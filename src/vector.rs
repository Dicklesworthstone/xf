//! Vector search for semantic similarity.
//!
//! Provides in-memory vector search with SIMD-accelerated dot product.
//! Vectors are loaded from `SQLite` at startup and searched using cosine
//! similarity (which equals dot product for L2-normalized vectors).

use crate::embedder::dot_product_simd;
use crate::storage::Storage;
use anyhow::{Result, ensure};
use fmmap::{MmapFile, MmapFileExt};
use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::str;

#[allow(dead_code)]
const VECTOR_INDEX_MAGIC: [u8; 4] = *b"XFVI";
#[allow(dead_code)]
const VECTOR_INDEX_VERSION: u16 = 1;
#[allow(dead_code)]
const VECTOR_INDEX_HEADER_LEN: usize = 32;
#[allow(dead_code)]
const VECTOR_INDEX_DOC_TYPE_ENCODING: u8 = 0;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct VectorIndexHeader {
    version: u16,
    doc_type_encoding: u8,
    dimension: u32,
    record_count: u64,
    offsets_start: u64,
}

#[allow(dead_code)]
fn parse_vector_index_header(bytes: &[u8]) -> Result<VectorIndexHeader> {
    ensure!(
        bytes.len() >= VECTOR_INDEX_HEADER_LEN,
        "vector index header truncated"
    );
    ensure!(
        bytes[0..4] == VECTOR_INDEX_MAGIC,
        "vector index magic mismatch"
    );

    let version = u16::from_le_bytes([bytes[4], bytes[5]]);
    let doc_type_encoding = bytes[6];
    let dimension = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
    let record_count = u64::from_le_bytes([
        bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17], bytes[18], bytes[19],
    ]);
    let offsets_start = u64::from_le_bytes([
        bytes[20], bytes[21], bytes[22], bytes[23], bytes[24], bytes[25], bytes[26], bytes[27],
    ]);

    Ok(VectorIndexHeader {
        version,
        doc_type_encoding,
        dimension,
        record_count,
        offsets_start,
    })
}

/// Encode `doc_type` string to u8 for compact storage.
/// Returns None for unknown types.
fn encode_doc_type(doc_type: &str) -> Option<u8> {
    match doc_type {
        "tweet" => Some(0),
        "like" => Some(1),
        "dm" => Some(2),
        "grok" => Some(3),
        _ => None,
    }
}

/// Decode `doc_type` u8 to string.
const fn decode_doc_type(value: u8) -> Option<&'static str> {
    match value {
        0 => Some("tweet"),
        1 => Some("like"),
        2 => Some("dm"),
        3 => Some("grok"),
        _ => None,
    }
}

/// Intern a `doc_type` string to a static reference.
///
/// Returns `&'static str` for known types, avoiding allocations.
/// Unknown types default to "tweet".
fn intern_doc_type(doc_type: &str) -> &'static str {
    match doc_type {
        "like" => "like",
        "dm" => "dm",
        "grok" => "grok",
        _ => "tweet", // Default for unknown types (including "tweet")
    }
}

fn validate_doc_type(value: u8) -> Result<()> {
    ensure!(value <= 3, "invalid doc_type encoding");
    Ok(())
}

#[allow(dead_code)]
fn validate_vector_index_layout(bytes: &[u8]) -> Result<()> {
    let header = parse_vector_index_header(bytes)?;

    ensure!(
        header.version == VECTOR_INDEX_VERSION,
        "unsupported vector index version"
    );
    ensure!(
        header.doc_type_encoding == VECTOR_INDEX_DOC_TYPE_ENCODING,
        "unsupported doc_type encoding"
    );
    ensure!(header.dimension > 0, "embedding dimension must be non-zero");
    let offsets_start = usize::try_from(header.offsets_start)
        .map_err(|_| anyhow::anyhow!("offsets start overflow"))?;
    ensure!(
        offsets_start >= VECTOR_INDEX_HEADER_LEN,
        "offsets start precedes header"
    );

    let offsets_bytes = header
        .record_count
        .checked_mul(8)
        .ok_or_else(|| anyhow::anyhow!("offset table size overflow"))?;
    let offsets_end = header
        .offsets_start
        .checked_add(offsets_bytes)
        .ok_or_else(|| anyhow::anyhow!("offset table end overflow"))?;
    let offsets_end =
        usize::try_from(offsets_end).map_err(|_| anyhow::anyhow!("offset table end overflow"))?;
    ensure!(
        offsets_end <= bytes.len(),
        "offset table exceeds file length"
    );

    let offsets_slice = &bytes[offsets_start..offsets_end];
    let record_base = offsets_end;
    let embedding_len = header.dimension as usize * 2;

    let mut last_offset = None;
    for chunk in offsets_slice.chunks_exact(8) {
        let offset = usize::try_from(u64::from_le_bytes([
            chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
        ]))
        .map_err(|_| anyhow::anyhow!("record offset overflow"))?;

        ensure!(offset >= record_base, "record offset precedes data section");
        ensure!(offset < bytes.len(), "record offset out of bounds");
        if let Some(prev) = last_offset {
            ensure!(offset >= prev, "record offsets not sorted");
        }
        last_offset = Some(offset);

        let record = &bytes[offset..];
        ensure!(record.len() >= 4, "record truncated");
        let doc_type = record[0];
        validate_doc_type(doc_type)?;
        let doc_id_len = u16::from_le_bytes([record[2], record[3]]) as usize;
        ensure!(doc_id_len > 0, "doc_id length must be non-zero");

        let header_len = 4usize;
        let total_len = header_len
            .checked_add(doc_id_len)
            .and_then(|v| v.checked_add(embedding_len))
            .ok_or_else(|| anyhow::anyhow!("record length overflow"))?;
        ensure!(record.len() >= total_len, "record length exceeds file");

        let doc_id_bytes = &record[header_len..header_len + doc_id_len];
        str::from_utf8(doc_id_bytes).map_err(|_| anyhow::anyhow!("doc_id is not valid UTF-8"))?;
    }

    Ok(())
}

#[derive(Debug, Clone, Copy)]
struct DocTypeFilter {
    allowed: [bool; 4],
}

impl DocTypeFilter {
    fn new(doc_types: Option<&[&str]>) -> Option<Self> {
        let doc_types = doc_types?;
        let mut allowed = [false; 4];
        for doc_type in doc_types {
            if let Some(code) = encode_doc_type(doc_type) {
                let idx = usize::from(code);
                if idx < allowed.len() {
                    allowed[idx] = true;
                }
            }
        }
        Some(Self { allowed })
    }

    fn allows(self, code: u8) -> bool {
        let idx = usize::from(code);
        self.allowed.get(idx).copied().unwrap_or(false)
    }
}

fn dot_product_f16_simd(query: &[f32], embedding: &[u8]) -> Option<f32> {
    use half::f16;
    use wide::f32x8;

    if embedding.len() != query.len().saturating_mul(2) {
        return None;
    }
    if query.is_empty() {
        return Some(0.0);
    }

    let mut sum = f32x8::ZERO;
    let mut idx = 0usize;
    while idx + 8 <= query.len() {
        let mut emb = [0.0f32; 8];
        for (lane, value) in emb.iter_mut().enumerate() {
            let byte_idx = (idx + lane) * 2;
            let arr = [embedding[byte_idx], embedding[byte_idx + 1]];
            *value = f16::from_le_bytes(arr).to_f32();
        }

        let mut q_arr = [0.0f32; 8];
        q_arr.copy_from_slice(&query[idx..idx + 8]);
        sum += f32x8::from(emb) * f32x8::from(q_arr);
        idx += 8;
    }

    let mut scalar_sum = sum.reduce_add();
    for (pos, value) in query.iter().enumerate().skip(idx) {
        let byte_idx = pos * 2;
        let arr = [embedding[byte_idx], embedding[byte_idx + 1]];
        scalar_sum += f16::from_le_bytes(arr).to_f32() * value;
    }

    Some(scalar_sum)
}

/// Default filename for the vector index file.
pub const VECTOR_INDEX_FILENAME: &str = "vector.idx";
/// Filename for the fast-tier vector index (two-tier search).
pub const VECTOR_INDEX_FAST_FILENAME: &str = "vector.fast.idx";
/// Filename for the quality-tier vector index (two-tier search).
pub const VECTOR_INDEX_QUALITY_FILENAME: &str = "vector.quality.idx";

/// Write a vector index file from embeddings.
///
/// This creates a compact binary file that can be memory-mapped for fast
/// semantic search without scanning `SQLite`.
///
/// # Arguments
///
/// * `index_path` - Directory where the vector index file will be written.
/// * `storage` - Storage instance to load embeddings from.
///
/// # File Format
///
/// The file has the following structure:
/// - Header (32 bytes): magic, version, dimension, record count, offsets pointer
/// - Offset table: array of u64 offsets to each record
/// - Records: `doc_type` (u8), reserved (u8), `doc_id_len` (u16), `doc_id`, embedding (F16)
///
/// Records are sorted by (`doc_type`, `doc_id`) for deterministic ordering.
///
/// # Errors
///
/// Returns an error if loading embeddings fails or if file I/O fails.
pub fn write_vector_index(
    index_path: &std::path::Path,
    storage: &Storage,
) -> Result<WriteVectorIndexStats> {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    // Load all embeddings from storage as raw F16 bytes
    let mut embeddings = storage.load_all_embeddings_raw()?;
    if embeddings.is_empty() {
        return Ok(WriteVectorIndexStats {
            record_count: 0,
            file_size: 0,
        });
    }

    // Determine dimension from first embedding (bytes / 2 since F16)
    let embedding_bytes = embeddings[0].2.len();
    ensure!(
        embedding_bytes > 0 && embedding_bytes % 2 == 0,
        "invalid embedding byte length: {embedding_bytes}"
    );

    // Validate embeddings before writing to avoid silent mislabeling.
    for (doc_id, doc_type, embedding_f16) in &embeddings {
        if encode_doc_type(doc_type).is_none() {
            return Err(anyhow::anyhow!(
                "unknown embedding doc_type '{doc_type}' for doc_id '{doc_id}'"
            ));
        }
        if embedding_f16.len() != embedding_bytes {
            return Err(anyhow::anyhow!(
                "embedding byte length mismatch for doc_id '{doc_id}' (type '{doc_type}'): expected {embedding_bytes}, got {}",
                embedding_f16.len()
            ));
        }
    }

    // Sort deterministically: by doc_type code, then by doc_id
    embeddings.sort_by(|a, b| {
        let type_a = encode_doc_type(&a.1).unwrap_or(255);
        let type_b = encode_doc_type(&b.1).unwrap_or(255);
        type_a.cmp(&type_b).then_with(|| a.0.cmp(&b.0))
    });

    let dimension = embedding_bytes / 2; // F16 = 2 bytes per float
    let record_count = embeddings.len() as u64;

    // Calculate layout
    let offsets_start = VECTOR_INDEX_HEADER_LEN as u64;
    let offsets_bytes = record_count * 8;
    let records_start = offsets_start + offsets_bytes;

    // Build offset table and record bytes
    let mut offsets: Vec<u64> = Vec::with_capacity(embeddings.len());
    let mut record_data: Vec<u8> = Vec::new();
    let mut current_offset = records_start;

    for (doc_id, doc_type, embedding_f16) in &embeddings {
        offsets.push(current_offset);

        let doc_type_code = encode_doc_type(doc_type).ok_or_else(|| {
            anyhow::anyhow!("unknown embedding doc_type '{doc_type}' for doc_id '{doc_id}'")
        })?;
        let doc_id_bytes = doc_id.as_bytes();
        let doc_id_len = u16::try_from(doc_id_bytes.len())
            .map_err(|_| anyhow::anyhow!("doc_id exceeds maximum length of 65535 bytes"))?;

        // Record: [doc_type(1), reserved(1), doc_id_len(2), doc_id, embedding]
        record_data.push(doc_type_code);
        record_data.push(0); // reserved
        record_data.extend_from_slice(&doc_id_len.to_le_bytes());
        record_data.extend_from_slice(doc_id_bytes);

        // Embedding is already F16 bytes from load_all_embeddings_raw
        record_data.extend_from_slice(embedding_f16);

        let record_len = 4 + doc_id_bytes.len() + embedding_bytes;
        current_offset += record_len as u64;
    }

    // Write to temp file, then rename atomically
    let final_path = index_path.join(VECTOR_INDEX_FILENAME);
    let temp_path = index_path.join(format!("{VECTOR_INDEX_FILENAME}.tmp"));

    let file = File::create(&temp_path)?;
    let mut writer = BufWriter::new(file);

    // Write header (32 bytes)
    let dimension_u32 = u32::try_from(dimension)
        .map_err(|_| anyhow::anyhow!("embedding dimension exceeds u32 maximum"))?;
    writer.write_all(&VECTOR_INDEX_MAGIC)?;
    writer.write_all(&VECTOR_INDEX_VERSION.to_le_bytes())?;
    writer.write_all(&[VECTOR_INDEX_DOC_TYPE_ENCODING])?; // doc_type_encoding
    writer.write_all(&[0u8])?; // padding
    writer.write_all(&dimension_u32.to_le_bytes())?;
    writer.write_all(&record_count.to_le_bytes())?;
    writer.write_all(&offsets_start.to_le_bytes())?;
    writer.write_all(&[0u8; 4])?; // reserved

    // Write offset table
    for offset in &offsets {
        writer.write_all(&offset.to_le_bytes())?;
    }

    // Write record data
    writer.write_all(&record_data)?;

    // Flush and sync
    writer.flush()?;
    let file = writer.into_inner()?;
    file.sync_all()?;
    drop(file);

    // Get file size before rename
    let file_size = std::fs::metadata(&temp_path)?.len();

    // Atomic rename
    std::fs::rename(&temp_path, &final_path)?;

    #[allow(clippy::cast_possible_truncation)]
    Ok(WriteVectorIndexStats {
        // Safe: record_count comes from embeddings.len() which is already usize
        record_count: record_count as usize,
        file_size,
    })
}

/// Write a named vector index file from embeddings filtered by model ID.
///
/// Generalized form of `write_vector_index` that writes to a specific filename
/// and optionally filters embeddings by `model_id`.
///
/// # Errors
///
/// Returns an error if loading embeddings fails or if file I/O fails.
pub fn write_vector_index_named(
    index_path: &std::path::Path,
    storage: &Storage,
    filename: &str,
    model_id_filter: Option<&str>,
) -> Result<WriteVectorIndexStats> {
    use std::fs::File;
    use std::io::{BufWriter, Write};

    let mut embeddings = storage.load_all_embeddings_raw_filtered(model_id_filter)?;
    if embeddings.is_empty() {
        return Ok(WriteVectorIndexStats {
            record_count: 0,
            file_size: 0,
        });
    }

    let embedding_bytes = embeddings[0].2.len();
    ensure!(
        embedding_bytes > 0 && embedding_bytes % 2 == 0,
        "invalid embedding byte length: {embedding_bytes}"
    );

    for (doc_id, doc_type, embedding_f16) in &embeddings {
        if encode_doc_type(doc_type).is_none() {
            return Err(anyhow::anyhow!(
                "unknown embedding doc_type '{doc_type}' for doc_id '{doc_id}'"
            ));
        }
        if embedding_f16.len() != embedding_bytes {
            return Err(anyhow::anyhow!(
                "embedding byte length mismatch for doc_id '{doc_id}' (type '{doc_type}'): expected {embedding_bytes}, got {}",
                embedding_f16.len()
            ));
        }
    }

    embeddings.sort_by(|a, b| {
        let type_a = encode_doc_type(&a.1).unwrap_or(255);
        let type_b = encode_doc_type(&b.1).unwrap_or(255);
        type_a.cmp(&type_b).then_with(|| a.0.cmp(&b.0))
    });

    let dimension = embedding_bytes / 2;
    let record_count = embeddings.len() as u64;

    let offsets_start = VECTOR_INDEX_HEADER_LEN as u64;
    let offsets_bytes = record_count * 8;
    let records_start = offsets_start + offsets_bytes;

    let mut offsets: Vec<u64> = Vec::with_capacity(embeddings.len());
    let mut record_data: Vec<u8> = Vec::new();
    let mut current_offset = records_start;

    for (doc_id, doc_type, embedding_f16) in &embeddings {
        offsets.push(current_offset);

        let doc_type_code = encode_doc_type(doc_type).ok_or_else(|| {
            anyhow::anyhow!("unknown embedding doc_type '{doc_type}' for doc_id '{doc_id}'")
        })?;
        let doc_id_bytes = doc_id.as_bytes();
        let doc_id_len = u16::try_from(doc_id_bytes.len())
            .map_err(|_| anyhow::anyhow!("doc_id exceeds maximum length of 65535 bytes"))?;

        record_data.push(doc_type_code);
        record_data.push(0);
        record_data.extend_from_slice(&doc_id_len.to_le_bytes());
        record_data.extend_from_slice(doc_id_bytes);
        record_data.extend_from_slice(embedding_f16);

        let record_len = 4 + doc_id_bytes.len() + embedding_bytes;
        current_offset += record_len as u64;
    }

    let final_path = index_path.join(filename);
    let temp_path = index_path.join(format!("{filename}.tmp"));

    let file = File::create(&temp_path)?;
    let mut writer = BufWriter::new(file);

    let dimension_u32 = u32::try_from(dimension)
        .map_err(|_| anyhow::anyhow!("embedding dimension exceeds u32 maximum"))?;
    writer.write_all(&VECTOR_INDEX_MAGIC)?;
    writer.write_all(&VECTOR_INDEX_VERSION.to_le_bytes())?;
    writer.write_all(&[VECTOR_INDEX_DOC_TYPE_ENCODING])?;
    writer.write_all(&[0u8])?;
    writer.write_all(&dimension_u32.to_le_bytes())?;
    writer.write_all(&record_count.to_le_bytes())?;
    writer.write_all(&offsets_start.to_le_bytes())?;
    writer.write_all(&[0u8; 4])?;

    for offset in &offsets {
        writer.write_all(&offset.to_le_bytes())?;
    }

    writer.write_all(&record_data)?;

    writer.flush()?;
    let file = writer.into_inner()?;
    file.sync_all()?;
    drop(file);

    let file_size = std::fs::metadata(&temp_path)?.len();
    std::fs::rename(&temp_path, &final_path)?;

    #[allow(clippy::cast_possible_truncation)]
    Ok(WriteVectorIndexStats {
        record_count: record_count as usize,
        file_size,
    })
}

/// Statistics from writing a vector index file.
#[derive(Debug, Clone, Copy)]
pub struct WriteVectorIndexStats {
    /// Number of embedding records written.
    pub record_count: usize,
    /// Total file size in bytes.
    pub file_size: u64,
}

/// Result of a vector search.
#[derive(Debug, Clone)]
pub struct VectorSearchResult {
    /// Document ID.
    pub doc_id: String,
    /// Document type (tweet, like, dm, grok).
    ///
    /// Uses `&'static str` to avoid allocations for the fixed set of types.
    pub doc_type: &'static str,
    /// Similarity score (cosine similarity, range -1.0 to 1.0).
    pub score: f32,
}

/// Entry in the min-heap for top-k selection (allocation-free).
///
/// Stores only the offset into the mmap buffer, deferring String allocation
/// until after the heap is finalized. This reduces allocations from O(n) to O(k).
#[derive(Debug, Clone, Copy)]
struct HeapEntry {
    score: f32,
    /// Offset into the mmap buffer where this record starts.
    offset: usize,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.offset == other.offset
    }
}

impl Eq for HeapEntry {}

impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: lower score is "greater" (gets popped first)
        // This way we keep the highest scores in the heap
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.offset.cmp(&other.offset))
    }
}

/// Entry in the min-heap for in-memory `VectorIndex` (allocation-free).
///
/// Stores only the index into the vectors Vec, deferring String access
/// until after the heap is finalized.
#[derive(Debug, Clone, Copy)]
struct IndexHeapEntry {
    score: f32,
    /// Index into the vectors Vec.
    idx: usize,
}

impl PartialEq for IndexHeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score && self.idx == other.idx
    }
}

impl Eq for IndexHeapEntry {}

impl PartialOrd for IndexHeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IndexHeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap: lower score is "greater" (gets popped first)
        other
            .score
            .total_cmp(&self.score)
            .then_with(|| self.idx.cmp(&other.idx))
    }
}

/// Mmap-backed vector index for fast similarity search without `SQLite`.
pub struct MmapVectorIndex {
    mmap: MmapFile,
    record_count: usize,
    dimension: usize,
    embedding_len: usize,
    offsets_range: std::ops::Range<usize>,
}

impl MmapVectorIndex {
    /// Open a memory-mapped vector index file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be mapped or fails validation.
    pub fn open(path: &std::path::Path) -> Result<Self> {
        let mmap = MmapFile::open(path)?;
        let bytes = mmap.as_slice();
        validate_vector_index_layout(bytes)?;

        let header = parse_vector_index_header(bytes)?;
        let record_count = usize::try_from(header.record_count)
            .map_err(|_| anyhow::anyhow!("record count overflow"))?;
        let dimension =
            usize::try_from(header.dimension).map_err(|_| anyhow::anyhow!("dimension overflow"))?;
        let embedding_len = dimension
            .checked_mul(2)
            .ok_or_else(|| anyhow::anyhow!("embedding length overflow"))?;

        let offsets_start = usize::try_from(header.offsets_start)
            .map_err(|_| anyhow::anyhow!("offsets start overflow"))?;
        let offsets_len = record_count
            .checked_mul(8)
            .ok_or_else(|| anyhow::anyhow!("offset table length overflow"))?;
        let offsets_end = offsets_start
            .checked_add(offsets_len)
            .ok_or_else(|| anyhow::anyhow!("offset table end overflow"))?;

        Ok(Self {
            mmap,
            record_count,
            dimension,
            embedding_len,
            offsets_range: offsets_start..offsets_end,
        })
    }

    /// Get number of records in the index.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.record_count
    }

    /// Check if the index is empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.record_count == 0
    }

    /// Get embedding dimension for the index.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Search for the top-k most similar vectors.
    #[must_use]
    pub fn search_top_k(
        &self,
        query: &[f32],
        k: usize,
        doc_types: Option<&[&str]>,
    ) -> Vec<VectorSearchResult> {
        if k == 0 || self.record_count == 0 || query.len() != self.dimension {
            return Vec::new();
        }

        let filter = DocTypeFilter::new(doc_types);
        let bytes = self.mmap.as_slice();
        let Some(offsets_bytes) = bytes.get(self.offsets_range.clone()) else {
            return Vec::new();
        };

        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(k + 1);

        // Phase 1: Scan all records, keeping only offsets in heap (no String allocations)
        for chunk in offsets_bytes.chunks_exact(8) {
            let offset = usize::try_from(u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]))
            .unwrap_or(usize::MAX);
            let Some(record) = bytes.get(offset..) else {
                continue;
            };
            if record.len() < 4 {
                continue;
            }

            let doc_type_code = record[0];
            if let Some(filter) = &filter {
                if !filter.allows(doc_type_code) {
                    continue;
                }
            }
            if decode_doc_type(doc_type_code).is_none() {
                continue;
            }

            let doc_id_len = u16::from_le_bytes([record[2], record[3]]) as usize;
            let doc_id_start = 4usize;
            let doc_id_end = doc_id_start.saturating_add(doc_id_len);
            let embedding_end = doc_id_end.saturating_add(self.embedding_len);
            if record.len() < embedding_end {
                continue;
            }

            // Validate UTF-8 without allocating
            let doc_id_bytes = &record[doc_id_start..doc_id_end];
            if str::from_utf8(doc_id_bytes).is_err() {
                continue;
            }

            let embedding_bytes = &record[doc_id_end..embedding_end];
            let Some(score) = dot_product_f16_simd(query, embedding_bytes) else {
                continue;
            };

            heap.push(HeapEntry { score, offset });

            if heap.len() > k {
                heap.pop();
            }
        }

        // Phase 2: Extract top-k and parse Strings only for final results
        let mut results: Vec<VectorSearchResult> = Vec::with_capacity(heap.len());
        for entry in heap {
            let Some(record) = bytes.get(entry.offset..) else {
                continue;
            };
            let doc_type_code = record[0];
            let Some(doc_type) = decode_doc_type(doc_type_code) else {
                continue;
            };
            let doc_id_len = u16::from_le_bytes([record[2], record[3]]) as usize;
            let doc_id_bytes = &record[4..4 + doc_id_len];
            let Ok(doc_id) = str::from_utf8(doc_id_bytes) else {
                continue;
            };
            results.push(VectorSearchResult {
                doc_id: doc_id.to_string(),
                doc_type, // Already &'static str from decode_doc_type
                score: entry.score,
            });
        }

        results.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
                .then_with(|| a.doc_type.cmp(b.doc_type))
        });

        results
    }

    /// Get counts of embeddings by document type.
    #[must_use]
    pub fn type_counts(&self) -> std::collections::HashMap<String, usize> {
        let mut counts = std::collections::HashMap::new();
        let bytes = self.mmap.as_slice();
        let Some(offsets_bytes) = bytes.get(self.offsets_range.clone()) else {
            return counts;
        };

        for chunk in offsets_bytes.chunks_exact(8) {
            let offset = usize::try_from(u64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]))
            .unwrap_or(usize::MAX);
            let Some(record) = bytes.get(offset..) else {
                continue;
            };
            if record.is_empty() {
                continue;
            }
            if let Some(doc_type) = decode_doc_type(record[0]) {
                *counts.entry(doc_type.to_string()).or_insert(0) += 1;
            }
        }
        counts
    }
}

/// Unified vector index interface wrapping either Mmap or In-Memory implementation.
pub enum VectorIndex {
    Mmap(MmapVectorIndex),
    InMemory(InMemoryVectorIndex),
}

impl VectorIndex {
    /// Load embeddings from a file into memory.
    ///
    /// Returns `Ok(None)` if the file doesn't exist.
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but cannot be read or is invalid.
    pub fn load_from_file(index_path: &std::path::Path) -> Result<Option<Self>> {
        InMemoryVectorIndex::load_from_file(index_path).map(|opt| opt.map(Self::InMemory))
    }

    /// Load embeddings directly from storage into memory.
    ///
    /// # Errors
    ///
    /// Returns an error if loading from storage fails.
    pub fn load_from_storage(storage: &Storage) -> Result<Self> {
        let index = InMemoryVectorIndex::load_from_storage(storage)?;
        Ok(Self::InMemory(index))
    }

    /// Try to load from file first (Mmap), fall back to storage (InMemory) if unavailable.
    ///
    /// # Errors
    ///
    /// Returns an error if both file loading and storage loading fail.
    pub fn load_from_file_or_storage(
        index_path: &std::path::Path,
        storage: &Storage,
    ) -> Result<Self> {
        // Try mmap first (fast path)
        let file_path = index_path.join(VECTOR_INDEX_FILENAME);
        if file_path.exists() {
            match MmapVectorIndex::open(&file_path) {
                Ok(index) => return Ok(Self::Mmap(index)),
                Err(e) => {
                    tracing::warn!("Failed to mmap vector index, falling back to DB: {}", e);
                }
            }
        }

        // Fall back to storage (slow path, loads into memory)
        let index = InMemoryVectorIndex::load_from_storage(storage)?;
        Ok(Self::InMemory(index))
    }

    /// Try to load a named index file via mmap.
    ///
    /// Returns `Ok(None)` if the file does not exist.
    /// Returns `Ok(Some(..))` if the file was opened successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but cannot be read or is invalid.
    pub fn load_named(index_path: &std::path::Path, filename: &str) -> Result<Option<Self>> {
        let file_path = index_path.join(filename);
        if !file_path.exists() {
            return Ok(None);
        }
        let mmap_index = MmapVectorIndex::open(&file_path)?;
        Ok(Some(Self::Mmap(mmap_index)))
    }

    /// Get number of records.
    #[must_use]
    pub fn len(&self) -> usize {
        match self {
            Self::Mmap(idx) => idx.len(),
            Self::InMemory(idx) => idx.len(),
        }
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match self {
            Self::Mmap(idx) => idx.is_empty(),
            Self::InMemory(idx) => idx.is_empty(),
        }
    }

    /// Get embedding dimension.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // Cannot be const due to match on enum
    pub fn dimension(&self) -> usize {
        match self {
            Self::Mmap(idx) => idx.dimension(),
            Self::InMemory(idx) => idx.dimension(),
        }
    }

    /// Get type counts.
    #[must_use]
    pub fn type_counts(&self) -> std::collections::HashMap<String, usize> {
        match self {
            Self::Mmap(idx) => idx.type_counts(),
            Self::InMemory(idx) => idx.type_counts(),
        }
    }

    /// Search for top-k results.
    #[must_use]
    pub fn search_top_k(
        &self,
        query: &[f32],
        k: usize,
        doc_types: Option<&[&str]>,
    ) -> Vec<VectorSearchResult> {
        match self {
            Self::Mmap(idx) => idx.search_top_k(query, k, doc_types),
            Self::InMemory(idx) => idx.search_top_k(query, k, doc_types),
        }
    }
}

/// In-memory vector index for fast similarity search.
pub struct InMemoryVectorIndex {
    /// All stored vectors with their metadata.
    /// Uses `&'static str` for `doc_type` to avoid allocations.
    vectors: Vec<(String, &'static str, Vec<f32>)>, // (doc_id, doc_type, embedding)
    /// Embedding dimension.
    dimension: usize,
}

impl InMemoryVectorIndex {
    /// Create a new empty vector index.
    #[must_use]
    pub const fn new(dimension: usize) -> Self {
        Self {
            vectors: Vec::new(),
            dimension,
        }
    }

    /// Load all embeddings from storage.
    ///
    /// # Errors
    ///
    /// Returns an error if loading from storage fails.
    pub fn load_from_storage(storage: &Storage) -> Result<Self> {
        let embeddings = storage.load_all_embeddings()?;

        let dimension = embeddings.first().map_or(384, |(_, _, v)| v.len());

        // Intern doc_type strings to avoid allocations
        let vectors = embeddings
            .into_iter()
            .map(|(doc_id, doc_type, embedding)| (doc_id, intern_doc_type(&doc_type), embedding))
            .collect();

        Ok(Self { vectors, dimension })
    }

    /// Load embeddings from a vector index file.
    ///
    /// This is the fast path for semantic search - it reads the pre-computed
    /// index file directly instead of scanning `SQLite`.
    ///
    /// Returns `None` if the file doesn't exist. Returns an error if the file
    /// exists but is corrupt or has an unsupported version.
    ///
    /// # Errors
    ///
    /// Returns an error if the file exists but cannot be read or is invalid.
    #[allow(clippy::missing_panics_doc, clippy::cast_possible_truncation)]
    pub fn load_from_file(index_path: &std::path::Path) -> Result<Option<Self>> {
        use half::f16;
        use std::fs::File;
        use std::io::Read;
        use tracing::warn;

        let file_path = index_path.join(VECTOR_INDEX_FILENAME);
        if !file_path.exists() {
            return Ok(None);
        }

        // Read entire file into memory
        let mut file = File::open(&file_path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;

        // Validate header and layout
        if let Err(e) = validate_vector_index_layout(&bytes) {
            warn!("Vector index file is invalid, falling back to DB: {}", e);
            return Ok(None);
        }

        let header = parse_vector_index_header(&bytes)?;
        let dimension = header.dimension as usize;
        let record_count = header.record_count as usize;
        let offsets_start = header.offsets_start as usize;
        let embedding_bytes = dimension * 2; // F16 = 2 bytes per float

        // Parse offset table
        let offsets_end = offsets_start + record_count * 8;
        let offsets_slice = &bytes[offsets_start..offsets_end];

        // Pre-allocate vector storage
        let mut vectors = Vec::with_capacity(record_count);

        // Parse each record
        for i in 0..record_count {
            let offset_bytes = &offsets_slice[i * 8..(i + 1) * 8];
            let offset = u64::from_le_bytes(offset_bytes.try_into().unwrap()) as usize;

            let record = &bytes[offset..];

            // Parse record header
            let doc_type_code = record[0];
            let doc_id_len = u16::from_le_bytes([record[2], record[3]]) as usize;

            // Extract doc_id
            let doc_id_start = 4;
            let doc_id_bytes = &record[doc_id_start..doc_id_start + doc_id_len];
            let doc_id = std::str::from_utf8(doc_id_bytes)
                .map_err(|_| anyhow::anyhow!("invalid UTF-8 in doc_id"))?
                .to_string();

            // Decode doc_type (already &'static str)
            let doc_type = decode_doc_type(doc_type_code)
                .ok_or_else(|| anyhow::anyhow!("unknown doc_type code: {doc_type_code}"))?;

            // Extract and convert embedding F16 -> f32
            let embedding_start = doc_id_start + doc_id_len;
            let embedding_slice = &record[embedding_start..embedding_start + embedding_bytes];
            let embedding: Vec<f32> = embedding_slice
                .chunks_exact(2)
                .map(|chunk| {
                    let arr: [u8; 2] = chunk.try_into().unwrap();
                    f16::from_le_bytes(arr).to_f32()
                })
                .collect();

            vectors.push((doc_id, doc_type, embedding));
        }

        Ok(Some(Self { vectors, dimension }))
    }

    /// Add a vector to the index.
    ///
    /// The `doc_type` is interned to a static reference.
    pub fn add(&mut self, doc_id: String, doc_type: &str, embedding: Vec<f32>) {
        debug_assert_eq!(
            embedding.len(),
            self.dimension,
            "embedding dimension mismatch"
        );
        self.vectors
            .push((doc_id, intern_doc_type(doc_type), embedding));
    }

    /// Get the number of vectors in the index.
    #[must_use]
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Check if the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Get counts of embeddings by document type.
    #[must_use]
    pub fn type_counts(&self) -> std::collections::HashMap<String, usize> {
        // Count using static strings first (no allocations during counting)
        let mut static_counts: std::collections::HashMap<&'static str, usize> =
            std::collections::HashMap::with_capacity(4);
        for (_, doc_type, _) in &self.vectors {
            *static_counts.entry(*doc_type).or_insert(0) += 1;
        }
        // Convert to String only for the final result (at most 4 entries)
        static_counts
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect()
    }

    /// Get the embedding dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Search for the top-k most similar vectors.
    ///
    /// Uses SIMD-accelerated dot product for fast computation.
    /// Results are sorted by score (descending), then by `doc_id` (ascending)
    /// for deterministic ordering.
    #[must_use]
    pub fn search_top_k(
        &self,
        query: &[f32],
        k: usize,
        doc_types: Option<&[&str]>,
    ) -> Vec<VectorSearchResult> {
        if k == 0 || self.is_empty() || query.len() != self.dimension {
            return Vec::new();
        }

        // Phase 1: Scan vectors, keeping only indices in heap (no String clones)
        let mut heap: BinaryHeap<IndexHeapEntry> = BinaryHeap::with_capacity(k + 1);

        for (idx, (_, doc_type, embedding)) in self.vectors.iter().enumerate() {
            // Filter by doc_type if specified
            if let Some(types) = doc_types {
                if !types.contains(doc_type) {
                    continue;
                }
            }

            // Compute similarity using SIMD dot product
            let score = dot_product_simd(query, embedding);

            heap.push(IndexHeapEntry { score, idx });

            // Keep only top-k by removing the minimum when heap exceeds k
            if heap.len() > k {
                heap.pop();
            }
        }

        // Phase 2: Extract top-k results (doc_type is Copy since it's &'static str)
        let mut results: Vec<VectorSearchResult> = Vec::with_capacity(heap.len());
        for entry in heap {
            let (doc_id, doc_type, _) = &self.vectors[entry.idx];
            results.push(VectorSearchResult {
                doc_id: doc_id.clone(),
                doc_type,
                score: entry.score,
            });
        }

        // Sort by score descending, then doc_id + doc_type ascending for determinism
        results.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
                .then_with(|| a.doc_type.cmp(b.doc_type))
        });

        results
    }

    /// Search with parallel processing for large indices.
    ///
    /// Uses rayon to parallelize the search across multiple CPU cores.
    /// Falls back to sequential search for small indices.
    #[cfg(feature = "parallel-search")]
    #[must_use]
    pub fn search_top_k_parallel(
        &self,
        query: &[f32],
        k: usize,
        doc_types: Option<&[&str]>,
    ) -> Vec<VectorSearchResult> {
        use rayon::prelude::*;

        const PARALLEL_THRESHOLD: usize = 10_000;
        const CHUNK_SIZE: usize = 1024;

        if self.vectors.len() < PARALLEL_THRESHOLD {
            return self.search_top_k(query, k, doc_types);
        }

        // Create indexed chunks to preserve absolute indices
        let indexed_chunks: Vec<_> = self
            .vectors
            .chunks(CHUNK_SIZE)
            .enumerate()
            .map(|(chunk_idx, chunk)| (chunk_idx * CHUNK_SIZE, chunk))
            .collect();

        // Parallel scan with thread-local heaps using indices (no String clones)
        let partial_results: Vec<Vec<IndexHeapEntry>> = indexed_chunks
            .par_iter()
            .map(|(base_idx, chunk)| {
                let mut local_heap: BinaryHeap<IndexHeapEntry> = BinaryHeap::with_capacity(k + 1);

                for (offset, (_, doc_type, embedding)) in chunk.iter().enumerate() {
                    if let Some(types) = doc_types {
                        if !types.contains(doc_type) {
                            continue;
                        }
                    }

                    let score = dot_product_simd(query, embedding);
                    let idx = base_idx + offset;

                    local_heap.push(IndexHeapEntry { score, idx });

                    if local_heap.len() > k {
                        local_heap.pop();
                    }
                }

                local_heap.into_vec()
            })
            .collect();

        // Merge thread-local results
        let mut final_heap: BinaryHeap<IndexHeapEntry> = BinaryHeap::with_capacity(k + 1);
        for entries in partial_results {
            for entry in entries {
                final_heap.push(entry);
                if final_heap.len() > k {
                    final_heap.pop();
                }
            }
        }

        // Only clone doc_id for the final k results (doc_type is Copy)
        let mut results: Vec<VectorSearchResult> = Vec::with_capacity(final_heap.len());
        for entry in final_heap {
            let (doc_id, doc_type, _) = &self.vectors[entry.idx];
            results.push(VectorSearchResult {
                doc_id: doc_id.clone(),
                doc_type,
                score: entry.score,
            });
        }

        results.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| a.doc_id.cmp(&b.doc_id))
                .then_with(|| a.doc_type.cmp(b.doc_type))
        });

        results
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::l2_normalize;

    #[allow(clippy::cast_precision_loss)]
    fn create_test_vector(seed: u32, dim: usize) -> Vec<f32> {
        let mut vec: Vec<f32> = (0..dim)
            .map(|i| ((seed as usize * 17 + i * 13) % 100) as f32 / 100.0)
            .collect();
        l2_normalize(&mut vec);
        vec
    }

    fn build_vector_index_bytes(dimension: u32, records: &[(u8, &str)]) -> Vec<u8> {
        let offsets_start = VECTOR_INDEX_HEADER_LEN as u64;
        let offsets_len = records.len() * 8;
        let records_start = offsets_start + offsets_len as u64;

        let mut offsets: Vec<u64> = Vec::with_capacity(records.len());
        let mut record_bytes: Vec<u8> = Vec::new();
        let mut current_offset = records_start;

        for (doc_type, doc_id) in records {
            offsets.push(current_offset);
            let doc_id_bytes = doc_id.as_bytes();
            let doc_id_len = u16::try_from(doc_id_bytes.len()).expect("doc_id length fits u16");
            let embedding_len = dimension as usize * 2;

            record_bytes.push(*doc_type);
            record_bytes.push(0);
            record_bytes.extend_from_slice(&doc_id_len.to_le_bytes());
            record_bytes.extend_from_slice(doc_id_bytes);
            record_bytes.extend(std::iter::repeat_n(0u8, embedding_len));

            let record_len = 4 + doc_id_bytes.len() + embedding_len;
            current_offset = current_offset
                .checked_add(record_len as u64)
                .expect("record offset overflow");
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&VECTOR_INDEX_MAGIC);
        bytes.extend_from_slice(&VECTOR_INDEX_VERSION.to_le_bytes());
        bytes.push(VECTOR_INDEX_DOC_TYPE_ENCODING);
        bytes.push(0);
        bytes.extend_from_slice(&dimension.to_le_bytes());
        bytes.extend_from_slice(&(records.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&offsets_start.to_le_bytes());
        bytes.extend_from_slice(&0u32.to_le_bytes());

        for offset in offsets {
            bytes.extend_from_slice(&offset.to_le_bytes());
        }
        bytes.extend_from_slice(&record_bytes);

        bytes
    }

    #[test]
    fn test_vector_index_layout_validation_ok() {
        let bytes = build_vector_index_bytes(3, &[(0, "doc1"), (1, "doc2")]);
        validate_vector_index_layout(&bytes).unwrap();
    }

    #[test]
    fn test_vector_index_layout_invalid_magic() {
        let mut bytes = build_vector_index_bytes(3, &[(0, "doc1")]);
        bytes[0] = b'Z';
        assert!(validate_vector_index_layout(&bytes).is_err());
    }

    #[test]
    fn test_vector_index_layout_invalid_offsets() {
        let mut bytes = build_vector_index_bytes(3, &[(0, "doc1")]);
        let offsets_start = u64::from_le_bytes([
            bytes[20], bytes[21], bytes[22], bytes[23], bytes[24], bytes[25], bytes[26], bytes[27],
        ]);
        let bad_offset = offsets_start + 8 + 10_000;
        let start = VECTOR_INDEX_HEADER_LEN;
        bytes[start..start + 8].copy_from_slice(&bad_offset.to_le_bytes());
        assert!(validate_vector_index_layout(&bytes).is_err());
    }

    #[test]
    fn test_vector_index_layout_invalid_doc_type() {
        let mut bytes = build_vector_index_bytes(3, &[(0, "doc1")]);
        let record_offset = usize::try_from(u64::from_le_bytes([
            bytes[VECTOR_INDEX_HEADER_LEN],
            bytes[VECTOR_INDEX_HEADER_LEN + 1],
            bytes[VECTOR_INDEX_HEADER_LEN + 2],
            bytes[VECTOR_INDEX_HEADER_LEN + 3],
            bytes[VECTOR_INDEX_HEADER_LEN + 4],
            bytes[VECTOR_INDEX_HEADER_LEN + 5],
            bytes[VECTOR_INDEX_HEADER_LEN + 6],
            bytes[VECTOR_INDEX_HEADER_LEN + 7],
        ]))
        .unwrap();
        bytes[record_offset] = 9;
        assert!(validate_vector_index_layout(&bytes).is_err());
    }

    #[test]
    fn test_vector_index_layout_invalid_doc_id_len() {
        let mut bytes = build_vector_index_bytes(3, &[(0, "doc1")]);
        let record_offset = usize::try_from(u64::from_le_bytes([
            bytes[VECTOR_INDEX_HEADER_LEN],
            bytes[VECTOR_INDEX_HEADER_LEN + 1],
            bytes[VECTOR_INDEX_HEADER_LEN + 2],
            bytes[VECTOR_INDEX_HEADER_LEN + 3],
            bytes[VECTOR_INDEX_HEADER_LEN + 4],
            bytes[VECTOR_INDEX_HEADER_LEN + 5],
            bytes[VECTOR_INDEX_HEADER_LEN + 6],
            bytes[VECTOR_INDEX_HEADER_LEN + 7],
        ]))
        .unwrap();
        let doc_id_len_offset = record_offset + 2;
        bytes[doc_id_len_offset..doc_id_len_offset + 2].copy_from_slice(&u16::MAX.to_le_bytes());
        assert!(validate_vector_index_layout(&bytes).is_err());
    }

    #[test]
    fn test_vector_index_layout_invalid_utf8() {
        let mut bytes = build_vector_index_bytes(3, &[(0, "doc1")]);
        let record_offset = usize::try_from(u64::from_le_bytes([
            bytes[VECTOR_INDEX_HEADER_LEN],
            bytes[VECTOR_INDEX_HEADER_LEN + 1],
            bytes[VECTOR_INDEX_HEADER_LEN + 2],
            bytes[VECTOR_INDEX_HEADER_LEN + 3],
            bytes[VECTOR_INDEX_HEADER_LEN + 4],
            bytes[VECTOR_INDEX_HEADER_LEN + 5],
            bytes[VECTOR_INDEX_HEADER_LEN + 6],
            bytes[VECTOR_INDEX_HEADER_LEN + 7],
        ]))
        .unwrap();
        let doc_id_len =
            u16::from_le_bytes([bytes[record_offset + 2], bytes[record_offset + 3]]) as usize;
        let doc_id_offset = record_offset + 4;
        if doc_id_len > 0 {
            bytes[doc_id_offset] = 0xFF;
        }
        assert!(validate_vector_index_layout(&bytes).is_err());
    }

    #[test]
    fn test_new_index() {
        let index = InMemoryVectorIndex::new(384);
        assert!(index.is_empty());
        assert_eq!(index.dimension(), 384);
    }

    #[test]
    fn test_add_and_len() {
        let mut index = InMemoryVectorIndex::new(4);
        assert_eq!(index.len(), 0);

        let mut v1 = vec![1.0, 0.0, 0.0, 0.0];
        l2_normalize(&mut v1);
        index.add("doc1".to_string(), "tweet", v1);

        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_search_empty() {
        let index = InMemoryVectorIndex::new(4);
        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.search_top_k(&query, 10, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_basic() {
        let mut index = InMemoryVectorIndex::new(4);

        // Add some test vectors
        let mut v1 = vec![1.0, 0.0, 0.0, 0.0];
        let mut v2 = vec![0.9, 0.1, 0.0, 0.0];
        let mut v3 = vec![0.0, 1.0, 0.0, 0.0];

        l2_normalize(&mut v1);
        l2_normalize(&mut v2);
        l2_normalize(&mut v3);

        index.add("doc1".to_string(), "tweet", v1.clone());
        index.add("doc2".to_string(), "tweet", v2);
        index.add("doc3".to_string(), "like", v3);

        // Search with query similar to v1
        let results = index.search_top_k(&v1, 2, None);

        assert_eq!(results.len(), 2);
        // doc1 should be first (exact match)
        assert_eq!(results[0].doc_id, "doc1");
        assert!((results[0].score - 1.0).abs() < 0.001);
        // doc2 should be second (similar)
        assert_eq!(results[1].doc_id, "doc2");
    }

    #[test]
    fn test_search_with_type_filter() {
        let mut index = InMemoryVectorIndex::new(4);

        let mut v1 = vec![1.0, 0.0, 0.0, 0.0];
        let mut v2 = vec![0.9, 0.1, 0.0, 0.0];
        let mut v3 = vec![0.8, 0.2, 0.0, 0.0];

        l2_normalize(&mut v1);
        l2_normalize(&mut v2);
        l2_normalize(&mut v3);

        index.add("doc1".to_string(), "tweet", v1.clone());
        index.add("doc2".to_string(), "like", v2);
        index.add("doc3".to_string(), "dm", v3);

        // Search only tweets
        let results = index.search_top_k(&v1, 10, Some(&["tweet"]));
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].doc_id, "doc1");

        // Search tweets and likes
        let results = index.search_top_k(&v1, 10, Some(&["tweet", "like"]));
        assert_eq!(results.len(), 2);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_search_limit() {
        let mut index = InMemoryVectorIndex::new(4);

        // Add 10 vectors
        for i in 0..10 {
            let mut v = vec![1.0, (i as f32) / 10.0, 0.0, 0.0];
            l2_normalize(&mut v);
            index.add(format!("doc{i}"), "tweet", v);
        }

        let query = vec![1.0, 0.0, 0.0, 0.0];
        let results = index.search_top_k(&query, 3, None);
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_search_deterministic() {
        let mut index = InMemoryVectorIndex::new(8);

        // Add vectors with varying similarities
        for i in 0..20 {
            index.add(format!("doc{i}"), "tweet", create_test_vector(i, 8));
        }

        let query = create_test_vector(100, 8);

        // Run search multiple times
        let results1 = index.search_top_k(&query, 5, None);
        let results2 = index.search_top_k(&query, 5, None);
        let results3 = index.search_top_k(&query, 5, None);

        // Results should be identical
        for i in 0..5 {
            assert_eq!(results1[i].doc_id, results2[i].doc_id);
            assert_eq!(results2[i].doc_id, results3[i].doc_id);
        }
    }

    #[test]
    fn test_search_deterministic_with_same_id_types() {
        let mut index = InMemoryVectorIndex::new(4);
        let mut v = vec![1.0, 0.0, 0.0, 0.0];
        l2_normalize(&mut v);

        index.add("same".to_string(), "tweet", v.clone());
        index.add("same".to_string(), "like", v.clone());

        let results = index.search_top_k(&v, 2, None);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].doc_id, "same");
        assert_eq!(results[1].doc_id, "same");
        assert_eq!(results[0].doc_type, "like");
        assert_eq!(results[1].doc_type, "tweet");
    }

    #[test]
    fn test_search_scores_descending() {
        let mut index = InMemoryVectorIndex::new(4);

        let mut v1 = vec![1.0, 0.0, 0.0, 0.0];
        let mut v2 = vec![0.7, 0.3, 0.0, 0.0];
        let mut v3 = vec![0.5, 0.5, 0.0, 0.0];

        l2_normalize(&mut v1);
        l2_normalize(&mut v2);
        l2_normalize(&mut v3);

        index.add("doc1".to_string(), "tweet", v1.clone());
        index.add("doc2".to_string(), "tweet", v2);
        index.add("doc3".to_string(), "tweet", v3);

        let results = index.search_top_k(&v1, 10, None);

        // Verify scores are in descending order
        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "Results not sorted by score"
            );
        }
    }

    #[test]
    fn test_zero_k() {
        let mut index = InMemoryVectorIndex::new(4);
        let mut v = vec![1.0, 0.0, 0.0, 0.0];
        l2_normalize(&mut v);
        index.add("doc1".to_string(), "tweet", v.clone());

        let results = index.search_top_k(&v, 0, None);
        assert!(results.is_empty());
    }

    #[test]
    fn test_dimension_mismatch() {
        let mut index = InMemoryVectorIndex::new(4);
        let mut v = vec![1.0, 0.0, 0.0, 0.0];
        l2_normalize(&mut v);
        index.add("doc1".to_string(), "tweet", v);

        // Query with wrong dimension
        let wrong_query = vec![1.0, 0.0, 0.0, 0.0, 0.0]; // 5 dimensions
        let results = index.search_top_k(&wrong_query, 10, None);
        assert!(results.is_empty());
    }

    // Tests for doc_type encoding
    #[test]
    fn test_encode_doc_type() {
        assert_eq!(encode_doc_type("tweet"), Some(0));
        assert_eq!(encode_doc_type("like"), Some(1));
        assert_eq!(encode_doc_type("dm"), Some(2));
        assert_eq!(encode_doc_type("grok"), Some(3));
        assert_eq!(encode_doc_type("unknown"), None);
    }

    #[test]
    fn test_decode_doc_type() {
        assert_eq!(decode_doc_type(0), Some("tweet"));
        assert_eq!(decode_doc_type(1), Some("like"));
        assert_eq!(decode_doc_type(2), Some("dm"));
        assert_eq!(decode_doc_type(3), Some("grok"));
        assert_eq!(decode_doc_type(4), None);
    }

    #[test]
    fn test_doc_type_roundtrip() {
        for doc_type in &["tweet", "like", "dm", "grok"] {
            let encoded = encode_doc_type(doc_type).unwrap();
            let decoded = decode_doc_type(encoded).unwrap();
            assert_eq!(*doc_type, decoded);
        }
    }

    // Tests for write_vector_index
    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_write_vector_index_empty() {
        let storage = Storage::open_memory().unwrap();
        let temp_dir = tempfile::tempdir().unwrap();

        let stats = write_vector_index(temp_dir.path(), &storage).unwrap();
        assert_eq!(stats.record_count, 0);
        assert_eq!(stats.file_size, 0);

        // No file should be created for empty index
        assert!(!temp_dir.path().join(VECTOR_INDEX_FILENAME).exists());
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_write_vector_index_single_embedding() {
        let storage = Storage::open_memory().unwrap();
        let temp_dir = tempfile::tempdir().unwrap();

        // Store a single embedding
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        storage
            .store_embedding("doc1", "tweet", &embedding, None)
            .unwrap();

        let stats = write_vector_index(temp_dir.path(), &storage).unwrap();
        assert_eq!(stats.record_count, 1);
        assert!(stats.file_size > 0);

        // File should exist and pass validation
        let file_path = temp_dir.path().join(VECTOR_INDEX_FILENAME);
        assert!(file_path.exists());

        let bytes = std::fs::read(&file_path).unwrap();
        validate_vector_index_layout(&bytes).unwrap();
    }

    #[test]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    fn test_write_vector_index_deterministic_ordering() {
        let storage = Storage::open_memory().unwrap();
        let temp_dir = tempfile::tempdir().unwrap();

        // Store embeddings in non-sorted order
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        storage
            .store_embedding("z_doc", "dm", &embedding, None)
            .unwrap();
        storage
            .store_embedding("a_doc", "tweet", &embedding, None)
            .unwrap();
        storage
            .store_embedding("m_doc", "like", &embedding, None)
            .unwrap();

        write_vector_index(temp_dir.path(), &storage).unwrap();

        let file_path = temp_dir.path().join(VECTOR_INDEX_FILENAME);
        let bytes = std::fs::read(&file_path).unwrap();

        // Parse and verify ordering: should be sorted by doc_type (tweet=0, like=1, dm=2)
        // then by doc_id within each type
        let header = parse_vector_index_header(&bytes).unwrap();
        assert_eq!(header.record_count, 3);

        // Read first record's doc_type (should be tweet = 0)
        let first_offset = u64::from_le_bytes(
            bytes[VECTOR_INDEX_HEADER_LEN..VECTOR_INDEX_HEADER_LEN + 8]
                .try_into()
                .unwrap(),
        ) as usize;
        assert_eq!(bytes[first_offset], 0); // tweet

        // Second record should be like = 1
        let second_offset = u64::from_le_bytes(
            bytes[VECTOR_INDEX_HEADER_LEN + 8..VECTOR_INDEX_HEADER_LEN + 16]
                .try_into()
                .unwrap(),
        ) as usize;
        assert_eq!(bytes[second_offset], 1); // like

        // Third record should be dm = 2
        let third_offset = u64::from_le_bytes(
            bytes[VECTOR_INDEX_HEADER_LEN + 16..VECTOR_INDEX_HEADER_LEN + 24]
                .try_into()
                .unwrap(),
        ) as usize;
        assert_eq!(bytes[third_offset], 2); // dm
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_write_vector_index_multiple_runs_identical() {
        let storage = Storage::open_memory().unwrap();
        let temp_dir1 = tempfile::tempdir().unwrap();
        let temp_dir2 = tempfile::tempdir().unwrap();

        // Store some embeddings
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        storage
            .store_embedding("doc1", "tweet", &embedding, None)
            .unwrap();
        storage
            .store_embedding("doc2", "like", &embedding, None)
            .unwrap();

        // Write twice
        write_vector_index(temp_dir1.path(), &storage).unwrap();
        write_vector_index(temp_dir2.path(), &storage).unwrap();

        // Files should be byte-for-byte identical
        let bytes1 = std::fs::read(temp_dir1.path().join(VECTOR_INDEX_FILENAME)).unwrap();
        let bytes2 = std::fs::read(temp_dir2.path().join(VECTOR_INDEX_FILENAME)).unwrap();
        assert_eq!(
            bytes1, bytes2,
            "Multiple writes should produce identical output"
        );
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_write_vector_index_header_values() {
        let storage = Storage::open_memory().unwrap();
        let temp_dir = tempfile::tempdir().unwrap();

        // Store embeddings with dimension 384
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        storage
            .store_embedding("doc1", "tweet", &embedding, None)
            .unwrap();
        storage
            .store_embedding("doc2", "tweet", &embedding, None)
            .unwrap();

        write_vector_index(temp_dir.path(), &storage).unwrap();

        let bytes = std::fs::read(temp_dir.path().join(VECTOR_INDEX_FILENAME)).unwrap();
        let header = parse_vector_index_header(&bytes).unwrap();

        assert_eq!(header.version, VECTOR_INDEX_VERSION);
        assert_eq!(header.doc_type_encoding, VECTOR_INDEX_DOC_TYPE_ENCODING);
        assert_eq!(header.dimension, 384);
        assert_eq!(header.record_count, 2);
        assert_eq!(header.offsets_start, VECTOR_INDEX_HEADER_LEN as u64);
    }

    // Tests for load_from_file (reader)
    #[test]
    fn test_load_from_file_nonexistent() {
        let temp_dir = tempfile::tempdir().unwrap();
        let result = InMemoryVectorIndex::load_from_file(temp_dir.path()).unwrap();
        assert!(result.is_none(), "Should return None for missing file");
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_load_from_file_roundtrip() {
        let storage = Storage::open_memory().unwrap();
        let temp_dir = tempfile::tempdir().unwrap();

        // Store some embeddings with different types
        let embedding1: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        let embedding2: Vec<f32> = (0..384).map(|i| (384 - i) as f32 / 384.0).collect();
        storage
            .store_embedding("doc1", "tweet", &embedding1, None)
            .unwrap();
        storage
            .store_embedding("doc2", "like", &embedding2, None)
            .unwrap();

        // Write to file
        let stats = write_vector_index(temp_dir.path(), &storage).unwrap();
        assert_eq!(stats.record_count, 2);

        // Load from file
        let index = InMemoryVectorIndex::load_from_file(temp_dir.path())
            .unwrap()
            .expect("Should load successfully");

        assert_eq!(index.len(), 2);
        assert_eq!(index.dimension(), 384);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_load_from_file_matches_storage() {
        let storage = Storage::open_memory().unwrap();
        let temp_dir = tempfile::tempdir().unwrap();

        // Store embeddings
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        storage
            .store_embedding("doc1", "tweet", &embedding, None)
            .unwrap();

        // Write to file
        write_vector_index(temp_dir.path(), &storage).unwrap();

        // Load from both sources
        let from_file = InMemoryVectorIndex::load_from_file(temp_dir.path())
            .unwrap()
            .expect("Should load from file");
        let from_storage = InMemoryVectorIndex::load_from_storage(&storage).unwrap();

        // Should have same data
        assert_eq!(from_file.len(), from_storage.len());
        assert_eq!(from_file.dimension(), from_storage.dimension());

        // Search should return same results
        let query: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        let results_file = from_file.search_top_k(&query, 10, None);
        let results_storage = from_storage.search_top_k(&query, 10, None);

        assert_eq!(results_file.len(), results_storage.len());
        for (rf, rs) in results_file.iter().zip(results_storage.iter()) {
            assert_eq!(rf.doc_id, rs.doc_id);
            assert_eq!(rf.doc_type, rs.doc_type);
            // Scores should be very close (F16 precision loss is minimal)
            assert!(
                (rf.score - rs.score).abs() < 0.001,
                "Scores differ: {} vs {}",
                rf.score,
                rs.score
            );
        }
    }

    #[test]
    fn test_load_from_file_invalid_magic() {
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join(VECTOR_INDEX_FILENAME);

        // Write invalid file
        let mut bytes = vec![0u8; 100];
        bytes[0..4].copy_from_slice(b"XXXX"); // Wrong magic
        std::fs::write(&file_path, &bytes).unwrap();

        // Should return None (fall back to DB)
        let result = InMemoryVectorIndex::load_from_file(temp_dir.path()).unwrap();
        assert!(result.is_none(), "Should return None for invalid magic");
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_load_from_file_or_storage_prefers_file() {
        let storage = Storage::open_memory().unwrap();
        let temp_dir = tempfile::tempdir().unwrap();

        // Store embeddings in both
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        storage
            .store_embedding("doc1", "tweet", &embedding, None)
            .unwrap();

        // Write to file
        write_vector_index(temp_dir.path(), &storage).unwrap();

        // Add more to storage (file is now stale)
        storage
            .store_embedding("doc2", "like", &embedding, None)
            .unwrap();

        // load_from_file_or_storage should load from file (2 in storage, 1 in file)
        let index = VectorIndex::load_from_file_or_storage(temp_dir.path(), &storage).unwrap();

        // Should have file version (1 record, not 2)
        assert_eq!(index.len(), 1, "Should load from file, not storage");
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_load_from_file_or_storage_falls_back() {
        let storage = Storage::open_memory().unwrap();
        let temp_dir = tempfile::tempdir().unwrap();

        // Store embedding but don't write file
        let embedding: Vec<f32> = (0..384).map(|i| i as f32 / 384.0).collect();
        storage
            .store_embedding("doc1", "tweet", &embedding, None)
            .unwrap();

        // load_from_file_or_storage should fall back to storage
        let index = VectorIndex::load_from_file_or_storage(temp_dir.path(), &storage).unwrap();

        assert_eq!(index.len(), 1, "Should fall back to storage");
    }

    // ========================================================================
    // xf-70: Vector index regression tests
    // ========================================================================

    #[test]
    fn test_load_from_file_truncated_header() {
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join(VECTOR_INDEX_FILENAME);

        // Write truncated header (only 16 bytes instead of 32)
        let mut bytes = vec![0u8; 16];
        bytes[0..4].copy_from_slice(&VECTOR_INDEX_MAGIC);
        bytes[4..6].copy_from_slice(&VECTOR_INDEX_VERSION.to_le_bytes());
        std::fs::write(&file_path, &bytes).unwrap();

        // Should return None (invalid file)
        let result = InMemoryVectorIndex::load_from_file(temp_dir.path()).unwrap();
        assert!(result.is_none(), "Should return None for truncated header");
    }

    #[test]
    fn test_load_from_file_version_mismatch() {
        let temp_dir = tempfile::tempdir().unwrap();
        let file_path = temp_dir.path().join(VECTOR_INDEX_FILENAME);

        // Write header with wrong version (99 instead of 1)
        let mut bytes = vec![0u8; 100];
        bytes[0..4].copy_from_slice(&VECTOR_INDEX_MAGIC);
        bytes[4..6].copy_from_slice(&99u16.to_le_bytes()); // Wrong version
        bytes[6] = VECTOR_INDEX_DOC_TYPE_ENCODING;
        bytes[7] = 0; // padding
        bytes[8..12].copy_from_slice(&384u32.to_le_bytes()); // dimension
        bytes[12..20].copy_from_slice(&0u64.to_le_bytes()); // record_count
        bytes[20..28].copy_from_slice(&32u64.to_le_bytes()); // offset_table_offset
        bytes[28..32].copy_from_slice(&[0u8; 4]); // reserved
        std::fs::write(&file_path, &bytes).unwrap();

        // Should return None (unsupported version)
        let result = InMemoryVectorIndex::load_from_file(temp_dir.path()).unwrap();
        assert!(result.is_none(), "Should return None for version mismatch");
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_load_from_file_truncated_record() {
        let temp_dir = tempfile::tempdir().unwrap();
        let storage = Storage::open_memory().unwrap();

        // Store an embedding
        let embedding: Vec<f32> = (0..384).map(|i| (i as f32) / 384.0).collect();
        storage
            .store_embedding("doc1", "tweet", &embedding, None)
            .unwrap();

        // Write valid file
        write_vector_index(temp_dir.path(), &storage).unwrap();

        // Read and truncate
        let file_path = temp_dir.path().join(VECTOR_INDEX_FILENAME);
        let bytes = std::fs::read(&file_path).unwrap();

        // Truncate mid-record (remove last 100 bytes)
        let truncated = &bytes[..bytes.len().saturating_sub(100)];
        std::fs::write(&file_path, truncated).unwrap();

        // Should return None (truncated record data)
        let result = InMemoryVectorIndex::load_from_file(temp_dir.path()).unwrap();
        assert!(result.is_none(), "Should return None for truncated record");
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_load_from_file_corrupted_offset_table() {
        let temp_dir = tempfile::tempdir().unwrap();
        let storage = Storage::open_memory().unwrap();

        // Store multiple embeddings
        for i in 0..5 {
            let embedding: Vec<f32> = (0..384).map(|j| ((i * 384 + j) as f32) / 1920.0).collect();
            storage
                .store_embedding(&format!("doc{i}"), "tweet", &embedding, None)
                .unwrap();
        }

        // Write valid file
        write_vector_index(temp_dir.path(), &storage).unwrap();

        // Read and corrupt offset table
        let file_path = temp_dir.path().join(VECTOR_INDEX_FILENAME);
        let mut bytes = std::fs::read(&file_path).unwrap();

        // Corrupt offset table entry (set to impossibly large value)
        if bytes.len() > 40 {
            bytes[32..40].copy_from_slice(&u64::MAX.to_le_bytes());
        }
        std::fs::write(&file_path, &bytes).unwrap();

        // Should return None (corrupted offset)
        let result = InMemoryVectorIndex::load_from_file(temp_dir.path()).unwrap();
        assert!(result.is_none(), "Should return None for corrupted offset");
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_search_isomorphism_file_vs_storage() {
        // Verify that searching the file-loaded index produces identical
        // results to the storage-loaded index
        let storage = Storage::open_memory().unwrap();
        let temp_dir = tempfile::tempdir().unwrap();

        // Create diverse embeddings
        let embeddings: Vec<Vec<f32>> = (0..10)
            .map(|i| {
                (0..384)
                    .map(|j| {
                        // Create somewhat distinct embeddings
                        let base = (i * 100 + j) as f32;
                        f32::midpoint(base.sin(), base.cos())
                    })
                    .collect()
            })
            .collect();

        // Store embeddings with different types
        for (i, emb) in embeddings.iter().enumerate() {
            let doc_type = if i % 3 == 0 {
                "tweet"
            } else if i % 3 == 1 {
                "like"
            } else {
                "dm"
            };
            storage
                .store_embedding(&format!("doc{i}"), doc_type, emb, None)
                .unwrap();
        }

        // Write to file
        write_vector_index(temp_dir.path(), &storage).unwrap();

        // Load both ways
        let file_index = InMemoryVectorIndex::load_from_file(temp_dir.path())
            .unwrap()
            .expect("Should load from file");
        let storage_index = InMemoryVectorIndex::load_from_storage(&storage).unwrap();

        // Use first embedding as query
        let query = &embeddings[0];

        // Search both indices
        let file_results = file_index.search_top_k(query, 5, None);
        let storage_results = storage_index.search_top_k(query, 5, None);

        // Verify isomorphism
        assert_eq!(
            file_results.len(),
            storage_results.len(),
            "Same number of results"
        );

        for (file_r, storage_r) in file_results.iter().zip(storage_results.iter()) {
            assert_eq!(file_r.doc_id, storage_r.doc_id, "Same doc_id");
            assert_eq!(file_r.doc_type, storage_r.doc_type, "Same doc_type");
            // Allow small floating point differences
            let score_diff = (file_r.score - storage_r.score).abs();
            assert!(
                score_diff < 1e-5,
                "Scores should match: {} vs {} (diff: {})",
                file_r.score,
                storage_r.score,
                score_diff
            );
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_search_isomorphism_with_type_filter() {
        // Verify type-filtered searches produce identical results
        let storage = Storage::open_memory().unwrap();
        let temp_dir = tempfile::tempdir().unwrap();

        // Create embeddings
        for i in 0..20 {
            let embedding: Vec<f32> = (0..384).map(|j| ((i * 50 + j) as f32).sin()).collect();
            let doc_type = match i % 4 {
                0 => "tweet",
                1 => "like",
                2 => "dm",
                _ => "grok",
            };
            storage
                .store_embedding(&format!("doc{i}"), doc_type, &embedding, None)
                .unwrap();
        }

        // Write to file
        write_vector_index(temp_dir.path(), &storage).unwrap();

        // Load both ways
        let file_index = InMemoryVectorIndex::load_from_file(temp_dir.path())
            .unwrap()
            .expect("Should load from file");
        let storage_index = InMemoryVectorIndex::load_from_storage(&storage).unwrap();

        // Query embedding
        let query: Vec<f32> = (0..384).map(|j| (j as f32).sin()).collect();

        // Search with type filter
        let filter = ["tweet", "dm"];
        let file_results = file_index.search_top_k(&query, 10, Some(&filter));
        let storage_results = storage_index.search_top_k(&query, 10, Some(&filter));

        // Verify same results
        assert_eq!(file_results.len(), storage_results.len());
        for (f, s) in file_results.iter().zip(storage_results.iter()) {
            assert_eq!(f.doc_id, s.doc_id);
            assert_eq!(f.doc_type, s.doc_type);
            assert!(
                (f.score - s.score).abs() < 1e-5,
                "Score mismatch: {} vs {}",
                f.score,
                s.score
            );
        }

        // Verify only filtered types returned
        for r in &file_results {
            assert!(
                r.doc_type == "tweet" || r.doc_type == "dm",
                "Unexpected type: {}",
                r.doc_type
            );
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_search_order_determinism() {
        // Verify search results are deterministic (same order across runs)
        let storage = Storage::open_memory().unwrap();
        let temp_dir = tempfile::tempdir().unwrap();

        // Create embeddings with similar scores to test tie-breaking
        for i in 0..10 {
            let embedding: Vec<f32> = (0..384)
                .map(|j| {
                    if j == 0 {
                        1.0 // All have same first component
                    } else {
                        (i as f32).mul_add(0.001, j as f32 * 0.0001)
                    }
                })
                .collect();
            storage
                .store_embedding(&format!("doc{i}"), "tweet", &embedding, None)
                .unwrap();
        }

        write_vector_index(temp_dir.path(), &storage).unwrap();
        let index = InMemoryVectorIndex::load_from_file(temp_dir.path())
            .unwrap()
            .expect("Should load");

        // Run search multiple times
        let query: Vec<f32> = (0..384).map(|j| if j == 0 { 1.0 } else { 0.0 }).collect();

        let results1 = index.search_top_k(&query, 5, None);
        let results2 = index.search_top_k(&query, 5, None);
        let results3 = index.search_top_k(&query, 5, None);

        // All runs should produce identical results
        for i in 0..results1.len() {
            assert_eq!(results1[i].doc_id, results2[i].doc_id);
            assert_eq!(results1[i].doc_id, results3[i].doc_id);
        }
    }

    #[test]
    fn test_write_and_load_named_index() {
        let storage = Storage::open_memory().unwrap();
        let temp_dir = tempfile::tempdir().unwrap();

        // Store embeddings with model_id "fast"
        let emb = create_test_vector(1, 384);
        storage
            .store_embedding_with_model("doc1", "tweet", &emb, None, "fast")
            .unwrap();
        let emb2 = create_test_vector(2, 384);
        storage
            .store_embedding_with_model("doc2", "tweet", &emb2, None, "fast")
            .unwrap();

        // Write named index
        let stats = write_vector_index_named(
            temp_dir.path(),
            &storage,
            VECTOR_INDEX_FAST_FILENAME,
            Some("fast"),
        )
        .unwrap();
        assert_eq!(stats.record_count, 2);
        assert!(stats.file_size > 0);

        // Load via VectorIndex::load_named
        let loaded = VectorIndex::load_named(temp_dir.path(), VECTOR_INDEX_FAST_FILENAME)
            .unwrap()
            .expect("named index should exist");
        assert_eq!(loaded.len(), 2);
    }

    #[test]
    fn test_named_coexists_with_default() {
        let storage = Storage::open_memory().unwrap();
        let temp_dir = tempfile::tempdir().unwrap();

        // Store "default" embeddings
        let emb = create_test_vector(1, 384);
        storage
            .store_embedding("doc1", "tweet", &emb, None)
            .unwrap();

        // Store "fast" embeddings
        let fast_emb = create_test_vector(2, 384);
        storage
            .store_embedding_with_model("doc1", "tweet", &fast_emb, None, "fast")
            .unwrap();
        storage
            .store_embedding_with_model("doc2", "tweet", &fast_emb, None, "fast")
            .unwrap();

        // Write both
        let default_stats = write_vector_index(temp_dir.path(), &storage).unwrap();
        let fast_stats = write_vector_index_named(
            temp_dir.path(),
            &storage,
            VECTOR_INDEX_FAST_FILENAME,
            Some("fast"),
        )
        .unwrap();

        // Default includes all embeddings (default + fast)
        assert_eq!(default_stats.record_count, 3);
        // Fast only includes fast embeddings
        assert_eq!(fast_stats.record_count, 2);

        // Both files should exist
        assert!(temp_dir.path().join(VECTOR_INDEX_FILENAME).exists());
        assert!(temp_dir.path().join(VECTOR_INDEX_FAST_FILENAME).exists());

        // Load both
        let default_idx = VectorIndex::load_named(temp_dir.path(), VECTOR_INDEX_FILENAME)
            .unwrap()
            .expect("default index should exist");
        let fast_idx = VectorIndex::load_named(temp_dir.path(), VECTOR_INDEX_FAST_FILENAME)
            .unwrap()
            .expect("fast index should exist");

        assert_eq!(default_idx.len(), 3);
        assert_eq!(fast_idx.len(), 2);
    }

    #[test]
    fn test_load_named_nonexistent() {
        let temp_dir = tempfile::tempdir().unwrap();
        let result =
            VectorIndex::load_named(temp_dir.path(), VECTOR_INDEX_QUALITY_FILENAME).unwrap();
        assert!(result.is_none());
    }
}
