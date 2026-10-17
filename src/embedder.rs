//! Embedder trait and types for text embedding.
//!
//! Provides a common interface for converting text into dense vectors
//! for semantic similarity search.

use asupersync::Cx;
use frankensearch_core::{
    Embedder as FrankensearchEmbedder, ModelCategory as FrankensearchModelCategory, SearchError,
    SearchFuture, SearchResult as FrankensearchSearchResult,
};
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Category of embedding model (for benchmark classification).
///
/// This distinguishes between fast static embedders (feature hashing, Model2Vec)
/// and slower but more capable transformer-based embedders (MiniLM, BGE, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelCategory {
    /// Static/hashing embedders: FNV-1a hash, Model2Vec, static-retrieval-mrl.
    /// Very fast (~0ms per embedding), no GPU required.
    StaticEmbedder,
    /// Transformer-based embedders: MiniLM, BGE, Nomic, Gemma (ONNX inference).
    /// Slower (~5-100ms per embedding) but more semantically aware.
    TransformerEmbedder,
}

impl std::fmt::Display for ModelCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::StaticEmbedder => write!(f, "static"),
            Self::TransformerEmbedder => write!(f, "transformer"),
        }
    }
}

/// Errors that can occur during embedding operations.
#[derive(Debug, Error)]
pub enum EmbedderError {
    /// The embedder is not available (e.g., model files missing).
    #[error("embedder unavailable: {0}")]
    Unavailable(String),

    /// Failed to generate an embedding.
    #[error("embedding failed: {0}")]
    EmbeddingFailed(String),

    /// Invalid input provided to the embedder.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Internal error during embedding.
    #[error("internal error: {0}")]
    Internal(String),
}

/// Result type for embedder operations.
pub type EmbedderResult<T> = Result<T, EmbedderError>;

/// Split the complete original text into overlapping UTF-8 byte ranges.
/// Each window contains at most 800 characters and overlaps the next by 20%.
/// Transformer implementations additionally enforce their tokenizer's limit.
/// No markdown, code, URLs, or tail text is discarded.
pub(crate) fn overlapping_chunks(
    text: &str,
    fits: impl Fn(&str) -> EmbedderResult<bool>,
) -> EmbedderResult<Vec<std::ops::Range<usize>>> {
    let mut boundaries: Vec<usize> = text.char_indices().map(|(offset, _)| offset).collect();
    boundaries.push(text.len());
    let chars = boundaries.len() - 1;
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < chars {
        let mut end = (start + 800).min(chars);
        if !fits(&text[boundaries[start]..boundaries[end]])? {
            // Find a fitting prefix without ever relying on silent truncation.
            let mut low = start;
            let mut high = end;
            while low + 1 < high {
                let mid = low + (high - low) / 2;
                if fits(&text[boundaries[start]..boundaries[mid]])? {
                    low = mid;
                } else {
                    high = mid;
                }
            }
            end = low;
            if end == start {
                return Err(EmbedderError::InvalidInput(
                    "a single character exceeds the embedding token budget".to_string(),
                ));
            }
        }
        chunks.push(boundaries[start]..boundaries[end]);
        if end == chars {
            break;
        }
        let overlap = (end - start) / 5;
        start = end - overlap;
    }
    Ok(chunks)
}

/// Information about an embedder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbedderInfo {
    /// Unique identifier for this embedder (e.g., "fnv1a-384", "minilm-384").
    pub id: String,
    /// Human-readable model name (registry key).
    pub model_name: String,
    /// Output dimension of embeddings.
    pub dimension: usize,
    /// Whether this is a semantic (ML-based) embedder.
    pub is_semantic: bool,
    /// Whether this embedder supports MRL truncation.
    pub supports_mrl: bool,
    /// Model category for benchmark classification.
    pub category: ModelCategory,
}

/// Trait for text embedding implementations.
///
/// Embedders convert text into fixed-dimension dense vectors that can be
/// compared using cosine similarity (or dot product for normalized vectors).
///
/// # Implementations
///
/// - [`HashEmbedder`](crate::hash_embedder::HashEmbedder): Fast, deterministic
///   hash-based embeddings using FNV-1a. Always available, ~0ms per embedding.
///
/// - `FastEmbedder` (optional): ML-based semantic embeddings using `MiniLM`.
///   Requires model files, ~5ms per embedding.
pub trait Embedder: Send + Sync {
    /// Full-document windows, expressed as byte ranges in the original text.
    /// Models with a token limit must override this to check every window.
    fn document_chunks(&self, text: &str) -> EmbedderResult<Vec<std::ops::Range<usize>>> {
        overlapping_chunks(text, |_| Ok(true))
    }

    /// Embed a single text into a dense vector.
    ///
    /// The returned vector is L2-normalized (unit length).
    ///
    /// # Errors
    ///
    /// Returns an error if the input is invalid or embedding fails.
    fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>>;

    /// Embed multiple texts in a batch.
    ///
    /// Default implementation calls `embed` for each text.
    /// Implementations may override for better batching performance.
    ///
    /// # Errors
    ///
    /// Returns an error if any input is invalid or embedding fails.
    fn embed_batch(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }

    /// Get the output dimension of embeddings.
    fn dimension(&self) -> usize;

    /// Get the unique identifier for this embedder.
    fn id(&self) -> &str;

    /// Human-readable model name (registry key).
    ///
    /// Defaults to `id()` if not overridden.
    fn model_name(&self) -> &str {
        self.id()
    }

    /// Check if this is a semantic (ML-based) embedder.
    ///
    /// Hash-based embedders return `false`, ML embedders return `true`.
    fn is_semantic(&self) -> bool;

    /// Get the model category for benchmark classification.
    ///
    /// Defaults to `TransformerEmbedder` for semantic models,
    /// `StaticEmbedder` for non-semantic models.
    fn category(&self) -> ModelCategory {
        if self.is_semantic() {
            ModelCategory::TransformerEmbedder
        } else {
            ModelCategory::StaticEmbedder
        }
    }

    /// Whether this model supports Matryoshka truncation.
    ///
    /// Defaults to `false`.
    fn supports_mrl(&self) -> bool {
        false
    }

    /// Truncate an embedding to a target dimension.
    ///
    /// By default, truncation is supported only when `supports_mrl()` is true.
    fn truncate_embedding(&self, embedding: &[f32], target_dim: usize) -> EmbedderResult<Vec<f32>> {
        if !self.supports_mrl() {
            return Err(EmbedderError::InvalidInput(
                "model does not support MRL truncation".to_string(),
            ));
        }
        if target_dim == 0 || target_dim > embedding.len() {
            return Err(EmbedderError::InvalidInput(format!(
                "invalid target_dim {target_dim} for embedding length {}",
                embedding.len()
            )));
        }
        Ok(embedding[..target_dim].to_vec())
    }

    /// Truncate a batch of embeddings to a target dimension.
    fn truncate_batch(
        &self,
        embeddings: &[Vec<f32>],
        target_dim: usize,
    ) -> EmbedderResult<Vec<Vec<f32>>> {
        embeddings
            .iter()
            .map(|e| self.truncate_embedding(e, target_dim))
            .collect()
    }

    /// Get information about this embedder.
    fn info(&self) -> EmbedderInfo {
        EmbedderInfo {
            id: self.id().to_string(),
            model_name: self.model_name().to_string(),
            dimension: self.dimension(),
            is_semantic: self.is_semantic(),
            supports_mrl: self.supports_mrl(),
            category: self.category(),
        }
    }
}

/// L2-normalize a vector in place.
///
/// After normalization, the vector has unit length (L2 norm = 1.0).
/// This allows cosine similarity to be computed as a simple dot product.
#[inline]
pub fn l2_normalize(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

/// Compute the dot product of two vectors.
///
/// For L2-normalized vectors, this equals cosine similarity.
#[inline]
#[must_use]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute the dot product using SIMD (8 floats per iteration).
///
/// Falls back to scalar for remainder elements.
///
/// # Panics
///
/// Panics if the internal SIMD chunk conversion fails (should be unreachable
/// because chunks are always length 8).
#[inline]
#[must_use]
pub fn dot_product_simd(a: &[f32], b: &[f32]) -> f32 {
    use wide::f32x8;

    debug_assert_eq!(a.len(), b.len(), "vectors must have same dimension");

    let chunks_a = a.chunks_exact(8);
    let chunks_b = b.chunks_exact(8);
    let remainder_a = chunks_a.remainder();
    let remainder_b = chunks_b.remainder();

    let mut sum = f32x8::ZERO;
    for (ca, cb) in chunks_a.zip(chunks_b) {
        let arr_a: [f32; 8] = ca.try_into().unwrap();
        let arr_b: [f32; 8] = cb.try_into().unwrap();
        sum += f32x8::from(arr_a) * f32x8::from(arr_b);
    }

    let mut scalar_sum: f32 = sum.reduce_add();
    for (a, b) in remainder_a.iter().zip(remainder_b) {
        scalar_sum += a * b;
    }
    scalar_sum
}

fn map_embedder_error(model: &str, err: EmbedderError) -> SearchError {
    match err {
        EmbedderError::Unavailable(reason) => SearchError::EmbedderUnavailable {
            model: model.to_string(),
            reason,
        },
        EmbedderError::InvalidInput(value) => SearchError::InvalidConfig {
            field: "embedder_input".to_string(),
            value,
            reason: "invalid embedder input".to_string(),
        },
        other => SearchError::EmbeddingFailed {
            model: model.to_string(),
            source: Box::new(other),
        },
    }
}

const fn map_model_category(category: ModelCategory) -> FrankensearchModelCategory {
    match category {
        ModelCategory::StaticEmbedder => FrankensearchModelCategory::StaticEmbedder,
        ModelCategory::TransformerEmbedder => FrankensearchModelCategory::TransformerEmbedder,
    }
}

/// Adapter from xf's synchronous embedder trait to frankensearch's async trait.
pub struct FrankensearchEmbedderAdapter {
    inner: Box<dyn Embedder>,
    adapter_id: String,
}

impl FrankensearchEmbedderAdapter {
    #[must_use]
    pub fn new(inner: Box<dyn Embedder>) -> Self {
        let adapter_id = format!("xf-adapter::{}", inner.id());
        Self { inner, adapter_id }
    }

    #[must_use]
    pub fn into_inner(self) -> Box<dyn Embedder> {
        self.inner
    }
}

impl FrankensearchEmbedder for FrankensearchEmbedderAdapter {
    fn embed<'a>(&'a self, _cx: &'a Cx, text: &'a str) -> SearchFuture<'a, Vec<f32>> {
        Box::pin(async move {
            self.inner
                .embed(text)
                .map_err(|err| map_embedder_error(self.inner.model_name(), err))
        })
    }

    fn embed_batch<'a>(
        &'a self,
        _cx: &'a Cx,
        texts: &'a [&'a str],
    ) -> SearchFuture<'a, Vec<Vec<f32>>> {
        Box::pin(async move {
            self.inner
                .embed_batch(texts)
                .map_err(|err| map_embedder_error(self.inner.model_name(), err))
        })
    }

    fn dimension(&self) -> usize {
        self.inner.dimension()
    }

    fn id(&self) -> &str {
        &self.adapter_id
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn is_semantic(&self) -> bool {
        self.inner.is_semantic()
    }

    fn category(&self) -> FrankensearchModelCategory {
        map_model_category(self.inner.category())
    }

    fn supports_mrl(&self) -> bool {
        self.inner.supports_mrl()
    }

    fn truncate_embedding(
        &self,
        embedding: &[f32],
        target_dim: usize,
    ) -> FrankensearchSearchResult<Vec<f32>> {
        self.inner
            .truncate_embedding(embedding, target_dim)
            .map_err(|err| map_embedder_error(self.inner.model_name(), err))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_preserve_every_byte_and_overlap_without_splitting_unicode() {
        let text = format!(
            "{}{}",
            "α🦀中文 code_with_marks **text**\n".repeat(200),
            "UNIQUE TAIL"
        );
        let chunks = overlapping_chunks(&text, |s| Ok(s.chars().count() <= 211)).unwrap();
        assert_eq!(chunks.first().unwrap().start, 0);
        assert_eq!(chunks.last().unwrap().end, text.len());
        for range in &chunks {
            assert!(text.is_char_boundary(range.start) && text.is_char_boundary(range.end));
            assert!(text[range.clone()].chars().count() <= 211);
        }
        for pair in chunks.windows(2) {
            assert!(pair[0].start < pair[1].start);
            assert!(pair[0].end > pair[1].start);
            assert!(pair[0].end < pair[1].end);
        }
        assert!(text[chunks.last().unwrap().clone()].ends_with("UNIQUE TAIL"));
        assert_eq!(overlapping_chunks("", |_| Ok(true)).unwrap().len(), 0);
        assert_eq!(overlapping_chunks("é", |_| Ok(true)).unwrap(), vec![0..2]);
        assert!(overlapping_chunks("x", |_| Ok(false)).is_err());
    }

    #[test]
    fn test_l2_normalize() {
        let mut vec = vec![3.0, 4.0];
        l2_normalize(&mut vec);

        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((vec[0] - 0.6).abs() < 1e-6);
        assert!((vec[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let mut vec = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut vec);
        // Should not panic, vector remains zero
        assert!(vec.iter().all(|x| *x == 0.0));
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert!((result - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_dot_product_simd() {
        // Test with 16 elements (exactly 2 SIMD chunks)
        let a: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..16).map(|i| (i * 2) as f32).collect();

        let scalar = dot_product(&a, &b);
        let simd = dot_product_simd(&a, &b);

        assert!((scalar - simd).abs() < 1e-4);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_dot_product_simd_with_remainder() {
        // Test with 19 elements (2 chunks + 3 remainder)
        let a: Vec<f32> = (0..19).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..19).map(|i| (i + 1) as f32).collect();

        let scalar = dot_product(&a, &b);
        let simd = dot_product_simd(&a, &b);

        assert!((scalar - simd).abs() < 1e-4);
    }

    #[test]
    fn test_normalized_dot_is_cosine() {
        let mut a = vec![1.0, 2.0, 3.0];
        let mut b = vec![4.0, 5.0, 6.0];

        l2_normalize(&mut a);
        l2_normalize(&mut b);

        let cosine = dot_product(&a, &b);
        // Cosine similarity of [1,2,3] and [4,5,6] = 32 / (√14 * √77) ≈ 0.9746
        assert!((cosine - 0.9746).abs() < 0.001);
    }
}
