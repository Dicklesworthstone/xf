//! Model2Vec embedding model backend.
//!
//! Implements static embeddings using Model2Vec's distilled word2vec-style approach:
//! subword tokenization + static embedding lookup + mean pooling.
//!
//! # Key Features
//!
//! - Extremely fast inference (~0ms, no transformer computation)
//! - No ONNX runtime needed (pure embedding lookup)
//! - Memory: ~32M params × 4 bytes = ~128MB resident for potion-retrieval-32M
//!
//! # Model Files
//!
//! - `tokenizer.json` - HuggingFace tokenizer
//! - `model.safetensors` - Static embedding weights
//! - `config.json` - Model configuration (optional)

use std::path::{Path, PathBuf};

use safetensors::SafeTensors;
use tokenizers::Tokenizer;

#[cfg(not(feature = "frankensearch-migration"))]
use crate::embedder::l2_normalize;
use crate::embedder::{Embedder, EmbedderError, EmbedderResult, ModelCategory};
#[cfg(feature = "frankensearch-migration")]
use frankensearch_embed::Model2VecEmbedder as FsModel2VecEmbedder;

/// Model name constant for potion-retrieval-32M.
pub const MODEL_POTION_32M: &str = "potion-retrieval-32M";

/// Model name constant for potion-multilingual-128M.
pub const MODEL_POTION_MULTI_128M: &str = "potion-multilingual-128M";

/// Required model files.
const REQUIRED_FILES: &[&str] = &["tokenizer.json", "model.safetensors"];

/// Model2Vec embedder using static embedding lookup.
///
/// This embedder loads a tokenizer and an embedding matrix, then performs:
/// 1. Subword tokenization
/// 2. Embedding lookup for each token
/// 3. Mean pooling over tokens
/// 4. L2 normalization
pub struct Model2VecEmbedder {
    /// Subword tokenizer (BPE or WordPiece from teacher model).
    #[cfg_attr(feature = "frankensearch-migration", allow(dead_code))]
    tokenizer: Tokenizer,
    /// Static embedding matrix [vocab_size × dims].
    #[cfg_attr(feature = "frankensearch-migration", allow(dead_code))]
    embeddings: Vec<Vec<f32>>,
    /// Output dimensions.
    dimensions: usize,
    /// Model identifier (e.g., "potion-retrieval-32M").
    name: String,
    /// Vocabulary size.
    vocab_size: usize,
    #[cfg(feature = "frankensearch-migration")]
    delegate: FsModel2VecEmbedder,
}

impl Model2VecEmbedder {
    /// Load the model from a directory containing model files.
    ///
    /// # Errors
    ///
    /// Returns an error if the model files are missing or invalid.
    pub fn load_from_dir(model_dir: &Path, model_name: &str) -> EmbedderResult<Self> {
        // Validate required files
        for file in REQUIRED_FILES {
            let file_path = model_dir.join(file);
            if !file_path.exists() {
                return Err(EmbedderError::Unavailable(format!(
                    "missing required model file: {}",
                    file_path.display()
                )));
            }
        }

        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedderError::Internal(format!("failed to load tokenizer: {e}")))?;

        // Load embeddings from safetensors
        let embeddings_path = model_dir.join("model.safetensors");
        let embeddings_data = std::fs::read(&embeddings_path)
            .map_err(|e| EmbedderError::Internal(format!("failed to read embeddings: {e}")))?;

        let safetensors = SafeTensors::deserialize(&embeddings_data)
            .map_err(|e| EmbedderError::Internal(format!("failed to parse safetensors: {e}")))?;

        // Find the embedding tensor - Model2Vec typically stores as "embeddings" or "embedding"
        let tensor_name = Self::find_embedding_tensor_name(&safetensors)?;
        let tensor = safetensors.tensor(&tensor_name).map_err(|e| {
            EmbedderError::Internal(format!("failed to get tensor {tensor_name}: {e}"))
        })?;

        // Validate tensor shape [vocab_size, dims]
        let shape = tensor.shape();
        if shape.len() != 2 {
            return Err(EmbedderError::Internal(format!(
                "expected 2D tensor, got shape: {shape:?}"
            )));
        }
        let vocab_size = shape[0];
        let dimensions = shape[1];

        // Convert tensor data to embedding vectors
        let embeddings = Self::tensor_to_embeddings(tensor.data(), vocab_size, dimensions)?;

        tracing::info!(
            model = model_name,
            vocab_size = vocab_size,
            dimensions = dimensions,
            "Model2Vec embedder loaded"
        );

        #[cfg(feature = "frankensearch-migration")]
        let delegate = FsModel2VecEmbedder::load_with_name(model_dir, model_name).map_err(|e| {
            EmbedderError::Unavailable(format!(
                "frankensearch model2vec load failed for {}: {e}",
                model_dir.display()
            ))
        })?;

        Ok(Self {
            tokenizer,
            embeddings,
            dimensions,
            name: model_name.to_string(),
            vocab_size,
            #[cfg(feature = "frankensearch-migration")]
            delegate,
        })
    }

    /// Find the embedding tensor name in a safetensors file.
    fn find_embedding_tensor_name(safetensors: &SafeTensors<'_>) -> EmbedderResult<String> {
        let names: Vec<String> = safetensors.names().into_iter().cloned().collect();

        // Try common embedding tensor names
        for candidate in &["embeddings", "embedding", "word_embeddings", "embed", "emb"] {
            if names.contains(&candidate.to_string()) {
                return Ok((*candidate).to_string());
            }
        }

        // If only one tensor, use it
        if names.len() == 1 {
            return Ok(names[0].clone());
        }

        Err(EmbedderError::Internal(format!(
            "could not find embedding tensor. Available: {names:?}"
        )))
    }

    /// Convert raw tensor bytes to embedding vectors.
    fn tensor_to_embeddings(
        data: &[u8],
        vocab_size: usize,
        dimensions: usize,
    ) -> EmbedderResult<Vec<Vec<f32>>> {
        // Expect f32 data (4 bytes per float)
        let expected_bytes = vocab_size * dimensions * 4;
        if data.len() != expected_bytes {
            return Err(EmbedderError::Internal(format!(
                "tensor size mismatch: expected {expected_bytes} bytes, got {}",
                data.len()
            )));
        }

        let mut embeddings = Vec::with_capacity(vocab_size);
        for v in 0..vocab_size {
            let mut row = Vec::with_capacity(dimensions);
            for d in 0..dimensions {
                let offset = (v * dimensions + d) * 4;
                let bytes: [u8; 4] = data[offset..offset + 4].try_into().map_err(|_| {
                    EmbedderError::Internal("byte slice conversion failed".to_string())
                })?;
                row.push(f32::from_le_bytes(bytes));
            }
            embeddings.push(row);
        }

        Ok(embeddings)
    }

    /// Try to load from standard model locations.
    ///
    /// Searches in order:
    /// 1. `~/.cache/xf/models/<model_name>`
    /// 2. `~/.local/share/xf/models/<model_name>`
    /// 3. `~/.cache/huggingface/hub/models--minishlab--<model_name>`
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be found.
    pub fn try_load(model_name: &str) -> EmbedderResult<Self> {
        let candidates = Self::model_search_paths(model_name);

        for candidate in &candidates {
            if candidate.exists() {
                // For HuggingFace hub cache, we need to find the snapshot directory
                if candidate.to_string_lossy().contains("huggingface") {
                    if let Some(snapshot_dir) = Self::find_hf_snapshot(candidate) {
                        if let Ok(embedder) = Self::load_from_dir(&snapshot_dir, model_name) {
                            return Ok(embedder);
                        }
                    }
                } else if let Ok(embedder) = Self::load_from_dir(candidate, model_name) {
                    return Ok(embedder);
                }
            }
        }

        Err(EmbedderError::Unavailable(format!(
            "{model_name} model not found. Searched: {}",
            candidates
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )))
    }

    /// Find the latest snapshot directory in a HuggingFace hub cache.
    fn find_hf_snapshot(hub_path: &Path) -> Option<PathBuf> {
        let snapshots_dir = hub_path.join("snapshots");
        if !snapshots_dir.exists() {
            return None;
        }

        // Get the most recent snapshot (usually only one)
        std::fs::read_dir(&snapshots_dir)
            .ok()?
            .filter_map(Result::ok)
            .filter(|e| e.file_type().ok().is_some_and(|ft| ft.is_dir()))
            .map(|e| e.path())
            .max_by_key(|p| {
                std::fs::metadata(p)
                    .and_then(|m| m.modified())
                    .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
            })
    }

    /// Get standard model search paths.
    #[must_use]
    pub fn model_search_paths(model_name: &str) -> Vec<PathBuf> {
        let mut paths = Vec::new();

        // xf cache directory
        if let Some(cache) = dirs::cache_dir() {
            paths.push(cache.join("xf").join("models").join(model_name));
        }

        // xf data directory
        if let Some(data) = dirs::data_local_dir() {
            paths.push(data.join("xf").join("models").join(model_name));
        }

        // HuggingFace hub cache
        if let Some(cache) = dirs::cache_dir() {
            paths.push(
                cache
                    .join("huggingface")
                    .join("hub")
                    .join(format!("models--minishlab--{model_name}")),
            );
        }

        paths
    }

    /// Check if a specific model is available.
    #[must_use]
    pub fn is_available(model_name: &str) -> bool {
        Self::try_load(model_name).is_ok()
    }

    /// Get the vocabulary size.
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Embed a single text using static lookup + mean pooling.
    fn embed_internal(&self, text: &str) -> EmbedderResult<Vec<f32>> {
        #[cfg(feature = "frankensearch-migration")]
        {
            self
                .delegate
                .embed_sync(text)
                .map_err(|e| EmbedderError::EmbeddingFailed(e.to_string()))
        }

        #[cfg(not(feature = "frankensearch-migration"))]
        {
            if text.is_empty() {
                return Err(EmbedderError::InvalidInput("empty text".to_string()));
            }

            // Tokenize
            let encoding = self
                .tokenizer
                .encode(text, false)
                .map_err(|e| EmbedderError::EmbeddingFailed(format!("tokenization failed: {e}")))?;

            let token_ids = encoding.get_ids();

            if token_ids.is_empty() {
                return Err(EmbedderError::InvalidInput(
                    "text tokenizes to empty sequence".to_string(),
                ));
            }

            // Mean pool over token embeddings
            let mut sum = vec![0.0f32; self.dimensions];
            let mut count = 0usize;

            for &token_id in token_ids {
                let idx = token_id as usize;
                if idx < self.vocab_size {
                    let row = &self.embeddings[idx];
                    for (s, &r) in sum.iter_mut().zip(row.iter()) {
                        *s += r;
                    }
                    count += 1;
                }
                // OOV tokens are silently skipped (common in Model2Vec)
            }

            if count == 0 {
                return Err(EmbedderError::EmbeddingFailed(
                    "all tokens were OOV".to_string(),
                ));
            }

            // Compute mean
            #[allow(clippy::cast_precision_loss)]
            let inv = 1.0 / count as f32;
            for s in &mut sum {
                *s *= inv;
            }

            // L2 normalize
            l2_normalize(&mut sum);

            Ok(sum)
        }
    }
}

impl std::fmt::Debug for Model2VecEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Model2VecEmbedder")
            .field("name", &self.name)
            .field("dimensions", &self.dimensions)
            .field("vocab_size", &self.vocab_size)
            .finish_non_exhaustive()
    }
}

impl Embedder for Model2VecEmbedder {
    fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>> {
        self.embed_internal(text)
    }

    fn embed_batch(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        // Model2Vec is fast enough that parallel batch processing helps
        use rayon::prelude::*;

        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Check for empty texts
        if texts.iter().any(|t| t.is_empty()) {
            return Err(EmbedderError::InvalidInput(
                "batch contains empty text".to_string(),
            ));
        }

        texts
            .par_iter()
            .map(|text| self.embed_internal(text))
            .collect()
    }

    fn dimension(&self) -> usize {
        self.dimensions
    }

    fn id(&self) -> &'static str {
        "model2vec"
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    fn is_semantic(&self) -> bool {
        true // Model2Vec uses distilled semantic knowledge
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::StaticEmbedder
    }

    fn supports_mrl(&self) -> bool {
        false // Model2Vec doesn't support MRL truncation
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(MODEL_POTION_32M, "potion-retrieval-32M");
        assert_eq!(MODEL_POTION_MULTI_128M, "potion-multilingual-128M");
    }

    #[test]
    fn test_required_files() {
        assert!(REQUIRED_FILES.contains(&"tokenizer.json"));
        assert!(REQUIRED_FILES.contains(&"model.safetensors"));
    }

    #[test]
    fn test_model_search_paths() {
        let paths = Model2VecEmbedder::model_search_paths(MODEL_POTION_32M);
        assert!(!paths.is_empty());

        // Should include xf cache path
        assert!(paths.iter().any(|p| p.to_string_lossy().contains("xf")));
    }

    #[test]
    fn test_tensor_to_embeddings_small() {
        // Create a small test tensor: 2 vocab × 3 dims
        // Using f32 little-endian bytes
        let data: Vec<u8> = vec![
            // Row 0: [1.0, 2.0, 3.0]
            0x00, 0x00, 0x80, 0x3F, // 1.0
            0x00, 0x00, 0x00, 0x40, // 2.0
            0x00, 0x00, 0x40, 0x40, // 3.0
            // Row 1: [4.0, 5.0, 6.0]
            0x00, 0x00, 0x80, 0x40, // 4.0
            0x00, 0x00, 0xA0, 0x40, // 5.0
            0x00, 0x00, 0xC0, 0x40, // 6.0
        ];

        let embeddings = Model2VecEmbedder::tensor_to_embeddings(&data, 2, 3).unwrap();

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 3);
        assert_eq!(embeddings[1].len(), 3);

        assert!((embeddings[0][0] - 1.0).abs() < 1e-6);
        assert!((embeddings[0][1] - 2.0).abs() < 1e-6);
        assert!((embeddings[0][2] - 3.0).abs() < 1e-6);

        assert!((embeddings[1][0] - 4.0).abs() < 1e-6);
        assert!((embeddings[1][1] - 5.0).abs() < 1e-6);
        assert!((embeddings[1][2] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_size_mismatch() {
        // Wrong size data
        let data = vec![0u8; 10]; // Not a valid tensor size
        let result = Model2VecEmbedder::tensor_to_embeddings(&data, 2, 3);
        assert!(result.is_err());
    }

    // Integration tests require actual model files
    #[test]
    #[ignore = "requires model files"]
    fn test_embed_produces_correct_dimension() {
        let embedder = Model2VecEmbedder::try_load(MODEL_POTION_32M).expect("model not available");

        let embedding = embedder.embed("hello world").unwrap();
        assert_eq!(embedding.len(), embedder.dimension());

        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "embedding not normalized: {norm}"
        );
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_deterministic_output() {
        let embedder = Model2VecEmbedder::try_load(MODEL_POTION_32M).expect("model not available");

        let text = "determinism test";
        let emb1 = embedder.embed(text).unwrap();
        let emb2 = embedder.embed(text).unwrap();

        assert_eq!(emb1, emb2, "same input should produce identical output");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_batch_embedding() {
        let embedder = Model2VecEmbedder::try_load(MODEL_POTION_32M).expect("model not available");

        let texts = vec!["hello", "world", "rust programming"];
        let embeddings = embedder.embed_batch(&texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), embedder.dimension());

            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_similarity_ordering() {
        use crate::embedder::dot_product;

        let embedder = Model2VecEmbedder::try_load(MODEL_POTION_32M).expect("model not available");

        let query = "programming languages";
        let doc1 = "rust is a systems programming language";
        let doc2 = "i love eating pizza";

        let q_emb = embedder.embed(query).unwrap();
        let d1_emb = embedder.embed(doc1).unwrap();
        let d2_emb = embedder.embed(doc2).unwrap();

        let sim_1 = dot_product(&q_emb, &d1_emb);
        let sim_2 = dot_product(&q_emb, &d2_emb);

        // doc1 should be more similar to query
        assert!(
            sim_1 > sim_2,
            "doc1 ({sim_1}) should be more similar than doc2 ({sim_2})"
        );
    }
}
