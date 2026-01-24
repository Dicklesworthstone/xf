//! Static embedding model with Matryoshka (MRL) truncation support.
//!
//! Implements the static-retrieval-mrl-en-v1 embedder using ONNX Runtime.
//! This is a "static" model that uses distilled/frozen transformer weights
//! for very fast inference while supporting MRL dimension truncation.
//!
//! # Key Features
//!
//! - Very fast inference (~50-100x faster than full transformers)
//! - Matryoshka Representation Learning (MRL) support
//! - L2 normalization with proper re-normalization after truncation
//!
//! # Model Files
//!
//! - `model.onnx` - ONNX model weights
//! - `tokenizer.json` - HuggingFace tokenizer
//! - `config.json` - Model configuration

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ort::session::{Session, builder::GraphOptimizationLevel};
use tokenizers::Tokenizer;

use crate::embedder::{Embedder, EmbedderError, EmbedderResult, ModelCategory, l2_normalize};

/// Model name constant.
pub const MODEL_NAME: &str = "static-retrieval-mrl-en-v1";

/// Default native embedding dimension.
const NATIVE_DIMS: usize = 1024;

/// Default max sequence length.
const MAX_SEQ_LEN: usize = 512;

/// Required model files.
const REQUIRED_FILES: &[&str] = &["model.onnx", "tokenizer.json"];

/// Static embedding model with MRL truncation support.
pub struct StaticMrlEmbedder {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    native_dims: usize,
    target_dims: Option<usize>,
    name: String,
    max_seq_len: usize,
}

impl StaticMrlEmbedder {
    /// Load the model from a directory containing model files.
    ///
    /// # Errors
    ///
    /// Returns an error if the model files are missing or invalid.
    pub fn load_from_dir(model_dir: &Path) -> EmbedderResult<Self> {
        Self::load_from_dir_with_dims(model_dir, None)
    }

    /// Load the model from a directory with optional MRL dimension truncation.
    ///
    /// # Errors
    ///
    /// Returns an error if the model files are missing or invalid.
    pub fn load_from_dir_with_dims(
        model_dir: &Path,
        target_dims: Option<usize>,
    ) -> EmbedderResult<Self> {
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

        // Load ONNX model
        let model_path = model_dir.join("model.onnx");
        let session = Session::builder()
            .map_err(|e| EmbedderError::Internal(format!("failed to create session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| EmbedderError::Internal(format!("failed to set optimization level: {e}")))?
            .with_intra_threads(rayon::current_num_threads())
            .map_err(|e| EmbedderError::Internal(format!("failed to set thread count: {e}")))?
            .commit_from_file(&model_path)
            .map_err(|e| EmbedderError::Internal(format!("failed to load ONNX model: {e}")))?;

        // Validate target_dims if specified
        if let Some(dims) = target_dims {
            if dims == 0 || dims > NATIVE_DIMS {
                return Err(EmbedderError::InvalidInput(format!(
                    "invalid target_dims {dims}: must be 1..={NATIVE_DIMS}"
                )));
            }
        }

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            native_dims: NATIVE_DIMS,
            target_dims,
            name: MODEL_NAME.to_string(),
            max_seq_len: MAX_SEQ_LEN,
        })
    }

    /// Try to load from standard model locations.
    ///
    /// Searches in order:
    /// 1. `~/.cache/xf/models/static-retrieval-mrl-en-v1`
    /// 2. `~/.local/share/xf/models/static-retrieval-mrl-en-v1`
    /// 3. `~/.cache/huggingface/hub/models--sentence-transformers--static-retrieval-mrl-en-v1`
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be found.
    pub fn try_load() -> EmbedderResult<Self> {
        Self::try_load_with_dims(None)
    }

    /// Try to load from standard locations with MRL truncation.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be found.
    pub fn try_load_with_dims(target_dims: Option<usize>) -> EmbedderResult<Self> {
        let candidates = Self::model_search_paths();

        for candidate in &candidates {
            if candidate.exists() {
                if let Ok(embedder) = Self::load_from_dir_with_dims(candidate, target_dims) {
                    return Ok(embedder);
                }
            }
        }

        Err(EmbedderError::Unavailable(format!(
            "{MODEL_NAME} model not found. Searched: {}",
            candidates
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )))
    }

    /// Get standard model search paths.
    #[must_use]
    pub fn model_search_paths() -> Vec<PathBuf> {
        let mut paths = Vec::new();

        // xf cache directory
        if let Some(cache) = dirs::cache_dir() {
            paths.push(cache.join("xf").join("models").join(MODEL_NAME));
        }

        // xf data directory
        if let Some(data) = dirs::data_local_dir() {
            paths.push(data.join("xf").join("models").join(MODEL_NAME));
        }

        // HuggingFace hub cache
        if let Some(cache) = dirs::cache_dir() {
            paths.push(
                cache
                    .join("huggingface")
                    .join("hub")
                    .join(format!("models--sentence-transformers--{MODEL_NAME}"))
                    .join("snapshots"),
            );
        }

        paths
    }

    /// Check if the model is available.
    #[must_use]
    pub fn is_available() -> bool {
        Self::try_load().is_ok()
    }

    /// Get the native embedding dimension (before truncation).
    #[must_use]
    pub const fn native_dimension(&self) -> usize {
        self.native_dims
    }

    /// Get the target MRL dimension (if truncation is enabled).
    #[must_use]
    pub const fn target_dimension(&self) -> Option<usize> {
        self.target_dims
    }

    /// Embed texts and return the raw embeddings before truncation.
    #[allow(clippy::significant_drop_tightening, clippy::cast_precision_loss)]
    fn embed_raw(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Tokenize all texts
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| EmbedderError::EmbeddingFailed(format!("tokenization failed: {e}")))?;

        // Find max length and prepare batched inputs
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len().min(self.max_seq_len))
            .max()
            .unwrap_or(0);

        if max_len == 0 {
            return Err(EmbedderError::InvalidInput(
                "all texts tokenize to empty".to_string(),
            ));
        }

        let batch_size = texts.len();

        // Build padded input_ids and attention_mask tensors
        let mut input_ids = vec![0i64; batch_size * max_len];
        let mut attention_mask = vec![0i64; batch_size * max_len];

        for (i, encoding) in encodings.iter().enumerate() {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let len = ids.len().min(max_len);

            for j in 0..len {
                input_ids[i * max_len + j] = i64::from(ids[j]);
                attention_mask[i * max_len + j] = i64::from(mask[j]);
            }
        }

        // Run ONNX inference
        let session = self
            .session
            .lock()
            .map_err(|e| EmbedderError::Internal(format!("session lock poisoned: {e}")))?;

        // Create input tensors with proper shape
        let input_ids_array = ndarray::Array2::from_shape_vec((batch_size, max_len), input_ids)
            .map_err(|e| {
                EmbedderError::Internal(format!("failed to create input_ids array: {e}"))
            })?;

        let attention_mask_array =
            ndarray::Array2::from_shape_vec((batch_size, max_len), attention_mask).map_err(
                |e| EmbedderError::Internal(format!("failed to create attention_mask array: {e}")),
            )?;

        let outputs = session
            .run(
                ort::inputs![
                    "input_ids" => input_ids_array.view(),
                    "attention_mask" => attention_mask_array.view(),
                ]
                .map_err(|e| EmbedderError::Internal(format!("failed to create inputs: {e}")))?,
            )
            .map_err(|e| EmbedderError::EmbeddingFailed(format!("ONNX inference failed: {e}")))?;

        // Extract embeddings from output
        // Static models typically output [batch_size, seq_len, hidden_dim] or [batch_size, hidden_dim]
        let output_tensor = &outputs[0];

        let embeddings = output_tensor.try_extract_tensor::<f32>().map_err(|e| {
            EmbedderError::Internal(format!("failed to extract output tensor: {e}"))
        })?;

        let shape = embeddings.shape();

        // Handle different output shapes
        let results: Vec<Vec<f32>> = if shape.len() == 3 {
            // [batch_size, seq_len, hidden_dim] - need mean pooling
            let hidden_dim = shape[2];
            let seq_len = shape[1];

            (0..batch_size)
                .map(|b| {
                    // Mean pool over sequence dimension with attention mask
                    let mut pooled = vec![0.0f32; hidden_dim];
                    let mut token_count = 0;

                    for s in 0..seq_len {
                        if attention_mask_array[[b, s]] == 1 {
                            for h in 0..hidden_dim {
                                pooled[h] += embeddings[[b, s, h]];
                            }
                            token_count += 1;
                        }
                    }

                    if token_count > 0 {
                        for val in &mut pooled {
                            *val /= token_count as f32;
                        }
                    }

                    pooled
                })
                .collect()
        } else if shape.len() == 2 {
            // [batch_size, hidden_dim] - already pooled
            (0..batch_size)
                .map(|b| (0..shape[1]).map(|h| embeddings[[b, h]]).collect())
                .collect()
        } else {
            return Err(EmbedderError::Internal(format!(
                "unexpected output shape: {shape:?}"
            )));
        };

        Ok(results)
    }
}

impl std::fmt::Debug for StaticMrlEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StaticMrlEmbedder")
            .field("name", &self.name)
            .field("native_dims", &self.native_dims)
            .field("target_dims", &self.target_dims)
            .field("max_seq_len", &self.max_seq_len)
            .finish_non_exhaustive()
    }
}

impl Embedder for StaticMrlEmbedder {
    fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>> {
        if text.is_empty() {
            return Err(EmbedderError::InvalidInput("empty text".to_string()));
        }

        let mut embeddings = self.embed_raw(&[text])?;
        let mut embedding = embeddings
            .pop()
            .ok_or_else(|| EmbedderError::Internal("no embedding returned".to_string()))?;

        // Apply MRL truncation if configured
        if let Some(target) = self.target_dims {
            embedding.truncate(target);
        }

        // L2 normalize (critical: must re-normalize after truncation)
        l2_normalize(&mut embedding);

        Ok(embedding)
    }

    fn embed_batch(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Check for empty texts
        if texts.iter().any(|t| t.is_empty()) {
            return Err(EmbedderError::InvalidInput(
                "batch contains empty text".to_string(),
            ));
        }

        let mut embeddings = self.embed_raw(texts)?;

        // Apply MRL truncation and normalization
        for embedding in &mut embeddings {
            if let Some(target) = self.target_dims {
                embedding.truncate(target);
            }
            l2_normalize(embedding);
        }

        Ok(embeddings)
    }

    fn dimension(&self) -> usize {
        self.target_dims.unwrap_or(self.native_dims)
    }

    fn id(&self) -> &'static str {
        "static-mrl"
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn category(&self) -> ModelCategory {
        ModelCategory::StaticEmbedder
    }

    fn supports_mrl(&self) -> bool {
        true
    }

    fn truncate_embedding(&self, embedding: &[f32], target_dim: usize) -> EmbedderResult<Vec<f32>> {
        if target_dim == 0 || target_dim > self.native_dims {
            return Err(EmbedderError::InvalidInput(format!(
                "invalid target_dim {target_dim}: must be 1..={}",
                self.native_dims
            )));
        }

        let mut truncated = embedding[..target_dim].to_vec();
        l2_normalize(&mut truncated);
        Ok(truncated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants() {
        assert_eq!(MODEL_NAME, "static-retrieval-mrl-en-v1");
        assert_eq!(NATIVE_DIMS, 1024);
        assert_eq!(MAX_SEQ_LEN, 512);
    }

    #[test]
    fn test_required_files() {
        assert!(REQUIRED_FILES.contains(&"model.onnx"));
        assert!(REQUIRED_FILES.contains(&"tokenizer.json"));
    }

    #[test]
    fn test_model_search_paths() {
        let paths = StaticMrlEmbedder::model_search_paths();
        assert!(!paths.is_empty());
    }

    #[test]
    fn test_invalid_target_dims() {
        // Create a temp dir with mock files
        let temp_dir = tempfile::tempdir().unwrap();
        std::fs::write(temp_dir.path().join("model.onnx"), "mock").unwrap();
        std::fs::write(temp_dir.path().join("tokenizer.json"), "{}").unwrap();

        // target_dims = 0 should fail
        let result = StaticMrlEmbedder::load_from_dir_with_dims(temp_dir.path(), Some(0));
        assert!(result.is_err());

        // target_dims > NATIVE_DIMS should fail
        let result =
            StaticMrlEmbedder::load_from_dir_with_dims(temp_dir.path(), Some(NATIVE_DIMS + 1));
        assert!(result.is_err());
    }

    // Integration tests require actual model files
    #[test]
    #[ignore = "requires model files"]
    fn test_embed_produces_correct_dimension() {
        let embedder = StaticMrlEmbedder::try_load().expect("model not available");

        let embedding = embedder.embed("hello world").unwrap();
        assert_eq!(embedding.len(), NATIVE_DIMS);

        // Check normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "embedding not normalized: {norm}"
        );
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_mrl_truncation() {
        let embedder =
            StaticMrlEmbedder::try_load_with_dims(Some(256)).expect("model not available");

        let embedding = embedder.embed("hello world").unwrap();
        assert_eq!(embedding.len(), 256);

        // Check normalization after truncation
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "truncated embedding not normalized: {norm}"
        );
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_batch_embedding() {
        let embedder = StaticMrlEmbedder::try_load().expect("model not available");

        let texts = vec!["hello", "world", "rust programming"];
        let embeddings = embedder.embed_batch(&texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        for emb in &embeddings {
            assert_eq!(emb.len(), NATIVE_DIMS);

            let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_mrl_preserves_similarity_ordering() {
        use crate::embedder::dot_product;

        let embedder = StaticMrlEmbedder::try_load().expect("model not available");

        let query = "programming languages";
        let doc1 = "rust is a systems programming language";
        let doc2 = "i love eating pizza";

        // Full dimension embeddings
        let q_full = embedder.embed(query).unwrap();
        let d1_full = embedder.embed(doc1).unwrap();
        let d2_full = embedder.embed(doc2).unwrap();

        let sim_full_1 = dot_product(&q_full, &d1_full);
        let sim_full_2 = dot_product(&q_full, &d2_full);

        // doc1 should be more similar to query
        assert!(
            sim_full_1 > sim_full_2,
            "full: doc1 ({sim_full_1}) should be more similar than doc2 ({sim_full_2})"
        );

        // Truncated embeddings (256 dims)
        let q_trunc = embedder.truncate_embedding(&q_full, 256).unwrap();
        let d1_trunc = embedder.truncate_embedding(&d1_full, 256).unwrap();
        let d2_trunc = embedder.truncate_embedding(&d2_full, 256).unwrap();

        let sim_trunc_1 = dot_product(&q_trunc, &d1_trunc);
        let sim_trunc_2 = dot_product(&q_trunc, &d2_trunc);

        // MRL should preserve similarity ordering
        assert!(
            sim_trunc_1 > sim_trunc_2,
            "truncated: doc1 ({sim_trunc_1}) should be more similar than doc2 ({sim_trunc_2})"
        );
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_deterministic_output() {
        let embedder = StaticMrlEmbedder::try_load().expect("model not available");

        let text = "determinism test";
        let emb1 = embedder.embed(text).unwrap();
        let emb2 = embedder.embed(text).unwrap();

        assert_eq!(emb1, emb2, "same input should produce identical output");
    }
}
