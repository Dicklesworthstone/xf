//! Mixedbread AI mxbai-rerank-xsmall-v1 cross-encoder reranker via ONNX.
//!
//! Implements a ~100M parameter cross-encoder model for CPU-based reranking.
//! Slightly slower than FlashRank nano but offers better quality.
//!
//! # Model Source
//!
//! - <https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1>
//! - License: Apache-2.0
//!
//! # Usage
//!
//! ```rust,ignore
//! let reranker = MxbaiReranker::load()?;
//! let scores = reranker.rerank("what is rust?", &["Rust is a language", "Python is cool"])?;
//! ```

use std::path::{Path, PathBuf};
use std::sync::Mutex;

use ndarray::Array2;
use ort::session::{Session, builder::GraphOptimizationLevel};
use tokenizers::Tokenizer;

use crate::reranker::{Reranker, RerankerError, RerankerResult};

/// Model name constant.
pub const MODEL_NAME: &str = "mxbai-rerank-xsmall";

/// Default max sequence length for cross-encoder.
const MAX_SEQ_LEN: usize = 512;

/// Batch size for ONNX inference.
const BATCH_SIZE: usize = 16;

/// Required model files.
const REQUIRED_FILES: &[&str] = &["model.onnx", "tokenizer.json"];

/// Mixedbread AI mxbai-rerank-xsmall-v1 cross-encoder reranker.
///
/// Uses a distilled cross-encoder (~100M params) for quality/speed balance.
pub struct MxbaiReranker {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    max_length: usize,
    name: String,
    _model_dir: PathBuf,
}

impl MxbaiReranker {
    /// Load the model from a directory containing model files.
    ///
    /// # Errors
    ///
    /// Returns an error if the model files are missing or invalid.
    pub fn load_from_dir(model_dir: &Path) -> RerankerResult<Self> {
        // Validate required files
        for file in REQUIRED_FILES {
            let file_path = model_dir.join(file);
            if !file_path.exists() {
                return Err(RerankerError::Unavailable(format!(
                    "missing required model file: {}",
                    file_path.display()
                )));
            }
        }

        // Load tokenizer
        let tokenizer_path = model_dir.join("tokenizer.json");
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| RerankerError::Internal(format!("failed to load tokenizer: {e}")))?;

        // Load ONNX model
        let model_path = model_dir.join("model.onnx");
        let session = Session::builder()
            .map_err(|e| RerankerError::Internal(format!("failed to create session builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::Level3)
            .map_err(|e| RerankerError::Internal(format!("failed to set optimization level: {e}")))?
            .with_intra_threads(rayon::current_num_threads())
            .map_err(|e| RerankerError::Internal(format!("failed to set thread count: {e}")))?
            .commit_from_file(&model_path)
            .map_err(|e| RerankerError::Internal(format!("failed to load ONNX model: {e}")))?;

        Ok(Self {
            session: Mutex::new(session),
            tokenizer,
            max_length: MAX_SEQ_LEN,
            name: MODEL_NAME.to_string(),
            _model_dir: model_dir.to_path_buf(),
        })
    }

    /// Try to load from standard model locations.
    ///
    /// Searches in order:
    /// 1. `~/.cache/xf/models/mxbai-rerank-xsmall`
    /// 2. `~/.local/share/xf/models/mxbai-rerank-xsmall`
    /// 3. `~/.cache/huggingface/hub/models--mixedbread-ai--mxbai-rerank-xsmall-v1`
    ///
    /// # Errors
    ///
    /// Returns an error if the model is not found in any location.
    pub fn load() -> RerankerResult<Self> {
        let home = dirs::home_dir()
            .ok_or_else(|| RerankerError::Unavailable("cannot determine home directory".into()))?;

        let search_paths = [
            home.join(".cache/xf/models/mxbai-rerank-xsmall"),
            home.join(".local/share/xf/models/mxbai-rerank-xsmall"),
            home.join(
                ".cache/huggingface/hub/models--mixedbread-ai--mxbai-rerank-xsmall-v1/snapshots",
            ),
        ];

        for path in &search_paths {
            if path.exists() {
                // For HuggingFace hub, look for snapshot directory
                let model_dir = if path.ends_with("snapshots") {
                    find_latest_snapshot(path)?
                } else {
                    path.clone()
                };

                if has_required_files(&model_dir) {
                    return Self::load_from_dir(&model_dir);
                }
            }
        }

        Err(RerankerError::Unavailable(format!(
            "mxbai-rerank-xsmall model not found in standard locations: {:?}",
            search_paths
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
        )))
    }

    /// Batch score multiple query-document pairs.
    #[allow(clippy::significant_drop_tightening)] // Session lock needed throughout inference
    #[allow(clippy::too_many_lines)] // Inference logic is self-contained
    fn score_batch(&self, query: &str, documents: &[&str]) -> RerankerResult<Vec<f32>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        // Tokenize all pairs
        let mut all_input_ids = Vec::with_capacity(documents.len());
        let mut all_attention_masks = Vec::with_capacity(documents.len());
        let mut all_type_ids = Vec::with_capacity(documents.len());

        for doc in documents {
            let encoding = self
                .tokenizer
                .encode((query, *doc), true)
                .map_err(|e| RerankerError::Internal(format!("tokenization failed: {e}")))?;

            // Truncate if needed
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();

            let len = ids.len().min(self.max_length);
            all_input_ids.push(ids[..len].to_vec());
            all_attention_masks.push(mask[..len].to_vec());
            all_type_ids.push(type_ids[..len].to_vec());
        }

        // Pad to uniform length
        let max_len = all_input_ids.iter().map(Vec::len).max().unwrap_or(0);

        let mut input_ids_flat = Vec::with_capacity(documents.len() * max_len);
        let mut attention_mask_flat = Vec::with_capacity(documents.len() * max_len);
        let mut type_ids_flat = Vec::with_capacity(documents.len() * max_len);

        for i in 0..documents.len() {
            let ids = &all_input_ids[i];
            let mask = &all_attention_masks[i];
            let types = &all_type_ids[i];

            // Add tokens
            for &id in ids {
                input_ids_flat.push(i64::from(id));
            }
            for &m in mask {
                attention_mask_flat.push(i64::from(m));
            }
            for &t in types {
                type_ids_flat.push(i64::from(t));
            }

            // Pad to max_len
            let padding = max_len - ids.len();
            input_ids_flat.extend(std::iter::repeat_n(0i64, padding));
            attention_mask_flat.extend(std::iter::repeat_n(0i64, padding));
            type_ids_flat.extend(std::iter::repeat_n(0i64, padding));
        }

        // Create tensors
        let batch_size = documents.len();
        let input_ids =
            Array2::from_shape_vec((batch_size, max_len), input_ids_flat).map_err(|e| {
                RerankerError::Internal(format!("failed to create input_ids tensor: {e}"))
            })?;
        let attention_mask = Array2::from_shape_vec((batch_size, max_len), attention_mask_flat)
            .map_err(|e| {
                RerankerError::Internal(format!("failed to create attention_mask tensor: {e}"))
            })?;
        let token_type_ids =
            Array2::from_shape_vec((batch_size, max_len), type_ids_flat).map_err(|e| {
                RerankerError::Internal(format!("failed to create token_type_ids tensor: {e}"))
            })?;

        // Run inference
        let session = self
            .session
            .lock()
            .map_err(|e| RerankerError::Internal(format!("session lock failed: {e}")))?;

        let inputs = ort::inputs![
            "input_ids" => input_ids.view(),
            "attention_mask" => attention_mask.view(),
            "token_type_ids" => token_type_ids.view(),
        ]
        .map_err(|e| RerankerError::Internal(format!("failed to create inputs: {e}")))?;

        let outputs = session
            .run(inputs)
            .map_err(|e| RerankerError::RerankFailed(format!("ONNX inference failed: {e}")))?;

        // Extract scores from output
        // mxbai outputs logits of shape (batch_size, 1) or (batch_size,)
        // Try named outputs first, fall back to index-based access
        let scores: Vec<f32> = if let Some(output) = outputs.get("logits") {
            let tensor = output
                .try_extract_tensor::<f32>()
                .map_err(|e| RerankerError::Internal(format!("failed to extract logits: {e}")))?;
            tensor
                .as_slice()
                .ok_or_else(|| RerankerError::Internal("non-contiguous tensor".into()))?
                .iter()
                .take(batch_size)
                .copied()
                .collect()
        } else if let Some(output) = outputs.get("output") {
            let tensor = output
                .try_extract_tensor::<f32>()
                .map_err(|e| RerankerError::Internal(format!("failed to extract output: {e}")))?;
            tensor
                .as_slice()
                .ok_or_else(|| RerankerError::Internal("non-contiguous tensor".into()))?
                .iter()
                .take(batch_size)
                .copied()
                .collect()
        } else if let Some(output) = outputs.get("sentence_embedding") {
            // Some models use different output names
            let tensor = output.try_extract_tensor::<f32>().map_err(|e| {
                RerankerError::Internal(format!("failed to extract sentence_embedding: {e}"))
            })?;
            tensor
                .as_slice()
                .ok_or_else(|| RerankerError::Internal("non-contiguous tensor".into()))?
                .iter()
                .take(batch_size)
                .copied()
                .collect()
        } else {
            // Try first output by index
            if outputs.len() > 0 {
                let output = &outputs[0];
                let tensor = output.try_extract_tensor::<f32>().map_err(|e| {
                    RerankerError::Internal(format!("failed to extract first output: {e}"))
                })?;
                tensor
                    .as_slice()
                    .ok_or_else(|| RerankerError::Internal("non-contiguous tensor".into()))?
                    .iter()
                    .take(batch_size)
                    .copied()
                    .collect()
            } else {
                return Err(RerankerError::Internal("no output tensor found".into()));
            }
        };

        // Apply sigmoid for probability-like scores
        let scores: Vec<f32> = scores
            .into_iter()
            .map(|s| 1.0 / (1.0 + (-s).exp()))
            .collect();

        Ok(scores)
    }
}

impl Reranker for MxbaiReranker {
    fn rerank(&self, query: &str, documents: &[&str]) -> RerankerResult<Vec<f32>> {
        if documents.is_empty() {
            return Ok(vec![]);
        }

        // Process in batches
        let mut all_scores = Vec::with_capacity(documents.len());

        for chunk in documents.chunks(BATCH_SIZE) {
            let batch_scores = self.score_batch(query, chunk)?;
            all_scores.extend(batch_scores);
        }

        Ok(all_scores)
    }

    fn model_name(&self) -> &str {
        &self.name
    }

    fn max_length(&self) -> usize {
        self.max_length
    }
}

/// Find the latest snapshot directory in a HuggingFace hub cache.
fn find_latest_snapshot(snapshots_dir: &Path) -> RerankerResult<PathBuf> {
    let mut entries: Vec<_> = std::fs::read_dir(snapshots_dir)
        .map_err(|e| RerankerError::Unavailable(format!("cannot read snapshots dir: {e}")))?
        .filter_map(Result::ok)
        .filter(|e| e.path().is_dir())
        .collect();

    entries.sort_by(|a, b| {
        b.metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .cmp(&a.metadata().ok().and_then(|m| m.modified().ok()))
    });

    entries
        .first()
        .map(std::fs::DirEntry::path)
        .ok_or_else(|| RerankerError::Unavailable("no snapshot found".into()))
}

/// Check if a directory has all required model files.
fn has_required_files(dir: &Path) -> bool {
    REQUIRED_FILES.iter().all(|f| dir.join(f).exists())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_name() {
        assert_eq!(MODEL_NAME, "mxbai-rerank-xsmall");
    }

    #[test]
    fn test_has_required_files_missing() {
        let temp = std::env::temp_dir().join("mxbai_test_empty");
        let _ = std::fs::create_dir_all(&temp);
        assert!(!has_required_files(&temp));
        let _ = std::fs::remove_dir_all(&temp);
    }

    #[test]
    fn test_batch_size() {
        assert_eq!(BATCH_SIZE, 16);
    }

    #[test]
    fn test_max_seq_len() {
        assert_eq!(MAX_SEQ_LEN, 512);
    }
}
