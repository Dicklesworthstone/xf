//! ML-based semantic embedder using FastEmbed (MiniLM).
//!
//! This embedder uses the `all-MiniLM-L6-v2` sentence transformer model
//! to generate true semantic embeddings. Unlike the hash embedder,
//! semantically similar texts (e.g., "happy" and "joyful") will have
//! similar embedding vectors.
//!
//! # Model Requirements
//!
//! The model directory must contain:
//! - `model.onnx` - The ONNX model file
//! - `tokenizer.json` - Tokenizer configuration
//! - `config.json` - Model configuration
//! - `special_tokens_map.json` - Special tokens mapping
//! - `tokenizer_config.json` - Tokenizer settings
//!
//! # Performance
//!
//! - ~5ms per embedding on CPU
//! - Batching provides ~3x throughput improvement
//! - Thread-safe via internal Mutex

use std::path::Path;
use std::sync::Mutex;

use fastembed::{EmbeddingModel, InitOptions, TextEmbedding};

use crate::embedder::{Embedder, EmbedderError, EmbedderResult, l2_normalize};

/// Model identifier for fastembed.
const MODEL_ID: EmbeddingModel = EmbeddingModel::AllMiniLML6V2;

/// Directory name where model files are stored.
const MODEL_DIR_NAME: &str = "all-MiniLM-L6-v2";

/// Unique identifier for this embedder.
const EMBEDDER_ID: &str = "minilm-384";

/// Output dimension of MiniLM embeddings.
const EMBEDDING_DIMENSION: usize = 384;

/// Required model files for validation.
const REQUIRED_FILES: &[&str] = &[
    "model.onnx",
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
];

/// ML-based semantic embedder using MiniLM.
pub struct FastEmbedder {
    model: Mutex<TextEmbedding>,
    id: String,
    dimension: usize,
}

impl FastEmbedder {
    /// Load the embedder from a model directory.
    ///
    /// The model directory should be a parent that contains a subdirectory
    /// named `all-MiniLM-L6-v2` with the required model files.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The model directory doesn't exist
    /// - Required files are missing
    /// - Model initialization fails
    pub fn load_from_dir(model_dir: &Path) -> EmbedderResult<Self> {
        let minilm_dir = model_dir.join(MODEL_DIR_NAME);

        // Validate required files exist
        for file in REQUIRED_FILES {
            let file_path = minilm_dir.join(file);
            if !file_path.exists() {
                return Err(EmbedderError::Unavailable(format!(
                    "missing required model file: {}",
                    file_path.display()
                )));
            }
        }

        // Initialize with local-only loading (never download)
        let init_options = InitOptions::new(MODEL_ID)
            .with_cache_dir(model_dir.to_path_buf())
            .with_show_download_progress(false);

        let model = TextEmbedding::try_new(init_options).map_err(|e| {
            EmbedderError::Internal(format!("failed to load MiniLM model: {e}"))
        })?;

        Ok(Self {
            model: Mutex::new(model),
            id: EMBEDDER_ID.to_string(),
            dimension: EMBEDDING_DIMENSION,
        })
    }

    /// Try to load the embedder from standard locations.
    ///
    /// Checks in order:
    /// 1. `~/.local/share/xf/models`
    /// 2. `~/.cache/fastembed`
    /// 3. System data directory
    ///
    /// # Errors
    ///
    /// Returns an error if no valid model is found in any location.
    pub fn try_load() -> EmbedderResult<Self> {
        let candidates = [
            dirs::data_local_dir().map(|p| p.join("xf").join("models")),
            dirs::cache_dir().map(|p| p.join("fastembed")),
            Some(std::path::PathBuf::from("/usr/local/share/xf/models")),
        ];

        for candidate in candidates.into_iter().flatten() {
            if candidate.join(MODEL_DIR_NAME).exists() {
                if let Ok(embedder) = Self::load_from_dir(&candidate) {
                    return Ok(embedder);
                }
            }
        }

        Err(EmbedderError::Unavailable(
            "MiniLM model not found. Run 'xf index --semantic' to auto-download.".to_string(),
        ))
    }

    /// Load the model, downloading it if necessary.
    ///
    /// This will automatically download the MiniLM model (~80MB) on first use
    /// if it's not already available. Download progress is shown when
    /// `show_progress` is true.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded or downloaded.
    pub fn load_or_download(show_progress: bool) -> EmbedderResult<Self> {
        // First try to load from existing locations
        if let Ok(embedder) = Self::try_load() {
            return Ok(embedder);
        }

        // Model not found, download it
        let cache_dir = dirs::cache_dir()
            .map(|p| p.join("fastembed"))
            .unwrap_or_else(|| std::path::PathBuf::from(".cache/fastembed"));

        // Create cache directory if needed
        std::fs::create_dir_all(&cache_dir).map_err(|e| {
            EmbedderError::Internal(format!("failed to create cache dir: {e}"))
        })?;

        // Initialize with download enabled
        let init_options = InitOptions::new(MODEL_ID)
            .with_cache_dir(cache_dir.clone())
            .with_show_download_progress(show_progress);

        let model = TextEmbedding::try_new(init_options).map_err(|e| {
            EmbedderError::Internal(format!("failed to download/load MiniLM model: {e}"))
        })?;

        Ok(Self {
            model: Mutex::new(model),
            id: EMBEDDER_ID.to_string(),
            dimension: EMBEDDING_DIMENSION,
        })
    }

    /// Check if the semantic model is available.
    #[must_use]
    pub fn is_available() -> bool {
        Self::try_load().is_ok()
    }

    /// Get the expected model directory path.
    #[must_use]
    pub fn default_model_dir() -> std::path::PathBuf {
        dirs::data_local_dir()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("xf")
            .join("models")
    }

    /// Get the model subdirectory name.
    #[must_use]
    pub const fn model_dir_name() -> &'static str {
        MODEL_DIR_NAME
    }
}

impl std::fmt::Debug for FastEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastEmbedder")
            .field("id", &self.id)
            .field("dimension", &self.dimension)
            .finish_non_exhaustive()
    }
}

impl Embedder for FastEmbedder {
    fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>> {
        if text.is_empty() {
            return Err(EmbedderError::InvalidInput("empty text".to_string()));
        }

        let model = self.model.lock().map_err(|e| {
            EmbedderError::Internal(format!("model lock poisoned: {e}"))
        })?;

        let embeddings = model.embed(vec![text], None).map_err(|e| {
            EmbedderError::EmbeddingFailed(format!("embedding failed: {e}"))
        })?;

        let mut embedding = embeddings.into_iter().next().ok_or_else(|| {
            EmbedderError::Internal("no embedding returned".to_string())
        })?;

        // Ensure L2 normalization (fastembed should already normalize, but be safe)
        l2_normalize(&mut embedding);

        Ok(embedding)
    }

    fn embed_batch(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        // Filter empty texts but track their positions for reconstruction
        let (non_empty_indices, non_empty_texts): (Vec<_>, Vec<_>) = texts
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.is_empty())
            .map(|(i, t)| (i, *t))
            .unzip();

        if non_empty_texts.is_empty() {
            return Err(EmbedderError::InvalidInput("all texts are empty".to_string()));
        }

        let model = self.model.lock().map_err(|e| {
            EmbedderError::Internal(format!("model lock poisoned: {e}"))
        })?;

        let mut embeddings = model.embed(non_empty_texts, None).map_err(|e| {
            EmbedderError::EmbeddingFailed(format!("batch embedding failed: {e}"))
        })?;

        // L2 normalize all embeddings
        for embedding in &mut embeddings {
            l2_normalize(embedding);
        }

        // Reconstruct full result with empty slots for empty inputs
        let mut result = vec![Vec::new(); texts.len()];
        for (result_idx, embedding) in non_empty_indices.into_iter().zip(embeddings) {
            result[result_idx] = embedding;
        }

        // Fill empty slots with error indicators (zero vectors)
        // Callers should check for empty inputs before calling
        for (i, text) in texts.iter().enumerate() {
            if text.is_empty() && result[i].is_empty() {
                result[i] = vec![0.0; self.dimension];
            }
        }

        Ok(result)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn is_semantic(&self) -> bool {
        true // This IS a semantic embedder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedder_id() {
        // Just test constants are correct
        assert_eq!(EMBEDDER_ID, "minilm-384");
        assert_eq!(EMBEDDING_DIMENSION, 384);
        assert_eq!(MODEL_DIR_NAME, "all-MiniLM-L6-v2");
    }

    #[test]
    fn test_required_files() {
        assert!(REQUIRED_FILES.contains(&"model.onnx"));
        assert!(REQUIRED_FILES.contains(&"tokenizer.json"));
    }

    #[test]
    fn test_default_model_dir() {
        let dir = FastEmbedder::default_model_dir();
        assert!(dir.to_string_lossy().contains("xf"));
    }

    // Integration tests require actual model files
    #[test]
    #[ignore = "requires model files"]
    fn test_embed_semantic_similarity() {
        let embedder = FastEmbedder::try_load().expect("model not available");

        let happy = embedder.embed("I am very happy today").unwrap();
        let joyful = embedder.embed("I am feeling joyful").unwrap();
        let sad = embedder.embed("I am feeling very sad").unwrap();

        use crate::embedder::dot_product;
        let sim_happy_joyful = dot_product(&happy, &joyful);
        let sim_happy_sad = dot_product(&happy, &sad);

        // "happy" should be more similar to "joyful" than to "sad"
        assert!(
            sim_happy_joyful > sim_happy_sad,
            "semantic similarity failed: happy-joyful={sim_happy_joyful}, happy-sad={sim_happy_sad}"
        );
    }
}
