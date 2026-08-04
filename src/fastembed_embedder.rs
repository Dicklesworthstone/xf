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
use asupersync::Cx;
use asupersync::runtime::{Runtime, RuntimeBuilder};
use frankensearch_core::Embedder as FrankensearchEmbedder;
use frankensearch_embed::FastEmbedEmbedder as FsFastEmbedder;

/// Directory name where model files are stored.
const MODEL_DIR_NAME: &str = "all-MiniLM-L6-v2";

/// Unique identifier for this embedder.
const EMBEDDER_ID: &str = "minilm-384";

/// Output dimension of MiniLM embeddings.
#[allow(dead_code)]
const EMBEDDING_DIMENSION: usize = 384;

/// Required model files for validation.
#[allow(dead_code)]
const REQUIRED_FILES: &[&str] = &[
    "model.onnx",
    "tokenizer.json",
    "config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
];

/// ML-based semantic embedder using MiniLM.
pub struct FastEmbedder {
    backend: FastEmbedBackend,
    id: String,
    dimension: usize,
}

enum FastEmbedBackend {
    Frankensearch {
        runtime: Runtime,
        delegate: FsFastEmbedder,
    },
}

fn map_fs_error(context: &str, err: impl std::fmt::Display) -> EmbedderError {
    EmbedderError::EmbeddingFailed(format!("frankensearch fastembed {context} failed: {err}"))
}

/// Generic FastEmbed-based embedder for multiple models.
pub struct FastEmbedModelEmbedder {
    model: Mutex<TextEmbedding>,
    id: String,
    model_name: String,
    dimension: usize,
    supports_mrl: bool,
}

impl FastEmbedModelEmbedder {
    /// Load or download the specified FastEmbed model.
    ///
    /// # Errors
    ///
    /// Returns an error if the model cannot be loaded or downloaded.
    pub fn load_or_download(
        model: EmbeddingModel,
        model_name: &str,
        cache_dir: Option<std::path::PathBuf>,
        show_progress: bool,
        supports_mrl: bool,
    ) -> EmbedderResult<Self> {
        let mut init = InitOptions::new(model).with_show_download_progress(show_progress);
        if let Some(dir) = cache_dir {
            init = init.with_cache_dir(dir);
        }

        let mut embedding = TextEmbedding::try_new(init)
            .map_err(|e| EmbedderError::Internal(format!("failed to load FastEmbed model: {e}")))?;

        let dim = {
            // Probe a single embedding to derive dimension. fastembed 5.x
            // tightened embed() to take &mut self, so both the probe below
            // and the per-instance embed calls in embed()/embed_batch()
            // need mutable bindings.
            let probe = embedding
                .embed(vec!["dimension probe"], None)
                .map_err(|e| EmbedderError::EmbeddingFailed(format!("probe failed: {e}")))?;
            probe.first().map_or(0, Vec::len)
        };

        if dim == 0 {
            return Err(EmbedderError::Internal(
                "failed to determine embedding dimension".to_string(),
            ));
        }

        Ok(Self {
            model: Mutex::new(embedding),
            id: format!("fastembed-{model_name}"),
            model_name: model_name.to_string(),
            dimension: dim,
            supports_mrl,
        })
    }
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
        let delegate = FsFastEmbedder::load_with_name(model_dir, MODEL_DIR_NAME)
            .map_err(|e| EmbedderError::Unavailable(e.to_string()))?;
        let runtime = RuntimeBuilder::current_thread()
            .build()
            .map_err(|e| EmbedderError::Internal(format!("runtime init failed: {e}")))?;
        let dimension = FrankensearchEmbedder::dimension(&delegate);

        Ok(Self {
            backend: FastEmbedBackend::Frankensearch { runtime, delegate },
            id: EMBEDDER_ID.to_string(),
            dimension,
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
        let _ = show_progress;

        // First try to load from existing locations
        if let Ok(embedder) = Self::try_load() {
            return Ok(embedder);
        }

        Err(EmbedderError::Unavailable(
            "MiniLM model not found. Run 'xf index --semantic' to provision local assets."
                .to_string(),
        ))
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

        let FastEmbedBackend::Frankensearch { runtime, delegate } = &self.backend;
        // `Cx::for_testing()` is a test-internals-only constructor and is not
        // available in a production feature set. `block_on` installs an ambient
        // Cx backed by this runtime's drivers, so take that one instead.
        runtime
            .block_on(async {
                let cx = Cx::current().expect("block_on installs an ambient Cx");
                delegate.embed(&cx, text).await
            })
            .map_err(|e| map_fs_error("embed", e))
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
            return Err(EmbedderError::InvalidInput(
                "all texts are empty".to_string(),
            ));
        }

        let FastEmbedBackend::Frankensearch { runtime, delegate } = &self.backend;
        let embeddings = runtime
            .block_on(async {
                let cx = Cx::current().expect("block_on installs an ambient Cx");
                delegate.embed_batch(&cx, &non_empty_texts).await
            })
            .map_err(|e| map_fs_error("embed_batch", e))?;

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

    fn model_name(&self) -> &str {
        MODEL_DIR_NAME
    }

    fn is_semantic(&self) -> bool {
        true // This IS a semantic embedder
    }
}

impl std::fmt::Debug for FastEmbedModelEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FastEmbedModelEmbedder")
            .field("id", &self.id)
            .field("model_name", &self.model_name)
            .field("dimension", &self.dimension)
            .finish_non_exhaustive()
    }
}

impl Embedder for FastEmbedModelEmbedder {
    fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>> {
        if text.is_empty() {
            return Err(EmbedderError::InvalidInput("empty text".to_string()));
        }

        let embeddings = {
            let mut model = self
                .model
                .lock()
                .map_err(|e| EmbedderError::Internal(format!("model lock poisoned: {e}")))?;
            model
                .embed(vec![text], None)
                .map_err(|e| EmbedderError::EmbeddingFailed(format!("embedding failed: {e}")))?
        };

        let mut embedding = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| EmbedderError::Internal("no embedding returned".to_string()))?;

        l2_normalize(&mut embedding);
        Ok(embedding)
    }

    fn embed_batch(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let (non_empty_indices, non_empty_texts): (Vec<_>, Vec<_>) = texts
            .iter()
            .enumerate()
            .filter(|(_, t)| !t.is_empty())
            .map(|(i, t)| (i, *t))
            .unzip();

        if non_empty_texts.is_empty() {
            return Err(EmbedderError::InvalidInput(
                "all texts are empty".to_string(),
            ));
        }

        let mut embeddings = {
            let mut model = self
                .model
                .lock()
                .map_err(|e| EmbedderError::Internal(format!("model lock poisoned: {e}")))?;
            model.embed(non_empty_texts, None).map_err(|e| {
                EmbedderError::EmbeddingFailed(format!("batch embedding failed: {e}"))
            })?
        };

        for embedding in &mut embeddings {
            l2_normalize(embedding);
        }

        let mut result = vec![Vec::new(); texts.len()];
        for (result_idx, embedding) in non_empty_indices.into_iter().zip(embeddings) {
            result[result_idx] = embedding;
        }

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

    fn model_name(&self) -> &str {
        &self.model_name
    }

    fn is_semantic(&self) -> bool {
        true
    }

    fn supports_mrl(&self) -> bool {
        self.supports_mrl
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

    #[test]
    fn test_minilm_baseline_constants() {
        // MiniLM baseline should have fixed characteristics
        // This is used as the reference point for all model comparisons

        // Fixed 384-dimensional output (no MRL support)
        assert_eq!(EMBEDDING_DIMENSION, 384);

        // Model identifier matches sentence-transformers convention
        assert!(MODEL_DIR_NAME.contains("MiniLM"));

        // Required files for ONNX inference
        assert_eq!(REQUIRED_FILES.len(), 5);
        assert!(REQUIRED_FILES.contains(&"model.onnx"));
        assert!(REQUIRED_FILES.contains(&"tokenizer.json"));
        assert!(REQUIRED_FILES.contains(&"config.json"));
        assert!(REQUIRED_FILES.contains(&"special_tokens_map.json"));
        assert!(REQUIRED_FILES.contains(&"tokenizer_config.json"));
    }

    #[test]
    fn test_embedder_id_format() {
        // ID format: "minilm-{dimension}"
        assert!(EMBEDDER_ID.starts_with("minilm-"));
        let dim_str = EMBEDDER_ID.strip_prefix("minilm-").unwrap();
        let dim: usize = dim_str.parse().unwrap();
        assert_eq!(dim, EMBEDDING_DIMENSION);
    }

    #[test]
    fn test_model_dir_name_convention() {
        // Should match HuggingFace model ID convention
        // all-MiniLM-L6-v2 = all layers, MiniLM architecture, 6 layers, version 2
        assert!(MODEL_DIR_NAME.starts_with("all-"));
        assert!(MODEL_DIR_NAME.contains("MiniLM"));
        assert!(MODEL_DIR_NAME.contains("L6")); // 6 transformer layers
        assert!(MODEL_DIR_NAME.ends_with("v2")); // Version 2
    }

    // Integration tests require actual model files
    #[test]
    #[ignore = "requires model files"]
    fn test_embed_semantic_similarity() {
        use crate::embedder::dot_product;

        let embedder = FastEmbedder::try_load().expect("model not available");

        let happy = embedder.embed("I am very happy today").unwrap();
        let joyful = embedder.embed("I am feeling joyful").unwrap();
        let sad = embedder.embed("I am feeling very sad").unwrap();

        let sim_happy_joyful = dot_product(&happy, &joyful);
        let sim_happy_sad = dot_product(&happy, &sad);

        // "happy" should be more similar to "joyful" than to "sad"
        assert!(
            sim_happy_joyful > sim_happy_sad,
            "semantic similarity failed: happy-joyful={sim_happy_joyful}, happy-sad={sim_happy_sad}"
        );
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_minilm_output_dimensions() {
        let embedder = FastEmbedder::try_load().expect("model not available");

        // Dimension should always be 384 (no MRL support)
        assert_eq!(embedder.dimension(), 384);

        // Embed a test text and verify output dimension
        let embedding = embedder.embed("test text").unwrap();
        assert_eq!(embedding.len(), 384);
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_minilm_l2_normalization() {
        let embedder = FastEmbedder::try_load().expect("model not available");

        let embedding = embedder.embed("any text").unwrap();

        // L2 norm should be approximately 1.0 (unit vector)
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-5,
            "embedding not normalized: L2 norm = {norm}"
        );
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_minilm_batch_matches_individual() {
        let embedder = FastEmbedder::try_load().expect("model not available");

        let texts = ["first text", "second text", "third text"];

        // Embed individually
        let individual: Vec<Vec<f32>> = texts.iter().map(|t| embedder.embed(t).unwrap()).collect();

        // Embed as batch
        let batch = embedder.embed_batch(&texts).unwrap();

        // Results should match
        assert_eq!(individual.len(), batch.len());
        for (i, (ind, bat)) in individual.iter().zip(batch.iter()).enumerate() {
            assert_eq!(ind.len(), bat.len(), "dimension mismatch at index {i}");
            for (a, b) in ind.iter().zip(bat.iter()) {
                assert!(
                    (a - b).abs() < 1e-5,
                    "value mismatch at index {i}: {a} vs {b}"
                );
            }
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_minilm_is_semantic() {
        let embedder = FastEmbedder::try_load().expect("model not available");

        // MiniLM is a semantic embedder (not a hash-based one)
        assert!(embedder.is_semantic());
    }
}
