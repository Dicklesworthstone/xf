//! Model registry for embedders and rerankers.
//!
//! Provides a single place to map model names to concrete backends and
//! enforces shared validation rules (MRL dimensions, availability).

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::embedder::FrankensearchEmbedderAdapter;
use crate::embedder::{Embedder, EmbedderError, EmbedderResult, ModelCategory};
use crate::fastembed_embedder::FastEmbedModelEmbedder;
use crate::flashrank_reranker::FlashRankReranker;
use crate::hash_embedder::{DEFAULT_DIMENSION as HASH_DEFAULT_DIM, HashEmbedder};
use crate::model2vec_embedder::Model2VecEmbedder;
use crate::mxbai_reranker::MxbaiReranker;
use crate::reranker::FrankensearchRerankerAdapter;
use crate::reranker::{Reranker, RerankerError, RerankerResult};
use crate::static_mrl_embedder::StaticMrlEmbedder;

use fastembed::EmbeddingModel;

/// Canonical embedder model keys.
pub const EMBEDDER_HASH_FNV1A_384: &str = "hash-fnv1a-384";
pub const EMBEDDER_MINILM_L6_V2: &str = "all-MiniLM-L6-v2";
pub const EMBEDDER_BGE_SMALL_EN_V15: &str = "bge-small-en-v1.5";
pub const EMBEDDER_NOMIC_V15: &str = "nomic-embed-text-v1.5";
pub const EMBEDDER_E5_SMALL: &str = "multilingual-e5-small";
pub const EMBEDDER_STATIC_MRL_EN_V1: &str = "static-retrieval-mrl-en-v1";
pub const EMBEDDER_POTION_RETRIEVAL_32M: &str = "potion-retrieval-32M";
pub const EMBEDDER_POTION_MULTI_128M: &str = "potion-multilingual-128M";
pub const EMBEDDER_EMBEDDINGGEMMA_300M: &str = "embeddinggemma-300m";

/// Canonical reranker model keys.
pub const RERANKER_NONE: &str = "none";
pub const RERANKER_FLASHRANK_NANO: &str = "flashrank-nano";
pub const RERANKER_MXBAI_XSMALL_V1: &str = "mxbai-rerank-xsmall-v1";

/// Information about a model in the registry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    /// Canonical model name (registry key).
    pub name: String,
    /// Model category for benchmark classification.
    pub category: ModelCategory,
    /// Backend type (e.g., "hash", "fastembed", "model2vec").
    pub backend: String,
    /// Whether this model supports MRL truncation.
    pub supports_mrl: bool,
    /// Native embedding dimensions.
    pub native_dims: usize,
    /// Approximate model size in MB (None if not downloaded or unknown).
    pub size_mb: Option<f64>,
    /// Whether the model is downloaded and available locally.
    pub downloaded: bool,
    /// Whether the model is currently available for use.
    pub available: bool,
}

/// Embedder configuration.
#[derive(Debug, Clone)]
pub struct EmbedderConfig {
    pub model: String,
    pub dimensions: Option<usize>,
    pub show_progress: bool,
    pub cache_dir: Option<PathBuf>,
}

impl EmbedderConfig {
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            dimensions: None,
            show_progress: false,
            cache_dir: None,
        }
    }
}

/// Reranker configuration.
#[derive(Debug, Clone)]
pub struct RerankerConfig {
    pub model: String,
    pub show_progress: bool,
    pub cache_dir: Option<PathBuf>,
}

impl RerankerConfig {
    #[must_use]
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            show_progress: false,
            cache_dir: None,
        }
    }
}

/// Registry for embedding and reranker backends.
#[derive(Debug, Clone, Default)]
pub struct ModelRegistry;

impl ModelRegistry {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Return a list of available embedder names.
    #[must_use]
    pub fn embedder_names(&self) -> Vec<&'static str> {
        vec![
            EMBEDDER_HASH_FNV1A_384,
            EMBEDDER_MINILM_L6_V2,
            EMBEDDER_BGE_SMALL_EN_V15,
            EMBEDDER_NOMIC_V15,
            EMBEDDER_E5_SMALL,
            EMBEDDER_STATIC_MRL_EN_V1,
            EMBEDDER_POTION_RETRIEVAL_32M,
            EMBEDDER_POTION_MULTI_128M,
            EMBEDDER_EMBEDDINGGEMMA_300M,
        ]
    }

    /// Return a list of available reranker names.
    #[must_use]
    pub fn reranker_names(&self) -> Vec<&'static str> {
        vec![
            RERANKER_NONE,
            RERANKER_FLASHRANK_NANO,
            RERANKER_MXBAI_XSMALL_V1,
        ]
    }

    /// Whether an embedder name is known to the registry.
    #[must_use]
    pub fn has_embedder(&self, name: &str) -> bool {
        Self::canonical_embedder_name(name).is_some()
    }

    /// Resolve any accepted alias to the canonical registry key
    /// (e.g. `sentence-transformers/all-MiniLM-L6-v2` → `all-MiniLM-L6-v2`).
    #[must_use]
    pub fn canonical_name(name: &str) -> Option<&'static str> {
        Self::canonical_embedder_name(name)
    }

    /// Whether a reranker name is known to the registry.
    #[must_use]
    pub fn has_reranker(&self, name: &str) -> bool {
        Self::canonical_reranker_name(name).is_some()
    }

    /// Convenience method to create an embedder by name.
    ///
    /// Uses default configuration (no MRL truncation, no progress bar).
    pub fn embedder_by_name(&self, name: &str) -> EmbedderResult<Box<dyn Embedder>> {
        self.embedder(&EmbedderConfig::new(name))
    }

    /// Create a frankensearch-compatible embedder adapter by name.
    pub fn frankensearch_embedder_by_name(
        &self,
        name: &str,
    ) -> EmbedderResult<FrankensearchEmbedderAdapter> {
        self.embedder_by_name(name)
            .map(FrankensearchEmbedderAdapter::new)
    }

    /// Convenience method to create a reranker by name.
    ///
    /// Uses default configuration (no progress bar).
    pub fn reranker_by_name(&self, name: &str) -> RerankerResult<Option<Box<dyn Reranker>>> {
        self.reranker(&RerankerConfig::new(name))
    }

    /// Create a frankensearch-compatible reranker adapter by name.
    pub fn frankensearch_reranker_by_name(
        &self,
        name: &str,
    ) -> RerankerResult<Option<FrankensearchRerankerAdapter>> {
        self.reranker_by_name(name)
            .map(|reranker| reranker.map(FrankensearchRerankerAdapter::new))
    }

    /// List all known models with their metadata.
    ///
    /// Returns information about each model including category, backend,
    /// dimensions, MRL support, and availability status.
    #[must_use]
    pub fn list_models(&self) -> Vec<ModelInfo> {
        use crate::fastembed_embedder::fastembed_model_downloaded;
        vec![
            ModelInfo {
                name: EMBEDDER_HASH_FNV1A_384.to_string(),
                category: ModelCategory::StaticEmbedder,
                backend: "hash".to_string(),
                supports_mrl: false,
                native_dims: HASH_DEFAULT_DIM,
                size_mb: Some(0.0), // No model files
                downloaded: true,   // Always available
                available: true,
            },
            ModelInfo {
                name: EMBEDDER_MINILM_L6_V2.to_string(),
                category: ModelCategory::TransformerEmbedder,
                backend: "fastembed".to_string(),
                supports_mrl: false,
                native_dims: 384,
                size_mb: Some(80.0),
                downloaded: fastembed_model_downloaded(&EmbeddingModel::AllMiniLML6V2),
                available: true, // fastembed auto-downloads on first use
            },
            ModelInfo {
                name: EMBEDDER_BGE_SMALL_EN_V15.to_string(),
                category: ModelCategory::TransformerEmbedder,
                backend: "fastembed".to_string(),
                supports_mrl: false,
                native_dims: 384,
                size_mb: Some(130.0),
                downloaded: fastembed_model_downloaded(&EmbeddingModel::BGESmallENV15),
                available: true,
            },
            ModelInfo {
                name: EMBEDDER_NOMIC_V15.to_string(),
                category: ModelCategory::TransformerEmbedder,
                backend: "fastembed".to_string(),
                supports_mrl: true, // Nomic supports MRL
                native_dims: 768,
                size_mb: Some(560.0),
                downloaded: fastembed_model_downloaded(&EmbeddingModel::NomicEmbedTextV15),
                available: true,
            },
            ModelInfo {
                name: EMBEDDER_E5_SMALL.to_string(),
                category: ModelCategory::TransformerEmbedder,
                backend: "fastembed".to_string(),
                supports_mrl: false,
                native_dims: 384,
                size_mb: Some(470.0),
                downloaded: fastembed_model_downloaded(&EmbeddingModel::MultilingualE5Small),
                available: true,
            },
            ModelInfo {
                name: EMBEDDER_STATIC_MRL_EN_V1.to_string(),
                category: ModelCategory::StaticEmbedder,
                backend: "onnx".to_string(),
                supports_mrl: true,
                native_dims: 1024,
                size_mb: Some(100.0),
                downloaded: StaticMrlEmbedder::is_available(),
                available: StaticMrlEmbedder::is_available(),
            },
            ModelInfo {
                name: EMBEDDER_POTION_RETRIEVAL_32M.to_string(),
                category: ModelCategory::StaticEmbedder,
                backend: "model2vec".to_string(),
                supports_mrl: false,
                native_dims: 256,
                size_mb: Some(32.0),
                downloaded: Model2VecEmbedder::is_available(EMBEDDER_POTION_RETRIEVAL_32M),
                available: Model2VecEmbedder::is_available(EMBEDDER_POTION_RETRIEVAL_32M),
            },
            ModelInfo {
                name: EMBEDDER_POTION_MULTI_128M.to_string(),
                category: ModelCategory::StaticEmbedder,
                backend: "model2vec".to_string(),
                supports_mrl: false,
                native_dims: 256,
                size_mb: Some(128.0),
                downloaded: Model2VecEmbedder::is_available(EMBEDDER_POTION_MULTI_128M),
                available: Model2VecEmbedder::is_available(EMBEDDER_POTION_MULTI_128M),
            },
            ModelInfo {
                name: EMBEDDER_EMBEDDINGGEMMA_300M.to_string(),
                category: ModelCategory::TransformerEmbedder,
                backend: "fastembed".to_string(),
                supports_mrl: true,
                native_dims: 768,
                size_mb: Some(600.0),
                downloaded: false,
                available: false, // Not implemented yet
            },
        ]
    }

    /// Build an embedder from configuration.
    pub fn embedder(&self, config: &EmbedderConfig) -> EmbedderResult<Box<dyn Embedder>> {
        let name = Self::canonical_embedder_name(&config.model).ok_or_else(|| {
            EmbedderError::InvalidInput(format!("unknown embedder: {}", config.model))
        })?;

        // Use a stable cache dir for fastembed models unless the caller
        // supplied one, so that index-time downloads and search-time loads
        // resolve the same location regardless of the current directory.
        let fastembed_cache = || {
            Some(
                config
                    .cache_dir
                    .clone()
                    .unwrap_or_else(crate::fastembed_embedder::default_fastembed_cache_dir),
            )
        };

        let mut embedder: Box<dyn Embedder> = match name {
            EMBEDDER_HASH_FNV1A_384 => {
                let dim = config.dimensions.unwrap_or(HASH_DEFAULT_DIM);
                Box::new(HashEmbedder::new(dim))
            }
            EMBEDDER_MINILM_L6_V2 => Box::new(FastEmbedModelEmbedder::load_or_download(
                EmbeddingModel::AllMiniLML6V2,
                name,
                fastembed_cache(),
                config.show_progress,
                false,
            )?),
            EMBEDDER_BGE_SMALL_EN_V15 => Box::new(FastEmbedModelEmbedder::load_or_download(
                EmbeddingModel::BGESmallENV15,
                name,
                fastembed_cache(),
                config.show_progress,
                false,
            )?),
            EMBEDDER_NOMIC_V15 => Box::new(FastEmbedModelEmbedder::load_or_download(
                EmbeddingModel::NomicEmbedTextV15,
                name,
                fastembed_cache(),
                config.show_progress,
                true,
            )?),
            EMBEDDER_E5_SMALL => Box::new(FastEmbedModelEmbedder::load_or_download(
                EmbeddingModel::MultilingualE5Small,
                name,
                fastembed_cache(),
                config.show_progress,
                false,
            )?),
            EMBEDDER_STATIC_MRL_EN_V1 => {
                let dims = config.dimensions;
                Box::new(
                    StaticMrlEmbedder::try_load_with_dims(dims)
                        .map_err(|e| EmbedderError::Unavailable(format!("{e}")))?,
                )
            }
            EMBEDDER_POTION_RETRIEVAL_32M => Box::new(
                Model2VecEmbedder::try_load(EMBEDDER_POTION_RETRIEVAL_32M)
                    .map_err(|e| EmbedderError::Unavailable(format!("{e}")))?,
            ),
            EMBEDDER_POTION_MULTI_128M => Box::new(
                Model2VecEmbedder::try_load(EMBEDDER_POTION_MULTI_128M)
                    .map_err(|e| EmbedderError::Unavailable(format!("{e}")))?,
            ),
            EMBEDDER_EMBEDDINGGEMMA_300M => {
                return Err(EmbedderError::Unavailable(
                    "embeddinggemma-300m backend not implemented yet".to_string(),
                ));
            }
            _ => {
                return Err(EmbedderError::InvalidInput(format!(
                    "unsupported embedder: {name}"
                )));
            }
        };

        if let Some(target_dim) = config.dimensions {
            if target_dim != embedder.dimension() {
                if embedder.supports_mrl() {
                    embedder = Box::new(TruncateEmbedder::new(embedder, target_dim)?);
                } else {
                    return Err(EmbedderError::InvalidInput(format!(
                        "model {} does not support MRL truncation",
                        embedder.model_name()
                    )));
                }
            }
        }

        Ok(embedder)
    }

    /// Build a reranker from configuration.
    pub fn reranker(&self, config: &RerankerConfig) -> RerankerResult<Option<Box<dyn Reranker>>> {
        let name = Self::canonical_reranker_name(&config.model).ok_or_else(|| {
            RerankerError::InvalidInput(format!("unknown reranker: {}", config.model))
        })?;

        match name {
            RERANKER_NONE => Ok(None),
            RERANKER_FLASHRANK_NANO => {
                let reranker = FlashRankReranker::load()?;
                Ok(Some(Box::new(reranker)))
            }
            RERANKER_MXBAI_XSMALL_V1 => {
                let reranker = MxbaiReranker::load()?;
                Ok(Some(Box::new(reranker)))
            }
            _ => Err(RerankerError::InvalidInput(format!(
                "unsupported reranker: {name}"
            ))),
        }
    }

    fn canonical_embedder_name(name: &str) -> Option<&'static str> {
        let lower = name.to_lowercase();
        match lower.as_str() {
            "hash" | "fnv1a" | "fnv1a-384" | "hash-fnv1a-384" => Some(EMBEDDER_HASH_FNV1A_384),
            "all-minilm-l6-v2" | "sentence-transformers/all-minilm-l6-v2" => {
                Some(EMBEDDER_MINILM_L6_V2)
            }
            "bge-small-en-v1.5" | "baai/bge-small-en-v1.5" => Some(EMBEDDER_BGE_SMALL_EN_V15),
            "nomic-embed-text-v1.5" | "nomic-ai/nomic-embed-text-v1.5" => Some(EMBEDDER_NOMIC_V15),
            "multilingual-e5-small" | "intfloat/multilingual-e5-small" => Some(EMBEDDER_E5_SMALL),
            "static-retrieval-mrl-en-v1" | "sentence-transformers/static-retrieval-mrl-en-v1" => {
                Some(EMBEDDER_STATIC_MRL_EN_V1)
            }
            "potion-retrieval-32m" | "minishlab/potion-retrieval-32m" => {
                Some(EMBEDDER_POTION_RETRIEVAL_32M)
            }
            "potion-multilingual-128m" | "minishlab/potion-multilingual-128m" => {
                Some(EMBEDDER_POTION_MULTI_128M)
            }
            "embeddinggemma-300m" | "google/embeddinggemma-300m" => {
                Some(EMBEDDER_EMBEDDINGGEMMA_300M)
            }
            _ => None,
        }
    }

    fn canonical_reranker_name(name: &str) -> Option<&'static str> {
        let lower = name.to_lowercase();
        match lower.as_str() {
            "none" | "off" => Some(RERANKER_NONE),
            "flashrank" | "flashrank-nano" | "flashrank/ms-marco-nano" => {
                Some(RERANKER_FLASHRANK_NANO)
            }
            "mxbai-rerank-xsmall-v1" | "mixedbread-ai/mxbai-rerank-xsmall-v1" => {
                Some(RERANKER_MXBAI_XSMALL_V1)
            }
            _ => None,
        }
    }
}

/// Embedder wrapper that truncates embeddings to a target dimension.
struct TruncateEmbedder {
    inner: Box<dyn Embedder>,
    target_dim: usize,
}

impl TruncateEmbedder {
    fn new(inner: Box<dyn Embedder>, target_dim: usize) -> EmbedderResult<Self> {
        if !inner.supports_mrl() {
            return Err(EmbedderError::InvalidInput(
                "model does not support MRL truncation".to_string(),
            ));
        }
        if target_dim == 0 || target_dim > inner.dimension() {
            return Err(EmbedderError::InvalidInput(format!(
                "target dimension {target_dim} must be between 1 and {} (native dimension)",
                inner.dimension()
            )));
        }
        Ok(Self { inner, target_dim })
    }
}

impl Embedder for TruncateEmbedder {
    fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>> {
        let embedding = self.inner.embed(text)?;
        self.inner.truncate_embedding(&embedding, self.target_dim)
    }

    fn embed_batch(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        let embeddings = self.inner.embed_batch(texts)?;
        self.inner.truncate_batch(&embeddings, self.target_dim)
    }

    fn dimension(&self) -> usize {
        self.target_dim
    }

    fn id(&self) -> &str {
        self.inner.id()
    }

    fn model_name(&self) -> &str {
        self.inner.model_name()
    }

    fn is_semantic(&self) -> bool {
        self.inner.is_semantic()
    }

    fn supports_mrl(&self) -> bool {
        self.inner.supports_mrl()
    }

    fn category(&self) -> ModelCategory {
        self.inner.category()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_embedder_names() {
        assert_eq!(
            ModelRegistry::canonical_embedder_name("sentence-transformers/all-MiniLM-L6-v2"),
            Some(EMBEDDER_MINILM_L6_V2)
        );
        assert_eq!(
            ModelRegistry::canonical_embedder_name("minishlab/potion-retrieval-32M"),
            Some(EMBEDDER_POTION_RETRIEVAL_32M)
        );
    }

    #[test]
    fn test_canonical_reranker_names() {
        assert_eq!(
            ModelRegistry::canonical_reranker_name("flashrank/ms-marco-nano"),
            Some(RERANKER_FLASHRANK_NANO)
        );
        assert_eq!(
            ModelRegistry::canonical_reranker_name("none"),
            Some(RERANKER_NONE)
        );
    }

    #[test]
    fn test_list_models() {
        let registry = ModelRegistry::new();
        let models = registry.list_models();

        // Should have entries for all known embedders
        assert!(!models.is_empty());

        // Hash embedder should always be available
        let hash = models.iter().find(|m| m.name == EMBEDDER_HASH_FNV1A_384);
        assert!(hash.is_some());
        let hash = hash.unwrap();
        assert!(hash.available);
        assert!(hash.downloaded);
        assert_eq!(hash.category, ModelCategory::StaticEmbedder);
        assert_eq!(hash.backend, "hash");

        // MiniLM should be a transformer
        let minilm = models.iter().find(|m| m.name == EMBEDDER_MINILM_L6_V2);
        assert!(minilm.is_some());
        let minilm = minilm.unwrap();
        assert_eq!(minilm.category, ModelCategory::TransformerEmbedder);
        assert_eq!(minilm.backend, "fastembed");
    }

    #[test]
    fn test_embedder_by_name() {
        let registry = ModelRegistry::new();

        // Hash embedder should work
        let embedder = registry.embedder_by_name("hash");
        assert!(embedder.is_ok());
        let embedder = embedder.unwrap();
        assert_eq!(embedder.category(), ModelCategory::StaticEmbedder);

        // Unknown embedder should fail
        let unknown = registry.embedder_by_name("unknown-model-xyz");
        assert!(unknown.is_err());
    }

    #[test]
    fn test_reranker_by_name() {
        let registry = ModelRegistry::new();

        // "none" reranker should return None (no reranker)
        let none_reranker = registry.reranker_by_name("none");
        assert!(none_reranker.is_ok());
        assert!(none_reranker.unwrap().is_none());

        // Unknown reranker should fail
        let unknown = registry.reranker_by_name("unknown-reranker-xyz");
        assert!(unknown.is_err());
    }

    #[test]
    fn test_model_category_display() {
        assert_eq!(format!("{}", ModelCategory::StaticEmbedder), "static");
        assert_eq!(
            format!("{}", ModelCategory::TransformerEmbedder),
            "transformer"
        );
    }

    /// Test-only embedder that reports MRL support for validation testing.
    struct MockMrlEmbedder {
        dim: usize,
    }

    impl Embedder for MockMrlEmbedder {
        fn embed(&self, _text: &str) -> EmbedderResult<Vec<f32>> {
            Ok(vec![0.0; self.dim])
        }

        fn dimension(&self) -> usize {
            self.dim
        }

        fn id(&self) -> &'static str {
            "mock-mrl"
        }

        fn is_semantic(&self) -> bool {
            true
        }

        fn supports_mrl(&self) -> bool {
            true
        }
    }

    #[test]
    fn test_truncate_embedder_rejects_oversized_dimension() {
        let inner: Box<dyn Embedder> = Box::new(MockMrlEmbedder { dim: 384 });
        let result = TruncateEmbedder::new(inner, 999);
        let err = result.err().expect("should fail for oversized dimension");
        assert!(
            err.to_string().contains("target dimension 999"),
            "error should mention the invalid dimension: {err}"
        );
    }

    #[test]
    fn test_truncate_embedder_rejects_zero_dimension() {
        let inner: Box<dyn Embedder> = Box::new(MockMrlEmbedder { dim: 384 });
        let result = TruncateEmbedder::new(inner, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_truncate_embedder_accepts_valid_dimension() {
        let inner: Box<dyn Embedder> = Box::new(MockMrlEmbedder { dim: 384 });
        let result = TruncateEmbedder::new(inner, 128);
        assert!(result.is_ok());
        assert_eq!(result.unwrap().dimension(), 128);
    }

    #[test]
    fn test_truncate_embedder_accepts_native_dimension() {
        // target_dim == native_dim is handled by the registry (skips TruncateEmbedder),
        // but TruncateEmbedder::new should still accept it as valid.
        let inner: Box<dyn Embedder> = Box::new(MockMrlEmbedder { dim: 384 });
        let result = TruncateEmbedder::new(inner, 384);
        assert!(result.is_ok());
    }

    #[test]
    fn test_truncate_embedder_rejects_non_mrl_model() {
        use crate::hash_embedder::HashEmbedder;
        let inner: Box<dyn Embedder> = Box::new(HashEmbedder::new(384));
        let result = TruncateEmbedder::new(inner, 128);
        let err = result.err().expect("should fail for non-MRL model");
        assert!(err.to_string().contains("MRL"));
    }
}
