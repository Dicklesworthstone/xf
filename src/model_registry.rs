//! Model registry for embedders and rerankers.
//!
//! Provides a single place to map model names to concrete backends and
//! enforces shared validation rules (MRL dimensions, availability).

use std::path::PathBuf;

use crate::embedder::{Embedder, EmbedderError, EmbedderResult};
use crate::fastembed_embedder::FastEmbedModelEmbedder;
use crate::hash_embedder::{DEFAULT_DIMENSION as HASH_DEFAULT_DIM, HashEmbedder};
use crate::reranker::{Reranker, RerankerError, RerankerResult};

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

    /// Whether a reranker name is known to the registry.
    #[must_use]
    pub fn has_reranker(&self, name: &str) -> bool {
        Self::canonical_reranker_name(name).is_some()
    }

    /// Build an embedder from configuration.
    pub fn embedder(&self, config: &EmbedderConfig) -> EmbedderResult<Box<dyn Embedder>> {
        let name = Self::canonical_embedder_name(&config.model).ok_or_else(|| {
            EmbedderError::InvalidInput(format!("unknown embedder: {}", config.model))
        })?;

        let mut embedder: Box<dyn Embedder> = match name {
            EMBEDDER_HASH_FNV1A_384 => {
                let dim = config.dimensions.unwrap_or(HASH_DEFAULT_DIM);
                Box::new(HashEmbedder::new(dim))
            }
            EMBEDDER_MINILM_L6_V2 => Box::new(FastEmbedModelEmbedder::load_or_download(
                EmbeddingModel::AllMiniLML6V2,
                name,
                config.cache_dir.clone(),
                config.show_progress,
            )?),
            EMBEDDER_BGE_SMALL_EN_V15 => Box::new(FastEmbedModelEmbedder::load_or_download(
                EmbeddingModel::BGESmallENV15,
                name,
                config.cache_dir.clone(),
                config.show_progress,
            )?),
            EMBEDDER_NOMIC_V15 => Box::new(FastEmbedModelEmbedder::load_or_download(
                EmbeddingModel::NomicEmbedTextV15,
                name,
                config.cache_dir.clone(),
                config.show_progress,
            )?),
            EMBEDDER_E5_SMALL => Box::new(FastEmbedModelEmbedder::load_or_download(
                EmbeddingModel::MultilingualE5Small,
                name,
                config.cache_dir.clone(),
                config.show_progress,
            )?),
            EMBEDDER_STATIC_MRL_EN_V1 => {
                return Err(EmbedderError::Unavailable(
                    "static-retrieval-mrl-en-v1 backend not implemented yet".to_string(),
                ));
            }
            EMBEDDER_POTION_RETRIEVAL_32M => {
                return Err(EmbedderError::Unavailable(
                    "model2vec potion-retrieval-32M backend not implemented yet".to_string(),
                ));
            }
            EMBEDDER_POTION_MULTI_128M => {
                return Err(EmbedderError::Unavailable(
                    "model2vec potion-multilingual-128M backend not implemented yet".to_string(),
                ));
            }
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
            RERANKER_FLASHRANK_NANO => Err(RerankerError::Unavailable(
                "flashrank-nano backend not implemented yet".to_string(),
            )),
            RERANKER_MXBAI_XSMALL_V1 => Err(RerankerError::Unavailable(
                "mxbai-rerank-xsmall-v1 backend not implemented yet".to_string(),
            )),
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
}
