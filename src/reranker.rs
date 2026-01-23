//! Reranker trait and types for cross-encoder style reranking.
//!
//! Rerankers take a query and candidate documents, returning relevance scores
//! suitable for re-sorting a top-N candidate set.

use thiserror::Error;

/// Errors that can occur during reranking operations.
#[derive(Debug, Error)]
pub enum RerankerError {
    /// The reranker is not available (model files missing, gated, etc).
    #[error("reranker unavailable: {0}")]
    Unavailable(String),

    /// Failed to score candidates.
    #[error("rerank failed: {0}")]
    RerankFailed(String),

    /// Invalid input provided to the reranker.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Internal error during reranking.
    #[error("internal error: {0}")]
    Internal(String),
}

/// Result type for reranker operations.
pub type RerankerResult<T> = Result<T, RerankerError>;

/// Information about a reranker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerankerInfo {
    /// Human-readable model name (registry key).
    pub model_name: String,
    /// Maximum supported input length (tokens, if known).
    pub max_length: usize,
}

/// Trait for CPU reranker implementations.
pub trait Reranker: Send + Sync {
    /// Score a list of documents for a given query.
    ///
    /// Returns a score per document, aligned to input order.
    fn rerank(&self, query: &str, documents: &[&str]) -> RerankerResult<Vec<f32>>;

    /// Human-readable model name (registry key).
    fn model_name(&self) -> &str;

    /// Maximum input length supported by this reranker (tokens, if known).
    fn max_length(&self) -> usize;

    /// Get information about this reranker.
    fn info(&self) -> RerankerInfo {
        RerankerInfo {
            model_name: self.model_name().to_string(),
            max_length: self.max_length(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyReranker;

    impl Reranker for DummyReranker {
        fn rerank(&self, _query: &str, documents: &[&str]) -> RerankerResult<Vec<f32>> {
            Ok(vec![0.0; documents.len()])
        }

        #[allow(clippy::unnecessary_literal_bound)]
        fn model_name(&self) -> &str {
            "dummy"
        }

        fn max_length(&self) -> usize {
            128
        }
    }

    #[test]
    fn test_reranker_info() {
        let rr = DummyReranker;
        let info = rr.info();
        assert_eq!(info.model_name, "dummy");
        assert_eq!(info.max_length, 128);
    }
}
