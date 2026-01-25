//! Reranking step for search pipeline.
//!
//! Provides a second-pass reranking using cross-encoder models to improve
//! relevance ordering of initial retrieval results.
//!
//! # Example
//!
//! ```rust,ignore
//! let rerank_step = RerankStep::new(&registry, "flashrank-nano")?;
//! rerank_step.rerank("what is rust?", &mut candidates)?;
//! ```

use crate::model::SearchResult;
use crate::model_registry::ModelRegistry;
use crate::reranker::{Reranker, RerankerError, RerankerResult};

/// Default number of candidates to rerank.
const DEFAULT_TOP_K_RERANK: usize = 100;

/// Minimum candidates required before reranking is performed.
const DEFAULT_MIN_CANDIDATES: usize = 5;

/// Reranking step for search pipeline integration.
///
/// Receives top-N candidates from initial retrieval and re-scores them
/// using a cross-encoder model for better relevance ordering.
pub struct RerankStep {
    reranker: Box<dyn Reranker>,
    top_k_rerank: usize,
    min_candidates: usize,
}

impl RerankStep {
    /// Create a new rerank step with the specified reranker model.
    ///
    /// # Arguments
    ///
    /// * `registry` - Model registry for loading reranker
    /// * `reranker_name` - Name of the reranker model to use
    ///
    /// # Errors
    ///
    /// Returns an error if the reranker model cannot be loaded.
    pub fn new(registry: &ModelRegistry, reranker_name: &str) -> RerankerResult<Option<Self>> {
        let reranker = registry.reranker_by_name(reranker_name)?;

        Ok(reranker.map(|r| Self {
            reranker: r,
            top_k_rerank: DEFAULT_TOP_K_RERANK,
            min_candidates: DEFAULT_MIN_CANDIDATES,
        }))
    }

    /// Create a rerank step with a pre-loaded reranker.
    ///
    /// Useful for daemon mode where the reranker is already instantiated.
    #[must_use]
    pub fn with_reranker(reranker: Box<dyn Reranker>) -> Self {
        Self {
            reranker,
            top_k_rerank: DEFAULT_TOP_K_RERANK,
            min_candidates: DEFAULT_MIN_CANDIDATES,
        }
    }

    /// Set the maximum number of candidates to rerank.
    #[must_use]
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k_rerank = top_k;
        self
    }

    /// Set the minimum candidates threshold.
    ///
    /// Reranking is skipped if fewer candidates are provided.
    #[must_use]
    pub const fn with_min_candidates(mut self, min: usize) -> Self {
        self.min_candidates = min;
        self
    }

    /// Get the name of the reranker model being used.
    #[must_use]
    pub fn model_name(&self) -> &str {
        self.reranker.model_name()
    }

    /// Rerank the candidates in place.
    ///
    /// Modifies the candidate list by:
    /// 1. Setting `rerank_score` on reranked candidates
    /// 2. Re-sorting by rerank score (descending), with original score as tiebreaker
    ///
    /// If there are fewer candidates than `min_candidates`, reranking is skipped.
    /// Only the top `top_k_rerank` candidates are reranked; the rest remain unchanged.
    ///
    /// # Errors
    ///
    /// Returns an error if the reranker fails to score the candidates.
    pub fn rerank(&self, query: &str, candidates: &mut [SearchResult]) -> RerankerResult<()> {
        if candidates.len() < self.min_candidates {
            tracing::debug!(
                count = candidates.len(),
                min = self.min_candidates,
                "Skipping rerank: too few candidates"
            );
            return Ok(());
        }

        // Only rerank top-K candidates
        let rerank_count = candidates.len().min(self.top_k_rerank);
        let texts: Vec<&str> = candidates[..rerank_count]
            .iter()
            .map(|r| r.text.as_str())
            .collect();

        tracing::info!(
            query = query,
            candidates = rerank_count,
            model = self.reranker.model_name(),
            "Reranking candidates"
        );

        let start = std::time::Instant::now();
        let scores = self.reranker.rerank(query, &texts)?;
        let elapsed = start.elapsed();

        // Calculate score statistics for logging
        let (top_score, bottom_score) = scores
            .iter()
            .fold((f32::NEG_INFINITY, f32::INFINITY), |(max, min), &s| {
                (max.max(s), min.min(s))
            });

        tracing::info!(
            elapsed_ms = elapsed.as_millis(),
            top_score = %top_score,
            bottom_score = %bottom_score,
            "Reranking complete"
        );

        // Verify score count matches
        if scores.len() != rerank_count {
            return Err(RerankerError::Internal(format!(
                "reranker returned {} scores, expected {}",
                scores.len(),
                rerank_count
            )));
        }

        // Apply reranker scores
        for (i, score) in scores.iter().enumerate() {
            candidates[i].rerank_score = Some(*score);
        }

        // Stable sort: by rerank_score descending, then original score for ties
        candidates[..rerank_count].sort_by(|a, b| {
            let ra = a.rerank_score.unwrap_or(f32::NEG_INFINITY);
            let rb = b.rerank_score.unwrap_or(f32::NEG_INFINITY);
            rb.partial_cmp(&ra)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    b.score
                        .partial_cmp(&a.score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SearchResultType;
    use chrono::Utc;
    use std::sync::Mutex;

    /// Mock reranker for testing that returns predetermined scores.
    struct MockReranker {
        scores: Mutex<Vec<f32>>,
        name: String,
    }

    impl MockReranker {
        fn with_scores(scores: Vec<f32>) -> Self {
            Self {
                scores: Mutex::new(scores),
                name: "mock-reranker".to_string(),
            }
        }

        #[allow(dead_code)]
        fn failing() -> Self {
            Self {
                scores: Mutex::new(vec![]),
                name: "failing-reranker".to_string(),
            }
        }
    }

    impl Reranker for MockReranker {
        fn rerank(&self, _query: &str, documents: &[&str]) -> RerankerResult<Vec<f32>> {
            let scores = self.scores.lock().unwrap();
            if scores.is_empty() && !documents.is_empty() {
                return Err(RerankerError::RerankFailed("mock failure".into()));
            }
            Ok(scores.iter().take(documents.len()).copied().collect())
        }

        fn model_name(&self) -> &str {
            &self.name
        }

        fn max_length(&self) -> usize {
            512
        }
    }

    fn make_result(id: &str, text: &str, score: f32) -> SearchResult {
        SearchResult {
            result_type: SearchResultType::Tweet,
            id: id.to_string(),
            text: text.to_string(),
            created_at: Utc::now(),
            score,
            highlights: vec![],
            metadata: serde_json::Value::Null,
            rerank_score: None,
        }
    }

    #[test]
    fn test_rerank_reorders_candidates() {
        let reranker = MockReranker::with_scores(vec![0.3, 0.9, 0.5]);
        let step = RerankStep::with_reranker(Box::new(reranker)).with_min_candidates(2);

        let mut candidates = vec![
            make_result("A", "Text A", 0.9),
            make_result("B", "Text B", 0.8),
            make_result("C", "Text C", 0.7),
        ];

        step.rerank("query", &mut candidates).unwrap();

        // B should be first (0.9 rerank score)
        assert_eq!(candidates[0].id, "B");
        assert_eq!(candidates[0].rerank_score, Some(0.9));

        // C should be second (0.5 rerank score)
        assert_eq!(candidates[1].id, "C");
        assert_eq!(candidates[1].rerank_score, Some(0.5));

        // A should be third (0.3 rerank score)
        assert_eq!(candidates[2].id, "A");
        assert_eq!(candidates[2].rerank_score, Some(0.3));
    }

    #[test]
    fn test_rerank_skips_for_few_candidates() {
        let reranker = MockReranker::with_scores(vec![0.9, 0.8]);
        let step = RerankStep::with_reranker(Box::new(reranker)).with_min_candidates(5);

        let mut candidates = vec![
            make_result("A", "Text A", 0.9),
            make_result("B", "Text B", 0.8),
        ];

        step.rerank("query", &mut candidates).unwrap();

        // Should not have been reranked
        assert!(candidates[0].rerank_score.is_none());
        assert!(candidates[1].rerank_score.is_none());
    }

    #[test]
    fn test_rerank_only_top_k() {
        let reranker = MockReranker::with_scores(vec![0.5, 0.9]);
        let step = RerankStep::with_reranker(Box::new(reranker))
            .with_top_k(2)
            .with_min_candidates(1);

        let mut candidates = vec![
            make_result("A", "Text A", 0.9),
            make_result("B", "Text B", 0.8),
            make_result("C", "Text C", 0.7),
        ];

        step.rerank("query", &mut candidates).unwrap();

        // Only first two should have rerank scores
        assert!(candidates[0].rerank_score.is_some());
        assert!(candidates[1].rerank_score.is_some());
        // Third item should not be reranked
        assert!(candidates[2].rerank_score.is_none());
    }

    #[test]
    fn test_rerank_stable_sort_on_tie() {
        // When rerank scores are equal, original score breaks ties
        let reranker = MockReranker::with_scores(vec![0.5, 0.5]);
        let step = RerankStep::with_reranker(Box::new(reranker)).with_min_candidates(1);

        let mut candidates = vec![
            make_result("Higher", "Higher original", 0.9),
            make_result("Lower", "Lower original", 0.7),
        ];

        step.rerank("query", &mut candidates).unwrap();

        // Higher original score wins tie
        assert_eq!(candidates[0].id, "Higher");
    }

    #[test]
    fn test_rerank_error_propagates() {
        let reranker = MockReranker::failing();
        let step = RerankStep::with_reranker(Box::new(reranker)).with_min_candidates(1);

        let mut candidates = vec![
            make_result("A", "Text A", 0.9),
            make_result("B", "Text B", 0.8),
        ];

        assert!(step.rerank("query", &mut candidates).is_err());
    }

    #[test]
    fn test_model_name() {
        let reranker = MockReranker::with_scores(vec![]);
        let step = RerankStep::with_reranker(Box::new(reranker));

        assert_eq!(step.model_name(), "mock-reranker");
    }

    #[test]
    fn test_empty_candidates() {
        let reranker = MockReranker::with_scores(vec![]);
        let step = RerankStep::with_reranker(Box::new(reranker)).with_min_candidates(1);

        let mut candidates: Vec<SearchResult> = vec![];
        step.rerank("query", &mut candidates).unwrap();

        assert!(candidates.is_empty());
    }
}
