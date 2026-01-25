//! Reranker quality evaluation framework.
//!
//! Measures the quality improvement that reranking provides over initial retrieval,
//! computing NDCG, MRR, MAP, Precision, and Recall metrics at various cutoffs.
//!
//! # Usage
//!
//! ```rust,ignore
//! let benchmark = RerankQualityBenchmark::new(corpus)
//!     .with_embedder(Box::new(embedder))
//!     .with_reranker(Box::new(reranker));
//! let reports = benchmark.run()?;
//! ```

use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use super::datasets::{BenchmarkCorpus, QueryType};
use super::metrics::{
    bootstrap_mean_diff, compute_mrr, map_at_k, ndcg_at_k, percentile, precision_at_k, recall_at_k,
};
use crate::embedder::Embedder;
use crate::reranker::Reranker;

/// Configuration for rerank quality benchmark.
#[derive(Debug, Clone)]
pub struct RerankQualityConfig {
    /// Maximum number of candidates to retrieve for reranking.
    pub top_k_retrieve: usize,
    /// Cutoffs for computing metrics.
    pub k_values: Vec<usize>,
    /// Number of bootstrap iterations for significance testing.
    pub n_bootstrap: usize,
    /// Random seed for reproducibility.
    pub seed: u64,
}

impl Default for RerankQualityConfig {
    fn default() -> Self {
        Self {
            top_k_retrieve: 100,
            k_values: vec![5, 10, 20, 50, 100],
            n_bootstrap: 1000,
            seed: 42,
        }
    }
}

/// Metrics computed at various cutoff values.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MetricsAtK {
    /// NDCG at k=5.
    pub ndcg_5: f64,
    /// NDCG at k=10.
    pub ndcg_10: f64,
    /// NDCG at k=20.
    pub ndcg_20: f64,
    /// Mean Reciprocal Rank.
    pub mrr: f64,
    /// Mean Average Precision at k=10.
    pub map_10: f64,
    /// Precision at k=1.
    pub precision_1: f64,
    /// Precision at k=3.
    pub precision_3: f64,
    /// Precision at k=5.
    pub precision_5: f64,
    /// Precision at k=10.
    pub precision_10: f64,
    /// Recall at k=10.
    pub recall_10: f64,
    /// Recall at k=20.
    pub recall_20: f64,
    /// Recall at k=50.
    pub recall_50: f64,
    /// Recall at k=100.
    pub recall_100: f64,
}

/// Percentage lift over baseline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LiftMetrics {
    /// NDCG@10 lift percentage.
    pub ndcg_10_lift_pct: f64,
    /// MRR lift percentage.
    pub mrr_lift_pct: f64,
    /// Precision@5 lift percentage.
    pub precision_5_lift_pct: f64,
    /// Whether the NDCG@10 improvement is statistically significant.
    pub ndcg_10_significant: bool,
}

/// Latency statistics for the pipeline.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LatencyStats {
    /// Mean latency in milliseconds.
    pub mean_ms: f64,
    /// 50th percentile latency.
    pub p50_ms: f64,
    /// 95th percentile latency.
    pub p95_ms: f64,
    /// 99th percentile latency.
    pub p99_ms: f64,
    /// Maximum latency.
    pub max_ms: f64,
}

impl LatencyStats {
    /// Create from a list of millisecond values.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn from_millis(values: &[f64]) -> Self {
        if values.is_empty() {
            return Self::default();
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        Self {
            mean_ms: mean,
            p50_ms: percentile(values, 0.50).unwrap_or(0.0),
            p95_ms: percentile(values, 0.95).unwrap_or(0.0),
            p99_ms: percentile(values, 0.99).unwrap_or(0.0),
            max_ms: values.iter().copied().fold(0.0_f64, f64::max),
        }
    }
}

/// Full quality report for an embedder+reranker combination.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityReport {
    /// Embedder model name.
    pub embedder: String,
    /// Reranker model name (or "none" for baseline).
    pub reranker: String,
    /// Overall metrics.
    pub metrics: MetricsAtK,
    /// Lift over no-rerank baseline.
    pub lift_over_baseline: LiftMetrics,
    /// Metrics broken down by query type.
    pub by_query_type: HashMap<String, MetricsAtK>,
    /// Latency statistics.
    pub latency: LatencyStats,
    /// Number of queries evaluated.
    pub num_queries: usize,
}

/// Accumulator for computing aggregate metrics across queries.
#[derive(Debug, Default)]
struct MetricsAccumulator {
    ndcg_5_sum: f64,
    ndcg_10_sum: f64,
    ndcg_20_sum: f64,
    mrr_sum: f64,
    map_10_sum: f64,
    precision_1_sum: f64,
    precision_3_sum: f64,
    precision_5_sum: f64,
    precision_10_sum: f64,
    recall_10_sum: f64,
    recall_20_sum: f64,
    recall_50_sum: f64,
    recall_100_sum: f64,
    count: usize,
    ndcg_10_values: Vec<f64>,
}

impl MetricsAccumulator {
    fn add(&mut self, relevance: &[u8], total_relevant: usize) {
        self.ndcg_5_sum += ndcg_at_k(relevance, 5);
        let ndcg_10 = ndcg_at_k(relevance, 10);
        self.ndcg_10_sum += ndcg_10;
        self.ndcg_10_values.push(ndcg_10);
        self.ndcg_20_sum += ndcg_at_k(relevance, 20);
        self.mrr_sum += compute_mrr(relevance);
        self.map_10_sum += map_at_k(relevance, 10);
        self.precision_1_sum += precision_at_k(relevance, 1);
        self.precision_3_sum += precision_at_k(relevance, 3);
        self.precision_5_sum += precision_at_k(relevance, 5);
        self.precision_10_sum += precision_at_k(relevance, 10);
        self.recall_10_sum += recall_at_k(relevance, 10, total_relevant);
        self.recall_20_sum += recall_at_k(relevance, 20, total_relevant);
        self.recall_50_sum += recall_at_k(relevance, 50, total_relevant);
        self.recall_100_sum += recall_at_k(relevance, 100, total_relevant);
        self.count += 1;
    }

    #[allow(clippy::cast_precision_loss)]
    fn finalize(&self) -> MetricsAtK {
        if self.count == 0 {
            return MetricsAtK::default();
        }
        let n = self.count as f64;
        MetricsAtK {
            ndcg_5: self.ndcg_5_sum / n,
            ndcg_10: self.ndcg_10_sum / n,
            ndcg_20: self.ndcg_20_sum / n,
            mrr: self.mrr_sum / n,
            map_10: self.map_10_sum / n,
            precision_1: self.precision_1_sum / n,
            precision_3: self.precision_3_sum / n,
            precision_5: self.precision_5_sum / n,
            precision_10: self.precision_10_sum / n,
            recall_10: self.recall_10_sum / n,
            recall_20: self.recall_20_sum / n,
            recall_50: self.recall_50_sum / n,
            recall_100: self.recall_100_sum / n,
        }
    }
}

/// Reranker quality benchmark runner.
pub struct RerankQualityBenchmark {
    corpus: BenchmarkCorpus,
    config: RerankQualityConfig,
    embedders: Vec<Box<dyn Embedder>>,
    rerankers: Vec<Box<dyn Reranker>>,
}

impl RerankQualityBenchmark {
    /// Create a new benchmark with the given corpus.
    #[must_use]
    pub fn new(corpus: BenchmarkCorpus) -> Self {
        Self {
            corpus,
            config: RerankQualityConfig::default(),
            embedders: Vec::new(),
            rerankers: Vec::new(),
        }
    }

    /// Set custom configuration.
    #[must_use]
    pub fn with_config(mut self, config: RerankQualityConfig) -> Self {
        self.config = config;
        self
    }

    /// Add an embedder to evaluate.
    #[must_use]
    pub fn with_embedder(mut self, embedder: Box<dyn Embedder>) -> Self {
        self.embedders.push(embedder);
        self
    }

    /// Add a reranker to evaluate.
    #[must_use]
    pub fn with_reranker(mut self, reranker: Box<dyn Reranker>) -> Self {
        self.rerankers.push(reranker);
        self
    }

    /// Run the benchmark and produce quality reports.
    pub fn run(&self) -> Result<Vec<QualityReport>> {
        let mut reports = Vec::new();

        // Pre-embed all documents
        let doc_texts: Vec<&str> = self.corpus.corpus.iter().map(|d| d.text.as_str()).collect();

        for embedder in &self.embedders {
            tracing::info!(embedder = embedder.model_name(), "Embedding corpus");
            let doc_embeddings = embedder.embed_batch(&doc_texts)?;

            // Baseline: no reranking
            let (baseline_result, baseline_ndcg_values) =
                self.evaluate_pipeline(embedder.as_ref(), &doc_embeddings, None)?;
            tracing::info!(
                embedder = embedder.model_name(),
                ndcg_10 = format!("{:.4}", baseline_result.metrics.ndcg_10),
                mrr = format!("{:.4}", baseline_result.metrics.mrr),
                "Baseline (no rerank)"
            );

            let baseline_report = QualityReport {
                embedder: embedder.model_name().to_string(),
                reranker: "none".to_string(),
                metrics: baseline_result.metrics.clone(),
                lift_over_baseline: LiftMetrics::default(),
                by_query_type: baseline_result.by_query_type,
                latency: baseline_result.latency,
                num_queries: baseline_result.num_queries,
            };
            reports.push(baseline_report);

            // Evaluate with each reranker
            for reranker in &self.rerankers {
                let (result, reranked_ndcg_values) =
                    self.evaluate_pipeline(embedder.as_ref(), &doc_embeddings, Some(reranker.as_ref()))?;

                // Compute lift and significance
                let lift = compute_lift(&baseline_result.metrics, &result.metrics);
                let (_, _, significant) = bootstrap_mean_diff(
                    &baseline_ndcg_values,
                    &reranked_ndcg_values,
                    self.config.n_bootstrap,
                    self.config.seed,
                );

                let lift_with_sig = LiftMetrics {
                    ndcg_10_significant: significant,
                    ..lift
                };

                tracing::info!(
                    embedder = embedder.model_name(),
                    reranker = reranker.model_name(),
                    ndcg_10 = format!("{:.4}", result.metrics.ndcg_10),
                    lift = format!("{:+.1}%", lift_with_sig.ndcg_10_lift_pct),
                    significant = significant,
                    "Reranked results"
                );

                reports.push(QualityReport {
                    embedder: embedder.model_name().to_string(),
                    reranker: reranker.model_name().to_string(),
                    metrics: result.metrics,
                    lift_over_baseline: lift_with_sig,
                    by_query_type: result.by_query_type,
                    latency: result.latency,
                    num_queries: result.num_queries,
                });
            }
        }

        Ok(reports)
    }

    /// Evaluate a single pipeline configuration.
    #[allow(clippy::too_many_lines)]
    fn evaluate_pipeline(
        &self,
        embedder: &dyn Embedder,
        doc_embeddings: &[Vec<f32>],
        reranker: Option<&dyn Reranker>,
    ) -> Result<(PipelineResult, Vec<f64>)> {
        let mut metrics_accum = MetricsAccumulator::default();
        let mut type_accum: HashMap<QueryType, MetricsAccumulator> = HashMap::new();
        let mut latencies = Vec::new();

        for query in &self.corpus.queries {
            let start = Instant::now();

            // Embed query
            let query_embedding = embedder.embed(&query.text)?;

            // Compute similarities and rank
            let mut scored: Vec<(usize, f32)> = doc_embeddings
                .iter()
                .enumerate()
                .map(|(i, doc_emb)| {
                    let sim = cosine_similarity(&query_embedding, doc_emb);
                    (i, sim)
                })
                .collect();

            // Sort by similarity (descending)
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Take top-k candidates
            let candidates: Vec<usize> = scored
                .iter()
                .take(self.config.top_k_retrieve)
                .map(|(i, _)| *i)
                .collect();

            // Optionally rerank
            let final_ranking = if let Some(rr) = reranker {
                let texts: Vec<&str> = candidates
                    .iter()
                    .map(|&i| self.corpus.corpus[i].text.as_str())
                    .collect();
                let rerank_scores = rr.rerank(&query.text, &texts)?;
                let mut rescored: Vec<(usize, f32)> =
                    candidates.into_iter().zip(rerank_scores).collect();
                rescored.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                rescored.into_iter().map(|(i, _)| i).collect()
            } else {
                candidates
            };

            let elapsed = start.elapsed();
            latencies.push(duration_to_ms(elapsed));

            // Build relevance vector from ranking
            let relevance: Vec<u8> = final_ranking
                .iter()
                .map(|&i| {
                    let doc_id = &self.corpus.corpus[i].id;
                    query.relevants.get(doc_id).copied().unwrap_or(0)
                })
                .collect();

            let total_relevant = self.corpus.total_relevant(query);
            metrics_accum.add(&relevance, total_relevant);

            // Accumulate by query type
            if let Some(qtype) = &query.query_type {
                type_accum
                    .entry(*qtype)
                    .or_default()
                    .add(&relevance, total_relevant);
            }
        }

        let metrics = metrics_accum.finalize();
        let ndcg_values = metrics_accum.ndcg_10_values.clone();
        let by_query_type = type_accum
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.finalize()))
            .collect();

        Ok((
            PipelineResult {
                metrics,
                by_query_type,
                latency: LatencyStats::from_millis(&latencies),
                num_queries: self.corpus.queries.len(),
            },
            ndcg_values,
        ))
    }
}

/// Internal result from pipeline evaluation.
struct PipelineResult {
    metrics: MetricsAtK,
    by_query_type: HashMap<String, MetricsAtK>,
    latency: LatencyStats,
    num_queries: usize,
}

/// Compute percentage lift from baseline to improved.
fn compute_lift(baseline: &MetricsAtK, improved: &MetricsAtK) -> LiftMetrics {
    LiftMetrics {
        ndcg_10_lift_pct: pct_lift(baseline.ndcg_10, improved.ndcg_10),
        mrr_lift_pct: pct_lift(baseline.mrr, improved.mrr),
        precision_5_lift_pct: pct_lift(baseline.precision_5, improved.precision_5),
        ndcg_10_significant: false,
    }
}

/// Compute percentage lift.
fn pct_lift(baseline: f64, improved: f64) -> f64 {
    if baseline == 0.0 {
        return 0.0;
    }
    ((improved - baseline) / baseline) * 100.0
}

/// Compute cosine similarity between two vectors.
#[allow(clippy::cast_precision_loss)]
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Convert Duration to milliseconds.
fn duration_to_ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Classify a query into a category based on its text.
#[must_use]
pub fn classify_query(query: &str) -> QueryType {
    let lower = query.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();

    // Check for question words
    if words
        .first()
        .is_some_and(|w| ["who", "what", "when", "where", "why", "how"].contains(w))
    {
        return QueryType::Factual;
    }

    // Check for entity mentions
    if query.contains('@') || query.contains('#') {
        return QueryType::Entity;
    }

    // Check for temporal references
    let temporal_words = [
        "yesterday",
        "today",
        "last",
        "recent",
        "this week",
        "month",
        "year",
        "2024",
        "2025",
        "january",
        "february",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
    ];
    if temporal_words.iter().any(|tw| lower.contains(tw)) {
        return QueryType::Temporal;
    }

    // Check for conversational keywords
    let conversational_words = ["dm", "message", "conversation", "chat", "reply", "thread"];
    if conversational_words.iter().any(|cw| lower.contains(cw)) {
        return QueryType::Conversational;
    }

    // Default to topical
    QueryType::Topical
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pct_lift_positive() {
        assert!((pct_lift(0.5, 0.6) - 20.0).abs() < 0.01);
    }

    #[test]
    fn test_pct_lift_negative() {
        assert!((pct_lift(0.5, 0.4) - (-20.0)).abs() < 0.01);
    }

    #[test]
    fn test_pct_lift_zero_baseline() {
        assert_eq!(pct_lift(0.0, 0.5), 0.0);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 0.0001);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 0.0001);
    }

    #[test]
    fn test_classify_query_question() {
        assert_eq!(
            classify_query("how to learn rust programming"),
            QueryType::Factual
        );
        assert_eq!(
            classify_query("what is machine learning"),
            QueryType::Factual
        );
    }

    #[test]
    fn test_classify_query_entity() {
        assert_eq!(
            classify_query("tweets mentioning @elonmusk"),
            QueryType::Entity
        );
        assert_eq!(classify_query("posts with #rust"), QueryType::Entity);
    }

    #[test]
    fn test_classify_query_temporal() {
        assert_eq!(
            classify_query("messages from yesterday"),
            QueryType::Temporal
        );
        assert_eq!(classify_query("posts from March 2024"), QueryType::Temporal);
    }

    #[test]
    fn test_classify_query_conversational() {
        assert_eq!(
            classify_query("dm about project"),
            QueryType::Conversational
        );
        assert_eq!(
            classify_query("conversation with alice"),
            QueryType::Conversational
        );
    }

    #[test]
    fn test_classify_query_topical() {
        assert_eq!(
            classify_query("machine learning models"),
            QueryType::Topical
        );
    }

    #[test]
    fn test_latency_stats_from_millis() {
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let stats = LatencyStats::from_millis(&values);
        assert!((stats.mean_ms - 30.0).abs() < 0.01);
        assert!(stats.max_ms >= 50.0);
    }

    #[test]
    fn test_latency_stats_empty() {
        let stats = LatencyStats::from_millis(&[]);
        assert_eq!(stats.mean_ms, 0.0);
    }

    #[test]
    fn test_metrics_accumulator() {
        let mut accum = MetricsAccumulator::default();
        // Perfect ranking: all relevant at top
        accum.add(&[2, 1, 0, 0, 0], 2);
        let metrics = accum.finalize();
        assert!((metrics.ndcg_5 - 1.0).abs() < 0.01);
        assert!((metrics.mrr - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_compute_lift() {
        let baseline = MetricsAtK {
            ndcg_10: 0.5,
            mrr: 0.4,
            precision_5: 0.3,
            ..Default::default()
        };
        let improved = MetricsAtK {
            ndcg_10: 0.6,
            mrr: 0.5,
            precision_5: 0.4,
            ..Default::default()
        };
        let lift = compute_lift(&baseline, &improved);
        assert!((lift.ndcg_10_lift_pct - 20.0).abs() < 0.01);
        assert!((lift.mrr_lift_pct - 25.0).abs() < 0.01);
    }
}
