//! Benchmark runner for embedding and reranker models.
//!
//! This module implements the benchmark harness for the embedding/reranker bake-off.
//! Key components:
//!
//! - [`BenchmarkConfig`]: Configurable warmup/measurement parameters
//! - [`FullBenchmarkResult`]: Complete benchmark output consumed by report generator
//! - [`run_embedding_benchmark`]: Execute a single model benchmark
//! - [`evaluate_embedder_quality`]: Compute quality metrics against ground truth

use std::collections::HashMap;
use std::time::{Duration, Instant};

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::benchmark::datasets::BenchmarkCorpus;
use crate::benchmark::metrics::{
    QualityMetrics, SpeedMetrics, coefficient_of_variation, compute_mrr, map_at_k, ndcg_at_k,
    percentile, precision_at_k, recall_at_k,
};
use crate::embedder::{Embedder, ModelCategory};

/// Configuration for a benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Number of warmup iterations (results discarded).
    pub warmup_iters: usize,
    /// Number of measurement iterations.
    pub measure_iters: usize,
    /// Batch size for embedding calls.
    pub batch_size: usize,
    /// MRL dimensions to test (if model supports MRL).
    #[serde(default)]
    pub mrl_dims: Vec<usize>,
    /// Whether to run quality evaluation.
    #[serde(default = "default_eval_quality")]
    pub eval_quality: bool,
}

const fn default_eval_quality() -> bool {
    true
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iters: 10,
            measure_iters: 100,
            batch_size: 32,
            mrl_dims: vec![64, 128, 256, 512, 768, 1024],
            eval_quality: true,
        }
    }
}

/// Benchmark metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkMetadata {
    pub model: String,
    pub dimension: usize,
    pub batch_size: usize,
    pub warmup_iters: usize,
    pub measure_iters: usize,
}

/// Simple benchmark result for embedding throughput and latency.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub metadata: BenchmarkMetadata,
    pub speed: SpeedMetrics,
    #[serde(default)]
    pub latency_ms: Vec<f64>,
}

/// Complete benchmark result for one model, consumed by report generator.
///
/// This is the primary output struct matching bd-120 specification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FullBenchmarkResult {
    /// Model name identifier.
    pub model_name: String,
    /// Model category (StaticEmbedder or TransformerEmbedder).
    pub category: ModelCategory,
    /// Whether model meets eligibility criteria (release_date >= 2025-11-01).
    pub eligible: bool,
    /// Model release date if known.
    #[serde(default)]
    pub release_date: Option<String>,

    /// Quality metrics from corpus evaluation.
    pub quality: QualityMetrics,

    /// Speed metrics from benchmark phases.
    pub speed: SpeedMetrics,

    /// Total number of benchmark requests.
    pub total_requests: usize,
    /// Number of failed requests.
    pub errors: usize,
    /// Number of OOM errors.
    pub oom_count: usize,
    /// Number of panics (recovered).
    pub panics: usize,

    /// Peak RSS memory in MB.
    pub peak_rss_mb: f64,
    /// Steady-state RSS memory in MB.
    pub steady_rss_mb: f64,
    /// Model file size in MB.
    pub model_file_size_mb: f64,
    /// Peak thread count.
    pub peak_threads: usize,

    /// MRL dimensions tested (if model supports MRL).
    #[serde(default)]
    pub mrl_dims_tested: Vec<usize>,
    /// Quality at each MRL dimension: (dims, ndcg@10).
    #[serde(default)]
    pub mrl_quality_at_dims: Vec<(usize, f64)>,

    /// Reranker lift in NDCG@10 vs no-rerank (reranker benchmarks only).
    #[serde(default)]
    pub reranker_lift: Option<f64>,

    /// Raw latency measurements in ms.
    #[serde(default)]
    pub latency_ms: Vec<f64>,
}

impl FullBenchmarkResult {
    /// Error rate as fraction (0.0-1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn error_rate(&self) -> f64 {
        if self.total_requests == 0 {
            return 0.0;
        }
        self.errors as f64 / self.total_requests as f64
    }

    /// Is this result reliable enough for winner selection?
    #[must_use]
    pub fn is_reliable(&self) -> bool {
        self.error_rate() < 0.01 && self.oom_count == 0 && self.panics == 0
    }
}

/// System information for benchmark context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemInfo {
    pub cpu_model: String,
    pub cores: usize,
    pub ram_gb: f64,
    #[serde(default)]
    pub avx_features: Vec<String>,
}

impl Default for SystemInfo {
    fn default() -> Self {
        Self {
            cpu_model: "unknown".into(),
            cores: std::thread::available_parallelism()
                .map(std::num::NonZero::get)
                .unwrap_or(1),
            ram_gb: 0.0,
            avx_features: vec![],
        }
    }
}

/// Full benchmark report with multiple model results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    /// Timestamp of benchmark run.
    pub timestamp: String,
    /// System information.
    pub system: SystemInfo,
    /// Results for each model.
    pub results: Vec<FullBenchmarkResult>,
}

/// Evaluate embedder quality against a labeled corpus.
///
/// Computes NDCG@10, MRR@10, MAP@10, Recall@100, Precision@10 by:
/// 1. Embedding all corpus documents
/// 2. For each query, compute cosine similarity to all docs
/// 3. Rank by similarity and compare to ground truth relevance
#[allow(clippy::cast_precision_loss)]
pub fn evaluate_embedder_quality(
    embedder: &dyn Embedder,
    corpus: &BenchmarkCorpus,
) -> Result<QualityMetrics> {
    if corpus.queries.is_empty() {
        return Ok(QualityMetrics::default());
    }

    // Embed all corpus documents
    let doc_texts: Vec<&str> = corpus.corpus.iter().map(|d| d.text.as_str()).collect();
    let doc_embeddings = embedder.embed_batch(&doc_texts)?;

    // Build doc_id -> index map (used for relevance lookup)
    let _doc_id_to_idx: HashMap<&str, usize> = corpus
        .corpus
        .iter()
        .enumerate()
        .map(|(i, d)| (d.id.as_str(), i))
        .collect();

    let mut ndcg_scores = Vec::with_capacity(corpus.queries.len());
    let mut mrr_scores = Vec::with_capacity(corpus.queries.len());
    let mut map_scores = Vec::with_capacity(corpus.queries.len());
    let mut recall_scores = Vec::with_capacity(corpus.queries.len());
    let mut precision_scores = Vec::with_capacity(corpus.queries.len());

    for query in &corpus.queries {
        // Embed query
        let query_emb = embedder.embed_batch(&[query.text.as_str()])?;
        let query_vec = &query_emb[0];

        // Compute cosine similarities (embeddings should be normalized)
        let mut scores: Vec<(usize, f32)> = doc_embeddings
            .iter()
            .enumerate()
            .map(|(i, doc_emb)| {
                let sim = dot_product(query_vec, doc_emb);
                (i, sim)
            })
            .collect();

        // Sort by similarity descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Map ranked results to relevance grades
        let ranked_rels: Vec<u8> = scores
            .iter()
            .take(100) // For recall@100
            .map(|(idx, _)| {
                let doc_id = &corpus.corpus[*idx].id;
                query.relevants.get(doc_id).copied().unwrap_or(0)
            })
            .collect();

        let total_relevant = corpus.total_relevant(query);

        ndcg_scores.push(ndcg_at_k(&ranked_rels, 10));
        mrr_scores.push(compute_mrr(&ranked_rels));
        map_scores.push(map_at_k(&ranked_rels, 10));
        recall_scores.push(recall_at_k(&ranked_rels, 100, total_relevant));
        precision_scores.push(precision_at_k(&ranked_rels, 10));
    }

    Ok(QualityMetrics {
        ndcg_at_10: mean(&ndcg_scores),
        mrr_at_10: mean(&mrr_scores),
        map_at_10: mean(&map_scores),
        recall_at_100: mean(&recall_scores),
        precision_at_10: mean(&precision_scores),
    })
}

/// Compute dot product between two embedding vectors.
#[inline]
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Compute mean of a slice.
#[inline]
#[allow(clippy::cast_precision_loss)]
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Run an embedding benchmark using a factory to capture cold start time.
#[allow(clippy::cast_precision_loss)]
pub fn run_embedding_benchmark<F>(
    factory: F,
    corpus: &BenchmarkCorpus,
    config: &BenchmarkConfig,
) -> Result<BenchmarkResult>
where
    F: FnOnce() -> Result<Box<dyn Embedder>>,
{
    let cold_start_begin = Instant::now();
    let embedder = factory()?;
    let cold_start_ms = cold_start_begin.elapsed().as_secs_f64() * 1000.0;

    let texts: Vec<&str> = corpus
        .corpus
        .iter()
        .take(config.batch_size)
        .map(|doc| doc.text.as_str())
        .collect();

    // Time-to-first-result (single batch)
    let ttfr_begin = Instant::now();
    let _ = embedder.embed_batch(&texts)?;
    let ttfr_ms = ttfr_begin.elapsed().as_secs_f64() * 1000.0;

    // Warmup phase
    for _ in 0..config.warmup_iters {
        let _ = embedder.embed_batch(&texts)?;
    }

    // Measurement phase
    let mut latencies = Vec::with_capacity(config.measure_iters);
    let measure_start = Instant::now();
    for _ in 0..config.measure_iters {
        let iter_start = Instant::now();
        let _ = embedder.embed_batch(&texts)?;
        let elapsed = iter_start.elapsed().as_secs_f64() * 1000.0;
        latencies.push(elapsed);
    }
    let total_elapsed = measure_start.elapsed();

    let warm_p50 = percentile(&latencies, 0.5).unwrap_or(0.0);
    let warm_p95 = percentile(&latencies, 0.95).unwrap_or(0.0);
    let warm_p99 = percentile(&latencies, 0.99).unwrap_or(0.0);
    let warm_max = latencies
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
        .unwrap_or(0.0);
    let jitter = coefficient_of_variation(&latencies);
    let total_docs = (config.measure_iters * config.batch_size) as f64;
    let throughput_docs_per_sec = if total_elapsed > Duration::ZERO {
        total_docs / total_elapsed.as_secs_f64()
    } else {
        0.0
    };

    let metadata = BenchmarkMetadata {
        model: embedder.model_name().to_string(),
        dimension: embedder.dimension(),
        batch_size: config.batch_size,
        warmup_iters: config.warmup_iters,
        measure_iters: config.measure_iters,
    };

    let speed = SpeedMetrics {
        cold_start_ms,
        warm_latency_p50_ms: warm_p50,
        warm_latency_p95_ms: warm_p95,
        warm_latency_p99_ms: warm_p99,
        warm_latency_max_ms: warm_max,
        throughput_docs_per_sec,
        time_to_first_result_ms: ttfr_ms,
        jitter_cv: jitter,
    };

    Ok(BenchmarkResult {
        metadata,
        speed,
        latency_ms: latencies,
    })
}

/// Run a full benchmark producing a `FullBenchmarkResult`.
///
/// This includes:
/// 1. Cold start and TTFR measurement
/// 2. Warmup phase (discarded)
/// 3. Measurement phase with percentile tracking
/// 4. Quality evaluation against corpus (if enabled)
/// 5. MRL sweep (if model supports MRL)
#[allow(clippy::cast_precision_loss, clippy::too_many_lines)]
pub fn run_full_benchmark<F>(
    factory: F,
    corpus: &BenchmarkCorpus,
    config: &BenchmarkConfig,
) -> Result<FullBenchmarkResult>
where
    F: FnOnce() -> Result<Box<dyn Embedder>>,
{
    // 1. Cold start phase
    let cold_start_begin = Instant::now();
    let embedder = factory()?;
    let cold_start_ms = cold_start_begin.elapsed().as_secs_f64() * 1000.0;

    let model_name = embedder.model_name().to_string();
    let category = embedder.category();
    let supports_mrl = embedder.supports_mrl();

    let texts: Vec<&str> = corpus
        .corpus
        .iter()
        .take(config.batch_size)
        .map(|doc| doc.text.as_str())
        .collect();

    // Time-to-first-result
    let ttfr_begin = Instant::now();
    let _ = embedder.embed_batch(&texts)?;
    let ttfr_ms = ttfr_begin.elapsed().as_secs_f64() * 1000.0;

    // 2. Warmup phase (discard results)
    for _ in 0..config.warmup_iters {
        let _ = embedder.embed_batch(&texts)?;
    }

    // 3. Measurement phase
    let mut latencies = Vec::with_capacity(config.measure_iters);
    let mut errors = 0usize;
    let mut oom_count = 0usize;
    let measure_start = Instant::now();

    for _ in 0..config.measure_iters {
        let iter_start = Instant::now();
        match embedder.embed_batch(&texts) {
            Ok(_) => {
                let elapsed = iter_start.elapsed().as_secs_f64() * 1000.0;
                latencies.push(elapsed);
            }
            Err(e) => {
                errors += 1;
                let err_str = e.to_string().to_lowercase();
                if err_str.contains("oom") || err_str.contains("out of memory") {
                    oom_count += 1;
                }
            }
        }
    }
    let total_elapsed = measure_start.elapsed();

    // Compute speed metrics
    let warm_p50 = percentile(&latencies, 0.5).unwrap_or(0.0);
    let warm_p95 = percentile(&latencies, 0.95).unwrap_or(0.0);
    let warm_p99 = percentile(&latencies, 0.99).unwrap_or(0.0);
    let warm_max = latencies
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .copied()
        .unwrap_or(0.0);
    let jitter = coefficient_of_variation(&latencies);

    let successful_iters = config.measure_iters - errors;
    let total_docs = (successful_iters * config.batch_size) as f64;
    let throughput_docs_per_sec = if total_elapsed > Duration::ZERO && successful_iters > 0 {
        total_docs / total_elapsed.as_secs_f64()
    } else {
        0.0
    };

    let speed = SpeedMetrics {
        cold_start_ms,
        warm_latency_p50_ms: warm_p50,
        warm_latency_p95_ms: warm_p95,
        warm_latency_p99_ms: warm_p99,
        warm_latency_max_ms: warm_max,
        throughput_docs_per_sec,
        time_to_first_result_ms: ttfr_ms,
        jitter_cv: jitter,
    };

    // 4. Quality evaluation
    let quality = if config.eval_quality && !corpus.queries.is_empty() {
        evaluate_embedder_quality(embedder.as_ref(), corpus)?
    } else {
        QualityMetrics::default()
    };

    // 5. MRL sweep (if supported)
    // For now, return empty - actual MRL evaluation would require
    // dimension-aware embedding which needs model-specific support
    let mrl_dims_tested: Vec<usize> = vec![];
    let mrl_quality_at_dims: Vec<(usize, f64)> = vec![];
    let _ = (supports_mrl, &config.mrl_dims); // Suppress unused warnings

    // Resource metrics (basic for now)
    let peak_threads = rayon::current_num_threads();

    Ok(FullBenchmarkResult {
        model_name,
        category,
        eligible: true, // TODO: Check against candidates list
        release_date: None,
        quality,
        speed,
        total_requests: config.measure_iters,
        errors,
        oom_count,
        panics: 0,
        peak_rss_mb: 0.0,        // TODO: Implement RSS measurement
        steady_rss_mb: 0.0,      // TODO: Implement RSS measurement
        model_file_size_mb: 0.0, // TODO: Implement model size measurement
        peak_threads,
        mrl_dims_tested,
        mrl_quality_at_dims,
        reranker_lift: None,
        latency_ms: latencies,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::benchmark::datasets::{CorpusDoc, CorpusStats, CrossValidationSplits, LabeledQuery};
    use crate::hash_embedder::HashEmbedder;

    fn make_test_corpus() -> BenchmarkCorpus {
        BenchmarkCorpus {
            corpus_version: Some("test-v1".into()),
            generation_seed: Some(42),
            generated_at: None,
            corpus: vec![
                CorpusDoc {
                    id: "doc1".into(),
                    text: "hello world rust programming".into(),
                    doc_type: "tweet".into(),
                    metadata: serde_json::Value::Null,
                },
                CorpusDoc {
                    id: "doc2".into(),
                    text: "another document about testing".into(),
                    doc_type: "tweet".into(),
                    metadata: serde_json::Value::Null,
                },
                CorpusDoc {
                    id: "doc3".into(),
                    text: "rust is great for systems programming".into(),
                    doc_type: "tweet".into(),
                    metadata: serde_json::Value::Null,
                },
            ],
            queries: vec![LabeledQuery {
                id: "q1".into(),
                text: "rust programming".into(),
                relevants: [("doc1".into(), 2), ("doc3".into(), 2)]
                    .into_iter()
                    .collect(),
                query_type: None,
                difficulty: None,
                source_doc_id: None,
                split: None,
                category: None,
            }],
            splits: CrossValidationSplits::default(),
            stats: CorpusStats::default(),
        }
    }

    #[test]
    fn test_benchmark_runs() {
        let corpus = make_test_corpus();
        let config = BenchmarkConfig {
            warmup_iters: 1,
            measure_iters: 2,
            batch_size: 2,
            ..Default::default()
        };

        let result =
            run_embedding_benchmark(|| Ok(Box::new(HashEmbedder::default())), &corpus, &config)
                .expect("benchmark failed");

        assert!(result.speed.warm_latency_p50_ms >= 0.0);
        assert!(result.speed.throughput_docs_per_sec >= 0.0);
        assert_eq!(result.metadata.batch_size, 2);
    }

    #[test]
    fn test_full_benchmark_runs() {
        let corpus = make_test_corpus();
        let config = BenchmarkConfig {
            warmup_iters: 1,
            measure_iters: 2,
            batch_size: 2,
            eval_quality: false, // Skip quality eval for speed
            ..Default::default()
        };

        let result = run_full_benchmark(|| Ok(Box::new(HashEmbedder::default())), &corpus, &config)
            .expect("benchmark failed");

        assert!(result.speed.warm_latency_p50_ms >= 0.0);
        assert!(result.speed.throughput_docs_per_sec >= 0.0);
        assert_eq!(result.total_requests, 2);
        assert_eq!(result.errors, 0);
        assert!(result.is_reliable());
    }

    #[test]
    fn test_full_benchmark_result_error_rate() {
        let result = FullBenchmarkResult {
            model_name: "test".into(),
            category: ModelCategory::StaticEmbedder,
            eligible: true,
            release_date: None,
            quality: QualityMetrics::default(),
            speed: SpeedMetrics::default(),
            total_requests: 100,
            errors: 5,
            oom_count: 0,
            panics: 0,
            peak_rss_mb: 0.0,
            steady_rss_mb: 0.0,
            model_file_size_mb: 0.0,
            peak_threads: 1,
            mrl_dims_tested: vec![],
            mrl_quality_at_dims: vec![],
            reranker_lift: None,
            latency_ms: vec![],
        };

        assert!((result.error_rate() - 0.05).abs() < 1e-6);
        assert!(!result.is_reliable()); // 5% error rate > 1% threshold
    }

    #[test]
    fn test_full_benchmark_result_is_reliable() {
        let mut result = FullBenchmarkResult {
            model_name: "test".into(),
            category: ModelCategory::StaticEmbedder,
            eligible: true,
            release_date: None,
            quality: QualityMetrics::default(),
            speed: SpeedMetrics::default(),
            total_requests: 100,
            errors: 0,
            oom_count: 0,
            panics: 0,
            peak_rss_mb: 0.0,
            steady_rss_mb: 0.0,
            model_file_size_mb: 0.0,
            peak_threads: 1,
            mrl_dims_tested: vec![],
            mrl_quality_at_dims: vec![],
            reranker_lift: None,
            latency_ms: vec![],
        };

        assert!(result.is_reliable());

        result.oom_count = 1;
        assert!(!result.is_reliable());

        result.oom_count = 0;
        result.panics = 1;
        assert!(!result.is_reliable());
    }

    #[test]
    fn test_evaluate_quality_empty_queries() {
        let corpus = BenchmarkCorpus {
            corpus_version: None,
            generation_seed: None,
            generated_at: None,
            corpus: vec![CorpusDoc {
                id: "doc1".into(),
                text: "test".into(),
                doc_type: "tweet".into(),
                metadata: serde_json::Value::Null,
            }],
            queries: vec![], // Empty queries
            splits: CrossValidationSplits::default(),
            stats: CorpusStats::default(),
        };

        let embedder = HashEmbedder::default();
        let quality = evaluate_embedder_quality(&embedder, &corpus).unwrap();

        assert!((quality.ndcg_at_10 - 0.0).abs() < f64::EPSILON);
        assert!((quality.mrr_at_10 - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert!((result - 32.0).abs() < 1e-6); // 1*4 + 2*5 + 3*6 = 32
    }

    #[test]
    fn test_mean() {
        assert!((mean(&[1.0, 2.0, 3.0, 4.0, 5.0]) - 3.0).abs() < f64::EPSILON);
        assert!((mean(&[]) - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_benchmark_config_default() {
        let config = BenchmarkConfig::default();
        assert_eq!(config.warmup_iters, 10);
        assert_eq!(config.measure_iters, 100);
        assert_eq!(config.batch_size, 32);
        assert!(config.eval_quality);
        assert!(!config.mrl_dims.is_empty());
    }

    #[test]
    fn test_system_info_default() {
        let info = SystemInfo::default();
        assert_eq!(info.cpu_model, "unknown");
        assert!(info.cores > 0);
    }
}
