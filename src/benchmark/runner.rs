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
            cores: std::thread::available_parallelism().map_or(1, std::num::NonZero::get),
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

/// MRL dimension sweep output: `(dims_tested, quality_at_dims)`.
pub type MrlSweepResult = (Vec<usize>, Vec<(usize, f64)>);

/// Evaluate MRL NDCG@10 across specified target dimensions.
#[allow(clippy::cast_precision_loss)]
pub fn evaluate_mrl_quality_sweep(
    embedder: &dyn Embedder,
    corpus: &BenchmarkCorpus,
    mrl_dims: &[usize],
) -> Result<MrlSweepResult> {
    if !embedder.supports_mrl() || corpus.queries.is_empty() || mrl_dims.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let max_dim = embedder.dimension();
    let valid_dims: Vec<usize> = mrl_dims
        .iter()
        .copied()
        .filter(|&d| d > 0 && d <= max_dim)
        .collect();
    if valid_dims.is_empty() {
        return Ok((Vec::new(), Vec::new()));
    }

    let doc_texts: Vec<&str> = corpus.corpus.iter().map(|d| d.text.as_str()).collect();
    let full_doc_embeddings = embedder.embed_batch(&doc_texts)?;
    let query_texts: Vec<&str> = corpus.queries.iter().map(|q| q.text.as_str()).collect();
    let full_query_embeddings = embedder.embed_batch(&query_texts)?;

    let mut dims_tested = Vec::with_capacity(valid_dims.len());
    let mut quality_at_dims = Vec::with_capacity(valid_dims.len());

    for dim in valid_dims {
        let Ok(mut trunc_docs) = embedder.truncate_batch(&full_doc_embeddings, dim) else {
            continue;
        };
        let Ok(mut trunc_queries) = embedder.truncate_batch(&full_query_embeddings, dim) else {
            continue;
        };
        for v in &mut trunc_docs {
            crate::embedder::l2_normalize(v);
        }
        for v in &mut trunc_queries {
            crate::embedder::l2_normalize(v);
        }

        let mut ndcg_scores = Vec::with_capacity(corpus.queries.len());
        for (q_idx, query) in corpus.queries.iter().enumerate() {
            let query_vec = &trunc_queries[q_idx];
            let mut scores: Vec<(usize, f32)> = trunc_docs
                .iter()
                .enumerate()
                .map(|(i, doc_emb)| (i, dot_product(query_vec, doc_emb)))
                .collect();
            scores.sort_by(|a, b| b.1.total_cmp(&a.1));
            let ranked_rels: Vec<u8> = scores
                .iter()
                .take(10)
                .map(|(idx, _)| {
                    let doc_id = &corpus.corpus[*idx].id;
                    query.relevants.get(doc_id).copied().unwrap_or(0)
                })
                .collect();
            ndcg_scores.push(ndcg_at_k(&ranked_rels, 10));
        }

        dims_tested.push(dim);
        quality_at_dims.push((dim, mean(&ndcg_scores)));
    }

    Ok((dims_tested, quality_at_dims))
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
    let (mrl_dims_tested, mrl_quality_at_dims) =
        if supports_mrl && config.eval_quality && !corpus.queries.is_empty() {
            evaluate_mrl_quality_sweep(embedder.as_ref(), corpus, &config.mrl_dims)?
        } else {
            (Vec::new(), Vec::new())
        };

    // Resource and model registry metrics
    let peak_threads = rayon::current_num_threads();
    let (peak_rss_mb, steady_rss_mb) = measure_process_rss_mb();
    let registry = crate::model_registry::ModelRegistry::new();
    let canonical_name = crate::model_registry::ModelRegistry::canonical_embedder_name(&model_name);
    let eligible = canonical_name.is_some();
    let model_file_size_mb = canonical_name
        .and_then(|canon| registry.list_models().into_iter().find(|m| m.name == canon))
        .and_then(|m| m.size_mb)
        .unwrap_or(0.0);

    Ok(FullBenchmarkResult {
        model_name,
        category,
        eligible,
        release_date: None,
        quality,
        speed,
        total_requests: config.measure_iters,
        errors,
        oom_count,
        panics: 0,
        peak_rss_mb,
        steady_rss_mb,
        model_file_size_mb,
        peak_threads,
        mrl_dims_tested,
        mrl_quality_at_dims,
        reranker_lift: None,
        latency_ms: latencies,
    })
}

/// Parse `(peak_rss_mb, steady_rss_mb)` (`VmHWM` and `VmRSS`) from `/proc/self/status` text.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn parse_proc_status_rss_mb(status_text: &str) -> (f64, f64) {
    let mut vm_hwm_kib = None;
    let mut vm_rss_kib = None;
    for line in status_text.lines() {
        if let Some(rest) = line.strip_prefix("VmHWM:") {
            if let Some(val_str) = rest.split_whitespace().next() {
                if let Ok(kb) = val_str.parse::<u64>() {
                    vm_hwm_kib = Some(kb);
                }
            }
        } else if let Some(rest) = line.strip_prefix("VmRSS:") {
            if let Some(val_str) = rest.split_whitespace().next() {
                if let Ok(kb) = val_str.parse::<u64>() {
                    vm_rss_kib = Some(kb);
                }
            }
        }
    }
    let steady_mb = vm_rss_kib.map_or(0.0, |kb| kb as f64 / 1024.0);
    let peak_mb = vm_hwm_kib.map_or(steady_mb, |kb| (kb as f64 / 1024.0).max(steady_mb));
    (peak_mb, steady_mb)
}

fn measure_process_rss_mb() -> (f64, f64) {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            return parse_proc_status_rss_mb(&status);
        }
    }
    (0.0, 0.0)
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
        assert_ne!(config.mrl_dims, Vec::<usize>::new());
    }

    #[test]
    fn test_system_info_default() {
        let info = SystemInfo::default();
        assert_eq!(info.cpu_model, "unknown");
        assert!(info.cores > 0);
    }

    #[test]
    fn test_parse_proc_status_rss_mb() {
        let sample = "\
Name:\txf
VmPeak:\t  524288 kB
VmSize:\t  512000 kB
VmHWM:\t  131072 kB
VmRSS:\t   65536 kB
Threads:\t8
";
        let (peak_mb, steady_mb) = parse_proc_status_rss_mb(sample);
        assert!((peak_mb - 128.0).abs() < 1e-6);
        assert!((steady_mb - 64.0).abs() < 1e-6);

        let empty = parse_proc_status_rss_mb("");
        assert!((empty.0 - 0.0).abs() < 1e-6);
        assert!((empty.1 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_full_benchmark_populates_rss_and_eligibility() {
        let corpus = make_test_corpus();
        let config = BenchmarkConfig {
            warmup_iters: 1,
            measure_iters: 1,
            batch_size: 2,
            eval_quality: false,
            ..Default::default()
        };
        let result = run_full_benchmark(|| Ok(Box::new(HashEmbedder::default())), &corpus, &config)
            .expect("full benchmark failed");
        assert!(result.eligible);
        #[cfg(target_os = "linux")]
        {
            assert!(result.steady_rss_mb > 0.0);
            assert!(result.peak_rss_mb >= result.steady_rss_mb);
        }
    }

    #[test]
    fn test_full_benchmark_mrl_sweep() {
        struct MockMrlEmbedder(HashEmbedder);
        impl Embedder for MockMrlEmbedder {
            fn embed(&self, text: &str) -> crate::embedder::EmbedderResult<Vec<f32>> {
                self.0.embed(text)
            }
            fn dimension(&self) -> usize {
                self.0.dimension()
            }
            fn id(&self) -> &'static str {
                "potion-multilingual-128M"
            }
            fn is_semantic(&self) -> bool {
                false
            }
            fn supports_mrl(&self) -> bool {
                true
            }
        }

        let corpus = make_test_corpus();
        let config = BenchmarkConfig {
            warmup_iters: 1,
            measure_iters: 1,
            batch_size: 2,
            mrl_dims: vec![64, 128, 384, 1024],
            eval_quality: true,
        };
        let result = run_full_benchmark(
            || Ok(Box::new(MockMrlEmbedder(HashEmbedder::default()))),
            &corpus,
            &config,
        )
        .expect("full benchmark with MRL sweep failed");

        assert_eq!(result.mrl_dims_tested, vec![64, 128, 384]);
        assert_eq!(result.mrl_quality_at_dims.len(), 3);
        for (dim, ndcg) in &result.mrl_quality_at_dims {
            assert!([64, 128, 384].contains(dim));
            assert!((0.0..=1.0).contains(ndcg));
        }
    }
}
