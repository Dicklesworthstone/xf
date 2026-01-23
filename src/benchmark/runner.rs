//! Benchmark runner for embedding and reranker models.

use std::time::{Duration, Instant};

use anyhow::Result;
use serde::Serialize;

use crate::benchmark::datasets::BenchmarkCorpus;
use crate::benchmark::metrics::{percentile, SpeedMetrics};
use crate::embedder::Embedder;

/// Configuration for a benchmark run.
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub warmup_iters: usize,
    pub measure_iters: usize,
    pub batch_size: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            warmup_iters: 10,
            measure_iters: 100,
            batch_size: 32,
        }
    }
}

/// Benchmark metadata.
#[derive(Debug, Clone, Serialize)]
pub struct BenchmarkMetadata {
    pub model: String,
    pub dimension: usize,
    pub batch_size: usize,
    pub warmup_iters: usize,
    pub measure_iters: usize,
}

/// Benchmark result for embedding throughput and latency.
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub metadata: BenchmarkMetadata,
    pub speed: SpeedMetrics,
    pub latency_ms: Vec<f64>,
}

/// Run an embedding benchmark using a factory to capture cold start time.
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
        throughput_docs_per_sec,
        time_to_first_result_ms: ttfr_ms,
    };

    Ok(BenchmarkResult {
        metadata,
        speed,
        latency_ms: latencies,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hash_embedder::HashEmbedder;

    #[test]
    fn test_benchmark_runs() {
        let corpus = BenchmarkCorpus {
            corpus: vec![
                crate::benchmark::datasets::CorpusDoc {
                    id: "doc1".into(),
                    text: "hello world".into(),
                    doc_type: "tweet".into(),
                    metadata: serde_json::Value::Null,
                },
                crate::benchmark::datasets::CorpusDoc {
                    id: "doc2".into(),
                    text: "another doc".into(),
                    doc_type: "tweet".into(),
                    metadata: serde_json::Value::Null,
                },
            ],
            queries: vec![],
        };

        let config = BenchmarkConfig {
            warmup_iters: 1,
            measure_iters: 2,
            batch_size: 2,
        };

        let result = run_embedding_benchmark(
            || Ok(Box::new(HashEmbedder::default())),
            &corpus,
            &config,
        )
        .expect("benchmark failed");

        assert!(result.speed.warm_latency_p50_ms >= 0.0);
        assert!(result.speed.throughput_docs_per_sec >= 0.0);
        assert_eq!(result.metadata.batch_size, 2);
    }
}
