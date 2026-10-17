//! Reporters for benchmark results.

use std::fmt::Write as FmtWrite;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use anyhow::Result;
use serde::Serialize;

use crate::benchmark::metrics::{QualityMetrics, ReliabilityMetrics, SpeedMetrics};
use crate::benchmark::runner::{BenchmarkMetadata, BenchmarkResult, FullBenchmarkResult};

/// Serializable report wrapper.
#[derive(Debug, Serialize)]
pub struct BenchmarkReport<'a> {
    pub metadata: &'a BenchmarkMetadata,
    pub speed: &'a SpeedMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<&'a QualityMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reliability: Option<ReliabilityMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peak_rss_mb: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub steady_rss_mb: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub model_file_size_mb: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eligible: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mrl_dims_tested: Option<&'a [usize]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mrl_quality_at_dims: Option<&'a [(usize, f64)]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reranker_lift: Option<f64>,
    pub latency_ms: &'a [f64],
}

/// Write JSON report.
pub fn write_json(path: &Path, result: &BenchmarkResult) -> Result<()> {
    let report = BenchmarkReport {
        metadata: &result.metadata,
        speed: &result.speed,
        quality: None,
        reliability: None,
        peak_rss_mb: None,
        steady_rss_mb: None,
        model_file_size_mb: None,
        eligible: None,
        mrl_dims_tested: None,
        mrl_quality_at_dims: None,
        reranker_lift: None,
        latency_ms: &result.latency_ms,
    };
    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Write full JSON report including quality, reliability, RSS, MRL, and reranker lift.
pub fn write_full_json(
    path: &Path,
    metadata: &BenchmarkMetadata,
    result: &FullBenchmarkResult,
) -> Result<()> {
    let reliability = ReliabilityMetrics {
        error_rate: result.error_rate(),
        oom_threshold_batch: if result.oom_count == 0 {
            metadata.batch_size
        } else {
            0
        },
        fallback_rate: 0.0,
    };
    let report = BenchmarkReport {
        metadata,
        speed: &result.speed,
        quality: Some(&result.quality),
        reliability: Some(reliability),
        peak_rss_mb: Some(result.peak_rss_mb),
        steady_rss_mb: Some(result.steady_rss_mb),
        model_file_size_mb: Some(result.model_file_size_mb),
        eligible: Some(result.eligible),
        mrl_dims_tested: if result.mrl_dims_tested.is_empty() {
            None
        } else {
            Some(&result.mrl_dims_tested)
        },
        mrl_quality_at_dims: if result.mrl_quality_at_dims.is_empty() {
            None
        } else {
            Some(&result.mrl_quality_at_dims)
        },
        reranker_lift: result.reranker_lift,
        latency_ms: &result.latency_ms,
    };
    let json = serde_json::to_string_pretty(&report)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Write Markdown report.
pub fn write_markdown(path: &Path, result: &BenchmarkResult) -> Result<()> {
    let mut out = String::new();
    out.push_str("# Benchmark Report\n\n");
    let _ = writeln!(out, "Model: `{}`\n", result.metadata.model);
    out.push_str("| Metric | Value |\n|---|---|\n");
    let _ = writeln!(
        out,
        "| Cold start (ms) | {:.2} |",
        result.speed.cold_start_ms
    );
    let _ = writeln!(
        out,
        "| TTFR (ms) | {:.2} |",
        result.speed.time_to_first_result_ms
    );
    let _ = writeln!(
        out,
        "| Warm p50 (ms) | {:.2} |",
        result.speed.warm_latency_p50_ms
    );
    let _ = writeln!(
        out,
        "| Warm p95 (ms) | {:.2} |",
        result.speed.warm_latency_p95_ms
    );
    let _ = writeln!(
        out,
        "| Warm p99 (ms) | {:.2} |",
        result.speed.warm_latency_p99_ms
    );
    let _ = writeln!(
        out,
        "| Throughput (docs/sec) | {:.2} |",
        result.speed.throughput_docs_per_sec
    );
    std::fs::write(path, out)?;
    Ok(())
}

/// Write full Markdown report including quality, reliability, RSS, MRL, and reranker lift.
pub fn write_full_markdown(
    path: &Path,
    metadata: &BenchmarkMetadata,
    result: &FullBenchmarkResult,
) -> Result<()> {
    let mut out = String::new();
    out.push_str("# Benchmark Report\n\n");
    let _ = writeln!(
        out,
        "Model: `{}` (dim: {}, eligible: {})\n",
        metadata.model, metadata.dimension, result.eligible
    );
    out.push_str("## Speed & Resource Metrics\n\n");
    out.push_str("| Metric | Value |\n|---|---|\n");
    let _ = writeln!(
        out,
        "| Cold start (ms) | {:.2} |",
        result.speed.cold_start_ms
    );
    let _ = writeln!(
        out,
        "| TTFR (ms) | {:.2} |",
        result.speed.time_to_first_result_ms
    );
    let _ = writeln!(
        out,
        "| Warm p50 (ms) | {:.2} |",
        result.speed.warm_latency_p50_ms
    );
    let _ = writeln!(
        out,
        "| Warm p95 (ms) | {:.2} |",
        result.speed.warm_latency_p95_ms
    );
    let _ = writeln!(
        out,
        "| Warm p99 (ms) | {:.2} |",
        result.speed.warm_latency_p99_ms
    );
    let _ = writeln!(
        out,
        "| Throughput (docs/sec) | {:.2} |",
        result.speed.throughput_docs_per_sec
    );
    let _ = writeln!(out, "| Peak RSS (MB) | {:.2} |", result.peak_rss_mb);
    let _ = writeln!(out, "| Steady RSS (MB) | {:.2} |", result.steady_rss_mb);
    let _ = writeln!(
        out,
        "| Model Size (MB) | {:.2} |",
        result.model_file_size_mb
    );
    let _ = writeln!(out, "| Error Rate | {:.4} |", result.error_rate());

    out.push_str("\n## Quality Metrics\n\n");
    out.push_str("| Metric | Value |\n|---|---|\n");
    let _ = writeln!(out, "| NDCG@10 | {:.4} |", result.quality.ndcg_at_10);
    let _ = writeln!(out, "| MRR@10 | {:.4} |", result.quality.mrr_at_10);
    let _ = writeln!(out, "| MAP@10 | {:.4} |", result.quality.map_at_10);
    let _ = writeln!(out, "| Recall@100 | {:.4} |", result.quality.recall_at_100);
    let _ = writeln!(
        out,
        "| Precision@10 | {:.4} |",
        result.quality.precision_at_10
    );
    if let Some(lift) = result.reranker_lift {
        let _ = writeln!(out, "| Reranker NDCG@10 Lift (%) | {lift:+.2}% |");
    }

    if !result.mrl_quality_at_dims.is_empty() {
        out.push_str("\n## MRL Dimension Sweep\n\n");
        out.push_str("| Dimension | NDCG@10 |\n|---|---|\n");
        for (dim, ndcg) in &result.mrl_quality_at_dims {
            let _ = writeln!(out, "| {dim} | {ndcg:.4} |");
        }
    }

    std::fs::write(path, out)?;
    Ok(())
}

/// Write CSV report.
pub fn write_csv(path: &Path, result: &BenchmarkResult) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "model,dimension,batch_size,warmup_iters,measure_iters,cold_start_ms,ttfr_ms,warm_p50_ms,warm_p95_ms,warm_p99_ms,throughput_docs_per_sec"
    )?;
    writeln!(
        writer,
        "{},{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2}",
        result.metadata.model,
        result.metadata.dimension,
        result.metadata.batch_size,
        result.metadata.warmup_iters,
        result.metadata.measure_iters,
        result.speed.cold_start_ms,
        result.speed.time_to_first_result_ms,
        result.speed.warm_latency_p50_ms,
        result.speed.warm_latency_p95_ms,
        result.speed.warm_latency_p99_ms,
        result.speed.throughput_docs_per_sec
    )?;
    writer.flush()?;
    Ok(())
}

/// Write full CSV report including quality, RSS, and reranker lift columns.
pub fn write_full_csv(
    path: &Path,
    metadata: &BenchmarkMetadata,
    result: &FullBenchmarkResult,
) -> Result<()> {
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "model,dimension,batch_size,warmup_iters,measure_iters,cold_start_ms,ttfr_ms,warm_p50_ms,warm_p95_ms,warm_p99_ms,throughput_docs_per_sec,ndcg_at_10,mrr_at_10,map_at_10,recall_at_100,precision_at_10,peak_rss_mb,steady_rss_mb,reranker_lift_pct"
    )?;
    let lift_str = result
        .reranker_lift
        .map_or_else(String::new, |v| format!("{v:.2}"));
    writeln!(
        writer,
        "{},{},{},{},{},{:.2},{:.2},{:.2},{:.2},{:.2},{:.2},{:.4},{:.4},{:.4},{:.4},{:.4},{:.2},{:.2},{}",
        metadata.model,
        metadata.dimension,
        metadata.batch_size,
        metadata.warmup_iters,
        metadata.measure_iters,
        result.speed.cold_start_ms,
        result.speed.time_to_first_result_ms,
        result.speed.warm_latency_p50_ms,
        result.speed.warm_latency_p95_ms,
        result.speed.warm_latency_p99_ms,
        result.speed.throughput_docs_per_sec,
        result.quality.ndcg_at_10,
        result.quality.mrr_at_10,
        result.quality.map_at_10,
        result.quality.recall_at_100,
        result.quality.precision_at_10,
        result.peak_rss_mb,
        result.steady_rss_mb,
        lift_str
    )?;
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::ModelCategory;

    #[test]
    fn test_write_and_write_full_reporters() {
        let dir = tempfile::tempdir().unwrap();
        let metadata = BenchmarkMetadata {
            model: "hash-fnv1a-384".to_string(),
            dimension: 384,
            batch_size: 16,
            warmup_iters: 2,
            measure_iters: 5,
        };
        let speed = SpeedMetrics {
            cold_start_ms: 1.2,
            warm_latency_p50_ms: 0.5,
            warm_latency_p95_ms: 0.8,
            warm_latency_p99_ms: 0.9,
            warm_latency_max_ms: 1.0,
            throughput_docs_per_sec: 32000.0,
            time_to_first_result_ms: 1.5,
            jitter_cv: 0.05,
        };
        let simple = BenchmarkResult {
            metadata: metadata.clone(),
            speed,
            latency_ms: vec![0.5, 0.6],
        };

        let simple_json = dir.path().join("simple.json");
        let simple_md = dir.path().join("simple.md");
        let simple_csv = dir.path().join("simple.csv");
        write_json(&simple_json, &simple).unwrap();
        write_markdown(&simple_md, &simple).unwrap();
        write_csv(&simple_csv, &simple).unwrap();
        assert!(simple_json.exists());
        assert!(simple_md.exists());
        assert!(simple_csv.exists());

        let full = FullBenchmarkResult {
            model_name: "hash-fnv1a-384".to_string(),
            category: ModelCategory::StaticEmbedder,
            eligible: true,
            release_date: None,
            quality: QualityMetrics {
                ndcg_at_10: 0.75,
                mrr_at_10: 0.80,
                map_at_10: 0.70,
                recall_at_100: 0.95,
                precision_at_10: 0.50,
            },
            speed,
            total_requests: 5,
            errors: 0,
            oom_count: 0,
            panics: 0,
            peak_rss_mb: 64.0,
            steady_rss_mb: 48.0,
            model_file_size_mb: 0.0,
            peak_threads: 4,
            mrl_dims_tested: vec![128, 256],
            mrl_quality_at_dims: vec![(128, 0.70), (256, 0.74)],
            reranker_lift: Some(4.5),
            latency_ms: vec![0.5, 0.6],
        };

        let full_json = dir.path().join("full.json");
        let full_md = dir.path().join("full.md");
        let full_csv = dir.path().join("full.csv");
        write_full_json(&full_json, &metadata, &full).unwrap();
        write_full_markdown(&full_md, &metadata, &full).unwrap();
        write_full_csv(&full_csv, &metadata, &full).unwrap();

        let json_str = std::fs::read_to_string(&full_json).unwrap();
        assert!(json_str.contains("\"ndcg_at_10\": 0.75"));
        assert!(json_str.contains("\"reranker_lift\": 4.5"));
        let md_str = std::fs::read_to_string(&full_md).unwrap();
        assert!(md_str.contains("MRL Dimension Sweep"));
        assert!(md_str.contains("+4.50%"));
        let csv_str = std::fs::read_to_string(&full_csv).unwrap();
        assert!(csv_str.contains("ndcg_at_10"));
    }
}
