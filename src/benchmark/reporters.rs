//! Reporters for benchmark results.

use std::fs::File;
use std::io::{Write, BufWriter};
use std::path::Path;

use anyhow::Result;
use serde::Serialize;

use crate::benchmark::metrics::{QualityMetrics, ReliabilityMetrics, SpeedMetrics};
use crate::benchmark::runner::{BenchmarkMetadata, BenchmarkResult};

/// Serializable report wrapper.
#[derive(Debug, Serialize)]
pub struct BenchmarkReport<'a> {
    pub metadata: &'a BenchmarkMetadata,
    pub speed: &'a SpeedMetrics,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quality: Option<&'a QualityMetrics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reliability: Option<&'a ReliabilityMetrics>,
    pub latency_ms: &'a [f64],
}

/// Write JSON report.
pub fn write_json(path: &Path, result: &BenchmarkResult) -> Result<()> {
    let report = BenchmarkReport {
        metadata: &result.metadata,
        speed: &result.speed,
        quality: None,
        reliability: None,
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
    out.push_str(&format!("Model: `{}`\n\n", result.metadata.model));
    out.push_str("| Metric | Value |\n|---|---|\n");
    out.push_str(&format!(
        "| Cold start (ms) | {:.2} |\n",
        result.speed.cold_start_ms
    ));
    out.push_str(&format!(
        "| TTFR (ms) | {:.2} |\n",
        result.speed.time_to_first_result_ms
    ));
    out.push_str(&format!(
        "| Warm p50 (ms) | {:.2} |\n",
        result.speed.warm_latency_p50_ms
    ));
    out.push_str(&format!(
        "| Warm p95 (ms) | {:.2} |\n",
        result.speed.warm_latency_p95_ms
    ));
    out.push_str(&format!(
        "| Warm p99 (ms) | {:.2} |\n",
        result.speed.warm_latency_p99_ms
    ));
    out.push_str(&format!(
        "| Throughput (docs/sec) | {:.2} |\n",
        result.speed.throughput_docs_per_sec
    ));
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
