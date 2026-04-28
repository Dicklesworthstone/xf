//! Benchmark metrics for embedding and reranker evaluation.
//!
//! This module provides metric computation functions and structs for the bake-off
//! benchmark harness. Key components:
//!
//! - [`SpeedMetrics`]: Latency, throughput, and cold-start timing
//! - [`QualityMetrics`]: IR metrics (NDCG, MRR, MAP, recall, precision)
//! - [`ReliabilityMetrics`]: Error rates and stability measures
//!
//! Metric computation functions:
//! - [`ndcg_at_k`], [`compute_mrr`], [`map_at_k`]: Quality metrics from ranked relevance
//! - [`percentile`], [`coefficient_of_variation`]: Statistical aggregation
//! - [`bootstrap_ci`], [`bootstrap_mean_diff`]: Significance testing

use rand::{RngExt, SeedableRng, rngs::StdRng};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Speed metrics collected during benchmark runs.
///
/// These metrics capture both cold-start behavior and steady-state performance,
/// which is critical for comparing static vs transformer embedders.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct SpeedMetrics {
    /// Time to load model + first embed (cold start).
    pub cold_start_ms: f64,
    /// 50th percentile of warm iteration latencies.
    pub warm_latency_p50_ms: f64,
    /// 95th percentile of warm iteration latencies.
    pub warm_latency_p95_ms: f64,
    /// 99th percentile of warm iteration latencies.
    pub warm_latency_p99_ms: f64,
    /// Maximum observed latency across all warm iterations.
    pub warm_latency_max_ms: f64,
    /// Average throughput (batch_size * iterations / total_time).
    pub throughput_docs_per_sec: f64,
    /// Cold start + first inference time.
    pub time_to_first_result_ms: f64,
    /// Coefficient of variation (std/mean) for latency jitter measurement.
    pub jitter_cv: f64,
}

/// Quality metrics computed against ground truth corpus.
///
/// Relevance judgments use graded scale: 2 = highly relevant, 1 = partially relevant, 0 = not.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Normalized Discounted Cumulative Gain at rank 10 (primary quality metric).
    pub ndcg_at_10: f64,
    /// Mean Reciprocal Rank at rank 10.
    pub mrr_at_10: f64,
    /// Mean Average Precision at rank 10.
    pub map_at_10: f64,
    /// Recall at rank 100 (fraction of relevant docs retrieved in top 100).
    pub recall_at_100: f64,
    /// Precision at rank 10 (fraction of top-10 that are relevant).
    pub precision_at_10: f64,
}

/// Reliability metrics for a benchmark run.
///
/// Used by [`crate::benchmark::runner::BenchmarkResult`] to track stability.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct ReliabilityMetrics {
    /// Fraction of requests that failed (errors / total_requests).
    pub error_rate: f64,
    /// Largest batch size that succeeded without OOM.
    pub oom_threshold_batch: usize,
    /// Fraction of requests that fell back to a simpler method.
    pub fallback_rate: f64,
}

/// Compute a percentile from a list of values.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn percentile(values: &[f64], pct: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let clamped = pct.clamp(0.0, 1.0);
    let idx = ((sorted.len() - 1) as f64 * clamped).round() as usize;
    sorted.get(idx).copied()
}

/// Compute MRR from a ranked list of graded relevance (0 = not relevant).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn compute_mrr(relevance: &[u8]) -> f64 {
    for (idx, rel) in relevance.iter().enumerate() {
        if *rel > 0 {
            return 1.0 / (idx as f64 + 1.0);
        }
    }
    0.0
}

/// Compute precision@k from graded relevance.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn precision_at_k(relevance: &[u8], k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let hits = relevance.iter().take(k).filter(|r| **r > 0).count();
    hits as f64 / k as f64
}

/// Compute recall@k given total relevant count.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn recall_at_k(relevance: &[u8], k: usize, total_relevant: usize) -> f64 {
    if total_relevant == 0 {
        return 0.0;
    }
    let hits = relevance.iter().take(k).filter(|r| **r > 0).count();
    hits as f64 / total_relevant as f64
}

/// Compute DCG for graded relevance.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::cast_lossless)]
pub fn dcg(relevance: &[u8], k: usize) -> f64 {
    relevance
        .iter()
        .take(k)
        .enumerate()
        .map(|(i, rel)| {
            let gain = f64::from(2_u32.pow(u32::from(*rel)) - 1);
            let denom = (i as f64 + 2.0).log2();
            gain / denom
        })
        .sum()
}

/// Compute nDCG@k from graded relevance.
#[must_use]
pub fn ndcg_at_k(relevance: &[u8], k: usize) -> f64 {
    if relevance.is_empty() || k == 0 {
        return 0.0;
    }
    let actual = dcg(relevance, k);
    let mut ideal = relevance.to_vec();
    ideal.sort_by(|a, b| b.cmp(a));
    let ideal_dcg = dcg(&ideal, k);
    if ideal_dcg == 0.0 {
        0.0
    } else {
        actual / ideal_dcg
    }
}

/// Compute Mean Average Precision at k.
///
/// For each relevant document found in the top-k, compute precision at that rank,
/// then average over all relevant documents.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::cast_lossless)]
pub fn map_at_k(relevance: &[u8], k: usize) -> f64 {
    if k == 0 || relevance.is_empty() {
        return 0.0;
    }
    let total_relevant = relevance.iter().filter(|r| **r > 0).count();
    if total_relevant == 0 {
        return 0.0;
    }

    let mut sum_precision = 0.0;
    let mut relevant_count: u32 = 0;

    for (i, &rel) in relevance.iter().take(k).enumerate() {
        if rel > 0 {
            relevant_count += 1;
            sum_precision += f64::from(relevant_count) / (i + 1) as f64;
        }
    }

    sum_precision / total_relevant as f64
}

/// Compute coefficient of variation (std/mean) for jitter measurement.
///
/// Returns 0.0 if mean is zero or slice is empty.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn coefficient_of_variation(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    if mean == 0.0 {
        return 0.0;
    }
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
    variance.sqrt() / mean
}

/// Compute coefficient of variation from Duration values (convenience wrapper).
#[must_use]
pub fn cv_from_durations(values: &[Duration]) -> f64 {
    let ms: Vec<f64> = values.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    coefficient_of_variation(&ms)
}

/// Compute percentile from Duration values (convenience wrapper).
#[must_use]
pub fn percentile_from_durations(values: &[Duration], pct: f64) -> Option<f64> {
    let ms: Vec<f64> = values.iter().map(|d| d.as_secs_f64() * 1000.0).collect();
    percentile(&ms, pct)
}

/// Compute maximum latency from Duration values.
#[must_use]
pub fn max_latency_ms(values: &[Duration]) -> f64 {
    values
        .iter()
        .map(|d| d.as_secs_f64() * 1000.0)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(0.0)
}

/// Bootstrap confidence interval for a single sample mean.
///
/// Returns (lower_bound, upper_bound) for the specified confidence level.
/// Uses deterministic seed for reproducibility.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn bootstrap_ci(values: &[f64], n_bootstrap: usize, confidence: f64, seed: u64) -> (f64, f64) {
    if values.is_empty() || n_bootstrap == 0 {
        return (0.0, 0.0);
    }

    let n = values.len();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut means = Vec::with_capacity(n_bootstrap);

    for _ in 0..n_bootstrap {
        let sample_mean: f64 =
            (0..n).map(|_| values[rng.random_range(0..n)]).sum::<f64>() / n as f64;
        means.push(sample_mean);
    }

    means.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let alpha = (1.0 - confidence) / 2.0;
    let lo_idx = (alpha * means.len() as f64).floor() as usize;
    let hi_idx = ((1.0 - alpha) * means.len() as f64).ceil() as usize;
    (means[lo_idx], means[hi_idx.min(means.len() - 1)])
}

/// Bootstrap confidence interval for mean difference between two samples.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn bootstrap_mean_diff(
    baseline: &[f64],
    improved: &[f64],
    n_bootstrap: usize,
    seed: u64,
) -> (f64, f64, bool) {
    if baseline.is_empty() || improved.is_empty() || n_bootstrap == 0 {
        return (0.0, 0.0, false);
    }
    let n = baseline.len().min(improved.len());
    let mut rng = StdRng::seed_from_u64(seed);
    let mut diffs = Vec::with_capacity(n_bootstrap);
    for _ in 0..n_bootstrap {
        let mut base_sum = 0.0;
        let mut imp_sum = 0.0;
        for _ in 0..n {
            let idx = rng.random_range(0..n);
            base_sum += baseline[idx];
            imp_sum += improved[idx];
        }
        diffs.push((imp_sum / n as f64) - (base_sum / n as f64));
    }
    diffs.sort_by(f64::total_cmp);
    let low = diffs[(n_bootstrap as f64 * 0.05) as usize];
    let high = diffs[(n_bootstrap as f64 * 0.95) as usize];
    let significant = low > 0.0;
    (low, high, significant)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_mrr_first_relevant() {
        let relevance = vec![0, 1, 0, 0];
        let mrr = compute_mrr(&relevance);
        assert!((mrr - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_ndcg_perfect() {
        let relevance = vec![2, 1, 0, 0];
        let ndcg = ndcg_at_k(&relevance, 4);
        assert!((ndcg - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_precision_recall() {
        let relevance = vec![1, 0, 1, 0, 1];
        assert!((precision_at_k(&relevance, 3) - (2.0 / 3.0)).abs() < 1e-6);
        assert!((recall_at_k(&relevance, 3, 3) - (2.0 / 3.0)).abs() < 1e-6);
    }

    // === MAP tests ===

    #[test]
    fn test_map_all_relevant() {
        // All top-k are relevant: AP = (1/1 + 2/2 + 3/3) / 3 = 1.0
        let relevance = vec![1, 1, 1];
        assert!((map_at_k(&relevance, 3) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_map_alternating() {
        // Relevant at positions 1, 3: AP = (1/1 + 2/3) / 2 = 0.833...
        let relevance = vec![1, 0, 1, 0];
        let expected = f64::midpoint(1.0 / 1.0, 2.0 / 3.0);
        assert!((map_at_k(&relevance, 4) - expected).abs() < 1e-6);
    }

    #[test]
    fn test_map_no_relevant() {
        let relevance = vec![0, 0, 0];
        assert!(map_at_k(&relevance, 3).abs() < f64::EPSILON);
    }

    #[test]
    fn test_map_empty() {
        let relevance: Vec<u8> = vec![];
        assert!(map_at_k(&relevance, 10).abs() < f64::EPSILON);
    }

    // === NDCG edge cases ===

    #[test]
    fn test_ndcg_worst_ranking() {
        // All relevant docs at bottom
        let relevance = vec![0, 0, 0, 2, 1];
        let score = ndcg_at_k(&relevance, 5);
        assert!(score < 0.5, "Worst ranking should score low: {score}");
    }

    #[test]
    fn test_ndcg_empty() {
        let relevance: Vec<u8> = vec![];
        assert!(ndcg_at_k(&relevance, 10).abs() < f64::EPSILON);
    }

    #[test]
    fn test_ndcg_no_relevant() {
        let relevance = vec![0, 0, 0];
        assert!(ndcg_at_k(&relevance, 3).abs() < f64::EPSILON);
    }

    // === MRR edge cases ===

    #[test]
    fn test_mrr_first_position() {
        let relevance = vec![1, 0, 0];
        assert!((compute_mrr(&relevance) - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mrr_third_position() {
        let relevance = vec![0, 0, 1, 0];
        assert!((compute_mrr(&relevance) - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_mrr_no_relevant() {
        let relevance = vec![0, 0, 0];
        assert!(compute_mrr(&relevance).abs() < f64::EPSILON);
    }

    // === Percentile edge cases ===

    #[test]
    fn test_percentile_single_value() {
        let values = vec![42.0];
        assert!((percentile(&values, 0.5).unwrap() - 42.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_percentile_median() {
        let values: Vec<f64> = (1..=100).map(f64::from).collect();
        let p50 = percentile(&values, 0.50).unwrap();
        assert!((p50 - 50.0).abs() < 1.5, "p50 should be ~50: {p50}");
    }

    #[test]
    fn test_percentile_p99() {
        let values: Vec<f64> = (1..=100).map(f64::from).collect();
        let p99 = percentile(&values, 0.99).unwrap();
        assert!(p99 >= 99.0, "p99 should be >=99: {p99}");
    }

    // === Coefficient of variation tests ===

    #[test]
    fn test_cv_constant_values() {
        let values: Vec<f64> = (0..10).map(|_| 50.0).collect();
        assert!(coefficient_of_variation(&values).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cv_variable_values() {
        let values = vec![10.0, 100.0];
        let cv = coefficient_of_variation(&values);
        assert!(cv > 0.5, "High variability should give high CV: {cv}");
    }

    #[test]
    fn test_cv_empty() {
        assert!(coefficient_of_variation(&[]).abs() < f64::EPSILON);
    }

    #[test]
    fn test_cv_from_durations() {
        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(20),
            Duration::from_millis(30),
        ];
        let cv = cv_from_durations(&durations);
        assert!(cv > 0.0);
        assert!(cv < 1.0);
    }

    // === Bootstrap CI tests ===

    #[test]
    fn test_bootstrap_ci_narrow_for_tight_data() {
        let values: Vec<f64> = (0..100).map(|i| f64::from(i).mul_add(0.01, 50.0)).collect();
        let (lo, hi) = bootstrap_ci(&values, 1000, 0.95, 42);
        assert!(
            (hi - lo) < 1.0,
            "Tight data should give narrow CI: [{lo}, {hi}]"
        );
    }

    #[test]
    fn test_bootstrap_ci_deterministic() {
        let values: Vec<f64> = (0..50).map(f64::from).collect();
        let (lo1, hi1) = bootstrap_ci(&values, 100, 0.95, 42);
        let (lo2, hi2) = bootstrap_ci(&values, 100, 0.95, 42);
        assert!((lo1 - lo2).abs() < f64::EPSILON);
        assert!((hi1 - hi2).abs() < f64::EPSILON);
    }

    #[test]
    fn test_bootstrap_ci_empty() {
        let (lo, hi) = bootstrap_ci(&[], 100, 0.95, 42);
        assert!(lo.abs() < f64::EPSILON);
        assert!(hi.abs() < f64::EPSILON);
    }

    // === Max latency test ===

    #[test]
    fn test_max_latency() {
        let durations = vec![
            Duration::from_millis(10),
            Duration::from_millis(50),
            Duration::from_millis(30),
        ];
        assert!((max_latency_ms(&durations) - 50.0).abs() < 0.01);
    }

    // === Property tests ===

    proptest! {
        #[test]
        fn percentiles_monotonic(vals in proptest::collection::vec(0.0f64..1000.0, 1..100)) {
            let p10 = percentile(&vals, 0.1).unwrap();
            let p50 = percentile(&vals, 0.5).unwrap();
            let p90 = percentile(&vals, 0.9).unwrap();
            prop_assert!(p10 <= p50 + f64::EPSILON);
            prop_assert!(p50 <= p90 + f64::EPSILON);
        }

        #[test]
        fn ndcg_bounded(rels in proptest::collection::vec(0u8..3, 1..20)) {
            let score = ndcg_at_k(&rels, 10);
            prop_assert!((0.0..=1.0).contains(&score), "NDCG must be in [0,1]: {score}");
        }

        #[test]
        fn map_bounded(rels in proptest::collection::vec(0u8..2, 1..20)) {
            let score = map_at_k(&rels, 10);
            prop_assert!((0.0..=1.0).contains(&score), "MAP must be in [0,1]: {score}");
        }

        #[test]
        fn cv_non_negative(vals in proptest::collection::vec(0.0f64..1000.0, 1..100)) {
            let cv = coefficient_of_variation(&vals);
            prop_assert!(cv >= 0.0, "CV must be non-negative: {cv}");
        }
    }
}
