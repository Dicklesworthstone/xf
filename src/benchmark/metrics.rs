//! Benchmark metrics for embedding and reranker evaluation.

use rand::{Rng, SeedableRng, rngs::StdRng};
use serde::Serialize;

/// Speed metrics for a benchmark run.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct SpeedMetrics {
    pub cold_start_ms: f64,
    pub warm_latency_p50_ms: f64,
    pub warm_latency_p95_ms: f64,
    pub warm_latency_p99_ms: f64,
    pub throughput_docs_per_sec: f64,
    pub time_to_first_result_ms: f64,
}

/// Quality metrics for a benchmark run.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct QualityMetrics {
    pub mrr_at_10: f64,
    pub ndcg_at_10: f64,
    pub recall_at_100: f64,
    pub precision_at_10: f64,
}

/// Reliability metrics for a benchmark run.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct ReliabilityMetrics {
    pub error_rate: f64,
    pub oom_threshold_batch: usize,
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
            let idx = rng.gen_range(0..n);
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

    proptest! {
        #[test]
        fn percentiles_monotonic(vals in proptest::collection::vec(0.0f64..1000.0, 1..100)) {
            let p10 = percentile(&vals, 0.1).unwrap();
            let p50 = percentile(&vals, 0.5).unwrap();
            let p90 = percentile(&vals, 0.9).unwrap();
            prop_assert!(p10 <= p50 + f64::EPSILON);
            prop_assert!(p50 <= p90 + f64::EPSILON);
        }
    }
}
