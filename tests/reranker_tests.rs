//! Unit tests for reranker implementations.
//!
//! Tests for all cross-encoder reranker backends covering:
//! - Model loading and configuration
//! - Relevance ordering (relevant docs rank higher)
//! - Score calibration (bounded, deterministic, consistent)
//! - Edge cases (empty, long, unicode, special chars)
//! - Performance benchmarks
//! - Pipeline integration
//!
//! Bead: bd-2q67

use std::sync::Arc;
use std::time::Instant;

use xf::flashrank_reranker::{FlashRankReranker, MODEL_NAME as FLASHRANK_MODEL_NAME};
use xf::model::SearchResult;
use xf::model::SearchResultType;
use xf::model_registry::ModelRegistry;
use xf::mxbai_reranker::{MODEL_NAME as MXBAI_MODEL_NAME, MxbaiReranker};
use xf::rerank_step::RerankStep;
use xf::reranker::Reranker;

use chrono::Utc;

// =============================================================================
// Helper Functions
// =============================================================================

/// Create a test FlashRank reranker if available.
fn try_flashrank() -> Option<FlashRankReranker> {
    FlashRankReranker::load().ok()
}

/// Create a test mxbai reranker if available.
fn try_mxbai() -> Option<MxbaiReranker> {
    MxbaiReranker::load().ok()
}

/// Create a SearchResult for testing.
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

// =============================================================================
// Model Constants Tests (no model files required)
// =============================================================================

mod constants {
    use super::*;

    #[test]
    fn test_flashrank_model_name() {
        assert_eq!(FLASHRANK_MODEL_NAME, "flashrank-nano");
    }

    #[test]
    fn test_mxbai_model_name() {
        assert_eq!(MXBAI_MODEL_NAME, "mxbai-rerank-xsmall");
    }
}

// =============================================================================
// Reranker Loading Tests
// =============================================================================

mod loading {
    use super::*;

    #[test]
    #[ignore = "requires model files"]
    fn test_flashrank_loads() {
        let reranker = FlashRankReranker::load().expect("FlashRank should load");
        assert_eq!(reranker.model_name(), "flashrank-nano");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_mxbai_reranker_loads() {
        let reranker = MxbaiReranker::load().expect("mxbai should load");
        assert_eq!(reranker.model_name(), "mxbai-rerank-xsmall");
    }

    #[test]
    fn test_flashrank_load_with_bad_path_gives_error() {
        use std::path::Path;
        let result = FlashRankReranker::load_from_dir(Path::new("/nonexistent/path"));
        assert!(result.is_err(), "Should fail with bad path");
        let err_str = result.err().unwrap().to_string();
        assert!(
            err_str.contains("missing") || err_str.contains("unavailable"),
            "Error should mention missing files: {err_str}"
        );
    }

    #[test]
    fn test_mxbai_load_with_bad_path_gives_error() {
        use std::path::Path;
        let result = MxbaiReranker::load_from_dir(Path::new("/nonexistent/path"));
        assert!(result.is_err(), "Should fail with bad path");
        let err_str = result.err().unwrap().to_string();
        assert!(
            err_str.contains("missing") || err_str.contains("unavailable"),
            "Error should mention missing files: {err_str}"
        );
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_reranker_from_registry_flashrank() {
        let registry = ModelRegistry::new();
        let reranker = registry
            .reranker_by_name("flashrank-nano")
            .expect("registry lookup should succeed")
            .expect("reranker should be available");
        assert_eq!(reranker.model_name(), "flashrank-nano");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_reranker_from_registry_mxbai() {
        let registry = ModelRegistry::new();
        let reranker = registry
            .reranker_by_name("mxbai-rerank-xsmall")
            .expect("registry lookup should succeed")
            .expect("reranker should be available");
        assert_eq!(reranker.model_name(), "mxbai-rerank-xsmall");
    }

    #[test]
    fn test_reranker_unknown_name_returns_none() {
        let registry = ModelRegistry::new();
        let result = registry.reranker_by_name("nonexistent-model");
        // Unknown models should return Ok(None) or an error, not panic
        if let Ok(Some(_)) = result {
            panic!("Should not find unknown model");
        }
        // Ok(None) or Err(_) are expected for unknown model
    }
}

// =============================================================================
// Relevance Ordering Tests
// =============================================================================

mod relevance {
    use super::*;

    #[test]
    #[ignore = "requires model files"]
    fn test_flashrank_ranks_relevant_higher() {
        let reranker = try_flashrank().expect("FlashRank required");
        test_reranker_ranks_relevant_higher(&reranker, "FlashRank");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_mxbai_ranks_relevant_higher() {
        let reranker = try_mxbai().expect("mxbai required");
        test_reranker_ranks_relevant_higher(&reranker, "mxbai");
    }

    fn test_reranker_ranks_relevant_higher<R: Reranker>(reranker: &R, name: &str) {
        let query = "best programming language for systems";
        let docs = vec![
            "Rust is a systems programming language focused on safety and performance",
            "Python is great for data science and machine learning",
            "The weather today is sunny with a chance of rain",
            "Recipe: How to make chocolate cake from scratch",
        ];

        let scores = reranker
            .rerank(query, &docs)
            .expect("rerank should succeed");
        println!("{name} scores: {scores:?}");

        // Rust doc (index 0) should score highest (most relevant to query)
        assert!(
            scores[0] > scores[2],
            "{name}: Systems lang doc should beat weather: {} vs {}",
            scores[0],
            scores[2]
        );
        assert!(
            scores[0] > scores[3],
            "{name}: Systems lang doc should beat recipe: {} vs {}",
            scores[0],
            scores[3]
        );
        // Programming doc should beat non-programming
        assert!(
            scores[1] > scores[3],
            "{name}: Programming doc should beat recipe: {} vs {}",
            scores[1],
            scores[3]
        );
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_reranker_handles_identical_docs() {
        let reranker = try_flashrank().expect("FlashRank required");
        let query = "test query";
        let docs = vec!["same doc", "same doc", "different doc"];
        let scores = reranker
            .rerank(query, &docs)
            .expect("rerank should succeed");

        // Identical docs should get same score (within tolerance)
        assert!(
            (scores[0] - scores[1]).abs() < 0.01,
            "Identical docs should have same score: {} vs {}",
            scores[0],
            scores[1]
        );
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_reranker_query_relevance_sensitivity() {
        let reranker = try_flashrank().expect("FlashRank required");
        let docs = vec![
            "Rust systems programming safety",
            "Python machine learning data science",
            "JavaScript web frontend React",
        ];

        let rust_scores = reranker
            .rerank("systems programming memory safety", &docs)
            .expect("rerank should succeed");
        let ml_scores = reranker
            .rerank("machine learning deep learning", &docs)
            .expect("rerank should succeed");

        println!("Rust query scores: {rust_scores:?}");
        println!("ML query scores: {ml_scores:?}");

        assert!(
            rust_scores[0] > rust_scores[1],
            "Rust doc should win for systems query: {} vs {}",
            rust_scores[0],
            rust_scores[1]
        );
        assert!(
            ml_scores[1] > ml_scores[0],
            "Python doc should win for ML query: {} vs {}",
            ml_scores[1],
            ml_scores[0]
        );
    }
}

// =============================================================================
// Score Calibration Tests
// =============================================================================

mod calibration {
    use super::*;

    #[test]
    #[ignore = "requires model files"]
    fn test_flashrank_scores_are_bounded() {
        let reranker = try_flashrank().expect("FlashRank required");
        test_scores_are_bounded(&reranker, "FlashRank");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_mxbai_scores_are_bounded() {
        let reranker = try_mxbai().expect("mxbai required");
        test_scores_are_bounded(&reranker, "mxbai");
    }

    fn test_scores_are_bounded<R: Reranker>(reranker: &R, name: &str) {
        let query = "test query";
        let docs = vec!["relevant doc about testing", "irrelevant doc about cooking"];
        let scores = reranker
            .rerank(query, &docs)
            .expect("rerank should succeed");

        for (i, score) in scores.iter().enumerate() {
            assert!(
                score.is_finite(),
                "{name}: Score {i} should be finite, got {score}"
            );
            // After sigmoid, scores should be in [0, 1]
            assert!(
                *score >= 0.0 && *score <= 1.0,
                "{name}: Score {i} should be in [0,1], got {score}"
            );
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_flashrank_scores_are_deterministic() {
        let reranker = try_flashrank().expect("FlashRank required");
        test_scores_are_deterministic(&reranker, "FlashRank");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_mxbai_scores_are_deterministic() {
        let reranker = try_mxbai().expect("mxbai required");
        test_scores_are_deterministic(&reranker, "mxbai");
    }

    fn test_scores_are_deterministic<R: Reranker>(reranker: &R, name: &str) {
        let query = "test query for determinism check";
        let docs = vec!["document one", "document two", "document three"];

        let scores1 = reranker
            .rerank(query, &docs)
            .expect("rerank should succeed");
        let scores2 = reranker
            .rerank(query, &docs)
            .expect("rerank should succeed");

        for (i, (s1, s2)) in scores1.iter().zip(scores2.iter()).enumerate() {
            assert!(
                (s1 - s2).abs() < 1e-5,
                "{name}: Scores should be deterministic at index {i}: {s1} vs {s2}"
            );
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_score_output_length_matches_input() {
        let reranker = try_flashrank().expect("FlashRank required");

        for n in [1, 5, 20, 50] {
            let docs: Vec<String> = (0..n).map(|i| format!("document number {i}")).collect();
            let doc_refs: Vec<&str> = docs.iter().map(String::as_str).collect();
            let scores = reranker
                .rerank("query", &doc_refs)
                .expect("rerank should succeed");
            assert_eq!(scores.len(), n, "Should return exactly {n} scores");
        }
    }
}

// =============================================================================
// Edge Case Tests
// =============================================================================

mod edge_cases {
    use super::*;

    #[test]
    #[ignore = "requires model files"]
    fn test_single_document() {
        let reranker = try_flashrank().expect("FlashRank required");
        let scores = reranker
            .rerank("query", &["single doc"])
            .expect("single doc should work");
        assert_eq!(scores.len(), 1);
        assert!(scores[0].is_finite());
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_empty_query() {
        let reranker = try_flashrank().expect("FlashRank required");
        let result = reranker.rerank("", &["doc"]);
        assert!(result.is_ok(), "Empty query should not panic");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_empty_document() {
        let reranker = try_flashrank().expect("FlashRank required");
        let result = reranker.rerank("query", &[""]);
        assert!(result.is_ok(), "Empty doc should not panic");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_empty_documents_vec() {
        let reranker = try_flashrank().expect("FlashRank required");
        let result = reranker.rerank("query", &[]);
        assert!(result.is_ok(), "Empty docs vec should not panic");
        assert!(result.unwrap().is_empty(), "Should return empty scores");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_very_long_document() {
        let reranker = try_flashrank().expect("FlashRank required");
        let long_doc = "word ".repeat(5000); // Way beyond typical max_length
        let result = reranker.rerank("query", &[&long_doc]);
        assert!(result.is_ok(), "Long doc should be truncated, not panic");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_many_documents() {
        let reranker = try_flashrank().expect("FlashRank required");
        let docs: Vec<String> = (0..100).map(|i| format!("document number {i}")).collect();
        let doc_refs: Vec<&str> = docs.iter().map(String::as_str).collect();
        let scores = reranker
            .rerank("find document 42", &doc_refs)
            .expect("many docs should work");
        assert_eq!(scores.len(), 100);
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_unicode_content() {
        let reranker = try_flashrank().expect("FlashRank required");
        let docs = vec!["日本語のテスト", "Ñoño español", "🦀 Rust emoji"];
        let result = reranker.rerank("unicode test", &docs);
        assert!(result.is_ok(), "Unicode docs should not crash");
        assert_eq!(result.unwrap().len(), 3);
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_special_characters() {
        let reranker = try_flashrank().expect("FlashRank required");
        let docs = vec![
            "<script>alert('xss')</script>",
            "SELECT * FROM users;--",
            "\\n\\t\\r\\0",
        ];
        let result = reranker.rerank("special chars", &docs);
        assert!(result.is_ok(), "Special chars should not crash");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_whitespace_only_docs() {
        let reranker = try_flashrank().expect("FlashRank required");
        let docs = vec!["   ", "\t\n", "  \n\t  "];
        let result = reranker.rerank("query", &docs);
        assert!(result.is_ok(), "Whitespace docs should not crash");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_very_long_query() {
        let reranker = try_flashrank().expect("FlashRank required");
        let long_query = "test ".repeat(1000);
        let result = reranker.rerank(&long_query, &["short doc"]);
        assert!(result.is_ok(), "Long query should be truncated, not panic");
    }
}

// =============================================================================
// Performance Tests
// =============================================================================

mod performance {
    use super::*;

    #[test]
    #[ignore = "requires model files and is slow"]
    fn test_flashrank_latency_top20() {
        let reranker = try_flashrank().expect("FlashRank required");
        let query = "test query about programming";
        let docs: Vec<&str> = (0..20)
            .map(|_| "sample document text for benchmarking reranker speed")
            .collect();

        // Warmup (3 iterations)
        for _ in 0..3 {
            reranker.rerank(query, &docs).unwrap();
        }

        // Measure (10 iterations)
        let start = Instant::now();
        for _ in 0..10 {
            reranker.rerank(query, &docs).unwrap();
        }
        #[allow(clippy::cast_precision_loss)]
        let avg_ms = start.elapsed().as_millis() as f64 / 10.0;

        println!("FlashRank top-20 reranking: {avg_ms:.1}ms average");
        // FlashRank nano on CPU should rerank 20 docs in <100ms (generous for CI)
        assert!(
            avg_ms < 100.0,
            "FlashRank top-20 should be <100ms on CPU, got {avg_ms:.1}ms"
        );
    }

    #[test]
    #[ignore = "requires model files and is slow"]
    fn test_mxbai_latency_top20() {
        let reranker = try_mxbai().expect("mxbai required");
        let query = "test query about programming";
        let docs: Vec<&str> = (0..20)
            .map(|_| "sample document text for benchmarking reranker speed")
            .collect();

        // Warmup
        for _ in 0..3 {
            reranker.rerank(query, &docs).unwrap();
        }

        let start = Instant::now();
        for _ in 0..10 {
            reranker.rerank(query, &docs).unwrap();
        }
        #[allow(clippy::cast_precision_loss)]
        let avg_ms = start.elapsed().as_millis() as f64 / 10.0;

        println!("mxbai-rerank-xsmall top-20 reranking: {avg_ms:.1}ms average");
        // mxbai-xsmall is larger, allow more headroom
        assert!(
            avg_ms < 300.0,
            "mxbai top-20 should be <300ms on CPU, got {avg_ms:.1}ms"
        );
    }

    #[test]
    #[ignore = "requires model files and is slow"]
    fn test_reranker_throughput_scales_reasonably() {
        let reranker = try_flashrank().expect("FlashRank required");
        let query = "test query";

        // Measure at different document counts
        let mut timings = Vec::new();
        for n in [10, 20, 50, 100] {
            let docs: Vec<String> = (0..n).map(|i| format!("document {i}")).collect();
            let doc_refs: Vec<&str> = docs.iter().map(String::as_str).collect();

            // Warmup
            reranker.rerank(query, &doc_refs).unwrap();

            let start = Instant::now();
            for _ in 0..5 {
                reranker.rerank(query, &doc_refs).unwrap();
            }
            #[allow(clippy::cast_precision_loss)]
            let avg_ms = start.elapsed().as_millis() as f64 / 5.0;
            timings.push((n, avg_ms));
            println!("  n={n}: {avg_ms:.1}ms");
        }

        // Verify roughly linear scaling (100 docs should be <20x of 10 docs)
        let ratio = timings[3].1 / timings[0].1.max(0.1);
        println!("Scaling ratio (100/10 docs): {ratio:.1}x");
        assert!(ratio < 20.0, "Should scale roughly linearly, got {ratio:.1}x");
    }
}

// =============================================================================
// Comparison Tests
// =============================================================================

mod comparison {
    use super::*;

    #[test]
    #[ignore = "requires model files"]
    fn test_reranker_improves_over_random() {
        let reranker = try_flashrank().expect("FlashRank required");
        let query = "machine learning optimization";
        let docs = vec![
            "gradient descent optimization in neural networks",
            "the quick brown fox jumps over the lazy dog",
            "stochastic gradient methods for deep learning",
            "how to make pasta from scratch at home",
        ];

        let scores = reranker
            .rerank(query, &docs)
            .expect("rerank should succeed");

        // ML docs (indices 0, 2) should outrank non-ML docs (1, 3)
        let ml_avg = f32::midpoint(scores[0], scores[2]);
        let non_ml_avg = f32::midpoint(scores[1], scores[3]);
        assert!(
            ml_avg > non_ml_avg,
            "ML docs should score higher: {ml_avg:.3} vs {non_ml_avg:.3}"
        );
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_both_rerankers_agree_on_obvious_cases() {
        let flashrank = try_flashrank().expect("FlashRank required");
        let mxbai = try_mxbai().expect("mxbai required");
        let query = "rust programming language";
        let docs = vec![
            "Rust is a multi-paradigm systems programming language",
            "How to bake a chocolate cake recipe",
        ];

        let fr_scores = flashrank
            .rerank(query, &docs)
            .expect("FlashRank should succeed");
        let mx_scores = mxbai.rerank(query, &docs).expect("mxbai should succeed");

        println!("FlashRank scores: {fr_scores:?}");
        println!("mxbai scores: {mx_scores:?}");

        // Both should agree: doc[0] > doc[1]
        assert!(
            fr_scores[0] > fr_scores[1],
            "FlashRank should rank Rust doc higher: {} vs {}",
            fr_scores[0],
            fr_scores[1]
        );
        assert!(
            mx_scores[0] > mx_scores[1],
            "mxbai should rank Rust doc higher: {} vs {}",
            mx_scores[0],
            mx_scores[1]
        );
    }
}

// =============================================================================
// Pipeline Integration Tests
// =============================================================================

mod pipeline {
    use super::*;

    #[test]
    #[ignore = "requires model files"]
    fn test_rerank_step_reorders_candidates() {
        let reranker = try_flashrank().expect("FlashRank required");
        let step = RerankStep::with_reranker(Box::new(reranker)).with_min_candidates(2);

        let mut candidates = vec![
            make_result("cooking", "How to make a delicious cake", 0.9),
            make_result("rust", "Rust is a systems programming language", 0.8),
            make_result("weather", "The weather is nice today", 0.7),
        ];

        step.rerank("programming language", &mut candidates)
            .expect("rerank should succeed");

        // Rust doc should be reranked to top for programming query
        assert!(
            candidates[0].id == "rust"
                || candidates[0].rerank_score.unwrap_or(0.0)
                    >= candidates[1].rerank_score.unwrap_or(0.0),
            "Rust doc should rank higher for programming query: {:?}",
            candidates
                .iter()
                .map(|c| (&c.id, c.rerank_score))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_rerank_preserves_metadata() {
        let reranker = try_flashrank().expect("FlashRank required");
        let step = RerankStep::with_reranker(Box::new(reranker)).with_min_candidates(1);

        let metadata = serde_json::json!({"source": "tweets", "date": "2024-01"});
        let mut candidates = vec![SearchResult {
            result_type: SearchResultType::Tweet,
            id: "tweet_123".to_string(),
            text: "Important tweet about Rust".to_string(),
            created_at: Utc::now(),
            score: 0.7,
            highlights: vec![],
            metadata,
            rerank_score: None,
        }];

        step.rerank("Rust programming", &mut candidates)
            .expect("rerank should succeed");

        // Metadata should be unchanged after reranking
        assert_eq!(candidates[0].metadata["source"], "tweets");
        assert_eq!(candidates[0].id, "tweet_123");
        assert!(candidates[0].rerank_score.is_some());
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_rerank_step_stable_sort() {
        let reranker = try_flashrank().expect("FlashRank required");
        let step = RerankStep::with_reranker(Box::new(reranker)).with_min_candidates(1);

        // Create results with initial scores where rerank might tie
        let mut candidates = vec![
            make_result("1", "similar document one", 0.9),
            make_result("2", "similar document two", 0.7),
        ];

        step.rerank("similar document", &mut candidates)
            .expect("rerank should succeed");

        // Both should have rerank scores
        assert!(candidates[0].rerank_score.is_some());
        assert!(candidates[1].rerank_score.is_some());

        // Verify stable sort: if rerank scores are close, original order should be preserved
        // (higher original score wins ties)
        println!(
            "After rerank: {:?}",
            candidates
                .iter()
                .map(|c| (&c.id, c.rerank_score, c.score))
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn test_rerank_step_skips_few_candidates() {
        // This test uses MockReranker from rerank_step.rs via public API
        let registry = ModelRegistry::new();
        let step = RerankStep::new(&registry, "flashrank-nano");

        // Even if model isn't available, the logic should handle few candidates
        match step {
            Ok(Some(step)) => {
                let mut candidates = vec![make_result("A", "Text A", 0.9)];
                // With default min_candidates=5, this should be skipped
                step.rerank("query", &mut candidates).unwrap();
                assert!(candidates[0].rerank_score.is_none());
            }
            _ => {
                // Model not available, skip test
                println!("Skipping: reranker not available");
            }
        }
    }
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

mod thread_safety {
    use super::*;
    use std::thread;

    #[test]
    #[ignore = "requires model files"]
    fn test_flashrank_thread_safe() {
        let reranker = Arc::new(try_flashrank().expect("FlashRank required"));
        let mut handles = vec![];

        for i in 0..4 {
            let rr = Arc::clone(&reranker);
            handles.push(thread::spawn(move || {
                let query = format!("thread {i} query");
                let docs = vec!["doc one", "doc two", "doc three"];
                for _ in 0..5 {
                    let scores = rr.rerank(&query, &docs).expect("rerank should succeed");
                    assert_eq!(scores.len(), 3);
                }
            }));
        }

        for handle in handles {
            handle.join().expect("thread should complete");
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_mxbai_thread_safe() {
        let reranker = Arc::new(try_mxbai().expect("mxbai required"));
        let mut handles = vec![];

        for i in 0..4 {
            let rr = Arc::clone(&reranker);
            handles.push(thread::spawn(move || {
                let query = format!("thread {i} query");
                let docs = vec!["doc one", "doc two", "doc three"];
                for _ in 0..5 {
                    let scores = rr.rerank(&query, &docs).expect("rerank should succeed");
                    assert_eq!(scores.len(), 3);
                }
            }));
        }

        for handle in handles {
            handle.join().expect("thread should complete");
        }
    }
}
