//! Unit tests for static embedder implementations.
//!
//! Tests for static embedding backends:
//! - static-retrieval-mrl-en-v1 (ONNX-based with MRL support)
//! - Model2Vec potion-retrieval-32M (static lookup)
//!
//! Covers:
//! - Model loading and configuration
//! - Embedding quality (semantic similarity)
//! - MRL truncation and re-normalization
//! - Normalization verification
//! - Batch processing
//! - Edge cases (unicode, empty, long text)
//! - Determinism
//! - Performance baselines
//!
//! Bead: bd-3783

use xf::embedder::{Embedder, EmbedderResult, ModelCategory};
use xf::hash_embedder::HashEmbedder;
use xf::model_registry::{
    EMBEDDER_HASH_FNV1A_384, EMBEDDER_POTION_MULTI_128M, EMBEDDER_POTION_RETRIEVAL_32M,
    EMBEDDER_STATIC_MRL_EN_V1, EmbedderConfig, ModelRegistry,
};
use xf::model2vec_embedder::{MODEL_POTION_32M, MODEL_POTION_MULTI_128M, Model2VecEmbedder};
use xf::static_mrl_embedder::StaticMrlEmbedder;

/// Cosine similarity between two vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Check if a vector is approximately L2-normalized.
fn is_normalized(vec: &[f32], tolerance: f32) -> bool {
    let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    (norm - 1.0).abs() < tolerance
}

/// Helper to create an embedder from the registry.
fn create_embedder(model: &str) -> EmbedderResult<Box<dyn Embedder>> {
    let registry = ModelRegistry::new();
    let config = EmbedderConfig::new(model);
    registry.embedder(&config)
}

// =============================================================================
// Model Constants Tests (no model files required)
// =============================================================================

mod constants {
    use super::*;

    #[test]
    fn test_static_mrl_model_name() {
        assert_eq!(EMBEDDER_STATIC_MRL_EN_V1, "static-retrieval-mrl-en-v1");
    }

    #[test]
    fn test_model2vec_potion_model_name() {
        assert_eq!(EMBEDDER_POTION_RETRIEVAL_32M, "potion-retrieval-32M");
        assert_eq!(MODEL_POTION_32M, "potion-retrieval-32M");
    }

    #[test]
    fn test_model2vec_multilingual_model_name() {
        assert_eq!(EMBEDDER_POTION_MULTI_128M, "potion-multilingual-128M");
        assert_eq!(MODEL_POTION_MULTI_128M, "potion-multilingual-128M");
    }

    #[test]
    fn test_registry_has_static_models() {
        let registry = ModelRegistry::new();
        assert!(registry.has_embedder(EMBEDDER_STATIC_MRL_EN_V1));
        assert!(registry.has_embedder(EMBEDDER_POTION_RETRIEVAL_32M));
        assert!(registry.has_embedder(EMBEDDER_POTION_MULTI_128M));
    }

    #[test]
    fn test_hash_embedder_is_always_available() {
        let registry = ModelRegistry::new();
        assert!(registry.has_embedder(EMBEDDER_HASH_FNV1A_384));

        // Hash embedder should always work
        let embedder = create_embedder(EMBEDDER_HASH_FNV1A_384).unwrap();
        assert_eq!(embedder.dimension(), 384);
        assert!(!embedder.is_semantic());
    }

    #[test]
    fn test_static_model_info() {
        let registry = ModelRegistry::new();
        let models = registry.list_models();

        // Static MRL info
        let static_mrl = models
            .iter()
            .find(|m| m.name == EMBEDDER_STATIC_MRL_EN_V1)
            .unwrap();
        assert_eq!(static_mrl.native_dims, 1024);
        assert!(static_mrl.supports_mrl);
        assert_eq!(static_mrl.category, ModelCategory::StaticEmbedder);
        assert_eq!(static_mrl.backend, "onnx");

        // Model2Vec potion info
        let potion = models
            .iter()
            .find(|m| m.name == EMBEDDER_POTION_RETRIEVAL_32M)
            .unwrap();
        assert_eq!(potion.native_dims, 256);
        assert!(!potion.supports_mrl);
        assert_eq!(potion.category, ModelCategory::StaticEmbedder);
        assert_eq!(potion.backend, "model2vec");
    }

    #[test]
    fn test_static_embedder_category() {
        // Hash embedder
        let hash = HashEmbedder::default();
        assert_eq!(hash.category(), ModelCategory::StaticEmbedder);
        assert!(!hash.is_semantic());
    }
}

// =============================================================================
// Hash Embedder Tests (always available - no model files required)
// =============================================================================

mod hash_embedder {
    use super::*;

    #[test]
    fn test_hash_embedder_dimensions() {
        let embedder = HashEmbedder::new(256);
        assert_eq!(embedder.dimension(), 256);

        let embedder = HashEmbedder::default();
        assert_eq!(embedder.dimension(), 384);
    }

    #[test]
    fn test_hash_embedder_produces_normalized_vectors() {
        let embedder = HashEmbedder::default();
        let embedding = embedder.embed("hello world").unwrap();

        assert!(
            is_normalized(&embedding, 1e-5),
            "Hash embedding should be L2-normalized"
        );
    }

    #[test]
    fn test_hash_embedder_is_deterministic() {
        let embedder = HashEmbedder::default();
        let e1 = embedder.embed("determinism test").unwrap();
        let e2 = embedder.embed("determinism test").unwrap();
        assert_eq!(e1, e2, "Same text should produce identical embeddings");
    }

    #[test]
    fn test_hash_embedder_different_texts_differ() {
        let embedder = HashEmbedder::default();
        let e1 = embedder.embed("hello world").unwrap();
        let e2 = embedder.embed("goodbye world").unwrap();
        assert_ne!(
            e1, e2,
            "Different texts should produce different embeddings"
        );
    }

    #[test]
    fn test_hash_embedder_case_insensitive() {
        let embedder = HashEmbedder::default();
        let e1 = embedder.embed("Hello World").unwrap();
        let e2 = embedder.embed("hello world").unwrap();
        let e3 = embedder.embed("HELLO WORLD").unwrap();
        assert_eq!(e1, e2);
        assert_eq!(e2, e3);
    }

    #[test]
    fn test_hash_embedder_empty_text_returns_error() {
        let embedder = HashEmbedder::default();
        let result = embedder.embed("");
        assert!(result.is_err());
    }

    #[test]
    fn test_hash_embedder_no_valid_tokens() {
        let embedder = HashEmbedder::default();
        // Only single-char tokens, all filtered out
        let embedding = embedder.embed("a b c !").unwrap();
        assert_eq!(embedding.len(), 384);
        // Should still be normalized
        assert!(is_normalized(&embedding, 1e-4));
    }

    #[test]
    fn test_hash_embedder_unicode() {
        let embedder = HashEmbedder::default();
        let embedding = embedder.embed("日本語テスト café naïve").unwrap();
        assert_eq!(embedding.len(), 384);
        assert!(is_normalized(&embedding, 1e-5));
    }

    #[test]
    fn test_hash_embedder_batch() {
        let embedder = HashEmbedder::default();
        let texts = ["hello world", "rust programming", "machine learning"];
        let batch = embedder.embed_batch(&texts).unwrap();

        assert_eq!(batch.len(), 3);
        for (i, text) in texts.iter().enumerate() {
            let single = embedder.embed(text).unwrap();
            assert_eq!(
                batch[i], single,
                "Batch result should match individual embed"
            );
        }
    }

    #[test]
    fn test_hash_embedder_similar_texts_more_similar() {
        let embedder = HashEmbedder::default();
        let e_rust = embedder.embed("rust programming language").unwrap();
        let e_rust2 = embedder.embed("rust programming").unwrap();
        let e_python = embedder.embed("python scripting language").unwrap();

        let sim_rust_rust2 = cosine_similarity(&e_rust, &e_rust2);
        let sim_rust_python = cosine_similarity(&e_rust, &e_python);

        assert!(
            sim_rust_rust2 > sim_rust_python,
            "Related texts should be more similar: {} vs {}",
            sim_rust_rust2,
            sim_rust_python
        );
    }

    #[test]
    fn test_hash_embedder_does_not_support_mrl() {
        let embedder = HashEmbedder::default();
        assert!(!embedder.supports_mrl());
    }

    #[test]
    fn test_hash_embedder_info() {
        let embedder = HashEmbedder::new(256);
        let info = embedder.info();

        assert_eq!(info.id, "fnv1a-256");
        assert_eq!(info.dimension, 256);
        assert!(!info.is_semantic);
        assert!(!info.supports_mrl);
        assert_eq!(info.category, ModelCategory::StaticEmbedder);
    }
}

// =============================================================================
// Model2Vec Tensor Parsing Tests (no model files required)
// =============================================================================

mod model2vec_parsing {
    use super::*;

    #[test]
    fn test_model_search_paths_not_empty() {
        let paths = Model2VecEmbedder::model_search_paths(MODEL_POTION_32M);
        assert!(!paths.is_empty(), "Should have search paths");
        assert!(paths.iter().any(|p| p.to_string_lossy().contains("xf")));
    }

    #[test]
    fn test_model_search_paths_include_huggingface() {
        let paths = Model2VecEmbedder::model_search_paths(MODEL_POTION_32M);
        assert!(paths.iter().any(|p| {
            let s = p.to_string_lossy();
            s.contains("huggingface") || s.contains("models")
        }));
    }
}

// =============================================================================
// Static MRL Search Paths Tests (no model files required)
// =============================================================================

mod static_mrl_paths {
    use super::*;

    #[test]
    fn test_static_mrl_search_paths_not_empty() {
        let paths = StaticMrlEmbedder::model_search_paths();
        assert!(!paths.is_empty(), "Should have search paths");
    }
}

// =============================================================================
// Model Loading Tests (require model files)
// =============================================================================

mod loading {
    use super::*;

    #[test]
    #[ignore = "requires model files"]
    fn test_static_mrl_loads() {
        let embedder = StaticMrlEmbedder::try_load().expect("model should load");
        assert_eq!(embedder.model_name(), "static-retrieval-mrl-en-v1");
        assert!(embedder.supports_mrl());
        assert!(embedder.is_semantic());
        assert_eq!(embedder.category(), ModelCategory::StaticEmbedder);
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_static_mrl_dimensions() {
        let embedder = StaticMrlEmbedder::try_load().expect("model should load");
        assert_eq!(embedder.dimension(), 1024);
        assert_eq!(embedder.native_dimension(), 1024);
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_model2vec_loads() {
        let embedder = Model2VecEmbedder::try_load(MODEL_POTION_32M).expect("model should load");
        assert_eq!(embedder.model_name(), MODEL_POTION_32M);
        assert_eq!(embedder.dimension(), 256);
        assert!(!embedder.supports_mrl());
        assert!(embedder.is_semantic()); // Model2Vec uses distilled semantics
    }

    #[test]
    fn test_model_not_found_gives_clear_error() {
        // Try to load a model that doesn't exist
        let result = Model2VecEmbedder::try_load("nonexistent-model-xyz");
        assert!(result.is_err());
        let err = result.unwrap_err();
        let err_str = err.to_string();
        assert!(
            err_str.contains("not found") || err_str.contains("unavailable"),
            "Error should indicate model not found: {}",
            err_str
        );
    }
}

// =============================================================================
// Embedding Quality Tests (require model files)
// =============================================================================

mod quality {
    use super::*;

    #[test]
    #[ignore = "requires model files"]
    fn test_static_mrl_semantic_similarity() {
        let embedder = StaticMrlEmbedder::try_load().unwrap();

        let q = embedder.embed("coffee espresso latte").unwrap();
        let d_good = embedder.embed("best cafe downtown morning brew").unwrap();
        let d_bad = embedder.embed("quantum mechanics wave function").unwrap();

        let sim_good = cosine_similarity(&q, &d_good);
        let sim_bad = cosine_similarity(&q, &d_bad);

        assert!(
            sim_good > sim_bad,
            "Related texts should be more similar: {} vs {}",
            sim_good,
            sim_bad
        );
        assert!(sim_good > 0.4, "Related texts should have >0.4 similarity");
        assert!(sim_bad < 0.6, "Unrelated texts should have <0.6 similarity");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_model2vec_semantic_similarity() {
        let embedder = Model2VecEmbedder::try_load(MODEL_POTION_32M).unwrap();

        let q = embedder.embed("programming languages").unwrap();
        let d_good = embedder.embed("software development coding").unwrap();
        let d_bad = embedder.embed("tropical fish aquarium care").unwrap();

        let sim_good = cosine_similarity(&q, &d_good);
        let sim_bad = cosine_similarity(&q, &d_bad);

        assert!(
            sim_good > sim_bad,
            "Model2Vec should rank relevant higher: {} vs {}",
            sim_good,
            sim_bad
        );
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_embeddings_are_normalized() {
        let mrl = StaticMrlEmbedder::try_load().unwrap();
        let m2v = Model2VecEmbedder::try_load(MODEL_POTION_32M).unwrap();

        for embedder in [&mrl as &dyn Embedder, &m2v as &dyn Embedder] {
            let embedding = embedder.embed("test text for normalization").unwrap();
            assert!(
                is_normalized(&embedding, 0.01),
                "{}: embedding should be L2-normalized",
                embedder.model_name()
            );
        }
    }
}

// =============================================================================
// MRL Truncation Tests (require model files) - CRITICAL
// =============================================================================

mod mrl {
    use super::*;

    #[test]
    #[ignore = "requires model files"]
    fn test_mrl_truncation_produces_correct_length() {
        let embedder = StaticMrlEmbedder::try_load().unwrap();
        let full = embedder.embed("test sentence for MRL").unwrap();
        assert_eq!(full.len(), 1024);

        for dims in [64, 128, 256, 512, 768] {
            let truncated = embedder.truncate_embedding(&full, dims).unwrap();
            assert_eq!(truncated.len(), dims, "Truncated should be {} dims", dims);
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_mrl_renormalization_after_truncation() {
        // CRITICAL TEST: After MRL truncation, vectors MUST be re-normalized.
        let embedder = StaticMrlEmbedder::try_load().unwrap();
        let full = embedder.embed("test re-normalization").unwrap();

        // Verify full embedding is normalized
        assert!(
            is_normalized(&full, 0.01),
            "Full embedding should be normalized"
        );

        for dims in [64, 128, 256, 512, 768] {
            let truncated = embedder.truncate_embedding(&full, dims).unwrap();
            assert!(
                is_normalized(&truncated, 0.01),
                "Truncated to {} dims: must be re-normalized after MRL truncation",
                dims
            );
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_mrl_quality_degrades_gracefully() {
        let embedder = StaticMrlEmbedder::try_load().unwrap();
        let q = embedder.embed("coffee espresso latte").unwrap();
        let d1 = embedder.embed("best cafe downtown morning brew").unwrap();
        let d2 = embedder.embed("quantum mechanics wave function").unwrap();

        // Quality should degrade gracefully with fewer dimensions
        // but relevance ordering should be preserved
        for dims in [64, 128, 256, 512, 768, 1024] {
            let qt = embedder.truncate_embedding(&q, dims).unwrap();
            let d1t = embedder.truncate_embedding(&d1, dims).unwrap();
            let d2t = embedder.truncate_embedding(&d2, dims).unwrap();

            let sim_good = cosine_similarity(&qt, &d1t);
            let sim_bad = cosine_similarity(&qt, &d2t);

            assert!(
                sim_good > sim_bad,
                "Relevance ordering broken at dims={}: {} vs {}",
                dims,
                sim_good,
                sim_bad
            );
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_mrl_batch_truncation() {
        let embedder = StaticMrlEmbedder::try_load().unwrap();
        let texts = ["hello world", "foo bar", "test sentence"];
        let text_refs: Vec<&str> = texts.iter().map(|s| *s).collect();
        let full = embedder.embed_batch(&text_refs).unwrap();

        let truncated = embedder.truncate_batch(&full, 256).unwrap();
        assert_eq!(truncated.len(), 3);
        for (i, t) in truncated.iter().enumerate() {
            assert_eq!(t.len(), 256, "Batch item {} should be 256 dims", i);
            assert!(
                is_normalized(t, 0.01),
                "Batch item {} should be normalized after truncation",
                i
            );
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_mrl_invalid_dims_rejected() {
        let embedder = StaticMrlEmbedder::try_load().unwrap();
        let full = embedder.embed("test").unwrap();

        // Requesting more dims than available should error
        let result = embedder.truncate_embedding(&full, 2048);
        assert!(result.is_err(), "Should reject dims > model dimensions");

        // Requesting 0 dims should error
        let result = embedder.truncate_embedding(&full, 0);
        assert!(result.is_err(), "Should reject 0 dims");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_model2vec_does_not_support_mrl() {
        let embedder = Model2VecEmbedder::try_load(MODEL_POTION_32M).unwrap();
        assert!(!embedder.supports_mrl());

        let full = embedder.embed("test").unwrap();
        let result = embedder.truncate_embedding(&full, 128);
        assert!(result.is_err(), "Model2Vec should reject truncation");
    }
}

// =============================================================================
// Batch Processing Tests (require model files)
// =============================================================================

mod batch {
    use super::*;

    #[test]
    #[ignore = "requires model files"]
    fn test_static_mrl_batch_matches_individual() {
        let embedder = StaticMrlEmbedder::try_load().unwrap();
        let texts = ["hello world", "foo bar", "test three"];
        let text_refs: Vec<&str> = texts.iter().map(|s| *s).collect();

        let batch = embedder.embed_batch(&text_refs).unwrap();
        for (i, text) in texts.iter().enumerate() {
            let single = embedder.embed(text).unwrap();
            let diff: f32 = batch[i]
                .iter()
                .zip(single.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(
                diff < 0.001,
                "Batch and individual should match at index {}, diff={}",
                i,
                diff
            );
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_model2vec_batch_matches_individual() {
        let embedder = Model2VecEmbedder::try_load(MODEL_POTION_32M).unwrap();
        let texts = ["hello world", "foo bar", "test three"];
        let text_refs: Vec<&str> = texts.iter().map(|s| *s).collect();

        let batch = embedder.embed_batch(&text_refs).unwrap();
        for (i, text) in texts.iter().enumerate() {
            let single = embedder.embed(text).unwrap();
            let diff: f32 = batch[i]
                .iter()
                .zip(single.iter())
                .map(|(a, b)| (a - b).abs())
                .sum();
            assert!(
                diff < 0.001,
                "Batch and individual should match at index {}, diff={}",
                i,
                diff
            );
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_empty_batch() {
        let embedder = StaticMrlEmbedder::try_load().unwrap();
        let results = embedder.embed_batch(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_large_batch() {
        let embedder = StaticMrlEmbedder::try_load().unwrap();
        let texts: Vec<String> = (0..50).map(|i| format!("document number {}", i)).collect();
        let text_refs: Vec<&str> = texts.iter().map(|s| s.as_str()).collect();
        let results = embedder.embed_batch(&text_refs).unwrap();
        assert_eq!(results.len(), 50);
    }
}

// =============================================================================
// Edge Cases Tests (require model files)
// =============================================================================

mod edge_cases {
    use super::*;

    #[test]
    #[ignore = "requires model files"]
    fn test_empty_text_handled() {
        let embedder = StaticMrlEmbedder::try_load().unwrap();
        let result = embedder.embed("");
        assert!(result.is_err(), "Empty text should return error");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_very_long_text() {
        let embedder = StaticMrlEmbedder::try_load().unwrap();
        let long_text = "word ".repeat(10000);
        let result = embedder.embed(&long_text);
        assert!(result.is_ok(), "Long text should not panic");
        assert_eq!(result.unwrap().len(), embedder.dimension());
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_emoji_handling() {
        let embedder = StaticMrlEmbedder::try_load().unwrap();
        let result = embedder.embed("Great day! 🌞☕🎉").unwrap();
        assert_eq!(result.len(), embedder.dimension());
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_cjk_characters() {
        let embedder = StaticMrlEmbedder::try_load().unwrap();
        let result = embedder.embed("日本語テスト 中文测试").unwrap();
        assert_eq!(result.len(), embedder.dimension());
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_special_characters() {
        let embedder = StaticMrlEmbedder::try_load().unwrap();
        for text in ["<script>alert(1)</script>", "\n\t\r", "test\0text"] {
            let result = embedder.embed(text);
            assert!(result.is_ok(), "Should handle special chars: {:?}", text);
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_model2vec_unknown_tokens() {
        let embedder = Model2VecEmbedder::try_load(MODEL_POTION_32M).unwrap();
        // Made-up words should still produce valid embeddings
        let result = embedder.embed("xyzzy plugh zork");
        assert!(result.is_ok(), "Unknown tokens should be handled");
        let embedding = result.unwrap();
        assert_eq!(embedding.len(), 256);
        assert!(is_normalized(&embedding, 0.1));
    }
}

// =============================================================================
// Determinism Tests (require model files)
// =============================================================================

mod determinism {
    use super::*;

    #[test]
    #[ignore = "requires model files"]
    fn test_static_mrl_deterministic() {
        let embedder = StaticMrlEmbedder::try_load().unwrap();
        let text = "determinism test";
        let e1 = embedder.embed(text).unwrap();
        let e2 = embedder.embed(text).unwrap();
        assert_eq!(e1, e2, "Same text should always produce same embedding");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_model2vec_deterministic() {
        let embedder = Model2VecEmbedder::try_load(MODEL_POTION_32M).unwrap();
        let text = "determinism for model2vec";
        let e1 = embedder.embed(text).unwrap();
        let e2 = embedder.embed(text).unwrap();
        assert_eq!(e1, e2);
    }
}

// =============================================================================
// Performance Baseline Tests (require model files)
// =============================================================================

mod performance {
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore = "requires model files"]
    fn test_model2vec_is_fast() {
        let embedder = Model2VecEmbedder::try_load(MODEL_POTION_32M).unwrap();

        // Warmup
        for _ in 0..5 {
            embedder.embed("warmup").unwrap();
        }

        let start = Instant::now();
        for _ in 0..1000 {
            embedder.embed("static lookup speed test").unwrap();
        }
        let avg_us = start.elapsed().as_micros() as f64 / 1000.0;

        // Model2Vec should be very fast (sub-millisecond)
        assert!(
            avg_us < 1000.0,
            "Model2Vec should be <1ms per embed, got {}us",
            avg_us
        );
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_static_mrl_latency() {
        let embedder = StaticMrlEmbedder::try_load().unwrap();

        // Warmup
        for _ in 0..5 {
            embedder.embed("warmup").unwrap();
        }

        let start = Instant::now();
        for _ in 0..100 {
            embedder
                .embed("benchmark latency measurement text")
                .unwrap();
        }
        let avg_ms = start.elapsed().as_millis() as f64 / 100.0;

        // Static MRL uses ONNX but should still be fast
        assert!(
            avg_ms < 50.0,
            "static-mrl should be <50ms per embed, got {}ms",
            avg_ms
        );
    }
}

// =============================================================================
// Registry Integration Tests
// =============================================================================

mod registry {
    use super::*;

    #[test]
    fn test_static_embedders_in_registry() {
        let registry = ModelRegistry::new();
        let names = registry.embedder_names();

        assert!(names.contains(&EMBEDDER_HASH_FNV1A_384));
        assert!(names.contains(&EMBEDDER_STATIC_MRL_EN_V1));
        assert!(names.contains(&EMBEDDER_POTION_RETRIEVAL_32M));
        assert!(names.contains(&EMBEDDER_POTION_MULTI_128M));
    }

    #[test]
    fn test_canonical_name_resolution() {
        let registry = ModelRegistry::new();

        // Various aliases should resolve to the same model
        assert!(registry.has_embedder("hash"));
        assert!(registry.has_embedder("fnv1a"));
        assert!(registry.has_embedder("fnv1a-384"));
        assert!(registry.has_embedder("hash-fnv1a-384"));

        assert!(registry.has_embedder("potion-retrieval-32m"));
        assert!(registry.has_embedder("minishlab/potion-retrieval-32M"));

        assert!(registry.has_embedder("static-retrieval-mrl-en-v1"));
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_registry_creates_embedder() {
        let registry = ModelRegistry::new();

        // Hash embedder should always work
        let hash = registry.embedder_by_name("hash").unwrap();
        assert_eq!(hash.category(), ModelCategory::StaticEmbedder);
        assert!(!hash.is_semantic());

        // If static-mrl is available
        if let Ok(mrl) = registry.embedder_by_name(EMBEDDER_STATIC_MRL_EN_V1) {
            assert_eq!(mrl.category(), ModelCategory::StaticEmbedder);
            assert!(mrl.supports_mrl());
        }

        // If model2vec is available
        if let Ok(m2v) = registry.embedder_by_name(EMBEDDER_POTION_RETRIEVAL_32M) {
            assert_eq!(m2v.category(), ModelCategory::StaticEmbedder);
            assert!(!m2v.supports_mrl());
        }
    }
}

// =============================================================================
// Model2Vec Mean Pooling Tests (require model files)
// =============================================================================

mod model2vec_specific {
    use super::*;

    #[test]
    #[ignore = "requires model files"]
    fn test_model2vec_mean_pooling_order_invariance() {
        let embedder = Model2VecEmbedder::try_load(MODEL_POTION_32M).unwrap();

        // Same words in different order should produce similar embeddings
        // (mean pooling is order-invariant for bag-of-words)
        let e1 = embedder.embed("hello world foo").unwrap();
        let e2 = embedder.embed("world foo hello").unwrap();

        let sim = cosine_similarity(&e1, &e2);
        // Note: tokenization may differ slightly, so not exact equality
        // but they should be very similar
        assert!(
            sim > 0.9,
            "Mean pooling should be mostly order-invariant: {}",
            sim
        );
    }
}
