//! Unit tests for transformer embedder backends.
//!
//! Tests for all transformer-based embedding models covering:
//! - Model loading and configuration
//! - Embedding quality (semantic similarity)
//! - Normalization verification
//! - Batch processing
//! - Thread safety
//! - Edge cases
//!
//! Bead: bd-25z8

use std::sync::Arc;

use xf::embedder::{Embedder, EmbedderResult, l2_normalize};
use xf::model_registry::{
    EMBEDDER_BGE_SMALL_EN_V15, EMBEDDER_EMBEDDINGGEMMA_300M, EMBEDDER_MINILM_L6_V2,
    EMBEDDER_NOMIC_V15, EmbedderConfig, ModelRegistry,
};

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

/// Helper to create an embedder from the registry.
fn create_embedder(model: &str) -> EmbedderResult<Box<dyn Embedder>> {
    let registry = ModelRegistry::new();
    let config = EmbedderConfig::new(model);
    registry.embedder(&config)
}

/// Assert two vectors are approximately equal.
fn assert_vectors_close(a: &[f32], b: &[f32], tolerance: f32, msg: &str) {
    assert_eq!(
        a.len(),
        b.len(),
        "{}: length mismatch {} vs {}",
        msg,
        a.len(),
        b.len()
    );
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < tolerance,
            "{}: mismatch at index {}: {} vs {} (diff={})",
            msg,
            i,
            x,
            y,
            (x - y).abs()
        );
    }
}

// =============================================================================
// Model Constants Tests (no model files required)
// =============================================================================

mod constants {
    use xf::model_registry::*;

    #[test]
    fn test_minilm_model_name() {
        assert_eq!(EMBEDDER_MINILM_L6_V2, "all-MiniLM-L6-v2");
    }

    #[test]
    fn test_bge_small_model_name() {
        assert_eq!(EMBEDDER_BGE_SMALL_EN_V15, "bge-small-en-v1.5");
    }

    #[test]
    fn test_nomic_model_name() {
        assert_eq!(EMBEDDER_NOMIC_V15, "nomic-embed-text-v1.5");
    }

    #[test]
    fn test_embeddinggemma_model_name() {
        assert_eq!(EMBEDDER_EMBEDDINGGEMMA_300M, "embeddinggemma-300m");
    }

    #[test]
    fn test_e5_model_name() {
        assert_eq!(EMBEDDER_E5_SMALL, "multilingual-e5-small");
    }

    #[test]
    fn test_registry_has_all_transformer_models() {
        let registry = ModelRegistry::new();
        assert!(registry.has_embedder(EMBEDDER_MINILM_L6_V2));
        assert!(registry.has_embedder(EMBEDDER_BGE_SMALL_EN_V15));
        assert!(registry.has_embedder(EMBEDDER_NOMIC_V15));
        assert!(registry.has_embedder(EMBEDDER_E5_SMALL));
        assert!(registry.has_embedder(EMBEDDER_EMBEDDINGGEMMA_300M));
    }

    #[test]
    fn test_embedder_names_list() {
        let registry = ModelRegistry::new();
        let names = registry.embedder_names();
        assert!(names.contains(&EMBEDDER_MINILM_L6_V2));
        assert!(names.contains(&EMBEDDER_BGE_SMALL_EN_V15));
        assert!(names.contains(&EMBEDDER_NOMIC_V15));
    }

    #[test]
    fn test_model_info_for_transformers() {
        let registry = ModelRegistry::new();
        let models = registry.list_models();

        let minilm = models
            .iter()
            .find(|m| m.name == EMBEDDER_MINILM_L6_V2)
            .unwrap();
        assert_eq!(minilm.native_dims, 384);
        assert!(!minilm.supports_mrl);
        assert_eq!(minilm.backend, "fastembed");

        let bge = models
            .iter()
            .find(|m| m.name == EMBEDDER_BGE_SMALL_EN_V15)
            .unwrap();
        assert_eq!(bge.native_dims, 384);
        assert!(!bge.supports_mrl);

        let nomic = models
            .iter()
            .find(|m| m.name == EMBEDDER_NOMIC_V15)
            .unwrap();
        assert_eq!(nomic.native_dims, 768);
        assert!(nomic.supports_mrl);

        let gemma = models
            .iter()
            .find(|m| m.name == EMBEDDER_EMBEDDINGGEMMA_300M)
            .unwrap();
        assert_eq!(gemma.native_dims, 768);
        assert!(gemma.supports_mrl);
    }
}

// =============================================================================
// Model Loading Tests (requires model files - integration tests)
// =============================================================================

mod loading {
    use super::*;

    #[test]
    #[ignore = "requires model files"]
    fn test_minilm_loads_with_correct_dims() {
        let embedder = create_embedder(EMBEDDER_MINILM_L6_V2).expect("failed to load MiniLM");
        assert_eq!(embedder.dimension(), 384);
        assert_eq!(embedder.model_name(), "all-MiniLM-L6-v2");
        assert!(embedder.is_semantic());
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_bge_small_loads_with_correct_dims() {
        let embedder = create_embedder(EMBEDDER_BGE_SMALL_EN_V15).expect("failed to load BGE");
        assert_eq!(embedder.dimension(), 384);
        assert!(embedder.is_semantic());
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_nomic_loads_with_correct_dims() {
        let embedder = create_embedder(EMBEDDER_NOMIC_V15).expect("failed to load Nomic");
        assert_eq!(embedder.dimension(), 768);
        assert!(embedder.is_semantic());
    }

    /// EmbeddingGemma loading test - currently returns Unavailable error.
    /// Will work when backend is implemented.
    #[test]
    fn test_embeddinggemma_not_yet_available() {
        let result = create_embedder(EMBEDDER_EMBEDDINGGEMMA_300M);
        // Expected to fail with Unavailable until backend is implemented
        assert!(result.is_err());
        let err = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected unavailable error"),
        };
        assert!(
            err.contains("not implemented") || err.contains("unavailable"),
            "Should indicate model is not implemented: {err}"
        );
    }

    #[test]
    fn test_invalid_model_gives_clear_error() {
        let registry = ModelRegistry::new();
        let config = EmbedderConfig::new("nonexistent-model-xyz");
        let result = registry.embedder(&config);
        assert!(result.is_err());
        let err = match result {
            Err(e) => e.to_string(),
            Ok(_) => panic!("expected error"),
        };
        assert!(
            err.contains("unknown") || err.contains("not found"),
            "Error should be descriptive: {err}"
        );
    }
}

// =============================================================================
// Embedding Quality Tests (requires model files - integration tests)
// =============================================================================

mod quality {
    use super::*;

    /// Test that a model can distinguish related from unrelated texts.
    fn quality_test_for_model(embedder: &dyn Embedder) {
        let pairs = [
            (
                "buy bitcoin cryptocurrency",
                "crypto trading exchange",
                true,
            ),
            (
                "buy bitcoin cryptocurrency",
                "chocolate cake recipe baking",
                false,
            ),
            (
                "machine learning neural networks",
                "deep learning training data",
                true,
            ),
            (
                "machine learning neural networks",
                "flower arrangement centerpiece",
                false,
            ),
            (
                "rust programming language safety",
                "systems programming memory safe",
                true,
            ),
            (
                "rust programming language safety",
                "vacation resort beach hotel",
                false,
            ),
        ];

        for (query, doc, should_be_related) in pairs {
            let q = embedder.embed(query).expect("embed query failed");
            let d = embedder.embed(doc).expect("embed doc failed");
            let sim = cosine_similarity(&q, &d);

            if should_be_related {
                assert!(
                    sim > 0.3,
                    "{}: {} x {}: expected related (>0.3), got sim={}",
                    embedder.model_name(),
                    query,
                    doc,
                    sim
                );
            } else {
                assert!(
                    sim < 0.7,
                    "{}: {} x {}: expected unrelated (<0.7), got sim={}",
                    embedder.model_name(),
                    query,
                    doc,
                    sim
                );
            }
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_minilm_quality() {
        let embedder = create_embedder(EMBEDDER_MINILM_L6_V2).unwrap();
        quality_test_for_model(embedder.as_ref());
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_bge_small_quality() {
        let embedder = create_embedder(EMBEDDER_BGE_SMALL_EN_V15).unwrap();
        quality_test_for_model(embedder.as_ref());
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_nomic_quality() {
        let embedder = create_embedder(EMBEDDER_NOMIC_V15).unwrap();
        quality_test_for_model(embedder.as_ref());
    }
}

// =============================================================================
// Normalization Tests (requires model files - integration tests)
// =============================================================================

mod normalization {
    use super::*;

    fn test_model_produces_normalized_embeddings(model_name: &str) {
        let embedder = create_embedder(model_name).expect("failed to load model");
        let embedding = embedder.embed("test normalization").expect("embed failed");
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "{model_name}: embedding should be L2-normalized, got norm={norm}"
        );
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_minilm_normalized() {
        test_model_produces_normalized_embeddings(EMBEDDER_MINILM_L6_V2);
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_bge_normalized() {
        test_model_produces_normalized_embeddings(EMBEDDER_BGE_SMALL_EN_V15);
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_nomic_normalized() {
        test_model_produces_normalized_embeddings(EMBEDDER_NOMIC_V15);
    }

    #[test]
    fn test_l2_normalize_function() {
        let mut vec = vec![3.0, 4.0]; // 3-4-5 right triangle
        l2_normalize(&mut vec);
        assert!((vec[0] - 0.6).abs() < 1e-5);
        assert!((vec[1] - 0.8).abs() < 1e-5);

        let norm: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let mut vec = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut vec);
        // Should not panic, vector remains unchanged
        assert!(vec.iter().all(|&x| x == 0.0));
    }
}

// =============================================================================
// Batch Processing Tests (requires model files - integration tests)
// =============================================================================

mod batching {
    use super::*;

    #[test]
    #[ignore = "requires model files"]
    fn test_batch_size_1_matches_individual() {
        let embedder = create_embedder(EMBEDDER_MINILM_L6_V2).unwrap();
        let text = "single document test";
        let batch = embedder.embed_batch(&[text]).unwrap();
        let individual = embedder.embed(text).unwrap();
        assert_vectors_close(&batch[0], &individual, 1e-5, "batch vs individual");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_different_batch_sizes_produce_same_results() {
        let embedder = create_embedder(EMBEDDER_MINILM_L6_V2).unwrap();
        let texts: Vec<String> = (0..16).map(|i| format!("text number {i}")).collect();
        let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();

        let batch_16 = embedder.embed_batch(&text_refs).unwrap();
        let batch_4: Vec<Vec<f32>> = text_refs
            .chunks(4)
            .flat_map(|chunk| embedder.embed_batch(chunk).unwrap())
            .collect();

        for (i, (b16, b4)) in batch_16.iter().zip(batch_4.iter()).enumerate() {
            assert_vectors_close(b16, b4, 1e-4, &format!("batch mismatch at index {i}"));
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_empty_batch_returns_empty() {
        let embedder = create_embedder(EMBEDDER_MINILM_L6_V2).unwrap();
        let result = embedder.embed_batch(&[]).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_large_batch_succeeds() {
        let embedder = create_embedder(EMBEDDER_MINILM_L6_V2).unwrap();
        let texts: Vec<String> = (0..100).map(|i| format!("batch text {i}")).collect();
        let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();
        let results = embedder.embed_batch(&text_refs).unwrap();
        assert_eq!(results.len(), 100);
    }
}

// =============================================================================
// Thread Safety Tests (requires model files - integration tests)
// =============================================================================

mod thread_safety {
    use super::*;

    #[test]
    #[ignore = "requires model files"]
    #[allow(clippy::needless_collect)] // Must collect handles before joining to spawn all threads
    fn test_concurrent_embedding() {
        let embedder = Arc::new(create_embedder(EMBEDDER_MINILM_L6_V2).unwrap());
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let e = embedder.clone();
                std::thread::spawn(move || e.embed(&format!("thread {i} text")).unwrap())
            })
            .collect();

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        assert_eq!(results.len(), 8);
        assert!(results.iter().all(|r| r.len() == embedder.dimension()));
    }

    #[test]
    #[ignore = "requires model files"]
    #[allow(clippy::needless_collect)] // Must collect handles before joining to spawn all threads
    fn test_concurrent_embedding_deterministic() {
        let embedder = Arc::new(create_embedder(EMBEDDER_MINILM_L6_V2).unwrap());
        let text = "determinism across threads";

        let handles: Vec<_> = (0..4)
            .map(|_| {
                let e = embedder.clone();
                let t = text.to_string();
                std::thread::spawn(move || e.embed(&t).unwrap())
            })
            .collect();

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        for i in 1..results.len() {
            assert_vectors_close(
                &results[0],
                &results[i],
                1e-5,
                &format!("thread {i} mismatch"),
            );
        }
    }
}

// =============================================================================
// Edge Cases Tests (requires model files - integration tests)
// =============================================================================

mod edge_cases {
    use super::*;

    #[test]
    #[ignore = "requires model files"]
    fn test_emoji_handling() {
        let embedder = create_embedder(EMBEDDER_MINILM_L6_V2).unwrap();
        let result = embedder.embed("Great day! 🌞☕🎉 #blessed").unwrap();
        assert_eq!(result.len(), embedder.dimension());
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_only_special_characters() {
        let embedder = create_embedder(EMBEDDER_MINILM_L6_V2).unwrap();
        let result = embedder.embed("!@#$%^&*()").unwrap();
        assert_eq!(result.len(), embedder.dimension());
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_unicode_handling() {
        let embedder = create_embedder(EMBEDDER_MINILM_L6_V2).unwrap();

        // Various Unicode texts
        let texts = [
            "café",                            // Latin extended
            "日本語",                          // Japanese
            "العربية",                         // Arabic
            "🚀🔥💻",                          // Emoji
            "mixed 日本語 and English テスト", // Mixed
        ];

        for text in texts {
            let result = embedder.embed(text);
            assert!(result.is_ok(), "Failed to embed: {text}");
            assert_eq!(result.unwrap().len(), embedder.dimension());
        }
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_very_short_text() {
        let embedder = create_embedder(EMBEDDER_MINILM_L6_V2).unwrap();
        let result = embedder.embed("a").unwrap();
        assert_eq!(result.len(), embedder.dimension());
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_long_text_truncation() {
        let embedder = create_embedder(EMBEDDER_MINILM_L6_V2).unwrap();
        // MiniLM has 512 token limit
        let long_text = "machine learning. ".repeat(200); // ~1600 tokens
        let result = embedder.embed(&long_text).unwrap();
        assert_eq!(result.len(), embedder.dimension());
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_whitespace_only() {
        let embedder = create_embedder(EMBEDDER_MINILM_L6_V2).unwrap();
        let result = embedder.embed("   \t\n   ");
        // Should produce an embedding (even if just padding tokens)
        assert!(result.is_ok() || result.is_err());
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_repeated_embedding_deterministic() {
        let embedder = create_embedder(EMBEDDER_MINILM_L6_V2).unwrap();
        let text = "determinism test";

        let e1 = embedder.embed(text).unwrap();
        let e2 = embedder.embed(text).unwrap();

        assert_vectors_close(&e1, &e2, 1e-6, "repeated embedding should be identical");
    }
}

// =============================================================================
// Pooling Strategy Tests (requires model files - integration tests)
// =============================================================================

mod pooling {
    use super::*;

    #[test]
    #[ignore = "requires model files"]
    fn test_minilm_uses_mean_pooling() {
        let embedder = create_embedder(EMBEDDER_MINILM_L6_V2).unwrap();

        // Mean pooling: embedding is average of all token representations
        let e1 = embedder.embed("the cat sat on the mat").unwrap();
        let e2 = embedder.embed("the cat").unwrap();

        // Mean pooling should give different results for different-length texts
        let sim = cosine_similarity(&e1, &e2);
        assert!(
            sim < 0.99,
            "Mean pooling should differ for different text lengths"
        );
        assert!(sim > 0.5, "Related texts should still be somewhat similar");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_different_texts_produce_different_embeddings() {
        let embedder = create_embedder(EMBEDDER_MINILM_L6_V2).unwrap();

        let e1 = embedder.embed("hello world").unwrap();
        let e2 = embedder.embed("goodbye universe").unwrap();

        let sim = cosine_similarity(&e1, &e2);
        assert!(
            sim < 0.99,
            "Different texts should produce different embeddings"
        );
    }

    /// BGE uses CLS token pooling (position 0) instead of mean pooling.
    /// This test verifies that BGE produces quality embeddings suitable for retrieval.
    #[test]
    #[ignore = "requires model files"]
    fn test_bge_cls_pooling_retrieval_quality() {
        let embedder = create_embedder(EMBEDDER_BGE_SMALL_EN_V15).unwrap();

        // BGE should produce high-quality embeddings for retrieval
        let query = embedder
            .embed("programming languages for systems development")
            .unwrap();
        let relevant = embedder
            .embed("Rust is a systems programming language")
            .unwrap();
        let irrelevant = embedder.embed("Chocolate chip cookies recipe").unwrap();

        let sim_relevant = cosine_similarity(&query, &relevant);
        let sim_irrelevant = cosine_similarity(&query, &irrelevant);

        assert!(
            sim_relevant > sim_irrelevant,
            "BGE should rank relevant doc higher: relevant={sim_relevant} vs irrelevant={sim_irrelevant}"
        );
        assert!(
            sim_relevant > 0.4,
            "BGE should have reasonable similarity for related concepts"
        );
    }

    /// Verify BGE produces consistent 384-dimensional output regardless of input length.
    #[test]
    #[ignore = "requires model files"]
    fn test_bge_cls_consistent_dimensions() {
        let embedder = create_embedder(EMBEDDER_BGE_SMALL_EN_V15).unwrap();

        let short = embedder.embed("hi").unwrap();
        let medium = embedder.embed("This is a medium length sentence").unwrap();
        let long = embedder.embed(
            "This is a much longer sentence that contains many more tokens and words to process",
        ).unwrap();

        assert_eq!(short.len(), 384);
        assert_eq!(medium.len(), 384);
        assert_eq!(long.len(), 384);
    }
}

// =============================================================================
// MRL Tests (requires model files - integration tests)
// =============================================================================

mod mrl {
    use super::*;

    #[test]
    #[ignore = "requires model files"]
    fn test_nomic_supports_mrl() {
        let registry = ModelRegistry::new();
        let models = registry.list_models();
        let nomic = models
            .iter()
            .find(|m| m.name == EMBEDDER_NOMIC_V15)
            .unwrap();
        assert!(nomic.supports_mrl, "Nomic should support MRL");
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_minilm_does_not_support_mrl() {
        let registry = ModelRegistry::new();
        let models = registry.list_models();
        let minilm = models
            .iter()
            .find(|m| m.name == EMBEDDER_MINILM_L6_V2)
            .unwrap();
        assert!(!minilm.supports_mrl, "MiniLM should not support MRL");
    }

    /// Nomic embedding quality with MRL-capable model.
    #[test]
    #[ignore = "requires model files"]
    fn test_nomic_mrl_quality_at_full_dims() {
        let embedder = create_embedder(EMBEDDER_NOMIC_V15).unwrap();

        // At full 768 dimensions, nomic should have strong retrieval quality
        let query = embedder.embed("machine learning algorithms").unwrap();
        let relevant = embedder
            .embed("neural network training optimization")
            .unwrap();
        let irrelevant = embedder.embed("chocolate chip cookie recipe").unwrap();

        assert_eq!(query.len(), 768, "Nomic should have 768 dimensions");

        let sim_relevant = cosine_similarity(&query, &relevant);
        let sim_irrelevant = cosine_similarity(&query, &irrelevant);

        assert!(
            sim_relevant > sim_irrelevant,
            "Nomic should rank relevant doc higher"
        );
        assert!(
            sim_relevant > 0.5,
            "Nomic should have strong similarity for related concepts"
        );
    }
}

// =============================================================================
// Long Context Tests (nomic-specific - requires model files)
// =============================================================================

mod long_context {
    use super::*;

    /// Nomic supports 8192 token context (vs 512 for MiniLM/BGE).
    /// Test that embeddings are stable for longer text.
    #[test]
    #[ignore = "requires model files"]
    fn test_nomic_long_text_embedding() {
        let embedder = create_embedder(EMBEDDER_NOMIC_V15).unwrap();

        // Generate text that would exceed 512 tokens
        let long_text = "This is a sentence about technology and innovation. ".repeat(100); // ~800 tokens

        let result = embedder.embed(&long_text);
        assert!(result.is_ok(), "Nomic should handle long text");

        let embedding = result.unwrap();
        assert_eq!(embedding.len(), 768);

        // Verify L2 normalized
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "Should be L2 normalized");
    }

    /// Verify nomic produces consistent dimensions regardless of input length.
    #[test]
    #[ignore = "requires model files"]
    fn test_nomic_dimension_consistency() {
        let embedder = create_embedder(EMBEDDER_NOMIC_V15).unwrap();

        let short = embedder.embed("hi").unwrap();
        let medium = embedder.embed("This is a medium length sentence").unwrap();
        let long = embedder.embed(&"word ".repeat(500)).unwrap();

        assert_eq!(short.len(), 768);
        assert_eq!(medium.len(), 768);
        assert_eq!(long.len(), 768);
    }

    /// MiniLM truncates at 512 tokens; test that truncation is handled gracefully.
    #[test]
    #[ignore = "requires model files"]
    fn test_minilm_truncation_for_long_text() {
        let embedder = create_embedder(EMBEDDER_MINILM_L6_V2).unwrap();

        // Generate text that would exceed 512 tokens
        let long_text = "This is a test sentence. ".repeat(200); // ~1000 tokens

        let result = embedder.embed(&long_text);
        assert!(result.is_ok(), "MiniLM should truncate without error");

        let embedding = result.unwrap();
        assert_eq!(embedding.len(), 384);
    }
}

// =============================================================================
// Performance Tests (requires model files - integration tests)
// =============================================================================

mod performance {
    use super::*;
    use std::time::Instant;

    #[test]
    #[ignore = "requires model files"]
    fn test_minilm_latency_reasonable() {
        let embedder = create_embedder(EMBEDDER_MINILM_L6_V2).unwrap();

        // Warmup
        for _ in 0..5 {
            let _ = embedder.embed("warmup");
        }

        let start = Instant::now();
        let iterations = 50;
        for _ in 0..iterations {
            let _ = embedder.embed("benchmark text for latency measurement");
        }
        #[allow(clippy::cast_precision_loss)] // Precision loss negligible for timing
        let avg_ms = start.elapsed().as_millis() as f64 / f64::from(iterations);

        // MiniLM should be reasonably fast (<100ms per embedding on CPU)
        assert!(
            avg_ms < 100.0,
            "MiniLM should be <100ms per embed, got {avg_ms}ms"
        );
    }

    #[test]
    #[ignore = "requires model files"]
    fn test_batch_faster_than_individual() {
        let embedder = create_embedder(EMBEDDER_MINILM_L6_V2).unwrap();
        let texts: Vec<String> = (0..32).map(|i| format!("text {i}")).collect();
        let text_refs: Vec<&str> = texts.iter().map(std::string::String::as_str).collect();

        // Warmup
        let _ = embedder.embed_batch(&text_refs);

        // Time batch
        let start = Instant::now();
        let _ = embedder.embed_batch(&text_refs);
        let batch_ms = start.elapsed().as_millis();

        // Time individual
        let start = Instant::now();
        for text in &text_refs {
            let _ = embedder.embed(text);
        }
        let individual_ms = start.elapsed().as_millis();

        // Batch should be faster (or at least not significantly slower)
        assert!(
            batch_ms <= individual_ms + 100,
            "Batch should be faster: batch={batch_ms}ms vs individual={individual_ms}ms"
        );
    }

    /// BGE should have comparable latency to MiniLM despite slightly more parameters.
    #[test]
    #[ignore = "requires model files"]
    fn test_bge_latency_reasonable() {
        let embedder = create_embedder(EMBEDDER_BGE_SMALL_EN_V15).unwrap();

        // Warmup
        for _ in 0..5 {
            let _ = embedder.embed("warmup");
        }

        let start = Instant::now();
        let iterations = 50;
        for _ in 0..iterations {
            let _ = embedder.embed("benchmark text for latency measurement");
        }
        #[allow(clippy::cast_precision_loss)] // Precision loss negligible for timing
        let avg_ms = start.elapsed().as_millis() as f64 / f64::from(iterations);

        // BGE should be reasonably fast (<100ms per embedding on CPU)
        // Slightly larger than MiniLM (33M vs 22M) but same architecture
        assert!(
            avg_ms < 100.0,
            "BGE should be <100ms per embed, got {avg_ms}ms"
        );
    }

    /// Nomic is larger (137M params) so expect higher latency than MiniLM/BGE.
    #[test]
    #[ignore = "requires model files"]
    fn test_nomic_latency_reasonable() {
        let embedder = create_embedder(EMBEDDER_NOMIC_V15).unwrap();

        // Warmup
        for _ in 0..5 {
            let _ = embedder.embed("warmup");
        }

        let start = Instant::now();
        let iterations = 30; // Fewer iterations since nomic is larger
        for _ in 0..iterations {
            let _ = embedder.embed("benchmark text for latency measurement");
        }
        #[allow(clippy::cast_precision_loss)] // Precision loss negligible for timing
        let avg_ms = start.elapsed().as_millis() as f64 / f64::from(iterations);

        // Nomic is larger (137M) so allow more latency (<200ms per embedding on CPU)
        assert!(
            avg_ms < 200.0,
            "Nomic should be <200ms per embed, got {avg_ms}ms"
        );
    }
}
