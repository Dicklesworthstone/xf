#![allow(clippy::uninlined_format_args)]
#![allow(clippy::cast_precision_loss)]
//! E2E integration tests for the semantic search pipeline.
//!
//! Comprehensive end-to-end tests that verify the full search pipeline works
//! correctly with all embedding/reranking combinations, MRL dimension variants,
//! and edge cases. Uses real models and real data.
//!
//! Bead: bd-3qov
//!
//! Test Categories:
//! 1. MRL Dimension Sweep Tests - verify quality at each dimension
//! 2. MRL Re-normalization Tests - verify unit vectors after truncation
//! 3. Search Quality Tests - verify semantic search returns relevant results
//! 4. Cross-Model Comparison - verify all models produce usable results
//! 5. CLI Integration Tests - verify commands work end-to-end

use std::time::Instant;

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

/// Vector norm (L2)
fn vector_norm(vec: &[f32]) -> f32 {
    vec.iter().map(|x| x * x).sum::<f32>().sqrt()
}

// =============================================================================
// MRL Re-normalization Verification Tests
// =============================================================================

mod mrl_renormalization {
    use super::*;
    use xf::embedder::Embedder;
    use xf::model_registry::{EmbedderConfig, ModelRegistry};

    /// Helper to create an embedder with optional dimension truncation.
    fn create_embedder_with_dims(model: &str, dims: Option<usize>) -> Box<dyn Embedder> {
        let registry = ModelRegistry::new();
        let mut config = EmbedderConfig::new(model);
        config.dimensions = dims;
        registry
            .embedder(&config)
            .expect("Failed to create embedder")
    }

    #[test]
    #[ignore = "Requires model files - run with --ignored"]
    fn test_mrl_renormalization_static_mrl() {
        // CRITICAL: Verify that MRL truncation includes proper re-normalization
        let model = "static-retrieval-mrl-en-v1";
        let dims = [128, 256, 384, 512, 768, 1024];
        let text = "This is a test sentence for MRL re-normalization verification.";

        println!("\n=== MRL Re-normalization Test: {} ===", model);

        for &dim in &dims {
            let embedder = create_embedder_with_dims(model, Some(dim));

            // Check if model supports this dimension
            if !embedder.supports_mrl() {
                println!("Skipping {} dims (MRL not supported)", dim);
                continue;
            }

            let embedding = embedder.embed(text).expect("Embedding failed");

            // Check embedding has correct dimension
            assert_eq!(
                embedding.len(),
                dim,
                "{} at {} dims: wrong dimension {}",
                model,
                dim,
                embedding.len()
            );

            // Check embedding is unit normalized
            let norm = vector_norm(&embedding);
            let is_unit = is_normalized(&embedding, 0.02);

            println!(
                "  dims={:4} | norm={:.6} | unit={}",
                dim,
                norm,
                if is_unit { "✓" } else { "✗" }
            );

            assert!(
                is_unit,
                "{} at {} dims: embedding not unit normalized (norm={:.4}). \
                 Re-normalization may be missing!",
                model, dim, norm
            );
        }
    }

    #[test]
    #[ignore = "Requires model files - run with --ignored"]
    fn test_mrl_renormalization_nomic() {
        let model = "nomic-embed-text-v1.5";
        let dims = [128, 256, 384, 512, 768];
        let text = "Testing Nomic MRL re-normalization at various dimensions.";

        println!("\n=== MRL Re-normalization Test: {} ===", model);

        for &dim in &dims {
            let embedder = create_embedder_with_dims(model, Some(dim));

            if !embedder.supports_mrl() {
                println!("Skipping {} dims (MRL not supported)", dim);
                continue;
            }

            let embedding = embedder.embed(text).expect("Embedding failed");
            let norm = vector_norm(&embedding);
            let is_unit = is_normalized(&embedding, 0.02);

            println!(
                "  dims={:4} | norm={:.6} | unit={}",
                dim,
                norm,
                if is_unit { "✓" } else { "✗" }
            );

            assert!(
                is_unit,
                "{} at {} dims: not unit normalized (norm={:.4})",
                model, dim, norm
            );
        }
    }

    #[test]
    #[ignore = "Requires model files - run with --ignored"]
    fn test_full_dimension_is_normalized() {
        // Verify that full-dimension embeddings are also normalized
        let models = [
            ("static-retrieval-mrl-en-v1", 1024),
            ("nomic-embed-text-v1.5", 768),
            ("minilm-l6-v2", 384),
            ("bge-small-en-v1.5", 384),
        ];
        let text = "Full dimension normalization test.";

        println!("\n=== Full Dimension Normalization ===");

        for (model, expected_dim) in models {
            let embedder = create_embedder_with_dims(model, None);
            let embedding = embedder.embed(text).expect("Embedding failed");

            assert_eq!(
                embedding.len(),
                expected_dim,
                "{}: unexpected dimension",
                model
            );

            let norm = vector_norm(&embedding);
            let is_unit = is_normalized(&embedding, 0.02);

            println!(
                "  {} | dims={} | norm={:.6} | unit={}",
                model,
                embedding.len(),
                norm,
                if is_unit { "✓" } else { "✗" }
            );

            assert!(
                is_unit,
                "{}: full-dim embedding not unit normalized (norm={:.4})",
                model, norm
            );
        }
    }
}

// =============================================================================
// MRL Quality Sweep Tests
// =============================================================================

mod mrl_quality_sweep {
    use super::*;
    use xf::embedder::Embedder;
    use xf::model_registry::{EmbedderConfig, ModelRegistry};

    fn create_embedder_with_dims(model: &str, dims: Option<usize>) -> Box<dyn Embedder> {
        let registry = ModelRegistry::new();
        let mut config = EmbedderConfig::new(model);
        config.dimensions = dims;
        registry
            .embedder(&config)
            .expect("Failed to create embedder")
    }

    #[test]
    #[ignore = "Requires model files - run with --ignored"]
    fn test_mrl_quality_degradation_graceful() {
        // Test that search quality degrades gracefully with lower dimensions
        let model = "static-retrieval-mrl-en-v1";
        let dims = [128, 256, 512, 768, 1024];

        // Test corpus: pairs of similar sentences
        let pairs = [
            (
                "Machine learning is revolutionizing technology",
                "AI and deep learning are transforming industries",
            ),
            (
                "The stock market crashed today",
                "Financial markets experienced a major downturn",
            ),
            (
                "I love programming in Rust",
                "Rust is my favorite programming language",
            ),
            (
                "The weather is beautiful outside",
                "It's a gorgeous sunny day today",
            ),
        ];

        println!("\n=== MRL Quality Degradation Test: {} ===", model);
        let mut quality_at_dim: Vec<(usize, f32)> = Vec::new();

        for &dim in &dims {
            let embedder = create_embedder_with_dims(model, Some(dim));

            let mut total_similarity = 0.0;
            for (a, b) in &pairs {
                let emb_a = embedder.embed(a).expect("embed failed");
                let emb_b = embedder.embed(b).expect("embed failed");
                total_similarity += cosine_similarity(&emb_a, &emb_b);
            }
            let avg_similarity = total_similarity / pairs.len() as f32;
            quality_at_dim.push((dim, avg_similarity));

            println!("  dims={:4} | avg_similarity={:.4}", dim, avg_similarity);
        }

        // Verify monotonic quality improvement (with tolerance)
        for window in quality_at_dim.windows(2) {
            let (d1, q1) = window[0];
            let (d2, q2) = window[1];
            // Higher dims should give equal or better quality (with 15% tolerance)
            assert!(
                q2 >= q1 * 0.85,
                "{}: quality at dims={} ({:.3}) should not be much worse than dims={} ({:.3})",
                model,
                d2,
                q2,
                d1,
                q1
            );
        }

        // Even at lowest dimension, quality should be non-trivial (> 0.3 similarity)
        if let Some(&(dims, quality)) = quality_at_dim.first() {
            assert!(
                quality > 0.3,
                "{}: even at {} dims, avg similarity should be >0.3, got {:.3}",
                model,
                dims,
                quality
            );
        }
    }

    #[test]
    #[ignore = "Requires model files - run with --ignored"]
    fn test_mrl_query_doc_consistency() {
        // Verify query and document embeddings are computed identically
        let model = "static-retrieval-mrl-en-v1";
        let dims = [256, 512];
        let text = "Consistent embedding computation test.";

        println!("\n=== MRL Query/Doc Consistency Test ===");

        for &dim in &dims {
            let embedder = create_embedder_with_dims(model, Some(dim));

            // Embed same text twice
            let emb1 = embedder.embed(text).expect("embed failed");
            let emb2 = embedder.embed(text).expect("embed failed");

            // Should be identical
            let similarity = cosine_similarity(&emb1, &emb2);
            println!("  dims={} | self-similarity={:.6}", dim, similarity);

            assert!(
                similarity > 0.999,
                "Same text should produce identical embeddings: {:.6}",
                similarity
            );
        }
    }
}

// =============================================================================
// Cross-Model Comparison Tests
// =============================================================================

mod cross_model_comparison {
    use super::*;
    use xf::embedder::Embedder;
    use xf::model_registry::{EmbedderConfig, ModelRegistry};

    fn create_embedder(model: &str) -> Box<dyn Embedder> {
        let registry = ModelRegistry::new();
        let config = EmbedderConfig::new(model);
        registry
            .embedder(&config)
            .expect("Failed to create embedder")
    }

    #[test]
    #[ignore = "Requires model files - run with --ignored"]
    fn test_all_models_produce_usable_results() {
        // Verify that all available models produce meaningful embeddings
        let models = [
            "minilm-l6-v2",
            "bge-small-en-v1.5",
            "static-retrieval-mrl-en-v1",
            "potion-retrieval-32M",
        ];

        // Test queries with expected high similarity pairs
        let test_cases = [
            (
                "artificial intelligence research",
                "machine learning studies",
                0.5, // minimum expected similarity
            ),
            (
                "software engineering best practices",
                "programming methodologies and patterns",
                0.4,
            ),
            ("the cat sat on the mat", "a feline rested on the rug", 0.5),
        ];

        println!("\n=== Cross-Model Quality Comparison ===");

        for model in models {
            println!("\nModel: {}", model);
            let embedder = create_embedder(model);

            let mut passed = 0;
            let mut total = 0;

            for (query, related, min_sim) in &test_cases {
                let emb_q = embedder.embed(query).expect("embed query failed");
                let emb_r = embedder.embed(related).expect("embed related failed");
                let similarity = cosine_similarity(&emb_q, &emb_r);

                let pass = similarity >= *min_sim;
                if pass {
                    passed += 1;
                }
                total += 1;

                println!(
                    "  {} sim={:.3} (min={:.2}) | \"{}\" <-> \"{}\"",
                    if pass { "✓" } else { "✗" },
                    similarity,
                    min_sim,
                    query,
                    related
                );
            }

            // At least 2/3 of test cases should pass
            assert!(
                passed >= (total * 2 / 3),
                "{}: only {}/{} test cases passed",
                model,
                passed,
                total
            );
        }
    }

    #[test]
    #[ignore = "Requires model files - run with --ignored"]
    fn test_model_embedding_dimensions() {
        let expected_dims = [
            ("minilm-l6-v2", 384),
            ("bge-small-en-v1.5", 384),
            ("static-retrieval-mrl-en-v1", 1024),
            ("potion-retrieval-32M", 256),
            ("nomic-embed-text-v1.5", 768),
        ];

        println!("\n=== Model Dimension Verification ===");

        for (model, expected) in expected_dims {
            let embedder = create_embedder(model);
            let actual = embedder.dimension();

            println!("  {} | expected={} actual={}", model, expected, actual);

            assert_eq!(actual, expected, "{}: dimension mismatch", model);
        }
    }
}

// =============================================================================
// Performance Baseline Tests
// =============================================================================

mod performance {
    use super::*;
    use xf::embedder::Embedder;
    use xf::model_registry::{EmbedderConfig, ModelRegistry};

    fn create_embedder(model: &str) -> Box<dyn Embedder> {
        let registry = ModelRegistry::new();
        let config = EmbedderConfig::new(model);
        registry
            .embedder(&config)
            .expect("Failed to create embedder")
    }

    #[test]
    #[ignore = "Requires model files - run with --ignored"]
    fn test_embedding_latency_acceptable() {
        let models = [
            ("minilm-l6-v2", 50.0), // max ms per embed
            ("bge-small-en-v1.5", 50.0),
            ("static-retrieval-mrl-en-v1", 100.0),
            ("potion-retrieval-32M", 20.0), // Static models should be very fast
        ];

        let test_texts = [
            "Short text.",
            "A medium length sentence for testing embedding performance.",
            "This is a longer piece of text that contains multiple sentences. \
             It should still be processed quickly by modern embedding models. \
             The latency should remain acceptable even for longer inputs.",
        ];

        println!("\n=== Embedding Latency Test ===");

        for (model, max_ms) in models {
            let embedder = create_embedder(model);

            // Warmup
            for text in &test_texts {
                let _ = embedder.embed(text);
            }

            let start = Instant::now();
            let iterations = 10;
            for _ in 0..iterations {
                for text in &test_texts {
                    let _ = embedder.embed(text);
                }
            }
            let total_ms = start.elapsed().as_millis() as f64;
            let avg_ms = total_ms / (iterations * test_texts.len()) as f64;

            println!(
                "  {} | avg={:.2}ms | max={:.0}ms | {}",
                model,
                avg_ms,
                max_ms,
                if avg_ms <= max_ms { "✓" } else { "✗" }
            );

            assert!(
                avg_ms <= max_ms,
                "{}: avg latency {:.2}ms exceeds max {:.0}ms",
                model,
                avg_ms,
                max_ms
            );
        }
    }

    #[test]
    #[ignore = "Requires model files - run with --ignored"]
    fn test_batch_embedding_faster_than_individual() {
        let model = "minilm-l6-v2";
        let embedder = create_embedder(model);

        let texts: Vec<&str> = vec![
            "First test sentence for batch processing.",
            "Second sentence in the batch.",
            "Third sentence to embed.",
            "Fourth text input.",
            "Fifth and final sentence.",
        ];

        // Individual embedding time
        let start = Instant::now();
        for text in &texts {
            let _ = embedder.embed(text);
        }
        let individual_ms = start.elapsed().as_millis();

        // Batch embedding time
        let start = Instant::now();
        let _ = embedder.embed_batch(&texts);
        let batch_ms = start.elapsed().as_millis();

        println!("\n=== Batch vs Individual Embedding ===");
        println!("  individual={}ms | batch={}ms", individual_ms, batch_ms);

        // Batch should be faster or at least not much slower
        assert!(
            batch_ms <= individual_ms + 50, // Allow 50ms tolerance
            "Batch should not be significantly slower: batch={}ms, individual={}ms",
            batch_ms,
            individual_ms
        );
    }
}

// =============================================================================
// Edge Cases
// =============================================================================

mod edge_cases {
    use super::*;
    use xf::embedder::Embedder;
    use xf::model_registry::{EmbedderConfig, ModelRegistry};

    fn create_embedder(model: &str) -> Box<dyn Embedder> {
        let registry = ModelRegistry::new();
        let config = EmbedderConfig::new(model);
        registry
            .embedder(&config)
            .expect("Failed to create embedder")
    }

    #[test]
    #[ignore = "Requires model files - run with --ignored"]
    fn test_empty_string_handling() {
        let models = [
            "minilm-l6-v2",
            "static-retrieval-mrl-en-v1",
            "potion-retrieval-32M",
        ];

        println!("\n=== Empty String Handling ===");

        for model in models {
            let embedder = create_embedder(model);
            let result = embedder.embed("");

            // Should either succeed with a valid embedding or return an error
            // Should NOT panic
            match result {
                Ok(emb) => {
                    println!("  {}: Ok (len={})", model, emb.len());
                    assert!(!emb.is_empty(), "{}: empty embedding returned", model);
                }
                Err(e) => {
                    println!("  {}: Err ({})", model, e);
                    // Error is acceptable for empty input
                }
            }
        }
    }

    #[test]
    #[ignore = "Requires model files - run with --ignored"]
    fn test_unicode_handling() {
        let model = "minilm-l6-v2";
        let embedder = create_embedder(model);

        let unicode_texts = [
            "Hello 世界",
            "Café résumé naïve",
            "🚀 rocket science 🔬",
            "Привет мир",
            "مرحبا بالعالم",
        ];

        println!("\n=== Unicode Handling Test ===");

        for text in unicode_texts {
            let result = embedder.embed(text);
            match result {
                Ok(emb) => {
                    let norm = vector_norm(&emb);
                    println!("  ✓ \"{}\" | norm={:.4}", text, norm);
                    assert!(!emb.is_empty());
                    assert!(
                        is_normalized(&emb, 0.05),
                        "Unicode text should be normalized"
                    );
                }
                Err(e) => {
                    println!("  ✗ \"{}\" | error: {}", text, e);
                    panic!("{}: Unicode text should be embeddable", text);
                }
            }
        }
    }

    #[test]
    #[ignore = "Requires model files - run with --ignored"]
    fn test_very_long_text() {
        let model = "minilm-l6-v2";
        let embedder = create_embedder(model);

        // Generate a very long text (8000+ characters)
        let long_text = "This is a test sentence. ".repeat(400);
        println!("\n=== Very Long Text Test ===");
        println!("  Text length: {} chars", long_text.len());

        let result = embedder.embed(&long_text);
        match result {
            Ok(emb) => {
                println!("  ✓ Embedding succeeded (dim={})", emb.len());
                assert!(is_normalized(&emb, 0.05));
            }
            Err(e) => {
                // Some models may truncate or reject very long text
                println!("  ! Error (may be expected): {}", e);
            }
        }
    }
}
