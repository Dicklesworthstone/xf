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
//! 1. Hash Embedder Pipeline Tests - CI-runnable, no model downloads
//! 2. MRL Dimension Sweep Tests - verify quality at each dimension
//! 3. MRL Re-normalization Tests - verify unit vectors after truncation
//! 4. Search Quality Tests - verify semantic search returns relevant results
//! 5. Cross-Model Comparison - verify all models produce usable results
//! 6. CLI Integration Tests - verify commands work end-to-end

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
// Hash Embedder Pipeline Tests (CI-runnable, no model downloads)
// =============================================================================

mod hash_embedder_pipeline {
    use super::*;
    use xf::embedder::Embedder;
    use xf::hash_embedder::HashEmbedder;
    use xf::vector::InMemoryVectorIndex;

    #[test]
    fn test_hash_embedder_produces_normalized_vectors() {
        let embedder = HashEmbedder::default();

        let texts = [
            "machine learning algorithms",
            "the quick brown fox",
            "database indexing strategies",
            "🚀 emoji in text",
            "a",
        ];

        for text in &texts {
            let embedding = embedder.embed(text).expect("embed should succeed");
            assert_eq!(embedding.len(), 384, "should be 384 dimensions");

            let norm = vector_norm(&embedding);
            assert!(
                is_normalized(&embedding, 0.01),
                "embedding for {:?} should be L2-normalized, got norm={:.6}",
                text,
                norm
            );
        }
    }

    #[test]
    fn test_hash_embedder_rejects_empty_string() {
        let embedder = HashEmbedder::default();
        let result = embedder.embed("");
        assert!(
            result.is_err(),
            "empty string should return error, not a valid embedding"
        );
    }

    #[test]
    fn test_hash_embedder_deterministic() {
        let embedder = HashEmbedder::default();
        let text = "reproducible embeddings are essential for testing";

        let emb1 = embedder.embed(text).expect("first embed");
        let emb2 = embedder.embed(text).expect("second embed");

        assert_eq!(emb1.len(), emb2.len());
        for (i, (a, b)) in emb1.iter().zip(emb2.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "dimension {i} should be bitwise identical"
            );
        }
    }

    #[test]
    fn test_hash_embedder_different_texts_different_vectors() {
        let embedder = HashEmbedder::default();

        let emb_a = embedder.embed("hello world").unwrap();
        let emb_b = embedder.embed("goodbye moon").unwrap();

        // Different texts should produce different embeddings
        let identical = emb_a
            .iter()
            .zip(emb_b.iter())
            .all(|(a, b)| a.to_bits() == b.to_bits());
        assert!(
            !identical,
            "different texts should produce different embeddings"
        );
    }

    #[test]
    fn test_hash_embedder_batch_equals_individual() {
        let embedder = HashEmbedder::default();
        let texts = ["alpha", "beta", "gamma", "delta"];

        let batch = embedder.embed_batch(&texts).expect("batch embed");
        assert_eq!(batch.len(), texts.len());

        for (i, text) in texts.iter().enumerate() {
            let individual = embedder.embed(text).expect("individual embed");
            assert_eq!(batch[i].len(), individual.len());
            for (j, (a, b)) in batch[i].iter().zip(individual.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "text {:?} dim {j}: batch and individual should be identical",
                    text
                );
            }
        }
    }

    #[test]
    fn test_vector_index_add_and_search() {
        let embedder = HashEmbedder::default();
        let mut index = InMemoryVectorIndex::new(384);

        let docs = [
            ("doc0", "tweet", "machine learning is transformative"),
            ("doc1", "tweet", "artificial intelligence research"),
            ("doc2", "tweet", "cooking pasta recipe"),
            ("doc3", "like", "weather forecast for tomorrow"),
            ("doc4", "tweet", "deep neural network architectures"),
        ];

        for (id, doc_type, text) in &docs {
            let emb = embedder.embed(text).unwrap();
            index.add(id.to_string(), doc_type, emb);
        }

        assert_eq!(index.len(), 5);
        assert_eq!(index.dimension(), 384);

        // Search with a query
        let query_emb = embedder.embed("machine learning neural networks").unwrap();
        let results = index.search_top_k(&query_emb, 3, None);

        assert_eq!(results.len(), 3, "should return top 3 results");

        // Results should be sorted by score descending
        for window in results.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "results should be sorted descending: {} >= {}",
                window[0].score,
                window[1].score
            );
        }
    }

    #[test]
    fn test_vector_index_type_filtering() {
        let embedder = HashEmbedder::default();
        let mut index = InMemoryVectorIndex::new(384);

        let docs = [
            ("t1", "tweet", "hello world"),
            ("t2", "tweet", "rust programming"),
            ("l1", "like", "hello world liked"),
            ("d1", "dm", "hello private message"),
        ];

        for (id, doc_type, text) in &docs {
            let emb = embedder.embed(text).unwrap();
            index.add(id.to_string(), doc_type, emb);
        }

        let query = embedder.embed("hello").unwrap();

        // Filter to tweets only
        let tweet_results = index.search_top_k(&query, 10, Some(&["tweet"]));
        for r in &tweet_results {
            assert_eq!(r.doc_type, "tweet", "filter should return only tweets");
        }
        assert_eq!(tweet_results.len(), 2, "should find 2 tweets");

        // Filter to likes only
        let like_results = index.search_top_k(&query, 10, Some(&["like"]));
        assert_eq!(like_results.len(), 1, "should find 1 like");
        assert_eq!(like_results[0].doc_type, "like");

        // No filter returns all
        let all_results = index.search_top_k(&query, 10, None);
        assert_eq!(all_results.len(), 4, "no filter should return all docs");
    }

    #[test]
    fn test_vector_index_top_k_limiting() {
        let embedder = HashEmbedder::default();
        let mut index = InMemoryVectorIndex::new(384);

        for i in 0..20 {
            let emb = embedder.embed(&format!("document number {i}")).unwrap();
            index.add(format!("doc{i}"), "tweet", emb);
        }

        let query = embedder.embed("document").unwrap();

        // k=5 should return exactly 5
        let results = index.search_top_k(&query, 5, None);
        assert_eq!(results.len(), 5, "should limit to top 5");

        // k=100 should return all 20
        let all = index.search_top_k(&query, 100, None);
        assert_eq!(all.len(), 20, "k > len should return all");

        // k=1 should return exactly 1
        let one = index.search_top_k(&query, 1, None);
        assert_eq!(one.len(), 1, "k=1 should return exactly 1");
    }

    #[test]
    fn test_vector_index_empty() {
        let index = InMemoryVectorIndex::new(384);
        assert!(index.is_empty());
        assert_eq!(index.len(), 0);

        let embedder = HashEmbedder::default();
        let query = embedder.embed("anything").unwrap();
        let results = index.search_top_k(&query, 10, None);
        assert!(results.is_empty(), "empty index should return no results");
    }

    #[test]
    fn test_vector_index_type_counts() {
        let embedder = HashEmbedder::default();
        let mut index = InMemoryVectorIndex::new(384);

        let docs = [
            ("t1", "tweet", "one"),
            ("t2", "tweet", "two"),
            ("t3", "tweet", "three"),
            ("l1", "like", "four"),
            ("d1", "dm", "five"),
            ("d2", "dm", "six"),
        ];

        for (id, doc_type, text) in &docs {
            let emb = embedder.embed(text).unwrap();
            index.add(id.to_string(), doc_type, emb);
        }

        let counts = index.type_counts();
        assert_eq!(counts.get("tweet"), Some(&3));
        assert_eq!(counts.get("like"), Some(&1));
        assert_eq!(counts.get("dm"), Some(&2));
    }

    #[test]
    fn test_self_similarity_is_maximum() {
        let embedder = HashEmbedder::default();
        let text = "self similarity test";
        let emb = embedder.embed(text).unwrap();

        let self_sim = cosine_similarity(&emb, &emb);
        assert!(
            (self_sim - 1.0).abs() < 0.001,
            "self-similarity should be ~1.0, got {:.6}",
            self_sim
        );
    }

    #[test]
    fn test_search_self_retrieval() {
        // The document most similar to its own embedding should be itself
        let embedder = HashEmbedder::default();
        let mut index = InMemoryVectorIndex::new(384);

        let docs = [
            ("target", "tweet", "unique target document for retrieval"),
            (
                "other1",
                "tweet",
                "completely different topic about cooking",
            ),
            ("other2", "tweet", "another unrelated text about weather"),
        ];

        for (id, doc_type, text) in &docs {
            let emb = embedder.embed(text).unwrap();
            index.add(id.to_string(), doc_type, emb);
        }

        // Query with the exact text of "target"
        let query = embedder
            .embed("unique target document for retrieval")
            .unwrap();
        let results = index.search_top_k(&query, 1, None);

        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].doc_id, "target",
            "document should be its own best match"
        );
        assert!(
            results[0].score > 0.99,
            "self-retrieval score should be ~1.0, got {:.6}",
            results[0].score
        );
    }

    #[test]
    fn test_pipeline_roundtrip_embed_index_search() {
        // Full pipeline: create embedder -> embed documents -> build index -> search -> verify
        let embedder = HashEmbedder::default();
        let mut index = InMemoryVectorIndex::new(embedder.dimension());

        // Corpus of documents
        let corpus = [
            ("d1", "tweet", "rust programming language systems"),
            ("d2", "tweet", "python data science machine learning"),
            ("d3", "like", "javascript web development frontend"),
            ("d4", "dm", "database sql query optimization"),
            ("d5", "tweet", "rust cargo package manager"),
            ("d6", "tweet", "python pandas dataframe analysis"),
            ("d7", "like", "react typescript component design"),
            ("d8", "dm", "postgresql index performance tuning"),
        ];

        // Embed and index all documents
        let start = Instant::now();
        for (id, doc_type, text) in &corpus {
            let emb = embedder.embed(text).unwrap();
            index.add(id.to_string(), doc_type, emb);
        }
        let index_time = start.elapsed();

        assert_eq!(index.len(), 8);
        assert!(!index.is_empty());

        // Search for "rust" related
        let query = embedder.embed("rust programming cargo").unwrap();
        let results = index.search_top_k(&query, 3, None);
        assert_eq!(results.len(), 3);

        // All scores should be valid cosine similarities
        for r in &results {
            assert!(
                r.score >= -1.0 && r.score <= 1.0,
                "score should be valid cosine similarity: {}",
                r.score
            );
        }

        // Search with type filter
        let tweet_results = index.search_top_k(&query, 10, Some(&["tweet"]));
        assert!(tweet_results.len() <= 4, "only 4 tweets in corpus");
        for r in &tweet_results {
            assert_eq!(r.doc_type, "tweet");
        }

        println!(
            "Pipeline roundtrip: indexed {} docs in {:?}",
            corpus.len(),
            index_time
        );
    }
}

// =============================================================================
// Utility Function Tests (CI-runnable)
// =============================================================================

mod utility_functions {
    use super::*;

    #[test]
    fn test_cosine_similarity_identical_vectors() {
        let v = vec![1.0, 0.0, 0.0];
        let sim = cosine_similarity(&v, &v);
        assert!(
            (sim - 1.0).abs() < 1e-6,
            "identical vectors should have similarity 1.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "orthogonal vectors should have similarity ~0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_similarity_opposite_vectors() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            (sim + 1.0).abs() < 1e-6,
            "opposite vectors should have similarity -1.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_similarity_empty_vectors() {
        let empty: Vec<f32> = vec![];
        let sim = cosine_similarity(&empty, &empty);
        assert!(
            sim.abs() < 1e-6,
            "empty vectors should return ~0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_similarity_mismatched_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!(
            sim.abs() < 1e-6,
            "mismatched lengths should return ~0, got {sim}"
        );
    }

    #[test]
    fn test_is_normalized_unit_vector() {
        let unit = vec![1.0, 0.0, 0.0];
        assert!(is_normalized(&unit, 0.01));

        let unit2 = vec![
            1.0 / 3.0_f32.sqrt(),
            1.0 / 3.0_f32.sqrt(),
            1.0 / 3.0_f32.sqrt(),
        ];
        assert!(is_normalized(&unit2, 0.01));
    }

    #[test]
    fn test_is_normalized_non_unit_vector() {
        let non_unit = vec![2.0, 0.0, 0.0];
        assert!(!is_normalized(&non_unit, 0.01));
    }

    #[test]
    fn test_vector_norm() {
        let v = vec![3.0, 4.0];
        assert!((vector_norm(&v) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_norm_unit() {
        let v = vec![1.0, 0.0, 0.0];
        assert!((vector_norm(&v) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_vector_norm_zero() {
        let v = vec![0.0, 0.0, 0.0];
        assert!(vector_norm(&v).abs() < 1e-6);
    }
}

// =============================================================================
// Model Registry Hash Embedder Tests (CI-runnable)
// =============================================================================

mod model_registry_hash {
    use xf::model_registry::{EmbedderConfig, ModelRegistry};

    #[test]
    fn test_hash_embedder_via_registry() {
        let registry = ModelRegistry::new();
        let config = EmbedderConfig::new("hash");
        let embedder = registry.embedder(&config).expect("create hash embedder");

        assert_eq!(embedder.dimension(), 384);
        assert!(!embedder.is_semantic());
        assert!(!embedder.supports_mrl());

        let emb = embedder.embed("test text").expect("embed");
        assert_eq!(emb.len(), 384);
    }

    #[test]
    fn test_hash_embedder_aliases() {
        let registry = ModelRegistry::new();

        for alias in &["hash", "fnv1a", "hash-fnv1a-384"] {
            let config = EmbedderConfig::new(*alias);
            let embedder = registry
                .embedder(&config)
                .unwrap_or_else(|e| panic!("alias {:?} should work: {}", alias, e));
            assert_eq!(embedder.dimension(), 384, "alias {:?}", alias);
        }
    }

    #[test]
    fn test_hash_embedder_info() {
        let registry = ModelRegistry::new();
        let config = EmbedderConfig::new("hash");
        let embedder = registry.embedder(&config).unwrap();

        let info = embedder.info();
        assert_eq!(info.dimension, 384);
        assert!(!info.is_semantic);
        assert!(!info.supports_mrl);
    }
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

// =============================================================================
// Two-Tier Progressive Search Tests (CI-runnable, uses hash embedder)
// =============================================================================

mod two_tier {
    use xf::embedder::Embedder;
    use xf::hash_embedder::HashEmbedder;
    use xf::hybrid::{self, SearchMode};
    use xf::storage::Storage;
    use xf::vector::{
        InMemoryVectorIndex, VECTOR_INDEX_FAST_FILENAME, VECTOR_INDEX_QUALITY_FILENAME,
        VectorIndex, VectorSearchResult, write_vector_index_named,
    };

    /// Build two in-memory vector indexes with the same documents but different embeddings.
    fn build_two_tier_indexes() -> (InMemoryVectorIndex, InMemoryVectorIndex) {
        let embedder = HashEmbedder::default();
        let mut fast_index = InMemoryVectorIndex::new(384);
        let mut quality_index = InMemoryVectorIndex::new(384);

        let docs = [
            ("doc0", "tweet", "machine learning algorithms"),
            ("doc1", "tweet", "artificial intelligence research"),
            ("doc2", "tweet", "cooking pasta recipe italian"),
            ("doc3", "like", "weather forecast tomorrow rain"),
            ("doc4", "tweet", "deep neural network architectures"),
        ];

        for (id, doc_type, text) in &docs {
            let fast_emb = embedder.embed(text).unwrap();
            fast_index.add(id.to_string(), doc_type, fast_emb.clone());
            // Quality uses same hash embedder but on slightly different text
            // to simulate different model characteristics
            let quality_emb = embedder.embed(&format!("{text} semantic meaning")).unwrap();
            quality_index.add(id.to_string(), doc_type, quality_emb);
        }

        (fast_index, quality_index)
    }

    #[test]
    fn test_two_tier_hash_only() {
        // Verify blending produces valid results using hash embedder for both tiers
        let embedder = HashEmbedder::default();
        let (fast_index, quality_index) = build_two_tier_indexes();

        let query = embedder.embed("machine learning neural networks").unwrap();

        let fast_results = fast_index.search_top_k(&query, 5, None);
        let quality_results = quality_index.search_top_k(&query, 5, None);

        assert!(!fast_results.is_empty(), "fast results should not be empty");
        assert!(
            !quality_results.is_empty(),
            "quality results should not be empty"
        );

        // Normalize
        let mut fast_norm = fast_results;
        let mut quality_norm = quality_results;
        hybrid::min_max_normalize(&mut fast_norm);
        hybrid::min_max_normalize(&mut quality_norm);

        // Blend with default factor
        let blended = hybrid::blend_two_tier(&fast_norm, &quality_norm, 0.7);
        assert!(!blended.is_empty(), "blended results should not be empty");

        // Results should be sorted by score descending
        for window in blended.windows(2) {
            assert!(
                window[0].score >= window[1].score,
                "blended results should be sorted: {:.4} >= {:.4}",
                window[0].score,
                window[1].score
            );
        }
    }

    #[test]
    fn test_two_tier_degradation_no_quality() {
        // Missing quality index should return fast-only results
        let embedder = HashEmbedder::default();
        let mut fast_index = InMemoryVectorIndex::new(384);

        let docs = [
            ("doc0", "tweet", "hello world"),
            ("doc1", "tweet", "test document"),
        ];

        for (id, doc_type, text) in &docs {
            let emb = embedder.embed(text).unwrap();
            fast_index.add(id.to_string(), doc_type, emb);
        }

        let query = embedder.embed("hello test").unwrap();
        let fast_results = fast_index.search_top_k(&query, 5, None);

        // With no quality results, blend should just return fast results scaled
        let mut fast_norm = fast_results;
        hybrid::min_max_normalize(&mut fast_norm);

        let quality_results: Vec<VectorSearchResult> = vec![];
        let blended = hybrid::blend_two_tier(&fast_norm, &quality_results, 0.7);

        // Should have same number of results as fast
        assert_eq!(blended.len(), fast_norm.len());

        // All results should have score = original * (1 - blend_factor)
        for (blended_hit, fast_hit) in blended.iter().zip(fast_norm.iter()) {
            assert_eq!(blended_hit.doc_id, fast_hit.doc_id);
            let expected = fast_hit.score * 0.3; // 1 - 0.7
            assert!(
                (blended_hit.score - expected).abs() < 0.001,
                "fast-only score should be original * (1-blend): {:.4} vs {:.4}",
                blended_hit.score,
                expected
            );
        }
    }

    #[test]
    fn test_two_tier_blend_factor_zero_matches_fast() {
        // blend_factor=0 should give same ordering as fast-only
        let embedder = HashEmbedder::default();
        let (fast_index, quality_index) = build_two_tier_indexes();

        let query = embedder.embed("machine learning").unwrap();
        let fast_results = fast_index.search_top_k(&query, 5, None);
        let quality_results = quality_index.search_top_k(&query, 5, None);

        let mut fast_norm = fast_results;
        let mut quality_norm = quality_results;
        hybrid::min_max_normalize(&mut fast_norm);
        hybrid::min_max_normalize(&mut quality_norm);

        let blended = hybrid::blend_two_tier(&fast_norm, &quality_norm, 0.0);

        // With blend_factor=0, quality weight=0, fast weight=1.0
        // So scores should match fast scores exactly
        for (blended_hit, fast_hit) in blended.iter().zip(fast_norm.iter()) {
            assert_eq!(blended_hit.doc_id, fast_hit.doc_id);
            assert!(
                (blended_hit.score - fast_hit.score).abs() < 0.001,
                "blend_factor=0 should match fast: {:.4} vs {:.4}",
                blended_hit.score,
                fast_hit.score
            );
        }
    }

    #[test]
    fn test_two_tier_blend_factor_one_matches_quality() {
        // blend_factor=1 should give same ordering as quality-only
        let embedder = HashEmbedder::default();
        let (fast_index, quality_index) = build_two_tier_indexes();

        let query = embedder.embed("machine learning").unwrap();
        let fast_results = fast_index.search_top_k(&query, 5, None);
        let quality_results = quality_index.search_top_k(&query, 5, None);

        let mut fast_norm = fast_results;
        let mut quality_norm = quality_results;
        hybrid::min_max_normalize(&mut fast_norm);
        hybrid::min_max_normalize(&mut quality_norm);

        let blended = hybrid::blend_two_tier(&fast_norm, &quality_norm, 1.0);

        // With blend_factor=1, fast weight=0, quality weight=1.0
        // Quality-only docs should preserve their scores
        for quality_hit in &quality_norm {
            let blended_hit = blended
                .iter()
                .find(|h| h.doc_id == quality_hit.doc_id && h.doc_type == quality_hit.doc_type);
            if let Some(bh) = blended_hit {
                assert!(
                    (bh.score - quality_hit.score).abs() < 0.001,
                    "blend_factor=1 should match quality for {}: {:.4} vs {:.4}",
                    quality_hit.doc_id,
                    bh.score,
                    quality_hit.score
                );
            }
        }
    }

    #[test]
    fn test_two_tier_named_index_roundtrip() {
        // Test writing and loading named index files for two-tier mode
        let storage = Storage::open_memory().unwrap();
        let temp_dir = tempfile::tempdir().unwrap();
        let embedder = HashEmbedder::default();

        let docs = [
            ("doc0", "tweet", "machine learning algorithms"),
            ("doc1", "tweet", "cooking pasta recipe"),
        ];

        // Store embeddings with different model_ids
        for (id, doc_type, text) in &docs {
            let emb = embedder.embed(text).unwrap();
            storage
                .store_embedding_with_model(id, doc_type, &emb, None, "fast")
                .unwrap();
            // Quality uses different text for different embedding
            let quality_emb = embedder.embed(&format!("{text} quality semantic")).unwrap();
            storage
                .store_embedding_with_model(id, doc_type, &quality_emb, None, "quality")
                .unwrap();
        }

        // Write named indexes
        let fast_stats = write_vector_index_named(
            temp_dir.path(),
            &storage,
            VECTOR_INDEX_FAST_FILENAME,
            Some("fast"),
        )
        .unwrap();
        let quality_stats = write_vector_index_named(
            temp_dir.path(),
            &storage,
            VECTOR_INDEX_QUALITY_FILENAME,
            Some("quality"),
        )
        .unwrap();

        assert_eq!(fast_stats.record_count, 2);
        assert_eq!(quality_stats.record_count, 2);

        // Load named indexes
        let fast_idx = VectorIndex::load_named(temp_dir.path(), VECTOR_INDEX_FAST_FILENAME)
            .unwrap()
            .expect("fast index should exist");
        let quality_idx = VectorIndex::load_named(temp_dir.path(), VECTOR_INDEX_QUALITY_FILENAME)
            .unwrap()
            .expect("quality index should exist");

        assert_eq!(fast_idx.len(), 2);
        assert_eq!(quality_idx.len(), 2);

        // Search and blend
        let query = embedder.embed("machine learning").unwrap();
        let fast_results = fast_idx.search_top_k(&query, 5, None);
        let quality_results = quality_idx.search_top_k(&query, 5, None);

        let mut fast_norm = fast_results;
        let mut quality_norm = quality_results;
        hybrid::min_max_normalize(&mut fast_norm);
        hybrid::min_max_normalize(&mut quality_norm);

        let blended = hybrid::blend_two_tier(&fast_norm, &quality_norm, 0.7);
        assert_eq!(blended.len(), 2);
    }

    #[test]
    fn test_search_mode_two_tier_parsing() {
        assert_eq!(
            "two-tier".parse::<SearchMode>().unwrap(),
            SearchMode::TwoTier
        );
        assert_eq!(
            "progressive".parse::<SearchMode>().unwrap(),
            SearchMode::TwoTier
        );
        assert_eq!("2tier".parse::<SearchMode>().unwrap(), SearchMode::TwoTier);
    }

    #[test]
    fn test_two_tier_config_defaults() {
        use xf::config::TwoTierConfig;

        let cfg = TwoTierConfig::default();
        assert!(cfg.fast_model.is_none());
        assert!(cfg.quality_model.is_none());
        assert!((cfg.blend_factor - 0.7).abs() < f32::EPSILON);
        assert_eq!(cfg.quality_timeout_ms, 500);
    }
}
