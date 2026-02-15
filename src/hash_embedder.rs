//! FNV-1a hash-based embedder for fast, deterministic text embeddings.
//!
//! This embedder uses feature hashing (the "hashing trick") to convert text
//! into fixed-dimension dense vectors. It's always available, requires no
//! model files, and runs in ~0ms.
//!
//! # Algorithm
//!
//! 1. Tokenize text on non-alphanumeric boundaries
//! 2. Hash each token using FNV-1a (64-bit)
//! 3. Use hash to determine:
//!    - Index: `hash % dimension`
//!    - Sign: MSB of hash (bit 63) determines +1 or -1
//! 4. L2-normalize the resulting vector
//!
//! # Properties
//!
//! - **Deterministic**: Same input always produces same output
//! - **Fast**: No ML inference, pure hash computation
//! - **Sparse-ish**: Most dimensions are 0 for short texts
//! - **Not semantic**: "happy" and "joyful" won't be similar
//!
//! Use this as a fallback or for hybrid search with keyword-aware matching.

use crate::embedder::{Embedder, EmbedderError, EmbedderResult, l2_normalize};
#[cfg(feature = "frankensearch-migration")]
use frankensearch_embed::{HashAlgorithm as FsHashAlgorithm, HashEmbedder as FsHashEmbedder};

/// FNV-1a 64-bit offset basis.
#[cfg(not(feature = "frankensearch-migration"))]
const FNV_OFFSET_BASIS: u64 = 0xcbf2_9ce4_8422_2325;

/// FNV-1a 64-bit prime.
#[cfg(not(feature = "frankensearch-migration"))]
const FNV_PRIME: u64 = 0x0100_0000_01b3;

/// Default embedding dimension (matches `MiniLM` for compatibility).
pub const DEFAULT_DIMENSION: usize = 384;

/// Minimum token length to include (filter single-char tokens).
const MIN_TOKEN_LEN: usize = 2;

/// FNV-1a hash-based embedder.
#[derive(Debug, Clone)]
pub struct HashEmbedder {
    dimension: usize,
    id: String,
    #[cfg(feature = "frankensearch-migration")]
    delegate: FsHashEmbedder,
}

impl HashEmbedder {
    /// Create a new hash embedder with the specified dimension.
    ///
    /// # Panics
    ///
    /// Panics if dimension is 0.
    #[must_use]
    pub fn new(dimension: usize) -> Self {
        assert!(dimension > 0, "dimension must be positive");
        Self {
            dimension,
            id: format!("fnv1a-{dimension}"),
            #[cfg(feature = "frankensearch-migration")]
            delegate: FsHashEmbedder::new(dimension, FsHashAlgorithm::FnvModular),
        }
    }

    /// Create a hash embedder with the default dimension (384).
    #[must_use]
    pub fn default_dimension() -> Self {
        Self::new(DEFAULT_DIMENSION)
    }

    /// Compute FNV-1a hash of a byte slice.
    #[inline]
    #[cfg(not(feature = "frankensearch-migration"))]
    fn fnv1a_hash(bytes: &[u8]) -> u64 {
        let mut hash = FNV_OFFSET_BASIS;
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(FNV_PRIME);
        }
        hash
    }

    /// Tokenize text into lowercase alphanumeric tokens.
    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split(|c: char| !c.is_alphanumeric())
            .filter(|s| s.len() >= MIN_TOKEN_LEN)
            .map(String::from)
            .collect()
    }

    fn uniform_fallback(&self) -> Vec<f32> {
        let mut embedding = vec![1.0f32; self.dimension];
        l2_normalize(&mut embedding);
        embedding
    }

    /// Embed a list of tokens into a vector.
    #[cfg(not(feature = "frankensearch-migration"))]
    fn embed_tokens_legacy(&self, tokens: &[String]) -> Vec<f32> {
        let mut embedding = vec![0.0f32; self.dimension];

        for token in tokens {
            let hash = Self::fnv1a_hash(token.as_bytes());

            // Hash determines:
            // - Index: hash % dimension
            // - Sign: MSB of hash (bit 63)
            let dim = u64::try_from(self.dimension).unwrap_or(u64::MAX);
            let idx = usize::try_from(hash % dim).unwrap_or(0);
            let sign = if (hash >> 63) == 0 { 1.0 } else { -1.0 };

            embedding[idx] += sign;
        }

        l2_normalize(&mut embedding);
        embedding
    }
}

impl Default for HashEmbedder {
    fn default() -> Self {
        Self::default_dimension()
    }
}

impl Embedder for HashEmbedder {
    fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>> {
        if text.is_empty() {
            return Err(EmbedderError::InvalidInput("empty text".to_string()));
        }

        let tokens = Self::tokenize(text);

        if tokens.is_empty() {
            // Preserve low-signal fallback semantics in both legacy and migration modes.
            return Ok(self.uniform_fallback());
        }

        #[cfg(feature = "frankensearch-migration")]
        {
            let canonical = tokens.join(" ");
            let embedding = self.delegate.embed_sync(&canonical);
            if embedding.len() != self.dimension {
                return Err(EmbedderError::EmbeddingFailed(format!(
                    "delegate dimension mismatch: expected {}, got {}",
                    self.dimension,
                    embedding.len()
                )));
            }
            Ok(embedding)
        }

        #[cfg(not(feature = "frankensearch-migration"))]
        {
            Ok(self.embed_tokens_legacy(&tokens))
        }
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn is_semantic(&self) -> bool {
        false // Hash embedder is lexical, not semantic
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::dot_product;

    #[test]
    fn test_new() {
        let embedder = HashEmbedder::new(256);
        assert_eq!(embedder.dimension(), 256);
        assert_eq!(embedder.id(), "fnv1a-256");
        assert!(!embedder.is_semantic());
    }

    #[test]
    fn test_default() {
        let embedder = HashEmbedder::default();
        assert_eq!(embedder.dimension(), DEFAULT_DIMENSION);
    }

    #[test]
    #[should_panic(expected = "dimension must be positive")]
    fn test_zero_dimension_panics() {
        let _ = HashEmbedder::new(0);
    }

    #[test]
    #[cfg(not(feature = "frankensearch-migration"))]
    fn test_fnv1a_hash() {
        // Known FNV-1a test vectors
        let hash = HashEmbedder::fnv1a_hash(b"");
        assert_eq!(hash, FNV_OFFSET_BASIS);

        let hash = HashEmbedder::fnv1a_hash(b"a");
        assert_ne!(hash, FNV_OFFSET_BASIS);
    }

    #[test]
    fn test_tokenize() {
        let tokens = HashEmbedder::tokenize("Hello, World! This is a test.");
        assert!(tokens.contains(&"hello".to_string()));
        assert!(tokens.contains(&"world".to_string()));
        assert!(tokens.contains(&"this".to_string()));
        assert!(tokens.contains(&"is".to_string()));
        assert!(tokens.contains(&"test".to_string()));
        // "a" should be filtered (< 2 chars)
        assert!(!tokens.contains(&"a".to_string()));
    }

    #[test]
    fn test_embed_basic() {
        let embedder = HashEmbedder::default();
        let embedding = embedder.embed("hello world").unwrap();

        assert_eq!(embedding.len(), DEFAULT_DIMENSION);

        // Check L2 normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_embed_deterministic() {
        let embedder = HashEmbedder::default();

        let e1 = embedder.embed("rust programming").unwrap();
        let e2 = embedder.embed("rust programming").unwrap();

        assert_eq!(e1, e2);
    }

    #[test]
    fn test_embed_empty_returns_error() {
        let embedder = HashEmbedder::default();
        let result = embedder.embed("");
        assert!(result.is_err());
    }

    #[test]
    fn test_embed_no_valid_tokens() {
        let embedder = HashEmbedder::default();
        // Only single-char tokens, all filtered out
        let embedding = embedder.embed("a b c !").unwrap();

        // Should return uniform normalized vector
        assert_eq!(embedding.len(), DEFAULT_DIMENSION);
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Relaxed tolerance due to floating point precision with high dimensions
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_similar_texts_have_higher_similarity() {
        let embedder = HashEmbedder::default();

        let e_rust = embedder.embed("rust programming language").unwrap();
        let e_rust2 = embedder.embed("rust programming").unwrap();
        let e_python = embedder.embed("python scripting language").unwrap();

        let sim_rust_rust2 = dot_product(&e_rust, &e_rust2);
        let sim_rust_python = dot_product(&e_rust, &e_python);

        // "rust programming language" should be more similar to "rust programming"
        // than to "python scripting language"
        assert!(sim_rust_rust2 > sim_rust_python);
    }

    #[test]
    fn test_embed_batch() {
        let embedder = HashEmbedder::default();
        let texts = ["hello world", "rust programming", "machine learning"];

        let embeddings = embedder.embed_batch(&texts).unwrap();

        assert_eq!(embeddings.len(), 3);
        for embedding in &embeddings {
            assert_eq!(embedding.len(), DEFAULT_DIMENSION);
        }

        // Each should match individual embed
        for (i, text) in texts.iter().enumerate() {
            let single = embedder.embed(text).unwrap();
            assert_eq!(embeddings[i], single);
        }
    }

    #[test]
    fn test_unicode_support() {
        let embedder = HashEmbedder::default();

        // Should handle Unicode without panicking
        let embedding = embedder.embed("日本語テスト café naïve").unwrap();
        assert_eq!(embedding.len(), DEFAULT_DIMENSION);

        // Should be deterministic for Unicode
        let e2 = embedder.embed("日本語テスト café naïve").unwrap();
        assert_eq!(embedding, e2);
    }

    #[test]
    fn test_case_insensitive() {
        let embedder = HashEmbedder::default();

        let e1 = embedder.embed("Hello World").unwrap();
        let e2 = embedder.embed("hello world").unwrap();
        let e3 = embedder.embed("HELLO WORLD").unwrap();

        // All should produce identical embeddings
        assert_eq!(e1, e2);
        assert_eq!(e2, e3);
    }

    #[test]
    fn test_info() {
        let embedder = HashEmbedder::new(256);
        let info = embedder.info();

        assert_eq!(info.id, "fnv1a-256");
        assert_eq!(info.dimension, 256);
        assert!(!info.is_semantic);
    }
}
