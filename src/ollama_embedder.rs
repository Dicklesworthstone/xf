//! Ollama-based embedder for true semantic text embeddings.
//!
//! This embedder uses a locally running Ollama instance to generate
//! ML-based semantic embeddings using models like nomic-embed-text.
//!
//! # Requirements
//!
//! - Ollama must be running locally (default: http://localhost:11434)
//! - An embedding model must be available (e.g., `ollama pull nomic-embed-text`)
//!
//! # Properties
//!
//! - **Semantic**: "happy" and "joyful" will have high similarity
//! - **Slower**: ~50-100ms per embedding (network + inference)
//! - **Requires Ollama**: Not available if Ollama isn't running

use crate::embedder::{l2_normalize, Embedder, EmbedderError, EmbedderResult};
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::net::TcpStream;
use std::time::Duration;

/// Default Ollama API endpoint.
const DEFAULT_OLLAMA_URL: &str = "localhost:11434";

/// Default embedding model.
const DEFAULT_MODEL: &str = "nomic-embed-text";

/// Dimension for nomic-embed-text model.
const NOMIC_DIMENSION: usize = 768;

/// Request timeout in seconds.
const TIMEOUT_SECS: u64 = 30;

/// Ollama embedding request.
#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    prompt: &'a str,
}

/// Ollama embedding response.
#[derive(Deserialize)]
struct EmbeddingResponse {
    embedding: Vec<f64>,
}

/// Ollama-based semantic embedder.
#[derive(Debug, Clone)]
pub struct OllamaEmbedder {
    host: String,
    model: String,
    dimension: usize,
    id: String,
}

impl OllamaEmbedder {
    /// Create a new Ollama embedder with default settings.
    ///
    /// Uses `nomic-embed-text` model on `localhost:11434`.
    #[must_use]
    pub fn new() -> Self {
        Self::with_model(DEFAULT_MODEL)
    }

    /// Create an Ollama embedder with a specific model.
    #[must_use]
    pub fn with_model(model: &str) -> Self {
        let dimension = match model {
            "nomic-embed-text" => NOMIC_DIMENSION,
            "all-minilm" | "all-MiniLM-L6-v2" => 384,
            "mxbai-embed-large" => 1024,
            _ => NOMIC_DIMENSION, // Default assumption
        };

        Self {
            host: DEFAULT_OLLAMA_URL.to_string(),
            model: model.to_string(),
            dimension,
            id: format!("ollama-{model}"),
        }
    }

    /// Create an Ollama embedder with custom host and model.
    #[must_use]
    pub fn with_host_and_model(host: &str, model: &str, dimension: usize) -> Self {
        Self {
            host: host.to_string(),
            model: model.to_string(),
            dimension,
            id: format!("ollama-{model}"),
        }
    }

    /// Check if Ollama is available.
    #[must_use]
    pub fn is_available(&self) -> bool {
        TcpStream::connect_timeout(
            &self.host.parse().unwrap_or_else(|_| "127.0.0.1:11434".parse().unwrap()),
            Duration::from_secs(2),
        )
        .is_ok()
    }

    /// Make HTTP request to Ollama API.
    fn request(&self, text: &str) -> EmbedderResult<Vec<f32>> {
        let addr = self.host.parse().map_err(|e| {
            EmbedderError::Unavailable(format!("invalid host address: {e}"))
        })?;

        let mut stream = TcpStream::connect_timeout(&addr, Duration::from_secs(TIMEOUT_SECS))
            .map_err(|e| EmbedderError::Unavailable(format!("connection failed: {e}")))?;

        stream
            .set_read_timeout(Some(Duration::from_secs(TIMEOUT_SECS)))
            .ok();
        stream
            .set_write_timeout(Some(Duration::from_secs(TIMEOUT_SECS)))
            .ok();

        let request_body = serde_json::to_string(&EmbeddingRequest {
            model: &self.model,
            prompt: text,
        })
        .map_err(|e| EmbedderError::Internal(format!("JSON serialization failed: {e}")))?;

        let http_request = format!(
            "POST /api/embeddings HTTP/1.1\r\n\
             Host: {}\r\n\
             Content-Type: application/json\r\n\
             Content-Length: {}\r\n\
             Connection: close\r\n\
             \r\n\
             {}",
            self.host,
            request_body.len(),
            request_body
        );

        stream
            .write_all(http_request.as_bytes())
            .map_err(|e| EmbedderError::EmbeddingFailed(format!("write failed: {e}")))?;

        let mut response = String::new();
        stream
            .read_to_string(&mut response)
            .map_err(|e| EmbedderError::EmbeddingFailed(format!("read failed: {e}")))?;

        // Parse HTTP response - find JSON body after headers
        let body_start = response.find("\r\n\r\n").or_else(|| response.find("\n\n"));
        let body = match body_start {
            Some(idx) => &response[idx..].trim(),
            None => {
                return Err(EmbedderError::EmbeddingFailed(
                    "invalid HTTP response".to_string(),
                ))
            }
        };

        // Handle chunked transfer encoding
        let json_body = if body.starts_with(|c: char| c.is_ascii_hexdigit()) {
            // Chunked: skip chunk size line
            body.lines()
                .skip(1)
                .take_while(|line| !line.is_empty() && *line != "0")
                .collect::<Vec<_>>()
                .join("")
        } else {
            body.to_string()
        };

        let parsed: EmbeddingResponse = serde_json::from_str(&json_body).map_err(|e| {
            EmbedderError::EmbeddingFailed(format!("JSON parse failed: {e}, body: {json_body}"))
        })?;

        // Convert f64 to f32 and normalize
        let mut embedding: Vec<f32> = parsed.embedding.iter().map(|&x| x as f32).collect();
        l2_normalize(&mut embedding);

        Ok(embedding)
    }
}

impl Default for OllamaEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

impl Embedder for OllamaEmbedder {
    fn embed(&self, text: &str) -> EmbedderResult<Vec<f32>> {
        if text.is_empty() {
            return Err(EmbedderError::InvalidInput("empty text".to_string()));
        }

        self.request(text)
    }

    fn embed_batch(&self, texts: &[&str]) -> EmbedderResult<Vec<Vec<f32>>> {
        // Ollama doesn't have native batch API, so we process sequentially
        // Could be parallelized with rayon for better throughput
        texts.iter().map(|t| self.embed(t)).collect()
    }

    fn dimension(&self) -> usize {
        self.dimension
    }

    fn id(&self) -> &str {
        &self.id
    }

    fn is_semantic(&self) -> bool {
        true // This IS a semantic embedder
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let embedder = OllamaEmbedder::new();
        assert_eq!(embedder.model, "nomic-embed-text");
        assert_eq!(embedder.dimension(), NOMIC_DIMENSION);
        assert!(embedder.is_semantic());
    }

    #[test]
    fn test_with_model() {
        let embedder = OllamaEmbedder::with_model("all-minilm");
        assert_eq!(embedder.model, "all-minilm");
        assert_eq!(embedder.dimension(), 384);
    }

    #[test]
    fn test_empty_input() {
        let embedder = OllamaEmbedder::new();
        let result = embedder.embed("");
        assert!(result.is_err());
    }

    // Integration test - only runs if Ollama is available
    #[test]
    #[ignore = "requires running Ollama instance"]
    fn test_embed_integration() {
        let embedder = OllamaEmbedder::new();
        if !embedder.is_available() {
            println!("Ollama not available, skipping integration test");
            return;
        }

        let embedding = embedder.embed("hello world").unwrap();
        assert_eq!(embedding.len(), NOMIC_DIMENSION);

        // Check L2 normalization
        let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    #[test]
    #[ignore = "requires running Ollama instance"]
    fn test_semantic_similarity() {
        use crate::embedder::dot_product;

        let embedder = OllamaEmbedder::new();
        if !embedder.is_available() {
            return;
        }

        let e_happy = embedder.embed("I am very happy today").unwrap();
        let e_joyful = embedder.embed("I am feeling joyful").unwrap();
        let e_sad = embedder.embed("I am feeling very sad").unwrap();

        let sim_happy_joyful = dot_product(&e_happy, &e_joyful);
        let sim_happy_sad = dot_product(&e_happy, &e_sad);

        // Happy and joyful should be more similar than happy and sad
        assert!(
            sim_happy_joyful > sim_happy_sad,
            "Expected happy-joyful ({sim_happy_joyful}) > happy-sad ({sim_happy_sad})"
        );
    }
}
