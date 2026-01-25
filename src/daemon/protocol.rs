//! Daemon protocol types for MessagePack-over-UDS communication.
//!
//! The protocol uses length-prefixed MessagePack envelopes for reliable framing.

use serde::{Deserialize, Serialize};

/// Protocol version. Increment on breaking changes.
pub const PROTOCOL_VERSION: u8 = 1;

/// Envelope wrapping all messages for versioning and request correlation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope {
    /// Protocol version (always 1 for now).
    pub version: u8,
    /// Client request ID for response correlation.
    pub id: u64,
    /// MessagePack-encoded payload bytes.
    pub payload: Vec<u8>,
}

impl Envelope {
    /// Create a new envelope with the given request ID and payload.
    #[must_use]
    pub fn new(id: u64, payload: Vec<u8>) -> Self {
        Self {
            version: PROTOCOL_VERSION,
            id,
            payload,
        }
    }

    /// Encode request into envelope payload.
    pub fn from_request(id: u64, request: &Request) -> Result<Self, rmp_serde::encode::Error> {
        let payload = rmp_serde::to_vec(request)?;
        Ok(Self::new(id, payload))
    }

    /// Encode response into envelope payload.
    pub fn from_response(id: u64, response: &Response) -> Result<Self, rmp_serde::encode::Error> {
        let payload = rmp_serde::to_vec(response)?;
        Ok(Self::new(id, payload))
    }

    /// Decode request from envelope payload.
    pub fn to_request(&self) -> Result<Request, rmp_serde::decode::Error> {
        rmp_serde::from_slice(&self.payload)
    }

    /// Decode response from envelope payload.
    pub fn to_response(&self) -> Result<Response, rmp_serde::decode::Error> {
        rmp_serde::from_slice(&self.payload)
    }
}

/// Client requests to the daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Request {
    /// Health check - returns uptime and loaded model count.
    Health,

    /// Embed texts using specified model.
    Embed {
        /// Texts to embed.
        texts: Vec<String>,
        /// Model name (from registry).
        model: String,
        /// Optional MRL truncation dimension.
        dims: Option<usize>,
    },

    /// Rerank documents for a query.
    Rerank {
        /// Query text.
        query: String,
        /// Documents to rerank.
        documents: Vec<String>,
        /// Reranker model name.
        model: String,
    },

    /// Get daemon status - loaded models, memory, uptime.
    Status,

    /// Request graceful shutdown.
    Shutdown,
}

/// Daemon responses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Response {
    /// Health check response.
    Health {
        /// Daemon uptime in seconds.
        uptime_secs: u64,
        /// Number of models currently loaded in memory.
        models_loaded: usize,
    },

    /// Embedding results.
    Embeddings {
        /// Embedding vectors, one per input text.
        vectors: Vec<Vec<f32>>,
    },

    /// Reranking scores.
    Scores {
        /// Relevance scores, one per input document.
        scores: Vec<f32>,
    },

    /// Daemon status.
    Status {
        /// Uptime in seconds.
        uptime_secs: u64,
        /// Loaded model information.
        models: Vec<LoadedModelInfo>,
        /// Resident set size in MB.
        rss_mb: f64,
        /// Total requests served since startup.
        requests_served: u64,
        /// Current in-flight requests.
        in_flight: usize,
        /// Current queue length.
        queue_len: usize,
    },

    /// Error response.
    Error {
        /// Error code.
        code: u16,
        /// Error message.
        message: String,
    },

    /// Shutdown acknowledgment.
    Shutdown {
        /// Whether shutdown was accepted.
        ok: bool,
    },
}

/// Information about a loaded model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoadedModelInfo {
    /// Model name.
    pub name: String,
    /// Model type ("embedder" or "reranker").
    pub model_type: String,
    /// When the model was loaded (unix timestamp).
    pub loaded_at: u64,
    /// Number of requests served by this model.
    pub requests_served: u64,
    /// Last request time (unix timestamp, 0 if never used).
    pub last_used: u64,
}

/// Error codes for daemon responses.
pub mod error_codes {
    /// Unknown or internal error.
    pub const INTERNAL: u16 = 1;
    /// Invalid request format.
    pub const INVALID_REQUEST: u16 = 2;
    /// Unknown model name.
    pub const UNKNOWN_MODEL: u16 = 3;
    /// Model loading failed.
    pub const MODEL_LOAD_FAILED: u16 = 4;
    /// Embedding failed.
    pub const EMBEDDING_FAILED: u16 = 5;
    /// Reranking failed.
    pub const RERANK_FAILED: u16 = 6;
    /// Protocol version mismatch.
    pub const VERSION_MISMATCH: u16 = 7;
    /// Server overloaded (too many in-flight requests).
    pub const OVERLOADED: u16 = 8;
}

impl Response {
    /// Create an error response.
    #[must_use]
    pub fn error(code: u16, message: impl Into<String>) -> Self {
        Self::Error {
            code,
            message: message.into(),
        }
    }

    /// Create an internal error response.
    #[must_use]
    pub fn internal_error(message: impl Into<String>) -> Self {
        Self::error(error_codes::INTERNAL, message)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_envelope_roundtrip() {
        let req = Request::Health;
        let envelope = Envelope::from_request(42, &req).unwrap();

        assert_eq!(envelope.version, PROTOCOL_VERSION);
        assert_eq!(envelope.id, 42);

        let decoded = envelope.to_request().unwrap();
        assert!(matches!(decoded, Request::Health));
    }

    #[test]
    fn test_embed_request_roundtrip() {
        let req = Request::Embed {
            texts: vec!["hello".to_string(), "world".to_string()],
            model: "all-MiniLM-L6-v2".to_string(),
            dims: Some(256),
        };
        let envelope = Envelope::from_request(1, &req).unwrap();
        let decoded = envelope.to_request().unwrap();

        match decoded {
            Request::Embed { texts, model, dims } => {
                assert_eq!(texts, vec!["hello", "world"]);
                assert_eq!(model, "all-MiniLM-L6-v2");
                assert_eq!(dims, Some(256));
            }
            _ => panic!("expected Embed request"),
        }
    }

    #[test]
    fn test_response_roundtrip() {
        let resp = Response::Embeddings {
            vectors: vec![vec![0.1, 0.2, 0.3], vec![0.4, 0.5, 0.6]],
        };
        let envelope = Envelope::from_response(123, &resp).unwrap();
        let decoded = envelope.to_response().unwrap();

        match decoded {
            Response::Embeddings { vectors } => {
                assert_eq!(vectors.len(), 2);
                assert_eq!(vectors[0], vec![0.1, 0.2, 0.3]);
            }
            _ => panic!("expected Embeddings response"),
        }
    }

    #[test]
    fn test_error_response() {
        let resp = Response::error(error_codes::UNKNOWN_MODEL, "model not found");
        match resp {
            Response::Error { code, message } => {
                assert_eq!(code, error_codes::UNKNOWN_MODEL);
                assert_eq!(message, "model not found");
            }
            _ => panic!("expected Error response"),
        }
    }

    #[test]
    fn test_status_response_roundtrip() {
        let resp = Response::Status {
            uptime_secs: 3600,
            models: vec![LoadedModelInfo {
                name: "all-MiniLM-L6-v2".to_string(),
                model_type: "embedder".to_string(),
                loaded_at: 1700000000,
                requests_served: 100,
                last_used: 1700003600,
            }],
            rss_mb: 256.5,
            requests_served: 1000,
            in_flight: 2,
            queue_len: 0,
        };
        let envelope = Envelope::from_response(99, &resp).unwrap();
        let decoded = envelope.to_response().unwrap();

        match decoded {
            Response::Status {
                uptime_secs,
                models,
                in_flight,
                queue_len,
                ..
            } => {
                assert_eq!(uptime_secs, 3600);
                assert_eq!(models.len(), 1);
                assert_eq!(models[0].name, "all-MiniLM-L6-v2");
                assert_eq!(in_flight, 2);
                assert_eq!(queue_len, 0);
            }
            _ => panic!("expected Status response"),
        }
    }
}
