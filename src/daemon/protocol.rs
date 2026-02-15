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
    pub const fn new(id: u64, payload: Vec<u8>) -> Self {
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

    /// Submit a background embedding job.
    SubmitEmbeddingJob {
        /// Path to the SQLite database containing documents.
        db_path: String,
        /// Path to the index directory (for writing vector index).
        index_path: String,
        /// Model ID for storing embeddings ("fast", "quality", or custom).
        model_id: String,
        /// Use two-tier embedding (fast + quality).
        two_tier: bool,
        /// Fast tier model name (if two_tier).
        fast_model: Option<String>,
        /// Quality tier model name (if two_tier).
        quality_model: Option<String>,
    },

    /// Get status of embedding jobs for a database.
    EmbeddingJobStatus {
        /// Path to the database (or None for all jobs).
        db_path: Option<String>,
    },

    /// Cancel embedding jobs.
    CancelEmbeddingJob {
        /// Path to the database.
        db_path: String,
        /// Specific model to cancel (or None for all models).
        model_id: Option<String>,
    },
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

    /// Embedding job submitted successfully.
    JobSubmitted {
        /// Job ID assigned by the daemon.
        job_id: i64,
        /// Number of documents to embed.
        estimated_docs: i64,
    },

    /// Embedding job status response.
    JobStatus {
        /// List of jobs matching the query.
        jobs: Vec<EmbeddingJobInfo>,
    },

    /// Job cancellation response.
    JobCancelled {
        /// Number of jobs cancelled.
        cancelled_count: usize,
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

/// Information about an embedding job (for protocol responses).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingJobInfo {
    /// Job ID.
    pub id: i64,
    /// Database path.
    pub db_path: String,
    /// Model ID (e.g., "fast", "quality").
    pub model_id: String,
    /// Job status ("pending", "in_progress", "completed", "failed").
    pub status: String,
    /// Total documents to process.
    pub total_docs: i64,
    /// Documents completed so far.
    pub completed_docs: i64,
    /// Progress percentage (0.0 to 100.0).
    pub progress_pct: f32,
    /// Creation timestamp (unix).
    pub created_at: i64,
    /// Start timestamp (unix), 0 if not started.
    pub started_at: i64,
    /// Completion timestamp (unix), 0 if not completed.
    pub completed_at: i64,
    /// Error message if failed.
    pub error_message: Option<String>,
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
    /// Job-related error.
    pub const JOB_ERROR: u16 = 9;
    /// Database not found or inaccessible.
    pub const DB_NOT_FOUND: u16 = 10;
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
    fn test_internal_error() {
        let resp = Response::internal_error("something broke");
        match resp {
            Response::Error { code, message } => {
                assert_eq!(code, error_codes::INTERNAL);
                assert_eq!(message, "something broke");
            }
            _ => panic!("expected Error response"),
        }
    }

    #[test]
    fn test_rerank_request_roundtrip() {
        let req = Request::Rerank {
            query: "search query".to_string(),
            documents: vec!["doc one".to_string(), "doc two".to_string()],
            model: "cross-encoder".to_string(),
        };
        let envelope = Envelope::from_request(7, &req).unwrap();
        let decoded = envelope.to_request().unwrap();

        match decoded {
            Request::Rerank {
                query,
                documents,
                model,
            } => {
                assert_eq!(query, "search query");
                assert_eq!(documents, vec!["doc one", "doc two"]);
                assert_eq!(model, "cross-encoder");
            }
            _ => panic!("expected Rerank request"),
        }
    }

    #[test]
    fn test_shutdown_request_roundtrip() {
        let req = Request::Shutdown;
        let envelope = Envelope::from_request(10, &req).unwrap();
        let decoded = envelope.to_request().unwrap();
        assert!(matches!(decoded, Request::Shutdown));
    }

    #[test]
    fn test_status_request_roundtrip() {
        let req = Request::Status;
        let envelope = Envelope::from_request(11, &req).unwrap();
        let decoded = envelope.to_request().unwrap();
        assert!(matches!(decoded, Request::Status));
    }

    #[test]
    fn test_health_response_roundtrip() {
        let resp = Response::Health {
            uptime_secs: 120,
            models_loaded: 3,
        };
        let envelope = Envelope::from_response(50, &resp).unwrap();
        let decoded = envelope.to_response().unwrap();

        match decoded {
            Response::Health {
                uptime_secs,
                models_loaded,
            } => {
                assert_eq!(uptime_secs, 120);
                assert_eq!(models_loaded, 3);
            }
            _ => panic!("expected Health response"),
        }
    }

    #[test]
    fn test_shutdown_response_roundtrip() {
        let resp = Response::Shutdown { ok: true };
        let envelope = Envelope::from_response(60, &resp).unwrap();
        let decoded = envelope.to_response().unwrap();

        match decoded {
            Response::Shutdown { ok } => assert!(ok),
            _ => panic!("expected Shutdown response"),
        }

        // Also test ok: false
        let resp_false = Response::Shutdown { ok: false };
        let envelope2 = Envelope::from_response(61, &resp_false).unwrap();
        let decoded2 = envelope2.to_response().unwrap();
        match decoded2 {
            Response::Shutdown { ok } => assert!(!ok),
            _ => panic!("expected Shutdown response"),
        }
    }

    #[test]
    fn test_scores_response_roundtrip() {
        let resp = Response::Scores {
            scores: vec![0.95, 0.72, 0.31],
        };
        let envelope = Envelope::from_response(70, &resp).unwrap();
        let decoded = envelope.to_response().unwrap();

        match decoded {
            Response::Scores { scores } => {
                assert_eq!(scores.len(), 3);
                assert!((scores[0] - 0.95).abs() < f32::EPSILON);
                assert!((scores[1] - 0.72).abs() < f32::EPSILON);
                assert!((scores[2] - 0.31).abs() < f32::EPSILON);
            }
            _ => panic!("expected Scores response"),
        }
    }

    #[test]
    fn test_envelope_new_sets_version() {
        let envelope = Envelope::new(99, vec![1, 2, 3]);
        assert_eq!(envelope.version, PROTOCOL_VERSION);
        assert_eq!(envelope.id, 99);
        assert_eq!(envelope.payload, vec![1, 2, 3]);
    }

    #[test]
    fn test_embed_request_no_dims() {
        let req = Request::Embed {
            texts: vec!["test".to_string()],
            model: "default".to_string(),
            dims: None,
        };
        let envelope = Envelope::from_request(5, &req).unwrap();
        let decoded = envelope.to_request().unwrap();

        match decoded {
            Request::Embed { dims, .. } => {
                assert!(dims.is_none());
            }
            _ => panic!("expected Embed request"),
        }
    }

    #[test]
    fn test_status_response_roundtrip() {
        let resp = Response::Status {
            uptime_secs: 3600,
            models: vec![LoadedModelInfo {
                name: "all-MiniLM-L6-v2".to_string(),
                model_type: "embedder".to_string(),
                loaded_at: 1_700_000_000,
                requests_served: 100,
                last_used: 1_700_003_600,
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

    #[test]
    fn test_submit_embedding_job_roundtrip() {
        let req = Request::SubmitEmbeddingJob {
            db_path: "/path/to/xf.db".to_string(),
            index_path: "/path/to/index".to_string(),
            model_id: "quality".to_string(),
            two_tier: true,
            fast_model: Some("hash-fnv1a-384".to_string()),
            quality_model: Some("all-MiniLM-L6-v2".to_string()),
        };
        let envelope = Envelope::from_request(100, &req).unwrap();
        let decoded = envelope.to_request().unwrap();

        match decoded {
            Request::SubmitEmbeddingJob {
                db_path,
                two_tier,
                fast_model,
                ..
            } => {
                assert_eq!(db_path, "/path/to/xf.db");
                assert!(two_tier);
                assert_eq!(fast_model, Some("hash-fnv1a-384".to_string()));
            }
            _ => panic!("expected SubmitEmbeddingJob request"),
        }
    }

    #[test]
    fn test_embedding_job_status_roundtrip() {
        let req = Request::EmbeddingJobStatus {
            db_path: Some("/path/to/xf.db".to_string()),
        };
        let envelope = Envelope::from_request(101, &req).unwrap();
        let decoded = envelope.to_request().unwrap();

        match decoded {
            Request::EmbeddingJobStatus { db_path } => {
                assert_eq!(db_path, Some("/path/to/xf.db".to_string()));
            }
            _ => panic!("expected EmbeddingJobStatus request"),
        }
    }

    #[test]
    fn test_cancel_embedding_job_roundtrip() {
        let req = Request::CancelEmbeddingJob {
            db_path: "/path/to/xf.db".to_string(),
            model_id: Some("fast".to_string()),
        };
        let envelope = Envelope::from_request(102, &req).unwrap();
        let decoded = envelope.to_request().unwrap();

        match decoded {
            Request::CancelEmbeddingJob { db_path, model_id } => {
                assert_eq!(db_path, "/path/to/xf.db");
                assert_eq!(model_id, Some("fast".to_string()));
            }
            _ => panic!("expected CancelEmbeddingJob request"),
        }
    }

    #[test]
    fn test_job_submitted_response_roundtrip() {
        let resp = Response::JobSubmitted {
            job_id: 42,
            estimated_docs: 68000,
        };
        let envelope = Envelope::from_response(103, &resp).unwrap();
        let decoded = envelope.to_response().unwrap();

        match decoded {
            Response::JobSubmitted {
                job_id,
                estimated_docs,
            } => {
                assert_eq!(job_id, 42);
                assert_eq!(estimated_docs, 68000);
            }
            _ => panic!("expected JobSubmitted response"),
        }
    }

    #[test]
    fn test_job_status_response_roundtrip() {
        let resp = Response::JobStatus {
            jobs: vec![EmbeddingJobInfo {
                id: 1,
                db_path: "/path/to/db".to_string(),
                model_id: "fast".to_string(),
                status: "in_progress".to_string(),
                total_docs: 1000,
                completed_docs: 500,
                progress_pct: 50.0,
                created_at: 1_700_000_000,
                started_at: 1_700_000_100,
                completed_at: 0,
                error_message: None,
            }],
        };
        let envelope = Envelope::from_response(104, &resp).unwrap();
        let decoded = envelope.to_response().unwrap();

        match decoded {
            Response::JobStatus { jobs } => {
                assert_eq!(jobs.len(), 1);
                assert_eq!(jobs[0].id, 1);
                assert_eq!(jobs[0].status, "in_progress");
                assert!((jobs[0].progress_pct - 50.0).abs() < f32::EPSILON);
            }
            _ => panic!("expected JobStatus response"),
        }
    }

    #[test]
    fn test_job_cancelled_response_roundtrip() {
        let resp = Response::JobCancelled { cancelled_count: 2 };
        let envelope = Envelope::from_response(105, &resp).unwrap();
        let decoded = envelope.to_response().unwrap();

        match decoded {
            Response::JobCancelled { cancelled_count } => {
                assert_eq!(cancelled_count, 2);
            }
            _ => panic!("expected JobCancelled response"),
        }
    }
}
