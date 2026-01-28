//! Comprehensive tests for the daemon module.
//!
//! Tests cover:
//! - Protocol serialization/deserialization
//! - Resource configuration
//! - Memory monitoring
//! - Client configuration

use xf::daemon::{
    Envelope, IoPriority, LoadedModelInfo, MemoryMonitor, PROTOCOL_VERSION, Request,
    ResourceConfig, Response, error_codes, get_process_rss_mb,
};

// =============================================================================
// Protocol Serialization Tests
// =============================================================================

mod protocol_tests {
    use super::*;

    #[test]
    fn test_all_request_variants_roundtrip() {
        let requests = vec![
            Request::Health,
            Request::Embed {
                texts: vec!["hello".into(), "world".into()],
                model: "all-MiniLM-L6-v2".into(),
                dims: None,
            },
            Request::Embed {
                texts: vec!["test".into()],
                model: "default".into(),
                dims: Some(256),
            },
            Request::Rerank {
                query: "test query".into(),
                documents: vec!["doc1".into(), "doc2".into()],
                model: "flashrank".into(),
            },
            Request::Status,
            Request::Shutdown,
        ];

        for req in &requests {
            let bytes = rmp_serde::to_vec(req).expect("serialize request");
            let decoded: Request = rmp_serde::from_slice(&bytes).expect("deserialize request");
            assert_eq!(format!("{decoded:?}"), format!("{req:?}"));
        }
    }

    #[test]
    fn test_all_response_variants_roundtrip() {
        let responses = vec![
            Response::Health {
                uptime_secs: 42,
                models_loaded: 2,
            },
            Response::Embeddings {
                vectors: vec![vec![1.0, 2.0, 3.0]],
            },
            Response::Scores {
                scores: vec![0.9, 0.5, 0.1],
            },
            Response::Status {
                uptime_secs: 100,
                models: vec![LoadedModelInfo {
                    name: "minilm".into(),
                    model_type: "embedder".into(),
                    loaded_at: 1_700_000_000,
                    requests_served: 50,
                    last_used: 1_700_000_100,
                }],
                rss_mb: 128.5,
                requests_served: 1000,
                in_flight: 2,
                queue_len: 0,
            },
            Response::Error {
                code: error_codes::UNKNOWN_MODEL,
                message: "not found".into(),
            },
            Response::Shutdown { ok: true },
        ];

        for resp in &responses {
            let bytes = rmp_serde::to_vec(resp).expect("serialize response");
            let decoded: Response = rmp_serde::from_slice(&bytes).expect("deserialize response");
            assert_eq!(format!("{decoded:?}"), format!("{resp:?}"));
        }
    }

    #[test]
    #[allow(clippy::cast_possible_truncation)]
    fn test_envelope_framing() {
        let envelope = Envelope::from_request(12345, &Request::Health).expect("create envelope");
        assert_eq!(envelope.version, PROTOCOL_VERSION);
        assert_eq!(envelope.id, 12345);

        // Test length-prefixed encoding (what goes on the wire)
        let payload = rmp_serde::to_vec(&envelope).expect("serialize envelope");
        let mut buf = Vec::new();
        buf.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        buf.extend_from_slice(&payload);

        // Decode
        let len = u32::from_be_bytes(buf[0..4].try_into().unwrap()) as usize;
        let decoded: Envelope =
            rmp_serde::from_slice(&buf[4..4 + len]).expect("deserialize envelope");
        assert_eq!(decoded.version, PROTOCOL_VERSION);
        assert_eq!(decoded.id, 12345);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_large_payload_serialization() {
        // 100 embeddings of 768 dimensions each
        let embeddings: Vec<Vec<f32>> = (0..100)
            .map(|_| (0..768).map(|i| i as f32 * 0.001).collect())
            .collect();
        let resp = Response::Embeddings {
            vectors: embeddings,
        };
        let bytes = rmp_serde::to_vec(&resp).expect("serialize large embeddings");
        println!("100x768 embeddings serialized to {} bytes", bytes.len());
        assert!(
            bytes.len() < 400_000,
            "Should be compact: {} bytes",
            bytes.len()
        );

        let decoded: Response = rmp_serde::from_slice(&bytes).expect("deserialize");
        match decoded {
            Response::Embeddings { vectors } => assert_eq!(vectors.len(), 100),
            _ => panic!("expected Embeddings"),
        }
    }

    #[test]
    fn test_error_codes_are_distinct() {
        let codes = [
            error_codes::INTERNAL,
            error_codes::INVALID_REQUEST,
            error_codes::UNKNOWN_MODEL,
            error_codes::MODEL_LOAD_FAILED,
            error_codes::EMBEDDING_FAILED,
            error_codes::RERANK_FAILED,
            error_codes::VERSION_MISMATCH,
            error_codes::OVERLOADED,
        ];
        let mut seen = std::collections::HashSet::new();
        for code in codes {
            assert!(seen.insert(code), "Duplicate error code: {code}");
        }
    }

    #[test]
    fn test_error_code_values() {
        assert_eq!(error_codes::INTERNAL, 1);
        assert_eq!(error_codes::INVALID_REQUEST, 2);
        assert_eq!(error_codes::UNKNOWN_MODEL, 3);
        assert_eq!(error_codes::MODEL_LOAD_FAILED, 4);
        assert_eq!(error_codes::EMBEDDING_FAILED, 5);
        assert_eq!(error_codes::RERANK_FAILED, 6);
        assert_eq!(error_codes::VERSION_MISMATCH, 7);
        assert_eq!(error_codes::OVERLOADED, 8);
    }

    #[test]
    fn test_response_error_helper() {
        let resp = Response::error(error_codes::UNKNOWN_MODEL, "model not found");
        match resp {
            Response::Error { code, message } => {
                assert_eq!(code, error_codes::UNKNOWN_MODEL);
                assert_eq!(message, "model not found");
            }
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn test_response_internal_error_helper() {
        let resp = Response::internal_error("something went wrong");
        match resp {
            Response::Error { code, message } => {
                assert_eq!(code, error_codes::INTERNAL);
                assert_eq!(message, "something went wrong");
            }
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn test_empty_texts_embed() {
        let req = Request::Embed {
            texts: vec![],
            model: "test".into(),
            dims: None,
        };
        let bytes = rmp_serde::to_vec(&req).expect("serialize");
        let decoded: Request = rmp_serde::from_slice(&bytes).expect("deserialize");
        match decoded {
            Request::Embed { texts, .. } => assert!(texts.is_empty()),
            _ => panic!("expected Embed"),
        }
    }

    #[test]
    fn test_envelope_from_response() {
        let resp = Response::Health {
            uptime_secs: 42,
            models_loaded: 1,
        };
        let envelope = Envelope::from_response(999, &resp).expect("create envelope");
        assert_eq!(envelope.id, 999);
        let decoded = envelope.to_response().expect("decode response");
        match decoded {
            Response::Health {
                uptime_secs,
                models_loaded,
            } => {
                assert_eq!(uptime_secs, 42);
                assert_eq!(models_loaded, 1);
            }
            _ => panic!("expected Health"),
        }
    }

    #[test]
    fn test_loaded_model_info_serialization() {
        let info = LoadedModelInfo {
            name: "test-model".into(),
            model_type: "embedder".into(),
            loaded_at: 1_700_000_000,
            requests_served: 42,
            last_used: 1_700_001_000,
        };
        let bytes = rmp_serde::to_vec(&info).expect("serialize");
        let decoded: LoadedModelInfo = rmp_serde::from_slice(&bytes).expect("deserialize");
        assert_eq!(decoded.name, "test-model");
        assert_eq!(decoded.model_type, "embedder");
        assert_eq!(decoded.requests_served, 42);
    }
}

// =============================================================================
// Resource Configuration Tests
// =============================================================================

mod resource_tests {
    use super::*;

    #[test]
    fn test_resource_config_defaults() {
        let config = ResourceConfig::default();
        assert_eq!(config.nice_level, 10);
        assert!(matches!(config.io_priority, IoPriority::Idle));
        assert_eq!(config.memory_limit_mb, 2048);
        assert!(config.max_threads >= 1);
        // Idle timeout should be at least a minute
        assert!(config.idle_timeout.as_secs() >= 60);
    }

    #[test]
    fn test_io_priority_idle_default() {
        let config = ResourceConfig::default();
        match config.io_priority {
            IoPriority::Idle => (),
            _ => panic!("expected Idle as default IO priority"),
        }
    }

    #[test]
    fn test_thread_pool_sizing() {
        let cpus = num_cpus::get();
        let config = ResourceConfig::default();
        // Default should be capped at reasonable number
        assert!(config.max_threads <= cpus);
        assert!(config.max_threads >= 1);
    }

    #[test]
    fn test_effective_threads_caps_at_cpus() {
        let cpus = num_cpus::get();
        let config = ResourceConfig {
            max_threads: 1000, // Unreasonably high
            ..Default::default()
        };
        assert_eq!(config.effective_threads(), cpus);
    }

    #[test]
    fn test_effective_threads_respects_minimum() {
        let config = ResourceConfig {
            max_threads: 1,
            ..Default::default()
        };
        assert_eq!(config.effective_threads(), 1);
    }

    // Note: from_toml is tested in src/daemon/resource.rs unit tests
    // These tests focus on the public API

    #[test]
    fn test_get_process_rss() {
        let rss = get_process_rss_mb();
        assert!(rss >= 0.0, "RSS should be non-negative: {rss}");
        // On non-Linux/macOS platforms, this returns 0.0
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        assert!(rss > 0.0, "RSS should be positive on Linux/macOS: {rss}");
        assert!(rss < 10000.0, "RSS should be reasonable: {rss}");
        println!("Current process RSS: {rss:.2} MB");
    }
}

// =============================================================================
// Memory Monitor Tests
// =============================================================================

mod memory_monitor_tests {
    use super::*;

    #[test]
    fn test_memory_monitor_creation() {
        let mut monitor = MemoryMonitor::new(0.85);
        // Should not panic
        let _ = monitor.is_under_pressure();
    }

    #[test]
    fn test_memory_monitor_high_threshold() {
        let mut monitor = MemoryMonitor::new(0.99);
        // With 99% threshold, we're unlikely to be under pressure
        // Just verify it doesn't panic
        let _ = monitor.is_under_pressure();
    }

    #[test]
    fn test_memory_monitor_available_mb() {
        let monitor = MemoryMonitor::new(0.85);
        let available = monitor.available_mb();
        // Should return some value (even if 0 on unsupported platforms)
        println!("Available memory: {available} MB");
    }
}

// =============================================================================
// Wire Format Tests
// =============================================================================

mod wire_format_tests {
    use super::*;

    #[test]
    fn test_msgpack_is_compact() {
        let req = Request::Health;
        let bytes = rmp_serde::to_vec(&req).unwrap();
        // Health request should be very small
        assert!(
            bytes.len() < 20,
            "Health request too large: {} bytes",
            bytes.len()
        );
    }

    #[test]
    fn test_embeddings_binary_format() {
        // Verify embeddings are stored efficiently
        let vectors: Vec<Vec<f32>> = vec![vec![1.0; 384]];
        let resp = Response::Embeddings { vectors };
        let bytes = rmp_serde::to_vec(&resp).unwrap();

        // 384 floats * 4 bytes = 1536 bytes minimum, plus overhead
        // MsgPack may use more due to array framing but should be efficient
        println!("384-dim embedding response size: {} bytes", bytes.len());
        assert!(
            bytes.len() < 3000,
            "Embeddings too large: {} bytes",
            bytes.len()
        );
    }

    #[test]
    fn test_envelope_version_field() {
        let envelope = Envelope::from_request(1, &Request::Health).unwrap();
        assert_eq!(envelope.version, 1, "Current protocol version should be 1");
    }

    #[test]
    fn test_request_id_correlation() {
        // Multiple requests should have different IDs
        let e1 = Envelope::from_request(1, &Request::Health).unwrap();
        let e2 = Envelope::from_request(2, &Request::Status).unwrap();
        assert_ne!(e1.id, e2.id);
    }
}

// =============================================================================
// Error Response Tests
// =============================================================================

mod error_tests {
    use super::*;

    #[test]
    fn test_error_response_structure() {
        let resp = Response::Error {
            code: error_codes::INTERNAL,
            message: "test error".into(),
        };
        let bytes = rmp_serde::to_vec(&resp).unwrap();
        let decoded: Response = rmp_serde::from_slice(&bytes).unwrap();

        match decoded {
            Response::Error { code, message } => {
                assert_eq!(code, error_codes::INTERNAL);
                assert_eq!(message, "test error");
            }
            _ => panic!("expected Error"),
        }
    }

    #[test]
    fn test_all_error_codes_serialize() {
        for code in [
            error_codes::INTERNAL,
            error_codes::INVALID_REQUEST,
            error_codes::UNKNOWN_MODEL,
            error_codes::MODEL_LOAD_FAILED,
            error_codes::EMBEDDING_FAILED,
            error_codes::RERANK_FAILED,
            error_codes::VERSION_MISMATCH,
            error_codes::OVERLOADED,
        ] {
            let resp = Response::error(code, "test");
            let bytes = rmp_serde::to_vec(&resp).unwrap();
            let decoded: Response = rmp_serde::from_slice(&bytes).unwrap();
            match decoded {
                Response::Error { code: c, .. } => assert_eq!(c, code),
                _ => panic!("expected Error"),
            }
        }
    }
}
