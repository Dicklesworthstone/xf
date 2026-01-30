//! Comprehensive tests for the daemon module.
//!
//! Tests cover:
//! - Protocol serialization/deserialization
//! - Resource configuration
//! - Memory monitoring
//! - Client configuration (ClientConfig, DaemonClient, DaemonConfig)
//! - ModelManager (loading, caching, eviction)
//! - Info types (HealthInfo, StatusInfo)
//! - Wire format
//! - Service files (systemd, launchd)

use std::path::PathBuf;
use std::time::Duration;
use xf::daemon::{
    ClientConfig, DaemonClient, DaemonConfig, Envelope, HealthInfo, IoPriority, LoadedModelInfo,
    MemoryMonitor, ModelManager, PROTOCOL_VERSION, Request, ResourceConfig, Response, StatusInfo,
    error_codes, get_process_rss_mb,
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

// =============================================================================
// Client Configuration Tests
// =============================================================================

mod client_config_tests {
    use super::*;

    #[test]
    fn test_client_config_defaults() {
        let config = ClientConfig::default();
        // Socket path should contain xf-daemon
        assert!(config.socket_path.to_string_lossy().contains("xf-daemon"));
        // PID path should contain xf-daemon
        assert!(config.pid_path.to_string_lossy().contains("xf-daemon"));
        // Connect timeout should be reasonable
        assert!(config.connect_timeout >= Duration::from_millis(100));
        assert!(config.connect_timeout <= Duration::from_secs(10));
        // Request timeout should be longer than connect timeout
        assert!(config.request_timeout > config.connect_timeout);
        // Max retries should be positive
        assert!(config.max_retries >= 1);
        // Auto-spawn enabled by default
        assert!(config.auto_spawn);
        // Spawn wait should be reasonable
        assert!(config.spawn_wait >= Duration::from_secs(1));
    }

    #[test]
    fn test_client_config_with_socket_path() {
        let custom_path = PathBuf::from("/custom/path/daemon.sock");
        let config = ClientConfig::default().with_socket_path(custom_path.clone());
        assert_eq!(config.socket_path, custom_path);
        // Other fields should remain default
        assert!(config.auto_spawn);
    }

    #[test]
    fn test_client_config_without_auto_spawn() {
        let config = ClientConfig::default().without_auto_spawn();
        assert!(!config.auto_spawn);
        // Other fields should remain default
        assert!(config.socket_path.to_string_lossy().contains("xf-daemon"));
    }

    #[test]
    fn test_client_config_builder_chaining() {
        let config = ClientConfig::default()
            .with_socket_path(PathBuf::from("/tmp/test.sock"))
            .without_auto_spawn();

        assert_eq!(config.socket_path, PathBuf::from("/tmp/test.sock"));
        assert!(!config.auto_spawn);
    }

    #[test]
    fn test_client_config_debug_impl() {
        let config = ClientConfig::default();
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("ClientConfig"));
        assert!(debug_str.contains("socket_path"));
        assert!(debug_str.contains("auto_spawn"));
    }

    #[test]
    fn test_client_config_clone() {
        let original = ClientConfig::default()
            .with_socket_path(PathBuf::from("/tmp/clone-test.sock"))
            .without_auto_spawn();

        let cloned = original.clone();

        assert_eq!(cloned.socket_path, original.socket_path);
        assert_eq!(cloned.auto_spawn, original.auto_spawn);
        assert_eq!(cloned.max_retries, original.max_retries);
    }

    #[test]
    fn test_socket_path_ends_with_sock() {
        let config = ClientConfig::default();
        assert!(
            config.socket_path.to_string_lossy().ends_with(".sock"),
            "Default socket path should end with .sock"
        );
    }

    #[test]
    fn test_pid_path_ends_with_pid() {
        let config = ClientConfig::default();
        assert!(
            config.pid_path.to_string_lossy().ends_with(".pid"),
            "Default pid path should end with .pid"
        );
    }
}

// =============================================================================
// DaemonClient Tests (no running daemon required)
// =============================================================================

mod daemon_client_tests {
    use super::*;

    #[test]
    fn test_daemon_client_new() {
        let client = DaemonClient::new();
        // Should create client without panic
        // Daemon should not be running in test environment
        assert!(!client.is_daemon_running());
    }

    #[test]
    fn test_daemon_client_with_config() {
        let config = ClientConfig::default().without_auto_spawn();
        let client = DaemonClient::with_config(config);
        assert!(!client.is_daemon_running());
    }

    #[test]
    fn test_daemon_client_with_socket_path() {
        let client = DaemonClient::with_socket_path(PathBuf::from("/tmp/nonexistent.sock"));
        assert!(!client.is_daemon_running());
    }

    #[test]
    fn test_daemon_client_default_trait() {
        let client = DaemonClient::default();
        // Should be equivalent to DaemonClient::new()
        assert!(!client.is_daemon_running());
    }

    #[test]
    fn test_daemon_pid_when_not_running() {
        // Use a custom config with BOTH a nonexistent socket AND a nonexistent pid file
        // to ensure the test doesn't read from a real daemon's pid file
        let config = ClientConfig {
            socket_path: PathBuf::from("/tmp/nonexistent-test.sock"),
            pid_path: PathBuf::from("/tmp/nonexistent-test.pid"),
            ..ClientConfig::default()
        };
        let client = DaemonClient::with_config(config);
        // PID should be None when daemon isn't running (no pid file exists)
        assert!(client.daemon_pid().is_none());
    }

    #[test]
    fn test_multiple_clients() {
        // Multiple clients should be able to coexist
        let client1 = DaemonClient::new();
        let client2 = DaemonClient::new();
        let client3 = DaemonClient::with_socket_path(PathBuf::from("/tmp/other.sock"));

        // All should report daemon not running
        assert!(!client1.is_daemon_running());
        assert!(!client2.is_daemon_running());
        assert!(!client3.is_daemon_running());
    }
}

// =============================================================================
// DaemonConfig Tests
// =============================================================================

mod daemon_config_tests {
    use super::*;

    #[test]
    fn test_daemon_config_defaults() {
        let config = DaemonConfig::default();
        // Socket path should contain xf-daemon
        assert!(config.socket_path.to_string_lossy().contains("xf-daemon"));
        // PID path should contain xf-daemon
        assert!(config.pid_path.to_string_lossy().contains("xf-daemon"));
        // Idle timeout should be reasonable
        assert!(config.idle_timeout >= Duration::from_secs(60));
        // Max models should be at least 1
        assert!(config.max_models >= 1);
        // Should have resource config
        assert!(config.resources.nice_level > 0);
    }

    #[test]
    fn test_daemon_config_with_paths() {
        let socket_path = PathBuf::from("/tmp/custom-daemon.sock");
        let pid_path = PathBuf::from("/tmp/custom-daemon.pid");
        let config = DaemonConfig::with_paths(socket_path.clone(), pid_path.clone());

        assert_eq!(config.socket_path, socket_path);
        assert_eq!(config.pid_path, pid_path);
    }

    #[test]
    fn test_daemon_config_with_idle_timeout() {
        let timeout = Duration::from_secs(600);
        let config = DaemonConfig::default().with_idle_timeout(timeout);
        assert_eq!(config.idle_timeout, timeout);
    }

    #[test]
    fn test_daemon_config_with_max_models() {
        let config = DaemonConfig::default().with_max_models(8);
        assert_eq!(config.max_models, 8);
    }

    #[test]
    fn test_daemon_config_builder_chaining() {
        let config = DaemonConfig::default()
            .with_idle_timeout(Duration::from_secs(120))
            .with_max_models(2);

        assert_eq!(config.idle_timeout, Duration::from_secs(120));
        assert_eq!(config.max_models, 2);
    }

    #[test]
    fn test_daemon_config_debug_impl() {
        let config = DaemonConfig::default();
        let debug_str = format!("{config:?}");
        assert!(debug_str.contains("DaemonConfig"));
        assert!(debug_str.contains("socket_path"));
        assert!(debug_str.contains("idle_timeout"));
    }

    #[test]
    fn test_daemon_config_clone() {
        let original = DaemonConfig::default()
            .with_idle_timeout(Duration::from_secs(300))
            .with_max_models(6);

        let cloned = original.clone();

        assert_eq!(cloned.socket_path, original.socket_path);
        assert_eq!(cloned.idle_timeout, original.idle_timeout);
        assert_eq!(cloned.max_models, original.max_models);
    }

    #[test]
    fn test_socket_path_includes_user() {
        let config = DaemonConfig::default();
        let socket_str = config.socket_path.to_string_lossy();
        // Should be in /tmp and contain xf-daemon
        assert!(socket_str.starts_with("/tmp/"));
        assert!(socket_str.contains("xf-daemon"));
    }

    #[test]
    fn test_daemon_config_with_resources() {
        let resources = ResourceConfig {
            nice_level: 15,
            max_threads: 4,
            memory_limit_mb: 1024,
            ..Default::default()
        };
        let config = DaemonConfig::default().with_resources(resources);

        assert_eq!(config.resources.nice_level, 15);
        assert_eq!(config.resources.max_threads, 4);
        assert_eq!(config.resources.memory_limit_mb, 1024);
    }
}

// =============================================================================
// ModelManager Tests
// =============================================================================

mod model_manager_tests {
    use super::*;

    #[test]
    fn test_model_manager_creation() {
        let manager = ModelManager::new(4);
        assert_eq!(manager.total_models(), 0);
    }

    #[test]
    fn test_model_manager_creation_with_zero() {
        // Edge case: zero max_models
        let manager = ModelManager::new(0);
        assert_eq!(manager.total_models(), 0);
    }

    #[test]
    fn test_model_manager_creation_with_large_limit() {
        let manager = ModelManager::new(1000);
        assert_eq!(manager.total_models(), 0);
    }

    #[test]
    fn test_model_manager_loaded_models_empty() {
        let manager = ModelManager::new(4);
        let models = manager.loaded_models();
        assert!(models.is_empty());
    }

    #[test]
    fn test_model_manager_hash_embedder() {
        let mut manager = ModelManager::new(4);
        // Hash embedder is always available (no ML model needed)
        let result = manager.get_embedder("hash");
        assert!(result.is_ok());
        assert_eq!(manager.total_models(), 1);
    }

    #[test]
    fn test_model_manager_hash_embedder_cached() {
        let mut manager = ModelManager::new(4);
        // Load once
        manager.get_embedder("hash").unwrap();
        assert_eq!(manager.total_models(), 1);

        // Load again - should reuse cache
        manager.get_embedder("hash").unwrap();
        assert_eq!(manager.total_models(), 1);
    }

    #[test]
    fn test_model_manager_none_reranker() {
        let mut manager = ModelManager::new(4);
        // "none" reranker should return None without loading anything
        let result = manager.get_reranker("none");
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
        assert_eq!(manager.total_models(), 0);
    }

    #[test]
    fn test_model_manager_empty_reranker() {
        let mut manager = ModelManager::new(4);
        // Empty string should return None like "none"
        let result = manager.get_reranker("");
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
        assert_eq!(manager.total_models(), 0);
    }

    #[test]
    fn test_model_manager_unknown_embedder() {
        let mut manager = ModelManager::new(4);
        let result = manager.get_embedder("nonexistent-model-xyz");
        assert!(result.is_err());
        assert_eq!(manager.total_models(), 0);
    }

    #[test]
    fn test_model_manager_unload_all() {
        let mut manager = ModelManager::new(4);
        manager.get_embedder("hash").unwrap();
        assert_eq!(manager.total_models(), 1);

        manager.unload_all();
        assert_eq!(manager.total_models(), 0);
    }
}

// =============================================================================
// HealthInfo and StatusInfo Tests
// =============================================================================

mod info_types_tests {
    use super::*;

    #[test]
    fn test_health_info_serialization() {
        let info = HealthInfo {
            uptime_secs: 42,
            models_loaded: 2,
        };
        let json = serde_json::to_string(&info).expect("serialize HealthInfo");
        assert!(json.contains("42"));
        assert!(json.contains("uptime_secs"));
        assert!(json.contains("models_loaded"));
    }

    #[test]
    fn test_health_info_debug() {
        let info = HealthInfo {
            uptime_secs: 100,
            models_loaded: 3,
        };
        let debug_str = format!("{info:?}");
        assert!(debug_str.contains("HealthInfo"));
    }

    #[test]
    fn test_health_info_clone() {
        let original = HealthInfo {
            uptime_secs: 50,
            models_loaded: 1,
        };
        let cloned = original.clone();
        assert_eq!(cloned.uptime_secs, original.uptime_secs);
        assert_eq!(cloned.models_loaded, original.models_loaded);
    }

    #[test]
    fn test_status_info_serialization() {
        let info = StatusInfo {
            uptime_secs: 1000,
            models: vec![LoadedModelInfo {
                name: "test-model".into(),
                model_type: "embedder".into(),
                loaded_at: 1_700_000_000,
                requests_served: 500,
                last_used: 1_700_001_000,
            }],
            rss_mb: 256.5,
            requests_served: 10000,
            in_flight: 5,
            queue_len: 2,
        };
        let json = serde_json::to_string(&info).expect("serialize StatusInfo");
        assert!(json.contains("1000"));
        assert!(json.contains("test-model"));
        assert!(json.contains("256.5"));
    }

    #[test]
    fn test_status_info_debug() {
        let info = StatusInfo {
            uptime_secs: 200,
            models: vec![],
            rss_mb: 128.0,
            requests_served: 100,
            in_flight: 0,
            queue_len: 0,
        };
        let debug_str = format!("{info:?}");
        assert!(debug_str.contains("StatusInfo"));
    }

    #[test]
    #[allow(clippy::float_cmp)]
    fn test_status_info_clone() {
        let original = StatusInfo {
            uptime_secs: 300,
            models: vec![],
            rss_mb: 64.0,
            requests_served: 50,
            in_flight: 1,
            queue_len: 0,
        };
        let cloned = original.clone();
        assert_eq!(cloned.uptime_secs, original.uptime_secs);
        assert_eq!(cloned.rss_mb, original.rss_mb);
        assert_eq!(cloned.in_flight, original.in_flight);
    }

    #[test]
    fn test_status_info_with_multiple_models() {
        let info = StatusInfo {
            uptime_secs: 500,
            models: vec![
                LoadedModelInfo {
                    name: "embedder1".into(),
                    model_type: "embedder".into(),
                    loaded_at: 1_700_000_000,
                    requests_served: 100,
                    last_used: 1_700_000_500,
                },
                LoadedModelInfo {
                    name: "reranker1".into(),
                    model_type: "reranker".into(),
                    loaded_at: 1_700_000_100,
                    requests_served: 50,
                    last_used: 1_700_000_400,
                },
            ],
            rss_mb: 512.0,
            requests_served: 1000,
            in_flight: 3,
            queue_len: 1,
        };
        assert_eq!(info.models.len(), 2);
        let json = serde_json::to_string(&info).expect("serialize");
        assert!(json.contains("embedder1"));
        assert!(json.contains("reranker1"));
    }
}

// =============================================================================
// Service File Validation Tests
// =============================================================================

mod service_file_tests {
    #[test]
    fn test_systemd_service_file_valid() {
        let content = include_str!("../scripts/daemon/xf-daemon.service");
        // Basic INI section validation
        assert!(content.contains("[Unit]"), "Missing [Unit] section");
        assert!(content.contains("[Service]"), "Missing [Service] section");
        assert!(content.contains("[Install]"), "Missing [Install] section");
        // Required directives
        assert!(content.contains("ExecStart="), "Missing ExecStart");
        assert!(content.contains("ExecStop="), "Missing ExecStop");
        assert!(content.contains("Type="), "Missing Type");
        // Resource limits
        assert!(content.contains("Nice="), "Missing Nice level");
        assert!(content.contains("MemoryMax="), "Missing MemoryMax");
        // Security hardening
        assert!(
            content.contains("NoNewPrivileges=true"),
            "Missing NoNewPrivileges"
        );
        assert!(content.contains("ProtectSystem="), "Missing ProtectSystem");
    }

    #[test]
    fn test_launchd_plist_valid() {
        let content = include_str!("../scripts/daemon/com.dicklesworthstone.xf-daemon.plist");
        // Basic XML validation
        assert!(
            content.contains("<!DOCTYPE plist"),
            "Missing DOCTYPE declaration"
        );
        assert!(content.contains("<key>Label</key>"), "Missing Label key");
        assert!(
            content.contains("<key>ProgramArguments</key>"),
            "Missing ProgramArguments"
        );
        assert!(
            content.contains("<key>RunAtLoad</key>"),
            "Missing RunAtLoad"
        );
        assert!(
            content.contains("<key>KeepAlive</key>"),
            "Missing KeepAlive"
        );
        // Resource settings
        assert!(content.contains("<key>Nice</key>"), "Missing Nice key");
        assert!(
            content.contains("<key>LowPriorityIO</key>"),
            "Missing LowPriorityIO"
        );
        assert!(
            content.contains("<key>ProcessType</key>"),
            "Missing ProcessType"
        );
        assert!(
            content.contains("<string>Background</string>"),
            "Missing Background process type"
        );
    }

    #[test]
    fn test_install_scripts_exist() {
        let systemd_script = include_str!("../scripts/daemon/install-systemd.sh");
        assert!(
            systemd_script.contains("systemctl --user"),
            "systemd script should use user units"
        );
        assert!(
            systemd_script.contains("daemon-reload"),
            "systemd script should reload daemon"
        );

        let launchd_script = include_str!("../scripts/daemon/install-launchd.sh");
        assert!(
            launchd_script.contains("launchctl"),
            "launchd script should use launchctl"
        );
        assert!(
            launchd_script.contains("bootstrap"),
            "launchd script should bootstrap agent"
        );
    }

    #[test]
    fn test_readme_exists() {
        let readme = include_str!("../scripts/daemon/README.md");
        assert!(readme.contains("systemd"), "README should mention systemd");
        assert!(readme.contains("launchd"), "README should mention launchd");
        assert!(
            readme.contains("install"),
            "README should have install instructions"
        );
    }
}
