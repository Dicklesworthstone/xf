//! Daemon core with Unix Domain Socket listener and request handling.
//!
//! The daemon keeps embedding and reranking models warm in memory,
//! accepting requests over a Unix Domain Socket with MessagePack protocol.
//!
//! Note: This module is only functional on Unix platforms. On Windows,
//! the daemon types exist but attempting to run will return an error.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use tokio::fs;
#[cfg(unix)]
use tokio::io::{AsyncReadExt, AsyncWriteExt};
#[cfg(unix)]
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::Mutex;

use super::models::ModelManager;
use super::protocol::{Envelope, PROTOCOL_VERSION, Request, Response, error_codes};

/// Default idle timeout before daemon shuts down (30 minutes).
const DEFAULT_IDLE_TIMEOUT_SECS: u64 = 30 * 60;

/// Default maximum number of models to keep loaded.
const DEFAULT_MAX_MODELS: usize = 4;

/// Maximum concurrent in-flight requests before returning overload error.
const MAX_IN_FLIGHT: usize = 64;

/// Daemon configuration.
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    /// Path to Unix Domain Socket.
    pub socket_path: PathBuf,
    /// Path to PID file.
    pub pid_path: PathBuf,
    /// Idle timeout before auto-shutdown.
    pub idle_timeout: Duration,
    /// Maximum models to keep loaded.
    pub max_models: usize,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        // Get user identifier for unique socket paths (safe alternative to getuid)
        let user_id = std::env::var("USER")
            .or_else(|_| std::env::var("USERNAME"))
            .unwrap_or_else(|_| "default".to_string());
        let socket_path = PathBuf::from(format!("/tmp/xf-daemon-{user_id}.sock"));
        let pid_path = PathBuf::from(format!("/tmp/xf-daemon-{user_id}.pid"));

        Self {
            socket_path,
            pid_path,
            idle_timeout: Duration::from_secs(DEFAULT_IDLE_TIMEOUT_SECS),
            max_models: DEFAULT_MAX_MODELS,
        }
    }
}

impl DaemonConfig {
    /// Create config with custom paths.
    #[must_use]
    pub fn with_paths(socket_path: PathBuf, pid_path: PathBuf) -> Self {
        Self {
            socket_path,
            pid_path,
            ..Self::default()
        }
    }

    /// Set idle timeout.
    #[must_use]
    pub const fn with_idle_timeout(mut self, timeout: Duration) -> Self {
        self.idle_timeout = timeout;
        self
    }

    /// Set maximum models.
    #[must_use]
    pub const fn with_max_models(mut self, max: usize) -> Self {
        self.max_models = max;
        self
    }
}

/// Shared daemon state.
struct DaemonState {
    models: ModelManager,
    start_time: Instant,
    last_request: Instant,
    requests_served: AtomicU64,
    in_flight: AtomicU64,
    shutdown_requested: bool,
}

impl DaemonState {
    fn new(max_models: usize) -> Self {
        let now = Instant::now();
        Self {
            models: ModelManager::new(max_models),
            start_time: now,
            last_request: now,
            requests_served: AtomicU64::new(0),
            in_flight: AtomicU64::new(0),
            shutdown_requested: false,
        }
    }

    fn uptime_secs(&self) -> u64 {
        self.start_time.elapsed().as_secs()
    }

    fn touch(&mut self) {
        self.last_request = Instant::now();
    }
}

/// The model daemon.
pub struct ModelDaemon {
    config: DaemonConfig,
    state: Arc<Mutex<DaemonState>>,
}

impl ModelDaemon {
    /// Create a new daemon with the given configuration.
    #[must_use]
    pub fn new(config: DaemonConfig) -> Self {
        let state = Arc::new(Mutex::new(DaemonState::new(config.max_models)));
        Self { config, state }
    }

    /// Run the daemon, accepting connections until shutdown.
    pub async fn run(&self) -> anyhow::Result<()> {
        // Clean up stale socket if it exists
        if self.config.socket_path.exists() {
            fs::remove_file(&self.config.socket_path).await?;
        }

        // Bind the Unix socket
        let listener = UnixListener::bind(&self.config.socket_path)?;

        // Set socket permissions to owner-only (0600)
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let perms = std::fs::Permissions::from_mode(0o600);
            std::fs::set_permissions(&self.config.socket_path, perms)?;
        }

        // Write PID file
        let pid = std::process::id();
        fs::write(&self.config.pid_path, pid.to_string()).await?;

        tracing::info!(
            socket = %self.config.socket_path.display(),
            pid = pid,
            "daemon ready"
        );

        // Accept loop with idle timeout
        loop {
            let idle_timeout = {
                let state = self.state.lock().await;
                if state.shutdown_requested {
                    tracing::info!("shutdown requested, exiting accept loop");
                    break;
                }
                let elapsed = state.last_request.elapsed();
                drop(state); // Release lock early
                self.config.idle_timeout.checked_sub(elapsed)
            };

            let Some(remaining) = idle_timeout else {
                tracing::info!("idle timeout reached, shutting down");
                break;
            };

            tokio::select! {
                result = listener.accept() => {
                    match result {
                        Ok((stream, _addr)) => {
                            let state = Arc::clone(&self.state);
                            tokio::spawn(async move {
                                if let Err(e) = handle_connection(stream, state).await {
                                    tracing::warn!(error = %e, "connection error");
                                }
                            });
                        }
                        Err(e) => {
                            tracing::error!(error = %e, "accept error");
                        }
                    }
                }
                () = tokio::time::sleep(remaining) => {
                    tracing::info!("idle timeout reached, shutting down");
                    break;
                }
                _ = tokio::signal::ctrl_c() => {
                    tracing::info!("received SIGINT, shutting down");
                    break;
                }
            }
        }

        // Cleanup
        self.cleanup().await;
        Ok(())
    }

    /// Clean up socket and PID files.
    async fn cleanup(&self) {
        // Unload models
        {
            let mut state = self.state.lock().await;
            state.models.unload_all();
        }

        // Remove socket
        if let Err(e) = fs::remove_file(&self.config.socket_path).await {
            tracing::warn!(error = %e, "failed to remove socket file");
        }

        // Remove PID file
        if let Err(e) = fs::remove_file(&self.config.pid_path).await {
            tracing::warn!(error = %e, "failed to remove PID file");
        }

        tracing::info!("daemon cleanup complete");
    }
}

/// Handle a single client connection.
#[allow(clippy::cast_possible_truncation)] // Message size is validated < 10MB
async fn handle_connection(
    mut stream: UnixStream,
    state: Arc<Mutex<DaemonState>>,
) -> anyhow::Result<()> {
    // Check in-flight limit and increment counter atomically
    let in_flight = state.lock().await.in_flight.load(Ordering::Relaxed);
    if in_flight >= MAX_IN_FLIGHT as u64 {
        let response = Response::error(error_codes::OVERLOADED, "server overloaded");
        let envelope = Envelope::from_response(0, &response)?;
        let bytes = rmp_serde::to_vec(&envelope)?;
        stream.write_u32(bytes.len() as u32).await?;
        stream.write_all(&bytes).await?;
        return Ok(());
    }

    // Increment in-flight counter
    state.lock().await.in_flight.fetch_add(1, Ordering::Relaxed);

    // Read length-prefixed message
    let len = stream.read_u32().await?;
    if len > 10 * 1024 * 1024 {
        // 10MB limit
        anyhow::bail!("message too large: {len} bytes");
    }

    let mut buf = vec![0u8; len as usize];
    stream.read_exact(&mut buf).await?;

    // Decode envelope
    let envelope: Envelope = rmp_serde::from_slice(&buf)?;

    // Validate protocol version
    if envelope.version != PROTOCOL_VERSION {
        let response = Response::error(
            error_codes::VERSION_MISMATCH,
            format!(
                "protocol version {} not supported, expected {PROTOCOL_VERSION}",
                envelope.version
            ),
        );
        let resp_envelope = Envelope::from_response(envelope.id, &response)?;
        let bytes = rmp_serde::to_vec(&resp_envelope)?;
        stream.write_u32(bytes.len() as u32).await?;
        stream.write_all(&bytes).await?;

        state.lock().await.in_flight.fetch_sub(1, Ordering::Relaxed);
        return Ok(());
    }

    // Decode and handle request
    let request: Request = rmp_serde::from_slice(&envelope.payload)?;
    let response = handle_request(request, &state).await;

    // Send response
    let resp_envelope = Envelope::from_response(envelope.id, &response)?;
    let bytes = rmp_serde::to_vec(&resp_envelope)?;
    stream.write_u32(bytes.len() as u32).await?;
    stream.write_all(&bytes).await?;

    // Decrement in-flight and update stats
    {
        let mut s = state.lock().await;
        s.in_flight.fetch_sub(1, Ordering::Relaxed);
        s.requests_served.fetch_add(1, Ordering::Relaxed);
        s.touch();
    }

    Ok(())
}

/// Handle a decoded request.
#[allow(clippy::significant_drop_tightening)] // Lock must be held while accessing model references
async fn handle_request(request: Request, state: &Arc<Mutex<DaemonState>>) -> Response {
    match request {
        Request::Health => {
            let s = state.lock().await;
            Response::Health {
                uptime_secs: s.uptime_secs(),
                models_loaded: s.models.total_models(),
            }
        }

        Request::Embed { texts, model, dims } => {
            let mut s = state.lock().await;

            // Get embedder
            let embedder = match s.models.get_embedder(&model) {
                Ok(e) => e,
                Err(e) => {
                    return Response::error(error_codes::MODEL_LOAD_FAILED, e.to_string());
                }
            };

            // Convert texts to &str references
            let text_refs: Vec<&str> = texts.iter().map(String::as_str).collect();

            // Embed
            let embeddings = match embedder.embed_batch(&text_refs) {
                Ok(e) => e,
                Err(e) => {
                    return Response::error(error_codes::EMBEDDING_FAILED, e.to_string());
                }
            };

            // Apply MRL truncation if requested
            let vectors = if let Some(target_dim) = dims {
                if embedder.supports_mrl() && target_dim < embedder.dimension() {
                    match embedder.truncate_batch(&embeddings, target_dim) {
                        Ok(t) => t,
                        Err(e) => {
                            return Response::error(error_codes::EMBEDDING_FAILED, e.to_string());
                        }
                    }
                } else {
                    embeddings
                }
            } else {
                embeddings
            };

            Response::Embeddings { vectors }
        }

        Request::Rerank {
            query,
            documents,
            model,
        } => {
            let mut s = state.lock().await;

            // Get reranker
            let reranker = match s.models.get_reranker(&model) {
                Ok(Some(r)) => r,
                Ok(None) => {
                    return Response::error(error_codes::UNKNOWN_MODEL, "reranker 'none' selected");
                }
                Err(e) => {
                    return Response::error(error_codes::MODEL_LOAD_FAILED, e.to_string());
                }
            };

            // Convert documents to &str references
            let doc_refs: Vec<&str> = documents.iter().map(String::as_str).collect();

            // Rerank
            let scores = match reranker.rerank(&query, &doc_refs) {
                Ok(s) => s,
                Err(e) => {
                    return Response::error(error_codes::RERANK_FAILED, e.to_string());
                }
            };

            Response::Scores { scores }
        }

        Request::Status => {
            let s = state.lock().await;

            // Get RSS (resident set size)
            let rss_mb = get_rss_mb();

            Response::Status {
                uptime_secs: s.uptime_secs(),
                models: s.models.loaded_models(),
                rss_mb,
                requests_served: s.requests_served.load(Ordering::Relaxed),
                // Safety: in_flight is bounded by MAX_IN_FLIGHT (64), fits in usize
                #[allow(clippy::cast_possible_truncation)]
                in_flight: s.in_flight.load(Ordering::Relaxed) as usize,
                queue_len: 0, // No queue in current implementation
            }
        }

        Request::Shutdown => {
            state.lock().await.shutdown_requested = true;
            Response::Shutdown { ok: true }
        }
    }
}

/// Get current process RSS in MB.
fn get_rss_mb() -> f64 {
    #[cfg(target_os = "linux")]
    {
        if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
            for line in status.lines() {
                if let Some(kb) = line.strip_prefix("VmRSS:") {
                    let kb = kb.trim().trim_end_matches(" kB").trim();
                    if let Ok(kb) = kb.parse::<f64>() {
                        return kb / 1024.0;
                    }
                }
            }
        }
        0.0
    }

    #[cfg(target_os = "macos")]
    {
        // macOS: use mach APIs via libc
        0.0 // Placeholder - would need mach APIs
    }

    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = DaemonConfig::default();
        assert!(config.socket_path.to_string_lossy().contains("xf-daemon"));
        assert!(config.pid_path.to_string_lossy().contains("xf-daemon"));
        assert_eq!(config.idle_timeout, Duration::from_secs(30 * 60));
        assert_eq!(config.max_models, DEFAULT_MAX_MODELS);
    }

    #[test]
    fn test_config_builder() {
        let config = DaemonConfig::default()
            .with_idle_timeout(Duration::from_secs(60))
            .with_max_models(8);

        assert_eq!(config.idle_timeout, Duration::from_secs(60));
        assert_eq!(config.max_models, 8);
    }

    #[test]
    fn test_daemon_state_new() {
        let state = DaemonState::new(4);
        assert_eq!(state.models.total_models(), 0);
        assert!(!state.shutdown_requested);
        assert_eq!(state.requests_served.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_daemon_state_touch() {
        let mut state = DaemonState::new(4);
        let before = state.last_request;
        std::thread::sleep(std::time::Duration::from_millis(10));
        state.touch();
        assert!(state.last_request > before);
    }

    #[tokio::test]
    async fn test_handle_health_request() {
        let state = Arc::new(Mutex::new(DaemonState::new(4)));
        let response = handle_request(Request::Health, &state).await;

        match response {
            Response::Health {
                uptime_secs,
                models_loaded,
            } => {
                assert!(uptime_secs < 5); // Should be very quick
                assert_eq!(models_loaded, 0);
            }
            _ => panic!("expected Health response"),
        }
    }

    #[tokio::test]
    async fn test_handle_embed_request() {
        let state = Arc::new(Mutex::new(DaemonState::new(4)));

        let response = handle_request(
            Request::Embed {
                texts: vec!["hello world".to_string()],
                model: "hash".to_string(),
                dims: None,
            },
            &state,
        )
        .await;

        match response {
            Response::Embeddings { vectors } => {
                assert_eq!(vectors.len(), 1);
                assert!(!vectors[0].is_empty());
            }
            Response::Error { message, .. } => {
                panic!("unexpected error: {message}");
            }
            _ => panic!("expected Embeddings response"),
        }
    }

    #[tokio::test]
    async fn test_handle_status_request() {
        let state = Arc::new(Mutex::new(DaemonState::new(4)));

        // Load a model first
        {
            let mut s = state.lock().await;
            s.models.get_embedder("hash").unwrap();
        }

        let response = handle_request(Request::Status, &state).await;

        match response {
            Response::Status {
                uptime_secs,
                models,
                in_flight,
                queue_len,
                ..
            } => {
                assert!(uptime_secs < 5);
                assert_eq!(models.len(), 1);
                assert_eq!(in_flight, 0);
                assert_eq!(queue_len, 0);
            }
            _ => panic!("expected Status response"),
        }
    }

    #[tokio::test]
    async fn test_handle_shutdown_request() {
        let state = Arc::new(Mutex::new(DaemonState::new(4)));

        let response = handle_request(Request::Shutdown, &state).await;

        match response {
            Response::Shutdown { ok } => {
                assert!(ok);
            }
            _ => panic!("expected Shutdown response"),
        }

        // Verify shutdown was requested
        let s = state.lock().await;
        assert!(s.shutdown_requested);
        drop(s);
    }

    #[tokio::test]
    async fn test_handle_unknown_model() {
        let state = Arc::new(Mutex::new(DaemonState::new(4)));

        let response = handle_request(
            Request::Embed {
                texts: vec!["hello".to_string()],
                model: "unknown-xyz".to_string(),
                dims: None,
            },
            &state,
        )
        .await;

        match response {
            Response::Error { code, .. } => {
                assert_eq!(code, error_codes::MODEL_LOAD_FAILED);
            }
            _ => panic!("expected Error response"),
        }
    }

    #[test]
    fn test_get_rss_mb() {
        let rss = get_rss_mb();
        // Should return something >= 0
        assert!(rss >= 0.0);
    }
}
