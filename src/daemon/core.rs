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
use super::protocol::{
    EmbeddingJobInfo, Envelope, PROTOCOL_VERSION, Request, Response, error_codes,
};
use super::resource::{ResourceConfig, apply_resource_settings, get_process_rss_mb};
use super::worker::{EmbeddingJobConfig, EmbeddingWorker, EmbeddingWorkerHandle};

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
    /// Resource management settings.
    pub resources: ResourceConfig,
}

impl Default for DaemonConfig {
    fn default() -> Self {
        let resources = ResourceConfig::default();

        // Get user identifier for unique socket paths (safe alternative to getuid)
        // Sanitize to prevent path traversal attacks via malicious USER env var
        let user_id = std::env::var("USER")
            .or_else(|_| std::env::var("USERNAME"))
            .unwrap_or_else(|_| "default".to_string())
            .chars()
            .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
            .take(64) // Limit length to prevent overly long paths
            .collect::<String>();
        let user_id = if user_id.is_empty() {
            "default".to_string()
        } else {
            user_id
        };

        // Use socket path from ResourceConfig if set, otherwise default
        let socket_path = resources
            .socket_path
            .clone()
            .unwrap_or_else(|| PathBuf::from(format!("/tmp/xf-daemon-{user_id}.sock")));
        let pid_path = PathBuf::from(format!("/tmp/xf-daemon-{user_id}.pid"));

        Self {
            socket_path,
            pid_path,
            idle_timeout: resources.idle_timeout,
            max_models: DEFAULT_MAX_MODELS,
            resources,
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

    /// Set resource configuration.
    #[must_use]
    pub fn with_resources(mut self, resources: ResourceConfig) -> Self {
        // Update paths if ResourceConfig specifies them
        if let Some(socket) = &resources.socket_path {
            self.socket_path.clone_from(socket);
        }
        self.idle_timeout = resources.idle_timeout;
        self.resources = resources;
        self
    }

    /// Load configuration from file and environment.
    ///
    /// # Errors
    /// Returns an error if the config file exists but cannot be parsed.
    pub fn load(config_path: Option<&PathBuf>) -> anyhow::Result<Self> {
        let resources = ResourceConfig::load(config_path)?;
        Ok(Self::default().with_resources(resources))
    }
}

/// Atomic counters accessible without holding the daemon state lock.
///
/// Extracted from `DaemonState` so that `InFlightGuard::drop()` can update
/// counters synchronously without spawning an unreliable async task.
struct DaemonCounters {
    requests_served: AtomicU64,
    in_flight: AtomicU64,
}

/// Shared daemon state.
struct DaemonState {
    models: ModelManager,
    start_time: Instant,
    last_request: Instant,
    counters: Arc<DaemonCounters>,
    shutdown_requested: bool,
    /// Handle to the background embedding worker.
    worker_handle: Option<EmbeddingWorkerHandle>,
}

impl DaemonState {
    fn new(max_models: usize) -> Self {
        let now = Instant::now();
        Self {
            models: ModelManager::new(max_models),
            start_time: now,
            last_request: now,
            counters: Arc::new(DaemonCounters {
                requests_served: AtomicU64::new(0),
                in_flight: AtomicU64::new(0),
            }),
            shutdown_requested: false,
            worker_handle: None,
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
    ///
    /// Note: The embedding worker is spawned when `run()` is called.
    #[must_use]
    pub fn new(config: DaemonConfig) -> Self {
        let state = Arc::new(Mutex::new(DaemonState::new(config.max_models)));
        Self { config, state }
    }

    /// Initialize and spawn the background embedding worker.
    async fn init_worker(&self) {
        let (worker, worker_handle) = EmbeddingWorker::new();

        // Store handle in state
        {
            let mut state = self.state.lock().await;
            state.worker_handle = Some(worker_handle);
        }

        // Spawn the worker task
        tokio::spawn(async move {
            worker.run().await;
        });

        tracing::info!("Embedding worker initialized");
    }

    /// Run the daemon, accepting connections until shutdown.
    ///
    /// # Errors
    /// Returns an error on Windows as Unix Domain Sockets are not supported.
    #[cfg(unix)]
    pub async fn run(&self) -> anyhow::Result<()> {
        // Apply resource settings (nice, ionice, memory limits, thread pools)
        apply_resource_settings(&self.config.resources)?;

        // Initialize the background embedding worker
        self.init_worker().await;

        // Clean up stale socket if it exists
        if self.config.socket_path.exists() {
            fs::remove_file(&self.config.socket_path).await?;
        }

        // Bind the Unix socket
        // Note: There's a brief TOCTOU window between bind and chmod where the socket
        // has default permissions. The risk is minimal because:
        // 1. /tmp has sticky bit preventing other users from accessing files
        // 2. The window is microseconds (bind → chmod)
        // 3. An attacker would need to predict exact timing and path
        let listener = UnixListener::bind(&self.config.socket_path)?;

        // Set socket permissions to owner-only (0600) immediately after bind
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

    /// Run the daemon, accepting connections until shutdown.
    ///
    /// # Errors
    /// Always returns an error on Windows as Unix Domain Sockets are not supported.
    #[cfg(not(unix))]
    pub async fn run(&self) -> anyhow::Result<()> {
        anyhow::bail!("daemon is not supported on Windows (requires Unix Domain Sockets)")
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

/// RAII guard for in-flight request counter.
///
/// Holds an `Arc` to shared atomic counters so that `Drop` can decrement
/// synchronously without spawning an async task (which could be lost during shutdown).
struct InFlightGuard {
    counters: Arc<DaemonCounters>,
}

impl InFlightGuard {
    async fn acquire(state: &Arc<Mutex<DaemonState>>) -> Option<Self> {
        let s = state.lock().await;
        let current = s.counters.in_flight.load(Ordering::Relaxed);
        if current >= MAX_IN_FLIGHT as u64 {
            return None;
        }
        s.counters.in_flight.fetch_add(1, Ordering::Relaxed);
        let counters = Arc::clone(&s.counters);
        drop(s);
        Some(Self { counters })
    }
}

impl Drop for InFlightGuard {
    fn drop(&mut self) {
        self.counters.in_flight.fetch_sub(1, Ordering::Relaxed);
        self.counters
            .requests_served
            .fetch_add(1, Ordering::Relaxed);
    }
}

/// Handle a single client connection.
#[cfg(unix)]
#[allow(clippy::cast_possible_truncation)] // Message size is validated < 10MB
async fn handle_connection(
    mut stream: UnixStream,
    state: Arc<Mutex<DaemonState>>,
) -> anyhow::Result<()> {
    // Acquire in-flight guard
    let Some(_guard) = InFlightGuard::acquire(&state).await else {
        let response = Response::error(error_codes::OVERLOADED, "server overloaded");
        let envelope = Envelope::from_response(0, &response)?;
        let bytes = rmp_serde::to_vec(&envelope)?;
        stream.write_u32(bytes.len() as u32).await?;
        stream.write_all(&bytes).await?;
        return Ok(());
    };

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

    Ok(())
}

/// Handle a decoded request.
#[allow(clippy::significant_drop_tightening)] // Lock must be held while accessing model references
async fn handle_request(request: Request, state: &Arc<Mutex<DaemonState>>) -> Response {
    match request {
        Request::Health => {
            let mut s = state.lock().await;
            s.touch();
            Response::Health {
                uptime_secs: s.uptime_secs(),
                models_loaded: s.models.total_models(),
            }
        }

        Request::Embed { texts, model, dims } => {
            let mut s = state.lock().await;
            s.touch();

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
            if let Some(target_dim) = dims {
                if target_dim == embedder.dimension() {
                    Response::Embeddings {
                        vectors: embeddings,
                    }
                } else if embedder.supports_mrl() {
                    match embedder.truncate_batch(&embeddings, target_dim) {
                        Ok(t) => Response::Embeddings { vectors: t },
                        Err(e) => Response::error(error_codes::EMBEDDING_FAILED, e.to_string()),
                    }
                } else {
                    Response::error(
                        error_codes::INVALID_REQUEST,
                        format!(
                            "model {} does not support MRL truncation",
                            embedder.model_name()
                        ),
                    )
                }
            } else {
                Response::Embeddings {
                    vectors: embeddings,
                }
            }
        }

        Request::Rerank {
            query,
            documents,
            model,
        } => {
            let mut s = state.lock().await;
            s.touch();

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
            let mut s = state.lock().await;
            s.touch();

            // Get RSS (resident set size)
            let rss_mb = get_process_rss_mb();

            Response::Status {
                uptime_secs: s.uptime_secs(),
                models: s.models.loaded_models(),
                rss_mb,
                requests_served: s.counters.requests_served.load(Ordering::Relaxed),
                // Safety: in_flight is bounded by MAX_IN_FLIGHT (64), fits in usize
                #[allow(clippy::cast_possible_truncation)]
                in_flight: s.counters.in_flight.load(Ordering::Relaxed) as usize,
                queue_len: 0, // No queue in current implementation
            }
        }

        Request::Shutdown => {
            let mut s = state.lock().await;
            s.touch();
            s.shutdown_requested = true;
            Response::Shutdown { ok: true }
        }

        Request::SubmitEmbeddingJob {
            db_path,
            index_path,
            model_id: _model_id,
            two_tier,
            fast_model,
            quality_model,
        } => {
            let mut s = state.lock().await;
            s.touch();

            // Validate paths
            let db_path = std::path::PathBuf::from(&db_path);
            let index_path = std::path::PathBuf::from(&index_path);

            if !db_path.exists() {
                return Response::error(
                    error_codes::DB_NOT_FOUND,
                    format!("Database not found: {}", db_path.display()),
                );
            }

            // Open storage to count documents and create job entry
            let storage = match crate::storage::Storage::open(&db_path) {
                Ok(s) => s,
                Err(e) => {
                    return Response::error(
                        error_codes::DB_NOT_FOUND,
                        format!("Failed to open database: {}", e),
                    );
                }
            };

            // Count documents
            let doc_count = match count_embeddable_docs(&storage) {
                Ok(count) => count,
                Err(e) => {
                    return Response::error(
                        error_codes::INTERNAL,
                        format!("Failed to count documents: {}", e),
                    );
                }
            };

            // Cancel any existing jobs for this database
            let db_path_str = db_path.to_string_lossy().to_string();
            if let Err(e) = storage.cancel_embedding_jobs(&db_path_str, None) {
                tracing::warn!("Failed to cancel existing jobs: {}", e);
            }

            // Create job entries
            let job_id = if two_tier {
                // Create fast job entry
                storage
                    .upsert_embedding_job(&db_path_str, "fast", doc_count)
                    .ok();
                // Create quality job entry (this is the one we return)
                match storage.upsert_embedding_job(&db_path_str, "quality", doc_count) {
                    Ok(id) => id,
                    Err(e) => {
                        return Response::error(
                            error_codes::JOB_ERROR,
                            format!("Failed to create job: {}", e),
                        );
                    }
                }
            } else {
                match storage.upsert_embedding_job(&db_path_str, "default", doc_count) {
                    Ok(id) => id,
                    Err(e) => {
                        return Response::error(
                            error_codes::JOB_ERROR,
                            format!("Failed to create job: {}", e),
                        );
                    }
                }
            };

            // Submit to worker
            if let Some(ref handle) = s.worker_handle {
                let config = EmbeddingJobConfig {
                    db_path,
                    index_path,
                    two_tier,
                    fast_model,
                    quality_model,
                };
                if let Err(e) = handle.submit(config).await {
                    return Response::error(
                        error_codes::JOB_ERROR,
                        format!("Failed to submit job: {}", e),
                    );
                }
            } else {
                return Response::error(error_codes::INTERNAL, "Embedding worker not available");
            }

            Response::JobSubmitted {
                job_id,
                estimated_docs: doc_count,
            }
        }

        Request::EmbeddingJobStatus { db_path } => {
            let mut s = state.lock().await;
            s.touch();

            let jobs = match db_path {
                Some(path) => {
                    let storage = match crate::storage::Storage::open(&path) {
                        Ok(s) => s,
                        Err(e) => {
                            return Response::error(
                                error_codes::DB_NOT_FOUND,
                                format!("Failed to open database: {}", e),
                            );
                        }
                    };
                    match storage.get_job_status(&path) {
                        Ok(jobs) => jobs,
                        Err(e) => {
                            return Response::error(
                                error_codes::INTERNAL,
                                format!("Failed to get job status: {}", e),
                            );
                        }
                    }
                }
                None => {
                    // No way to get all jobs without knowing which databases exist
                    // Return empty list
                    Vec::new()
                }
            };

            let job_infos: Vec<EmbeddingJobInfo> = jobs
                .into_iter()
                .map(|j| {
                    let progress_pct = if j.total_docs > 0 {
                        (j.completed_docs as f32 / j.total_docs as f32) * 100.0
                    } else {
                        0.0
                    };
                    EmbeddingJobInfo {
                        id: j.id,
                        db_path: j.db_path,
                        model_id: j.model_id,
                        status: j.status.as_str().to_string(),
                        total_docs: j.total_docs,
                        completed_docs: j.completed_docs,
                        progress_pct,
                        created_at: j.created_at.timestamp(),
                        started_at: j.started_at.map_or(0, |t| t.timestamp()),
                        completed_at: j.completed_at.map_or(0, |t| t.timestamp()),
                        error_message: j.error_message,
                    }
                })
                .collect();

            Response::JobStatus { jobs: job_infos }
        }

        Request::CancelEmbeddingJob { db_path, model_id } => {
            let mut s = state.lock().await;
            s.touch();

            let db_path_buf = std::path::PathBuf::from(&db_path);

            // Cancel in database
            let storage = match crate::storage::Storage::open(&db_path_buf) {
                Ok(s) => s,
                Err(e) => {
                    return Response::error(
                        error_codes::DB_NOT_FOUND,
                        format!("Failed to open database: {}", e),
                    );
                }
            };

            let cancelled = match storage.cancel_embedding_jobs(&db_path, model_id.as_deref()) {
                Ok(count) => count,
                Err(e) => {
                    return Response::error(
                        error_codes::JOB_ERROR,
                        format!("Failed to cancel jobs: {}", e),
                    );
                }
            };

            // Signal worker to cancel
            if let Some(ref handle) = s.worker_handle {
                if let Err(e) = handle.cancel(db_path_buf, model_id).await {
                    tracing::warn!("Failed to signal worker cancellation: {}", e);
                }
            }

            Response::JobCancelled {
                cancelled_count: cancelled,
            }
        }
    }
}

/// Count embeddable documents in storage.
fn count_embeddable_docs(storage: &crate::storage::Storage) -> anyhow::Result<i64> {
    let tweets = storage.get_all_tweets(None)?.len();
    let likes = storage
        .get_all_likes(None)?
        .iter()
        .filter(|l| l.full_text.as_ref().is_some_and(|t| !t.is_empty()))
        .count();
    let dms = storage
        .get_all_dms(None)?
        .iter()
        .filter(|d| !d.text.is_empty())
        .count();
    let grok = storage
        .get_all_grok_messages(None)?
        .iter()
        .filter(|m| !m.message.is_empty())
        .count();

    Ok((tweets + likes + dms + grok) as i64)
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
        assert_eq!(state.counters.requests_served.load(Ordering::Relaxed), 0);
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
        let rss = get_process_rss_mb();
        // Should return something >= 0
        assert!(rss >= 0.0);
    }

    #[test]
    fn test_with_paths() {
        let config = DaemonConfig::with_paths(
            PathBuf::from("/tmp/custom.sock"),
            PathBuf::from("/tmp/custom.pid"),
        );
        assert_eq!(config.socket_path, PathBuf::from("/tmp/custom.sock"));
        assert_eq!(config.pid_path, PathBuf::from("/tmp/custom.pid"));
        // Other defaults should still hold
        assert_eq!(config.max_models, DEFAULT_MAX_MODELS);
        assert_eq!(config.idle_timeout, Duration::from_secs(30 * 60));
    }

    #[test]
    fn test_with_resources() {
        let resources = ResourceConfig {
            socket_path: Some(PathBuf::from("/tmp/res.sock")),
            idle_timeout: Duration::from_secs(120),
            ..ResourceConfig::default()
        };
        let config = DaemonConfig::default().with_resources(resources);
        assert_eq!(config.socket_path, PathBuf::from("/tmp/res.sock"));
        assert_eq!(config.idle_timeout, Duration::from_secs(120));
    }

    #[test]
    fn test_with_resources_no_socket_override() {
        let resources = ResourceConfig {
            socket_path: None,
            idle_timeout: Duration::from_secs(60),
            ..ResourceConfig::default()
        };
        let default_socket = DaemonConfig::default().socket_path;
        let config = DaemonConfig::default().with_resources(resources);
        // Socket path should stay at default when ResourceConfig has None
        assert_eq!(config.socket_path, default_socket);
        assert_eq!(config.idle_timeout, Duration::from_secs(60));
    }

    #[test]
    fn test_uptime_secs() {
        let state = DaemonState::new(4);
        // Just created, uptime should be 0 or very close
        assert!(state.uptime_secs() < 2);
    }

    #[test]
    fn test_daemon_state_requests_served_starts_zero() {
        let state = DaemonState::new(4);
        assert_eq!(state.counters.requests_served.load(Ordering::Relaxed), 0);
        assert_eq!(state.counters.in_flight.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_model_daemon_new() {
        let config = DaemonConfig::default().with_max_models(2);
        let daemon = ModelDaemon::new(config);
        assert_eq!(daemon.config.max_models, 2);
    }

    #[tokio::test]
    async fn test_handle_rerank_none_model() {
        let state = Arc::new(Mutex::new(DaemonState::new(4)));

        let response = handle_request(
            Request::Rerank {
                query: "test query".to_string(),
                documents: vec!["doc".to_string()],
                model: "none".to_string(),
            },
            &state,
        )
        .await;

        match response {
            Response::Error { code, .. } => {
                assert_eq!(code, error_codes::UNKNOWN_MODEL);
            }
            _ => panic!("expected Error response for 'none' reranker"),
        }
    }
}
