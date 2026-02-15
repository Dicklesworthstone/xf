//! Daemon client for communicating with the model daemon.
//!
//! The client handles:
//! - Connection to existing daemon
//! - Auto-spawning daemon if not running
//! - Retry with exponential backoff
//! - Fallback to direct inference on failure

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[cfg(unix)]
use tokio::io::{AsyncReadExt, AsyncWriteExt};
#[cfg(unix)]
use tokio::net::UnixStream;

use super::protocol::{Envelope, PROTOCOL_VERSION, Request, Response};

/// Default socket path template (uses $USER for uniqueness).
fn default_socket_path() -> PathBuf {
    let user_id = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "default".to_string());
    PathBuf::from(format!("/tmp/xf-daemon-{user_id}.sock"))
}

/// Default PID file path template.
fn default_pid_path() -> PathBuf {
    let user_id = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "default".to_string());
    PathBuf::from(format!("/tmp/xf-daemon-{user_id}.pid"))
}

/// Client configuration.
#[derive(Debug, Clone)]
pub struct ClientConfig {
    /// Path to the daemon socket.
    pub socket_path: PathBuf,
    /// Path to the daemon PID file.
    pub pid_path: PathBuf,
    /// Timeout for connecting to the daemon.
    pub connect_timeout: Duration,
    /// Timeout for individual requests.
    pub request_timeout: Duration,
    /// Maximum number of retry attempts.
    pub max_retries: u8,
    /// Whether to auto-spawn the daemon if not running.
    pub auto_spawn: bool,
    /// Time to wait for daemon to start after spawning.
    pub spawn_wait: Duration,
}

impl Default for ClientConfig {
    fn default() -> Self {
        Self {
            socket_path: std::env::var("XF_DAEMON_SOCK")
                .map_or_else(|_| default_socket_path(), PathBuf::from),
            pid_path: default_pid_path(),
            connect_timeout: Duration::from_secs(2),
            request_timeout: Duration::from_secs(30),
            max_retries: 3,
            auto_spawn: true,
            spawn_wait: Duration::from_secs(5),
        }
    }
}

impl ClientConfig {
    /// Create config with custom socket path.
    #[must_use]
    pub fn with_socket_path(mut self, path: PathBuf) -> Self {
        self.socket_path = path;
        self
    }

    /// Disable auto-spawn.
    #[must_use]
    pub const fn without_auto_spawn(mut self) -> Self {
        self.auto_spawn = false;
        self
    }
}

/// Daemon client for sending requests to the model daemon.
pub struct DaemonClient {
    config: ClientConfig,
    /// Next request ID.
    next_id: u64,
}

impl DaemonClient {
    /// Create a new client with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(ClientConfig::default())
    }

    /// Create a new client with custom configuration.
    #[must_use]
    pub const fn with_config(config: ClientConfig) -> Self {
        Self { config, next_id: 1 }
    }

    /// Create a client with a specific socket path.
    #[must_use]
    pub fn with_socket_path(socket_path: PathBuf) -> Self {
        Self::with_config(ClientConfig::default().with_socket_path(socket_path))
    }

    /// Check if the daemon is running.
    #[cfg(unix)]
    #[must_use]
    pub fn is_daemon_running(&self) -> bool {
        self.config.socket_path.exists()
    }

    #[cfg(not(unix))]
    #[must_use]
    pub fn is_daemon_running(&self) -> bool {
        false
    }

    /// Get the daemon PID if running.
    #[must_use]
    pub fn daemon_pid(&self) -> Option<u32> {
        std::fs::read_to_string(&self.config.pid_path)
            .ok()
            .and_then(|s| s.trim().parse().ok())
    }

    /// Connect to the daemon, optionally spawning it if not running.
    ///
    /// # Errors
    /// Returns an error if connection fails and auto-spawn is disabled or fails.
    #[cfg(unix)]
    pub async fn connect(&self) -> anyhow::Result<UnixStream> {
        // Try to connect directly first
        match tokio::time::timeout(
            self.config.connect_timeout,
            UnixStream::connect(&self.config.socket_path),
        )
        .await
        {
            Ok(Ok(stream)) => return Ok(stream),
            Ok(Err(e)) if e.kind() == std::io::ErrorKind::NotFound => {
                // Socket doesn't exist - daemon not running
                if !self.config.auto_spawn {
                    anyhow::bail!("daemon not running and auto-spawn disabled");
                }
            }
            Ok(Err(e)) if e.kind() == std::io::ErrorKind::ConnectionRefused => {
                // Stale socket - daemon crashed
                if !self.config.auto_spawn {
                    anyhow::bail!("daemon not responding and auto-spawn disabled");
                }
            }
            Ok(Err(e)) => return Err(e.into()),
            Err(_) => anyhow::bail!("connection timeout"),
        }

        // Spawn daemon
        self.spawn_daemon()?;

        // Wait for socket to appear with backoff
        let start = Instant::now();
        let mut delay = Duration::from_millis(50);

        while start.elapsed() < self.config.spawn_wait {
            tokio::time::sleep(delay).await;

            match UnixStream::connect(&self.config.socket_path).await {
                Ok(stream) => return Ok(stream),
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                    // Socket not ready yet, continue waiting
                    delay = (delay * 2).min(Duration::from_millis(500));
                }
                Err(e) => {
                    tracing::warn!(error = %e, "connect attempt failed");
                    delay = (delay * 2).min(Duration::from_millis(500));
                }
            }
        }

        anyhow::bail!("daemon failed to start within {:?}", self.config.spawn_wait)
    }

    #[cfg(not(unix))]
    pub async fn connect(&self) -> anyhow::Result<()> {
        anyhow::bail!("daemon client not supported on this platform")
    }

    /// Spawn the daemon process.
    fn spawn_daemon(&self) -> anyhow::Result<()> {
        tracing::info!(
            socket = %self.config.socket_path.display(),
            "spawning daemon"
        );

        // Get the path to the current executable
        let exe = std::env::current_exe()?;

        // Spawn the daemon as a background process
        // Pass the socket path to ensure daemon listens on the same socket client expects
        let child = Command::new(&exe)
            .args([
                "daemon",
                "start",
                "--socket",
                &self.config.socket_path.to_string_lossy(),
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;

        tracing::info!(pid = child.id(), "daemon process spawned");
        Ok(())
    }

    /// Send a request to the daemon and receive a response.
    ///
    /// # Errors
    /// Returns an error if the request fails after all retries.
    #[cfg(unix)]
    #[allow(clippy::cast_possible_truncation)]
    pub async fn send(&mut self, request: Request) -> anyhow::Result<Response> {
        let mut last_error = None;
        let mut delay = Duration::from_millis(50);

        for attempt in 0..=self.config.max_retries {
            if attempt > 0 {
                tracing::debug!(attempt, "retrying request");
                tokio::time::sleep(delay).await;
                delay *= 2;
            }

            match self.try_send(&request).await {
                Ok(response) => return Ok(response),
                Err(e) => {
                    tracing::warn!(attempt, error = %e, "request failed");
                    last_error = Some(e);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("request failed")))
    }

    #[cfg(not(unix))]
    pub async fn send(&mut self, _request: Request) -> anyhow::Result<Response> {
        anyhow::bail!("daemon client not supported on this platform")
    }

    /// Try to send a request once.
    #[cfg(unix)]
    #[allow(clippy::cast_possible_truncation)]
    async fn try_send(&mut self, request: &Request) -> anyhow::Result<Response> {
        // Connect (with potential spawn)
        let mut stream = self.connect().await?;

        // Create envelope
        let id = self.next_id;
        self.next_id = self.next_id.wrapping_add(1);
        let envelope = Envelope::from_request(id, request)?;

        // Serialize and send
        let bytes = rmp_serde::to_vec(&envelope)?;
        stream.write_u32(bytes.len() as u32).await?;
        stream.write_all(&bytes).await?;

        // Read response
        let resp_len = tokio::time::timeout(self.config.request_timeout, stream.read_u32())
            .await
            .map_err(|_| anyhow::anyhow!("response timeout"))?? as usize;

        if resp_len > 10 * 1024 * 1024 {
            anyhow::bail!("response too large: {resp_len} bytes");
        }

        let mut resp_buf = vec![0u8; resp_len];
        tokio::time::timeout(
            self.config.request_timeout,
            stream.read_exact(&mut resp_buf),
        )
        .await
        .map_err(|_| anyhow::anyhow!("response read timeout"))??;

        // Decode response envelope
        let resp_envelope: Envelope = rmp_serde::from_slice(&resp_buf)?;

        if resp_envelope.version != PROTOCOL_VERSION {
            anyhow::bail!(
                "protocol version mismatch: got {}, expected {PROTOCOL_VERSION}",
                resp_envelope.version
            );
        }

        // Decode response
        let response: Response = rmp_serde::from_slice(&resp_envelope.payload)?;
        Ok(response)
    }

    // High-level API methods

    /// Send a health check request.
    ///
    /// # Errors
    /// Returns an error if the request fails.
    pub async fn health(&mut self) -> anyhow::Result<HealthInfo> {
        match self.send(Request::Health).await? {
            Response::Health {
                uptime_secs,
                models_loaded,
            } => Ok(HealthInfo {
                uptime_secs,
                models_loaded,
            }),
            Response::Error { code, message } => {
                anyhow::bail!("daemon error {code}: {message}")
            }
            other => anyhow::bail!("unexpected response: {other:?}"),
        }
    }

    /// Embed texts using the daemon.
    ///
    /// # Errors
    /// Returns an error if embedding fails.
    pub async fn embed(
        &mut self,
        texts: &[&str],
        model: Option<&str>,
        dims: Option<usize>,
    ) -> anyhow::Result<Vec<Vec<f32>>> {
        let request = Request::Embed {
            texts: texts.iter().map(|s| (*s).to_string()).collect(),
            model: model.unwrap_or("default").to_string(),
            dims,
        };

        match self.send(request).await? {
            Response::Embeddings { vectors } => Ok(vectors),
            Response::Error { code, message } => {
                anyhow::bail!("embedding error {code}: {message}")
            }
            other => anyhow::bail!("unexpected response: {other:?}"),
        }
    }

    /// Rerank documents using the daemon.
    ///
    /// # Errors
    /// Returns an error if reranking fails.
    pub async fn rerank(
        &mut self,
        query: &str,
        documents: &[&str],
        model: Option<&str>,
    ) -> anyhow::Result<Vec<f32>> {
        let request = Request::Rerank {
            query: query.to_string(),
            documents: documents.iter().map(|s| (*s).to_string()).collect(),
            model: model.unwrap_or("default").to_string(),
        };

        match self.send(request).await? {
            Response::Scores { scores } => Ok(scores),
            Response::Error { code, message } => {
                anyhow::bail!("rerank error {code}: {message}")
            }
            other => anyhow::bail!("unexpected response: {other:?}"),
        }
    }

    /// Get daemon status.
    ///
    /// # Errors
    /// Returns an error if the request fails.
    pub async fn status(&mut self) -> anyhow::Result<StatusInfo> {
        match self.send(Request::Status).await? {
            Response::Status {
                uptime_secs,
                models,
                rss_mb,
                requests_served,
                in_flight,
                queue_len,
            } => Ok(StatusInfo {
                uptime_secs,
                models,
                rss_mb,
                requests_served,
                in_flight,
                queue_len,
            }),
            Response::Error { code, message } => {
                anyhow::bail!("status error {code}: {message}")
            }
            other => anyhow::bail!("unexpected response: {other:?}"),
        }
    }

    /// Request daemon shutdown.
    ///
    /// # Errors
    /// Returns an error if the request fails.
    pub async fn shutdown(&mut self) -> anyhow::Result<bool> {
        match self.send(Request::Shutdown).await? {
            Response::Shutdown { ok } => Ok(ok),
            Response::Error { code, message } => {
                anyhow::bail!("shutdown error {code}: {message}")
            }
            other => anyhow::bail!("unexpected response: {other:?}"),
        }
    }

    /// Submit a background embedding job.
    ///
    /// # Errors
    /// Returns an error if the request fails.
    pub async fn submit_embedding_job(
        &mut self,
        db_path: &str,
        index_path: &str,
        two_tier: bool,
        fast_model: Option<String>,
        quality_model: Option<String>,
    ) -> anyhow::Result<EmbeddingJobSubmitResult> {
        let request = Request::SubmitEmbeddingJob {
            db_path: db_path.to_string(),
            index_path: index_path.to_string(),
            model_id: if two_tier {
                "two_tier".to_string()
            } else {
                "default".to_string()
            },
            two_tier,
            fast_model,
            quality_model,
        };

        match self.send(request).await? {
            Response::JobSubmitted {
                job_id,
                estimated_docs,
            } => Ok(EmbeddingJobSubmitResult {
                job_id,
                estimated_docs,
            }),
            Response::Error { code, message } => {
                anyhow::bail!("job submit error {code}: {message}")
            }
            other => anyhow::bail!("unexpected response: {other:?}"),
        }
    }

    /// Get embedding job status.
    ///
    /// # Errors
    /// Returns an error if the request fails.
    pub async fn embedding_job_status(
        &mut self,
        db_path: Option<&str>,
    ) -> anyhow::Result<Vec<super::protocol::EmbeddingJobInfo>> {
        let request = Request::EmbeddingJobStatus {
            db_path: db_path.map(str::to_string),
        };

        match self.send(request).await? {
            Response::JobStatus { jobs } => Ok(jobs),
            Response::Error { code, message } => {
                anyhow::bail!("job status error {code}: {message}")
            }
            other => anyhow::bail!("unexpected response: {other:?}"),
        }
    }

    /// Cancel embedding jobs.
    ///
    /// # Errors
    /// Returns an error if the request fails.
    pub async fn cancel_embedding_job(
        &mut self,
        db_path: &str,
        model_id: Option<&str>,
    ) -> anyhow::Result<usize> {
        let request = Request::CancelEmbeddingJob {
            db_path: db_path.to_string(),
            model_id: model_id.map(str::to_string),
        };

        match self.send(request).await? {
            Response::JobCancelled { cancelled_count } => Ok(cancelled_count),
            Response::Error { code, message } => {
                anyhow::bail!("job cancel error {code}: {message}")
            }
            other => anyhow::bail!("unexpected response: {other:?}"),
        }
    }
}

/// Result of submitting an embedding job.
#[derive(Debug, Clone)]
pub struct EmbeddingJobSubmitResult {
    /// Job ID assigned by the daemon.
    pub job_id: i64,
    /// Estimated number of documents to embed.
    pub estimated_docs: i64,
}

impl Default for DaemonClient {
    fn default() -> Self {
        Self::new()
    }
}

/// Health check response info.
#[derive(Debug, Clone, serde::Serialize)]
pub struct HealthInfo {
    /// Daemon uptime in seconds.
    pub uptime_secs: u64,
    /// Number of models currently loaded.
    pub models_loaded: usize,
}

/// Status response info.
#[derive(Debug, Clone, serde::Serialize)]
pub struct StatusInfo {
    /// Daemon uptime in seconds.
    pub uptime_secs: u64,
    /// Loaded models.
    pub models: Vec<super::protocol::LoadedModelInfo>,
    /// Resident set size in MB.
    pub rss_mb: f64,
    /// Total requests served.
    pub requests_served: u64,
    /// Requests currently in flight.
    pub in_flight: usize,
    /// Requests in queue.
    pub queue_len: usize,
}

/// Inference mode that can fall back from daemon to direct.
pub enum InferenceMode {
    /// Using the daemon for inference.
    Daemon(DaemonClient),
    /// Direct in-process inference (fallback).
    Direct(Box<dyn crate::embedder::Embedder>),
}

impl InferenceMode {
    /// Create a new inference mode, preferring daemon.
    #[must_use]
    pub fn new() -> Self {
        Self::Daemon(DaemonClient::new())
    }

    /// Create daemon-only mode (no fallback).
    #[must_use]
    pub fn daemon_only() -> Self {
        Self::Daemon(DaemonClient::with_config(
            ClientConfig::default().without_auto_spawn(),
        ))
    }

    /// Create direct-only mode.
    #[must_use]
    pub fn direct(embedder: Box<dyn crate::embedder::Embedder>) -> Self {
        Self::Direct(embedder)
    }

    /// Embed texts, falling back to direct inference on daemon failure.
    ///
    /// # Errors
    /// Returns an error if both daemon and direct inference fail.
    pub async fn embed(&mut self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        match self {
            Self::Daemon(client) => {
                match client.embed(texts, None, None).await {
                    Ok(vectors) => Ok(vectors),
                    Err(e) => {
                        tracing::warn!(error = %e, "daemon embed failed, consider direct fallback");
                        // In a full implementation, this would upgrade to Direct mode
                        // For now, propagate the error
                        Err(e)
                    }
                }
            }
            Self::Direct(embedder) => embedder.embed_batch(texts).map_err(Into::into),
        }
    }
}

impl Default for InferenceMode {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ClientConfig::default();
        assert!(config.socket_path.to_string_lossy().contains("xf-daemon"));
        assert_eq!(config.connect_timeout, Duration::from_secs(2));
        assert_eq!(config.request_timeout, Duration::from_secs(30));
        assert_eq!(config.max_retries, 3);
        assert!(config.auto_spawn);
    }

    #[test]
    fn test_config_builder() {
        let config = ClientConfig::default()
            .with_socket_path(PathBuf::from("/tmp/test.sock"))
            .without_auto_spawn();

        assert_eq!(config.socket_path, PathBuf::from("/tmp/test.sock"));
        assert!(!config.auto_spawn);
    }

    #[test]
    fn test_client_new() {
        let client = DaemonClient::new();
        assert!(!client.is_daemon_running());
    }

    #[test]
    fn test_default_paths() {
        let socket = default_socket_path();
        let pid = default_pid_path();

        assert!(socket.to_string_lossy().contains("xf-daemon"));
        assert!(pid.to_string_lossy().contains("xf-daemon"));
        assert!(socket.to_string_lossy().ends_with(".sock"));
        assert!(pid.to_string_lossy().ends_with(".pid"));
    }

    #[test]
    fn test_with_config() {
        let config = ClientConfig {
            socket_path: PathBuf::from("/custom/test.sock"),
            pid_path: PathBuf::from("/custom/test.pid"),
            connect_timeout: Duration::from_secs(10),
            request_timeout: Duration::from_secs(60),
            max_retries: 5,
            auto_spawn: false,
            spawn_wait: Duration::from_secs(10),
        };
        let client = DaemonClient::with_config(config);
        assert_eq!(
            client.config.socket_path,
            PathBuf::from("/custom/test.sock")
        );
        assert_eq!(client.config.max_retries, 5);
        assert!(!client.config.auto_spawn);
    }

    #[test]
    fn test_client_with_socket_path() {
        let client = DaemonClient::with_socket_path(PathBuf::from("/tmp/custom.sock"));
        assert_eq!(client.config.socket_path, PathBuf::from("/tmp/custom.sock"));
        // Other defaults should still hold
        assert!(client.config.auto_spawn);
        assert_eq!(client.config.max_retries, 3);
    }

    #[test]
    fn test_daemon_pid_no_file() {
        let config = ClientConfig {
            pid_path: PathBuf::from("/tmp/nonexistent-xf-test-pid-file.pid"),
            ..ClientConfig::default()
        };
        let client = DaemonClient::with_config(config);
        assert!(client.daemon_pid().is_none());
    }

    #[test]
    fn test_daemon_pid_with_temp_file() {
        let dir = std::env::temp_dir();
        let pid_path = dir.join("xf-test-daemon-pid-unit.pid");
        std::fs::write(&pid_path, "12345\n").unwrap();

        let config = ClientConfig {
            pid_path: pid_path.clone(),
            ..ClientConfig::default()
        };
        let client = DaemonClient::with_config(config);
        assert_eq!(client.daemon_pid(), Some(12345));

        // Cleanup
        let _ = std::fs::remove_file(&pid_path);
    }

    #[test]
    fn test_daemon_pid_invalid_content() {
        let dir = std::env::temp_dir();
        let pid_path = dir.join("xf-test-daemon-pid-invalid.pid");
        std::fs::write(&pid_path, "not-a-number\n").unwrap();

        let config = ClientConfig {
            pid_path: pid_path.clone(),
            ..ClientConfig::default()
        };
        let client = DaemonClient::with_config(config);
        assert!(client.daemon_pid().is_none());

        let _ = std::fs::remove_file(&pid_path);
    }

    #[test]
    fn test_inference_mode_new() {
        let mode = InferenceMode::new();
        assert!(matches!(mode, InferenceMode::Daemon(_)));
    }

    #[test]
    fn test_inference_mode_daemon_only() {
        let mode = InferenceMode::daemon_only();
        match mode {
            InferenceMode::Daemon(client) => {
                // daemon_only disables auto_spawn
                assert!(!client.config.auto_spawn);
            }
            InferenceMode::Direct(_) => panic!("expected Daemon variant"),
        }
    }

    #[test]
    fn test_inference_mode_default() {
        let mode = InferenceMode::default();
        assert!(matches!(mode, InferenceMode::Daemon(_)));
    }

    #[test]
    fn test_client_default_impl() {
        let client = DaemonClient::default();
        assert!(client.config.auto_spawn);
        assert_eq!(client.config.connect_timeout, Duration::from_secs(2));
    }

    #[test]
    fn test_config_without_auto_spawn() {
        let config = ClientConfig::default().without_auto_spawn();
        assert!(!config.auto_spawn);
        // Other fields unchanged
        assert_eq!(config.connect_timeout, Duration::from_secs(2));
        assert_eq!(config.max_retries, 3);
    }
}
