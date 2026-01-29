//! E2E lifecycle tests for the daemon.
//!
//! These tests spawn real daemon processes and verify:
//! - Startup and health check
//! - Graceful shutdown
//! - Socket permissions
//! - PID file handling
//! - Signal handling (SIGINT, SIGTERM)
//! - Idle timeout shutdown
//! - Protocol version handling (bd-1nya)
//! - Error responses (bd-1nya)
//! - Client auto-spawn and retry logic (bd-2zll)
//! - Config file and environment variable handling (bd-35ft)
//! - LRU eviction and memory pressure (bd-3k2x)
//! - Daemon reranker error handling (bd-2cnh)
//! - MRL dimension truncation via daemon (bd-ou8b)
//!
//! # Test Infrastructure
//!
//! The `DaemonProcess` struct provides reusable test infrastructure for spawning
//! real daemon processes in isolated temporary directories.
//!
//! # Running Tests
//!
//! **Important**: These tests must run serially because they share the global
//! PID file at `/tmp/xf-daemon-$USER.pid`.
//!
//! ```bash
//! # Run all daemon E2E tests (serially)
//! cargo test --test daemon_e2e -- --test-threads=1
//!
//! # Run a specific test with logging
//! RUST_LOG=debug cargo test --test daemon_e2e test_daemon_start_and_health_check -- --nocapture
//! ```

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::time::Duration;

use serial_test::serial;
use tempfile::TempDir;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::time::sleep;
use xf::daemon::{
    ClientConfig, DaemonClient, Envelope, PROTOCOL_VERSION, Request, Response, error_codes,
};

// =============================================================================
// Test Infrastructure
// =============================================================================

/// Test daemon process wrapper with automatic cleanup.
///
/// Spawns a real `xf daemon` process and provides utilities for
/// interacting with it during tests. Automatically cleans up on drop.
struct DaemonProcess {
    /// The spawned child process.
    child: Child,
    /// Path to the daemon's Unix socket.
    socket_path: PathBuf,
    /// Path to the daemon's PID file.
    pid_path: PathBuf,
    /// Temporary directory holding test files (cleaned up on drop).
    _temp_dir: TempDir,
}

impl DaemonProcess {
    /// Spawn a daemon with default configuration.
    ///
    /// Creates unique socket and PID paths in a temporary directory.
    async fn spawn() -> anyhow::Result<Self> {
        Self::spawn_default().await
    }

    /// Spawn a daemon with custom idle timeout.
    ///
    /// # Arguments
    /// * `idle_timeout_secs` - Idle timeout in seconds (0 = no timeout)
    async fn spawn_with_timeout(idle_timeout_secs: u64) -> anyhow::Result<Self> {
        let temp_dir = TempDir::new()?;
        let socket_path = temp_dir.path().join("daemon.sock");

        // PID file uses default location based on USER env var
        let user_id = std::env::var("USER")
            .or_else(|_| std::env::var("USERNAME"))
            .unwrap_or_else(|_| "default".to_string());
        let pid_path = PathBuf::from(format!("/tmp/xf-daemon-{user_id}.pid"));

        let xf_binary = find_xf_binary()?;

        tracing::info!(
            binary = %xf_binary.display(),
            socket = %socket_path.display(),
            pid = %pid_path.display(),
            idle_timeout_secs,
            "spawning test daemon"
        );

        let mut cmd = Command::new(&xf_binary);
        cmd.arg("daemon")
            .arg("start")
            .arg("--foreground")
            .arg("--socket")
            .arg(&socket_path)
            .arg("--idle-timeout")
            .arg(idle_timeout_secs.to_string())
            .env("XF_DAEMON_SOCK", &socket_path)
            .env("RUST_LOG", "xf=debug")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let child = cmd.spawn()?;
        let child_pid = child.id();

        tracing::info!(
            pid = child_pid,
            "daemon process spawned, waiting for socket"
        );

        // Wait for socket to appear
        wait_for_socket(&socket_path, Duration::from_secs(10)).await?;

        tracing::info!(
            socket = %socket_path.display(),
            "daemon socket ready"
        );

        Ok(Self {
            child,
            socket_path,
            pid_path,
            _temp_dir: temp_dir,
        })
    }

    /// Spawn a daemon with a specific max-models limit.
    ///
    /// # Arguments
    /// * `max_models` - Maximum number of models to keep loaded
    async fn spawn_with_max_models(max_models: usize) -> anyhow::Result<Self> {
        let temp_dir = TempDir::new()?;
        let socket_path = temp_dir.path().join("daemon.sock");

        let user_id = std::env::var("USER")
            .or_else(|_| std::env::var("USERNAME"))
            .unwrap_or_else(|_| "default".to_string());
        let pid_path = PathBuf::from(format!("/tmp/xf-daemon-{user_id}.pid"));

        let xf_binary = find_xf_binary()?;

        tracing::info!(
            binary = %xf_binary.display(),
            socket = %socket_path.display(),
            max_models,
            "spawning test daemon with max_models limit"
        );

        let mut cmd = Command::new(&xf_binary);
        cmd.arg("daemon")
            .arg("start")
            .arg("--foreground")
            .arg("--socket")
            .arg(&socket_path)
            .arg("--max-models")
            .arg(max_models.to_string())
            .env("XF_DAEMON_SOCK", &socket_path)
            .env("RUST_LOG", "xf=debug")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let child = cmd.spawn()?;
        let child_pid = child.id();

        tracing::info!(
            pid = child_pid,
            "daemon process spawned, waiting for socket"
        );

        wait_for_socket(&socket_path, Duration::from_secs(10)).await?;

        tracing::info!(socket = %socket_path.display(), "daemon socket ready");

        Ok(Self {
            child,
            socket_path,
            pid_path,
            _temp_dir: temp_dir,
        })
    }

    /// Spawn a daemon with default configuration (5 minute idle timeout).
    async fn spawn_default() -> anyhow::Result<Self> {
        let temp_dir = TempDir::new()?;
        let socket_path = temp_dir.path().join("daemon.sock");

        // PID file uses default location based on USER env var
        let user_id = std::env::var("USER")
            .or_else(|_| std::env::var("USERNAME"))
            .unwrap_or_else(|_| "default".to_string());
        let pid_path = PathBuf::from(format!("/tmp/xf-daemon-{user_id}.pid"));

        let xf_binary = find_xf_binary()?;

        tracing::info!(
            binary = %xf_binary.display(),
            socket = %socket_path.display(),
            pid = %pid_path.display(),
            "spawning test daemon"
        );

        let mut cmd = Command::new(&xf_binary);
        cmd.arg("daemon")
            .arg("start")
            .arg("--foreground")
            .arg("--socket")
            .arg(&socket_path)
            .env("XF_DAEMON_SOCK", &socket_path)
            .env("RUST_LOG", "xf=debug")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let child = cmd.spawn()?;
        let child_pid = child.id();

        tracing::info!(
            pid = child_pid,
            "daemon process spawned, waiting for socket"
        );

        // Wait for socket to appear
        wait_for_socket(&socket_path, Duration::from_secs(10)).await?;

        tracing::info!(
            socket = %socket_path.display(),
            "daemon socket ready"
        );

        Ok(Self {
            child,
            socket_path,
            pid_path,
            _temp_dir: temp_dir,
        })
    }

    /// Gracefully stop the daemon by sending SIGINT.
    ///
    /// Note: The daemon only handles SIGINT for graceful shutdown, not SIGTERM.
    #[cfg(unix)]
    async fn stop(&mut self) -> anyhow::Result<()> {
        use nix::sys::signal::{Signal, kill};
        use nix::unistd::Pid;

        tracing::info!(pid = self.child.id(), "stopping daemon");

        // Graceful shutdown via SIGINT (daemon handles ctrl_c)
        #[allow(clippy::cast_possible_wrap)]
        let pid_i32 = self.child.id() as i32;
        let _ = kill(Pid::from_raw(pid_i32), Signal::SIGINT);

        // Wait for process to exit with timeout
        let start = std::time::Instant::now();
        let timeout_duration = Duration::from_secs(5);

        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => {
                    tracing::info!(?status, "daemon stopped");
                    return Ok(());
                }
                Ok(None) => {
                    if start.elapsed() > timeout_duration {
                        tracing::warn!("daemon didn't stop gracefully, killing");
                        let _ = self.child.kill();
                        let _ = self.child.wait();
                        anyhow::bail!("daemon didn't stop within timeout");
                    }
                    sleep(Duration::from_millis(50)).await;
                }
                Err(e) => {
                    tracing::warn!(error = %e, "error waiting for daemon");
                    return Err(e.into());
                }
            }
        }
    }

    #[cfg(not(unix))]
    async fn stop(&mut self) -> anyhow::Result<()> {
        let _ = self.child.kill();
        let _ = self.child.wait();
        Ok(())
    }

    /// Check if the daemon process is still running.
    fn is_running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// Get the process ID of the daemon.
    fn pid(&self) -> u32 {
        self.child.id()
    }

    /// Wait for the daemon to exit with timeout.
    async fn wait_with_timeout(
        &mut self,
        timeout_duration: Duration,
    ) -> anyhow::Result<std::process::ExitStatus> {
        let start = std::time::Instant::now();

        loop {
            match self.child.try_wait() {
                Ok(Some(status)) => return Ok(status),
                Ok(None) => {
                    if start.elapsed() > timeout_duration {
                        anyhow::bail!("daemon didn't exit within timeout");
                    }
                    sleep(Duration::from_millis(50)).await;
                }
                Err(e) => return Err(e.into()),
            }
        }
    }
}

impl Drop for DaemonProcess {
    fn drop(&mut self) {
        // Best-effort cleanup: try to kill the process if still running
        if self.is_running() {
            tracing::warn!(
                pid = self.child.id(),
                "daemon still running in drop, killing"
            );
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

/// Wait for a socket file to appear with timeout.
async fn wait_for_socket(path: &std::path::Path, timeout_duration: Duration) -> anyhow::Result<()> {
    let start = std::time::Instant::now();

    while start.elapsed() < timeout_duration {
        if path.exists() {
            return Ok(());
        }
        sleep(Duration::from_millis(50)).await;
    }

    anyhow::bail!(
        "socket {} did not appear within {:?}",
        path.display(),
        timeout_duration
    )
}

/// Find the xf binary, checking both debug and release builds.
fn find_xf_binary() -> anyhow::Result<PathBuf> {
    let target_dir =
        std::env::var("CARGO_TARGET_DIR").map_or_else(|_| PathBuf::from("target"), PathBuf::from);

    // Prefer release build if it exists
    let release_path = target_dir.join("release/xf");
    let debug_path = target_dir.join("debug/xf");

    if release_path.exists() {
        return Ok(release_path);
    }

    if debug_path.exists() {
        return Ok(debug_path);
    }

    // Try PATH as fallback
    if let Ok(path) = which::which("xf") {
        return Ok(path);
    }

    anyhow::bail!(
        "xf binary not found. Run `cargo build` or `cargo build --release` first.\n\
         Searched: {}, {}, PATH",
        release_path.display(),
        debug_path.display()
    )
}

// =============================================================================
// Lifecycle Tests
// =============================================================================

/// Test that daemon starts and responds to health checks.
#[tokio::test]
#[serial]
async fn test_daemon_start_and_health_check() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());
    let health = client.health().await.expect("health check");

    assert!(
        health.uptime_secs < 10,
        "Daemon uptime should be less than 10s, got {}s",
        health.uptime_secs
    );

    tracing::info!(
        uptime = health.uptime_secs,
        models = health.models_loaded,
        "Health check passed"
    );
}

/// Test that daemon stops gracefully and cleans up socket file.
#[tokio::test]
#[serial]
async fn test_daemon_graceful_stop() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let mut daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let socket_path = daemon.socket_path.clone();

    assert!(socket_path.exists(), "Socket should exist before stop");

    daemon.stop().await.expect("graceful stop");

    tokio::time::sleep(Duration::from_millis(200)).await;

    assert!(
        !socket_path.exists(),
        "Socket should be removed after stop: {}",
        socket_path.display()
    );

    // Note: PID file at /tmp/xf-daemon-$USER.pid is shared globally
    // and may not be cleaned up if other tests are running

    tracing::info!("Graceful stop and cleanup verified");
}

/// Test that socket has correct permissions (0600 - owner only).
#[tokio::test]
#[serial]
#[cfg(unix)]
async fn test_socket_permissions_are_0600() {
    use std::os::unix::fs::PermissionsExt;

    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    let metadata = std::fs::metadata(&daemon.socket_path).expect("socket metadata");
    let mode = metadata.permissions().mode() & 0o777;

    assert_eq!(
        mode, 0o600,
        "Socket should be owner-only (0600), got {mode:o}"
    );

    tracing::info!(mode = format!("{:o}", mode), "Socket permissions verified");
}

/// Test that PID file is written with a valid process ID.
///
/// Note: When running tests in parallel, the global PID file may contain
/// the PID of another daemon instance. This test verifies the PID file
/// exists and contains a valid PID, but not necessarily this daemon's PID.
/// For accurate PID verification, run tests with `--test-threads=1`.
#[tokio::test]
#[serial]
async fn test_pid_file_is_written() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    // PID file should exist
    assert!(
        daemon.pid_path.exists(),
        "PID file should exist at {}",
        daemon.pid_path.display()
    );

    // PID file should contain a valid number
    let pid_content = std::fs::read_to_string(&daemon.pid_path).expect("read PID file");
    let file_pid: u32 = pid_content.trim().parse().expect("parse PID");
    assert!(file_pid > 0, "PID should be positive");

    // Note: file_pid may not match daemon.pid() if tests run in parallel
    // because all daemons share the same PID file path
    tracing::info!(
        file_pid,
        process_pid = daemon.pid(),
        "PID file verified (shared file, may differ in parallel)"
    );
}

/// Test that idle timeout triggers automatic shutdown.
#[tokio::test]
#[serial]
async fn test_idle_timeout_triggers_shutdown() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    // Spawn with very short idle timeout (2 seconds)
    let mut daemon = DaemonProcess::spawn_with_timeout(2)
        .await
        .expect("spawn daemon with timeout");

    let socket_path = daemon.socket_path.clone();

    tracing::info!("Waiting for idle timeout (2s + buffer)...");

    // Wait longer than the timeout
    tokio::time::sleep(Duration::from_secs(4)).await;

    // Daemon should have exited
    let status = daemon.child.try_wait().expect("check daemon status");

    assert!(
        status.is_some(),
        "Daemon should have exited after idle timeout"
    );

    // Socket should be cleaned up
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        !socket_path.exists(),
        "Socket should be removed after idle timeout"
    );

    tracing::info!(?status, "Idle timeout shutdown verified");
}

/// Test that SIGINT causes clean shutdown.
#[tokio::test]
#[serial]
#[cfg(unix)]
async fn test_sigint_clean_shutdown() {
    use nix::sys::signal::{Signal, kill};
    use nix::unistd::Pid;

    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let mut daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let socket_path = daemon.socket_path.clone();
    let pid = daemon.pid();

    tracing::info!(pid, "Sending SIGINT");

    #[allow(clippy::cast_possible_wrap)]
    let pid_i32 = pid as i32;
    kill(Pid::from_raw(pid_i32), Signal::SIGINT).expect("send SIGINT");

    let status = daemon
        .wait_with_timeout(Duration::from_secs(5))
        .await
        .expect("wait for daemon");

    assert!(status.success(), "Daemon should exit cleanly on SIGINT");

    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(
        !socket_path.exists(),
        "Socket should be cleaned up on SIGINT"
    );

    tracing::info!(?status, "SIGINT handling verified");
}

/// Test that SIGTERM terminates the daemon.
///
/// Note: Currently the daemon does not handle SIGTERM for graceful shutdown,
/// it only handles SIGINT. SIGTERM will kill the process without cleanup.
#[tokio::test]
#[serial]
#[cfg(unix)]
async fn test_sigterm_terminates_daemon() {
    use nix::sys::signal::{Signal, kill};
    use nix::unistd::Pid;

    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let mut daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let pid = daemon.pid();

    tracing::info!(pid, "Sending SIGTERM");

    #[allow(clippy::cast_possible_wrap)]
    let pid_i32 = pid as i32;
    kill(Pid::from_raw(pid_i32), Signal::SIGTERM).expect("send SIGTERM");

    let status = daemon
        .wait_with_timeout(Duration::from_secs(5))
        .await
        .expect("wait for daemon");

    // SIGTERM causes immediate termination (exit code 15 = signal 15)
    // The daemon doesn't get a chance to clean up
    assert!(!status.success(), "SIGTERM should cause non-zero exit");

    tracing::info!(?status, "SIGTERM termination verified");
}

/// Test that daemon status endpoint provides expected info.
#[tokio::test]
#[serial]
async fn test_daemon_status_endpoint() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());
    let status = client.status().await.expect("status request");

    assert!(
        status.uptime_secs < 10,
        "Uptime should be reasonable: {}s",
        status.uptime_secs
    );
    assert!(status.rss_mb > 0.0, "RSS should be positive");
    // Note: in_flight may be non-zero during parallel test runs

    tracing::info!(
        uptime = status.uptime_secs,
        rss_mb = status.rss_mb,
        models = status.models.len(),
        requests = status.requests_served,
        in_flight = status.in_flight,
        "Status endpoint verified"
    );
}

/// Test that multiple clients can connect sequentially.
#[tokio::test]
#[serial]
async fn test_multiple_sequential_clients() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    for i in 0..5 {
        let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());
        let health = client.health().await.expect("health check");
        tracing::info!(
            iteration = i,
            uptime = health.uptime_secs,
            "Client {} connected",
            i
        );
    }

    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());
    let status = client.status().await.expect("final status");

    assert!(
        status.requests_served >= 5,
        "Should have served at least 5 requests, got {}",
        status.requests_served
    );

    tracing::info!(
        requests = status.requests_served,
        "Multiple sequential clients verified"
    );
}

/// Test that daemon handles rapid reconnections.
#[tokio::test]
#[serial]
async fn test_rapid_reconnections() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    let mut handles = vec![];
    for i in 0..10 {
        let socket = daemon.socket_path.clone();
        handles.push(tokio::spawn(async move {
            let mut client = DaemonClient::with_socket_path(socket);
            client.health().await.map(|h| (i, h))
        }));
    }

    let results: Vec<_> = futures::future::join_all(handles).await;
    let successes = results
        .iter()
        .filter(|r| r.as_ref().ok().and_then(|r| r.as_ref().ok()).is_some())
        .count();

    assert!(
        successes >= 8,
        "At least 80% of rapid connections should succeed, got {successes}/10"
    );

    tracing::info!(successes, "Rapid reconnections test passed");
}

/// Test daemon restarts cleanly after normal shutdown.
#[tokio::test]
#[serial]
async fn test_daemon_restart_after_shutdown() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    // First daemon
    let mut daemon1 = DaemonProcess::spawn().await.expect("spawn first daemon");
    let mut client1 = DaemonClient::with_socket_path(daemon1.socket_path.clone());
    client1.health().await.expect("first health check");

    daemon1.stop().await.expect("stop first daemon");

    // Second daemon
    let daemon2 = DaemonProcess::spawn().await.expect("spawn second daemon");
    let mut client2 = DaemonClient::with_socket_path(daemon2.socket_path.clone());
    let health2 = client2.health().await.expect("second health check");

    assert!(
        health2.uptime_secs < 5,
        "Second daemon should be fresh, uptime: {}",
        health2.uptime_secs
    );

    tracing::info!("Daemon restart test passed");
}

// =============================================================================
// Lock File and PID File Tests (bd-376e)
// =============================================================================

/// Test that PID file contains the correct process ID.
///
/// This is a more rigorous version of test_pid_file_is_written that verifies
/// the PID in the file matches the actual daemon process ID when tests run
/// serially.
#[tokio::test]
#[serial]
async fn test_pid_file_contains_correct_process_id() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    // PID file should exist
    assert!(
        daemon.pid_path.exists(),
        "PID file should exist at {}",
        daemon.pid_path.display()
    );

    // Read and parse the PID
    let pid_content = std::fs::read_to_string(&daemon.pid_path).expect("read PID file");
    let file_pid: u32 = pid_content.trim().parse().expect("parse PID");

    // In serial mode, the PID should match our daemon
    let process_pid = daemon.pid();
    assert_eq!(
        file_pid, process_pid,
        "PID file should contain daemon's PID: file={file_pid}, process={process_pid}"
    );

    tracing::info!(
        file_pid,
        process_pid,
        "PID file contains correct process ID"
    );
}

/// Test that starting a second daemon on the same socket path is handled.
///
/// The current implementation removes stale sockets on startup, which means
/// the second daemon will take over. This test documents this behavior.
/// A proper lock file implementation would prevent the second daemon from starting.
#[tokio::test]
#[serial]
async fn test_second_daemon_on_same_socket_takes_over() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    // Create a shared temp directory for both daemons
    let temp_dir = tempfile::TempDir::new().expect("create temp dir");
    let socket_path = temp_dir.path().join("shared.sock");

    // Spawn first daemon
    let xf_binary = find_xf_binary().expect("find xf binary");
    let mut cmd1 = Command::new(&xf_binary);
    cmd1.arg("daemon")
        .arg("start")
        .arg("--foreground")
        .arg("--socket")
        .arg(&socket_path)
        .env("XF_DAEMON_SOCK", &socket_path)
        .env("RUST_LOG", "xf=debug")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child1 = cmd1.spawn().expect("spawn first daemon");
    wait_for_socket(&socket_path, Duration::from_secs(10))
        .await
        .expect("first socket");

    let pid1 = child1.id();
    tracing::info!(pid1, "First daemon started");

    // Try to spawn second daemon on same socket
    // This should either fail or take over (current behavior: takes over)
    let mut cmd2 = Command::new(&xf_binary);
    cmd2.arg("daemon")
        .arg("start")
        .arg("--foreground")
        .arg("--socket")
        .arg(&socket_path)
        .env("XF_DAEMON_SOCK", &socket_path)
        .env("RUST_LOG", "xf=debug")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let mut child2 = cmd2.spawn().expect("spawn second daemon");
    let pid2 = child2.id();

    // Wait for the second daemon to potentially take over the socket
    // The socket file gets removed and recreated, so wait for it to reappear
    sleep(Duration::from_millis(500)).await;

    // The second daemon removes the socket, breaking the first daemon's listener.
    // This documents the current "last writer wins" behavior.
    // A proper implementation would use advisory locks to prevent this.
    tracing::info!(
        pid1,
        pid2,
        "Second daemon spawned (may have taken over socket)"
    );

    // Verify the socket is still operational (second daemon took over)
    if socket_path.exists() {
        let mut client = DaemonClient::with_socket_path(socket_path.clone());
        match client.health().await {
            Ok(health) => {
                tracing::info!(
                    uptime = health.uptime_secs,
                    "Socket operational after takeover"
                );
            }
            Err(e) => {
                tracing::warn!(error = %e, "Socket not responding after takeover");
            }
        }
    }

    // Clean up both processes
    let _ = child1.kill();
    let _ = child1.wait();
    let _ = child2.kill();
    let _ = child2.wait();

    tracing::info!("Socket takeover behavior documented");
}

/// Test that a stale PID file (pointing to non-existent process) doesn't prevent startup.
#[tokio::test]
#[serial]
#[cfg(unix)]
async fn test_stale_pid_file_does_not_block_startup() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    // Get the default PID path
    let user_id = std::env::var("USER")
        .or_else(|_| std::env::var("USERNAME"))
        .unwrap_or_else(|_| "default".to_string());
    let pid_path = PathBuf::from(format!("/tmp/xf-daemon-{user_id}.pid"));

    // Create a stale PID file with a non-existent PID
    // Using 4194304 (max PID + 1 on most Linux systems) ensures it doesn't exist
    let stale_pid = 4_194_304u32;
    std::fs::write(&pid_path, stale_pid.to_string()).expect("write stale PID");
    tracing::info!(stale_pid, "Created stale PID file");

    // Daemon should start despite stale PID file
    // (PID file is overwritten, not checked)
    let daemon = DaemonProcess::spawn()
        .await
        .expect("spawn daemon with stale PID");

    // Verify daemon is running and PID file was updated
    let new_pid_content = std::fs::read_to_string(&pid_path).expect("read new PID");
    let new_pid: u32 = new_pid_content.trim().parse().expect("parse new PID");

    assert_ne!(
        new_pid, stale_pid,
        "PID file should be updated from stale PID"
    );
    assert_eq!(
        new_pid,
        daemon.pid(),
        "PID file should contain new daemon's PID"
    );

    tracing::info!(
        stale_pid,
        new_pid,
        "Stale PID file was overwritten successfully"
    );
}

/// Test that PID file is removed after clean shutdown (SIGINT).
#[tokio::test]
#[serial]
#[cfg(unix)]
async fn test_pid_file_removed_on_clean_shutdown() {
    use nix::sys::signal::{Signal, kill};
    use nix::unistd::Pid;

    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let mut daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let pid_path = daemon.pid_path.clone();

    // Verify PID file exists
    assert!(pid_path.exists(), "PID file should exist before shutdown");

    // Send SIGINT for graceful shutdown
    #[allow(clippy::cast_possible_wrap)]
    let pid_i32 = daemon.pid() as i32;
    kill(Pid::from_raw(pid_i32), Signal::SIGINT).expect("send SIGINT");

    // Wait for daemon to exit
    daemon
        .wait_with_timeout(Duration::from_secs(5))
        .await
        .expect("wait for shutdown");

    // Give filesystem a moment to sync
    sleep(Duration::from_millis(100)).await;

    // PID file should be removed
    assert!(
        !pid_path.exists(),
        "PID file should be removed after clean shutdown: {}",
        pid_path.display()
    );

    tracing::info!("PID file cleanup on clean shutdown verified");
}

/// Test that PID file remains after crash (SIGKILL).
///
/// When a daemon crashes or is killed forcefully, it doesn't get a chance
/// to clean up its PID file. This test verifies the PID file persists,
/// which is important for stale detection logic.
#[tokio::test]
#[serial]
#[cfg(unix)]
async fn test_pid_file_persists_after_crash() {
    use nix::sys::signal::{Signal, kill};
    use nix::unistd::Pid;

    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let mut daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let pid_path = daemon.pid_path.clone();
    let original_pid = daemon.pid();

    // Verify PID file exists with correct content
    assert!(pid_path.exists(), "PID file should exist");
    let pid_content = std::fs::read_to_string(&pid_path).expect("read PID");
    let file_pid: u32 = pid_content.trim().parse().expect("parse PID");
    assert_eq!(file_pid, original_pid);

    // Kill daemon forcefully (SIGKILL - no cleanup opportunity)
    #[allow(clippy::cast_possible_wrap)]
    let pid_i32 = original_pid as i32;
    kill(Pid::from_raw(pid_i32), Signal::SIGKILL).expect("send SIGKILL");

    // Wait for process to be reaped
    let _ = daemon.child.wait();
    sleep(Duration::from_millis(100)).await;

    // PID file should still exist (no cleanup was possible)
    assert!(
        pid_path.exists(),
        "PID file should persist after SIGKILL crash"
    );

    // And should still contain the original (now stale) PID
    let stale_content = std::fs::read_to_string(&pid_path).expect("read stale PID");
    let stale_pid: u32 = stale_content.trim().parse().expect("parse stale PID");
    assert_eq!(
        stale_pid, original_pid,
        "Stale PID file should contain crashed daemon's PID"
    );

    tracing::info!(
        original_pid,
        "PID file persistence after crash verified (stale PID: {})",
        stale_pid
    );

    // Clean up the stale PID file
    let _ = std::fs::remove_file(&pid_path);
}

/// Test that socket file is removed on clean shutdown.
#[tokio::test]
#[serial]
async fn test_socket_removed_on_clean_shutdown() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let mut daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let socket_path = daemon.socket_path.clone();

    assert!(socket_path.exists(), "Socket should exist before shutdown");

    daemon.stop().await.expect("graceful stop");
    sleep(Duration::from_millis(200)).await;

    assert!(
        !socket_path.exists(),
        "Socket should be removed after clean shutdown"
    );

    tracing::info!("Socket cleanup on clean shutdown verified");
}

/// Test that socket file persists after crash (for stale detection).
#[tokio::test]
#[serial]
#[cfg(unix)]
async fn test_socket_persists_after_crash() {
    use nix::sys::signal::{Signal, kill};
    use nix::unistd::Pid;

    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let mut daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let socket_path = daemon.socket_path.clone();

    assert!(socket_path.exists(), "Socket should exist");

    // Kill forcefully
    #[allow(clippy::cast_possible_wrap)]
    let pid_i32 = daemon.pid() as i32;
    kill(Pid::from_raw(pid_i32), Signal::SIGKILL).expect("send SIGKILL");

    let _ = daemon.child.wait();
    sleep(Duration::from_millis(100)).await;

    // Socket file may or may not persist depending on kernel behavior
    // This is system-dependent - document the actual behavior
    let socket_exists = socket_path.exists();
    tracing::info!(
        socket_exists,
        "Socket persistence after SIGKILL: {}",
        if socket_exists {
            "persists (stale socket)"
        } else {
            "removed by kernel"
        }
    );

    // Clean up if it exists
    let _ = std::fs::remove_file(&socket_path);
}

// =============================================================================
// Protocol Version and Error Handling Tests (bd-1nya)
// =============================================================================

/// Test that protocol version mismatch returns appropriate error.
#[tokio::test]
#[serial]
#[cfg(unix)]
#[allow(clippy::cast_possible_truncation)] // Test message sizes are always small
async fn test_protocol_version_mismatch_rejected() {
    use tokio::net::UnixStream;

    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    // Connect directly to socket
    let mut stream = UnixStream::connect(&daemon.socket_path)
        .await
        .expect("connect to socket");

    // Create an envelope with wrong protocol version
    let bad_envelope = Envelope {
        version: 99, // Wrong version (current is 1)
        id: 1,
        payload: rmp_serde::to_vec(&Request::Health).expect("serialize request"),
    };

    let bytes = rmp_serde::to_vec(&bad_envelope).expect("serialize envelope");

    // Write length-prefixed message
    stream
        .write_u32(bytes.len() as u32)
        .await
        .expect("write length");
    stream.write_all(&bytes).await.expect("write payload");

    // Read response
    let len = stream.read_u32().await.expect("read response length");
    let mut buf = vec![0u8; len as usize];
    stream.read_exact(&mut buf).await.expect("read response");

    let response_envelope: Envelope = rmp_serde::from_slice(&buf).expect("decode envelope");
    let response: Response =
        rmp_serde::from_slice(&response_envelope.payload).expect("decode response");

    match response {
        Response::Error { code, message } => {
            assert_eq!(
                code,
                error_codes::VERSION_MISMATCH,
                "Expected VERSION_MISMATCH error code"
            );
            assert!(
                message.contains("version"),
                "Error message should mention version: {message}"
            );
            tracing::info!(
                code,
                message = message.as_str(),
                "Version mismatch error received"
            );
        }
        _ => panic!("Expected Error response, got {response:?}"),
    }
}

/// Test that requests for unknown models return appropriate error.
#[tokio::test]
#[serial]
async fn test_unknown_model_returns_error() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    // Try to embed with non-existent model
    let result = client
        .embed(&["test text"], Some("nonexistent-model-xyz-12345"), None)
        .await;

    assert!(result.is_err(), "Unknown model should return error");
    let err = result.unwrap_err();
    tracing::info!(error = %err, "Unknown model error received");

    // Error should mention model load failure
    let err_str = err.to_string();
    assert!(
        err_str.contains("load") || err_str.contains("MODEL") || err_str.contains("not found"),
        "Error should indicate model issue: {err_str}"
    );
}

/// Test that requests for unknown reranker return appropriate error.
#[tokio::test]
#[serial]
async fn test_unknown_reranker_returns_error() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    // Try to rerank with non-existent model
    let result = client
        .rerank("test query", &["doc1", "doc2"], Some("fake-reranker-model"))
        .await;

    assert!(result.is_err(), "Unknown reranker should return error");
    let err = result.unwrap_err();
    tracing::info!(error = %err, "Unknown reranker error received");
}

/// Test that oversized requests are rejected.
#[tokio::test]
#[serial]
#[cfg(unix)]
async fn test_oversized_request_rejected() {
    use tokio::net::UnixStream;

    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    // Connect directly to socket
    let mut stream = UnixStream::connect(&daemon.socket_path)
        .await
        .expect("connect to socket");

    // Claim we're sending 20MB (exceeds 10MB limit)
    let oversized_len: u32 = 20 * 1024 * 1024;
    stream
        .write_u32(oversized_len)
        .await
        .expect("write oversized length");

    // Try to read response - connection should be closed or error
    let mut buf = [0u8; 1];
    let result = stream.read(&mut buf).await;

    // Either error or 0 bytes (connection closed)
    match result {
        Ok(0) => {
            tracing::info!("Connection closed after oversized request (expected)");
        }
        Ok(n) => {
            tracing::warn!(bytes = n, "Unexpected data received");
        }
        Err(e) => {
            tracing::info!(error = %e, "Connection error after oversized request (expected)");
        }
    }

    // Verify daemon is still healthy after rejecting oversized request
    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());
    let health = client.health().await;
    assert!(
        health.is_ok(),
        "Daemon should still be healthy after rejecting oversized request"
    );

    tracing::info!("Oversized request rejection verified");
}

/// Test that malformed request data is handled gracefully.
#[tokio::test]
#[serial]
#[cfg(unix)]
#[allow(clippy::cast_possible_truncation)] // Test message sizes are always small
async fn test_malformed_request_handled() {
    use tokio::net::UnixStream;

    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    // Connect directly to socket
    let mut stream = UnixStream::connect(&daemon.socket_path)
        .await
        .expect("connect to socket");

    // Send garbage data that's not valid MessagePack
    let garbage = b"this is not valid messagepack data at all!!!";
    stream
        .write_u32(garbage.len() as u32)
        .await
        .expect("write length");
    stream.write_all(garbage).await.expect("write garbage");

    // The daemon should either close the connection or return an error
    // Either way, it shouldn't crash
    sleep(Duration::from_millis(100)).await;

    // Verify daemon is still running by making a valid request with a new connection
    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());
    let health = client.health().await;

    assert!(
        health.is_ok(),
        "Daemon should still be healthy after malformed request"
    );
    tracing::info!("Daemon survived malformed request");
}

/// Test that the daemon returns correct protocol version in responses.
#[tokio::test]
#[serial]
#[cfg(unix)]
#[allow(clippy::cast_possible_truncation)] // Test message sizes are always small
async fn test_response_has_correct_protocol_version() {
    use tokio::net::UnixStream;

    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    // Connect directly to socket
    let mut stream = UnixStream::connect(&daemon.socket_path)
        .await
        .expect("connect to socket");

    // Create a valid health request
    let envelope = Envelope::from_request(42, &Request::Health).expect("create envelope");
    let bytes = rmp_serde::to_vec(&envelope).expect("serialize");

    // Send request
    stream
        .write_u32(bytes.len() as u32)
        .await
        .expect("write length");
    stream.write_all(&bytes).await.expect("write payload");

    // Read response
    let len = stream.read_u32().await.expect("read response length");
    let mut buf = vec![0u8; len as usize];
    stream.read_exact(&mut buf).await.expect("read response");

    let response_envelope: Envelope = rmp_serde::from_slice(&buf).expect("decode envelope");

    assert_eq!(
        response_envelope.version, PROTOCOL_VERSION,
        "Response should have current protocol version"
    );
    assert_eq!(
        response_envelope.id, 42,
        "Response should have matching request ID"
    );

    tracing::info!(
        version = response_envelope.version,
        id = response_envelope.id,
        "Protocol version in response verified"
    );
}

/// Test request ID correlation in responses.
#[tokio::test]
#[serial]
#[cfg(unix)]
#[allow(clippy::cast_possible_truncation)] // Test message sizes are always small
async fn test_request_id_correlation() {
    use tokio::net::UnixStream;

    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    // Test with multiple request IDs
    for request_id in [1u64, 42, 12345, u64::MAX] {
        let mut stream = UnixStream::connect(&daemon.socket_path)
            .await
            .expect("connect to socket");

        let envelope =
            Envelope::from_request(request_id, &Request::Health).expect("create envelope");
        let bytes = rmp_serde::to_vec(&envelope).expect("serialize");

        stream
            .write_u32(bytes.len() as u32)
            .await
            .expect("write length");
        stream.write_all(&bytes).await.expect("write payload");

        let len = stream.read_u32().await.expect("read response length");
        let mut buf = vec![0u8; len as usize];
        stream.read_exact(&mut buf).await.expect("read response");

        let response_envelope: Envelope = rmp_serde::from_slice(&buf).expect("decode envelope");

        assert_eq!(
            response_envelope.id, request_id,
            "Response ID should match request ID"
        );

        tracing::info!(request_id, "Request ID correlation verified");
    }
}

// =============================================================================
// Client Auto-Spawn and Retry Tests (bd-2zll)
// =============================================================================

/// Test that a client with auto_spawn disabled fails immediately when no daemon is running.
#[tokio::test]
#[serial]
async fn test_no_auto_spawn_when_disabled() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let temp_dir = TempDir::new().expect("create temp dir");
    let socket_path = temp_dir.path().join("no-spawn.sock");

    let config = ClientConfig::default()
        .with_socket_path(socket_path.clone())
        .without_auto_spawn();
    let mut client = DaemonClient::with_config(config);

    // Should fail immediately without spawning
    let result = client.health().await;
    assert!(result.is_err(), "health check should fail without daemon");
    assert!(
        !socket_path.exists(),
        "socket should not exist - daemon should not have been spawned"
    );

    tracing::info!("auto_spawn disabled correctly prevents daemon spawn");
}

/// Test that connection timeout is enforced when daemon is not running.
#[tokio::test]
#[serial]
async fn test_connection_timeout_without_daemon() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let temp_dir = TempDir::new().expect("create temp dir");
    let socket_path = temp_dir.path().join("timeout-test.sock");

    let config = ClientConfig::default()
        .with_socket_path(socket_path)
        .without_auto_spawn();
    let mut client = DaemonClient::with_config(config);

    let start = std::time::Instant::now();
    let result = client.health().await;
    let elapsed = start.elapsed();

    assert!(result.is_err(), "should fail when no daemon is running");
    // Should not wait forever - connect_timeout is 2s by default
    assert!(
        elapsed < Duration::from_secs(10),
        "should not wait forever, elapsed: {elapsed:?}"
    );

    tracing::info!(
        elapsed_ms = elapsed.as_millis(),
        "connection timeout enforced correctly"
    );
}

/// Test that is_daemon_running returns false when socket doesn't exist.
#[tokio::test]
#[serial]
async fn test_is_daemon_running_false_when_no_socket() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let temp_dir = TempDir::new().expect("create temp dir");
    let socket_path = temp_dir.path().join("nonexistent.sock");

    let client = DaemonClient::with_socket_path(socket_path);
    assert!(
        !client.is_daemon_running(),
        "should report not running when socket doesn't exist"
    );

    tracing::info!("is_daemon_running correctly returns false for nonexistent socket");
}

/// Test that is_daemon_running returns true when socket exists.
#[tokio::test]
#[serial]
async fn test_is_daemon_running_true_when_socket_exists() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    let client = DaemonClient::with_socket_path(daemon.socket_path.clone());
    assert!(
        client.is_daemon_running(),
        "should report running when daemon socket exists"
    );

    tracing::info!("is_daemon_running correctly detects running daemon");
}

/// Test that daemon_pid returns None when PID file doesn't exist.
#[tokio::test]
#[serial]
async fn test_daemon_pid_returns_none_when_no_pid_file() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let temp_dir = TempDir::new().expect("create temp dir");
    let socket_path = temp_dir.path().join("no-pid.sock");

    // Create a client that points to a nonexistent PID file
    let mut config = ClientConfig::default().with_socket_path(socket_path);
    config.pid_path = temp_dir.path().join("nonexistent.pid");
    let client = DaemonClient::with_config(config);

    assert_eq!(
        client.daemon_pid(),
        None,
        "should return None when PID file doesn't exist"
    );

    tracing::info!("daemon_pid correctly returns None for missing PID file");
}

/// Test that client can make multiple sequential health checks.
#[tokio::test]
#[serial]
async fn test_multiple_sequential_health_checks() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    // Make 5 sequential health checks
    for i in 0..5 {
        let health = client
            .health()
            .await
            .unwrap_or_else(|e| panic!("health check {i} failed: {e}"));
        assert!(
            health.uptime_secs < 30,
            "uptime should be reasonable at iteration {i}"
        );
    }

    tracing::info!("5 sequential health checks succeeded");
}

/// Test that spawn_wait configuration affects auto-spawn behavior.
///
/// When auto_spawn is disabled, the client should fail quickly regardless
/// of spawn_wait setting.
#[tokio::test]
#[serial]
async fn test_spawn_wait_with_no_auto_spawn() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let temp_dir = TempDir::new().expect("create temp dir");
    let socket_path = temp_dir.path().join("spawn-wait.sock");

    let mut config = ClientConfig::default()
        .with_socket_path(socket_path)
        .without_auto_spawn();
    // Set a long spawn_wait - should not matter with auto_spawn=false
    config.spawn_wait = Duration::from_secs(30);

    let mut client = DaemonClient::with_config(config);

    let start = std::time::Instant::now();
    let result = client.health().await;
    let elapsed = start.elapsed();

    assert!(result.is_err());
    // Should NOT wait 30 seconds - auto_spawn is disabled, so spawn_wait is irrelevant
    assert!(
        elapsed < Duration::from_secs(5),
        "should not honor spawn_wait when auto_spawn is disabled, elapsed: {elapsed:?}"
    );

    tracing::info!(
        elapsed_ms = elapsed.as_millis(),
        "spawn_wait correctly ignored when auto_spawn disabled"
    );
}

// =============================================================================
// Config File and Environment Variable E2E Tests (bd-35ft)
// =============================================================================

/// Test that ResourceConfig loads valid TOML correctly.
#[test]
fn test_resource_config_load_valid_toml() {
    use xf::daemon::ResourceConfig;

    let config_content = r#"
[daemon]
nice_level = 15
memory_limit_mb = 1024
max_threads = 2
idle_timeout_secs = 600
io_priority = "best_effort"
socket_path = "/tmp/test-xf.sock"
"#;

    let mut file = tempfile::NamedTempFile::new().expect("create temp file");
    std::io::Write::write_all(&mut file, config_content.as_bytes()).expect("write config");
    let path = file.path().to_path_buf();

    let config = ResourceConfig::load(Some(&path)).expect("load config");

    assert_eq!(config.nice_level, 15);
    assert_eq!(config.memory_limit_mb, 1024);
    assert_eq!(config.max_threads, 2);
    assert_eq!(config.idle_timeout, Duration::from_secs(600));

    tracing::info!(?config, "Valid TOML config loaded correctly");
}

/// Test that ResourceConfig handles invalid TOML gracefully.
#[test]
fn test_resource_config_load_invalid_toml() {
    use xf::daemon::ResourceConfig;

    let config_content = "this is not valid toml {{{";

    let mut file = tempfile::NamedTempFile::new().expect("create temp file");
    std::io::Write::write_all(&mut file, config_content.as_bytes()).expect("write config");
    let path = file.path().to_path_buf();

    let result = ResourceConfig::load(Some(&path));
    assert!(result.is_err(), "invalid TOML should return an error");

    tracing::info!(
        error = %result.unwrap_err(),
        "Invalid TOML correctly rejected"
    );
}

/// Test that ResourceConfig uses defaults when file doesn't exist.
#[test]
fn test_resource_config_missing_file_uses_defaults() {
    use xf::daemon::ResourceConfig;

    let path = PathBuf::from("/nonexistent/path/config.toml");
    let config = ResourceConfig::load(Some(&path)).expect("should use defaults");

    // Should use defaults
    assert_eq!(config.nice_level, 10);
    assert_eq!(config.memory_limit_mb, 2048);
    assert_eq!(config.idle_timeout, Duration::from_secs(30 * 60));

    tracing::info!("Missing file correctly uses defaults");
}

/// Test that DaemonConfig::load produces valid default config.
#[test]
fn test_daemon_config_load_defaults() {
    use xf::daemon::DaemonConfig;

    let config = DaemonConfig::load(None).expect("load default config");

    assert!(
        config.max_models > 0,
        "max_models should be positive by default"
    );
    assert!(
        config.idle_timeout.as_secs() > 0,
        "idle_timeout should be positive by default"
    );

    tracing::info!(
        max_models = config.max_models,
        idle_timeout_secs = config.idle_timeout.as_secs(),
        socket = %config.socket_path.display(),
        "DaemonConfig defaults loaded"
    );
}

/// Test that DaemonConfig loads resources from a TOML config file.
#[test]
fn test_daemon_config_load_from_toml_file() {
    use xf::daemon::DaemonConfig;

    let config_content = r"
[daemon]
nice_level = 5
memory_limit_mb = 512
max_threads = 1
idle_timeout_secs = 120
";

    let mut file = tempfile::NamedTempFile::new().expect("create temp file");
    std::io::Write::write_all(&mut file, config_content.as_bytes()).expect("write config");
    let path = file.path().to_path_buf();

    let config = DaemonConfig::load(Some(&path)).expect("load config from file");

    assert_eq!(config.resources.nice_level, 5);
    assert_eq!(config.resources.memory_limit_mb, 512);
    assert_eq!(config.resources.max_threads, 1);

    tracing::info!("DaemonConfig loaded from TOML file correctly");
}

/// Test that effective_threads clamps to available CPUs.
#[test]
fn test_effective_threads_clamps_to_cpu_count() {
    use xf::daemon::ResourceConfig;

    let cpus = num_cpus::get();

    // Set max_threads way higher than CPU count
    let config_content = r"
[daemon]
max_threads = 99999
";
    let mut file = tempfile::NamedTempFile::new().expect("create temp file");
    std::io::Write::write_all(&mut file, config_content.as_bytes()).expect("write config");
    let path = file.path().to_path_buf();

    let config = ResourceConfig::load(Some(&path)).expect("load config");

    assert_eq!(
        config.effective_threads(),
        cpus,
        "effective_threads should be clamped to CPU count ({cpus})"
    );

    tracing::info!(
        max_threads = config.max_threads,
        effective = config.effective_threads(),
        cpus,
        "effective_threads correctly clamped"
    );
}

/// Test that daemon uses idle timeout from config.
#[tokio::test]
#[serial]
async fn test_daemon_respects_config_idle_timeout() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    // Spawn daemon with very short idle timeout
    let daemon = DaemonProcess::spawn_with_timeout(3)
        .await
        .expect("spawn daemon with short timeout");

    // Verify daemon is running
    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());
    let health = client.health().await.expect("initial health check");
    assert!(health.uptime_secs < 5, "daemon should be freshly started");

    tracing::info!(
        uptime = health.uptime_secs,
        "daemon started with short idle timeout"
    );

    // Note: We don't wait for the idle timeout here because that's already
    // covered by test_idle_timeout_triggers_shutdown. This test just verifies
    // the config value is accepted without error.
}

/// Test that DaemonConfig socket path override works end-to-end.
#[tokio::test]
#[serial]
async fn test_daemon_custom_socket_path() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let temp_dir = TempDir::new().expect("create temp dir");
    let custom_socket = temp_dir.path().join("custom-daemon.sock");

    let xf_binary = find_xf_binary().expect("find xf binary");

    let child = Command::new(&xf_binary)
        .arg("daemon")
        .arg("start")
        .arg("--foreground")
        .arg("--socket")
        .arg(&custom_socket)
        .arg("--idle-timeout")
        .arg("300")
        .env("XF_DAEMON_SOCK", &custom_socket)
        .env("RUST_LOG", "xf=debug")
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn daemon");

    // Wait for socket to appear
    wait_for_socket(&custom_socket, Duration::from_secs(10))
        .await
        .expect("custom socket should appear");

    // Connect to the custom socket
    let mut client = DaemonClient::with_socket_path(custom_socket.clone());
    let health = client
        .health()
        .await
        .expect("health check on custom socket");
    assert!(health.uptime_secs < 10);

    tracing::info!(
        socket = %custom_socket.display(),
        uptime = health.uptime_secs,
        "custom socket path works end-to-end"
    );

    // Cleanup: create a wrapper to stop the process
    let mut daemon = DaemonProcess {
        child,
        socket_path: custom_socket,
        pid_path: PathBuf::from("/tmp/nonexistent.pid"),
        _temp_dir: temp_dir,
    };
    let _ = daemon.stop().await;
}

// =============================================================================
// LRU Eviction and Memory Pressure Tests (bd-3k2x)
// =============================================================================

/// Test that a single model loads and appears in status.
#[tokio::test]
#[serial]
async fn test_single_model_loading_and_status() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn_with_max_models(4)
        .await
        .expect("spawn daemon");

    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    // Load hash embedder via embed request
    let result = client
        .embed(&["test sentence"], Some("hash"), None)
        .await
        .expect("embed with hash");
    assert!(!result.is_empty(), "should return embeddings");
    assert!(!result[0].is_empty(), "embedding should have dimensions");

    // Check status shows the loaded model
    let status = client.status().await.expect("status check");
    assert_eq!(status.models.len(), 1, "should have exactly 1 model loaded");
    assert_eq!(status.models[0].name, "hash");
    assert_eq!(status.models[0].model_type, "embedder");
    assert!(
        status.models[0].requests_served >= 1,
        "should have served at least 1 request"
    );

    tracing::info!(
        model = %status.models[0].name,
        requests = status.models[0].requests_served,
        "single model loading verified"
    );
}

/// Test that requests_served counter increments correctly.
#[tokio::test]
#[serial]
async fn test_model_requests_served_counter() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn_with_max_models(4)
        .await
        .expect("spawn daemon");

    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    let request_count = 5;
    for i in 0..request_count {
        client
            .embed(&[&format!("test {i}")], Some("hash"), None)
            .await
            .unwrap_or_else(|e| panic!("embed request {i} failed: {e}"));
    }

    let status = client.status().await.expect("status check");
    assert_eq!(status.models.len(), 1);
    assert!(
        status.models[0].requests_served >= request_count,
        "expected >= {request_count} requests, got {}",
        status.models[0].requests_served
    );

    tracing::info!(
        requests = status.models[0].requests_served,
        "request counter verified after {request_count} requests"
    );
}

/// Test that overall requests_served on status tracks total requests.
#[tokio::test]
#[serial]
async fn test_total_requests_served_counter() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    // Make several requests (health + embeds)
    let _ = client.health().await.expect("health check");
    for i in 0..3 {
        let _ = client
            .embed(&[&format!("req {i}")], Some("hash"), None)
            .await
            .expect("embed");
    }

    let status = client.status().await.expect("status check");
    // At least 4 requests: 1 health + 3 embeds (status itself counts too)
    assert!(
        status.requests_served >= 4,
        "expected >= 4 total requests served, got {}",
        status.requests_served
    );

    tracing::info!(
        total = status.requests_served,
        "total requests_served counter verified"
    );
}

/// Test that RSS memory reporting returns a reasonable value.
#[tokio::test]
#[serial]
async fn test_status_rss_mb_is_reasonable() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    let status = client.status().await.expect("status check");

    // RSS should be positive on Linux (our CI/test platform)
    #[cfg(target_os = "linux")]
    assert!(
        status.rss_mb > 0.0,
        "RSS should be positive on Linux, got {}",
        status.rss_mb
    );

    // RSS should be reasonable (< 1GB for a daemon that hasn't loaded ML models)
    assert!(
        status.rss_mb < 1024.0,
        "RSS should be under 1GB without ML models, got {} MB",
        status.rss_mb
    );

    tracing::info!(rss_mb = status.rss_mb, "RSS memory reporting verified");
}

/// Test that status shows no models when daemon just started.
#[tokio::test]
#[serial]
async fn test_no_models_loaded_on_fresh_start() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    let status = client.status().await.expect("status check");
    assert!(
        status.models.is_empty(),
        "fresh daemon should have no models loaded, got {}",
        status.models.len()
    );

    tracing::info!("fresh daemon correctly starts with no models");
}

/// Test that max_models=1 limits the loaded model count.
///
/// Since only the hash embedder is guaranteed available without model downloads,
/// this test verifies the max_models argument is accepted and the daemon operates
/// correctly with a restricted model limit.
#[tokio::test]
#[serial]
async fn test_max_models_limits_loaded_count() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn_with_max_models(1)
        .await
        .expect("spawn daemon with max_models=1");

    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    // Load hash embedder
    client
        .embed(&["test"], Some("hash"), None)
        .await
        .expect("embed with hash");

    let status = client.status().await.expect("status check");
    assert!(
        status.models.len() <= 1,
        "should have at most 1 model loaded with max_models=1, got {}",
        status.models.len()
    );

    tracing::info!(loaded = status.models.len(), "max_models=1 limit honored");
}

/// Test that embedding results are deterministic across multiple calls.
#[tokio::test]
#[serial]
async fn test_hash_embedder_deterministic_results() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    let text = "deterministic test input";
    let result1 = client
        .embed(&[text], Some("hash"), None)
        .await
        .expect("first embed");
    let result2 = client
        .embed(&[text], Some("hash"), None)
        .await
        .expect("second embed");

    assert_eq!(
        result1.len(),
        result2.len(),
        "should return same number of embeddings"
    );
    assert_eq!(
        result1[0].len(),
        result2[0].len(),
        "embeddings should have same dimensions"
    );

    // Compare bitwise for exact equality (hash embedder should be perfectly deterministic)
    for (i, (a, b)) in result1[0].iter().zip(result2[0].iter()).enumerate() {
        assert_eq!(
            a.to_bits(),
            b.to_bits(),
            "embedding dimension {i} should be identical across calls"
        );
    }

    tracing::info!(
        dims = result1[0].len(),
        "hash embedder determinism verified"
    );
}

/// Test batch embedding with multiple texts.
#[tokio::test]
#[serial]
async fn test_batch_embedding() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    let texts = &["first text", "second text", "third text"];
    let results = client
        .embed(texts, Some("hash"), None)
        .await
        .expect("batch embed");

    assert_eq!(
        results.len(),
        texts.len(),
        "should return one embedding per input text"
    );

    // Each embedding should have the same dimension
    let dim = results[0].len();
    for (i, emb) in results.iter().enumerate() {
        assert_eq!(
            emb.len(),
            dim,
            "embedding {i} should have same dimension as first ({dim})"
        );
    }

    // Different inputs should produce different embeddings
    assert_ne!(
        results[0], results[1],
        "different texts should produce different embeddings"
    );

    tracing::info!(count = results.len(), dim, "batch embedding verified");
}

/// Test that in_flight counter is zero when no requests are active.
#[tokio::test]
#[serial]
async fn test_in_flight_zero_when_idle() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    // Do a request first to ensure daemon is warmed up
    let _ = client.health().await.expect("health check");

    // Small delay to let in_flight settle
    sleep(Duration::from_millis(50)).await;

    let status = client.status().await.expect("status check");
    // The status request itself may be in_flight=1, but after response it should be 0
    // We check that it's at most 1 (the status request itself)
    assert!(
        status.in_flight <= 1,
        "in_flight should be 0 or 1 (status request itself), got {}",
        status.in_flight
    );

    tracing::info!(
        in_flight = status.in_flight,
        "in_flight counter is low when idle"
    );
}

// =============================================================================
// Daemon Reranker E2E Tests (bd-2cnh)
// =============================================================================

/// Test that reranking with 'none' reranker returns an error.
///
/// The "none" reranker is a sentinel indicating no reranking should be done.
/// When explicitly requested, the daemon returns an UNKNOWN_MODEL error.
#[tokio::test]
#[serial]
async fn test_rerank_none_reranker_returns_error() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    let result = client
        .rerank("test query", &["doc one", "doc two"], Some("none"))
        .await;

    assert!(result.is_err(), "'none' reranker should return error");
    let err_str = result.unwrap_err().to_string();
    tracing::info!(
        error = err_str.as_str(),
        "'none' reranker correctly rejected"
    );
}

/// Test that reranking with an unknown model returns an error.
#[tokio::test]
#[serial]
async fn test_rerank_unknown_model_returns_error() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    let result = client
        .rerank(
            "machine learning",
            &["doc1", "doc2"],
            Some("totally-fake-reranker-xyz"),
        )
        .await;

    assert!(result.is_err(), "unknown reranker should return error");
    let err_str = result.unwrap_err().to_string();
    tracing::info!(error = err_str.as_str(), "unknown reranker error verified");
}

/// Test rerank with empty documents list.
#[tokio::test]
#[serial]
async fn test_rerank_empty_documents() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    // Empty docs with "none" reranker - should still error on the reranker
    let result = client.rerank("query", &[], Some("none")).await;

    // Expected: error because "none" reranker is not a real reranker
    assert!(
        result.is_err(),
        "empty documents with 'none' reranker should error"
    );

    tracing::info!("empty documents rerank correctly handled");
}

/// Test rerank protocol via raw request (using "none" model to test error path).
#[tokio::test]
#[serial]
#[cfg(unix)]
#[allow(clippy::cast_possible_truncation)]
async fn test_rerank_protocol_error_path() {
    use tokio::net::UnixStream;

    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    let mut stream = UnixStream::connect(&daemon.socket_path)
        .await
        .expect("connect to socket");

    // Send a rerank request with "none" reranker via raw protocol
    let request = Request::Rerank {
        query: "test query".to_string(),
        documents: vec!["document one".to_string(), "document two".to_string()],
        model: "none".to_string(),
    };

    let envelope = Envelope::from_request(100, &request).expect("create envelope");
    let bytes = rmp_serde::to_vec(&envelope).expect("serialize");

    stream
        .write_u32(bytes.len() as u32)
        .await
        .expect("write length");
    stream.write_all(&bytes).await.expect("write payload");

    // Read response
    let len = stream.read_u32().await.expect("read response length");
    let mut buf = vec![0u8; len as usize];
    stream.read_exact(&mut buf).await.expect("read response");

    let resp_envelope: Envelope = rmp_serde::from_slice(&buf).expect("decode envelope");
    assert_eq!(resp_envelope.id, 100, "response ID should match request");

    let response: Response =
        rmp_serde::from_slice(&resp_envelope.payload).expect("decode response");

    match response {
        Response::Error { code, message } => {
            assert_eq!(
                code,
                error_codes::UNKNOWN_MODEL,
                "expected UNKNOWN_MODEL error for 'none' reranker"
            );
            tracing::info!(
                code,
                message = message.as_str(),
                "rerank 'none' correctly returns UNKNOWN_MODEL error via protocol"
            );
        }
        _ => panic!("expected Error response for 'none' reranker, got {response:?}"),
    }
}

/// Test that daemon remains healthy after reranker errors.
#[tokio::test]
#[serial]
async fn test_daemon_healthy_after_reranker_errors() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    // Trigger multiple reranker errors
    for model in &["none", "fake-model-1", "fake-model-2"] {
        let _ = client.rerank("query", &["doc"], Some(model)).await;
    }

    // Daemon should still be healthy
    let health = client.health().await.expect("health after errors");
    assert!(
        health.uptime_secs < 30,
        "daemon should still be running after errors"
    );

    // Embed should still work
    let embed = client
        .embed(&["still works"], Some("hash"), None)
        .await
        .expect("embed after reranker errors");
    assert!(!embed.is_empty());

    tracing::info!(
        uptime = health.uptime_secs,
        "daemon remains healthy after reranker errors"
    );
}

/// Test rerank with no explicit model (client uses "default" internally).
///
/// When `None` is passed for the model, the client substitutes `"default"`.
/// The daemon should handle this gracefully without crashing.
#[tokio::test]
#[serial]
async fn test_rerank_with_no_model_specified() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    // None becomes "default" via client.rerank()'s unwrap_or("default")
    let result = client.rerank("test query", &["doc"], None).await;

    // Result depends on what the "default" reranker resolves to.
    // The important thing is the daemon doesn't crash.
    tracing::info!(
        success = result.is_ok(),
        "rerank with no model specified handled without crash"
    );

    // Daemon should still be healthy regardless
    let health = client.health().await.expect("health check");
    assert!(health.uptime_secs < 30);
}

// =============================================================================
// MRL Dimension Truncation E2E Tests (bd-ou8b)
// =============================================================================

/// Test that embedding with dims=None returns full native dimensions.
///
/// The hash embedder has a native dimension of 384. Without MRL truncation,
/// the full embedding should be returned.
#[tokio::test]
#[serial]
async fn test_embed_without_dims_returns_full_dimensions() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    let result = client
        .embed(&["test text for full dimensions"], Some("hash"), None)
        .await
        .expect("embed without dims");

    assert_eq!(result.len(), 1);
    // Hash embedder defaults to 384 dimensions
    assert_eq!(
        result[0].len(),
        384,
        "hash embedder should return 384 dimensions by default"
    );

    tracing::info!(
        dims = result[0].len(),
        "full dimensions verified without MRL"
    );
}

/// Test that embedding with dims equal to native dimension succeeds.
///
/// When dims matches the model's native dimension, embeddings are returned
/// without any truncation.
#[tokio::test]
#[serial]
async fn test_embed_dims_equals_native_succeeds() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    // Request exactly the native dimension (384 for hash embedder)
    let result = client
        .embed(&["test text"], Some("hash"), Some(384))
        .await
        .expect("embed with native dims");

    assert_eq!(result.len(), 1);
    assert_eq!(
        result[0].len(),
        384,
        "should return embeddings at native dimension"
    );

    tracing::info!("dims=native (384) correctly returns full embeddings");
}

/// Test that non-MRL embedder rejects truncation to different dimension.
///
/// The hash embedder does not support MRL. Requesting dims != native should
/// return an INVALID_REQUEST error.
#[tokio::test]
#[serial]
async fn test_embed_non_mrl_rejects_truncation() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    // Hash embedder is 384 dims, doesn't support MRL
    // Requesting 128 should fail with INVALID_REQUEST
    let result = client.embed(&["test text"], Some("hash"), Some(128)).await;

    assert!(
        result.is_err(),
        "non-MRL embedder should reject dimension truncation"
    );

    let err_str = result.unwrap_err().to_string();
    tracing::info!(
        error = err_str.as_str(),
        "non-MRL embedder correctly rejects truncation"
    );
}

/// Test dimension truncation via raw protocol.
///
/// Sends an embed request with dims parameter via raw protocol to verify
/// the envelope correctly carries the dims field end-to-end.
#[tokio::test]
#[serial]
#[cfg(unix)]
#[allow(clippy::cast_possible_truncation)]
async fn test_dims_parameter_protocol_roundtrip() {
    use tokio::net::UnixStream;

    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    let mut stream = UnixStream::connect(&daemon.socket_path)
        .await
        .expect("connect to socket");

    // Send embed request with dims=384 (native, should succeed)
    let request = Request::Embed {
        texts: vec!["protocol test".to_string()],
        model: "hash".to_string(),
        dims: Some(384),
    };

    let envelope = Envelope::from_request(200, &request).expect("create envelope");
    let bytes = rmp_serde::to_vec(&envelope).expect("serialize");

    stream
        .write_u32(bytes.len() as u32)
        .await
        .expect("write length");
    stream.write_all(&bytes).await.expect("write payload");

    let len = stream.read_u32().await.expect("read response length");
    let mut buf = vec![0u8; len as usize];
    stream.read_exact(&mut buf).await.expect("read response");

    let resp_envelope: Envelope = rmp_serde::from_slice(&buf).expect("decode envelope");
    assert_eq!(resp_envelope.id, 200);

    let response: Response =
        rmp_serde::from_slice(&resp_envelope.payload).expect("decode response");

    match response {
        Response::Embeddings { vectors } => {
            assert_eq!(vectors.len(), 1);
            assert_eq!(vectors[0].len(), 384);
            tracing::info!("dims=384 protocol roundtrip succeeded");
        }
        Response::Error { code, message } => {
            panic!("unexpected error {code}: {message}");
        }
        _ => panic!("unexpected response: {response:?}"),
    }
}

/// Test that non-MRL truncation error includes model name.
///
/// Verifies the error response contains useful information about
/// which model doesn't support MRL.
#[tokio::test]
#[serial]
#[cfg(unix)]
#[allow(clippy::cast_possible_truncation)]
async fn test_non_mrl_truncation_error_message() {
    use tokio::net::UnixStream;

    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    let mut stream = UnixStream::connect(&daemon.socket_path)
        .await
        .expect("connect to socket");

    // Request truncation on non-MRL embedder
    let request = Request::Embed {
        texts: vec!["test".to_string()],
        model: "hash".to_string(),
        dims: Some(128), // Not native, hash doesn't support MRL
    };

    let envelope = Envelope::from_request(201, &request).expect("create envelope");
    let bytes = rmp_serde::to_vec(&envelope).expect("serialize");

    stream
        .write_u32(bytes.len() as u32)
        .await
        .expect("write length");
    stream.write_all(&bytes).await.expect("write payload");

    let len = stream.read_u32().await.expect("read response length");
    let mut buf = vec![0u8; len as usize];
    stream.read_exact(&mut buf).await.expect("read response");

    let resp_envelope: Envelope = rmp_serde::from_slice(&buf).expect("decode envelope");
    let response: Response =
        rmp_serde::from_slice(&resp_envelope.payload).expect("decode response");

    match response {
        Response::Error { code, message } => {
            assert_eq!(
                code,
                error_codes::INVALID_REQUEST,
                "should return INVALID_REQUEST for non-MRL truncation"
            );
            assert!(
                message.contains("MRL") || message.contains("truncation"),
                "error message should mention MRL: {message}"
            );
            tracing::info!(
                code,
                message = message.as_str(),
                "non-MRL truncation error message verified"
            );
        }
        _ => panic!("expected Error response, got {response:?}"),
    }
}

/// Test that daemon remains functional after MRL truncation errors.
#[tokio::test]
#[serial]
async fn test_daemon_healthy_after_mrl_errors() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    // Trigger MRL errors with non-MRL embedder
    for dims in [64, 128, 256, 512] {
        let _ = client.embed(&["test"], Some("hash"), Some(dims)).await;
    }

    // Daemon should still be healthy and serving valid requests
    let health = client.health().await.expect("health after MRL errors");
    assert!(health.uptime_secs < 30);

    let embed = client
        .embed(&["still works"], Some("hash"), None)
        .await
        .expect("embed after MRL errors");
    assert_eq!(embed[0].len(), 384);

    tracing::info!(
        uptime = health.uptime_secs,
        "daemon healthy after MRL truncation errors"
    );
}

/// Test batch embedding with dims parameter (native dims).
#[tokio::test]
#[serial]
async fn test_batch_embed_with_native_dims() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=debug,daemon_e2e=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    let texts = &["text one", "text two", "text three"];
    let result = client
        .embed(texts, Some("hash"), Some(384))
        .await
        .expect("batch embed with native dims");

    assert_eq!(result.len(), 3, "should return 3 embeddings");
    for (i, emb) in result.iter().enumerate() {
        assert_eq!(emb.len(), 384, "embedding {i} should have 384 dims");
    }

    tracing::info!("batch embed with native dims verified");
}
