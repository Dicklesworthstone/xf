//! E2E lifecycle tests for the daemon.
//!
//! These tests spawn real daemon processes and verify:
//! - Startup and health check
//! - Graceful shutdown
//! - Socket permissions
//! - PID file handling
//! - Signal handling (SIGINT, SIGTERM)
//! - Idle timeout shutdown
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
use tokio::time::sleep;
use xf::daemon::DaemonClient;

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
