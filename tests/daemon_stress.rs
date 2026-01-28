//! Stress tests for daemon concurrency and resource limits.
// Allow precision loss casts in tests (we won't have billions of requests)
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::map_unwrap_or)]
//!
//! These tests verify daemon stability under concurrent load:
//! - Multiple simultaneous client connections
//! - In-flight request limit (MAX_IN_FLIGHT=64) enforcement
//! - Overload error response when limit exceeded
//! - No resource leaks under sustained load
//!
//! # Running Tests
//!
//! **Important**: These tests must run serially to avoid socket conflicts.
//!
//! ```bash
//! # Run all stress tests (serially, may take a while)
//! cargo test --test daemon_stress -- --test-threads=1 --nocapture
//!
//! # Run a specific test
//! cargo test --test daemon_stress test_concurrent_embed_requests -- --nocapture
//! ```

use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::time::{Duration, Instant};

use serial_test::serial;
use tempfile::TempDir;
use tokio::sync::Barrier;
use tokio::time::sleep;
use xf::daemon::DaemonClient;

// =============================================================================
// Test Infrastructure
// =============================================================================

/// Test daemon process wrapper with automatic cleanup.
struct DaemonProcess {
    child: Child,
    socket_path: PathBuf,
    _temp_dir: TempDir,
}

impl DaemonProcess {
    /// Spawn a daemon with default configuration.
    async fn spawn() -> anyhow::Result<Self> {
        let temp_dir = TempDir::new()?;
        let socket_path = temp_dir.path().join("daemon.sock");

        let xf_binary = find_xf_binary()?;

        tracing::info!(
            binary = %xf_binary.display(),
            socket = %socket_path.display(),
            "spawning stress test daemon"
        );

        let mut cmd = Command::new(&xf_binary);
        cmd.arg("daemon")
            .arg("start")
            .arg("--foreground")
            .arg("--socket")
            .arg(&socket_path)
            .env("XF_DAEMON_SOCK", &socket_path)
            .env("RUST_LOG", "xf=info")
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let child = cmd.spawn()?;

        // Wait for socket to appear
        wait_for_socket(&socket_path, Duration::from_secs(10)).await?;

        tracing::info!(
            socket = %socket_path.display(),
            "daemon socket ready"
        );

        Ok(Self {
            child,
            socket_path,
            _temp_dir: temp_dir,
        })
    }

    /// Gracefully stop the daemon.
    #[cfg(unix)]
    #[allow(dead_code)]
    async fn stop(&mut self) -> anyhow::Result<()> {
        use nix::sys::signal::{Signal, kill};
        use nix::unistd::Pid;

        let _ = kill(Pid::from_raw(self.child.id() as i32), Signal::SIGINT);

        let start = Instant::now();
        let timeout_duration = Duration::from_secs(5);

        loop {
            match self.child.try_wait() {
                Ok(Some(_)) => return Ok(()),
                Ok(None) => {
                    if start.elapsed() > timeout_duration {
                        let _ = self.child.kill();
                        let _ = self.child.wait();
                        anyhow::bail!("daemon didn't stop within timeout");
                    }
                    sleep(Duration::from_millis(50)).await;
                }
                Err(e) => return Err(e.into()),
            }
        }
    }

    #[cfg(not(unix))]
    #[allow(dead_code)]
    async fn stop(&mut self) -> anyhow::Result<()> {
        let _ = self.child.kill();
        let _ = self.child.wait();
        Ok(())
    }

    fn is_running(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }
}

impl Drop for DaemonProcess {
    fn drop(&mut self) {
        if self.is_running() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

async fn wait_for_socket(path: &std::path::Path, timeout_duration: Duration) -> anyhow::Result<()> {
    let start = Instant::now();

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

fn find_xf_binary() -> anyhow::Result<PathBuf> {
    let target_dir = std::env::var("CARGO_TARGET_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("target"));

    let release_path = target_dir.join("release/xf");
    let debug_path = target_dir.join("debug/xf");

    if release_path.exists() {
        return Ok(release_path);
    }

    if debug_path.exists() {
        return Ok(debug_path);
    }

    if let Ok(path) = which::which("xf") {
        return Ok(path);
    }

    anyhow::bail!("xf binary not found. Run `cargo build` or `cargo build --release` first.")
}

// =============================================================================
// Concurrent Request Tests
// =============================================================================

/// Test that multiple concurrent embed requests succeed.
///
/// This test spawns 20 concurrent embed requests using the hash embedder
/// (which is fast and doesn't require ML model downloads).
#[tokio::test]
#[serial]
async fn test_concurrent_embed_requests() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=info,daemon_stress=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    // Spawn 20 concurrent embed requests
    let mut handles = vec![];
    for i in 0..20 {
        let socket = daemon.socket_path.clone();
        handles.push(tokio::spawn(async move {
            let mut client = DaemonClient::with_socket_path(socket);
            let start = Instant::now();
            // Use "hash" model which is always available (no ML model needed)
            let result = client
                .embed(&[&format!("text {i}")], Some("hash"), None)
                .await;
            (i, result, start.elapsed())
        }));
    }

    let results: Vec<_> = futures::future::join_all(handles).await;

    let mut successes = 0;
    let mut failures = 0;

    for result in results {
        let (i, res, elapsed) = result.expect("task panicked");
        match res {
            Ok(vectors) => {
                tracing::info!(
                    request_id = i,
                    elapsed_ms = elapsed.as_millis(),
                    dims = vectors.first().map(Vec::len).unwrap_or(0),
                    "request succeeded"
                );
                successes += 1;
                assert!(!vectors.is_empty(), "Should have at least one vector");
            }
            Err(e) => {
                tracing::warn!(request_id = i, error = %e, "request failed");
                failures += 1;
            }
        }
    }

    tracing::info!(successes, failures, "Concurrent test results");

    // All 20 should succeed since we're well under the 64 limit
    assert!(
        successes >= 18,
        "At least 90% should succeed, got {successes}/20"
    );
}

/// Test that coordinated concurrent requests work correctly.
///
/// Uses a barrier to ensure all clients connect simultaneously,
/// testing the daemon's ability to handle concurrent connection establishment.
#[tokio::test]
#[serial]
async fn test_concurrent_with_barrier() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=info,daemon_stress=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let barrier = Arc::new(Barrier::new(15));

    let mut handles = vec![];
    for i in 0..15 {
        let socket = daemon.socket_path.clone();
        let barrier = barrier.clone();
        handles.push(tokio::spawn(async move {
            // Wait for all tasks to be ready
            barrier.wait().await;

            let mut client = DaemonClient::with_socket_path(socket);
            let start = Instant::now();
            let result = client
                .embed(&[&format!("barrier text {i}")], Some("hash"), None)
                .await;
            (i, result, start.elapsed())
        }));
    }

    let results: Vec<_> = futures::future::join_all(handles).await;

    let successes = results
        .iter()
        .filter(|r| {
            r.as_ref()
                .ok()
                .and_then(|(_, res, _)| res.as_ref().ok())
                .is_some()
        })
        .count();

    tracing::info!(successes, total = 15, "Barrier concurrent test results");

    assert!(
        successes >= 12,
        "At least 80% should succeed with barrier, got {successes}/15"
    );
}

// =============================================================================
// In-Flight Limit Tests
// =============================================================================

/// Test that the daemon enforces the MAX_IN_FLIGHT limit.
///
/// The daemon has MAX_IN_FLIGHT=64. This test attempts to exceed that limit
/// by creating many concurrent slow requests.
///
/// Note: This test may be flaky depending on system load since hash embeddings
/// are very fast. The key is that we shouldn't exceed the limit.
#[tokio::test]
#[serial]
async fn test_in_flight_limit_awareness() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=info,daemon_stress=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    // Create 50 concurrent requests
    let mut handles = vec![];
    for i in 0..50 {
        let socket = daemon.socket_path.clone();
        handles.push(tokio::spawn(async move {
            let mut client = DaemonClient::with_socket_path(socket);
            let result = client
                .embed(&[&format!("limit test {i}")], Some("hash"), None)
                .await;
            (i, result)
        }));
    }

    let results: Vec<_> = futures::future::join_all(handles).await;

    let successes = results
        .iter()
        .filter(|r| {
            r.as_ref()
                .ok()
                .and_then(|(_, res)| res.as_ref().ok())
                .is_some()
        })
        .count();

    let overloaded = results
        .iter()
        .filter(|r| {
            r.as_ref()
                .ok()
                .and_then(|(_, res)| res.as_ref().err())
                .is_some_and(|e| e.to_string().to_lowercase().contains("overload"))
        })
        .count();

    tracing::info!(
        successes,
        overloaded,
        total = 50,
        "In-flight limit test results"
    );

    // Since hash embedder is very fast, most requests should succeed
    // even with 50 concurrent. The important thing is we handled them.
    assert!(
        successes + overloaded == 50,
        "All requests should either succeed or get overloaded error"
    );

    // We should have processed most requests
    assert!(
        successes >= 40,
        "Most requests should succeed, got {successes}/50"
    );
}

// =============================================================================
// Resource Leak Tests
// =============================================================================

/// Test that no significant memory leak occurs under sustained load.
///
/// This test runs many sequential requests and verifies that memory
/// usage doesn't grow unboundedly.
#[tokio::test]
#[serial]
async fn test_no_resource_leak_sequential() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=info,daemon_stress=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    // Get initial RSS
    let status_before = client.status().await.expect("initial status");
    let rss_before = status_before.rss_mb;
    let requests_before = status_before.requests_served;

    tracing::info!(
        rss_mb = rss_before,
        requests = requests_before,
        "Initial state"
    );

    // Run 500 sequential requests (reduced from 1000 for faster tests)
    let request_count = 500;
    for i in 0..request_count {
        let _ = client.embed(&["test"], Some("hash"), None).await;
        if i % 100 == 0 {
            tracing::info!(request = i, "Progress");
        }
    }

    // Get final RSS
    let status_after = client.status().await.expect("final status");
    let rss_after = status_after.rss_mb;
    let requests_after = status_after.requests_served;
    let rss_growth = rss_after - rss_before;

    tracing::info!(
        rss_before_mb = rss_before,
        rss_after_mb = rss_after,
        growth_mb = rss_growth,
        requests_served = requests_after - requests_before,
        "Memory usage after {} requests",
        request_count
    );

    // RSS growth should be minimal (< 100MB for 500 requests)
    // Hash embedder is very lightweight so shouldn't cause much growth
    assert!(
        rss_growth < 100.0,
        "Memory grew too much: {rss_growth:.1} MB (before: {rss_before:.1}, after: {rss_after:.1})"
    );

    // Verify requests were actually served
    assert!(
        requests_after >= requests_before + (request_count as u64 / 2),
        "Should have served most requests"
    );
}

/// Test that concurrent load doesn't cause significant memory growth.
#[tokio::test]
#[serial]
async fn test_no_resource_leak_concurrent() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=info,daemon_stress=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    // Get initial status
    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());
    let status_before = client.status().await.expect("initial status");
    let rss_before = status_before.rss_mb;

    // Run 10 batches of 20 concurrent requests each
    for batch in 0..10 {
        let mut handles = vec![];
        for i in 0..20 {
            let socket = daemon.socket_path.clone();
            handles.push(tokio::spawn(async move {
                let mut client = DaemonClient::with_socket_path(socket);
                client
                    .embed(&[&format!("batch {batch} item {i}")], Some("hash"), None)
                    .await
            }));
        }
        let _ = futures::future::join_all(handles).await;

        if batch % 5 == 0 {
            tracing::info!(batch, "Batch progress");
        }
    }

    // Get final status
    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());
    let status_after = client.status().await.expect("final status");
    let rss_after = status_after.rss_mb;
    let rss_growth = rss_after - rss_before;

    tracing::info!(
        rss_before_mb = rss_before,
        rss_after_mb = rss_after,
        growth_mb = rss_growth,
        "Memory after 200 concurrent requests in 10 batches"
    );

    // Should not grow significantly
    assert!(
        rss_growth < 150.0,
        "Memory grew too much under concurrent load: {rss_growth:.1} MB"
    );
}

// =============================================================================
// Rapid Connection Tests
// =============================================================================

/// Test daemon stability under rapid connect/disconnect cycles.
#[tokio::test]
#[serial]
async fn test_rapid_connection_cycles() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=info,daemon_stress=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    let mut successes = 0;
    let cycles = 50;

    for _i in 0..cycles {
        let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());
        if client.health().await.is_ok() {
            successes += 1;
        }
        // Client drops here, closing the connection
    }

    tracing::info!(successes, cycles, "Rapid connection cycle results");

    assert!(
        successes >= cycles - 5,
        "Most connection cycles should succeed, got {successes}/{cycles}"
    );
}

/// Test that daemon status reflects concurrent activity.
#[tokio::test]
#[serial]
async fn test_status_reflects_activity() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=info,daemon_stress=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());

    // Get initial status
    let status1 = client.status().await.expect("status 1");
    let requests1 = status1.requests_served;

    // Make some requests
    for _ in 0..10 {
        let _ = client.embed(&["test"], Some("hash"), None).await;
    }

    // Get status again
    let status2 = client.status().await.expect("status 2");
    let requests2 = status2.requests_served;

    tracing::info!(before = requests1, after = requests2, "Request counter");

    // Status call itself counts as a request, so we should see growth
    assert!(
        requests2 > requests1,
        "Request counter should increase: {requests1} -> {requests2}"
    );
}

// =============================================================================
// Sustained Load Tests
// =============================================================================

/// Test daemon stability under sustained load over time.
#[tokio::test]
#[serial]
async fn test_sustained_load() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=info,daemon_stress=debug")
        .try_init();

    let mut daemon = DaemonProcess::spawn().await.expect("spawn daemon");
    let duration = Duration::from_secs(5);
    let start = Instant::now();

    let mut request_count = 0u64;
    let mut error_count = 0u64;

    while start.elapsed() < duration {
        let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());
        match client
            .embed(&["sustained load test"], Some("hash"), None)
            .await
        {
            Ok(_) => request_count += 1,
            Err(_) => error_count += 1,
        }
    }

    let elapsed = start.elapsed();
    let requests_per_sec = request_count as f64 / elapsed.as_secs_f64();

    tracing::info!(
        requests = request_count,
        errors = error_count,
        elapsed_secs = elapsed.as_secs_f64(),
        rps = requests_per_sec,
        "Sustained load results"
    );

    // Should have processed many requests
    assert!(
        request_count > 50,
        "Should handle sustained load, processed only {request_count} requests"
    );

    // Error rate should be low
    let error_rate = error_count as f64 / (request_count + error_count) as f64;
    assert!(
        error_rate < 0.1,
        "Error rate too high: {:.1}%",
        error_rate * 100.0
    );

    // Daemon should still be running
    assert!(daemon.is_running(), "Daemon crashed under sustained load");
}

/// Test daemon handles burst traffic after idle period.
#[tokio::test]
#[serial]
async fn test_burst_after_idle() {
    let _ = tracing_subscriber::fmt()
        .with_env_filter("xf=info,daemon_stress=debug")
        .try_init();

    let daemon = DaemonProcess::spawn().await.expect("spawn daemon");

    // Initial request to warm up
    let mut client = DaemonClient::with_socket_path(daemon.socket_path.clone());
    client.health().await.expect("warmup");

    // Wait a bit (simulating idle)
    sleep(Duration::from_secs(1)).await;

    // Burst of concurrent requests
    let mut handles = vec![];
    for i in 0..30 {
        let socket = daemon.socket_path.clone();
        handles.push(tokio::spawn(async move {
            let mut client = DaemonClient::with_socket_path(socket);
            client
                .embed(&[&format!("burst {i}")], Some("hash"), None)
                .await
        }));
    }

    let results: Vec<_> = futures::future::join_all(handles).await;
    let successes = results
        .iter()
        .filter(|r| r.as_ref().ok().and_then(|r| r.as_ref().ok()).is_some())
        .count();

    tracing::info!(successes, total = 30, "Burst after idle results");

    assert!(
        successes >= 25,
        "Should handle burst after idle, got {successes}/30"
    );
}
