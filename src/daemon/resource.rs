//! Resource management for the model daemon.
//!
//! Applies nice/ionice settings, memory limits, and thread pool bounds
//! to make the daemon a good system citizen.
//!
//! # Platform Support
//!
//! - Linux: Full support (nice via renice, ionice via command)
//! - macOS: Partial support (nice via renice)
//! - Other Unix: Basic support (nice only)
//! - Windows: No support (daemon doesn't run on Windows)

use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;

/// Resource configuration for the daemon.
#[derive(Debug, Clone)]
pub struct ResourceConfig {
    /// CPU nice level (0-19 for non-root, -20 to 19 for root).
    /// Higher values = lower priority. Default: 10.
    pub nice_level: i32,

    /// I/O priority policy.
    pub io_priority: IoPriority,

    /// Memory limit in MB. Default: 2048.
    pub memory_limit_mb: u64,

    /// Maximum threads for model inference. Default: min(4, cpus/2).
    pub max_threads: usize,

    /// Idle timeout before daemon auto-shutdowns.
    pub idle_timeout: Duration,

    /// Socket path override. If None, uses default /tmp/xf-daemon-{uid}.sock.
    pub socket_path: Option<PathBuf>,
}

impl Default for ResourceConfig {
    fn default() -> Self {
        let cpus = num_cpus::get();
        Self {
            nice_level: 10,
            io_priority: IoPriority::Idle,
            memory_limit_mb: 2048,
            max_threads: (cpus / 2).clamp(1, 4),
            idle_timeout: Duration::from_secs(30 * 60),
            socket_path: None,
        }
    }
}

impl ResourceConfig {
    /// Load config from TOML file at the given path.
    ///
    /// If the file doesn't exist, returns default config.
    /// Environment variables can override file values:
    /// - `XF_DAEMON_NICE`: CPU nice level
    /// - `XF_DAEMON_MEMORY_MB`: Memory limit
    /// - `XF_DAEMON_THREADS`: Max threads
    /// - `XF_DAEMON_TIMEOUT`: Idle timeout in seconds
    /// - `XF_DAEMON_SOCKET`: Socket path
    #[allow(clippy::missing_errors_doc)]
    pub fn load(path: Option<&PathBuf>) -> anyhow::Result<Self> {
        let mut config = if let Some(p) = path {
            if p.exists() {
                let content = std::fs::read_to_string(p)?;
                Self::from_toml(&content)?
            } else {
                Self::default()
            }
        } else {
            // Try default config locations
            let default_paths = [
                dirs::config_dir().map(|d| d.join("xf/daemon.toml")),
                Some(PathBuf::from("/etc/xf/daemon.toml")),
            ];

            let mut found = None;
            for maybe_path in default_paths.into_iter().flatten() {
                if maybe_path.exists() {
                    let content = std::fs::read_to_string(&maybe_path)?;
                    found = Some(Self::from_toml(&content)?);
                    tracing::info!(path = %maybe_path.display(), "loaded config");
                    break;
                }
            }
            found.unwrap_or_default()
        };

        // Apply environment variable overrides
        config.apply_env_overrides();

        Ok(config)
    }

    /// Parse config from TOML string.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn from_toml(content: &str) -> anyhow::Result<Self> {
        let table: toml::Table = toml::from_str(content)?;
        let mut config = Self::default();

        if let Some(daemon) = table.get("daemon").and_then(toml::Value::as_table) {
            if let Some(v) = daemon.get("nice_level").and_then(toml::Value::as_integer) {
                config.nice_level = v as i32;
            }
            if let Some(v) = daemon
                .get("memory_limit_mb")
                .and_then(toml::Value::as_integer)
            {
                config.memory_limit_mb = v as u64;
            }
            if let Some(v) = daemon.get("max_threads").and_then(toml::Value::as_integer) {
                config.max_threads = v as usize;
            }
            if let Some(v) = daemon
                .get("idle_timeout_secs")
                .and_then(toml::Value::as_integer)
            {
                config.idle_timeout = Duration::from_secs(v as u64);
            }
            if let Some(v) = daemon.get("socket_path").and_then(toml::Value::as_str) {
                config.socket_path = Some(PathBuf::from(v));
            }
            if let Some(v) = daemon.get("io_priority").and_then(toml::Value::as_str) {
                config.io_priority = match v.to_lowercase().as_str() {
                    "best_effort" | "best-effort" | "besteffort" => IoPriority::BestEffort,
                    "none" => IoPriority::None,
                    // "idle" and anything else defaults to Idle
                    _ => IoPriority::Idle,
                };
            }
        }

        Ok(config)
    }

    /// Apply environment variable overrides.
    fn apply_env_overrides(&mut self) {
        if let Ok(v) = std::env::var("XF_DAEMON_NICE") {
            if let Ok(n) = v.parse::<i32>() {
                self.nice_level = n;
            }
        }
        if let Ok(v) = std::env::var("XF_DAEMON_MEMORY_MB") {
            if let Ok(n) = v.parse::<u64>() {
                self.memory_limit_mb = n;
            }
        }
        if let Ok(v) = std::env::var("XF_DAEMON_THREADS") {
            if let Ok(n) = v.parse::<usize>() {
                self.max_threads = n;
            }
        }
        if let Ok(v) = std::env::var("XF_DAEMON_TIMEOUT") {
            if let Ok(n) = v.parse::<u64>() {
                self.idle_timeout = Duration::from_secs(n);
            }
        }
        if let Ok(v) = std::env::var("XF_DAEMON_SOCKET") {
            self.socket_path = Some(PathBuf::from(v));
        }
    }

    /// Compute effective max threads based on config and system.
    #[must_use]
    pub fn effective_threads(&self) -> usize {
        let cpus = num_cpus::get();
        self.max_threads.min(cpus).max(1)
    }
}

/// I/O priority policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum IoPriority {
    /// Idle priority - only run I/O when system is idle.
    #[default]
    Idle,
    /// Best effort priority - normal I/O scheduling.
    BestEffort,
    /// Don't set any I/O priority.
    None,
}

/// Apply all resource settings from config.
///
/// This should be called early in daemon startup, before binding sockets
/// or loading models.
///
/// # Errors
///
/// Returns Ok even if some settings fail - non-critical failures are logged.
pub fn apply_resource_settings(config: &ResourceConfig) -> anyhow::Result<()> {
    tracing::info!(
        nice_level = config.nice_level,
        io_priority = ?config.io_priority,
        memory_limit_mb = config.memory_limit_mb,
        max_threads = config.effective_threads(),
        idle_timeout_secs = config.idle_timeout.as_secs(),
        "applying resource settings"
    );

    // Apply CPU nice level via renice command
    if let Err(e) = apply_cpu_nice(config.nice_level) {
        tracing::warn!(error = %e, level = config.nice_level, "failed to set nice level");
    }

    // Apply I/O priority
    if config.io_priority != IoPriority::None {
        if let Err(e) = apply_io_priority(config.io_priority) {
            tracing::warn!(error = %e, "failed to set I/O priority");
        }
    }

    // Configure thread pools
    configure_thread_pools(config.effective_threads());

    // Note: Memory limits via setrlimit require unsafe, so we skip that.
    // The daemon relies on LRU eviction to manage memory instead.
    tracing::info!(
        limit_mb = config.memory_limit_mb,
        "memory limit configured (enforced via model LRU eviction)"
    );

    Ok(())
}

/// Apply CPU nice level using the `renice` command.
#[cfg(unix)]
fn apply_cpu_nice(level: i32) -> std::io::Result<()> {
    let pid = std::process::id();

    let output = Command::new("renice")
        .args(["-n", &level.to_string(), "-p", &pid.to_string()])
        .output()?;

    if output.status.success() {
        tracing::info!(level, "CPU nice level applied via renice");
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::warn!(level, stderr = %stderr.trim(), "renice failed");
        // Not a hard error - we continue even if nice fails
    }
    Ok(())
}

#[cfg(not(unix))]
fn apply_cpu_nice(_level: i32) -> std::io::Result<()> {
    tracing::info!("CPU nice not supported on this platform");
    Ok(())
}

/// Apply I/O priority using the `ionice` command (Linux only).
#[cfg(target_os = "linux")]
fn apply_io_priority(policy: IoPriority) -> std::io::Result<()> {
    let pid = std::process::id();

    // ionice -c class -p pid
    // class 3 = idle, class 2 = best-effort
    let class = match policy {
        IoPriority::Idle => "3",
        IoPriority::BestEffort => "2",
        IoPriority::None => return Ok(()),
    };

    let output = Command::new("ionice")
        .args(["-c", class, "-p", &pid.to_string()])
        .output()?;

    if output.status.success() {
        tracing::info!(class, "I/O priority set via ionice");
    } else {
        let stderr = String::from_utf8_lossy(&output.stderr);
        tracing::warn!(class, stderr = %stderr.trim(), "ionice failed");
    }

    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn apply_io_priority(_policy: IoPriority) -> std::io::Result<()> {
    tracing::info!("I/O priority (ionice) not supported on this platform, skipping");
    Ok(())
}

/// Configure thread pools for inference.
fn configure_thread_pools(max_threads: usize) {
    // Configure rayon thread pool
    if let Err(e) = rayon::ThreadPoolBuilder::new()
        .num_threads(max_threads)
        .build_global()
    {
        // This can fail if rayon is already initialized elsewhere
        tracing::debug!(error = %e, "failed to configure rayon thread pool (may already be initialized)");
    } else {
        tracing::info!(threads = max_threads, "rayon thread pool configured");
    }

    // Note: ONNX runtime thread configuration would require unsafe set_var
    // in Rust 2024 edition. The runtime will use its default thread count.
    tracing::debug!(
        threads = max_threads,
        "thread pool limit target (ONNX may use default)"
    );
}

/// Memory pressure monitor.
///
/// Periodically checks system memory and can signal when pressure is high.
#[derive(Debug)]
pub struct MemoryMonitor {
    /// Memory pressure threshold (0.0 - 1.0). Default: 0.85 (85% used).
    threshold: f64,
    /// Last measured available MB.
    last_available_mb: u64,
}

impl Default for MemoryMonitor {
    fn default() -> Self {
        Self {
            threshold: 0.85,
            last_available_mb: 0,
        }
    }
}

impl MemoryMonitor {
    /// Create a new memory monitor with the given threshold.
    #[must_use]
    #[allow(clippy::missing_const_for_fn)] // clamp is not const
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold: threshold.clamp(0.5, 0.99),
            last_available_mb: 0,
        }
    }

    /// Check if system is under memory pressure.
    ///
    /// Returns true if memory usage exceeds threshold.
    #[allow(clippy::cast_precision_loss)] // Precision loss acceptable for memory stats
    pub fn is_under_pressure(&mut self) -> bool {
        let (total, available) = get_system_memory();

        if total == 0 {
            return false;
        }

        self.last_available_mb = available;
        let used_ratio = 1.0 - (available as f64 / total as f64);

        if used_ratio > self.threshold {
            tracing::warn!(
                used_ratio = format!("{:.1}%", used_ratio * 100.0),
                threshold = format!("{:.1}%", self.threshold * 100.0),
                available_mb = available,
                "memory pressure detected"
            );
            return true;
        }

        false
    }

    /// Get last measured available memory in MB.
    #[must_use]
    pub const fn available_mb(&self) -> u64 {
        self.last_available_mb
    }
}

/// Get system memory info (total MB, available MB).
#[cfg(target_os = "linux")]
fn get_system_memory() -> (u64, u64) {
    // Parse /proc/meminfo
    let Ok(content) = std::fs::read_to_string("/proc/meminfo") else {
        return (0, 0);
    };

    let mut total_kb = 0u64;
    let mut available_kb = 0u64;

    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            if let Ok(kb) = rest.trim().trim_end_matches(" kB").trim().parse::<u64>() {
                total_kb = kb;
            }
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            if let Ok(kb) = rest.trim().trim_end_matches(" kB").trim().parse::<u64>() {
                available_kb = kb;
            }
        }
    }

    (total_kb / 1024, available_kb / 1024)
}

#[cfg(target_os = "macos")]
fn get_system_memory() -> (u64, u64) {
    // macOS: Use sysctl for total, vm_stat for available
    // Get total memory via sysctl
    let total_mb = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<u64>().ok())
        .map(|bytes| bytes / 1024 / 1024)
        .unwrap_or(0);

    // Get page size and free/inactive pages via vm_stat
    let vm_stat = Command::new("vm_stat").output().ok();
    let available_mb = vm_stat
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| {
            let mut free_pages = 0u64;
            let mut inactive_pages = 0u64;
            let page_size = 16384u64; // Assume 16KB pages on Apple Silicon

            for line in s.lines() {
                if line.starts_with("Pages free:") {
                    if let Some(n) = line.split(':').nth(1) {
                        free_pages = n.trim().trim_end_matches('.').parse().unwrap_or(0);
                    }
                } else if line.starts_with("Pages inactive:") {
                    if let Some(n) = line.split(':').nth(1) {
                        inactive_pages = n.trim().trim_end_matches('.').parse().unwrap_or(0);
                    }
                }
            }

            (free_pages + inactive_pages) * page_size / 1024 / 1024
        })
        .unwrap_or(0);

    (total_mb, available_mb)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn get_system_memory() -> (u64, u64) {
    (0, 0)
}

/// Get current process RSS in MB.
#[cfg(target_os = "linux")]
#[must_use]
pub fn get_process_rss_mb() -> f64 {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        for line in status.lines() {
            if let Some(rest) = line.strip_prefix("VmRSS:") {
                let kb_str = rest.trim().trim_end_matches(" kB").trim();
                if let Ok(kb) = kb_str.parse::<f64>() {
                    return kb / 1024.0;
                }
            }
        }
    }
    0.0
}

#[cfg(target_os = "macos")]
#[must_use]
pub fn get_process_rss_mb() -> f64 {
    // macOS: Use ps command to get RSS
    let pid = std::process::id();
    Command::new("ps")
        .args(["-o", "rss=", "-p", &pid.to_string()])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse::<f64>().ok())
        .map(|kb| kb / 1024.0)
        .unwrap_or(0.0)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
#[must_use]
pub fn get_process_rss_mb() -> f64 {
    0.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ResourceConfig::default();
        assert_eq!(config.nice_level, 10);
        assert_eq!(config.io_priority, IoPriority::Idle);
        assert_eq!(config.memory_limit_mb, 2048);
        assert!(config.max_threads >= 1);
        assert_eq!(config.idle_timeout, Duration::from_secs(30 * 60));
    }

    #[test]
    fn test_config_from_toml() {
        let toml = r#"
[daemon]
nice_level = 15
memory_limit_mb = 1024
max_threads = 2
idle_timeout_secs = 600
io_priority = "best_effort"
socket_path = "/tmp/test.sock"
"#;

        let config = ResourceConfig::from_toml(toml).unwrap();
        assert_eq!(config.nice_level, 15);
        assert_eq!(config.memory_limit_mb, 1024);
        assert_eq!(config.max_threads, 2);
        assert_eq!(config.idle_timeout, Duration::from_secs(600));
        assert_eq!(config.io_priority, IoPriority::BestEffort);
        assert_eq!(config.socket_path, Some(PathBuf::from("/tmp/test.sock")));
    }

    #[test]
    fn test_config_effective_threads() {
        let cpus = num_cpus::get();

        let config = ResourceConfig {
            max_threads: 1000,
            ..Default::default()
        };
        assert_eq!(config.effective_threads(), cpus);

        let config2 = ResourceConfig {
            max_threads: 1,
            ..Default::default()
        };
        assert_eq!(config2.effective_threads(), 1);
    }

    #[test]
    fn test_memory_monitor() {
        let mut monitor = MemoryMonitor::new(0.99);
        // With 99% threshold, we shouldn't be under pressure in normal conditions
        let under_pressure = monitor.is_under_pressure();
        // Don't assert the result since it depends on actual system state
        // but verify it doesn't panic
        let _ = under_pressure;
        // available_mb() returns u64, which is always >= 0
        let _available = monitor.available_mb();
    }

    #[test]
    fn test_get_process_rss() {
        let rss = get_process_rss_mb();
        // Should return something >= 0 on supported platforms
        assert!(rss >= 0.0);
    }

    #[test]
    fn test_get_system_memory() {
        let (total, available) = get_system_memory();
        // On supported platforms, should return non-zero values
        #[cfg(any(target_os = "linux", target_os = "macos"))]
        {
            assert!(total > 0, "total memory should be > 0");
            assert!(available > 0, "available memory should be > 0");
            assert!(available <= total, "available should be <= total");
        }
        // On unsupported platforms, should return (0, 0)
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            assert_eq!(total, 0);
            assert_eq!(available, 0);
        }
    }
}
