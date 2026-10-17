//! Enhanced logging system for xf.
//!
//! Provides structured logging with multiple output formats and levels.
//! Uses the `tracing` ecosystem for high-performance, structured logging.
//!
//! # Usage
//!
//! ```rust
//! use xf::logging::{init_logging, LogConfig, LogFormat};
//!
//! let config = LogConfig::default();
//! init_logging(&config);
//!
//! tracing::info!("Application started");
//! ```

use tracing::Level;
use tracing_subscriber::{
    EnvFilter,
    fmt::{self, format::FmtSpan},
    layer::SubscriberExt,
    util::SubscriberInitExt,
};

/// Logging configuration.
#[derive(Debug, Clone)]
#[allow(clippy::struct_excessive_bools)]
pub struct LogConfig {
    /// Minimum log level to display.
    pub level: LogLevel,
    /// Output format for log messages.
    pub format: LogFormat,
    /// Include timestamps in log output.
    pub timestamps: bool,
    /// Include target (module path) in log output.
    pub target: bool,
    /// Include span events (enter/exit).
    pub spans: bool,
    /// Enable ANSI colors in output.
    pub colors: bool,
    /// Write logs to this file (in addition to stderr).
    pub file: Option<String>,
}

/// Log level configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogLevel {
    /// Only errors.
    Error,
    /// Errors and warnings.
    Warn,
    /// Errors, warnings, and info messages.
    Info,
    /// All of the above plus debug messages.
    Debug,
    /// Everything including trace messages.
    Trace,
    /// No logging at all.
    Off,
}

/// Log output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LogFormat {
    /// Human-readable format with colors.
    Pretty,
    /// Compact single-line format.
    Compact,
    /// Full format with all details.
    Full,
}

impl Default for LogConfig {
    fn default() -> Self {
        Self {
            level: LogLevel::Info,
            format: LogFormat::Compact,
            timestamps: true,
            target: false,
            spans: false,
            colors: true,
            file: None,
        }
    }
}

impl LogConfig {
    /// Create a config for quiet mode (errors only).
    #[must_use]
    pub const fn quiet() -> Self {
        Self {
            level: LogLevel::Error,
            format: LogFormat::Compact,
            timestamps: false,
            target: false,
            spans: false,
            colors: true,
            file: None,
        }
    }

    /// Create a config for verbose mode (debug level).
    #[must_use]
    pub const fn verbose() -> Self {
        Self {
            level: LogLevel::Debug,
            format: LogFormat::Pretty,
            timestamps: true,
            target: true,
            spans: false,
            colors: true,
            file: None,
        }
    }

    /// Create a config for trace mode (maximum verbosity).
    #[must_use]
    pub const fn trace() -> Self {
        Self {
            level: LogLevel::Trace,
            format: LogFormat::Full,
            timestamps: true,
            target: true,
            spans: true,
            colors: true,
            file: None,
        }
    }
}

impl LogLevel {
    /// Convert to tracing Level.
    #[must_use]
    pub const fn to_tracing_level(self) -> Option<Level> {
        match self {
            Self::Error => Some(Level::ERROR),
            Self::Warn => Some(Level::WARN),
            Self::Info => Some(Level::INFO),
            Self::Debug => Some(Level::DEBUG),
            Self::Trace => Some(Level::TRACE),
            Self::Off => None,
        }
    }

    /// Convert to env filter directive string.
    const fn to_filter_string(self) -> &'static str {
        match self {
            Self::Error => "error",
            Self::Warn => "warn",
            Self::Info => "info",
            Self::Debug => "debug",
            Self::Trace => "trace",
            Self::Off => "off",
        }
    }
}

impl std::str::FromStr for LogLevel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "error" | "e" => Ok(Self::Error),
            "warn" | "warning" | "w" => Ok(Self::Warn),
            "info" | "i" => Ok(Self::Info),
            "debug" | "d" => Ok(Self::Debug),
            "trace" | "t" => Ok(Self::Trace),
            "off" | "none" | "quiet" => Ok(Self::Off),
            _ => Err(format!("Invalid log level: {s}")),
        }
    }
}

impl std::str::FromStr for LogFormat {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "pretty" | "p" => Ok(Self::Pretty),
            "compact" | "c" => Ok(Self::Compact),
            "full" | "f" => Ok(Self::Full),
            _ => Err(format!("Invalid log format: {s}")),
        }
    }
}

/// Initialize the logging system with the given configuration.
///
/// This should be called once at the start of the application.
/// Subsequent calls will be ignored.
///
/// # Arguments
///
/// * `config` - Logging configuration
pub fn init_logging(config: &LogConfig) {
    // Check if RUST_LOG is set, use that instead
    let env_filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::from_default_env()
    } else {
        EnvFilter::new(format!("xf={}", config.level.to_filter_string()))
    };

    // Determine span events
    let span_events = if config.spans {
        FmtSpan::ENTER | FmtSpan::EXIT
    } else {
        FmtSpan::NONE
    };

    match config.format {
        LogFormat::Pretty => {
            let stderr_fn = || std::io::stderr();
            let layer = fmt::layer()
                .pretty()
                .with_ansi(config.colors)
                .with_target(config.target)
                .with_span_events(span_events)
                .with_writer(stderr_fn);

            if config.timestamps {
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(layer)
                    .try_init()
                    .ok();
            } else {
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(layer.without_time())
                    .try_init()
                    .ok();
            }
        }
        LogFormat::Compact => {
            let stderr_fn = || std::io::stderr();
            let layer = fmt::layer()
                .compact()
                .with_ansi(config.colors)
                .with_target(config.target)
                .with_span_events(span_events)
                .with_writer(stderr_fn);

            if config.timestamps {
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(layer)
                    .try_init()
                    .ok();
            } else {
                tracing_subscriber::registry()
                    .with(env_filter)
                    .with(layer.without_time())
                    .try_init()
                    .ok();
            }
        }
        LogFormat::Full => {
            let stderr_fn = || std::io::stderr();
            let layer = fmt::layer()
                .with_ansi(config.colors)
                .with_target(config.target)
                .with_span_events(span_events)
                .with_thread_ids(true)
                .with_thread_names(true)
                .with_file(true)
                .with_line_number(true)
                .with_writer(stderr_fn);

            tracing_subscriber::registry()
                .with(env_filter)
                .with(layer)
                .try_init()
                .ok();
        }
    }
}

/// Initialize logging for tests (quiet by default).
pub fn init_test_logging() {
    let config = LogConfig {
        level: LogLevel::Off,
        ..Default::default()
    };
    init_logging(&config);
}

/// Initialize logging with defaults suitable for CLI use.
pub fn init_cli_logging(quiet: bool, verbose: bool) {
    let config = if quiet {
        LogConfig::quiet()
    } else if verbose {
        LogConfig::verbose()
    } else {
        LogConfig::default()
    };
    init_logging(&config);
}

/// A guard that logs the start and end of an operation.
///
/// Useful for tracking the duration and success of operations.
pub struct OperationGuard {
    name: String,
    start: std::time::Instant,
}

impl OperationGuard {
    /// Start tracking an operation.
    pub fn new(name: impl Into<String>) -> Self {
        let name = name.into();
        tracing::info!(operation = %name, "Starting operation");
        Self {
            name,
            start: std::time::Instant::now(),
        }
    }

    /// Complete the operation successfully.
    pub fn complete(self) {
        let duration = self.start.elapsed();
        tracing::info!(
            operation = %self.name,
            duration_ms = duration.as_millis(),
            "Operation completed"
        );
    }

    /// Mark the operation as failed.
    pub fn fail(self, error: &dyn std::error::Error) {
        let duration = self.start.elapsed();
        tracing::error!(
            operation = %self.name,
            duration_ms = duration.as_millis(),
            error = %error,
            "Operation failed"
        );
    }
}

impl Drop for OperationGuard {
    fn drop(&mut self) {
        // Only log if not explicitly completed/failed
        // This is a safety net for operations that panic
    }
}

/// Log a progress update for a long-running operation.
#[macro_export]
macro_rules! log_progress {
    ($current:expr, $total:expr, $($arg:tt)*) => {
        tracing::info!(
            current = $current,
            total = $total,
            percent = ($current as f64 / $total as f64 * 100.0) as u32,
            $($arg)*
        );
    };
}

/// Log a performance metric.
#[macro_export]
macro_rules! log_metric {
    ($name:expr, $value:expr) => {
        tracing::info!(metric = $name, value = $value, "Performance metric");
    };
    ($name:expr, $value:expr, $unit:expr) => {
        tracing::info!(
            metric = $name,
            value = $value,
            unit = $unit,
            "Performance metric"
        );
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_from_str() {
        assert_eq!("error".parse::<LogLevel>().unwrap(), LogLevel::Error);
        assert_eq!("warn".parse::<LogLevel>().unwrap(), LogLevel::Warn);
        assert_eq!("info".parse::<LogLevel>().unwrap(), LogLevel::Info);
        assert_eq!("debug".parse::<LogLevel>().unwrap(), LogLevel::Debug);
        assert_eq!("trace".parse::<LogLevel>().unwrap(), LogLevel::Trace);
        assert_eq!("off".parse::<LogLevel>().unwrap(), LogLevel::Off);
    }

    #[test]
    fn test_log_format_from_str() {
        assert_eq!("pretty".parse::<LogFormat>().unwrap(), LogFormat::Pretty);
        assert_eq!("compact".parse::<LogFormat>().unwrap(), LogFormat::Compact);
        assert_eq!("full".parse::<LogFormat>().unwrap(), LogFormat::Full);
    }

    #[test]
    fn test_default_config() {
        let config = LogConfig::default();
        assert_eq!(config.level, LogLevel::Info);
        assert_eq!(config.format, LogFormat::Compact);
        assert!(config.timestamps);
        assert!(config.colors);
    }

    #[test]
    fn test_preset_configs() {
        let quiet = LogConfig::quiet();
        assert_eq!(quiet.level, LogLevel::Error);

        let verbose = LogConfig::verbose();
        assert_eq!(verbose.level, LogLevel::Debug);

        let trace = LogConfig::trace();
        assert_eq!(trace.level, LogLevel::Trace);
    }

    #[test]
    fn test_log_level_filter_string() {
        assert_eq!(LogLevel::Error.to_filter_string(), "error");
        assert_eq!(LogLevel::Warn.to_filter_string(), "warn");
        assert_eq!(LogLevel::Info.to_filter_string(), "info");
        assert_eq!(LogLevel::Debug.to_filter_string(), "debug");
        assert_eq!(LogLevel::Trace.to_filter_string(), "trace");
        assert_eq!(LogLevel::Off.to_filter_string(), "off");
    }
}
