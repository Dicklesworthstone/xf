//! Health check types for `xf doctor`.
//!
//! This module defines common structures used by archive, database, and index
//! diagnostics. Individual checks live in their respective modules.

use serde::Serialize;

/// High-level category for a health check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckCategory {
    Archive,
    Database,
    Index,
    Performance,
}

/// Status for an individual health check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CheckStatus {
    Pass,
    Warning,
    Error,
}

impl CheckStatus {
    /// Whether the check is healthy enough for continued operation.
    #[must_use]
    pub const fn is_ok(self) -> bool {
        matches!(self, Self::Pass)
    }
}

/// Single health check result.
#[derive(Debug, Clone, Serialize)]
pub struct HealthCheck {
    pub category: CheckCategory,
    pub name: String,
    pub status: CheckStatus,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub suggestion: Option<String>,
}

/// Table row counts (and optional size) for reporting.
#[derive(Debug, Clone, Serialize)]
pub struct TableStat {
    pub name: String,
    pub rows: i64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<i64>,
}

// ============================================================================
// Archive Structure Validation (xf-11.4.1)
// ============================================================================

use chrono::{Datelike, Utc};
use glob::glob;
use std::collections::HashSet;
use std::fs;
use std::path::Path;
use std::time::Instant;
use tracing::{debug, info, warn};

use crate::ArchiveParser;

/// File requirement specification.
struct FileRequirement {
    pattern: &'static str,
    required: bool,
    description: &'static str,
}

/// Expected files in an X archive.
const ARCHIVE_FILES: &[FileRequirement] = &[
    FileRequirement {
        pattern: "data/tweets.js",
        required: false, // Could be split into parts
        description: "Main tweets file",
    },
    FileRequirement {
        pattern: "data/tweets-part*.js",
        required: false, // Alternative to single file
        description: "Tweets parts",
    },
    FileRequirement {
        pattern: "data/direct-messages.js",
        required: false,
        description: "Direct messages",
    },
    FileRequirement {
        pattern: "data/direct-messages-group*.js",
        required: false,
        description: "Group DM parts",
    },
    FileRequirement {
        pattern: "data/like.js",
        required: false,
        description: "Likes/favorites",
    },
    FileRequirement {
        pattern: "data/likes.js",
        required: false,
        description: "Likes/favorites (alternate filename)",
    },
    FileRequirement {
        pattern: "data/follower.js",
        required: false,
        description: "Followers list",
    },
    FileRequirement {
        pattern: "data/following.js",
        required: false,
        description: "Following list",
    },
    FileRequirement {
        pattern: "data/block.js",
        required: false,
        description: "Blocked accounts",
    },
    FileRequirement {
        pattern: "data/mute.js",
        required: false,
        description: "Muted accounts",
    },
    FileRequirement {
        pattern: "data/grok-chat-item*.js",
        required: false,
        description: "Grok AI conversations",
    },
];

/// Check that required archive files are present.
///
/// # Errors
/// Returns error if glob pattern matching fails.
pub fn check_required_files(archive_path: &Path) -> crate::Result<Vec<HealthCheck>> {
    let mut checks = Vec::new();
    let mut has_tweets = false;

    for req in ARCHIVE_FILES {
        let req_path = Path::new(req.pattern);
        let req_dir = req_path.parent().unwrap_or_else(|| Path::new(""));
        let req_filename = req_path
            .file_name()
            .ok_or_else(|| {
                crate::XfError::invalid_archive(format!("Invalid pattern: {}", req.pattern))
            })?
            .to_string_lossy();

        let search_dir = archive_path.join(req_dir);
        let escaped_dir = glob::Pattern::escape(search_dir.to_string_lossy().as_ref());
        let full_pattern = format!(
            "{}{}{}",
            escaped_dir,
            std::path::MAIN_SEPARATOR,
            req_filename
        );

        debug!("Checking for pattern: {}", full_pattern);

        let matches: Vec<_> = glob(&full_pattern)
            .map_err(|e| crate::XfError::invalid_archive(format!("Invalid glob pattern: {e}")))?
            .filter_map(Result::ok)
            .collect();

        let exists = !matches.is_empty();

        if req.pattern.contains("tweets") && exists {
            has_tweets = true;
        }

        let status = if exists {
            CheckStatus::Pass
        } else if req.required {
            CheckStatus::Error
        } else {
            // Optional files that are missing are just info, not warnings
            continue; // Skip optional missing files from output
        };

        checks.push(HealthCheck {
            category: CheckCategory::Archive,
            name: format!("File: {} ({})", req.pattern, req.description),
            status,
            message: if exists {
                format!("Found {} file(s)", matches.len())
            } else {
                "Not found".into()
            },
            suggestion: if !exists && req.required {
                Some("Ensure archive was fully extracted".into())
            } else {
                None
            },
        });
    }

    // Special check: must have at least tweets.js or tweets-part*.js
    if !has_tweets {
        checks.push(HealthCheck {
            category: CheckCategory::Archive,
            name: "Tweets data".into(),
            status: CheckStatus::Error,
            message: "No tweets.js or tweets-part*.js found".into(),
            suggestion: Some(
                "Archive must contain tweet data. Check if archive was fully extracted.".into(),
            ),
        });
    }

    Ok(checks)
}

/// Validate JSON structure of archive files.
///
/// # Errors
/// Returns error if file reading or parsing fails.
pub fn check_json_structure(archive_path: &Path) -> crate::Result<Vec<HealthCheck>> {
    let mut checks = Vec::new();

    let files_to_check = [
        ("data/tweets.js", "Tweets"),
        ("data/like.js", "Likes"),
        ("data/likes.js", "Likes"),
        ("data/direct-messages.js", "DMs"),
        ("data/follower.js", "Followers"),
        ("data/following.js", "Following"),
    ];

    for (file, label) in files_to_check {
        let path = archive_path.join(file);
        if !path.exists() {
            continue;
        }

        debug!("Validating JSON structure: {}", file);

        match validate_js_wrapped_json(&path) {
            Ok((count, warnings)) => {
                let status = if warnings.is_empty() {
                    CheckStatus::Pass
                } else {
                    CheckStatus::Warning
                };
                checks.push(HealthCheck {
                    category: CheckCategory::Archive,
                    name: format!("Parse: {label}"),
                    status,
                    message: format!("{count} items parsed"),
                    suggestion: warnings.first().cloned(),
                });
            }
            Err(e) => {
                warn!("Parse error for {}: {}", file, e);
                checks.push(HealthCheck {
                    category: CheckCategory::Archive,
                    name: format!("Parse: {label}"),
                    status: CheckStatus::Error,
                    message: format!("Parse error: {e}"),
                    suggestion: Some("Check if file is corrupted or incomplete".into()),
                });
            }
        }
    }

    Ok(checks)
}

/// Validate a JavaScript-wrapped JSON file and return item count with warnings.
fn validate_js_wrapped_json(path: &Path) -> crate::Result<(usize, Vec<String>)> {
    let start = Instant::now();
    let content =
        fs::read_to_string(path).map_err(|e| crate::XfError::path_error("read", path, e))?;

    // Strip JS wrapper: window.YTD.tweets.part0 = [...]
    let mut parts = content.splitn(2, '=');
    let _prefix = parts.next().ok_or_else(|| {
        crate::XfError::parse_error(
            path.display().to_string(),
            "Invalid JS wrapper (missing prefix)".to_string(),
        )
    })?;
    let json_part = parts.next().ok_or_else(|| {
        crate::XfError::parse_error(
            path.display().to_string(),
            "Invalid JS wrapper (missing '=')".to_string(),
        )
    })?;
    let mut json_str = json_part.trim();
    if let Some(stripped) = json_str.strip_suffix(';') {
        json_str = stripped.trim_end();
    }

    // Parse as generic JSON array
    let items: Vec<serde_json::Value> = serde_json::from_str(json_str).map_err(|e| {
        crate::XfError::parse_error(path.display().to_string(), format!("Invalid JSON: {e}"))
    })?;

    let warnings = Vec::new();
    // Could add specific field validation here if needed

    if content.len() >= 5 * 1024 * 1024 {
        info!(
            "Parsed {} ({} bytes) in {}ms",
            path.display(),
            content.len(),
            start.elapsed().as_millis()
        );
    }

    Ok((items.len(), warnings))
}

/// Check for duplicate tweet IDs in the archive.
///
/// # Errors
/// Returns error if parsing fails.
pub fn check_duplicate_ids(archive_path: &Path) -> crate::Result<HealthCheck> {
    let parser = ArchiveParser::new(archive_path);
    let start = Instant::now();
    let tweets = parser.parse_tweets()?;
    if tweets.len() >= 100_000 {
        info!(
            "Parsed {} tweets for duplicate check in {}ms",
            tweets.len(),
            start.elapsed().as_millis()
        );
    }
    Ok(check_duplicate_ids_in_tweets(&tweets))
}

/// Check for duplicate tweet IDs in a pre-parsed tweet collection.
#[must_use]
pub fn check_duplicate_ids_in_tweets(tweets: &[crate::Tweet]) -> HealthCheck {
    let mut seen_ids: HashSet<String> = HashSet::new();
    let mut duplicates: Vec<String> = Vec::new();

    for tweet in tweets {
        if !seen_ids.insert(tweet.id.clone()) {
            duplicates.push(tweet.id.clone());
        }
    }

    HealthCheck {
        category: CheckCategory::Archive,
        name: "Duplicate Tweet IDs".into(),
        status: if duplicates.is_empty() {
            CheckStatus::Pass
        } else {
            CheckStatus::Warning
        },
        message: if duplicates.is_empty() {
            format!("{} unique tweet IDs", seen_ids.len())
        } else {
            format!("{} duplicate IDs found", duplicates.len())
        },
        suggestion: if duplicates.is_empty() {
            None
        } else {
            Some(format!(
                "Duplicate IDs: {}{}",
                duplicates[..3.min(duplicates.len())].join(", "),
                if duplicates.len() > 3 { "..." } else { "" }
            ))
        },
    }
}

/// Check timestamp consistency in tweets.
///
/// # Errors
/// Returns error if parsing fails.
pub fn check_timestamp_consistency(archive_path: &Path) -> crate::Result<HealthCheck> {
    let parser = ArchiveParser::new(archive_path);
    let start = Instant::now();
    let tweets = parser.parse_tweets()?;
    if tweets.len() >= 100_000 {
        info!(
            "Parsed {} tweets for timestamp check in {}ms",
            tweets.len(),
            start.elapsed().as_millis()
        );
    }
    Ok(check_timestamp_consistency_in_tweets(&tweets))
}

/// Check timestamp consistency in a pre-parsed tweet collection.
#[must_use]
pub fn check_timestamp_consistency_in_tweets(tweets: &[crate::Tweet]) -> HealthCheck {
    let mut issues: Vec<String> = Vec::new();
    let now = Utc::now();
    let twitter_launch_year = 2006;

    for tweet in tweets {
        // Check for future dates
        if tweet.created_at > now {
            issues.push(format!("{}: future date", tweet.id));
        }
        // Check for impossibly old dates (before Twitter existed)
        if tweet.created_at.year() < twitter_launch_year {
            issues.push(format!("{}: before {twitter_launch_year}", tweet.id));
        }
    }

    HealthCheck {
        category: CheckCategory::Archive,
        name: "Timestamp Validity".into(),
        status: if issues.is_empty() {
            CheckStatus::Pass
        } else {
            CheckStatus::Warning
        },
        message: if issues.is_empty() {
            format!("All {} timestamps valid", tweets.len())
        } else {
            format!("{} timestamp issues found", issues.len())
        },
        suggestion: if issues.is_empty() {
            None
        } else {
            Some(format!(
                "Issues: {}{}",
                issues[..3.min(issues.len())].join("; "),
                if issues.len() > 3 { "..." } else { "" }
            ))
        },
    }
}

/// Run all archive validation checks.
///
/// # Errors
/// Returns error if any check fails to execute.
pub fn validate_archive(archive_path: &Path) -> crate::Result<Vec<HealthCheck>> {
    let mut all_checks = Vec::new();

    // File presence checks
    all_checks.extend(check_required_files(archive_path)?);

    // JSON structure validation
    all_checks.extend(check_json_structure(archive_path)?);

    // Duplicate ID and timestamp checks (only if tweets exist)
    // Parse tweets ONCE and run both checks on the same data
    let tweets_path = archive_path.join("data/tweets.js");

    let data_dir = archive_path.join("data");
    let escaped_dir = glob::Pattern::escape(data_dir.to_string_lossy().as_ref());
    let pattern = format!(
        "{}{}{}",
        escaped_dir,
        std::path::MAIN_SEPARATOR,
        "tweets-part*.js"
    );

    let has_tweets = tweets_path.exists() || glob(&pattern).is_ok_and(|mut g| g.next().is_some());

    if has_tweets {
        let parser = ArchiveParser::new(archive_path);
        let start = Instant::now();
        let tweets = parser.parse_tweets()?;
        if tweets.len() >= 100_000 {
            info!(
                "Parsed {} tweets for validation checks in {}ms",
                tweets.len(),
                start.elapsed().as_millis()
            );
        }

        // Run both checks on the same parsed tweets (no double-parsing)
        all_checks.push(check_duplicate_ids_in_tweets(&tweets));
        all_checks.push(check_timestamp_consistency_in_tweets(&tweets));
    }

    Ok(all_checks)
}

// ============================================================================
// Performance Benchmarks (xf-11.4.4)
// ============================================================================

use crate::SearchEngine;

/// Performance thresholds for query latency (milliseconds).
mod thresholds {
    /// Acceptable: under 50ms
    pub const QUERY_ACCEPTABLE_MS: f64 = 50.0;
    /// Slow: over 100ms is a warning
    pub const QUERY_SLOW_MS: f64 = 100.0;

    /// Index load time thresholds
    pub const LOAD_ACCEPTABLE_MS: f64 = 500.0;
    pub const LOAD_SLOW_MS: f64 = 1000.0;

    /// Number of iterations for benchmark stability
    pub const BENCHMARK_ITERATIONS: usize = 10;
}

/// Latency statistics from a benchmark run.
#[derive(Debug, Clone, Serialize)]
pub struct LatencyStats {
    pub min_ms: f64,
    pub max_ms: f64,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub iterations: usize,
}

impl LatencyStats {
    /// Compute statistics from a vector of durations in milliseconds.
    fn from_durations(durations: &mut [f64]) -> Self {
        durations.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = durations.len();
        let sum: f64 = durations.iter().sum();
        let n_f64 = f64::from(u32::try_from(n).unwrap_or(u32::MAX));

        Self {
            min_ms: durations.first().copied().unwrap_or(0.0),
            max_ms: durations.last().copied().unwrap_or(0.0),
            mean_ms: if n > 0 { sum / n_f64 } else { 0.0 },
            p50_ms: percentile(durations, 50),
            p95_ms: percentile(durations, 95),
            p99_ms: percentile(durations, 99),
            iterations: n,
        }
    }

    /// Format as a concise string for health check messages.
    #[must_use]
    fn format_summary(&self) -> String {
        format!(
            "p50={:.1}ms, p95={:.1}ms, p99={:.1}ms (n={})",
            self.p50_ms, self.p95_ms, self.p99_ms, self.iterations
        )
    }
}

/// Calculate a percentile from a sorted slice.
fn percentile(sorted: &[f64], pct: usize) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (pct * sorted.len() / 100).min(sorted.len() - 1);
    sorted[idx]
}

/// Benchmark index load time.
///
/// Opens the index multiple times and measures load latency.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn benchmark_index_load(index_path: &Path) -> HealthCheck {
    let mut durations = Vec::with_capacity(3);

    // Fewer iterations for load test since it's more expensive
    for _ in 0..3 {
        let start = Instant::now();
        let result = SearchEngine::open(index_path);
        let elapsed_ms = start.elapsed().as_secs_f64() * 1000.0;

        match result {
            Ok(_) => durations.push(elapsed_ms),
            Err(e) => {
                return HealthCheck {
                    category: CheckCategory::Performance,
                    name: "Index Load Time".to_string(),
                    status: CheckStatus::Error,
                    message: format!("Failed to load index: {e}"),
                    suggestion: Some("Verify index exists and is not corrupted".to_string()),
                };
            }
        }
    }

    let latency_stats = LatencyStats::from_durations(&mut durations);
    let median = latency_stats.p50_ms;

    let (check_status, suggestion) = if median < thresholds::LOAD_ACCEPTABLE_MS {
        (CheckStatus::Pass, None)
    } else if median < thresholds::LOAD_SLOW_MS {
        (
            CheckStatus::Warning,
            Some("Index load is slow. Consider rebuilding with 'xf index --force'".to_string()),
        )
    } else {
        (
            CheckStatus::Warning,
            Some("Index load is very slow. Consider SSD storage or index rebuild".to_string()),
        )
    };

    HealthCheck {
        category: CheckCategory::Performance,
        name: "Index Load Time".to_string(),
        status: check_status,
        message: format!("{median:.0}ms"),
        suggestion,
    }
}

/// Benchmark simple single-word queries.
#[must_use]
pub fn benchmark_simple_query(engine: &SearchEngine) -> HealthCheck {
    let test_queries = ["the", "and", "test", "hello", "world"];
    let mut durations = Vec::with_capacity(thresholds::BENCHMARK_ITERATIONS * test_queries.len());

    for query in test_queries {
        for _ in 0..thresholds::BENCHMARK_ITERATIONS {
            let start = Instant::now();
            let _ = engine.search(query, None, 10);
            durations.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    let latency = LatencyStats::from_durations(&mut durations);
    let (check_status, suggestion) = evaluate_query_latency(&latency);

    HealthCheck {
        category: CheckCategory::Performance,
        name: "Simple Query Latency".to_string(),
        status: check_status,
        message: latency.format_summary(),
        suggestion,
    }
}

/// Benchmark phrase queries (multi-word, quoted).
#[must_use]
pub fn benchmark_phrase_query(engine: &SearchEngine) -> HealthCheck {
    let test_queries = ["\"hello world\"", "\"the quick\"", "\"test message\""];
    let mut durations = Vec::with_capacity(thresholds::BENCHMARK_ITERATIONS * test_queries.len());

    for query in test_queries {
        for _ in 0..thresholds::BENCHMARK_ITERATIONS {
            let start = Instant::now();
            let _ = engine.search(query, None, 10);
            durations.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    let latency = LatencyStats::from_durations(&mut durations);
    let (check_status, suggestion) = evaluate_query_latency(&latency);

    HealthCheck {
        category: CheckCategory::Performance,
        name: "Phrase Query Latency".to_string(),
        status: check_status,
        message: latency.format_summary(),
        suggestion,
    }
}

/// Benchmark complex boolean queries.
#[must_use]
pub fn benchmark_complex_query(engine: &SearchEngine) -> HealthCheck {
    let test_queries = [
        "hello AND world",
        "test OR example",
        "NOT spam",
        "(hello OR hi) AND world",
    ];
    let mut durations = Vec::with_capacity(thresholds::BENCHMARK_ITERATIONS * test_queries.len());

    for query in test_queries {
        for _ in 0..thresholds::BENCHMARK_ITERATIONS {
            let start = Instant::now();
            let _ = engine.search(query, None, 10);
            durations.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    let latency = LatencyStats::from_durations(&mut durations);
    let (check_status, suggestion) = evaluate_query_latency(&latency);

    HealthCheck {
        category: CheckCategory::Performance,
        name: "Complex Query Latency".to_string(),
        status: check_status,
        message: latency.format_summary(),
        suggestion,
    }
}

/// Benchmark FTS5 queries via `SQLite`.
#[must_use]
pub fn benchmark_fts5_query(storage: &crate::Storage) -> HealthCheck {
    let test_queries = ["the", "test", "hello", "and"];
    let mut durations = Vec::with_capacity(thresholds::BENCHMARK_ITERATIONS * test_queries.len());

    for query in test_queries {
        for _ in 0..thresholds::BENCHMARK_ITERATIONS {
            let start = Instant::now();
            // Use search_tweets which queries via FTS5
            let _ = storage.search_tweets(query, 10);
            durations.push(start.elapsed().as_secs_f64() * 1000.0);
        }
    }

    let latency = LatencyStats::from_durations(&mut durations);
    let (check_status, suggestion) = evaluate_query_latency(&latency);

    HealthCheck {
        category: CheckCategory::Performance,
        name: "FTS5 Query Latency".to_string(),
        status: check_status,
        message: latency.format_summary(),
        suggestion,
    }
}

/// Evaluate query latency against thresholds.
fn evaluate_query_latency(latency: &LatencyStats) -> (CheckStatus, Option<String>) {
    let p95 = latency.p95_ms;

    if p95 < thresholds::QUERY_ACCEPTABLE_MS {
        (CheckStatus::Pass, None)
    } else if p95 < thresholds::QUERY_SLOW_MS {
        (
            CheckStatus::Warning,
            Some(
                "Query latency is elevated. Consider rebuilding with 'xf index --force'"
                    .to_string(),
            ),
        )
    } else {
        (
            CheckStatus::Warning,
            Some("Query latency is high. Consider index optimization or SSD storage".to_string()),
        )
    }
}

/// Run all performance benchmarks.
///
/// Returns a vector of health checks covering index load time,
/// simple/phrase/complex query latencies, and FTS5 performance.
pub fn run_performance_benchmarks(
    index_path: &Path,
    engine: &SearchEngine,
    storage: &crate::Storage,
) -> Vec<HealthCheck> {
    let mut checks = Vec::with_capacity(5);

    info!("Running performance benchmarks...");

    // Index load time
    debug!("Benchmarking index load time");
    checks.push(benchmark_index_load(index_path));

    // Simple queries
    debug!("Benchmarking simple queries");
    checks.push(benchmark_simple_query(engine));

    // Phrase queries
    debug!("Benchmarking phrase queries");
    checks.push(benchmark_phrase_query(engine));

    // Complex boolean queries
    debug!("Benchmarking complex queries");
    checks.push(benchmark_complex_query(engine));

    // FTS5 queries
    debug!("Benchmarking FTS5 queries");
    checks.push(benchmark_fts5_query(storage));

    info!("Performance benchmarks complete: {} checks", checks.len());

    checks
}

// ============================================================================
// Styled Rendering (xf-11.4.5)
// ============================================================================

use crate::output::Output;
use crate::theme::Theme;

/// Summary of health check results for rendering and serialization.
#[derive(Debug, Clone, Serialize)]
pub struct DoctorSummary {
    pub passed: usize,
    pub warnings: usize,
    pub errors: usize,
    pub total: usize,
}

impl DoctorSummary {
    /// Compute summary from a slice of health checks.
    #[must_use]
    pub fn from_checks(checks: &[HealthCheck]) -> Self {
        let mut passed = 0;
        let mut warnings = 0;
        let mut errors = 0;
        for check in checks {
            match check.status {
                CheckStatus::Pass => passed += 1,
                CheckStatus::Warning => warnings += 1,
                CheckStatus::Error => errors += 1,
            }
        }
        Self {
            passed,
            warnings,
            errors,
            total: checks.len(),
        }
    }
}

/// Collects doctor data and renders styled output via the Output abstraction.
pub struct DoctorRenderer<'a> {
    checks: &'a [HealthCheck],
    summary: &'a DoctorSummary,
    suggestions: &'a [String],
    runtime_ms: u64,
    output: &'a Output,
    theme: &'a Theme,
}

impl<'a> DoctorRenderer<'a> {
    /// Create a new doctor renderer.
    #[must_use]
    pub const fn new(
        checks: &'a [HealthCheck],
        summary: &'a DoctorSummary,
        suggestions: &'a [String],
        runtime_ms: u64,
        output: &'a Output,
        theme: &'a Theme,
    ) -> Self {
        Self {
            checks,
            summary,
            suggestions,
            runtime_ms,
            output,
            theme,
        }
    }

    /// Render the full doctor report.
    pub fn render(&self) {
        if self.output.is_styled() {
            self.render_styled();
        } else {
            self.render_plain();
        }
    }

    /// Render styled output with markup and theme colors.
    fn render_styled(&self) {
        let width = self.output.width().min(80);

        // Title banner
        let rule = "\u{2550}".repeat(width);
        self.output
            .print(&format!("[{}]{rule}[/]", self.theme.primary));
        self.output.print(&format!(
            "[{} bold]  \u{1fa7a} XF HEALTH CHECK[/]",
            self.theme.heading
        ));
        self.output
            .print(&format!("[{}]{rule}[/]", self.theme.primary));
        self.output.print("");

        // Group by category
        let mut current_category: Option<CheckCategory> = None;
        for check in self.checks {
            if current_category != Some(check.category) {
                current_category = Some(check.category);
                let icon = category_icon(check.category);
                let name = category_name(check.category);
                self.output.print("");
                self.output
                    .print(&format!("[{} bold]{icon} {name}[/]", self.theme.info));
                let sep = "\u{2500}".repeat(width.min(60));
                self.output
                    .print(&format!("[{}]{sep}[/]", self.theme.border));
            }

            self.render_check_styled(check);
        }

        // Summary footer
        self.output.print("");
        self.output
            .print(&format!("[{}]{rule}[/]", self.theme.primary));
        self.output.print(&format!(
            "  [{}]{} \u{2713}[/] passed  [{}]{} \u{26a0}[/] warnings  [{}]{} \u{2717}[/] errors  [{}]({} total, {}ms)[/]",
            self.theme.success,
            self.summary.passed,
            self.theme.warning,
            self.summary.warnings,
            self.theme.error,
            self.summary.errors,
            self.theme.muted,
            self.summary.total,
            self.runtime_ms,
        ));

        // Suggestions
        if !self.suggestions.is_empty() {
            self.output.print("");
            self.output.print(&format!(
                "[{} bold]\u{1f4a1} Suggestions:[/]",
                self.theme.info
            ));
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                self.output.print(&format!(
                    "  [{}]{}.[/] {suggestion}",
                    self.theme.muted,
                    i + 1
                ));
            }
        }
    }

    /// Render a single check in styled mode.
    fn render_check_styled(&self, check: &HealthCheck) {
        let (icon, color) = status_icon_and_color(check.status, self.theme);
        self.output.print(&format!(
            "  [{color}]{icon}[/] {}: {}",
            check.name, check.message
        ));
    }

    /// Render plain text output for non-TTY contexts.
    fn render_plain(&self) {
        self.output.print("XF HEALTH CHECK");
        self.output.print(&"=".repeat(60));
        self.output.print("");

        let mut current_category: Option<CheckCategory> = None;
        for check in self.checks {
            if current_category != Some(check.category) {
                current_category = Some(check.category);
                self.output.print("");
                self.output.print(category_name(check.category));
                self.output.print(&"-".repeat(40));
            }

            let tag = status_tag(check.status);
            self.output
                .print(&format!("  {tag} {}: {}", check.name, check.message));
        }

        // Summary
        self.output.print("");
        self.output.print(&format!(
            "{} passed, {} warnings, {} errors ({} total, {}ms)",
            self.summary.passed,
            self.summary.warnings,
            self.summary.errors,
            self.summary.total,
            self.runtime_ms,
        ));

        if !self.suggestions.is_empty() {
            self.output.print("");
            self.output.print("Suggestions:");
            for suggestion in self.suggestions {
                self.output.print(&format!("  * {suggestion}"));
            }
        }
    }
}

/// Get display name for a check category.
#[must_use]
pub const fn category_name(cat: CheckCategory) -> &'static str {
    match cat {
        CheckCategory::Archive => "Archive",
        CheckCategory::Database => "Database",
        CheckCategory::Index => "Index",
        CheckCategory::Performance => "Performance",
    }
}

/// Get icon for a check category.
#[must_use]
pub const fn category_icon(cat: CheckCategory) -> &'static str {
    match cat {
        CheckCategory::Archive => "\u{1f4c1}",
        CheckCategory::Database => "\u{1f5c4}\u{fe0f}",
        CheckCategory::Index => "\u{1f50d}",
        CheckCategory::Performance => "\u{26a1}",
    }
}

/// Get status icon and theme color key for a check status.
#[must_use]
pub const fn status_icon_and_color(status: CheckStatus, theme: &Theme) -> (&'static str, &str) {
    match status {
        CheckStatus::Pass => ("\u{2713}", theme.success),
        CheckStatus::Warning => ("\u{26a0}", theme.warning),
        CheckStatus::Error => ("\u{2717}", theme.error),
    }
}

/// Get plain text status tag (for non-TTY output).
#[must_use]
pub const fn status_tag(status: CheckStatus) -> &'static str {
    match status {
        CheckStatus::Pass => "[OK]",
        CheckStatus::Warning => "[WARN]",
        CheckStatus::Error => "[FAIL]",
    }
}

// ============================================================================
// Tests (xf-11.4.6)
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use tempfile::TempDir;

    // ======================== Helper Functions ========================

    /// Create a minimal valid archive structure for testing.
    fn create_test_archive(tweets: &str) -> TempDir {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        // Write tweets.js with JS wrapper
        let tweets_content = format!("window.YTD.tweets.part0 = {tweets}");
        std::fs::write(data_dir.join("tweets.js"), tweets_content).unwrap();

        dir
    }

    /// Create a broken archive with missing required structure.
    fn create_broken_archive() -> TempDir {
        // No data directory at all
        TempDir::new().unwrap()
    }

    /// Create a minimal Tweet for testing.
    fn make_tweet(id: &str, text: &str, created_at: chrono::DateTime<Utc>) -> crate::Tweet {
        crate::Tweet {
            id: id.into(),
            created_at,
            full_text: text.into(),
            source: None,
            favorite_count: 0,
            retweet_count: 0,
            lang: None,
            in_reply_to_status_id: None,
            in_reply_to_user_id: None,
            in_reply_to_screen_name: None,
            is_retweet: false,
            hashtags: Vec::new(),
            user_mentions: Vec::new(),
            urls: Vec::new(),
            media: Vec::new(),
        }
    }

    // ======================== File Presence Tests ========================

    #[test]
    fn test_check_required_files_valid() {
        let archive = create_test_archive("[]");
        let checks = check_required_files(archive.path()).unwrap();

        // Should have at least one check
        assert!(!checks.is_empty());

        // No errors when tweets.js exists
        let errors: Vec<_> = checks
            .iter()
            .filter(|c| c.status == CheckStatus::Error)
            .collect();
        assert!(errors.is_empty(), "Expected no errors, got: {errors:?}");
    }

    #[test]
    fn test_check_required_files_missing_tweets() {
        let archive = create_broken_archive();
        std::fs::create_dir_all(archive.path().join("data")).unwrap();
        // data/ exists but no tweets.js

        let checks = check_required_files(archive.path()).unwrap();

        // Should have error for missing tweets
        let tweet_error = checks
            .iter()
            .find(|c| c.name == "Tweets data" && c.status == CheckStatus::Error);
        assert!(
            tweet_error.is_some(),
            "Expected error for missing tweets data"
        );
    }

    #[test]
    fn test_check_required_files_with_parts() {
        let dir = TempDir::new().unwrap();
        let data_dir = dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        // Create tweet parts instead of single file
        std::fs::write(
            data_dir.join("tweets-part1.js"),
            "window.YTD.tweets.part1 = []",
        )
        .unwrap();
        std::fs::write(
            data_dir.join("tweets-part2.js"),
            "window.YTD.tweets.part2 = []",
        )
        .unwrap();

        let checks = check_required_files(dir.path()).unwrap();

        // No error for tweets - parts count as valid
        let tweet_error = checks
            .iter()
            .find(|c| c.name == "Tweets data" && c.status == CheckStatus::Error);
        assert!(tweet_error.is_none(), "Should accept tweet parts");
    }

    // ======================== JSON Validation Tests ========================

    #[test]
    fn test_check_json_structure_valid() {
        let tweets = r#"[{"tweet": {"id": "123", "full_text": "Hello"}}]"#;
        let archive = create_test_archive(tweets);

        let checks = check_json_structure(archive.path()).unwrap();

        let parse_check = checks.iter().find(|c| c.name.contains("Parse: Tweets"));
        assert!(parse_check.is_some(), "Should have tweets parse check");
        assert_eq!(parse_check.unwrap().status, CheckStatus::Pass);
    }

    #[test]
    fn test_check_json_structure_invalid() {
        let archive = create_test_archive("not valid json at all");

        let checks = check_json_structure(archive.path()).unwrap();

        // Should have error for invalid JSON
        let error = checks
            .iter()
            .find(|c| c.name.contains("Parse") && c.status == CheckStatus::Error);
        assert!(error.is_some(), "Expected parse error for invalid JSON");
    }

    #[test]
    fn test_check_json_structure_empty_array() {
        let archive = create_test_archive("[]");

        let checks = check_json_structure(archive.path()).unwrap();

        // Empty array should still pass (0 items parsed)
        let parse_check = checks.iter().find(|c| c.name.contains("Parse: Tweets"));
        if let Some(check) = parse_check {
            assert_eq!(check.status, CheckStatus::Pass);
            assert!(check.message.contains("0 items"));
        }
    }

    // ======================== Duplicate ID Tests ========================

    #[test]
    fn test_check_duplicate_ids_none() {
        let tweets = vec![
            make_tweet("1", "Tweet 1", Utc::now()),
            make_tweet("2", "Tweet 2", Utc::now()),
        ];

        let check = check_duplicate_ids_in_tweets(&tweets);

        assert_eq!(check.status, CheckStatus::Pass);
        assert!(check.message.contains("2 unique"));
    }

    #[test]
    fn test_check_duplicate_ids_found() {
        let tweets = vec![
            make_tweet("1", "Tweet 1", Utc::now()),
            make_tweet("1", "Tweet 2 (duplicate ID)", Utc::now()),
        ];

        let check = check_duplicate_ids_in_tweets(&tweets);

        assert_eq!(check.status, CheckStatus::Warning);
        assert!(check.message.contains("duplicate"));
        assert!(check.suggestion.is_some());
    }

    // ======================== Timestamp Tests ========================

    #[test]
    fn test_check_timestamp_consistency_valid() {
        let valid_date = chrono::Utc.with_ymd_and_hms(2023, 6, 15, 12, 0, 0).unwrap();
        let tweets = vec![make_tweet("1", "Tweet", valid_date)];

        let check = check_timestamp_consistency_in_tweets(&tweets);

        assert_eq!(check.status, CheckStatus::Pass);
    }

    #[test]
    fn test_check_timestamp_consistency_future_date() {
        let future = Utc::now() + chrono::Duration::days(365);
        let tweets = vec![make_tweet("1", "Future tweet", future)];

        let check = check_timestamp_consistency_in_tweets(&tweets);

        assert_eq!(check.status, CheckStatus::Warning);
        assert!(check.message.contains("issue"));
    }

    #[test]
    fn test_check_timestamp_consistency_before_twitter() {
        let old = chrono::Utc.with_ymd_and_hms(2000, 1, 1, 0, 0, 0).unwrap();
        let tweets = vec![make_tweet("1", "Old tweet", old)];

        let check = check_timestamp_consistency_in_tweets(&tweets);

        assert_eq!(check.status, CheckStatus::Warning);
        assert!(check.message.contains("issue"));
    }

    // ======================== Latency Stats Tests ========================

    #[test]
    #[allow(clippy::float_cmp)] // Exact values expected in test
    fn test_latency_stats_from_durations() {
        let mut durations = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let stats = LatencyStats::from_durations(&mut durations);

        assert_eq!(stats.min_ms, 10.0);
        assert_eq!(stats.max_ms, 50.0);
        assert!((stats.mean_ms - 30.0).abs() < 0.01);
        assert_eq!(stats.iterations, 5);
    }

    #[test]
    #[allow(clippy::float_cmp)] // Exact values expected in test
    fn test_latency_stats_empty() {
        let mut durations: Vec<f64> = vec![];
        let stats = LatencyStats::from_durations(&mut durations);

        assert_eq!(stats.min_ms, 0.0);
        assert_eq!(stats.max_ms, 0.0);
        assert_eq!(stats.mean_ms, 0.0);
        assert_eq!(stats.iterations, 0);
    }

    #[test]
    fn test_latency_stats_format_summary() {
        let stats = LatencyStats {
            min_ms: 1.0,
            max_ms: 100.0,
            mean_ms: 25.0,
            p50_ms: 20.0,
            p95_ms: 80.0,
            p99_ms: 95.0,
            iterations: 100,
        };

        let summary = stats.format_summary();
        assert!(summary.contains("p50=20.0ms"));
        assert!(summary.contains("p95=80.0ms"));
        assert!(summary.contains("n=100"));
    }

    // ======================== Health Check Status Tests ========================

    #[test]
    fn test_check_status_is_ok() {
        assert!(CheckStatus::Pass.is_ok());
        assert!(!CheckStatus::Warning.is_ok());
        assert!(!CheckStatus::Error.is_ok());
    }

    // ======================== Full Validation Tests ========================

    #[test]
    fn test_validate_archive_healthy() {
        let tweets = r#"[{"tweet": {"id": "123", "full_text": "Hello", "created_at": "2023-01-01T12:00:00.000Z"}}]"#;
        let archive = create_test_archive(tweets);

        let checks = validate_archive(archive.path()).unwrap();

        // Should have multiple checks
        assert!(!checks.is_empty());

        // Most should pass for a healthy archive
        let pass_count = checks
            .iter()
            .filter(|c| c.status == CheckStatus::Pass)
            .count();
        assert!(pass_count > 0, "Healthy archive should have passing checks");
    }

    #[test]
    fn test_validate_archive_empty() {
        let archive = create_test_archive("[]");

        let checks = validate_archive(archive.path()).unwrap();

        // Empty but valid structure
        assert!(!checks.is_empty());
    }

    // ======================== Rendering Tests ========================

    #[test]
    fn test_doctor_summary_from_checks() {
        let checks = vec![
            HealthCheck {
                category: CheckCategory::Archive,
                name: "Test 1".into(),
                status: CheckStatus::Pass,
                message: "OK".into(),
                suggestion: None,
            },
            HealthCheck {
                category: CheckCategory::Database,
                name: "Test 2".into(),
                status: CheckStatus::Warning,
                message: "Warn".into(),
                suggestion: Some("Fix it".into()),
            },
            HealthCheck {
                category: CheckCategory::Index,
                name: "Test 3".into(),
                status: CheckStatus::Error,
                message: "Fail".into(),
                suggestion: Some("Rebuild".into()),
            },
        ];

        let summary = DoctorSummary::from_checks(&checks);
        assert_eq!(summary.passed, 1);
        assert_eq!(summary.warnings, 1);
        assert_eq!(summary.errors, 1);
        assert_eq!(summary.total, 3);
    }

    #[test]
    fn test_doctor_summary_all_passing() {
        let checks = vec![
            HealthCheck {
                category: CheckCategory::Archive,
                name: "A".into(),
                status: CheckStatus::Pass,
                message: "OK".into(),
                suggestion: None,
            },
            HealthCheck {
                category: CheckCategory::Database,
                name: "B".into(),
                status: CheckStatus::Pass,
                message: "OK".into(),
                suggestion: None,
            },
        ];

        let summary = DoctorSummary::from_checks(&checks);
        assert_eq!(summary.passed, 2);
        assert_eq!(summary.warnings, 0);
        assert_eq!(summary.errors, 0);
    }

    #[test]
    fn test_doctor_summary_empty() {
        let summary = DoctorSummary::from_checks(&[]);
        assert_eq!(summary.passed, 0);
        assert_eq!(summary.warnings, 0);
        assert_eq!(summary.errors, 0);
        assert_eq!(summary.total, 0);
    }

    #[test]
    fn test_category_name_all() {
        assert_eq!(category_name(CheckCategory::Archive), "Archive");
        assert_eq!(category_name(CheckCategory::Database), "Database");
        assert_eq!(category_name(CheckCategory::Index), "Index");
        assert_eq!(category_name(CheckCategory::Performance), "Performance");
    }

    #[test]
    fn test_category_icon_all() {
        // Each category should have a non-empty icon
        for cat in [
            CheckCategory::Archive,
            CheckCategory::Database,
            CheckCategory::Index,
            CheckCategory::Performance,
        ] {
            assert!(!category_icon(cat).is_empty());
        }
    }

    #[test]
    fn test_status_icon_and_color() {
        let theme = Theme::dark();

        let (icon, color) = status_icon_and_color(CheckStatus::Pass, &theme);
        assert_eq!(icon, "\u{2713}");
        assert_eq!(color, theme.success);

        let (icon, color) = status_icon_and_color(CheckStatus::Warning, &theme);
        assert_eq!(icon, "\u{26a0}");
        assert_eq!(color, theme.warning);

        let (icon, color) = status_icon_and_color(CheckStatus::Error, &theme);
        assert_eq!(icon, "\u{2717}");
        assert_eq!(color, theme.error);
    }

    #[test]
    fn test_status_tag() {
        assert_eq!(status_tag(CheckStatus::Pass), "[OK]");
        assert_eq!(status_tag(CheckStatus::Warning), "[WARN]");
        assert_eq!(status_tag(CheckStatus::Error), "[FAIL]");
    }

    #[test]
    fn test_renderer_plain_no_panic() {
        let checks = vec![
            HealthCheck {
                category: CheckCategory::Archive,
                name: "Test".into(),
                status: CheckStatus::Pass,
                message: "OK".into(),
                suggestion: None,
            },
            HealthCheck {
                category: CheckCategory::Database,
                name: "DB Test".into(),
                status: CheckStatus::Warning,
                message: "Slow".into(),
                suggestion: Some("Optimize".into()),
            },
        ];
        let summary = DoctorSummary::from_checks(&checks);
        let suggestions = vec!["Optimize".to_string()];
        let output = Output::new_piped();
        let theme = Theme::dark();

        let renderer = DoctorRenderer::new(&checks, &summary, &suggestions, 42, &output, &theme);
        // Should not panic
        renderer.render();
    }

    #[test]
    fn test_renderer_styled_no_panic() {
        let checks = vec![HealthCheck {
            category: CheckCategory::Performance,
            name: "Speed".into(),
            status: CheckStatus::Error,
            message: "Slow".into(),
            suggestion: Some("Upgrade".into()),
        }];
        let summary = DoctorSummary::from_checks(&checks);
        let suggestions = vec!["Upgrade".to_string()];
        let output = Output::new_tty();
        let theme = Theme::dark();

        let renderer = DoctorRenderer::new(&checks, &summary, &suggestions, 100, &output, &theme);
        // Should not panic
        renderer.render();
    }

    #[test]
    fn test_renderer_empty_checks() {
        let checks: Vec<HealthCheck> = vec![];
        let summary = DoctorSummary::from_checks(&checks);
        let suggestions: Vec<String> = vec![];
        let output = Output::new_piped();
        let theme = Theme::dark();

        let renderer = DoctorRenderer::new(&checks, &summary, &suggestions, 0, &output, &theme);
        // Should not panic on empty
        renderer.render();
    }
}
