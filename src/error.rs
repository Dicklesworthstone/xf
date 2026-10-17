//! Custom error types for xf.
//!
//! Provides structured error handling with detailed context for better
//! diagnostics and user experience.

use std::path::PathBuf;
use thiserror::Error;

/// Primary error type for xf operations.
///
/// Each variant provides specific context about what went wrong,
/// enabling better error messages and programmatic error handling.
#[derive(Error, Debug)]
pub enum XfError {
    // =========================================================================
    // Archive Errors
    // =========================================================================
    /// Archive directory not found at the specified path.
    #[error("Archive not found at '{path}'")]
    ArchiveNotFound { path: PathBuf },

    /// Archive exists but is missing the expected structure.
    #[error("Invalid archive structure: {reason}")]
    InvalidArchive { reason: String },

    /// Required file missing from archive.
    #[error("Missing required file in archive: {file}")]
    MissingArchiveFile { file: String },

    /// Failed to parse archive data file.
    #[error("Failed to parse '{file}': {reason}")]
    ParseError { file: String, reason: String },

    /// Archive manifest is invalid or corrupt.
    #[error("Invalid manifest: {reason}")]
    InvalidManifest { reason: String },

    // =========================================================================
    // Database Errors
    // =========================================================================
    /// Database file not found (not yet indexed).
    #[error(
        "No indexed archive found. Run 'xf index <archive_path>' first.\nExpected database at: {path}"
    )]
    DatabaseNotFound { path: PathBuf },

    /// Database schema version mismatch.
    #[error(
        "Database schema version mismatch: expected {expected}, found {found}. Consider re-indexing with --force."
    )]
    SchemaMismatch { expected: i32, found: i32 },

    /// Database is locked by another process.
    #[error(
        "Database is locked. Ensure no other xf processes are running.\nIf the problem persists, remove the lock files:\n  rm {path}-wal {path}-shm"
    )]
    DatabaseLocked { path: PathBuf },

    /// Database operation failed.
    #[error("Database error: {0}")]
    DatabaseError(#[from] rusqlite::Error),

    // =========================================================================
    // Search Index Errors
    // =========================================================================
    /// Search index not found.
    #[error("Search index not found at '{path}'. Re-run indexing.")]
    IndexNotFound { path: PathBuf },

    /// Search index is corrupted.
    #[error("Search index corrupted: {reason}. Re-index with 'xf index --force'.")]
    IndexCorrupted { reason: String },

    /// Search query parsing failed.
    #[error("Invalid search query: {reason}")]
    InvalidQuery { reason: String },

    /// Search operation failed.
    #[error("Search error: {0}")]
    SearchError(String),

    /// Tantivy-specific error.
    #[error("Search engine error: {0}")]
    TantivyError(#[from] tantivy::TantivyError),

    // =========================================================================
    // IO Errors
    // =========================================================================
    /// File read/write error.
    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    /// Path-specific IO error with context.
    #[error("Failed to {operation} '{path}': {source}")]
    PathError {
        operation: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    // =========================================================================
    // Configuration Errors
    // =========================================================================
    /// Configuration file parsing error.
    #[error("Invalid configuration in '{path}': {reason}")]
    ConfigError { path: PathBuf, reason: String },

    /// Environment variable error.
    #[error("Invalid environment variable {var}: {reason}")]
    EnvVarError { var: String, reason: String },

    // =========================================================================
    // Data Validation Errors
    // =========================================================================
    /// Invalid date format in archive data.
    #[error("Invalid date format '{value}' in {context}")]
    InvalidDate { value: String, context: String },

    /// Invalid tweet ID.
    #[error("Invalid tweet ID: {id}")]
    InvalidTweetId { id: String },

    /// Data not found.
    #[error("{item_type} with ID '{id}' not found")]
    NotFound { item_type: &'static str, id: String },

    // =========================================================================
    // CLI Errors
    // =========================================================================
    /// Invalid command-line argument.
    #[error("Invalid argument: {reason}")]
    InvalidArgument { reason: String },

    /// Unsupported operation.
    #[error("{operation} is not yet implemented")]
    NotImplemented { operation: String },

    // =========================================================================
    // Generic Errors
    // =========================================================================
    /// Catch-all for other errors with context.
    #[error("{context}: {source}")]
    WithContext {
        context: String,
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Wrapped anyhow error for gradual migration.
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

/// Result type alias for xf operations.
pub type Result<T> = std::result::Result<T, XfError>;

impl XfError {
    /// Create an archive not found error.
    pub fn archive_not_found(path: impl Into<PathBuf>) -> Self {
        Self::ArchiveNotFound { path: path.into() }
    }

    /// Create an invalid archive error.
    pub fn invalid_archive(reason: impl Into<String>) -> Self {
        Self::InvalidArchive {
            reason: reason.into(),
        }
    }

    /// Create a parse error.
    pub fn parse_error(file: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::ParseError {
            file: file.into(),
            reason: reason.into(),
        }
    }

    /// Create a database not found error.
    pub fn database_not_found(path: impl Into<PathBuf>) -> Self {
        Self::DatabaseNotFound { path: path.into() }
    }

    /// Create an index not found error.
    pub fn index_not_found(path: impl Into<PathBuf>) -> Self {
        Self::IndexNotFound { path: path.into() }
    }

    /// Create an invalid query error.
    pub fn invalid_query(reason: impl Into<String>) -> Self {
        Self::InvalidQuery {
            reason: reason.into(),
        }
    }

    /// Create a not found error.
    pub fn not_found(item_type: &'static str, id: impl Into<String>) -> Self {
        Self::NotFound {
            item_type,
            id: id.into(),
        }
    }

    /// Create a path error with context.
    pub fn path_error(
        operation: &'static str,
        path: impl Into<PathBuf>,
        source: std::io::Error,
    ) -> Self {
        Self::PathError {
            operation,
            path: path.into(),
            source,
        }
    }

    /// Wrap an error with additional context.
    pub fn with_context<E>(context: impl Into<String>, source: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::WithContext {
            context: context.into(),
            source: Box::new(source),
        }
    }

    /// Check if this error is recoverable (user can fix it).
    #[must_use]
    pub const fn is_recoverable(&self) -> bool {
        matches!(
            self,
            Self::ArchiveNotFound { .. }
                | Self::DatabaseNotFound { .. }
                | Self::IndexNotFound { .. }
                | Self::InvalidQuery { .. }
                | Self::InvalidArgument { .. }
                | Self::NotFound { .. }
        )
    }

    /// Check if this error suggests re-indexing.
    #[must_use]
    pub const fn suggests_reindex(&self) -> bool {
        matches!(
            self,
            Self::SchemaMismatch { .. } | Self::IndexCorrupted { .. } | Self::IndexNotFound { .. }
        )
    }

    /// Get a suggestion for how to fix this error, if applicable.
    #[must_use]
    pub const fn suggestion(&self) -> Option<&'static str> {
        match self {
            Self::ArchiveNotFound { .. } => {
                Some("Verify the archive path and ensure the X data export is extracted.")
            }
            Self::DatabaseNotFound { .. } => {
                Some("Run 'xf index <archive_path>' to create the database.")
            }
            Self::IndexNotFound { .. } | Self::IndexCorrupted { .. } => {
                Some("Run 'xf index --force <archive_path>' to rebuild the index.")
            }
            Self::SchemaMismatch { .. } => {
                Some("Run 'xf index --force <archive_path>' to upgrade the schema.")
            }
            Self::DatabaseLocked { .. } => {
                Some("Close other xf instances or remove stale lock files.")
            }
            Self::InvalidQuery { .. } => Some(
                "Check query syntax. Use quotes for phrases, AND/OR for boolean, * for wildcards.",
            ),
            _ => None,
        }
    }
}

/// Extension trait for adding context to Results.
pub trait ResultExt<T> {
    /// Add context to an error.
    ///
    /// # Errors
    ///
    /// Returns the original error wrapped with additional context.
    fn context(self, context: impl Into<String>) -> Result<T>;

    /// Add context lazily (only evaluated on error).
    ///
    /// # Errors
    ///
    /// Returns the original error wrapped with additional context.
    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String;
}

impl<T, E> ResultExt<T> for std::result::Result<T, E>
where
    E: std::error::Error + Send + Sync + 'static,
{
    fn context(self, context: impl Into<String>) -> Result<T> {
        self.map_err(|e| XfError::with_context(context, e))
    }

    fn with_context<F>(self, f: F) -> Result<T>
    where
        F: FnOnce() -> String,
    {
        self.map_err(|e| XfError::with_context(f(), e))
    }
}

// =============================================================================
// CLI Error Formatting Utilities
// =============================================================================

use colored::Colorize;

/// Format a structured CLI error with explanation and suggestions.
///
/// # Arguments
/// * `title` - Brief error title (e.g., "Conflicting options")
/// * `explanation` - What went wrong and why
/// * `suggestions` - List of actionable suggestions
///
/// # Returns
/// A formatted error string ready for display.
#[must_use]
pub fn format_error(title: &str, explanation: &str, suggestions: &[&str]) -> String {
    use std::fmt::Write;

    let mut output = format!("{} {}", "✗".red().bold(), title.bold());

    if !explanation.is_empty() {
        let _ = write!(output, "\n\n   {explanation}");
    }

    if !suggestions.is_empty() {
        output.push_str("\n\n   ");
        if suggestions.len() == 1 {
            let _ = write!(output, "{} {}", "Hint:".cyan(), suggestions[0]);
        } else {
            let _ = write!(output, "{}:", "Try".cyan());
            for suggestion in suggestions {
                let _ = write!(output, "\n     {} {}", "•".dimmed(), suggestion);
            }
        }
    }

    output
}

/// Calculate the Levenshtein edit distance between two strings.
///
/// This is used for "did you mean?" suggestions when users make typos.
#[must_use]
pub fn levenshtein_distance(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 {
        return b_len;
    }
    if b_len == 0 {
        return a_len;
    }

    // Use two rows instead of full matrix for space efficiency
    let mut prev_row: Vec<usize> = (0..=b_len).collect();
    let mut curr_row: Vec<usize> = vec![0; b_len + 1];

    for (i, a_char) in a_chars.iter().enumerate() {
        curr_row[0] = i + 1;

        for (j, b_char) in b_chars.iter().enumerate() {
            let cost = usize::from(a_char != b_char);
            curr_row[j + 1] = (prev_row[j + 1] + 1)
                .min(curr_row[j] + 1)
                .min(prev_row[j] + cost);
        }

        std::mem::swap(&mut prev_row, &mut curr_row);
    }

    prev_row[b_len]
}

/// Find the best match from a list of candidates for a given input.
///
/// Returns `Some(match)` if a sufficiently close match is found,
/// `None` otherwise.
///
/// # Arguments
/// * `input` - The user's input (possibly a typo)
/// * `candidates` - List of valid options
/// * `max_distance` - Maximum edit distance to consider (default: 2)
#[must_use]
pub fn find_closest_match<'a>(
    input: &str,
    candidates: &[&'a str],
    max_distance: Option<usize>,
) -> Option<&'a str> {
    let max_dist = max_distance.unwrap_or(2);
    let input_lower = input.to_lowercase();

    candidates
        .iter()
        .map(|&candidate| {
            let candidate_lower = candidate.to_lowercase();
            let distance = levenshtein_distance(&input_lower, &candidate_lower);
            (candidate, distance)
        })
        .filter(|(_, distance)| *distance <= max_dist && *distance > 0)
        .min_by_key(|(_, distance)| *distance)
        .map(|(candidate, _)| candidate)
}

/// Format a "did you mean?" suggestion.
#[must_use]
pub fn format_did_you_mean(suggestion: &str) -> String {
    format!("Did you mean '{}'?", suggestion.green())
}

/// Format an error for an unknown value with "did you mean?" support.
///
/// # Arguments
/// * `kind` - The kind of value (e.g., "type", "field", "command")
/// * `input` - The user's input
/// * `valid_options` - List of valid options
///
/// # Returns
/// A formatted error string with suggestions if available.
pub fn format_unknown_value_error(kind: &str, input: &str, valid_options: &[&str]) -> String {
    let title = format!("Unknown {kind}: '{input}'");

    let mut suggestions = Vec::new();

    // Check for close matches
    if let Some(closest) = find_closest_match(input, valid_options, None) {
        suggestions.push(format_did_you_mean(closest));
    }

    // Show valid options if list is short
    if valid_options.len() <= 8 {
        suggestions.push(format!("Valid {kind}s: {}", valid_options.join(", ")));
    }

    let suggestion_refs: Vec<&str> = suggestions.iter().map(String::as_str).collect();
    format_error(&title, "", &suggestion_refs)
}

/// Standard valid data types for type filtering.
pub const VALID_DATA_TYPES: &[&str] = &["tweet", "like", "dm", "grok", "bookmark"];

/// Standard valid output fields for --fields.
pub const VALID_OUTPUT_FIELDS: &[&str] = &[
    "result_type",
    "id",
    "text",
    "created_at",
    "score",
    "highlights",
    "metadata",
    "rerank_score",
];

/// Standard valid config keys.
pub const VALID_CONFIG_KEYS: &[&str] = &[
    "db",
    "paths.db",
    "index",
    "paths.index",
    "archive",
    "paths.archive",
    "search.default_limit",
    "search.highlight",
    "search.fuzzy",
    "search.min_score",
    "search.cache_size",
    "indexing.parallel",
    "indexing.buffer_size_mb",
    "indexing.threads",
    "indexing.skip_types",
    "output.format",
    "output.colors",
    "output.quiet",
    "output.timings",
    "semantic.model",
    "semantic.dimensions",
    "semantic.reranker",
    "semantic.rerank",
    "semantic.daemon",
    "semantic.rerank_top",
    "semantic.two_tier.fast_model",
    "semantic.two_tier.quality_model",
    "semantic.two_tier.blend_factor",
    "semantic.two_tier.quality_timeout_ms",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        let err = XfError::archive_not_found("/path/to/archive");
        assert!(err.to_string().contains("/path/to/archive"));
    }

    #[test]
    fn test_error_suggestions() {
        let err = XfError::database_not_found("/path/to/db");
        assert!(err.suggestion().is_some());
        assert!(err.is_recoverable());
    }

    #[test]
    fn test_reindex_suggestion() {
        let err = XfError::IndexCorrupted {
            reason: "checksum mismatch".to_string(),
        };
        assert!(err.suggests_reindex());
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let xf_err: XfError = io_err.into();
        assert!(matches!(xf_err, XfError::IoError(_)));
    }

    #[test]
    fn test_from_rusqlite_error() {
        // This test verifies the From impl exists
        fn accepts_xf_error(_: XfError) {}
        let sqlite_err = rusqlite::Error::InvalidQuery;
        accepts_xf_error(sqlite_err.into());
    }

    // =========================================================================
    // Levenshtein Distance Tests
    // =========================================================================

    #[test]
    fn levenshtein_identical_strings() {
        assert_eq!(levenshtein_distance("hello", "hello"), 0);
    }

    #[test]
    fn levenshtein_one_char_difference() {
        assert_eq!(levenshtein_distance("hello", "hallo"), 1);
        assert_eq!(levenshtein_distance("cat", "car"), 1);
    }

    #[test]
    fn levenshtein_insertions_deletions() {
        assert_eq!(levenshtein_distance("cat", "cats"), 1);
        assert_eq!(levenshtein_distance("cats", "cat"), 1);
    }

    #[test]
    fn levenshtein_empty_strings() {
        assert_eq!(levenshtein_distance("", "hello"), 5);
        assert_eq!(levenshtein_distance("hello", ""), 5);
        assert_eq!(levenshtein_distance("", ""), 0);
    }

    #[test]
    fn find_closest_match_typo() {
        let candidates = ["tweet", "like", "dm", "grok"];
        assert_eq!(find_closest_match("twet", &candidates, None), Some("tweet"));
        assert_eq!(find_closest_match("dm", &candidates, None), None); // exact match not returned
        assert_eq!(find_closest_match("xyz", &candidates, None), None);
    }

    #[test]
    fn find_closest_match_case_insensitive() {
        let candidates = ["Tweet", "Like", "DM", "Grok"];
        assert_eq!(find_closest_match("TWET", &candidates, None), Some("Tweet"));
        assert_eq!(find_closest_match("gork", &candidates, None), Some("Grok"));
    }

    #[test]
    fn format_error_single_suggestion() {
        let output = format_error("Test Error", "Something went wrong", &["Try this"]);
        assert!(output.contains("Test Error"));
        assert!(output.contains("Something went wrong"));
        assert!(output.contains("Try this"));
    }

    #[test]
    fn format_error_multiple_suggestions() {
        let output = format_error(
            "Test Error",
            "Something went wrong",
            &["First option", "Second option"],
        );
        assert!(output.contains("First option"));
        assert!(output.contains("Second option"));
    }

    #[test]
    fn format_unknown_value_with_suggestion() {
        let output = format_unknown_value_error("type", "twet", VALID_DATA_TYPES);
        assert!(output.contains("Unknown type"));
        assert!(output.contains("twet"));
        assert!(output.contains("tweet")); // did you mean
    }
}
