//! Styled error and warning message panels.
//!
//! Provides structured error messaging with error codes, suggestions,
//! and documentation links. Supports both styled (TTY) and plain (piped) output.
//!
//! # Example
//!
//! ```rust,ignore
//! use xf::messages::{MessagePanel, ErrorCode};
//! use xf::output::Output;
//!
//! let output = Output::new_tty();
//! MessagePanel::error("Archive not found", "Could not locate archive")
//!     .with_code(ErrorCode::ArchiveNotFound)
//!     .with_suggestion("Check the path is correct")
//!     .render(&output);
//! ```

use crate::output::Output;
use crate::theme::Theme;
use std::path::Path;

/// Error codes for structured error reporting.
///
/// Error codes are organized by category:
/// - E1xx: Archive errors
/// - E2xx: Index errors
/// - E3xx: Search errors
/// - E4xx: Database errors
/// - E5xx: Config errors
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ErrorCode {
    // Archive errors (E1xx)
    /// Archive file or directory not found
    ArchiveNotFound = 100,
    /// Archive is corrupted or unreadable
    ArchiveCorrupt = 101,
    /// Insufficient permissions to access archive
    ArchivePermission = 102,
    /// Archive format is invalid
    ArchiveInvalid = 103,

    // Index errors (E2xx)
    /// Search index not found
    IndexNotFound = 200,
    /// Index is outdated relative to archive
    IndexStale = 201,
    /// Index is corrupted
    IndexCorrupt = 202,
    /// Index is locked by another process
    IndexLocked = 203,

    // Search errors (E3xx)
    /// No results found for query
    NoResults = 300,
    /// Query syntax is invalid
    QueryInvalid = 301,
    /// Query is too complex to process
    QueryTooComplex = 302,

    // Database errors (E4xx)
    /// Cannot connect to database
    DatabaseConnection = 400,
    /// Database schema mismatch
    DatabaseSchema = 401,
    /// Database is locked by another process
    DatabaseLocked = 402,

    // Config errors (E5xx)
    /// Config file not found
    ConfigNotFound = 500,
    /// Config file is invalid
    ConfigInvalid = 501,
}

impl ErrorCode {
    /// Get the error code as a string (e.g., "E100").
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::ArchiveNotFound => "E100",
            Self::ArchiveCorrupt => "E101",
            Self::ArchivePermission => "E102",
            Self::ArchiveInvalid => "E103",
            Self::IndexNotFound => "E200",
            Self::IndexStale => "E201",
            Self::IndexCorrupt => "E202",
            Self::IndexLocked => "E203",
            Self::NoResults => "E300",
            Self::QueryInvalid => "E301",
            Self::QueryTooComplex => "E302",
            Self::DatabaseConnection => "E400",
            Self::DatabaseSchema => "E401",
            Self::DatabaseLocked => "E402",
            Self::ConfigNotFound => "E500",
            Self::ConfigInvalid => "E501",
        }
    }

    /// Get the error category name.
    #[must_use]
    pub const fn category(&self) -> &'static str {
        match self {
            Self::ArchiveNotFound
            | Self::ArchiveCorrupt
            | Self::ArchivePermission
            | Self::ArchiveInvalid => "Archive",
            Self::IndexNotFound | Self::IndexStale | Self::IndexCorrupt | Self::IndexLocked => {
                "Index"
            }
            Self::NoResults | Self::QueryInvalid | Self::QueryTooComplex => "Search",
            Self::DatabaseConnection | Self::DatabaseSchema | Self::DatabaseLocked => "Database",
            Self::ConfigNotFound | Self::ConfigInvalid => "Config",
        }
    }

    /// Get the documentation URL for this error code.
    #[must_use]
    pub fn docs_url(&self) -> String {
        format!("https://xf.dev/errors/{}", self.as_str())
    }
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Message severity level.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MessageLevel {
    /// Critical error that stops execution
    Error,
    /// Warning that doesn't stop execution
    Warning,
    /// Informational message
    Info,
    /// Success message
    Success,
}

impl MessageLevel {
    /// Get the styled icon for this level.
    #[must_use]
    pub const fn icon(&self) -> &'static str {
        match self {
            Self::Error => "✗",
            Self::Warning => "⚠",
            Self::Info => "ℹ",
            Self::Success => "✓",
        }
    }

    /// Get the label for this level.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Error => "ERROR",
            Self::Warning => "WARNING",
            Self::Info => "INFO",
            Self::Success => "SUCCESS",
        }
    }

    /// Get the plain text icon for non-TTY output.
    #[must_use]
    pub const fn plain_icon(&self) -> &'static str {
        match self {
            Self::Error => "[ERROR]",
            Self::Warning => "[WARN]",
            Self::Info => "[INFO]",
            Self::Success => "[OK]",
        }
    }
}

/// A styled message panel for errors, warnings, and info.
///
/// Supports error codes, suggestions, and documentation links.
#[derive(Clone, Debug)]
pub struct MessagePanel {
    /// Severity level
    pub level: MessageLevel,
    /// Short title
    pub title: String,
    /// Main message content
    pub message: String,
    /// Optional error code
    pub code: Option<ErrorCode>,
    /// Optional additional details
    pub details: Option<String>,
    /// List of suggestions
    pub suggestions: Vec<String>,
    /// Whether to show documentation link
    pub show_docs_link: bool,
}

impl MessagePanel {
    /// Create a new message panel.
    #[must_use]
    pub fn new(level: MessageLevel, title: &str, message: &str) -> Self {
        Self {
            level,
            title: title.to_string(),
            message: message.to_string(),
            code: None,
            details: None,
            suggestions: vec![],
            show_docs_link: true,
        }
    }

    /// Create an error panel.
    #[must_use]
    pub fn error(title: &str, message: &str) -> Self {
        Self::new(MessageLevel::Error, title, message)
    }

    /// Create a warning panel.
    #[must_use]
    pub fn warning(title: &str, message: &str) -> Self {
        Self::new(MessageLevel::Warning, title, message)
    }

    /// Create an info panel.
    #[must_use]
    pub fn info(title: &str, message: &str) -> Self {
        Self::new(MessageLevel::Info, title, message)
    }

    /// Create a success panel.
    #[must_use]
    pub fn success(title: &str, message: &str) -> Self {
        Self::new(MessageLevel::Success, title, message)
    }

    /// Add an error code.
    #[must_use]
    pub const fn with_code(mut self, code: ErrorCode) -> Self {
        self.code = Some(code);
        self
    }

    /// Add additional details.
    #[must_use]
    pub fn with_details(mut self, details: &str) -> Self {
        self.details = Some(details.to_string());
        self
    }

    /// Add a suggestion.
    #[must_use]
    pub fn with_suggestion(mut self, suggestion: &str) -> Self {
        self.suggestions.push(suggestion.to_string());
        self
    }

    /// Add multiple suggestions.
    #[must_use]
    pub fn with_suggestions(mut self, suggestions: Vec<String>) -> Self {
        self.suggestions.extend(suggestions);
        self
    }

    /// Hide the documentation link.
    #[must_use]
    pub const fn hide_docs_link(mut self) -> Self {
        self.show_docs_link = false;
        self
    }

    /// Render the panel to output.
    pub fn render(&self, output: &Output) {
        if !output.is_styled() {
            self.render_plain(output);
            return;
        }

        self.render_styled(output);
    }

    /// Render to stderr.
    pub fn render_to_stderr(&self, output: &Output) {
        if !output.is_stderr_styled() {
            self.render_plain_stderr();
            return;
        }

        self.render_styled_stderr(output);
    }

    /// Render with styling.
    #[allow(clippy::option_if_let_else)]
    fn render_styled(&self, output: &Output) {
        let theme = Theme::for_terminal();

        let (icon_color, border_color) = match self.level {
            MessageLevel::Error => (theme.error, theme.error),
            MessageLevel::Warning => (theme.warning, theme.warning),
            MessageLevel::Info => (theme.info, theme.info),
            MessageLevel::Success => (theme.success, theme.success),
        };

        // Build content
        let mut lines = Vec::new();

        // Icon, code, and title
        let icon = self.level.icon();
        let header = if let Some(code) = &self.code {
            format!(
                "[{icon_color} bold]{icon} {code}[/] [bold]{}[/]",
                self.title
            )
        } else {
            format!("[{icon_color} bold]{icon}[/] [bold]{}[/]", self.title)
        };
        lines.push(header);

        // Message
        if !self.message.is_empty() {
            lines.push(String::new());
            lines.push(self.message.clone());
        }

        // Details
        if let Some(ref details) = self.details {
            lines.push(String::new());
            lines.push(format!("[{}]{}[/]", theme.muted, details));
        }

        // Suggestions
        if !self.suggestions.is_empty() {
            lines.push(String::new());
            lines.push(format!("[{}]Suggestions:[/]", theme.info));
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                lines.push(format!("  {}. {suggestion}", i + 1));
            }
        }

        // Docs link
        if self.show_docs_link {
            if let Some(code) = &self.code {
                lines.push(String::new());
                lines.push(format!("[{} underline]{}[/]", theme.link, code.docs_url()));
            }
        }

        // Render as a box
        output.print(&format!("[{border_color}]┌{}┐[/]", "─".repeat(60)));
        for line in &lines {
            output.print(&format!(
                "[{border_color}]│[/] {line:<58} [{border_color}]│[/]"
            ));
        }
        output.print(&format!("[{border_color}]└{}┘[/]", "─".repeat(60)));
    }

    /// Render with styling to stderr.
    #[allow(clippy::option_if_let_else)]
    fn render_styled_stderr(&self, output: &Output) {
        let theme = Theme::for_terminal();

        let (icon_color, border_color) = match self.level {
            MessageLevel::Error => (theme.error, theme.error),
            MessageLevel::Warning => (theme.warning, theme.warning),
            MessageLevel::Info => (theme.info, theme.info),
            MessageLevel::Success => (theme.success, theme.success),
        };

        let icon = self.level.icon();
        let header = if let Some(code) = &self.code {
            format!(
                "[{icon_color} bold]{icon} {code}[/] [bold]{}[/]",
                self.title
            )
        } else {
            format!("[{icon_color} bold]{icon}[/] [bold]{}[/]", self.title)
        };

        // Use output.eprint() to get proper styled stderr output
        output.eprint(&format!("[{border_color}]┌{}┐[/]", "─".repeat(60)));
        output.eprint(&format!(
            "[{border_color}]│[/] {header:<58} [{border_color}]│[/]"
        ));

        if !self.message.is_empty() {
            output.eprint(&format!(
                "[{border_color}]│[/] {:<58} [{border_color}]│[/]",
                ""
            ));
            output.eprint(&format!(
                "[{border_color}]│[/] {:<58} [{border_color}]│[/]",
                self.message
            ));
        }

        if let Some(ref details) = self.details {
            output.eprint(&format!(
                "[{border_color}]│[/] {:<58} [{border_color}]│[/]",
                ""
            ));
            output.eprint(&format!(
                "[{border_color}]│[/] [{}]{:<56}[/] [{border_color}]│[/]",
                theme.muted, details
            ));
        }

        if !self.suggestions.is_empty() {
            output.eprint(&format!(
                "[{border_color}]│[/] {:<58} [{border_color}]│[/]",
                ""
            ));
            output.eprint(&format!(
                "[{border_color}]│[/] [{}]Suggestions:[/]{:<47} [{border_color}]│[/]",
                theme.info, ""
            ));
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                let line = format!("  {}. {suggestion}", i + 1);
                output.eprint(&format!(
                    "[{border_color}]│[/] {line:<58} [{border_color}]│[/]"
                ));
            }
        }

        if self.show_docs_link {
            if let Some(code) = &self.code {
                output.eprint(&format!(
                    "[{border_color}]│[/] {:<58} [{border_color}]│[/]",
                    ""
                ));
                output.eprint(&format!(
                    "[{border_color}]│[/] [{} underline]{:<56}[/] [{border_color}]│[/]",
                    theme.link,
                    code.docs_url()
                ));
            }
        }

        output.eprint(&format!("[{border_color}]└{}┘[/]", "─".repeat(60)));
    }

    /// Render as plain text.
    fn render_plain(&self, output: &Output) {
        let icon = self.level.plain_icon();

        // Header
        if let Some(code) = &self.code {
            output.print(&format!("{icon} {code} {}", self.title));
        } else {
            output.print(&format!("{icon} {}", self.title));
        }

        // Message
        if !self.message.is_empty() {
            output.print(&self.message);
        }

        // Details
        if let Some(ref details) = self.details {
            output.print(&format!("  {details}"));
        }

        // Suggestions
        if !self.suggestions.is_empty() {
            output.print("Suggestions:");
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                output.print(&format!("  {}. {suggestion}", i + 1));
            }
        }

        // Docs link
        if self.show_docs_link {
            if let Some(code) = &self.code {
                output.print(&format!("See: {}", code.docs_url()));
            }
        }
    }

    /// Render as plain text to stderr.
    fn render_plain_stderr(&self) {
        let icon = self.level.plain_icon();

        if let Some(code) = &self.code {
            eprintln!("{icon} {code} {}", self.title);
        } else {
            eprintln!("{icon} {}", self.title);
        }

        if !self.message.is_empty() {
            eprintln!("{}", self.message);
        }

        if let Some(ref details) = self.details {
            eprintln!("  {details}");
        }

        if !self.suggestions.is_empty() {
            eprintln!("Suggestions:");
            for (i, suggestion) in self.suggestions.iter().enumerate() {
                eprintln!("  {}. {suggestion}", i + 1);
            }
        }

        if self.show_docs_link {
            if let Some(code) = &self.code {
                eprintln!("See: {}", code.docs_url());
            }
        }
    }

    // Pre-built error panels for common errors

    /// Create panel for archive not found error.
    #[must_use]
    pub fn archive_not_found(path: &Path) -> Self {
        Self::error(
            "Archive not found",
            &format!("Could not find archive at: {}", path.display()),
        )
        .with_code(ErrorCode::ArchiveNotFound)
        .with_suggestion("Check the archive path is correct")
        .with_suggestion("Run 'xf index <path>' to index a new archive")
        .with_suggestion("Use --archive to specify an alternative path")
    }

    /// Create panel for index outdated warning.
    #[must_use]
    pub fn index_outdated() -> Self {
        Self::warning(
            "Index is outdated",
            "The search index is older than the archive and may be missing documents",
        )
        .with_code(ErrorCode::IndexStale)
        .with_suggestion("Run 'xf index' to rebuild the index")
    }

    /// Create panel for no results info.
    #[must_use]
    pub fn no_results(query: &str) -> Self {
        Self::info(
            "No results found",
            &format!("No documents match: \"{query}\""),
        )
        .with_code(ErrorCode::NoResults)
        .with_suggestion("Try broader search terms")
        .with_suggestion("Check spelling and try variations")
        .with_suggestion("Use --types to search specific content types")
        .hide_docs_link()
    }

    /// Create panel for invalid query error.
    #[must_use]
    pub fn query_invalid(query: &str, reason: &str) -> Self {
        Self::error("Invalid query", &format!("Could not parse: \"{query}\""))
            .with_code(ErrorCode::QueryInvalid)
            .with_details(reason)
            .with_suggestion("Check query syntax")
            .with_suggestion("Use quotes for exact phrases")
    }

    /// Create panel for database locked error.
    #[must_use]
    pub fn database_locked() -> Self {
        Self::error("Database locked", "Another process is using the database")
            .with_code(ErrorCode::DatabaseLocked)
            .with_suggestion("Wait for the other process to finish")
            .with_suggestion("Check for stale lock files")
    }
}

/// Collector for aggregating multiple warnings.
#[derive(Clone, Debug, Default)]
pub struct WarningCollector {
    warnings: Vec<MessagePanel>,
}

impl WarningCollector {
    /// Create a new empty collector.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            warnings: Vec::new(),
        }
    }

    /// Add a warning panel.
    pub fn add(&mut self, warning: MessagePanel) {
        self.warnings.push(warning);
    }

    /// Add a simple warning with title and message.
    pub fn add_simple(&mut self, title: &str, message: &str) {
        self.warnings.push(MessagePanel::warning(title, message));
    }

    /// Check if the collector is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.warnings.is_empty()
    }

    /// Get the number of warnings.
    #[must_use]
    pub fn count(&self) -> usize {
        self.warnings.len()
    }

    /// Render all warnings in a single summary panel.
    pub fn render(&self, output: &Output) {
        if self.warnings.is_empty() {
            return;
        }

        if !output.is_styled() {
            self.render_plain(output);
            return;
        }

        let theme = Theme::for_terminal();

        // Header
        output.print(&format!(
            "[{} bold]⚠ {} warning(s)[/]",
            theme.warning,
            self.warnings.len()
        ));
        output.print("");

        // List warnings
        for (i, warning) in self.warnings.iter().enumerate() {
            output.print(&format!(
                "[bold]{}.[/] {} - {}",
                i + 1,
                warning.title,
                warning.message
            ));
        }
    }

    /// Render plain text summary.
    fn render_plain(&self, output: &Output) {
        output.print(&format!("[WARN] {} warning(s):", self.warnings.len()));
        for (i, warning) in self.warnings.iter().enumerate() {
            output.print(&format!(
                "  {}. {} - {}",
                i + 1,
                warning.title,
                warning.message
            ));
        }
    }

    /// Render each warning as a separate panel.
    pub fn render_individually(&self, output: &Output) {
        for warning in &self.warnings {
            warning.render(output);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_code_as_string() {
        assert_eq!(ErrorCode::ArchiveNotFound.as_str(), "E100");
        assert_eq!(ErrorCode::IndexStale.as_str(), "E201");
        assert_eq!(ErrorCode::NoResults.as_str(), "E300");
        assert_eq!(ErrorCode::DatabaseConnection.as_str(), "E400");
        assert_eq!(ErrorCode::ConfigNotFound.as_str(), "E500");
    }

    #[test]
    fn test_error_code_category() {
        assert_eq!(ErrorCode::ArchiveNotFound.category(), "Archive");
        assert_eq!(ErrorCode::IndexStale.category(), "Index");
        assert_eq!(ErrorCode::NoResults.category(), "Search");
        assert_eq!(ErrorCode::DatabaseConnection.category(), "Database");
        assert_eq!(ErrorCode::ConfigNotFound.category(), "Config");
    }

    #[test]
    fn test_error_code_docs_url() {
        let url = ErrorCode::ArchiveNotFound.docs_url();
        assert!(url.contains("E100"));
        assert!(url.starts_with("https://"));
    }

    #[test]
    fn test_error_code_display() {
        assert_eq!(format!("{}", ErrorCode::ArchiveNotFound), "E100");
        assert_eq!(format!("{}", ErrorCode::NoResults), "E300");
    }

    #[test]
    fn test_message_level_icons() {
        assert_eq!(MessageLevel::Error.icon(), "✗");
        assert_eq!(MessageLevel::Warning.icon(), "⚠");
        assert_eq!(MessageLevel::Info.icon(), "ℹ");
        assert_eq!(MessageLevel::Success.icon(), "✓");
    }

    #[test]
    fn test_message_level_labels() {
        assert_eq!(MessageLevel::Error.label(), "ERROR");
        assert_eq!(MessageLevel::Warning.label(), "WARNING");
        assert_eq!(MessageLevel::Info.label(), "INFO");
        assert_eq!(MessageLevel::Success.label(), "SUCCESS");
    }

    #[test]
    fn test_message_level_plain_icons() {
        assert_eq!(MessageLevel::Error.plain_icon(), "[ERROR]");
        assert_eq!(MessageLevel::Warning.plain_icon(), "[WARN]");
        assert_eq!(MessageLevel::Info.plain_icon(), "[INFO]");
        assert_eq!(MessageLevel::Success.plain_icon(), "[OK]");
    }

    #[test]
    fn test_error_panel_has_correct_level() {
        let panel = MessagePanel::error("Test", "Message");
        assert_eq!(panel.level, MessageLevel::Error);
    }

    #[test]
    fn test_warning_panel_has_correct_level() {
        let panel = MessagePanel::warning("Test", "Message");
        assert_eq!(panel.level, MessageLevel::Warning);
    }

    #[test]
    fn test_info_panel_has_correct_level() {
        let panel = MessagePanel::info("Test", "Message");
        assert_eq!(panel.level, MessageLevel::Info);
    }

    #[test]
    fn test_success_panel_has_correct_level() {
        let panel = MessagePanel::success("Test", "Message");
        assert_eq!(panel.level, MessageLevel::Success);
    }

    #[test]
    fn test_suggestions_accumulate() {
        let panel = MessagePanel::error("E", "M")
            .with_suggestion("First")
            .with_suggestion("Second")
            .with_suggestion("Third");
        assert_eq!(panel.suggestions.len(), 3);
    }

    #[test]
    fn test_with_suggestions_vec() {
        let panel = MessagePanel::error("E", "M").with_suggestions(vec!["A".into(), "B".into()]);
        assert_eq!(panel.suggestions.len(), 2);
    }

    #[test]
    fn test_with_code() {
        let panel = MessagePanel::error("E", "M").with_code(ErrorCode::NoResults);
        assert_eq!(panel.code, Some(ErrorCode::NoResults));
    }

    #[test]
    fn test_with_details() {
        let panel = MessagePanel::error("E", "M").with_details("Additional context");
        assert_eq!(panel.details, Some("Additional context".to_string()));
    }

    #[test]
    fn test_hide_docs_link() {
        let panel = MessagePanel::error("E", "M")
            .with_code(ErrorCode::NoResults)
            .hide_docs_link();
        assert!(!panel.show_docs_link);
    }

    #[test]
    fn test_prebuilt_archive_not_found() {
        let panel = MessagePanel::archive_not_found(Path::new("/test"));
        assert_eq!(panel.code, Some(ErrorCode::ArchiveNotFound));
        assert_eq!(panel.level, MessageLevel::Error);
        assert!(!panel.suggestions.is_empty());
        assert!(panel.message.contains("/test"));
    }

    #[test]
    fn test_prebuilt_index_outdated() {
        let panel = MessagePanel::index_outdated();
        assert_eq!(panel.code, Some(ErrorCode::IndexStale));
        assert_eq!(panel.level, MessageLevel::Warning);
    }

    #[test]
    fn test_prebuilt_no_results() {
        let panel = MessagePanel::no_results("test query");
        assert_eq!(panel.code, Some(ErrorCode::NoResults));
        assert!(!panel.show_docs_link);
        assert!(panel.message.contains("test query"));
    }

    #[test]
    fn test_prebuilt_query_invalid() {
        let panel = MessagePanel::query_invalid("bad query", "Unexpected token");
        assert_eq!(panel.code, Some(ErrorCode::QueryInvalid));
        assert!(panel.details.is_some());
    }

    #[test]
    fn test_prebuilt_database_locked() {
        let panel = MessagePanel::database_locked();
        assert_eq!(panel.code, Some(ErrorCode::DatabaseLocked));
        assert_eq!(panel.level, MessageLevel::Error);
    }

    #[test]
    fn test_warning_collector_empty() {
        let collector = WarningCollector::new();
        assert!(collector.is_empty());
        assert_eq!(collector.count(), 0);
    }

    #[test]
    fn test_warning_collector_add() {
        let mut collector = WarningCollector::new();
        collector.add_simple("W1", "Warning 1");
        collector.add_simple("W2", "Warning 2");

        assert!(!collector.is_empty());
        assert_eq!(collector.count(), 2);
    }

    #[test]
    fn test_warning_collector_add_panel() {
        let mut collector = WarningCollector::new();
        collector.add(MessagePanel::warning("Title", "Message").with_code(ErrorCode::IndexStale));

        assert_eq!(collector.count(), 1);
        assert_eq!(collector.warnings[0].code, Some(ErrorCode::IndexStale));
    }

    #[test]
    fn test_render_styled_no_panic() {
        let panel = MessagePanel::error("Test", "Message")
            .with_code(ErrorCode::ArchiveNotFound)
            .with_details("Details")
            .with_suggestion("Try this");
        let output = Output::new_tty();

        // Should not panic
        panel.render(&output);
    }

    #[test]
    fn test_render_plain_no_panic() {
        let panel = MessagePanel::error("Test", "Message")
            .with_code(ErrorCode::ArchiveNotFound)
            .with_details("Details")
            .with_suggestion("Try this");
        let output = Output::new_piped();

        // Should not panic
        panel.render(&output);
    }

    #[test]
    fn test_warning_collector_render_no_panic() {
        let mut collector = WarningCollector::new();
        collector.add_simple("W1", "Warning 1");
        collector.add_simple("W2", "Warning 2");

        let output = Output::new_piped();

        // Should not panic
        collector.render(&output);
        collector.render_individually(&output);
    }

    #[test]
    fn test_warning_collector_default() {
        let collector = WarningCollector::default();
        assert!(collector.is_empty());
    }
}
