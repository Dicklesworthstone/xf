//! Statistics dashboard rendering with styled terminal output.
//!
//! Provides styled tables and panels for displaying archive statistics,
//! using the Theme system for consistent styling across the CLI.
//!
//! # Example
//!
//! ```rust,ignore
//! use xf::stats_dashboard::StatsDashboard;
//! use xf::output::Output;
//! use xf::storage::AllCounts;
//!
//! let output = Output::new_tty();
//! let dashboard = StatsDashboard::new(&stats)
//!     .with_storage_info(archive_bytes, index_bytes, db_bytes);
//!
//! dashboard.render(&output);
//! ```

use crate::output::Output;
use crate::storage::AllCounts;
use crate::theme::Theme;
use crate::{format_number, format_relative_date};
use std::path::PathBuf;

/// Format bytes as human-readable string (KB, MB, GB, TB).
///
/// Uses binary units (1024 bytes = 1 KB) for consistency with
/// typical filesystem displays.
///
/// # Examples
///
/// ```rust,ignore
/// assert_eq!(humanize_bytes(0), "0 B");
/// assert_eq!(humanize_bytes(1024), "1.00 KB");
/// assert_eq!(humanize_bytes(1024 * 1024), "1.00 MB");
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn humanize_bytes(bytes: u64) -> String {
    const KB: u64 = 1024;
    const MB: u64 = KB * 1024;
    const GB: u64 = MB * 1024;
    const TB: u64 = GB * 1024;

    if bytes >= TB {
        format!("{:.2} TB", bytes as f64 / TB as f64)
    } else if bytes >= GB {
        format!("{:.2} GB", bytes as f64 / GB as f64)
    } else if bytes >= MB {
        format!("{:.2} MB", bytes as f64 / MB as f64)
    } else if bytes >= KB {
        format!("{:.2} KB", bytes as f64 / KB as f64)
    } else {
        format!("{bytes} B")
    }
}

/// Format a percentage from a value and total.
///
/// Returns "—" if total is zero to avoid division by zero.
///
/// # Examples
///
/// ```rust,ignore
/// assert_eq!(format_percentage(25.0, 100.0), "25.0%");
/// assert_eq!(format_percentage(1.0, 0.0), "—");
/// ```
#[must_use]
pub fn format_percentage(value: f64, total: f64) -> String {
    if total == 0.0 {
        return "—".to_string();
    }
    format!("{:.1}%", (value / total) * 100.0)
}

/// Storage size information for the dashboard.
#[derive(Clone, Debug, Default)]
pub struct StorageInfo {
    /// Size of the original archive (ZIP or extracted)
    pub archive_bytes: u64,
    /// Size of the search index directory
    pub index_bytes: u64,
    /// Size of the SQLite database
    pub database_bytes: u64,
    /// Path to the archive
    pub archive_path: Option<PathBuf>,
}

impl StorageInfo {
    /// Total size of all storage components.
    #[must_use]
    pub const fn total_bytes(&self) -> u64 {
        self.archive_bytes + self.index_bytes + self.database_bytes
    }
}

/// Statistics dashboard renderer.
///
/// Renders archive statistics with styled output when in TTY mode,
/// falling back to plain text otherwise.
pub struct StatsDashboard<'a> {
    stats: &'a AllCounts,
    storage: Option<StorageInfo>,
    show_storage: bool,
    compact: bool,
}

impl<'a> StatsDashboard<'a> {
    /// Create a new dashboard for the given statistics.
    #[must_use]
    pub const fn new(stats: &'a AllCounts) -> Self {
        Self {
            stats,
            storage: None,
            show_storage: false,
            compact: false,
        }
    }

    /// Add storage information to display.
    #[must_use]
    pub fn with_storage(mut self, storage: StorageInfo) -> Self {
        self.storage = Some(storage);
        self.show_storage = true;
        self
    }

    /// Enable or disable storage panel display.
    #[must_use]
    pub const fn show_storage(mut self, show: bool) -> Self {
        self.show_storage = show;
        self
    }

    /// Enable compact mode (fewer blank lines).
    #[must_use]
    pub const fn compact(mut self, compact: bool) -> Self {
        self.compact = compact;
        self
    }

    /// Render the dashboard to output.
    pub fn render(&self, output: &Output) {
        if !output.is_styled() {
            self.render_plain(output);
            return;
        }

        self.render_styled(output);
    }

    /// Render with styling for TTY output.
    fn render_styled(&self, output: &Output) {
        let theme = Theme::for_terminal();

        // Overview panel
        self.render_overview_panel(output, &theme);

        // Content breakdown
        self.render_content_breakdown(output, &theme);

        // Storage panel (if available)
        if self.show_storage {
            if let Some(ref storage) = self.storage {
                self.render_storage_panel(output, &theme, storage);
            }
        }

        // Date range panel
        self.render_date_range(output, &theme);
    }

    /// Render overview panel showing total counts.
    fn render_overview_panel(&self, output: &Output, theme: &Theme) {
        output.print(&theme.format_heading("Archive Overview"));
        output.print(&theme.format_muted(&"─".repeat(40)));

        let total_docs = self.stats.tweets_count
            + self.stats.likes_count
            + self.stats.dms_count
            + self.stats.grok_messages_count;

        output.print(&format!(
            "  {:<20} {}",
            theme.format_muted("Total Documents:"),
            theme.format_primary(&format_number(total_docs))
        ));

        if !self.compact {
            output.print("");
        }
    }

    /// Render content breakdown with bars.
    #[allow(clippy::cast_precision_loss)]
    fn render_content_breakdown(&self, output: &Output, theme: &Theme) {
        output.print(&theme.format_heading("Content Breakdown"));
        output.print(&theme.format_muted(&"─".repeat(40)));

        let total = (self.stats.tweets_count
            + self.stats.likes_count
            + self.stats.dms_count
            + self.stats.grok_messages_count) as f64;

        let items = [
            ("Tweets", self.stats.tweets_count, theme.primary),
            ("Likes", self.stats.likes_count, theme.error),
            ("DM Messages", self.stats.dms_count, theme.secondary),
            (
                "Grok Messages",
                self.stats.grok_messages_count,
                theme.accent,
            ),
        ];

        let max_count = items.iter().map(|(_, c, _)| *c).max().unwrap_or(1);

        for (name, count, color) in items {
            let pct = format_percentage(count as f64, total);
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            let bar_len = if max_count > 0 {
                ((count as f64 / max_count as f64) * 20.0) as usize
            } else {
                0
            };
            let bar = "█".repeat(bar_len);

            output.print(&format!(
                "  {:<15} {:>10}  {:>6}  [{}]{bar}[/]",
                theme.format_muted(name),
                theme.format_primary(&format_number(count)),
                theme.format_muted(&pct),
                color
            ));
        }

        // Additional counts
        output.print("");
        output.print(&theme.format_muted("  Additional:"));
        output.print(&format!(
            "    {:<18} {}",
            theme.format_muted("DM Conversations:"),
            format_number(self.stats.dm_conversations_count)
        ));
        output.print(&format!(
            "    {:<18} {}",
            theme.format_muted("Followers:"),
            format_number(self.stats.followers_count)
        ));
        output.print(&format!(
            "    {:<18} {}",
            theme.format_muted("Following:"),
            format_number(self.stats.following_count)
        ));
        output.print(&format!(
            "    {:<18} {}",
            theme.format_muted("Blocks:"),
            format_number(self.stats.blocks_count)
        ));
        output.print(&format!(
            "    {:<18} {}",
            theme.format_muted("Mutes:"),
            format_number(self.stats.mutes_count)
        ));

        if !self.compact {
            output.print("");
        }
    }

    /// Render storage panel showing sizes.
    #[allow(clippy::cast_precision_loss)]
    fn render_storage_panel(&self, output: &Output, theme: &Theme, storage: &StorageInfo) {
        output.print(&theme.format_heading("Storage"));
        output.print(&theme.format_muted(&"─".repeat(40)));

        let total = storage.total_bytes() as f64;

        let components = [
            ("Archive", storage.archive_bytes),
            ("Index", storage.index_bytes),
            ("Database", storage.database_bytes),
        ];

        for (name, size) in components {
            let pct = format_percentage(size as f64, total);
            output.print(&format!(
                "  {:<15} {:>12}  {:>6}",
                theme.format_muted(name),
                humanize_bytes(size),
                theme.format_muted(&pct)
            ));
        }

        output.print(&theme.format_muted(&"─".repeat(40)));
        output.print(&format!(
            "  {:<15} {}",
            theme.format_muted("Total:"),
            theme.format_primary(&humanize_bytes(storage.total_bytes()))
        ));

        if let Some(ref path) = storage.archive_path {
            output.print(&format!(
                "  {:<15} {}",
                theme.format_muted("Path:"),
                theme.format_muted(&path.display().to_string())
            ));
        }

        if !self.compact {
            output.print("");
        }
    }

    /// Render date range information.
    fn render_date_range(&self, output: &Output, theme: &Theme) {
        if self.stats.first_tweet_date.is_none() && self.stats.last_tweet_date.is_none() {
            return;
        }

        output.print(&theme.format_heading("Date Range"));
        output.print(&theme.format_muted(&"─".repeat(40)));

        if let Some(first) = self.stats.first_tweet_date {
            output.print(&format!(
                "  {:<15} {}",
                theme.format_muted("First tweet:"),
                theme.format_timestamp(&format_relative_date(first))
            ));
        }

        if let Some(last) = self.stats.last_tweet_date {
            output.print(&format!(
                "  {:<15} {}",
                theme.format_muted("Last tweet:"),
                theme.format_timestamp(&format_relative_date(last))
            ));
        }

        // Calculate span
        if let (Some(first), Some(last)) = (self.stats.first_tweet_date, self.stats.last_tweet_date)
        {
            let days = (last - first).num_days();
            let years = days / 365;
            let remaining_days = days % 365;

            let span = if years > 0 {
                format!("{years} years, {remaining_days} days")
            } else {
                format!("{days} days")
            };

            output.print(&format!("  {:<15} {}", theme.format_muted("Span:"), span));
        }
    }

    /// Render as plain text for piped output.
    fn render_plain(&self, output: &Output) {
        output.print("=== Archive Statistics ===");
        output.print("");

        // Counts
        output.print(&format!(
            "Tweets: {}",
            format_number(self.stats.tweets_count)
        ));
        output.print(&format!("Likes: {}", format_number(self.stats.likes_count)));
        output.print(&format!(
            "DM Conversations: {}",
            format_number(self.stats.dm_conversations_count)
        ));
        output.print(&format!(
            "DM Messages: {}",
            format_number(self.stats.dms_count)
        ));
        output.print(&format!(
            "Grok Messages: {}",
            format_number(self.stats.grok_messages_count)
        ));
        output.print(&format!(
            "Followers: {}",
            format_number(self.stats.followers_count)
        ));
        output.print(&format!(
            "Following: {}",
            format_number(self.stats.following_count)
        ));
        output.print(&format!(
            "Blocks: {}",
            format_number(self.stats.blocks_count)
        ));
        output.print(&format!("Mutes: {}", format_number(self.stats.mutes_count)));

        // Storage
        if self.show_storage {
            if let Some(ref storage) = self.storage {
                output.print("");
                output.print("Storage:");
                output.print(&format!(
                    "  Archive:  {}",
                    humanize_bytes(storage.archive_bytes)
                ));
                output.print(&format!(
                    "  Index:    {}",
                    humanize_bytes(storage.index_bytes)
                ));
                output.print(&format!(
                    "  Database: {}",
                    humanize_bytes(storage.database_bytes)
                ));
                output.print(&format!(
                    "  Total:    {}",
                    humanize_bytes(storage.total_bytes())
                ));
            }
        }

        // Date range
        if let (Some(first), Some(last)) = (self.stats.first_tweet_date, self.stats.last_tweet_date)
        {
            output.print("");
            output.print(&format!("First tweet: {}", first.format("%Y-%m-%d %H:%M")));
            output.print(&format!("Last tweet: {}", last.format("%Y-%m-%d %H:%M")));
        }
    }

    /// Get a summary string for testing.
    #[must_use]
    pub fn summary(&self) -> String {
        let total = self.stats.tweets_count
            + self.stats.likes_count
            + self.stats.dms_count
            + self.stats.grok_messages_count;

        format!(
            "Total: {} docs, Tweets: {}, Likes: {}",
            format_number(total),
            format_number(self.stats.tweets_count),
            format_number(self.stats.likes_count)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_stats() -> AllCounts {
        AllCounts {
            tweets_count: 12345,
            likes_count: 5000,
            dms_count: 1000,
            dm_conversations_count: 50,
            followers_count: 500,
            following_count: 300,
            blocks_count: 10,
            mutes_count: 5,
            grok_messages_count: 100,
            first_tweet_date: Some(Utc::now() - chrono::Duration::days(365)),
            last_tweet_date: Some(Utc::now()),
        }
    }

    #[test]
    fn test_humanize_bytes_small() {
        assert_eq!(humanize_bytes(0), "0 B");
        assert_eq!(humanize_bytes(512), "512 B");
        assert_eq!(humanize_bytes(1023), "1023 B");
    }

    #[test]
    fn test_humanize_bytes_kb() {
        assert_eq!(humanize_bytes(1024), "1.00 KB");
        assert_eq!(humanize_bytes(1536), "1.50 KB");
        assert_eq!(humanize_bytes(1024 * 100), "100.00 KB");
    }

    #[test]
    fn test_humanize_bytes_mb() {
        assert_eq!(humanize_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(humanize_bytes(1024 * 1024 * 50), "50.00 MB");
    }

    #[test]
    fn test_humanize_bytes_gb() {
        assert_eq!(humanize_bytes(1024 * 1024 * 1024), "1.00 GB");
        assert_eq!(humanize_bytes(1024 * 1024 * 1024 * 5), "5.00 GB");
    }

    #[test]
    fn test_humanize_bytes_tb() {
        assert_eq!(humanize_bytes(1024_u64 * 1024 * 1024 * 1024), "1.00 TB");
    }

    #[test]
    fn test_format_percentage() {
        assert_eq!(format_percentage(25.0, 100.0), "25.0%");
        assert_eq!(format_percentage(33.0, 100.0), "33.0%");
        assert_eq!(format_percentage(1.0, 3.0), "33.3%");
    }

    #[test]
    fn test_format_percentage_zero_total() {
        assert_eq!(format_percentage(10.0, 0.0), "—");
        assert_eq!(format_percentage(0.0, 0.0), "—");
    }

    #[test]
    fn test_storage_info_total() {
        let storage = StorageInfo {
            archive_bytes: 1000,
            index_bytes: 500,
            database_bytes: 200,
            archive_path: None,
        };
        assert_eq!(storage.total_bytes(), 1700);
    }

    #[test]
    fn test_dashboard_new() {
        let stats = make_stats();
        let dashboard = StatsDashboard::new(&stats);
        assert!(!dashboard.show_storage);
        assert!(!dashboard.compact);
    }

    #[test]
    fn test_dashboard_with_storage() {
        let stats = make_stats();
        let storage = StorageInfo {
            archive_bytes: 1024 * 1024 * 100,
            index_bytes: 1024 * 1024 * 50,
            database_bytes: 1024 * 1024 * 10,
            archive_path: Some(PathBuf::from("/tmp/archive")),
        };

        let dashboard = StatsDashboard::new(&stats).with_storage(storage);
        assert!(dashboard.show_storage);
    }

    #[test]
    fn test_dashboard_summary() {
        let stats = make_stats();
        let dashboard = StatsDashboard::new(&stats);
        let summary = dashboard.summary();

        assert!(summary.contains("18,445")); // Total: 12345 + 5000 + 1000 + 100
        assert!(summary.contains("12,345")); // Tweets
        assert!(summary.contains("5,000")); // Likes
    }

    #[test]
    fn test_render_styled_no_panic() {
        let stats = make_stats();
        let dashboard = StatsDashboard::new(&stats);
        let output = Output::new_tty();

        // Should not panic
        dashboard.render(&output);
    }

    #[test]
    fn test_render_plain_no_panic() {
        let stats = make_stats();
        let dashboard = StatsDashboard::new(&stats);
        let output = Output::new_piped();

        // Should not panic
        dashboard.render(&output);
    }

    #[test]
    fn test_render_with_storage_no_panic() {
        let stats = make_stats();
        let storage = StorageInfo {
            archive_bytes: 1024 * 1024 * 100,
            index_bytes: 1024 * 1024 * 50,
            database_bytes: 1024 * 1024 * 10,
            archive_path: Some(PathBuf::from("/tmp/archive")),
        };

        let dashboard = StatsDashboard::new(&stats)
            .with_storage(storage)
            .compact(true);
        let output = Output::new_tty();

        // Should not panic
        dashboard.render(&output);
    }

    #[test]
    fn test_render_empty_dates_no_panic() {
        let stats = AllCounts {
            tweets_count: 0,
            likes_count: 0,
            dms_count: 0,
            dm_conversations_count: 0,
            followers_count: 0,
            following_count: 0,
            blocks_count: 0,
            mutes_count: 0,
            grok_messages_count: 0,
            first_tweet_date: None,
            last_tweet_date: None,
        };

        let dashboard = StatsDashboard::new(&stats);
        let output = Output::new_piped();

        // Should not panic
        dashboard.render(&output);
    }

    #[test]
    fn test_content_breakdown_percentages() {
        let stats = AllCounts {
            tweets_count: 50,
            likes_count: 30,
            dms_count: 15,
            dm_conversations_count: 5,
            followers_count: 0,
            following_count: 0,
            blocks_count: 0,
            mutes_count: 0,
            grok_messages_count: 5,
            first_tweet_date: None,
            last_tweet_date: None,
        };

        // Total = 100, so percentages should be exact
        let total = 100.0;
        assert_eq!(format_percentage(50.0, total), "50.0%");
        assert_eq!(format_percentage(30.0, total), "30.0%");
        assert_eq!(format_percentage(15.0, total), "15.0%");
        assert_eq!(format_percentage(5.0, total), "5.0%");

        // Verify dashboard doesn't panic with these values
        let dashboard = StatsDashboard::new(&stats);
        dashboard.render(&Output::new_piped());
    }
}
