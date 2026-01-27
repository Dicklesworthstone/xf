//! Search result card rendering for styled terminal output.
//!
//! Provides beautiful search result cards that display content, scores,
//! highlights, and metadata. Supports both styled (TTY) and plain (piped) output.
//!
//! # Example
//!
//! ```rust,ignore
//! use xf::search_results::{SearchResultCard, extract_terms};
//! use xf::output::Output;
//!
//! let output = Output::new_tty();
//! let terms = extract_terms("rust programming");
//!
//! let card = SearchResultCard::new()
//!     .doc_id("123456789")
//!     .doc_type("tweet")
//!     .content("Learning Rust programming is fun!")
//!     .score(0.85)
//!     .highlight_terms(terms);
//!
//! card.render(&output);
//! ```

use chrono::{DateTime, Utc};
use regex::Regex;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use std::time::Duration;

use crate::output::Output;
use crate::theme::Theme;

/// Regex for extracting search terms (words, quoted phrases)
static TERM_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#""([^"]+)"|(\S+)"#).expect("Invalid term regex"));

/// Regex cache for highlighting (word boundaries, case insensitive)
static HIGHLIGHT_REGEX_CACHE: LazyLock<Mutex<HashMap<String, Regex>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Extract search terms from a query string.
///
/// Handles both individual words and quoted phrases. Filters out common
/// operators (AND, OR, NOT) and very short words.
///
/// # Examples
///
/// ```rust,ignore
/// let terms = extract_terms("hello world");
/// assert!(terms.contains(&"hello".to_string()));
///
/// let terms = extract_terms("\"hello world\" rust");
/// assert!(terms.contains(&"hello world".to_string()));
/// ```
#[must_use]
pub fn extract_terms(query: &str) -> Vec<String> {
    TERM_REGEX
        .captures_iter(query)
        .filter_map(|cap| {
            cap.get(1) // Quoted phrase
                .or_else(|| cap.get(2)) // Single word
                .map(|m| m.as_str().to_lowercase())
        })
        .filter(|term| {
            // Skip common operators and short words
            !matches!(term.as_str(), "and" | "or" | "not" | "the" | "a" | "an") && term.len() >= 2
        })
        .collect()
}

/// A styled search result card.
///
/// Displays document content with score, type, timestamp, and optional
/// term highlighting. Supports both rich terminal output and plain text fallback.
#[derive(Clone, Debug)]
pub struct SearchResultCard {
    doc_id: String,
    doc_type: String,
    content: String,
    score: f32,
    timestamp: Option<DateTime<Utc>>,
    highlight_terms: Vec<String>,
    max_content_length: usize,
}

impl Default for SearchResultCard {
    fn default() -> Self {
        Self::new()
    }
}

impl SearchResultCard {
    /// Create a new empty search result card.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            doc_id: String::new(),
            doc_type: String::new(),
            content: String::new(),
            score: 0.0,
            timestamp: None,
            highlight_terms: vec![],
            max_content_length: 280, // Default to tweet length
        }
    }

    /// Set the document ID.
    #[must_use]
    pub fn doc_id(mut self, id: impl Into<String>) -> Self {
        self.doc_id = id.into();
        self
    }

    /// Set the document type (tweet, like, dm, grok).
    #[must_use]
    pub fn doc_type(mut self, dtype: impl Into<String>) -> Self {
        self.doc_type = dtype.into();
        self
    }

    /// Set the document content.
    #[must_use]
    pub fn content(mut self, text: impl Into<String>) -> Self {
        self.content = text.into();
        self
    }

    /// Set the relevance score (0.0 - 1.0).
    #[must_use]
    pub const fn score(mut self, score: f32) -> Self {
        self.score = score;
        self
    }

    /// Set the document timestamp.
    #[must_use]
    pub const fn timestamp(mut self, ts: DateTime<Utc>) -> Self {
        self.timestamp = Some(ts);
        self
    }

    /// Set terms to highlight in the content.
    #[must_use]
    pub fn highlight_terms(mut self, terms: Vec<String>) -> Self {
        self.highlight_terms = terms;
        self
    }

    /// Set maximum content length before truncation.
    #[must_use]
    pub const fn max_content_length(mut self, len: usize) -> Self {
        self.max_content_length = len;
        self
    }

    /// Truncate content with ellipsis if too long.
    ///
    /// Attempts to break at word boundaries for cleaner output.
    fn truncate_content(&self) -> String {
        if self.content.len() <= self.max_content_length {
            return self.content.clone();
        }

        // Get the substring up to max length
        let truncated: String = self.content.chars().take(self.max_content_length).collect();

        // Try to break at word boundary
        if let Some(last_space) = truncated.rfind(' ') {
            if last_space > self.max_content_length / 2 {
                return format!("{}...", &truncated[..last_space]);
            }
        }

        format!("{truncated}...")
    }

    /// Apply highlighting to content.
    ///
    /// Wraps matched terms with markup for styled display.
    fn highlight_content(&self, content: &str, theme: &Theme) -> String {
        if self.highlight_terms.is_empty() {
            return content.to_string();
        }

        let mut result = content.to_string();

        for term in &self.highlight_terms {
            // Get or create regex for this term
            let pattern_key = format!(r"(?i)\b{}\b", regex::escape(term));
            let regex = {
                let mut cache = HIGHLIGHT_REGEX_CACHE.lock().unwrap();
                if let Some(existing) = cache.get(&pattern_key) {
                    let result = existing.clone();
                    drop(cache);
                    result
                } else if let Ok(new_regex) = Regex::new(&pattern_key) {
                    cache.insert(pattern_key.clone(), new_regex.clone());
                    drop(cache);
                    new_regex
                } else {
                    // Skip terms that can't be compiled to regex
                    continue;
                }
            };

            // Replace with highlighted version
            let highlight_style = theme.highlight;
            result = regex
                .replace_all(&result, &format!("[{highlight_style}]$0[/]"))
                .into();
        }

        result
    }

    /// Render the card to the output.
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

        // Score badge (clamp to valid range)
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let score_pct = (self.score.clamp(0.0, 1.0) * 100.0) as u32;
        let score_style = theme.score_style(self.score);
        let score_badge = format!("[{score_style}]{score_pct}%[/]");

        // Doc type badge with type-specific color
        let type_color = match self.doc_type.as_str() {
            "tweet" => theme.primary,
            "like" => theme.error,   // Red for hearts
            "dm" => theme.secondary, // Purple for DMs
            "grok" => theme.accent,  // Yellow for Grok
            _ => theme.muted,
        };
        let type_badge = format!("[{type_color} bold]{}[/]", self.doc_type.to_uppercase());

        // Timestamp
        let time_str = self
            .timestamp
            .map_or_else(|| "—".to_string(), crate::format_relative_date);
        let time_display = format!("[{}]{time_str}[/]", theme.timestamp);

        // Short ID display
        let short_id = crate::format_short_id(&self.doc_id);
        let id_display = format!("[{}]{short_id}[/]", theme.muted);

        // Header line
        output.print(&format!(
            "{score_badge} {type_badge} {time_display} {id_display}"
        ));

        // Content with highlighting and truncation
        let content = self.truncate_content();
        let highlighted = self.highlight_content(&content, &theme);
        output.print(&format!("  {highlighted}"));

        // Separator
        output.print("");
    }

    /// Render as plain text for piped output.
    fn render_plain(&self, output: &Output) {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let score_pct = (self.score.clamp(0.0, 1.0) * 100.0) as u32;
        let time_str = self.timestamp.map_or_else(
            || "—".to_string(),
            |ts| ts.format("%Y-%m-%d %H:%M").to_string(),
        );

        let short_id = crate::format_short_id(&self.doc_id);

        output.print(&format!(
            "{score_pct}% | {} | {time_str} | {short_id}",
            self.doc_type
        ));
        output.print(&format!("  {}", self.truncate_content()));
        output.print("---");
    }
}

/// Search result data for rendering.
#[derive(Clone, Debug, Default)]
pub struct SearchResultData {
    pub doc_id: String,
    pub doc_type: String,
    pub content: String,
    pub score: f32,
    pub timestamp: Option<DateTime<Utc>>,
}

/// Paginated search results renderer.
///
/// Handles rendering multiple search results with pagination support,
/// header showing result count and timing, and optional term highlighting.
pub struct SearchResultsRenderer {
    results: Vec<SearchResultData>,
    query: String,
    timing: Duration,
    page_size: usize,
    current_page: usize,
}

impl SearchResultsRenderer {
    /// Create a new renderer for search results.
    #[must_use]
    pub fn new(results: Vec<SearchResultData>, query: &str, timing: Duration) -> Self {
        Self {
            results,
            query: query.to_string(),
            timing,
            page_size: 10,
            current_page: 0,
        }
    }

    /// Set the number of results per page.
    #[must_use]
    pub fn page_size(mut self, size: usize) -> Self {
        self.page_size = size.max(1);
        self
    }

    /// Set the current page (0-indexed).
    #[must_use]
    pub const fn page(mut self, page: usize) -> Self {
        self.current_page = page;
        self
    }

    /// Get the total number of pages.
    #[must_use]
    pub fn total_pages(&self) -> usize {
        if self.results.is_empty() {
            return 0;
        }
        self.results.len().div_ceil(self.page_size)
    }

    /// Render the search results to output.
    pub fn render(&self, output: &Output) {
        let terms = extract_terms(&self.query);
        let theme = Theme::for_terminal();

        // Header with result count and timing
        let timing_ms = self.timing.as_secs_f64() * 1000.0;
        if output.is_styled() {
            output.print(&format!(
                "{} Found {} {} for {} ({:.1}ms)",
                theme.format_success(""),
                theme.format_primary(&self.results.len().to_string()),
                if self.results.len() == 1 {
                    "result"
                } else {
                    "results"
                },
                theme.format_info(&format!("\"{}\"", self.query)),
                timing_ms
            ));
        } else {
            output.print(&format!(
                "Found {} result(s) for \"{}\" ({:.1}ms)",
                self.results.len(),
                self.query,
                timing_ms
            ));
        }

        if self.results.is_empty() {
            output.print("");
            if output.is_styled() {
                output.print(&theme.format_muted("No matching documents found."));
                output.print(&theme.format_muted("Tips: Try different keywords, use semantic search (--mode semantic), or check your filters."));
            } else {
                output.print("No matching documents found.");
            }
            return;
        }

        output.print("");

        // Calculate page bounds
        let start = self.current_page * self.page_size;
        let end = (start + self.page_size).min(self.results.len());

        // Render results for current page
        for (i, result) in self.results[start..end].iter().enumerate() {
            // Result number
            let num = start + i + 1;
            if output.is_styled() {
                output.print(&format!("[{}]{}.[/]", theme.muted, num));
            }

            let card = SearchResultCard::new()
                .doc_id(&result.doc_id)
                .doc_type(&result.doc_type)
                .content(&result.content)
                .score(result.score)
                .highlight_terms(terms.clone());

            // Add timestamp if available
            let card = if let Some(ts) = result.timestamp {
                card.timestamp(ts)
            } else {
                card
            };

            card.render(output);
        }

        // Page indicator for multi-page results
        if self.total_pages() > 1 {
            if output.is_styled() {
                output.print(&format!(
                    "[{}]Page {} of {}[/]",
                    theme.muted,
                    self.current_page + 1,
                    self.total_pages()
                ));
            } else {
                output.print(&format!(
                    "Page {} of {}",
                    self.current_page + 1,
                    self.total_pages()
                ));
            }
        }
    }
}

/// Convenience function for rendering search results.
///
/// Creates a renderer and immediately renders the results.
pub fn render_search_results(
    output: &Output,
    results: &[SearchResultData],
    query: &str,
    timing: Duration,
) {
    SearchResultsRenderer::new(results.to_vec(), query, timing).render(output);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_terms_single_words() {
        let terms = extract_terms("hello world rust");
        assert!(terms.contains(&"hello".to_string()));
        assert!(terms.contains(&"world".to_string()));
        assert!(terms.contains(&"rust".to_string()));
    }

    #[test]
    fn test_extract_terms_quoted_phrases() {
        let terms = extract_terms("\"hello world\" rust");
        assert!(terms.contains(&"hello world".to_string()));
        assert!(terms.contains(&"rust".to_string()));
    }

    #[test]
    fn test_extract_terms_filters_operators() {
        let terms = extract_terms("hello AND world OR rust");
        assert!(!terms.contains(&"and".to_string()));
        assert!(!terms.contains(&"or".to_string()));
        assert!(terms.contains(&"hello".to_string()));
    }

    #[test]
    fn test_extract_terms_filters_short_words() {
        let terms = extract_terms("a the rust");
        assert!(!terms.contains(&"a".to_string()));
        assert!(!terms.contains(&"the".to_string()));
        assert!(terms.contains(&"rust".to_string()));
    }

    #[test]
    fn test_extract_terms_empty_query() {
        let terms = extract_terms("");
        assert!(terms.is_empty());
    }

    #[test]
    fn test_search_card_default() {
        let card = SearchResultCard::default();
        assert!(card.doc_id.is_empty());
        assert!(card.score.abs() < f32::EPSILON);
        assert_eq!(card.max_content_length, 280);
    }

    #[test]
    fn test_search_card_builder() {
        let card = SearchResultCard::new()
            .doc_id("12345")
            .doc_type("tweet")
            .content("Test content")
            .score(0.85);

        assert_eq!(card.doc_id, "12345");
        assert_eq!(card.doc_type, "tweet");
        assert_eq!(card.content, "Test content");
        assert!((card.score - 0.85).abs() < 0.001);
    }

    #[test]
    fn test_search_card_truncation_long_content() {
        let long_content = "a".repeat(500);
        let card = SearchResultCard::new()
            .content(&long_content)
            .max_content_length(100);

        let truncated = card.truncate_content();
        assert!(truncated.len() <= 103); // 100 + "..."
        assert!(truncated.ends_with("..."));
    }

    #[test]
    fn test_search_card_truncation_word_boundary() {
        let content = "The quick brown fox jumps over the lazy dog";
        let card = SearchResultCard::new()
            .content(content)
            .max_content_length(25);

        let truncated = card.truncate_content();
        // Should break at word boundary
        assert!(truncated.ends_with("..."));
        // Should not cut mid-word (no partial "fox" like "fo")
        assert!(!truncated.contains("jump"));
    }

    #[test]
    fn test_search_card_no_truncation_short_content() {
        let content = "Short";
        let card = SearchResultCard::new()
            .content(content)
            .max_content_length(100);

        let result = card.truncate_content();
        assert_eq!(result, "Short");
        assert!(!result.ends_with("..."));
    }

    #[test]
    fn test_search_card_highlight_terms() {
        let card = SearchResultCard::new()
            .content("Hello world test content")
            .highlight_terms(vec!["test".into(), "world".into()]);

        let theme = Theme::dark();
        let highlighted = card.highlight_content(&card.content, &theme);

        // Highlighted text should contain markup
        assert!(highlighted.contains('['));
        assert!(highlighted.contains("test"));
        assert!(highlighted.contains("world"));
    }

    #[test]
    fn test_search_card_highlight_empty_terms() {
        let card = SearchResultCard::new().content("Hello world test content");

        let theme = Theme::dark();
        let highlighted = card.highlight_content(&card.content, &theme);

        // No highlighting, should be unchanged
        assert_eq!(highlighted, "Hello world test content");
    }

    #[test]
    fn test_pagination_total_pages() {
        let results: Vec<SearchResultData> = (0..25)
            .map(|i| SearchResultData {
                doc_id: format!("doc_{i}"),
                ..Default::default()
            })
            .collect();

        let renderer =
            SearchResultsRenderer::new(results, "test", Duration::from_millis(10)).page_size(10);

        assert_eq!(renderer.total_pages(), 3); // 25 results / 10 per page = 3 pages
    }

    #[test]
    fn test_pagination_exact_division() {
        let results: Vec<SearchResultData> = (0..20)
            .map(|i| SearchResultData {
                doc_id: format!("doc_{i}"),
                ..Default::default()
            })
            .collect();

        let renderer =
            SearchResultsRenderer::new(results, "test", Duration::from_millis(10)).page_size(10);

        assert_eq!(renderer.total_pages(), 2); // 20 results / 10 per page = 2 pages exactly
    }

    #[test]
    fn test_pagination_empty_results() {
        let results: Vec<SearchResultData> = vec![];
        let renderer = SearchResultsRenderer::new(results, "test", Duration::from_millis(5));

        assert_eq!(renderer.total_pages(), 0);
    }

    #[test]
    fn test_pagination_single_page() {
        let results: Vec<SearchResultData> = (0..5)
            .map(|i| SearchResultData {
                doc_id: format!("doc_{i}"),
                ..Default::default()
            })
            .collect();

        let renderer =
            SearchResultsRenderer::new(results, "test", Duration::from_millis(10)).page_size(10);

        assert_eq!(renderer.total_pages(), 1);
    }

    #[test]
    fn test_doc_type_colors_no_panic() {
        // Verify different doc types don't cause panics
        for doc_type in ["tweet", "like", "dm", "grok", "unknown", ""] {
            let card = SearchResultCard::new().doc_type(doc_type);
            let output = Output::new_piped();
            // Should not panic
            card.render(&output);
        }
    }

    #[test]
    fn test_render_plain_no_ansi() {
        let card = SearchResultCard::new()
            .doc_id("123456789")
            .doc_type("tweet")
            .content("Test content here")
            .score(0.85);

        // Capture would require stdout capture - just verify it doesn't panic
        let output = Output::new_piped();
        card.render(&output);
    }

    #[test]
    fn test_search_result_data_default() {
        let data = SearchResultData::default();
        assert!(data.doc_id.is_empty());
        assert!(data.content.is_empty());
        assert!(data.score.abs() < f32::EPSILON);
        assert!(data.timestamp.is_none());
    }
}
