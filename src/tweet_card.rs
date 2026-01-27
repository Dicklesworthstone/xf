//! Tweet card rendering for styled terminal output.
//!
//! Provides beautiful tweet display cards with content styling for hashtags,
//! mentions, URLs, and optional engagement metrics. Supports both styled
//! (TTY) and plain (piped) output.
//!
//! # Example
//!
//! ```rust,ignore
//! use xf::tweet_card::TweetCard;
//! use xf::output::Output;
//! use xf::model::Tweet;
//!
//! let output = Output::new_tty();
//! let card = TweetCard::new(&tweet)
//!     .show_engagement(true)
//!     .highlight_terms(vec!["rust".into()]);
//!
//! card.render(&output);
//! ```

use regex::Regex;
use std::sync::LazyLock;

use crate::model::Tweet;
use crate::output::Output;
use crate::theme::Theme;

/// Pre-compiled regex for hashtags
static HASHTAG_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"#\w+").expect("Invalid hashtag regex"));

/// Pre-compiled regex for mentions
static MENTION_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"@\w+").expect("Invalid mention regex"));

/// Pre-compiled regex for URLs
static URL_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"https?://\S+").expect("Invalid URL regex"));

/// A styled tweet card.
///
/// Displays a tweet with styled content, engagement metrics, and metadata.
/// Content styling includes hashtag, mention, and URL highlighting.
#[derive(Clone, Debug)]
pub struct TweetCard<'a> {
    tweet: &'a Tweet,
    show_engagement: bool,
    show_media: bool,
    highlight_terms: Vec<String>,
    compact: bool,
}

impl<'a> TweetCard<'a> {
    /// Create a new tweet card for the given tweet.
    #[must_use]
    pub const fn new(tweet: &'a Tweet) -> Self {
        Self {
            tweet,
            show_engagement: true,
            show_media: true,
            highlight_terms: vec![],
            compact: false,
        }
    }

    /// Set whether to show engagement metrics (likes, retweets).
    #[must_use]
    pub const fn show_engagement(mut self, show: bool) -> Self {
        self.show_engagement = show;
        self
    }

    /// Set whether to show media indicators.
    #[must_use]
    pub const fn show_media(mut self, show: bool) -> Self {
        self.show_media = show;
        self
    }

    /// Set terms to highlight in the content.
    #[must_use]
    pub fn highlight_terms(mut self, terms: Vec<String>) -> Self {
        self.highlight_terms = terms;
        self
    }

    /// Set compact mode for nested displays.
    #[must_use]
    pub const fn compact(mut self, compact: bool) -> Self {
        self.compact = compact;
        self
    }

    /// Check if the tweet has any engagement.
    #[must_use]
    pub const fn has_engagement(&self) -> bool {
        self.tweet.favorite_count > 0 || self.tweet.retweet_count > 0
    }

    /// Check if the tweet has media attachments.
    #[must_use]
    pub fn has_media(&self) -> bool {
        !self.tweet.media.is_empty()
    }

    /// Check if the tweet is a reply.
    #[must_use]
    pub const fn is_reply(&self) -> bool {
        self.tweet.in_reply_to_status_id.is_some()
    }

    /// Render the tweet card to output.
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

        // Header line: short ID, timestamp, indicators
        let header = self.render_header(&theme);
        output.print(&header);

        // Tweet content with styling
        let content = self.render_content(&theme);
        output.print(&format!("  {content}"));

        // Engagement metrics
        if self.show_engagement && self.has_engagement() {
            let engagement = self.render_engagement(&theme);
            output.print(&format!("  {engagement}"));
        }

        // Media indicators
        if self.show_media && self.has_media() {
            let media = self.render_media(&theme);
            output.print(&format!("  {media}"));
        }

        // Separator (unless compact)
        if !self.compact {
            output.print("");
        }
    }

    /// Render the header line.
    fn render_header(&self, theme: &Theme) -> String {
        let short_id = crate::format_short_id(&self.tweet.id);
        let timestamp = crate::format_relative_date(self.tweet.created_at);

        let mut parts = vec![
            format!("[{}]{}[/]", theme.primary, short_id),
            format!("[{}]{timestamp}[/]", theme.timestamp),
        ];

        // Indicators
        let mut indicators = Vec::new();

        if self.is_reply() {
            indicators.push(format!("[{}]Reply[/]", theme.muted));
        }

        if self.tweet.is_retweet {
            indicators.push(format!("[{}]RT[/]", theme.muted));
        }

        if !indicators.is_empty() {
            parts.push(indicators.join(" "));
        }

        parts.join(" | ")
    }

    /// Render the tweet content with styled hashtags, mentions, and URLs.
    fn render_content(&self, theme: &Theme) -> String {
        let mut text = self.tweet.full_text.clone();

        // Apply search term highlighting first (takes precedence)
        for term in &self.highlight_terms {
            if let Ok(pattern) = Regex::new(&format!(r"(?i)\b{}\b", regex::escape(term))) {
                let highlight_style = theme.highlight;
                text = pattern
                    .replace_all(&text, format!("[{highlight_style}]$0[/]"))
                    .into_owned();
            }
        }

        // Style hashtags
        text = HASHTAG_REGEX
            .replace_all(&text, format!("[{}]$0[/]", theme.hashtag))
            .into_owned();

        // Style mentions
        text = MENTION_REGEX
            .replace_all(&text, format!("[{}]$0[/]", theme.mention))
            .into_owned();

        // Style URLs
        text = URL_REGEX
            .replace_all(&text, format!("[{} underline]$0[/]", theme.link))
            .into_owned();

        text
    }

    /// Render engagement metrics.
    fn render_engagement(&self, theme: &Theme) -> String {
        let mut parts = Vec::new();

        if self.tweet.favorite_count > 0 {
            parts.push(format!(
                "[{}]{}[/] likes",
                theme.error,
                crate::format_number(self.tweet.favorite_count)
            ));
        }

        if self.tweet.retweet_count > 0 {
            parts.push(format!(
                "[{}]{}[/] retweets",
                theme.success,
                crate::format_number(self.tweet.retweet_count)
            ));
        }

        format!("[{}]{}[/]", theme.muted, parts.join(" | "))
    }

    /// Render media indicators.
    fn render_media(&self, theme: &Theme) -> String {
        let media_types: Vec<_> = self
            .tweet
            .media
            .iter()
            .map(|m| m.media_type.as_str())
            .collect();

        // Count by type
        let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
        for t in &media_types {
            *counts.entry(t).or_insert(0) += 1;
        }

        let parts: Vec<String> = counts
            .into_iter()
            .map(|(k, v)| {
                if v > 1 {
                    format!("{v} {k}s")
                } else {
                    k.to_string()
                }
            })
            .collect();

        format!("[{}][{}][/]", theme.muted, parts.join(", "))
    }

    /// Render as plain text for piped output.
    fn render_plain(&self, output: &Output) {
        let short_id = crate::format_short_id(&self.tweet.id);
        let timestamp = self.tweet.created_at.format("%Y-%m-%d %H:%M");

        // Header
        let mut header_parts = vec![short_id, timestamp.to_string()];

        if self.is_reply() {
            header_parts.push("Reply".to_string());
        }
        if self.tweet.is_retweet {
            header_parts.push("RT".to_string());
        }

        output.print(&header_parts.join(" | "));

        // Content
        output.print(&format!("  {}", self.tweet.full_text));

        // Engagement
        if self.show_engagement && self.has_engagement() {
            let mut parts = Vec::new();
            if self.tweet.favorite_count > 0 {
                parts.push(format!(
                    "{} likes",
                    crate::format_number(self.tweet.favorite_count)
                ));
            }
            if self.tweet.retweet_count > 0 {
                parts.push(format!(
                    "{} retweets",
                    crate::format_number(self.tweet.retweet_count)
                ));
            }
            output.print(&format!("  {}", parts.join(" | ")));
        }

        // Media
        if self.show_media && self.has_media() {
            let media_count = self.tweet.media.len();
            output.print(&format!("  [{media_count} media]"));
        }

        output.print("---");
    }
}

/// Display a conversation thread with tree structure.
pub struct ConversationThread<'a> {
    tweets: &'a [Tweet],
    highlight_terms: Vec<String>,
}

impl<'a> ConversationThread<'a> {
    /// Create a new conversation thread display.
    #[must_use]
    pub const fn new(tweets: &'a [Tweet]) -> Self {
        Self {
            tweets,
            highlight_terms: vec![],
        }
    }

    /// Set terms to highlight in all tweets.
    #[must_use]
    pub fn highlight_terms(mut self, terms: Vec<String>) -> Self {
        self.highlight_terms = terms;
        self
    }

    /// Render the thread.
    pub fn render(&self, output: &Output) {
        if self.tweets.is_empty() {
            return;
        }

        let theme = Theme::for_terminal();

        for (i, tweet) in self.tweets.iter().enumerate() {
            let is_first = i == 0;
            let is_last = i == self.tweets.len() - 1;

            // Tree structure prefix for non-first tweets
            if !is_first && output.is_styled() {
                let prefix = if is_last { "└─" } else { "├─" };
                output.print(&format!("[{}]{prefix}[/]", theme.muted));
            }

            TweetCard::new(tweet)
                .highlight_terms(self.highlight_terms.clone())
                .compact(!is_last)
                .render(output);
        }
    }
}

/// Convenience function to render a single tweet.
pub fn render_tweet(output: &Output, tweet: &Tweet) {
    TweetCard::new(tweet).render(output);
}

/// Convenience function to render a tweet with search highlights.
pub fn render_tweet_with_highlights(output: &Output, tweet: &Tweet, terms: Vec<String>) {
    TweetCard::new(tweet).highlight_terms(terms).render(output);
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn make_tweet() -> Tweet {
        Tweet {
            id: "1234567890123456789".into(),
            created_at: Utc::now(),
            full_text: "Hello #rust @rustlang https://rust-lang.org".into(),
            source: None,
            favorite_count: 100,
            retweet_count: 50,
            lang: None,
            in_reply_to_status_id: None,
            in_reply_to_user_id: None,
            in_reply_to_screen_name: None,
            is_retweet: false,
            hashtags: vec!["rust".into()],
            user_mentions: vec![],
            urls: vec![],
            media: vec![],
        }
    }

    #[test]
    fn test_tweet_card_new() {
        let tweet = make_tweet();
        let card = TweetCard::new(&tweet);
        assert!(card.show_engagement);
        assert!(card.show_media);
        assert!(!card.compact);
    }

    #[test]
    fn test_tweet_card_builder() {
        let tweet = make_tweet();
        let card = TweetCard::new(&tweet)
            .show_engagement(false)
            .show_media(false)
            .compact(true)
            .highlight_terms(vec!["test".into()]);

        assert!(!card.show_engagement);
        assert!(!card.show_media);
        assert!(card.compact);
        assert_eq!(card.highlight_terms.len(), 1);
    }

    #[test]
    fn test_has_engagement() {
        let mut tweet = make_tweet();
        let card = TweetCard::new(&tweet);
        assert!(card.has_engagement());

        tweet.favorite_count = 0;
        tweet.retweet_count = 0;
        let card = TweetCard::new(&tweet);
        assert!(!card.has_engagement());
    }

    #[test]
    fn test_has_media() {
        let mut tweet = make_tweet();
        let card = TweetCard::new(&tweet);
        assert!(!card.has_media());

        tweet.media.push(crate::model::TweetMedia {
            id: "1".into(),
            media_type: "photo".into(),
            url: "https://example.com/photo.jpg".into(),
            local_path: None,
        });
        let card = TweetCard::new(&tweet);
        assert!(card.has_media());
    }

    #[test]
    fn test_is_reply() {
        let mut tweet = make_tweet();
        let card = TweetCard::new(&tweet);
        assert!(!card.is_reply());

        tweet.in_reply_to_status_id = Some("999".into());
        let card = TweetCard::new(&tweet);
        assert!(card.is_reply());
    }

    #[test]
    fn test_hashtag_regex() {
        assert!(HASHTAG_REGEX.is_match("#rust"));
        assert!(HASHTAG_REGEX.is_match("#Rust2024"));
        assert!(!HASHTAG_REGEX.is_match("rust"));
    }

    #[test]
    fn test_mention_regex() {
        assert!(MENTION_REGEX.is_match("@user"));
        assert!(MENTION_REGEX.is_match("@Some_User123"));
        // Note: @domain in email@domain.com does match - this is expected
        // because we only check for @ followed by word chars
        assert!(!MENTION_REGEX.is_match("no mention here"));
    }

    #[test]
    fn test_url_regex() {
        assert!(URL_REGEX.is_match("https://example.com"));
        assert!(URL_REGEX.is_match("http://test.org/path"));
        assert!(!URL_REGEX.is_match("example.com"));
    }

    #[test]
    fn test_render_content_styling() {
        let tweet = make_tweet();
        let card = TweetCard::new(&tweet);
        let theme = Theme::for_terminal();

        let content = card.render_content(&theme);

        // Should contain styling markup
        assert!(content.contains('['));
        assert!(content.contains(']'));
        // Should still contain original content
        assert!(content.contains("rust"));
        assert!(content.contains("rustlang"));
    }

    #[test]
    fn test_render_header() {
        let tweet = make_tweet();
        let card = TweetCard::new(&tweet);
        let theme = Theme::for_terminal();

        let header = card.render_header(&theme);

        // Should contain ID (truncated)
        assert!(header.contains("1234"));
    }

    #[test]
    fn test_render_header_with_reply() {
        let mut tweet = make_tweet();
        tweet.in_reply_to_status_id = Some("999".into());

        let card = TweetCard::new(&tweet);
        let theme = Theme::for_terminal();

        let header = card.render_header(&theme);
        assert!(header.contains("Reply"));
    }

    #[test]
    fn test_render_header_with_retweet() {
        let mut tweet = make_tweet();
        tweet.is_retweet = true;

        let card = TweetCard::new(&tweet);
        let theme = Theme::for_terminal();

        let header = card.render_header(&theme);
        assert!(header.contains("RT"));
    }

    #[test]
    fn test_render_engagement() {
        let tweet = make_tweet();
        let card = TweetCard::new(&tweet);
        let theme = Theme::for_terminal();

        let engagement = card.render_engagement(&theme);
        assert!(engagement.contains("100"));
        assert!(engagement.contains("50"));
        assert!(engagement.contains("likes"));
        assert!(engagement.contains("retweets"));
    }

    #[test]
    fn test_render_no_panic_styled() {
        let tweet = make_tweet();
        let card = TweetCard::new(&tweet);
        let output = Output::new_tty();

        // Should not panic
        card.render(&output);
    }

    #[test]
    fn test_render_no_panic_plain() {
        let tweet = make_tweet();
        let card = TweetCard::new(&tweet);
        let output = Output::new_piped();

        // Should not panic
        card.render(&output);
    }

    #[test]
    fn test_highlight_terms() {
        let tweet = Tweet {
            full_text: "The quick brown fox jumps over the lazy dog".into(),
            ..make_tweet()
        };

        let card = TweetCard::new(&tweet).highlight_terms(vec!["quick".into(), "fox".into()]);

        let theme = Theme::for_terminal();
        let content = card.render_content(&theme);

        // Should contain highlight markup for matched terms
        assert!(content.contains('['));
    }

    #[test]
    fn test_conversation_thread_empty() {
        let tweets: Vec<Tweet> = vec![];
        let thread = ConversationThread::new(&tweets);
        let output = Output::new_piped();

        // Should not panic on empty
        thread.render(&output);
    }

    #[test]
    fn test_conversation_thread_single() {
        let tweets = vec![make_tweet()];
        let thread = ConversationThread::new(&tweets);
        let output = Output::new_piped();

        // Should not panic
        thread.render(&output);
    }

    #[test]
    fn test_conversation_thread_multiple() {
        let tweets = vec![make_tweet(), make_tweet(), make_tweet()];
        let thread = ConversationThread::new(&tweets).highlight_terms(vec!["rust".into()]);
        let output = Output::new_piped();

        // Should not panic
        thread.render(&output);
    }
}
