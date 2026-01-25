//! Theme system for consistent terminal styling.
//!
//! Provides semantic color palettes and style definitions that adapt to
//! terminal color support levels (TrueColor, 256-color, 16-color, no-color).
//!
//! # Example
//!
//! ```rust,ignore
//! use xf::theme::Theme;
//!
//! let theme = Theme::for_terminal();
//! println!("{}", theme.format_success("Operation completed"));
//! ```

use crate::output::ColorSupport;

/// Comprehensive theme for consistent terminal styling.
///
/// Stores rich_rust markup style strings for different semantic elements.
/// The Output module interprets these when rendering.
#[derive(Clone, Debug)]
pub struct Theme {
    // Primary palette
    pub primary: &'static str,
    pub secondary: &'static str,
    pub accent: &'static str,
    pub muted: &'static str,

    // Semantic colors
    pub success: &'static str,
    pub warning: &'static str,
    pub error: &'static str,
    pub info: &'static str,

    // Text hierarchy
    pub heading: &'static str,
    pub body: &'static str,
    pub caption: &'static str,

    // UI element colors
    pub border: &'static str,
    pub highlight: &'static str,

    // Tweet-specific colors
    pub username: &'static str,
    pub hashtag: &'static str,
    pub mention: &'static str,
    pub link: &'static str,
    pub timestamp: &'static str,

    // Score thresholds
    pub score_high: &'static str,
    pub score_medium: &'static str,
    pub score_low: &'static str,

    // Color support level
    pub color_support: ColorSupport,

    // Accessibility flag
    pub high_contrast: bool,
}

impl Default for Theme {
    fn default() -> Self {
        Self::dark()
    }
}

impl Theme {
    /// Dark theme with TrueColor support (default)
    #[must_use]
    pub const fn dark() -> Self {
        Self {
            primary: "bright_blue",
            secondary: "magenta",
            accent: "yellow",
            muted: "bright_black",

            success: "green",
            warning: "yellow",
            error: "red",
            info: "cyan",

            heading: "bold white",
            body: "white",
            caption: "bright_black",

            border: "bright_black",
            highlight: "bold yellow",

            username: "bright_blue",
            hashtag: "cyan",
            mention: "magenta",
            link: "bright_cyan",
            timestamp: "bright_black",

            score_high: "bold green",
            score_medium: "bold yellow",
            score_low: "bold red",

            color_support: ColorSupport::TrueColor,
            high_contrast: false,
        }
    }

    /// Dark theme for 256-color terminals
    #[must_use]
    pub const fn dark_256() -> Self {
        Self {
            primary: "blue",
            secondary: "magenta",
            accent: "yellow",
            muted: "bright_black",

            success: "green",
            warning: "yellow",
            error: "red",
            info: "cyan",

            heading: "bold white",
            body: "white",
            caption: "bright_black",

            border: "bright_black",
            highlight: "bold yellow",

            username: "blue",
            hashtag: "cyan",
            mention: "magenta",
            link: "cyan",
            timestamp: "bright_black",

            score_high: "bold green",
            score_medium: "bold yellow",
            score_low: "bold red",

            color_support: ColorSupport::Extended,
            high_contrast: false,
        }
    }

    /// Light theme
    #[must_use]
    pub const fn light() -> Self {
        Self {
            primary: "blue",
            secondary: "magenta",
            accent: "yellow",
            muted: "bright_black",

            success: "green",
            warning: "yellow",
            error: "red",
            info: "cyan",

            heading: "bold black",
            body: "black",
            caption: "bright_black",

            border: "bright_black",
            highlight: "bold yellow",

            username: "blue",
            hashtag: "blue",
            mention: "magenta",
            link: "cyan",
            timestamp: "bright_black",

            score_high: "bold green",
            score_medium: "bold yellow",
            score_low: "bold red",

            color_support: ColorSupport::TrueColor,
            high_contrast: false,
        }
    }

    /// Minimal theme (basic 16 colors)
    #[must_use]
    pub const fn minimal() -> Self {
        Self {
            primary: "blue",
            secondary: "magenta",
            accent: "yellow",
            muted: "bright_black",

            success: "green",
            warning: "yellow",
            error: "red",
            info: "cyan",

            heading: "bold",
            body: "",
            caption: "dim",

            border: "dim",
            highlight: "bold",

            username: "blue",
            hashtag: "cyan",
            mention: "magenta",
            link: "cyan",
            timestamp: "dim",

            score_high: "green",
            score_medium: "yellow",
            score_low: "red",

            color_support: ColorSupport::Basic,
            high_contrast: false,
        }
    }

    /// High contrast theme (accessibility)
    #[must_use]
    pub const fn high_contrast() -> Self {
        Self {
            primary: "bright_cyan",
            secondary: "bright_magenta",
            accent: "bright_yellow",
            muted: "white",

            success: "bright_green",
            warning: "bright_yellow",
            error: "bright_red",
            info: "bright_cyan",

            heading: "bold bright_white",
            body: "bright_white",
            caption: "white",

            border: "white",
            highlight: "bold bright_yellow",

            username: "bright_cyan",
            hashtag: "bright_cyan",
            mention: "bright_magenta",
            link: "bright_cyan",
            timestamp: "white",

            score_high: "bold bright_green",
            score_medium: "bold bright_yellow",
            score_low: "bold bright_red",

            color_support: ColorSupport::Basic,
            high_contrast: true,
        }
    }

    /// No-color theme (plain text)
    #[must_use]
    pub const fn no_color() -> Self {
        Self {
            primary: "",
            secondary: "",
            accent: "",
            muted: "",

            success: "",
            warning: "",
            error: "",
            info: "",

            heading: "",
            body: "",
            caption: "",

            border: "",
            highlight: "",

            username: "",
            hashtag: "",
            mention: "",
            link: "",
            timestamp: "",

            score_high: "",
            score_medium: "",
            score_low: "",

            color_support: ColorSupport::None,
            high_contrast: false,
        }
    }

    /// List available built-in theme names
    #[must_use]
    pub const fn available_themes() -> &'static [&'static str] {
        &[
            "dark",
            "dark-256",
            "light",
            "minimal",
            "high-contrast",
            "no-color",
        ]
    }

    /// Get theme by name (case-insensitive, supports aliases)
    #[must_use]
    pub fn by_name(name: &str) -> Option<Self> {
        match name.to_lowercase().replace(['-', '_'], "").as_str() {
            "dark" => Some(Self::dark()),
            "dark256" => Some(Self::dark_256()),
            "light" => Some(Self::light()),
            "minimal" | "basic" => Some(Self::minimal()),
            "highcontrast" | "hc" => Some(Self::high_contrast()),
            "nocolor" | "none" => Some(Self::no_color()),
            _ => None,
        }
    }

    /// Get theme appropriate for detected color support
    #[must_use]
    pub const fn for_color_support(support: ColorSupport) -> Self {
        match support {
            ColorSupport::TrueColor => Self::dark(),
            ColorSupport::Extended => Self::dark_256(),
            ColorSupport::Basic => Self::minimal(),
            ColorSupport::None => Self::no_color(),
        }
    }

    /// Auto-detect best theme for current terminal
    #[must_use]
    pub fn for_terminal() -> Self {
        Self::for_color_support(ColorSupport::detect())
    }

    /// Get style markup for a score value (0.0 - 1.0)
    #[must_use]
    pub const fn score_style(&self, score: f32) -> &'static str {
        if score >= 0.8 {
            self.score_high
        } else if score >= 0.5 {
            self.score_medium
        } else {
            self.score_low
        }
    }

    /// Format text with heading style
    #[must_use]
    pub fn format_heading(&self, text: &str) -> String {
        if self.heading.is_empty() {
            text.to_string()
        } else {
            format!("[{}]{text}[/]", self.heading)
        }
    }

    /// Format text with error style
    #[must_use]
    pub fn format_error(&self, text: &str) -> String {
        if self.error.is_empty() {
            text.to_string()
        } else {
            format!("[{}]{text}[/]", self.error)
        }
    }

    /// Format text with warning style
    #[must_use]
    pub fn format_warning(&self, text: &str) -> String {
        if self.warning.is_empty() {
            text.to_string()
        } else {
            format!("[{}]{text}[/]", self.warning)
        }
    }

    /// Format text with success style
    #[must_use]
    pub fn format_success(&self, text: &str) -> String {
        if self.success.is_empty() {
            text.to_string()
        } else {
            format!("[{}]{text}[/]", self.success)
        }
    }

    /// Format text with info style
    #[must_use]
    pub fn format_info(&self, text: &str) -> String {
        if self.info.is_empty() {
            text.to_string()
        } else {
            format!("[{}]{text}[/]", self.info)
        }
    }

    /// Format text with muted style
    #[must_use]
    pub fn format_muted(&self, text: &str) -> String {
        if self.muted.is_empty() {
            text.to_string()
        } else {
            format!("[{}]{text}[/]", self.muted)
        }
    }

    /// Format text with primary color
    #[must_use]
    pub fn format_primary(&self, text: &str) -> String {
        if self.primary.is_empty() {
            text.to_string()
        } else {
            format!("[{}]{text}[/]", self.primary)
        }
    }

    /// Format text with score-based style
    #[must_use]
    pub fn format_score(&self, text: &str, score: f32) -> String {
        let style = self.score_style(score);
        if style.is_empty() {
            text.to_string()
        } else {
            format!("[{style}]{text}[/]")
        }
    }

    /// Format a username
    #[must_use]
    pub fn format_username(&self, text: &str) -> String {
        if self.username.is_empty() {
            text.to_string()
        } else {
            format!("[{}]{text}[/]", self.username)
        }
    }

    /// Format a hashtag
    #[must_use]
    pub fn format_hashtag(&self, text: &str) -> String {
        if self.hashtag.is_empty() {
            text.to_string()
        } else {
            format!("[{}]{text}[/]", self.hashtag)
        }
    }

    /// Format a mention
    #[must_use]
    pub fn format_mention(&self, text: &str) -> String {
        if self.mention.is_empty() {
            text.to_string()
        } else {
            format!("[{}]{text}[/]", self.mention)
        }
    }

    /// Format a link
    #[must_use]
    pub fn format_link(&self, text: &str) -> String {
        if self.link.is_empty() {
            text.to_string()
        } else {
            format!("[{}]{text}[/]", self.link)
        }
    }

    /// Format a timestamp
    #[must_use]
    pub fn format_timestamp(&self, text: &str) -> String {
        if self.timestamp.is_empty() {
            text.to_string()
        } else {
            format!("[{}]{text}[/]", self.timestamp)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_theme_is_dark() {
        let theme = Theme::default();
        assert_eq!(theme.color_support, ColorSupport::TrueColor);
    }

    #[test]
    fn test_all_builtin_themes_valid() {
        // Verify each theme can be created without panicking
        let themes = [
            Theme::dark(),
            Theme::dark_256(),
            Theme::light(),
            Theme::minimal(),
            Theme::high_contrast(),
            Theme::no_color(),
        ];
        for theme in themes {
            // Should not panic
            let _ = theme.format_heading("Test");
            let _ = theme.format_error("Test");
            let _ = theme.format_score("Test", 0.5);
        }
    }

    #[test]
    fn test_theme_by_name_finds_all() {
        for name in Theme::available_themes() {
            assert!(Theme::by_name(name).is_some(), "Theme '{name}' not found");
        }
    }

    #[test]
    fn test_theme_by_name_aliases() {
        assert!(Theme::by_name("dark256").is_some());
        assert!(Theme::by_name("dark-256").is_some());
        assert!(Theme::by_name("highcontrast").is_some());
        assert!(Theme::by_name("high-contrast").is_some());
        assert!(Theme::by_name("hc").is_some());
        assert!(Theme::by_name("nocolor").is_some());
        assert!(Theme::by_name("no-color").is_some());
        assert!(Theme::by_name("basic").is_some());
    }

    #[test]
    fn test_theme_by_name_unknown_returns_none() {
        assert!(Theme::by_name("nonexistent").is_none());
        assert!(Theme::by_name("").is_none());
    }

    #[test]
    fn test_score_style_thresholds() {
        let theme = Theme::dark();

        // High scores (>= 0.8)
        assert_eq!(theme.score_style(0.95), theme.score_high);
        assert_eq!(theme.score_style(0.80), theme.score_high);

        // Medium scores (>= 0.5, < 0.8)
        assert_eq!(theme.score_style(0.55), theme.score_medium);
        assert_eq!(theme.score_style(0.50), theme.score_medium);

        // Low scores (< 0.5)
        assert_eq!(theme.score_style(0.15), theme.score_low);
        assert_eq!(theme.score_style(0.49), theme.score_low);
    }

    #[test]
    fn test_high_contrast_flag() {
        assert!(Theme::high_contrast().high_contrast);
        assert!(!Theme::dark().high_contrast);
        assert!(!Theme::light().high_contrast);
    }

    #[test]
    fn test_no_color_theme_has_empty_styles() {
        let theme = Theme::no_color();
        assert_eq!(theme.color_support, ColorSupport::None);
        assert!(theme.primary.is_empty());
        assert!(theme.error.is_empty());
    }

    #[test]
    fn test_color_support_for_each_theme() {
        assert_eq!(Theme::dark().color_support, ColorSupport::TrueColor);
        assert_eq!(Theme::dark_256().color_support, ColorSupport::Extended);
        assert_eq!(Theme::minimal().color_support, ColorSupport::Basic);
        assert_eq!(Theme::no_color().color_support, ColorSupport::None);
    }

    #[test]
    fn test_for_color_support_returns_appropriate_theme() {
        let tc = Theme::for_color_support(ColorSupport::TrueColor);
        assert_eq!(tc.color_support, ColorSupport::TrueColor);

        let ext = Theme::for_color_support(ColorSupport::Extended);
        assert_eq!(ext.color_support, ColorSupport::Extended);

        let basic = Theme::for_color_support(ColorSupport::Basic);
        assert_eq!(basic.color_support, ColorSupport::Basic);

        let none = Theme::for_color_support(ColorSupport::None);
        assert_eq!(none.color_support, ColorSupport::None);
    }

    #[test]
    fn test_format_heading_with_style() {
        let theme = Theme::dark();
        let result = theme.format_heading("Title");
        assert!(result.contains("[bold white]"));
        assert!(result.contains("Title"));
        assert!(result.contains("[/]"));
    }

    #[test]
    fn test_format_heading_no_color() {
        let theme = Theme::no_color();
        let result = theme.format_heading("Title");
        assert_eq!(result, "Title");
        assert!(!result.contains('['));
    }

    #[test]
    fn test_format_score_high() {
        let theme = Theme::dark();
        let result = theme.format_score("0.95", 0.95);
        assert!(result.contains("[bold green]"));
    }

    #[test]
    fn test_format_score_medium() {
        let theme = Theme::dark();
        let result = theme.format_score("0.65", 0.65);
        assert!(result.contains("[bold yellow]"));
    }

    #[test]
    fn test_format_score_low() {
        let theme = Theme::dark();
        let result = theme.format_score("0.25", 0.25);
        assert!(result.contains("[bold red]"));
    }

    #[test]
    fn test_format_username() {
        let theme = Theme::dark();
        let result = theme.format_username("@user");
        assert!(result.contains("[bright_blue]"));
        assert!(result.contains("@user"));
    }

    #[test]
    fn test_format_hashtag() {
        let theme = Theme::dark();
        let result = theme.format_hashtag("#topic");
        assert!(result.contains("[cyan]"));
        assert!(result.contains("#topic"));
    }

    #[test]
    fn test_for_terminal_does_not_panic() {
        // Just verify it doesn't panic
        let theme = Theme::for_terminal();
        let _ = theme.color_support;
    }
}
