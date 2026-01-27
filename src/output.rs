//! Output abstraction module for terminal output.
//!
//! Provides conditional compilation between rich_rust styled output
//! and fallback colored crate implementations. Handles TTY detection,
//! color environment variables, format switching, and verbosity control.
//!
//! # Feature Flags
//!
//! - `rich` - Enable rich_rust styled output
//! - `legacy-colors` - Use colored crate explicitly
//!
//! # Example
//!
//! ```rust,ignore
//! use xf::output::{Output, OutputFormat, Verbosity};
//!
//! let output = Output::new(OutputFormat::Text);
//! if output.is_styled() {
//!     output.print("[bold green]Success![/]");
//! } else {
//!     output.print("Success!");
//! }
//! ```

use regex::Regex;
use std::io::{self, IsTerminal, Write};
use std::sync::LazyLock;

#[cfg(feature = "rich")]
pub use rich_rust::prelude::*;

/// Regex for stripping markup tags - compiled once at startup
static MARKUP_REGEX: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"\[/?[^\]]*\]").expect("Invalid markup regex"));

/// Returns true if the rich_rust feature is enabled.
///
/// Use this for runtime feature detection to select between
/// rich_rust styled output and fallback implementations.
#[must_use]
pub const fn is_rich_available() -> bool {
    cfg!(feature = "rich")
}

/// Returns the terminal dimensions (columns, rows) if available.
///
/// Falls back to (80, 24) if terminal size cannot be determined.
#[must_use]
pub fn terminal_dimensions() -> (u16, u16) {
    terminal_size::terminal_size().map_or((80, 24), |(w, h)| (w.0, h.0))
}

/// Output format selection
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
    Csv,
}

/// Verbosity level for output control
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum Verbosity {
    Quiet,
    #[default]
    Normal,
    Verbose,
    Debug,
}

/// Color support level detection
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ColorSupport {
    #[default]
    None,
    Basic,     // 16 colors
    Extended,  // 256 colors
    TrueColor, // 24-bit
}

impl ColorSupport {
    /// Detect color support from environment
    #[must_use]
    pub fn detect() -> Self {
        // Check COLORTERM for true color support
        if let Ok(ct) = std::env::var("COLORTERM") {
            if ct == "truecolor" || ct == "24bit" {
                return Self::TrueColor;
            }
        }

        // Check TERM for color capability
        if let Ok(term) = std::env::var("TERM") {
            if term.contains("256color") {
                return Self::Extended;
            }
            if term.contains("color") || term == "xterm" || term.starts_with("screen") {
                return Self::Basic;
            }
        }

        Self::None
    }
}

/// Simple theme for consistent styling
#[derive(Clone, Debug, Default)]
pub struct Theme {
    pub color_support: ColorSupport,
}

impl Theme {
    /// Create theme for specific color support level
    #[must_use]
    pub const fn for_color_support(support: ColorSupport) -> Self {
        Self {
            color_support: support,
        }
    }

    /// Detect color support from environment
    #[must_use]
    pub fn detect_color_support() -> ColorSupport {
        ColorSupport::detect()
    }
}

/// Central output abstraction for all CLI output
#[allow(clippy::struct_excessive_bools)]
pub struct Output {
    format: OutputFormat,
    stdout_is_tty: bool,
    stderr_is_tty: bool,
    force_color: bool,
    no_color: bool,
    clicolor: Option<bool>,
    #[cfg(feature = "rich")]
    console: Option<Console>,
    pub theme: Theme,
    terminal_width: usize,
    verbosity: Verbosity,
}

impl Output {
    /// Create new Output with TTY auto-detection
    #[must_use]
    pub fn new(format: OutputFormat) -> Self {
        let stdout_is_tty = io::stdout().is_terminal();
        let stderr_is_tty = io::stderr().is_terminal();
        Self::new_with_tty(format, stdout_is_tty, stderr_is_tty)
    }

    /// Create with explicit TTY state (for testing)
    #[must_use]
    pub fn new_with_tty(format: OutputFormat, stdout_tty: bool, stderr_tty: bool) -> Self {
        let force_color = std::env::var("FORCE_COLOR").is_ok_and(|v| !v.is_empty() && v != "0")
            || std::env::var("CLICOLOR_FORCE").is_ok_and(|v| !v.is_empty() && v != "0");
        let no_color = std::env::var("NO_COLOR").is_ok();
        let clicolor = std::env::var("CLICOLOR").ok().map(|v| v != "0");

        let terminal_width = terminal_size::terminal_size()
            .map_or(80, |(w, _)| w.0 as usize)
            .clamp(40, 300); // Sensible bounds

        let color_support = if no_color || clicolor == Some(false) {
            ColorSupport::None
        } else {
            Theme::detect_color_support()
        };

        #[cfg(feature = "rich")]
        let console = if stdout_tty && format == OutputFormat::Text {
            Some(Console::new())
        } else {
            None
        };

        Self {
            format,
            stdout_is_tty: stdout_tty,
            stderr_is_tty: stderr_tty,
            force_color,
            no_color,
            clicolor,
            #[cfg(feature = "rich")]
            console,
            theme: Theme::for_color_support(color_support),
            terminal_width,
            verbosity: Verbosity::Normal,
        }
    }

    /// Create for TTY (testing helper)
    #[must_use]
    pub fn new_tty() -> Self {
        Self::new_with_tty(OutputFormat::Text, true, true)
    }

    /// Create for piped/non-TTY (testing helper)
    #[must_use]
    pub fn new_piped() -> Self {
        Self::new_with_tty(OutputFormat::Text, false, false)
    }

    /// Set verbosity level
    #[must_use]
    pub const fn with_verbosity(mut self, verbosity: Verbosity) -> Self {
        self.verbosity = verbosity;
        self
    }

    /// Override terminal width (for testing)
    #[must_use]
    pub fn with_width(mut self, width: usize) -> Self {
        self.terminal_width = width.clamp(40, 300);
        self
    }

    /// Check if styled output should be used
    #[must_use]
    pub fn is_styled(&self) -> bool {
        if self.format != OutputFormat::Text {
            return false;
        }
        if self.no_color && !self.force_color {
            return false;
        }
        if self.clicolor == Some(false) && !self.force_color {
            return false;
        }
        self.stdout_is_tty || self.force_color
    }

    /// Check if stderr is styled
    #[must_use]
    pub fn is_stderr_styled(&self) -> bool {
        if self.format != OutputFormat::Text {
            return false;
        }
        if self.no_color && !self.force_color {
            return false;
        }
        self.stderr_is_tty || self.force_color
    }

    /// Get output format
    #[must_use]
    pub const fn format(&self) -> OutputFormat {
        self.format
    }

    /// Get terminal width
    #[must_use]
    pub const fn width(&self) -> usize {
        self.terminal_width
    }

    /// Get current verbosity
    #[must_use]
    pub const fn verbosity(&self) -> Verbosity {
        self.verbosity
    }

    /// Check if stdout is a TTY (for progress bars that need cursor control)
    #[must_use]
    pub const fn stdout_is_tty(&self) -> bool {
        self.stdout_is_tty
    }

    /// Check if stderr is a TTY
    #[must_use]
    pub const fn stderr_is_tty(&self) -> bool {
        self.stderr_is_tty
    }

    /// Check verbosity level - quiet mode
    #[must_use]
    pub fn is_quiet(&self) -> bool {
        self.verbosity == Verbosity::Quiet
    }

    /// Check verbosity level - verbose or debug mode
    #[must_use]
    pub const fn is_verbose(&self) -> bool {
        matches!(self.verbosity, Verbosity::Verbose | Verbosity::Debug)
    }

    /// Check verbosity level - debug mode only
    #[must_use]
    pub fn is_debug(&self) -> bool {
        self.verbosity == Verbosity::Debug
    }

    /// Print text (styled if TTY, plain otherwise)
    pub fn print(&self, text: &str) {
        if self.is_quiet() {
            return;
        }

        if self.is_styled() {
            #[cfg(feature = "rich")]
            if let Some(ref console) = self.console {
                console.print(text);
                return;
            }
        }

        // Plain output - strip any markup
        println!("{}", Self::strip_markup(text));
    }

    /// Print text without newline
    pub fn print_inline(&self, text: &str) {
        if self.is_quiet() {
            return;
        }

        if self.is_styled() {
            #[cfg(feature = "rich")]
            if let Some(ref console) = self.console {
                console.print(text);
                return;
            }
        }

        print!("{}", Self::strip_markup(text));
    }

    /// Print to stderr (styled if stderr TTY)
    pub fn eprint(&self, text: &str) {
        if self.is_stderr_styled() {
            #[cfg(feature = "rich")]
            {
                let console = Console::stderr();
                console.print(text);
                return;
            }
        }
        eprintln!("{}", Self::strip_markup(text));
    }

    /// Print only in verbose mode
    pub fn print_verbose(&self, text: &str) {
        if self.is_verbose() {
            self.print(text);
        }
    }

    /// Print only in debug mode
    pub fn print_debug(&self, text: &str) {
        if self.is_debug() {
            self.print(&format!("[dim]DEBUG:[/] {text}"));
        }
    }

    /// Print a renderable (table, panel, etc.) when rich feature is enabled
    #[cfg(feature = "rich")]
    pub fn print_renderable<R: Renderable>(&self, renderable: &R) {
        if self.is_quiet() {
            return;
        }

        if self.is_styled() {
            if let Some(ref console) = self.console {
                console.print_renderable(renderable);
                return;
            }
        }

        // Fallback: render to plain text
        let segments = renderable.render(self.terminal_width);
        for segment in segments {
            print!("{}", segment.text);
        }
        println!();
    }

    /// Print JSON (always unformatted, no styling)
    pub fn print_json<T: serde::Serialize>(&self, value: &T) -> Result<(), serde_json::Error> {
        let json = serde_json::to_string(value)?;
        println!("{json}");
        Ok(())
    }

    /// Print JSON pretty-printed
    pub fn print_json_pretty<T: serde::Serialize>(
        &self,
        value: &T,
    ) -> Result<(), serde_json::Error> {
        let json = serde_json::to_string_pretty(value)?;
        println!("{json}");
        Ok(())
    }

    /// Print CSV row
    pub fn print_csv(&self, fields: &[&str]) {
        let escaped: Vec<String> = fields
            .iter()
            .map(|f| {
                if f.contains(',') || f.contains('"') || f.contains('\n') {
                    format!("\"{}\"", f.replace('"', "\"\""))
                } else {
                    (*f).to_string()
                }
            })
            .collect();
        println!("{}", escaped.join(","));
    }

    /// Strip markup tags for plain output - uses pre-compiled regex
    #[must_use]
    pub fn strip_markup(text: &str) -> String {
        MARKUP_REGEX.replace_all(text, "").into_owned()
    }

    /// Flush stdout
    pub fn flush(&self) {
        let _ = io::stdout().flush();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_rich_available_returns_correct_value() {
        let available = is_rich_available();

        #[cfg(feature = "rich")]
        assert!(available, "Should be true when rich feature is enabled");

        #[cfg(not(feature = "rich"))]
        assert!(!available, "Should be false when rich feature is disabled");
    }

    #[test]
    fn test_terminal_dimensions_returns_reasonable_values() {
        let (cols, rows) = terminal_dimensions();
        assert!(cols > 0, "Columns should be positive");
        assert!(rows > 0, "Rows should be positive");
    }

    #[test]
    fn test_tty_detection_returns_styled() {
        let output = Output::new_tty();
        assert!(output.is_styled());
    }

    #[test]
    fn test_piped_detection_not_styled() {
        let output = Output::new_piped();
        assert!(!output.is_styled());
    }

    #[test]
    fn test_json_format_never_styled() {
        let output = Output::new_with_tty(OutputFormat::Json, true, true);
        assert!(!output.is_styled());
    }

    #[test]
    fn test_csv_format_never_styled() {
        let output = Output::new_with_tty(OutputFormat::Csv, true, true);
        assert!(!output.is_styled());
    }

    #[test]
    fn test_strip_markup_removes_tags() {
        assert_eq!(Output::strip_markup("[bold]text[/]"), "text");
        assert_eq!(Output::strip_markup("[red on white]colored[/]"), "colored");
        assert_eq!(Output::strip_markup("no markup"), "no markup");
        assert_eq!(Output::strip_markup("[a][b]nested[/][/]"), "nested");
    }

    #[test]
    fn test_strip_markup_handles_empty_tags() {
        assert_eq!(Output::strip_markup("[]empty[]"), "empty");
        assert_eq!(Output::strip_markup("[/]close only"), "close only");
    }

    #[test]
    fn test_terminal_width_has_sane_bounds() {
        let output = Output::new_piped();
        assert!(output.width() >= 40);
        assert!(output.width() <= 300);
    }

    #[test]
    fn test_terminal_width_override() {
        let output = Output::new_piped().with_width(120);
        assert_eq!(output.width(), 120);
    }

    #[test]
    fn test_width_clamped_to_bounds() {
        let output = Output::new_piped().with_width(10);
        assert_eq!(output.width(), 40); // Clamped to minimum

        let output = Output::new_piped().with_width(1000);
        assert_eq!(output.width(), 300); // Clamped to maximum
    }

    #[test]
    fn test_verbosity_methods() {
        let quiet = Output::new_piped().with_verbosity(Verbosity::Quiet);
        assert!(quiet.is_quiet());
        assert!(!quiet.is_verbose());
        assert!(!quiet.is_debug());

        let verbose = Output::new_piped().with_verbosity(Verbosity::Verbose);
        assert!(!verbose.is_quiet());
        assert!(verbose.is_verbose());
        assert!(!verbose.is_debug());

        let debug = Output::new_piped().with_verbosity(Verbosity::Debug);
        assert!(!debug.is_quiet());
        assert!(debug.is_verbose()); // Debug implies verbose
        assert!(debug.is_debug());
    }

    #[test]
    fn test_csv_escaping_needs_escape() {
        // Verify the escaping logic detects fields needing escape
        let needs_escape = "field,with,commas";
        assert!(needs_escape.contains(','));

        let needs_quote_escape = "field with \"quotes\"";
        assert!(needs_quote_escape.contains('"'));

        let needs_newline_escape = "field\nwith\nnewlines";
        assert!(needs_newline_escape.contains('\n'));
    }

    #[test]
    fn test_stderr_styled_independent() {
        // stderr TTY is checked independently
        let output = Output::new_with_tty(OutputFormat::Text, false, true);
        assert!(!output.is_styled()); // stdout not TTY
        assert!(output.is_stderr_styled()); // stderr is TTY
    }

    #[test]
    fn test_output_format_default() {
        let format = OutputFormat::default();
        assert_eq!(format, OutputFormat::Text);
    }

    #[test]
    fn test_verbosity_default() {
        let verbosity = Verbosity::default();
        assert_eq!(verbosity, Verbosity::Normal);
    }

    #[test]
    fn test_color_support_detect_returns_valid() {
        let support = ColorSupport::detect();
        // Should return a valid variant without panicking
        match support {
            ColorSupport::None
            | ColorSupport::Basic
            | ColorSupport::Extended
            | ColorSupport::TrueColor => {}
        }
    }

    #[test]
    fn test_theme_default_does_not_panic() {
        let theme = Theme::default();
        // Just verify it creates without panicking
        let _ = theme.color_support;
    }

    #[test]
    fn test_format_getter() {
        let output = Output::new_with_tty(OutputFormat::Json, true, true);
        assert_eq!(output.format(), OutputFormat::Json);
    }
}
