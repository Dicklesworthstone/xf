//! Output abstraction module for terminal output.
//!
//! Provides conditional compilation between rich_rust styled output
//! and fallback colored crate implementations.
//!
//! # Feature Flags
//!
//! - `rich` - Enable rich_rust styled output
//! - `legacy-colors` - Use colored crate explicitly
//!
//! # Example
//!
//! ```rust
//! use xf::output::is_rich_available;
//!
//! if is_rich_available() {
//!     println!("Using rich_rust for styled output");
//! } else {
//!     println!("Using fallback colored output");
//! }
//! ```

#[cfg(feature = "rich")]
pub use rich_rust::prelude::*;

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
}
