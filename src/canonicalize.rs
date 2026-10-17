//! Text canonicalization for consistent embeddings.
//!
//! This module provides a preprocessing pipeline that normalizes text before
//! embedding to ensure deterministic results and filter out noise.
//!
//! # Pipeline
//!
//! 1. **Unicode NFC normalization** - Ensures consistent character representation
//! 2. **Markdown stripping** - Removes formatting syntax (bold, italic, links)
//! 3. **Code block collapsing** - Keeps first 20 + last 10 lines of code
//! 4. **Whitespace normalization** - Collapses runs, trims edges
//! 5. **Low-signal filtering** - Removes short acknowledgments ("OK", "Done")
//! 6. **Truncation** - Limits to 2000 characters
//!
//! # Usage
//!
//! ```ignore
//! use xf::canonicalize::canonicalize_for_embedding;
//!
//! let text = "**Bold** text with `code`";
//! let canonical = canonicalize_for_embedding(text);
//! assert_eq!(canonical, "Bold text with code");
//! ```

use ring::digest::{self, SHA256};
use unicode_normalization::UnicodeNormalization;

/// Maximum characters to keep after canonicalization.
pub const MAX_EMBED_CHARS: usize = 2000;

/// Number of lines to keep from the start of code blocks.
const CODE_HEAD_LINES: usize = 20;

/// Number of lines to keep from the end of code blocks.
const CODE_TAIL_LINES: usize = 10;

/// Low-signal content that should be filtered out entirely.
const LOW_SIGNAL_CONTENT: &[&str] = &[
    "ok",
    "done",
    "done.",
    "got it",
    "got it.",
    "understood",
    "understood.",
    "sure",
    "sure.",
    "yes",
    "no",
    "thanks",
    "thanks.",
    "thank you",
    "thank you.",
    "lgtm",
    "👍",
    "✓",
];

/// Canonicalize text for embedding.
///
/// Applies the full preprocessing pipeline to ensure consistent,
/// deterministic embeddings.
#[must_use]
pub fn canonicalize_for_embedding(text: &str) -> String {
    // Step 1: Unicode NFC normalization
    let normalized: String = text.nfc().collect();

    // Step 2: Strip markdown and collapse code blocks
    let stripped = strip_markdown_and_code(&normalized);

    // Step 3: Normalize whitespace
    let whitespace_normalized = normalize_whitespace(&stripped);

    // Step 4: Filter low-signal content
    let filtered = filter_low_signal(&whitespace_normalized);

    // Step 5: Truncate to max length
    truncate_to_chars(&filtered, MAX_EMBED_CHARS)
}

/// Compute SHA256 hash of text for deduplication.
#[must_use]
pub fn content_hash(text: &str) -> [u8; 32] {
    let digest = digest::digest(&SHA256, text.as_bytes());
    let mut hash = [0u8; 32];
    hash.copy_from_slice(digest.as_ref());
    hash
}

/// Compute SHA256 hash and return as hex string.
#[must_use]
pub fn content_hash_hex(text: &str) -> String {
    let hash = content_hash(text);
    hex_encode(&hash)
}

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;

    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

/// Strip markdown formatting and collapse code blocks.
fn strip_markdown_and_code(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_code_block = false;
    let mut code_block_lang = String::new();
    let mut code_lines: Vec<&str> = Vec::new();

    for line in text.lines() {
        if line.starts_with("```") {
            if in_code_block {
                // End of code block - collapse it
                result.push_str(&collapse_code_block(&code_block_lang, &code_lines));
                result.push('\n');
                code_lines.clear();
                code_block_lang.clear();
                in_code_block = false;
            } else {
                // Start of code block
                in_code_block = true;
                code_block_lang = line.trim_start_matches('`').trim().to_string();
            }
        } else if in_code_block {
            code_lines.push(line);
        } else {
            // Strip markdown from regular text
            let stripped = strip_markdown_line(line);
            if !stripped.is_empty() {
                result.push_str(&stripped);
                result.push('\n');
            }
        }
    }

    // Handle unclosed code block
    if in_code_block && !code_lines.is_empty() {
        result.push_str(&collapse_code_block(&code_block_lang, &code_lines));
        result.push('\n');
    }

    result
}

/// Collapse a code block to head + tail lines.
fn collapse_code_block(lang: &str, lines: &[&str]) -> String {
    let lang_label = if lang.is_empty() {
        "code".to_string()
    } else {
        format!("code: {lang}")
    };

    if lines.len() <= CODE_HEAD_LINES + CODE_TAIL_LINES {
        // Short enough to keep in full
        format!("[{lang_label}] {}", lines.join(" "))
    } else {
        // Collapse middle: first N + last M lines
        let head: Vec<_> = lines.iter().take(CODE_HEAD_LINES).copied().collect();
        let tail: Vec<_> = lines
            .iter()
            .skip(lines.len() - CODE_TAIL_LINES)
            .copied()
            .collect();
        let omitted = lines.len() - CODE_HEAD_LINES - CODE_TAIL_LINES;
        format!(
            "[{lang_label}] {} [...{omitted} lines...] {}",
            head.join(" "),
            tail.join(" ")
        )
    }
}

/// Strip markdown formatting from a single line.
fn strip_markdown_line(line: &str) -> String {
    let mut result = line.to_string();

    // Remove bold/italic markers
    result = result.replace("**", "");
    result = result.replace("__", "");
    result = result.replace('*', "");
    result = result.replace('_', " ");

    // Remove inline code backticks
    result = result.replace('`', "");

    // Convert links [text](url) to just text
    result = strip_markdown_links(&result);

    // Remove headers (# prefix)
    result = result.trim_start_matches('#').trim_start().to_string();

    // Remove blockquote prefix
    result = result.trim_start_matches('>').trim_start().to_string();

    // Remove list markers
    result = strip_list_marker(&result);

    result
}

/// Remove markdown link syntax, keeping only the text.
fn strip_markdown_links(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '[' {
            // Collect link text until ]
            let mut link_text = String::new();
            let mut found_close = false;
            for ch in chars.by_ref() {
                if ch == ']' {
                    found_close = true;
                    break;
                }
                link_text.push(ch);
            }

            if found_close && chars.peek() == Some(&'(') {
                // Skip the URL in parentheses
                chars.next(); // consume '('
                let mut depth = 1;
                for ch in chars.by_ref() {
                    if ch == '(' {
                        depth += 1;
                    } else if ch == ')' {
                        depth -= 1;
                        if depth == 0 {
                            break;
                        }
                    }
                }
                result.push_str(&link_text);
            } else {
                // Not a link, keep the brackets
                result.push('[');
                result.push_str(&link_text);
                if found_close {
                    result.push(']');
                }
            }
        } else {
            result.push(c);
        }
    }

    result
}

/// Remove list markers (-, *, 1., etc.) from the start of a line.
fn strip_list_marker(line: &str) -> String {
    let trimmed = line.trim_start();

    // Unordered list markers
    if trimmed.starts_with("- ") || trimmed.starts_with("* ") || trimmed.starts_with("+ ") {
        return trimmed[2..].to_string();
    }

    // Ordered list markers (1., 2., etc.)
    if let Some(dot_pos) = trimmed.find('.') {
        let prefix = &trimmed[..dot_pos];
        if prefix.chars().all(|c| c.is_ascii_digit()) && trimmed.len() > dot_pos + 1 {
            let after_dot = &trimmed[dot_pos + 1..];
            if let Some(stripped) = after_dot.strip_prefix(' ') {
                return stripped.to_string();
            }
        }
    }

    line.to_string()
}

/// Normalize whitespace: collapse runs, trim edges.
fn normalize_whitespace(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut prev_whitespace = true; // Start as true to trim leading

    for c in text.chars() {
        if c.is_whitespace() {
            if !prev_whitespace {
                result.push(' ');
                prev_whitespace = true;
            }
        } else {
            result.push(c);
            prev_whitespace = false;
        }
    }

    // Trim trailing whitespace
    result.trim_end().to_string()
}

/// Filter out low-signal content.
fn filter_low_signal(text: &str) -> String {
    let trimmed = text.trim();
    let lower = trimmed.to_lowercase();

    // If entire text is low-signal, return empty
    for pattern in LOW_SIGNAL_CONTENT {
        if lower == *pattern {
            return String::new();
        }
    }

    text.to_string()
}

/// Truncate text to a maximum number of characters.
fn truncate_to_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        text.to_string()
    } else {
        text.chars().take(max_chars).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unicode_normalization() {
        // Composed vs decomposed form should produce same result
        let composed = "café";
        let decomposed = "cafe\u{0301}"; // e + combining acute accent

        let c1 = canonicalize_for_embedding(composed);
        let c2 = canonicalize_for_embedding(decomposed);

        assert_eq!(c1, c2);
    }

    #[test]
    fn test_strip_bold_italic() {
        let text = "This is **bold** and *italic* and __also bold__";
        let result = canonicalize_for_embedding(text);
        assert!(!result.contains('*'));
        assert!(result.contains("bold"));
        assert!(result.contains("italic"));
    }

    #[test]
    fn test_strip_inline_code() {
        let text = "Use the `print` function";
        let result = canonicalize_for_embedding(text);
        assert!(!result.contains('`'));
        assert!(result.contains("print"));
    }

    #[test]
    fn test_strip_links() {
        let text = "Check out [this link](https://example.com) for more";
        let result = canonicalize_for_embedding(text);
        assert!(!result.contains("https://"));
        assert!(result.contains("this link"));
    }

    #[test]
    fn test_strip_headers() {
        let text = "# Header\n## Subheader\nContent";
        let result = canonicalize_for_embedding(text);
        assert!(!result.starts_with('#'));
        assert!(result.contains("Header"));
        assert!(result.contains("Content"));
    }

    #[test]
    fn test_strip_list_markers() {
        let text = "- Item one\n* Item two\n1. Item three";
        let result = canonicalize_for_embedding(text);
        assert!(result.contains("Item one"));
        assert!(result.contains("Item two"));
        assert!(result.contains("Item three"));
    }

    #[test]
    fn test_code_block_collapse() {
        let mut lines: Vec<&str> = (0..50).map(|_| "code line").collect();
        let result = collapse_code_block("rust", &lines);

        assert!(result.contains("[code: rust]"));
        assert!(result.contains("..."));

        // Short code block should not be collapsed
        lines.truncate(10);
        let result = collapse_code_block("rust", &lines);
        assert!(!result.contains("..."));
    }

    #[test]
    fn test_whitespace_normalization() {
        let text = "Multiple   spaces\t\tand\n\nnewlines";
        let result = normalize_whitespace(text);
        assert!(!result.contains("  ")); // No double spaces
        assert!(result.contains(' ')); // Single spaces preserved
    }

    #[test]
    fn test_low_signal_filtering() {
        let low_signal = ["ok", "Done.", "Thanks", "LGTM", "👍"];
        for text in low_signal {
            let result = canonicalize_for_embedding(text);
            assert!(result.is_empty(), "Expected '{text}' to be filtered");
        }

        // Non-low-signal should pass through
        let result = canonicalize_for_embedding("This is actual content");
        assert_ne!(result, "");
    }

    #[test]
    fn test_truncation() {
        let long_text: String = "a".repeat(3000);
        let result = canonicalize_for_embedding(&long_text);
        assert_eq!(result.chars().count(), MAX_EMBED_CHARS);
    }

    #[test]
    fn test_content_hash() {
        let text = "Hello, world!";
        let hash = content_hash(text);
        assert_eq!(hash.len(), 32);

        // Same input should produce same hash
        let hash2 = content_hash(text);
        assert_eq!(hash, hash2);

        // Different input should produce different hash
        let hash3 = content_hash("Different text");
        assert_ne!(hash, hash3);
    }

    #[test]
    fn test_content_hash_hex() {
        let text = "test";
        let hex = content_hash_hex(text);
        assert_eq!(hex.len(), 64); // 32 bytes * 2 hex chars
        assert!(hex.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_full_pipeline() {
        let text = r#"
# Important Note

This is **bold** and has a [link](https://example.com).

```rust
fn main() {
    println!("Hello");
}
```

Thanks for reading!
"#;

        let result = canonicalize_for_embedding(text);

        // Should have stripped markdown
        assert!(!result.contains("**"));
        assert!(!result.contains("```"));
        assert!(!result.contains("https://"));

        // Should preserve content
        assert!(result.contains("Important Note"));
        assert!(result.contains("bold"));
        assert!(result.contains("link"));
        assert!(result.contains("code"));
    }

    #[test]
    fn test_empty_input() {
        let result = canonicalize_for_embedding("");
        assert_eq!(result, "");
    }

    #[test]
    fn test_whitespace_only_input() {
        let result = canonicalize_for_embedding("   \n\t  ");
        assert_eq!(result, "");
    }
}
