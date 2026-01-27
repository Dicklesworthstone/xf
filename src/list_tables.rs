//! Generic list table rendering for CLI list commands.
//!
//! Provides a `ListTable<T>` abstraction for rendering sorted, paginated tables
//! in both styled (rich_rust) and plain (TSV) modes.

use crate::output::Output;

/// Sort key extractor type alias to reduce complexity.
type SortKeyFn<T> = Box<dyn Fn(&T) -> i64>;

/// A column definition for a list table.
pub struct TableColumn<T> {
    name: String,
    extractor: Box<dyn Fn(&T) -> String>,
    sort_key: Option<SortKeyFn<T>>,
    right_align: bool,
}

impl<T> TableColumn<T> {
    /// Create a new column with a name and value extractor.
    pub fn new(name: &str, extractor: impl Fn(&T) -> String + 'static) -> Self {
        Self {
            name: name.to_string(),
            extractor: Box::new(extractor),
            sort_key: None,
            right_align: false,
        }
    }

    /// Right-align this column's values.
    #[must_use]
    pub const fn right(mut self) -> Self {
        self.right_align = true;
        self
    }

    /// Make this column sortable by a numeric key.
    #[must_use]
    pub fn sortable(mut self, key_fn: impl Fn(&T) -> i64 + 'static) -> Self {
        self.sort_key = Some(Box::new(key_fn));
        self
    }
}

/// A generic list table with configurable columns, sorting, and pagination.
pub struct ListTable<T: Clone> {
    items: Vec<T>,
    columns: Vec<TableColumn<T>>,
    title: String,
    sorted_by: Option<usize>,
    sort_descending: bool,
    page_size: Option<usize>,
}

impl<T: Clone> ListTable<T> {
    /// Create a new list table with a title and items.
    #[must_use]
    pub fn new(title: &str, items: Vec<T>) -> Self {
        Self {
            items,
            columns: vec![],
            title: title.to_string(),
            sorted_by: None,
            sort_descending: true,
            page_size: None,
        }
    }

    /// Add a column definition.
    #[must_use]
    pub fn add_column(mut self, column: TableColumn<T>) -> Self {
        self.columns.push(column);
        self
    }

    /// Sort by a column index (0-based), descending by default.
    #[must_use]
    pub const fn sort_by(mut self, column_idx: usize, descending: bool) -> Self {
        self.sorted_by = Some(column_idx);
        self.sort_descending = descending;
        self
    }

    /// Limit display to a page size.
    #[must_use]
    pub fn page_size(mut self, size: usize) -> Self {
        self.page_size = Some(size.max(1));
        self
    }

    /// Get items sorted according to the current sort column.
    fn sorted_items(&self) -> Vec<T> {
        let mut items = self.items.clone();

        if let Some(col_idx) = self.sorted_by {
            if let Some(col) = self.columns.get(col_idx) {
                if let Some(ref sort_key) = col.sort_key {
                    items.sort_by(|a, b| {
                        let ka = sort_key(a);
                        let kb = sort_key(b);
                        if self.sort_descending {
                            kb.cmp(&ka)
                        } else {
                            ka.cmp(&kb)
                        }
                    });
                }
            }
        }

        items
    }

    /// Total item count (before pagination).
    #[must_use]
    pub fn total(&self) -> usize {
        self.items.len()
    }

    /// Render the table to the given output.
    pub fn render(&self, output: &Output) {
        if self.items.is_empty() {
            output.print(&format!("{}: (empty)", self.title));
            return;
        }

        let items = self.sorted_items();
        let display_items = self
            .page_size
            .map_or(&items[..], |ps| &items[..items.len().min(ps)]);

        if output.is_styled() {
            self.render_styled(display_items, output);
        } else {
            self.render_plain(display_items, output);
        }

        // Footer with count
        if let Some(page_size) = self.page_size {
            if self.items.len() > page_size {
                output.print(&format!(
                    "  Showing {} of {} items",
                    page_size,
                    self.items.len()
                ));
            } else {
                output.print(&format!("  Total: {} items", self.items.len()));
            }
        } else {
            output.print(&format!("  Total: {} items", self.items.len()));
        }
    }

    /// Render styled table with aligned columns and header.
    #[allow(clippy::cast_possible_truncation)]
    fn render_styled(&self, items: &[T], output: &Output) {
        // Compute column widths
        let mut widths: Vec<usize> = self
            .columns
            .iter()
            .map(|col| col.name.len())
            .collect();

        // Measure all cells
        let rows: Vec<Vec<String>> = items
            .iter()
            .map(|item| {
                self.columns
                    .iter()
                    .enumerate()
                    .map(|(i, col)| {
                        let val = (col.extractor)(item);
                        widths[i] = widths[i].max(val.len());
                        val
                    })
                    .collect()
            })
            .collect();

        // Cap widths at terminal width
        let term_width = output.width();
        let total_padding = self.columns.len().saturating_sub(1) * 3; // " | " separators
        let available = term_width.saturating_sub(total_padding + 4); // margins
        let total_width: usize = widths.iter().sum();
        if total_width > available {
            // Proportionally shrink
            for w in &mut widths {
                *w = (*w * available) / total_width.max(1);
                *w = (*w).max(3);
            }
        }

        // Title
        output.print(&format!("\n  {}", self.title));

        // Header
        let header: String = self
            .columns
            .iter()
            .zip(&widths)
            .map(|(col, &w)| {
                if col.right_align {
                    format!("{:>w$}", col.name)
                } else {
                    format!("{:<w$}", col.name)
                }
            })
            .collect::<Vec<_>>()
            .join(" | ");
        output.print(&format!("  {header}"));

        // Separator
        let sep: String = widths.iter().map(|&w| "-".repeat(w)).collect::<Vec<_>>().join("-+-");
        output.print(&format!("  {sep}"));

        // Rows
        for row in &rows {
            let line: String = row
                .iter()
                .zip(&self.columns)
                .zip(&widths)
                .map(|((val, col), &w)| {
                    let truncated = if val.len() > w {
                        format!("{}...", &val[..w.saturating_sub(3)])
                    } else {
                        val.clone()
                    };
                    if col.right_align {
                        format!("{truncated:>w$}")
                    } else {
                        format!("{truncated:<w$}")
                    }
                })
                .collect::<Vec<_>>()
                .join(" | ");
            output.print(&format!("  {line}"));
        }
    }

    /// Render plain TSV output (for piped/non-TTY contexts).
    fn render_plain(&self, items: &[T], _output: &Output) {
        // TSV header
        let headers: Vec<&str> = self.columns.iter().map(|c| c.name.as_str()).collect();
        println!("{}", headers.join("\t"));

        // TSV rows
        for item in items {
            let cells: Vec<String> = self
                .columns
                .iter()
                .map(|col| {
                    let value = (col.extractor)(item);
                    // Escape tabs and newlines in TSV
                    value.replace('\t', "\\t").replace('\n', "\\n")
                })
                .collect();
            println!("{}", cells.join("\t"));
        }
    }
}

/// Render a hashtag frequency list.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_possible_wrap, clippy::cast_sign_loss)]
pub fn render_hashtag_list(hashtags: &[(String, usize)], output: &Output) {
    let max_count = hashtags.iter().map(|(_, c)| *c).max().unwrap_or(1);

    let table = ListTable::new("Hashtags", hashtags.to_vec())
        .add_column(TableColumn::new("#", |item: &(String, usize)| {
            format!("#{}", item.0)
        }))
        .add_column(
            TableColumn::new("Count", |item: &(String, usize)| item.1.to_string())
                .right()
                .sortable(|item: &(String, usize)| item.1 as i64),
        )
        .add_column(TableColumn::new("", move |item: &(String, usize)| {
            let bar_len = ((item.1 as f64 / max_count as f64) * 20.0) as usize;
            "\u{2588}".repeat(bar_len)
        }))
        .sort_by(1, true);

    table.render(output);
}

/// Render a mention frequency list.
#[allow(clippy::cast_precision_loss, clippy::cast_possible_wrap)]
pub fn render_mention_list(mentions: &[(String, usize)], output: &Output) {
    let total: usize = mentions.iter().map(|(_, c)| c).sum();

    let table = ListTable::new("Mentions", mentions.to_vec())
        .add_column(TableColumn::new("User", |item: &(String, usize)| {
            format!("@{}", item.0)
        }))
        .add_column(
            TableColumn::new("Count", |item: &(String, usize)| item.1.to_string())
                .right()
                .sortable(|item: &(String, usize)| item.1 as i64),
        )
        .add_column(
            TableColumn::new("%", move |item: &(String, usize)| {
                if total > 0 {
                    format!("{:.1}%", (item.1 as f64 / total as f64) * 100.0)
                } else {
                    "\u{2014}".to_string()
                }
            })
            .right(),
        )
        .sort_by(1, true);

    table.render(output);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_table_basic() {
        let items: Vec<(&str, i32)> = vec![("apple", 10), ("banana", 20), ("cherry", 15)];
        let table = ListTable::new("Fruits", items)
            .add_column(TableColumn::new("Name", |item: &(&str, i32)| item.0.to_string()))
            .add_column(TableColumn::new("Count", |item: &(&str, i32)| item.1.to_string()));

        assert_eq!(table.total(), 3);
    }

    #[test]
    fn test_list_table_sorting_descending() {
        let items: Vec<(&str, i64)> = vec![("a", 10), ("b", 30), ("c", 20)];
        let table = ListTable::new("Test", items)
            .add_column(TableColumn::new("Name", |item: &(&str, i64)| item.0.to_string()))
            .add_column(
                TableColumn::new("Count", |item: &(&str, i64)| item.1.to_string())
                    .sortable(|item: &(&str, i64)| item.1),
            )
            .sort_by(1, true);

        let sorted = table.sorted_items();
        assert_eq!(sorted[0].1, 30);
        assert_eq!(sorted[1].1, 20);
        assert_eq!(sorted[2].1, 10);
    }

    #[test]
    fn test_list_table_sorting_ascending() {
        let items: Vec<(&str, i64)> = vec![("a", 30), ("b", 10), ("c", 20)];
        let table = ListTable::new("Test", items)
            .add_column(TableColumn::new("Name", |item: &(&str, i64)| item.0.to_string()))
            .add_column(
                TableColumn::new("Count", |item: &(&str, i64)| item.1.to_string())
                    .sortable(|item: &(&str, i64)| item.1),
            )
            .sort_by(1, false);

        let sorted = table.sorted_items();
        assert_eq!(sorted[0].1, 10);
        assert_eq!(sorted[1].1, 20);
        assert_eq!(sorted[2].1, 30);
    }

    #[test]
    fn test_tsv_escaping() {
        let value = "hello\tworld\ntest";
        let escaped = value.replace('\t', "\\t").replace('\n', "\\n");
        assert_eq!(escaped, "hello\\tworld\\ntest");
    }

    #[test]
    fn test_page_size() {
        let items: Vec<(String, i32)> = (0..100).map(|i| (format!("item{i}"), i)).collect();
        let table = ListTable::new("Test", items)
            .add_column(TableColumn::new("Name", |item: &(String, i32)| item.0.clone()))
            .page_size(10);

        assert_eq!(table.page_size, Some(10));
        assert_eq!(table.total(), 100);
    }

    #[test]
    fn test_page_size_minimum() {
        let items = vec![("a", 1)];
        let table = ListTable::new("Test", items).page_size(0);
        assert_eq!(table.page_size, Some(1)); // Clamped to 1
    }

    #[test]
    fn test_column_right_align() {
        let col =
            TableColumn::<(String, i32)>::new("Test", |item: &(String, i32)| item.0.clone())
                .right();
        assert!(col.right_align);
    }

    #[test]
    fn test_empty_list() {
        let items: Vec<(&str, usize)> = vec![];
        let table = ListTable::new("Empty", items)
            .add_column(TableColumn::new("Name", |item: &(&str, usize)| {
                item.0.to_string()
            }));

        assert!(table.items.is_empty());
        assert_eq!(table.total(), 0);
    }

    #[test]
    fn test_render_empty_list_no_panic() {
        let items: Vec<(&str, usize)> = vec![];
        let table = ListTable::new("Empty", items)
            .add_column(TableColumn::new("Name", |item: &(&str, usize)| {
                item.0.to_string()
            }));
        let output = Output::new_piped();
        // Should not panic
        table.render(&output);
    }

    #[test]
    fn test_sorted_items_no_sort_column() {
        let items: Vec<(&str, i32)> = vec![("c", 3), ("a", 1), ("b", 2)];
        let table = ListTable::new("Test", items)
            .add_column(TableColumn::new("Name", |item: &(&str, i32)| {
                item.0.to_string()
            }));

        // No sort_by called - should return items in original order
        let sorted = table.sorted_items();
        assert_eq!(sorted[0].0, "c");
        assert_eq!(sorted[1].0, "a");
        assert_eq!(sorted[2].0, "b");
    }
}
