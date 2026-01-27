//! Interactive REPL for xf.
//!
//! Provides a command-driven shell with history, basic search, and help.
//! Supports styled prompts, history display, and status line when running in TTY.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use colored::Colorize;
use rustyline::completion::{Completer, Pair};
use rustyline::error::ReadlineError;
use rustyline::highlight::Highlighter;
use rustyline::hint::Hinter;
use rustyline::history::DefaultHistory;
use rustyline::validate::Validator;
use rustyline::{CompletionType, Config, EditMode, Editor, Helper};
use std::borrow::Cow;
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use tracing::{debug, info, trace, warn};

use crate::output::Output;
use crate::theme::Theme;
use crate::{
    CONTENT_DIVIDER_WIDTH, SearchEngine, SearchResult, Storage, csv_escape_text, format_number,
    format_number_usize, format_relative_date, format_short_id,
};

// =============================================================================
// Search Mode Enum
// =============================================================================

/// Search mode for the REPL (mirrors CLI search modes).
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum SearchMode {
    /// Hybrid search combining lexical and semantic
    #[default]
    Hybrid,
    /// Lexical search only (BM25)
    Lexical,
    /// Semantic search only (vector similarity)
    Semantic,
}

impl std::fmt::Display for SearchMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Hybrid => write!(f, "hybrid"),
            Self::Lexical => write!(f, "lexical"),
            Self::Semantic => write!(f, "semantic"),
        }
    }
}

impl std::str::FromStr for SearchMode {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "hybrid" | "h" => Ok(Self::Hybrid),
            "lexical" | "l" | "lex" => Ok(Self::Lexical),
            "semantic" | "s" | "sem" => Ok(Self::Semantic),
            _ => Err(format!(
                "Unknown search mode: '{}'. Use 'hybrid', 'lexical', or 'semantic'.",
                s
            )),
        }
    }
}

// =============================================================================
// Index Status Enum
// =============================================================================

/// Status of the search index for display in the REPL.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum IndexStatus {
    /// Index is ready and up to date
    #[default]
    Ready,
    /// Index exists but may be stale
    Stale,
    /// Index is missing or not initialized
    Missing,
}

impl IndexStatus {
    /// Get a short label for the status.
    #[must_use]
    pub const fn label(&self) -> &'static str {
        match self {
            Self::Ready => "ready",
            Self::Stale => "stale",
            Self::Missing => "missing",
        }
    }
}

// =============================================================================
// REPL Prompt
// =============================================================================

/// Styled REPL prompt with mode badge and arrow.
#[derive(Clone, Debug)]
pub struct ReplPrompt {
    theme: Theme,
}

impl Default for ReplPrompt {
    fn default() -> Self {
        Self {
            theme: Theme::default(),
        }
    }
}

impl ReplPrompt {
    /// Create a new styled prompt with the given theme.
    #[must_use]
    pub fn new(theme: &Theme) -> Self {
        Self {
            theme: theme.clone(),
        }
    }

    /// Render the prompt string, styled if output supports it.
    #[must_use]
    pub fn render(&self, output: &Output, mode: SearchMode) -> String {
        if !output.is_styled() {
            return format!("xf ({})> ", mode);
        }

        // Return plain string for rustyline (it doesn't support rich markup)
        // The actual styling happens via colored crate in the banner
        format!("xf ({})❯ ", mode)
    }

    /// Render the prompt with result count context.
    #[must_use]
    pub fn render_with_results(&self, output: &Output, mode: SearchMode, count: usize) -> String {
        if !output.is_styled() {
            return format!("xf ({}) [{}]> ", mode, format_number_usize(count));
        }

        format!(
            "xf ({}) [{}]❯ ",
            mode,
            format_number_usize(count)
        )
    }

    /// Render the prompt with conversation context.
    #[must_use]
    pub fn render_in_conversation(&self, output: &Output, mode: SearchMode, conv_id: &str) -> String {
        let snippet = conv_id.get(..8.min(conv_id.len())).unwrap_or(conv_id);
        if !output.is_styled() {
            return format!("xf ({}) [dm:{}]> ", mode, snippet);
        }

        format!("xf ({}) [dm:{}]❯ ", mode, snippet)
    }
}

// =============================================================================
// History Entry and Display
// =============================================================================

/// A single entry in the REPL search history.
#[derive(Clone, Debug)]
pub struct HistoryEntry {
    /// The search query
    pub query: String,
    /// When the search was executed
    pub timestamp: DateTime<Utc>,
    /// Number of results returned
    pub result_count: usize,
    /// Search duration in milliseconds
    pub duration_ms: f64,
}

/// REPL history display renderer.
#[derive(Clone, Debug)]
pub struct ReplHistory {
    entries: Vec<HistoryEntry>,
    theme: Theme,
}

impl ReplHistory {
    /// Create a new history display with entries and theme.
    #[must_use]
    pub fn new(entries: &[HistoryEntry], theme: &Theme) -> Self {
        Self {
            entries: entries.to_vec(),
            theme: theme.clone(),
        }
    }

    /// Render the history display.
    pub fn render(&self, output: &Output) {
        if self.entries.is_empty() {
            if output.is_styled() {
                output.print(&self.theme.format_muted("No search history"));
            } else {
                println!("No search history");
            }
            return;
        }

        if !output.is_styled() {
            println!("# | Query | Results | Time");
            println!("--|-------|---------|-----");
            for (i, entry) in self.entries.iter().enumerate() {
                println!(
                    "{}: {} ({} results, {:.1}ms)",
                    i + 1, entry.query, entry.result_count, entry.duration_ms
                );
            }
            return;
        }

        // Styled output using rich_rust Table if available
        #[cfg(feature = "rich")]
        {
            use rich_rust::prelude::*;

            let mut table = Table::new()
                .title("📜 Search History");

            table = table
                .with_column(Column::new("#").width(4))
                .with_column(Column::new("Query"))
                .with_column(Column::new("Results").justify(JustifyMethod::Right).width(8))
                .with_column(Column::new("Time").justify(JustifyMethod::Right).width(10));

            for (i, entry) in self.entries.iter().enumerate() {
                table.add_row_cells([
                    &(i + 1).to_string(),
                    &entry.query,
                    &entry.result_count.to_string(),
                    &format!("{:.1}ms", entry.duration_ms),
                ]);
            }

            output.print_renderable(&table);
            return;
        }

        // Fallback: simple styled output using colored crate
        #[cfg(not(feature = "rich"))]
        {
            println!("{}", "📜 Search History".bold().cyan());
            println!("{}", "─".repeat(60).dimmed());
            println!(
                "{:>4} {} {:>8} {:>10}",
                "#".dimmed(),
                "Query".dimmed(),
                "Results".dimmed(),
                "Time".dimmed()
            );
            for (i, entry) in self.entries.iter().enumerate() {
                let query_display = if entry.query.len() > 40 {
                    format!("{}...", &entry.query[..37])
                } else {
                    entry.query.clone()
                };
                println!(
                    "{:>4} {} {:>8} {:>10}",
                    (i + 1).to_string().cyan(),
                    query_display,
                    entry.result_count.to_string().green(),
                    format!("{:.1}ms", entry.duration_ms).dimmed()
                );
            }
        }
    }
}

// =============================================================================
// Status Line
// =============================================================================

/// Status line display for the REPL showing archive and index info.
#[derive(Clone, Debug)]
pub struct ReplStatusLine {
    archive_path: PathBuf,
    doc_count: usize,
    index_status: IndexStatus,
    theme: Theme,
}

impl ReplStatusLine {
    /// Create a new status line.
    #[must_use]
    pub fn new(
        archive_path: PathBuf,
        doc_count: usize,
        index_status: IndexStatus,
        theme: &Theme,
    ) -> Self {
        Self {
            archive_path,
            doc_count,
            index_status,
            theme: theme.clone(),
        }
    }

    /// Render the status line as a string.
    #[must_use]
    pub fn render(&self, output: &Output) -> String {
        let archive_name = self.archive_path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy();

        if !output.is_styled() {
            return format!(
                "Archive: {} | {} docs | Index: {}",
                archive_name,
                self.doc_count,
                self.index_status.label()
            );
        }

        let status_icon = match self.index_status {
            IndexStatus::Ready => "●".green().to_string(),
            IndexStatus::Stale => "●".yellow().to_string(),
            IndexStatus::Missing => "●".red().to_string(),
        };

        format!(
            "{} {} | {} docs",
            status_icon,
            archive_name.dimmed(),
            format_number_usize(self.doc_count).cyan()
        )
    }
}

// =============================================================================
// Help Display
// =============================================================================

/// Render styled help for REPL commands.
pub fn render_repl_help(output: &Output) {
    if !output.is_styled() {
        println!("REPL Commands:");
        println!("  :help       - Show this help");
        println!("  :mode MODE  - Switch mode (hybrid|lexical|semantic)");
        println!("  :history    - Show search history");
        println!("  :clear      - Clear screen");
        println!("  :limit N    - Set result limit");
        println!("  :type TYPE  - Filter by type (tweet|like|dm|grok)");
        println!("  :quit       - Exit REPL");
        println!();
        println!("Search Commands:");
        println!("  search <query>  - Search indexed content (s)");
        println!("  list [target]   - List tweets, likes, dms, etc (l)");
        println!("  refine <filter> - Filter current results (r)");
        println!("  more            - Show next page of results (m)");
        println!("  show <number>   - Show full result details");
        println!("  export [format] - Export results as json/csv (e)");
        println!("  stats           - Show archive statistics");
        return;
    }

    // Styled output
    #[cfg(feature = "rich")]
    {
        use rich_rust::prelude::*;

        let mut table = Table::new()
            .title("🔧 REPL Commands");

        table = table
            .with_column(Column::new("Command").style(Style::new().bold()))
            .with_column(Column::new("Description"));

        let commands = [
            (":help", "Show this help message"),
            (":mode <mode>", "Switch search mode (hybrid|lexical|semantic)"),
            (":history", "Show search history"),
            (":clear", "Clear screen"),
            (":limit <n>", "Set result limit"),
            (":type <type>", "Filter by doc type (tweet|like|dm|grok)"),
            (":quit", "Exit REPL"),
        ];

        for (cmd, desc) in commands {
            table.add_row_cells([cmd, desc]);
        }

        output.print_renderable(&table);
        return;
    }

    // Fallback: colored crate styling
    #[cfg(not(feature = "rich"))]
    {
        println!("{}", "🔧 REPL Commands".bold().cyan());
        println!("{}", "─".repeat(60).dimmed());
        let commands = [
            (":help", "Show this help message"),
            (":mode <mode>", "Switch search mode (hybrid|lexical|semantic)"),
            (":history", "Show search history"),
            (":clear", "Clear screen"),
            (":limit <n>", "Set result limit"),
            (":type <type>", "Filter by doc type (tweet|like|dm|grok)"),
            (":quit", "Exit REPL"),
        ];

        for (cmd, desc) in commands {
            println!("  {} {}", cmd.cyan().bold(), desc.dimmed());
        }
    }
}

/// Clear the terminal screen.
pub fn clear_screen(output: &Output) {
    if output.stdout_is_tty() {
        print!("\x1B[2J\x1B[1;1H");
        std::io::stdout().flush().ok();
    }
}

/// Configuration for the REPL session.
#[derive(Debug, Clone)]
pub struct ReplConfig {
    /// Custom prompt string
    pub prompt: String,
    /// Number of results per page
    pub page_size: usize,
    /// Disable history file
    pub no_history: bool,
    /// Path to history file (None = use default `~/.xf_history`)
    pub history_file: Option<PathBuf>,
}

impl Default for ReplConfig {
    fn default() -> Self {
        Self {
            prompt: "xf> ".to_string(),
            page_size: 10,
            no_history: false,
            history_file: None,
        }
    }
}

/// REPL session state.
pub struct ReplSession {
    storage: Storage,
    search: SearchEngine,
    last_results: Vec<SearchResult>,
    last_query: Option<String>,
    history_path: Option<PathBuf>,
    prompt_context: PromptContext,
    /// Current offset for pagination
    current_offset: usize,
    /// Page size for results
    page_size: usize,
    /// Custom prompt string
    prompt_str: String,
    /// Last selected result index (for $_)
    last_selected: Option<usize>,
    /// Named variables (for $name)
    named_vars: HashMap<String, String>,
}

#[derive(Default)]
#[allow(dead_code)]
enum PromptContext {
    #[default]
    Normal,
    WithResults(usize),
    InConversation(String),
}

#[derive(Debug)]
enum Command {
    Search { query: String },
    List { target: ListTarget },
    Refine { filter: String },
    More,
    Show { index: usize },
    Export { format: ExportFormat },
    Stats,
    Help { command: Option<String> },
    Set { name: String, value: String },
    Quit,
}

#[derive(Debug, Clone, Copy)]
enum ListTarget {
    Tweets,
    Likes,
    Dms,
    Conversations,
    Followers,
    Following,
    Blocks,
    Mutes,
}

#[derive(Debug, Clone, Copy)]
enum ExportFormat {
    Json,
    Csv,
}

// =============================================================================
// Tab Completion
// =============================================================================

/// Commands available in the REPL for completion.
const COMMANDS: &[&str] = &[
    "search", "s", "list", "l", "refine", "r", "more", "m", "show", "export", "e", "stats", "set",
    "help", "h", "?", "quit", "exit", "q",
];

/// List targets for completion.
const LIST_TARGETS: &[&str] = &[
    "tweets",
    "likes",
    "favorites",
    "dms",
    "dm",
    "messages",
    "conversations",
    "convos",
    "followers",
    "following",
    "blocks",
    "blocked",
    "mutes",
    "muted",
];

/// Export formats for completion.
const EXPORT_FORMATS: &[&str] = &["json", "csv"];

/// Tab completion helper for xf REPL.
#[derive(Default)]
struct XfCompleter;

impl XfCompleter {
    /// Determine completion context from the input line and cursor position.
    fn get_completions(&self, line: &str, pos: usize) -> Vec<Pair> {
        let line_to_cursor = &line[..pos];
        trace!(line = %line_to_cursor, pos, "Computing completions");

        // Check if cursor is inside quotes - don't complete
        if self.is_inside_quotes(line_to_cursor) {
            trace!("Inside quotes, no completions");
            return Vec::new();
        }

        let parts: Vec<&str> = line_to_cursor.split_whitespace().collect();

        if parts.is_empty() || (parts.len() == 1 && !line_to_cursor.ends_with(' ')) {
            // Complete command
            let prefix = parts.first().copied().unwrap_or("");
            return self.complete_command(prefix);
        }

        let command = parts[0].to_lowercase();

        // If we just finished typing a command (ends with space), suggest next token
        if line_to_cursor.ends_with(' ') {
            return self.complete_after_command(&command, &parts[1..]);
        }

        // Complete partial token
        let partial = parts.last().copied().unwrap_or("");
        self.complete_partial(&command, partial, &parts[1..parts.len().saturating_sub(1)])
    }

    /// Check if cursor position is inside a quoted string.
    #[allow(clippy::unused_self)]
    fn is_inside_quotes(&self, text: &str) -> bool {
        let mut in_single = false;
        let mut in_double = false;
        let mut prev_char = ' ';

        for c in text.chars() {
            if c == '\'' && prev_char != '\\' && !in_double {
                in_single = !in_single;
            } else if c == '"' && prev_char != '\\' && !in_single {
                in_double = !in_double;
            }
            prev_char = c;
        }

        in_single || in_double
    }

    /// Complete a command name.
    #[allow(clippy::unused_self)]
    fn complete_command(&self, prefix: &str) -> Vec<Pair> {
        let prefix_lower = prefix.to_lowercase();
        let mut completions: Vec<Pair> = COMMANDS
            .iter()
            .filter(|cmd| cmd.starts_with(&prefix_lower))
            .map(|cmd| Pair {
                display: (*cmd).to_string(),
                replacement: (*cmd).to_string(),
            })
            .collect();

        completions.sort_by(|a, b| a.display.cmp(&b.display));
        completions.dedup_by(|a, b| a.display == b.display);

        debug!(count = completions.len(), "Command completions");
        completions
    }

    /// Complete after a command has been typed.
    fn complete_after_command(&self, command: &str, args: &[&str]) -> Vec<Pair> {
        match command {
            "list" | "l" if args.is_empty() => self.complete_list_targets(""),
            "export" | "e" if args.is_empty() => self.complete_export_formats(""),
            "help" | "h" | "?" if args.is_empty() => self.complete_help_topics(""),
            _ => Vec::new(),
        }
    }

    /// Complete a partial token based on command context.
    fn complete_partial(&self, command: &str, partial: &str, _prior_args: &[&str]) -> Vec<Pair> {
        match command {
            "list" | "l" => self.complete_list_targets(partial),
            "export" | "e" => self.complete_export_formats(partial),
            "help" | "h" | "?" => self.complete_help_topics(partial),
            _ => Vec::new(),
        }
    }

    /// Complete list targets.
    #[allow(clippy::unused_self)]
    fn complete_list_targets(&self, prefix: &str) -> Vec<Pair> {
        let prefix_lower = prefix.to_lowercase();
        let mut completions: Vec<Pair> = LIST_TARGETS
            .iter()
            .filter(|t| t.starts_with(&prefix_lower))
            .map(|t| Pair {
                display: (*t).to_string(),
                replacement: (*t).to_string(),
            })
            .collect();

        completions.sort_by(|a, b| a.display.cmp(&b.display));
        completions.dedup_by(|a, b| a.display == b.display);
        completions
    }

    /// Complete export formats.
    #[allow(clippy::unused_self)]
    fn complete_export_formats(&self, prefix: &str) -> Vec<Pair> {
        let prefix_lower = prefix.to_lowercase();
        EXPORT_FORMATS
            .iter()
            .filter(|f| f.starts_with(&prefix_lower))
            .map(|f| Pair {
                display: (*f).to_string(),
                replacement: (*f).to_string(),
            })
            .collect()
    }

    /// Complete help topics (command names).
    #[allow(clippy::unused_self)]
    fn complete_help_topics(&self, prefix: &str) -> Vec<Pair> {
        let prefix_lower = prefix.to_lowercase();
        // Only primary commands, not aliases
        let topics = [
            "search", "list", "refine", "more", "show", "export", "stats", "quit",
        ];
        topics
            .iter()
            .filter(|t| t.starts_with(&prefix_lower))
            .map(|t| Pair {
                display: (*t).to_string(),
                replacement: (*t).to_string(),
            })
            .collect()
    }
}

impl Completer for XfCompleter {
    type Candidate = Pair;

    fn complete(
        &self,
        line: &str,
        pos: usize,
        _ctx: &rustyline::Context<'_>,
    ) -> std::result::Result<(usize, Vec<Pair>), ReadlineError> {
        let completions = self.get_completions(line, pos);

        // Find the start of the word being completed
        let line_to_cursor = &line[..pos];
        let word_start = line_to_cursor
            .rfind(|c: char| c.is_whitespace())
            .map_or(0, |i| i + 1);

        Ok((word_start, completions))
    }
}

impl Hinter for XfCompleter {
    type Hint = String;

    fn hint(&self, _line: &str, _pos: usize, _ctx: &rustyline::Context<'_>) -> Option<String> {
        // No hints for now - could add command suggestions later
        None
    }
}

impl Highlighter for XfCompleter {
    fn highlight<'l>(&self, line: &'l str, _pos: usize) -> Cow<'l, str> {
        Cow::Borrowed(line)
    }

    fn highlight_char(&self, _line: &str, _pos: usize) -> bool {
        false
    }
}

impl Validator for XfCompleter {}

impl Helper for XfCompleter {}

/// Run the REPL session.
///
/// # Errors
///
/// Returns an error if readline setup, history persistence, or command execution fails.
pub fn run(storage: Storage, search: SearchEngine, repl_config: ReplConfig) -> Result<()> {
    let rl_config = Config::builder()
        .history_ignore_space(true)
        .history_ignore_dups(true)?
        .completion_type(CompletionType::List)
        .edit_mode(EditMode::Emacs)
        .build();

    let mut rl: Editor<XfCompleter, DefaultHistory> = Editor::with_config(rl_config)?;
    rl.set_helper(Some(XfCompleter));

    // Determine history path
    let history_path = if repl_config.no_history {
        None
    } else {
        Some(repl_config.history_file.unwrap_or_else(|| {
            dirs::home_dir()
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".xf_history")
        }))
    };

    debug!(
        prompt = %repl_config.prompt,
        page_size = repl_config.page_size,
        history_path = ?history_path,
        "REPL configuration"
    );

    let mut session = ReplSession {
        storage,
        search,
        last_results: Vec::new(),
        last_query: None,
        history_path,
        prompt_context: PromptContext::Normal,
        current_offset: 0,
        page_size: repl_config.page_size,
        prompt_str: repl_config.prompt,
        last_selected: None,
        named_vars: HashMap::new(),
    };

    // Load history if enabled
    if let Some(ref path) = session.history_path {
        let _ = rl.load_history(path);
    }

    info!("Starting xf REPL session");
    print_startup_banner(&session.storage);

    loop {
        let prompt = session.format_prompt();
        match rl.readline(&prompt) {
            Ok(line) => {
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                if !matches!(line, "quit" | "exit" | "q") {
                    rl.add_history_entry(line)?;
                }

                debug!(command = %line, "REPL command");
                match session.execute(line) {
                    Ok(true) => {}
                    Ok(false) => break,
                    Err(e) => {
                        warn!(error = %e, "REPL command failed");
                        eprintln!("{}: {e}", "Error".red());
                    }
                }
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
            }
            Err(ReadlineError::Eof) => {
                println!();
                break;
            }
            Err(e) => {
                return Err(anyhow::anyhow!(e)).context("Readline failed");
            }
        }
    }

    // Save history if enabled
    if let Some(ref path) = session.history_path {
        rl.save_history(path)?;
    }
    info!("Ended xf REPL session");
    println!("{}", "Session ended.".dimmed());
    Ok(())
}

impl ReplSession {
    fn format_prompt(&self) -> String {
        // Extract the base prompt (without trailing "> " if present)
        let base = self.prompt_str.trim_end_matches("> ").trim_end_matches('>');
        match &self.prompt_context {
            PromptContext::Normal => self.prompt_str.clone(),
            PromptContext::WithResults(n) => {
                let count = format_number_usize(*n);
                format!("{base} [{count}]> ")
            }
            PromptContext::InConversation(id) => {
                let snippet = id.get(..8.min(id.len())).unwrap_or(id);
                format!("{base} [dm:{snippet}]> ")
            }
        }
    }

    /// Execute a command line, handling pipes and variable substitution.
    fn execute(&mut self, input: &str) -> Result<bool> {
        // Handle pipes: split on | and execute each command sequentially
        let pipe_segments: Vec<&str> = input.split('|').map(str::trim).collect();

        for segment in pipe_segments {
            if segment.is_empty() {
                continue;
            }

            // Substitute variables in the segment
            let substituted = self.substitute_vars(segment);
            debug!(original = %segment, substituted = %substituted, "Variable substitution");

            // Parse and execute
            if !self.execute_single(&substituted)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Execute a single command (after variable substitution).
    fn execute_single(&mut self, input: &str) -> Result<bool> {
        let command = parse_command(input)?;
        match command {
            Command::Search { query } => {
                self.run_search(&query)?;
            }
            Command::List { target } => {
                self.run_list(target)?;
            }
            Command::Refine { filter } => {
                self.run_refine(&filter)?;
            }
            Command::More => {
                self.run_more()?;
            }
            Command::Show { index } => {
                self.run_show(index)?;
            }
            Command::Export { format } => {
                self.run_export(format)?;
            }
            Command::Stats => {
                self.run_stats()?;
            }
            Command::Help { command } => {
                print_help(command.as_deref());
            }
            Command::Set { name, value } => {
                self.run_set(&name, &value);
            }
            Command::Quit => return Ok(false),
        }
        Ok(true)
    }

    /// Substitute variables in input text.
    ///
    /// Supports:
    /// - `$1`, `$2`, ... = Nth result ID (1-indexed)
    /// - `$_` = last selected result ID
    /// - `$*` = all result IDs (space-separated)
    /// - `$name` = named variable value
    fn substitute_vars(&self, input: &str) -> String {
        let mut result = String::with_capacity(input.len());
        let chars: Vec<char> = input.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if chars[i] == '$' && i + 1 < chars.len() {
                let start = i + 1;

                // Check for special cases first
                if chars[start] == '_' {
                    // Check if this is $_ alone (not $_abc which would be variable _abc)
                    let next_idx = start + 1;
                    let is_standalone = next_idx >= chars.len()
                        || (!chars[next_idx].is_alphanumeric() && chars[next_idx] != '_');

                    if is_standalone {
                        // $_ = last selected result
                        if let Some(idx) = self.last_selected {
                            if let Some(r) = self.last_results.get(idx) {
                                result.push_str(&r.id);
                            }
                        }
                        i += 2;
                        continue;
                    }
                    // Fall through to named variable handling for $_abc
                }
                if chars[start] == '*' {
                    // $* = all result IDs
                    let ids: Vec<&str> = self.last_results.iter().map(|r| r.id.as_str()).collect();
                    result.push_str(&ids.join(" "));
                    i += 2;
                    continue;
                } else if chars[start].is_ascii_digit() {
                    // $1, $2, ... = Nth result ID (1-indexed)
                    let mut end = start;
                    while end < chars.len() && chars[end].is_ascii_digit() {
                        end += 1;
                    }
                    let num_str: String = chars[start..end].iter().collect();
                    if let Ok(n) = num_str.parse::<usize>() {
                        if n > 0 {
                            if let Some(r) = self.last_results.get(n - 1) {
                                result.push_str(&r.id);
                            }
                        }
                    }
                    i = end;
                    continue;
                } else if chars[start].is_alphabetic() || chars[start] == '_' {
                    // $name = named variable
                    let mut end = start;
                    while end < chars.len() && (chars[end].is_alphanumeric() || chars[end] == '_') {
                        end += 1;
                    }
                    let name: String = chars[start..end].iter().collect();
                    if let Some(val) = self.named_vars.get(&name) {
                        result.push_str(val);
                    }
                    i = end;
                    continue;
                }
            }

            result.push(chars[i]);
            i += 1;
        }

        result
    }

    /// Set a named variable.
    fn run_set(&mut self, name: &str, value: &str) {
        debug!(name = %name, value = %value, "Setting variable");
        self.named_vars.insert(name.to_string(), value.to_string());
        println!("{} = {}", format!("${name}").bold(), value);
    }

    fn run_search(&mut self, query: &str) -> Result<()> {
        let results = self.search.search(query, None, 100)?;
        let count = results.len();
        self.last_results = results;
        self.last_query = Some(query.to_string());
        self.current_offset = 0;
        self.last_selected = None; // Reset: new search invalidates previous selection
        self.prompt_context = PromptContext::WithResults(count);

        println!(
            "{} {}",
            format_number_usize(count).bold(),
            "results".dimmed()
        );
        print_results(&self.last_results, 0, self.page_size);
        Ok(())
    }

    fn run_stats(&self) -> Result<()> {
        let stats = self.storage.get_stats()?;
        println!("{}", "Archive Statistics".bold().cyan());
        println!("{}", "─".repeat(CONTENT_DIVIDER_WIDTH));
        println!("  {:<20} {}", "Tweets:", format_number(stats.tweets_count));
        println!("  {:<20} {}", "Likes:", format_number(stats.likes_count));
        println!(
            "  {:<20} {}",
            "DM Messages:",
            format_number(stats.dms_count)
        );
        println!(
            "  {:<20} {}",
            "DM Conversations:",
            format_number(stats.dm_conversations_count)
        );
        println!(
            "  {:<20} {}",
            "Grok Messages:",
            format_number(stats.grok_messages_count)
        );
        println!(
            "  {:<20} {}",
            "Followers:",
            format_number(stats.followers_count)
        );
        println!(
            "  {:<20} {}",
            "Following:",
            format_number(stats.following_count)
        );
        println!("  {:<20} {}", "Blocks:", format_number(stats.blocks_count));
        println!("  {:<20} {}", "Mutes:", format_number(stats.mutes_count));
        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn run_list(&self, target: ListTarget) -> Result<()> {
        debug!(?target, "Listing items");
        match target {
            ListTarget::Tweets => {
                let tweets = self.storage.get_all_tweets(None)?;
                println!(
                    "{} {}",
                    format_number_usize(tweets.len()).bold(),
                    "tweets".dimmed()
                );
                for tweet in tweets.iter().take(self.page_size) {
                    let text = truncate_text(&tweet.full_text, 60);
                    println!(
                        "  {} {}",
                        format_relative_date(tweet.created_at).dimmed(),
                        text
                    );
                }
                if tweets.len() > self.page_size {
                    println!(
                        "{}",
                        format!(
                            "… {} more",
                            format_number_usize(tweets.len() - self.page_size)
                        )
                        .dimmed()
                    );
                }
            }
            ListTarget::Likes => {
                let likes = self.storage.get_all_likes(None)?;
                println!(
                    "{} {}",
                    format_number_usize(likes.len()).bold(),
                    "likes".dimmed()
                );
                for like in likes.iter().take(self.page_size) {
                    let text = like.full_text.as_deref().unwrap_or("[no text]");
                    let text = truncate_text(text, 60);
                    println!("  {text}");
                }
                if likes.len() > self.page_size {
                    println!(
                        "{}",
                        format!(
                            "… {} more",
                            format_number_usize(likes.len() - self.page_size)
                        )
                        .dimmed()
                    );
                }
            }
            ListTarget::Dms => {
                let dms = self.storage.get_all_dms(None)?;
                println!(
                    "{} {}",
                    format_number_usize(dms.len()).bold(),
                    "DM messages".dimmed()
                );
                for dm in dms.iter().take(self.page_size) {
                    let text = truncate_text(&dm.text, 60);
                    println!(
                        "  {} {}",
                        format_relative_date(dm.created_at).dimmed(),
                        text
                    );
                }
                if dms.len() > self.page_size {
                    println!(
                        "{}",
                        format!("… {} more", format_number_usize(dms.len() - self.page_size))
                            .dimmed()
                    );
                }
            }
            ListTarget::Conversations => {
                let stats = self.storage.get_stats()?;
                println!(
                    "{} {}",
                    format_number(stats.dm_conversations_count).bold(),
                    "DM conversations".dimmed()
                );
            }
            ListTarget::Followers => {
                let followers = self.storage.get_all_followers(None)?;
                println!(
                    "{} {}",
                    format_number_usize(followers.len()).bold(),
                    "followers".dimmed()
                );
                for f in followers.iter().take(self.page_size) {
                    println!("  {}", format_short_id(&f.account_id));
                }
                if followers.len() > self.page_size {
                    println!(
                        "{}",
                        format!(
                            "… {} more",
                            format_number_usize(followers.len() - self.page_size)
                        )
                        .dimmed()
                    );
                }
            }
            ListTarget::Following => {
                let following = self.storage.get_all_following(None)?;
                println!(
                    "{} {}",
                    format_number_usize(following.len()).bold(),
                    "following".dimmed()
                );
                for f in following.iter().take(self.page_size) {
                    println!("  {}", format_short_id(&f.account_id));
                }
                if following.len() > self.page_size {
                    println!(
                        "{}",
                        format!(
                            "… {} more",
                            format_number_usize(following.len() - self.page_size)
                        )
                        .dimmed()
                    );
                }
            }
            ListTarget::Blocks => {
                let blocks = self.storage.get_all_blocks(None)?;
                println!(
                    "{} {}",
                    format_number_usize(blocks.len()).bold(),
                    "blocked accounts".dimmed()
                );
                for b in blocks.iter().take(self.page_size) {
                    println!("  {}", format_short_id(&b.account_id));
                }
                if blocks.len() > self.page_size {
                    println!(
                        "{}",
                        format!(
                            "… {} more",
                            format_number_usize(blocks.len() - self.page_size)
                        )
                        .dimmed()
                    );
                }
            }
            ListTarget::Mutes => {
                let mutes = self.storage.get_all_mutes(None)?;
                println!(
                    "{} {}",
                    format_number_usize(mutes.len()).bold(),
                    "muted accounts".dimmed()
                );
                for m in mutes.iter().take(self.page_size) {
                    println!("  {}", format_short_id(&m.account_id));
                }
                if mutes.len() > self.page_size {
                    println!(
                        "{}",
                        format!(
                            "… {} more",
                            format_number_usize(mutes.len() - self.page_size)
                        )
                        .dimmed()
                    );
                }
            }
        }
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)] // Consistent return type with other run_* methods
    fn run_refine(&mut self, filter: &str) -> Result<()> {
        if self.last_results.is_empty() {
            warn!("No cached results to refine");
            println!("{}", "No results to refine. Run a search first.".yellow());
            return Ok(());
        }

        let filter_lower = filter.to_lowercase();
        let filtered: Vec<SearchResult> = self
            .last_results
            .iter()
            .filter(|r| r.text.to_lowercase().contains(&filter_lower))
            .cloned()
            .collect();

        let count = filtered.len();
        debug!(original = self.last_results.len(), filtered = count, filter = %filter, "Refined results");

        self.last_results = filtered;
        self.current_offset = 0;
        self.last_selected = None; // Reset: refine changes indices, invalidating previous selection
        self.prompt_context = PromptContext::WithResults(count);

        println!(
            "{} {} (filtered by '{}')",
            format_number_usize(count).bold(),
            "results".dimmed(),
            filter.yellow()
        );
        print_results(&self.last_results, 0, self.page_size);
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)] // Consistent return type with other run_* methods
    fn run_more(&mut self) -> Result<()> {
        if self.last_results.is_empty() {
            println!("{}", "No results. Run a search first.".yellow());
            return Ok(());
        }

        let total = self.last_results.len();
        let new_offset = self.current_offset + self.page_size;

        if new_offset >= total {
            println!("{}", "No more results.".dimmed());
            return Ok(());
        }

        self.current_offset = new_offset;
        debug!(offset = new_offset, total, "Showing more results");

        print_results(&self.last_results, self.current_offset, self.page_size);
        let start = self.current_offset + 1;
        let end = (self.current_offset + self.page_size).min(total);
        println!(
            "{}",
            format!(
                "Showing {}-{} of {}",
                format_number_usize(start),
                format_number_usize(end),
                format_number_usize(total)
            )
            .dimmed()
        );
        Ok(())
    }

    #[allow(clippy::unnecessary_wraps)] // Consistent return type with other run_* methods
    fn run_show(&mut self, index: usize) -> Result<()> {
        if self.last_results.is_empty() {
            println!("{}", "No results. Run a search first.".yellow());
            return Ok(());
        }

        if index == 0 || index > self.last_results.len() {
            println!(
                "{}",
                format!("Invalid index. Use 1-{}.", self.last_results.len()).red()
            );
            return Ok(());
        }

        // Update last_selected for $_ variable
        self.last_selected = Some(index - 1);
        debug!(last_selected = index - 1, "Updated last selected");

        let result = &self.last_results[index - 1];
        debug!(index, result_type = %result.result_type, "Showing result details");

        println!("{}", "─".repeat(CONTENT_DIVIDER_WIDTH));
        println!("{}: {}", "Type".dimmed(), result.result_type);
        println!("{}: {}", "ID".dimmed(), result.id);
        if result.created_at.timestamp() > 0 {
            println!(
                "{}: {}",
                "Date".dimmed(),
                format_relative_date(result.created_at)
            );
        }
        println!("{}: {:.2}", "Score".dimmed(), result.score);
        println!();
        println!("{}", result.text);
        println!("{}", "─".repeat(CONTENT_DIVIDER_WIDTH));
        Ok(())
    }

    fn run_export(&self, format: ExportFormat) -> Result<()> {
        if self.last_results.is_empty() {
            println!("{}", "No results to export. Run a search first.".yellow());
            return Ok(());
        }

        debug!(
            ?format,
            count = self.last_results.len(),
            "Exporting results"
        );

        match format {
            ExportFormat::Json => {
                let json = serde_json::to_string_pretty(&self.last_results)?;
                println!("{json}");
            }
            ExportFormat::Csv => {
                println!("id,type,score,created_at,text");
                for r in &self.last_results {
                    let created = r.created_at.to_rfc3339();
                    // Escape quotes and replace newlines/carriage returns for valid CSV
                    let text_escaped = csv_escape_text(&r.text);
                    println!(
                        "{},{},{:.2},{},\"{}\"",
                        r.id, r.result_type, r.score, created, text_escaped
                    );
                }
            }
        }

        println!(
            "{}",
            format!(
                "Exported {} results",
                format_number_usize(self.last_results.len())
            )
            .dimmed()
        );
        Ok(())
    }
}

fn parse_command(input: &str) -> Result<Command> {
    let parts: Vec<&str> = input.split_whitespace().collect();
    if parts.is_empty() {
        anyhow::bail!("Empty command");
    }

    match parts[0] {
        "search" | "s" => {
            let query = parts[1..].join(" ");
            if query.is_empty() {
                anyhow::bail!("Search query cannot be empty.");
            }
            Ok(Command::Search { query })
        }
        "list" | "l" => {
            let target = parts.get(1).copied().unwrap_or("tweets");
            let target = parse_list_target(target)?;
            Ok(Command::List { target })
        }
        "refine" | "r" => {
            let filter = parts[1..].join(" ");
            if filter.is_empty() {
                anyhow::bail!("Refine requires a filter term.");
            }
            Ok(Command::Refine { filter })
        }
        "more" | "m" => Ok(Command::More),
        "show" => {
            let idx_str = parts
                .get(1)
                .ok_or_else(|| anyhow::anyhow!("Usage: show <number>"))?;
            let index: usize = idx_str
                .parse()
                .map_err(|_| anyhow::anyhow!("Invalid number: {idx_str}"))?;
            Ok(Command::Show { index })
        }
        "export" | "e" => {
            let fmt = parts.get(1).copied().unwrap_or("json");
            let format = match fmt {
                "json" => ExportFormat::Json,
                "csv" => ExportFormat::Csv,
                _ => anyhow::bail!("Unknown export format: {fmt}. Use 'json' or 'csv'."),
            };
            Ok(Command::Export { format })
        }
        "stats" => Ok(Command::Stats),
        "set" => {
            if parts.len() < 3 {
                anyhow::bail!("Usage: set <name> <value>");
            }
            let name = parts[1].trim_start_matches('$').to_string();
            // Variable names must:
            // 1. Not be empty
            // 2. Not start with a digit (since $123 means "result index 123")
            // 3. Contain only alphanumeric chars and underscores
            if name.is_empty()
                || name.chars().next().is_some_and(|c| c.is_ascii_digit())
                || !name.chars().all(|c| c.is_alphanumeric() || c == '_')
            {
                anyhow::bail!(
                    "Invalid variable name '{}'. Names must start with a letter or underscore, and contain only alphanumeric characters and underscores.",
                    parts[1]
                );
            }
            let value = parts[2..].join(" ");
            Ok(Command::Set { name, value })
        }
        "help" | "h" | "?" => Ok(Command::Help {
            command: parts.get(1).map(ToString::to_string),
        }),
        "quit" | "exit" | "q" => Ok(Command::Quit),
        _ => anyhow::bail!(
            "Unknown command: {}. Type 'help' for available commands.",
            parts[0]
        ),
    }
}

fn parse_list_target(s: &str) -> Result<ListTarget> {
    match s {
        "tweets" | "t" => Ok(ListTarget::Tweets),
        "likes" | "favorites" => Ok(ListTarget::Likes),
        "dms" | "dm" | "messages" => Ok(ListTarget::Dms),
        "conversations" | "convos" => Ok(ListTarget::Conversations),
        "followers" => Ok(ListTarget::Followers),
        "following" => Ok(ListTarget::Following),
        "blocks" | "blocked" => Ok(ListTarget::Blocks),
        "mutes" | "muted" => Ok(ListTarget::Mutes),
        _ => anyhow::bail!(
            "Unknown list target: {s}. Options: tweets, likes, dms, conversations, followers, following, blocks, mutes"
        ),
    }
}

fn print_results(results: &[SearchResult], offset: usize, page_size: usize) {
    for (idx, result) in results.iter().skip(offset).take(page_size).enumerate() {
        let text = truncate_text(&result.text, 80);
        println!(
            "{:>3}. [{}] {}",
            offset + idx + 1,
            result.result_type.to_string().cyan(),
            text
        );
    }
    let remaining = results.len().saturating_sub(offset + page_size);
    if remaining > 0 {
        println!(
            "{}",
            format!(
                "… {} more results available (type 'more')",
                format_number_usize(remaining)
            )
            .dimmed()
        );
    }
}

fn truncate_text(text: &str, max_len: usize) -> String {
    let text = text.replace('\n', " ").replace('\r', "");
    let char_count = text.chars().count();
    if char_count <= max_len {
        text
    } else if max_len <= 3 {
        // Can't fit any text + "...", just truncate without ellipsis
        text.chars().take(max_len).collect()
    } else {
        let truncated: String = text.chars().take(max_len - 3).collect();
        format!("{truncated}...")
    }
}

/// Print the startup banner with archive statistics.
fn print_startup_banner(storage: &Storage) {
    const VERSION: &str = env!("CARGO_PKG_VERSION");

    // Get archive stats
    let stats = storage.get_stats();
    let archive_info = storage.get_archive_info().ok().flatten();

    let username = archive_info
        .as_ref()
        .map_or("unknown", |info| info.username.as_str());

    println!();
    println!(
        "{}",
        "╭────────────────────────────────────────────────────────────╮".dimmed()
    );
    // Inner width is 60 chars: "  xf shell " (11) + "vX.X.X" (1+VERSION.len()) + padding
    let header_padding = " ".repeat(60usize.saturating_sub(12 + VERSION.len()));
    println!(
        "{}  {} {}{}{}",
        "│".dimmed(),
        "xf shell".cyan().bold(),
        format!("v{VERSION}").dimmed(),
        header_padding,
        "│".dimmed()
    );
    println!(
        "{}",
        "│                                                            │".dimmed()
    );

    // Archive info
    let archive_line = format!("  Archive: @{username}");
    // Use chars().count() for proper alignment with potential non-ASCII usernames
    let padding = 60usize.saturating_sub(archive_line.chars().count());
    println!(
        "{}{}{}{}",
        "│".dimmed(),
        archive_line,
        " ".repeat(padding),
        "│".dimmed()
    );

    // Stats line
    if let Ok(ref s) = stats {
        let stats_line = format!(
            "  Tweets: {} • DMs: {} • Likes: {}",
            format_number(s.tweets_count),
            format_number(s.dms_count),
            format_number(s.likes_count)
        );
        // Use chars().count() since "•" is a multi-byte UTF-8 character
        let padding = 60usize.saturating_sub(stats_line.chars().count());
        println!(
            "{}{}{}{}",
            "│".dimmed(),
            stats_line,
            " ".repeat(padding),
            "│".dimmed()
        );
    }

    println!(
        "{}",
        "│                                                            │".dimmed()
    );
    // Inner width is 60 chars. "  " prefix (2) + text (41) = 43, need 17 spaces padding.
    println!(
        "{}  {}{}",
        "│".dimmed(),
        "Type 'help' for commands, 'quit' to exit".dimmed(),
        "                 │".dimmed()
    );
    println!(
        "{}",
        "╰────────────────────────────────────────────────────────────╯".dimmed()
    );
    println!();
}

fn print_help(command: Option<&str>) {
    match command {
        Some("search" | "s") => {
            println!("{}", "search <query>".cyan());
            println!("  Search all indexed content (tweets, DMs, likes, Grok)");
            println!("  Aliases: s");
            println!("  Example: search hello world");
        }
        Some("list" | "l") => {
            println!("{}", "list [target]".cyan());
            println!("  List items from the archive");
            println!("  Aliases: l");
            println!(
                "  Targets: tweets, likes, dms, conversations, followers, following, blocks, mutes"
            );
            println!("  Example: list tweets");
        }
        Some("refine" | "r") => {
            println!("{}", "refine <filter>".cyan());
            println!("  Filter the current search results by a term");
            println!("  Aliases: r");
            println!("  Example: refine hello");
        }
        Some("more" | "m") => {
            println!("{}", "more".cyan());
            println!("  Show the next page of results");
            println!("  Aliases: m");
        }
        Some("show") => {
            println!("{}", "show <number>".cyan());
            println!("  Show full details of a result by its number");
            println!("  Example: show 1");
        }
        Some("export" | "e") => {
            println!("{}", "export [format]".cyan());
            println!("  Export current search results");
            println!("  Aliases: e");
            println!("  Formats: json (default), csv");
            println!("  Example: export csv");
        }
        Some("stats") => {
            println!("{}", "stats".cyan());
            println!("  Show archive statistics");
        }
        Some("set") => {
            println!("{}", "set <name> <value>".cyan());
            println!("  Set a named variable for later use");
            println!("  Example: set myquery rust programming");
            println!("  Use: search $myquery");
        }
        Some("quit" | "exit" | "q") => {
            println!("{}", "quit".cyan());
            println!("  Exit the REPL");
            println!("  Aliases: exit, q");
        }
        Some("help" | "h" | "?") => {
            println!("{}", "help [command]".cyan());
            println!("  Show help for all commands or a specific command");
            println!("  Aliases: h, ?");
        }
        _ => {
            println!("{}", "Commands:".bold().cyan());
            println!("  search <query>  - search indexed content (s)");
            println!("  list [target]   - list tweets, likes, dms, etc (l)");
            println!("  refine <filter> - filter current results (r)");
            println!("  more            - show next page of results (m)");
            println!("  show <number>   - show full result details");
            println!("  export [format] - export results as json/csv (e)");
            println!("  stats           - show archive statistics");
            println!("  set <n> <val>   - set a named variable");
            println!("  help [command]  - show help (h, ?)");
            println!("  quit            - exit (exit, q)");
            println!();
            println!("{}", "Variables:".dimmed());
            println!("  {} $1, $2, ... = Nth result ID", "•".dimmed());
            println!("  {} $_ = last shown result ID", "•".dimmed());
            println!("  {} $* = all result IDs", "•".dimmed());
            println!("  {} $name = named variable", "•".dimmed());
            println!();
            println!("{}", "Pipes:".dimmed());
            println!(
                "  {} Use | to chain commands: search rust | refine async",
                "•".dimmed()
            );
            println!();
            println!("{}", "Tips:".dimmed());
            println!("  {} Use Ctrl+C to cancel, Ctrl+D to quit", "•".dimmed());
            println!("  {} Arrow keys for history navigation", "•".dimmed());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ======================== Command Parsing Tests ========================

    #[test]
    fn test_parse_search_command() {
        let cmd = parse_command("search hello world").unwrap();
        assert!(matches!(cmd, Command::Search { query } if query == "hello world"));
    }

    #[test]
    fn test_parse_search_alias() {
        let cmd = parse_command("s rust programming").unwrap();
        assert!(matches!(cmd, Command::Search { query } if query == "rust programming"));
    }

    #[test]
    fn test_parse_search_empty_query_fails() {
        let result = parse_command("search");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_list_default_target() {
        let cmd = parse_command("list").unwrap();
        assert!(matches!(
            cmd,
            Command::List {
                target: ListTarget::Tweets
            }
        ));
    }

    #[test]
    fn test_parse_list_with_target() {
        let cmd = parse_command("list likes").unwrap();
        assert!(matches!(
            cmd,
            Command::List {
                target: ListTarget::Likes
            }
        ));
    }

    #[test]
    fn test_parse_list_alias() {
        let cmd = parse_command("l dms").unwrap();
        assert!(matches!(
            cmd,
            Command::List {
                target: ListTarget::Dms
            }
        ));
    }

    #[test]
    fn test_parse_list_target_aliases() {
        // Test various target aliases
        assert!(matches!(
            parse_command("list favorites").unwrap(),
            Command::List {
                target: ListTarget::Likes
            }
        ));
        assert!(matches!(
            parse_command("list dm").unwrap(),
            Command::List {
                target: ListTarget::Dms
            }
        ));
        assert!(matches!(
            parse_command("list convos").unwrap(),
            Command::List {
                target: ListTarget::Conversations
            }
        ));
        assert!(matches!(
            parse_command("list blocked").unwrap(),
            Command::List {
                target: ListTarget::Blocks
            }
        ));
        assert!(matches!(
            parse_command("list muted").unwrap(),
            Command::List {
                target: ListTarget::Mutes
            }
        ));
    }

    #[test]
    fn test_parse_list_invalid_target() {
        let result = parse_command("list invalid_target");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_refine_command() {
        let cmd = parse_command("refine keyword").unwrap();
        assert!(matches!(cmd, Command::Refine { filter } if filter == "keyword"));
    }

    #[test]
    fn test_parse_refine_alias() {
        let cmd = parse_command("r multiple words here").unwrap();
        assert!(matches!(cmd, Command::Refine { filter } if filter == "multiple words here"));
    }

    #[test]
    fn test_parse_refine_empty_fails() {
        let result = parse_command("refine");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_more_command() {
        let cmd = parse_command("more").unwrap();
        assert!(matches!(cmd, Command::More));
    }

    #[test]
    fn test_parse_more_alias() {
        let cmd = parse_command("m").unwrap();
        assert!(matches!(cmd, Command::More));
    }

    #[test]
    fn test_parse_show_command() {
        let cmd = parse_command("show 5").unwrap();
        assert!(matches!(cmd, Command::Show { index: 5 }));
    }

    #[test]
    fn test_parse_show_no_number_fails() {
        let result = parse_command("show");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_show_invalid_number_fails() {
        let result = parse_command("show abc");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_export_default_format() {
        let cmd = parse_command("export").unwrap();
        assert!(matches!(
            cmd,
            Command::Export {
                format: ExportFormat::Json
            }
        ));
    }

    #[test]
    fn test_parse_export_csv() {
        let cmd = parse_command("export csv").unwrap();
        assert!(matches!(
            cmd,
            Command::Export {
                format: ExportFormat::Csv
            }
        ));
    }

    #[test]
    fn test_parse_export_alias() {
        let cmd = parse_command("e json").unwrap();
        assert!(matches!(
            cmd,
            Command::Export {
                format: ExportFormat::Json
            }
        ));
    }

    #[test]
    fn test_parse_export_invalid_format() {
        let result = parse_command("export xml");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_stats_command() {
        let cmd = parse_command("stats").unwrap();
        assert!(matches!(cmd, Command::Stats));
    }

    #[test]
    fn test_parse_help_command() {
        let cmd = parse_command("help").unwrap();
        assert!(matches!(cmd, Command::Help { command: None }));
    }

    #[test]
    fn test_parse_help_with_topic() {
        let cmd = parse_command("help search").unwrap();
        assert!(matches!(cmd, Command::Help { command: Some(ref c) } if c == "search"));
    }

    #[test]
    fn test_parse_help_aliases() {
        assert!(matches!(
            parse_command("h").unwrap(),
            Command::Help { command: None }
        ));
        assert!(matches!(
            parse_command("?").unwrap(),
            Command::Help { command: None }
        ));
    }

    #[test]
    fn test_parse_quit_command() {
        let cmd = parse_command("quit").unwrap();
        assert!(matches!(cmd, Command::Quit));
    }

    #[test]
    fn test_parse_quit_aliases() {
        assert!(matches!(parse_command("exit").unwrap(), Command::Quit));
        assert!(matches!(parse_command("q").unwrap(), Command::Quit));
    }

    #[test]
    fn test_parse_unknown_command() {
        let result = parse_command("unknown_command");
        assert!(result.is_err());
    }

    // ======================== Prompt Formatting Tests ========================

    #[test]
    fn test_prompt_context_normal() {
        let ctx = PromptContext::Normal;
        // We can't easily test ReplSession::format_prompt directly,
        // but we can verify the enum variants exist and are used correctly
        assert!(matches!(ctx, PromptContext::Normal));
    }

    #[test]
    fn test_prompt_context_with_results() {
        let ctx = PromptContext::WithResults(42);
        assert!(matches!(ctx, PromptContext::WithResults(42)));
    }

    #[test]
    fn test_prompt_context_in_conversation() {
        let ctx = PromptContext::InConversation("abc123".to_string());
        assert!(matches!(ctx, PromptContext::InConversation(_)));
    }

    // ======================== Helper Function Tests ========================

    #[test]
    fn test_truncate_text_short() {
        let result = truncate_text("short text", 80);
        assert_eq!(result, "short text");
    }

    #[test]
    fn test_truncate_text_long() {
        let long_text = "a".repeat(100);
        let result = truncate_text(&long_text, 50);
        assert!(result.ends_with("..."));
        assert!(result.len() <= 50);
    }

    #[test]
    fn test_truncate_text_removes_newlines() {
        let result = truncate_text("line1\nline2\rline3", 80);
        assert!(!result.contains('\n'));
        assert!(!result.contains('\r'));
    }

    #[test]
    fn test_truncate_text_exact_length() {
        let text = "exactly ten";
        let result = truncate_text(text, 11);
        assert_eq!(result, text);
    }

    #[test]
    fn test_truncate_text_very_small_max_len() {
        // When max_len <= 3, we can't fit text + "...", so just truncate
        let result = truncate_text("hello", 2);
        assert_eq!(result, "he");
        assert_eq!(result.len(), 2);

        let result = truncate_text("hello", 3);
        assert_eq!(result, "hel");
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_truncate_text_boundary_max_len() {
        // max_len = 4: just enough for 1 char + "..."
        let result = truncate_text("hello", 4);
        assert_eq!(result, "h...");
        assert_eq!(result.len(), 4);
    }

    // ======================== ReplConfig Tests ========================

    #[test]
    fn test_repl_config_default() {
        let config = ReplConfig::default();
        assert_eq!(config.prompt, "xf> ");
        assert_eq!(config.page_size, 10);
        assert!(!config.no_history);
        assert!(config.history_file.is_none());
    }

    #[test]
    fn test_repl_config_custom_values() {
        let config = ReplConfig {
            prompt: "custom> ".to_string(),
            page_size: 25,
            no_history: true,
            history_file: Some(PathBuf::from("/tmp/test_history")),
        };
        assert_eq!(config.prompt, "custom> ");
        assert_eq!(config.page_size, 25);
        assert!(config.no_history);
        assert_eq!(
            config.history_file,
            Some(PathBuf::from("/tmp/test_history"))
        );
    }

    #[test]
    fn test_repl_config_clone() {
        let config = ReplConfig {
            prompt: "test> ".to_string(),
            page_size: 15,
            no_history: false,
            history_file: None,
        };
        let cloned = config.clone();
        assert_eq!(cloned.prompt, config.prompt);
        assert_eq!(cloned.page_size, config.page_size);
        assert_eq!(cloned.no_history, config.no_history);
        assert_eq!(cloned.history_file, config.history_file);
    }

    // ======================== Prompt Base Extraction Tests ========================

    /// Helper to test prompt base extraction logic (same as `format_prompt` uses).
    fn extract_prompt_base(prompt_str: &str) -> &str {
        prompt_str.trim_end_matches("> ").trim_end_matches('>')
    }

    #[test]
    fn test_extract_prompt_base_default() {
        assert_eq!(extract_prompt_base("xf> "), "xf");
    }

    #[test]
    fn test_extract_prompt_base_custom() {
        assert_eq!(extract_prompt_base("myshell> "), "myshell");
    }

    #[test]
    fn test_extract_prompt_base_with_arrow_only() {
        assert_eq!(extract_prompt_base(">"), "");
    }

    #[test]
    fn test_extract_prompt_base_no_suffix() {
        assert_eq!(extract_prompt_base("test"), "test");
    }

    #[test]
    fn test_extract_prompt_base_with_spaces() {
        assert_eq!(extract_prompt_base("my app> "), "my app");
    }

    // ======================== ListTarget Tests ========================

    #[test]
    fn test_list_target_debug() {
        // Verify Debug trait is implemented
        let target = ListTarget::Tweets;
        let debug_str = format!("{target:?}");
        assert!(debug_str.contains("Tweets"));
    }

    #[test]
    fn test_list_target_copy() {
        let target = ListTarget::Likes;
        let copied = target;
        assert!(matches!(copied, ListTarget::Likes));
    }

    // ======================== ExportFormat Tests ========================

    #[test]
    fn test_export_format_debug() {
        let format = ExportFormat::Json;
        let debug_str = format!("{format:?}");
        assert!(debug_str.contains("Json"));
    }

    #[test]
    fn test_export_format_copy() {
        let format = ExportFormat::Csv;
        let copied = format;
        assert!(matches!(copied, ExportFormat::Csv));
    }

    // ======================== Edge Case Tests ========================

    #[test]
    fn test_parse_command_whitespace_handling() {
        // Leading/trailing whitespace should be handled
        let cmd = parse_command("  search  query  ").unwrap();
        assert!(matches!(cmd, Command::Search { query } if query == "query"));
    }

    #[test]
    fn test_parse_list_requires_lowercase() {
        // Targets must be lowercase
        let result = parse_command("list TWEETS");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_export_requires_lowercase() {
        // Formats must be lowercase
        let result = parse_command("export JSON");
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_input_fails() {
        let result = parse_command("");
        assert!(result.is_err());
    }

    #[test]
    fn test_whitespace_only_fails() {
        let result = parse_command("   ");
        assert!(result.is_err());
    }

    // ======================== Tab Completion Tests ========================

    #[test]
    fn test_complete_command_empty() {
        let completer = XfCompleter;
        let completions = completer.get_completions("", 0);
        // Should return all commands
        assert!(!completions.is_empty());
        assert!(completions.iter().any(|p| p.display == "search"));
        assert!(completions.iter().any(|p| p.display == "list"));
        assert!(completions.iter().any(|p| p.display == "quit"));
    }

    #[test]
    fn test_complete_command_partial() {
        let completer = XfCompleter;
        let completions = completer.get_completions("se", 2);
        assert!(completions.iter().any(|p| p.display == "search"));
        // Should not include unrelated commands
        assert!(!completions.iter().any(|p| p.display == "quit"));
    }

    #[test]
    fn test_complete_command_s_aliases() {
        let completer = XfCompleter;
        let completions = completer.get_completions("s", 1);
        // s, search, show, stats should all match
        assert!(completions.iter().any(|p| p.display == "s"));
        assert!(completions.iter().any(|p| p.display == "search"));
        assert!(completions.iter().any(|p| p.display == "show"));
        assert!(completions.iter().any(|p| p.display == "stats"));
    }

    #[test]
    fn test_complete_list_target_empty() {
        let completer = XfCompleter;
        let completions = completer.get_completions("list ", 5);
        // Should return all list targets
        assert!(completions.iter().any(|p| p.display == "tweets"));
        assert!(completions.iter().any(|p| p.display == "likes"));
        assert!(completions.iter().any(|p| p.display == "dms"));
    }

    #[test]
    fn test_complete_list_target_partial() {
        let completer = XfCompleter;
        let completions = completer.get_completions("list tw", 7);
        assert!(completions.iter().any(|p| p.display == "tweets"));
        assert!(!completions.iter().any(|p| p.display == "likes"));
    }

    #[test]
    fn test_complete_list_alias() {
        let completer = XfCompleter;
        let completions = completer.get_completions("l ", 2);
        // l is alias for list, should complete targets
        assert!(completions.iter().any(|p| p.display == "tweets"));
    }

    #[test]
    fn test_complete_export_format_empty() {
        let completer = XfCompleter;
        let completions = completer.get_completions("export ", 7);
        assert!(completions.iter().any(|p| p.display == "json"));
        assert!(completions.iter().any(|p| p.display == "csv"));
    }

    #[test]
    fn test_complete_export_format_partial() {
        let completer = XfCompleter;
        let completions = completer.get_completions("export j", 8);
        assert!(completions.iter().any(|p| p.display == "json"));
        assert!(!completions.iter().any(|p| p.display == "csv"));
    }

    #[test]
    fn test_complete_help_topics() {
        let completer = XfCompleter;
        let completions = completer.get_completions("help ", 5);
        assert!(completions.iter().any(|p| p.display == "search"));
        assert!(completions.iter().any(|p| p.display == "list"));
    }

    #[test]
    fn test_complete_no_completions_after_search() {
        let completer = XfCompleter;
        // After "search " we don't complete anything (query is free-form)
        let completions = completer.get_completions("search ", 7);
        assert!(completions.is_empty());
    }

    #[test]
    fn test_complete_no_completions_inside_quotes() {
        let completer = XfCompleter;
        // Inside quotes should not complete
        let completions = completer.get_completions("search \"se", 10);
        assert!(completions.is_empty());
    }

    #[test]
    fn test_complete_no_completions_inside_single_quotes() {
        let completer = XfCompleter;
        let completions = completer.get_completions("search 'se", 10);
        assert!(completions.is_empty());
    }

    #[test]
    fn test_is_inside_quotes_false() {
        let completer = XfCompleter;
        assert!(!completer.is_inside_quotes("search query"));
        assert!(!completer.is_inside_quotes("\"complete\""));
        assert!(!completer.is_inside_quotes("'complete'"));
    }

    #[test]
    fn test_is_inside_quotes_true() {
        let completer = XfCompleter;
        assert!(completer.is_inside_quotes("search \"query"));
        assert!(completer.is_inside_quotes("search 'query"));
    }

    #[test]
    fn test_completion_deterministic_order() {
        let completer = XfCompleter;
        let completions1 = completer.get_completions("", 0);
        let completions2 = completer.get_completions("", 0);
        assert_eq!(completions1.len(), completions2.len());
        for (a, b) in completions1.iter().zip(completions2.iter()) {
            assert_eq!(a.display, b.display);
        }
    }

    #[test]
    fn test_completion_no_duplicates() {
        let completer = XfCompleter;
        let completions = completer.get_completions("", 0);
        let mut seen = std::collections::HashSet::new();
        for c in &completions {
            assert!(
                seen.insert(&c.display),
                "Duplicate completion: {}",
                c.display
            );
        }
    }

    // ======================== Set Command Parsing Tests ========================

    #[test]
    fn test_parse_set_command() {
        let cmd = parse_command("set myvar hello").unwrap();
        assert!(matches!(cmd, Command::Set { name, value } if name == "myvar" && value == "hello"));
    }

    #[test]
    fn test_parse_set_with_dollar_prefix() {
        // $myvar should be stripped to myvar
        let cmd = parse_command("set $myvar hello").unwrap();
        assert!(matches!(cmd, Command::Set { name, value } if name == "myvar" && value == "hello"));
    }

    #[test]
    fn test_parse_set_multiword_value() {
        let cmd = parse_command("set query rust programming language").unwrap();
        assert!(
            matches!(cmd, Command::Set { name, value } if name == "query" && value == "rust programming language")
        );
    }

    #[test]
    fn test_parse_set_missing_value_fails() {
        let result = parse_command("set varname");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_set_missing_name_fails() {
        let result = parse_command("set");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_set_empty_name_fails() {
        // set $ value should fail since $ alone is not a valid name
        let result = parse_command("set $ value");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid variable name")
        );
    }

    #[test]
    fn test_parse_set_invalid_chars_fails() {
        // Variable names with special characters should fail
        let result = parse_command("set $a$b value");
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("Invalid variable name")
        );
    }

    #[test]
    fn test_parse_set_valid_underscore_name() {
        // Underscore should be valid in variable names
        let cmd = parse_command("set my_var hello").unwrap();
        assert!(
            matches!(cmd, Command::Set { name, value } if name == "my_var" && value == "hello")
        );
    }

    #[test]
    fn test_parse_set_valid_underscore_prefix() {
        // Variable names starting with underscore should be valid
        let cmd = parse_command("set _private secret").unwrap();
        assert!(
            matches!(cmd, Command::Set { name, value } if name == "_private" && value == "secret")
        );
    }

    #[test]
    fn test_parse_set_numeric_name_fails() {
        // Variable names starting with digit should be rejected
        // because $123 is interpreted as "result index 123", not "variable named 123"
        let result = parse_command("set 123 value");
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Invalid variable name") && err.contains("start with a letter"),
            "Expected error about invalid name starting with digit, got: {err}"
        );
    }

    #[test]
    fn test_parse_set_numeric_prefix_fails() {
        // Even partial numeric prefix should fail
        let result = parse_command("set 1abc value");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_set_valid_with_numbers() {
        // Numbers in the middle/end of name are fine
        let cmd = parse_command("set var123 test").unwrap();
        assert!(matches!(cmd, Command::Set { name, value } if name == "var123" && value == "test"));

        let cmd = parse_command("set my2cents opinion").unwrap();
        assert!(
            matches!(cmd, Command::Set { name, value } if name == "my2cents" && value == "opinion")
        );
    }

    // ======================== Variable Substitution Tests ========================

    /// Helper to create a minimal test session state for variable substitution tests.
    fn create_test_session_vars() -> (Vec<SearchResult>, HashMap<String, String>, Option<usize>) {
        use crate::model::SearchResultType;
        use chrono::Utc;

        let results = vec![
            SearchResult {
                id: "tweet_001".to_string(),
                result_type: SearchResultType::Tweet,
                text: "First tweet".to_string(),
                score: 1.0,
                created_at: Utc::now(),
                highlights: vec![],
                metadata: serde_json::json!({}),
                rerank_score: None,
            },
            SearchResult {
                id: "tweet_002".to_string(),
                result_type: SearchResultType::Tweet,
                text: "Second tweet".to_string(),
                score: 0.9,
                created_at: Utc::now(),
                highlights: vec![],
                metadata: serde_json::json!({}),
                rerank_score: None,
            },
            SearchResult {
                id: "tweet_003".to_string(),
                result_type: SearchResultType::Tweet,
                text: "Third tweet".to_string(),
                score: 0.8,
                created_at: Utc::now(),
                highlights: vec![],
                metadata: serde_json::json!({}),
                rerank_score: None,
            },
        ];

        let mut named_vars = HashMap::new();
        named_vars.insert("myquery".to_string(), "rust async".to_string());
        named_vars.insert("user".to_string(), "alice".to_string());
        named_vars.insert("_private".to_string(), "secret_value".to_string());

        (results, named_vars, Some(1)) // last_selected = index 1 (second result)
    }

    /// Helper to test variable substitution without full `ReplSession`.
    fn substitute_vars_test(
        input: &str,
        results: &[SearchResult],
        named_vars: &HashMap<String, String>,
        last_selected: Option<usize>,
    ) -> String {
        // Replicate the substitution logic for testing
        let mut result = String::with_capacity(input.len());
        let chars: Vec<char> = input.chars().collect();
        let mut i = 0;

        while i < chars.len() {
            if chars[i] == '$' && i + 1 < chars.len() {
                let start = i + 1;

                if chars[start] == '_' {
                    // Check if this is $_ alone (not $_abc which would be variable _abc)
                    let next_idx = start + 1;
                    let is_standalone = next_idx >= chars.len()
                        || (!chars[next_idx].is_alphanumeric() && chars[next_idx] != '_');

                    if is_standalone {
                        if let Some(idx) = last_selected {
                            if let Some(r) = results.get(idx) {
                                result.push_str(&r.id);
                            }
                        }
                        i += 2;
                        continue;
                    }
                    // Fall through to named variable handling for $_abc
                }
                if chars[start] == '*' {
                    let ids: Vec<&str> = results.iter().map(|r| r.id.as_str()).collect();
                    result.push_str(&ids.join(" "));
                    i += 2;
                    continue;
                } else if chars[start].is_ascii_digit() {
                    let mut end = start;
                    while end < chars.len() && chars[end].is_ascii_digit() {
                        end += 1;
                    }
                    let num_str: String = chars[start..end].iter().collect();
                    if let Ok(n) = num_str.parse::<usize>() {
                        if n > 0 {
                            if let Some(r) = results.get(n - 1) {
                                result.push_str(&r.id);
                            }
                        }
                    }
                    i = end;
                    continue;
                } else if chars[start].is_alphabetic() || chars[start] == '_' {
                    let mut end = start;
                    while end < chars.len() && (chars[end].is_alphanumeric() || chars[end] == '_') {
                        end += 1;
                    }
                    let name: String = chars[start..end].iter().collect();
                    if let Some(val) = named_vars.get(&name) {
                        result.push_str(val);
                    }
                    i = end;
                    continue;
                }
            }

            result.push(chars[i]);
            i += 1;
        }

        result
    }

    #[test]
    fn test_substitute_var_numeric_first() {
        let (results, named_vars, last_selected) = create_test_session_vars();
        let result = substitute_vars_test("show $1", &results, &named_vars, last_selected);
        assert_eq!(result, "show tweet_001");
    }

    #[test]
    fn test_substitute_var_numeric_second() {
        let (results, named_vars, last_selected) = create_test_session_vars();
        let result = substitute_vars_test("show $2", &results, &named_vars, last_selected);
        assert_eq!(result, "show tweet_002");
    }

    #[test]
    fn test_substitute_var_numeric_out_of_range() {
        let (results, named_vars, last_selected) = create_test_session_vars();
        // $99 doesn't exist, should be empty
        let result = substitute_vars_test("show $99", &results, &named_vars, last_selected);
        assert_eq!(result, "show ");
    }

    #[test]
    fn test_substitute_var_underscore() {
        let (results, named_vars, last_selected) = create_test_session_vars();
        // $_ = last_selected which is index 1 = tweet_002
        let result = substitute_vars_test("show $_", &results, &named_vars, last_selected);
        assert_eq!(result, "show tweet_002");
    }

    #[test]
    fn test_substitute_var_star() {
        let (results, named_vars, last_selected) = create_test_session_vars();
        let result = substitute_vars_test("echo $*", &results, &named_vars, last_selected);
        assert_eq!(result, "echo tweet_001 tweet_002 tweet_003");
    }

    #[test]
    fn test_substitute_var_named() {
        let (results, named_vars, last_selected) = create_test_session_vars();
        let result = substitute_vars_test("search $myquery", &results, &named_vars, last_selected);
        assert_eq!(result, "search rust async");
    }

    #[test]
    fn test_substitute_var_named_multiple() {
        let (results, named_vars, last_selected) = create_test_session_vars();
        let result = substitute_vars_test(
            "search $myquery by $user",
            &results,
            &named_vars,
            last_selected,
        );
        assert_eq!(result, "search rust async by alice");
    }

    #[test]
    fn test_substitute_var_unknown_named() {
        let (results, named_vars, last_selected) = create_test_session_vars();
        // $unknown should be empty since it's not defined
        let result = substitute_vars_test("search $unknown", &results, &named_vars, last_selected);
        assert_eq!(result, "search ");
    }

    #[test]
    fn test_substitute_var_underscore_prefix() {
        let (results, named_vars, last_selected) = create_test_session_vars();
        // $_private should be the variable _private, NOT $_ + literal "private"
        let result = substitute_vars_test("search $_private", &results, &named_vars, last_selected);
        assert_eq!(result, "search secret_value");
    }

    #[test]
    fn test_substitute_underscore_alone() {
        let (results, named_vars, last_selected) = create_test_session_vars();
        // $_ alone should be last selected (index 1 = tweet_002)
        let result = substitute_vars_test("show $_", &results, &named_vars, last_selected);
        assert_eq!(result, "show tweet_002");
    }

    #[test]
    fn test_substitute_underscore_with_space() {
        let (results, named_vars, last_selected) = create_test_session_vars();
        // "$_ foo" should be last selected + " foo"
        let result = substitute_vars_test("show $_ foo", &results, &named_vars, last_selected);
        assert_eq!(result, "show tweet_002 foo");
    }

    #[test]
    fn test_substitute_no_vars() {
        let (results, named_vars, last_selected) = create_test_session_vars();
        let result =
            substitute_vars_test("search plain text", &results, &named_vars, last_selected);
        assert_eq!(result, "search plain text");
    }

    #[test]
    fn test_substitute_dollar_at_end() {
        let (results, named_vars, last_selected) = create_test_session_vars();
        // $ at end of string should be kept as-is
        let result = substitute_vars_test("search $", &results, &named_vars, last_selected);
        assert_eq!(result, "search $");
    }

    #[test]
    fn test_substitute_mixed_vars() {
        let (results, named_vars, last_selected) = create_test_session_vars();
        let result = substitute_vars_test(
            "from $1 to $2 query $myquery",
            &results,
            &named_vars,
            last_selected,
        );
        assert_eq!(result, "from tweet_001 to tweet_002 query rust async");
    }

    // ======================== Pipe Parsing Tests ========================

    #[test]
    fn test_pipe_split_simple() {
        let input = "search rust | refine async";
        let segments: Vec<&str> = input.split('|').map(str::trim).collect();
        assert_eq!(segments.len(), 2);
        assert_eq!(segments[0], "search rust");
        assert_eq!(segments[1], "refine async");
    }

    #[test]
    fn test_pipe_split_multiple() {
        let input = "search rust | refine async | export json";
        let segments: Vec<&str> = input.split('|').map(str::trim).collect();
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0], "search rust");
        assert_eq!(segments[1], "refine async");
        assert_eq!(segments[2], "export json");
    }

    #[test]
    fn test_pipe_split_no_pipe() {
        let input = "search rust programming";
        let segments: Vec<&str> = input.split('|').map(str::trim).collect();
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0], "search rust programming");
    }

    #[test]
    fn test_pipe_split_with_extra_spaces() {
        let input = "search rust  |  refine async  |  more";
        let segments: Vec<&str> = input.split('|').map(str::trim).collect();
        assert_eq!(segments.len(), 3);
        assert_eq!(segments[0], "search rust");
        assert_eq!(segments[1], "refine async");
        assert_eq!(segments[2], "more");
    }

    // ======================== SearchMode Tests ========================

    #[test]
    fn test_search_mode_display() {
        assert_eq!(SearchMode::Hybrid.to_string(), "hybrid");
        assert_eq!(SearchMode::Lexical.to_string(), "lexical");
        assert_eq!(SearchMode::Semantic.to_string(), "semantic");
    }

    #[test]
    fn test_search_mode_default() {
        assert_eq!(SearchMode::default(), SearchMode::Hybrid);
    }

    #[test]
    fn test_search_mode_from_str() {
        assert_eq!("hybrid".parse::<SearchMode>().unwrap(), SearchMode::Hybrid);
        assert_eq!("h".parse::<SearchMode>().unwrap(), SearchMode::Hybrid);
        assert_eq!("lexical".parse::<SearchMode>().unwrap(), SearchMode::Lexical);
        assert_eq!("l".parse::<SearchMode>().unwrap(), SearchMode::Lexical);
        assert_eq!("lex".parse::<SearchMode>().unwrap(), SearchMode::Lexical);
        assert_eq!("semantic".parse::<SearchMode>().unwrap(), SearchMode::Semantic);
        assert_eq!("s".parse::<SearchMode>().unwrap(), SearchMode::Semantic);
        assert_eq!("sem".parse::<SearchMode>().unwrap(), SearchMode::Semantic);
    }

    #[test]
    fn test_search_mode_from_str_case_insensitive() {
        assert_eq!("HYBRID".parse::<SearchMode>().unwrap(), SearchMode::Hybrid);
        assert_eq!("Lexical".parse::<SearchMode>().unwrap(), SearchMode::Lexical);
        assert_eq!("SEMANTIC".parse::<SearchMode>().unwrap(), SearchMode::Semantic);
    }

    #[test]
    fn test_search_mode_from_str_invalid() {
        assert!("unknown".parse::<SearchMode>().is_err());
        assert!("".parse::<SearchMode>().is_err());
    }

    // ======================== IndexStatus Tests ========================

    #[test]
    fn test_index_status_default() {
        assert_eq!(IndexStatus::default(), IndexStatus::Ready);
    }

    #[test]
    fn test_index_status_label() {
        assert_eq!(IndexStatus::Ready.label(), "ready");
        assert_eq!(IndexStatus::Stale.label(), "stale");
        assert_eq!(IndexStatus::Missing.label(), "missing");
    }

    // ======================== ReplPrompt Tests ========================

    #[test]
    fn test_repl_prompt_render_plain() {
        let output = Output::new_piped();
        let prompt = ReplPrompt::default();
        let rendered = prompt.render(&output, SearchMode::Hybrid);

        assert!(!rendered.contains('['), "Plain prompt should not have markup");
        assert!(rendered.contains("xf"));
        assert!(rendered.contains("hybrid"));
    }

    #[test]
    fn test_repl_prompt_render_modes() {
        let output = Output::new_piped();
        let prompt = ReplPrompt::default();

        for mode in [SearchMode::Hybrid, SearchMode::Lexical, SearchMode::Semantic] {
            let rendered = prompt.render(&output, mode);
            assert!(rendered.contains(&mode.to_string()));
        }
    }

    #[test]
    fn test_repl_prompt_with_results() {
        let output = Output::new_piped();
        let prompt = ReplPrompt::default();
        let rendered = prompt.render_with_results(&output, SearchMode::Hybrid, 42);

        assert!(rendered.contains("42"));
        assert!(rendered.contains("hybrid"));
    }

    #[test]
    fn test_repl_prompt_in_conversation() {
        let output = Output::new_piped();
        let prompt = ReplPrompt::default();
        let rendered = prompt.render_in_conversation(&output, SearchMode::Hybrid, "conv_123456789");

        assert!(rendered.contains("dm:"));
        assert!(rendered.contains("conv_123")); // Truncated to 8 chars
    }

    // ======================== HistoryEntry Tests ========================

    #[test]
    fn test_history_entry_creation() {
        let entry = HistoryEntry {
            query: "test query".into(),
            timestamp: chrono::Utc::now(),
            result_count: 10,
            duration_ms: 5.5,
        };
        assert_eq!(entry.query, "test query");
        assert_eq!(entry.result_count, 10);
        assert!((entry.duration_ms - 5.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_repl_history_empty() {
        let history = ReplHistory::new(&[], &Theme::default());
        assert!(history.entries.is_empty());
    }

    #[test]
    fn test_repl_history_with_entries() {
        let entries = vec![
            HistoryEntry {
                query: "first".into(),
                timestamp: chrono::Utc::now(),
                result_count: 10,
                duration_ms: 5.5,
            },
            HistoryEntry {
                query: "second".into(),
                timestamp: chrono::Utc::now(),
                result_count: 20,
                duration_ms: 7.2,
            },
        ];
        let history = ReplHistory::new(&entries, &Theme::default());
        assert_eq!(history.entries.len(), 2);
    }

    // ======================== ReplStatusLine Tests ========================

    #[test]
    fn test_status_line_render_plain() {
        let status_line = ReplStatusLine::new(
            PathBuf::from("/test/archive"),
            1000,
            IndexStatus::Ready,
            &Theme::default(),
        );
        let output = Output::new_piped();
        let rendered = status_line.render(&output);

        assert!(rendered.contains("1000"));
        assert!(rendered.contains("ready"));
        assert!(rendered.contains("archive"));
    }

    #[test]
    fn test_status_line_render_variants() {
        let theme = Theme::default();
        let output = Output::new_tty();

        for status in [IndexStatus::Ready, IndexStatus::Stale, IndexStatus::Missing] {
            let status_line = ReplStatusLine::new(
                PathBuf::from("/test/archive"),
                1000,
                status,
                &theme,
            );
            let rendered = status_line.render(&output);
            assert!(rendered.contains('●'), "Status line should have indicator");
        }
    }
}
