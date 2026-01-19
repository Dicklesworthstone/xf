//! CLI definitions for xf.
//!
//! Uses clap for argument parsing with derive macros.

use clap::{Args, Parser, Subcommand, ValueEnum};
use std::path::PathBuf;

/// xf - Ultra-fast X data archive search
#[derive(Parser, Debug)]
#[command(name = "xf")]
#[command(author = "Jeffrey Emanuel <jeff@jeffreyemanuel.dev>")]
#[command(version = concat!(
    env!("CARGO_PKG_VERSION"),
    "\n  Built: ", env!("VERGEN_BUILD_TIMESTAMP"),
    "\n  Rustc: ", env!("VERGEN_RUSTC_SEMVER"),
    "\n  Target: ", env!("VERGEN_CARGO_TARGET_TRIPLE"),
))]
#[command(about = "Ultra-fast CLI for searching X data archives")]
#[command(long_about = r#"
xf (x_find) - A blazingly fast command-line tool for indexing and searching
your X data archive.

Features:
  - Full-text search with BM25 ranking
  - Search tweets, likes, DMs, and Grok chats
  - Sub-millisecond query latency via Tantivy
  - SQLite storage for metadata queries
  - JSON and human-readable output formats

Quick start:
  1. Download your data from x.com/settings/download_your_data
  2. Run: xf index /path/to/your-archive
  3. Search: xf search "your query"
"#)]
#[command(after_help = r#"Common tasks:
  xf search "query"               # Search tweets, likes, DMs, grok chats
  xf search "query" --types dm    # Search DMs only
  xf tweet <id> --thread          # View a tweet thread
  xf export tweets --format csv   # Export tweets to CSV
  xf doctor                       # Check archive/index health
"#)]
pub struct Cli {
    /// Path to the database file
    #[arg(long, env = "XF_DB", global = true)]
    pub db: Option<PathBuf>,

    /// Path to the search index directory
    #[arg(long, env = "XF_INDEX", global = true)]
    pub index: Option<PathBuf>,

    /// Output format
    #[arg(long, short = 'f', default_value = "text", global = true)]
    pub format: OutputFormat,

    /// Be verbose (show debug info)
    #[arg(long, short = 'v', global = true)]
    pub verbose: bool,

    /// Be quiet (suppress non-error output)
    #[arg(long, short = 'q', global = true)]
    pub quiet: bool,

    /// Disable colored output (also respects `NO_COLOR` env var)
    #[arg(long, global = true)]
    pub no_color: bool,

    #[command(subcommand)]
    pub command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Import and index an X data archive from a zip file
    Import(ImportArgs),

    /// Index an X data archive
    Index(IndexArgs),

    /// Search the indexed archive
    Search(SearchArgs),

    /// Show archive statistics
    Stats(StatsArgs),

    /// Show information about a specific tweet
    Tweet(TweetArgs),

    /// List available data in the archive
    List(ListArgs),

    /// Export data in various formats
    Export(ExportArgs),

    /// Show or manage configuration
    Config(ConfigArgs),

    /// Update xf to the latest version
    Update,

    /// Generate shell completions
    Completions(CompletionsArgs),

    /// Check archive, database, and index health
    Doctor(DoctorArgs),

    /// Launch interactive REPL mode
    Shell(ShellArgs),
}

#[derive(Args, Debug)]
#[command(after_help = r#"Examples:
  xf import ~/Downloads/twitter-2026-01-09-abc123.zip
  xf import archive.zip -o ~/my_x_data
  xf import archive.zip --no-index
"#)]
pub struct ImportArgs {
    /// Path to the X data archive zip file
    pub zip_file: PathBuf,

    /// Extract to this directory (default: `~/my_x_history`)
    #[arg(long, short = 'o')]
    pub output: Option<PathBuf>,

    /// Extract only, don't index
    #[arg(long)]
    pub no_index: bool,

    /// Overwrite existing extraction
    #[arg(long, short = 'F')]
    pub force: bool,
}

#[derive(Args, Debug)]
pub struct IndexArgs {
    /// Path to the X data archive directory (defaults to `/data/projects/my_twitter_data`)
    pub archive_path: Option<PathBuf>,

    /// Force full re-index (delete existing data)
    #[arg(long, short = 'F')]
    pub force: bool,

    /// Only index specific data types
    #[arg(long, value_delimiter = ',')]
    pub only: Option<Vec<DataType>>,

    /// Skip specific data types
    #[arg(long, value_delimiter = ',')]
    pub skip: Option<Vec<DataType>>,

    /// Number of parallel workers
    #[arg(long, short = 'j', default_value = "0")]
    pub jobs: usize,
}

#[derive(Args, Debug)]
#[command(after_help = r#"Examples:
  xf search "hello world"              # Basic full-text search
  xf search "rust" --types tweet       # Search only tweets
  xf search "meeting" --types dm       # Search DMs
  xf search "2024" --since "last week" # Recent content
  xf search "bug" --limit 50           # More results
"#)]
pub struct SearchArgs {
    /// Search query
    pub query: String,

    /// Filter by data type (tweet, like, dm, grok, all)
    #[arg(long, short = 't', value_delimiter = ',')]
    pub types: Option<Vec<SearchType>>,

    /// Maximum number of results
    #[arg(long, short = 'n', default_value = "20")]
    pub limit: usize,

    /// Skip first N results (for pagination)
    #[arg(long, default_value = "0")]
    pub offset: usize,

    /// Sort order
    #[arg(long, short = 's', default_value = "relevance")]
    pub sort: SortOrder,

    /// Show only results from this date onwards
    #[arg(
        long,
        long_help = "Show only results from this date onwards (tweets, DMs, Grok). Likes without timestamps are excluded.\n\nFormats: 2024-01-15, 2024-01, \"last week\", \"3 days ago\", \"yesterday\"\nExample: --since \"last month\""
    )]
    pub since: Option<String>,

    /// Show only results until this date
    #[arg(
        long,
        long_help = "Show only results until this date (tweets, DMs, Grok). Likes without timestamps are excluded.\n\nFormats: 2024-01-15, 2024-01, \"last week\", \"3 days ago\", \"yesterday\"\nExample: --until \"yesterday\""
    )]
    pub until: Option<String>,

    /// Search only in replies
    #[arg(long)]
    pub replies_only: bool,

    /// Exclude replies from results
    #[arg(long)]
    pub no_replies: bool,

    /// Show full conversation context for DM searches.
    ///
    /// Requires --types dm. Displays all messages in matching conversations
    /// with search hits highlighted. Works with text and JSON formats.
    #[arg(long, short = 'c')]
    pub context: bool,

    /// Fields to include in output
    #[arg(long, value_delimiter = ',')]
    pub fields: Option<Vec<String>>,

    /// Search mode: lexical (keyword), semantic (meaning), or hybrid (both)
    #[arg(long, short = 'm', default_value = "hybrid")]
    pub mode: crate::hybrid::SearchMode,

    /// Embedder for semantic search: hash (fast, lexical) or ollama (true semantic, requires Ollama)
    #[arg(long, short = 'e', default_value = "hash")]
    pub embedder: EmbedderType,

    /// Ollama model to use (only with --embedder ollama)
    #[arg(long, default_value = "nomic-embed-text")]
    pub ollama_model: String,
}

/// Embedder type for semantic search.
#[derive(ValueEnum, Clone, Debug, Default, PartialEq, Eq)]
pub enum EmbedderType {
    /// Fast hash-based embeddings (not truly semantic)
    #[default]
    Hash,
    /// Ollama-based ML embeddings (true semantic, requires running Ollama)
    Ollama,
}

#[derive(Args, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct StatsArgs {
    /// Show comprehensive analytics dashboard (temporal, engagement, content)
    #[arg(long, short = 'd')]
    pub detailed: bool,

    /// Show top hashtags with counts
    #[arg(long)]
    pub hashtags: bool,

    /// Show top mentioned users with counts
    #[arg(long)]
    pub mentions: bool,

    /// Show temporal analytics (activity patterns, gaps, sparklines)
    #[arg(long)]
    pub temporal: bool,

    /// Show engagement analytics (likes distribution, top tweets)
    #[arg(long)]
    pub engagement: bool,

    /// Show content analysis (media/link ratios, length distribution)
    #[arg(long)]
    pub content: bool,

    /// Number of top items to show
    #[arg(long, short = 'n', default_value = "10")]
    pub top: usize,
}

#[derive(Args, Debug)]
pub struct TweetArgs {
    /// Tweet ID to show
    pub id: String,

    /// Show thread context (replies)
    #[arg(long, short = 't')]
    pub thread: bool,

    /// Show engagement metrics
    #[arg(long, short = 'e')]
    pub engagement: bool,
}

#[derive(Args, Debug)]
pub struct ListArgs {
    /// What to list
    #[arg(default_value = "files")]
    pub what: ListTarget,

    /// Limit number of items
    #[arg(long, short = 'n', default_value = "50")]
    pub limit: usize,
}

#[derive(Args, Debug)]
pub struct ExportArgs {
    /// What to export
    pub what: ExportTarget,

    /// Output file path (stdout if not specified)
    #[arg(long, short = 'o')]
    pub output: Option<PathBuf>,

    /// Export format
    #[arg(long, short = 'f', default_value = "json")]
    pub format: ExportFormat,

    /// Limit number of items
    #[arg(long, short = 'n')]
    pub limit: Option<usize>,
}

#[derive(Args, Debug)]
pub struct ConfigArgs {
    /// Show current configuration
    #[arg(long)]
    pub show: bool,

    /// Set a configuration value
    #[arg(long)]
    pub set: Option<String>,

    /// Path to archive (sets default)
    #[arg(long)]
    pub archive: Option<PathBuf>,
}

#[derive(Args, Debug, Clone)]
pub struct CompletionsArgs {
    /// Shell to generate completions for
    pub shell: clap_complete::Shell,
}

#[derive(Args, Debug)]
pub struct DoctorArgs {
    /// Path to the X data archive directory (overrides config)
    #[arg(long)]
    pub archive: Option<PathBuf>,

    /// Apply safe, idempotent repairs when issues are found
    #[arg(long)]
    pub fix: bool,
}

#[derive(Args, Debug)]
pub struct ShellArgs {
    /// Custom prompt string (default: "xf> ")
    #[arg(long, default_value = "xf> ")]
    pub prompt: String,

    /// Number of results per page (default: 10)
    #[arg(long, default_value = "10")]
    pub page_size: usize,

    /// Disable history file
    #[arg(long)]
    pub no_history: bool,

    /// Path to history file (default: `~/.xf_history`)
    #[arg(long)]
    pub history_file: Option<PathBuf>,
}

#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
pub enum DataType {
    Tweet,
    Like,
    Dm,
    Grok,
    Follower,
    Following,
    Block,
    Mute,
    All,
}

#[derive(ValueEnum, Clone, Debug, PartialEq, Eq)]
pub enum SearchType {
    Tweet,
    Like,
    Dm,
    Grok,
    All,
}

impl SearchType {
    #[must_use]
    pub fn all_content() -> Vec<Self> {
        vec![Self::Tweet, Self::Like, Self::Dm, Self::Grok]
    }
}

impl DataType {
    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::Tweet,
            Self::Like,
            Self::Dm,
            Self::Grok,
            Self::Follower,
            Self::Following,
            Self::Block,
            Self::Mute,
        ]
    }
}

#[derive(ValueEnum, Clone, Debug, Default)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
    JsonPretty,
    Compact,
    Csv,
}

#[derive(ValueEnum, Clone, Debug, Default)]
pub enum SortOrder {
    #[default]
    Relevance,
    Date,
    DateDesc,
    Engagement,
}

#[derive(ValueEnum, Clone, Debug, Default)]
pub enum ListTarget {
    #[default]
    Files,
    Tweets,
    Likes,
    Dms,
    Conversations,
    Followers,
    Following,
    Blocks,
    Mutes,
}

#[derive(ValueEnum, Clone, Debug, Default)]
pub enum ExportTarget {
    #[default]
    Tweets,
    Likes,
    Dms,
    Followers,
    Following,
    All,
}

#[derive(ValueEnum, Clone, Debug, Default)]
pub enum ExportFormat {
    #[default]
    Json,
    Jsonl,
    Csv,
}
