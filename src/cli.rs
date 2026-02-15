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

    /// Output format (env: XF_OUTPUT_FORMAT, TOON_DEFAULT_FORMAT)
    #[arg(long, short = 'f', default_value = "text", global = true)]
    pub format: OutputFormat,

    /// Increase verbosity (-v for verbose, -vv for debug)
    #[arg(long, short = 'v', global = true, action = clap::ArgAction::Count)]
    pub verbose: u8,

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

    /// Run embedding/reranker benchmarks
    Benchmark(BenchmarkArgs),

    /// Show machine-readable CLI documentation (JSON)
    RobotDocs(RobotDocsArgs),

    /// Manage embedding and reranker models
    Models(ModelsArgs),

    /// Manage the warm model daemon
    Daemon(DaemonArgs),
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
#[allow(clippy::struct_excessive_bools)]
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

    /// Enable semantic embeddings (uses ML model, ~100 items/sec on CPU)
    ///
    /// Downloads the MiniLM model (~80MB) on first use. Enables true
    /// semantic search where "happy" matches "joyful".
    #[arg(long, short = 's')]
    pub semantic: bool,

    /// Generate two-tier embeddings (fast + quality) for progressive search.
    ///
    /// Builds both a fast index (hash-based) and a quality index (MiniLM)
    /// to enable `--mode two-tier` search with score blending.
    #[arg(long)]
    pub two_tier: bool,

    /// Skip embedding generation entirely.
    ///
    /// Useful for quick indexing when only lexical search is needed.
    /// You can generate embeddings later with `xf daemon` background jobs.
    #[arg(long)]
    pub no_embeddings: bool,

    /// Force synchronous embedding (block until complete).
    ///
    /// By default, embeddings are generated in the background via the daemon.
    /// Use this flag to wait for embedding completion before returning.
    #[arg(long)]
    pub sync_embeddings: bool,
}

#[derive(Args, Debug)]
#[command(after_help = r#"Examples:
  xf search "hello world"              # Basic full-text search
  xf search "rust" --types tweet       # Search only tweets
  xf search "meeting" --types dm       # Search DMs
  xf search "2024" --since "last week" # Recent content
  xf search "bug" --limit 50           # More results
"#)]
#[allow(clippy::struct_excessive_bools)]
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

    /// Embedder model override
    #[arg(long)]
    pub model: Option<String>,

    /// MRL dimension override (256, 512, 768, or 1024)
    #[arg(long, value_parser = validate_mrl_dims)]
    pub dimensions: Option<usize>,

    /// Enable reranking stage
    #[arg(long)]
    pub rerank: bool,

    /// Reranker model override
    #[arg(long)]
    pub reranker: Option<String>,

    /// Number of candidates to rerank
    #[arg(long, default_value = "100")]
    pub rerank_top: usize,

    /// Force daemon usage
    #[arg(long, conflicts_with = "no_daemon")]
    pub daemon: bool,

    /// Force direct inference (no daemon)
    #[arg(long, conflicts_with = "daemon")]
    pub no_daemon: bool,

    /// Two-tier search: use only fast (hash) results, skip quality refinement.
    ///
    /// Equivalent to --mode two-tier with blend_factor=0.0.
    /// Fastest option, uses only hash-based similarity.
    #[arg(long, conflicts_with_all = ["quality_only", "blend_factor"])]
    pub fast_only: bool,

    /// Two-tier search: wait for quality results only, skip fast phase display.
    ///
    /// Equivalent to --mode two-tier with blend_factor=1.0.
    /// Highest quality, but no instant results.
    #[arg(long, conflicts_with_all = ["fast_only", "blend_factor"])]
    pub quality_only: bool,

    /// Two-tier search: blend factor for combining fast and quality results.
    ///
    /// Range: 0.0 (fast-only) to 1.0 (quality-only). Default: 0.7.
    /// Implies --mode two-tier if not already set.
    #[arg(long, value_parser = validate_blend_factor, conflicts_with_all = ["fast_only", "quality_only"])]
    pub blend_factor: Option<f32>,
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
    /// Shell to generate completions for (bash, zsh, fish, powershell, elvish)
    pub shell: crate::completions::Shell,
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

#[derive(Args, Debug)]
#[command(after_help = r#"Examples:
  xf benchmark --corpus tests/fixtures/benchmark_corpus.json --model all-MiniLM-L6-v2
  xf benchmark --corpus tests/fixtures/benchmark_corpus.json --model bge-small-en-v1.5 --batch-size 64
  xf benchmark --corpus tests/fixtures/benchmark_corpus.json --model hash-fnv1a-384 --output-dir results/
"#)]
pub struct BenchmarkArgs {
    /// Corpus JSON file
    #[arg(long)]
    pub corpus: PathBuf,

    /// Embedder model to benchmark
    #[arg(long, default_value = "all-MiniLM-L6-v2")]
    pub model: String,

    /// Optional MRL dimension override
    #[arg(long)]
    pub dimensions: Option<usize>,

    /// Batch size per embedding call
    #[arg(long, default_value = "32")]
    pub batch_size: usize,

    /// Warmup iterations (discarded)
    #[arg(long, default_value = "10")]
    pub warmup_iters: usize,

    /// Measurement iterations
    #[arg(long, default_value = "100")]
    pub measure_iters: usize,

    /// Output directory for reports
    #[arg(long, default_value = "results")]
    pub output_dir: PathBuf,
}

#[derive(Args, Debug)]
pub struct RobotDocsArgs {
    /// Documentation topic (default: all)
    #[arg(default_value = "all")]
    pub topic: String,
}

#[derive(Args, Debug)]
pub struct ModelsArgs {
    #[command(subcommand)]
    pub command: ModelsCommand,
}

#[derive(Subcommand, Debug)]
pub enum ModelsCommand {
    /// List available embedding and reranker models
    List(ModelsListArgs),

    /// Show detailed information about a specific model
    Info(ModelsInfoArgs),
}

#[derive(Args, Debug)]
pub struct ModelsListArgs {
    /// Filter by model type (embedder, reranker)
    #[arg(long, short = 't')]
    pub model_type: Option<ModelType>,

    /// Show only available/downloaded models
    #[arg(long)]
    pub available: bool,
}

#[derive(Args, Debug)]
pub struct ModelsInfoArgs {
    /// Model name to show info for
    pub name: String,
}

#[derive(Args, Debug)]
#[command(after_help = r#"Examples:
  xf daemon start                    # Start the daemon (background, auto-shutdown on idle)
  xf daemon start --foreground       # Run in foreground (blocks until shutdown)
  xf daemon stop                     # Stop the running daemon
  xf daemon status                   # Show daemon status and loaded models
"#)]
pub struct DaemonArgs {
    #[command(subcommand)]
    pub command: DaemonCommand,
}

#[derive(Subcommand, Debug)]
pub enum DaemonCommand {
    /// Start the model daemon
    Start(DaemonStartArgs),

    /// Stop the running daemon
    Stop,

    /// Show daemon status
    Status,
}

#[derive(Args, Debug)]
pub struct DaemonStartArgs {
    /// Run in foreground (default: detach and exit)
    #[arg(long)]
    pub foreground: bool,

    /// Path to socket file
    #[arg(long)]
    pub socket: Option<PathBuf>,

    /// Idle timeout in seconds (0 = no timeout)
    #[arg(long, default_value = "300")]
    pub idle_timeout: u64,

    /// Maximum models to keep loaded
    #[arg(long, default_value = "4")]
    pub max_models: usize,

    /// Path to config file
    #[arg(long, short = 'c')]
    pub config: Option<PathBuf>,
}

#[derive(ValueEnum, Clone, Debug)]
pub enum ModelType {
    Embedder,
    Reranker,
}

/// Validate MRL dimensions (must be a positive integer).
fn validate_mrl_dims(s: &str) -> Result<usize, String> {
    let dims: usize = s.parse().map_err(|_| format!("invalid dimension: {s}"))?;
    if dims == 0 {
        return Err("dimension must be greater than 0".to_string());
    }
    Ok(dims)
}

/// Validate blend factor (must be between 0.0 and 1.0).
fn validate_blend_factor(s: &str) -> Result<f32, String> {
    let factor: f32 = s
        .parse()
        .map_err(|_| format!("invalid blend factor: {s}"))?;
    if !(0.0..=1.0).contains(&factor) {
        return Err("blend factor must be between 0.0 and 1.0".to_string());
    }
    Ok(factor)
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

#[derive(ValueEnum, Clone, Debug, Default, PartialEq, Eq)]
pub enum OutputFormat {
    #[default]
    Text,
    Json,
    JsonPretty,
    Compact,
    Csv,
    /// Token-optimized output notation (40-60% fewer tokens than JSON)
    Toon,
}

impl OutputFormat {
    /// Get format from environment variables.
    /// Precedence: XF_OUTPUT_FORMAT > TOON_DEFAULT_FORMAT
    #[must_use]
    pub fn from_env() -> Option<Self> {
        if let Ok(val) = std::env::var("XF_OUTPUT_FORMAT") {
            return Self::from_str(&val);
        }
        if let Ok(val) = std::env::var("TOON_DEFAULT_FORMAT") {
            return Self::from_str(&val);
        }
        None
    }

    fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "text" => Some(Self::Text),
            "json" => Some(Self::Json),
            "json-pretty" | "json_pretty" | "jsonpretty" => Some(Self::JsonPretty),
            "compact" => Some(Self::Compact),
            "csv" => Some(Self::Csv),
            "toon" => Some(Self::Toon),
            _ => None,
        }
    }

    /// Resolve format: CLI explicit value wins, then env vars, then default
    #[must_use]
    pub fn resolve(cli_format: Self) -> Self {
        // If CLI specified a non-default format, use it
        if cli_format != Self::Text {
            return cli_format;
        }
        // Check environment variables
        Self::from_env().unwrap_or(Self::Text)
    }
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
