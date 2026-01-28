//! xf - Ultra-fast X data archive search CLI
//!
//! Main entry point for the xf command-line tool.

use anyhow::{Context, Result};
use chrono::{DateTime, Datelike, NaiveDate, TimeZone, Utc};
use clap::Parser;
use colored::{Colorize, control};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::ThreadPoolBuilder;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufReader, IsTerminal};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};
use tracing::{info, warn};
use tracing_subscriber::EnvFilter;

use xf::benchmark::datasets::BenchmarkCorpus;
use xf::benchmark::reporters;
use xf::benchmark::runner::{BenchmarkConfig, run_embedding_benchmark};
use xf::canonicalize::canonicalize_for_embedding;
use xf::cli;
use xf::config::Config;
use xf::date_parser;
use xf::embedder::Embedder;
use xf::fastembed_embedder::FastEmbedder;
use xf::hash_embedder::HashEmbedder;
use xf::hybrid::{self, SearchMode};
use xf::model_registry::{EmbedderConfig, ModelRegistry};
use xf::output::{Output, OutputFormat as OutFmt, Verbosity};
use xf::repl;
use xf::search;
use xf::stats_analytics::{self, ContentStats, EngagementStats, TemporalStats};
use xf::vector::{VECTOR_INDEX_FILENAME, VectorIndex, write_vector_index};
use xf::{
    ArchiveParser, ArchiveStats, CONTENT_DIVIDER_WIDTH, Cli, Commands, DataType, ExportFormat,
    ExportTarget, HEADER_DIVIDER_WIDTH, ListTarget, OutputFormat, SearchEngine, SearchResult,
    SearchResultType, SearchType, SortOrder, Storage, TweetUrl, VALID_CONFIG_KEYS,
    VALID_OUTPUT_FIELDS, csv_escape_text, find_closest_match, format_bytes, format_duration,
    format_error, format_number, format_number_u64, format_number_usize, format_optional_date,
    format_relative_date, format_short_id,
};

/// Cache container for the `VectorIndex`.
struct VectorIndexCache {
    index: OnceLock<VectorIndex>,
    meta: OnceLock<CacheMeta>,
    init_lock: Mutex<()>,
}

impl VectorIndexCache {
    const fn new() -> Self {
        Self {
            index: OnceLock::new(),
            meta: OnceLock::new(),
            init_lock: Mutex::new(()),
        }
    }

    fn meta(&self) -> Option<&CacheMeta> {
        self.meta.get()
    }

    #[cfg(test)]
    fn index(&self) -> Option<&VectorIndex> {
        self.index.get()
    }

    fn warn_if_stale(&self, storage: &Storage, db_path: &Path) {
        let Some(meta) = self.meta.get() else {
            return;
        };
        match meta.is_stale(storage, db_path) {
            Ok(true) => warn!("VectorIndex cache may be stale; restart xf to reload embeddings."),
            Ok(false) => {}
            Err(err) => warn!("VectorIndex cache staleness check failed: {err}"),
        }
    }

    fn load<'a>(
        &'a self,
        storage: &Storage,
        db_path: &Path,
        index_path: &Path,
    ) -> Result<(&'a VectorIndex, bool)> {
        if let Some(index) = self.index.get() {
            self.warn_if_stale(storage, db_path);
            return Ok((index, false));
        }

        let _guard = self
            .init_lock
            .lock()
            .map_err(|err| anyhow::anyhow!("lock vector index initialization: {err}"))?;
        if let Some(index) = self.index.get() {
            self.warn_if_stale(storage, db_path);
            return Ok((index, false));
        }

        info!("Loading VectorIndex (first search)...");
        let start = Instant::now();

        // Try file first (fast path), fall back to storage (slow path)
        let index = VectorIndex::load_from_file_or_storage(index_path, storage)?;

        // Collect stats for staleness checking
        let type_counts = index.type_counts();
        let embedding_count = index.len();

        let db_mtime = db_path
            .metadata()
            .and_then(|meta| meta.modified())
            .context("read database mtime")?;
        let meta = CacheMeta {
            db_mtime,
            embedding_count,
            type_counts,
        };
        let _ = self.meta.set(meta);

        let _ = self.index.set(index);
        info!(
            "VectorIndex loaded: {} embeddings in {:?}",
            embedding_count,
            start.elapsed()
        );

        self.index
            .get()
            .map(|idx| (idx, true))
            .ok_or_else(|| anyhow::anyhow!("VectorIndex cache not initialized"))
    }
}

/// Global cached `VectorIndex` for semantic search.
/// Initialized on first search, reused for subsequent searches.
static VECTOR_INDEX_CACHE: VectorIndexCache = VectorIndexCache::new();

/// Global cached `FastEmbedder` for semantic search.
/// Falls back to `HashEmbedder` if the ML model is not available.
static SEMANTIC_EMBEDDER: OnceLock<Option<FastEmbedder>> = OnceLock::new();

/// Get the semantic embedder, loading it on first use.
/// Returns `None` if the FastEmbed model is not available (falls back to hash embedder).
fn get_semantic_embedder() -> Option<&'static FastEmbedder> {
    SEMANTIC_EMBEDDER
        .get_or_init(|| match FastEmbedder::try_load() {
            Ok(embedder) => {
                info!("Loaded FastEmbed model for semantic search");
                Some(embedder)
            }
            Err(e) => {
                warn!("FastEmbed model not available, using hash embedder: {e}");
                None
            }
        })
        .as_ref()
}

#[derive(Debug, Clone)]
struct CacheMeta {
    #[allow(dead_code)]
    db_mtime: SystemTime,
    embedding_count: usize,
    type_counts: HashMap<String, usize>,
}

impl CacheMeta {
    fn is_stale(&self, storage: &Storage, db_path: &Path) -> Result<bool> {
        let db_mtime = db_path
            .metadata()
            .and_then(|meta| meta.modified())
            .context("read database mtime")?;
        if db_mtime != self.db_mtime {
            return Ok(true);
        }

        let current_count =
            usize::try_from(storage.embedding_count()?).context("convert embedding count")?;
        Ok(current_count != self.embedding_count)
    }
}

/// Convert CLI output format to the output module's format enum.
///
/// The output module uses a simplified format enum (Text, Json, Csv) while
/// the CLI has additional variants (JsonPretty, Compact, Toon). Non-text
/// formats are mapped to Json/Csv to disable styling.
#[allow(clippy::needless_pass_by_value)] // const fn requires owned value
const fn cli_format_to_output(format: OutputFormat) -> OutFmt {
    match format {
        OutputFormat::Text | OutputFormat::Compact => OutFmt::Text,
        OutputFormat::Json | OutputFormat::JsonPretty | OutputFormat::Toon => OutFmt::Json,
        OutputFormat::Csv => OutFmt::Csv,
    }
}

fn main() -> Result<()> {
    let mut cli = Cli::parse();

    // Resolve format: CLI explicit value wins, then env vars (XF_OUTPUT_FORMAT, TOON_DEFAULT_FORMAT)
    cli.format = OutputFormat::resolve(cli.format);

    // Handle --no-color flag, NO_COLOR env var, and non-interactive output
    if should_disable_color(&cli) {
        control::set_override(false);
    }

    // Setup logging
    // Default to xf-only logs so machine-readable output stays clean.
    let machine_output = matches!(
        cli.format,
        OutputFormat::Json | OutputFormat::JsonPretty | OutputFormat::Csv | OutputFormat::Toon
    );
    // For machine-readable formats, force quiet unless user explicitly asked for verbose
    let effective_quiet = cli.quiet || (machine_output && cli.verbose == 0);
    let verbosity = Verbosity::from_flags(effective_quiet, cli.verbose);
    let env_filter = if std::env::var("RUST_LOG").is_ok() {
        EnvFilter::from_default_env()
    } else {
        EnvFilter::new(format!("xf={}", verbosity.log_filter()))
    };

    tracing_subscriber::fmt()
        .with_env_filter(env_filter)
        .with_target(false)
        .without_time()
        .with_writer(std::io::stderr)
        .init();

    // Create output abstraction for styled terminal output
    let output = Output::new(cli_format_to_output(cli.format.clone())).with_verbosity(verbosity);

    // Run the appropriate command
    match &cli.command {
        None => {
            print_quickstart();
            Ok(())
        }
        Some(Commands::Import(args)) => cmd_import(&cli, args, &output),
        Some(Commands::Index(args)) => cmd_index(&cli, args, &output),
        Some(Commands::Search(args)) => cmd_search(&cli, args, &output),
        Some(Commands::Stats(args)) => cmd_stats(&cli, args, &output),
        Some(Commands::Tweet(args)) => cmd_tweet(&cli, args, &output),
        Some(Commands::List(args)) => cmd_list(&cli, args, &output),
        Some(Commands::Export(args)) => cmd_export(&cli, args, &output),
        Some(Commands::Config(args)) => cmd_config(&cli, args, &output),
        Some(Commands::Update) => {
            cmd_update(&output);
            Ok(())
        }
        Some(Commands::Completions(args)) => {
            cmd_completions(&cli, args);
            Ok(())
        }
        Some(Commands::Doctor(args)) => cmd_doctor(&cli, args, &output),
        Some(Commands::Shell(args)) => cmd_shell(&cli, args, &output),
        Some(Commands::Benchmark(args)) => cmd_benchmark(&cli, args, &output),
        Some(Commands::RobotDocs(args)) => cmd_robot_docs(args, &output),
        Some(Commands::Models(args)) => cmd_models(args, &output),
        Some(Commands::Daemon(args)) => cmd_daemon(args, &output),
    }
}

/// Print a colorful quickstart guide when xf is run with no arguments.
#[allow(clippy::too_many_lines)]
fn print_quickstart() {
    let version = env!("CARGO_PKG_VERSION");

    // Box-drawing characters
    let tl = "╭"; // top-left
    let tr = "╮"; // top-right
    let bl = "╰"; // bottom-left
    let br = "╯"; // bottom-right
    let h = "─"; // horizontal
    let v = "│"; // vertical

    let width = 78;
    let inner = width - 2;

    // Helper to create a horizontal line
    let hline =
        |left: &str, right: &str| -> String { format!("{}{}{}", left, h.repeat(inner), right) };

    // Helper to pad a line to fill the box
    let pad = |text: &str| -> String {
        let visible_len = console::measure_text_width(text);
        let padding = inner.saturating_sub(visible_len);
        format!("{v} {text}{}{v}", " ".repeat(padding.saturating_sub(1)))
    };

    // Header
    println!("{}", hline(tl, tr).bright_cyan());
    println!("{}", pad(""));
    println!(
        "{}",
        pad(&format!(
            "{}  {}",
            "xf".bold().bright_cyan(),
            format!("v{version}").dimmed()
        ))
    );
    println!(
        "{}",
        pad(&"Ultra-fast CLI for searching your X data archive"
            .italic()
            .to_string())
    );
    println!("{}", pad(""));
    println!(
        "{}",
        format!("{v}{}{v}", h.repeat(inner).dimmed()).bright_cyan()
    );

    // Quick Start section
    println!("{}", pad(""));
    println!(
        "{}",
        pad(&format!("{}  Getting Started", "1.".bold().yellow()))
    );
    println!("{}", pad(""));
    println!(
        "{}",
        pad(&format!(
            "   Download your archive from: {}",
            "x.com/settings/download_your_data".cyan()
        ))
    );
    println!(
        "{}",
        pad("   (X emails you when it's ready, usually 24-48 hours)")
    );
    println!("{}", pad(""));
    println!(
        "{}",
        pad(&format!("{}  Extract Your Archive", "2.".bold().yellow()))
    );
    println!("{}", pad(""));
    println!(
        "{}",
        pad(&format!(
            "   {}",
            "unzip ~/Downloads/twitter-*.zip -d ~/my_twitter_data".bright_green()
        ))
    );
    println!("{}", pad(""));
    println!(
        "{}",
        pad(&format!("{}  Index Your Data", "3.".bold().yellow()))
    );
    println!("{}", pad(""));
    println!(
        "{}",
        pad(&format!(
            "   {}  {}",
            "xf index".bright_green(),
            "# Uses default path: /data/projects/my_twitter_data".dimmed()
        ))
    );
    println!(
        "{}",
        pad(&format!(
            "   {}  {}",
            "xf index ~/other/path".bright_green(),
            "# Or specify a custom path".dimmed()
        ))
    );
    println!(
        "{}",
        pad("   (Takes ~5-30 seconds depending on archive size)")
    );
    println!("{}", pad(""));
    println!(
        "{}",
        format!("{v}{}{v}", h.repeat(inner).dimmed()).bright_cyan()
    );

    // Example searches section
    println!("{}", pad(""));
    println!(
        "{}",
        pad(&format!("{}", "Example Searches".bold().bright_magenta()))
    );
    println!("{}", pad(""));
    println!(
        "{}",
        pad(&format!(
            "   {}  {}",
            "xf search \"machine learning\"".bright_green(),
            "# Find tweets about ML".dimmed()
        ))
    );
    println!(
        "{}",
        pad(&format!(
            "   {}  {}",
            "xf search \"dinner plans\" --types dm".bright_green(),
            "# Search your DMs".dimmed()
        ))
    );
    println!(
        "{}",
        pad(&format!(
            "   {}  {}",
            "xf search \"conference\" --types dm --context".bright_green(),
            "# DMs with full convo".dimmed()
        ))
    );
    println!(
        "{}",
        pad(&format!(
            "   {}  {}",
            "xf search \"interesting article\" --types like".bright_green(),
            "# Tweets you liked".dimmed()
        ))
    );
    println!(
        "{}",
        pad(&format!(
            "   {}  {}",
            "xf search \"bug fix\" --since \"last month\"".bright_green(),
            "# Recent tweets only".dimmed()
        ))
    );
    println!(
        "{}",
        pad(&format!(
            "   {}  {}",
            "xf search \"project update\" --format json".bright_green(),
            "# JSON output".dimmed()
        ))
    );
    println!("{}", pad(""));
    println!(
        "{}",
        format!("{v}{}{v}", h.repeat(inner).dimmed()).bright_cyan()
    );

    // More commands section
    println!("{}", pad(""));
    println!(
        "{}",
        pad(&format!("{}", "More Commands".bold().bright_magenta()))
    );
    println!("{}", pad(""));
    println!(
        "{}",
        pad(&format!(
            "   {}  {}",
            "xf stats".bright_green(),
            "# Archive overview (counts, date range)".dimmed()
        ))
    );
    println!(
        "{}",
        pad(&format!(
            "   {}  {}",
            "xf stats --detailed".bright_green(),
            "# Full analytics dashboard".dimmed()
        ))
    );
    println!(
        "{}",
        pad(&format!(
            "   {}  {}",
            "xf list tweets --limit 20".bright_green(),
            "# Browse recent tweets".dimmed()
        ))
    );
    println!(
        "{}",
        pad(&format!(
            "   {}  {}",
            "xf list conversations".bright_green(),
            "# See all DM threads".dimmed()
        ))
    );
    println!(
        "{}",
        pad(&format!(
            "   {}  {}",
            "xf tweet 1234567890 --thread".bright_green(),
            "# View a tweet thread".dimmed()
        ))
    );
    println!(
        "{}",
        pad(&format!(
            "   {}  {}",
            "xf export tweets --format csv -o tweets.csv".bright_green(),
            "# Export to CSV".dimmed()
        ))
    );
    println!(
        "{}",
        pad(&format!(
            "   {}  {}",
            "xf shell".bright_green(),
            "# Interactive REPL mode".dimmed()
        ))
    );
    println!(
        "{}",
        pad(&format!(
            "   {}  {}",
            "xf doctor".bright_green(),
            "# Check archive/index health".dimmed()
        ))
    );
    println!("{}", pad(""));
    println!(
        "{}",
        format!("{v}{}{v}", h.repeat(inner).dimmed()).bright_cyan()
    );

    // Footer
    println!("{}", pad(""));
    println!(
        "{}",
        pad(&format!(
            "Documentation: {}",
            "https://github.com/Dicklesworthstone/xf".cyan().underline()
        ))
    );
    println!(
        "{}",
        pad(&format!(
            "Run {} for all options",
            "xf --help".bright_green()
        ))
    );
    println!("{}", pad(""));
    println!("{}", hline(bl, br).bright_cyan());
}

fn no_color_env_set() -> bool {
    match std::env::var("NO_COLOR") {
        Err(std::env::VarError::NotPresent) => false,
        Ok(_) | Err(std::env::VarError::NotUnicode(_)) => true,
    }
}

fn should_disable_color(cli: &Cli) -> bool {
    cli.no_color || no_color_env_set() || !std::io::stdout().is_terminal()
}

fn get_db_path(cli: &Cli) -> PathBuf {
    if let Some(db) = &cli.db {
        return db.clone();
    }
    let config = Config::load();
    config.db_path()
}

fn get_index_path(cli: &Cli) -> PathBuf {
    if let Some(index) = &cli.index {
        return index.clone();
    }
    let config = Config::load();
    config.index_path()
}

/// Import an X data archive from a zip file.
///
/// Extracts the archive to a standard location and optionally indexes it.
#[allow(clippy::too_many_lines)]
fn cmd_import(cli: &Cli, args: &cli::ImportArgs, output: &Output) -> Result<()> {
    // Validate zip file exists
    if !args.zip_file.exists() {
        anyhow::bail!(
            "{}",
            format_error(
                "Zip file not found",
                &format!("The file '{}' does not exist.", args.zip_file.display()),
                &[
                    "Check the path for typos",
                    "Download your data from x.com/settings/download_your_data",
                ],
            )
        );
    }

    // Determine output directory
    let output_dir = args.output.clone().unwrap_or_else(|| {
        dirs::home_dir()
            .unwrap_or_else(|| PathBuf::from("."))
            .join("my_x_history")
    });

    // Check if output already exists
    if output_dir.exists() && !args.force {
        anyhow::bail!(
            "{}",
            format_error(
                "Directory already exists",
                &format!("'{}' already exists.", output_dir.display()),
                &[
                    "Use --force to overwrite",
                    "Choose a different output with -o <path>",
                ],
            )
        );
    }

    output.print("");
    output.print("[bold bright_cyan]Importing X data archive...[/]");
    output.print("");

    // Create progress bar for extraction
    let pb = ProgressBar::new_spinner();
    pb.set_style(
        ProgressStyle::default_spinner()
            .template("{spinner:.cyan} {msg}")
            .unwrap(),
    );
    pb.enable_steady_tick(Duration::from_millis(80));
    pb.set_message(format!(
        "Extracting {}...",
        args.zip_file
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
    ));

    // Open and extract zip file
    let file = File::open(&args.zip_file)
        .with_context(|| format!("Failed to open '{}'", args.zip_file.display()))?;
    let reader = BufReader::new(file);
    let mut archive = zip::ZipArchive::new(reader)
        .with_context(|| format!("Failed to read zip file '{}'", args.zip_file.display()))?;

    // Create output directory
    if args.force && output_dir.exists() {
        fs::remove_dir_all(&output_dir)?;
    }
    fs::create_dir_all(&output_dir)?;

    // Extract files
    let total_files = archive.len();
    let mut extracted_size: u64 = 0;

    for i in 0..total_files {
        let mut file = archive.by_index(i)?;
        let outpath = match file.enclosed_name() {
            Some(path) => output_dir.join(path),
            None => continue,
        };

        if file.is_dir() {
            fs::create_dir_all(&outpath)?;
        } else {
            if let Some(parent) = outpath.parent() {
                if !parent.exists() {
                    fs::create_dir_all(parent)?;
                }
            }
            let mut outfile = File::create(&outpath)?;
            io::copy(&mut file, &mut outfile)?;
            extracted_size += file.size();
        }

        if i % 100 == 0 {
            pb.set_message(format!("Extracting... ({}/{} files)", i + 1, total_files));
        }
    }

    pb.finish_and_clear();

    // Format extracted size
    let size_str = format_bytes(extracted_size);

    output.print(&format!(
        "  [bold green]✓[/] Extracted to [cyan]{}[/]",
        output_dir.display()
    ));
    output.print(&format!(
        "    [dim]→[/] [bold]{size_str}[/] in {total_files} files"
    ));
    output.print("");

    // Index unless --no-index
    if args.no_index {
        output.print("[dim]  · Skipping indexing (--no-index)[/]");
        output.print("");
        output.print(&format!(
            "  Run [bright_green]xf index {}[/] to index later.",
            output_dir.display()
        ));
    } else {
        // Create index args and call cmd_index
        let index_args = cli::IndexArgs {
            archive_path: Some(output_dir.clone()),
            force: true, // Always force since this is a fresh import
            only: None,
            skip: None,
            jobs: 0,
            semantic: false, // Don't use semantic embeddings by default for import
        };

        cmd_index(cli, &index_args, output)?;

        // Print welcome box with stats
        print_import_welcome(&output_dir, cli)?;
    }

    output.print("");
    Ok(())
}

/// Print a beautiful welcome box after successful import.
fn print_import_welcome(_archive_path: &PathBuf, cli: &Cli) -> Result<()> {
    let db_path = get_db_path(cli);
    let storage = Storage::open(&db_path)?;

    // Get stats
    let stats = storage.get_stats()?;

    let date_range = match (stats.first_tweet_date, stats.last_tweet_date) {
        (Some(first), Some(_last)) => format!("since {}", first.format("%b %Y")),
        _ => String::new(),
    };

    println!();

    // Box drawing
    let h = "─";
    let tl = "╭";
    let tr = "╮";
    let bl = "╰";
    let br = "╯";
    let v = "│";
    let width = 48;
    let inner = width - 2;

    let hline = format!("{}{}{}", tl, h.repeat(inner), tr).bright_cyan();
    let bline = format!("{}{}{}", bl, h.repeat(inner), br).bright_cyan();

    let pad = |text: &str| -> String {
        let visible_len = console::measure_text_width(text);
        // inner = width - 2 (for the two │ chars), minus 1 for the leading space
        let padding = inner.saturating_sub(visible_len).saturating_sub(1);
        format!(
            "{} {}{}{}",
            v.bright_cyan(),
            text,
            " ".repeat(padding),
            v.bright_cyan()
        )
    };

    println!("{hline}");
    println!("{}", pad(""));
    println!(
        "{}",
        pad(&format!(
            "  {}",
            "Welcome to your X archive!".bold().bright_magenta()
        ))
    );
    println!("{}", pad(""));

    if stats.tweets_count > 0 {
        println!(
            "{}",
            pad(&format!(
                "  Tweets:    {:>6}  {}",
                format_number(stats.tweets_count).bold(),
                date_range.dimmed()
            ))
        );
    }
    if stats.likes_count > 0 {
        println!(
            "{}",
            pad(&format!(
                "  Likes:     {:>6}",
                format_number(stats.likes_count).bold()
            ))
        );
    }
    if stats.dms_count > 0 {
        println!(
            "{}",
            pad(&format!(
                "  DMs:       {:>6}  {}",
                format_number(stats.dms_count).bold(),
                format!("in {} conversations", stats.dm_conversations_count).dimmed()
            ))
        );
    }
    if stats.grok_messages_count > 0 {
        println!(
            "{}",
            pad(&format!(
                "  Grok:      {:>6}",
                format_number(stats.grok_messages_count).bold()
            ))
        );
    }

    println!("{}", pad(""));
    println!(
        "{}",
        pad(&format!(
            "  Try: {}",
            "xf search \"your first tweet\"".bright_green()
        ))
    );
    println!("{}", pad(""));
    println!("{bline}");

    Ok(())
}

#[allow(clippy::too_many_lines)]
fn cmd_index(cli: &Cli, args: &cli::IndexArgs, output: &Output) -> Result<()> {
    // Use provided path or fall back to config/default
    let config = Config::load();
    let default_path = config
        .paths
        .archive
        .unwrap_or_else(|| PathBuf::from(xf::DEFAULT_ARCHIVE_PATH));
    let archive_path = args.archive_path.as_ref().unwrap_or(&default_path);

    // Validate archive path
    if !archive_path.exists() {
        anyhow::bail!(
            "{}",
            format_error(
                "Archive not found",
                &format!("The path '{}' does not exist.", archive_path.display()),
                &[
                    "Check the path for typos",
                    "Ensure the archive is extracted (not still a .zip file)",
                    "Download your data from x.com/settings/download_your_data",
                ],
            )
        );
    }

    let data_path = archive_path.join("data");
    if !data_path.exists() {
        anyhow::bail!(
            "{}",
            format_error(
                "Invalid archive structure",
                &format!(
                    "No 'data' directory found at '{}'.\n   This doesn't look like a valid X data archive.",
                    archive_path.display()
                ),
                &[
                    "Ensure you're pointing to the extracted archive root",
                    "The archive should contain a 'data' folder with .js files",
                ],
            )
        );
    }

    if args.jobs > 0 {
        ThreadPoolBuilder::new()
            .num_threads(args.jobs)
            .build_global()
            .context("Failed to configure rayon thread pool")?;
    }

    // Setup database and index paths
    let db_path = get_db_path(cli);
    let index_path = get_index_path(cli);

    // Create data directory
    if let Some(parent) = db_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::create_dir_all(&index_path)?;

    // Handle force flag
    if args.force {
        if db_path.exists() {
            std::fs::remove_file(&db_path)?;
        }
        if index_path.exists() {
            std::fs::remove_dir_all(&index_path)?;
            std::fs::create_dir_all(&index_path)?;
        }
        info!("Cleared existing data");
    }

    let index_start = Instant::now();

    output.print("[bold cyan]Indexing X data archive...[/]");
    output.print(&format!("  Archive: {}", archive_path.display()));
    output.print(&format!("  Database: {}", db_path.display()));
    output.print(&format!("  Index: {}", index_path.display()));
    output.print("");

    // Parse archive
    let parser = ArchiveParser::new(archive_path);

    // Open storage and search engine
    let mut storage = Storage::open(&db_path)?;
    let search_engine = SearchEngine::open(&index_path)?;
    let mut writer = search_engine.writer(100_000_000)?;

    // Parse and store manifest
    let manifest = parser.parse_manifest()?;
    storage.store_archive_info(&manifest)?;
    output.print(&format!(
        "  [green]✓[/] Archive for @{} ({})",
        manifest.username,
        manifest.display_name.as_deref().unwrap_or("Unknown")
    ));

    // Determine what to index
    let mut data_types = args.only.as_ref().map_or_else(
        || {
            args.skip.as_ref().map_or_else(DataType::all, |skip| {
                DataType::all()
                    .into_iter()
                    .filter(|t| !skip.contains(t))
                    .collect()
            })
        },
        Clone::clone,
    );

    if let Some(only) = &args.only {
        if only.iter().any(|t| matches!(t, DataType::All)) {
            data_types = DataType::all();
        }
    }

    if let Some(skip) = &args.skip {
        if skip.iter().any(|t| matches!(t, DataType::All)) {
            data_types.clear();
        }
    }

    if data_types.is_empty() {
        anyhow::bail!(
            "{}",
            format_error(
                "No data types selected",
                "Your filters excluded all data types.",
                &[
                    "Remove --skip all",
                    "Use --only tweet,like,dm,grok,follower,following,block,mute",
                    "Run 'xf index <archive_path>' to index everything",
                ],
            )
        );
    }

    // Progress bar (hidden when stdout is non-tty)
    let use_progress = std::io::stdout().is_terminal();
    let pb = if use_progress {
        let pb = ProgressBar::new(data_types.len() as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.cyan} [{elapsed_precise}] {bar:40.cyan/blue} {pos}/{len} ETA {eta_precise} {msg}",
                )
                .unwrap()
                .progress_chars("█▓▒░"),
        );
        pb.enable_steady_tick(Duration::from_millis(120));
        pb
    } else {
        ProgressBar::hidden()
    };

    let log_line = |line: &str| {
        if use_progress {
            pb.println(line);
        } else {
            output.print(line);
        }
    };

    // Index each data type
    for data_type in &data_types {
        let item_start = Instant::now();
        match data_type {
            DataType::Tweet => {
                pb.set_message("tweets");
                let tweets = parser.parse_tweets()?;
                storage.store_tweets(&tweets)?;
                search_engine.index_tweets(&mut writer, &tweets)?;
                let elapsed = format_duration(item_start.elapsed());
                log_line(&format!(
                    "  [green]✓[/] [bold]{}[/] tweets [dim]({elapsed})[/]",
                    format_number_usize(tweets.len())
                ));
            }
            DataType::Like => {
                pb.set_message("likes");
                let likes = parser.parse_likes()?;
                storage.store_likes(&likes)?;
                search_engine.index_likes(&mut writer, &likes)?;
                let elapsed = format_duration(item_start.elapsed());
                log_line(&format!(
                    "  [green]✓[/] [bold]{}[/] likes [dim]({elapsed})[/]",
                    format_number_usize(likes.len())
                ));
            }
            DataType::Dm => {
                pb.set_message("DMs");
                let convos = parser.parse_direct_messages()?;
                let msg_count: usize = convos.iter().map(|c| c.messages.len()).sum();
                storage.store_dm_conversations(&convos)?;
                search_engine.index_dms(&mut writer, &convos)?;
                let elapsed = format_duration(item_start.elapsed());
                log_line(&format!(
                    "  [green]✓[/] [bold]{}[/] DM conversations ([bold]{}[/] messages) [dim]({elapsed})[/]",
                    format_number_usize(convos.len()),
                    format_number_usize(msg_count)
                ));
            }
            DataType::Grok => {
                pb.set_message("Grok");
                let messages = parser.parse_grok_messages()?;
                storage.store_grok_messages(&messages)?;
                search_engine.index_grok_messages(&mut writer, &messages)?;
                let elapsed = format_duration(item_start.elapsed());
                log_line(&format!(
                    "  [green]✓[/] [bold]{}[/] Grok messages [dim]({elapsed})[/]",
                    format_number_usize(messages.len())
                ));
            }
            DataType::Follower => {
                pb.set_message("followers");
                let followers = parser.parse_followers()?;
                storage.store_followers(&followers)?;
                let elapsed = format_duration(item_start.elapsed());
                log_line(&format!(
                    "  [green]✓[/] [bold]{}[/] followers [dim]({elapsed})[/]",
                    format_number_usize(followers.len())
                ));
            }
            DataType::Following => {
                pb.set_message("following");
                let following = parser.parse_following()?;
                storage.store_following(&following)?;
                let elapsed = format_duration(item_start.elapsed());
                log_line(&format!(
                    "  [green]✓[/] [bold]{}[/] following [dim]({elapsed})[/]",
                    format_number_usize(following.len())
                ));
            }
            DataType::Block => {
                pb.set_message("blocks");
                let blocks = parser.parse_blocks()?;
                storage.store_blocks(&blocks)?;
                let elapsed = format_duration(item_start.elapsed());
                log_line(&format!(
                    "  [green]✓[/] [bold]{}[/] blocks [dim]({elapsed})[/]",
                    format_number_usize(blocks.len())
                ));
            }
            DataType::Mute => {
                pb.set_message("mutes");
                let mutes = parser.parse_mutes()?;
                storage.store_mutes(&mutes)?;
                let elapsed = format_duration(item_start.elapsed());
                log_line(&format!(
                    "  [green]✓[/] [bold]{}[/] mutes [dim]({elapsed})[/]",
                    format_number_usize(mutes.len())
                ));
            }
            DataType::All => {
                // Already handled by DataType::all()
            }
        }
        pb.inc(1);
    }

    pb.finish_and_clear();

    // Commit search index
    writer.commit()?;
    search_engine.reload()?;

    // Generate embeddings for semantic search
    let embed_config = xf::EmbeddingConfig {
        show_progress: !cli.quiet,
        use_semantic: args.semantic,
        model: None,
        dimensions: None,
    };
    xf::generate_embeddings_with_config(&storage, &embed_config)?;

    // Write vector index file for fast semantic search
    let vector_stats = write_vector_index(&index_path, &storage)?;
    if !cli.quiet && vector_stats.record_count > 0 {
        output.print(&format!(
            "  [green]✓[/] Vector index written ({} records, {})",
            format_number_usize(vector_stats.record_count),
            format_bytes(vector_stats.file_size)
        ));
    }

    let total_elapsed = format_duration(index_start.elapsed());

    output.print("");
    output.print(&format!(
        "[green]✓[/] [bold]Indexing complete in {total_elapsed}[/]"
    ));
    output.print(&format!(
        "  Total documents indexed: [bold]{}[/]",
        format_number_u64(search_engine.doc_count())
    ));
    output.print("");
    output.print("Run [bold]xf search <query>[/] to search your archive.");

    Ok(())
}

#[allow(clippy::too_many_lines)]
fn cmd_search(cli: &Cli, args: &cli::SearchArgs, output: &Output) -> Result<()> {
    let db_path = get_db_path(cli);
    let index_path = get_index_path(cli);
    let config = Config::load();

    if !db_path.exists() {
        anyhow::bail!(
            "{}",
            format_error(
                "No archive indexed yet",
                "Before searching, you need to index your X data archive.",
                &[
                    "1. Download your data from x.com/settings/download_your_data",
                    "2. Run: xf index ~/Downloads/twitter-archive",
                    "Then try your search again!",
                ],
            )
        );
    }

    if !index_path.join("meta.json").exists() {
        anyhow::bail!(
            "{}",
            format_error(
                "Search index missing",
                &format!(
                    "Database exists but search index not found at '{}'.",
                    index_path.display()
                ),
                &["Run 'xf index <archive_path>' to rebuild the search index"],
            )
        );
    }

    if args.replies_only && args.no_replies {
        anyhow::bail!(
            "{}",
            format_error(
                "Conflicting options",
                "--replies-only and --no-replies cannot be used together.\n   These flags are mutually exclusive.",
                &[
                    "--replies-only    Show only replies",
                    "--no-replies      Exclude replies from results",
                ],
            )
        );
    }

    if args.context {
        if !matches!(
            cli.format,
            OutputFormat::Text | OutputFormat::Json | OutputFormat::JsonPretty | OutputFormat::Toon
        ) {
            anyhow::bail!("--context only supports text, json, json-pretty, or toon output.");
        }
        if let Some(types) = &args.types {
            if types.len() != 1 || !types.contains(&SearchType::Dm) {
                anyhow::bail!("--context only supports --types dm.");
            }
        }
    }

    if let Some(fields) = &args.fields {
        if args.context {
            anyhow::bail!("--fields is not supported with --context.");
        }
        if !matches!(
            cli.format,
            OutputFormat::Json | OutputFormat::JsonPretty | OutputFormat::Toon
        ) {
            anyhow::bail!("--fields is only supported with --format json, json-pretty, or toon.");
        }
        validate_output_fields(fields)?;
    }

    let search_engine = SearchEngine::open(&index_path)?;
    let storage = Storage::open(&db_path)?;

    // Convert data types to search doc types
    let doc_types: Option<Vec<search::DocType>> = if args.context {
        Some(vec![search::DocType::DirectMessage])
    } else {
        args.types.as_ref().and_then(|types| {
            if types.iter().any(|t| matches!(t, SearchType::All)) {
                return None;
            }
            Some(
                types
                    .iter()
                    .filter_map(|t| match t {
                        SearchType::Tweet => Some(search::DocType::Tweet),
                        SearchType::Like => Some(search::DocType::Like),
                        SearchType::Dm => Some(search::DocType::DirectMessage),
                        SearchType::Grok => Some(search::DocType::GrokMessage),
                        SearchType::All => None,
                    })
                    .collect(),
            )
        })
    };

    // Load vector index for semantic/hybrid search (cached per process)
    let vector_index = if matches!(args.mode, SearchMode::Semantic | SearchMode::Hybrid) {
        let index = load_vector_index_cached(&storage, &db_path, &index_path)?;
        if matches!(args.mode, SearchMode::Semantic)
            && !has_embeddings_for_types(doc_types.as_deref())
        {
            anyhow::bail!(
                "{}",
                format_error(
                    "No embeddings found",
                    "Semantic search requires embeddings. Your archive may need re-indexing.",
                    &["Run 'xf index <archive_path> --force' to rebuild with embeddings"],
                )
            );
        }
        Some(index)
    } else {
        None
    };

    let since = match args.since.as_deref() {
        Some(value) => Some(parse_date_arg("--since", value, false, cli.verbose > 0)?),
        None => None,
    };
    let until = match args.until.as_deref() {
        Some(value) => Some(parse_date_arg("--until", value, true, cli.verbose > 0)?),
        None => None,
    };

    let limit_target = args.limit.saturating_add(args.offset);
    let needs_post_filter =
        since.is_some() || until.is_some() || args.replies_only || args.no_replies;
    let needs_full_sort = !matches!(args.sort, SortOrder::Relevance);
    let max_docs = if needs_post_filter || needs_full_sort {
        usize::try_from(search_engine.doc_count()).unwrap_or(usize::MAX)
    } else {
        limit_target
    };

    // Time the search operation
    let search_start = Instant::now();

    // Perform search based on mode
    let mut results = match args.mode {
        SearchMode::Lexical => {
            // Original lexical-only search
            let mut fetch_limit = limit_target.min(max_docs);
            loop {
                let mut batch =
                    search_engine.search(&args.query, doc_types.as_deref(), fetch_limit)?;
                if needs_post_filter {
                    apply_search_filters(
                        &mut batch,
                        since,
                        until,
                        args.replies_only,
                        args.no_replies,
                    );
                }

                if (batch.len() >= limit_target && !needs_full_sort) || fetch_limit >= max_docs {
                    break batch;
                }

                let next = fetch_limit
                    .saturating_mul(2)
                    .max(fetch_limit.saturating_add(1));
                fetch_limit = next.min(max_docs);
            }
        }

        SearchMode::Semantic => {
            // Semantic-only search using vector similarity
            let vector_index = vector_index
                .ok_or_else(|| anyhow::anyhow!("vector index required for semantic"))?;
            // Use explicit model if provided, otherwise FastEmbedder fallback
            #[allow(unused_assignments)]
            let mut embedder_box: Option<Box<dyn Embedder>> = None;
            #[allow(unused_assignments)]
            let mut hash_embedder_fallback: Option<HashEmbedder> = None;
            let embedder: &dyn Embedder = if let Some(model) = &args.model {
                if let Some(dims) = args.dimensions {
                    validate_mrl_dims(dims)?;
                }
                let registry = ModelRegistry::new();
                let mut cfg = EmbedderConfig::new(model);
                cfg.dimensions = args.dimensions;
                cfg.show_progress = false;
                let boxed = registry.embedder(&cfg)?;
                embedder_box = Some(boxed);
                embedder_box.as_ref().unwrap().as_ref()
            } else if let Some(fe) = get_semantic_embedder() {
                info!("Using semantic embeddings for search");
                fe
            } else {
                warn!("FastEmbed model not available, using hash embeddings for search");
                hash_embedder_fallback = Some(HashEmbedder::default());
                hash_embedder_fallback.as_ref().unwrap()
            };
            let canonical_query = canonicalize_for_embedding(&args.query);

            if canonical_query.is_empty() {
                Vec::new()
            } else {
                let query_embedding = embedder.embed(&canonical_query)?;

                // Convert doc_types to string slices for vector search
                let type_strs: Option<Vec<&str>> = doc_types
                    .as_ref()
                    .map(|types| types.iter().map(|t| t.as_str()).collect());

                let semantic_hits = vector_index.search_top_k(
                    &query_embedding,
                    limit_target.saturating_mul(hybrid::CANDIDATE_MULTIPLIER),
                    type_strs.as_deref(),
                );

                let lookups: Vec<_> = semantic_hits
                    .iter()
                    .map(|hit| search::DocLookup::with_type(&hit.doc_id, hit.doc_type))
                    .collect();
                let fetched = search_engine.get_by_ids(&lookups)?;

                // Look up full results from search engine by doc_id + type
                let mut results = Vec::new();
                for (hit, result) in semantic_hits.into_iter().zip(fetched) {
                    if let Some(mut result) = result {
                        result.score = hit.score;
                        results.push(result);
                    }
                }

                if needs_post_filter {
                    apply_search_filters(
                        &mut results,
                        since,
                        until,
                        args.replies_only,
                        args.no_replies,
                    );
                }
                results
            }
        }

        SearchMode::Hybrid => {
            // Hybrid search using RRF fusion
            // Use explicit model if provided, otherwise FastEmbedder fallback
            #[allow(unused_assignments)]
            let mut embedder_box: Option<Box<dyn Embedder>> = None;
            #[allow(unused_assignments)]
            let mut hash_embedder_fallback: Option<HashEmbedder> = None;
            let embedder: &dyn Embedder = if let Some(model) = &args.model {
                if let Some(dims) = args.dimensions {
                    validate_mrl_dims(dims)?;
                }
                let registry = ModelRegistry::new();
                let mut cfg = EmbedderConfig::new(model);
                cfg.dimensions = args.dimensions;
                cfg.show_progress = false;
                let boxed = registry.embedder(&cfg)?;
                embedder_box = Some(boxed);
                embedder_box.as_ref().unwrap().as_ref()
            } else if let Some(fe) = get_semantic_embedder() {
                fe
            } else {
                hash_embedder_fallback = Some(HashEmbedder::default());
                hash_embedder_fallback.as_ref().unwrap()
            };
            let canonical_query = canonicalize_for_embedding(&args.query);
            let candidate_count = hybrid::candidate_count(args.limit, args.offset);

            // Get lexical results
            let lexical_results =
                search_engine.search(&args.query, doc_types.as_deref(), candidate_count)?;

            // Get semantic results (if embeddings exist and query canonicalizes)
            let semantic_results = get_semantic_results(
                vector_index,
                embedder,
                &canonical_query,
                doc_types.as_deref(),
                candidate_count,
            );

            // Fuse results using RRF
            // Pass limit + offset as the limit, and 0 for offset, so the common
            // pagination code at the end handles offset consistently with other modes
            let fused = hybrid::rrf_fuse(
                &lexical_results,
                &semantic_results,
                args.limit.saturating_add(args.offset),
                0,
            );

            // Convert fused hits back to SearchResults
            let mut lookups = Vec::new();
            let mut lookup_indices = Vec::new();
            for (idx, hit) in fused.iter().enumerate() {
                if hit.lexical_rank.is_none() {
                    let lookup = if hit.doc_type.is_empty() {
                        search::DocLookup::new(hit.doc_id)
                    } else {
                        search::DocLookup::with_type(hit.doc_id, hit.doc_type)
                    };
                    lookups.push(lookup);
                    lookup_indices.push(idx);
                }
            }

            let fetched = if lookups.is_empty() {
                Vec::new()
            } else {
                search_engine.get_by_ids(&lookups)?
            };

            let mut fetched_by_index = vec![None; fused.len()];
            for (idx, result) in lookup_indices.into_iter().zip(fetched) {
                fetched_by_index[idx] = result;
            }

            let mut results = Vec::new();
            for (idx, hit) in fused.iter().enumerate() {
                // Prefer lexical result (has full data)
                if let Some(rank) = hit.lexical_rank {
                    let mut result = lexical_results[rank].clone();
                    result.score = hit.score;
                    results.push(result);
                } else if let Some(mut result) = fetched_by_index[idx].take() {
                    result.score = hit.score;
                    results.push(result);
                }
            }

            if needs_post_filter {
                apply_search_filters(
                    &mut results,
                    since,
                    until,
                    args.replies_only,
                    args.no_replies,
                );
            }
            results
        }
    };

    apply_search_sort(&mut results, &args.sort);

    // Apply offset
    let mut results: Vec<_> = results.into_iter().skip(args.offset).collect();
    if args.limit == 0 {
        results.clear();
    } else if results.len() > args.limit {
        results.truncate(args.limit);
    }

    let search_elapsed = search_start.elapsed();

    if results.is_empty() {
        if matches!(cli.format, OutputFormat::Json | OutputFormat::JsonPretty) {
            output.print("[]");
            return Ok(());
        }

        output.print(&format!(
            "[yellow]No results found[/] for \"[bold]{}[/]\"\n",
            args.query
        ));
        output.print("[dim]  Try:[/]");
        output.print("[dim]    • Using different keywords[/]");
        output.print("[dim]    • Checking your spelling[/]");
        if args.since.is_some() || args.until.is_some() {
            output.print("[dim]    • Removing date filters[/]");
        }
        if let Some(types) = &args.types {
            if types.len() == 1 {
                output.print("[dim]    •[/] Searching other data types: [cyan]xf search \"...\" --types tweet,dm,like[/]");
            }
        }
        return Ok(());
    }

    if !config.search.highlight {
        for result in &mut results {
            result.highlights.clear();
        }
    }

    if args.context {
        let contexts = build_dm_context(&results, &storage)?;
        output_dm_context(cli, &contexts, config.search.highlight, output)?;
        return Ok(());
    }

    // Output results
    match cli.format {
        OutputFormat::Json => {
            if let Some(fields) = &args.fields {
                let filtered = filter_results_fields(&results, fields)?;
                output.print(&serde_json::to_string(&filtered)?);
            } else {
                output.print(&serde_json::to_string(&results)?);
            }
        }
        OutputFormat::JsonPretty => {
            if let Some(fields) = &args.fields {
                let filtered = filter_results_fields(&results, fields)?;
                output.print(&serde_json::to_string_pretty(&filtered)?);
            } else {
                output.print(&serde_json::to_string_pretty(&results)?);
            }
        }
        OutputFormat::Csv => {
            output.print("type,id,created_at,score,text");
            for r in &results {
                // Escape quotes and replace newlines/carriage returns for valid CSV
                let text_escaped = csv_escape_text(&r.text);
                output.print(&format!(
                    "{},{},{},{:.4},\"{}\"",
                    r.result_type,
                    r.id,
                    r.created_at.to_rfc3339(),
                    r.score,
                    text_escaped
                ));
            }
        }
        OutputFormat::Compact => {
            for r in &results {
                output.print(&format!(
                    "[{}] {} | {}",
                    r.result_type,
                    r.id,
                    truncate(&r.text, 100)
                ));
            }
        }
        OutputFormat::Text => {
            let timing_str = format_duration(search_elapsed);

            output.print(&format!(
                "Found [bold]{}[/] results for \"[bold]{}[/]\" in [dim]{timing_str}[/]\n",
                format_number_usize(results.len()),
                args.query
            ));

            for (i, r) in results.iter().enumerate() {
                print_result(i + 1, r, output);
            }
        }
        OutputFormat::Toon => {
            let json = if let Some(fields) = &args.fields {
                let filtered = filter_results_fields(&results, fields)?;
                serde_json::to_value(&filtered)?
            } else {
                serde_json::to_value(&results)?
            };
            output.print(&toon_rust::encode(json, None));
        }
    }

    Ok(())
}

fn load_vector_index_cached(
    storage: &Storage,
    db_path: &Path,
    index_path: &Path,
) -> Result<&'static VectorIndex> {
    let (index, _loaded_now) = VECTOR_INDEX_CACHE.load(storage, db_path, index_path)?;
    Ok(index)
}

fn has_embeddings_for_types(doc_types: Option<&[search::DocType]>) -> bool {
    let Some(meta) = VECTOR_INDEX_CACHE.meta() else {
        return true;
    };
    if meta.embedding_count == 0 {
        return false;
    }

    let Some(types) = doc_types else {
        return true;
    };

    types.iter().any(|doc_type| {
        meta.type_counts
            .get(doc_type.as_str())
            .copied()
            .unwrap_or(0)
            > 0
    })
}

/// Get semantic search results from the vector index.
///
/// Returns empty vector if vector index is None, query is empty, or embedding fails.
fn get_semantic_results(
    vector_index: Option<&VectorIndex>,
    embedder: &dyn Embedder,
    canonical_query: &str,
    doc_types: Option<&[search::DocType]>,
    candidate_count: usize,
) -> Vec<xf::vector::VectorSearchResult> {
    let Some(vector_index) = vector_index else {
        return Vec::new();
    };

    if canonical_query.is_empty() {
        return Vec::new();
    }

    let Ok(query_embedding) = embedder.embed(canonical_query) else {
        return Vec::new();
    };

    let type_strs: Option<Vec<&str>> =
        doc_types.map(|types| types.iter().map(|t| t.as_str()).collect());

    vector_index.search_top_k(&query_embedding, candidate_count, type_strs.as_deref())
}

#[derive(Serialize)]
struct DmConversationContext {
    conversation_id: String,
    messages: Vec<DmContextMessage>,
}

#[derive(Serialize)]
struct DmContextMessage {
    id: String,
    sender_id: String,
    recipient_id: String,
    text: String,
    created_at: DateTime<Utc>,
    urls: Vec<TweetUrl>,
    media_urls: Vec<String>,
    is_match: bool,
    highlights: Vec<String>,
}

fn build_dm_context(
    results: &[SearchResult],
    storage: &Storage,
) -> Result<Vec<DmConversationContext>> {
    let mut conversation_order = Vec::new();
    let mut seen = HashSet::new();
    let mut matches: HashMap<String, HashMap<String, Vec<String>>> = HashMap::new();

    for result in results {
        let conv_id = result
            .metadata
            .get("conversation_id")
            .and_then(|value| value.as_str())
            .ok_or_else(|| anyhow::anyhow!("DM result missing conversation_id metadata"))?;
        let conv_id = conv_id.to_string();

        if seen.insert(conv_id.clone()) {
            conversation_order.push(conv_id.clone());
        }

        matches
            .entry(conv_id)
            .or_default()
            .entry(result.id.clone())
            .or_insert_with(|| result.highlights.clone());
    }

    let mut contexts = Vec::with_capacity(conversation_order.len());
    for conversation_id in conversation_order {
        let messages = storage.get_conversation_messages(&conversation_id)?;
        let mut context_messages = Vec::with_capacity(messages.len());

        let message_matches = matches.get(&conversation_id);
        for message in messages {
            let (is_match, highlights) = match message_matches {
                Some(match_map) => match_map
                    .get(&message.id)
                    .map_or((false, Vec::new()), |items| (true, items.clone())),
                None => (false, Vec::new()),
            };

            context_messages.push(DmContextMessage {
                id: message.id,
                sender_id: message.sender_id,
                recipient_id: message.recipient_id,
                text: message.text,
                created_at: message.created_at,
                urls: message.urls,
                media_urls: message.media_urls,
                is_match,
                highlights,
            });
        }

        contexts.push(DmConversationContext {
            conversation_id,
            messages: context_messages,
        });
    }

    Ok(contexts)
}

fn output_dm_context(
    cli: &Cli,
    contexts: &[DmConversationContext],
    highlight_enabled: bool,
    output: &Output,
) -> Result<()> {
    match cli.format {
        OutputFormat::Json => {
            output.print(&serde_json::to_string(contexts)?);
        }
        OutputFormat::JsonPretty => {
            output.print(&serde_json::to_string_pretty(contexts)?);
        }
        OutputFormat::Text => {
            print_dm_context_text(contexts, highlight_enabled, output);
        }
        OutputFormat::Toon => {
            let json = serde_json::to_value(contexts)?;
            output.print(&toon_rust::encode(json, None));
        }
        _ => {
            anyhow::bail!("--context only supports text, json, json-pretty, or toon output.");
        }
    }
    Ok(())
}

fn print_dm_context_text(
    contexts: &[DmConversationContext],
    highlight_enabled: bool,
    output: &Output,
) {
    for context in contexts {
        output.print(&format!(
            "[bold cyan]Conversation[/] [dim]{}[/]",
            context.conversation_id
        ));
        output.print(&format!("[dim]{}[/]", "─".repeat(CONTENT_DIVIDER_WIDTH)));

        for message in &context.messages {
            let timestamp = format_relative_date(message.created_at);
            output.print(&format!(
                "[dim]{} {} → {}[/]",
                timestamp,
                format_short_id(&message.sender_id),
                format_short_id(&message.recipient_id)
            ));

            let lines = textwrap::wrap(&message.text, 78);
            for line in lines {
                if highlight_enabled && message.is_match {
                    output.print(&format!("  [bold yellow]{line}[/]"));
                } else {
                    output.print(&format!("  {line}"));
                }
            }
            output.print("");
        }
    }
}

fn print_result(num: usize, result: &SearchResult, output: &Output) {
    let type_badge = match result.result_type {
        SearchResultType::Tweet => "[on blue]TWEET[/]",
        SearchResultType::Like => "[on magenta]LIKE[/]",
        SearchResultType::DirectMessage => "[on green]DM[/]",
        SearchResultType::GrokMessage => "[on yellow]GROK[/]",
    };

    // Result number is bold for easy scanning, ID is shown but dimmed
    // Score is hidden in text output (kept in JSON for programmatic use)
    output.print(&format!(
        "[bold]{}[/]. {} [dim]{}[/]",
        num,
        type_badge,
        format_short_id(&result.id)
    ));

    // Use highlighted text if available, otherwise use plain text
    let display_text = if result.highlights.is_empty() {
        result.text.clone()
    } else {
        // Convert HTML highlights to ANSI colors
        // Tantivy uses <b> tags for highlighting
        html_highlights_to_ansi(&result.highlights[0])
    };

    // Word wrap the text
    let wrapped = textwrap::wrap(&display_text, 78);
    for line in wrapped {
        output.print(&format!("   {line}"));
    }

    if result.created_at.timestamp() > 0 {
        output.print(&format!(
            "   [dim]{}[/]",
            format_relative_date(result.created_at)
        ));
    }

    output.print("");
}

/// Convert HTML-style highlights (from Tantivy) to ANSI colored text
fn html_highlights_to_ansi(html: &str) -> String {
    // Tantivy uses <b>...</b> for highlighting
    // We'll convert these to ANSI bold + yellow (or strip tags if color is disabled)
    if !control::SHOULD_COLORIZE.should_colorize()
        || no_color_env_set()
        || !std::io::stdout().is_terminal()
    {
        return html.replace("<b>", "").replace("</b>", "");
    }
    let mut result = html.to_string();

    // Replace opening tags with ANSI escape for bold yellow
    result = result.replace("<b>", "\x1b[1;33m");
    // Replace closing tags with ANSI reset
    result = result.replace("</b>", "\x1b[0m");

    result
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else if max_len <= 3 {
        // Can't fit any text + "...", just truncate without ellipsis
        // Find a valid UTF-8 char boundary
        let mut end = max_len;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        s[..end].to_string()
    } else {
        // Find a valid UTF-8 char boundary to avoid panic on multi-byte chars
        let mut end = max_len - 3;
        while end > 0 && !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

fn parse_date_arg(
    label: &str,
    value: &str,
    prefer_end: bool,
    verbose: bool,
) -> Result<DateTime<Utc>> {
    let parsed = date_parser::parse_date_flexible(value, prefer_end)
        .map_err(|err| anyhow::anyhow!("{label} date '{value}' could not be parsed: {err}"))?;

    if verbose {
        eprintln!("Parsed {label} '{value}' as {}", parsed.to_rfc3339());
    }

    Ok(parsed)
}

fn is_reply(result: &SearchResult) -> bool {
    if result.result_type != SearchResultType::Tweet {
        return false;
    }
    result
        .metadata
        .get("in_reply_to")
        .and_then(|v| v.as_str())
        .is_some()
}

fn apply_search_filters(
    results: &mut Vec<SearchResult>,
    since: Option<DateTime<Utc>>,
    until: Option<DateTime<Utc>>,
    replies_only: bool,
    no_replies: bool,
) {
    if since.is_some() || until.is_some() {
        results.retain(|r| {
            if matches!(r.result_type, SearchResultType::Like) {
                return false;
            }
            if let Some(since_dt) = since {
                if r.created_at < since_dt {
                    return false;
                }
            }
            if let Some(until_dt) = until {
                if r.created_at > until_dt {
                    return false;
                }
            }
            true
        });
    }

    if replies_only {
        results.retain(is_reply);
    } else if no_replies {
        results.retain(|r| !is_reply(r));
    }
}

fn engagement_score(result: &SearchResult) -> i64 {
    if result.result_type != SearchResultType::Tweet {
        return 0;
    }
    let favs = result
        .metadata
        .get("favorite_count")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(0);
    let rts = result
        .metadata
        .get("retweet_count")
        .and_then(serde_json::Value::as_i64)
        .unwrap_or(0);
    favs.saturating_add(rts)
}

fn apply_search_sort(results: &mut [SearchResult], sort: &SortOrder) {
    use std::cmp::Ordering;

    match sort {
        SortOrder::Relevance => {}
        SortOrder::Date => {
            results.sort_by(|a, b| {
                let cmp = a.created_at.cmp(&b.created_at);
                if cmp == Ordering::Equal {
                    b.score.total_cmp(&a.score)
                } else {
                    cmp
                }
            });
        }
        SortOrder::DateDesc => {
            results.sort_by(|a, b| {
                let cmp = b.created_at.cmp(&a.created_at);
                if cmp == Ordering::Equal {
                    b.score.total_cmp(&a.score)
                } else {
                    cmp
                }
            });
        }
        SortOrder::Engagement => {
            results.sort_by(|a, b| {
                let cmp = engagement_score(b).cmp(&engagement_score(a));
                if cmp == Ordering::Equal {
                    b.created_at.cmp(&a.created_at)
                } else {
                    cmp
                }
            });
        }
    }
}

#[cfg(test)]
mod search_filter_tests {
    use super::apply_search_filters;
    use chrono::{TimeZone, Utc};
    use xf::{SearchResult, SearchResultType};

    fn make_result(
        result_type: SearchResultType,
        created_at: chrono::DateTime<Utc>,
    ) -> SearchResult {
        SearchResult {
            result_type,
            id: "id".to_string(),
            text: "text".to_string(),
            created_at,
            score: 1.0,
            highlights: Vec::new(),
            metadata: serde_json::json!({}),
            rerank_score: None,
        }
    }

    #[test]
    fn date_filters_exclude_likes_without_timestamps() {
        let mut results = vec![
            make_result(SearchResultType::Like, Utc.timestamp_opt(0, 0).unwrap()),
            make_result(
                SearchResultType::Tweet,
                Utc.with_ymd_and_hms(2024, 1, 15, 12, 0, 0)
                    .single()
                    .unwrap(),
            ),
        ];

        let until = Some(Utc.with_ymd_and_hms(2024, 2, 1, 0, 0, 0).single().unwrap());
        apply_search_filters(&mut results, None, until, false, false);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].result_type, SearchResultType::Tweet);
    }
}

fn validate_output_fields(fields: &[String]) -> Result<()> {
    for field in fields {
        if !VALID_OUTPUT_FIELDS.contains(&field.as_str()) {
            let mut suggestions = Vec::new();

            // Check for close matches (typos)
            if let Some(closest) = find_closest_match(field, VALID_OUTPUT_FIELDS, None) {
                suggestions.push(format!("Did you mean '{closest}'?"));
            }

            suggestions.push(format!("Valid fields: {}", VALID_OUTPUT_FIELDS.join(", ")));

            let suggestion_refs: Vec<&str> = suggestions.iter().map(String::as_str).collect();
            anyhow::bail!(
                "{}",
                format_error(&format!("Unknown field: '{field}'"), "", &suggestion_refs,)
            );
        }
    }
    Ok(())
}

fn validate_mrl_dims(dims: usize) -> Result<()> {
    match dims {
        256 | 512 | 768 | 1024 => Ok(()),
        _ => anyhow::bail!("MRL dimensions must be 256, 512, 768, or 1024; got {dims}"),
    }
}

fn filter_results_fields(
    results: &[SearchResult],
    fields: &[String],
) -> Result<Vec<serde_json::Value>> {
    let mut filtered = Vec::with_capacity(results.len());
    for result in results {
        let value = serde_json::to_value(result)?;
        let obj = value.as_object().ok_or_else(|| {
            anyhow::anyhow!("Failed to serialize search result for field filtering.")
        })?;
        let mut new_obj = serde_json::Map::new();
        for field in fields {
            if let Some(val) = obj.get(field) {
                new_obj.insert(field.clone(), val.clone());
            }
        }
        filtered.push(serde_json::Value::Object(new_obj));
    }
    Ok(filtered)
}

#[allow(clippy::too_many_lines)]
fn cmd_stats(cli: &Cli, args: &cli::StatsArgs, output: &Output) -> Result<()> {
    let db_path = get_db_path(cli);

    if !db_path.exists() {
        anyhow::bail!(
            "{}",
            format_error(
                "No archive indexed yet",
                "You need to index your X data archive first.",
                &["Run: xf index ~/Downloads/twitter-archive"],
            )
        );
    }

    let storage = Storage::open(&db_path)?;
    let stats = storage.get_stats()?;

    // --detailed shows all analytics (temporal + engagement + content)
    let show_temporal = args.temporal || args.detailed;
    let show_engagement = args.engagement || args.detailed;
    let show_content = args.content || args.detailed;

    // Show progress for large archives when computing detailed analytics
    if args.detailed && stats.tweets_count > 10_000 {
        output.eprint("Computing detailed analytics...");
    }

    // Temporal analytics uses efficient SQL aggregations
    let temporal = if show_temporal {
        Some(TemporalStats::compute(&storage)?)
    } else {
        None
    };

    // Engagement analytics
    let engagement = if show_engagement {
        Some(EngagementStats::compute(&storage, args.top)?)
    } else {
        None
    };

    // Content analytics - also provides top_hashtags and top_mentions efficiently
    let content = if show_content || args.hashtags || args.mentions {
        Some(ContentStats::compute(&storage, args.top)?)
    } else {
        None
    };

    // Extract top_hashtags/mentions from ContentStats if requested separately
    #[allow(clippy::cast_possible_truncation)]
    let top_hashtags = if args.hashtags && !show_content {
        content.as_ref().map(|c| {
            c.top_hashtags
                .iter()
                .map(|t| CountItem {
                    value: t.tag.clone(),
                    count: t.count as usize,
                })
                .collect::<Vec<_>>()
        })
    } else {
        None
    };

    #[allow(clippy::cast_possible_truncation)]
    let top_mentions = if args.mentions && !show_content {
        content.as_ref().map(|c| {
            c.top_mentions
                .iter()
                .map(|t| CountItem {
                    value: t.tag.clone(),
                    count: t.count as usize,
                })
                .collect::<Vec<_>>()
        })
    } else {
        None
    };

    let needs_extended =
        show_temporal || show_engagement || show_content || args.hashtags || args.mentions;

    // For backward compatibility with JSON output, include monthly breakdown in detailed
    let detailed = if args.detailed && temporal.is_some() {
        temporal
            .as_ref()
            .map(|t| build_monthly_counts_from_daily(&t.daily_counts))
    } else {
        None
    };

    match cli.format {
        OutputFormat::Json | OutputFormat::JsonPretty => {
            if needs_extended {
                let extended = StatsExtended {
                    stats,
                    detailed,
                    top_hashtags,
                    top_mentions,
                    temporal,
                    engagement,
                    content,
                };
                let json = if matches!(cli.format, OutputFormat::JsonPretty) {
                    serde_json::to_string_pretty(&extended)?
                } else {
                    serde_json::to_string(&extended)?
                };
                output.print(&json);
            } else {
                let json = if matches!(cli.format, OutputFormat::JsonPretty) {
                    serde_json::to_string_pretty(&stats)?
                } else {
                    serde_json::to_string(&stats)?
                };
                output.print(&json);
            }
        }
        OutputFormat::Toon => {
            if needs_extended {
                let extended = StatsExtended {
                    stats,
                    detailed,
                    top_hashtags,
                    top_mentions,
                    temporal,
                    engagement,
                    content,
                };
                let json = serde_json::to_value(&extended)?;
                output.print(&toon_rust::encode(json, None));
            } else {
                let json = serde_json::to_value(&stats)?;
                output.print(&toon_rust::encode(json, None));
            }
        }
        _ => {
            // Show fancy banner for --detailed mode
            if args.detailed {
                output.print(&format!(
                    "[bright_blue]{}[/]",
                    "═".repeat(HEADER_DIVIDER_WIDTH)
                ));
                output.print("[bold on bright_blue]              ARCHIVE ANALYTICS DASHBOARD              [/]");
                output.print(&format!(
                    "[bright_blue]{}[/]",
                    "═".repeat(HEADER_DIVIDER_WIDTH)
                ));
                output.print("");
            }

            output.print("[bold cyan]Overview[/]");
            output.print(&format!("[dim]{}[/]", "─".repeat(CONTENT_DIVIDER_WIDTH)));
            output.print(&format!(
                "  [dim]{:<20}[/] [bold]{:>10}[/]",
                "Tweets:",
                format_number(stats.tweets_count)
            ));
            output.print(&format!(
                "  [dim]{:<20}[/] [bold]{:>10}[/]",
                "Likes:",
                format_number(stats.likes_count)
            ));
            output.print(&format!(
                "  [dim]{:<20}[/] [bold]{:>10}[/]",
                "DM Conversations:",
                format_number(stats.dm_conversations_count)
            ));
            output.print(&format!(
                "  [dim]{:<20}[/] [bold]{:>10}[/]",
                "DM Messages:",
                format_number(stats.dms_count)
            ));
            output.print(&format!(
                "  [dim]{:<20}[/] [bold]{:>10}[/]",
                "Grok Messages:",
                format_number(stats.grok_messages_count)
            ));
            output.print(&format!(
                "  [dim]{:<20}[/] [bold]{:>10}[/]",
                "Followers:",
                format_number(stats.followers_count)
            ));
            output.print(&format!(
                "  [dim]{:<20}[/] [bold]{:>10}[/]",
                "Following:",
                format_number(stats.following_count)
            ));
            output.print(&format!(
                "  [dim]{:<20}[/] [bold]{:>10}[/]",
                "Blocks:",
                format_number(stats.blocks_count)
            ));
            output.print(&format!(
                "  [dim]{:<20}[/] [bold]{:>10}[/]",
                "Mutes:",
                format_number(stats.mutes_count)
            ));
            output.print(&format!("[dim]{}[/]", "─".repeat(CONTENT_DIVIDER_WIDTH)));

            if let (Some(first), Some(last)) = (stats.first_tweet_date, stats.last_tweet_date) {
                output.print(&format!(
                    "  [dim]{}[/] [bold]{}[/]",
                    "First tweet:",
                    format_relative_date(first)
                ));
                output.print(&format!(
                    "  [dim]{}[/] [bold]{}[/]",
                    "Last tweet:",
                    format_relative_date(last)
                ));
            }

            if let Some(detailed) = detailed {
                if !detailed.is_empty() {
                    output.print("");
                    output.print("[bold cyan]Tweets by Month[/]");
                    output.print(&format!("[dim]{}[/]", "─".repeat(CONTENT_DIVIDER_WIDTH)));
                    for entry in detailed {
                        output.print(&format!(
                            "  {:04}-{:02}: [bold]{}[/]",
                            entry.year,
                            entry.month,
                            format_number_usize(entry.count)
                        ));
                    }
                }
            }

            if let Some(items) = top_hashtags {
                if !items.is_empty() {
                    output.print("");
                    output.print("[bold cyan]Top Hashtags[/]");
                    output.print(&format!("[dim]{}[/]", "─".repeat(CONTENT_DIVIDER_WIDTH)));
                    for item in items {
                        output.print(&format!(
                            "  {:<20} [bold]{}[/]",
                            item.value,
                            format_number_usize(item.count)
                        ));
                    }
                }
            }

            if let Some(items) = top_mentions {
                if !items.is_empty() {
                    output.print("");
                    output.print("[bold cyan]Top Mentions[/]");
                    output.print(&format!("[dim]{}[/]", "─".repeat(CONTENT_DIVIDER_WIDTH)));
                    for item in items {
                        output.print(&format!(
                            "  {:<20} [bold]{}[/]",
                            item.value,
                            format_number_usize(item.count)
                        ));
                    }
                }
            }

            #[allow(clippy::cast_possible_wrap)]
            if let Some(ref temporal) = temporal {
                output.print("");
                output.print("[bold cyan]Temporal Patterns[/]");
                output.print(&format!("[dim]{}[/]", "─".repeat(CONTENT_DIVIDER_WIDTH)));

                // Activity sparkline
                let sparkline = stats_analytics::sparkline_from_daily(&temporal.daily_counts, 50);
                output.print(&format!("  Activity: [dim]{sparkline}[/]"));

                // Key metrics
                output.print(&format!(
                    "  [dim]{:<25}[/] [bold]{:>10}[/]",
                    "Active days:",
                    format_number_u64(temporal.active_days_count)
                ));
                output.print(&format!(
                    "  [dim]{:<25}[/] [bold]{:>10}[/]",
                    "Total days in range:",
                    format_number_u64(temporal.total_days_in_range)
                ));
                output.print(&format!(
                    "  [dim]{:<25}[/] [bold]{:>10.1}[/]",
                    "Avg tweets/active day:", temporal.avg_tweets_per_active_day
                ));

                // Most active day
                if let Some(day) = temporal.most_active_day {
                    output.print(&format!(
                        "  [dim]{:<25}[/] [bold]{}[/] ([bold]{}[/])",
                        "Most active day:",
                        format_naive_date(day),
                        format_number_u64(temporal.most_active_day_count)
                    ));
                }

                // Most active hour
                let hour_label = format!("{:02}:00", temporal.most_active_hour);
                output.print(&format!(
                    "  [dim]{:<25}[/] [bold]{}[/] ([bold]{}[/])",
                    "Most active hour:",
                    hour_label,
                    format_number_u64(temporal.most_active_hour_count)
                ));

                // Longest gap
                if temporal.longest_gap_days > 1 {
                    let gap_info = if let (Some(start), Some(end)) =
                        (temporal.longest_gap_start, temporal.longest_gap_end)
                    {
                        format!(
                            "{} days ({} to {})",
                            format_number(temporal.longest_gap_days),
                            format_naive_date(start),
                            format_naive_date(end)
                        )
                    } else {
                        format!("{} days", format_number(temporal.longest_gap_days))
                    };
                    output.print(&format!(
                        "  [dim]{:<25}[/] [yellow]{}[/]",
                        "Longest gap:", gap_info
                    ));
                }

                // Hourly distribution
                output.print("");
                output.print(&format!("  [dim]{} (00-23):[/]", "Hourly distribution"));
                let hourly_sparkline =
                    stats_analytics::format_hourly_sparkline(&temporal.hourly_distribution);
                output.print(&format!("  {hourly_sparkline}"));

                // Day of week distribution
                output.print("");
                output.print(&format!("  [dim]{}:[/]", "Day of week"));
                let dow_chart =
                    stats_analytics::format_dow_distribution(&temporal.dow_distribution);
                for line in dow_chart.lines() {
                    output.print(&format!("  {line}"));
                }
            }

            #[allow(clippy::cast_possible_wrap)]
            if let Some(ref engagement) = engagement {
                output.print("");
                output.print("[bold cyan]Engagement Analytics[/]");
                output.print(&format!("[dim]{}[/]", "─".repeat(CONTENT_DIVIDER_WIDTH)));

                // Summary metrics
                output.print(&format!(
                    "  Total Likes: [bold]{}[/] | Total Retweets: [bold]{}[/]",
                    format_number_u64(engagement.total_likes),
                    format_number_u64(engagement.total_retweets)
                ));
                output.print(&format!(
                    "  Average per Tweet: [bold]{:.1}[/] | Median: [bold]{}[/]",
                    engagement.avg_engagement,
                    format_number_u64(engagement.median_engagement)
                ));

                // Trend sparkline
                if !engagement.monthly_trend.is_empty() {
                    output.print("");
                    output.print(&format!("  [dim]{} (monthly avg):[/]", "Engagement trend"));
                    let trend_sparkline =
                        stats_analytics::sparkline_from_monthly(&engagement.monthly_trend, 24);
                    output.print(&format!("  {trend_sparkline}"));
                }

                // Likes histogram
                output.print("");
                output.print(&format!("  [dim]{}:[/]", "Likes distribution"));
                let histogram =
                    stats_analytics::format_likes_histogram(&engagement.likes_histogram);
                for line in histogram.lines() {
                    output.print(&format!("  {line}"));
                }

                // Top performing tweets
                if !engagement.top_tweets.is_empty() {
                    output.print("");
                    output.print(&format!("  [dim]{}:[/]", "Top performing tweets"));
                    for (i, tweet) in engagement.top_tweets.iter().enumerate() {
                        output.print(&format!(
                            "  {}. [[bold]{}[/] [dim]♥[/] [bold]{}[/]] \"[dim]{}[/]\" ({})",
                            i + 1,
                            format_number_u64(tweet.likes),
                            format_number_u64(tweet.retweets),
                            tweet.text_preview,
                            format_relative_date(tweet.created_at)
                        ));
                    }
                }
            }

            #[allow(clippy::cast_possible_wrap)]
            if let Some(ref content) = content {
                output.print("");
                output.print("[bold cyan]Content Analysis[/]");
                output.print(&format!("[dim]{}[/]", "─".repeat(CONTENT_DIVIDER_WIDTH)));

                // Content type ratios
                output.print(&format!(
                    "  [dim]{:<25}[/] [bold]{:>6.1}%[/]",
                    "Tweets with media:", content.media_ratio
                ));
                output.print(&format!(
                    "  [dim]{:<25}[/] [bold]{:>6.1}%[/]",
                    "Tweets with links:", content.link_ratio
                ));
                output.print(&format!(
                    "  [dim]{:<25}[/] [bold]{:>6.1}%[/]",
                    "Replies:", content.reply_ratio
                ));
                output.print(&format!(
                    "  [dim]{:<25}[/] [bold]{:>10}[/]",
                    "Self-threads:",
                    format_number_u64(content.thread_count)
                ));
                output.print(&format!(
                    "  [dim]{:<25}[/] [bold]{:>10}[/]",
                    "Standalone tweets:",
                    format_number_u64(content.standalone_count)
                ));

                // Tweet length
                output.print("");
                output.print(&format!(
                    "  [dim]{:<25}[/] [bold]{:.1} chars[/]",
                    "Average tweet length:", content.avg_tweet_length
                ));
                output.print("");
                output.print(&format!("  [dim]{}:[/]", "Length distribution"));
                let length_chart =
                    stats_analytics::format_length_distribution(&content.length_distribution);
                for line in length_chart.lines() {
                    output.print(&format!("  {line}"));
                }

                // Top hashtags
                if !content.top_hashtags.is_empty() {
                    output.print("");
                    output.print(&format!("  [dim]{}:[/]", "Top hashtags"));
                    for tag in content.top_hashtags.iter().take(6) {
                        output.print(&format!(
                            "    #{:<20} [bold]{}[/]",
                            tag.tag,
                            format_number_u64(tag.count)
                        ));
                    }
                }

                // Top mentions
                if !content.top_mentions.is_empty() {
                    output.print("");
                    output.print(&format!("  [dim]{}:[/]", "Top mentions"));
                    for mention in content.top_mentions.iter().take(6) {
                        output.print(&format!(
                            "    @{:<20} [bold]{}[/]",
                            mention.tag,
                            format_number_u64(mention.count)
                        ));
                    }
                }
            }
        }
    }

    Ok(())
}

#[derive(Serialize)]
struct StatsExtended {
    stats: ArchiveStats,
    #[serde(skip_serializing_if = "Option::is_none")]
    detailed: Option<Vec<StatsPeriod>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_hashtags: Option<Vec<CountItem>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_mentions: Option<Vec<CountItem>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temporal: Option<TemporalStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    engagement: Option<EngagementStats>,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<ContentStats>,
}

#[derive(Serialize)]
struct StatsPeriod {
    year: i32,
    month: u32,
    count: usize,
}

#[derive(Serialize)]
struct CountItem {
    value: String,
    count: usize,
}

/// Build monthly counts from pre-computed daily counts (efficient SQL-based approach).
#[allow(clippy::cast_possible_truncation)]
fn build_monthly_counts_from_daily(
    daily_counts: &[stats_analytics::DailyCount],
) -> Vec<StatsPeriod> {
    use std::collections::BTreeMap;

    let mut counts: BTreeMap<(i32, u32), usize> = BTreeMap::new();
    for day in daily_counts {
        let key = (day.date.year(), day.date.month());
        *counts.entry(key).or_insert(0) += day.count as usize;
    }

    counts
        .into_iter()
        .map(|((year, month), count)| StatsPeriod { year, month, count })
        .collect()
}

fn format_naive_date(date: NaiveDate) -> String {
    date.and_hms_opt(0, 0, 0)
        .map(|dt| Utc.from_utc_datetime(&dt))
        .map_or_else(
            || date.format("%b %d, %Y").to_string(),
            format_relative_date,
        )
}

fn cmd_tweet(cli: &Cli, args: &cli::TweetArgs, output: &Output) -> Result<()> {
    let db_path = get_db_path(cli);
    let storage = Storage::open(&db_path)?;

    if args.thread {
        return cmd_tweet_thread(cli, &storage, args, output);
    }

    let tweet = storage.get_tweet(&args.id)?;

    match tweet {
        Some(t) => match cli.format {
            OutputFormat::Json | OutputFormat::JsonPretty => {
                let json = if matches!(cli.format, OutputFormat::JsonPretty) {
                    serde_json::to_string_pretty(&t)?
                } else {
                    serde_json::to_string(&t)?
                };
                output.print(&json);
            }
            OutputFormat::Toon => {
                let json = serde_json::to_value(&t)?;
                output.print(&toon_rust::encode(json, None));
            }
            _ => {
                output.print(&format!("[dim]{}[/]", "─".repeat(CONTENT_DIVIDER_WIDTH)));
                output.print(&t.full_text);
                output.print(&format!("[dim]{}[/]", "─".repeat(CONTENT_DIVIDER_WIDTH)));
                output.print(&format!(
                    "  [dim]ID: {}  Date: {}[/]",
                    t.id,
                    format_relative_date(t.created_at)
                ));
                if args.engagement {
                    output.print(&format!(
                        "  [bold]{}[/] likes  [bold]{}[/] retweets",
                        format_number(t.favorite_count),
                        format_number(t.retweet_count)
                    ));
                }
                if !t.hashtags.is_empty() {
                    output.print(&format!("  Hashtags: [blue]{}[/]", t.hashtags.join(", ")));
                }
                if let Some(reply_to) = &t.in_reply_to_screen_name {
                    output.print(&format!("  [dim]Reply to:[/] [bold]@{reply_to}[/]"));
                }
            }
        },
        None => {
            output.print(&format!("[red]Tweet {} not found.[/]", args.id));
        }
    }

    Ok(())
}

#[allow(clippy::too_many_lines)]
fn cmd_list(cli: &Cli, args: &cli::ListArgs, output: &Output) -> Result<()> {
    let db_path = get_db_path(cli);

    if matches!(args.what, ListTarget::Files) {
        let config = Config::load();
        let Some(archive_path) = config.paths.archive else {
            output.print("[yellow]No archive path configured. Use 'xf config --archive <path>' or set XF_ARCHIVE.[/]");
            return Ok(());
        };

        if !archive_path.exists() {
            output.print(&format!(
                "[red]Archive path not found: {}[/]",
                archive_path.display()
            ));
            return Ok(());
        }

        let parser = ArchiveParser::new(&archive_path);
        let files = parser.list_data_files()?;
        if files.is_empty() {
            output.print("[yellow]No data files found in archive.[/]");
            return Ok(());
        }

        output.print(&format!(
            "[dim]Showing[/] [bold]{}[/] files:\n",
            format_number_usize(files.len())
        ));
        for file in &files {
            output.print(file);
        }
        return Ok(());
    }

    if !db_path.exists() {
        anyhow::bail!(
            "{}",
            format_error(
                "No archive indexed yet",
                "You need to index your X data archive first.",
                &["Run: xf index ~/Downloads/twitter-archive"],
            )
        );
    }

    let storage = Storage::open(&db_path)?;
    let limit = Some(args.limit);

    match args.what {
        ListTarget::Files => unreachable!(),
        ListTarget::Tweets => {
            let tweets = storage.get_all_tweets(limit)?;
            output.print(&format!(
                "[dim]Showing[/] [bold]{}[/] tweets:\n",
                format_number_usize(tweets.len())
            ));
            for tweet in &tweets {
                let date = format_relative_date(tweet.created_at);
                let text = truncate_text(&tweet.full_text, 80);
                output.print(&format!(
                    "[dim]{} {}[/] {}",
                    date,
                    format_short_id(&tweet.id),
                    text
                ));
            }
        }
        ListTarget::Likes => {
            let likes = storage.get_all_likes(limit)?;
            output.print(&format!(
                "[dim]Showing[/] [bold]{}[/] likes:\n",
                format_number_usize(likes.len())
            ));
            for like in &likes {
                let text = like
                    .full_text
                    .as_ref()
                    .map_or_else(|| "[No text]".to_string(), |t| truncate_text(t, 80));
                output.print(&format!(
                    "[dim]{}[/] {}",
                    format_short_id(&like.tweet_id),
                    text
                ));
            }
        }
        ListTarget::Dms => {
            let dms = storage.get_all_dms(limit)?;
            output.print(&format!(
                "[dim]Showing[/] [bold]{}[/] DM messages:\n",
                format_number_usize(dms.len())
            ));
            for dm in &dms {
                let date = format_relative_date(dm.created_at);
                let text = truncate_text(&dm.text, 60);
                output.print(&format!(
                    "[dim]{} {} → {}[/] {}",
                    date,
                    format_short_id(&dm.sender_id),
                    format_short_id(&dm.recipient_id),
                    text
                ));
            }
        }
        ListTarget::Conversations => {
            let conversations = storage.get_dm_conversation_summaries(limit)?;
            output.print(&format!(
                "[dim]Showing[/] [bold]{}[/] conversations:\n",
                format_number_usize(conversations.len())
            ));
            for convo in &conversations {
                let participants = if convo.participant_ids.is_empty() {
                    "[unknown]".to_string()
                } else {
                    convo
                        .participant_ids
                        .iter()
                        .map(|id| format_short_id(id))
                        .collect::<Vec<_>>()
                        .join(", ")
                };
                let first = format_optional_date(convo.first_message_at);
                let last = format_optional_date(convo.last_message_at);
                output.print(&format!(
                    "[dim]{}[/] [bold]{}[/] msgs  [dim]{} → {}  {}[/]",
                    format_short_id(&convo.conversation_id),
                    format_number(convo.message_count),
                    first,
                    last,
                    participants
                ));
            }
        }
        ListTarget::Followers => {
            let followers = storage.get_all_followers(limit)?;
            output.print(&format!(
                "[dim]Showing[/] [bold]{}[/] followers:\n",
                format_number_usize(followers.len())
            ));
            for follower in &followers {
                let link = follower.user_link.as_deref().unwrap_or("[no link]");
                output.print(&format!(
                    "[dim]{} {}[/]",
                    format_short_id(&follower.account_id),
                    link
                ));
            }
        }
        ListTarget::Following => {
            let following = storage.get_all_following(limit)?;
            output.print(&format!(
                "[dim]Showing[/] [bold]{}[/] following:\n",
                format_number_usize(following.len())
            ));
            for f in &following {
                let link = f.user_link.as_deref().unwrap_or("[no link]");
                output.print(&format!(
                    "[dim]{} {}[/]",
                    format_short_id(&f.account_id),
                    link
                ));
            }
        }
        ListTarget::Blocks => {
            let blocks = storage.get_all_blocks(limit)?;
            output.print(&format!(
                "[dim]Showing[/] [bold]{}[/] blocks:\n",
                format_number_usize(blocks.len())
            ));
            for block in &blocks {
                let link = block.user_link.as_deref().unwrap_or("[no link]");
                output.print(&format!(
                    "[dim]{} {}[/]",
                    format_short_id(&block.account_id),
                    link
                ));
            }
        }
        ListTarget::Mutes => {
            let mutes = storage.get_all_mutes(limit)?;
            output.print(&format!(
                "[dim]Showing[/] [bold]{}[/] mutes:\n",
                format_number_usize(mutes.len())
            ));
            for mute in &mutes {
                let link = mute.user_link.as_deref().unwrap_or("[no link]");
                output.print(&format!(
                    "[dim]{} {}[/]",
                    format_short_id(&mute.account_id),
                    link
                ));
            }
        }
    }

    Ok(())
}

/// Truncate text to a maximum length, adding ellipsis if needed.
/// Uses character count, not byte count, to properly handle UTF-8.
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

#[allow(clippy::too_many_lines)]
fn cmd_export(cli: &Cli, args: &cli::ExportArgs, output: &Output) -> Result<()> {
    let db_path = get_db_path(cli);

    if !db_path.exists() {
        anyhow::bail!(
            "{}",
            format_error(
                "No archive indexed yet",
                "You need to index your X data archive first.",
                &["Run: xf index ~/Downloads/twitter-archive"],
            )
        );
    }

    let storage = Storage::open(&db_path)?;

    // Build export data based on target
    let export_data = match args.what {
        ExportTarget::Tweets => {
            let tweets = storage.get_all_tweets(args.limit)?;
            format_export(&tweets, &args.format)?
        }
        ExportTarget::Likes => {
            let likes = storage.get_all_likes(args.limit)?;
            format_export(&likes, &args.format)?
        }
        ExportTarget::Dms => {
            let dms = storage.get_all_dms(args.limit)?;
            format_export(&dms, &args.format)?
        }
        ExportTarget::Followers => {
            let followers = storage.get_all_followers(args.limit)?;
            format_export(&followers, &args.format)?
        }
        ExportTarget::Following => {
            let following = storage.get_all_following(args.limit)?;
            format_export(&following, &args.format)?
        }
        ExportTarget::All => {
            // For "all", we create a combined structure
            let tweets = storage.get_all_tweets(args.limit)?;
            let likes = storage.get_all_likes(args.limit)?;
            let dms = storage.get_all_dms(args.limit)?;
            let followers = storage.get_all_followers(args.limit)?;
            let following = storage.get_all_following(args.limit)?;

            match args.format {
                ExportFormat::Json => {
                    let combined = serde_json::json!({
                        "tweets": tweets,
                        "likes": likes,
                        "dms": dms,
                        "followers": followers,
                        "following": following
                    });
                    serde_json::to_string_pretty(&combined)?
                }
                ExportFormat::Jsonl => {
                    let mut jsonl_lines = Vec::new();
                    for t in &tweets {
                        jsonl_lines.push(format!(
                            r#"{{"type":"tweet","data":{}}}"#,
                            serde_json::to_string(t)?
                        ));
                    }
                    for l in &likes {
                        jsonl_lines.push(format!(
                            r#"{{"type":"like","data":{}}}"#,
                            serde_json::to_string(l)?
                        ));
                    }
                    for d in &dms {
                        jsonl_lines.push(format!(
                            r#"{{"type":"dm","data":{}}}"#,
                            serde_json::to_string(d)?
                        ));
                    }
                    for f in &followers {
                        jsonl_lines.push(format!(
                            r#"{{"type":"follower","data":{}}}"#,
                            serde_json::to_string(f)?
                        ));
                    }
                    for f in &following {
                        jsonl_lines.push(format!(
                            r#"{{"type":"following","data":{}}}"#,
                            serde_json::to_string(f)?
                        ));
                    }
                    jsonl_lines.join("\n")
                }
                ExportFormat::Csv => {
                    anyhow::bail!(
                        "CSV export not supported for 'all' target. Export individual types instead."
                    );
                }
            }
        }
    };

    // Write to file or stdout
    if let Some(path) = &args.output {
        std::fs::write(path, &export_data)?;
        output.print(&format!(
            "[green]✓[/] Exported to [bold]{}[/]",
            path.display()
        ));
    } else {
        output.print(&export_data);
    }

    Ok(())
}

/// Format data for export based on the specified format
fn format_export<T: serde::Serialize>(data: &[T], format: &ExportFormat) -> Result<String> {
    match format {
        ExportFormat::Json => Ok(serde_json::to_string_pretty(data)?),
        ExportFormat::Jsonl => {
            let lines: Vec<String> = data
                .iter()
                .map(|item| serde_json::to_string(item))
                .collect::<std::result::Result<_, _>>()?;
            Ok(lines.join("\n"))
        }
        ExportFormat::Csv => {
            if data.is_empty() {
                return Ok(String::new());
            }
            // Use serde_json to get a consistent field representation
            let first = serde_json::to_value(&data[0])?;
            if let serde_json::Value::Object(map) = first {
                let headers: Vec<&str> = map.keys().map(String::as_str).collect();
                let mut output = headers.join(",");
                output.push('\n');

                for item in data {
                    let val = serde_json::to_value(item)?;
                    if let serde_json::Value::Object(obj) = val {
                        let row: Vec<String> = headers
                            .iter()
                            .map(|&h| obj.get(h).map(csv_escape).unwrap_or_default())
                            .collect();
                        output.push_str(&row.join(","));
                        output.push('\n');
                    }
                }
                Ok(output)
            } else {
                anyhow::bail!("Data structure not suitable for CSV export");
            }
        }
    }
}

/// Escape a JSON value for CSV output
fn csv_escape(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => String::new(),
        serde_json::Value::Bool(b) => b.to_string(),
        serde_json::Value::Number(n) => n.to_string(),
        serde_json::Value::String(s) => {
            // Quote if contains comma, quote, newline, or carriage return per RFC 4180
            let sanitized = csv_escape_text(s);
            if sanitized.contains(',') || sanitized.contains('"') {
                format!("\"{sanitized}\"")
            } else {
                sanitized
            }
        }
        serde_json::Value::Array(arr) => {
            let inner = serde_json::to_string(arr).unwrap_or_default();
            let sanitized = csv_escape_text(&inner);
            format!("\"{sanitized}\"")
        }
        serde_json::Value::Object(obj) => {
            let inner = serde_json::to_string(obj).unwrap_or_default();
            let sanitized = csv_escape_text(&inner);
            format!("\"{sanitized}\"")
        }
    }
}

fn cmd_config(cli: &Cli, args: &cli::ConfigArgs, output: &Output) -> Result<()> {
    let mut config = Config::load();
    let set_present = args.set.is_some();
    let archive_present = args.archive.is_some();

    if let Some(set) = &args.set {
        apply_config_set(&mut config, set)?;
    }

    if let Some(archive) = &args.archive {
        config.paths.archive = Some(archive.clone());
    }

    if set_present || archive_present {
        config
            .save()
            .with_context(|| "Failed to save config file".to_string())?;
        output.print("[green]✓ Updated configuration[/]");
    }
    if args.show {
        output.print("[bold cyan]Current Configuration[/]");
        output.print(&format!("  Database: {}", get_db_path(cli).display()));
        output.print(&format!("  Index: {}", get_index_path(cli).display()));
        if let Some(archive) = &config.paths.archive {
            output.print(&format!("  Archive: {}", archive.display()));
        }
    }
    Ok(())
}

fn cmd_tweet_thread(
    cli: &Cli,
    storage: &Storage,
    args: &cli::TweetArgs,
    output: &Output,
) -> Result<()> {
    let thread = storage.get_tweet_thread(&args.id)?;

    if thread.is_empty() {
        output.print(&format!("[red]Tweet {} not found.[/]", args.id));
        return Ok(());
    }

    match cli.format {
        OutputFormat::Json | OutputFormat::JsonPretty => {
            let json = if matches!(cli.format, OutputFormat::JsonPretty) {
                serde_json::to_string_pretty(&thread)?
            } else {
                serde_json::to_string(&thread)?
            };
            output.print(&json);
        }
        OutputFormat::Toon => {
            let json = serde_json::to_value(&thread)?;
            output.print(&toon_rust::encode(json, None));
        }
        _ => {
            output.print("[bold cyan]Thread[/]");
            output.print(&format!("[dim]{}[/]", "─".repeat(CONTENT_DIVIDER_WIDTH)));
            for tweet in &thread {
                let date = format_relative_date(tweet.created_at);
                let text = truncate_text(&tweet.full_text, 100);
                output.print(&format!(
                    "[dim]{} {}[/] {}",
                    date,
                    format_short_id(&tweet.id),
                    text
                ));
                if args.engagement {
                    output.print(&format!(
                        "  [bold]{}[/] likes  [bold]{}[/] retweets",
                        format_number(tweet.favorite_count),
                        format_number(tweet.retweet_count)
                    ));
                }
            }
        }
    }

    Ok(())
}

fn apply_config_set(config: &mut Config, raw: &str) -> Result<()> {
    let (key, value) = raw
        .split_once('=')
        .map(|(k, v)| (k.trim(), v.trim()))
        .ok_or_else(|| anyhow::anyhow!("Invalid --set format. Use key=value."))?;

    if key.is_empty() {
        anyhow::bail!("Invalid --set key. Use key=value.");
    }

    match key {
        "db" | "paths.db" => {
            config.paths.db = parse_optional_path(value);
        }
        "index" | "paths.index" => {
            config.paths.index = parse_optional_path(value);
        }
        "archive" | "paths.archive" => {
            config.paths.archive = parse_optional_path(value);
        }
        "search.default_limit" => {
            config.search.default_limit = parse_usize(value, key)?;
        }
        "search.highlight" => {
            config.search.highlight = parse_bool(value, key)?;
        }
        "search.fuzzy" => {
            config.search.fuzzy = parse_bool(value, key)?;
        }
        "search.min_score" => {
            let parsed = parse_f32(value, key)?;
            if !(0.0..=1.0).contains(&parsed) {
                anyhow::bail!("{key} must be between 0.0 and 1.0.");
            }
            config.search.min_score = parsed;
        }
        "search.cache_size" => {
            config.search.cache_size = parse_usize(value, key)?;
        }
        "indexing.parallel" => {
            config.indexing.parallel = parse_bool(value, key)?;
        }
        "indexing.buffer_size_mb" => {
            config.indexing.buffer_size_mb = parse_usize(value, key)?;
        }
        "indexing.threads" => {
            config.indexing.threads = parse_usize(value, key)?;
        }
        "indexing.skip_types" => {
            config.indexing.skip_types = parse_csv_list(value);
        }
        "output.format" => {
            if value.is_empty() {
                anyhow::bail!("output.format cannot be empty.");
            }
            config.output.format = value.to_string();
        }
        "output.colors" => {
            config.output.colors = parse_bool(value, key)?;
        }
        "output.quiet" => {
            config.output.quiet = parse_bool(value, key)?;
        }
        "output.timings" => {
            config.output.timings = parse_bool(value, key)?;
        }
        _ => {
            let mut suggestions = Vec::new();

            // Check for close matches (typos)
            if let Some(closest) = find_closest_match(key, VALID_CONFIG_KEYS, Some(3)) {
                suggestions.push(format!("Did you mean '{closest}'?"));
            }

            suggestions.push("Run 'xf config --show' to see current configuration".to_string());

            let suggestion_refs: Vec<&str> = suggestions.iter().map(String::as_str).collect();
            anyhow::bail!(
                "{}",
                format_error(
                    &format!("Unknown config key: '{key}'"),
                    "",
                    &suggestion_refs,
                )
            );
        }
    }

    Ok(())
}

fn parse_optional_path(value: &str) -> Option<PathBuf> {
    if value.is_empty() {
        None
    } else {
        Some(PathBuf::from(value))
    }
}

fn parse_bool(value: &str, key: &str) -> Result<bool> {
    match value.to_lowercase().as_str() {
        "true" | "1" | "yes" | "y" | "on" => Ok(true),
        "false" | "0" | "no" | "n" | "off" => Ok(false),
        _ => anyhow::bail!("Invalid boolean value for {key}: {value}"),
    }
}

fn parse_usize(value: &str, key: &str) -> Result<usize> {
    value
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid integer value for {key}: {value}"))
}

fn parse_f32(value: &str, key: &str) -> Result<f32> {
    value
        .parse()
        .map_err(|_| anyhow::anyhow!("Invalid float value for {key}: {value}"))
}

fn parse_csv_list(value: &str) -> Vec<String> {
    if value.is_empty() {
        return Vec::new();
    }
    value
        .split(',')
        .map(|entry| entry.trim().to_string())
        .filter(|entry| !entry.is_empty())
        .collect()
}

fn cmd_update(output: &Output) {
    output.print("[cyan]Checking for updates...[/]");
    output.print("To update, run:\n  [bold]curl -fsSL https://raw.githubusercontent.com/Dicklesworthstone/xf/main/install.sh | bash[/]");
}

fn cmd_completions(_cli: &Cli, args: &cli::CompletionsArgs) {
    use xf::output::{Output, OutputFormat as OutFmt};

    // Create output - no_color is handled by Output's env detection
    let output = Output::new(OutFmt::Text);
    xf::completions::generate_completions(args.shell, &output);
}

// ============================================================================
// Doctor Command (xf-11.4.5)
// ============================================================================

use xf::doctor::{self, CheckCategory, CheckStatus, DoctorRenderer, DoctorSummary, HealthCheck};

/// Full doctor output for JSON format.
#[derive(Debug, Serialize)]
struct DoctorOutput {
    checks: Vec<HealthCheck>,
    summary: DoctorSummary,
    suggestions: Vec<String>,
    runtime_ms: u64,
}

#[allow(clippy::too_many_lines)]
fn cmd_doctor(cli: &Cli, args: &cli::DoctorArgs, output: &Output) -> Result<()> {
    let start = Instant::now();
    let mut all_checks: Vec<HealthCheck> = Vec::new();

    // Resolve paths
    let db_path = get_db_path(cli);
    let index_path = get_index_path(cli);

    // Get archive path from args or config
    let config = Config::load();
    let archive_path = args.archive.clone().or(config.paths.archive);

    info!("Running xf doctor...");

    // ========== Archive Checks ==========
    if let Some(ref archive) = archive_path {
        if archive.exists() {
            info!("Checking archive at: {}", archive.display());
            match doctor::validate_archive(archive) {
                Ok(checks) => all_checks.extend(checks),
                Err(e) => {
                    warn!("Archive validation failed: {}", e);
                    all_checks.push(HealthCheck {
                        category: CheckCategory::Archive,
                        name: "Archive Validation".into(),
                        status: CheckStatus::Error,
                        message: format!("Failed: {e}"),
                        suggestion: Some("Check archive path and permissions".into()),
                    });
                }
            }
        } else {
            all_checks.push(HealthCheck {
                category: CheckCategory::Archive,
                name: "Archive Path".into(),
                status: CheckStatus::Warning,
                message: format!("Path does not exist: {}", archive.display()),
                suggestion: Some("Provide a valid archive path with --archive".into()),
            });
        }
    } else {
        all_checks.push(HealthCheck {
            category: CheckCategory::Archive,
            name: "Archive Path".into(),
            status: CheckStatus::Warning,
            message: "No archive path configured".into(),
            suggestion: Some("Use --archive or set via 'xf config --archive <path>'".into()),
        });
    }

    // ========== Database Checks ==========
    if db_path.exists() {
        info!("Checking database at: {}", db_path.display());
        match Storage::open(&db_path) {
            Ok(storage) => {
                let db_checks = storage.database_health_checks();
                all_checks.extend(db_checks);

                // ========== Index Checks ==========
                if index_path.join("meta.json").exists() {
                    info!("Checking index at: {}", index_path.display());
                    match SearchEngine::open(&index_path) {
                        Ok(engine) => {
                            let index_checks = engine.index_health_checks(&storage);
                            all_checks.extend(index_checks);

                            // ========== Performance Checks ==========
                            info!("Running performance benchmarks...");
                            let perf_checks =
                                doctor::run_performance_benchmarks(&index_path, &engine, &storage);
                            all_checks.extend(perf_checks);

                            // ========== Vector Index Checks ==========
                            info!("Checking vector index...");
                            all_checks.extend(check_vector_index_health(&index_path, &storage));
                        }
                        Err(e) => {
                            warn!("Failed to open index: {}", e);
                            all_checks.push(HealthCheck {
                                category: CheckCategory::Index,
                                name: "Index Open".into(),
                                status: CheckStatus::Error,
                                message: format!("Failed to open: {e}"),
                                suggestion: Some(
                                    "Run 'xf index' to rebuild the search index".into(),
                                ),
                            });
                        }
                    }
                } else {
                    all_checks.push(HealthCheck {
                        category: CheckCategory::Index,
                        name: "Index Directory".into(),
                        status: CheckStatus::Warning,
                        message: format!("No index found at {}", index_path.display()),
                        suggestion: Some(
                            "Run 'xf index <archive_path>' to create the index".into(),
                        ),
                    });
                }
            }
            Err(e) => {
                warn!("Failed to open database: {}", e);
                all_checks.push(HealthCheck {
                    category: CheckCategory::Database,
                    name: "Database Open".into(),
                    status: CheckStatus::Error,
                    message: format!("Failed to open: {e}"),
                    suggestion: Some("Run 'xf index <archive_path>' to create the database".into()),
                });
            }
        }
    } else {
        all_checks.push(HealthCheck {
            category: CheckCategory::Database,
            name: "Database File".into(),
            status: CheckStatus::Warning,
            message: format!("No database found at {}", db_path.display()),
            suggestion: Some("Run 'xf index <archive_path>' to create the database".into()),
        });
    }

    // ========== Apply Fixes (--fix) ==========
    if args.fix {
        info!("Applying safe fixes...");
        if db_path.exists() {
            match Storage::open(&db_path) {
                Ok(mut storage) => {
                    let db_checks = storage.database_health_checks();
                    let fts_issue = db_checks
                        .iter()
                        .any(|check| check.name.starts_with("FTS") && !check.status.is_ok());
                    let dm_issue = db_checks
                        .iter()
                        .any(|check| check.name == "Orphaned DM messages" && !check.status.is_ok());

                    let mut applied_any = false;

                    if fts_issue {
                        match storage.rebuild_fts_tables() {
                            Ok(stats) => {
                                applied_any = true;
                                all_checks.push(HealthCheck {
                                    category: CheckCategory::Database,
                                    name: "Auto-fix (FTS rebuild)".into(),
                                    status: CheckStatus::Pass,
                                    message: format!(
                                        "fts_tweets={}; fts_likes={}; fts_dms={}; fts_grok={}",
                                        stats.tweets, stats.likes, stats.dms, stats.grok
                                    ),
                                    suggestion: None,
                                });
                            }
                            Err(err) => {
                                all_checks.push(HealthCheck {
                                    category: CheckCategory::Database,
                                    name: "Auto-fix (FTS rebuild)".into(),
                                    status: CheckStatus::Error,
                                    message: format!("Failed to rebuild FTS tables: {err}"),
                                    suggestion: Some(
                                        "Run 'xf index --force' to rebuild the database.".into(),
                                    ),
                                });
                            }
                        }
                    }

                    if dm_issue {
                        match storage.rebuild_dm_conversations() {
                            Ok(rebuilt) => {
                                applied_any = true;
                                all_checks.push(HealthCheck {
                                    category: CheckCategory::Database,
                                    name: "Auto-fix (DM conversations)".into(),
                                    status: CheckStatus::Pass,
                                    message: format!("Rebuilt {rebuilt} conversations"),
                                    suggestion: None,
                                });
                            }
                            Err(err) => {
                                all_checks.push(HealthCheck {
                                    category: CheckCategory::Database,
                                    name: "Auto-fix (DM conversations)".into(),
                                    status: CheckStatus::Error,
                                    message: format!("Failed to rebuild DM conversations: {err}"),
                                    suggestion: Some(
                                        "Run 'xf index --force' to rebuild DM conversations."
                                            .into(),
                                    ),
                                });
                            }
                        }
                    }

                    match storage.optimize() {
                        Ok(()) => {
                            all_checks.push(HealthCheck {
                                category: CheckCategory::Database,
                                name: "Auto-fix (SQLite optimize)".into(),
                                status: CheckStatus::Pass,
                                message: "PRAGMA optimize completed".into(),
                                suggestion: None,
                            });
                        }
                        Err(err) => {
                            all_checks.push(HealthCheck {
                                category: CheckCategory::Database,
                                name: "Auto-fix (SQLite optimize)".into(),
                                status: CheckStatus::Warning,
                                message: format!("Optimize failed: {err}"),
                                suggestion: Some(
                                    "Database may be locked by another process.".into(),
                                ),
                            });
                        }
                    }

                    if !applied_any && !fts_issue && !dm_issue {
                        all_checks.push(HealthCheck {
                            category: CheckCategory::Database,
                            name: "Auto-fix".into(),
                            status: CheckStatus::Pass,
                            message: "No structural issues detected".into(),
                            suggestion: None,
                        });
                    }
                }
                Err(err) => {
                    all_checks.push(HealthCheck {
                        category: CheckCategory::Database,
                        name: "Auto-fix".into(),
                        status: CheckStatus::Error,
                        message: format!("Failed to open database: {err}"),
                        suggestion: Some(
                            "Run 'xf index <archive_path>' to create the database".into(),
                        ),
                    });
                }
            }
        } else {
            all_checks.push(HealthCheck {
                category: CheckCategory::Database,
                name: "Auto-fix".into(),
                status: CheckStatus::Error,
                message: format!("No database found at {}", db_path.display()),
                suggestion: Some("Run 'xf index <archive_path>' to create the database".into()),
            });
        }
    }

    // ========== Sort Checks ==========
    // Sort by category (Archive -> Database -> Index -> Performance), then by name
    all_checks.sort_by(|a, b| {
        let cat_order = |c: &CheckCategory| match c {
            CheckCategory::Archive => 0,
            CheckCategory::Database => 1,
            CheckCategory::Index => 2,
            CheckCategory::Performance => 3,
        };
        cat_order(&a.category)
            .cmp(&cat_order(&b.category))
            .then_with(|| a.name.cmp(&b.name))
    });

    // ========== Build Summary ==========
    let summary = DoctorSummary::from_checks(&all_checks);
    let has_errors = summary.errors > 0;

    // Collect unique suggestions (sorted for deterministic output)
    let mut suggestions: Vec<String> = all_checks
        .iter()
        .filter_map(|c| c.suggestion.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();
    suggestions.sort();

    #[allow(clippy::cast_possible_truncation)]
    let runtime_ms = start.elapsed().as_millis() as u64; // Safe: health check won't run 584M years

    // ========== Output ==========
    match cli.format {
        OutputFormat::Json => {
            let doctor_output = DoctorOutput {
                checks: all_checks,
                summary,
                suggestions,
                runtime_ms,
            };
            output.print(&serde_json::to_string(&doctor_output)?);
        }
        OutputFormat::JsonPretty => {
            let doctor_output = DoctorOutput {
                checks: all_checks,
                summary,
                suggestions,
                runtime_ms,
            };
            output.print(&serde_json::to_string_pretty(&doctor_output)?);
        }
        OutputFormat::Toon => {
            let doctor_output = DoctorOutput {
                checks: all_checks,
                summary,
                suggestions,
                runtime_ms,
            };
            let json = serde_json::to_value(&doctor_output)?;
            output.print(&toon_rust::encode(json, None));
        }
        _ => {
            // Styled text output via DoctorRenderer
            use xf::theme::Theme;

            let theme = Theme::for_terminal();
            let renderer = DoctorRenderer::new(
                &all_checks,
                &summary,
                &suggestions,
                runtime_ms,
                output,
                &theme,
            );
            renderer.render();
        }
    }

    // Exit code based on severity
    if has_errors {
        std::process::exit(1);
    }
    Ok(())
}

/// Launch interactive REPL shell.
fn cmd_shell(cli: &Cli, args: &cli::ShellArgs, _output: &Output) -> Result<()> {
    let db_path = get_db_path(cli);
    let index_path = get_index_path(cli);

    // Check that DB exists
    if !db_path.exists() {
        anyhow::bail!(
            "{}",
            format_error(
                "No archive indexed yet",
                "The interactive shell requires an indexed archive.",
                &[
                    "1. Download your data from x.com/settings/download_your_data",
                    "2. Run: xf index ~/Downloads/twitter-archive",
                    "3. Then run: xf shell",
                ],
            )
        );
    }

    // Check that index exists
    if !index_path.exists() {
        anyhow::bail!(
            "{}",
            format_error(
                "Search index missing",
                &format!(
                    "Database exists but search index not found at '{}'.",
                    index_path.display()
                ),
                &["Run 'xf index <archive_path>' to rebuild the search index"],
            )
        );
    }

    info!(
        db = %db_path.display(),
        index = %index_path.display(),
        "Starting REPL shell"
    );

    let storage = Storage::open(&db_path)?;
    let search = SearchEngine::open(&index_path)?;

    let config = repl::ReplConfig {
        prompt: args.prompt.clone(),
        page_size: args.page_size,
        no_history: args.no_history,
        history_file: args.history_file.clone(),
    };

    repl::run(storage, search, config)
}

fn cmd_benchmark(_cli: &Cli, args: &cli::BenchmarkArgs, output: &Output) -> Result<()> {
    let corpus = BenchmarkCorpus::load(&args.corpus)
        .with_context(|| format!("Failed to load corpus: {}", args.corpus.display()))?;
    corpus
        .validate()
        .with_context(|| "Corpus validation failed")?;

    let registry = ModelRegistry::new();
    let mut config = EmbedderConfig::new(&args.model);
    config.dimensions = args.dimensions;
    config.show_progress = true;

    let bench_cfg = BenchmarkConfig {
        warmup_iters: args.warmup_iters,
        measure_iters: args.measure_iters,
        batch_size: args.batch_size,
        ..Default::default()
    };

    let model_name = args.model.clone();
    let result = run_embedding_benchmark(
        || {
            let embedder = registry.embedder(&config)?;
            Ok(embedder)
        },
        &corpus,
        &bench_cfg,
    )
    .with_context(|| format!("Benchmark failed for model {model_name}"))?;

    let output_dir = &args.output_dir;
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output dir: {}", output_dir.display()))?;

    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let base = output_dir.join(format!(
        "benchmark_{}_{}",
        result.metadata.model.replace('/', "_"),
        timestamp
    ));
    let json_path = base.with_extension("json");
    let md_path = base.with_extension("md");
    let csv_path = base.with_extension("csv");

    reporters::write_json(&json_path, &result)?;
    reporters::write_markdown(&md_path, &result)?;
    reporters::write_csv(&csv_path, &result)?;

    output.print(&format!(
        "[green]✓[/] Wrote benchmark reports:\n  {}\n  {}\n  {}",
        json_path.display(),
        md_path.display(),
        csv_path.display()
    ));

    Ok(())
}

// ============================================================================
// Robot Docs Command
// ============================================================================

fn cmd_robot_docs(args: &cli::RobotDocsArgs, output: &Output) -> Result<()> {
    if !xf::robot_docs::is_valid_topic(&args.topic) {
        output.eprint(&format!(
            "Unknown topic: '{}'. Available topics: {}",
            args.topic,
            xf::robot_docs::TOPICS.join(", ")
        ));
        std::process::exit(2);
    }

    let docs = xf::robot_docs::generate(&args.topic);
    output.print(&serde_json::to_string_pretty(&docs)?);
    Ok(())
}

// ============================================================================
// Models Command
// ============================================================================

#[allow(clippy::too_many_lines)]
fn cmd_models(args: &cli::ModelsArgs, output: &Output) -> Result<()> {
    use cli::{ModelType, ModelsCommand};

    let registry = ModelRegistry::new();

    match &args.command {
        ModelsCommand::List(list_args) => {
            let models = registry.list_models();

            // Filter by type if specified
            let filtered: Vec<_> = models
                .into_iter()
                .filter(|m| {
                    list_args
                        .model_type
                        .as_ref()
                        .is_none_or(|model_type| match model_type {
                            ModelType::Embedder => {
                                registry.embedder_names().contains(&m.name.as_str())
                            }
                            ModelType::Reranker => {
                                registry.reranker_names().contains(&m.name.as_str())
                            }
                        })
                })
                .filter(|m| !list_args.available || m.available)
                .collect();

            if output.format() == OutFmt::Json {
                output.print_json_pretty(&filtered)?;
            } else {
                // Text format
                output.print("[bold underline]Embedding Models[/]");
                output.print("");

                let embedders: Vec<_> = filtered
                    .iter()
                    .filter(|m| registry.embedder_names().contains(&m.name.as_str()))
                    .collect();

                if embedders.is_empty() {
                    output.print("  No embedders found.");
                } else {
                    for model in &embedders {
                        let status = if model.available {
                            "[green]✓[/]"
                        } else {
                            "[red]✗[/]"
                        };
                        let mrl = if model.supports_mrl { " [MRL]" } else { "" };
                        output.print(&format!(
                            "  {} {} ({}, {}d){mrl}",
                            status, model.name, model.backend, model.native_dims
                        ));
                    }
                }

                output.print("");
                output.print("[bold underline]Reranker Models[/]");
                output.print("");

                let rerankers: Vec<_> = filtered
                    .iter()
                    .filter(|m| registry.reranker_names().contains(&m.name.as_str()))
                    .collect();

                if rerankers.is_empty() {
                    output.print("  No rerankers found.");
                } else {
                    for model in &rerankers {
                        let status = if model.available {
                            "[green]✓[/]"
                        } else {
                            "[red]✗[/]"
                        };
                        output.print(&format!("  {} {} ({})", status, model.name, model.backend));
                    }
                }
                output.print("");
            }
        }
        ModelsCommand::Info(info_args) => {
            let models = registry.list_models();
            let model = models.iter().find(|m| m.name == info_args.name);

            if let Some(m) = model {
                if output.format() == OutFmt::Json {
                    output.print_json_pretty(m)?;
                } else {
                    output.print(&format!("[bold underline]Model: {}[/]", m.name));
                    output.print("");
                    output.print(&format!("  Backend:     {}", m.backend));
                    output.print(&format!("  Category:    {}", m.category));
                    output.print(&format!("  Dimensions:  {}", m.native_dims));
                    output.print(&format!(
                        "  MRL Support: {}",
                        if m.supports_mrl { "yes" } else { "no" }
                    ));
                    output.print(&format!(
                        "  Size:        {}",
                        m.size_mb
                            .map_or_else(|| "unknown".to_string(), |s| format!("{s:.1} MB"))
                    ));
                    output.print(&format!(
                        "  Available:   {}",
                        if m.available { "yes" } else { "no" }
                    ));
                    output.print(&format!(
                        "  Downloaded:  {}",
                        if m.downloaded { "yes" } else { "no" }
                    ));
                    output.print("");
                }
            } else {
                // Try to suggest a close match
                let all_names: Vec<_> = models.iter().map(|m| m.name.as_str()).collect();
                if let Some(suggestion) = find_closest_match(&info_args.name, &all_names, None) {
                    output.eprint(&format!(
                        "Unknown model: '{}'. Did you mean '{}'?",
                        info_args.name, suggestion
                    ));
                } else {
                    output.eprint(&format!(
                        "Unknown model: '{}'. Run 'xf models list' to see available models.",
                        info_args.name
                    ));
                }
                std::process::exit(2);
            }
        }
    }

    Ok(())
}

/// Handle daemon subcommand.
#[allow(clippy::too_many_lines)]
fn cmd_daemon(args: &cli::DaemonArgs, output: &Output) -> Result<()> {
    use cli::DaemonCommand;
    use xf::daemon::{DaemonClient, DaemonConfig, ModelDaemon};

    match &args.command {
        DaemonCommand::Start(start_args) => {
            // Load config
            let mut config = DaemonConfig::load(start_args.config.as_ref())?;

            // Apply CLI overrides
            if let Some(socket) = &start_args.socket {
                config.socket_path.clone_from(socket);
            }
            if start_args.idle_timeout > 0 {
                config.idle_timeout = Duration::from_secs(start_args.idle_timeout);
            } else {
                config.idle_timeout = Duration::MAX; // No timeout
            }
            config.max_models = start_args.max_models;

            if start_args.foreground {
                // Run daemon in foreground
                output.print(&format!(
                    "[blue]ℹ[/] Starting daemon on {} (foreground mode)",
                    config.socket_path.display()
                ));
                output.print(&format!(
                    "[blue]ℹ[/] Idle timeout: {}s, max models: {}",
                    config.idle_timeout.as_secs(),
                    config.max_models
                ));

                let daemon = ModelDaemon::new(config);
                let runtime = tokio::runtime::Runtime::new()?;
                runtime.block_on(daemon.run())?;
            } else {
                // Spawn daemon as background process
                output.print("[blue]ℹ[/] Spawning daemon in background...");

                let exe = std::env::current_exe()?;
                let mut cmd = std::process::Command::new(&exe);
                cmd.args(["daemon", "start", "--foreground"]);

                if let Some(socket) = &start_args.socket {
                    cmd.args(["--socket", &socket.to_string_lossy()]);
                }
                cmd.args(["--idle-timeout", &start_args.idle_timeout.to_string()]);
                cmd.args(["--max-models", &start_args.max_models.to_string()]);

                cmd.stdin(std::process::Stdio::null())
                    .stdout(std::process::Stdio::null())
                    .stderr(std::process::Stdio::null())
                    .spawn()?;

                output.print("[green]✓[/] Daemon spawned");
            }
        }

        DaemonCommand::Stop => {
            output.print("[blue]ℹ[/] Stopping daemon...");

            let mut client = DaemonClient::new();
            let runtime = tokio::runtime::Runtime::new()?;

            match runtime.block_on(client.shutdown()) {
                Ok(_) => output.print("[green]✓[/] Daemon stopped"),
                Err(e) => {
                    if e.to_string().contains("not running") {
                        output.eprint("[yellow]⚠[/] Daemon is not running");
                    } else {
                        return Err(e.context("Failed to stop daemon"));
                    }
                }
            }
        }

        DaemonCommand::Status => {
            let mut client = DaemonClient::new();
            let runtime = tokio::runtime::Runtime::new()?;

            match runtime.block_on(client.status()) {
                Ok(status) => {
                    if output.format() == OutFmt::Json {
                        output.print_json_pretty(&status)?;
                    } else {
                        output.print("[bold underline]Daemon Status[/]");
                        output.print("");
                        output.print(&format!("  Uptime:          {}s", status.uptime_secs));
                        output.print(&format!("  Memory (RSS):    {:.1} MB", status.rss_mb));
                        output.print(&format!("  Requests served: {}", status.requests_served));
                        output.print(&format!("  In-flight:       {}", status.in_flight));
                        output.print(&format!("  Queue length:    {}", status.queue_len));
                        output.print("");
                        output.print("[bold underline]Loaded Models[/]");
                        output.print("");

                        if status.models.is_empty() {
                            output.print("  No models loaded.");
                        } else {
                            for model in &status.models {
                                output.print(&format!(
                                    "  [cyan]{}[/] ({}) - {} requests",
                                    model.name, model.model_type, model.requests_served
                                ));
                            }
                        }
                    }
                }
                Err(e) => {
                    if e.to_string().contains("not running")
                        || e.to_string().contains("No such file")
                    {
                        output.eprint("[yellow]⚠[/] Daemon is not running");
                    } else {
                        return Err(e.context("Failed to get daemon status"));
                    }
                }
            }
        }
    }

    Ok(())
}

// ============================================================================
// Vector Index Health Checks
// ============================================================================

/// Run health checks on the vector index file.
///
/// Checks for:
/// - File existence
/// - File size and record count
/// - Consistency with database embedding count
#[allow(clippy::too_many_lines, clippy::cast_possible_wrap)]
fn check_vector_index_health(index_path: &Path, storage: &Storage) -> Vec<HealthCheck> {
    let mut checks = Vec::new();
    let vector_file = index_path.join(VECTOR_INDEX_FILENAME);

    if !vector_file.exists() {
        checks.push(HealthCheck {
            category: CheckCategory::Index,
            name: "Vector Index File".into(),
            status: CheckStatus::Warning,
            message: "Not found (semantic search will load from DB)".into(),
            suggestion: Some("Run 'xf index --force' to regenerate vector index".into()),
        });
        return checks;
    }

    // Check file size
    match std::fs::metadata(&vector_file) {
        Ok(meta) => {
            let size = meta.len();
            let size_str = format_bytes(size);

            if size < 32 {
                checks.push(HealthCheck {
                    category: CheckCategory::Index,
                    name: "Vector Index File".into(),
                    status: CheckStatus::Error,
                    message: format!("File too small ({size_str})"),
                    suggestion: Some("Run 'xf index --force' to regenerate".into()),
                });
                return checks;
            }

            // Try to read header to get record count
            match std::fs::read(&vector_file) {
                Ok(bytes) => {
                    // Parse record count from header (bytes 12-19)
                    if bytes.len() >= 20 {
                        let record_count = u64::from_le_bytes([
                            bytes[12], bytes[13], bytes[14], bytes[15], bytes[16], bytes[17],
                            bytes[18], bytes[19],
                        ]);

                        // Check consistency with DB
                        let db_count = storage.embedding_count().unwrap_or(0);
                        let diff = (record_count as i64 - db_count).abs();

                        let (status, message, suggestion) = if diff == 0 {
                            (
                                CheckStatus::Pass,
                                format!(
                                    "Found ({} records, {})",
                                    format_number_u64(record_count),
                                    size_str
                                ),
                                None,
                            )
                        } else if diff <= 10 || (db_count > 0 && diff * 100 / db_count <= 1) {
                            (
                                CheckStatus::Warning,
                                format!(
                                    "Found ({} records, DB has {}, diff: {})",
                                    format_number_u64(record_count),
                                    format_number(db_count),
                                    diff
                                ),
                                Some("Minor mismatch. Run 'xf index --force' to regenerate".into()),
                            )
                        } else {
                            (
                                CheckStatus::Warning,
                                format!(
                                    "Stale ({} records, DB has {})",
                                    format_number_u64(record_count),
                                    format_number(db_count)
                                ),
                                Some("Run 'xf index --force' to regenerate vector index".into()),
                            )
                        };

                        checks.push(HealthCheck {
                            category: CheckCategory::Index,
                            name: "Vector Index File".into(),
                            status,
                            message,
                            suggestion,
                        });
                    } else {
                        checks.push(HealthCheck {
                            category: CheckCategory::Index,
                            name: "Vector Index File".into(),
                            status: CheckStatus::Error,
                            message: "Header truncated".into(),
                            suggestion: Some("Run 'xf index --force' to regenerate".into()),
                        });
                    }
                }
                Err(e) => {
                    checks.push(HealthCheck {
                        category: CheckCategory::Index,
                        name: "Vector Index File".into(),
                        status: CheckStatus::Error,
                        message: format!("Cannot read: {e}"),
                        suggestion: Some("Check file permissions".into()),
                    });
                }
            }
        }
        Err(e) => {
            checks.push(HealthCheck {
                category: CheckCategory::Index,
                name: "Vector Index File".into(),
                status: CheckStatus::Error,
                message: format!("Cannot stat: {e}"),
                suggestion: Some("Check file permissions".into()),
            });
        }
    }

    checks
}

#[cfg(test)]
mod cache_tests {
    use super::CacheMeta;
    use super::Storage;
    use super::VectorIndexCache;
    use anyhow::{Context, Result};
    use std::collections::HashMap;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::SystemTime;
    use tempfile::tempdir;

    fn create_storage() -> Result<(tempfile::TempDir, PathBuf, PathBuf, Storage)> {
        let dir = tempdir().context("create temp dir")?;
        let db_path = dir.path().join("xf-test.db");
        let index_path = dir.path().join("index");
        std::fs::create_dir_all(&index_path).context("create index dir")?;
        let storage = Storage::open(&db_path)?;
        Ok((dir, db_path, index_path, storage))
    }

    fn build_meta(storage: &Storage, db_path: &Path) -> Result<CacheMeta> {
        let db_mtime = db_path
            .metadata()
            .and_then(|meta| meta.modified())
            .context("read database mtime")?;
        let embedding_count =
            usize::try_from(storage.embedding_count()?).context("convert embedding count")?;
        Ok(CacheMeta {
            db_mtime,
            embedding_count,
            type_counts: HashMap::new(),
        })
    }

    fn store_sample_embedding(storage: &Storage, doc_id: &str) -> Result<()> {
        storage.store_embedding(doc_id, "tweet", &[0.1, 0.2], None)
    }

    #[test]
    fn test_fresh_when_no_changes() -> Result<()> {
        let (_dir, db_path, _index_path, storage) = create_storage()?;
        store_sample_embedding(&storage, "doc-1")?;
        let meta = build_meta(&storage, &db_path)?;
        assert!(!meta.is_stale(&storage, &db_path)?);
        Ok(())
    }

    #[test]
    fn test_stale_after_embedding_added() -> Result<()> {
        let (_dir, db_path, _index_path, storage) = create_storage()?;
        store_sample_embedding(&storage, "doc-1")?;
        let meta = build_meta(&storage, &db_path)?;
        store_sample_embedding(&storage, "doc-2")?;
        assert!(meta.is_stale(&storage, &db_path)?);
        Ok(())
    }

    #[test]
    fn test_stale_after_embedding_deleted() -> Result<()> {
        let (_dir, db_path, _index_path, storage) = create_storage()?;
        store_sample_embedding(&storage, "doc-1")?;
        store_sample_embedding(&storage, "doc-2")?;
        let meta = build_meta(&storage, &db_path)?;
        storage.clear_embeddings()?;
        assert!(meta.is_stale(&storage, &db_path)?);
        Ok(())
    }

    #[test]
    fn test_stale_after_db_file_modified() -> Result<()> {
        let (_dir, db_path, _index_path, storage) = create_storage()?;
        store_sample_embedding(&storage, "doc-1")?;
        let mut meta = build_meta(&storage, &db_path)?;
        meta.db_mtime = SystemTime::UNIX_EPOCH;
        assert!(meta.is_stale(&storage, &db_path)?);
        Ok(())
    }

    #[test]
    fn test_stale_with_missing_db_file() -> Result<()> {
        let (_dir, _db_path, _index_path, storage) = create_storage()?;
        let missing_path = PathBuf::from("missing-xf-db.sqlite");
        let meta = CacheMeta {
            db_mtime: SystemTime::UNIX_EPOCH,
            embedding_count: 0,
            type_counts: HashMap::new(),
        };
        assert!(meta.is_stale(&storage, &missing_path).is_err());
        Ok(())
    }

    #[test]
    fn test_vector_index_initializes_once() -> Result<()> {
        let (_dir, db_path, index_path, storage) = create_storage()?;
        store_sample_embedding(&storage, "doc-1")?;
        let cache = VectorIndexCache::new();
        let (first, loaded_first) = cache.load(&storage, &db_path, &index_path)?;
        let (second, loaded_second) = cache.load(&storage, &db_path, &index_path)?;
        assert!(loaded_first);
        assert!(!loaded_second);
        assert!(std::ptr::eq(first, second));
        Ok(())
    }

    #[test]
    fn test_vector_index_handles_empty_embeddings() -> Result<()> {
        let (_dir, db_path, index_path, storage) = create_storage()?;
        let cache = VectorIndexCache::new();
        let (_index, loaded) = cache.load(&storage, &db_path, &index_path)?;
        assert!(loaded);
        let meta = cache.meta().context("cache meta missing")?;
        assert_eq!(meta.embedding_count, 0);
        Ok(())
    }

    #[test]
    fn test_concurrent_cache_access_after_init() -> Result<()> {
        let (_dir, db_path, index_path, storage) = create_storage()?;
        store_sample_embedding(&storage, "doc-1")?;
        let cache = Arc::new(VectorIndexCache::new());
        let _ = cache.load(&storage, &db_path, &index_path)?;

        let mut handles = Vec::new();
        for _ in 0..4 {
            let cache = Arc::clone(&cache);
            handles.push(std::thread::spawn(move || cache.index().is_some()));
        }

        for handle in handles {
            assert!(handle.join().unwrap_or(false));
        }

        Ok(())
    }

    // =========================================================================
    // Cache Performance Tests (xf-pkc)
    // =========================================================================

    /// Perf smoke check: verify second load is faster than first (cache hit).
    ///
    /// Marked as #[ignore] because timing is environment-dependent.
    /// Run manually with: `cargo test test_cache_hit_faster_than_cold -- --ignored`
    #[test]
    #[ignore = "Perf comparison is environment-dependent; run manually for validation"]
    fn test_cache_hit_faster_than_cold() -> Result<()> {
        use std::time::Instant;

        let (_dir, db_path, index_path, storage) = create_storage()?;

        // Add multiple embeddings to make the timing difference more noticeable
        for i in 0..100 {
            storage.store_embedding(&format!("doc-{i}"), "tweet", &[0.1, 0.2, 0.3], None)?;
        }

        let cache = VectorIndexCache::new();

        // First load (cold)
        let cold_start = Instant::now();
        let (_, loaded_first) = cache.load(&storage, &db_path, &index_path)?;
        let cold_duration = cold_start.elapsed();
        assert!(loaded_first, "First load should be fresh");

        // Second load (warm - should be instant)
        let warm_start = Instant::now();
        let (_, loaded_second) = cache.load(&storage, &db_path, &index_path)?;
        let warm_duration = warm_start.elapsed();
        assert!(!loaded_second, "Second load should be cached");

        eprintln!(
            "Cache perf: cold={cold_duration:?} warm={warm_duration:?} (speedup={:.1}x)",
            cold_duration.as_secs_f64() / warm_duration.as_secs_f64().max(1e-9)
        );

        // Warm should be faster - we just verify the cache is working, not a specific speedup.
        // In practice, speedup is typically 1000x+, but timing tests are environment-dependent.
        assert!(
            warm_duration < cold_duration,
            "Cached load ({warm_duration:?}) should be faster than cold load ({cold_duration:?})"
        );

        Ok(())
    }

    /// Test loading with a moderately large embedding set.
    ///
    /// This tests that the cache handles thousands of embeddings without
    /// memory issues. The size is bounded to avoid CI timeouts.
    #[test]
    fn test_large_embedding_set() -> Result<()> {
        let (_dir, db_path, index_path, storage) = create_storage()?;

        // Add 5000 embeddings - large enough to be meaningful, small enough for CI
        let embedding_count = 5000;
        for i in 0..embedding_count {
            let doc_type = match i % 4 {
                0 => "tweet",
                1 => "like",
                2 => "dm",
                _ => "grok",
            };
            storage.store_embedding(&format!("doc-{i}"), doc_type, &[0.1, 0.2, 0.3], None)?;
        }

        let cache = VectorIndexCache::new();
        let (index, loaded) = cache.load(&storage, &db_path, &index_path)?;

        assert!(loaded, "Should load fresh");
        assert_eq!(index.len(), embedding_count);

        // Verify type counts are tracked
        let meta = cache.meta().context("meta should be set")?;
        assert_eq!(meta.embedding_count, embedding_count);
        assert!(
            !meta.type_counts.is_empty(),
            "type_counts should be populated"
        );

        // Verify search still works
        let query = vec![0.1, 0.2, 0.3];
        let results = index.search_top_k(&query, 10, None);
        assert_eq!(results.len(), 10);

        Ok(())
    }

    /// Test that `loaded_now` flag is correctly set on first vs subsequent loads.
    #[test]
    fn test_loaded_now_flag_semantics() -> Result<()> {
        let (_dir, db_path, index_path, storage) = create_storage()?;
        store_sample_embedding(&storage, "doc-1")?;

        let cache = VectorIndexCache::new();

        // First load: loaded_now should be true
        let (_, loaded_now_1) = cache.load(&storage, &db_path, &index_path)?;
        assert!(loaded_now_1, "First load: loaded_now should be true");

        // Second load: loaded_now should be false
        let (_, loaded_now_2) = cache.load(&storage, &db_path, &index_path)?;
        assert!(!loaded_now_2, "Second load: loaded_now should be false");

        // Third load: still false
        let (_, loaded_now_3) = cache.load(&storage, &db_path, &index_path)?;
        assert!(!loaded_now_3, "Third load: loaded_now should be false");

        // Fourth load from a different thread: still false
        let cache = Arc::new(cache);
        let cache_clone = Arc::clone(&cache);
        let storage_clone = Storage::open(&db_path)?;
        let handle = std::thread::spawn(move || -> Result<bool> {
            let (_, loaded_now) = cache_clone.load(&storage_clone, &db_path, &index_path)?;
            Ok(loaded_now)
        });
        let loaded_now_thread = handle.join().expect("thread should complete")?;
        assert!(
            !loaded_now_thread,
            "Load from thread: loaded_now should be false"
        );

        Ok(())
    }
}
