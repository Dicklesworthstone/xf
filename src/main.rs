//! xf - Ultra-fast X data archive search CLI
//!
//! Main entry point for the xf command-line tool.

use anyhow::{Context, Result};
use chrono::{DateTime, Datelike, NaiveDate, TimeZone, Utc};
use clap::Parser;
use colored::control;
use rayon::ThreadPoolBuilder;
use serde::Serialize;
use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{self, BufReader, IsTerminal};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime};
use tracing::{info, warn};
use xf::logging::{LogConfig, init_logging};

use xf::benchmark::datasets::BenchmarkCorpus;
use xf::benchmark::reporters;
use xf::benchmark::runner::{BenchmarkConfig, BenchmarkMetadata, run_full_benchmark};
use xf::canonicalize::canonicalize_for_embedding;
use xf::cli;
use xf::config::Config;
use xf::date_parser;
use xf::embedder::Embedder;
use xf::hybrid::{self, SearchMode};
use xf::model_registry::{EmbedderConfig, ModelRegistry};
use xf::output::{Output, OutputFormat as OutFmt, Verbosity};
use xf::repl;
use xf::search;
use xf::stats_analytics::{self, ContentStats, EngagementStats, TemporalStats};
use xf::vector::{
    VECTOR_INDEX_FAST_FILENAME, VECTOR_INDEX_FILENAME, VECTOR_INDEX_QUALITY_FILENAME, VectorIndex,
    write_vector_index_named,
};
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
    /// Fast-tier index for two-tier search.
    fast_index: OnceLock<Option<VectorIndex>>,
    fast_meta: OnceLock<CacheMeta>,
    /// Quality-tier index for two-tier search.
    quality_index: OnceLock<Option<VectorIndex>>,
    quality_meta: OnceLock<CacheMeta>,
}

impl VectorIndexCache {
    const fn new() -> Self {
        Self {
            index: OnceLock::new(),
            meta: OnceLock::new(),
            init_lock: Mutex::new(()),
            fast_index: OnceLock::new(),
            fast_meta: OnceLock::new(),
            quality_index: OnceLock::new(),
            quality_meta: OnceLock::new(),
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
            Ok(true) => {
                let panel = xf::messages::MessagePanel::index_outdated();
                warn!(
                    "{}: {} (VectorIndex cache may be stale; restart xf to reload embeddings.)",
                    panel.title, panel.message
                );
            }
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

    /// Load the fast-tier vector index (returns None if file doesn't exist).
    fn load_fast(&self, index_path: &Path) -> Option<&VectorIndex> {
        self.fast_index
            .get_or_init(|| {
                match VectorIndex::load_named(index_path, xf::vector::VECTOR_INDEX_FAST_FILENAME) {
                    Ok(Some(idx)) => {
                        let meta = CacheMeta {
                            db_mtime: SystemTime::UNIX_EPOCH,
                            embedding_count: idx.len(),
                            type_counts: idx.type_counts(),
                        };
                        let _ = self.fast_meta.set(meta);
                        info!("Fast vector index loaded: {} embeddings", idx.len());
                        Some(idx)
                    }
                    Ok(None) => {
                        info!("Fast vector index not found");
                        None
                    }
                    Err(e) => {
                        warn!("Failed to load fast vector index: {e}");
                        None
                    }
                }
            })
            .as_ref()
    }

    /// Load the quality-tier vector index (returns None if file doesn't exist).
    fn load_quality(&self, index_path: &Path) -> Option<&VectorIndex> {
        self.quality_index
            .get_or_init(|| {
                match VectorIndex::load_named(index_path, xf::vector::VECTOR_INDEX_QUALITY_FILENAME)
                {
                    Ok(Some(idx)) => {
                        let meta = CacheMeta {
                            db_mtime: SystemTime::UNIX_EPOCH,
                            embedding_count: idx.len(),
                            type_counts: idx.type_counts(),
                        };
                        let _ = self.quality_meta.set(meta);
                        info!("Quality vector index loaded: {} embeddings", idx.len());
                        Some(idx)
                    }
                    Ok(None) => {
                        info!("Quality vector index not found");
                        None
                    }
                    Err(e) => {
                        warn!("Failed to load quality vector index: {e}");
                        None
                    }
                }
            })
            .as_ref()
    }
}

/// Global cached `VectorIndex` for semantic search.
/// Initialized on first search, reused for subsequent searches.
static VECTOR_INDEX_CACHE: VectorIndexCache = VectorIndexCache::new();

/// Resolve the default query embedder from what the index was actually built
/// with (GH #9), and make any degraded hash fallback LOUD.
///
/// Used by every search path when the user did not name a model explicitly
/// (no `--model`, no `XF_MODEL`, no config entry). The resolver embeds
/// queries in the same vector space as the stored document embeddings —
/// previously the search path probed xf-specific model directories that
/// `xf index --semantic` never populates and silently fell back to the hash
/// embedder (nDCG@10 = 0 on a real gold set).
fn resolve_default_query_embedder(storage: &Storage, context: &str) -> Box<dyn Embedder> {
    let resolved = xf::embedder_resolution::resolve_query_embedder(storage);
    if resolved.degraded {
        warn!("{context}: {}", resolved.detail);
        // tracing WARN is invisible in `--format json | jq` pipelines; write
        // an unmissable banner to stderr as well.
        eprintln!(
            "WARNING: {context} DEGRADED — {}. Falling back to the hash embedder, so semantic \
             ranking will be effectively random. Pass --model / set XF_MODEL to the model used \
             at index time, or re-run 'xf index --semantic'.",
            resolved.detail
        );
    } else if resolved.hash_index && context.contains("semantic") {
        // Correct embedder, but the index holds only hash embeddings: real
        // semantic quality is impossible; say so once, quietly but visibly.
        eprintln!(
            "note: {} — for true semantic search re-index with 'xf index --semantic'",
            resolved.detail
        );
    } else {
        info!("{context}: {}", resolved.detail);
    }
    resolved.embedder
}

#[derive(Debug, Clone)]
struct CacheMeta {
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
        OutputFormat::Json
        | OutputFormat::JsonPretty
        | OutputFormat::Toon
        | OutputFormat::Jsonl => OutFmt::Json,
        OutputFormat::Csv => OutFmt::Csv,
    }
}

fn main() -> Result<()> {
    let mut cli = Cli::parse();
    let config = Config::load();

    // Resolve format: CLI explicit value wins, then env vars, then config.output.format
    if cli.format == OutputFormat::Text {
        if let Some(env_fmt) = OutputFormat::from_env() {
            cli.format = env_fmt;
        } else if let Some(cfg_fmt) = OutputFormat::from_config_str(&config.output.format) {
            cli.format = cfg_fmt;
        }
    }

    // Handle --no-color flag, NO_COLOR env var, config.output.colors, and non-interactive output
    if should_disable_color(&cli) || !config.output.colors {
        control::set_override(false);
    }

    if config.output.quiet {
        cli.quiet = true;
    }

    // Setup logging
    // Default to xf-only logs so machine-readable output stays clean.
    let machine_output = matches!(
        cli.format,
        OutputFormat::Json
            | OutputFormat::JsonPretty
            | OutputFormat::Csv
            | OutputFormat::Toon
            | OutputFormat::Jsonl
    );
    // For machine-readable formats, force quiet unless user explicitly asked for verbose
    let effective_quiet = cli.quiet || (machine_output && cli.verbose == 0);
    let verbosity = Verbosity::from_flags(effective_quiet, cli.verbose);

    // Create output abstraction for styled terminal output
    let output = Output::new(cli_format_to_output(cli.format.clone())).with_verbosity(verbosity);

    // Initialize structured logging via xf::logging
    let log_config = LogConfig::from_verbosity(verbosity, output.is_stderr_styled());
    init_logging(&log_config);

    // Run the appropriate command
    match &cli.command {
        None => {
            print_quickstart(&output);
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
        Some(Commands::Models(args)) => cmd_models(&cli, args, &output),
        Some(Commands::Daemon(args)) => cmd_daemon(&cli, args, &output),
    }
}

/// Print a colorful quickstart guide when xf is run with no arguments.
#[allow(clippy::too_many_lines)]
fn print_quickstart(output: &Output) {
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

    // Helper to pad a line to fill the box - strips markup for width calculation
    let pad = |text: &str| -> String {
        let visible_len = console::measure_text_width(&Output::strip_markup(text));
        let padding = inner.saturating_sub(visible_len);
        format!("{v} {text}{}{v}", " ".repeat(padding.saturating_sub(1)))
    };

    // Header
    output.print(&format!("[bright_cyan]{}[/]", hline(tl, tr)));
    output.print(&pad(""));
    output.print(&pad(&format!(
        "[bold bright_cyan]xf[/]  [dim]v{version}[/]"
    )));
    output.print(&pad(
        "[italic]Ultra-fast CLI for searching your X data archive[/]",
    ));
    output.print(&pad(""));
    output.print(&format!(
        "[bright_cyan]{v}[dim]{}[/]{v}[/]",
        h.repeat(inner)
    ));

    // Quick Start section
    output.print(&pad(""));
    output.print(&pad("[bold yellow]1.[/]  Getting Started"));
    output.print(&pad(""));
    output.print(&pad(
        "   Download your archive from: [cyan]x.com/settings/download_your_data[/]",
    ));
    output.print(&pad(
        "   (X emails you when it's ready, usually 24-48 hours)",
    ));
    output.print(&pad(""));
    output.print(&pad("[bold yellow]2.[/]  Extract Your Archive"));
    output.print(&pad(""));
    output.print(&pad(
        "   [bright_green]unzip ~/Downloads/twitter-*.zip -d ~/my_twitter_data[/]",
    ));
    output.print(&pad(""));
    output.print(&pad("[bold yellow]3.[/]  Index Your Data"));
    output.print(&pad(""));
    output.print(&pad(
        "   [bright_green]xf index[/]  [dim]# Uses default path: /data/projects/my_twitter_data[/]",
    ));
    output.print(&pad(
        "   [bright_green]xf index ~/other/path[/]  [dim]# Or specify a custom path[/]",
    ));
    output.print(&pad("   (Takes ~5-30 seconds depending on archive size)"));
    output.print(&pad(""));
    output.print(&format!(
        "[bright_cyan]{v}[dim]{}[/]{v}[/]",
        h.repeat(inner)
    ));

    // Example searches section
    output.print(&pad(""));
    output.print(&pad("[bold bright_magenta]Example Searches[/]"));
    output.print(&pad(""));
    output.print(&pad(
        "   [bright_green]xf search \"machine learning\"[/]  [dim]# Find tweets about ML[/]",
    ));
    output.print(&pad(
        "   [bright_green]xf search \"dinner plans\" --types dm[/]  [dim]# Search your DMs[/]",
    ));
    output.print(&pad(
        "   [bright_green]xf search \"conference\" --types dm --context[/]  [dim]# DMs with full convo[/]",
    ));
    output.print(&pad(
        "   [bright_green]xf search \"interesting article\" --types like[/]  [dim]# Tweets you liked[/]",
    ));
    output.print(&pad(
        "   [bright_green]xf search \"bug fix\" --since \"last month\"[/]  [dim]# Recent tweets only[/]",
    ));
    output.print(&pad(
        "   [bright_green]xf search \"project update\" --format json[/]  [dim]# JSON output[/]",
    ));
    output.print(&pad(""));
    output.print(&format!(
        "[bright_cyan]{v}[dim]{}[/]{v}[/]",
        h.repeat(inner)
    ));

    // More commands section
    output.print(&pad(""));
    output.print(&pad("[bold bright_magenta]More Commands[/]"));
    output.print(&pad(""));
    output.print(&pad(
        "   [bright_green]xf stats[/]  [dim]# Archive overview (counts, date range)[/]",
    ));
    output.print(&pad(
        "   [bright_green]xf stats --detailed[/]  [dim]# Full analytics dashboard[/]",
    ));
    output.print(&pad(
        "   [bright_green]xf list tweets --limit 20[/]  [dim]# Browse recent tweets[/]",
    ));
    output.print(&pad(
        "   [bright_green]xf list conversations[/]  [dim]# See all DM threads[/]",
    ));
    output.print(&pad(
        "   [bright_green]xf tweet 1234567890 --thread[/]  [dim]# View a tweet thread[/]",
    ));
    output.print(&pad(
        "   [bright_green]xf export tweets --format csv -o tweets.csv[/]  [dim]# Export to CSV[/]",
    ));
    output.print(&pad(
        "   [bright_green]xf shell[/]  [dim]# Interactive REPL mode[/]",
    ));
    output.print(&pad(
        "   [bright_green]xf doctor[/]  [dim]# Check archive/index health[/]",
    ));
    output.print(&pad(""));
    output.print(&format!(
        "[bright_cyan]{v}[dim]{}[/]{v}[/]",
        h.repeat(inner)
    ));

    // Footer
    output.print(&pad(""));
    output.print(&pad(
        "Documentation: [cyan underline]https://github.com/Dicklesworthstone/xf[/]",
    ));
    output.print(&pad("Run [bright_green]xf --help[/] for all options"));
    output.print(&pad(""));
    output.print(&format!("[bright_cyan]{}[/]", hline(bl, br)));
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
    let extract_msg = format!(
        "Extracting {}...",
        args.zip_file
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
    );
    let mut pb = xf::progress::ProgressIndicator::new_spinner(&extract_msg, output);

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
            pb.set_message(&format!("Extracting... ({}/{} files)", i + 1, total_files));
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
            two_tier: false,
            no_embeddings: false,
            sync_embeddings: false, // Allow background embedding for import
            model: None,
            dimensions: None,
        };

        cmd_index(cli, &index_args, output)?;

        // Print welcome box with stats
        print_import_welcome(&output_dir, cli, output)?;
    }

    output.print("");
    Ok(())
}

/// Print a beautiful welcome box after successful import.
fn print_import_welcome(_archive_path: &PathBuf, cli: &Cli, output: &Output) -> Result<()> {
    let db_path = get_db_path(cli);
    let storage = Storage::open(&db_path)?;

    // Get stats
    let stats = storage.get_stats()?;

    let date_range = match (stats.first_tweet_date, stats.last_tweet_date) {
        (Some(first), Some(_last)) => format!("since {}", first.format("%b %Y")),
        _ => String::new(),
    };

    output.print("");

    // Box drawing
    let h = "─";
    let tl = "╭";
    let tr = "╮";
    let bl = "╰";
    let br = "╯";
    let v = "│";
    let width = 48;
    let inner = width - 2;

    let hline = format!("[bright_cyan]{}{}{}[/]", tl, h.repeat(inner), tr);
    let bline = format!("[bright_cyan]{}{}{}[/]", bl, h.repeat(inner), br);

    // Pad closure - strips markup for width calculation
    let pad = |text: &str| -> String {
        let visible_len = console::measure_text_width(&Output::strip_markup(text));
        // inner = width - 2 (for the two │ chars), minus 1 for the leading space
        let padding = inner.saturating_sub(visible_len).saturating_sub(1);
        format!(
            "[bright_cyan]{v}[/] {text}{}[bright_cyan]{v}[/]",
            " ".repeat(padding)
        )
    };

    output.print(&hline);
    output.print(&pad(""));
    output.print(&pad("  [bold bright_magenta]Welcome to your X archive![/]"));
    output.print(&pad(""));

    if stats.tweets_count > 0 {
        let tweets_str = format_number(stats.tweets_count);
        output.print(&pad(&format!(
            "  Tweets:    [bold]{tweets_str:>6}[/]  [dim]{date_range}[/]"
        )));
    }
    if stats.likes_count > 0 {
        let likes_str = format_number(stats.likes_count);
        output.print(&pad(&format!("  Likes:     [bold]{likes_str:>6}[/]")));
    }
    if stats.dms_count > 0 {
        let dms_str = format_number(stats.dms_count);
        let convos = stats.dm_conversations_count;
        output.print(&pad(&format!(
            "  DMs:       [bold]{dms_str:>6}[/]  [dim]in {convos} conversations[/]"
        )));
    }
    if stats.grok_messages_count > 0 {
        let grok_str = format_number(stats.grok_messages_count);
        output.print(&pad(&format!("  Grok:      [bold]{grok_str:>6}[/]")));
    }
    if stats.bookmarks_count > 0 {
        let bookmarks_str = format_number(stats.bookmarks_count);
        output.print(&pad(&format!("  Bookmarks: [bold]{bookmarks_str:>6}[/]")));
    }

    output.print(&pad(""));
    output.print(&pad(
        "  Try: [bright_green]xf search \"your first tweet\"[/]",
    ));
    output.print(&pad(""));
    output.print(&bline);

    Ok(())
}

/// Try to submit embedding job to daemon for background processing.
async fn try_background_embedding(
    db_path: &Path,
    index_path: &Path,
    two_tier: bool,
    fast_model: Option<String>,
    quality_model: Option<String>,
) -> Result<xf::daemon::EmbeddingJobSubmitResult> {
    use xf::daemon::DaemonClient;

    let mut client = DaemonClient::new();

    // Check if daemon is running (don't auto-spawn just for job submission)
    if !client.is_daemon_running() {
        anyhow::bail!("daemon not running");
    }

    client
        .submit_embedding_job(
            &db_path.to_string_lossy(),
            &index_path.to_string_lossy(),
            two_tier,
            fast_model,
            quality_model,
        )
        .await
}

/// Generate embeddings synchronously (blocking).
fn generate_embeddings_sync(
    quiet: bool,
    args: &cli::IndexArgs,
    config: &Config,
    storage: &Storage,
    index_path: &Path,
    output: &Output,
) -> Result<()> {
    let two_tier_config = &config.semantic.two_tier;
    let effective_model = config
        .semantic
        .effective_model(args.model.as_deref())
        .map(str::to_string);
    let effective_dims = config.semantic.effective_dimensions(args.dimensions);
    if let Some(dims) = effective_dims {
        validate_mrl_dims(dims)?;
    }
    let use_semantic = args.semantic
        || effective_model.as_deref().is_some_and(|m| {
            !matches!(
                m.to_ascii_lowercase().as_str(),
                "hash" | "fnv1a" | "fnv1a-384" | "hash-fnv1a-384"
            )
        });
    let embed_config = xf::EmbeddingConfig {
        show_progress: !quiet,
        use_semantic,
        model: effective_model,
        dimensions: effective_dims,
        two_tier: args.two_tier,
        fast_model: two_tier_config.fast_model.clone(),
        quality_model: two_tier_config.quality_model.clone(),
    };
    xf::generate_embeddings_with_config(storage, &embed_config)?;

    if args.two_tier {
        let fast_stats = write_vector_index_named(
            index_path,
            storage,
            VECTOR_INDEX_FAST_FILENAME,
            Some("fast"),
        )?;
        if !quiet && fast_stats.record_count > 0 {
            output.print(&format!(
                "  [green]✓[/] Fast vector index written ({} records, {})",
                format_number_usize(fast_stats.record_count),
                format_bytes(fast_stats.file_size)
            ));
        }

        let quality_stats = write_vector_index_named(
            index_path,
            storage,
            VECTOR_INDEX_QUALITY_FILENAME,
            Some("quality"),
        )?;
        if !quiet && quality_stats.record_count > 0 {
            output.print(&format!(
                "  [green]✓[/] Quality vector index written ({} records, {})",
                format_number_usize(quality_stats.record_count),
                format_bytes(quality_stats.file_size)
            ));
        }

        // Write the primary vector.idx from the quality tier (or fast tier if quality is empty)
        // so single-tier semantic search works without mixing 256d fast + 384d quality vectors.
        let primary_model_id = if quality_stats.record_count > 0 {
            "quality"
        } else {
            "fast"
        };
        let vector_stats = write_vector_index_named(
            index_path,
            storage,
            xf::vector::VECTOR_INDEX_FILENAME,
            Some(primary_model_id),
        )?;
        if storage.get_embedding_model("default")?.is_none() {
            if let Some(model_name) = storage.get_embedding_model(primary_model_id)? {
                storage.set_embedding_model("default", &model_name)?;
            }
        }
        if !quiet && vector_stats.record_count > 0 {
            output.print(&format!(
                "  [green]✓[/] Vector index written ({} records, {})",
                format_number_usize(vector_stats.record_count),
                format_bytes(vector_stats.file_size)
            ));
        }
    } else {
        // Write vector index file for fast semantic search (scoped to "default" model_id)
        let vector_stats = write_vector_index_named(
            index_path,
            storage,
            xf::vector::VECTOR_INDEX_FILENAME,
            Some("default"),
        )?;
        if !quiet && vector_stats.record_count > 0 {
            output.print(&format!(
                "  [green]✓[/] Vector index written ({} records, {})",
                format_number_usize(vector_stats.record_count),
                format_bytes(vector_stats.file_size)
            ));
        }
    }

    Ok(())
}

#[allow(clippy::too_many_lines)]
fn cmd_index(cli: &Cli, args: &cli::IndexArgs, output: &Output) -> Result<()> {
    // Use provided path or fall back to config/default
    let config = Config::load();
    let default_path = config
        .paths
        .archive
        .clone()
        .unwrap_or_else(|| PathBuf::from(xf::DEFAULT_ARCHIVE_PATH));
    let raw_archive_path = args.archive_path.as_ref().unwrap_or(&default_path);

    // Validate archive path
    if !raw_archive_path.exists() {
        if output.is_stderr_styled() {
            xf::messages::MessagePanel::archive_not_found(raw_archive_path)
                .render_to_stderr(output);
        }
        anyhow::bail!(
            "{}",
            format_error(
                "Archive not found",
                &format!("The path '{}' does not exist.", raw_archive_path.display()),
                &[
                    "Check the path for typos",
                    "Download your data from x.com/settings/download_your_data",
                ],
            )
        );
    }

    // If a .zip file was passed directly to `xf index`, extract it automatically
    let extracted_archive_dir: Option<PathBuf> = if raw_archive_path.is_file()
        && raw_archive_path
            .extension()
            .is_some_and(|ext| ext.eq_ignore_ascii_case("zip"))
    {
        let candidate = raw_archive_path.with_extension("");
        let out_dir = if candidate.is_file() {
            raw_archive_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("extracted_x_archive")
        } else {
            candidate
        };
        if !out_dir.join("data").exists() || args.force {
            output.print(&format!(
                "  [cyan]⟳[/] Extracting ZIP archive '{}' to '{}'...",
                raw_archive_path.display(),
                out_dir.display()
            ));
            let file = File::open(raw_archive_path)
                .with_context(|| format!("Failed to open '{}'", raw_archive_path.display()))?;
            let reader = BufReader::new(file);
            let mut archive = zip::ZipArchive::new(reader).with_context(|| {
                format!("Failed to read zip file '{}'", raw_archive_path.display())
            })?;
            fs::create_dir_all(&out_dir)?;
            for i in 0..archive.len() {
                let mut entry = archive.by_index(i)?;
                let Some(rel_path) = entry.enclosed_name() else {
                    continue;
                };
                let outpath = out_dir.join(rel_path);
                if entry.is_dir() {
                    fs::create_dir_all(&outpath)?;
                } else {
                    if let Some(parent) = outpath.parent() {
                        if !parent.exists() {
                            fs::create_dir_all(parent)?;
                        }
                    }
                    let mut outfile = File::create(&outpath)?;
                    io::copy(&mut entry, &mut outfile)?;
                }
            }
        }
        Some(out_dir)
    } else {
        None
    };
    let archive_path = extracted_archive_dir.as_ref().unwrap_or(raw_archive_path);

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
                    "Ensure you're pointing to the extracted archive root (or a valid .zip archive)",
                    "The archive should contain a 'data' folder with .js files",
                ],
            )
        );
    }

    let threads = if args.jobs > 0 {
        args.jobs
    } else if !config.indexing.parallel {
        1
    } else {
        config.indexing.threads
    };
    if threads > 0 {
        let _ = ThreadPoolBuilder::new().num_threads(threads).build_global();
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

    let index_timer = xf::perf::Timer::start(xf::perf::INDEX_FULL);

    output.print("[bold cyan]Indexing X data archive...[/]");
    output.print(&format!("  Archive: {}", archive_path.display()));
    output.print(&format!("  Database: {}", db_path.display()));
    output.print(&format!("  Index: {}", index_path.display()));
    output.print("");

    // Parse archive
    let parser = ArchiveParser::new(archive_path);

    // Open storage and search engine
    let storage_timer = xf::perf::Timer::start(xf::perf::STORAGE_OPEN);
    let mut storage = Storage::open(&db_path)?;
    let _ = storage_timer.stop();
    let search_engine = SearchEngine::open(&index_path)?;
    let buffer_bytes = config
        .indexing
        .buffer_size_mb
        .max(15)
        .saturating_mul(1_000_000);
    let mut writer = search_engine.writer(buffer_bytes)?;

    // Parse and store manifest, account, and profile metadata
    let manifest = parser.parse_manifest()?;
    storage.store_archive_info(&manifest)?;
    if let Ok(Some(account)) = parser.parse_account() {
        let _ = storage.store_account(&account);
    }
    if let Ok(Some(profile)) = parser.parse_profile() {
        let _ = storage.store_profile(&profile);
    }
    output.print(&format!(
        "  [green]✓[/] Archive for @{} ({})",
        manifest.username,
        manifest.display_name.as_deref().unwrap_or("Unknown")
    ));

    // Determine what to index
    let mut data_types = args.only.as_ref().map_or_else(
        || {
            args.skip.as_ref().map_or_else(
                || {
                    if config.indexing.skip_types.is_empty() {
                        DataType::all()
                    } else {
                        use clap::ValueEnum;
                        let skip_set: Vec<DataType> = config
                            .indexing
                            .skip_types
                            .iter()
                            .filter_map(|s| DataType::from_str(s.trim(), true).ok())
                            .collect();
                        DataType::all()
                            .into_iter()
                            .filter(|t| !skip_set.contains(t))
                            .collect()
                    }
                },
                |skip| {
                    DataType::all()
                        .into_iter()
                        .filter(|t| !skip.contains(t))
                        .collect()
                },
            )
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

    // Progress bar (hidden when stdout is non-tty or quiet)
    let mut pb =
        xf::progress::ProgressIndicator::new_bar(data_types.len() as u64, "indexing", output);

    // Tweet text lookup for cross-referencing (used by bookmarks)
    let mut tweet_texts: std::collections::HashMap<String, String> =
        std::collections::HashMap::new();

    // Index each data type
    for data_type in &data_types {
        let item_start = Instant::now();
        match data_type {
            DataType::Tweet => {
                pb.set_message("tweets");
                let tweets = parser.parse_tweets()?;
                // Build tweet text lookup for bookmark cross-referencing
                for tweet in &tweets {
                    tweet_texts.insert(tweet.id.clone(), tweet.full_text.clone());
                }
                storage.store_tweets(&tweets)?;
                search_engine.clear_doc_type(&writer, search::DocType::Tweet);
                search_engine.index_tweets(&mut writer, &tweets)?;
                let elapsed = format_duration(item_start.elapsed());
                pb.print_line(
                    &format!(
                        "  [green]✓[/] [bold]{}[/] tweets [dim]({elapsed})[/]",
                        format_number_usize(tweets.len())
                    ),
                    output,
                );
            }
            DataType::Like => {
                pb.set_message("likes");
                let likes = parser.parse_likes()?;
                for like in &likes {
                    if let Some(ref text) = like.full_text {
                        if !text.is_empty() {
                            tweet_texts
                                .entry(like.tweet_id.clone())
                                .or_insert_with(|| text.clone());
                        }
                    }
                }
                storage.store_likes(&likes)?;
                search_engine.clear_doc_type(&writer, search::DocType::Like);
                search_engine.index_likes(&mut writer, &likes)?;
                let elapsed = format_duration(item_start.elapsed());
                pb.print_line(
                    &format!(
                        "  [green]✓[/] [bold]{}[/] likes [dim]({elapsed})[/]",
                        format_number_usize(likes.len())
                    ),
                    output,
                );
            }
            DataType::Dm => {
                pb.set_message("DMs");
                let convos = parser.parse_direct_messages()?;
                let msg_count: usize = convos.iter().map(|c| c.messages.len()).sum();
                storage.store_dm_conversations(&convos)?;
                search_engine.clear_doc_type(&writer, search::DocType::DirectMessage);
                search_engine.index_dms(&mut writer, &convos)?;
                let elapsed = format_duration(item_start.elapsed());
                pb.print_line(
                    &format!(
                        "  [green]✓[/] [bold]{}[/] DM conversations ([bold]{}[/] messages) [dim]({elapsed})[/]",
                        format_number_usize(convos.len()),
                        format_number_usize(msg_count)
                    ),
                    output,
                );
            }
            DataType::Grok => {
                pb.set_message("Grok");
                let messages = parser.parse_grok_messages()?;
                storage.store_grok_messages(&messages)?;
                search_engine.clear_doc_type(&writer, search::DocType::GrokMessage);
                search_engine.index_grok_messages(&mut writer, &messages)?;
                let elapsed = format_duration(item_start.elapsed());
                pb.print_line(
                    &format!(
                        "  [green]✓[/] [bold]{}[/] Grok messages [dim]({elapsed})[/]",
                        format_number_usize(messages.len())
                    ),
                    output,
                );
            }
            DataType::Bookmark => {
                pb.set_message("bookmarks");
                // Supplement in-memory tweet_texts with any previously stored tweets/likes in SQLite
                if let Ok(stored_texts) = storage.load_tweet_text_lookup() {
                    for (id, text) in stored_texts {
                        tweet_texts.entry(id).or_insert(text);
                    }
                }
                let bookmarks = parser.parse_bookmarks(&tweet_texts)?;
                storage.store_bookmarks(&bookmarks)?;
                search_engine.clear_doc_type(&writer, search::DocType::Bookmark);
                search_engine.index_bookmarks(&mut writer, &bookmarks)?;
                let with_text = bookmarks.iter().filter(|b| b.full_text.is_some()).count();
                let elapsed = format_duration(item_start.elapsed());
                pb.print_line(
                    &format!(
                        "  [green]✓[/] [bold]{}[/] bookmarks ({} with text) [dim]({elapsed})[/]",
                        format_number_usize(bookmarks.len()),
                        format_number_usize(with_text)
                    ),
                    output,
                );
            }
            DataType::Follower => {
                pb.set_message("followers");
                let followers = parser.parse_followers()?;
                storage.store_followers(&followers)?;
                let elapsed = format_duration(item_start.elapsed());
                pb.print_line(
                    &format!(
                        "  [green]✓[/] [bold]{}[/] followers [dim]({elapsed})[/]",
                        format_number_usize(followers.len())
                    ),
                    output,
                );
            }
            DataType::Following => {
                pb.set_message("following");
                let following = parser.parse_following()?;
                storage.store_following(&following)?;
                let elapsed = format_duration(item_start.elapsed());
                pb.print_line(
                    &format!(
                        "  [green]✓[/] [bold]{}[/] following [dim]({elapsed})[/]",
                        format_number_usize(following.len())
                    ),
                    output,
                );
            }
            DataType::Block => {
                pb.set_message("blocks");
                let blocks = parser.parse_blocks()?;
                storage.store_blocks(&blocks)?;
                let elapsed = format_duration(item_start.elapsed());
                pb.print_line(
                    &format!(
                        "  [green]✓[/] [bold]{}[/] blocks [dim]({elapsed})[/]",
                        format_number_usize(blocks.len())
                    ),
                    output,
                );
            }
            DataType::Mute => {
                pb.set_message("mutes");
                let mutes = parser.parse_mutes()?;
                storage.store_mutes(&mutes)?;
                let elapsed = format_duration(item_start.elapsed());
                pb.print_line(
                    &format!(
                        "  [green]✓[/] [bold]{}[/] mutes [dim]({elapsed})[/]",
                        format_number_usize(mutes.len())
                    ),
                    output,
                );
            }
            DataType::All => {
                // Already handled by DataType::all()
            }
        }
        pb.inc(1);
    }

    pb.finish_and_clear();

    // Commit search index
    let commit_timer = xf::perf::Timer::start(xf::perf::INDEX_COMMIT);
    writer.commit()?;
    let _ = commit_timer.stop();
    search_engine.reload()?;

    // Handle embeddings based on flags
    let effective_dims = config.semantic.effective_dimensions(args.dimensions);
    if args.no_embeddings {
        // Skip embedding generation entirely
        if !cli.quiet {
            output.print("  [yellow]⚠[/] Skipping embeddings (--no-embeddings)");
            output.print("    Lexical search will work immediately.");
            output
                .print("    Run `xf index` again without --no-embeddings to add semantic search.");
        }
    } else if args.sync_embeddings || effective_dims.is_some() {
        // Force synchronous embedding (or when custom MRL dimensions are specified)
        generate_embeddings_sync(cli.quiet, args, &config, &storage, &index_path, output)?;
    } else {
        // Try background embedding via daemon, fall back to sync
        let quality_model_for_job = if args.two_tier {
            config.semantic.two_tier.quality_model.clone()
        } else {
            config
                .semantic
                .effective_model(args.model.as_deref())
                .map(str::to_string)
                .or_else(|| args.semantic.then(|| "all-MiniLM-L6-v2".to_string()))
        };
        let runtime = tokio::runtime::Runtime::new()?;
        let background_succeeded = runtime.block_on(async {
            try_background_embedding(
                &db_path,
                &index_path,
                args.two_tier,
                config.semantic.two_tier.fast_model.clone(),
                quality_model_for_job,
            )
            .await
        });

        match background_succeeded {
            Ok(result) => {
                if !cli.quiet {
                    output.print(&format!(
                        "  [cyan]⟳[/] Embeddings generating in background (job #{})",
                        result.job_id
                    ));
                    #[allow(clippy::cast_sign_loss)]
                    let est_docs = result.estimated_docs as u64;
                    output.print(&format!(
                        "    {} documents queued",
                        format_number_u64(est_docs)
                    ));
                    output.print("    Run `xf daemon status` to check progress");
                    output.print("");
                    output.print("    [dim]Lexical search works immediately.[/]");
                    output.print(
                        "    [dim]Hybrid/semantic search improves as embeddings complete.[/]",
                    );
                }
            }
            Err(e) => {
                // Daemon not available, fall back to sync
                if !cli.quiet {
                    info!("Background embedding unavailable ({}), using sync", e);
                }
                generate_embeddings_sync(cli.quiet, args, &config, &storage, &index_path, output)?;
            }
        }
    }

    let total_elapsed = format_duration(index_timer.stop());

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
    let mut config = Config::load();

    // Determine effective search mode and blend factor from CLI flags
    let effective_mode = if args.fast_only || args.quality_only || args.blend_factor.is_some() {
        // These flags imply two-tier mode
        SearchMode::TwoTier
    } else {
        args.mode
    };

    // Override blend factor based on CLI flags
    if args.fast_only {
        config.semantic.two_tier.blend_factor = 0.0;
    } else if args.quality_only {
        config.semantic.two_tier.blend_factor = 1.0;
    } else if let Some(factor) = args.blend_factor {
        config.semantic.two_tier.blend_factor = factor;
    }

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

    let storage = Storage::open(&db_path)?;
    let has_tantivy_meta = index_path.join("meta.json").exists();
    let sql_doc_count =
        usize::try_from(storage.indexable_document_count().unwrap_or(0)).unwrap_or(0);

    if !has_tantivy_meta && sql_doc_count == 0 {
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
            OutputFormat::Text
                | OutputFormat::Json
                | OutputFormat::JsonPretty
                | OutputFormat::Toon
                | OutputFormat::Jsonl
        ) {
            anyhow::bail!(
                "--context only supports text, json, json-pretty, jsonl, or toon output."
            );
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
            OutputFormat::Json
                | OutputFormat::JsonPretty
                | OutputFormat::Toon
                | OutputFormat::Jsonl
        ) {
            anyhow::bail!(
                "--fields is only supported with --format json, json-pretty, jsonl, or toon."
            );
        }
        validate_output_fields(fields)?;
    }

    let search_engine = if has_tantivy_meta {
        match SearchEngine::open(&index_path) {
            Ok(engine) if engine.doc_count() > 0 || sql_doc_count == 0 => Some(engine),
            Ok(_) => {
                info!(
                    "Tantivy index is empty while SQLite has {sql_doc_count} docs; using SQLite FTS5 fallback"
                );
                None
            }
            Err(err) => {
                if sql_doc_count > 0 {
                    warn!("Failed to open Tantivy index ({err}); falling back to SQLite FTS5");
                    None
                } else {
                    return Err(err);
                }
            }
        }
    } else {
        info!(
            "Tantivy index missing at '{}'; falling back to SQLite FTS5 ({sql_doc_count} docs)",
            index_path.display()
        );
        None
    };

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
                        SearchType::Bookmark => Some(search::DocType::Bookmark),
                        SearchType::All => None,
                    })
                    .collect(),
            )
        })
    };

    // Load vector index for semantic/hybrid/two-tier search (cached per process)
    let vector_index = if matches!(
        effective_mode,
        SearchMode::Semantic | SearchMode::Hybrid | SearchMode::TwoTier
    ) {
        let index = load_vector_index_cached(&storage, &db_path, &index_path)?;
        if matches!(effective_mode, SearchMode::Semantic)
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

    let cli_limit_passed = std::env::args_os().any(|a| {
        let s = a.to_string_lossy();
        s == "--limit"
            || s == "-n"
            || s.starts_with("--limit=")
            || (s.starts_with("-n") && s.len() > 2 && s[2..].chars().all(|c| c.is_ascii_digit()))
    });
    let effective_limit = if cli_limit_passed {
        args.limit
    } else {
        config.search.default_limit
    };

    let explicit_reranker = config.semantic.effective_reranker(args.reranker.as_deref());
    let reranker_name = explicit_reranker.unwrap_or(xf::model_registry::RERANKER_FLASHRANK_NANO);
    let rerank_enabled = (config.semantic.is_rerank_enabled(args.rerank)
        || args.reranker.is_some())
        && !matches!(reranker_name.to_ascii_lowercase().as_str(), "none" | "off");
    let effective_rerank_top = if args.rerank_top != 100 {
        args.rerank_top
    } else if config.semantic.rerank_top > 0 {
        config.semantic.rerank_top
    } else {
        100
    };

    let limit_target = effective_limit.saturating_add(args.offset);
    let retrieval_target = if rerank_enabled {
        limit_target.max(effective_rerank_top)
    } else {
        limit_target
    };
    let needs_post_filter =
        since.is_some() || until.is_some() || args.replies_only || args.no_replies;
    let fast_field_date_order = if matches!(effective_mode, SearchMode::Lexical)
        && !needs_post_filter
        && !rerank_enabled
        && search_engine.is_some()
    {
        match args.sort {
            SortOrder::Date => Some(tantivy::Order::Asc),
            SortOrder::DateDesc => Some(tantivy::Order::Desc),
            _ => None,
        }
    } else {
        None
    };
    let needs_full_sort =
        !matches!(args.sort, SortOrder::Relevance) && fast_field_date_order.is_none();
    let total_indexed_docs = search_engine.as_ref().map_or(sql_doc_count, |se| {
        usize::try_from(se.doc_count()).unwrap_or(usize::MAX)
    });
    let max_docs = if needs_post_filter || needs_full_sort {
        total_indexed_docs.max(retrieval_target)
    } else {
        retrieval_target
    };

    let run_lexical = |fetch_limit: usize| -> Result<Vec<xf::SearchResult>> {
        search_engine.as_ref().map_or_else(
            || storage.search_fts(&args.query, doc_types.as_deref(), fetch_limit),
            |engine| {
                engine.search_ordered(
                    &args.query,
                    doc_types.as_deref(),
                    fetch_limit,
                    fast_field_date_order,
                )
            },
        )
    };

    let embed_query = |embedder: &dyn Embedder,
                       text: &str,
                       dims: Option<usize>|
     -> xf::embedder::EmbedderResult<Vec<f32>> {
        let try_daemon =
            !args.no_daemon && (args.daemon || (config.semantic.daemon && embedder.is_semantic()));
        if try_daemon {
            let mut client_cfg = xf::daemon::ClientConfig::default();
            if !args.daemon {
                client_cfg = client_cfg.without_auto_spawn();
            }
            let mut client = xf::daemon::DaemonClient::with_config(client_cfg);
            if args.daemon || client.is_daemon_running() {
                if let Ok(rt) = tokio::runtime::Runtime::new() {
                    if let Ok(mut vecs) =
                        rt.block_on(client.embed(&[text], Some(embedder.model_name()), dims))
                    {
                        if !vecs.is_empty() {
                            return Ok(vecs.swap_remove(0));
                        }
                    }
                }
            }
        }
        embedder.embed(text)
    };

    let hydrate_lookups =
        |lookups: &[search::DocLookup<'_>]| -> Result<Vec<Option<xf::SearchResult>>> {
            if lookups.is_empty() {
                return Ok(Vec::new());
            }
            if let Some(ref engine) = search_engine {
                engine.get_by_ids(lookups)
            } else {
                let all_fts =
                    storage.search_fts("", doc_types.as_deref(), total_indexed_docs.max(1000))?;
                let mut by_id_type: std::collections::HashMap<(String, String), xf::SearchResult> =
                    std::collections::HashMap::with_capacity(all_fts.len());
                for item in all_fts {
                    by_id_type.insert((item.id.clone(), item.result_type.to_string()), item);
                }
                Ok(lookups
                    .iter()
                    .map(|l| {
                        l.doc_type.map_or_else(
                            || {
                                by_id_type
                                    .iter()
                                    .find(|((id, _), _)| id == l.id)
                                    .map(|(_, v)| v.clone())
                            },
                            |dt| by_id_type.get(&(l.id.to_string(), dt.to_string())).cloned(),
                        )
                    })
                    .collect())
            }
        };

    // Time the search operation against performance budgets
    let search_budget = if search_engine.is_none() {
        xf::perf::STORAGE_FTS
    } else if doc_types.is_some() || needs_post_filter {
        xf::perf::SEARCH_FILTERED
    } else if args.query.contains('*') {
        xf::perf::SEARCH_WILDCARD
    } else if args.query.contains('"') {
        xf::perf::SEARCH_PHRASE
    } else if args.query.contains(" AND ")
        || args.query.contains(" OR ")
        || args.query.contains(" NOT ")
        || !matches!(effective_mode, SearchMode::Lexical)
    {
        xf::perf::SEARCH_COMPLEX
    } else {
        xf::perf::SEARCH_SIMPLE
    };
    let search_timer = xf::perf::Timer::start(search_budget);

    // Perform search based on mode
    let mut results = match effective_mode {
        SearchMode::Lexical => {
            // Original lexical-only search (with SQLite FTS5 fallback when Tantivy is absent)
            let mut fetch_limit = retrieval_target.min(max_docs);
            loop {
                let mut batch = run_lexical(fetch_limit)?;
                if needs_post_filter {
                    apply_search_filters(
                        &mut batch,
                        since,
                        until,
                        args.replies_only,
                        args.no_replies,
                    );
                }

                if (batch.len() >= retrieval_target && !needs_full_sort) || fetch_limit >= max_docs
                {
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
            // Model resolution: CLI flag > config file/env > FastEmbedder > hash fallback
            let effective_model = config.semantic.effective_model(args.model.as_deref());
            let effective_dims = config.semantic.effective_dimensions(args.dimensions);
            let embedder_box: Box<dyn Embedder> = if let Some(model) = effective_model {
                if let Some(dims) = effective_dims {
                    validate_mrl_dims(dims)?;
                }
                let registry = ModelRegistry::new();
                let mut cfg = EmbedderConfig::new(model);
                cfg.dimensions = effective_dims;
                cfg.show_progress = false;
                let boxed = registry.embedder(&cfg)?;
                info!("Using configured embedder: {}", model);
                boxed
            } else {
                // GH #9: resolve from what the index was built with; loud on
                // any degraded hash fallback.
                resolve_default_query_embedder(&storage, "semantic search")
            };
            let embedder: &dyn Embedder = embedder_box.as_ref();
            let canonical_query = canonicalize_for_embedding(&args.query);

            if canonical_query.is_empty() {
                Vec::new()
            } else {
                let query_embedding = embed_query(embedder, &canonical_query, effective_dims)?;

                // Convert doc_types to string slices for vector search
                let type_strs: Option<Vec<&str>> = doc_types
                    .as_ref()
                    .map(|types| types.iter().map(|t| t.as_str()).collect());

                let max_vec_docs = vector_index.len().max(retrieval_target);
                let mut fetch_limit = if needs_full_sort {
                    max_vec_docs
                } else {
                    retrieval_target
                        .saturating_mul(hybrid::CANDIDATE_MULTIPLIER)
                        .min(max_vec_docs)
                };

                loop {
                    let semantic_hits = vector_index.search_top_k(
                        &query_embedding,
                        fetch_limit,
                        type_strs.as_deref(),
                    );

                    let lookups: Vec<_> = semantic_hits
                        .iter()
                        .map(|hit| search::DocLookup::with_type(&hit.doc_id, hit.doc_type))
                        .collect();
                    let fetched = hydrate_lookups(&lookups)?;

                    // Look up full results by doc_id + type
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

                    if (results.len() >= retrieval_target && !needs_full_sort)
                        || fetch_limit >= max_vec_docs
                    {
                        break results;
                    }

                    let next = fetch_limit
                        .saturating_mul(2)
                        .max(fetch_limit.saturating_add(1));
                    fetch_limit = next.min(max_vec_docs);
                }
            }
        }

        SearchMode::Hybrid => {
            // Hybrid search using RRF fusion
            // Model resolution: CLI flag > config file/env > FastEmbedder > hash fallback
            let effective_model = config.semantic.effective_model(args.model.as_deref());
            let effective_dims = config.semantic.effective_dimensions(args.dimensions);
            let embedder_box: Box<dyn Embedder> = if let Some(model) = effective_model {
                if let Some(dims) = effective_dims {
                    validate_mrl_dims(dims)?;
                }
                let registry = ModelRegistry::new();
                let mut cfg = EmbedderConfig::new(model);
                cfg.dimensions = effective_dims;
                cfg.show_progress = false;
                let boxed = registry.embedder(&cfg)?;
                info!("Using configured embedder: {}", model);
                boxed
            } else {
                // GH #9: resolve from what the index was built with; loud on
                // any degraded hash fallback.
                resolve_default_query_embedder(&storage, "hybrid search (semantic half)")
            };
            let embedder: &dyn Embedder = embedder_box.as_ref();
            let canonical_query = canonicalize_for_embedding(&args.query);
            let base_candidates = hybrid::candidate_count(effective_limit, args.offset)
                .max(100)
                .max(retrieval_target);
            let candidate_count = if needs_post_filter || needs_full_sort {
                max_docs.max(base_candidates)
            } else {
                base_candidates
            };
            let fuse_limit = if needs_post_filter || needs_full_sort || rerank_enabled {
                candidate_count
            } else {
                retrieval_target
            };

            // Get lexical results
            let lexical_results = run_lexical(candidate_count)?;

            // Get semantic results (if embeddings exist and query canonicalizes)
            let semantic_results = vector_index.map_or_else(Vec::new, |vec_idx| {
                if canonical_query.is_empty() {
                    Vec::new()
                } else if let Ok(query_embedding) =
                    embed_query(embedder, &canonical_query, effective_dims)
                {
                    let type_strs: Option<Vec<&str>> = doc_types
                        .as_ref()
                        .map(|types| types.iter().map(|t| t.as_str()).collect());
                    vec_idx.search_top_k(&query_embedding, candidate_count, type_strs.as_deref())
                } else {
                    Vec::new()
                }
            });

            // Fuse results using RRF before filtering/sorting so filters never starve top-K
            let fused = hybrid::rrf_fuse(&lexical_results, &semantic_results, fuse_limit, 0);

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

            let fetched = hydrate_lookups(&lookups)?;

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

        SearchMode::TwoTier => {
            // Two-tier progressive search: fast results refined by quality model
            let two_tier_cfg = &config.semantic.two_tier;
            let effective_dims = config.semantic.effective_dimensions(args.dimensions);

            // Phase 1: Fast semantic search + lexical search → RRF fuse
            // Try fast index first, fall back to default index
            let fast_index = VECTOR_INDEX_CACHE.load_fast(&index_path);
            let use_index = fast_index.or(vector_index);

            // Create fast embedder (hash-based by default)
            let fast_model = two_tier_cfg
                .fast_model
                .as_deref()
                .or_else(|| config.semantic.effective_model(args.model.as_deref()));
            let fast_embedder_box: Box<dyn Embedder> = if let Some(model) = fast_model {
                let registry = ModelRegistry::new();
                let mut cfg = EmbedderConfig::new(model);
                cfg.dimensions = effective_dims;
                cfg.show_progress = false;
                registry.embedder(&cfg)?
            } else {
                // GH #9: resolve from what the index was built with; loud on
                // any degraded hash fallback.
                resolve_default_query_embedder(&storage, "two-tier search (fast tier)")
            };
            let fast_embedder: &dyn Embedder = fast_embedder_box.as_ref();

            let canonical_query = canonicalize_for_embedding(&args.query);
            let base_candidates = hybrid::candidate_count(effective_limit, args.offset)
                .max(100)
                .max(retrieval_target);
            let candidate_count = if needs_post_filter || needs_full_sort {
                max_docs.max(base_candidates)
            } else {
                base_candidates
            };
            let fuse_limit = if needs_post_filter || needs_full_sort || rerank_enabled {
                candidate_count
            } else {
                retrieval_target
            };

            // Lexical results
            let lexical_results = run_lexical(candidate_count)?;

            // Fast semantic results
            let fast_semantic = use_index.map_or_else(Vec::new, |vec_idx| {
                if canonical_query.is_empty() {
                    Vec::new()
                } else if let Ok(query_embedding) =
                    embed_query(fast_embedder, &canonical_query, effective_dims)
                {
                    let type_strs: Option<Vec<&str>> = doc_types
                        .as_ref()
                        .map(|types| types.iter().map(|t| t.as_str()).collect());
                    vec_idx.search_top_k(&query_embedding, candidate_count, type_strs.as_deref())
                } else {
                    Vec::new()
                }
            });

            // Phase 2: Quality refinement (if quality index exists)
            let quality_index = VECTOR_INDEX_CACHE.load_quality(&index_path);

            let final_semantic_results = if let Some(q_index) = quality_index {
                // Create quality embedder
                let quality_model = two_tier_cfg
                    .quality_model
                    .as_deref()
                    .or_else(|| config.semantic.effective_model(args.model.as_deref()));
                let quality_embedder_box: Box<dyn Embedder> = if let Some(model) = quality_model {
                    let registry = ModelRegistry::new();
                    let mut cfg = EmbedderConfig::new(model);
                    cfg.dimensions = effective_dims;
                    cfg.show_progress = false;
                    registry.embedder(&cfg)?
                } else {
                    // GH #9: resolve from what the index was built with; loud
                    // on any degraded hash fallback.
                    resolve_default_query_embedder(&storage, "two-tier search (quality tier)")
                };
                let quality_embedder: &dyn Embedder = quality_embedder_box.as_ref();

                // Quality semantic search
                let type_strs: Option<Vec<&str>> = doc_types
                    .as_ref()
                    .map(|types| types.iter().map(|t| t.as_str()).collect());

                if canonical_query.is_empty() {
                    fast_semantic
                } else if let Ok(query_embedding) =
                    embed_query(quality_embedder, &canonical_query, effective_dims)
                {
                    let quality_hits = q_index.search_top_k(
                        &query_embedding,
                        candidate_count,
                        type_strs.as_deref(),
                    );

                    // Normalize and blend fast + quality
                    let mut fast_normalized = fast_semantic;
                    let mut quality_normalized = quality_hits;
                    hybrid::min_max_normalize(&mut fast_normalized);
                    hybrid::min_max_normalize(&mut quality_normalized);

                    hybrid::blend_two_tier(
                        &fast_normalized,
                        &quality_normalized,
                        two_tier_cfg.blend_factor,
                    )
                } else {
                    warn!("Quality embedding failed; returning Phase 1 results only");
                    fast_semantic
                }
            } else {
                info!("Quality index not found; returning Phase 1 results only");
                fast_semantic
            };

            // Re-fuse with lexical using the refined semantic results
            let final_fused =
                hybrid::rrf_fuse(&lexical_results, &final_semantic_results, fuse_limit, 0);

            // Convert fused hits back to SearchResults (same pattern as Hybrid)
            let mut lookups = Vec::new();
            let mut lookup_indices = Vec::new();
            for (idx, hit) in final_fused.iter().enumerate() {
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

            let fetched = hydrate_lookups(&lookups)?;

            let mut fetched_by_index = vec![None; final_fused.len()];
            for (idx, result) in lookup_indices.into_iter().zip(fetched) {
                fetched_by_index[idx] = result;
            }

            let mut results = Vec::new();
            for (idx, hit) in final_fused.iter().enumerate() {
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

    if config.search.min_score > 0.0 {
        results.retain(|r| r.score >= config.search.min_score);
    }

    if rerank_enabled && !results.is_empty() {
        let rerank_count = results.len().min(effective_rerank_top);
        let try_daemon = !args.no_daemon && (args.daemon || config.semantic.daemon);
        let mut reranked_via_daemon = false;
        if try_daemon && rerank_count > 0 {
            let mut client_cfg = xf::daemon::ClientConfig::default();
            if !args.daemon {
                client_cfg = client_cfg.without_auto_spawn();
            }
            let mut client = xf::daemon::DaemonClient::with_config(client_cfg);
            if args.daemon || client.is_daemon_running() {
                let doc_texts: Vec<&str> = results[..rerank_count]
                    .iter()
                    .map(|r| r.text.as_str())
                    .collect();
                if let Ok(rt) = tokio::runtime::Runtime::new() {
                    if let Ok(scores) =
                        rt.block_on(client.rerank(&args.query, &doc_texts, Some(reranker_name)))
                    {
                        if xf::rerank_step::RerankStep::apply_scores(
                            &mut results,
                            rerank_count,
                            &scores,
                        )
                        .is_ok()
                        {
                            reranked_via_daemon = true;
                        }
                    }
                }
            }
        }
        if !reranked_via_daemon {
            let registry = ModelRegistry::new();
            let step = match xf::rerank_step::RerankStep::new(&registry, reranker_name) {
                Ok(Some(step)) => step,
                Ok(None) => xf::rerank_step::RerankStep::with_reranker(Box::new(
                    xf::reranker::HashReranker::new(),
                )),
                Err(xf::reranker::RerankerError::Unavailable(reason)) => {
                    warn!(
                        "Reranker '{}' unavailable ({}); falling back to deterministic hash-reranker",
                        reranker_name, reason
                    );
                    xf::rerank_step::RerankStep::with_reranker(Box::new(
                        xf::reranker::HashReranker::new(),
                    ))
                }
                Err(err) => return Err(err.into()),
            }
            .with_top_k(effective_rerank_top)
            .with_min_candidates(1);
            step.rerank(&args.query, &mut results)?;
        }
    }

    apply_search_sort(&mut results, &args.sort);

    // Apply offset
    let mut results: Vec<_> = results.into_iter().skip(args.offset).collect();
    if effective_limit == 0 {
        results.clear();
    } else if results.len() > effective_limit {
        results.truncate(effective_limit);
    }

    let search_elapsed = search_timer.stop();

    if results.is_empty() {
        match cli.format {
            OutputFormat::Json | OutputFormat::JsonPretty => {
                output.print_data("[]");
                return Ok(());
            }
            OutputFormat::Jsonl | OutputFormat::Compact => {
                return Ok(());
            }
            OutputFormat::Csv => {
                output.print_data("type,id,created_at,score,text");
                return Ok(());
            }
            OutputFormat::Toon => {
                let json = serde_json::Value::Array(Vec::new());
                output.print_data(&toon_rust::encode(json, None));
                return Ok(());
            }
            OutputFormat::Text => {}
        }

        if output.is_styled() {
            let mut panel = xf::messages::MessagePanel::no_results(&args.query);
            if args.since.is_some() || args.until.is_some() {
                panel = panel.with_suggestion("Removing date filters (--since / --until)");
            }
            if let Some(types) = &args.types {
                if types.len() == 1 {
                    panel = panel.with_suggestion(
                        "Searching other data types: xf search \"...\" --types tweet,dm,like",
                    );
                }
            }
            panel.render(output);
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
                output.print_data(&serde_json::to_string(&filtered)?);
            } else {
                output.print_data(&serde_json::to_string(&results)?);
            }
        }
        OutputFormat::JsonPretty => {
            if let Some(fields) = &args.fields {
                let filtered = filter_results_fields(&results, fields)?;
                output.print_data(&serde_json::to_string_pretty(&filtered)?);
            } else {
                output.print_data(&serde_json::to_string_pretty(&results)?);
            }
        }
        OutputFormat::Jsonl => {
            if let Some(fields) = &args.fields {
                for value in filter_results_fields(&results, fields)? {
                    output.print_data(&serde_json::to_string(&value)?);
                }
            } else {
                for r in &results {
                    output.print_data(&serde_json::to_string(r)?);
                }
            }
        }
        OutputFormat::Csv => {
            output.print_data("type,id,created_at,score,text");
            for r in &results {
                // Escape quotes and replace newlines/carriage returns for valid CSV
                let text_escaped = csv_escape_text(&r.text);
                output.print_data(&format!(
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
            let query_terms = if config.search.highlight {
                xf::search_results::extract_terms(&args.query)
            } else {
                Vec::new()
            };

            output.print(&format!(
                "Found [bold]{}[/] results for \"[bold]{}[/]\" in [dim]{timing_str}[/]\n",
                format_number_usize(results.len()),
                args.query
            ));

            for (i, r) in results.iter().enumerate() {
                print_result(i + 1, r, &query_terms, output);
            }
        }
        OutputFormat::Toon => {
            let json = if let Some(fields) = &args.fields {
                let filtered = filter_results_fields(&results, fields)?;
                serde_json::to_value(&filtered)?
            } else {
                serde_json::to_value(&results)?
            };
            output.print_data(&toon_rust::encode(json, None));
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
            output.print_data(&serde_json::to_string(contexts)?);
        }
        OutputFormat::Jsonl => {
            for context in contexts {
                output.print_data(&serde_json::to_string(context)?);
            }
        }
        OutputFormat::JsonPretty => {
            output.print_data(&serde_json::to_string_pretty(contexts)?);
        }
        OutputFormat::Text => {
            print_dm_context_text(contexts, highlight_enabled, output);
        }
        OutputFormat::Toon => {
            let json = serde_json::to_value(contexts)?;
            output.print_data(&toon_rust::encode(json, None));
        }
        _ => {
            anyhow::bail!(
                "--context only supports text, json, json-pretty, jsonl, or toon output."
            );
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

fn print_result(num: usize, result: &SearchResult, query_terms: &[String], output: &Output) {
    if output.is_styled() {
        output.print(&format!("[bold]{num}.[/]"));
        let doc_type_str = match result.result_type {
            SearchResultType::Tweet => "tweet",
            SearchResultType::Like => "like",
            SearchResultType::DirectMessage => "dm",
            SearchResultType::GrokMessage => "grok",
            SearchResultType::Bookmark => "bookmark",
        };
        let mut card = xf::search_results::SearchResultCard::new()
            .doc_id(&result.id)
            .doc_type(doc_type_str)
            .content(&result.text)
            .score(result.score)
            .max_content_length(560)
            .highlight_terms(query_terms.to_vec());
        if result.created_at.timestamp() > 0 {
            card = card.timestamp(result.created_at);
        }
        card.render(output);
        return;
    }

    let type_badge = match result.result_type {
        SearchResultType::Tweet => "[on blue]TWEET[/]",
        SearchResultType::Like => "[on magenta]LIKE[/]",
        SearchResultType::DirectMessage => "[on green]DM[/]",
        SearchResultType::GrokMessage => "[on yellow]GROK[/]",
        SearchResultType::Bookmark => "[on cyan]BOOKMARK[/]",
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
    [
        "in_reply_to",
        "in_reply_to_status_id",
        "in_reply_to_user_id",
    ]
    .iter()
    .any(|key| {
        result
            .metadata
            .get(*key)
            .and_then(|v| v.as_str())
            .is_some_and(|s| !s.is_empty())
    })
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
            if matches!(
                r.result_type,
                SearchResultType::Like | SearchResultType::Bookmark
            ) {
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

    let storage_timer = xf::perf::Timer::start(xf::perf::STORAGE_OPEN);
    let storage = Storage::open(&db_path)?;
    let _ = storage_timer.stop();
    let stats_timer = xf::perf::Timer::start(xf::perf::STORAGE_STATS);
    let stats = storage.get_stats()?;
    let _ = stats_timer.stop();

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
        OutputFormat::Json | OutputFormat::JsonPretty | OutputFormat::Jsonl => {
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
                output.print_data(&json);
            } else {
                let json = if matches!(cli.format, OutputFormat::JsonPretty) {
                    serde_json::to_string_pretty(&stats)?
                } else {
                    serde_json::to_string(&stats)?
                };
                output.print_data(&json);
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
                output.print_data(&toon_rust::encode(json, None));
            } else {
                let json = serde_json::to_value(&stats)?;
                output.print_data(&toon_rust::encode(json, None));
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

            if output.is_styled() {
                let index_path = get_index_path(cli);
                let config = Config::load();
                let archive_path = config.paths.archive;
                let storage_info = xf::stats_dashboard::StorageInfo {
                    archive_bytes: archive_path.as_deref().map_or(0, path_size_bytes),
                    index_bytes: path_size_bytes(&index_path),
                    database_bytes: path_size_bytes(&db_path),
                    archive_path,
                };
                let counts = xf::storage::AllCounts::from(&stats);
                xf::stats_dashboard::StatsDashboard::new(&counts)
                    .with_storage(storage_info)
                    .render(output);
            } else {
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
                if stats.bookmarks_count > 0 {
                    output.print(&format!(
                        "  [dim]{:<20}[/] [bold]{:>10}[/]",
                        "Bookmarks:",
                        format_number(stats.bookmarks_count)
                    ));
                }
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
                    if output.is_styled() {
                        let pairs: Vec<(String, usize)> = items
                            .iter()
                            .map(|item| (item.value.clone(), item.count))
                            .collect();
                        xf::list_tables::render_hashtag_list(&pairs, output);
                    } else {
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
            }

            if let Some(items) = top_mentions {
                if !items.is_empty() {
                    if output.is_styled() {
                        let pairs: Vec<(String, usize)> = items
                            .iter()
                            .map(|item| (item.value.clone(), item.count))
                            .collect();
                        xf::list_tables::render_mention_list(&pairs, output);
                    } else {
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

fn path_size_bytes(path: &Path) -> u64 {
    let Ok(meta) = fs::metadata(path) else {
        return 0;
    };
    if meta.is_file() {
        return meta.len();
    }
    let Ok(entries) = fs::read_dir(path) else {
        return 0;
    };
    entries
        .flatten()
        .map(|entry| path_size_bytes(&entry.path()))
        .sum()
}

fn cmd_tweet(cli: &Cli, args: &cli::TweetArgs, output: &Output) -> Result<()> {
    let db_path = get_db_path(cli);
    let storage = Storage::open(&db_path)?;

    if args.thread {
        return cmd_tweet_thread(cli, &storage, args, output);
    }

    let lookup_timer = xf::perf::Timer::start(xf::perf::STORAGE_LOOKUP);
    let tweet = storage.get_tweet(&args.id)?;
    let _ = lookup_timer.stop();

    match tweet {
        Some(t) => match cli.format {
            OutputFormat::Json | OutputFormat::JsonPretty | OutputFormat::Jsonl => {
                let json = if matches!(cli.format, OutputFormat::JsonPretty) {
                    serde_json::to_string_pretty(&t)?
                } else {
                    serde_json::to_string(&t)?
                };
                output.print_data(&json);
            }
            OutputFormat::Toon => {
                let json = serde_json::to_value(&t)?;
                output.print_data(&toon_rust::encode(json, None));
            }
            _ => {
                if output.is_styled() {
                    xf::tweet_card::TweetCard::new(&t)
                        .show_engagement(args.engagement)
                        .render(output);
                } else {
                    output.print(&format!("[dim]{}[/]", "─".repeat(CONTENT_DIVIDER_WIDTH)));
                    output.print(&t.full_text);
                }
                output.print(&format!("[dim]{}[/]", "─".repeat(CONTENT_DIVIDER_WIDTH)));
                output.print(&format!(
                    "  [dim]ID: {}  Date: {}[/]",
                    t.id,
                    format_relative_date(t.created_at)
                ));
                if args.engagement && !output.is_styled() {
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

fn print_list_machine_output<T: Serialize>(
    cli: &Cli,
    items: &[T],
    output: &Output,
) -> Result<bool> {
    match cli.format {
        OutputFormat::Json => {
            output.print_data(&serde_json::to_string(items)?);
            Ok(true)
        }
        OutputFormat::JsonPretty => {
            output.print_data(&serde_json::to_string_pretty(items)?);
            Ok(true)
        }
        OutputFormat::Jsonl => {
            for item in items {
                output.print_data(&serde_json::to_string(item)?);
            }
            Ok(true)
        }
        OutputFormat::Toon => {
            let json = serde_json::to_value(items)?;
            output.print_data(&toon_rust::encode(json, None));
            Ok(true)
        }
        OutputFormat::Csv => {
            let csv = format_export(items, &ExportFormat::Csv)?;
            output.print_data(csv.trim_end_matches('\n'));
            Ok(true)
        }
        OutputFormat::Text | OutputFormat::Compact => Ok(false),
    }
}

#[allow(clippy::too_many_lines)]
fn cmd_list(cli: &Cli, args: &cli::ListArgs, output: &Output) -> Result<()> {
    let db_path = get_db_path(cli);

    if matches!(args.what, ListTarget::Files) {
        let config = Config::load();
        let Some(archive_path) = config.paths.archive else {
            if matches!(cli.format, OutputFormat::Csv) {
                output.print_data("file");
                return Ok(());
            }
            if print_list_machine_output::<String>(cli, &[], output)? {
                return Ok(());
            }
            output.print("[yellow]No archive path configured. Use 'xf config --archive <path>' or set XF_ARCHIVE.[/]");
            return Ok(());
        };

        if !archive_path.exists() {
            if matches!(cli.format, OutputFormat::Csv) {
                output.print_data("file");
                return Ok(());
            }
            if print_list_machine_output::<String>(cli, &[], output)? {
                return Ok(());
            }
            output.print(&format!(
                "[red]Archive path not found: {}[/]",
                archive_path.display()
            ));
            return Ok(());
        }

        let parser = ArchiveParser::new(&archive_path);
        let files = parser.list_data_files()?;
        if matches!(cli.format, OutputFormat::Csv) {
            output.print_data("file");
            for file in &files {
                output.print_data(&csv_escape(&serde_json::Value::String(file.clone())));
            }
            return Ok(());
        }
        if print_list_machine_output(cli, &files, output)? {
            return Ok(());
        }

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
            if print_list_machine_output(cli, &tweets, output)? {
                return Ok(());
            }
            if output.is_styled() {
                use xf::list_tables::{ListTable, TableColumn};
                ListTable::new("Tweets", tweets)
                    .add_column(TableColumn::new("Date", |t: &xf::Tweet| {
                        format_relative_date(t.created_at)
                    }))
                    .add_column(TableColumn::new("ID", |t: &xf::Tweet| {
                        format_short_id(&t.id)
                    }))
                    .add_column(TableColumn::new("Text", |t: &xf::Tweet| {
                        truncate_text(&t.full_text, 80)
                    }))
                    .render(output);
                return Ok(());
            }
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
            if print_list_machine_output(cli, &likes, output)? {
                return Ok(());
            }
            if output.is_styled() {
                use xf::list_tables::{ListTable, TableColumn};
                ListTable::new("Likes", likes)
                    .add_column(TableColumn::new("Tweet ID", |l: &xf::Like| {
                        format_short_id(&l.tweet_id)
                    }))
                    .add_column(TableColumn::new("Text", |l: &xf::Like| {
                        l.full_text
                            .as_ref()
                            .map_or_else(|| "[No text]".to_string(), |t| truncate_text(t, 80))
                    }))
                    .render(output);
                return Ok(());
            }
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
            if print_list_machine_output(cli, &dms, output)? {
                return Ok(());
            }
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
            if print_list_machine_output(cli, &conversations, output)? {
                return Ok(());
            }
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
            if print_list_machine_output(cli, &followers, output)? {
                return Ok(());
            }
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
            if print_list_machine_output(cli, &following, output)? {
                return Ok(());
            }
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
            if print_list_machine_output(cli, &blocks, output)? {
                return Ok(());
            }
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
            if print_list_machine_output(cli, &mutes, output)? {
                return Ok(());
            }
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
        ListTarget::Bookmarks => {
            let bookmarks = storage.get_all_bookmarks(limit)?;
            if print_list_machine_output(cli, &bookmarks, output)? {
                return Ok(());
            }
            output.print(&format!(
                "[dim]Showing[/] [bold]{}[/] bookmarks:\n",
                format_number_usize(bookmarks.len())
            ));
            for bm in &bookmarks {
                let text = bm
                    .full_text
                    .as_ref()
                    .map_or_else(|| "[No text]".to_string(), |t| truncate_text(t, 80));
                output.print(&format!(
                    "[dim]{}[/] {}",
                    format_short_id(&bm.tweet_id),
                    text
                ));
            }
        }
        ListTarget::Grok => {
            let grok = storage.get_all_grok_messages(limit)?;
            if print_list_machine_output(cli, &grok, output)? {
                return Ok(());
            }
            output.print(&format!(
                "[dim]Showing[/] [bold]{}[/] Grok messages:\n",
                format_number_usize(grok.len())
            ));
            for msg in &grok {
                let date = format_relative_date(msg.created_at);
                let text = truncate_text(&msg.message, 60);
                output.print(&format!(
                    "[dim]{} {} ({})[/] {}",
                    date,
                    format_short_id(&msg.chat_id),
                    msg.sender,
                    text
                ));
            }
        }
        ListTarget::Account => {
            let account = storage.get_account()?;
            let profile = storage.get_profile()?;
            let info = storage.get_archive_info()?;
            let combined = serde_json::json!({
                "archive_info": info,
                "account": account,
                "profile": profile,
            });
            if print_list_machine_output(cli, std::slice::from_ref(&combined), output)? {
                return Ok(());
            }
            output.print("[bold cyan]Account & Profile[/]");
            output.print(&format!("[dim]{}[/]", "─".repeat(CONTENT_DIVIDER_WIDTH)));
            if let Some(ref acc) = account {
                output.print(&format!(
                    "  [dim]Username:[/]     [bold]@{}[/]",
                    acc.username
                ));
                output.print(&format!("  [dim]Account ID:[/]   {}", acc.account_id));
                if let Some(ref name) = acc.display_name {
                    output.print(&format!("  [dim]Display Name:[/] {name}"));
                }
                if let Some(ref email) = acc.email {
                    output.print(&format!("  [dim]Email:[/]        {email}"));
                }
            } else if let Some(ref inf) = info {
                output.print(&format!(
                    "  [dim]Username:[/]     [bold]@{}[/]",
                    inf.username
                ));
                output.print(&format!("  [dim]Account ID:[/]   {}", inf.account_id));
                if let Some(ref name) = inf.display_name {
                    output.print(&format!("  [dim]Display Name:[/] {name}"));
                }
            }
            if let Some(ref prof) = profile {
                if let Some(ref bio) = prof.bio {
                    output.print(&format!("  [dim]Bio:[/]          {bio}"));
                }
                if let Some(ref loc) = prof.location {
                    output.print(&format!("  [dim]Location:[/]     {loc}"));
                }
                if let Some(ref web) = prof.website {
                    output.print(&format!("  [dim]Website:[/]      {web}"));
                }
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
    let format = ExportFormat::from_output(&cli.format);

    // Build export data based on target
    let export_data = match args.what {
        ExportTarget::Tweets => {
            let tweets = storage.get_all_tweets(args.limit)?;
            format_export(&tweets, &format)?
        }
        ExportTarget::Likes => {
            let likes = storage.get_all_likes(args.limit)?;
            format_export(&likes, &format)?
        }
        ExportTarget::Dms => {
            let dms = storage.get_all_dms(args.limit)?;
            format_export(&dms, &format)?
        }
        ExportTarget::Bookmarks => {
            let bookmarks = storage.get_all_bookmarks(args.limit)?;
            format_export(&bookmarks, &format)?
        }
        ExportTarget::Grok => {
            let grok = storage.get_all_grok_messages(args.limit)?;
            format_export(&grok, &format)?
        }
        ExportTarget::Followers => {
            let followers = storage.get_all_followers(args.limit)?;
            format_export(&followers, &format)?
        }
        ExportTarget::Following => {
            let following = storage.get_all_following(args.limit)?;
            format_export(&following, &format)?
        }
        ExportTarget::Blocks => {
            let blocks = storage.get_all_blocks(args.limit)?;
            format_export(&blocks, &format)?
        }
        ExportTarget::Mutes => {
            let mutes = storage.get_all_mutes(args.limit)?;
            format_export(&mutes, &format)?
        }
        ExportTarget::All => {
            // For "all", we create a combined structure
            let tweets = storage.get_all_tweets(args.limit)?;
            let likes = storage.get_all_likes(args.limit)?;
            let dms = storage.get_all_dms(args.limit)?;
            let bookmarks = storage.get_all_bookmarks(args.limit)?;
            let grok = storage.get_all_grok_messages(args.limit)?;
            let followers = storage.get_all_followers(args.limit)?;
            let following = storage.get_all_following(args.limit)?;
            let blocks = storage.get_all_blocks(args.limit)?;
            let mutes = storage.get_all_mutes(args.limit)?;

            match format {
                ExportFormat::Json => {
                    let combined = serde_json::json!({
                        "tweets": tweets,
                        "likes": likes,
                        "dms": dms,
                        "bookmarks": bookmarks,
                        "grok": grok,
                        "followers": followers,
                        "following": following,
                        "blocks": blocks,
                        "mutes": mutes
                    });
                    serde_json::to_string_pretty(&combined)?
                }
                ExportFormat::Toon => {
                    let combined = serde_json::json!({
                        "tweets": tweets,
                        "likes": likes,
                        "dms": dms,
                        "bookmarks": bookmarks,
                        "grok": grok,
                        "followers": followers,
                        "following": following,
                        "blocks": blocks,
                        "mutes": mutes
                    });
                    toon_rust::encode(combined, None)
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
                    for bm in &bookmarks {
                        jsonl_lines.push(format!(
                            r#"{{"type":"bookmark","data":{}}}"#,
                            serde_json::to_string(bm)?
                        ));
                    }
                    for g in &grok {
                        jsonl_lines.push(format!(
                            r#"{{"type":"grok","data":{}}}"#,
                            serde_json::to_string(g)?
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
                    for b in &blocks {
                        jsonl_lines.push(format!(
                            r#"{{"type":"block","data":{}}}"#,
                            serde_json::to_string(b)?
                        ));
                    }
                    for m in &mutes {
                        jsonl_lines.push(format!(
                            r#"{{"type":"mute","data":{}}}"#,
                            serde_json::to_string(m)?
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
        output.print_data(&export_data);
    }

    Ok(())
}

/// Format data for export based on the specified format
fn format_export<T: serde::Serialize>(data: &[T], format: &ExportFormat) -> Result<String> {
    match format {
        ExportFormat::Json => Ok(serde_json::to_string_pretty(data)?),
        ExportFormat::Toon => {
            let json = serde_json::to_value(data)?;
            Ok(toon_rust::encode(json, None))
        }
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
        let db = get_db_path(cli);
        let index = get_index_path(cli);
        match cli.format {
            OutputFormat::Json | OutputFormat::Jsonl => {
                let val = serde_json::json!({
                    "database": db,
                    "index": index,
                    "archive": config.paths.archive,
                    "config": config,
                });
                output.print_data(&serde_json::to_string(&val)?);
            }
            OutputFormat::JsonPretty => {
                let val = serde_json::json!({
                    "database": db,
                    "index": index,
                    "archive": config.paths.archive,
                    "config": config,
                });
                output.print_data(&serde_json::to_string_pretty(&val)?);
            }
            OutputFormat::Toon => {
                let val = serde_json::json!({
                    "database": db,
                    "index": index,
                    "archive": config.paths.archive,
                    "config": config,
                });
                output.print_data(&toon_rust::encode(val, None));
            }
            _ => {
                output.print("[bold cyan]Current Configuration[/]");
                output.print(&format!("  Database: {}", db.display()));
                output.print(&format!("  Index: {}", index.display()));
                if let Some(archive) = &config.paths.archive {
                    output.print(&format!("  Archive: {}", archive.display()));
                }
                output.print(&format!(
                    "  Search: default_limit={}, highlight={}, fuzzy={}, min_score={}, cache_size={}",
                    config.search.default_limit,
                    config.search.highlight,
                    config.search.fuzzy,
                    config.search.min_score,
                    config.search.cache_size
                ));
                output.print(&format!(
                    "  Indexing: parallel={}, buffer_size_mb={}, threads={}, skip_types={:?}",
                    config.indexing.parallel,
                    config.indexing.buffer_size_mb,
                    config.indexing.threads,
                    config.indexing.skip_types
                ));
                output.print(&format!(
                    "  Output: format={}, colors={}, quiet={}, timings={}",
                    config.output.format,
                    config.output.colors,
                    config.output.quiet,
                    config.output.timings
                ));
                output.print(&format!(
                    "  Semantic: model={}, dimensions={:?}, reranker={}, rerank={}, daemon={}, rerank_top={}",
                    config.semantic.model.as_deref().unwrap_or("default"),
                    config.semantic.dimensions,
                    config.semantic.reranker.as_deref().unwrap_or("default"),
                    config.semantic.rerank,
                    config.semantic.daemon,
                    config.semantic.rerank_top
                ));
                output.print(&format!(
                    "  Two-Tier: fast_model={}, quality_model={}, blend_factor={}, quality_timeout_ms={}",
                    config.semantic.two_tier.fast_model.as_deref().unwrap_or("default"),
                    config.semantic.two_tier.quality_model.as_deref().unwrap_or("default"),
                    config.semantic.two_tier.blend_factor,
                    config.semantic.two_tier.quality_timeout_ms
                ));
            }
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
        OutputFormat::Json | OutputFormat::JsonPretty | OutputFormat::Jsonl => {
            let json = if matches!(cli.format, OutputFormat::JsonPretty) {
                serde_json::to_string_pretty(&thread)?
            } else {
                serde_json::to_string(&thread)?
            };
            output.print_data(&json);
        }
        OutputFormat::Toon => {
            let json = serde_json::to_value(&thread)?;
            output.print_data(&toon_rust::encode(json, None));
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

fn apply_semantic_config_set(config: &mut Config, key: &str, value: &str) -> Result<bool> {
    match key {
        "semantic.model" => {
            if value.is_empty() {
                anyhow::bail!("semantic.model cannot be empty.");
            }
            config.semantic.model = Some(value.to_string());
        }
        "semantic.dimensions" => {
            if value.is_empty() || value.eq_ignore_ascii_case("none") {
                config.semantic.dimensions = None;
            } else {
                config.semantic.dimensions = Some(parse_usize(value, key)?);
            }
        }
        "semantic.reranker" => {
            if value.is_empty() {
                anyhow::bail!("semantic.reranker cannot be empty.");
            }
            config.semantic.reranker = Some(value.to_string());
        }
        "semantic.rerank" => {
            config.semantic.rerank = parse_bool(value, key)?;
        }
        "semantic.daemon" => {
            config.semantic.daemon = parse_bool(value, key)?;
        }
        "semantic.rerank_top" => {
            config.semantic.rerank_top = parse_usize(value, key)?;
        }
        "semantic.two_tier.fast_model" => {
            if value.is_empty() {
                anyhow::bail!("semantic.two_tier.fast_model cannot be empty.");
            }
            config.semantic.two_tier.fast_model = Some(value.to_string());
        }
        "semantic.two_tier.quality_model" => {
            if value.is_empty() {
                anyhow::bail!("semantic.two_tier.quality_model cannot be empty.");
            }
            config.semantic.two_tier.quality_model = Some(value.to_string());
        }
        "semantic.two_tier.blend_factor" => {
            let parsed = parse_f32(value, key)?;
            if !(0.0..=1.0).contains(&parsed) {
                anyhow::bail!("{key} must be between 0.0 and 1.0.");
            }
            config.semantic.two_tier.blend_factor = parsed;
        }
        "semantic.two_tier.quality_timeout_ms" => {
            config.semantic.two_tier.quality_timeout_ms = value
                .parse()
                .map_err(|_| anyhow::anyhow!("Invalid integer value for {key}: {value}"))?;
        }
        _ => return Ok(false),
    }
    Ok(true)
}

fn apply_config_set(config: &mut Config, raw: &str) -> Result<()> {
    let (key, value) = raw
        .split_once('=')
        .map(|(k, v)| (k.trim(), v.trim()))
        .ok_or_else(|| anyhow::anyhow!("Invalid --set format. Use key=value."))?;

    if key.is_empty() {
        anyhow::bail!("Invalid --set key. Use key=value.");
    }

    if apply_semantic_config_set(config, key, value)? {
        return Ok(());
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

    // ========== Apply Fixes (--fix) ==========
    let mut fix_checks: Vec<HealthCheck> = Vec::new();
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
                                fix_checks.push(HealthCheck {
                                    category: CheckCategory::Database,
                                    name: "Auto-fix (FTS rebuild)".into(),
                                    status: CheckStatus::Pass,
                                    message: format!(
                                        "fts_tweets={}; fts_likes={}; fts_dms={}; fts_grok={}; fts_bookmarks={}",
                                        stats.tweets,
                                        stats.likes,
                                        stats.dms,
                                        stats.grok,
                                        stats.bookmarks
                                    ),
                                    suggestion: None,
                                });
                            }
                            Err(err) => {
                                fix_checks.push(HealthCheck {
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
                                fix_checks.push(HealthCheck {
                                    category: CheckCategory::Database,
                                    name: "Auto-fix (DM conversations)".into(),
                                    status: CheckStatus::Pass,
                                    message: format!("Rebuilt {rebuilt} conversations"),
                                    suggestion: None,
                                });
                            }
                            Err(err) => {
                                fix_checks.push(HealthCheck {
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

                    let vector_file = index_path.join(VECTOR_INDEX_FILENAME);
                    let needs_vector_rebuild = index_path.exists()
                        && storage.embedding_count().unwrap_or(0) > 0
                        && (!vector_file.exists()
                            || std::fs::metadata(&vector_file).map_or(true, |m| m.len() < 32));
                    if needs_vector_rebuild {
                        let rebuild_res = if !storage
                            .load_all_embeddings_raw_filtered(Some("quality"))
                            .unwrap_or_default()
                            .is_empty()
                        {
                            xf::vector::write_vector_index_named(
                                &index_path,
                                &storage,
                                VECTOR_INDEX_FILENAME,
                                Some("quality"),
                            )
                        } else if !storage
                            .load_all_embeddings_raw_filtered(Some("fast"))
                            .unwrap_or_default()
                            .is_empty()
                        {
                            xf::vector::write_vector_index_named(
                                &index_path,
                                &storage,
                                VECTOR_INDEX_FILENAME,
                                Some("fast"),
                            )
                        } else {
                            xf::vector::write_vector_index(&index_path, &storage)
                        };
                        match rebuild_res {
                            Ok(_stats) => {
                                applied_any = true;
                                fix_checks.push(HealthCheck {
                                    category: CheckCategory::Index,
                                    name: "Auto-fix (Vector index)".into(),
                                    status: CheckStatus::Pass,
                                    message: "Rebuilt vector.idx from SQLite embeddings".into(),
                                    suggestion: None,
                                });
                            }
                            Err(err) => {
                                fix_checks.push(HealthCheck {
                                    category: CheckCategory::Index,
                                    name: "Auto-fix (Vector index)".into(),
                                    status: CheckStatus::Warning,
                                    message: format!("Failed to rebuild vector.idx: {err}"),
                                    suggestion: Some(
                                        "Run 'xf index --force' to rebuild vector index.".into(),
                                    ),
                                });
                            }
                        }
                    }

                    match storage.optimize() {
                        Ok(()) => {
                            fix_checks.push(HealthCheck {
                                category: CheckCategory::Database,
                                name: "Auto-fix (SQLite optimize)".into(),
                                status: CheckStatus::Pass,
                                message: "PRAGMA optimize completed".into(),
                                suggestion: None,
                            });
                        }
                        Err(err) => {
                            fix_checks.push(HealthCheck {
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
                        fix_checks.push(HealthCheck {
                            category: CheckCategory::Database,
                            name: "Auto-fix".into(),
                            status: CheckStatus::Pass,
                            message: "No structural issues detected".into(),
                            suggestion: None,
                        });
                    }
                }
                Err(err) => {
                    fix_checks.push(HealthCheck {
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
            fix_checks.push(HealthCheck {
                category: CheckCategory::Database,
                name: "Auto-fix".into(),
                status: CheckStatus::Error,
                message: format!("No database found at {}", db_path.display()),
                suggestion: Some("Run 'xf index <archive_path>' to create the database".into()),
            });
        }
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

    all_checks.extend(fix_checks);

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
        OutputFormat::Json | OutputFormat::Jsonl => {
            let doctor_output = DoctorOutput {
                checks: all_checks,
                summary,
                suggestions,
                runtime_ms,
            };
            output.print_data(&serde_json::to_string(&doctor_output)?);
        }
        OutputFormat::JsonPretty => {
            let doctor_output = DoctorOutput {
                checks: all_checks,
                summary,
                suggestions,
                runtime_ms,
            };
            output.print_data(&serde_json::to_string_pretty(&doctor_output)?);
        }
        OutputFormat::Toon => {
            let doctor_output = DoctorOutput {
                checks: all_checks,
                summary,
                suggestions,
                runtime_ms,
            };
            let json = serde_json::to_value(&doctor_output)?;
            output.print_data(&toon_rust::encode(json, None));
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

    let storage = Storage::open(&db_path)?;
    let stats = storage.get_stats()?;
    let sql_doc_count = usize::try_from(
        stats.tweets_count
            + stats.likes_count
            + stats.dms_count
            + stats.grok_messages_count
            + stats.bookmarks_count,
    )
    .unwrap_or(0);
    let has_tantivy_meta = index_path.join("meta.json").is_file();

    // Check that index exists (unless SQLite has indexed docs for FTS5 fallback)
    if !has_tantivy_meta && sql_doc_count == 0 {
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

    let search = if has_tantivy_meta {
        match SearchEngine::open(&index_path) {
            Ok(engine) if engine.doc_count() > 0 || sql_doc_count == 0 => Some(engine),
            Ok(_) => {
                info!(
                    "Tantivy index is empty while SQLite has {sql_doc_count} docs; using SQLite FTS5 fallback in shell"
                );
                None
            }
            Err(err) if sql_doc_count > 0 => {
                warn!("Failed to open Tantivy index ({err}); using SQLite FTS5 fallback in shell");
                None
            }
            Err(err) => return Err(err),
        }
    } else {
        info!(
            "Tantivy index missing at '{}'; using SQLite FTS5 fallback in shell ({sql_doc_count} docs)",
            index_path.display()
        );
        None
    };

    let config = repl::ReplConfig {
        prompt: args.prompt.clone(),
        page_size: args.page_size,
        no_history: args.no_history,
        history_file: args.history_file.clone(),
        index_path: Some(index_path),
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
    let embedder_dim = std::cell::Cell::new(0usize);
    let mut full_result = run_full_benchmark(
        || {
            let embedder = registry.embedder(&config)?;
            embedder_dim.set(embedder.dimension());
            Ok(embedder)
        },
        &corpus,
        &bench_cfg,
    )
    .with_context(|| format!("Benchmark failed for model {model_name}"))?;

    if !args.reranker.is_empty() && !args.reranker.eq_ignore_ascii_case("none") {
        if let Some(reranker) = registry
            .reranker_by_name(&args.reranker)
            .with_context(|| format!("Failed to load reranker: {}", args.reranker))?
        {
            let embedder = registry.embedder(&config).with_context(|| {
                format!("Failed to load embedder for rerank eval: {model_name}")
            })?;
            let rerank_bench = xf::benchmark::rerank_quality::RerankQualityBenchmark::new(corpus)
                .with_embedder(embedder)
                .with_reranker(reranker);
            let reports = rerank_bench
                .run()
                .with_context(|| "Reranker quality benchmark failed")?;
            if let Some(rerank_report) = reports.iter().find(|r| r.reranker != "none") {
                full_result.reranker_lift = Some(rerank_report.lift_over_baseline.ndcg_10_lift_pct);
            }
        }
    }

    let metadata = BenchmarkMetadata {
        model: full_result.model_name.clone(),
        dimension: embedder_dim.get(),
        batch_size: args.batch_size,
        warmup_iters: args.warmup_iters,
        measure_iters: args.measure_iters,
    };

    let output_dir = &args.output_dir;
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("Failed to create output dir: {}", output_dir.display()))?;

    let timestamp = chrono::Utc::now().format("%Y%m%d_%H%M%S");
    let base = output_dir.join(format!(
        "benchmark_{}_{}",
        metadata.model.replace('/', "_"),
        timestamp
    ));
    let json_path = base.with_extension("json");
    let md_path = base.with_extension("md");
    let csv_path = base.with_extension("csv");

    reporters::write_full_json(&json_path, &metadata, &full_result)?;
    reporters::write_full_markdown(&md_path, &metadata, &full_result)?;
    reporters::write_full_csv(&csv_path, &metadata, &full_result)?;

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
    output.print_data(&serde_json::to_string_pretty(&docs)?);
    Ok(())
}

// ============================================================================
// Models Command
// ============================================================================

#[allow(clippy::too_many_lines)]
fn cmd_models(cli: &Cli, args: &cli::ModelsArgs, output: &Output) -> Result<()> {
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

            if print_list_machine_output(cli, &filtered, output)? {
                return Ok(());
            }

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
        ModelsCommand::Info(info_args) => {
            let models = registry.list_models();
            let model = models.iter().find(|m| m.name == info_args.name);

            if let Some(m) = model {
                match cli.format {
                    OutputFormat::Json | OutputFormat::Jsonl => {
                        output.print_json(m)?;
                    }
                    OutputFormat::JsonPretty => {
                        output.print_json_pretty(m)?;
                    }
                    OutputFormat::Toon => {
                        let json = serde_json::to_value(m)?;
                        output.print_data(&toon_rust::encode(json, None));
                    }
                    OutputFormat::Csv => {
                        let csv = format_export(std::slice::from_ref(m), &ExportFormat::Csv)?;
                        output.print_data(csv.trim_end_matches('\n'));
                    }
                    OutputFormat::Text | OutputFormat::Compact => {
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
fn cmd_daemon(cli: &Cli, args: &cli::DaemonArgs, output: &Output) -> Result<()> {
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
            if let Some(timeout_secs) = start_args.idle_timeout {
                if timeout_secs > 0 {
                    config.idle_timeout = Duration::from_secs(timeout_secs);
                } else {
                    config.idle_timeout = Duration::MAX; // No timeout
                }
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

                let mut cmd = std::process::Command::new(std::env::current_exe()?);
                cmd.args(["daemon", "start", "--foreground"]);

                if let Some(config_path) = &start_args.config {
                    cmd.args(["--config", &config_path.to_string_lossy()]);
                }
                if let Some(socket) = &start_args.socket {
                    cmd.args(["--socket", &socket.to_string_lossy()]);
                }
                if let Some(timeout_secs) = start_args.idle_timeout {
                    cmd.args(["--idle-timeout", &timeout_secs.to_string()]);
                }
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
                Ok(status) => match cli.format {
                    OutputFormat::Json | OutputFormat::Jsonl => {
                        output.print_json(&status)?;
                    }
                    OutputFormat::JsonPretty => {
                        output.print_json_pretty(&status)?;
                    }
                    OutputFormat::Toon => {
                        let json = serde_json::to_value(&status)?;
                        output.print_data(&toon_rust::encode(json, None));
                    }
                    _ => {
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

                        // Show embedding job status if available
                        show_embedding_job_status(&mut client, &runtime, output);
                    }
                },
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

/// Show embedding job status from the daemon.
fn show_embedding_job_status(
    client: &mut xf::daemon::DaemonClient,
    runtime: &tokio::runtime::Runtime,
    output: &Output,
) {
    use xf::daemon::EmbeddingJobInfo;

    // Try to get job status from the default database
    // Since we don't know which DB to query, we query for all jobs by passing None
    // But the daemon needs a DB path, so we check the default database
    let db_path = get_db_path(&Cli::parse_from::<[_; 1], &str>(["xf"]));

    if let Ok(jobs) = runtime
        .block_on(client.embedding_job_status(Some(&db_path.to_string_lossy())))
        as Result<Vec<EmbeddingJobInfo>, _>
    {
        if !jobs.is_empty() {
            output.print("");
            output.print("[bold underline]Embedding Jobs[/]");
            output.print("");

            for job in &jobs {
                let status_icon = match job.status.as_str() {
                    "pending" => "[yellow]○[/]",
                    "in_progress" => "[cyan]⟳[/]",
                    "completed" => "[green]✓[/]",
                    "failed" => "[red]✗[/]",
                    _ => "?",
                };

                let progress = if job.total_docs > 0 {
                    format!("{:.1}%", job.progress_pct)
                } else {
                    "0%".to_string()
                };

                #[allow(clippy::cast_sign_loss)]
                let completed = job.completed_docs as u64;
                #[allow(clippy::cast_sign_loss)]
                let total = job.total_docs as u64;
                output.print(&format!(
                    "  {} [bold]{}[/] — {} ({}/{})",
                    status_icon,
                    job.model_id,
                    progress,
                    format_number_u64(completed),
                    format_number_u64(total)
                ));

                if let Some(ref err) = job.error_message {
                    output.print(&format!("    [red]Error:[/] {err}"));
                }
            }
        }
    }
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

                        // Check consistency with DB (accounting for two-tier fast+quality rows)
                        let db_count = storage.embedding_count().unwrap_or(0);
                        let diff = if record_count as i64 * 2 == db_count {
                            0
                        } else {
                            (record_count as i64 - db_count).abs()
                        };

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

    for (filename, label) in [
        (VECTOR_INDEX_FAST_FILENAME, "Vector Index (fast tier)"),
        (VECTOR_INDEX_QUALITY_FILENAME, "Vector Index (quality tier)"),
    ] {
        let tier_file = index_path.join(filename);
        if tier_file.exists() {
            match std::fs::metadata(&tier_file) {
                Ok(meta) if meta.len() >= 32 => {
                    checks.push(HealthCheck {
                        category: CheckCategory::Index,
                        name: label.into(),
                        status: CheckStatus::Pass,
                        message: format!("Found ({})", format_bytes(meta.len())),
                        suggestion: None,
                    });
                }
                Ok(meta) => {
                    checks.push(HealthCheck {
                        category: CheckCategory::Index,
                        name: label.into(),
                        status: CheckStatus::Error,
                        message: format!("File too small ({})", format_bytes(meta.len())),
                        suggestion: Some("Run 'xf index --two-tier --force' to regenerate".into()),
                    });
                }
                Err(e) => {
                    checks.push(HealthCheck {
                        category: CheckCategory::Index,
                        name: label.into(),
                        status: CheckStatus::Error,
                        message: format!("Cannot stat: {e}"),
                        suggestion: Some("Check file permissions".into()),
                    });
                }
            }
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
