//! xf - Ultra-fast X data archive search
//!
//! This library provides the core functionality for indexing and searching
//! X data archives exported from x.com.
//!
//! # Modules
//!
//! - [`cli`] - Command-line interface definitions
//! - [`error`] - Custom error types with rich context
//! - [`model`] - Data models for X archive data
//! - [`parser`] - Archive parsing and data extraction
//! - [`search`] - Tantivy-based full-text search engine
//! - [`storage`] - `SQLite` storage layer

pub mod benchmark;
pub mod canonicalize;
pub mod cli;
pub mod completions;
pub mod config;
pub mod daemon;
pub mod date_parser;
pub mod doctor;
pub mod embedder;
pub mod error;
pub mod fastembed_embedder;
pub mod flashrank_reranker;
pub mod hash_embedder;
pub mod hybrid;
pub mod list_tables;
pub mod logging;
pub mod messages;
pub mod model;
pub mod model2vec_embedder;
pub mod model_registry;
pub mod mxbai_reranker;
pub mod output;
pub mod parser;
pub mod perf;
pub mod progress;
pub mod repl;
pub mod rerank_step;
pub mod reranker;
pub mod robot_docs;
pub mod search;
pub mod search_results;
pub mod static_mrl_embedder;
pub mod stats_analytics;
pub mod stats_dashboard;
pub mod storage;
pub mod theme;
pub mod tweet_card;
pub mod vector;

pub use cli::*;
pub use error::{
    Result, ResultExt, VALID_CONFIG_KEYS, VALID_DATA_TYPES, VALID_OUTPUT_FIELDS, XfError,
    find_closest_match, format_did_you_mean, format_error, format_unknown_value_error,
};
pub use model::*;
pub use parser::ArchiveParser;
pub use search::SearchEngine;
pub use storage::Storage;

use chrono::{DateTime, Datelike, Utc};

/// Default database filename
pub const DEFAULT_DB_NAME: &str = "xf.db";

/// Default index directory name
pub const DEFAULT_INDEX_DIR: &str = "xf_index";

/// Default archive path - if user extracts to this location, xf works without arguments
pub const DEFAULT_ARCHIVE_PATH: &str = "/data/projects/my_twitter_data";

/// Standard width for content dividers in CLI output
pub const CONTENT_DIVIDER_WIDTH: usize = 60;

/// Standard width for major header dividers in CLI output
pub const HEADER_DIVIDER_WIDTH: usize = 70;

const BYTES_PER_KB: u64 = 1024;
const BYTES_PER_MB: u64 = 1024 * 1024;
const BYTES_PER_GB: u64 = 1024 * 1024 * 1024;

/// Get the default data directory for xf
#[must_use]
pub fn default_data_dir() -> std::path::PathBuf {
    dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("."))
        .join("xf")
}

/// Get the default database path
#[must_use]
pub fn default_db_path() -> std::path::PathBuf {
    default_data_dir().join(DEFAULT_DB_NAME)
}

/// Get the default index path
#[must_use]
pub fn default_index_path() -> std::path::PathBuf {
    default_data_dir().join(DEFAULT_INDEX_DIR)
}

/// Format an integer with thousands separators.
#[must_use]
pub fn format_number(value: i64) -> String {
    let abs = value.unsigned_abs().to_string();
    let mut out = String::with_capacity(abs.len() + abs.len() / 3);

    for (idx, ch) in abs.chars().rev().enumerate() {
        if idx > 0 && idx % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }

    let mut formatted: String = out.chars().rev().collect();
    if value < 0 {
        formatted.insert(0, '-');
    }
    formatted
}

/// Format an unsigned integer with thousands separators.
#[must_use]
pub fn format_number_u64(value: u64) -> String {
    let mut out = String::with_capacity(24);

    for (idx, ch) in value.to_string().chars().rev().enumerate() {
        if idx > 0 && idx % 3 == 0 {
            out.push(',');
        }
        out.push(ch);
    }

    out.chars().rev().collect()
}

/// Format a usize with thousands separators.
#[must_use]
pub fn format_number_usize(value: usize) -> String {
    format_number_u64(u64::try_from(value).unwrap_or(u64::MAX))
}

/// Format a datetime as a human-friendly relative string.
///
/// Uses smart thresholds for readability:
/// - < 1 minute: "just now"
/// - < 1 hour: "Nm ago"
/// - < 24 hours: "Nh ago"
/// - < 7 days: "Nd ago"
/// - Same calendar year: "Mon D"
/// - Different year: "Mon D, YYYY"
#[must_use]
pub fn format_relative_date(dt: DateTime<Utc>) -> String {
    format_relative_date_with_base(dt, Utc::now())
}

/// Format a datetime relative to a fixed base time (useful for tests).
#[must_use]
pub fn format_relative_date_with_base(dt: DateTime<Utc>, now: DateTime<Utc>) -> String {
    let duration = now.signed_duration_since(dt);

    // Handle future dates (shouldn't happen, but be safe)
    if duration.num_seconds() < 0 {
        return dt.format("%b %d, %Y").to_string();
    }

    let seconds = duration.num_seconds();
    let minutes = duration.num_minutes();
    let hours = duration.num_hours();
    let days = duration.num_days();

    if seconds < 60 {
        "just now".to_string()
    } else if minutes < 60 {
        format!("{minutes}m ago")
    } else if hours < 24 {
        format!("{hours}h ago")
    } else if days < 7 {
        format!("{days}d ago")
    } else if dt.year() == now.year() {
        // Same calendar year: "Jan 15"
        dt.format("%b %d").to_string()
    } else {
        // Different year: "Jan 15, 2023"
        dt.format("%b %d, %Y").to_string()
    }
}

/// Format an optional datetime with human-friendly output.
#[must_use]
pub fn format_optional_date(value: Option<DateTime<Utc>>) -> String {
    value.map_or_else(|| "unknown".to_string(), format_relative_date)
}

/// Format a duration into a short human-friendly string.
#[must_use]
pub fn format_duration(duration: std::time::Duration) -> String {
    let secs = duration.as_secs();
    if secs == 0 {
        let ms = duration.as_secs_f64() * 1000.0;
        return format!("{ms:.1}ms");
    }
    if secs < 60 {
        return format!("{:.2}s", duration.as_secs_f64());
    }
    if secs < 3600 {
        let minutes = secs / 60;
        let seconds = secs % 60;
        return format!("{minutes}m {seconds:02}s");
    }
    let hours = secs / 3600;
    let minutes = (secs % 3600) / 60;
    format!("{hours}h {minutes:02}m")
}

/// Generate embeddings for all documents in the archive.
///
/// This function creates embeddings for tweets, likes, DMs, and Grok messages
/// using the hash-based embedder, storing them in the `SQLite` embeddings table.
///
/// # Errors
///
/// Returns an error if any storage query fails or if embedding generation
/// encounters an unexpected failure.
///
/// # Panics
///
/// Panics only if the progress bar template is invalid (a programming error).
/// Configuration for embedding generation.
#[derive(Debug, Clone)]
pub struct EmbeddingConfig {
    /// Show progress bar during embedding generation.
    pub show_progress: bool,
    /// Use semantic (ML-based) embeddings instead of hash embeddings.
    /// When true, uses FastEmbed with MiniLM model for true semantic similarity.
    /// When false, uses hash-based embeddings (faster but lexical, not semantic).
    pub use_semantic: bool,
    /// Optional embedder model name (registry key).
    pub model: Option<String>,
    /// Optional MRL dimension override.
    pub dimensions: Option<usize>,
    /// Enable two-tier embedding generation (fast + quality models).
    pub two_tier: bool,
    /// Fast model name for two-tier mode.
    pub fast_model: Option<String>,
    /// Quality model name for two-tier mode.
    pub quality_model: Option<String>,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            show_progress: true,
            use_semantic: false,
            model: None,
            dimensions: None,
            two_tier: false,
            fast_model: None,
            quality_model: None,
        }
    }
}

#[allow(clippy::too_many_lines)]
pub fn generate_embeddings(storage: &Storage, show_progress: bool) -> Result<()> {
    generate_embeddings_with_config(
        storage,
        &EmbeddingConfig {
            show_progress,
            use_semantic: false,
            model: None,
            dimensions: None,
            two_tier: false,
            fast_model: None,
            quality_model: None,
        },
    )
}

/// Generate embeddings with full configuration options.
///
/// When `config.use_semantic` is true, uses the FastEmbed MiniLM model for
/// true semantic embeddings. This is slower (~100 items/sec) but enables
/// semantic search where "happy" matches "joyful".
///
/// When `config.use_semantic` is false (default), uses hash-based embeddings
/// which are much faster but only capture lexical similarity.
///
/// # Errors
///
/// Returns an error if database operations fail or if the semantic model
/// is requested but not available.
///
/// # Panics
///
/// Panics only if the progress bar template is invalid (a programming error).
#[allow(clippy::too_many_lines)]
pub fn generate_embeddings_with_config(storage: &Storage, config: &EmbeddingConfig) -> Result<()> {
    use crate::canonicalize::{canonicalize_for_embedding, content_hash};
    use crate::embedder::Embedder;
    use crate::hash_embedder::HashEmbedder;
    use crate::model_registry::{EmbedderConfig as RegistryConfig, ModelRegistry};
    use colored::Colorize;
    use indicatif::{ProgressBar, ProgressStyle};
    use rayon::prelude::*;
    use std::collections::{HashMap, HashSet};
    use std::time::Instant;

    use tracing::{info, warn};

    // Type alias for embedding records: (doc_id, doc_type, embedding, content_hash)
    type EmbedRecord = (String, String, Vec<f32>, Option<[u8; 32]>);

    const EMBED_CHUNK_SIZE: usize = 1000;

    // Two-tier mode: run two separate embedding passes
    if config.two_tier {
        if config.show_progress {
            println!();
            println!(
                "{}",
                "Two-tier embedding: generating fast + quality embeddings..."
                    .bold()
                    .cyan()
            );
        }

        // Pass 1: Fast model
        let fast_model = config
            .fast_model
            .clone()
            .unwrap_or_else(|| "hash-fnv1a-384".to_string());
        if config.show_progress {
            println!(
                "  {} Fast pass (model: {})",
                "→".cyan(),
                fast_model.dimmed()
            );
        }
        let fast_config = EmbeddingConfig {
            show_progress: config.show_progress,
            use_semantic: false,
            model: Some(fast_model),
            dimensions: config.dimensions,
            two_tier: false,
            fast_model: None,
            quality_model: None,
        };
        generate_embeddings_for_model(storage, &fast_config, "fast")?;

        // Pass 2: Quality model
        let quality_model = config
            .quality_model
            .clone()
            .unwrap_or_else(|| "all-MiniLM-L6-v2".to_string());
        if config.show_progress {
            println!(
                "  {} Quality pass (model: {})",
                "→".cyan(),
                quality_model.dimmed()
            );
        }
        let quality_config = EmbeddingConfig {
            show_progress: config.show_progress,
            use_semantic: true,
            model: Some(quality_model),
            dimensions: config.dimensions,
            two_tier: false,
            fast_model: None,
            quality_model: None,
        };
        generate_embeddings_for_model(storage, &quality_config, "quality")?;

        return Ok(());
    }

    let embed_start = Instant::now();

    // Create the appropriate embedder based on config
    let registry = ModelRegistry::new();
    let embedder_box: Box<dyn Embedder> = if let Some(model) = &config.model {
        let mut cfg = RegistryConfig::new(model);
        cfg.dimensions = config.dimensions;
        cfg.show_progress = config.show_progress;
        registry
            .embedder(&cfg)
            .map_err(|e| anyhow::anyhow!("Failed to load embedder {model}: {e}"))?
    } else if config.use_semantic {
        info!("Loading semantic embedder (default MiniLM)...");
        let mut cfg = RegistryConfig::new("all-MiniLM-L6-v2");
        cfg.dimensions = config.dimensions;
        cfg.show_progress = config.show_progress;
        let embedder = registry
            .embedder(&cfg)
            .map_err(|e| anyhow::anyhow!("Failed to load semantic model: {e}"))?;
        info!("Semantic embedder loaded successfully");
        embedder
    } else {
        Box::new(HashEmbedder::default())
    };
    let embedder: &dyn Embedder = embedder_box.as_ref();
    // Use smaller batches for semantic embeddings (more memory intensive)
    let store_batch_size = if embedder.is_semantic() { 50 } else { 100 };

    if config.show_progress {
        println!();
        let mode = if embedder.is_semantic() {
            "semantic (ML)"
        } else {
            "hash-based"
        };
        println!(
            "{}",
            format!("Generating {mode} embeddings...").bold().cyan()
        );
    }

    // Fetch all collections first to pre-allocate
    let tweets = storage.get_all_tweets(None)?;
    let likes = storage.get_all_likes(None)?;
    let dms = storage.get_all_dms(None)?;
    let grok_msgs = storage.get_all_grok_messages(None)?;

    // Pre-allocate with known capacity
    let capacity = tweets.len() + likes.len() + dms.len() + grok_msgs.len();
    let mut docs: Vec<(String, String, &'static str)> = Vec::with_capacity(capacity);

    // Tweets
    for tweet in &tweets {
        docs.push((tweet.id.clone(), tweet.full_text.clone(), "tweet"));
    }

    // Likes (only if they have text)
    for like in &likes {
        if let Some(ref text) = like.full_text {
            if !text.is_empty() {
                docs.push((like.tweet_id.clone(), text.clone(), "like"));
            }
        }
    }

    // DMs
    for dm in &dms {
        if !dm.text.is_empty() {
            docs.push((dm.id.clone(), dm.text.clone(), "dm"));
        }
    }

    // Grok messages
    for msg in &grok_msgs {
        if !msg.message.is_empty() {
            // Use same doc_id format as Tantivy indexing (search.rs:344-350)
            let doc_id = format!(
                "{}_{}_{}_{}",
                msg.chat_id,
                msg.created_at.timestamp(),
                msg.created_at.timestamp_subsec_nanos(),
                msg.sender
            );
            docs.push((doc_id, msg.message.clone(), "grok"));
        }
    }

    if docs.is_empty() {
        if config.show_progress {
            println!("  {} No documents to embed", "⚠".yellow());
        }
        return Ok(());
    }

    let existing_hashes_by_doc = storage.load_embedding_hashes_by_doc()?;
    let mut existing_hashes: HashSet<[u8; 32]> = HashSet::new();
    for by_type in existing_hashes_by_doc.values() {
        for hash in by_type.values() {
            existing_hashes.insert(*hash);
        }
    }

    // Create progress bar
    let pb = if config.show_progress {
        let pb = ProgressBar::new(docs.len() as u64);
        let style = ProgressStyle::default_bar()
            .template("  {spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("█▓░");
        pb.set_style(style);
        Some(pb)
    } else {
        None
    };

    // Generate and store embeddings in batches
    let mut stored_count = 0;
    let mut reused_count = 0;
    let mut skipped_count = 0;

    for chunk in docs.chunks(EMBED_CHUNK_SIZE) {
        let mut batch: Vec<EmbedRecord> = Vec::new();
        let mut candidates: Vec<(String, &'static str, String, [u8; 32])> = Vec::new();

        for (doc_id, text, doc_type) in chunk {
            // Canonicalize text
            let canonical = canonicalize_for_embedding(text);
            if canonical.is_empty() {
                skipped_count += 1;
                if let Some(ref pb) = pb {
                    pb.inc(1);
                }
                continue;
            }

            // Compute content hash for deduplication
            let hash = content_hash(&canonical);

            // Skip if this doc already has the same content hash.
            if let Some(existing_hash) = existing_hashes_by_doc
                .get(doc_id)
                .and_then(|by_type| by_type.get(*doc_type))
            {
                if existing_hash == &hash {
                    skipped_count += 1;
                    if let Some(ref pb) = pb {
                        pb.inc(1);
                    }
                    continue;
                }
            }

            candidates.push((doc_id.clone(), *doc_type, canonical, hash));
            if let Some(ref pb) = pb {
                pb.inc(1);
            }
        }

        let mut batch_cache: HashMap<[u8; 32], Vec<f32>> = HashMap::new();
        let mut needed_hashes: Vec<[u8; 32]> = Vec::new();
        let mut needed_hashes_set: HashSet<[u8; 32]> = HashSet::new();
        for (_, _, _, hash) in &candidates {
            if existing_hashes.contains(hash) && needed_hashes_set.insert(*hash) {
                needed_hashes.push(*hash);
            }
        }

        if !needed_hashes.is_empty() {
            let fetched = storage.load_embeddings_by_hashes(&needed_hashes)?;
            for (hash, embedding) in fetched {
                batch_cache.insert(hash, embedding);
            }
        }

        let mut new_hashes: Vec<(String, String, [u8; 32])> = Vec::new();
        let mut new_hashes_set: HashSet<[u8; 32]> = HashSet::new();
        for (doc_id, _doc_type, canonical, hash) in &candidates {
            if batch_cache.contains_key(hash) {
                continue;
            }
            if new_hashes_set.insert(*hash) {
                new_hashes.push((doc_id.clone(), canonical.clone(), *hash));
            }
        }

        if !new_hashes.is_empty() {
            let computed_embeddings: Vec<([u8; 32], Vec<f32>)> = new_hashes
                .par_iter()
                .filter_map(
                    |(doc_id, canonical, hash)| match embedder.embed(canonical) {
                        Ok(embedding) => Some((*hash, embedding)),
                        Err(e) => {
                            warn!("Failed to embed doc {}: {}", doc_id, e);
                            None
                        }
                    },
                )
                .collect();

            for (hash, embedding) in computed_embeddings {
                batch_cache.insert(hash, embedding);
            }
        }

        let mut seen_new_hashes: HashSet<[u8; 32]> = HashSet::new();
        for (doc_id, doc_type, _canonical, hash) in candidates {
            // Reuse an existing embedding if identical content exists.
            if let Some(existing_embedding) = batch_cache.get(&hash) {
                batch.push((
                    doc_id,
                    doc_type.to_string(),
                    existing_embedding.clone(),
                    Some(hash),
                ));

                if existing_hashes.contains(&hash) || !seen_new_hashes.insert(hash) {
                    reused_count += 1;
                }
            } else {
                skipped_count += 1;
            }
        }

        // Store batch
        if !batch.is_empty() {
            for chunk in batch.chunks(store_batch_size) {
                storage.store_embeddings_batch(chunk)?;
                stored_count += chunk.len();
            }
        }
    }

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    let embed_elapsed = format_duration(embed_start.elapsed());
    let generated_count = stored_count.saturating_sub(reused_count);
    if config.show_progress {
        println!(
            "  {} {} embeddings stored {}",
            "✓".green(),
            format_number_usize(stored_count).bold(),
            format!("({embed_elapsed})").dimmed()
        );
        if reused_count > 0 {
            println!(
                "  {} {} reused from identical content",
                "·".dimmed(),
                format_number_usize(reused_count).dimmed()
            );
        }
        if generated_count > 0 && reused_count > 0 {
            println!(
                "  {} {} generated",
                "·".dimmed(),
                format_number_usize(generated_count).dimmed()
            );
        }
        if skipped_count > 0 {
            println!(
                "  {} {} skipped (empty or unchanged)",
                "·".dimmed(),
                format_number_usize(skipped_count).dimmed()
            );
        }
    }

    Ok(())
}

/// Generate embeddings for a specific model and store with the given `model_id`.
///
/// Used by two-tier mode to run separate passes for fast and quality models.
///
/// # Errors
///
/// Returns an error if embedding generation or storage fails.
#[allow(clippy::too_many_lines)]
fn generate_embeddings_for_model(
    storage: &Storage,
    config: &EmbeddingConfig,
    model_id: &str,
) -> Result<()> {
    use crate::canonicalize::{canonicalize_for_embedding, content_hash};
    use crate::embedder::Embedder;
    use crate::hash_embedder::HashEmbedder;
    use crate::model_registry::{EmbedderConfig as RegistryConfig, ModelRegistry};
    use colored::Colorize;
    use indicatif::{ProgressBar, ProgressStyle};
    use rayon::prelude::*;
    use std::collections::{HashMap, HashSet};
    use std::time::Instant;

    type EmbedRecord = (String, String, Vec<f32>, Option<[u8; 32]>);
    const EMBED_CHUNK_SIZE: usize = 1000;
    let embed_start = Instant::now();

    let registry = ModelRegistry::new();
    let embedder_box: Box<dyn Embedder> = if let Some(model) = &config.model {
        let mut cfg = RegistryConfig::new(model);
        cfg.dimensions = config.dimensions;
        cfg.show_progress = config.show_progress;
        registry
            .embedder(&cfg)
            .map_err(|e| anyhow::anyhow!("Failed to load embedder {model}: {e}"))?
    } else if config.use_semantic {
        let mut cfg = RegistryConfig::new("all-MiniLM-L6-v2");
        cfg.dimensions = config.dimensions;
        cfg.show_progress = config.show_progress;
        registry
            .embedder(&cfg)
            .map_err(|e| anyhow::anyhow!("Failed to load semantic model: {e}"))?
    } else {
        Box::new(HashEmbedder::default())
    };
    let embedder: &dyn Embedder = embedder_box.as_ref();
    let store_batch_size = if embedder.is_semantic() { 50 } else { 100 };

    // Fetch all documents
    let tweets = storage.get_all_tweets(None)?;
    let likes = storage.get_all_likes(None)?;
    let dms = storage.get_all_dms(None)?;
    let grok_msgs = storage.get_all_grok_messages(None)?;

    let capacity = tweets.len() + likes.len() + dms.len() + grok_msgs.len();
    let mut docs: Vec<(String, String, &'static str)> = Vec::with_capacity(capacity);

    for tweet in &tweets {
        docs.push((tweet.id.clone(), tweet.full_text.clone(), "tweet"));
    }
    for like in &likes {
        if let Some(ref text) = like.full_text {
            if !text.is_empty() {
                docs.push((like.tweet_id.clone(), text.clone(), "like"));
            }
        }
    }
    for dm in &dms {
        if !dm.text.is_empty() {
            docs.push((dm.id.clone(), dm.text.clone(), "dm"));
        }
    }
    for msg in &grok_msgs {
        if !msg.message.is_empty() {
            let doc_id = format!(
                "{}_{}_{}_{}",
                msg.chat_id,
                msg.created_at.timestamp(),
                msg.created_at.timestamp_subsec_nanos(),
                msg.sender
            );
            docs.push((doc_id, msg.message.clone(), "grok"));
        }
    }

    if docs.is_empty() {
        if config.show_progress {
            println!("    {} No documents to embed", "⚠".yellow());
        }
        return Ok(());
    }

    // Load hashes scoped to this model_id to avoid cross-model dedup collisions.
    // Without filtering, the quality pass would skip all docs because the fast
    // pass already stored the same content hashes (different model, same text).
    let existing_hashes_by_doc = storage.load_embedding_hashes_by_doc_for_model(model_id)?;
    let mut existing_hashes: HashSet<[u8; 32]> = HashSet::new();
    for by_type in existing_hashes_by_doc.values() {
        for hash_val in by_type.values() {
            existing_hashes.insert(*hash_val);
        }
    }

    let pb = if config.show_progress {
        let pb = ProgressBar::new(docs.len() as u64);
        let style = ProgressStyle::default_bar()
            .template("    {spinner:.green} [{bar:40.cyan/blue}] {pos}/{len} ({per_sec:.cyan} docs/sec) [{elapsed_precise}] ETA: {eta}")
            .unwrap_or_else(|_| ProgressStyle::default_bar())
            .progress_chars("█▓░");
        pb.set_style(style);
        Some(pb)
    } else {
        None
    };

    let mut stored_count = 0;
    let mut reused_count = 0;
    let mut skipped_count = 0;

    for chunk in docs.chunks(EMBED_CHUNK_SIZE) {
        let mut batch: Vec<EmbedRecord> = Vec::new();
        let mut candidates: Vec<(String, &'static str, String, [u8; 32])> = Vec::new();

        for (doc_id, text, doc_type) in chunk {
            let canonical = canonicalize_for_embedding(text);
            if canonical.is_empty() {
                skipped_count += 1;
                if let Some(ref pb) = pb {
                    pb.inc(1);
                }
                continue;
            }
            let hash = content_hash(&canonical);
            if let Some(existing_hash) = existing_hashes_by_doc
                .get(doc_id)
                .and_then(|by_type| by_type.get(*doc_type))
            {
                if existing_hash == &hash {
                    skipped_count += 1;
                    if let Some(ref pb) = pb {
                        pb.inc(1);
                    }
                    continue;
                }
            }
            candidates.push((doc_id.clone(), *doc_type, canonical, hash));
            if let Some(ref pb) = pb {
                pb.inc(1);
            }
        }

        let mut batch_cache: HashMap<[u8; 32], Vec<f32>> = HashMap::new();
        let mut needed_hashes: Vec<[u8; 32]> = Vec::new();
        let mut needed_hashes_set: HashSet<[u8; 32]> = HashSet::new();
        for (_, _, _, hash) in &candidates {
            if existing_hashes.contains(hash) && needed_hashes_set.insert(*hash) {
                needed_hashes.push(*hash);
            }
        }

        if !needed_hashes.is_empty() {
            let fetched = storage.load_embeddings_by_hashes_for_model(&needed_hashes, model_id)?;
            for (hash, embedding) in fetched {
                batch_cache.insert(hash, embedding);
            }
        }

        let mut new_hashes: Vec<(String, String, [u8; 32])> = Vec::new();
        let mut new_hashes_set: HashSet<[u8; 32]> = HashSet::new();
        for (doc_id, _doc_type, canonical, hash) in &candidates {
            if batch_cache.contains_key(hash) {
                continue;
            }
            if new_hashes_set.insert(*hash) {
                new_hashes.push((doc_id.clone(), canonical.clone(), *hash));
            }
        }

        if !new_hashes.is_empty() {
            let computed_embeddings: Vec<([u8; 32], Vec<f32>)> = new_hashes
                .par_iter()
                .filter_map(
                    |(doc_id, canonical, hash)| match embedder.embed(canonical) {
                        Ok(embedding) => Some((*hash, embedding)),
                        Err(e) => {
                            tracing::warn!("Failed to embed doc {}: {}", doc_id, e);
                            None
                        }
                    },
                )
                .collect();

            for (hash, embedding) in computed_embeddings {
                batch_cache.insert(hash, embedding);
            }
        }

        let mut seen_new_hashes: HashSet<[u8; 32]> = HashSet::new();
        for (doc_id, doc_type, _canonical, hash) in candidates {
            if let Some(existing_embedding) = batch_cache.get(&hash) {
                batch.push((
                    doc_id,
                    doc_type.to_string(),
                    existing_embedding.clone(),
                    Some(hash),
                ));

                if existing_hashes.contains(&hash) || !seen_new_hashes.insert(hash) {
                    reused_count += 1;
                }
            } else {
                skipped_count += 1;
            }
        }

        if !batch.is_empty() {
            for chunk in batch.chunks(store_batch_size) {
                storage.store_embeddings_batch_with_model(chunk, model_id)?;
                stored_count += chunk.len();
            }
        }
    }

    if let Some(pb) = pb {
        pb.finish_and_clear();
    }

    let embed_elapsed_secs = embed_start.elapsed().as_secs_f64();
    let embed_elapsed = format_duration(embed_start.elapsed());
    let generated_count = stored_count.saturating_sub(reused_count);
    let docs_per_sec = if embed_elapsed_secs > 0.0 {
        (stored_count as f64 / embed_elapsed_secs) as u64
    } else {
        0
    };
    if config.show_progress {
        println!(
            "    {} {} embeddings stored ({} docs/sec) {}",
            "✓".green(),
            format_number_usize(stored_count).bold(),
            format_number_u64(docs_per_sec).cyan(),
            format!("({embed_elapsed})").dimmed()
        );
        if reused_count > 0 {
            println!(
                "    {} {} reused from identical content",
                "·".dimmed(),
                format_number_usize(reused_count).dimmed()
            );
        }
        if generated_count > 0 && reused_count > 0 {
            println!(
                "    {} {} generated",
                "·".dimmed(),
                format_number_usize(generated_count).dimmed()
            );
        }
        if skipped_count > 0 {
            println!(
                "    {} {} skipped (empty or unchanged)",
                "·".dimmed(),
                format_number_usize(skipped_count).dimmed()
            );
        }
    }

    Ok(())
}

/// Escape text for CSV by sanitizing newlines and quotes.
#[must_use]
pub fn csv_escape_text(text: &str) -> String {
    text.replace('"', "\"\"").replace(['\n', '\r'], " ")
}

/// Format a long identifier as a short token (e.g., 1234...6789).
#[must_use]
pub fn format_short_id(id: &str) -> String {
    let chars: Vec<char> = id.chars().collect();
    if chars.len() <= 10 {
        return id.to_string();
    }
    let start: String = chars.iter().take(4).collect();
    let end: String = chars.iter().rev().take(4).rev().collect();
    format!("{start}...{end}")
}

/// Format bytes into a human-friendly string.
#[must_use]
pub fn format_bytes(bytes: u64) -> String {
    if bytes < BYTES_PER_KB {
        format!("{bytes} B")
    } else if bytes < BYTES_PER_MB {
        format_bytes_with_unit(bytes, BYTES_PER_KB, "KB")
    } else if bytes < BYTES_PER_GB {
        format_bytes_with_unit(bytes, BYTES_PER_MB, "MB")
    } else {
        format_bytes_with_unit(bytes, BYTES_PER_GB, "GB")
    }
}

/// Format bytes for signed input, clamping negatives to zero.
#[must_use]
pub fn format_bytes_i64(bytes: i64) -> String {
    let bytes = u64::try_from(bytes.max(0)).unwrap_or(0);
    format_bytes(bytes)
}

fn format_bytes_with_unit(bytes: u64, unit: u64, suffix: &str) -> String {
    let whole = bytes / unit;
    let tenths = (bytes % unit) * 10 / unit;
    format!("{whole}.{tenths} {suffix}")
}

#[cfg(test)]
mod tests {
    use super::{
        Like, Storage, Tweet, TweetUrl, csv_escape_text, format_bytes_i64, format_duration,
        format_number, format_relative_date_with_base, format_short_id, generate_embeddings,
    };
    use crate::canonicalize::{canonicalize_for_embedding, content_hash};
    use crate::embedder::Embedder;
    use crate::hash_embedder::HashEmbedder;
    use crate::model::{DirectMessage, DmConversation, GrokMessage};
    use anyhow::Result;
    use chrono::{DateTime, Duration, TimeZone, Utc};
    use std::time::Duration as StdDuration;

    type EmbedBatch = Vec<(String, String, Vec<f32>, Option<[u8; 32]>)>;

    #[test]
    fn format_number_adds_separators() {
        assert_eq!(format_number(0), "0");
        assert_eq!(format_number(999), "999");
        assert_eq!(format_number(1000), "1,000");
        assert_eq!(format_number(12_345_678), "12,345,678");
        assert_eq!(format_number(-12_345), "-12,345");
    }

    #[test]
    fn format_relative_date_thresholds() {
        let base = Utc
            .with_ymd_and_hms(2025, 1, 10, 12, 0, 0)
            .single()
            .unwrap();

        assert_eq!(
            format_relative_date_with_base(base - Duration::seconds(30), base),
            "just now"
        );
        assert_eq!(
            format_relative_date_with_base(base - Duration::minutes(5), base),
            "5m ago"
        );
        assert_eq!(
            format_relative_date_with_base(base - Duration::hours(3), base),
            "3h ago"
        );
        assert_eq!(
            format_relative_date_with_base(base - Duration::days(2), base),
            "2d ago"
        );

        let same_year = Utc.with_ymd_and_hms(2025, 1, 1, 0, 0, 0).single().unwrap();
        assert_eq!(format_relative_date_with_base(same_year, base), "Jan 01");

        let different_year = Utc
            .with_ymd_and_hms(2024, 12, 11, 0, 0, 0)
            .single()
            .unwrap();
        assert_eq!(
            format_relative_date_with_base(different_year, base),
            "Dec 11, 2024"
        );

        let future = base + Duration::days(2);
        assert_eq!(
            format_relative_date_with_base(future, base),
            future.format("%b %d, %Y").to_string()
        );
    }

    #[test]
    fn csv_escape_text_sanitizes_newlines_and_quotes() {
        let input = "Hello\r\n\"world\", ok";
        let escaped = csv_escape_text(input);
        assert_eq!(escaped, "Hello  \"\"world\"\", ok");
    }

    #[test]
    fn format_duration_thresholds() {
        assert_eq!(format_duration(StdDuration::from_millis(120)), "120.0ms");
        assert_eq!(format_duration(StdDuration::from_millis(1500)), "1.50s");
        assert_eq!(format_duration(StdDuration::from_secs(75)), "1m 15s");
        assert_eq!(format_duration(StdDuration::from_secs(7260)), "2h 01m");
    }

    #[test]
    fn format_short_id_truncates_long_ids() {
        assert_eq!(format_short_id("short"), "short");
        assert_eq!(format_short_id("1234567890123"), "1234...0123");
    }

    #[test]
    fn format_bytes_i64_clamps_negative() {
        assert_eq!(format_bytes_i64(-5), "0 B");
    }

    fn seed_storage(storage: &mut Storage, base: DateTime<Utc>) -> Result<()> {
        let tweet = Tweet {
            id: "t1".to_string(),
            created_at: base,
            full_text: "hello world".to_string(),
            source: None,
            favorite_count: 0,
            retweet_count: 0,
            lang: None,
            in_reply_to_status_id: None,
            in_reply_to_user_id: None,
            in_reply_to_screen_name: None,
            is_retweet: false,
            hashtags: Vec::new(),
            user_mentions: Vec::new(),
            urls: vec![TweetUrl {
                url: "https://example.com".to_string(),
                expanded_url: None,
                display_url: None,
            }],
            media: Vec::new(),
        };
        storage.store_tweets(&[tweet])?;

        let like = Like {
            tweet_id: "l1".to_string(),
            full_text: Some("liked tweet".to_string()),
            expanded_url: None,
        };
        storage.store_likes(&[like])?;

        let dm = DirectMessage {
            id: "dm1".to_string(),
            conversation_id: "conv1".to_string(),
            sender_id: "u1".to_string(),
            recipient_id: "u2".to_string(),
            text: "dm body".to_string(),
            created_at: base,
            urls: Vec::new(),
            media_urls: Vec::new(),
        };
        storage.store_dm_conversations(&[DmConversation {
            conversation_id: "conv1".to_string(),
            messages: vec![dm],
        }])?;

        let grok = GrokMessage {
            chat_id: "chat1".to_string(),
            message: "grok response".to_string(),
            sender: "user".to_string(),
            created_at: base,
            grok_mode: None,
        };
        storage.store_grok_messages(&[grok])?;

        Ok(())
    }

    fn generate_embeddings_sequential(storage: &Storage) -> Result<()> {
        let embedder = HashEmbedder::default();
        let mut batch: EmbedBatch = Vec::new();

        for tweet in storage.get_all_tweets(None)? {
            let canonical = canonicalize_for_embedding(&tweet.full_text);
            if canonical.is_empty() {
                continue;
            }
            let hash = content_hash(&canonical);
            let embedding = embedder.embed(&canonical)?;
            batch.push((tweet.id, "tweet".to_string(), embedding, Some(hash)));
        }

        for like in storage.get_all_likes(None)? {
            let Some(text) = like.full_text else {
                continue;
            };
            if text.is_empty() {
                continue;
            }
            let canonical = canonicalize_for_embedding(&text);
            if canonical.is_empty() {
                continue;
            }
            let hash = content_hash(&canonical);
            let embedding = embedder.embed(&canonical)?;
            batch.push((like.tweet_id, "like".to_string(), embedding, Some(hash)));
        }

        for dm in storage.get_all_dms(None)? {
            if dm.text.is_empty() {
                continue;
            }
            let canonical = canonicalize_for_embedding(&dm.text);
            if canonical.is_empty() {
                continue;
            }
            let hash = content_hash(&canonical);
            let embedding = embedder.embed(&canonical)?;
            batch.push((dm.id, "dm".to_string(), embedding, Some(hash)));
        }

        for msg in storage.get_all_grok_messages(None)? {
            if msg.message.is_empty() {
                continue;
            }
            let canonical = canonicalize_for_embedding(&msg.message);
            if canonical.is_empty() {
                continue;
            }
            let hash = content_hash(&canonical);
            let embedding = embedder.embed(&canonical)?;
            let doc_id = format!(
                "{}_{}_{}_{}",
                msg.chat_id,
                msg.created_at.timestamp(),
                msg.created_at.timestamp_subsec_nanos(),
                msg.sender
            );
            batch.push((doc_id, "grok".to_string(), embedding, Some(hash)));
        }

        if !batch.is_empty() {
            storage.store_embeddings_batch(&batch)?;
        }

        Ok(())
    }

    #[test]
    fn generate_embeddings_matches_sequential_output() -> Result<()> {
        let base = Utc
            .with_ymd_and_hms(2025, 1, 10, 12, 0, 0)
            .single()
            .unwrap();

        let mut storage_seq = Storage::open_memory()?;
        seed_storage(&mut storage_seq, base)?;
        generate_embeddings_sequential(&storage_seq)?;
        let mut seq = storage_seq.load_all_embeddings()?;

        let mut storage_par = Storage::open_memory()?;
        seed_storage(&mut storage_par, base)?;
        generate_embeddings(&storage_par, false)?;
        let mut par = storage_par.load_all_embeddings()?;

        seq.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
        par.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));

        assert_eq!(seq.len(), par.len());
        for ((id_a, ty_a, emb_a), (id_b, ty_b, emb_b)) in seq.iter().zip(par.iter()) {
            assert_eq!(id_a, id_b);
            assert_eq!(ty_a, ty_b);
            assert_eq!(emb_a.len(), emb_b.len());
            for (a, b) in emb_a.iter().zip(emb_b.iter()) {
                assert!((a - b).abs() < 1e-6);
            }
        }

        Ok(())
    }
}
