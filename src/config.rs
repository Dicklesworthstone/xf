//! Configuration system for xf.
//!
//! Provides layered configuration from multiple sources:
//!
//! 1. **Compiled defaults** - Sensible defaults built into the binary
//! 2. **User config file** - `~/.config/xf/config.toml`
//! 3. **Environment variables** - `XF_*` prefix
//! 4. **CLI arguments** - Highest priority, always wins
//!
//! # Example Configuration File
//!
//! ```toml
//! [paths]
//! db = "~/.local/share/xf/xf.db"
//! index = "~/.local/share/xf/xf_index"
//!
//! [search]
//! default_limit = 20
//! highlight = true
//!
//! [indexing]
//! parallel = true
//! buffer_size_mb = 256
//!
//! [output]
//! format = "text"
//! colors = true
//! ```

use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tracing::{debug, info, warn};

/// Main configuration structure for xf.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Path-related configuration.
    pub paths: PathsConfig,
    /// Search behavior configuration.
    pub search: SearchConfig,
    /// Indexing behavior configuration.
    pub indexing: IndexingConfig,
    /// Output formatting configuration.
    pub output: OutputConfig,
    /// Semantic search configuration.
    pub semantic: SemanticConfig,
}

/// Path configuration for database and index locations.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PathsConfig {
    /// Path to the `SQLite` database file.
    /// Environment variable: `XF_DB`
    pub db: Option<PathBuf>,

    /// Path to the Tantivy search index directory.
    /// Environment variable: `XF_INDEX`
    pub index: Option<PathBuf>,

    /// Default archive path (for repeated indexing).
    pub archive: Option<PathBuf>,
}

/// Search behavior configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SearchConfig {
    /// Default number of results to return.
    /// Environment variable: `XF_LIMIT`
    pub default_limit: usize,

    /// Enable search result highlighting.
    pub highlight: bool,

    /// Enable fuzzy matching for typo tolerance.
    pub fuzzy: bool,

    /// Minimum score threshold for results (0.0 - 1.0).
    pub min_score: f32,

    /// Cache size for search results (number of queries).
    pub cache_size: usize,
}

/// Indexing behavior configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct IndexingConfig {
    /// Enable parallel parsing (uses all CPU cores).
    pub parallel: bool,

    /// Memory buffer size for indexing (in MB).
    pub buffer_size_mb: usize,

    /// Number of threads for parallel operations (0 = auto).
    pub threads: usize,

    /// Skip specific data types during indexing.
    pub skip_types: Vec<String>,
}

/// Output formatting configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct OutputConfig {
    /// Default output format: text, json, json-pretty, compact, csv.
    pub format: String,

    /// Enable colored output.
    pub colors: bool,

    /// Suppress non-essential output (progress bars, etc.).
    pub quiet: bool,

    /// Show timing information for operations.
    pub timings: bool,
}

/// Semantic search configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SemanticConfig {
    /// Default embedding model.
    /// Environment variable: `XF_MODEL`
    pub model: Option<String>,

    /// Default MRL dimensions (256, 512, 768, or 1024).
    /// Environment variable: `XF_DIMS`
    pub dimensions: Option<usize>,

    /// Default reranker model.
    /// Environment variable: `XF_RERANKER`
    pub reranker: Option<String>,

    /// Enable reranking by default.
    pub rerank: bool,

    /// Use daemon for model inference by default.
    pub daemon: bool,

    /// Number of candidates to rerank.
    pub rerank_top: usize,
}

impl Default for SearchConfig {
    fn default() -> Self {
        Self {
            default_limit: 20,
            highlight: true,
            fuzzy: false,
            min_score: 0.0,
            cache_size: 1000,
        }
    }
}

impl Default for IndexingConfig {
    fn default() -> Self {
        Self {
            parallel: true,
            buffer_size_mb: 256,
            threads: 0, // Auto-detect
            skip_types: vec![],
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            format: "text".to_string(),
            colors: true,
            quiet: false,
            timings: false,
        }
    }
}

impl Default for SemanticConfig {
    fn default() -> Self {
        Self {
            model: None,
            dimensions: None,
            reranker: None,
            rerank: false,
            daemon: true,
            rerank_top: 100,
        }
    }
}

impl SemanticConfig {
    /// Resolve the effective embedding model name.
    ///
    /// Priority: CLI argument > config file/env > None (use runtime default).
    #[must_use]
    pub fn effective_model<'a>(&'a self, cli_override: Option<&'a str>) -> Option<&'a str> {
        cli_override.or(self.model.as_deref())
    }

    /// Resolve the effective MRL dimensions.
    ///
    /// Priority: CLI argument > config file/env > None (use model native).
    #[must_use]
    pub fn effective_dimensions(&self, cli_override: Option<usize>) -> Option<usize> {
        cli_override.or(self.dimensions)
    }

    /// Resolve the effective reranker model name.
    ///
    /// Priority: CLI argument > config file/env > None (no reranking).
    #[must_use]
    pub fn effective_reranker<'a>(&'a self, cli_override: Option<&'a str>) -> Option<&'a str> {
        cli_override.or(self.reranker.as_deref())
    }

    /// Whether reranking is enabled (CLI flag or config).
    #[must_use]
    pub const fn is_rerank_enabled(&self, cli_rerank: bool) -> bool {
        cli_rerank || self.rerank
    }
}

impl Config {
    /// Load configuration from all sources.
    ///
    /// Priority (highest to lowest):
    /// 1. Environment variables
    /// 2. User config file (~/.config/xf/config.toml)
    /// 3. Compiled defaults
    pub fn load() -> Self {
        let mut config = Self::default();

        // Load from user config file
        if let Some(user_config) = Self::load_user_config() {
            config.merge(user_config);
        }

        // Override from environment variables
        config.apply_env_overrides();

        // Expand ~ in configured paths after all overrides are applied
        config.expand_tilde_paths();

        debug!("Configuration loaded: {:?}", config);
        config
    }

    /// Load configuration from a specific file.
    pub fn load_from_file(path: &PathBuf) -> Option<Self> {
        if !path.exists() {
            debug!("Config file not found: {}", path.display());
            return None;
        }

        match std::fs::read_to_string(path) {
            Ok(content) => match toml::from_str::<Self>(&content) {
                Ok(mut config) => {
                    config.expand_tilde_paths();
                    info!("Loaded config from: {}", path.display());
                    Some(config)
                }
                Err(e) => {
                    warn!("Failed to parse config file {}: {}", path.display(), e);
                    None
                }
            },
            Err(e) => {
                warn!("Failed to read config file {}: {}", path.display(), e);
                None
            }
        }
    }

    /// Load the user configuration file from the standard location.
    fn load_user_config() -> Option<Self> {
        let config_path = Self::user_config_path()?;
        Self::load_from_file(&config_path)
    }

    /// Get the path to the user configuration file.
    #[must_use]
    pub fn user_config_path() -> Option<PathBuf> {
        dirs::config_dir().map(|p| p.join("xf").join("config.toml"))
    }

    /// Apply environment variable overrides.
    fn apply_env_overrides(&mut self) {
        // Path overrides
        if let Ok(db) = std::env::var("XF_DB") {
            self.paths.db = Some(PathBuf::from(db));
        }
        if let Ok(index) = std::env::var("XF_INDEX") {
            self.paths.index = Some(PathBuf::from(index));
        }
        if let Ok(archive) = std::env::var("XF_ARCHIVE") {
            self.paths.archive = Some(PathBuf::from(archive));
        }

        // Search overrides
        if let Ok(limit) = std::env::var("XF_LIMIT") {
            if let Ok(n) = limit.parse() {
                self.search.default_limit = n;
            }
        }

        // Output overrides
        if let Ok(format) = std::env::var("XF_FORMAT") {
            self.output.format = format;
        }
        if std::env::var("XF_NO_COLOR").is_ok() || std::env::var("NO_COLOR").is_ok() {
            self.output.colors = false;
        }
        if std::env::var("XF_QUIET").is_ok() {
            self.output.quiet = true;
        }

        // Indexing overrides
        if let Ok(buffer) = std::env::var("XF_BUFFER_MB") {
            if let Ok(n) = buffer.parse() {
                self.indexing.buffer_size_mb = n;
            }
        }
        if let Ok(threads) = std::env::var("XF_THREADS") {
            if let Ok(n) = threads.parse() {
                self.indexing.threads = n;
            }
        }

        // Semantic overrides
        if let Ok(model) = std::env::var("XF_MODEL") {
            self.semantic.model = Some(model);
        }
        if let Ok(dims) = std::env::var("XF_DIMS") {
            if let Ok(n) = dims.parse() {
                self.semantic.dimensions = Some(n);
            }
        }
        if let Ok(reranker) = std::env::var("XF_RERANKER") {
            self.semantic.reranker = Some(reranker);
        }
        if std::env::var("XF_RERANK").is_ok() {
            self.semantic.rerank = true;
        }
        if let Ok(val) = std::env::var("XF_DAEMON") {
            self.semantic.daemon = val != "0" && val.to_lowercase() != "false";
        }
    }

    fn expand_tilde_paths(&mut self) {
        self.paths.db = self.paths.db.clone().map(expand_tilde_path);
        self.paths.index = self.paths.index.clone().map(expand_tilde_path);
        self.paths.archive = self.paths.archive.clone().map(expand_tilde_path);
    }

    /// Merge another config into this one (other takes precedence).
    fn merge(&mut self, other: Self) {
        // Paths
        if other.paths.db.is_some() {
            self.paths.db = other.paths.db;
        }
        if other.paths.index.is_some() {
            self.paths.index = other.paths.index;
        }
        if other.paths.archive.is_some() {
            self.paths.archive = other.paths.archive;
        }

        // Search (always override if present in other)
        self.search.default_limit = other.search.default_limit;
        self.search.highlight = other.search.highlight;
        self.search.fuzzy = other.search.fuzzy;
        self.search.min_score = other.search.min_score;
        self.search.cache_size = other.search.cache_size;

        // Indexing
        self.indexing.parallel = other.indexing.parallel;
        self.indexing.buffer_size_mb = other.indexing.buffer_size_mb;
        self.indexing.threads = other.indexing.threads;
        if !other.indexing.skip_types.is_empty() {
            self.indexing.skip_types = other.indexing.skip_types;
        }

        // Output
        self.output.format = other.output.format;
        self.output.colors = other.output.colors;
        self.output.quiet = other.output.quiet;
        self.output.timings = other.output.timings;

        // Semantic
        if other.semantic.model.is_some() {
            self.semantic.model = other.semantic.model;
        }
        if other.semantic.dimensions.is_some() {
            self.semantic.dimensions = other.semantic.dimensions;
        }
        if other.semantic.reranker.is_some() {
            self.semantic.reranker = other.semantic.reranker;
        }
        self.semantic.rerank = other.semantic.rerank;
        self.semantic.daemon = other.semantic.daemon;
        self.semantic.rerank_top = other.semantic.rerank_top;
    }

    /// Get the database path, using defaults if not configured.
    pub fn db_path(&self) -> PathBuf {
        self.paths.db.clone().unwrap_or_else(crate::default_db_path)
    }

    /// Get the index path, using defaults if not configured.
    pub fn index_path(&self) -> PathBuf {
        self.paths
            .index
            .clone()
            .unwrap_or_else(crate::default_index_path)
    }

    /// Save the current configuration to the user config file.
    ///
    /// # Errors
    ///
    /// Returns an error if the config directory cannot be determined,
    /// the parent directory cannot be created, or the file cannot be written.
    pub fn save(&self) -> std::io::Result<()> {
        let config_path = Self::user_config_path().ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "Could not determine config directory",
            )
        })?;

        // Create parent directory if needed
        if let Some(parent) = config_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let content = toml::to_string_pretty(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;

        std::fs::write(&config_path, content)?;
        info!("Saved config to: {}", config_path.display());
        Ok(())
    }

    /// Generate a default configuration file content.
    #[must_use]
    pub fn default_config_content() -> String {
        let config = Self::default();
        toml::to_string_pretty(&config).unwrap_or_default()
    }
}

fn expand_tilde_path(path: PathBuf) -> PathBuf {
    let path_str = path.to_string_lossy();
    if path_str == "~" {
        return dirs::home_dir().unwrap_or(path);
    }
    let rest = path_str
        .strip_prefix("~/")
        .or_else(|| path_str.strip_prefix("~\\"));

    match (rest, dirs::home_dir()) {
        (Some(suffix), Some(home)) => home.join(suffix),
        _ => path,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.search.default_limit, 20);
        assert!(config.indexing.parallel);
        assert!(config.output.colors);
    }

    #[test]
    fn test_config_serialization() {
        let config = Config::default();
        let toml = toml::to_string(&config).unwrap();
        let parsed: Config = toml::from_str(&toml).unwrap();
        assert_eq!(config.search.default_limit, parsed.search.default_limit);
    }

    #[test]
    fn test_config_merge() {
        let mut base = Config::default();
        let mut other = Config::default();
        other.search.default_limit = 50;
        other.paths.db = Some(PathBuf::from("/custom/path"));

        base.merge(other);

        assert_eq!(base.search.default_limit, 50);
        assert_eq!(base.paths.db, Some(PathBuf::from("/custom/path")));
    }

    #[test]
    fn test_default_config_content() {
        let content = Config::default_config_content();
        assert!(content.contains("[paths]"));
        assert!(content.contains("[search]"));
        assert!(content.contains("[indexing]"));
        assert!(content.contains("[output]"));
    }

    #[test]
    fn test_semantic_config_defaults() {
        let sc = SemanticConfig::default();
        assert!(sc.model.is_none());
        assert!(sc.dimensions.is_none());
        assert!(sc.reranker.is_none());
        assert!(!sc.rerank);
        assert!(sc.daemon);
        assert_eq!(sc.rerank_top, 100);
    }

    #[test]
    fn test_semantic_effective_model_cli_wins() {
        let sc = SemanticConfig {
            model: Some("config-model".to_string()),
            ..Default::default()
        };
        // CLI override should win
        assert_eq!(sc.effective_model(Some("cli-model")), Some("cli-model"));
    }

    #[test]
    fn test_semantic_effective_model_config_fallback() {
        let sc = SemanticConfig {
            model: Some("config-model".to_string()),
            ..Default::default()
        };
        // No CLI override → config value
        assert_eq!(sc.effective_model(None), Some("config-model"));
    }

    #[test]
    fn test_semantic_effective_model_none() {
        let sc = SemanticConfig::default();
        // No CLI, no config → None
        assert_eq!(sc.effective_model(None), None);
    }

    #[test]
    fn test_semantic_effective_dimensions() {
        let sc = SemanticConfig {
            dimensions: Some(512),
            ..Default::default()
        };
        assert_eq!(sc.effective_dimensions(Some(256)), Some(256)); // CLI wins
        assert_eq!(sc.effective_dimensions(None), Some(512)); // Config fallback
    }

    #[test]
    fn test_semantic_effective_reranker() {
        let sc = SemanticConfig {
            reranker: Some("flashrank-nano".to_string()),
            ..Default::default()
        };
        assert_eq!(sc.effective_reranker(Some("mxbai")), Some("mxbai")); // CLI wins
        assert_eq!(sc.effective_reranker(None), Some("flashrank-nano")); // Config fallback
    }

    #[test]
    fn test_semantic_is_rerank_enabled() {
        let sc_off = SemanticConfig::default();
        assert!(!sc_off.is_rerank_enabled(false));
        assert!(sc_off.is_rerank_enabled(true)); // CLI flag enables it

        let sc_on = SemanticConfig {
            rerank: true,
            ..Default::default()
        };
        assert!(sc_on.is_rerank_enabled(false)); // Config enables it
        assert!(sc_on.is_rerank_enabled(true)); // Both on
    }

    #[test]
    fn test_semantic_config_toml_roundtrip() {
        let sc = SemanticConfig {
            model: Some("all-MiniLM-L6-v2".to_string()),
            dimensions: Some(384),
            reranker: Some("flashrank-nano".to_string()),
            rerank: true,
            daemon: false,
            rerank_top: 50,
        };
        let toml_str = toml::to_string(&sc).unwrap();
        let parsed: SemanticConfig = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.model.as_deref(), Some("all-MiniLM-L6-v2"));
        assert_eq!(parsed.dimensions, Some(384));
        assert_eq!(parsed.reranker.as_deref(), Some("flashrank-nano"));
        assert!(parsed.rerank);
        assert!(!parsed.daemon);
        assert_eq!(parsed.rerank_top, 50);
    }

    #[test]
    fn test_semantic_config_partial_toml() {
        let toml_str = r#"
            model = "nomic-embed-text-v1.5"
            rerank = true
        "#;
        let parsed: SemanticConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(parsed.model.as_deref(), Some("nomic-embed-text-v1.5"));
        assert!(parsed.rerank);
        // Other fields should be defaults
        assert!(parsed.dimensions.is_none());
        assert!(parsed.reranker.is_none());
        assert!(parsed.daemon);
        assert_eq!(parsed.rerank_top, 100);
    }

    #[test]
    fn test_config_merge_semantic() {
        let mut base = Config::default();
        let mut other = Config::default();
        other.semantic.model = Some("custom-model".to_string());
        other.semantic.reranker = Some("custom-reranker".to_string());
        other.semantic.rerank = true;

        base.merge(other);

        assert_eq!(base.semantic.model.as_deref(), Some("custom-model"));
        assert_eq!(base.semantic.reranker.as_deref(), Some("custom-reranker"));
        assert!(base.semantic.rerank);
    }

    #[test]
    fn test_config_merge_semantic_none_doesnt_overwrite() {
        let mut base = Config::default();
        base.semantic.model = Some("existing-model".to_string());

        let other = Config::default(); // model is None
        base.merge(other);

        // None should not overwrite existing value
        assert_eq!(base.semantic.model.as_deref(), Some("existing-model"));
    }
}
