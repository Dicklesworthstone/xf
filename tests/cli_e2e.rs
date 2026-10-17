//! End-to-end CLI tests for xf.
//!
//! These tests run the actual xf binary and verify:
//! - Command-line interface behavior
//! - Output format and content
//! - Error handling and messages
//! - Integration between all components
//!
//! # Test Organization
//!
//! Tests are organized by command:
//! - `test_index_*` - Index command tests
//! - `test_search_*` - Search command tests
//! - `test_stats_*` - Stats command tests
//! - `test_cli_*` - General CLI tests (flags, help, version)
//!
//! # Logging
//!
//! All tests use detailed logging for debugging:
//! - Test start/end timestamps
//! - Command output capture
//! - Timing information

use assert_cmd::Command;
use assert_cmd::cargo::cargo_bin_cmd;
use predicates::prelude::*;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tempfile::TempDir;
use xf::canonicalize::canonicalize_for_embedding;
use xf::embedder::Embedder;
use xf::hash_embedder::HashEmbedder;
use xf::hybrid;
use xf::model::SearchResult;
use xf::search::{DocLookup, DocType, SearchEngine};
use xf::storage::Storage;
use xf::vector::VectorIndex;

// =============================================================================
// Test Utilities
// =============================================================================

/// Log a test event with timestamp
macro_rules! test_log {
    ($($arg:tt)*) => {
        let timestamp = chrono::Utc::now().format("%H:%M:%S%.3f");
        eprintln!("[TEST {}] {}", timestamp, format!($($arg)*));
    };
}

/// Create a test archive with the given data files
fn create_test_archive(
    tweets: Option<&str>,
    likes: Option<&str>,
    followers: Option<&str>,
    following: Option<&str>,
    dms: Option<&str>,
) -> (TempDir, PathBuf) {
    let temp_dir = TempDir::new().expect("Failed to create temp directory");
    let data_dir = temp_dir.path().join("data");
    fs::create_dir_all(&data_dir).expect("Failed to create data directory");

    if let Some(content) = tweets {
        fs::write(data_dir.join("tweets.js"), content).expect("Failed to write tweets.js");
    }
    if let Some(content) = likes {
        fs::write(data_dir.join("like.js"), content).expect("Failed to write like.js");
    }
    if let Some(content) = followers {
        fs::write(data_dir.join("follower.js"), content).expect("Failed to write follower.js");
    }
    if let Some(content) = following {
        fs::write(data_dir.join("following.js"), content).expect("Failed to write following.js");
    }
    if let Some(content) = dms {
        fs::write(data_dir.join("direct-messages.js"), content)
            .expect("Failed to write direct-messages.js");
    }
    fs::write(data_dir.join("manifest.js"), SAMPLE_MANIFEST).expect("Failed to write manifest.js");

    let archive_path = temp_dir.path().to_path_buf();
    (temp_dir, archive_path)
}

/// Create a minimal valid test archive
fn create_minimal_archive() -> (TempDir, PathBuf) {
    create_test_archive(
        Some(SAMPLE_TWEETS),
        Some(SAMPLE_LIKES),
        Some(SAMPLE_FOLLOWERS),
        Some(SAMPLE_FOLLOWING),
        None,
    )
}

/// Create an archive with Unicode content for edge case testing
fn create_unicode_archive() -> (TempDir, PathBuf) {
    create_test_archive(Some(SAMPLE_UNICODE_TWEETS), None, None, None, None)
}

/// Create an archive with empty data files
fn create_empty_archive() -> (TempDir, PathBuf) {
    create_test_archive(
        Some("window.YTD.tweets.part0 = []"),
        Some("window.YTD.like.part0 = []"),
        Some("window.YTD.follower.part0 = []"),
        Some("window.YTD.following.part0 = []"),
        None,
    )
}

/// Get the xf command ready for testing
fn xf_cmd() -> Command {
    cargo_bin_cmd!("xf")
}

fn parse_search_results(output: &std::process::Output) -> Vec<SearchResult> {
    assert!(
        output.status.success(),
        "xf search failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let missing_json_message = format!(
        "Expected JSON output, got: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json_start = stdout
        .char_indices()
        .find(|(_, ch)| *ch == '[' || *ch == '{')
        .map(|(idx, _)| idx)
        .expect(&missing_json_message);

    let json_slice = &stdout[json_start..];
    let parse_message = format!(
        "Failed to parse JSON output\nstdout: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_str(json_slice).expect(&parse_message)
}

fn parse_stats_json(output: &std::process::Output) -> Value {
    assert!(
        output.status.success(),
        "xf stats failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let missing_json_message = format!(
        "Expected JSON output, got: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let json_start = stdout
        .char_indices()
        .find(|(_, ch)| *ch == '{' || *ch == '[')
        .map(|(idx, _)| idx)
        .expect(&missing_json_message);

    let json_slice = &stdout[json_start..];
    let parse_message = format!(
        "Failed to parse JSON output\nstdout: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_str(json_slice).expect(&parse_message)
}

fn extract_stat_value(output: &str, label: &str) -> Option<String> {
    output.lines().find_map(|line| {
        let line = line.trim_end();
        line.find(label).map(|idx| {
            let value = line[idx + label.len()..].trim();
            value.replace(',', "")
        })
    })
}

fn format_id_scores(results: &[SearchResult]) -> String {
    results
        .iter()
        .map(|result| format!("{}:{:.6}", result.id, result.score))
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_id_scores_pairs(pairs: &[(String, f32)]) -> String {
    pairs
        .iter()
        .map(|(id, score)| format!("{id}:{score:.6}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn expected_semantic_results(
    db_path: &Path,
    index_path: &Path,
    query: &str,
    limit: usize,
    doc_types: &[DocType],
) -> Vec<SearchResult> {
    let storage = Storage::open(db_path).expect("Failed to open storage");
    let search_engine = SearchEngine::open(index_path).expect("Failed to open search index");
    let vector_index =
        VectorIndex::load_from_storage(&storage).expect("Failed to load vector index");

    let canonical_query = canonicalize_for_embedding(query);
    assert!(
        !canonical_query.is_empty(),
        "Canonical query should not be empty"
    );

    let embedder = HashEmbedder::default();
    let query_embedding = embedder
        .embed(&canonical_query)
        .expect("Failed to embed query");

    let type_strs: Vec<&str> = doc_types.iter().map(|t| t.as_str()).collect();
    let semantic_hits = vector_index.search_top_k(
        &query_embedding,
        limit.saturating_mul(hybrid::CANDIDATE_MULTIPLIER),
        Some(&type_strs),
    );

    let lookups: Vec<_> = semantic_hits
        .iter()
        .map(|hit| DocLookup::with_type(&hit.doc_id, hit.doc_type))
        .collect();
    let fetched = search_engine
        .get_by_ids(&lookups)
        .expect("Failed to fetch semantic results");

    let mut results = Vec::new();
    for (hit, result) in semantic_hits.into_iter().zip(fetched) {
        if let Some(mut result) = result {
            result.score = hit.score;
            results.push(result);
        }
    }

    if results.len() > limit {
        results.truncate(limit);
    }

    results
}

fn expected_hybrid_scores(
    db_path: &Path,
    index_path: &Path,
    query: &str,
    limit: usize,
    offset: usize,
    doc_types: &[DocType],
) -> Vec<(String, f32)> {
    let storage = Storage::open(db_path).expect("Failed to open storage");
    let search_engine = SearchEngine::open(index_path).expect("Failed to open search index");
    let vector_index =
        VectorIndex::load_from_storage(&storage).expect("Failed to load vector index");

    let canonical_query = canonicalize_for_embedding(query);
    let embedder = HashEmbedder::default();
    let candidate_count = hybrid::candidate_count(limit, offset);

    let lexical_results = search_engine
        .search(query, Some(doc_types), candidate_count)
        .expect("Failed to run lexical search");

    let semantic_results = if canonical_query.is_empty() {
        Vec::new()
    } else {
        let query_embedding = embedder
            .embed(&canonical_query)
            .expect("Failed to embed query");
        let type_strs: Vec<&str> = doc_types.iter().map(|t| t.as_str()).collect();
        vector_index.search_top_k(&query_embedding, candidate_count, Some(&type_strs))
    };

    let fused = hybrid::rrf_fuse(
        &lexical_results,
        &semantic_results,
        limit.saturating_add(offset),
        0,
    );

    fused
        .into_iter()
        .map(|hit| (hit.doc_id.to_string(), hit.score))
        .collect()
}

// =============================================================================
// Sample Test Data
// =============================================================================

const SAMPLE_TWEETS: &str = r#"window.YTD.tweets.part0 = [
    {
        "tweet": {
            "id_str": "1234567890123456789",
            "created_at": "Wed Jan 08 12:00:00 +0000 2025",
            "full_text": "Hello world! This is a test tweet about Rust programming. #rust #programming",
            "source": "<a href=\"https://x.com\">X Web App</a>",
            "favorite_count": "42",
            "retweet_count": "7",
            "lang": "en",
            "entities": {
                "hashtags": [{"text": "rust"}, {"text": "programming"}],
                "user_mentions": [],
                "urls": []
            }
        }
    },
    {
        "tweet": {
            "id_str": "1234567890123456790",
            "created_at": "Thu Jan 09 14:30:00 +0000 2025",
            "full_text": "Learning about Tantivy search engine. It's incredibly fast for full-text search!",
            "source": "<a href=\"https://x.com\">X Web App</a>",
            "favorite_count": "100",
            "retweet_count": "25",
            "lang": "en",
            "entities": {
                "hashtags": [{"text": "search"}, {"text": "tantivy"}],
                "user_mentions": [],
                "urls": []
            }
        }
    },
    {
        "tweet": {
            "id_str": "1234567890123456791",
            "created_at": "Fri Jan 10 09:15:00 +0000 2025",
            "full_text": "SQLite is an amazing embedded database. Perfect for local data storage.",
            "source": "<a href=\"https://x.com\">X Web App</a>",
            "favorite_count": "55",
            "retweet_count": "12",
            "lang": "en",
            "entities": {
                "hashtags": [{"text": "sqlite"}, {"text": "database"}],
                "user_mentions": [],
                "urls": []
            }
        }
    }
]"#;

const SAMPLE_MANIFEST: &str = r#"window.YTD.manifest.part0 = {
    "userInfo": {
        "accountId": "999999999",
        "userName": "test_user",
        "displayName": "Test User"
    },
    "archiveInfo": {
        "sizeBytes": "1234",
        "generationDate": "2025-01-01T00:00:00Z",
        "isPartialArchive": false
    }
}"#;

const SAMPLE_LIKES: &str = r#"window.YTD.like.part0 = [
    {
        "like": {
            "tweetId": "9876543210987654321",
            "fullText": "Great article about database optimization techniques and query performance",
            "expandedUrl": "https://example.com/article"
        }
    },
    {
        "like": {
            "tweetId": "9876543210987654322",
            "fullText": "Interesting thread on Rust async programming patterns",
            "expandedUrl": "https://example.com/thread"
        }
    }
]"#;

const SAMPLE_FOLLOWERS: &str = r#"window.YTD.follower.part0 = [
    {"follower": {"accountId": "111111111", "userLink": "https://x.com/user111"}},
    {"follower": {"accountId": "222222222", "userLink": "https://x.com/user222"}},
    {"follower": {"accountId": "333333333", "userLink": "https://x.com/user333"}}
]"#;

const SAMPLE_FOLLOWING: &str = r#"window.YTD.following.part0 = [
    {"following": {"accountId": "444444444", "userLink": "https://x.com/user444"}},
    {"following": {"accountId": "555555555", "userLink": "https://x.com/user555"}}
]"#;

const SAMPLE_UNICODE_TWEETS: &str = r#"window.YTD.tweets.part0 = [
    {
        "tweet": {
            "id_str": "1000000000000000001",
            "created_at": "Wed Jan 08 12:00:00 +0000 2025",
            "full_text": "Testing emoji support: 🦀 Rust is awesome! 🚀 Let's go!",
            "source": "<a href=\"https://x.com\">X Web App</a>",
            "favorite_count": "10",
            "retweet_count": "2",
            "lang": "en",
            "entities": {"hashtags": [], "user_mentions": [], "urls": []}
        }
    },
    {
        "tweet": {
            "id_str": "1000000000000000002",
            "created_at": "Thu Jan 09 14:30:00 +0000 2025",
            "full_text": "日本語のテストツイート。Unicode handling test!",
            "source": "<a href=\"https://x.com\">X Web App</a>",
            "favorite_count": "5",
            "retweet_count": "1",
            "lang": "ja",
            "entities": {"hashtags": [], "user_mentions": [], "urls": []}
        }
    },
    {
        "tweet": {
            "id_str": "1000000000000000003",
            "created_at": "Fri Jan 10 09:15:00 +0000 2025",
            "full_text": "Special chars: <>&\"' and newlines\nare handled correctly",
            "source": "<a href=\"https://x.com\">X Web App</a>",
            "favorite_count": "3",
            "retweet_count": "0",
            "lang": "en",
            "entities": {"hashtags": [], "user_mentions": [], "urls": []}
        }
    }
]"#;

// =============================================================================
// Help and Version Tests
// =============================================================================

#[test]
fn test_cli_help() {
    test_log!("Starting test_cli_help");
    let start = Instant::now();

    let mut cmd = xf_cmd();
    cmd.arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("xf"))
        .stdout(predicate::str::contains("Usage"));

    test_log!("test_cli_help completed in {:?}", start.elapsed());
}

#[test]
fn test_search_help_examples_and_types() {
    test_log!("Starting test_search_help_examples_and_types");
    let start = Instant::now();

    let mut cmd = xf_cmd();
    cmd.arg("search")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Examples:"))
        .stdout(predicate::str::contains("xf search \"hello world\""))
        .stdout(predicate::str::contains("Formats:"))
        .stdout(predicate::str::contains("tweet"))
        .stdout(predicate::str::contains("like"))
        .stdout(predicate::str::contains("dm"))
        .stdout(predicate::str::contains("grok"))
        .stdout(predicate::str::contains("follower").not())
        .stdout(predicate::str::contains("following").not())
        .stdout(predicate::str::contains("block").not())
        .stdout(predicate::str::contains("mute").not());

    test_log!(
        "test_search_help_examples_and_types completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_cli_version() {
    test_log!("Starting test_cli_version");
    let start = Instant::now();

    let mut cmd = xf_cmd();
    cmd.arg("--version")
        .assert()
        .success()
        .stdout(predicate::str::contains("xf"));

    test_log!("test_cli_version completed in {:?}", start.elapsed());
}

#[test]
fn test_cli_no_args() {
    test_log!("Starting test_cli_no_args");
    let start = Instant::now();

    let mut cmd = xf_cmd();
    // Running with no args should show help or error
    let output = cmd.output().expect("Failed to run command");

    // Either succeeds with help or fails with usage hint
    assert!(output.status.success() || !output.stderr.is_empty());

    test_log!("test_cli_no_args completed in {:?}", start.elapsed());
}

// =============================================================================
// Index Command Tests
// =============================================================================

#[test]
fn test_index_valid_archive() {
    test_log!("Starting test_index_valid_archive");
    let start = Instant::now();

    let (_temp_dir, archive_path) = create_minimal_archive();
    let output_dir = TempDir::new().expect("Failed to create output dir");
    let db_path = output_dir.path().join("test.db");
    let index_path = output_dir.path().join("test_index");

    test_log!("Archive path: {:?}", archive_path);
    test_log!("Database path: {:?}", db_path);
    test_log!("Index path: {:?}", index_path);

    let mut cmd = xf_cmd();
    cmd.arg("index")
        .arg(&archive_path)
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .success();

    // Verify database was created
    assert!(db_path.exists(), "Database file should exist");

    // Verify index directory was created
    assert!(index_path.exists(), "Index directory should exist");

    test_log!(
        "test_index_valid_archive completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_index_empty_archive() {
    test_log!("Starting test_index_empty_archive");
    let start = Instant::now();

    let (_temp_dir, archive_path) = create_empty_archive();
    let output_dir = TempDir::new().expect("Failed to create output dir");
    let db_path = output_dir.path().join("test.db");
    let index_path = output_dir.path().join("test_index");

    let mut cmd = xf_cmd();
    cmd.arg("index")
        .arg(&archive_path)
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .success();

    test_log!(
        "test_index_empty_archive completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_index_nonexistent_path() {
    test_log!("Starting test_index_nonexistent_path");
    let start = Instant::now();

    let mut cmd = xf_cmd();
    cmd.arg("index")
        .arg("/nonexistent/path/to/archive")
        .assert()
        .failure();

    test_log!(
        "test_index_nonexistent_path completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_index_unicode_content() {
    test_log!("Starting test_index_unicode_content");
    let start = Instant::now();

    let (_temp_dir, archive_path) = create_unicode_archive();
    let output_dir = TempDir::new().expect("Failed to create output dir");
    let db_path = output_dir.path().join("test.db");
    let index_path = output_dir.path().join("test_index");

    test_log!("Testing with Unicode content archive");

    let mut cmd = xf_cmd();
    cmd.arg("index")
        .arg(&archive_path)
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .success();

    test_log!(
        "test_index_unicode_content completed in {:?}",
        start.elapsed()
    );
}

// =============================================================================
// Search Command Tests
// =============================================================================

/// Helper to create an indexed archive and return paths.
/// Returns (`archive_temp`, `output_dir`, `db_path`, `index_path`).
/// Note: `archive_temp` must be kept alive to prevent cleanup during tests,
/// even though we've already indexed the data.
fn create_indexed_archive() -> (TempDir, TempDir, PathBuf, PathBuf) {
    let (archive_temp, archive_path) = create_minimal_archive();
    let output_dir = TempDir::new().expect("Failed to create output dir");
    let db_path = output_dir.path().join("test.db");
    let index_path = output_dir.path().join("test_index");

    // Index the archive
    let mut cmd = xf_cmd();
    cmd.arg("index")
        .arg(&archive_path)
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .success();

    // Return archive_temp to keep it alive (prevents early cleanup)
    (archive_temp, output_dir, db_path, index_path)
}

#[test]
fn test_search_basic_query() {
    test_log!("Starting test_search_basic_query");
    let start = Instant::now();

    let (_archive_temp, _output_dir, db_path, index_path) = create_indexed_archive();

    test_log!("Searching for 'rust'");

    let mut cmd = xf_cmd();
    cmd.arg("search")
        .arg("rust")
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .success()
        .stdout(predicate::str::contains("Rust").or(predicate::str::contains("rust")));

    test_log!("test_search_basic_query completed in {:?}", start.elapsed());
}

#[test]
fn test_search_no_results() {
    test_log!("Starting test_search_no_results");
    let start = Instant::now();

    let (_archive_temp, _output_dir, db_path, index_path) = create_indexed_archive();

    test_log!("Searching for 'xyznonexistent123'");

    let mut cmd = xf_cmd();
    cmd.arg("search")
        .arg("xyznonexistent123")
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .success();

    test_log!("test_search_no_results completed in {:?}", start.elapsed());
}

#[test]
fn test_search_with_limit() {
    test_log!("Starting test_search_with_limit");
    let start = Instant::now();

    let (_archive_temp, _output_dir, db_path, index_path) = create_indexed_archive();

    test_log!("Searching with limit 1");

    let mut cmd = xf_cmd();
    cmd.arg("search")
        .arg("tweet")
        .arg("--limit")
        .arg("1")
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .success();

    test_log!("test_search_with_limit completed in {:?}", start.elapsed());
}

#[test]
fn test_search_type_filter_tweets() {
    test_log!("Starting test_search_type_filter_tweets");
    let start = Instant::now();

    let (_archive_temp, _output_dir, db_path, index_path) = create_indexed_archive();

    test_log!("Searching with type filter: tweet");

    let mut cmd = xf_cmd();
    cmd.arg("search")
        .arg("rust")
        .arg("--types")
        .arg("tweet")
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .success();

    test_log!(
        "test_search_type_filter_tweets completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_search_type_filter_likes() {
    test_log!("Starting test_search_type_filter_likes");
    let start = Instant::now();

    let (_archive_temp, _output_dir, db_path, index_path) = create_indexed_archive();

    test_log!("Searching with type filter: like");

    let mut cmd = xf_cmd();
    cmd.arg("search")
        .arg("database")
        .arg("--types")
        .arg("like")
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .success();

    test_log!(
        "test_search_type_filter_likes completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_search_with_named_period_filters() {
    test_log!("Starting test_search_with_named_period_filters");
    let start = Instant::now();

    let (_archive_temp, _output_dir, db_path, index_path) = create_indexed_archive();

    test_log!("Searching with named period filters");

    let mut cmd = xf_cmd();
    cmd.arg("search")
        .arg("rust")
        .arg("--since")
        .arg("Jan 2025")
        .arg("--until")
        .arg("Jan 2025")
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .success()
        .stdout(predicate::str::contains("Rust").or(predicate::str::contains("rust")));

    test_log!(
        "test_search_with_named_period_filters completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_search_verbose_date_parse_output() {
    test_log!("Starting test_search_verbose_date_parse_output");
    let start = Instant::now();

    let (_archive_temp, _output_dir, db_path, index_path) = create_indexed_archive();

    test_log!("Searching with verbose date parsing output");

    let mut cmd = xf_cmd();
    cmd.arg("search")
        .arg("rust")
        .arg("--since")
        .arg("2025-01-09")
        .arg("--verbose")
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .success()
        .stderr(predicate::str::contains("Parsed --since '2025-01-09' as"));

    test_log!(
        "test_search_verbose_date_parse_output completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_search_invalid_date_expression() {
    test_log!("Starting test_search_invalid_date_expression");
    let start = Instant::now();

    let (_archive_temp, _output_dir, db_path, index_path) = create_indexed_archive();

    test_log!("Searching with invalid date expression");

    let mut cmd = xf_cmd();
    cmd.arg("search")
        .arg("rust")
        .arg("--since")
        .arg("notadate")
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("could not be parsed"));

    test_log!(
        "test_search_invalid_date_expression completed in {:?}",
        start.elapsed()
    );
}

// =============================================================================
// Stats Command Tests
// =============================================================================

#[test]
fn test_stats_after_indexing() {
    test_log!("Starting test_stats_after_indexing");
    let start = Instant::now();

    let (_archive_temp, _output_dir, db_path, index_path) = create_indexed_archive();

    test_log!("Getting stats");

    let mut cmd = xf_cmd();
    cmd.arg("stats")
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .success()
        // Should contain count information
        .stdout(
            predicate::str::contains("tweet")
                .or(predicate::str::contains("Tweet"))
                .or(predicate::str::contains("3")),
        );

    test_log!(
        "test_stats_after_indexing completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_stats_nonexistent_db() {
    test_log!("Starting test_stats_nonexistent_db");
    let start = Instant::now();

    let mut cmd = xf_cmd();
    cmd.arg("stats")
        .arg("--db")
        .arg("/nonexistent/path/to/db.db")
        .assert()
        .failure();

    test_log!(
        "test_stats_nonexistent_db completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_stats_text_output_counts() {
    test_log!("Starting test_stats_text_output_counts");
    let start = Instant::now();

    let (_archive_temp, _output_dir, db_path, index_path) = create_indexed_archive();

    let mut cmd = xf_cmd();
    let output = cmd
        .arg("stats")
        .arg("--no-color")
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .env("NO_COLOR", "1")
        .output()
        .expect("Failed to run stats command");

    assert!(
        output.status.success(),
        "xf stats failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let stdout = String::from_utf8_lossy(&output.stdout);
    let expectations = [
        ("Tweets:", "3"),
        ("Likes:", "2"),
        ("DM Conversations:", "0"),
        ("DM Messages:", "0"),
        ("Grok Messages:", "0"),
        ("Followers:", "3"),
        ("Following:", "2"),
        ("Blocks:", "0"),
        ("Mutes:", "0"),
    ];

    for (label, expected) in expectations {
        let value = extract_stat_value(&stdout, label)
            .unwrap_or_else(|| panic!("Missing stats line for label: {label}"));
        assert_eq!(value, expected, "Mismatch for {label}");
    }

    test_log!(
        "test_stats_text_output_counts completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_stats_json_output_counts() {
    test_log!("Starting test_stats_json_output_counts");
    let start = Instant::now();

    let (_archive_temp, _output_dir, db_path, index_path) = create_indexed_archive();

    let mut cmd = xf_cmd();
    let output = cmd
        .arg("stats")
        .arg("--format")
        .arg("json")
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .output()
        .expect("Failed to run stats command");

    let json = parse_stats_json(&output);
    assert_eq!(json["tweets_count"].as_i64(), Some(3));
    assert_eq!(json["likes_count"].as_i64(), Some(2));
    assert_eq!(json["dm_conversations_count"].as_i64(), Some(0));
    assert_eq!(json["dms_count"].as_i64(), Some(0));
    assert_eq!(json["grok_messages_count"].as_i64(), Some(0));
    assert_eq!(json["followers_count"].as_i64(), Some(3));
    assert_eq!(json["following_count"].as_i64(), Some(2));
    assert_eq!(json["blocks_count"].as_i64(), Some(0));
    assert_eq!(json["mutes_count"].as_i64(), Some(0));
    assert!(json["first_tweet_date"].is_string());
    assert!(json["last_tweet_date"].is_string());

    test_log!(
        "test_stats_json_output_counts completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_stats_json_detailed_output() {
    test_log!("Starting test_stats_json_detailed_output");
    let start = Instant::now();

    let (_archive_temp, _output_dir, db_path, index_path) = create_indexed_archive();

    let mut cmd = xf_cmd();
    let output = cmd
        .arg("stats")
        .arg("--format")
        .arg("json")
        .arg("--detailed")
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .output()
        .expect("Failed to run stats command");

    let json = parse_stats_json(&output);
    let stats = json
        .get("stats")
        .and_then(|value| value.as_object())
        .expect("Missing stats object in detailed output");
    assert_eq!(stats.get("tweets_count").and_then(Value::as_i64), Some(3));
    assert_eq!(stats.get("likes_count").and_then(Value::as_i64), Some(2));
    assert!(json.get("detailed").and_then(Value::as_array).is_some());

    test_log!(
        "test_stats_json_detailed_output completed in {:?}",
        start.elapsed()
    );
}

// =============================================================================
// Output Format Tests
// =============================================================================

#[test]
fn test_search_json_output() {
    test_log!("Starting test_search_json_output");
    let start = Instant::now();

    let (_archive_temp, _output_dir, db_path, index_path) = create_indexed_archive();

    test_log!("Searching with JSON output format");

    let mut cmd = xf_cmd();
    let output = cmd
        .arg("search")
        .arg("rust")
        .arg("--format")
        .arg("json")
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .output()
        .expect("Failed to run command");

    if output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        // JSON output should be valid JSON or empty
        if !stdout.trim().is_empty() {
            // Should at least have JSON-like structure
            assert!(
                stdout.contains('[') || stdout.contains('{'),
                "JSON output should contain JSON structure: {stdout}"
            );
        }
    }

    test_log!("test_search_json_output completed in {:?}", start.elapsed());
}

#[test]
fn test_search_semantic_score_semantics() {
    test_log!("Starting test_search_semantic_score_semantics");
    let start = Instant::now();

    let (_archive_temp, _output_dir, db_path, index_path) = create_indexed_archive();
    let query = "rust";
    let limit = 3usize;
    let doc_types = [DocType::Tweet];

    let mut cmd = xf_cmd();
    let output = cmd
        .arg("search")
        .arg(query)
        .arg("--mode")
        .arg("semantic")
        .arg("--types")
        .arg("tweet")
        .arg("--limit")
        .arg(limit.to_string())
        .arg("--format")
        .arg("json")
        .arg("--quiet") // Suppress INFO logs that pollute JSON output
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .output()
        .expect("Failed to run command");

    let results = parse_search_results(&output);
    let expected_results =
        expected_semantic_results(&db_path, &index_path, query, limit, &doc_types);

    assert!(
        !expected_results.is_empty(),
        "Expected semantic results to be non-empty"
    );

    let actual_ids: Vec<_> = results.iter().map(|r| r.id.clone()).collect();
    let expected_ids: Vec<_> = expected_results.iter().map(|r| r.id.clone()).collect();
    assert_eq!(
        actual_ids,
        expected_ids,
        "Semantic result ids mismatch\nactual: {}\nexpected: {}",
        format_id_scores(&results),
        format_id_scores(&expected_results)
    );

    for (idx, (actual, expected_result)) in results.iter().zip(expected_results.iter()).enumerate()
    {
        let delta = (actual.score - expected_result.score).abs();
        assert!(
            delta <= 1e-6,
            "Semantic score mismatch at idx {idx} id {}: actual {:.6} expected {:.6}\nactual: {}\nexpected: {}",
            actual.id,
            actual.score,
            expected_result.score,
            format_id_scores(&results),
            format_id_scores(&expected_results)
        );
    }

    test_log!(
        "test_search_semantic_score_semantics completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_search_hybrid_score_semantics() {
    test_log!("Starting test_search_hybrid_score_semantics");
    let start = Instant::now();

    let (_archive_temp, _output_dir, db_path, index_path) = create_indexed_archive();
    let query = "rust";
    let limit = 3usize;
    let offset = 0usize;
    let doc_types = [DocType::Tweet];

    let mut cmd = xf_cmd();
    let output = cmd
        .arg("search")
        .arg(query)
        .arg("--mode")
        .arg("hybrid")
        .arg("--types")
        .arg("tweet")
        .arg("--limit")
        .arg(limit.to_string())
        .arg("--format")
        .arg("json")
        .arg("--quiet") // Suppress INFO logs that pollute JSON output
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .output()
        .expect("Failed to run command");

    let results = parse_search_results(&output);
    let expected = expected_hybrid_scores(&db_path, &index_path, query, limit, offset, &doc_types);

    assert!(
        !expected.is_empty(),
        "Expected hybrid results to be non-empty"
    );

    let actual_pairs: Vec<(String, f32)> =
        results.iter().map(|r| (r.id.clone(), r.score)).collect();
    let actual_ids: Vec<_> = actual_pairs.iter().map(|(id, _)| id.clone()).collect();
    let expected_ids: Vec<_> = expected.iter().map(|(id, _)| id.clone()).collect();

    assert_eq!(
        actual_ids,
        expected_ids,
        "Hybrid result ids mismatch\nactual: {}\nexpected: {}",
        format_id_scores_pairs(&actual_pairs),
        format_id_scores_pairs(&expected)
    );

    for (idx, ((actual_id, actual_score), (_, expected_score))) in
        actual_pairs.iter().zip(expected.iter()).enumerate()
    {
        let delta = (*actual_score - *expected_score).abs();
        assert!(
            delta <= 1e-6,
            "Hybrid score mismatch at idx {idx} id {}: actual {:.6} expected {:.6}\nactual: {}\nexpected: {}",
            actual_id,
            actual_score,
            expected_score,
            format_id_scores_pairs(&actual_pairs),
            format_id_scores_pairs(&expected)
        );
    }

    test_log!(
        "test_search_hybrid_score_semantics completed in {:?}",
        start.elapsed()
    );
}

// =============================================================================
// Error Handling Tests
// =============================================================================

#[test]
fn test_invalid_command() {
    test_log!("Starting test_invalid_command");
    let start = Instant::now();

    let mut cmd = xf_cmd();
    cmd.arg("nonexistent_command")
        .assert()
        .failure()
        .stderr(predicate::str::contains("error").or(predicate::str::contains("unrecognized")));

    test_log!("test_invalid_command completed in {:?}", start.elapsed());
}

#[test]
fn test_missing_required_args() {
    test_log!("Starting test_missing_required_args");
    let start = Instant::now();

    // Index with non-existent archive path should fail
    let mut cmd = xf_cmd();
    cmd.arg("index")
        .arg("/nonexistent/path/to/archive")
        .assert()
        .failure();

    // Search without query should fail
    let mut cmd = xf_cmd();
    cmd.arg("search").assert().failure();

    test_log!(
        "test_missing_required_args completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_search_missing_index() {
    test_log!("Starting test_search_missing_index");
    let start = Instant::now();

    let output_dir = TempDir::new().expect("Failed to create output dir");
    let db_path = output_dir.path().join("test.db");
    let index_path = output_dir.path().join("missing_index");

    fs::write(&db_path, "").expect("Failed to create test db file");

    let mut cmd = xf_cmd();
    cmd.arg("search")
        .arg("rust")
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("Search index missing"));

    test_log!(
        "test_search_missing_index completed in {:?}",
        start.elapsed()
    );
}

// =============================================================================
// Quiet/Verbose Mode Tests
// =============================================================================

#[test]
fn test_quiet_mode() {
    test_log!("Starting test_quiet_mode");
    let start = Instant::now();

    let (_temp_dir, archive_path) = create_minimal_archive();
    let output_dir = TempDir::new().expect("Failed to create output dir");
    let db_path = output_dir.path().join("test.db");
    let index_path = output_dir.path().join("test_index");

    let mut cmd = xf_cmd();
    cmd.arg("index")
        .arg(&archive_path)
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .arg("--quiet")
        .assert()
        .success();

    test_log!("test_quiet_mode completed in {:?}", start.elapsed());
}

#[test]
fn test_verbose_mode() {
    test_log!("Starting test_verbose_mode");
    let start = Instant::now();

    let (_temp_dir, archive_path) = create_minimal_archive();
    let output_dir = TempDir::new().expect("Failed to create output dir");
    let db_path = output_dir.path().join("test.db");
    let index_path = output_dir.path().join("test_index");

    let mut cmd = xf_cmd();
    cmd.arg("index")
        .arg(&archive_path)
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .arg("--verbose")
        .assert()
        .success();

    test_log!("test_verbose_mode completed in {:?}", start.elapsed());
}

// =============================================================================
// Performance Tests (Basic)
// =============================================================================

#[test]
fn test_index_performance_basic() {
    test_log!("Starting test_index_performance_basic");
    let start = Instant::now();

    let (_temp_dir, archive_path) = create_minimal_archive();
    let output_dir = TempDir::new().expect("Failed to create output dir");
    let db_path = output_dir.path().join("test.db");
    let index_path = output_dir.path().join("test_index");

    let index_start = Instant::now();

    let mut cmd = xf_cmd();
    cmd.arg("index")
        .arg(&archive_path)
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .success();

    let index_time = index_start.elapsed();
    test_log!("Indexing took {:?}", index_time);

    // Basic performance check - indexing a small archive should be fast
    assert!(
        index_time.as_secs() < 30,
        "Indexing small archive took too long: {index_time:?}"
    );

    test_log!(
        "test_index_performance_basic completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_search_performance_basic() {
    test_log!("Starting test_search_performance_basic");
    let start = Instant::now();

    let (_archive_temp, _output_dir, db_path, index_path) = create_indexed_archive();

    let search_start = Instant::now();

    let mut cmd = xf_cmd();
    cmd.arg("search")
        .arg("rust")
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .success();

    let search_time = search_start.elapsed();
    test_log!("Search took {:?}", search_time);

    // Search should be very fast (sub-second)
    assert!(
        search_time.as_secs() < 5,
        "Search took too long: {search_time:?}"
    );

    test_log!(
        "test_search_performance_basic completed in {:?}",
        start.elapsed()
    );
}

// =============================================================================
// Doctor Command Tests (xf-11.4.6)
// =============================================================================

#[test]
fn test_doctor_with_valid_archive() {
    test_log!("Starting test_doctor_with_valid_archive");
    let start = Instant::now();

    let (_archive_temp, archive_path) = create_minimal_archive();
    let output_dir = TempDir::new().expect("Failed to create output dir");
    let db_path = output_dir.path().join("test.db");
    let index_path = output_dir.path().join("test_index");

    // First index the archive
    let mut cmd = xf_cmd();
    cmd.arg("index")
        .arg(&archive_path)
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .success();

    // Then run doctor with archive path
    let mut cmd = xf_cmd();
    cmd.arg("doctor")
        .arg("--archive")
        .arg(&archive_path)
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .success()
        .stdout(predicate::str::contains("Archive").or(predicate::str::contains("passed")));

    test_log!(
        "test_doctor_with_valid_archive completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_doctor_json_output() {
    test_log!("Starting test_doctor_json_output");
    let start = Instant::now();

    let (_archive_temp, archive_path) = create_minimal_archive();
    let output_dir = TempDir::new().expect("Failed to create output dir");
    let db_path = output_dir.path().join("test.db");
    let index_path = output_dir.path().join("test_index");

    // First index the archive
    let mut cmd = xf_cmd();
    cmd.arg("index")
        .arg(&archive_path)
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .success();

    // Run doctor with JSON output (use --quiet to suppress info logs)
    let mut cmd = xf_cmd();
    let output = cmd
        .arg("--quiet") // Suppress info logs that could pollute JSON output
        .arg("doctor")
        .arg("--archive")
        .arg(&archive_path)
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .arg("--format")
        .arg("json")
        .output()
        .expect("Failed to run command");

    assert!(output.status.success(), "Doctor command should succeed");

    let stdout = String::from_utf8_lossy(&output.stdout);
    // Validate JSON structure
    let json: serde_json::Value = serde_json::from_str(&stdout).unwrap_or_else(|e| {
        test_log!("Doctor JSON parse failed: {e}. Output: {stdout}");
        serde_json::Value::Null
    });
    assert!(
        json.is_object(),
        "Output should be valid JSON object. Output: {stdout}"
    );

    // Check expected fields in JSON output
    assert!(json.get("checks").is_some(), "Should have 'checks' field");
    assert!(json.get("summary").is_some(), "Should have 'summary' field");
    assert!(
        json.get("runtime_ms").is_some(),
        "Should have 'runtime_ms' field"
    );

    test_log!("test_doctor_json_output completed in {:?}", start.elapsed());
}

#[test]
fn test_doctor_without_archive() {
    test_log!("Starting test_doctor_without_archive");
    let start = Instant::now();

    let output_dir = TempDir::new().expect("Failed to create output dir");
    let db_path = output_dir.path().join("nonexistent.db");
    let index_path = output_dir.path().join("nonexistent_index");

    // Run doctor without archive or database - should warn but not crash
    let mut cmd = xf_cmd();
    cmd.arg("doctor")
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .success() // Should succeed even with warnings
        .stdout(predicate::str::contains("warning").or(predicate::str::contains("Warning")));

    test_log!(
        "test_doctor_without_archive completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_doctor_performance_check() {
    test_log!("Starting test_doctor_performance_check");
    let start = Instant::now();

    let (_archive_temp, archive_path) = create_minimal_archive();
    let output_dir = TempDir::new().expect("Failed to create output dir");
    let db_path = output_dir.path().join("test.db");
    let index_path = output_dir.path().join("test_index");

    // Index first
    let mut cmd = xf_cmd();
    cmd.arg("index")
        .arg(&archive_path)
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .success();

    // Run doctor and check performance benchmarks are included
    let mut cmd = xf_cmd();
    cmd.arg("doctor")
        .arg("--archive")
        .arg(&archive_path)
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .success()
        // Should include performance checks
        .stdout(predicate::str::contains("Performance").or(predicate::str::contains("Latency")));

    // Doctor should complete reasonably fast
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs() < 30,
        "Doctor command took too long: {elapsed:?}"
    );

    test_log!("test_doctor_performance_check completed in {:?}", elapsed);
}

// =============================================================================
// Shell Command Tests (xf-11.3.4)
// =============================================================================

#[test]
fn test_shell_help() {
    test_log!("Starting test_shell_help");
    let start = Instant::now();

    let mut cmd = xf_cmd();
    cmd.arg("shell")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("Launch interactive REPL mode"))
        .stdout(predicate::str::contains("--prompt"))
        .stdout(predicate::str::contains("--page-size"))
        .stdout(predicate::str::contains("--no-history"))
        .stdout(predicate::str::contains("--history-file"));

    test_log!("test_shell_help completed in {:?}", start.elapsed());
}

#[test]
fn test_shell_requires_database() {
    test_log!("Starting test_shell_requires_database");
    let start = Instant::now();

    let output_dir = TempDir::new().expect("Failed to create output dir");
    let db_path = output_dir.path().join("nonexistent.db");
    let index_path = output_dir.path().join("nonexistent_index");

    // Shell should fail when database doesn't exist
    let mut cmd = xf_cmd();
    cmd.arg("shell")
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("No archive indexed yet"));

    test_log!(
        "test_shell_requires_database completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_shell_requires_index() {
    test_log!("Starting test_shell_requires_index");
    let start = Instant::now();

    let output_dir = TempDir::new().expect("Failed to create output dir");
    let db_path = output_dir.path().join("test.db");
    let index_path = output_dir.path().join("nonexistent_index");

    // Create an empty db file
    fs::write(&db_path, "").expect("Failed to create test db file");

    // Shell should fail when index doesn't exist
    let mut cmd = xf_cmd();
    cmd.arg("shell")
        .arg("--db")
        .arg(&db_path)
        .arg("--index")
        .arg(&index_path)
        .assert()
        .failure()
        .stderr(predicate::str::contains("Search index missing"));

    test_log!(
        "test_shell_requires_index completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_shell_custom_prompt_parsing() {
    test_log!("Starting test_shell_custom_prompt_parsing");
    let start = Instant::now();

    // Verify that custom prompt option is parsed correctly via help
    let mut cmd = xf_cmd();
    cmd.arg("shell")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("--prompt"))
        .stdout(predicate::str::contains("Custom prompt string"));

    test_log!(
        "test_shell_custom_prompt_parsing completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_shell_page_size_parsing() {
    test_log!("Starting test_shell_page_size_parsing");
    let start = Instant::now();

    // Verify that page-size option is parsed correctly via help
    let mut cmd = xf_cmd();
    cmd.arg("shell")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("--page-size"))
        .stdout(predicate::str::contains("Number of results per page"));

    test_log!(
        "test_shell_page_size_parsing completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_shell_no_history_parsing() {
    test_log!("Starting test_shell_no_history_parsing");
    let start = Instant::now();

    // Verify that no-history option is parsed correctly via help
    let mut cmd = xf_cmd();
    cmd.arg("shell")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("--no-history"))
        .stdout(predicate::str::contains("Disable history file"));

    test_log!(
        "test_shell_no_history_parsing completed in {:?}",
        start.elapsed()
    );
}

#[test]
fn test_shell_history_file_parsing() {
    test_log!("Starting test_shell_history_file_parsing");
    let start = Instant::now();

    // Verify that history-file option is parsed correctly via help
    let mut cmd = xf_cmd();
    cmd.arg("shell")
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("--history-file"))
        .stdout(predicate::str::contains("Path to history file"));

    test_log!(
        "test_shell_history_file_parsing completed in {:?}",
        start.elapsed()
    );
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_machine_formats_never_suppressed_by_effective_quiet() {
    let (_archive_temp, _output_dir, db_path, index_path) = create_indexed_archive();

    // 1. Empty lexical search with --format json emits "[]"
    let out = xf_cmd()
        .args([
            "search",
            "xyznonexistent123",
            "--mode",
            "lexical",
            "--format",
            "json",
            "--db",
            db_path.to_str().unwrap(),
            "--index",
            index_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_eq!(String::from_utf8_lossy(&out.stdout).trim(), "[]");

    // 2. Non-empty search with --format csv emits header + rows
    let out = xf_cmd()
        .args([
            "search",
            "rust",
            "--format",
            "csv",
            "--db",
            db_path.to_str().unwrap(),
            "--index",
            index_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let csv_stdout = String::from_utf8_lossy(&out.stdout);
    assert!(csv_stdout.starts_with("type,id,created_at,score,text\n"));
    assert!(csv_stdout.lines().count() >= 2);

    // 3. Compact search preserves [tweet] bracket prefix
    let out = xf_cmd()
        .args([
            "search",
            "rust",
            "--format",
            "compact",
            "--db",
            db_path.to_str().unwrap(),
            "--index",
            index_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let compact_stdout = String::from_utf8_lossy(&out.stdout);
    assert!(compact_stdout.contains("[tweet]"));

    // 4. xf list tweets --format json emits valid JSON array
    let out = xf_cmd()
        .args([
            "list",
            "tweets",
            "--format",
            "json",
            "--db",
            db_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let list_json: Value = serde_json::from_slice(&out.stdout)
        .expect("xf list tweets --format json must be valid JSON");
    assert!(list_json.is_array());
    assert_ne!(list_json.as_array().unwrap().len(), 0);

    // 5. xf export tweets --format json and --format toon emit non-empty stdout
    let out = xf_cmd()
        .args([
            "export",
            "tweets",
            "--format",
            "json",
            "--db",
            db_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let export_json: Value = serde_json::from_slice(&out.stdout)
        .expect("xf export tweets --format json must be valid JSON");
    assert!(export_json.is_array());

    let out = xf_cmd()
        .args([
            "export",
            "tweets",
            "--format",
            "toon",
            "--db",
            db_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    assert_ne!(String::from_utf8_lossy(&out.stdout).trim(), "");

    // 6. xf config --show --format json emits valid JSON object
    let out = xf_cmd()
        .args([
            "config",
            "--show",
            "--format",
            "json",
            "--db",
            db_path.to_str().unwrap(),
            "--index",
            index_path.to_str().unwrap(),
        ])
        .output()
        .unwrap();
    assert!(out.status.success());
    let cfg_json: Value = serde_json::from_slice(&out.stdout)
        .expect("xf config --show --format json must be valid JSON");
    assert!(cfg_json.is_object());

    // 7. xf models list --format json emits valid JSON array
    let out = xf_cmd()
        .args(["models", "list", "--format", "json"])
        .output()
        .unwrap();
    assert!(out.status.success());
    let models_json: Value = serde_json::from_slice(&out.stdout)
        .expect("xf models list --format json must be valid JSON");
    assert!(models_json.is_array());
}

#[test]
fn test_search_fts5_fallback_when_tantivy_index_absent() {
    let (_archive_temp, output_dir, db_path, _index_path) = create_indexed_archive();
    let absent_index_path = output_dir.path().join("absent_tantivy_index");

    let output = xf_cmd()
        .args([
            "search",
            "rust",
            "--mode",
            "lexical",
            "--format",
            "json",
            "--quiet",
            "--db",
            db_path.to_str().unwrap(),
            "--index",
            absent_index_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run FTS5 fallback search");

    let results = parse_search_results(&output);
    assert!(
        !results.is_empty(),
        "SQLite FTS5 fallback should return results when Tantivy index is absent"
    );
    assert!(
        results
            .iter()
            .any(|r| r.text.to_lowercase().contains("rust"))
    );
}

#[test]
fn test_search_rerank_hash_reranker_and_date_sort() {
    let (_archive_temp, _output_dir, db_path, index_path) = create_indexed_archive();

    let output = xf_cmd()
        .args([
            "search",
            "rust",
            "--rerank",
            "--reranker",
            "hash-reranker",
            "--format",
            "json",
            "--quiet",
            "--db",
            db_path.to_str().unwrap(),
            "--index",
            index_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run reranked search");

    let results = parse_search_results(&output);
    assert!(!results.is_empty());
    assert!(
        results.iter().all(|r| r.rerank_score.is_some()),
        "Expected rerank_score to be populated when --rerank is set"
    );

    // Verify --sort date orders ascending by created_at
    let asc_out = xf_cmd()
        .args([
            "search",
            "tweet OR search OR database",
            "--mode",
            "lexical",
            "--types",
            "tweet",
            "--sort",
            "date",
            "--format",
            "json",
            "--quiet",
            "--db",
            db_path.to_str().unwrap(),
            "--index",
            index_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run date-sorted search");
    let asc_results = parse_search_results(&asc_out);
    assert!(asc_results.len() >= 2);
    for pair in asc_results.windows(2) {
        assert!(pair[0].created_at <= pair[1].created_at);
    }
}

#[test]
fn test_env_limit_and_format_overrides() {
    let (_archive_temp, _output_dir, db_path, index_path) = create_indexed_archive();

    let output = xf_cmd()
        .env("XF_LIMIT", "1")
        .env("XF_FORMAT", "json")
        .args([
            "search",
            "rust",
            "--quiet",
            "--db",
            db_path.to_str().unwrap(),
            "--index",
            index_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run search with XF_LIMIT and XF_FORMAT");

    let results = parse_search_results(&output);
    assert_eq!(results.len(), 1, "XF_LIMIT=1 should cap default limit to 1");
}

#[test]
fn test_doctor_fix_rebuilds_corrupted_vector_index() {
    let (archive_temp, _output_dir, db_path, index_path) = create_indexed_archive();
    let vector_file = index_path.join("vector.idx");
    fs::write(&vector_file, b"").expect("truncate vector.idx");

    let output = xf_cmd()
        .args([
            "doctor",
            "--fix",
            "--archive",
            archive_temp.path().to_str().unwrap(),
            "--format",
            "json",
            "--quiet",
            "--db",
            db_path.to_str().unwrap(),
            "--index",
            index_path.to_str().unwrap(),
        ])
        .output()
        .expect("Failed to run xf doctor --fix");

    assert!(
        output.status.success(),
        "xf doctor --fix failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let meta = fs::metadata(&vector_file).expect("vector.idx metadata");
    assert!(
        meta.len() >= 32,
        "Expected xf doctor --fix to rebuild vector.idx, got {} bytes",
        meta.len()
    );
}
