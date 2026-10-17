//! Machine-readable CLI documentation for agent consumption.
//!
//! Provides JSON output describing xf commands, flags, examples,
//! exit codes, and output formats. Used by `xf robot-docs <topic>`.

use serde::Serialize;
use serde_json::json;

/// Schema version for the robot-docs output format.
pub const SCHEMA_VERSION: &str = "1.0.0";

/// Top-level robot-docs response.
#[derive(Debug, Serialize)]
pub struct RobotDocs {
    pub version: String,
    pub schema_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub quickstart: Option<Quickstart>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub commands: Option<Vec<CommandDoc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub examples: Option<Vec<Example>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_codes: Option<Vec<ExitCode>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_formats: Option<Vec<OutputFormatDoc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub schemas: Option<Vec<SchemaDoc>>,
}

/// Quick-start guide for agents.
#[derive(Debug, Serialize)]
pub struct Quickstart {
    pub summary: String,
    pub steps: Vec<String>,
    pub tips: Vec<String>,
}

/// Documentation for a single CLI command.
#[derive(Debug, Serialize)]
pub struct CommandDoc {
    pub name: String,
    pub description: String,
    pub usage: String,
    pub flags: Vec<FlagDoc>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_formats: Option<Vec<String>>,
}

/// Documentation for a CLI flag.
#[derive(Debug, Serialize)]
pub struct FlagDoc {
    pub name: String,
    pub short: Option<String>,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub default: Option<String>,
    pub required: bool,
}

/// A usage example.
#[derive(Debug, Serialize)]
pub struct Example {
    pub description: String,
    pub command: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub notes: Option<String>,
}

/// Exit code documentation.
#[derive(Debug, Serialize)]
pub struct ExitCode {
    pub code: i32,
    pub meaning: String,
    pub commands: Vec<String>,
}

/// Output format documentation.
#[derive(Debug, Serialize)]
pub struct OutputFormatDoc {
    pub name: String,
    pub flag: String,
    pub description: String,
    pub mime_type: String,
}

/// JSON Schema documentation for a command's output format.
#[derive(Debug, Serialize)]
pub struct SchemaDoc {
    /// Command or output type name (e.g., "search", "stats", "doctor")
    pub name: String,
    /// Human-readable description of this output
    pub description: String,
    /// JSON Schema (draft-07) describing the output shape
    pub schema: serde_json::Value,
}

/// Available documentation topics.
pub const TOPICS: &[&str] = &[
    "all",
    "quickstart",
    "commands",
    "examples",
    "exit-codes",
    "output-formats",
    "schemas",
];

/// Generate robot-docs for the given topic.
#[must_use]
pub fn generate(topic: &str) -> RobotDocs {
    let all = topic == "all";
    RobotDocs {
        version: env!("CARGO_PKG_VERSION").to_string(),
        schema_version: SCHEMA_VERSION.to_string(),
        quickstart: if all || topic == "quickstart" {
            Some(quickstart())
        } else {
            None
        },
        commands: if all || topic == "commands" {
            Some(commands())
        } else {
            None
        },
        examples: if all || topic == "examples" {
            Some(examples())
        } else {
            None
        },
        exit_codes: if all || topic == "exit-codes" {
            Some(exit_codes())
        } else {
            None
        },
        output_formats: if all || topic == "output-formats" {
            Some(output_formats())
        } else {
            None
        },
        schemas: if all || topic == "schemas" {
            Some(schemas())
        } else {
            None
        },
    }
}

/// Check if a topic string is valid.
#[must_use]
pub fn is_valid_topic(topic: &str) -> bool {
    TOPICS.contains(&topic)
}

fn quickstart() -> Quickstart {
    Quickstart {
        summary: "xf indexes and searches your X (Twitter) data archive locally using full-text search, SQLite, and optional vector embeddings.".into(),
        steps: vec![
            "Download your data archive from x.com/settings/download_your_data".into(),
            "Import and index: xf import ~/Downloads/twitter-archive.zip".into(),
            "Search: xf search 'your query'".into(),
            "For existing extracted archives: xf index ~/path/to/archive".into(),
        ],
        tips: vec![
            "Use --format json for machine-readable output".into(),
            "Use --format toon for compact agent-friendly output".into(),
            "Use xf stats for archive overview".into(),
            "Use xf doctor to check archive health".into(),
            "Use xf shell for interactive REPL mode".into(),
        ],
    }
}

#[allow(clippy::too_many_lines)]
fn commands() -> Vec<CommandDoc> {
    let all_formats = || {
        Some(vec![
            "text".into(),
            "json".into(),
            "json-pretty".into(),
            "compact".into(),
            "csv".into(),
            "toon".into(),
            "jsonl".into(),
        ])
    };
    vec![
        CommandDoc {
            name: "import".into(),
            description: "Import and index an X data archive from a zip file".into(),
            usage: "xf import <ZIP_FILE> [OPTIONS]".into(),
            flags: vec![
                FlagDoc {
                    name: "--output".into(),
                    short: Some("-o".into()),
                    description: "Extract to this directory (default: ~/my_x_history)".into(),
                    default: Some("~/my_x_history".into()),
                    required: false,
                },
                FlagDoc {
                    name: "--no-index".into(),
                    short: None,
                    description: "Extract only, don't index".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--force".into(),
                    short: Some("-F".into()),
                    description: "Overwrite existing extraction".into(),
                    default: None,
                    required: false,
                },
            ],
            output_formats: None,
        },
        CommandDoc {
            name: "index".into(),
            description: "Index an extracted X data archive or .zip file for search".into(),
            usage: "xf index [ARCHIVE_PATH] [OPTIONS]".into(),
            flags: vec![
                FlagDoc {
                    name: "--force".into(),
                    short: Some("-F".into()),
                    description: "Rebuild index from scratch".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--only".into(),
                    short: None,
                    description: "Only index specific data types (tweet,like,dm,grok,bookmark,follower,following,block,mute)".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--skip".into(),
                    short: None,
                    description: "Skip specific data types".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--jobs".into(),
                    short: Some("-j".into()),
                    description: "Number of parallel worker threads".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--semantic".into(),
                    short: Some("-s".into()),
                    description: "Generate embeddings for semantic search".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--two-tier".into(),
                    short: None,
                    description: "Generate both fast and quality embeddings for progressive two-tier search".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--sync-embeddings".into(),
                    short: None,
                    description: "Generate embeddings synchronously instead of backgrounding to daemon".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--model".into(),
                    short: None,
                    description: "Embedding model override (default, minilm, potion-128M, mrl, bge-small, nomic, hash)".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--dimensions".into(),
                    short: None,
                    description: "MRL embedding dimension override".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--db".into(),
                    short: None,
                    description: "Path to SQLite database (global flag)".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--index".into(),
                    short: None,
                    description: "Path to search index directory (global flag)".into(),
                    default: None,
                    required: false,
                },
            ],
            output_formats: None,
        },
        CommandDoc {
            name: "search".into(),
            description: "Full-text, semantic, and hybrid search over indexed archive".into(),
            usage: "xf search <QUERY> [OPTIONS]".into(),
            flags: vec![
                FlagDoc {
                    name: "--limit".into(),
                    short: Some("-n".into()),
                    description: "Maximum results to return (env: XF_LIMIT)".into(),
                    default: Some("20".into()),
                    required: false,
                },
                FlagDoc {
                    name: "--types".into(),
                    short: Some("-t".into()),
                    description: "Filter by document type (tweet, like, dm, grok, bookmark, all)".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--since".into(),
                    short: None,
                    description: "Filter results after date (YYYY-MM-DD or natural language)".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--until".into(),
                    short: None,
                    description: "Filter results before date (YYYY-MM-DD or natural language)".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--mode".into(),
                    short: Some("-m".into()),
                    description: "Search mode: hybrid, lexical, or semantic".into(),
                    default: Some("hybrid".into()),
                    required: false,
                },
                FlagDoc {
                    name: "--replies-only".into(),
                    short: None,
                    description: "Only show replies".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--no-replies".into(),
                    short: None,
                    description: "Exclude replies".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--sort".into(),
                    short: None,
                    description: "Sort order (relevance, date, date-asc, engagement)".into(),
                    default: Some("relevance".into()),
                    required: false,
                },
                FlagDoc {
                    name: "--offset".into(),
                    short: None,
                    description: "Skip first N results (pagination)".into(),
                    default: Some("0".into()),
                    required: false,
                },
                FlagDoc {
                    name: "--context".into(),
                    short: Some("-C".into()),
                    description: "Return full DM conversation context for matching messages".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--rerank".into(),
                    short: None,
                    description: "Enable cross-encoder reranking of top candidates".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--reranker".into(),
                    short: None,
                    description: "Reranker model override (flashrank-nano, mxbai-xsmall, hash-reranker, none)".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--rerank-top".into(),
                    short: None,
                    description: "Number of top candidates to rerank".into(),
                    default: Some("20".into()),
                    required: false,
                },
                FlagDoc {
                    name: "--model".into(),
                    short: None,
                    description: "Embedding model override for query embedding".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--dimensions".into(),
                    short: None,
                    description: "MRL embedding dimension override".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--daemon".into(),
                    short: None,
                    description: "Route model inference through warm daemon".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--no-daemon".into(),
                    short: None,
                    description: "Disable daemon usage, force direct inference".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--fields".into(),
                    short: None,
                    description: "Select specific output fields (id,text,created_at,type,score,rerank_score,highlights,metadata)".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--fast-only".into(),
                    short: None,
                    description: "Only use fast tier in two-tier search".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--quality-only".into(),
                    short: None,
                    description: "Only use quality tier in two-tier search".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--blend-factor".into(),
                    short: None,
                    description: "Two-tier score blend factor (0.0=fast, 1.0=quality)".into(),
                    default: None,
                    required: false,
                },
            ],
            output_formats: all_formats(),
        },
        CommandDoc {
            name: "stats".into(),
            description: "Show archive statistics and analytics".into(),
            usage: "xf stats [OPTIONS]".into(),
            flags: vec![
                FlagDoc {
                    name: "--detailed".into(),
                    short: Some("-d".into()),
                    description: "Show detailed breakdown by time period and all analytics".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--temporal".into(),
                    short: None,
                    description: "Include temporal patterns".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--engagement".into(),
                    short: None,
                    description: "Include engagement analytics".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--content".into(),
                    short: None,
                    description: "Include content analysis".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--hashtags".into(),
                    short: None,
                    description: "Show top hashtags".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--mentions".into(),
                    short: None,
                    description: "Show top mentioned users".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--top".into(),
                    short: None,
                    description: "Limit for top lists".into(),
                    default: Some("10".into()),
                    required: false,
                },
            ],
            output_formats: all_formats(),
        },
        CommandDoc {
            name: "tweet".into(),
            description: "Show detailed information about a specific tweet".into(),
            usage: "xf tweet <ID> [OPTIONS]".into(),
            flags: vec![
                FlagDoc {
                    name: "--replies".into(),
                    short: Some("-r".into()),
                    description: "Include replies to this tweet".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--thread".into(),
                    short: Some("-t".into()),
                    description: "Show full conversation thread".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--engagement".into(),
                    short: Some("-e".into()),
                    description: "Include engagement details".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--raw".into(),
                    short: None,
                    description: "Show raw JSON from archive".into(),
                    default: None,
                    required: false,
                },
            ],
            output_formats: all_formats(),
        },
        CommandDoc {
            name: "list".into(),
            description: "List data from the archive (tweets, likes, bookmarks, dms, grok, conversations, followers, following, blocks, mutes, account, files, stats)".into(),
            usage: "xf list [TARGET] [OPTIONS]".into(),
            flags: vec![
                FlagDoc {
                    name: "--limit".into(),
                    short: Some("-n".into()),
                    description: "Maximum items to list".into(),
                    default: Some("20".into()),
                    required: false,
                },
                FlagDoc {
                    name: "--count".into(),
                    short: Some("-c".into()),
                    description: "Show counts only".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--files".into(),
                    short: None,
                    description: "Show archive files".into(),
                    default: None,
                    required: false,
                },
            ],
            output_formats: all_formats(),
        },
        CommandDoc {
            name: "export".into(),
            description: "Export data in various formats (tweets, likes, dms, bookmarks, grok, followers, following, blocks, mutes, all)".into(),
            usage: "xf export <TARGET> [OPTIONS]".into(),
            flags: vec![
                FlagDoc {
                    name: "--output".into(),
                    short: Some("-o".into()),
                    description: "Output file (stdout if not specified)".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--limit".into(),
                    short: Some("-n".into()),
                    description: "Maximum items to export".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--format".into(),
                    short: Some("-f".into()),
                    description: "Global output format; export writes json, jsonl, csv, or toon".into(),
                    default: Some("json".into()),
                    required: false,
                },
            ],
            output_formats: Some(vec![
                "json".into(),
                "jsonl".into(),
                "csv".into(),
                "toon".into(),
            ]),
        },
        CommandDoc {
            name: "config".into(),
            description: "Show or manage configuration".into(),
            usage: "xf config [OPTIONS]".into(),
            flags: vec![
                FlagDoc {
                    name: "--show".into(),
                    short: None,
                    description: "Show current configuration".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--path".into(),
                    short: None,
                    description: "Show config file path".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--archive".into(),
                    short: None,
                    description: "Set default archive path".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--set".into(),
                    short: None,
                    description: "Set a configuration value (KEY=VALUE)".into(),
                    default: None,
                    required: false,
                },
            ],
            output_formats: all_formats(),
        },
        CommandDoc {
            name: "update".into(),
            description: "Check for updates and show upgrade instructions".into(),
            usage: "xf update".into(),
            flags: vec![],
            output_formats: all_formats(),
        },
        CommandDoc {
            name: "completions".into(),
            description: "Generate shell completions (bash, zsh, fish, powershell, elvish)".into(),
            usage: "xf completions <SHELL>".into(),
            flags: vec![],
            output_formats: None,
        },
        CommandDoc {
            name: "doctor".into(),
            description: "Check archive, database, and index health".into(),
            usage: "xf doctor [OPTIONS]".into(),
            flags: vec![
                FlagDoc {
                    name: "--archive".into(),
                    short: None,
                    description: "Path to the X data archive directory".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--fix".into(),
                    short: None,
                    description: "Apply safe, idempotent repairs when issues are found".into(),
                    default: None,
                    required: false,
                },
            ],
            output_formats: all_formats(),
        },
        CommandDoc {
            name: "shell".into(),
            description: "Launch interactive REPL mode".into(),
            usage: "xf shell [OPTIONS]".into(),
            flags: vec![
                FlagDoc {
                    name: "--prompt".into(),
                    short: None,
                    description: "Custom prompt string".into(),
                    default: Some("xf> ".into()),
                    required: false,
                },
                FlagDoc {
                    name: "--page-size".into(),
                    short: None,
                    description: "Results per page".into(),
                    default: Some("10".into()),
                    required: false,
                },
                FlagDoc {
                    name: "--no-history".into(),
                    short: None,
                    description: "Disable history persistence".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--history-file".into(),
                    short: None,
                    description: "Custom history file path".into(),
                    default: None,
                    required: false,
                },
            ],
            output_formats: None,
        },
        CommandDoc {
            name: "benchmark".into(),
            description: "Run embedding and reranker performance/quality benchmarks".into(),
            usage: "xf benchmark [OPTIONS]".into(),
            flags: vec![
                FlagDoc {
                    name: "--model".into(),
                    short: None,
                    description: "Embedding model to benchmark".into(),
                    default: Some("all-MiniLM-L6-v2".into()),
                    required: false,
                },
                FlagDoc {
                    name: "--dimensions".into(),
                    short: None,
                    description: "MRL truncation dimension".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--corpus".into(),
                    short: None,
                    description: "Path to benchmark corpus JSON".into(),
                    default: Some("tests/fixtures/benchmark_corpus.json".into()),
                    required: false,
                },
                FlagDoc {
                    name: "--warmup-iters".into(),
                    short: None,
                    description: "Number of warmup iterations".into(),
                    default: Some("10".into()),
                    required: false,
                },
                FlagDoc {
                    name: "--measure-iters".into(),
                    short: None,
                    description: "Number of measurement iterations".into(),
                    default: Some("100".into()),
                    required: false,
                },
                FlagDoc {
                    name: "--batch-size".into(),
                    short: None,
                    description: "Batch size for throughput benchmark".into(),
                    default: Some("32".into()),
                    required: false,
                },
                FlagDoc {
                    name: "--reranker".into(),
                    short: None,
                    description: "Reranker model to benchmark".into(),
                    default: Some("none".into()),
                    required: false,
                },
                FlagDoc {
                    name: "--output-dir".into(),
                    short: Some("-o".into()),
                    description: "Output directory for benchmark reports (alias: --output)".into(),
                    default: Some("results".into()),
                    required: false,
                },
            ],
            output_formats: all_formats(),
        },
        CommandDoc {
            name: "robot-docs".into(),
            description: "Show machine-readable CLI documentation (JSON)".into(),
            usage: "xf robot-docs [TOPIC]".into(),
            flags: vec![],
            output_formats: all_formats(),
        },
        CommandDoc {
            name: "models".into(),
            description: "Manage embedding and reranker models (list, show)".into(),
            usage: "xf models <SUBCOMMAND>".into(),
            flags: vec![],
            output_formats: all_formats(),
        },
        CommandDoc {
            name: "daemon".into(),
            description: "Manage the warm model daemon (start, stop, status, submit, cancel)".into(),
            usage: "xf daemon <SUBCOMMAND>".into(),
            flags: vec![],
            output_formats: all_formats(),
        },
    ]
}

fn examples() -> Vec<Example> {
    vec![
        Example {
            description: "Import a zip archive".into(),
            command: "xf import ~/Downloads/twitter-2026-01-09-abc123.zip".into(),
            notes: Some("Extracts and indexes in one step".into()),
        },
        Example {
            description: "Index an already-extracted archive".into(),
            command: "xf index ~/my_x_history".into(),
            notes: None,
        },
        Example {
            description: "Search tweets as JSON".into(),
            command: "xf search 'hello world' --format json".into(),
            notes: None,
        },
        Example {
            description: "Search with date range filter".into(),
            command: "xf search 'topic' --since 2024-01-01 --until 2024-12-31".into(),
            notes: None,
        },
        Example {
            description: "Search only DMs".into(),
            command: "xf search 'secret' --types dm".into(),
            notes: None,
        },
        Example {
            description: "Get archive statistics as JSON".into(),
            command: "xf stats --format json".into(),
            notes: None,
        },
        Example {
            description: "Get compact agent-friendly output".into(),
            command: "xf search 'query' --format toon".into(),
            notes: Some("TOON format minimizes token usage for LLM consumption".into()),
        },
        Example {
            description: "Check archive health".into(),
            command: "xf doctor".into(),
            notes: Some("Returns exit code 1 if errors found".into()),
        },
        Example {
            description: "Check health and auto-fix".into(),
            command: "xf doctor --fix".into(),
            notes: Some(
                "Applies safe repairs (FTS rebuild, vector index rebuild, PRAGMA optimize)".into(),
            ),
        },
        Example {
            description: "Launch interactive shell".into(),
            command: "xf shell".into(),
            notes: Some("Supports :help, :mode, :history, :quit commands".into()),
        },
        Example {
            description: "Show specific tweet details".into(),
            command: "xf tweet 1234567890 --format json".into(),
            notes: None,
        },
        Example {
            description: "List recent tweets".into(),
            command: "xf list tweets --limit 10 --format json".into(),
            notes: None,
        },
    ]
}

fn exit_codes() -> Vec<ExitCode> {
    vec![
        ExitCode {
            code: 0,
            meaning: "Success".into(),
            commands: vec!["all".into()],
        },
        ExitCode {
            code: 1,
            meaning: "Error or health check failures found".into(),
            commands: vec!["all".into(), "doctor".into()],
        },
        ExitCode {
            code: 2,
            meaning: "Invalid arguments or usage error".into(),
            commands: vec!["all".into()],
        },
    ]
}

fn output_formats() -> Vec<OutputFormatDoc> {
    vec![
        OutputFormatDoc {
            name: "text".into(),
            flag: "--format text".into(),
            description: "Human-readable styled text (default for TTY)".into(),
            mime_type: "text/plain".into(),
        },
        OutputFormatDoc {
            name: "json".into(),
            flag: "--format json".into(),
            description: "Compact JSON on a single line".into(),
            mime_type: "application/json".into(),
        },
        OutputFormatDoc {
            name: "json-pretty".into(),
            flag: "--format json-pretty".into(),
            description: "Pretty-printed JSON with indentation".into(),
            mime_type: "application/json".into(),
        },
        OutputFormatDoc {
            name: "compact".into(),
            flag: "--format compact".into(),
            description: "Single-line summary per item".into(),
            mime_type: "text/plain".into(),
        },
        OutputFormatDoc {
            name: "csv".into(),
            flag: "--format csv".into(),
            description: "Comma-separated values (search, list, export)".into(),
            mime_type: "text/csv".into(),
        },
        OutputFormatDoc {
            name: "toon".into(),
            flag: "--format toon".into(),
            description: "Compact TOON encoding optimized for LLM token efficiency".into(),
            mime_type: "application/x-toon".into(),
        },
        OutputFormatDoc {
            name: "jsonl".into(),
            flag: "--format jsonl".into(),
            description: "JSON Lines (newline-delimited JSON objects)".into(),
            mime_type: "application/x-ndjson".into(),
        },
    ]
}

#[allow(clippy::too_many_lines)]
fn schemas() -> Vec<SchemaDoc> {
    vec![
        SchemaDoc {
            name: "search".into(),
            description: "Output of `xf search` with --format json. Returns an array of search results.".into(),
            schema: json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["result_type", "id", "text", "created_at", "score", "highlights", "metadata"],
                    "properties": {
                        "result_type": {
                            "type": "string",
                            "enum": ["tweet", "like", "direct_message", "grok_message", "bookmark"],
                            "description": "Type of content matched"
                        },
                        "id": {
                            "type": "string",
                            "description": "Unique identifier for the content item"
                        },
                        "text": {
                            "type": "string",
                            "description": "Full text content"
                        },
                        "created_at": {
                            "type": "string",
                            "format": "date-time",
                            "description": "ISO 8601 timestamp"
                        },
                        "score": {
                            "type": "number",
                            "description": "BM25 or hybrid RRF relevance score"
                        },
                        "highlights": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Text fragments with matching terms highlighted"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "Type-specific metadata (varies by result_type)"
                        },
                        "rerank_score": {
                            "type": "number",
                            "description": "Cross-encoder reranker score (present only when --rerank is used)"
                        }
                    }
                }
            }),
        },
        SchemaDoc {
            name: "stats".into(),
            description: "Output of `xf stats` with --format json. Shape depends on flags.".into(),
            schema: json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["stats"],
                "properties": {
                    "stats": {
                        "type": "object",
                        "required": [
                            "tweets_count", "likes_count", "dms_count",
                            "dm_conversations_count", "followers_count", "following_count",
                            "blocks_count", "mutes_count", "grok_messages_count",
                            "bookmarks_count", "index_built_at"
                        ],
                        "properties": {
                            "tweets_count": { "type": "integer" },
                            "likes_count": { "type": "integer" },
                            "dms_count": { "type": "integer" },
                            "dm_conversations_count": { "type": "integer" },
                            "followers_count": { "type": "integer" },
                            "following_count": { "type": "integer" },
                            "blocks_count": { "type": "integer" },
                            "mutes_count": { "type": "integer" },
                            "grok_messages_count": { "type": "integer" },
                            "bookmarks_count": { "type": "integer" },
                            "first_tweet_date": {
                                "type": ["string", "null"],
                                "format": "date-time"
                            },
                            "last_tweet_date": {
                                "type": ["string", "null"],
                                "format": "date-time"
                            },
                            "index_built_at": {
                                "type": "string",
                                "format": "date-time"
                            }
                        }
                    },
                    "detailed": {
                        "type": "array",
                        "description": "Monthly breakdown (present with --detailed)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "year": { "type": "integer" },
                                "month": { "type": "integer" },
                                "count": { "type": "integer" }
                            }
                        }
                    },
                    "top_hashtags": {
                        "type": "array",
                        "description": "Top hashtags (present with --detailed or --hashtags)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "value": { "type": "string" },
                                "count": { "type": "integer" }
                            }
                        }
                    },
                    "top_mentions": {
                        "type": "array",
                        "description": "Top mentions (present with --detailed or --mentions)",
                        "items": {
                            "type": "object",
                            "properties": {
                                "value": { "type": "string" },
                                "count": { "type": "integer" }
                            }
                        }
                    },
                    "temporal": {
                        "type": "object",
                        "description": "Temporal analytics (present with --detailed or --temporal)"
                    },
                    "engagement": {
                        "type": "object",
                        "description": "Engagement analytics (present with --detailed or --engagement)"
                    },
                    "content": {
                        "type": "object",
                        "description": "Content analysis (present with --detailed or --content)"
                    }
                }
            }),
        },
        SchemaDoc {
            name: "tweet".into(),
            description: "Output of `xf tweet <ID>` with --format json. Returns a single tweet object.".into(),
            schema: json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": [
                    "id", "created_at", "full_text", "favorite_count",
                    "retweet_count", "is_retweet", "hashtags",
                    "user_mentions", "urls", "media"
                ],
                "properties": {
                    "id": { "type": "string" },
                    "created_at": {
                        "type": "string",
                        "format": "date-time"
                    },
                    "full_text": {
                        "type": "string",
                        "description": "Complete tweet text"
                    },
                    "source": {
                        "type": ["string", "null"],
                        "description": "Client used to post (e.g. 'Twitter Web App')"
                    },
                    "favorite_count": { "type": "integer" },
                    "retweet_count": { "type": "integer" },
                    "lang": {
                        "type": ["string", "null"],
                        "description": "ISO 639-1 language code"
                    },
                    "in_reply_to_status_id": { "type": ["string", "null"] },
                    "in_reply_to_user_id": { "type": ["string", "null"] },
                    "in_reply_to_screen_name": { "type": ["string", "null"] },
                    "is_retweet": { "type": "boolean" },
                    "hashtags": {
                        "type": "array",
                        "items": { "type": "string" }
                    },
                    "user_mentions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": { "type": "string" },
                                "screen_name": { "type": "string" },
                                "name": { "type": "string" }
                            }
                        }
                    },
                    "urls": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "url": { "type": "string" },
                                "expanded_url": { "type": "string" },
                                "display_url": { "type": "string" }
                            }
                        }
                    },
                    "media": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "media_url": { "type": "string" },
                                "media_type": { "type": "string" },
                                "expanded_url": { "type": ["string", "null"] }
                            }
                        }
                    }
                }
            }),
        },
        SchemaDoc {
            name: "doctor".into(),
            description: "Output of `xf doctor` with --format json. Returns health check results.".into(),
            schema: json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["checks", "summary", "suggestions", "runtime_ms"],
                "properties": {
                    "checks": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "required": ["category", "name", "status", "message"],
                            "properties": {
                                "category": {
                                    "type": "string",
                                    "enum": ["archive", "database", "index", "performance"],
                                    "description": "Check category"
                                },
                                "name": {
                                    "type": "string",
                                    "description": "Short check name"
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pass", "warning", "error"],
                                    "description": "Check result"
                                },
                                "message": {
                                    "type": "string",
                                    "description": "Human-readable result message"
                                },
                                "suggestion": {
                                    "type": "string",
                                    "description": "Remediation suggestion (present for warnings/errors)"
                                }
                            }
                        }
                    },
                    "summary": {
                        "type": "object",
                        "required": ["passed", "warnings", "errors", "total"],
                        "properties": {
                            "passed": { "type": "integer" },
                            "warnings": { "type": "integer" },
                            "errors": { "type": "integer" },
                            "total": { "type": "integer" }
                        }
                    },
                    "suggestions": {
                        "type": "array",
                        "items": { "type": "string" },
                        "description": "Overall suggestions for fixing issues"
                    },
                    "runtime_ms": {
                        "type": "integer",
                        "description": "Total check duration in milliseconds"
                    }
                }
            }),
        },
        SchemaDoc {
            name: "list-tweets".into(),
            description: "Output of `xf list tweets` with --format json. Array of Tweet objects.".into(),
            schema: json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "array",
                "items": { "$ref": "#/definitions/tweet" },
                "definitions": {
                    "tweet": {
                        "type": "object",
                        "description": "Same schema as the tweet command output",
                        "required": ["id", "created_at", "full_text"],
                        "properties": {
                            "id": { "type": "string" },
                            "created_at": { "type": "string", "format": "date-time" },
                            "full_text": { "type": "string" },
                            "favorite_count": { "type": "integer" },
                            "retweet_count": { "type": "integer" },
                            "is_retweet": { "type": "boolean" }
                        }
                    }
                }
            }),
        },
        SchemaDoc {
            name: "list-likes".into(),
            description: "Output of `xf list likes` with --format json. Array of Like objects.".into(),
            schema: json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["tweet_id"],
                    "properties": {
                        "tweet_id": {
                            "type": "string",
                            "description": "ID of the liked tweet"
                        },
                        "full_text": {
                            "type": ["string", "null"],
                            "description": "Text of the liked tweet (if available)"
                        },
                        "expanded_url": {
                            "type": ["string", "null"],
                            "description": "URL to the liked tweet"
                        }
                    }
                }
            }),
        },
        SchemaDoc {
            name: "list-conversations".into(),
            description: "Output of `xf list conversations` with --format json. Array of DM conversation summaries.".into(),
            schema: json!({
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["conversation_id", "participant_ids", "message_count"],
                    "properties": {
                        "conversation_id": { "type": "string" },
                        "participant_ids": {
                            "type": "array",
                            "items": { "type": "string" }
                        },
                        "message_count": { "type": "integer" },
                        "first_message_at": {
                            "type": ["string", "null"],
                            "format": "date-time"
                        },
                        "last_message_at": {
                            "type": ["string", "null"],
                            "format": "date-time"
                        }
                    }
                }
            }),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_all() {
        let docs = generate("all");
        assert_eq!(docs.schema_version, SCHEMA_VERSION);
        assert!(docs.quickstart.is_some());
        assert!(docs.commands.is_some());
        assert!(docs.examples.is_some());
        assert!(docs.exit_codes.is_some());
        assert!(docs.output_formats.is_some());
        assert!(docs.schemas.is_some());
    }

    #[test]
    fn test_generate_quickstart_only() {
        let docs = generate("quickstart");
        assert!(docs.quickstart.is_some());
        assert!(docs.commands.is_none());
        assert!(docs.examples.is_none());
    }

    #[test]
    fn test_generate_commands_only() {
        let docs = generate("commands");
        assert!(docs.quickstart.is_none());
        assert!(docs.commands.is_some());
        let cmds = docs.commands.unwrap();
        assert!(!cmds.is_empty());
        // Should include key commands
        let names: Vec<&str> = cmds.iter().map(|c| c.name.as_str()).collect();
        assert!(names.contains(&"search"));
        assert!(names.contains(&"index"));
        assert!(names.contains(&"doctor"));
        assert!(names.contains(&"robot-docs"));
    }

    #[test]
    fn test_generate_examples_only() {
        let docs = generate("examples");
        assert!(docs.examples.is_some());
        let examples = docs.examples.unwrap();
        assert!(!examples.is_empty());
        // Each example should have a command
        for ex in &examples {
            assert_ne!(ex.command, "");
            assert_ne!(ex.description, "");
        }
    }

    #[test]
    fn test_generate_exit_codes() {
        let docs = generate("exit-codes");
        assert!(docs.exit_codes.is_some());
        let codes = docs.exit_codes.unwrap();
        // Must have at least 0 and 1
        assert!(codes.iter().any(|c| c.code == 0));
        assert!(codes.iter().any(|c| c.code == 1));
    }

    #[test]
    fn test_generate_output_formats() {
        let docs = generate("output-formats");
        assert!(docs.output_formats.is_some());
        let formats = docs.output_formats.unwrap();
        let names: Vec<&str> = formats.iter().map(|f| f.name.as_str()).collect();
        assert!(names.contains(&"json"));
        assert!(names.contains(&"toon"));
        assert!(names.contains(&"csv"));
        assert!(names.contains(&"text"));
    }

    #[test]
    fn test_generate_unknown_topic_returns_empty() {
        let docs = generate("nonexistent");
        assert!(docs.quickstart.is_none());
        assert!(docs.commands.is_none());
        assert!(docs.examples.is_none());
        assert!(docs.exit_codes.is_none());
        assert!(docs.output_formats.is_none());
        assert!(docs.schemas.is_none());
        // But version is always present
        assert_ne!(docs.version, "");
    }

    #[test]
    fn test_is_valid_topic() {
        assert!(is_valid_topic("all"));
        assert!(is_valid_topic("quickstart"));
        assert!(is_valid_topic("commands"));
        assert!(is_valid_topic("examples"));
        assert!(is_valid_topic("exit-codes"));
        assert!(is_valid_topic("output-formats"));
        assert!(is_valid_topic("schemas"));
        assert!(!is_valid_topic("nonexistent"));
        assert!(!is_valid_topic(""));
    }

    #[test]
    fn test_version_matches_cargo() {
        let docs = generate("all");
        assert_eq!(docs.version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn test_json_serializable() {
        let docs = generate("all");
        let json = serde_json::to_string(&docs);
        assert!(json.is_ok());
        let json_str = json.unwrap();
        assert!(json_str.contains("schema_version"));
        assert!(json_str.contains("quickstart"));
    }

    #[test]
    fn test_search_command_has_output_formats() {
        let docs = generate("commands");
        let cmds = docs.commands.unwrap();
        let search = cmds.iter().find(|c| c.name == "search").unwrap();
        assert!(search.output_formats.is_some());
        let formats = search.output_formats.as_ref().unwrap();
        assert!(formats.contains(&"json".to_string()));
        assert!(formats.contains(&"toon".to_string()));
    }

    #[test]
    fn test_topics_list_includes_all() {
        assert!(TOPICS.contains(&"all"));
        assert!(TOPICS.len() >= 6);
    }

    #[test]
    fn test_generate_schemas_only() {
        let docs = generate("schemas");
        assert!(docs.schemas.is_some());
        assert!(docs.quickstart.is_none());
        assert!(docs.commands.is_none());
        assert!(docs.examples.is_none());
    }

    #[test]
    fn test_schemas_cover_key_commands() {
        let docs = generate("schemas");
        let schemas = docs.schemas.unwrap();
        let names: Vec<&str> = schemas.iter().map(|s| s.name.as_str()).collect();
        assert!(names.contains(&"search"));
        assert!(names.contains(&"stats"));
        assert!(names.contains(&"tweet"));
        assert!(names.contains(&"doctor"));
        assert!(names.contains(&"list-tweets"));
        assert!(names.contains(&"list-likes"));
        assert!(names.contains(&"list-conversations"));
    }

    #[test]
    fn test_schemas_have_valid_structure() {
        let docs = generate("schemas");
        let schemas = docs.schemas.unwrap();
        for schema_doc in &schemas {
            assert_ne!(schema_doc.name, "");
            assert_ne!(schema_doc.description, "");
            // Each schema should have a $schema key and a type
            let schema = &schema_doc.schema;
            assert!(
                schema.get("$schema").is_some(),
                "Missing $schema in {}",
                schema_doc.name
            );
            assert!(
                schema.get("type").is_some(),
                "Missing type in {}",
                schema_doc.name
            );
        }
    }

    #[test]
    fn test_schemas_serializable() {
        let docs = generate("schemas");
        let json = serde_json::to_string_pretty(&docs);
        assert!(json.is_ok());
        let json_str = json.unwrap();
        assert!(json_str.contains("\"schemas\""));
        assert!(json_str.contains("\"$schema\""));
        assert!(json_str.contains("draft-07"));
    }

    #[test]
    fn test_cli_subcommands_and_output_formats_documented_in_robot_docs() {
        use clap::{CommandFactory, ValueEnum};

        let docs = generate("all");
        let doc_commands: std::collections::HashSet<String> =
            docs.commands.unwrap().into_iter().map(|c| c.name).collect();

        let cli_cmd = crate::cli::Cli::command();
        for sub in cli_cmd.get_subcommands() {
            let name = sub.get_name();
            if name == "help" {
                continue;
            }
            assert!(
                doc_commands.contains(name),
                "CLI subcommand '{name}' is missing from robot_docs::commands()"
            );
        }

        let doc_formats: std::collections::HashSet<String> = docs
            .output_formats
            .unwrap()
            .into_iter()
            .map(|f| f.name)
            .collect();

        for variant in crate::cli::OutputFormat::value_variants() {
            if let Some(pv) = variant.to_possible_value() {
                let name = pv.get_name();
                assert!(
                    doc_formats.contains(name),
                    "OutputFormat variant '{name}' is missing from robot_docs::output_formats()"
                );
            }
        }
    }
}
