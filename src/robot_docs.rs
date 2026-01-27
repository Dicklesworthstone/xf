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
            description: "Index an extracted X data archive for search".into(),
            usage: "xf index <ARCHIVE_PATH> [OPTIONS]".into(),
            flags: vec![
                FlagDoc {
                    name: "--force".into(),
                    short: Some("-F".into()),
                    description: "Rebuild index from scratch".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--db".into(),
                    short: None,
                    description: "Path to SQLite database".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--index-dir".into(),
                    short: None,
                    description: "Path to search index directory".into(),
                    default: None,
                    required: false,
                },
            ],
            output_formats: None,
        },
        CommandDoc {
            name: "search".into(),
            description: "Full-text search over indexed archive".into(),
            usage: "xf search <QUERY> [OPTIONS]".into(),
            flags: vec![
                FlagDoc {
                    name: "--limit".into(),
                    short: Some("-n".into()),
                    description: "Maximum results to return".into(),
                    default: Some("20".into()),
                    required: false,
                },
                FlagDoc {
                    name: "--type".into(),
                    short: Some("-t".into()),
                    description: "Filter by document type (tweet, like, dm, grok)".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--since".into(),
                    short: None,
                    description: "Filter results after date (YYYY-MM-DD)".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--until".into(),
                    short: None,
                    description: "Filter results before date (YYYY-MM-DD)".into(),
                    default: None,
                    required: false,
                },
                FlagDoc {
                    name: "--mode".into(),
                    short: None,
                    description: "Search mode: lexical, semantic, or hybrid".into(),
                    default: Some("lexical".into()),
                    required: false,
                },
            ],
            output_formats: Some(vec![
                "text".into(),
                "json".into(),
                "json-pretty".into(),
                "csv".into(),
                "toon".into(),
            ]),
        },
        CommandDoc {
            name: "stats".into(),
            description: "Show archive statistics and analytics".into(),
            usage: "xf stats [OPTIONS]".into(),
            flags: vec![FlagDoc {
                name: "--section".into(),
                short: Some("-s".into()),
                description:
                    "Show specific section (overview, timeline, hashtags, mentions, engagement)"
                        .into(),
                default: None,
                required: false,
            }],
            output_formats: Some(vec![
                "text".into(),
                "json".into(),
                "json-pretty".into(),
                "toon".into(),
            ]),
        },
        CommandDoc {
            name: "tweet".into(),
            description: "Show detailed information about a specific tweet".into(),
            usage: "xf tweet <ID> [OPTIONS]".into(),
            flags: vec![],
            output_formats: Some(vec![
                "text".into(),
                "json".into(),
                "json-pretty".into(),
                "toon".into(),
            ]),
        },
        CommandDoc {
            name: "list".into(),
            description: "List data from the archive (tweets, likes, dms, etc.)".into(),
            usage: "xf list <TARGET> [OPTIONS]".into(),
            flags: vec![FlagDoc {
                name: "--limit".into(),
                short: Some("-n".into()),
                description: "Maximum items to list".into(),
                default: Some("20".into()),
                required: false,
            }],
            output_formats: Some(vec![
                "text".into(),
                "json".into(),
                "json-pretty".into(),
                "csv".into(),
                "toon".into(),
            ]),
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
            output_formats: Some(vec![
                "text".into(),
                "json".into(),
                "json-pretty".into(),
                "toon".into(),
            ]),
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
            ],
            output_formats: None,
        },
        CommandDoc {
            name: "export".into(),
            description: "Export data in various formats".into(),
            usage: "xf export <TARGET> [OPTIONS]".into(),
            flags: vec![FlagDoc {
                name: "--format".into(),
                short: Some("-f".into()),
                description: "Export format (json, csv, markdown)".into(),
                default: Some("json".into()),
                required: false,
            }],
            output_formats: None,
        },
        CommandDoc {
            name: "robot-docs".into(),
            description: "Show machine-readable CLI documentation (JSON)".into(),
            usage: "xf robot-docs [TOPIC]".into(),
            flags: vec![],
            output_formats: Some(vec!["json".into()]),
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
            command: "xf search 'secret' --type dm".into(),
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
            notes: Some("Applies safe repairs (FTS rebuild, PRAGMA optimize)".into()),
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
                            "enum": ["tweet", "like", "direct_message", "grok_message"],
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
                            "description": "BM25 relevance score"
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
                            "index_built_at"
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
            assert!(!ex.command.is_empty());
            assert!(!ex.description.is_empty());
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
        assert!(!docs.version.is_empty());
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
            assert!(!schema_doc.name.is_empty());
            assert!(!schema_doc.description.is_empty());
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
}
