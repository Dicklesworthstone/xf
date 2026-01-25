//! Tantivy-based full-text search engine for X data.
//!
//! Provides ultra-fast search with BM25 ranking, prefix matching, and phrase queries.

use crate::doctor::{CheckCategory, CheckStatus, HealthCheck};
use crate::format_bytes;
use crate::model::{DmConversation, GrokMessage, Like, SearchResult, SearchResultType, Tweet};
use crate::storage::Storage;
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tantivy::collector::TopDocs;
use tantivy::query::{AllQuery, BooleanQuery, Occur, Query, QueryParser, TermQuery, TermSetQuery};
use tantivy::schema::{
    FAST, Field, INDEXED, IndexRecordOption, STORED, STRING, Schema, TextFieldIndexing,
    TextOptions, Value,
};
use tantivy::snippet::SnippetGenerator;
use tantivy::{Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term, doc};
use tracing::info;

/// Parse metadata JSON, avoiding parse overhead for empty objects.
#[inline]
fn parse_metadata(s: &str) -> serde_json::Value {
    // Fast path: empty metadata is common, skip parsing entirely.
    // Preserve original semantics: "{}" → empty object, "null" → null.
    match s {
        "{}" => serde_json::Value::Object(serde_json::Map::new()),
        "null" => serde_json::Value::Null,
        _ => serde_json::from_str(s).unwrap_or_default(),
    }
}

/// Schema field names
const FIELD_ID: &str = "id";
const FIELD_TEXT: &str = "text";
const FIELD_TEXT_PREFIX: &str = "text_prefix";
const FIELD_TYPE: &str = "type";
const FIELD_CREATED_AT: &str = "created_at";
const FIELD_METADATA: &str = "metadata";

const LARGE_INDEX_BYTES: u64 = 500 * 1024 * 1024;
const MAX_DOC_TYPES: usize = 4;

const fn epoch_utc() -> DateTime<Utc> {
    DateTime::<Utc>::from_timestamp(0, 0).unwrap()
}

fn build_lookup_query(
    id_field: Field,
    type_field: Field,
    lookups: &[DocLookup<'_>],
) -> Option<Box<dyn Query>> {
    let mut untyped_ids: Vec<&str> = Vec::new();
    let mut typed_ids: HashMap<&str, Vec<&str>> = HashMap::new();

    for lookup in lookups {
        if let Some(doc_type) = lookup.doc_type {
            typed_ids.entry(doc_type).or_default().push(lookup.id);
        } else {
            untyped_ids.push(lookup.id);
        }
    }

    let mut subqueries: Vec<(Occur, Box<dyn Query>)> = Vec::new();

    if !untyped_ids.is_empty() {
        let terms = untyped_ids
            .iter()
            .map(|id| Term::from_field_text(id_field, id));
        let id_query = TermSetQuery::new(terms);
        subqueries.push((Occur::Should, Box::new(id_query)));
    }

    for (doc_type, ids) in typed_ids {
        let terms = ids.iter().map(|id| Term::from_field_text(id_field, id));
        let id_query = TermSetQuery::new(terms);
        let type_query = TermQuery::new(
            Term::from_field_text(type_field, doc_type),
            IndexRecordOption::Basic,
        );
        let typed_filter_query = BooleanQuery::new(vec![
            (Occur::Must, Box::new(id_query)),
            (Occur::Must, Box::new(type_query)),
        ]);
        subqueries.push((Occur::Should, Box::new(typed_filter_query)));
    }

    if subqueries.is_empty() {
        return None;
    }

    if subqueries.len() == 1 {
        Some(subqueries.swap_remove(0).1)
    } else {
        Some(Box::new(BooleanQuery::new(subqueries)))
    }
}

fn doc_to_search_result(
    doc: &TantivyDocument,
    id_field: Field,
    text_field: Field,
    type_field: Field,
    created_at_field: Field,
    metadata_field: Field,
) -> (SearchResult, String) {
    let id = doc
        .get_first(id_field)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let text = doc
        .get_first(text_field)
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    let doc_type_str = doc
        .get_first(type_field)
        .and_then(|v| v.as_str())
        .unwrap_or("tweet");

    let created_at_ts = doc
        .get_first(created_at_field)
        .and_then(|v| v.as_i64())
        .unwrap_or(0);

    let metadata_str = doc
        .get_first(metadata_field)
        .and_then(|v| v.as_str())
        .unwrap_or("{}");

    let result_type = match doc_type_str {
        "like" => SearchResultType::Like,
        "dm" => SearchResultType::DirectMessage,
        "grok" => SearchResultType::GrokMessage,
        _ => SearchResultType::Tweet,
    };

    let result = SearchResult {
        result_type,
        id,
        text,
        created_at: DateTime::from_timestamp(created_at_ts, 0).unwrap_or_else(epoch_utc),
        score: 1.0,
        highlights: vec![],
        metadata: parse_metadata(metadata_str),
        rerank_score: None,
    };

    (result, doc_type_str.to_string())
}

/// Document types stored in the index
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DocType {
    Tweet,
    Like,
    DirectMessage,
    GrokMessage,
}

/// Document lookup key for batch retrieval.
#[derive(Debug, Clone, Copy)]
pub struct DocLookup<'a> {
    pub id: &'a str,
    pub doc_type: Option<&'a str>,
}

impl<'a> DocLookup<'a> {
    #[must_use]
    pub const fn new(id: &'a str) -> Self {
        Self { id, doc_type: None }
    }

    #[must_use]
    pub const fn with_type(id: &'a str, doc_type: &'a str) -> Self {
        Self {
            id,
            doc_type: Some(doc_type),
        }
    }
}

impl DocType {
    /// Get the string representation of the doc type.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Tweet => "tweet",
            Self::Like => "like",
            Self::DirectMessage => "dm",
            Self::GrokMessage => "grok",
        }
    }

    #[allow(dead_code)]
    fn from_str(s: &str) -> Option<Self> {
        match s {
            "tweet" => Some(Self::Tweet),
            "like" => Some(Self::Like),
            "dm" => Some(Self::DirectMessage),
            "grok" => Some(Self::GrokMessage),
            _ => None,
        }
    }
}

/// Build the Tantivy schema
fn build_schema() -> Schema {
    let mut schema_builder = Schema::builder();

    // ID field - stored but not indexed for search
    schema_builder.add_text_field(FIELD_ID, STRING | STORED);

    // Main text field - tokenized for full-text search
    let text_options = TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer("default")
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        )
        .set_stored();
    schema_builder.add_text_field(FIELD_TEXT, text_options);

    // Prefix text field - for edge n-gram style prefix matching
    let prefix_options = TextOptions::default().set_indexing_options(
        TextFieldIndexing::default()
            .set_tokenizer("default")
            .set_index_option(IndexRecordOption::Basic),
    );
    schema_builder.add_text_field(FIELD_TEXT_PREFIX, prefix_options);

    // Document type - exact match only
    schema_builder.add_text_field(FIELD_TYPE, STRING | STORED);

    // Created at timestamp - for sorting and range queries
    schema_builder.add_i64_field(FIELD_CREATED_AT, INDEXED | STORED | FAST);

    // Metadata JSON - stored for retrieval
    schema_builder.add_text_field(FIELD_METADATA, STORED);

    schema_builder.build()
}

/// Search engine wrapping Tantivy
pub struct SearchEngine {
    index: Index,
    schema: Schema,
    reader: IndexReader,
    index_path: Option<PathBuf>,
}

impl SearchEngine {
    /// Create or open an index at the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the index directory cannot be created or opened.
    pub fn open(index_path: impl AsRef<Path>) -> Result<Self> {
        let index_path = index_path.as_ref();
        std::fs::create_dir_all(index_path)?;

        let schema = build_schema();

        let index = if index_path.join("meta.json").exists() {
            Index::open_in_dir(index_path)
                .with_context(|| format!("Failed to open index at {}", index_path.display()))?
        } else {
            Index::create_in_dir(index_path, schema.clone())
                .with_context(|| format!("Failed to create index at {}", index_path.display()))?
        };

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()?;

        Ok(Self {
            index,
            schema,
            reader,
            index_path: Some(index_path.to_path_buf()),
        })
    }

    /// Create an in-memory index (for testing).
    ///
    /// # Errors
    ///
    /// Returns an error if the in-memory index cannot be created.
    pub fn open_memory() -> Result<Self> {
        let schema = build_schema();
        let index = Index::create_in_ram(schema.clone());

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()?;

        Ok(Self {
            index,
            schema,
            reader,
            index_path: None,
        })
    }

    /// Return the on-disk index path when available.
    #[must_use]
    pub fn index_path(&self) -> Option<&Path> {
        self.index_path.as_deref()
    }

    /// Get a writer for indexing.
    ///
    /// # Errors
    ///
    /// Returns an error if the writer cannot be created.
    pub fn writer(&self, heap_size: usize) -> Result<IndexWriter> {
        self.index
            .writer(heap_size)
            .context("Failed to create index writer")
    }

    /// Reload the reader to see committed changes.
    ///
    /// # Errors
    ///
    /// Returns an error if the reader cannot be reloaded.
    pub fn reload(&self) -> Result<()> {
        self.reader.reload()?;
        Ok(())
    }

    /// Get schema fields
    fn get_fields(&self) -> (Field, Field, Field, Field, Field, Field) {
        (
            self.schema.get_field(FIELD_ID).unwrap(),
            self.schema.get_field(FIELD_TEXT).unwrap(),
            self.schema.get_field(FIELD_TEXT_PREFIX).unwrap(),
            self.schema.get_field(FIELD_TYPE).unwrap(),
            self.schema.get_field(FIELD_CREATED_AT).unwrap(),
            self.schema.get_field(FIELD_METADATA).unwrap(),
        )
    }

    /// Index tweets.
    ///
    /// # Errors
    ///
    /// Returns an error if any document cannot be added to the index.
    pub fn index_tweets(&self, writer: &mut IndexWriter, tweets: &[Tweet]) -> Result<usize> {
        let (id_field, text_field, prefix_field, type_field, created_at_field, metadata_field) =
            self.get_fields();

        let mut count = 0;
        for tweet in tweets {
            // Generate prefix terms
            let prefixes = generate_prefixes(&tweet.full_text);

            let metadata = serde_json::json!({
                "favorite_count": tweet.favorite_count,
                "retweet_count": tweet.retweet_count,
                "in_reply_to": tweet.in_reply_to_screen_name,
                "hashtags": tweet.hashtags,
                "source": tweet.source,
            });

            writer.add_document(doc!(
                id_field => tweet.id.clone(),
                text_field => tweet.full_text.clone(),
                prefix_field => prefixes,
                type_field => DocType::Tweet.as_str(),
                created_at_field => tweet.created_at.timestamp(),
                metadata_field => metadata.to_string(),
            ))?;
            count += 1;
        }

        info!("Indexed {} tweets", count);
        Ok(count)
    }

    /// Index likes.
    ///
    /// # Errors
    ///
    /// Returns an error if any document cannot be added to the index.
    pub fn index_likes(&self, writer: &mut IndexWriter, likes: &[Like]) -> Result<usize> {
        let (id_field, text_field, prefix_field, type_field, created_at_field, metadata_field) =
            self.get_fields();

        let mut count = 0;
        for like in likes {
            if let Some(text) = &like.full_text {
                if text.is_empty() {
                    continue;
                }

                let prefixes = generate_prefixes(text);

                let metadata = serde_json::json!({
                    "expanded_url": like.expanded_url,
                });

                writer.add_document(doc!(
                    id_field => like.tweet_id.clone(),
                    text_field => text.clone(),
                    prefix_field => prefixes,
                    type_field => DocType::Like.as_str(),
                    created_at_field => 0i64, // Likes don't have timestamps
                    metadata_field => metadata.to_string(),
                ))?;
                count += 1;
            }
        }

        info!("Indexed {} likes", count);
        Ok(count)
    }

    /// Index direct messages.
    ///
    /// # Errors
    ///
    /// Returns an error if any document cannot be added to the index.
    pub fn index_dms(
        &self,
        writer: &mut IndexWriter,
        conversations: &[DmConversation],
    ) -> Result<usize> {
        let (id_field, text_field, prefix_field, type_field, created_at_field, metadata_field) =
            self.get_fields();

        let mut count = 0;
        for conv in conversations {
            for msg in &conv.messages {
                let prefixes = generate_prefixes(&msg.text);

                let metadata = serde_json::json!({
                    "conversation_id": conv.conversation_id,
                    "sender_id": msg.sender_id,
                    "recipient_id": msg.recipient_id,
                });

                writer.add_document(doc!(
                    id_field => msg.id.clone(),
                    text_field => msg.text.clone(),
                    prefix_field => prefixes,
                    type_field => DocType::DirectMessage.as_str(),
                    created_at_field => msg.created_at.timestamp(),
                    metadata_field => metadata.to_string(),
                ))?;
                count += 1;
            }
        }

        info!("Indexed {} DMs", count);
        Ok(count)
    }

    /// Index Grok messages.
    ///
    /// # Errors
    ///
    /// Returns an error if any document cannot be added to the index.
    pub fn index_grok_messages(
        &self,
        writer: &mut IndexWriter,
        messages: &[GrokMessage],
    ) -> Result<usize> {
        let (id_field, text_field, prefix_field, type_field, created_at_field, metadata_field) =
            self.get_fields();

        let mut count = 0;
        for msg in messages {
            let prefixes = generate_prefixes(&msg.message);

            let metadata = serde_json::json!({
                "chat_id": msg.chat_id,
                "sender": msg.sender,
                "grok_mode": msg.grok_mode,
            });

            // Use chat_id + timestamp_nanos + sender for better uniqueness
            let doc_id = format!(
                "{}_{}_{}_{}",
                msg.chat_id,
                msg.created_at.timestamp(),
                msg.created_at.timestamp_subsec_nanos(),
                msg.sender
            );

            writer.add_document(doc!(
                id_field => doc_id,
                text_field => msg.message.clone(),
                prefix_field => prefixes,
                type_field => DocType::GrokMessage.as_str(),
                created_at_field => msg.created_at.timestamp(),
                metadata_field => metadata.to_string(),
            ))?;
            count += 1;
        }

        info!("Indexed {} Grok messages", count);
        Ok(count)
    }

    /// Search the index.
    ///
    /// # Errors
    ///
    /// Returns an error if the query cannot be parsed or the search fails.
    #[allow(clippy::too_many_lines)]
    pub fn search(
        &self,
        query_str: &str,
        doc_types: Option<&[DocType]>,
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        if limit == 0 {
            return Ok(Vec::new());
        }
        let searcher = self.reader.searcher();
        let (id_field, text_field, prefix_field, type_field, created_at_field, metadata_field) =
            self.get_fields();

        // Build query
        let trimmed = query_str.trim();
        let mut enable_highlights = true;
        let base_query: Box<dyn Query> = if trimmed.is_empty() {
            enable_highlights = false;
            Box::new(AllQuery)
        } else {
            // Check if query contains quoted phrases - prefix_field doesn't have positions
            // indexed (IndexRecordOption::Basic), so phrase queries would fail on it.
            // Only include prefix_field for queries without phrases to enable prefix matching.
            let has_phrase = trimmed.contains('"');
            let fields = if has_phrase {
                vec![text_field]
            } else {
                vec![text_field, prefix_field]
            };
            let query_parser = QueryParser::for_index(&self.index, fields);
            query_parser
                .parse_query(trimmed)
                .map_err(|e| anyhow::anyhow!("Invalid search query: {e}"))?
        };

        // Apply type filter if specified
        let query: Box<dyn Query> = if let Some(types) = doc_types {
            let type_queries: Vec<(Occur, Box<dyn Query>)> = types
                .iter()
                .map(|t| {
                    (
                        Occur::Should,
                        Box::new(TermQuery::new(
                            Term::from_field_text(type_field, t.as_str()),
                            IndexRecordOption::Basic,
                        )) as Box<dyn Query>,
                    )
                })
                .collect();

            let type_filter = BooleanQuery::new(type_queries);

            Box::new(BooleanQuery::new(vec![
                (Occur::Must, base_query),
                (Occur::Must, Box::new(type_filter)),
            ]))
        } else {
            base_query
        };

        // Execute search
        let top_docs = searcher.search(&query, &TopDocs::with_limit(limit))?;

        // Create snippet generator for highlighting when query has terms
        let snippet_generator = if enable_highlights {
            Some(SnippetGenerator::create(&searcher, &query, text_field)?)
        } else {
            None
        };

        // Collect results
        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;

            let id = doc
                .get_first(id_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let text = doc
                .get_first(text_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let doc_type_str = doc
                .get_first(type_field)
                .and_then(|v| v.as_str())
                .unwrap_or("tweet");

            let created_at_ts = doc
                .get_first(created_at_field)
                .and_then(|v| v.as_i64())
                .unwrap_or(0);

            let metadata_str = doc
                .get_first(metadata_field)
                .and_then(|v| v.as_str())
                .unwrap_or("{}");

            let result_type = match doc_type_str {
                "like" => SearchResultType::Like,
                "dm" => SearchResultType::DirectMessage,
                "grok" => SearchResultType::GrokMessage,
                _ => SearchResultType::Tweet,
            };

            let highlights = snippet_generator
                .as_ref()
                .map_or_else(Vec::new, |generator| {
                    let snippet = generator.snippet_from_doc(&doc);
                    if snippet.is_empty() {
                        vec![]
                    } else {
                        let html = snippet.to_html();
                        vec![html]
                    }
                });

            results.push(SearchResult {
                result_type,
                id,
                text,
                created_at: DateTime::from_timestamp(created_at_ts, 0).unwrap_or_else(epoch_utc),
                score,
                highlights,
                metadata: parse_metadata(metadata_str),
                rerank_score: None,
            });
        }

        Ok(results)
    }

    /// Get a single document by its ID.
    ///
    /// Returns the document if found, None if not found.
    ///
    /// # Errors
    ///
    /// Returns an error if the search operation fails.
    pub fn get_by_id(&self, doc_id: &str) -> Result<Option<SearchResult>> {
        self.get_by_id_impl(doc_id, None)
    }

    /// Get a single document by its ID and type.
    ///
    /// Useful when multiple document types can share the same ID
    /// (e.g., tweets vs likes).
    ///
    /// # Errors
    ///
    /// Returns an error if the search operation fails.
    pub fn get_by_id_and_type(&self, doc_id: &str, doc_type: &str) -> Result<Option<SearchResult>> {
        self.get_by_id_impl(doc_id, Some(doc_type))
    }

    /// Get multiple documents by ID in a single query.
    ///
    /// Returns results aligned to the input order. Missing IDs yield `None`.
    ///
    /// # Errors
    ///
    /// Returns an error if the search operation fails.
    pub fn get_by_ids(&self, lookups: &[DocLookup<'_>]) -> Result<Vec<Option<SearchResult>>> {
        if lookups.is_empty() {
            return Ok(Vec::new());
        }

        let searcher = self.reader.searcher();
        let (id_field, text_field, _prefix_field, type_field, created_at_field, metadata_field) =
            self.get_fields();

        let Some(query) = build_lookup_query(id_field, type_field, lookups) else {
            return Ok(vec![None; lookups.len()]);
        };

        let max_docs = lookups
            .iter()
            .map(|lookup| {
                if lookup.doc_type.is_some() {
                    1
                } else {
                    MAX_DOC_TYPES
                }
            })
            .sum::<usize>()
            .max(lookups.len());

        let top_docs = searcher.search(&query, &TopDocs::with_limit(max_docs))?;

        // Single map: id -> (doc_type -> result). Eliminates redundant clones.
        let mut results_by_type: HashMap<String, HashMap<String, SearchResult>> =
            HashMap::with_capacity(lookups.len());

        for (_score, doc_address) in top_docs {
            let doc: TantivyDocument = searcher.doc(doc_address)?;
            let (result, doc_type) = doc_to_search_result(
                &doc,
                id_field,
                text_field,
                type_field,
                created_at_field,
                metadata_field,
            );

            results_by_type
                .entry(result.id.clone())
                .or_default()
                .entry(doc_type)
                .or_insert(result);
        }

        let mut ordered = Vec::with_capacity(lookups.len());
        for lookup in lookups {
            let result = results_by_type.get(lookup.id).and_then(|by_type| {
                lookup.doc_type.map_or_else(
                    // No specific type requested: return any result for this id
                    || by_type.values().next().cloned(),
                    // Specific type requested: look it up
                    |doc_type| by_type.get(doc_type).cloned(),
                )
            });
            ordered.push(result);
        }

        Ok(ordered)
    }

    fn get_by_id_impl(&self, doc_id: &str, doc_type: Option<&str>) -> Result<Option<SearchResult>> {
        let searcher = self.reader.searcher();
        let (id_field, text_field, _prefix_field, type_field, created_at_field, metadata_field) =
            self.get_fields();

        let id_query = TermQuery::new(
            Term::from_field_text(id_field, doc_id),
            IndexRecordOption::Basic,
        );

        let query: Box<dyn Query> = if let Some(doc_type) = doc_type {
            let type_query = TermQuery::new(
                Term::from_field_text(type_field, doc_type),
                IndexRecordOption::Basic,
            );
            Box::new(BooleanQuery::new(vec![
                (Occur::Must, Box::new(id_query)),
                (Occur::Must, Box::new(type_query)),
            ]))
        } else {
            Box::new(id_query)
        };

        // Execute search (limit 1 since IDs should be unique per type)
        let top_docs = searcher.search(&query, &TopDocs::with_limit(1))?;

        if let Some((_score, doc_address)) = top_docs.into_iter().next() {
            let doc: TantivyDocument = searcher.doc(doc_address)?;

            let id = doc
                .get_first(id_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let text = doc
                .get_first(text_field)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            let doc_type_str = doc
                .get_first(type_field)
                .and_then(|v| v.as_str())
                .unwrap_or("tweet");

            let created_at_ts = doc
                .get_first(created_at_field)
                .and_then(|v| v.as_i64())
                .unwrap_or(0);

            let metadata_str = doc
                .get_first(metadata_field)
                .and_then(|v| v.as_str())
                .unwrap_or("{}");

            let result_type = match doc_type_str {
                "like" => SearchResultType::Like,
                "dm" => SearchResultType::DirectMessage,
                "grok" => SearchResultType::GrokMessage,
                _ => SearchResultType::Tweet,
            };

            Ok(Some(SearchResult {
                result_type,
                id,
                text,
                created_at: DateTime::from_timestamp(created_at_ts, 0).unwrap_or_else(epoch_utc),
                score: 1.0, // ID lookup has no relevance score
                highlights: vec![],
                metadata: parse_metadata(metadata_str),
                rerank_score: None,
            }))
        } else {
            Ok(None)
        }
    }

    /// Run Tantivy index health checks for `xf doctor`.
    #[must_use]
    pub fn index_health_checks(&self, storage: &Storage) -> Vec<HealthCheck> {
        vec![
            self.check_index_directory(),
            self.check_index_version(),
            self.check_segment_count(),
            self.check_document_count(storage),
            self.check_sample_query(),
            self.check_index_size(),
        ]
    }

    fn check_index_directory(&self) -> HealthCheck {
        let Some(index_path) = self.index_path.as_deref() else {
            return HealthCheck {
                category: CheckCategory::Index,
                name: "Index Directory".to_string(),
                status: CheckStatus::Warning,
                message: "In-memory index; no directory to inspect".to_string(),
                suggestion: None,
            };
        };

        if !index_path.exists() {
            return HealthCheck {
                category: CheckCategory::Index,
                name: "Index Directory".to_string(),
                status: CheckStatus::Error,
                message: format!("Index not found at {}", index_path.display()),
                suggestion: Some("Run 'xf index' to create the index".to_string()),
            };
        }

        if !index_path.is_dir() {
            return HealthCheck {
                category: CheckCategory::Index,
                name: "Index Directory".to_string(),
                status: CheckStatus::Error,
                message: format!("Index path is not a directory: {}", index_path.display()),
                suggestion: Some("Run 'xf index --force' to rebuild the index".to_string()),
            };
        }

        let meta_path = index_path.join("meta.json");
        if !meta_path.exists() {
            return HealthCheck {
                category: CheckCategory::Index,
                name: "Index Directory".to_string(),
                status: CheckStatus::Error,
                message: "Missing meta.json - index may be corrupted".to_string(),
                suggestion: Some("Run 'xf index --force' to rebuild the index".to_string()),
            };
        }

        HealthCheck {
            category: CheckCategory::Index,
            name: "Index Directory".to_string(),
            status: CheckStatus::Pass,
            message: format!("Found at {}", index_path.display()),
            suggestion: None,
        }
    }

    fn check_index_version(&self) -> HealthCheck {
        match self.index.load_metas() {
            Ok(_) => HealthCheck {
                category: CheckCategory::Index,
                name: "Index Version".to_string(),
                status: CheckStatus::Pass,
                message: format!("Compatible with {}", tantivy::version_string()),
                suggestion: None,
            },
            Err(err) => HealthCheck {
                category: CheckCategory::Index,
                name: "Index Version".to_string(),
                status: CheckStatus::Error,
                message: format!("Index metadata unreadable: {err}"),
                suggestion: Some("Run 'xf index --force' to rebuild the index".to_string()),
            },
        }
    }

    fn check_segment_count(&self) -> HealthCheck {
        let segment_count = self.reader.searcher().segment_readers().len();
        let (status, suggestion) = if segment_count == 0 {
            (
                CheckStatus::Warning,
                Some("Run 'xf index --force' to rebuild the index".to_string()),
            )
        } else if segment_count <= 10 {
            (CheckStatus::Pass, None)
        } else {
            (
                CheckStatus::Warning,
                Some("Run 'xf index --force' to rebuild and compact the index".to_string()),
            )
        };

        HealthCheck {
            category: CheckCategory::Index,
            name: "Segment Count".to_string(),
            status,
            message: format!("{segment_count} segments"),
            suggestion,
        }
    }

    fn check_document_count(&self, storage: &Storage) -> HealthCheck {
        let index_count = i64::try_from(self.reader.searcher().num_docs()).unwrap_or(i64::MAX);

        match storage.indexable_document_count() {
            Ok(db_count) => {
                let diff = (index_count - db_count).abs();
                let percent = if db_count > 0 {
                    diff.saturating_mul(100) / db_count
                } else {
                    0
                };

                let status = if diff == 0 {
                    CheckStatus::Pass
                } else if percent <= 1 || diff <= 10 {
                    CheckStatus::Warning
                } else {
                    CheckStatus::Error
                };

                let suggestion = if diff == 0 {
                    None
                } else {
                    Some(
                        "Run 'xf index --force' to rebuild the index (ignore if you skipped data types)"
                            .to_string(),
                    )
                };

                HealthCheck {
                    category: CheckCategory::Index,
                    name: "Document Count".to_string(),
                    status,
                    message: format!(
                        "Index: {index_count}, DB indexable: {db_count} (diff: {diff})"
                    ),
                    suggestion,
                }
            }
            Err(err) => HealthCheck {
                category: CheckCategory::Index,
                name: "Document Count".to_string(),
                status: CheckStatus::Error,
                message: format!("Failed to read DB counts: {err}"),
                suggestion: Some("Run 'xf doctor' after fixing database errors".to_string()),
            },
        }
    }

    fn check_sample_query(&self) -> HealthCheck {
        let start = Instant::now();
        let result = self.search("test", None, 1);
        let duration_ms = start.elapsed().as_millis();

        match result {
            Ok(_) => {
                let status = if duration_ms < 10 {
                    CheckStatus::Pass
                } else {
                    CheckStatus::Warning
                };

                let suggestion = if duration_ms >= 10 {
                    Some("Consider rebuilding the index with 'xf index --force'".to_string())
                } else {
                    None
                };

                HealthCheck {
                    category: CheckCategory::Index,
                    name: "Sample Query".to_string(),
                    status,
                    message: format!("{duration_ms}ms"),
                    suggestion,
                }
            }
            Err(err) => HealthCheck {
                category: CheckCategory::Index,
                name: "Sample Query".to_string(),
                status: CheckStatus::Error,
                message: format!("Query failed: {err}"),
                suggestion: Some("Index may be corrupted. Try 'xf index --force'".to_string()),
            },
        }
    }

    fn check_index_size(&self) -> HealthCheck {
        let Some(index_path) = self.index_path.as_deref() else {
            return HealthCheck {
                category: CheckCategory::Index,
                name: "Index Size".to_string(),
                status: CheckStatus::Warning,
                message: "In-memory index; size unavailable".to_string(),
                suggestion: None,
            };
        };

        match directory_size_bytes(index_path) {
            Ok(size_bytes) => {
                let is_large = size_bytes > LARGE_INDEX_BYTES;
                let status = if is_large {
                    CheckStatus::Warning
                } else {
                    CheckStatus::Pass
                };
                let suggestion = if is_large {
                    Some("Large index. Consider 'xf index --force' to rebuild".to_string())
                } else {
                    None
                };

                HealthCheck {
                    category: CheckCategory::Index,
                    name: "Index Size".to_string(),
                    status,
                    message: format_bytes(size_bytes),
                    suggestion,
                }
            }
            Err(err) => HealthCheck {
                category: CheckCategory::Index,
                name: "Index Size".to_string(),
                status: CheckStatus::Error,
                message: format!("Failed to read index size: {err}"),
                suggestion: Some("Check index directory permissions".to_string()),
            },
        }
    }

    /// Get document count.
    #[must_use]
    pub fn doc_count(&self) -> u64 {
        self.reader.searcher().num_docs()
    }

    /// Delete all documents and reset the index.
    ///
    /// # Errors
    ///
    /// Returns an error if the index cannot be cleared or committed.
    pub fn clear(&self) -> Result<()> {
        let mut writer = self.writer(50_000_000)?;
        writer.delete_all_documents()?;
        writer.commit()?;
        self.reload()?;
        Ok(())
    }
}

fn directory_size_bytes(path: &Path) -> std::io::Result<u64> {
    let mut total = 0u64;
    let mut stack = vec![path.to_path_buf()];

    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(dir)? {
            let entry = entry?;
            let metadata = entry.metadata()?;
            if metadata.is_dir() {
                stack.push(entry.path());
            } else if metadata.is_file() {
                total = total.saturating_add(metadata.len());
            }
        }
    }

    Ok(total)
}

/// Generate prefix terms for edge n-gram style matching.
/// Uses character count (not byte count) to properly handle UTF-8.
fn generate_prefixes(text: &str) -> String {
    // Estimate capacity: ~7 prefixes per word on average, ~9 chars each
    let word_count_estimate = text.split(|c: char| !c.is_alphanumeric()).take(100).count();
    let capacity_estimate = word_count_estimate * 7 * 9;

    let mut result = String::with_capacity(capacity_estimate);
    let mut chars_buf: Vec<char> = Vec::with_capacity(32);

    for word in text
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.chars().count() >= 2)
        .take(100)
    {
        // Reuse char buffer instead of allocating new String per word
        chars_buf.clear();
        chars_buf.extend(word.chars().flat_map(char::to_lowercase));
        let char_count = chars_buf.len();

        // Generate 2-char to 15-char prefixes (by character count)
        for len in 2..=char_count.min(15) {
            if !result.is_empty() {
                result.push(' ');
            }
            // Write directly to result instead of allocating intermediate String
            for &c in &chars_buf[..len] {
                result.push(c);
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::DirectMessage;

    fn create_test_tweet(id: &str, text: &str) -> Tweet {
        Tweet {
            id: id.to_string(),
            created_at: Utc::now(),
            full_text: text.to_string(),
            source: Some("test".to_string()),
            favorite_count: 0,
            retweet_count: 0,
            lang: Some("en".to_string()),
            in_reply_to_status_id: None,
            in_reply_to_user_id: None,
            in_reply_to_screen_name: None,
            is_retweet: false,
            hashtags: vec![],
            user_mentions: vec![],
            urls: vec![],
            media: vec![],
        }
    }

    fn create_test_like(tweet_id: &str, text: Option<&str>) -> Like {
        Like {
            tweet_id: tweet_id.to_string(),
            full_text: text.map(str::to_string),
            expanded_url: None,
        }
    }

    fn create_test_grok_message(chat_id: &str, message: &str) -> GrokMessage {
        GrokMessage {
            chat_id: chat_id.to_string(),
            message: message.to_string(),
            sender: "user".to_string(),
            created_at: Utc::now(),
            grok_mode: None,
        }
    }

    #[test]
    fn test_generate_prefixes_punctuation() {
        let text = "hello,world";
        let prefixes = generate_prefixes(text);
        assert!(prefixes.contains("he"));
        assert!(prefixes.contains("hello"));
        assert!(prefixes.contains("wo"));
        assert!(prefixes.contains("world"));
    }

    #[test]
    fn test_generate_prefixes() {
        let text = "hello world";
        let prefixes = generate_prefixes(text);
        assert!(prefixes.contains("he"));
        assert!(prefixes.contains("hel"));
        assert!(prefixes.contains("hell"));
        assert!(prefixes.contains("hello"));
        assert!(prefixes.contains("wo"));
        assert!(prefixes.contains("wor"));
        assert!(prefixes.contains("worl"));
        assert!(prefixes.contains("world"));
    }

    #[test]
    fn test_generate_prefixes_short_words() {
        // Words shorter than 2 chars should be skipped
        let text = "a b c hello";
        let prefixes = generate_prefixes(text);
        assert!(!prefixes.contains('a'));
        assert!(!prefixes.contains('b'));
        assert!(!prefixes.contains('c'));
        assert!(prefixes.contains("he"));
    }

    #[test]
    fn test_generate_prefixes_empty() {
        let text = "";
        let prefixes = generate_prefixes(text);
        assert!(prefixes.is_empty());
    }

    #[test]
    fn test_generate_prefixes_long_word() {
        // Prefixes should be capped at 15 chars
        let text = "supercalifragilisticexpialidocious";
        let prefixes = generate_prefixes(text);
        assert!(prefixes.contains("su"));
        assert!(prefixes.contains("supercalifragil")); // 15 chars
        // Should not contain full word (>15 chars)
    }

    #[test]
    fn test_doc_type_as_str() {
        assert_eq!(DocType::Tweet.as_str(), "tweet");
        assert_eq!(DocType::Like.as_str(), "like");
        assert_eq!(DocType::DirectMessage.as_str(), "dm");
        assert_eq!(DocType::GrokMessage.as_str(), "grok");
    }

    #[test]
    fn test_doc_type_from_str() {
        assert_eq!(DocType::from_str("tweet"), Some(DocType::Tweet));
        assert_eq!(DocType::from_str("like"), Some(DocType::Like));
        assert_eq!(DocType::from_str("dm"), Some(DocType::DirectMessage));
        assert_eq!(DocType::from_str("grok"), Some(DocType::GrokMessage));
        assert_eq!(DocType::from_str("invalid"), None);
    }

    #[test]
    fn test_search_engine_memory() {
        let engine = SearchEngine::open_memory().unwrap();
        assert_eq!(engine.doc_count(), 0);
    }

    #[test]
    fn test_search_engine_index_and_search() {
        let engine = SearchEngine::open_memory().unwrap();
        let mut writer = engine.writer(15_000_000).unwrap();

        let tweets = vec![create_test_tweet("123", "Hello world this is a test tweet")];

        engine.index_tweets(&mut writer, &tweets).unwrap();
        writer.commit().unwrap();
        engine.reload().unwrap();

        let results = engine.search("hello", None, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "123");
    }

    #[test]
    fn test_search_engine_empty_query_returns_all() {
        let engine = SearchEngine::open_memory().unwrap();
        let mut writer = engine.writer(15_000_000).unwrap();

        let tweets = vec![
            create_test_tweet("1", "First tweet"),
            create_test_tweet("2", "Second tweet"),
        ];

        engine.index_tweets(&mut writer, &tweets).unwrap();
        writer.commit().unwrap();
        engine.reload().unwrap();

        let results = engine.search("", None, 10).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_engine_multiple_tweets() {
        let engine = SearchEngine::open_memory().unwrap();
        let mut writer = engine.writer(15_000_000).unwrap();

        let tweets = vec![
            create_test_tweet("1", "Rust programming language is great"),
            create_test_tweet("2", "Python is also a programming language"),
            create_test_tweet("3", "Hello world example"),
        ];

        engine.index_tweets(&mut writer, &tweets).unwrap();
        writer.commit().unwrap();
        engine.reload().unwrap();

        // Search for "programming" should find 2 tweets
        let results = engine.search("programming", None, 10).unwrap();
        assert_eq!(results.len(), 2);

        // Search for "rust" should find 1 tweet
        let results = engine.search("rust", None, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "1");
    }

    #[test]
    fn test_search_engine_prefix_matching() {
        let engine = SearchEngine::open_memory().unwrap();
        let mut writer = engine.writer(15_000_000).unwrap();

        let tweets = vec![create_test_tweet("1", "Hello world example")];
        engine.index_tweets(&mut writer, &tweets).unwrap();
        writer.commit().unwrap();
        engine.reload().unwrap();

        // Prefix matches should work on the edge n-gram field.
        let results = engine.search("he", None, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "1");

        let results = engine.search("wor", None, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "1");

        let results = engine.search("xq", None, 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_engine_type_filter() {
        let engine = SearchEngine::open_memory().unwrap();
        let mut writer = engine.writer(15_000_000).unwrap();

        // Index tweets and likes
        let tweets = vec![create_test_tweet("tweet1", "Hello world tweet")];
        let likes = vec![create_test_like("like1", Some("Hello world like"))];

        engine.index_tweets(&mut writer, &tweets).unwrap();
        engine.index_likes(&mut writer, &likes).unwrap();
        writer.commit().unwrap();
        engine.reload().unwrap();

        // Search without filter should find both
        let results = engine.search("hello", None, 10).unwrap();
        assert_eq!(results.len(), 2);

        // Search with tweet filter should find only tweets
        let results = engine.search("hello", Some(&[DocType::Tweet]), 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].result_type, SearchResultType::Tweet);

        // Search with like filter should find only likes
        let results = engine.search("hello", Some(&[DocType::Like]), 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].result_type, SearchResultType::Like);
    }

    #[test]
    fn test_search_engine_limit() {
        let engine = SearchEngine::open_memory().unwrap();
        let mut writer = engine.writer(15_000_000).unwrap();

        let tweets: Vec<Tweet> = (0..10)
            .map(|i| create_test_tweet(&format!("{i}"), "common search term"))
            .collect();

        engine.index_tweets(&mut writer, &tweets).unwrap();
        writer.commit().unwrap();
        engine.reload().unwrap();

        // Limit to 5 results
        let results = engine.search("common", None, 5).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_search_engine_no_results() {
        let engine = SearchEngine::open_memory().unwrap();
        let mut writer = engine.writer(15_000_000).unwrap();

        let tweets = vec![create_test_tweet("1", "Hello world")];
        engine.index_tweets(&mut writer, &tweets).unwrap();
        writer.commit().unwrap();
        engine.reload().unwrap();

        // Search for something not in the index
        let results = engine.search("nonexistent", None, 10).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_engine_index_likes() {
        let engine = SearchEngine::open_memory().unwrap();
        let mut writer = engine.writer(15_000_000).unwrap();

        let likes = vec![
            create_test_like("like1", Some("Great Rust content")),
            create_test_like("like2", None), // Likes without text should be skipped
        ];

        let count = engine.index_likes(&mut writer, &likes).unwrap();
        writer.commit().unwrap();
        engine.reload().unwrap();

        assert_eq!(count, 1); // Only one like has text

        let results = engine.search("rust", None, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].result_type, SearchResultType::Like);
    }

    #[test]
    fn test_search_engine_index_dms() {
        let engine = SearchEngine::open_memory().unwrap();
        let mut writer = engine.writer(15_000_000).unwrap();

        let conversations = vec![DmConversation {
            conversation_id: "conv1".to_string(),
            messages: vec![
                DirectMessage {
                    id: "dm1".to_string(),
                    conversation_id: "conv1".to_string(),
                    sender_id: "user1".to_string(),
                    recipient_id: "user2".to_string(),
                    text: "Hello from direct message".to_string(),
                    created_at: Utc::now(),
                    urls: vec![],
                    media_urls: vec![],
                },
                DirectMessage {
                    id: "dm2".to_string(),
                    conversation_id: "conv1".to_string(),
                    sender_id: "user2".to_string(),
                    recipient_id: "user1".to_string(),
                    text: "Reply to direct message".to_string(),
                    created_at: Utc::now(),
                    urls: vec![],
                    media_urls: vec![],
                },
            ],
        }];

        let count = engine.index_dms(&mut writer, &conversations).unwrap();
        writer.commit().unwrap();
        engine.reload().unwrap();

        assert_eq!(count, 2);

        let results = engine.search("direct", None, 10).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_engine_index_grok() {
        let engine = SearchEngine::open_memory().unwrap();
        let mut writer = engine.writer(15_000_000).unwrap();

        let messages = vec![
            create_test_grok_message("chat1", "What is artificial intelligence?"),
            create_test_grok_message("chat1", "AI is a field of computer science"),
        ];

        let count = engine.index_grok_messages(&mut writer, &messages).unwrap();
        writer.commit().unwrap();
        engine.reload().unwrap();

        assert_eq!(count, 2);

        let results = engine.search("artificial", None, 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].result_type, SearchResultType::GrokMessage);
    }

    #[test]
    fn test_search_engine_clear() {
        let engine = SearchEngine::open_memory().unwrap();

        // Use a scope to ensure writer is dropped before clear
        {
            let mut writer = engine.writer(15_000_000).unwrap();
            let tweets = vec![create_test_tweet("1", "Hello world")];
            engine.index_tweets(&mut writer, &tweets).unwrap();
            writer.commit().unwrap();
        }
        engine.reload().unwrap();

        assert_eq!(engine.doc_count(), 1);

        engine.clear().unwrap();
        assert_eq!(engine.doc_count(), 0);
    }

    #[test]
    fn test_search_engine_doc_count() {
        let engine = SearchEngine::open_memory().unwrap();
        let mut writer = engine.writer(15_000_000).unwrap();

        assert_eq!(engine.doc_count(), 0);

        let tweets = vec![
            create_test_tweet("1", "Tweet one"),
            create_test_tweet("2", "Tweet two"),
        ];
        engine.index_tweets(&mut writer, &tweets).unwrap();
        writer.commit().unwrap();
        engine.reload().unwrap();

        assert_eq!(engine.doc_count(), 2);
    }

    #[test]
    fn test_search_result_metadata() {
        let engine = SearchEngine::open_memory().unwrap();
        let mut writer = engine.writer(15_000_000).unwrap();

        let tweets = vec![Tweet {
            id: "123".to_string(),
            created_at: Utc::now(),
            full_text: "Hello world".to_string(),
            source: Some("Web".to_string()),
            favorite_count: 10,
            retweet_count: 5,
            lang: Some("en".to_string()),
            in_reply_to_status_id: None,
            in_reply_to_user_id: None,
            in_reply_to_screen_name: Some("someone".to_string()),
            is_retweet: false,
            hashtags: vec!["test".to_string()],
            user_mentions: vec![],
            urls: vec![],
            media: vec![],
        }];
        engine.index_tweets(&mut writer, &tweets).unwrap();
        writer.commit().unwrap();
        engine.reload().unwrap();

        let results = engine.search("hello", None, 10).unwrap();
        assert_eq!(results.len(), 1);

        let metadata = &results[0].metadata;
        assert_eq!(metadata["favorite_count"], 10);
        assert_eq!(metadata["retweet_count"], 5);
        assert_eq!(metadata["in_reply_to"], "someone");
        assert_eq!(metadata["source"], "Web");
    }

    #[test]
    fn test_search_with_multiple_type_filters() {
        let engine = SearchEngine::open_memory().unwrap();
        let mut writer = engine.writer(15_000_000).unwrap();

        let tweets = vec![create_test_tweet("tweet1", "common search term")];
        let likes = vec![create_test_like("like1", Some("common search term"))];
        let grok = vec![create_test_grok_message("chat1", "common search term")];

        engine.index_tweets(&mut writer, &tweets).unwrap();
        engine.index_likes(&mut writer, &likes).unwrap();
        engine.index_grok_messages(&mut writer, &grok).unwrap();
        writer.commit().unwrap();
        engine.reload().unwrap();

        // Filter by tweet and like (exclude grok)
        let results = engine
            .search("common", Some(&[DocType::Tweet, DocType::Like]), 10)
            .unwrap();
        assert_eq!(results.len(), 2);

        let has_tweet = results
            .iter()
            .any(|r| r.result_type == SearchResultType::Tweet);
        let has_like = results
            .iter()
            .any(|r| r.result_type == SearchResultType::Like);
        let has_grok = results
            .iter()
            .any(|r| r.result_type == SearchResultType::GrokMessage);

        assert!(has_tweet);
        assert!(has_like);
        assert!(!has_grok);
    }

    #[test]
    fn test_get_by_id_and_type_disambiguates() {
        let engine = SearchEngine::open_memory().unwrap();
        let mut writer = engine.writer(15_000_000).unwrap();

        let tweets = vec![create_test_tweet("42", "tweet text")];
        let likes = vec![create_test_like("42", Some("like text"))];

        engine.index_tweets(&mut writer, &tweets).unwrap();
        engine.index_likes(&mut writer, &likes).unwrap();
        writer.commit().unwrap();
        engine.reload().unwrap();

        let tweet = engine
            .get_by_id_and_type("42", DocType::Tweet.as_str())
            .unwrap()
            .unwrap();
        assert_eq!(tweet.result_type, SearchResultType::Tweet);
        assert_eq!(tweet.text, "tweet text");

        let like = engine
            .get_by_id_and_type("42", DocType::Like.as_str())
            .unwrap()
            .unwrap();
        assert_eq!(like.result_type, SearchResultType::Like);
        assert_eq!(like.text, "like text");
    }

    #[test]
    fn test_get_by_ids_empty() {
        let engine = SearchEngine::open_memory().unwrap();
        let results = engine.get_by_ids(&[]).unwrap();
        assert!(results.is_empty());
    }

    #[test]
    fn test_get_by_ids_preserves_order_and_duplicates() {
        let engine = SearchEngine::open_memory().unwrap();
        let mut writer = engine.writer(15_000_000).unwrap();

        let tweets = vec![
            create_test_tweet("1", "tweet one"),
            create_test_tweet("2", "tweet two"),
        ];

        engine.index_tweets(&mut writer, &tweets).unwrap();
        writer.commit().unwrap();
        engine.reload().unwrap();

        let lookups = vec![
            DocLookup::new("2"),
            DocLookup::new("1"),
            DocLookup::new("1"),
        ];

        let results = engine.get_by_ids(&lookups).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].as_ref().unwrap().id, "2");
        assert_eq!(results[1].as_ref().unwrap().id, "1");
        assert_eq!(results[2].as_ref().unwrap().id, "1");
    }

    #[test]
    fn test_get_by_ids_missing_ids() {
        let engine = SearchEngine::open_memory().unwrap();
        let mut writer = engine.writer(15_000_000).unwrap();

        let tweets = vec![create_test_tweet("1", "tweet one")];
        engine.index_tweets(&mut writer, &tweets).unwrap();
        writer.commit().unwrap();
        engine.reload().unwrap();

        let lookups = vec![DocLookup::new("1"), DocLookup::new("missing")];
        let results = engine.get_by_ids(&lookups).unwrap();

        assert!(results[0].is_some());
        assert!(results[1].is_none());
    }

    #[test]
    fn test_get_by_ids_mixed_types() {
        let engine = SearchEngine::open_memory().unwrap();
        let mut writer = engine.writer(15_000_000).unwrap();

        let tweets = vec![
            create_test_tweet("42", "tweet 42"),
            create_test_tweet("99", "tweet 99"),
        ];
        let likes = vec![create_test_like("42", Some("like text"))];

        engine.index_tweets(&mut writer, &tweets).unwrap();
        engine.index_likes(&mut writer, &likes).unwrap();
        writer.commit().unwrap();
        engine.reload().unwrap();

        let lookups = vec![
            DocLookup::with_type("42", DocType::Like.as_str()),
            DocLookup::with_type("42", DocType::Tweet.as_str()),
            DocLookup::with_type("99", DocType::Tweet.as_str()),
        ];

        let results = engine.get_by_ids(&lookups).unwrap();
        assert_eq!(results.len(), 3);
        assert_eq!(
            results[0].as_ref().unwrap().result_type,
            SearchResultType::Like
        );
        assert_eq!(results[0].as_ref().unwrap().text, "like text");
        assert_eq!(
            results[1].as_ref().unwrap().result_type,
            SearchResultType::Tweet
        );
        assert_eq!(results[1].as_ref().unwrap().text, "tweet 42");
        assert_eq!(
            results[2].as_ref().unwrap().result_type,
            SearchResultType::Tweet
        );
        assert_eq!(results[2].as_ref().unwrap().text, "tweet 99");
    }

    #[test]
    fn test_get_by_ids_untyped_handles_multiple_doc_types() {
        let engine = SearchEngine::open_memory().unwrap();
        let mut writer = engine.writer(15_000_000).unwrap();

        let tweets = vec![
            create_test_tweet("42", "tweet 42"),
            create_test_tweet("99", "tweet 99"),
        ];
        let likes = vec![create_test_like("42", Some("like text"))];

        engine.index_tweets(&mut writer, &tweets).unwrap();
        engine.index_likes(&mut writer, &likes).unwrap();
        writer.commit().unwrap();
        engine.reload().unwrap();

        let lookups = vec![DocLookup::new("42"), DocLookup::new("99")];
        let results = engine.get_by_ids(&lookups).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].is_some());
        assert_eq!(results[1].as_ref().unwrap().id, "99");
    }

    #[test]
    fn test_search_engine_highlights() {
        let engine = SearchEngine::open_memory().unwrap();
        let mut writer = engine.writer(15_000_000).unwrap();

        let tweets = vec![create_test_tweet(
            "123",
            "The Rust programming language is fast and memory-safe",
        )];

        engine.index_tweets(&mut writer, &tweets).unwrap();
        writer.commit().unwrap();
        engine.reload().unwrap();

        let results = engine.search("rust", None, 10).unwrap();
        assert_eq!(results.len(), 1);

        // Highlights should contain the search term wrapped in <b> tags
        assert!(!results[0].highlights.is_empty());
        let highlight = &results[0].highlights[0];
        assert!(highlight.contains("<b>"));
        assert!(highlight.contains("</b>"));
        // The highlight should contain "Rust" (case-insensitive match)
        assert!(highlight.to_lowercase().contains("rust"));
    }
}
