//! `SQLite` storage for X archive data.
//!
//! Provides persistent storage with optimized schema for fast queries.

use crate::doctor::{CheckCategory, CheckStatus, HealthCheck, TableStat};
use crate::model::{
    ArchiveInfo, ArchiveStats, Block, Bookmark, DirectMessage, DmConversation,
    DmConversationSummary, Follower, Following, GrokMessage, Like, Mute, Tweet,
};
use crate::{format_bytes_i64, format_number};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use rusqlite::{Connection, params};
use std::collections::HashMap;
use std::path::Path;
use tracing::info;

const SCHEMA_VERSION: i32 = 6;
// SQLite default limit on host parameters is usually 999 or 32766.
// We use a safe batch size to avoid "too many SQL variables" errors.
const SQLITE_BATCH_SIZE: usize = 900;

const fn epoch_utc() -> DateTime<Utc> {
    DateTime::<Utc>::from_timestamp(0, 0).unwrap()
}

fn parse_rfc3339_or_epoch(value: Option<String>) -> DateTime<Utc> {
    value
        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
        .map_or_else(epoch_utc, |dt| dt.with_timezone(&Utc))
}

fn parse_rfc3339_opt(value: Option<String>) -> Option<DateTime<Utc>> {
    value
        .and_then(|s| DateTime::parse_from_rfc3339(&s).ok())
        .map(|dt| dt.with_timezone(&Utc))
}

/// `SQLite` storage manager
pub struct Storage {
    conn: Connection,
}

/// Embedding record: (doc_id, doc_type, embedding, content_hash).
type EmbeddingRecord = (String, String, Vec<f32>, Option<[u8; 32]>);

/// Summary of FTS rebuild results.
#[derive(Debug, Clone, Copy)]
pub struct FtsRebuildStats {
    pub tweets: usize,
    pub likes: usize,
    pub dms: usize,
    pub grok: usize,
    pub bookmarks: usize,
}

/// Status of an embedding job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbeddingJobStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
}

impl EmbeddingJobStatus {
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::InProgress => "in_progress",
            Self::Completed => "completed",
            Self::Failed => "failed",
        }
    }

    #[must_use]
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Self {
        match s {
            "in_progress" => Self::InProgress,
            "completed" => Self::Completed,
            "failed" => Self::Failed,
            _ => Self::Pending,
        }
    }
}

/// Information about an embedding job.
#[derive(Debug, Clone)]
pub struct EmbeddingJob {
    pub id: i64,
    pub db_path: String,
    pub model_id: String,
    pub status: EmbeddingJobStatus,
    pub total_docs: i64,
    pub completed_docs: i64,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    pub error_message: Option<String>,
}

/// Aggregate counts and date bounds for archive tables.
#[derive(Debug, Clone)]
pub struct AllCounts {
    pub tweets_count: i64,
    pub likes_count: i64,
    pub dms_count: i64,
    pub dm_conversations_count: i64,
    pub followers_count: i64,
    pub following_count: i64,
    pub blocks_count: i64,
    pub mutes_count: i64,
    pub grok_messages_count: i64,
    pub bookmarks_count: i64,
    pub first_tweet_date: Option<DateTime<Utc>>,
    pub last_tweet_date: Option<DateTime<Utc>>,
}

impl Storage {
    /// Open or create the database at the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the database cannot be opened or initialized.
    pub fn open(db_path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(db_path.as_ref()).with_context(|| {
            format!("Failed to open database at {}", db_path.as_ref().display())
        })?;

        // Set pragmas for performance
        conn.execute_batch(
            "
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous = NORMAL;
            PRAGMA foreign_keys = ON;
            PRAGMA cache_size = -64000;
            PRAGMA temp_store = MEMORY;
            ",
        )?;

        let storage = Self { conn };
        storage.migrate()?;
        Ok(storage)
    }

    /// Open an in-memory database (for testing).
    ///
    /// # Errors
    ///
    /// Returns an error if the in-memory database cannot be initialized.
    pub fn open_memory() -> Result<Self> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(
            "
            PRAGMA foreign_keys = ON;
            PRAGMA temp_store = MEMORY;
            ",
        )?;
        let storage = Self { conn };
        storage.migrate()?;
        Ok(storage)
    }

    /// Get a reference to the underlying database connection.
    ///
    /// This is useful for modules that need to execute custom queries.
    #[must_use]
    pub const fn connection(&self) -> &Connection {
        &self.conn
    }

    /// Run database migrations
    fn migrate(&self) -> Result<()> {
        let current_version = self.get_schema_version();

        if current_version < SCHEMA_VERSION {
            info!(
                "Migrating database from version {} to {}",
                current_version, SCHEMA_VERSION
            );

            // Embeddings are derived data. For schema updates we drop/recreate to
            // guarantee a clean, consistent layout.
            if current_version < 3 {
                self.conn.execute("DROP TABLE IF EXISTS embeddings", [])?;
            }

            // v4: Add model_id column to embeddings table BEFORE create_schema(),
            // so the column exists when we try to create indexes on it.
            if (3..4).contains(&current_version) {
                self.migrate_v4_add_model_id()?;
            }

            self.create_schema()?;

            self.set_schema_version(SCHEMA_VERSION)?;
        }

        Ok(())
    }

    fn get_schema_version(&self) -> i32 {
        let result: Result<i32, _> = self.conn.query_row(
            "SELECT value FROM meta WHERE key = 'schema_version'",
            [],
            |row| {
                let value: String = row.get(0)?;
                Ok(value.parse().unwrap_or(0))
            },
        );

        // Treat missing schema table as version 0.
        result.unwrap_or_default()
    }

    fn set_schema_version(&self, version: i32) -> Result<()> {
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES ('schema_version', ?)",
            params![version.to_string()],
        )?;
        Ok(())
    }

    /// Migrate from schema v3 to v4: add model_id column to embeddings.
    fn migrate_v4_add_model_id(&self) -> Result<()> {
        // Check if model_id column already exists (idempotent)
        let has_model_id: bool = self
            .conn
            .prepare("SELECT model_id FROM embeddings LIMIT 0")
            .is_ok();

        if has_model_id {
            return Ok(());
        }

        info!("Migrating embeddings table: adding model_id column");
        self.conn.execute_batch(
            r"
            -- Add model_id column with default value
            ALTER TABLE embeddings ADD COLUMN model_id TEXT NOT NULL DEFAULT 'default';

            -- Drop old primary key index and create new composite one
            DROP INDEX IF EXISTS sqlite_autoindex_embeddings_1;
            CREATE UNIQUE INDEX IF NOT EXISTS idx_embeddings_pk
                ON embeddings(doc_id, doc_type, model_id);

            -- Add model_id index for filtered queries
            CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_id);
            ",
        )?;

        Ok(())
    }

    #[allow(clippy::too_many_lines)]
    fn create_schema(&self) -> Result<()> {
        self.conn.execute_batch(
            r"
            -- Metadata table
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            -- Archive info
            CREATE TABLE IF NOT EXISTS archive_info (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                account_id TEXT NOT NULL,
                username TEXT NOT NULL,
                display_name TEXT,
                archive_size_bytes INTEGER,
                generation_date TEXT,
                is_partial INTEGER DEFAULT 0,
                indexed_at TEXT NOT NULL
            );

            -- Tweets
            CREATE TABLE IF NOT EXISTS tweets (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                full_text TEXT NOT NULL,
                source TEXT,
                favorite_count INTEGER DEFAULT 0,
                retweet_count INTEGER DEFAULT 0,
                lang TEXT,
                in_reply_to_status_id TEXT,
                in_reply_to_user_id TEXT,
                in_reply_to_screen_name TEXT,
                is_retweet INTEGER DEFAULT 0,
                hashtags_json TEXT,
                mentions_json TEXT,
                urls_json TEXT,
                media_json TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_tweets_created_at ON tweets(created_at);
            CREATE INDEX IF NOT EXISTS idx_tweets_in_reply_to ON tweets(in_reply_to_status_id);

            -- Likes
            CREATE TABLE IF NOT EXISTS likes (
                tweet_id TEXT PRIMARY KEY,
                full_text TEXT,
                expanded_url TEXT
            );

            -- DM Conversations
            CREATE TABLE IF NOT EXISTS dm_conversations (
                conversation_id TEXT PRIMARY KEY,
                participant_ids TEXT,
                message_count INTEGER DEFAULT 0,
                first_message_at TEXT,
                last_message_at TEXT
            );

            -- Direct Messages
            CREATE TABLE IF NOT EXISTS direct_messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                sender_id TEXT NOT NULL,
                recipient_id TEXT NOT NULL,
                text TEXT NOT NULL,
                created_at TEXT NOT NULL,
                urls_json TEXT,
                media_urls_json TEXT,
                FOREIGN KEY (conversation_id) REFERENCES dm_conversations(conversation_id)
            );
            CREATE INDEX IF NOT EXISTS idx_dm_conversation ON direct_messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_dm_created_at ON direct_messages(created_at);

            -- Followers
            CREATE TABLE IF NOT EXISTS followers (
                account_id TEXT PRIMARY KEY,
                user_link TEXT
            );

            -- Following
            CREATE TABLE IF NOT EXISTS following (
                account_id TEXT PRIMARY KEY,
                user_link TEXT
            );

            -- Blocks
            CREATE TABLE IF NOT EXISTS blocks (
                account_id TEXT PRIMARY KEY,
                user_link TEXT
            );

            -- Mutes
            CREATE TABLE IF NOT EXISTS mutes (
                account_id TEXT PRIMARY KEY,
                user_link TEXT
            );

            -- Bookmarks
            CREATE TABLE IF NOT EXISTS bookmarks (
                tweet_id TEXT PRIMARY KEY,
                full_text TEXT
            );

            -- Grok messages
            CREATE TABLE IF NOT EXISTS grok_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id TEXT NOT NULL,
                message TEXT NOT NULL,
                sender TEXT NOT NULL,
                created_at TEXT NOT NULL,
                grok_mode TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_grok_chat_id ON grok_messages(chat_id);
            CREATE INDEX IF NOT EXISTS idx_grok_created_at ON grok_messages(created_at);

            -- Full-text search virtual tables (standalone, not content-synced)
            CREATE VIRTUAL TABLE IF NOT EXISTS fts_tweets USING fts5(
                tweet_id,
                full_text
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS fts_likes USING fts5(
                tweet_id,
                full_text
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS fts_dms USING fts5(
                dm_id,
                text
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS fts_grok USING fts5(
                grok_id,
                message
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS fts_bookmarks USING fts5(
                tweet_id,
                full_text
            );

            -- Embeddings for semantic search
            CREATE TABLE IF NOT EXISTS embeddings (
                doc_id TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                embedding BLOB NOT NULL,
                content_hash BLOB,
                created_at TEXT NOT NULL,
                model_id TEXT NOT NULL DEFAULT 'default',
                PRIMARY KEY (doc_id, doc_type, model_id)
            );
            CREATE INDEX IF NOT EXISTS idx_embeddings_type ON embeddings(doc_type);
            CREATE INDEX IF NOT EXISTS idx_embeddings_hash ON embeddings(content_hash);
            CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_id);

            -- Background embedding job tracking
            CREATE TABLE IF NOT EXISTS embedding_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                db_path TEXT NOT NULL,
                model_id TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                total_docs INTEGER NOT NULL,
                completed_docs INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                started_at TEXT,
                completed_at TEXT,
                error_message TEXT,
                UNIQUE(db_path, model_id)
            );
            CREATE INDEX IF NOT EXISTS idx_embedding_jobs_status ON embedding_jobs(status);
            ",
        )?;

        Ok(())
    }

    /// Store archive info.
    ///
    /// # Errors
    ///
    /// Returns an error if the database insert fails.
    pub fn store_archive_info(&self, info: &ArchiveInfo) -> Result<()> {
        self.conn.execute(
            r"
            INSERT OR REPLACE INTO archive_info
            (id, account_id, username, display_name, archive_size_bytes, generation_date, is_partial, indexed_at)
            VALUES (1, ?, ?, ?, ?, ?, ?, ?)
            ",
            params![
                info.account_id,
                info.username,
                info.display_name,
                info.archive_size_bytes,
                info.generation_date.to_rfc3339(),
                i32::from(info.is_partial),
                Utc::now().to_rfc3339(),
            ],
        )?;
        Ok(())
    }

    /// Retrieve stored archive info.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_archive_info(&self) -> Result<Option<ArchiveInfo>> {
        let result = self.conn.query_row(
            "SELECT account_id, username, display_name, archive_size_bytes, generation_date, is_partial
             FROM archive_info WHERE id = 1",
            [],
            |row| {
                let generation_date_str: String = row.get(4)?;
                let generation_date =
                    chrono::DateTime::parse_from_rfc3339(&generation_date_str).map_or_else(
                        |_| Utc::now(),
                        |dt| dt.with_timezone(&Utc),
                    );

                Ok(ArchiveInfo {
                    account_id: row.get(0)?,
                    username: row.get(1)?,
                    display_name: row.get(2)?,
                    archive_size_bytes: row.get(3)?,
                    generation_date,
                    is_partial: row.get::<_, i32>(5)? != 0,
                })
            },
        );

        match result {
            Ok(info) => Ok(Some(info)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Store tweets in a transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if any tweet insert fails.
    pub fn store_tweets(&mut self, tweets: &[Tweet]) -> Result<usize> {
        let tx = self.conn.transaction()?;
        let mut count = 0;

        {
            // FTS5 doesn't support INSERT OR REPLACE, so we must delete first to avoid duplicates.
            // Batch delete for performance: one DELETE with IN clause instead of N individual DELETEs.
            if !tweets.is_empty() {
                for chunk in tweets.chunks(SQLITE_BATCH_SIZE) {
                    let placeholders: String =
                        chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                    let delete_sql =
                        format!("DELETE FROM fts_tweets WHERE tweet_id IN ({placeholders})");
                    let mut delete_stmt = tx.prepare_cached(&delete_sql)?;
                    delete_stmt.execute(rusqlite::params_from_iter(chunk.iter().map(|t| &t.id)))?;
                }
            }

            let mut stmt = tx.prepare(
                r"
                INSERT OR REPLACE INTO tweets
                (id, created_at, full_text, source, favorite_count, retweet_count, lang,
                 in_reply_to_status_id, in_reply_to_user_id, in_reply_to_screen_name,
                 is_retweet, hashtags_json, mentions_json, urls_json, media_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ",
            )?;
            let mut fts_stmt =
                tx.prepare("INSERT INTO fts_tweets (tweet_id, full_text) VALUES (?, ?)")?;

            for tweet in tweets {
                stmt.execute(params![
                    tweet.id,
                    tweet.created_at.to_rfc3339(),
                    tweet.full_text,
                    tweet.source,
                    tweet.favorite_count,
                    tweet.retweet_count,
                    tweet.lang,
                    tweet.in_reply_to_status_id,
                    tweet.in_reply_to_user_id,
                    tweet.in_reply_to_screen_name,
                    i32::from(tweet.is_retweet),
                    serde_json::to_string(&tweet.hashtags)?,
                    serde_json::to_string(&tweet.user_mentions)?,
                    serde_json::to_string(&tweet.urls)?,
                    serde_json::to_string(&tweet.media)?,
                ])?;
                fts_stmt.execute(params![&tweet.id, &tweet.full_text])?;
                count += 1;
            }
        }

        tx.commit()?;
        info!("Stored {} tweets", count);
        Ok(count)
    }

    /// Store likes in a transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if any like insert fails.
    pub fn store_likes(&mut self, likes: &[Like]) -> Result<usize> {
        let tx = self.conn.transaction()?;
        let mut count = 0;

        {
            // FTS5 batch delete for all likes to avoid stale rows when text is removed.
            let like_ids: Vec<_> = likes.iter().map(|l| l.tweet_id.as_str()).collect();
            if !like_ids.is_empty() {
                for chunk in like_ids.chunks(SQLITE_BATCH_SIZE) {
                    let placeholders: String =
                        chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                    let delete_sql =
                        format!("DELETE FROM fts_likes WHERE tweet_id IN ({placeholders})");
                    let mut delete_stmt = tx.prepare_cached(&delete_sql)?;
                    delete_stmt.execute(rusqlite::params_from_iter(chunk.iter()))?;
                }
            }

            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO likes (tweet_id, full_text, expanded_url) VALUES (?, ?, ?)",
            )?;
            let mut fts_stmt =
                tx.prepare("INSERT INTO fts_likes (tweet_id, full_text) VALUES (?, ?)")?;

            for like in likes {
                stmt.execute(params![like.tweet_id, like.full_text, like.expanded_url])?;
                if let Some(text) = &like.full_text {
                    if !text.is_empty() {
                        fts_stmt.execute(params![&like.tweet_id, text])?;
                    }
                }
                count += 1;
            }
        }

        tx.commit()?;
        info!("Stored {} likes", count);
        Ok(count)
    }

    /// Store DM conversations and messages.
    ///
    /// # Errors
    ///
    /// Returns an error if any conversation or message insert fails.
    pub fn store_dm_conversations(&mut self, conversations: &[DmConversation]) -> Result<usize> {
        let tx = self.conn.transaction()?;
        let mut message_count = 0;

        {
            // Collect all message IDs for batch FTS delete
            let all_msg_ids: Vec<&str> = conversations
                .iter()
                .flat_map(|c| c.messages.iter().map(|m| m.id.as_str()))
                .collect();
            if !all_msg_ids.is_empty() {
                for chunk in all_msg_ids.chunks(SQLITE_BATCH_SIZE) {
                    let placeholders: String =
                        chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                    let delete_sql = format!("DELETE FROM fts_dms WHERE dm_id IN ({placeholders})");
                    let mut delete_stmt = tx.prepare_cached(&delete_sql)?;
                    delete_stmt.execute(rusqlite::params_from_iter(chunk.iter()))?;
                }
            }

            let mut conv_stmt = tx.prepare(
                r"
                INSERT OR REPLACE INTO dm_conversations
                (conversation_id, participant_ids, message_count, first_message_at, last_message_at)
                VALUES (?, ?, ?, ?, ?)
                ",
            )?;

            let mut msg_stmt = tx.prepare(
                r"
                INSERT OR REPLACE INTO direct_messages
                (id, conversation_id, sender_id, recipient_id, text, created_at, urls_json, media_urls_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ",
            )?;

            let mut fts_stmt = tx.prepare("INSERT INTO fts_dms (dm_id, text) VALUES (?, ?)")?;

            for conv in conversations {
                // Get participant IDs and date range
                let mut participant_ids: Vec<&str> = conv
                    .messages
                    .iter()
                    .flat_map(|m| vec![m.sender_id.as_str(), m.recipient_id.as_str()])
                    .collect();
                participant_ids.sort_unstable();
                participant_ids.dedup();

                let first_msg = conv.messages.iter().min_by_key(|m| m.created_at);
                let last_msg = conv.messages.iter().max_by_key(|m| m.created_at);

                conv_stmt.execute(params![
                    conv.conversation_id,
                    participant_ids.join(","),
                    i64::try_from(conv.messages.len()).unwrap_or(i64::MAX),
                    first_msg.map(|m| m.created_at.to_rfc3339()),
                    last_msg.map(|m| m.created_at.to_rfc3339()),
                ])?;

                for msg in &conv.messages {
                    msg_stmt.execute(params![
                        msg.id,
                        conv.conversation_id,
                        msg.sender_id,
                        msg.recipient_id,
                        msg.text,
                        msg.created_at.to_rfc3339(),
                        serde_json::to_string(&msg.urls)?,
                        serde_json::to_string(&msg.media_urls)?,
                    ])?;
                    fts_stmt.execute(params![&msg.id, &msg.text])?;
                    message_count += 1;
                }
            }
        }
        tx.commit()?;
        info!(
            "Stored {} conversations with {} messages",
            conversations.len(),
            message_count
        );
        Ok(message_count)
    }

    /// Store followers.
    ///
    /// # Errors
    ///
    /// Returns an error if any follower insert fails.
    pub fn store_followers(&mut self, followers: &[Follower]) -> Result<usize> {
        let tx = self.conn.transaction()?;
        let mut count = 0;

        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO followers (account_id, user_link) VALUES (?, ?)",
            )?;

            for f in followers {
                stmt.execute(params![f.account_id, f.user_link])?;
                count += 1;
            }
        }

        tx.commit()?;
        info!("Stored {} followers", count);
        Ok(count)
    }

    /// Store following.
    ///
    /// # Errors
    ///
    /// Returns an error if any following insert fails.
    pub fn store_following(&mut self, following: &[Following]) -> Result<usize> {
        let tx = self.conn.transaction()?;
        let mut count = 0;

        {
            let mut stmt = tx.prepare(
                "INSERT OR REPLACE INTO following (account_id, user_link) VALUES (?, ?)",
            )?;

            for f in following {
                stmt.execute(params![f.account_id, f.user_link])?;
                count += 1;
            }
        }

        tx.commit()?;
        info!("Stored {} following", count);
        Ok(count)
    }

    /// Store blocks.
    ///
    /// # Errors
    ///
    /// Returns an error if any block insert fails.
    pub fn store_blocks(&mut self, blocks: &[Block]) -> Result<usize> {
        let tx = self.conn.transaction()?;
        let mut count = 0;

        {
            let mut stmt =
                tx.prepare("INSERT OR REPLACE INTO blocks (account_id, user_link) VALUES (?, ?)")?;

            for b in blocks {
                stmt.execute(params![b.account_id, b.user_link])?;
                count += 1;
            }
        }

        tx.commit()?;
        info!("Stored {} blocks", count);
        Ok(count)
    }

    /// Store mutes.
    ///
    /// # Errors
    ///
    /// Returns an error if any mute insert fails.
    pub fn store_mutes(&mut self, mutes: &[Mute]) -> Result<usize> {
        let tx = self.conn.transaction()?;
        let mut count = 0;

        {
            let mut stmt =
                tx.prepare("INSERT OR REPLACE INTO mutes (account_id, user_link) VALUES (?, ?)")?;

            for m in mutes {
                stmt.execute(params![m.account_id, m.user_link])?;
                count += 1;
            }
        }

        tx.commit()?;
        info!("Stored {} mutes", count);
        Ok(count)
    }

    /// Store Grok messages.
    ///
    /// # Errors
    ///
    /// Returns an error if any Grok message insert fails.
    pub fn store_grok_messages(&mut self, messages: &[GrokMessage]) -> Result<usize> {
        let tx = self.conn.transaction()?;
        let mut count = 0;

        {
            // Clear existing data to ensure FTS and main table correspondence.
            // Unlike tweets/likes/dms which have natural primary keys for INSERT OR REPLACE,
            // grok_messages uses auto-increment, so we clear and repopulate.
            tx.execute("DELETE FROM fts_grok", [])?;
            tx.execute("DELETE FROM grok_messages", [])?;

            let mut stmt = tx.prepare(
                r"
                INSERT INTO grok_messages (chat_id, message, sender, created_at, grok_mode)
                VALUES (?, ?, ?, ?, ?)
                ",
            )?;
            let mut fts_stmt =
                tx.prepare("INSERT INTO fts_grok (grok_id, message) VALUES (?, ?)")?;

            for msg in messages {
                stmt.execute(params![
                    msg.chat_id,
                    msg.message,
                    msg.sender,
                    msg.created_at.to_rfc3339(),
                    msg.grok_mode,
                ])?;
                // Use the auto-generated id as grok_id for proper FTS joining
                let grok_id = tx.last_insert_rowid();
                fts_stmt.execute(params![grok_id.to_string(), &msg.message])?;
                count += 1;
            }
        }

        tx.commit()?;
        info!("Stored {} Grok messages", count);
        Ok(count)
    }

    /// Store bookmarks in a transaction.
    ///
    /// # Errors
    ///
    /// Returns an error if any bookmark insert fails.
    pub fn store_bookmarks(&mut self, bookmarks: &[Bookmark]) -> Result<usize> {
        let tx = self.conn.transaction()?;
        let mut count = 0;

        {
            // FTS5 batch delete for all bookmarks to avoid stale rows.
            let bm_ids: Vec<_> = bookmarks.iter().map(|b| b.tweet_id.as_str()).collect();
            if !bm_ids.is_empty() {
                for chunk in bm_ids.chunks(SQLITE_BATCH_SIZE) {
                    let placeholders: String =
                        chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
                    let delete_sql =
                        format!("DELETE FROM fts_bookmarks WHERE tweet_id IN ({placeholders})");
                    let mut delete_stmt = tx.prepare_cached(&delete_sql)?;
                    delete_stmt.execute(rusqlite::params_from_iter(chunk.iter()))?;
                }
            }

            let mut stmt =
                tx.prepare("INSERT OR REPLACE INTO bookmarks (tweet_id, full_text) VALUES (?, ?)")?;
            let mut fts_stmt =
                tx.prepare("INSERT INTO fts_bookmarks (tweet_id, full_text) VALUES (?, ?)")?;

            for bookmark in bookmarks {
                stmt.execute(params![bookmark.tweet_id, bookmark.full_text])?;
                if let Some(text) = &bookmark.full_text {
                    if !text.is_empty() {
                        fts_stmt.execute(params![&bookmark.tweet_id, text])?;
                    }
                }
                count += 1;
            }
        }

        tx.commit()?;
        info!("Stored {} bookmarks", count);
        Ok(count)
    }

    /// Get archive statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if statistics queries fail.
    pub fn get_stats(&self) -> Result<ArchiveStats> {
        let counts = self.get_all_counts()?;
        let index_built_at = self.get_index_built_at()?.unwrap_or_else(Utc::now);

        Ok(ArchiveStats {
            tweets_count: counts.tweets_count,
            likes_count: counts.likes_count,
            dms_count: counts.dms_count,
            dm_conversations_count: counts.dm_conversations_count,
            followers_count: counts.followers_count,
            following_count: counts.following_count,
            blocks_count: counts.blocks_count,
            mutes_count: counts.mutes_count,
            grok_messages_count: counts.grok_messages_count,
            bookmarks_count: counts.bookmarks_count,
            first_tweet_date: counts.first_tweet_date,
            last_tweet_date: counts.last_tweet_date,
            index_built_at,
        })
    }

    /// Get the index build timestamp from the archive metadata, if present.
    ///
    /// # Errors
    ///
    /// Returns an error if the query fails.
    pub fn get_index_built_at(&self) -> Result<Option<DateTime<Utc>>> {
        let result: std::result::Result<String, _> = self.conn.query_row(
            "SELECT indexed_at FROM archive_info WHERE id = 1",
            [],
            |row| row.get(0),
        );

        match result {
            Ok(ts) => Ok(parse_rfc3339_opt(Some(ts))),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(err) => Err(err.into()),
        }
    }

    /// Get all archive counts and tweet date bounds in a single query.
    ///
    /// # Errors
    ///
    /// Returns an error if the consolidated query fails.
    pub fn get_all_counts(&self) -> Result<AllCounts> {
        let query = r"
            SELECT
                (SELECT COUNT(*) FROM tweets) AS tweets_count,
                (SELECT COUNT(*) FROM likes) AS likes_count,
                (SELECT COUNT(*) FROM direct_messages) AS dms_count,
                (SELECT COUNT(*) FROM dm_conversations) AS dm_conversations_count,
                (SELECT COUNT(*) FROM followers) AS followers_count,
                (SELECT COUNT(*) FROM following) AS following_count,
                (SELECT COUNT(*) FROM blocks) AS blocks_count,
                (SELECT COUNT(*) FROM mutes) AS mutes_count,
                (SELECT COUNT(*) FROM grok_messages) AS grok_messages_count,
                (SELECT COUNT(*) FROM bookmarks) AS bookmarks_count,
                (SELECT MIN(created_at) FROM tweets) AS first_tweet_date,
                (SELECT MAX(created_at) FROM tweets) AS last_tweet_date
        ";

        Ok(self.conn.query_row(query, [], |row| {
            let first_tweet_date: Option<String> = row.get(10)?;
            let last_tweet_date: Option<String> = row.get(11)?;
            Ok(AllCounts {
                tweets_count: row.get(0)?,
                likes_count: row.get(1)?,
                dms_count: row.get(2)?,
                dm_conversations_count: row.get(3)?,
                followers_count: row.get(4)?,
                following_count: row.get(5)?,
                blocks_count: row.get(6)?,
                mutes_count: row.get(7)?,
                grok_messages_count: row.get(8)?,
                bookmarks_count: row.get(9)?,
                first_tweet_date: parse_rfc3339_opt(first_tweet_date),
                last_tweet_date: parse_rfc3339_opt(last_tweet_date),
            })
        })?)
    }

    /// Get the count of documents expected in the Tantivy index.
    ///
    /// Uses a single consolidated query instead of 4 separate round-trips.
    ///
    /// # Errors
    ///
    /// Returns an error if the count query fails.
    pub fn indexable_document_count(&self) -> Result<i64> {
        let query = r"
            SELECT
                (SELECT COUNT(*) FROM tweets) +
                (SELECT COUNT(*) FROM likes WHERE full_text IS NOT NULL AND full_text != '') +
                (SELECT COUNT(*) FROM direct_messages) +
                (SELECT COUNT(*) FROM grok_messages) +
                (SELECT COUNT(*) FROM bookmarks WHERE full_text IS NOT NULL AND full_text != '')
        ";

        Ok(self.conn.query_row(query, [], |row| row.get(0))?)
    }

    /// Run database health checks for `xf doctor`.
    #[must_use]
    pub fn database_health_checks(&self) -> Vec<HealthCheck> {
        let mut checks = Vec::new();

        checks.push(self.check_integrity());
        checks.push(self.check_schema_version());
        checks.extend(self.check_fts_integrity());
        checks.extend(self.check_fts_orphaned());
        checks.extend(self.check_fts_missing());
        checks.push(self.check_orphaned_dm_messages());
        checks.push(self.check_grok_fts_counts());
        checks.push(self.check_table_stats());

        checks
    }

    /// Apply safe, idempotent database optimizations.
    ///
    /// # Errors
    ///
    /// Returns an error if any pragma fails.
    pub fn optimize(&self) -> Result<()> {
        self.conn.execute_batch(
            "
            PRAGMA optimize;
            PRAGMA wal_checkpoint(TRUNCATE);
            ",
        )?;
        Ok(())
    }

    /// Rebuild all FTS5 tables from source tables.
    ///
    /// This is safe and idempotent because FTS tables are derived data.
    ///
    /// # Errors
    ///
    /// Returns an error if the rebuild fails.
    pub fn rebuild_fts_tables(&mut self) -> Result<FtsRebuildStats> {
        let tx = self.conn.transaction()?;

        tx.execute("DELETE FROM fts_tweets", [])?;
        let tweets = tx.execute(
            "INSERT INTO fts_tweets (tweet_id, full_text) SELECT id, full_text FROM tweets",
            [],
        )?;

        tx.execute("DELETE FROM fts_likes", [])?;
        let likes = tx.execute(
            "INSERT INTO fts_likes (tweet_id, full_text)
             SELECT tweet_id, full_text FROM likes WHERE full_text IS NOT NULL AND full_text != ''",
            [],
        )?;

        tx.execute("DELETE FROM fts_dms", [])?;
        let dms = tx.execute(
            "INSERT INTO fts_dms (dm_id, text) SELECT id, text FROM direct_messages",
            [],
        )?;

        tx.execute("DELETE FROM fts_grok", [])?;
        let grok = tx.execute(
            "INSERT INTO fts_grok (grok_id, message)
             SELECT CAST(id AS TEXT), message FROM grok_messages",
            [],
        )?;

        tx.execute("DELETE FROM fts_bookmarks", [])?;
        let bookmarks = tx.execute(
            "INSERT INTO fts_bookmarks (tweet_id, full_text)
             SELECT tweet_id, full_text FROM bookmarks WHERE full_text IS NOT NULL AND full_text != ''",
            [],
        )?;

        tx.commit()?;

        Ok(FtsRebuildStats {
            tweets,
            likes,
            dms,
            grok,
            bookmarks,
        })
    }

    /// Rebuild DM conversation summaries from direct messages.
    ///
    /// This avoids dropping conversations while foreign keys are enabled.
    ///
    /// # Errors
    ///
    /// Returns an error if any SQL statement fails.
    pub fn rebuild_dm_conversations(&mut self) -> Result<usize> {
        let tx = self.conn.transaction()?;

        let rebuilt = tx.execute(
            r"
            INSERT OR REPLACE INTO dm_conversations
                (conversation_id, participant_ids, message_count, first_message_at, last_message_at)
            SELECT
                dm.conversation_id,
                (
                    SELECT group_concat(participant_id, ',')
                    FROM (
                        SELECT sender_id AS participant_id
                        FROM direct_messages dm2
                        WHERE dm2.conversation_id = dm.conversation_id
                        UNION
                        SELECT recipient_id AS participant_id
                        FROM direct_messages dm3
                        WHERE dm3.conversation_id = dm.conversation_id
                        ORDER BY participant_id
                    )
                ) AS participant_ids,
                COUNT(*) AS message_count,
                MIN(created_at) AS first_message_at,
                MAX(created_at) AS last_message_at
            FROM direct_messages dm
            GROUP BY dm.conversation_id
            ",
            [],
        )?;

        tx.execute(
            "
            DELETE FROM dm_conversations
            WHERE conversation_id NOT IN (SELECT DISTINCT conversation_id FROM direct_messages)
            ",
            [],
        )?;

        tx.commit()?;
        Ok(rebuilt)
    }

    /// Collect per-table row counts and optional size statistics.
    ///
    /// # Errors
    ///
    /// Returns an error if a table count query fails.
    pub fn database_table_stats(&self) -> Result<Vec<TableStat>> {
        let tables = [
            "tweets",
            "likes",
            "direct_messages",
            "dm_conversations",
            "grok_messages",
            "bookmarks",
            "followers",
            "following",
            "blocks",
            "mutes",
            "fts_tweets",
            "fts_likes",
            "fts_dms",
            "fts_grok",
            "fts_bookmarks",
        ];

        let has_dbstat = self.dbstat_available();
        let mut stats = Vec::with_capacity(tables.len());

        for table in tables {
            let rows = self.table_row_count(table)?;
            let bytes = if has_dbstat {
                self.table_size_bytes(table)
            } else {
                None
            };
            stats.push(TableStat {
                name: table.to_string(),
                rows,
                bytes,
            });
        }

        Ok(stats)
    }

    fn check_integrity(&self) -> HealthCheck {
        match self
            .conn
            .query_row("PRAGMA integrity_check", [], |row| row.get::<_, String>(0))
        {
            Ok(result) => {
                if result == "ok" {
                    HealthCheck {
                        category: CheckCategory::Database,
                        name: "PRAGMA integrity_check".to_string(),
                        status: CheckStatus::Pass,
                        message: "ok".to_string(),
                        suggestion: None,
                    }
                } else {
                    HealthCheck {
                        category: CheckCategory::Database,
                        name: "PRAGMA integrity_check".to_string(),
                        status: CheckStatus::Error,
                        message: format!("Integrity check failed: {result}"),
                        suggestion: Some(
                            "Database corruption detected. Re-index or restore from backup."
                                .to_string(),
                        ),
                    }
                }
            }
            Err(err) => HealthCheck {
                category: CheckCategory::Database,
                name: "PRAGMA integrity_check".to_string(),
                status: CheckStatus::Error,
                message: format!("Integrity check failed: {err}"),
                suggestion: Some("Re-index the archive to rebuild the database.".to_string()),
            },
        }
    }

    fn check_schema_version(&self) -> HealthCheck {
        let current = self.get_schema_version();
        if current == SCHEMA_VERSION {
            HealthCheck {
                category: CheckCategory::Database,
                name: "Schema version".to_string(),
                status: CheckStatus::Pass,
                message: format!("schema_version={current}"),
                suggestion: None,
            }
        } else {
            HealthCheck {
                category: CheckCategory::Database,
                name: "Schema version".to_string(),
                status: CheckStatus::Error,
                message: format!("schema_version={current}, expected={SCHEMA_VERSION}"),
                suggestion: Some(
                    "Run 'xf index --force' to rebuild the database and embeddings.".to_string(),
                ),
            }
        }
    }

    fn check_fts_integrity(&self) -> Vec<HealthCheck> {
        let tables = [
            "fts_tweets",
            "fts_likes",
            "fts_dms",
            "fts_grok",
            "fts_bookmarks",
        ];
        let mut checks = Vec::with_capacity(tables.len());

        for table in tables {
            let sql = format!("INSERT INTO {table}({table}) VALUES('integrity-check')");
            let result = self.conn.execute(&sql, []);
            let (status, message, suggestion) = match result {
                Ok(_) => (CheckStatus::Pass, "ok".to_string(), None),
                Err(err) => (
                    CheckStatus::Error,
                    format!("Integrity check failed: {err}"),
                    Some("Run 'xf index --force' to rebuild FTS tables.".to_string()),
                ),
            };

            checks.push(HealthCheck {
                category: CheckCategory::Database,
                name: format!("FTS5 integrity ({table})"),
                status,
                message,
                suggestion,
            });
        }

        checks
    }

    fn check_fts_orphaned(&self) -> Vec<HealthCheck> {
        vec![
            self.check_count(
                "FTS orphaned rows (tweets)",
                "SELECT COUNT(*) FROM fts_tweets fts LEFT JOIN tweets t ON fts.tweet_id = t.id WHERE t.id IS NULL",
                "Run 'xf index --force' to rebuild FTS tables.",
            ),
            self.check_count(
                "FTS orphaned rows (likes)",
                "SELECT COUNT(*) FROM fts_likes fts LEFT JOIN likes l ON fts.tweet_id = l.tweet_id WHERE l.tweet_id IS NULL",
                "Run 'xf index --force' to rebuild FTS tables.",
            ),
            self.check_count(
                "FTS orphaned rows (dms)",
                "SELECT COUNT(*) FROM fts_dms fts LEFT JOIN direct_messages dm ON fts.dm_id = dm.id WHERE dm.id IS NULL",
                "Run 'xf index --force' to rebuild FTS tables.",
            ),
            self.check_count(
                "FTS orphaned rows (bookmarks)",
                "SELECT COUNT(*) FROM fts_bookmarks fts LEFT JOIN bookmarks b ON fts.tweet_id = b.tweet_id WHERE b.tweet_id IS NULL",
                "Run 'xf index --force' to rebuild FTS tables.",
            ),
        ]
    }

    fn check_fts_missing(&self) -> Vec<HealthCheck> {
        vec![
            self.check_count(
                "FTS missing rows (tweets)",
                "SELECT COUNT(*) FROM tweets t LEFT JOIN fts_tweets fts ON fts.tweet_id = t.id WHERE fts.tweet_id IS NULL",
                "Run 'xf index --force' to rebuild FTS tables.",
            ),
            self.check_count(
                "FTS missing rows (likes)",
                "SELECT COUNT(*) FROM likes l LEFT JOIN fts_likes fts ON fts.tweet_id = l.tweet_id WHERE l.full_text IS NOT NULL AND l.full_text != '' AND fts.tweet_id IS NULL",
                "Run 'xf index --force' to rebuild FTS tables.",
            ),
            self.check_count(
                "FTS missing rows (dms)",
                "SELECT COUNT(*) FROM direct_messages dm LEFT JOIN fts_dms fts ON fts.dm_id = dm.id WHERE fts.dm_id IS NULL",
                "Run 'xf index --force' to rebuild FTS tables.",
            ),
            self.check_count(
                "FTS missing rows (bookmarks)",
                "SELECT COUNT(*) FROM bookmarks b LEFT JOIN fts_bookmarks fts ON fts.tweet_id = b.tweet_id WHERE b.full_text IS NOT NULL AND b.full_text != '' AND fts.tweet_id IS NULL",
                "Run 'xf index --force' to rebuild FTS tables.",
            ),
        ]
    }

    fn check_orphaned_dm_messages(&self) -> HealthCheck {
        self.check_count(
            "Orphaned DM messages",
            "SELECT COUNT(*) FROM direct_messages dm LEFT JOIN dm_conversations conv ON dm.conversation_id = conv.conversation_id WHERE conv.conversation_id IS NULL",
            "Run 'xf index --force' to rebuild DM conversations.",
        )
    }

    fn check_grok_fts_counts(&self) -> HealthCheck {
        let grok_count = self.table_row_count("grok_messages");
        let fts_count = self.table_row_count("fts_grok");

        match (grok_count, fts_count) {
            (Ok(grok_rows), Ok(fts_rows)) => {
                if grok_rows == fts_rows {
                    HealthCheck {
                        category: CheckCategory::Database,
                        name: "FTS row count (grok)".to_string(),
                        status: CheckStatus::Pass,
                        message: format!("grok_messages={grok_rows}, fts_grok={fts_rows}"),
                        suggestion: None,
                    }
                } else {
                    HealthCheck {
                        category: CheckCategory::Database,
                        name: "FTS row count (grok)".to_string(),
                        status: CheckStatus::Warning,
                        message: format!("grok_messages={grok_rows}, fts_grok={fts_rows}"),
                        suggestion: Some("Run 'xf index --force' to rebuild Grok FTS.".to_string()),
                    }
                }
            }
            (Err(err), _) | (_, Err(err)) => HealthCheck {
                category: CheckCategory::Database,
                name: "FTS row count (grok)".to_string(),
                status: CheckStatus::Error,
                message: format!("Failed to read row counts: {err}"),
                suggestion: Some("Run 'xf index --force' to rebuild Grok FTS.".to_string()),
            },
        }
    }

    fn check_table_stats(&self) -> HealthCheck {
        match self.database_table_stats() {
            Ok(stats) => {
                let message = format_table_stats(&stats);
                HealthCheck {
                    category: CheckCategory::Database,
                    name: "Table stats".to_string(),
                    status: CheckStatus::Pass,
                    message,
                    suggestion: None,
                }
            }
            Err(err) => HealthCheck {
                category: CheckCategory::Database,
                name: "Table stats".to_string(),
                status: CheckStatus::Error,
                message: format!("Failed to collect table stats: {err}"),
                suggestion: None,
            },
        }
    }

    fn check_count(&self, name: &str, sql: &str, suggestion: &str) -> HealthCheck {
        match self.conn.query_row(sql, [], |row| row.get::<_, i64>(0)) {
            Ok(count) => {
                if count == 0 {
                    HealthCheck {
                        category: CheckCategory::Database,
                        name: name.to_string(),
                        status: CheckStatus::Pass,
                        message: "0 rows".to_string(),
                        suggestion: None,
                    }
                } else {
                    HealthCheck {
                        category: CheckCategory::Database,
                        name: name.to_string(),
                        status: CheckStatus::Warning,
                        message: format!("{count} rows"),
                        suggestion: Some(suggestion.to_string()),
                    }
                }
            }
            Err(err) => HealthCheck {
                category: CheckCategory::Database,
                name: name.to_string(),
                status: CheckStatus::Error,
                message: format!("Query failed: {err}"),
                suggestion: Some(suggestion.to_string()),
            },
        }
    }

    fn table_row_count(&self, table: &str) -> Result<i64> {
        let sql = format!("SELECT COUNT(*) FROM {table}");
        self.conn
            .query_row(&sql, [], |row| row.get(0))
            .with_context(|| format!("Failed to count rows for table {table}"))
    }

    fn dbstat_available(&self) -> bool {
        let result: std::result::Result<i64, _> = self.conn.query_row(
            "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='dbstat'",
            [],
            |row| row.get(0),
        );
        result.unwrap_or(0) > 0
    }

    fn table_size_bytes(&self, table: &str) -> Option<i64> {
        let sql = "SELECT SUM(pgsize) FROM dbstat WHERE name = ?1";
        self.conn
            .query_row(sql, [table], |row| row.get::<_, Option<i64>>(0))
            .ok()
            .flatten()
    }

    /// Search tweets using FTS5.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn search_tweets(&self, query: &str, limit: usize) -> Result<Vec<Tweet>> {
        let limit = limit_to_i64(limit);
        let mut stmt = self.conn.prepare(
            r"
            SELECT t.id, t.created_at, t.full_text, t.source, t.favorite_count, t.retweet_count,
                   t.lang, t.in_reply_to_status_id, t.in_reply_to_user_id, t.in_reply_to_screen_name,
                   t.is_retweet, t.hashtags_json, t.mentions_json, t.urls_json, t.media_json
            FROM tweets t
            JOIN fts_tweets fts ON t.id = fts.tweet_id
            WHERE fts_tweets MATCH ?
            ORDER BY rank
            LIMIT ?
            ",
        )?;

        let tweets = stmt
            .query_map(params![query, limit], |row| {
                Ok(Tweet {
                    id: row.get(0)?,
                    created_at: parse_rfc3339_or_epoch(row.get::<_, Option<String>>(1)?),
                    full_text: row.get(2)?,
                    source: row.get(3)?,
                    favorite_count: row.get(4)?,
                    retweet_count: row.get(5)?,
                    lang: row.get(6)?,
                    in_reply_to_status_id: row.get(7)?,
                    in_reply_to_user_id: row.get(8)?,
                    in_reply_to_screen_name: row.get(9)?,
                    is_retweet: row.get::<_, i32>(10)? != 0,
                    hashtags: serde_json::from_str(&row.get::<_, String>(11)?).unwrap_or_default(),
                    user_mentions: serde_json::from_str(&row.get::<_, String>(12)?)
                        .unwrap_or_default(),
                    urls: serde_json::from_str(&row.get::<_, String>(13)?).unwrap_or_default(),
                    media: serde_json::from_str(&row.get::<_, String>(14)?).unwrap_or_default(),
                })
            })?
            .filter_map(std::result::Result::ok)
            .collect();

        Ok(tweets)
    }

    /// Search likes using FTS5.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn search_likes(&self, query: &str, limit: usize) -> Result<Vec<Like>> {
        let limit = limit_to_i64(limit);
        let mut stmt = self.conn.prepare(
            r"
            SELECT l.tweet_id, l.full_text, l.expanded_url
            FROM likes l
            JOIN fts_likes fts ON l.tweet_id = fts.tweet_id
            WHERE fts_likes MATCH ?
            ORDER BY rank
            LIMIT ?
            ",
        )?;

        let likes = stmt
            .query_map(params![query, limit], |row| {
                Ok(Like {
                    tweet_id: row.get(0)?,
                    full_text: row.get(1)?,
                    expanded_url: row.get(2)?,
                })
            })?
            .filter_map(std::result::Result::ok)
            .collect();

        Ok(likes)
    }

    /// Search DMs using FTS5.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn search_dms(&self, query: &str, limit: usize) -> Result<Vec<DirectMessage>> {
        let limit = limit_to_i64(limit);
        let mut stmt = self.conn.prepare(
            r"
            SELECT dm.id, dm.conversation_id, dm.sender_id, dm.recipient_id, dm.text,
                   dm.created_at, dm.urls_json, dm.media_urls_json
            FROM direct_messages dm
            JOIN fts_dms fts ON dm.id = fts.dm_id
            WHERE fts_dms MATCH ?
            ORDER BY rank
            LIMIT ?
            ",
        )?;

        let dms = stmt
            .query_map(params![query, limit], |row| {
                Ok(DirectMessage {
                    id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    sender_id: row.get(2)?,
                    recipient_id: row.get(3)?,
                    text: row.get(4)?,
                    created_at: parse_rfc3339_or_epoch(row.get::<_, Option<String>>(5)?),
                    urls: serde_json::from_str(&row.get::<_, String>(6)?).unwrap_or_default(),
                    media_urls: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or_default(),
                })
            })?
            .filter_map(std::result::Result::ok)
            .collect();

        Ok(dms)
    }

    /// Get all messages for a conversation, ordered by timestamp.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_conversation_messages(&self, conversation_id: &str) -> Result<Vec<DirectMessage>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT id, conversation_id, sender_id, recipient_id, text, created_at, urls_json, media_urls_json
            FROM direct_messages
            WHERE conversation_id = ?
            ORDER BY created_at ASC, id ASC
            ",
        )?;

        let messages = stmt
            .query_map(params![conversation_id], |row| {
                Ok(DirectMessage {
                    id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    sender_id: row.get(2)?,
                    recipient_id: row.get(3)?,
                    text: row.get(4)?,
                    created_at: parse_rfc3339_or_epoch(row.get::<_, Option<String>>(5)?),
                    urls: serde_json::from_str(&row.get::<_, String>(6)?).unwrap_or_default(),
                    media_urls: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or_default(),
                })
            })?
            .filter_map(std::result::Result::ok)
            .collect();

        Ok(messages)
    }

    /// Search Grok messages using FTS5.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn search_grok(&self, query: &str, limit: usize) -> Result<Vec<GrokMessage>> {
        let limit = limit_to_i64(limit);
        let mut stmt = self.conn.prepare(
            r"
            SELECT g.chat_id, g.message, g.sender, g.created_at, g.grok_mode
            FROM grok_messages g
            JOIN fts_grok fts ON g.id = CAST(fts.grok_id AS INTEGER)
            WHERE fts_grok MATCH ?
            ORDER BY rank
            LIMIT ?
            ",
        )?;

        let messages = stmt
            .query_map(params![query, limit], |row| {
                Ok(GrokMessage {
                    chat_id: row.get(0)?,
                    message: row.get(1)?,
                    sender: row.get(2)?,
                    created_at: parse_rfc3339_or_epoch(row.get::<_, Option<String>>(3)?),
                    grok_mode: row.get(4)?,
                })
            })?
            .filter_map(std::result::Result::ok)
            .collect();

        Ok(messages)
    }

    /// Get a tweet by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_tweet(&self, id: &str) -> Result<Option<Tweet>> {
        let result = self.conn.query_row(
            r"
            SELECT id, created_at, full_text, source, favorite_count, retweet_count,
                   lang, in_reply_to_status_id, in_reply_to_user_id, in_reply_to_screen_name,
                   is_retweet, hashtags_json, mentions_json, urls_json, media_json
            FROM tweets WHERE id = ?
            ",
            params![id],
            |row| {
                Ok(Tweet {
                    id: row.get(0)?,
                    created_at: parse_rfc3339_or_epoch(row.get::<_, Option<String>>(1)?),
                    full_text: row.get(2)?,
                    source: row.get(3)?,
                    favorite_count: row.get(4)?,
                    retweet_count: row.get(5)?,
                    lang: row.get(6)?,
                    in_reply_to_status_id: row.get(7)?,
                    in_reply_to_user_id: row.get(8)?,
                    in_reply_to_screen_name: row.get(9)?,
                    is_retweet: row.get::<_, i32>(10)? != 0,
                    hashtags: serde_json::from_str(&row.get::<_, String>(11)?).unwrap_or_default(),
                    user_mentions: serde_json::from_str(&row.get::<_, String>(12)?)
                        .unwrap_or_default(),
                    urls: serde_json::from_str(&row.get::<_, String>(13)?).unwrap_or_default(),
                    media: serde_json::from_str(&row.get::<_, String>(14)?).unwrap_or_default(),
                })
            },
        );

        match result {
            Ok(tweet) => Ok(Some(tweet)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Get replies to a tweet by parent ID, ordered by creation time.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_tweet_replies(&self, parent_id: &str) -> Result<Vec<Tweet>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT id, created_at, full_text, source, favorite_count, retweet_count,
                   lang, in_reply_to_status_id, in_reply_to_user_id, in_reply_to_screen_name,
                   is_retweet, hashtags_json, mentions_json, urls_json, media_json
            FROM tweets
            WHERE in_reply_to_status_id = ?
            ORDER BY created_at ASC
            ",
        )?;

        let tweets = stmt
            .query_map(params![parent_id], |row| {
                Ok(Tweet {
                    id: row.get(0)?,
                    created_at: parse_rfc3339_or_epoch(row.get::<_, Option<String>>(1)?),
                    full_text: row.get(2)?,
                    source: row.get(3)?,
                    favorite_count: row.get(4)?,
                    retweet_count: row.get(5)?,
                    lang: row.get(6)?,
                    in_reply_to_status_id: row.get(7)?,
                    in_reply_to_user_id: row.get(8)?,
                    in_reply_to_screen_name: row.get(9)?,
                    is_retweet: row.get::<_, i32>(10)? != 0,
                    hashtags: serde_json::from_str(&row.get::<_, String>(11)?).unwrap_or_default(),
                    user_mentions: serde_json::from_str(&row.get::<_, String>(12)?)
                        .unwrap_or_default(),
                    urls: serde_json::from_str(&row.get::<_, String>(13)?).unwrap_or_default(),
                    media: serde_json::from_str(&row.get::<_, String>(14)?).unwrap_or_default(),
                })
            })?
            .filter_map(std::result::Result::ok)
            .collect();

        Ok(tweets)
    }

    /// Get a tweet thread rooted at the earliest ancestor, including all replies.
    ///
    /// Uses recursive CTEs to fetch the entire thread in a single query,
    /// eliminating N+1 query patterns.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_tweet_thread(&self, id: &str) -> Result<Vec<Tweet>> {
        // Single recursive CTE query that:
        // 1. Finds all ancestors of the given tweet (going up the reply chain)
        // 2. Identifies the root (earliest ancestor)
        // 3. Finds all descendants from the root (going down)
        // This eliminates N+1 queries - one query instead of potentially 50+
        let mut stmt = self.conn.prepare(
            r"
            WITH RECURSIVE
                -- Find all ancestors by traversing up the reply chain
                ancestors(id, parent_id, depth) AS (
                    SELECT id, in_reply_to_status_id, 0
                    FROM tweets WHERE id = ?1
                    UNION ALL
                    SELECT t.id, t.in_reply_to_status_id, a.depth + 1
                    FROM tweets t
                    JOIN ancestors a ON t.id = a.parent_id
                    WHERE a.parent_id IS NOT NULL AND a.parent_id != ''
                      AND a.depth < 100  -- Prevent infinite loops
                ),
                -- The root is the ancestor with no parent in our archive
                root AS (
                    SELECT id FROM ancestors
                    WHERE parent_id IS NULL OR parent_id = ''
                       OR parent_id NOT IN (SELECT id FROM tweets)
                    ORDER BY depth DESC
                    LIMIT 1
                ),
                -- Find all descendants from the root
                thread_ids(id) AS (
                    SELECT id FROM root
                    UNION ALL
                    SELECT t.id
                    FROM tweets t
                    JOIN thread_ids ti ON t.in_reply_to_status_id = ti.id
                )
            SELECT DISTINCT t.id, t.created_at, t.full_text, t.source,
                   t.favorite_count, t.retweet_count, t.lang,
                   t.in_reply_to_status_id, t.in_reply_to_user_id,
                   t.in_reply_to_screen_name, t.is_retweet,
                   t.hashtags_json, t.mentions_json, t.urls_json, t.media_json
            FROM tweets t
            WHERE t.id IN (SELECT id FROM thread_ids)
            ORDER BY t.created_at ASC
            ",
        )?;

        let tweets = stmt
            .query_map(params![id], |row| {
                Ok(Tweet {
                    id: row.get(0)?,
                    created_at: parse_rfc3339_or_epoch(row.get::<_, Option<String>>(1)?),
                    full_text: row.get(2)?,
                    source: row.get(3)?,
                    favorite_count: row.get(4)?,
                    retweet_count: row.get(5)?,
                    lang: row.get(6)?,
                    in_reply_to_status_id: row.get(7)?,
                    in_reply_to_user_id: row.get(8)?,
                    in_reply_to_screen_name: row.get(9)?,
                    is_retweet: row.get::<_, i32>(10)? != 0,
                    hashtags: serde_json::from_str(&row.get::<_, String>(11)?).unwrap_or_default(),
                    user_mentions: serde_json::from_str(&row.get::<_, String>(12)?)
                        .unwrap_or_default(),
                    urls: serde_json::from_str(&row.get::<_, String>(13)?).unwrap_or_default(),
                    media: serde_json::from_str(&row.get::<_, String>(14)?).unwrap_or_default(),
                })
            })?
            .collect::<std::result::Result<Vec<_>, _>>()?;

        Ok(tweets)
    }

    /// Get all tweets, optionally limited.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_all_tweets(&self, limit: Option<usize>) -> Result<Vec<Tweet>> {
        // Use parameterized query for better query plan caching.
        // SQLite treats LIMIT -1 as "no limit", so we use that for unlimited queries.
        const QUERY: &str = r"SELECT id, created_at, full_text, source, favorite_count, retweet_count,
                   lang, in_reply_to_status_id, in_reply_to_user_id, in_reply_to_screen_name,
                   is_retweet, hashtags_json, mentions_json, urls_json, media_json
                FROM tweets ORDER BY created_at DESC LIMIT ?";

        let limit_param: i64 = limit.map_or(-1, |l| i64::try_from(l).unwrap_or(i64::MAX));
        let mut stmt = self.conn.prepare_cached(QUERY)?;
        let tweets = stmt
            .query_map([limit_param], |row| {
                Ok(Tweet {
                    id: row.get(0)?,
                    created_at: parse_rfc3339_or_epoch(row.get::<_, Option<String>>(1)?),
                    full_text: row.get(2)?,
                    source: row.get(3)?,
                    favorite_count: row.get(4)?,
                    retweet_count: row.get(5)?,
                    lang: row.get(6)?,
                    in_reply_to_status_id: row.get(7)?,
                    in_reply_to_user_id: row.get(8)?,
                    in_reply_to_screen_name: row.get(9)?,
                    is_retweet: row.get::<_, i32>(10)? != 0,
                    hashtags: serde_json::from_str(&row.get::<_, String>(11)?).unwrap_or_default(),
                    user_mentions: serde_json::from_str(&row.get::<_, String>(12)?)
                        .unwrap_or_default(),
                    urls: serde_json::from_str(&row.get::<_, String>(13)?).unwrap_or_default(),
                    media: serde_json::from_str(&row.get::<_, String>(14)?).unwrap_or_default(),
                })
            })?
            .filter_map(std::result::Result::ok)
            .collect();

        Ok(tweets)
    }

    /// Get all likes, optionally limited.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_all_likes(&self, limit: Option<usize>) -> Result<Vec<Like>> {
        const QUERY: &str = "SELECT tweet_id, full_text, expanded_url FROM likes LIMIT ?";
        let limit_param: i64 = limit.map_or(-1, |l| i64::try_from(l).unwrap_or(i64::MAX));

        let mut stmt = self.conn.prepare_cached(QUERY)?;
        let likes = stmt
            .query_map([limit_param], |row| {
                Ok(Like {
                    tweet_id: row.get(0)?,
                    full_text: row.get(1)?,
                    expanded_url: row.get(2)?,
                })
            })?
            .filter_map(std::result::Result::ok)
            .collect();

        Ok(likes)
    }

    /// Get all DM conversations with messages, optionally limited.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_all_dms(&self, limit: Option<usize>) -> Result<Vec<DirectMessage>> {
        const QUERY: &str = r"SELECT id, conversation_id, sender_id, recipient_id, text,
                   created_at, urls_json, media_urls_json
                FROM direct_messages ORDER BY created_at DESC LIMIT ?";
        let limit_param: i64 = limit.map_or(-1, |l| i64::try_from(l).unwrap_or(i64::MAX));

        let mut stmt = self.conn.prepare_cached(QUERY)?;
        let dms = stmt
            .query_map([limit_param], |row| {
                Ok(DirectMessage {
                    id: row.get(0)?,
                    conversation_id: row.get(1)?,
                    sender_id: row.get(2)?,
                    recipient_id: row.get(3)?,
                    text: row.get(4)?,
                    created_at: parse_rfc3339_or_epoch(row.get::<_, Option<String>>(5)?),
                    urls: serde_json::from_str(&row.get::<_, String>(6)?).unwrap_or_default(),
                    media_urls: serde_json::from_str(&row.get::<_, String>(7)?).unwrap_or_default(),
                })
            })?
            .filter_map(std::result::Result::ok)
            .collect();

        Ok(dms)
    }

    /// Get DM conversation summaries, optionally limited.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_dm_conversation_summaries(
        &self,
        limit: Option<usize>,
    ) -> Result<Vec<DmConversationSummary>> {
        const QUERY: &str = r"SELECT conversation_id, participant_ids, message_count,
                   first_message_at, last_message_at
                FROM dm_conversations
                ORDER BY last_message_at DESC LIMIT ?";
        let limit_param: i64 = limit.map_or(-1, |l| i64::try_from(l).unwrap_or(i64::MAX));

        let mut stmt = self.conn.prepare_cached(QUERY)?;
        let summaries = stmt
            .query_map([limit_param], |row| {
                let participants: String = row.get(1)?;
                let participant_ids = participants
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect();

                Ok(DmConversationSummary {
                    conversation_id: row.get(0)?,
                    participant_ids,
                    message_count: row.get(2)?,
                    first_message_at: parse_rfc3339_opt(row.get::<_, Option<String>>(3)?),
                    last_message_at: parse_rfc3339_opt(row.get::<_, Option<String>>(4)?),
                })
            })?
            .filter_map(std::result::Result::ok)
            .collect();

        Ok(summaries)
    }

    /// Get all followers, optionally limited.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_all_followers(&self, limit: Option<usize>) -> Result<Vec<Follower>> {
        const QUERY: &str = "SELECT account_id, user_link FROM followers LIMIT ?";
        let limit_param: i64 = limit.map_or(-1, |l| i64::try_from(l).unwrap_or(i64::MAX));

        let mut stmt = self.conn.prepare_cached(QUERY)?;
        let followers = stmt
            .query_map([limit_param], |row| {
                Ok(Follower {
                    account_id: row.get(0)?,
                    user_link: row.get(1)?,
                })
            })?
            .filter_map(std::result::Result::ok)
            .collect();

        Ok(followers)
    }

    /// Get all following, optionally limited.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_all_following(&self, limit: Option<usize>) -> Result<Vec<Following>> {
        const QUERY: &str = "SELECT account_id, user_link FROM following LIMIT ?";
        let limit_param: i64 = limit.map_or(-1, |l| i64::try_from(l).unwrap_or(i64::MAX));

        let mut stmt = self.conn.prepare_cached(QUERY)?;
        let following = stmt
            .query_map([limit_param], |row| {
                Ok(Following {
                    account_id: row.get(0)?,
                    user_link: row.get(1)?,
                })
            })?
            .filter_map(std::result::Result::ok)
            .collect();

        Ok(following)
    }

    /// Get all blocks, optionally limited.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_all_blocks(&self, limit: Option<usize>) -> Result<Vec<Block>> {
        const QUERY: &str = "SELECT account_id, user_link FROM blocks LIMIT ?";
        let limit_param: i64 = limit.map_or(-1, |l| i64::try_from(l).unwrap_or(i64::MAX));

        let mut stmt = self.conn.prepare_cached(QUERY)?;
        let blocks = stmt
            .query_map([limit_param], |row| {
                Ok(Block {
                    account_id: row.get(0)?,
                    user_link: row.get(1)?,
                })
            })?
            .filter_map(std::result::Result::ok)
            .collect();

        Ok(blocks)
    }

    /// Get all mutes, optionally limited.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_all_mutes(&self, limit: Option<usize>) -> Result<Vec<Mute>> {
        const QUERY: &str = "SELECT account_id, user_link FROM mutes LIMIT ?";
        let limit_param: i64 = limit.map_or(-1, |l| i64::try_from(l).unwrap_or(i64::MAX));

        let mut stmt = self.conn.prepare_cached(QUERY)?;
        let mutes = stmt
            .query_map([limit_param], |row| {
                Ok(Mute {
                    account_id: row.get(0)?,
                    user_link: row.get(1)?,
                })
            })?
            .filter_map(std::result::Result::ok)
            .collect();

        Ok(mutes)
    }

    /// Get all Grok messages, optionally limited.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_all_grok_messages(&self, limit: Option<usize>) -> Result<Vec<GrokMessage>> {
        const QUERY: &str = r"SELECT chat_id, message, sender, created_at, grok_mode
                FROM grok_messages ORDER BY created_at DESC LIMIT ?";
        let limit_param: i64 = limit.map_or(-1, |l| i64::try_from(l).unwrap_or(i64::MAX));

        let mut stmt = self.conn.prepare_cached(QUERY)?;
        let messages = stmt
            .query_map([limit_param], |row| {
                Ok(GrokMessage {
                    chat_id: row.get(0)?,
                    message: row.get(1)?,
                    sender: row.get(2)?,
                    created_at: parse_rfc3339_or_epoch(row.get::<_, Option<String>>(3)?),
                    grok_mode: row.get(4)?,
                })
            })?
            .filter_map(std::result::Result::ok)
            .collect();

        Ok(messages)
    }

    // ============================================================
    // Embeddings (Semantic Search)
    // ============================================================

    /// Store an embedding for a document.
    ///
    /// The embedding is stored as a BLOB with F16 quantization.
    ///
    /// # Errors
    ///
    /// Returns an error if the database insert fails.
    pub fn store_embedding(
        &self,
        doc_id: &str,
        doc_type: &str,
        embedding: &[f32],
        content_hash: Option<&[u8; 32]>,
    ) -> Result<()> {
        self.store_embedding_with_model(doc_id, doc_type, embedding, content_hash, "default")
    }

    /// Store an embedding for a document with a specific model ID.
    ///
    /// The `model_id` distinguishes embeddings from different models
    /// (e.g., "fast", "quality", "default").
    ///
    /// # Errors
    ///
    /// Returns an error if the database insert fails.
    pub fn store_embedding_with_model(
        &self,
        doc_id: &str,
        doc_type: &str,
        embedding: &[f32],
        content_hash: Option<&[u8; 32]>,
        model_id: &str,
    ) -> Result<()> {
        use half::f16;

        // Convert f32 to f16 for storage (50% smaller)
        let f16_bytes: Vec<u8> = embedding
            .iter()
            .flat_map(|&f| f16::from_f32(f).to_le_bytes())
            .collect();

        self.conn.execute(
            r"
            INSERT OR REPLACE INTO embeddings
            (doc_id, doc_type, embedding, content_hash, created_at, model_id)
            VALUES (?, ?, ?, ?, ?, ?)
            ",
            params![
                doc_id,
                doc_type,
                f16_bytes,
                content_hash.map(<[u8; 32]>::as_slice),
                Utc::now().to_rfc3339(),
                model_id,
            ],
        )?;

        Ok(())
    }

    /// Store multiple embeddings in a batch.
    ///
    /// More efficient than calling `store_embedding` repeatedly.
    ///
    /// # Errors
    ///
    /// Returns an error if any database insert fails.
    pub fn store_embeddings_batch(&self, embeddings: &[EmbeddingRecord]) -> Result<usize> {
        self.store_embeddings_batch_with_model(embeddings, "default")
    }

    /// Store multiple embeddings in a batch with a specific model ID.
    ///
    /// # Errors
    ///
    /// Returns an error if any database insert fails.
    pub fn store_embeddings_batch_with_model(
        &self,
        embeddings: &[EmbeddingRecord],
        model_id: &str,
    ) -> Result<usize> {
        use half::f16;

        let tx = self.conn.unchecked_transaction()?;
        let mut count = 0;

        {
            let mut stmt = tx.prepare(
                r"
                INSERT OR REPLACE INTO embeddings
                (doc_id, doc_type, embedding, content_hash, created_at, model_id)
                VALUES (?, ?, ?, ?, ?, ?)
                ",
            )?;

            let now = Utc::now().to_rfc3339();

            for (doc_id, doc_type, embedding, content_hash) in embeddings {
                let f16_bytes: Vec<u8> = embedding
                    .iter()
                    .flat_map(|&f| f16::from_f32(f).to_le_bytes())
                    .collect();

                stmt.execute(params![
                    doc_id,
                    doc_type,
                    f16_bytes,
                    content_hash.as_ref().map(<[u8; 32]>::as_slice),
                    &now,
                    model_id,
                ])?;
                count += 1;
            }
        }

        tx.commit()?;
        Ok(count)
    }

    /// Get an embedding by document ID and type.
    ///
    /// Returns the embedding as f32 values (converted from stored F16).
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    ///
    /// # Panics
    ///
    /// Panics if stored embedding bytes are not aligned to 2-byte F16 chunks.
    pub fn get_embedding(&self, doc_id: &str, doc_type: &str) -> Result<Option<Vec<f32>>> {
        use half::f16;

        let result: rusqlite::Result<Vec<u8>> = self.conn.query_row(
            "SELECT embedding FROM embeddings WHERE doc_id = ? AND doc_type = ?",
            params![doc_id, doc_type],
            |row| row.get(0),
        );

        match result {
            Ok(bytes) => {
                // Convert F16 bytes back to f32
                let floats: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let arr: [u8; 2] = chunk.try_into().unwrap();
                        f16::from_le_bytes(arr).to_f32()
                    })
                    .collect();
                Ok(Some(floats))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Get the content hash for a document's embedding.
    ///
    /// Returns None if the document is missing or has no stored hash.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_embedding_hash(&self, doc_id: &str, doc_type: &str) -> Result<Option<[u8; 32]>> {
        let result: rusqlite::Result<Option<Vec<u8>>> = self.conn.query_row(
            "SELECT content_hash FROM embeddings WHERE doc_id = ? AND doc_type = ?",
            params![doc_id, doc_type],
            |row| row.get(0),
        );

        match result {
            Ok(Some(bytes)) => {
                let arr: [u8; 32] = bytes
                    .as_slice()
                    .try_into()
                    .map_err(|_| anyhow::anyhow!("Invalid content_hash length"))?;
                Ok(Some(arr))
            }
            Ok(None) | Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Get an embedding by content hash.
    ///
    /// Returns None if no embedding with the given hash exists.
    ///
    /// Content hashes are global; embeddings can be reused across types when the
    /// canonicalized text is identical.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    ///
    /// # Panics
    ///
    /// Panics if stored embedding bytes are not aligned to 2-byte F16 chunks.
    pub fn get_embedding_by_hash(&self, content_hash: &[u8; 32]) -> Result<Option<Vec<f32>>> {
        use half::f16;

        let result: rusqlite::Result<Vec<u8>> = self.conn.query_row(
            "SELECT embedding FROM embeddings WHERE content_hash = ? LIMIT 1",
            params![content_hash.as_slice()],
            |row| row.get(0),
        );

        match result {
            Ok(bytes) => {
                let floats: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let arr: [u8; 2] = chunk.try_into().unwrap();
                        f16::from_le_bytes(arr).to_f32()
                    })
                    .collect();
                Ok(Some(floats))
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Load existing content hashes keyed by document ID and type.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails or stored hashes are invalid.
    pub fn load_embedding_hashes_by_doc(
        &self,
    ) -> Result<HashMap<String, HashMap<String, [u8; 32]>>> {
        let mut stmt = self.conn.prepare(
            "SELECT doc_id, doc_type, content_hash FROM embeddings WHERE content_hash IS NOT NULL",
        )?;

        let rows = stmt.query_map([], |row| {
            let doc_id: String = row.get(0)?;
            let doc_type: String = row.get(1)?;
            let bytes: Vec<u8> = row.get(2)?;
            let hash: [u8; 32] = match bytes.as_slice().try_into() {
                Ok(hash) => hash,
                Err(_) => {
                    return Err(rusqlite::Error::FromSqlConversionFailure(
                        bytes.len(),
                        rusqlite::types::Type::Blob,
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "Invalid content_hash length",
                        )),
                    ));
                }
            };
            Ok((doc_id, doc_type, hash))
        })?;

        let mut map: HashMap<String, HashMap<String, [u8; 32]>> = HashMap::new();
        for row in rows {
            let (doc_id, doc_type, hash) = row?;
            map.entry(doc_id).or_default().insert(doc_type, hash);
        }

        Ok(map)
    }

    /// Load content hashes keyed by (doc_id, doc_type) for a specific model.
    ///
    /// Only returns hashes for embeddings stored with the given `model_id`.
    /// Used by two-tier embedding generation to avoid cross-model hash collisions.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails or stored hashes are invalid.
    pub fn load_embedding_hashes_by_doc_for_model(
        &self,
        model_id: &str,
    ) -> Result<HashMap<String, HashMap<String, [u8; 32]>>> {
        let mut stmt = self.conn.prepare(
            "SELECT doc_id, doc_type, content_hash FROM embeddings \
             WHERE content_hash IS NOT NULL AND model_id = ?",
        )?;

        let rows = stmt.query_map(params![model_id], |row| {
            let doc_id: String = row.get(0)?;
            let doc_type: String = row.get(1)?;
            let bytes: Vec<u8> = row.get(2)?;
            let hash: [u8; 32] = match bytes.as_slice().try_into() {
                Ok(hash) => hash,
                Err(_) => {
                    return Err(rusqlite::Error::FromSqlConversionFailure(
                        bytes.len(),
                        rusqlite::types::Type::Blob,
                        Box::new(std::io::Error::new(
                            std::io::ErrorKind::InvalidData,
                            "Invalid content_hash length",
                        )),
                    ));
                }
            };
            Ok((doc_id, doc_type, hash))
        })?;

        let mut map: HashMap<String, HashMap<String, [u8; 32]>> = HashMap::new();
        for row in rows {
            let (doc_id, doc_type, hash) = row?;
            map.entry(doc_id).or_default().insert(doc_type, hash);
        }

        Ok(map)
    }

    /// Load content hashes for a specific set of document IDs and types.
    ///
    /// Returns a map keyed by `doc_id` -> `doc_type` -> `content_hash` for the
    /// provided lookups that exist in the embeddings table.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails or stored hashes are invalid.
    pub fn load_embedding_hashes_for_docs(
        &self,
        lookups: &[(String, String)],
    ) -> Result<HashMap<String, HashMap<String, [u8; 32]>>> {
        if lookups.is_empty() {
            return Ok(HashMap::new());
        }

        let mut map: HashMap<String, HashMap<String, [u8; 32]>> = HashMap::new();
        let chunk_size = SQLITE_BATCH_SIZE / 2;

        for chunk in lookups.chunks(chunk_size.max(1)) {
            let mut sql = String::from(
                "SELECT doc_id, doc_type, content_hash FROM embeddings WHERE content_hash IS NOT NULL AND (",
            );
            for i in 0..chunk.len() {
                if i > 0 {
                    sql.push_str(" OR ");
                }
                sql.push_str("(doc_id = ? AND doc_type = ?)");
            }
            sql.push(')');

            let mut stmt = self.conn.prepare(&sql)?;
            let params = rusqlite::params_from_iter(
                chunk
                    .iter()
                    .flat_map(|(doc_id, doc_type)| [doc_id.as_str(), doc_type.as_str()]),
            );
            let rows = stmt.query_map(params, |row| {
                let doc_id: String = row.get(0)?;
                let doc_type: String = row.get(1)?;
                let bytes: Vec<u8> = row.get(2)?;
                let hash: [u8; 32] = match bytes.as_slice().try_into() {
                    Ok(hash) => hash,
                    Err(_) => {
                        return Err(rusqlite::Error::FromSqlConversionFailure(
                            bytes.len(),
                            rusqlite::types::Type::Blob,
                            Box::new(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                "Invalid content_hash length",
                            )),
                        ));
                    }
                };
                Ok((doc_id, doc_type, hash))
            })?;

            for row in rows {
                let (doc_id, doc_type, hash) = row?;
                map.entry(doc_id).or_default().insert(doc_type, hash);
            }
        }

        Ok(map)
    }

    /// Load embeddings by content hash in batches.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails or stored embeddings are invalid.
    ///
    /// # Panics
    ///
    /// Panics if stored embedding bytes are not aligned to 2-byte F16 chunks.
    pub fn load_embeddings_by_hashes(
        &self,
        hashes: &[[u8; 32]],
    ) -> Result<HashMap<[u8; 32], Vec<f32>>> {
        use half::f16;

        if hashes.is_empty() {
            return Ok(HashMap::new());
        }

        let mut results: HashMap<[u8; 32], Vec<f32>> = HashMap::new();

        for chunk in hashes.chunks(SQLITE_BATCH_SIZE) {
            let mut sql = String::from(
                "SELECT content_hash, embedding FROM embeddings WHERE content_hash IN (",
            );
            for i in 0..chunk.len() {
                if i > 0 {
                    sql.push(',');
                }
                sql.push('?');
            }
            sql.push(')');

            let mut stmt = self.conn.prepare(&sql)?;
            let params = rusqlite::params_from_iter(chunk.iter().map(<[u8; 32]>::as_slice));
            let rows = stmt.query_map(params, |row| {
                let hash_bytes: Vec<u8> = row.get(0)?;
                let hash: [u8; 32] = match hash_bytes.as_slice().try_into() {
                    Ok(hash) => hash,
                    Err(_) => {
                        return Err(rusqlite::Error::FromSqlConversionFailure(
                            hash_bytes.len(),
                            rusqlite::types::Type::Blob,
                            Box::new(std::io::Error::new(
                                std::io::ErrorKind::InvalidData,
                                "Invalid content_hash length",
                            )),
                        ));
                    }
                };
                let bytes: Vec<u8> = row.get(1)?;
                let floats: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let arr: [u8; 2] = chunk.try_into().unwrap();
                        f16::from_le_bytes(arr).to_f32()
                    })
                    .collect();
                Ok((hash, floats))
            })?;

            for row in rows {
                let (hash, embedding) = row?;
                results.entry(hash).or_insert(embedding);
            }
        }

        Ok(results)
    }

    /// Load embeddings by content hashes, filtered to a specific model.
    ///
    /// Only returns embeddings that were stored with the given `model_id`.
    /// Used by two-tier embedding generation to reuse embeddings within
    /// the same model without cross-model contamination.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    ///
    /// # Panics
    ///
    /// Panics if the stored embedding bytes are not a multiple of 2 (corrupted data).
    pub fn load_embeddings_by_hashes_for_model(
        &self,
        hashes: &[[u8; 32]],
        model_id: &str,
    ) -> Result<HashMap<[u8; 32], Vec<f32>>> {
        use half::f16;

        if hashes.is_empty() {
            return Ok(HashMap::new());
        }

        let mut results: HashMap<[u8; 32], Vec<f32>> = HashMap::new();

        for chunk in hashes.chunks(SQLITE_BATCH_SIZE) {
            let mut sql = String::from(
                "SELECT content_hash, embedding FROM embeddings \
                 WHERE model_id = ? AND content_hash IN (",
            );
            for i in 0..chunk.len() {
                if i > 0 {
                    sql.push(',');
                }
                sql.push('?');
            }
            sql.push(')');

            let mut stmt = self.conn.prepare(&sql)?;
            let mut param_values: Vec<Box<dyn rusqlite::types::ToSql>> =
                Vec::with_capacity(1 + chunk.len());
            param_values.push(Box::new(model_id.to_string()));
            for h in chunk {
                param_values.push(Box::new(h.to_vec()));
            }
            let rows = stmt.query_map(
                rusqlite::params_from_iter(param_values.iter().map(AsRef::as_ref)),
                |row| {
                    let hash_bytes: Vec<u8> = row.get(0)?;
                    let hash: [u8; 32] = match hash_bytes.as_slice().try_into() {
                        Ok(hash) => hash,
                        Err(_) => {
                            return Err(rusqlite::Error::FromSqlConversionFailure(
                                hash_bytes.len(),
                                rusqlite::types::Type::Blob,
                                Box::new(std::io::Error::new(
                                    std::io::ErrorKind::InvalidData,
                                    "Invalid content_hash length",
                                )),
                            ));
                        }
                    };
                    let bytes: Vec<u8> = row.get(1)?;
                    let floats: Vec<f32> = bytes
                        .chunks_exact(2)
                        .map(|chunk| {
                            let arr: [u8; 2] = chunk.try_into().unwrap();
                            f16::from_le_bytes(arr).to_f32()
                        })
                        .collect();
                    Ok((hash, floats))
                },
            )?;

            for row in rows {
                let (hash, embedding) = row?;
                results.entry(hash).or_insert(embedding);
            }
        }

        Ok(results)
    }

    /// Load all embeddings for vector search.
    ///
    /// Returns tuples of (`doc_id`, `doc_type`, embedding).
    /// Embeddings are converted to f32 for fast dot product computation.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    ///
    /// # Panics
    ///
    /// Panics if stored embedding bytes are not aligned to 2-byte F16 chunks.
    pub fn load_all_embeddings(&self) -> Result<Vec<(String, String, Vec<f32>)>> {
        self.load_all_embeddings_filtered(None)
    }

    /// Load all embeddings, optionally filtered by model ID.
    ///
    /// When `model_id` is `Some`, only embeddings with that model are returned.
    /// When `None`, all embeddings are returned (backward-compatible).
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    ///
    /// # Panics
    ///
    /// Panics if the stored embedding bytes are not a multiple of 2 (corrupted data).
    pub fn load_all_embeddings_filtered(
        &self,
        model_id: Option<&str>,
    ) -> Result<Vec<(String, String, Vec<f32>)>> {
        use half::f16;

        let query = if model_id.is_some() {
            "SELECT doc_id, doc_type, embedding FROM embeddings WHERE model_id = ?"
        } else {
            "SELECT doc_id, doc_type, embedding FROM embeddings"
        };

        let mut stmt = self.conn.prepare(query)?;

        let rows = if let Some(mid) = model_id {
            stmt.query_map(params![mid], |row| {
                let doc_id: String = row.get(0)?;
                let doc_type: String = row.get(1)?;
                let bytes: Vec<u8> = row.get(2)?;
                let floats: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let arr: [u8; 2] = chunk.try_into().unwrap();
                        f16::from_le_bytes(arr).to_f32()
                    })
                    .collect();
                Ok((doc_id, doc_type, floats))
            })?
            .filter_map(std::result::Result::ok)
            .collect()
        } else {
            stmt.query_map([], |row| {
                let doc_id: String = row.get(0)?;
                let doc_type: String = row.get(1)?;
                let bytes: Vec<u8> = row.get(2)?;
                let floats: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let arr: [u8; 2] = chunk.try_into().unwrap();
                        f16::from_le_bytes(arr).to_f32()
                    })
                    .collect();
                Ok((doc_id, doc_type, floats))
            })?
            .filter_map(std::result::Result::ok)
            .collect()
        };

        Ok(rows)
    }

    /// Load all embeddings in raw F16 byte format for vector index writing.
    ///
    /// Returns tuples of (`doc_id`, `doc_type`, raw F16 bytes).
    /// This preserves exact values without float conversion drift.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn load_all_embeddings_raw(&self) -> Result<Vec<(String, String, Vec<u8>)>> {
        self.load_all_embeddings_raw_filtered(None)
    }

    /// Load all embeddings in raw F16 format, optionally filtered by model ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn load_all_embeddings_raw_filtered(
        &self,
        model_id: Option<&str>,
    ) -> Result<Vec<(String, String, Vec<u8>)>> {
        let query = if model_id.is_some() {
            "SELECT doc_id, doc_type, embedding FROM embeddings WHERE model_id = ?"
        } else {
            "SELECT doc_id, doc_type, embedding FROM embeddings"
        };

        let mut stmt = self.conn.prepare(query)?;

        let rows = if let Some(mid) = model_id {
            stmt.query_map(params![mid], |row| {
                let doc_id: String = row.get(0)?;
                let doc_type: String = row.get(1)?;
                let bytes: Vec<u8> = row.get(2)?;
                Ok((doc_id, doc_type, bytes))
            })?
            .filter_map(std::result::Result::ok)
            .collect()
        } else {
            stmt.query_map([], |row| {
                let doc_id: String = row.get(0)?;
                let doc_type: String = row.get(1)?;
                let bytes: Vec<u8> = row.get(2)?;
                Ok((doc_id, doc_type, bytes))
            })?
            .filter_map(std::result::Result::ok)
            .collect()
        };

        Ok(rows)
    }

    /// Load embeddings filtered by document type.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    ///
    /// # Panics
    ///
    /// Panics if stored embedding bytes are not aligned to 2-byte F16 chunks.
    pub fn load_embeddings_by_type(&self, doc_type: &str) -> Result<Vec<(String, Vec<f32>)>> {
        use half::f16;

        let mut stmt = self
            .conn
            .prepare("SELECT doc_id, embedding FROM embeddings WHERE doc_type = ?")?;

        let rows = stmt.query_map(params![doc_type], |row| {
            let doc_id: String = row.get(0)?;
            let bytes: Vec<u8> = row.get(1)?;

            let floats: Vec<f32> = bytes
                .chunks_exact(2)
                .map(|chunk| {
                    let arr: [u8; 2] = chunk.try_into().unwrap();
                    f16::from_le_bytes(arr).to_f32()
                })
                .collect();

            Ok((doc_id, floats))
        })?;

        let embeddings: Vec<_> = rows.filter_map(std::result::Result::ok).collect();
        Ok(embeddings)
    }

    /// Check if an embedding exists by content hash.
    ///
    /// Useful for skipping re-embedding of unchanged content.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn embedding_exists_by_hash(&self, content_hash: &[u8; 32]) -> Result<bool> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM embeddings WHERE content_hash = ?",
            params![content_hash.as_slice()],
            |row| row.get(0),
        )?;
        Ok(count > 0)
    }

    /// Get the total count of embeddings.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn embedding_count(&self) -> Result<i64> {
        let count: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM embeddings", [], |row| row.get(0))?;
        Ok(count)
    }

    /// Delete all embeddings.
    ///
    /// # Errors
    ///
    /// Returns an error if the database delete fails.
    pub fn clear_embeddings(&self) -> Result<()> {
        self.conn.execute("DELETE FROM embeddings", [])?;
        Ok(())
    }

    // =========================================================================
    // Embedding Job Management
    // =========================================================================

    /// Create or update an embedding job.
    ///
    /// Uses UPSERT to handle the case where a job already exists for the same
    /// (db_path, model_id) pair - in that case, the existing job is reset.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub fn upsert_embedding_job(
        &self,
        db_path: &str,
        model_id: &str,
        total_docs: i64,
    ) -> Result<i64> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            r"
            INSERT INTO embedding_jobs (db_path, model_id, status, total_docs, completed_docs, created_at)
            VALUES (?1, ?2, 'pending', ?3, 0, ?4)
            ON CONFLICT(db_path, model_id) DO UPDATE SET
                status = 'pending',
                total_docs = ?3,
                completed_docs = 0,
                created_at = ?4,
                started_at = NULL,
                completed_at = NULL,
                error_message = NULL
            ",
            params![db_path, model_id, total_docs, now],
        )?;

        let job_id: i64 = self.conn.query_row(
            "SELECT id FROM embedding_jobs WHERE db_path = ? AND model_id = ?",
            params![db_path, model_id],
            |row| row.get(0),
        )?;

        Ok(job_id)
    }

    /// Mark a job as in-progress.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub fn start_embedding_job(&self, job_id: i64) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "UPDATE embedding_jobs SET status = 'in_progress', started_at = ? WHERE id = ?",
            params![now, job_id],
        )?;
        Ok(())
    }

    /// Update job progress (increment completed count).
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub fn update_job_progress(&self, job_id: i64, completed_docs: i64) -> Result<()> {
        self.conn.execute(
            "UPDATE embedding_jobs SET completed_docs = ? WHERE id = ?",
            params![completed_docs, job_id],
        )?;
        Ok(())
    }

    /// Mark a job as completed successfully.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub fn complete_embedding_job(&self, job_id: i64) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            r"
            UPDATE embedding_jobs
            SET status = 'completed', completed_at = ?, completed_docs = total_docs
            WHERE id = ?
            ",
            params![now, job_id],
        )?;
        Ok(())
    }

    /// Mark a job as failed with an error message.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub fn fail_embedding_job(&self, job_id: i64, error_message: &str) -> Result<()> {
        let now = Utc::now().to_rfc3339();
        self.conn.execute(
            "UPDATE embedding_jobs SET status = 'failed', completed_at = ?, error_message = ? WHERE id = ?",
            params![now, error_message, job_id],
        )?;
        Ok(())
    }

    /// Get all pending or in-progress jobs (for resumption on daemon restart).
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_pending_jobs(&self) -> Result<Vec<EmbeddingJob>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT id, db_path, model_id, status, total_docs, completed_docs,
                   created_at, started_at, completed_at, error_message
            FROM embedding_jobs
            WHERE status IN ('pending', 'in_progress')
            ORDER BY created_at ASC
            ",
        )?;

        let jobs = stmt.query_map([], |row| {
            Ok(EmbeddingJob {
                id: row.get(0)?,
                db_path: row.get(1)?,
                model_id: row.get(2)?,
                status: EmbeddingJobStatus::from_str(&row.get::<_, String>(3)?),
                total_docs: row.get(4)?,
                completed_docs: row.get(5)?,
                created_at: parse_rfc3339_or_epoch(row.get(6)?),
                started_at: parse_rfc3339_opt(row.get(7)?),
                completed_at: parse_rfc3339_opt(row.get(8)?),
                error_message: row.get(9)?,
            })
        })?;

        let result: Vec<_> = jobs.filter_map(std::result::Result::ok).collect();
        Ok(result)
    }

    /// Get job status for a specific database path.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_job_status(&self, db_path: &str) -> Result<Vec<EmbeddingJob>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT id, db_path, model_id, status, total_docs, completed_docs,
                   created_at, started_at, completed_at, error_message
            FROM embedding_jobs
            WHERE db_path = ?
            ORDER BY model_id ASC
            ",
        )?;

        let jobs = stmt.query_map(params![db_path], |row| {
            Ok(EmbeddingJob {
                id: row.get(0)?,
                db_path: row.get(1)?,
                model_id: row.get(2)?,
                status: EmbeddingJobStatus::from_str(&row.get::<_, String>(3)?),
                total_docs: row.get(4)?,
                completed_docs: row.get(5)?,
                created_at: parse_rfc3339_or_epoch(row.get(6)?),
                started_at: parse_rfc3339_opt(row.get(7)?),
                completed_at: parse_rfc3339_opt(row.get(8)?),
                error_message: row.get(9)?,
            })
        })?;

        let result: Vec<_> = jobs.filter_map(std::result::Result::ok).collect();
        Ok(result)
    }

    /// Get a specific job by ID.
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_embedding_job(&self, job_id: i64) -> Result<Option<EmbeddingJob>> {
        let result = self.conn.query_row(
            r"
            SELECT id, db_path, model_id, status, total_docs, completed_docs,
                   created_at, started_at, completed_at, error_message
            FROM embedding_jobs
            WHERE id = ?
            ",
            params![job_id],
            |row| {
                Ok(EmbeddingJob {
                    id: row.get(0)?,
                    db_path: row.get(1)?,
                    model_id: row.get(2)?,
                    status: EmbeddingJobStatus::from_str(&row.get::<_, String>(3)?),
                    total_docs: row.get(4)?,
                    completed_docs: row.get(5)?,
                    created_at: parse_rfc3339_or_epoch(row.get(6)?),
                    started_at: parse_rfc3339_opt(row.get(7)?),
                    completed_at: parse_rfc3339_opt(row.get(8)?),
                    error_message: row.get(9)?,
                })
            },
        );

        match result {
            Ok(job) => Ok(Some(job)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    /// Cancel jobs for a given database path and optional model.
    ///
    /// Returns the number of jobs cancelled.
    ///
    /// # Errors
    ///
    /// Returns an error if the database operation fails.
    pub fn cancel_embedding_jobs(&self, db_path: &str, model_id: Option<&str>) -> Result<usize> {
        let now = Utc::now().to_rfc3339();

        let count = match model_id {
            Some(model) => self.conn.execute(
                r"
                UPDATE embedding_jobs
                SET status = 'failed', completed_at = ?, error_message = 'Cancelled'
                WHERE db_path = ? AND model_id = ? AND status IN ('pending', 'in_progress')
                ",
                params![now, db_path, model],
            )?,
            None => self.conn.execute(
                r"
                UPDATE embedding_jobs
                SET status = 'failed', completed_at = ?, error_message = 'Cancelled'
                WHERE db_path = ? AND status IN ('pending', 'in_progress')
                ",
                params![now, db_path],
            )?,
        };

        Ok(count)
    }

    /// Get all jobs (for status display).
    ///
    /// # Errors
    ///
    /// Returns an error if the database query fails.
    pub fn get_all_jobs(&self) -> Result<Vec<EmbeddingJob>> {
        let mut stmt = self.conn.prepare(
            r"
            SELECT id, db_path, model_id, status, total_docs, completed_docs,
                   created_at, started_at, completed_at, error_message
            FROM embedding_jobs
            ORDER BY created_at DESC
            LIMIT 100
            ",
        )?;

        let jobs = stmt.query_map([], |row| {
            Ok(EmbeddingJob {
                id: row.get(0)?,
                db_path: row.get(1)?,
                model_id: row.get(2)?,
                status: EmbeddingJobStatus::from_str(&row.get::<_, String>(3)?),
                total_docs: row.get(4)?,
                completed_docs: row.get(5)?,
                created_at: parse_rfc3339_or_epoch(row.get(6)?),
                started_at: parse_rfc3339_opt(row.get(7)?),
                completed_at: parse_rfc3339_opt(row.get(8)?),
                error_message: row.get(9)?,
            })
        })?;

        let result: Vec<_> = jobs.filter_map(std::result::Result::ok).collect();
        Ok(result)
    }
}

fn format_table_stats(stats: &[TableStat]) -> String {
    if stats.is_empty() {
        return "no tables found".to_string();
    }

    stats
        .iter()
        .map(|stat| {
            let size = stat
                .bytes
                .map_or_else(|| "size unavailable".to_string(), format_bytes_i64);
            format!("{}: {} rows ({size})", stat.name, format_number(stat.rows))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

fn limit_to_i64(limit: usize) -> i64 {
    i64::try_from(limit).unwrap_or(i64::MAX)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TweetUrl;
    use chrono::Duration;
    use rusqlite::params;
    use std::time::Instant;

    fn create_test_tweet(id: &str, text: &str) -> Tweet {
        Tweet {
            id: id.to_string(),
            created_at: Utc::now(),
            full_text: text.to_string(),
            source: None,
            favorite_count: 0,
            retweet_count: 0,
            lang: None,
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
            expanded_url: Some("https://x.com/status/123".to_string()),
        }
    }

    fn create_test_dm(id: &str, text: &str) -> DirectMessage {
        DirectMessage {
            id: id.to_string(),
            conversation_id: "test_conv".to_string(),
            sender_id: "user1".to_string(),
            recipient_id: "user2".to_string(),
            text: text.to_string(),
            created_at: Utc::now(),
            urls: vec![],
            media_urls: vec![],
        }
    }

    fn create_test_grok_message(chat_id: &str, message: &str) -> GrokMessage {
        GrokMessage {
            chat_id: chat_id.to_string(),
            message: message.to_string(),
            sender: "user".to_string(),
            created_at: Utc::now(),
            grok_mode: Some("fun".to_string()),
        }
    }

    #[test]
    fn test_create_database() {
        let storage = Storage::open_memory().unwrap();
        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.tweets_count, 0);
    }

    #[test]
    fn test_store_and_retrieve_tweets() {
        let mut storage = Storage::open_memory().unwrap();

        let tweets = vec![
            create_test_tweet("1", "First tweet about Rust"),
            create_test_tweet("2", "Second tweet about programming"),
        ];

        let count = storage.store_tweets(&tweets).unwrap();
        assert_eq!(count, 2);

        // Verify count
        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.tweets_count, 2);

        // Retrieve by ID
        let tweet = storage.get_tweet("1").unwrap();
        assert!(tweet.is_some());
        let tweet = tweet.unwrap();
        assert_eq!(tweet.id, "1");
        assert_eq!(tweet.full_text, "First tweet about Rust");
    }

    #[test]
    fn test_get_tweet_not_found() {
        let storage = Storage::open_memory().unwrap();
        let tweet = storage.get_tweet("nonexistent").unwrap();
        assert!(tweet.is_none());
    }

    #[test]
    fn test_get_tweet_thread() {
        let mut storage = Storage::open_memory().unwrap();

        let root_date = DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let reply_first_date = DateTime::parse_from_rfc3339("2024-01-01T00:01:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let reply_followup_date = DateTime::parse_from_rfc3339("2024-01-01T00:02:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let branch_date = DateTime::parse_from_rfc3339("2024-01-01T00:03:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let root = Tweet {
            id: "1".to_string(),
            created_at: root_date,
            full_text: "Root tweet".to_string(),
            source: None,
            favorite_count: 0,
            retweet_count: 0,
            lang: None,
            in_reply_to_status_id: None,
            in_reply_to_user_id: None,
            in_reply_to_screen_name: None,
            is_retweet: false,
            hashtags: vec![],
            user_mentions: vec![],
            urls: vec![],
            media: vec![],
        };
        let reply = Tweet {
            id: "2".to_string(),
            created_at: reply_first_date,
            full_text: "Reply tweet".to_string(),
            source: None,
            favorite_count: 0,
            retweet_count: 0,
            lang: None,
            in_reply_to_status_id: Some("1".to_string()),
            in_reply_to_user_id: None,
            in_reply_to_screen_name: None,
            is_retweet: false,
            hashtags: vec![],
            user_mentions: vec![],
            urls: vec![],
            media: vec![],
        };
        let reply2 = Tweet {
            id: "3".to_string(),
            created_at: reply_followup_date,
            full_text: "Reply to reply".to_string(),
            source: None,
            favorite_count: 0,
            retweet_count: 0,
            lang: None,
            in_reply_to_status_id: Some("2".to_string()),
            in_reply_to_user_id: None,
            in_reply_to_screen_name: None,
            is_retweet: false,
            hashtags: vec![],
            user_mentions: vec![],
            urls: vec![],
            media: vec![],
        };
        let branch = Tweet {
            id: "4".to_string(),
            created_at: branch_date,
            full_text: "Branch reply".to_string(),
            source: None,
            favorite_count: 0,
            retweet_count: 0,
            lang: None,
            in_reply_to_status_id: Some("1".to_string()),
            in_reply_to_user_id: None,
            in_reply_to_screen_name: None,
            is_retweet: false,
            hashtags: vec![],
            user_mentions: vec![],
            urls: vec![],
            media: vec![],
        };

        storage
            .store_tweets(&[root, reply, reply2, branch])
            .unwrap();

        let thread = storage.get_tweet_thread("3").unwrap();
        assert_eq!(thread.len(), 4);
        assert_eq!(thread[0].id, "1");
        assert_eq!(thread[1].id, "2");
        assert_eq!(thread[2].id, "3");
        assert_eq!(thread[3].id, "4");
    }

    #[test]
    fn test_store_and_search_tweets_fts() {
        let mut storage = Storage::open_memory().unwrap();

        let tweets = vec![
            create_test_tweet("1", "Rust programming is awesome"),
            create_test_tweet("2", "Python programming is also good"),
            create_test_tweet("3", "Hello world example"),
        ];

        storage.store_tweets(&tweets).unwrap();

        // Search for "programming"
        let results = storage.search_tweets("programming", 10).unwrap();
        assert_eq!(results.len(), 2);

        // Search for "rust"
        let results = storage.search_tweets("rust", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "1");
    }

    #[test]
    fn test_store_likes() {
        let mut storage = Storage::open_memory().unwrap();

        let likes = vec![
            create_test_like("like1", Some("Great content")),
            create_test_like("like2", Some("Another liked tweet")),
            create_test_like("like3", None),
        ];

        let count = storage.store_likes(&likes).unwrap();
        assert_eq!(count, 3);

        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.likes_count, 3);
    }

    #[test]
    fn test_search_likes_fts() {
        let mut storage = Storage::open_memory().unwrap();

        let likes = vec![
            create_test_like("like1", Some("Rust programming content")),
            create_test_like("like2", Some("Python content")),
        ];

        storage.store_likes(&likes).unwrap();

        let results = storage.search_likes("rust", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tweet_id, "like1");
    }

    #[test]
    fn test_store_likes_removes_stale_fts_entries() {
        let mut storage = Storage::open_memory().unwrap();

        let likes = vec![create_test_like("like1", Some("Rust content"))];
        storage.store_likes(&likes).unwrap();

        let results = storage.search_likes("rust", 10).unwrap();
        assert_eq!(results.len(), 1);

        let likes_updated = vec![create_test_like("like1", None)];
        storage.store_likes(&likes_updated).unwrap();

        let results = storage.search_likes("rust", 10).unwrap();
        assert_eq!(results.len(), 0);
    }

    #[test]
    fn test_store_dm_conversations() {
        let mut storage = Storage::open_memory().unwrap();

        let conversations = vec![
            DmConversation {
                conversation_id: "conv1".to_string(),
                messages: vec![
                    create_test_dm("dm1", "Hello!"),
                    create_test_dm("dm2", "Hi there!"),
                ],
            },
            DmConversation {
                conversation_id: "conv2".to_string(),
                messages: vec![create_test_dm("dm3", "Another conversation")],
            },
        ];

        let count = storage.store_dm_conversations(&conversations).unwrap();
        assert_eq!(count, 3); // Total messages

        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.dm_conversations_count, 2);
        assert_eq!(stats.dms_count, 3);
    }

    #[test]
    fn test_search_dms_fts() {
        let mut storage = Storage::open_memory().unwrap();

        let conversations = vec![DmConversation {
            conversation_id: "conv1".to_string(),
            messages: vec![
                create_test_dm("dm1", "Let's discuss Rust"),
                create_test_dm("dm2", "What about Python?"),
            ],
        }];

        storage.store_dm_conversations(&conversations).unwrap();

        let results = storage.search_dms("rust", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, "dm1");
    }

    #[test]
    fn test_get_conversation_messages() {
        let mut storage = Storage::open_memory().unwrap();

        let base_time = DateTime::parse_from_rfc3339("2024-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let later_time = base_time + Duration::minutes(5);

        let conversation = DmConversation {
            conversation_id: "conv1".to_string(),
            messages: vec![
                DirectMessage {
                    id: "dm2".to_string(),
                    conversation_id: "conv1".to_string(),
                    sender_id: "user2".to_string(),
                    recipient_id: "user1".to_string(),
                    text: "Second message".to_string(),
                    created_at: later_time,
                    urls: vec![],
                    media_urls: vec![],
                },
                DirectMessage {
                    id: "dm1".to_string(),
                    conversation_id: "conv1".to_string(),
                    sender_id: "user1".to_string(),
                    recipient_id: "user2".to_string(),
                    text: "First message".to_string(),
                    created_at: base_time,
                    urls: vec![],
                    media_urls: vec![],
                },
            ],
        };

        storage.store_dm_conversations(&[conversation]).unwrap();

        let messages = storage.get_conversation_messages("conv1").unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].id, "dm1");
        assert_eq!(messages[1].id, "dm2");
    }

    #[test]
    fn test_get_conversation_messages_empty() {
        // Empty/missing conversation_id should return empty vec, not error
        let storage = Storage::open_memory().unwrap();
        let messages = storage.get_conversation_messages("nonexistent").unwrap();
        assert!(messages.is_empty());
    }

    #[test]
    fn test_get_conversation_messages_single() {
        let mut storage = Storage::open_memory().unwrap();

        let conversation = DmConversation {
            conversation_id: "single_conv".to_string(),
            messages: vec![DirectMessage {
                id: "dm_only".to_string(),
                conversation_id: "single_conv".to_string(),
                sender_id: "alice".to_string(),
                recipient_id: "bob".to_string(),
                text: "Single message".to_string(),
                created_at: DateTime::parse_from_rfc3339("2024-06-15T12:00:00Z")
                    .unwrap()
                    .with_timezone(&Utc),
                urls: vec![],
                media_urls: vec![],
            }],
        };

        storage.store_dm_conversations(&[conversation]).unwrap();

        let messages = storage.get_conversation_messages("single_conv").unwrap();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].id, "dm_only");
        assert_eq!(messages[0].sender_id, "alice");
        assert_eq!(messages[0].text, "Single message");
    }

    #[test]
    fn test_get_conversation_messages_field_preservation() {
        // Verify URLs and media_urls round-trip correctly
        let mut storage = Storage::open_memory().unwrap();

        let conversation = DmConversation {
            conversation_id: "rich_conv".to_string(),
            messages: vec![DirectMessage {
                id: "dm_rich".to_string(),
                conversation_id: "rich_conv".to_string(),
                sender_id: "user1".to_string(),
                recipient_id: "user2".to_string(),
                text: "Check out this link!".to_string(),
                created_at: DateTime::parse_from_rfc3339("2024-07-01T09:30:00Z")
                    .unwrap()
                    .with_timezone(&Utc),
                urls: vec![TweetUrl {
                    url: "https://t.co/abc".to_string(),
                    expanded_url: Some("https://example.com/article".to_string()),
                    display_url: Some("example.com/article".to_string()),
                }],
                media_urls: vec![
                    "https://pbs.twimg.com/media/abc.jpg".to_string(),
                    "https://pbs.twimg.com/media/def.png".to_string(),
                ],
            }],
        };

        storage.store_dm_conversations(&[conversation]).unwrap();

        let messages = storage.get_conversation_messages("rich_conv").unwrap();
        assert_eq!(messages.len(), 1);

        let msg = &messages[0];
        assert_eq!(msg.urls.len(), 1);
        assert_eq!(msg.urls[0].url, "https://t.co/abc");
        assert_eq!(
            msg.urls[0].expanded_url,
            Some("https://example.com/article".to_string())
        );

        assert_eq!(msg.media_urls.len(), 2);
        assert_eq!(msg.media_urls[0], "https://pbs.twimg.com/media/abc.jpg");
        assert_eq!(msg.media_urls[1], "https://pbs.twimg.com/media/def.png");
    }

    #[test]
    fn test_get_conversation_messages_id_tiebreaker() {
        // When timestamps are identical, id should be used as tiebreaker
        let mut storage = Storage::open_memory().unwrap();

        let same_time = DateTime::parse_from_rfc3339("2024-01-01T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc);

        let conversation = DmConversation {
            conversation_id: "tie_conv".to_string(),
            messages: vec![
                DirectMessage {
                    id: "dm_z".to_string(),
                    conversation_id: "tie_conv".to_string(),
                    sender_id: "user1".to_string(),
                    recipient_id: "user2".to_string(),
                    text: "Message Z".to_string(),
                    created_at: same_time,
                    urls: vec![],
                    media_urls: vec![],
                },
                DirectMessage {
                    id: "dm_a".to_string(),
                    conversation_id: "tie_conv".to_string(),
                    sender_id: "user2".to_string(),
                    recipient_id: "user1".to_string(),
                    text: "Message A".to_string(),
                    created_at: same_time,
                    urls: vec![],
                    media_urls: vec![],
                },
            ],
        };

        storage.store_dm_conversations(&[conversation]).unwrap();

        let messages = storage.get_conversation_messages("tie_conv").unwrap();
        assert_eq!(messages.len(), 2);
        // dm_a should come before dm_z (alphabetical order)
        assert_eq!(messages[0].id, "dm_a");
        assert_eq!(messages[1].id, "dm_z");
    }

    #[test]
    fn test_store_followers() {
        let mut storage = Storage::open_memory().unwrap();

        let followers = vec![
            Follower {
                account_id: "123".to_string(),
                user_link: Some("https://x.com/user123".to_string()),
            },
            Follower {
                account_id: "456".to_string(),
                user_link: None,
            },
        ];

        let count = storage.store_followers(&followers).unwrap();
        assert_eq!(count, 2);

        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.followers_count, 2);
    }

    #[test]
    fn test_store_following() {
        let mut storage = Storage::open_memory().unwrap();

        let following = vec![Following {
            account_id: "789".to_string(),
            user_link: Some("https://x.com/user789".to_string()),
        }];

        let count = storage.store_following(&following).unwrap();
        assert_eq!(count, 1);

        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.following_count, 1);
    }

    #[test]
    fn test_store_blocks() {
        let mut storage = Storage::open_memory().unwrap();

        let blocks = vec![Block {
            account_id: "blocked1".to_string(),
            user_link: None,
        }];
        let count = storage.store_blocks(&blocks).unwrap();
        assert_eq!(count, 1);

        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.blocks_count, 1);
    }

    #[test]
    fn test_store_mutes() {
        let mut storage = Storage::open_memory().unwrap();

        let mutes = vec![
            Mute {
                account_id: "muted1".to_string(),
                user_link: None,
            },
            Mute {
                account_id: "muted2".to_string(),
                user_link: Some("https://x.com/muted2".to_string()),
            },
        ];

        let count = storage.store_mutes(&mutes).unwrap();
        assert_eq!(count, 2);

        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.mutes_count, 2);
    }

    #[test]
    fn test_store_grok_messages() {
        let mut storage = Storage::open_memory().unwrap();

        let messages = vec![
            create_test_grok_message("chat1", "What is AI?"),
            create_test_grok_message("chat1", "AI is artificial intelligence"),
            create_test_grok_message("chat2", "Different topic"),
        ];

        let count = storage.store_grok_messages(&messages).unwrap();
        assert_eq!(count, 3);

        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.grok_messages_count, 3);
    }

    #[test]
    fn test_search_grok_fts() {
        let mut storage = Storage::open_memory().unwrap();

        let messages = vec![
            create_test_grok_message("chat1", "Machine learning algorithms"),
            create_test_grok_message("chat2", "Web development basics"),
        ];

        storage.store_grok_messages(&messages).unwrap();

        let results = storage.search_grok("machine", 10).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].message.contains("Machine learning"));
    }

    #[test]
    fn test_store_archive_info() {
        let storage = Storage::open_memory().unwrap();

        let info = ArchiveInfo {
            account_id: "12345".to_string(),
            username: "testuser".to_string(),
            display_name: Some("Test User".to_string()),
            archive_size_bytes: 1_024_000,
            generation_date: Utc::now(),
            is_partial: false,
        };

        storage.store_archive_info(&info).unwrap();
        // No error means success
    }

    #[test]
    fn test_get_stats_with_data() {
        let mut storage = Storage::open_memory().unwrap();

        // Store various data types
        storage
            .store_tweets(&[create_test_tweet("1", "Tweet")])
            .unwrap();
        storage
            .store_likes(&[create_test_like("l1", Some("Like"))])
            .unwrap();
        storage
            .store_followers(&[Follower {
                account_id: "f1".to_string(),
                user_link: None,
            }])
            .unwrap();
        storage
            .store_following(&[Following {
                account_id: "fo1".to_string(),
                user_link: None,
            }])
            .unwrap();
        storage
            .store_blocks(&[Block {
                account_id: "b1".to_string(),
                user_link: None,
            }])
            .unwrap();
        storage
            .store_mutes(&[Mute {
                account_id: "m1".to_string(),
                user_link: None,
            }])
            .unwrap();
        storage
            .store_grok_messages(&[create_test_grok_message("c1", "Grok")])
            .unwrap();

        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.tweets_count, 1);
        assert_eq!(stats.likes_count, 1);
        assert_eq!(stats.followers_count, 1);
        assert_eq!(stats.following_count, 1);
        assert_eq!(stats.blocks_count, 1);
        assert_eq!(stats.mutes_count, 1);
        assert_eq!(stats.grok_messages_count, 1);
    }

    #[test]
    fn test_get_all_counts_empty_db() {
        let storage = Storage::open_memory().unwrap();

        let counts = storage.get_all_counts().unwrap();
        assert_eq!(counts.tweets_count, 0);
        assert_eq!(counts.likes_count, 0);
        assert_eq!(counts.dms_count, 0);
        assert_eq!(counts.dm_conversations_count, 0);
        assert_eq!(counts.followers_count, 0);
        assert_eq!(counts.following_count, 0);
        assert_eq!(counts.blocks_count, 0);
        assert_eq!(counts.mutes_count, 0);
        assert_eq!(counts.grok_messages_count, 0);
        assert_eq!(counts.first_tweet_date, None);
        assert_eq!(counts.last_tweet_date, None);
    }

    #[test]
    fn test_get_all_counts_only_dms() {
        let mut storage = Storage::open_memory().unwrap();

        storage
            .store_dm_conversations(&[DmConversation {
                conversation_id: "conv1".to_string(),
                messages: vec![create_test_dm("dm1", "Hello")],
            }])
            .unwrap();

        let counts = storage.get_all_counts().unwrap();
        assert_eq!(counts.tweets_count, 0);
        assert_eq!(counts.likes_count, 0);
        assert_eq!(counts.dms_count, 1);
        assert_eq!(counts.dm_conversations_count, 1);
        assert_eq!(counts.followers_count, 0);
        assert_eq!(counts.following_count, 0);
        assert_eq!(counts.blocks_count, 0);
        assert_eq!(counts.mutes_count, 0);
        assert_eq!(counts.grok_messages_count, 0);
        assert_eq!(counts.first_tweet_date, None);
        assert_eq!(counts.last_tweet_date, None);
    }

    #[test]
    fn test_get_all_counts_only_tweets() {
        let mut storage = Storage::open_memory().unwrap();

        storage
            .store_tweets(&[
                create_test_tweet("1", "Only tweet one"),
                create_test_tweet("2", "Only tweet two"),
            ])
            .unwrap();

        let counts = storage.get_all_counts().unwrap();
        assert_eq!(counts.tweets_count, 2);
        assert_eq!(counts.likes_count, 0);
        assert_eq!(counts.dms_count, 0);
        assert_eq!(counts.dm_conversations_count, 0);
        assert_eq!(counts.followers_count, 0);
        assert_eq!(counts.following_count, 0);
        assert_eq!(counts.blocks_count, 0);
        assert_eq!(counts.mutes_count, 0);
        assert_eq!(counts.grok_messages_count, 0);
        assert!(counts.first_tweet_date.is_some());
        assert!(counts.last_tweet_date.is_some());
    }

    #[test]
    fn test_get_all_counts_unicode_text() {
        let mut storage = Storage::open_memory().unwrap();

        storage
            .store_tweets(&[create_test_tweet("1", "Unicode test 🦀🚀")])
            .unwrap();

        let counts = storage.get_all_counts().unwrap();
        assert_eq!(counts.tweets_count, 1);
        assert_eq!(counts.likes_count, 0);
        assert_eq!(counts.dms_count, 0);
        assert_eq!(counts.dm_conversations_count, 0);
        assert_eq!(counts.followers_count, 0);
        assert_eq!(counts.following_count, 0);
        assert_eq!(counts.blocks_count, 0);
        assert_eq!(counts.mutes_count, 0);
        assert_eq!(counts.grok_messages_count, 0);
        assert!(counts.first_tweet_date.is_some());
        assert!(counts.last_tweet_date.is_some());
    }

    #[test]
    fn test_get_all_counts() {
        let mut storage = Storage::open_memory().unwrap();

        let early_date = DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let late_date = DateTime::parse_from_rfc3339("2024-12-31T23:59:59Z")
            .unwrap()
            .with_timezone(&Utc);

        let tweets = vec![
            Tweet {
                id: "1".to_string(),
                created_at: early_date,
                full_text: "Early tweet".to_string(),
                source: None,
                favorite_count: 0,
                retweet_count: 0,
                lang: None,
                in_reply_to_status_id: None,
                in_reply_to_user_id: None,
                in_reply_to_screen_name: None,
                is_retweet: false,
                hashtags: vec![],
                user_mentions: vec![],
                urls: vec![],
                media: vec![],
            },
            Tweet {
                id: "2".to_string(),
                created_at: late_date,
                full_text: "Late tweet".to_string(),
                source: None,
                favorite_count: 0,
                retweet_count: 0,
                lang: None,
                in_reply_to_status_id: None,
                in_reply_to_user_id: None,
                in_reply_to_screen_name: None,
                is_retweet: false,
                hashtags: vec![],
                user_mentions: vec![],
                urls: vec![],
                media: vec![],
            },
        ];
        storage.store_tweets(&tweets).unwrap();
        storage
            .store_likes(&[create_test_like("l1", Some("Like"))])
            .unwrap();
        storage
            .store_dm_conversations(&[DmConversation {
                conversation_id: "conv1".to_string(),
                messages: vec![
                    create_test_dm("dm1", "Hello"),
                    create_test_dm("dm2", "World"),
                ],
            }])
            .unwrap();
        storage
            .store_followers(&[Follower {
                account_id: "f1".to_string(),
                user_link: None,
            }])
            .unwrap();
        storage
            .store_following(&[Following {
                account_id: "fo1".to_string(),
                user_link: None,
            }])
            .unwrap();
        storage
            .store_blocks(&[Block {
                account_id: "b1".to_string(),
                user_link: None,
            }])
            .unwrap();
        storage
            .store_mutes(&[Mute {
                account_id: "m1".to_string(),
                user_link: None,
            }])
            .unwrap();
        storage
            .store_grok_messages(&[create_test_grok_message("c1", "Grok")])
            .unwrap();

        let counts = storage.get_all_counts().unwrap();
        assert_eq!(counts.tweets_count, 2);
        assert_eq!(counts.likes_count, 1);
        assert_eq!(counts.dms_count, 2);
        assert_eq!(counts.dm_conversations_count, 1);
        assert_eq!(counts.followers_count, 1);
        assert_eq!(counts.following_count, 1);
        assert_eq!(counts.blocks_count, 1);
        assert_eq!(counts.mutes_count, 1);
        assert_eq!(counts.grok_messages_count, 1);
        assert_eq!(counts.first_tweet_date, Some(early_date));
        assert_eq!(counts.last_tweet_date, Some(late_date));
    }

    #[test]
    #[ignore = "Large-count regression test; run manually for perf validation"]
    fn test_get_all_counts_large_counts() {
        let storage = Storage::open_memory().unwrap();

        let tx = storage.conn.unchecked_transaction().unwrap();
        let created_at = "2024-01-01T00:00:00Z";
        for i in 0..120_000u32 {
            let id = format!("tweet_{i}");
            tx.execute(
                "INSERT INTO tweets (id, created_at, full_text) VALUES (?1, ?2, ?3)",
                params![id, created_at, "bulk tweet"],
            )
            .unwrap();
        }
        tx.commit().unwrap();

        let counts = storage.get_all_counts().unwrap();
        assert_eq!(counts.tweets_count, 120_000);
        assert_eq!(counts.likes_count, 0);
        assert_eq!(counts.dms_count, 0);
        assert!(counts.first_tweet_date.is_some());
        assert!(counts.last_tweet_date.is_some());
    }

    #[test]
    #[ignore = "Perf comparison is environment-dependent; run manually"]
    fn test_get_all_counts_perf_vs_individual_queries() {
        let storage = Storage::open_memory().unwrap();

        let tx = storage.conn.unchecked_transaction().unwrap();
        let created_at = "2024-01-01T00:00:00Z";
        for i in 0..50_000u32 {
            let id = format!("tweet_{i}");
            tx.execute(
                "INSERT INTO tweets (id, created_at, full_text) VALUES (?1, ?2, ?3)",
                params![id, created_at, "bulk tweet"],
            )
            .unwrap();
        }
        tx.commit().unwrap();

        let iterations = 25;
        let mut all_counts_total = Duration::zero();
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = storage.get_all_counts().unwrap();
            all_counts_total += Duration::from_std(start.elapsed()).unwrap();
        }

        let mut individual_total = Duration::zero();
        for _ in 0..iterations {
            let start = Instant::now();
            let _tweets: i64 = storage
                .conn
                .query_row("SELECT COUNT(*) FROM tweets", [], |row| row.get(0))
                .unwrap();
            let _likes: i64 = storage
                .conn
                .query_row("SELECT COUNT(*) FROM likes", [], |row| row.get(0))
                .unwrap();
            let _dms: i64 = storage
                .conn
                .query_row("SELECT COUNT(*) FROM direct_messages", [], |row| row.get(0))
                .unwrap();
            let _dm_conversations: i64 = storage
                .conn
                .query_row("SELECT COUNT(*) FROM dm_conversations", [], |row| {
                    row.get(0)
                })
                .unwrap();
            let _followers: i64 = storage
                .conn
                .query_row("SELECT COUNT(*) FROM followers", [], |row| row.get(0))
                .unwrap();
            let _following: i64 = storage
                .conn
                .query_row("SELECT COUNT(*) FROM following", [], |row| row.get(0))
                .unwrap();
            let _blocks: i64 = storage
                .conn
                .query_row("SELECT COUNT(*) FROM blocks", [], |row| row.get(0))
                .unwrap();
            let _mutes: i64 = storage
                .conn
                .query_row("SELECT COUNT(*) FROM mutes", [], |row| row.get(0))
                .unwrap();
            let _grok: i64 = storage
                .conn
                .query_row("SELECT COUNT(*) FROM grok_messages", [], |row| row.get(0))
                .unwrap();
            let _first: Option<String> = storage
                .conn
                .query_row("SELECT MIN(created_at) FROM tweets", [], |row| row.get(0))
                .unwrap();
            let _last: Option<String> = storage
                .conn
                .query_row("SELECT MAX(created_at) FROM tweets", [], |row| row.get(0))
                .unwrap();
            individual_total += Duration::from_std(start.elapsed()).unwrap();
        }

        let all_avg_us = all_counts_total.num_microseconds().unwrap_or(0) / i64::from(iterations);
        let ind_avg_us = individual_total.num_microseconds().unwrap_or(0) / i64::from(iterations);

        eprintln!(
            "get_all_counts avg {all_avg_us}us vs individual {ind_avg_us}us (iters={iterations})"
        );
        assert!(
            all_counts_total <= individual_total,
            "Expected get_all_counts to be faster (avg {all_avg_us}us vs {ind_avg_us}us)",
        );
    }

    #[test]
    fn test_tweet_upsert() {
        let mut storage = Storage::open_memory().unwrap();

        // Store initial tweet
        storage
            .store_tweets(&[create_test_tweet("1", "Original text")])
            .unwrap();

        // Store updated tweet with same ID
        let updated = Tweet {
            id: "1".to_string(),
            created_at: Utc::now(),
            full_text: "Updated text".to_string(),
            source: Some("test".to_string()),
            favorite_count: 100, // Changed
            retweet_count: 50,   // Changed
            lang: Some("en".to_string()),
            in_reply_to_status_id: None,
            in_reply_to_user_id: None,
            in_reply_to_screen_name: None,
            is_retweet: false,
            hashtags: vec![],
            user_mentions: vec![],
            urls: vec![],
            media: vec![],
        };
        storage.store_tweets(&[updated]).unwrap();

        // Should still have 1 tweet (upsert, not insert)
        let stats = storage.get_stats().unwrap();
        assert_eq!(stats.tweets_count, 1);

        // Verify updated content
        let tweet = storage.get_tweet("1").unwrap().unwrap();
        assert_eq!(tweet.full_text, "Updated text");
        assert_eq!(tweet.favorite_count, 100);
    }

    #[test]
    fn test_search_limit() {
        let mut storage = Storage::open_memory().unwrap();

        // Store many tweets
        let tweets: Vec<Tweet> = (0..20)
            .map(|i| create_test_tweet(&format!("{i}"), "common search term"))
            .collect();
        storage.store_tweets(&tweets).unwrap();

        // Search with limit
        let results = storage.search_tweets("common", 5).unwrap();
        assert_eq!(results.len(), 5);
    }

    #[test]
    fn test_stats_date_range() {
        let mut storage = Storage::open_memory().unwrap();

        let early_date = DateTime::parse_from_rfc3339("2020-01-01T00:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let late_date = DateTime::parse_from_rfc3339("2024-12-31T23:59:59Z")
            .unwrap()
            .with_timezone(&Utc);

        let tweets = vec![
            Tweet {
                id: "1".to_string(),
                created_at: early_date,
                full_text: "Early tweet".to_string(),
                source: None,
                favorite_count: 0,
                retweet_count: 0,
                lang: None,
                in_reply_to_status_id: None,
                in_reply_to_user_id: None,
                in_reply_to_screen_name: None,
                is_retweet: false,
                hashtags: vec![],
                user_mentions: vec![],
                urls: vec![],
                media: vec![],
            },
            Tweet {
                id: "2".to_string(),
                created_at: late_date,
                full_text: "Late tweet".to_string(),
                source: None,
                favorite_count: 0,
                retweet_count: 0,
                lang: None,
                in_reply_to_status_id: None,
                in_reply_to_user_id: None,
                in_reply_to_screen_name: None,
                is_retweet: false,
                hashtags: vec![],
                user_mentions: vec![],
                urls: vec![],
                media: vec![],
            },
        ];

        storage.store_tweets(&tweets).unwrap();

        let stats = storage.get_stats().unwrap();
        assert!(stats.first_tweet_date.is_some());
        assert!(stats.last_tweet_date.is_some());

        let first = stats.first_tweet_date.unwrap();
        let last = stats.last_tweet_date.unwrap();
        assert!(first < last);
    }

    #[test]
    fn test_schema_version() {
        let storage = Storage::open_memory().unwrap();
        let version = storage.get_schema_version();
        assert_eq!(version, SCHEMA_VERSION);
    }

    #[test]
    fn test_migrate_embeddings_schema_v3() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute_batch(
            r"
            PRAGMA foreign_keys = ON;
            CREATE TABLE meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            INSERT INTO meta (key, value) VALUES ('schema_version', '2');
            CREATE TABLE embeddings (
                doc_id TEXT PRIMARY KEY,
                doc_type TEXT NOT NULL,
                embedding BLOB NOT NULL,
                content_hash BLOB,
                created_at TEXT NOT NULL
            );
            ",
        )
        .unwrap();

        let storage = Storage { conn };
        storage.migrate().unwrap();

        let version = storage.get_schema_version();
        assert_eq!(version, SCHEMA_VERSION);

        let schema_sql: String = storage
            .conn
            .query_row(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='embeddings'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert!(schema_sql.contains("PRIMARY KEY (doc_id, doc_type, model_id)"));
    }

    #[test]
    fn test_embedding_lookup_is_type_aware() {
        let storage = Storage::open_memory().unwrap();

        let emb_tweet = vec![0.1_f32, 0.2];
        let emb_like = vec![0.3_f32, 0.4];
        let hash_tweet = [1_u8; 32];
        let hash_like = [2_u8; 32];

        storage
            .store_embedding("123", "tweet", &emb_tweet, Some(&hash_tweet))
            .unwrap();
        storage
            .store_embedding("123", "like", &emb_like, Some(&hash_like))
            .unwrap();

        let got_tweet = storage.get_embedding("123", "tweet").unwrap();
        let got_like = storage.get_embedding("123", "like").unwrap();
        let got_missing = storage.get_embedding("123", "dm").unwrap();

        let assert_vec_approx = |left: &[f32], right: &[f32]| {
            assert_eq!(left.len(), right.len());
            for (a, b) in left.iter().zip(right.iter()) {
                assert!((a - b).abs() < 1e-3, "expected {b} ~= {a}");
            }
        };

        let got_tweet = got_tweet.expect("tweet embedding missing");
        let got_like = got_like.expect("like embedding missing");

        assert_vec_approx(&got_tweet, &emb_tweet);
        assert_vec_approx(&got_like, &emb_like);
        assert_eq!(got_missing, None);

        let hash_tweet_got = storage.get_embedding_hash("123", "tweet").unwrap();
        let hash_like_got = storage.get_embedding_hash("123", "like").unwrap();
        let hash_missing = storage.get_embedding_hash("123", "dm").unwrap();

        assert_eq!(hash_tweet_got, Some(hash_tweet));
        assert_eq!(hash_like_got, Some(hash_like));
        assert_eq!(hash_missing, None);
    }

    #[test]
    fn test_load_embedding_hashes_by_doc_matches_get() {
        let storage = Storage::open_memory().unwrap();

        let emb_tweet = vec![0.9_f32, 0.8];
        let emb_like = vec![0.7_f32, 0.6];
        let hash_tweet = [5_u8; 32];
        let hash_like = [6_u8; 32];

        storage
            .store_embedding("999", "tweet", &emb_tweet, Some(&hash_tweet))
            .unwrap();
        storage
            .store_embedding("999", "like", &emb_like, Some(&hash_like))
            .unwrap();

        let hashes_by_doc = storage.load_embedding_hashes_by_doc().unwrap();
        let tweet_hash = hashes_by_doc
            .get("999")
            .and_then(|by_type| by_type.get("tweet"))
            .copied();
        let like_hash = hashes_by_doc
            .get("999")
            .and_then(|by_type| by_type.get("like"))
            .copied();

        assert_eq!(
            tweet_hash,
            storage.get_embedding_hash("999", "tweet").unwrap()
        );
        assert_eq!(
            like_hash,
            storage.get_embedding_hash("999", "like").unwrap()
        );
        assert_eq!(
            hashes_by_doc
                .get("999")
                .and_then(|by_type| by_type.get("dm")),
            None
        );
    }

    #[test]
    fn test_load_embedding_hashes_for_docs_subset() {
        let storage = Storage::open_memory().unwrap();

        let emb_tweet = vec![0.1_f32, 0.2];
        let emb_like = vec![0.3_f32, 0.4];
        let hash_tweet = [11_u8; 32];
        let hash_like = [12_u8; 32];

        storage
            .store_embedding("alpha", "tweet", &emb_tweet, Some(&hash_tweet))
            .unwrap();
        storage
            .store_embedding("beta", "like", &emb_like, Some(&hash_like))
            .unwrap();

        let lookups = vec![
            ("alpha".to_string(), "tweet".to_string()),
            ("beta".to_string(), "like".to_string()),
            ("missing".to_string(), "dm".to_string()),
        ];
        let hashes = storage.load_embedding_hashes_for_docs(&lookups).unwrap();

        assert_eq!(
            hashes.get("alpha").and_then(|by_type| by_type.get("tweet")),
            Some(&hash_tweet)
        );
        assert_eq!(
            hashes.get("beta").and_then(|by_type| by_type.get("like")),
            Some(&hash_like)
        );
        assert_eq!(hashes.get("missing"), None);
    }

    #[test]
    fn test_load_embedding_hashes_for_docs_chunking() {
        let storage = Storage::open_memory().unwrap();

        let emb = vec![0.5_f32, 0.6];
        let hash = [33_u8; 32];
        storage
            .store_embedding("doc-0", "tweet", &emb, Some(&hash))
            .unwrap();

        let chunk_size = SQLITE_BATCH_SIZE / 2;
        let mut lookups: Vec<(String, String)> = Vec::with_capacity(chunk_size + 2);
        for i in 0..=chunk_size {
            lookups.push((format!("doc-{i}"), "tweet".to_string()));
        }

        let hashes = storage.load_embedding_hashes_for_docs(&lookups).unwrap();
        assert_eq!(
            hashes.get("doc-0").and_then(|by_type| by_type.get("tweet")),
            Some(&hash)
        );
        assert_eq!(
            hashes
                .get(&format!("doc-{chunk_size}"))
                .and_then(|by_type| by_type.get("tweet")),
            None
        );
    }

    #[test]
    fn test_load_embeddings_by_hashes() {
        let storage = Storage::open_memory().unwrap();

        let emb_tweet = vec![0.12_f32, 0.34];
        let emb_like = vec![0.56_f32, 0.78];
        let hash_tweet = [7_u8; 32];
        let hash_like = [8_u8; 32];

        storage
            .store_embedding("321", "tweet", &emb_tweet, Some(&hash_tweet))
            .unwrap();
        storage
            .store_embedding("321", "like", &emb_like, Some(&hash_like))
            .unwrap();

        let hashes = vec![hash_tweet, hash_like, [9_u8; 32]];
        let embeddings = storage.load_embeddings_by_hashes(&hashes).unwrap();
        assert_eq!(embeddings.len(), 2);

        let assert_vec_approx = |left: &[f32], right: &[f32]| {
            assert_eq!(left.len(), right.len());
            for (a, b) in left.iter().zip(right.iter()) {
                assert!((a - b).abs() < 1e-3, "expected {b} ~= {a}");
            }
        };

        assert_vec_approx(embeddings.get(&hash_tweet).unwrap(), &emb_tweet);
        assert_vec_approx(embeddings.get(&hash_like).unwrap(), &emb_like);
    }

    #[test]
    fn test_load_embeddings_by_type_avoids_collisions() {
        let storage = Storage::open_memory().unwrap();

        let emb_tweet = vec![0.11_f32, 0.22];
        let emb_like = vec![0.33_f32, 0.44];
        let hash_tweet = [3_u8; 32];
        let hash_like = [4_u8; 32];

        storage
            .store_embedding("42", "tweet", &emb_tweet, Some(&hash_tweet))
            .unwrap();
        storage
            .store_embedding("42", "like", &emb_like, Some(&hash_like))
            .unwrap();

        let all = storage.load_all_embeddings().unwrap();
        assert_eq!(all.len(), 2);

        let tweets = storage.load_embeddings_by_type("tweet").unwrap();
        assert_eq!(tweets.len(), 1);
        assert_eq!(tweets[0].0, "42");

        let likes = storage.load_embeddings_by_type("like").unwrap();
        assert_eq!(likes.len(), 1);
        assert_eq!(likes[0].0, "42");

        let assert_vec_approx = |left: &[f32], right: &[f32]| {
            assert_eq!(left.len(), right.len());
            for (a, b) in left.iter().zip(right.iter()) {
                assert!((a - b).abs() < 1e-3, "expected {b} ~= {a}");
            }
        };

        assert_vec_approx(&tweets[0].1, &emb_tweet);
        assert_vec_approx(&likes[0].1, &emb_like);
    }

    #[test]
    fn test_database_health_checks_pass() {
        let storage = Storage::open_memory().unwrap();
        let checks = storage.database_health_checks();

        let integrity = checks
            .iter()
            .find(|c| c.name == "PRAGMA integrity_check")
            .expect("integrity check missing");
        assert_eq!(integrity.status, CheckStatus::Pass);

        let schema = checks
            .iter()
            .find(|c| c.name == "Schema version")
            .expect("schema check missing");
        assert_eq!(schema.status, CheckStatus::Pass);
    }

    #[test]
    fn test_database_health_orphaned_fts() {
        let storage = Storage::open_memory().unwrap();

        storage
            .connection()
            .execute(
                "INSERT INTO fts_tweets (tweet_id, full_text) VALUES ('orphan', 'text')",
                [],
            )
            .unwrap();

        let checks = storage.database_health_checks();
        let orphaned = checks
            .iter()
            .find(|c| c.name == "FTS orphaned rows (tweets)")
            .expect("orphan check missing");
        assert_eq!(orphaned.status, CheckStatus::Warning);
    }

    #[test]
    fn test_store_embedding_with_model_id() {
        let storage = Storage::open_memory().unwrap();
        let embedding = vec![0.1_f32, 0.2, 0.3, 0.4];

        // Store with default model_id
        storage
            .store_embedding("doc1", "tweet", &embedding, None)
            .unwrap();

        // Store same doc with different model_id
        storage
            .store_embedding_with_model("doc1", "tweet", &embedding, None, "fast")
            .unwrap();
        storage
            .store_embedding_with_model("doc1", "tweet", &embedding, None, "quality")
            .unwrap();

        // All three should coexist (different PKs)
        let all = storage.load_all_embeddings().unwrap();
        let doc1_count = all.iter().filter(|(id, _, _)| id == "doc1").count();
        assert_eq!(
            doc1_count, 3,
            "should have 3 embeddings for doc1 (default, fast, quality)"
        );
    }

    #[test]
    fn test_load_filtered_by_model() {
        let storage = Storage::open_memory().unwrap();
        let emb_fast = vec![1.0_f32, 0.0, 0.0, 0.0];
        let emb_quality = vec![0.0_f32, 1.0, 0.0, 0.0];

        storage
            .store_embedding_with_model("doc1", "tweet", &emb_fast, None, "fast")
            .unwrap();
        storage
            .store_embedding_with_model("doc1", "tweet", &emb_quality, None, "quality")
            .unwrap();
        storage
            .store_embedding_with_model("doc2", "like", &emb_fast, None, "fast")
            .unwrap();

        // Filter by "fast" model
        let fast_only = storage.load_all_embeddings_filtered(Some("fast")).unwrap();
        assert_eq!(fast_only.len(), 2);

        // Filter by "quality" model
        let quality_only = storage
            .load_all_embeddings_filtered(Some("quality"))
            .unwrap();
        assert_eq!(quality_only.len(), 1);
        assert_eq!(quality_only[0].0, "doc1");

        // No filter returns all
        let all = storage.load_all_embeddings_filtered(None).unwrap();
        assert_eq!(all.len(), 3);

        // Raw filtered
        let fast_raw = storage
            .load_all_embeddings_raw_filtered(Some("fast"))
            .unwrap();
        assert_eq!(fast_raw.len(), 2);
    }

    #[test]
    fn test_schema_migration_v5() {
        // Open in-memory creates schema at current version (now v6 after
        // the bookmarks_count migration) with embedding_jobs table.
        // Verify the tables exist and work. The hard-coded version here
        // tracks SCHEMA_VERSION at the top of this file; bump both
        // together when adding a new migration.
        let storage = Storage::open_memory().unwrap();
        let version = storage.get_schema_version();
        assert_eq!(version, SCHEMA_VERSION);

        // Store with model_id should work
        storage
            .store_embedding_with_model("d1", "tweet", &[0.5; 4], None, "fast")
            .unwrap();
        storage
            .store_embedding_with_model("d1", "tweet", &[0.5; 4], None, "quality")
            .unwrap();

        let fast = storage.load_all_embeddings_filtered(Some("fast")).unwrap();
        assert_eq!(fast.len(), 1);
        let quality = storage
            .load_all_embeddings_filtered(Some("quality"))
            .unwrap();
        assert_eq!(quality.len(), 1);
    }

    #[test]
    #[allow(clippy::type_complexity)]
    fn test_store_embeddings_batch_with_model() {
        let storage = Storage::open_memory().unwrap();
        let batch: Vec<(String, String, Vec<f32>, Option<[u8; 32]>)> = vec![
            ("d1".into(), "tweet".into(), vec![0.1; 4], None),
            ("d2".into(), "like".into(), vec![0.2; 4], None),
        ];

        let count = storage
            .store_embeddings_batch_with_model(&batch, "fast")
            .unwrap();
        assert_eq!(count, 2);

        let fast = storage.load_all_embeddings_filtered(Some("fast")).unwrap();
        assert_eq!(fast.len(), 2);

        // Default model should have zero
        let default_embs = storage
            .load_all_embeddings_filtered(Some("default"))
            .unwrap();
        assert_eq!(default_embs.len(), 0);
    }

    #[test]
    fn test_load_hashes_and_embeddings_by_model() {
        let storage = Storage::open_memory().unwrap();
        let hash_a = [0xAA_u8; 32];
        let hash_b = [0xBB_u8; 32];
        let emb_fast = vec![1.0_f32, 0.0, 0.0, 0.0];
        let emb_quality = vec![0.0_f32, 1.0, 0.0, 0.0];

        // Store same doc with different models and different content hashes
        storage
            .store_embedding_with_model("doc1", "tweet", &emb_fast, Some(&hash_a), "fast")
            .unwrap();
        storage
            .store_embedding_with_model("doc1", "tweet", &emb_quality, Some(&hash_b), "quality")
            .unwrap();

        // load_embedding_hashes_by_doc_for_model: should isolate by model
        let fast_hashes = storage
            .load_embedding_hashes_by_doc_for_model("fast")
            .unwrap();
        assert_eq!(fast_hashes.len(), 1);
        assert_eq!(fast_hashes["doc1"]["tweet"], hash_a);

        let quality_hashes = storage
            .load_embedding_hashes_by_doc_for_model("quality")
            .unwrap();
        assert_eq!(quality_hashes.len(), 1);
        assert_eq!(quality_hashes["doc1"]["tweet"], hash_b);

        // Nonexistent model returns empty
        let none_hashes = storage
            .load_embedding_hashes_by_doc_for_model("nonexistent")
            .unwrap();
        assert!(none_hashes.is_empty());

        // load_embeddings_by_hashes_for_model: should return only matching model
        let fast_embs = storage
            .load_embeddings_by_hashes_for_model(&[hash_a], "fast")
            .unwrap();
        assert_eq!(fast_embs.len(), 1);
        // First component should be ~1.0 (fast embedding)
        assert!(fast_embs[&hash_a][0] > 0.5);

        let quality_embs = storage
            .load_embeddings_by_hashes_for_model(&[hash_b], "quality")
            .unwrap();
        assert_eq!(quality_embs.len(), 1);
        // Second component should be ~1.0 (quality embedding)
        assert!(quality_embs[&hash_b][1] > 0.5);

        // Cross-model lookup should return empty (hash_a belongs to fast, not quality)
        let cross = storage
            .load_embeddings_by_hashes_for_model(&[hash_a], "quality")
            .unwrap();
        assert!(cross.is_empty());
    }

    #[test]
    fn test_embedding_job_lifecycle() {
        let storage = Storage::open_memory().unwrap();

        // Create a job
        let job_id = storage
            .upsert_embedding_job("/path/to/db", "fast", 1000)
            .unwrap();
        assert!(job_id > 0);

        // Check job was created with pending status
        let job = storage.get_embedding_job(job_id).unwrap().unwrap();
        assert_eq!(job.status, EmbeddingJobStatus::Pending);
        assert_eq!(job.total_docs, 1000);
        assert_eq!(job.completed_docs, 0);

        // Start the job
        storage.start_embedding_job(job_id).unwrap();
        let job = storage.get_embedding_job(job_id).unwrap().unwrap();
        assert_eq!(job.status, EmbeddingJobStatus::InProgress);
        assert!(job.started_at.is_some());

        // Update progress
        storage.update_job_progress(job_id, 500).unwrap();
        let job = storage.get_embedding_job(job_id).unwrap().unwrap();
        assert_eq!(job.completed_docs, 500);

        // Complete the job
        storage.complete_embedding_job(job_id).unwrap();
        let job = storage.get_embedding_job(job_id).unwrap().unwrap();
        assert_eq!(job.status, EmbeddingJobStatus::Completed);
        assert!(job.completed_at.is_some());
        assert_eq!(job.completed_docs, job.total_docs);
    }

    #[test]
    fn test_embedding_job_upsert_resets() {
        let storage = Storage::open_memory().unwrap();

        // Create a job
        let job_id1 = storage
            .upsert_embedding_job("/path/to/db", "fast", 1000)
            .unwrap();
        storage.start_embedding_job(job_id1).unwrap();
        storage.update_job_progress(job_id1, 500).unwrap();

        // Upsert same db_path + model_id resets the job
        let job_id2 = storage
            .upsert_embedding_job("/path/to/db", "fast", 2000)
            .unwrap();
        assert_eq!(job_id1, job_id2); // Same ID

        let job = storage.get_embedding_job(job_id2).unwrap().unwrap();
        assert_eq!(job.status, EmbeddingJobStatus::Pending);
        assert_eq!(job.total_docs, 2000);
        assert_eq!(job.completed_docs, 0);
        assert!(job.started_at.is_none());
    }

    #[test]
    fn test_embedding_job_fail() {
        let storage = Storage::open_memory().unwrap();

        let job_id = storage
            .upsert_embedding_job("/path/to/db", "quality", 500)
            .unwrap();
        storage.start_embedding_job(job_id).unwrap();
        storage.fail_embedding_job(job_id, "Disk full").unwrap();

        let job = storage.get_embedding_job(job_id).unwrap().unwrap();
        assert_eq!(job.status, EmbeddingJobStatus::Failed);
        assert_eq!(job.error_message, Some("Disk full".to_string()));
    }

    #[test]
    fn test_get_pending_jobs() {
        let storage = Storage::open_memory().unwrap();

        // Create multiple jobs
        storage.upsert_embedding_job("/db1", "fast", 100).unwrap();
        let job2_id = storage
            .upsert_embedding_job("/db1", "quality", 200)
            .unwrap();
        storage.start_embedding_job(job2_id).unwrap();
        let job3_id = storage.upsert_embedding_job("/db2", "fast", 300).unwrap();
        storage.start_embedding_job(job3_id).unwrap();
        storage.complete_embedding_job(job3_id).unwrap();

        // Get pending jobs (pending + in_progress)
        let pending = storage.get_pending_jobs().unwrap();
        assert_eq!(pending.len(), 2); // job1 (pending) and job2 (in_progress)
    }

    #[test]
    fn test_cancel_embedding_jobs() {
        let storage = Storage::open_memory().unwrap();

        storage.upsert_embedding_job("/db1", "fast", 100).unwrap();
        storage
            .upsert_embedding_job("/db1", "quality", 200)
            .unwrap();
        storage.upsert_embedding_job("/db2", "fast", 300).unwrap();

        // Cancel all jobs for /db1
        let cancelled = storage.cancel_embedding_jobs("/db1", None).unwrap();
        assert_eq!(cancelled, 2);

        // Cancel specific model for /db2
        let cancelled = storage.cancel_embedding_jobs("/db2", Some("fast")).unwrap();
        assert_eq!(cancelled, 1);

        // All should be failed now
        let pending = storage.get_pending_jobs().unwrap();
        assert_eq!(pending.len(), 0);
    }

    #[test]
    fn test_get_job_status_for_db() {
        let storage = Storage::open_memory().unwrap();

        storage.upsert_embedding_job("/db1", "fast", 100).unwrap();
        storage
            .upsert_embedding_job("/db1", "quality", 200)
            .unwrap();
        storage.upsert_embedding_job("/db2", "fast", 300).unwrap();

        let jobs = storage.get_job_status("/db1").unwrap();
        assert_eq!(jobs.len(), 2);
        assert!(jobs.iter().all(|j| j.db_path == "/db1"));
    }
}
