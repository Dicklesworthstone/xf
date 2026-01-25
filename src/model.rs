//! Data models for X archive data.
//!
//! These structures represent the normalized form of X data after parsing
//! from the JavaScript export format.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A tweet from the archive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tweet {
    pub id: String,
    pub created_at: DateTime<Utc>,
    pub full_text: String,
    pub source: Option<String>,
    pub favorite_count: i64,
    pub retweet_count: i64,
    pub lang: Option<String>,
    pub in_reply_to_status_id: Option<String>,
    pub in_reply_to_user_id: Option<String>,
    pub in_reply_to_screen_name: Option<String>,
    pub is_retweet: bool,
    pub hashtags: Vec<String>,
    pub user_mentions: Vec<UserMention>,
    pub urls: Vec<TweetUrl>,
    pub media: Vec<TweetMedia>,
}

/// A user mention in a tweet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserMention {
    pub id: String,
    pub screen_name: String,
    pub name: Option<String>,
}

/// A URL in a tweet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TweetUrl {
    pub url: String,
    pub expanded_url: Option<String>,
    pub display_url: Option<String>,
}

/// Media attached to a tweet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TweetMedia {
    pub id: String,
    pub media_type: String,
    pub url: String,
    pub local_path: Option<String>,
}

/// A liked tweet
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Like {
    pub tweet_id: String,
    pub full_text: Option<String>,
    pub expanded_url: Option<String>,
}

/// A direct message conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DmConversation {
    pub conversation_id: String,
    pub messages: Vec<DirectMessage>,
}

/// Summary information for a DM conversation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DmConversationSummary {
    pub conversation_id: String,
    pub participant_ids: Vec<String>,
    pub message_count: i64,
    pub first_message_at: Option<DateTime<Utc>>,
    pub last_message_at: Option<DateTime<Utc>>,
}

/// A direct message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectMessage {
    pub id: String,
    pub conversation_id: String,
    pub sender_id: String,
    pub recipient_id: String,
    pub text: String,
    pub created_at: DateTime<Utc>,
    pub urls: Vec<TweetUrl>,
    pub media_urls: Vec<String>,
}

/// A follower
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Follower {
    pub account_id: String,
    pub user_link: Option<String>,
}

/// A following (account the user follows)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Following {
    pub account_id: String,
    pub user_link: Option<String>,
}

/// A blocked account
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub account_id: String,
    pub user_link: Option<String>,
}

/// A muted account
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Mute {
    pub account_id: String,
    pub user_link: Option<String>,
}

/// Account information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub account_id: String,
    pub username: String,
    pub display_name: Option<String>,
    pub email: Option<String>,
    pub created_at: Option<DateTime<Utc>>,
    pub created_via: Option<String>,
}

/// Profile information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Profile {
    pub bio: Option<String>,
    pub website: Option<String>,
    pub location: Option<String>,
    pub avatar_url: Option<String>,
    pub header_url: Option<String>,
}

/// Grok chat message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GrokMessage {
    pub chat_id: String,
    pub message: String,
    pub sender: String,
    pub created_at: DateTime<Utc>,
    pub grok_mode: Option<String>,
}

/// Archive metadata from manifest.js
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveInfo {
    pub account_id: String,
    pub username: String,
    pub display_name: Option<String>,
    pub archive_size_bytes: i64,
    pub generation_date: DateTime<Utc>,
    pub is_partial: bool,
}

/// Statistics about the indexed archive
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchiveStats {
    pub tweets_count: i64,
    pub likes_count: i64,
    pub dms_count: i64,
    pub dm_conversations_count: i64,
    pub followers_count: i64,
    pub following_count: i64,
    pub blocks_count: i64,
    pub mutes_count: i64,
    pub grok_messages_count: i64,
    pub first_tweet_date: Option<DateTime<Utc>>,
    pub last_tweet_date: Option<DateTime<Utc>>,
    pub index_built_at: DateTime<Utc>,
}

/// Search result item
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub result_type: SearchResultType,
    pub id: String,
    pub text: String,
    pub created_at: DateTime<Utc>,
    pub score: f32,
    pub highlights: Vec<String>,
    pub metadata: serde_json::Value,
    /// Reranker score if reranking was applied, None otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rerank_score: Option<f32>,
}

/// Type of search result
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SearchResultType {
    Tweet,
    Like,
    DirectMessage,
    GrokMessage,
}

impl std::fmt::Display for SearchResultType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Tweet => write!(f, "tweet"),
            Self::Like => write!(f, "like"),
            Self::DirectMessage => write!(f, "dm"),
            Self::GrokMessage => write!(f, "grok"),
        }
    }
}
