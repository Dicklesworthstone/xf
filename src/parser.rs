//! X archive data parser.
//!
//! Handles parsing the JavaScript-wrapped JSON format used in X data exports.
//! Files are formatted as: `window.YTD.<datatype>.part0 = [...]`

use crate::model::{
    Account, ArchiveInfo, Block, Bookmark, DirectMessage, DmConversation, Follower, Following,
    GrokMessage, Like, Mute, Profile, Tweet, TweetMedia, TweetUrl, UserMention,
};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use glob::glob;
use rayon::prelude::*;
use serde_json::Value;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use tracing::info;
use walkdir::WalkDir;

/// Parser for X archive data
pub struct ArchiveParser {
    archive_path: std::path::PathBuf,
}

impl ArchiveParser {
    pub fn new(archive_path: impl AsRef<Path>) -> Self {
        Self {
            archive_path: archive_path.as_ref().to_path_buf(),
        }
    }

    /// Parse the JavaScript file format and extract JSON
    #[allow(clippy::unused_self)]
    fn parse_js_file(&self, content: &str) -> Result<Value> {
        // Format: window.YTD.<type>.part<n> = [...]
        // Extract everything after the first '=' and trim whitespace/semicolon.
        let mut parts = content.splitn(2, '=');
        let _prefix = parts
            .next()
            .context("Invalid JS file format: missing prefix")?;
        let json_part = parts
            .next()
            .context("Invalid JS file format: no '=' found")?;

        let mut json_str = json_part.trim();
        if let Some(stripped) = json_str.strip_suffix(';') {
            json_str = stripped.trim_end();
        }

        serde_json::from_str(json_str).context("Failed to parse JSON from JS file")
    }

    /// Read and parse a JS data file
    fn read_data_file(&self, filename: &str) -> Result<Value> {
        let path = self.archive_path.join("data").join(filename);
        if !path.exists() {
            return Ok(Value::Array(vec![]));
        }
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        self.parse_js_file(&content)
    }

    /// Read and parse a required JS data file.
    fn read_required_data_file(&self, filename: &str) -> Result<Value> {
        let path = self.archive_path.join("data").join(filename);
        if !path.exists() {
            anyhow::bail!("Required archive file missing: {}", path.display());
        }
        let content = std::fs::read_to_string(&path)
            .with_context(|| format!("Failed to read {}", path.display()))?;
        self.parse_js_file(&content)
    }

    /// Parse X's date format: "Fri Jan 09 15:12:21 +0000 2026"
    fn parse_x_date(date_str: &str) -> Option<DateTime<Utc>> {
        // X format: "Fri Jan 09 15:12:21 +0000 2026"
        DateTime::parse_from_str(date_str, "%a %b %d %H:%M:%S %z %Y")
            .ok()
            .map(|dt| dt.with_timezone(&Utc))
    }

    /// Parse ISO 8601 date format
    fn parse_iso_date(date_str: &str) -> Option<DateTime<Utc>> {
        DateTime::parse_from_rfc3339(date_str)
            .ok()
            .map(|dt| dt.with_timezone(&Utc))
    }

    /// Parse archive metadata from manifest.js.
    ///
    /// # Errors
    ///
    /// Returns an error if the manifest file cannot be read or parsed.
    pub fn parse_manifest(&self) -> Result<ArchiveInfo> {
        let data = self.read_required_data_file("manifest.js")?;
        if !data.is_object() {
            anyhow::bail!("Invalid manifest format: expected JSON object");
        }

        let user_info = &data["userInfo"];
        let archive_info = &data["archiveInfo"];

        Ok(ArchiveInfo {
            account_id: user_info["accountId"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            username: user_info["userName"]
                .as_str()
                .unwrap_or_default()
                .to_string(),
            display_name: user_info["displayName"].as_str().map(String::from),
            archive_size_bytes: Self::parse_i64(&archive_info["sizeBytes"]).unwrap_or(0),
            generation_date: archive_info["generationDate"]
                .as_str()
                .and_then(Self::parse_iso_date)
                .unwrap_or_else(|| DateTime::<Utc>::from_timestamp(0, 0).unwrap_or_else(Utc::now)),
            is_partial: archive_info["isPartialArchive"].as_bool().unwrap_or(false),
        })
    }

    /// Parse all tweets from tweets.js.
    ///
    /// # Errors
    ///
    /// Returns an error if the tweets file cannot be read or parsed.
    pub fn parse_tweets(&self) -> Result<Vec<Tweet>> {
        info!("Parsing tweets...");

        let mut files = Vec::new();
        let tweets_path = self.archive_path.join("data").join("tweets.js");
        if tweets_path.exists() {
            files.push(tweets_path);
        }
        files.extend(self.collect_data_files("tweets-part*.js")?);

        if files.is_empty() {
            info!("No tweet files found.");
            return Ok(Vec::new());
        }

        let mut tweets: Vec<Tweet> = Vec::new();
        let mut seen_ids: HashSet<String> = HashSet::new();

        for path in files {
            let content = std::fs::read_to_string(&path)
                .with_context(|| format!("Failed to read {}", path.display()))?;
            let data = self.parse_js_file(&content)?;
            let Some(items) = data.as_array() else {
                continue;
            };

            let file_tweets: Vec<Tweet> = items
                .par_iter()
                .filter_map(|item| {
                    let tweet = &item["tweet"];
                    Some(Tweet {
                        id: tweet["id_str"].as_str()?.to_string(),
                        created_at: tweet["created_at"].as_str().and_then(Self::parse_x_date)?,
                        full_text: tweet["full_text"].as_str()?.to_string(),
                        source: tweet["source"].as_str().map(|s| {
                            // Extract text from HTML anchor tag
                            s.split('>')
                                .nth(1)
                                .and_then(|s| s.split('<').next())
                                .unwrap_or(s)
                                .to_string()
                        }),
                        favorite_count: Self::parse_i64(&tweet["favorite_count"]).unwrap_or(0),
                        retweet_count: Self::parse_i64(&tweet["retweet_count"]).unwrap_or(0),
                        lang: tweet["lang"].as_str().map(String::from),
                        in_reply_to_status_id: tweet["in_reply_to_status_id_str"]
                            .as_str()
                            .map(String::from),
                        in_reply_to_user_id: tweet["in_reply_to_user_id_str"]
                            .as_str()
                            .map(String::from),
                        in_reply_to_screen_name: tweet["in_reply_to_screen_name"]
                            .as_str()
                            .map(String::from),
                        is_retweet: tweet["retweeted"].as_bool().unwrap_or(false),
                        hashtags: Self::parse_hashtags(&tweet["entities"]["hashtags"]),
                        user_mentions: Self::parse_user_mentions(
                            &tweet["entities"]["user_mentions"],
                        ),
                        urls: Self::parse_urls(&tweet["entities"]["urls"]),
                        media: Self::parse_media(&tweet["entities"]["media"]),
                    })
                })
                .collect();

            for tweet in file_tweets {
                if seen_ids.insert(tweet.id.clone()) {
                    tweets.push(tweet);
                }
            }
        }

        info!("Parsed {} tweets", tweets.len());
        Ok(tweets)
    }

    fn parse_hashtags(value: &Value) -> Vec<String> {
        Self::as_array_or_empty(value)
            .iter()
            .filter_map(|h| h["text"].as_str().map(String::from))
            .collect()
    }

    fn parse_user_mentions(value: &Value) -> Vec<UserMention> {
        Self::as_array_or_empty(value)
            .iter()
            .filter_map(|m| {
                Some(UserMention {
                    id: m["id_str"].as_str()?.to_string(),
                    screen_name: m["screen_name"].as_str()?.to_string(),
                    name: m["name"].as_str().map(String::from),
                })
            })
            .collect()
    }

    fn parse_urls(value: &Value) -> Vec<TweetUrl> {
        Self::as_array_or_empty(value)
            .iter()
            .filter_map(|u| {
                Some(TweetUrl {
                    url: u["url"].as_str()?.to_string(),
                    expanded_url: u["expanded_url"].as_str().map(String::from),
                    display_url: u["display_url"].as_str().map(String::from),
                })
            })
            .collect()
    }

    fn parse_media(value: &Value) -> Vec<TweetMedia> {
        Self::as_array_or_empty(value)
            .iter()
            .filter_map(|m| {
                Some(TweetMedia {
                    id: m["id_str"].as_str()?.to_string(),
                    media_type: m["type"].as_str().unwrap_or("photo").to_string(),
                    url: m["media_url_https"]
                        .as_str()
                        .or_else(|| m["media_url"].as_str())?
                        .to_string(),
                    local_path: None,
                })
            })
            .collect()
    }

    fn as_array_or_empty(value: &Value) -> &[Value] {
        value.as_array().map_or(&[], |items| items.as_slice())
    }

    fn parse_i64(value: &Value) -> Option<i64> {
        if let Some(n) = value.as_i64() {
            return Some(n);
        }
        value.as_str().and_then(|s| s.parse().ok())
    }

    /// Parse all likes from like.js or likes.js.
    ///
    /// # Errors
    ///
    /// Returns an error if the likes file cannot be read or parsed.
    pub fn parse_likes(&self) -> Result<Vec<Like>> {
        info!("Parsing likes...");

        let mut files = Vec::new();
        let like_path = self.archive_path.join("data").join("like.js");
        if like_path.exists() {
            files.push(like_path);
        }
        let likes_path = self.archive_path.join("data").join("likes.js");
        if likes_path.exists() {
            files.push(likes_path);
        }

        if files.is_empty() {
            info!("No like files found.");
            return Ok(Vec::new());
        }

        let mut likes = Vec::new();
        let mut seen_ids: HashSet<String> = HashSet::new();

        for path in files {
            let content = std::fs::read_to_string(&path)
                .with_context(|| format!("Failed to read {}", path.display()))?;
            let data = self.parse_js_file(&content)?;
            let Some(items) = data.as_array() else {
                continue;
            };

            let file_likes: Vec<Like> = items
                .par_iter()
                .filter_map(|item| {
                    let like = &item["like"];
                    Some(Like {
                        tweet_id: like["tweetId"].as_str()?.to_string(),
                        full_text: like["fullText"].as_str().map(String::from),
                        expanded_url: like["expandedUrl"].as_str().map(String::from),
                    })
                })
                .collect();

            for like in file_likes {
                if seen_ids.insert(like.tweet_id.clone()) {
                    likes.push(like);
                }
            }
        }

        info!("Parsed {} likes", likes.len());
        Ok(likes)
    }

    /// Parse direct messages from direct-messages.js.
    ///
    /// # Errors
    ///
    /// Returns an error if the direct messages file cannot be read or parsed.
    pub fn parse_direct_messages(&self) -> Result<Vec<DmConversation>> {
        info!("Parsing direct messages...");

        let mut files = Vec::new();
        let dm_path = self.archive_path.join("data").join("direct-messages.js");
        if dm_path.exists() {
            files.push(dm_path);
        }
        files.extend(self.collect_data_files("direct-messages-group*.js")?);

        if files.is_empty() {
            info!("No direct message files found.");
            return Ok(Vec::new());
        }

        let mut conversations: HashMap<String, Vec<DirectMessage>> = HashMap::new();
        let mut seen_ids: HashSet<String> = HashSet::new();

        for path in files {
            let content = std::fs::read_to_string(&path)
                .with_context(|| format!("Failed to read {}", path.display()))?;
            let data = self.parse_js_file(&content)?;
            let Some(items) = data.as_array() else {
                continue;
            };

            for item in items {
                let conv = &item["dmConversation"];
                let Some(conversation_id) = conv["conversationId"].as_str().map(String::from)
                else {
                    continue;
                };

                if let Some(messages) = conv["messages"].as_array() {
                    for msg in messages {
                        let mc = &msg["messageCreate"];
                        let Some(id) = mc["id"].as_str().map(String::from) else {
                            continue;
                        };
                        if !seen_ids.insert(id.clone()) {
                            continue;
                        }

                        let Some(sender_id) = mc["senderId"].as_str().map(String::from) else {
                            continue;
                        };
                        let Some(recipient_id) = mc["recipientId"].as_str().map(String::from)
                        else {
                            continue;
                        };
                        let Some(text) = mc["text"].as_str().map(String::from) else {
                            continue;
                        };
                        let Some(created_at) =
                            mc["createdAt"].as_str().and_then(Self::parse_iso_date)
                        else {
                            continue;
                        };

                        let media_urls = mc["mediaUrls"]
                            .as_array()
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|u| u.as_str().map(String::from))
                                    .collect()
                            })
                            .unwrap_or_default();

                        let dm = DirectMessage {
                            id,
                            conversation_id: conversation_id.clone(),
                            sender_id,
                            recipient_id,
                            text,
                            created_at,
                            urls: Self::parse_dm_urls(&mc["urls"]),
                            media_urls,
                        };

                        conversations
                            .entry(conversation_id.clone())
                            .or_default()
                            .push(dm);
                    }
                }
            }
        }

        let mut output: Vec<DmConversation> = conversations
            .into_iter()
            .map(|(conversation_id, mut messages)| {
                messages.sort_by_key(|m| m.created_at);
                DmConversation {
                    conversation_id,
                    messages,
                }
            })
            .collect();

        output.sort_by(|a, b| a.conversation_id.cmp(&b.conversation_id));

        let total_messages: usize = output.iter().map(|c| c.messages.len()).sum();
        info!(
            "Parsed {} DM conversations with {} total messages",
            output.len(),
            total_messages
        );
        Ok(output)
    }

    fn parse_dm_urls(value: &Value) -> Vec<TweetUrl> {
        Self::as_array_or_empty(value)
            .iter()
            .filter_map(|u| {
                Some(TweetUrl {
                    url: u["url"].as_str()?.to_string(),
                    expanded_url: u["expanded"].as_str().map(String::from),
                    display_url: u["display"].as_str().map(String::from),
                })
            })
            .collect()
    }

    /// Parse followers from follower.js.
    ///
    /// # Errors
    ///
    /// Returns an error if the followers file cannot be read or parsed.
    pub fn parse_followers(&self) -> Result<Vec<Follower>> {
        info!("Parsing follower.js...");
        let data = self.read_data_file("follower.js")?;

        let followers: Vec<Follower> = Self::as_array_or_empty(&data)
            .par_iter()
            .filter_map(|item| {
                let f = &item["follower"];
                Some(Follower {
                    account_id: f["accountId"].as_str()?.to_string(),
                    user_link: f["userLink"].as_str().map(String::from),
                })
            })
            .collect();

        info!("Parsed {} followers", followers.len());
        Ok(followers)
    }

    /// Parse following from following.js.
    ///
    /// # Errors
    ///
    /// Returns an error if the following file cannot be read or parsed.
    pub fn parse_following(&self) -> Result<Vec<Following>> {
        info!("Parsing following.js...");
        let data = self.read_data_file("following.js")?;

        let following: Vec<Following> = Self::as_array_or_empty(&data)
            .par_iter()
            .filter_map(|item| {
                let f = &item["following"];
                Some(Following {
                    account_id: f["accountId"].as_str()?.to_string(),
                    user_link: f["userLink"].as_str().map(String::from),
                })
            })
            .collect();

        info!("Parsed {} following", following.len());
        Ok(following)
    }

    /// Parse blocks from block.js.
    ///
    /// # Errors
    ///
    /// Returns an error if the blocks file cannot be read or parsed.
    pub fn parse_blocks(&self) -> Result<Vec<Block>> {
        info!("Parsing block.js...");
        let data = self.read_data_file("block.js")?;

        let blocks: Vec<Block> = Self::as_array_or_empty(&data)
            .par_iter()
            .filter_map(|item| {
                let b = &item["blocking"];
                Some(Block {
                    account_id: b["accountId"].as_str()?.to_string(),
                    user_link: b["userLink"].as_str().map(String::from),
                })
            })
            .collect();

        info!("Parsed {} blocks", blocks.len());
        Ok(blocks)
    }

    /// Parse mutes from mute.js.
    ///
    /// # Errors
    ///
    /// Returns an error if the mutes file cannot be read or parsed.
    pub fn parse_mutes(&self) -> Result<Vec<Mute>> {
        info!("Parsing mute.js...");
        let data = self.read_data_file("mute.js")?;

        let mutes: Vec<Mute> = Self::as_array_or_empty(&data)
            .par_iter()
            .filter_map(|item| {
                let m = &item["muting"];
                Some(Mute {
                    account_id: m["accountId"].as_str()?.to_string(),
                    user_link: m["userLink"].as_str().map(String::from),
                })
            })
            .collect();

        info!("Parsed {} mutes", mutes.len());
        Ok(mutes)
    }

    /// Parse account info from account.js.
    ///
    /// # Errors
    ///
    /// Returns an error if the account file cannot be read or parsed.
    pub fn parse_account(&self) -> Result<Option<Account>> {
        info!("Parsing account.js...");
        let data = self.read_data_file("account.js")?;

        let account = data.as_array().and_then(|arr| arr.first()).map(|item| {
            let a = &item["account"];
            Account {
                account_id: a["accountId"].as_str().unwrap_or_default().to_string(),
                username: a["username"].as_str().unwrap_or_default().to_string(),
                display_name: a["accountDisplayName"].as_str().map(String::from),
                email: a["email"].as_str().map(String::from),
                created_at: a["createdAt"].as_str().and_then(Self::parse_iso_date),
                created_via: a["createdVia"].as_str().map(String::from),
            }
        });

        Ok(account)
    }

    /// Parse profile info from profile.js.
    ///
    /// # Errors
    ///
    /// Returns an error if the profile file cannot be read or parsed.
    pub fn parse_profile(&self) -> Result<Option<Profile>> {
        info!("Parsing profile.js...");
        let data = self.read_data_file("profile.js")?;

        let profile = data.as_array().and_then(|arr| arr.first()).map(|item| {
            let p = &item["profile"];
            let desc = &p["description"];
            Profile {
                bio: desc["bio"].as_str().map(String::from),
                website: desc["website"].as_str().map(String::from),
                location: desc["location"].as_str().map(String::from),
                avatar_url: p["avatarMediaUrl"].as_str().map(String::from),
                header_url: p["headerMediaUrl"].as_str().map(String::from),
            }
        });

        Ok(profile)
    }

    /// Parse Grok chat messages from grok-chat-item.js.
    ///
    /// # Errors
    ///
    /// Returns an error if the Grok messages file cannot be read or parsed.
    pub fn parse_grok_messages(&self) -> Result<Vec<GrokMessage>> {
        info!("Parsing grok-chat-item.js...");
        let data = self.read_data_file("grok-chat-item.js")?;

        let messages: Vec<GrokMessage> = Self::as_array_or_empty(&data)
            .par_iter()
            .filter_map(|item| {
                let g = &item["grokChatItem"];
                Some(GrokMessage {
                    chat_id: g["chatId"].as_str()?.to_string(),
                    message: g["message"].as_str()?.to_string(),
                    sender: g["sender"].as_str().unwrap_or("unknown").to_string(),
                    created_at: g["createdAt"].as_str().and_then(Self::parse_iso_date)?,
                    grok_mode: g["grokMode"].as_str().map(String::from),
                })
            })
            .collect();

        info!("Parsed {} Grok messages", messages.len());
        Ok(messages)
    }

    /// Parse bookmarks from bookmarks.js.
    ///
    /// Bookmarks only contain tweet IDs, so we cross-reference with the provided
    /// tweet lookup to get full text when the bookmarked tweet is in the user's archive.
    ///
    /// # Errors
    ///
    /// Returns an error if the bookmarks file cannot be read or parsed.
    pub fn parse_bookmarks(&self, tweet_texts: &HashMap<String, String>) -> Result<Vec<Bookmark>> {
        info!("Parsing bookmarks...");
        let data = self.read_data_file("bookmarks.js")?;

        let items = Self::as_array_or_empty(&data);
        if items.is_empty() {
            info!("No bookmarks found.");
            return Ok(Vec::new());
        }

        let mut bookmarks = Vec::new();
        let mut seen_ids: HashSet<String> = HashSet::new();

        for item in items {
            let bm = &item["bookmark"];
            let Some(tweet_id) = bm["tweetId"].as_str().map(String::from) else {
                continue;
            };
            if !seen_ids.insert(tweet_id.clone()) {
                continue;
            }
            let full_text = tweet_texts.get(&tweet_id).cloned();
            bookmarks.push(Bookmark {
                tweet_id,
                full_text,
            });
        }

        let with_text = bookmarks.iter().filter(|b| b.full_text.is_some()).count();
        info!(
            "Parsed {} bookmarks ({} with full text from tweets.js)",
            bookmarks.len(),
            with_text
        );
        Ok(bookmarks)
    }

    /// List all available data files in the archive.
    ///
    /// # Errors
    ///
    /// Returns an error if the archive directory cannot be read.
    pub fn list_data_files(&self) -> Result<Vec<String>> {
        let data_path = self.archive_path.join("data");
        let mut files = Vec::new();

        for entry in WalkDir::new(&data_path).max_depth(1) {
            let entry = entry?;
            if entry.file_type().is_file() {
                if let Some(name) = entry.file_name().to_str() {
                    if std::path::Path::new(name)
                        .extension()
                        .is_some_and(|ext| ext.eq_ignore_ascii_case("js"))
                    {
                        files.push(name.to_string());
                    }
                }
            }
        }

        files.sort();
        Ok(files)
    }

    fn collect_data_files(&self, pattern: &str) -> Result<Vec<std::path::PathBuf>> {
        let data_dir = self.archive_path.join("data");
        // Escape the directory path to safely handle characters like [, ], * in the path itself
        let escaped_dir = glob::Pattern::escape(data_dir.to_string_lossy().as_ref());
        let full_pattern = format!("{}{}{}", escaped_dir, std::path::MAIN_SEPARATOR, pattern);

        let mut paths: Vec<_> = glob(&full_pattern)
            .map_err(|e| anyhow::anyhow!("Invalid glob pattern: {e}"))?
            .filter_map(std::result::Result::ok)
            .collect();
        paths.sort();
        Ok(paths)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{Datelike, Timelike};
    use tempfile::TempDir;

    #[test]
    fn test_parse_x_date() {
        let date = ArchiveParser::parse_x_date("Fri Jan 09 15:12:21 +0000 2026");
        assert!(date.is_some());
        let dt = date.unwrap();
        assert_eq!(dt.year(), 2026);
        assert_eq!(dt.month(), 1);
        assert_eq!(dt.day(), 9);
        assert_eq!(dt.hour(), 15);
        assert_eq!(dt.minute(), 12);
        assert_eq!(dt.second(), 21);
    }

    #[test]
    fn test_parse_x_date_invalid() {
        let date = ArchiveParser::parse_x_date("invalid date string");
        assert!(date.is_none());
    }

    #[test]
    fn test_parse_x_date_empty() {
        let date = ArchiveParser::parse_x_date("");
        assert!(date.is_none());
    }

    #[test]
    fn test_parse_iso_date() {
        let date = ArchiveParser::parse_iso_date("2025-11-06T23:32:43.358Z");
        assert!(date.is_some());
        let dt = date.unwrap();
        assert_eq!(dt.year(), 2025);
        assert_eq!(dt.month(), 11);
        assert_eq!(dt.day(), 6);
    }

    #[test]
    fn test_parse_iso_date_invalid() {
        let date = ArchiveParser::parse_iso_date("not a date");
        assert!(date.is_none());
    }

    #[test]
    fn test_parse_js_file_valid() {
        let temp_dir = TempDir::new().unwrap();
        let parser = ArchiveParser::new(temp_dir.path());

        let content = r#"window.YTD.tweets.part0 = [{"tweet": {"id": "123"}}]"#;
        let result = parser.parse_js_file(content);
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(value.is_array());
        assert_eq!(value.as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_parse_js_file_empty_array() {
        let temp_dir = TempDir::new().unwrap();
        let parser = ArchiveParser::new(temp_dir.path());

        let content = r"window.YTD.likes.part0 = []";
        let result = parser.parse_js_file(content);
        assert!(result.is_ok());

        let value = result.unwrap();
        assert!(value.is_array());
        assert_eq!(value.as_array().unwrap().len(), 0);
    }

    #[test]
    fn test_parse_js_file_invalid_no_equals() {
        let temp_dir = TempDir::new().unwrap();
        let parser = ArchiveParser::new(temp_dir.path());

        let content = r"window.YTD.tweets.part0";
        let result = parser.parse_js_file(content);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_js_file_invalid_json() {
        let temp_dir = TempDir::new().unwrap();
        let parser = ArchiveParser::new(temp_dir.path());

        let content = r"window.YTD.tweets.part0 = {invalid json";
        let result = parser.parse_js_file(content);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_manifest_missing_file() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let parser = ArchiveParser::new(temp_dir.path());
        let result = parser.parse_manifest();
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_hashtags() {
        let json = serde_json::json!([
            {"text": "rust"},
            {"text": "programming"},
            {"text": "code"}
        ]);
        let hashtags = ArchiveParser::parse_hashtags(&json);
        assert_eq!(hashtags.len(), 3);
        assert!(hashtags.contains(&"rust".to_string()));
        assert!(hashtags.contains(&"programming".to_string()));
        assert!(hashtags.contains(&"code".to_string()));
    }

    #[test]
    fn test_parse_hashtags_empty() {
        let json = serde_json::json!([]);
        let hashtags = ArchiveParser::parse_hashtags(&json);
        assert!(hashtags.is_empty());
    }

    #[test]
    fn test_parse_hashtags_null() {
        let json = serde_json::json!(null);
        let hashtags = ArchiveParser::parse_hashtags(&json);
        assert!(hashtags.is_empty());
    }

    #[test]
    fn test_parse_user_mentions() {
        let json = serde_json::json!([
            {"id_str": "123", "screen_name": "alice", "name": "Alice Smith"},
            {"id_str": "456", "screen_name": "bob"}
        ]);
        let mentions = ArchiveParser::parse_user_mentions(&json);
        assert_eq!(mentions.len(), 2);
        assert_eq!(mentions[0].id, "123");
        assert_eq!(mentions[0].screen_name, "alice");
        assert_eq!(mentions[0].name, Some("Alice Smith".to_string()));
        assert_eq!(mentions[1].id, "456");
        assert_eq!(mentions[1].screen_name, "bob");
        assert_eq!(mentions[1].name, None);
    }

    #[test]
    fn test_parse_user_mentions_missing_required_fields() {
        let json = serde_json::json!([
            {"id_str": "123"},  // missing screen_name
            {"screen_name": "bob"}  // missing id_str
        ]);
        let mentions = ArchiveParser::parse_user_mentions(&json);
        assert!(mentions.is_empty());
    }

    #[test]
    fn test_parse_urls() {
        let json = serde_json::json!([
            {
                "url": "https://t.co/abc",
                "expanded_url": "https://example.com/page",
                "display_url": "example.com/page"
            },
            {
                "url": "https://t.co/xyz"
            }
        ]);
        let urls = ArchiveParser::parse_urls(&json);
        assert_eq!(urls.len(), 2);
        assert_eq!(urls[0].url, "https://t.co/abc");
        assert_eq!(
            urls[0].expanded_url,
            Some("https://example.com/page".to_string())
        );
        assert_eq!(urls[0].display_url, Some("example.com/page".to_string()));
        assert_eq!(urls[1].url, "https://t.co/xyz");
        assert_eq!(urls[1].expanded_url, None);
    }

    #[test]
    fn test_parse_media() {
        let json = serde_json::json!([
            {
                "id_str": "media123",
                "type": "photo",
                "media_url_https": "https://pbs.twimg.com/media/123.jpg"
            },
            {
                "id_str": "media456",
                "type": "video",
                "media_url": "https://pbs.twimg.com/media/456.mp4"
            }
        ]);
        let media = ArchiveParser::parse_media(&json);
        assert_eq!(media.len(), 2);
        assert_eq!(media[0].id, "media123");
        assert_eq!(media[0].media_type, "photo");
        assert_eq!(media[0].url, "https://pbs.twimg.com/media/123.jpg");
        assert_eq!(media[1].id, "media456");
        assert_eq!(media[1].media_type, "video");
        assert_eq!(media[1].url, "https://pbs.twimg.com/media/456.mp4");
    }

    #[test]
    fn test_parse_media_default_type() {
        let json = serde_json::json!([
            {
                "id_str": "media123",
                "media_url_https": "https://pbs.twimg.com/media/123.jpg"
            }
        ]);
        let media = ArchiveParser::parse_media(&json);
        assert_eq!(media.len(), 1);
        assert_eq!(media[0].media_type, "photo"); // default
    }

    #[test]
    fn test_parse_dm_urls() {
        let json = serde_json::json!([
            {
                "url": "https://t.co/test",
                "expanded": "https://example.com",
                "display": "example.com"
            }
        ]);
        let urls = ArchiveParser::parse_dm_urls(&json);
        assert_eq!(urls.len(), 1);
        assert_eq!(urls[0].url, "https://t.co/test");
        assert_eq!(
            urls[0].expanded_url,
            Some("https://example.com".to_string())
        );
        assert_eq!(urls[0].display_url, Some("example.com".to_string()));
    }

    #[test]
    fn test_archive_parser_new() {
        let parser = ArchiveParser::new("/some/path");
        assert_eq!(parser.archive_path, std::path::PathBuf::from("/some/path"));
    }

    #[test]
    fn test_read_data_file_nonexistent() {
        let temp_dir = TempDir::new().unwrap();
        let parser = ArchiveParser::new(temp_dir.path());

        // Reading a nonexistent file should return an empty array
        let result = parser.read_data_file("nonexistent.js");
        assert!(result.is_ok());
        let value = result.unwrap();
        assert!(value.is_array());
        assert!(value.as_array().unwrap().is_empty());
    }

    #[test]
    fn test_read_data_file_with_data() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let content = r#"window.YTD.test.part0 = [{"key": "value"}]"#;
        std::fs::write(data_dir.join("test.js"), content).unwrap();

        let parser = ArchiveParser::new(temp_dir.path());
        let result = parser.read_data_file("test.js");
        assert!(result.is_ok());
        let value = result.unwrap();
        assert!(value.is_array());
        assert_eq!(value.as_array().unwrap().len(), 1);
    }

    #[test]
    fn test_list_data_files() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        // Create some test files
        std::fs::write(data_dir.join("tweets.js"), "").unwrap();
        std::fs::write(data_dir.join("likes.js"), "").unwrap();
        std::fs::write(data_dir.join("other.txt"), "").unwrap(); // should be ignored

        let parser = ArchiveParser::new(temp_dir.path());
        let files = parser.list_data_files().unwrap();

        assert!(files.contains(&"tweets.js".to_string()));
        assert!(files.contains(&"likes.js".to_string()));
        assert!(!files.contains(&"other.txt".to_string()));
    }

    // =========================================================================
    // Full Parsing Tests (end-to-end)
    // =========================================================================

    #[test]
    fn test_parse_tweets_full() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let content = r#"window.YTD.tweets.part0 = [
            {
                "tweet": {
                    "id_str": "1234567890",
                    "created_at": "Fri Jan 10 12:00:00 +0000 2025",
                    "full_text": "Hello world! #test @mention",
                    "source": "<a href=\"https://x.com\">X Web App</a>",
                    "favorite_count": "42",
                    "retweet_count": "7",
                    "lang": "en",
                    "entities": {
                        "hashtags": [{"text": "test"}],
                        "user_mentions": [{"id_str": "999", "screen_name": "mention", "name": "User"}],
                        "urls": []
                    }
                }
            }
        ]"#;
        std::fs::write(data_dir.join("tweets.js"), content).unwrap();

        let parser = ArchiveParser::new(temp_dir.path());
        let tweets = parser.parse_tweets().unwrap();

        assert_eq!(tweets.len(), 1);
        assert_eq!(tweets[0].id, "1234567890");
        assert_eq!(tweets[0].full_text, "Hello world! #test @mention");
        assert_eq!(tweets[0].favorite_count, 42);
        assert_eq!(tweets[0].retweet_count, 7);
        assert_eq!(tweets[0].hashtags, vec!["test".to_string()]);
        assert_eq!(tweets[0].user_mentions.len(), 1);
        assert_eq!(tweets[0].user_mentions[0].screen_name, "mention");
    }

    #[test]
    fn test_parse_tweets_parts_combines_and_dedupes() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let part1 = r#"window.YTD.tweets.part0 = [
            {
                "tweet": {
                    "id_str": "t1",
                    "created_at": "Fri Jan 10 12:00:00 +0000 2025",
                    "full_text": "First part",
                    "source": "web",
                    "favorite_count": "1",
                    "retweet_count": "0",
                    "lang": "en",
                    "entities": {"hashtags": [], "user_mentions": [], "urls": []}
                }
            }
        ]"#;

        let part2 = r#"window.YTD.tweets.part1 = [
            {
                "tweet": {
                    "id_str": "t2",
                    "created_at": "Fri Jan 10 12:01:00 +0000 2025",
                    "full_text": "Second part",
                    "source": "web",
                    "favorite_count": "2",
                    "retweet_count": "0",
                    "lang": "en",
                    "entities": {"hashtags": [], "user_mentions": [], "urls": []}
                }
            },
            {
                "tweet": {
                    "id_str": "t1",
                    "created_at": "Fri Jan 10 12:00:00 +0000 2025",
                    "full_text": "Duplicate from part2",
                    "source": "web",
                    "favorite_count": "1",
                    "retweet_count": "0",
                    "lang": "en",
                    "entities": {"hashtags": [], "user_mentions": [], "urls": []}
                }
            }
        ]"#;

        std::fs::write(data_dir.join("tweets-part1.js"), part1).unwrap();
        std::fs::write(data_dir.join("tweets-part2.js"), part2).unwrap();

        let parser = ArchiveParser::new(temp_dir.path());
        let tweets = parser.parse_tweets().unwrap();

        let ids: std::collections::HashSet<_> = tweets.iter().map(|t| t.id.as_str()).collect();
        assert_eq!(tweets.len(), 2);
        assert!(ids.contains("t1"));
        assert!(ids.contains("t2"));
    }

    #[test]
    fn test_parse_tweets_with_retweet() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let content = r#"window.YTD.tweets.part0 = [
            {
                "tweet": {
                    "id_str": "111",
                    "created_at": "Fri Jan 10 12:00:00 +0000 2025",
                    "full_text": "RT @someone: Original tweet content",
                    "source": "web",
                    "favorite_count": "0",
                    "retweet_count": "0",
                    "retweeted": true,
                    "entities": {"hashtags": [], "user_mentions": [], "urls": []}
                }
            }
        ]"#;
        std::fs::write(data_dir.join("tweets.js"), content).unwrap();

        let parser = ArchiveParser::new(temp_dir.path());
        let tweets = parser.parse_tweets().unwrap();

        assert_eq!(tweets.len(), 1);
        assert!(tweets[0].is_retweet);
    }

    #[test]
    fn test_parse_tweets_reply() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let content = r#"window.YTD.tweets.part0 = [
            {
                "tweet": {
                    "id_str": "222",
                    "created_at": "Fri Jan 10 12:00:00 +0000 2025",
                    "full_text": "@user This is a reply",
                    "source": "web",
                    "favorite_count": "5",
                    "retweet_count": "1",
                    "in_reply_to_status_id_str": "111",
                    "in_reply_to_user_id_str": "999",
                    "in_reply_to_screen_name": "user",
                    "entities": {"hashtags": [], "user_mentions": [], "urls": []}
                }
            }
        ]"#;
        std::fs::write(data_dir.join("tweets.js"), content).unwrap();

        let parser = ArchiveParser::new(temp_dir.path());
        let tweets = parser.parse_tweets().unwrap();

        assert_eq!(tweets.len(), 1);
        assert_eq!(tweets[0].in_reply_to_status_id, Some("111".to_string()));
        assert_eq!(tweets[0].in_reply_to_user_id, Some("999".to_string()));
        assert_eq!(tweets[0].in_reply_to_screen_name, Some("user".to_string()));
    }

    #[test]
    fn test_parse_likes_full() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let content = r#"window.YTD.like.part0 = [
            {
                "like": {
                    "tweetId": "9876543210",
                    "fullText": "Great content!",
                    "expandedUrl": "https://x.com/user/status/9876543210"
                }
            },
            {
                "like": {
                    "tweetId": "9876543211"
                }
            }
        ]"#;
        std::fs::write(data_dir.join("like.js"), content).unwrap();

        let parser = ArchiveParser::new(temp_dir.path());
        let likes = parser.parse_likes().unwrap();

        assert_eq!(likes.len(), 2);
        assert_eq!(likes[0].tweet_id, "9876543210");
        assert_eq!(likes[0].full_text, Some("Great content!".to_string()));
        assert_eq!(
            likes[0].expanded_url,
            Some("https://x.com/user/status/9876543210".to_string())
        );
        assert_eq!(likes[1].tweet_id, "9876543211");
        assert_eq!(likes[1].full_text, None);
    }

    #[test]
    fn test_parse_likes_fallback_likes_js() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let content = r#"window.YTD.likes.part0 = [
            {
                "like": {
                    "tweetId": "111222333",
                    "fullText": "Likes file variant",
                    "expandedUrl": "https://example.com/likes"
                }
            }
        ]"#;
        std::fs::write(data_dir.join("likes.js"), content).unwrap();

        let parser = ArchiveParser::new(temp_dir.path());
        let likes = parser.parse_likes().unwrap();

        assert_eq!(likes.len(), 1);
        assert_eq!(likes[0].tweet_id, "111222333");
        assert_eq!(likes[0].full_text, Some("Likes file variant".to_string()));
        assert_eq!(
            likes[0].expanded_url,
            Some("https://example.com/likes".to_string())
        );
    }

    #[test]
    fn test_parse_direct_messages_full() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let content = r#"window.YTD.direct_messages.part0 = [
            {
                "dmConversation": {
                    "conversationId": "conv123",
                    "messages": [
                        {
                            "messageCreate": {
                                "id": "msg1",
                                "senderId": "user1",
                                "recipientId": "user2",
                                "text": "Hello!",
                                "createdAt": "2025-01-10T12:00:00.000Z",
                                "urls": [],
                                "mediaUrls": []
                            }
                        },
                        {
                            "messageCreate": {
                                "id": "msg2",
                                "senderId": "user2",
                                "recipientId": "user1",
                                "text": "Hi there!",
                                "createdAt": "2025-01-10T12:01:00.000Z",
                                "urls": [],
                                "mediaUrls": []
                            }
                        }
                    ]
                }
            }
        ]"#;
        std::fs::write(data_dir.join("direct-messages.js"), content).unwrap();

        let parser = ArchiveParser::new(temp_dir.path());
        let conversations = parser.parse_direct_messages().unwrap();

        assert_eq!(conversations.len(), 1);
        assert_eq!(conversations[0].conversation_id, "conv123");
        assert_eq!(conversations[0].messages.len(), 2);
        assert_eq!(conversations[0].messages[0].id, "msg1");
        assert_eq!(conversations[0].messages[0].text, "Hello!");
        assert_eq!(conversations[0].messages[1].id, "msg2");
        assert_eq!(conversations[0].messages[1].text, "Hi there!");
    }

    #[test]
    fn test_parse_direct_messages_group_parts() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let part1 = r#"window.YTD.direct_messages.part0 = [
            {
                "dmConversation": {
                    "conversationId": "convA",
                    "messages": [
                        {
                            "messageCreate": {
                                "id": "msg1",
                                "senderId": "user1",
                                "recipientId": "user2",
                                "text": "First",
                                "createdAt": "2025-01-10T12:00:00.000Z",
                                "urls": [],
                                "mediaUrls": []
                            }
                        }
                    ]
                }
            }
        ]"#;

        let part2 = r#"window.YTD.direct_messages.part1 = [
            {
                "dmConversation": {
                    "conversationId": "convA",
                    "messages": [
                        {
                            "messageCreate": {
                                "id": "msg2",
                                "senderId": "user2",
                                "recipientId": "user1",
                                "text": "Second",
                                "createdAt": "2025-01-10T12:01:00.000Z",
                                "urls": [],
                                "mediaUrls": []
                            }
                        }
                    ]
                }
            }
        ]"#;

        std::fs::write(data_dir.join("direct-messages-group1.js"), part1).unwrap();
        std::fs::write(data_dir.join("direct-messages-group2.js"), part2).unwrap();

        let parser = ArchiveParser::new(temp_dir.path());
        let conversations = parser.parse_direct_messages().unwrap();

        assert_eq!(conversations.len(), 1);
        assert_eq!(conversations[0].conversation_id, "convA");
        assert_eq!(conversations[0].messages.len(), 2);
        assert_eq!(conversations[0].messages[0].id, "msg1");
        assert_eq!(conversations[0].messages[1].id, "msg2");
    }

    #[test]
    fn test_parse_grok_messages_full() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        // Note: The actual file is grok-chat-item.js with grokChatItem key
        let content = r#"window.YTD.grok_chat_item.part0 = [
            {
                "grokChatItem": {
                    "chatId": "chat123",
                    "message": "What is Rust?",
                    "sender": "user",
                    "createdAt": "2025-01-10T12:00:00.000Z",
                    "grokMode": "regular"
                }
            },
            {
                "grokChatItem": {
                    "chatId": "chat123",
                    "message": "Rust is a programming language...",
                    "sender": "grok",
                    "createdAt": "2025-01-10T12:00:01.000Z",
                    "grokMode": "regular"
                }
            }
        ]"#;
        std::fs::write(data_dir.join("grok-chat-item.js"), content).unwrap();

        let parser = ArchiveParser::new(temp_dir.path());
        let messages = parser.parse_grok_messages().unwrap();

        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].chat_id, "chat123");
        assert_eq!(messages[0].message, "What is Rust?");
        assert_eq!(messages[0].sender, "user");
        assert_eq!(messages[1].sender, "grok");
    }

    #[test]
    fn test_parse_followers_full() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let content = r#"window.YTD.follower.part0 = [
            {"follower": {"accountId": "111", "userLink": "https://x.com/user111"}},
            {"follower": {"accountId": "222"}}
        ]"#;
        std::fs::write(data_dir.join("follower.js"), content).unwrap();

        let parser = ArchiveParser::new(temp_dir.path());
        let followers = parser.parse_followers().unwrap();

        assert_eq!(followers.len(), 2);
        assert_eq!(followers[0].account_id, "111");
        assert_eq!(
            followers[0].user_link,
            Some("https://x.com/user111".to_string())
        );
        assert_eq!(followers[1].account_id, "222");
        assert_eq!(followers[1].user_link, None);
    }

    #[test]
    fn test_parse_following_full() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let content = r#"window.YTD.following.part0 = [
            {"following": {"accountId": "333", "userLink": "https://x.com/user333"}}
        ]"#;
        std::fs::write(data_dir.join("following.js"), content).unwrap();

        let parser = ArchiveParser::new(temp_dir.path());
        let following = parser.parse_following().unwrap();

        assert_eq!(following.len(), 1);
        assert_eq!(following[0].account_id, "333");
    }

    #[test]
    fn test_parse_blocks_full() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let content = r#"window.YTD.block.part0 = [
            {"blocking": {"accountId": "444", "userLink": "https://x.com/blocked"}}
        ]"#;
        std::fs::write(data_dir.join("block.js"), content).unwrap();

        let parser = ArchiveParser::new(temp_dir.path());
        let blocks = parser.parse_blocks().unwrap();

        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].account_id, "444");
    }

    #[test]
    fn test_parse_mutes_full() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let content = r#"window.YTD.mute.part0 = [
            {"muting": {"accountId": "555", "userLink": "https://x.com/muted"}}
        ]"#;
        std::fs::write(data_dir.join("mute.js"), content).unwrap();

        let parser = ArchiveParser::new(temp_dir.path());
        let mutes = parser.parse_mutes().unwrap();

        assert_eq!(mutes.len(), 1);
        assert_eq!(mutes[0].account_id, "555");
    }

    // =========================================================================
    // Edge Case Tests
    // =========================================================================

    #[test]
    fn test_parse_tweet_with_media() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let content = r#"window.YTD.tweets.part0 = [
            {
                "tweet": {
                    "id_str": "333",
                    "created_at": "Fri Jan 10 12:00:00 +0000 2025",
                    "full_text": "Check out this image!",
                    "source": "web",
                    "favorite_count": "10",
                    "retweet_count": "2",
                    "entities": {
                        "hashtags": [],
                        "user_mentions": [],
                        "urls": [],
                        "media": [
                            {
                                "id_str": "media111",
                                "type": "photo",
                                "media_url_https": "https://pbs.twimg.com/media/test.jpg"
                            }
                        ]
                    }
                }
            }
        ]"#;
        std::fs::write(data_dir.join("tweets.js"), content).unwrap();

        let parser = ArchiveParser::new(temp_dir.path());
        let tweets = parser.parse_tweets().unwrap();

        assert_eq!(tweets.len(), 1);
        assert_eq!(tweets[0].media.len(), 1);
        assert_eq!(tweets[0].media[0].id, "media111");
        assert_eq!(tweets[0].media[0].media_type, "photo");
    }

    #[test]
    fn test_parse_tweet_with_urls() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let content = r#"window.YTD.tweets.part0 = [
            {
                "tweet": {
                    "id_str": "444",
                    "created_at": "Fri Jan 10 12:00:00 +0000 2025",
                    "full_text": "Check this out: https://t.co/abc",
                    "source": "web",
                    "favorite_count": "5",
                    "retweet_count": "1",
                    "entities": {
                        "hashtags": [],
                        "user_mentions": [],
                        "urls": [
                            {
                                "url": "https://t.co/abc",
                                "expanded_url": "https://example.com/article",
                                "display_url": "example.com/article"
                            }
                        ]
                    }
                }
            }
        ]"#;
        std::fs::write(data_dir.join("tweets.js"), content).unwrap();

        let parser = ArchiveParser::new(temp_dir.path());
        let tweets = parser.parse_tweets().unwrap();

        assert_eq!(tweets.len(), 1);
        assert_eq!(tweets[0].urls.len(), 1);
        assert_eq!(tweets[0].urls[0].url, "https://t.co/abc");
        assert_eq!(
            tweets[0].urls[0].expanded_url,
            Some("https://example.com/article".to_string())
        );
    }

    #[test]
    fn test_parse_empty_tweet_entities() {
        let temp_dir = TempDir::new().unwrap();
        let data_dir = temp_dir.path().join("data");
        std::fs::create_dir_all(&data_dir).unwrap();

        let content = r#"window.YTD.tweets.part0 = [
            {
                "tweet": {
                    "id_str": "555",
                    "created_at": "Fri Jan 10 12:00:00 +0000 2025",
                    "full_text": "Simple tweet without entities",
                    "source": "web",
                    "favorite_count": "0",
                    "retweet_count": "0"
                }
            }
        ]"#;
        std::fs::write(data_dir.join("tweets.js"), content).unwrap();

        let parser = ArchiveParser::new(temp_dir.path());
        let tweets = parser.parse_tweets().unwrap();

        assert_eq!(tweets.len(), 1);
        assert!(tweets[0].hashtags.is_empty());
        assert!(tweets[0].user_mentions.is_empty());
        assert!(tweets[0].urls.is_empty());
        assert!(tweets[0].media.is_empty());
    }
}
