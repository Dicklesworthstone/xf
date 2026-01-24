//! Enhanced statistics and analytics for archive data.
//!
//! This module provides advanced analytics beyond basic counts, including:
//! - Temporal analysis (activity patterns over time)
//! - Engagement metrics (likes, retweets distribution)
//! - Content analysis (media ratios, hashtags, mentions)

use crate::storage::Storage;
use crate::{Result, format_number_u64};
use chrono::{DateTime, Datelike, NaiveDate, Utc};
use serde::Serialize;

/// Temporal statistics showing activity patterns over time.
#[derive(Debug, Clone, Serialize)]
pub struct TemporalStats {
    /// Tweets per day for the entire archive period
    pub daily_counts: Vec<DailyCount>,
    /// Tweets per hour of day (0-23), aggregated across all days
    pub hourly_distribution: [u64; 24],
    /// Tweets per day of week (0=Sunday, 6=Saturday)
    pub dow_distribution: [u64; 7],
    /// Longest gap between tweets
    pub longest_gap_days: i64,
    /// Start date of the longest gap
    pub longest_gap_start: Option<NaiveDate>,
    /// End date of the longest gap
    pub longest_gap_end: Option<NaiveDate>,
    /// Day with most tweets
    pub most_active_day: Option<NaiveDate>,
    /// Tweet count on most active day
    pub most_active_day_count: u64,
    /// Most active hour (0-23)
    pub most_active_hour: u8,
    /// Tweet count for most active hour
    pub most_active_hour_count: u64,
    /// Average tweets per day (on days with activity)
    pub avg_tweets_per_active_day: f64,
    /// Total days with at least one tweet
    pub active_days_count: u64,
    /// Total days in archive range
    pub total_days_in_range: u64,
}

/// A single day's tweet count.
#[derive(Debug, Clone, Serialize)]
pub struct DailyCount {
    pub date: NaiveDate,
    pub count: u64,
}

impl TemporalStats {
    /// Compute temporal statistics from the storage.
    ///
    /// Uses SQL aggregations for efficiency on large datasets.
    ///
    /// # Errors
    ///
    /// Returns an error if database queries fail.
    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
    pub fn compute(storage: &Storage) -> Result<Self> {
        // Get daily counts using SQL
        let daily_counts = Self::query_daily_counts(storage)?;

        // Get hourly distribution
        let hourly_distribution = Self::query_hourly_distribution(storage)?;

        // Compute day-of-week distribution from daily_counts (avoids extra query)
        let dow_distribution = Self::compute_dow_from_daily(&daily_counts);

        // Compute derived metrics
        let (longest_gap_days, longest_gap_start, longest_gap_end) =
            Self::find_longest_gap(&daily_counts);

        let (most_active_day, most_active_day_count) = daily_counts
            .iter()
            .max_by_key(|d| d.count)
            .map_or((None, 0), |d| (Some(d.date), d.count));

        #[allow(clippy::cast_possible_truncation)]
        let (most_active_hour, most_active_hour_count) = hourly_distribution
            .iter()
            .enumerate()
            .max_by_key(|(_, count)| *count)
            .map_or((0, 0), |(hour, count)| (hour as u8, *count));

        let active_days_count = daily_counts.len() as u64;
        let total_tweets: u64 = daily_counts.iter().map(|d| d.count).sum();
        let avg_tweets_per_active_day = if active_days_count > 0 {
            total_tweets as f64 / active_days_count as f64
        } else {
            0.0
        };

        let total_days_in_range =
            if let (Some(first), Some(last)) = (daily_counts.first(), daily_counts.last()) {
                (last.date - first.date).num_days() as u64 + 1
            } else {
                0
            };

        Ok(Self {
            daily_counts,
            hourly_distribution,
            dow_distribution,
            longest_gap_days,
            longest_gap_start,
            longest_gap_end,
            most_active_day,
            most_active_day_count,
            most_active_hour,
            most_active_hour_count,
            avg_tweets_per_active_day,
            active_days_count,
            total_days_in_range,
        })
    }

    /// Query daily tweet counts from the database.
    #[allow(clippy::cast_sign_loss)]
    fn query_daily_counts(storage: &Storage) -> Result<Vec<DailyCount>> {
        let query = r"
            SELECT DATE(created_at) as day, COUNT(*) as count
            FROM tweets
            WHERE created_at IS NOT NULL
            GROUP BY day
            ORDER BY day
        ";

        let conn = storage.connection();
        let mut stmt = conn.prepare(query)?;
        let rows = stmt.query_map([], |row| {
            let day_str: Option<String> = row.get(0)?;
            let count: i64 = row.get(1)?;
            Ok((day_str, count as u64))
        })?;

        let mut counts = Vec::new();
        for row in rows {
            let (day_str, count) = row?;
            if let Some(s) = day_str {
                if let Ok(date) = NaiveDate::parse_from_str(&s, "%Y-%m-%d") {
                    counts.push(DailyCount { date, count });
                }
            }
        }

        Ok(counts)
    }

    /// Query hourly distribution (tweets per hour of day).
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    fn query_hourly_distribution(storage: &Storage) -> Result<[u64; 24]> {
        let query = r"
            SELECT CAST(strftime('%H', created_at) AS INTEGER) as hour, COUNT(*) as count
            FROM tweets
            WHERE created_at IS NOT NULL
            GROUP BY hour
            ORDER BY hour
        ";

        let conn = storage.connection();
        let mut stmt = conn.prepare(query)?;
        let rows = stmt.query_map([], |row| {
            let hour: Option<i64> = row.get(0)?;
            let count: i64 = row.get(1)?;
            Ok((hour, count as u64))
        })?;

        let mut distribution = [0u64; 24];
        for row in rows {
            let (hour, count) = row?;
            if let Some(h) = hour {
                if (0..24).contains(&h) {
                    distribution[h as usize] = count;
                }
            }
        }

        Ok(distribution)
    }

    /// Compute day-of-week distribution from daily counts.
    ///
    /// This avoids an extra database query by computing DOW from the dates
    /// we already have in `daily_counts`.
    fn compute_dow_from_daily(daily_counts: &[DailyCount]) -> [u64; 7] {
        let mut distribution = [0u64; 7];
        for dc in daily_counts {
            // chrono weekday: Mon=0, Tue=1, ..., Sun=6
            // We want: Sun=0, Mon=1, ..., Sat=6 (to match SQLite strftime('%w'))
            let dow = dc.date.weekday().num_days_from_sunday() as usize;
            distribution[dow] += dc.count;
        }
        distribution
    }

    /// Find the longest gap between consecutive days with tweets.
    fn find_longest_gap(
        daily_counts: &[DailyCount],
    ) -> (i64, Option<NaiveDate>, Option<NaiveDate>) {
        if daily_counts.len() < 2 {
            return (0, None, None);
        }

        let mut max_gap = 0i64;
        let mut gap_start: Option<NaiveDate> = None;
        let mut gap_end: Option<NaiveDate> = None;

        for window in daily_counts.windows(2) {
            let gap = (window[1].date - window[0].date).num_days();
            if gap > max_gap {
                max_gap = gap;
                gap_start = Some(window[0].date);
                gap_end = Some(window[1].date);
            }
        }

        (max_gap, gap_start, gap_end)
    }
}

// ============================================================================
// Engagement Analytics
// ============================================================================

/// Engagement metrics for the archive showing how tweets performed.
#[derive(Debug, Clone, Serialize)]
pub struct EngagementStats {
    /// Distribution of likes across tweets
    pub likes_histogram: Vec<LikesBucket>,
    /// Top N tweets by total engagement (likes + retweets)
    pub top_tweets: Vec<TopTweet>,
    /// Average engagement per tweet
    pub avg_engagement: f64,
    /// Median engagement
    pub median_engagement: u64,
    /// Total likes received across all tweets
    pub total_likes: u64,
    /// Total retweets received
    pub total_retweets: u64,
    /// Engagement trend over time (monthly averages)
    pub monthly_trend: Vec<MonthlyEngagement>,
}

/// A bucket in the likes histogram.
#[derive(Debug, Clone, Serialize)]
pub struct LikesBucket {
    /// Label for this bucket (e.g., "0", "1-5", "6-10")
    pub label: String,
    /// Minimum value in range (inclusive)
    pub min: u64,
    /// Maximum value in range (inclusive)
    pub max: u64,
    /// Number of tweets in this bucket
    pub count: u64,
    /// Percentage of total tweets
    pub percentage: f64,
}

/// A top-performing tweet by engagement.
#[derive(Debug, Clone, Serialize)]
pub struct TopTweet {
    /// Tweet ID
    pub id: String,
    /// First 50 characters of tweet text
    pub text_preview: String,
    /// When the tweet was created
    pub created_at: DateTime<Utc>,
    /// Number of likes
    pub likes: u64,
    /// Number of retweets
    pub retweets: u64,
    /// Total engagement (likes + retweets)
    pub total_engagement: u64,
}

/// Monthly engagement average.
#[derive(Debug, Clone, Serialize)]
pub struct MonthlyEngagement {
    /// Month in YYYY-MM format
    pub month: String,
    /// Average engagement for this month
    pub avg_engagement: f64,
}

impl EngagementStats {
    /// Compute engagement statistics from the storage.
    ///
    /// # Errors
    ///
    /// Returns an error if database queries fail.
    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
    pub fn compute(storage: &Storage, top_n: usize) -> Result<Self> {
        let likes_histogram = Self::query_likes_histogram(storage)?;
        let top_tweets = Self::query_top_tweets(storage, top_n)?;
        let (total_likes, total_retweets, avg_engagement, median_engagement) =
            Self::query_engagement_totals(storage)?;
        let monthly_trend = Self::query_monthly_trend(storage)?;

        Ok(Self {
            likes_histogram,
            top_tweets,
            avg_engagement,
            median_engagement,
            total_likes,
            total_retweets,
            monthly_trend,
        })
    }

    /// Query likes histogram with predefined buckets.
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation
    )]
    fn query_likes_histogram(storage: &Storage) -> Result<Vec<LikesBucket>> {
        // Get total tweet count first
        let total_query = "SELECT COUNT(*) FROM tweets";
        let conn = storage.connection();
        let total_count: i64 = conn.query_row(total_query, [], |row| row.get(0))?;
        let total_count = total_count as u64;

        // Define buckets with SQL CASE logic
        let query = r"
            SELECT
                CASE
                    WHEN favorite_count = 0 THEN 0
                    WHEN favorite_count BETWEEN 1 AND 5 THEN 1
                    WHEN favorite_count BETWEEN 6 AND 10 THEN 2
                    WHEN favorite_count BETWEEN 11 AND 25 THEN 3
                    WHEN favorite_count BETWEEN 26 AND 50 THEN 4
                    WHEN favorite_count BETWEEN 51 AND 100 THEN 5
                    WHEN favorite_count BETWEEN 101 AND 500 THEN 6
                    ELSE 7
                END as bucket,
                COUNT(*) as count
            FROM tweets
            WHERE favorite_count IS NOT NULL
            GROUP BY bucket
            ORDER BY bucket
        ";

        let mut stmt = conn.prepare(query)?;
        let rows = stmt.query_map([], |row| {
            let bucket: i64 = row.get(0)?;
            let count: i64 = row.get(1)?;
            Ok((bucket as usize, count as u64))
        })?;

        // Define bucket ranges and labels
        let bucket_defs = [
            ("0", 0u64, 0u64),
            ("1-5", 1, 5),
            ("6-10", 6, 10),
            ("11-25", 11, 25),
            ("26-50", 26, 50),
            ("51-100", 51, 100),
            ("101-500", 101, 500),
            ("500+", 501, u64::MAX),
        ];

        let mut buckets: Vec<LikesBucket> = bucket_defs
            .iter()
            .map(|(label, min, max)| LikesBucket {
                label: (*label).to_string(),
                min: *min,
                max: *max,
                count: 0,
                percentage: 0.0,
            })
            .collect();

        for row in rows {
            let (bucket_idx, count) = row?;
            if bucket_idx < buckets.len() {
                buckets[bucket_idx].count = count;
                buckets[bucket_idx].percentage = if total_count > 0 {
                    (count as f64 / total_count as f64) * 100.0
                } else {
                    0.0
                };
            }
        }

        Ok(buckets)
    }

    /// Query top N tweets by total engagement.
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_wrap)]
    fn query_top_tweets(storage: &Storage, limit: usize) -> Result<Vec<TopTweet>> {
        let query = r"
            SELECT id, full_text, created_at, favorite_count, retweet_count,
                   (COALESCE(favorite_count, 0) + COALESCE(retweet_count, 0)) as total_engagement
            FROM tweets
            WHERE favorite_count IS NOT NULL OR retweet_count IS NOT NULL
            ORDER BY total_engagement DESC
            LIMIT ?
        ";

        let conn = storage.connection();
        let mut stmt = conn.prepare(query)?;
        let limit_i64 = i64::try_from(limit).unwrap_or(i64::MAX);
        let rows = stmt.query_map([limit_i64], |row| {
            let id: String = row.get(0)?;
            let full_text: String = row.get(1)?;
            let created_at_str: String = row.get(2)?;
            let likes: i64 = row.get::<_, Option<i64>>(3)?.unwrap_or(0);
            let retweets: i64 = row.get::<_, Option<i64>>(4)?.unwrap_or(0);
            let total: i64 = row.get(5)?;
            Ok((id, full_text, created_at_str, likes, retweets, total))
        })?;

        let mut top_tweets = Vec::new();
        for row in rows {
            let (id, full_text, created_at_str, likes, retweets, total) = row?;

            // Parse date - try ISO format first, then X format
            let created_at = DateTime::parse_from_rfc3339(&created_at_str)
                .map(|dt| dt.with_timezone(&Utc))
                .or_else(|_| {
                    DateTime::parse_from_str(&created_at_str, "%a %b %d %H:%M:%S %z %Y")
                        .map(|dt| dt.with_timezone(&Utc))
                })
                .unwrap_or_else(|_| DateTime::<Utc>::from_timestamp(0, 0).unwrap());

            // Truncate text to ~50 chars at word boundary
            let text_preview = truncate_text(&full_text, 50);

            top_tweets.push(TopTweet {
                id,
                text_preview,
                created_at,
                likes: likes as u64,
                retweets: retweets as u64,
                total_engagement: total as u64,
            });
        }

        Ok(top_tweets)
    }

    /// Query total engagement metrics.
    ///
    /// Computes sum, average, and median engagement in two queries, reusing
    /// the count from the first query to avoid a subquery scan in the median.
    #[allow(clippy::cast_sign_loss, clippy::cast_precision_loss)]
    fn query_engagement_totals(storage: &Storage) -> Result<(u64, u64, f64, u64)> {
        let query = r"
            SELECT
                COALESCE(SUM(favorite_count), 0) as total_likes,
                COALESCE(SUM(retweet_count), 0) as total_retweets,
                COALESCE(AVG(favorite_count + retweet_count), 0) as avg_engagement,
                COUNT(*) as tweet_count
            FROM tweets
        ";

        let conn = storage.connection();
        let (total_likes, total_retweets, avg_engagement, tweet_count): (i64, i64, f64, i64) = conn
            .query_row(query, [], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?;

        // Query median using pre-computed offset (avoids subquery scan)
        let median_offset = tweet_count / 2;
        let median_query = r"
            SELECT favorite_count + retweet_count as engagement
            FROM tweets
            WHERE favorite_count IS NOT NULL
            ORDER BY engagement
            LIMIT 1 OFFSET ?
        ";

        let median: i64 = conn
            .query_row(median_query, [median_offset], |row| row.get(0))
            .unwrap_or(0);

        Ok((
            total_likes as u64,
            total_retweets as u64,
            avg_engagement,
            median as u64,
        ))
    }

    /// Query monthly engagement trend.
    #[allow(clippy::cast_sign_loss)]
    fn query_monthly_trend(storage: &Storage) -> Result<Vec<MonthlyEngagement>> {
        let query = r"
            SELECT strftime('%Y-%m', created_at) as month,
                   AVG(COALESCE(favorite_count, 0) + COALESCE(retweet_count, 0)) as avg_engagement
            FROM tweets
            WHERE created_at IS NOT NULL
            GROUP BY month
            ORDER BY month
        ";

        let conn = storage.connection();
        let mut stmt = conn.prepare(query)?;
        let rows = stmt.query_map([], |row| {
            let month: Option<String> = row.get(0)?;
            let avg: f64 = row.get(1)?;
            Ok((month, avg))
        })?;

        let mut trend = Vec::new();
        for row in rows {
            let (month, avg) = row?;
            if let Some(m) = month {
                trend.push(MonthlyEngagement {
                    month: m,
                    avg_engagement: avg,
                });
            }
        }

        Ok(trend)
    }
}

// ============================================================================
// Content Analysis
// ============================================================================

/// Content breakdown and interaction patterns.
#[derive(Debug, Clone, Serialize)]
pub struct ContentStats {
    /// Percentage of tweets with media attachments
    pub media_ratio: f64,
    /// Percentage of tweets with links
    pub link_ratio: f64,
    /// Percentage of tweets that are replies
    pub reply_ratio: f64,
    /// Number of tweets that are part of self-threads
    pub thread_count: u64,
    /// Number of standalone tweets (non-reply, non-thread)
    pub standalone_count: u64,
    /// Total tweet count
    pub total_count: u64,
    /// Average tweet length in characters
    pub avg_tweet_length: f64,
    /// Distribution of tweet lengths by bucket
    pub length_distribution: Vec<LengthBucket>,
    /// Top hashtags with counts
    pub top_hashtags: Vec<TagCount>,
    /// Top mentioned users with counts
    pub top_mentions: Vec<TagCount>,
}

/// A length distribution bucket.
#[derive(Debug, Clone, Serialize)]
pub struct LengthBucket {
    /// Label for this bucket (e.g., "0-50")
    pub label: String,
    /// Number of tweets in this bucket
    pub count: u64,
    /// Percentage of total tweets
    pub percentage: f64,
}

/// A hashtag or mention with its count.
#[derive(Debug, Clone, Serialize)]
pub struct TagCount {
    /// The tag or username
    pub tag: String,
    /// Number of occurrences
    pub count: u64,
}

impl ContentStats {
    /// Compute content statistics from the storage.
    ///
    /// # Errors
    ///
    /// Returns an error if database queries fail.
    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
    pub fn compute(storage: &Storage, top_n: usize) -> Result<Self> {
        let (total_count, media_count, link_count, reply_count, thread_count, standalone_count) =
            Self::query_content_counts(storage)?;

        let media_ratio = if total_count > 0 {
            (media_count as f64 / total_count as f64) * 100.0
        } else {
            0.0
        };

        let link_ratio = if total_count > 0 {
            (link_count as f64 / total_count as f64) * 100.0
        } else {
            0.0
        };

        let reply_ratio = if total_count > 0 {
            (reply_count as f64 / total_count as f64) * 100.0
        } else {
            0.0
        };

        let avg_tweet_length = Self::query_avg_length(storage)?;
        let length_distribution = Self::query_length_distribution(storage)?;
        let top_hashtags = Self::query_top_hashtags(storage, top_n)?;
        let top_mentions = Self::query_top_mentions(storage, top_n)?;

        Ok(Self {
            media_ratio,
            link_ratio,
            reply_ratio,
            thread_count,
            standalone_count,
            total_count,
            avg_tweet_length,
            length_distribution,
            top_hashtags,
            top_mentions,
        })
    }

    /// Query content type counts.
    #[allow(clippy::cast_sign_loss)]
    fn query_content_counts(storage: &Storage) -> Result<(u64, u64, u64, u64, u64, u64)> {
        let conn = storage.connection();

        // Total tweets
        let total: i64 = conn.query_row("SELECT COUNT(*) FROM tweets", [], |row| row.get(0))?;

        // Tweets with media
        let media: i64 = conn.query_row(
            "SELECT COUNT(*) FROM tweets WHERE media_json IS NOT NULL AND media_json != '[]' AND media_json != ''",
            [],
            |row| row.get(0),
        )?;

        // Tweets with URLs
        let links: i64 = conn.query_row(
            "SELECT COUNT(*) FROM tweets WHERE urls_json IS NOT NULL AND urls_json != '[]' AND urls_json != ''",
            [],
            |row| row.get(0),
        )?;

        // Replies (has in_reply_to_status_id)
        let replies: i64 = conn.query_row(
            "SELECT COUNT(*) FROM tweets WHERE in_reply_to_status_id IS NOT NULL AND in_reply_to_status_id != ''",
            [],
            |row| row.get(0),
        )?;

        // Self-threads: replies where in_reply_to_user_id matches our user
        // We need to get our user_id from archive_info
        let threads: i64 = conn
            .query_row(
                r"
                SELECT COUNT(*) FROM tweets t
                WHERE t.in_reply_to_status_id IS NOT NULL
                  AND t.in_reply_to_user_id = (SELECT account_id FROM archive_info LIMIT 1)
            ",
                [],
                |row| row.get(0),
            )
            .unwrap_or(0);

        let standalone = total - replies;

        Ok((
            total as u64,
            media as u64,
            links as u64,
            replies as u64,
            threads as u64,
            standalone as u64,
        ))
    }

    /// Query average tweet length.
    fn query_avg_length(storage: &Storage) -> Result<f64> {
        let conn = storage.connection();
        // Use COALESCE to handle empty tables where AVG returns NULL
        let avg: f64 = conn.query_row(
            "SELECT COALESCE(AVG(LENGTH(full_text)), 0) FROM tweets",
            [],
            |row| row.get(0),
        )?;
        Ok(avg)
    }

    /// Query tweet length distribution.
    #[allow(
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation
    )]
    fn query_length_distribution(storage: &Storage) -> Result<Vec<LengthBucket>> {
        let conn = storage.connection();

        // Get total for percentages
        let total: i64 = conn.query_row("SELECT COUNT(*) FROM tweets", [], |row| row.get(0))?;
        let total = total as u64;

        let query = r"
            SELECT
                CASE
                    WHEN LENGTH(full_text) <= 50 THEN 0
                    WHEN LENGTH(full_text) <= 140 THEN 1
                    WHEN LENGTH(full_text) <= 280 THEN 2
                    ELSE 3
                END as bucket,
                COUNT(*) as count
            FROM tweets
            GROUP BY bucket
            ORDER BY bucket
        ";

        let mut stmt = conn.prepare(query)?;
        let rows = stmt.query_map([], |row| {
            let bucket: i64 = row.get(0)?;
            let count: i64 = row.get(1)?;
            Ok((bucket as usize, count as u64))
        })?;

        let bucket_labels = ["0-50", "51-140", "141-280", "280+"];
        let mut buckets: Vec<LengthBucket> = bucket_labels
            .iter()
            .map(|label| LengthBucket {
                label: (*label).to_string(),
                count: 0,
                percentage: 0.0,
            })
            .collect();

        for row in rows {
            let (bucket_idx, count) = row?;
            if bucket_idx < buckets.len() {
                buckets[bucket_idx].count = count;
                buckets[bucket_idx].percentage = if total > 0 {
                    (count as f64 / total as f64) * 100.0
                } else {
                    0.0
                };
            }
        }

        Ok(buckets)
    }

    /// Query top hashtags from the `hashtags_json` column.
    #[allow(clippy::cast_sign_loss)]
    fn query_top_hashtags(storage: &Storage, limit: usize) -> Result<Vec<TagCount>> {
        let conn = storage.connection();

        // The hashtags are stored as JSON array in hashtags_json column
        // We need to parse them and count
        let query = "SELECT hashtags_json FROM tweets WHERE hashtags_json IS NOT NULL AND hashtags_json != '[]' AND hashtags_json != ''";
        let mut stmt = conn.prepare(query)?;

        let mut counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();

        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;

        for json_str in rows.flatten() {
            // Parse JSON array of hashtags
            if let Ok(tags) = serde_json::from_str::<Vec<String>>(&json_str) {
                for tag in tags {
                    let tag_lower = tag.to_lowercase();
                    *counts.entry(tag_lower).or_default() += 1;
                }
            }
        }

        // Sort by count and take top N
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by_key(|b| std::cmp::Reverse(b.1));

        Ok(sorted
            .into_iter()
            .take(limit)
            .map(|(tag, count)| TagCount { tag, count })
            .collect())
    }

    /// Query top mentions from the `mentions_json` column.
    #[allow(clippy::cast_sign_loss)]
    fn query_top_mentions(storage: &Storage, limit: usize) -> Result<Vec<TagCount>> {
        let conn = storage.connection();

        let query = "SELECT mentions_json FROM tweets WHERE mentions_json IS NOT NULL AND mentions_json != '[]' AND mentions_json != ''";
        let mut stmt = conn.prepare(query)?;

        let mut counts: std::collections::HashMap<String, u64> = std::collections::HashMap::new();

        let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;

        for json_str in rows.flatten() {
            // Parse JSON array of mention objects
            if let Ok(mentions) = serde_json::from_str::<Vec<serde_json::Value>>(&json_str) {
                for mention in mentions {
                    if let Some(screen_name) = mention.get("screen_name").and_then(|v| v.as_str()) {
                        let name_lower = screen_name.to_lowercase();
                        *counts.entry(name_lower).or_default() += 1;
                    }
                }
            }
        }

        // Sort by count and take top N
        let mut sorted: Vec<_> = counts.into_iter().collect();
        sorted.sort_by_key(|b| std::cmp::Reverse(b.1));

        Ok(sorted
            .into_iter()
            .take(limit)
            .map(|(tag, count)| TagCount { tag, count })
            .collect())
    }
}

/// Format length distribution as a horizontal bar chart.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::manual_checked_ops)]
pub fn format_length_distribution(distribution: &[LengthBucket]) -> String {
    let max_count = distribution.iter().map(|b| b.count).max().unwrap_or(1);

    distribution
        .iter()
        .map(|bucket| {
            let bar_len = if max_count > 0 {
                let scaled = bucket.count.saturating_mul(15) / max_count;
                usize::try_from(scaled).unwrap_or(usize::MAX)
            } else {
                0
            };
            format!(
                "{:>7} {} {:>5.1}%",
                bucket.label,
                "█".repeat(bar_len),
                bucket.percentage
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Format top tags as a compact inline list.
#[must_use]
pub fn format_top_tags(tags: &[TagCount], prefix: &str) -> String {
    if tags.is_empty() {
        return String::new();
    }

    tags.iter()
        .take(6)
        .map(|t| format!("{}{} ({})", prefix, t.tag, format_number_u64(t.count)))
        .collect::<Vec<_>>()
        .join("  ")
}

/// Truncate text to approximately `max_len` characters at a word boundary.
/// Uses character count, not byte count, to properly handle UTF-8.
#[must_use]
fn truncate_text(text: &str, max_len: usize) -> String {
    // Normalize whitespace first
    let text = text.replace('\n', " ").replace('\r', "");
    let char_count = text.chars().count();

    if char_count <= max_len {
        return text;
    }

    if max_len <= 3 {
        // Can't fit any text + "...", just truncate without ellipsis
        return text.chars().take(max_len).collect();
    }

    // Take max_len - 3 characters to leave room for "..."
    let truncated: String = text.chars().take(max_len - 3).collect();

    // Try to find a word boundary (space) to break at
    truncated.rfind(' ').map_or_else(
        || format!("{truncated}..."),
        |last_space| format!("{}...", &truncated[..last_space]),
    )
}

/// Format likes histogram as a horizontal bar chart.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::manual_checked_ops)]
pub fn format_likes_histogram(histogram: &[LikesBucket]) -> String {
    let max_count = histogram.iter().map(|b| b.count).max().unwrap_or(1);

    histogram
        .iter()
        .map(|bucket| {
            let bar_len = if max_count > 0 {
                let scaled = bucket.count.saturating_mul(20) / max_count;
                usize::try_from(scaled).unwrap_or(usize::MAX)
            } else {
                0
            };
            format!(
                "{:>7} {} {:>5.1}%",
                bucket.label,
                "█".repeat(bar_len),
                bucket.percentage
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Generate a sparkline from monthly engagement data.
#[must_use]
pub fn sparkline_from_monthly(monthly: &[MonthlyEngagement], width: usize) -> String {
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let values: Vec<u64> = monthly.iter().map(|m| m.avg_engagement as u64).collect();
    sparkline(&values, width)
}

/// Generate an ASCII sparkline from a slice of values.
///
/// Uses Unicode block characters: ▁▂▃▄▅▆▇█
///
/// # Arguments
/// * `values` - The values to visualize
/// * `width` - Target width (values will be bucketed if len > width)
///
/// # Returns
/// A string of sparkline characters
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn sparkline(values: &[u64], width: usize) -> String {
    if values.is_empty() || width == 0 {
        return String::new();
    }

    let blocks = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    // Bucket values if we have more than width
    let bucketed: Vec<u64> = if values.len() <= width {
        values.to_vec()
    } else {
        let bucket_size = values.len().div_ceil(width);
        values
            .chunks(bucket_size)
            .map(|chunk| chunk.iter().sum::<u64>() / chunk.len() as u64)
            .collect()
    };

    let max = *bucketed.iter().max().unwrap_or(&1);
    if max == 0 {
        return "▁".repeat(bucketed.len().min(width));
    }

    bucketed
        .iter()
        .take(width)
        .map(|&v| {
            let idx = ((v as f64 / max as f64) * 7.0) as usize;
            blocks[idx.min(7)]
        })
        .collect()
}

/// Generate a sparkline from daily counts.
#[must_use]
pub fn sparkline_from_daily(daily_counts: &[DailyCount], width: usize) -> String {
    let values: Vec<u64> = daily_counts.iter().map(|d| d.count).collect();
    sparkline(&values, width)
}

/// Format day-of-week distribution as a mini-bar chart.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
pub fn format_dow_distribution(distribution: &[u64; 7]) -> String {
    let days = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
    let max = *distribution.iter().max().unwrap_or(&1);

    days.iter()
        .zip(distribution.iter())
        .map(|(day, &count)| {
            let bar_len = if max > 0 {
                ((count as f64 / max as f64) * 10.0) as usize
            } else {
                0
            };
            format!("{day}: {}", "█".repeat(bar_len))
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Format hourly distribution as a 24-hour sparkline.
#[must_use]
pub fn format_hourly_sparkline(distribution: &[u64; 24]) -> String {
    sparkline(distribution, 24)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{ArchiveInfo, Tweet, TweetMedia, TweetUrl, UserMention};
    use crate::storage::Storage;
    use tracing::debug;

    #[test]
    fn test_sparkline_empty() {
        assert_eq!(sparkline(&[], 10), "");
    }

    #[test]
    fn test_sparkline_single_value() {
        let result = sparkline(&[5], 1);
        assert_eq!(result.chars().count(), 1);
        assert_eq!(result, "█"); // Single value is max
    }

    #[test]
    fn test_sparkline_values() {
        let values = vec![1, 5, 10, 8, 3, 1];
        let result = sparkline(&values, 6);
        assert_eq!(result.chars().count(), 6);
        // The highest value (10) should produce █
        assert!(result.contains('█'));
        // The lowest values (1) should produce ▁
        assert!(result.contains('▁'));
    }

    #[test]
    fn test_sparkline_all_zeros() {
        let values = vec![0, 0, 0, 0];
        let result = sparkline(&values, 4);
        assert_eq!(result, "▁▁▁▁");
    }

    #[test]
    fn test_sparkline_bucketing() {
        // 12 values bucketed into 6
        let values = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
        let result = sparkline(&values, 6);
        assert_eq!(result.chars().count(), 6);
    }

    #[test]
    fn test_find_longest_gap_empty() {
        let (gap, start, end) = TemporalStats::find_longest_gap(&[]);
        assert_eq!(gap, 0);
        assert!(start.is_none());
        assert!(end.is_none());
    }

    #[test]
    fn test_find_longest_gap_single() {
        let counts = vec![DailyCount {
            date: NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(),
            count: 5,
        }];
        let (gap, start, end) = TemporalStats::find_longest_gap(&counts);
        assert_eq!(gap, 0);
        assert!(start.is_none());
        assert!(end.is_none());
    }

    #[test]
    fn test_find_longest_gap_normal() {
        let counts = vec![
            DailyCount {
                date: NaiveDate::from_ymd_opt(2023, 1, 1).unwrap(),
                count: 5,
            },
            DailyCount {
                date: NaiveDate::from_ymd_opt(2023, 1, 5).unwrap(),
                count: 3,
            }, // 4 day gap
            DailyCount {
                date: NaiveDate::from_ymd_opt(2023, 1, 20).unwrap(),
                count: 2,
            }, // 15 day gap (longest)
            DailyCount {
                date: NaiveDate::from_ymd_opt(2023, 1, 22).unwrap(),
                count: 1,
            }, // 2 day gap
        ];
        let (gap, start, end) = TemporalStats::find_longest_gap(&counts);
        assert_eq!(gap, 15);
        assert_eq!(start, Some(NaiveDate::from_ymd_opt(2023, 1, 5).unwrap()));
        assert_eq!(end, Some(NaiveDate::from_ymd_opt(2023, 1, 20).unwrap()));
    }

    fn base_tweet(id: &str, created_at: &str, text: &str) -> Tweet {
        let created_at = DateTime::parse_from_rfc3339(created_at)
            .unwrap()
            .with_timezone(&Utc);
        Tweet {
            id: id.to_string(),
            created_at,
            full_text: text.to_string(),
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
            urls: Vec::new(),
            media: Vec::new(),
        }
    }

    fn storage_with_tweets(tweets: &[Tweet], account_id: &str) -> Storage {
        let mut storage = Storage::open_memory().unwrap();
        let info = ArchiveInfo {
            account_id: account_id.to_string(),
            username: "tester".to_string(),
            display_name: None,
            archive_size_bytes: 0,
            generation_date: Utc::now(),
            is_partial: false,
        };
        storage.store_archive_info(&info).unwrap();
        storage.store_tweets(tweets).unwrap();
        storage
    }

    fn assert_approx(actual: f64, expected: f64, epsilon: f64) {
        let diff = (actual - expected).abs();
        assert!(
            diff <= epsilon,
            "expected {expected:.3}, got {actual:.3} (diff {diff:.3})"
        );
    }

    #[test]
    fn test_temporal_hourly_distribution() {
        debug!("test_temporal_hourly_distribution: setup");
        let tweets = vec![
            base_tweet("t1", "2023-01-01T09:00:00Z", "Morning"),
            base_tweet("t2", "2023-01-01T09:30:00Z", "Also morning"),
            base_tweet("t3", "2023-01-01T21:00:00Z", "Evening"),
        ];
        let storage = storage_with_tweets(&tweets, "user-1");
        let stats = TemporalStats::compute(&storage).unwrap();
        assert_eq!(stats.hourly_distribution[9], 2);
        assert_eq!(stats.hourly_distribution[21], 1);
        assert_eq!(stats.active_days_count, 1);
        debug!("test_temporal_hourly_distribution: done");
    }

    #[test]
    fn test_engagement_histogram_buckets() {
        debug!("test_engagement_histogram_buckets: setup");
        let mut tweets = Vec::new();
        for (idx, favorites) in [0, 2, 7, 20, 30, 70, 200, 700].iter().enumerate() {
            let mut tweet = base_tweet(&format!("t{idx}"), "2023-01-02T10:00:00Z", "Engagement");
            tweet.favorite_count = *favorites;
            tweets.push(tweet);
        }
        let storage = storage_with_tweets(&tweets, "user-1");
        let stats = EngagementStats::compute(&storage, 5).unwrap();
        let counts: Vec<u64> = stats.likes_histogram.iter().map(|b| b.count).collect();
        assert_eq!(counts, vec![1, 1, 1, 1, 1, 1, 1, 1]);
        assert_approx(stats.likes_histogram[0].percentage, 12.5, 0.01);
        debug!("test_engagement_histogram_buckets: done");
    }

    #[test]
    fn test_top_tweets_ordering() {
        debug!("test_top_tweets_ordering: setup");
        let mut tweets = Vec::new();
        let mut a = base_tweet("a", "2023-01-03T00:00:00Z", "A");
        a.favorite_count = 10;
        a.retweet_count = 5; // total 15
        tweets.push(a);
        let mut b = base_tweet("b", "2023-01-04T00:00:00Z", "B");
        b.favorite_count = 100;
        b.retweet_count = 20; // total 120
        tweets.push(b);
        let mut c = base_tweet("c", "2023-01-05T00:00:00Z", "C");
        c.favorite_count = 50;
        c.retweet_count = 10; // total 60
        tweets.push(c);
        let storage = storage_with_tweets(&tweets, "user-1");
        let stats = EngagementStats::compute(&storage, 3).unwrap();
        assert_eq!(stats.top_tweets[0].total_engagement, 120);
        assert_eq!(stats.top_tweets[1].total_engagement, 60);
        debug!("test_top_tweets_ordering: done");
    }

    #[test]
    fn test_content_hashtag_extraction() {
        debug!("test_content_hashtag_extraction: setup");
        let mut t1 = base_tweet("t1", "2023-02-01T00:00:00Z", "Hello");
        t1.hashtags = vec!["Rust".to_string(), "Programming".to_string()];
        let mut t2 = base_tweet("t2", "2023-02-02T00:00:00Z", "More");
        t2.hashtags = vec!["rust".to_string()];
        let mut t3 = base_tweet("t3", "2023-02-03T00:00:00Z", "Tech");
        t3.hashtags = vec!["Tech".to_string()];
        let storage = storage_with_tweets(&[t1, t2, t3], "user-1");
        let stats = ContentStats::compute(&storage, 5).unwrap();
        assert_eq!(stats.top_hashtags[0].tag, "rust");
        assert_eq!(stats.top_hashtags[0].count, 2);
        debug!("test_content_hashtag_extraction: done");
    }

    #[test]
    fn test_content_media_ratio() {
        debug!("test_content_media_ratio: setup");
        let mut tweets = Vec::new();
        for idx in 0..10 {
            let mut tweet = base_tweet(&format!("t{idx}"), "2023-03-01T00:00:00Z", "Media");
            if idx < 3 {
                tweet.media = vec![TweetMedia {
                    id: format!("m{idx}"),
                    media_type: "photo".to_string(),
                    url: "https://example.com".to_string(),
                    local_path: None,
                }];
            }
            tweets.push(tweet);
        }
        let storage = storage_with_tweets(&tweets, "user-1");
        let stats = ContentStats::compute(&storage, 5).unwrap();
        assert_approx(stats.media_ratio, 30.0, 0.01);
        debug!("test_content_media_ratio: done");
    }

    #[test]
    fn test_thread_detection() {
        debug!("test_thread_detection: setup");
        let account_id = "user-123";
        let t1 = base_tweet("t1", "2023-04-01T00:00:00Z", "Root");
        let mut t2 = base_tweet("t2", "2023-04-01T00:10:00Z", "Thread reply");
        t2.in_reply_to_status_id = Some("t1".to_string());
        t2.in_reply_to_user_id = Some(account_id.to_string());
        let mut t3 = base_tweet("t3", "2023-04-01T00:20:00Z", "Another thread");
        t3.in_reply_to_status_id = Some("t2".to_string());
        t3.in_reply_to_user_id = Some(account_id.to_string());
        let mut t4 = base_tweet("t4", "2023-04-01T01:00:00Z", "Reply to other");
        t4.in_reply_to_status_id = Some("x1".to_string());
        t4.in_reply_to_user_id = Some("other-user".to_string());
        let storage = storage_with_tweets(&[t1, t2, t3, t4], account_id);
        let stats = ContentStats::compute(&storage, 5).unwrap();
        assert_eq!(stats.thread_count, 2);
        assert_eq!(stats.total_count, 4);
        debug!("test_thread_detection: done");
    }

    #[test]
    fn test_empty_archive_stats() {
        debug!("test_empty_archive_stats: setup");
        let storage = storage_with_tweets(&[], "user-1");
        let temporal = TemporalStats::compute(&storage).unwrap();
        assert!(temporal.daily_counts.is_empty());
        assert_eq!(temporal.total_days_in_range, 0);
        let engagement = EngagementStats::compute(&storage, 5).unwrap();
        assert_eq!(engagement.total_likes, 0);
        let content = ContentStats::compute(&storage, 5).unwrap();
        assert_eq!(content.total_count, 0);
        debug!("test_empty_archive_stats: done");
    }

    #[test]
    fn test_single_tweet_archive() {
        debug!("test_single_tweet_archive: setup");
        let tweet = base_tweet("t1", "2023-05-01T12:00:00Z", "Solo");
        let storage = storage_with_tweets(&[tweet], "user-1");
        let temporal = TemporalStats::compute(&storage).unwrap();
        assert_eq!(temporal.active_days_count, 1);
        assert_eq!(temporal.total_days_in_range, 1);
        assert_eq!(temporal.longest_gap_days, 0);
        let engagement = EngagementStats::compute(&storage, 5).unwrap();
        assert_eq!(engagement.top_tweets.len(), 1);
        let content = ContentStats::compute(&storage, 5).unwrap();
        assert_eq!(content.total_count, 1);
        debug!("test_single_tweet_archive: done");
    }

    #[test]
    fn test_temporal_stats_performance_smoke() {
        debug!("test_temporal_stats_performance_smoke: setup");
        let mut tweets = Vec::new();
        for day in 0..365 {
            let date = NaiveDate::from_ymd_opt(2023, 1, 1)
                .unwrap()
                .checked_add_days(chrono::Days::new(day))
                .unwrap();
            let created_at = date.and_hms_opt(12, 0, 0).unwrap();
            let mut tweet = base_tweet(
                &format!("t{day}"),
                &created_at.format("%Y-%m-%dT%H:%M:%SZ").to_string(),
                "Load test",
            );
            tweet.favorite_count = 1;
            tweets.push(tweet);
        }
        let storage = storage_with_tweets(&tweets, "user-1");
        let start = std::time::Instant::now();
        let _ = TemporalStats::compute(&storage).unwrap();
        let elapsed = start.elapsed();
        assert!(
            elapsed < std::time::Duration::from_secs(2),
            "TemporalStats::compute took {elapsed:?}"
        );
        debug!("test_temporal_stats_performance_smoke: done");
    }

    #[test]
    fn test_mentions_and_links() {
        debug!("test_mentions_and_links: setup");
        let mut tweet = base_tweet("t1", "2023-06-01T00:00:00Z", "Hello");
        tweet.user_mentions = vec![UserMention {
            id: "u1".to_string(),
            screen_name: "Friend".to_string(),
            name: Some("Friend".to_string()),
        }];
        tweet.urls = vec![TweetUrl {
            url: "https://t.co/test".to_string(),
            expanded_url: Some("https://example.com".to_string()),
            display_url: Some("example.com".to_string()),
        }];
        let storage = storage_with_tweets(&[tweet], "user-1");
        let stats = ContentStats::compute(&storage, 5).unwrap();
        assert_eq!(stats.top_mentions[0].tag, "friend");
        assert_eq!(stats.top_mentions[0].count, 1);
        assert_approx(stats.link_ratio, 100.0, 0.01);
        debug!("test_mentions_and_links: done");
    }

    #[test]
    fn test_engagement_monthly_trend() {
        debug!("test_engagement_monthly_trend: setup");
        let mut jan = base_tweet("t1", "2023-01-15T00:00:00Z", "Jan");
        jan.favorite_count = 10;
        let mut feb = base_tweet("t2", "2023-02-15T00:00:00Z", "Feb");
        feb.favorite_count = 20;
        let storage = storage_with_tweets(&[jan, feb], "user-1");
        let stats = EngagementStats::compute(&storage, 5).unwrap();
        assert_eq!(stats.monthly_trend.len(), 2);
        assert_eq!(stats.monthly_trend[0].month, "2023-01");
        assert_eq!(stats.monthly_trend[1].month, "2023-02");
        assert_approx(stats.monthly_trend[0].avg_engagement, 10.0, 0.01);
        debug!("test_engagement_monthly_trend: done");
    }

    #[test]
    fn test_avg_length_and_distribution() {
        debug!("test_avg_length_and_distribution: setup");
        let short = base_tweet("t1", "2023-07-01T00:00:00Z", "short");
        let long_text = "L".repeat(200);
        let mut long = base_tweet("t2", "2023-07-02T00:00:00Z", &long_text);
        long.favorite_count = 1;
        let storage = storage_with_tweets(&[short, long], "user-1");
        let stats = ContentStats::compute(&storage, 5).unwrap();
        assert!(stats.avg_tweet_length >= 5.0);
        assert_eq!(stats.length_distribution.len(), 4);
        assert_eq!(stats.total_count, 2);
        debug!("test_avg_length_and_distribution: done");
    }

    #[test]
    fn test_engagement_with_nulls_safe() {
        debug!("test_engagement_with_nulls_safe: setup");
        let storage = Storage::open_memory().unwrap();
        let info = ArchiveInfo {
            account_id: "user-1".to_string(),
            username: "tester".to_string(),
            display_name: None,
            archive_size_bytes: 0,
            generation_date: Utc::now(),
            is_partial: false,
        };
        storage.store_archive_info(&info).unwrap();
        storage
            .connection()
            .execute(
                r"
                INSERT INTO tweets
                (id, created_at, full_text, source, favorite_count, retweet_count, lang,
                 in_reply_to_status_id, in_reply_to_user_id, in_reply_to_screen_name,
                 is_retweet, hashtags_json, mentions_json, urls_json, media_json)
                VALUES (?, ?, ?, ?, NULL, 2, NULL, NULL, NULL, NULL, 0, NULL, NULL, NULL, NULL)
                ",
                ["null-1", "2023-08-01T00:00:00Z", "Null engagement", ""],
            )
            .unwrap();
        let stats = EngagementStats::compute(&storage, 5).unwrap();
        assert_eq!(stats.top_tweets.len(), 1);
        assert_eq!(stats.total_likes, 0);
        assert_eq!(stats.total_retweets, 2);
        debug!("test_engagement_with_nulls_safe: done");
    }

    #[test]
    fn test_longest_gap_calculation() {
        debug!("test_longest_gap_calculation: setup");
        let tweets = vec![
            base_tweet("t1", "2023-01-01T00:00:00Z", "A"),
            base_tweet("t2", "2023-01-10T00:00:00Z", "B"),
            base_tweet("t3", "2023-01-12T00:00:00Z", "C"),
        ];
        let storage = storage_with_tweets(&tweets, "user-1");
        let stats = TemporalStats::compute(&storage).unwrap();
        assert_eq!(stats.longest_gap_days, 9);
        assert_eq!(
            stats.longest_gap_start,
            Some(NaiveDate::from_ymd_opt(2023, 1, 1).unwrap())
        );
        assert_eq!(
            stats.longest_gap_end,
            Some(NaiveDate::from_ymd_opt(2023, 1, 10).unwrap())
        );
        debug!("test_longest_gap_calculation: done");
    }

    #[test]
    fn test_avg_engagement_matches_totals() {
        debug!("test_avg_engagement_matches_totals: setup");
        let mut t1 = base_tweet("t1", "2023-09-01T00:00:00Z", "A");
        t1.favorite_count = 10;
        t1.retweet_count = 0;
        let mut t2 = base_tweet("t2", "2023-09-02T00:00:00Z", "B");
        t2.favorite_count = 0;
        t2.retweet_count = 10;
        let storage = storage_with_tweets(&[t1, t2], "user-1");
        let stats = EngagementStats::compute(&storage, 5).unwrap();
        assert_eq!(stats.total_likes, 10);
        assert_eq!(stats.total_retweets, 10);
        assert_approx(stats.avg_engagement, 10.0, 0.01);
        debug!("test_avg_engagement_matches_totals: done");
    }

    #[test]
    fn test_truncate_text_boundary() {
        debug!("test_truncate_text_boundary: setup");
        let text = "This is a sentence with words";
        let truncated = truncate_text(text, 10);
        assert!(truncated.ends_with("..."));
        assert!(truncated.len() <= 13);
        debug!("test_truncate_text_boundary: done");
    }

    #[test]
    fn test_format_helpers() {
        debug!("test_format_helpers: setup");
        let dist = [
            LengthBucket {
                label: "0-50".to_string(),
                count: 2,
                percentage: 50.0,
            },
            LengthBucket {
                label: "51-140".to_string(),
                count: 2,
                percentage: 50.0,
            },
        ];
        let formatted = format_length_distribution(&dist);
        assert!(formatted.contains("0-50"));
        let tags = vec![
            TagCount {
                tag: "rust".to_string(),
                count: 2,
            },
            TagCount {
                tag: "cli".to_string(),
                count: 1,
            },
        ];
        let list = format_top_tags(&tags, "#");
        assert!(list.contains("#rust"));
        let likes = vec![LikesBucket {
            label: "0".to_string(),
            min: 0,
            max: 0,
            count: 1,
            percentage: 100.0,
        }];
        let formatted_likes = format_likes_histogram(&likes);
        assert!(formatted_likes.contains('0'));
        debug!("test_format_helpers: done");
    }
}
