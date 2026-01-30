//! Hybrid search with Reciprocal Rank Fusion (RRF).
//!
//! Combines lexical (keyword) and semantic (vector) search results
//! using RRF scoring for optimal relevance.
//!
//! # Algorithm
//!
//! RRF score = Σ 1/(K + rank + 1)
//!
//! Where:
//! - K = 60 (constant, empirically optimal)
//! - rank = position in result list (0-indexed)
//!
//! Results appearing in both lists get scores from both, naturally
//! boosting documents that match both keyword and meaning.
//!
//! # Tie-Breaking
//!
//! For deterministic ordering:
//! 1. RRF score (descending)
//! 2. Appears in both lists (bonus)
//! 3. Document ID (ascending)

use crate::model::{SearchResult, SearchResultType};
use crate::vector::VectorSearchResult;
use clap::ValueEnum;
use std::cmp::Ordering;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct DocKey<'a> {
    id: &'a str,
    doc_type: &'a str,
}

impl<'a> DocKey<'a> {
    #[allow(clippy::missing_const_for_fn)]
    fn new(id: &'a str, doc_type: &'a str) -> Self {
        Self { id, doc_type }
    }
}

/// RRF constant K. Empirically, K=60 works well for most use cases.
const RRF_K: f32 = 60.0;

/// Multiplier for candidate fetching. Fetch 3x more candidates from each
/// source to ensure good coverage after fusion.
pub const CANDIDATE_MULTIPLIER: usize = 3;

/// Search mode for hybrid search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, ValueEnum)]
pub enum SearchMode {
    /// Keyword-only search using BM25.
    #[value(alias = "keyword", alias = "bm25")]
    Lexical,
    /// Semantic-only search using vector similarity.
    #[value(alias = "vector", alias = "embedding")]
    Semantic,
    /// Hybrid search combining lexical and semantic with RRF (default).
    #[default]
    #[value(alias = "rrf", alias = "both")]
    Hybrid,
    /// Two-tier progressive search: fast results refined by quality model.
    #[value(alias = "progressive", alias = "2tier")]
    TwoTier,
}

impl std::fmt::Display for SearchMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Lexical => write!(f, "lexical"),
            Self::Semantic => write!(f, "semantic"),
            Self::Hybrid => write!(f, "hybrid"),
            Self::TwoTier => write!(f, "two-tier"),
        }
    }
}

impl std::str::FromStr for SearchMode {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "lexical" | "keyword" | "bm25" => Ok(Self::Lexical),
            "semantic" | "vector" | "embedding" => Ok(Self::Semantic),
            "hybrid" | "rrf" | "both" => Ok(Self::Hybrid),
            "two-tier" | "twotier" | "two_tier" | "progressive" | "2tier" => Ok(Self::TwoTier),
            _ => Err(format!(
                "unknown search mode: '{s}'. Use 'lexical', 'semantic', 'hybrid', or 'two-tier'"
            )),
        }
    }
}

/// Hybrid score tracking for RRF fusion.
#[derive(Debug, Default, Clone)]
struct HybridScore {
    /// RRF score (sum of 1/(K+rank+1) from each list).
    rrf: f32,
    /// Rank in lexical results (if present).
    lexical_rank: Option<usize>,
    /// Rank in semantic results (if present).
    semantic_rank: Option<usize>,
}

/// Fused search hit combining information from both sources.
#[derive(Debug, Clone)]
pub struct FusedHit<'a> {
    /// Document ID.
    pub doc_id: &'a str,
    /// Document type.
    pub doc_type: &'a str,
    /// Fused RRF score.
    pub score: f32,
    /// Index into the lexical results (if present).
    pub lexical_rank: Option<usize>,
    /// Whether this hit appeared in both lexical and semantic results.
    pub in_both: bool,
}

const fn result_type_str(result_type: SearchResultType) -> &'static str {
    match result_type {
        SearchResultType::Tweet => "tweet",
        SearchResultType::Like => "like",
        SearchResultType::DirectMessage => "dm",
        SearchResultType::GrokMessage => "grok",
    }
}

/// Fuse lexical and semantic search results using RRF.
///
/// # Arguments
///
/// * `lexical` - Results from keyword (Tantivy) search
/// * `semantic` - Results from vector similarity search
/// * `limit` - Maximum number of results to return
/// * `offset` - Number of results to skip (for pagination)
///
/// # Returns
///
/// Fused results sorted by RRF score, with deterministic tie-breaking.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn rrf_fuse<'a>(
    lexical: &'a [SearchResult],
    semantic: &'a [VectorSearchResult],
    limit: usize,
    offset: usize,
) -> Vec<FusedHit<'a>> {
    if limit == 0 {
        return Vec::new();
    }

    let mut scores: HashMap<DocKey<'a>, HybridScore> =
        HashMap::with_capacity(lexical.len() + semantic.len());

    // Process lexical results (rank 0, 1, 2, ...)
    for (rank, hit) in lexical.iter().enumerate() {
        let doc_type = result_type_str(hit.result_type);
        let key = DocKey::new(hit.id.as_str(), doc_type);
        let entry = scores.entry(key).or_default();
        entry.rrf += 1.0 / (RRF_K + rank as f32 + 1.0);
        entry.lexical_rank = Some(rank);
    }

    // Process semantic results (rank 0, 1, 2, ...)
    for (rank, hit) in semantic.iter().enumerate() {
        let key = DocKey::new(hit.doc_id.as_str(), hit.doc_type);
        let entry = scores.entry(key).or_default();
        entry.rrf += 1.0 / (RRF_K + rank as f32 + 1.0);
        entry.semantic_rank = Some(rank);
    }

    // Convert to fused hits
    let mut fused: Vec<FusedHit<'a>> = scores
        .into_iter()
        .map(|(key, score)| {
            let in_both = score.lexical_rank.is_some() && score.semantic_rank.is_some();
            FusedHit {
                doc_id: key.id,
                doc_type: key.doc_type,
                score: score.rrf,
                lexical_rank: score.lexical_rank,
                in_both,
            }
        })
        .collect();

    // Sort with deterministic tie-breaking
    fused.sort_by(|a, b| {
        // Level 1: RRF score (descending)
        b.score
            .total_cmp(&a.score)
            // Level 2: Prefer hits appearing in both lists
            .then_with(|| match (b.in_both, a.in_both) {
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                _ => Ordering::Equal,
            })
            // Level 3: Document ID (ascending) for determinism
            .then_with(|| a.doc_id.cmp(b.doc_id))
            .then_with(|| a.doc_type.cmp(b.doc_type))
    });

    // Apply offset and limit
    let start = offset.min(fused.len());
    let end = start.saturating_add(limit).min(fused.len());

    fused.into_iter().skip(start).take(end - start).collect()
}

/// Calculate the number of candidates to fetch for hybrid search.
///
/// Fetches 3x the limit from each source to ensure good coverage
/// after RRF fusion filters results.
#[must_use]
pub const fn candidate_count(limit: usize, offset: usize) -> usize {
    limit
        .saturating_add(offset)
        .saturating_mul(CANDIDATE_MULTIPLIER)
}

/// Normalize scores to [0, 1] using min-max scaling.
///
/// If all scores are equal or the input is empty, scores are set to 1.0.
pub fn min_max_normalize(results: &mut [VectorSearchResult]) {
    if results.is_empty() {
        return;
    }

    let min = results
        .iter()
        .map(|r| r.score)
        .fold(f32::INFINITY, f32::min);
    let max = results
        .iter()
        .map(|r| r.score)
        .fold(f32::NEG_INFINITY, f32::max);

    let range = max - min;
    if range.abs() < f32::EPSILON {
        // All same score — assign 1.0
        for r in results.iter_mut() {
            r.score = 1.0;
        }
    } else {
        for r in results.iter_mut() {
            r.score = (r.score - min) / range;
        }
    }
}

/// Blend fast and quality search results with weighted scoring.
///
/// `blend_factor` controls the weight: 0.0 = fast-only, 1.0 = quality-only.
/// Documents appearing in both lists get blended scores.
/// Documents in only one list get that score scaled by the appropriate weight.
///
/// Results are sorted by blended score descending.
#[must_use]
pub fn blend_two_tier(
    fast: &[VectorSearchResult],
    quality: &[VectorSearchResult],
    blend_factor: f32,
) -> Vec<VectorSearchResult> {
    let fast_weight = 1.0 - blend_factor;
    let quality_weight = blend_factor;

    // Build lookup from quality results by (doc_id, doc_type)
    let mut quality_map: HashMap<(&str, &str), f32> = HashMap::with_capacity(quality.len());
    for hit in quality {
        quality_map.insert((hit.doc_id.as_str(), hit.doc_type), hit.score);
    }

    let mut seen: std::collections::HashSet<(String, &str)> = std::collections::HashSet::new();
    let mut blended: Vec<VectorSearchResult> = Vec::with_capacity(fast.len() + quality.len());

    // Process fast results, blending with quality if available
    for hit in fast {
        let key = (hit.doc_id.as_str(), hit.doc_type);
        let fast_score = hit.score * fast_weight;
        let quality_score = quality_map.remove(&key).map_or(0.0, |s| s * quality_weight);

        blended.push(VectorSearchResult {
            doc_id: hit.doc_id.clone(),
            doc_type: hit.doc_type,
            score: fast_score + quality_score,
        });
        seen.insert((hit.doc_id.clone(), hit.doc_type));
    }

    // Add quality-only results (not in fast)
    for hit in quality {
        if !seen.contains(&(hit.doc_id.clone(), hit.doc_type)) {
            blended.push(VectorSearchResult {
                doc_id: hit.doc_id.clone(),
                doc_type: hit.doc_type,
                score: hit.score * quality_weight,
            });
        }
    }

    // Sort by blended score descending, then doc_id ascending for determinism
    blended.sort_by(|a, b| {
        b.score
            .total_cmp(&a.score)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
            .then_with(|| a.doc_type.cmp(b.doc_type))
    });

    blended
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::SearchResultType;
    use chrono::Utc;

    fn make_lexical_hit(id: &str, score: f32, result_type: SearchResultType) -> SearchResult {
        SearchResult {
            result_type,
            id: id.to_string(),
            text: format!("Text for {id}"),
            created_at: Utc::now(),
            score,
            highlights: vec![],
            metadata: serde_json::Value::Null,
            rerank_score: None,
        }
    }

    /// Intern a `doc_type` string to `&'static str` for tests.
    fn intern_doc_type(doc_type: &str) -> &'static str {
        match doc_type {
            "like" => "like",
            "dm" => "dm",
            "grok" => "grok",
            _ => "tweet", // Default for "tweet" and unknown types
        }
    }

    fn make_semantic_hit(doc_id: &str, score: f32, doc_type: &str) -> VectorSearchResult {
        VectorSearchResult {
            doc_id: doc_id.to_string(),
            doc_type: intern_doc_type(doc_type),
            score,
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Hash)]
    struct LegacyDocKey {
        id: String,
        doc_type: String,
    }

    #[derive(Debug, Clone)]
    struct LegacyFusedHit {
        doc_id: String,
        doc_type: String,
        score: f32,
        lexical_rank: Option<usize>,
        in_both: bool,
    }

    #[allow(clippy::cast_precision_loss)]
    fn legacy_rrf_fuse(
        lexical: &[SearchResult],
        semantic: &[VectorSearchResult],
        limit: usize,
        offset: usize,
    ) -> Vec<LegacyFusedHit> {
        if limit == 0 {
            return Vec::new();
        }

        let mut scores: HashMap<LegacyDocKey, HybridScore> = HashMap::new();

        for (rank, hit) in lexical.iter().enumerate() {
            let doc_type = hit.result_type.to_string();
            let key = LegacyDocKey {
                id: hit.id.clone(),
                doc_type: doc_type.clone(),
            };
            let entry = scores.entry(key).or_default();
            entry.rrf += 1.0 / (RRF_K + rank as f32 + 1.0);
            entry.lexical_rank = Some(rank);
        }

        for (rank, hit) in semantic.iter().enumerate() {
            let key = LegacyDocKey {
                id: hit.doc_id.clone(),
                doc_type: hit.doc_type.to_string(), // &'static str -> String
            };
            let entry = scores.entry(key).or_default();
            entry.rrf += 1.0 / (RRF_K + rank as f32 + 1.0);
            entry.semantic_rank = Some(rank);
        }

        let mut fused: Vec<LegacyFusedHit> = scores
            .into_iter()
            .map(|(key, score)| {
                let in_both = score.lexical_rank.is_some() && score.semantic_rank.is_some();
                LegacyFusedHit {
                    doc_id: key.id,
                    doc_type: key.doc_type,
                    score: score.rrf,
                    lexical_rank: score.lexical_rank,
                    in_both,
                }
            })
            .collect();

        fused.sort_by(|a, b| {
            b.score
                .total_cmp(&a.score)
                .then_with(|| match (b.in_both, a.in_both) {
                    (true, false) => Ordering::Greater,
                    (false, true) => Ordering::Less,
                    _ => Ordering::Equal,
                })
                .then_with(|| a.doc_id.cmp(&b.doc_id))
                .then_with(|| a.doc_type.cmp(&b.doc_type))
        });

        let start = offset.min(fused.len());
        let end = start.saturating_add(limit).min(fused.len());

        fused.into_iter().skip(start).take(end - start).collect()
    }

    fn next_u32(seed: &mut u64) -> u32 {
        *seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
        (*seed >> 32) as u32
    }

    fn doc_type_for(idx: u32) -> (SearchResultType, &'static str) {
        match idx % 4 {
            0 => (SearchResultType::Tweet, "tweet"),
            1 => (SearchResultType::Like, "like"),
            2 => (SearchResultType::DirectMessage, "dm"),
            _ => (SearchResultType::GrokMessage, "grok"),
        }
    }

    fn usize_to_f32(value: usize) -> f32 {
        f32::from(u16::try_from(value).unwrap_or(u16::MAX))
    }

    #[test]
    fn test_rrf_basic() {
        let lexical = vec![
            make_lexical_hit("A", 10.0, SearchResultType::Tweet),
            make_lexical_hit("B", 8.0, SearchResultType::Tweet),
            make_lexical_hit("C", 6.0, SearchResultType::Tweet),
        ];
        let semantic = vec![
            make_semantic_hit("A", 0.9, "tweet"),
            make_semantic_hit("D", 0.8, "tweet"),
            make_semantic_hit("B", 0.7, "tweet"),
        ];

        let fused = rrf_fuse(&lexical, &semantic, 10, 0);

        // A should be first (appears in both, rank 0 in both)
        assert_eq!(fused[0].doc_id, "A");
        assert!(fused[0].in_both);

        // B should be second (appears in both, but lower combined rank)
        assert_eq!(fused[1].doc_id, "B");
        assert!(fused[1].in_both);
    }

    #[test]
    fn test_rrf_scoring() {
        let lexical = vec![make_lexical_hit("A", 10.0, SearchResultType::Tweet)]; // rank 0
        let semantic = vec![make_semantic_hit("A", 0.9, "tweet")]; // rank 0

        let fused = rrf_fuse(&lexical, &semantic, 10, 0);

        // RRF score = 1/(60+0+1) + 1/(60+0+1) = 2/61 ≈ 0.0328
        let expected_score = 2.0 / 61.0;
        assert!((fused[0].score - expected_score).abs() < 0.001);
    }

    #[test]
    fn test_rrf_single_source() {
        // Only lexical results
        let lexical = vec![
            make_lexical_hit("A", 10.0, SearchResultType::Tweet),
            make_lexical_hit("B", 8.0, SearchResultType::Tweet),
        ];
        let semantic: Vec<VectorSearchResult> = vec![];

        let fused = rrf_fuse(&lexical, &semantic, 10, 0);

        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].doc_id, "A");
        assert!(!fused[0].in_both);

        // Score should be 1/(60+0+1) = 1/61
        let expected_score = 1.0 / 61.0;
        assert!((fused[0].score - expected_score).abs() < 0.001);
    }

    #[test]
    fn test_rrf_limit() {
        let lexical = vec![
            make_lexical_hit("A", 10.0, SearchResultType::Tweet),
            make_lexical_hit("B", 8.0, SearchResultType::Tweet),
            make_lexical_hit("C", 6.0, SearchResultType::Tweet),
        ];
        let semantic: Vec<VectorSearchResult> = vec![];

        let fused = rrf_fuse(&lexical, &semantic, 2, 0);

        assert_eq!(fused.len(), 2);
    }

    #[test]
    fn test_rrf_offset() {
        let lexical = vec![
            make_lexical_hit("A", 10.0, SearchResultType::Tweet),
            make_lexical_hit("B", 8.0, SearchResultType::Tweet),
            make_lexical_hit("C", 6.0, SearchResultType::Tweet),
        ];
        let semantic: Vec<VectorSearchResult> = vec![];

        let fused = rrf_fuse(&lexical, &semantic, 10, 1);

        assert_eq!(fused.len(), 2);
        assert_eq!(fused[0].doc_id, "B"); // A is skipped
    }

    #[test]
    fn test_rrf_empty() {
        let lexical: Vec<SearchResult> = vec![];
        let semantic: Vec<VectorSearchResult> = vec![];

        let fused = rrf_fuse(&lexical, &semantic, 10, 0);

        assert!(fused.is_empty());
    }

    #[test]
    fn test_rrf_zero_limit() {
        let lexical = vec![make_lexical_hit("A", 10.0, SearchResultType::Tweet)];
        let semantic = vec![make_semantic_hit("A", 0.9, "tweet")];

        let fused = rrf_fuse(&lexical, &semantic, 0, 0);

        assert!(fused.is_empty());
    }

    #[test]
    fn test_rrf_deterministic() {
        let lexical = vec![
            make_lexical_hit("A", 5.0, SearchResultType::Tweet),
            make_lexical_hit("B", 5.0, SearchResultType::Tweet),
            make_lexical_hit("C", 5.0, SearchResultType::Tweet),
        ];
        let semantic: Vec<VectorSearchResult> = vec![];

        // Run multiple times
        let fused1 = rrf_fuse(&lexical, &semantic, 10, 0);
        let fused2 = rrf_fuse(&lexical, &semantic, 10, 0);
        let fused3 = rrf_fuse(&lexical, &semantic, 10, 0);

        // Order should be identical
        for i in 0..3 {
            assert_eq!(fused1[i].doc_id, fused2[i].doc_id);
            assert_eq!(fused2[i].doc_id, fused3[i].doc_id);
        }
    }

    #[test]
    fn test_rrf_both_bonus() {
        // Same RRF score, but "both" should rank higher
        let lexical = vec![
            make_lexical_hit("solo_lex", 10.0, SearchResultType::Tweet), // rank 0
            make_lexical_hit("both", 5.0, SearchResultType::Tweet),      // rank 1
        ];
        let semantic = vec![
            make_semantic_hit("solo_sem", 0.9, "tweet"), // rank 0
            make_semantic_hit("both", 0.5, "tweet"),     // rank 1
        ];

        let fused = rrf_fuse(&lexical, &semantic, 10, 0);

        // "both" should be first because it appears in both lists
        // and has combined score from both (even though individual ranks are lower)
        assert_eq!(fused[0].doc_id, "both");
        assert!(fused[0].in_both);
    }

    #[test]
    fn test_search_mode_parsing() {
        assert_eq!(
            "lexical".parse::<SearchMode>().unwrap(),
            SearchMode::Lexical
        );
        assert_eq!(
            "keyword".parse::<SearchMode>().unwrap(),
            SearchMode::Lexical
        );
        assert_eq!(
            "semantic".parse::<SearchMode>().unwrap(),
            SearchMode::Semantic
        );
        assert_eq!(
            "vector".parse::<SearchMode>().unwrap(),
            SearchMode::Semantic
        );
        assert_eq!("hybrid".parse::<SearchMode>().unwrap(), SearchMode::Hybrid);
        assert_eq!("rrf".parse::<SearchMode>().unwrap(), SearchMode::Hybrid);
        assert!("invalid".parse::<SearchMode>().is_err());
    }

    #[test]
    fn test_candidate_count() {
        assert_eq!(candidate_count(10, 0), 30); // 10 * 3
        assert_eq!(candidate_count(10, 5), 45); // (10 + 5) * 3
        assert_eq!(candidate_count(0, 0), 0);
    }

    #[test]
    fn test_rrf_separates_types_with_same_id() {
        let lexical = vec![
            make_lexical_hit("42", 10.0, SearchResultType::Tweet),
            make_lexical_hit("42", 9.0, SearchResultType::Like),
        ];
        let semantic = vec![make_semantic_hit("42", 0.8, "like")];

        let fused = rrf_fuse(&lexical, &semantic, 10, 0);
        let matching: Vec<_> = fused.iter().filter(|hit| hit.doc_id == "42").collect();

        assert_eq!(matching.len(), 2);
        assert!(matching.iter().any(|hit| hit.doc_type == "tweet"));
        assert!(matching.iter().any(|hit| hit.doc_type == "like"));
    }

    #[test]
    fn test_rrf_isomorphic_legacy_randomized() {
        let mut seed = 42u64;

        for _case in 0..25 {
            let lexical_len = (next_u32(&mut seed) % 10 + 1) as usize;
            let semantic_len = (next_u32(&mut seed) % 10) as usize;
            let limit = (next_u32(&mut seed) % 10 + 1) as usize;
            let offset = (next_u32(&mut seed) % 3) as usize;

            let mut lexical = Vec::with_capacity(lexical_len);
            for i in 0..lexical_len {
                let id = format!("doc{}", next_u32(&mut seed) % 7);
                let (result_type, _) = doc_type_for(next_u32(&mut seed));
                let score = usize_to_f32(lexical_len.saturating_sub(i) + 1);
                lexical.push(make_lexical_hit(&id, score, result_type));
            }

            let mut semantic = Vec::with_capacity(semantic_len);
            for i in 0..semantic_len {
                let id = format!("doc{}", next_u32(&mut seed) % 7);
                let (_, doc_type) = doc_type_for(next_u32(&mut seed));
                let score = usize_to_f32(semantic_len.saturating_sub(i) + 1) / 10.0;
                semantic.push(make_semantic_hit(&id, score, doc_type));
            }

            let fused = rrf_fuse(&lexical, &semantic, limit, offset);
            let legacy = legacy_rrf_fuse(&lexical, &semantic, limit, offset);

            assert_eq!(fused.len(), legacy.len());
            for (new, old) in fused.iter().zip(legacy.iter()) {
                assert_eq!(new.doc_id, old.doc_id);
                assert_eq!(new.doc_type, old.doc_type);
                assert_eq!(new.in_both, old.in_both);
                assert_eq!(new.lexical_rank, old.lexical_rank);
                assert_eq!(new.score.to_bits(), old.score.to_bits());
            }
        }
    }

    #[test]
    fn test_rrf_borrows_doc_id_from_lexical() {
        let lexical = vec![make_lexical_hit("borrowed", 1.0, SearchResultType::Tweet)];
        let semantic: Vec<VectorSearchResult> = vec![];

        let fused = rrf_fuse(&lexical, &semantic, 10, 0);
        let fused_ptr = fused[0].doc_id.as_ptr();
        let lexical_ptr = lexical[0].id.as_ptr();
        assert_eq!(fused_ptr, lexical_ptr);
    }

    #[test]
    fn test_rrf_borrows_doc_id_from_semantic() {
        let lexical: Vec<SearchResult> = vec![];
        let semantic = vec![make_semantic_hit("borrowed", 0.9, "tweet")];

        let fused = rrf_fuse(&lexical, &semantic, 10, 0);
        let fused_ptr = fused[0].doc_id.as_ptr();
        let semantic_ptr = semantic[0].doc_id.as_ptr();
        assert_eq!(fused_ptr, semantic_ptr);
    }

    // =========================================================================
    // TwoTier search mode tests
    // =========================================================================

    #[test]
    fn test_search_mode_two_tier_parsing() {
        assert_eq!(
            "two-tier".parse::<SearchMode>().unwrap(),
            SearchMode::TwoTier
        );
        assert_eq!(
            "twotier".parse::<SearchMode>().unwrap(),
            SearchMode::TwoTier
        );
        assert_eq!(
            "two_tier".parse::<SearchMode>().unwrap(),
            SearchMode::TwoTier
        );
        assert_eq!(
            "progressive".parse::<SearchMode>().unwrap(),
            SearchMode::TwoTier
        );
        assert_eq!("2tier".parse::<SearchMode>().unwrap(), SearchMode::TwoTier);
    }

    #[test]
    fn test_search_mode_two_tier_display() {
        assert_eq!(SearchMode::TwoTier.to_string(), "two-tier");
    }

    #[test]
    fn test_min_max_normalize_basic() {
        let mut results = vec![
            make_semantic_hit("a", 0.2, "tweet"),
            make_semantic_hit("b", 0.8, "tweet"),
            make_semantic_hit("c", 0.5, "tweet"),
        ];
        min_max_normalize(&mut results);

        // min=0.2, max=0.8, range=0.6
        // a: (0.2-0.2)/0.6 = 0.0
        // b: (0.8-0.2)/0.6 = 1.0
        // c: (0.5-0.2)/0.6 = 0.5
        assert!((results[0].score - 0.0).abs() < 0.001);
        assert!((results[1].score - 1.0).abs() < 0.001);
        assert!((results[2].score - 0.5).abs() < 0.001);
    }

    #[test]
    fn test_min_max_normalize_all_same() {
        let mut results = vec![
            make_semantic_hit("a", 0.5, "tweet"),
            make_semantic_hit("b", 0.5, "tweet"),
        ];
        min_max_normalize(&mut results);

        // All same → all 1.0
        assert!((results[0].score - 1.0).abs() < f32::EPSILON);
        assert!((results[1].score - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_min_max_normalize_empty() {
        let mut results: Vec<VectorSearchResult> = vec![];
        min_max_normalize(&mut results); // Should not panic
        assert!(results.is_empty());
    }

    #[test]
    fn test_blend_two_tier_all_fast() {
        let fast = vec![
            make_semantic_hit("a", 1.0, "tweet"),
            make_semantic_hit("b", 0.5, "tweet"),
        ];
        let quality: Vec<VectorSearchResult> = vec![];

        // blend_factor=0 means fast-only
        let blended = blend_two_tier(&fast, &quality, 0.0);
        assert_eq!(blended.len(), 2);
        assert_eq!(blended[0].doc_id, "a");
        assert!((blended[0].score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_blend_two_tier_all_quality() {
        let fast: Vec<VectorSearchResult> = vec![];
        let quality = vec![
            make_semantic_hit("a", 1.0, "tweet"),
            make_semantic_hit("b", 0.5, "tweet"),
        ];

        // blend_factor=1.0 means quality-only
        let blended = blend_two_tier(&fast, &quality, 1.0);
        assert_eq!(blended.len(), 2);
        assert_eq!(blended[0].doc_id, "a");
        assert!((blended[0].score - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_blend_two_tier_overlap() {
        let fast = vec![
            make_semantic_hit("a", 0.9, "tweet"),
            make_semantic_hit("b", 0.5, "tweet"),
        ];
        let quality = vec![
            make_semantic_hit("a", 0.8, "tweet"),
            make_semantic_hit("c", 0.7, "tweet"),
        ];

        // blend_factor=0.5 → 50% fast + 50% quality
        let blended = blend_two_tier(&fast, &quality, 0.5);
        assert_eq!(blended.len(), 3); // a, b, c

        // "a" should be first: 0.9*0.5 + 0.8*0.5 = 0.85
        let a = blended.iter().find(|h| h.doc_id == "a").unwrap();
        assert!((a.score - 0.85).abs() < 0.001);

        // "c" quality-only: 0.7*0.5 = 0.35
        let c = blended.iter().find(|h| h.doc_id == "c").unwrap();
        assert!((c.score - 0.35).abs() < 0.001);

        // "b" fast-only: 0.5*0.5 = 0.25
        let b = blended.iter().find(|h| h.doc_id == "b").unwrap();
        assert!((b.score - 0.25).abs() < 0.001);
    }

    #[test]
    fn test_blend_two_tier_disjoint() {
        let fast = vec![make_semantic_hit("a", 1.0, "tweet")];
        let quality = vec![make_semantic_hit("b", 1.0, "tweet")];

        let blended = blend_two_tier(&fast, &quality, 0.7);
        assert_eq!(blended.len(), 2);

        let a = blended.iter().find(|h| h.doc_id == "a").unwrap();
        assert!((a.score - 0.3).abs() < 0.001); // 1.0 * 0.3

        let b = blended.iter().find(|h| h.doc_id == "b").unwrap();
        assert!((b.score - 0.7).abs() < 0.001); // 1.0 * 0.7
    }
}
