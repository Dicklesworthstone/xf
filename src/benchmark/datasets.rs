//! Benchmark dataset definitions, loader, and ground truth generation.
//!
//! This module provides data structures for the benchmark corpus and
//! automated ground truth generation via synthetic query + relevance grading.
//!
//! Key types:
//! - [`BenchmarkCorpus`]: Full corpus with docs, queries, splits, and stats
//! - [`LabeledQuery`]: Query with graded relevance judgments
//! - [`QueryType`], [`QueryDifficulty`]: Query classification enums
//!
//! Ground truth generation follows a 3-stage approach:
//! 1. Synthetic query generation from documents (rule-based, no LLM)
//! 2. Candidate pooling via BM25 + embedder
//! 3. Automated relevance grading via term overlap heuristics

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Query type for classification and balanced sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryType {
    /// Entity mention queries: "tweets mentioning @user"
    Entity,
    /// Temporal queries: "messages from March 2024"
    Temporal,
    /// Topical queries: "discussions about AI safety"
    Topical,
    /// Conversational queries: "DM conversations about project deadlines"
    Conversational,
    /// Factual queries: "tweets with links to arxiv papers"
    Factual,
}

impl fmt::Display for QueryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Entity => write!(f, "entity"),
            Self::Temporal => write!(f, "temporal"),
            Self::Topical => write!(f, "topical"),
            Self::Conversational => write!(f, "conversational"),
            Self::Factual => write!(f, "factual"),
        }
    }
}

/// Query difficulty for stratified sampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueryDifficulty {
    /// Easy: >=5 highly relevant docs, distinctive terms.
    Easy,
    /// Medium: 2-4 highly relevant docs, some ambiguity.
    Medium,
    /// Hard: 0-1 highly relevant docs, requires semantic understanding.
    Hard,
}

impl fmt::Display for QueryDifficulty {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Easy => write!(f, "easy"),
            Self::Medium => write!(f, "medium"),
            Self::Hard => write!(f, "hard"),
        }
    }
}

/// Assign difficulty based on number of highly relevant documents.
impl QueryDifficulty {
    #[must_use]
    pub const fn from_relevant_count(highly_relevant: usize) -> Self {
        match highly_relevant {
            5.. => Self::Easy,
            2..=4 => Self::Medium,
            _ => Self::Hard,
        }
    }
}

/// Cross-validation split assignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SplitAssignment {
    /// Development set for hyperparameter tuning.
    Dev,
    /// Test set for final evaluation.
    Test,
}

impl fmt::Display for SplitAssignment {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Dev => write!(f, "dev"),
            Self::Test => write!(f, "test"),
        }
    }
}

/// Document in the benchmark corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusDoc {
    /// Unique document identifier.
    pub id: String,
    /// Document text content.
    pub text: String,
    /// Document type: "tweet", "dm", "grok", "cass".
    #[serde(rename = "type")]
    pub doc_type: String,
    /// Optional metadata (created_at, author, etc.).
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// Labeled query with graded relevance judgments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabeledQuery {
    /// Unique query identifier.
    pub id: String,
    /// Query text.
    pub text: String,
    /// Relevance judgments: doc_id -> grade (2=highly, 1=partially, 0=not).
    pub relevants: HashMap<String, u8>,
    /// Query type for classification.
    #[serde(default)]
    pub query_type: Option<QueryType>,
    /// Query difficulty for stratified sampling.
    #[serde(default)]
    pub difficulty: Option<QueryDifficulty>,
    /// Source document ID (for synthetically generated queries).
    #[serde(default)]
    pub source_doc_id: Option<String>,
    /// Cross-validation split assignment.
    #[serde(default)]
    pub split: Option<SplitAssignment>,
    /// Legacy category field (deprecated, use query_type).
    #[serde(default)]
    pub category: Option<String>,
}

impl LabeledQuery {
    /// Count of highly relevant documents (grade == 2).
    #[must_use]
    pub fn highly_relevant_count(&self) -> usize {
        self.relevants.values().filter(|&&g| g == 2).count()
    }

    /// Count of all relevant documents (grade > 0).
    #[must_use]
    pub fn relevant_count(&self) -> usize {
        self.relevants.values().filter(|&&g| g > 0).count()
    }
}

/// Cross-validation splits metadata.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CrossValidationSplits {
    /// Query IDs in development set.
    pub dev: Vec<String>,
    /// Query IDs in test set.
    pub test: Vec<String>,
}

impl CrossValidationSplits {
    /// Check if a query is in the dev split.
    #[must_use]
    pub fn is_dev(&self, query_id: &str) -> bool {
        self.dev.iter().any(|id| id == query_id)
    }

    /// Check if a query is in the test split.
    #[must_use]
    pub fn is_test(&self, query_id: &str) -> bool {
        self.test.iter().any(|id| id == query_id)
    }
}

/// Corpus statistics for validation and reporting.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CorpusStats {
    /// Total number of documents.
    pub total_docs: usize,
    /// Total number of queries.
    pub total_queries: usize,
    /// Average relevant documents per query.
    pub avg_relevants_per_query: f64,
    /// Distribution by difficulty: easy, medium, hard.
    pub difficulty_distribution: HashMap<String, usize>,
    /// Distribution by query type.
    pub type_distribution: HashMap<String, usize>,
    /// Distribution by document type.
    pub doc_type_distribution: HashMap<String, usize>,
}

/// Full benchmark corpus (docs + queries + splits + metadata).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkCorpus {
    /// Corpus version identifier for reproducibility.
    #[serde(default)]
    pub corpus_version: Option<String>,
    /// Random seed used for generation.
    #[serde(default)]
    pub generation_seed: Option<u64>,
    /// Generation timestamp.
    #[serde(default)]
    pub generated_at: Option<String>,
    /// Document corpus.
    pub corpus: Vec<CorpusDoc>,
    /// Labeled queries with relevance judgments.
    pub queries: Vec<LabeledQuery>,
    /// Cross-validation splits.
    #[serde(default)]
    pub splits: CrossValidationSplits,
    /// Corpus statistics (computed on load).
    #[serde(default)]
    pub stats: CorpusStats,
}

impl BenchmarkCorpus {
    /// Load corpus from JSON.
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let mut corpus: Self = serde_json::from_str(&raw)?;
        corpus.compute_stats();
        Ok(corpus)
    }

    /// Save corpus to JSON.
    pub fn save(&self, path: &Path) -> Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Validate corpus for duplicates and relevance grades.
    pub fn validate(&self) -> Result<()> {
        let mut ids = HashSet::new();
        for doc in &self.corpus {
            if !ids.insert(doc.id.clone()) {
                anyhow::bail!("duplicate document id: {}", doc.id);
            }
        }

        for q in &self.queries {
            for (doc_id, grade) in &q.relevants {
                if *grade > 2 {
                    anyhow::bail!("invalid relevance grade {} for query {}", grade, q.id);
                }
                if !ids.contains(doc_id) {
                    anyhow::bail!("query {} references unknown doc id {}", q.id, doc_id);
                }
            }
        }

        // Validate splits cover all queries
        let split_ids: HashSet<_> = self
            .splits
            .dev
            .iter()
            .chain(self.splits.test.iter())
            .collect();
        for q in &self.queries {
            if !split_ids.contains(&q.id) && q.split.is_none() {
                anyhow::bail!("query {} not assigned to any split", q.id);
            }
        }

        Ok(())
    }

    /// Validate corpus with relaxed rules (for partial corpora).
    pub fn validate_relaxed(&self) -> Result<()> {
        let mut ids = HashSet::new();
        for doc in &self.corpus {
            if !ids.insert(doc.id.clone()) {
                anyhow::bail!("duplicate document id: {}", doc.id);
            }
        }

        for q in &self.queries {
            for (doc_id, grade) in &q.relevants {
                if *grade > 2 {
                    anyhow::bail!("invalid relevance grade {} for query {}", grade, q.id);
                }
                if !ids.contains(doc_id) {
                    anyhow::bail!("query {} references unknown doc id {}", q.id, doc_id);
                }
            }
        }
        Ok(())
    }

    /// Total relevant documents for a query.
    #[must_use]
    pub fn total_relevant(&self, query: &LabeledQuery) -> usize {
        query.relevants.values().filter(|v| **v > 0).count()
    }

    /// Compute and update corpus statistics.
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_stats(&mut self) {
        let total_docs = self.corpus.len();
        let total_queries = self.queries.len();

        let total_relevants: usize = self.queries.iter().map(LabeledQuery::relevant_count).sum();
        let avg_relevants_per_query = if total_queries > 0 {
            total_relevants as f64 / total_queries as f64
        } else {
            0.0
        };

        let mut difficulty_distribution: HashMap<String, usize> = HashMap::new();
        let mut type_distribution: HashMap<String, usize> = HashMap::new();

        for q in &self.queries {
            if let Some(diff) = &q.difficulty {
                *difficulty_distribution.entry(diff.to_string()).or_default() += 1;
            }
            if let Some(qtype) = &q.query_type {
                *type_distribution.entry(qtype.to_string()).or_default() += 1;
            }
        }

        let mut doc_type_distribution: HashMap<String, usize> = HashMap::new();
        for doc in &self.corpus {
            *doc_type_distribution
                .entry(doc.doc_type.clone())
                .or_default() += 1;
        }

        self.stats = CorpusStats {
            total_docs,
            total_queries,
            avg_relevants_per_query,
            difficulty_distribution,
            type_distribution,
            doc_type_distribution,
        };
    }

    /// Get queries by split assignment.
    #[must_use]
    pub fn queries_by_split(&self, split: SplitAssignment) -> Vec<&LabeledQuery> {
        self.queries
            .iter()
            .filter(|q| q.split == Some(split))
            .collect()
    }

    /// Get queries by type.
    #[must_use]
    pub fn queries_by_type(&self, query_type: QueryType) -> Vec<&LabeledQuery> {
        self.queries
            .iter()
            .filter(|q| q.query_type == Some(query_type))
            .collect()
    }

    /// Get queries by difficulty.
    #[must_use]
    pub fn queries_by_difficulty(&self, difficulty: QueryDifficulty) -> Vec<&LabeledQuery> {
        self.queries
            .iter()
            .filter(|q| q.difficulty == Some(difficulty))
            .collect()
    }

    /// Get document by ID.
    #[must_use]
    pub fn get_doc(&self, id: &str) -> Option<&CorpusDoc> {
        self.corpus.iter().find(|d| d.id == id)
    }

    /// Get query by ID.
    #[must_use]
    pub fn get_query(&self, id: &str) -> Option<&LabeledQuery> {
        self.queries.iter().find(|q| q.id == id)
    }
}

// ============================================================================
// Ground Truth Generation (Rule-Based, Deterministic)
// ============================================================================

/// Grade relevance using reproducible heuristics (no LLM dependency).
///
/// Relevance levels: 2=highly relevant, 1=partially relevant, 0=not relevant.
#[must_use]
#[allow(clippy::cast_precision_loss, clippy::bool_to_int_with_if)]
pub fn grade_relevance(query: &str, doc_text: &str) -> u8 {
    let query_lower = query.to_lowercase();
    let doc_lower = doc_text.to_lowercase();

    // Extract query terms (words > 2 chars, excluding common stop words)
    let stop_words: HashSet<&str> = [
        "the",
        "and",
        "for",
        "are",
        "but",
        "not",
        "you",
        "all",
        "can",
        "had",
        "her",
        "was",
        "one",
        "our",
        "out",
        "has",
        "have",
        "been",
        "this",
        "that",
        "with",
        "they",
        "from",
        "about",
        "what",
        "which",
        "when",
        "where",
        "who",
        "will",
        "would",
        "there",
        "their",
        "some",
        "into",
        "than",
        "then",
        "them",
        "these",
        "those",
        "only",
        "other",
        "more",
        "most",
        "just",
        "over",
        "such",
        "also",
        "back",
        "after",
        "tweets",
        "tweet",
        "messages",
        "message",
        "discussions",
        "about",
    ]
    .into_iter()
    .collect();

    let query_terms: HashSet<&str> = query_lower
        .split_whitespace()
        .filter(|w| w.len() > 2 && !stop_words.contains(w))
        .collect();

    if query_terms.is_empty() {
        return 0;
    }

    let doc_terms: HashSet<&str> = doc_lower.split_whitespace().collect();

    let overlap = query_terms.intersection(&doc_terms).count();
    let query_coverage = overlap as f64 / query_terms.len() as f64;

    // Check for exact phrase match (strong signal)
    let has_exact_phrase = query_terms.len() > 1 && {
        let phrase: Vec<_> = query_terms.iter().copied().collect();
        phrase.iter().all(|term| doc_lower.contains(term))
    };

    // Check for entity overlap (@ mentions, # hashtags)
    let entity_match = extract_entities(&query_lower)
        .iter()
        .any(|e| doc_lower.contains(e));

    if has_exact_phrase || (query_coverage > 0.8 && entity_match) {
        2 // Highly relevant
    } else if query_coverage > 0.4 || entity_match {
        1 // Partially relevant
    } else {
        0 // Not relevant
    }
}

/// Extract entities (@ mentions, # hashtags) from text.
#[must_use]
pub fn extract_entities(text: &str) -> Vec<String> {
    let mut entities = Vec::new();

    for word in text.split_whitespace() {
        let trimmed = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '@' && c != '#');
        if trimmed.starts_with('@') || trimmed.starts_with('#') {
            entities.push(trimmed.to_lowercase());
        }
    }

    entities
}

/// Extract key phrases from document text (simple n-gram approach).
#[must_use]
pub fn extract_keyphrases(text: &str, max_phrases: usize) -> Vec<String> {
    let words: Vec<&str> = text
        .split_whitespace()
        .filter(|w| w.len() > 3)
        .take(50) // Limit to first 50 words for efficiency
        .collect();

    let mut phrases = Vec::new();

    // Extract 2-grams and 3-grams
    for window in words.windows(2).take(max_phrases) {
        phrases.push(window.join(" ").to_lowercase());
    }

    if phrases.len() < max_phrases && words.len() >= 3 {
        for window in words.windows(3).take(max_phrases - phrases.len()) {
            phrases.push(window.join(" ").to_lowercase());
        }
    }

    phrases.truncate(max_phrases);
    phrases
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_doc(id: &str, text: &str, doc_type: &str) -> CorpusDoc {
        CorpusDoc {
            id: id.into(),
            text: text.into(),
            doc_type: doc_type.into(),
            metadata: serde_json::Value::Null,
        }
    }

    fn make_query(id: &str, text: &str, relevants: &[(&str, u8)]) -> LabeledQuery {
        LabeledQuery {
            id: id.into(),
            text: text.into(),
            relevants: relevants
                .iter()
                .map(|(k, v)| ((*k).to_string(), *v))
                .collect(),
            query_type: None,
            difficulty: None,
            source_doc_id: None,
            split: None,
            category: None,
        }
    }

    fn make_corpus(docs: Vec<CorpusDoc>, queries: Vec<LabeledQuery>) -> BenchmarkCorpus {
        BenchmarkCorpus {
            corpus_version: Some("test-v1".into()),
            generation_seed: Some(42),
            generated_at: None,
            corpus: docs,
            queries,
            splits: CrossValidationSplits::default(),
            stats: CorpusStats::default(),
        }
    }

    #[test]
    fn test_validate_detects_duplicate_ids() {
        let corpus = make_corpus(
            vec![
                make_doc("doc1", "a", "tweet"),
                make_doc("doc1", "b", "tweet"),
            ],
            vec![],
        );
        assert!(corpus.validate_relaxed().is_err());
    }

    #[test]
    fn test_validate_detects_invalid_relevance() {
        let corpus = make_corpus(
            vec![make_doc("doc1", "a", "tweet")],
            vec![make_query("q1", "test", &[("doc1", 3)])], // Invalid grade 3
        );
        assert!(corpus.validate_relaxed().is_err());
    }

    #[test]
    fn test_validate_detects_unknown_doc() {
        let corpus = make_corpus(
            vec![make_doc("doc1", "a", "tweet")],
            vec![make_query("q1", "test", &[("doc_unknown", 1)])],
        );
        assert!(corpus.validate_relaxed().is_err());
    }

    #[test]
    fn test_validate_passes_valid_corpus() {
        let corpus = make_corpus(
            vec![
                make_doc("doc1", "hello world", "tweet"),
                make_doc("doc2", "test document", "tweet"),
            ],
            vec![make_query("q1", "hello", &[("doc1", 2), ("doc2", 1)])],
        );
        assert!(corpus.validate_relaxed().is_ok());
    }

    #[test]
    fn test_compute_stats() {
        let mut corpus = make_corpus(
            vec![
                make_doc("doc1", "a", "tweet"),
                make_doc("doc2", "b", "dm"),
                make_doc("doc3", "c", "tweet"),
            ],
            vec![
                make_query("q1", "test", &[("doc1", 2), ("doc2", 1)]),
                make_query("q2", "test2", &[("doc3", 2)]),
            ],
        );
        corpus.compute_stats();

        assert_eq!(corpus.stats.total_docs, 3);
        assert_eq!(corpus.stats.total_queries, 2);
        assert!((corpus.stats.avg_relevants_per_query - 1.5).abs() < 0.01);
        assert_eq!(corpus.stats.doc_type_distribution.get("tweet"), Some(&2));
        assert_eq!(corpus.stats.doc_type_distribution.get("dm"), Some(&1));
    }

    #[test]
    fn test_query_relevant_count() {
        let q = make_query("q1", "test", &[("doc1", 2), ("doc2", 1), ("doc3", 0)]);
        assert_eq!(q.highly_relevant_count(), 1);
        assert_eq!(q.relevant_count(), 2);
    }

    #[test]
    fn test_difficulty_from_relevant_count() {
        assert_eq!(
            QueryDifficulty::from_relevant_count(5),
            QueryDifficulty::Easy
        );
        assert_eq!(
            QueryDifficulty::from_relevant_count(10),
            QueryDifficulty::Easy
        );
        assert_eq!(
            QueryDifficulty::from_relevant_count(3),
            QueryDifficulty::Medium
        );
        assert_eq!(
            QueryDifficulty::from_relevant_count(1),
            QueryDifficulty::Hard
        );
        assert_eq!(
            QueryDifficulty::from_relevant_count(0),
            QueryDifficulty::Hard
        );
    }

    // === Relevance grading tests ===

    #[test]
    fn test_grade_relevance_highly_relevant() {
        let query = "rust programming language";
        let doc = "I love the rust programming language for systems programming";
        assert_eq!(grade_relevance(query, doc), 2);
    }

    #[test]
    fn test_grade_relevance_partially_relevant() {
        let query = "rust programming language";
        let doc = "Python is a great programming language";
        assert!(grade_relevance(query, doc) <= 1);
    }

    #[test]
    fn test_grade_relevance_not_relevant() {
        let query = "rust programming language";
        let doc = "The weather is nice today";
        assert_eq!(grade_relevance(query, doc), 0);
    }

    #[test]
    fn test_grade_relevance_entity_match() {
        let query = "tweets mentioning @elonmusk";
        let doc = "Just saw @elonmusk tweet about SpaceX launch";
        let grade = grade_relevance(query, doc);
        assert!(
            grade >= 1,
            "Entity match should be at least partially relevant"
        );
    }

    #[test]
    fn test_grade_relevance_deterministic() {
        let query = "test query about AI";
        let doc = "This is a document about AI and machine learning";
        let grade1 = grade_relevance(query, doc);
        let grade2 = grade_relevance(query, doc);
        assert_eq!(grade1, grade2, "Grading must be deterministic");
    }

    // === Entity extraction tests ===

    #[test]
    fn test_extract_entities_mentions() {
        let text = "Hey @alice and @Bob, check this out!";
        let entities = extract_entities(text);
        assert!(entities.contains(&"@alice".to_string()));
        assert!(entities.contains(&"@bob".to_string()));
    }

    #[test]
    fn test_extract_entities_hashtags() {
        let text = "Love #RustLang and #Programming";
        let entities = extract_entities(text);
        assert!(entities.contains(&"#rustlang".to_string()));
        assert!(entities.contains(&"#programming".to_string()));
    }

    #[test]
    fn test_extract_entities_empty() {
        let text = "No entities here";
        assert!(extract_entities(text).is_empty());
    }

    // === Keyphrase extraction tests ===

    #[test]
    fn test_extract_keyphrases() {
        let text = "The quick brown fox jumps over the lazy dog";
        let phrases = extract_keyphrases(text, 3);
        assert!(!phrases.is_empty());
        assert!(phrases.len() <= 3);
    }

    #[test]
    fn test_extract_keyphrases_respects_limit() {
        let text = "one two three four five six seven eight nine ten";
        let phrases = extract_keyphrases(text, 2);
        assert!(phrases.len() <= 2);
    }

    // === Query type/difficulty display tests ===

    #[test]
    fn test_query_type_display() {
        assert_eq!(QueryType::Entity.to_string(), "entity");
        assert_eq!(QueryType::Temporal.to_string(), "temporal");
        assert_eq!(QueryType::Topical.to_string(), "topical");
        assert_eq!(QueryType::Conversational.to_string(), "conversational");
        assert_eq!(QueryType::Factual.to_string(), "factual");
    }

    #[test]
    fn test_query_difficulty_display() {
        assert_eq!(QueryDifficulty::Easy.to_string(), "easy");
        assert_eq!(QueryDifficulty::Medium.to_string(), "medium");
        assert_eq!(QueryDifficulty::Hard.to_string(), "hard");
    }

    #[test]
    fn test_split_assignment_display() {
        assert_eq!(SplitAssignment::Dev.to_string(), "dev");
        assert_eq!(SplitAssignment::Test.to_string(), "test");
    }

    // === Cross-validation splits tests ===

    #[test]
    fn test_splits_membership() {
        let splits = CrossValidationSplits {
            dev: vec!["q1".into(), "q2".into()],
            test: vec!["q3".into(), "q4".into()],
        };
        assert!(splits.is_dev("q1"));
        assert!(splits.is_dev("q2"));
        assert!(!splits.is_dev("q3"));
        assert!(splits.is_test("q3"));
        assert!(splits.is_test("q4"));
        assert!(!splits.is_test("q1"));
    }

    // === Corpus query methods tests ===

    #[test]
    fn test_get_doc() {
        let corpus = make_corpus(
            vec![
                make_doc("doc1", "hello", "tweet"),
                make_doc("doc2", "world", "dm"),
            ],
            vec![],
        );
        assert!(corpus.get_doc("doc1").is_some());
        assert!(corpus.get_doc("doc2").is_some());
        assert!(corpus.get_doc("doc3").is_none());
    }

    #[test]
    fn test_queries_by_type() {
        let mut q1 = make_query("q1", "entity query", &[]);
        q1.query_type = Some(QueryType::Entity);
        let mut q2 = make_query("q2", "topical query", &[]);
        q2.query_type = Some(QueryType::Topical);
        let mut q3 = make_query("q3", "another entity", &[]);
        q3.query_type = Some(QueryType::Entity);

        let corpus = make_corpus(vec![], vec![q1, q2, q3]);
        let entity_queries = corpus.queries_by_type(QueryType::Entity);
        assert_eq!(entity_queries.len(), 2);
    }

    #[test]
    fn test_queries_by_difficulty() {
        let mut q1 = make_query("q1", "easy", &[]);
        q1.difficulty = Some(QueryDifficulty::Easy);
        let mut q2 = make_query("q2", "hard", &[]);
        q2.difficulty = Some(QueryDifficulty::Hard);

        let corpus = make_corpus(vec![], vec![q1, q2]);
        assert_eq!(corpus.queries_by_difficulty(QueryDifficulty::Easy).len(), 1);
        assert_eq!(corpus.queries_by_difficulty(QueryDifficulty::Hard).len(), 1);
        assert_eq!(
            corpus.queries_by_difficulty(QueryDifficulty::Medium).len(),
            0
        );
    }
}
