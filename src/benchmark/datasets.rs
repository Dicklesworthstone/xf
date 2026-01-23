//! Benchmark dataset definitions and loader.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Document in the benchmark corpus.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorpusDoc {
    pub id: String,
    pub text: String,
    #[serde(rename = "type")]
    pub doc_type: String,
    #[serde(default)]
    pub metadata: serde_json::Value,
}

/// Labeled query with graded relevance judgments.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabeledQuery {
    pub id: String,
    pub text: String,
    pub relevants: HashMap<String, u8>,
    #[serde(default)]
    pub category: Option<String>,
}

/// Full benchmark corpus (docs + queries).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkCorpus {
    pub corpus: Vec<CorpusDoc>,
    pub queries: Vec<LabeledQuery>,
}

impl BenchmarkCorpus {
    /// Load corpus from JSON.
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)?;
        let corpus: Self = serde_json::from_str(&raw)?;
        Ok(corpus)
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
        Ok(())
    }

    /// Total relevant documents for a query.
    #[must_use]
    pub fn total_relevant(&self, query: &LabeledQuery) -> usize {
        query.relevants.values().filter(|v| **v > 0).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_detects_duplicate_ids() {
        let corpus = BenchmarkCorpus {
            corpus: vec![
                CorpusDoc {
                    id: "doc1".into(),
                    text: "a".into(),
                    doc_type: "tweet".into(),
                    metadata: serde_json::Value::Null,
                },
                CorpusDoc {
                    id: "doc1".into(),
                    text: "b".into(),
                    doc_type: "tweet".into(),
                    metadata: serde_json::Value::Null,
                },
            ],
            queries: vec![],
        };
        assert!(corpus.validate().is_err());
    }
}
