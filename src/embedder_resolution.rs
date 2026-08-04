//! Search-time resolution of the default query embedder.
//!
//! `xf index` records which embedder produced the stored document embeddings
//! (meta key `embedding_model.<model_id>`). At search time, when the user has
//! not explicitly named a model (via `--model`, `XF_MODEL`, or the config
//! file), the query embedder MUST live in the same vector space as the stored
//! document embeddings — otherwise cosine similarity degenerates to noise.
//!
//! Historically the index path loaded MiniLM through the model registry
//! (`fastembed`, which caches under `.fastembed_cache/`) while the search path
//! probed xf-specific model directories that `xf index --semantic` never
//! populates. The result: `xf index --semantic` succeeded, then `xf search
//! --mode semantic` silently fell back to the hash embedder and scored
//! nDCG@10 = 0 (GitHub issue #9). This module fixes that by resolving the
//! query embedder from what the index was actually built with:
//!
//! 1. The embedder recorded at index time (authoritative, written by every
//!    indexing pass since this module was introduced).
//! 2. For legacy databases without the recorded key: an xf-layout model
//!    directory if one exists (preserves previous behavior), otherwise the
//!    stored embeddings themselves are inspected — hash embeddings of short
//!    texts are mostly exact zeros, transformer embeddings are fully dense.
//! 3. The hash embedder, as an explicit, *loud* last resort.

use crate::embedder::Embedder;
use crate::fastembed_embedder::FastEmbedder;
use crate::hash_embedder::HashEmbedder;
use crate::model_registry::{
    EMBEDDER_HASH_FNV1A_384, EMBEDDER_MINILM_L6_V2, EmbedderConfig, ModelRegistry,
};
use crate::storage::Storage;

/// Model id used for single-pass (non two-tier) embeddings.
pub const DEFAULT_MODEL_ID: &str = "default";

/// Fraction of exactly-zero components above which a stored embedding is
/// considered hash-based. FNV-1a feature hashing leaves most of its 384
/// buckets untouched for short texts (tweets/DMs), while transformer
/// embeddings (MiniLM et al.) are fully dense — even after f16 quantization
/// exact zeros are vanishingly rare.
pub const HASH_ZERO_FRACTION_THRESHOLD: f32 = 0.25;

/// How the default query embedder should be obtained.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueryEmbedderPlan {
    /// Load this registry model — the same loader the index path uses.
    LoadModel {
        model: String,
        source: PlanSource,
    },
    /// Use the hash embedder: the index itself was built with hash
    /// embeddings, so the hash embedder is the *correct* query embedder.
    HashIndex {
        /// Whether this was recorded at index time (vs inferred).
        recorded: bool,
    },
    /// A legacy xf-layout model directory exists; load MiniLM from it.
    LegacyXfLayout,
    /// Stored embeddings look semantic but the model cannot be determined
    /// (unknown dimension, nothing recorded). Only loud hash fallback is
    /// possible.
    UnknownSemanticIndex { dimension: usize },
}

/// Where a [`QueryEmbedderPlan::LoadModel`] decision came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanSource {
    /// The embedder model recorded in the database at index time.
    RecordedMeta,
    /// Inferred from the stored embeddings of a legacy database.
    InferredFromStoredEmbeddings,
}

/// Outcome of resolving the default query embedder.
pub struct ResolvedQueryEmbedder {
    /// The embedder to use for query embedding.
    pub embedder: Box<dyn Embedder>,
    /// Human-readable description of how (and why) it was chosen.
    pub detail: String,
    /// True when a semantic embedder was expected (the index holds semantic
    /// embeddings) but the hash embedder had to be used instead. Results
    /// will be severely degraded and callers MUST surface this loudly.
    pub degraded: bool,
    /// True when the index itself was built with hash embeddings (so hash
    /// query embedding is correct, but no true semantic search is possible).
    pub hash_index: bool,
}

impl std::fmt::Debug for ResolvedQueryEmbedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ResolvedQueryEmbedder")
            .field("embedder_id", &self.embedder.id())
            .field("detail", &self.detail)
            .field("degraded", &self.degraded)
            .field("hash_index", &self.hash_index)
            .finish()
    }
}

/// Heuristic: does a stored embedding look like the FNV-1a hash embedder's
/// output (mostly exact zeros) rather than a dense transformer embedding?
#[must_use]
pub fn looks_like_hash_embedding(embedding: &[f32]) -> bool {
    if embedding.is_empty() {
        return true;
    }
    #[allow(clippy::cast_precision_loss)]
    let zero_fraction =
        embedding.iter().filter(|v| **v == 0.0).count() as f32 / embedding.len() as f32;
    zero_fraction > HASH_ZERO_FRACTION_THRESHOLD
}

/// Decide how the default query embedder should be obtained, without loading
/// any model.
///
/// `legacy_xf_layout_present` reports whether an xf-layout MiniLM directory
/// exists (see [`FastEmbedder::try_load`]); it is injected so the decision
/// logic stays deterministic and testable.
#[must_use]
pub fn plan_query_embedder(storage: &Storage, legacy_xf_layout_present: bool) -> QueryEmbedderPlan {
    // 1. Authoritative: the embedder recorded when the index was built.
    if let Ok(Some(recorded)) = storage.get_embedding_model(DEFAULT_MODEL_ID) {
        let canonical = ModelRegistry::canonical_name(&recorded);
        if canonical == Some(EMBEDDER_HASH_FNV1A_384) {
            return QueryEmbedderPlan::HashIndex { recorded: true };
        }
        return QueryEmbedderPlan::LoadModel {
            model: recorded,
            source: PlanSource::RecordedMeta,
        };
    }

    // 2. Legacy xf-layout model directory (preserves pre-existing behavior
    //    for users who provisioned models manually).
    if legacy_xf_layout_present {
        return QueryEmbedderPlan::LegacyXfLayout;
    }

    // 3. Legacy database: infer the embedder family from the embeddings
    //    themselves.
    if let Ok(Some(sample)) = storage.sample_embedding(DEFAULT_MODEL_ID) {
        if !looks_like_hash_embedding(&sample) {
            if sample.len() == 384 {
                // Dense 384-dim vectors from the default pipeline can only
                // come from `xf index --semantic` (MiniLM).
                return QueryEmbedderPlan::LoadModel {
                    model: EMBEDDER_MINILM_L6_V2.to_string(),
                    source: PlanSource::InferredFromStoredEmbeddings,
                };
            }
            return QueryEmbedderPlan::UnknownSemanticIndex {
                dimension: sample.len(),
            };
        }
    }

    QueryEmbedderPlan::HashIndex { recorded: false }
}

/// Resolve the default query embedder for search.
///
/// Never fails: on any error the hash embedder is returned with `degraded`
/// set so callers can warn loudly instead of silently returning noise.
#[must_use]
pub fn resolve_query_embedder(storage: &Storage) -> ResolvedQueryEmbedder {
    let legacy_present = FastEmbedder::is_available();
    let plan = plan_query_embedder(storage, legacy_present);
    resolve_plan(storage, plan)
}

fn resolve_plan(storage: &Storage, plan: QueryEmbedderPlan) -> ResolvedQueryEmbedder {
    match plan {
        QueryEmbedderPlan::LoadModel { model, source } => {
            let registry = ModelRegistry::new();
            let mut cfg = EmbedderConfig::new(&model);
            cfg.show_progress = false;
            match registry.embedder(&cfg) {
                Ok(embedder) => {
                    let how = match source {
                        PlanSource::RecordedMeta => "recorded at index time",
                        PlanSource::InferredFromStoredEmbeddings => {
                            "inferred from stored embeddings"
                        }
                    };
                    ResolvedQueryEmbedder {
                        embedder,
                        detail: format!("using semantic model '{model}' ({how})"),
                        degraded: false,
                        hash_index: false,
                    }
                }
                Err(e) => ResolvedQueryEmbedder {
                    embedder: Box::new(HashEmbedder::default()),
                    detail: format!(
                        "the index was built with semantic model '{model}' but it could not \
                         be loaded: {e}"
                    ),
                    degraded: true,
                    hash_index: false,
                },
            }
        }
        QueryEmbedderPlan::LegacyXfLayout => match FastEmbedder::try_load() {
            Ok(embedder) => ResolvedQueryEmbedder {
                embedder: Box::new(embedder),
                detail: "using MiniLM from xf model directory".to_string(),
                degraded: false,
                hash_index: false,
            },
            // The directory probe passed but the actual load failed; fall
            // back to the remaining plan steps (meta was already absent).
            Err(_) => resolve_plan(storage, plan_query_embedder(storage, false)),
        },
        QueryEmbedderPlan::HashIndex { recorded } => ResolvedQueryEmbedder {
            embedder: Box::new(HashEmbedder::default()),
            detail: if recorded {
                "index was built with hash embeddings (recorded at index time)".to_string()
            } else {
                "index appears to contain hash embeddings".to_string()
            },
            degraded: false,
            hash_index: true,
        },
        QueryEmbedderPlan::UnknownSemanticIndex { dimension } => ResolvedQueryEmbedder {
            embedder: Box::new(HashEmbedder::default()),
            detail: format!(
                "the index holds dense {dimension}-dim semantic embeddings but the model that \
                 produced them is unknown; pass --model (or set XF_MODEL) to the model used at \
                 index time"
            ),
            degraded: true,
            hash_index: false,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embedder::Embedder as _;

    fn temp_storage() -> (tempfile::TempDir, Storage) {
        let dir = tempfile::tempdir().expect("create tempdir");
        let storage = Storage::open(dir.path().join("test.db")).expect("open storage");
        (dir, storage)
    }

    #[test]
    fn test_hash_embedding_detected_as_hash() {
        let embedder = HashEmbedder::default();
        let embedding = embedder.embed("tips for haggling with overseas factories").unwrap();
        assert!(
            looks_like_hash_embedding(&embedding),
            "hash embedder output should be recognized as hash-based"
        );
    }

    #[test]
    fn test_dense_embedding_detected_as_semantic() {
        // Simulate a transformer embedding: fully dense unit vector.
        let mut dense = vec![0.03f32; 384];
        dense[0] = -0.5;
        assert!(!looks_like_hash_embedding(&dense));
    }

    #[test]
    fn test_empty_embedding_treated_as_hash() {
        assert!(looks_like_hash_embedding(&[]));
    }

    #[test]
    fn test_plan_prefers_recorded_semantic_model() {
        let (_dir, storage) = temp_storage();
        storage
            .set_embedding_model(DEFAULT_MODEL_ID, EMBEDDER_MINILM_L6_V2)
            .unwrap();
        let plan = plan_query_embedder(&storage, false);
        assert_eq!(
            plan,
            QueryEmbedderPlan::LoadModel {
                model: EMBEDDER_MINILM_L6_V2.to_string(),
                source: PlanSource::RecordedMeta,
            }
        );
    }

    #[test]
    fn test_plan_recorded_meta_wins_over_legacy_layout() {
        let (_dir, storage) = temp_storage();
        storage
            .set_embedding_model(DEFAULT_MODEL_ID, EMBEDDER_HASH_FNV1A_384)
            .unwrap();
        let plan = plan_query_embedder(&storage, true);
        assert_eq!(plan, QueryEmbedderPlan::HashIndex { recorded: true });
    }

    #[test]
    fn test_plan_recorded_hash_model() {
        let (_dir, storage) = temp_storage();
        storage
            .set_embedding_model(DEFAULT_MODEL_ID, EMBEDDER_HASH_FNV1A_384)
            .unwrap();
        let plan = plan_query_embedder(&storage, false);
        assert_eq!(plan, QueryEmbedderPlan::HashIndex { recorded: true });
    }

    #[test]
    fn test_plan_legacy_layout_when_no_meta() {
        let (_dir, storage) = temp_storage();
        let plan = plan_query_embedder(&storage, true);
        assert_eq!(plan, QueryEmbedderPlan::LegacyXfLayout);
    }

    #[test]
    fn test_plan_infers_semantic_from_dense_stored_embedding() {
        // Regression test for issue #9: a database indexed by `xf index
        // --semantic` (before the model was recorded in meta) must resolve to
        // the MiniLM loader, not the hash embedder.
        let (_dir, storage) = temp_storage();
        let dense = vec![0.05f32; 384];
        storage
            .store_embedding("doc1", "tweet", &dense, None)
            .unwrap();
        let plan = plan_query_embedder(&storage, false);
        assert_eq!(
            plan,
            QueryEmbedderPlan::LoadModel {
                model: EMBEDDER_MINILM_L6_V2.to_string(),
                source: PlanSource::InferredFromStoredEmbeddings,
            }
        );
    }

    #[test]
    fn test_plan_infers_hash_from_sparse_stored_embedding() {
        let (_dir, storage) = temp_storage();
        let embedder = HashEmbedder::default();
        let embedding = embedder.embed("a short tweet about rust").unwrap();
        storage
            .store_embedding("doc1", "tweet", &embedding, None)
            .unwrap();
        let plan = plan_query_embedder(&storage, false);
        assert_eq!(plan, QueryEmbedderPlan::HashIndex { recorded: false });
    }

    #[test]
    fn test_plan_unknown_dense_dimension() {
        let (_dir, storage) = temp_storage();
        let dense = vec![0.05f32; 768];
        storage
            .store_embedding("doc1", "tweet", &dense, None)
            .unwrap();
        let plan = plan_query_embedder(&storage, false);
        assert_eq!(plan, QueryEmbedderPlan::UnknownSemanticIndex { dimension: 768 });
    }

    #[test]
    fn test_plan_empty_database_defaults_to_hash() {
        let (_dir, storage) = temp_storage();
        let plan = plan_query_embedder(&storage, false);
        assert_eq!(plan, QueryEmbedderPlan::HashIndex { recorded: false });
    }

    #[test]
    fn test_resolve_unloadable_model_is_loud_degraded_hash_fallback() {
        // Regression test for issue #9 part 2: when a semantic model was
        // expected but cannot be loaded, the fallback must be flagged as
        // degraded (never silent) and the detail must name the model.
        let (_dir, storage) = temp_storage();
        // `embeddinggemma-300m` is registered but its backend is not
        // implemented, so loading fails deterministically without network.
        storage
            .set_embedding_model(DEFAULT_MODEL_ID, "embeddinggemma-300m")
            .unwrap();
        let resolved = resolve_query_embedder(&storage);
        assert!(resolved.degraded, "fallback must be marked degraded");
        assert!(!resolved.hash_index);
        assert_eq!(resolved.embedder.id(), "fnv1a-384");
        assert!(
            resolved.detail.contains("embeddinggemma-300m"),
            "detail should name the failing model: {}",
            resolved.detail
        );
    }

    #[test]
    fn test_resolve_hash_index_is_not_degraded() {
        let (_dir, storage) = temp_storage();
        storage
            .set_embedding_model(DEFAULT_MODEL_ID, EMBEDDER_HASH_FNV1A_384)
            .unwrap();
        let resolved = resolve_query_embedder(&storage);
        assert!(!resolved.degraded);
        assert!(resolved.hash_index);
        assert_eq!(resolved.embedder.id(), "fnv1a-384");
    }

    #[test]
    fn test_resolve_unknown_semantic_index_is_degraded() {
        let (_dir, storage) = temp_storage();
        let dense = vec![0.05f32; 768];
        storage
            .store_embedding("doc1", "tweet", &dense, None)
            .unwrap();
        let resolved = resolve_plan(&storage, plan_query_embedder(&storage, false));
        assert!(resolved.degraded);
        assert!(resolved.detail.contains("768"));
    }
}
