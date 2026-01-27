//! Model manager for lazy loading and LRU eviction.
//!
//! Manages embedders and rerankers, loading them on first use and evicting
//! least-recently-used models when the maximum is exceeded.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use crate::embedder::{Embedder, EmbedderError};
use crate::model_registry::{EmbedderConfig, ModelRegistry, RerankerConfig};
use crate::reranker::{Reranker, RerankerError};

use super::protocol::LoadedModelInfo;

/// Entry for a loaded embedder.
struct EmbedderEntry {
    embedder: Box<dyn Embedder>,
    loaded_at: u64,
    last_used: Instant,
    requests_served: AtomicU64,
}

/// Entry for a loaded reranker.
struct RerankerEntry {
    reranker: Box<dyn Reranker>,
    loaded_at: u64,
    last_used: Instant,
    requests_served: AtomicU64,
}

/// Manages model loading, caching, and eviction.
pub struct ModelManager {
    embedders: HashMap<String, EmbedderEntry>,
    rerankers: HashMap<String, RerankerEntry>,
    registry: ModelRegistry,
    max_models: usize,
}

impl ModelManager {
    /// Create a new model manager.
    ///
    /// # Arguments
    /// * `max_models` - Maximum number of models to keep loaded. When exceeded,
    ///   the least recently used model is evicted.
    #[must_use]
    pub fn new(max_models: usize) -> Self {
        Self {
            embedders: HashMap::new(),
            rerankers: HashMap::new(),
            registry: ModelRegistry::new(),
            max_models,
        }
    }

    /// Get or load an embedder by name.
    ///
    /// Models are loaded on first access and cached. If loading would exceed
    /// `max_models`, the least recently used model is evicted first.
    ///
    /// # Panics
    ///
    /// This function uses `expect` internally but will not panic in practice
    /// because the HashMap accesses are immediately after confirmed insertions.
    pub fn get_embedder(&mut self, name: &str) -> Result<&dyn Embedder, EmbedderError> {
        // Update last_used if already loaded
        if let Some(entry) = self.embedders.get_mut(name) {
            entry.last_used = Instant::now();
            entry.requests_served.fetch_add(1, Ordering::Relaxed);
            // Safety: We know the entry exists because get_mut succeeded
            return Ok(self
                .embedders
                .get(name)
                .expect("entry exists")
                .embedder
                .as_ref());
        }

        // Evict if necessary
        if self.total_models() >= self.max_models {
            self.evict_lru();
        }

        // Load the embedder
        let config = EmbedderConfig::new(name);
        let embedder = self.registry.embedder(&config)?;

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());

        let entry = EmbedderEntry {
            embedder,
            loaded_at: now,
            last_used: Instant::now(),
            requests_served: AtomicU64::new(1),
        };

        self.embedders.insert(name.to_string(), entry);
        // Safety: We just inserted this entry
        Ok(self
            .embedders
            .get(name)
            .expect("just inserted")
            .embedder
            .as_ref())
    }

    /// Get or load a reranker by name.
    ///
    /// Returns `None` for the "none" reranker.
    ///
    /// # Panics
    ///
    /// This function uses `expect` internally but will not panic in practice
    /// because the HashMap accesses are immediately after confirmed insertions.
    pub fn get_reranker(&mut self, name: &str) -> Result<Option<&dyn Reranker>, RerankerError> {
        // Handle "none" case
        if name == "none" || name.is_empty() {
            return Ok(None);
        }

        // Update last_used if already loaded
        if let Some(entry) = self.rerankers.get_mut(name) {
            entry.last_used = Instant::now();
            entry.requests_served.fetch_add(1, Ordering::Relaxed);
            // Safety: We know the entry exists because get_mut succeeded
            return Ok(Some(
                self.rerankers
                    .get(name)
                    .expect("entry exists")
                    .reranker
                    .as_ref(),
            ));
        }

        // Evict if necessary
        if self.total_models() >= self.max_models {
            self.evict_lru();
        }

        // Load the reranker
        let config = RerankerConfig::new(name);
        let reranker = self.registry.reranker(&config)?;

        let Some(reranker) = reranker else {
            return Ok(None);
        };

        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());

        let entry = RerankerEntry {
            reranker,
            loaded_at: now,
            last_used: Instant::now(),
            requests_served: AtomicU64::new(1),
        };

        self.rerankers.insert(name.to_string(), entry);
        // Safety: We just inserted this entry
        Ok(Some(
            self.rerankers
                .get(name)
                .expect("just inserted")
                .reranker
                .as_ref(),
        ))
    }

    /// Total number of loaded models.
    #[must_use]
    pub fn total_models(&self) -> usize {
        self.embedders.len() + self.rerankers.len()
    }

    /// Get information about all loaded models.
    #[must_use]
    pub fn loaded_models(&self) -> Vec<LoadedModelInfo> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());

        let mut models = Vec::with_capacity(self.embedders.len() + self.rerankers.len());

        for (name, entry) in &self.embedders {
            models.push(LoadedModelInfo {
                name: name.clone(),
                model_type: "embedder".to_string(),
                loaded_at: entry.loaded_at,
                requests_served: entry.requests_served.load(Ordering::Relaxed),
                last_used: now.saturating_sub(entry.last_used.elapsed().as_secs()),
            });
        }

        for (name, entry) in &self.rerankers {
            models.push(LoadedModelInfo {
                name: name.clone(),
                model_type: "reranker".to_string(),
                loaded_at: entry.loaded_at,
                requests_served: entry.requests_served.load(Ordering::Relaxed),
                last_used: now.saturating_sub(entry.last_used.elapsed().as_secs()),
            });
        }

        models
    }

    /// Evict the least recently used model.
    fn evict_lru(&mut self) {
        // Find LRU embedder
        let lru_embedder = self
            .embedders
            .iter()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(k, e)| (k.clone(), e.last_used));

        // Find LRU reranker
        let lru_reranker = self
            .rerankers
            .iter()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(k, e)| (k.clone(), e.last_used));

        // Evict the older one
        match (lru_embedder, lru_reranker) {
            (Some((ek, et)), Some((rk, rt))) => {
                if et < rt {
                    tracing::info!(model = %ek, "evicting LRU embedder");
                    self.embedders.remove(&ek);
                } else {
                    tracing::info!(model = %rk, "evicting LRU reranker");
                    self.rerankers.remove(&rk);
                }
            }
            (Some((ek, _)), None) => {
                tracing::info!(model = %ek, "evicting LRU embedder");
                self.embedders.remove(&ek);
            }
            (None, Some((rk, _))) => {
                tracing::info!(model = %rk, "evicting LRU reranker");
                self.rerankers.remove(&rk);
            }
            (None, None) => {}
        }
    }

    /// Unload all models.
    pub fn unload_all(&mut self) {
        self.embedders.clear();
        self.rerankers.clear();
        tracing::info!("unloaded all models");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_manager_new() {
        let manager = ModelManager::new(5);
        assert_eq!(manager.total_models(), 0);
        assert_eq!(manager.max_models, 5);
    }

    #[test]
    fn test_get_hash_embedder() {
        let mut manager = ModelManager::new(5);

        // Hash embedder should always be available
        let result = manager.get_embedder("hash");
        assert!(result.is_ok());

        // Should now be loaded
        assert_eq!(manager.total_models(), 1);

        // Getting again should reuse cached
        let result2 = manager.get_embedder("hash");
        assert!(result2.is_ok());
        assert_eq!(manager.total_models(), 1);
    }

    #[test]
    fn test_loaded_models_info() {
        let mut manager = ModelManager::new(5);

        manager.get_embedder("hash").unwrap();

        let models = manager.loaded_models();
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].name, "hash");
        assert_eq!(models[0].model_type, "embedder");
        assert!(models[0].requests_served >= 1);
    }

    #[test]
    fn test_unknown_embedder() {
        let mut manager = ModelManager::new(5);
        let result = manager.get_embedder("unknown-model-xyz");
        assert!(result.is_err());
    }

    #[test]
    fn test_none_reranker() {
        let mut manager = ModelManager::new(5);
        let result = manager.get_reranker("none");
        assert!(result.is_ok());
        assert!(result.unwrap().is_none());
        assert_eq!(manager.total_models(), 0);
    }

    #[test]
    fn test_eviction() {
        let mut manager = ModelManager::new(1);

        // Load first embedder
        manager.get_embedder("hash").unwrap();
        assert_eq!(manager.total_models(), 1);

        // Wait a tiny bit so timestamps differ
        std::thread::sleep(std::time::Duration::from_millis(10));

        // Loading a second should evict the first
        // Note: This would fail if hash is the only available embedder
        // In practice, we'd need a second available model
        // For now, just verify eviction logic works
        manager.evict_lru();
        assert_eq!(manager.total_models(), 0);
    }

    #[test]
    fn test_unload_all() {
        let mut manager = ModelManager::new(5);

        manager.get_embedder("hash").unwrap();
        assert_eq!(manager.total_models(), 1);

        manager.unload_all();
        assert_eq!(manager.total_models(), 0);
    }
}
