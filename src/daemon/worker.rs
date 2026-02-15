//! Background embedding worker for the daemon.
//!
//! Handles embedding jobs submitted to the daemon, processing them in the background
//! with progress tracking and cancellation support.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use tokio::sync::mpsc;
use tracing::{error, info, warn};

use crate::storage::Storage;

/// Configuration for an embedding job.
#[derive(Debug, Clone)]
pub struct EmbeddingJobConfig {
    /// Path to the SQLite database.
    pub db_path: PathBuf,
    /// Path to the index directory.
    pub index_path: PathBuf,
    /// Whether to use two-tier embedding.
    pub two_tier: bool,
    /// Fast tier model name.
    pub fast_model: Option<String>,
    /// Quality tier model name.
    pub quality_model: Option<String>,
}

/// Message types for the worker channel.
#[derive(Debug)]
pub enum WorkerMessage {
    /// Submit a new embedding job.
    Submit(EmbeddingJobConfig),
    /// Cancel jobs for a database path.
    Cancel {
        db_path: PathBuf,
        model_id: Option<String>,
    },
    /// Shutdown the worker.
    Shutdown,
}

/// Handle to communicate with the embedding worker.
#[derive(Clone)]
pub struct EmbeddingWorkerHandle {
    sender: mpsc::Sender<WorkerMessage>,
}

impl EmbeddingWorkerHandle {
    /// Submit a new embedding job.
    pub async fn submit(
        &self,
        config: EmbeddingJobConfig,
    ) -> Result<(), mpsc::error::SendError<WorkerMessage>> {
        self.sender.send(WorkerMessage::Submit(config)).await
    }

    /// Cancel jobs for a database path.
    pub async fn cancel(
        &self,
        db_path: PathBuf,
        model_id: Option<String>,
    ) -> Result<(), mpsc::error::SendError<WorkerMessage>> {
        self.sender
            .send(WorkerMessage::Cancel { db_path, model_id })
            .await
    }

    /// Request worker shutdown.
    pub async fn shutdown(&self) -> Result<(), mpsc::error::SendError<WorkerMessage>> {
        self.sender.send(WorkerMessage::Shutdown).await
    }
}

/// Background embedding worker.
pub struct EmbeddingWorker {
    receiver: mpsc::Receiver<WorkerMessage>,
    /// Flag to signal cancellation of the current job.
    cancel_flag: Arc<AtomicBool>,
}

impl EmbeddingWorker {
    /// Create a new worker and its handle.
    #[must_use]
    pub fn new() -> (Self, EmbeddingWorkerHandle) {
        let (sender, receiver) = mpsc::channel(32);
        let cancel_flag = Arc::new(AtomicBool::new(false));

        let worker = Self {
            receiver,
            cancel_flag,
        };
        let handle = EmbeddingWorkerHandle { sender };

        (worker, handle)
    }

    /// Run the worker loop.
    pub async fn run(mut self) {
        info!("Embedding worker started");

        // Resume any pending jobs on startup
        if let Err(e) = self.resume_pending_jobs().await {
            error!("Failed to resume pending jobs: {}", e);
        }

        while let Some(msg) = self.receiver.recv().await {
            match msg {
                WorkerMessage::Submit(config) => {
                    self.cancel_flag.store(false, Ordering::SeqCst);
                    if let Err(e) = self.process_job(config).await {
                        error!("Embedding job failed: {}", e);
                    }
                }
                WorkerMessage::Cancel { db_path, model_id } => {
                    self.cancel_flag.store(true, Ordering::SeqCst);
                    if let Err(e) = self.cancel_jobs(&db_path, model_id.as_deref()).await {
                        error!("Failed to cancel jobs: {}", e);
                    }
                }
                WorkerMessage::Shutdown => {
                    info!("Embedding worker shutting down");
                    break;
                }
            }
        }

        info!("Embedding worker stopped");
    }

    /// Resume pending jobs from the database.
    #[allow(clippy::unused_async)]
    async fn resume_pending_jobs(&self) -> anyhow::Result<()> {
        // We need a database to check for pending jobs
        // For now, skip this - the daemon will track jobs per-database
        // and resume them when those databases are accessed
        Ok(())
    }

    /// Cancel jobs for a database path.
    #[allow(clippy::unused_async)]
    async fn cancel_jobs(&self, db_path: &Path, model_id: Option<&str>) -> anyhow::Result<()> {
        let storage = Storage::open(db_path)?;
        let db_path_str = db_path.to_string_lossy();
        let cancelled = storage.cancel_embedding_jobs(&db_path_str, model_id)?;
        info!("Cancelled {} embedding jobs for {}", cancelled, db_path_str);
        Ok(())
    }

    /// Process a single embedding job.
    #[allow(clippy::unused_async)]
    async fn process_job(&self, config: EmbeddingJobConfig) -> anyhow::Result<()> {
        let db_path_str = config.db_path.to_string_lossy().to_string();
        let start = Instant::now();

        info!("Starting embedding job for {}", db_path_str);

        // Open the database (fresh connection for counting)
        let storage = Storage::open(&config.db_path)?;

        // Count total documents
        let total_docs = self.count_documents(&storage)?;
        drop(storage); // Release connection

        if total_docs == 0 {
            info!("No documents to embed for {}", db_path_str);
            return Ok(());
        }

        // Determine which passes to run
        let passes: Vec<(&str, String, bool)> = if config.two_tier {
            let fast = config
                .fast_model
                .clone()
                .unwrap_or_else(|| "hash-fnv1a-384".to_string());
            let quality = config
                .quality_model
                .clone()
                .unwrap_or_else(|| "all-MiniLM-L6-v2".to_string());
            vec![("fast", fast, false), ("quality", quality, true)]
        } else {
            // Single-pass with default hash embedder
            vec![("default", "hash-fnv1a-384".to_string(), false)]
        };

        for (model_id, model_name, use_semantic) in passes {
            if self.cancel_flag.load(Ordering::SeqCst) {
                warn!("Embedding job cancelled for {}", db_path_str);
                return Ok(());
            }

            // Open fresh storage connection for this pass
            let storage = Storage::open(&config.db_path)?;

            // Create job in database (may already exist from submit)
            let job_id = storage.upsert_embedding_job(&db_path_str, model_id, total_docs)?;
            storage.start_embedding_job(job_id)?;

            info!(
                "Processing {} pass (model: {}) for {}",
                model_id, model_name, db_path_str
            );

            // Generate embeddings (sync operation)
            let db_path = config.db_path.clone();
            let model_id_owned = model_id.to_string();
            let model_name_owned = model_name.clone();

            match self.generate_embeddings_with_progress_sync(
                &db_path,
                &model_id_owned,
                &model_name_owned,
                use_semantic,
                job_id,
            ) {
                Ok(()) => {
                    // Re-open for completion update
                    let storage = Storage::open(&config.db_path)?;
                    storage.complete_embedding_job(job_id)?;
                    info!("{} pass completed for {}", model_id, db_path_str);
                }
                Err(e) => {
                    let err_msg = e.to_string();
                    // Re-open for failure update
                    if let Ok(storage) = Storage::open(&config.db_path) {
                        let _ = storage.fail_embedding_job(job_id, &err_msg);
                    }
                    error!("{} pass failed for {}: {}", model_id, db_path_str, err_msg);
                    return Err(e);
                }
            }
        }

        // Write vector indices
        self.write_vector_indices_sync(&config)?;

        let elapsed = start.elapsed();
        info!(
            "Embedding job completed for {} in {:.1}s",
            db_path_str,
            elapsed.as_secs_f64()
        );

        Ok(())
    }

    /// Count total embeddable documents.
    #[allow(clippy::unused_self)]
    fn count_documents(&self, storage: &Storage) -> anyhow::Result<i64> {
        let tweets = storage.get_all_tweets(None)?.len();
        let likes = storage
            .get_all_likes(None)?
            .iter()
            .filter(|l| l.full_text.as_ref().is_some_and(|t| !t.is_empty()))
            .count();
        let dms = storage
            .get_all_dms(None)?
            .iter()
            .filter(|d| !d.text.is_empty())
            .count();
        let grok = storage
            .get_all_grok_messages(None)?
            .iter()
            .filter(|m| !m.message.is_empty())
            .count();

        #[allow(clippy::cast_possible_wrap)]
        Ok((tweets + likes + dms + grok) as i64)
    }

    /// Generate embeddings with progress tracking.
    ///
    /// Opens its own Storage connection to avoid Send/Sync issues.
    #[allow(clippy::missing_const_for_fn, clippy::too_many_lines)]
    fn generate_embeddings_with_progress_sync(
        &self,
        db_path: &Path,
        model_id: &str,
        model_name: &str,
        use_semantic: bool,
        job_id: i64,
    ) -> anyhow::Result<()> {
        use crate::canonicalize::{canonicalize_for_embedding, content_hash};
        use crate::embedder::Embedder;
        use crate::hash_embedder::HashEmbedder;
        use crate::model_registry::{EmbedderConfig as RegistryConfig, ModelRegistry};
        use rayon::prelude::*;
        use std::collections::{HashMap, HashSet};

        type EmbedRecord = (String, String, Vec<f32>, Option<[u8; 32]>);
        const EMBED_CHUNK_SIZE: usize = 1000;
        const PROGRESS_UPDATE_INTERVAL: usize = 100;

        let storage = Storage::open(db_path)?;

        // Create the embedder
        let registry = ModelRegistry::new();
        let mut cfg = RegistryConfig::new(model_name);
        cfg.show_progress = false;
        let embedder_box: Box<dyn Embedder> = if use_semantic {
            registry.embedder(&cfg)?
        } else {
            // Try to get the model from registry, fall back to hash embedder
            match registry.embedder(&cfg) {
                Ok(e) => e,
                Err(_) => Box::new(HashEmbedder::default()),
            }
        };
        let embedder: &dyn Embedder = embedder_box.as_ref();
        let store_batch_size = if embedder.is_semantic() { 50 } else { 100 };

        // Fetch all documents
        let tweets = storage.get_all_tweets(None)?;
        let likes = storage.get_all_likes(None)?;
        let dms = storage.get_all_dms(None)?;
        let grok_msgs = storage.get_all_grok_messages(None)?;

        let capacity = tweets.len() + likes.len() + dms.len() + grok_msgs.len();
        let mut docs: Vec<(String, String, &'static str)> = Vec::with_capacity(capacity);

        for tweet in &tweets {
            docs.push((tweet.id.clone(), tweet.full_text.clone(), "tweet"));
        }
        for like in &likes {
            if let Some(ref text) = like.full_text {
                if !text.is_empty() {
                    docs.push((like.tweet_id.clone(), text.clone(), "like"));
                }
            }
        }
        for dm in &dms {
            if !dm.text.is_empty() {
                docs.push((dm.id.clone(), dm.text.clone(), "dm"));
            }
        }
        for msg in &grok_msgs {
            if !msg.message.is_empty() {
                let doc_id = format!(
                    "{}_{}_{}_{}",
                    msg.chat_id,
                    msg.created_at.timestamp(),
                    msg.created_at.timestamp_subsec_nanos(),
                    msg.sender
                );
                docs.push((doc_id, msg.message.clone(), "grok"));
            }
        }

        if docs.is_empty() {
            return Ok(());
        }

        // Load existing hashes for this model
        let existing_hashes_by_doc = storage.load_embedding_hashes_by_doc_for_model(model_id)?;
        let mut existing_hashes: HashSet<[u8; 32]> = HashSet::new();
        for by_type in existing_hashes_by_doc.values() {
            for hash_val in by_type.values() {
                existing_hashes.insert(*hash_val);
            }
        }

        let mut completed_docs: i64 = 0;
        let mut last_progress_update = 0;

        for chunk in docs.chunks(EMBED_CHUNK_SIZE) {
            // Check cancellation
            if self.cancel_flag.load(Ordering::SeqCst) {
                return Err(anyhow::anyhow!("Job cancelled"));
            }

            let mut batch: Vec<EmbedRecord> = Vec::new();
            let mut candidates: Vec<(String, &'static str, String, [u8; 32])> = Vec::new();

            for (doc_id, text, doc_type) in chunk {
                let canonical = canonicalize_for_embedding(text);
                if canonical.is_empty() {
                    completed_docs += 1;
                    continue;
                }
                let hash = content_hash(&canonical);
                if let Some(existing_hash) = existing_hashes_by_doc
                    .get(doc_id)
                    .and_then(|by_type| by_type.get(*doc_type))
                {
                    if existing_hash == &hash {
                        completed_docs += 1;
                        continue;
                    }
                }
                candidates.push((doc_id.clone(), *doc_type, canonical, hash));
            }

            // Load existing embeddings for reuse
            let mut batch_cache: HashMap<[u8; 32], Vec<f32>> = HashMap::new();
            let mut needed_hashes: Vec<[u8; 32]> = Vec::new();
            let mut needed_hashes_set: HashSet<[u8; 32]> = HashSet::new();
            for (_, _, _, hash) in &candidates {
                if existing_hashes.contains(hash) && needed_hashes_set.insert(*hash) {
                    needed_hashes.push(*hash);
                }
            }

            if !needed_hashes.is_empty() {
                let fetched =
                    storage.load_embeddings_by_hashes_for_model(&needed_hashes, model_id)?;
                for (hash, embedding) in fetched {
                    batch_cache.insert(hash, embedding);
                }
            }

            // Find hashes that need computation
            let mut new_hashes: Vec<(String, String, [u8; 32])> = Vec::new();
            let mut new_hashes_set: HashSet<[u8; 32]> = HashSet::new();
            for (doc_id, _doc_type, canonical, hash) in &candidates {
                if batch_cache.contains_key(hash) {
                    continue;
                }
                if new_hashes_set.insert(*hash) {
                    new_hashes.push((doc_id.clone(), canonical.clone(), *hash));
                }
            }

            // Compute new embeddings
            if !new_hashes.is_empty() {
                let computed_embeddings: Vec<([u8; 32], Vec<f32>)> = new_hashes
                    .par_iter()
                    .filter_map(
                        |(doc_id, canonical, hash)| match embedder.embed(canonical) {
                            Ok(embedding) => Some((*hash, embedding)),
                            Err(e) => {
                                warn!("Failed to embed doc {}: {}", doc_id, e);
                                None
                            }
                        },
                    )
                    .collect();

                for (hash, embedding) in computed_embeddings {
                    batch_cache.insert(hash, embedding);
                }
            }

            // Build batch for storage
            let mut seen_new_hashes: HashSet<[u8; 32]> = HashSet::new();
            for (doc_id, doc_type, _canonical, hash) in candidates {
                if let Some(existing_embedding) = batch_cache.get(&hash) {
                    batch.push((
                        doc_id,
                        doc_type.to_string(),
                        existing_embedding.clone(),
                        Some(hash),
                    ));
                    if !existing_hashes.contains(&hash) {
                        seen_new_hashes.insert(hash);
                    }
                    // Only count as completed if embedding was successfully obtained
                    completed_docs += 1;
                }
                // Note: documents with failed embeddings are intentionally not counted
                // as completed - they will be retried on next job run
            }

            // Store batch
            if !batch.is_empty() {
                for store_chunk in batch.chunks(store_batch_size) {
                    storage.store_embeddings_batch_with_model(store_chunk, model_id)?;
                }
            }

            // Update existing hashes set
            for hash in seen_new_hashes {
                existing_hashes.insert(hash);
            }

            // Update progress periodically
            #[allow(clippy::cast_possible_wrap)]
            if completed_docs - last_progress_update >= PROGRESS_UPDATE_INTERVAL as i64 {
                storage.update_job_progress(job_id, completed_docs)?;
                last_progress_update = completed_docs;
            }
        }

        // Final progress update
        storage.update_job_progress(job_id, completed_docs)?;

        Ok(())
    }

    /// Write vector indices after embedding generation (sync version).
    #[allow(clippy::unused_self)]
    fn write_vector_indices_sync(&self, config: &EmbeddingJobConfig) -> anyhow::Result<()> {
        use crate::vector::{
            VECTOR_INDEX_FAST_FILENAME, VECTOR_INDEX_QUALITY_FILENAME, write_vector_index,
            write_vector_index_named,
        };

        let index_path = &config.index_path;
        let storage = Storage::open(&config.db_path)?;

        // Always write the default vector index (all embeddings)
        write_vector_index(index_path, &storage)?;
        info!("Wrote vector index to {}", index_path.display());

        if config.two_tier {
            // Also write named tier indices for two-tier search
            write_vector_index_named(
                index_path,
                &storage,
                VECTOR_INDEX_FAST_FILENAME,
                Some("fast"),
            )?;
            info!(
                "Wrote fast vector index to {}",
                index_path.join(VECTOR_INDEX_FAST_FILENAME).display()
            );

            write_vector_index_named(
                index_path,
                &storage,
                VECTOR_INDEX_QUALITY_FILENAME,
                Some("quality"),
            )?;
            info!(
                "Wrote quality vector index to {}",
                index_path.join(VECTOR_INDEX_QUALITY_FILENAME).display()
            );
        }

        Ok(())
    }
}

impl Default for EmbeddingWorker {
    fn default() -> Self {
        Self::new().0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_worker_handle_clone() {
        let (_worker, handle) = EmbeddingWorker::new();
        let _handle2 = handle;
    }

    #[test]
    fn test_job_config() {
        let config = EmbeddingJobConfig {
            db_path: PathBuf::from("/path/to/db"),
            index_path: PathBuf::from("/path/to/index"),
            two_tier: true,
            fast_model: Some("hash".to_string()),
            quality_model: Some("miniLM".to_string()),
        };
        assert!(config.two_tier);
        assert_eq!(config.fast_model, Some("hash".to_string()));
    }
}
