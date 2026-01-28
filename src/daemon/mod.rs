//! Model daemon for keeping embedding and reranking models warm in memory.
//!
//! The daemon accepts requests over a Unix Domain Socket using MessagePack
//! protocol, providing fast warm model inference for the CLI.
//!
//! # Architecture
//!
//! ```text
//! CLI Command
//!     ↓
//! DaemonClient (connects to UDS)
//!     ↓
//! ModelDaemon (accept loop)
//!     ↓
//! ModelManager (lazy load, LRU eviction)
//!     ↓
//! Embedder/Reranker backends
//! ```
//!
//! # Protocol
//!
//! Messages are length-prefixed MessagePack envelopes:
//! - 4-byte big-endian length
//! - MessagePack-encoded Envelope { version, id, payload }
//! - Payload contains Request or Response
//!
//! # Usage
//!
//! ```ignore
//! use xf::daemon::{DaemonConfig, ModelDaemon};
//!
//! let config = DaemonConfig::default()
//!     .with_idle_timeout(Duration::from_secs(300))
//!     .with_max_models(4);
//!
//! let daemon = ModelDaemon::new(config);
//! daemon.run().await?;
//! ```

mod client;
mod core;
mod models;
mod protocol;
mod resource;

pub use client::{ClientConfig, DaemonClient, HealthInfo, InferenceMode, StatusInfo};
pub use core::{DaemonConfig, ModelDaemon};
pub use models::ModelManager;
pub use protocol::{Envelope, LoadedModelInfo, PROTOCOL_VERSION, Request, Response, error_codes};
pub use resource::{IoPriority, MemoryMonitor, ResourceConfig, apply_resource_settings, get_process_rss_mb};
