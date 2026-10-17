# xf Architecture

This document describes the internal architecture of xf, the ultra-fast X data archive search tool.

## System Overview

xf uses a hybrid storage strategy combining a full-text search engine with a relational database:

```
┌─────────────────────────────────────────────────────────────────┐
│                        X Data Archive                            │
│                    (ZIP or extracted folder)                     │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Archive Parser                              │
│                       (parser.rs)                                │
│  • Parses JavaScript-wrapped JSON files                         │
│  • Normalizes data into Rust structs                            │
│  • Parallel processing with rayon                               │
└─────────────────────────────┬───────────────────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│    Tantivy Search       │     │    SQLite Storage       │
│     (search.rs)         │     │     (storage.rs)        │
├─────────────────────────┤     ├─────────────────────────┤
│ • Full-text indexing    │     │ • Metadata storage      │
│ • BM25 ranking          │     │ • FTS5 fallback search  │
│ • Sub-ms queries        │     │ • Statistics queries    │
│ • Prefix matching       │     │ • Structured queries    │
└─────────────────────────┘     └─────────────────────────┘
              │                               │
              └───────────────┬───────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                         CLI Layer                                │
│                        (main.rs)                                 │
│  • Command parsing (clap)                                       │
│  • Output formatting                                            │
│  • Progress indicators                                          │
└─────────────────────────────────────────────────────────────────┘
```

## Data Flow

### Indexing Flow

```
1. User runs: xf index ~/twitter-archive

2. Archive Parser:
   ├── Parse manifest.js → ArchiveInfo
   ├── Parse tweets.js → Vec<Tweet>
   ├── Parse like.js → Vec<Like>
   ├── Parse direct-messages.js → Vec<DmConversation>
   ├── Parse grok-chat-item.js → Vec<GrokMessage>
   └── Parse follower.js, following.js, etc.

3. Storage Layer:
   ├── Create SQLite database
   ├── Store all records with indexes
   └── Build FTS5 virtual tables

4. Search Engine:
   ├── Create Tantivy index
   ├── Index text fields with BM25
   └── Generate prefix n-grams
```

### Search Flow

```
1. User runs: xf search "rust programming"

2. Query Parsing:
   ├── Parse boolean operators (AND, OR, NOT)
   ├── Handle phrase queries ("exact match")
   └── Apply type filters (--types tweet,dm)

3. Tantivy Search:
   ├── Execute query against index
   ├── Rank results by BM25 score
   └── Return top N document IDs

4. Result Enrichment:
   ├── Fetch full records from SQLite
   └── Format output (text, JSON, CSV)
```

## Module Details

### parser.rs - Archive Parser

The parser handles X's unique data export format:

```javascript
// X export format (tweets.js):
window.YTD.tweets.part0 = [
  {"tweet": {"id": "123", "full_text": "Hello world", ...}},
  ...
]
```

Key responsibilities:
- Strip JavaScript wrapper to extract JSON
- Parse date formats (X uses multiple formats)
- Handle missing/optional fields gracefully
- Parallel parsing with rayon for large archives

### search.rs - Tantivy Search Engine

Schema design:

| Field | Type | Purpose |
|-------|------|---------|
| id | STRING, STORED | Unique document identifier |
| text | TEXT, STORED | Main searchable content |
| text_prefix | TEXT | Edge n-grams for prefix matching |
| type | STRING, STORED | Document type (tweet, like, dm, grok) |
| created_at | I64, INDEXED, FAST | Timestamp for sorting/filtering |
| metadata | TEXT, STORED | JSON blob for extra data |

Key features:
- BM25 ranking for relevance
- Phrase queries with position indexing
- Boolean query support (AND, OR, NOT)
- Sub-millisecond query latency

### storage.rs - SQLite Storage

Table structure (simplified):

```sql
-- Main tweets table
CREATE TABLE tweets (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    full_text TEXT NOT NULL,
    source TEXT,
    favorite_count INTEGER,
    retweet_count INTEGER,
    -- ... more fields
);

-- FTS5 virtual table for fallback search
CREATE VIRTUAL TABLE fts_tweets USING fts5(
    tweet_id,
    full_text,
    content='tweets',
    content_rowid='rowid'
);
```

Key features:
- WAL mode for concurrent reads
- Prepared statements for performance
- FTS5 for backup search capability
- Efficient batch inserts

### vector.rs - Persistent Mmap Vector Search + File Format

The persistent vector index (`vector.idx`, `vector.fast.idx`, `vector.quality.idx`) uses a compact, versioned binary format (`XFVI`) with memory-mapped zero-copy reads and zero-allocation SIMD (`wide::f32x8`) dot-product scoring over IEEE 754 `f16` embeddings.

#### Binary Layout (Little-Endian)

Header (fixed 32 bytes):

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0 | 4 | magic | `XFVI` |
| 4 | 2 | version | `1` |
| 6 | 1 | doc_type_encoding | `0` = enum |
| 7 | 1 | reserved | `0` |
| 8 | 4 | dimension | Embedding dimension |
| 12 | 8 | record_count | Number of records |
| 20 | 8 | offsets_start | Byte offset to offsets table |
| 28 | 4 | reserved | `0` |

Offsets table:

```
u64 offsets[record_count]  // absolute byte offsets to each record
```

Record layout (variable length):

| Field | Size | Notes |
|-------|------|-------|
| doc_type | 1 | enum (`0` tweet, `1` like, `2` dm, `3` grok, `4` bookmark) |
| reserved | 1 | `0` |
| doc_id_len | 2 | u16 byte length |
| doc_id | N | UTF-8 bytes |
| embedding | 2*dimension | raw f16 bytes, little-endian |

#### Validation Rules

- Magic must match `XFVI`.
- Version must be supported.
- `dimension > 0`.
- `offsets_start` must be >= header length.
- Offsets table must fit in file.
- Record offsets must be sorted, in-bounds, and point into data section.
- Each record must fit within file bounds and have valid UTF-8 `doc_id`.
- `doc_type` must be in the enum range (`0..=4`).

### hybrid.rs, reranker.rs & daemon/ - Fusion, Reranking & Model Server

- **Hybrid RRF Fusion (`hybrid.rs`)**: Combines lexical BM25 rankings with semantic vector rankings via Reciprocal Rank Fusion (`K = 60`), plus two-tier progressive search (`--mode two-tier`) blending fast static embeddings with quality transformer embeddings.
- **Cross-Encoder Reranking (`reranker.rs`, `flashrank_reranker.rs`, `mxbai_reranker.rs`, `rerank_step.rs`)**: Optional post-retrieval reranking (`--rerank`, `--reranker`, `--rerank-top`) with deterministic zero-dependency `hash-reranker` fallback when ONNX models are not downloaded.
- **Daemon Mode (`daemon/`)**: Background Unix domain socket server speaking length-prefixed MessagePack (`rmp_serde`) to keep embedders and rerankers warm across CLI invocations and execute background embedding jobs.

## Design Decisions

### Why Both Tantivy and SQLite?

| Capability | Tantivy | SQLite |
|------------|---------|--------|
| Full-text search | Excellent | Good (FTS5) |
| BM25 ranking | Native | Manual |
| Query latency | <1ms | 5-50ms |
| Structured queries | Limited | Excellent |
| Storage efficiency | Moderate | Excellent |
| Metadata storage | JSON blob | Native tables |

The hybrid approach gives us:
- **Tantivy** for blazing-fast text search
- **SQLite** for structured queries, statistics, and automatic FTS5 fallback when the Tantivy index directory is absent

### Why Parse JavaScript Files?

X exports data as JavaScript files (`window.YTD.* = [...]`) rather than
pure JSON. This is likely for browser-based viewing in their archive
viewer. We strip the JS wrapper and parse the underlying JSON.

### Why Parallel Parsing?

Large archives can have hundreds of MB of data. Parallel parsing with
rayon provides 3-4x speedup on typical multi-core systems.

## Performance Characteristics

| Operation | Target | Typical |
|-----------|--------|---------|
| Simple search | <1ms | 0.3ms |
| Phrase search | <5ms | 1-2ms |
| Index 10K tweets | <1s | 500ms |
| Index 100K tweets | <10s | 5s |
| Storage lookup | <1ms | 0.1ms |
| Statistics query | <10ms | 5ms |

## Future Considerations

1. **Remote Search**: Optional local HTTP API mode
2. **Multiple Archives**: Unified search across multiple X accounts
3. **Additional Export Formats**: Markdown and HTML archive reports
