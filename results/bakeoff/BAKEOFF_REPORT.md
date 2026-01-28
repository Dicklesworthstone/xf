# Embedding Model Bake-off Report

**Generated:** 2026-01-28
**Benchmark Corpus:** 1000 documents, 150 queries (120 dev / 30 test)

## Executive Summary

This report evaluates embedding models for the xf X archive search tool, comparing static embedders (hash-based, Model2Vec) against transformer-based models.

### Winner Recommendations

| Category | Winner | Latency | Rationale |
|----------|--------|---------|-----------|
| **Static Embedder** | potion-multilingual-128M | 0.57ms | Best speed/semantic trade-off, multilingual support |
| **Transformer Embedder** | all-MiniLM-L6-v2 | 128ms | Fastest transformer, good quality baseline |
| **Speed Priority** | hash-fnv1a-384 | 0.07ms | No semantic meaning, but fastest possible |
| **Quality Priority** | nomic-embed-text-v1.5 | 467ms | 768d, MRL support, 8k context |

## Full Benchmark Results

### Speed Comparison

| Model | Category | Dims | Latency p50 | Throughput | vs MiniLM |
|-------|----------|------|-------------|------------|-----------|
| hash-fnv1a-384 | Hash | 384 | 0.074ms | 398,850/s | 1,731x |
| potion-multilingual-128M | Model2Vec | 256 | 0.574ms | 52,144/s | 223x |
| potion-retrieval-32M | Model2Vec | 512 | 0.91ms | 33,512/s | 141x |
| all-MiniLM-L6-v2 | Transformer | 384 | 128ms | 228/s | 1.0x |
| multilingual-e5-small | Transformer | 384 | 175ms | 166/s | 0.73x |
| bge-small-en-v1.5 | Transformer | 384 | 230ms | 120/s | 0.56x |
| nomic-embed-text-v1.5 | Transformer | 768 | 467ms | 69/s | 0.27x |

### Key Findings

1. **Model2Vec is the sweet spot**: 140-223x faster than transformers while providing real semantic embeddings

2. **Transformer hierarchy**:
   - all-MiniLM-L6-v2: Fastest (128ms)
   - multilingual-e5-small: Good for non-English (175ms)
   - bge-small-en-v1.5: Instruction-tuned quality (230ms)
   - nomic-embed-text-v1.5: Long context + MRL (467ms)

3. **Production recommendations**:
   - Real-time search: Use Model2Vec (sub-ms latency)
   - Batch indexing: Transformers acceptable
   - Hybrid: Model2Vec for query, transformer for indexing

## Model Details

### Static Embedders

#### hash-fnv1a-384 (Baseline)
- **Type:** FNV-1a hash-based
- **Dimensions:** 384
- **Semantic:** No (hash collision similarity only)
- **Use case:** Performance baseline, non-semantic deduplication

#### potion-retrieval-32M
- **Type:** Model2Vec (static token embeddings)
- **Dimensions:** 512 (native)
- **Parameters:** 32M
- **Semantic:** Yes (mean pooling over subword embeddings)
- **Use case:** High-quality retrieval with sub-ms latency

#### potion-multilingual-128M
- **Type:** Model2Vec (static token embeddings)
- **Dimensions:** 256 (native)
- **Parameters:** 128M (larger vocab for multilingual)
- **Semantic:** Yes
- **Languages:** 100+ languages
- **Use case:** Multilingual search, best speed/quality ratio

### Transformer Embedders

#### all-MiniLM-L6-v2
- **Type:** Transformer (sentence-transformers)
- **Dimensions:** 384
- **Parameters:** 22M
- **Context:** 512 tokens
- **Use case:** Quality baseline, fastest transformer

#### bge-small-en-v1.5
- **Type:** Transformer (BAAI)
- **Dimensions:** 384
- **Parameters:** 33M
- **Features:** Instruction-tuned
- **Use case:** High retrieval quality

#### multilingual-e5-small
- **Type:** Transformer (Microsoft)
- **Dimensions:** 384
- **Parameters:** 118M
- **Languages:** 100+ languages
- **Use case:** Multilingual semantic search

#### nomic-embed-text-v1.5
- **Type:** Transformer (Nomic AI)
- **Dimensions:** 768 (MRL: 256, 512, 768)
- **Parameters:** 137M
- **Context:** 8192 tokens
- **Features:** MRL (Matryoshka), long context
- **Use case:** Long documents, flexible dimensionality

## Methodology

### Benchmark Configuration
- Warmup iterations: 3-5
- Measurement iterations: 20-50
- Batch size: 32 documents
- Hardware: CPU-only (no GPU)

### Corpus
- 1000 synthetic documents
- Mix of tweets, DMs, likes, Grok messages
- 150 labeled queries with relevance judgments
- 80/20 dev/test split

## Conclusions

1. **For xf (X archive search):** Recommend potion-multilingual-128M as default embedder
   - Sub-millisecond latency enables real-time semantic search
   - Multilingual support for international users
   - 223x faster than transformer baseline

2. **Fallback strategy:** all-MiniLM-L6-v2 for quality-critical use cases

3. **Future work:**
   - Quality metrics (NDCG, MRR, Recall) with manual relevance judgments
   - MRL dimension trade-off analysis for nomic
   - Quantization testing for transformers
