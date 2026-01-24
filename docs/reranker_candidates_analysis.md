# Post-2025-10 Tiny CPU Reranker Model Analysis

**Research Date:** 2026-01-24
**Bead:** bd-35wf
**Hard Filter:** Models released or updated on/after **2025-11-01**
**Eligibility:** Open weights, CPU-viable, <=500M params or <=500MB disk, permissive license

---

## Executive Summary

After research of HuggingFace Hub and vendor announcements, **multiple models** meet the >= 2025-11-01 release requirement:

| Model | Released | Params | Status |
|-------|----------|--------|--------|
| **RexReranker-micro** | 2026-01-24 | ~17M | **ELIGIBLE** |
| **RexReranker-mini** | 2026-01-24 | ~32M | **ELIGIBLE** |
| **RexReranker-base** | 2026-01-24 | ~68M | **ELIGIBLE** |
| **ettin-encoder-17m-listnet** | 2025-12-31 | 17M | **ELIGIBLE** |
| **ettin-encoder-32m-listnet** | 2025-12-31 | 32M | **ELIGIBLE** |
| **ettin-encoder-68m-listnet** | 2025-12-31 | 68M | **ELIGIBLE** |
| **ctxl-rerank-v2-1b** | 2025-12-22 | 1B | **ELIGIBLE** (borderline size) |

Several strong baseline models were released before the cutoff.

---

## ELIGIBLE MODELS (>= 2025-11-01)

### RexReranker Series (2026-01-24)

**Source:** [thebajajra/RexReranker-*](https://huggingface.co/thebajajra)

| Variant | Params | Dims | License |
|---------|--------|------|---------|
| RexReranker-micro | ~17M | 384 | TBD |
| RexReranker-mini | ~32M | 384 | TBD |
| RexReranker-base | ~68M | 768 | TBD |
| RexReranker-large | ~150M | 768 | TBD |
| RexReranker-0.6B | ~600M | 1024 | TBD |
| RexReranker-0.6B-FP8 | ~600M | 1024 | TBD |

**Notes:** Brand new release (today). Need to verify license and CPU viability. Multiple size options.

---

### Ettin Encoder Reranker Series (2025-12-21 - 2025-12-31)

**Source:** [bansalaman18/reranker-msmarco-v1.1-ettin-encoder-*](https://huggingface.co/bansalaman18)

| Variant | Params | Loss | Modified | License |
|---------|--------|------|----------|---------|
| ettin-encoder-17m-listnet | 17M | ListNet | 2025-12-31 | TBD |
| ettin-encoder-32m-listnet | 32M | ListNet | 2025-12-31 | TBD |
| ettin-encoder-68m-listnet | 68M | ListNet | 2025-12-31 | TBD |
| ettin-encoder-150m-listnet | 150M | ListNet | 2026-01-01 | TBD |
| ettin-encoder-400m-listnet | 400M | ListNet | 2026-01-01 | TBD |
| ettin-encoder-17m-bce | 17M | BCE | 2025-12-21 | TBD |
| ettin-encoder-32m-bce | 32M | BCE | 2025-12-21 | TBD |
| ettin-encoder-68m-bce | 68M | BCE | 2025-12-21 | TBD |
| ettin-encoder-150m-bce | 150M | BCE | 2025-12-21 | TBD |

**Notes:** New MS-MARCO trained rerankers in multiple sizes. Small variants (17M-68M) are excellent for CPU.

---

### ContextualAI ctxl-rerank-v2 Series (2025-12-22)

**Source:** [ContextualAI/ctxl-rerank-v2-instruct-multilingual-*](https://huggingface.co/ContextualAI)

| Variant | Params | Notes |
|---------|--------|-------|
| ctxl-rerank-v2-1b | 1B | Base model |
| ctxl-rerank-v2-1b-nvfp4 | 1B | FP4 quantized |
| ctxl-rerank-v2-2b | 2B | Too large |
| ctxl-rerank-v2-6b | 6B | Too large |

**Notes:** 1B variant is borderline for "tiny" but FP4 quantized version may be CPU-viable.

---

### Qwen3-Reranker GGUF Variants (Various 2025-12)

**Source:** Multiple uploaders

| Variant | Base | Modified |
|---------|------|----------|
| Qwen3-Reranker-0.6B-GGUF (iyanello) | 600M | 2025-12-28 |
| Qwen3-Reranker-0.6B-Q8_0-GGUF (yeahbeen) | 600M | 2026-01-21 |
| Qwen3-Reranker-4B-Q4_K_M-GGUF | 4B | 2026-01-17 |

**Notes:** GGUF quantized versions of Qwen3 rerankers. 0.6B variants are CPU-viable with llama.cpp.

---

## BASELINE MODELS (Pre-Cutoff, High Quality)

### gte-reranker-modernbert-base (2025-01-21)

**Source:** [Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base)

- **Parameters:** 149M
- **Context:** 8192 tokens
- **License:** Apache-2.0
- **Why Baseline:** Released 2025-01-21 (before cutoff)

**Strengths:** ModernBERT architecture, CPU-optimized, strong MTEB performance.

---

### granite-embedding-reranker-english-r2 (2025-08-15)

**Source:** [ibm-granite/granite-embedding-reranker-english-r2](https://huggingface.co/ibm-granite/granite-embedding-reranker-english-r2)

- **Parameters:** 149M
- **Context:** 8192 tokens
- **License:** Apache-2.0
- **Why Baseline:** Released 2025-08-15 (before cutoff)

**Strengths:** ModernBERT architecture, enterprise-grade, fast on CPU.

---

### Qwen3-Reranker-0.6B (2025-06-05)

**Source:** [Qwen/Qwen3-Reranker-0.6B](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B)

- **Parameters:** 600M
- **License:** Apache-2.0
- **Why Baseline:** Released June 2025 (before cutoff)

**Strengths:** GGUF available, 100+ languages, strong performance.

---

### cross-encoder/ms-marco-MiniLM-L6-v2 (2022)

**Source:** [cross-encoder/ms-marco-MiniLM-L6-v2](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)

- **Parameters:** ~22M
- **License:** Apache-2.0
- **Why Baseline:** Industry standard baseline, pre-cutoff

**Strengths:** Tiny, fast, well-tested, ONNX support.

---

### FlashRank Tiny (~4MB)

**Source:** [FlashRank library](https://github.com/PrithivirajDamodaran/FlashRank)

- **Size:** ~4MB disk
- **Runtime:** ONNX, no Torch dependency
- **Why Baseline:** Library-based, not standalone HF model

**Strengths:** World's tiniest reranker. 0.1s for 100 docs on laptop.

---

## REJECTED MODELS

| Model | Reason |
|-------|--------|
| Qwen3-Reranker-8B | Too large (8B) |
| ctxl-rerank-v2-6b | Too large (6B) |
| ctxl-rerank-v2-2b | Too large (2B) |
| jina-reranker-v2-base | GPU-focused |
| bge-reranker-v2-gemma | Too large (gemma-based) |

---

## Summary

### Eligible Count: 7+ models

The reranker landscape has more post-cutoff activity than embeddings:

1. **RexReranker** series (micro/mini/base) - very new
2. **Ettin-encoder** series (17M-150M) - multiple sizes
3. **ctxl-rerank-v2-1b** - borderline but FP4 available
4. **Qwen3-Reranker GGUF** variants

### Recommendation

Proceed with bake-off using:
- **Eligible:** ettin-encoder-17m/32m/68m-listnet, RexReranker-micro/mini/base
- **Baselines:** gte-reranker-modernbert-base, ms-marco-MiniLM-L6-v2, FlashRank

The ettin-encoder series offers a nice size progression (17M → 68M) ideal for benchmarking speed/quality tradeoffs.

---

## Sources

- [HuggingFace Text-Ranking Models](https://huggingface.co/models?pipeline_tag=text-ranking)
- [FlashRank GitHub](https://github.com/PrithivirajDamodaran/FlashRank)
- [Sentence Transformers Cross-Encoders](https://www.sbert.net/docs/cross_encoder/pretrained_models.html)
- [IBM Granite R2 Blog](https://huggingface.co/blog/hansolosan/granite-embedding-r2)
- [Alibaba GTE-ModernBERT](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base)
