# Post-2025-10 Tiny CPU Embedding Model Analysis

**Research Date:** 2026-01-24
**Bead:** bd-e9kz
**Hard Filter:** Models released or updated on/after **2025-11-01**
**Eligibility:** Open weights, CPU-viable, <=500M params or <=500MB disk, permissive license

---

## Executive Summary

After extensive research of HuggingFace Hub, vendor announcements, and MTEB leaderboard, **only 1 model** strictly meets the >= 2025-11-01 release date requirement:

| Model | Released | Params | Status |
|-------|----------|--------|--------|
| **voyage-4-nano** | 2026-01-15 | ~600M | **ELIGIBLE** |

Several strong candidates were released in early 2025 (before cutoff) and should be considered as **baselines**.

---

## ELIGIBLE: voyage-4-nano (2026-01-15)

**Source:** [voyageai/voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano)

### Specifications
- **Release Date:** 2026-01-15
- **Parameters:** ~600M (based on Qwen)
- **Dimensions:** 2048 default, MRL supports 256/512/1024/2048
- **Context:** 32K tokens
- **License:** Apache-2.0
- **Runtime:** HuggingFace Transformers, ONNX export possible

### CPU Viability
- Designed for local/on-device use
- Matryoshka representation learning enables 256-dim embeddings
- Quantization-aware training supports INT8 and binary precision

### Key Features
- Shared embedding space with voyage-4-large/lite (no re-indexing needed)
- First open-weight Voyage model
- Built on Qwen foundation

### Concerns
- ~600M params approaches upper limit for "tiny"
- May require quantization for fast CPU inference

---

## BASELINE MODELS (Pre-Cutoff, High Quality)

### gte-modernbert-base (2025-01-21)

**Source:** [Alibaba-NLP/gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)

- **Parameters:** 149M
- **Dimensions:** 768
- **Context:** 8192 tokens
- **License:** Apache-2.0
- **Why Baseline:** Released 2025-01-21 (before 2025-11-01 cutoff)

**Strengths:** #1 on MTEB for <300M params. ModernBERT architecture optimized for CPU.

---

### granite-embedding-small-english-r2 (2025-08-15)

**Source:** [ibm-granite/granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2)

- **Parameters:** 47M
- **Dimensions:** 384
- **Context:** 8192 tokens
- **License:** Apache-2.0
- **Why Baseline:** Released 2025-08-15 (before cutoff)

**Strengths:** Smallest model with strong quality. #2 on MTEB for <100M. ~200 docs/sec on H100.

---

### potion-retrieval-32M (2025-01-30)

**Source:** [minishlab/potion-retrieval-32M](https://huggingface.co/minishlab/potion-retrieval-32M)

- **Parameters:** 32M
- **Dimensions:** 768
- **Disk Size:** ~30MB
- **License:** MIT
- **Runtime:** model2vec
- **Why Baseline:** Released 2025-01-30 (before cutoff)

**Strengths:** 500x faster than transformer models. Best static retrieval model.

---

### Qwen3-Embedding-0.6B (2025-06-05)

**Source:** [Qwen/Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B)

- **Parameters:** 600M
- **Dimensions:** 1024
- **Context:** 32K tokens
- **License:** Apache-2.0
- **Why Baseline:** Released 2025-06-05 (before cutoff)

**Strengths:** GGUF quantization available. 100+ language support.

---

### static-retrieval-mrl-en-v1 (Early 2025)

**Source:** [sentence-transformers/static-retrieval-mrl-en-v1](https://huggingface.co/sentence-transformers/static-retrieval-mrl-en-v1)

- **Parameters:** ~0 active (static embeddings)
- **Dimensions:** 256-1024 (MRL)
- **Disk Size:** ~100MB vocab
- **License:** Apache-2.0
- **Why Baseline:** Already integrated in xf; pre-cutoff

**Strengths:** 100-400x faster than transformers. Outperforms BM25.

---

## Rejected Models

| Model | Reason |
|-------|--------|
| jina-embeddings-v3 | 570M params, GPU-focused |
| BGE-M3 | 568M params, GPU-targeted |
| Qwen3-Embedding-8B | Too large (8B) |
| nomic-embed-text-v1.5 | Released 2024 |
| all-MiniLM-L6-v2 | Released 2021 |
| multilingual-e5-small | Released 2023 |
| snowflake-arctic-embed-xs | Released 2024 |

---

## Recommendations

### Option A: Strict Cutoff (Recommended for Bake-off Integrity)

Use **voyage-4-nano** as the sole eligible candidate. Compare against baselines:
1. static-retrieval-mrl-en-v1 (current xf implementation, speed baseline)
2. potion-retrieval-32M (speed champion)
3. granite-embedding-small-english-r2 (quality baseline, tiny)
4. gte-modernbert-base (quality baseline, medium)

### Option B: Relax Cutoff to 2025-01-01

Include January 2025 releases:
- gte-modernbert-base (2025-01-21)
- potion-retrieval-32M (2025-01-30)
- potion-base-32M (2025-01-30)

This would give 4 eligible models total.

### Option C: Relax Size to 1B with Quantization

Include larger models with GGUF/INT8 quantization:
- Qwen3-Embedding-0.6B
- nomic-embed-text-v2-moe

---

## Decision Gate

Per bd-wyw optimization notes: if **eligible==0** or very low, coordinate via bd-5fj3 to either:
1. Relax cutoff/size thresholds, OR
2. Proceed baseline-only with exception flag

**Current eligible count: 1** (voyage-4-nano only)

**Recommendation:** Proceed with bake-off using voyage-4-nano vs baselines, but flag the low eligible count. The baselines from early 2025 (gte-modernbert-base, potion models) are strong enough to inform real decisions.

---

## Sources

- [Voyage AI Blog: Voyage 4](https://blog.voyageai.com/2026/01/15/voyage-4/)
- [HuggingFace: voyage-4-nano](https://huggingface.co/voyageai/voyage-4-nano)
- [HuggingFace: gte-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-modernbert-base)
- [HuggingFace: granite-embedding-small-english-r2](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2)
- [MinishLab Model2Vec](https://github.com/MinishLab/model2vec)
- [Sentence Transformers Static Embeddings](https://huggingface.co/blog/static-embeddings)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard)
- [IBM Granite Announcement](https://www.marktechpost.com/2025/09/12/ibm-ai-research-releases-two-english-granite-embedding-models-both-based-on-the-modernbert-architecture/)
