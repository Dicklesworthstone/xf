#!/usr/bin/env python3
"""Generate a synthetic benchmark corpus with graded relevance labels.

This script produces a stable, PII-free dataset for benchmark harness tests.
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone

DOC_TYPES = [
    ("tweet", 1000),
    ("dm", 500),
    ("grok", 500),
    ("cass", 1000),
]

TOPICS = [
    "rust", "sqlite", "embeddings", "indexing", "benchmark",
    "daemon", "reranker", "vector", "hashing", "search",
    "tokenizer", "onnx", "metrics", "recall", "precision",
]


def build_docs(seed: int) -> list[dict]:
    rng = random.Random(seed)
    docs = []
    for doc_type, count in DOC_TYPES:
        for i in range(count):
            topic = rng.choice(TOPICS)
            text = f"{doc_type} {i} about {topic} with details on {rng.choice(TOPICS)}"
            docs.append({
                "id": f"{doc_type}-{i}",
                "text": text,
                "type": doc_type,
                "metadata": {"topic": topic},
            })
    rng.shuffle(docs)
    return docs


def build_queries(docs: list[dict], seed: int, query_count: int = 150) -> list[dict]:
    rng = random.Random(seed)
    queries = []
    # Build topic buckets
    by_topic = {}
    for doc in docs:
        by_topic.setdefault(doc["metadata"]["topic"], []).append(doc)

    topics = list(by_topic.keys())
    for i in range(query_count):
        topic = topics[i % len(topics)]
        candidates = by_topic[topic]
        rng.shuffle(candidates)
        relevants = {}
        for rank, doc in enumerate(candidates[:10]):
            relevants[doc["id"]] = 2 if rank < 3 else 1
        queries.append({
            "id": f"q{i}",
            "text": f"{topic} search query {i}",
            "relevants": relevants,
            "category": "Keyword" if i % 2 == 0 else "Short",
        })
    return queries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--queries", type=int, default=150)
    args = parser.parse_args()

    docs = build_docs(args.seed)
    queries = build_queries(docs, args.seed, args.queries)

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "corpus": docs,
        "queries": queries,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
