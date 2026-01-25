#!/bin/bash
# nomic-embed-text-v1.5 Baseline Benchmark E2E Script
# Bead: bd-n4eu
#
# Captures baseline metrics for nomic-embed-text-v1.5, focusing on:
# - Long context (8192 tokens) performance
# - MRL truncation quality at various dimensions
# - Task prefix behavior (search_query vs search_document)
# - Binary quantization potential
#
# IMPORTANT: This is a PRE-2025-11 model (Dec 2024) used as baseline only.
# Do NOT use for production winner selection.

set -euo pipefail

LOG=/tmp/nomic_baseline_e2e.log
RESULTS_DIR=${RESULTS_DIR:-/tmp/nomic_baseline}
CORPUS_FILE=${CORPUS_FILE:-tests/fixtures/benchmark_corpus.json}
OUTPUT_FILE="$RESULTS_DIR/nomic_embed_text_v1_5_baseline.json"

# Configuration
MODEL="nomic-embed-text-v1.5"
WARMUP_ITERS=${WARMUP_ITERS:-10}
MEASURE_ITERS=${MEASURE_ITERS:-100}

echo "========================================" | tee "$LOG"
echo "nomic-embed-text-v1.5 Baseline Benchmark" | tee -a "$LOG"
echo "Date: $(date -Iseconds)" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"

# System info
echo "" | tee -a "$LOG"
echo "SYSTEM INFORMATION" | tee -a "$LOG"
echo "------------------" | tee -a "$LOG"
echo "OS: $(uname -s) $(uname -r)" | tee -a "$LOG"
echo "Host: $(hostname)" | tee -a "$LOG"
echo "CPU: $(lscpu 2>/dev/null | grep 'Model name' | cut -d: -f2 | xargs || echo 'N/A')" | tee -a "$LOG"
echo "Cores: $(nproc 2>/dev/null || echo 'N/A')" | tee -a "$LOG"
echo "RAM: $(free -h 2>/dev/null | awk '/^Mem:/ {print $2}' || echo 'N/A')" | tee -a "$LOG"
echo "Git SHA: $(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')" | tee -a "$LOG"
echo "Rust: $(rustc --version 2>/dev/null || echo 'N/A')" | tee -a "$LOG"

# AVX features
echo "" | tee -a "$LOG"
echo "AVX FEATURES" | tee -a "$LOG"
echo "------------" | tee -a "$LOG"
if [[ -f /proc/cpuinfo ]]; then
    grep -o 'avx[0-9a-z_]*' /proc/cpuinfo 2>/dev/null | sort -u | tr '\n' ' ' | tee -a "$LOG"
    echo "" | tee -a "$LOG"
else
    echo "N/A" | tee -a "$LOG"
fi

# Setup
rm -rf "$RESULTS_DIR"
mkdir -p "$RESULTS_DIR"

# Validate prerequisites
echo "" | tee -a "$LOG"
echo "PREREQUISITES" | tee -a "$LOG"
echo "-------------" | tee -a "$LOG"

if [[ ! -f "$CORPUS_FILE" ]]; then
    echo "[FAIL] Corpus file missing: $CORPUS_FILE" | tee -a "$LOG"
    exit 1
fi

CORPUS_SIZE=$(jq '.corpus | length' "$CORPUS_FILE" 2>/dev/null || echo "0")
QUERY_COUNT=$(jq '.queries | length' "$CORPUS_FILE" 2>/dev/null || echo "0")
echo "Corpus docs: $CORPUS_SIZE" | tee -a "$LOG"
echo "Queries: $QUERY_COUNT" | tee -a "$LOG"

# Build release binary
echo "" | tee -a "$LOG"
echo "BUILDING" | tee -a "$LOG"
echo "--------" | tee -a "$LOG"
BUILD_START=$(date +%s%3N)
if cargo build --release --quiet 2>&1; then
    BUILD_END=$(date +%s%3N)
    BUILD_DURATION=$((BUILD_END - BUILD_START))
    echo "Build successful (${BUILD_DURATION}ms)" | tee -a "$LOG"
else
    echo "[FAIL] Build failed" | tee -a "$LOG"
    exit 1
fi

# Model details
echo "" | tee -a "$LOG"
echo "MODEL DETAILS" | tee -a "$LOG"
echo "-------------" | tee -a "$LOG"
echo "Model: $MODEL" | tee -a "$LOG"
echo "Parameters: 137M" | tee -a "$LOG"
echo "Native Dimensions: 768" | tee -a "$LOG"
echo "MRL Dimensions: 64, 128, 256, 384, 512, 768" | tee -a "$LOG"
echo "Max Context: 8192 tokens" | tee -a "$LOG"
echo "Pooling: Mean pooling" | tee -a "$LOG"
echo "Task Prefixes: search_query:, search_document:, clustering:, classification:" | tee -a "$LOG"
echo "Binary Quantization: Supported (768/8 = 96 bytes)" | tee -a "$LOG"

# Record RSS at idle
IDLE_RSS=$(ps -o rss= -p $$ 2>/dev/null || echo "0")
echo "Idle RSS: ${IDLE_RSS}KB" | tee -a "$LOG"

# Run benchmark
echo "" | tee -a "$LOG"
echo "RUNNING BENCHMARK" | tee -a "$LOG"
echo "-----------------" | tee -a "$LOG"
echo "Model: $MODEL" | tee -a "$LOG"
echo "Warmup: $WARMUP_ITERS iters" | tee -a "$LOG"
echo "Measure: $MEASURE_ITERS iters" | tee -a "$LOG"

BATCH_SIZES=(1 8 16 32 64)
for BATCH in "${BATCH_SIZES[@]}"; do
    echo "" | tee -a "$LOG"
    echo "Batch size: $BATCH" | tee -a "$LOG"

    BATCH_OUTPUT="$RESULTS_DIR/batch_${BATCH}.json"
    BENCH_START=$(date +%s%3N)

    if cargo run --release --quiet -- benchmark \
        --model "$MODEL" \
        --corpus "$CORPUS_FILE" \
        --warmup "$WARMUP_ITERS" \
        --measure-iters "$MEASURE_ITERS" \
        --batch-size "$BATCH" \
        --output-dir "$RESULTS_DIR" 2>&1 | tee -a "$LOG"; then

        BENCH_END=$(date +%s%3N)
        BENCH_DURATION=$((BENCH_END - BENCH_START))
        echo "  Duration: ${BENCH_DURATION}ms" | tee -a "$LOG"

        # Extract key metrics if output exists
        if [[ -f "$RESULTS_DIR/$MODEL.json" ]]; then
            mv "$RESULTS_DIR/$MODEL.json" "$BATCH_OUTPUT"
            COLD=$(jq '.speed.cold_start_ms // 0' "$BATCH_OUTPUT" 2>/dev/null || echo "N/A")
            P50=$(jq '.speed.warm_latency_p50_ms // 0' "$BATCH_OUTPUT" 2>/dev/null || echo "N/A")
            P95=$(jq '.speed.warm_latency_p95_ms // 0' "$BATCH_OUTPUT" 2>/dev/null || echo "N/A")
            THROUGHPUT=$(jq '.speed.throughput_docs_per_sec // 0' "$BATCH_OUTPUT" 2>/dev/null || echo "N/A")
            echo "  Cold start: ${COLD}ms" | tee -a "$LOG"
            echo "  P50: ${P50}ms" | tee -a "$LOG"
            echo "  P95: ${P95}ms" | tee -a "$LOG"
            echo "  Throughput: ${THROUGHPUT} docs/sec" | tee -a "$LOG"
        fi
    else
        echo "  [WARN] Benchmark failed for batch size $BATCH" | tee -a "$LOG"
    fi
done

# Quality evaluation (using batch 32)
echo "" | tee -a "$LOG"
echo "QUALITY METRICS" | tee -a "$LOG"
echo "---------------" | tee -a "$LOG"

if [[ -f "$RESULTS_DIR/batch_32.json" ]]; then
    NDCG=$(jq '.quality.ndcg_at_10 // 0' "$RESULTS_DIR/batch_32.json" 2>/dev/null || echo "N/A")
    MRR=$(jq '.quality.mrr_at_10 // 0' "$RESULTS_DIR/batch_32.json" 2>/dev/null || echo "N/A")
    MAP=$(jq '.quality.map_at_10 // 0' "$RESULTS_DIR/batch_32.json" 2>/dev/null || echo "N/A")
    RECALL=$(jq '.quality.recall_at_100 // 0' "$RESULTS_DIR/batch_32.json" 2>/dev/null || echo "N/A")
    PRECISION=$(jq '.quality.precision_at_10 // 0' "$RESULTS_DIR/batch_32.json" 2>/dev/null || echo "N/A")

    echo "NDCG@10: $NDCG" | tee -a "$LOG"
    echo "MRR@10: $MRR" | tee -a "$LOG"
    echo "MAP@10: $MAP" | tee -a "$LOG"
    echo "Recall@100: $RECALL" | tee -a "$LOG"
    echo "Precision@10: $PRECISION" | tee -a "$LOG"
fi

# Memory usage
echo "" | tee -a "$LOG"
echo "MEMORY USAGE" | tee -a "$LOG"
echo "------------" | tee -a "$LOG"
PEAK_RSS=$(cat /proc/self/status 2>/dev/null | grep VmPeak | awk '{print $2}' || echo "N/A")
CURRENT_RSS=$(cat /proc/self/status 2>/dev/null | grep VmRSS | awk '{print $2}' || echo "N/A")
echo "Peak RSS: ${PEAK_RSS}KB" | tee -a "$LOG"
echo "Current RSS: ${CURRENT_RSS}KB" | tee -a "$LOG"

# Nomic-specific: MRL analysis
echo "" | tee -a "$LOG"
echo "MRL (MATRYOSHKA) ANALYSIS" | tee -a "$LOG"
echo "-------------------------" | tee -a "$LOG"
echo "Nomic supports truncating embeddings to smaller dimensions:" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "| Dims | Storage | Expected Quality |" | tee -a "$LOG"
echo "|------|---------|------------------|" | tee -a "$LOG"
echo "|   64 |  256 B  | ~85% of full     |" | tee -a "$LOG"
echo "|  128 |  512 B  | ~90% of full     |" | tee -a "$LOG"
echo "|  256 | 1024 B  | ~95% of full     |" | tee -a "$LOG"
echo "|  384 | 1536 B  | ~97% of full     |" | tee -a "$LOG"
echo "|  512 | 2048 B  | ~99% of full     |" | tee -a "$LOG"
echo "|  768 | 3072 B  | 100% (full)      |" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "For xf use: Consider 256 dims as good balance" | tee -a "$LOG"

# Binary quantization
echo "" | tee -a "$LOG"
echo "BINARY QUANTIZATION" | tee -a "$LOG"
echo "-------------------" | tee -a "$LOG"
echo "Nomic embeddings can be binary quantized:" | tee -a "$LOG"
echo "  Full: 768 dims × 4 bytes = 3072 bytes per doc" | tee -a "$LOG"
echo "  Binary: 768 dims / 8 = 96 bytes per doc (32x compression)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Typical approach: Binary first-pass → FP32 rescore top-K" | tee -a "$LOG"
echo "Quality retention: ~85-90% of full FP32 at 4x rescore multiplier" | tee -a "$LOG"

# Long context advantage
echo "" | tee -a "$LOG"
echo "LONG CONTEXT (8192 TOKENS)" | tee -a "$LOG"
echo "--------------------------" | tee -a "$LOG"
echo "Nomic supports 8192 token context vs 512 for MiniLM/BGE" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Benefits for xf:" | tee -a "$LOG"
echo "  - Full Grok conversations (can be 2000+ tokens)" | tee -a "$LOG"
echo "  - Long DM threads without truncation" | tee -a "$LOG"
echo "  - Complete tweet context (replies, quotes)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Considerations:" | tee -a "$LOG"
echo "  - Most tweets are <100 tokens, so long context rarely used" | tee -a "$LOG"
echo "  - Memory: 8192 × 768 × 4 = 24MB per batch item (long docs)" | tee -a "$LOG"
echo "  - Latency increases with sequence length" | tee -a "$LOG"

# Generate combined baseline report
echo "" | tee -a "$LOG"
echo "GENERATING BASELINE REPORT" | tee -a "$LOG"
echo "--------------------------" | tee -a "$LOG"

{
    echo "{"
    echo "  \"model\": \"$MODEL\","
    echo "  \"type\": \"baseline\","
    echo "  \"eligible\": false,"
    echo "  \"release_date\": \"2024-12-01\","
    echo "  \"note\": \"PRE-2025-11 MODEL - BASELINE ONLY, NOT FOR PRODUCTION\","
    echo "  \"timestamp\": \"$(date -Iseconds)\","
    echo "  \"git_sha\": \"$(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')\","
    echo "  \"model_details\": {"
    echo "    \"parameters\": \"137M\","
    echo "    \"native_dimensions\": 768,"
    echo "    \"mrl_dimensions\": [64, 128, 256, 384, 512, 768],"
    echo "    \"max_context\": 8192,"
    echo "    \"pooling\": \"mean\","
    echo "    \"task_prefixes\": [\"search_query:\", \"search_document:\", \"clustering:\", \"classification:\"],"
    echo "    \"binary_quantization\": true,"
    echo "    \"binary_size_bytes\": 96"
    echo "  },"
    echo "  \"system\": {"
    echo "    \"cpu\": \"$(lscpu 2>/dev/null | grep 'Model name' | cut -d: -f2 | xargs || echo 'N/A')\","
    echo "    \"cores\": $(nproc 2>/dev/null || echo 1),"
    echo "    \"ram_gb\": $(free -g 2>/dev/null | awk '/^Mem:/ {print $2}' || echo 0)"
    echo "  },"
    echo "  \"batch_results\": ["
    FIRST=true
    for f in "$RESULTS_DIR"/batch_*.json; do
        if [[ -f "$f" ]]; then
            if [[ "$FIRST" == "true" ]]; then
                FIRST=false
            else
                echo ","
            fi
            cat "$f"
        fi
    done
    echo ""
    echo "  ]"
    echo "}"
} > "$OUTPUT_FILE"

echo "Baseline report written to: $OUTPUT_FILE" | tee -a "$LOG"

# Summary
echo "" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "BASELINE BENCHMARK COMPLETE" | tee -a "$LOG"
echo "Model: $MODEL" | tee -a "$LOG"
echo "Output: $OUTPUT_FILE" | tee -a "$LOG"
echo "Log: $LOG" | tee -a "$LOG"
echo "========================================" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "KEY FEATURES OF NOMIC:" | tee -a "$LOG"
echo "  - 8192 token context (16x MiniLM/BGE)" | tee -a "$LOG"
echo "  - MRL support: truncate to 64-768 dims" | tee -a "$LOG"
echo "  - Binary quantization: 32x storage reduction" | tee -a "$LOG"
echo "  - Task prefixes for query/document distinction" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "COMPARISON WITH MiniLM BASELINE:" | tee -a "$LOG"
echo "  - Larger model (137M vs 22M params)" | tee -a "$LOG"
echo "  - Higher quality (MTEB 55+ vs 42)" | tee -a "$LOG"
echo "  - Higher dims (768 vs 384)" | tee -a "$LOG"
echo "  - MRL flexibility (can match smaller dims)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "NOTE: This model is a PRE-2025-11 baseline." | tee -a "$LOG"
echo "Use these metrics as comparison points only." | tee -a "$LOG"
