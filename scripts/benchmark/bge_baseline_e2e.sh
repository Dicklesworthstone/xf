#!/bin/bash
# bge-small-en-v1.5 Baseline Benchmark E2E Script
# Bead: bd-2ro
#
# Captures baseline metrics for bge-small-en-v1.5, focusing on:
# - CLS pooling performance
# - Instruction prefix impact on retrieval quality
# - Comparison with MiniLM baseline
#
# IMPORTANT: This is a PRE-2025-11 model used as baseline only.
# Do NOT use for production winner selection.

set -euo pipefail

LOG=/tmp/bge_baseline_e2e.log
RESULTS_DIR=${RESULTS_DIR:-/tmp/bge_baseline}
CORPUS_FILE=${CORPUS_FILE:-tests/fixtures/benchmark_corpus.json}
OUTPUT_FILE="$RESULTS_DIR/bge_small_en_v1_5_baseline.json"

# Configuration
MODEL="bge-small-en-v1.5"
WARMUP_ITERS=${WARMUP_ITERS:-10}
MEASURE_ITERS=${MEASURE_ITERS:-100}

echo "========================================" | tee "$LOG"
echo "bge-small-en-v1.5 Baseline Benchmark" | tee -a "$LOG"
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

# Check model availability
echo "" | tee -a "$LOG"
echo "MODEL DETAILS" | tee -a "$LOG"
echo "-------------" | tee -a "$LOG"
echo "Model: $MODEL" | tee -a "$LOG"
echo "Parameters: 33.4M" | tee -a "$LOG"
echo "Dimensions: 384" | tee -a "$LOG"
echo "Max Context: 512 tokens" | tee -a "$LOG"
echo "Pooling: CLS (position 0)" | tee -a "$LOG"
echo "Instruction: Uses query prefix for retrieval" | tee -a "$LOG"
echo "MRL: No" | tee -a "$LOG"

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

# BGE-specific: instruction prefix comparison
echo "" | tee -a "$LOG"
echo "INSTRUCTION PREFIX ANALYSIS" | tee -a "$LOG"
echo "---------------------------" | tee -a "$LOG"
echo "BGE uses instruction prefix for queries:" | tee -a "$LOG"
echo "  'Represent this sentence for searching relevant passages: '" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "Expected impact: +2-5% retrieval quality vs no prefix" | tee -a "$LOG"
echo "Note: FastEmbed handles prefix automatically for queries" | tee -a "$LOG"

# CLS vs Mean pooling comparison
echo "" | tee -a "$LOG"
echo "POOLING COMPARISON (vs MiniLM)" | tee -a "$LOG"
echo "------------------------------" | tee -a "$LOG"
echo "BGE: CLS pooling (position 0 of hidden state)" | tee -a "$LOG"
echo "MiniLM: Mean pooling (average of all token representations)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "CLS pooling advantages:" | tee -a "$LOG"
echo "  - Faster (no averaging across sequence)" | tee -a "$LOG"
echo "  - Trained specifically for retrieval" | tee -a "$LOG"
echo "CLS pooling disadvantages:" | tee -a "$LOG"
echo "  - May lose long-context information" | tee -a "$LOG"
echo "  - Requires model trained for CLS extraction" | tee -a "$LOG"

# Generate combined baseline report
echo "" | tee -a "$LOG"
echo "GENERATING BASELINE REPORT" | tee -a "$LOG"
echo "--------------------------" | tee -a "$LOG"

{
    echo "{"
    echo "  \"model\": \"$MODEL\","
    echo "  \"type\": \"baseline\","
    echo "  \"eligible\": false,"
    echo "  \"release_date\": \"2023-09-12\","
    echo "  \"note\": \"PRE-2025-11 MODEL - BASELINE ONLY, NOT FOR PRODUCTION\","
    echo "  \"timestamp\": \"$(date -Iseconds)\","
    echo "  \"git_sha\": \"$(git rev-parse --short HEAD 2>/dev/null || echo 'N/A')\","
    echo "  \"model_details\": {"
    echo "    \"parameters\": \"33.4M\","
    echo "    \"dimensions\": 384,"
    echo "    \"max_context\": 512,"
    echo "    \"pooling\": \"cls\","
    echo "    \"instruction_prefix\": true,"
    echo "    \"mrl\": false"
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
echo "KEY DIFFERENCES FROM MiniLM BASELINE:" | tee -a "$LOG"
echo "  - CLS pooling instead of mean pooling" | tee -a "$LOG"
echo "  - Instruction prefix for query embeddings" | tee -a "$LOG"
echo "  - Higher MTEB retrieval score (~51 vs ~42)" | tee -a "$LOG"
echo "  - Slightly larger (33M vs 22M params)" | tee -a "$LOG"
echo "" | tee -a "$LOG"
echo "NOTE: This model is a PRE-2025-11 baseline." | tee -a "$LOG"
echo "Use these metrics as comparison points only." | tee -a "$LOG"
