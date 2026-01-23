#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CORPUS="$ROOT/tests/fixtures/benchmark_corpus.json"
LOG="$ROOT/tests/e2e/results/corpus_validate_$(date +%Y%m%d_%H%M%S).log"
SUMMARY="$ROOT/tests/e2e/results/corpus_summary.json"

mkdir -p "$ROOT/tests/e2e/results"

log() { echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*" | tee -a "$LOG"; }

log "Validating corpus: $CORPUS"

DOC_COUNT=$(jq '.corpus | length' "$CORPUS")
QUERY_COUNT=$(jq '.queries | length' "$CORPUS")

log "Documents: $DOC_COUNT"
log "Queries:   $QUERY_COUNT"

# Type distribution
TYPE_COUNTS=$(jq -r '.corpus[] | .type' "$CORPUS" | sort | uniq -c | awk '{print $2 ":" $1}')
log "Type counts: $TYPE_COUNTS"

# PII scans (simple regex)
if rg -n "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}" "$CORPUS" >/dev/null; then
  log "PII check failed: found email-like pattern"
  exit 1
fi
if rg -n "\\+?[0-9][0-9\\- ]{8,}" "$CORPUS" >/dev/null; then
  log "PII check failed: found phone-like pattern"
  exit 1
fi

log "PII scan passed"

jq -n \
  --arg generated "$(date -u +"%Y-%m-%dT%H:%M:%SZ")" \
  --argjson docs "$DOC_COUNT" \
  --argjson queries "$QUERY_COUNT" \
  --arg type_counts "$TYPE_COUNTS" \
  '{generated_at:$generated, documents:$docs, queries:$queries, type_counts:$type_counts}' > "$SUMMARY"

log "Summary written: $SUMMARY"
log "Done"
