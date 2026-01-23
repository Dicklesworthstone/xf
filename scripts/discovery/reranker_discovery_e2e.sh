#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
LOG="$ROOT/docs/reranker_candidates_2025_11+_log.txt"
OUT_JSON="$ROOT/docs/reranker_candidates_2025_11+.json"
OUT_MD="$ROOT/docs/reranker_candidates_2025_11+.md"
SUMMARY="$ROOT/docs/reranker_candidates_2025_11+_summary.json"
CUTOFF="2025-11-01"

mkdir -p "$ROOT/docs"

stamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

echo "[start] $(stamp)" | tee "$LOG"

echo "[run] reranker discovery" | tee -a "$LOG"
python3 "$ROOT/scripts/discovery/reranker_discovery.py" \
  --cutoff "$CUTOFF" \
  --output-json "$OUT_JSON" \
  --output-md "$OUT_MD" \
  --log "$LOG"

ELIGIBLE_COUNT=$(jq '.eligible | length' "$OUT_JSON")
BASELINE_COUNT=$(jq '.baseline | length' "$OUT_JSON")
REJECT_COUNT=$(jq '.rejected | length' "$OUT_JSON")

BAD_COUNT=$(jq --arg cutoff "$CUTOFF" '[.eligible[] | select(.last_modified != null) | select(.last_modified < ($cutoff + "T00:00:00Z"))] | length' "$OUT_JSON")
if [[ "$BAD_COUNT" != "0" ]]; then
  echo "[fail] Found eligible models older than cutoff" | tee -a "$LOG"
  exit 1
fi

BAD_ELIGIBLE_FLAGS=$(jq '[.eligible[] | select(.flags | index("license_unknown") or index("size_unknown") or index("date_unknown"))] | length' "$OUT_JSON")
BAD_ELIGIBLE_WEIGHTS=$(jq '[.eligible[] | select(.reject_reason == "no_weight_files")] | length' "$OUT_JSON")
if [[ "$BAD_ELIGIBLE_FLAGS" != "0" || "$BAD_ELIGIBLE_WEIGHTS" != "0" ]]; then
  echo "[fail] Eligible models include unknown license/size/date or missing weights" | tee -a "$LOG"
  exit 1
fi

jq -n \
  --arg generated "$(stamp)" \
  --argjson eligible "$ELIGIBLE_COUNT" \
  --argjson baseline "$BASELINE_COUNT" \
  --argjson rejected "$REJECT_COUNT" \
  '{generated_at:$generated, eligible:$eligible, baseline:$baseline, rejected:$rejected}' > "$SUMMARY"

echo "[counts] eligible=$ELIGIBLE_COUNT baseline=$BASELINE_COUNT rejected=$REJECT_COUNT" | tee -a "$LOG"

echo "[done] $(stamp)" | tee -a "$LOG"
