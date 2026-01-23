#!/usr/bin/env python3
"""Discover post-2025-10 embedding models from HF API with detailed logging.
Outputs JSON + Markdown + log file.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
import urllib.parse
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple

CUTOFF_DEFAULT = "2025-11-01"
PIPELINE_TAG = "feature-extraction"
BASE_URL = "https://huggingface.co/api/models"
MODEL_INFO_URL = "https://huggingface.co/api/models/{model_id}"
USER_AGENT = "xf-bakeoff-discovery/0.1"

BASELINES = [
    # Common historical baselines (pre-2025-11)
    "sentence-transformers/all-MiniLM-L6-v2",
    "BAAI/bge-small-en-v1.5",
    "intfloat/multilingual-e5-small",
    "sentence-transformers/static-retrieval-mrl-en-v1",
    "minishlab/potion-retrieval-32M",
    "minishlab/potion-multilingual-128M",
]

NONCOMMERCIAL_LICENSE_HINTS = {
    "cc-by-nc",
    "cc-by-nc-4.0",
    "cc-by-nc-sa",
    "cc-by-nc-sa-4.0",
    "non-commercial",
    "gated",
}

WEIGHT_EXTS = {
    "safetensors",
    "bin",
    "onnx",
    "gguf",
    "ggml",
    "pt",
    "pth",
    "mlmodel",
}

CPU_FORMAT_HINTS = {
    "onnx": "onnxruntime",
    "gguf": "ggml/llama.cpp",
    "ggml": "ggml/llama.cpp",
    "mlmodel": "coreml",
}


def parse_date(value: Optional[str]) -> Optional[dt.datetime]:
    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return dt.datetime.fromisoformat(value)
    except Exception:
        return None


def format_date(value: Optional[dt.datetime]) -> str:
    if not value:
        return ""
    return value.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def request_json(url: str) -> Tuple[List[dict], dict]:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req) as resp:
        data = resp.read().decode("utf-8")
        headers = dict(resp.headers)
    return json.loads(data), headers


def request_model_info(model_id: str) -> Optional[dict]:
    url = MODEL_INFO_URL.format(model_id=urllib.parse.quote(model_id, safe="/"))
    for attempt in range(5):
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            if exc.code == 429:
                retry_after = exc.headers.get("Retry-After")
                wait = int(retry_after) if retry_after and retry_after.isdigit() else (2 ** attempt)
                time.sleep(min(wait, 30))
                continue
            return None
        except Exception:
            return None
    return None


def parse_next_link(link_header: Optional[str]) -> Optional[str]:
    if not link_header:
        return None
    parts = link_header.split(",")
    for part in parts:
        segs = part.split(";")
        if len(segs) < 2:
            continue
        url_part = segs[0].strip()
        rel_part = ";".join(segs[1:])
        if 'rel="next"' in rel_part:
            if url_part.startswith("<") and url_part.endswith(">"):
                return url_part[1:-1]
            return url_part
    return None


def compute_weight_sizes(siblings: List[dict]) -> Tuple[Optional[float], Optional[float], List[str]]:
    sizes = []
    formats = set()
    for s in siblings:
        fname = s.get("rfilename") or ""
        size = s.get("size")
        ext = fname.split(".")[-1].lower() if "." in fname else ""
        if ext in WEIGHT_EXTS:
            if isinstance(size, (int, float)):
                sizes.append(size)
            formats.add(ext)
        elif ext:
            formats.add(ext)
    min_mb = None
    max_mb = None
    if sizes:
        min_mb = min(sizes) / (1024 * 1024)
        max_mb = max(sizes) / (1024 * 1024)
    return min_mb, max_mb, sorted(formats)


def detect_cpu_runtimes(formats: List[str]) -> List[str]:
    runtimes = set()
    for f in formats:
        if f in CPU_FORMAT_HINTS:
            runtimes.add(CPU_FORMAT_HINTS[f])
    if not runtimes:
        runtimes.add("pytorch")
    return sorted(runtimes)


def is_noncommercial(license_str: Optional[str]) -> bool:
    if not license_str:
        return False
    lower = license_str.lower()
    return any(hint in lower for hint in NONCOMMERCIAL_LICENSE_HINTS)


def entry_from_model(model: dict, info: Optional[dict], cutoff_dt: dt.datetime) -> dict:
    model_id = model.get("modelId")
    created = parse_date(model.get("createdAt"))
    modified = parse_date(model.get("lastModified"))
    info_created = parse_date(info.get("createdAt") if info else None)
    info_modified = parse_date(info.get("lastModified") if info else None)
    if not created and info_created:
        created = info_created
    if not modified and info_modified:
        modified = info_modified

    siblings = info.get("siblings", []) if info else []
    min_mb, max_mb, formats = compute_weight_sizes(siblings)
    runtimes = detect_cpu_runtimes(formats)
    license_str = (info or {}).get("license") or model.get("license") or ""
    tags = (info or {}).get("tags") or model.get("tags") or []
    library = model.get("library_name") or (info or {}).get("library_name")

    flags = []
    if not license_str:
        flags.append("license_unknown")
    if min_mb is None:
        flags.append("size_unknown")

    tiny = None
    if min_mb is not None:
        tiny = min_mb <= 500.0

    status = "eligible"
    reject_reason = None
    if is_noncommercial(license_str):
        status = "reject"
        reject_reason = "noncommercial_license"
    if tiny is False:
        status = "reject"
        reject_reason = "size_gt_500mb"

    # Ensure eligibility by date
    if modified and modified < cutoff_dt:
        status = "baseline"

    return {
        "model_id": model_id,
        "created_at": format_date(created),
        "last_modified": format_date(modified),
        "pipeline_tag": model.get("pipeline_tag"),
        "library": library,
        "license": license_str,
        "formats": formats,
        "cpu_runtimes": runtimes,
        "min_weight_mb": round(min_mb, 2) if min_mb is not None else None,
        "max_weight_mb": round(max_mb, 2) if max_mb is not None else None,
        "tags": tags,
        "flags": flags,
        "status": status,
        "reject_reason": reject_reason,
        "url": f"https://huggingface.co/{model_id}" if model_id else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", default=CUTOFF_DEFAULT)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--limit-pages", type=int, default=0)
    args = parser.parse_args()

    cutoff_dt = parse_date(args.cutoff + "T00:00:00Z")
    if not cutoff_dt:
        print("Invalid cutoff date", file=sys.stderr)
        return 1

    log_lines = []
    def log(msg: str) -> None:
        print(msg)
        log_lines.append(msg)

    log(f"[start] cutoff={args.cutoff} pipeline_tag={PIPELINE_TAG}")
    start = time.time()

    url = f"{BASE_URL}?pipeline_tag={PIPELINE_TAG}&sort=lastModified&direction=-1&limit=100"
    page = 0
    all_models: List[dict] = []

    while url:
        page += 1
        data, headers = request_json(url)
        log(f"[page {page}] fetched {len(data)} models")
        all_models.extend(data)
        if args.limit_pages and page >= args.limit_pages:
            log("[page-limit] stopping due to limit-pages")
            break

        # Early cutoff: if last model is older than cutoff, stop
        if data:
            last_mod = parse_date(data[-1].get("lastModified"))
            if last_mod and last_mod < cutoff_dt:
                log("[cutoff] reached models older than cutoff; stopping")
                break
        url = parse_next_link(headers.get("Link"))
        if url:
            time.sleep(0.2)

    log(f"[scan] total models scanned={len(all_models)}")

    eligible = []
    baseline = []
    rejected = []
    seen = set()

    # Process models that meet cutoff (or baseline ones)
    for model in all_models:
        model_id = model.get("modelId")
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        modified = parse_date(model.get("lastModified"))
        created = parse_date(model.get("createdAt"))
        is_recent = (modified and modified >= cutoff_dt) or (created and created >= cutoff_dt)
        if not is_recent and model_id not in BASELINES:
            continue

        info = request_model_info(model_id)
        entry = entry_from_model(model, info, cutoff_dt)
        if entry["status"] == "eligible":
            eligible.append(entry)
        elif entry["status"] == "baseline":
            baseline.append(entry)
        else:
            rejected.append(entry)
        time.sleep(0.2)

    # Add explicit baselines if missing
    for model_id in BASELINES:
        if model_id in seen:
            continue
        info = request_model_info(model_id)
        if not info:
            continue
        model_stub = {
            "modelId": model_id,
            "createdAt": info.get("createdAt"),
            "lastModified": info.get("lastModified"),
            "pipeline_tag": info.get("pipeline_tag"),
            "library_name": info.get("library_name"),
            "tags": info.get("tags"),
            "license": info.get("license"),
        }
        entry = entry_from_model(model_stub, info, cutoff_dt)
        entry["status"] = "baseline"
        baseline.append(entry)

    # Sort by lastModified desc
    def sort_key(e: dict) -> str:
        return e.get("last_modified") or ""

    eligible.sort(key=sort_key, reverse=True)
    baseline.sort(key=sort_key, reverse=True)
    rejected.sort(key=sort_key, reverse=True)

    output = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "cutoff_date": args.cutoff,
        "pipeline_tag": PIPELINE_TAG,
        "eligible": eligible,
        "baseline": baseline,
        "rejected": rejected,
        "stats": {
            "eligible": len(eligible),
            "baseline": len(baseline),
            "rejected": len(rejected),
        },
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Markdown
    def md_table(items: List[dict]) -> str:
        lines = ["| Model | Last Modified | Size MB (min/max) | License | Formats | Flags |",
                 "|---|---|---|---|---|---|"]
        for e in items:
            size = ""
            if e.get("min_weight_mb") is not None or e.get("max_weight_mb") is not None:
                size = f"{e.get('min_weight_mb')}/{e.get('max_weight_mb')}"
            flags = ",".join(e.get("flags", []))
            lines.append("| {model} | {lm} | {size} | {lic} | {fmt} | {flags} |".format(
                model=f"[{e['model_id']}]({e['url']})" if e.get("url") else e.get("model_id"),
                lm=e.get("last_modified") or "",
                size=size,
                lic=e.get("license") or "",
                fmt=",".join(e.get("formats", [])),
                flags=flags,
            ))
        return "\n".join(lines)

    md = [
        f"# Embedding Candidates (>= {args.cutoff})",
        "",
        f"Generated: {output['generated_at']}",
        "",
        "## Eligible",
        md_table(eligible),
        "",
        "## Baseline",
        md_table(baseline),
        "",
        "## Rejected",
        md_table(rejected),
        "",
    ]

    with open(args.output_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    elapsed = time.time() - start
    log(f"[done] eligible={len(eligible)} baseline={len(baseline)} rejected={len(rejected)} elapsed={elapsed:.2f}s")

    with open(args.log, "w", encoding="utf-8") as f:
        f.write("\n".join(log_lines) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
