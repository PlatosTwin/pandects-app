"""
FP4 (HPC) vs FP8 (official API) sanity probe for the A prompt.

The HPC run serves DeepSeek-V4-Pro under FP4 (marlin MoE). The official
DeepSeek API serves the same model at FP8. This script replays the EXACT A
prompts (system+user, candidate-shortlisted, tx-context, no rationale) for a
sample of sections the HPC FP4 run got wrong vs the GPT gold, and asks the
FP8 API the same thing. If FP8 recovers a meaningful share of the misses, the
quantization is the bottleneck, not the prompt or the model.

Misses are restricted to gold-NONEMPTY sections (real classification misses,
not abstention judgement calls where gold itself is noisy).

Run from repo root:
  caffeinate -i etl/.venv/bin/python \
      etl/src/etl/models/taxonomy/deepseek/fp8_probe.py \
      --n 20 --seed 4242
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from sqlalchemy import text

from enrichment_lib import build_engine

DEEPSEEK_DIR = Path(__file__).resolve().parent
DATA_DIR = DEEPSEEK_DIR / "data"

API_KEY = os.environ["DEEPSEEK_API_KEY"]
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-pro"


def _load(path: Path) -> dict[str, set[str]]:
    """section_uuid -> set(categories); error rows skipped, []-rows kept."""
    out: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error"):
                continue
            sid = r.get("section_uuid")
            if not isinstance(sid, str):
                continue
            out[sid] = {
                str(c).strip() for c in (r.get("categories") or [])
                if str(c).strip()
            }
    return out


def _load_batch_rows(path: Path) -> dict[str, dict[str, Any]]:
    """section_uuid -> {messages, allowed_category_ids}."""
    out: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            secs = r.get("sections") or []
            if not secs:
                continue
            sid = secs[0].get("section_uuid")
            if not isinstance(sid, str):
                continue
            out[sid] = {
                "messages": r["messages"],
                "allowed": set(r.get("allowed_category_ids") or []),
            }
    return out


def _taxonomy() -> dict[str, dict[str, str]]:
    engine, schema = build_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                f"""SELECT standard_id, label, l2_label
                    FROM {schema}.taxonomy_node_meta"""
            )
        ).mappings().all()
    return {
        str(r["standard_id"]): {
            "label": str(r["label"]), "l2": str(r["l2_label"]),
        }
        for r in rows
    }


def _fmt(ids: set[str], tax: dict[str, dict[str, str]]) -> str:
    if not ids:
        return "[]"
    parts = []
    for c in sorted(ids):
        n = tax.get(c)
        parts.append(f"{n['l2']} > {n['label']}" if n else f"?{c}")
    return " | ".join(parts)


def _call_fp8(client: OpenAI, messages: list[dict[str, str]],
              sid: str) -> tuple[set[str], str | None]:
    """Return (categories, error). Parses {"assignments":[{...}]} JSON."""
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=messages,  # pyright: ignore[reportArgumentType]
            temperature=0.0,
            max_tokens=2000,
            response_format={"type": "json_object"},
        )
    except Exception as e:  # noqa: BLE001 - probe: surface any API failure
        return set(), f"api_error: {e}"
    raw = resp.choices[0].message.content or ""
    try:
        obj = json.loads(raw)
        asg = obj.get("assignments") or []
        for a in asg:
            if a.get("section_uuid") == sid or len(asg) == 1:
                return {
                    str(c).strip() for c in (a.get("categories") or [])
                    if str(c).strip()
                }, None
        return set(), "no_matching_assignment"
    except (json.JSONDecodeError, AttributeError, TypeError) as e:
        return set(), f"parse_error: {e} :: {raw[:200]!r}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pred", default="gold_hybrid_A_2026-05-17.out.jsonl")
    p.add_argument("--batch", default="gold_hybrid_A_2026-05-17.jsonl")
    p.add_argument("--gold", default=str(DATA_DIR / "gold_new_taxonomy.out.jsonl"))
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save", default=str(DATA_DIR / "fp8_probe_results.jsonl"))
    args = p.parse_args()

    logging.basicConfig(
        level="INFO", format="%(asctime)s %(levelname)s :: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("fp8_probe")

    gold = _load(Path(args.gold))
    fp4 = _load(DATA_DIR / args.pred)
    batch = _load_batch_rows(DATA_DIR / args.batch)
    tax = _taxonomy()

    misses = sorted(
        sid for sid in (set(gold) & set(fp4) & set(batch))
        if gold[sid] and gold[sid] != fp4[sid]
    )
    log.info("gold-nonempty FP4 misses available: %d", len(misses))
    rng = random.Random(args.seed)
    sample = rng.sample(misses, min(args.n, len(misses)))
    log.info("probing %d sections through FP8 (%s @ %s)",
             len(sample), MODEL, BASE_URL)

    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    results: dict[str, tuple[set[str], str | None]] = {}

    def _job(sid: str) -> tuple[str, set[str], str | None]:
        cats, err = _call_fp8(client, batch[sid]["messages"], sid)
        return sid, cats, err

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_job, sid): sid for sid in sample}
        done = 0
        for fut in as_completed(futs):
            sid, cats, err = fut.result()
            results[sid] = (cats, err)
            done += 1
            log.info("  %d/%d done (sid=%s%s)", done, len(sample),
                     sid[:8], f" ERR {err}" if err else "")

    fp8_right = fp8_eq_fp4 = fp8_changed_still_wrong = 0
    invalid = api_err = 0
    save_rows = []
    log.info("=" * 78)
    for sid in sample:
        g, f4 = gold[sid], fp4[sid]
        f8, err = results[sid]
        allowed = batch[sid]["allowed"]
        bad_ids = {c for c in f8 if c not in allowed}
        if err:
            api_err += 1
            verdict = f"ERROR {err[:60]}"
        else:
            if bad_ids:
                invalid += 1
            if f8 == g:
                fp8_right += 1
                verdict = "FP8 ✓ recovered (== gold)"
            elif f8 == f4:
                fp8_eq_fp4 += 1
                verdict = "FP8 == FP4 (no change, still wrong)"
            else:
                fp8_changed_still_wrong += 1
                verdict = "FP8 changed, still != gold"
        log.info("sid %s :: %s", sid[:8], verdict)
        log.info("   gold: %s", _fmt(g, tax))
        log.info("   FP4 : %s", _fmt(f4, tax))
        log.info("   FP8 : %s%s", _fmt(f8, tax),
                 f"  [INVALID ids: {bad_ids}]" if bad_ids else "")
        save_rows.append({
            "section_uuid": sid, "gold": sorted(g), "fp4": sorted(f4),
            "fp8": sorted(f8), "fp8_error": err,
            "fp8_invalid_ids": sorted(bad_ids),
        })
    n = len(sample)
    log.info("=" * 78)
    log.info("SUMMARY over %d FP4 misses (gold-nonempty):", n)
    log.info("  FP8 recovered (now == gold) : %d (%.1f%%)",
             fp8_right, 100 * fp8_right / n if n else 0.0)
    log.info("  FP8 same as FP4 (no change) : %d", fp8_eq_fp4)
    log.info("  FP8 changed but still wrong : %d", fp8_changed_still_wrong)
    log.info("  FP8 invalid ids / api errs  : %d / %d", invalid, api_err)
    Path(args.save).write_text(
        "\n".join(json.dumps(r) for r in save_rows) + "\n", encoding="utf-8")
    log.info("raw results -> %s", args.save)
    log.info("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
