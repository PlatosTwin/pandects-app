"""
Generate a fresh gold set on the CURRENT (new 140-leaf) taxonomy.

This is the reference set for the hybrid-retriever recall gate. It uses a
strong model with the FULL taxonomy in the prompt (not retrieval-limited) so
it approximates "what a careful model that saw the whole taxonomy would
choose" — exactly what the retriever must preserve.

Section selection = section_select.select_sections (the maintainer's
eligibility rules). The full-taxonomy system prompt and the strict per-row
JSON schema are reused verbatim from generate_batches.py /
generate_openai_gold.py so this set is comparable to the DeepSeek pipeline.

Output (per-section, resumable): data/gold_new_taxonomy.out.jsonl
  {"section_uuid", "agreement_uuid", "categories", "error"}

Run from repo root (spends OpenAI tokens):
  caffeinate -i etl/.venv/bin/python \
      etl/src/etl/models/taxonomy/deepseek/gold_generate.py \
      --num-sections 750 --resume
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from enrichment_lib import build_engine, load_envs
from generate_batches import (
    _build_system_instructions,
    _build_user_payload,
    _fetch_taxonomy,
)
from generate_openai_gold import _build_response_schema
from section_select import SelectedSection, select_sections

DEEPSEEK_DIR = Path(__file__).resolve().parent
DATA_DIR = DEEPSEEK_DIR / "data"


def _load_done(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("section_uuid")
            if isinstance(sid, str):
                done.add(sid)
    return done


def _section_dicts(chunk: list[SelectedSection]) -> list[dict[str, Any]]:
    return [
        {
            "section_uuid": s.section_uuid,
            "agreement_uuid": s.agreement_uuid,
            "article_title": s.article_title,
            "section_title": s.section_title,
            "_body": s.body,
        }
        for s in chunk
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-sections", type=int, default=750)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--pool-agreements", type=int, default=250)
    parser.add_argument("--min-words", type=int, default=10)
    parser.add_argument("--max-words", type=int, default=3000)
    parser.add_argument("--sections-per-request", type=int, default=1)
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning-effort", default="high",
                        choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--token-budget", type=int, default=20_000_000)
    parser.add_argument("--out", default=str(DATA_DIR / "gold_new_taxonomy.out.jsonl"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s :: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("gold_generate")

    load_envs()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY not set")
        return 2
    client = OpenAI(api_key=api_key)

    sections = select_sections(
        num_sections=args.num_sections,
        seed=args.seed,
        pool_agreements=args.pool_agreements,
        min_words=args.min_words,
        max_words=args.max_words,
        log=log,
    )
    if not sections:
        log.error("no sections selected")
        return 1

    engine, schema = build_engine()
    with engine.begin() as conn:
        taxonomy = _fetch_taxonomy(conn, schema)
    system_instructions = _build_system_instructions(taxonomy)
    log.info("taxonomy rows in prompt: %d", len(taxonomy))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out_path) if args.resume else set()
    if done:
        log.info("resume: %d section_uuids already present; skipping", len(done))
    open_mode = "a" if (args.resume and out_path.exists()) else "w"

    spr = max(1, args.sections_per_request)
    pending = [s for s in sections if s.section_uuid not in done]
    chunks = [pending[i : i + spr] for i in range(0, len(pending), spr)]

    usage_total = 0
    ok = failed = api_errors = parse_errors = 0
    t0 = time.monotonic()

    with out_path.open(open_mode, encoding="utf-8") as fh:
        for ci, chunk in enumerate(chunks):
            if usage_total >= args.token_budget:
                log.warning("token budget reached (%d); stopping at chunk %d/%d",
                            usage_total, ci, len(chunks))
                break
            sec_dicts = _section_dicts(chunk)
            uuids = [s.section_uuid for s in chunk]
            agg = {s.section_uuid: s.agreement_uuid for s in chunk}
            user = _build_user_payload(sec_dicts)
            try:
                resp = client.responses.create(
                    model=args.model,
                    reasoning={"effort": args.reasoning_effort},
                    instructions=system_instructions,
                    input=[{"role": "user", "content": user}],
                    text={"format": {
                        "type": "json_schema",
                        "name": "section_category_assignments",
                        "strict": True,
                        "schema": _build_response_schema(uuids),
                    }},
                )
            except Exception as exc:  # noqa: BLE001
                api_errors += 1
                log.error("chunk %d API error: %s", ci, exc)
                for sid in uuids:
                    fh.write(json.dumps({"section_uuid": sid,
                                         "agreement_uuid": agg[sid],
                                         "categories": [], "error": "api_error"},
                                        ensure_ascii=False) + "\n")
                    fh.flush()
                    failed += 1
                continue

            usage_total += getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0
            try:
                parsed = json.loads(resp.output_text)
                assignments = {
                    a["section_uuid"]: [str(c).strip() for c in a.get("categories", [])
                                        if str(c).strip()]
                    for a in parsed.get("assignments", [])
                }
                row_parse_error = False
            except (ValueError, KeyError) as exc:
                parse_errors += 1
                row_parse_error = True
                log.error("chunk %d parse error: %s :: head=%r",
                          ci, exc, resp.output_text[:200])
                assignments = {}

            for sid in uuids:
                if sid in assignments:
                    fh.write(json.dumps({"section_uuid": sid,
                                         "agreement_uuid": agg[sid],
                                         "categories": assignments[sid],
                                         "error": None}, ensure_ascii=False) + "\n")
                    ok += 1
                else:
                    fh.write(json.dumps(
                        {"section_uuid": sid, "agreement_uuid": agg[sid],
                         "categories": [],
                         "error": "parse_error" if row_parse_error
                         else "missing_in_response"},
                        ensure_ascii=False) + "\n")
                    failed += 1
                fh.flush()

            if (ci + 1) % 25 == 0 or ci + 1 == len(chunks):
                el = time.monotonic() - t0
                log.info("chunks=%d/%d ok=%d failed=%d api_err=%d parse_err=%d "
                         "tokens=%d el=%.0fs (%.2fs/chunk)",
                         ci + 1, len(chunks), ok, failed, api_errors,
                         parse_errors, usage_total, el, el / (ci + 1))

    log.info("done: ok=%d failed=%d api_errors=%d parse_errors=%d tokens=%d -> %s",
             ok, failed, api_errors, parse_errors, usage_total, out_path)
    return 0 if api_errors == 0 and parse_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
