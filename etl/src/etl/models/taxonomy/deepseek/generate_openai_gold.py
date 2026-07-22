"""
One-off: run gpt-5.4-mini over the same batch.jsonl rows that batch_inference.py
consumes on HPC, so we have a gold-label comparison set for the DeepSeek run.

Input rows match the shape produced by generate_batches.py:
  {"sections": [{"section_uuid", "agreement_uuid"}, ...],
   "messages": [{system}, {user}]}

Output mirrors batch_inference.py's per-section JSONL:
  {"section_uuid", "agreement_uuid", "categories", "error"}

Notes:
- Synchronous OpenAI Responses API with strict JSON-schema constrained output.
- Per-row schema pins section_uuid to the row's actual UUIDs (same belt-and-
  suspenders as batch_inference.py) so the model can't hallucinate UUIDs.
- Tracks cumulative usage tokens; stops before --token-budget to avoid
  blowing past the user-set cap (default 9.9M).
- Writes line-by-line with flush, and supports --resume to skip section_uuids
  already present in the output file (so re-running after a crash / token
  cap doesn't redo work).
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

from dotenv import load_dotenv
from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[6]
DEEPSEEK_DIR = Path(__file__).resolve().parent
DATA_DIR = DEEPSEEK_DIR / "data"
BACKEND_ENV = REPO_ROOT / "backend" / ".env"


def _build_response_schema(section_uuids: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "minItems": len(section_uuids),
                "maxItems": len(section_uuids),
                "items": {
                    "type": "object",
                    "properties": {
                        "section_uuid": {"type": "string", "enum": section_uuids},
                        "categories": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["section_uuid", "categories"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["assignments"],
        "additionalProperties": False,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_processed_section_uuids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    done: set[str] = set()
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


def _extract_system_user(messages: list[dict[str, Any]]) -> tuple[str, str]:
    system_parts: list[str] = []
    user_parts: list[str] = []
    for m in messages:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
    return "\n\n".join(system_parts), "\n\n".join(user_parts)


def _write_record(
    fh: Any,
    section_uuid: str,
    agreement_uuid: str | None,
    categories: list[str],
    error: str | None,
) -> None:
    fh.write(
        json.dumps(
            {
                "section_uuid": section_uuid,
                "agreement_uuid": agreement_uuid,
                "categories": categories,
                "error": error,
            },
            ensure_ascii=False,
        )
        + "\n"
    )
    fh.flush()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DATA_DIR / "batch.jsonl")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "batch_gold.out.jsonl")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--reasoning-effort", default="high",
                        choices=["minimal", "low", "medium", "high"])
    parser.add_argument("--token-budget", type=int, default=9_900_000,
                        help="Hard cap on cumulative usage.total_tokens. Stops before the next "
                             "request that would risk crossing this. Default: 9.9M (under the "
                             "user-stated 10M ceiling).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip section_uuids already present in --output (so re-running "
                             "after a crash or budget exhaustion doesn't redo work).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s :: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("generate_openai_gold")

    load_dotenv(BACKEND_ENV)
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        log.error("OPENAI_API_KEY not set (looked in env and %s)", BACKEND_ENV)
        return 2
    client = OpenAI(api_key=api_key)

    log.info("reading rows from %s", args.input)
    rows = _load_jsonl(args.input)
    log.info("loaded %d rows", len(rows))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    already_done: set[str] = set()
    open_mode = "w"
    if args.resume:
        already_done = _load_processed_section_uuids(args.output)
        if already_done:
            log.info("resume: %d section_uuids already present in %s; will skip",
                     len(already_done), args.output)
            open_mode = "a"

    usage_total = 0
    ok = 0
    failed = 0
    api_errors = 0
    skipped = 0
    parse_errors = 0
    t_start = time.monotonic()

    with args.output.open(open_mode, encoding="utf-8") as out_fh:
        for row_idx, row in enumerate(rows):
            sections = row["sections"]
            section_uuids = [s["section_uuid"] for s in sections]
            agreement_uuid_by_sid = {s["section_uuid"]: s.get("agreement_uuid") for s in sections}

            if args.resume and all(sid in already_done for sid in section_uuids):
                skipped += len(section_uuids)
                continue

            if usage_total >= args.token_budget:
                log.warning(
                    "token budget reached (%d >= %d); stopping at row %d/%d",
                    usage_total, args.token_budget, row_idx, len(rows),
                )
                break

            system_content, user_content = _extract_system_user(row["messages"])
            schema = _build_response_schema(section_uuids)

            try:
                resp = client.responses.create(
                    model=args.model,
                    reasoning={"effort": args.reasoning_effort},
                    instructions=system_content,
                    input=[{"role": "user", "content": user_content}],
                    text={
                        "format": {
                            "type": "json_schema",
                            "name": "section_category_assignments",
                            "strict": True,
                            "schema": schema,
                        }
                    },
                )
            except Exception as exc:
                api_errors += 1
                log.error("row %d API error: %s", row_idx, exc)
                for sid in section_uuids:
                    _write_record(out_fh, sid, agreement_uuid_by_sid.get(sid), [], "api_error")
                    failed += 1
                continue

            usage_total += getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0

            try:
                parsed = json.loads(resp.output_text)
                assignments = {
                    a["section_uuid"]: [str(c).strip() for c in a.get("categories", []) if str(c).strip()]
                    for a in parsed.get("assignments", [])
                }
                row_parse_error = False
            except (ValueError, KeyError) as exc:
                parse_errors += 1
                row_parse_error = True
                log.error("row %d parse error: %s :: head=%r", row_idx, exc, resp.output_text[:200])
                assignments = {}

            for sid in section_uuids:
                if sid in assignments:
                    _write_record(out_fh, sid, agreement_uuid_by_sid.get(sid),
                                  assignments[sid], None)
                    ok += 1
                else:
                    err = "parse_error" if row_parse_error else "missing_in_response"
                    _write_record(out_fh, sid, agreement_uuid_by_sid.get(sid), [], err)
                    failed += 1

            if (row_idx + 1) % 10 == 0 or row_idx + 1 == len(rows):
                elapsed = time.monotonic() - t_start
                log.info(
                    "progress: rows=%d/%d ok=%d failed=%d api_err=%d parse_err=%d "
                    "tokens=%d/%d (%.1f%%) elapsed=%.1fs (%.2fs/row)",
                    row_idx + 1, len(rows), ok, failed, api_errors, parse_errors,
                    usage_total, args.token_budget,
                    100.0 * usage_total / args.token_budget if args.token_budget else 0.0,
                    elapsed, elapsed / (row_idx + 1) if row_idx >= 0 else 0.0,
                )

    log.info(
        "done: ok=%d failed=%d api_errors=%d parse_errors=%d skipped(resume)=%d "
        "tokens_used=%d/%d elapsed=%.1fs",
        ok, failed, api_errors, parse_errors, skipped,
        usage_total, args.token_budget, time.monotonic() - t_start,
    )
    return 0 if failed == 0 and api_errors == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
