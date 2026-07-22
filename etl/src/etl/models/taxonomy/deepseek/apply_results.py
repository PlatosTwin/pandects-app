"""
Apply DeepSeek taxonomy inference results back into the local sections table.

Reads the JSONL produced by batch_inference.py (lines like
{"section_uuid": ..., "categories": [...], "error": null}) and writes the
JSON-encoded list of standard_ids to sections.section_standard_id_gold_label
(matching how h_taxonomy_asset writes labels in LLM mode).

Run from repo root, e.g.:
  caffeinate -i etl/.venv/bin/python \
      etl/src/etl/models/taxonomy/deepseek/apply_results.py \
      --input data/smoke_2026-04-25.out.jsonl \
      --model-name deepseek-v4-pro \
      --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import bindparam, create_engine, text


REPO_ROOT = Path(__file__).resolve().parents[6]
DEEPSEEK_DIR = Path(__file__).resolve().parent
DATA_DIR = DEEPSEEK_DIR / "data"
BACKEND_ENV = REPO_ROOT / "backend" / ".env"


def _build_engine() -> tuple[Any, str]:
    load_dotenv(BACKEND_ENV)
    user = os.environ["MARIADB_USER"]
    password = os.environ["MARIADB_PASSWORD"]
    host = os.environ["MARIADB_HOST"]
    database = os.environ["MARIADB_DATABASE"]
    url = f"mysql+mysqldb://{user}:{password}@{host}/{database}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True), database


def _serialize_labels(categories: list[str]) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for cat in categories:
        cleaned = (cat or "").strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            out.append(cleaned)
    return json.dumps(out, ensure_ascii=False)


def _load_results(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_num, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logging.warning("skipping malformed line %d: %s", line_num, exc)
    return rows


def _resolve_input(arg: str) -> Path:
    p = Path(arg)
    if not p.is_absolute():
        if (DATA_DIR / p).exists():
            p = DATA_DIR / p
        else:
            p = Path.cwd() / p
    return p


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True,
                        help="Path to results JSONL (relative paths are resolved against data/).")
    parser.add_argument("--model-name", default="deepseek-v4-pro",
                        help="Value to write to sections.gold_label_model.")
    parser.add_argument("--include-failed", action="store_true",
                        help="Apply rows that have an 'error' set (default: skip).")
    parser.add_argument("--include-empty", action="store_true",
                        help="Write []-payloads when categories is empty (default: skip).")
    parser.add_argument("--batch-size", type=int, default=250)
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and summarize but do not write to the DB.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s :: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("apply_results")

    in_path = _resolve_input(args.input)
    log.info("reading results from %s", in_path)
    rows = _load_results(in_path)
    log.info("loaded %d result rows", len(rows))

    updates: list[dict[str, str]] = []
    skipped_error = 0
    skipped_empty = 0
    for row in rows:
        if row.get("error") and not args.include_failed:
            skipped_error += 1
            continue
        sid = row.get("section_uuid")
        if not isinstance(sid, str):
            log.warning("row missing section_uuid: %r", row)
            continue
        cats = row.get("categories") or []
        if not cats and not args.include_empty:
            skipped_empty += 1
            continue
        updates.append(
            {
                "section_uuid": sid,
                "gold_label_payload": _serialize_labels(list(cats)),
                "model_name": args.model_name,
            }
        )

    log.info(
        "prepared %d updates (skipped %d errored, %d empty)",
        len(updates),
        skipped_error,
        skipped_empty,
    )
    if not updates:
        log.info("nothing to apply")
        return 0

    if args.dry_run:
        log.info("dry-run: showing first 5 updates")
        for u in updates[:5]:
            log.info("  %s -> %s", u["section_uuid"], u["gold_label_payload"])
        return 0

    engine, schema = _build_engine()
    update_sql = text(
        f"""
        UPDATE {schema}.sections
        SET section_standard_id_gold_label = :gold_label_payload,
            gold_label_model = :model_name
        WHERE section_uuid = :section_uuid
        """
    )
    affected_rows = 0
    with engine.begin() as conn:
        for start in range(0, len(updates), args.batch_size):
            chunk = updates[start : start + args.batch_size]
            result = conn.execute(update_sql, chunk)
            affected_rows += int(result.rowcount or 0)
            log.info(
                "wrote chunk %d-%d (rowcount=%s)",
                start,
                start + len(chunk),
                result.rowcount,
            )

        # Sanity check matching uuids in DB.
        section_uuids = [u["section_uuid"] for u in updates]
        existing = conn.execute(
            text(
                f"""
                SELECT COUNT(*) AS c FROM {schema}.sections
                WHERE section_uuid IN :section_uuids
                """
            ).bindparams(bindparam("section_uuids", expanding=True)),
            {"section_uuids": tuple(section_uuids)},
        ).scalar()
        log.info("section_uuids matched in DB: %s/%s", existing, len(section_uuids))

    log.info("done: total rows updated=%d", affected_rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
