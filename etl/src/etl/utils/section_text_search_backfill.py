"""Backfill canonical text resumably for every current searchable section.

The destination table must be created before this command runs. Production
backfills should target an unindexed shadow table, validate coverage, build its
FULLTEXT index, and then swap tables.

Usage (from repo root):
    caffeinate -i etl/.venv/bin/python -m etl.utils.section_text_search_backfill \
        --schema pdx --target-table section_text_search_v2
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

from dotenv import load_dotenv

from etl.utils.db_env import build_engine_from_env, validate_schema_name
from etl.utils.section_text_search import (
    refresh_section_text_search,
    select_section_text_backfill_agreements,
    select_section_text_drifted_agreements,
)


_TABLE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_table_name(table_name: str) -> str:
    if not _TABLE_NAME_RE.fullmatch(table_name):
        raise ValueError(f"Invalid table name: {table_name!r}")
    return table_name


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--schema", default="pdx")
    _ = parser.add_argument("--target-table", default="section_text_search_v2")
    _ = parser.add_argument(
        "--env-file",
        type=Path,
        default=Path("backend/.env"),
        help="MariaDB environment file (default: backend/.env).",
    )
    _ = parser.add_argument("--agreement-batch-size", type=int, default=100)
    _ = parser.add_argument("--write-batch-size", type=int, default=1000)
    _ = parser.add_argument(
        "--drift-only",
        action="store_true",
        help=(
            "Refresh only agreements with missing, stale, or wrong-version target rows. "
            "Use for the final catch-up after building the shadow FULLTEXT index."
        ),
    )
    _ = parser.add_argument(
        "--after-agreement-uuid",
        default="",
        help="Resume strictly after this agreement UUID.",
    )
    _ = parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Stop after this many committed batches (useful for canaries).",
    )
    args = parser.parse_args(argv)
    args.schema = validate_schema_name(str(args.schema))
    args.target_table = _validate_table_name(str(args.target_table))
    if args.agreement_batch_size <= 0:
        parser.error("--agreement-batch-size must be positive")
    if args.write_batch_size <= 0:
        parser.error("--write-batch-size must be positive")
    if args.max_batches is not None and args.max_batches <= 0:
        parser.error("--max-batches must be positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    _ = load_dotenv(args.env_file, override=False)
    db = build_engine_from_env()
    engine = db.get_engine()
    cursor = str(args.after_agreement_uuid)
    batches = 0
    agreements_processed = 0
    sections_refreshed = 0

    while args.max_batches is None or batches < args.max_batches:
        with engine.begin() as conn:
            if args.drift_only:
                agreement_uuids = select_section_text_drifted_agreements(
                    conn,
                    args.schema,
                    table_name=args.target_table,
                    after_agreement_uuid=cursor,
                    limit=args.agreement_batch_size,
                )
            else:
                agreement_uuids = select_section_text_backfill_agreements(
                    conn,
                    args.schema,
                    after_agreement_uuid=cursor,
                    limit=args.agreement_batch_size,
                )
            if not agreement_uuids:
                break
            refreshed = refresh_section_text_search(
                conn,
                args.schema,
                agreement_uuids,
                table_name=args.target_table,
                write_batch_size=args.write_batch_size,
            )

        cursor = agreement_uuids[-1]
        batches += 1
        agreements_processed += len(agreement_uuids)
        sections_refreshed += refreshed
        print(
            "Committed batch "
            + f"{batches}: agreements={len(agreement_uuids)}, sections={refreshed}, "
            + f"resume_after={cursor}",
            flush=True,
        )

    print(
        "Done: "
        + f"batches={batches}, agreements={agreements_processed}, "
        + f"sections={sections_refreshed}, resume_after={cursor}",
        flush=True,
    )


if __name__ == "__main__":
    main()
