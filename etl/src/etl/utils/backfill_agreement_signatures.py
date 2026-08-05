# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false, reportMissingTypeStubs=false
"""Backfill agreement content signatures from locally stored page text.

Computes full-text signatures and cover identity for the existing corpus from
the pdx.pages table (processed_page_content ordered by page_order) so the
staging dedupe guard has a complete corpus to compare against. No EDGAR
requests are made.

Usage (from repo root):
    caffeinate -i etl/.venv/bin/python -m etl.utils.backfill_agreement_signatures \
        --schema pdx --create-tables

Idempotent: --create-tables uses CREATE TABLE IF NOT EXISTS, and signatures are
upserted, so re-runs are safe. By default only agreements without a stored
signature are processed; use --recompute-all to refresh everything.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Connection

if __package__ in {None, ""}:
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from etl.domain.dedupe_signatures import (
    compute_document_signature,
    extract_cover_identity,
)
from etl.utils.agreement_signature_store import (
    StoredSignature,
    create_signature_tables,
    mark_reconciled,
    upsert_signatures,
)
from etl.utils.reset_stuck_agreements import (
    build_engine_from_env,
    validate_schema_name,
)

_BATCH_SIZE = 25


def _select_backfill_targets(
    conn: Connection, schema: str, *, only_missing: bool, all_sources: bool
) -> list[tuple[str, str]]:
    """Agreements with page text, ordered for deterministic progress.

    Defaults to source='edgar' because the staging dedupe guard only reconciles
    edgar agreements (URL-derived identity); --all-sources widens the scope.
    """
    missing_filter = (
        f"""
        AND NOT EXISTS (
            SELECT 1 FROM {schema}.agreement_signatures s
            WHERE s.agreement_uuid = a.agreement_uuid
        )
        """
        if only_missing
        else ""
    )
    source_filter = "" if all_sources else "AND a.source <=> 'edgar'"
    rows = conn.execute(
        text(
            f"""
            SELECT a.agreement_uuid, a.url
            FROM {schema}.agreements a
            WHERE EXISTS (
                SELECT 1 FROM {schema}.pages p
                WHERE p.agreement_uuid = a.agreement_uuid
            )
            {missing_filter}
            {source_filter}
            ORDER BY a.agreement_uuid
            """
        )
    ).fetchall()
    return [(str(row[0]), str(row[1])) for row in rows]


def load_agreement_text_from_pages(
    conn: Connection, schema: str, agreement_uuid: str, *, pages_table: str = "pages"
) -> tuple[str, int]:
    """Rebuild the rendered agreement text from stored page content.

    Mirrors _render_agreement_text_and_page_count in etl.domain.a_staging:
    text is non-empty stripped pages joined with blank lines, and page_count is
    the number of page rows — which equals the staging-side split count because
    split_to_pages drops empty fragments before pages are persisted, so the
    <50%-content survivor guard sees the same page_count on both paths.
    """
    rows = conn.execute(
        text(
            f"""
            SELECT processed_page_content
            FROM {schema}.{pages_table}
            WHERE agreement_uuid = :agreement_uuid
            ORDER BY page_order ASC
            """
        ),
        {"agreement_uuid": agreement_uuid},
    ).fetchall()
    pages = [str(row[0] or "").strip() for row in rows]
    return "\n\n".join(page for page in pages if page), len(pages)


def compute_stored_signature_from_pages(
    conn: Connection,
    schema: str,
    agreement_uuid: str,
    url: str,
    *,
    pages_table: str = "pages",
) -> StoredSignature:
    rendered_text, page_count = load_agreement_text_from_pages(
        conn, schema, agreement_uuid, pages_table=pages_table
    )
    return StoredSignature(
        agreement_uuid=agreement_uuid,
        url=url,
        signature=compute_document_signature(rendered_text, page_count=page_count),
        cover=extract_cover_identity(rendered_text),
    )


def _clear_satisfied_pending(conn: Connection, schema: str) -> int:
    result = conn.execute(
        text(
            f"""
            DELETE p
            FROM {schema}.agreement_signature_pending p
            JOIN {schema}.agreement_signatures s
                ON s.agreement_uuid = p.agreement_uuid
            """
        )
    )
    return int(result.rowcount or 0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--schema", default="pdx")
    _ = parser.add_argument(
        "--create-tables",
        action="store_true",
        help="Idempotently create the signature tables before backfilling.",
    )
    _ = parser.add_argument(
        "--recompute-all",
        action="store_true",
        help="Recompute signatures even for agreements that already have one.",
    )
    _ = parser.add_argument(
        "--all-sources",
        action="store_true",
        help="Include non-edgar agreements (the dedupe guard only scans edgar).",
    )
    _ = parser.add_argument(
        "--leave-unreconciled",
        action="store_true",
        help=(
            "Leave backfilled signatures unreconciled so the next staging run "
            "sweeps the whole backlog through the dedupe guard. Default marks "
            "them reconciled: the historical backlog goes through the manual "
            "audit-then-delete process, not automatic deletion."
        ),
    )
    _ = parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args(argv)
    args.schema = validate_schema_name(args.schema)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    db = build_engine_from_env()
    engine = db.get_engine()

    if args.create_tables:
        with engine.begin() as conn:
            create_signature_tables(conn, args.schema)
        print(f"Ensured signature tables exist in schema {args.schema}.")

    with engine.begin() as conn:
        targets = _select_backfill_targets(
            conn,
            args.schema,
            only_missing=not args.recompute_all,
            all_sources=args.all_sources,
        )
    if args.limit is not None:
        targets = targets[: args.limit]
    print(f"Backfilling signatures for {len(targets)} agreements.")

    processed = 0
    failed: list[str] = []
    for batch_start in range(0, len(targets), _BATCH_SIZE):
        batch = targets[batch_start : batch_start + _BATCH_SIZE]
        rows: list[StoredSignature] = []
        with engine.begin() as conn:
            for agreement_uuid, url in batch:
                try:
                    rows.append(
                        compute_stored_signature_from_pages(
                            conn, args.schema, agreement_uuid, url
                        )
                    )
                except Exception as exc:
                    failed.append(agreement_uuid)
                    print(f"WARNING: failed to compute signature for {url}: {exc}")
            upsert_signatures(conn, args.schema, rows)
            if not args.leave_unreconciled:
                mark_reconciled(conn, args.schema, [row.agreement_uuid for row in rows])
        processed += len(rows)
        if processed % 500 < _BATCH_SIZE:
            print(f"Progress: {processed}/{len(targets)}")

    with engine.begin() as conn:
        cleared = _clear_satisfied_pending(conn, args.schema)
    print(
        f"Done: {processed} signatures upserted, {len(failed)} failures, "
        f"{cleared} satisfied pending entries cleared."
    )
    if failed:
        print("Failed agreement_uuids: " + ", ".join(failed))


if __name__ == "__main__":
    main()
