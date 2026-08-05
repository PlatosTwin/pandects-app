# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false
"""Materialize the dedupe-eval text corpus into pdx.dedupe_eval_documents.

Reads every agreement referenced by etl/data/dedupe_eval/labels.json and copies
its rendered page text into a dedicated local table so the eval no longer
depends on the pdx.dedupe_bak_20260805{,b,c}_pages disaster-recovery backups.
Text is resolved from pdx.pages and those three backup generations; a document
missing from all of them is a hard failure. Re-runs are idempotent (upsert by
agreement_uuid).

By default the result is verified against the committed manifest
etl/data/dedupe_eval/corpus_manifest.json and any drift is a hard failure;
--write-manifest regenerates the manifest instead (only for intentional
dataset changes -- commit the result).

Usage (from repo root):
    caffeinate -i etl/.venv/bin/python -m etl.utils.build_dedupe_eval_corpus
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

from sqlalchemy import text
from sqlalchemy.engine import Connection

if __package__ in {None, ""}:
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from etl.utils.backfill_agreement_signatures import load_agreement_text_from_pages
from etl.utils.db_env import build_engine_from_env, validate_schema_name
from etl.utils.dedupe_eval_dataset import (
    CORPUS_MANIFEST_PATH,
    DOCUMENTS_TABLE,
    load_corpus_manifest,
    load_labels,
    manifest_drift,
    sha256_text,
)

# Live table first, then the three backup generations, matching the legacy
# eval's fallback order for survivor/negative text.
_PAGES_FIRST: Final[tuple[str, ...]] = (
    "pages",
    "dedupe_bak_20260805_pages",
    "dedupe_bak_20260805b_pages",
    "dedupe_bak_20260805c_pages",
)
# Deleted losers of true-dupe pairs live in the backups; the legacy eval
# preferred backup text for them, so the corpus does too.
_BACKUP_FIRST: Final[tuple[str, ...]] = _PAGES_FIRST[1:] + _PAGES_FIRST[:1]


@dataclass(frozen=True)
class MaterializedDocument:
    agreement_uuid: str
    text: str
    page_count: int
    source: str
    sha256: str


def _resolve_document(
    conn: Connection, schema: str, agreement_uuid: str, tables: tuple[str, ...]
) -> MaterializedDocument | None:
    for table in tables:
        rendered, page_count = load_agreement_text_from_pages(
            conn, schema, agreement_uuid, pages_table=table
        )
        if rendered:
            return MaterializedDocument(
                agreement_uuid=agreement_uuid,
                text=rendered,
                page_count=page_count,
                source=table,
                sha256=sha256_text(rendered),
            )
    return None


def _collect_documents(conn: Connection, schema: str) -> list[MaterializedDocument]:
    pairs = load_labels()
    backup_first: set[str] = set()
    pages_first: set[str] = set()
    for pair in pairs:
        if pair.is_dupe:
            backup_first.add(pair.loser.agreement_uuid)
        else:
            pages_first.add(pair.loser.agreement_uuid)
        pages_first.add(pair.survivor.agreement_uuid)

    documents: dict[str, MaterializedDocument] = {}
    missing: list[str] = []
    conflicting: list[str] = []
    for agreement_uuid in sorted(backup_first | pages_first):
        resolved: list[MaterializedDocument] = []
        if agreement_uuid in backup_first:
            doc = _resolve_document(conn, schema, agreement_uuid, _BACKUP_FIRST)
            if doc is not None:
                resolved.append(doc)
        if agreement_uuid in pages_first:
            doc = _resolve_document(conn, schema, agreement_uuid, _PAGES_FIRST)
            if doc is not None:
                resolved.append(doc)
        if not resolved:
            missing.append(agreement_uuid)
            continue
        # A uuid used both as a dupe loser and as a survivor/negative side must
        # resolve to the same text either way; otherwise one row per uuid
        # cannot reproduce the legacy per-role loading.
        if len(resolved) == 2 and (
            resolved[0].sha256 != resolved[1].sha256
            or resolved[0].page_count != resolved[1].page_count
        ):
            conflicting.append(agreement_uuid)
            continue
        documents[agreement_uuid] = resolved[0]
    if missing:
        raise RuntimeError(
            "No page text found in pages or any dedupe_bak_*_pages table for: "
            + ", ".join(missing)
        )
    if conflicting:
        raise RuntimeError(
            "Backup-preferred and live-preferred text disagree for: "
            + ", ".join(conflicting)
        )
    return [documents[uuid] for uuid in sorted(documents)]


def _create_documents_table(conn: Connection, schema: str) -> None:
    _ = conn.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.{DOCUMENTS_TABLE} (
                agreement_uuid CHAR(36) NOT NULL,
                `text` LONGTEXT NOT NULL,
                page_count INT NOT NULL,
                source VARCHAR(64) NOT NULL,
                sha256 CHAR(64) NOT NULL,
                materialized_at DATETIME NOT NULL DEFAULT UTC_TIMESTAMP(),
                PRIMARY KEY (agreement_uuid)
            ) DEFAULT CHARSET = utf8mb4
            """
        )
    )


def _upsert_documents(
    conn: Connection, schema: str, documents: list[MaterializedDocument]
) -> None:
    for doc in documents:
        _ = conn.execute(
            text(
                f"""
                INSERT INTO {schema}.{DOCUMENTS_TABLE}
                    (agreement_uuid, `text`, page_count, source, sha256,
                     materialized_at)
                VALUES
                    (:agreement_uuid, :text, :page_count, :source, :sha256,
                     UTC_TIMESTAMP())
                ON DUPLICATE KEY UPDATE
                    `text` = VALUES(`text`),
                    page_count = VALUES(page_count),
                    source = VALUES(source),
                    sha256 = VALUES(sha256),
                    materialized_at = UTC_TIMESTAMP()
                """
            ),
            {
                "agreement_uuid": doc.agreement_uuid,
                "text": doc.text,
                "page_count": doc.page_count,
                "source": doc.source,
                "sha256": doc.sha256,
            },
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--schema", default="pdx")
    _ = parser.add_argument(
        "--write-manifest",
        action="store_true",
        help=(
            "Regenerate corpus_manifest.json from the materialized documents "
            "instead of verifying against the committed manifest."
        ),
    )
    args = parser.parse_args(argv)
    args.schema = validate_schema_name(args.schema)
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    db = build_engine_from_env()
    with db.get_engine().begin() as conn:
        documents = _collect_documents(conn, args.schema)
        _create_documents_table(conn, args.schema)
        _upsert_documents(conn, args.schema, documents)

    by_source: dict[str, int] = {}
    for doc in documents:
        by_source[doc.source] = by_source.get(doc.source, 0) + 1
    print(
        f"Materialized {len(documents)} documents into"
        + f" {args.schema}.{DOCUMENTS_TABLE}: "
        + ", ".join(f"{count} from {source}" for source, count in sorted(by_source.items()))
    )

    shas = {doc.agreement_uuid: doc.sha256 for doc in documents}
    if args.write_manifest:
        manifest = {
            "dataset": "dedupe_eval_corpus",
            "generated_on": date.today().isoformat(),
            "total_documents": len(shas),
            "documents": {uuid: shas[uuid] for uuid in sorted(shas)},
        }
        _ = CORPUS_MANIFEST_PATH.write_text(json.dumps(manifest, indent=1) + "\n")
        print(f"Wrote manifest {CORPUS_MANIFEST_PATH}")
    else:
        expected = load_corpus_manifest()
        problems = manifest_drift(expected, shas)
        extra = sorted(set(shas) - set(expected))
        problems.extend(f"document {uuid} not in manifest" for uuid in extra)
        if problems:
            details = "\n  ".join(problems)
            raise RuntimeError(
                "Materialized corpus does not match the committed manifest"
                + f" {CORPUS_MANIFEST_PATH}:\n  {details}"
            )
        print("Materialized corpus matches the committed manifest.")


if __name__ == "__main__":
    main()
