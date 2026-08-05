# pyright: reportAny=false
"""Loaders for the dedupe-guard labeled evaluation dataset.

The dataset is a clean, isolated artifact with two parts:

- ``etl/data/dedupe_eval/labels.json`` (committed): every labeled pair with an
  orientation-insensitive pair id, both sides' agreement metadata frozen at
  adjudication time, the label, and adjudication provenance.
- ``pdx.dedupe_eval_documents`` (local-only table): the full page text of every
  agreement referenced by labels.json, materialized once by
  ``etl.utils.build_dedupe_eval_corpus``. The committed manifest
  ``etl/data/dedupe_eval/corpus_manifest.json`` (uuid -> sha256) lets consumers
  detect a missing or drifted corpus and fail loudly.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

from sqlalchemy import text
from sqlalchemy.engine import Connection

DEDUPE_EVAL_DATA_DIR: Final[Path] = (
    Path(__file__).resolve().parents[3] / "data" / "dedupe_eval"
)
LABELS_PATH: Final[Path] = DEDUPE_EVAL_DATA_DIR / "labels.json"
CORPUS_MANIFEST_PATH: Final[Path] = DEDUPE_EVAL_DATA_DIR / "corpus_manifest.json"
DOCUMENTS_TABLE: Final[str] = "dedupe_eval_documents"

LABEL_DUPLICATE: Final[str] = "duplicate"
LABEL_NOT_DUPLICATE: Final[str] = "not_duplicate"

_REBUILD_HINT: Final[str] = (
    "run `etl/.venv/bin/python -m etl.utils.build_dedupe_eval_corpus` to "
    "(re)materialize the corpus"
)


def pair_identity(uuid_a: str, uuid_b: str) -> str:
    """Orientation-insensitive pair id: the two uuids sorted and joined."""
    lo, hi = sorted((uuid_a, uuid_b))
    return f"{lo}__{hi}"


def sha256_text(document_text: str) -> str:
    return hashlib.sha256(document_text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PairSide:
    agreement_uuid: str
    url: str
    filing_date: date | None
    target: str | None
    acquirer: str | None


@dataclass(frozen=True)
class EvalPair:
    pair_id: str
    is_dupe: bool
    batch: str
    loser: PairSide
    survivor: PairSide


def _parse_side(raw: Mapping[str, object]) -> PairSide:
    filing_raw = raw["filing_date"]
    return PairSide(
        agreement_uuid=str(raw["agreement_uuid"]),
        url=str(raw["url"]),
        filing_date=None if filing_raw is None else date.fromisoformat(str(filing_raw)),
        target=None if raw["target"] is None else str(raw["target"]),
        acquirer=None if raw["acquirer"] is None else str(raw["acquirer"]),
    )


def load_labels(path: Path = LABELS_PATH) -> list[EvalPair]:
    """Parse and validate labels.json; raises ValueError on any inconsistency."""
    raw = json.loads(path.read_text())
    pairs: list[EvalPair] = []
    seen_ids: set[str] = set()
    for entry in raw["pairs"]:
        label = str(entry["label"])
        if label not in {LABEL_DUPLICATE, LABEL_NOT_DUPLICATE}:
            raise ValueError(f"Unknown label {label!r} in {path}")
        loser = _parse_side(entry["loser"])
        survivor = _parse_side(entry["survivor"])
        if loser.agreement_uuid == survivor.agreement_uuid:
            raise ValueError(f"Pair {entry['pair_id']} has identical sides in {path}")
        pair_id = str(entry["pair_id"])
        expected_id = pair_identity(loser.agreement_uuid, survivor.agreement_uuid)
        if pair_id != expected_id:
            raise ValueError(
                f"pair_id {pair_id} does not match its sides"
                + f" (expected {expected_id}) in {path}"
            )
        if pair_id in seen_ids:
            raise ValueError(f"Duplicate pair {pair_id} in {path}")
        seen_ids.add(pair_id)
        pairs.append(
            EvalPair(
                pair_id=pair_id,
                is_dupe=label == LABEL_DUPLICATE,
                batch=str(entry["batch"]),
                loser=loser,
                survivor=survivor,
            )
        )
    counts = raw["counts"]
    positives = sum(1 for pair in pairs if pair.is_dupe)
    negatives = len(pairs) - positives
    if positives != int(counts["positive"]) or negatives != int(counts["negative"]):
        raise ValueError(
            f"Label counts drifted in {path}: header says"
            + f" {counts['positive']}/{counts['negative']},"
            + f" pairs hold {positives}/{negatives}"
        )
    return pairs


def load_corpus_manifest(path: Path = CORPUS_MANIFEST_PATH) -> dict[str, str]:
    """Parse corpus_manifest.json into uuid -> sha256, validating its total."""
    raw = json.loads(path.read_text())
    documents = {str(uuid): str(sha) for uuid, sha in raw["documents"].items()}
    if len(documents) != int(raw["total_documents"]):
        raise ValueError(
            f"Corpus manifest total_documents={raw['total_documents']} does not"
            + f" match {len(documents)} listed documents in {path}"
        )
    return documents


def missing_from_manifest(
    pairs: list[EvalPair], manifest: Mapping[str, str]
) -> list[str]:
    """Uuids referenced by labeled pairs but absent from the corpus manifest."""
    referenced = {
        uuid
        for pair in pairs
        for uuid in (pair.loser.agreement_uuid, pair.survivor.agreement_uuid)
    }
    return sorted(referenced - set(manifest))


def manifest_drift(
    expected: Mapping[str, str], actual: Mapping[str, str]
) -> list[str]:
    """Describe differences between the manifest and the materialized corpus.

    Returns human-readable drift lines; empty means the corpus covers the
    manifest exactly. Extra documents in ``actual`` are not drift.
    """
    problems: list[str] = []
    for uuid in sorted(expected):
        if uuid not in actual:
            problems.append(f"missing document {uuid}")
        elif actual[uuid] != expected[uuid]:
            problems.append(
                f"sha256 mismatch for {uuid}:"
                + f" expected {expected[uuid]}, found {actual[uuid]}"
            )
    return problems


def verify_corpus(
    conn: Connection, schema: str, manifest: Mapping[str, str]
) -> None:
    """Fail loudly unless every manifest document is materialized and unchanged."""
    rows = conn.execute(
        text(
            f"""
            SELECT agreement_uuid, sha256
            FROM {schema}.{DOCUMENTS_TABLE}
            """
        )
    ).fetchall()
    actual = {str(row[0]): str(row[1]) for row in rows}
    problems = manifest_drift(manifest, actual)
    if problems:
        details = "\n  ".join(problems)
        raise RuntimeError(
            f"dedupe eval corpus in {schema}.{DOCUMENTS_TABLE} is incomplete or"
            + f" drifted; {_REBUILD_HINT}:\n  {details}"
        )


def load_document(
    conn: Connection, schema: str, agreement_uuid: str, expected_sha256: str
) -> tuple[str, int]:
    """Fetch one document's text and page_count, verifying its content hash."""
    row = conn.execute(
        text(
            f"""
            SELECT `text`, page_count
            FROM {schema}.{DOCUMENTS_TABLE}
            WHERE agreement_uuid = :agreement_uuid
            """
        ),
        {"agreement_uuid": agreement_uuid},
    ).fetchone()
    if row is None:
        raise RuntimeError(
            f"Document {agreement_uuid} is missing from"
            + f" {schema}.{DOCUMENTS_TABLE}; {_REBUILD_HINT}"
        )
    document_text = str(row[0])
    actual_sha = sha256_text(document_text)
    if actual_sha != expected_sha256:
        raise RuntimeError(
            f"Document {agreement_uuid} in {schema}.{DOCUMENTS_TABLE} drifted"
            + f" from the committed manifest (sha256 {actual_sha} !="
            + f" {expected_sha256}); {_REBUILD_HINT}"
        )
    return document_text, int(row[1])
