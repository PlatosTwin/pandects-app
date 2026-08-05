# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportMissingTypeStubs=false
"""Persistence for agreement content signatures (duplicate-agreement guard).

Tables are created by the migration
etl/sql/migrations/20260805_agreement_signatures.sql; assets assert their
existence and never issue DDL. `create_signature_tables` mirrors that DDL for
the backfill utility's --create-tables flag.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final

import numpy as np
from datasketch import MinHash
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection

from etl.domain.dedupe_signatures import CoverIdentity, DocumentSignature

SIGNATURES_TABLE: Final[str] = "agreement_signatures"
PENDING_TABLE: Final[str] = "agreement_signature_pending"
REVIEW_TABLE: Final[str] = "agreement_dedupe_review"
SIGNATURE_TABLE_NAMES: Final[tuple[str, ...]] = (
    SIGNATURES_TABLE,
    PENDING_TABLE,
    REVIEW_TABLE,
)

_CREATE_TABLE_DDL: Final[tuple[str, ...]] = (
    f"""
    CREATE TABLE IF NOT EXISTS {SIGNATURES_TABLE} (
        agreement_uuid CHAR(36) NOT NULL PRIMARY KEY,
        url VARCHAR(512) NOT NULL,
        content_fingerprint CHAR(64) NOT NULL,
        minhash_json MEDIUMTEXT NOT NULL,
        shingle_count INT NOT NULL,
        char_count BIGINT NOT NULL,
        page_count INT NULL,
        dated_as_of DATE NULL,
        party_tokens_json TEXT NULL,
        amends_and_restates TINYINT(1) NOT NULL DEFAULT 0,
        computed_at DATETIME NOT NULL DEFAULT UTC_TIMESTAMP(),
        KEY idx_agreement_signatures_fingerprint (content_fingerprint),
        KEY idx_agreement_signatures_dated_as_of (dated_as_of)
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {PENDING_TABLE} (
        agreement_uuid CHAR(36) NOT NULL PRIMARY KEY,
        url VARCHAR(512) NOT NULL,
        reason VARCHAR(255) NOT NULL,
        attempts INT NOT NULL DEFAULT 0,
        first_recorded_at DATETIME NOT NULL DEFAULT UTC_TIMESTAMP(),
        last_attempt_at DATETIME NULL,
        last_error TEXT NULL
    )
    """,
    f"""
    CREATE TABLE IF NOT EXISTS {REVIEW_TABLE} (
        review_id BIGINT NOT NULL AUTO_INCREMENT PRIMARY KEY,
        created_at DATETIME NOT NULL DEFAULT UTC_TIMESTAMP(),
        new_agreement_uuid CHAR(36) NOT NULL,
        new_url VARCHAR(512) NOT NULL,
        existing_agreement_uuid CHAR(36) NOT NULL,
        existing_url VARCHAR(512) NOT NULL,
        jaccard DOUBLE NULL,
        containment DOUBLE NULL,
        fingerprint_match TINYINT(1) NOT NULL DEFAULT 0,
        reason VARCHAR(255) NOT NULL,
        resolved TINYINT(1) NOT NULL DEFAULT 0,
        UNIQUE KEY uq_agreement_dedupe_review_pair (new_agreement_uuid, existing_agreement_uuid)
    )
    """,
)


@dataclass(frozen=True)
class StoredSignature:
    agreement_uuid: str
    url: str
    signature: DocumentSignature
    cover: CoverIdentity


@dataclass(frozen=True)
class PendingSignature:
    agreement_uuid: str
    url: str
    reason: str
    attempts: int


def create_signature_tables(conn: Connection, schema: str) -> None:
    """Idempotently create the signature tables (backfill utility only)."""
    for ddl in _CREATE_TABLE_DDL:
        qualified = ddl.replace(
            "CREATE TABLE IF NOT EXISTS ", f"CREATE TABLE IF NOT EXISTS {schema}."
        )
        _ = conn.execute(text(qualified))


def serialize_minhash(minhash: MinHash) -> str:
    return json.dumps([int(value) for value in minhash.hashvalues.tolist()])


def deserialize_minhash(raw: str) -> MinHash:
    hashvalues = np.array([int(value) for value in json.loads(raw)], dtype=np.uint64)
    minhash = MinHash(num_perm=len(hashvalues))
    minhash.hashvalues = hashvalues
    return minhash


def upsert_signatures(
    conn: Connection, schema: str, rows: Sequence[StoredSignature]
) -> None:
    if not rows:
        return
    sql = text(
        f"""
        REPLACE INTO {schema}.{SIGNATURES_TABLE} (
            agreement_uuid,
            url,
            content_fingerprint,
            minhash_json,
            shingle_count,
            char_count,
            page_count,
            dated_as_of,
            party_tokens_json,
            amends_and_restates
        ) VALUES (
            :agreement_uuid,
            :url,
            :content_fingerprint,
            :minhash_json,
            :shingle_count,
            :char_count,
            :page_count,
            :dated_as_of,
            :party_tokens_json,
            :amends_and_restates
        )
        """
    )
    _ = conn.execute(
        sql,
        [
            {
                "agreement_uuid": row.agreement_uuid,
                "url": row.url,
                "content_fingerprint": row.signature.content_fingerprint,
                "minhash_json": serialize_minhash(row.signature.minhash),
                "shingle_count": row.signature.shingle_count,
                "char_count": row.signature.char_count,
                "page_count": row.signature.page_count,
                "dated_as_of": row.cover.dated_as_of,
                "party_tokens_json": json.dumps(sorted(row.cover.party_tokens)),
                "amends_and_restates": int(row.cover.amends_and_restates),
            }
            for row in rows
        ],
    )


def load_all_signatures(conn: Connection, schema: str) -> list[StoredSignature]:
    rows = conn.execute(
        text(
            f"""
            SELECT agreement_uuid, url, content_fingerprint, minhash_json,
                   shingle_count, char_count, page_count, dated_as_of,
                   party_tokens_json, amends_and_restates
            FROM {schema}.{SIGNATURES_TABLE}
            """
        )
    ).mappings().all()
    stored: list[StoredSignature] = []
    for row in rows:
        dated_as_of_raw = row["dated_as_of"]
        dated_as_of = dated_as_of_raw if isinstance(dated_as_of_raw, date) else None
        party_tokens_raw = row["party_tokens_json"]
        party_tokens: frozenset[str] = frozenset()
        if party_tokens_raw:
            party_tokens = frozenset(
                str(token) for token in json.loads(str(party_tokens_raw))
            )
        stored.append(
            StoredSignature(
                agreement_uuid=str(row["agreement_uuid"]),
                url=str(row["url"]),
                signature=DocumentSignature(
                    content_fingerprint=str(row["content_fingerprint"]),
                    minhash=deserialize_minhash(str(row["minhash_json"])),
                    shingle_count=int(row["shingle_count"]),
                    char_count=int(row["char_count"]),
                    page_count=int(row["page_count"] or 0),
                ),
                cover=CoverIdentity(
                    dated_as_of=dated_as_of,
                    party_tokens=party_tokens,
                    amends_and_restates=bool(row["amends_and_restates"]),
                ),
            )
        )
    return stored


def delete_signatures(conn: Connection, schema: str, agreement_uuids: Iterable[str]) -> None:
    target = tuple(sorted({uuid for uuid in agreement_uuids if uuid}))
    if not target:
        return
    sql = text(
        f"DELETE FROM {schema}.{SIGNATURES_TABLE} WHERE agreement_uuid IN :agreement_uuids"
    ).bindparams(bindparam("agreement_uuids", expanding=True))
    _ = conn.execute(sql, {"agreement_uuids": target})


def record_pending(
    conn: Connection,
    schema: str,
    *,
    agreement_uuid: str,
    url: str,
    reason: str,
    error: str | None = None,
) -> None:
    """Record (or bump) a pending-reconciliation entry so a later run retries."""
    sql = text(
        f"""
        INSERT INTO {schema}.{PENDING_TABLE}
            (agreement_uuid, url, reason, attempts, last_attempt_at, last_error)
        VALUES (:agreement_uuid, :url, :reason, 1, UTC_TIMESTAMP(), :error)
        ON DUPLICATE KEY UPDATE
            attempts = attempts + 1,
            reason = VALUES(reason),
            last_attempt_at = UTC_TIMESTAMP(),
            last_error = VALUES(last_error)
        """
    )
    _ = conn.execute(
        sql,
        {"agreement_uuid": agreement_uuid, "url": url, "reason": reason, "error": error},
    )


def clear_pending(conn: Connection, schema: str, agreement_uuids: Iterable[str]) -> None:
    target = tuple(sorted({uuid for uuid in agreement_uuids if uuid}))
    if not target:
        return
    sql = text(
        f"DELETE FROM {schema}.{PENDING_TABLE} WHERE agreement_uuid IN :agreement_uuids"
    ).bindparams(bindparam("agreement_uuids", expanding=True))
    _ = conn.execute(sql, {"agreement_uuids": target})


def load_pending(conn: Connection, schema: str, *, limit: int) -> list[PendingSignature]:
    rows = conn.execute(
        text(
            f"""
            SELECT agreement_uuid, url, reason, attempts
            FROM {schema}.{PENDING_TABLE}
            ORDER BY attempts ASC, first_recorded_at ASC
            LIMIT :limit
            """
        ),
        {"limit": limit},
    ).mappings().all()
    return [
        PendingSignature(
            agreement_uuid=str(row["agreement_uuid"]),
            url=str(row["url"]),
            reason=str(row["reason"]),
            attempts=int(row["attempts"]),
        )
        for row in rows
    ]


def insert_review(
    conn: Connection,
    schema: str,
    *,
    new_agreement_uuid: str,
    new_url: str,
    existing_agreement_uuid: str,
    existing_url: str,
    jaccard: float | None,
    containment: float | None,
    fingerprint_match: bool,
    reason: str,
) -> None:
    sql = text(
        f"""
        INSERT IGNORE INTO {schema}.{REVIEW_TABLE} (
            new_agreement_uuid, new_url, existing_agreement_uuid, existing_url,
            jaccard, containment, fingerprint_match, reason
        ) VALUES (
            :new_agreement_uuid, :new_url, :existing_agreement_uuid, :existing_url,
            :jaccard, :containment, :fingerprint_match, :reason
        )
        """
    )
    _ = conn.execute(
        sql,
        {
            "new_agreement_uuid": new_agreement_uuid,
            "new_url": new_url,
            "existing_agreement_uuid": existing_agreement_uuid,
            "existing_url": existing_url,
            "jaccard": jaccard,
            "containment": containment,
            "fingerprint_match": int(fingerprint_match),
            "reason": reason,
        },
    )
