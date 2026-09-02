from __future__ import annotations

import hashlib
from collections.abc import Sequence

from sqlalchemy import bindparam, inspect, text
from sqlalchemy.engine import Connection

from etl.domain.g_sections import build_section_search_text


_REQUIRED_COLUMNS = frozenset(
    {
        "section_uuid",
        "agreement_uuid",
        "xml_version",
        "source_xml_sha256",
        "normalized_text",
    }
)
_DEFAULT_WRITE_BATCH_SIZE = 1000
_DEFAULT_AGREEMENT_QUERY_BATCH_SIZE = 100


def _qualified_table(schema: str, table_name: str) -> str:
    if not schema:
        return table_name
    return f"{schema}.{table_name}"


def _assert_section_text_search_schema(
    conn: Connection,
    schema: str,
    table_name: str,
) -> None:
    table_schema = schema or None
    columns = {
        str(column["name"])
        for column in inspect(conn).get_columns(table_name, schema=table_schema)
    }
    missing = sorted(_REQUIRED_COLUMNS - columns)
    if missing:
        qualified_table = _qualified_table(schema, table_name)
        raise RuntimeError(
            f"{qualified_table} is missing required columns: {', '.join(missing)}. "
            + "Apply the section text search schema before refreshing it."
        )


def _upsert_sql(conn: Connection, qualified_table: str):
    columns_and_values = f"""
        {qualified_table} (
            section_uuid,
            agreement_uuid,
            xml_version,
            source_xml_sha256,
            normalized_text
        ) VALUES (
            :section_uuid,
            :agreement_uuid,
            :xml_version,
            :source_xml_sha256,
            :normalized_text
        )
    """
    if conn.dialect.name == "sqlite":
        return text(
            f"""
            INSERT INTO {columns_and_values}
            ON CONFLICT(section_uuid) DO UPDATE SET
                agreement_uuid = excluded.agreement_uuid,
                xml_version = excluded.xml_version,
                source_xml_sha256 = excluded.source_xml_sha256,
                normalized_text = excluded.normalized_text
            WHERE agreement_uuid <> excluded.agreement_uuid
               OR xml_version IS NOT excluded.xml_version
               OR source_xml_sha256 IS NOT excluded.source_xml_sha256
               OR normalized_text <> excluded.normalized_text
            """
        )
    return text(
        f"""
        INSERT INTO {columns_and_values}
        ON DUPLICATE KEY UPDATE
            agreement_uuid = IF(
                NOT (agreement_uuid <=> VALUES(agreement_uuid)),
                VALUES(agreement_uuid), agreement_uuid
            ),
            xml_version = IF(
                NOT (xml_version <=> VALUES(xml_version)),
                VALUES(xml_version), xml_version
            ),
            source_xml_sha256 = IF(
                NOT (source_xml_sha256 <=> VALUES(source_xml_sha256)),
                VALUES(source_xml_sha256), source_xml_sha256
            ),
            normalized_text = IF(
                NOT (normalized_text <=> VALUES(normalized_text)),
                VALUES(normalized_text), normalized_text
            )
        """
    )


def _delete_stale_rows(
    conn: Connection,
    *,
    section_text_table: str,
    sections_table: str,
    xml_table: str,
    agreement_uuids: Sequence[str],
) -> None:
    _ = conn.execute(
        text(
            f"""
            DELETE FROM {section_text_table}
            WHERE agreement_uuid IN :agreement_uuids
              AND NOT EXISTS (
                  SELECT 1
                  FROM {sections_table} s
                  JOIN {xml_table} x
                    ON x.agreement_uuid = s.agreement_uuid
                   AND x.version = s.xml_version
                  WHERE s.section_uuid = {section_text_table}.section_uuid
                    AND s.agreement_uuid = {section_text_table}.agreement_uuid
                    AND x.latest = 1
                    AND (x.status IS NULL OR x.status = 'verified')
              )
            """
        ).bindparams(bindparam("agreement_uuids", expanding=True)),
        {"agreement_uuids": tuple(agreement_uuids)},
    )


def refresh_section_text_search(
    conn: Connection,
    schema: str,
    agreement_uuids: Sequence[str],
    *,
    table_name: str = "section_text_search",
    write_batch_size: int = _DEFAULT_WRITE_BATCH_SIZE,
    agreement_query_batch_size: int = _DEFAULT_AGREEMENT_QUERY_BATCH_SIZE,
) -> int:
    """Synchronize searchable text for the latest eligible section versions.

    Work is bounded to the supplied agreements. Source reads and stale-row deletes
    operate on bounded agreement sets to avoid per-agreement database round trips,
    while writes are chunked independently. The caller owns the transaction, so
    text upserts and stale-row deletion commit atomically with section extraction.
    """
    if write_batch_size <= 0:
        raise ValueError("write_batch_size must be positive.")
    if agreement_query_batch_size <= 0:
        raise ValueError("agreement_query_batch_size must be positive.")

    target_uuids = tuple(sorted({uuid for uuid in agreement_uuids if uuid}))
    if not target_uuids:
        return 0

    _assert_section_text_search_schema(conn, schema, table_name)

    section_text_table = _qualified_table(schema, table_name)
    sections_table = _qualified_table(schema, "sections")
    xml_table = _qualified_table(schema, "xml")
    source_sql = text(
        f"""
        SELECT
            s.section_uuid,
            s.agreement_uuid,
            s.xml_version,
            s.xml_content
        FROM {sections_table} s
        JOIN {xml_table} x
          ON x.agreement_uuid = s.agreement_uuid
         AND x.version = s.xml_version
        WHERE s.agreement_uuid IN :agreement_uuids
          AND x.latest = 1
          AND (x.status IS NULL OR x.status = 'verified')
        ORDER BY s.agreement_uuid, s.section_uuid
        """
    ).bindparams(bindparam("agreement_uuids", expanding=True))
    upsert_sql = _upsert_sql(conn, section_text_table)

    refreshed = 0
    for start in range(0, len(target_uuids), agreement_query_batch_size):
        agreement_batch = target_uuids[start : start + agreement_query_batch_size]
        result = conn.execute(
            source_sql,
            {"agreement_uuids": agreement_batch},
        ).mappings()
        while True:
            source_batch = result.fetchmany(write_batch_size)
            if not source_batch:
                break

            text_rows: list[dict[str, object]] = []
            for row in source_batch:
                xml_content = row["xml_content"]
                if not isinstance(xml_content, str):
                    raise TypeError(
                        f"sections.xml_content for {row['section_uuid']} must be a string."
                    )
                _, normalized_text = build_section_search_text(xml_content)
                text_rows.append(
                    {
                        "section_uuid": row["section_uuid"],
                        "agreement_uuid": row["agreement_uuid"],
                        "xml_version": row["xml_version"],
                        "source_xml_sha256": hashlib.sha256(
                            xml_content.encode("utf-8")
                        ).digest(),
                        "normalized_text": normalized_text,
                    }
                )

            _ = conn.execute(upsert_sql, text_rows)
            refreshed += len(text_rows)

        _delete_stale_rows(
            conn,
            section_text_table=section_text_table,
            sections_table=sections_table,
            xml_table=xml_table,
            agreement_uuids=agreement_batch,
        )

    return refreshed


def select_section_text_backfill_agreements(
    conn: Connection,
    schema: str,
    *,
    after_agreement_uuid: str,
    limit: int,
) -> list[str]:
    """Return a deterministic keyset page of current searchable agreements."""
    if limit <= 0:
        raise ValueError("limit must be positive.")
    latest_sections_table = _qualified_table(schema, "latest_sections_search")
    return [
        str(agreement_uuid)
        for agreement_uuid in conn.execute(
            text(
                f"""
                SELECT agreement_uuid
                FROM {latest_sections_table}
                WHERE agreement_uuid > :after_agreement_uuid
                GROUP BY agreement_uuid
                ORDER BY agreement_uuid
                LIMIT :limit
                """
            ),
            {"after_agreement_uuid": after_agreement_uuid, "limit": limit},
        ).scalars()
    ]


def _drifted_agreements_sql(
    dialect_name: str,
    *,
    latest_table: str,
    sections_table: str,
    target_table: str,
) -> str:
    version_diff = _version_diff_expression(dialect_name)
    source_hash_diff = _source_hash_diff_expression(dialect_name)
    return f"""
        SELECT drift.agreement_uuid
        FROM (
            SELECT missing_or_stale.agreement_uuid
            FROM (
                SELECT l.agreement_uuid
                FROM {latest_table} l
                JOIN {sections_table} s
                  ON s.section_uuid = l.section_uuid
                LEFT JOIN {target_table} t
                  ON t.section_uuid = l.section_uuid
                WHERE l.agreement_uuid > :after_agreement_uuid
                  AND (
                      t.section_uuid IS NULL
                      OR {version_diff}
                      OR {source_hash_diff}
                  )
                GROUP BY l.agreement_uuid
                ORDER BY l.agreement_uuid
                LIMIT :limit
            ) missing_or_stale

            UNION

            SELECT orphaned.agreement_uuid
            FROM (
                SELECT t.agreement_uuid
                FROM {target_table} t
                LEFT JOIN {latest_table} l
                  ON l.section_uuid = t.section_uuid
                WHERE t.agreement_uuid > :after_agreement_uuid
                  AND l.section_uuid IS NULL
                GROUP BY t.agreement_uuid
                ORDER BY t.agreement_uuid
                LIMIT :limit
            ) orphaned
        ) drift
        ORDER BY drift.agreement_uuid
        LIMIT :limit
    """


def select_section_text_drifted_agreements(
    conn: Connection,
    schema: str,
    *,
    table_name: str,
    after_agreement_uuid: str,
    limit: int,
) -> list[str]:
    """Return a keyset page whose target rows differ from the current section set."""
    if limit <= 0:
        raise ValueError("limit must be positive.")

    sql = _drifted_agreements_sql(
        conn.dialect.name,
        latest_table=_qualified_table(schema, "latest_sections_search"),
        sections_table=_qualified_table(schema, "sections"),
        target_table=_qualified_table(schema, table_name),
    )
    return [
        str(agreement_uuid)
        for agreement_uuid in conn.execute(
            text(sql),
            {"after_agreement_uuid": after_agreement_uuid, "limit": limit},
        ).scalars()
    ]


def select_section_text_drift_details(
    conn: Connection,
    schema: str,
    *,
    table_name: str,
    agreement_uuids: Sequence[str],
) -> list[dict[str, str]]:
    """Return every drifted section for the given agreements with its reason."""
    target_uuids = tuple(sorted({uuid for uuid in agreement_uuids if uuid}))
    if not target_uuids:
        return []

    latest_table = _qualified_table(schema, "latest_sections_search")
    sections_table = _qualified_table(schema, "sections")
    target_table = _qualified_table(schema, table_name)
    version_diff = _version_diff_expression(conn.dialect.name)
    source_hash_diff = _source_hash_diff_expression(conn.dialect.name)
    rows = conn.execute(
        text(
            f"""
            SELECT
                l.section_uuid,
                l.agreement_uuid,
                CASE
                    WHEN t.section_uuid IS NULL THEN 'missing text row'
                    WHEN {version_diff} THEN 'xml_version differs from sections'
                    ELSE 'source_xml_sha256 differs from sections.xml_content'
                END AS reason
            FROM {latest_table} l
            JOIN {sections_table} s
              ON s.section_uuid = l.section_uuid
            LEFT JOIN {target_table} t
              ON t.section_uuid = l.section_uuid
            WHERE l.agreement_uuid IN :agreement_uuids
              AND (
                  t.section_uuid IS NULL
                  OR {version_diff}
                  OR {source_hash_diff}
              )

            UNION ALL

            SELECT
                t.section_uuid,
                t.agreement_uuid,
                'text row has no latest_sections_search row' AS reason
            FROM {target_table} t
            LEFT JOIN {latest_table} l
              ON l.section_uuid = t.section_uuid
            WHERE t.agreement_uuid IN :agreement_uuids
              AND l.section_uuid IS NULL
            """
        ).bindparams(bindparam("agreement_uuids", expanding=True)),
        {"agreement_uuids": target_uuids},
    ).mappings()
    return sorted(
        (
            {
                "section_uuid": str(row["section_uuid"]),
                "agreement_uuid": str(row["agreement_uuid"]),
                "reason": str(row["reason"]),
            }
            for row in rows
        ),
        key=lambda row: (row["agreement_uuid"], row["section_uuid"]),
    )


def prune_section_text_search(
    conn: Connection,
    schema: str,
    agreement_uuids: Sequence[str],
    *,
    table_name: str = "section_text_search",
) -> int:
    """Delete text rows for the given agreements whose section left latest_sections_search."""
    target_uuids = tuple(sorted({uuid for uuid in agreement_uuids if uuid}))
    if not target_uuids:
        return 0
    if not inspect(conn).has_table(table_name, schema=schema or None):
        return 0

    section_text_table = _qualified_table(schema, table_name)
    latest_table = _qualified_table(schema, "latest_sections_search")
    result = conn.execute(
        text(
            f"""
            DELETE FROM {section_text_table}
            WHERE agreement_uuid IN :agreement_uuids
              AND NOT EXISTS (
                  SELECT 1
                  FROM {latest_table} l
                  WHERE l.section_uuid = {section_text_table}.section_uuid
              )
            """
        ).bindparams(bindparam("agreement_uuids", expanding=True)),
        {"agreement_uuids": target_uuids},
    )
    return int(result.rowcount or 0)


def _version_diff_expression(dialect_name: str) -> str:
    if dialect_name == "sqlite":
        return "t.xml_version IS NOT s.xml_version"
    return "NOT (t.xml_version <=> s.xml_version)"


def _source_hash_diff_expression(dialect_name: str) -> str:
    if dialect_name == "sqlite":
        return (
            "t.source_xml_sha256 IS NULL "
            "OR t.source_xml_sha256 <> sha256(s.xml_content)"
        )
    return (
        "t.source_xml_sha256 IS NULL "
        "OR t.source_xml_sha256 <> UNHEX(SHA2(s.xml_content, 256))"
    )
