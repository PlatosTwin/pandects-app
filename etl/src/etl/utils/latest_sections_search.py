# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
from __future__ import annotations

import json
from collections.abc import Sequence

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection


def _qualified_table(schema: str, table_name: str) -> str:
    if not schema:
        return table_name
    return f"{schema}.{table_name}"


def _parse_serialized_standard_ids(raw: object) -> list[str]:
    if raw is None:
        return []
    if not isinstance(raw, str):
        raise TypeError(f"Unsupported section_standard_ids value: {type(raw)!r}")

    serialized = raw.strip()
    if not serialized:
        return []

    try:
        parsed = json.loads(serialized)
    except json.JSONDecodeError:
        return [serialized]

    if isinstance(parsed, str):
        return [parsed]
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        raise ValueError("section_standard_ids must be a string or JSON list of strings.")

    deduped_ids: list[str] = []
    seen: set[str] = set()
    for item in parsed:
        if item in seen:
            continue
        seen.add(item)
        deduped_ids.append(item)
    return deduped_ids


def refresh_latest_sections_search(
    conn: Connection,
    schema: str,
    agreement_uuids: Sequence[str],
) -> int:
    target_uuids = tuple(sorted({agreement_uuid for agreement_uuid in agreement_uuids if agreement_uuid}))
    if not target_uuids:
        return 0

    latest_sections_search_table = _qualified_table(schema, "latest_sections_search")
    latest_sections_search_standard_ids_table = _qualified_table(
        schema,
        "latest_sections_search_standard_ids",
    )
    agreements_table = _qualified_table(schema, "agreements")
    sections_table = _qualified_table(schema, "sections")
    xml_table = _qualified_table(schema, "xml")

    delete_sql = text(
        f"""
        DELETE FROM {latest_sections_search_table}
        WHERE agreement_uuid IN :agreement_uuids
        """
    ).bindparams(bindparam("agreement_uuids", expanding=True))
    _ = conn.execute(delete_sql, {"agreement_uuids": target_uuids})

    delete_standard_ids_sql = text(
        f"""
        DELETE FROM {latest_sections_search_standard_ids_table}
        WHERE agreement_uuid IN :agreement_uuids
        """
    ).bindparams(bindparam("agreement_uuids", expanding=True))
    _ = conn.execute(delete_standard_ids_sql, {"agreement_uuids": target_uuids})

    insert_sql = text(
        f"""
        INSERT INTO {latest_sections_search_table} (
            section_uuid,
            agreement_uuid,
            filing_date,
            prob_filing,
            filing_company_name,
            filing_company_cik,
            form_type,
            exhibit_type,
            target,
            acquirer,
            transaction_price_total,
            transaction_price_stock,
            transaction_price_cash,
            transaction_price_assets,
            transaction_consideration,
            target_type,
            acquirer_type,
            target_counsel,
            acquirer_counsel,
            target_industry,
            acquirer_industry,
            announce_date,
            close_date,
            deal_status,
            attitude,
            deal_type,
            purpose,
            target_pe,
            acquirer_pe,
            verified,
            url,
            section_standard_ids,
            article_title,
            section_title
        )
        SELECT
            s.section_uuid,
            s.agreement_uuid,
            a.filing_date,
            a.prob_filing,
            a.filing_company_name,
            a.filing_company_cik,
            a.form_type,
            a.exhibit_type,
            a.target,
            a.acquirer,
            a.transaction_price_total,
            a.transaction_price_stock,
            a.transaction_price_cash,
            a.transaction_price_assets,
            a.transaction_consideration,
            a.target_type,
            a.acquirer_type,
            a.target_counsel,
            a.acquirer_counsel,
            a.target_industry,
            a.acquirer_industry,
            a.announce_date,
            a.close_date,
            a.deal_status,
            a.attitude,
            a.deal_type,
            a.purpose,
            a.target_pe,
            a.acquirer_pe,
            a.verified,
            a.url,
            COALESCE(s.section_standard_id_gold_label, s.section_standard_id),
            s.article_title,
            s.section_title
        FROM {sections_table} s
        JOIN {agreements_table} a
            ON a.agreement_uuid = s.agreement_uuid
        JOIN {xml_table} x
            ON x.agreement_uuid = s.agreement_uuid
           AND x.version = s.xml_version
        WHERE s.agreement_uuid IN :agreement_uuids
          AND x.latest = 1
          AND (x.status IS NULL OR x.status = 'verified')
        """
    ).bindparams(bindparam("agreement_uuids", expanding=True))
    result = conn.execute(insert_sql, {"agreement_uuids": target_uuids})

    standard_id_source_rows = conn.execute(
        text(
            f"""
            SELECT section_uuid, agreement_uuid, section_standard_ids
            FROM {latest_sections_search_table}
            WHERE agreement_uuid IN :agreement_uuids
            """
        ).bindparams(bindparam("agreement_uuids", expanding=True)),
        {"agreement_uuids": target_uuids},
    ).mappings()

    standard_id_rows: list[dict[str, str]] = []
    for row in standard_id_source_rows:
        section_uuid = str(row["section_uuid"])
        agreement_uuid = str(row["agreement_uuid"])
        for standard_id in _parse_serialized_standard_ids(row["section_standard_ids"]):
            standard_id_rows.append(
                {
                    "standard_id": standard_id,
                    "section_uuid": section_uuid,
                    "agreement_uuid": agreement_uuid,
                }
            )

    if standard_id_rows:
        _ = conn.execute(
            text(
                f"""
                INSERT INTO {latest_sections_search_standard_ids_table} (
                    standard_id,
                    section_uuid,
                    agreement_uuid
                ) VALUES (
                    :standard_id,
                    :section_uuid,
                    :agreement_uuid
                )
                """
            ),
            standard_id_rows,
        )
    return int(result.rowcount or 0)
