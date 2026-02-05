# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection

from etl.domain.f_xml import count_article_tags


MIN_ARTICLE_TAGS = 5


@dataclass(frozen=True)
class GatingCounts:
    agreements_gated: int
    pages_gated: int
    tagged_outputs_gated: int
    xml_gated: int


def apply_gating(
    conn: Connection,
    schema: str,
    min_article_tags: int = MIN_ARTICLE_TAGS,
) -> GatingCounts:
    agreements_gated = apply_agreement_gating(conn, schema)
    pages_gated = apply_pages_gating(conn, schema)
    tagged_outputs_gated = apply_tagged_outputs_gating(conn, schema)
    xml_gated = apply_xml_gating(conn, schema, min_article_tags)
    _set_validation_priority(conn, schema)
    return GatingCounts(
        agreements_gated=agreements_gated,
        pages_gated=pages_gated,
        tagged_outputs_gated=tagged_outputs_gated,
        xml_gated=xml_gated,
    )


def apply_agreement_gating(conn: Connection, schema: str) -> int:
    agreements_table = f"{schema}.agreements"
    stmt = text(
        f"""
        UPDATE {agreements_table}
        SET gated = CASE
            WHEN prob_filing < 0.75 AND status IS NULL THEN 1
            ELSE 0
        END
        """
    )
    result = conn.execute(stmt)
    return int(result.rowcount or 0)


def apply_pages_gating(conn: Connection, schema: str) -> int:
    pages_table = f"{schema}.pages"
    stmt = text(
        f"""
        UPDATE {pages_table} p
        LEFT JOIN (
            WITH sigs AS (
                SELECT
                    agreement_uuid,
                    COUNT(page_uuid) AS ct_pages,
                    MAX(CASE WHEN source_page_type = 'sig' THEN page_type_prob_sig ELSE 0 END) AS prob_sig,
                    SUM(CASE WHEN source_page_type = 'sig' THEN 1 ELSE 0 END) AS ct_sig,
                    SUM(CASE WHEN source_page_type = 'back_matter' THEN 1 ELSE 0 END) AS ct_back_matter,
                    SUM(CASE WHEN gold_label IS NOT NULL THEN 1 ELSE 0 END) AS ct_gold
                FROM {pages_table}
                GROUP BY agreement_uuid
            )
            SELECT DISTINCT agreement_uuid
            FROM sigs
            WHERE ct_back_matter >= 1 AND ct_sig = 0 AND ct_gold = 0
        ) g
            ON g.agreement_uuid = p.agreement_uuid
        SET p.gated = CASE
            WHEN g.agreement_uuid IS NULL THEN 0
            ELSE 1
        END
        """
    )
    result = conn.execute(stmt)
    return int(result.rowcount or 0)


def apply_tagged_outputs_gating(conn: Connection, schema: str) -> int:
    tagged_outputs_table = f"{schema}.tagged_outputs"
    pages_table = f"{schema}.pages"
    gated_agreements = sorted(_fetch_label_error_agreements(conn, schema))

    _ = conn.execute(text(f"UPDATE {tagged_outputs_table} SET gated = 0"))
    if not gated_agreements:
        return 0

    stmt = text(
        f"""
        UPDATE {tagged_outputs_table} t
        JOIN {pages_table} p
            ON t.page_uuid = p.page_uuid
        SET t.gated = 1
        WHERE p.agreement_uuid IN :gated_agreements
        """
    ).bindparams(bindparam("gated_agreements", expanding=True))
    result = conn.execute(stmt, {"gated_agreements": gated_agreements})
    return int(result.rowcount or 0)


def apply_xml_gating(
    conn: Connection,
    schema: str,
    min_article_tags: int = MIN_ARTICLE_TAGS,
) -> int:
    xml_table = f"{schema}.xml"
    low_article_uuids = _fetch_low_article_agreements(conn, schema, min_article_tags)
    if not low_article_uuids:
        stmt = text(
            f"""
            UPDATE {xml_table} x
            SET x.gated = CASE WHEN x.status = 'invalid' THEN 1 ELSE 0 END
            """
        )
        result = conn.execute(stmt)
    else:
        stmt = text(
            f"""
            UPDATE {xml_table} x
            SET x.gated = CASE
                WHEN x.status = 'invalid' THEN 1
                WHEN x.agreement_uuid IN :low_article_uuids
                    AND (x.status IS NULL OR x.status != 'verified')
                THEN 1
                ELSE 0
            END
            """
        ).bindparams(bindparam("low_article_uuids", expanding=True))
        result = conn.execute(stmt, {"low_article_uuids": sorted(low_article_uuids)})
    return int(result.rowcount or 0)


def _fetch_label_error_agreements(conn: Connection, schema: str) -> set[str]:
    tagged_outputs_table = f"{schema}.tagged_outputs"
    pages_table = f"{schema}.pages"
    rows = conn.execute(
        text(
            f"""
            SELECT DISTINCT p.agreement_uuid
            FROM {tagged_outputs_table} t
            JOIN {pages_table} p
                ON t.page_uuid = p.page_uuid
            WHERE t.label_error = 1
            """
        )
    ).scalars()
    return {row for row in rows if row is not None}


def _fetch_low_article_agreements(
    conn: Connection,
    schema: str,
    min_article_tags: int,
) -> set[str]:
    pages_table = f"{schema}.pages"
    tagged_outputs_table = f"{schema}.tagged_outputs"
    rows = conn.execute(
        text(
            f"""
            WITH eligible AS (
                SELECT p.agreement_uuid
                FROM {pages_table} p
                LEFT JOIN {tagged_outputs_table} t
                    ON t.page_uuid = p.page_uuid
                WHERE p.source_page_type = 'body'
                GROUP BY p.agreement_uuid
                HAVING SUM(
                    CASE
                        WHEN COALESCE(
                            t.tagged_text_gold,
                            t.tagged_text_corrected,
                            t.tagged_text
                        ) IS NOT NULL
                        THEN 1 ELSE 0
                    END
                ) = SUM(CASE WHEN p.source_page_type = 'body' THEN 1 ELSE 0 END)
            )
            SELECT
                p.agreement_uuid,
                COALESCE(
                    t.tagged_text_gold,
                    t.tagged_text_corrected,
                    t.tagged_text,
                    p.processed_page_content
                ) AS tagged_output
            FROM {pages_table} p
            LEFT JOIN {tagged_outputs_table} t
                ON t.page_uuid = p.page_uuid
            JOIN eligible e
                ON e.agreement_uuid = p.agreement_uuid
            WHERE p.source_page_type = 'body'
            ORDER BY p.agreement_uuid, p.page_order
            """
        )
    ).mappings()

    counts: dict[str, int] = {}
    for row in rows:
        agreement_uuid = row["agreement_uuid"]
        tagged_output = row["tagged_output"] or ""
        counts[agreement_uuid] = counts.get(agreement_uuid, 0) + count_article_tags(
            tagged_output
        )

    return {uuid for uuid, count in counts.items() if count < min_article_tags}


def _set_validation_priority(conn: Connection, schema: str) -> None:
    _ = conn.execute(
        text(f"UPDATE {schema}.agreements SET validation_priority = 1 - prob_filing")
    )
    _ = conn.execute(
        text(
            f"""
            UPDATE {schema}.pages p
            JOIN (
                SELECT agreement_uuid,
                    SUM(CASE WHEN gated = 1 THEN 1 ELSE 0 END) AS ct_flagged
                FROM {schema}.pages
                GROUP BY agreement_uuid
            ) g
                ON g.agreement_uuid = p.agreement_uuid
            SET p.validation_priority = g.ct_flagged
            """
        )
    )
    _ = conn.execute(text(f"UPDATE {schema}.tagged_outputs SET validation_priority = 1"))
    _ = conn.execute(text(f"UPDATE {schema}.xml SET validation_priority = 1"))
