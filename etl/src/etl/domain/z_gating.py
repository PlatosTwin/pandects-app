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
    tagged_outputs_gated = apply_tagged_outputs_gating(conn, schema, min_article_tags)
    xml_gated = apply_xml_gating(conn, schema)
    return GatingCounts(
        agreements_gated=agreements_gated,
        pages_gated=pages_gated,
        tagged_outputs_gated=tagged_outputs_gated,
        xml_gated=xml_gated,
    )


def apply_agreement_gating(conn: Connection, schema: str) -> int:
    stmt = text(
        f"""
        UPDATE {schema}.agreements
        SET gated = CASE
            WHEN NOT (
                (
                    (prob_filing < 0.90 AND status = 'verified')
                    OR (
                        prob_filing > 0.90
                        AND (status = 'verified' OR status IS NULL)
                    )
                )
                AND exhibit_type IS NOT NULL
                AND (paginated IS NULL OR paginated)
            )
            THEN 1
            ELSE 0
        END
        """
    )
    result = conn.execute(stmt)
    return int(result.rowcount or 0)


def apply_pages_gating(conn: Connection, schema: str) -> int:
    stmt = text(
        f"""
        UPDATE {schema}.pages p
        LEFT JOIN (
            SELECT DISTINCT agreement_uuid
            FROM (
                SELECT agreement_uuid
                FROM (
                    SELECT
                        agreement_uuid,
                        page_order,
                        page_type_prob_front_matter,
                        page_type_prob_toc,
                        page_type_prob_body,
                        page_type_prob_sig,
                        page_type_prob_back_matter,
                        gold_label,
                        MIN(
                            CASE
                                WHEN source_page_type = 'sig'
                                AND page_type_prob_sig >= 0.95
                                THEN page_order
                            END
                        ) OVER (PARTITION BY agreement_uuid) AS sig_cutoff_page
                    FROM {schema}.pages
                ) sub
                WHERE
                    (
                        page_type_prob_front_matter BETWEEN 0.3 AND 0.7
                        OR page_type_prob_toc BETWEEN 0.3 AND 0.7
                        OR page_type_prob_body BETWEEN 0.3 AND 0.7
                        OR page_type_prob_sig BETWEEN 0.3 AND 0.7
                        OR page_type_prob_back_matter BETWEEN 0.3 AND 0.7
                    )
                    AND (
                        page_order <= sig_cutoff_page
                        OR sig_cutoff_page IS NULL
                    )
                    AND gold_label IS NULL
                UNION
                SELECT agreement_uuid
                FROM (
                    SELECT agreement_uuid
                    FROM (
                        SELECT
                            agreement_uuid,
                            page_order,
                            type_rank,
                            MAX(type_rank) OVER (
                                PARTITION BY agreement_uuid
                                ORDER BY page_order
                                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                            ) AS max_prev_type_rank
                        FROM (
                            SELECT
                                agreement_uuid,
                                page_order,
                                CASE
                                    WHEN source_page_type = 'front_matter' THEN 1
                                    WHEN source_page_type = 'toc' THEN 2
                                    WHEN source_page_type = 'body' THEN 3
                                    WHEN source_page_type = 'sig' THEN 4
                                    WHEN source_page_type = 'back_matter' THEN 5
                                    ELSE 99
                                END AS type_rank
                            FROM {schema}.pages
                            WHERE gold_label IS NULL
                        ) PageRanks
                    ) RankedPages
                    WHERE max_prev_type_rank IS NOT NULL
                        AND type_rank < max_prev_type_rank
                ) out_of_order
            ) gated
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


def apply_tagged_outputs_gating(
    conn: Connection,
    schema: str,
    min_article_tags: int = MIN_ARTICLE_TAGS,
) -> int:
    label_error_uuids = _fetch_label_error_agreements(conn, schema)
    low_article_uuids = _fetch_low_article_agreements(conn, schema, min_article_tags)
    gated_agreements = sorted(label_error_uuids | low_article_uuids)

    _ = conn.execute(text(f"UPDATE {schema}.tagged_outputs SET gated = 0"))
    if not gated_agreements:
        return 0

    stmt = text(
        f"""
        UPDATE {schema}.tagged_outputs t
        JOIN {schema}.pages p
            ON t.page_uuid = p.page_uuid
        SET t.gated = 1
        WHERE p.agreement_uuid IN :gated_agreements
        """
    ).bindparams(bindparam("gated_agreements", expanding=True))
    result = conn.execute(stmt, {"gated_agreements": gated_agreements})
    return int(result.rowcount or 0)


def apply_xml_gating(conn: Connection, schema: str) -> int:
    stmt = text(
        f"""
        UPDATE {schema}.xml
        SET gated = CASE
            WHEN status = 'invalid' THEN 1
            ELSE 0
        END
        """
    )
    result = conn.execute(stmt)
    return int(result.rowcount or 0)


def _fetch_label_error_agreements(conn: Connection, schema: str) -> set[str]:
    rows = conn.execute(
        text(
            f"""
            SELECT DISTINCT p.agreement_uuid
            FROM {schema}.tagged_outputs t
            JOIN {schema}.pages p
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
    rows = conn.execute(
        text(
            f"""
            WITH eligible AS (
                SELECT p.agreement_uuid
                FROM {schema}.pages p
                LEFT JOIN {schema}.tagged_outputs t
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
            FROM {schema}.pages p
            LEFT JOIN {schema}.tagged_outputs t
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
