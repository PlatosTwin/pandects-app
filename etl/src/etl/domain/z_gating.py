# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Connection

from etl.utils.pipeline_state_sql import canonical_stage_state_sql

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
    _ = min_article_tags
    stage_state_table = "tmp_stage_state_for_gating"
    _ = conn.execute(text(f"DROP TEMPORARY TABLE IF EXISTS {stage_state_table}"))
    _ = conn.execute(
        text(
            f"""
            CREATE TEMPORARY TABLE {stage_state_table} AS
            {canonical_stage_state_sql(schema)}
            """
        )
    )
    _ = conn.execute(
        text(f"ALTER TABLE {stage_state_table} ADD PRIMARY KEY (agreement_uuid)")
    )

    agreements_gated = apply_agreement_gating(
        conn, schema, stage_state_table=stage_state_table
    )
    pages_gated = apply_pages_gating(
        conn, schema, stage_state_table=stage_state_table
    )
    tagged_outputs_gated = apply_tagged_outputs_gating(conn, schema)
    xml_gated = apply_xml_gating(conn, schema, stage_state_table=stage_state_table)
    _set_validation_priority(conn, schema)
    return GatingCounts(
        agreements_gated=agreements_gated,
        pages_gated=pages_gated,
        tagged_outputs_gated=tagged_outputs_gated,
        xml_gated=xml_gated,
    )


def apply_agreement_gating(
    conn: Connection,
    schema: str,
    *,
    stage_state_table: str,
) -> int:
    agreements_table = f"{schema}.agreements"
    stmt = text(
        f"""
        UPDATE {agreements_table} a
        LEFT JOIN {stage_state_table} s
            ON s.agreement_uuid = a.agreement_uuid
        SET a.gated = CASE
            WHEN s.current_stage = '0_staging' AND s.color = 'red' THEN 1
            ELSE 0
        END
        WHERE (
            a.gated IS NULL
            OR a.gated != CASE
                WHEN s.current_stage = '0_staging' AND s.color = 'red' THEN 1
                ELSE 0
            END
        )
        """
    )
    result = conn.execute(stmt)
    return int(result.rowcount or 0)


def apply_pages_gating(
    conn: Connection,
    schema: str,
    *,
    stage_state_table: str,
) -> int:
    pages_table = f"{schema}.pages"
    stmt = text(
        f"""
        UPDATE {pages_table} p
        LEFT JOIN {stage_state_table} s
            ON s.agreement_uuid = p.agreement_uuid
        SET p.gated = CASE
            WHEN s.current_stage = '1_pre_processing' AND s.color = 'red' THEN 1
            ELSE 0
        END
        WHERE (
            p.gated IS NULL
            OR p.gated != CASE
                WHEN s.current_stage = '1_pre_processing' AND s.color = 'red' THEN 1
                ELSE 0
            END
        )
        """
    )
    result = conn.execute(stmt)
    return int(result.rowcount or 0)


def apply_tagged_outputs_gating(conn: Connection, schema: str) -> int:
    tagged_outputs_table = f"{schema}.tagged_outputs"
    pages_table = f"{schema}.pages"
    _ = conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_flagged_tagged_agreements"))
    _ = conn.execute(
        text(
            """
            CREATE TEMPORARY TABLE tmp_flagged_tagged_agreements (
                agreement_uuid VARCHAR(64) NOT NULL PRIMARY KEY
            ) ENGINE=MEMORY
            """
        )
    )
    _ = conn.execute(
        text(
            f"""
            INSERT INTO tmp_flagged_tagged_agreements (agreement_uuid)
            SELECT DISTINCT p.agreement_uuid
            FROM {tagged_outputs_table} t
            JOIN {pages_table} p
                ON t.page_uuid = p.page_uuid
            WHERE t.label_error = 1
              AND p.agreement_uuid IS NOT NULL
            """
        )
    )

    stmt = text(
        f"""
        UPDATE {tagged_outputs_table} t
        JOIN {pages_table} p
            ON t.page_uuid = p.page_uuid
        LEFT JOIN tmp_flagged_tagged_agreements g
            ON g.agreement_uuid = p.agreement_uuid
        SET t.gated = CASE
            WHEN g.agreement_uuid IS NULL THEN 0
            ELSE 1
        END
        WHERE (
            t.gated IS NULL
            OR t.gated != CASE
                WHEN g.agreement_uuid IS NULL THEN 0
                ELSE 1
            END
        )
        """
    )
    result = conn.execute(stmt)
    return int(result.rowcount or 0)


def apply_xml_gating(
    conn: Connection,
    schema: str,
    *,
    stage_state_table: str,
) -> int:
    xml_table = f"{schema}.xml"
    stmt = text(
        f"""
        UPDATE {xml_table} x
        JOIN {stage_state_table} s
            ON s.agreement_uuid = x.agreement_uuid
        SET x.gated = CASE
            WHEN s.current_stage = '3_xml' AND s.color = 'red' THEN 1
            ELSE 0
        END
        WHERE (
            x.latest = 1
            AND (
                x.gated IS NULL
                OR x.gated != CASE
                    WHEN s.current_stage = '3_xml' AND s.color = 'red' THEN 1
                    ELSE 0
                END
            )
        )
        """
    )
    result = conn.execute(stmt)
    return int(result.rowcount or 0)


def _set_validation_priority(conn: Connection, schema: str) -> None:
    _ = conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_page_flag_counts"))
    _ = conn.execute(
        text(
            """
            CREATE TEMPORARY TABLE tmp_page_flag_counts (
                agreement_uuid VARCHAR(64) NOT NULL PRIMARY KEY,
                ct_flagged BIGINT NOT NULL
            ) ENGINE=MEMORY
            """
        )
    )
    _ = conn.execute(
        text(
            f"""
            INSERT INTO tmp_page_flag_counts (agreement_uuid, ct_flagged)
            SELECT
                agreement_uuid,
                SUM(CASE WHEN gated = 1 THEN 1 ELSE 0 END) AS ct_flagged
            FROM {schema}.pages
            WHERE agreement_uuid IS NOT NULL
            GROUP BY agreement_uuid
            """
        )
    )

    _ = conn.execute(
        text(
            f"""
            UPDATE {schema}.agreements
            SET validation_priority = 1 - prob_filing
            WHERE NOT (validation_priority <=> (1 - prob_filing))
            """
        )
    )
    _ = conn.execute(
        text(
            f"""
            UPDATE {schema}.pages p
            JOIN tmp_page_flag_counts g
                ON g.agreement_uuid = p.agreement_uuid
            SET p.validation_priority = g.ct_flagged
            WHERE NOT (p.validation_priority <=> g.ct_flagged)
            """
        )
    )
    _ = conn.execute(
        text(
            f"""
            UPDATE {schema}.tagged_outputs t
            JOIN {schema}.pages p
                ON p.page_uuid = t.page_uuid
            LEFT JOIN tmp_page_flag_counts g
                ON g.agreement_uuid = p.agreement_uuid
            SET t.validation_priority = COALESCE(g.ct_flagged, 0)
            WHERE NOT (t.validation_priority <=> COALESCE(g.ct_flagged, 0))
            """
        )
    )
    _ = conn.execute(
        text(
            f"""
            UPDATE {schema}.xml
            SET validation_priority = 1
            WHERE NOT (validation_priority <=> 1)
            """
        )
    )
