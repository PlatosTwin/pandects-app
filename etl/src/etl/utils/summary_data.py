"""Refresh the summary_data table from agreement-level aggregates."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from dagster import AssetExecutionContext
from sqlalchemy import text

from etl.defs.resources import DBResource


def refresh_summary_data(
    context: AssetExecutionContext | None,
    db: DBResource,
) -> None:
    """Rebuild summary_data from current agreements, pages, and sections."""
    schema = db.database
    engine = db.get_engine()

    summary_table = f"{schema}.summary_data"
    status_summary_table = f"{schema}.agreement_status_summary"
    agreements_table = f"{schema}.agreements"
    xml_table = f"{schema}.xml"
    pages_table = f"{schema}.pages"
    sections_table = f"{schema}.sections"
    tagged_outputs_table = f"{schema}.tagged_outputs"

    with engine.begin() as conn:
        _ = conn.execute(text(f"TRUNCATE TABLE {summary_table}"))
        _ = conn.execute(text(f"TRUNCATE TABLE {status_summary_table}"))
        _ = conn.execute(
            text(
                f"""
                INSERT INTO {summary_table} (
                    year,
                    form_type,
                    exhibit_type,
                    target_type,
                    acquirer_type,
                    deal_type,
                    attitude,
                    purpose,
                    count_agreements,
                    count_pages,
                    count_sections,
                    count_distinct_acquirer,
                    count_distinct_target,
                    sum_transaction_value_total,
                    count_verified
                )
                WITH eligible_xml AS (
                    SELECT DISTINCT agreement_uuid
                    FROM {xml_table}
                    WHERE status IS NULL OR status = 'verified'
                )
                SELECT
                    year(date(filing_date)) as year,
                    a.form_type,
                    a.exhibit_type,
                    a.target_type,
                    a.acquirer_type,
                    a.deal_type,
                    a.attitude,
                    a.purpose,
                    COUNT(*) AS count_agreements,
                    COALESCE(SUM(p.page_count), 0) AS count_pages,
                    COALESCE(SUM(s.section_count), 0) AS count_sections,
                    COUNT(DISTINCT a.acquirer) AS count_distinct_acquirer,
                    COUNT(DISTINCT a.target) AS count_distinct_target,
                    COALESCE(SUM(a.transaction_price_total), 0) AS sum_transaction_value_total,
                    SUM(CASE WHEN a.verified THEN 1 ELSE 0 END) AS count_verified
                FROM eligible_xml AS x
                JOIN {agreements_table} AS a
                    ON a.agreement_uuid = x.agreement_uuid
                LEFT JOIN (
                    SELECT p.agreement_uuid, COUNT(*) AS page_count
                    FROM {pages_table} AS p
                    WHERE p.agreement_uuid IN (SELECT agreement_uuid FROM eligible_xml)
                    GROUP BY p.agreement_uuid
                ) AS p
                    ON p.agreement_uuid = a.agreement_uuid
                LEFT JOIN (
                    SELECT s.agreement_uuid, COUNT(*) AS section_count
                    FROM {sections_table} AS s
                    WHERE s.agreement_uuid IN (SELECT agreement_uuid FROM eligible_xml)
                    GROUP BY s.agreement_uuid
                ) AS s
                    ON s.agreement_uuid = a.agreement_uuid
                GROUP BY
                    1,
                    2,
                    3,
                    4,
                    5,
                    6,
                    7,
                    8
                """
            )
        )
        _ = conn.execute(
            text(
                f"""
                INSERT INTO {status_summary_table} (
                    year,
                    color,
                    current_stage,
                    count
                )
                WITH green AS (
                    SELECT
                        YEAR(DATE(filing_date)) AS year,
                        'green' AS color,
                        'processed' AS current_stage,
                        COUNT(DISTINCT x.agreement_uuid) AS count
                    FROM {xml_table} x
                    JOIN {agreements_table} a
                        ON x.agreement_uuid = a.agreement_uuid
                    WHERE (x.status IS NULL OR x.status = 'verified')
                        AND x.latest = 1
                    GROUP BY 1, 2, 3
                ),
                yellow_a AS (
                    SELECT
                        YEAR(DATE(filing_date)) AS year,
                        'yellow' AS color,
                        '0_staging' AS current_stage,
                        COUNT(DISTINCT a.agreement_uuid) AS count
                    FROM {agreements_table} a
                    LEFT JOIN {pages_table} p
                        ON a.agreement_uuid = p.agreement_uuid
                    WHERE p.agreement_uuid IS NULL
                        AND a.gated = 0
                    GROUP BY 1, 2, 3
                ),
                red_a AS (
                    SELECT
                        YEAR(DATE(filing_date)) AS year,
                        'red' AS color,
                        '0_staging' AS current_stage,
                        COUNT(DISTINCT a.agreement_uuid) AS count
                    FROM {agreements_table} a
                    LEFT JOIN {pages_table} p
                        ON a.agreement_uuid = p.agreement_uuid
                    WHERE p.agreement_uuid IS NULL
                        AND a.gated = 1
                    GROUP BY 1, 2, 3
                ),
                gated_pages AS (
                    SELECT DISTINCT
                        agreement_uuid
                    FROM {pages_table}
                    WHERE gated = 1
                ),
                tagged AS (
                    SELECT DISTINCT agreement_uuid
                    FROM {tagged_outputs_table} t
                    JOIN {pages_table} p ON t.page_uuid = p.page_uuid
                ),
                gated_tagged AS (
                    SELECT DISTINCT p.agreement_uuid
                    FROM {tagged_outputs_table} t
                    JOIN {pages_table} p
                        ON t.page_uuid = p.page_uuid
                    WHERE t.gated = 1
                ),
                yellow_b AS (
                    SELECT
                        YEAR(DATE(filing_date)) AS year,
                        'yellow' AS color,
                        '1_pre_processing' AS current_stage,
                        COUNT(DISTINCT a.agreement_uuid) AS count
                    FROM {agreements_table} a
                    JOIN {pages_table} p
                        ON a.agreement_uuid = p.agreement_uuid
                    LEFT JOIN gated_pages
                        ON p.agreement_uuid = gated_pages.agreement_uuid
                    LEFT JOIN tagged
                        ON a.agreement_uuid = tagged.agreement_uuid
                    WHERE gated_pages.agreement_uuid IS NULL
                        AND tagged.agreement_uuid is null
                        AND (paginated is null or paginated)
                    GROUP BY 1, 2, 3
                ),
                red_b AS (
                    SELECT
                        YEAR(DATE(filing_date)) AS year,
                        'red' AS color,
                        '1_pre_processing' AS current_stage,
                        COUNT(DISTINCT a.agreement_uuid) AS count
                    FROM {agreements_table} a
                    JOIN {pages_table} p
                        ON a.agreement_uuid = p.agreement_uuid
                    JOIN gated_pages
                        ON p.agreement_uuid = gated_pages.agreement_uuid
                    GROUP BY 1, 2, 3
                ),
                gray_b AS (
                    SELECT
                        YEAR(DATE(filing_date)) AS year,
                        'gray' AS color,
                        '1_pre_processing' AS current_stage,
                        COUNT(DISTINCT agreement_uuid) AS count
                    FROM {agreements_table}
                    WHERE paginated = False
                    GROUP BY 1, 2, 3
                ),
                yellow_c AS (
                    SELECT
                        YEAR(DATE(filing_date)) AS year,
                        'yellow' AS color,
                        '2_tagging' AS current_stage,
                        COUNT(DISTINCT p.agreement_uuid) AS count
                    FROM {tagged_outputs_table} t
                    JOIN {pages_table} p
                        ON t.page_uuid = p.page_uuid
                    JOIN {agreements_table} a
                        ON p.agreement_uuid = a.agreement_uuid
                    LEFT JOIN {xml_table} x
                        ON a.agreement_uuid = x.agreement_uuid
                    LEFT JOIN gated_tagged
                        ON a.agreement_uuid = gated_tagged.agreement_uuid
                    WHERE (
                        x.agreement_uuid IS NULL
                        OR EXISTS (
                            SELECT 1
                            FROM {pages_table} p_upd
                            JOIN {tagged_outputs_table} t_upd
                                ON t_upd.page_uuid = p_upd.page_uuid
                            WHERE p_upd.agreement_uuid = a.agreement_uuid
                                AND p_upd.source_page_type = 'body'
                                AND t_upd.updated_date > x.created_date
                        )
                    )
                    AND gated_tagged.agreement_uuid IS NULL
                    GROUP BY 1, 2, 3
                ),
                red_c AS (
                    SELECT
                        YEAR(DATE(filing_date)) AS year,
                        'red' AS color,
                        '2_tagging' AS current_stage,
                        COUNT(DISTINCT p.agreement_uuid) AS count
                    FROM {tagged_outputs_table} t
                    JOIN {pages_table} p
                        ON t.page_uuid = p.page_uuid
                    JOIN {agreements_table} a
                        ON p.agreement_uuid = a.agreement_uuid
                    JOIN gated_tagged
                        ON a.agreement_uuid = gated_tagged.agreement_uuid
                    GROUP BY 1, 2, 3
                ),
                red_d AS (
                    SELECT
                        YEAR(DATE(filing_date)) AS year,
                        'red' AS color,
                        '3_xml' AS current_stage,
                        COUNT(x.agreement_uuid) AS count
                    FROM {xml_table} x
                    JOIN {agreements_table} a
                        ON x.agreement_uuid = a.agreement_uuid
                    WHERE x.gated = 1
                    GROUP BY 1, 2, 3
                )
                SELECT * FROM green
                UNION ALL SELECT * FROM yellow_a
                UNION ALL SELECT * FROM red_a
                UNION ALL SELECT * FROM yellow_b
                UNION ALL SELECT * FROM red_b
                UNION ALL SELECT * FROM gray_b
                UNION ALL SELECT * FROM yellow_c
                UNION ALL SELECT * FROM red_c
                UNION ALL SELECT * FROM red_d
                """
            )
        )

    if context is not None:
        context.log.info("summary_data refreshed.")
