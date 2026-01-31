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
                -- Deduplicate XML rows per agreement to avoid double-counting.
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
                FROM (
                    SELECT DISTINCT agreement_uuid
                    FROM {xml_table}
                    WHERE status IS NULL OR status = 'verified'
                ) AS x
                JOIN {agreements_table} AS a
                    ON a.agreement_uuid = x.agreement_uuid
                LEFT JOIN (
                    SELECT p.agreement_uuid, COUNT(*) AS page_count
                    FROM {pages_table} AS p
                    WHERE p.agreement_uuid IN (
                        SELECT DISTINCT agreement_uuid
                        FROM {xml_table}
                        WHERE status IS NULL OR status = 'verified'
                    )
                    GROUP BY p.agreement_uuid
                ) AS p
                    ON p.agreement_uuid = a.agreement_uuid
                LEFT JOIN (
                    SELECT s.agreement_uuid, COUNT(*) AS section_count
                    FROM {sections_table} AS s
                    WHERE s.agreement_uuid IN (
                        SELECT DISTINCT agreement_uuid
                        FROM {xml_table}
                        WHERE status IS NULL OR status = 'verified'
                    )
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
                WITH latest_xml AS (
                    SELECT
                        agreement_uuid,
                        MAX(created_date) AS created_date
                    FROM {xml_table}
                    GROUP BY agreement_uuid
                ),
                green AS (
                    SELECT
                        YEAR(DATE(filing_date)) AS year,
                        'green' AS color,
                        'processed' AS current_stage,
                        COUNT(DISTINCT x.agreement_uuid) AS count
                    FROM {xml_table} x
                    JOIN latest_xml
                        ON x.agreement_uuid = latest_xml.agreement_uuid
                        AND x.created_date = latest_xml.created_date
                    JOIN {agreements_table} a
                        ON x.agreement_uuid = a.agreement_uuid
                    WHERE x.status IS NULL
                        OR x.status = 'verified'
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
                        AND (
                            (prob_filing < 0.75 AND status = 'verified')
                            OR (
                                prob_filing > 0.75
                                AND (status = 'verified' OR status IS NULL)
                            )
                        )
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
                        AND prob_filing < 0.75
                        AND status IS NULL
                    GROUP BY 1, 2, 3
                ),
                low_conf AS (
                    SELECT DISTINCT
                        agreement_uuid
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
                        FROM {pages_table}
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
                ),
                out_of_order AS (
                    WITH PageRanks AS (
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
                        FROM {pages_table}
                        WHERE gold_label IS NULL
                    ),
                    RankedPages AS (
                        SELECT
                            agreement_uuid,
                            page_order,
                            type_rank,
                            MAX(type_rank) OVER (
                                PARTITION BY agreement_uuid
                                ORDER BY page_order
                                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
                            ) AS max_prev_type_rank
                        FROM PageRanks
                    )
                    SELECT DISTINCT
                        agreement_uuid
                    FROM RankedPages
                    WHERE max_prev_type_rank IS NOT NULL
                        AND type_rank < max_prev_type_rank
                ),
                tagged as (select distinct agreement_uuid from {tagged_outputs_table} join {pages_table} using(page_uuid)),
                yellow_b AS (
                    SELECT
                        YEAR(DATE(filing_date)) AS year,
                        'yellow' AS color,
                        '1_pre_processing' AS current_stage,
                        COUNT(DISTINCT a.agreement_uuid) AS count
                    FROM {agreements_table} a
                    JOIN {pages_table} p
                        ON a.agreement_uuid = p.agreement_uuid
                    LEFT JOIN low_conf
                        ON p.agreement_uuid = low_conf.agreement_uuid
                    LEFT JOIN out_of_order
                        ON p.agreement_uuid = out_of_order.agreement_uuid
                    LEFT JOIN tagged
                        ON a.agreement_uuid = tagged.agreement_uuid
                    WHERE low_conf.agreement_uuid IS NULL
                        AND out_of_order.agreement_uuid IS NULL
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
                    LEFT JOIN low_conf
                        ON p.agreement_uuid = low_conf.agreement_uuid
                    LEFT JOIN out_of_order
                        ON p.agreement_uuid = out_of_order.agreement_uuid
                    WHERE
                        (
                            low_conf.agreement_uuid IS NOT NULL
                            OR out_of_order.agreement_uuid IS NOT NULL
                        )
                    GROUP BY 1, 2, 3
                ),
                gray_b AS (
                    select
                        YEAR(DATE(filing_date)) AS year,
                        'gray' AS color,
                        '1_pre_processing' AS current_stage,
                        COUNT(DISTINCT agreement_uuid) AS count
                    from pdx.agreements
                        where paginated = False
                    group by 1,2,3
                ),
                label_errs AS (
                    SELECT DISTINCT
                        agreement_uuid
                    FROM {tagged_outputs_table} t
                    JOIN {pages_table} p
                        ON t.page_uuid = p.page_uuid
                    WHERE t.label_error
                ),
                repairs AS (
                    SELECT DISTINCT
                        agreement_uuid
                    FROM {schema}.ai_repair_requests r
                    JOIN {pages_table} p
                        ON r.page_uuid = p.page_uuid
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
                    LEFT JOIN repairs
                        ON a.agreement_uuid = repairs.agreement_uuid
                    LEFT JOIN label_errs
                        ON a.agreement_uuid = label_errs.agreement_uuid
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
                    AND CASE
                        WHEN repairs.agreement_uuid IS NOT NULL
                            THEN label_errs.agreement_uuid IS NULL
                        ELSE TRUE
                    END
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
                    JOIN label_errs
                        ON a.agreement_uuid = label_errs.agreement_uuid
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
                    WHERE x.status = 'invalid'
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
