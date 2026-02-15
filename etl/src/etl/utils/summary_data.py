"""Refresh the summary_data table from agreement-level aggregates."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from dagster import AssetExecutionContext, DagsterRunStatus, RunsFilter
from sqlalchemy import text
from sqlalchemy.engine import Connection

from etl.defs.resources import DBResource

IN_FLIGHT_RUN_STATUSES = (
    DagsterRunStatus.QUEUED,
    DagsterRunStatus.NOT_STARTED,
    DagsterRunStatus.MANAGED,
    DagsterRunStatus.STARTING,
    DagsterRunStatus.STARTED,
    DagsterRunStatus.CANCELING,
)


def _get_other_in_flight_run_ids(context: AssetExecutionContext) -> list[str]:
    """Return in-flight Dagster run ids excluding the current run."""
    runs_filter = RunsFilter(statuses=list(IN_FLIGHT_RUN_STATUSES))
    in_flight_runs = context.instance.get_runs(filters=runs_filter)
    return [run.run_id for run in in_flight_runs if run.run_id != context.run_id]


def _ensure_deal_type_summary_table(
    conn: Connection,
    *,
    schema: str,
    table: str,
) -> None:
    """Ensure deal type summary table exists."""
    _ = conn.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.{table} (
                year INT NOT NULL,
                deal_type VARCHAR(64) NOT NULL,
                `count` BIGINT NOT NULL,
                PRIMARY KEY (year, deal_type)
            )
            """
        )
    )


def refresh_summary_data(
    context: AssetExecutionContext | None,
    db: DBResource,
) -> None:
    """Rebuild summary_data from current agreements, pages, and sections."""
    if context is not None:
        other_in_flight_run_ids = _get_other_in_flight_run_ids(context)
        if other_in_flight_run_ids:
            context.log.info(
                "Skipping summary_data refresh for run %s; %d other Dagster run(s) are still in flight.",
                context.run_id,
                len(other_in_flight_run_ids),
            )
            return

    schema = db.database
    engine = db.get_engine()

    summary_table = f"{schema}.summary_data"
    status_summary_table = f"{schema}.agreement_status_summary"
    deal_type_summary_table = f"{schema}.agreement_deal_type_summary"
    agreements_table = f"{schema}.agreements"
    xml_table = f"{schema}.xml"
    pages_table = f"{schema}.pages"
    sections_table = f"{schema}.sections"
    tagged_outputs_table = f"{schema}.tagged_outputs"

    lock_name = f"{schema}.summary_data_refresh"
    lock_timeout_seconds = 300

    with engine.begin() as conn:
        lock_result = conn.execute(
            text("SELECT GET_LOCK(:lock_name, :timeout_seconds) AS got_lock"),
            {"lock_name": lock_name, "timeout_seconds": lock_timeout_seconds},
        ).scalar()

        if lock_result != 1:
            if context is not None:
                context.log.warning(
                    f"Skipping summary_data refresh; lock busy ({lock_name})."
                )
            return

        try:
            _ensure_deal_type_summary_table(
                conn,
                schema=schema,
                table="agreement_deal_type_summary",
            )
            _ = conn.execute(text(f"TRUNCATE TABLE {summary_table}"))
            _ = conn.execute(text(f"TRUNCATE TABLE {status_summary_table}"))
            _ = conn.execute(text(f"TRUNCATE TABLE {deal_type_summary_table}"))
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
                FROM {agreements_table} AS a
                LEFT JOIN (
                    SELECT p.agreement_uuid, COUNT(*) AS page_count
                    FROM {pages_table} AS p
                    GROUP BY p.agreement_uuid
                ) AS p
                    ON p.agreement_uuid = a.agreement_uuid
                LEFT JOIN (
                    SELECT s.agreement_uuid, COUNT(*) AS section_count
                    FROM {sections_table} AS s
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
                    FROM {sections_table} x
                    JOIN {agreements_table} a
                        ON x.agreement_uuid = a.agreement_uuid
                    GROUP BY 1, 2, 3
                ),
                -- agreement staged, awaiting pre-processing
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
                        AND (a.paginated IS NULL OR a.paginated = TRUE)
                    GROUP BY 1, 2, 3
                ),
                -- agreement staged, but needs validation
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
                        AND (a.paginated IS NULL OR a.paginated = TRUE)
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
                -- agreement in pages and ready to get tagged
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
                    GROUP BY 1, 2, 3
                ),
                -- agreement in pages but page classes need validation before tagging
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
                    LEFT JOIN tagged on a.agreement_uuid = tagged.agreement_uuid
                    WHERE
                        tagged.agreement_uuid is null
                    GROUP BY 1, 2, 3
                ),
                -- non-paginated agreements
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
                -- agreement tagged and ready to XML
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
                -- agreement tagged but needs validation before getting XML'd
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
                -- agreement XML'd and ready to get section'd
                yellow_d as (
                    SELECT
                         YEAR(DATE(filing_date)) AS year,
                        'yellow' AS color,
                        '3_xml' AS current_stage,
                        COUNT(x.agreement_uuid) AS count
                    FROM {xml_table} x
                    JOIN {agreements_table} a
                        ON x.agreement_uuid = a.agreement_uuid
                    LEFT JOIN {sections_table} s on s.agreement_uuid = a.agreement_uuid
                    WHERE x.gated = 0
                        AND latest
                        AND s.agreement_uuid IS NULL
                    GROUP BY 1, 2, 3
                ),
                -- agreement XML'd but needs validation before getting section'd
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
                UNION ALL SELECT * from yellow_d
                UNION ALL SELECT * FROM red_d
                """
                )
            )
            _ = conn.execute(
                text(
                    f"""
                INSERT INTO {deal_type_summary_table} (
                    year,
                    deal_type,
                    `count`
                )
                WITH eligible_xml AS (
                    SELECT DISTINCT agreement_uuid
                    FROM {xml_table}
                    WHERE status IS NULL OR status = 'verified'
                )
                SELECT
                    YEAR(DATE(a.filing_date)) AS year,
                    COALESCE(a.deal_type, 'unknown') AS deal_type,
                    COUNT(*) AS `count`
                FROM eligible_xml AS x
                JOIN {agreements_table} AS a
                    ON a.agreement_uuid = x.agreement_uuid
                WHERE YEAR(DATE(a.filing_date)) IS NOT NULL
                GROUP BY
                    YEAR(DATE(a.filing_date)),
                    COALESCE(a.deal_type, 'unknown')
                """
                )
            )
        finally:
            _ = conn.execute(
                text("SELECT RELEASE_LOCK(:lock_name)"), {"lock_name": lock_name}
            )

    if context is not None:
        context.log.info("summary_data refreshed.")
