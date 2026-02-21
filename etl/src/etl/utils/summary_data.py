"""Refresh the summary_data table from agreement-level aggregates."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from dagster import AssetExecutionContext
from sqlalchemy import text
from sqlalchemy.engine import Connection

from etl.defs.resources import DBResource
from etl.utils.pipeline_state_sql import canonical_stage_state_sql

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


def _build_summary_temp_tables(conn: Connection, *, schema: str) -> None:
    pages_table = f"{schema}.pages"
    sections_table = f"{schema}.sections"
    xml_table = f"{schema}.xml"

    _ = conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_page_counts"))
    _ = conn.execute(
        text(
            f"""
            CREATE TEMPORARY TABLE tmp_page_counts AS
            SELECT
                agreement_uuid,
                COUNT(*) AS page_count
            FROM {pages_table}
            WHERE agreement_uuid IS NOT NULL
            GROUP BY agreement_uuid
            """
        )
    )
    _ = conn.execute(text("ALTER TABLE tmp_page_counts ADD PRIMARY KEY (agreement_uuid)"))

    _ = conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_pages_agg"))
    _ = conn.execute(
        text(
            """
            CREATE TEMPORARY TABLE tmp_pages_agg AS
            SELECT
                pc.agreement_uuid,
                pc.page_count
            FROM tmp_page_counts pc
            """
        )
    )
    _ = conn.execute(text("ALTER TABLE tmp_pages_agg ADD PRIMARY KEY (agreement_uuid)"))

    _ = conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_sections_agg"))
    _ = conn.execute(
        text(
            f"""
            CREATE TEMPORARY TABLE tmp_sections_agg AS
            SELECT
                s.agreement_uuid,
                COUNT(*) AS section_count
            FROM {sections_table} s
            JOIN {xml_table} x
                ON x.agreement_uuid = s.agreement_uuid
               AND x.version = s.xml_version
               AND x.latest = 1
            WHERE s.agreement_uuid IS NOT NULL
            GROUP BY s.agreement_uuid
            """
        )
    )
    _ = conn.execute(text("ALTER TABLE tmp_sections_agg ADD PRIMARY KEY (agreement_uuid)"))

    _ = conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_xml_latest"))
    _ = conn.execute(
        text(
            f"""
            CREATE TEMPORARY TABLE tmp_xml_latest AS
            SELECT
                agreement_uuid,
                version,
                created_date,
                status
            FROM {xml_table}
            WHERE latest = 1
              AND agreement_uuid IS NOT NULL
            """
        )
    )
    _ = conn.execute(text("ALTER TABLE tmp_xml_latest ADD PRIMARY KEY (agreement_uuid)"))

    _ = conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_xml_eligible"))
    _ = conn.execute(
        text(
            """
            CREATE TEMPORARY TABLE tmp_xml_eligible AS
            SELECT agreement_uuid
            FROM tmp_xml_latest
            WHERE status = 'verified'
            """
        )
    )
    _ = conn.execute(text("ALTER TABLE tmp_xml_eligible ADD PRIMARY KEY (agreement_uuid)"))

    _ = conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_stage_state"))
    _ = conn.execute(
        text(
            f"""
            CREATE TEMPORARY TABLE tmp_stage_state AS
            {canonical_stage_state_sql(schema, include_year=True)}
            """
        )
    )
    _ = conn.execute(text("ALTER TABLE tmp_stage_state ADD PRIMARY KEY (agreement_uuid)"))


def _refresh_summary_table(conn: Connection, *, schema: str) -> None:
    summary_table = f"{schema}.summary_data"
    agreements_table = f"{schema}.agreements"
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
            LEFT JOIN tmp_pages_agg p
                ON p.agreement_uuid = a.agreement_uuid
            LEFT JOIN tmp_sections_agg s
                ON s.agreement_uuid = a.agreement_uuid
            WHERE COALESCE(LOWER(a.status), '') <> 'invalid'
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


def _refresh_status_summary_table(conn: Connection, *, schema: str) -> None:
    status_summary_table = f"{schema}.agreement_status_summary"
    agreements_table = f"{schema}.agreements"
    _ = conn.execute(
        text(
            f"""
            INSERT INTO {status_summary_table} (
                year,
                color,
                current_stage,
                count
            )
            SELECT
                s.year,
                s.color,
                s.current_stage,
                COUNT(*) AS count
            FROM tmp_stage_state s
            JOIN {agreements_table} a
                ON a.agreement_uuid = s.agreement_uuid
            WHERE s.year IS NOT NULL
              AND COALESCE(LOWER(a.status), '') <> 'invalid'
            GROUP BY s.year, s.color, s.current_stage
            """
        )
    )


def _refresh_deal_type_summary_table(conn: Connection, *, schema: str) -> None:
    deal_type_summary_table = f"{schema}.agreement_deal_type_summary"
    agreements_table = f"{schema}.agreements"
    _ = conn.execute(
        text(
            f"""
            INSERT INTO {deal_type_summary_table} (
                year,
                deal_type,
                `count`
            )
            SELECT
                YEAR(DATE(a.filing_date)) AS year,
                COALESCE(a.deal_type, 'unknown') AS deal_type,
                COUNT(*) AS `count`
            FROM tmp_xml_eligible x
            JOIN {agreements_table} a
                ON a.agreement_uuid = x.agreement_uuid
            WHERE YEAR(DATE(a.filing_date)) IS NOT NULL
              AND COALESCE(LOWER(a.status), '') <> 'invalid'
            GROUP BY
                YEAR(DATE(a.filing_date)),
                COALESCE(a.deal_type, 'unknown')
            """
        )
    )


def refresh_summary_data(
    context: AssetExecutionContext | None,
    db: DBResource,
) -> None:
    """Rebuild summary_data from current agreements, pages, and sections."""
    schema = db.database
    engine = db.get_engine()

    summary_table = f"{schema}.summary_data"
    status_summary_table = f"{schema}.agreement_status_summary"
    deal_type_summary_table = f"{schema}.agreement_deal_type_summary"

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
            _build_summary_temp_tables(conn, schema=schema)
            _refresh_summary_table(conn, schema=schema)
            _refresh_status_summary_table(conn, schema=schema)
            _refresh_deal_type_summary_table(conn, schema=schema)
        finally:
            _ = conn.execute(
                text("SELECT RELEASE_LOCK(:lock_name)"), {"lock_name": lock_name}
            )

    if context is not None:
        context.log.info("summary_data refreshed.")
