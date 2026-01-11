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
    agreements_table = f"{schema}.agreements"
    pages_table = f"{schema}.pages"
    sections_table = f"{schema}.sections"

    with engine.begin() as conn:
        _ = conn.execute(text(f"TRUNCATE TABLE {summary_table}"))
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
                    SELECT agreement_uuid, COUNT(*) AS page_count
                    FROM {pages_table}
                    GROUP BY agreement_uuid
                ) AS p
                    ON p.agreement_uuid = a.agreement_uuid
                LEFT JOIN (
                    SELECT agreement_uuid, COUNT(*) AS section_count
                    FROM {sections_table}
                    GROUP BY agreement_uuid
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

    if context is not None:
        context.log.info("summary_data refreshed.")
