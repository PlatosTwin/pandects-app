"""Refresh the summary_data table from agreement-level aggregates."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false

from collections import defaultdict
from decimal import Decimal
from dagster import AssetExecutionContext
from sqlalchemy import text
from sqlalchemy.engine import Connection

from backend.summary_specs import METADATA_FIELD_COVERAGE_CONFIG
from etl.defs.resources import DBResource
from etl.utils.pipeline_state_sql import canonical_stage_state_sql


_TREND_TARGET_BUCKET_ORDER = ("public", "private")
_TREND_BUYER_BUCKET_ORDER = (
    "public_buyer",
    "private_strategic",
    "private_equity",
    "other",
)


def _quantile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    sorted_values = sorted(values)
    index = (len(sorted_values) - 1) * percentile
    lower_index = int(index)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    if lower_index == upper_index:
        return sorted_values[lower_index]
    weight = index - lower_index
    return sorted_values[lower_index] + (
        (sorted_values[upper_index] - sorted_values[lower_index]) * weight
    )


def _coerce_int(value: object | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, Decimal):
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return None
    if isinstance(value, float):
        if not value.is_integer():
            return None
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped and stripped.lstrip("-").isdigit():
            try:
                return int(stripped)
            except (TypeError, ValueError, OverflowError):
                return None
    return None


def _coerce_float(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            return float(value)
        except (TypeError, ValueError, OverflowError):
            return None
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except (TypeError, ValueError, OverflowError):
            return None
    return None


def _summary_eligible_agreement_where_sql(*, alias: str = "a") -> str:
    return (
        f"COALESCE(LOWER({alias}.status), '') <> 'invalid'\n"
        f"              AND NOT (COALESCE({alias}.gated, 0) = 1 AND COALESCE({alias}.verified, 0) = 0)"
    )


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


def _ensure_agreement_overview_summary_table(
    conn: Connection,
    *,
    schema: str,
    table: str,
) -> None:
    _ = conn.execute(
        text(
            f"""
            CREATE TABLE IF NOT EXISTS {schema}.{table} (
                singleton_key TINYINT NOT NULL,
                metadata_covered_agreements BIGINT NULL,
                metadata_coverage_pct DECIMAL(5, 1) NULL,
                taxonomy_covered_sections BIGINT NULL,
                taxonomy_coverage_pct DECIMAL(5, 1) NULL,
                latest_filing_date DATE NULL,
                PRIMARY KEY (singleton_key)
            )
            """
        )
    )
    _ = conn.execute(
        text(
            f"""
            ALTER TABLE {schema}.{table}
            ADD COLUMN IF NOT EXISTS metadata_covered_agreements BIGINT NULL
            """
        )
    )
    _ = conn.execute(
        text(
            f"""
            ALTER TABLE {schema}.{table}
            ADD COLUMN IF NOT EXISTS taxonomy_covered_sections BIGINT NULL
            """
        )
    )


def _ensure_agreement_trends_summary_tables(conn: Connection, *, schema: str) -> None:
    table_sql = (
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.agreement_ownership_mix_summary (
            year INT NOT NULL,
            target_bucket VARCHAR(32) NOT NULL,
            deal_count BIGINT NOT NULL,
            total_transaction_value DECIMAL(24, 2) NOT NULL,
            PRIMARY KEY (year, target_bucket)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.agreement_ownership_deal_size_summary (
            year INT NOT NULL,
            target_bucket VARCHAR(32) NOT NULL,
            deal_count BIGINT NOT NULL,
            p25_transaction_value DECIMAL(24, 2) NULL,
            median_transaction_value DECIMAL(24, 2) NULL,
            p75_transaction_value DECIMAL(24, 2) NULL,
            PRIMARY KEY (year, target_bucket)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.agreement_buyer_type_matrix_summary (
            target_bucket VARCHAR(32) NOT NULL,
            buyer_bucket VARCHAR(32) NOT NULL,
            deal_count BIGINT NOT NULL,
            median_transaction_value DECIMAL(24, 2) NULL,
            PRIMARY KEY (target_bucket, buyer_bucket)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.agreement_target_industry_summary (
            year INT NOT NULL,
            industry VARCHAR(255) NOT NULL,
            deal_count BIGINT NOT NULL,
            total_transaction_value DECIMAL(24, 2) NOT NULL,
            PRIMARY KEY (year, industry)
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {schema}.agreement_industry_pairing_summary (
            target_industry VARCHAR(255) NOT NULL,
            acquirer_industry VARCHAR(255) NOT NULL,
            deal_count BIGINT NOT NULL,
            total_transaction_value DECIMAL(24, 2) NOT NULL,
            PRIMARY KEY (target_industry, acquirer_industry)
        )
        """,
    )
    for sql in table_sql:
        _ = conn.execute(text(sql))


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

    _ = conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_agreement_trends_base"))
    _ = conn.execute(
        text(
            f"""
            CREATE TEMPORARY TABLE tmp_agreement_trends_base AS
            SELECT
                a.agreement_uuid,
                CASE
                    WHEN a.filing_date IS NULL THEN NULL
                    WHEN TRIM(a.filing_date) REGEXP '^[0-9]{{4}}' THEN
                        CAST(SUBSTRING(TRIM(a.filing_date), 1, 4) AS UNSIGNED)
                    ELSE NULL
                END AS filing_year,
                CAST(COALESCE(a.transaction_price_total, 0) AS DECIMAL(24, 2)) AS transaction_price_total,
                CASE
                    WHEN LOWER(TRIM(COALESCE(a.target_type, ''))) = 'public' THEN 'public'
                    WHEN LOWER(TRIM(COALESCE(a.target_type, ''))) = 'private'
                         OR COALESCE(a.target_pe, 0) = 1 THEN 'private'
                    ELSE NULL
                END AS target_bucket,
                CASE
                    WHEN COALESCE(a.acquirer_pe, 0) = 1 THEN 'private_equity'
                    WHEN LOWER(TRIM(COALESCE(a.acquirer_type, ''))) = 'public' THEN 'public_buyer'
                    WHEN LOWER(TRIM(COALESCE(a.acquirer_type, ''))) = 'private' THEN 'private_strategic'
                    ELSE 'other'
                END AS buyer_bucket,
                COALESCE(NULLIF(TRIM(a.target_industry), ''), 'Unspecified') AS target_industry,
                COALESCE(NULLIF(TRIM(a.acquirer_industry), ''), 'Unspecified') AS acquirer_industry
            FROM {schema}.agreements a
            JOIN tmp_xml_latest x
                ON x.agreement_uuid = a.agreement_uuid
            WHERE (x.status IS NULL OR x.status = 'verified')
              AND {_summary_eligible_agreement_where_sql(alias='a')}
            """
        )
    )
    _ = conn.execute(text("ALTER TABLE tmp_agreement_trends_base ADD PRIMARY KEY (agreement_uuid)"))

    _ = conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_filter_option_base"))
    _ = conn.execute(
        text(
            f"""
            CREATE TEMPORARY TABLE tmp_filter_option_base AS
            SELECT
                a.agreement_uuid,
                a.target,
                a.acquirer,
                a.target_industry,
                a.acquirer_industry
            FROM {schema}.agreements a
            JOIN tmp_xml_latest x
                ON x.agreement_uuid = a.agreement_uuid
            JOIN tmp_sections_agg s
                ON s.agreement_uuid = a.agreement_uuid
            WHERE (x.status IS NULL OR x.status = 'verified')
              AND {_summary_eligible_agreement_where_sql(alias='a')}
            """
        )
    )
    _ = conn.execute(text("ALTER TABLE tmp_filter_option_base ADD PRIMARY KEY (agreement_uuid)"))

    _ = conn.execute(text("DROP TEMPORARY TABLE IF EXISTS tmp_agreement_index_base"))
    _ = conn.execute(
        text(
            f"""
            CREATE TEMPORARY TABLE tmp_agreement_index_base AS
            SELECT
                a.agreement_uuid,
                CASE
                    WHEN a.filing_date IS NULL THEN NULL
                    WHEN TRIM(a.filing_date) REGEXP '^[0-9]{{4}}' THEN
                        CAST(SUBSTRING(TRIM(a.filing_date), 1, 4) AS UNSIGNED)
                    ELSE NULL
                END AS year,
                a.target,
                a.acquirer,
                COALESCE(a.verified, 0) AS verified
            FROM {schema}.agreements a
            JOIN tmp_xml_eligible x
                ON x.agreement_uuid = a.agreement_uuid
            WHERE {_summary_eligible_agreement_where_sql(alias='a')}
            """
        )
    )
    _ = conn.execute(text("ALTER TABLE tmp_agreement_index_base ADD PRIMARY KEY (agreement_uuid)"))


def _refresh_filter_option_summary_table(conn: Connection, *, schema: str) -> None:
    filter_table = f"{schema}.agreement_filter_option_summary"
    insert_rows = (
        (
            "targets",
            """
            SELECT 'targets' AS field_name, target AS option_value, COUNT(*) AS agreement_count
            FROM tmp_filter_option_base
            WHERE target IS NOT NULL AND TRIM(target) <> ''
            GROUP BY target
            """,
        ),
        (
            "acquirers",
            """
            SELECT 'acquirers' AS field_name, acquirer AS option_value, COUNT(*) AS agreement_count
            FROM tmp_filter_option_base
            WHERE acquirer IS NOT NULL AND TRIM(acquirer) <> ''
            GROUP BY acquirer
            """,
        ),
        (
            "target_industries",
            """
            SELECT 'target_industries' AS field_name, target_industry AS option_value, COUNT(*) AS agreement_count
            FROM tmp_filter_option_base
            WHERE target_industry IS NOT NULL AND TRIM(target_industry) <> ''
            GROUP BY target_industry
            """,
        ),
        (
            "acquirer_industries",
            """
            SELECT 'acquirer_industries' AS field_name, acquirer_industry AS option_value, COUNT(*) AS agreement_count
            FROM tmp_filter_option_base
            WHERE acquirer_industry IS NOT NULL AND TRIM(acquirer_industry) <> ''
            GROUP BY acquirer_industry
            """,
        ),
        (
            "target_counsels",
            f"""
            SELECT 'target_counsels' AS field_name, c.canonical_name AS option_value, COUNT(*) AS agreement_count
            FROM {schema}.agreement_counsel ac
            JOIN {schema}.counsel c
                ON c.counsel_id = ac.counsel_id
            JOIN tmp_filter_option_base f
                ON f.agreement_uuid = ac.agreement_uuid
            WHERE ac.side = 'target'
              AND c.canonical_name IS NOT NULL
              AND TRIM(c.canonical_name) <> ''
            GROUP BY c.canonical_name
            """,
        ),
        (
            "acquirer_counsels",
            f"""
            SELECT 'acquirer_counsels' AS field_name, c.canonical_name AS option_value, COUNT(*) AS agreement_count
            FROM {schema}.agreement_counsel ac
            JOIN {schema}.counsel c
                ON c.counsel_id = ac.counsel_id
            JOIN tmp_filter_option_base f
                ON f.agreement_uuid = ac.agreement_uuid
            WHERE ac.side = 'acquirer'
              AND c.canonical_name IS NOT NULL
              AND TRIM(c.canonical_name) <> ''
            GROUP BY c.canonical_name
            """,
        ),
    )
    for _, select_sql in insert_rows:
        _ = conn.execute(
            text(
                f"""
                INSERT INTO {filter_table} (
                    field_name,
                    option_value,
                    agreement_count
                )
                {select_sql}
                """
            )
        )


def _refresh_metadata_field_coverage_summary_table(conn: Connection, *, schema: str) -> None:
    coverage_table = f"{schema}.agreement_metadata_field_coverage_summary"
    agreements_table = f"{schema}.agreements"
    agreement_where_parts = ["1 = 1"]
    agreement_where_parts.append("COALESCE(LOWER(a.status), '') <> 'invalid'")
    agreement_where_parts.append(
        "NOT (COALESCE(a.gated, 0) = 1 AND COALESCE(a.verified, 0) = 0)"
    )
    ingested_where = "\n                      AND ".join(agreement_where_parts)
    processed_where_parts = [*agreement_where_parts]
    processed_where_parts.append(
        "EXISTS ("
        f"SELECT 1 FROM {schema}.xml x "
        "WHERE x.agreement_uuid = a.agreement_uuid "
        "AND x.latest = 1 "
        "AND (x.status IS NULL OR x.status = 'verified')"
        ")"
    )
    processed_where = "\n                      AND ".join(processed_where_parts)

    aggregate_select_lines: list[str] = []
    for config in METADATA_FIELD_COVERAGE_CONFIG:
        field = str(config["field"])
        eligible_sql = str(config["eligible_sql"])
        covered_sql = str(config["covered_sql"])
        aggregate_select_lines.extend(
            [
                (
                    "SUM(CASE WHEN a._coverage_scope_ingested = 1 "
                    f"AND {eligible_sql} THEN 1 ELSE 0 END) "
                    f"AS {field}_ingested_eligible_agreements"
                ),
                (
                    "SUM(CASE WHEN a._coverage_scope_ingested = 1 "
                    f"AND {eligible_sql} AND {covered_sql} THEN 1 ELSE 0 END) "
                    f"AS {field}_ingested_covered_agreements"
                ),
                (
                    "SUM(CASE WHEN a._coverage_scope_processed = 1 "
                    f"AND {eligible_sql} THEN 1 ELSE 0 END) "
                    f"AS {field}_processed_eligible_agreements"
                ),
                (
                    "SUM(CASE WHEN a._coverage_scope_processed = 1 "
                    f"AND {eligible_sql} AND {covered_sql} THEN 1 ELSE 0 END) "
                    f"AS {field}_processed_covered_agreements"
                ),
            ]
        )
    aggregate_select = ",\n                            ".join(aggregate_select_lines)
    aggregate_rows = (
        conn.execute(
            text(
                f"""
                SELECT
                    {aggregate_select}
                FROM (
                    SELECT
                        a.*,
                        1 AS _coverage_scope_ingested,
                        0 AS _coverage_scope_processed
                    FROM {agreements_table} a
                    WHERE {ingested_where}
                    UNION ALL
                    SELECT
                        a.*,
                        0 AS _coverage_scope_ingested,
                        1 AS _coverage_scope_processed
                    FROM {agreements_table} a
                    WHERE {processed_where}
                ) a
                """
            )
        )
        .mappings()
        .all()
    )
    row_dict = dict(aggregate_rows[0]) if aggregate_rows else {}

    insert_rows: list[dict[str, object | None]] = []
    for config in METADATA_FIELD_COVERAGE_CONFIG:
        field = str(config["field"])
        ingested_eligible_agreements = _coerce_int(
            row_dict.get(f"{field}_ingested_eligible_agreements")
        ) or 0
        ingested_covered_agreements = _coerce_int(
            row_dict.get(f"{field}_ingested_covered_agreements")
        ) or 0
        processed_eligible_agreements = _coerce_int(
            row_dict.get(f"{field}_processed_eligible_agreements")
        ) or 0
        processed_covered_agreements = _coerce_int(
            row_dict.get(f"{field}_processed_covered_agreements")
        ) or 0
        insert_rows.append(
            {
                "field_name": field,
                "label": config["label"],
                "ingested_eligible_agreements": ingested_eligible_agreements,
                "ingested_covered_agreements": ingested_covered_agreements,
                "ingested_coverage_pct": (
                    round((ingested_covered_agreements / ingested_eligible_agreements) * 100, 1)
                    if ingested_eligible_agreements > 0
                    else None
                ),
                "processed_eligible_agreements": processed_eligible_agreements,
                "processed_covered_agreements": processed_covered_agreements,
                "processed_coverage_pct": (
                    round((processed_covered_agreements / processed_eligible_agreements) * 100, 1)
                    if processed_eligible_agreements > 0
                    else None
                ),
                "note": config["note"],
            }
        )
    if insert_rows:
        _ = conn.execute(
            text(
                f"""
                INSERT INTO {coverage_table} (
                    field_name,
                    label,
                    ingested_eligible_agreements,
                    ingested_covered_agreements,
                    ingested_coverage_pct,
                    processed_eligible_agreements,
                    processed_covered_agreements,
                    processed_coverage_pct,
                    note
                ) VALUES (
                    :field_name,
                    :label,
                    :ingested_eligible_agreements,
                    :ingested_covered_agreements,
                    :ingested_coverage_pct,
                    :processed_eligible_agreements,
                    :processed_covered_agreements,
                    :processed_coverage_pct,
                    :note
                )
                """
            ),
            insert_rows,
        )


def _refresh_counsel_leaderboard_summary_table(conn: Connection, *, schema: str) -> None:
    leaderboard_table = f"{schema}.agreement_counsel_leaderboard_summary"
    _ = conn.execute(
        text(
            f"""
            INSERT INTO {leaderboard_table} (
                side,
                counsel_key,
                counsel,
                year,
                deal_count,
                total_transaction_value
            )
            SELECT
                CASE ac.side
                    WHEN 'acquirer' THEN 'buy_side'
                    WHEN 'target' THEN 'sell_side'
                    ELSE ac.side
                END AS side,
                c.canonical_name_normalized AS counsel_key,
                c.canonical_name AS counsel,
                CASE
                    WHEN a.filing_date IS NULL THEN NULL
                    WHEN TRIM(a.filing_date) REGEXP '^[0-9]{{4}}' THEN
                        CAST(SUBSTRING(TRIM(a.filing_date), 1, 4) AS UNSIGNED)
                    ELSE NULL
                END AS year,
                COUNT(*) AS deal_count,
                COALESCE(SUM(CAST(COALESCE(a.transaction_price_total, 0) AS DECIMAL(24, 2))), 0) AS total_transaction_value
            FROM {schema}.agreement_counsel ac
            JOIN {schema}.counsel c
                ON c.counsel_id = ac.counsel_id
            JOIN {schema}.agreements a
                ON a.agreement_uuid = ac.agreement_uuid
            JOIN tmp_xml_latest x
                ON x.agreement_uuid = a.agreement_uuid
            WHERE (x.status IS NULL OR x.status = 'verified')
              AND {_summary_eligible_agreement_where_sql(alias='a')}
              AND c.canonical_name IS NOT NULL
              AND TRIM(c.canonical_name) <> ''
              AND c.canonical_name_normalized IS NOT NULL
              AND TRIM(c.canonical_name_normalized) <> ''
            GROUP BY side, counsel_key, counsel, year
            HAVING year IS NOT NULL
            """
        )
    )


def _refresh_agreement_index_summary_table(conn: Connection, *, schema: str) -> None:
    index_table = f"{schema}.agreement_index_summary"
    _ = conn.execute(
        text(
            f"""
            INSERT INTO {index_table} (
                agreement_uuid,
                year,
                target,
                acquirer,
                verified
            )
            SELECT
                agreement_uuid,
                year,
                target,
                acquirer,
                verified
            FROM tmp_agreement_index_base
            """
        )
    )


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
            WHERE {_summary_eligible_agreement_where_sql(alias='a')}
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
              AND {_summary_eligible_agreement_where_sql(alias='a')}
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
              AND {_summary_eligible_agreement_where_sql(alias='a')}
            GROUP BY
                YEAR(DATE(a.filing_date)),
                COALESCE(a.deal_type, 'unknown')
            """
        )
    )


def _refresh_agreement_overview_summary_table(conn: Connection, *, schema: str) -> None:
    overview_summary_table = f"{schema}.agreement_overview_summary"
    agreements_table = f"{schema}.agreements"
    sections_table = f"{schema}.sections"

    _ = conn.execute(
        text(
            f"""
            INSERT INTO {overview_summary_table} (
                singleton_key,
                metadata_covered_agreements,
                metadata_coverage_pct,
                taxonomy_covered_sections,
                taxonomy_coverage_pct,
                latest_filing_date
            )
            SELECT
                1 AS singleton_key,
                SUM(CASE WHEN COALESCE(a.metadata, 0) = 1 THEN 1 ELSE 0 END) AS metadata_covered_agreements,
                ROUND(
                    100.0 * SUM(CASE WHEN COALESCE(a.metadata, 0) = 1 THEN 1 ELSE 0 END)
                    / NULLIF(COUNT(*), 0),
                    1
                ) AS metadata_coverage_pct,
                (
                    SELECT SUM(
                        CASE
                            WHEN (
                                (
                                    s.section_standard_id IS NOT NULL
                                    AND TRIM(s.section_standard_id) <> ''
                                    AND TRIM(s.section_standard_id) <> '[]'
                                )
                                OR (
                                    s.section_standard_id_gold_label IS NOT NULL
                                    AND TRIM(s.section_standard_id_gold_label) <> ''
                                    AND TRIM(s.section_standard_id_gold_label) <> '[]'
                                )
                            ) THEN 1
                            ELSE 0
                        END
                    )
                    FROM {sections_table} s
                    JOIN tmp_xml_latest x
                        ON x.agreement_uuid = s.agreement_uuid
                       AND x.version = s.xml_version
                    JOIN {agreements_table} a2
                        ON a2.agreement_uuid = s.agreement_uuid
                    WHERE {_summary_eligible_agreement_where_sql(alias='a2')}
                ) AS taxonomy_covered_sections,
                (
                    SELECT ROUND(
                        100.0 * SUM(
                            CASE
                                WHEN (
                                    (
                                        s.section_standard_id IS NOT NULL
                                        AND TRIM(s.section_standard_id) <> ''
                                        AND TRIM(s.section_standard_id) <> '[]'
                                    )
                                    OR (
                                        s.section_standard_id_gold_label IS NOT NULL
                                        AND TRIM(s.section_standard_id_gold_label) <> ''
                                        AND TRIM(s.section_standard_id_gold_label) <> '[]'
                                    )
                                ) THEN 1
                                ELSE 0
                            END
                        ) / NULLIF(COUNT(*), 0),
                        1
                    )
                    FROM {sections_table} s
                    JOIN tmp_xml_latest x
                        ON x.agreement_uuid = s.agreement_uuid
                       AND x.version = s.xml_version
                    JOIN {agreements_table} a2
                        ON a2.agreement_uuid = s.agreement_uuid
                    WHERE {_summary_eligible_agreement_where_sql(alias='a2')}
                ) AS taxonomy_coverage_pct,
                (
                    SELECT MAX(
                        CASE
                            WHEN a3.filing_date IS NULL THEN NULL
                            WHEN TRIM(a3.filing_date) REGEXP '^[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}' THEN
                                SUBSTRING(TRIM(a3.filing_date), 1, 10)
                            ELSE NULL
                        END
                    )
                    FROM {agreements_table} a3
                    WHERE a3.filing_date IS NOT NULL
                      AND {_summary_eligible_agreement_where_sql(alias='a3')}
                ) AS latest_filing_date
            FROM tmp_xml_eligible x
            JOIN {agreements_table} a
                ON a.agreement_uuid = x.agreement_uuid
            WHERE {_summary_eligible_agreement_where_sql(alias='a')}
            """
        )
    )


def _refresh_agreement_trends_summary_tables(conn: Connection, *, schema: str) -> None:
    ownership_mix_table = f"{schema}.agreement_ownership_mix_summary"
    ownership_deal_size_table = f"{schema}.agreement_ownership_deal_size_summary"
    buyer_type_matrix_table = f"{schema}.agreement_buyer_type_matrix_summary"
    target_industry_table = f"{schema}.agreement_target_industry_summary"
    industry_pairing_table = f"{schema}.agreement_industry_pairing_summary"
    filter_option_table = f"{schema}.agreement_filter_option_summary"
    metadata_field_coverage_table = f"{schema}.agreement_metadata_field_coverage_summary"
    counsel_leaderboard_table = f"{schema}.agreement_counsel_leaderboard_summary"
    agreement_index_table = f"{schema}.agreement_index_summary"

    _ = conn.execute(
        text(
            f"""
            INSERT INTO {ownership_mix_table} (
                year,
                target_bucket,
                deal_count,
                total_transaction_value
            )
            SELECT
                filing_year,
                target_bucket,
                COUNT(*) AS deal_count,
                COALESCE(SUM(transaction_price_total), 0) AS total_transaction_value
            FROM tmp_agreement_trends_base
            WHERE filing_year IS NOT NULL
              AND target_bucket IS NOT NULL
            GROUP BY filing_year, target_bucket
            """
        )
    )

    _ = conn.execute(
        text(
            f"""
            INSERT INTO {target_industry_table} (
                year,
                industry,
                deal_count,
                total_transaction_value
            )
            SELECT
                filing_year,
                target_industry,
                COUNT(*) AS deal_count,
                COALESCE(SUM(transaction_price_total), 0) AS total_transaction_value
            FROM tmp_agreement_trends_base
            WHERE filing_year IS NOT NULL
            GROUP BY filing_year, target_industry
            """
        )
    )

    _ = conn.execute(
        text(
            f"""
            INSERT INTO {industry_pairing_table} (
                target_industry,
                acquirer_industry,
                deal_count,
                total_transaction_value
            )
            SELECT
                target_industry,
                acquirer_industry,
                COUNT(*) AS deal_count,
                COALESCE(SUM(transaction_price_total), 0) AS total_transaction_value
            FROM tmp_agreement_trends_base
            GROUP BY target_industry, acquirer_industry
            """
        )
    )

    eligible_value_rows = conn.execute(
        text(
            """
            SELECT
                filing_year,
                target_bucket,
                buyer_bucket,
                CAST(transaction_price_total AS DOUBLE) AS transaction_price_total
            FROM tmp_agreement_trends_base
            WHERE filing_year IS NOT NULL
              AND target_bucket IS NOT NULL
              AND transaction_price_total > 0
            ORDER BY filing_year ASC, target_bucket ASC, buyer_bucket ASC, transaction_price_total ASC
            """
        )
    ).mappings().all()

    ownership_values: dict[tuple[int, str], list[float]] = defaultdict(list)
    buyer_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in eligible_value_rows:
        year = _coerce_int(row.get("filing_year"))
        target_bucket = row.get("target_bucket")
        buyer_bucket = row.get("buyer_bucket")
        transaction_value = _coerce_float(row.get("transaction_price_total"))
        if year is None:
            continue
        if not isinstance(target_bucket, str):
            continue
        if not isinstance(buyer_bucket, str):
            continue
        if transaction_value is None:
            continue
        ownership_values[(year, target_bucket)].append(transaction_value)
        buyer_values[(target_bucket, buyer_bucket)].append(transaction_value)

    ownership_count_rows = conn.execute(
        text(
            f"""
            SELECT year, target_bucket, deal_count
            FROM {ownership_mix_table}
            ORDER BY year ASC, target_bucket ASC
            """
        )
    ).mappings().all()
    ownership_deal_size_rows = []
    for row in ownership_count_rows:
        year = _coerce_int(row.get("year"))
        target_bucket = row.get("target_bucket")
        deal_count = row.get("deal_count")
        if year is None or not isinstance(target_bucket, str):
            continue
        count_value = int(deal_count or 0)
        values = ownership_values.get((year, target_bucket), [])
        ownership_deal_size_rows.append(
            {
                "year": year,
                "target_bucket": target_bucket,
                "deal_count": count_value,
                "p25_transaction_value": _quantile(values, 0.25),
                "median_transaction_value": _quantile(values, 0.5),
                "p75_transaction_value": _quantile(values, 0.75),
            }
        )
    if ownership_deal_size_rows:
        _ = conn.execute(
            text(
                f"""
                INSERT INTO {ownership_deal_size_table} (
                    year,
                    target_bucket,
                    deal_count,
                    p25_transaction_value,
                    median_transaction_value,
                    p75_transaction_value
                ) VALUES (
                    :year,
                    :target_bucket,
                    :deal_count,
                    :p25_transaction_value,
                    :median_transaction_value,
                    :p75_transaction_value
                )
                """
            ),
            ownership_deal_size_rows,
        )

    buyer_count_rows = conn.execute(
        text(
            """
            SELECT
                target_bucket,
                buyer_bucket,
                COUNT(*) AS deal_count
            FROM tmp_agreement_trends_base
            WHERE target_bucket IS NOT NULL
            GROUP BY target_bucket, buyer_bucket
            ORDER BY target_bucket ASC, buyer_bucket ASC
            """
        )
    ).mappings().all()
    buyer_matrix_rows = []
    for row in buyer_count_rows:
        target_bucket = row.get("target_bucket")
        buyer_bucket = row.get("buyer_bucket")
        deal_count = row.get("deal_count")
        if not isinstance(target_bucket, str) or not isinstance(buyer_bucket, str):
            continue
        values = buyer_values.get((target_bucket, buyer_bucket), [])
        buyer_matrix_rows.append(
            {
                "target_bucket": target_bucket,
                "buyer_bucket": buyer_bucket,
                "deal_count": int(deal_count or 0),
                "median_transaction_value": _quantile(values, 0.5),
            }
        )
    if buyer_matrix_rows:
        _ = conn.execute(
            text(
                f"""
                INSERT INTO {buyer_type_matrix_table} (
                    target_bucket,
                    buyer_bucket,
                    deal_count,
                    median_transaction_value
                ) VALUES (
                    :target_bucket,
                    :buyer_bucket,
                    :deal_count,
                    :median_transaction_value
                )
                """
            ),
            buyer_matrix_rows,
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
    overview_summary_table = f"{schema}.agreement_overview_summary"
    ownership_mix_table = f"{schema}.agreement_ownership_mix_summary"
    ownership_deal_size_table = f"{schema}.agreement_ownership_deal_size_summary"
    buyer_type_matrix_table = f"{schema}.agreement_buyer_type_matrix_summary"
    target_industry_table = f"{schema}.agreement_target_industry_summary"
    industry_pairing_table = f"{schema}.agreement_industry_pairing_summary"
    filter_option_table = f"{schema}.agreement_filter_option_summary"
    metadata_field_coverage_table = f"{schema}.agreement_metadata_field_coverage_summary"
    counsel_leaderboard_table = f"{schema}.agreement_counsel_leaderboard_summary"
    agreement_index_table = f"{schema}.agreement_index_summary"

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
            _ensure_agreement_overview_summary_table(
                conn,
                schema=schema,
                table="agreement_overview_summary",
            )
            _ensure_agreement_trends_summary_tables(conn, schema=schema)
            _ = conn.execute(text(f"TRUNCATE TABLE {summary_table}"))
            _ = conn.execute(text(f"TRUNCATE TABLE {status_summary_table}"))
            _ = conn.execute(text(f"TRUNCATE TABLE {deal_type_summary_table}"))
            _ = conn.execute(text(f"TRUNCATE TABLE {overview_summary_table}"))
            _ = conn.execute(text(f"TRUNCATE TABLE {ownership_mix_table}"))
            _ = conn.execute(text(f"TRUNCATE TABLE {ownership_deal_size_table}"))
            _ = conn.execute(text(f"TRUNCATE TABLE {buyer_type_matrix_table}"))
            _ = conn.execute(text(f"TRUNCATE TABLE {target_industry_table}"))
            _ = conn.execute(text(f"TRUNCATE TABLE {industry_pairing_table}"))
            _ = conn.execute(text(f"TRUNCATE TABLE {filter_option_table}"))
            _ = conn.execute(text(f"TRUNCATE TABLE {metadata_field_coverage_table}"))
            _ = conn.execute(text(f"TRUNCATE TABLE {counsel_leaderboard_table}"))
            _ = conn.execute(text(f"TRUNCATE TABLE {agreement_index_table}"))
            _build_summary_temp_tables(conn, schema=schema)
            _refresh_summary_table(conn, schema=schema)
            _refresh_status_summary_table(conn, schema=schema)
            _refresh_deal_type_summary_table(conn, schema=schema)
            _refresh_agreement_overview_summary_table(conn, schema=schema)
            _refresh_agreement_trends_summary_tables(conn, schema=schema)
            _refresh_filter_option_summary_table(conn, schema=schema)
            _refresh_metadata_field_coverage_summary_table(conn, schema=schema)
            _refresh_counsel_leaderboard_summary_table(conn, schema=schema)
            _refresh_agreement_index_summary_table(conn, schema=schema)
        finally:
            _ = conn.execute(
                text("SELECT RELEASE_LOCK(:lock_name)"), {"lock_name": lock_name}
            )

    if context is not None:
        context.log.info("summary_data refreshed.")
