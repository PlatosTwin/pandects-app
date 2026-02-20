"""Canonical SQL selectors for stage state, queue selection, and gating alignment."""

from __future__ import annotations


def _table_names(schema: str) -> dict[str, str]:
    return {
        "agreements": f"{schema}.agreements",
        "pages": f"{schema}.pages",
        "tagged_outputs": f"{schema}.tagged_outputs",
        "xml": f"{schema}.xml",
        "sections": f"{schema}.sections",
        "ai_repair_processed_spans": f"{schema}.ai_repair_processed_spans",
    }


def canonical_components_cte_sql(schema: str) -> str:
    tables = _table_names(schema)
    agreements_table = tables["agreements"]
    pages_table = tables["pages"]
    tagged_outputs_table = tables["tagged_outputs"]
    xml_table = tables["xml"]
    sections_table = tables["sections"]

    return f"""
    WITH page_counts AS (
        SELECT
            agreement_uuid,
            COUNT(*) AS page_count
        FROM {pages_table}
        WHERE agreement_uuid IS NOT NULL
        GROUP BY agreement_uuid
    ),
    body_tag_state AS (
        SELECT
            p.agreement_uuid,
            SUM(
                CASE
                    WHEN COALESCE(p.gold_label, p.source_page_type) = 'body' THEN 1
                    ELSE 0
                END
            ) AS body_page_count,
            SUM(
                CASE
                    WHEN COALESCE(p.gold_label, p.source_page_type) = 'body'
                        AND t.page_uuid IS NOT NULL
                    THEN 1
                    ELSE 0
                END
            ) AS tagged_body_page_count
        FROM {pages_table} p
        LEFT JOIN {tagged_outputs_table} t
            ON t.page_uuid = p.page_uuid
        WHERE p.agreement_uuid IS NOT NULL
        GROUP BY p.agreement_uuid
    ),
    tagged_body_updated AS (
        SELECT
            p.agreement_uuid,
            MAX(t.updated_date) AS max_body_tagged_updated_date
        FROM {pages_table} p
        JOIN {tagged_outputs_table} t
            ON t.page_uuid = p.page_uuid
        WHERE p.agreement_uuid IS NOT NULL
          AND COALESCE(p.gold_label, p.source_page_type) = 'body'
        GROUP BY p.agreement_uuid
    ),
    xml_latest AS (
        SELECT
            agreement_uuid,
            version,
            created_date,
            status,
            status_reason_code,
            ai_repair_attempted
        FROM {xml_table}
        WHERE latest = 1
          AND agreement_uuid IS NOT NULL
    ),
    sections_latest AS (
        SELECT
            x.agreement_uuid,
            COUNT(s.section_uuid) AS section_count
        FROM xml_latest x
        LEFT JOIN {sections_table} s
            ON s.agreement_uuid = x.agreement_uuid
           AND s.xml_version = x.version
        GROUP BY x.agreement_uuid
    ),
    page_gating_agreements AS (
        SELECT
            agreement_uuid
        FROM {pages_table}
        WHERE agreement_uuid IS NOT NULL
        GROUP BY agreement_uuid
        HAVING
            SUM(
                CASE
                    WHEN source_page_type = 'back_matter' THEN 1
                    ELSE 0
                END
            ) >= 1
            AND SUM(
                CASE
                    WHEN source_page_type = 'sig' THEN 1
                    ELSE 0
                END
            ) = 0
            AND SUM(
                CASE
                    WHEN gold_label IS NOT NULL THEN 1
                    ELSE 0
                END
            ) = 0
    ),
    state_components AS (
        SELECT
            a.agreement_uuid,
            YEAR(DATE(a.filing_date)) AS filing_year,
            CASE
                WHEN a.source = 'dma' THEN 0
                WHEN a.status IS NULL THEN 1
                ELSE 0
            END AS agreement_is_gated,
            CASE
                WHEN pga.agreement_uuid IS NULL THEN 0
                ELSE 1
            END AS page_is_gated,
            CASE
                WHEN a.paginated = FALSE THEN 1
                ELSE 0
            END AS is_not_paginated,
            COALESCE(pc.page_count, 0) AS page_count,
            COALESCE(bts.body_page_count, 0) AS body_page_count,
            COALESCE(bts.tagged_body_page_count, 0) AS tagged_body_page_count,
            CASE
                WHEN xl.agreement_uuid IS NULL THEN 0
                ELSE 1
            END AS has_latest_xml,
            xl.version AS latest_xml_version,
            xl.status AS latest_xml_status,
            xl.status_reason_code AS latest_xml_reason_code,
            COALESCE(xl.ai_repair_attempted, 0) AS latest_xml_ai_repair_attempted,
            xl.created_date AS latest_xml_created_date,
            COALESCE(sl.section_count, 0) AS latest_section_count,
            CASE
                WHEN tbu.max_body_tagged_updated_date IS NOT NULL
                     AND xl.created_date IS NOT NULL
                     AND tbu.max_body_tagged_updated_date > xl.created_date
                THEN 1
                ELSE 0
            END AS has_stale_body_tags
        FROM {agreements_table} a
        LEFT JOIN page_counts pc
            ON pc.agreement_uuid = a.agreement_uuid
        LEFT JOIN body_tag_state bts
            ON bts.agreement_uuid = a.agreement_uuid
        LEFT JOIN tagged_body_updated tbu
            ON tbu.agreement_uuid = a.agreement_uuid
        LEFT JOIN xml_latest xl
            ON xl.agreement_uuid = a.agreement_uuid
        LEFT JOIN sections_latest sl
            ON sl.agreement_uuid = a.agreement_uuid
        LEFT JOIN page_gating_agreements pga
            ON pga.agreement_uuid = a.agreement_uuid
    )
    """


def _stage_current_case_sql() -> str:
    return """
    CASE
        WHEN is_not_paginated = 1 THEN '1_pre_processing'
        WHEN page_count = 0 THEN '0_staging'
        WHEN body_page_count = 0 OR tagged_body_page_count < body_page_count THEN '1_pre_processing'
        WHEN has_latest_xml = 0 THEN '2_tagging'
        WHEN latest_xml_status IS NULL THEN '3_xml'
        WHEN latest_xml_status = 'invalid' THEN '3_xml'
        WHEN latest_xml_status = 'verified'
             AND (latest_section_count = 0 OR has_stale_body_tags = 1) THEN '3_xml'
        WHEN latest_xml_status = 'verified'
             AND latest_section_count > 0
             AND has_stale_body_tags = 0 THEN 'processed'
        ELSE '3_xml'
    END
    """


def _stage_color_case_sql() -> str:
    return """
    CASE
        WHEN is_not_paginated = 1 THEN 'gray'
        WHEN page_count = 0 THEN CASE WHEN agreement_is_gated = 1 THEN 'red' ELSE 'yellow' END
        WHEN body_page_count = 0 THEN 'red'
        WHEN tagged_body_page_count < body_page_count THEN CASE WHEN page_is_gated = 1 THEN 'red' ELSE 'yellow' END
        WHEN has_latest_xml = 0 THEN 'yellow'
        WHEN latest_xml_status IS NULL THEN 'red'
        WHEN latest_xml_status = 'invalid' THEN 'red'
        WHEN latest_xml_status = 'verified'
             AND (latest_section_count = 0 OR has_stale_body_tags = 1) THEN 'yellow'
        WHEN latest_xml_status = 'verified'
             AND latest_section_count > 0
             AND has_stale_body_tags = 0 THEN 'green'
        ELSE 'red'
    END
    """


def canonical_stage_state_sql(schema: str, *, include_year: bool = False) -> str:
    year_col = "filing_year AS year," if include_year else ""
    return f"""
    {canonical_components_cte_sql(schema)}
    SELECT
        agreement_uuid,
        {year_col}
        {_stage_current_case_sql()} AS current_stage,
        {_stage_color_case_sql()} AS color
    FROM state_components
    """


def canonical_pre_processing_queue_sql(schema: str) -> str:
    return f"""
    {canonical_components_cte_sql(schema)}
    SELECT agreement_uuid
    FROM state_components
    WHERE is_not_paginated = 0
      AND page_count = 0
      AND agreement_is_gated = 0
      AND agreement_uuid > :last_uuid
    ORDER BY agreement_uuid ASC
    LIMIT :batch_size
    """


def canonical_tagging_queue_sql(schema: str) -> str:
    return f"""
    {canonical_components_cte_sql(schema)}
    SELECT agreement_uuid
    FROM state_components
    WHERE is_not_paginated = 0
      AND page_count > 0
      AND body_page_count > 0
      AND tagged_body_page_count < body_page_count
      AND page_is_gated = 0
      AND agreement_uuid > :last_uuid
    ORDER BY agreement_uuid ASC
    LIMIT :batch_size
    """


def canonical_fresh_xml_build_queue_sql(schema: str) -> str:
    return f"""
    {canonical_components_cte_sql(schema)}
    SELECT agreement_uuid
    FROM state_components
    WHERE agreement_uuid > :last_uuid
      AND body_page_count > 0
      AND tagged_body_page_count = body_page_count
      AND (
            has_latest_xml = 0
            OR (
                has_latest_xml = 1
                AND latest_xml_status = 'verified'
                AND has_stale_body_tags = 1
            )
      )
    ORDER BY agreement_uuid ASC
    LIMIT :limit
    """


def canonical_fresh_xml_verify_queue_sql(schema: str, *, scoped: bool = False) -> str:
    scoped_clause = "AND agreement_uuid IN :auuids" if scoped else "AND agreement_uuid > :last_uuid"
    return f"""
    {canonical_components_cte_sql(schema)}
    SELECT agreement_uuid
    FROM state_components
    WHERE has_latest_xml = 1
      AND latest_xml_status IS NULL
      AND latest_xml_ai_repair_attempted = 0
      {scoped_clause}
    ORDER BY agreement_uuid ASC
    LIMIT :lim
    """


def canonical_fresh_sections_queue_sql(schema: str, *, scoped: bool = False) -> str:
    scoped_clause = "AND agreement_uuid IN :auuids" if scoped else "AND agreement_uuid > :last_uuid"
    return f"""
    {canonical_components_cte_sql(schema)}
    SELECT agreement_uuid
    FROM state_components
    WHERE has_latest_xml = 1
      AND latest_xml_status = 'verified'
      AND latest_section_count = 0
      AND has_stale_body_tags = 0
      {scoped_clause}
    ORDER BY agreement_uuid ASC
    LIMIT :lim
    """


def canonical_post_repair_build_queue_sql(schema: str, *, scoped: bool = False) -> str:
    scoped_clause = "AND agreement_uuid IN :target_uuids" if scoped else ""
    return f"""
    {canonical_components_cte_sql(schema)}
    SELECT agreement_uuid
    FROM state_components
    WHERE has_latest_xml = 1
      AND latest_xml_status = 'invalid'
      AND latest_xml_ai_repair_attempted = 1
      AND body_page_count > 0
      AND tagged_body_page_count = body_page_count
      AND has_stale_body_tags = 1
      {scoped_clause}
    ORDER BY agreement_uuid ASC
    LIMIT :limit
    """


def canonical_post_repair_verify_queue_sql(schema: str, *, scoped: bool = False) -> str:
    scoped_clause = "AND agreement_uuid IN :auuids" if scoped else ""
    return f"""
    {canonical_components_cte_sql(schema)}
    SELECT agreement_uuid
    FROM state_components
    WHERE has_latest_xml = 1
      AND latest_xml_status IS NULL
      AND latest_xml_ai_repair_attempted = 1
      {scoped_clause}
    ORDER BY agreement_uuid ASC
    LIMIT :lim
    """


def canonical_ai_repair_enqueue_queue_sql(
    schema: str,
    *,
    status_clause_sql: str,
) -> str:
    tables = _table_names(schema)
    pages_table = tables["pages"]
    tagged_outputs_table = tables["tagged_outputs"]
    ai_repair_processed_spans_table = tables["ai_repair_processed_spans"]

    return f"""
    {canonical_components_cte_sql(schema)},
    eligible_agreements AS (
        SELECT agreement_uuid
        FROM state_components
        WHERE has_latest_xml = 1
          AND latest_xml_status = 'invalid'
          AND latest_xml_reason_code IN :reason_codes
    )
    SELECT
        p.agreement_uuid
    FROM
        {pages_table} p
        JOIN eligible_agreements ea
            ON ea.agreement_uuid = p.agreement_uuid
        JOIN {tagged_outputs_table} t
            USING (page_uuid)
        CROSS JOIN JSON_TABLE(
            t.spans,
            '$[*]' COLUMNS (
                entity VARCHAR(255) PATH '$.entity',
                start_char INT PATH '$.start_char',
                end_char INT PATH '$.end_char'
            )
        ) AS jt
    WHERE
        NOT EXISTS (
            SELECT 1
            FROM {ai_repair_processed_spans_table} ps
            WHERE ps.page_uuid = p.page_uuid
              AND ps.entity = CAST(jt.entity AS CHAR) COLLATE utf8mb4_unicode_ci
              AND ps.start_char = jt.start_char
              AND ps.end_char = jt.end_char
              AND ps.status IN {status_clause_sql}
        )
    GROUP BY
        p.agreement_uuid
    ORDER BY
        COUNT(DISTINCT p.page_uuid) ASC,
        p.agreement_uuid
    LIMIT
        :lim
    """
