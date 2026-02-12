"""Assemble tagged sections into XML documents for each agreement."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

import pandas as pd

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import bindparam, text

from etl.defs.c_tagging_asset import tagging_asset
from etl.defs.e_reconcile_tags import reconcile_tags
from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.f_xml import generate_xml
from etl.domain.z_gating import apply_gating, apply_tagged_outputs_gating, apply_xml_gating
from etl.utils.db_utils import upsert_xml
from etl.utils.run_config import is_batched, is_cleanup_mode
from etl.utils.summary_data import refresh_summary_data


@dg.asset(deps=[tagging_asset, reconcile_tags], name="6_xml_asset")
def xml_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> None:
    """
    Assemble tagged sections into XML documents.

    Re-creates XML for agreements where tagged_outputs have been updated since the last XML creation.
    Maintains version numbers and tracks creation dates.
    
    Args:
        context: Dagster execution context.
        db: Database resource for connection.
        pipeline_config: Pipeline configuration for mode.
    """
    # batching controls
    agreement_batch_size = pipeline_config.xml_agreement_batch_size
    batched = is_batched(context, pipeline_config)

    engine = db.get_engine()
    schema = db.database
    agreements_table = f"{schema}.agreements"
    pages_table = f"{schema}.pages"
    tagged_outputs_table = f"{schema}.tagged_outputs"
    xml_table = f"{schema}.xml"
    is_cleanup = is_cleanup_mode(context, pipeline_config)

    context.log.info(
        f"Running XML generation in {'CLEANUP' if is_cleanup else 'FROM_SCRATCH'} mode"
    )

    with engine.begin() as conn:
        _ = apply_tagged_outputs_gating(conn, db.database)

    last_uuid = ''
    while True:
        with engine.begin() as conn:
            # Fetch batch of agreements where:
            # 1. Either no XML exists yet, OR
            # 2. tagged_outputs have been updated after the last XML creation date
            agreement_uuids = (
                conn.execute(
                    text(
                        f"""
                        SELECT DISTINCT
                            a.agreement_uuid
                        FROM
                            {agreements_table} a
                        JOIN {pages_table} p
                            ON p.agreement_uuid = a.agreement_uuid
                        LEFT JOIN {tagged_outputs_table} t
                            ON t.page_uuid = p.page_uuid
                        LEFT JOIN {xml_table} x
                            ON x.agreement_uuid = a.agreement_uuid
                            AND x.latest = 1
                        LEFT JOIN (
                            SELECT DISTINCT p2.agreement_uuid
                            FROM {tagged_outputs_table} t2
                            JOIN {pages_table} p2
                                ON t2.page_uuid = p2.page_uuid
                            WHERE t2.gated = 1
                        ) gated
                            ON gated.agreement_uuid = a.agreement_uuid
                        WHERE a.agreement_uuid > :last_uuid
                        AND p.source_page_type = 'body'
                        AND COALESCE(t.tagged_text_gold, t.tagged_text_corrected, t.tagged_text) IS NOT NULL
                        AND gated.agreement_uuid IS NULL
                        AND (
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
                        GROUP BY a.agreement_uuid
                        HAVING
                        -- ensure that there is at least one body page and that all body pages are tagged
                        SUM(CASE WHEN p.source_page_type = 'body' THEN 1 ELSE 0 END) > 0
                        AND SUM(
                            CASE WHEN p.source_page_type = 'body'
                                AND COALESCE(t.tagged_text_corrected, t.tagged_text) IS NOT NULL
                                THEN 1 ELSE 0 END
                        ) = SUM(CASE WHEN p.source_page_type = 'body' THEN 1 ELSE 0 END)
                        ORDER BY a.agreement_uuid
                        LIMIT :limit;
                """
                    ),
                    {"limit": agreement_batch_size, "last_uuid": last_uuid},
                )
                .scalars()
                .all()
            )

            # If none left, we're done
            if not agreement_uuids:
                break

            # Fetch every page and its tagged output for those agreements
            rows = (
                conn.execute(
                    text(
                        f"""
                    SELECT
                    p.agreement_uuid,
                    p.page_uuid,
                    p.page_order,
                    coalesce(p.gold_label, p.source_page_type) as source_page_type,
                    coalesce(tgo.tagged_text_gold, tgo.tagged_text_corrected, tgo.tagged_text, p.processed_page_content) as tagged_output,
                    url,
                    acquirer,
                    target,
                    filing_date,
                    source_is_txt,
                    source_is_html
                    FROM {pages_table} p
                    JOIN {agreements_table} a on p.agreement_uuid = a.agreement_uuid
                    LEFT JOIN {tagged_outputs_table} tgo
                    ON p.page_uuid = tgo.page_uuid
                    WHERE p.agreement_uuid IN :uuids
                    ORDER BY p.agreement_uuid, p.page_order
                """
                    ),
                    {"uuids": tuple(agreement_uuids)},
                )
                .mappings()
                .fetchall()
            )

            df = pd.DataFrame(rows)
            # Determine version: new agreements get v1, updated pages increment version
            existing_versions = conn.execute(
                text(f"""
                    SELECT agreement_uuid, MAX(version) as max_version
                    FROM {xml_table}
                    WHERE agreement_uuid IN :uuids
                    GROUP BY agreement_uuid
                """),
                {"uuids": tuple(agreement_uuids)},
            ).mappings().fetchall()
            
            version_map = {row["agreement_uuid"]: row["max_version"] + 1 for row in existing_versions}
            
            # Generate XML from tagged pages
            xml, xml_generation_failures = generate_xml(df, version_map)
            for failure in xml_generation_failures:
                context.log.warning(
                    "Skipping XML generation due to parse error for agreement_uuid=%s: %s",
                    failure.agreement_uuid,
                    failure.error,
                )

            if not xml:
                context.log.warning(
                    "Skipping XML upsert for this batch because all %s agreements failed XML parsing",
                    len(agreement_uuids),
                )
                last_uuid = agreement_uuids[-1]
                if batched:
                    break
                continue

            generated_agreement_uuids = [item.agreement_uuid for item in xml]

            try:
                upsert_xml(xml, db.database, conn)
                _ = conn.execute(
                    text(
                        f"""
                        UPDATE {xml_table} x
                        JOIN (
                            SELECT agreement_uuid, MAX(version) AS max_version
                            FROM {xml_table}
                            WHERE agreement_uuid IN :uuids
                            GROUP BY agreement_uuid
                        ) m ON x.agreement_uuid = m.agreement_uuid
                        SET x.latest = CASE
                            WHEN x.version = m.max_version THEN 1
                            ELSE 0
                        END
                        WHERE x.agreement_uuid IN :uuids
                        """
                    ).bindparams(bindparam("uuids", expanding=True)),
                    {"uuids": generated_agreement_uuids},
                )
                context.log.info(
                    f"Successfully generated XML for {len(generated_agreement_uuids)} agreements"
                )
            except Exception as e:
                context.log.error(f"Error upserting XML: {e}")
                raise RuntimeError(e)
            
            last_uuid = agreement_uuids[-1]
        if batched:
            break

    with engine.begin() as conn:
        _ = apply_xml_gating(conn, db.database)
        _ = apply_gating(conn, db.database)

    refresh_summary_data(context, db)
