# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import text

from etl.defs.e_reconcile_tags import reconcile_tags
from etl.defs.f_xml_asset import xml_verify_asset
from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.g_sections import extract_sections_from_xml
from etl.domain.z_gating import apply_gating, apply_xml_gating
from etl.utils.db_utils import upsert_sections
from etl.utils.run_config import is_batched
from etl.utils.summary_data import refresh_summary_data


@dg.asset(deps=[xml_verify_asset, reconcile_tags], name="6_sections_asset")
def sections_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> None:
    # batching controls
    agreement_batch_size = pipeline_config.sections_agreement_batch_size
    batched = is_batched(context, pipeline_config)

    engine = db.get_engine()
    schema = db.database
    xml_table = f"{schema}.xml"
    sections_table = f"{schema}.sections"
    last_uuid = ""

    with engine.begin() as conn:
        _ = apply_xml_gating(conn, db.database)
        _ = apply_gating(conn, db.database)

    while True:
        with engine.begin() as conn:
            rows = (
                conn.execute(
                    text(
                        f"""
                        SELECT m.xml, m.agreement_uuid, m.version AS xml_version
                        FROM {xml_table} AS m
                        LEFT JOIN {sections_table} AS s
                          ON m.agreement_uuid = s.agreement_uuid
                            AND s.xml_version = m.version
                        WHERE m.agreement_uuid > :last
                          AND s.agreement_uuid IS NULL
                          AND m.latest = 1
                          AND m.gated = 0
                        ORDER BY m.agreement_uuid
                        LIMIT :lim
                        """
                    ),
                    {"last": last_uuid, "lim": agreement_batch_size},
                )
                .mappings()
                .fetchall()
            )

            if not rows:
                break

            staged = []
            for r in rows:
                xml_str = r["xml"]
                agr_uuid = r["agreement_uuid"]
                xml_version = r["xml_version"]
                secs = extract_sections_from_xml(xml_str)
                for s in secs:
                    staged.append(
                        {
                            "agreement_uuid": agr_uuid,
                            "section_uuid": s["section_uuid"],
                            "article_title": s["article_title"],
                            "article_title_normed": s["article_title_normed"],
                            "article_order": s.get("article_order"),
                            "section_title": s["section_title"],
                            "section_title_normed": s["section_title_normed"],
                            "section_order": s.get("section_order"),
                            "xml_content": s["xml_content"],
                            "xml_version": xml_version,
                        }
                    )

            if staged:
                upsert_sections(staged, db.database, conn)
                context.log.info(f"sections_asset: upserted {len(staged)} sections from {len(rows)} agreements")

            last_uuid = rows[-1]["agreement_uuid"]

        if batched:
            break

    refresh_summary_data(context, db)
