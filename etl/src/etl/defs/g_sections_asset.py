# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
from typing import List, Optional

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import bindparam, text

from etl.defs.e_reconcile_tags import reconcile_tags
from etl.defs.f_xml_asset import xml_verify_asset
from etl.defs.f_xml_repair_cycle_asset import post_repair_verify_xml_asset
from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.g_sections import extract_sections_from_xml
from etl.utils.db_utils import upsert_sections
from etl.utils.post_asset_refresh import run_post_asset_refresh
from etl.utils.latest_sections_search import refresh_latest_sections_search
from etl.utils.pipeline_state_sql import canonical_fresh_sections_queue_sql
from etl.utils.run_config import runs_single_batch


def _run_sections_for_agreements(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    *,
    target_agreement_uuids: Optional[List[str]],
    log_prefix: str,
) -> List[str]:
    agreement_batch_size = pipeline_config.xml_agreement_batch_size
    single_batch_run = runs_single_batch(context, pipeline_config)

    engine = db.get_engine()
    schema = db.database
    xml_table = f"{schema}.xml"
    last_uuid = ""
    processed_agreement_uuids: List[str] = []

    scoped_uuids = sorted(set(target_agreement_uuids or []))
    use_scope = len(scoped_uuids) > 0
    if use_scope:
        if len(scoped_uuids) > agreement_batch_size:
            raise ValueError(
                f"{log_prefix}: received more upstream agreements than xml_agreement_batch_size; "
                + "run-scoped sections extraction accepts at most one upstream XML batch."
            )

    while True:
        with engine.begin() as conn:
            if use_scope:
                agreement_uuids = (
                    conn.execute(
                        text(canonical_fresh_sections_queue_sql(schema, scoped=True)).bindparams(
                            bindparam("auuids", expanding=True)
                        ),
                        {"auuids": tuple(scoped_uuids), "lim": agreement_batch_size},
                    )
                    .scalars()
                    .all()
                )
            else:
                agreement_uuids = (
                    conn.execute(
                        text(canonical_fresh_sections_queue_sql(schema, scoped=False)),
                        {"last_uuid": last_uuid, "lim": agreement_batch_size},
                    )
                    .scalars()
                    .all()
                )

            if not agreement_uuids:
                break

            rows = (
                conn.execute(
                    text(
                        f"""
                        SELECT m.xml, m.agreement_uuid, m.version AS xml_version
                        FROM {xml_table} AS m
                        WHERE m.agreement_uuid IN :auuids
                          AND m.latest = 1
                        ORDER BY m.agreement_uuid
                        """
                    ).bindparams(bindparam("auuids", expanding=True)),
                    {"auuids": tuple(agreement_uuids)},
                )
                .mappings()
                .fetchall()
            )

            staged = []
            for r in rows:
                xml_str = r["xml"]
                agr_uuid = str(r["agreement_uuid"])
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
                processed_agreement_uuids.append(agr_uuid)

            if staged:
                upsert_sections(staged, db.database, conn)
                context.log.info(
                    "%s: upserted %s sections from %s agreements",
                    log_prefix,
                    len(staged),
                    len(rows),
                )
            if rows:
                refreshed = refresh_latest_sections_search(conn, db.database, agreement_uuids)
                context.log.info(
                    "%s: refreshed latest_sections_search for %s agreements (%s rows).",
                    log_prefix,
                    len(agreement_uuids),
                    refreshed,
                )

            if use_scope:
                break

            last_uuid = str(agreement_uuids[-1])

        if single_batch_run:
            break

    run_post_asset_refresh(context, db, pipeline_config)
    return sorted(set(processed_agreement_uuids))


@dg.asset(deps=[xml_verify_asset, reconcile_tags, post_repair_verify_xml_asset], name="6_sections_asset")
def sections_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> List[str]:
    # Backward-compatible generic sections run (not run-scoped).
    return _run_sections_for_agreements(
        context,
        db,
        pipeline_config,
        target_agreement_uuids=None,
        log_prefix="sections_asset",
    )


@dg.asset(
    name="6-1_sections_from_fresh_xml",
    ins={"verified_fresh_agreement_uuids": dg.AssetIn(key=xml_verify_asset.key)},
)
def sections_from_fresh_xml_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    verified_fresh_agreement_uuids: List[str],
) -> List[str]:
    scoped_uuids = sorted(set(verified_fresh_agreement_uuids))
    if len(scoped_uuids) > pipeline_config.xml_agreement_batch_size:
        context.log.warning(
            "sections_from_fresh_xml_asset: upstream agreement scope has %s uuids, "
            + "which exceeds xml_agreement_batch_size=%s; falling back to canonical sections queue.",
            len(scoped_uuids),
            pipeline_config.xml_agreement_batch_size,
        )
        target_agreement_uuids = None
    else:
        target_agreement_uuids = scoped_uuids or None
    return _run_sections_for_agreements(
        context,
        db,
        pipeline_config,
        target_agreement_uuids=target_agreement_uuids,
        log_prefix="sections_from_fresh_xml_asset",
    )


@dg.asset(
    name="6-2_sections_from_repair_xml",
    ins={"verified_repair_agreement_uuids": dg.AssetIn(key=post_repair_verify_xml_asset.key)},
)
def sections_from_repair_xml_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    verified_repair_agreement_uuids: List[str],
) -> List[str]:
    return _run_sections_for_agreements(
        context,
        db,
        pipeline_config,
        target_agreement_uuids=verified_repair_agreement_uuids,
        log_prefix="sections_from_repair_xml_asset",
    )
