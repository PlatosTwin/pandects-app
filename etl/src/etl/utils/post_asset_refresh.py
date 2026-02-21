# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
"""Shared end-of-asset refresh helper (gating + summary data)."""

from dagster import AssetExecutionContext
from sqlalchemy.engine import Connection

from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.z_gating import MIN_ARTICLE_TAGS, apply_gating
from etl.utils.summary_data import refresh_summary_data

_MAIN_STAGE_REFRESH_ASSET_NAMES = {
    "1_staging_asset",
    "2_pre_processing_asset",
    "3_tagging_asset",
    "6_sections_asset",
    "6-1_sections_from_fresh_xml",
    "6-2_sections_from_repair_xml",
}


def run_pre_asset_gating(
    context: AssetExecutionContext,
    db: DBResource,
    conn: Connection | None = None,
) -> None:
    """Run gating before asset logic to ensure gated filters are current."""
    if conn is None:
        with db.get_engine().begin() as gating_conn:
            counts = apply_gating(gating_conn, db.database, min_article_tags=MIN_ARTICLE_TAGS)
    else:
        counts = apply_gating(conn, db.database, min_article_tags=MIN_ARTICLE_TAGS)

    context.log.info(
        "Pre-asset gating refreshed: agreements=%s pages=%s tagged_outputs=%s xml=%s",
        counts.agreements_gated,
        counts.pages_gated,
        counts.tagged_outputs_gated,
        counts.xml_gated,
    )


def run_post_asset_refresh(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    conn: Connection | None = None,
) -> None:
    """
    Run end-of-stage gating + summary refresh for the main stage-ending assets.

    For non-stage-ending assets this is a no-op by design.
    """
    if not pipeline_config.refresh:
        return
    asset_name = context.asset_key.path[-1] if context.asset_key.path else ""
    if asset_name not in _MAIN_STAGE_REFRESH_ASSET_NAMES:
        return

    if conn is None:
        with db.get_engine().begin() as refresh_conn:
            _ = apply_gating(refresh_conn, db.database, min_article_tags=MIN_ARTICLE_TAGS)
    else:
        _ = apply_gating(conn, db.database, min_article_tags=MIN_ARTICLE_TAGS)
    refresh_summary_data(context, db)
