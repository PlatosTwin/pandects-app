# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
"""Shared end-of-asset refresh helper (gating + summary data)."""

from dagster import AssetExecutionContext
from sqlalchemy.engine import Connection

from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.z_gating import MIN_ARTICLE_TAGS, apply_gating
from etl.utils.summary_data import refresh_summary_data


def run_post_asset_refresh(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    conn: Connection | None = None,
) -> None:
    """Run end-of-asset gating + summary refresh when enabled."""
    if not pipeline_config.refresh:
        context.log.info("Post-asset refresh disabled (pipeline_config.refresh=false).")
        return

    if conn is None:
        with db.get_engine().begin() as refresh_conn:
            _ = apply_gating(refresh_conn, db.database, min_article_tags=MIN_ARTICLE_TAGS)
    else:
        _ = apply_gating(conn, db.database, min_article_tags=MIN_ARTICLE_TAGS)
    refresh_summary_data(context, db)
