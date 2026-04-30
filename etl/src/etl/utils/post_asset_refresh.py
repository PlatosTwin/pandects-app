# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
"""Shared end-of-job refresh helper (gating + summary data)."""

from dagster import AssetExecutionContext
from sqlalchemy.engine import Connection

from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.gating import apply_gating
from etl.utils.summary_data import refresh_summary_data

_END_OF_JOB_REFRESH_ASSET_NAMES = {
    "07_taxonomy_asset",
    "10_tx_metadata_asset",
    "10-02_regular_ingest_tx_metadata_web_search_asset",
    "10-04_ingestion_cleanup_a_tx_metadata_web_search_asset",
    "10-06_ingestion_cleanup_b_tx_metadata_web_search_asset",
    "10-08_ingestion_cleanup_c_tx_metadata_web_search_asset",
    "10-10_ingestion_cleanup_d_tx_metadata_web_search_asset",
    "11_embed_sections",
}


def run_pre_asset_gating(
    context: AssetExecutionContext,
    db: DBResource,
    conn: Connection | None = None,
) -> None:
    """Run gating before asset logic to ensure gated filters are current."""
    if conn is None:
        with db.get_engine().begin() as gating_conn:
            counts = apply_gating(gating_conn, db.database)
    else:
        counts = apply_gating(conn, db.database)

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
    Run a final gating + summary refresh for terminal assets only.

    For non-terminal assets this is a no-op by design.
    """
    if not pipeline_config.refresh:
        return
    asset_name = context.asset_key.path[-1] if context.asset_key.path else ""
    if asset_name not in _END_OF_JOB_REFRESH_ASSET_NAMES:
        return

    if conn is None:
        with db.get_engine().begin() as refresh_conn:
            _ = apply_gating(refresh_conn, db.database)
    else:
        _ = apply_gating(conn, db.database)
    refresh_summary_data(context, db)
