# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
import dagster as dg
from dagster import AssetExecutionContext

from etl.defs.resources import DBResource
from etl.domain.gating import apply_gating
from etl.utils.summary_data import refresh_summary_data


@dg.asset(name="99_gating")
def gating_asset(
    context: AssetExecutionContext,
    db: DBResource,
) -> None:
    """Manual maintenance asset for gating and summary refresh."""
    engine = db.get_engine()
    with engine.begin() as conn:
        context.log.info("Gating: applying criteria + priority updates.")
        gating_counts = apply_gating(conn, db.database)

    context.log.info(
        "Gating refreshed: agreements=%s, pages=%s, tagged_outputs=%s, xml=%s",
        gating_counts.agreements_gated,
        gating_counts.pages_gated,
        gating_counts.tagged_outputs_gated,
        gating_counts.xml_gated,
    )
    refresh_summary_data(context, db)
