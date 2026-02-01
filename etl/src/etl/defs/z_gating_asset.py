# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
import dagster as dg
from dagster import AssetExecutionContext

from etl.defs.resources import DBResource
from etl.domain.z_gating import MIN_ARTICLE_TAGS, apply_gating


@dg.asset(name="z_gating")
def gating_asset(
    context: AssetExecutionContext,
    db: DBResource,
) -> None:
    engine = db.get_engine()
    with engine.begin() as conn:
        counts = apply_gating(conn, db.database, min_article_tags=MIN_ARTICLE_TAGS)

    context.log.info(
        f"Gating refreshed: agreements={counts.agreements_gated}, pages={counts.pages_gated}, tagged_outputs={counts.tagged_outputs_gated}, xml={counts.xml_gated}"
    )
