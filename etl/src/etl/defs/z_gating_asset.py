# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import text

from etl.defs.resources import DBResource
from etl.domain.z_gating import (
    MIN_ARTICLE_TAGS,
    apply_agreement_gating,
    apply_pages_gating,
    apply_tagged_outputs_gating,
    apply_xml_gating,
)


@dg.asset(name="z_gating")
def gating_asset(
    context: AssetExecutionContext,
    db: DBResource,
) -> None:
    engine = db.get_engine()
    with engine.begin() as conn:
        context.log.info("Gating: updating validation_priority.")
        _ = conn.execute(text(f"UPDATE {db.database}.agreements SET validation_priority = 1"))
        _ = conn.execute(text(f"UPDATE {db.database}.pages SET validation_priority = 1"))
        _ = conn.execute(text(f"UPDATE {db.database}.tagged_outputs SET validation_priority = 1"))

        context.log.info("Gating: applying agreements criteria.")
        agreements_gated = apply_agreement_gating(conn, db.database)
        context.log.info("Gating: applying pages criteria.")
        pages_gated = apply_pages_gating(conn, db.database)
        context.log.info("Gating: applying tagged_outputs criteria.")
        tagged_outputs_gated = apply_tagged_outputs_gating(conn, db.database)
        context.log.info("Gating: applying xml criteria.")
        xml_gated = apply_xml_gating(conn, db.database, min_article_tags=MIN_ARTICLE_TAGS)

    context.log.info(
        f"Gating refreshed: agreements={agreements_gated}, pages={pages_gated}, tagged_outputs={tagged_outputs_gated}, xml={xml_gated}"
    )
