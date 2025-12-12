import dagster as dg
from sqlalchemy import text

from etl.defs.f_xml_asset import xml_asset
from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.g_sections import extract_sections_from_xml
from etl.utils.db_utils import upsert_sections
from etl.utils.run_config import get_int_tag, is_batched


@dg.asset(deps=[xml_asset], name="7_sections_asset")
def sections_asset(context, db: DBResource, pipeline_config: PipelineConfig) -> None:
    # batching controls
    ag_bs_tag = get_int_tag(context, "agreement_batch_size")
    agreement_batch_size = ag_bs_tag if ag_bs_tag is not None else pipeline_config.xml_agreement_batch_size
    batched = is_batched(context, pipeline_config)

    engine = db.get_engine()
    last_uuid = ""

    while True:
        with engine.begin() as conn:
            rows = (
                conn.execute(
                    text(
                        """
                        SELECT m.xml, m.agreement_uuid
                        FROM pdx.xml AS m
                        LEFT JOIN pdx.sections AS s
                          ON m.agreement_uuid = s.agreement_uuid
                        WHERE m.agreement_uuid > :last
                          AND s.agreement_uuid IS NULL
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
                        }
                    )

            if staged:
                upsert_sections(staged, conn)
                context.log.info(f"sections_asset: upserted {len(staged)} sections from {len(rows)} agreements")

            last_uuid = rows[-1]["agreement_uuid"]

        if is_batched:
            break


