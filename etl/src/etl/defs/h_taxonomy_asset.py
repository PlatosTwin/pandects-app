# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import text
from typing import cast

from etl.defs.g_sections_asset import sections_asset
from etl.defs.resources import DBResource, PipelineConfig, TaxonomyModel
from etl.domain.h_taxonomy import (
    TaxonomyPredictor,
    TaxonomyRow,
    ContextProtocol as TaxonomyContext,
    apply_standard_ids_to_xml,
    predict_taxonomy,
)
from etl.domain.f_xml import XMLData
from etl.utils.db_utils import upsert_xml
from etl.utils.run_config import is_batched, is_cleanup_mode

@dg.asset(deps=[sections_asset], name="8_taxonomy_asset")
def taxonomy_asset(
    context: AssetExecutionContext,
    db: DBResource,
    taxonomy_model: TaxonomyModel,
    pipeline_config: PipelineConfig,
) -> None:
    # batching controls
    agreement_batch_size = pipeline_config.taxonomy_agreement_batch_size
    batched = is_batched(context, pipeline_config)

    engine = db.get_engine()
    last_uuid = ""

    model = cast(TaxonomyPredictor, cast(object, taxonomy_model.model()))
    is_cleanup = is_cleanup_mode(context, pipeline_config)
    context.log.info(
        f"Running taxonomy in {'CLEANUP' if is_cleanup else 'FROM_SCRATCH'} mode"
    )

    while True:
        with engine.begin() as conn:
            # 1) Pick a batch of agreements that still have sections without a taxonomy label
            agr_rows = (
                conn.execute(
                    text(
                        """
                        SELECT DISTINCT agreement_uuid
                        FROM pdx.sections
                        WHERE agreement_uuid > :last
                          AND section_standard_id IS NULL
                        ORDER BY agreement_uuid
                        LIMIT :lim
                        """
                    ),
                    {"last": last_uuid, "lim": agreement_batch_size},
                )
                .mappings()
                .fetchall()
            )

            if not agr_rows:
                break

            agr_list = [r["agreement_uuid"] for r in agr_rows]

            # 2) Fetch sections needing labels for these agreements
            sec_rows = (
                conn.execute(
                    text(
                        """
                        SELECT section_uuid,
                               agreement_uuid,
                               article_title,
                               section_title,
                               xml_content
                        FROM pdx.sections
                        WHERE agreement_uuid IN :agreements
                          AND section_standard_id IS NULL
                        ORDER BY agreement_uuid, section_uuid
                        """
                    ),
                    {"agreements": tuple(agr_list)},
                )
                .mappings()
                .fetchall()
            )

            if not sec_rows:
                last_uuid = agr_rows[-1]["agreement_uuid"]
                continue

            # 3) Prepare model inputs + predict
            rows: list[TaxonomyRow] = [
                cast(TaxonomyRow, cast(object, dict(r))) for r in sec_rows
            ]
            sec_idx, preds = predict_taxonomy(
                rows, model, cast(TaxonomyContext, cast(object, context))
            )

            # 4) Update pdx.sections with labels and alt probabilities
            upd_rows: list[dict[str, object]] = []
            for meta, pred in zip(sec_idx, preds):
                label = cast(str, cast(object, pred.get("label")))
                alt_probs = pred.get("alt_probs") or [0.0, 0.0, 0.0]
                upd_rows.append(
                    {
                        "section_uuid": meta["section_uuid"],
                        "label": label,
                        "a": float(alt_probs[0]) if len(alt_probs) > 0 else 0.0,
                        "b": float(alt_probs[1]) if len(alt_probs) > 1 else 0.0,
                        "c": float(alt_probs[2]) if len(alt_probs) > 2 else 0.0,
                    }
                )

            upd_sql = text(
                """
                UPDATE pdx.sections
                SET section_standard_id = :label,
                    alt_label_a_prob = :a,
                    alt_label_b_prob = :b,
                    alt_label_c_prob = :c
                WHERE section_uuid = :section_uuid
                """
            )
            for i in range(0, len(upd_rows), 250):
                batch = upd_rows[i : i + 250]
                _ = conn.execute(upd_sql, batch)

            # 5) Update pdx.xml documents in place with section standardId
            # Build per-agreement mapping of section_uuid -> label
            by_agr: dict[str, dict[str, str]] = {}
            for meta, pred in zip(sec_idx, preds):
                label = cast(str, cast(object, pred.get("label")))
                by_agr.setdefault(meta["agreement_uuid"], {})[
                    meta["section_uuid"]
                ] = label

            xml_rows = (
                conn.execute(
                    text(
                        """
                        SELECT agreement_uuid, xml
                        FROM pdx.xml
                        WHERE agreement_uuid IN :agreements
                          AND is_latest_version = 1
                        """
                    ),
                    {"agreements": tuple(agr_list)},
                )
                .mappings()
                .fetchall()
            )

            staged_xml: list[XMLData] = []
            for r in xml_rows:
                agr_uuid = r["agreement_uuid"]
                xml_str = r["xml"]
                mapping = by_agr.get(agr_uuid, {})
                if not mapping:
                    continue
                new_xml = apply_standard_ids_to_xml(xml_str, mapping)
                staged_xml.append(XMLData(agreement_uuid=agr_uuid, xml=new_xml))

            if staged_xml:
                upsert_xml(staged_xml, conn)
            context.log.info(
                f"taxonomy_asset: batch updated {len(upd_rows)} sections across {len(agr_list)} agreements; upserted {len(staged_xml)} XMLs"
            )

            last_uuid = agr_rows[-1]["agreement_uuid"]

        if batched:
            break
