# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import bindparam, text
from typing import cast

from etl.defs.g_sections_asset import sections_asset
from etl.defs.resources import DBResource, PipelineConfig, TaxonomyModel, TaxonomyMode
from etl.domain.h_taxonomy import (
    TaxonomyPredictor,
    TaxonomyRow,
    ContextProtocol as TaxonomyContext,
    apply_standard_ids_to_xml,
    predict_taxonomy,
)
from etl.domain.f_xml import XMLData
from etl.utils.db_utils import upsert_xml
from etl.utils.latest_sections_search import refresh_latest_sections_search
from etl.utils.post_asset_refresh import run_post_asset_refresh
from etl.utils.run_config import runs_single_batch


def _normalized_gold_label(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None
    cleaned = raw_value.strip()
    if cleaned in {"", "[]"}:
        return None
    return cleaned


@dg.asset(deps=[sections_asset], name="7_taxonomy_asset")
def taxonomy_asset(
    context: AssetExecutionContext,
    db: DBResource,
    taxonomy_model: TaxonomyModel,
    pipeline_config: PipelineConfig,
) -> None:
    # batching controls
    agreement_batch_size = pipeline_config.taxonomy_agreement_batch_size
    single_batch_run = runs_single_batch(context, pipeline_config)

    engine = db.get_engine()
    schema = db.database
    sections_table = f"{schema}.sections"
    xml_table = f"{schema}.xml"
    last_uuid = ""
    mode = pipeline_config.taxonomy_mode

    model: TaxonomyPredictor | None = None
    if mode == TaxonomyMode.INFERENCE:
        model = cast(TaxonomyPredictor, cast(object, taxonomy_model.model()))
    context.log.info("Running taxonomy in mode=%s", mode.value)

    while True:
        with engine.begin() as conn:
            # 1) Pick a batch of agreements for this mode
            agreement_where_clause = (
                "s.section_standard_id IS NULL"
                if mode == TaxonomyMode.INFERENCE
                else (
                    "s.section_standard_id_gold_label IS NOT NULL "
                    "AND TRIM(s.section_standard_id_gold_label) <> '' "
                    "AND TRIM(s.section_standard_id_gold_label) <> '[]'"
                )
            )
            agr_rows = (
                conn.execute(
                    text(
                        f"""
                        SELECT DISTINCT s.agreement_uuid
                        FROM {sections_table} s
                        JOIN {xml_table} x
                          ON x.agreement_uuid = s.agreement_uuid
                         AND x.version = s.xml_version
                        WHERE s.agreement_uuid > :last
                          AND x.latest = 1
                          AND x.status = 'verified'
                          AND {agreement_where_clause}
                        ORDER BY s.agreement_uuid
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

            # 2) Fetch sections and labels for this mode
            if mode == TaxonomyMode.INFERENCE:
                section_sql = text(
                    f"""
                    SELECT s.section_uuid,
                           s.agreement_uuid,
                           s.article_title,
                           s.section_title,
                           s.xml_content,
                           s.section_standard_id_gold_label
                    FROM {sections_table} s
                    JOIN {xml_table} x
                      ON x.agreement_uuid = s.agreement_uuid
                     AND x.version = s.xml_version
                    WHERE s.agreement_uuid IN :agreements
                      AND x.latest = 1
                      AND x.status = 'verified'
                      AND s.section_standard_id IS NULL
                    ORDER BY s.agreement_uuid, s.section_uuid
                    """
                ).bindparams(bindparam("agreements", expanding=True))
            else:
                section_sql = text(
                    f"""
                    SELECT s.section_uuid,
                           s.agreement_uuid,
                           s.section_standard_id_gold_label
                    FROM {sections_table} s
                    JOIN {xml_table} x
                      ON x.agreement_uuid = s.agreement_uuid
                     AND x.version = s.xml_version
                    WHERE s.agreement_uuid IN :agreements
                      AND x.latest = 1
                      AND x.status = 'verified'
                      AND s.section_standard_id_gold_label IS NOT NULL
                      AND TRIM(s.section_standard_id_gold_label) <> ''
                      AND TRIM(s.section_standard_id_gold_label) <> '[]'
                    ORDER BY s.agreement_uuid, s.section_uuid
                    """
                ).bindparams(bindparam("agreements", expanding=True))

            sec_rows = (
                conn.execute(
                    section_sql,
                    {"agreements": tuple(agr_list)},
                )
                .mappings()
                .fetchall()
            )

            if not sec_rows:
                last_uuid = agr_rows[-1]["agreement_uuid"]
                continue

            # 3) Build section updates + XML mapping
            upd_rows: list[dict[str, object]] = []
            by_agr: dict[str, dict[str, str]] = {}

            if mode == TaxonomyMode.INFERENCE:
                if model is None:
                    raise RuntimeError("Taxonomy model was not initialized for inference mode.")
                rows: list[TaxonomyRow] = [
                    cast(TaxonomyRow, cast(object, dict(r))) for r in sec_rows
                ]
                sec_idx, preds = predict_taxonomy(
                    rows, model, cast(TaxonomyContext, cast(object, context))
                )
                gold_label_by_section_uuid = {
                    cast(str, r["section_uuid"]): cast(str | None, r.get("section_standard_id_gold_label"))
                    for r in sec_rows
                }
                for meta, pred in zip(sec_idx, preds):
                    inferred_label = cast(str, cast(object, pred.get("label")))
                    alt_probs = pred.get("alt_probs") or [0.0, 0.0, 0.0]
                    upd_rows.append(
                        {
                            "section_uuid": meta["section_uuid"],
                            "label": inferred_label,
                            "a": float(alt_probs[0]) if len(alt_probs) > 0 else 0.0,
                            "b": float(alt_probs[1]) if len(alt_probs) > 1 else 0.0,
                            "c": float(alt_probs[2]) if len(alt_probs) > 2 else 0.0,
                        }
                    )

                    gold_label = gold_label_by_section_uuid.get(meta["section_uuid"])
                    normalized_gold_label = _normalized_gold_label(gold_label)
                    label_for_xml = normalized_gold_label if normalized_gold_label is not None else inferred_label
                    by_agr.setdefault(meta["agreement_uuid"], {})[
                        meta["section_uuid"]
                    ] = label_for_xml
            else:
                for row in sec_rows:
                    agreement_uuid = cast(str, row["agreement_uuid"])
                    section_uuid = cast(str, row["section_uuid"])
                    gold_label = _normalized_gold_label(
                        cast(str | None, row["section_standard_id_gold_label"])
                    )
                    if gold_label is None:
                        continue
                    by_agr.setdefault(agreement_uuid, {})[section_uuid] = gold_label

            if upd_rows:
                upd_sql = text(
                    f"""
                    UPDATE {sections_table}
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

            xml_rows = (
                conn.execute(
                    text(
                        f"""
                        SELECT m.agreement_uuid, m.xml, m.version
                        FROM {xml_table} m
                        WHERE m.agreement_uuid IN :agreements
                          AND m.latest = 1
                          AND m.status = 'verified'
                        """
                    ).bindparams(bindparam("agreements", expanding=True)),
                    {"agreements": tuple(agr_list)},
                )
                .mappings()
                .fetchall()
            )

            staged_xml: list[XMLData] = []
            for r in xml_rows:
                agr_uuid = r["agreement_uuid"]
                xml_str = r["xml"]
                xml_version = r.get("version", 1)  # Get the existing version
                mapping = by_agr.get(agr_uuid, {})
                if not mapping:
                    continue
                new_xml = apply_standard_ids_to_xml(xml_str, mapping)
                # Preserve the existing version—taxonomy updates don't increment
                staged_xml.append(XMLData(
                    agreement_uuid=agr_uuid,
                    xml=new_xml,
                    version=xml_version
                ))

            if staged_xml:
                upsert_xml(staged_xml, db.database, conn)
            refreshed = refresh_latest_sections_search(conn, db.database, agr_list)
            context.log.info(
                (
                    "taxonomy_asset: batch updated %s sections across %s agreements; "
                    "upserted %s XMLs; refreshed latest_sections_search rows=%s"
                ),
                len(upd_rows),
                len(agr_list),
                len(staged_xml),
                refreshed,
            )

            last_uuid = agr_rows[-1]["agreement_uuid"]

        if single_batch_run:
            break

    run_post_asset_refresh(context, db, pipeline_config)
