import dagster as dg
from sqlalchemy import text
from typing import Any, Dict, List

from etl.defs.g_sections_asset import sections_asset
from etl.defs.resources import DBResource, PipelineConfig, TaxonomyModel
from etl.domain.h_taxonomy import strip_xml_tags_to_text, apply_standard_ids_to_xml
from etl.domain.xml import XMLData
from etl.utils.db_utils import upsert_xml


@dg.asset(deps=[sections_asset], name="8_taxonomy_asset")
def taxonomy_asset(
    context,
    db: DBResource,
    taxonomy_model: TaxonomyModel,
    pipeline_config: PipelineConfig,
) -> None:
    # batching controls
    ag_bs_tag = context.run.tags.get("agreement_batch_size")
    run_scope_tag = context.run.tags.get("run_scope")
    agreement_batch_size: int = int(ag_bs_tag) if ag_bs_tag else pipeline_config.agreement_batch_size
    is_batched: bool = (
        run_scope_tag == "batched"
        if run_scope_tag is not None
        else pipeline_config.is_batched()
    )

    engine = db.get_engine()
    last_uuid = ""

    model = taxonomy_model.model()

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

            # 3) Prepare model inputs
            inputs: List[Dict[str, Any]] = []
            sec_idx: List[Dict[str, Any]] = []  # keep metadata alongside
            for r in sec_rows:
                text_block = strip_xml_tags_to_text(r["xml_content"])
                inputs.append(
                    {
                        "article_title": r.get("article_title") or "",
                        "section_title": r.get("section_title") or "",
                        "section_text": text_block,
                    }
                )
                sec_idx.append({
                    "section_uuid": r["section_uuid"],
                    "agreement_uuid": r["agreement_uuid"],
                })

            preds = model.predict(inputs)

            # 4) Update pdx.sections with labels and alt probabilities
            upd_rows: List[Dict[str, Any]] = []
            for meta, pred in zip(sec_idx, preds):
                label = pred.get("label")
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
                conn.execute(upd_sql, batch)

            # 5) Update pdx.xml documents in place with section standardId
            # Build per-agreement mapping of section_uuid -> label
            by_agr: Dict[str, Dict[str, str]] = {}
            for meta, pred in zip(sec_idx, preds):
                by_agr.setdefault(meta["agreement_uuid"], {})[meta["section_uuid"]] = pred.get("label")

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

            staged_xml: List[XMLData] = []
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
                    f"taxonomy_asset: updated {len(upd_rows)} sections across {len(agr_list)} agreements; upserted {len(staged_xml)} XMLs"
                )

            last_uuid = agr_rows[-1]["agreement_uuid"]

        if is_batched:
            break


