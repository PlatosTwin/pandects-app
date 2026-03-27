"""Post-repair XML rebuild and verification assets for the repair cycle job."""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportPrivateUsage=false, reportExplicitAny=false

import io
import json
import xml.etree.ElementTree as ET
from typing import Any, Dict, List

import dagster as dg
import pandas as pd
from dagster import AssetExecutionContext
from sqlalchemy import bindparam, text

from etl.defs.e_reconcile_tags import reconcile_tags
from etl.defs.f_xml_asset import (
    XML_REASON_TAG_TREE_RENDER_FAILURE,
    XML_REASON_XML_PARSE_FAILURE,
    XML_VERIFY_BATCH_SCOPE_REPAIR,
    _apply_xml_verify_batch_output,
    _build_xml_verify_batch_request_body,
    _fetch_unpulled_xml_verify_batch,
    _load_xml_verify_batch_agreement_uuids,
    find_hard_rule_violations,
    _mark_xml_verify_batch_pulled,
    _oai_client,
    _parse_custom_id,
    _render_tag_tree_from_root,
    _resume_xml_verify_batch,
    _set_xml_status_with_reasons,
    _upsert_xml_verify_batch_row,
)
from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.f_xml import generate_xml
from etl.utils.db_utils import upsert_xml
from etl.utils.batch_keys import agreement_version_batch_key
from etl.utils.openai_batch import poll_batch_until_terminal
from etl.utils.post_asset_refresh import run_post_asset_refresh
from etl.utils.latest_sections_search import refresh_latest_sections_search
from etl.utils.pipeline_state_sql import (
    canonical_post_repair_build_queue_sql,
    canonical_post_repair_verify_queue_sql,
)
from etl.utils.schema_guards import assert_tables_exist


@dg.asset(
    name="5-4_post_repair_build_xml",
    ins={"reconciled_agreement_uuids": dg.AssetIn(key=reconcile_tags.key)},
)
def post_repair_build_xml_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    reconciled_agreement_uuids: List[str],
) -> List[str]:
    """
    Rebuild XML only for agreements that:
      - currently have latest XML marked invalid,
      - have been attempted by AI repair, and
      - have newer tagged outputs than that invalid XML version.
    """
    agreement_batch_size = pipeline_config.xml_agreement_batch_size
    target_agreement_uuids = sorted(set(reconciled_agreement_uuids))
    if not target_agreement_uuids:
        context.log.info("post_repair_build_xml_asset: no upstream agreements from reconcile_tags.")
        run_post_asset_refresh(context, db, pipeline_config)
        return []
    if len(target_agreement_uuids) > agreement_batch_size:
        raise ValueError(
            "post_repair_build_xml_asset received more upstream agreements than xml_agreement_batch_size; "
            + "run-scoped XML rebuild accepts at most one upstream reconciliation batch."
        )

    engine = db.get_engine()
    schema = db.database
    agreements_table = f"{schema}.agreements"
    pages_table = f"{schema}.pages"
    tagged_outputs_table = f"{schema}.tagged_outputs"
    xml_table = f"{schema}.xml"

    if pipeline_config.resume_openai_batches:
        with engine.begin() as conn:
            stranded_verify_batch = _fetch_unpulled_xml_verify_batch(
                conn,
                schema,
                batch_scope=XML_VERIFY_BATCH_SCOPE_REPAIR,
            )
        if stranded_verify_batch is not None:
            context.log.info(
                "post_repair_build_xml_asset: deferring new rebuilds because unpulled verify batch %s is waiting to resume.",
                stranded_verify_batch["batch_id"],
            )
            run_post_asset_refresh(context, db, pipeline_config)
            return []

    with engine.begin() as conn:
        agreement_uuids = (
            conn.execute(
                text(canonical_post_repair_build_queue_sql(schema, scoped=True)).bindparams(
                    bindparam("target_uuids", expanding=True)
                ),
                {"limit": agreement_batch_size, "target_uuids": target_agreement_uuids},
            )
            .scalars()
            .all()
        )

        if not agreement_uuids:
            run_post_asset_refresh(context, db, pipeline_config)
            return []

        rows = (
            conn.execute(
                text(
                    f"""
                    SELECT
                        p.agreement_uuid,
                        p.page_uuid,
                        p.page_order,
                        COALESCE(p.gold_label, p.source_page_type) AS source_page_type,
                        CASE
                            WHEN COALESCE(p.gold_label, p.source_page_type) = 'body' THEN
                                COALESCE(
                                    tgo.tagged_text_gold,
                                    tgo.tagged_text_corrected,
                                    tgo.tagged_text,
                                    p.processed_page_content
                                )
                            ELSE p.processed_page_content
                        END AS tagged_output,
                        url,
                        acquirer,
                        target,
                        filing_date,
                        source_is_txt,
                        source_is_html
                    FROM {pages_table} p
                    JOIN {agreements_table} a
                        ON p.agreement_uuid = a.agreement_uuid
                    LEFT JOIN {tagged_outputs_table} tgo
                        ON p.page_uuid = tgo.page_uuid
                    WHERE p.agreement_uuid IN :uuids
                    ORDER BY p.agreement_uuid, p.page_order
                    """
                ),
                {"uuids": tuple(agreement_uuids)},
            )
            .mappings()
            .fetchall()
        )

        df = pd.DataFrame(rows)
        existing_versions = conn.execute(
            text(
                f"""
                SELECT agreement_uuid, MAX(version) AS max_version
                FROM {xml_table}
                WHERE agreement_uuid IN :uuids
                GROUP BY agreement_uuid
                """
            ),
            {"uuids": tuple(agreement_uuids)},
        ).mappings().fetchall()
        version_map = {
            str(row["agreement_uuid"]): int(row["max_version"]) + 1
            for row in existing_versions
        }

        xml, xml_generation_failures = generate_xml(df, version_map)
        for failure in xml_generation_failures:
            context.log.warning(
                "post_repair_build_xml_asset: skipping XML generation due to parse error for agreement_uuid=%s: %s",
                failure.agreement_uuid,
                failure.error,
            )

        if not xml:
            context.log.warning(
                "post_repair_build_xml_asset: all %s agreements failed XML parsing for this batch.",
                len(agreement_uuids),
            )
            run_post_asset_refresh(context, db, pipeline_config)
            return []

        generated_agreement_uuids = [str(item.agreement_uuid) for item in xml]
        upsert_xml(xml, db.database, conn)
        _ = conn.execute(
            text(
                f"""
                UPDATE {xml_table} x
                JOIN (
                    SELECT agreement_uuid, MAX(version) AS max_version
                    FROM {xml_table}
                    WHERE agreement_uuid IN :uuids
                    GROUP BY agreement_uuid
                ) m ON x.agreement_uuid = m.agreement_uuid
                SET x.latest = CASE
                    WHEN x.version = m.max_version THEN 1
                    ELSE 0
                END
                WHERE x.agreement_uuid IN :uuids
                """
            ).bindparams(bindparam("uuids", expanding=True)),
            {"uuids": generated_agreement_uuids},
        )
        marked = conn.execute(
            text(
                f"""
                UPDATE {xml_table}
                SET ai_repair_attempted = 1
                WHERE agreement_uuid IN :uuids
                  AND latest = 1
                  AND NOT (ai_repair_attempted <=> 1)
                """
            ).bindparams(bindparam("uuids", expanding=True)),
            {"uuids": generated_agreement_uuids},
        ).rowcount or 0
        refreshed = refresh_latest_sections_search(
            conn,
            db.database,
            generated_agreement_uuids,
        )
        context.log.info(
            "post_repair_build_xml_asset: generated XML for %s agreements; marked ai_repair_attempted on %s latest rows; refreshed latest_sections_search rows=%s.",
            len(generated_agreement_uuids),
            int(marked),
            refreshed,
        )

    run_post_asset_refresh(context, db, pipeline_config)
    return sorted(set(generated_agreement_uuids))


@dg.asset(
    name="5-5_post_repair_verify_xml",
    ins={"rebuilt_agreement_uuids": dg.AssetIn(key=post_repair_build_xml_asset.key)},
)
def post_repair_verify_xml_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
    rebuilt_agreement_uuids: List[str],
) -> List[str]:
    """
    Verify only latest XML rows that came through the AI-repair cycle.
    """
    agreement_batch_size = pipeline_config.xml_agreement_batch_size
    resume_openai_batches = pipeline_config.resume_openai_batches
    target_agreement_uuids = sorted(set(rebuilt_agreement_uuids))

    engine = db.get_engine()
    schema = db.database
    xml_table = f"{schema}.xml"
    client = _oai_client()

    with engine.begin() as conn:
        assert_tables_exist(conn, schema=schema, table_names=("xml_verify_batches", "xml_status_reasons"))

    if resume_openai_batches:
        with engine.begin() as conn:
            stranded_batch = _fetch_unpulled_xml_verify_batch(
                conn,
                schema,
                batch_scope=XML_VERIFY_BATCH_SCOPE_REPAIR,
            )
        if stranded_batch is not None:
            try:
                stranded_agreement_uuids = _load_xml_verify_batch_agreement_uuids(
                    client,
                    stranded_batch,
                )
            except Exception as e:
                context.log.warning(
                    "post_repair_verify_xml_asset: failed to load agreement scope for unpulled batch %s: %s",
                    stranded_batch["batch_id"],
                    e,
                )
            else:
                target_scope = set(target_agreement_uuids)
                stranded_scope = set(stranded_agreement_uuids)
                if not target_scope or target_scope != stranded_scope:
                    return _resume_xml_verify_batch(
                        context,
                        engine,
                        db,
                        pipeline_config,
                        client,
                        schema=schema,
                        xml_table=xml_table,
                        batch_scope=XML_VERIFY_BATCH_SCOPE_REPAIR,
                        batch_row=stranded_batch,
                        agreement_uuids=stranded_agreement_uuids,
                        log_prefix="post_repair_verify_xml_asset",
                        hard_invalid_updated=0,
                    )

    if not target_agreement_uuids:
        context.log.info("post_repair_verify_xml_asset: no upstream agreements from post_repair_build_xml_asset.")
        run_post_asset_refresh(context, db, pipeline_config)
        return []
    if len(target_agreement_uuids) > agreement_batch_size:
        raise ValueError(
            "post_repair_verify_xml_asset received more upstream agreements than xml_agreement_batch_size; "
            + "run-scoped XML verification accepts at most one upstream rebuild batch."
        )

    queue_q = text(canonical_post_repair_verify_queue_sql(schema, scoped=True)).bindparams(
        bindparam("auuids", expanding=True)
    )
    with engine.begin() as conn:
        eligible_uuids = conn.execute(
            queue_q,
            {"lim": agreement_batch_size, "auuids": target_agreement_uuids},
        ).scalars().all()
    if not eligible_uuids:
        context.log.info(
            "post_repair_verify_xml_asset: no upstream-selected latest ai_repair_attempted XML rows with status IS NULL."
        )
        run_post_asset_refresh(context, db, pipeline_config)
        return []

    select_q = text(
        f"""
        SELECT agreement_uuid, version, xml
        FROM {xml_table}
        WHERE agreement_uuid IN :auuids
          AND latest = 1
        ORDER BY agreement_uuid ASC
        """
    ).bindparams(bindparam("auuids", expanding=True))
    with engine.begin() as conn:
        rows = conn.execute(
            select_q,
            {"auuids": tuple(eligible_uuids)},
        ).mappings().fetchall()

    selected_for_verify = [str(row["agreement_uuid"]) for row in rows]

    lines: List[Dict[str, Any]] = []
    hard_invalid_rows: List[Dict[str, Any]] = []
    for row in rows:
        agreement_uuid = str(row["agreement_uuid"])
        version = int(row["version"])
        xml_text = row["xml"]
        try:
            root = ET.fromstring(str(xml_text))
        except Exception as e:
            hard_invalid_rows.append(
                {
                    "agreement_uuid": agreement_uuid,
                    "version": version,
                    "reason_rows": [
                        {
                            "reason_code": XML_REASON_XML_PARSE_FAILURE,
                            "reason_detail": f"XML parse failure: {e}",
                            "page_uuid": None,
                        }
                    ],
                }
            )
            continue

        hard_rule_violations = find_hard_rule_violations(root)
        if hard_rule_violations:
            reason_rows: List[Dict[str, Any]] = []
            for violation in hard_rule_violations:
                if violation.page_uuids:
                    for page_uuid in violation.page_uuids:
                        reason_rows.append(
                            {
                                "reason_code": violation.reason_code,
                                "reason_detail": violation.reason_detail,
                                "page_uuid": page_uuid,
                            }
                        )
                else:
                    reason_rows.append(
                        {
                            "reason_code": violation.reason_code,
                            "reason_detail": violation.reason_detail,
                            "page_uuid": None,
                        }
                    )
            hard_invalid_rows.append(
                {
                    "agreement_uuid": agreement_uuid,
                    "version": version,
                    "reason_rows": reason_rows,
                }
            )
            continue

        try:
            tag_tree = _render_tag_tree_from_root(root)
        except Exception as e:
            hard_invalid_rows.append(
                {
                    "agreement_uuid": agreement_uuid,
                    "version": version,
                    "reason_rows": [
                        {
                            "reason_code": XML_REASON_TAG_TREE_RENDER_FAILURE,
                            "reason_detail": f"Tag tree render failure: {e}",
                            "page_uuid": None,
                        }
                    ],
                }
            )
            continue

        custom_id = f"{agreement_uuid}|{version}"
        lines.append(
            _build_xml_verify_batch_request_body(
                custom_id=custom_id,
                tag_tree=tag_tree,
                model="gpt-5-mini",
            )
        )

    hard_invalid_updated = 0
    if hard_invalid_rows:
        xml_status_reasons_table = f"{schema}.xml_status_reasons"
        with engine.begin() as conn:
            for row in hard_invalid_rows:
                hard_invalid_updated += _set_xml_status_with_reasons(
                    conn,
                    xml_table,
                    xml_status_reasons_table,
                    agreement_uuid=str(row["agreement_uuid"]),
                    version=int(row["version"]),
                    status="invalid",
                    reason_rows=list(row["reason_rows"]),
                )

        context.log.info(
            "post_repair_verify_xml_asset: hard-rule invalidated %s XML rows before LLM.",
            len(hard_invalid_rows),
        )

    if not lines:
        context.log.info(
            "post_repair_verify_xml_asset: no LLM submissions required after hard-rule checks; hard_invalid_updated=%s",
            hard_invalid_updated,
        )
        run_post_asset_refresh(context, db, pipeline_config)
        return selected_for_verify

    llm_targets = sorted({_parse_custom_id(str(line["custom_id"])) for line in lines})
    if not llm_targets:
        raise ValueError(
            "post_repair_verify_xml_asset: no (agreement_uuid, version) targets derived from LLM lines."
        )
    verify_batch_key = agreement_version_batch_key(llm_targets)
    context.log.info(
        "post_repair_verify_xml_asset: selected agreements=%s, llm_requests=%s, hard_invalid=%s",
        len(selected_for_verify),
        len(lines),
        len(hard_invalid_rows),
    )

    if resume_openai_batches:
        with engine.begin() as conn:
            existing_batch = _fetch_unpulled_xml_verify_batch(
                conn,
                schema,
                batch_scope=XML_VERIFY_BATCH_SCOPE_REPAIR,
                batch_key=verify_batch_key,
            )
        if existing_batch is not None:
            context.log.info(
                "post_repair_verify_xml_asset: resuming matching unpulled batch %s for batch_key=%s.",
                existing_batch["batch_id"],
                verify_batch_key[:12],
            )
            return _resume_xml_verify_batch(
                context,
                engine,
                db,
                pipeline_config,
                client,
                schema=schema,
                xml_table=xml_table,
                batch_scope=XML_VERIFY_BATCH_SCOPE_REPAIR,
                batch_row=existing_batch,
                agreement_uuids=selected_for_verify,
                log_prefix="post_repair_verify_xml_asset",
                hard_invalid_updated=hard_invalid_updated,
            )

    jsonl_buf = io.StringIO()
    for line in lines:
        _ = jsonl_buf.write(json.dumps(line, ensure_ascii=False) + "\n")
    jsonl_bytes = io.BytesIO(jsonl_buf.getvalue().encode("utf-8"))
    jsonl_bytes.name = "post_repair_xml_verify_requests.jsonl"

    input_file = client.files.create(purpose="batch", file=jsonl_bytes)
    completion_window = "24h"
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
    )
    with engine.begin() as conn:
        _upsert_xml_verify_batch_row(
            conn,
            schema,
            batch=batch,
            completion_window=completion_window,
            request_total=len(lines),
            batch_scope=XML_VERIFY_BATCH_SCOPE_REPAIR,
            batch_key=verify_batch_key,
        )
    context.log.info(
        "post_repair_verify_xml_asset: created batch %s with %s requests; polling until complete.",
        batch.id,
        len(lines),
    )

    final_batch = poll_batch_until_terminal(
        context,
        client,
        batch.id,
        log_prefix="post_repair_verify_xml_asset",
    )
    with engine.begin() as conn:
        _upsert_xml_verify_batch_row(
            conn,
            schema,
            batch=final_batch,
            completion_window=completion_window,
            request_total=len(lines),
            batch_scope=XML_VERIFY_BATCH_SCOPE_REPAIR,
            batch_key=verify_batch_key,
        )

    if final_batch.status != "completed":
        context.log.warning(
            "post_repair_verify_xml_asset: batch %s ended with status=%s; no status updates applied.",
            final_batch.id,
            final_batch.status,
        )
        with engine.begin() as conn:
            _mark_xml_verify_batch_pulled(conn, schema, final_batch.id)
        run_post_asset_refresh(context, db, pipeline_config)
        return selected_for_verify

    updated, parse_errors = _apply_xml_verify_batch_output(
        context=context,
        engine=engine,
        client=client,
        xml_table=xml_table,
        xml_status_reasons_table=f"{schema}.xml_status_reasons",
        batch=final_batch,
        log_prefix="post_repair_verify_xml_asset",
    )
    with engine.begin() as conn:
        _mark_xml_verify_batch_pulled(conn, schema, final_batch.id)
    context.log.info(
        "post_repair_verify_xml_asset: batch %s completed; updated=%s, parse_errors=%s, hard_invalid_updated=%s",
        final_batch.id,
        updated,
        parse_errors,
        hard_invalid_updated,
    )

    run_post_asset_refresh(context, db, pipeline_config)
    return selected_for_verify
