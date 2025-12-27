"""Transaction metadata enrichment via OpenAI web search (standard API).

This asset selects earliest agreements missing metadata, issues one standard
Responses API call per agreement with the web_search tool, parses the JSON
output directly from `response.output_text`, and updates `pdx.agreements`,
setting metadata=1 for successfully parsed rows (even if values are unknown/null).

Runs in batched mode only; in non-batched mode it logs a warning and exits.
Default batch size is configured via `PipelineConfig.tx_metadata_agreement_batch_size`.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

from __future__ import annotations

import os
from typing import Any, Dict, List, Tuple

import dagster as dg
from sqlalchemy import text
from openai import OpenAI

from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.i_tx_metadata import (
    build_tx_metadata_request_body,
    build_tx_metadata_update_params,
    parse_tx_metadata_response,
)
from etl.utils.run_config import is_batched


def _oai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required for tx_metadata_asset.")
    return OpenAI(api_key=api_key)


@dg.asset(deps=[], name="9_tx_metadata_asset")
def tx_metadata_asset(
    context: dg.AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> None:
    # Enforce batched-only mode
    if not is_batched(context, pipeline_config):
        context.log.warning("tx_metadata_asset runs only in batched mode; skipping.")
        return

    batch_size = pipeline_config.tx_metadata_agreement_batch_size

    engine = db.get_engine()

    # Select earliest agreements without metadata
    select_q = text(
        """
        SELECT 
            agreement_uuid, 
            target,
            acquirer,
            filing_date
        FROM 
            pdx.agreements
        WHERE 
            COALESCE(metadata, 0) = 0
        ORDER BY 
            (filing_date IS NULL) ASC, 
            filing_date ASC, 
            agreement_uuid ASC
        LIMIT :lim
        """
    )

    with engine.begin() as conn:
        rows = conn.execute(select_q, {"lim": batch_size}).mappings().fetchall()
        agreements = [dict(r) for r in rows]

    if not agreements:
        context.log.info("tx_metadata_asset: no agreements need metadata.")
        return

    context.log.info(f"tx_metadata_asset: selected {len(agreements)} agreements for enrichment")

    # Use standard Responses API per agreement (web_search is not supported in batch API)
    model_name = "gpt-5"
    client = _oai_client()

    success_data: List[Tuple[str, Dict[str, Any]]] = []  # (agreement_uuid, parsed_obj)
    attempted = 0
    parse_errors = 0

    for agreement in agreements:
        attempted += 1
        agr_uuid: str = agreement["agreement_uuid"]
        body = build_tx_metadata_request_body(agreement, model=model_name)
        try:
            resp = client.responses.create(**body)  # type: ignore[arg-type]
            obj = parse_tx_metadata_response(resp)
            success_data.append((agr_uuid, obj))
            context.log.info(f"tx_metadata_asset: successfully parsed response for {agr_uuid}")
        except Exception as e:
            parse_errors += 1
            context.log.warning(f"tx_metadata_asset: failed to parse response for {agr_uuid}: {e}")

    # Persist: only for parsed successes
    update_q = text(
        """
        UPDATE pdx.agreements
        SET 
            transaction_consideration = :consideration,
            transaction_price_cash = :price_cash,
            transaction_price_stock = :price_stock,
            transaction_price_assets = :price_assets,
            transaction_price_total = :price_total,
            target_type = :target_type,
            acquirer_type = :acquirer_type,
            target_pe = :target_pe,
            acquirer_pe = :acquirer_pe,
            target_industry = :target_industry,
            acquirer_industry = :acquirer_industry,
            sources = :sources,
            metadata = 1
        WHERE agreement_uuid = :uuid
        """
    )

    updated = 0
    skipped_due_to_error = 0
    param_rows: List[Dict[str, Any]] = []
    for uuid, obj in success_data:
        try:
            param_rows.append(
                build_tx_metadata_update_params(
                    agreement_uuid=uuid, tx_metadata_obj=obj
                )
            )
        except Exception as e:
            skipped_due_to_error += 1
            context.log.warning(f"tx_metadata_asset: invalid parsed object for {uuid}: {e}")

    if param_rows:
        with engine.begin() as conn:
            for params in param_rows:
                _ = conn.execute(update_q, params)
                updated += 1

    parsed_ok = len(success_data)
    context.log.info(
        f"tx_metadata_asset: attempted={attempted}, parsed={parsed_ok}, updated={updated}, parse_errors={parse_errors}, skipped_due_to_error={skipped_due_to_error}"
    )
