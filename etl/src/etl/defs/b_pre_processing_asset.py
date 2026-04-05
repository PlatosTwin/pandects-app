"""Split agreements into pages, classify page types, and format text.

Respects pre_processing_mode via `PipelineConfig`.
Writes pages to `pdx.pages`; FROM_SCRATCH inserts, CLEANUP updates.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

from typing import cast

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import bindparam, text

from etl.defs.a_staging_asset import regular_ingest_staging_asset, staging_asset
from etl.defs.resources import (
    ClassifierModel,
    DBResource,
    PipelineConfig,
    ReviewModel,
)
from etl.domain.b_pre_processing import (
    AgreementRow,
    ClassifierModelProtocol,
    CleanupRow,
    ContextProtocol as PreProcessContext,
    ReviewModelProtocol,
    cleanup,
    pre_process,
)
from etl.utils.db_utils import upsert_pages
from etl.utils.post_asset_refresh import run_post_asset_refresh, run_pre_asset_gating
from etl.utils.pipeline_state_sql import canonical_pre_processing_queue_sql
from etl.utils.run_config import is_pre_processing_cleanup_mode, runs_single_batch


def _run_pre_processing_from_scratch(
    context: AssetExecutionContext,
    *,
    db: DBResource,
    pipeline_config: PipelineConfig,
    classifier_model: ClassifierModel,
    review_model: ReviewModel,
    target_agreement_uuids: list[str] | None,
    log_prefix: str,
) -> list[str]:
    last_uuid = ""
    engine = db.get_engine()
    raw_inference_model = classifier_model.model()
    inference_model = cast(
        ClassifierModelProtocol,
        cast(object, raw_inference_model),
    )
    review_inference_model = cast(
        ReviewModelProtocol,
        cast(object, review_model.model(page_classifier=raw_inference_model)),
    )
    single_batch_run = runs_single_batch(context, pipeline_config)
    batch_size = pipeline_config.pre_processing_agreement_batch_size
    schema = db.database
    agreements_table = f"{schema}.agreements"
    scoped_uuids = sorted(set(target_agreement_uuids or []))
    processed_agreement_uuids: set[str] = set()

    while True:
        with engine.begin() as conn:
            if scoped_uuids:
                agreement_uuids = (
                    conn.execute(
                        text(canonical_pre_processing_queue_sql(schema, scoped=True)).bindparams(
                            bindparam("auuids", expanding=True)
                        ),
                        {
                            "auuids": tuple(scoped_uuids),
                            "batch_size": max(batch_size, len(scoped_uuids)),
                        },
                    )
                    .scalars()
                    .all()
                )
            else:
                agreement_uuids = (
                    conn.execute(
                        text(canonical_pre_processing_queue_sql(schema)),
                        {"last_uuid": last_uuid, "batch_size": batch_size},
                    )
                    .scalars()
                    .all()
                )
            if not agreement_uuids:
                break

            rows = (
                conn.execute(
                    text(
                        f"""
                        SELECT agreement_uuid, url
                        FROM {agreements_table}
                        WHERE agreement_uuid IN :uuids
                        ORDER BY agreement_uuid ASC
                        """
                    ).bindparams(bindparam("uuids", expanding=True)),
                    {"uuids": agreement_uuids},
                )
                .fetchall()
            )

            if not rows:
                break

            agreements: list[AgreementRow] = [
                {"agreement_uuid": r[0], "url": r[1]} for r in rows
            ]
            staged_pages, pagination_statuses = pre_process(
                cast(PreProcessContext, cast(object, context)),
                agreements,
                inference_model,
                review_inference_model,
            )

            if pagination_statuses:
                _ = conn.execute(
                    text(
                        f"""
                    UPDATE {agreements_table}
                    SET paginated = :paginated
                    WHERE agreement_uuid = :agreement_uuid
                      AND NOT (paginated <=> :paginated)
                    """
                    ),
                    [
                        {
                            "agreement_uuid": agreement_uuid,
                            "paginated": paginated,
                        }
                        for agreement_uuid, paginated in pagination_statuses.items()
                    ],
                )
                processed_agreement_uuids.update(str(agreement_uuid) for agreement_uuid in pagination_statuses)

            if staged_pages:
                try:
                    upsert_pages(staged_pages, "insert", db.database, conn)
                    agreement_count = len(
                        {p.agreement_uuid for p in staged_pages if p.agreement_uuid}
                    )
                    processed_agreement_uuids.update(
                        str(p.agreement_uuid)
                        for p in staged_pages
                        if p.agreement_uuid
                    )
                    context.log.info(
                        "%s: successfully processed %s pages from %s agreements",
                        log_prefix,
                        len(staged_pages),
                        agreement_count,
                    )
                except Exception as e:
                    context.log.error(f"{log_prefix}: error upserting pages: {e}")
                    raise RuntimeError(e)

            if scoped_uuids:
                break

            last_uuid = rows[-1][0]

        if single_batch_run:
            break

    return sorted(processed_agreement_uuids)


@dg.asset(deps=[staging_asset], name="2_pre_processing_asset")
def pre_processing_asset(
    context: AssetExecutionContext,
    db: DBResource,
    classifier_model: ClassifierModel,
    review_model: ReviewModel,
    pipeline_config: PipelineConfig,
) -> None:
    """Split agreements into pages, classify page types, and process HTML into formatted text.

    In cleanup mode, processes only existing unprocessed agreements.

    Args:
        context: Dagster execution context.
        db: Database resource for connection.
        classifier_model: Model for page classification.
        review_model: Model for agreement-level manual-review prediction.
        pipeline_config: Pipeline configuration.
    """
    last_uuid: str = ""
    engine = db.get_engine()
    raw_inference_model = classifier_model.model()
    inference_model = cast(
        ClassifierModelProtocol,
        cast(object, raw_inference_model),
    )
    review_inference_model = cast(
        ReviewModelProtocol,
        cast(object, review_model.model(page_classifier=raw_inference_model)),
    )

    is_cleanup = is_pre_processing_cleanup_mode(context, pipeline_config)
    single_batch_run = runs_single_batch(context, pipeline_config)

    batch_size = pipeline_config.pre_processing_agreement_batch_size
    schema = db.database
    pages_table = f"{schema}.pages"

    run_pre_asset_gating(context, db)
    
    if not is_cleanup:
        context.log.info("Running pre-processing in FROM_SCRATCH mode")
        processed_agreement_uuids = _run_pre_processing_from_scratch(
            context,
            db=db,
            pipeline_config=pipeline_config,
            classifier_model=classifier_model,
            review_model=review_model,
            target_agreement_uuids=None,
            log_prefix="pre_processing_asset",
        )
        context.log.info(
            "pre_processing_asset: processed %s agreements in FROM_SCRATCH mode",
            len(processed_agreement_uuids),
        )

    else:
        context.log.info("Running pre-processing in CLEANUP mode")

        totals_dict = {"agreements": 0, "pages": 0}

        # Since we've already split the agreements into pages, we can just fetch the pages
        while True:
            with engine.begin() as conn:
                # Identify agreements with missing derived outputs, then fetch all pages for
                # those agreements so the CRF and review model always see the full sequence.
                # Use UNION to split OR for index-friendly execution.
                result = conn.execute(
                    text(
                        f"""
                    WITH
                    batch_a AS (
                        SELECT DISTINCT agreement_uuid
                        FROM {pages_table}
                        WHERE agreement_uuid > :last_uuid
                          AND gold_label IS NULL
                          AND processed_page_content IS NULL
                    ),
                    batch_b AS (
                        SELECT DISTINCT agreement_uuid
                        FROM {pages_table}
                        WHERE agreement_uuid > :last_uuid
                          AND gold_label IS NULL
                          AND source_page_type IS NULL
                    ),
                    batch_c AS (
                        SELECT DISTINCT agreement_uuid
                        FROM {pages_table}
                        WHERE agreement_uuid > :last_uuid
                          AND gold_label IS NULL
                          AND review_flag IS NULL
                    ),
                    batch_d AS (
                        SELECT DISTINCT agreement_uuid
                        FROM {pages_table}
                        WHERE agreement_uuid > :last_uuid
                          AND gold_label IS NULL
                          AND validation_priority IS NULL
                    ),
                    agreement_batch AS (
                        (SELECT agreement_uuid FROM batch_a)
                        UNION
                        (SELECT agreement_uuid FROM batch_b)
                        UNION
                        (SELECT agreement_uuid FROM batch_c)
                        UNION
                        (SELECT agreement_uuid FROM batch_d)
                        ORDER BY 1
                        LIMIT :batch_size
                    )
                    SELECT
                        p.agreement_uuid,
                        p.page_uuid,
                        p.raw_page_content,
                        p.page_order,
                        p.source_is_txt,
                        p.source_is_html,
                        p.gold_label,
                        p.processed_page_content,
                        p.source_page_type,
                        p.page_type_prob_front_matter,
                        p.page_type_prob_toc,
                        p.page_type_prob_body,
                        p.page_type_prob_sig,
                        p.page_type_prob_back_matter,
                        p.postprocess_modified,
                        p.review_flag,
                        p.validation_priority
                    FROM {pages_table} p
                    WHERE p.agreement_uuid IN (SELECT agreement_uuid FROM agreement_batch)
                    ORDER BY p.page_order, p.page_uuid
                    """
                    ),
                    {"last_uuid": last_uuid, "batch_size": batch_size},
                )
                rows = result.fetchall()
                if not rows:
                    break
                last_uuid = max(r[0] for r in rows)

                # Process existing pages
                pages = [
                    cast(
                        CleanupRow,
                        cast(
                            object,
                            {
                                "agreement_uuid": cast(str, r[0]),
                                "page_uuid": cast(str | None, r[1]),
                                "content": cast(str, r[2]),
                                "page_order": cast(int, r[3]),
                                "is_txt": cast(bool, r[4]),
                                "is_html": cast(bool, r[5]),
                                "gold_label": cast(str | None, r[6]),
                                "processed_page_content": cast(str | None, r[7]),
                                "source_page_type": cast(str | None, r[8]),
                                "page_type_prob_front_matter": cast(float | None, r[9]),
                                "page_type_prob_toc": cast(float | None, r[10]),
                                "page_type_prob_body": cast(float | None, r[11]),
                                "page_type_prob_sig": cast(float | None, r[12]),
                                "page_type_prob_back_matter": cast(float | None, r[13]),
                                "postprocess_modified": cast(bool | None, r[14]),
                                "review_flag": cast(bool | None, r[15]),
                                "validation_priority": cast(float | None, r[16]),
                            },
                        ),
                    )
                    for r in rows
                ]
                staged_pages = cleanup(pages, inference_model, review_inference_model)

                if staged_pages:
                    try:
                        upsert_pages(
                            staged_pages, "update", db.database, conn
                        )
                        
                        num_agr = len(set(p["agreement_uuid"] for p in pages))
                        num_pages = len(staged_pages)
                        totals_dict["agreements"] += num_agr
                        totals_dict["pages"] += num_pages

                        context.log.info(
                            f"Successfully updated {num_pages} pages from {num_agr} agreements. Total pages: {totals_dict['pages']}. Total agreements: {totals_dict['agreements']}"
                        )
                        
                    except Exception as e:
                        context.log.error(f"Error upserting pages: {e}")
                        raise RuntimeError(e)

            if single_batch_run:
                break

    run_post_asset_refresh(context, db, pipeline_config)


@dg.asset(
    name="2-1_regular_ingest_pre_processing_asset",
    ins={"staged_agreement_uuids": dg.AssetIn(key=regular_ingest_staging_asset.key)},
)
def regular_ingest_pre_processing_asset(
    context: AssetExecutionContext,
    db: DBResource,
    classifier_model: ClassifierModel,
    review_model: ReviewModel,
    pipeline_config: PipelineConfig,
    staged_agreement_uuids: list[str],
) -> list[str]:
    run_pre_asset_gating(context, db)
    if is_pre_processing_cleanup_mode(context, pipeline_config):
        raise ValueError("regular_ingest_pre_processing_asset requires FROM_SCRATCH pre_processing_mode.")
    processed_agreement_uuids = _run_pre_processing_from_scratch(
        context,
        db=db,
        pipeline_config=pipeline_config,
        classifier_model=classifier_model,
        review_model=review_model,
        target_agreement_uuids=staged_agreement_uuids,
        log_prefix="regular_ingest_pre_processing_asset",
    )
    run_post_asset_refresh(context, db, pipeline_config)
    return processed_agreement_uuids
