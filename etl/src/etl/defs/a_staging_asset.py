# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
# pyright: reportMissingTypeStubs=false
"""Stage new filings and record pipeline run metadata.

Respects CLEANUP vs FROM_SCRATCH modes via `PipelineConfig`.
Returns the number of new filings processed.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, cast

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import text
from sqlalchemy.engine import Connection

from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.a_staging import (
    FilingMetadata,
    fetch_new_filings_sec_index,
    # To switch to DMA corpus flow, uncomment the line below and comment out fetch_new_filings_sec_index:
    # fetch_new_filings_dma_corpus,
)
from etl.models.exhibit_classifier.exhibit_classifier import ExhibitClassifier
from etl.utils.db_utils import upsert_agreements as _upsert_agreements
from etl.utils.post_asset_refresh import run_post_asset_refresh
from etl.utils.run_config import is_cleanup_mode

UpsertAgreements = Callable[[Sequence[FilingMetadata], str, Connection], None]
upsert_agreements = cast(UpsertAgreements, _upsert_agreements)


def _get_exhibit_classifier_path() -> Path:
    """Get the path to the exhibit classifier model."""
    base_dir = Path(__file__).resolve().parents[1]
    return base_dir / "models" / "exhibit_classifier" / "model_files" / "exhibit-classifier.joblib"


class _DagsterLogAdapter:
    """Adapts Dagster context.log to the _Logger protocol expected by staging functions."""

    def __init__(self, context: AssetExecutionContext):
        self._context = context

    def info(self, msg: str) -> None:
        self._context.log.info(msg)


class _DagsterContextAdapter:
    """Adapts AssetExecutionContext to the _Context protocol expected by staging functions."""

    def __init__(self, context: AssetExecutionContext):
        self._log = _DagsterLogAdapter(context)

    @property
    def log(self) -> _DagsterLogAdapter:
        return self._log


def _update_pipeline_run_progress(
    conn: Connection,
    pipeline_runs_table: str,
    run_id: int,
    pulled_to: datetime,
    rows_inserted: int,
) -> None:
    _ = conn.execute(
        text(
            f"""
            UPDATE {pipeline_runs_table}
            SET last_pulled_to = :pulled_to,
                rows_inserted = :count
            WHERE run_id = :run_id
            """
        ),
        {"run_id": run_id, "pulled_to": pulled_to, "count": rows_inserted},
    )


@dg.asset(name="1_staging_asset")
def staging_asset(
    context: AssetExecutionContext, 
    db: DBResource, 
    pipeline_config: PipelineConfig
) -> int:
    """Stage new filings day-by-day with incremental commits.
    
    Processes one day at a time, committing after each day. If the run crashes
    mid-way (e.g., on day 89 of 90), the next run resumes from where it left off
    rather than re-processing all days.
    
    In cleanup mode, skips fetching new filings and only processes existing 
    unprocessed agreements.
    
    Args:
        context: Dagster execution context.
        db: Database resource for connection.
        pipeline_config: Pipeline configuration for mode.
        
    Returns:
        Total number of new filings processed across all days.
    """
    is_cleanup = is_cleanup_mode(context, pipeline_config)

    if is_cleanup:
        context.log.info("CLEANUP mode. Skipping staging step.")
        run_post_asset_refresh(context, db, pipeline_config)
        return 0

    engine = db.get_engine()
    schema = db.database
    pipeline_runs_table = f"{schema}.pipeline_runs"
    context.log.info("Running staging in FROM_SCRATCH mode")

    # Get the most recent pull timestamp (even from a failed/terminated run)
    with engine.begin() as conn:
        last_run: datetime | None = conn.execute(
            text(
                f"""
                SELECT last_pulled_to
                FROM {pipeline_runs_table}
                WHERE last_pulled_to IS NOT NULL
                ORDER BY last_pulled_to DESC, run_time DESC
                LIMIT 1
                """
            )
        ).scalar_one_or_none()
        if last_run is None:
            last_run = datetime(2020, 12, 31)

    # Load exhibit classifier once for all days
    classifier = ExhibitClassifier.load(_get_exhibit_classifier_path())
    staging_context = _DagsterContextAdapter(context)

    # Calculate total days to process
    days_to_fetch = pipeline_config.staging_days_to_fetch
    
    # Insert STARTED record for this run and get the run_id.
    # Use date-level granularity (midnight) for last_pulled_from/to
    last_run_date = datetime.combine(last_run.date(), datetime.min.time())
    with engine.begin() as conn:
        result = conn.execute(
            text(
                f"""
                INSERT INTO {pipeline_runs_table}
                (run_time, last_pulled_from, last_pulled_to, status, rows_inserted)
                VALUES (UTC_TIMESTAMP(), :from_ts, :to_ts, 'STARTED', 0)
                """
            ),
            {
                "from_ts": last_run_date,
                "to_ts": last_run_date,  # Will be updated after successful completion
            },
        )
        run_id = result.lastrowid

    total_count = 0
    base_date = last_run.date()
    latest_pulled_to = last_run_date
    
    try:
        # Process day by day
        # Note: get_sec_index_urls fetches index for (start_date + day_offset), so we pass
        # base_date as start_date and let the function fetch the correct day's index.
        for day_offset in range(days_to_fetch):
            # The actual index date being fetched (start_date + 1 day due to how get_sec_index_urls works)
            index_date = base_date + timedelta(days=day_offset + 1)
            start_date_for_fetch = (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")
            context.log.info(
                f"Processing day {day_offset + 1}/{days_to_fetch}: fetching index for {index_date}"
            )
            
            # Fetch and classify filings for this single day
            filings = fetch_new_filings_sec_index(
                exhibit_classifier=classifier,
                context=staging_context,
                start_date=start_date_for_fetch,
                pipeline_config=pipeline_config,
                days_override=1,  # Process exactly one day
            )
            # To switch to DMA corpus: 
            # filings = fetch_new_filings_dma_corpus(since=start_date_for_fetch)
            
            day_count = len(filings)
            total_count += day_count
            
            # Commit this day's filings; keep pipeline_runs untouched until the end.
            # Each day is its own transaction for crash recovery.
            with engine.begin() as conn:
                if filings:
                    try:
                        upsert_agreements(filings, db.database, conn)
                        context.log.info(f"  Upserted {day_count} agreements for {index_date}")
                    except Exception as e:
                        context.log.error(f"Error upserting agreements for {index_date}: {e}")
                        raise RuntimeError(e)
                else:
                    context.log.info(f"  No M&A filings found for {index_date}")

            latest_pulled_to = datetime.combine(index_date, datetime.min.time())
            with engine.begin() as conn:
                _update_pipeline_run_progress(
                    conn, pipeline_runs_table, run_id, latest_pulled_to, total_count
                )

        # DMA corpus flow (commented out - one-time batch processing):
        # # For DMA corpus flow: process all records in one batch (one-time run)
        # # Pass None to skip date filtering and get all rows from the CSV
        # context.log.info("Fetching all DMA corpus filings (one-time run)")
        # filings = fetch_new_filings_dma_corpus(since=None)
        # 
        # total_count = len(filings)
        # 
        # # Commit all filings in a single transaction
        # with engine.begin() as conn:
        #     if filings:
        #         try:
        #             upsert_agreements(filings, conn)
        #             context.log.info(f"Upserted {total_count} agreements from DMA corpus")
        #         except Exception as e:
        #             context.log.error(f"Error upserting agreements: {e}")
        #             raise RuntimeError(e)
        #     else:
        #         context.log.info("No M&A filings found in DMA corpus")
        #     
        #     # Update last_pulled_to to now (since this is a one-time run)
        #     _update_last_pulled_to(conn, run_id, now)
        #     
        #     # Update rows_inserted count
        #     _ = conn.execute(
        #         text(
        #             """
        #             UPDATE pdx.pipeline_runs
        #             SET rows_inserted = :count
        #             WHERE run_id = :run_id
        #             """
        #         ),
        #         {"run_id": run_id, "count": total_count},
        #     )

        context.log.info(f"Staging complete: {total_count} total filings across {days_to_fetch} days")
        run_post_asset_refresh(context, db, pipeline_config)

        # Update pipeline_runs at the last possible moment.
        with engine.begin() as conn:
            _ = conn.execute(
                text(
                    f"""
                    UPDATE {pipeline_runs_table}
                    SET last_pulled_to = :pulled_to,
                        rows_inserted = :count,
                        status = 'SUCCEEDED'
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id, "pulled_to": latest_pulled_to, "count": total_count},
            )
        return total_count
    except Exception:
        with engine.begin() as conn:
            _ = conn.execute(
                text(
                    f"""
                    UPDATE {pipeline_runs_table}
                    SET status = 'FAILED'
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id},
            )
        raise
