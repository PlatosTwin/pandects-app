# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
# pyright: reportMissingTypeStubs=false
"""Stage new filings and record pipeline run metadata.

Respects CLEANUP vs FROM_SCRATCH modes via `PipelineConfig`.
Returns the number of new filings processed.
"""

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
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
    # To switch back to DMA corpus flow, uncomment the line below and comment out fetch_new_filings_sec_index:
    # fetch_new_filings_dma_corpus,
)
from etl.models.exhibit_classifier.exhibit_classifier import ExhibitClassifier
from etl.utils.db_utils import upsert_agreements as _upsert_agreements
from etl.utils.run_config import is_cleanup_mode

UpsertAgreements = Callable[[Sequence[FilingMetadata], Connection], None]
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


def _update_last_pulled_to(conn: Connection, run_id: int, pulled_to: datetime) -> None:
    """Update the last_pulled_to timestamp for the current run."""
    _ = conn.execute(
        text(
            """
            UPDATE pdx.pipeline_runs
            SET last_pulled_to = :pulled_to
            WHERE run_id = :run_id
            """
        ),
        {"run_id": run_id, "pulled_to": pulled_to},
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
        return 0

    engine = db.get_engine()
    context.log.info("Running staging in FROM_SCRATCH mode")

    # Get last pull timestamp
    with engine.begin() as conn:
        last_run: datetime | None = conn.execute(
            text(
                """
                SELECT last_pulled_to
                FROM pdx.pipeline_runs
                ORDER BY run_time DESC LIMIT 1
                """
            )
        ).scalar_one_or_none()
        if last_run is None:
            last_run = datetime(1970, 1, 1)

    # last_run = datetime(2021, 1, 1)  # TODO: Remove this override once production-ready
    now = datetime.now(timezone.utc)

    # Load exhibit classifier once for all days
    # To switch back to DMA corpus flow, see comments in fetch_new_filings_sec_index call below
    classifier = ExhibitClassifier.load(_get_exhibit_classifier_path())
    staging_context = _DagsterContextAdapter(context)

    # Calculate total days to process
    days_to_fetch = pipeline_config.staging_days_to_fetch
    
    # Insert STARTED record for this run and get the run_id
    # Use date-level granularity (midnight) for last_pulled_from/to
    last_run_date = datetime.combine(last_run.date(), datetime.min.time())
    with engine.begin() as conn:
        result = conn.execute(
            text(
                """
                INSERT INTO pdx.pipeline_runs
                (run_time, last_pulled_from, last_pulled_to, status, rows_inserted)
                VALUES (:run_time, :from_ts, :to_ts, 'STARTED', 0)
                """
            ),
            {
                "run_time": now,
                "from_ts": last_run_date,
                "to_ts": last_run_date,  # Will be updated as we process each day
            },
        )
        run_id = result.lastrowid

    total_count = 0
    base_date = last_run.date()
    
    # Process day by day
    # Note: get_sec_index_urls fetches index for (start_date + day_offset), so we pass
    # base_date as start_date and let the function fetch the correct day's index.
    for day_offset in range(days_to_fetch):
        # The actual index date being fetched (start_date + 1 day due to how get_sec_index_urls works)
        index_date = base_date + timedelta(days=day_offset + 1)
        start_date_for_fetch = (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")
        context.log.info(f"Processing day {day_offset + 1}/{days_to_fetch}: fetching index for {index_date}")
        
        # Fetch and classify filings for this single day
        filings = fetch_new_filings_sec_index(
            exhibit_classifier=classifier,
            context=staging_context,
            start_date=start_date_for_fetch,
            pipeline_config=pipeline_config,
            days_override=1,  # Process exactly one day
        )
        # To switch to DMA corpus: filings = fetch_new_filings_dma_corpus(since=start_date_for_fetch)
        
        day_count = len(filings)
        total_count += day_count
        
        # Commit this day's filings and update progress
        # Each day is its own transaction for crash recovery
        with engine.begin() as conn:
            if filings:
                try:
                    upsert_agreements(filings, conn)
                    context.log.info(f"  Upserted {day_count} agreements for {index_date}")
                except Exception as e:
                    context.log.error(f"Error upserting agreements for {index_date}: {e}")
                    raise RuntimeError(e)
            else:
                context.log.info(f"  No M&A filings found for {index_date}")
            
            # Update last_pulled_to to the date we actually processed (midnight, date-level granularity)
            pulled_to_date = datetime.combine(index_date, datetime.min.time())
            _update_last_pulled_to(conn, run_id, pulled_to_date)
            
            # Update rows_inserted count
            _ = conn.execute(
                text(
                    """
                    UPDATE pdx.pipeline_runs
                    SET rows_inserted = :count
                    WHERE run_id = :run_id
                    """
                ),
                {"run_id": run_id, "count": total_count},
            )

    # Mark run as SUCCEEDED after all days complete
    with engine.begin() as conn:
        _ = conn.execute(
            text(
                """
                UPDATE pdx.pipeline_runs
                SET status = 'SUCCEEDED'
                WHERE run_id = :run_id
                """
            ),
            {"run_id": run_id},
        )

    context.log.info(f"Staging complete: {total_count} total filings across {days_to_fetch} days")
    return total_count
