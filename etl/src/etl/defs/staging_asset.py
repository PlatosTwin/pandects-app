"""Staging asset for fetching and upserting new filings."""

from etl.defs.resources import DBResource, PipelineConfig
from sqlalchemy import text
from etl.domain.staging import fetch_new_filings
from datetime import datetime, timezone
import dagster as dg
from etl.utils.db_utils import upsert_agreements
from typing import Optional


@dg.asset
def staging_asset(
    context: dg.AssetExecutionContext, 
    db: DBResource, 
    pipeline_config: PipelineConfig
) -> int:
    """Track pipeline runs and upsert full agreements in batches of 250.
    
    In cleanup mode, skips fetching new filings and only processes existing 
    unprocessed agreements.
    
    Args:
        context: Dagster execution context.
        db: Database resource for connection.
        pipeline_config: Pipeline configuration for mode.
        
    Returns:
        Number of new filings processed.
    """
    engine = db.get_engine()
    is_cleanup = pipeline_config.is_cleanup_mode()

    # Override mode from job context if available
    if hasattr(context, "job_def") and hasattr(context.job_def, "config"):
        job_config = context.job_def.config
        if hasattr(job_config, "mode"):
            is_cleanup = job_config.mode.value == "cleanup"

    if is_cleanup:
        context.log.info("CLEANUP mode. Skipping staging step.")
        return 0

    context.log.info("Running staging in FROM_SCRATCH mode")

    # Get last pull timestamp
    with engine.begin() as conn:
        last_run: Optional[datetime] = conn.execute(
            text(
                "SELECT last_pulled_to "
                "FROM pdx.pipeline_runs "
                "ORDER BY run_time DESC LIMIT 1"
            )
        ).scalar_one_or_none() or datetime(1970, 1, 1)

    now = datetime.now(timezone.utc)

    # Fetch new filings
    filings = fetch_new_filings(since=last_run.isoformat())
    count = len(filings)

    # Use now as pulled_to_ts if no filings, else use max filing date
    pulled_to_ts = max([f.filing_date for f in filings]) if filings else now
    context.log.info(f"FROM_SCRATCH mode: Fetched {count} new filings")

    # Transactional run
    with engine.begin() as conn:
        # Insert STARTED record
        conn.execute(
            text(
                "INSERT INTO pdx.pipeline_runs "
                "(run_time, last_pulled_from, last_pulled_to, status, rows_inserted) "
                "VALUES (:run_time, :from_ts, :to_ts, 'STARTED', :count)"
            ),
            {
                "run_time": now,
                "from_ts": last_run,
                "to_ts": pulled_to_ts,
                "count": count,
            },
        )

        # Upsert agreements
        try:
            upsert_agreements(filings, conn)
            context.log.info(f"Successfully upserted {len(filings)} agreements")
        except Exception as e:
            context.log.error(f"Error upserting agreements: {e}")
            raise RuntimeError(e)

        # Mark as SUCCEEDED
        conn.execute(
            text(
                "UPDATE pdx.pipeline_runs "
                "SET status = 'SUCCEEDED' "
                "WHERE run_time = :run_time"
            ),
            {"run_time": now},
        )

    return count
