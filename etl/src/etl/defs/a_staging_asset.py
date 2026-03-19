# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
# pyright: reportMissingTypeStubs=false
"""Stage new filings and record pipeline run metadata.

Returns the number of new filings processed.
"""

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, cast

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import text
from sqlalchemy.engine import Connection

from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.a_staging import (
    FilingMetadata,
    SecDailyIndexUnavailable,
    fetch_new_filings_sec_index,
)
from etl.models.exhibit_classifier.exhibit_classifier import ExhibitClassifier
from etl.utils.db_utils import upsert_agreements as _upsert_agreements
from etl.utils.post_asset_refresh import run_post_asset_refresh

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


def _latest_sec_index_date_available(today: date | None = None) -> date:
    """Return the most recent date whose daily SEC index we should expect."""
    reference_date = today if today is not None else date.today()
    return reference_date - timedelta(days=1)


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

    Args:
        context: Dagster execution context.
        db: Database resource for connection.
        pipeline_config: Pipeline configuration.

    Returns:
        Total number of new filings processed across all days.
    """
    engine = db.get_engine()
    schema = db.database
    pipeline_runs_table = f"{schema}.pipeline_runs"
    context.log.info("Running staging")

    # Resume from the latest recorded pull boundary even if the previous run failed later.
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

    classifier = ExhibitClassifier.load(_get_exhibit_classifier_path())
    staging_context = _DagsterContextAdapter(context)

    days_to_fetch = pipeline_config.staging_days_to_fetch

    # Record the run before processing so day-by-day progress survives partial failures.
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
                "to_ts": last_run_date,
            },
        )
        run_id = result.lastrowid

    total_count = 0
    base_date = last_run.date()
    latest_pulled_to = last_run_date
    processed_days = 0
    latest_expected_index_date = _latest_sec_index_date_available()

    try:
        for day_offset in range(days_to_fetch):
            # `fetch_new_filings_sec_index(..., days_override=1)` reads the next index day after
            # `start_date`, so `start_date_for_fetch` intentionally lags the target index date.
            index_date = base_date + timedelta(days=day_offset + 1)
            start_date_for_fetch = (base_date + timedelta(days=day_offset)).strftime("%Y-%m-%d")
            context.log.info(
                f"Processing day {day_offset + 1}/{days_to_fetch}: fetching index for {index_date}"
            )

            try:
                filings = fetch_new_filings_sec_index(
                    exhibit_classifier=classifier,
                    context=staging_context,
                    start_date=start_date_for_fetch,
                    pipeline_config=pipeline_config,
                    days_override=1,
                )
            except SecDailyIndexUnavailable:
                if index_date > latest_expected_index_date:
                    context.log.info(
                        (
                            "Stopping staging early: SEC daily index for {index_date} is not "
                            + "available, and the availability cutoff is "
                            + "{latest_expected_index_date}."
                        ).format(
                            index_date=index_date,
                            latest_expected_index_date=latest_expected_index_date,
                        )
                    )
                    break
                context.log.info(
                    (
                        "SEC daily index for {index_date} is missing, but it is on or before "
                        + "the availability cutoff {latest_expected_index_date}; treating as an "
                        + "empty day."
                    ).format(
                        index_date=index_date,
                        latest_expected_index_date=latest_expected_index_date,
                    )
                )
                filings = []
            day_count = len(filings)
            total_count += day_count

            # Persist each day independently so retries resume from the last completed date.
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
            processed_days += 1

        context.log.info(
            f"Staging complete: {total_count} total filings across {processed_days} processed days"
        )
        run_post_asset_refresh(context, db, pipeline_config)

        # Only mark the run successful after downstream refresh work completes.
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
