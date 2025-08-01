from etl.defs.resources import DBResource
from sqlalchemy import text
from etl.domain.staging import fetch_new_filings
from datetime import datetime, timezone
import dagster as dg
from etl.utils.db_utils import upsert_agreements


@dg.asset
def staging_asset(db: DBResource):
    """
    Tracks pipeline runs and upserts full agreements in batches of 250.

    1. Reads last pull timestamp.
    2. Fetches new filings.
    3. Inserts a STARTED record.
    4. Upserts into pdx.agreements (all columns) in batches of 250.
    5. Marks the run SUCCEEDED.
    6. Returns the number of new filings.
    """
    engine = db.get_engine()

    # 1. Get last pull timestamp
    with engine.begin() as conn:
        last_run = conn.execute(
            text(
                "SELECT last_pulled_to "
                "FROM pdx.pipeline_runs "
                "ORDER BY run_time DESC LIMIT 1"
            )
        ).scalar_one_or_none() or datetime(1970, 1, 1)

    now = datetime.now(timezone.utc)

    # 2. Fetch new filings
    filings = fetch_new_filings(since=last_run.isoformat())
    count = len(filings)

    # Use now as pulled_to_ts if no filings, else use now or last filing date
    pulled_to_ts = max([f.transaction_date for f in filings])

    # 3-5 Transactional run
    with engine.begin() as conn:
        # 3. STARTED record
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

        # 4. Prepare upsert rows
        try:
            upsert_agreements(filings, conn)
        except Exception as e:
            print(f"Error upserting pages: {e}")
            raise RuntimeError(e)
        
        # 5. Mark (if) SUCCEEDED
        conn.execute(
            text(
                "UPDATE pdx.pipeline_runs "
                "SET status = 'SUCCEEDED' "
                "WHERE run_time = :run_time"
            ),
            {"run_time": now},
        )

    # 6. Return count for downstream dependencies
    return count
