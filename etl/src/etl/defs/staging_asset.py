from etl.defs.resources import DBResource
from sqlalchemy import text
from etl.domain.staging import fetch_new_filings
from datetime import datetime, timezone
import dagster as dg


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

    # 1) Get last pull timestamp
    with engine.begin() as conn:
        last_run = conn.execute(
            text(
                "SELECT last_pulled_to "
                "FROM pdx.pipeline_runs "
                "ORDER BY run_time DESC LIMIT 1"
            )
        ).scalar_one_or_none() or datetime(1970, 1, 1)

    now = datetime.now(timezone.utc)

    # 2) Fetch new filings
    filings, pulled_to_ts = fetch_new_filings(since=last_run.isoformat())
    count = len(filings)

    # 3-5) Transactional run
    with engine.begin() as conn:
        # 3) STARTED record
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

        # 4) Prepare upsert rows
        rows = []
        for f in filings:
            rows.append(
                {
                    "agreement_uuid": f.agreement_uuid,
                    "url": f.url,
                    "target": f.target,
                    "acquirer": f.acquirer,
                    "filing_date": f.filing_date,
                    "transaction_date": f.transaction_date,
                    "transaction_price": f.transaction_price,
                    "transaction_type": f.transaction_type,
                    "transaction_consideration": f.transaction_consideration,
                    "consideration_type": f.consideration_type,
                    "target_type": f.target_type,
                }
            )

        upsert_sql = text(
            """
            INSERT INTO pdx.agreements (
              agreement_uuid,
              url,
              target,
              acquirer,
              transaction_date,
              transaction_price,
              transaction_type,
              transaction_consideration,
              consideration_type,
              target_type
            ) VALUES (
              :agreement_uuid,
              :url,
              :target,
              :acquirer,
              :transaction_date,
              :transaction_price,
              :transaction_type,
              :transaction_consideration,
              :consideration_type,
              :target_type
            )
            ON DUPLICATE KEY UPDATE
              url                      = VALUES(url),
              target                   = VALUES(target),
              acquirer                 = VALUES(acquirer),
              transaction_date         = VALUES(transaction_date),
              transaction_price        = VALUES(transaction_price),
              transaction_type         = VALUES(transaction_type),
              transaction_consideration = VALUES(transaction_consideration),
              consideration_type       = VALUES(consideration_type),
              target_type              = VALUES(target_type)
        """
        )

        # execute in batches of 250
        for i in range(0, count, 250):
            batch = rows[i : i + 250]
            conn.execute(upsert_sql, batch)

        # 5) Mark SUCCEEDED
        conn.execute(
            text(
                "UPDATE pdx.pipeline_runs "
                "SET status = 'SUCCEEDED' "
                "WHERE run_time = :run_time"
            ),
            {"run_time": now},
        )

    # 6) Return count for downstream dependencies
    return count
