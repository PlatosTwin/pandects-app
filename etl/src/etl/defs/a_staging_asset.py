# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
# pyright: reportMissingTypeStubs=false
"""Stage new filings and record pipeline run metadata.

Returns the number of new filings processed.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, cast

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection

from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.a_staging import (
    ExhibitSignature,
    FilingMetadata,
    SecDailyIndexUnavailable,
    fetch_exhibit_signature,
    fetch_new_filings_sec_index,
)
from etl.models.exhibit_classifier.exhibit_classifier import ExhibitClassifier
from etl.utils.db_utils import upsert_agreements as _upsert_agreements
from etl.utils.post_asset_refresh import run_post_asset_refresh

UpsertAgreements = Callable[[Sequence[FilingMetadata], str, Connection], None]
upsert_agreements = cast(UpsertAgreements, _upsert_agreements)
_CROSS_DAY_DEDUPE_LOOKBACK_DAYS = 30
_CROSS_DAY_DEDUPE_MINHASH_THRESHOLD = 0.85


@dataclass(frozen=True)
class _PersistedAgreement:
    agreement_uuid: str
    url: str
    filing_date: date
    ingested_date: datetime | None
    secondary_filing_url: str | None


@dataclass(frozen=True)
class _DuplicateResolution:
    survivor: _PersistedAgreement
    loser: _PersistedAgreement


def _normalize_datetime(value: object | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, date):
        return datetime.combine(value, datetime.min.time())
    raise TypeError(f"Unexpected datetime value: {value!r}")


def _normalize_date(value: object | None) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    raise TypeError(f"Unexpected date value: {value!r}")


def _agreement_sort_key(agreement: _PersistedAgreement) -> tuple[date, datetime, str]:
    ingested_date = agreement.ingested_date or datetime.max
    return agreement.filing_date, ingested_date, agreement.url


def _pick_duplicate_survivor(
    left: _PersistedAgreement,
    right: _PersistedAgreement,
) -> tuple[_PersistedAgreement, _PersistedAgreement]:
    if _agreement_sort_key(left) <= _agreement_sort_key(right):
        return left, right
    return right, left


def _select_agreements_by_urls(
    conn: Connection,
    agreements_table: str,
    urls: Iterable[str],
) -> list[_PersistedAgreement]:
    target_urls = tuple(sorted({url for url in urls if url}))
    if not target_urls:
        return []
    sql = text(
        f"""
        SELECT agreement_uuid, url, filing_date, ingested_date, secondary_filing_url
        FROM {agreements_table}
        WHERE url IN :urls
          AND filing_date IS NOT NULL
        """
    ).bindparams(bindparam("urls", expanding=True))
    rows = conn.execute(sql, {"urls": target_urls}).mappings().all()
    return [
        _PersistedAgreement(
            agreement_uuid=str(row["agreement_uuid"]),
            url=str(row["url"]),
            filing_date=_normalize_date(row["filing_date"]),
            ingested_date=_normalize_datetime(row["ingested_date"]),
            secondary_filing_url=cast(str | None, row["secondary_filing_url"]),
        )
        for row in rows
    ]


def _select_existing_agreement_uuids(
    conn: Connection,
    agreements_table: str,
    agreement_uuids: Iterable[str],
) -> set[str]:
    target_uuids = tuple(sorted({agreement_uuid for agreement_uuid in agreement_uuids if agreement_uuid}))
    if not target_uuids:
        return set()
    sql = text(
        f"""
        SELECT agreement_uuid
        FROM {agreements_table}
        WHERE agreement_uuid IN :agreement_uuids
        """
    ).bindparams(bindparam("agreement_uuids", expanding=True))
    rows = conn.execute(sql, {"agreement_uuids": target_uuids}).scalars().all()
    return {str(row) for row in rows}


def _select_reconciliation_candidates(
    conn: Connection,
    agreements_table: str,
    *,
    window_start: date,
    window_end: date,
) -> list[_PersistedAgreement]:
    sql = text(
        f"""
        SELECT agreement_uuid, url, filing_date, ingested_date, secondary_filing_url
        FROM {agreements_table}
        WHERE filing_date BETWEEN :window_start AND :window_end
          AND source <=> 'edgar'
          AND filing_date IS NOT NULL
        """
    )
    rows = conn.execute(
        sql,
        {"window_start": window_start, "window_end": window_end},
    ).mappings().all()
    return [
        _PersistedAgreement(
            agreement_uuid=str(row["agreement_uuid"]),
            url=str(row["url"]),
            filing_date=_normalize_date(row["filing_date"]),
            ingested_date=_normalize_datetime(row["ingested_date"]),
            secondary_filing_url=cast(str | None, row["secondary_filing_url"]),
        )
        for row in rows
    ]


def _build_signature_cache(
    context: AssetExecutionContext,
    urls: Iterable[str],
) -> dict[str, ExhibitSignature | None]:
    signatures: dict[str, ExhibitSignature | None] = {}
    for url in sorted({url for url in urls if url}):
        try:
            signatures[url] = fetch_exhibit_signature(url)
        except Exception as exc:
            context.log.info(f"Cross-day de-dupe: failed to fetch signature for {url}: {exc}")
            signatures[url] = None
    return signatures


def _build_duplicate_resolutions(
    ingested_agreements: Sequence[_PersistedAgreement],
    candidate_agreements: Sequence[_PersistedAgreement],
    signatures_by_url: Mapping[str, ExhibitSignature | None],
) -> list[_DuplicateResolution]:
    candidate_by_uuid = {agreement.agreement_uuid: agreement for agreement in candidate_agreements}
    parent: dict[str, str] = {agreement.agreement_uuid: agreement.agreement_uuid for agreement in candidate_agreements}

    def find(agreement_uuid: str) -> str:
        current = parent[agreement_uuid]
        if current != agreement_uuid:
            parent[agreement_uuid] = find(current)
        return parent[agreement_uuid]

    def union(left_uuid: str, right_uuid: str) -> None:
        left_root = find(left_uuid)
        right_root = find(right_uuid)
        if left_root == right_root:
            return
        left_agreement = candidate_by_uuid[left_root]
        right_agreement = candidate_by_uuid[right_root]
        survivor, loser = _pick_duplicate_survivor(left_agreement, right_agreement)
        parent[loser.agreement_uuid] = survivor.agreement_uuid

    for ingested in ingested_agreements:
        ingested_signature = signatures_by_url.get(ingested.url)
        if ingested_signature is None:
            continue
        for candidate in candidate_agreements:
            if candidate.agreement_uuid == ingested.agreement_uuid:
                continue
            day_delta = abs((candidate.filing_date - ingested.filing_date).days)
            if day_delta > _CROSS_DAY_DEDUPE_LOOKBACK_DAYS:
                continue
            candidate_signature = signatures_by_url.get(candidate.url)
            if candidate_signature is None:
                continue
            fingerprints_match = ingested_signature.content_fingerprint == candidate_signature.content_fingerprint
            if not fingerprints_match:
                similarity = float(ingested_signature.minhash.jaccard(candidate_signature.minhash))
                if similarity < _CROSS_DAY_DEDUPE_MINHASH_THRESHOLD:
                    continue
            union(ingested.agreement_uuid, candidate.agreement_uuid)

    grouped: dict[str, list[_PersistedAgreement]] = {}
    for agreement in candidate_agreements:
        root = find(agreement.agreement_uuid)
        grouped.setdefault(root, []).append(agreement)

    ingested_uuid_set = {agreement.agreement_uuid for agreement in ingested_agreements}
    resolutions: list[_DuplicateResolution] = []
    for group in grouped.values():
        if len(group) < 2:
            continue
        sorted_group = sorted(group, key=_agreement_sort_key)
        survivor = sorted_group[0]
        if not any(agreement.agreement_uuid in ingested_uuid_set for agreement in group):
            continue
        for loser in sorted_group[1:]:
            resolutions.append(_DuplicateResolution(survivor=survivor, loser=loser))
    return resolutions


def _merge_secondary_filing_urls(
    conn: Connection,
    agreements_table: str,
    resolutions: Sequence[_DuplicateResolution],
) -> None:
    updates: dict[str, str] = {}
    for resolution in resolutions:
        if resolution.loser.url == resolution.survivor.url:
            continue
        _ = updates.setdefault(resolution.survivor.agreement_uuid, resolution.loser.url)
    if not updates:
        return
    sql = text(
        f"""
        UPDATE {agreements_table}
        SET secondary_filing_url = COALESCE(secondary_filing_url, :secondary_filing_url)
        WHERE agreement_uuid = :agreement_uuid
        """
    )
    _ = conn.execute(
        sql,
        [
            {"agreement_uuid": agreement_uuid, "secondary_filing_url": secondary_url}
            for agreement_uuid, secondary_url in updates.items()
        ],
    )


def _delete_duplicate_agreements(
    conn: Connection,
    schema: str,
    loser_uuids: Sequence[str],
) -> None:
    target_uuids = tuple(sorted({agreement_uuid for agreement_uuid in loser_uuids if agreement_uuid}))
    if not target_uuids:
        return

    agreement_tables = (
        "xml_status_reasons",
        "latest_sections_search_standard_ids",
        "latest_sections_search",
        "section_text_search",
        "sections",
        "xml",
    )
    page_tables = (
        "ai_repair_processed_spans",
        "ai_repair_rulings",
        "ai_repair_requests",
        "ai_repair_full_pages",
    )

    for table_name in agreement_tables:
        delete_sql = text(
            f"DELETE FROM {schema}.{table_name} WHERE agreement_uuid IN :agreement_uuids"
        ).bindparams(bindparam("agreement_uuids", expanding=True))
        _ = conn.execute(delete_sql, {"agreement_uuids": target_uuids})

    for table_name in page_tables:
        delete_sql = text(
            f"""
            DELETE d
            FROM {schema}.{table_name} d
            JOIN {schema}.pages p
                ON p.page_uuid = d.page_uuid
            WHERE p.agreement_uuid IN :agreement_uuids
            """
        ).bindparams(bindparam("agreement_uuids", expanding=True))
        _ = conn.execute(delete_sql, {"agreement_uuids": target_uuids})

    delete_agreements_sql = text(
        f"DELETE FROM {schema}.agreements WHERE agreement_uuid IN :agreement_uuids"
    ).bindparams(bindparam("agreement_uuids", expanding=True))
    _ = conn.execute(delete_agreements_sql, {"agreement_uuids": target_uuids})


def _reconcile_cross_day_duplicates(
    context: AssetExecutionContext,
    conn: Connection,
    schema: str,
    ingested_urls: Sequence[str],
) -> tuple[int, set[str]]:
    agreements_table = f"{schema}.agreements"
    ingested_agreements = _select_agreements_by_urls(conn, agreements_table, ingested_urls)
    if not ingested_agreements:
        context.log.info("Cross-day de-dupe stage: no ingested agreements to scan.")
        return 0, set()

    filing_dates = [agreement.filing_date for agreement in ingested_agreements]
    window_start = min(filing_dates) - timedelta(days=_CROSS_DAY_DEDUPE_LOOKBACK_DAYS)
    window_end = max(filing_dates) + timedelta(days=_CROSS_DAY_DEDUPE_LOOKBACK_DAYS)
    candidate_agreements = _select_reconciliation_candidates(
        conn,
        agreements_table,
        window_start=window_start,
        window_end=window_end,
    )
    context.log.info(
        "Cross-day de-dupe stage: scanning "
        + f"{len(ingested_agreements)} ingested agreements against {len(candidate_agreements)} "
        + f"candidates in filing-date window {window_start} to {window_end}."
    )
    signatures_by_url = _build_signature_cache(
        context,
        [agreement.url for agreement in candidate_agreements],
    )
    resolutions = _build_duplicate_resolutions(
        ingested_agreements,
        candidate_agreements,
        signatures_by_url,
    )
    if not resolutions:
        context.log.info("Cross-day de-dupe stage: found 0 duplicate agreements.")
        return 0, set()

    _merge_secondary_filing_urls(conn, agreements_table, resolutions)
    deleted_uuids = {
        resolution.loser.agreement_uuid
        for resolution in resolutions
        if resolution.loser.agreement_uuid
    }
    _delete_duplicate_agreements(
        conn,
        schema,
        sorted(deleted_uuids),
    )
    context.log.info(
        "Cross-day de-dupe stage: removed "
        + f"{len(resolutions)} duplicate agreements across "
        + f"{len({resolution.survivor.agreement_uuid for resolution in resolutions})} clusters."
    )
    return len(resolutions), deleted_uuids


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


def _run_staging(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> tuple[int, list[str]]:
    engine = db.get_engine()
    schema = db.database
    agreements_table = f"{schema}.agreements"
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
    ingested_urls: list[str] = []
    inserted_agreement_uuids: set[str] = set()

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
            ingested_urls.extend(filing.url for filing in filings if filing.url)

            # Persist each day independently so retries resume from the last completed date.
            with engine.begin() as conn:
                if filings:
                    try:
                        existing_uuids = _select_existing_agreement_uuids(
                            conn,
                            agreements_table,
                            [filing.agreement_uuid for filing in filings],
                        )
                        inserted_agreement_uuids.update(
                            filing.agreement_uuid
                            for filing in filings
                            if filing.agreement_uuid not in existing_uuids
                        )
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
        with engine.begin() as conn:
            _, deleted_uuids = _reconcile_cross_day_duplicates(context, conn, schema, ingested_urls)
        scoped_agreement_uuids = sorted(inserted_agreement_uuids - deleted_uuids)
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
        return total_count, scoped_agreement_uuids
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


@dg.asset(name="1_staging_asset")
def staging_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig
) -> int:
    """Stage new filings day-by-day with incremental commits."""
    total_count, _ = _run_staging(context, db, pipeline_config)
    return total_count


@dg.asset(name="1-1_regular_ingest_staging_asset")
def regular_ingest_staging_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> list[str]:
    """Stage filings and return only newly inserted, non-deduped agreement UUIDs."""
    _, scoped_agreement_uuids = _run_staging(context, db, pipeline_config)
    context.log.info(
        "regular_ingest_staging_asset: scoped %s newly inserted non-deduped agreements.",
        len(scoped_agreement_uuids),
    )
    return scoped_agreement_uuids
