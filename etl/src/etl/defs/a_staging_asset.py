# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
# pyright: reportMissingTypeStubs=false
"""Stage new filings and record pipeline run metadata.

Returns the number of new filings processed.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, cast
from zoneinfo import ZoneInfo

import dagster as dg
from dagster import AssetExecutionContext
from datasketch import MinHashLSH
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection, Engine

from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.a_staging import (
    FilingMetadata,
    SecDailyIndexUnavailable,
    StagedFiling,
    fetch_document_signature,
    fetch_new_filings_sec_index,
)
from etl.domain.dedupe_signatures import (
    LSH_INDEX_THRESHOLD,
    MINHASH_NUM_PERM,
    DuplicateAction,
    SurvivorCandidate,
    decide_duplicate,
    normalize_party_name,
    party_metadata_conflict,
    pick_survivor,
    same_edgar_accession,
)
from etl.models.exhibit_classifier.exhibit_classifier import ExhibitClassifier
from etl.utils.agreement_signature_store import (
    SIGNATURE_TABLE_NAMES,
    StoredSignature,
    clear_pending,
    delete_signatures,
    insert_review,
    load_all_signatures,
    load_pending,
    mark_reconciled,
    record_pending,
    record_pending_if_absent,
    select_unreconciled_uuids,
    upsert_signatures,
)
from etl.utils.logical_job_runs import (
    mark_logical_run_stage_completed,
    should_skip_managed_stage,
    start_or_resume_logical_run,
)
from etl.utils.db_utils import upsert_agreements as _upsert_agreements
from etl.utils.post_asset_refresh import run_post_asset_refresh
from etl.utils.schema_guards import assert_tables_exist

UpsertAgreements = Callable[[Sequence[FilingMetadata], str, Connection], None]
upsert_agreements = cast(UpsertAgreements, _upsert_agreements)
_PENDING_RETRY_LIMIT = 25
_EASTERN_TZ = ZoneInfo("America/New_York")


def _today_eastern() -> date:
    return datetime.now(_EASTERN_TZ).date()


def _parse_staging_target_date(raw: str) -> date:
    value = raw.strip()
    if not value:
        raise ValueError(
            "pipeline_config.staging_target_date is required (yyyy-mm-dd)."
        )
    try:
        target_date = datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise ValueError(
            f"pipeline_config.staging_target_date must be in yyyy-mm-dd format, got {raw!r}."
        ) from exc
    today_eastern = _today_eastern()
    if target_date >= today_eastern:
        raise ValueError(
            f"pipeline_config.staging_target_date ({target_date}) must be before "
            f"today's date in Eastern time ({today_eastern})."
        )
    return target_date


@dataclass(frozen=True)
class _CorpusAgreement:
    agreement_uuid: str
    url: str
    filing_date: date
    ingested_date: datetime | None
    target: str | None
    acquirer: str | None


@dataclass(frozen=True)
class _CorpusItem:
    agreement: _CorpusAgreement
    stored: StoredSignature


@dataclass(frozen=True)
class _DuplicateResolution:
    survivor_uuid: str
    survivor_url: str
    loser_uuid: str
    loser_url: str


@dataclass(frozen=True)
class _ReviewFlag:
    new_agreement_uuid: str
    new_url: str
    existing_agreement_uuid: str
    existing_url: str
    jaccard: float
    containment: float
    fingerprint_match: bool
    reason: str


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


def _select_edgar_agreements(
    conn: Connection,
    agreements_table: str,
) -> list[_CorpusAgreement]:
    sql = text(
        f"""
        SELECT agreement_uuid, url, filing_date, ingested_date, target, acquirer
        FROM {agreements_table}
        WHERE source <=> 'edgar'
          AND filing_date IS NOT NULL
        """
    )
    rows = conn.execute(sql).mappings().all()
    return [
        _CorpusAgreement(
            agreement_uuid=str(row["agreement_uuid"]),
            url=str(row["url"]),
            filing_date=_normalize_date(row["filing_date"]),
            ingested_date=_normalize_datetime(row["ingested_date"]),
            target=cast(str | None, row["target"]),
            acquirer=cast(str | None, row["acquirer"]),
        )
        for row in rows
    ]


def _survivor_input(item: _CorpusItem) -> SurvivorCandidate:
    return SurvivorCandidate(
        agreement_uuid=item.agreement.agreement_uuid,
        url=item.agreement.url,
        filing_date=item.agreement.filing_date,
        ingested_date=item.agreement.ingested_date,
        page_count=item.stored.signature.page_count,
    )


def _normalized_party_pair(agreement: _CorpusAgreement) -> tuple[str, str] | None:
    """Normalized (target, acquirer) pair, or None when either side is unknown."""
    if not agreement.target or not agreement.acquirer:
        return None
    target = normalize_party_name(agreement.target)
    acquirer = normalize_party_name(agreement.acquirer)
    if not target or not acquirer:
        return None
    return target, acquirer


def _plan_corpus_dedupe(
    new_agreement_uuids: set[str],
    items: Sequence[_CorpusItem],
) -> tuple[list[_DuplicateResolution], list[_ReviewFlag]]:
    """Plan duplicate resolutions for new agreements against the whole corpus.

    Pure planning over in-memory items: LSH surfaces content candidates, exact
    fingerprints and party-identity matches force additional comparisons, and
    decide_duplicate gates every merge. Party identity alone never deletes.
    """
    items_by_uuid = {item.agreement.agreement_uuid: item for item in items}

    lsh = MinHashLSH(threshold=LSH_INDEX_THRESHOLD, num_perm=MINHASH_NUM_PERM)
    fingerprint_index: dict[str, list[str]] = {}
    party_index: dict[tuple[str, str], list[str]] = {}
    for item in items:
        uuid = item.agreement.agreement_uuid
        lsh.insert(uuid, item.stored.signature.minhash)
        fingerprint_index.setdefault(item.stored.signature.content_fingerprint, []).append(uuid)
        party_pair = _normalized_party_pair(item.agreement)
        if party_pair is not None:
            party_index.setdefault(party_pair, []).append(uuid)

    parent: dict[str, str] = {uuid: uuid for uuid in items_by_uuid}

    def find(uuid: str) -> str:
        current = parent[uuid]
        if current != uuid:
            parent[uuid] = find(current)
        return parent[uuid]

    def union(left_uuid: str, right_uuid: str) -> None:
        left_root = find(left_uuid)
        right_root = find(right_uuid)
        if left_root == right_root:
            return
        survivor, loser = pick_survivor(
            _survivor_input(items_by_uuid[left_root]),
            _survivor_input(items_by_uuid[right_root]),
        )
        parent[loser.agreement_uuid] = survivor.agreement_uuid

    reviews: list[_ReviewFlag] = []
    seen_pairs: set[frozenset[str]] = set()

    for new_uuid in sorted(new_agreement_uuids):
        new_item = items_by_uuid.get(new_uuid)
        if new_item is None:
            continue
        content_candidates = {
            str(key) for key in lsh.query(new_item.stored.signature.minhash)
        }
        content_candidates.update(
            fingerprint_index.get(new_item.stored.signature.content_fingerprint, [])
        )
        party_candidates: set[str] = set()
        party_pair = _normalized_party_pair(new_item.agreement)
        if party_pair is not None:
            # Party-identity trigger: force a content comparison regardless of
            # date distance. NEVER sufficient to auto-dedupe on its own.
            party_candidates.update(party_index.get(party_pair, []))
        for other_uuid in sorted(content_candidates | party_candidates):
            if other_uuid == new_uuid:
                continue
            pair = frozenset({new_uuid, other_uuid})
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            other_item = items_by_uuid[other_uuid]
            new_key = (
                new_item.agreement.filing_date,
                new_item.agreement.ingested_date or datetime.min,
            )
            other_key = (
                other_item.agreement.filing_date,
                other_item.agreement.ingested_date or datetime.min,
            )
            if new_key >= other_key:
                newer_item, older_item = new_item, other_item
            else:
                newer_item, older_item = other_item, new_item
            decision = decide_duplicate(
                newer_item.stored.signature,
                newer_item.stored.cover,
                older_item.stored.signature,
                older_item.stored.cover,
                party_match=other_uuid in party_candidates,
                party_conflict=party_metadata_conflict(
                    new_item.agreement.target,
                    new_item.agreement.acquirer,
                    other_item.agreement.target,
                    other_item.agreement.acquirer,
                ),
                same_accession=same_edgar_accession(
                    new_item.agreement.url, other_item.agreement.url
                ),
            )
            if decision.action is DuplicateAction.AUTO_DEDUPE:
                union(new_uuid, other_uuid)
            elif decision.action is DuplicateAction.REVIEW:
                reviews.append(
                    _ReviewFlag(
                        new_agreement_uuid=new_uuid,
                        new_url=new_item.agreement.url,
                        existing_agreement_uuid=other_uuid,
                        existing_url=other_item.agreement.url,
                        jaccard=decision.scores.jaccard,
                        containment=decision.scores.containment,
                        fingerprint_match=decision.fingerprint_match,
                        reason=decision.reason,
                    )
                )

    grouped: dict[str, list[str]] = {}
    for uuid in items_by_uuid:
        grouped.setdefault(find(uuid), []).append(uuid)

    resolutions: list[_DuplicateResolution] = []
    for root, group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        survivor_item = items_by_uuid[root]
        for loser_uuid in sorted(uuid for uuid in group if uuid != root):
            resolutions.append(
                _DuplicateResolution(
                    survivor_uuid=survivor_item.agreement.agreement_uuid,
                    survivor_url=survivor_item.agreement.url,
                    loser_uuid=loser_uuid,
                    loser_url=items_by_uuid[loser_uuid].agreement.url,
                )
            )
    return resolutions, reviews


def _merge_secondary_filing_urls(
    conn: Connection,
    agreements_table: str,
    resolutions: Sequence[_DuplicateResolution],
) -> None:
    updates: dict[str, str] = {}
    for resolution in resolutions:
        if resolution.loser_url == resolution.survivor_url:
            continue
        _ = updates.setdefault(resolution.survivor_uuid, resolution.loser_url)
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


# Aligned with the manual-dedupe table set (dedupe_bak_20260805_* backups):
# every live table keyed on agreement_uuid or (via pages) page_uuid.
_AGREEMENT_KEYED_DELETE_TABLES = (
    "xml_status_reasons",
    "latest_sections_search_standard_ids",
    "latest_sections_search",
    "section_text_search",
    "clauses",
    "sections",
    "xml",
    "agreement_counsel",
    "agreement_index_summary",
    "ai_repair_source_verdicts",
    "transaction_charts",
    "tx_metadata_web_failures",
    "pipeline_job_run_agreements",
)
_PAGE_KEYED_DELETE_TABLES = (
    "ai_repair_processed_spans",
    "ai_repair_rulings",
    "ai_repair_requests",
    "ai_repair_full_pages",
    "tagged_outputs",
    "ner_training_data",
    "ner_training_realign_tracking",
    "preprocess_backfill_tracking",
)

# Curated deal metadata carried from a deleted loser into survivor NULLs.
# Pipeline-state flags (status, verified, gated, ...) are deliberately absent.
_CARRY_OVER_METADATA_COLUMNS = (
    "target",
    "acquirer",
    "announce_date",
    "close_date",
    "transaction_price_total",
    "transaction_price_stock",
    "transaction_price_cash",
    "transaction_price_assets",
    "transaction_consideration",
    "target_type",
    "acquirer_type",
    "target_industry",
    "acquirer_industry",
    "deal_status",
    "attitude",
    "deal_type",
    "purpose",
    "target_pe",
    "acquirer_pe",
    "target_counsel",
    "acquirer_counsel",
    "metadata_sources",
)


def _carry_over_loser_metadata(
    conn: Connection,
    schema: str,
    resolutions: Sequence[_DuplicateResolution],
) -> None:
    """COALESCE-copy each loser's curated metadata into survivor NULLs."""
    if not resolutions:
        return
    set_clause = ", ".join(
        f"s.{column} = COALESCE(s.{column}, l.{column})"
        for column in _CARRY_OVER_METADATA_COLUMNS
    )
    sql = text(
        f"""
        UPDATE {schema}.agreements s
        JOIN {schema}.agreements l
            ON l.agreement_uuid = :loser_uuid
        SET {set_clause}
        WHERE s.agreement_uuid = :survivor_uuid
        """
    )
    for resolution in resolutions:
        _ = conn.execute(
            sql,
            {
                "survivor_uuid": resolution.survivor_uuid,
                "loser_uuid": resolution.loser_uuid,
            },
        )


def _delete_duplicate_agreements(
    conn: Connection,
    schema: str,
    loser_uuids: Sequence[str],
) -> None:
    target_uuids = tuple(sorted({agreement_uuid for agreement_uuid in loser_uuids if agreement_uuid}))
    if not target_uuids:
        return

    # Page-keyed tables first (they join through pages), then agreement-keyed
    # tables, then pages, then the agreements row last.
    for table_name in _PAGE_KEYED_DELETE_TABLES:
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

    for table_name in _AGREEMENT_KEYED_DELETE_TABLES:
        delete_sql = text(
            f"DELETE FROM {schema}.{table_name} WHERE agreement_uuid IN :agreement_uuids"
        ).bindparams(bindparam("agreement_uuids", expanding=True))
        _ = conn.execute(delete_sql, {"agreement_uuids": target_uuids})

    delete_pages_sql = text(
        f"DELETE FROM {schema}.pages WHERE agreement_uuid IN :agreement_uuids"
    ).bindparams(bindparam("agreement_uuids", expanding=True))
    _ = conn.execute(delete_pages_sql, {"agreement_uuids": target_uuids})

    delete_agreements_sql = text(
        f"DELETE FROM {schema}.agreements WHERE agreement_uuid IN :agreement_uuids"
    ).bindparams(bindparam("agreement_uuids", expanding=True))
    _ = conn.execute(delete_agreements_sql, {"agreement_uuids": target_uuids})


def _retry_pending_signatures(
    context: AssetExecutionContext,
    engine: Engine,
    schema: str,
) -> None:
    """Retry signature computation for candidates recorded in pending state.

    EDGAR fetches happen OUTSIDE any DB transaction; each outcome is written in
    its own short transaction. Recovered signatures land unreconciled, so the
    reconciliation pass that follows scans them in the same run.
    """
    with engine.begin() as conn:
        pending_entries = load_pending(conn, schema, limit=_PENDING_RETRY_LIMIT)
        if not pending_entries:
            return
        agreements_table = f"{schema}.agreements"
        existing_uuids = _select_existing_agreement_uuids(
            conn, agreements_table, [entry.agreement_uuid for entry in pending_entries]
        )

    for entry in pending_entries:
        if entry.agreement_uuid not in existing_uuids:
            with engine.begin() as conn:
                clear_pending(conn, schema, [entry.agreement_uuid])
            continue
        try:
            result = fetch_document_signature(entry.url)
        except Exception as exc:
            context.log.warning(
                f"Dedupe guard: pending signature retry failed for {entry.url}: {exc}"
            )
            with engine.begin() as conn:
                record_pending(
                    conn,
                    schema,
                    agreement_uuid=entry.agreement_uuid,
                    url=entry.url,
                    reason=entry.reason,
                    error=str(exc),
                )
            continue
        with engine.begin() as conn:
            if result is None:
                context.log.warning(
                    f"Dedupe guard: unsupported file type for {entry.url}; "
                    "cannot compute signature, clearing pending entry."
                )
                clear_pending(conn, schema, [entry.agreement_uuid])
                continue
            signature, cover = result
            upsert_signatures(
                conn,
                schema,
                [
                    StoredSignature(
                        agreement_uuid=entry.agreement_uuid,
                        url=entry.url,
                        signature=signature,
                        cover=cover,
                    )
                ],
            )
            clear_pending(conn, schema, [entry.agreement_uuid])
        context.log.info(
            f"Dedupe guard: recovered pending signature for {entry.url}."
        )


def _reconcile_corpus_duplicates(
    context: AssetExecutionContext,
    conn: Connection,
    schema: str,
) -> tuple[int, set[str]]:
    """Corpus-wide duplicate reconciliation with a persisted scan scope.

    Replaces the old +/-30-day filing-date window: each pass scans every
    signature not yet marked `reconciled_at` against ALL persisted signatures
    via an LSH index built at run time, so re-filings months later
    (10-Q/10-K/S-4 exhibit copies) are caught. The scope is persisted on
    agreement_signatures rather than passed in memory, so pending-retry
    recoveries get scanned and a crash between ingest commits and
    reconciliation cannot orphan a run's agreements from scanning.
    """
    agreements_table = f"{schema}.agreements"

    stored_signatures = load_all_signatures(conn, schema)
    stored_by_uuid = {stored.agreement_uuid: stored for stored in stored_signatures}
    corpus_agreements = _select_edgar_agreements(conn, agreements_table)

    items: list[_CorpusItem] = []
    missing: list[_CorpusAgreement] = []
    for agreement in corpus_agreements:
        stored = stored_by_uuid.get(agreement.agreement_uuid)
        if stored is None:
            missing.append(agreement)
        else:
            items.append(_CorpusItem(agreement=agreement, stored=stored))

    if missing:
        context.log.warning(
            f"Dedupe guard: {len(missing)} edgar agreements have no persisted "
            "signature; recording them as pending. Run "
            "etl/src/etl/utils/backfill_agreement_signatures.py to backfill in bulk."
        )
        for agreement in missing:
            record_pending_if_absent(
                conn,
                schema,
                agreement_uuid=agreement.agreement_uuid,
                url=agreement.url,
                reason="missing_signature",
            )

    item_uuids = {item.agreement.agreement_uuid for item in items}
    new_uuids = select_unreconciled_uuids(conn, schema) & item_uuids
    if not new_uuids:
        context.log.info("Dedupe guard: no unreconciled agreements with signatures to scan.")
        return 0, set()

    context.log.info(
        f"Dedupe guard: scanning {len(new_uuids)} unreconciled agreements against "
        f"{len(items)} persisted signatures (corpus-wide)."
    )
    resolutions, reviews = _plan_corpus_dedupe(new_uuids, items)

    for review in reviews:
        context.log.warning(
            "DEDUPE REVIEW NEEDED: content similarity hit without confirmed "
            f"identity ({review.reason}); NOT auto-deleting. "
            f"new={review.new_url} existing={review.existing_url} "
            f"jaccard={review.jaccard:.3f} containment={review.containment:.3f}"
        )
        insert_review(
            conn,
            schema,
            new_agreement_uuid=review.new_agreement_uuid,
            new_url=review.new_url,
            existing_agreement_uuid=review.existing_agreement_uuid,
            existing_url=review.existing_url,
            jaccard=review.jaccard,
            containment=review.containment,
            fingerprint_match=review.fingerprint_match,
            reason=review.reason,
        )

    deleted_uuids: set[str] = set()
    if resolutions:
        _carry_over_loser_metadata(conn, schema, resolutions)
        _merge_secondary_filing_urls(conn, agreements_table, resolutions)
        deleted_uuids = {resolution.loser_uuid for resolution in resolutions}
        _delete_duplicate_agreements(conn, schema, sorted(deleted_uuids))
        delete_signatures(conn, schema, deleted_uuids)
        clear_pending(conn, schema, deleted_uuids)
        context.log.info(
            "Dedupe guard: removed "
            + f"{len(resolutions)} duplicate agreements across "
            + f"{len({resolution.survivor_uuid for resolution in resolutions})} clusters."
        )
    else:
        context.log.info("Dedupe guard: found 0 duplicate agreements.")

    # Mark the scanned scope only after all writes above succeed; on rollback
    # the scope stays unreconciled and the next run rescans it.
    mark_reconciled(conn, schema, sorted(new_uuids - deleted_uuids))
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

    def warning(self, msg: str) -> None:
        self._context.log.warning(msg)


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


def _persist_staged_signatures(
    context: AssetExecutionContext,
    conn: Connection,
    schema: str,
    staged_filings: Sequence[StagedFiling],
) -> None:
    """Persist signatures for staged filings; record failures as pending."""
    rows: list[StoredSignature] = []
    for staged in staged_filings:
        if staged.signature is None or staged.cover_identity is None:
            context.log.warning(
                "Dedupe guard: no signature computed for "
                f"{staged.metadata.url}; recording pending reconciliation."
            )
            record_pending(
                conn,
                schema,
                agreement_uuid=staged.metadata.agreement_uuid,
                url=staged.metadata.url,
                reason="signature_computation_failed",
            )
            continue
        rows.append(
            StoredSignature(
                agreement_uuid=staged.metadata.agreement_uuid,
                url=staged.metadata.url,
                signature=staged.signature,
                cover=staged.cover_identity,
            )
        )
    upsert_signatures(conn, schema, rows)


def _run_staging(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> tuple[int, list[str]]:
    target_date = _parse_staging_target_date(pipeline_config.staging_target_date)
    engine = db.get_engine()
    schema = db.database
    agreements_table = f"{schema}.agreements"
    pipeline_runs_table = f"{schema}.pipeline_runs"
    context.log.info(f"Running staging through target date {target_date}")

    # Resume from the latest recorded pull boundary even if the previous run failed later.
    with engine.begin() as conn:
        assert_tables_exist(conn, schema=schema, table_names=SIGNATURE_TABLE_NAMES)
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

    days_to_fetch = max(0, (target_date - last_run.date()).days)
    if days_to_fetch == 0:
        context.log.info(
            f"Target date {target_date} is on or before last pulled date "
            f"{last_run.date()}; no days to process."
        )

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
                staged_filings = fetch_new_filings_sec_index(
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
                staged_filings = []
            filings = [staged.metadata for staged in staged_filings]
            day_count = len(filings)
            total_count += day_count

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
                        _persist_staged_signatures(context, conn, schema, staged_filings)
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
        # EDGAR retry fetches run outside any write transaction; the
        # reconciliation pass itself is one transaction and scopes itself from
        # the persisted reconciled_at flag (not the in-memory ingest list).
        _retry_pending_signatures(context, engine, schema)
        with engine.begin() as conn:
            _, deleted_uuids = _reconcile_corpus_duplicates(context, conn, schema)
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


@dg.asset(name="01_staging_asset")
def staging_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig
) -> int:
    """Stage new filings day-by-day with incremental commits."""
    total_count, _ = _run_staging(context, db, pipeline_config)
    return total_count


@dg.asset(name="01-01_regular_ingest_staging_asset")
def regular_ingest_staging_asset(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> list[str]:
    """Stage filings and return only newly inserted, non-deduped agreement UUIDs."""
    logical_run = start_or_resume_logical_run(
        context,
        db=db,
        pipeline_config=pipeline_config,
        job_name="regular_ingest",
        initial_stage="regular_ingest_staging",
        selected_agreement_uuids=[],
    )
    resumed_scope = logical_run.agreement_uuids if logical_run is not None else []
    should_skip, current_stage = should_skip_managed_stage(
        db=db,
        job_name="regular_ingest",
        stage_name="regular_ingest_staging",
    )
    if should_skip:
        context.log.info(
            "regular_ingest_staging_asset: skipping because logical run already reached %s.",
            current_stage,
        )
        return resumed_scope

    _, scoped_agreement_uuids = _run_staging(context, db, pipeline_config)
    logical_run = start_or_resume_logical_run(
        context,
        db=db,
        pipeline_config=pipeline_config,
        job_name="regular_ingest",
        initial_stage="regular_ingest_staging",
        selected_agreement_uuids=scoped_agreement_uuids,
    )
    final_scope = logical_run.agreement_uuids if logical_run is not None else sorted(set(scoped_agreement_uuids))
    if final_scope:
        mark_logical_run_stage_completed(
            db=db,
            job_name="regular_ingest",
            stage_name="regular_ingest_staging",
        )
    context.log.info(
        "regular_ingest_staging_asset: scoped %s newly inserted non-deduped agreements.",
        len(final_scope),
    )
    return final_scope
