"""Backfill auto-verified status for qualifying EDGAR agreements.

Mode overview:
- dry-run: scan candidate agreements and report matches (no writes)
- apply: scan candidate agreements and update matching rows to status='verified'
"""
# pyright: reportPrivateUsage=false, reportArgumentType=false, reportAny=false

from __future__ import annotations

import argparse
import logging
import os
import threading
import time
from collections import deque
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import requests
from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection
from requests.models import Response
from requests.sessions import Session

from etl.defs.resources import DBResource, PipelineConfig
from etl.domain.a_staging import (
    _FETCH_MAX_RETRIES,
    _FETCH_RETRY_BACKOFF_BASE,
    _render_agreement_text_and_page_count,
    should_auto_verify_agreement,
)
from etl.utils.sec_utils import SEC_USER_AGENT


_ENV_PATH = Path(__file__).resolve().parents[3] / ".env"
_UPDATE_BATCH_SIZE = 200
_SCAN_BATCH_SIZE = 100


@dataclass(frozen=True)
class AgreementCandidate:
    agreement_uuid: str
    url: str


@dataclass
class ScanStats:
    scanned: int = 0
    matched: int = 0
    updated: int = 0
    skipped_fetch_failures: int = 0
    skipped_non_matches: int = 0


@dataclass(frozen=True)
class BatchOutcome:
    checked: int
    matched_uuids: list[str]
    failed: int


class _SecRateLimitedFetcher:
    def __init__(self, *, max_requests: int, window_seconds: float) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._request_times: deque[float] = deque()
        self._rate_lock = threading.Lock()
        self._thread_local = threading.local()
        self._sessions: list[Session] = []
        self._sessions_lock = threading.Lock()

    def close(self) -> None:
        for session in self._sessions:
            session.close()

    def _acquire_rate_slot(self) -> None:
        if self._max_requests <= 0 or self._window_seconds <= 0:
            return
        while True:
            with self._rate_lock:
                now = time.monotonic()
                while self._request_times and now - self._request_times[0] >= self._window_seconds:
                    _ = self._request_times.popleft()
                if len(self._request_times) < self._max_requests:
                    self._request_times.append(now)
                    return
                wait_seconds = self._window_seconds - (now - self._request_times[0])
            time.sleep(wait_seconds)

    def _get_session(self) -> Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = Session()
            self._thread_local.session = session
            with self._sessions_lock:
                self._sessions.append(session)
        return session

    def rate_limited_get(self, url: str, **kwargs: object) -> Response:
        self._acquire_rate_slot()
        session = self._get_session()
        last_exception: Exception | None = None
        for attempt in range(_FETCH_MAX_RETRIES):
            try:
                response = session.get(url, **kwargs)
                response.raise_for_status()
                return response
            except requests.exceptions.HTTPError as exc:
                if exc.response is not None and exc.response.status_code in {429, 500, 502, 503, 504}:
                    last_exception = exc
                    if attempt < _FETCH_MAX_RETRIES - 1:
                        wait_time = _FETCH_RETRY_BACKOFF_BASE ** (attempt + 1)
                        time.sleep(wait_time)
                        continue
                raise
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                last_exception = exc
                if attempt < _FETCH_MAX_RETRIES - 1:
                    wait_time = _FETCH_RETRY_BACKOFF_BASE ** (attempt + 1)
                    time.sleep(wait_time)
                    continue
                raise
        if last_exception:
            raise last_exception
        raise RuntimeError("Unexpected: retry loop exited without result")


def _load_env(path: Path) -> None:
    if not path.exists():
        logging.warning("Env file not found at %s; using existing environment.", path)
        return
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _build_db_resource() -> DBResource:
    return DBResource(
        user=os.environ["MARIADB_USER"],
        password=os.environ["MARIADB_PASSWORD"],
        host=os.environ["MARIADB_HOST"],
        port=os.environ["MARIADB_PORT"],
        database=os.environ["MARIADB_DATABASE"],
    )


def _iter_candidates(
    conn: Connection,
    schema: str,
    limit: int | None,
) -> list[AgreementCandidate]:
    limit_sql = f"\nLIMIT {int(limit)}" if limit is not None else ""
    query = text(
        f"""
        SELECT agreement_uuid, url
        FROM {schema}.agreements
        WHERE source = 'edgar'
          AND status IS NULL
          AND url IS NOT NULL
        ORDER BY agreement_uuid{limit_sql}
        """
    )
    return [
        AgreementCandidate(
            agreement_uuid=str(cast(object, row["agreement_uuid"])),
            url=str(cast(object, row["url"])),
        )
        for row in conn.execute(query).mappings().fetchall()
    ]


def _assert_agreements_status_supports_verified(conn: Connection, schema: str) -> None:
    result = conn.execute(
        text(
            """
            SELECT COLUMN_TYPE
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = :schema
              AND TABLE_NAME = 'agreements'
              AND COLUMN_NAME = 'status'
            """
        ),
        {"schema": schema},
    ).scalar_one_or_none()
    column_type = str(result or "")
    normalized = column_type.casefold()
    if "verified" not in normalized:
        raise RuntimeError(
            "agreements.status does not support 'verified' yet. "
            "Run the agreements-status migration before running this backfill."
        )


def _fetch_exhibit_content_with_get(
    exhibit_url: str,
    rate_limited_get: Callable[..., Response],
    *,
    timeout: float = 25.0,
) -> tuple[str, bool, bool] | None:
    path = exhibit_url.split("?", 1)[0]
    suffix = path.lower().rsplit(".", 1)[-1] if "." in path else ""
    if suffix not in {"txt", "htm", "html"}:
        return None
    headers = {"User-Agent": SEC_USER_AGENT}
    response = rate_limited_get(exhibit_url, headers=headers, timeout=timeout)
    content = response.text
    if suffix in {"htm", "html"}:
        return content, False, True
    return content, True, False


def agreement_matches_auto_verify_rule(
    url: str,
    rate_limited_get: Callable[..., Response],
) -> bool:
    fetch_result = _fetch_exhibit_content_with_get(url, rate_limited_get)
    if fetch_result is None:
        return False
    content, is_txt, is_html = fetch_result
    rendered_text, page_count = _render_agreement_text_and_page_count(
        content,
        is_txt=is_txt,
        is_html=is_html,
    )
    return should_auto_verify_agreement(rendered_text, page_count)


def collect_matching_agreement_uuids(
    candidates: list[AgreementCandidate],
    *,
    max_workers: int,
    rate_limit_max_requests: int,
    rate_limit_window_seconds: float,
    stats: ScanStats | None = None,
) -> BatchOutcome:
    resolved_stats = stats or ScanStats()
    matched_uuids: list[str] = []
    fetcher = _SecRateLimitedFetcher(
        max_requests=rate_limit_max_requests,
        window_seconds=rate_limit_window_seconds,
    )
    try:
        def scan_candidate(candidate: AgreementCandidate) -> tuple[str, bool]:
            is_match = agreement_matches_auto_verify_rule(
                candidate.url,
                fetcher.rate_limited_get,
            )
            return candidate.agreement_uuid, is_match

        if max_workers <= 1:
            for candidate in candidates:
                resolved_stats.scanned += 1
                try:
                    agreement_uuid, is_match = scan_candidate(candidate)
                except requests.exceptions.RequestException as exc:
                    resolved_stats.skipped_fetch_failures += 1
                    _ = exc
                    continue
                if is_match:
                    matched_uuids.append(agreement_uuid)
                    resolved_stats.matched += 1
                else:
                    resolved_stats.skipped_non_matches += 1
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(scan_candidate, candidate): candidate
                    for candidate in candidates
                }
                for future in as_completed(futures):
                    candidate = futures[future]
                    resolved_stats.scanned += 1
                    try:
                        agreement_uuid, is_match = future.result()
                    except requests.exceptions.RequestException as exc:
                        resolved_stats.skipped_fetch_failures += 1
                        _ = exc
                        continue
                    if is_match:
                        matched_uuids.append(agreement_uuid)
                        resolved_stats.matched += 1
                    else:
                        resolved_stats.skipped_non_matches += 1
    finally:
        fetcher.close()
    return BatchOutcome(
        checked=resolved_stats.scanned,
        matched_uuids=matched_uuids,
        failed=resolved_stats.skipped_fetch_failures,
    )


def apply_status_updates(
    conn: Connection,
    schema: str,
    agreement_uuids: list[str],
) -> int:
    if not agreement_uuids:
        return 0
    update_sql = text(
        f"""
        UPDATE {schema}.agreements
        SET status = 'verified'
        WHERE agreement_uuid IN :agreement_uuids
          AND status IS NULL
        """
    ).bindparams(bindparam("agreement_uuids", expanding=True))
    total_updated = 0
    for start in range(0, len(agreement_uuids), _UPDATE_BATCH_SIZE):
        batch = agreement_uuids[start : start + _UPDATE_BATCH_SIZE]
        result = conn.execute(update_sql, {"agreement_uuids": batch})
        total_updated += int(result.rowcount or 0)
    return total_updated


def _parse_args() -> argparse.Namespace:
    pipeline_config = PipelineConfig()
    parser = argparse.ArgumentParser(
        description="Backfill status='verified' for qualifying EDGAR agreements."
    )
    _ = parser.add_argument(
        "--mode",
        choices=("dry-run", "apply"),
        required=True,
        help="dry-run to report matches, apply to update matching rows.",
    )
    _ = parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on candidate agreements to scan.",
    )
    _ = parser.add_argument(
        "--max-workers",
        type=int,
        default=pipeline_config.staging_max_workers,
        help="Parallel workers for SEC fetches.",
    )
    _ = parser.add_argument(
        "--rate-limit-max-requests",
        type=int,
        default=pipeline_config.staging_rate_limit_max_requests,
        help="Max SEC requests per rate-limit window.",
    )
    _ = parser.add_argument(
        "--rate-limit-window-seconds",
        type=float,
        default=pipeline_config.staging_rate_limit_window_seconds,
        help="SEC rate-limit window in seconds.",
    )
    return parser.parse_args()


def run(
    mode: str,
    limit: int | None,
    *,
    max_workers: int,
    rate_limit_max_requests: int,
    rate_limit_window_seconds: float,
) -> ScanStats:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s backfill_auto_verify_edgar: %(message)s",
    )
    _load_env(_ENV_PATH)
    db = _build_db_resource()
    engine = db.get_engine()

    with engine.connect() as conn:
        _assert_agreements_status_supports_verified(conn, db.database)
        candidates = _iter_candidates(conn, db.database, limit)

    logging.info(
        "Scanning %s candidate agreements with max_workers=%s rate_limit=%s/%ss",
        len(candidates),
        max_workers,
        rate_limit_max_requests,
        rate_limit_window_seconds,
    )
    stats = ScanStats()
    total_batches = (len(candidates) + _SCAN_BATCH_SIZE - 1) // _SCAN_BATCH_SIZE
    for batch_index, start in enumerate(range(0, len(candidates), _SCAN_BATCH_SIZE), start=1):
        batch = candidates[start : start + _SCAN_BATCH_SIZE]
        outcome = collect_matching_agreement_uuids(
            batch,
            max_workers=max_workers,
            rate_limit_max_requests=rate_limit_max_requests,
            rate_limit_window_seconds=rate_limit_window_seconds,
        )
        stats.scanned += outcome.checked
        stats.matched += len(outcome.matched_uuids)
        stats.skipped_fetch_failures += outcome.failed
        stats.skipped_non_matches += outcome.checked - len(outcome.matched_uuids) - outcome.failed

        if mode == "apply" and outcome.matched_uuids:
            with engine.begin() as conn:
                stats.updated += apply_status_updates(conn, db.database, outcome.matched_uuids)

        logging.info(
            "Batch %s/%s: checked=%s verified=%s failed=%s",
            batch_index,
            total_batches,
            outcome.checked,
            len(outcome.matched_uuids),
            outcome.failed,
        )

    logging.info(
        "Complete: checked=%s verified=%s failed=%s updated=%s",
        stats.scanned,
        stats.matched,
        stats.skipped_fetch_failures,
        stats.updated,
    )

    return stats


def _run_cli() -> None:
    args = _parse_args()
    mode = cast(str, args.mode)
    limit = cast(int | None, args.limit)
    max_workers = cast(int, args.max_workers)
    rate_limit_max_requests = cast(int, args.rate_limit_max_requests)
    rate_limit_window_seconds = cast(float, args.rate_limit_window_seconds)
    _ = run(
        mode,
        limit,
        max_workers=max_workers,
        rate_limit_max_requests=rate_limit_max_requests,
        rate_limit_window_seconds=rate_limit_window_seconds,
    )


if __name__ == "__main__":
    _run_cli()
