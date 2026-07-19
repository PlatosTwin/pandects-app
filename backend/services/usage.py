from __future__ import annotations

import hmac
import hashlib
import logging
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, date, timedelta, timezone
from threading import Event, Lock, Thread
from collections.abc import Callable
from typing import Any, Protocol, cast

from flask import Flask, Response, current_app, request, g
from sqlalchemy import tuple_
from sqlalchemy.exc import IntegrityError, SQLAlchemyError


# Canonical latency histogram bounds for every usage rollup (api_key, web
# session, MCP tools) so the dashboard can render one bucket legend.
LATENCY_BUCKET_BOUNDS_MS = (25, 50, 100, 250, 500, 1000, 2000, 5000, 10000)


def _api_route_template() -> str | None:
    rule = request.url_rule
    if rule is not None and rule.rule:
        return rule.rule
    path = request.path
    if path:
        return path
    return None


def _ip_hash(value: str | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    secret = os.environ.get("AUTH_SECRET_KEY")
    if not isinstance(secret, str) or not secret.strip():
        return None
    digest = hmac.new(secret.encode("utf-8"), value.strip().encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()


def request_country() -> str | None:
    """Coarse request geo without storing PII.

    Prefers an ISO-3166 alpha-2 country code from a CDN header (uppercase),
    falling back to the Fly edge region that accepted the connection
    (lowercase 3-letter code) — distinguishable from countries by case and
    length. Returns None when neither is present.
    """
    for header in ("CF-IPCountry", "X-Vercel-IP-Country"):
        raw = request.headers.get(header)
        if isinstance(raw, str):
            value = raw.strip().upper()
            if len(value) == 2 and value.isalpha() and value != "XX":
                return value
    region = request.headers.get("Fly-Region")
    if isinstance(region, str):
        value = region.strip().lower()
        if 2 <= len(value) <= 4 and value.isalpha():
            return value
    return None


def _usage_event_sample_rate(status_code: int, *, sample_rate_2xx: float, sample_rate_3xx: float) -> float:
    if status_code >= 400:
        return 1.0
    if 300 <= status_code <= 399:
        return sample_rate_3xx
    return sample_rate_2xx


def _init_latency_buckets(bounds: tuple[int, ...]) -> list[int]:
    return [0] * (len(bounds) + 1)


def _latency_bucket_index(bounds: tuple[int, ...], elapsed_ms: int) -> int:
    for idx, bound in enumerate(bounds):
        if elapsed_ms <= bound:
            return idx
    return len(bounds)


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass(frozen=True)
class UsageEvent:
    api_key_id: str
    occurred_at: datetime
    day: date
    hour: datetime
    route: str
    method: str
    status_code: int
    status_class: int
    latency_ms: int
    request_bytes: int
    response_bytes: int
    ip_hash: str | None
    user_agent: str | None
    include_request_event: bool
    country: str | None = None


@dataclass
class UsageHourlyAggregate:
    count: int
    total_ms: int
    max_ms: int
    buckets: list[int]
    request_bytes: int
    response_bytes: int


class AccessContextLike(Protocol):
    tier: str
    api_key_id: str | None


class MockAuthUsageLike(Protocol):
    def record_usage(self, *, api_key_id: str) -> None:
        ...


FlushSnapshot = tuple[
    dict[tuple[str, date], int],
    dict[tuple[str, datetime, str, str, int], UsageHourlyAggregate],
    dict[tuple[str, date], dict[str, str | None]],
    list[UsageEvent],
]


class UsageBuffer:
    def __init__(
        self,
        *,
        app: Flask,
        db: Any,
        ApiUsageDaily: Any,
        ApiUsageHourly: Any,
        ApiUsageDailyIp: Any,
        ApiRequestEvent: Any,
        latency_bucket_bounds: tuple[int, ...],
        flush_interval_seconds: float,
        max_pending_events: int,
        request_event_retention_days: int = 0,
    ) -> None:
        self._app = app
        self._db = db
        self._ApiUsageDaily = ApiUsageDaily
        self._ApiUsageHourly = ApiUsageHourly
        self._ApiUsageDailyIp = ApiUsageDailyIp
        self._ApiRequestEvent = ApiRequestEvent
        self._latency_bucket_bounds = latency_bucket_bounds
        self._flush_interval_seconds = max(0.1, float(flush_interval_seconds))
        self._max_pending_events = max(1, int(max_pending_events))
        self._request_event_retention_days = max(0, int(request_event_retention_days))
        # Start the prune clock at boot so N workers don't all fire a
        # retention DELETE on their first flush after every deploy. With a
        # 180-day window, drifting one prune interval is immaterial.
        self._last_prune = time.time()
        self._lock = Lock()
        self._flush_event = Event()
        self._stop_event = Event()
        self._last_flush = time.time()
        self._pending_daily: dict[tuple[str, date], int] = {}
        self._pending_hourly: dict[tuple[str, datetime, str, str, int], UsageHourlyAggregate] = {}
        self._pending_ips: dict[tuple[str, date], dict[str, str | None]] = {}
        self._pending_events: list[UsageEvent] = []
        self._pending_count = 0
        self._force_flush = False
        self._thread = Thread(target=self._run, name="usage-buffer", daemon=True)
        self._thread.start()

    def enqueue(self, event: UsageEvent) -> None:
        with self._lock:
            daily_key = (event.api_key_id, event.day)
            self._pending_daily[daily_key] = self._pending_daily.get(daily_key, 0) + 1

            hourly_key = (
                event.api_key_id,
                event.hour,
                event.route,
                event.method,
                event.status_class,
            )
            bucket_index = _latency_bucket_index(self._latency_bucket_bounds, event.latency_ms)
            hourly = self._pending_hourly.get(hourly_key)
            if hourly is None:
                buckets = _init_latency_buckets(self._latency_bucket_bounds)
                buckets[bucket_index] = 1
                hourly = UsageHourlyAggregate(
                    count=1,
                    total_ms=event.latency_ms,
                    max_ms=event.latency_ms,
                    buckets=buckets,
                    request_bytes=event.request_bytes,
                    response_bytes=event.response_bytes,
                )
                self._pending_hourly[hourly_key] = hourly
            else:
                hourly.count += 1
                hourly.total_ms += event.latency_ms
                hourly.max_ms = max(hourly.max_ms, event.latency_ms)
                hourly.buckets[bucket_index] += 1
                hourly.request_bytes += event.request_bytes
                hourly.response_bytes += event.response_bytes

            if event.ip_hash:
                ip_key = (event.api_key_id, event.day)
                ip_map = self._pending_ips.setdefault(ip_key, {})
                if event.ip_hash not in ip_map:
                    ip_map[event.ip_hash] = event.country

            if event.include_request_event:
                self._pending_events.append(event)

            self._pending_count += 1
            self._flush_event.set()
            if self._pending_count >= self._max_pending_events:
                self._force_flush = True
                self._flush_event.set()

    def flush(self) -> None:
        snapshot = self._drain()
        if snapshot is None:
            return
        self._flush_snapshot(snapshot)

    def stop(self) -> None:
        self._stop_event.set()
        self._flush_event.set()

    def _drain(self):
        with self._lock:
            if self._pending_count == 0:
                return None
            snapshot = (
                dict(self._pending_daily),
                dict(self._pending_hourly),
                {key: dict(values) for key, values in self._pending_ips.items()},
                list(self._pending_events),
            )
            self._pending_daily.clear()
            self._pending_hourly.clear()
            self._pending_ips.clear()
            self._pending_events.clear()
            self._pending_count = 0
            self._force_flush = False
            self._last_flush = time.time()
            return snapshot

    def _run(self) -> None:
        while not self._stop_event.is_set():
            signaled = self._flush_event.wait(timeout=self._flush_interval_seconds)
            self._flush_event.clear()
            if self._stop_event.is_set():
                break
            if not signaled:
                self.flush()
                continue
            with self._lock:
                force_flush = self._force_flush
            now = time.time()
            if force_flush or (now - self._last_flush) >= self._flush_interval_seconds:
                self.flush()
        self.flush()

    def _flush_snapshot(self, snapshot: FlushSnapshot) -> None:
        daily, hourly, ips, events = snapshot
        if not daily and not hourly and not ips and not events:
            return
        with self._app.app_context():
            try:
                daily_existing: dict[tuple[str, date], Any] = {}
                daily_keys = list(daily.keys())
                if daily_keys:
                    existing_daily_rows = (
                        self._ApiUsageDaily.query.filter(
                            tuple_(
                                self._ApiUsageDaily.api_key_id,
                                self._ApiUsageDaily.day,
                            ).in_(daily_keys)
                        ).all()
                    )
                    daily_existing = {
                        (row.api_key_id, row.day): row for row in existing_daily_rows
                    }
                for (api_key_id, day), count in daily.items():
                    row = daily_existing.get((api_key_id, day))
                    if row is None:
                        self._db.session.add(
                            self._ApiUsageDaily(api_key_id=api_key_id, day=day, count=count)
                        )
                    else:
                        row.count = int(row.count) + int(count)

                hourly_existing: dict[tuple[str, datetime, str, str, int], Any] = {}
                hourly_keys = list(hourly.keys())
                if hourly_keys:
                    existing_hourly_rows = (
                        self._ApiUsageHourly.query.filter(
                            tuple_(
                                self._ApiUsageHourly.api_key_id,
                                self._ApiUsageHourly.hour,
                                self._ApiUsageHourly.route,
                                self._ApiUsageHourly.method,
                                self._ApiUsageHourly.status_class,
                            ).in_(hourly_keys)
                        ).all()
                    )
                    hourly_existing = {
                        (
                            row.api_key_id,
                            row.hour,
                            row.route,
                            row.method,
                            row.status_class,
                        ): row
                        for row in existing_hourly_rows
                    }
                for (api_key_id, hour, route, method, status_class), agg in hourly.items():
                    row = hourly_existing.get(
                        (api_key_id, hour, route, method, status_class)
                    )
                    if row is None:
                        self._db.session.add(
                            self._ApiUsageHourly(
                                api_key_id=api_key_id,
                                hour=hour,
                                route=route,
                                method=method,
                                status_class=status_class,
                                count=agg.count,
                                total_ms=agg.total_ms,
                                max_ms=agg.max_ms,
                                latency_buckets=agg.buckets,
                                request_bytes=agg.request_bytes,
                                response_bytes=agg.response_bytes,
                            )
                        )
                    else:
                        row.count = int(row.count) + agg.count
                        row.total_ms = int(row.total_ms) + agg.total_ms
                        row.max_ms = max(int(row.max_ms), agg.max_ms)
                        buckets = row.latency_buckets
                        if not isinstance(buckets, list) or len(buckets) != len(agg.buckets):
                            buckets = _init_latency_buckets(self._latency_bucket_bounds)
                        else:
                            # Copy before mutating: reassigning the same list
                            # object to a JSON column is not detected as a
                            # change by SQLAlchemy, so merged buckets would
                            # silently fail to persist.
                            buckets = list(buckets)
                        for idx, value in enumerate(agg.buckets):
                            buckets[idx] = int(buckets[idx]) + int(value)
                        row.latency_buckets = buckets
                        row.request_bytes = int(row.request_bytes) + agg.request_bytes
                        row.response_bytes = int(row.response_bytes) + agg.response_bytes

                pending_ip_entries = [
                    (api_key_id, day, ip_hash, country)
                    for (api_key_id, day), ip_hashes in ips.items()
                    for ip_hash, country in ip_hashes.items()
                ]
                existing_ip_keys: set[tuple[str, date, str]] = set()
                if pending_ip_entries:
                    existing_ip_rows = (
                        self._ApiUsageDailyIp.query.filter(
                            tuple_(
                                self._ApiUsageDailyIp.api_key_id,
                                self._ApiUsageDailyIp.day,
                                self._ApiUsageDailyIp.ip_hash,
                            ).in_(
                                [(a, d, i) for a, d, i, _ in pending_ip_entries]
                            )
                        ).all()
                    )
                    existing_ip_keys = {
                        (row.api_key_id, row.day, row.ip_hash)
                        for row in existing_ip_rows
                    }
                for api_key_id, day, ip_hash, country in pending_ip_entries:
                    if (api_key_id, day, ip_hash) in existing_ip_keys:
                        continue
                    self._db.session.add(
                        self._ApiUsageDailyIp(
                            api_key_id=api_key_id,
                            day=day,
                            ip_hash=ip_hash,
                            first_seen_at=_utc_now_naive(),
                            country=country,
                        )
                    )
                    existing_ip_keys.add((api_key_id, day, ip_hash))

                for event in events:
                    if not event.include_request_event:
                        continue
                    self._db.session.add(
                        self._ApiRequestEvent(
                            api_key_id=event.api_key_id,
                            occurred_at=event.occurred_at,
                            route=event.route,
                            method=event.method,
                            status_code=event.status_code,
                            status_class=event.status_class,
                            latency_ms=event.latency_ms,
                            request_bytes=event.request_bytes,
                            response_bytes=event.response_bytes,
                            ip_hash=event.ip_hash,
                            user_agent=event.user_agent,
                            country=event.country,
                        )
                    )

                self._db.session.commit()
                self._prune_request_events_if_due()
            except SQLAlchemyError:
                self._db.session.rollback()

    _PRUNE_INTERVAL_SECONDS = 6 * 3600

    def _prune_request_events_if_due(self) -> None:
        """Bounded retention for the raw event table (runs inside the flush
        app context). Disabled when retention_days is 0."""
        if self._request_event_retention_days <= 0:
            return
        now = time.time()
        if now - self._last_prune < self._PRUNE_INTERVAL_SECONDS:
            return
        self._last_prune = now
        cutoff = _utc_now_naive() - timedelta(days=self._request_event_retention_days)
        try:
            _ = self._ApiRequestEvent.query.filter(
                self._ApiRequestEvent.occurred_at < cutoff
            ).delete(synchronize_session=False)
            self._db.session.commit()
        except SQLAlchemyError:
            self._db.session.rollback()


def build_usage_event(
    *,
    ctx: AccessContextLike,
    response: Response,
    request_ip_address: Callable[[], str | None],
    request_user_agent: Callable[[], str | None],
    sample_rate_2xx: float,
    sample_rate_3xx: float,
) -> UsageEvent | None:
    if ctx.tier != "api_key" or not ctx.api_key_id:
        return None
    if not request.path.startswith("/v1/"):
        return None
    if request.path.startswith("/v1/auth/"):
        return None

    route = _api_route_template()
    if route is None:
        return None
    now = _utc_now_naive()
    status_code = int(response.status_code)
    status_class = status_code // 100
    elapsed_ms = 0
    start = getattr(g, "request_start", None)
    if isinstance(start, (int, float)):
        elapsed_ms = max(0, int((time.perf_counter() - start) * 1000))
    req_bytes_int = int(request.content_length or 0)
    resp_bytes_int = int(response.content_length or 0)
    ip_hash = _ip_hash(request_ip_address())
    sample_rate = _usage_event_sample_rate(
        status_code, sample_rate_2xx=sample_rate_2xx, sample_rate_3xx=sample_rate_3xx
    )
    include_request_event = random.random() < sample_rate
    user_agent = request_user_agent()

    return UsageEvent(
        api_key_id=ctx.api_key_id,
        occurred_at=now,
        day=now.date(),
        hour=now.replace(minute=0, second=0, microsecond=0),
        route=route,
        method=request.method,
        status_code=status_code,
        status_class=status_class,
        latency_ms=elapsed_ms,
        request_bytes=req_bytes_int,
        response_bytes=resp_bytes_int,
        ip_hash=ip_hash,
        user_agent=user_agent,
        include_request_event=include_request_event,
        country=request_country(),
    )


def record_api_key_usage(
    *,
    ctx: AccessContextLike,
    response: Response,
    db: Any,
    ApiUsageDaily: Any,
    ApiUsageHourly: Any,
    ApiUsageDailyIp: Any,
    ApiRequestEvent: Any,
    auth_is_mocked: Callable[[], bool],
    mock_auth: MockAuthUsageLike,
    request_ip_address: Callable[[], str | None],
    request_user_agent: Callable[[], str | None],
    sample_rate_2xx: float,
    sample_rate_3xx: float,
    latency_bucket_bounds: tuple[int, ...],
    usage_buffer: UsageBuffer | None = None,
) -> Response:
    if auth_is_mocked():
        if ctx.api_key_id is not None:
            mock_auth.record_usage(api_key_id=ctx.api_key_id)
        return response

    event = build_usage_event(
        ctx=ctx,
        response=response,
        request_ip_address=request_ip_address,
        request_user_agent=request_user_agent,
        sample_rate_2xx=sample_rate_2xx,
        sample_rate_3xx=sample_rate_3xx,
    )
    if event is None:
        return response
    if usage_buffer is None:
        _persist_usage_event(
            event=event,
            db=db,
            ApiUsageDaily=ApiUsageDaily,
            ApiUsageHourly=ApiUsageHourly,
            ApiUsageDailyIp=ApiUsageDailyIp,
            ApiRequestEvent=ApiRequestEvent,
            latency_bucket_bounds=latency_bucket_bounds,
        )
        return response

    usage_buffer.enqueue(event)
    return response


def _persist_usage_event(
    *,
    event: UsageEvent,
    db: Any,
    ApiUsageDaily: Any,
    ApiUsageHourly: Any,
    ApiUsageDailyIp: Any,
    ApiRequestEvent: Any,
    latency_bucket_bounds: tuple[int, ...],
) -> None:
    try:
        row = ApiUsageDaily.query.filter_by(api_key_id=event.api_key_id, day=event.day).first()
        if row is None:
            row = ApiUsageDaily(api_key_id=event.api_key_id, day=event.day, count=1)
            db.session.add(row)
        else:
            row.count = int(row.count) + 1

        hourly = ApiUsageHourly.query.filter_by(
            api_key_id=event.api_key_id,
            hour=event.hour,
            route=event.route,
            method=event.method,
            status_class=event.status_class,
        ).first()
        bucket_index = _latency_bucket_index(latency_bucket_bounds, event.latency_ms)
        if hourly is None:
            buckets = _init_latency_buckets(latency_bucket_bounds)
            buckets[bucket_index] = 1
            hourly = ApiUsageHourly(
                api_key_id=event.api_key_id,
                hour=event.hour,
                route=event.route,
                method=event.method,
                status_class=event.status_class,
                count=1,
                total_ms=event.latency_ms,
                max_ms=event.latency_ms,
                latency_buckets=buckets,
                request_bytes=event.request_bytes,
                response_bytes=event.response_bytes,
            )
            db.session.add(hourly)
        else:
            hourly.count = int(hourly.count) + 1
            hourly.total_ms = int(hourly.total_ms) + event.latency_ms
            hourly.max_ms = max(int(hourly.max_ms), event.latency_ms)
            buckets = hourly.latency_buckets
            if not isinstance(buckets, list) or len(buckets) != len(latency_bucket_bounds) + 1:
                buckets = _init_latency_buckets(latency_bucket_bounds)
            else:
                # Copy before mutating: reassigning the same list object to a
                # JSON column is not detected as a change by SQLAlchemy.
                buckets = list(buckets)
            buckets[bucket_index] = int(buckets[bucket_index]) + 1
            hourly.latency_buckets = buckets
            hourly.request_bytes = int(hourly.request_bytes) + event.request_bytes
            hourly.response_bytes = int(hourly.response_bytes) + event.response_bytes

        if event.ip_hash is not None:
            existing_ip = ApiUsageDailyIp.query.filter_by(
                api_key_id=event.api_key_id, day=event.day, ip_hash=event.ip_hash
            ).first()
            if existing_ip is None:
                db.session.add(
                    ApiUsageDailyIp(
                        api_key_id=event.api_key_id,
                        day=event.day,
                        ip_hash=event.ip_hash,
                        first_seen_at=event.occurred_at,
                        country=event.country,
                    )
                )

        if event.include_request_event:
            db.session.add(
                ApiRequestEvent(
                    api_key_id=event.api_key_id,
                    occurred_at=event.occurred_at,
                    route=event.route,
                    method=event.method,
                    status_code=event.status_code,
                    status_class=event.status_class,
                    latency_ms=event.latency_ms,
                    request_bytes=event.request_bytes,
                    response_bytes=event.response_bytes,
                    ip_hash=event.ip_hash,
                    user_agent=event.user_agent,
                    country=event.country,
                )
            )
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()


# ── Generic hourly rollups (MCP tool calls, web-session traffic) ──────────


@dataclass
class HourlyRollupAggregate:
    count: int
    total_ms: int
    max_ms: int
    buckets: list[int]
    request_bytes: int
    response_bytes: int


def _flush_rollup_rows(
    *,
    db: Any,
    model: Any,
    key_columns: tuple[str, ...],
    latency_bucket_bounds: tuple[int, ...],
    pending: dict[tuple[object, ...], HourlyRollupAggregate],
) -> None:
    """Upsert aggregated rollup rows for a model whose natural key is
    ``key_columns`` and whose metric columns match ApiUsageHourly's. Caller
    owns the session (commit/rollback).

    Existing rows are selected FOR UPDATE so concurrent workers merging into
    the same natural key serialize instead of losing increments (no-op on
    SQLite, which is single-writer anyway). A concurrent INSERT of the same
    key still raises IntegrityError at commit — callers retry once, at which
    point the row exists and this merges into it.
    """
    if not pending:
        return
    key_attrs = [getattr(model, name) for name in key_columns]
    existing_rows = (
        model.query.filter(tuple_(*key_attrs).in_(list(pending.keys())))
        .with_for_update()
        .all()
    )
    existing = {
        tuple(getattr(row, name) for name in key_columns): row for row in existing_rows
    }
    for key, agg in pending.items():
        row = existing.get(key)
        if row is None:
            row = model(**dict(zip(key_columns, key)))
            row.count = agg.count
            row.total_ms = agg.total_ms
            row.max_ms = agg.max_ms
            row.latency_buckets = agg.buckets
            row.request_bytes = agg.request_bytes
            row.response_bytes = agg.response_bytes
            db.session.add(row)
        else:
            row.count = int(row.count) + agg.count
            row.total_ms = int(row.total_ms) + agg.total_ms
            row.max_ms = max(int(row.max_ms), agg.max_ms)
            buckets = row.latency_buckets
            if not isinstance(buckets, list) or len(buckets) != len(agg.buckets):
                buckets = _init_latency_buckets(latency_bucket_bounds)
            else:
                # Copy before mutating: reassigning the same list object to a
                # JSON column is not detected as a change by SQLAlchemy.
                buckets = list(buckets)
            for idx, value in enumerate(agg.buckets):
                buckets[idx] = int(buckets[idx]) + int(value)
            row.latency_buckets = buckets
            row.request_bytes = int(row.request_bytes) + agg.request_bytes
            row.response_bytes = int(row.response_bytes) + agg.response_bytes


def _commit_rollup_pending(
    *,
    db: Any,
    model: Any,
    key_columns: tuple[str, ...],
    latency_bucket_bounds: tuple[int, ...],
    pending: dict[tuple[object, ...], HourlyRollupAggregate],
) -> None:
    """Flush + commit pending rollups, retrying once on IntegrityError.

    Two workers can race to INSERT the same natural key; the loser's whole
    commit fails, so without a retry an entire snapshot of aggregates would
    be dropped. On retry the SELECT finds the winner's row and merges.
    """
    for attempt in (0, 1):
        try:
            _flush_rollup_rows(
                db=db,
                model=model,
                key_columns=key_columns,
                latency_bucket_bounds=latency_bucket_bounds,
                pending=pending,
            )
            db.session.commit()
            return
        except IntegrityError:
            db.session.rollback()
            if attempt == 1:
                raise
        except SQLAlchemyError:
            db.session.rollback()
            raise


class HourlyRollupBuffer:
    """Buffered writer for hourly usage rollups keyed by an arbitrary
    natural-key tuple. Same lifecycle as UsageBuffer: a daemon thread drains
    pending aggregates every flush interval (or when the pending count hits
    the cap)."""

    def __init__(
        self,
        *,
        app: Flask,
        db: Any,
        model: Any,
        key_columns: tuple[str, ...],
        latency_bucket_bounds: tuple[int, ...],
        flush_interval_seconds: float,
        max_pending_events: int,
        thread_name: str,
    ) -> None:
        self._app = app
        self._db = db
        self._model = model
        self._key_columns = key_columns
        self._latency_bucket_bounds = latency_bucket_bounds
        self._flush_interval_seconds = max(0.1, float(flush_interval_seconds))
        self._max_pending_events = max(1, int(max_pending_events))
        self._lock = Lock()
        self._flush_event = Event()
        self._stop_event = Event()
        self._last_flush = time.time()
        self._pending: dict[tuple[object, ...], HourlyRollupAggregate] = {}
        self._pending_count = 0
        self._force_flush = False
        self._thread = Thread(target=self._run, name=thread_name, daemon=True)
        self._thread.start()

    def record(
        self,
        *,
        key: tuple[object, ...],
        latency_ms: int,
        request_bytes: int = 0,
        response_bytes: int = 0,
    ) -> None:
        bucket_index = _latency_bucket_index(self._latency_bucket_bounds, latency_ms)
        with self._lock:
            agg = self._pending.get(key)
            if agg is None:
                buckets = _init_latency_buckets(self._latency_bucket_bounds)
                buckets[bucket_index] = 1
                self._pending[key] = HourlyRollupAggregate(
                    count=1,
                    total_ms=latency_ms,
                    max_ms=latency_ms,
                    buckets=buckets,
                    request_bytes=request_bytes,
                    response_bytes=response_bytes,
                )
            else:
                agg.count += 1
                agg.total_ms += latency_ms
                agg.max_ms = max(agg.max_ms, latency_ms)
                agg.buckets[bucket_index] += 1
                agg.request_bytes += request_bytes
                agg.response_bytes += response_bytes
            self._pending_count += 1
            self._flush_event.set()
            if self._pending_count >= self._max_pending_events:
                self._force_flush = True
                self._flush_event.set()

    def flush(self) -> None:
        with self._lock:
            if self._pending_count == 0:
                return
            pending = dict(self._pending)
            self._pending.clear()
            self._pending_count = 0
            self._force_flush = False
            self._last_flush = time.time()
        with self._app.app_context():
            try:
                _commit_rollup_pending(
                    db=self._db,
                    model=self._model,
                    key_columns=self._key_columns,
                    latency_bucket_bounds=self._latency_bucket_bounds,
                    pending=pending,
                )
            except SQLAlchemyError:
                self._db.session.rollback()

    def stop(self) -> None:
        self._stop_event.set()
        self._flush_event.set()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            signaled = self._flush_event.wait(timeout=self._flush_interval_seconds)
            self._flush_event.clear()
            if self._stop_event.is_set():
                break
            if not signaled:
                self.flush()
                continue
            with self._lock:
                force_flush = self._force_flush
            now = time.time()
            if force_flush or (now - self._last_flush) >= self._flush_interval_seconds:
                self.flush()
        self.flush()


def _rollup_buffer_config() -> tuple[bool, float, int]:
    enabled = os.environ.get("USAGE_LOG_BUFFER_ENABLED", "1").strip() != "0"
    try:
        flush_seconds = float(os.environ.get("USAGE_LOG_BUFFER_FLUSH_SECONDS", "1"))
    except ValueError:
        flush_seconds = 1.0
    try:
        max_events = int(os.environ.get("USAGE_LOG_BUFFER_MAX_EVENTS", "200"))
    except ValueError:
        max_events = 200
    return enabled, flush_seconds, max_events


def _get_rollup_buffer(
    *,
    extension_key: str,
    model: Any,
    key_columns: tuple[str, ...],
    thread_name: str,
) -> HourlyRollupBuffer | None:
    enabled, flush_seconds, max_events = _rollup_buffer_config()
    if not enabled:
        return None
    buffer = current_app.extensions.get(extension_key)
    if buffer is None:
        from backend.extensions import db as _db

        buffer = HourlyRollupBuffer(
            app=cast(Flask, current_app._get_current_object()),  # pyright: ignore[reportPrivateUsage]
            db=_db,
            model=model,
            key_columns=key_columns,
            latency_bucket_bounds=LATENCY_BUCKET_BOUNDS_MS,
            flush_interval_seconds=flush_seconds,
            max_pending_events=max_events,
            thread_name=thread_name,
        )
        current_app.extensions[extension_key] = buffer
    return cast(HourlyRollupBuffer, buffer)


def _record_hourly_rollup(
    *,
    extension_key: str,
    model: Any,
    key_columns: tuple[str, ...],
    thread_name: str,
    key: tuple[object, ...],
    latency_ms: int,
    request_bytes: int = 0,
    response_bytes: int = 0,
) -> None:
    buffer = _get_rollup_buffer(
        extension_key=extension_key,
        model=model,
        key_columns=key_columns,
        thread_name=thread_name,
    )
    if buffer is not None:
        buffer.record(
            key=key,
            latency_ms=latency_ms,
            request_bytes=request_bytes,
            response_bytes=response_bytes,
        )
        return
    from backend.extensions import db as _db

    bucket_index = _latency_bucket_index(LATENCY_BUCKET_BOUNDS_MS, latency_ms)
    buckets = _init_latency_buckets(LATENCY_BUCKET_BOUNDS_MS)
    buckets[bucket_index] = 1
    try:
        _commit_rollup_pending(
            db=_db,
            model=model,
            key_columns=key_columns,
            latency_bucket_bounds=LATENCY_BUCKET_BOUNDS_MS,
            pending={
                key: HourlyRollupAggregate(
                    count=1,
                    total_ms=latency_ms,
                    max_ms=latency_ms,
                    buckets=buckets,
                    request_bytes=request_bytes,
                    response_bytes=response_bytes,
                )
            },
        )
    except SQLAlchemyError:
        _db.session.rollback()


def record_mcp_tool_usage(
    *,
    user_id: str,
    client_id: str | None,
    tool_name: str,
    outcome: str,
    error_category: str | None,
    latency_ms: int,
    request_bytes: int = 0,
) -> None:
    """Roll an MCP tool call into mcp_usage_hourly. Never raises: usage
    accounting must not break tool responses."""
    from backend.models import McpUsageHourly

    status = "ok" if outcome == "ok" else (error_category or "error")
    now = _utc_now_naive()
    key = (
        user_id,
        (client_id or "").strip()[:128],
        now.replace(minute=0, second=0, microsecond=0),
        tool_name[:128],
        status[:32],
    )
    try:
        _record_hourly_rollup(
            extension_key="mcp_usage_buffer",
            model=McpUsageHourly,
            key_columns=("user_id", "client_id", "hour", "tool_name", "status"),
            thread_name="mcp-usage-buffer",
            key=key,
            latency_ms=max(0, int(latency_ms)),
            request_bytes=max(0, int(request_bytes)),
        )
    except Exception:
        logging.getLogger(__name__).exception("mcp_usage_rollup_failed")


def record_web_session_usage(
    *,
    ctx: AccessContextLike,
    response: Response,
    auth_is_mocked: Callable[[], bool],
) -> None:
    """Roll a session-authenticated (browser) API request into
    web_usage_hourly. Mirrors build_usage_event's route filters. Never
    raises."""
    user_id = getattr(ctx, "user_id", None)
    if ctx.tier != "user" or not isinstance(user_id, str) or not user_id:
        return
    if auth_is_mocked():
        return
    path = request.path
    if not path.startswith("/v1/"):
        return
    if path.startswith("/v1/auth/") or path.startswith("/v1/page-views"):
        return
    route = _api_route_template()
    if route is None:
        return
    now = _utc_now_naive()
    elapsed_ms = 0
    start = getattr(g, "request_start", None)
    if isinstance(start, (int, float)):
        elapsed_ms = max(0, int((time.perf_counter() - start) * 1000))
    from backend.models import WebUsageHourly

    key = (
        user_id,
        now.replace(minute=0, second=0, microsecond=0),
        route[:256],
        request.method[:8],
        int(response.status_code) // 100,
    )
    try:
        _record_hourly_rollup(
            extension_key="web_usage_buffer",
            model=WebUsageHourly,
            key_columns=("user_id", "hour", "route", "method", "status_class"),
            thread_name="web-usage-buffer",
            key=key,
            latency_ms=elapsed_ms,
            request_bytes=int(request.content_length or 0),
            response_bytes=int(response.content_length or 0),
        )
    except Exception:
        logging.getLogger(__name__).exception("web_usage_rollup_failed")
