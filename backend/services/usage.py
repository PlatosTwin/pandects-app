from __future__ import annotations

import hmac
import hashlib
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime, date
from threading import Event, Lock, Thread

from flask import request, g
from sqlalchemy.exc import SQLAlchemyError


def _api_route_template() -> str | None:
    rule = request.url_rule
    if rule is not None and isinstance(rule.rule, str) and rule.rule:
        return rule.rule
    path = request.path
    if isinstance(path, str) and path:
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


@dataclass
class UsageHourlyAggregate:
    count: int
    total_ms: int
    max_ms: int
    buckets: list[int]
    request_bytes: int
    response_bytes: int


class UsageBuffer:
    def __init__(
        self,
        *,
        app,
        db,
        ApiUsageDaily,
        ApiUsageHourly,
        ApiUsageDailyIp,
        ApiRequestEvent,
        latency_bucket_bounds: tuple[int, ...],
        flush_interval_seconds: float,
        max_pending_events: int,
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
        self._lock = Lock()
        self._flush_event = Event()
        self._stop_event = Event()
        self._last_flush = time.time()
        self._pending_daily: dict[tuple[str, date], int] = {}
        self._pending_hourly: dict[tuple[str, datetime, str, str, int], UsageHourlyAggregate] = {}
        self._pending_ips: dict[tuple[str, date], set[str]] = {}
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
                ip_set = self._pending_ips.setdefault(ip_key, set())
                ip_set.add(event.ip_hash)

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
                {key: set(values) for key, values in self._pending_ips.items()},
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

    def _flush_snapshot(self, snapshot) -> None:
        daily, hourly, ips, events = snapshot
        if not daily and not hourly and not ips and not events:
            return
        with self._app.app_context():
            try:
                for (api_key_id, day), count in daily.items():
                    row = self._ApiUsageDaily.query.filter_by(
                        api_key_id=api_key_id, day=day
                    ).first()
                    if row is None:
                        self._db.session.add(
                            self._ApiUsageDaily(api_key_id=api_key_id, day=day, count=count)
                        )
                    else:
                        row.count = int(row.count) + int(count)

                for (api_key_id, hour, route, method, status_class), agg in hourly.items():
                    row = self._ApiUsageHourly.query.filter_by(
                        api_key_id=api_key_id,
                        hour=hour,
                        route=route,
                        method=method,
                        status_class=status_class,
                    ).first()
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
                        for idx, value in enumerate(agg.buckets):
                            buckets[idx] = int(buckets[idx]) + int(value)
                        row.latency_buckets = buckets
                        row.request_bytes = int(row.request_bytes) + agg.request_bytes
                        row.response_bytes = int(row.response_bytes) + agg.response_bytes

                for (api_key_id, day), ip_hashes in ips.items():
                    if not ip_hashes:
                        continue
                    existing = {
                        row.ip_hash
                        for row in self._ApiUsageDailyIp.query.filter_by(
                            api_key_id=api_key_id, day=day
                        )
                        .filter(self._ApiUsageDailyIp.ip_hash.in_(list(ip_hashes)))
                        .all()
                    }
                    missing = ip_hashes - existing
                    for ip_hash in missing:
                        self._db.session.add(
                            self._ApiUsageDailyIp(
                                api_key_id=api_key_id,
                                day=day,
                                ip_hash=ip_hash,
                                first_seen_at=datetime.utcnow(),
                            )
                        )

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
                        )
                    )

                self._db.session.commit()
            except SQLAlchemyError:
                self._db.session.rollback()


def build_usage_event(
    *,
    ctx,
    response,
    request_ip_address,
    request_user_agent,
    sample_rate_2xx: float,
    sample_rate_3xx: float,
    latency_bucket_bounds: tuple[int, ...],
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
    now = datetime.utcnow()
    status_code = int(response.status_code)
    status_class = status_code // 100
    elapsed_ms = 0
    start = getattr(g, "request_start", None)
    if isinstance(start, (int, float)):
        elapsed_ms = max(0, int((time.perf_counter() - start) * 1000))
    req_bytes = request.content_length
    req_bytes_int = int(req_bytes) if isinstance(req_bytes, int) else 0
    resp_bytes = response.content_length
    resp_bytes_int = int(resp_bytes) if isinstance(resp_bytes, int) else 0
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
    )


def record_api_key_usage(
    *,
    ctx,
    response,
    db,
    ApiUsageDaily,
    ApiUsageHourly,
    ApiUsageDailyIp,
    ApiRequestEvent,
    auth_is_mocked,
    mock_auth,
    request_ip_address,
    request_user_agent,
    sample_rate_2xx: float,
    sample_rate_3xx: float,
    latency_bucket_bounds: tuple[int, ...],
    usage_buffer: UsageBuffer | None = None,
):
    if auth_is_mocked():
        mock_auth.record_usage(api_key_id=ctx.api_key_id)
        return response

    event = build_usage_event(
        ctx=ctx,
        response=response,
        request_ip_address=request_ip_address,
        request_user_agent=request_user_agent,
        sample_rate_2xx=sample_rate_2xx,
        sample_rate_3xx=sample_rate_3xx,
        latency_bucket_bounds=latency_bucket_bounds,
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
    db,
    ApiUsageDaily,
    ApiUsageHourly,
    ApiUsageDailyIp,
    ApiRequestEvent,
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
                )
            )
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
