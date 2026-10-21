"""Per-app background workers and the after_request usage recorder.

The async task runner carries fire-and-forget side effects (notification
emails); the usage buffer batches API usage rows. Both are created lazily per
Flask app and stored in ``app.extensions``.
"""

from __future__ import annotations

import os
from typing import cast

from flask import Response, current_app

from backend.auth.session_runtime import (
    AccessContext,
    _mock_auth,
    auth_is_mocked,
    request_ip_address,
    request_user_agent,
)
from backend.core.runtime_utils import current_app_object
from backend.extensions import db
from backend.models import ApiRequestEvent, ApiUsageDaily, ApiUsageDailyIp, ApiUsageHourly
from backend.services.async_tasks import AsyncTaskRunner
from backend.services.usage import (
    LATENCY_BUCKET_BOUNDS_MS,
    UsageBuffer,
    record_api_key_usage,
    record_web_session_usage,
)

# Sample every successful request at current traffic volume; at ~1-2k calls
# per quarter a 0.05 rate left the raw event table (and latency percentiles)
# effectively empty. Retention below keeps the table bounded instead.
USAGE_SAMPLE_RATE_2XX = float(os.environ.get("USAGE_SAMPLE_RATE_2XX", "1.0"))
USAGE_SAMPLE_RATE_3XX = float(os.environ.get("USAGE_SAMPLE_RATE_3XX", "1.0"))
API_REQUEST_EVENTS_RETENTION_DAYS = int(os.environ.get("API_REQUEST_EVENTS_RETENTION_DAYS", "180"))
USAGE_BUFFER_ENABLED = os.environ.get("USAGE_LOG_BUFFER_ENABLED", "1").strip() != "0"
USAGE_BUFFER_FLUSH_SECONDS = float(os.environ.get("USAGE_LOG_BUFFER_FLUSH_SECONDS", "1"))
USAGE_BUFFER_MAX_EVENTS = int(os.environ.get("USAGE_LOG_BUFFER_MAX_EVENTS", "200"))
ASYNC_SIDE_EFFECTS_ENABLED = os.environ.get("ASYNC_SIDE_EFFECTS_ENABLED", "1").strip() != "0"
ASYNC_SIDE_EFFECTS_MAX_QUEUE = int(os.environ.get("ASYNC_SIDE_EFFECTS_MAX_QUEUE", "100"))


def usage_buffer() -> UsageBuffer | None:
    if not USAGE_BUFFER_ENABLED:
        return None
    buffer = cast(UsageBuffer | None, current_app.extensions.get("usage_buffer"))
    if buffer is None:
        buffer = UsageBuffer(
            app=current_app_object(),
            db=db,
            ApiUsageDaily=ApiUsageDaily,
            ApiUsageHourly=ApiUsageHourly,
            ApiUsageDailyIp=ApiUsageDailyIp,
            ApiRequestEvent=ApiRequestEvent,
            latency_bucket_bounds=LATENCY_BUCKET_BOUNDS_MS,
            flush_interval_seconds=USAGE_BUFFER_FLUSH_SECONDS,
            max_pending_events=USAGE_BUFFER_MAX_EVENTS,
            request_event_retention_days=API_REQUEST_EVENTS_RETENTION_DAYS,
        )
        current_app.extensions["usage_buffer"] = buffer
    return buffer


def async_task_runner() -> AsyncTaskRunner | None:
    if not ASYNC_SIDE_EFFECTS_ENABLED:
        return None
    runner = cast(AsyncTaskRunner | None, current_app.extensions.get("async_task_runner"))
    if runner is None:
        runner = AsyncTaskRunner(
            app=current_app_object(),
            max_queue_size=ASYNC_SIDE_EFFECTS_MAX_QUEUE,
        )
        current_app.extensions["async_task_runner"] = runner
    return runner


def record_request_usage(ctx: AccessContext, response: Response) -> Response:
    """Roll one finished request into the web-session and API-key usage tables."""
    record_web_session_usage(ctx=ctx, response=response, auth_is_mocked=auth_is_mocked)
    return record_api_key_usage(
        ctx=ctx,
        response=response,
        db=db,
        ApiUsageDaily=ApiUsageDaily,
        ApiUsageHourly=ApiUsageHourly,
        ApiUsageDailyIp=ApiUsageDailyIp,
        ApiRequestEvent=ApiRequestEvent,
        auth_is_mocked=auth_is_mocked,
        mock_auth=_mock_auth,
        request_ip_address=request_ip_address,
        request_user_agent=request_user_agent,
        sample_rate_2xx=USAGE_SAMPLE_RATE_2XX,
        sample_rate_3xx=USAGE_SAMPLE_RATE_3XX,
        latency_bucket_bounds=LATENCY_BUCKET_BOUNDS_MS,
        usage_buffer=usage_buffer(),
    )


__all__ = [
    "ASYNC_SIDE_EFFECTS_ENABLED",
    "async_task_runner",
    "record_request_usage",
    "usage_buffer",
]
