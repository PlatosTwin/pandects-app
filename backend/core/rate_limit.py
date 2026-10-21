"""In-process rate limiting for ``/v1`` routes.

Three layers, all fixed one-minute windows held in process memory:

- a per-caller bucket keyed by access tier (API key, user, or anonymous IP);
  paginated search reads get their own higher-ceiling bucket;
- per-endpoint, per-IP limits for abuse-prone POSTs (sign-up, login, ...);
- a per-account login-failure throttle with a back-off window.

Counts are local to each process: on a horizontally scaled deployment the
effective limit is N times the per-process threshold.
"""

from __future__ import annotations

import hashlib
import os
import time
from threading import Lock

from flask import abort, request

from backend.auth.session_runtime import AccessContext, request_ip_address
from backend.core.errors import json_error

RATE_LIMIT_WINDOW_SECONDS = 60.0
RATE_LIMIT_MAX_KEYS = int(os.environ.get("RATE_LIMIT_MAX_KEYS", "50000"))
RATE_LIMIT_PRUNE_INTERVAL_SECONDS = float(
    os.environ.get("RATE_LIMIT_PRUNE_INTERVAL_SECONDS", "30")
)

rate_limit_lock = Lock()
rate_limit_state: dict[str, dict[str, float | int]] = {}
endpoint_rate_limit_state: dict[str, dict[str, float | int]] = {}
# Read and written by prune_rate_limit_state; tests reset it through this module.
rate_limit_last_prune_at = 0.0

# Read-only search endpoints are paginated, so a single user session legitimately
# fans out into many GETs. Give those reads their own higher-ceiling bucket
# (separate key prefix) so deep pagination is not throttled by — and does not
# starve — the general per-tier limit shared by all other /v1 traffic.
SEARCH_READ_RATE_LIMIT_PATHS = frozenset(
    {
        "/v1/sections",
        "/v1/search/agreements",
        "/v1/tax-clauses",
    }
)
SEARCH_READ_RATE_LIMITS_BY_TIER = {
    "anonymous": 300,
    "user": 600,
    "api_key": 1200,
}

ENDPOINT_RATE_LIMITS: dict[tuple[str, str], int] = {
    ("POST", "/v1/page-views"): 60,
    ("POST", "/v1/auth/flag-inaccurate"): 10,
    ("POST", "/v1/auth/signup/password"): 5,
    ("POST", "/v1/auth/login/password"): 10,
    ("POST", "/v1/auth/password-reset/request"): 3,
    ("POST", "/v1/auth/email/verify/resend"): 5,
    ("POST", "/v1/auth/oauth/register"): 10,
}

# Per-account login-failure throttle. Layers on top of the per-IP endpoint
# limit so an attacker can't spread credential-stuffing attempts across many
# IPs against the same email. Keyed on SHA-256(email) so plaintext addresses
# never sit in process memory or logs.
account_login_failure_lock = Lock()
account_login_failure_state: dict[str, dict[str, float | int]] = {}
ACCOUNT_LOGIN_FAILURE_WINDOW_SECONDS = 15 * 60
ACCOUNT_LOGIN_FAILURE_LIMIT = 8
ACCOUNT_LOGIN_BACKOFF_SECONDS = 15 * 60
ACCOUNT_LOGIN_STATE_MAX_KEYS = int(os.environ.get("ACCOUNT_LOGIN_STATE_MAX_KEYS", "50000"))


def _abort_rate_limited(message: str, *, retry_after: int) -> None:
    abort(
        json_error(
            429,
            error="rate_limited",
            message=message,
            headers={"Retry-After": str(retry_after)},
        )
    )


def rate_limit_key(ctx: AccessContext) -> tuple[str, int]:
    if ctx.tier == "api_key" and ctx.api_key_id:
        return f"api_key:{ctx.api_key_id}", 300
    if ctx.tier == "user" and ctx.user_id:
        return f"user:{ctx.user_id}", 120
    ip = request_ip_address() or "unknown"
    return f"anon:{ip}", 60


def is_search_read_request() -> bool:
    if request.method != "GET":
        return False
    path = request.path
    return (
        path in SEARCH_READ_RATE_LIMIT_PATHS
        or path == "/v1/filter-options"
        or path.startswith("/v1/filter-options/")
    )


def apply_search_read_rate_limit(
    ctx: AccessContext, key: str, per_minute: int
) -> tuple[str, int]:
    if not is_search_read_request():
        return key, per_minute
    return f"search:{key}", SEARCH_READ_RATE_LIMITS_BY_TIER.get(ctx.tier, per_minute)


def endpoint_rate_limit_key(method: str, path: str) -> tuple[str, int] | None:
    limit = ENDPOINT_RATE_LIMITS.get((method, path))
    if limit is None:
        return None
    ip = request_ip_address() or "unknown"
    return f"endpoint:{method}:{path}:ip:{ip}", limit


def prune_rate_limit_state(now: float) -> None:
    """Drop expired windows and cap key cardinality. Caller holds rate_limit_lock."""
    global rate_limit_last_prune_at
    if (now - rate_limit_last_prune_at) < RATE_LIMIT_PRUNE_INTERVAL_SECONDS:
        return

    cutoff = now - RATE_LIMIT_WINDOW_SECONDS
    for state in (rate_limit_state, endpoint_rate_limit_state):
        stale_keys = [key for key, payload in state.items() if float(payload["ts"]) < cutoff]
        for key in stale_keys:
            _ = state.pop(key, None)
        overflow = len(state) - RATE_LIMIT_MAX_KEYS
        if overflow > 0:
            oldest_keys = [
                key
                for key, _payload in sorted(
                    state.items(),
                    key=lambda item: float(item[1]["ts"]),
                )[:overflow]
            ]
            for key in oldest_keys:
                _ = state.pop(key, None)

    rate_limit_last_prune_at = now


def _consume(state: dict[str, dict[str, float | int]], key: str, per_minute: int) -> int | None:
    """Count one request against ``key``; return Retry-After seconds when over."""
    now = time.time()
    window = RATE_LIMIT_WINDOW_SECONDS
    with rate_limit_lock:
        prune_rate_limit_state(now)
        current = state.get(key)
        if current is None or (now - float(current["ts"])) >= window:
            state[key] = {"ts": now, "count": 1}
            return None

        count = int(current["count"]) + 1
        current["count"] = count
        if count <= per_minute:
            return None
        return max(1, int(window - (now - float(current["ts"]))))


def check_rate_limit(ctx: AccessContext) -> None:
    if not request.path.startswith("/v1/"):
        return
    key, per_minute = rate_limit_key(ctx)
    key, per_minute = apply_search_read_rate_limit(ctx, key, per_minute)
    retry_after = _consume(rate_limit_state, key, per_minute)
    if retry_after is not None:
        _abort_rate_limited("Too many requests. Please retry shortly.", retry_after=retry_after)


def check_endpoint_rate_limit() -> None:
    if not request.path.startswith("/v1/"):
        return
    key_limit = endpoint_rate_limit_key(request.method, request.path)
    if key_limit is None:
        return
    key, per_minute = key_limit
    retry_after = _consume(endpoint_rate_limit_state, key, per_minute)
    if retry_after is not None:
        _abort_rate_limited("Too many requests. Please retry shortly.", retry_after=retry_after)


def account_login_key(email_normalized: str) -> str:
    return hashlib.sha256(email_normalized.encode("utf-8")).hexdigest()


def _prune_account_login_state_locked(now: float) -> None:
    cutoff = now - ACCOUNT_LOGIN_FAILURE_WINDOW_SECONDS
    stale_keys = [
        key
        for key, payload in account_login_failure_state.items()
        if float(payload["ts"]) < cutoff and float(payload.get("blocked_until", 0)) <= now
    ]
    for key in stale_keys:
        _ = account_login_failure_state.pop(key, None)
    overflow = len(account_login_failure_state) - ACCOUNT_LOGIN_STATE_MAX_KEYS
    if overflow > 0:
        oldest_keys = [
            existing_key
            for existing_key, _payload in sorted(
                account_login_failure_state.items(),
                key=lambda item: float(item[1]["ts"]),
            )[:overflow]
        ]
        for existing_key in oldest_keys:
            _ = account_login_failure_state.pop(existing_key, None)


def check_account_login_rate_limit(email_normalized: str) -> None:
    key = account_login_key(email_normalized)
    now = time.time()
    with account_login_failure_lock:
        _prune_account_login_state_locked(now)
        state = account_login_failure_state.get(key)
        if state is None:
            return
        blocked_until = float(state.get("blocked_until", 0))
        if blocked_until <= now:
            return
        retry_after = max(1, int(blocked_until - now))
    _abort_rate_limited(
        "Too many failed sign-in attempts for this account. Please retry shortly.",
        retry_after=retry_after,
    )


def record_account_login_failure(email_normalized: str) -> None:
    key = account_login_key(email_normalized)
    now = time.time()
    with account_login_failure_lock:
        state = account_login_failure_state.get(key)
        if state is None or (
            now - float(state["ts"]) >= ACCOUNT_LOGIN_FAILURE_WINDOW_SECONDS
            and float(state.get("blocked_until", 0)) <= now
        ):
            account_login_failure_state[key] = {
                "ts": now,
                "count": 1,
                "blocked_until": 0.0,
            }
            return
        new_count = int(state["count"]) + 1
        state["count"] = new_count
        state["ts"] = now
        if new_count >= ACCOUNT_LOGIN_FAILURE_LIMIT:
            state["blocked_until"] = now + ACCOUNT_LOGIN_BACKOFF_SECONDS


def record_account_login_success(email_normalized: str) -> None:
    key = account_login_key(email_normalized)
    with account_login_failure_lock:
        _ = account_login_failure_state.pop(key, None)


def reset_rate_limit_state() -> None:
    """Clear every bucket (tests)."""
    global rate_limit_last_prune_at
    with rate_limit_lock:
        rate_limit_state.clear()
        endpoint_rate_limit_state.clear()
        rate_limit_last_prune_at = 0.0
    with account_login_failure_lock:
        account_login_failure_state.clear()
