from __future__ import annotations

import secrets
import time
from collections.abc import Callable

from flask import Flask, Response, abort, g, request


def capture_request_start() -> None:
    g.request_start = time.perf_counter()


def auth_rate_limit_guard(
    *,
    current_access_context: Callable[[], object],
    csrf_required: Callable[[str], bool],
    check_rate_limit: Callable[[object], None],
    check_endpoint_rate_limit: Callable[[], None],
    csrf_cookie_name: str,
) -> None:
    ctx = current_access_context()
    g.access_ctx = ctx
    if csrf_required(request.path):
        csrf_cookie = request.cookies.get(csrf_cookie_name)
        csrf_header = request.headers.get("X-CSRF-Token")
        if (
            not isinstance(csrf_cookie, str)
            or not isinstance(csrf_header, str)
            or not csrf_cookie
            or not secrets.compare_digest(csrf_cookie, csrf_header)
        ):
            abort(403, description="Missing or invalid CSRF token.")
    check_rate_limit(ctx)
    check_endpoint_rate_limit()


def set_security_headers(
    response: Response, *, is_running_on_fly: Callable[[], bool]
) -> Response:
    _ = response.headers.setdefault("X-Content-Type-Options", "nosniff")
    _ = response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    _ = response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    origin = request.headers.get("Origin")
    if isinstance(origin, str) and origin.strip():
        existing = response.headers.get("Vary")
        if existing:
            if "Origin" not in {part.strip() for part in existing.split(",")}:
                response.headers["Vary"] = f"{existing}, Origin"
        else:
            response.headers["Vary"] = "Origin"
    if request.path.startswith("/v1/"):
        _ = response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
        )
    if is_running_on_fly():
        _ = response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=15552000; includeSubDomains",
        )
    return response


def register_request_hooks(
    target_app: Flask,
    *,
    capture_request_start: Callable[[], None],
    auth_rate_limit_guard: Callable[[], None],
    record_api_key_usage: Callable[[Response], Response],
    set_security_headers: Callable[[Response], Response],
) -> None:
    _before_req_start: object = target_app.before_request(capture_request_start)
    _before_req_guard: object = target_app.before_request(auth_rate_limit_guard)
    _after_req_usage: object = target_app.after_request(record_api_key_usage)
    _after_req_headers: object = target_app.after_request(set_security_headers)
