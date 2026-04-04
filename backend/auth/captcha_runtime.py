from __future__ import annotations

import json
import os
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request

from flask import abort

from backend.auth.session_runtime import is_running_on_fly, request_ip_address
from backend.core.errors import json_error as _json_error
from backend.core.runtime_utils import urlopen_read_bytes as _urlopen_read_bytes


def turnstile_enabled() -> bool:
    raw = os.environ.get("TURNSTILE_ENABLED", "").strip()
    if raw == "1":
        return True
    if raw == "0":
        return False
    if not is_running_on_fly():
        return False
    return bool(os.environ.get("TURNSTILE_SITE_KEY", "").strip()) and bool(
        os.environ.get("TURNSTILE_SECRET_KEY", "").strip()
    )


def turnstile_site_key() -> str:
    site_key = os.environ.get("TURNSTILE_SITE_KEY", "").strip()
    if not site_key:
        abort(503, description="Captcha is not configured (missing TURNSTILE_SITE_KEY).")
    return site_key


def turnstile_secret_key() -> str:
    secret = os.environ.get("TURNSTILE_SECRET_KEY", "").strip()
    if not secret:
        abort(503, description="Captcha is not configured (missing TURNSTILE_SECRET_KEY).")
    return secret


def require_captcha_token(data: dict[str, object]) -> str:
    token = data.get("captcha_token")
    if not isinstance(token, str) or not token.strip():
        abort(
            _json_error(
                412,
                error="captcha_required",
                message="Captcha is required to create an account.",
            )
        )
    return token.strip()


def verify_turnstile_token(*, token: str) -> None:
    ip_address = request_ip_address()
    payload: dict[str, str] = {"secret": turnstile_secret_key(), "response": token}
    if ip_address:
        payload["remoteip"] = ip_address
    body = urlencode(payload).encode("utf-8")
    req = Request(
        "https://challenges.cloudflare.com/turnstile/v0/siteverify",
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded", "Accept": "application/json"},
        method="POST",
    )
    try:
        raw = _urlopen_read_bytes(req, timeout=15).decode("utf-8")
    except (HTTPError, URLError):
        abort(503, description="Captcha verification is unavailable right now.")
    try:
        result_obj = cast(object, json.loads(raw))
    except json.JSONDecodeError:
        abort(503, description="Captcha verification returned invalid data.")
    if not isinstance(result_obj, dict):
        abort(
            _json_error(
                412,
                error="captcha_failed",
                message="Captcha verification failed. Please retry.",
            )
        )
    result = cast(dict[str, object], result_obj)
    if result.get("success") is not True:
        abort(
            _json_error(
                412,
                error="captcha_failed",
                message="Captcha verification failed. Please retry.",
            )
        )
