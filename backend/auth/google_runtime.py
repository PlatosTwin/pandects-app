from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request

from flask import abort, redirect, request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from werkzeug.wrappers.response import Response as WerkzeugResponse

from backend.auth.email_runtime import frontend_base_url, public_api_base_url
from backend.auth.session_runtime import (
    _GOOGLE_OAUTH_COOKIE_MAX_AGE,
    _GOOGLE_OAUTH_COOKIE_NAME,
    _GOOGLE_OAUTH_NONCE_COOKIE_NAME,
    cookie_settings,
    is_running_on_fly,
    request_ip_address,
)
from backend.core.errors import json_error as _json_error
from backend.core.runtime_utils import urlopen_read_bytes as _urlopen_read_bytes


def google_oauth_client_id() -> str:
    client_id = os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "").strip()
    if not client_id:
        abort(503, description="Google auth is not configured (missing GOOGLE_OAUTH_CLIENT_ID).")
    return client_id


def google_oauth_client_secret() -> str:
    client_secret = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", "").strip()
    if not client_secret:
        abort(
            503,
            description="Google auth is not configured (missing GOOGLE_OAUTH_CLIENT_SECRET).",
        )
    return client_secret


def google_oauth_redirect_uri() -> str:
    return f"{public_api_base_url()}/v1/auth/google/callback"


def google_oauth_flow_enabled() -> bool:
    return os.environ.get("GOOGLE_OAUTH_FLOW_ENABLED", "").strip() == "1"


def encode_frontend_hash_params(params: dict[str, str]) -> str:
    return urlencode(params, quote_via=quote)


def google_oauth_cookie_serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("AUTH_SECRET_KEY")
    if not secret:
        abort(503, description="Auth is not configured (missing AUTH_SECRET_KEY).")
    return URLSafeTimedSerializer(secret_key=secret, salt="pandects-google-oauth-cookie")


def google_oauth_pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def set_google_oauth_cookie(resp: WerkzeugResponse, payload: dict[str, str]) -> None:
    value = google_oauth_cookie_serializer().dumps(payload)
    samesite, secure = cookie_settings()
    resp.set_cookie(
        _GOOGLE_OAUTH_COOKIE_NAME,
        value,
        max_age=_GOOGLE_OAUTH_COOKIE_MAX_AGE,
        httponly=True,
        secure=secure,
        samesite=samesite.capitalize() if samesite != "none" else "None",
        path="/v1/auth/google/callback",
    )


def load_google_oauth_cookie() -> dict[str, str] | None:
    raw = request.cookies.get(_GOOGLE_OAUTH_COOKIE_NAME)
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        raw_payload_obj = cast(
            object,
            google_oauth_cookie_serializer().loads(
                raw, max_age=_GOOGLE_OAUTH_COOKIE_MAX_AGE
            ),
        )
    except (BadSignature, SignatureExpired):
        return None
    if not isinstance(raw_payload_obj, dict):
        return None
    payload = cast(dict[str, object], raw_payload_obj)
    if not all(isinstance(v, str) for v in payload.values()):
        return None
    return cast(dict[str, str], payload)


def clear_google_oauth_cookie(resp: WerkzeugResponse) -> None:
    samesite, secure = cookie_settings()
    resp.delete_cookie(
        _GOOGLE_OAUTH_COOKIE_NAME,
        path="/v1/auth/google/callback",
        secure=secure,
        samesite=samesite.capitalize() if samesite != "none" else "None",
    )


def set_google_nonce_cookie(resp: WerkzeugResponse, nonce: str) -> None:
    samesite, secure = cookie_settings()
    resp.set_cookie(
        _GOOGLE_OAUTH_NONCE_COOKIE_NAME,
        nonce,
        max_age=_GOOGLE_OAUTH_COOKIE_MAX_AGE,
        httponly=True,
        secure=secure,
        samesite=samesite.capitalize() if samesite != "none" else "None",
        path="/v1/auth/google/credential",
    )


def google_nonce_cookie_value() -> str | None:
    raw = request.cookies.get(_GOOGLE_OAUTH_NONCE_COOKIE_NAME)
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip()


def clear_google_nonce_cookie(resp: WerkzeugResponse) -> None:
    samesite, secure = cookie_settings()
    resp.delete_cookie(
        _GOOGLE_OAUTH_NONCE_COOKIE_NAME,
        path="/v1/auth/google/credential",
        secure=secure,
        samesite=samesite.capitalize() if samesite != "none" else "None",
    )


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


def frontend_google_callback_redirect(*, token: str | None, next_path: str | None, error: str | None):
    fragment: dict[str, str] = {}
    if token:
        fragment["session_token"] = token
    if next_path:
        fragment["next"] = next_path
    if error:
        fragment["error"] = error
    url = f"{frontend_base_url()}/auth/google/callback"
    if fragment:
        url = f"{url}#{encode_frontend_hash_params(fragment)}"
    resp = redirect(url)
    resp.headers["Cache-Control"] = "no-store"
    clear_google_oauth_cookie(resp)
    return resp


def google_fetch_json(url: str, *, data: dict[str, str] | None = None) -> dict[str, object]:
    headers = {"Accept": "application/json"}
    body = None
    if data is not None:
        body = urlencode(data).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = Request(url, data=body, headers=headers, method="POST" if data is not None else "GET")
    try:
        raw = _urlopen_read_bytes(req, timeout=15).decode("utf-8")
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            err_parsed_obj = cast(object, json.loads(raw))
        except json.JSONDecodeError:
            err_parsed_obj = None
        err_payload: dict[str, object] | None = (
            cast(dict[str, object], err_parsed_obj) if isinstance(err_parsed_obj, dict) else None
        )
        if isinstance(err_payload, dict):
            err_code = err_payload.get("error")
            err_desc = err_payload.get("error_description")
            if isinstance(err_code, str) and isinstance(err_desc, str):
                abort(
                    502,
                    description=f"Google auth failed ({err_code}): {err_desc}",
                )
            if isinstance(err_code, str):
                abort(502, description=f"Google auth failed ({err_code}).")
        abort(502, description=f"Google auth failed (HTTP {e.code}).")
    except URLError:
        abort(502, description="Google auth failed (network error).")
    try:
        parsed_obj = cast(object, json.loads(raw))
    except json.JSONDecodeError:
        abort(502, description="Google auth failed (invalid JSON response).")
    return cast(dict[str, object], parsed_obj) if isinstance(parsed_obj, dict) else {}
