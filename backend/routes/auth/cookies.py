"""Signed-cookie helpers for the auth blueprint.

Five short-lived signed cookies are used by the auth flows. Each has its own
serializer (distinct itsdangerous salt) so a cookie issued for one purpose
cannot be replayed as another.
"""

from __future__ import annotations

import os

from flask import abort, request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from backend.auth.session_runtime import cookie_settings


_ZITADEL_LINK_COOKIE_NAME = "pdcts_zitadel_link"
_ZITADEL_LINK_COOKIE_MAX_AGE = 60 * 10
_ZITADEL_WEB_COOKIE_NAME = "pdcts_zitadel_web"
_ZITADEL_WEB_COOKIE_MAX_AGE = 60 * 10
_ZITADEL_PENDING_COOKIE_NAME = "pdcts_zitadel_pending"
_ZITADEL_PENDING_COOKIE_MAX_AGE = 60 * 20
_OAUTH_BROWSER_COOKIE_NAME = "pdcts_oauth_browser"
_OAUTH_BROWSER_COOKIE_MAX_AGE = 60 * 20
_OAUTH_AUTHORIZE_COOKIE_NAME = "pdcts_oauth_authorize"
_OAUTH_AUTHORIZE_COOKIE_MAX_AGE = 60 * 20


def _zitadel_link_cookie_serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("AUTH_SECRET_KEY")
    if not secret:
        abort(503, description="Auth is not configured (missing AUTH_SECRET_KEY).")
    return URLSafeTimedSerializer(secret_key=secret, salt="pandects-zitadel-link-cookie")


def _zitadel_web_cookie_serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("AUTH_SECRET_KEY")
    if not secret:
        abort(503, description="Auth is not configured (missing AUTH_SECRET_KEY).")
    return URLSafeTimedSerializer(secret_key=secret, salt="pandects-zitadel-web-cookie")


def _zitadel_pending_cookie_serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("AUTH_SECRET_KEY")
    if not secret:
        abort(503, description="Auth is not configured (missing AUTH_SECRET_KEY).")
    return URLSafeTimedSerializer(secret_key=secret, salt="pandects-zitadel-pending-cookie")


def _oauth_browser_cookie_serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("AUTH_SECRET_KEY")
    if not secret:
        abort(503, description="Auth is not configured (missing AUTH_SECRET_KEY).")
    return URLSafeTimedSerializer(secret_key=secret, salt="pandects-oauth-browser-cookie")


def _oauth_authorize_cookie_serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("AUTH_SECRET_KEY")
    if not secret:
        abort(503, description="Auth is not configured (missing AUTH_SECRET_KEY).")
    return URLSafeTimedSerializer(secret_key=secret, salt="pandects-oauth-authorize-cookie")


def _set_signed_cookie(
    *,
    name: str,
    serializer: URLSafeTimedSerializer,
    payload: dict[str, str],
    max_age: int,
    path: str,
    resp,
) -> None:
    value = serializer.dumps(payload)
    samesite, secure = cookie_settings()
    resp.set_cookie(
        name,
        value,
        max_age=max_age,
        httponly=True,
        secure=secure,
        samesite=samesite.capitalize() if samesite != "none" else "None",
        path=path,
    )


def _load_signed_cookie(
    *,
    name: str,
    serializer: URLSafeTimedSerializer,
    max_age: int,
) -> dict[str, str] | None:
    raw = request.cookies.get(name)
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        payload_obj = serializer.loads(raw, max_age=max_age)
    except (BadSignature, SignatureExpired):
        return None
    if not isinstance(payload_obj, dict):
        return None
    payload = payload_obj
    if not all(isinstance(value, str) for value in payload.values()):
        return None
    return payload


def _clear_signed_cookie(*, name: str, path: str, resp) -> None:
    samesite, secure = cookie_settings()
    resp.delete_cookie(
        name,
        path=path,
        secure=secure,
        samesite=samesite.capitalize() if samesite != "none" else "None",
    )


def _set_zitadel_link_cookie(resp, payload: dict[str, str]) -> None:
    _set_signed_cookie(
        name=_ZITADEL_LINK_COOKIE_NAME,
        serializer=_zitadel_link_cookie_serializer(),
        payload=payload,
        max_age=_ZITADEL_LINK_COOKIE_MAX_AGE,
        path="/v1/auth/external-subjects/zitadel/complete",
        resp=resp,
    )


def _load_zitadel_link_cookie() -> dict[str, str] | None:
    return _load_signed_cookie(
        name=_ZITADEL_LINK_COOKIE_NAME,
        serializer=_zitadel_link_cookie_serializer(),
        max_age=_ZITADEL_LINK_COOKIE_MAX_AGE,
    )


def _clear_zitadel_link_cookie(resp) -> None:
    _clear_signed_cookie(
        name=_ZITADEL_LINK_COOKIE_NAME,
        path="/v1/auth/external-subjects/zitadel/complete",
        resp=resp,
    )


def _set_zitadel_web_cookie(resp, payload: dict[str, str]) -> None:
    _set_signed_cookie(
        name=_ZITADEL_WEB_COOKIE_NAME,
        serializer=_zitadel_web_cookie_serializer(),
        payload=payload,
        max_age=_ZITADEL_WEB_COOKIE_MAX_AGE,
        path="/v1/auth/zitadel/complete",
        resp=resp,
    )


def _load_zitadel_web_cookie() -> dict[str, str] | None:
    return _load_signed_cookie(
        name=_ZITADEL_WEB_COOKIE_NAME,
        serializer=_zitadel_web_cookie_serializer(),
        max_age=_ZITADEL_WEB_COOKIE_MAX_AGE,
    )


def _clear_zitadel_web_cookie(resp) -> None:
    _clear_signed_cookie(
        name=_ZITADEL_WEB_COOKIE_NAME,
        path="/v1/auth/zitadel/complete",
        resp=resp,
    )


def _set_zitadel_pending_cookie(resp, payload: dict[str, str]) -> None:
    _set_signed_cookie(
        name=_ZITADEL_PENDING_COOKIE_NAME,
        serializer=_zitadel_pending_cookie_serializer(),
        payload=payload,
        max_age=_ZITADEL_PENDING_COOKIE_MAX_AGE,
        path="/v1/auth/zitadel/finalize",
        resp=resp,
    )


def _load_zitadel_pending_cookie() -> dict[str, str] | None:
    return _load_signed_cookie(
        name=_ZITADEL_PENDING_COOKIE_NAME,
        serializer=_zitadel_pending_cookie_serializer(),
        max_age=_ZITADEL_PENDING_COOKIE_MAX_AGE,
    )


def _clear_zitadel_pending_cookie(resp) -> None:
    _clear_signed_cookie(
        name=_ZITADEL_PENDING_COOKIE_NAME,
        path="/v1/auth/zitadel/finalize",
        resp=resp,
    )


def _set_oauth_browser_cookie(resp, payload: dict[str, str]) -> None:
    _set_signed_cookie(
        name=_OAUTH_BROWSER_COOKIE_NAME,
        serializer=_oauth_browser_cookie_serializer(),
        payload=payload,
        max_age=_OAUTH_BROWSER_COOKIE_MAX_AGE,
        path="/v1/auth/oauth",
        resp=resp,
    )


def _load_oauth_browser_cookie() -> dict[str, str] | None:
    return _load_signed_cookie(
        name=_OAUTH_BROWSER_COOKIE_NAME,
        serializer=_oauth_browser_cookie_serializer(),
        max_age=_OAUTH_BROWSER_COOKIE_MAX_AGE,
    )


def _clear_oauth_browser_cookie(resp) -> None:
    _clear_signed_cookie(
        name=_OAUTH_BROWSER_COOKIE_NAME,
        path="/v1/auth/oauth",
        resp=resp,
    )


def _set_oauth_authorize_cookie(resp, payload: dict[str, str]) -> None:
    _set_signed_cookie(
        name=_OAUTH_AUTHORIZE_COOKIE_NAME,
        serializer=_oauth_authorize_cookie_serializer(),
        payload=payload,
        max_age=_OAUTH_AUTHORIZE_COOKIE_MAX_AGE,
        path="/v1/auth/oauth/authorize",
        resp=resp,
    )


def _load_oauth_authorize_cookie() -> dict[str, str] | None:
    return _load_signed_cookie(
        name=_OAUTH_AUTHORIZE_COOKIE_NAME,
        serializer=_oauth_authorize_cookie_serializer(),
        max_age=_OAUTH_AUTHORIZE_COOKIE_MAX_AGE,
    )


def _clear_oauth_authorize_cookie(resp) -> None:
    _clear_signed_cookie(
        name=_OAUTH_AUTHORIZE_COOKIE_NAME,
        path="/v1/auth/oauth/authorize",
        resp=resp,
    )
