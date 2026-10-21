"""Request access resolution: who is calling, and may they call this route.

``current_access_context()`` maps a request's credentials (session cookie,
``X-API-Key``, or ``Authorization: Bearer``) to an ``AccessContext``.
``require_user`` / ``require_verified_user`` gate user-only routes.
``temporary_access_gate`` is the temporary lockdown before_request hook.
"""

from __future__ import annotations

from datetime import datetime
from typing import cast

from flask import abort, g, request
from sqlalchemy.exc import SQLAlchemyError

from backend.auth.api_keys import lookup_api_key
from backend.auth.session_runtime import (
    _SESSION_COOKIE_NAME,
    AccessContext,
    _mock_auth,
    auth_db_is_configured,
    auth_is_mocked,
    auth_lookup_failed,
    auth_session_transport,
    load_session_token,
    note_auth_lookup_failed,
)
from backend.core.config import temporary_access_lockdown_enabled
from backend.core.runtime_utils import utc_today
from backend.extensions import db
from backend.models import AuthUser


def user_id_is_verified(user_id: str) -> bool:
    if auth_is_mocked():
        user = _mock_auth.get_user(user_id)
        return user is not None and user.email_verified_at is not None
    if not auth_db_is_configured():
        return False
    try:
        user = db.session.get(AuthUser, user_id)
    except SQLAlchemyError:
        note_auth_lookup_failed()
        return False
    if user is None:
        return False
    return cast(object, user.email_verified_at) is not None


def _verified_session_user_context(token: str) -> AccessContext | None:
    user_id = load_session_token(token)
    if not user_id:
        return None
    if auth_is_mocked():
        mock_user = _mock_auth.get_user(user_id)
        if mock_user is not None and mock_user.email_verified_at is not None:
            return AccessContext(tier="user", user_id=user_id)
        return None
    if auth_db_is_configured():
        try:
            user = db.session.get(AuthUser, user_id)
        except SQLAlchemyError:
            note_auth_lookup_failed()
            user = None
        if user is not None and cast(object, user.email_verified_at) is not None:
            return AccessContext(tier="user", user_id=user_id)
    return None


def _api_key_context(api_key_raw: str) -> AccessContext | None:
    if auth_is_mocked():
        api_key = _mock_auth.lookup_api_key(api_key_raw)
        if api_key is None:
            return None
        api_key_user_id = cast(object, api_key.user_id)
        api_key_id = cast(object, api_key.id)
        if not isinstance(api_key_user_id, str) or not user_id_is_verified(api_key_user_id):
            return None
        if api_key_id is not None:
            g.api_key_last_used_candidate_id = str(api_key_id)
        return AccessContext(
            tier="api_key",
            user_id=api_key_user_id,
            api_key_id=api_key_id if isinstance(api_key_id, str) else None,
        )
    if not auth_db_is_configured():
        return None
    try:
        db_api_key = lookup_api_key(api_key_raw)
    except SQLAlchemyError:
        note_auth_lookup_failed()
        db_api_key = None
    if db_api_key is None:
        return None
    db_user_id = cast(object, db_api_key.user_id)
    db_key_id = cast(object, db_api_key.id)
    if not isinstance(db_user_id, str) or not user_id_is_verified(db_user_id):
        return None
    return AccessContext(
        tier="api_key",
        user_id=db_user_id,
        api_key_id=db_key_id if isinstance(db_key_id, str) else None,
    )


def current_access_context() -> AccessContext:
    """Resolve request auth (cookie session, X-API-Key, or bearer) into AccessContext.

    Precedence: session cookie (cookie transport only), then API key, then
    bearer session token; anything unverified falls through to anonymous.
    """
    cached = cast(object, getattr(g, "access_ctx", None))
    if isinstance(cached, AccessContext):
        return cached

    if auth_session_transport() == "cookie":
        cookie_token = request.cookies.get(_SESSION_COOKIE_NAME)
        if isinstance(cookie_token, str) and cookie_token.strip():
            ctx = _verified_session_user_context(cookie_token.strip())
            if ctx is not None:
                return ctx

    api_key_raw = request.headers.get("X-API-Key")
    api_key_raw = api_key_raw.strip() if isinstance(api_key_raw, str) else ""
    if api_key_raw:
        ctx = _api_key_context(api_key_raw)
        if ctx is not None:
            return ctx

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.removeprefix("Bearer ").strip()
        ctx = _verified_session_user_context(token)
        if ctx is not None:
            return ctx

    return AccessContext(tier="anonymous")


def _is_deleted_account_email(email: object) -> bool:
    return (
        isinstance(email, str)
        and email.startswith("deleted+")
        and email.endswith("@deleted.invalid")
    )


def require_user() -> tuple[AuthUser, AccessContext]:
    ctx = current_access_context()
    if not ctx.user_id or ctx.tier == "api_key":
        abort(401, description="Sign in required.")
    if auth_is_mocked():
        mock_user = _mock_auth.get_user(ctx.user_id)
        if mock_user is None:
            abort(401, description="Invalid session.")
        if _is_deleted_account_email(mock_user.email):
            abort(401, description="Account deleted.")
        auth_user = AuthUser()
        auth_user.id = mock_user.id
        auth_user.email = mock_user.email
        auth_user.email_verified_at = mock_user.email_verified_at
        auth_user.created_at = mock_user.created_at
        return auth_user, ctx
    try:
        user = db.session.get(AuthUser, ctx.user_id)
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")
    if user is None:
        abort(401, description="Invalid session.")
    if _is_deleted_account_email(cast(object, user.email)):
        abort(401, description="Account deleted.")
    return user, ctx


def require_verified_user() -> tuple[AuthUser, AccessContext]:
    user, ctx = require_user()
    if cast(object, user.email_verified_at) is None:
        abort(403, description="Email address not verified.")
    return user, ctx


# --- Temporary access restrictions -------------------------------------------
# New account creation is disabled and the data API is limited to accounts that
# already existed before the current day, surfaced in-app as a "check back in
# September" notice. To reopen access, remove temporary_access_gate (and its
# before_request registration in backend.app._register_request_hooks) plus the
# OAuth register guard in backend/routes/auth/__init__.py.
REGISTRATION_DISABLED_MESSAGE = (
    "New user creation is temporarily disabled. Please check back in September."
)
DATA_API_DISABLED_MESSAGE = (
    "This API is temporarily disabled for non-authenticated users. "
    "Please check back in September."
)


def access_account_predates_today(ctx: AccessContext) -> bool:
    """True only for an authenticated account created before the current UTC day."""
    if not ctx.is_authenticated or ctx.user_id is None:
        return False
    user: object
    if auth_is_mocked():
        user = _mock_auth.get_user(ctx.user_id)
    elif auth_db_is_configured():
        try:
            user = db.session.get(AuthUser, ctx.user_id)
        except SQLAlchemyError:
            note_auth_lookup_failed()
            return False
    else:
        return False
    created_at = cast(object, getattr(user, "created_at", None)) if user is not None else None
    if not isinstance(created_at, datetime):
        return False
    return created_at.date() < utc_today()


def temporary_access_gate() -> None:
    """Block new registration and restrict the data API to pre-today accounts."""
    if not temporary_access_lockdown_enabled():
        return
    if request.method == "OPTIONS":
        return
    path = request.path
    if request.method == "POST" and path == "/v1/auth/signup/password":
        abort(403, description=REGISTRATION_DISABLED_MESSAGE)
    # Only the data API is gated; /v1/auth/* (sign-in, session, captcha, etc.)
    # stays reachable so existing users can still authenticate, and
    # /v1/page-views stays open because anonymous-visitor telemetry is the
    # point of that endpoint (it exposes no data).
    if not path.startswith("/v1/") or path.startswith("/v1/auth/"):
        return
    if path == "/v1/page-views":
        return
    ctx = cast(object, getattr(g, "access_ctx", None))
    if not isinstance(ctx, AccessContext):
        ctx = current_access_context()
    if access_account_predates_today(ctx):
        return
    if auth_lookup_failed():
        # The request carried credentials we could not check; a 403 here would
        # tell a signed-in user they have no access. 503 lets the client retry.
        abort(503, description="Auth backend is unavailable right now.")
    abort(403, description=DATA_API_DISABLED_MESSAGE)


__all__ = [
    "DATA_API_DISABLED_MESSAGE",
    "REGISTRATION_DISABLED_MESSAGE",
    "access_account_predates_today",
    "current_access_context",
    "require_user",
    "require_verified_user",
    "temporary_access_gate",
    "user_id_is_verified",
]
