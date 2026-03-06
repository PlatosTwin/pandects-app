from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import secrets
import uuid
from collections.abc import Callable
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from threading import Lock
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode, urlsplit
from urllib.request import Request

from flask import Flask, abort, current_app, redirect, request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.elements import ColumnElement
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.wrappers.response import Response as WerkzeugResponse

from backend.core.config import effective_auth_database_uri as _effective_auth_database_uri
from backend.core.errors import json_error as _json_error
from backend.core.runtime_utils import (
    request_ip_address as _request_ip_address_util,
    request_user_agent as _request_user_agent_util,
    urlopen_read_bytes as _urlopen_read_bytes,
    utc_datetime_from_ms as _utc_datetime_from_ms,
    utc_now as _utc_now,
    utc_today as _utc_today,
)
from backend.extensions import db
from backend.models import (
    AuthPasswordResetToken,
    AuthSession,
    AuthSignonEvent,
    AuthUser,
    LegalAcceptance,
)


def is_email_like(value: str) -> bool:
    if not value or value.strip() != value:
        return False
    if any(ch.isspace() for ch in value):
        return False
    if value.count("@") != 1:
        return False
    local, domain = value.split("@", 1)
    if not local or not domain:
        return False
    if "." not in domain:
        return False
    if domain.startswith(".") or domain.endswith("."):
        return False
    return True


def normalize_email(email: str) -> str:
    return email.strip().lower()


def frontend_base_url() -> str:
    base = os.environ.get("PUBLIC_FRONTEND_BASE_URL", "").strip().rstrip("/")
    if base:
        return base
    if current_app.debug:
        return "http://localhost:8080"
    abort(503, description="Google auth is not configured (missing PUBLIC_FRONTEND_BASE_URL).")


def public_api_base_url() -> str:
    base = os.environ.get("PUBLIC_API_BASE_URL", "").strip().rstrip("/")
    if base:
        return base
    if current_app.debug:
        return "http://127.0.0.1:5113"
    abort(503, description="Google auth is not configured (missing PUBLIC_API_BASE_URL).")


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


def safe_next_path(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip()
    if not value or len(value) > 2048:
        return None
    if any(ord(ch) < 32 for ch in value):
        return None
    parsed = urlsplit(value)
    if parsed.scheme or parsed.netloc:
        return None
    if parsed.fragment:
        return None
    path = parsed.path
    if not path.startswith("/") or path.startswith("//"):
        return None
    if "\\" in path:
        return None
    return value


AUTH_MODE = os.environ.get("AUTH_MODE", "").strip().lower()


def is_running_on_fly() -> bool:
    return bool(os.environ.get("FLY_APP_NAME") or os.environ.get("FLY_REGION"))


_SESSION_COOKIE_NAME = "pdcts_session"
_CSRF_COOKIE_NAME = "pdcts_csrf"
_GOOGLE_OAUTH_COOKIE_NAME = "pdcts_google_oauth"
_GOOGLE_OAUTH_NONCE_COOKIE_NAME = "pdcts_google_nonce"
_GOOGLE_OAUTH_COOKIE_MAX_AGE = 60 * 10


def auth_session_transport() -> str:
    raw = os.environ.get("AUTH_SESSION_TRANSPORT", "").strip().lower()
    if raw in ("cookie", "bearer"):
        return raw
    return "cookie"


def cookie_samesite() -> str:
    raw = os.environ.get("SESSION_COOKIE_SAMESITE", "").strip().lower()
    if raw in ("lax", "strict", "none"):
        return raw
    return "none" if is_running_on_fly() else "lax"


def cookie_secure() -> bool:
    if os.environ.get("SESSION_COOKIE_SECURE", "").strip() == "0":
        return False
    if os.environ.get("SESSION_COOKIE_SECURE", "").strip() == "1":
        return True
    return is_running_on_fly()


def cookie_settings() -> tuple[str, bool]:
    samesite = cookie_samesite()
    secure = cookie_secure()
    if samesite == "none" and not secure:
        raise RuntimeError("SESSION_COOKIE_SAMESITE=None requires SESSION_COOKIE_SECURE=1.")
    return samesite, secure


def set_auth_cookies(resp: WerkzeugResponse, *, session_token: str) -> None:
    max_age = 60 * 60 * 24 * 14
    csrf_token = secrets.token_urlsafe(32)
    samesite, secure = cookie_settings()
    resp.set_cookie(
        _SESSION_COOKIE_NAME,
        session_token,
        max_age=max_age,
        httponly=True,
        secure=secure,
        samesite=samesite.capitalize() if samesite != "none" else "None",
        path="/",
    )
    resp.set_cookie(
        _CSRF_COOKIE_NAME,
        csrf_token,
        max_age=max_age,
        httponly=False,
        secure=secure,
        samesite=samesite.capitalize() if samesite != "none" else "None",
        path="/",
    )


def set_csrf_cookie(resp: WerkzeugResponse, value: str, *, max_age: int) -> None:
    samesite, secure = cookie_settings()
    resp.set_cookie(
        _CSRF_COOKIE_NAME,
        value,
        max_age=max_age,
        httponly=False,
        secure=secure,
        samesite=samesite.capitalize() if samesite != "none" else "None",
        path="/",
    )


def csrf_cookie_value() -> str | None:
    existing = request.cookies.get(_CSRF_COOKIE_NAME)
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    return None


def clear_auth_cookies(resp: WerkzeugResponse) -> None:
    samesite, secure = cookie_settings()
    resp.delete_cookie(
        _SESSION_COOKIE_NAME,
        path="/",
        secure=secure,
        samesite=samesite.capitalize() if samesite != "none" else "None",
    )
    resp.delete_cookie(
        _CSRF_COOKIE_NAME,
        path="/",
        secure=secure,
        samesite=samesite.capitalize() if samesite != "none" else "None",
    )


def csrf_required(path: str) -> bool:
    if auth_session_transport() != "cookie":
        return False
    if not path.startswith("/v1/"):
        return False
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return False
    if request.cookies.get(_SESSION_COOKIE_NAME):
        return True
    return path in (
        "/v1/auth/login",
        "/v1/auth/register",
        "/v1/auth/google/credential",
        "/v1/auth/password/forgot",
        "/v1/auth/password/reset",
        "/v1/auth/logout",
    )


def auth_is_mocked() -> bool:
    return AUTH_MODE == "mock" and bool(current_app.debug)


AUTH_DATABASE_URI = _effective_auth_database_uri()


_LEGAL_DOCS: dict[str, dict[str, str]] = {
    "tos": {
        "version": "2025-12-21",
        "sha256": "a73094a3ddd71a5c4b2bb86d53e9b2fc776c6d4fb9cd8cbb57233615d9fec7dc",
    },
    "privacy": {
        "version": "2025-12-21",
        "sha256": "889cdb5d64a2a30c2788d985e318863db92fc402beb7146c2c404147a0ef43b4",
    },
}


def ensure_auth_tables_exist(target_app: Flask) -> None:
    with target_app.app_context():
        if AUTH_DATABASE_URI is not None or auth_is_mocked():
            return
        db.create_all(bind_key="auth")


def auth_db_is_configured() -> bool:
    if auth_is_mocked():
        return True
    if AUTH_DATABASE_URI is not None:
        return True
    return not is_running_on_fly()


def require_auth_db() -> None:
    if auth_db_is_configured():
        if auth_is_mocked():
            return
        try:
            if AuthUser.query.limit(1).first() is not None:
                pass
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")
        else:
            return
    abort(
        503,
        description=(
            "Auth is not configured (missing AUTH_DATABASE_URI / DATABASE_URL). "
            "Search is available in limited mode."
        ),
    )


class AccessContext:
    def __init__(
        self,
        *,
        tier: str,
        user_id: str | None = None,
        api_key_id: str | None = None,
    ):
        self.tier = tier
        self.user_id = user_id
        self.api_key_id = api_key_id

    @property
    def is_authenticated(self) -> bool:
        return self.user_id is not None


@dataclass(frozen=True)
class _MockAuthUser:
    id: str
    email: str
    password_hash: str | None
    email_verified_at: datetime | None
    created_at: datetime


@dataclass
class _MockApiKey:
    id: str
    user_id: str
    name: str | None
    prefix: str
    key_hash: str
    created_at: datetime
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None


_API_KEY_MIN_HASH_CHECKS = 5
_DUMMY_API_KEY_HASH = generate_password_hash("pdcts_dummy_api_key")


class _MockAuthStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._users_by_id: dict[str, _MockAuthUser] = {}
        self._users_by_email: dict[str, str] = {}
        self._tokens: dict[str, str] = {}
        self._reset_tokens: dict[str, tuple[str, str, datetime, bool]] = {}
        self._api_keys_by_id: dict[str, _MockApiKey] = {}
        self._api_keys_by_prefix: dict[str, list[str]] = defaultdict(list)
        self._usage_daily: dict[tuple[str, date], int] = defaultdict(int)

    def create_user(self, *, email: str, password: str) -> _MockAuthUser:
        user_id = str(uuid.uuid4())
        user = _MockAuthUser(
            id=user_id,
            email=email,
            password_hash=generate_password_hash(password),
            email_verified_at=None,
            created_at=_utc_now(),
        )
        with self._lock:
            if email in self._users_by_email:
                abort(409, description="An account with this email already exists.")
            self._users_by_id[user_id] = user
            self._users_by_email[email] = user_id
        return user

    def authenticate(self, *, email: str, password: str) -> _MockAuthUser | None:
        with self._lock:
            user_id = self._users_by_email.get(email)
            user = self._users_by_id.get(user_id) if user_id else None
        if user is None or not user.password_hash:
            return None
        return user if check_password_hash(user.password_hash, password) else None

    def get_user_by_email(self, email: str) -> _MockAuthUser | None:
        with self._lock:
            user_id = self._users_by_email.get(email)
            return self._users_by_id.get(user_id) if user_id else None

    def get_user(self, user_id: str) -> _MockAuthUser | None:
        with self._lock:
            return self._users_by_id.get(user_id)

    def mark_email_verified(self, user_id: str) -> bool:
        with self._lock:
            user = self._users_by_id.get(user_id)
            if user is None:
                return False
            if user.email_verified_at is not None:
                return True
            self._users_by_id[user_id] = _MockAuthUser(
                id=user.id,
                email=user.email,
                password_hash=user.password_hash,
                email_verified_at=_utc_now(),
                created_at=user.created_at,
            )
            return True

    def issue_session_token(self, *, user_id: str) -> str:
        token = f"mock_{uuid.uuid4().hex}{uuid.uuid4().hex}"
        with self._lock:
            self._tokens[token] = user_id
        return token

    def load_session_token(self, token: str) -> str | None:
        with self._lock:
            return self._tokens.get(token)

    def revoke_session_token(self, token: str) -> None:
        with self._lock:
            _ = self._tokens.pop(token, None)

    def issue_password_reset_token(self, *, user_id: str, email: str) -> str:
        token = secrets.token_urlsafe(48)
        expires_at = _utc_now() + timedelta(seconds=password_reset_max_age_seconds())
        with self._lock:
            self._reset_tokens[token] = (user_id, email, expires_at, False)
        return token

    def consume_password_reset_token(self, token: str) -> tuple[str, str] | None:
        now = _utc_now()
        with self._lock:
            entry = self._reset_tokens.get(token)
            if entry is None:
                return None
            user_id, email, expires_at, used = entry
            if used or expires_at <= now:
                return None
            self._reset_tokens[token] = (user_id, email, expires_at, True)
        return user_id, email

    def set_user_password(self, *, user_id: str, password: str) -> bool:
        with self._lock:
            user = self._users_by_id.get(user_id)
            if user is None:
                return False
            self._users_by_id[user_id] = _MockAuthUser(
                id=user.id,
                email=user.email,
                password_hash=generate_password_hash(password),
                email_verified_at=user.email_verified_at or _utc_now(),
                created_at=user.created_at,
            )
        return True

    def create_api_key(
        self, *, user_id: str, name: str | None
    ) -> tuple[_MockApiKey, str]:
        token = f"pdcts_{uuid.uuid4().hex}{uuid.uuid4().hex}"
        prefix = token[: 6 + 12]
        key = _MockApiKey(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=name.strip() if isinstance(name, str) and name.strip() else None,
            prefix=prefix,
            key_hash=generate_password_hash(token),
            created_at=_utc_now(),
        )
        with self._lock:
            self._api_keys_by_id[key.id] = key
            self._api_keys_by_prefix[prefix].append(key.id)
        return key, token

    def list_api_keys(self, *, user_id: str) -> list[_MockApiKey]:
        with self._lock:
            keys = [k for k in self._api_keys_by_id.values() if k.user_id == user_id]
        return sorted(keys, key=lambda k: k.created_at, reverse=True)

    def revoke_api_key(self, *, user_id: str, key_id: str) -> bool:
        with self._lock:
            key = self._api_keys_by_id.get(key_id)
            if key is None or key.user_id != user_id:
                return False
            if key.revoked_at is None:
                key.revoked_at = _utc_now()
            return True

    def lookup_api_key(self, raw_key: str) -> _MockApiKey | None:
        if not raw_key.startswith("pdcts_"):
            return None
        prefix = raw_key[: 6 + 12]
        with self._lock:
            candidates = [
                self._api_keys_by_id[key_id]
                for key_id in self._api_keys_by_prefix.get(prefix, [])
            ]
        checks = 0
        for candidate in candidates:
            if candidate.revoked_at is not None:
                continue
            checks += 1
            if check_password_hash(candidate.key_hash, raw_key):
                candidate.last_used_at = _utc_now()
                return candidate
        for _ in range(max(0, _API_KEY_MIN_HASH_CHECKS - checks)):
            _ = check_password_hash(_DUMMY_API_KEY_HASH, raw_key)
        return None

    def record_usage(self, *, api_key_id: str) -> None:
        with self._lock:
            self._usage_daily[(api_key_id, _utc_today())] += 1

    def usage_for_user(
        self,
        *,
        user_id: str,
        period: str = "1m",
        api_key_id: str | None = None,
    ) -> tuple[list[dict[str, object]], int]:
        period_cutoffs = {
            "1w": 6,
            "1m": 29,
            "1y": 364,
            "all": None,
        }
        cutoff_days = period_cutoffs.get(period, 29)
        cutoff = _utc_today() - timedelta(days=cutoff_days) if cutoff_days is not None else None
        with self._lock:
            key_ids = [
                k.id for k in self._api_keys_by_id.values() if k.user_id == user_id
            ]
            if not key_ids:
                return [], 0
            if api_key_id is not None:
                if api_key_id not in key_ids:
                    return [], 0
                scoped_key_ids = {api_key_id}
            else:
                scoped_key_ids = set(key_ids)
            by_day: dict[str, int] = defaultdict(int)
            total = 0
            for (usage_key_id, day), count in self._usage_daily.items():
                if usage_key_id not in scoped_key_ids:
                    continue
                if cutoff is not None and day < cutoff:
                    continue
                day_str = day.isoformat()
                by_day[day_str] += int(count)
                total += int(count)
        return (
            [{"day": day, "count": by_day[day]} for day in sorted(by_day)],
            total,
        )


_mock_auth = _MockAuthStore()


def email_verification_serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("AUTH_SECRET_KEY")
    if not secret:
        abort(503, description="Auth is not configured (missing AUTH_SECRET_KEY).")
    return URLSafeTimedSerializer(secret_key=secret, salt="pandects-email-verify")


def email_verification_max_age_seconds() -> int:
    raw = os.environ.get("EMAIL_VERIFICATION_TOKEN_MAX_AGE_SECONDS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            abort(503, description="Invalid EMAIL_VERIFICATION_TOKEN_MAX_AGE_SECONDS.")
        if value <= 0:
            abort(503, description="Invalid EMAIL_VERIFICATION_TOKEN_MAX_AGE_SECONDS.")
        return value
    return 60 * 60 * 24 * 7


def password_reset_serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("AUTH_SECRET_KEY")
    if not secret:
        abort(503, description="Auth is not configured (missing AUTH_SECRET_KEY).")
    return URLSafeTimedSerializer(secret_key=secret, salt="pandects-password-reset")


def password_reset_max_age_seconds() -> int:
    raw = os.environ.get("PASSWORD_RESET_TOKEN_MAX_AGE_SECONDS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            abort(503, description="Invalid PASSWORD_RESET_TOKEN_MAX_AGE_SECONDS.")
        if value <= 0:
            abort(503, description="Invalid PASSWORD_RESET_TOKEN_MAX_AGE_SECONDS.")
        return value
    return 60 * 60


def issue_email_verification_token(*, user_id: str, email: str) -> str:
    serializer = email_verification_serializer()
    return serializer.dumps({"user_id": user_id, "email": email})


def load_email_verification_token(token: str) -> tuple[str, str] | None:
    serializer = email_verification_serializer()
    try:
        raw_payload_obj = cast(
            object, serializer.loads(token, max_age=email_verification_max_age_seconds())
        )
    except (BadSignature, SignatureExpired):
        return None
    if not isinstance(raw_payload_obj, dict):
        return None
    payload = cast(dict[str, object], raw_payload_obj)
    user_id = payload.get("user_id")
    email = payload.get("email")
    if not isinstance(user_id, str) or not isinstance(email, str):
        return None
    return user_id, email


def request_ip_address() -> str | None:
    return _request_ip_address_util(is_running_on_fly=is_running_on_fly())


def request_user_agent() -> str | None:
    return _request_user_agent_util()


def issue_password_reset_token(*, user_id: str, email: str) -> str:
    if auth_is_mocked():
        return _mock_auth.issue_password_reset_token(user_id=user_id, email=email)
    serializer = password_reset_serializer()
    reset_id = str(uuid.uuid4())
    now = _utc_now()
    expires_at = now + timedelta(seconds=password_reset_max_age_seconds())
    reset_token = AuthPasswordResetToken()
    reset_token.id = reset_id
    reset_token.user_id = user_id
    reset_token.created_at = now
    reset_token.expires_at = expires_at
    reset_token.ip_address = request_ip_address()
    reset_token.user_agent = request_user_agent()
    db.session.add(reset_token)
    db.session.commit()
    return serializer.dumps({"user_id": user_id, "email": email, "reset_id": reset_id})


def load_password_reset_token(
    token: str,
) -> tuple[str, str, AuthPasswordResetToken | None] | None:
    if auth_is_mocked():
        parsed = _mock_auth.consume_password_reset_token(token)
        if parsed is None:
            return None
        user_id, email = parsed
        return user_id, email, None
    serializer = password_reset_serializer()
    try:
        raw_payload_obj = cast(
            object, serializer.loads(token, max_age=password_reset_max_age_seconds())
        )
    except (BadSignature, SignatureExpired):
        return None
    if not isinstance(raw_payload_obj, dict):
        return None
    payload = cast(dict[str, object], raw_payload_obj)
    user_id = payload.get("user_id")
    email = payload.get("email")
    reset_id = payload.get("reset_id")
    if not isinstance(user_id, str) or not isinstance(email, str) or not isinstance(reset_id, str):
        return None
    try:
        row = cast(
            AuthPasswordResetToken | None,
            AuthPasswordResetToken.query.filter_by(id=reset_id, user_id=user_id, used_at=None)
            .first(),
        )
    except SQLAlchemyError:
        return None
    if row is None:
        return None
    expires_at = cast(object, row.expires_at)
    if not isinstance(expires_at, datetime) or expires_at <= _utc_now():
        return None
    return user_id, email, row


def resend_api_key() -> str | None:
    key = os.environ.get("RESEND_API_KEY")
    key = key.strip() if isinstance(key, str) else ""
    return key or None


def resend_from_email() -> str | None:
    sender = os.environ.get("RESEND_FROM_EMAIL")
    sender = sender.strip() if isinstance(sender, str) else ""
    if not sender:
        return None
    return sender


def resend_template_id() -> str | None:
    template_id = os.environ.get("RESEND_TEMPLATE_ID")
    template_id = template_id.strip() if isinstance(template_id, str) else ""
    return template_id or None


def resend_forgot_password_template_id() -> str | None:
    template_id = os.environ.get("RESEND_FORGOT_PASSWORD_TEMPLATE_ID")
    template_id = template_id.strip() if isinstance(template_id, str) else ""
    return template_id or "forgot-password"


def send_resend_text_email(*, to_email: str, subject: str, text: str) -> None:
    api_key = resend_api_key()
    sender = resend_from_email()
    if api_key is None or sender is None:
        if current_app.testing:
            return
        current_app.logger.warning(
            "Signup notification skipped (missing RESEND_API_KEY/RESEND_FROM_EMAIL)."
        )
        return

    if current_app.testing:
        return

    payload: dict[str, object] = {
        "from": sender,
        "to": [to_email],
        "subject": subject,
        "text": text,
        "headers": {"X-Entity-Ref-ID": uuid.uuid4().hex},
    }
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        "https://api.resend.com/emails",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "PandectsBackend/1.0 (+https://pandects.org)",
        },
        method="POST",
    )
    try:
        _ = _urlopen_read_bytes(req, timeout=15)
    except HTTPError as e:
        try:
            details = e.read().decode("utf-8", errors="replace")
        except Exception:
            details = ""
        current_app.logger.error(
            "Resend signup notification failed (HTTP %s): %s", e.code, details
        )
    except URLError as e:
        current_app.logger.error("Resend signup notification failed (network error): %s", e)


def send_resend_template_email(
    *,
    to_email: str,
    subject: str,
    variables: dict[str, object],
    template_id: str,
) -> None:
    api_key = resend_api_key()
    sender = resend_from_email()
    if api_key is None or sender is None:
        missing: list[str] = []
        if api_key is None:
            missing.append("RESEND_API_KEY")
        if sender is None:
            missing.append("RESEND_FROM_EMAIL")
        if current_app.testing:
            return
        abort(503, description=f"Email is not configured (missing {', '.join(missing)}).")

    if current_app.testing:
        return

    if not template_id:
        abort(503, description="Email is not configured (missing template id).")

    payload: dict[str, object] = {
        "from": sender,
        "to": [to_email],
        "subject": subject,
        "template": {
            "id": template_id,
            "variables": variables,
        },
        "headers": {"X-Entity-Ref-ID": uuid.uuid4().hex},
    }
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        "https://api.resend.com/emails",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "PandectsBackend/1.0 (+https://pandects.org)",
        },
        method="POST",
    )
    try:
        _ = _urlopen_read_bytes(req, timeout=15)
    except HTTPError as e:
        try:
            details = e.read().decode("utf-8", errors="replace")
        except Exception:
            details = ""
        current_app.logger.error("Resend email failed (HTTP %s): %s", e.code, details)
        abort(503, description="Email delivery failed.")
    except URLError as e:
        current_app.logger.error("Resend email failed (network error): %s", e)
        abort(503, description="Email delivery failed.")


def send_email_verification_email(*, to_email: str, token: str) -> None:
    verify_url = f"{frontend_base_url()}/auth/verify-email#token={quote(token)}"
    subject = "Verify your email for Pandects"
    year = str(_utc_now().year)
    template_id = resend_template_id()
    if template_id is None:
        abort(503, description="Email is not configured (missing RESEND_TEMPLATE_ID).")
    send_resend_template_email(
        to_email=to_email,
        subject=subject,
        variables={"VERIFY_URL": verify_url, "YEAR": year},
        template_id=template_id,
    )


def send_password_reset_email(*, to_email: str, token: str) -> None:
    reset_url = f"{frontend_base_url()}/auth/reset-password#token={quote(token)}"
    subject = "Reset your Pandects password"
    year = str(_utc_now().year)
    send_resend_template_email(
        to_email=to_email,
        subject=subject,
        variables={"RESET_URL": reset_url, "YEAR": year},
        template_id=resend_forgot_password_template_id() or "",
    )


def session_token_hash(token: str) -> str:
    secret = os.environ.get("AUTH_SECRET_KEY")
    if not isinstance(secret, str) or not secret.strip():
        abort(503, description="Auth is not configured (missing AUTH_SECRET_KEY).")
    digest = hmac.new(secret.encode("utf-8"), token.encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()


def issue_session_token(user_id: str) -> str:
    if auth_is_mocked():
        return _mock_auth.issue_session_token(user_id=user_id)
    token = secrets.token_urlsafe(48)
    now = _utc_now()
    expires_at = now + timedelta(days=14)
    ip_address = request_ip_address()
    user_agent = request_user_agent()
    session = AuthSession()
    session.user_id = user_id
    session.token_hash = session_token_hash(token)
    session.created_at = now
    session.expires_at = expires_at
    session.ip_address = ip_address
    session.user_agent = user_agent
    db.session.add(session)
    db.session.commit()
    return token


def load_session_token(token: str) -> str | None:
    if auth_is_mocked():
        return _mock_auth.load_session_token(token)
    if not token:
        return None
    try:
        session = cast(
            AuthSession | None,
            AuthSession.query.filter_by(
                token_hash=session_token_hash(token), revoked_at=None
            ).first(),
        )
    except SQLAlchemyError:
        return None
    if session is None:
        return None
    expires_at = cast(object, session.expires_at)
    if not isinstance(expires_at, datetime) or expires_at <= _utc_now():
        return None
    user_id = cast(object, session.user_id)
    return user_id if isinstance(user_id, str) else None


def revoke_session_token(token: str) -> None:
    if auth_is_mocked():
        _mock_auth.revoke_session_token(token)
        return
    if not token:
        return
    now = _utc_now()
    try:
        _ = AuthSession.query.filter_by(token_hash=session_token_hash(token)).update(
            {"revoked_at": now}, synchronize_session=False
        )
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
        return


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


def auth_enumeration_delay(*, sleeper: Callable[[], None]) -> None:
    sleeper()


def require_legal_acceptance(data: dict[str, object]) -> datetime:
    legal = data.get("legal")
    if not isinstance(legal, dict):
        abort(
            _json_error(
                412,
                error="legal_required",
                message="Legal acceptance required to create an account.",
            )
        )
    legal_dict = cast(dict[str, object], legal)
    checked_at_ms = legal_dict.get("checked_at_ms")
    checked_at = _utc_datetime_from_ms(checked_at_ms, field="legal.checked_at_ms")
    docs = legal_dict.get("docs")
    if not isinstance(docs, list):
        abort(400, description="legal.docs must be an array.")
    docs_list = cast(list[object], docs)
    normalized: list[str] = []
    for doc in docs_list:
        if not isinstance(doc, str):
            abort(400, description="legal.docs must contain strings.")
        normalized.append(doc.strip().lower())
    if set(normalized) != {"tos", "privacy", "license"}:
        abort(400, description="legal.docs must include tos, privacy, and license.")
    return checked_at


def legal_acceptance_columns() -> tuple[
    ColumnElement[str],
    ColumnElement[str],
    ColumnElement[str],
    ColumnElement[str],
]:
    return (
        cast(ColumnElement[str], LegalAcceptance.document),
        cast(ColumnElement[str], LegalAcceptance.version),
        cast(ColumnElement[str], LegalAcceptance.document_hash),
        cast(ColumnElement[str], LegalAcceptance.user_id),
    )


def user_has_current_legal_acceptances(*, user_id: str) -> bool:
    expected_rows = {(doc, meta["version"], meta["sha256"]) for doc, meta in _LEGAL_DOCS.items()}
    document_col, version_col, document_hash_col, user_id_col = legal_acceptance_columns()
    try:
        rows_raw = cast(
            list[tuple[object, object, object]],
            db.session.execute(
                select(
                    document_col,
                    version_col,
                    document_hash_col,
                ).where(
                    user_id_col == user_id,
                    document_col.in_(list(_LEGAL_DOCS.keys())),
                )
            ).all(),
        )
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")
    found: set[tuple[str, str, str]] = set()
    for doc_raw, ver_raw, hash_raw in rows_raw:
        if not isinstance(doc_raw, str) or not isinstance(ver_raw, str):
            continue
        hash_value = hash_raw if isinstance(hash_raw, str) else ""
        found.add((doc_raw, ver_raw, hash_value.strip()))
    return expected_rows.issubset(found)


def ensure_current_legal_acceptances(*, user_id: str, checked_at: datetime) -> None:
    now = _utc_now()
    ip_address = request_ip_address()
    user_agent = request_user_agent()
    document_col, version_col, document_hash_col, user_id_col = legal_acceptance_columns()
    try:
        existing_rows = cast(
            list[tuple[object, object, object]],
            db.session.execute(
                select(
                    document_col,
                    version_col,
                    document_hash_col,
                ).where(
                    user_id_col == user_id,
                    document_col.in_(list(_LEGAL_DOCS.keys())),
                )
            ).all(),
        )
        existing_set: set[tuple[str, str, str]] = set()
        for doc_raw, ver_raw, hash_raw in existing_rows:
            if not isinstance(doc_raw, str) or not isinstance(ver_raw, str):
                continue
            hash_value = hash_raw if isinstance(hash_raw, str) else ""
            existing_set.add((doc_raw, ver_raw, hash_value.strip()))
        for doc, meta in _LEGAL_DOCS.items():
            key = (doc, meta["version"], meta["sha256"])
            if key in existing_set:
                continue
            acceptance = LegalAcceptance()
            acceptance.user_id = user_id
            acceptance.document = doc
            acceptance.version = meta["version"]
            acceptance.document_hash = meta["sha256"]
            acceptance.checked_at = checked_at
            acceptance.submitted_at = now
            acceptance.ip_address = ip_address
            acceptance.user_agent = user_agent
            db.session.add(acceptance)
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")


def record_signon_event(*, user_id: str, provider: str, action: str) -> None:
    if auth_is_mocked():
        return
    ip_address = request_ip_address()
    user_agent = request_user_agent()
    event = AuthSignonEvent()
    event.user_id = user_id
    event.provider = provider
    event.action = action
    event.ip_address = ip_address
    event.user_agent = user_agent
    db.session.add(event)


__all__ = [
    "AUTH_DATABASE_URI",
    "AUTH_MODE",
    "AccessContext",
    "_CSRF_COOKIE_NAME",
    "_GOOGLE_OAUTH_COOKIE_NAME",
    "_GOOGLE_OAUTH_NONCE_COOKIE_NAME",
    "_LEGAL_DOCS",
    "_SESSION_COOKIE_NAME",
    "_mock_auth",
    "auth_db_is_configured",
    "auth_is_mocked",
    "auth_session_transport",
    "clear_auth_cookies",
    "clear_google_nonce_cookie",
    "clear_google_oauth_cookie",
    "cookie_settings",
    "csrf_cookie_value",
    "csrf_required",
    "email_verification_max_age_seconds",
    "email_verification_serializer",
    "encode_frontend_hash_params",
    "ensure_auth_tables_exist",
    "ensure_current_legal_acceptances",
    "frontend_base_url",
    "frontend_google_callback_redirect",
    "google_fetch_json",
    "google_nonce_cookie_value",
    "google_oauth_client_id",
    "google_oauth_client_secret",
    "google_oauth_cookie_serializer",
    "google_oauth_flow_enabled",
    "google_oauth_pkce_pair",
    "google_oauth_redirect_uri",
    "is_email_like",
    "is_running_on_fly",
    "issue_email_verification_token",
    "issue_password_reset_token",
    "issue_session_token",
    "legal_acceptance_columns",
    "load_email_verification_token",
    "load_google_oauth_cookie",
    "load_password_reset_token",
    "load_session_token",
    "normalize_email",
    "password_reset_max_age_seconds",
    "password_reset_serializer",
    "record_signon_event",
    "request_ip_address",
    "request_user_agent",
    "require_auth_db",
    "require_captcha_token",
    "require_legal_acceptance",
    "resend_api_key",
    "resend_forgot_password_template_id",
    "resend_from_email",
    "resend_template_id",
    "revoke_session_token",
    "safe_next_path",
    "send_email_verification_email",
    "send_password_reset_email",
    "send_resend_template_email",
    "send_resend_text_email",
    "session_token_hash",
    "set_auth_cookies",
    "set_csrf_cookie",
    "set_google_nonce_cookie",
    "set_google_oauth_cookie",
    "turnstile_enabled",
    "turnstile_secret_key",
    "turnstile_site_key",
    "user_has_current_legal_acceptances",
    "verify_turnstile_token",
]
