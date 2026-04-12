from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from threading import Lock
from typing import cast
from urllib.parse import urlsplit

from flask import Flask, abort, current_app, request
from sqlalchemy import inspect, text
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.wrappers.response import Response as WerkzeugResponse

from backend.core.config import effective_auth_database_uri as _effective_auth_database_uri
from backend.core.runtime_utils import (
    request_ip_address as _request_ip_address_util,
    request_user_agent as _request_user_agent_util,
    utc_now as _utc_now,
    utc_today as _utc_today,
)
from backend.extensions import db
from backend.models import AuthSession, AuthUser


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
    if path in {
        "/v1/auth/oauth/register",
        "/v1/auth/oauth/token",
        "/v1/auth/oauth/browser-session",
    }:
        return False
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return False
    if request.cookies.get(_SESSION_COOKIE_NAME):
        return True
    return path in (
        "/v1/auth/logout",
    )


def auth_is_mocked() -> bool:
    return AUTH_MODE == "mock" and bool(current_app.debug)


AUTH_DATABASE_URI = _effective_auth_database_uri()


def ensure_auth_tables_exist(target_app: Flask) -> None:
    with target_app.app_context():
        if auth_is_mocked():
            return
        if AUTH_DATABASE_URI is None:
            db.create_all(bind_key="auth")
        ensure_auth_schema_upgrades(target_app)


def ensure_auth_schema_upgrades(target_app: Flask) -> None:
    with target_app.app_context():
        engine = db.engines.get("auth")
        if engine is None:
            return
        inspector = inspect(engine)
        if "api_keys" not in inspector.get_table_names():
            return
        api_key_columns = {column["name"] for column in inspector.get_columns("api_keys")}
        if "deleted_at" in api_key_columns:
            return
        column_type = "TIMESTAMP" if engine.dialect.name == "postgresql" else "DATETIME"
        with engine.begin() as conn:
            conn.execute(text(f"ALTER TABLE api_keys ADD COLUMN deleted_at {column_type} NULL"))


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
    deleted_at: datetime | None = None


_API_KEY_MIN_HASH_CHECKS = 5
_DUMMY_API_KEY_HASH = generate_password_hash("pdcts_dummy_api_key")


class _MockAuthStore:
    def __init__(self) -> None:
        self._lock = Lock()
        self._users_by_id: dict[str, _MockAuthUser] = {}
        self._users_by_email: dict[str, str] = {}
        self._tokens: dict[str, str] = {}
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

    def create_api_key(
        self,
        *,
        user_id: str,
        name: str | None,
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
            keys = [
                k
                for k in self._api_keys_by_id.values()
                if k.user_id == user_id and k.deleted_at is None
            ]
        return sorted(keys, key=lambda k: k.created_at, reverse=True)

    def revoke_api_key(self, *, user_id: str, key_id: str) -> bool:
        with self._lock:
            key = self._api_keys_by_id.get(key_id)
            if key is None or key.user_id != user_id:
                return False
            if key.revoked_at is None:
                key.revoked_at = _utc_now()
            return True

    def permanently_delete_api_key(self, *, user_id: str, key_id: str) -> bool:
        with self._lock:
            key = self._api_keys_by_id.get(key_id)
            if key is None or key.user_id != user_id:
                return False
            if key.deleted_at is not None:
                return True
            prefix = key.prefix
            ids = self._api_keys_by_prefix.get(prefix, [])
            if key_id in ids:
                remaining = [kid for kid in ids if kid != key_id]
                if remaining:
                    self._api_keys_by_prefix[prefix] = remaining
                else:
                    del self._api_keys_by_prefix[prefix]
            now = _utc_now()
            key.deleted_at = now
            if key.revoked_at is None:
                key.revoked_at = now
            key.key_hash = _DUMMY_API_KEY_HASH
            key.name = None
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
            if candidate.revoked_at is not None or candidate.deleted_at is not None:
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
                visible_key_ids = {
                    k.id
                    for k in self._api_keys_by_id.values()
                    if k.user_id == user_id and k.deleted_at is None
                }
                if api_key_id not in visible_key_ids:
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


def request_ip_address() -> str | None:
    return _request_ip_address_util(is_running_on_fly=is_running_on_fly())


def request_user_agent() -> str | None:
    return _request_user_agent_util()


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
