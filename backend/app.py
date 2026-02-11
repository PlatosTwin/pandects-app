import os
import sys
from typing import cast
from pathlib import Path
import time
import random
import hmac
import hashlib
import base64
from threading import Lock
from datetime import datetime, date, timedelta
import uuid
import re
import click
import json
import math
from flask import Flask, jsonify, request, abort, Response, g, current_app, has_app_context
from flask import redirect
from flask import make_response
from flask_cors import CORS
from flask_smorest import Blueprint
from flask.views import MethodView
from boto3.session import Session
from collections import defaultdict
from marshmallow import Schema, fields, ValidationError, EXCLUDE
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import HTTPException, InternalServerError
from dataclasses import dataclass
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import (
    create_engine,
    inspect,
    MetaData,
    Table,
    Column,
    CHAR,
    Integer,
    TEXT,
    text,
    or_,
    and_,
    case,
    func,
    cast as sql_cast,
    desc,
    asc,
)
from sqlalchemy.dialects.mysql import LONGTEXT, TINYTEXT
from dotenv import load_dotenv
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import secrets

from backend.extensions import db, api
from backend.models import (
    ApiKey,
    ApiRequestEvent,
    ApiUsageDaily,
    ApiUsageDailyIp,
    ApiUsageHourly,
    AuthPasswordResetToken,
    AuthSession,
    AuthSignonEvent,
    AuthUser,
    LegalAcceptance,
)
from backend.schemas.auth import (
    AuthApiKeySchema,
    AuthDeleteAccountSchema,
    AuthEmailSchema,
    AuthFlagInaccurateSchema,
    AuthGoogleCredentialSchema,
    AuthLoginSchema,
    AuthPasswordResetSchema,
    AuthRegisterSchema,
    AuthTokenSchema,
)
from backend.services.async_tasks import AsyncTaskRunner
from backend.services.usage import UsageBuffer, record_api_key_usage

# Load env vars from `backend/.env` regardless of the process working directory.
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
# Also allow a repo/root `.env` (or process env) to supply values without overriding.
load_dotenv()

# ── Simple in-process caching ─────────────────────────────────────────────
_FILTER_OPTIONS_TTL_SECONDS = int(os.environ.get("FILTER_OPTIONS_TTL_SECONDS", "21600"))
_filter_options_cache = {"ts": 0.0, "payload": None}
_filter_options_lock = Lock()
_TAXONOMY_TTL_SECONDS = int(os.environ.get("TAXONOMY_TTL_SECONDS", "21600"))
_taxonomy_cache = {"ts": 0.0, "payload": None}
_taxonomy_lock = Lock()
_AGREEMENTS_SUMMARY_TTL_SECONDS = int(
    os.environ.get("AGREEMENTS_SUMMARY_TTL_SECONDS", "60")
)
_agreements_summary_cache = {"ts": 0.0, "payload": None}
_agreements_summary_lock = Lock()

# ── Simple in-process rate limiting ──────────────────────────────────────
_rate_limit_lock = Lock()
_rate_limit_state: dict[str, dict[str, float | int]] = {}
_endpoint_rate_limit_state: dict[str, dict[str, float | int]] = {}

# ── Simple in-process caching for dumps ───────────────────────────────────
_DUMPS_CACHE_TTL_SECONDS = int(os.environ.get("DUMPS_CACHE_TTL_SECONDS", "300"))
_dumps_cache = {"ts": 0.0, "payload": None}
_dumps_cache_lock = Lock()
_DUMPS_MANIFEST_CACHE_TTL_SECONDS = int(
    os.environ.get("DUMPS_MANIFEST_CACHE_TTL_SECONDS", "1800")
)
_dumps_manifest_cache: dict[str, dict[str, object]] = {}
_dumps_manifest_cache_lock = Lock()

# ── API usage logging ─────────────────────────────────────────────────────
_USAGE_SAMPLE_RATE_2XX = float(os.environ.get("USAGE_SAMPLE_RATE_2XX", "0.05"))
_USAGE_SAMPLE_RATE_3XX = float(os.environ.get("USAGE_SAMPLE_RATE_3XX", "0.05"))
_LATENCY_BUCKET_BOUNDS_MS = (25, 50, 100, 250, 500, 1000, 2000, 5000, 10000)
_API_KEY_MIN_HASH_CHECKS = 5
_DUMMY_API_KEY_HASH = generate_password_hash("pdcts_dummy_api_key")
_USAGE_BUFFER_ENABLED = os.environ.get("USAGE_LOG_BUFFER_ENABLED", "1").strip() != "0"
_USAGE_BUFFER_FLUSH_SECONDS = float(os.environ.get("USAGE_LOG_BUFFER_FLUSH_SECONDS", "1"))
_USAGE_BUFFER_MAX_EVENTS = int(os.environ.get("USAGE_LOG_BUFFER_MAX_EVENTS", "200"))
_ASYNC_SIDE_EFFECTS_ENABLED = os.environ.get("ASYNC_SIDE_EFFECTS_ENABLED", "1").strip() != "0"
_ASYNC_SIDE_EFFECTS_MAX_QUEUE = int(os.environ.get("ASYNC_SIDE_EFFECTS_MAX_QUEUE", "100"))


def _usage_buffer() -> UsageBuffer | None:
    if not _USAGE_BUFFER_ENABLED:
        return None
    buffer = current_app.extensions.get("usage_buffer")
    if buffer is None:
        app_obj = cast(Flask, current_app._get_current_object())  # pyright: ignore[reportAttributeAccessIssue]
        buffer = UsageBuffer(
            app=app_obj,
            db=db,
            ApiUsageDaily=ApiUsageDaily,
            ApiUsageHourly=ApiUsageHourly,
            ApiUsageDailyIp=ApiUsageDailyIp,
            ApiRequestEvent=ApiRequestEvent,
            latency_bucket_bounds=_LATENCY_BUCKET_BOUNDS_MS,
            flush_interval_seconds=_USAGE_BUFFER_FLUSH_SECONDS,
            max_pending_events=_USAGE_BUFFER_MAX_EVENTS,
        )
        current_app.extensions["usage_buffer"] = buffer
    return buffer


def _async_task_runner() -> AsyncTaskRunner | None:
    if not _ASYNC_SIDE_EFFECTS_ENABLED:
        return None
    runner = current_app.extensions.get("async_task_runner")
    if runner is None:
        app_obj = cast(Flask, current_app._get_current_object())  # pyright: ignore[reportAttributeAccessIssue]
        runner = AsyncTaskRunner(
            app=app_obj,
            max_queue_size=_ASYNC_SIDE_EFFECTS_MAX_QUEUE,
        )
        current_app.extensions["async_task_runner"] = runner
    return runner

# ── CORS origins ──────────────────────────────────────────────────────────
_DEFAULT_CORS_ORIGINS = (
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://pandects.org",
    "https://www.pandects.org",
    "https://docs.pandects.org",
    "https://www.docs.pandects.org",
)


def _cors_origins() -> list[str]:
    raw = os.environ.get("CORS_ORIGINS", "").strip()
    if not raw:
        return list(_DEFAULT_CORS_ORIGINS)
    origins = [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]
    if "*" in origins:
        raise RuntimeError(
            "CORS_ORIGINS cannot include '*' when supports_credentials=True. "
            "Specify explicit origins instead."
        )
    return origins or list(_DEFAULT_CORS_ORIGINS)


# ── Auth mode ─────────────────────────────────────────────────────────────
AUTH_MODE = os.environ.get("AUTH_MODE", "").strip().lower()

def _is_running_on_fly() -> bool:
    return bool(os.environ.get("FLY_APP_NAME") or os.environ.get("FLY_REGION"))

# ── Session transport / cookies ───────────────────────────────────────────

_SESSION_COOKIE_NAME = "pdcts_session"
_CSRF_COOKIE_NAME = "pdcts_csrf"
_GOOGLE_OAUTH_COOKIE_NAME = "pdcts_google_oauth"
_GOOGLE_OAUTH_NONCE_COOKIE_NAME = "pdcts_google_nonce"
_GOOGLE_OAUTH_COOKIE_MAX_AGE = 60 * 10


def _auth_session_transport() -> str:
    raw = os.environ.get("AUTH_SESSION_TRANSPORT", "").strip().lower()
    if raw in ("cookie", "bearer"):
        return raw
    return "cookie"


def _cookie_samesite() -> str:
    raw = os.environ.get("SESSION_COOKIE_SAMESITE", "").strip().lower()
    if raw in ("lax", "strict", "none"):
        return raw
    return "none" if _is_running_on_fly() else "lax"


def _cookie_secure() -> bool:
    if os.environ.get("SESSION_COOKIE_SECURE", "").strip() == "0":
        return False
    if os.environ.get("SESSION_COOKIE_SECURE", "").strip() == "1":
        return True
    return _is_running_on_fly()


def _cookie_settings() -> tuple[str, bool]:
    samesite = _cookie_samesite()
    secure = _cookie_secure()
    if samesite == "none" and not secure:
        raise RuntimeError("SESSION_COOKIE_SAMESITE=None requires SESSION_COOKIE_SECURE=1.")
    return samesite, secure


def _set_auth_cookies(resp: Response, *, session_token: str) -> None:
    max_age = 60 * 60 * 24 * 14
    csrf_token = secrets.token_urlsafe(32)
    samesite, secure = _cookie_settings()
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


def _ensure_csrf_cookie(resp: Response) -> None:
    existing = request.cookies.get(_CSRF_COOKIE_NAME)
    if isinstance(existing, str) and existing.strip():
        return
    max_age = 60 * 60 * 24 * 14
    csrf_token = secrets.token_urlsafe(32)
    _set_csrf_cookie(resp, csrf_token, max_age=max_age)


def _set_csrf_cookie(resp: Response, value: str, *, max_age: int) -> None:
    samesite, secure = _cookie_settings()
    resp.set_cookie(
        _CSRF_COOKIE_NAME,
        value,
        max_age=max_age,
        httponly=False,
        secure=secure,
        samesite=samesite.capitalize() if samesite != "none" else "None",
        path="/",
    )


def _csrf_cookie_value() -> str | None:
    existing = request.cookies.get(_CSRF_COOKIE_NAME)
    if isinstance(existing, str) and existing.strip():
        return existing.strip()
    return None


def _clear_auth_cookies(resp: Response) -> None:
    samesite, secure = _cookie_settings()
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


def _csrf_required(path: str) -> bool:
    if _auth_session_transport() != "cookie":
        return False
    if not path.startswith("/v1/"):
        return False
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return False
    # Require CSRF for session-cookie authenticated requests and for endpoints that
    # establish/tear down a session (login/logout), even before a session exists.
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


def _auth_is_mocked() -> bool:
    return AUTH_MODE == "mock" and bool(current_app.debug)


# ── Auth DB configuration (separate DB; local sqlite placeholder by default) ──

def _normalize_database_uri(uri: str) -> str:
    normalized = uri.strip()
    if normalized.startswith("postgres://"):
        normalized = f"postgresql://{normalized[len('postgres://'):]}"
    if normalized.startswith("postgresql://") and "connect_timeout=" not in normalized:
        joiner = "&" if "?" in normalized else "?"
        normalized = f"{normalized}{joiner}connect_timeout=5"
    return normalized


def _effective_auth_database_uri() -> str | None:
    auth_uri = os.environ.get("AUTH_DATABASE_URI")
    auth_uri = auth_uri.strip() if isinstance(auth_uri, str) else ""
    db_url = os.environ.get("DATABASE_URL")
    db_url = db_url.strip() if isinstance(db_url, str) else ""

    raw = auth_uri or db_url
    if not raw:
        return None
    return _normalize_database_uri(raw)


AUTH_DATABASE_URI = _effective_auth_database_uri()


def _configure_auth_bind(target_app: Flask) -> None:
    if AUTH_DATABASE_URI is not None:
        target_app.config["SQLALCHEMY_BINDS"] = {"auth": AUTH_DATABASE_URI}
    else:
        target_app.config["SQLALCHEMY_BINDS"] = {
            "auth": f"sqlite:///{Path(__file__).with_name('auth_dev.sqlite')}"
        }

# ── OpenAPI / Flask-Smorest configuration ───────────────────────────────
def _configure_openapi(target_app: Flask) -> None:
    target_app.config.update(
        {
            "API_TITLE": "Pandects API",
            "API_VERSION": "v1",
            "OPENAPI_VERSION": "3.0.2",
            "API_SPEC_OPTIONS": {
                "servers": [
                    {
                        "url": "https://api.pandects.org",
                        "description": "Production API",
                    },
                    {
                        "url": "http://localhost:5113",
                        "description": "Local development API",
                    },
                ]
            },
            "OPENAPI_URL_PREFIX": "/",
            "OPENAPI_SWAGGER_UI_PATH": "/swagger-ui",
            "OPENAPI_SWAGGER_UI_URL": "https://cdn.jsdelivr.net/npm/swagger-ui-dist/",
            "MAX_CONTENT_LENGTH": None,
        }
    )
    target_app.config["MAX_CONTENT_LENGTH"] = _max_content_length()


def _max_content_length() -> int:
    raw = os.environ.get("MAX_CONTENT_LENGTH_BYTES", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError as exc:
            raise RuntimeError("MAX_CONTENT_LENGTH_BYTES must be an integer.") from exc
        if value <= 0:
            raise RuntimeError("MAX_CONTENT_LENGTH_BYTES must be positive.")
        return value
    return 1 * 1024 * 1024


_MAIN_SCHEMA_TOKEN = "__main_schema__"


def _main_db_schema_from_env() -> str:
    return os.environ.get("MAIN_DB_SCHEMA", "pdx").strip()


def _main_db_uri_from_env() -> str:
    raw = os.environ.get("MAIN_DATABASE_URI", "").strip()
    if raw:
        return raw
    db_user = os.environ["MARIADB_USER"]
    db_pass = os.environ["MARIADB_PASSWORD"]
    db_host = os.environ["MARIADB_HOST"]
    db_name = os.environ["MARIADB_DATABASE"]
    return f"mysql+pymysql://{db_user}:{db_pass}@{db_host}:3306/{db_name}"


def _schema_translate_map(schema: str | None) -> dict[str, str | None]:
    value = schema.strip() if isinstance(schema, str) else ""
    return {_MAIN_SCHEMA_TOKEN: value or None}


def _schema_prefix() -> str:
    if has_app_context():
        raw = current_app.config.get("MAIN_DB_SCHEMA", "")
        value = raw.strip() if isinstance(raw, str) else ""
    else:
        value = _main_db_schema_from_env()
    return f"{value}." if value else ""


def _configure_main_db(target_app: Flask) -> None:
    if "SQLALCHEMY_DATABASE_URI" not in target_app.config:
        target_app.config["SQLALCHEMY_DATABASE_URI"] = _main_db_uri_from_env()
    target_app.config.setdefault("MAIN_DB_SCHEMA", _main_db_schema_from_env())
    target_app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    engine_options = dict(target_app.config.get("SQLALCHEMY_ENGINE_OPTIONS", {}) or {})
    execution_options = dict(engine_options.get("execution_options", {}) or {})
    execution_options.setdefault(
        "schema_translate_map",
        _schema_translate_map(target_app.config.get("MAIN_DB_SCHEMA")),
    )
    engine_options["execution_options"] = execution_options
    target_app.config["SQLALCHEMY_ENGINE_OPTIONS"] = engine_options


def _configure_extensions(target_app: Flask) -> None:
    api.init_app(target_app)
    db.init_app(target_app)


def _configure_cors(target_app: Flask) -> None:
    CORS(
        target_app,
        resources={
            r"/v1/*": {
                "origins": _cors_origins()
            }
        },
        methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-CSRF-Token"],
        supports_credentials=True,
    )


def _configure_app(
    target_app: Flask, *, config_overrides: dict[str, object] | None = None
) -> None:
    _configure_auth_bind(target_app)
    _configure_openapi(target_app)
    if config_overrides:
        target_app.config.update(config_overrides)
    _configure_main_db(target_app)
    _configure_extensions(target_app)
    _configure_cors(target_app)



# ── JSON error responses for API routes ──────────────────────────────────


def _handle_http_exception(err: HTTPException):
    if request.path.startswith("/v1/"):
        resp = jsonify({"error": err.name, "message": err.description})
        resp.status_code = err.code or 500
        return resp
    return err


def _handle_internal_server_error(err: InternalServerError):
    if request.path.startswith("/v1/"):
        current_app.logger.exception("Unhandled API exception: %s", err)
        resp = jsonify(
            {"error": "Internal Server Error", "message": "Unexpected server error."}
        )
        resp.status_code = 500
        return resp
    return err


def _handle_sqlalchemy_error(err: SQLAlchemyError):
    if request.path.startswith("/v1/"):
        current_app.logger.exception("Database error: %s", err)
        resp = jsonify(
            {"error": "Service Unavailable", "message": "Database is unavailable."}
        )
        resp.status_code = 503
        return resp
    raise err


def _register_error_handlers(target_app: Flask) -> None:
    target_app.register_error_handler(HTTPException, _handle_http_exception)
    target_app.register_error_handler(InternalServerError, _handle_internal_server_error)
    target_app.register_error_handler(SQLAlchemyError, _handle_sqlalchemy_error)


def _json_error(
    status: int, *, error: str, message: str, headers: dict[str, str] | None = None
) -> Response:
    """Build a consistent JSON error response for API handlers."""
    resp = make_response(jsonify({"error": error, "message": message}), status)
    if headers:
        resp.headers.update(headers)
    return resp


def _status_response(status: str, *, code: int = 200) -> Response:
    """Build a standard status JSON response."""
    return make_response(jsonify({"status": status}), code)

# —— Bulk data setup ——————��———————————————————————————————————————————————
R2_BUCKET_NAME = "pandects-bulk"
R2_ENDPOINT = "https://7b5e7846d94ee35b35e21999fc4fad5b.r2.cloudflarestorage.com"
PUBLIC_DEV_BASE = "https://bulk.pandects.org"

client = None
if os.environ.get("R2_ACCESS_KEY_ID") and os.environ.get("R2_SECRET_ACCESS_KEY"):
    session = Session()
    client = session.client(
        service_name="s3",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        endpoint_url=R2_ENDPOINT,
    )

# ── Database configuration ───────────────────────────────────────────────

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


def _ensure_auth_tables_exist(target_app: Flask) -> None:
    with target_app.app_context():
        if AUTH_DATABASE_URI is not None or _auth_is_mocked():
            return
        db.create_all(bind_key="auth")

def _auth_db_is_configured() -> bool:
    if _auth_is_mocked():
        return True
    if AUTH_DATABASE_URI is not None:
        return True
    # Local development: fall back to the bundled sqlite auth DB.
    # Production (Fly): require an explicit AUTH_DATABASE_URI.
    return not _is_running_on_fly()


def _require_auth_db() -> None:
    if _auth_db_is_configured():
        if _auth_is_mocked():
            return
        try:
            AuthUser.query.limit(1).all()
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
            created_at=datetime.utcnow(),
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
                email_verified_at=datetime.utcnow(),
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
            self._tokens.pop(token, None)

    def issue_password_reset_token(self, *, user_id: str, email: str) -> str:
        token = secrets.token_urlsafe(48)
        expires_at = datetime.utcnow() + timedelta(seconds=_password_reset_max_age_seconds())
        with self._lock:
            self._reset_tokens[token] = (user_id, email, expires_at, False)
        return token

    def consume_password_reset_token(self, token: str) -> tuple[str, str] | None:
        now = datetime.utcnow()
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
                email_verified_at=user.email_verified_at or datetime.utcnow(),
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
            created_at=datetime.utcnow(),
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
                key.revoked_at = datetime.utcnow()
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
                candidate.last_used_at = datetime.utcnow()
                return candidate
        for _ in range(max(0, _API_KEY_MIN_HASH_CHECKS - checks)):
            check_password_hash(_DUMMY_API_KEY_HASH, raw_key)
        return None

    def record_usage(self, *, api_key_id: str) -> None:
        with self._lock:
            self._usage_daily[(api_key_id, _utc_today())] += 1

    def usage_for_user(self, *, user_id: str) -> tuple[list[dict[str, object]], int]:
        cutoff = _utc_today() - timedelta(days=29)
        with self._lock:
            key_ids = [
                k.id for k in self._api_keys_by_id.values() if k.user_id == user_id
            ]
            if not key_ids:
                return [], 0
            by_day: dict[str, int] = defaultdict(int)
            total = 0
            for (api_key_id, day), count in self._usage_daily.items():
                if api_key_id not in key_ids:
                    continue
                if day < cutoff:
                    continue
                day_str = day.isoformat()
                by_day[day_str] += int(count)
                total += int(count)
        return (
            [{"day": day, "count": by_day[day]} for day in sorted(by_day)],
            total,
        )


_mock_auth = _MockAuthStore()


def _auth_serializer() -> URLSafeTimedSerializer | None:
    secret = os.environ.get("AUTH_SECRET_KEY")
    if not secret:
        return None
    return URLSafeTimedSerializer(secret_key=secret, salt="pandects-auth")


def _email_verification_serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("AUTH_SECRET_KEY")
    if not secret:
        abort(503, description="Auth is not configured (missing AUTH_SECRET_KEY).")
    return URLSafeTimedSerializer(secret_key=secret, salt="pandects-email-verify")


def _email_verification_max_age_seconds() -> int:
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


def _password_reset_serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("AUTH_SECRET_KEY")
    if not secret:
        abort(503, description="Auth is not configured (missing AUTH_SECRET_KEY).")
    return URLSafeTimedSerializer(secret_key=secret, salt="pandects-password-reset")


def _password_reset_max_age_seconds() -> int:
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


def _issue_email_verification_token(*, user_id: str, email: str) -> str:
    serializer = _email_verification_serializer()
    return serializer.dumps({"user_id": user_id, "email": email})


def _load_email_verification_token(token: str) -> tuple[str, str] | None:
    serializer = _email_verification_serializer()
    try:
        payload = serializer.loads(token, max_age=_email_verification_max_age_seconds())
    except (BadSignature, SignatureExpired):
        return None
    user_id = payload.get("user_id")
    email = payload.get("email")
    if not isinstance(user_id, str) or not isinstance(email, str):
        return None
    return user_id, email


def _issue_password_reset_token(*, user_id: str, email: str) -> str:
    if _auth_is_mocked():
        return _mock_auth.issue_password_reset_token(user_id=user_id, email=email)
    serializer = _password_reset_serializer()
    reset_id = str(uuid.uuid4())
    now = datetime.utcnow()
    expires_at = now + timedelta(seconds=_password_reset_max_age_seconds())
    reset_token = AuthPasswordResetToken()
    reset_token.id = reset_id
    reset_token.user_id = user_id
    reset_token.created_at = now
    reset_token.expires_at = expires_at
    reset_token.ip_address = _request_ip_address()
    reset_token.user_agent = _request_user_agent()
    db.session.add(reset_token)
    db.session.commit()
    return serializer.dumps({"user_id": user_id, "email": email, "reset_id": reset_id})


def _load_password_reset_token(
    token: str,
) -> tuple[str, str, AuthPasswordResetToken | None] | None:
    if _auth_is_mocked():
        parsed = _mock_auth.consume_password_reset_token(token)
        if parsed is None:
            return None
        user_id, email = parsed
        return user_id, email, None
    serializer = _password_reset_serializer()
    try:
        payload = serializer.loads(token, max_age=_password_reset_max_age_seconds())
    except (BadSignature, SignatureExpired):
        return None
    user_id = payload.get("user_id")
    email = payload.get("email")
    reset_id = payload.get("reset_id")
    if not isinstance(user_id, str) or not isinstance(email, str) or not isinstance(reset_id, str):
        return None
    try:
        row = (
            AuthPasswordResetToken.query.filter_by(id=reset_id, user_id=user_id, used_at=None)
            .first()
        )
    except SQLAlchemyError:
        return None
    if row is None or row.expires_at <= datetime.utcnow():
        return None
    return user_id, email, row


def _resend_api_key() -> str | None:
    key = os.environ.get("RESEND_API_KEY")
    key = key.strip() if isinstance(key, str) else ""
    return key or None


def _resend_from_email() -> str | None:
    sender = os.environ.get("RESEND_FROM_EMAIL")
    sender = sender.strip() if isinstance(sender, str) else ""
    if not sender:
        return None
   
    return sender


def _resend_template_id() -> str | None:
    template_id = os.environ.get("RESEND_TEMPLATE_ID")
    template_id = template_id.strip() if isinstance(template_id, str) else ""
    return template_id or None


def _resend_forgot_password_template_id() -> str | None:
    template_id = os.environ.get("RESEND_FORGOT_PASSWORD_TEMPLATE_ID")
    template_id = template_id.strip() if isinstance(template_id, str) else ""
    return template_id or "forgot-password"


_SIGNUP_NOTIFICATION_EMAIL = "nmbogdan@alumni.stanford.edu"


def _send_resend_text_email(*, to_email: str, subject: str, text: str) -> None:
    api_key = _resend_api_key()
    sender = _resend_from_email()
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
        # Prevent Gmail conversation threading from collapsing the message behind "..." in mobile clients.
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
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=15) as resp:
            resp.read()
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


def _send_signup_notification_email(*, new_user_email: str) -> None:
    subject = "New Pandects signup"
    text = f"{new_user_email} just signed up as a new user on Pandects."
    runner = _async_task_runner()
    if runner is None:
        _send_resend_text_email(to_email=_SIGNUP_NOTIFICATION_EMAIL, subject=subject, text=text)
        return
    if not runner.enqueue(
        lambda: _send_resend_text_email(
            to_email=_SIGNUP_NOTIFICATION_EMAIL, subject=subject, text=text
        )
    ):
        _send_resend_text_email(to_email=_SIGNUP_NOTIFICATION_EMAIL, subject=subject, text=text)


def _send_flag_notification_email(
    *,
    user_email: str,
    submitted_at: datetime,
    source: str,
    agreement_uuid: str,
    section_uuid: str | None,
    message: str,
    request_follow_up: bool,
    issue_types: list[str],
) -> None:
    lines = [
        "Pandects issue report",
        "",
        f"User: {user_email}",
        f"Date/time (UTC): {submitted_at.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"Context: {source}",
        f"Agreement UUID: {agreement_uuid}",
    ]
    if section_uuid:
        lines.append(f"Section UUID: {section_uuid}")
    if issue_types:
        lines.extend(["", "Issue categories:", ", ".join(issue_types)])
    if message:
        lines.extend(["", "Report details:", message])
    else:
        lines.extend(["", "Report details: (none provided)"])
    lines.append(f"Request follow-up: {'Yes' if request_follow_up else 'No'}")
    text = "\n".join(lines)
    subject = "Pandects: Issue report"
    api_key = _resend_api_key()
    sender = _resend_from_email()
    if api_key is None or sender is None:
        if current_app.testing:
            return
        current_app.logger.warning(
            "Flag notification skipped (missing RESEND_API_KEY/RESEND_FROM_EMAIL)."
        )
        return
    if current_app.testing:
        return
    _send_resend_text_email(
        to_email=_SIGNUP_NOTIFICATION_EMAIL, subject=subject, text=text
    )


def _send_resend_template_email(
    *,
    to_email: str,
    subject: str,
    variables: dict[str, object],
    template_id: str,
) -> None:
    api_key = _resend_api_key()
    sender = _resend_from_email()
    if api_key is None or sender is None:
        missing = []
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
        # Prevent Gmail conversation threading from collapsing the message behind "..." in mobile clients.
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
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=15) as resp:
            resp.read()
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


def _send_email_verification_email(*, to_email: str, token: str) -> None:
    verify_url = f"{_frontend_base_url()}/auth/verify-email#token={quote(token)}"
    subject = "Verify your email for Pandects"
    year = str(datetime.utcnow().year)
    template_id = _resend_template_id()
    if template_id is None:
        abort(503, description="Email is not configured (missing RESEND_TEMPLATE_ID).")
    _send_resend_template_email(
        to_email=to_email,
        subject=subject,
        variables={"VERIFY_URL": verify_url, "YEAR": year},
        template_id=template_id,
    )


def _send_password_reset_email(*, to_email: str, token: str) -> None:
    reset_url = f"{_frontend_base_url()}/auth/reset-password#token={quote(token)}"
    subject = "Reset your Pandects password"
    year = str(datetime.utcnow().year)
    _send_resend_template_email(
        to_email=to_email,
        subject=subject,
        variables={"RESET_URL": reset_url, "YEAR": year},
        template_id=_resend_forgot_password_template_id() or "",
    )


def _session_token_hash(token: str) -> str:
    secret = os.environ.get("AUTH_SECRET_KEY")
    if not isinstance(secret, str) or not secret.strip():
        abort(503, description="Auth is not configured (missing AUTH_SECRET_KEY).")
    digest = hmac.new(secret.encode("utf-8"), token.encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()


def _issue_session_token(user_id: str) -> str:
    if _auth_is_mocked():
        return _mock_auth.issue_session_token(user_id=user_id)
    token = secrets.token_urlsafe(48)
    now = datetime.utcnow()
    expires_at = now + timedelta(days=14)
    ip_address = _request_ip_address()
    user_agent = _request_user_agent()
    session = AuthSession()
    session.user_id = user_id
    session.token_hash = _session_token_hash(token)
    session.created_at = now
    session.expires_at = expires_at
    session.ip_address = ip_address
    session.user_agent = user_agent
    db.session.add(session)
    db.session.commit()
    return token


def _load_session_token(token: str) -> str | None:
    if _auth_is_mocked():
        return _mock_auth.load_session_token(token)
    if not token:
        return None
    try:
        session = (
            AuthSession.query.filter_by(token_hash=_session_token_hash(token))
            .filter(AuthSession.revoked_at.is_(None))
            .first()
        )
    except SQLAlchemyError:
        return None
    if session is None:
        return None
    if session.expires_at <= datetime.utcnow():
        return None
    return session.user_id


def _revoke_session_token(token: str) -> None:
    if _auth_is_mocked():
        _mock_auth.revoke_session_token(token)
        return
    if not token:
        return
    now = datetime.utcnow()
    try:
        AuthSession.query.filter_by(token_hash=_session_token_hash(token)).update(
            {"revoked_at": now}, synchronize_session=False
        )
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
        return


_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def _is_email_like(value: str) -> bool:
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


def _frontend_base_url() -> str:
    base = os.environ.get("PUBLIC_FRONTEND_BASE_URL", "").strip().rstrip("/")
    if base:
        return base
    if current_app.debug:
        return "http://localhost:8080"
    abort(503, description="Google auth is not configured (missing PUBLIC_FRONTEND_BASE_URL).")


def _public_api_base_url() -> str:
    base = os.environ.get("PUBLIC_API_BASE_URL", "").strip().rstrip("/")
    if base:
        return base
    if current_app.debug:
        return "http://127.0.0.1:5113"
    abort(503, description="Google auth is not configured (missing PUBLIC_API_BASE_URL).")


def _google_oauth_client_id() -> str:
    client_id = os.environ.get("GOOGLE_OAUTH_CLIENT_ID", "").strip()
    if not client_id:
        abort(503, description="Google auth is not configured (missing GOOGLE_OAUTH_CLIENT_ID).")
    return client_id


def _google_oauth_client_secret() -> str:
    client_secret = os.environ.get("GOOGLE_OAUTH_CLIENT_SECRET", "").strip()
    if not client_secret:
        abort(
            503,
            description="Google auth is not configured (missing GOOGLE_OAUTH_CLIENT_SECRET).",
        )
    return client_secret


def _google_oauth_redirect_uri() -> str:
    return f"{_public_api_base_url()}/v1/auth/google/callback"


def _google_oauth_flow_enabled() -> bool:
    return os.environ.get("GOOGLE_OAUTH_FLOW_ENABLED", "").strip() == "1"


def _google_oauth_cookie_serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("AUTH_SECRET_KEY")
    if not secret:
        abort(503, description="Auth is not configured (missing AUTH_SECRET_KEY).")
    return URLSafeTimedSerializer(secret_key=secret, salt="pandects-google-oauth-cookie")


def _google_oauth_pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("utf-8")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def _set_google_oauth_cookie(resp: Response, payload: dict[str, str]) -> None:
    value = _google_oauth_cookie_serializer().dumps(payload)
    samesite, secure = _cookie_settings()
    resp.set_cookie(
        _GOOGLE_OAUTH_COOKIE_NAME,
        value,
        max_age=_GOOGLE_OAUTH_COOKIE_MAX_AGE,
        httponly=True,
        secure=secure,
        samesite=samesite.capitalize() if samesite != "none" else "None",
        path="/v1/auth/google/callback",
    )


def _load_google_oauth_cookie() -> dict[str, str] | None:
    raw = request.cookies.get(_GOOGLE_OAUTH_COOKIE_NAME)
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        payload = _google_oauth_cookie_serializer().loads(
            raw, max_age=_GOOGLE_OAUTH_COOKIE_MAX_AGE
        )
    except (BadSignature, SignatureExpired):
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _clear_google_oauth_cookie(resp: Response) -> None:
    samesite, secure = _cookie_settings()
    resp.delete_cookie(
        _GOOGLE_OAUTH_COOKIE_NAME,
        path="/v1/auth/google/callback",
        secure=secure,
        samesite=samesite.capitalize() if samesite != "none" else "None",
    )


def _set_google_nonce_cookie(resp: Response, nonce: str) -> None:
    samesite, secure = _cookie_settings()
    resp.set_cookie(
        _GOOGLE_OAUTH_NONCE_COOKIE_NAME,
        nonce,
        max_age=_GOOGLE_OAUTH_COOKIE_MAX_AGE,
        httponly=True,
        secure=secure,
        samesite=samesite.capitalize() if samesite != "none" else "None",
        path="/v1/auth/google/credential",
    )


def _google_nonce_cookie_value() -> str | None:
    raw = request.cookies.get(_GOOGLE_OAUTH_NONCE_COOKIE_NAME)
    if not isinstance(raw, str) or not raw.strip():
        return None
    return raw.strip()


def _clear_google_nonce_cookie(resp: Response) -> None:
    samesite, secure = _cookie_settings()
    resp.delete_cookie(
        _GOOGLE_OAUTH_NONCE_COOKIE_NAME,
        path="/v1/auth/google/credential",
        secure=secure,
        samesite=samesite.capitalize() if samesite != "none" else "None",
    )


def _turnstile_enabled() -> bool:
    raw = os.environ.get("TURNSTILE_ENABLED", "").strip()
    if raw == "1":
        return True
    if raw == "0":
        return False
    if not _is_running_on_fly():
        return False
    return bool(os.environ.get("TURNSTILE_SITE_KEY", "").strip()) and bool(
        os.environ.get("TURNSTILE_SECRET_KEY", "").strip()
    )


def _turnstile_site_key() -> str:
    site_key = os.environ.get("TURNSTILE_SITE_KEY", "").strip()
    if not site_key:
        abort(503, description="Captcha is not configured (missing TURNSTILE_SITE_KEY).")
    return site_key


def _turnstile_secret_key() -> str:
    secret = os.environ.get("TURNSTILE_SECRET_KEY", "").strip()
    if not secret:
        abort(503, description="Captcha is not configured (missing TURNSTILE_SECRET_KEY).")
    return secret


def _require_captcha_token(data: dict) -> str:
    token = data.get("captchaToken")
    if not isinstance(token, str) or not token.strip():
        abort(
            _json_error(
                412,
                error="captcha_required",
                message="Captcha is required to create an account.",
            )
        )
    return token.strip()


def _verify_turnstile_token(*, token: str) -> None:
    ip_address = _request_ip_address()
    payload: dict[str, str] = {"secret": _turnstile_secret_key(), "response": token}
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
        with urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
    except (HTTPError, URLError):
        abort(503, description="Captcha verification is unavailable right now.")
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        abort(503, description="Captcha verification returned invalid data.")
    if not isinstance(result, dict) or result.get("success") is not True:
        abort(
            _json_error(
                412,
                error="captcha_failed",
                message="Captcha verification failed. Please retry.",
            )
        )


def _encode_frontend_hash_params(params: dict[str, str]) -> str:
    return urlencode(params, quote_via=quote)


def _frontend_google_callback_redirect(*, token: str | None, next_path: str | None, error: str | None):
    fragment: dict[str, str] = {}
    if token:
        fragment["sessionToken"] = token
    if next_path:
        fragment["next"] = next_path
    if error:
        fragment["error"] = error
    url = f"{_frontend_base_url()}/auth/google/callback"
    if fragment:
        url = f"{url}#{_encode_frontend_hash_params(fragment)}"
    resp = redirect(url)
    resp.headers["Cache-Control"] = "no-store"
    _clear_google_oauth_cookie(resp)
    return resp


def _google_fetch_json(url: str, *, data: dict[str, str] | None = None) -> dict[str, object]:
    headers = {"Accept": "application/json"}
    body = None
    if data is not None:
        body = urlencode(data).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = Request(url, data=body, headers=headers, method="POST" if data is not None else "GET")
    try:
        with urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8")
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        try:
            err_payload = json.loads(raw)
        except json.JSONDecodeError:
            err_payload = None
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
    except URLError as e:
        abort(502, description="Google auth failed (network error).")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        abort(502, description="Google auth failed (invalid JSON response).")
    return payload if isinstance(payload, dict) else {}


def _google_verify_id_token(id_token: str, *, expected_nonce: str | None = None) -> str:
    try:
        import jwt
        from jwt import PyJWKClient
        from jwt.exceptions import InvalidTokenError
        from jwt.exceptions import PyJWKClientError
    except ImportError:
        abort(503, description="Google auth is unavailable (missing PyJWT dependency).")

    global _google_jwk_client
    try:
        client = _google_jwk_client
    except NameError:
        client = None

    if client is None:
        client = PyJWKClient("https://www.googleapis.com/oauth2/v3/certs")
        _google_jwk_client = client

    try:
        signing_key = client.get_signing_key_from_jwt(id_token).key
        payload = jwt.decode(
            id_token,
            signing_key,
            algorithms=["RS256"],
            audience=_google_oauth_client_id(),
            issuer=["accounts.google.com", "https://accounts.google.com"],
            leeway=60,
        )
    except PyJWKClientError:
        abort(503, description="Google auth is temporarily unavailable.")
    except InvalidTokenError:
        abort(401, description="Invalid Google credential.")

    email = payload.get("email")
    if not isinstance(email, str) or not email.strip():
        abort(401, description="Google token missing email.")

    email_verified = payload.get("email_verified")
    if email_verified is not True:
        abort(401, description="Google email is not verified.")

    if expected_nonce:
        nonce = payload.get("nonce")
        if not isinstance(nonce, str) or not nonce.strip():
            abort(401, description="Google token missing nonce.")
        if not secrets.compare_digest(nonce, expected_nonce):
            abort(401, description="Google token nonce mismatch.")

    normalized = _normalize_email(email)
    if not _is_email_like(normalized):
        abort(401, description="Invalid email address.")
    return normalized


def _safe_next_path(value: str | None) -> str | None:
    if not value:
        return None
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value.startswith("/"):
        return None
    if value.startswith("//"):
        return None
    return value


def _redact_agreement_xml(
    xml_content: str, *, focus_section_uuid: str | None, neighbor_sections: int
) -> str:
    if __package__:
        from .redaction import redact_agreement_xml
    else:
        from redaction import redact_agreement_xml

    try:
        return redact_agreement_xml(
            xml_content,
            focus_section_uuid=focus_section_uuid,
            neighbor_sections=neighbor_sections,
        )
    except ValueError as exc:
        abort(400, description=str(exc))


def _normalize_email(email: str) -> str:
    return email.strip().lower()


def _require_json() -> dict:
    """Read a JSON object body or abort with a 400 error."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        abort(400, description="Expected JSON object body.")
    return data


def _load_json(schema: Schema) -> dict:
    """Validate a JSON body against a Marshmallow schema."""
    data = _require_json()
    try:
        return schema.load(data, unknown=EXCLUDE)
    except ValidationError as exc:
        current_app.logger.debug("Validation error: %s", exc.messages)
        abort(_json_error(400, error="validation_error", message="Invalid request body."))


def _load_query(schema: Schema) -> dict:
    """Validate query args against a Marshmallow schema."""
    try:
        return schema.load(request.args, unknown=EXCLUDE)
    except ValidationError as exc:
        current_app.logger.debug("Validation error: %s", exc.messages)
        abort(_json_error(400, error="validation_error", message="Invalid query parameters."))


def _pagination_metadata(*, total_count: int, page: int, page_size: int) -> dict[str, object]:
    """Build standard pagination dict for list responses."""
    total_pages = math.ceil(total_count / page_size) if total_count else 0
    has_prev = page > 1
    has_next = page < total_pages
    prev_num = page - 1 if has_prev else None
    next_num = page + 1 if has_next else None
    return {
        "page": page,
        "pageSize": page_size,
        "totalCount": total_count,
        "totalPages": total_pages,
        "hasNext": has_next,
        "hasPrev": has_prev,
        "nextNum": next_num,
        "prevNum": prev_num,
    }


def _auth_enumeration_delay() -> None:
    time.sleep(random.uniform(0.15, 0.35))


def _utc_datetime_from_ms(value: object, *, field: str) -> datetime:
    if not isinstance(value, int):
        abort(400, description=f"{field} must be an integer (milliseconds since epoch).")
    if value <= 0:
        abort(400, description=f"{field} must be a positive integer.")
    seconds = value / 1000.0
    if not math.isfinite(seconds):
        abort(400, description=f"{field} must be a finite integer.")
    return datetime.utcfromtimestamp(seconds)


def _request_ip_address() -> str | None:
    # Only trust proxy-provided client IP headers when running behind Fly's edge.
    # Otherwise, headers like X-Forwarded-For are trivially spoofable by clients.
    if _is_running_on_fly():
        fly_client_ip = request.headers.get("Fly-Client-IP")
        if isinstance(fly_client_ip, str) and fly_client_ip.strip():
            return fly_client_ip.strip()

        forwarded_for = request.headers.get("X-Forwarded-For")
        if isinstance(forwarded_for, str) and forwarded_for.strip():
            first = forwarded_for.split(",", 1)[0].strip()
            return first or None
    remote = request.remote_addr
    return remote.strip() if isinstance(remote, str) and remote.strip() else None


def _request_user_agent() -> str | None:
    ua = request.headers.get("User-Agent")
    if not isinstance(ua, str):
        return None
    ua = ua.strip()
    if not ua:
        return None
    return ua[:512]


def _utc_today() -> date:
    return datetime.utcnow().date()




def _require_legal_acceptance(data: dict) -> datetime:
    legal = data.get("legal")
    if not isinstance(legal, dict):
        abort(
            _json_error(
                412,
                error="legal_required",
                message="Legal acceptance required to create an account.",
            )
        )
    checked_at_ms = legal.get("checkedAtMs")
    checked_at = _utc_datetime_from_ms(checked_at_ms, field="legal.checkedAtMs")
    docs = legal.get("docs")
    if not isinstance(docs, list):
        abort(400, description="legal.docs must be an array.")
    normalized = []
    for doc in docs:
        if not isinstance(doc, str):
            abort(400, description="legal.docs must contain strings.")
        normalized.append(doc.strip().lower())
    if set(normalized) != {"tos", "privacy", "license"}:
        abort(400, description="legal.docs must include tos, privacy, and license.")
    return checked_at


def _user_has_current_legal_acceptances(*, user_id: str) -> bool:
    expected_rows = {(doc, meta["version"], meta["sha256"]) for doc, meta in _LEGAL_DOCS.items()}
    try:
        rows = (
            LegalAcceptance.query.with_entities(
                LegalAcceptance.document,
                LegalAcceptance.version,
                LegalAcceptance.document_hash,
            )
            .filter_by(user_id=user_id)
            .filter(LegalAcceptance.document.in_(list(_LEGAL_DOCS.keys())))
            .all()
        )
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")
    found = {(doc, ver, (h or "").strip()) for (doc, ver, h) in rows}
    return expected_rows.issubset(found)


def _ensure_current_legal_acceptances(*, user_id: str, checked_at: datetime) -> None:
    now = datetime.utcnow()
    ip_address = _request_ip_address()
    user_agent = _request_user_agent()
    try:
        existing = (
            LegalAcceptance.query.with_entities(
                LegalAcceptance.document, LegalAcceptance.version, LegalAcceptance.document_hash
            )
            .filter_by(user_id=user_id)
            .filter(LegalAcceptance.document.in_(list(_LEGAL_DOCS.keys())))
            .all()
        )
        existing_set = {(doc, ver, (h or "").strip()) for (doc, ver, h) in existing}
        for doc, meta in _LEGAL_DOCS.items():
            key = (doc, meta["version"], meta["sha256"])
            if key in existing_set:
                continue
            db.session.add(
                LegalAcceptance(
                    user_id=user_id,
                    document=doc,
                    version=meta["version"],
                    document_hash=meta["sha256"],
                    checked_at=checked_at,
                    submitted_at=now,
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
            )
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")


def _record_signon_event(*, user_id: str, provider: str, action: str) -> None:
    if _auth_is_mocked():
        return
    ip_address = _request_ip_address()
    user_agent = _request_user_agent()
    db.session.add(
        AuthSignonEvent(
            user_id=user_id,
            provider=provider,
            action=action,
            ip_address=ip_address,
            user_agent=user_agent,
        )
    )


def _lookup_api_key(raw_key: str) -> ApiKey | None:
    if _auth_is_mocked():
        return None
    raw_key = raw_key.strip()
    if not raw_key.startswith("pdcts_"):
        return None
    prefix = raw_key[: 6 + 12]  # "pdcts_" + 12 chars
    candidates = ApiKey.query.filter_by(prefix=prefix, revoked_at=None).limit(25).all()
    checks = 0
    for candidate in candidates:
        checks += 1
        if check_password_hash(candidate.key_hash, raw_key):
            candidate.last_used_at = datetime.utcnow()
            db.session.commit()
            return candidate
    for _ in range(max(0, _API_KEY_MIN_HASH_CHECKS - checks)):
        check_password_hash(_DUMMY_API_KEY_HASH, raw_key)
    return None


def _current_access_context() -> AccessContext:
    """Resolve request auth (cookie session or X-API-Key) into AccessContext; cached on g."""
    cached = getattr(g, "access_ctx", None)
    if isinstance(cached, AccessContext):
        return cached

    if _auth_session_transport() == "cookie":
        cookie_token = request.cookies.get(_SESSION_COOKIE_NAME)
        if isinstance(cookie_token, str) and cookie_token.strip():
            user_id = _load_session_token(cookie_token.strip())
            if user_id:
                if _auth_is_mocked():
                    user = _mock_auth.get_user(user_id)
                    if user is not None and user.email_verified_at is not None:
                        return AccessContext(tier="user", user_id=user_id)
                elif _auth_db_is_configured():
                    try:
                        user = db.session.get(AuthUser, user_id)
                    except SQLAlchemyError:
                        user = None
                    if user is not None and user.email_verified_at is not None:
                        return AccessContext(tier="user", user_id=user_id)

    api_key_raw = request.headers.get("X-API-Key")
    api_key_raw = api_key_raw.strip() if isinstance(api_key_raw, str) else ""
    if api_key_raw:
        if _auth_is_mocked():
            api_key = _mock_auth.lookup_api_key(api_key_raw)
            if api_key is not None:
                if _user_id_is_verified(api_key.user_id):
                    return AccessContext(
                        tier="api_key", user_id=api_key.user_id, api_key_id=api_key.id
                    )
        elif _auth_db_is_configured():
            try:
                api_key = _lookup_api_key(api_key_raw)
            except SQLAlchemyError:
                api_key = None
            if api_key is not None:
                if _user_id_is_verified(api_key.user_id):
                    return AccessContext(
                        tier="api_key", user_id=api_key.user_id, api_key_id=api_key.id
                    )

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        token = auth_header.removeprefix("Bearer ").strip()
        user_id = _load_session_token(token)
        if user_id:
            if _auth_is_mocked():
                user = _mock_auth.get_user(user_id)
                if user is not None and user.email_verified_at is not None:
                    return AccessContext(tier="user", user_id=user_id)
            elif _auth_db_is_configured():
                try:
                    user = db.session.get(AuthUser, user_id)
                except SQLAlchemyError:
                    user = None
                if user is not None and user.email_verified_at is not None:
                    return AccessContext(tier="user", user_id=user_id)

    return AccessContext(tier="anonymous")


def _create_api_key(*, user_id: str, name: str | None) -> tuple[ApiKey, str]:
    token = f"pdcts_{uuid.uuid4().hex}{uuid.uuid4().hex}"
    prefix = token[: 6 + 12]
    key = ApiKey(
        user_id=user_id,
        name=name.strip() if isinstance(name, str) and name.strip() else None,
        prefix=prefix,
        key_hash=generate_password_hash(token),
    )
    db.session.add(key)
    db.session.commit()
    return key, token


def _require_user() -> tuple[AuthUser, AccessContext]:
    ctx = _current_access_context()
    if not ctx.user_id or ctx.tier == "api_key":
        abort(401, description="Sign in required.")
    if _auth_is_mocked():
        user = _mock_auth.get_user(ctx.user_id)
        if user is None:
            abort(401, description="Invalid session.")
        if user.email.startswith("deleted+") and user.email.endswith("@deleted.invalid"):
            abort(401, description="Account deleted.")
        return (
            AuthUser(
                id=user.id,
                email=user.email,
                email_verified_at=user.email_verified_at,
                created_at=user.created_at,
            ),
            ctx,
        )
    try:
        user = db.session.get(AuthUser, ctx.user_id)
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")
    if user is None:
        abort(401, description="Invalid session.")
    if user.email.startswith("deleted+") and user.email.endswith("@deleted.invalid"):
        abort(401, description="Account deleted.")
    return user, ctx


def _require_verified_user() -> tuple[AuthUser, AccessContext]:
    user, ctx = _require_user()
    if user.email_verified_at is None:
        abort(403, description="Email address not verified.")
    return user, ctx


def _user_id_is_verified(user_id: str) -> bool:
    if _auth_is_mocked():
        user = _mock_auth.get_user(user_id)
        return user is not None and user.email_verified_at is not None
    if not _auth_db_is_configured():
        return False
    try:
        user = db.session.get(AuthUser, user_id)
    except SQLAlchemyError:
        return False
    return user is not None and user.email_verified_at is not None


def _rate_limit_key(ctx: AccessContext) -> tuple[str, int]:
    if ctx.tier == "api_key" and ctx.api_key_id:
        return f"api_key:{ctx.api_key_id}", 120
    if ctx.tier == "user" and ctx.user_id:
        return f"user:{ctx.user_id}", 300
    ip = _request_ip_address() or "unknown"
    return f"anon:{ip}", 60


_ENDPOINT_RATE_LIMITS: dict[tuple[str, str], int] = {
    ("POST", "/v1/auth/login"): 10,
    ("POST", "/v1/auth/register"): 5,
    ("POST", "/v1/auth/email/resend"): 5,
    ("POST", "/v1/auth/password/forgot"): 5,
    ("POST", "/v1/auth/password/reset"): 10,
    ("POST", "/v1/auth/google/credential"): 10,
    ("POST", "/v1/auth/flag-inaccurate"): 10,
}


def _endpoint_rate_limit_key(method: str, path: str) -> tuple[str, int] | None:
    limit = _ENDPOINT_RATE_LIMITS.get((method, path))
    if limit is None:
        return None
    ip = _request_ip_address() or "unknown"
    return f"endpoint:{method}:{path}:ip:{ip}", limit


def _check_rate_limit(ctx: AccessContext) -> None:
    if not request.path.startswith("/v1/"):
        return

    key, per_minute = _rate_limit_key(ctx)
    now = time.time()
    window = 60.0
    with _rate_limit_lock:
        state = _rate_limit_state.get(key)
        if state is None or (now - float(state["ts"])) >= window:
            _rate_limit_state[key] = {"ts": now, "count": 1}
            return

        count = int(state["count"]) + 1
        state["count"] = count
        if count <= per_minute:
            return

        retry_after = max(1, int(window - (now - float(state["ts"]))))
    abort(
        _json_error(
            429,
            error="rate_limited",
            message="Too many requests. Please retry shortly.",
            headers={"Retry-After": str(retry_after)},
        )
    )


def _check_endpoint_rate_limit() -> None:
    if not request.path.startswith("/v1/"):
        return

    key_limit = _endpoint_rate_limit_key(request.method, request.path)
    if key_limit is None:
        return
    key, per_minute = key_limit
    now = time.time()
    window = 60.0
    with _rate_limit_lock:
        state = _endpoint_rate_limit_state.get(key)
        if state is None or (now - float(state["ts"])) >= window:
            _endpoint_rate_limit_state[key] = {"ts": now, "count": 1}
            return

        count = int(state["count"]) + 1
        state["count"] = count
        if count <= per_minute:
            return

        retry_after = max(1, int(window - (now - float(state["ts"]))))
    abort(
        _json_error(
            429,
            error="rate_limited",
            message="Too many requests. Please retry shortly.",
            headers={"Retry-After": str(retry_after)},
        )
    )


def _capture_request_start() -> None:
    g.request_start = time.perf_counter()


def _auth_rate_limit_guard():
    ctx = _current_access_context()
    g.access_ctx = ctx
    if _csrf_required(request.path):
        csrf_cookie = request.cookies.get(_CSRF_COOKIE_NAME)
        csrf_header = request.headers.get("X-CSRF-Token")
        if (
            not isinstance(csrf_cookie, str)
            or not isinstance(csrf_header, str)
            or not csrf_cookie
            or not secrets.compare_digest(csrf_cookie, csrf_header)
        ):
            abort(403, description="Missing or invalid CSRF token.")
    _check_rate_limit(ctx)
    _check_endpoint_rate_limit()


def _record_api_key_usage(response):
    ctx = _current_access_context()
    return record_api_key_usage(
        ctx=ctx,
        response=response,
        db=db,
        ApiUsageDaily=ApiUsageDaily,
        ApiUsageHourly=ApiUsageHourly,
        ApiUsageDailyIp=ApiUsageDailyIp,
        ApiRequestEvent=ApiRequestEvent,
        auth_is_mocked=_auth_is_mocked,
        mock_auth=_mock_auth,
        request_ip_address=_request_ip_address,
        request_user_agent=_request_user_agent,
        sample_rate_2xx=_USAGE_SAMPLE_RATE_2XX,
        sample_rate_3xx=_USAGE_SAMPLE_RATE_3XX,
        latency_bucket_bounds=_LATENCY_BUCKET_BOUNDS_MS,
        usage_buffer=_usage_buffer(),
    )


def _set_security_headers(response: Response):
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    origin = request.headers.get("Origin")
    if isinstance(origin, str) and origin.strip():
        existing = response.headers.get("Vary")
        if existing:
            if "Origin" not in {part.strip() for part in existing.split(",")}:
                response.headers["Vary"] = f"{existing}, Origin"
        else:
            response.headers["Vary"] = "Origin"
    if request.path.startswith("/v1/"):
        response.headers.setdefault(
            "Content-Security-Policy",
            "default-src 'none'; frame-ancestors 'none'; base-uri 'none'",
        )
    if _is_running_on_fly():
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=15552000; includeSubDomains",
        )
    return response


def _register_request_hooks(target_app: Flask) -> None:
    target_app.before_request(_capture_request_start)
    target_app.before_request(_auth_rate_limit_guard)
    target_app.after_request(_record_api_key_usage)
    target_app.after_request(_set_security_headers)

# ── Reflect existing tables via standalone engine ─────────────────────────
_SKIP_MAIN_DB_REFLECTION = os.environ.get("SKIP_MAIN_DB_REFLECTION", "").strip() == "1"
metadata = MetaData()

if not _SKIP_MAIN_DB_REFLECTION:
    engine = create_engine(
        _main_db_uri_from_env(),
        execution_options={
            "schema_translate_map": _schema_translate_map(_main_db_schema_from_env())
        },
    )

    agreements_table = Table(
        "agreements",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    pages_table = Table(
        "pages",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    xml_table = Table(
        "xml",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    taxonomy_l1_table = Table(
        "taxonomy_l1",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    taxonomy_l2_table = Table(
        "taxonomy_l2",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
    taxonomy_l3_table = Table(
        "taxonomy_l3",
        metadata,
        schema=_MAIN_SCHEMA_TOKEN,
        autoload_with=engine,
    )
else:
    # Test mode: avoid connecting to the main DB at import time.
    engine = None
    agreements_table = Table(
        "agreements",
        metadata,
        Column("agreement_uuid", CHAR(36), primary_key=True),
        Column("filing_date", TEXT, nullable=True),
        Column("prob_filing", TEXT, nullable=True),
        Column("filing_company_name", TEXT, nullable=True),
        Column("filing_company_cik", TEXT, nullable=True),
        Column("form_type", TEXT, nullable=True),
        Column("exhibit_type", TEXT, nullable=True),
        Column("target", TEXT, nullable=True),
        Column("acquirer", TEXT, nullable=True),
        Column("transaction_price_total", TEXT, nullable=True),
        Column("transaction_price_stock", TEXT, nullable=True),
        Column("transaction_price_cash", TEXT, nullable=True),
        Column("transaction_price_assets", TEXT, nullable=True),
        Column("transaction_consideration", TEXT, nullable=True),
        Column("target_type", TEXT, nullable=True),
        Column("acquirer_type", TEXT, nullable=True),
        Column("target_industry", TEXT, nullable=True),
        Column("acquirer_industry", TEXT, nullable=True),
        Column("announce_date", TEXT, nullable=True),
        Column("close_date", TEXT, nullable=True),
        Column("deal_status", TEXT, nullable=True),
        Column("attitude", TEXT, nullable=True),
        Column("deal_type", TEXT, nullable=True),
        Column("purpose", TEXT, nullable=True),
        Column("target_pe", Integer, nullable=True),
        Column("acquirer_pe", Integer, nullable=True),
        Column("verified", Integer, nullable=True),
        Column("transaction_size", Integer, nullable=True),
        Column("transaction_type", TEXT, nullable=True),
        Column("consideration_type", TEXT, nullable=True),
        Column("url", TEXT, nullable=True),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    pages_table = Table(
        "pages",
        metadata,
        Column("page_uuid", CHAR(36), primary_key=True),
        Column("agreement_uuid", CHAR(36), nullable=False),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    xml_table = Table(
        "xml",
        metadata,
        Column("agreement_uuid", CHAR(36), primary_key=True),
        Column("xml", TEXT, nullable=True),
        Column("version", Integer, nullable=True),
        Column("status", TEXT, nullable=True),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    taxonomy_l1_table = Table(
        "taxonomy_l1",
        metadata,
        Column("standard_id", TEXT, primary_key=True),
        Column("label", TEXT, nullable=True),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    taxonomy_l2_table = Table(
        "taxonomy_l2",
        metadata,
        Column("standard_id", TEXT, primary_key=True),
        Column("label", TEXT, nullable=True),
        Column("parent_id", TEXT, nullable=True),
        schema=_MAIN_SCHEMA_TOKEN,
    )
    taxonomy_l3_table = Table(
        "taxonomy_l3",
        metadata,
        Column("standard_id", TEXT, primary_key=True),
        Column("label", TEXT, nullable=True),
        Column("parent_id", TEXT, nullable=True),
        schema=_MAIN_SCHEMA_TOKEN,
    )

_SECTION_TEXT_TYPE = LONGTEXT if not _SKIP_MAIN_DB_REFLECTION else TEXT
_SECTION_ID_TYPE = TINYTEXT if not _SKIP_MAIN_DB_REFLECTION else TEXT

sections_table = Table(
    "sections",
    metadata,
    Column("agreement_uuid", CHAR(36), nullable=False),
    Column("section_uuid", CHAR(36), primary_key=True),
    Column("article_title", TEXT, nullable=False),
    Column("section_title", TEXT, nullable=False),
    Column("xml_content", _SECTION_TEXT_TYPE, nullable=False),
    Column("article_standard_id", _SECTION_ID_TYPE, nullable=False),
    Column("section_standard_id", _SECTION_ID_TYPE, nullable=False),
    Column("section_standard_id_gold_label", _SECTION_ID_TYPE, nullable=True),
    schema=_MAIN_SCHEMA_TOKEN,
)


# ── SQLAlchemy models mapping ───────────────────────────────────────
class Sections(db.Model):
    __table__ = sections_table


class Agreements(db.Model):
    __table__ = agreements_table


class XML(db.Model):
    __table__ = xml_table


class Pages(db.Model):
    __table__ = pages_table


class TaxonomyL1(db.Model):
    __table__ = taxonomy_l1_table


class TaxonomyL2(db.Model):
    __table__ = taxonomy_l2_table


class TaxonomyL3(db.Model):
    __table__ = taxonomy_l3_table


def _coalesced_section_standard_ids():
    return func.coalesce(
        Sections.section_standard_id_gold_label,
        Sections.section_standard_id,
    )


def _agreement_year_expr():
    bind = db.session.get_bind()
    if bind is not None and bind.dialect.name == "sqlite":
        return sql_cast(func.substr(Agreements.filing_date, 1, 4), Integer)
    return func.year(func.str_to_date(Agreements.filing_date, "%Y-%m-%d"))


def _xml_eligible_latest_subquery():
    """
    Returns subquery (agreement_uuid, version) for XML rows that are
    latest per agreement and have pdx.xml.status IS NULL OR status = 'verified'.
    """
    status_ok = or_(XML.status.is_(None), XML.status == "verified")
    max_version_sq = (
        db.session.query(
            XML.agreement_uuid,
            func.max(XML.version).label("max_version"),
        )
        .filter(status_ok)
        .group_by(XML.agreement_uuid)
        .subquery()
    )
    return (
        db.session.query(
            XML.agreement_uuid.label("agreement_uuid"),
            XML.version.label("version"),
        )
        .join(
            max_version_sq,
            and_(
                XML.agreement_uuid == max_version_sq.c.agreement_uuid,
                XML.version == max_version_sq.c.max_version,
            ),
        )
        .filter(status_ok)
        .subquery()
    )


def _xml_agreements_subquery():
    """
    Returns subquery of agreement_uuids that have eligible XML
    (latest version, pdx.xml.status IS NULL OR status = 'verified').
    """
    eligible = _xml_eligible_latest_subquery()
    return (
        db.session.query(eligible.c.agreement_uuid.distinct().label("agreement_uuid"))
        .select_from(eligible)
        .subquery()
    )


def _is_agreement_section_eligible(agreement_uuid: str, section_uuid: str | None) -> bool:
    """True iff agreement has pdx.xml status null/verified and (if section_uuid) section exists."""
    xml_agreements = _xml_agreements_subquery()
    if not section_uuid:
        return (
            db.session.query(xml_agreements.c.agreement_uuid)
            .filter(xml_agreements.c.agreement_uuid == agreement_uuid)
            .first()
            is not None
        )
    return (
        db.session.query(Sections.section_uuid)
        .join(
            xml_agreements,
            Sections.agreement_uuid == xml_agreements.c.agreement_uuid,
        )
        .filter(
            Sections.agreement_uuid == agreement_uuid,
            Sections.section_uuid == section_uuid,
        )
        .first()
        is not None
    )


def _parse_section_standard_ids(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        if not all(isinstance(item, str) for item in raw):
            raise ValueError("section_standard_id must contain string values.")
        return raw
    if isinstance(raw, str):
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or not all(
            isinstance(item, str) for item in parsed
        ):
            raise ValueError("section_standard_id must be a JSON list of strings.")
        return parsed
    raise TypeError(f"Unsupported section_standard_id type: {type(raw)!r}")


def _expand_taxonomy_standard_ids(standard_ids: list[str]) -> list[str]:
    if not standard_ids:
        return []

    standard_ids_set = {value for value in standard_ids if value}
    if not standard_ids_set:
        return []

    l1_ids = {
        row.standard_id
        for row in db.session.query(TaxonomyL1.standard_id)
        .filter(TaxonomyL1.standard_id.in_(standard_ids_set))
        .all()
    }
    l2_ids = {
        row.standard_id
        for row in db.session.query(TaxonomyL2.standard_id)
        .filter(TaxonomyL2.standard_id.in_(standard_ids_set))
        .all()
    }
    l3_ids = {
        row.standard_id
        for row in db.session.query(TaxonomyL3.standard_id)
        .filter(TaxonomyL3.standard_id.in_(standard_ids_set))
        .all()
    }

    expanded_l2_ids = set()
    expanded_l3_ids = set()
    if l1_ids:
        expanded_l2_ids.update(
            row.standard_id
            for row in db.session.query(TaxonomyL2.standard_id)
            .filter(TaxonomyL2.parent_id.in_(l1_ids))
            .all()
        )
        expanded_l3_ids.update(
            row.standard_id
            for row in db.session.query(TaxonomyL3.standard_id)
            .join(TaxonomyL2, TaxonomyL3.parent_id == TaxonomyL2.standard_id)
            .filter(TaxonomyL2.parent_id.in_(l1_ids))
            .all()
        )
    if l2_ids:
        expanded_l3_ids.update(
            row.standard_id
            for row in db.session.query(TaxonomyL3.standard_id)
            .filter(TaxonomyL3.parent_id.in_(l2_ids))
            .all()
        )

    return list(
        standard_ids_set | l1_ids | l2_ids | l3_ids | expanded_l2_ids | expanded_l3_ids
    )


# ── Define search blueprint and schemas ──────────────────────────────────
search_blp = Blueprint(
    "search",
    "search",
    url_prefix="/v1/search",
    description="Search merger agreement sections",
)

dumps_blp = Blueprint(
    "dumps",
    "dumps",
    url_prefix="/v1/dumps",
    description="Access metadata about bulk data on Cloudflare",
)

taxonomy_blp = Blueprint(
    "taxonomy",
    "taxonomy",
    url_prefix="/v1/taxonomy",
    description="Access the Pandects agreement taxonomy",
)

agreements_blp = Blueprint(
    "agreements",
    "agreements",
    url_prefix="/v1/agreements",
    description="Retrieve full text for a given agreement",
)

sections_blp = Blueprint(
    "sections",
    "sections",
    url_prefix="/v1/sections",
    description="Retrieve full text for a given section",
)


class SearchArgsSchema(Schema):
    year = fields.List(
        fields.Int(),
        load_default=[],
        metadata={
            "description": "Agreement year filter. Repeat query key for multiple values.",
            "example": [2022, 2023],
        },
    )
    target = fields.List(
        fields.Str(),
        load_default=[],
        metadata={
            "description": "Exact target company names to include.",
            "example": ["Slack Technologies, Inc."],
        },
    )
    acquirer = fields.List(
        fields.Str(),
        load_default=[],
        metadata={
            "description": "Exact acquirer company names to include.",
            "example": ["salesforce.com, inc."],
        },
    )
    standardId = fields.List(
        fields.Str(),
        load_default=[],
        metadata={
            "description": (
                "Clause taxonomy standard IDs. Parent IDs expand to include descendant "
                "taxonomy nodes."
            ),
            "example": ["1.1", "1.2.3"],
        },
    )
    # Transaction price filters
    transactionPriceTotal = fields.List(
        fields.Str(),
        load_default=[],
        metadata={
            "description": (
                "Reserved for future filtering by total transaction price. Accepted but "
                "currently ignored by the query engine."
            )
        },
    )
    transactionPriceStock = fields.List(
        fields.Str(),
        load_default=[],
        metadata={
            "description": (
                "Reserved for future filtering by stock consideration value. Accepted but "
                "currently ignored by the query engine."
            )
        },
    )
    transactionPriceCash = fields.List(
        fields.Str(),
        load_default=[],
        metadata={
            "description": (
                "Reserved for future filtering by cash consideration value. Accepted but "
                "currently ignored by the query engine."
            )
        },
    )
    transactionPriceAssets = fields.List(
        fields.Str(),
        load_default=[],
        metadata={
            "description": (
                "Reserved for future filtering by asset consideration value. Accepted but "
                "currently ignored by the query engine."
            )
        },
    )
    # New filters from DB definition
    transactionConsideration = fields.List(
        fields.Str(),
        load_default=[],
        metadata={
            "description": "Transaction consideration category values from the agreement record."
        },
    )
    targetType = fields.List(
        fields.Str(),
        load_default=[],
        metadata={
            "description": "Target company type values (for example `public` or `private`)."
        },
    )
    acquirerType = fields.List(
        fields.Str(),
        load_default=[],
        metadata={
            "description": "Acquirer company type values (for example `public` or `private`)."
        },
    )
    targetIndustry = fields.List(
        fields.Str(),
        load_default=[],
        metadata={"description": "Target industry values from normalized agreement metadata."},
    )
    acquirerIndustry = fields.List(
        fields.Str(),
        load_default=[],
        metadata={"description": "Acquirer industry values from normalized agreement metadata."},
    )
    dealStatus = fields.List(
        fields.Str(),
        load_default=[],
        metadata={"description": "Deal status values (for example announced, completed, terminated)."},
    )
    attitude = fields.List(
        fields.Str(),
        load_default=[],
        metadata={"description": "Deal attitude values (for example friendly or hostile)."},
    )
    dealType = fields.List(
        fields.Str(),
        load_default=[],
        metadata={"description": "Deal type values from agreement metadata."},
    )
    purpose = fields.List(
        fields.Str(),
        load_default=[],
        metadata={"description": "Strategic purpose values from agreement metadata."},
    )
    targetPe = fields.List(
        fields.Str(),
        load_default=[],
        metadata={
            "description": "Target private-equity backed filter. Supported values: `true`, `false`."
        },
    )
    acquirerPe = fields.List(
        fields.Str(),
        load_default=[],
        metadata={
            "description": "Acquirer private-equity backed filter. Supported values: `true`, `false`."
        },
    )
    # Legacy filters (kept for backward compatibility, but deprecated)
    transactionSize = fields.List(
        fields.Str(),
        load_default=[],
        metadata={
            "description": (
                "Legacy transaction size bucket labels (for example `100M - 250M`, `20B+`)."
            ),
            "deprecated": True,
        },
    )
    transactionType = fields.List(
        fields.Str(),
        load_default=[],
        metadata={
            "description": "Legacy transaction type labels (`Strategic`, `Financial`).",
            "deprecated": True,
        },
    )
    considerationType = fields.List(
        fields.Str(),
        load_default=[],
        metadata={
            "description": "Legacy consideration labels (`All stock`, `All cash`, `Mixed`).",
            "deprecated": True,
        },
    )
    # Text filters
    agreementUuid = fields.Str(
        load_default=None,
        allow_none=True,
        metadata={"description": "Filter to one agreement UUID."},
    )
    sectionUuid = fields.Str(
        load_default=None,
        allow_none=True,
        metadata={"description": "Filter to one section UUID."},
    )
    # Sort parameters
    sortBy = fields.Str(
        load_default="year",
        validate=lambda x: x in ["year", "target", "acquirer"],
        metadata={"description": "Sort key. One of: `year`, `target`, `acquirer`."},
    )
    sortDirection = fields.Str(
        load_default="desc",
        validate=lambda x: x in ["asc", "desc"],
        metadata={"description": "Sort direction. One of: `asc`, `desc`."},
    )
    page = fields.Int(
        load_default=1,
        metadata={"description": "1-based page number.", "example": 1},
    )
    pageSize = fields.Int(
        load_default=25,
        metadata={
            "description": (
                "Page size. Effective max is 10 for unauthenticated callers and 100 for "
                "authenticated callers."
            ),
            "example": 25,
        },
    )


class SectionItemSchema(Schema):
    id = fields.Str(metadata={"description": "Canonical identifier for this section result."})
    agreementUuid = fields.Str(metadata={"description": "Agreement UUID for the section."})
    sectionUuid = fields.Str(metadata={"description": "Section UUID."})
    standardId = fields.List(
        fields.Str(),
        metadata={"description": "Matched taxonomy standard IDs for this section."},
    )
    xml = fields.Str(
        metadata={"description": "Section XML content. Full content requires authentication."}
    )
    articleTitle = fields.Str(metadata={"description": "Article heading that contains the section."})
    sectionTitle = fields.Str(metadata={"description": "Section heading text."})
    acquirer = fields.Str(metadata={"description": "Acquirer company name."})
    target = fields.Str(metadata={"description": "Target company name."})
    year = fields.Int(metadata={"description": "Agreement year."})
    verified = fields.Bool(metadata={"description": "Whether this result is from verified content."})


class AccessInfoSchema(Schema):
    tier = fields.Str(
        required=True,
        metadata={"description": "Access tier used to shape response limits and content visibility."},
    )
    message = fields.Str(
        required=False,
        allow_none=True,
        metadata={"description": "Optional human-readable access message for the caller."},
    )


class SearchResponseSchema(Schema):
    results = fields.List(
        fields.Nested(SectionItemSchema),
        metadata={"description": "Page of search results."},
    )
    access = fields.Nested(
        AccessInfoSchema,
        metadata={"description": "Access context applied to this response."},
    )
    page = fields.Int(metadata={"description": "Current 1-based page number."})
    pageSize = fields.Int(metadata={"description": "Effective page size for this response."})
    totalCount = fields.Int(metadata={"description": "Total number of matching sections."})
    totalPages = fields.Int(metadata={"description": "Total pages at the effective page size."})
    hasNext = fields.Bool(metadata={"description": "Whether a next page exists."})
    hasPrev = fields.Bool(metadata={"description": "Whether a previous page exists."})
    nextNum = fields.Int(
        allow_none=True,
        metadata={"description": "Next page number when `hasNext` is true."},
    )
    prevNum = fields.Int(
        allow_none=True,
        metadata={"description": "Previous page number when `hasPrev` is true."},
    )


class DumpEntrySchema(Schema):
    timestamp = fields.Str(
        required=True,
        metadata={"description": "Dump timestamp label (derived from object prefix)."},
    )
    sql = fields.Url(
        required=False,
        allow_none=True,
        metadata={"description": "Public URL for the compressed SQL dump file."},
    )
    sha256 = fields.Str(
        required=False,
        allow_none=True,
        metadata={"description": "SHA-256 digest value for the dump when available."},
    )
    sha256_url = fields.Url(
        required=False,
        allow_none=True,
        metadata={"description": "Public URL to a `.sha256` checksum file."},
    )
    manifest = fields.Url(
        required=False,
        allow_none=True,
        metadata={"description": "Public URL to a manifest JSON file for the dump."},
    )
    size_bytes = fields.Int(
        required=False,
        allow_none=True,
        metadata={"description": "Dump file size in bytes when available."},
    )
    warning = fields.Str(
        required=False,
        allow_none=True,
        metadata={"description": "Warning message when metadata is incomplete."},
    )


class AgreementArgsSchema(Schema):
    focusSectionUuid = fields.Str(
        required=False,
        allow_none=True,
        metadata={
            "description": (
                "Optional section UUID used when redacting anonymous responses to keep a "
                "focused neighborhood visible."
            )
        },
    )
    neighborSections = fields.Int(
        load_default=1,
        metadata={
            "description": (
                "Number of neighboring sections to include around `focusSectionUuid` when "
                "response XML is redacted."
            ),
            "example": 1,
        },
    )


class AgreementsIndexArgsSchema(Schema):
    page = fields.Int(load_default=1)
    pageSize = fields.Int(load_default=25)
    sortBy = fields.Str(load_default="year")
    sortDir = fields.Str(load_default="desc")
    query = fields.Str(load_default="")


class AgreementResponseSchema(Schema):
    year = fields.Int(metadata={"description": "Agreement year."})
    target = fields.Str(metadata={"description": "Target company name."})
    acquirer = fields.Str(metadata={"description": "Acquirer company name."})
    filing_date = fields.Str(allow_none=True, metadata={"description": "SEC filing date, when available."})
    prob_filing = fields.Float(allow_none=True, metadata={"description": "Model confidence score for filing linkage."})
    filing_company_name = fields.Str(
        allow_none=True,
        metadata={"description": "Filing entity name in SEC metadata."},
    )
    filing_company_cik = fields.Str(
        allow_none=True,
        metadata={"description": "Filing entity CIK in SEC metadata."},
    )
    form_type = fields.Str(allow_none=True, metadata={"description": "SEC form type."})
    exhibit_type = fields.Str(allow_none=True, metadata={"description": "SEC exhibit type."})
    transaction_price_total = fields.Float(
        allow_none=True,
        metadata={"description": "Total transaction value when available."},
    )
    transaction_price_stock = fields.Float(
        allow_none=True,
        metadata={"description": "Stock portion of transaction consideration."},
    )
    transaction_price_cash = fields.Float(
        allow_none=True,
        metadata={"description": "Cash portion of transaction consideration."},
    )
    transaction_price_assets = fields.Float(
        allow_none=True,
        metadata={"description": "Asset portion of transaction consideration."},
    )
    transaction_consideration = fields.Str(
        allow_none=True,
        metadata={"description": "High-level consideration type classification."},
    )
    target_type = fields.Str(allow_none=True, metadata={"description": "Target company type."})
    acquirer_type = fields.Str(allow_none=True, metadata={"description": "Acquirer company type."})
    target_industry = fields.Str(
        allow_none=True,
        metadata={"description": "Target industry classification."},
    )
    acquirer_industry = fields.Str(
        allow_none=True,
        metadata={"description": "Acquirer industry classification."},
    )
    announce_date = fields.Str(allow_none=True, metadata={"description": "Public deal announcement date."})
    close_date = fields.Str(allow_none=True, metadata={"description": "Deal close date, when available."})
    deal_status = fields.Str(allow_none=True, metadata={"description": "Current status of the deal."})
    attitude = fields.Str(allow_none=True, metadata={"description": "Deal attitude classification."})
    deal_type = fields.Str(allow_none=True, metadata={"description": "Deal type classification."})
    purpose = fields.Str(allow_none=True, metadata={"description": "Deal purpose classification."})
    target_pe = fields.Bool(allow_none=True, metadata={"description": "Whether target is private-equity backed."})
    acquirer_pe = fields.Bool(allow_none=True, metadata={"description": "Whether acquirer is private-equity backed."})
    url = fields.Str(metadata={"description": "Source filing URL."})
    xml = fields.Str(metadata={"description": "Agreement XML content (may be redacted for anonymous access)."})
    isRedacted = fields.Bool(
        required=False,
        metadata={"description": "Present and true when XML has been redacted for access control."},
    )


class SectionResponseSchema(Schema):
    agreementUuid = fields.Str(metadata={"description": "Agreement UUID that owns this section."})
    sectionUuid = fields.Str(metadata={"description": "Section UUID."})
    articleStandardId = fields.Str(
        metadata={"description": "Taxonomy standard ID for the parent article."}
    )
    sectionStandardId = fields.List(
        fields.Str(),
        metadata={"description": "Taxonomy standard IDs for this section."},
    )
    xml = fields.Str(metadata={"description": "Section XML content."})
    articleTitle = fields.Str(metadata={"description": "Parent article heading."})
    sectionTitle = fields.Str(metadata={"description": "Section heading text."})


# ── Auth request schemas ──────────────────────────────────────────────────
# ── Route definitions ───────────────────────────────────────

@agreements_blp.route("/<string:agreement_uuid>")
class AgreementResource(MethodView):
    @agreements_blp.doc(
        operationId="getAgreement",
        summary="Retrieve agreement text by UUID",
        description=(
            "Returns agreement metadata and XML content. For anonymous callers, XML can be "
            "redacted based on `focusSectionUuid` and `neighborSections`."
        ),
        parameters=[
            {
                "in": "path",
                "name": "agreement_uuid",
                "required": True,
                "schema": {"type": "string", "minLength": 1},
                "description": "Agreement UUID.",
                "example": "8f89fe31-f77e-45dc-91e0-c5e38fe4004f",
            }
        ],
    )
    @agreements_blp.arguments(AgreementArgsSchema, location="query")
    @agreements_blp.response(200, AgreementResponseSchema)
    def get(self, args, agreement_uuid) -> dict[str, object]:
        ctx = _current_access_context()
        focus_section_uuid = args.get("focusSectionUuid")
        if focus_section_uuid is not None:
            focus_section_uuid = focus_section_uuid.strip()
            if not _UUID_RE.match(focus_section_uuid):
                abort(400, description="Invalid focusSectionUuid.")
        neighbor_sections_int = args["neighborSections"]

        # Only serve agreements with pdx.xml status null or 'verified' (latest version).
        year_expr = _agreement_year_expr().label("year")
        eligible = _xml_eligible_latest_subquery()
        row = (
            db.session.query(
                year_expr,
                Agreements.target,
                Agreements.acquirer,
                Agreements.filing_date,
                Agreements.prob_filing,
                Agreements.filing_company_name,
                Agreements.filing_company_cik,
                Agreements.form_type,
                Agreements.exhibit_type,
                Agreements.transaction_price_total,
                Agreements.transaction_price_stock,
                Agreements.transaction_price_cash,
                Agreements.transaction_price_assets,
                Agreements.transaction_consideration,
                Agreements.target_type,
                Agreements.acquirer_type,
                Agreements.target_industry,
                Agreements.acquirer_industry,
                Agreements.announce_date,
                Agreements.close_date,
                Agreements.deal_status,
                Agreements.attitude,
                Agreements.deal_type,
                Agreements.purpose,
                Agreements.target_pe,
                Agreements.acquirer_pe,
                Agreements.url,
                XML.xml,
            )
            .join(eligible, Agreements.agreement_uuid == eligible.c.agreement_uuid)
            .join(
                XML,
                and_(
                    XML.agreement_uuid == eligible.c.agreement_uuid,
                    XML.version == eligible.c.version,
                ),
            )
            .filter(Agreements.agreement_uuid == agreement_uuid)
            .first()
        )

        if row is None:
            abort(404)

        xml_content = row.xml
        payload = {
            "year": row.year,
            "target": row.target,
            "acquirer": row.acquirer,
            "filing_date": row.filing_date,
            "prob_filing": row.prob_filing,
            "filing_company_name": row.filing_company_name,
            "filing_company_cik": row.filing_company_cik,
            "form_type": row.form_type,
            "exhibit_type": row.exhibit_type,
            "transaction_price_total": row.transaction_price_total,
            "transaction_price_stock": row.transaction_price_stock,
            "transaction_price_cash": row.transaction_price_cash,
            "transaction_price_assets": row.transaction_price_assets,
            "transaction_consideration": row.transaction_consideration,
            "target_type": row.target_type,
            "acquirer_type": row.acquirer_type,
            "target_industry": row.target_industry,
            "acquirer_industry": row.acquirer_industry,
            "announce_date": row.announce_date,
            "close_date": row.close_date,
            "deal_status": row.deal_status,
            "attitude": row.attitude,
            "deal_type": row.deal_type,
            "purpose": row.purpose,
            "target_pe": row.target_pe,
            "acquirer_pe": row.acquirer_pe,
            "url": row.url,
        }
        if not ctx.is_authenticated:
            redacted_xml = _redact_agreement_xml(
                xml_content,
                focus_section_uuid=focus_section_uuid,
                neighbor_sections=neighbor_sections_int,
            )
            payload["xml"] = redacted_xml
            payload["isRedacted"] = True
            return payload
        payload["xml"] = xml_content
        return payload


@sections_blp.route("/<string:section_uuid>")
class SectionResource(MethodView):
    @sections_blp.doc(
        operationId="getSection",
        summary="Retrieve section text by UUID",
        description="Returns one section payload including taxonomy IDs and XML content.",
        parameters=[
            {
                "in": "path",
                "name": "section_uuid",
                "required": True,
                "schema": {"type": "string", "minLength": 1},
                "description": "Section UUID.",
                "example": "5f7e1853-60ed-4f1c-b5fe-a4f5e237f97e",
            }
        ],
    )
    @sections_blp.response(200, SectionResponseSchema)
    def get(self, section_uuid: str) -> dict[str, object]:
        section_uuid = section_uuid.strip()
        if not _UUID_RE.match(section_uuid):
            abort(400, description="Invalid sectionUuid.")

        section_standard_ids_expr = _coalesced_section_standard_ids().label(
            "section_standard_ids"
        )
        xml_agreements = _xml_agreements_subquery()
        row = (
            db.session.query(
                Sections.agreement_uuid,
                Sections.section_uuid,
                Sections.article_standard_id,
                section_standard_ids_expr,
                Sections.xml_content,
                Sections.article_title,
                Sections.section_title,
            )
            .join(
                xml_agreements,
                Sections.agreement_uuid == xml_agreements.c.agreement_uuid,
            )
            .filter(Sections.section_uuid == section_uuid)
            .first()
        )

        if row is None:
            abort(404)

        (
            agreement_uuid,
            section_uuid,
            article_standard_id,
            section_standard_ids_raw,
            xml_content,
            article_title,
            section_title,
        ) = row

        section_standard_ids = _parse_section_standard_ids(section_standard_ids_raw)

        return {
            "agreementUuid": agreement_uuid,
            "sectionUuid": section_uuid,
            "articleStandardId": article_standard_id,
            "sectionStandardId": section_standard_ids,
            "xml": xml_content,
            "articleTitle": article_title,
            "sectionTitle": section_title,
        }


def get_agreements_index() -> dict[str, object]:
    ctx = _current_access_context()
    args = _load_query(AgreementsIndexArgsSchema())
    page = int(args["page"])
    page_size = int(args["pageSize"])
    sort_by = str(args["sortBy"] or "year")
    sort_dir = str(args["sortDir"] or "desc")
    query = str(args.get("query") or "").strip()

    if page < 1:
        page = 1

    max_page_size = 100 if ctx.is_authenticated else 10
    if page_size < 1 or page_size > max_page_size:
        page_size = min(25, max_page_size)

    year_expr = _agreement_year_expr()
    sort_map = {
        "year": year_expr,
        "target": Agreements.target,
        "acquirer": Agreements.acquirer,
    }
    sort_column = sort_map.get(sort_by, year_expr)
    sort_direction = sort_dir.lower()
    order_by = sort_column.desc() if sort_direction == "desc" else sort_column.asc()

    xml_agreements = _xml_agreements_subquery()
    q = (
        db.session.query(
            Agreements.agreement_uuid,
            year_expr.label("year"),
            Agreements.target,
            Agreements.acquirer,
            Agreements.verified,
        )
        .join(
            xml_agreements,
            Agreements.agreement_uuid == xml_agreements.c.agreement_uuid,
        )
    )
    count_q = (
        db.session.query(func.count(xml_agreements.c.agreement_uuid))
        .select_from(Agreements)
        .join(
            xml_agreements,
            Agreements.agreement_uuid == xml_agreements.c.agreement_uuid,
        )
    )

    if query:
        if query.isdigit():
            year_value = int(query)
            q = q.filter(year_expr == year_value)
            count_q = count_q.filter(year_expr == year_value)
        else:
            like = f"{query}%"
            filters = or_(
                Agreements.target.ilike(like),
                Agreements.acquirer.ilike(like),
            )
            q = q.filter(filters)
            count_q = count_q.filter(filters)

    q = q.order_by(order_by, Agreements.agreement_uuid)

    total_count = count_q.scalar()
    total_count = int(total_count or 0)
    offset = (page - 1) * page_size
    items = q.offset(offset).limit(page_size).all()
    meta = _pagination_metadata(total_count=total_count, page=page, page_size=page_size)

    results = [
        {
            "agreementUuid": row.agreement_uuid,
            "year": row.year,
            "target": row.target,
            "acquirer": row.acquirer,
            "considerationType": None,
            "totalConsideration": None,
            "targetIndustry": None,
            "acquirerIndustry": None,
            "verified": bool(row.verified) if row.verified is not None else False,
        }
        for row in items
    ]

    return {"results": results, **meta}


def get_agreements_status_summary() -> dict[str, object]:
    latest_filing_date: object | None = (
        db.session.query(func.max(Agreements.filing_date))
        .filter(Agreements.filing_date.isnot(None), Agreements.filing_date != "")
        .scalar()
    )
    if isinstance(latest_filing_date, (date, datetime)):
        latest_filing_date = latest_filing_date.isoformat()
    elif latest_filing_date is not None:
        latest_filing_date = str(latest_filing_date)
    rows = (
        db.session.execute(
            text(
                f"""
                SELECT
                    year,
                    color,
                    current_stage,
                    count
                FROM {_schema_prefix()}agreement_status_summary
                WHERE year IS NOT NULL
                ORDER BY year ASC, current_stage ASC, color ASC
                """
            )
        )
        .mappings()
        .all()
    )

    years = [
        {
            "year": int(row["year"]),
            "color": row["color"],
            "currentStage": row["current_stage"],
            "count": int(row["count"] or 0),
        }
        for row in rows
    ]
    return {"years": years, "latestFilingDate": latest_filing_date}


def get_agreements_summary() -> dict[str, int]:
    now = time.time()
    with _agreements_summary_lock:
        cached_payload = _agreements_summary_cache["payload"]
        cached_ts = _agreements_summary_cache["ts"]
        cache_is_valid = cached_payload is not None and (
            now - cached_ts < _AGREEMENTS_SUMMARY_TTL_SECONDS
        )
    if cache_is_valid:
        return cached_payload

    row = db.session.execute(
        text(
            f"""
            SELECT
              COALESCE(SUM(count_agreements), 0) AS agreements,
              COALESCE(SUM(count_sections), 0) AS sections,
              COALESCE(SUM(count_pages), 0) AS pages
            FROM {_schema_prefix()}summary_data
            """
        )
    ).mappings().first()

    payload = {
        "agreements": int((row or {}).get("agreements", 0) or 0),
        "sections": int((row or {}).get("sections", 0) or 0),
        "pages": int((row or {}).get("pages", 0) or 0),
    }
    with _agreements_summary_lock:
        _agreements_summary_cache["payload"] = payload
        _agreements_summary_cache["ts"] = now

    return payload


def get_filter_options() -> tuple[Response, int] | Response:
    """Fetch distinct targets, acquirers, and industries from the database"""
    now = time.time()
    with _filter_options_lock:
        cached_payload = _filter_options_cache["payload"]
        cached_ts = _filter_options_cache["ts"]
        cache_is_valid = cached_payload is not None and (
            now - cached_ts < _FILTER_OPTIONS_TTL_SECONDS
        )
    if cache_is_valid:
        resp = jsonify(cached_payload)
        resp.headers["Cache-Control"] = f"public, max-age={_FILTER_OPTIONS_TTL_SECONDS}"
        return resp, 200

    # Restrict to agreements with sections and with pdx.xml status null or 'verified'.
    _xml_eligible = (
        "EXISTS ("
        "  SELECT 1 FROM {t}xml x "
        "  WHERE x.agreement_uuid = a.agreement_uuid "
        "    AND (x.status IS NULL OR x.status = 'verified')"
        ")"
    ).format(t=_schema_prefix())
    _has_sections = (
        "EXISTS ("
        "  SELECT 1 FROM {t}sections s "
        "  WHERE s.agreement_uuid = a.agreement_uuid"
        ")"
    ).format(t=_schema_prefix())
    targets = [
        row[0]
        for row in db.session.execute(
            text(
                f"""
                SELECT DISTINCT a.target
                FROM {_schema_prefix()}agreements a
                WHERE a.target IS NOT NULL
                  AND a.target <> ''
                  AND {_has_sections}
                  AND {_xml_eligible}
                ORDER BY a.target
                """
            )
        ).fetchall()
    ]
    acquirers = [
        row[0]
        for row in db.session.execute(
            text(
                f"""
                SELECT DISTINCT a.acquirer
                FROM {_schema_prefix()}agreements a
                WHERE a.acquirer IS NOT NULL
                  AND a.acquirer <> ''
                  AND {_has_sections}
                  AND {_xml_eligible}
                ORDER BY a.acquirer
                """
            )
        ).fetchall()
    ]
    target_industries = [
        row[0]
        for row in db.session.execute(
            text(
                f"""
                SELECT DISTINCT a.target_industry
                FROM {_schema_prefix()}agreements a
                WHERE a.target_industry IS NOT NULL
                  AND a.target_industry <> ''
                  AND {_has_sections}
                  AND {_xml_eligible}
                ORDER BY a.target_industry
                """
            )
        ).fetchall()
    ]
    acquirer_industries = [
        row[0]
        for row in db.session.execute(
            text(
                f"""
                SELECT DISTINCT a.acquirer_industry
                FROM {_schema_prefix()}agreements a
                WHERE a.acquirer_industry IS NOT NULL
                  AND a.acquirer_industry <> ''
                  AND {_has_sections}
                  AND {_xml_eligible}
                ORDER BY a.acquirer_industry
                """
            )
        ).fetchall()
    ]

    payload = {
        "targets": targets,
        "acquirers": acquirers,
        "targetIndustries": target_industries,
        "acquirerIndustries": acquirer_industries,
    }
    with _filter_options_lock:
        _filter_options_cache["payload"] = payload
        _filter_options_cache["ts"] = now

    resp = jsonify(payload)
    resp.headers["Cache-Control"] = f"public, max-age={_FILTER_OPTIONS_TTL_SECONDS}"
    return resp, 200


def _taxonomy_tree() -> dict[str, object]:
    l1_rows = db.session.query(
        TaxonomyL1.standard_id,
        TaxonomyL1.label,
    ).all()
    l2_rows = db.session.query(
        TaxonomyL2.standard_id,
        TaxonomyL2.label,
        TaxonomyL2.parent_id,
    ).all()
    l3_rows = db.session.query(
        TaxonomyL3.standard_id,
        TaxonomyL3.label,
        TaxonomyL3.parent_id,
    ).all()

    l2_by_parent: dict[str, list[object]] = defaultdict(list)
    for row in l2_rows:
        if not isinstance(row.parent_id, str) or not isinstance(row.label, str):
            raise ValueError("taxonomy_l2 has invalid parent_id or label.")
        l2_by_parent[row.parent_id].append(row)

    l3_by_parent: dict[str, list[object]] = defaultdict(list)
    for row in l3_rows:
        if not isinstance(row.parent_id, str) or not isinstance(row.label, str):
            raise ValueError("taxonomy_l3 has invalid parent_id or label.")
        l3_by_parent[row.parent_id].append(row)

    validated_l1_rows = []
    for l1 in l1_rows:
        if not isinstance(l1.standard_id, str) or not isinstance(l1.label, str):
            raise ValueError("taxonomy_l1 has invalid standard_id or label.")
        validated_l1_rows.append(l1)

    tree: dict[str, object] = {}
    for l1 in sorted(validated_l1_rows, key=lambda r: r.label):
        l2_children: dict[str, object] = {}
        for l2 in sorted(l2_by_parent.get(l1.standard_id, []), key=lambda r: r.label):
            if not isinstance(l2.standard_id, str) or not isinstance(l2.label, str):
                raise ValueError("taxonomy_l2 has invalid standard_id or label.")
            l3_children: dict[str, object] = {}
            for l3 in sorted(l3_by_parent.get(l2.standard_id, []), key=lambda r: r.label):
                if not isinstance(l3.standard_id, str) or not isinstance(l3.label, str):
                    raise ValueError("taxonomy_l3 has invalid standard_id or label.")
                l3_children[l3.label] = {"id": l3.standard_id}
            l2_children[l2.label] = {"id": l2.standard_id, "children": l3_children}
        tree[l1.label] = {"id": l1.standard_id, "children": l2_children}

    return tree


def _get_taxonomy_payload_cached() -> tuple[dict[str, object], bool]:
    now = time.time()
    with _taxonomy_lock:
        cached_payload = _taxonomy_cache["payload"]
        cached_ts = _taxonomy_cache["ts"]
        cache_is_valid = cached_payload is not None and (
            now - cached_ts < _TAXONOMY_TTL_SECONDS
        )
    if cache_is_valid:
        return cached_payload, True

    payload = _taxonomy_tree()
    with _taxonomy_lock:
        _taxonomy_cache["payload"] = payload
        _taxonomy_cache["ts"] = now

    return payload, False


def _register_main_routes(target_app: Flask) -> None:
    target_app.add_url_rule(
        "/v1/agreements-index", view_func=get_agreements_index, methods=["GET"]
    )
    target_app.add_url_rule(
        "/v1/agreements-summary", view_func=get_agreements_summary, methods=["GET"]
    )
    target_app.add_url_rule(
        "/v1/agreements-status-summary",
        view_func=get_agreements_status_summary,
        methods=["GET"],
    )
    target_app.add_url_rule(
        "/v1/filter-options", view_func=get_filter_options, methods=["GET"]
    )


@search_blp.route("")
class SearchResource(MethodView):
    @search_blp.doc(
        operationId="searchSections",
        summary="Search agreement sections",
        description=(
            "Searches sections using structured filters and taxonomy IDs. For list filters, "
            "repeat query keys (for example `year=2023&year=2024`)."
        ),
    )
    @search_blp.arguments(SearchArgsSchema, location="query")
    @search_blp.response(200, SearchResponseSchema)
    def get(self, args) -> dict[str, object]:
        ctx = _current_access_context()
        years = args["year"]
        targets = args["target"]
        acquirers = args["acquirer"]
        standard_ids = args["standardId"]
        # New filters from DB definition
        transaction_price_totals = args["transactionPriceTotal"]
        transaction_price_stocks = args["transactionPriceStock"]
        transaction_price_cashes = args["transactionPriceCash"]
        transaction_price_assets = args["transactionPriceAssets"]
        transaction_considerations = args["transactionConsideration"]
        target_types = args["targetType"]
        acquirer_types = args["acquirerType"]
        target_industries = args["targetIndustry"]
        acquirer_industries = args["acquirerIndustry"]
        deal_statuses = args["dealStatus"]
        attitudes = args["attitude"]
        deal_types = args["dealType"]
        purposes = args["purpose"]
        target_pes = args["targetPe"]
        acquirer_pes = args["acquirerPe"]
        # Text filters
        agreement_uuid = args["agreementUuid"]
        section_uuid = args["sectionUuid"]
        # Legacy filters (kept for backward compatibility)
        transaction_sizes = args["transactionSize"]
        transaction_types = args["transactionType"]
        consideration_types = args["considerationType"]
        # Sort parameters
        sort_by = args["sortBy"]
        sort_direction = args["sortDirection"]

        # pagination parameters
        page = args["page"]
        page_size = args["pageSize"]

        # Validate pagination parameters
        if page < 1:
            page = 1
        max_page_size = 100 if ctx.is_authenticated else 10
        if page_size < 1 or page_size > max_page_size:
            page_size = min(25, max_page_size)

        section_standard_ids_expr = _coalesced_section_standard_ids()

        year_expr = _agreement_year_expr()

        # build the base ORM query
        xml_agreements = _xml_agreements_subquery()
        q = (
            db.session.query(
                Sections.section_uuid,
                Sections.agreement_uuid,
                section_standard_ids_expr.label("section_standard_ids"),
                Sections.xml_content,
                Sections.article_title,
                Sections.section_title,
                Agreements.acquirer,
                Agreements.target,
                year_expr.label("year"),
                Agreements.verified,
            )
            .join(Agreements, Sections.agreement_uuid == Agreements.agreement_uuid)
            .join(
                xml_agreements,
                Agreements.agreement_uuid == xml_agreements.c.agreement_uuid,
            )
        )

        # apply filters only when provided - now handling multiple values
        if years:
            q = q.filter(year_expr.in_(years))

        if targets:
            q = q.filter(Agreements.target.in_(targets))

        if acquirers:
            q = q.filter(Agreements.acquirer.in_(acquirers))

        if standard_ids:
            expanded_standard_ids = _expand_taxonomy_standard_ids(standard_ids)
            if expanded_standard_ids:
                json_filters = [
                    func.json_contains(
                        section_standard_ids_expr, func.json_quote(value)
                    )
                    for value in expanded_standard_ids
                ]
                q = q.filter(or_(*json_filters))

        # Transaction Size filter - convert ranges to DB values
        if transaction_sizes:
            size_conditions = []
            for size_range in transaction_sizes:
                if size_range == "100M - 250M":
                    size_conditions.append(
                        db.and_(
                            Agreements.transaction_size >= 100000000,
                            Agreements.transaction_size < 250000000,
                        )
                    )
                elif size_range == "250M - 500M":
                    size_conditions.append(
                        db.and_(
                            Agreements.transaction_size >= 250000000,
                            Agreements.transaction_size < 500000000,
                        )
                    )
                elif size_range == "500M - 750M":
                    size_conditions.append(
                        db.and_(
                            Agreements.transaction_size >= 500000000,
                            Agreements.transaction_size < 750000000,
                        )
                    )
                elif size_range == "750M - 1B":
                    size_conditions.append(
                        db.and_(
                            Agreements.transaction_size >= 750000000,
                            Agreements.transaction_size < 1000000000,
                        )
                    )
                elif size_range == "1B - 5B":
                    size_conditions.append(
                        db.and_(
                            Agreements.transaction_size >= 1000000000,
                            Agreements.transaction_size < 5000000000,
                        )
                    )
                elif size_range == "5B - 10B":
                    size_conditions.append(
                        db.and_(
                            Agreements.transaction_size >= 5000000000,
                            Agreements.transaction_size < 10000000000,
                        )
                    )
                elif size_range == "10B - 20B":
                    size_conditions.append(
                        db.and_(
                            Agreements.transaction_size >= 10000000000,
                            Agreements.transaction_size < 20000000000,
                        )
                    )
                elif size_range == "20B+":
                    size_conditions.append(Agreements.transaction_size >= 20000000000)
            if size_conditions:
                q = q.filter(db.or_(*size_conditions))

        # Transaction Type filter
        if transaction_types:
            # Convert frontend values to DB enum values
            db_transaction_types = []
            for t_type in transaction_types:
                if t_type == "Strategic":
                    db_transaction_types.append("strategic")
                elif t_type == "Financial":
                    db_transaction_types.append("financial")
            if db_transaction_types:
                q = q.filter(Agreements.transaction_type.in_(db_transaction_types))

        # Consideration Type filter
        if consideration_types:
            # Convert frontend values to DB enum values
            db_consideration_types = []
            for c_type in consideration_types:
                if c_type == "All stock":
                    db_consideration_types.append("stock")
                elif c_type == "All cash":
                    db_consideration_types.append("cash")
                elif c_type == "Mixed":
                    db_consideration_types.append("mixed")
            if db_consideration_types:
                q = q.filter(Agreements.consideration_type.in_(db_consideration_types))

        # Target Type filter
        if target_types:
            # Frontend sends lowercase values, use them directly
            q = q.filter(Agreements.target_type.in_(target_types))

        # New filters from DB definition (disabled for now, ready to enable)
        # Transaction Price filters - will be enabled when frontend is ready
        # if transaction_price_totals:
        #     q = q.filter(Agreements.transaction_price_total.in_(transaction_price_totals))
        # if transaction_price_stocks:
        #     q = q.filter(Agreements.transaction_price_stock.in_(transaction_price_stocks))
        # if transaction_price_cashes:
        #     q = q.filter(Agreements.transaction_price_cash.in_(transaction_price_cashes))
        # if transaction_price_assets:
        #     q = q.filter(Agreements.transaction_price_assets.in_(transaction_price_assets))

        # Transaction Consideration filter
        if transaction_considerations:
            q = q.filter(Agreements.transaction_consideration.in_(transaction_considerations))

        # Acquirer Type filter
        if acquirer_types:
            # Convert frontend values to DB enum values (same as target_type)
            db_acquirer_types = []
            for a_type in acquirer_types:
                if a_type == "public":
                    db_acquirer_types.append("public")
                elif a_type == "private":
                    db_acquirer_types.append("private")
            if db_acquirer_types:
                q = q.filter(Agreements.acquirer_type.in_(db_acquirer_types))

        # Target Industry filter
        if target_industries:
            q = q.filter(Agreements.target_industry.in_(target_industries))

        # Acquirer Industry filter
        if acquirer_industries:
            q = q.filter(Agreements.acquirer_industry.in_(acquirer_industries))

        # Deal Status filter
        if deal_statuses:
            q = q.filter(Agreements.deal_status.in_(deal_statuses))

        # Attitude filter
        if attitudes:
            q = q.filter(Agreements.attitude.in_(attitudes))

        # Deal Type filter
        if deal_types:
            q = q.filter(Agreements.deal_type.in_(deal_types))

        # Purpose filter
        if purposes:
            q = q.filter(Agreements.purpose.in_(purposes))

        # Target PE filter
        if target_pes:
            db_target_pes = []
            for pe in target_pes:
                if pe == "true":
                    db_target_pes.append(1)
                elif pe == "false":
                    db_target_pes.append(0)
            if db_target_pes:
                q = q.filter(Agreements.target_pe.in_(db_target_pes))

        # Acquirer PE filter
        if acquirer_pes:
            db_acquirer_pes = []
            for pe in acquirer_pes:
                if pe == "true":
                    db_acquirer_pes.append(1)
                elif pe == "false":
                    db_acquirer_pes.append(0)
            if db_acquirer_pes:
                q = q.filter(Agreements.acquirer_pe.in_(db_acquirer_pes))

        # Agreement UUID filter
        if agreement_uuid and agreement_uuid.strip():
            q = q.filter(Agreements.agreement_uuid == agreement_uuid.strip())

        # Section UUID filter
        if section_uuid and section_uuid.strip():
            q = q.filter(Sections.section_uuid == section_uuid.strip())

        # Apply sorting based on sort_by and sort_direction
        if sort_by == "year":
            year_expr = _agreement_year_expr()
            if sort_direction == "desc":
                q = q.order_by(desc(year_expr))
            else:
                q = q.order_by(asc(year_expr))
        elif sort_by == "target":
            if sort_direction == "desc":
                q = q.order_by(desc(Agreements.target))
            else:
                q = q.order_by(asc(Agreements.target))
        elif sort_by == "acquirer":
            if sort_direction == "desc":
                q = q.order_by(desc(Agreements.acquirer))
            else:
                q = q.order_by(asc(Agreements.acquirer))

        count_subquery = (
            q.order_by(None)
            .with_entities(Sections.section_uuid)
            .distinct()
            .subquery()
        )
        total_count = db.session.query(func.count()).select_from(count_subquery).scalar()
        total_count = int(total_count or 0)
        offset = (page - 1) * page_size
        items = q.offset(offset).limit(page_size).all()
        meta = _pagination_metadata(total_count=total_count, page=page, page_size=page_size)

        # marshal into JSON with pagination metadata
        results = [
            {
                "id": r.section_uuid,
                "agreementUuid": r.agreement_uuid,
                "sectionUuid": r.section_uuid,
                "standardId": _parse_section_standard_ids(r.section_standard_ids),
                "xml": r.xml_content,
                "articleTitle": r.article_title,
                "sectionTitle": r.section_title,
                "acquirer": r.acquirer,
                "target": r.target,
                "year": r.year,
                "verified": r.verified,
            }
            for r in items
        ]

        # Return results with pagination metadata
        return {
            "results": results,
            "access": {
                "tier": ctx.tier,
                "message": None
                if ctx.is_authenticated
                else "Limited mode: sign in to view clause text and unlock full pagination.",
            },
            **meta,
        }


@taxonomy_blp.route("")
class TaxonomyResource(MethodView):
    @taxonomy_blp.doc(
        operationId="getTaxonomy",
        summary="Retrieve clause taxonomy",
        description=(
            "Returns the hierarchical Pandects taxonomy tree keyed by standard ID."
        ),
        responses={
            200: {
                "description": "OK",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "additionalProperties": {
                                "type": "object",
                                "required": ["id"],
                                "properties": {
                                    "id": {"type": "string"},
                                    "children": {
                                        "type": "object",
                                        "additionalProperties": {"type": "object"},
                                    },
                                },
                            },
                        }
                    }
                },
            }
        }
    )
    def get(self) -> Response:
        payload, _ = _get_taxonomy_payload_cached()
        resp = jsonify(payload)
        resp.headers["Cache-Control"] = f"public, max-age={_TAXONOMY_TTL_SECONDS}"
        return resp


@dumps_blp.route("")  # blueprint already has url_prefix="/v1/dumps"
class DumpListResource(MethodView):
    @dumps_blp.doc(
        operationId="listDumps",
        summary="List available bulk dumps",
        description=(
            "Returns newest-first metadata for publicly available database dump artifacts."
        ),
    )
    @dumps_blp.response(200, DumpEntrySchema(many=True))
    def get(self) -> list[dict[str, object]]:
        now = time.time()
        with _dumps_cache_lock:
            cached_payload = _dumps_cache["payload"]
            cached_ts = _dumps_cache["ts"]
            cache_is_valid = cached_payload is not None and (
                now - cached_ts < _DUMPS_CACHE_TTL_SECONDS
            )
        if cache_is_valid:
            return cached_payload
        if client is None:
            return []
        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=R2_BUCKET_NAME, Prefix="dumps/")

        dumps_map = defaultdict(dict)
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                etag = obj.get("ETag")
                filename = key.rsplit("/", 1)[-1]

                if filename.endswith(".sql.gz.manifest.json"):
                    prefix = filename[: -len(".sql.gz.manifest.json")]
                    dumps_map[prefix]["manifest"] = key
                    if isinstance(etag, str):
                        dumps_map[prefix]["manifest_etag"] = etag.strip('"')

                elif filename.endswith(".sql.gz.sha256"):
                    prefix = filename[: -len(".sql.gz.sha256")]
                    dumps_map[prefix]["sha256"] = key

                elif filename.endswith(".sql.gz"):
                    prefix = filename[: -len(".sql.gz")]
                    dumps_map[prefix]["sql"] = key

                elif filename.endswith(".json"):
                    prefix = filename[: -len(".json")]
                    dumps_map[prefix]["manifest"] = key

        dump_list = []
        for prefix, files in sorted(dumps_map.items(), reverse=True):
            label = prefix.replace("db_dump_", "")
            entry = {"timestamp": label}

            if "sql" in files:
                entry["sql"] = f"{PUBLIC_DEV_BASE}/{files['sql']}"

            if "sha256" in files:
                entry["sha256_url"] = f"{PUBLIC_DEV_BASE}/{files['sha256']}"

            if "manifest" in files:
                entry["manifest"] = f"{PUBLIC_DEV_BASE}/{files['manifest']}"
                manifest_key = files["manifest"]
                manifest_etag = files.get("manifest_etag")
                cached_manifest = None
                now = time.time()
                if manifest_key and isinstance(manifest_etag, str):
                    with _dumps_manifest_cache_lock:
                        cached_manifest = _dumps_manifest_cache.get(manifest_key)
                        if cached_manifest is not None:
                            cache_age = now - float(cached_manifest.get("ts", 0.0))
                            if (
                                cached_manifest.get("etag") != manifest_etag
                                or cache_age >= _DUMPS_MANIFEST_CACHE_TTL_SECONDS
                            ):
                                cached_manifest = None
                if cached_manifest is not None:
                    data = cached_manifest.get("payload") or {}
                else:
                    try:
                        body = client.get_object(
                            Bucket=R2_BUCKET_NAME, Key=files["manifest"]
                        )["Body"].read()
                        data = json.loads(body)
                        if manifest_key and isinstance(manifest_etag, str):
                            with _dumps_manifest_cache_lock:
                                _dumps_manifest_cache[manifest_key] = {
                                    "etag": manifest_etag,
                                    "payload": data,
                                    "ts": now,
                                }
                    except Exception as e:
                        entry["warning"] = f"couldn't read manifest: {e}"
                        data = {}
                if "size_bytes" in data:
                    entry["size_bytes"] = data["size_bytes"]
                if "sha256" in data:
                    entry["sha256"] = data["sha256"]

            dump_list.append(entry)

        with _dumps_cache_lock:
            _dumps_cache["payload"] = dump_list
            _dumps_cache["ts"] = now

        return dump_list


def _register_blueprints() -> None:
    api.register_blueprint(search_blp)
    api.register_blueprint(dumps_blp)
    api.register_blueprint(taxonomy_blp)
    api.register_blueprint(agreements_blp)
    api.register_blueprint(sections_blp)


def _register_app(target_app: Flask) -> None:
    from backend.routes.auth import register_auth_routes

    _register_error_handlers(target_app)
    _register_request_hooks(target_app)
    _register_blueprints()
    _register_main_routes(target_app)
    register_auth_routes(target_app, app_module=sys.modules[__name__])


def create_app(*, config_overrides: dict[str, object] | None = None) -> Flask:
    target_app = Flask(__name__)
    _configure_app(target_app, config_overrides=config_overrides)
    _register_app(target_app)
    _ensure_auth_tables_exist(target_app)
    return target_app


def create_test_app(*, config_overrides: dict[str, object] | None = None) -> Flask:
    test_app = create_app(config_overrides=config_overrides)
    test_app.testing = True
    return test_app


app = create_app()

# ── CLI command for auth DB initialization ───────────────────────────────
@app.cli.command("init-auth-db")
def init_auth_db():
    """Create the auth tables in the configured auth database bind."""
    if _auth_is_mocked():
        raise click.ClickException("Auth DB is not available in mock auth mode.")
    if not _auth_db_is_configured():
        raise click.ClickException(
            "Auth DB is not configured. Set AUTH_DATABASE_URI or DATABASE_URL."
        )

    with app.app_context():
        db.create_all(bind_key="auth")
        engine = db.engines.get("auth")
        if engine is None:
            raise click.ClickException("Auth DB bind is missing.")
        inspector = inspect(engine)
        expected = {
            "auth_users",
            "auth_sessions",
            "auth_password_reset_tokens",
            "api_keys",
            "api_usage_daily",
            "api_usage_hourly",
            "api_request_events",
            "api_usage_daily_ips",
        }
        existing = set(inspector.get_table_names())
        missing = sorted(expected - existing)
        if missing:
            raise click.ClickException(
                f"Auth DB initialization failed (missing tables: {', '.join(missing)})."
            )
    click.echo("Auth DB initialized.")


# ── CLI command for OpenAPI spec generation ──────────────────────────────
@app.cli.command("gen-openapi")
def gen_openapi():
    """Generate an OpenAPI3 YAML spec for your Flask-Smorest API."""
    with app.test_request_context():
        yaml_spec = api.spec.to_yaml()
        Path("openapi.yaml").write_text(yaml_spec)
        click.echo("Wrote openapi.yaml")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "").strip().lower() in ("1", "true", "yes")
    app.run(debug=debug, port=port)
