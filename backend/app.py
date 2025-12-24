import os
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
import html as _html
import math
from flask import Flask, jsonify, request, abort, Response, g
from flask import redirect
from flask import make_response
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_smorest import Api, Blueprint
from flask.views import MethodView
import boto3
from collections import defaultdict
from marshmallow import Schema, fields
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
)
from sqlalchemy.dialects.mysql import LONGTEXT, TINYTEXT
from dotenv import load_dotenv
from urllib.parse import urlencode, quote
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import secrets

# Load env vars from `backend/.env` regardless of the process working directory.
load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
# Also allow a repo/root `.env` (or process env) to supply values without overriding.
load_dotenv()

# ── Simple in-process caching ─────────────────────────────────────────────
_FILTER_OPTIONS_TTL_SECONDS = int(os.environ.get("FILTER_OPTIONS_TTL_SECONDS", "21600"))
_filter_options_cache = {"ts": 0.0, "payload": None}
_filter_options_lock = Lock()

# ── Simple in-process rate limiting ──────────────────────────────────────
_rate_limit_lock = Lock()
_rate_limit_state: dict[str, dict[str, float | int]] = {}
_endpoint_rate_limit_state: dict[str, dict[str, float | int]] = {}

# ── API usage logging ─────────────────────────────────────────────────────
_USAGE_SAMPLE_RATE_2XX = float(os.environ.get("USAGE_SAMPLE_RATE_2XX", "0.05"))
_USAGE_SAMPLE_RATE_3XX = float(os.environ.get("USAGE_SAMPLE_RATE_3XX", "0.05"))
_LATENCY_BUCKET_BOUNDS_MS = (25, 50, 100, 250, 500, 1000, 2000, 5000, 10000)
_API_KEY_MIN_HASH_CHECKS = 5
_DUMMY_API_KEY_HASH = generate_password_hash("pdcts_dummy_api_key")

# ── Flask setup ───────────────────────────────────────────────────────────
app = Flask(__name__)

# ── CORS origins ──────────────────────────────────────────────────────────
_DEFAULT_CORS_ORIGINS = (
    "http://localhost:8080",
    "http://127.0.0.1:8080",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "https://pandects.org",
    "https://www.pandects.org",
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
    if not path.startswith("/api/"):
        return False
    if request.method in ("GET", "HEAD", "OPTIONS"):
        return False
    # Require CSRF for session-cookie authenticated requests and for endpoints that
    # establish/tear down a session (login/logout), even before a session exists.
    if request.cookies.get(_SESSION_COOKIE_NAME):
        return True
    return path in (
        "/api/auth/login",
        "/api/auth/register",
        "/api/auth/google/credential",
        "/api/auth/password/forgot",
        "/api/auth/password/reset",
        "/api/auth/logout",
    )


def _auth_is_mocked() -> bool:
    return AUTH_MODE == "mock" and bool(app.debug)


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
if AUTH_DATABASE_URI is not None:
    app.config["SQLALCHEMY_BINDS"] = {"auth": AUTH_DATABASE_URI}
else:
    app.config["SQLALCHEMY_BINDS"] = {
        "auth": f"sqlite:///{Path(__file__).with_name('auth_dev.sqlite')}"
    }

# ── OpenAPI / Flask-Smorest configuration ───────────────────────────────
app.config.update(
    {
        "API_TITLE": "Pandects API",
        "API_VERSION": "v1",
        "OPENAPI_VERSION": "3.0.2",
        "OPENAPI_URL_PREFIX": "/",
        "OPENAPI_SWAGGER_UI_PATH": "/swagger-ui",
        "OPENAPI_SWAGGER_UI_URL": "https://cdn.jsdelivr.net/npm/swagger-ui-dist/",
    }
)

api = Api(app)

# ── JSON error responses for API routes ──────────────────────────────────


@app.errorhandler(HTTPException)
def _handle_http_exception(err: HTTPException):
    if request.path.startswith("/api/"):
        return jsonify({"error": err.name, "message": err.description}), err.code
    return err


@app.errorhandler(InternalServerError)
def _handle_internal_server_error(err: InternalServerError):
    if request.path.startswith("/api/"):
        app.logger.exception("Unhandled API exception: %s", err)
        return (
            jsonify({"error": "Internal Server Error", "message": "Unexpected server error."}),
            500,
        )
    return err

# ── CORS setup ────────────────────────────────────────────────────────────
CORS(
    app,
    resources={
        r"/api/*": {
            "origins": _cors_origins()
        }
    },
    methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-API-Key", "X-CSRF-Token"],
    supports_credentials=True,
)

# —— Bulk data setup ——————��———————————————————————————————————————————————
R2_BUCKET_NAME = "pandects-bulk"
R2_ENDPOINT = "https://7b5e7846d94ee35b35e21999fc4fad5b.r2.cloudflarestorage.com"
PUBLIC_DEV_BASE = "https://bulk.pandects.org"

client = None
if os.environ.get("R2_ACCESS_KEY_ID") and os.environ.get("R2_SECRET_ACCESS_KEY"):
    session = boto3.session.Session()
    client = session.client(
        service_name="s3",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        endpoint_url=R2_ENDPOINT,
    )

# ── Database configuration ───────────────────────────────────────────────
DB_USER = os.environ["MARIADB_USER"]
DB_PASS = os.environ["MARIADB_PASSWORD"]
DB_HOST = os.environ["MARIADB_HOST"]
DB_NAME = os.environ["MARIADB_DATABASE"]

app.config["SQLALCHEMY_DATABASE_URI"] = (
    f"mysql+pymysql://{DB_USER}:{DB_PASS}@{DB_HOST}:3306/{DB_NAME}"
)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)

# ── Auth models (bind: "auth") ───────────────────────────────────────────


class AuthUser(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "auth_users"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = db.Column(db.String(320), unique=True, index=True, nullable=False)
    password_hash = db.Column(db.Text, nullable=True)
    email_verified_at = db.Column(db.DateTime, nullable=True)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class AuthSession(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "auth_sessions"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(
        db.String(36), db.ForeignKey("auth_users.id"), index=True, nullable=False
    )
    token_hash = db.Column(db.String(64), unique=True, index=True, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)
    revoked_at = db.Column(db.DateTime, nullable=True)
    last_used_at = db.Column(db.DateTime, nullable=True)
    ip_address = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.String(512), nullable=True)


class AuthPasswordResetToken(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "auth_password_reset_tokens"

    id = db.Column(db.String(36), primary_key=True)
    user_id = db.Column(
        db.String(36), db.ForeignKey("auth_users.id"), index=True, nullable=False
    )
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)
    used_at = db.Column(db.DateTime, nullable=True)
    ip_address = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.String(512), nullable=True)


class ApiKey(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "api_keys"

    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = db.Column(
        db.String(36), db.ForeignKey("auth_users.id"), index=True, nullable=False
    )
    name = db.Column(db.String(120), nullable=True)
    prefix = db.Column(db.String(18), index=True, nullable=False)
    key_hash = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    last_used_at = db.Column(db.DateTime, nullable=True)
    revoked_at = db.Column(db.DateTime, nullable=True)


class ApiUsageDaily(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "api_usage_daily"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    api_key_id = db.Column(
        db.String(36), db.ForeignKey("api_keys.id"), index=True, nullable=False
    )
    day = db.Column(db.Date, index=True, nullable=False)
    count = db.Column(db.Integer, nullable=False, default=0)

    __table_args__ = (db.UniqueConstraint("api_key_id", "day"),)


class ApiUsageHourly(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "api_usage_hourly"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    api_key_id = db.Column(
        db.String(36), db.ForeignKey("api_keys.id"), index=True, nullable=False
    )
    hour = db.Column(db.DateTime, index=True, nullable=False)
    route = db.Column(db.String(256), nullable=False)
    method = db.Column(db.String(8), nullable=False)
    status_class = db.Column(db.Integer, nullable=False)
    count = db.Column(db.Integer, nullable=False, default=0)
    total_ms = db.Column(db.Integer, nullable=False, default=0)
    max_ms = db.Column(db.Integer, nullable=False, default=0)
    latency_buckets = db.Column(db.JSON, nullable=True)
    request_bytes = db.Column(db.Integer, nullable=False, default=0)
    response_bytes = db.Column(db.Integer, nullable=False, default=0)

    __table_args__ = (
        db.UniqueConstraint(
            "api_key_id", "hour", "route", "method", "status_class"
        ),
        db.Index("ix_api_usage_hourly_route_method", "route", "method"),
    )


class ApiRequestEvent(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "api_request_events"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    api_key_id = db.Column(
        db.String(36), db.ForeignKey("api_keys.id"), index=True, nullable=False
    )
    occurred_at = db.Column(db.DateTime, index=True, nullable=False)
    route = db.Column(db.String(256), nullable=False)
    method = db.Column(db.String(8), nullable=False)
    status_code = db.Column(db.Integer, nullable=False)
    status_class = db.Column(db.Integer, nullable=False)
    latency_ms = db.Column(db.Integer, nullable=False)
    request_bytes = db.Column(db.Integer, nullable=True)
    response_bytes = db.Column(db.Integer, nullable=True)
    ip_hash = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.String(512), nullable=True)

    __table_args__ = (
        db.Index("ix_api_request_events_key_time", "api_key_id", "occurred_at"),
        db.Index("ix_api_request_events_ip_time", "ip_hash", "occurred_at"),
    )


class ApiUsageDailyIp(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "api_usage_daily_ips"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    api_key_id = db.Column(
        db.String(36), db.ForeignKey("api_keys.id"), index=True, nullable=False
    )
    day = db.Column(db.Date, index=True, nullable=False)
    ip_hash = db.Column(db.String(64), nullable=False)
    first_seen_at = db.Column(db.DateTime, nullable=False)

    __table_args__ = (
        db.UniqueConstraint("api_key_id", "day", "ip_hash"),
        db.Index("ix_api_usage_daily_ips_key_day", "api_key_id", "day"),
    )


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


class LegalAcceptance(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "legal_acceptances"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(
        db.String(36), db.ForeignKey("auth_users.id"), index=True, nullable=False
    )
    document = db.Column(db.String(24), nullable=False)
    version = db.Column(db.String(64), nullable=False)
    # Nullable for legacy rows created before document hashing existed (or if policy
    # intentionally omits hashing for certain documents), but always populated for
    # new Terms/Privacy acceptances.
    document_hash = db.Column(db.String(64), nullable=True)
    checked_at = db.Column(db.DateTime, nullable=False)
    submitted_at = db.Column(db.DateTime, nullable=False)
    ip_address = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.String(512), nullable=True)

    __table_args__ = (
        db.Index("ix_legal_acceptances_user_doc_ver", "user_id", "document", "version"),
    )


class AuthSignonEvent(db.Model):
    __bind_key__ = "auth"
    __tablename__ = "auth_signon_events"

    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    user_id = db.Column(
        db.String(36), db.ForeignKey("auth_users.id"), index=True, nullable=False
    )
    provider = db.Column(db.String(32), nullable=False)  # "email" | "google"
    action = db.Column(db.String(32), nullable=False)  # "register" | "login"
    occurred_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    ip_address = db.Column(db.String(64), nullable=True)
    user_agent = db.Column(db.String(512), nullable=True)


def _ensure_auth_tables_exist() -> None:
    if AUTH_DATABASE_URI is not None or _auth_is_mocked():
        return
    with app.app_context():
        db.create_all(bind_key="auth")


_ensure_auth_tables_exist()

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
    db.session.add(
        AuthPasswordResetToken(
            id=reset_id,
            user_id=user_id,
            created_at=now,
            expires_at=expires_at,
            ip_address=_request_ip_address(),
            user_agent=_request_user_agent(),
        )
    )
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
        if app.testing:
            return
        app.logger.warning("Signup notification skipped (missing RESEND_API_KEY/RESEND_FROM_EMAIL).")
        return

    if app.testing:
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
        app.logger.error("Resend signup notification failed (HTTP %s): %s", e.code, details)
    except URLError as e:
        app.logger.error("Resend signup notification failed (network error): %s", e)


def _send_signup_notification_email(*, new_user_email: str) -> None:
    subject = "New Pandects signup"
    text = f"{new_user_email} just signed up as a new user on Pandects."
    _send_resend_text_email(to_email=_SIGNUP_NOTIFICATION_EMAIL, subject=subject, text=text)


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
        if app.testing:
            return
        abort(503, description=f"Email is not configured (missing {', '.join(missing)}).")

    if app.testing:
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
        app.logger.error("Resend email failed (HTTP %s): %s", e.code, details)
        abort(503, description="Email delivery failed.")
    except URLError as e:
        app.logger.error("Resend email failed (network error): %s", e)
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
    session = AuthSession(
        user_id=user_id,
        token_hash=_session_token_hash(token),
        created_at=now,
        expires_at=expires_at,
        ip_address=ip_address,
        user_agent=user_agent,
    )
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
    if " " in value:
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
    if app.debug:
        return "http://localhost:8080"
    abort(503, description="Google auth is not configured (missing PUBLIC_FRONTEND_BASE_URL).")


def _public_api_base_url() -> str:
    base = os.environ.get("PUBLIC_API_BASE_URL", "").strip().rstrip("/")
    if base:
        return base
    if app.debug:
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
    return f"{_public_api_base_url()}/api/auth/google/callback"


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
        path="/api/auth/google/callback",
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
        path="/api/auth/google/callback",
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
        path="/api/auth/google/credential",
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
        path="/api/auth/google/credential",
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
            Response(
                response=json.dumps(
                    {
                        "error": "captcha_required",
                        "message": "Captcha is required to create an account.",
                    }
                ),
                status=412,
                mimetype="application/json",
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
            Response(
                response=json.dumps(
                    {
                        "error": "captcha_failed",
                        "message": "Captcha verification failed. Please retry.",
                    }
                ),
                status=412,
                mimetype="application/json",
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
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        abort(400, description="Expected JSON object body.")
    return data


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


def _api_route_template() -> str | None:
    rule = request.url_rule
    if rule is not None and isinstance(rule.rule, str) and rule.rule:
        return rule.rule
    path = request.path
    if isinstance(path, str) and path:
        return path
    return None


def _ip_hash(value: str | None) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    secret = os.environ.get("AUTH_SECRET_KEY")
    if not isinstance(secret, str) or not secret.strip():
        return None
    digest = hmac.new(secret.encode("utf-8"), value.strip().encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()


def _usage_event_sample_rate(status_code: int) -> float:
    if status_code >= 400:
        return 1.0
    if 300 <= status_code <= 399:
        return _USAGE_SAMPLE_RATE_3XX
    return _USAGE_SAMPLE_RATE_2XX


def _utc_today() -> date:
    return datetime.utcnow().date()


def _init_latency_buckets() -> list[int]:
    return [0] * (len(_LATENCY_BUCKET_BOUNDS_MS) + 1)


def _latency_bucket_index(elapsed_ms: int) -> int:
    for idx, bound in enumerate(_LATENCY_BUCKET_BOUNDS_MS):
        if elapsed_ms <= bound:
            return idx
    return len(_LATENCY_BUCKET_BOUNDS_MS)


def _require_legal_acceptance(data: dict) -> datetime:
    legal = data.get("legal")
    if not isinstance(legal, dict):
        abort(
            Response(
                response=json.dumps(
                    {
                        "error": "legal_required",
                        "message": "Legal acceptance required to create an account.",
                    }
                ),
                status=412,
                mimetype="application/json",
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
    ("POST", "/api/auth/login"): 10,
    ("POST", "/api/auth/register"): 5,
    ("POST", "/api/auth/email/resend"): 5,
    ("POST", "/api/auth/password/forgot"): 5,
    ("POST", "/api/auth/password/reset"): 10,
    ("POST", "/api/auth/google/credential"): 10,
}


def _endpoint_rate_limit_key(method: str, path: str) -> tuple[str, int] | None:
    limit = _ENDPOINT_RATE_LIMITS.get((method, path))
    if limit is None:
        return None
    ip = _request_ip_address() or "unknown"
    return f"endpoint:{method}:{path}:ip:{ip}", limit


def _check_rate_limit(ctx: AccessContext) -> None:
    if not request.path.startswith("/api/"):
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
        Response(
            response=json.dumps(
                {
                    "error": "rate_limited",
                    "message": "Too many requests. Please retry shortly.",
                }
            ),
            status=429,
            mimetype="application/json",
            headers={"Retry-After": str(retry_after)},
        )
    )


def _check_endpoint_rate_limit() -> None:
    if not request.path.startswith("/api/"):
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
        Response(
            response=json.dumps(
                {
                    "error": "rate_limited",
                    "message": "Too many requests. Please retry shortly.",
                }
            ),
            status=429,
            mimetype="application/json",
            headers={"Retry-After": str(retry_after)},
        )
    )


@app.before_request
def _capture_request_start() -> None:
    g.request_start = time.perf_counter()


@app.before_request
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


@app.after_request
def _record_api_key_usage(response):
    ctx = _current_access_context()
    if ctx.tier != "api_key" or not ctx.api_key_id:
        return response
    if not request.path.startswith("/api/"):
        return response
    if request.path.startswith("/api/auth/"):
        return response

    if _auth_is_mocked():
        _mock_auth.record_usage(api_key_id=ctx.api_key_id)
        return response
    try:
        route = _api_route_template()
        if route is None:
            return response
        now = datetime.utcnow()
        today = now.date()
        hour = now.replace(minute=0, second=0, microsecond=0)
        status_code = int(response.status_code)
        status_class = status_code // 100
        elapsed_ms = 0
        start = getattr(g, "request_start", None)
        if isinstance(start, (int, float)):
            elapsed_ms = max(0, int((time.perf_counter() - start) * 1000))
        req_bytes = request.content_length
        req_bytes_int = int(req_bytes) if isinstance(req_bytes, int) else 0
        resp_bytes = response.content_length
        resp_bytes_int = int(resp_bytes) if isinstance(resp_bytes, int) else 0

        row = ApiUsageDaily.query.filter_by(api_key_id=ctx.api_key_id, day=today).first()
        if row is None:
            row = ApiUsageDaily(api_key_id=ctx.api_key_id, day=today, count=1)
            db.session.add(row)
        else:
            row.count = int(row.count) + 1

        hourly = ApiUsageHourly.query.filter_by(
            api_key_id=ctx.api_key_id,
            hour=hour,
            route=route,
            method=request.method,
            status_class=status_class,
        ).first()
        bucket_index = _latency_bucket_index(elapsed_ms)
        if hourly is None:
            buckets = _init_latency_buckets()
            buckets[bucket_index] = 1
            hourly = ApiUsageHourly(
                api_key_id=ctx.api_key_id,
                hour=hour,
                route=route,
                method=request.method,
                status_class=status_class,
                count=1,
                total_ms=elapsed_ms,
                max_ms=elapsed_ms,
                latency_buckets=buckets,
                request_bytes=req_bytes_int,
                response_bytes=resp_bytes_int,
            )
            db.session.add(hourly)
        else:
            hourly.count = int(hourly.count) + 1
            hourly.total_ms = int(hourly.total_ms) + elapsed_ms
            hourly.max_ms = max(int(hourly.max_ms), elapsed_ms)
            buckets = hourly.latency_buckets
            if not isinstance(buckets, list) or len(buckets) != len(_LATENCY_BUCKET_BOUNDS_MS) + 1:
                buckets = _init_latency_buckets()
            buckets[bucket_index] = int(buckets[bucket_index]) + 1
            hourly.latency_buckets = buckets
            hourly.request_bytes = int(hourly.request_bytes) + req_bytes_int
            hourly.response_bytes = int(hourly.response_bytes) + resp_bytes_int

        ip_hash = _ip_hash(_request_ip_address())
        if ip_hash is not None:
            existing_ip = ApiUsageDailyIp.query.filter_by(
                api_key_id=ctx.api_key_id, day=today, ip_hash=ip_hash
            ).first()
            if existing_ip is None:
                db.session.add(
                    ApiUsageDailyIp(
                        api_key_id=ctx.api_key_id,
                        day=today,
                        ip_hash=ip_hash,
                        first_seen_at=now,
                    )
                )

        sample_rate = _usage_event_sample_rate(status_code)
        if random.random() < sample_rate:
            user_agent = _request_user_agent()
            db.session.add(
                ApiRequestEvent(
                    api_key_id=ctx.api_key_id,
                    occurred_at=now,
                    route=route,
                    method=request.method,
                    status_code=status_code,
                    status_class=status_class,
                    latency_ms=elapsed_ms,
                    request_bytes=req_bytes if isinstance(req_bytes, int) else None,
                    response_bytes=resp_bytes if isinstance(resp_bytes, int) else None,
                    ip_hash=ip_hash,
                    user_agent=user_agent,
                )
            )
        db.session.commit()
    except SQLAlchemyError:
        return response
    return response


@app.after_request
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
    if request.path.startswith("/api/"):
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

# ── Reflect existing tables via standalone engine ─────────────────────────
_SKIP_MAIN_DB_REFLECTION = os.environ.get("SKIP_MAIN_DB_REFLECTION", "").strip() == "1"
metadata = MetaData()

if not _SKIP_MAIN_DB_REFLECTION:
    engine = create_engine(app.config["SQLALCHEMY_DATABASE_URI"])

    agreements_table = Table(
        "agreements",
        metadata,
        autoload_with=engine,
    )
    xml_table = Table(
        "xml",
        metadata,
        autoload_with=engine,
    )
    taxonomy_table = Table(
        "taxonomy",
        metadata,
        autoload_with=engine,
    )
else:
    # Test mode: avoid connecting to the main DB at import time.
    engine = None
    agreements_table = Table(
        "agreements",
        metadata,
        Column("uuid", CHAR(36), primary_key=True),
    )
    xml_table = Table(
        "xml",
        metadata,
        Column("agreement_uuid", CHAR(36), primary_key=True),
    )
    taxonomy_table = Table("taxonomy", metadata, Column("id", Integer, primary_key=True))

sections_table = Table(
    "sections",
    metadata,
    Column("agreement_uuid", CHAR(36), nullable=False),
    Column("section_uuid", CHAR(36), primary_key=True),
    Column("article_title", TEXT, nullable=False),
    Column("section_title", TEXT, nullable=False),
    Column("xml_content", LONGTEXT, nullable=False),
    Column("article_standard_id", TINYTEXT, nullable=False),
    Column("section_standard_id", TINYTEXT, nullable=False),
    schema="mna",
)


# ── SQLAlchemy models mapping ───────────────────────────────────────
class Sections(db.Model):
    __table__ = sections_table


class Agreements(db.Model):
    __table__ = agreements_table


class XML(db.Model):
    __table__ = xml_table


class Taxonomy(db.Model):
    __table__ = taxonomy_table


# ── Define search blueprint and schemas ──────────────────────────────────
search_blp = Blueprint(
    "search",
    "search",
    url_prefix="/api/search",
    description="Search merger agreement sections",
)

dumps_blp = Blueprint(
    "dumps",
    "dumps",
    url_prefix="/api/dumps",
    description="Access metadata about bulk data on Cloudflare",
)

agreements_blp = Blueprint(
    "agreements",
    "agreements",
    url_prefix="/api/agreements",
    description="Retrieve full text for a given agreement",
)


class SearchArgsSchema(Schema):
    year = fields.List(fields.Int(), load_default=[])
    target = fields.List(fields.Str(), load_default=[])
    acquirer = fields.List(fields.Str(), load_default=[])
    standardId = fields.List(fields.Str(), load_default=[])
    transactionSize = fields.List(fields.Str(), load_default=[])
    transactionType = fields.List(fields.Str(), load_default=[])
    considerationType = fields.List(fields.Str(), load_default=[])
    targetType = fields.List(fields.Str(), load_default=[])
    page = fields.Int(load_default=1)
    pageSize = fields.Int(load_default=25)


class SectionItemSchema(Schema):
    id = fields.Str()
    agreementUuid = fields.Str()
    sectionUuid = fields.Str()
    standardId = fields.Str(allow_none=True)
    xml = fields.Str()
    articleTitle = fields.Str()
    sectionTitle = fields.Str()
    acquirer = fields.Str()
    target = fields.Str()
    year = fields.Int()
    verified = fields.Bool()


class AccessInfoSchema(Schema):
    tier = fields.Str(required=True)
    message = fields.Str(required=False, allow_none=True)


class SearchResponseSchema(Schema):
    results = fields.List(fields.Nested(SectionItemSchema))
    access = fields.Nested(AccessInfoSchema)
    page = fields.Int()
    pageSize = fields.Int()
    totalCount = fields.Int()
    totalPages = fields.Int()
    hasNext = fields.Bool()
    hasPrev = fields.Bool()
    nextNum = fields.Int(allow_none=True)
    prevNum = fields.Int(allow_none=True)


class DumpEntrySchema(Schema):
    timestamp = fields.Str(required=True)
    sql = fields.Url(required=False, allow_none=True)
    sha256 = fields.Str(required=False, allow_none=True)
    sha256_url = fields.Url(required=False, allow_none=True)
    manifest = fields.Url(required=False, allow_none=True)
    size_bytes = fields.Int(required=False, allow_none=True)
    warning = fields.Str(required=False, allow_none=True)


class AgreementArgsSchema(Schema):
    focusSectionUuid = fields.Str(required=False, allow_none=True)
    neighborSections = fields.Int(load_default=1)


class AgreementResponseSchema(Schema):
    year = fields.Int()
    target = fields.Str()
    acquirer = fields.Str()
    url = fields.Str()
    xml = fields.Str()
    isRedacted = fields.Bool(required=False)


# ── Route definitions ───────────────────────────────────────

@agreements_blp.route("/<string:agreement_uuid>")
class AgreementResource(MethodView):
    @agreements_blp.arguments(AgreementArgsSchema, location="query")
    @agreements_blp.response(200, AgreementResponseSchema)
    def get(self, args, agreement_uuid):
        ctx = _current_access_context()
        focus_section_uuid = args.get("focusSectionUuid")
        if focus_section_uuid is not None:
            focus_section_uuid = focus_section_uuid.strip()
            if not _UUID_RE.match(focus_section_uuid):
                abort(400, description="Invalid focusSectionUuid.")
        neighbor_sections_int = args["neighborSections"]

        # query year, target, acquirer, xml, url for this agreement_uuid
        row = (
            db.session.query(
                Agreements.year,
                Agreements.target,
                Agreements.acquirer,
                Agreements.url,
                XML.xml,
            )
            .join(XML, XML.agreement_uuid == Agreements.uuid)
            .filter(Agreements.uuid == agreement_uuid)
            .first()
        )

        if row is None:
            abort(404)

        year, target, acquirer, url, xml_content = row
        if not ctx.is_authenticated:
            redacted_xml = _redact_agreement_xml(
                xml_content,
                focus_section_uuid=focus_section_uuid,
                neighbor_sections=neighbor_sections_int,
            )
            return {
                "year": year,
                "target": target,
                "acquirer": acquirer,
                "url": url,
                "xml": redacted_xml,
                "isRedacted": True,
            }
        return {
            "year": year,
            "target": target,
            "acquirer": acquirer,
            "url": url,
            "xml": xml_content,
        }


@app.route("/api/agreements-index", methods=["GET"])
def get_agreements_index():
    ctx = _current_access_context()
    page = request.args.get("page", default=1, type=int)
    page_size = request.args.get("pageSize", default=25, type=int)
    sort_by = request.args.get("sortBy", default="year", type=str)
    sort_dir = request.args.get("sortDir", default="desc", type=str)
    query = (request.args.get("query") or "").strip()

    if page < 1:
        page = 1

    max_page_size = 100 if ctx.is_authenticated else 10
    if page_size < 1 or page_size > max_page_size:
        page_size = min(25, max_page_size)

    sort_map = {
        "year": Agreements.year,
        "target": Agreements.target,
        "acquirer": Agreements.acquirer,
    }
    sort_column = sort_map.get(sort_by, Agreements.year)
    sort_direction = sort_dir.lower()
    order_by = sort_column.desc() if sort_direction == "desc" else sort_column.asc()

    q = db.session.query(
        Agreements.uuid,
        Agreements.year,
        Agreements.target,
        Agreements.acquirer,
        Agreements.verified,
    )

    if query:
        like = f"%{query}%"
        q = q.filter(
            or_(
                Agreements.year.ilike(like),
                Agreements.target.ilike(like),
                Agreements.acquirer.ilike(like),
            )
        )

    q = q.order_by(order_by, Agreements.uuid)

    try:
        paginated = q.paginate(page=page, per_page=page_size, error_out=False)
    except Exception:
        abort(400, description="Invalid pagination request.")

    results = [
        {
            "agreementUuid": row.uuid,
            "year": row.year,
            "target": row.target,
            "acquirer": row.acquirer,
            "considerationType": None,
            "totalConsideration": None,
            "targetIndustry": None,
            "acquirerIndustry": None,
            "verified": bool(row.verified) if row.verified is not None else False,
        }
        for row in paginated.items
    ]

    return {
        "results": results,
        "page": paginated.page,
        "pageSize": paginated.per_page,
        "totalCount": paginated.total,
        "totalPages": paginated.pages,
        "hasNext": paginated.has_next,
        "hasPrev": paginated.has_prev,
        "nextNum": paginated.next_num,
        "prevNum": paginated.prev_num,
    }


@app.route("/api/agreements-summary", methods=["GET"])
def get_agreements_summary():
    agreements_count = db.session.execute(
        text("SELECT COUNT(*) FROM mna.agreements")
    ).scalar()
    sections_count = db.session.execute(
        text("SELECT COUNT(*) FROM mna.sections")
    ).scalar()
    pages_count = db.session.execute(
        text("SELECT COUNT(*) FROM mna.pages")
    ).scalar()

    return {
        "agreements": int(agreements_count or 0),
        "sections": int(sections_count or 0),
        "pages": int(pages_count or 0),
    }


@app.route("/api/filter-options", methods=["GET"])
def get_filter_options():
    """Fetch distinct targets and acquirers from the database"""
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

    # Use EXISTS to avoid row explosion from joining into per-section tables.
    # This keeps filter options aligned with the agreements that actually have searchable sections.
    targets = [
        row[0]
        for row in db.session.execute(
            text(
                """
                SELECT DISTINCT a.target
                FROM mna.agreements a
                WHERE a.target IS NOT NULL
                  AND a.target <> ''
                  AND EXISTS (
                    SELECT 1
                    FROM mna.sections s
                    WHERE s.agreement_uuid = a.uuid
                  )
                ORDER BY a.target
                """
            )
        ).fetchall()
    ]
    acquirers = [
        row[0]
        for row in db.session.execute(
            text(
                """
                SELECT DISTINCT a.acquirer
                FROM mna.agreements a
                WHERE a.acquirer IS NOT NULL
                  AND a.acquirer <> ''
                  AND EXISTS (
                    SELECT 1
                    FROM mna.sections s
                    WHERE s.agreement_uuid = a.uuid
                  )
                ORDER BY a.acquirer
                """
            )
        ).fetchall()
    ]

    payload = {"targets": targets, "acquirers": acquirers}
    with _filter_options_lock:
        _filter_options_cache["payload"] = payload
        _filter_options_cache["ts"] = now

    resp = jsonify(payload)
    resp.headers["Cache-Control"] = f"public, max-age={_FILTER_OPTIONS_TTL_SECONDS}"
    return resp, 200


@search_blp.route("")
class SearchResource(MethodView):
    @search_blp.arguments(SearchArgsSchema, location="query")
    @search_blp.response(200, SearchResponseSchema)
    def get(self, args):
        ctx = _current_access_context()
        # @app.route("/api/search", methods=["GET"])
        # def search_sections():
        # pull in optional query params - now supporting multiple values
        years = args["year"]
        targets = args["target"]
        acquirers = args["acquirer"]
        standard_ids = args["standardId"]
        transaction_sizes = args["transactionSize"]
        transaction_types = args["transactionType"]
        consideration_types = args["considerationType"]
        target_types = args["targetType"]

        # pagination parameters
        page = args["page"]
        page_size = args["pageSize"]

        # Validate pagination parameters
        if page < 1:
            page = 1
        max_page_size = 100 if ctx.is_authenticated else 10
        if page_size < 1 or page_size > max_page_size:
            page_size = min(25, max_page_size)

        # build the base ORM query
        q = db.session.query(
            Sections.section_uuid,
            Sections.agreement_uuid,
            Sections.section_standard_id,
            Sections.xml_content,
            Sections.article_title,
            Sections.section_title,
            Agreements.acquirer,
            Agreements.target,
            Agreements.year,
            Agreements.verified,
        ).join(Agreements, Sections.agreement_uuid == Agreements.uuid)

        # apply filters only when provided - now handling multiple values
        if years:
            q = q.filter(Agreements.year.in_(years))

        if targets:
            q = q.filter(Agreements.target.in_(targets))

        if acquirers:
            q = q.filter(Agreements.acquirer.in_(acquirers))

        if standard_ids:
            q = (
                q.join(Taxonomy, Sections.section_standard_id == Taxonomy.standard_id)
                .filter(Taxonomy.type == "section")
                .filter(Taxonomy.standard_id.in_(standard_ids))
            )

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
            # Convert frontend values to DB enum values
            db_target_types = []
            for t_type in target_types:
                if t_type == "Public":
                    db_target_types.append("public")
                elif t_type == "Private":
                    db_target_types.append("private")
            if db_target_types:
                q = q.filter(Agreements.target_type.in_(db_target_types))

        # Use SQLAlchemy's paginate() method
        try:
            paginated = q.paginate(page=page, per_page=page_size, error_out=False)
        except Exception:
            abort(400, description="Invalid pagination request.")

        # marshal into JSON with pagination metadata
        results = [
            {
                "id": r.section_uuid,
                "agreementUuid": r.agreement_uuid,
                "sectionUuid": r.section_uuid,
                "standardId": r.section_standard_id,
                "xml": r.xml_content,
                "articleTitle": r.article_title,
                "sectionTitle": r.section_title,
                "acquirer": r.acquirer,
                "target": r.target,
                "year": r.year,
                "verified": r.verified,
            }
            for r in paginated.items
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
            "page": paginated.page,
            "pageSize": paginated.per_page,
            "totalCount": paginated.total,
            "totalPages": paginated.pages,
            "hasNext": paginated.has_next,
            "hasPrev": paginated.has_prev,
            "nextNum": paginated.next_num,
            "prevNum": paginated.prev_num,
        }


# Register search blueprint
api.register_blueprint(search_blp)


@dumps_blp.route("")  # blueprint already has url_prefix="/api/dumps"
class DumpListResource(MethodView):
    @dumps_blp.response(200, DumpEntrySchema(many=True))
    def get(self):
        if client is None:
            return []
        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=R2_BUCKET_NAME, Prefix="dumps/")

        dumps_map = defaultdict(dict)
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                filename = key.rsplit("/", 1)[-1]

                if filename.endswith(".sql.gz.manifest.json"):
                    prefix = filename[: -len(".sql.gz.manifest.json")]
                    dumps_map[prefix]["manifest"] = key

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
                try:
                    body = client.get_object(
                        Bucket=R2_BUCKET_NAME, Key=files["manifest"]
                    )["Body"].read()
                    data = json.loads(body)
                    if "size_bytes" in data:
                        entry["size_bytes"] = data["size_bytes"]
                    if "sha256" in data:
                        entry["sha256"] = data["sha256"]
                except Exception as e:
                    entry["warning"] = f"couldn't read manifest: {e}"

            dump_list.append(entry)

        return dump_list


# Register dumps blueprint
api.register_blueprint(dumps_blp)

# Register agreements blueprint
api.register_blueprint(agreements_blp)

# ── Auth routes ──────────────────────────────────────────────────────────


@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    _require_auth_db()
    data = _require_json()
    checked_at = _require_legal_acceptance(data)
    if _turnstile_enabled():
        captcha_token = _require_captcha_token(data)
        _verify_turnstile_token(token=captcha_token)
    email_raw = data.get("email")
    password = data.get("password")
    if not isinstance(email_raw, str) or not isinstance(password, str):
        abort(400, description="Email and password are required.")
    email = _normalize_email(email_raw)
    if not _is_email_like(email):
        abort(400, description="Invalid email address.")
    if len(password) < 8:
        abort(400, description="Password must be at least 8 characters.")

    if _auth_is_mocked():
        existing = _mock_auth.get_user_by_email(email)
        user = existing or _mock_auth.create_user(email=email, password=password)
        verify_token = None
        if user.email_verified_at is None:
            verify_token = _issue_email_verification_token(user_id=user.id, email=user.email)
        payload: dict[str, object] = {
            "status": "verification_required",
            "user": {"id": user.id, "email": user.email, "createdAt": user.created_at.isoformat()},
        }
        if (
            verify_token
            and os.environ.get("EMAIL_VERIFICATION_DEBUG_TOKEN", "").strip() == "1"
            and app.debug
        ):
            payload["debugToken"] = verify_token
        resp = make_response(jsonify(payload), 201)
        resp.headers["Cache-Control"] = "no-store"
        _clear_auth_cookies(resp)
        return resp

    try:
        existing = AuthUser.query.filter_by(email=email).first()
        if existing is not None:
            verify_token = None
            if existing.email_verified_at is None:
                verify_token = _issue_email_verification_token(
                    user_id=existing.id, email=existing.email
                )
                _send_email_verification_email(to_email=existing.email, token=verify_token)
            payload: dict[str, object] = {
                "status": "verification_required",
                "user": {
                    "id": existing.id,
                    "email": existing.email,
                    "createdAt": existing.created_at.isoformat(),
                },
            }
            if (
                verify_token
                and os.environ.get("EMAIL_VERIFICATION_DEBUG_TOKEN", "").strip() == "1"
                and app.debug
            ):
                payload["debugToken"] = verify_token
            _auth_enumeration_delay()
            resp = make_response(jsonify(payload), 201)
            resp.headers["Cache-Control"] = "no-store"
            _clear_auth_cookies(resp)
            return resp

        now = datetime.utcnow()
        ip_address = _request_ip_address()
        user_agent = _request_user_agent()
        user = AuthUser(
            email=email,
            password_hash=generate_password_hash(password),
            email_verified_at=None,
        )
        db.session.add(user)
        db.session.flush()
        for doc, meta in _LEGAL_DOCS.items():
            db.session.add(
                LegalAcceptance(
                    user_id=user.id,
                    document=doc,
                    version=meta["version"],
                    document_hash=meta["sha256"],
                    checked_at=checked_at,
                    submitted_at=now,
                    ip_address=ip_address,
                    user_agent=user_agent,
                )
            )
        _record_signon_event(user_id=user.id, provider="email", action="register")
        verify_token = _issue_email_verification_token(user_id=user.id, email=user.email)
        _send_email_verification_email(to_email=user.email, token=verify_token)
        _send_signup_notification_email(new_user_email=user.email)
        db.session.commit()
    except HTTPException:
        db.session.rollback()
        raise
    except SQLAlchemyError:
        db.session.rollback()
        abort(503, description="Auth backend is unavailable right now.")

    payload: dict[str, object] = {
        "status": "verification_required",
        "user": {"id": user.id, "email": user.email, "createdAt": user.created_at.isoformat()},
    }
    if os.environ.get("EMAIL_VERIFICATION_DEBUG_TOKEN", "").strip() == "1" and app.debug:
        payload["debugToken"] = verify_token
    resp = make_response(jsonify(payload), 201)
    resp.headers["Cache-Control"] = "no-store"
    _clear_auth_cookies(resp)
    return resp


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    _require_auth_db()
    data = _require_json()
    email_raw = data.get("email")
    password = data.get("password")
    if not isinstance(email_raw, str) or not isinstance(password, str):
        abort(400, description="Email and password are required.")
    email = _normalize_email(email_raw)

    if _auth_is_mocked():
        user = _mock_auth.authenticate(email=email, password=password)
        if user is None:
            _auth_enumeration_delay()
            abort(401, description="Invalid credentials.")
        if user.email_verified_at is None:
            abort(403, description="Email address not verified.")
        token = _issue_session_token(user.id)
        payload: dict[str, object] = {"user": {"id": user.id, "email": user.email}}
        if _auth_session_transport() == "bearer":
            payload["sessionToken"] = token
        resp = make_response(jsonify(payload))
        resp.headers["Cache-Control"] = "no-store"
        if _auth_session_transport() == "cookie":
            _set_auth_cookies(resp, session_token=token)
        return resp

    try:
        user = AuthUser.query.filter_by(email=email).first()
        if user is None or not user.password_hash:
            _auth_enumeration_delay()
            abort(401, description="Invalid credentials.")
        if not check_password_hash(user.password_hash, password):
            _auth_enumeration_delay()
            abort(401, description="Invalid credentials.")
        if user.email_verified_at is None:
            abort(403, description="Email address not verified.")

        _record_signon_event(user_id=user.id, provider="email", action="login")
        db.session.commit()
        token = _issue_session_token(user.id)
        payload: dict[str, object] = {"user": {"id": user.id, "email": user.email}}
        if _auth_session_transport() == "bearer":
            payload["sessionToken"] = token
        resp = make_response(jsonify(payload))
        resp.headers["Cache-Control"] = "no-store"
        if _auth_session_transport() == "cookie":
            _set_auth_cookies(resp, session_token=token)
        return resp
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")


@app.route("/api/auth/email/resend", methods=["POST"])
def auth_resend_email_verification():
    _require_auth_db()
    data = _require_json()
    email_raw = data.get("email")
    if not isinstance(email_raw, str) or not email_raw.strip():
        abort(400, description="Email is required.")
    email = _normalize_email(email_raw)
    if not _is_email_like(email):
        abort(400, description="Invalid email address.")

    if _auth_is_mocked():
        user = _mock_auth.get_user_by_email(email)
        if user is not None and user.email_verified_at is None:
            verify_token = _issue_email_verification_token(user_id=user.id, email=user.email)
            _send_email_verification_email(to_email=user.email, token=verify_token)
        _auth_enumeration_delay()
        resp = make_response(jsonify({"status": "sent"}))
        resp.headers["Cache-Control"] = "no-store"
        return resp

    try:
        user = AuthUser.query.filter_by(email=email).first()
        if user is not None and user.email_verified_at is None:
            verify_token = _issue_email_verification_token(user_id=user.id, email=user.email)
            _send_email_verification_email(to_email=user.email, token=verify_token)
    except HTTPException:
        db.session.rollback()
        raise
    except SQLAlchemyError:
        db.session.rollback()
        abort(503, description="Auth backend is unavailable right now.")

    _auth_enumeration_delay()
    resp = make_response(jsonify({"status": "sent"}))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/auth/password/forgot", methods=["POST"])
def auth_password_forgot():
    _require_auth_db()
    data = _require_json()
    email_raw = data.get("email")
    if not isinstance(email_raw, str) or not email_raw.strip():
        abort(400, description="Email is required.")
    email = _normalize_email(email_raw)
    if not _is_email_like(email):
        abort(400, description="Invalid email address.")

    if _auth_is_mocked():
        user = _mock_auth.get_user_by_email(email)
        if user is not None and not (
            user.email.startswith("deleted+") and user.email.endswith("@deleted.invalid")
        ):
            token = _issue_password_reset_token(user_id=user.id, email=user.email)
            _send_password_reset_email(to_email=user.email, token=token)
        _auth_enumeration_delay()
        resp = make_response(jsonify({"status": "sent"}))
        resp.headers["Cache-Control"] = "no-store"
        return resp

    try:
        user = AuthUser.query.filter_by(email=email).first()
        if user is not None and not (
            user.email.startswith("deleted+") and user.email.endswith("@deleted.invalid")
        ):
            token = _issue_password_reset_token(user_id=user.id, email=user.email)
            _send_password_reset_email(to_email=user.email, token=token)
    except HTTPException:
        db.session.rollback()
        raise
    except SQLAlchemyError:
        db.session.rollback()
        abort(503, description="Auth backend is unavailable right now.")

    _auth_enumeration_delay()
    resp = make_response(jsonify({"status": "sent"}))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/auth/password/reset", methods=["POST"])
def auth_password_reset():
    _require_auth_db()
    data = _require_json()
    token = data.get("token")
    password = data.get("password")
    if not isinstance(token, str) or not token.strip():
        abort(400, description="Missing reset token.")
    if not isinstance(password, str):
        abort(400, description="Password is required.")
    if len(password) < 8:
        abort(400, description="Password must be at least 8 characters.")

    if _auth_is_mocked():
        parsed = _load_password_reset_token(token.strip())
        if parsed is None:
            abort(400, description="Invalid or expired reset token.")
        user_id, email, _row = parsed
        user = _mock_auth.get_user(user_id)
        if user is None or user.email != email:
            abort(400, description="Invalid or expired reset token.")
        if not _mock_auth.set_user_password(user_id=user_id, password=password):
            abort(400, description="Invalid or expired reset token.")
        resp = make_response(jsonify({"status": "ok"}))
        resp.headers["Cache-Control"] = "no-store"
        _clear_auth_cookies(resp)
        return resp

    try:
        parsed = _load_password_reset_token(token.strip())
        if parsed is None:
            abort(400, description="Invalid or expired reset token.")
        user_id, email, row = parsed
        user = db.session.get(AuthUser, user_id)
        if user is None or user.email != email:
            abort(400, description="Invalid or expired reset token.")
        if user.email.startswith("deleted+") and user.email.endswith("@deleted.invalid"):
            abort(400, description="Invalid or expired reset token.")
        user.password_hash = generate_password_hash(password)
        if user.email_verified_at is None:
            user.email_verified_at = datetime.utcnow()
        now = datetime.utcnow()
        if row is not None:
            row.used_at = now
        AuthSession.query.filter_by(user_id=user.id, revoked_at=None).update(
            {"revoked_at": now}, synchronize_session=False
        )
        db.session.commit()
    except HTTPException:
        db.session.rollback()
        raise
    except SQLAlchemyError:
        db.session.rollback()
        abort(503, description="Auth backend is unavailable right now.")

    resp = make_response(jsonify({"status": "ok"}))
    resp.headers["Cache-Control"] = "no-store"
    _clear_auth_cookies(resp)
    return resp


@app.route("/api/auth/email/verify", methods=["POST"])
def auth_verify_email():
    _require_auth_db()
    data = _require_json()
    token = data.get("token")
    if not isinstance(token, str) or not token.strip():
        abort(400, description="Missing verification token.")
    parsed = _load_email_verification_token(token.strip())
    if parsed is None:
        abort(400, description="Invalid or expired verification token.")
    user_id, email = parsed

    if _auth_is_mocked():
        user = _mock_auth.get_user(user_id)
        if user is None or user.email != email:
            abort(400, description="Invalid verification token.")
        _mock_auth.mark_email_verified(user_id)
        resp = make_response(jsonify({"status": "ok"}))
        resp.headers["Cache-Control"] = "no-store"
        return resp

    try:
        user = db.session.get(AuthUser, user_id)
        if user is None or user.email != email:
            abort(400, description="Invalid verification token.")
        if user.email.startswith("deleted+") and user.email.endswith("@deleted.invalid"):
            abort(400, description="Invalid verification token.")
        if user.email_verified_at is None:
            user.email_verified_at = datetime.utcnow()
            db.session.commit()
    except HTTPException:
        db.session.rollback()
        raise
    except SQLAlchemyError:
        db.session.rollback()
        abort(503, description="Auth backend is unavailable right now.")

    resp = make_response(jsonify({"status": "ok"}))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/auth/email/verify", methods=["GET"])
def auth_verify_email_get():
    _require_auth_db()
    token = request.args.get("token")
    if not isinstance(token, str) or not token.strip():
        abort(400, description="Missing verification token.")
    parsed = _load_email_verification_token(token.strip())
    if parsed is None:
        abort(400, description="Invalid or expired verification token.")
    user_id, email = parsed

    if _auth_is_mocked():
        user = _mock_auth.get_user(user_id)
        if user is None or user.email != email:
            abort(400, description="Invalid verification token.")
        _mock_auth.mark_email_verified(user_id)
    else:
        try:
            user = db.session.get(AuthUser, user_id)
            if user is None or user.email != email:
                abort(400, description="Invalid verification token.")
            if user.email.startswith("deleted+") and user.email.endswith("@deleted.invalid"):
                abort(400, description="Invalid verification token.")
            if user.email_verified_at is None:
                user.email_verified_at = datetime.utcnow()
                db.session.commit()
        except HTTPException:
            db.session.rollback()
            raise
        except SQLAlchemyError:
            db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")

    dest = f"{_frontend_base_url()}/account?emailVerified=1"
    resp = redirect(dest, code=303)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    _require_auth_db()
    user, _ctx = _require_verified_user()
    resp = make_response(jsonify({"user": {"id": user.id, "email": user.email}}))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/auth/csrf", methods=["GET"])
def auth_csrf():
    _require_auth_db()
    if _auth_session_transport() == "cookie":
        existing = _csrf_cookie_value()
        token = existing or secrets.token_urlsafe(32)
        resp = make_response(jsonify({"status": "ok", "csrfToken": token}))
        resp.headers["Cache-Control"] = "no-store"
        if existing is None:
            _set_csrf_cookie(resp, token, max_age=60 * 60 * 24 * 14)
        return resp
    resp = make_response(jsonify({"status": "ok"}))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/auth/health", methods=["GET"])
def auth_health():
    if _auth_is_mocked():
        resp = make_response(jsonify({"status": "ok"}))
        resp.headers["Cache-Control"] = "no-store"
        return resp
    if not _auth_db_is_configured():
        abort(
            503,
            description=(
                "Auth is not configured (missing AUTH_DATABASE_URI / DATABASE_URL). "
                "Search is available in limited mode."
            ),
        )
    engine = db.engines.get("auth")
    if engine is None:
        abort(503, description="Auth backend is unavailable right now.")
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")
    resp = make_response(jsonify({"status": "ok"}))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/auth/api-keys", methods=["GET"])
def auth_list_api_keys():
    _require_auth_db()
    user, _ctx = _require_verified_user()
    if _auth_is_mocked():
        keys = _mock_auth.list_api_keys(user_id=user.id)
        resp = make_response(
            jsonify(
            {
                "keys": [
                    {
                        "id": k.id,
                        "name": k.name,
                        "prefix": k.prefix,
                        "createdAt": k.created_at.isoformat(),
                        "lastUsedAt": k.last_used_at.isoformat() if k.last_used_at else None,
                        "revokedAt": k.revoked_at.isoformat() if k.revoked_at else None,
                    }
                    for k in keys
                ]
            }
            )
        )
        resp.headers["Cache-Control"] = "no-store"
        return resp
    try:
        keys = (
            ApiKey.query.filter_by(user_id=user.id)
            .order_by(ApiKey.created_at.desc())
            .all()
        )
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")
    resp = make_response(
        jsonify(
            {
                "keys": [
                    {
                        "id": k.id,
                        "name": k.name,
                        "prefix": k.prefix,
                        "createdAt": k.created_at.isoformat(),
                        "lastUsedAt": k.last_used_at.isoformat() if k.last_used_at else None,
                        "revokedAt": k.revoked_at.isoformat() if k.revoked_at else None,
                    }
                    for k in keys
                ]
            }
        )
    )
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/auth/api-keys", methods=["POST"])
def auth_create_api_key():
    _require_auth_db()
    user, _ctx = _require_verified_user()
    data = _require_json()
    name = data.get("name")
    if name is not None and not isinstance(name, str):
        abort(400, description="Key name must be a string.")
    if isinstance(name, str):
        name = name.strip() or None
        if name is not None and len(name) > 120:
            abort(400, description="Key name is too long.")
    if _auth_is_mocked():
        key, plaintext = _mock_auth.create_api_key(user_id=user.id, name=name)
        resp = make_response(
            jsonify(
            {
                "apiKey": {
                    "id": key.id,
                    "name": key.name,
                    "prefix": key.prefix,
                    "createdAt": key.created_at.isoformat(),
                },
                "apiKeyPlaintext": plaintext,
            }
            )
        )
        resp.headers["Cache-Control"] = "no-store"
        return resp
    try:
        key, plaintext = _create_api_key(user_id=user.id, name=name)
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")
    resp = make_response(
        jsonify(
            {
                "apiKey": {
                    "id": key.id,
                    "name": key.name,
                    "prefix": key.prefix,
                    "createdAt": key.created_at.isoformat(),
                },
                "apiKeyPlaintext": plaintext,
            }
        )
    )
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/auth/api-keys/<string:key_id>", methods=["DELETE"])
def auth_revoke_api_key(key_id: str):
    _require_auth_db()
    user, _ctx = _require_verified_user()
    if not _UUID_RE.match(key_id):
        abort(404)
    if _auth_is_mocked():
        if not _mock_auth.revoke_api_key(user_id=user.id, key_id=key_id):
            abort(404)
        resp = make_response(jsonify({"status": "revoked"}))
        resp.headers["Cache-Control"] = "no-store"
        return resp
    try:
        key = ApiKey.query.filter_by(id=key_id, user_id=user.id).first()
        if key is None:
            abort(404)
        if key.revoked_at is None:
            key.revoked_at = datetime.utcnow()
            db.session.commit()
        resp = make_response(jsonify({"status": "revoked"}))
        resp.headers["Cache-Control"] = "no-store"
        return resp
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")


@app.route("/api/auth/usage", methods=["GET"])
def auth_usage():
    _require_auth_db()
    user, _ctx = _require_verified_user()
    if _auth_is_mocked():
        by_day, total = _mock_auth.usage_for_user(user_id=user.id)
        resp = make_response(jsonify({"byDay": by_day, "total": total}))
        resp.headers["Cache-Control"] = "no-store"
        return resp
    cutoff = _utc_today() - timedelta(days=29)
    try:
        key_ids = [k.id for k in ApiKey.query.filter_by(user_id=user.id).all()]
        if not key_ids:
            return jsonify({"byDay": [], "total": 0})

        rows = (
            ApiUsageDaily.query.filter(ApiUsageDaily.api_key_id.in_(key_ids))
            .filter(ApiUsageDaily.day >= cutoff)
            .order_by(ApiUsageDaily.day.asc())
            .all()
        )
        by_day: dict[str, int] = defaultdict(int)
        total = 0
        for row in rows:
            day_str = row.day.isoformat()
            by_day[day_str] += int(row.count)
            total += int(row.count)

        resp = make_response(
            jsonify(
                {
                    "byDay": [{"day": day, "count": by_day[day]} for day in sorted(by_day)],
                    "total": total,
                }
            )
        )
        resp.headers["Cache-Control"] = "no-store"
        return resp
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")


@app.route("/api/auth/account/delete", methods=["POST"])
def auth_delete_account():
    _require_auth_db()
    user, _ctx = _require_verified_user()
    data = _require_json()
    confirm = data.get("confirm")
    if confirm != "Delete":
        abort(400, description='Type "Delete" to confirm.')

    if _auth_is_mocked():
        abort(501, description="Account deletion is unavailable in mock auth mode.")

    try:
        now = datetime.utcnow()
        tombstone = f"deleted+{uuid.uuid4().hex}@deleted.invalid"
        user.email = tombstone
        user.password_hash = None
        AuthSession.query.filter_by(user_id=user.id, revoked_at=None).update(
            {"revoked_at": now}, synchronize_session=False
        )
        ApiKey.query.filter_by(user_id=user.id, revoked_at=None).update(
            {"revoked_at": now}, synchronize_session=False
        )
        db.session.commit()
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")

    resp = make_response(jsonify({"status": "deleted"}))
    resp.headers["Cache-Control"] = "no-store"
    _clear_auth_cookies(resp)
    return resp


@app.route("/api/auth/google/start", methods=["GET"])
def auth_google_start():
    _require_auth_db()
    if _auth_is_mocked():
        abort(501, description="Google auth is unavailable in mock auth mode.")
    if not _google_oauth_flow_enabled():
        abort(404)

    client_id = _google_oauth_client_id()
    redirect_uri = _google_oauth_redirect_uri()

    next_path = _safe_next_path(request.args.get("next")) or "/account"
    state = secrets.token_urlsafe(32)
    code_verifier, code_challenge = _google_oauth_pkce_pair()
    nonce = secrets.token_urlsafe(32)
    cookie_payload = {
        "state": state,
        "code_verifier": code_verifier,
        "nonce": nonce,
        "next": next_path,
    }

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "nonce": nonce,
    }
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    resp = redirect(auth_url)
    resp.headers["Cache-Control"] = "no-store"
    _set_google_oauth_cookie(resp, cookie_payload)
    return resp


@app.route("/api/auth/google/client-id", methods=["GET"])
def auth_google_client_id():
    _require_auth_db()
    if _auth_is_mocked():
        abort(501, description="Google auth is unavailable in mock auth mode.")
    nonce = secrets.token_urlsafe(32)
    resp = make_response(jsonify({"clientId": _google_oauth_client_id(), "nonce": nonce}))
    resp.headers["Cache-Control"] = "no-store"
    _set_google_nonce_cookie(resp, nonce)
    return resp


@app.route("/api/auth/captcha/site-key", methods=["GET"])
def auth_captcha_site_key():
    _require_auth_db()
    if not _turnstile_enabled():
        payload: dict[str, object] = {"enabled": False}
        if app.debug:
            payload["debug"] = {
                "TURNSTILE_ENABLED": os.environ.get("TURNSTILE_ENABLED"),
                "has_site_key": bool(os.environ.get("TURNSTILE_SITE_KEY", "").strip()),
                "has_secret_key": bool(os.environ.get("TURNSTILE_SECRET_KEY", "").strip()),
            }
        resp = make_response(jsonify(payload))
        resp.headers["Cache-Control"] = "no-store"
        return resp
    payload: dict[str, object] = {"enabled": True, "siteKey": _turnstile_site_key()}
    if app.debug:
        payload["debug"] = {
            "TURNSTILE_ENABLED": os.environ.get("TURNSTILE_ENABLED"),
            "has_site_key": True,
            "has_secret_key": bool(os.environ.get("TURNSTILE_SECRET_KEY", "").strip()),
        }
    resp = make_response(jsonify(payload))
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/auth/google/callback", methods=["GET"])
def auth_google_callback():
    _require_auth_db()
    if _auth_is_mocked():
        abort(501, description="Google auth is unavailable in mock auth mode.")
    if not _google_oauth_flow_enabled():
        abort(404)

    error = request.args.get("error")
    if isinstance(error, str) and error.strip():
        return _frontend_google_callback_redirect(token=None, next_path="/account", error=error)

    state = request.args.get("state")
    code = request.args.get("code")
    if not isinstance(state, str) or not state.strip():
        return _frontend_google_callback_redirect(
            token=None, next_path="/account", error="missing_state"
        )
    if not isinstance(code, str) or not code.strip():
        return _frontend_google_callback_redirect(
            token=None, next_path="/account", error="missing_code"
        )

    cookie_payload = _load_google_oauth_cookie()
    if not cookie_payload:
        return _frontend_google_callback_redirect(
            token=None, next_path="/account", error="invalid_state"
        )

    expected_state = cookie_payload.get("state")
    if not isinstance(expected_state, str) or not expected_state.strip():
        return _frontend_google_callback_redirect(
            token=None, next_path="/account", error="invalid_state"
        )
    if not secrets.compare_digest(expected_state, state):
        return _frontend_google_callback_redirect(
            token=None, next_path="/account", error="invalid_state"
        )

    code_verifier = cookie_payload.get("code_verifier")
    nonce = cookie_payload.get("nonce")
    next_path = _safe_next_path(cookie_payload.get("next")) if cookie_payload else None
    if not isinstance(code_verifier, str) or not code_verifier.strip():
        return _frontend_google_callback_redirect(
            token=None, next_path=next_path or "/account", error="invalid_state"
        )
    if not isinstance(nonce, str) or not nonce.strip():
        return _frontend_google_callback_redirect(
            token=None, next_path=next_path or "/account", error="invalid_state"
        )

    token_payload = _google_fetch_json(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": _google_oauth_client_id(),
            "client_secret": _google_oauth_client_secret(),
            "redirect_uri": _google_oauth_redirect_uri(),
            "grant_type": "authorization_code",
            "code_verifier": code_verifier,
        },
    )

    id_token = token_payload.get("id_token")
    if not isinstance(id_token, str) or not id_token.strip():
        return _frontend_google_callback_redirect(
            token=None, next_path=next_path or "/account", error="missing_id_token"
        )

    try:
        normalized = _google_verify_id_token(id_token, expected_nonce=nonce)
    except HTTPException as e:
        return _frontend_google_callback_redirect(
            token=None, next_path=next_path or "/account", error=f"google_{e.code}"
        )

    try:
        user = AuthUser.query.filter_by(email=normalized).first()
        if user is None:
            return _frontend_google_callback_redirect(
                token=None,
                next_path=next_path or "/account",
                error="legal_required",
            )
        if not _user_has_current_legal_acceptances(user_id=user.id):
            return _frontend_google_callback_redirect(
                token=None,
                next_path=next_path or "/account",
                error="legal_required",
            )
        if user.email_verified_at is None:
            user.email_verified_at = datetime.utcnow()
        _record_signon_event(user_id=user.id, provider="google", action="login")
        db.session.commit()
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")

    token = _issue_session_token(user.id)
    if _auth_session_transport() == "cookie":
        dest = f"{_frontend_base_url()}{(next_path or '/account')}"
        resp = redirect(dest)
        resp.headers["Cache-Control"] = "no-store"
        _set_auth_cookies(resp, session_token=token)
        _clear_google_oauth_cookie(resp)
        return resp
    return _frontend_google_callback_redirect(
        token=token, next_path=next_path or "/account", error=None
    )


@app.route("/api/auth/google/credential", methods=["POST"])
def auth_google_credential():
    _require_auth_db()
    if _auth_is_mocked():
        abort(501, description="Google auth is unavailable in mock auth mode.")
    data = _require_json()
    credential = data.get("credential")
    if not isinstance(credential, str) or not credential.strip():
        abort(400, description="Missing Google credential.")

    expected_nonce = _google_nonce_cookie_value()
    if not expected_nonce:
        abort(400, description="Missing Google nonce.")
    normalized = _google_verify_id_token(credential, expected_nonce=expected_nonce)

    try:
        user = AuthUser.query.filter_by(email=normalized).first()
        if user is None:
            checked_at = _require_legal_acceptance(data)
            now = datetime.utcnow()
            ip_address = _request_ip_address()
            user_agent = _request_user_agent()
            user = AuthUser(email=normalized, password_hash=None, email_verified_at=now)
            db.session.add(user)
            db.session.flush()
            for doc, meta in _LEGAL_DOCS.items():
                db.session.add(
                    LegalAcceptance(
                        user_id=user.id,
                        document=doc,
                        version=meta["version"],
                        document_hash=meta["sha256"],
                        checked_at=checked_at,
                        submitted_at=now,
                        ip_address=ip_address,
                        user_agent=user_agent,
                    )
                )
            _record_signon_event(user_id=user.id, provider="google", action="register")
            _send_signup_notification_email(new_user_email=user.email)
            db.session.commit()
        elif not _user_has_current_legal_acceptances(user_id=user.id):
            checked_at = _require_legal_acceptance(data)
            _ensure_current_legal_acceptances(user_id=user.id, checked_at=checked_at)
            if user.email_verified_at is None:
                user.email_verified_at = datetime.utcnow()
            _record_signon_event(user_id=user.id, provider="google", action="login")
            db.session.commit()
        else:
            if user.email_verified_at is None:
                user.email_verified_at = datetime.utcnow()
            _record_signon_event(user_id=user.id, provider="google", action="login")
            db.session.commit()
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")

    token = _issue_session_token(user.id)
    payload: dict[str, object] = {"user": {"id": user.id, "email": user.email}}
    if _auth_session_transport() == "bearer":
        payload["sessionToken"] = token
    resp = make_response(jsonify(payload))
    resp.headers["Cache-Control"] = "no-store"
    if _auth_session_transport() == "cookie":
        _set_auth_cookies(resp, session_token=token)
    _clear_google_nonce_cookie(resp)
    return resp


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    _require_auth_db()
    resp = make_response(jsonify({"status": "ok"}))
    resp.headers["Cache-Control"] = "no-store"
    token = None
    if _auth_session_transport() == "cookie":
        cookie_token = request.cookies.get(_SESSION_COOKIE_NAME)
        token = cookie_token.strip() if isinstance(cookie_token, str) else None
    else:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.startswith("Bearer "):
            token = auth_header.removeprefix("Bearer ").strip()
    if token:
        _revoke_session_token(token)
    _clear_auth_cookies(resp)
    return resp


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
