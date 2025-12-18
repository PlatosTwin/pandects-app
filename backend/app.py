import os
from pathlib import Path
import time
from threading import Lock
from datetime import datetime, date, timedelta
import uuid
import re
import click
import json
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
    func,
    desc,
    Column,
    CHAR,
    Integer,
    TEXT,
    text,
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

# ── Flask setup ───────────────────────────────────────────────────────────
app = Flask(__name__)

# ── CORS origins ──────────────────────────────────────────────────────────
_DEFAULT_CORS_ORIGINS = (
    "http://localhost:8080",
    "http://127.0.0.1:8080",
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


def _auth_session_transport() -> str:
    raw = os.environ.get("AUTH_SESSION_TRANSPORT", "").strip().lower()
    if raw in ("cookie", "bearer"):
        return raw
    return "cookie" if _is_running_on_fly() else "bearer"


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
    samesite, secure = _cookie_settings()
    resp.set_cookie(
        _CSRF_COOKIE_NAME,
        csrf_token,
        max_age=max_age,
        httponly=False,
        secure=secure,
        samesite=samesite.capitalize() if samesite != "none" else "None",
        path="/",
    )


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
        "/api/auth/logout",
    )


def _auth_is_mocked() -> bool:
    return AUTH_MODE == "mock" and bool(app.debug)


# ── Auth DB configuration (separate DB; local sqlite placeholder by default) ──

def _normalize_database_uri(uri: str) -> str:
    normalized = uri.strip()
    if normalized.startswith("postgres://"):
        normalized = f"postgresql://{normalized[len('postgres://'):]}"
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
R2_ENDPOINT = "https://34730161d8a80dadcd289d6774ffff3d.r2.cloudflarestorage.com"
PUBLIC_DEV_BASE = "https://pub-d1f4ad8b64bd4b89a2d5c5ab58a4ebdf.r2.dev"

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
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


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
        self._api_keys_by_id: dict[str, _MockApiKey] = {}
        self._api_keys_by_prefix: dict[str, list[str]] = defaultdict(list)
        self._usage_daily: dict[tuple[str, date], int] = defaultdict(int)

    def create_user(self, *, email: str, password: str) -> _MockAuthUser:
        user_id = str(uuid.uuid4())
        user = _MockAuthUser(
            id=user_id,
            email=email,
            password_hash=generate_password_hash(password),
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

    def get_user(self, user_id: str) -> _MockAuthUser | None:
        with self._lock:
            return self._users_by_id.get(user_id)

    def issue_session_token(self, *, user_id: str) -> str:
        token = f"mock_{uuid.uuid4().hex}{uuid.uuid4().hex}"
        with self._lock:
            self._tokens[token] = user_id
        return token

    def load_session_token(self, token: str) -> str | None:
        with self._lock:
            return self._tokens.get(token)

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
        for candidate in candidates:
            if candidate.revoked_at is not None:
                continue
            if check_password_hash(candidate.key_hash, raw_key):
                candidate.last_used_at = datetime.utcnow()
                return candidate
        return None

    def record_usage(self, *, api_key_id: str) -> None:
        with self._lock:
            self._usage_daily[(api_key_id, date.today())] += 1

    def usage_for_user(self, *, user_id: str) -> tuple[list[dict[str, object]], int]:
        cutoff = date.today() - timedelta(days=29)
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


def _issue_session_token(user_id: str) -> str:
    if _auth_is_mocked():
        return _mock_auth.issue_session_token(user_id=user_id)
    serializer = _auth_serializer()
    if serializer is None:
        abort(503, description="Auth is not configured (missing AUTH_SECRET_KEY).")
    return serializer.dumps({"user_id": user_id})


def _load_session_token(token: str) -> str | None:
    if _auth_is_mocked():
        return _mock_auth.load_session_token(token)
    serializer = _auth_serializer()
    if serializer is None:
        return None
    try:
        payload = serializer.loads(token, max_age=60 * 60 * 24 * 14)
    except (BadSignature, SignatureExpired):
        return None
    user_id = payload.get("user_id")
    return user_id if isinstance(user_id, str) else None


_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


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
        return "http://127.0.0.1:5000"
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


def _google_state_serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("AUTH_SECRET_KEY")
    if not secret:
        abort(503, description="Auth is not configured (missing AUTH_SECRET_KEY).")
    return URLSafeTimedSerializer(secret_key=secret, salt="pandects-google-oauth")


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


def _google_verify_id_token(id_token: str) -> str:
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

    normalized = _normalize_email(email)
    if not _EMAIL_RE.match(normalized):
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
    from backend.redaction import redact_agreement_xml

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


def _lookup_api_key(raw_key: str) -> ApiKey | None:
    if _auth_is_mocked():
        return None
    raw_key = raw_key.strip()
    if not raw_key.startswith("pdcts_"):
        return None
    prefix = raw_key[: 6 + 12]  # "pdcts_" + 12 chars
    candidates = ApiKey.query.filter_by(prefix=prefix, revoked_at=None).limit(25).all()
    for candidate in candidates:
        if check_password_hash(candidate.key_hash, raw_key):
            candidate.last_used_at = datetime.utcnow()
            db.session.commit()
            return candidate
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
                    if user is not None:
                        return AccessContext(tier="user", user_id=user_id)
                elif _auth_db_is_configured():
                    try:
                        user = db.session.get(AuthUser, user_id)
                    except SQLAlchemyError:
                        user = None
                    if user is not None:
                        return AccessContext(tier="user", user_id=user_id)

    api_key_raw = request.headers.get("X-API-Key")
    api_key_raw = api_key_raw.strip() if isinstance(api_key_raw, str) else ""
    if api_key_raw:
        if _auth_is_mocked():
            api_key = _mock_auth.lookup_api_key(api_key_raw)
            if api_key is not None:
                return AccessContext(
                    tier="api_key", user_id=api_key.user_id, api_key_id=api_key.id
                )
        elif _auth_db_is_configured():
            try:
                api_key = _lookup_api_key(api_key_raw)
            except SQLAlchemyError:
                api_key = None
            if api_key is not None:
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
                if user is not None:
                    return AccessContext(tier="user", user_id=user_id)
            elif _auth_db_is_configured():
                try:
                    user = db.session.get(AuthUser, user_id)
                except SQLAlchemyError:
                    user = None
                if user is not None:
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
        return AuthUser(id=user.id, email=user.email, created_at=user.created_at), ctx
    try:
        user = db.session.get(AuthUser, ctx.user_id)
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")
    if user is None:
        abort(401, description="Invalid session.")
    if user.email.startswith("deleted+") and user.email.endswith("@deleted.invalid"):
        abort(401, description="Account deleted.")
    return user, ctx


def _rate_limit_key(ctx: AccessContext) -> tuple[str, int]:
    if ctx.tier == "api_key" and ctx.api_key_id:
        return f"api_key:{ctx.api_key_id}", 120
    if ctx.tier == "user" and ctx.user_id:
        return f"user:{ctx.user_id}", 300
    ip = _request_ip_address() or "unknown"
    return f"anon:{ip}", 60


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


@app.before_request
def _auth_rate_limit_guard():
    ctx = _current_access_context()
    g.access_ctx = ctx
    if _csrf_required(request.path):
        csrf_cookie = request.cookies.get(_CSRF_COOKIE_NAME, "")
        csrf_header = request.headers.get("X-CSRF-Token", "")
        if not csrf_cookie or csrf_cookie != csrf_header:
            abort(403, description="Missing or invalid CSRF token.")
    _check_rate_limit(ctx)


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
        today = date.today()
        row = ApiUsageDaily.query.filter_by(api_key_id=ctx.api_key_id, day=today).first()
        if row is None:
            row = ApiUsageDaily(api_key_id=ctx.api_key_id, day=today, count=1)
            db.session.add(row)
        else:
            row.count = int(row.count) + 1
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

    llm_output_table = Table(
        "llm_output",
        metadata,
        autoload_with=engine,
    )
    prompts_table = Table(
        "prompts",
        metadata,
        autoload_with=engine,
    )
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
    llm_output_table = Table("llm_output", metadata, Column("id", Integer, primary_key=True))
    prompts_table = Table("prompts", metadata, Column("id", Integer, primary_key=True))
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
class LLMOut(db.Model):
    __table__ = llm_output_table


class Prompts(db.Model):
    __table__ = prompts_table


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


# ── Route definitions ───────────────────────────────────────

# LLM routes - Only available in local development (debug mode)
@app.route("/api/llm/<string:page_uuid>", methods=["GET"])
def get_llm(page_uuid):
    # Check if running in debug mode (local development)
    if not app.debug:
        abort(404)  # Return 404 in production to hide the endpoint

    # pick the most-recent prompt for this page (excluding SKIP outputs)
    latest_prompt_id = (
        db.session.query(Prompts.prompt_id)
        .join(LLMOut, Prompts.prompt_id == LLMOut.prompt_id)
        .filter(LLMOut.page_uuid == page_uuid)
        .filter(func.coalesce(LLMOut.llm_output_corrected, LLMOut.llm_output) != "SKIP")
        .order_by(desc(Prompts.updated_at))
        .limit(1)
        .scalar()
    )
    if not latest_prompt_id:
        abort(404)

    # fetch the LLMOut record
    record = LLMOut.query.get_or_404((page_uuid, latest_prompt_id))

    return jsonify(
        {
            "pageUuid": record.page_uuid,
            "promptId": record.prompt_id,
            "llmOutput": record.llm_output,
            "llmOutputCorrected": record.llm_output_corrected,
        }
    )


@app.route("/api/llm/<string:page_uuid>/<string:prompt_id>", methods=["PUT"])
def update_llm(page_uuid, prompt_id):
    # Check if running in debug mode (local development)
    if not app.debug:
        abort(404)  # Return 404 in production to hide the endpoint

    data = request.get_json()
    corrected = data.get("llmOutputCorrected")
    if corrected is None:
        return jsonify({"error": "llmOutputCorrected is required"}), 400

    record = LLMOut.query.get_or_404((page_uuid, prompt_id))
    record.llm_output_corrected = corrected
    db.session.commit()
    return jsonify({"status": "updated"}), 200


@app.route("/api/agreements/<string:agreement_uuid>", methods=["GET"])
def get_agreement(agreement_uuid):
    ctx = _current_access_context()
    focus_section_uuid = request.args.get("focusSectionUuid")
    if focus_section_uuid is not None:
        focus_section_uuid = focus_section_uuid.strip()
        if not _UUID_RE.match(focus_section_uuid):
            abort(400, description="Invalid focusSectionUuid.")
    neighbor_sections = request.args.get("neighborSections", "1")
    try:
        neighbor_sections_int = int(neighbor_sections)
    except ValueError:
        abort(400, description="neighborSections must be an integer.")

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
        return jsonify(
            {
                "year": year,
                "target": target,
                "acquirer": acquirer,
                "url": url,
                "xml": redacted_xml,
                "isRedacted": True,
            }
        )
    return jsonify(
        {
            "year": year,
            "target": target,
            "acquirer": acquirer,
            "url": url,
            "xml": xml_content,
        }
    )


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

# ── Auth routes ──────────────────────────────────────────────────────────


@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    _require_auth_db()
    data = _require_json()
    checked_at = _require_legal_acceptance(data)
    email_raw = data.get("email")
    password = data.get("password")
    if not isinstance(email_raw, str) or not isinstance(password, str):
        abort(400, description="Email and password are required.")
    email = _normalize_email(email_raw)
    if not _EMAIL_RE.match(email):
        abort(400, description="Invalid email address.")
    if len(password) < 8:
        abort(400, description="Password must be at least 8 characters.")

    if _auth_is_mocked():
        user = _mock_auth.create_user(email=email, password=password)
        token = _issue_session_token(user.id)
        payload = {
            "user": {
                "id": user.id,
                "email": user.email,
                "createdAt": user.created_at.isoformat(),
            },
        }
        if _auth_session_transport() == "bearer":
            payload["sessionToken"] = token
        resp = make_response(jsonify(payload), 201)
        resp.headers["Cache-Control"] = "no-store"
        if _auth_session_transport() == "cookie":
            _set_auth_cookies(resp, session_token=token)
        return resp

    try:
        existing = AuthUser.query.filter_by(email=email).first()
        if existing is not None:
            abort(409, description="An account with this email already exists.")

        now = datetime.utcnow()
        ip_address = _request_ip_address()
        user_agent = _request_user_agent()
        user = AuthUser(email=email, password_hash=generate_password_hash(password))
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
        db.session.commit()
        token = _issue_session_token(user.id)
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")

    payload = {
        "user": {
            "id": user.id,
            "email": user.email,
            "createdAt": user.created_at.isoformat(),
        },
    }
    if _auth_session_transport() == "bearer":
        payload["sessionToken"] = token
    resp = make_response(jsonify(payload), 201)
    resp.headers["Cache-Control"] = "no-store"
    if _auth_session_transport() == "cookie":
        _set_auth_cookies(resp, session_token=token)
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
            abort(401, description="Invalid credentials.")
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
            abort(401, description="Invalid credentials.")
        if not check_password_hash(user.password_hash, password):
            abort(401, description="Invalid credentials.")

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


@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    _require_auth_db()
    user, _ctx = _require_user()
    return jsonify({"user": {"id": user.id, "email": user.email}})


@app.route("/api/auth/csrf", methods=["GET"])
def auth_csrf():
    _require_auth_db()
    resp = make_response(jsonify({"status": "ok"}))
    resp.headers["Cache-Control"] = "no-store"
    if _auth_session_transport() == "cookie":
        _ensure_csrf_cookie(resp)
    return resp


@app.route("/api/auth/api-keys", methods=["GET"])
def auth_list_api_keys():
    _require_auth_db()
    user, _ctx = _require_user()
    if _auth_is_mocked():
        keys = _mock_auth.list_api_keys(user_id=user.id)
        return jsonify(
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
    try:
        keys = (
            ApiKey.query.filter_by(user_id=user.id)
            .order_by(ApiKey.created_at.desc())
            .all()
        )
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")
    return jsonify(
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


@app.route("/api/auth/api-keys", methods=["POST"])
def auth_create_api_key():
    _require_auth_db()
    user, _ctx = _require_user()
    data = _require_json()
    name = data.get("name")
    if name is not None and not isinstance(name, str):
        abort(400, description="Key name must be a string.")
    if _auth_is_mocked():
        key, plaintext = _mock_auth.create_api_key(user_id=user.id, name=name)
        return jsonify(
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
    try:
        key, plaintext = _create_api_key(user_id=user.id, name=name)
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")
    return jsonify(
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


@app.route("/api/auth/api-keys/<string:key_id>", methods=["DELETE"])
def auth_revoke_api_key(key_id: str):
    _require_auth_db()
    user, _ctx = _require_user()
    if _auth_is_mocked():
        if not _mock_auth.revoke_api_key(user_id=user.id, key_id=key_id):
            abort(404)
        return jsonify({"status": "revoked"})
    try:
        key = ApiKey.query.filter_by(id=key_id, user_id=user.id).first()
        if key is None:
            abort(404)
        if key.revoked_at is None:
            key.revoked_at = datetime.utcnow()
            db.session.commit()
        return jsonify({"status": "revoked"})
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")


@app.route("/api/auth/usage", methods=["GET"])
def auth_usage():
    _require_auth_db()
    user, _ctx = _require_user()
    if _auth_is_mocked():
        by_day, total = _mock_auth.usage_for_user(user_id=user.id)
        return jsonify({"byDay": by_day, "total": total})
    cutoff = date.today() - timedelta(days=29)
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

        return jsonify(
            {
                "byDay": [{"day": day, "count": by_day[day]} for day in sorted(by_day)],
                "total": total,
            }
        )
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")


@app.route("/api/auth/account/delete", methods=["POST"])
def auth_delete_account():
    _require_auth_db()
    user, _ctx = _require_user()
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

    client_id = _google_oauth_client_id()
    redirect_uri = _google_oauth_redirect_uri()

    next_path = _safe_next_path(request.args.get("next"))
    state_payload = {"next": next_path or "/account"}
    state = _google_state_serializer().dumps(state_payload)

    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": "openid email profile",
        "state": state,
        "prompt": "select_account",
    }
    auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{urlencode(params)}"
    resp = redirect(auth_url)
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/auth/google/client-id", methods=["GET"])
def auth_google_client_id():
    _require_auth_db()
    if _auth_is_mocked():
        abort(501, description="Google auth is unavailable in mock auth mode.")
    return jsonify({"clientId": _google_oauth_client_id()})


@app.route("/api/auth/google/callback", methods=["GET"])
def auth_google_callback():
    _require_auth_db()
    if _auth_is_mocked():
        abort(501, description="Google auth is unavailable in mock auth mode.")

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

    try:
        state_payload = _google_state_serializer().loads(state, max_age=60 * 10)
    except (BadSignature, SignatureExpired):
        return _frontend_google_callback_redirect(
            token=None, next_path="/account", error="invalid_state"
        )

    next_path = None
    if isinstance(state_payload, dict):
        next_path = _safe_next_path(state_payload.get("next"))

    token_payload = _google_fetch_json(
        "https://oauth2.googleapis.com/token",
        data={
            "code": code,
            "client_id": _google_oauth_client_id(),
            "client_secret": _google_oauth_client_secret(),
            "redirect_uri": _google_oauth_redirect_uri(),
            "grant_type": "authorization_code",
        },
    )

    id_token = token_payload.get("id_token")
    if not isinstance(id_token, str) or not id_token.strip():
        return _frontend_google_callback_redirect(
            token=None, next_path=next_path or "/account", error="missing_id_token"
        )

    try:
        normalized = _google_verify_id_token(id_token)
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
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")

    token = _issue_session_token(user.id)
    if _auth_session_transport() == "cookie":
        dest = f"{_frontend_base_url()}{(next_path or '/account')}"
        resp = redirect(dest)
        resp.headers["Cache-Control"] = "no-store"
        _set_auth_cookies(resp, session_token=token)
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

    normalized = _google_verify_id_token(credential)

    try:
        user = AuthUser.query.filter_by(email=normalized).first()
        if user is None:
            checked_at = _require_legal_acceptance(data)
            now = datetime.utcnow()
            ip_address = _request_ip_address()
            user_agent = _request_user_agent()
            user = AuthUser(email=normalized, password_hash=None)
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
    return resp


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    _require_auth_db()
    resp = make_response(jsonify({"status": "ok"}))
    resp.headers["Cache-Control"] = "no-store"
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
        expected = {"auth_users", "api_keys", "api_usage_daily"}
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
