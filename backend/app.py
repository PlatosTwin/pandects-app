import os
import sys
import importlib
from typing import Callable, Protocol, SupportsInt, TypedDict, cast
from collections.abc import Iterable, Mapping
from pathlib import Path
import time
import random
import hmac
import hashlib
import base64
import binascii
from threading import Lock
from datetime import datetime, date, timedelta, timezone
import uuid
import re
import click
import json
import math
from collections import defaultdict
from flask import Flask, jsonify, request, abort, Response, g, current_app, has_app_context
from flask import redirect
from flask import make_response
from flask_cors import CORS
from flask_smorest import Blueprint
from boto3.session import Session
from marshmallow import Schema, ValidationError, EXCLUDE
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.exceptions import HTTPException, InternalServerError
from dataclasses import dataclass
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import (
    inspect,
    select,
)
from sqlalchemy.sql.elements import ColumnElement
from dotenv import load_dotenv
from urllib.parse import urlencode, quote, urlsplit
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
import secrets
from werkzeug.wrappers.response import Response as WerkzeugResponse

# Load env vars from `backend/.env` regardless of process working directory.
_ = load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
# Also allow repo/root `.env` (or process env) to supply values without overriding.
_ = load_dotenv()

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
from backend.models.main_db import (
    Agreements,
    LatestSectionsSearch,
    LatestSectionsSearchStandardId,
    NaicsSector,
    NaicsSubSector,
    Sections,
    TaxonomyL1,
    TaxonomyL2,
    TaxonomyL3,
    XML,
    agreement_latest_xml_join_condition as _agreement_latest_xml_join_condition,
    agreement_year_expr as _agreement_year_expr,
    coalesced_section_standard_ids as _coalesced_section_standard_ids,
    expand_taxonomy_standard_ids_cached as _expand_taxonomy_standard_ids_cached,
    main_db_schema_from_env as _main_db_schema_from_env,
    main_db_uri_from_env as _main_db_uri_from_env,
    metadata,
    parse_section_standard_ids as _parse_section_standard_ids,
    schema_translate_map as _schema_translate_map,
    section_latest_xml_join_condition as _section_latest_xml_join_condition,
    standard_id_filter_expr as _standard_id_filter_expr,
    year_from_filing_date_value as _year_from_filing_date_value,
)
from backend.routes.search import register_search_routes
from backend.routes.agreements import register_agreements_routes
from backend.routes.reference_data import register_reference_data_routes
from backend.services.async_tasks import AsyncTaskRunner
from backend.services.search_service import (
    estimated_latest_sections_search_table_rows as _svc_estimated_latest_sections_search_table_rows,
    estimated_query_row_count as _svc_estimated_query_row_count,
    search_total_count_metadata as _svc_search_total_count_metadata,
)
from backend.services.usage import UsageBuffer

# Contract surface consumed by `backend.routes.auth` via app_module indirection.
_AUTH_SCHEMA_EXPORTS = (
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


class _FilterOptionsPayload(TypedDict):
    targets: list[str]
    acquirers: list[str]
    target_industries: list[str]
    acquirer_industries: list[str]


class _FilterOptionsCache(TypedDict):
    ts: float
    payload: _FilterOptionsPayload | None


class _ObjectPayloadCache(TypedDict):
    ts: float
    payload: dict[str, object] | None


class _AgreementsSummaryPayload(TypedDict):
    agreements: int
    sections: int
    pages: int


class _AgreementsSummaryCache(TypedDict):
    ts: float
    payload: _AgreementsSummaryPayload | None


class _DumpsCache(TypedDict):
    ts: float
    payload: list[dict[str, object]] | None


class _DumpsManifestCacheEntry(TypedDict):
    etag: str
    payload: dict[str, object]
    ts: float

class _S3ListObject(TypedDict, total=False):
    Key: str
    ETag: str


class _S3ListPage(TypedDict, total=False):
    Contents: list[_S3ListObject]


class _S3Paginator(Protocol):
    def paginate(self, *, Bucket: str, Prefix: str) -> Iterable[_S3ListPage]:
        ...


class _S3BodyReader(Protocol):
    def read(self) -> bytes:
        ...


class _S3GetObjectResult(TypedDict):
    Body: _S3BodyReader


class _S3Client(Protocol):
    def get_paginator(self, operation_name: str) -> _S3Paginator:
        ...

    def get_object(self, *, Bucket: str, Key: str) -> _S3GetObjectResult:
        ...


class _HttpResponseReader(Protocol):
    def __enter__(self) -> "_HttpResponseReader":
        ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> bool | None:
        ...

    def read(self, amt: int | None = None) -> bytes:
        ...


class _Boto3SessionLike(Protocol):
    def client(
        self,
        service_name: str,
        *,
        aws_access_key_id: str,
        aws_secret_access_key: str,
        endpoint_url: str,
    ) -> object:
        ...


class _JwkSigningKeyLike(Protocol):
    key: str | bytes


class _JwkClientLike(Protocol):
    def get_signing_key_from_jwt(self, token: str) -> _JwkSigningKeyLike:
        ...


class _FlaskExtension(Protocol):
    def init_app(self, app: Flask, **kwargs: object) -> None:
        ...


class _ApiExtension(_FlaskExtension, Protocol):
    def register_blueprint(
        self, blp: Blueprint, *, parameters: object | None = None, **options: object
    ) -> None:
        ...


class _RegisterAuthRoutesFn(Protocol):
    def __call__(self, app: Flask, *, app_module: object) -> Blueprint:
        ...


_MAIN_SCHEMA_TOKEN = "__main_schema__"
_SKIP_MAIN_DB_REFLECTION = os.environ.get("SKIP_MAIN_DB_REFLECTION", "").strip() == "1"
_ENABLE_MAIN_DB_REFLECTION = (
    os.environ.get("ENABLE_MAIN_DB_REFLECTION", "1").strip() != "0"
)

# ── Simple in-process caching ─────────────────────────────────────────────
_FILTER_OPTIONS_TTL_SECONDS = int(os.environ.get("FILTER_OPTIONS_TTL_SECONDS", "21600"))
_filter_options_cache: _FilterOptionsCache = {"ts": 0.0, "payload": None}
_filter_options_lock = Lock()
_TAXONOMY_TTL_SECONDS = int(os.environ.get("TAXONOMY_TTL_SECONDS", "21600"))
_taxonomy_cache: _ObjectPayloadCache = {"ts": 0.0, "payload": None}
_taxonomy_lock = Lock()
_NAICS_TTL_SECONDS = int(os.environ.get("NAICS_TTL_SECONDS", "21600"))
_naics_cache: _ObjectPayloadCache = {"ts": 0.0, "payload": None}
_naics_lock = Lock()
_AGREEMENTS_SUMMARY_TTL_SECONDS = int(
    os.environ.get("AGREEMENTS_SUMMARY_TTL_SECONDS", "60")
)
_agreements_summary_cache: _AgreementsSummaryCache = {"ts": 0.0, "payload": None}
_agreements_summary_lock = Lock()

# ── Simple in-process rate limiting ──────────────────────────────────────
_rate_limit_lock = Lock()
_rate_limit_state: dict[str, dict[str, float | int]] = {}
_endpoint_rate_limit_state: dict[str, dict[str, float | int]] = {}
_RATE_LIMIT_WINDOW_SECONDS = 60.0
_RATE_LIMIT_MAX_KEYS = int(os.environ.get("RATE_LIMIT_MAX_KEYS", "50000"))
_RATE_LIMIT_PRUNE_INTERVAL_SECONDS = float(
    os.environ.get("RATE_LIMIT_PRUNE_INTERVAL_SECONDS", "30")
)
_rate_limit_last_prune_at = 0.0

# ── Simple in-process caching for dumps ───────────────────────────────────
_DUMPS_CACHE_TTL_SECONDS = int(os.environ.get("DUMPS_CACHE_TTL_SECONDS", "300"))
_dumps_cache: _DumpsCache = {"ts": 0.0, "payload": None}
_dumps_cache_lock = Lock()
_DUMPS_MANIFEST_CACHE_TTL_SECONDS = int(
    os.environ.get("DUMPS_MANIFEST_CACHE_TTL_SECONDS", "1800")
)
_dumps_manifest_cache: dict[str, _DumpsManifestCacheEntry] = {}
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
_API_KEY_LAST_USED_TOUCH_SECONDS = int(
    os.environ.get("API_KEY_LAST_USED_TOUCH_SECONDS", "300")
)
_API_KEY_LAST_USED_MAX_KEYS = int(os.environ.get("API_KEY_LAST_USED_MAX_KEYS", "50000"))
_api_key_last_used_touch_lock = Lock()
_api_key_last_used_touch_state: dict[str, float] = {}
_SEARCH_EXPLAIN_ESTIMATE_ENABLED = (
    os.environ.get("SEARCH_EXPLAIN_ESTIMATE_ENABLED", "1").strip() != "0"
)


def _usage_buffer() -> UsageBuffer | None:
    if not _USAGE_BUFFER_ENABLED:
        return None
    buffer = current_app.extensions.get("usage_buffer")
    if buffer is None:
        app_obj = _current_app_object()
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
        app_obj = _current_app_object()
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
            + "Specify explicit origins instead."
        )
    return origins or list(_DEFAULT_CORS_ORIGINS)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _urlopen_read_bytes(req: Request, *, timeout: float = 15) -> bytes:
    with cast(_HttpResponseReader, urlopen(req, timeout=timeout)) as resp:
        return resp.read()


def _current_app_object() -> Flask:
    app_obj = cast(object, current_app)
    getter = getattr(app_obj, "_get_current_object", None)
    if callable(getter):
        return cast(Flask, getter())
    return cast(Flask, app_obj)


def _app_config_map(app: Flask) -> dict[str, object]:
    return cast(dict[str, object], app.config)


def _to_int(value: object, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return int(stripped)
        except ValueError:
            try:
                return int(float(stripped))
            except ValueError:
                return default
    if isinstance(value, SupportsInt):
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return default
    return default


def _row_mapping_as_dict(row: object) -> dict[str, object]:
    if isinstance(row, dict):
        return cast(dict[str, object], row)
    if isinstance(row, Mapping):
        mapping_row = cast(Mapping[object, object], row)
        return {str(key): value for key, value in mapping_row.items()}
    mapping_obj = cast(object, getattr(row, "_mapping", None))
    if mapping_obj is None:
        return {}
    items = getattr(mapping_obj, "items", None)
    if not callable(items):
        return {}
    result: dict[str, object] = {}
    for key, value in cast(Iterable[tuple[object, object]], items()):
        if isinstance(key, str):
            result[key] = value
    return result


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
_google_jwk_client: object | None = None


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


def _set_auth_cookies(resp: WerkzeugResponse, *, session_token: str) -> None:
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


def _set_csrf_cookie(resp: WerkzeugResponse, value: str, *, max_age: int) -> None:
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


def _clear_auth_cookies(resp: WerkzeugResponse) -> None:
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
    config = _app_config_map(target_app)
    if AUTH_DATABASE_URI is not None:
        config["SQLALCHEMY_BINDS"] = {"auth": AUTH_DATABASE_URI}
    else:
        config["SQLALCHEMY_BINDS"] = {
            "auth": f"sqlite:///{Path(__file__).with_name('auth_dev.sqlite')}"
        }

# ── OpenAPI / Flask-Smorest configuration ───────────────────────────────
def _configure_openapi(target_app: Flask) -> None:
    config = _app_config_map(target_app)
    config.update(
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
    config["MAX_CONTENT_LENGTH"] = _max_content_length()


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


def _schema_prefix() -> str:
    if has_app_context():
        config = _app_config_map(_current_app_object())
        raw = config.get("MAIN_DB_SCHEMA", "")
        value = raw.strip() if isinstance(raw, str) else ""
    else:
        value = _main_db_schema_from_env()
    return f"{value}." if value else ""


def _configure_main_db(target_app: Flask) -> None:
    config = _app_config_map(target_app)
    if "SQLALCHEMY_DATABASE_URI" not in config:
        config["SQLALCHEMY_DATABASE_URI"] = _main_db_uri_from_env()
    _ = config.setdefault("MAIN_DB_SCHEMA", _main_db_schema_from_env())
    config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    configured_engine_options = config.get("SQLALCHEMY_ENGINE_OPTIONS", {})
    engine_options = (
        dict(cast(dict[str, object], configured_engine_options))
        if isinstance(configured_engine_options, dict)
        else {}
    )
    raw_execution_options = engine_options.get("execution_options", {})
    execution_options = (
        dict(cast(dict[str, object], raw_execution_options))
        if isinstance(raw_execution_options, dict)
        else {}
    )
    _ = execution_options.setdefault(
        "schema_translate_map",
        _schema_translate_map(cast(str | None, config.get("MAIN_DB_SCHEMA"))),
    )
    engine_options["execution_options"] = execution_options
    config["SQLALCHEMY_ENGINE_OPTIONS"] = engine_options


def _configure_extensions(target_app: Flask) -> None:
    cast(_FlaskExtension, cast(object, api)).init_app(target_app)
    cast(_FlaskExtension, cast(object, db)).init_app(target_app)


def _configure_cors(target_app: Flask) -> None:
    _ = CORS(
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
        config = _app_config_map(target_app)
        config.update(config_overrides)
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

client: _S3Client | None = None
if os.environ.get("R2_ACCESS_KEY_ID") and os.environ.get("R2_SECRET_ACCESS_KEY"):
    session = cast(_Boto3SessionLike, Session())
    raw_client = session.client(
        service_name="s3",
        aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        endpoint_url=R2_ENDPOINT,
    )
    client = cast(_S3Client, raw_client)

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
        expires_at = _utc_now() + timedelta(seconds=_password_reset_max_age_seconds())
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
        raw_payload_obj = cast(
            object, serializer.loads(token, max_age=_email_verification_max_age_seconds())
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


def _issue_password_reset_token(*, user_id: str, email: str) -> str:
    if _auth_is_mocked():
        return _mock_auth.issue_password_reset_token(user_id=user_id, email=email)
    serializer = _password_reset_serializer()
    reset_id = str(uuid.uuid4())
    now = _utc_now()
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
        raw_payload_obj = cast(
            object, serializer.loads(token, max_age=_password_reset_max_age_seconds())
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
            .first()
        )
    except SQLAlchemyError:
        return None
    if row is None:
        return None
    expires_at = cast(object, row.expires_at)
    if not isinstance(expires_at, datetime) or expires_at <= _utc_now():
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


def _send_email_verification_email(*, to_email: str, token: str) -> None:
    verify_url = f"{_frontend_base_url()}/auth/verify-email#token={quote(token)}"
    subject = "Verify your email for Pandects"
    year = str(_utc_now().year)
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
    year = str(_utc_now().year)
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
    now = _utc_now()
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
        session = cast(
            AuthSession | None,
            AuthSession.query.filter_by(
                token_hash=_session_token_hash(token), revoked_at=None
            ).first()
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


def _revoke_session_token(token: str) -> None:
    if _auth_is_mocked():
        _mock_auth.revoke_session_token(token)
        return
    if not token:
        return
    now = _utc_now()
    try:
        _ = AuthSession.query.filter_by(token_hash=_session_token_hash(token)).update(
            {"revoked_at": now}, synchronize_session=False
        )
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
        return


_SECTION_ID_RE = re.compile(
    r"^(?:[0-9a-fA-F]{16}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
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


def _set_google_oauth_cookie(resp: WerkzeugResponse, payload: dict[str, str]) -> None:
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
        raw_payload_obj = cast(
            object,
            _google_oauth_cookie_serializer().loads(
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


def _clear_google_oauth_cookie(resp: WerkzeugResponse) -> None:
    samesite, secure = _cookie_settings()
    resp.delete_cookie(
        _GOOGLE_OAUTH_COOKIE_NAME,
        path="/v1/auth/google/callback",
        secure=secure,
        samesite=samesite.capitalize() if samesite != "none" else "None",
    )


def _set_google_nonce_cookie(resp: WerkzeugResponse, nonce: str) -> None:
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


def _clear_google_nonce_cookie(resp: WerkzeugResponse) -> None:
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


def _require_captcha_token(data: dict[str, object]) -> str:
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


def _encode_frontend_hash_params(params: dict[str, str]) -> str:
    return urlencode(params, quote_via=quote)


def _frontend_google_callback_redirect(*, token: str | None, next_path: str | None, error: str | None):
    fragment: dict[str, str] = {}
    if token:
        fragment["session_token"] = token
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
    except URLError as e:
        abort(502, description="Google auth failed (network error).")
    try:
        parsed_obj = cast(object, json.loads(raw))
    except json.JSONDecodeError:
        abort(502, description="Google auth failed (invalid JSON response).")
    return cast(dict[str, object], parsed_obj) if isinstance(parsed_obj, dict) else {}


def _google_verify_id_token(id_token: str, *, expected_nonce: str | None = None) -> str:
    try:
        import jwt
        from jwt import PyJWKClient
        from jwt.exceptions import InvalidTokenError
        from jwt.exceptions import PyJWKClientError
    except ImportError:
        abort(503, description="Google auth is unavailable (missing PyJWT dependency).")

    global _google_jwk_client
    client = _google_jwk_client
    if client is None:
        client = PyJWKClient("https://www.googleapis.com/oauth2/v3/certs")
        _google_jwk_client = client

    client_obj = cast(_JwkClientLike, client)
    try:
        signing_key = client_obj.get_signing_key_from_jwt(id_token).key
        payload_obj = cast(
            object,
            jwt.decode(
            id_token,
            signing_key,
            algorithms=["RS256"],
            audience=_google_oauth_client_id(),
            issuer=["accounts.google.com", "https://accounts.google.com"],
            leeway=60,
            ),
        )
    except PyJWKClientError:
        abort(503, description="Google auth is temporarily unavailable.")
    except InvalidTokenError:
        abort(401, description="Invalid Google credential.")

    if not isinstance(payload_obj, dict):
        abort(401, description="Invalid Google credential.")
    payload = cast(dict[str, object], payload_obj)

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


def _require_json() -> dict[str, object]:
    """Read a JSON object body or abort with a 400 error."""
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        abort(400, description="Expected JSON object body.")
    return cast(dict[str, object], data)


def _load_json(schema: Schema) -> dict[str, object]:
    """Validate a JSON body against a Marshmallow schema."""
    data = _require_json()
    try:
        loaded = cast(object, schema.load(data, unknown=EXCLUDE))
    except ValidationError as exc:
        current_app.logger.debug("Validation error: %s", cast(object, exc.messages))
        abort(_json_error(400, error="validation_error", message="Invalid request body."))
    if not isinstance(loaded, dict):
        abort(400, description="Expected JSON object body.")
    return cast(dict[str, object], loaded)


def _load_query(schema: Schema) -> dict[str, object]:
    """Validate query args against a Marshmallow schema."""
    try:
        loaded = cast(object, schema.load(request.args, unknown=EXCLUDE))
    except ValidationError as exc:
        current_app.logger.debug("Validation error: %s", cast(object, exc.messages))
        abort(_json_error(400, error="validation_error", message="Invalid query parameters."))
    if not isinstance(loaded, dict):
        abort(400, description="Expected query object.")
    return cast(dict[str, object], loaded)


def _pagination_metadata(
    *,
    total_count: int,
    page: int,
    page_size: int,
    has_next_override: bool | None = None,
    total_count_is_approximate: bool = False,
) -> dict[str, object]:
    """Build standard pagination dict for list responses."""
    total_pages = math.ceil(total_count / page_size) if total_count else 0
    if total_count and page > total_pages:
        total_pages = page
    if has_next_override and total_pages <= page:
        total_pages = page + 1
    has_prev = page > 1
    has_next = has_next_override if has_next_override is not None else page < total_pages
    prev_num = page - 1 if has_prev else None
    next_num = page + 1 if has_next else None
    return {
        "page": page,
        "page_size": page_size,
        "total_count": total_count,
        "total_count_is_approximate": total_count_is_approximate,
        "total_pages": total_pages,
        "has_next": has_next,
        "has_prev": has_prev,
        "next_num": next_num,
        "prev_num": prev_num,
    }


def _estimated_query_row_count(query: object) -> int | None:
    return _svc_estimated_query_row_count(sys.modules[__name__], query)


def _estimated_latest_sections_search_table_rows() -> int | None:
    return _svc_estimated_latest_sections_search_table_rows(sys.modules[__name__])


def _search_total_count_metadata(
    *,
    query: object,
    page: int,
    page_size: int,
    item_count: int,
    has_next: bool,
    has_filters: bool,
) -> tuple[int, bool]:
    return _svc_search_total_count_metadata(
        sys.modules[__name__],
        query=query,
        page=page,
        page_size=page_size,
        item_count=item_count,
        has_next=has_next,
        has_filters=has_filters,
    )


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
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)


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
    return _utc_now().date()




def _require_legal_acceptance(data: dict[str, object]) -> datetime:
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


def _legal_acceptance_columns() -> tuple[
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


def _user_has_current_legal_acceptances(*, user_id: str) -> bool:
    expected_rows = {(doc, meta["version"], meta["sha256"]) for doc, meta in _LEGAL_DOCS.items()}
    document_col, version_col, document_hash_col, user_id_col = _legal_acceptance_columns()
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


def _ensure_current_legal_acceptances(*, user_id: str, checked_at: datetime) -> None:
    now = _utc_now()
    ip_address = _request_ip_address()
    user_agent = _request_user_agent()
    document_col, version_col, document_hash_col, user_id_col = _legal_acceptance_columns()
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


def _record_signon_event(*, user_id: str, provider: str, action: str) -> None:
    if _auth_is_mocked():
        return
    ip_address = _request_ip_address()
    user_agent = _request_user_agent()
    event = AuthSignonEvent()
    event.user_id = user_id
    event.provider = provider
    event.action = action
    event.ip_address = ip_address
    event.user_agent = user_agent
    db.session.add(event)


def _lookup_api_key(raw_key: str) -> ApiKey | None:
    if _auth_is_mocked():
        return None
    raw_key = raw_key.strip()
    if not raw_key.startswith("pdcts_"):
        return None
    prefix = raw_key[: 6 + 12]  # "pdcts_" + 12 chars
    candidates = cast(
        list[ApiKey],
        ApiKey.query.filter_by(prefix=prefix, revoked_at=None).limit(25).all(),
    )
    checks = 0
    for candidate in candidates:
        checks += 1
        key_hash = cast(object, candidate.key_hash)
        if isinstance(key_hash, str) and check_password_hash(key_hash, raw_key):
            candidate_id = cast(object, candidate.id)
            if candidate_id is not None:
                candidate_id_str = str(candidate_id)
                if _should_touch_api_key_last_used(candidate_id_str):
                    try:
                        candidate.last_used_at = _utc_now()
                        db.session.commit()
                    except SQLAlchemyError:
                        db.session.rollback()
            return candidate
    for _ in range(max(0, _API_KEY_MIN_HASH_CHECKS - checks)):
        _ = check_password_hash(_DUMMY_API_KEY_HASH, raw_key)
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
                    if user is not None and cast(object, user.email_verified_at) is not None:
                        return AccessContext(tier="user", user_id=user_id)

    api_key_raw = request.headers.get("X-API-Key")
    api_key_raw = api_key_raw.strip() if isinstance(api_key_raw, str) else ""
    if api_key_raw:
        if _auth_is_mocked():
            api_key = _mock_auth.lookup_api_key(api_key_raw)
            if api_key is not None:
                api_key_user_id = cast(object, api_key.user_id)
                api_key_id = cast(object, api_key.id)
                if isinstance(api_key_user_id, str) and _user_id_is_verified(api_key_user_id):
                    if api_key_id is not None:
                        g.api_key_last_used_candidate_id = str(api_key_id)
                    return AccessContext(
                        tier="api_key",
                        user_id=api_key_user_id,
                        api_key_id=api_key_id if isinstance(api_key_id, str) else None,
                    )
        elif _auth_db_is_configured():
            try:
                api_key = _lookup_api_key(api_key_raw)
            except SQLAlchemyError:
                api_key = None
            if api_key is not None:
                api_key_user_id = cast(object, api_key.user_id)
                api_key_id = cast(object, api_key.id)
                if isinstance(api_key_user_id, str) and _user_id_is_verified(api_key_user_id):
                    return AccessContext(
                        tier="api_key",
                        user_id=api_key_user_id,
                        api_key_id=api_key_id if isinstance(api_key_id, str) else None,
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
                if user is not None and cast(object, user.email_verified_at) is not None:
                    return AccessContext(tier="user", user_id=user_id)

    return AccessContext(tier="anonymous")


def _create_api_key(*, user_id: str, name: str | None) -> tuple[ApiKey, str]:
    token = f"pdcts_{uuid.uuid4().hex}{uuid.uuid4().hex}"
    prefix = token[: 6 + 12]
    key = ApiKey()
    key.user_id = user_id
    key.name = name.strip() if isinstance(name, str) and name.strip() else None
    key.prefix = prefix
    key.key_hash = generate_password_hash(token)
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
        auth_user = AuthUser()
        auth_user.id = user.id
        auth_user.email = user.email
        auth_user.email_verified_at = user.email_verified_at
        auth_user.created_at = user.created_at
        return (
            auth_user,
            ctx,
        )
    try:
        user = db.session.get(AuthUser, ctx.user_id)
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")
    if user is None:
        abort(401, description="Invalid session.")
    user_email = cast(object, user.email)
    if (
        isinstance(user_email, str)
        and user_email.startswith("deleted+")
        and user_email.endswith("@deleted.invalid")
    ):
        abort(401, description="Account deleted.")
    return user, ctx


def _require_verified_user() -> tuple[AuthUser, AccessContext]:
    user, ctx = _require_user()
    if cast(object, user.email_verified_at) is None:
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
    if user is None:
        return False
    return cast(object, user.email_verified_at) is not None


def _rate_limit_key(ctx: AccessContext) -> tuple[str, int]:
    if ctx.tier == "api_key" and ctx.api_key_id:
        return f"api_key:{ctx.api_key_id}", 300
    if ctx.tier == "user" and ctx.user_id:
        return f"user:{ctx.user_id}", 120
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


def _prune_rate_limit_state(now: float) -> None:
    global _rate_limit_last_prune_at
    if (now - _rate_limit_last_prune_at) < _RATE_LIMIT_PRUNE_INTERVAL_SECONDS:
        return

    cutoff = now - _RATE_LIMIT_WINDOW_SECONDS
    for state in (_rate_limit_state, _endpoint_rate_limit_state):
        stale_keys = [key for key, payload in state.items() if float(payload["ts"]) < cutoff]
        for key in stale_keys:
            _ = state.pop(key, None)
        overflow = len(state) - _RATE_LIMIT_MAX_KEYS
        if overflow > 0:
            oldest_keys = [
                key
                for key, _payload in sorted(
                    state.items(),
                    key=lambda item: float(item[1]["ts"]),
                )[:overflow]
            ]
            for key in oldest_keys:
                _ = state.pop(key, None)

    _rate_limit_last_prune_at = now


def _should_touch_api_key_last_used(key_id: str) -> bool:
    interval_seconds = float(max(0, _API_KEY_LAST_USED_TOUCH_SECONDS))
    now = time.time()
    with _api_key_last_used_touch_lock:
        if interval_seconds > 0:
            cutoff = now - (interval_seconds * 2.0)
            stale_keys = [
                existing_key
                for existing_key, touched_at in _api_key_last_used_touch_state.items()
                if touched_at < cutoff
            ]
            for stale_key in stale_keys:
                _ = _api_key_last_used_touch_state.pop(stale_key, None)
        overflow = len(_api_key_last_used_touch_state) - _API_KEY_LAST_USED_MAX_KEYS
        if overflow > 0:
            oldest_keys = [
                existing_key
                for existing_key, _touched_at in sorted(
                    _api_key_last_used_touch_state.items(),
                    key=lambda item: item[1],
                )[:overflow]
            ]
            for oldest_key in oldest_keys:
                _ = _api_key_last_used_touch_state.pop(oldest_key, None)
        last_touched_at = _api_key_last_used_touch_state.get(key_id)
        if interval_seconds > 0 and last_touched_at is not None:
            if (now - last_touched_at) < interval_seconds:
                return False
        _api_key_last_used_touch_state[key_id] = now
        return True


def _touch_api_key_last_used_if_needed() -> None:
    candidate_id = getattr(g, "api_key_last_used_candidate_id", None)
    if not isinstance(candidate_id, str) or not candidate_id:
        return
    if not _should_touch_api_key_last_used(candidate_id):
        return
    try:
        _ = ApiKey.query.filter_by(id=candidate_id, revoked_at=None).update(
            {"last_used_at": _utc_now()},
            synchronize_session=False,
        )
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()


def _check_rate_limit(ctx: AccessContext) -> None:
    if not request.path.startswith("/v1/"):
        return

    key, per_minute = _rate_limit_key(ctx)
    now = time.time()
    window = _RATE_LIMIT_WINDOW_SECONDS
    with _rate_limit_lock:
        _prune_rate_limit_state(now)
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
    window = _RATE_LIMIT_WINDOW_SECONDS
    with _rate_limit_lock:
        _prune_rate_limit_state(now)
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


def _record_api_key_usage(response: Response) -> Response:
    _touch_api_key_last_used_if_needed()
    usage_module = importlib.import_module("backend.services.usage")
    record_api_key_usage = cast(
        Callable[..., Response], getattr(usage_module, "record_api_key_usage")
    )
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
    if _is_running_on_fly():
        _ = response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=15552000; includeSubDomains",
        )
    return response


def _register_request_hooks(target_app: Flask) -> None:
    _before_req_start: object = target_app.before_request(_capture_request_start)
    _before_req_guard: object = target_app.before_request(_auth_rate_limit_guard)
    _after_req_usage: object = target_app.after_request(_record_api_key_usage)
    _after_req_headers: object = target_app.after_request(_set_security_headers)

# ── API helper compatibility shims ────────────────────────────────────────
def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _encode_agreements_cursor(agreement_uuid: str) -> str:
    payload = json.dumps({"agreement_uuid": agreement_uuid}, separators=(",", ":"))
    token = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")
    return token.rstrip("=")


def _decode_agreements_cursor(cursor_raw: str | None) -> str | None:
    if cursor_raw is None:
        return None
    cursor = cursor_raw.strip()
    if not cursor:
        return None
    padded = cursor + ("=" * (-len(cursor) % 4))
    try:
        decoded_bytes = base64.urlsafe_b64decode(padded.encode("ascii"))
        decoded_obj = cast(object, json.loads(decoded_bytes.decode("utf-8")))
    except (binascii.Error, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        abort(400, description="Invalid cursor.")
    if not isinstance(decoded_obj, dict):
        abort(400, description="Invalid cursor.")
    decoded_dict = cast(dict[str, object], decoded_obj)
    agreement_uuid = decoded_dict.get("agreement_uuid")
    if not isinstance(agreement_uuid, str) or not agreement_uuid.strip():
        abort(400, description="Invalid cursor.")
    return agreement_uuid

def _register_blueprints(target_app: Flask) -> None:
    api_ext = cast(_ApiExtension, cast(object, api))
    app_module = sys.modules[__name__]
    search_blp = register_search_routes(app_module=app_module)
    agreements_blp, sections_blp = register_agreements_routes(
        target_app,
        app_module=app_module,
    )
    taxonomy_blp, naics_blp, dumps_blp = register_reference_data_routes(
        app_module=app_module,
    )
    api_ext.register_blueprint(search_blp)
    api_ext.register_blueprint(agreements_blp)
    api_ext.register_blueprint(sections_blp)
    api_ext.register_blueprint(taxonomy_blp)
    api_ext.register_blueprint(naics_blp)
    api_ext.register_blueprint(dumps_blp)


# Contract surface consumed by extracted route/service modules via app_module indirection.
_PUBLIC_ROUTE_EXPORTS = (
    db,
    time,
    client,
    R2_BUCKET_NAME,
    Agreements,
    LatestSectionsSearch,
    LatestSectionsSearchStandardId,
    Sections,
    XML,
    TaxonomyL1,
    TaxonomyL2,
    TaxonomyL3,
    NaicsSector,
    NaicsSubSector,
    metadata,
    _ENABLE_MAIN_DB_REFLECTION,
    _MAIN_SCHEMA_TOKEN,
    _SKIP_MAIN_DB_REFLECTION,
    _SECTION_ID_RE,
    _SEARCH_EXPLAIN_ESTIMATE_ENABLED,
    _agreement_latest_xml_join_condition,
    _agreement_year_expr,
    _coalesced_section_standard_ids,
    _expand_taxonomy_standard_ids_cached,
    _parse_section_standard_ids,
    _section_latest_xml_join_condition,
    _standard_id_filter_expr,
    _year_from_filing_date_value,
    _current_access_context,
    _to_int,
    _row_mapping_as_dict,
    _schema_prefix,
    _load_query,
    _pagination_metadata,
    _dedupe_preserve_order,
    _encode_agreements_cursor,
    _decode_agreements_cursor,
    _redact_agreement_xml,
    _estimated_query_row_count,
    _estimated_latest_sections_search_table_rows,
    _search_total_count_metadata,
    _taxonomy_cache,
    _taxonomy_lock,
    _TAXONOMY_TTL_SECONDS,
    _naics_cache,
    _naics_lock,
    _NAICS_TTL_SECONDS,
    _dumps_cache,
    _dumps_cache_lock,
    _DUMPS_CACHE_TTL_SECONDS,
    _dumps_manifest_cache,
    _dumps_manifest_cache_lock,
    _DUMPS_MANIFEST_CACHE_TTL_SECONDS,
    _filter_options_cache,
    _filter_options_lock,
    _FILTER_OPTIONS_TTL_SECONDS,
    _agreements_summary_cache,
    _agreements_summary_lock,
    _AGREEMENTS_SUMMARY_TTL_SECONDS,
)


# Contract surface consumed by `backend.routes.auth` via app_module indirection.
_AUTH_ROUTE_HELPERS = (
    _set_auth_cookies,
    _set_csrf_cookie,
    _csrf_cookie_value,
    _clear_auth_cookies,
    _status_response,
    _require_auth_db,
    _issue_email_verification_token,
    _load_email_verification_token,
    _issue_password_reset_token,
    _load_password_reset_token,
    _send_signup_notification_email,
    _send_flag_notification_email,
    _send_email_verification_email,
    _send_password_reset_email,
    _issue_session_token,
    _revoke_session_token,
    _google_oauth_client_secret,
    _google_oauth_redirect_uri,
    _google_oauth_flow_enabled,
    _google_oauth_pkce_pair,
    _set_google_oauth_cookie,
    _load_google_oauth_cookie,
    _set_google_nonce_cookie,
    _google_nonce_cookie_value,
    _clear_google_nonce_cookie,
    _turnstile_enabled,
    _turnstile_site_key,
    _require_captcha_token,
    _verify_turnstile_token,
    _frontend_google_callback_redirect,
    _google_fetch_json,
    _google_verify_id_token,
    _safe_next_path,
    _load_json,
    _auth_enumeration_delay,
    _require_legal_acceptance,
    _user_has_current_legal_acceptances,
    _ensure_current_legal_acceptances,
    _record_signon_event,
    _create_api_key,
    _require_verified_user,
)


def _register_app(target_app: Flask) -> None:
    auth_routes = importlib.import_module("backend.routes.auth")

    _register_error_handlers(target_app)
    _register_request_hooks(target_app)
    _register_blueprints(target_app)
    register_auth_routes = cast(
        _RegisterAuthRoutesFn, getattr(auth_routes, "register_auth_routes")
    )
    _ = register_auth_routes(target_app, app_module=sys.modules[__name__])


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
        spec = api.spec
        if spec is None:
            raise click.ClickException("OpenAPI spec is unavailable.")
        yaml_spec = spec.to_yaml()
        _ = Path("openapi.yaml").write_text(yaml_spec)
        click.echo("Wrote openapi.yaml")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "").strip().lower() in ("1", "true", "yes")
    app.run(debug=debug, port=port)
