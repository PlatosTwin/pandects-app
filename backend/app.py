import os
from typing import Any, Protocol, TypedDict, cast
from collections.abc import Iterable
from pathlib import Path
import time
import random
from threading import Lock
from datetime import datetime
import uuid
import re
import click
from flask import Flask, request, abort, Response, g, current_app, has_app_context
from flask_smorest import Blueprint
from boto3.session import Session
from marshmallow import Schema
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import (
    inspect,
    text,
)
from dotenv import load_dotenv
from urllib.parse import urlencode
import secrets

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
    AuthExternalSubject,
    AuthOAuthAuthorizationCode,
    AuthOAuthClient,
    AuthOAuthSigningKey,
    AuthSession,
    AuthUser,
    LegalAcceptance,
)
from backend.schemas.auth import (
    AuthApiKeySchema,
    AuthDeleteAccountSchema,
    AuthExternalSubjectLinkSchema,
    AuthFlagInaccurateSchema,
    AuthPasswordLoginSchema,
    AuthPasswordResetConfirmSchema,
    AuthPasswordResetRequestSchema,
    AuthPasswordSignupSchema,
)
from backend.auth.mcp_runtime import (
    authenticate_external_identity as _authenticate_external_identity,
    resolve_mcp_identity_provider_name as _resolve_mcp_identity_provider_name,
)
from backend.models.main_db import (
    Agreements,
    AgreementCounsel,
    Clauses,
    Counsel,
    LatestSectionsSearch,
    NaicsSector,
    NaicsSubSector,
    Sections,
    TaxClauseAssignment,
    TaxClauseTaxonomyL1,
    TaxClauseTaxonomyL2,
    TaxClauseTaxonomyL3,
    TaxonomyL1,
    TaxonomyL2,
    TaxonomyL3,
    XML,
    agreement_latest_xml_join_condition as _agreement_latest_xml_join_condition,
    agreement_year_expr as _agreement_year_expr,
    coalesced_section_standard_ids as _coalesced_section_standard_ids,
    expand_tax_clause_taxonomy_standard_ids_cached as _expand_tax_clause_taxonomy_standard_ids_cached,
    expand_taxonomy_standard_ids_cached as _expand_taxonomy_standard_ids_cached,
    main_db_schema_from_env as _main_db_schema_from_env,
    metadata as _main_db_metadata,
    parse_section_standard_ids as _parse_section_standard_ids,
    section_latest_xml_join_condition as _section_latest_xml_join_condition,
    standard_id_filter_expr as _standard_id_filter_expr,
    standard_id_agreement_filter_expr as _standard_id_agreement_filter_expr,
    tax_clause_standard_id_filter_expr as _tax_clause_standard_id_filter_expr,
    year_from_filing_date_value as _year_from_filing_date_value,
)
from backend.routes.deps import (
    AgreementsDeps,
    AuthDeps,
    ReferenceDataDeps,
    SectionsDeps,
    SectionsServiceDeps,
)
from backend.routes.sections import register_sections_routes
from backend.routes.agreements import register_agreements_routes
from backend.routes.reference_data import register_reference_data_routes
from backend.routes.tax_clauses import TaxClausesDeps, register_tax_clauses_routes
from backend.routes.auth import register_auth_routes
from backend.services.tax_clauses_service import TaxClausesServiceDeps
from backend.mcp.routes import McpDeps, register_mcp_routes
from backend.core.config import (
    app_config_map as _app_config_map,
    configure_app as _configure_app_core,
)
from backend.core.errors import (
    json_error as _json_error,
    register_error_handlers as _register_error_handlers,
    status_response as _status_response,
)
from backend.core.hooks import (
    auth_rate_limit_guard as _auth_rate_limit_guard_core,
    capture_request_start as _capture_request_start_core,
    register_request_hooks as _register_request_hooks_core,
    set_security_headers as _set_security_headers_core,
)
from backend.core.runtime_utils import (
    current_app_object as _current_app_object,
    decode_agreements_cursor as _core_decode_agreements_cursor,
    dedupe_preserve_order as _dedupe_preserve_order,
    encode_agreements_cursor as _core_encode_agreements_cursor,
    load_json as _core_load_json,
    load_query as _core_load_query,
    pagination_metadata as _core_pagination_metadata,
    row_mapping_as_dict as _row_mapping_as_dict,
    to_int as _to_int,
    utc_now as _utc_now,
    utc_today as _utc_today,
)
from backend.auth.runtime import (
    AUTH_DATABASE_URI,
    AccessContext,
    _CSRF_COOKIE_NAME,
    _LEGAL_DOCS,
    _SESSION_COOKIE_NAME,
    _mock_auth,
    auth_db_is_configured as _auth_db_is_configured,
    auth_is_mocked as _auth_is_mocked,
    auth_session_transport as _auth_session_transport,
    clear_auth_cookies as _clear_auth_cookies,
    csrf_cookie_value as _csrf_cookie_value,
    csrf_required as _csrf_required,
    ensure_auth_schema_upgrades as _ensure_auth_schema_upgrades_auth_runtime,
    ensure_auth_tables_exist as _ensure_auth_tables_exist_auth_runtime,
    ensure_current_legal_acceptances as _ensure_current_legal_acceptances,
    frontend_base_url as _frontend_base_url,
    oauth_fetch_json as _oauth_fetch_json,
    issue_session_token as _issue_session_token,
    is_running_on_fly as _is_running_on_fly_auth_runtime,
    is_email_like as _is_email_like,
    load_session_token as _load_session_token,
    normalize_email as _normalize_email,
    record_signon_event as _record_signon_event,
    request_ip_address as _request_ip_address,
    request_user_agent as _request_user_agent,
    require_auth_db as _require_auth_db,
    require_captcha_token as _require_captcha_token,
    require_legal_acceptance as _require_legal_acceptance,
    revoke_session_token as _revoke_session_token,
    safe_next_path as _safe_next_path,
    send_resend_text_email as _send_resend_text_email,
    set_auth_cookies as _set_auth_cookies,
    set_csrf_cookie as _set_csrf_cookie,
    turnstile_enabled as _turnstile_enabled,
    turnstile_site_key as _turnstile_site_key,
    user_has_current_legal_acceptances as _user_has_current_legal_acceptances,
    verify_turnstile_token as _verify_turnstile_token,
)
from backend.services.async_tasks import AsyncTaskRunner
from backend.services.sections_service import (
    estimated_latest_sections_search_table_rows as _svc_estimated_latest_sections_search_table_rows,
    estimated_query_row_count as _svc_estimated_query_row_count,
    sections_total_count_metadata as _svc_sections_total_count_metadata,
)
from backend.services.usage import UsageBuffer
from backend.services.usage import record_api_key_usage as _record_api_key_usage_service

# Retained for tests and modules that still import `metadata` from `backend.app`.
metadata = _main_db_metadata

# Contract surface consumed by `backend.routes.auth` via app_module indirection.
_AUTH_SCHEMA_EXPORTS = (
    AuthApiKeySchema,
    AuthDeleteAccountSchema,
    AuthExternalSubjectLinkSchema,
    AuthFlagInaccurateSchema,
)


class _FilterOptionsPayload(TypedDict):
    targets: list[str]
    acquirers: list[str]
    target_counsels: list[str]
    acquirer_counsels: list[str]
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


class _FlaskExtension(Protocol):
    def init_app(self, app: Flask, **kwargs: object) -> None:
        ...


class _ApiExtension(_FlaskExtension, Protocol):
    def register_blueprint(
        self, blp: Blueprint, *, parameters: object | None = None, **options: object
    ) -> None:
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
_tax_clause_taxonomy_cache: _ObjectPayloadCache = {"ts": 0.0, "payload": None}
_tax_clause_taxonomy_lock = Lock()
_COUNSEL_TTL_SECONDS = int(os.environ.get("COUNSEL_TTL_SECONDS", "21600"))
_counsel_cache: _ObjectPayloadCache = {"ts": 0.0, "payload": None}
_counsel_lock = Lock()
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

# Legacy names retained for tests and app-module indirection.
_is_running_on_fly = _is_running_on_fly_auth_runtime
_ensure_auth_tables_exist = _ensure_auth_tables_exist_auth_runtime

def _schema_prefix() -> str:
    """Return the configured schema prefix for raw SQL assembly."""
    if has_app_context():
        config = _app_config_map(_current_app_object())
        raw = config.get("MAIN_DB_SCHEMA", "")
        value = raw.strip() if isinstance(raw, str) else ""
    else:
        value = _main_db_schema_from_env()
    return f"{value}." if value else ""


def _configure_app(
    target_app: Flask, *, config_overrides: dict[str, object] | None = None
) -> None:
    _configure_app_core(
        target_app,
        auth_database_uri=AUTH_DATABASE_URI,
        config_overrides=config_overrides,
    )



# ── JSON error responses for API routes ──────────────────────────────────

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

_SIGNUP_NOTIFICATION_EMAIL = "nmbogdan@alumni.stanford.edu"


def _send_signup_notification_email(*, new_user_email: str) -> None:
    subject = "New Pandects signup"
    text = f"{new_user_email} just signed up as a new user on Pandects."

    def _send() -> None:
        _send_resend_text_email(
            to_email=_SIGNUP_NOTIFICATION_EMAIL,
            subject=subject,
            text=text,
        )

    runner = _async_task_runner()
    if runner is None or not runner.enqueue(_send):
        _send()


def _send_flag_notification_email(
    *,
    user_email: str,
    submitted_at: datetime,
    source: str,
    agreement_uuid: str,
    section_uuid: str | None,
    message: str | None,
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
    _send_resend_text_email(
        to_email=_SIGNUP_NOTIFICATION_EMAIL, subject=subject, text=text
    )


_SECTION_ID_RE = re.compile(
    r"^(?:[0-9a-fA-F]{16}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)
_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


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


def _load_json(schema: Schema) -> dict[str, object]:
    return _core_load_json(schema, json_error=_json_error)


def _load_query(schema: Schema) -> dict[str, object]:
    return _core_load_query(schema, json_error=_json_error)


def _pagination_metadata(
    *,
    total_count: int,
    page: int,
    page_size: int,
    has_next_override: bool | None = None,
    total_count_is_approximate: bool = False,
) -> dict[str, object]:
    return _core_pagination_metadata(
        total_count=total_count,
        page=page,
        page_size=page_size,
        has_next_override=has_next_override,
        total_count_is_approximate=total_count_is_approximate,
    )


def _build_sections_service_deps() -> SectionsServiceDeps:
    return SectionsServiceDeps(
        db=db,
        AgreementCounsel=AgreementCounsel,
        Counsel=Counsel,
        LatestSectionsSearch=LatestSectionsSearch,
        Sections=Sections,
        _SEARCH_EXPLAIN_ESTIMATE_ENABLED=_SEARCH_EXPLAIN_ESTIMATE_ENABLED,
        _to_int=_to_int,
        _estimated_query_row_count=_estimated_query_row_count,
        _estimated_latest_sections_search_table_rows=_estimated_latest_sections_search_table_rows,
        _row_mapping_as_dict=_row_mapping_as_dict,
        _pagination_metadata=_pagination_metadata,
        _dedupe_preserve_order=_dedupe_preserve_order,
        _expand_taxonomy_standard_ids_cached=_expand_taxonomy_standard_ids_cached,
        _standard_id_filter_expr=_standard_id_filter_expr,
        _parse_section_standard_ids=_parse_section_standard_ids,
        _year_from_filing_date_value=_year_from_filing_date_value,
    )


def _estimated_query_row_count(query: object) -> int | None:
    return _svc_estimated_query_row_count(_build_sections_service_deps(), query)


def _estimated_latest_sections_search_table_rows() -> int | None:
    return _svc_estimated_latest_sections_search_table_rows(_build_sections_service_deps())


def _search_total_count_metadata(  # pyright: ignore[reportUnusedFunction]
    *,
    query: object,
    page: int,
    page_size: int,
    item_count: int,
    has_next: bool,
    has_filters: bool,
) -> tuple[int, bool]:
    total_count, is_approximate, _ = _svc_sections_total_count_metadata(
        _build_sections_service_deps(),
        query=query,
        page=page,
        page_size=page_size,
        item_count=item_count,
        has_next=has_next,
        has_filters=has_filters,
        count_mode="auto",
    )
    return total_count, is_approximate


def _auth_enumeration_delay() -> None:
    time.sleep(random.uniform(0.15, 0.35))


def _is_agreement_section_eligible(agreement_uuid: str, section_uuid: str | None) -> bool:
    if not agreement_uuid or not _UUID_RE.match(agreement_uuid):
        return False
    if section_uuid is not None and (not section_uuid or not _UUID_RE.match(section_uuid)):
        return False
    q = (
        db.session.query(Sections.section_uuid)
        .join(XML, _section_latest_xml_join_condition())
        .filter(Sections.agreement_uuid == agreement_uuid)
    )
    if section_uuid is not None:
        q = q.filter(Sections.section_uuid == section_uuid)
    return q.first() is not None


def _lookup_api_key(raw_key: str) -> ApiKey | None:
    if _auth_is_mocked():
        return None
    raw_key = raw_key.strip()
    if not raw_key.startswith("pdcts_"):
        return None
    prefix = raw_key[: 6 + 12]  # "pdcts_" + 12 chars
    candidates = cast(
        list[ApiKey],
        ApiKey.query.filter_by(prefix=prefix, revoked_at=None, deleted_at=None).limit(25).all(),
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


def _permanently_delete_api_key(*, user_id: str, key_id: str) -> bool:
    """Tombstone an API key while retaining usage rows that reference it."""
    try:
        key = ApiKey.query.filter_by(id=key_id, user_id=user_id).first()
        if key is None:
            return False
        if key.deleted_at is not None:
            return True
        now = _utc_now()
        key.deleted_at = now
        if key.revoked_at is None:
            key.revoked_at = now
        key.key_hash = _DUMMY_API_KEY_HASH
        key.name = None
        db.session.commit()
        return True
    except SQLAlchemyError:
        db.session.rollback()
        raise


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
    _capture_request_start_core()


def _auth_rate_limit_guard():
    _auth_rate_limit_guard_core(
        current_access_context=lambda: _current_access_context(),
        csrf_required=_csrf_required,
        check_rate_limit=lambda ctx: _check_rate_limit(cast(AccessContext, ctx)),
        check_endpoint_rate_limit=_check_endpoint_rate_limit,
        csrf_cookie_name=_CSRF_COOKIE_NAME,
    )


def _record_api_key_usage(response: Response) -> Response:
    _touch_api_key_last_used_if_needed()
    ctx = _current_access_context()
    return _record_api_key_usage_service(
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


def _set_security_headers(response: Response) -> Response:
    return _set_security_headers_core(response, is_running_on_fly=_is_running_on_fly)


def _register_request_hooks(target_app: Flask) -> None:
    _register_request_hooks_core(
        target_app,
        capture_request_start=_capture_request_start,
        auth_rate_limit_guard=_auth_rate_limit_guard,
        record_api_key_usage=_record_api_key_usage,
        set_security_headers=_set_security_headers,
    )

# Legacy helper names kept so route dependency wiring and tests can import one app module.
_encode_agreements_cursor = _core_encode_agreements_cursor
_decode_agreements_cursor = _core_decode_agreements_cursor

def _register_blueprints(target_app: Flask) -> None:
    api_ext = cast(_ApiExtension, cast(object, api))
    sections_deps, agreements_deps, reference_data_deps, tax_clauses_deps, _ = _build_route_deps()
    sections_list_blp = register_sections_routes(deps=sections_deps)
    agreements_blp, sections_blp, agreement_search_blp = register_agreements_routes(
        target_app,
        deps=agreements_deps,
    )
    taxonomy_blp, naics_blp, counsel_blp, dumps_blp, tax_clause_taxonomy_blp = register_reference_data_routes(
        deps=reference_data_deps,
    )
    tax_clauses_blp = register_tax_clauses_routes(deps=tax_clauses_deps)
    api_ext.register_blueprint(sections_list_blp)
    api_ext.register_blueprint(tax_clauses_blp)
    api_ext.register_blueprint(agreements_blp)
    api_ext.register_blueprint(sections_blp)
    api_ext.register_blueprint(agreement_search_blp)
    api_ext.register_blueprint(taxonomy_blp)
    api_ext.register_blueprint(tax_clause_taxonomy_blp)
    api_ext.register_blueprint(naics_blp)
    api_ext.register_blueprint(counsel_blp)
    api_ext.register_blueprint(dumps_blp)
    target_app.register_blueprint(
        register_mcp_routes(
            target_app,
            deps=McpDeps(
                sections_service_deps=sections_deps.sections_service_deps,
                agreements_deps=agreements_deps,
                reference_data_deps=reference_data_deps,
            ),
        )
    )


def _build_tax_clauses_service_deps() -> TaxClausesServiceDeps:
    return TaxClausesServiceDeps(
        db=db,
        AgreementCounsel=AgreementCounsel,
        Agreements=Agreements,
        Clauses=Clauses,
        Counsel=Counsel,
        TaxClauseAssignment=TaxClauseAssignment,
        _to_int=_to_int,
        _row_mapping_as_dict=_row_mapping_as_dict,
        _pagination_metadata=_pagination_metadata,
        _expand_tax_clause_taxonomy_standard_ids_cached=_expand_tax_clause_taxonomy_standard_ids_cached,
        _tax_clause_standard_id_filter_expr=_tax_clause_standard_id_filter_expr,
        _year_from_filing_date_value=_year_from_filing_date_value,
    )


def _build_route_deps() -> tuple[SectionsDeps, AgreementsDeps, ReferenceDataDeps, TaxClausesDeps, AuthDeps]:
    sections_service_deps = _build_sections_service_deps()
    sections_deps = SectionsDeps(
        _current_access_context=_current_access_context,
        sections_service_deps=sections_service_deps,
    )
    tax_clauses_deps = TaxClausesDeps(
        _current_access_context=_current_access_context,
        tax_clauses_service_deps=_build_tax_clauses_service_deps(),
    )
    agreements_deps = AgreementsDeps(
        Agreements=Agreements,
        AgreementCounsel=AgreementCounsel,
        Clauses=Clauses,
        Counsel=Counsel,
        LatestSectionsSearch=LatestSectionsSearch,
        Sections=Sections,
        TaxonomyL1=TaxonomyL1,
        TaxonomyL2=TaxonomyL2,
        TaxonomyL3=TaxonomyL3,
        TaxClauseAssignment=TaxClauseAssignment,
        XML=XML,
        _AGREEMENTS_SUMMARY_TTL_SECONDS=_AGREEMENTS_SUMMARY_TTL_SECONDS,
        _FILTER_OPTIONS_TTL_SECONDS=_FILTER_OPTIONS_TTL_SECONDS,
        _SECTION_ID_RE=_SECTION_ID_RE,
        _agreement_latest_xml_join_condition=_agreement_latest_xml_join_condition,
        _agreement_year_expr=_agreement_year_expr,
        _agreements_summary_cache=cast(
            dict[str, object], cast(object, _agreements_summary_cache)
        ),
        _agreements_summary_lock=_agreements_summary_lock,
        _coalesced_section_standard_ids=_coalesced_section_standard_ids,
        _current_access_context=_current_access_context,
        _decode_agreements_cursor=_decode_agreements_cursor,
        _expand_taxonomy_standard_ids_cached=_expand_taxonomy_standard_ids_cached,
        _encode_agreements_cursor=_encode_agreements_cursor,
        _filter_options_cache=cast(dict[str, object], cast(object, _filter_options_cache)),
        _filter_options_lock=_filter_options_lock,
        _load_query=_load_query,
        _pagination_metadata=_pagination_metadata,
        _parse_section_standard_ids=_parse_section_standard_ids,
        _redact_agreement_xml=_redact_agreement_xml,
        _row_mapping_as_dict=_row_mapping_as_dict,
        _schema_prefix=_schema_prefix,
        _section_latest_xml_join_condition=_section_latest_xml_join_condition,
        _standard_id_filter_expr=_standard_id_filter_expr,
        _standard_id_agreement_filter_expr=_standard_id_agreement_filter_expr,
        _estimated_query_row_count=_estimated_query_row_count,
        _to_int=_to_int,
        _year_from_filing_date_value=_year_from_filing_date_value,
        db=db,
        time=time,
    )
    reference_data_deps = ReferenceDataDeps(
        Counsel=Counsel,
        NaicsSector=NaicsSector,
        NaicsSubSector=NaicsSubSector,
        PUBLIC_DEV_BASE=PUBLIC_DEV_BASE,
        R2_BUCKET_NAME=R2_BUCKET_NAME,
        TaxClauseTaxonomyL1=TaxClauseTaxonomyL1,
        TaxClauseTaxonomyL2=TaxClauseTaxonomyL2,
        TaxClauseTaxonomyL3=TaxClauseTaxonomyL3,
        TaxonomyL1=TaxonomyL1,
        TaxonomyL2=TaxonomyL2,
        TaxonomyL3=TaxonomyL3,
        _DUMPS_CACHE_TTL_SECONDS=_DUMPS_CACHE_TTL_SECONDS,
        _DUMPS_MANIFEST_CACHE_TTL_SECONDS=_DUMPS_MANIFEST_CACHE_TTL_SECONDS,
        _COUNSEL_TTL_SECONDS=_COUNSEL_TTL_SECONDS,
        _NAICS_TTL_SECONDS=_NAICS_TTL_SECONDS,
        _TAXONOMY_TTL_SECONDS=_TAXONOMY_TTL_SECONDS,
        _counsel_cache=cast(dict[str, object], cast(object, _counsel_cache)),
        _counsel_lock=_counsel_lock,
        _dumps_cache=cast(dict[str, object], cast(object, _dumps_cache)),
        _dumps_cache_lock=_dumps_cache_lock,
        _dumps_manifest_cache=cast(dict[str, object], cast(object, _dumps_manifest_cache)),
        _dumps_manifest_cache_lock=_dumps_manifest_cache_lock,
        _naics_cache=cast(dict[str, object], cast(object, _naics_cache)),
        _naics_lock=_naics_lock,
        _tax_clause_taxonomy_cache=cast(
            dict[str, object], cast(object, _tax_clause_taxonomy_cache)
        ),
        _tax_clause_taxonomy_lock=_tax_clause_taxonomy_lock,
        _taxonomy_cache=cast(dict[str, object], cast(object, _taxonomy_cache)),
        _taxonomy_lock=_taxonomy_lock,
        client=client,
        db=db,
        time=time,
    )

    def _oauth_fetch_json_for_routes(
        url: str,
        *,
        data: dict[str, str] | None = None,
        json_body: dict[str, object] | None = None,
        headers: dict[str, str] | None = None,
        method: str | None = None,
    ) -> dict[str, object]:
        kwargs: dict[str, object] = {}
        if data is not None:
            kwargs["data"] = data
        if json_body is not None:
            kwargs["json_body"] = json_body
        if headers is not None:
            kwargs["headers"] = headers
        if method is not None:
            kwargs["method"] = method
        return cast(Any, _oauth_fetch_json)(url, **kwargs)

    def _verify_turnstile_token_for_routes(*, token: str) -> None:
        _verify_turnstile_token(token=token)

    def _authenticate_external_identity_for_routes(
        *, access_token: str, provider_name: str | None = None
    ):
        return _authenticate_external_identity(
            access_token=access_token, provider_name=provider_name
        )

    auth_deps = AuthDeps(
        ApiKey=ApiKey,
        ApiUsageDaily=ApiUsageDaily,
        AuthApiKeySchema=AuthApiKeySchema,
        AuthDeleteAccountSchema=AuthDeleteAccountSchema,
        AuthExternalSubject=AuthExternalSubject,
        AuthOAuthAuthorizationCode=AuthOAuthAuthorizationCode,
        AuthOAuthClient=AuthOAuthClient,
        AuthOAuthSigningKey=AuthOAuthSigningKey,
        AuthExternalSubjectLinkSchema=AuthExternalSubjectLinkSchema,
        AuthFlagInaccurateSchema=AuthFlagInaccurateSchema,
        AuthPasswordLoginSchema=AuthPasswordLoginSchema,
        AuthPasswordResetConfirmSchema=AuthPasswordResetConfirmSchema,
        AuthPasswordResetRequestSchema=AuthPasswordResetRequestSchema,
        AuthPasswordSignupSchema=AuthPasswordSignupSchema,
        AuthSession=AuthSession,
        AuthUser=AuthUser,
        LegalAcceptance=LegalAcceptance,
        _LEGAL_DOCS=_LEGAL_DOCS,
        _SESSION_COOKIE_NAME=_SESSION_COOKIE_NAME,
        _UUID_RE=_UUID_RE,
        _auth_db_is_configured=_auth_db_is_configured,
        _auth_enumeration_delay=_auth_enumeration_delay,
        _auth_is_mocked=_auth_is_mocked,
        _auth_session_transport=_auth_session_transport,
        _authenticate_external_identity=_authenticate_external_identity_for_routes,
        _clear_auth_cookies=_clear_auth_cookies,
        _create_api_key=_create_api_key,
        _permanently_delete_api_key=_permanently_delete_api_key,
        _csrf_cookie_value=_csrf_cookie_value,
        _ensure_current_legal_acceptances=_ensure_current_legal_acceptances,
        _frontend_base_url=_frontend_base_url,
        _oidc_fetch_json=_oauth_fetch_json_for_routes,
        _is_agreement_section_eligible=_is_agreement_section_eligible,
        _is_email_like=_is_email_like,
        _issue_session_token=_issue_session_token,
        _load_json=_load_json,
        _mock_auth=_mock_auth,
        _normalize_email=_normalize_email,
        _record_signon_event=_record_signon_event,
        _request_ip_address=_request_ip_address,
        _request_user_agent=_request_user_agent,
        _require_auth_db=_require_auth_db,
        _require_captcha_token=_require_captcha_token,
        _require_legal_acceptance=_require_legal_acceptance,
        _require_verified_user=_require_verified_user,
        _revoke_session_token=_revoke_session_token,
        _safe_next_path=_safe_next_path,
        _send_flag_notification_email=_send_flag_notification_email,
        _send_signup_notification_email=_send_signup_notification_email,
        _set_auth_cookies=_set_auth_cookies,
        _set_csrf_cookie=_set_csrf_cookie,
        _status_response=_status_response,
        _turnstile_enabled=_turnstile_enabled,
        _turnstile_site_key=_turnstile_site_key,
        _user_has_current_legal_acceptances=_user_has_current_legal_acceptances,
        _utc_now=_utc_now,
        _utc_today=_utc_today,
        _resolve_mcp_identity_provider_name=_resolve_mcp_identity_provider_name,
        _verify_turnstile_token=_verify_turnstile_token_for_routes,
        db=db,
        text=text,
        urlencode=urlencode,
    )
    return sections_deps, agreements_deps, reference_data_deps, tax_clauses_deps, auth_deps


def _register_app(target_app: Flask) -> None:
    _, _, _, _, auth_deps = _build_route_deps()

    _register_error_handlers(target_app)
    _register_request_hooks(target_app)
    _register_blueprints(target_app)
    _ = register_auth_routes(target_app, deps=auth_deps)


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
        _ensure_auth_schema_upgrades_auth_runtime(app)
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
