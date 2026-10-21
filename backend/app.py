"""Application composition: config, typed route deps, hooks, and blueprints.

Runtime logic lives elsewhere (see ``backend/ARCHITECTURE.md``):

- ``backend/core/rate_limit.py``: per-tier, per-endpoint, and per-account limits
- ``backend/auth/access.py``: request access context and user gates
- ``backend/auth/api_keys.py``: API key issuance and lookup
- ``backend/services/notifications.py``: maintainer notification emails
- ``backend/services/dumps.py``: R2 client and active dump version
- ``backend/services/side_effects.py``: async task runner and usage recording

Names below marked as re-exports keep ``backend.app.<name>`` working for
tests and tooling that still reach into this module.
"""

import os
import random
import re
import time
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, cast
from urllib.parse import urlencode

import click
from dotenv import load_dotenv
from flask import Flask, Response, abort, has_app_context
from flask_smorest import Blueprint
from marshmallow import Schema
from sqlalchemy import text
# Re-export: tests hash fixture passwords through `backend.app`.
from werkzeug.security import generate_password_hash as generate_password_hash

# Load env vars from `backend/.env` regardless of process working directory.
_ = load_dotenv(dotenv_path=Path(__file__).with_name(".env"))
# Also allow repo/root `.env` (or process env) to supply values without overriding.
_ = load_dotenv()

from backend.extensions import db, api
from backend.models import (
    ApiKey,
    ApiUsageDaily,
    AuthExternalSubject,
    AuthOAuthAuthorizationCode,
    AuthOAuthClient,
    AuthOAuthRefreshToken,
    AuthOAuthSigningKey,
    AuthOAuthUserGrant,
    AuthSession,
    AuthUser,
    Favorite,
    FavoriteProject,
    FavoriteProjectAssignment,
    FavoriteTag,
    FavoriteTagAssignment,
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
from backend.auth import access as _access
from backend.auth import api_keys as _api_keys
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
    LatestSectionsSearchStandardId,
    NaicsSector,
    NaicsSubSector,
    SectionTextSearch,
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
from backend.routes.favorites import FavoritesDeps, register_favorites_routes
from backend.services.tax_clauses_service import TaxClausesServiceDeps
from backend.mcp.routes import McpDeps, register_mcp_routes
from backend.routes.telemetry import register_telemetry_routes
from backend.core import rate_limit as _rate_limit
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
    capture_request_start as _capture_request_start,
    register_request_hooks as _register_request_hooks_core,
    set_security_headers as _set_security_headers_core,
)
from backend.core.runtime_utils import (
    current_app_object as _current_app_object,
    decode_agreements_cursor as _decode_agreements_cursor,
    dedupe_preserve_order as _dedupe_preserve_order,
    encode_agreements_cursor as _encode_agreements_cursor,
    load_json as _core_load_json,
    load_query as _core_load_query,
    pagination_metadata as _pagination_metadata,
    row_mapping_as_dict as _row_mapping_as_dict,
    to_int as _to_int,
    utc_now as _utc_now,
    utc_today as _utc_today,
)
from backend.auth.captcha_runtime import (
    require_captcha_token as _require_captcha_token,
    require_turnstile_configured as _require_turnstile_configured,
    turnstile_enabled as _turnstile_enabled,
    turnstile_site_key as _turnstile_site_key,
    verify_turnstile_token as _verify_turnstile_token,
)
from backend.auth.email_runtime import (
    assert_zitadel_notification_signing_key_configured as _assert_zitadel_notification_signing_key_configured,
    frontend_base_url as _frontend_base_url,
    is_email_like as _is_email_like,
    normalize_email as _normalize_email,
)
from backend.auth.legal_runtime import (
    _LEGAL_DOCS,
    ensure_current_legal_acceptances as _ensure_current_legal_acceptances,
    record_signon_event as _record_signon_event,
    require_legal_acceptance as _require_legal_acceptance,
    user_has_current_legal_acceptances as _user_has_current_legal_acceptances,
)
from backend.auth.oauth_runtime import (
    oauth_fetch_json as _oauth_fetch_json,
)
from backend.auth.session_runtime import (
    AUTH_DATABASE_URI,
    AccessContext,
    _CSRF_COOKIE_NAME,
    _SESSION_COOKIE_NAME,
    _mock_auth,
    auth_db_is_configured as _auth_db_is_configured,
    auth_is_mocked as _auth_is_mocked,
    auth_session_transport as _auth_session_transport,
    clear_auth_cookies as _clear_auth_cookies,
    csrf_cookie_value as _csrf_cookie_value,
    csrf_required as _csrf_required,
    ensure_auth_tables_exist as _ensure_auth_tables_exist,
    is_running_on_fly as _is_running_on_fly,
    issue_session_token as _issue_session_token,
    request_ip_address as _request_ip_address,
    request_user_agent as _request_user_agent,
    require_auth_db as _require_auth_db,
    revoke_session_token as _revoke_session_token,
    safe_next_path as _safe_next_path,
    set_auth_cookies as _set_auth_cookies,
    set_csrf_cookie as _set_csrf_cookie,
)
from backend.search_counts import (
    cached_exact_query_count as _search_counts_cached_exact_query_count,
)
from backend.services import dumps as _dumps
from backend.services import notifications as _notifications
from backend.services.sections_service import (
    estimated_latest_sections_search_table_rows as _svc_estimated_latest_sections_search_table_rows,
    estimated_query_row_count as _svc_estimated_query_row_count,
)
from backend.services.side_effects import record_request_usage as _record_request_usage

# Retained for tests and modules that still import `metadata` from `backend.app`.
metadata = _main_db_metadata

_SKIP_MAIN_DB_REFLECTION = os.environ.get("SKIP_MAIN_DB_REFLECTION", "").strip() == "1"
_ENABLE_MAIN_DB_REFLECTION = (
    os.environ.get("ENABLE_MAIN_DB_REFLECTION", "1").strip() != "0"
)

# ── Re-exports (tests and app-module indirection) ─────────────────────────
# Mutable state is the same object as in the owning module, so `.clear()` and
# item assignment through `backend.app` still act on live state. Scalars
# (limits, intervals) must be patched on the owning module instead.
_rate_limit_state = _rate_limit.rate_limit_state
_endpoint_rate_limit_state = _rate_limit.endpoint_rate_limit_state
_account_login_failure_state = _rate_limit.account_login_failure_state
_ENDPOINT_RATE_LIMITS = _rate_limit.ENDPOINT_RATE_LIMITS
_ACCOUNT_LOGIN_FAILURE_LIMIT = _rate_limit.ACCOUNT_LOGIN_FAILURE_LIMIT
_apply_search_read_rate_limit = _rate_limit.apply_search_read_rate_limit
_prune_rate_limit_state = _rate_limit.prune_rate_limit_state
_check_rate_limit = _rate_limit.check_rate_limit
_check_endpoint_rate_limit = _rate_limit.check_endpoint_rate_limit
_check_account_login_rate_limit = _rate_limit.check_account_login_rate_limit
_record_account_login_failure = _rate_limit.record_account_login_failure
_record_account_login_success = _rate_limit.record_account_login_success
_account_login_key = _rate_limit.account_login_key

_api_key_last_used_touch_state = _api_keys.api_key_last_used_touch_state
_DUMMY_API_KEY_HASH = _api_keys.DUMMY_API_KEY_HASH
_lookup_api_key = _api_keys.lookup_api_key
_create_api_key = _api_keys.create_api_key
_permanently_delete_api_key = _api_keys.permanently_delete_api_key
_should_touch_api_key_last_used = _api_keys.should_touch_api_key_last_used

_current_access_context = _access.current_access_context
_require_user = _access.require_user
_require_verified_user = _access.require_verified_user
_user_id_is_verified = _access.user_id_is_verified
_access_account_predates_today = _access.access_account_predates_today
_temporary_access_gate = _access.temporary_access_gate

_send_signup_notification_email = _notifications.send_signup_notification_email
_send_flag_notification_email = _notifications.send_flag_notification_email
_is_agreement_section_eligible = _notifications.is_agreement_section_eligible

R2_BUCKET_NAME = _dumps.R2_BUCKET_NAME
R2_ENDPOINT = _dumps.R2_ENDPOINT
PUBLIC_DEV_BASE = _dumps.PUBLIC_DEV_BASE
_fetch_latest_dump_version = _dumps.fetch_latest_dump_version
_r2_client: _dumps.S3Client | None = _dumps.build_r2_client()
client = _r2_client

# Contract surface consumed by `backend.routes.auth` via app_module indirection.
_AUTH_SCHEMA_EXPORTS = (
    AuthApiKeySchema,
    AuthDeleteAccountSchema,
    AuthExternalSubjectLinkSchema,
    AuthFlagInaccurateSchema,
)
_SECTION_ID_RE = re.compile(
    r"^(?:[0-9a-fA-F]{16}|[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12})$"
)
_UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


class _ApiExtension(Protocol):
    def init_app(self, app: Flask, **kwargs: object) -> None:
        ...

    def register_blueprint(
        self, blp: Blueprint, *, parameters: object | None = None, **options: object
    ) -> None:
        ...


# ── Cached exact result counts for paginated search ───────────────────────
# A search's match count depends only on its filter signature, so it is the
# same on every page. Memoizing it per signature turns one COUNT-per-page into
# one COUNT-per-search; values stay exact, just reused while fresh.
_SEARCH_COUNT_CACHE_TTL_SECONDS = float(os.environ.get("SEARCH_COUNT_CACHE_TTL_SECONDS", "120"))
_SEARCH_COUNT_CACHE_MAX_KEYS = int(os.environ.get("SEARCH_COUNT_CACHE_MAX_KEYS", "10000"))
_search_count_cache: dict[str, tuple[float, int]] = {}
_search_count_cache_lock = Lock()
_SEARCH_EXPLAIN_ESTIMATE_ENABLED = (
    os.environ.get("SEARCH_EXPLAIN_ESTIMATE_ENABLED", "1").strip() != "0"
)


def _cached_exact_query_count(query: object, *, cache_key: str | None) -> int:
    """Exact COUNT(*) for a search query, memoized per filter signature + dump.

    `cache_key=None` bypasses the cache for an authoritative fresh count.
    """
    namespaced_key = (
        None if cache_key is None else f"{_dumps.current_dump_version_token()}|{cache_key}"
    )
    return _search_counts_cached_exact_query_count(
        query=query,
        cache_key=namespaced_key,
        cache=_search_count_cache,
        lock=_search_count_cache_lock,
        ttl_seconds=_SEARCH_COUNT_CACHE_TTL_SECONDS,
        max_keys=_SEARCH_COUNT_CACHE_MAX_KEYS,
        to_int=_to_int,
        now=time.time,
    )


# ── Small adapters handed to route deps ───────────────────────────────────
def _schema_prefix() -> str:
    """Return the configured schema prefix for raw SQL assembly."""
    if has_app_context():
        config = _app_config_map(_current_app_object())
        raw = config.get("MAIN_DB_SCHEMA", "")
        value = raw.strip() if isinstance(raw, str) else ""
    else:
        value = _main_db_schema_from_env()
    return f"{value}." if value else ""


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


def _load_json(schema: Schema) -> dict[str, object]:
    return _core_load_json(schema, json_error=_json_error)


def _load_query(schema: Schema) -> dict[str, object]:
    return _core_load_query(schema, json_error=_json_error)


def _auth_enumeration_delay() -> None:
    time.sleep(random.uniform(0.15, 0.35))


# ── Sections service deps ─────────────────────────────────────────────────
def _build_sections_service_deps() -> SectionsServiceDeps:
    return SectionsServiceDeps(
        db=db,
        AgreementCounsel=AgreementCounsel,
        Agreements=Agreements,
        Counsel=Counsel,
        LatestSectionsSearch=LatestSectionsSearch,
        LatestSectionsSearchStandardId=LatestSectionsSearchStandardId,
        SectionTextSearch=SectionTextSearch,
        Sections=Sections,
        _SEARCH_EXPLAIN_ESTIMATE_ENABLED=_SEARCH_EXPLAIN_ESTIMATE_ENABLED,
        _to_int=_to_int,
        _cached_exact_query_count=_cached_exact_query_count,
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


_shared_sections_service_deps: SectionsServiceDeps | None = None


def _sections_service_deps_for_estimates() -> SectionsServiceDeps:
    """One deps instance for the row-estimate adapters, built on first use."""
    global _shared_sections_service_deps
    if _shared_sections_service_deps is None:
        _shared_sections_service_deps = _build_sections_service_deps()
    return _shared_sections_service_deps


def _estimated_query_row_count(query: object) -> int | None:
    return _svc_estimated_query_row_count(_sections_service_deps_for_estimates(), query)


def _estimated_latest_sections_search_table_rows() -> int | None:
    return _svc_estimated_latest_sections_search_table_rows(
        _sections_service_deps_for_estimates()
    )


def _build_tax_clauses_service_deps() -> TaxClausesServiceDeps:
    return TaxClausesServiceDeps(
        db=db,
        AgreementCounsel=AgreementCounsel,
        Agreements=Agreements,
        Clauses=Clauses,
        Counsel=Counsel,
        TaxClauseAssignment=TaxClauseAssignment,
        TaxClauseTaxonomyL1=TaxClauseTaxonomyL1,
        TaxClauseTaxonomyL2=TaxClauseTaxonomyL2,
        TaxClauseTaxonomyL3=TaxClauseTaxonomyL3,
        XML=XML,
        _to_int=_to_int,
        _cached_exact_query_count=_cached_exact_query_count,
        _row_mapping_as_dict=_row_mapping_as_dict,
        _pagination_metadata=_pagination_metadata,
        _expand_tax_clause_taxonomy_standard_ids_cached=_expand_tax_clause_taxonomy_standard_ids_cached,
        _tax_clause_standard_id_filter_expr=_tax_clause_standard_id_filter_expr,
        _year_from_filing_date_value=_year_from_filing_date_value,
    )


# ── Route deps ────────────────────────────────────────────────────────────
def _build_route_deps() -> tuple[SectionsDeps, AgreementsDeps, ReferenceDataDeps, TaxClausesDeps, AuthDeps]:
    sections_deps = SectionsDeps(
        _current_access_context=_current_access_context,
        sections_service_deps=_build_sections_service_deps(),
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
        LatestSectionsSearchStandardId=LatestSectionsSearchStandardId,
        SectionTextSearch=SectionTextSearch,
        Sections=Sections,
        TaxonomyL1=TaxonomyL1,
        TaxonomyL2=TaxonomyL2,
        TaxonomyL3=TaxonomyL3,
        TaxClauseAssignment=TaxClauseAssignment,
        TaxClauseTaxonomyL1=TaxClauseTaxonomyL1,
        TaxClauseTaxonomyL2=TaxClauseTaxonomyL2,
        TaxClauseTaxonomyL3=TaxClauseTaxonomyL3,
        XML=XML,
        _SECTION_ID_RE=_SECTION_ID_RE,
        _agreement_latest_xml_join_condition=_agreement_latest_xml_join_condition,
        _agreement_year_expr=_agreement_year_expr,
        _coalesced_section_standard_ids=_coalesced_section_standard_ids,
        _current_access_context=_current_access_context,
        _decode_agreements_cursor=_decode_agreements_cursor,
        _expand_taxonomy_standard_ids_cached=_expand_taxonomy_standard_ids_cached,
        _expand_tax_clause_taxonomy_standard_ids_cached=_expand_tax_clause_taxonomy_standard_ids_cached,
        _tax_clause_standard_id_filter_expr=_tax_clause_standard_id_filter_expr,
        _encode_agreements_cursor=_encode_agreements_cursor,
        _load_query=_load_query,
        _pagination_metadata=_pagination_metadata,
        _parse_section_standard_ids=_parse_section_standard_ids,
        _redact_agreement_xml=_redact_agreement_xml,
        _row_mapping_as_dict=_row_mapping_as_dict,
        _schema_prefix=_schema_prefix,
        _section_latest_xml_join_condition=_section_latest_xml_join_condition,
        _standard_id_filter_expr=_standard_id_filter_expr,
        _standard_id_agreement_filter_expr=_standard_id_agreement_filter_expr,
        _cached_exact_query_count=_cached_exact_query_count,
        _estimated_query_row_count=_estimated_query_row_count,
        _to_int=_to_int,
        _year_from_filing_date_value=_year_from_filing_date_value,
        db=db,
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
        client=_r2_client,
        db=db,
    )

    # The three wrappers below resolve their target through this module at
    # call time, so tests can swap `backend.app._oauth_fetch_json` and friends.
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
        AuthOAuthRefreshToken=AuthOAuthRefreshToken,
        AuthOAuthSigningKey=AuthOAuthSigningKey,
        AuthOAuthUserGrant=AuthOAuthUserGrant,
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
        _check_account_login_rate_limit=_check_account_login_rate_limit,
        _record_account_login_failure=_record_account_login_failure,
        _record_account_login_success=_record_account_login_success,
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
        _require_turnstile_configured=_require_turnstile_configured,
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


def _build_favorites_deps() -> FavoritesDeps:
    return FavoritesDeps(
        Favorite=Favorite,
        FavoriteProject=FavoriteProject,
        FavoriteProjectAssignment=FavoriteProjectAssignment,
        FavoriteTag=FavoriteTag,
        FavoriteTagAssignment=FavoriteTagAssignment,
        Sections=Sections,
        Clauses=Clauses,
        Agreements=Agreements,
        XML=XML,
        db=db,
        _section_latest_xml_join_condition=_section_latest_xml_join_condition,
        _agreement_latest_xml_join_condition=_agreement_latest_xml_join_condition,
        _coalesced_section_standard_ids=_coalesced_section_standard_ids,
        _parse_section_standard_ids=_parse_section_standard_ids,
        _require_auth_db=_require_auth_db,
        _require_verified_user=_require_verified_user,
        _auth_is_mocked=_auth_is_mocked,
        _load_json=_load_json,
        _utc_now=_utc_now,
    )


# ── Request hooks ─────────────────────────────────────────────────────────
def _auth_rate_limit_guard() -> None:
    _auth_rate_limit_guard_core(
        current_access_context=_current_access_context,
        csrf_required=_csrf_required,
        check_rate_limit=lambda ctx: _check_rate_limit(cast(AccessContext, ctx)),
        check_endpoint_rate_limit=_check_endpoint_rate_limit,
        csrf_cookie_name=_CSRF_COOKIE_NAME,
    )


def _record_api_key_usage(response: Response) -> Response:
    _api_keys.touch_api_key_last_used_if_needed()
    return _record_request_usage(_current_access_context(), response)


def _set_security_headers(response: Response) -> Response:
    return _set_security_headers_core(response, is_running_on_fly=_is_running_on_fly)


def _register_request_hooks(target_app: Flask) -> None:
    _register_request_hooks_core(
        target_app,
        capture_request_start=_capture_request_start,
        auth_rate_limit_guard=_auth_rate_limit_guard,
        record_api_key_usage=_record_api_key_usage,
        set_security_headers=_set_security_headers,
        populate_dump_version=_dumps.populate_dump_version,
        attach_dump_version_header=_dumps.attach_dump_version_header,
    )
    # Temporary lockdown gate; registered after the core guard so g.access_ctx is set.
    _temp_access_gate: object = target_app.before_request(_temporary_access_gate)


# ── Blueprints ────────────────────────────────────────────────────────────
def _register_blueprints(target_app: Flask) -> None:
    api_ext = cast(_ApiExtension, cast(object, api))
    sections_deps, agreements_deps, reference_data_deps, tax_clauses_deps, auth_deps = (
        _build_route_deps()
    )
    sections_list_blp = register_sections_routes(deps=sections_deps)
    agreements_blp, sections_blp, agreement_search_blp = register_agreements_routes(
        target_app,
        deps=agreements_deps,
    )
    taxonomy_blp, naics_blp, counsel_blp, dumps_blp, tax_clause_taxonomy_blp, changelog_blp = register_reference_data_routes(
        deps=reference_data_deps,
    )
    tax_clauses_blp = register_tax_clauses_routes(deps=tax_clauses_deps)
    api_ext.register_blueprint(sections_list_blp)
    api_ext.register_blueprint(agreements_blp)
    api_ext.register_blueprint(sections_blp)
    api_ext.register_blueprint(tax_clauses_blp)
    api_ext.register_blueprint(agreement_search_blp)
    api_ext.register_blueprint(taxonomy_blp)
    api_ext.register_blueprint(tax_clause_taxonomy_blp)
    api_ext.register_blueprint(naics_blp)
    api_ext.register_blueprint(counsel_blp)
    api_ext.register_blueprint(dumps_blp)
    api_ext.register_blueprint(changelog_blp)
    target_app.register_blueprint(register_telemetry_routes())
    target_app.register_blueprint(
        register_mcp_routes(
            target_app,
            deps=McpDeps(
                sections_service_deps=sections_deps.sections_service_deps,
                agreements_deps=agreements_deps,
                reference_data_deps=reference_data_deps,
                tax_clauses_service_deps=tax_clauses_deps.tax_clauses_service_deps,
            ),
        )
    )
    _ = register_auth_routes(target_app, deps=auth_deps)
    _ = register_favorites_routes(target_app, deps=_build_favorites_deps())


# ── App factory ───────────────────────────────────────────────────────────
def _configure_app(
    target_app: Flask, *, config_overrides: dict[str, object] | None = None
) -> None:
    _configure_app_core(
        target_app,
        auth_database_uri=AUTH_DATABASE_URI,
        config_overrides=config_overrides,
    )


def _healthz() -> tuple[str, int]:
    # Liveness only: Fly's deploy check. Stays off both DBs so a DB blip can't
    # mark every machine unhealthy; /v1/auth/health is the deeper check.
    return "ok", 200


def _register_app(target_app: Flask) -> None:
    _ = target_app.add_url_rule("/healthz", "healthz", _healthz, methods=["GET"])
    _register_error_handlers(target_app)
    _register_request_hooks(target_app)
    _register_blueprints(target_app)


def create_app(*, config_overrides: dict[str, object] | None = None) -> Flask:
    target_app = Flask(__name__)
    _configure_app(target_app, config_overrides=config_overrides)
    _assert_zitadel_notification_signing_key_configured()
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
    """Migrate the auth database to the current schema (Alembic upgrade head)."""
    if _auth_is_mocked():
        raise click.ClickException("Auth DB is not available in mock auth mode.")
    if not _auth_db_is_configured():
        raise click.ClickException(
            "Auth DB is not configured. Set AUTH_DATABASE_URI or DATABASE_URL."
        )

    from backend.init_auth_db import upgrade_auth_db

    try:
        url = upgrade_auth_db()
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(f"Auth DB migrated ({url}).")


# OpenAPI specs: `python -m backend.scripts.export_openapi` (see ARCHITECTURE.md, "API Contract").


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    debug = os.environ.get("FLASK_DEBUG", "").strip().lower() in ("1", "true", "yes")
    app.run(debug=debug, port=port)
