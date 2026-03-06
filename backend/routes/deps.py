from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from re import Pattern
from typing import Any, Protocol

from flask import Response
from marshmallow import Schema
from werkzeug.wrappers.response import Response as WerkzeugResponse


class AccessContextProtocol(Protocol):
    tier: str
    user_id: str | None
    api_key_id: str | None

    @property
    def is_authenticated(self) -> bool: ...


@dataclass(frozen=True)
class SearchServiceDeps:
    db: Any
    LatestSectionsSearch: Any
    Sections: Any
    _SEARCH_EXPLAIN_ESTIMATE_ENABLED: bool
    _to_int: Callable[..., int]
    _estimated_query_row_count: Callable[[object], int | None]
    _estimated_latest_sections_search_table_rows: Callable[[], int | None]
    _row_mapping_as_dict: Callable[..., dict[str, object]]
    _pagination_metadata: Callable[..., dict[str, object]]
    _dedupe_preserve_order: Callable[..., list[str]]
    _expand_taxonomy_standard_ids_cached: Callable[..., tuple[str, ...]]
    _standard_id_filter_expr: Callable[[list[str]], object]
    _parse_section_standard_ids: Callable[..., list[str]]
    _year_from_filing_date_value: Callable[..., int | None]


@dataclass(frozen=True)
class SearchDeps:
    _current_access_context: Callable[[], AccessContextProtocol]
    search_service_deps: SearchServiceDeps


@dataclass(frozen=True)
class AgreementsDeps:
    Agreements: Any
    Sections: Any
    XML: Any
    _AGREEMENTS_SUMMARY_TTL_SECONDS: int
    _FILTER_OPTIONS_TTL_SECONDS: int
    _SECTION_ID_RE: Pattern[str]
    _agreement_latest_xml_join_condition: Callable[[], object]
    _agreement_year_expr: Callable[..., Any]
    _agreements_summary_cache: dict[str, Any]
    _agreements_summary_lock: Any
    _coalesced_section_standard_ids: Callable[..., Any]
    _current_access_context: Callable[[], AccessContextProtocol]
    _decode_agreements_cursor: Callable[..., str | None]
    _encode_agreements_cursor: Callable[..., str]
    _filter_options_cache: dict[str, Any]
    _filter_options_lock: Any
    _load_query: Callable[..., dict[str, object]]
    _pagination_metadata: Callable[..., dict[str, object]]
    _parse_section_standard_ids: Callable[..., list[str]]
    _redact_agreement_xml: Callable[..., str]
    _row_mapping_as_dict: Callable[..., dict[str, object]]
    _schema_prefix: Callable[..., str]
    _section_latest_xml_join_condition: Callable[[], object]
    _to_int: Callable[..., int]
    db: Any
    time: Any


@dataclass(frozen=True)
class ReferenceDataDeps:
    NaicsSector: Any
    NaicsSubSector: Any
    PUBLIC_DEV_BASE: str
    R2_BUCKET_NAME: str
    TaxonomyL1: Any
    TaxonomyL2: Any
    TaxonomyL3: Any
    _DUMPS_CACHE_TTL_SECONDS: int
    _DUMPS_MANIFEST_CACHE_TTL_SECONDS: int
    _NAICS_TTL_SECONDS: int
    _TAXONOMY_TTL_SECONDS: int
    _dumps_cache: dict[str, Any]
    _dumps_cache_lock: Any
    _dumps_manifest_cache: dict[str, Any]
    _dumps_manifest_cache_lock: Any
    _naics_cache: dict[str, Any]
    _naics_lock: Any
    _taxonomy_cache: dict[str, Any]
    _taxonomy_lock: Any
    client: Any
    db: Any
    time: Any


@dataclass(frozen=True)
class AuthDeps:
    ApiKey: Any
    ApiUsageDaily: Any
    AuthApiKeySchema: type[Schema]
    AuthDeleteAccountSchema: type[Schema]
    AuthEmailSchema: type[Schema]
    AuthFlagInaccurateSchema: type[Schema]
    AuthGoogleCredentialSchema: type[Schema]
    AuthLoginSchema: type[Schema]
    AuthPasswordResetSchema: type[Schema]
    AuthRegisterSchema: type[Schema]
    AuthSession: Any
    AuthTokenSchema: type[Schema]
    AuthUser: Any
    LegalAcceptance: Any
    _LEGAL_DOCS: dict[str, dict[str, str]]
    _SESSION_COOKIE_NAME: str
    _UUID_RE: Pattern[str]
    _auth_db_is_configured: Callable[..., bool]
    _auth_enumeration_delay: Callable[..., None]
    _auth_is_mocked: Callable[..., bool]
    _auth_session_transport: Callable[..., str]
    _clear_auth_cookies: Callable[..., None]
    _clear_google_nonce_cookie: Callable[..., None]
    _clear_google_oauth_cookie: Callable[..., None]
    _create_api_key: Callable[..., tuple[Any, str]]
    _csrf_cookie_value: Callable[..., str | None]
    _ensure_current_legal_acceptances: Callable[..., None]
    _frontend_base_url: Callable[..., str]
    _frontend_google_callback_redirect: Callable[..., WerkzeugResponse]
    _google_fetch_json: Callable[..., dict[str, object]]
    _google_nonce_cookie_value: Callable[..., str | None]
    _google_oauth_client_id: Callable[..., str]
    _google_oauth_client_secret: Callable[..., str]
    _google_oauth_flow_enabled: Callable[..., bool]
    _google_oauth_pkce_pair: Callable[..., tuple[str, str]]
    _google_oauth_redirect_uri: Callable[..., str]
    _google_verify_id_token: Callable[..., str]
    _is_agreement_section_eligible: Callable[..., bool]
    _is_email_like: Callable[..., bool]
    _issue_email_verification_token: Callable[..., str]
    _issue_password_reset_token: Callable[..., str]
    _issue_session_token: Callable[..., str]
    _load_email_verification_token: Callable[..., tuple[str, str] | None]
    _load_google_oauth_cookie: Callable[..., dict[str, str] | None]
    _load_json: Callable[[Schema], dict[str, object]]
    _load_password_reset_token: Callable[[str], tuple[str, str, Any | None] | None]
    _mock_auth: Any
    _normalize_email: Callable[..., str]
    _record_signon_event: Callable[..., None]
    _request_ip_address: Callable[..., str | None]
    _request_user_agent: Callable[..., str | None]
    _require_auth_db: Callable[..., None]
    _require_captcha_token: Callable[..., str]
    _require_legal_acceptance: Callable[[dict[str, object]], datetime]
    _require_verified_user: Callable[[], tuple[Any, AccessContextProtocol]]
    _revoke_session_token: Callable[..., None]
    _safe_next_path: Callable[..., str | None]
    _send_email_verification_email: Callable[..., None]
    _send_flag_notification_email: Callable[..., None]
    _send_password_reset_email: Callable[..., None]
    _send_signup_notification_email: Callable[..., None]
    _set_auth_cookies: Callable[..., None]
    _set_csrf_cookie: Callable[..., None]
    _set_google_nonce_cookie: Callable[..., None]
    _set_google_oauth_cookie: Callable[..., None]
    _status_response: Callable[..., Response]
    _turnstile_enabled: Callable[..., bool]
    _turnstile_site_key: Callable[..., str]
    _user_has_current_legal_acceptances: Callable[..., bool]
    _utc_now: Callable[[], datetime]
    _utc_today: Callable[[], date]
    _verify_turnstile_token: Callable[..., None]
    check_password_hash: Callable[..., bool]
    db: Any
    generate_password_hash: Callable[..., str]
    text: Callable[[str], object]
    urlencode: Callable[..., str]
