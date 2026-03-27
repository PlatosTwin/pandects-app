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


class UserLikeProtocol(Protocol):
    id: str
    email: str
    created_at: datetime
    password_hash: str | None
    email_verified_at: datetime | None


class ExternalIdentityLikeProtocol(Protocol):
    issuer: str
    subject: str


class ApiKeyLikeProtocol(Protocol):
    id: str
    name: str | None
    prefix: str
    created_at: datetime
    last_used_at: datetime | None
    revoked_at: datetime | None


class PasswordResetTokenRowProtocol(Protocol):
    used_at: datetime | None


class ToIntProtocol(Protocol):
    def __call__(self, value: object, *, default: int = 0) -> int: ...


class RowMappingAsDictProtocol(Protocol):
    def __call__(self, row: object) -> dict[str, object]: ...


class PaginationMetadataProtocol(Protocol):
    def __call__(
        self,
        *,
        total_count: int,
        page: int,
        page_size: int,
        has_next_override: bool | None = None,
        total_count_is_approximate: bool = False,
    ) -> dict[str, object]: ...


class RedactAgreementXmlProtocol(Protocol):
    def __call__(
        self,
        xml_content: str,
        *,
        focus_section_uuid: str | None,
        neighbor_sections: int,
    ) -> str: ...


class CreateApiKeyProtocol(Protocol):
    def __call__(self, *, user_id: str, name: str | None) -> tuple[ApiKeyLikeProtocol, str]: ...


class EnsureCurrentLegalAcceptancesProtocol(Protocol):
    def __call__(self, *, user_id: str, checked_at: datetime) -> None: ...


class FrontendGoogleCallbackRedirectProtocol(Protocol):
    def __call__(
        self,
        *,
        token: str | None,
        next_path: str | None,
        error: str | None,
    ) -> WerkzeugResponse: ...


class GoogleFetchJsonProtocol(Protocol):
    def __call__(self, url: str, *, data: dict[str, str] | None = None) -> dict[str, object]: ...


class GoogleVerifyIdTokenProtocol(Protocol):
    def __call__(self, id_token: str, *, expected_nonce: str | None = None) -> str: ...


class IsAgreementSectionEligibleProtocol(Protocol):
    def __call__(self, agreement_uuid: str, section_uuid: str | None) -> bool: ...


class IssueEmailVerificationTokenProtocol(Protocol):
    def __call__(self, *, user_id: str, email: str) -> str: ...


class IssuePasswordResetTokenProtocol(Protocol):
    def __call__(self, *, user_id: str, email: str) -> str: ...


class RecordSignonEventProtocol(Protocol):
    def __call__(self, *, user_id: str, provider: str, action: str) -> None: ...


class AuthenticateExternalIdentityProtocol(Protocol):
    def __call__(
        self,
        *,
        access_token: str,
        provider_name: str | None = None,
    ) -> ExternalIdentityLikeProtocol: ...


class RequireVerifiedUserProtocol(Protocol):
    def __call__(self) -> tuple[UserLikeProtocol, AccessContextProtocol]: ...


class SendEmailVerificationEmailProtocol(Protocol):
    def __call__(self, *, to_email: str, token: str) -> None: ...


class SendPasswordResetEmailProtocol(Protocol):
    def __call__(self, *, to_email: str, token: str) -> None: ...


class SendSignupNotificationEmailProtocol(Protocol):
    def __call__(self, *, new_user_email: str) -> None: ...


class SendFlagNotificationEmailProtocol(Protocol):
    def __call__(
        self,
        *,
        user_email: str,
        submitted_at: datetime,
        source: str,
        agreement_uuid: str,
        section_uuid: str | None,
        message: str | None,
        request_follow_up: bool,
        issue_types: list[str],
    ) -> None: ...


class SetAuthCookiesProtocol(Protocol):
    def __call__(self, resp: WerkzeugResponse, *, session_token: str) -> None: ...


class SetCsrfCookieProtocol(Protocol):
    def __call__(self, resp: WerkzeugResponse, value: str, *, max_age: int) -> None: ...


class StatusResponseProtocol(Protocol):
    def __call__(self, status: str, *, code: int = 200) -> Response: ...


class UserHasCurrentLegalAcceptancesProtocol(Protocol):
    def __call__(self, *, user_id: str) -> bool: ...


class VerifyTurnstileTokenProtocol(Protocol):
    def __call__(self, *, token: str) -> None: ...


@dataclass(frozen=True)
class SectionsServiceDeps:
    db: Any
    LatestSectionsSearch: Any
    Sections: Any
    _SEARCH_EXPLAIN_ESTIMATE_ENABLED: bool
    _to_int: ToIntProtocol
    _estimated_query_row_count: Callable[[object], int | None]
    _estimated_latest_sections_search_table_rows: Callable[[], int | None]
    _row_mapping_as_dict: RowMappingAsDictProtocol
    _pagination_metadata: PaginationMetadataProtocol
    _dedupe_preserve_order: Callable[[list[str]], list[str]]
    _expand_taxonomy_standard_ids_cached: Callable[[tuple[str, ...]], tuple[str, ...]]
    _standard_id_filter_expr: Callable[[list[str]], object]
    _parse_section_standard_ids: Callable[[object], list[str]]
    _year_from_filing_date_value: Callable[[object], int | None]


@dataclass(frozen=True)
class SectionsDeps:
    _current_access_context: Callable[[], AccessContextProtocol]
    sections_service_deps: SectionsServiceDeps


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
    _decode_agreements_cursor: Callable[[str | None], str | None]
    _encode_agreements_cursor: Callable[[str], str]
    _filter_options_cache: dict[str, Any]
    _filter_options_lock: Any
    _load_query: Callable[[Schema], dict[str, object]]
    _pagination_metadata: PaginationMetadataProtocol
    _parse_section_standard_ids: Callable[[object], list[str]]
    _redact_agreement_xml: RedactAgreementXmlProtocol
    _row_mapping_as_dict: RowMappingAsDictProtocol
    _schema_prefix: Callable[[], str]
    _section_latest_xml_join_condition: Callable[[], object]
    _to_int: ToIntProtocol
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
    AuthExternalSubject: Any
    AuthExternalSubjectLinkSchema: type[Schema]
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
    _auth_db_is_configured: Callable[[], bool]
    _auth_enumeration_delay: Callable[[], None]
    _auth_is_mocked: Callable[[], bool]
    _auth_session_transport: Callable[[], str]
    _authenticate_external_identity: AuthenticateExternalIdentityProtocol
    _clear_auth_cookies: Callable[[WerkzeugResponse], None]
    _clear_google_nonce_cookie: Callable[[WerkzeugResponse], None]
    _clear_google_oauth_cookie: Callable[[WerkzeugResponse], None]
    _create_api_key: CreateApiKeyProtocol
    _csrf_cookie_value: Callable[[], str | None]
    _ensure_current_legal_acceptances: EnsureCurrentLegalAcceptancesProtocol
    _frontend_base_url: Callable[[], str]
    _frontend_google_callback_redirect: FrontendGoogleCallbackRedirectProtocol
    _google_fetch_json: GoogleFetchJsonProtocol
    _google_nonce_cookie_value: Callable[[], str | None]
    _google_oauth_client_id: Callable[[], str]
    _google_oauth_client_secret: Callable[[], str]
    _google_oauth_flow_enabled: Callable[[], bool]
    _google_oauth_pkce_pair: Callable[[], tuple[str, str]]
    _google_oauth_redirect_uri: Callable[[], str]
    _google_verify_id_token: GoogleVerifyIdTokenProtocol
    _is_agreement_section_eligible: IsAgreementSectionEligibleProtocol
    _is_email_like: Callable[[str], bool]
    _issue_email_verification_token: IssueEmailVerificationTokenProtocol
    _issue_password_reset_token: IssuePasswordResetTokenProtocol
    _issue_session_token: Callable[[str], str]
    _load_email_verification_token: Callable[[str], tuple[str, str] | None]
    _load_google_oauth_cookie: Callable[[], dict[str, str] | None]
    _load_json: Callable[[Schema], dict[str, object]]
    _load_password_reset_token: Callable[[str], tuple[str, str, PasswordResetTokenRowProtocol | None] | None]
    _mock_auth: Any
    _normalize_email: Callable[[str], str]
    _record_signon_event: RecordSignonEventProtocol
    _request_ip_address: Callable[[], str | None]
    _request_user_agent: Callable[[], str | None]
    _require_auth_db: Callable[[], None]
    _require_captcha_token: Callable[[dict[str, object]], str]
    _require_legal_acceptance: Callable[[dict[str, object]], datetime]
    _require_verified_user: RequireVerifiedUserProtocol
    _revoke_session_token: Callable[[str], None]
    _safe_next_path: Callable[[str | None], str | None]
    _send_email_verification_email: SendEmailVerificationEmailProtocol
    _send_flag_notification_email: SendFlagNotificationEmailProtocol
    _send_password_reset_email: SendPasswordResetEmailProtocol
    _send_signup_notification_email: SendSignupNotificationEmailProtocol
    _set_auth_cookies: SetAuthCookiesProtocol
    _set_csrf_cookie: SetCsrfCookieProtocol
    _set_google_nonce_cookie: Callable[[WerkzeugResponse, str], None]
    _set_google_oauth_cookie: Callable[[WerkzeugResponse, dict[str, str]], None]
    _status_response: StatusResponseProtocol
    _turnstile_enabled: Callable[[], bool]
    _turnstile_site_key: Callable[[], str]
    _user_has_current_legal_acceptances: UserHasCurrentLegalAcceptancesProtocol
    _utc_now: Callable[[], datetime]
    _utc_today: Callable[[], date]
    _resolve_mcp_identity_provider_name: Callable[[str | None], str]
    _verify_turnstile_token: VerifyTurnstileTokenProtocol
    check_password_hash: Callable[[str, str], bool]
    db: Any
    generate_password_hash: Callable[[str], str]
    text: Callable[[str], object]
    urlencode: Callable[[dict[str, str]], str]
