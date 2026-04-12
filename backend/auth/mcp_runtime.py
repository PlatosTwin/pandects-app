from __future__ import annotations

import json
import os
from dataclasses import dataclass
from base64 import b64encode
from functools import lru_cache
from typing import Any, cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from flask import request
from sqlalchemy.exc import SQLAlchemyError

from backend.auth.mcp_oauth_runtime import decode_access_token, mcp_oauth_issuer, public_pem_from_private_pem
from backend.auth.session_runtime import AccessContext
from backend.extensions import db
from backend.models import AuthExternalSubject, AuthOAuthSigningKey, AuthUser


class McpAuthError(Exception):
    def __init__(self, *, status_code: int, message: str, www_authenticate: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.www_authenticate = www_authenticate


@dataclass(frozen=True)
class McpPrincipal:
    access_context: AccessContext
    scopes: frozenset[str]
    issuer: str
    subject: str
    user_id: str


@dataclass(frozen=True)
class ExternalIdentity:
    issuer: str
    subject: str
    scopes: frozenset[str]
    audiences: frozenset[str]
    claims: dict[str, object]


_DEFAULT_MCP_SCOPES = (
    "sections:search",
    "agreements:search",
    "agreements:read",
    "agreements:read_fulltext",
)
_DEFAULT_ZITADEL_ROLE_SCOPE_MAP = {
    "sections_search": "sections:search",
    "agreements_search": "agreements:search",
    "agreements_read": "agreements:read",
    "agreements_read_fulltext": "agreements:read_fulltext",
}
_DEFAULT_JWT_ALGORITHMS = (
    "RS256",
    "RS384",
    "RS512",
    "PS256",
    "PS384",
    "PS512",
    "ES256",
    "ES384",
    "ES512",
)
_mcp_jwk_client: object | None = None
_mcp_identity_provider: "McpIdentityProvider | None" = None


def mcp_protocol_version() -> str:
    raw = os.environ.get("MCP_PROTOCOL_VERSION", "").strip()
    return raw or "2025-11-05"


def mcp_server_name() -> str:
    raw = os.environ.get("MCP_SERVER_NAME", "").strip()
    return raw or "pandects-mcp"


def mcp_server_version() -> str:
    raw = os.environ.get("MCP_SERVER_VERSION", "").strip()
    return raw or "0.1.0"


def mcp_identity_provider_name() -> str:
    raw = os.environ.get("MCP_IDENTITY_PROVIDER", "").strip().lower()
    return raw or "oidc"


def resolve_mcp_identity_provider_name(provider_name: str | None = None) -> str:
    if isinstance(provider_name, str) and provider_name.strip():
        return provider_name.strip().lower()
    return mcp_identity_provider_name()


def mcp_public_base_url() -> str:
    raw = os.environ.get("MCP_PUBLIC_BASE_URL", "").strip()
    if raw:
        return raw.rstrip("/")
    api_base = os.environ.get("PUBLIC_API_BASE_URL", "").strip()
    if api_base:
        return api_base.rstrip("/")
    return "http://localhost:5000"


def mcp_resource_url() -> str:
    raw = os.environ.get("MCP_RESOURCE_URL", "").strip()
    if raw:
        return raw
    return f"{mcp_public_base_url()}/mcp"


def mcp_resource_metadata_url() -> str:
    return f"{mcp_public_base_url()}/.well-known/oauth-protected-resource"


def mcp_oidc_issuer() -> str:
    raw = os.environ.get("MCP_OIDC_ISSUER", "").strip()
    if not raw:
        raise RuntimeError("MCP_OIDC_ISSUER is required for MCP auth.")
    return raw.rstrip("/")


def mcp_authorization_server_url() -> str:
    return mcp_oauth_issuer()


def mcp_oidc_audiences() -> tuple[str, ...]:
    raw = os.environ.get("MCP_OIDC_AUDIENCE", "").strip()
    if not raw:
        raise RuntimeError("MCP_OIDC_AUDIENCE is required for MCP auth.")
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def mcp_supported_scopes() -> tuple[str, ...]:
    raw = os.environ.get("MCP_SUPPORTED_SCOPES", "").strip()
    if not raw:
        return _DEFAULT_MCP_SCOPES
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def mcp_jwt_algorithms() -> tuple[str, ...]:
    raw = os.environ.get("MCP_OIDC_SIGNING_ALGORITHMS", "").strip()
    if not raw:
        return _DEFAULT_JWT_ALGORITHMS
    return tuple(part.strip() for part in raw.split(",") if part.strip())


@lru_cache(maxsize=1)
def _oidc_discovery_document() -> dict[str, object]:
    explicit = os.environ.get("MCP_OIDC_DISCOVERY_URL", "").strip()
    discovery_url = explicit or f"{mcp_oidc_issuer()}/.well-known/openid-configuration"
    try:
        with urlopen(discovery_url, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, URLError, ValueError) as exc:
        raise RuntimeError("Failed to load OIDC discovery document.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("OIDC discovery document was not a JSON object.")
    return cast(dict[str, object], payload)


def mcp_jwks_url() -> str:
    raw = os.environ.get("MCP_OIDC_JWKS_URL", "").strip()
    if raw:
        return raw
    discovered = _oidc_discovery_document().get("jwks_uri")
    if not isinstance(discovered, str) or not discovered.strip():
        raise RuntimeError("OIDC discovery document missing jwks_uri.")
    return discovered


def mcp_introspection_endpoint() -> str:
    raw = os.environ.get("MCP_OIDC_INTROSPECTION_ENDPOINT", "").strip()
    if raw:
        return raw
    discovered = _oidc_discovery_document().get("introspection_endpoint")
    if not isinstance(discovered, str) or not discovered.strip():
        raise RuntimeError("OIDC discovery document missing introspection_endpoint.")
    return discovered


def mcp_introspection_client_id() -> str | None:
    raw = os.environ.get("MCP_OIDC_INTROSPECTION_CLIENT_ID", "").strip()
    return raw or None


def mcp_introspection_client_secret() -> str | None:
    raw = os.environ.get("MCP_OIDC_INTROSPECTION_CLIENT_SECRET", "").strip()
    return raw or None


def mcp_protected_resource_metadata() -> dict[str, object]:
    return {
        "resource": mcp_resource_url(),
        "authorization_servers": [mcp_authorization_server_url()],
        "scopes_supported": list(mcp_supported_scopes()),
        "bearer_methods_supported": ["header"],
    }


def _bearer_challenge(*, error: str | None = None, description: str | None = None) -> str:
    attrs = [
        'realm="pandects-mcp"',
        f'resource_metadata="{mcp_resource_metadata_url()}"',
    ]
    if error:
        attrs.append(f'error="{error}"')
    if description:
        safe = description.replace('"', "'")
        attrs.append(f'error_description="{safe}"')
    return f"Bearer {', '.join(attrs)}"


def _authenticate_pandects_access_token(token: str) -> McpPrincipal | None:
    active_keys = AuthOAuthSigningKey.query.filter_by(active=True).all()
    for key in active_keys:
        try:
            payload = decode_access_token(
                token=token,
                public_key_pem=public_pem_from_private_pem(key.private_pem),
                audience=mcp_resource_url(),
            )
        except Exception:
            continue
        subject = payload.get("sub")
        if not isinstance(subject, str) or not subject.strip():
            raise McpAuthError(
                status_code=401,
                message="Bearer token missing subject.",
                www_authenticate=_bearer_challenge(
                    error="invalid_token",
                    description="The OAuth access token did not include a subject.",
                ),
            )
        scope_set = _scope_set(payload)
        user = db.session.get(AuthUser, subject.strip())
        if user is None or user.email_verified_at is None:
            raise McpAuthError(
                status_code=401,
                message="Linked Pandects account is not verified.",
                www_authenticate=_bearer_challenge(
                    error="invalid_token",
                    description="The token subject is not linked to a verified Pandects user.",
                ),
            )
        return McpPrincipal(
            access_context=AccessContext(tier="mcp", user_id=user.id),
            scopes=scope_set,
            issuer=mcp_oauth_issuer(),
            subject=subject.strip(),
            user_id=user.id,
        )
    return None


def _scope_set(payload: dict[str, object]) -> frozenset[str]:
    scopes: set[str] = set()

    from_scope = payload.get("scope")
    if isinstance(from_scope, str) and from_scope.strip():
        scopes.update(part for part in from_scope.split() if part)
    from_scp = payload.get("scp")
    if isinstance(from_scp, list):
        scopes.update(str(part).strip() for part in from_scp if str(part).strip())

    role_scope_map = dict(_DEFAULT_ZITADEL_ROLE_SCOPE_MAP)
    raw_mapping = os.environ.get("MCP_ZITADEL_ROLE_SCOPE_MAP", "").strip()
    if raw_mapping:
        for item in raw_mapping.split(","):
            role_key, sep, scope_name = item.partition("=")
            if not sep:
                continue
            normalized_role = role_key.strip()
            normalized_scope = scope_name.strip()
            if normalized_role and normalized_scope:
                role_scope_map[normalized_role] = normalized_scope

    def _add_role_claim_scopes(role_claim: object) -> None:
        if not isinstance(role_claim, dict):
            return
        for role_key in role_claim.keys():
            if not isinstance(role_key, str):
                continue
            normalized_role = role_key.strip()
            if not normalized_role:
                continue
            mapped_scope = role_scope_map.get(normalized_role)
            if mapped_scope:
                scopes.add(mapped_scope)
            elif normalized_role in _DEFAULT_MCP_SCOPES:
                scopes.add(normalized_role)

    for claim_key, claim_value in payload.items():
        if not isinstance(claim_key, str):
            continue
        normalized_key = claim_key.strip()
        if normalized_key in (
            "urn:zitadel:iam:org:project:roles",
            "urn:iam:org:project:roles",
        ) or (
            normalized_key.startswith("urn:zitadel:iam:org:project:")
            and normalized_key.endswith(":roles")
        ):
            _add_role_claim_scopes(claim_value)

    return frozenset(scopes)


def _audience_set(payload: dict[str, object]) -> frozenset[str]:
    audience_claim = payload.get("aud")
    if isinstance(audience_claim, str) and audience_claim.strip():
        return frozenset({audience_claim.strip()})
    if isinstance(audience_claim, list):
        return frozenset(
            str(part).strip() for part in audience_claim if str(part).strip()
        )
    return frozenset()


def _signing_key_from_token(token: str) -> object:
    try:
        from jwt import PyJWKClient
        from jwt.exceptions import PyJWKClientError
    except ImportError as exc:
        raise RuntimeError("PyJWT dependency is required for MCP auth.") from exc

    global _mcp_jwk_client
    client = _mcp_jwk_client
    if client is None:
        client = PyJWKClient(mcp_jwks_url())
        _mcp_jwk_client = client
    client_obj = cast(Any, client)
    try:
        return client_obj.get_signing_key_from_jwt(token).key
    except PyJWKClientError as exc:
        raise RuntimeError("Unable to load JWT signing key.") from exc


def _decode_access_token(token: str) -> dict[str, object]:
    try:
        import jwt
        from jwt.exceptions import InvalidTokenError
    except ImportError as exc:
        raise RuntimeError("PyJWT dependency is required for MCP auth.") from exc

    signing_key = cast(Any, _signing_key_from_token(token))
    audiences = mcp_oidc_audiences()
    audience: str | list[str]
    audience_values = list(audiences)
    if not audience_values:
        raise RuntimeError("MCP_OIDC_AUDIENCE is required for MCP auth.")
    audience = audience_values if len(audience_values) > 1 else audience_values[0]
    try:
        payload_obj = jwt.decode(
            token,
            signing_key,
            algorithms=list(mcp_jwt_algorithms()),
            audience=audience,
            issuer=mcp_oidc_issuer(),
            leeway=60,
        )
    except InvalidTokenError as exc:
        raise McpAuthError(
            status_code=401,
            message="Invalid bearer token.",
            www_authenticate=_bearer_challenge(
                error="invalid_token",
                description="The bearer access token is invalid or expired.",
            ),
        ) from exc
    if not isinstance(payload_obj, dict):
        raise McpAuthError(
            status_code=401,
            message="Invalid bearer token.",
            www_authenticate=_bearer_challenge(
                error="invalid_token",
                description="The bearer access token payload was invalid.",
            ),
        )
    return cast(dict[str, object], payload_obj)


def _introspect_access_token(token: str) -> dict[str, object]:
    client_id = mcp_introspection_client_id()
    client_secret = mcp_introspection_client_secret()
    if not client_id or not client_secret:
        raise RuntimeError(
            "MCP token introspection is not configured (missing client credentials)."
        )

    credentials = f"{client_id}:{client_secret}".encode("utf-8")
    auth_header = b64encode(credentials).decode("ascii")
    body = urlencode(
        {
            "token": token,
            "token_type_hint": "access_token",
        }
    ).encode("utf-8")
    req = Request(
        mcp_introspection_endpoint(),
        data=body,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {auth_header}",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=15) as response:
            payload_obj = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raw = exc.read().decode("utf-8", errors="replace")
        try:
            err_payload_obj = json.loads(raw)
        except ValueError:
            err_payload_obj = None
        if exc.code == 401:
            raise RuntimeError("MCP token introspection credentials were rejected.") from exc
        if isinstance(err_payload_obj, dict):
            err_code = err_payload_obj.get("error")
            err_description = err_payload_obj.get("error_description")
            if isinstance(err_code, str) and err_code == "invalid_client":
                raise RuntimeError(
                    "MCP token introspection credentials were rejected."
                ) from exc
            if isinstance(err_description, str) and err_description.strip():
                raise RuntimeError(
                    f"MCP token introspection failed: {err_description.strip()}"
                ) from exc
        raise RuntimeError("MCP token introspection failed.") from exc
    except (OSError, URLError, ValueError) as exc:
        raise RuntimeError("MCP token introspection failed.") from exc

    if not isinstance(payload_obj, dict):
        raise RuntimeError("MCP token introspection returned invalid data.")
    active = payload_obj.get("active")
    if active is not True:
        raise McpAuthError(
            status_code=401,
            message="Invalid bearer token.",
            www_authenticate=_bearer_challenge(
                error="invalid_token",
                description="The bearer access token is invalid or expired.",
            ),
        )
    return cast(dict[str, object], payload_obj)


def _normalize_external_identity(payload: dict[str, object]) -> ExternalIdentity:
    issuer = payload.get("iss")
    subject = payload.get("sub")
    if not isinstance(issuer, str) or not issuer.strip():
        raise McpAuthError(
            status_code=401,
            message="Bearer token missing issuer.",
            www_authenticate=_bearer_challenge(
                error="invalid_token",
                description="The bearer access token is missing an issuer claim.",
            ),
        )
    if not isinstance(subject, str) or not subject.strip():
        raise McpAuthError(
            status_code=401,
            message="Bearer token missing subject.",
            www_authenticate=_bearer_challenge(
                error="invalid_token",
                description="The bearer access token is missing a subject claim.",
            ),
        )
    return ExternalIdentity(
        issuer=issuer.rstrip("/"),
        subject=subject,
        scopes=_scope_set(payload),
        audiences=_audience_set(payload),
        claims=payload,
    )


class McpIdentityProvider:
    def authenticate_access_token(self, token: str) -> ExternalIdentity:
        raise NotImplementedError


class OidcMcpIdentityProvider(McpIdentityProvider):
    def authenticate_access_token(self, token: str) -> ExternalIdentity:
        payload = _decode_access_token(token)
        return _normalize_external_identity(payload)


class ZitadelMcpIdentityProvider(OidcMcpIdentityProvider):
    def authenticate_access_token(self, token: str) -> ExternalIdentity:
        try:
            return super().authenticate_access_token(token)
        except McpAuthError:
            raise
        except Exception:
            pass
        payload = _introspect_access_token(token)
        return _normalize_external_identity(payload)


class _ProviderRegistry(dict[str, type[McpIdentityProvider]]):
    def register(
        self, name: str, provider_cls: type[McpIdentityProvider]
    ) -> type[McpIdentityProvider]:
        self[name.strip().lower()] = provider_cls
        return provider_cls


_PROVIDER_REGISTRY = _ProviderRegistry()
_PROVIDER_REGISTRY.register("oidc", OidcMcpIdentityProvider)
_PROVIDER_REGISTRY.register("zitadel", ZitadelMcpIdentityProvider)


def register_mcp_identity_provider(
    name: str, provider_cls: type[McpIdentityProvider]
) -> type[McpIdentityProvider]:
    return _PROVIDER_REGISTRY.register(name, provider_cls)


def _provider_by_name(provider_name: str) -> McpIdentityProvider:
    provider_cls = _PROVIDER_REGISTRY.get(provider_name)
    if provider_cls is None:
        raise RuntimeError(f"Unsupported MCP identity provider: {provider_name}.")
    return provider_cls()


def _identity_provider() -> McpIdentityProvider:
    global _mcp_identity_provider
    provider = _mcp_identity_provider
    if provider is None:
        provider = _provider_by_name(mcp_identity_provider_name())
        _mcp_identity_provider = provider
    return provider


def authenticate_external_identity(
    *, access_token: str, provider_name: str | None = None
) -> ExternalIdentity:
    resolved_provider_name = resolve_mcp_identity_provider_name(provider_name)
    if resolved_provider_name == mcp_identity_provider_name():
        return _identity_provider().authenticate_access_token(access_token)
    return _provider_by_name(resolved_provider_name).authenticate_access_token(
        access_token
    )


def _linked_verified_user(*, issuer: str, subject: str) -> AuthUser:
    try:
        mapping = AuthExternalSubject.query.filter_by(issuer=issuer, subject=subject).first()
    except SQLAlchemyError as exc:
        raise RuntimeError("Auth backend is unavailable right now.") from exc
    if mapping is None:
        raise McpAuthError(
            status_code=401,
            message="No linked Pandects account for this token.",
            www_authenticate=_bearer_challenge(
                error="invalid_token",
                description="The token subject is not linked to a verified Pandects user.",
            ),
        )
    try:
        user = db.session.get(AuthUser, mapping.user_id)
    except SQLAlchemyError as exc:
        raise RuntimeError("Auth backend is unavailable right now.") from exc
    if user is None or user.email_verified_at is None:
        raise McpAuthError(
            status_code=401,
            message="Linked Pandects account is not verified.",
            www_authenticate=_bearer_challenge(
                error="invalid_token",
                description="The token subject is not linked to a verified Pandects user.",
            ),
        )
    return cast(AuthUser, user)


def authenticate_mcp_request() -> McpPrincipal:
    try:
        _ = mcp_authorization_server_url()
    except RuntimeError as exc:
        raise McpAuthError(
            status_code=503,
            message=str(exc),
            www_authenticate=_bearer_challenge(),
        ) from exc

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise McpAuthError(
            status_code=401,
            message="Bearer token required.",
            www_authenticate=_bearer_challenge(
                error="invalid_token",
                description="Send an OAuth access token in the Authorization header.",
            ),
        )

    token = auth_header.removeprefix("Bearer ").strip()
    if not token:
        raise McpAuthError(
            status_code=401,
            message="Bearer token required.",
            www_authenticate=_bearer_challenge(
                error="invalid_token",
                description="Send an OAuth access token in the Authorization header.",
            ),
        )

    try:
        pandects_principal = _authenticate_pandects_access_token(token)
        if pandects_principal is not None:
            return pandects_principal
        external_identity = authenticate_external_identity(access_token=token)
        user = _linked_verified_user(
            issuer=external_identity.issuer,
            subject=external_identity.subject,
        )
    except McpAuthError:
        raise
    except RuntimeError as exc:
        raise McpAuthError(
            status_code=503,
            message=str(exc),
            www_authenticate=_bearer_challenge(),
        ) from exc

    return McpPrincipal(
        access_context=AccessContext(tier="mcp", user_id=user.id),
        scopes=external_identity.scopes,
        issuer=external_identity.issuer,
        subject=external_identity.subject,
        user_id=user.id,
    )


__all__ = [
    "McpAuthError",
    "ExternalIdentity",
    "McpPrincipal",
    "McpIdentityProvider",
    "OidcMcpIdentityProvider",
    "ZitadelMcpIdentityProvider",
    "_mcp_jwk_client",
    "_mcp_identity_provider",
    "authenticate_external_identity",
    "mcp_identity_provider_name",
    "register_mcp_identity_provider",
    "resolve_mcp_identity_provider_name",
    "_normalize_external_identity",
    "authenticate_mcp_request",
    "mcp_protocol_version",
    "mcp_protected_resource_metadata",
    "mcp_server_name",
    "mcp_server_version",
]
