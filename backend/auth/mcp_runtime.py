from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, cast
from urllib.error import URLError
from urllib.request import urlopen

from flask import request
from sqlalchemy.exc import SQLAlchemyError

from backend.auth.session_runtime import AccessContext
from backend.extensions import db
from backend.models import AuthExternalSubject, AuthUser


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
    raw = os.environ.get("MCP_OIDC_AUTHORIZATION_SERVER_URL", "").strip()
    if raw:
        return raw.rstrip("/")
    return mcp_oidc_issuer()


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


def _scope_set(payload: dict[str, object]) -> frozenset[str]:
    from_scope = payload.get("scope")
    if isinstance(from_scope, str) and from_scope.strip():
        return frozenset(part for part in from_scope.split() if part)
    from_scp = payload.get("scp")
    if isinstance(from_scp, list):
        return frozenset(str(part).strip() for part in from_scp if str(part).strip())
    return frozenset()


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

    signing_key = _signing_key_from_token(token)
    audiences = mcp_oidc_audiences()
    audience: str | list[str]
    audience = list(audiences) if len(audiences) > 1 else audiences[0]
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


class _ProviderRegistry(dict[str, type[McpIdentityProvider]]):
    def register(
        self, name: str, provider_cls: type[McpIdentityProvider]
    ) -> type[McpIdentityProvider]:
        self[name.strip().lower()] = provider_cls
        return provider_cls


_PROVIDER_REGISTRY = _ProviderRegistry()
_PROVIDER_REGISTRY.register("oidc", OidcMcpIdentityProvider)


def register_mcp_identity_provider(
    name: str, provider_cls: type[McpIdentityProvider]
) -> type[McpIdentityProvider]:
    return _PROVIDER_REGISTRY.register(name, provider_cls)


def _identity_provider() -> McpIdentityProvider:
    global _mcp_identity_provider
    provider = _mcp_identity_provider
    if provider is None:
        provider_name = mcp_identity_provider_name()
        provider_cls = _PROVIDER_REGISTRY.get(provider_name)
        if provider_cls is None:
            raise RuntimeError(
                f"Unsupported MCP identity provider: {provider_name}."
            )
        provider = provider_cls()
        _mcp_identity_provider = provider
    return provider


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
        _ = mcp_oidc_issuer()
        _ = mcp_oidc_audiences()
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
        external_identity = _identity_provider().authenticate_access_token(token)
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
    "_mcp_jwk_client",
    "_mcp_identity_provider",
    "mcp_identity_provider_name",
    "register_mcp_identity_provider",
    "_normalize_external_identity",
    "authenticate_mcp_request",
    "mcp_protocol_version",
    "mcp_protected_resource_metadata",
    "mcp_server_name",
    "mcp_server_version",
]
