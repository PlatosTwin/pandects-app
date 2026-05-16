"""Environment-driven configuration getters for ZITADEL/OAuth flows.

Reading is centralized here so the route handlers stay focused on flow logic.
Each helper either returns a value or aborts the request with 503 when the
required environment variable is missing.
"""

from __future__ import annotations

import os
from base64 import urlsafe_b64encode
from hashlib import sha256

from flask import abort

from backend.auth.mcp_runtime import (
    mcp_jwks_url,
    mcp_jwt_algorithms,
    mcp_oidc_audiences,
    mcp_oidc_issuer,
    mcp_supported_scopes,
)
from backend.routes.deps import AuthDeps


def _zitadel_client_id() -> str:
    client_id = os.environ.get("MCP_ZITADEL_CLIENT_ID", "").strip()
    if not client_id:
        abort(503, description="ZITADEL linking is not configured (missing MCP_ZITADEL_CLIENT_ID).")
    return client_id


def _zitadel_redirect_uri(deps: AuthDeps) -> str:
    explicit = os.environ.get("MCP_ZITADEL_REDIRECT_URI", "").strip()
    if explicit:
        return explicit
    return f"{deps._frontend_base_url()}/auth/zitadel/callback"


def _website_zitadel_redirect_uri(deps: AuthDeps) -> str:
    explicit = os.environ.get("AUTH_ZITADEL_REDIRECT_URI", "").strip()
    if explicit:
        return explicit
    return f"{deps._frontend_base_url()}/auth/zitadel/callback"


def _zitadel_scopes() -> str:
    raw = os.environ.get("MCP_ZITADEL_SCOPES", "").strip()
    if raw:
        return " ".join(part for part in raw.split() if part)
    scopes: list[str] = [
        "openid",
        "profile",
        "email",
        *mcp_supported_scopes(),
        "urn:iam:org:project:roles",
        "urn:zitadel:iam:org:project:roles",
    ]
    project_id = _zitadel_project_id(optional=True)
    if project_id:
        scopes.append(f"urn:zitadel:iam:org:project:{project_id}:roles")
    return " ".join(scopes)


def _zitadel_audience() -> str | None:
    raw = os.environ.get("MCP_ZITADEL_AUDIENCE", "").strip()
    if raw:
        return raw
    audiences = mcp_oidc_audiences()
    return audiences[0] if audiences else None


def _zitadel_resource() -> str | None:
    raw = os.environ.get("MCP_ZITADEL_RESOURCE", "").strip()
    if raw:
        return raw
    return _zitadel_audience()


def _zitadel_authorization_endpoint() -> str:
    raw = os.environ.get("MCP_OIDC_AUTHORIZATION_ENDPOINT", "").strip()
    if raw:
        return raw
    return f"{mcp_oidc_issuer()}/oauth/v2/authorize"


def _zitadel_token_endpoint() -> str:
    raw = os.environ.get("MCP_OIDC_TOKEN_ENDPOINT", "").strip()
    if raw:
        return raw
    return f"{mcp_oidc_issuer()}/oauth/v2/token"


def _zitadel_api_token() -> str:
    raw = os.environ.get("AUTH_ZITADEL_API_TOKEN", "").strip()
    if not raw:
        abort(503, description="ZITADEL API access is not configured.")
    return raw


def _zitadel_api_client_id() -> str | None:
    raw = os.environ.get("AUTH_ZITADEL_API_CLIENT_ID", "").strip()
    return raw or None


def _zitadel_api_key_id() -> str | None:
    raw = os.environ.get("AUTH_ZITADEL_API_KEY_ID", "").strip()
    return raw or None


def _zitadel_api_private_key() -> str | None:
    raw = os.environ.get("AUTH_ZITADEL_API_PRIVATE_KEY", "").strip()
    if not raw:
        return None
    return raw.replace("\\n", "\n")


def _zitadel_google_idp_id() -> str:
    raw = os.environ.get("AUTH_ZITADEL_GOOGLE_IDP_ID", "").strip()
    if raw:
        return raw
    raw = os.environ.get("AUTH_ZITADEL_GOOGLE_IDP_HINT", "").strip()
    if raw:
        return raw
    abort(503, description="ZITADEL Google auth is not configured (missing AUTH_ZITADEL_GOOGLE_IDP_ID).")


def _zitadel_default_role_keys() -> tuple[str, ...]:
    raw = os.environ.get("AUTH_ZITADEL_DEFAULT_ROLE_KEYS", "").strip()
    if raw:
        return tuple(part.strip() for part in raw.split(",") if part.strip())
    return (
        "sections_search",
        "agreements_search",
        "agreements_read",
        "agreements_read_fulltext",
    )


def _zitadel_project_id(*, optional: bool = False) -> str | None:
    explicit = os.environ.get("AUTH_ZITADEL_PROJECT_ID", "").strip()
    if explicit:
        return explicit
    if optional:
        return None
    abort(
        503,
        description=(
            "ZITADEL project configuration is incomplete. "
            "Set AUTH_ZITADEL_PROJECT_ID."
        ),
    )


def _decode_zitadel_id_token(id_token: str) -> dict[str, object] | None:
    try:
        import jwt
        from jwt import PyJWKClient
        from jwt.exceptions import InvalidTokenError, PyJWKClientError
    except ImportError:
        return None

    try:
        jwk_client = PyJWKClient(mcp_jwks_url())
        signing_key = jwk_client.get_signing_key_from_jwt(id_token).key
        payload_obj = jwt.decode(
            id_token,
            signing_key,
            algorithms=list(mcp_jwt_algorithms()),
            audience=_zitadel_client_id(),
            issuer=mcp_oidc_issuer(),
            leeway=60,
        )
    except (InvalidTokenError, PyJWKClientError, RuntimeError):
        return None
    if not isinstance(payload_obj, dict):
        return None
    return payload_obj


def _build_pkce_challenge(code_verifier: str) -> str:
    digest = sha256(code_verifier.encode("utf-8")).digest()
    return urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
