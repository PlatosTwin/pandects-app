from __future__ import annotations

import base64
import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.rsa import RSAPrivateKey, RSAPublicKey

from backend.auth.runtime import public_api_base_url


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _b64url_uint(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def mcp_oauth_issuer() -> str:
    raw = os.environ.get("MCP_OAUTH_ISSUER", "").strip().rstrip("/")
    if raw:
        return raw
    return f"{public_api_base_url()}/v1/auth/oauth"


def mcp_oauth_authorization_endpoint() -> str:
    return f"{mcp_oauth_issuer()}/authorize"


def mcp_oauth_token_endpoint() -> str:
    return f"{mcp_oauth_issuer()}/token"


def mcp_oauth_registration_endpoint() -> str:
    return f"{mcp_oauth_issuer()}/register"


def mcp_oauth_jwks_uri() -> str:
    return f"{mcp_oauth_issuer()}/jwks.json"


def mcp_oauth_openid_configuration_url() -> str:
    return f"{mcp_oauth_issuer()}/.well-known/openid-configuration"


def mcp_oauth_authorization_server_metadata_url() -> str:
    return f"{mcp_oauth_issuer()}/.well-known/oauth-authorization-server"


def mcp_oauth_access_token_ttl_seconds() -> int:
    raw = os.environ.get("MCP_OAUTH_ACCESS_TOKEN_TTL_SECONDS", "").strip()
    try:
        parsed = int(raw)
    except ValueError:
        parsed = 600
    return max(60, parsed)


def mcp_oauth_authorization_code_ttl_seconds() -> int:
    raw = os.environ.get("MCP_OAUTH_AUTHORIZATION_CODE_TTL_SECONDS", "").strip()
    try:
        parsed = int(raw)
    except ValueError:
        parsed = 180
    return max(60, parsed)


def mcp_oauth_metadata() -> dict[str, Any]:
    issuer = mcp_oauth_issuer()
    return {
        "issuer": issuer,
        "authorization_endpoint": mcp_oauth_authorization_endpoint(),
        "token_endpoint": mcp_oauth_token_endpoint(),
        "registration_endpoint": mcp_oauth_registration_endpoint(),
        "jwks_uri": mcp_oauth_jwks_uri(),
        "subject_types_supported": ["public"],
        "id_token_signing_alg_values_supported": ["RS256"],
        "response_types_supported": ["code"],
        "grant_types_supported": ["authorization_code"],
        "token_endpoint_auth_methods_supported": ["none"],
        "code_challenge_methods_supported": ["S256"],
        "scopes_supported": [
            "sections:search",
            "agreements:search",
            "agreements:read",
            "agreements:read_fulltext",
        ],
    }


def generate_signing_keypair() -> tuple[str, str]:
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    private_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")
    return secrets.token_urlsafe(18), private_pem


def public_jwk_from_private_pem(*, kid: str, private_pem: str) -> dict[str, str]:
    private_key = cast(
        RSAPrivateKey,
        serialization.load_pem_private_key(private_pem.encode("utf-8"), password=None),
    )
    public_key: RSAPublicKey = private_key.public_key()
    public_numbers = public_key.public_numbers()
    return {
        "kty": "RSA",
        "kid": kid,
        "use": "sig",
        "alg": "RS256",
        "n": _b64url_uint(public_numbers.n),
        "e": _b64url_uint(public_numbers.e),
    }


def encode_access_token(*, private_pem: str, kid: str, claims: dict[str, Any]) -> str:
    return jwt.encode(
        claims,
        private_pem,
        algorithm="RS256",
        headers={"kid": kid, "typ": "JWT"},
    )


def decode_access_token(*, token: str, public_key_pem: str, audience: str) -> dict[str, Any]:
    payload = jwt.decode(
        token,
        public_key_pem,
        algorithms=["RS256"],
        issuer=mcp_oauth_issuer(),
        audience=audience,
        leeway=30,
    )
    if not isinstance(payload, dict):
        raise jwt.InvalidTokenError("JWT payload was not an object.")
    return payload


def public_pem_from_private_pem(private_pem: str) -> str:
    private_key = cast(
        RSAPrivateKey,
        serialization.load_pem_private_key(private_pem.encode("utf-8"), password=None),
    )
    public_key: RSAPublicKey = private_key.public_key()
    return public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("utf-8")


def access_token_claims(
    *,
    subject: str,
    audience: str,
    scope: str,
    token_id: str,
) -> dict[str, Any]:
    now = _utc_now()
    exp = now + timedelta(seconds=mcp_oauth_access_token_ttl_seconds())
    return {
        "iss": mcp_oauth_issuer(),
        "sub": subject,
        "aud": audience,
        "scope": scope,
        "iat": int(now.timestamp()),
        "nbf": int(now.timestamp()),
        "exp": int(exp.timestamp()),
        "jti": token_id,
    }


__all__ = [
    "access_token_claims",
    "decode_access_token",
    "encode_access_token",
    "generate_signing_keypair",
    "mcp_oauth_access_token_ttl_seconds",
    "mcp_oauth_authorization_code_ttl_seconds",
    "mcp_oauth_authorization_endpoint",
    "mcp_oauth_authorization_server_metadata_url",
    "mcp_oauth_issuer",
    "mcp_oauth_jwks_uri",
    "mcp_oauth_metadata",
    "mcp_oauth_openid_configuration_url",
    "mcp_oauth_registration_endpoint",
    "mcp_oauth_token_endpoint",
    "public_jwk_from_private_pem",
    "public_pem_from_private_pem",
]
