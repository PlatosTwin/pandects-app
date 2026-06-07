"""ZITADEL Management API transport.

Encapsulates the bearer-token lifecycle and the authenticated request helpers
used to call ZITADEL's v2/management APIs. Higher-level operations (user
creation, session checks, grants) live in routes/auth/__init__.py and call
into ``ZitadelApiClient`` for transport.
"""

from __future__ import annotations

import os
import secrets
import time

from flask import abort

from backend.auth.mcp_runtime import mcp_oidc_issuer
from backend.routes.auth.zitadel_config import (
    _zitadel_api_client_id,
    _zitadel_api_key_id,
    _zitadel_api_private_key,
    _zitadel_api_token,
    _zitadel_token_endpoint,
)
from backend.routes.deps import AuthDeps

# Module-level cache so tests can clear it between cases without reaching into
# the client instance owned by the blueprint closure.
_TOKEN_CACHE: dict[str, object] = {}


class ZitadelApiClient:
    """Thin wrapper around ``deps._oidc_fetch_json`` for ZITADEL APIs.

    Owns the access-token cache and prefixes every call with the ZITADEL
    issuer URL plus a fresh ``Authorization: Bearer`` header.
    """

    def __init__(self, deps: AuthDeps) -> None:
        self._deps = deps

    def access_token(self) -> str:
        explicit = _zitadel_api_token().strip() if os.environ.get("AUTH_ZITADEL_API_TOKEN", "").strip() else ""
        if explicit:
            return explicit

        client_id = _zitadel_api_client_id()
        key_id = _zitadel_api_key_id()
        private_key = _zitadel_api_private_key()
        if not client_id or not key_id or not private_key:
            abort(
                503,
                description=(
                    "ZITADEL API access is not configured. Set AUTH_ZITADEL_API_TOKEN or "
                    "AUTH_ZITADEL_API_CLIENT_ID, AUTH_ZITADEL_API_KEY_ID, and AUTH_ZITADEL_API_PRIVATE_KEY."
                ),
            )

        cached_token = _TOKEN_CACHE.get("token")
        cached_expires_at = _TOKEN_CACHE.get("expires_at")
        now = int(time.time())
        if isinstance(cached_token, str) and isinstance(cached_expires_at, int) and cached_expires_at - 30 > now:
            return cached_token

        try:
            import jwt
        except ImportError:
            abort(503, description="JWT support is unavailable for ZITADEL API authentication.")

        assertion = jwt.encode(
            {
                "iss": client_id,
                "sub": client_id,
                "aud": _zitadel_token_endpoint(),
                "iat": now,
                "exp": now + 300,
                "jti": secrets.token_urlsafe(24),
            },
            private_key,
            algorithm="RS256",
            headers={"kid": key_id},
        )
        token_payload = self._deps._oidc_fetch_json(
            _zitadel_token_endpoint(),
            data={
                "grant_type": "client_credentials",
                "scope": "urn:zitadel:iam:org:project:id:zitadel:aud",
                "client_assertion_type": "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
                "client_assertion": assertion,
            },
        )
        access_token = token_payload.get("access_token")
        expires_in = token_payload.get("expires_in")
        if not isinstance(access_token, str) or not access_token.strip():
            abort(502, description="ZITADEL did not return an API access token.")
        expires_at = now + (int(expires_in) if isinstance(expires_in, int) else 300)
        _TOKEN_CACHE["token"] = access_token.strip()
        _TOKEN_CACHE["expires_at"] = expires_at
        return access_token.strip()

    def request(
        self,
        *,
        path: str,
        method: str,
        json_body: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return self._deps._oidc_fetch_json(
            f"{mcp_oidc_issuer()}{path}",
            json_body=json_body,
            headers={"Authorization": f"Bearer {self.access_token()}"},
            method=method,
        )

    def json(
        self,
        *,
        path: str,
        json_body: dict[str, object],
    ) -> dict[str, object]:
        return self.request(path=path, method="POST", json_body=json_body)
