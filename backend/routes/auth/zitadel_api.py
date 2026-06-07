"""ZITADEL Management API client.

Encapsulates the bearer-token lifecycle and the higher-level user/session
operations that talk to ZITADEL's v2/management APIs. Code in
routes/auth/__init__.py owns the DB-coupled flows and calls into this
client for anything that touches the remote API.
"""

from __future__ import annotations

import os
import secrets
import time
import uuid

from flask import abort
from werkzeug.exceptions import HTTPException

from backend.auth.mcp_runtime import ExternalIdentity, mcp_oidc_issuer
from backend.routes.auth.zitadel_config import (
    _zitadel_api_client_id,
    _zitadel_api_key_id,
    _zitadel_api_private_key,
    _zitadel_api_token,
    _zitadel_default_role_keys,
    _zitadel_project_id,
    _zitadel_token_endpoint,
)
from backend.routes.deps import AuthDeps

# Module-level cache so tests can clear it between cases without reaching into
# the client instance owned by the blueprint closure.
_TOKEN_CACHE: dict[str, object] = {}


class ZitadelApiClient:
    """Thin wrapper around ``deps._oidc_fetch_json`` for ZITADEL APIs.

    Owns the access-token cache and prefixes every call with the ZITADEL
    issuer URL plus a fresh ``Authorization: Bearer`` header. Also exposes
    user/session/grant helpers that compose multiple transport calls.
    """

    def __init__(self, deps: AuthDeps) -> None:
        self._deps = deps

    # ---- Transport ---------------------------------------------------------

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

    # ---- Identity construction --------------------------------------------

    @staticmethod
    def build_external_identity(
        *,
        subject: str,
        email: str,
        email_verified: bool = True,
        display_name: str | None = None,
        given_name: str | None = None,
        family_name: str | None = None,
        picture: str | None = None,
    ) -> ExternalIdentity:
        claims: dict[str, object] = {
            "email": email,
            "email_verified": email_verified,
        }
        if isinstance(display_name, str) and display_name.strip():
            claims["name"] = display_name.strip()
        if isinstance(given_name, str) and given_name.strip():
            claims["given_name"] = given_name.strip()
        if isinstance(family_name, str) and family_name.strip():
            claims["family_name"] = family_name.strip()
        if isinstance(picture, str) and picture.strip():
            claims["picture"] = picture.strip()
        return ExternalIdentity(
            issuer=mcp_oidc_issuer(),
            subject=subject,
            scopes=frozenset(),
            audiences=frozenset(),
            claims=claims,
        )

    @staticmethod
    def human_payload(
        *,
        email: str,
        password: str,
        first_name: str | None = None,
        last_name: str | None = None,
        email_is_verified: bool,
        verify_url_template: str | None = None,
    ) -> dict[str, object]:
        display_name = " ".join(
            part.strip() for part in (first_name, last_name) if isinstance(part, str) and part.strip()
        )
        if not display_name:
            display_name = email
        profile: dict[str, object] = {"displayName": display_name}
        if isinstance(first_name, str) and first_name.strip():
            profile["givenName"] = first_name.strip()
        if isinstance(last_name, str) and last_name.strip():
            profile["familyName"] = last_name.strip()
        email_payload: dict[str, object] = {
            "email": email,
        }
        if isinstance(verify_url_template, str) and verify_url_template.strip():
            email_payload["sendCode"] = {"urlTemplate": verify_url_template.strip()}
        else:
            email_payload["isVerified"] = email_is_verified
        return {
            "userId": str(uuid.uuid4()),
            "username": email,
            "profile": profile,
            "email": email_payload,
            "password": {
                "password": password,
                "changeRequired": False,
            },
        }

    # ---- Grants / users / sessions ----------------------------------------

    def ensure_default_user_grant(self, *, user_id: str) -> None:
        role_keys = list(_zitadel_default_role_keys())
        if not role_keys:
            return
        project_id = _zitadel_project_id(optional=True)
        if not project_id:
            return
        payload = self.request(
            path="/management/v1/users/grants/_search",
            method="POST",
            json_body={
                "queries": [
                    {"user_id_query": {"user_id": user_id}},
                    {"project_id_query": {"project_id": project_id}},
                ]
            },
        )
        result = payload.get("result")
        existing_grant = result[0] if isinstance(result, list) and result else None
        existing_role_keys: set[str] = set()
        grant_id: str | None = None
        if isinstance(existing_grant, dict):
            raw_roles = existing_grant.get("roleKeys")
            if isinstance(raw_roles, list):
                existing_role_keys = {
                    str(role_key).strip()
                    for role_key in raw_roles
                    if str(role_key).strip()
                }
            raw_grant_id = existing_grant.get("id") or existing_grant.get("grantId")
            if isinstance(raw_grant_id, str) and raw_grant_id.strip():
                grant_id = raw_grant_id.strip()

        merged_role_keys = sorted(existing_role_keys.union(role_keys))
        if merged_role_keys == sorted(existing_role_keys):
            return

        if grant_id:
            self.request(
                path=f"/management/v1/users/{user_id}/grants/{grant_id}",
                method="PUT",
                json_body={
                    "projectId": project_id,
                    "roleKeys": merged_role_keys,
                },
            )
            return

        self.request(
            path=f"/management/v1/users/{user_id}/grants",
            method="POST",
            json_body={
                "projectId": project_id,
                "roleKeys": merged_role_keys,
            },
        )

    def create_human_user(
        self,
        *,
        email: str,
        password: str,
        first_name: str | None = None,
        last_name: str | None = None,
        email_is_verified: bool,
        verify_url_template: str | None = None,
    ) -> ExternalIdentity:
        created = self.json(
            path="/v2/users/human",
            json_body=self.human_payload(
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name,
                email_is_verified=email_is_verified,
                verify_url_template=verify_url_template,
            ),
        )
        user_id = created.get("userId") or created.get("user_id")
        if not isinstance(user_id, str) or not user_id.strip():
            abort(502, description="ZITADEL did not return a user identifier.")
        self.ensure_default_user_grant(user_id=user_id.strip())
        return self.build_external_identity(
            subject=user_id.strip(),
            email=email,
            email_verified=email_is_verified,
            display_name=" ".join(
                part.strip()
                for part in (first_name, last_name)
                if isinstance(part, str) and part.strip()
            )
            or email,
            given_name=first_name,
            family_name=last_name,
        )

    def session_identity(self, *, email: str, password: str) -> ExternalIdentity:
        created = self.json(
            path="/v2/sessions",
            json_body={"checks": {"user": {"loginName": email}}},
        )
        session_id = created.get("sessionId") or created.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            abort(502, description="ZITADEL did not return a session identifier.")
        self.request(
            path=f"/v2/sessions/{session_id.strip()}",
            method="PATCH",
            json_body={"checks": {"password": {"password": password}}},
        )
        session_payload = self.request(
            path=f"/v2/sessions/{session_id.strip()}",
            method="GET",
        )
        session = session_payload.get("session")
        if not isinstance(session, dict):
            abort(502, description="ZITADEL did not return session details.")
        factors = session.get("factors")
        if not isinstance(factors, dict):
            abort(502, description="ZITADEL did not return verified session factors.")
        user_factor = factors.get("user")
        if not isinstance(user_factor, dict):
            abort(401, description="Invalid email or password.")
        password_factor = factors.get("password")
        if not isinstance(password_factor, dict):
            abort(401, description="Invalid email or password.")
        user_id = user_factor.get("id")
        if not isinstance(user_id, str) or not user_id.strip():
            abort(502, description="ZITADEL did not return a user identifier.")
        try:
            return self.user_identity(user_id=user_id.strip())
        except HTTPException as exc:
            if exc.code == 400 and exc.description == "Auth provider did not return a verified email address.":
                abort(400, description="Verify your email before signing in.")
            raise

    def user_identity(self, *, user_id: str) -> ExternalIdentity:
        payload = self.request(
            path=f"/v2/users/{user_id}",
            method="GET",
        )
        user_payload = payload.get("user")
        if not isinstance(user_payload, dict):
            abort(502, description="ZITADEL did not return the user record.")
        human = user_payload.get("human")
        if not isinstance(human, dict):
            abort(502, description="ZITADEL did not return a human user record.")
        email_payload = human.get("email")
        if not isinstance(email_payload, dict):
            abort(502, description="ZITADEL did not return the user email.")
        email = email_payload.get("email")
        is_verified = email_payload.get("isVerified")
        if not isinstance(email, str) or not email.strip() or is_verified is not True:
            abort(400, description="Auth provider did not return a verified email address.")
        profile = human.get("profile")
        profile_dict = profile if isinstance(profile, dict) else {}
        display_name = profile_dict.get("displayName")
        given_name = profile_dict.get("givenName")
        family_name = profile_dict.get("familyName")
        avatar_url = profile_dict.get("avatarUrl")
        return self.build_external_identity(
            subject=user_id.strip(),
            email=email.strip(),
            display_name=display_name if isinstance(display_name, str) else email.strip(),
            given_name=given_name if isinstance(given_name, str) else None,
            family_name=family_name if isinstance(family_name, str) else None,
            picture=avatar_url if isinstance(avatar_url, str) else None,
        )

    def user_email_matches(self, *, user_id: str, email: str) -> bool:
        payload = self.request(
            path=f"/v2/users/{user_id}",
            method="GET",
        )
        user_payload = payload.get("user")
        if not isinstance(user_payload, dict):
            return False
        human = user_payload.get("human")
        if not isinstance(human, dict):
            return False
        email_payload = human.get("email")
        if not isinstance(email_payload, dict):
            return False
        remote_email = email_payload.get("email")
        if not isinstance(remote_email, str) or not remote_email.strip():
            return False
        return self._deps._normalize_email(remote_email) == email

    def send_email_verification(self, *, user_id: str, url_template: str) -> None:
        self.json(
            path=f"/v2/users/{user_id}/email/send",
            json_body={"sendCode": {"urlTemplate": url_template}},
        )

    def search_user_id_by_login_name(self, *, login_name: str) -> str | None:
        payload = self.request(
            path="/v2/users",
            method="POST",
            json_body={
                "pagination": {"limit": 1},
                "queries": [
                    {
                        "loginNameQuery": {
                            "loginName": login_name,
                            "method": "TEXT_QUERY_METHOD_EQUALS",
                        }
                    }
                ],
            },
        )
        result = payload.get("result")
        if not isinstance(result, list) or not result:
            return None
        first = result[0]
        if not isinstance(first, dict):
            return None
        user_id = first.get("userId")
        if not isinstance(user_id, str) or not user_id.strip():
            return None
        return user_id.strip()
