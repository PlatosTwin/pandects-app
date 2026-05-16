"""Flask blueprint for auth: register, login, password reset, API keys, Google OAuth."""

from __future__ import annotations

import json
import os
import secrets
import time
import uuid
from html import escape
from base64 import urlsafe_b64encode
from hashlib import sha256
from datetime import datetime, timedelta
from collections import defaultdict
from typing import cast
from urllib.parse import urlencode, urlparse, urlunparse

from flask import Blueprint, Flask, abort, jsonify, make_response, redirect, request, current_app
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.exceptions import HTTPException
from werkzeug.security import check_password_hash

from backend.auth.mcp_oauth_runtime import (
    access_token_claims,
    encode_access_token,
    generate_signing_keypair,
    mcp_oauth_access_token_ttl_seconds,
    mcp_oauth_authorization_code_ttl_seconds,
    mcp_oauth_authorization_endpoint,
    mcp_oauth_authorization_server_metadata_url,
    mcp_oauth_issuer,
    mcp_oauth_jwks_uri,
    mcp_oauth_metadata,
    mcp_oauth_openid_configuration_url,
    mcp_oauth_refresh_token_ttl_seconds,
    mcp_oauth_registration_endpoint,
    mcp_oauth_token_endpoint,
    public_jwk_from_private_pem,
)
from backend.auth.email_runtime import send_pandects_auth_email, verify_zitadel_signature
from backend.auth.mcp_runtime import (
    ExternalIdentity,
    McpAuthError,
    mcp_oidc_issuer,
    mcp_resource_url,
    mcp_supported_scopes,
)
from backend.routes.auth.cookies import (
    _clear_oauth_authorize_cookie,
    _clear_oauth_browser_cookie,
    _clear_zitadel_link_cookie,
    _clear_zitadel_pending_cookie,
    _clear_zitadel_web_cookie,
    _load_oauth_authorize_cookie,
    _load_oauth_browser_cookie,
    _load_zitadel_link_cookie,
    _load_zitadel_pending_cookie,
    _load_zitadel_web_cookie,
    _set_oauth_authorize_cookie,
    _set_oauth_browser_cookie,
    _set_zitadel_link_cookie,
    _set_zitadel_pending_cookie,
    _set_zitadel_web_cookie,
)
from backend.routes.auth.zitadel_config import (
    _build_pkce_challenge,
    _decode_zitadel_id_token,
    _website_zitadel_redirect_uri,
    _zitadel_api_client_id,
    _zitadel_api_key_id,
    _zitadel_api_private_key,
    _zitadel_api_token,
    _zitadel_audience,
    _zitadel_authorization_endpoint,
    _zitadel_client_id,
    _zitadel_default_role_keys,
    _zitadel_google_idp_id,
    _zitadel_project_id,
    _zitadel_redirect_uri,
    _zitadel_resource,
    _zitadel_scopes,
    _zitadel_token_endpoint,
)
from backend.routes.deps import AuthDeps

_ZITADEL_API_TOKEN_CACHE: dict[str, object] = {}
_OAUTH_LOOPBACK_REDIRECT_HOSTS = {"localhost", "127.0.0.1", "::1"}


def register_auth_routes(app: Flask, *, deps: AuthDeps) -> Blueprint:
    auth_blp = Blueprint("auth", "auth", url_prefix="/v1/auth")

    def _notification_dict(obj: object, *keys: str) -> dict[str, object] | None:
        if not isinstance(obj, dict):
            return None
        for key in keys:
            value = obj.get(key)
            if isinstance(value, dict):
                return value
        return None

    def _notification_string(obj: object, *keys: str) -> str | None:
        if not isinstance(obj, dict):
            return None
        for key in keys:
            value = obj.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    def _parse_auth_notification(payload: dict[str, object]) -> tuple[str, str, str, str | None] | None:
        template_data = _notification_dict(payload, "templateData", "template_data")
        context_info = _notification_dict(payload, "contextInfo", "context_info")
        args = _notification_dict(payload, "args")

        action_url = _notification_string(template_data, "url", "link")
        recipient = _notification_string(
            payload,
            "recipientEmail",
            "recipient_email",
            "recipient",
        ) or _notification_string(context_info, "recipientEmailAddress", "recipient_email_address")
        code = _notification_string(args, "code", "Code")

        if not isinstance(action_url, str) or not isinstance(recipient, str):
            return None

        if "/verify-email" in action_url:
            return ("verify-email", recipient, action_url, code)
        if "/reset-password/confirm" in action_url:
            return ("reset-password", recipient, action_url, code)
        return None

    def _external_link_payload(*, link, provider_name: str) -> dict[str, object]:
        return {
            "id": link.id,
            "provider": provider_name,
            "issuer": link.issuer,
            "subject": link.subject,
            "created_at": link.created_at.isoformat(),
        }

    def _link_external_identity_for_user(*, user, provider_name: str, external_identity):
        existing = deps.AuthExternalSubject.query.filter_by(
            issuer=external_identity.issuer,
            subject=external_identity.subject,
        ).first()
        if existing is not None:
            if existing.user_id != user.id:
                abort(409, description="External identity is already linked to another account.")
            return ("already_linked", existing, 200)

        link = deps.AuthExternalSubject(
            user_id=user.id,
            issuer=external_identity.issuer,
            subject=external_identity.subject,
        )
        deps.db.session.add(link)
        deps._record_signon_event(user_id=user.id, provider=provider_name, action="link")
        deps.db.session.commit()
        return ("linked", link, 201)

    def _website_auth_error_payload(message: str, *, code: int = 400):
        resp = make_response(jsonify({"message": message, "error": "auth_failed"}), code)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    def _zitadel_api_access_token() -> str:
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

        cached_token = _ZITADEL_API_TOKEN_CACHE.get("token")
        cached_expires_at = _ZITADEL_API_TOKEN_CACHE.get("expires_at")
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
        token_payload = deps._oidc_fetch_json(
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
        _ZITADEL_API_TOKEN_CACHE["token"] = access_token.strip()
        _ZITADEL_API_TOKEN_CACHE["expires_at"] = expires_at
        return access_token.strip()

    def _zitadel_api_json(
        *,
        path: str,
        json_body: dict[str, object],
    ) -> dict[str, object]:
        return _zitadel_api_request(path=path, method="POST", json_body=json_body)

    def _zitadel_api_request(
        *,
        path: str,
        method: str,
        json_body: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return deps._oidc_fetch_json(
            f"{mcp_oidc_issuer()}{path}",
            json_body=json_body,
            headers={"Authorization": f"Bearer {_zitadel_api_access_token()}"},
            method=method,
        )

    def _ensure_zitadel_default_user_grant(*, user_id: str) -> None:
        role_keys = list(_zitadel_default_role_keys())
        if not role_keys:
            return
        project_id = _zitadel_project_id(optional=True)
        if not project_id:
            return
        payload = _zitadel_api_request(
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
            _zitadel_api_request(
                path=f"/management/v1/users/{user_id}/grants/{grant_id}",
                method="PUT",
                json_body={
                    "projectId": project_id,
                    "roleKeys": merged_role_keys,
                },
            )
            return

        _zitadel_api_request(
            path=f"/management/v1/users/{user_id}/grants",
            method="POST",
            json_body={
                "projectId": project_id,
                "roleKeys": merged_role_keys,
            },
        )

    def _build_external_identity(
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

    def _zitadel_human_payload(
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

    def _create_zitadel_human_user(
        *,
        email: str,
        password: str,
        first_name: str | None = None,
        last_name: str | None = None,
        email_is_verified: bool,
        verify_url_template: str | None = None,
    ):
        created = _zitadel_api_json(
            path="/v2/users/human",
            json_body=_zitadel_human_payload(
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
        _ensure_zitadel_default_user_grant(user_id=user_id.strip())
        return _build_external_identity(
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

    def _local_user_by_email(*, email: str):
        return deps.AuthUser.query.filter_by(email=email).first()

    def _zitadel_session_identity(*, email: str, password: str):
        created = _zitadel_api_json(
            path="/v2/sessions",
            json_body={"checks": {"user": {"loginName": email}}},
        )
        session_id = created.get("sessionId") or created.get("session_id")
        if not isinstance(session_id, str) or not session_id.strip():
            abort(502, description="ZITADEL did not return a session identifier.")
        _zitadel_api_request(
            path=f"/v2/sessions/{session_id.strip()}",
            method="PATCH",
            json_body={"checks": {"password": {"password": password}}},
        )
        session_payload = _zitadel_api_request(
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
            return _zitadel_user_identity(user_id=user_id.strip())
        except HTTPException as exc:
            if exc.code == 400 and exc.description == "Auth provider did not return a verified email address.":
                abort(400, description="Verify your email before signing in.")
            raise

    def _zitadel_user_identity(*, user_id: str):
        payload = _zitadel_api_request(
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
        return _build_external_identity(
            subject=user_id.strip(),
            email=email.strip(),
            display_name=display_name if isinstance(display_name, str) else email.strip(),
            given_name=given_name if isinstance(given_name, str) else None,
            family_name=family_name if isinstance(family_name, str) else None,
            picture=avatar_url if isinstance(avatar_url, str) else None,
        )

    def _zitadel_user_email_matches(*, user_id: str, email: str) -> bool:
        payload = _zitadel_api_request(
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
        return deps._normalize_email(remote_email) == email

    def _ensure_zitadel_user_for_local_password_user(*, user, password: str):
        existing_link = deps.AuthExternalSubject.query.filter_by(
            user_id=user.id,
            issuer=mcp_oidc_issuer(),
        ).first()
        if existing_link is not None:
            return _build_external_identity(subject=existing_link.subject, email=user.email)
        return _create_zitadel_human_user(
            email=user.email,
            password=password,
            email_is_verified=True,
        )

    def _maybe_migrate_local_password_user(*, email: str, password: str):
        user = _local_user_by_email(email=email)
        if user is None or user.email_verified_at is None:
            return None
        if not isinstance(user.password_hash, str) or not user.password_hash.strip():
            return None
        if not check_password_hash(user.password_hash, password):
            return None
        return _ensure_zitadel_user_for_local_password_user(user=user, password=password)

    def _linked_zitadel_subject_for_email(*, email: str, allow_unverified: bool = False):
        user = _local_user_by_email(email=email)
        if user is None:
            return None
        if user.email_verified_at is None and not allow_unverified:
            return None
        existing_link = deps.AuthExternalSubject.query.filter_by(
            user_id=user.id,
            issuer=mcp_oidc_issuer(),
        ).first()
        if existing_link is not None:
            return existing_link.subject
        if isinstance(user.password_hash, str) and user.password_hash.strip():
            try:
                migrated = _create_zitadel_human_user(
                    email=user.email,
                    password=secrets.token_urlsafe(24),
                    email_is_verified=user.email_verified_at is not None,
                )
            except HTTPException as exc:
                description = exc.description if isinstance(exc.description, str) else ""
                if exc.code == 409 and "already exists" in description.lower():
                    existing_user_id = _search_zitadel_user_id_by_login_name(login_name=user.email)
                    if not isinstance(existing_user_id, str) or not existing_user_id.strip():
                        raise
                    migrated = _build_external_identity(
                        subject=existing_user_id.strip(),
                        email=user.email,
                    )
                else:
                    raise
            _resolve_user_from_external_identity(external_identity=migrated)
            deps.db.session.commit()
            return migrated.subject
        return None

    def _external_identity_verified_email(external_identity) -> str | None:
        claims = getattr(external_identity, "claims", None)
        if not isinstance(claims, dict):
            return None
        email = claims.get("email")
        email_verified = claims.get("email_verified")
        if not isinstance(email, str) or not email.strip():
            return None
        if email_verified is not True:
            return None
        normalized = deps._normalize_email(email)
        if not deps._is_email_like(normalized):
            return None
        return normalized

    def _resolve_user_from_external_identity(*, external_identity):
        existing_link = deps.AuthExternalSubject.query.filter_by(
            issuer=external_identity.issuer,
            subject=external_identity.subject,
        ).first()
        if existing_link is not None:
            user = deps.db.session.get(deps.AuthUser, existing_link.user_id)
            if user is None:
                abort(401, description="Linked account no longer exists.")
            if user.email_verified_at is None:
                user.email_verified_at = deps._utc_now()
            return ("login", user)

        email = _external_identity_verified_email(external_identity)
        if email is None:
            abort(400, description="Auth provider did not return a verified email address.")

        user = deps.AuthUser.query.filter_by(email=email).first()
        action = "login"
        if user is None:
            user = deps.AuthUser(
                email=email,
                password_hash=None,
                email_verified_at=deps._utc_now(),
            )
            deps.db.session.add(user)
            deps.db.session.flush()
            action = "register"
        elif user.email_verified_at is None:
            user.email_verified_at = deps._utc_now()

        existing_for_user = deps.AuthExternalSubject.query.filter_by(
            user_id=user.id,
            issuer=external_identity.issuer,
            subject=external_identity.subject,
        ).first()
        if existing_for_user is None:
            deps.db.session.add(
                deps.AuthExternalSubject(
                    user_id=user.id,
                    issuer=external_identity.issuer,
                    subject=external_identity.subject,
                )
            )
        return (action, user)

    def _linked_user_for_subject(*, issuer: str, subject: str):
        existing_link = deps.AuthExternalSubject.query.filter_by(
            issuer=issuer,
            subject=subject,
        ).first()
        if existing_link is None:
            return None
        return deps.db.session.get(deps.AuthUser, existing_link.user_id)

    def _full_name_from_parts(*parts: object) -> str | None:
        full_name = " ".join(
            part.strip() for part in parts if isinstance(part, str) and part.strip()
        )
        return full_name or None

    def _user_created_at_or_now(*, user) -> datetime:
        created_at = getattr(user, "created_at", None)
        if isinstance(created_at, datetime):
            return created_at
        return deps._utc_now()

    def _complete_website_auth_for_identity(
        *,
        external_identity,
        next_path: str,
        provider_name: str,
        notification_provider: str | None = None,
    ):
        action, user = _resolve_user_from_external_identity(external_identity=external_identity)
        claims = getattr(external_identity, "claims", None)
        signup_first_name = claims.get("given_name") if isinstance(claims, dict) else None
        signup_last_name = claims.get("family_name") if isinstance(claims, dict) else None
        signup_full_name = _full_name_from_parts(signup_first_name, signup_last_name)
        signup_provider = notification_provider or provider_name
        if not deps._user_has_current_legal_acceptances(user_id=user.id):
            deps.db.session.commit()
            if action == "register":
                deps._send_signup_notification_email(
                    new_user_email=user.email,
                    provider=signup_provider,
                    full_name=signup_full_name,
                    signed_up_at=_user_created_at_or_now(user=user),
                )
            return _pending_website_auth_response(
                user=user,
                next_path=next_path,
                action=action,
                provider_name=provider_name,
            )

        deps._record_signon_event(user_id=user.id, provider=provider_name, action=action)
        deps.db.session.commit()
        if action == "register":
            deps._send_signup_notification_email(
                new_user_email=user.email,
                provider=signup_provider,
                full_name=signup_full_name,
                signed_up_at=_user_created_at_or_now(user=user),
            )
        return _auth_success_response(user=user, next_path=next_path)

    def _pending_website_auth_response(*, user, next_path: str, action: str, provider_name: str):
        resp = make_response(
            jsonify(
                {
                    "status": "legal_required",
                    "next_path": next_path,
                    "user": {"id": user.id, "email": user.email},
                }
            )
        )
        resp.headers["Cache-Control"] = "no-store"
        _set_zitadel_pending_cookie(
            resp,
            {
                "user_id": user.id,
                "next": next_path,
                "action": action,
                "provider": provider_name,
            },
        )
        return resp

    def _resume_incomplete_signup_if_eligible(*, user, next_path: str, provider_name: str):
        subject = _ensure_linked_zitadel_subject_for_user(user=user)
        if subject is None:
            return None
        if not deps._user_has_current_legal_acceptances(user_id=user.id):
            deps.db.session.commit()
            return _pending_website_auth_response(
                user=user,
                next_path=next_path,
                action="register",
                provider_name=provider_name,
            )
        if user.email_verified_at is None:
            _send_zitadel_email_verification_for_user(user=user)
            deps.db.session.commit()
            return _verification_required_response(user=user, next_path=next_path)
        return _auth_success_response(user=user, next_path=next_path)

    def _zitadel_email_verify_url_template(*, deps: AuthDeps) -> str:
        return (
            f"{deps._frontend_base_url()}/verify-email"
            "?user_id={{.UserID}}&code={{.Code}}&org_id={{.OrgID}}"
        )

    def _send_zitadel_email_verification(*, user_id: str, url_template: str) -> None:
        _zitadel_api_json(
            path=f"/v2/users/{user_id}/email/send",
            json_body={"sendCode": {"urlTemplate": url_template}},
        )

    def _ensure_linked_zitadel_subject_for_user(*, user) -> str | None:
        existing_link = deps.AuthExternalSubject.query.filter_by(
            user_id=user.id,
            issuer=mcp_oidc_issuer(),
        ).first()
        if existing_link is not None and isinstance(existing_link.subject, str) and existing_link.subject.strip():
            return existing_link.subject.strip()
        searched_user_id = _search_zitadel_user_id_by_login_name(login_name=user.email)
        if isinstance(searched_user_id, str) and searched_user_id.strip():
            resolved_user_id = searched_user_id.strip()
            deps.db.session.add(
                deps.AuthExternalSubject(
                    user_id=user.id,
                    issuer=mcp_oidc_issuer(),
                    subject=resolved_user_id,
                )
            )
            deps.db.session.flush()
            return resolved_user_id
        return None

    def _send_zitadel_email_verification_for_user(*, user) -> bool:
        subject = _ensure_linked_zitadel_subject_for_user(user=user)
        if not isinstance(subject, str) or not subject.strip():
            return False
        url_template = _zitadel_email_verify_url_template(deps=deps)
        try:
            _send_zitadel_email_verification(user_id=subject, url_template=url_template)
            return True
        except HTTPException as exc:
            description = exc.description if isinstance(exc.description, str) else ""
            if exc.code in {400, 404} and "email not found" in description.lower():
                refreshed_subject = _search_zitadel_user_id_by_login_name(login_name=user.email)
                if (
                    isinstance(refreshed_subject, str)
                    and refreshed_subject.strip()
                    and refreshed_subject.strip() != subject
                ):
                    existing_link = deps.AuthExternalSubject.query.filter_by(
                        user_id=user.id,
                        issuer=mcp_oidc_issuer(),
                    ).first()
                    if existing_link is None:
                        deps.db.session.add(
                            deps.AuthExternalSubject(
                                user_id=user.id,
                                issuer=mcp_oidc_issuer(),
                                subject=refreshed_subject.strip(),
                            )
                        )
                    else:
                        existing_link.subject = refreshed_subject.strip()
                    deps.db.session.flush()
                    _send_zitadel_email_verification(
                        user_id=refreshed_subject.strip(),
                        url_template=url_template,
                    )
                    return True
                existing_link = deps.AuthExternalSubject.query.filter_by(
                    user_id=user.id,
                    issuer=mcp_oidc_issuer(),
                ).first()
                if existing_link is not None:
                    deps.db.session.delete(existing_link)
                    deps.db.session.flush()
                return False
            raise

    def _ensure_pending_signup_remote_user(
        *,
        user,
        password: str,
        first_name: str | None = None,
        last_name: str | None = None,
        send_verification: bool = False,
    ) -> str | None:
        subject = _ensure_linked_zitadel_subject_for_user(user=user)
        if isinstance(subject, str) and subject.strip():
            try:
                if _zitadel_user_email_matches(user_id=subject.strip(), email=user.email):
                    return subject.strip()
            except HTTPException as exc:
                if exc.code == 404:
                    existing_link = deps.AuthExternalSubject.query.filter_by(
                        user_id=user.id,
                        issuer=mcp_oidc_issuer(),
                    ).first()
                    if existing_link is not None:
                        deps.db.session.delete(existing_link)
                        deps.db.session.flush()
                else:
                    raise
        try:
            external_identity = _create_zitadel_human_user(
                email=user.email,
                password=password,
                first_name=first_name,
                last_name=last_name,
                email_is_verified=False,
                verify_url_template=(
                    _zitadel_email_verify_url_template(deps=deps) if send_verification else None
                ),
            )
        except HTTPException as exc:
            description = exc.description if isinstance(exc.description, str) else ""
            if exc.code == 409 and "already exists" in description.lower():
                existing_user_id = _search_zitadel_user_id_by_login_name(login_name=user.email)
                if not isinstance(existing_user_id, str) or not existing_user_id.strip():
                    return None
                external_identity = _build_external_identity(
                    subject=existing_user_id.strip(),
                    email=user.email,
                    email_verified=False,
                )
            else:
                raise
        existing_link = deps.AuthExternalSubject.query.filter_by(
            user_id=user.id,
            issuer=mcp_oidc_issuer(),
        ).first()
        if existing_link is None:
            deps.db.session.add(
                deps.AuthExternalSubject(
                    user_id=user.id,
                    issuer=mcp_oidc_issuer(),
                    subject=external_identity.subject,
                )
            )
        else:
            existing_link.subject = external_identity.subject
        deps.db.session.flush()
        return external_identity.subject if isinstance(external_identity.subject, str) else None

    def _verification_required_response(*, user, next_path: str):
        resp = make_response(
            jsonify(
                {
                    "status": "verification_required",
                    "next_path": next_path,
                    "user": {"id": user.id, "email": user.email},
                }
            )
        )
        resp.headers["Cache-Control"] = "no-store"
        return resp

    def _pending_login_block_response(*, user, next_path: str):
        has_legal = deps._user_has_current_legal_acceptances(user_id=user.id)
        if user.email_verified_at is None and not has_legal:
            abort(400, description="You need to accept the terms and verify your email before signing in.")
        if user.email_verified_at is None:
            abort(400, description="Verify your email before signing in.")
        if not has_legal:
            abort(400, description="You need to accept the terms before signing in.")
        return None

    def _ensure_pending_signup_user(
        *,
        external_identity,
    ):
        email_claim = external_identity.claims.get("email")
        if not isinstance(email_claim, str) or not email_claim.strip():
            abort(400, description="Auth provider did not return an email address.")
        email = deps._normalize_email(email_claim)
        user = deps.AuthUser.query.filter_by(email=email).first()
        if user is None:
            user = deps.AuthUser(
                email=email,
                password_hash=None,
                email_verified_at=None,
            )
            deps.db.session.add(user)
            deps.db.session.flush()
        existing_for_user = deps.AuthExternalSubject.query.filter_by(
            user_id=user.id,
            issuer=external_identity.issuer,
            subject=external_identity.subject,
        ).first()
        if existing_for_user is None:
            deps.db.session.add(
                deps.AuthExternalSubject(
                    user_id=user.id,
                    issuer=external_identity.issuer,
                    subject=external_identity.subject,
                )
            )
        return user

    def _delete_zitadel_user_if_linked(*, user) -> None:
        links = (
            deps.AuthExternalSubject.query.filter_by(user_id=user.id, issuer=mcp_oidc_issuer())
            .order_by(deps.AuthExternalSubject.id.asc())
            .all()
        )
        for link in links:
            try:
                _zitadel_api_request(
                    path=f"/v2/users/{link.subject}",
                    method="DELETE",
                )
            except HTTPException as exc:
                if exc.code == 404:
                    continue
                raise

    def _search_zitadel_user_id_by_login_name(*, login_name: str) -> str | None:
        payload = _zitadel_api_request(
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

    def _zitadel_google_identity_from_intent(*, payload: dict[str, object], user_id: str | None):
        idp_information = payload.get("idpInformation")
        if not isinstance(idp_information, dict):
            abort(502, description="ZITADEL did not return identity provider information.")
        raw_information = idp_information.get("rawInformation")
        raw_user = raw_information.get("User") if isinstance(raw_information, dict) else None
        if not isinstance(raw_user, dict):
            abort(502, description="ZITADEL did not return Google profile information.")
        external_subject = raw_user.get("sub")
        if not isinstance(external_subject, str) or not external_subject.strip():
            abort(502, description="Google did not return a stable user identifier.")
        idp_id = idp_information.get("idpId")
        if not isinstance(idp_id, str) or not idp_id.strip():
            abort(502, description="ZITADEL did not return the Google provider identifier.")

        resolved_user_id = user_id.strip() if isinstance(user_id, str) and user_id.strip() else None
        if resolved_user_id is None:
            email = raw_user.get("email")
            email_verified = raw_user.get("email_verified")
            if not isinstance(email, str) or not email.strip() or email_verified is not True:
                abort(400, description="Google did not return a verified email address.")
            existing_subject = _linked_zitadel_subject_for_email(email=deps._normalize_email(email))
            if isinstance(existing_subject, str) and existing_subject.strip():
                resolved_user_id = existing_subject.strip()
        if resolved_user_id is None:
            email = cast(str, raw_user.get("email"))
            display_name = raw_user.get("name")
            if not isinstance(display_name, str) or not display_name.strip():
                display_name = email.strip()
            given_name = raw_user.get("given_name")
            family_name = raw_user.get("family_name")
            create_payload: dict[str, object] = {
                "username": email.strip(),
                "profile": {
                    "displayName": display_name,
                },
                "email": {
                    "email": email.strip(),
                    "isVerified": True,
                },
                "idpLinks": [
                    {
                        "idpId": idp_id,
                        "userId": external_subject.strip(),
                        "userName": display_name,
                    }
                ],
            }
            if isinstance(given_name, str) and given_name.strip():
                cast(dict[str, object], create_payload["profile"])["givenName"] = given_name.strip()
            if isinstance(family_name, str) and family_name.strip():
                cast(dict[str, object], create_payload["profile"])["familyName"] = family_name.strip()
            try:
                created = _zitadel_api_json(path="/v2/users/human", json_body=create_payload)
                user_id_candidate = created.get("userId") or created.get("user_id")
                if not isinstance(user_id_candidate, str) or not user_id_candidate.strip():
                    abort(502, description="ZITADEL did not return a user identifier for the Google login.")
                _ensure_zitadel_default_user_grant(user_id=user_id_candidate.strip())
                resolved_user_id = user_id_candidate.strip()
            except HTTPException as exc:
                description = exc.description if isinstance(exc.description, str) else ""
                if exc.code == 409 and "already exists" in description.lower():
                    existing_user_id = _search_zitadel_user_id_by_login_name(login_name=email.strip())
                    if isinstance(existing_user_id, str) and existing_user_id.strip():
                        resolved_user_id = existing_user_id.strip()
                    else:
                        raise
                else:
                    raise

        claims: dict[str, object] = {
            "email": raw_user.get("email"),
            "email_verified": raw_user.get("email_verified"),
            "name": raw_user.get("name"),
            "given_name": raw_user.get("given_name"),
            "family_name": raw_user.get("family_name"),
            "picture": raw_user.get("picture"),
        }
        return ExternalIdentity(
            issuer=mcp_oidc_issuer(),
            subject=resolved_user_id,
            scopes=frozenset(),
            audiences=frozenset(),
            claims=claims,
        )

    def _auth_success_response(*, user, next_path: str, resp_code: int = 200):
        token = deps._issue_session_token(user.id)
        payload: dict[str, object] = {
            "status": "authenticated",
            "next_path": next_path,
            "user": {"id": user.id, "email": user.email},
        }
        if deps._auth_session_transport() == "bearer":
            payload["session_token"] = token
        resp = make_response(jsonify(payload), resp_code)
        resp.headers["Cache-Control"] = "no-store"
        if deps._auth_session_transport() == "cookie":
            deps._set_auth_cookies(resp, session_token=token)
        return resp

    def _mcp_access_token_response(
        *,
        access_token: str,
        next_path: str,
        token_payload: dict[str, object],
    ):
        payload: dict[str, object] = {
            "status": "mcp_token",
            "next_path": next_path,
            "access_token": access_token,
        }
        token_type = token_payload.get("token_type")
        if isinstance(token_type, str) and token_type.strip():
            payload["token_type"] = token_type.strip()
        expires_in = token_payload.get("expires_in")
        if isinstance(expires_in, int):
            payload["expires_in"] = expires_in
        elif isinstance(expires_in, float) and expires_in.is_integer():
            payload["expires_in"] = int(expires_in)
        scope = token_payload.get("scope")
        if isinstance(scope, str) and scope.strip():
            payload["scope"] = scope.strip()
        resp = make_response(jsonify(payload), 200)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    def _oauth_code_hash(code: str) -> str:
        return sha256(code.encode("utf-8")).hexdigest()

    def _oauth_scope_string(raw: str | None) -> str:
        allowed = set(mcp_supported_scopes())
        requested = [part.strip() for part in (raw or "").split() if part.strip()]
        if not requested:
            return " ".join(mcp_supported_scopes())
        deduped: list[str] = []
        for scope_name in requested:
            if scope_name not in allowed:
                abort(400, description=f"Unsupported OAuth scope: {scope_name}")
            if scope_name not in deduped:
                deduped.append(scope_name)
        return " ".join(deduped)

    def _oauth_redirect_uri_allowed(client, redirect_uri: str) -> bool:
        raw_uris = getattr(client, "redirect_uris", None)
        if not isinstance(raw_uris, list):
            return False
        registered = [str(item).strip() for item in raw_uris if str(item).strip()]
        if redirect_uri in registered:
            return True
        # RFC 8252 §7.3: loopback redirect URIs must allow any port.
        try:
            req = urlparse(redirect_uri)
        except Exception:
            return False
        if req.scheme != "http" or req.hostname not in _OAUTH_LOOPBACK_REDIRECT_HOSTS:
            return False
        for reg_uri in registered:
            try:
                reg = urlparse(reg_uri)
            except Exception:
                continue
            if (
                reg.scheme == "http"
                and reg.hostname in _OAUTH_LOOPBACK_REDIRECT_HOSTS
                and reg.path == req.path
                and reg.query == req.query
            ):
                return True
        return False

    def _oauth_client_supports_grant_type(client, grant_type: str) -> bool:
        raw_grant_types = getattr(client, "grant_types", None)
        if not isinstance(raw_grant_types, list):
            return False
        normalized = {
            str(item).strip()
            for item in raw_grant_types
            if isinstance(item, str) and item.strip()
        }
        return grant_type in normalized

    def _valid_oauth_redirect_uri(redirect_uri: str) -> bool:
        if "\\" in redirect_uri:
            return False
        try:
            parsed = urlparse(redirect_uri)
        except Exception:
            return False
        if parsed.fragment or parsed.username or parsed.password or not parsed.hostname:
            return False
        if parsed.scheme == "https":
            return True
        return parsed.scheme == "http" and parsed.hostname in _OAUTH_LOOPBACK_REDIRECT_HOSTS

    def _oauth_redirect_location(redirect_uri: str, params: dict[str, str]) -> str:
        parsed = urlparse(redirect_uri)
        encoded_params = urlencode(params)
        query = f"{parsed.query}&{encoded_params}" if parsed.query else encoded_params
        return urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, query, ""))

    def _oauth_active_signing_key():
        key = deps.AuthOAuthSigningKey.query.filter_by(active=True).first()
        if key is not None:
            return key
        kid, private_pem = generate_signing_keypair()
        key = deps.AuthOAuthSigningKey(kid=kid, private_pem=private_pem, active=True)
        deps.db.session.add(key)
        deps.db.session.commit()
        return key

    def _oauth_error_response(message: str, *, code: int = 400):
        resp = make_response(
            f"<html><body><h1>OAuth request failed</h1><p>{escape(message)}</p></body></html>",
            code,
        )
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        resp.headers["Cache-Control"] = "no-store"
        return resp

    def _oauth_redirect_error(redirect_uri: str, *, error: str, state: str | None = None):
        params = {"error": error}
        if isinstance(state, str) and state.strip():
            params["state"] = state.strip()
        return redirect(_oauth_redirect_location(redirect_uri, params), code=302)

    def _oauth_authorize_pending_payload(
        *,
        client_id: str,
        redirect_uri: str,
        state: str | None,
        scope: str,
        code_challenge: str,
        code_challenge_method: str,
    ) -> dict[str, str]:
        payload = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "scope": scope,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
        }
        if state:
            payload["state"] = state
        return payload

    def _oauth_authorize_bridge_response(*, pending_payload: dict[str, str]):
        next_path = "/v1/auth/oauth/authorize"
        login_url = f"{deps._frontend_base_url()}/login?{urlencode({'next': next_path})}"
        body = f"""<!doctype html>
<html><body>
<script>
window.location.replace({json.dumps(login_url)});
</script>
<noscript><meta http-equiv="refresh" content="0; url={escape(login_url, quote=True)}"></noscript>
</body></html>"""
        resp = make_response(body, 200)
        resp.headers["Content-Type"] = "text/html; charset=utf-8"
        resp.headers["Cache-Control"] = "no-store"
        _set_oauth_authorize_cookie(resp, pending_payload)
        return resp

    def _oauth_response_with_pending_cookie_cleared(resp):
        _clear_oauth_authorize_cookie(resp)
        return resp

    def _oauth_authenticated_user():
        cookie_payload = _load_oauth_browser_cookie()
        if cookie_payload is not None:
            user_id = cookie_payload.get("user_id")
            if isinstance(user_id, str) and user_id.strip():
                user = deps.db.session.get(deps.AuthUser, user_id.strip())
                if user is not None and cast(object, user.email_verified_at) is not None:
                    return user
        try:
            user, _ctx = deps._require_verified_user()
            return user
        except HTTPException:
            return None

    @auth_blp.get("/oauth/.well-known/openid-configuration")
    def oauth_openid_configuration():
        resp = make_response(jsonify(mcp_oauth_metadata()), 200)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.get("/oauth/.well-known/oauth-authorization-server")
    def oauth_authorization_server_metadata():
        resp = make_response(jsonify(mcp_oauth_metadata()), 200)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.get("/oauth/jwks.json")
    def oauth_jwks():
        keys = deps.AuthOAuthSigningKey.query.filter_by(active=True).all()
        payload = {
            "keys": [
                public_jwk_from_private_pem(kid=key.kid, private_pem=key.private_pem)
                for key in keys
            ]
        }
        resp = make_response(jsonify(payload), 200)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.post("/oauth/register")
    def oauth_register():
        deps._require_auth_db()
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            abort(400, description="Invalid client registration payload.")
        unsupported_keys = {
            "jwks",
            "jwks_uri",
            "logo_uri",
            "policy_uri",
            "tos_uri",
            "client_secret",
        }.intersection(data.keys())
        if unsupported_keys:
            abort(400, description="Unsupported OAuth client metadata.")
        redirect_uris = data.get("redirect_uris")
        if not isinstance(redirect_uris, list) or not redirect_uris:
            abort(400, description="redirect_uris must be a non-empty array.")
        normalized_redirect_uris: list[str] = []
        for raw_uri in redirect_uris:
            if not isinstance(raw_uri, str) or not raw_uri.strip():
                abort(400, description="redirect_uris must contain only non-empty strings.")
            normalized = raw_uri.strip()
            if not _valid_oauth_redirect_uri(normalized):
                abort(400, description="redirect_uris must be HTTPS URLs or HTTP loopback URLs without fragments.")
            if normalized in normalized_redirect_uris:
                continue
            normalized_redirect_uris.append(normalized)
        token_endpoint_auth_method = data.get("token_endpoint_auth_method", "none")
        if token_endpoint_auth_method != "none":
            abort(400, description="Only public clients with token_endpoint_auth_method=none are supported.")
        grant_types = data.get("grant_types", ["authorization_code"])
        if not isinstance(grant_types, list) or not all(
            isinstance(item, str) and item.strip() for item in grant_types
        ):
            abort(400, description="grant_types must be a non-empty array of strings.")
        normalized_grant_types = {
            item.strip() for item in grant_types if isinstance(item, str) and item.strip()
        }
        if not normalized_grant_types or "authorization_code" not in normalized_grant_types:
            abort(400, description="OAuth clients must support the authorization_code grant.")
        unsupported_grant_types = normalized_grant_types.difference(
            {"authorization_code", "refresh_token"}
        )
        if unsupported_grant_types:
            abort(400, description="Only authorization_code clients are supported.")

        response_types = data.get("response_types", ["code"])
        if not isinstance(response_types, list) or not all(
            isinstance(item, str) and item.strip() for item in response_types
        ):
            abort(400, description="response_types must be a non-empty array of strings.")
        normalized_response_types = {
            item.strip() for item in response_types if isinstance(item, str) and item.strip()
        }
        if normalized_response_types != {"code"}:
            abort(400, description="Only code response types are supported.")
        client = deps.AuthOAuthClient(
            client_id=secrets.token_urlsafe(24),
            client_name=(data.get("client_name") or None) if isinstance(data.get("client_name"), str) else None,
            redirect_uris=normalized_redirect_uris,
            token_endpoint_auth_method="none",
            grant_types=sorted(normalized_grant_types),
            response_types=["code"],
            created_by_ip=deps._request_ip_address(),
        )
        deps.db.session.add(client)
        deps.db.session.commit()
        payload = {
            "client_id": client.client_id,
            "client_id_issued_at": int(client.created_at.timestamp()),
            "redirect_uris": normalized_redirect_uris,
            "grant_types": sorted(normalized_grant_types),
            "response_types": ["code"],
            "token_endpoint_auth_method": "none",
        }
        if isinstance(client.client_name, str) and client.client_name.strip():
            payload["client_name"] = client.client_name.strip()
        resp = make_response(jsonify(payload), 201)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.post("/oauth/browser-session")
    def oauth_browser_session():
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        resp = make_response(jsonify({"status": "ok"}), 200)
        _set_oauth_browser_cookie(resp, {"user_id": user.id})
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.get("/oauth/authorize")
    def oauth_authorize():
        deps._require_auth_db()
        pending = _load_oauth_authorize_cookie()
        resuming_pending = pending is not None and not request.args
        client_id = request.args.get("client_id", "")
        redirect_uri = request.args.get("redirect_uri", "")
        response_type = request.args.get("response_type", "")
        state_raw = request.args.get("state", "")
        scope_raw = request.args.get("scope")
        code_challenge = request.args.get("code_challenge", "")
        code_challenge_method = request.args.get("code_challenge_method", "")
        if resuming_pending:
            assert pending is not None
            client_id = pending.get("client_id", "")
            redirect_uri = pending.get("redirect_uri", "")
            state_raw = pending.get("state", "")
            scope_raw = pending.get("scope")
            code_challenge = pending.get("code_challenge", "")
            code_challenge_method = pending.get("code_challenge_method", "")
        client_id = client_id.strip()
        redirect_uri = redirect_uri.strip()
        response_type = "code" if resuming_pending else response_type.strip()
        state = state_raw.strip() or None
        scope = _oauth_scope_string(scope_raw)
        code_challenge = code_challenge.strip()
        code_challenge_method = code_challenge_method.strip()
        if not client_id or not redirect_uri:
            resp = _oauth_error_response("Missing OAuth client_id or redirect_uri.")
            return _oauth_response_with_pending_cookie_cleared(resp) if resuming_pending else resp
        client = deps.AuthOAuthClient.query.filter_by(client_id=client_id).first()
        if client is None or not _oauth_redirect_uri_allowed(client, redirect_uri):
            resp = _oauth_error_response("Invalid OAuth client or redirect URI.")
            return _oauth_response_with_pending_cookie_cleared(resp) if resuming_pending else resp
        if response_type != "code":
            resp = _oauth_redirect_error(redirect_uri, error="unsupported_response_type", state=state)
            return _oauth_response_with_pending_cookie_cleared(resp) if resuming_pending else resp
        if not scope:
            resp = _oauth_redirect_error(redirect_uri, error="invalid_scope", state=state)
            return _oauth_response_with_pending_cookie_cleared(resp) if resuming_pending else resp
        if not code_challenge or code_challenge_method != "S256":
            resp = _oauth_redirect_error(redirect_uri, error="invalid_request", state=state)
            return _oauth_response_with_pending_cookie_cleared(resp) if resuming_pending else resp
        pending_payload = _oauth_authorize_pending_payload(
            client_id=client_id,
            redirect_uri=redirect_uri,
            state=state,
            scope=scope,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
        )

        user = _oauth_authenticated_user()
        if user is None:
            if deps._auth_session_transport() == "bearer":
                return _oauth_authorize_bridge_response(pending_payload=pending_payload)
            resp = redirect(
                f"{deps._frontend_base_url()}/login?{urlencode({'next': '/v1/auth/oauth/authorize'})}",
                code=302,
            )
            _set_oauth_authorize_cookie(resp, pending_payload)
            return resp

        if not deps._user_has_current_legal_acceptances(user_id=user.id):
            resp = redirect(
                f"{deps._frontend_base_url()}/login?{urlencode({'next': '/v1/auth/oauth/authorize'})}",
                code=302,
            )
            _set_oauth_authorize_cookie(resp, pending_payload)
            return resp
        linked_subject = _ensure_linked_zitadel_subject_for_user(user=user)
        if not linked_subject:
            linked_subject = _linked_zitadel_subject_for_email(email=user.email)
        if not linked_subject:
            resp = _oauth_error_response("Pandects could not link this account to MCP identity.", code=403)
            return _oauth_response_with_pending_cookie_cleared(resp) if resuming_pending else resp

        raw_code = secrets.token_urlsafe(32)
        code = deps.AuthOAuthAuthorizationCode(
            code_hash=_oauth_code_hash(raw_code),
            client_id=client.client_id,
            user_id=user.id,
            redirect_uri=redirect_uri,
            scope=scope,
            code_challenge=code_challenge,
            code_challenge_method=code_challenge_method,
            expires_at=deps._utc_now() + timedelta(seconds=mcp_oauth_authorization_code_ttl_seconds()),
        )
        deps.db.session.add(code)
        deps.db.session.commit()
        _params = {"code": raw_code}
        if state:
            _params["state"] = state
        resp = redirect(_oauth_redirect_location(redirect_uri, _params), code=302)
        _clear_oauth_browser_cookie(resp)
        _clear_oauth_authorize_cookie(resp)
        return resp

    def _issue_token_pair(*, user_id: str, client_id: str, scope: str) -> dict[str, object]:
        active_key = _oauth_active_signing_key()
        access_token = encode_access_token(
            private_pem=active_key.private_pem,
            kid=active_key.kid,
            claims=access_token_claims(
                subject=user_id,
                audience=mcp_resource_url(),
                scope=scope,
                token_id=str(uuid.uuid4()),
            ),
        )
        token_payload: dict[str, object] = {
            "access_token": access_token,
            "token_type": "Bearer",
            "expires_in": mcp_oauth_access_token_ttl_seconds(),
            "scope": scope,
        }
        client = deps.AuthOAuthClient.query.filter_by(client_id=client_id).first()
        if client is not None and _oauth_client_supports_grant_type(client, "refresh_token"):
            raw_refresh = secrets.token_urlsafe(32)
            refresh_record = deps.AuthOAuthRefreshToken(
                token_hash=_oauth_code_hash(raw_refresh),
                client_id=client_id,
                user_id=user_id,
                scope=scope,
                expires_at=deps._utc_now() + timedelta(seconds=mcp_oauth_refresh_token_ttl_seconds()),
            )
            deps.db.session.add(refresh_record)
            token_payload["refresh_token"] = raw_refresh
        return token_payload

    @auth_blp.post("/oauth/token")
    def oauth_token():
        deps._require_auth_db()
        grant_type = request.form.get("grant_type", "").strip()
        client_id = request.form.get("client_id", "").strip()

        if grant_type == "authorization_code":
            code = request.form.get("code", "").strip()
            redirect_uri = request.form.get("redirect_uri", "").strip()
            code_verifier = request.form.get("code_verifier", "").strip()
            if not code or not client_id or not redirect_uri or not code_verifier:
                abort(400, description="Missing OAuth token exchange fields.")
            client = deps.AuthOAuthClient.query.filter_by(client_id=client_id).first()
            if client is None or not _oauth_redirect_uri_allowed(client, redirect_uri):
                abort(400, description="Invalid OAuth client or redirect URI.")
            if not _oauth_client_supports_grant_type(client, "authorization_code"):
                abort(400, description="OAuth client does not support the authorization_code grant.")
            auth_code = deps.AuthOAuthAuthorizationCode.query.filter_by(
                code_hash=_oauth_code_hash(code),
                client_id=client.client_id,
            ).first()
            if auth_code is None or auth_code.used_at is not None or auth_code.expires_at < deps._utc_now():
                abort(400, description="Invalid or expired OAuth code.")
            if auth_code.redirect_uri != redirect_uri:
                abort(400, description="OAuth redirect URI mismatch.")
            expected_challenge = _build_pkce_challenge(code_verifier)
            if not secrets.compare_digest(expected_challenge, auth_code.code_challenge):
                abort(400, description="OAuth PKCE verification failed.")
            user = deps.db.session.get(deps.AuthUser, auth_code.user_id)
            if user is None or cast(object, user.email_verified_at) is None:
                abort(403, description="Linked Pandects account is not verified.")
            auth_code.used_at = deps._utc_now()
            token_payload = _issue_token_pair(
                user_id=user.id,
                client_id=client.client_id,
                scope=auth_code.scope,
            )
            deps.db.session.commit()
            resp = make_response(jsonify(token_payload), 200)
            resp.headers["Cache-Control"] = "no-store"
            return resp

        if grant_type == "refresh_token":
            raw_refresh = request.form.get("refresh_token", "").strip()
            if not raw_refresh or not client_id:
                abort(400, description="Missing OAuth refresh token fields.")
            client = deps.AuthOAuthClient.query.filter_by(client_id=client_id).first()
            if client is None:
                abort(400, description="Invalid OAuth client.")
            if not _oauth_client_supports_grant_type(client, "refresh_token"):
                abort(400, description="OAuth client does not support the refresh_token grant.")
            refresh_record = deps.AuthOAuthRefreshToken.query.filter_by(
                token_hash=_oauth_code_hash(raw_refresh),
                client_id=client.client_id,
            ).first()
            now = deps._utc_now()
            if (
                refresh_record is None
                or refresh_record.revoked_at is not None
                or refresh_record.used_at is not None
                or refresh_record.expires_at < now
            ):
                abort(400, description="Invalid or expired refresh token.")
            user = deps.db.session.get(deps.AuthUser, refresh_record.user_id)
            if user is None or cast(object, user.email_verified_at) is None:
                abort(403, description="Linked Pandects account is not verified.")
            # Rotate: mark old token used, issue a new pair.
            refresh_record.used_at = now
            token_payload = _issue_token_pair(
                user_id=user.id,
                client_id=client.client_id,
                scope=refresh_record.scope,
            )
            deps.db.session.commit()
            resp = make_response(jsonify(token_payload), 200)
            resp.headers["Cache-Control"] = "no-store"
            return resp

        abort(400, description="Unsupported grant_type.")

    @auth_blp.route("/me", methods=["GET"])
    def auth_me():
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        resp = make_response(jsonify({"user": {"id": user.id, "email": user.email}}))
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/external-subjects", methods=["GET"])
    def auth_list_external_subjects():
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        if deps._auth_is_mocked():
            abort(501, description="External identity linking is unavailable in mocked auth mode.")
        try:
            links = (
                deps.AuthExternalSubject.query.filter_by(user_id=user.id)
                .order_by(deps.AuthExternalSubject.created_at.desc())
                .all()
            )
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")
        resp = make_response(
            jsonify(
                {
                    "links": [
                        {
                            "id": link.id,
                            "issuer": link.issuer,
                            "subject": link.subject,
                            "created_at": link.created_at.isoformat(),
                        }
                        for link in links
                    ]
                }
            )
        )
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/external-subjects", methods=["POST"])
    def auth_link_external_subject():
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        if deps._auth_is_mocked():
            abort(501, description="External identity linking is unavailable in mocked auth mode.")
        data = deps._load_json(deps.AuthExternalSubjectLinkSchema())
        access_token = data.get("access_token")
        provider = data.get("provider")
        if not isinstance(access_token, str) or not access_token.strip():
            abort(400, description="External access token is required.")
        if provider is not None and not isinstance(provider, str):
            abort(400, description="Provider must be a string.")
        provider_name = deps._resolve_mcp_identity_provider_name(
            provider if isinstance(provider, str) else None
        )
        try:
            external_identity = deps._authenticate_external_identity(
                access_token=access_token.strip(),
                provider_name=provider_name,
            )
        except Exception as exc:
            message = str(exc)
            if "Unsupported MCP identity provider" in message:
                abort(400, description="Unsupported external identity provider.")
            if isinstance(exc, McpAuthError):
                if exc.status_code >= 500:
                    abort(503, description="External identity provider is unavailable right now.")
                abort(400, description="External access token is invalid or expired.")
            abort(503, description="External identity provider is unavailable right now.")

        try:
            status, link, code = _link_external_identity_for_user(
                user=user,
                provider_name=provider_name,
                external_identity=external_identity,
            )
        except HTTPException:
            deps.db.session.rollback()
            raise
        except SQLAlchemyError:
            deps.db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")

        resp = make_response(
            jsonify(
                {
                    "status": status,
                    "link": _external_link_payload(link=link, provider_name=provider_name),
                }
            ),
            code,
        )
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/external-subjects/<int:link_id>", methods=["DELETE"])
    def auth_unlink_external_subject(link_id: int):
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        if deps._auth_is_mocked():
            abort(501, description="External identity linking is unavailable in mocked auth mode.")
        try:
            link = deps.AuthExternalSubject.query.filter_by(id=link_id, user_id=user.id).first()
            if link is None:
                abort(404)
            deps.db.session.delete(link)
            deps.db.session.commit()
        except HTTPException:
            deps.db.session.rollback()
            raise
        except SQLAlchemyError:
            deps.db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")

        resp = deps._status_response("unlinked")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/external-subjects/zitadel/start", methods=["GET"])
    def auth_zitadel_link_start():
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        if deps._auth_is_mocked():
            abort(501, description="External identity linking is unavailable in mocked auth mode.")
        _ = user

        next_path = deps._safe_next_path(request.args.get("next")) or "/account"
        state = secrets.token_urlsafe(32)
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = _build_pkce_challenge(code_verifier)
        cookie_payload = {
            "state": state,
            "code_verifier": code_verifier,
            "next": next_path,
            "provider": "zitadel",
        }
        params = {
            "client_id": _zitadel_client_id(),
            "redirect_uri": _zitadel_redirect_uri(deps),
            "response_type": "code",
            "scope": _zitadel_scopes(),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        resource = _zitadel_resource()
        audience = _zitadel_audience()
        if resource:
            params["resource"] = resource
        if audience:
            params["audience"] = audience

        resp = make_response(
            jsonify(
                {
                    "authorize_url": f"{_zitadel_authorization_endpoint()}?{urlencode(params)}",
                }
            )
        )
        resp.headers["Cache-Control"] = "no-store"
        _set_zitadel_link_cookie(resp, cookie_payload)
        return resp

    @auth_blp.route("/external-subjects/zitadel/complete", methods=["POST"])
    def auth_zitadel_link_complete():
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        if deps._auth_is_mocked():
            abort(501, description="External identity linking is unavailable in mocked auth mode.")

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            abort(400, description="Invalid callback payload.")
        code = data.get("code")
        state = data.get("state")
        if not isinstance(code, str) or not code.strip():
            abort(400, description="Missing authorization code.")
        if not isinstance(state, str) or not state.strip():
            abort(400, description="Missing authorization state.")

        cookie_payload = _load_zitadel_link_cookie()
        if not cookie_payload:
            abort(400, description="Invalid ZITADEL authorization state.")
        expected_state = cookie_payload.get("state")
        code_verifier = cookie_payload.get("code_verifier")
        next_path = deps._safe_next_path(cookie_payload.get("next")) or "/account"
        provider_name = deps._resolve_mcp_identity_provider_name(cookie_payload.get("provider"))
        if (
            not isinstance(expected_state, str)
            or not expected_state.strip()
            or not secrets.compare_digest(expected_state, state.strip())
            or not isinstance(code_verifier, str)
            or not code_verifier.strip()
        ):
            resp = make_response(
                jsonify({"message": "Invalid ZITADEL authorization state."}),
                400,
            )
            resp.headers["Cache-Control"] = "no-store"
            _clear_zitadel_link_cookie(resp)
            return resp

        token_data = {
            "grant_type": "authorization_code",
            "client_id": _zitadel_client_id(),
            "code": code.strip(),
            "code_verifier": code_verifier,
            "redirect_uri": _zitadel_redirect_uri(deps),
        }
        resource = _zitadel_resource()
        audience = _zitadel_audience()
        if resource:
            token_data["resource"] = resource
        if audience:
            token_data["audience"] = audience

        try:
            token_payload = deps._oidc_fetch_json(_zitadel_token_endpoint(), data=token_data)
            access_token = token_payload.get("access_token")
            if not isinstance(access_token, str) or not access_token.strip():
                abort(502, description="ZITADEL token response did not include an access token.")
            external_identity = deps._authenticate_external_identity(
                access_token=access_token.strip(),
                provider_name=provider_name,
            )
            status, link, code = _link_external_identity_for_user(
                user=user,
                provider_name=provider_name,
                external_identity=external_identity,
            )
        except HTTPException:
            deps.db.session.rollback()
            raise
        except SQLAlchemyError:
            deps.db.session.rollback()
            current_app.logger.exception("ZITADEL link completion failed due to auth DB error.")
            abort(503, description="Auth backend is unavailable right now.")
        except Exception as exc:
            deps.db.session.rollback()
            current_app.logger.exception("ZITADEL link completion failed.")
            message = str(exc)
            if "Unsupported MCP identity provider" in message:
                abort(400, description="Unsupported external identity provider.")
            if isinstance(exc, McpAuthError):
                if exc.status_code >= 500:
                    abort(503, description="External identity provider is unavailable right now.")
                abort(400, description="External access token is invalid or expired.")
            abort(503, description="External identity provider is unavailable right now.")

        resp = make_response(
            jsonify(
                {
                    "status": status,
                    "link": _external_link_payload(link=link, provider_name=provider_name),
                    "return_to": next_path,
                }
            ),
            code,
        )
        resp.headers["Cache-Control"] = "no-store"
        _clear_zitadel_link_cookie(resp)
        return resp

    @auth_blp.route("/zitadel/start", methods=["GET"])
    def auth_zitadel_start():
        deps._require_auth_db()
        if deps._auth_is_mocked():
            abort(501, description="ZITADEL auth is unavailable in mocked auth mode.")

        next_path = deps._safe_next_path(request.args.get("next")) or "/account"
        provider = request.args.get("provider", "email")
        provider_name = provider.strip().lower() if isinstance(provider, str) else "email"
        if provider_name not in {"email", "google"}:
            abort(400, description="Unsupported auth provider.")
        prompt_raw = request.args.get("prompt")
        prompt = prompt_raw.strip().lower() if isinstance(prompt_raw, str) and prompt_raw.strip() else None
        allowed_prompts = {"login", "create", "select_account"}
        if prompt is not None and prompt not in allowed_prompts:
            abort(400, description="Unsupported auth prompt.")

        state = secrets.token_urlsafe(32)
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = _build_pkce_challenge(code_verifier)
        cookie_payload = {
            "state": state,
            "code_verifier": code_verifier,
            "next": next_path,
            "provider": provider_name,
        }
        params = {
            "client_id": _zitadel_client_id(),
            "redirect_uri": _website_zitadel_redirect_uri(deps),
            "response_type": "code",
            "scope": _zitadel_scopes(),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
        }
        resource = _zitadel_resource()
        audience = _zitadel_audience()
        if resource:
            params["resource"] = resource
        if audience:
            params["audience"] = audience
        if prompt is not None:
            params["prompt"] = prompt
        if provider_name == "google":
            idp_hint = os.environ.get("AUTH_ZITADEL_GOOGLE_IDP_HINT", "").strip()
            if idp_hint:
                params["idp_hint"] = idp_hint

        resp = make_response(
            jsonify(
                {
                    "authorize_url": f"{_zitadel_authorization_endpoint()}?{urlencode(params)}",
                }
            )
        )
        resp.headers["Cache-Control"] = "no-store"
        _set_zitadel_web_cookie(resp, cookie_payload)
        return resp

    @auth_blp.route("/mcp-token/start", methods=["GET"])
    def auth_mcp_token_start():
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        if deps._auth_is_mocked():
            abort(501, description="ZITADEL auth is unavailable in mocked auth mode.")
        _ = user

        next_path = deps._safe_next_path(request.args.get("next")) or "/account"
        state = secrets.token_urlsafe(32)
        code_verifier = secrets.token_urlsafe(64)
        code_challenge = _build_pkce_challenge(code_verifier)
        cookie_payload = {
            "state": state,
            "code_verifier": code_verifier,
            "next": next_path,
            "provider": "zitadel",
            "flow": "mcp_token",
        }
        params = {
            "client_id": _zitadel_client_id(),
            "redirect_uri": _website_zitadel_redirect_uri(deps),
            "response_type": "code",
            "scope": _zitadel_scopes(),
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "prompt": "login",
        }
        resource = _zitadel_resource()
        audience = _zitadel_audience()
        if resource:
            params["resource"] = resource
        if audience:
            params["audience"] = audience

        resp = make_response(
            jsonify(
                {
                    "authorize_url": f"{_zitadel_authorization_endpoint()}?{urlencode(params)}",
                }
            )
        )
        resp.headers["Cache-Control"] = "no-store"
        _set_zitadel_web_cookie(resp, cookie_payload)
        return resp

    @auth_blp.route("/zitadel/google/start", methods=["GET"])
    def auth_zitadel_google_start():
        deps._require_auth_db()
        if deps._auth_is_mocked():
            abort(501, description="ZITADEL auth is unavailable in mocked auth mode.")

        next_path = deps._safe_next_path(request.args.get("next")) or "/account"
        cookie_payload = {
            "next": next_path,
            "provider": "zitadel",
            "flow": "google_intent",
        }
        callback_url = _website_zitadel_redirect_uri(deps)
        try:
            start_payload = _zitadel_api_json(
                path="/v2/idp_intents",
                json_body={
                    "idpId": _zitadel_google_idp_id(),
                    "urls": {
                        "successUrl": callback_url,
                        "failureUrl": callback_url,
                    },
                },
            )
        except HTTPException:
            raise
        except Exception:
            current_app.logger.exception("ZITADEL Google intent start failed.")
            abort(503, description="External identity provider is unavailable right now.")
        authorize_url = start_payload.get("authUrl")
        if not isinstance(authorize_url, str) or not authorize_url.strip():
            abort(502, description="ZITADEL did not return a Google authorization URL.")

        resp = make_response(jsonify({"authorize_url": authorize_url.strip()}))
        resp.headers["Cache-Control"] = "no-store"
        _set_zitadel_web_cookie(resp, cookie_payload)
        return resp

    @auth_blp.route("/zitadel/complete", methods=["POST"])
    def auth_zitadel_complete():
        deps._require_auth_db()
        if deps._auth_is_mocked():
            abort(501, description="ZITADEL auth is unavailable in mocked auth mode.")

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            abort(400, description="Invalid callback payload.")
        code = data.get("code")
        state = data.get("state")
        intent_id = data.get("intent_id")
        intent_token = data.get("intent_token")
        user_id = data.get("user_id")
        has_oauth_code = isinstance(code, str) and bool(code.strip())
        has_intent = isinstance(intent_id, str) and bool(intent_id.strip()) and isinstance(intent_token, str) and bool(intent_token.strip())
        if not has_oauth_code and not has_intent:
            abort(400, description="Missing auth provider callback payload.")
        if has_oauth_code and (not isinstance(state, str) or not state.strip()):
            abort(400, description="Missing authorization state.")

        cookie_payload = _load_zitadel_web_cookie()
        if not cookie_payload:
            return _website_auth_error_payload("Invalid authorization state.")
        next_path = deps._safe_next_path(cookie_payload.get("next")) or "/account"
        provider_name = "zitadel"
        flow = cookie_payload.get("flow")

        try:
            if has_intent:
                if flow != "google_intent":
                    resp = _website_auth_error_payload("Invalid authorization state.")
                    _clear_zitadel_web_cookie(resp)
                    return resp
                retrieved = _zitadel_api_json(
                    path=f"/v2/idp_intents/{cast(str, intent_id).strip()}",
                    json_body={"idpIntentToken": cast(str, intent_token).strip()},
                )
                external_identity = _zitadel_google_identity_from_intent(
                    payload=retrieved,
                    user_id=user_id if isinstance(user_id, str) else None,
                )
            else:
                expected_state = cookie_payload.get("state")
                code_verifier = cookie_payload.get("code_verifier")
                if (
                    not isinstance(expected_state, str)
                    or not expected_state.strip()
                    or not secrets.compare_digest(expected_state, cast(str, state).strip())
                    or not isinstance(code_verifier, str)
                    or not code_verifier.strip()
                ):
                    resp = _website_auth_error_payload("Invalid authorization state.")
                    _clear_zitadel_web_cookie(resp)
                    return resp

                token_data = {
                    "grant_type": "authorization_code",
                    "client_id": _zitadel_client_id(),
                    "code": cast(str, code).strip(),
                    "code_verifier": code_verifier,
                    "redirect_uri": _website_zitadel_redirect_uri(deps),
                }
                resource = _zitadel_resource()
                audience = _zitadel_audience()
                if resource:
                    token_data["resource"] = resource
                if audience:
                    token_data["audience"] = audience

                token_payload = deps._oidc_fetch_json(_zitadel_token_endpoint(), data=token_data)
                access_token = token_payload.get("access_token")
                if not isinstance(access_token, str) or not access_token.strip():
                    abort(502, description="ZITADEL token response did not include an access token.")
                external_identity = deps._authenticate_external_identity(
                    access_token=access_token.strip(),
                    provider_name=provider_name,
                )
                if _external_identity_verified_email(external_identity) is None:
                    id_token = token_payload.get("id_token")
                    if isinstance(id_token, str) and id_token.strip():
                        id_claims = _decode_zitadel_id_token(id_token.strip())
                        if id_claims is not None:
                            merged_claims = dict(getattr(external_identity, "claims", {}))
                            merged_claims.update(id_claims)
                            scopes = getattr(external_identity, "scopes", frozenset())
                            audiences = getattr(external_identity, "audiences", frozenset())
                            external_identity = ExternalIdentity(
                                issuer=external_identity.issuer,
                                subject=external_identity.subject,
                                scopes=scopes if isinstance(scopes, frozenset) else frozenset(),
                            audiences=audiences if isinstance(audiences, frozenset) else frozenset(),
                            claims=merged_claims,
                        )
                if flow == "mcp_token":
                    user, _ctx = deps._require_verified_user()
                    _link_external_identity_for_user(
                        user=user,
                        provider_name=provider_name,
                        external_identity=external_identity,
                    )
                    resp = _mcp_access_token_response(
                        access_token=access_token.strip(),
                        next_path=next_path,
                        token_payload=token_payload,
                    )
                    _clear_zitadel_web_cookie(resp)
                    return resp
            resp = _complete_website_auth_for_identity(
                external_identity=external_identity,
                next_path=next_path,
                provider_name=provider_name,
                notification_provider="google" if flow == "google_intent" else None,
            )
        except HTTPException as exc:
            deps.db.session.rollback()
            current_app.logger.warning(
                "ZITADEL website auth HTTP error: "
                "code=%s description=%s has_intent=%s data_keys=%s user_id=%s intent_id=%s",
                exc.code,
                exc.description,
                has_intent,
                sorted(data.keys()),
                user_id if isinstance(user_id, str) else None,
                intent_id if isinstance(intent_id, str) else None,
            )
            raise
        except SQLAlchemyError:
            deps.db.session.rollback()
            current_app.logger.exception("ZITADEL website auth completion failed due to auth DB error.")
            abort(503, description="Auth backend is unavailable right now.")
        except Exception as exc:
            deps.db.session.rollback()
            current_app.logger.exception("ZITADEL website auth completion failed.")
            message = str(exc)
            if "Unsupported MCP identity provider" in message:
                abort(400, description="Unsupported external identity provider.")
            if isinstance(exc, McpAuthError):
                if exc.status_code >= 500:
                    abort(503, description="External identity provider is unavailable right now.")
                abort(400, description="External access token is invalid or expired.")
            abort(503, description="External identity provider is unavailable right now.")

        response_payload = resp.get_json(silent=True)
        _clear_zitadel_web_cookie(resp)
        if not (
            isinstance(response_payload, dict)
            and response_payload.get("status") == "legal_required"
        ):
            _clear_zitadel_pending_cookie(resp)
        return resp

    @auth_blp.route("/zitadel/finalize", methods=["POST"])
    def auth_zitadel_finalize():
        deps._require_auth_db()
        if deps._auth_is_mocked():
            abort(501, description="ZITADEL auth is unavailable in mocked auth mode.")

        pending = _load_zitadel_pending_cookie()
        if not pending:
            abort(400, description="No pending auth flow to complete.")

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            abort(400, description="Invalid legal acceptance payload.")
        checked_at = deps._require_legal_acceptance(data)

        user_id = pending.get("user_id")
        next_path = deps._safe_next_path(pending.get("next")) or "/account"
        action = pending.get("action") or "login"
        provider_name = pending.get("provider") or "zitadel"
        if not isinstance(user_id, str) or not user_id.strip():
            resp = _website_auth_error_payload("No pending auth flow to complete.")
            _clear_zitadel_pending_cookie(resp)
            return resp

        try:
            user = deps.db.session.get(deps.AuthUser, user_id)
            if user is None:
                abort(401, description="Pending account no longer exists.")
            deps._ensure_current_legal_acceptances(user_id=user.id, checked_at=checked_at)
            if user.email_verified_at is None:
                if not _send_zitadel_email_verification_for_user(user=user):
                    abort(
                        400,
                        description="Resume account setup from signup so we can send your verification email.",
                    )
                deps.db.session.commit()
                resp = _verification_required_response(user=user, next_path=next_path)
            else:
                deps._record_signon_event(user_id=user.id, provider=provider_name, action=action)
                deps.db.session.commit()
                resp = _auth_success_response(user=user, next_path=next_path)
        except HTTPException:
            deps.db.session.rollback()
            raise
        except SQLAlchemyError:
            deps.db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")

        _clear_zitadel_pending_cookie(resp)
        return resp

    @auth_blp.route("/login/password", methods=["POST"])
    def auth_password_login():
        deps._require_auth_db()
        if deps._auth_is_mocked():
            abort(501, description="Password auth is unavailable in mocked auth mode.")

        data = deps._load_json(deps.AuthPasswordLoginSchema())
        email_raw = data.get("email")
        password = data.get("password")
        next_raw = data.get("next")
        next_path = deps._safe_next_path(next_raw if isinstance(next_raw, str) else None) or "/account"
        if not isinstance(email_raw, str) or not email_raw.strip():
            abort(400, description="Email is required.")
        if not isinstance(password, str) or not password:
            abort(400, description="Password is required.")
        email = deps._normalize_email(email_raw)
        if not deps._is_email_like(email):
            abort(400, description="Enter a valid email address.")

        existing_user = _local_user_by_email(email=email)
        if existing_user is not None:
            existing_link = deps.AuthExternalSubject.query.filter_by(
                user_id=existing_user.id,
                issuer=mcp_oidc_issuer(),
            ).first()
            if existing_link is not None:
                _pending_login_block_response(user=existing_user, next_path=next_path)

        try:
            try:
                external_identity = _zitadel_session_identity(email=email, password=password)
            except HTTPException as exc:
                if exc.code == 400 and exc.description == "Verify your email before signing in.":
                    raise
                if exc.code in {400, 401, 403, 404, 409}:
                    migrated = _maybe_migrate_local_password_user(email=email, password=password)
                    if migrated is None:
                        abort(401, description="Invalid email or password.")
                    external_identity = migrated
                else:
                    raise
            resp = _complete_website_auth_for_identity(
                external_identity=external_identity,
                next_path=next_path,
                provider_name="zitadel",
            )
        except HTTPException:
            deps.db.session.rollback()
            raise
        except SQLAlchemyError:
            deps.db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")
        except Exception:
            deps.db.session.rollback()
            current_app.logger.exception("Custom password login failed.")
            abort(503, description="Remote auth provider is unavailable right now.")
        return resp

    @auth_blp.route("/signup/password", methods=["POST"])
    def auth_password_signup():
        deps._require_auth_db()
        if deps._auth_is_mocked():
            abort(501, description="Signup is unavailable in mocked auth mode.")

        data = deps._load_json(deps.AuthPasswordSignupSchema())
        email_raw = data.get("email")
        password = data.get("password")
        first_name = data.get("first_name")
        last_name = data.get("last_name")
        next_raw = data.get("next")
        next_path = deps._safe_next_path(next_raw if isinstance(next_raw, str) else None) or "/account"
        if not isinstance(email_raw, str) or not email_raw.strip():
            abort(400, description="Email is required.")
        if not isinstance(password, str) or not password:
            abort(400, description="Password is required.")
        if first_name is not None and not isinstance(first_name, str):
            abort(400, description="First name must be a string.")
        if last_name is not None and not isinstance(last_name, str):
            abort(400, description="Last name must be a string.")
        email = deps._normalize_email(email_raw)
        if not deps._is_email_like(email):
            abort(400, description="Enter a valid email address.")

        existing_user = _local_user_by_email(email=email)
        if existing_user is not None:
            has_current_legal = deps._user_has_current_legal_acceptances(user_id=existing_user.id)
            if existing_user.email_verified_at is None:
                remote_subject = _ensure_pending_signup_remote_user(
                    user=existing_user,
                    password=password,
                    first_name=first_name if isinstance(first_name, str) else None,
                    last_name=last_name if isinstance(last_name, str) else None,
                    send_verification=has_current_legal,
                )
                if isinstance(remote_subject, str) and remote_subject.strip():
                    if not has_current_legal:
                        deps.db.session.commit()
                        return _pending_website_auth_response(
                            user=existing_user,
                            next_path=next_path,
                            action="register",
                            provider_name="zitadel",
                        )
                    if _send_zitadel_email_verification_for_user(user=existing_user):
                        deps.db.session.commit()
                        return _verification_required_response(user=existing_user, next_path=next_path)
            resumed = _resume_incomplete_signup_if_eligible(
                user=existing_user,
                next_path=next_path,
                provider_name="zitadel",
            )
            if resumed is not None:
                return resumed
            abort(409, description="An account with that email already exists. Sign in or reset your password.")

        try:
            external_identity = _create_zitadel_human_user(
                email=email,
                password=password,
                first_name=first_name if isinstance(first_name, str) else None,
                last_name=last_name if isinstance(last_name, str) else None,
                email_is_verified=False,
            )
            user = _ensure_pending_signup_user(external_identity=external_identity)
            if deps._user_has_current_legal_acceptances(user_id=user.id):
                _send_zitadel_email_verification_for_user(user=user)
                deps.db.session.commit()
                deps._send_signup_notification_email(
                    new_user_email=user.email,
                    provider="password",
                    full_name=_full_name_from_parts(first_name, last_name),
                    signed_up_at=_user_created_at_or_now(user=user),
                )
                resp = _verification_required_response(user=user, next_path=next_path)
            else:
                deps.db.session.commit()
                deps._send_signup_notification_email(
                    new_user_email=user.email,
                    provider="password",
                    full_name=_full_name_from_parts(first_name, last_name),
                    signed_up_at=_user_created_at_or_now(user=user),
                )
                resp = _pending_website_auth_response(
                    user=user,
                    next_path=next_path,
                    action="register",
                    provider_name="zitadel",
                )
        except HTTPException:
            deps.db.session.rollback()
            raise
        except SQLAlchemyError:
            deps.db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")
        except Exception:
            deps.db.session.rollback()
            current_app.logger.exception("Custom signup failed.")
            abort(503, description="Remote auth provider is unavailable right now.")
        return resp

    @auth_blp.route("/email/verify/confirm", methods=["POST"])
    def auth_email_verify_confirm():
        deps._require_auth_db()
        if deps._auth_is_mocked():
            abort(501, description="Email verification is unavailable in mocked auth mode.")

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            abort(400, description="Invalid email verification payload.")
        user_id = data.get("user_id")
        code = data.get("code")
        next_path = (
            deps._safe_next_path(data.get("next") if isinstance(data.get("next"), str) else None)
            or "/account"
        )
        if not isinstance(user_id, str) or not user_id.strip():
            abort(400, description="User ID is required.")
        if not isinstance(code, str) or not code.strip():
            abort(400, description="Verification code is required.")

        try:
            _zitadel_api_json(
                path=f"/v2/users/{user_id.strip()}/email/verify",
                json_body={"verificationCode": code.strip()},
            )
            external_identity = _zitadel_user_identity(user_id=user_id.strip())
            resp = _complete_website_auth_for_identity(
                external_identity=external_identity,
                next_path=next_path,
                provider_name="zitadel",
            )
        except HTTPException:
            deps.db.session.rollback()
            raise
        except SQLAlchemyError:
            deps.db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")
        except Exception:
            deps.db.session.rollback()
            current_app.logger.exception("Email verification completion failed.")
            abort(503, description="Remote auth provider is unavailable right now.")
        return resp

    @auth_blp.route("/email/verify/resend", methods=["POST"])
    def auth_email_verify_resend():
        deps._require_auth_db()
        if deps._auth_is_mocked():
            abort(501, description="Email verification is unavailable in mocked auth mode.")

        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            abort(400, description="Invalid email verification payload.")
        email_raw = data.get("email")
        if not isinstance(email_raw, str) or not email_raw.strip():
            abort(400, description="Email is required.")
        email = deps._normalize_email(email_raw)
        if not deps._is_email_like(email):
            abort(400, description="Enter a valid email address.")

        user = _local_user_by_email(email=email)
        if user is None:
            abort(404, description="Account not found.")
        if user.email_verified_at is not None:
            abort(400, description="Email is already verified.")

        try:
            if not _send_zitadel_email_verification_for_user(user=user):
                abort(
                    400,
                    description="Resume account setup from signup so we can send your verification email.",
                )
            deps.db.session.commit()
        except HTTPException:
            deps.db.session.rollback()
            raise
        except SQLAlchemyError:
            deps.db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")
        except Exception:
            deps.db.session.rollback()
            current_app.logger.exception("Email verification resend failed.")
            abort(503, description="Remote auth provider is unavailable right now.")

        resp = make_response(
            jsonify({"status": "verification_required", "user": {"id": user.id, "email": user.email}})
        )
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/zitadel/notifications/email", methods=["POST"])
    def auth_zitadel_email_notification():
        raw_body = request.get_data(cache=True)
        signature_header = request.headers.get("Zitadel-Signature")
        if not verify_zitadel_signature(raw_body=raw_body, signature_header=signature_header):
            abort(401, description="Invalid ZITADEL notification signature.")

        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            abort(400, description="Invalid ZITADEL notification payload.")

        parsed = _parse_auth_notification(payload)
        if parsed is None:
            resp = make_response(jsonify({"status": "ignored"}), 202)
            resp.headers["Cache-Control"] = "no-store"
            return resp

        notification_type, recipient, action_url, code = parsed
        try:
            send_pandects_auth_email(
                notification_type=notification_type,
                to_email=recipient,
                action_url=action_url,
                code=code,
            )
        except HTTPException:
            raise
        except Exception:
            current_app.logger.exception("ZITADEL email notification handling failed.")
            abort(503, description="Email delivery is unavailable right now.")

        resp = make_response(jsonify({"status": "sent"}))
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/password-reset/request", methods=["POST"])
    def auth_password_reset_request():
        deps._require_auth_db()
        if deps._auth_is_mocked():
            abort(501, description="Password reset is unavailable in mocked auth mode.")

        data = deps._load_json(deps.AuthPasswordResetRequestSchema())
        email_raw = data.get("email")
        if not isinstance(email_raw, str) or not email_raw.strip():
            abort(400, description="Email is required.")
        email = deps._normalize_email(email_raw)
        if not deps._is_email_like(email):
            abort(400, description="Enter a valid email address.")

        try:
            subject = _linked_zitadel_subject_for_email(email=email, allow_unverified=True)
            if isinstance(subject, str) and subject.strip():
                _zitadel_api_json(
                    path=f"/v2/users/{subject}/password_reset",
                    json_body={
                        "sendLink": {
                            "urlTemplate": (
                                f"{deps._frontend_base_url()}/reset-password/confirm"
                                "?user_id={{.UserID}}&code={{.Code}}&org_id={{.OrgID}}"
                            )
                        }
                    },
                )
        except HTTPException as exc:
            if exc.code not in {400, 401, 403, 404, 409}:
                raise
        except SQLAlchemyError:
            deps.db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")
        except Exception:
            deps.db.session.rollback()
            current_app.logger.exception("Password reset request failed.")
            abort(503, description="Remote auth provider is unavailable right now.")

        resp = deps._status_response("requested")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/password-reset/confirm", methods=["POST"])
    def auth_password_reset_confirm():
        deps._require_auth_db()
        if deps._auth_is_mocked():
            abort(501, description="Password reset is unavailable in mocked auth mode.")

        data = deps._load_json(deps.AuthPasswordResetConfirmSchema())
        user_id = data.get("user_id")
        code = data.get("code")
        password = data.get("password")
        if not isinstance(user_id, str) or not user_id.strip():
            abort(400, description="User ID is required.")
        if not isinstance(code, str) or not code.strip():
            abort(400, description="Reset code is required.")
        if not isinstance(password, str) or not password:
            abort(400, description="New password is required.")

        try:
            _zitadel_api_json(
                path=f"/v2/users/{user_id.strip()}/password",
                json_body={
                    "verificationCode": code.strip(),
                    "newPassword": {
                        "password": password,
                        "changeRequired": False,
                    },
                },
            )
        except HTTPException:
            raise
        except Exception:
            current_app.logger.exception("Password reset confirmation failed.")
            abort(503, description="Remote auth provider is unavailable right now.")

        resp = deps._status_response("updated")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/csrf", methods=["GET"])
    def auth_csrf():
        deps._require_auth_db()
        if deps._auth_session_transport() == "cookie":
            existing = deps._csrf_cookie_value()
            token = existing or secrets.token_urlsafe(32)
            resp = make_response(jsonify({"status": "ok", "csrf_token": token}))
            resp.headers["Cache-Control"] = "no-store"
            if existing is None:
                deps._set_csrf_cookie(resp, token, max_age=60 * 60 * 24 * 14)
            return resp
        resp = deps._status_response("ok")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/health", methods=["GET"])
    def auth_health():
        if deps._auth_is_mocked():
            resp = deps._status_response("ok")
            resp.headers["Cache-Control"] = "no-store"
            return resp
        if not deps._auth_db_is_configured():
            abort(
                503,
                description=(
                    "Auth is not configured (missing AUTH_DATABASE_URI / DATABASE_URL). "
                    "Search is available in limited mode."
                ),
            )
        engine = deps.db.engines.get("auth")
        if engine is None:
            abort(503, description="Auth backend is unavailable right now.")
        try:
            with engine.connect() as conn:
                conn.execute(deps.text("SELECT 1"))
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")
        resp = deps._status_response("ok")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/api-keys", methods=["GET"])
    def auth_list_api_keys():
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        if deps._auth_is_mocked():
            keys = deps._mock_auth.list_api_keys(user_id=user.id)
            resp = make_response(
                jsonify(
                    {
                        "keys": [
                            {
                                "id": k.id,
                                "name": k.name,
                                "prefix": k.prefix,
                                "created_at": k.created_at.isoformat(),
                                "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
                                "revoked_at": k.revoked_at.isoformat() if k.revoked_at else None,
                            }
                            for k in keys
                        ]
                    }
                )
            )
            resp.headers["Cache-Control"] = "no-store"
            return resp
        try:
            keys = (
                deps.ApiKey.query.filter_by(user_id=user.id, deleted_at=None)
                .order_by(deps.ApiKey.created_at.desc())
                .all()
            )
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")
        resp = make_response(
            jsonify(
                {
                    "keys": [
                        {
                            "id": k.id,
                            "name": k.name,
                            "prefix": k.prefix,
                            "created_at": k.created_at.isoformat(),
                            "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
                            "revoked_at": k.revoked_at.isoformat() if k.revoked_at else None,
                        }
                        for k in keys
                    ]
                }
            )
        )
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/api-keys", methods=["POST"])
    def auth_create_api_key():
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        data = deps._load_json(deps.AuthApiKeySchema())
        name = data.get("name")
        if name is not None and not isinstance(name, str):
            abort(400, description="Key name must be a string.")
        if isinstance(name, str):
            name = name.strip() or None
            if name is not None and len(name) > 120:
                abort(400, description="Key name is too long.")
        if deps._auth_is_mocked():
            key, plaintext = deps._mock_auth.create_api_key(user_id=user.id, name=name)
            resp = make_response(
                jsonify(
                    {
                        "api_key": {
                            "id": key.id,
                            "name": key.name,
                            "prefix": key.prefix,
                            "created_at": key.created_at.isoformat(),
                        },
                        "api_key_plaintext": plaintext,
                    }
                )
            )
            resp.headers["Cache-Control"] = "no-store"
            return resp
        try:
            key, plaintext = deps._create_api_key(user_id=user.id, name=name)
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")
        resp = make_response(
            jsonify(
                {
                    "api_key": {
                        "id": key.id,
                        "name": key.name,
                        "prefix": key.prefix,
                        "created_at": key.created_at.isoformat(),
                    },
                    "api_key_plaintext": plaintext,
                }
            )
        )
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/api-keys/<string:key_id>", methods=["DELETE"])
    def auth_revoke_api_key(key_id: str):
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        if not deps._UUID_RE.match(key_id):
            abort(404)
        if deps._auth_is_mocked():
            if not deps._mock_auth.revoke_api_key(user_id=user.id, key_id=key_id):
                abort(404)
            resp = deps._status_response("revoked")
            resp.headers["Cache-Control"] = "no-store"
            return resp
        try:
            key = deps.ApiKey.query.filter_by(id=key_id, user_id=user.id).first()
            if key is None:
                abort(404)
            if key.revoked_at is None:
                key.revoked_at = deps._utc_now()
                deps.db.session.commit()
            resp = deps._status_response("revoked")
            resp.headers["Cache-Control"] = "no-store"
            return resp
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")

    @auth_blp.route("/api-keys/<string:key_id>/permanent", methods=["DELETE"])
    def auth_permanently_delete_api_key(key_id: str):
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        if not deps._UUID_RE.match(key_id):
            abort(404)
        if deps._auth_is_mocked():
            if not deps._mock_auth.permanently_delete_api_key(user_id=user.id, key_id=key_id):
                abort(404)
            resp = deps._status_response("deleted")
            resp.headers["Cache-Control"] = "no-store"
            return resp
        try:
            if not deps._permanently_delete_api_key(user_id=user.id, key_id=key_id):
                abort(404)
            resp = deps._status_response("deleted")
            resp.headers["Cache-Control"] = "no-store"
            return resp
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")

    @auth_blp.route("/usage", methods=["GET"])
    def auth_usage():
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        period = (request.args.get("period") or "1m").strip().lower()
        period_cutoffs = {
            "1w": 6,
            "1m": 29,
            "1y": 364,
            "all": None,
        }
        if period not in period_cutoffs:
            abort(400, description="period must be one of: 1w, 1m, 1y, all.")
        cutoff_days = period_cutoffs[period]
        cutoff = deps._utc_today() - timedelta(days=cutoff_days) if cutoff_days is not None else None
        raw_api_key_id = (request.args.get("api_key_id") or "").strip()
        api_key_id = raw_api_key_id or None
        if api_key_id is not None and not deps._UUID_RE.match(api_key_id):
            abort(400, description="api_key_id must be a UUID.")
        if deps._auth_is_mocked():
            if api_key_id is not None:
                key_ids = {k.id for k in deps._mock_auth.list_api_keys(user_id=user.id)}
                if api_key_id not in key_ids:
                    abort(404)
            by_day, total = deps._mock_auth.usage_for_user(
                user_id=user.id,
                period=period,
                api_key_id=api_key_id,
            )
            resp = make_response(jsonify({"by_day": by_day, "total": total}))
            resp.headers["Cache-Control"] = "no-store"
            return resp
        try:
            if api_key_id is not None:
                key = deps.ApiKey.query.filter_by(
                    user_id=user.id,
                    id=api_key_id,
                    deleted_at=None,
                ).first()
                if key is None:
                    abort(404)
                key_ids = [key.id]
            else:
                keys = deps.ApiKey.query.filter_by(user_id=user.id).all()
                key_ids = [k.id for k in keys]
            if not key_ids:
                return jsonify({"by_day": [], "total": 0})

            rows_query = deps.ApiUsageDaily.query.filter(deps.ApiUsageDaily.api_key_id.in_(key_ids))
            if cutoff is not None:
                rows_query = rows_query.filter(deps.ApiUsageDaily.day >= cutoff)
            rows = rows_query.order_by(deps.ApiUsageDaily.day.asc()).all()
            by_day: dict[str, int] = defaultdict(int)
            total = 0
            for row in rows:
                day_str = row.day.isoformat()
                by_day[day_str] += int(row.count)
                total += int(row.count)

            resp = make_response(
                jsonify(
                    {
                        "by_day": [{"day": day, "count": by_day[day]} for day in sorted(by_day)],
                        "total": total,
                    }
                )
            )
            resp.headers["Cache-Control"] = "no-store"
            return resp
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")

    @auth_blp.route("/account/delete", methods=["POST"])
    def auth_delete_account():
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        data = deps._load_json(deps.AuthDeleteAccountSchema())
        confirm = data.get("confirm")
        if confirm != "Delete":
            abort(400, description='Type "Delete" to confirm.')

        if deps._auth_is_mocked():
            abort(501, description="Account deletion is unavailable in mock auth mode.")

        try:
            now = deps._utc_now()
            _delete_zitadel_user_if_linked(user=user)
            deps.AuthExternalSubject.query.filter_by(user_id=user.id).delete(
                synchronize_session=False
            )
            tombstone = f"deleted+{uuid.uuid4().hex}@deleted.invalid"
            user.email = tombstone
            user.password_hash = None
            deps.AuthSession.query.filter_by(user_id=user.id, revoked_at=None).update(
                {"revoked_at": now}, synchronize_session=False
            )
            deps.ApiKey.query.filter_by(user_id=user.id, revoked_at=None).update(
                {"revoked_at": now}, synchronize_session=False
            )
            deps.db.session.commit()
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")

        resp = deps._status_response("deleted")
        resp.headers["Cache-Control"] = "no-store"
        deps._clear_auth_cookies(resp)
        return resp

    @auth_blp.route("/captcha/site-key", methods=["GET"])
    def auth_captcha_site_key():
        deps._require_auth_db()
        if not deps._turnstile_enabled():
            disabled_payload: dict[str, object] = {"enabled": False}
            if current_app.debug:
                disabled_payload["debug"] = {
                    "turnstile_enabled": os.environ.get("TURNSTILE_ENABLED"),
                    "has_site_key": bool(os.environ.get("TURNSTILE_SITE_KEY", "").strip()),
                    "has_secret_key": bool(os.environ.get("TURNSTILE_SECRET_KEY", "").strip()),
                }
            resp = make_response(jsonify(disabled_payload))
            resp.headers["Cache-Control"] = "no-store"
            return resp
        enabled_payload: dict[str, object] = {
            "enabled": True,
            "site_key": deps._turnstile_site_key(),
        }
        if current_app.debug:
            enabled_payload["debug"] = {
                "turnstile_enabled": os.environ.get("TURNSTILE_ENABLED"),
                "has_site_key": True,
                "has_secret_key": bool(os.environ.get("TURNSTILE_SECRET_KEY", "").strip()),
            }
        resp = make_response(jsonify(enabled_payload))
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/logout", methods=["POST"])
    def auth_logout():
        deps._require_auth_db()
        resp = deps._status_response("ok")
        resp.headers["Cache-Control"] = "no-store"
        token = None
        if deps._auth_session_transport() == "cookie":
            cookie_token = request.cookies.get(deps._SESSION_COOKIE_NAME)
            token = cookie_token.strip() if isinstance(cookie_token, str) else None
        else:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header.removeprefix("Bearer ").strip()
        if token:
            deps._revoke_session_token(token)
        deps._clear_auth_cookies(resp)
        return resp

    @auth_blp.route("/flag-inaccurate", methods=["POST"])
    def auth_flag_inaccurate():
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        data = deps._load_json(deps.AuthFlagInaccurateSchema())
        source_raw = data.get("source")
        source = source_raw.strip() if isinstance(source_raw, str) else ""
        agreement_uuid_raw = data.get("agreement_uuid")
        agreement_uuid = (
            agreement_uuid_raw.strip() if isinstance(agreement_uuid_raw, str) else ""
        )
        raw_message = data.get("message")
        message = raw_message.strip() if isinstance(raw_message, str) else None
        section_uuid = data.get("section_uuid")
        section_uuid = section_uuid.strip() if isinstance(section_uuid, str) else None
        request_follow_up = data.get("request_follow_up")
        if request_follow_up is None:
            request_follow_up = False
        if not isinstance(request_follow_up, bool):
            abort(400, description="request_follow_up must be a boolean.")
        issue_types = data.get("issue_types")
        if issue_types is None:
            issue_types = []
        if not isinstance(issue_types, list) or not all(
            isinstance(item, str) for item in issue_types
        ):
            abort(400, description="issue_types must be a list of strings.")
        issue_types = [item.strip() for item in issue_types if item and item.strip()]
        if not issue_types:
            abort(400, description="issue_types is required.")
        allowed_issue_types = {
            "Incorrect tagging (Article/Section)",
            "Corrupted formatting",
            "Incorrect taxonomy class",
            "Incorrect metadata",
            "Not an M&A agreement",
            "Something else",
        }
        if any(item not in allowed_issue_types for item in issue_types):
            abort(400, description="issue_types contains an invalid value.")
        if source not in ("search_result", "agreement_view"):
            abort(400, description="Invalid source. Use 'search_result' or 'agreement_view'.")
        if not agreement_uuid:
            abort(400, description="agreement_uuid is required.")
        if source == "search_result" and not section_uuid:
            abort(400, description="section_uuid is required when source is 'search_result'.")
        if source == "agreement_view":
            section_uuid = None
        if not current_app.testing and not deps._is_agreement_section_eligible(
            agreement_uuid, section_uuid
        ):
            abort(
                400,
                description="Agreement or section not found or not eligible for flagging.",
            )
        submitted_at = deps._utc_now()
        deps._send_flag_notification_email(
            user_email=user.email,
            submitted_at=submitted_at,
            source=source,
            agreement_uuid=agreement_uuid,
            section_uuid=section_uuid,
            message=message,
            request_follow_up=request_follow_up,
            issue_types=issue_types,
        )
        resp = make_response(jsonify({"status": "ok"}), 200)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    app.register_blueprint(auth_blp)
    return auth_blp
