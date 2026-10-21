"""Flask blueprint for auth: register, login, password reset, API keys, Google OAuth."""

from __future__ import annotations

import os
import secrets
import uuid
from hashlib import sha256
from datetime import datetime, timedelta
from functools import partial
from typing import cast
from urllib.parse import urlencode

from flask import Blueprint, Flask, abort, jsonify, make_response, request, current_app
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.exceptions import HTTPException
from werkzeug.security import check_password_hash

from backend.auth import account, identity_links
from backend.auth.account import AccountInputError
from backend.auth.email_runtime import send_pandects_auth_email, verify_zitadel_signature
from backend.auth.session_runtime import load_session_token as _load_session_token
from backend.auth.mcp_runtime import (
    ExternalIdentity,
    McpAuthError,
    mcp_oidc_issuer,
)
from backend.core.config import temporary_access_lockdown_enabled
from backend.routes.auth.cookies import (
    _clear_zitadel_link_cookie,
    _clear_zitadel_pending_cookie,
    _clear_zitadel_web_cookie,
    _load_zitadel_link_cookie,
    _load_zitadel_pending_cookie,
    _load_zitadel_web_cookie,
    _set_zitadel_link_cookie,
    _set_zitadel_pending_cookie,
    _set_zitadel_web_cookie,
)
from backend.routes.auth.mcp_oauth import register_mcp_oauth_routes
from backend.routes.auth.zitadel_api import (
    ZitadelApiClient,
    ZitadelEmailVerificationRequiredError,
)
from backend.routes.auth.zitadel_api import _TOKEN_CACHE
from backend.routes.auth.zitadel_config import (
    _build_pkce_challenge,
    _decode_zitadel_id_token,
    _website_zitadel_redirect_uri,
    _zitadel_audience,
    _zitadel_authorization_endpoint,
    _zitadel_client_id,
    _zitadel_google_idp_id,
    _zitadel_redirect_uri,
    _zitadel_resource,
    _zitadel_scopes,
    _zitadel_token_endpoint,
)
from backend.routes.deps import AuthDeps


# Re-export: tests clear the Zitadel API token cache through this module.
_ZITADEL_API_TOKEN_CACHE = _TOKEN_CACHE

# Password policy lives in backend.auth.account; local aliases keep call sites short.
_PASSWORD_MIN_LENGTH = account.PASSWORD_MIN_LENGTH
_PASSWORD_MAX_LENGTH = account.PASSWORD_MAX_LENGTH
_PASSWORD_LENGTH_MESSAGE = account.PASSWORD_LENGTH_MESSAGE
_PASSWORD_MAX_LENGTH_MESSAGE = account.PASSWORD_MAX_LENGTH_MESSAGE


def register_auth_routes(app: Flask, *, deps: AuthDeps) -> Blueprint:
    auth_blp = Blueprint("auth", "auth", url_prefix="/v1/auth")

    _parse_auth_notification = identity_links.parse_auth_notification
    _external_link_payload = identity_links.external_link_payload
    _full_name_from_parts = identity_links.full_name_from_parts
    _link_external_identity_for_user = partial(identity_links.link_external_identity_for_user, deps)
    _local_user_by_email = partial(identity_links.local_user_by_email, deps)
    _external_identity_verified_email = partial(
        identity_links.external_identity_verified_email, deps
    )
    _resolve_user_from_external_identity = partial(
        identity_links.resolve_user_from_external_identity, deps
    )
    _linked_user_for_subject = partial(identity_links.linked_user_for_subject, deps)

    def _website_auth_error_payload(message: str, *, code: int = 400):
        resp = make_response(jsonify({"message": message, "error": "auth_failed"}), code)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    zitadel_api = ZitadelApiClient(deps)

    def _ensure_zitadel_user_for_local_password_user(*, user, password: str):
        existing_link = deps.AuthExternalSubject.query.filter_by(
            user_id=user.id,
            issuer=mcp_oidc_issuer(),
        ).first()
        if existing_link is not None:
            return zitadel_api.build_external_identity(subject=existing_link.subject, email=user.email)
        return zitadel_api.create_human_user(
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
                migrated = zitadel_api.create_human_user(
                    email=user.email,
                    password=secrets.token_urlsafe(24),
                    email_is_verified=user.email_verified_at is not None,
                )
            except HTTPException as exc:
                description = exc.description if isinstance(exc.description, str) else ""
                if exc.code == 409 and "already exists" in description.lower():
                    existing_user_id = zitadel_api.search_user_id_by_login_name(login_name=user.email)
                    if not isinstance(existing_user_id, str) or not existing_user_id.strip():
                        raise
                    migrated = zitadel_api.build_external_identity(
                        subject=existing_user_id.strip(),
                        email=user.email,
                    )
                else:
                    raise
            _resolve_user_from_external_identity(external_identity=migrated)
            deps.db.session.commit()
            return migrated.subject
        return None

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
        if action == "register" and temporary_access_lockdown_enabled():
            # TEMPORARY (September reopen): new account creation is disabled, so
            # discard the just-created user/link instead of provisioning it.
            deps.db.session.rollback()
            abort(
                403,
                description=(
                    "New user creation is temporarily disabled. Please check back in September."
                ),
            )
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
        # Fully provisioned accounts must authenticate via /login/password.
        # Returning a session here would let any caller take over an account
        # by posting the signup endpoint with the victim's email — the
        # submitted password is never verified on this path.
        return None

    def _zitadel_email_verify_url_template(*, deps: AuthDeps) -> str:
        return (
            f"{deps._frontend_base_url()}/verify-email"
            "?user_id={{.UserID}}&code={{.Code}}&org_id={{.OrgID}}"
        )

    def _ensure_linked_zitadel_subject_for_user(*, user) -> str | None:
        existing_link = deps.AuthExternalSubject.query.filter_by(
            user_id=user.id,
            issuer=mcp_oidc_issuer(),
        ).first()
        if existing_link is not None and isinstance(existing_link.subject, str) and existing_link.subject.strip():
            return existing_link.subject.strip()
        searched_user_id = zitadel_api.search_user_id_by_login_name(login_name=user.email)
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
            zitadel_api.send_email_verification(user_id=subject, url_template=url_template)
            return True
        except HTTPException as exc:
            description = exc.description if isinstance(exc.description, str) else ""
            if exc.code in {400, 404} and "email not found" in description.lower():
                refreshed_subject = zitadel_api.search_user_id_by_login_name(login_name=user.email)
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
                    zitadel_api.send_email_verification(
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
                if zitadel_api.user_email_matches(user_id=subject.strip(), email=user.email):
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
            external_identity = zitadel_api.create_human_user(
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
                existing_user_id = zitadel_api.search_user_id_by_login_name(login_name=user.email)
                if not isinstance(existing_user_id, str) or not existing_user_id.strip():
                    return None
                external_identity = zitadel_api.build_external_identity(
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
        blocked_reason: str | None = None
        if user.email_verified_at is None and not has_legal:
            blocked_reason = "You need to accept the terms and verify your email before signing in."
        elif user.email_verified_at is None:
            blocked_reason = "Verify your email before signing in."
        elif not has_legal:
            blocked_reason = "You need to accept the terms before signing in."
        if blocked_reason is not None:
            # All three abort branches reveal "account exists with a Zitadel
            # link" via a local fast-path. Delay so the timing matches a real
            # Zitadel round-trip. Pay the delay only on the abort path —
            # adding it to the helper's pass-through (verified + legal user)
            # would slow the legitimate login by ~250ms and create a new
            # timing oracle vs. the no-link path.
            deps._auth_enumeration_delay()
            abort(400, description=blocked_reason)
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

    def _collect_zitadel_subjects_for_user(*, user) -> list[str]:
        links = (
            deps.AuthExternalSubject.query.filter_by(user_id=user.id, issuer=mcp_oidc_issuer())
            .order_by(deps.AuthExternalSubject.id.asc())
            .all()
        )
        return [
            link.subject
            for link in links
            if isinstance(link.subject, str) and link.subject.strip()
        ]

    def _delete_zitadel_subjects_best_effort(subjects: list[str], *, user_id: str) -> None:
        # Called AFTER the local tombstone is committed. A failure here
        # leaves a Zitadel orphan but does not roll back the local-side
        # privacy guarantee — losing the local row is the part the user
        # is asking for. Log the orphan so it can be cleaned up out of
        # band rather than aborting and leaving the caller's account in
        # a half-deleted state.
        for subject in subjects:
            try:
                zitadel_api.request(
                    path=f"/v2/users/{subject}",
                    method="DELETE",
                )
            except HTTPException as exc:
                if exc.code == 404:
                    continue
                current_app.logger.warning(
                    "account_delete_zitadel_orphan local_user_id=%s zitadel_subject=%s code=%s",
                    user_id,
                    subject,
                    exc.code,
                )

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
                created = zitadel_api.json(path="/v2/users/human", json_body=create_payload)
                user_id_candidate = created.get("userId") or created.get("user_id")
                if not isinstance(user_id_candidate, str) or not user_id_candidate.strip():
                    abort(502, description="ZITADEL did not return a user identifier for the Google login.")
                zitadel_api.ensure_default_user_grant(user_id=user_id_candidate.strip())
                resolved_user_id = user_id_candidate.strip()
            except HTTPException as exc:
                description = exc.description if isinstance(exc.description, str) else ""
                if exc.code == 409 and "already exists" in description.lower():
                    existing_user_id = zitadel_api.search_user_id_by_login_name(login_name=email.strip())
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

    register_mcp_oauth_routes(
        auth_blp,
        deps=deps,
        ensure_linked_zitadel_subject_for_user=_ensure_linked_zitadel_subject_for_user,
        linked_zitadel_subject_for_email=_linked_zitadel_subject_for_email,
    )

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
            start_payload = zitadel_api.json(
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
                retrieved = zitadel_api.json(
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
        # Cap only (no minimum): legacy accounts may have short passwords,
        # but unbounded input would still hit local hash verification in the
        # migration path.
        if len(password) > _PASSWORD_MAX_LENGTH:
            abort(400, description=_PASSWORD_MAX_LENGTH_MESSAGE)
        email = deps._normalize_email(email_raw)
        if not deps._is_email_like(email):
            abort(400, description="Enter a valid email address.")

        # Per-account throttle layered on top of the per-IP endpoint limit.
        # Prevents credential stuffing where attempts are spread across many
        # source IPs against the same email. Aborts 429 if backed off.
        deps._check_account_login_rate_limit(email)

        existing_user = _local_user_by_email(email=email)
        if existing_user is not None:
            existing_link = deps.AuthExternalSubject.query.filter_by(
                user_id=existing_user.id,
                issuer=mcp_oidc_issuer(),
            ).first()
            if existing_link is not None:
                # Helper aborts with 400 when this account is pending
                # (unverified email / missing legal). The helper internally
                # delays before aborting so the abort-vs-Zitadel timing is
                # indistinguishable. If it returns None we continue to the
                # real Zitadel call without an artificial delay.
                _pending_login_block_response(user=existing_user, next_path=next_path)

        try:
            try:
                external_identity = zitadel_api.session_identity(email=email, password=password)
            except ZitadelEmailVerificationRequiredError:
                # Reveals "account exists but unverified". Equalize timing
                # with the invalid-credentials path below.
                deps._auth_enumeration_delay()
                raise
            except HTTPException as exc:
                if exc.code in {400, 401, 403, 404, 409}:
                    migrated = _maybe_migrate_local_password_user(email=email, password=password)
                    if migrated is None:
                        # "No such user" (Zitadel 404, fast) and "wrong
                        # password" (Zitadel verification, slow) both reach
                        # this abort. Record the failure for the per-account
                        # throttle and add randomized jitter so the response
                        # times are indistinguishable.
                        deps._record_account_login_failure(email)
                        deps._auth_enumeration_delay()
                        abort(401, description="Invalid email or password.")
                    external_identity = migrated
                else:
                    raise
            resp = _complete_website_auth_for_identity(
                external_identity=external_identity,
                next_path=next_path,
                provider_name="zitadel",
            )
            deps._record_account_login_success(email)
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
        # Fail-closed on production deployments where Turnstile keys are
        # missing — better to surface a 503 than to silently lose the captcha
        # gate on a high-abuse endpoint.
        deps._require_turnstile_configured()
        if deps._turnstile_enabled():
            captcha_token = deps._require_captcha_token(data)
            deps._verify_turnstile_token(token=captcha_token)
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
        if not _PASSWORD_MIN_LENGTH <= len(password) <= _PASSWORD_MAX_LENGTH:
            abort(400, description=_PASSWORD_LENGTH_MESSAGE)
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
            # Existing fully-provisioned account. Return the same shape as a
            # fresh signup so an unauthenticated caller can't use this endpoint
            # to enumerate registered emails. The id is a throwaway UUID so the
            # response is shape-indistinguishable from a legitimate new signup.
            # The real user will be told to check their email; if no message
            # arrives, they can sign in or use the password-reset flow.
            # Log the hashed email rather than the raw address — the log
            # line is the enumeration signal the response is supposed to
            # mask, so anyone with log access shouldn't be able to read
            # which addresses already had accounts.
            current_app.logger.info(
                "signup_password_existing_account_masked email_sha256=%s",
                sha256(email.encode("utf-8")).hexdigest()[:16],
            )
            resp = make_response(
                jsonify(
                    {
                        "status": "verification_required",
                        "next_path": next_path,
                        "user": {"id": str(uuid.uuid4()), "email": email},
                    }
                )
            )
            resp.headers["Cache-Control"] = "no-store"
            return resp

        try:
            external_identity = zitadel_api.create_human_user(
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
            zitadel_api.json(
                path=f"/v2/users/{user_id.strip()}/email/verify",
                json_body={"verificationCode": code.strip()},
            )
            external_identity = zitadel_api.user_identity(user_id=user_id.strip())
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

        # Always respond with the same shape regardless of whether the account
        # exists or is already verified. Distinguishing those states would let an
        # unauthenticated caller enumerate registered emails.
        user = _local_user_by_email(email=email)
        if user is not None and user.email_verified_at is None:
            try:
                if _send_zitadel_email_verification_for_user(user=user):
                    deps.db.session.commit()
                else:
                    deps.db.session.rollback()
            except HTTPException:
                deps.db.session.rollback()
            except SQLAlchemyError:
                deps.db.session.rollback()
            except Exception:
                deps.db.session.rollback()
                current_app.logger.exception("Email verification resend failed.")

        resp = make_response(
            jsonify({"status": "verification_required", "user": {"email": email}})
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
        deps._require_turnstile_configured()
        if deps._turnstile_enabled():
            captcha_token = deps._require_captcha_token(data)
            deps._verify_turnstile_token(token=captcha_token)
        email_raw = data.get("email")
        if not isinstance(email_raw, str) or not email_raw.strip():
            abort(400, description="Email is required.")
        email = deps._normalize_email(email_raw)
        if not deps._is_email_like(email):
            abort(400, description="Enter a valid email address.")

        try:
            subject = _linked_zitadel_subject_for_email(email=email, allow_unverified=True)
            if isinstance(subject, str) and subject.strip():
                zitadel_api.json(
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

        # Same-shape response regardless of account existence — pair it with a
        # randomized delay so existing vs. non-existing addresses also can't
        # be distinguished by timing (the Zitadel call only fires for the
        # former).
        deps._auth_enumeration_delay()
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
        if not _PASSWORD_MIN_LENGTH <= len(password) <= _PASSWORD_MAX_LENGTH:
            abort(400, description=_PASSWORD_LENGTH_MESSAGE)

        try:
            zitadel_api.json(
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

    def _no_store_json(payload: object, *, status: int = 200):
        resp = make_response(jsonify(payload), status)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    def _no_store_status(status: str):
        resp = deps._status_response(status)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/api-keys", methods=["GET"])
    def auth_list_api_keys():
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        try:
            keys = account.list_api_keys(deps, user_id=user.id)
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")
        return _no_store_json({"keys": keys})

    @auth_blp.route("/api-keys", methods=["POST"])
    def auth_create_api_key():
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        data = deps._load_json(deps.AuthApiKeySchema())
        try:
            name = account.normalize_api_key_name(data.get("name"))
        except AccountInputError as exc:
            abort(400, description=exc.message)
        if deps._auth_is_mocked():
            key, plaintext = deps._mock_auth.create_api_key(user_id=user.id, name=name)
            return _no_store_json(account.created_api_key_payload(key, plaintext))
        try:
            key, plaintext = deps._create_api_key(user_id=user.id, name=name)
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")
        return _no_store_json(account.created_api_key_payload(key, plaintext))

    @auth_blp.route("/api-keys/<string:key_id>", methods=["DELETE"])
    def auth_revoke_api_key(key_id: str):
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        if not deps._UUID_RE.match(key_id):
            abort(404)
        try:
            revoked = account.revoke_api_key(deps, user_id=user.id, key_id=key_id)
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")
        if not revoked:
            abort(404)
        return _no_store_status("revoked")

    @auth_blp.route("/api-keys/<string:key_id>/permanent", methods=["DELETE"])
    def auth_permanently_delete_api_key(key_id: str):
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        if not deps._UUID_RE.match(key_id):
            abort(404)
        try:
            deleted = account.permanently_delete_api_key(deps, user_id=user.id, key_id=key_id)
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")
        if not deleted:
            abort(404)
        return _no_store_status("deleted")

    @auth_blp.route("/usage", methods=["GET"])
    def auth_usage():
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        period = (request.args.get("period") or "1m").strip().lower()
        if period not in account.USAGE_PERIOD_CUTOFF_DAYS:
            abort(400, description="period must be one of: 1w, 1m, 1y, all.")
        cutoff_days = account.USAGE_PERIOD_CUTOFF_DAYS[period]
        cutoff = deps._utc_today() - timedelta(days=cutoff_days) if cutoff_days is not None else None
        raw_api_key_id = (request.args.get("api_key_id") or "").strip()
        api_key_id = raw_api_key_id or None
        if api_key_id is not None and not deps._UUID_RE.match(api_key_id):
            abort(400, description="api_key_id must be a UUID.")
        try:
            usage = account.usage_by_day(
                deps,
                user_id=user.id,
                period=period,
                cutoff=cutoff,
                api_key_id=api_key_id,
            )
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")
        if usage is None:
            abort(404)
        return _no_store_json(usage)

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

        # Capture the Zitadel subjects BEFORE the local commit. We delete
        # them from Zitadel only after the local tombstone has committed —
        # otherwise a failed local commit would leave the Zitadel account
        # gone but the local row active, producing an "I'm deleted but
        # can't sign in" state that's impossible to recover from on the
        # next attempt.
        try:
            zitadel_subjects = _collect_zitadel_subjects_for_user(user=user)
            local_user_id = user.id
            now = deps._utc_now()
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

        _delete_zitadel_subjects_best_effort(zitadel_subjects, user_id=local_user_id)

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
            # Resolve the user before revocation so session duration can be
            # measured from login → logout in auth_signon_events.
            user_id = _load_session_token(token)
            if isinstance(user_id, str) and user_id:
                deps._record_signon_event(
                    user_id=user_id, provider="session", action="logout"
                )
            deps._revoke_session_token(token)
        deps._clear_auth_cookies(resp)
        return resp

    @auth_blp.route("/flag-inaccurate", methods=["POST"])
    def auth_flag_inaccurate():
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        data = deps._load_json(deps.AuthFlagInaccurateSchema())
        try:
            report = account.parse_flag_report(data)
        except AccountInputError as exc:
            abort(400, description=exc.message)
        if not current_app.testing and not deps._is_agreement_section_eligible(
            report.agreement_uuid, report.section_uuid
        ):
            abort(
                400,
                description="Agreement or section not found or not eligible for flagging.",
            )
        deps._send_flag_notification_email(
            user_email=user.email,
            submitted_at=deps._utc_now(),
            source=report.source,
            agreement_uuid=report.agreement_uuid,
            section_uuid=report.section_uuid,
            message=report.message,
            request_follow_up=report.request_follow_up,
            issue_types=report.issue_types,
        )
        return _no_store_json({"status": "ok"})

    app.register_blueprint(auth_blp)
    return auth_blp
