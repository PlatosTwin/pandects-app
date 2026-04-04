"""Flask blueprint for auth: register, login, password reset, API keys, Google OAuth."""

from __future__ import annotations

import os
import secrets
import uuid
from base64 import urlsafe_b64encode
from hashlib import sha256
from datetime import timedelta
from collections import defaultdict
from typing import Callable
from urllib.parse import urlencode

from flask import Blueprint, Flask, abort, jsonify, make_response, redirect, request, current_app
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.exceptions import HTTPException

from backend.auth.runtime import cookie_settings
from backend.auth.mcp_runtime import McpAuthError, mcp_oidc_audiences, mcp_oidc_issuer, mcp_supported_scopes
from backend.routes.deps import AuthDeps

_ZITADEL_LINK_COOKIE_NAME = "pdcts_zitadel_link"
_ZITADEL_LINK_COOKIE_MAX_AGE = 60 * 10


def _zitadel_link_cookie_serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("AUTH_SECRET_KEY")
    if not secret:
        abort(503, description="Auth is not configured (missing AUTH_SECRET_KEY).")
    return URLSafeTimedSerializer(secret_key=secret, salt="pandects-zitadel-link-cookie")


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


def _zitadel_scopes() -> str:
    raw = os.environ.get("MCP_ZITADEL_SCOPES", "").strip()
    if raw:
        return " ".join(part for part in raw.split() if part)
    return " ".join(("openid", "profile", "email", *mcp_supported_scopes()))


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


def _build_pkce_challenge(code_verifier: str) -> str:
    digest = sha256(code_verifier.encode("utf-8")).digest()
    return urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")


def _set_zitadel_link_cookie(resp, payload: dict[str, str]) -> None:
    value = _zitadel_link_cookie_serializer().dumps(payload)
    samesite, secure = cookie_settings()
    resp.set_cookie(
        _ZITADEL_LINK_COOKIE_NAME,
        value,
        max_age=_ZITADEL_LINK_COOKIE_MAX_AGE,
        httponly=True,
        secure=secure,
        samesite=samesite.capitalize() if samesite != "none" else "None",
        path="/v1/auth/external-subjects/zitadel/complete",
    )


def _load_zitadel_link_cookie() -> dict[str, str] | None:
    raw = request.cookies.get(_ZITADEL_LINK_COOKIE_NAME)
    if not isinstance(raw, str) or not raw.strip():
        return None
    try:
        payload_obj = _zitadel_link_cookie_serializer().loads(
            raw,
            max_age=_ZITADEL_LINK_COOKIE_MAX_AGE,
        )
    except (BadSignature, SignatureExpired):
        return None
    if not isinstance(payload_obj, dict):
        return None
    payload = payload_obj
    if not all(isinstance(value, str) for value in payload.values()):
        return None
    return payload


def _clear_zitadel_link_cookie(resp) -> None:
    samesite, secure = cookie_settings()
    resp.delete_cookie(
        _ZITADEL_LINK_COOKIE_NAME,
        path="/v1/auth/external-subjects/zitadel/complete",
        secure=secure,
        samesite=samesite.capitalize() if samesite != "none" else "None",
    )


def register_auth_routes(app: Flask, *, deps: AuthDeps) -> Blueprint:
    auth_blp = Blueprint("auth", "auth", url_prefix="/v1/auth")

    def _run_email_side_effect(*, label: str, email: str, fn: Callable[[], None]) -> bool:
        try:
            fn()
            return True
        except HTTPException as exc:
            current_app.logger.warning(
                "%s failed for %s (HTTP %s): %s",
                label,
                email,
                getattr(exc, "code", "unknown"),
                getattr(exc, "description", ""),
            )
            return False
        except Exception:
            current_app.logger.exception("%s failed for %s.", label, email)
            return False

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

    @auth_blp.route("/register", methods=["POST"])
    def auth_register():
        deps._require_auth_db()
        data = deps._load_json(deps.AuthRegisterSchema())
        checked_at = deps._require_legal_acceptance(data)
        if deps._turnstile_enabled():
            captcha_token = deps._require_captcha_token(data)
            deps._verify_turnstile_token(token=captcha_token)
        email_raw = data.get("email")
        password = data.get("password")
        if not isinstance(email_raw, str) or not isinstance(password, str):
            abort(400, description="Email and password are required.")
        email = deps._normalize_email(email_raw)
        if not deps._is_email_like(email):
            abort(400, description="Invalid email address.")
        if len(password) < 8:
            abort(400, description="Password must be at least 8 characters.")

        if deps._auth_is_mocked():
            existing = deps._mock_auth.get_user_by_email(email)
            user = existing or deps._mock_auth.create_user(email=email, password=password)
            verify_token = None
            if user.email_verified_at is None:
                verify_token = deps._issue_email_verification_token(
                    user_id=user.id, email=user.email
                )
            payload: dict[str, object] = {
                "status": "verification_required",
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "created_at": user.created_at.isoformat(),
                },
            }
            if (
                verify_token
                and os.environ.get("EMAIL_VERIFICATION_DEBUG_TOKEN", "").strip() == "1"
                and current_app.debug
            ):
                payload["debug_token"] = verify_token
            resp = make_response(jsonify(payload), 201)
            resp.headers["Cache-Control"] = "no-store"
            deps._clear_auth_cookies(resp)
            return resp

        try:
            existing = deps.AuthUser.query.filter_by(email=email).first()
        except SQLAlchemyError:
            deps.db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")

        if existing is not None:
            verify_token = None
            if existing.email_verified_at is None:
                verify_token = deps._issue_email_verification_token(
                    user_id=existing.id, email=existing.email
                )
                def _send_existing_verification_email() -> None:
                    deps._send_email_verification_email(
                        to_email=existing.email, token=verify_token
                    )

                _ = _run_email_side_effect(
                    label="Verification email delivery",
                    email=existing.email,
                    fn=_send_existing_verification_email,
                )
            payload = {
                "status": "verification_required",
                "user": {
                    "id": existing.id,
                    "email": existing.email,
                    "created_at": existing.created_at.isoformat(),
                },
            }
            if (
                verify_token
                and os.environ.get("EMAIL_VERIFICATION_DEBUG_TOKEN", "").strip() == "1"
                and current_app.debug
            ):
                payload["debug_token"] = verify_token
            deps._auth_enumeration_delay()
            resp = make_response(jsonify(payload), 201)
            resp.headers["Cache-Control"] = "no-store"
            deps._clear_auth_cookies(resp)
            return resp

        try:
            now = deps._utc_now()
            ip_address = deps._request_ip_address()
            user_agent = deps._request_user_agent()
            user = deps.AuthUser(
                email=email,
                password_hash=deps.generate_password_hash(password),
                email_verified_at=None,
            )
            deps.db.session.add(user)
            deps.db.session.flush()
            for doc, meta in deps._LEGAL_DOCS.items():
                deps.db.session.add(
                    deps.LegalAcceptance(
                        user_id=user.id,
                        document=doc,
                        version=meta["version"],
                        document_hash=meta["sha256"],
                        checked_at=checked_at,
                        submitted_at=now,
                        ip_address=ip_address,
                        user_agent=user_agent,
                    )
                )
            deps._record_signon_event(
                user_id=user.id, provider="email", action="register"
            )
            deps.db.session.commit()
        except SQLAlchemyError:
            deps.db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")

        verify_token = deps._issue_email_verification_token(
            user_id=user.id, email=user.email
        )
        def _send_new_user_verification_email() -> None:
            deps._send_email_verification_email(to_email=user.email, token=verify_token)

        _ = _run_email_side_effect(
            label="Verification email delivery",
            email=user.email,
            fn=_send_new_user_verification_email,
        )
        def _send_signup_notification() -> None:
            deps._send_signup_notification_email(new_user_email=user.email)

        _ = _run_email_side_effect(
            label="Signup notification delivery",
            email=user.email,
            fn=_send_signup_notification,
        )

        payload = {
            "status": "verification_required",
            "user": {
                "id": user.id,
                "email": user.email,
                "created_at": user.created_at.isoformat(),
            },
        }
        if os.environ.get("EMAIL_VERIFICATION_DEBUG_TOKEN", "").strip() == "1" and current_app.debug:
            payload["debug_token"] = verify_token
        resp = make_response(jsonify(payload), 201)
        resp.headers["Cache-Control"] = "no-store"
        deps._clear_auth_cookies(resp)
        return resp

    @auth_blp.route("/login", methods=["POST"])
    def auth_login():
        deps._require_auth_db()
        data = deps._load_json(deps.AuthLoginSchema())
        email_raw = data.get("email")
        password = data.get("password")
        if not isinstance(email_raw, str) or not isinstance(password, str):
            abort(400, description="Email and password are required.")
        email = deps._normalize_email(email_raw)

        if deps._auth_is_mocked():
            user = deps._mock_auth.authenticate(email=email, password=password)
            if user is None:
                deps._auth_enumeration_delay()
                abort(401, description="Invalid credentials.")
            if user.email_verified_at is None:
                abort(403, description="Email address not verified.")
            token = deps._issue_session_token(user.id)
            payload: dict[str, object] = {"user": {"id": user.id, "email": user.email}}
            if deps._auth_session_transport() == "bearer":
                payload["session_token"] = token
            resp = make_response(jsonify(payload))
            resp.headers["Cache-Control"] = "no-store"
            if deps._auth_session_transport() == "cookie":
                deps._set_auth_cookies(resp, session_token=token)
            return resp

        try:
            user = deps.AuthUser.query.filter_by(email=email).first()
            if user is None or not user.password_hash:
                deps._auth_enumeration_delay()
                abort(401, description="Invalid credentials.")
            if not deps.check_password_hash(user.password_hash, password):
                deps._auth_enumeration_delay()
                abort(401, description="Invalid credentials.")
            if user.email_verified_at is None:
                abort(403, description="Email address not verified.")

            deps._record_signon_event(user_id=user.id, provider="email", action="login")
            deps.db.session.commit()
            token = deps._issue_session_token(user.id)
            payload = {"user": {"id": user.id, "email": user.email}}
            if deps._auth_session_transport() == "bearer":
                payload["session_token"] = token
            resp = make_response(jsonify(payload))
            resp.headers["Cache-Control"] = "no-store"
            if deps._auth_session_transport() == "cookie":
                deps._set_auth_cookies(resp, session_token=token)
            return resp
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")

    @auth_blp.route("/email/resend", methods=["POST"])
    def auth_resend_email_verification():
        deps._require_auth_db()
        data = deps._load_json(deps.AuthEmailSchema())
        email_raw = data.get("email")
        if not isinstance(email_raw, str) or not email_raw.strip():
            abort(400, description="Email is required.")
        email = deps._normalize_email(email_raw)
        if not deps._is_email_like(email):
            abort(400, description="Invalid email address.")

        if deps._auth_is_mocked():
            user = deps._mock_auth.get_user_by_email(email)
            if user is not None and user.email_verified_at is None:
                verify_token = deps._issue_email_verification_token(
                    user_id=user.id, email=user.email
                )
                deps._send_email_verification_email(
                    to_email=user.email, token=verify_token
                )
            deps._auth_enumeration_delay()
            resp = deps._status_response("sent")
            resp.headers["Cache-Control"] = "no-store"
            return resp

        try:
            user = deps.AuthUser.query.filter_by(email=email).first()
        except SQLAlchemyError:
            deps.db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")
        if user is not None and user.email_verified_at is None:
            verify_token = deps._issue_email_verification_token(
                user_id=user.id, email=user.email
            )
            def _send_resend_email() -> None:
                deps._send_email_verification_email(
                    to_email=user.email, token=verify_token
                )

            _ = _run_email_side_effect(
                label="Verification resend delivery",
                email=user.email,
                fn=_send_resend_email,
            )

        deps._auth_enumeration_delay()
        resp = deps._status_response("sent")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/password/forgot", methods=["POST"])
    def auth_password_forgot():
        deps._require_auth_db()
        data = deps._load_json(deps.AuthEmailSchema())
        email_raw = data.get("email")
        if not isinstance(email_raw, str) or not email_raw.strip():
            abort(400, description="Email is required.")
        email = deps._normalize_email(email_raw)
        if not deps._is_email_like(email):
            abort(400, description="Invalid email address.")

        if deps._auth_is_mocked():
            user = deps._mock_auth.get_user_by_email(email)
            if user is not None and not (
                user.email.startswith("deleted+") and user.email.endswith("@deleted.invalid")
            ):
                token = deps._issue_password_reset_token(user_id=user.id, email=user.email)
                deps._send_password_reset_email(to_email=user.email, token=token)
            deps._auth_enumeration_delay()
            resp = deps._status_response("sent")
            resp.headers["Cache-Control"] = "no-store"
            return resp

        try:
            user = deps.AuthUser.query.filter_by(email=email).first()
        except SQLAlchemyError:
            deps.db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")
        if user is not None and not (
            user.email.startswith("deleted+") and user.email.endswith("@deleted.invalid")
        ):
            token = deps._issue_password_reset_token(user_id=user.id, email=user.email)
            def _send_password_reset() -> None:
                deps._send_password_reset_email(to_email=user.email, token=token)

            _ = _run_email_side_effect(
                label="Password reset email delivery",
                email=user.email,
                fn=_send_password_reset,
            )

        deps._auth_enumeration_delay()
        resp = deps._status_response("sent")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/password/reset", methods=["POST"])
    def auth_password_reset():
        deps._require_auth_db()
        data = deps._load_json(deps.AuthPasswordResetSchema())
        token = data.get("token")
        password = data.get("password")
        if not isinstance(token, str) or not token.strip():
            abort(400, description="Missing reset token.")
        if not isinstance(password, str):
            abort(400, description="Password is required.")
        if len(password) < 8:
            abort(400, description="Password must be at least 8 characters.")

        if deps._auth_is_mocked():
            parsed = deps._load_password_reset_token(token.strip())
            if parsed is None:
                abort(400, description="Invalid or expired reset token.")
            user_id, email, _row = parsed
            user = deps._mock_auth.get_user(user_id)
            if user is None or user.email != email:
                abort(400, description="Invalid or expired reset token.")
            if not deps._mock_auth.set_user_password(user_id=user_id, password=password):
                abort(400, description="Invalid or expired reset token.")
            resp = deps._status_response("ok")
            resp.headers["Cache-Control"] = "no-store"
            deps._clear_auth_cookies(resp)
            return resp

        try:
            parsed = deps._load_password_reset_token(token.strip())
            if parsed is None:
                abort(400, description="Invalid or expired reset token.")
            user_id, email, row = parsed
            user = deps.db.session.get(deps.AuthUser, user_id)
            if user is None or user.email != email:
                abort(400, description="Invalid or expired reset token.")
            if user.email.startswith("deleted+") and user.email.endswith("@deleted.invalid"):
                abort(400, description="Invalid or expired reset token.")
            user.password_hash = deps.generate_password_hash(password)
            if user.email_verified_at is None:
                user.email_verified_at = deps._utc_now()
            now = deps._utc_now()
            if row is not None:
                row.used_at = now
            deps.AuthSession.query.filter_by(user_id=user.id, revoked_at=None).update(
                {"revoked_at": now}, synchronize_session=False
            )
            deps.db.session.commit()
        except HTTPException:
            deps.db.session.rollback()
            raise
        except SQLAlchemyError:
            deps.db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")

        resp = deps._status_response("ok")
        resp.headers["Cache-Control"] = "no-store"
        deps._clear_auth_cookies(resp)
        return resp

    @auth_blp.route("/email/verify", methods=["POST"])
    def auth_verify_email():
        deps._require_auth_db()
        data = deps._load_json(deps.AuthTokenSchema())
        token = data.get("token")
        if not isinstance(token, str) or not token.strip():
            abort(400, description="Missing verification token.")
        parsed = deps._load_email_verification_token(token.strip())
        if parsed is None:
            abort(400, description="Invalid or expired verification token.")
        user_id, email = parsed

        if deps._auth_is_mocked():
            user = deps._mock_auth.get_user(user_id)
            if user is None or user.email != email:
                abort(400, description="Invalid verification token.")
            deps._mock_auth.mark_email_verified(user_id)
            resp = deps._status_response("ok")
            resp.headers["Cache-Control"] = "no-store"
            return resp

        try:
            user = deps.db.session.get(deps.AuthUser, user_id)
            if user is None or user.email != email:
                abort(400, description="Invalid verification token.")
            if user.email.startswith("deleted+") and user.email.endswith("@deleted.invalid"):
                abort(400, description="Invalid verification token.")
            if user.email_verified_at is None:
                user.email_verified_at = deps._utc_now()
                deps.db.session.commit()
        except HTTPException:
            deps.db.session.rollback()
            raise
        except SQLAlchemyError:
            deps.db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")

        resp = deps._status_response("ok")
        resp.headers["Cache-Control"] = "no-store"
        return resp

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
            token_payload = deps._google_fetch_json(_zitadel_token_endpoint(), data=token_data)
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
                deps.ApiKey.query.filter_by(user_id=user.id)
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
            keys_query = deps.ApiKey.query.filter_by(user_id=user.id)
            if api_key_id is not None:
                keys_query = keys_query.filter_by(id=api_key_id)
            keys = keys_query.all()
            if api_key_id is not None and not keys:
                abort(404)
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

    @auth_blp.route("/google/start", methods=["GET"])
    def auth_google_start():
        deps._require_auth_db()
        if deps._auth_is_mocked():
            abort(501, description="Google auth is unavailable in mock auth mode.")
        if not deps._google_oauth_flow_enabled():
            abort(404)

        client_id = deps._google_oauth_client_id()
        redirect_uri = deps._google_oauth_redirect_uri()

        next_path = deps._safe_next_path(request.args.get("next")) or "/account"
        state = secrets.token_urlsafe(32)
        code_verifier, code_challenge = deps._google_oauth_pkce_pair()
        nonce = secrets.token_urlsafe(32)
        cookie_payload = {
            "state": state,
            "code_verifier": code_verifier,
            "nonce": nonce,
            "next": next_path,
        }

        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "prompt": "select_account",
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "nonce": nonce,
        }
        auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{deps.urlencode(params)}"
        resp = redirect(auth_url)
        resp.headers["Cache-Control"] = "no-store"
        deps._set_google_oauth_cookie(resp, cookie_payload)
        return resp

    @auth_blp.route("/google/client-id", methods=["GET"])
    def auth_google_client_id():
        deps._require_auth_db()
        if deps._auth_is_mocked():
            abort(501, description="Google auth is unavailable in mock auth mode.")
        nonce = secrets.token_urlsafe(32)
        resp = make_response(jsonify({"client_id": deps._google_oauth_client_id(), "nonce": nonce}))
        resp.headers["Cache-Control"] = "no-store"
        deps._set_google_nonce_cookie(resp, nonce)
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

    @auth_blp.route("/google/callback", methods=["GET"])
    def auth_google_callback():
        deps._require_auth_db()
        if deps._auth_is_mocked():
            abort(501, description="Google auth is unavailable in mock auth mode.")
        if not deps._google_oauth_flow_enabled():
            abort(404)

        error = request.args.get("error")
        if isinstance(error, str) and error.strip():
            return deps._frontend_google_callback_redirect(
                token=None, next_path="/account", error=error
            )

        state = request.args.get("state")
        code = request.args.get("code")
        if not isinstance(state, str) or not state.strip():
            return deps._frontend_google_callback_redirect(
                token=None, next_path="/account", error="missing_state"
            )
        if not isinstance(code, str) or not code.strip():
            return deps._frontend_google_callback_redirect(
                token=None, next_path="/account", error="missing_code"
            )

        cookie_payload = deps._load_google_oauth_cookie()
        if not cookie_payload:
            return deps._frontend_google_callback_redirect(
                token=None, next_path="/account", error="invalid_state"
            )

        expected_state = cookie_payload.get("state")
        if not isinstance(expected_state, str) or not expected_state.strip():
            return deps._frontend_google_callback_redirect(
                token=None, next_path="/account", error="invalid_state"
            )
        if not secrets.compare_digest(expected_state, state):
            return deps._frontend_google_callback_redirect(
                token=None, next_path="/account", error="invalid_state"
            )

        code_verifier = cookie_payload.get("code_verifier")
        nonce = cookie_payload.get("nonce")
        next_path = deps._safe_next_path(cookie_payload.get("next")) if cookie_payload else None
        if not isinstance(code_verifier, str) or not code_verifier.strip():
            return deps._frontend_google_callback_redirect(
                token=None, next_path=next_path or "/account", error="invalid_state"
            )
        if not isinstance(nonce, str) or not nonce.strip():
            return deps._frontend_google_callback_redirect(
                token=None, next_path=next_path or "/account", error="invalid_state"
            )

        token_payload = deps._google_fetch_json(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": deps._google_oauth_client_id(),
                "client_secret": deps._google_oauth_client_secret(),
                "redirect_uri": deps._google_oauth_redirect_uri(),
                "grant_type": "authorization_code",
                "code_verifier": code_verifier,
            },
        )

        id_token = token_payload.get("id_token")
        if not isinstance(id_token, str) or not id_token.strip():
            return deps._frontend_google_callback_redirect(
                token=None, next_path=next_path or "/account", error="missing_id_token"
            )

        try:
            normalized = deps._google_verify_id_token(id_token, expected_nonce=nonce)
        except HTTPException as e:
            return deps._frontend_google_callback_redirect(
                token=None, next_path=next_path or "/account", error=f"google_{e.code}"
            )

        try:
            user = deps.AuthUser.query.filter_by(email=normalized).first()
            if user is None:
                return deps._frontend_google_callback_redirect(
                    token=None, next_path=next_path or "/account", error="legal_required"
                )
            if not deps._user_has_current_legal_acceptances(user_id=user.id):
                return deps._frontend_google_callback_redirect(
                    token=None, next_path=next_path or "/account", error="legal_required"
                )
            if user.email_verified_at is None:
                user.email_verified_at = deps._utc_now()
            deps._record_signon_event(user_id=user.id, provider="google", action="login")
            deps.db.session.commit()
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")

        token = deps._issue_session_token(user.id)
        if deps._auth_session_transport() == "cookie":
            dest = f"{deps._frontend_base_url()}{(next_path or '/account')}"
            resp = redirect(dest)
            resp.headers["Cache-Control"] = "no-store"
            deps._set_auth_cookies(resp, session_token=token)
            deps._clear_google_oauth_cookie(resp)
            return resp
        return deps._frontend_google_callback_redirect(
            token=token, next_path=next_path or "/account", error=None
        )

    @auth_blp.route("/google/credential", methods=["POST"])
    def auth_google_credential():
        deps._require_auth_db()
        if deps._auth_is_mocked():
            abort(501, description="Google auth is unavailable in mock auth mode.")
        data = deps._load_json(deps.AuthGoogleCredentialSchema())
        credential = data.get("credential")
        if not isinstance(credential, str) or not credential.strip():
            abort(400, description="Missing Google credential.")

        expected_nonce = deps._google_nonce_cookie_value()
        if not expected_nonce:
            abort(400, description="Missing Google nonce.")
        normalized = deps._google_verify_id_token(credential, expected_nonce=expected_nonce)

        try:
            user = deps.AuthUser.query.filter_by(email=normalized).first()
            if user is None:
                checked_at = deps._require_legal_acceptance(data)
                now = deps._utc_now()
                ip_address = deps._request_ip_address()
                user_agent = deps._request_user_agent()
                user = deps.AuthUser(
                    email=normalized, password_hash=None, email_verified_at=now
                )
                deps.db.session.add(user)
                deps.db.session.flush()
                for doc, meta in deps._LEGAL_DOCS.items():
                    deps.db.session.add(
                        deps.LegalAcceptance(
                            user_id=user.id,
                            document=doc,
                            version=meta["version"],
                            document_hash=meta["sha256"],
                            checked_at=checked_at,
                            submitted_at=now,
                            ip_address=ip_address,
                            user_agent=user_agent,
                        )
                    )
                deps._record_signon_event(
                    user_id=user.id, provider="google", action="register"
                )
                deps._send_signup_notification_email(new_user_email=user.email)
                deps.db.session.commit()
            elif not deps._user_has_current_legal_acceptances(user_id=user.id):
                checked_at = deps._require_legal_acceptance(data)
                deps._ensure_current_legal_acceptances(
                    user_id=user.id, checked_at=checked_at
                )
                if user.email_verified_at is None:
                    user.email_verified_at = deps._utc_now()
                deps._record_signon_event(user_id=user.id, provider="google", action="login")
                deps.db.session.commit()
            else:
                if user.email_verified_at is None:
                    user.email_verified_at = deps._utc_now()
                deps._record_signon_event(user_id=user.id, provider="google", action="login")
                deps.db.session.commit()
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")

        token = deps._issue_session_token(user.id)
        payload: dict[str, object] = {"user": {"id": user.id, "email": user.email}}
        if deps._auth_session_transport() == "bearer":
            payload["session_token"] = token
        resp = make_response(jsonify(payload))
        resp.headers["Cache-Control"] = "no-store"
        if deps._auth_session_transport() == "cookie":
            deps._set_auth_cookies(resp, session_token=token)
        deps._clear_google_nonce_cookie(resp)
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
