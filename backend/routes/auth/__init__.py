"""Flask blueprint for auth: register, login, password reset, API keys, Google OAuth."""

from __future__ import annotations

import os
import secrets
from datetime import timedelta
from functools import partial
from typing import cast
from urllib.parse import urlencode

from flask import Blueprint, Flask, abort, jsonify, make_response, request, current_app
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.exceptions import HTTPException

from backend.auth import account, identity_links
from backend.auth.account import AccountInputError
from backend.auth.email_runtime import send_pandects_auth_email, verify_zitadel_signature
from backend.auth.session_runtime import load_session_token as _load_session_token
from backend.auth.mcp_runtime import McpAuthError
from backend.routes.auth.cookies import (
    _clear_zitadel_link_cookie,
    _clear_zitadel_pending_cookie,
    _clear_zitadel_web_cookie,
    _load_zitadel_link_cookie,
    _load_zitadel_pending_cookie,
    _load_zitadel_web_cookie,
    _set_zitadel_link_cookie,
    _set_zitadel_web_cookie,
)
from backend.routes.auth.mcp_oauth import register_mcp_oauth_routes
from backend.routes.auth.website_auth import WebsiteAuthService, website_auth_error_response
from backend.routes.auth.zitadel_api import ZitadelApiClient
from backend.routes.auth.zitadel_api import _TOKEN_CACHE
from backend.routes.auth.zitadel_config import (
    _build_pkce_challenge,
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
    _link_external_identity_for_user = partial(identity_links.link_external_identity_for_user, deps)

    zitadel_api = ZitadelApiClient(deps)
    website_auth = WebsiteAuthService(deps, zitadel_api)

    register_mcp_oauth_routes(
        auth_blp,
        deps=deps,
        ensure_linked_zitadel_subject_for_user=website_auth.ensure_linked_zitadel_subject_for_user,
        linked_zitadel_subject_for_email=website_auth.linked_zitadel_subject_for_email,
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
            return website_auth_error_response("Invalid authorization state.")
        next_path = deps._safe_next_path(cookie_payload.get("next")) or "/account"
        provider_name = "zitadel"
        flow = cookie_payload.get("flow")

        try:
            if has_intent:
                if flow != "google_intent":
                    resp = website_auth_error_response("Invalid authorization state.")
                    _clear_zitadel_web_cookie(resp)
                    return resp
                retrieved = zitadel_api.json(
                    path=f"/v2/idp_intents/{cast(str, intent_id).strip()}",
                    json_body={"idpIntentToken": cast(str, intent_token).strip()},
                )
                external_identity = website_auth.google_identity_from_intent(
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
                    resp = website_auth_error_response("Invalid authorization state.")
                    _clear_zitadel_web_cookie(resp)
                    return resp

                external_identity, access_token, token_payload = (
                    website_auth.identity_from_authorization_code(
                        code=cast(str, code).strip(),
                        code_verifier=code_verifier,
                        provider_name=provider_name,
                    )
                )
                if flow == "mcp_token":
                    user, _ctx = deps._require_verified_user()
                    _link_external_identity_for_user(
                        user=user,
                        provider_name=provider_name,
                        external_identity=external_identity,
                    )
                    resp = website_auth.mcp_access_token_response(
                        access_token=access_token,
                        next_path=next_path,
                        token_payload=token_payload,
                    )
                    _clear_zitadel_web_cookie(resp)
                    return resp
            resp = website_auth.complete_website_auth_for_identity(
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
            resp = website_auth_error_response("No pending auth flow to complete.")
            _clear_zitadel_pending_cookie(resp)
            return resp

        try:
            user = deps.db.session.get(deps.AuthUser, user_id)
            if user is None:
                abort(401, description="Pending account no longer exists.")
            deps._ensure_current_legal_acceptances(user_id=user.id, checked_at=checked_at)
            if user.email_verified_at is None:
                if not website_auth.send_email_verification_for_user(user=user):
                    abort(
                        400,
                        description="Resume account setup from signup so we can send your verification email.",
                    )
                deps.db.session.commit()
                resp = website_auth.verification_required_response(user=user, next_path=next_path)
            else:
                deps._record_signon_event(user_id=user.id, provider=provider_name, action=action)
                deps.db.session.commit()
                resp = website_auth.auth_success_response(user=user, next_path=next_path)
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

        return website_auth.password_login(email=email, password=password, next_path=next_path)

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

        return website_auth.password_signup(
            email=email,
            password=password,
            first_name=first_name if isinstance(first_name, str) else None,
            last_name=last_name if isinstance(last_name, str) else None,
            next_path=next_path,
        )

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

        return website_auth.confirm_email_verification(
            user_id=user_id.strip(), code=code.strip(), next_path=next_path
        )

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
        website_auth.resend_email_verification(email=email)

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
            subject = website_auth.linked_zitadel_subject_for_email(email=email, allow_unverified=True)
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

        website_auth.delete_account(user=user)

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
