from __future__ import annotations

import os
import secrets
import uuid
from datetime import datetime, timedelta
from collections import defaultdict

from flask import Blueprint, abort, jsonify, make_response, redirect, request, current_app
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.exceptions import HTTPException


def register_auth_routes(app, *, app_module) -> Blueprint:
    auth_blp = Blueprint("auth", "auth", url_prefix="/api/auth")

    @auth_blp.route("/register", methods=["POST"])
    def auth_register():
        app_module._require_auth_db()
        data = app_module._load_json(app_module.AuthRegisterSchema())
        checked_at = app_module._require_legal_acceptance(data)
        if app_module._turnstile_enabled():
            captcha_token = app_module._require_captcha_token(data)
            app_module._verify_turnstile_token(token=captcha_token)
        email_raw = data.get("email")
        password = data.get("password")
        if not isinstance(email_raw, str) or not isinstance(password, str):
            abort(400, description="Email and password are required.")
        email = app_module._normalize_email(email_raw)
        if not app_module._is_email_like(email):
            abort(400, description="Invalid email address.")
        if len(password) < 8:
            abort(400, description="Password must be at least 8 characters.")

        if app_module._auth_is_mocked():
            existing = app_module._mock_auth.get_user_by_email(email)
            user = existing or app_module._mock_auth.create_user(email=email, password=password)
            verify_token = None
            if user.email_verified_at is None:
                verify_token = app_module._issue_email_verification_token(
                    user_id=user.id, email=user.email
                )
            payload: dict[str, object] = {
                "status": "verification_required",
                "user": {
                    "id": user.id,
                    "email": user.email,
                    "createdAt": user.created_at.isoformat(),
                },
            }
            if (
                verify_token
                and os.environ.get("EMAIL_VERIFICATION_DEBUG_TOKEN", "").strip() == "1"
                and current_app.debug
            ):
                payload["debugToken"] = verify_token
            resp = make_response(jsonify(payload), 201)
            resp.headers["Cache-Control"] = "no-store"
            app_module._clear_auth_cookies(resp)
            return resp

        try:
            existing = app_module.AuthUser.query.filter_by(email=email).first()
            if existing is not None:
                verify_token = None
                if existing.email_verified_at is None:
                    verify_token = app_module._issue_email_verification_token(
                        user_id=existing.id, email=existing.email
                    )
                    app_module._send_email_verification_email(
                        to_email=existing.email, token=verify_token
                    )
                payload = {
                    "status": "verification_required",
                    "user": {
                        "id": existing.id,
                        "email": existing.email,
                        "createdAt": existing.created_at.isoformat(),
                    },
                }
                if (
                    verify_token
                    and os.environ.get("EMAIL_VERIFICATION_DEBUG_TOKEN", "").strip() == "1"
                    and current_app.debug
                ):
                    payload["debugToken"] = verify_token
                app_module._auth_enumeration_delay()
                resp = make_response(jsonify(payload), 201)
                resp.headers["Cache-Control"] = "no-store"
                app_module._clear_auth_cookies(resp)
                return resp

            now = datetime.utcnow()
            ip_address = app_module._request_ip_address()
            user_agent = app_module._request_user_agent()
            user = app_module.AuthUser(
                email=email,
                password_hash=app_module.generate_password_hash(password),
                email_verified_at=None,
            )
            app_module.db.session.add(user)
            app_module.db.session.flush()
            for doc, meta in app_module._LEGAL_DOCS.items():
                app_module.db.session.add(
                    app_module.LegalAcceptance(
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
            app_module._record_signon_event(
                user_id=user.id, provider="email", action="register"
            )
            verify_token = app_module._issue_email_verification_token(
                user_id=user.id, email=user.email
            )
            app_module._send_email_verification_email(to_email=user.email, token=verify_token)
            app_module._send_signup_notification_email(new_user_email=user.email)
            app_module.db.session.commit()
        except HTTPException:
            app_module.db.session.rollback()
            raise
        except SQLAlchemyError:
            app_module.db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")

        payload = {
            "status": "verification_required",
            "user": {
                "id": user.id,
                "email": user.email,
                "createdAt": user.created_at.isoformat(),
            },
        }
        if os.environ.get("EMAIL_VERIFICATION_DEBUG_TOKEN", "").strip() == "1" and current_app.debug:
            payload["debugToken"] = verify_token
        resp = make_response(jsonify(payload), 201)
        resp.headers["Cache-Control"] = "no-store"
        app_module._clear_auth_cookies(resp)
        return resp

    @auth_blp.route("/login", methods=["POST"])
    def auth_login():
        app_module._require_auth_db()
        data = app_module._load_json(app_module.AuthLoginSchema())
        email_raw = data.get("email")
        password = data.get("password")
        if not isinstance(email_raw, str) or not isinstance(password, str):
            abort(400, description="Email and password are required.")
        email = app_module._normalize_email(email_raw)

        if app_module._auth_is_mocked():
            user = app_module._mock_auth.authenticate(email=email, password=password)
            if user is None:
                app_module._auth_enumeration_delay()
                abort(401, description="Invalid credentials.")
            if user.email_verified_at is None:
                abort(403, description="Email address not verified.")
            token = app_module._issue_session_token(user.id)
            payload: dict[str, object] = {"user": {"id": user.id, "email": user.email}}
            if app_module._auth_session_transport() == "bearer":
                payload["sessionToken"] = token
            resp = make_response(jsonify(payload))
            resp.headers["Cache-Control"] = "no-store"
            if app_module._auth_session_transport() == "cookie":
                app_module._set_auth_cookies(resp, session_token=token)
            return resp

        try:
            user = app_module.AuthUser.query.filter_by(email=email).first()
            if user is None or not user.password_hash:
                app_module._auth_enumeration_delay()
                abort(401, description="Invalid credentials.")
            if not app_module.check_password_hash(user.password_hash, password):
                app_module._auth_enumeration_delay()
                abort(401, description="Invalid credentials.")
            if user.email_verified_at is None:
                abort(403, description="Email address not verified.")

            app_module._record_signon_event(user_id=user.id, provider="email", action="login")
            app_module.db.session.commit()
            token = app_module._issue_session_token(user.id)
            payload = {"user": {"id": user.id, "email": user.email}}
            if app_module._auth_session_transport() == "bearer":
                payload["sessionToken"] = token
            resp = make_response(jsonify(payload))
            resp.headers["Cache-Control"] = "no-store"
            if app_module._auth_session_transport() == "cookie":
                app_module._set_auth_cookies(resp, session_token=token)
            return resp
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")

    @auth_blp.route("/email/resend", methods=["POST"])
    def auth_resend_email_verification():
        app_module._require_auth_db()
        data = app_module._load_json(app_module.AuthEmailSchema())
        email_raw = data.get("email")
        if not isinstance(email_raw, str) or not email_raw.strip():
            abort(400, description="Email is required.")
        email = app_module._normalize_email(email_raw)
        if not app_module._is_email_like(email):
            abort(400, description="Invalid email address.")

        if app_module._auth_is_mocked():
            user = app_module._mock_auth.get_user_by_email(email)
            if user is not None and user.email_verified_at is None:
                verify_token = app_module._issue_email_verification_token(
                    user_id=user.id, email=user.email
                )
                app_module._send_email_verification_email(
                    to_email=user.email, token=verify_token
                )
            app_module._auth_enumeration_delay()
            resp = app_module._status_response("sent")
            resp.headers["Cache-Control"] = "no-store"
            return resp

        try:
            user = app_module.AuthUser.query.filter_by(email=email).first()
            if user is not None and user.email_verified_at is None:
                verify_token = app_module._issue_email_verification_token(
                    user_id=user.id, email=user.email
                )
                app_module._send_email_verification_email(
                    to_email=user.email, token=verify_token
                )
        except HTTPException:
            app_module.db.session.rollback()
            raise
        except SQLAlchemyError:
            app_module.db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")

        app_module._auth_enumeration_delay()
        resp = app_module._status_response("sent")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/password/forgot", methods=["POST"])
    def auth_password_forgot():
        app_module._require_auth_db()
        data = app_module._load_json(app_module.AuthEmailSchema())
        email_raw = data.get("email")
        if not isinstance(email_raw, str) or not email_raw.strip():
            abort(400, description="Email is required.")
        email = app_module._normalize_email(email_raw)
        if not app_module._is_email_like(email):
            abort(400, description="Invalid email address.")

        if app_module._auth_is_mocked():
            user = app_module._mock_auth.get_user_by_email(email)
            if user is not None and not (
                user.email.startswith("deleted+") and user.email.endswith("@deleted.invalid")
            ):
                token = app_module._issue_password_reset_token(user_id=user.id, email=user.email)
                app_module._send_password_reset_email(to_email=user.email, token=token)
            app_module._auth_enumeration_delay()
            resp = app_module._status_response("sent")
            resp.headers["Cache-Control"] = "no-store"
            return resp

        try:
            user = app_module.AuthUser.query.filter_by(email=email).first()
            if user is not None and not (
                user.email.startswith("deleted+") and user.email.endswith("@deleted.invalid")
            ):
                token = app_module._issue_password_reset_token(user_id=user.id, email=user.email)
                app_module._send_password_reset_email(to_email=user.email, token=token)
        except HTTPException:
            app_module.db.session.rollback()
            raise
        except SQLAlchemyError:
            app_module.db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")

        app_module._auth_enumeration_delay()
        resp = app_module._status_response("sent")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/password/reset", methods=["POST"])
    def auth_password_reset():
        app_module._require_auth_db()
        data = app_module._load_json(app_module.AuthPasswordResetSchema())
        token = data.get("token")
        password = data.get("password")
        if not isinstance(token, str) or not token.strip():
            abort(400, description="Missing reset token.")
        if not isinstance(password, str):
            abort(400, description="Password is required.")
        if len(password) < 8:
            abort(400, description="Password must be at least 8 characters.")

        if app_module._auth_is_mocked():
            parsed = app_module._load_password_reset_token(token.strip())
            if parsed is None:
                abort(400, description="Invalid or expired reset token.")
            user_id, email, _row = parsed
            user = app_module._mock_auth.get_user(user_id)
            if user is None or user.email != email:
                abort(400, description="Invalid or expired reset token.")
            if not app_module._mock_auth.set_user_password(user_id=user_id, password=password):
                abort(400, description="Invalid or expired reset token.")
            resp = app_module._status_response("ok")
            resp.headers["Cache-Control"] = "no-store"
            app_module._clear_auth_cookies(resp)
            return resp

        try:
            parsed = app_module._load_password_reset_token(token.strip())
            if parsed is None:
                abort(400, description="Invalid or expired reset token.")
            user_id, email, row = parsed
            user = app_module.db.session.get(app_module.AuthUser, user_id)
            if user is None or user.email != email:
                abort(400, description="Invalid or expired reset token.")
            if user.email.startswith("deleted+") and user.email.endswith("@deleted.invalid"):
                abort(400, description="Invalid or expired reset token.")
            user.password_hash = app_module.generate_password_hash(password)
            if user.email_verified_at is None:
                user.email_verified_at = datetime.utcnow()
            now = datetime.utcnow()
            if row is not None:
                row.used_at = now
            app_module.AuthSession.query.filter_by(user_id=user.id, revoked_at=None).update(
                {"revoked_at": now}, synchronize_session=False
            )
            app_module.db.session.commit()
        except HTTPException:
            app_module.db.session.rollback()
            raise
        except SQLAlchemyError:
            app_module.db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")

        resp = app_module._status_response("ok")
        resp.headers["Cache-Control"] = "no-store"
        app_module._clear_auth_cookies(resp)
        return resp

    @auth_blp.route("/email/verify", methods=["POST"])
    def auth_verify_email():
        app_module._require_auth_db()
        data = app_module._load_json(app_module.AuthTokenSchema())
        token = data.get("token")
        if not isinstance(token, str) or not token.strip():
            abort(400, description="Missing verification token.")
        parsed = app_module._load_email_verification_token(token.strip())
        if parsed is None:
            abort(400, description="Invalid or expired verification token.")
        user_id, email = parsed

        if app_module._auth_is_mocked():
            user = app_module._mock_auth.get_user(user_id)
            if user is None or user.email != email:
                abort(400, description="Invalid verification token.")
            app_module._mock_auth.mark_email_verified(user_id)
            resp = app_module._status_response("ok")
            resp.headers["Cache-Control"] = "no-store"
            return resp

        try:
            user = app_module.db.session.get(app_module.AuthUser, user_id)
            if user is None or user.email != email:
                abort(400, description="Invalid verification token.")
            if user.email.startswith("deleted+") and user.email.endswith("@deleted.invalid"):
                abort(400, description="Invalid verification token.")
            if user.email_verified_at is None:
                user.email_verified_at = datetime.utcnow()
                app_module.db.session.commit()
        except HTTPException:
            app_module.db.session.rollback()
            raise
        except SQLAlchemyError:
            app_module.db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")

        resp = app_module._status_response("ok")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/me", methods=["GET"])
    def auth_me():
        app_module._require_auth_db()
        user, _ctx = app_module._require_verified_user()
        resp = make_response(jsonify({"user": {"id": user.id, "email": user.email}}))
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/csrf", methods=["GET"])
    def auth_csrf():
        app_module._require_auth_db()
        if app_module._auth_session_transport() == "cookie":
            existing = app_module._csrf_cookie_value()
            token = existing or secrets.token_urlsafe(32)
            resp = make_response(jsonify({"status": "ok", "csrfToken": token}))
            resp.headers["Cache-Control"] = "no-store"
            if existing is None:
                app_module._set_csrf_cookie(resp, token, max_age=60 * 60 * 24 * 14)
            return resp
        resp = app_module._status_response("ok")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/health", methods=["GET"])
    def auth_health():
        if app_module._auth_is_mocked():
            resp = app_module._status_response("ok")
            resp.headers["Cache-Control"] = "no-store"
            return resp
        if not app_module._auth_db_is_configured():
            abort(
                503,
                description=(
                    "Auth is not configured (missing AUTH_DATABASE_URI / DATABASE_URL). "
                    "Search is available in limited mode."
                ),
            )
        engine = app_module.db.engines.get("auth")
        if engine is None:
            abort(503, description="Auth backend is unavailable right now.")
        try:
            with engine.connect() as conn:
                conn.execute(app_module.text("SELECT 1"))
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")
        resp = app_module._status_response("ok")
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/api-keys", methods=["GET"])
    def auth_list_api_keys():
        app_module._require_auth_db()
        user, _ctx = app_module._require_verified_user()
        if app_module._auth_is_mocked():
            keys = app_module._mock_auth.list_api_keys(user_id=user.id)
            resp = make_response(
                jsonify(
                    {
                        "keys": [
                            {
                                "id": k.id,
                                "name": k.name,
                                "prefix": k.prefix,
                                "createdAt": k.created_at.isoformat(),
                                "lastUsedAt": k.last_used_at.isoformat() if k.last_used_at else None,
                                "revokedAt": k.revoked_at.isoformat() if k.revoked_at else None,
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
                app_module.ApiKey.query.filter_by(user_id=user.id)
                .order_by(app_module.ApiKey.created_at.desc())
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
                            "createdAt": k.created_at.isoformat(),
                            "lastUsedAt": k.last_used_at.isoformat() if k.last_used_at else None,
                            "revokedAt": k.revoked_at.isoformat() if k.revoked_at else None,
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
        app_module._require_auth_db()
        user, _ctx = app_module._require_verified_user()
        data = app_module._load_json(app_module.AuthApiKeySchema())
        name = data.get("name")
        if name is not None and not isinstance(name, str):
            abort(400, description="Key name must be a string.")
        if isinstance(name, str):
            name = name.strip() or None
            if name is not None and len(name) > 120:
                abort(400, description="Key name is too long.")
        if app_module._auth_is_mocked():
            key, plaintext = app_module._mock_auth.create_api_key(user_id=user.id, name=name)
            resp = make_response(
                jsonify(
                    {
                        "apiKey": {
                            "id": key.id,
                            "name": key.name,
                            "prefix": key.prefix,
                            "createdAt": key.created_at.isoformat(),
                        },
                        "apiKeyPlaintext": plaintext,
                    }
                )
            )
            resp.headers["Cache-Control"] = "no-store"
            return resp
        try:
            key, plaintext = app_module._create_api_key(user_id=user.id, name=name)
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")
        resp = make_response(
            jsonify(
                {
                    "apiKey": {
                        "id": key.id,
                        "name": key.name,
                        "prefix": key.prefix,
                        "createdAt": key.created_at.isoformat(),
                    },
                    "apiKeyPlaintext": plaintext,
                }
            )
        )
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/api-keys/<string:key_id>", methods=["DELETE"])
    def auth_revoke_api_key(key_id: str):
        app_module._require_auth_db()
        user, _ctx = app_module._require_verified_user()
        if not app_module._UUID_RE.match(key_id):
            abort(404)
        if app_module._auth_is_mocked():
            if not app_module._mock_auth.revoke_api_key(user_id=user.id, key_id=key_id):
                abort(404)
            resp = app_module._status_response("revoked")
            resp.headers["Cache-Control"] = "no-store"
            return resp
        try:
            key = app_module.ApiKey.query.filter_by(id=key_id, user_id=user.id).first()
            if key is None:
                abort(404)
            if key.revoked_at is None:
                key.revoked_at = datetime.utcnow()
                app_module.db.session.commit()
            resp = app_module._status_response("revoked")
            resp.headers["Cache-Control"] = "no-store"
            return resp
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")

    @auth_blp.route("/usage", methods=["GET"])
    def auth_usage():
        app_module._require_auth_db()
        user, _ctx = app_module._require_verified_user()
        if app_module._auth_is_mocked():
            by_day, total = app_module._mock_auth.usage_for_user(user_id=user.id)
            resp = make_response(jsonify({"byDay": by_day, "total": total}))
            resp.headers["Cache-Control"] = "no-store"
            return resp
        cutoff = app_module._utc_today() - timedelta(days=29)
        try:
            key_ids = [k.id for k in app_module.ApiKey.query.filter_by(user_id=user.id).all()]
            if not key_ids:
                return jsonify({"byDay": [], "total": 0})

            rows = (
                app_module.ApiUsageDaily.query.filter(app_module.ApiUsageDaily.api_key_id.in_(key_ids))
                .filter(app_module.ApiUsageDaily.day >= cutoff)
                .order_by(app_module.ApiUsageDaily.day.asc())
                .all()
            )
            by_day: dict[str, int] = defaultdict(int)
            total = 0
            for row in rows:
                day_str = row.day.isoformat()
                by_day[day_str] += int(row.count)
                total += int(row.count)

            resp = make_response(
                jsonify(
                    {
                        "byDay": [{"day": day, "count": by_day[day]} for day in sorted(by_day)],
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
        app_module._require_auth_db()
        user, _ctx = app_module._require_verified_user()
        data = app_module._load_json(app_module.AuthDeleteAccountSchema())
        confirm = data.get("confirm")
        if confirm != "Delete":
            abort(400, description='Type "Delete" to confirm.')

        if app_module._auth_is_mocked():
            abort(501, description="Account deletion is unavailable in mock auth mode.")

        try:
            now = datetime.utcnow()
            tombstone = f"deleted+{uuid.uuid4().hex}@deleted.invalid"
            user.email = tombstone
            user.password_hash = None
            app_module.AuthSession.query.filter_by(user_id=user.id, revoked_at=None).update(
                {"revoked_at": now}, synchronize_session=False
            )
            app_module.ApiKey.query.filter_by(user_id=user.id, revoked_at=None).update(
                {"revoked_at": now}, synchronize_session=False
            )
            app_module.db.session.commit()
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")

        resp = app_module._status_response("deleted")
        resp.headers["Cache-Control"] = "no-store"
        app_module._clear_auth_cookies(resp)
        return resp

    @auth_blp.route("/google/start", methods=["GET"])
    def auth_google_start():
        app_module._require_auth_db()
        if app_module._auth_is_mocked():
            abort(501, description="Google auth is unavailable in mock auth mode.")
        if not app_module._google_oauth_flow_enabled():
            abort(404)

        client_id = app_module._google_oauth_client_id()
        redirect_uri = app_module._google_oauth_redirect_uri()

        next_path = app_module._safe_next_path(request.args.get("next")) or "/account"
        state = secrets.token_urlsafe(32)
        code_verifier, code_challenge = app_module._google_oauth_pkce_pair()
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
        auth_url = f"https://accounts.google.com/o/oauth2/v2/auth?{app_module.urlencode(params)}"
        resp = redirect(auth_url)
        resp.headers["Cache-Control"] = "no-store"
        app_module._set_google_oauth_cookie(resp, cookie_payload)
        return resp

    @auth_blp.route("/google/client-id", methods=["GET"])
    def auth_google_client_id():
        app_module._require_auth_db()
        if app_module._auth_is_mocked():
            abort(501, description="Google auth is unavailable in mock auth mode.")
        nonce = secrets.token_urlsafe(32)
        resp = make_response(jsonify({"clientId": app_module._google_oauth_client_id(), "nonce": nonce}))
        resp.headers["Cache-Control"] = "no-store"
        app_module._set_google_nonce_cookie(resp, nonce)
        return resp

    @auth_blp.route("/captcha/site-key", methods=["GET"])
    def auth_captcha_site_key():
        app_module._require_auth_db()
        if not app_module._turnstile_enabled():
            payload: dict[str, object] = {"enabled": False}
            if current_app.debug:
                payload["debug"] = {
                    "TURNSTILE_ENABLED": os.environ.get("TURNSTILE_ENABLED"),
                    "has_site_key": bool(os.environ.get("TURNSTILE_SITE_KEY", "").strip()),
                    "has_secret_key": bool(os.environ.get("TURNSTILE_SECRET_KEY", "").strip()),
                }
            resp = make_response(jsonify(payload))
            resp.headers["Cache-Control"] = "no-store"
            return resp
        payload: dict[str, object] = {"enabled": True, "siteKey": app_module._turnstile_site_key()}
        if current_app.debug:
            payload["debug"] = {
                "TURNSTILE_ENABLED": os.environ.get("TURNSTILE_ENABLED"),
                "has_site_key": True,
                "has_secret_key": bool(os.environ.get("TURNSTILE_SECRET_KEY", "").strip()),
            }
        resp = make_response(jsonify(payload))
        resp.headers["Cache-Control"] = "no-store"
        return resp

    @auth_blp.route("/google/callback", methods=["GET"])
    def auth_google_callback():
        app_module._require_auth_db()
        if app_module._auth_is_mocked():
            abort(501, description="Google auth is unavailable in mock auth mode.")
        if not app_module._google_oauth_flow_enabled():
            abort(404)

        error = request.args.get("error")
        if isinstance(error, str) and error.strip():
            return app_module._frontend_google_callback_redirect(
                token=None, next_path="/account", error=error
            )

        state = request.args.get("state")
        code = request.args.get("code")
        if not isinstance(state, str) or not state.strip():
            return app_module._frontend_google_callback_redirect(
                token=None, next_path="/account", error="missing_state"
            )
        if not isinstance(code, str) or not code.strip():
            return app_module._frontend_google_callback_redirect(
                token=None, next_path="/account", error="missing_code"
            )

        cookie_payload = app_module._load_google_oauth_cookie()
        if not cookie_payload:
            return app_module._frontend_google_callback_redirect(
                token=None, next_path="/account", error="invalid_state"
            )

        expected_state = cookie_payload.get("state")
        if not isinstance(expected_state, str) or not expected_state.strip():
            return app_module._frontend_google_callback_redirect(
                token=None, next_path="/account", error="invalid_state"
            )
        if not secrets.compare_digest(expected_state, state):
            return app_module._frontend_google_callback_redirect(
                token=None, next_path="/account", error="invalid_state"
            )

        code_verifier = cookie_payload.get("code_verifier")
        nonce = cookie_payload.get("nonce")
        next_path = app_module._safe_next_path(cookie_payload.get("next")) if cookie_payload else None
        if not isinstance(code_verifier, str) or not code_verifier.strip():
            return app_module._frontend_google_callback_redirect(
                token=None, next_path=next_path or "/account", error="invalid_state"
            )
        if not isinstance(nonce, str) or not nonce.strip():
            return app_module._frontend_google_callback_redirect(
                token=None, next_path=next_path or "/account", error="invalid_state"
            )

        token_payload = app_module._google_fetch_json(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": app_module._google_oauth_client_id(),
                "client_secret": app_module._google_oauth_client_secret(),
                "redirect_uri": app_module._google_oauth_redirect_uri(),
                "grant_type": "authorization_code",
                "code_verifier": code_verifier,
            },
        )

        id_token = token_payload.get("id_token")
        if not isinstance(id_token, str) or not id_token.strip():
            return app_module._frontend_google_callback_redirect(
                token=None, next_path=next_path or "/account", error="missing_id_token"
            )

        try:
            normalized = app_module._google_verify_id_token(id_token, expected_nonce=nonce)
        except HTTPException as e:
            return app_module._frontend_google_callback_redirect(
                token=None, next_path=next_path or "/account", error=f"google_{e.code}"
            )

        try:
            user = app_module.AuthUser.query.filter_by(email=normalized).first()
            if user is None:
                return app_module._frontend_google_callback_redirect(
                    token=None, next_path=next_path or "/account", error="legal_required"
                )
            if not app_module._user_has_current_legal_acceptances(user_id=user.id):
                return app_module._frontend_google_callback_redirect(
                    token=None, next_path=next_path or "/account", error="legal_required"
                )
            if user.email_verified_at is None:
                user.email_verified_at = datetime.utcnow()
            app_module._record_signon_event(user_id=user.id, provider="google", action="login")
            app_module.db.session.commit()
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")

        token = app_module._issue_session_token(user.id)
        if app_module._auth_session_transport() == "cookie":
            dest = f"{app_module._frontend_base_url()}{(next_path or '/account')}"
            resp = redirect(dest)
            resp.headers["Cache-Control"] = "no-store"
            app_module._set_auth_cookies(resp, session_token=token)
            app_module._clear_google_oauth_cookie(resp)
            return resp
        return app_module._frontend_google_callback_redirect(
            token=token, next_path=next_path or "/account", error=None
        )

    @auth_blp.route("/google/credential", methods=["POST"])
    def auth_google_credential():
        app_module._require_auth_db()
        if app_module._auth_is_mocked():
            abort(501, description="Google auth is unavailable in mock auth mode.")
        data = app_module._load_json(app_module.AuthGoogleCredentialSchema())
        credential = data.get("credential")
        if not isinstance(credential, str) or not credential.strip():
            abort(400, description="Missing Google credential.")

        expected_nonce = app_module._google_nonce_cookie_value()
        if not expected_nonce:
            abort(400, description="Missing Google nonce.")
        normalized = app_module._google_verify_id_token(credential, expected_nonce=expected_nonce)

        try:
            user = app_module.AuthUser.query.filter_by(email=normalized).first()
            if user is None:
                checked_at = app_module._require_legal_acceptance(data)
                now = datetime.utcnow()
                ip_address = app_module._request_ip_address()
                user_agent = app_module._request_user_agent()
                user = app_module.AuthUser(
                    email=normalized, password_hash=None, email_verified_at=now
                )
                app_module.db.session.add(user)
                app_module.db.session.flush()
                for doc, meta in app_module._LEGAL_DOCS.items():
                    app_module.db.session.add(
                        app_module.LegalAcceptance(
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
                app_module._record_signon_event(
                    user_id=user.id, provider="google", action="register"
                )
                app_module._send_signup_notification_email(new_user_email=user.email)
                app_module.db.session.commit()
            elif not app_module._user_has_current_legal_acceptances(user_id=user.id):
                checked_at = app_module._require_legal_acceptance(data)
                app_module._ensure_current_legal_acceptances(
                    user_id=user.id, checked_at=checked_at
                )
                if user.email_verified_at is None:
                    user.email_verified_at = datetime.utcnow()
                app_module._record_signon_event(user_id=user.id, provider="google", action="login")
                app_module.db.session.commit()
            else:
                if user.email_verified_at is None:
                    user.email_verified_at = datetime.utcnow()
                app_module._record_signon_event(user_id=user.id, provider="google", action="login")
                app_module.db.session.commit()
        except SQLAlchemyError:
            abort(503, description="Auth backend is unavailable right now.")

        token = app_module._issue_session_token(user.id)
        payload: dict[str, object] = {"user": {"id": user.id, "email": user.email}}
        if app_module._auth_session_transport() == "bearer":
            payload["sessionToken"] = token
        resp = make_response(jsonify(payload))
        resp.headers["Cache-Control"] = "no-store"
        if app_module._auth_session_transport() == "cookie":
            app_module._set_auth_cookies(resp, session_token=token)
        app_module._clear_google_nonce_cookie(resp)
        return resp

    @auth_blp.route("/logout", methods=["POST"])
    def auth_logout():
        app_module._require_auth_db()
        resp = app_module._status_response("ok")
        resp.headers["Cache-Control"] = "no-store"
        token = None
        if app_module._auth_session_transport() == "cookie":
            cookie_token = request.cookies.get(app_module._SESSION_COOKIE_NAME)
            token = cookie_token.strip() if isinstance(cookie_token, str) else None
        else:
            auth_header = request.headers.get("Authorization", "")
            if auth_header.startswith("Bearer "):
                token = auth_header.removeprefix("Bearer ").strip()
        if token:
            app_module._revoke_session_token(token)
        app_module._clear_auth_cookies(resp)
        return resp

    app.register_blueprint(auth_blp)
    return auth_blp
