"""MCP OAuth routes: DCR, authorization, consent, token, JWKS, discovery.

Registers the eight OAuth/OIDC endpoints under ``/v1/auth/oauth/...`` on the
existing auth blueprint, plus the helpers that back them (PKCE / scope /
redirect-URI validation, refresh-token rotation, signing-key rollover,
stale-client sweep). The DB-coupled helpers that bridge MCP identity to local
accounts (``ensure_linked_zitadel_subject_for_user``,
``linked_zitadel_subject_for_email``) live in routes/auth/__init__.py and are
injected at registration time.
"""

from __future__ import annotations

import json
import secrets
import uuid
from datetime import timedelta
from hashlib import sha256
from html import escape
from typing import Callable, cast
from urllib.parse import urlencode, urlparse, urlunparse

from flask import Blueprint, abort, current_app, jsonify, make_response, redirect, request
from sqlalchemy import and_, or_, update
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.exceptions import HTTPException

from backend.auth.mcp_oauth_runtime import (
    access_token_claims,
    encode_access_token,
    generate_signing_keypair,
    mcp_oauth_access_token_ttl_seconds,
    mcp_oauth_authorization_code_ttl_seconds,
    mcp_oauth_metadata,
    mcp_oauth_refresh_token_ttl_seconds,
    public_jwk_from_private_pem,
)
from backend.auth.mcp_runtime import mcp_resource_url, mcp_supported_scopes
from backend.routes.auth.cookies import (
    _clear_oauth_authorize_cookie,
    _clear_oauth_browser_cookie,
    _load_oauth_browser_cookie,
    _set_oauth_browser_cookie,
)
from backend.routes.auth.zitadel_config import _build_pkce_challenge
from backend.routes.deps import AuthDeps

_OAUTH_LOOPBACK_REDIRECT_HOSTS = {"localhost", "127.0.0.1", "::1"}
# DCR clients are world-registerable. Sweep clients that were registered but
# never used after this grace period, and clients that haven't been touched
# in this long — keeps the table from accumulating abandoned/spammed entries.
_DCR_UNUSED_GRACE_HOURS = 24
_DCR_INACTIVE_DAYS = 90


def register_mcp_oauth_routes(
    auth_blp: Blueprint,
    *,
    deps: AuthDeps,
    ensure_linked_zitadel_subject_for_user: Callable[..., str | None],
    linked_zitadel_subject_for_email: Callable[..., str | None],
) -> None:
    """Attach the MCP OAuth routes and their private helpers to ``auth_blp``.

    The two ``*_subject_*`` callables are the DB-coupled bridge helpers from
    routes/auth/__init__.py, passed in so the OAuth code never has to touch
    Zitadel state directly.
    """

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

    def _oauth_scope_set(scope: str) -> set[str]:
        return {part for part in scope.split(" ") if part}

    def _normalize_granted_scope(scope: str) -> str:
        return " ".join(sorted(_oauth_scope_set(scope)))

    def _user_grant_for_client(*, user_id: str, client_id: str):
        return deps.AuthOAuthUserGrant.query.filter_by(
            user_id=user_id,
            client_id=client_id,
            revoked_at=None,
        ).first()

    def _user_has_grant_for_scope(*, user_id: str, client_id: str, requested_scope: str) -> bool:
        grant = _user_grant_for_client(user_id=user_id, client_id=client_id)
        if grant is None:
            return False
        granted = _oauth_scope_set(grant.scope) if isinstance(grant.scope, str) else set()
        return _oauth_scope_set(requested_scope).issubset(granted)

    def _consent_redirect_response(
        *,
        client,
        requested_scope: str,
        redirect_uri: str,
        response_type: str,
        state: str | None,
        code_challenge: str,
        code_challenge_method: str,
    ):
        """Redirect the browser to the SPA consent screen, carrying every
        param needed to (a) render the page and (b) replay /oauth/authorize
        after the user clicks Allow. The page POSTs /v1/auth/oauth/consent
        to record the grant, then navigates to /v1/auth/oauth/authorize."""
        client_name = (
            client.client_name.strip()
            if isinstance(client.client_name, str) and client.client_name.strip()
            else ""
        )
        params: dict[str, str] = {
            "client_id": client.client_id,
            "redirect_uri": redirect_uri,
            "response_type": response_type,
            "scope": requested_scope,
            "code_challenge": code_challenge,
            "code_challenge_method": code_challenge_method,
        }
        if client_name:
            params["client_name"] = client_name
        if state:
            params["state"] = state
        # Defense-in-depth: an open redirect here would be a phishing primitive.
        # The scheme/host/path are fixed by server config (_frontend_base_url)
        # and tainted values only enter via urlencode (query string), but enforce
        # the invariant explicitly so a future refactor that breaks it fails
        # loudly instead of silently turning into an open redirect.
        frontend_base = deps._frontend_base_url()
        url = f"{frontend_base}/oauth/consent?{urlencode(params)}"
        if not url.startswith(f"{frontend_base}/oauth/consent?"):
            abort(500, description="Refusing to build malformed consent redirect.")
        resp = redirect(url, code=302)
        resp.headers["Cache-Control"] = "no-store"
        return resp

    def _touch_oauth_client_last_used(client_id: str) -> None:
        """Best-effort: mark a DCR client as actively in use so the sweep keeps it."""
        try:
            deps.db.session.execute(
                update(deps.AuthOAuthClient)
                .where(deps.AuthOAuthClient.client_id == client_id)
                .values(last_used_at=deps._utc_now())
            )
        except SQLAlchemyError:
            current_app.logger.exception("Failed to update OAuth client last_used_at.")
            deps.db.session.rollback()

    def _sweep_stale_oauth_clients() -> None:
        """Evict DCR clients that look abandoned: never-used past the grace
        window, or used long ago. Runs best-effort from /oauth/register so the
        table doesn't grow without bound from spammed registrations. Failures
        are swallowed — sweep is a hygiene measure, not a correctness barrier."""
        now = deps._utc_now()
        unused_cutoff = now - timedelta(hours=_DCR_UNUSED_GRACE_HOURS)
        inactive_cutoff = now - timedelta(days=_DCR_INACTIVE_DAYS)
        try:
            stale_clients = deps.AuthOAuthClient.query.filter(
                or_(
                    and_(
                        deps.AuthOAuthClient.last_used_at.is_(None),
                        deps.AuthOAuthClient.created_at < unused_cutoff,
                    ),
                    deps.AuthOAuthClient.last_used_at < inactive_cutoff,
                )
            ).limit(500).all()
            if not stale_clients:
                return
            stale_ids = [c.client_id for c in stale_clients]
            # Delete every table that has a FK pointing at auth_oauth_clients
            # BEFORE the client row itself; we don't configure ON DELETE
            # CASCADE so on Postgres the parent DELETE would otherwise fail
            # with a FK violation. SQLite doesn't enforce FKs by default so
            # the symptom would only surface in production.
            deps.AuthOAuthRefreshToken.query.filter(
                deps.AuthOAuthRefreshToken.client_id.in_(stale_ids)
            ).delete(synchronize_session=False)
            deps.AuthOAuthAuthorizationCode.query.filter(
                deps.AuthOAuthAuthorizationCode.client_id.in_(stale_ids)
            ).delete(synchronize_session=False)
            deps.AuthOAuthUserGrant.query.filter(
                deps.AuthOAuthUserGrant.client_id.in_(stale_ids)
            ).delete(synchronize_session=False)
            deps.AuthOAuthClient.query.filter(
                deps.AuthOAuthClient.client_id.in_(stale_ids)
            ).delete(synchronize_session=False)
            deps.db.session.commit()
        except SQLAlchemyError:
            current_app.logger.exception("OAuth client sweep failed.")
            deps.db.session.rollback()

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

    def _oauth_resume_authorize_path(pending_payload: dict[str, str]) -> str:
        # Embed the full OAuth params in the post-login `next` URL so that resume
        # is bound to the request the client initiated, not to a server-side
        # cookie that a different user (e.g., on a shared device) could finalize.
        params = dict(pending_payload)
        params["response_type"] = "code"
        return f"/v1/auth/oauth/authorize?{urlencode(params)}"

    def _oauth_authorize_bridge_response(*, pending_payload: dict[str, str]):
        next_path = _oauth_resume_authorize_path(pending_payload)
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
        # Clear any stale pending cookie left over from prior versions; resume now
        # relies on the `next=` query string, not on this cookie.
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
        # World-open endpoint; sweep abandoned/spammed clients before adding a
        # new row so the table can't grow without bound.
        _sweep_stale_oauth_clients()
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

    @auth_blp.post("/oauth/consent")
    def oauth_consent():
        """Record a user's consent for a DCR-registered client to receive the
        requested scopes. Required before /oauth/authorize will mint a code.

        Auth: verified user session + CSRF (standard /v1/* path). The grant is
        scoped to (user, client) so cross-user replay is impossible. The
        client_id and scope are validated server-side; the caller can't grant
        a scope the client didn't register for or that the server doesn't
        support."""
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            abort(400, description="Invalid consent payload.")
        client_id = data.get("client_id")
        scope_raw = data.get("scope")
        if not isinstance(client_id, str) or not client_id.strip():
            abort(400, description="client_id is required.")
        if not isinstance(scope_raw, str) or not scope_raw.strip():
            abort(400, description="scope is required.")
        # Re-runs the same scope normalization the authorize endpoint applies,
        # so an unsupported scope is rejected with the identical error.
        scope = _oauth_scope_string(scope_raw)
        client = deps.AuthOAuthClient.query.filter_by(client_id=client_id.strip()).first()
        if client is None:
            abort(400, description="Invalid OAuth client.")

        now = deps._utc_now()
        normalized_scope = _normalize_granted_scope(scope)
        try:
            existing = _user_grant_for_client(user_id=user.id, client_id=client.client_id)
            if existing is None:
                grant = deps.AuthOAuthUserGrant(
                    user_id=user.id,
                    client_id=client.client_id,
                    scope=normalized_scope,
                    granted_at=now,
                )
                deps.db.session.add(grant)
                stored_scope = normalized_scope
            else:
                # Widen the existing grant to cover any additional scopes
                # the user is approving in this round; never narrow.
                merged = _oauth_scope_set(existing.scope) | _oauth_scope_set(normalized_scope)
                stored_scope = " ".join(sorted(merged))
                existing.scope = stored_scope
                existing.granted_at = now
                existing.revoked_at = None
            deps.db.session.commit()
        except SQLAlchemyError:
            deps.db.session.rollback()
            abort(503, description="Auth backend is unavailable right now.")

        resp = make_response(
            jsonify(
                {
                    "status": "granted",
                    "client_id": client.client_id,
                    "scope": stored_scope,
                }
            ),
            200,
        )
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
        client_id = request.args.get("client_id", "").strip()
        redirect_uri = request.args.get("redirect_uri", "").strip()
        response_type = request.args.get("response_type", "").strip()
        state = (request.args.get("state", "") or "").strip() or None
        scope = _oauth_scope_string(request.args.get("scope"))
        code_challenge = request.args.get("code_challenge", "").strip()
        code_challenge_method = request.args.get("code_challenge_method", "").strip()
        if not client_id or not redirect_uri:
            return _oauth_error_response("Missing OAuth client_id or redirect_uri.")
        client = deps.AuthOAuthClient.query.filter_by(client_id=client_id).first()
        if client is None or not _oauth_redirect_uri_allowed(client, redirect_uri):
            return _oauth_error_response("Invalid OAuth client or redirect URI.")
        if response_type != "code":
            return _oauth_redirect_error(redirect_uri, error="unsupported_response_type", state=state)
        if not scope:
            return _oauth_redirect_error(redirect_uri, error="invalid_scope", state=state)
        if not code_challenge or code_challenge_method != "S256":
            return _oauth_redirect_error(redirect_uri, error="invalid_request", state=state)
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
            next_path = _oauth_resume_authorize_path(pending_payload)
            return redirect(
                f"{deps._frontend_base_url()}/login?{urlencode({'next': next_path})}",
                code=302,
            )

        if not deps._user_has_current_legal_acceptances(user_id=user.id):
            next_path = _oauth_resume_authorize_path(pending_payload)
            return redirect(
                f"{deps._frontend_base_url()}/login?{urlencode({'next': next_path})}",
                code=302,
            )
        linked_subject = ensure_linked_zitadel_subject_for_user(user=user)
        if not linked_subject:
            linked_subject = linked_zitadel_subject_for_email(email=user.email)
        if not linked_subject:
            return _oauth_error_response("Pandects could not link this account to MCP identity.", code=403)

        # Per-client consent gate: a DCR client can be registered by anyone, so
        # don't mint a code until the user has explicitly granted this client
        # the requested scopes. Redirect the browser to the SPA consent screen,
        # which POSTs /v1/auth/oauth/consent to record the grant and then
        # re-navigates to /v1/auth/oauth/authorize.
        if not _user_has_grant_for_scope(
            user_id=user.id, client_id=client.client_id, requested_scope=scope
        ):
            return _consent_redirect_response(
                client=client,
                requested_scope=scope,
                redirect_uri=redirect_uri,
                response_type=response_type,
                state=state,
                code_challenge=code_challenge,
                code_challenge_method=code_challenge_method,
            )

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
        _touch_oauth_client_last_used(client.client_id)
        deps.db.session.commit()
        _params = {"code": raw_code}
        if state:
            _params["state"] = state
        resp = redirect(_oauth_redirect_location(redirect_uri, _params), code=302)
        _clear_oauth_browser_cookie(resp)
        _clear_oauth_authorize_cookie(resp)
        return resp

    def _issue_token_pair(
        *,
        user_id: str,
        client_id: str,
        scope: str,
        family_id: str | None = None,
    ) -> dict[str, object]:
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
                family_id=family_id or str(uuid.uuid4()),
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
            now = deps._utc_now()
            if auth_code is None or auth_code.used_at is not None or auth_code.expires_at < now:
                abort(400, description="Invalid or expired OAuth code.")
            if auth_code.redirect_uri != redirect_uri:
                abort(400, description="OAuth redirect URI mismatch.")
            expected_challenge = _build_pkce_challenge(code_verifier)
            if not secrets.compare_digest(expected_challenge, auth_code.code_challenge):
                abort(400, description="OAuth PKCE verification failed.")
            user = deps.db.session.get(deps.AuthUser, auth_code.user_id)
            if user is None or cast(object, user.email_verified_at) is None:
                abort(403, description="Linked Pandects account is not verified.")
            # Atomic claim: only succeeds if used_at IS NULL. Prevents two concurrent
            # token exchanges from minting parallel tokens for the same code.
            claim_result = deps.db.session.execute(
                update(deps.AuthOAuthAuthorizationCode)
                .where(
                    deps.AuthOAuthAuthorizationCode.id == auth_code.id,
                    deps.AuthOAuthAuthorizationCode.used_at.is_(None),
                )
                .values(used_at=now)
            )
            if claim_result.rowcount != 1:
                deps.db.session.rollback()
                abort(400, description="Invalid or expired OAuth code.")
            token_payload = _issue_token_pair(
                user_id=user.id,
                client_id=client.client_id,
                scope=auth_code.scope,
            )
            _touch_oauth_client_last_used(client.client_id)
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
            if refresh_record is None or refresh_record.expires_at < now:
                abort(400, description="Invalid or expired refresh token.")
            # Reuse detection: a presented token that has already been rotated or
            # revoked indicates theft. Revoke the entire token family (RFC 6819 §5.2.2.3).
            if refresh_record.used_at is not None or refresh_record.revoked_at is not None:
                _ = deps.db.session.execute(
                    update(deps.AuthOAuthRefreshToken)
                    .where(
                        deps.AuthOAuthRefreshToken.family_id == refresh_record.family_id,
                        deps.AuthOAuthRefreshToken.revoked_at.is_(None),
                    )
                    .values(revoked_at=now)
                )
                deps.db.session.commit()
                abort(400, description="Invalid or expired refresh token.")
            user = deps.db.session.get(deps.AuthUser, refresh_record.user_id)
            if user is None or cast(object, user.email_verified_at) is None:
                abort(403, description="Linked Pandects account is not verified.")
            # Atomic rotation: only succeeds if used_at IS NULL. A concurrent rotation
            # losing this race is treated as reuse and revokes the family.
            rotate_result = deps.db.session.execute(
                update(deps.AuthOAuthRefreshToken)
                .where(
                    deps.AuthOAuthRefreshToken.id == refresh_record.id,
                    deps.AuthOAuthRefreshToken.used_at.is_(None),
                    deps.AuthOAuthRefreshToken.revoked_at.is_(None),
                )
                .values(used_at=now)
            )
            if rotate_result.rowcount != 1:
                _ = deps.db.session.execute(
                    update(deps.AuthOAuthRefreshToken)
                    .where(
                        deps.AuthOAuthRefreshToken.family_id == refresh_record.family_id,
                        deps.AuthOAuthRefreshToken.revoked_at.is_(None),
                    )
                    .values(revoked_at=now)
                )
                deps.db.session.commit()
                abort(400, description="Invalid or expired refresh token.")
            token_payload = _issue_token_pair(
                user_id=user.id,
                client_id=client.client_id,
                scope=refresh_record.scope,
                family_id=refresh_record.family_id,
            )
            _touch_oauth_client_last_used(client.client_id)
            deps.db.session.commit()
            resp = make_response(jsonify(token_payload), 200)
            resp.headers["Cache-Control"] = "no-store"
            return resp

        abort(400, description="Unsupported grant_type.")
