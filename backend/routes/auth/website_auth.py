"""Website (first-party) ZITADEL auth flows behind the /v1/auth routes.

The route handlers in ``backend.routes.auth`` parse and validate requests; the
account work they share (resolving identities, pending signups, email
verification, legal gating, account deletion) lives here so each handler stays
short. Behavior is unchanged from when these were closures in
``register_auth_routes``: every ``abort`` keeps its status code and message.
"""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime
from hashlib import sha256
from typing import Any, cast

from flask import Response, abort, current_app, jsonify, make_response
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.exceptions import HTTPException
from werkzeug.security import check_password_hash

from backend.auth import identity_links
from backend.auth.mcp_runtime import ExternalIdentity, mcp_oidc_issuer
from backend.core.config import temporary_access_lockdown_enabled
from backend.routes.auth.cookies import _set_zitadel_pending_cookie
from backend.routes.auth.zitadel_api import (
    ZitadelApiClient,
    ZitadelEmailVerificationRequiredError,
)
from backend.routes.auth.zitadel_config import (
    _decode_zitadel_id_token,
    _website_zitadel_redirect_uri,
    _zitadel_audience,
    _zitadel_client_id,
    _zitadel_resource,
    _zitadel_token_endpoint,
)
from backend.routes.deps import AuthDeps


def no_store(resp: Response) -> Response:
    resp.headers["Cache-Control"] = "no-store"
    return resp


def website_auth_error_response(message: str, *, code: int = 400) -> Response:
    return no_store(make_response(jsonify({"message": message, "error": "auth_failed"}), code))


def _user_payload(user: Any) -> dict[str, object]:
    return {"id": user.id, "email": user.email}


def _verification_required_payload(*, next_path: str, user: dict[str, object]) -> Response:
    return no_store(
        make_response(
            jsonify({"status": "verification_required", "next_path": next_path, "user": user})
        )
    )


class WebsiteAuthService:
    def __init__(self, deps: AuthDeps, zitadel_api: ZitadelApiClient) -> None:
        self.deps = deps
        self.zitadel_api = zitadel_api

    # -- identity lookups -------------------------------------------------

    def _local_user_by_email(self, *, email: str) -> Any:
        return identity_links.local_user_by_email(self.deps, email=email)

    def _zitadel_link_for_user(self, *, user: Any) -> Any:
        return self.deps.AuthExternalSubject.query.filter_by(
            user_id=user.id,
            issuer=mcp_oidc_issuer(),
        ).first()

    def _set_zitadel_link_subject(self, *, user: Any, subject: str) -> None:
        existing_link = self._zitadel_link_for_user(user=user)
        if existing_link is None:
            self.deps.db.session.add(
                self.deps.AuthExternalSubject(
                    user_id=user.id,
                    issuer=mcp_oidc_issuer(),
                    subject=subject,
                )
            )
        else:
            existing_link.subject = subject
        self.deps.db.session.flush()

    def _delete_zitadel_link(self, *, user: Any) -> None:
        existing_link = self._zitadel_link_for_user(user=user)
        if existing_link is not None:
            self.deps.db.session.delete(existing_link)
            self.deps.db.session.flush()

    def _ensure_zitadel_user_for_local_password_user(self, *, user: Any, password: str):
        existing_link = self._zitadel_link_for_user(user=user)
        if existing_link is not None:
            return self.zitadel_api.build_external_identity(
                subject=existing_link.subject, email=user.email
            )
        return self.zitadel_api.create_human_user(
            email=user.email,
            password=password,
            email_is_verified=True,
        )

    def _maybe_migrate_local_password_user(self, *, email: str, password: str):
        user = self._local_user_by_email(email=email)
        if user is None or user.email_verified_at is None:
            return None
        if not isinstance(user.password_hash, str) or not user.password_hash.strip():
            return None
        if not check_password_hash(user.password_hash, password):
            return None
        return self._ensure_zitadel_user_for_local_password_user(user=user, password=password)

    def linked_zitadel_subject_for_email(self, *, email: str, allow_unverified: bool = False):
        user = self._local_user_by_email(email=email)
        if user is None:
            return None
        if user.email_verified_at is None and not allow_unverified:
            return None
        existing_link = self._zitadel_link_for_user(user=user)
        if existing_link is not None:
            return existing_link.subject
        if isinstance(user.password_hash, str) and user.password_hash.strip():
            try:
                migrated = self.zitadel_api.create_human_user(
                    email=user.email,
                    password=secrets.token_urlsafe(24),
                    email_is_verified=user.email_verified_at is not None,
                )
            except HTTPException as exc:
                description = exc.description if isinstance(exc.description, str) else ""
                if exc.code == 409 and "already exists" in description.lower():
                    existing_user_id = self.zitadel_api.search_user_id_by_login_name(
                        login_name=user.email
                    )
                    if not isinstance(existing_user_id, str) or not existing_user_id.strip():
                        raise
                    migrated = self.zitadel_api.build_external_identity(
                        subject=existing_user_id.strip(),
                        email=user.email,
                    )
                else:
                    raise
            _ = identity_links.resolve_user_from_external_identity(
                self.deps, external_identity=migrated
            )
            self.deps.db.session.commit()
            return migrated.subject
        return None

    def ensure_linked_zitadel_subject_for_user(self, *, user: Any) -> str | None:
        existing_link = self._zitadel_link_for_user(user=user)
        if (
            existing_link is not None
            and isinstance(existing_link.subject, str)
            and existing_link.subject.strip()
        ):
            return existing_link.subject.strip()
        searched_user_id = self.zitadel_api.search_user_id_by_login_name(login_name=user.email)
        if isinstance(searched_user_id, str) and searched_user_id.strip():
            resolved_user_id = searched_user_id.strip()
            self.deps.db.session.add(
                self.deps.AuthExternalSubject(
                    user_id=user.id,
                    issuer=mcp_oidc_issuer(),
                    subject=resolved_user_id,
                )
            )
            self.deps.db.session.flush()
            return resolved_user_id
        return None

    def _user_created_at_or_now(self, *, user: Any) -> datetime:
        created_at = getattr(user, "created_at", None)
        if isinstance(created_at, datetime):
            return created_at
        return self.deps._utc_now()

    def _send_signup_notification(
        self, *, user: Any, provider: str, full_name: str | None
    ) -> None:
        self.deps._send_signup_notification_email(
            new_user_email=user.email,
            provider=provider,
            full_name=full_name,
            signed_up_at=self._user_created_at_or_now(user=user),
        )

    # -- responses ----------------------------------------------------------

    def auth_success_response(self, *, user: Any, next_path: str, resp_code: int = 200) -> Response:
        token = self.deps._issue_session_token(user.id)
        payload: dict[str, object] = {
            "status": "authenticated",
            "next_path": next_path,
            "user": _user_payload(user),
        }
        if self.deps._auth_session_transport() == "bearer":
            payload["session_token"] = token
        resp = no_store(make_response(jsonify(payload), resp_code))
        if self.deps._auth_session_transport() == "cookie":
            self.deps._set_auth_cookies(resp, session_token=token)
        return resp

    def pending_website_auth_response(
        self, *, user: Any, next_path: str, action: str, provider_name: str
    ) -> Response:
        resp = no_store(
            make_response(
                jsonify(
                    {
                        "status": "legal_required",
                        "next_path": next_path,
                        "user": _user_payload(user),
                    }
                )
            )
        )
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

    def verification_required_response(self, *, user: Any, next_path: str) -> Response:
        return _verification_required_payload(next_path=next_path, user=_user_payload(user))

    def mcp_access_token_response(
        self,
        *,
        access_token: str,
        next_path: str,
        token_payload: dict[str, object],
    ) -> Response:
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
        return no_store(make_response(jsonify(payload), 200))

    # -- shared completion ----------------------------------------------------

    def complete_website_auth_for_identity(
        self,
        *,
        external_identity: Any,
        next_path: str,
        provider_name: str,
        notification_provider: str | None = None,
    ) -> Response:
        deps = self.deps
        action, user = identity_links.resolve_user_from_external_identity(
            deps, external_identity=external_identity
        )
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
        signup_full_name = identity_links.full_name_from_parts(signup_first_name, signup_last_name)
        signup_provider = notification_provider or provider_name
        if not deps._user_has_current_legal_acceptances(user_id=user.id):
            deps.db.session.commit()
            if action == "register":
                self._send_signup_notification(
                    user=user, provider=signup_provider, full_name=signup_full_name
                )
            return self.pending_website_auth_response(
                user=user,
                next_path=next_path,
                action=action,
                provider_name=provider_name,
            )

        deps._record_signon_event(user_id=user.id, provider=provider_name, action=action)
        deps.db.session.commit()
        if action == "register":
            self._send_signup_notification(
                user=user, provider=signup_provider, full_name=signup_full_name
            )
        return self.auth_success_response(user=user, next_path=next_path)

    # -- email verification ---------------------------------------------------

    def _email_verify_url_template(self) -> str:
        return (
            f"{self.deps._frontend_base_url()}/verify-email"
            "?user_id={{.UserID}}&code={{.Code}}&org_id={{.OrgID}}"
        )

    def send_email_verification_for_user(self, *, user: Any) -> bool:
        subject = self.ensure_linked_zitadel_subject_for_user(user=user)
        if not isinstance(subject, str) or not subject.strip():
            return False
        url_template = self._email_verify_url_template()
        try:
            self.zitadel_api.send_email_verification(user_id=subject, url_template=url_template)
            return True
        except HTTPException as exc:
            description = exc.description if isinstance(exc.description, str) else ""
            if not (exc.code in {400, 404} and "email not found" in description.lower()):
                raise
        # The linked ZITADEL user is gone or lost its email: relink to the user
        # ZITADEL now has for this login name, or drop the stale link.
        refreshed_subject = self.zitadel_api.search_user_id_by_login_name(login_name=user.email)
        if (
            isinstance(refreshed_subject, str)
            and refreshed_subject.strip()
            and refreshed_subject.strip() != subject
        ):
            self._set_zitadel_link_subject(user=user, subject=refreshed_subject.strip())
            self.zitadel_api.send_email_verification(
                user_id=refreshed_subject.strip(),
                url_template=url_template,
            )
            return True
        self._delete_zitadel_link(user=user)
        return False

    def confirm_email_verification(self, *, user_id: str, code: str, next_path: str) -> Response:
        deps = self.deps
        try:
            _ = self.zitadel_api.json(
                path=f"/v2/users/{user_id}/email/verify",
                json_body={"verificationCode": code},
            )
            external_identity = self.zitadel_api.user_identity(user_id=user_id)
            return self.complete_website_auth_for_identity(
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

    def resend_email_verification(self, *, email: str) -> None:
        """Best effort: never raises, so the caller's response cannot reveal the account state."""
        deps = self.deps
        user = self._local_user_by_email(email=email)
        if user is None or user.email_verified_at is not None:
            return
        try:
            if self.send_email_verification_for_user(user=user):
                deps.db.session.commit()
            else:
                deps.db.session.rollback()
        except (HTTPException, SQLAlchemyError):
            deps.db.session.rollback()
        except Exception:
            deps.db.session.rollback()
            current_app.logger.exception("Email verification resend failed.")

    # -- password login -------------------------------------------------------

    def _pending_login_block(self, *, user: Any) -> None:
        has_legal = self.deps._user_has_current_legal_acceptances(user_id=user.id)
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
            # adding it to the pass-through (verified + legal user) would slow
            # the legitimate login by ~250ms and create a new timing oracle
            # vs. the no-link path.
            self.deps._auth_enumeration_delay()
            abort(400, description=blocked_reason)

    def _password_identity(self, *, email: str, password: str):
        deps = self.deps
        try:
            return self.zitadel_api.session_identity(email=email, password=password)
        except ZitadelEmailVerificationRequiredError:
            # Reveals "account exists but unverified". Equalize timing
            # with the invalid-credentials path below.
            deps._auth_enumeration_delay()
            raise
        except HTTPException as exc:
            if exc.code not in {400, 401, 403, 404, 409}:
                raise
        migrated = self._maybe_migrate_local_password_user(email=email, password=password)
        if migrated is None:
            # "No such user" (Zitadel 404, fast) and "wrong password" (Zitadel
            # verification, slow) both reach this abort. Record the failure for
            # the per-account throttle and add randomized jitter so the
            # response times are indistinguishable.
            deps._record_account_login_failure(email)
            deps._auth_enumeration_delay()
            abort(401, description="Invalid email or password.")
        return migrated

    def password_login(self, *, email: str, password: str, next_path: str) -> Response:
        deps = self.deps
        # Per-account throttle layered on top of the per-IP endpoint limit.
        # Prevents credential stuffing where attempts are spread across many
        # source IPs against the same email. Aborts 429 if backed off.
        deps._check_account_login_rate_limit(email)

        existing_user = self._local_user_by_email(email=email)
        if existing_user is not None and self._zitadel_link_for_user(user=existing_user) is not None:
            # Aborts with 400 when this account is pending (unverified email /
            # missing legal), after a delay so the abort-vs-Zitadel timing is
            # indistinguishable. Otherwise continue to the real Zitadel call
            # without an artificial delay.
            self._pending_login_block(user=existing_user)

        try:
            external_identity = self._password_identity(email=email, password=password)
            resp = self.complete_website_auth_for_identity(
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

    # -- password signup ------------------------------------------------------

    def _ensure_pending_signup_remote_user(
        self,
        *,
        user: Any,
        password: str,
        first_name: str | None = None,
        last_name: str | None = None,
        send_verification: bool = False,
    ) -> str | None:
        subject = self.ensure_linked_zitadel_subject_for_user(user=user)
        if isinstance(subject, str) and subject.strip():
            try:
                if self.zitadel_api.user_email_matches(user_id=subject.strip(), email=user.email):
                    return subject.strip()
            except HTTPException as exc:
                if exc.code != 404:
                    raise
                self._delete_zitadel_link(user=user)
        try:
            external_identity = self.zitadel_api.create_human_user(
                email=user.email,
                password=password,
                first_name=first_name,
                last_name=last_name,
                email_is_verified=False,
                verify_url_template=(
                    self._email_verify_url_template() if send_verification else None
                ),
            )
        except HTTPException as exc:
            description = exc.description if isinstance(exc.description, str) else ""
            if not (exc.code == 409 and "already exists" in description.lower()):
                raise
            existing_user_id = self.zitadel_api.search_user_id_by_login_name(login_name=user.email)
            if not isinstance(existing_user_id, str) or not existing_user_id.strip():
                return None
            external_identity = self.zitadel_api.build_external_identity(
                subject=existing_user_id.strip(),
                email=user.email,
                email_verified=False,
            )
        self._set_zitadel_link_subject(user=user, subject=external_identity.subject)
        return external_identity.subject if isinstance(external_identity.subject, str) else None

    def _resume_incomplete_signup_if_eligible(
        self, *, user: Any, next_path: str, provider_name: str
    ) -> Response | None:
        subject = self.ensure_linked_zitadel_subject_for_user(user=user)
        if subject is None:
            return None
        if not self.deps._user_has_current_legal_acceptances(user_id=user.id):
            self.deps.db.session.commit()
            return self.pending_website_auth_response(
                user=user,
                next_path=next_path,
                action="register",
                provider_name=provider_name,
            )
        if user.email_verified_at is None:
            _ = self.send_email_verification_for_user(user=user)
            self.deps.db.session.commit()
            return self.verification_required_response(user=user, next_path=next_path)
        # Fully provisioned accounts must authenticate via /login/password.
        # Returning a session here would let any caller take over an account
        # by posting the signup endpoint with the victim's email — the
        # submitted password is never verified on this path.
        return None

    def _signup_for_existing_user(
        self,
        *,
        user: Any,
        email: str,
        password: str,
        first_name: str | None,
        last_name: str | None,
        next_path: str,
    ) -> Response:
        deps = self.deps
        if user.email_verified_at is None:
            has_current_legal = deps._user_has_current_legal_acceptances(user_id=user.id)
            remote_subject = self._ensure_pending_signup_remote_user(
                user=user,
                password=password,
                first_name=first_name,
                last_name=last_name,
                send_verification=has_current_legal,
            )
            if isinstance(remote_subject, str) and remote_subject.strip():
                if not has_current_legal:
                    deps.db.session.commit()
                    return self.pending_website_auth_response(
                        user=user,
                        next_path=next_path,
                        action="register",
                        provider_name="zitadel",
                    )
                if self.send_email_verification_for_user(user=user):
                    deps.db.session.commit()
                    return self.verification_required_response(user=user, next_path=next_path)
        resumed = self._resume_incomplete_signup_if_eligible(
            user=user,
            next_path=next_path,
            provider_name="zitadel",
        )
        if resumed is not None:
            return resumed
        # Existing fully-provisioned account. Return the same shape as a fresh
        # signup so an unauthenticated caller can't use this endpoint to
        # enumerate registered emails. The id is a throwaway UUID so the
        # response is shape-indistinguishable from a legitimate new signup.
        # The real user will be told to check their email; if no message
        # arrives, they can sign in or use the password-reset flow.
        # Log the hashed email rather than the raw address — the log line is
        # the enumeration signal the response is supposed to mask, so anyone
        # with log access shouldn't be able to read which addresses already
        # had accounts.
        current_app.logger.info(
            "signup_password_existing_account_masked email_sha256=%s",
            sha256(email.encode("utf-8")).hexdigest()[:16],
        )
        return _verification_required_payload(
            next_path=next_path, user={"id": str(uuid.uuid4()), "email": email}
        )

    def _ensure_pending_signup_user(self, *, external_identity: Any) -> Any:
        deps = self.deps
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

    def _signup_new_user(
        self,
        *,
        email: str,
        password: str,
        first_name: str | None,
        last_name: str | None,
        next_path: str,
    ) -> Response:
        deps = self.deps
        try:
            external_identity = self.zitadel_api.create_human_user(
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name,
                email_is_verified=False,
            )
            user = self._ensure_pending_signup_user(external_identity=external_identity)
            has_current_legal = deps._user_has_current_legal_acceptances(user_id=user.id)
            if has_current_legal:
                _ = self.send_email_verification_for_user(user=user)
            deps.db.session.commit()
            self._send_signup_notification(
                user=user,
                provider="password",
                full_name=identity_links.full_name_from_parts(first_name, last_name),
            )
            if has_current_legal:
                return self.verification_required_response(user=user, next_path=next_path)
            return self.pending_website_auth_response(
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

    def password_signup(
        self,
        *,
        email: str,
        password: str,
        first_name: str | None,
        last_name: str | None,
        next_path: str,
    ) -> Response:
        existing_user = self._local_user_by_email(email=email)
        if existing_user is not None:
            return self._signup_for_existing_user(
                user=existing_user,
                email=email,
                password=password,
                first_name=first_name,
                last_name=last_name,
                next_path=next_path,
            )
        return self._signup_new_user(
            email=email,
            password=password,
            first_name=first_name,
            last_name=last_name,
            next_path=next_path,
        )

    # -- ZITADEL callbacks ------------------------------------------------------

    def identity_from_authorization_code(
        self, *, code: str, code_verifier: str, provider_name: str
    ) -> tuple[Any, str, dict[str, object]]:
        """Exchange a website authorization code; returns (identity, access_token, token_payload)."""
        deps = self.deps
        token_data = {
            "grant_type": "authorization_code",
            "client_id": _zitadel_client_id(),
            "code": code,
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
        if identity_links.external_identity_verified_email(deps, external_identity) is None:
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
        return external_identity, access_token.strip(), token_payload

    def google_identity_from_intent(
        self, *, payload: dict[str, object], user_id: str | None
    ) -> ExternalIdentity:
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
            existing_subject = self.linked_zitadel_subject_for_email(
                email=self.deps._normalize_email(email)
            )
            if isinstance(existing_subject, str) and existing_subject.strip():
                resolved_user_id = existing_subject.strip()
        if resolved_user_id is None:
            resolved_user_id = self._create_google_zitadel_user(
                raw_user=raw_user, idp_id=idp_id, external_subject=external_subject.strip()
            )

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

    def _create_google_zitadel_user(
        self, *, raw_user: dict[Any, Any], idp_id: str, external_subject: str
    ) -> str:
        email = cast(str, raw_user.get("email"))
        display_name = raw_user.get("name")
        if not isinstance(display_name, str) or not display_name.strip():
            display_name = email.strip()
        given_name = raw_user.get("given_name")
        family_name = raw_user.get("family_name")
        profile: dict[str, object] = {"displayName": display_name}
        if isinstance(given_name, str) and given_name.strip():
            profile["givenName"] = given_name.strip()
        if isinstance(family_name, str) and family_name.strip():
            profile["familyName"] = family_name.strip()
        create_payload: dict[str, object] = {
            "username": email.strip(),
            "profile": profile,
            "email": {
                "email": email.strip(),
                "isVerified": True,
            },
            "idpLinks": [
                {
                    "idpId": idp_id,
                    "userId": external_subject,
                    "userName": display_name,
                }
            ],
        }
        try:
            created = self.zitadel_api.json(path="/v2/users/human", json_body=create_payload)
            user_id_candidate = created.get("userId") or created.get("user_id")
            if not isinstance(user_id_candidate, str) or not user_id_candidate.strip():
                abort(502, description="ZITADEL did not return a user identifier for the Google login.")
            self.zitadel_api.ensure_default_user_grant(user_id=user_id_candidate.strip())
            return user_id_candidate.strip()
        except HTTPException as exc:
            description = exc.description if isinstance(exc.description, str) else ""
            if exc.code == 409 and "already exists" in description.lower():
                existing_user_id = self.zitadel_api.search_user_id_by_login_name(
                    login_name=email.strip()
                )
                if isinstance(existing_user_id, str) and existing_user_id.strip():
                    return existing_user_id.strip()
            raise

    # -- account deletion -------------------------------------------------------

    def _collect_zitadel_subjects_for_user(self, *, user: Any) -> list[str]:
        links = (
            self.deps.AuthExternalSubject.query.filter_by(user_id=user.id, issuer=mcp_oidc_issuer())
            .order_by(self.deps.AuthExternalSubject.id.asc())
            .all()
        )
        return [
            link.subject
            for link in links
            if isinstance(link.subject, str) and link.subject.strip()
        ]

    def _delete_zitadel_subjects_best_effort(self, subjects: list[str], *, user_id: str) -> None:
        # Called AFTER the local tombstone is committed. A failure here
        # leaves a Zitadel orphan but does not roll back the local-side
        # privacy guarantee — losing the local row is the part the user
        # is asking for. Log the orphan so it can be cleaned up out of
        # band rather than aborting and leaving the caller's account in
        # a half-deleted state.
        for subject in subjects:
            try:
                _ = self.zitadel_api.request(
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

    def delete_account(self, *, user: Any) -> None:
        deps = self.deps
        # Capture the Zitadel subjects BEFORE the local commit. We delete
        # them from Zitadel only after the local tombstone has committed —
        # otherwise a failed local commit would leave the Zitadel account
        # gone but the local row active, producing an "I'm deleted but
        # can't sign in" state that's impossible to recover from on the
        # next attempt.
        try:
            zitadel_subjects = self._collect_zitadel_subjects_for_user(user=user)
            local_user_id = user.id
            now = deps._utc_now()
            deps.AuthExternalSubject.query.filter_by(user_id=user.id).delete(
                synchronize_session=False
            )
            user.email = f"deleted+{uuid.uuid4().hex}@deleted.invalid"
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

        self._delete_zitadel_subjects_best_effort(zitadel_subjects, user_id=local_user_id)
