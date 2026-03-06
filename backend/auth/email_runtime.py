from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request

from flask import abort, current_app
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from sqlalchemy.exc import SQLAlchemyError

from backend.core.runtime_utils import urlopen_read_bytes as _urlopen_read_bytes
from backend.core.runtime_utils import utc_now as _utc_now
from backend.extensions import db
from backend.models import AuthPasswordResetToken

from backend.auth.session_runtime import (
    _mock_auth,
    auth_is_mocked,
    request_ip_address,
    request_user_agent,
)


def is_email_like(value: str) -> bool:
    if not value or value.strip() != value:
        return False
    if any(ch.isspace() for ch in value):
        return False
    if value.count("@") != 1:
        return False
    local, domain = value.split("@", 1)
    if not local or not domain:
        return False
    if "." not in domain:
        return False
    if domain.startswith(".") or domain.endswith("."):
        return False
    return True


def normalize_email(email: str) -> str:
    return email.strip().lower()


def frontend_base_url() -> str:
    base = os.environ.get("PUBLIC_FRONTEND_BASE_URL", "").strip().rstrip("/")
    if base:
        return base
    if current_app.debug:
        return "http://localhost:8080"
    abort(503, description="Google auth is not configured (missing PUBLIC_FRONTEND_BASE_URL).")


def public_api_base_url() -> str:
    base = os.environ.get("PUBLIC_API_BASE_URL", "").strip().rstrip("/")
    if base:
        return base
    if current_app.debug:
        return "http://127.0.0.1:5113"
    abort(503, description="Google auth is not configured (missing PUBLIC_API_BASE_URL).")


def email_verification_serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("AUTH_SECRET_KEY")
    if not secret:
        abort(503, description="Auth is not configured (missing AUTH_SECRET_KEY).")
    return URLSafeTimedSerializer(secret_key=secret, salt="pandects-email-verify")


def email_verification_max_age_seconds() -> int:
    raw = os.environ.get("EMAIL_VERIFICATION_TOKEN_MAX_AGE_SECONDS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            abort(503, description="Invalid EMAIL_VERIFICATION_TOKEN_MAX_AGE_SECONDS.")
        if value <= 0:
            abort(503, description="Invalid EMAIL_VERIFICATION_TOKEN_MAX_AGE_SECONDS.")
        return value
    return 60 * 60 * 24 * 7


def password_reset_serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("AUTH_SECRET_KEY")
    if not secret:
        abort(503, description="Auth is not configured (missing AUTH_SECRET_KEY).")
    return URLSafeTimedSerializer(secret_key=secret, salt="pandects-password-reset")


def password_reset_max_age_seconds() -> int:
    raw = os.environ.get("PASSWORD_RESET_TOKEN_MAX_AGE_SECONDS", "").strip()
    if raw:
        try:
            value = int(raw)
        except ValueError:
            abort(503, description="Invalid PASSWORD_RESET_TOKEN_MAX_AGE_SECONDS.")
        if value <= 0:
            abort(503, description="Invalid PASSWORD_RESET_TOKEN_MAX_AGE_SECONDS.")
        return value
    return 60 * 60


def issue_email_verification_token(*, user_id: str, email: str) -> str:
    serializer = email_verification_serializer()
    return serializer.dumps({"user_id": user_id, "email": email})


def load_email_verification_token(token: str) -> tuple[str, str] | None:
    serializer = email_verification_serializer()
    try:
        raw_payload_obj = cast(
            object, serializer.loads(token, max_age=email_verification_max_age_seconds())
        )
    except (BadSignature, SignatureExpired):
        return None
    if not isinstance(raw_payload_obj, dict):
        return None
    payload = cast(dict[str, object], raw_payload_obj)
    user_id = payload.get("user_id")
    email = payload.get("email")
    if not isinstance(user_id, str) or not isinstance(email, str):
        return None
    return user_id, email


def issue_password_reset_token(*, user_id: str, email: str) -> str:
    if auth_is_mocked():
        return _mock_auth.issue_password_reset_token(user_id=user_id, email=email)
    serializer = password_reset_serializer()
    reset_id = str(uuid.uuid4())
    now = _utc_now()
    expires_at = now + timedelta(seconds=password_reset_max_age_seconds())
    reset_token = AuthPasswordResetToken()
    reset_token.id = reset_id
    reset_token.user_id = user_id
    reset_token.created_at = now
    reset_token.expires_at = expires_at
    reset_token.ip_address = request_ip_address()
    reset_token.user_agent = request_user_agent()
    db.session.add(reset_token)
    db.session.commit()
    return serializer.dumps({"user_id": user_id, "email": email, "reset_id": reset_id})


def load_password_reset_token(
    token: str,
) -> tuple[str, str, AuthPasswordResetToken | None] | None:
    if auth_is_mocked():
        parsed = _mock_auth.consume_password_reset_token(token)
        if parsed is None:
            return None
        user_id, email = parsed
        return user_id, email, None
    serializer = password_reset_serializer()
    try:
        raw_payload_obj = cast(
            object, serializer.loads(token, max_age=password_reset_max_age_seconds())
        )
    except (BadSignature, SignatureExpired):
        return None
    if not isinstance(raw_payload_obj, dict):
        return None
    payload = cast(dict[str, object], raw_payload_obj)
    user_id = payload.get("user_id")
    email = payload.get("email")
    reset_id = payload.get("reset_id")
    if not isinstance(user_id, str) or not isinstance(email, str) or not isinstance(reset_id, str):
        return None
    try:
        row = cast(
            AuthPasswordResetToken | None,
            AuthPasswordResetToken.query.filter_by(id=reset_id, user_id=user_id, used_at=None)
            .first(),
        )
    except SQLAlchemyError:
        return None
    if row is None:
        return None
    expires_at = cast(object, row.expires_at)
    if not isinstance(expires_at, datetime) or expires_at <= _utc_now():
        return None
    return user_id, email, row


def resend_api_key() -> str | None:
    key = os.environ.get("RESEND_API_KEY")
    key = key.strip() if isinstance(key, str) else ""
    return key or None


def resend_from_email() -> str | None:
    sender = os.environ.get("RESEND_FROM_EMAIL")
    sender = sender.strip() if isinstance(sender, str) else ""
    if not sender:
        return None
    return sender


def resend_template_id() -> str | None:
    template_id = os.environ.get("RESEND_TEMPLATE_ID")
    template_id = template_id.strip() if isinstance(template_id, str) else ""
    return template_id or None


def resend_forgot_password_template_id() -> str | None:
    template_id = os.environ.get("RESEND_FORGOT_PASSWORD_TEMPLATE_ID")
    template_id = template_id.strip() if isinstance(template_id, str) else ""
    return template_id or "forgot-password"


def send_resend_text_email(*, to_email: str, subject: str, text: str) -> None:
    api_key = resend_api_key()
    sender = resend_from_email()
    if api_key is None or sender is None:
        if current_app.testing:
            return
        current_app.logger.warning(
            "Signup notification skipped (missing RESEND_API_KEY/RESEND_FROM_EMAIL)."
        )
        return

    if current_app.testing:
        return

    payload: dict[str, object] = {
        "from": sender,
        "to": [to_email],
        "subject": subject,
        "text": text,
        "headers": {"X-Entity-Ref-ID": uuid.uuid4().hex},
    }
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        "https://api.resend.com/emails",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "PandectsBackend/1.0 (+https://pandects.org)",
        },
        method="POST",
    )
    try:
        _ = _urlopen_read_bytes(req, timeout=15)
    except HTTPError as e:
        try:
            details = e.read().decode("utf-8", errors="replace")
        except Exception:
            details = ""
        current_app.logger.error(
            "Resend signup notification failed (HTTP %s): %s", e.code, details
        )
    except URLError as e:
        current_app.logger.error("Resend signup notification failed (network error): %s", e)


def send_resend_template_email(
    *,
    to_email: str,
    subject: str,
    variables: dict[str, object],
    template_id: str,
) -> None:
    api_key = resend_api_key()
    sender = resend_from_email()
    if api_key is None or sender is None:
        missing: list[str] = []
        if api_key is None:
            missing.append("RESEND_API_KEY")
        if sender is None:
            missing.append("RESEND_FROM_EMAIL")
        if current_app.testing:
            return
        abort(503, description=f"Email is not configured (missing {', '.join(missing)}).")

    if current_app.testing:
        return

    if not template_id:
        abort(503, description="Email is not configured (missing template id).")

    payload: dict[str, object] = {
        "from": sender,
        "to": [to_email],
        "subject": subject,
        "template": {
            "id": template_id,
            "variables": variables,
        },
        "headers": {"X-Entity-Ref-ID": uuid.uuid4().hex},
    }
    body = json.dumps(payload).encode("utf-8")
    req = Request(
        "https://api.resend.com/emails",
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "PandectsBackend/1.0 (+https://pandects.org)",
        },
        method="POST",
    )
    try:
        _ = _urlopen_read_bytes(req, timeout=15)
    except HTTPError as e:
        try:
            details = e.read().decode("utf-8", errors="replace")
        except Exception:
            details = ""
        current_app.logger.error("Resend email failed (HTTP %s): %s", e.code, details)
        abort(503, description="Email delivery failed.")
    except URLError as e:
        current_app.logger.error("Resend email failed (network error): %s", e)
        abort(503, description="Email delivery failed.")


def send_email_verification_email(*, to_email: str, token: str) -> None:
    verify_url = f"{frontend_base_url()}/auth/verify-email#token={quote(token)}"
    subject = "Verify your email for Pandects"
    year = str(_utc_now().year)
    template_id = resend_template_id()
    if template_id is None:
        abort(503, description="Email is not configured (missing RESEND_TEMPLATE_ID).")
    send_resend_template_email(
        to_email=to_email,
        subject=subject,
        variables={"VERIFY_URL": verify_url, "YEAR": year},
        template_id=template_id,
    )


def send_password_reset_email(*, to_email: str, token: str) -> None:
    reset_url = f"{frontend_base_url()}/auth/reset-password#token={quote(token)}"
    subject = "Reset your Pandects password"
    year = str(_utc_now().year)
    send_resend_template_email(
        to_email=to_email,
        subject=subject,
        variables={"RESET_URL": reset_url, "YEAR": year},
        template_id=resend_forgot_password_template_id() or "",
    )
