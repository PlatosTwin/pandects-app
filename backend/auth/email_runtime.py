from __future__ import annotations

import hashlib
import hmac
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request

from flask import abort, current_app

from backend.core.runtime_utils import urlopen_read_bytes as _urlopen_read_bytes


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


def resend_from_name() -> str:
    sender_name = os.environ.get("RESEND_FROM_NAME")
    sender_name = sender_name.strip() if isinstance(sender_name, str) else ""
    return sender_name or "Pandects"


def resend_sender() -> str | None:
    sender = resend_from_email()
    if sender is None:
        return None
    if "<" in sender and ">" in sender:
        return sender
    return f"{resend_from_name()} <{sender}>"


def zitadel_notification_signing_key() -> str | None:
    key = os.environ.get("AUTH_ZITADEL_NOTIFICATION_SIGNING_KEY")
    key = key.strip() if isinstance(key, str) else ""
    return key or None


def verify_zitadel_signature(*, raw_body: bytes, signature_header: str | None) -> bool:
    signing_key = zitadel_notification_signing_key()
    if signing_key is None:
        current_app.logger.warning(
            "ZITADEL notification signature verification is unavailable (missing AUTH_ZITADEL_NOTIFICATION_SIGNING_KEY)."
        )
        return False
    if not isinstance(signature_header, str) or not signature_header.strip():
        return False

    parts: dict[str, str] = {}
    for piece in signature_header.split(","):
        key, sep, value = piece.strip().partition("=")
        if not sep or not key or not value:
            continue
        parts[key] = value

    timestamp = parts.get("t")
    signature = parts.get("v1")
    if not isinstance(timestamp, str) or not isinstance(signature, str):
        return False

    try:
        timestamp_int = int(timestamp)
    except ValueError:
        return False

    max_age_raw = os.environ.get("AUTH_ZITADEL_NOTIFICATION_MAX_AGE_SECONDS", "").strip()
    try:
        max_age_seconds = int(max_age_raw) if max_age_raw else 300
    except ValueError:
        max_age_seconds = 300

    now = int(time.time())
    if abs(now - timestamp_int) > max_age_seconds:
        return False

    signed_payload = timestamp.encode("utf-8") + b"." + raw_body
    computed = hmac.new(
        signing_key.encode("utf-8"),
        signed_payload,
        hashlib.sha256,
    ).hexdigest()
    return hmac.compare_digest(computed, signature)


def _emails_workspace_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "emails"


def render_react_email_template(*, template_name: str, props: dict[str, str]) -> str:
    emails_dir = _emails_workspace_dir()
    tsx_bin = emails_dir / "node_modules" / ".bin" / "tsx"
    script_path = emails_dir / "scripts" / "render-template.ts"
    if not tsx_bin.exists() or not script_path.exists():
        abort(503, description="Email rendering is not configured on this server.")

    try:
        proc = subprocess.run(
            [str(tsx_bin), str(script_path), template_name],
            cwd=str(emails_dir),
            input=json.dumps(props),
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        current_app.logger.exception("React Email render failed to start.")
        abort(503, description="Email rendering is unavailable right now.")

    if proc.returncode != 0:
        current_app.logger.error(
            "React Email render failed for %s: %s",
            template_name,
            proc.stderr.strip(),
        )
        abort(503, description="Email rendering is unavailable right now.")
    html = proc.stdout.strip()
    if not html:
        abort(503, description="Email rendering is unavailable right now.")
    return html


def _send_resend_email(
    *,
    to_email: str,
    subject: str,
    text: str | None = None,
    html: str | None = None,
    raise_on_error: bool,
) -> None:
    api_key = resend_api_key()
    sender = resend_sender()
    if api_key is None or sender is None:
        if current_app.testing:
            return
        current_app.logger.warning(
            "Email send skipped (missing RESEND_API_KEY/RESEND_FROM_EMAIL)."
        )
        return

    if current_app.testing:
        return

    payload: dict[str, object] = {
        "from": sender,
        "to": [to_email],
        "subject": subject,
        "headers": {"X-Entity-Ref-ID": uuid.uuid4().hex},
    }
    if isinstance(text, str) and text.strip():
        payload["text"] = text
    if isinstance(html, str) and html.strip():
        payload["html"] = html

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
            "Resend email delivery failed (HTTP %s): %s", e.code, details
        )
        if raise_on_error:
            raise
    except URLError as e:
        current_app.logger.error("Resend email delivery failed (network error): %s", e)
        if raise_on_error:
            raise


def send_resend_text_email(*, to_email: str, subject: str, text: str) -> None:
    _send_resend_email(to_email=to_email, subject=subject, text=text, raise_on_error=False)


def send_resend_html_email(*, to_email: str, subject: str, html: str, text: str | None = None) -> None:
    _send_resend_email(
        to_email=to_email,
        subject=subject,
        html=html,
        text=text,
        raise_on_error=True,
    )


def send_pandects_auth_email(
    *,
    notification_type: str,
    to_email: str,
    action_url: str,
    code: str | None = None,
) -> None:
    year = str(time.gmtime().tm_year)
    if notification_type == "verify-email":
        subject = "Verify your email"
        html = render_react_email_template(
            template_name="verify-email",
            props={"VERIFY_URL": action_url, "YEAR": year},
        )
        text = (
            "Verify your email\n\n"
            "Thanks for signing up. Verify your email to gain access to the API "
            "and unlock full search results.\n\n"
            f"Verify email: {action_url}\n\n"
            "If you didn't request this email, you can safely ignore it.\n"
            "Questions? Email nmbogdan@alumni.stanford.edu."
        )
    elif notification_type == "reset-password":
        subject = "Reset your password"
        html = render_react_email_template(
            template_name="reset-password",
            props={"RESET_URL": action_url, "YEAR": year},
        )
        text = (
            "Reset your password\n\n"
            "We received a request to reset your password. Use the link below to choose a new one.\n\n"
            f"Reset password: {action_url}\n\n"
            "If you didn't request this email, you can safely ignore it.\n"
            "Questions? Email nmbogdan@alumni.stanford.edu."
        )
    else:
        abort(400, description="Unsupported auth email notification.")

    if isinstance(code, str) and code.strip():
        text = f"{text}\n\nVerification code: {code.strip()}"
    send_resend_html_email(to_email=to_email, subject=subject, html=html, text=text)
