"""Maintainer notification emails: new sign-ups and user issue reports.

Both go to ``admin_notification_email()``. Delivery goes through Resend and is
a logged no-op when ``RESEND_API_KEY`` / ``RESEND_FROM_EMAIL`` are unset.
"""

from __future__ import annotations

import os
import re
from datetime import datetime

from backend.auth import email_runtime
from backend.extensions import db
from backend.models.main_db import XML, Sections, section_latest_xml_join_condition
from backend.services.side_effects import async_task_runner

# Recipient of sign-up and issue-report notifications. Defaults to the
# maintainer address that was hardcoded here before it became configurable.
_DEFAULT_ADMIN_NOTIFICATION_EMAIL = "nmbogdan@alumni.stanford.edu"

_UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")


def admin_notification_email() -> str:
    configured = os.environ.get("ADMIN_NOTIFICATION_EMAIL", "").strip()
    return configured or _DEFAULT_ADMIN_NOTIFICATION_EMAIL


def send_signup_notification_email(
    *,
    new_user_email: str,
    provider: str,
    signed_up_at: datetime,
    full_name: str | None = None,
) -> None:
    """Notify the maintainer of a sign-up and send the user's welcome email.

    Runs on the async side-effect queue when it is enabled and has room.
    """
    display_name = (
        full_name.strip() if isinstance(full_name, str) and full_name.strip() else "Not provided"
    )
    text = "\n".join(
        [
            "A new user just signed up on Pandects.",
            "",
            f"Full name: {display_name}",
            f"Email: {new_user_email}",
            f"Provider: {provider}",
            f"Signup date: {signed_up_at:%Y-%m-%d}",
            f"Signup time: {signed_up_at:%H:%M:%S} UTC",
        ]
    )
    recipient = admin_notification_email()

    def _send() -> None:
        email_runtime.send_resend_text_email(
            to_email=recipient,
            subject="New Pandects signup",
            text=text,
        )
        email_runtime.send_welcome_email(to_email=new_user_email, name=full_name)

    runner = async_task_runner()
    if runner is None or not runner.enqueue(_send):
        _send()


def send_flag_notification_email(
    *,
    user_email: str,
    submitted_at: datetime,
    source: str,
    agreement_uuid: str,
    section_uuid: str | None,
    message: str | None,
    request_follow_up: bool,
    issue_types: list[str],
) -> None:
    lines = [
        "Pandects issue report",
        "",
        f"User: {user_email}",
        f"Date/time (UTC): {submitted_at.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        f"Context: {source}",
        f"Agreement UUID: {agreement_uuid}",
    ]
    if section_uuid:
        lines.append(f"Section UUID: {section_uuid}")
    if issue_types:
        lines.extend(["", "Issue categories:", ", ".join(issue_types)])
    if message:
        lines.extend(["", "Report details:", message])
    else:
        lines.extend(["", "Report details: (none provided)"])
    lines.append(f"Request follow-up: {'Yes' if request_follow_up else 'No'}")
    email_runtime.send_resend_text_email(
        to_email=admin_notification_email(),
        subject="Pandects: Issue report",
        text="\n".join(lines),
    )


def is_agreement_section_eligible(agreement_uuid: str, section_uuid: str | None) -> bool:
    """True when an issue report targets a served agreement (and section)."""
    if not agreement_uuid or not _UUID_RE.match(agreement_uuid):
        return False
    if section_uuid is not None and (not section_uuid or not _UUID_RE.match(section_uuid)):
        return False
    q = (
        db.session.query(Sections.section_uuid)
        .join(XML, section_latest_xml_join_condition())
        .filter(Sections.agreement_uuid == agreement_uuid)
    )
    if section_uuid is not None:
        q = q.filter(Sections.section_uuid == section_uuid)
    return q.first() is not None


__all__ = [
    "admin_notification_email",
    "is_agreement_section_eligible",
    "send_flag_notification_email",
    "send_signup_notification_email",
]
