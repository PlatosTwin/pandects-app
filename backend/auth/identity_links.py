"""Linking external (Zitadel / OIDC) identities to local auth users.

Functions take the explicit ``AuthDeps``; ``backend/routes/auth`` binds them
once per app. Also holds the pure parsers for Zitadel notification webhooks.
"""

from __future__ import annotations

from typing import Any, cast

from flask import abort

from backend.routes.deps import AuthDeps


def _first_dict(obj: object, *keys: str) -> dict[str, object] | None:
    if not isinstance(obj, dict):
        return None
    mapping = cast(dict[str, object], obj)
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, dict):
            return cast(dict[str, object], value)
    return None


def _first_string(obj: object, *keys: str) -> str | None:
    if not isinstance(obj, dict):
        return None
    mapping = cast(dict[str, object], obj)
    for key in keys:
        value = mapping.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def parse_auth_notification(
    payload: dict[str, object],
) -> tuple[str, str, str, str | None] | None:
    """Zitadel email notification -> ``(kind, recipient, action_url, code)``.

    ``kind`` is ``"verify-email"`` or ``"reset-password"``; anything else is
    not ours to deliver and yields None.
    """
    template_data = _first_dict(payload, "templateData", "template_data")
    context_info = _first_dict(payload, "contextInfo", "context_info")
    args = _first_dict(payload, "args")

    action_url = _first_string(template_data, "url", "link")
    recipient = _first_string(
        payload,
        "recipientEmail",
        "recipient_email",
        "recipient",
    ) or _first_string(context_info, "recipientEmailAddress", "recipient_email_address")
    code = _first_string(args, "code", "Code")

    if action_url is None or recipient is None:
        return None
    if "/verify-email" in action_url:
        return ("verify-email", recipient, action_url, code)
    if "/reset-password/confirm" in action_url:
        return ("reset-password", recipient, action_url, code)
    return None


def full_name_from_parts(*parts: object) -> str | None:
    full_name = " ".join(part.strip() for part in parts if isinstance(part, str) and part.strip())
    return full_name or None


def external_link_payload(*, link: Any, provider_name: str) -> dict[str, object]:
    return {
        "id": link.id,
        "provider": provider_name,
        "issuer": link.issuer,
        "subject": link.subject,
        "created_at": link.created_at.isoformat(),
    }


def local_user_by_email(deps: AuthDeps, *, email: str) -> Any:
    return deps.AuthUser.query.filter_by(email=email).first()


def linked_user_for_subject(deps: AuthDeps, *, issuer: str, subject: str) -> Any:
    existing_link = deps.AuthExternalSubject.query.filter_by(
        issuer=issuer,
        subject=subject,
    ).first()
    if existing_link is None:
        return None
    return deps.db.session.get(deps.AuthUser, existing_link.user_id)


def link_external_identity_for_user(
    deps: AuthDeps, *, user: Any, provider_name: str, external_identity: Any
) -> tuple[str, Any, int]:
    """Attach an external identity to ``user``: ``(status, link, http_status)``.

    409 when the identity already belongs to a different account.
    """
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


def external_identity_verified_email(deps: AuthDeps, external_identity: Any) -> str | None:
    """The identity's email when the provider marked it verified, else None."""
    claims = cast(object, getattr(external_identity, "claims", None))
    if not isinstance(claims, dict):
        return None
    claims_map = cast(dict[str, object], claims)
    email = claims_map.get("email")
    if not isinstance(email, str) or not email.strip():
        return None
    if claims_map.get("email_verified") is not True:
        return None
    normalized = deps._normalize_email(email)
    if not deps._is_email_like(normalized):
        return None
    return normalized


def resolve_user_from_external_identity(
    deps: AuthDeps, *, external_identity: Any
) -> tuple[str, Any]:
    """Find or create the local user for an external identity: ``(action, user)``.

    ``action`` is ``"login"`` or ``"register"``. Adds the subject link when
    missing; the caller commits.
    """
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

    email = external_identity_verified_email(deps, external_identity)
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


__all__ = [
    "external_identity_verified_email",
    "external_link_payload",
    "full_name_from_parts",
    "link_external_identity_for_user",
    "linked_user_for_subject",
    "local_user_by_email",
    "parse_auth_notification",
    "resolve_user_from_external_identity",
]
