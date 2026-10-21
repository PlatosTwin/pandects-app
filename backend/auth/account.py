"""Account self-service: API key listings, usage rollups, issue-report input.

Functions take the explicit ``AuthDeps``. Client-facing input failures raise
``AccountInputError`` (the route answers 400 with its message); database
failures propagate as ``SQLAlchemyError``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import Any, cast

from backend.routes.deps import AuthDeps

# Server-side password policy. Zitadel enforces its own complexity rules;
# these bounds are defense-in-depth: a floor for new credentials and a cap
# so arbitrarily long inputs can't inflate password-hashing cost.
PASSWORD_MIN_LENGTH = 8
PASSWORD_MAX_LENGTH = 128
PASSWORD_LENGTH_MESSAGE = (
    f"Password must be between {PASSWORD_MIN_LENGTH} and {PASSWORD_MAX_LENGTH} characters."
)
PASSWORD_MAX_LENGTH_MESSAGE = f"Password must be at most {PASSWORD_MAX_LENGTH} characters."

USAGE_PERIOD_CUTOFF_DAYS: dict[str, int | None] = {
    "1w": 6,
    "1m": 29,
    "1y": 364,
    "all": None,
}

FLAG_ISSUE_TYPES = frozenset(
    {
        "Incorrect tagging (Article/Section)",
        "Corrupted formatting",
        "Incorrect taxonomy class",
        "Incorrect metadata",
        "Not an M&A agreement",
        "Something else",
    }
)
FLAG_SOURCES = ("search_result", "agreement_view")


class AccountInputError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message: str = message


# ── API keys ──────────────────────────────────────────────────────────────
def api_key_summary(key: Any) -> dict[str, object]:
    return {
        "id": key.id,
        "name": key.name,
        "prefix": key.prefix,
        "created_at": key.created_at.isoformat(),
        "last_used_at": key.last_used_at.isoformat() if key.last_used_at else None,
        "revoked_at": key.revoked_at.isoformat() if key.revoked_at else None,
    }


def created_api_key_payload(key: Any, plaintext: str) -> dict[str, object]:
    return {
        "api_key": {
            "id": key.id,
            "name": key.name,
            "prefix": key.prefix,
            "created_at": key.created_at.isoformat(),
        },
        "api_key_plaintext": plaintext,
    }


def normalize_api_key_name(name: object) -> str | None:
    if name is not None and not isinstance(name, str):
        raise AccountInputError("Key name must be a string.")
    if isinstance(name, str):
        stripped = name.strip() or None
        if stripped is not None and len(stripped) > 120:
            raise AccountInputError("Key name is too long.")
        return stripped
    return None


def list_api_keys(deps: AuthDeps, *, user_id: str) -> list[dict[str, object]]:
    """The user's keys, newest first; tombstoned keys are hidden."""
    if deps._auth_is_mocked():
        return [api_key_summary(k) for k in deps._mock_auth.list_api_keys(user_id=user_id)]
    keys = cast(
        list[Any],
        deps.ApiKey.query.filter_by(user_id=user_id, deleted_at=None)
        .order_by(deps.ApiKey.created_at.desc())
        .all(),
    )
    return [api_key_summary(k) for k in keys]


def revoke_api_key(deps: AuthDeps, *, user_id: str, key_id: str) -> bool:
    """Revoke one of the user's keys; False when it does not exist."""
    if deps._auth_is_mocked():
        return bool(deps._mock_auth.revoke_api_key(user_id=user_id, key_id=key_id))
    key = deps.ApiKey.query.filter_by(id=key_id, user_id=user_id).first()
    if key is None:
        return False
    if key.revoked_at is None:
        key.revoked_at = deps._utc_now()
        deps.db.session.commit()
    return True


def permanently_delete_api_key(deps: AuthDeps, *, user_id: str, key_id: str) -> bool:
    if deps._auth_is_mocked():
        return bool(deps._mock_auth.permanently_delete_api_key(user_id=user_id, key_id=key_id))
    return deps._permanently_delete_api_key(user_id=user_id, key_id=key_id)


# ── Usage ─────────────────────────────────────────────────────────────────
def usage_by_day(
    deps: AuthDeps,
    *,
    user_id: str,
    period: str,
    cutoff: date | None,
    api_key_id: str | None,
) -> dict[str, object] | None:
    """API request counts per day for the user's keys; None if ``api_key_id`` is not theirs."""
    if deps._auth_is_mocked():
        if api_key_id is not None:
            key_ids = {k.id for k in deps._mock_auth.list_api_keys(user_id=user_id)}
            if api_key_id not in key_ids:
                return None
        by_day_list, total_count = deps._mock_auth.usage_for_user(
            user_id=user_id,
            period=period,
            api_key_id=api_key_id,
        )
        return {"by_day": by_day_list, "total": total_count}

    if api_key_id is not None:
        key = deps.ApiKey.query.filter_by(user_id=user_id, id=api_key_id, deleted_at=None).first()
        if key is None:
            return None
        key_ids = [key.id]
    else:
        key_ids = [k.id for k in deps.ApiKey.query.filter_by(user_id=user_id).all()]
    if not key_ids:
        return {"by_day": [], "total": 0}

    rows_query = deps.ApiUsageDaily.query.filter(deps.ApiUsageDaily.api_key_id.in_(key_ids))
    if cutoff is not None:
        rows_query = rows_query.filter(deps.ApiUsageDaily.day >= cutoff)
    rows = cast(list[Any], rows_query.order_by(deps.ApiUsageDaily.day.asc()).all())
    by_day: dict[str, int] = defaultdict(int)
    total = 0
    for row in rows:
        by_day[row.day.isoformat()] += int(row.count)
        total += int(row.count)
    return {
        "by_day": [{"day": day, "count": by_day[day]} for day in sorted(by_day)],
        "total": total,
    }


# ── Issue reports ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FlagReport:
    source: str
    agreement_uuid: str
    section_uuid: str | None
    message: str | None
    request_follow_up: bool
    issue_types: list[str]


def parse_flag_report(data: dict[str, object]) -> FlagReport:
    """Validate a /flag-inaccurate body. Checks run in the order the API documents."""
    source_raw = data.get("source")
    source = source_raw.strip() if isinstance(source_raw, str) else ""
    agreement_uuid_raw = data.get("agreement_uuid")
    agreement_uuid = agreement_uuid_raw.strip() if isinstance(agreement_uuid_raw, str) else ""
    raw_message = data.get("message")
    message = raw_message.strip() if isinstance(raw_message, str) else None
    section_uuid_raw = data.get("section_uuid")
    section_uuid = section_uuid_raw.strip() if isinstance(section_uuid_raw, str) else None
    request_follow_up = data.get("request_follow_up")
    if request_follow_up is None:
        request_follow_up = False
    if not isinstance(request_follow_up, bool):
        raise AccountInputError("request_follow_up must be a boolean.")
    raw_issue_types = data.get("issue_types")
    if raw_issue_types is None:
        raw_issue_types = []
    if not isinstance(raw_issue_types, list) or not all(
        isinstance(item, str) for item in cast(list[object], raw_issue_types)
    ):
        raise AccountInputError("issue_types must be a list of strings.")
    issue_types = [item.strip() for item in cast(list[str], raw_issue_types) if item and item.strip()]
    if not issue_types:
        raise AccountInputError("issue_types is required.")
    if any(item not in FLAG_ISSUE_TYPES for item in issue_types):
        raise AccountInputError("issue_types contains an invalid value.")
    if source not in FLAG_SOURCES:
        raise AccountInputError("Invalid source. Use 'search_result' or 'agreement_view'.")
    if not agreement_uuid:
        raise AccountInputError("agreement_uuid is required.")
    if source == "search_result" and not section_uuid:
        raise AccountInputError("section_uuid is required when source is 'search_result'.")
    if source == "agreement_view":
        section_uuid = None
    return FlagReport(
        source=source,
        agreement_uuid=agreement_uuid,
        section_uuid=section_uuid,
        message=message,
        request_follow_up=request_follow_up,
        issue_types=issue_types,
    )


__all__ = [
    "AccountInputError",
    "FLAG_ISSUE_TYPES",
    "FlagReport",
    "PASSWORD_LENGTH_MESSAGE",
    "PASSWORD_MAX_LENGTH",
    "PASSWORD_MAX_LENGTH_MESSAGE",
    "PASSWORD_MIN_LENGTH",
    "USAGE_PERIOD_CUTOFF_DAYS",
    "api_key_summary",
    "created_api_key_payload",
    "list_api_keys",
    "normalize_api_key_name",
    "parse_flag_report",
    "permanently_delete_api_key",
    "revoke_api_key",
    "usage_by_day",
]
