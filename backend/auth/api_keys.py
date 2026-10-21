"""API key issuance, lookup, and last-used bookkeeping.

Keys look like ``pdcts_<64 hex>``; only a werkzeug hash and the first 18
characters (the lookup prefix) are stored.
"""

from __future__ import annotations

import os
import time
import uuid
from threading import Lock
from typing import cast

from flask import g
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.security import check_password_hash, generate_password_hash

from backend.auth.session_runtime import auth_is_mocked
from backend.core.runtime_utils import utc_now
from backend.extensions import db
from backend.models import ApiKey

API_KEY_PREFIX = "pdcts_"
_PREFIX_LENGTH = len(API_KEY_PREFIX) + 12
# Every lookup runs at least this many hash checks so response time does not
# reveal whether a prefix matched.
API_KEY_MIN_HASH_CHECKS = 5
DUMMY_API_KEY_HASH = generate_password_hash("pdcts_dummy_api_key")

API_KEY_LAST_USED_TOUCH_SECONDS = int(os.environ.get("API_KEY_LAST_USED_TOUCH_SECONDS", "300"))
API_KEY_LAST_USED_MAX_KEYS = int(os.environ.get("API_KEY_LAST_USED_MAX_KEYS", "50000"))
api_key_last_used_touch_lock = Lock()
api_key_last_used_touch_state: dict[str, float] = {}


def should_touch_api_key_last_used(key_id: str) -> bool:
    """Throttle ``last_used_at`` writes to one per key per touch interval."""
    interval_seconds = float(max(0, API_KEY_LAST_USED_TOUCH_SECONDS))
    now = time.time()
    with api_key_last_used_touch_lock:
        if interval_seconds > 0:
            cutoff = now - (interval_seconds * 2.0)
            stale_keys = [
                existing_key
                for existing_key, touched_at in api_key_last_used_touch_state.items()
                if touched_at < cutoff
            ]
            for stale_key in stale_keys:
                _ = api_key_last_used_touch_state.pop(stale_key, None)
        overflow = len(api_key_last_used_touch_state) - API_KEY_LAST_USED_MAX_KEYS
        if overflow > 0:
            oldest_keys = [
                existing_key
                for existing_key, _touched_at in sorted(
                    api_key_last_used_touch_state.items(),
                    key=lambda item: item[1],
                )[:overflow]
            ]
            for oldest_key in oldest_keys:
                _ = api_key_last_used_touch_state.pop(oldest_key, None)
        last_touched_at = api_key_last_used_touch_state.get(key_id)
        if interval_seconds > 0 and last_touched_at is not None:
            if (now - last_touched_at) < interval_seconds:
                return False
        api_key_last_used_touch_state[key_id] = now
        return True


def touch_api_key_last_used_if_needed() -> None:
    """after_request: persist ``last_used_at`` for a mock-auth key seen on this request."""
    candidate_id = cast(object, getattr(g, "api_key_last_used_candidate_id", None))
    if not isinstance(candidate_id, str) or not candidate_id:
        return
    if not should_touch_api_key_last_used(candidate_id):
        return
    try:
        _ = ApiKey.query.filter_by(id=candidate_id, revoked_at=None).update(
            {"last_used_at": utc_now()},
            synchronize_session=False,
        )
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()


def lookup_api_key(raw_key: str) -> ApiKey | None:
    if auth_is_mocked():
        return None
    raw_key = raw_key.strip()
    if not raw_key.startswith(API_KEY_PREFIX):
        return None
    prefix = raw_key[:_PREFIX_LENGTH]
    candidates = cast(
        list[ApiKey],
        ApiKey.query.filter_by(prefix=prefix, revoked_at=None, deleted_at=None).limit(25).all(),
    )
    checks = 0
    for candidate in candidates:
        checks += 1
        key_hash = cast(object, candidate.key_hash)
        if isinstance(key_hash, str) and check_password_hash(key_hash, raw_key):
            candidate_id = cast(object, candidate.id)
            if candidate_id is not None:
                if should_touch_api_key_last_used(str(candidate_id)):
                    try:
                        candidate.last_used_at = utc_now()
                        db.session.commit()
                    except SQLAlchemyError:
                        db.session.rollback()
            return candidate
    for _ in range(max(0, API_KEY_MIN_HASH_CHECKS - checks)):
        _ = check_password_hash(DUMMY_API_KEY_HASH, raw_key)
    return None


def create_api_key(*, user_id: str, name: str | None) -> tuple[ApiKey, str]:
    token = f"{API_KEY_PREFIX}{uuid.uuid4().hex}{uuid.uuid4().hex}"
    key = ApiKey()
    key.user_id = user_id
    key.name = name.strip() if isinstance(name, str) and name.strip() else None
    key.prefix = token[:_PREFIX_LENGTH]
    key.key_hash = generate_password_hash(token)
    db.session.add(key)
    db.session.commit()
    return key, token


def permanently_delete_api_key(*, user_id: str, key_id: str) -> bool:
    """Tombstone an API key while retaining usage rows that reference it."""
    try:
        key = ApiKey.query.filter_by(id=key_id, user_id=user_id).first()
        if key is None:
            return False
        if key.deleted_at is not None:
            return True
        now = utc_now()
        key.deleted_at = now
        if key.revoked_at is None:
            key.revoked_at = now
        key.key_hash = DUMMY_API_KEY_HASH
        key.name = None
        db.session.commit()
        return True
    except SQLAlchemyError:
        db.session.rollback()
        raise


__all__ = [
    "API_KEY_LAST_USED_MAX_KEYS",
    "API_KEY_LAST_USED_TOUCH_SECONDS",
    "API_KEY_MIN_HASH_CHECKS",
    "API_KEY_PREFIX",
    "DUMMY_API_KEY_HASH",
    "api_key_last_used_touch_state",
    "create_api_key",
    "lookup_api_key",
    "permanently_delete_api_key",
    "should_touch_api_key_last_used",
    "touch_api_key_last_used_if_needed",
]
