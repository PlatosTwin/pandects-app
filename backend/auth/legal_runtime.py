from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import cast

from flask import abort
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.sql.elements import ColumnElement

from backend.auth.session_runtime import auth_is_mocked, request_ip_address, request_user_agent
from backend.core.errors import json_error as _json_error
from backend.core.runtime_utils import utc_datetime_from_ms as _utc_datetime_from_ms
from backend.core.runtime_utils import utc_now as _utc_now
from backend.extensions import db
from backend.models import AuthSignonEvent, LegalAcceptance


_LEGAL_DOCS: dict[str, dict[str, str]] = {
    "tos": {
        "version": "2026-03-11",
        "sha256": "a8ec1487d9473166a8172e086cfd3a3e370c7c3200efdf6936d89660dce2b04b",
    },
    "privacy": {
        "version": "2026-03-11",
        "sha256": "ec056d325ada8c7801c82e9d3555098c7e0a3255ad4da82aaef8942292c6cd9d",
    },
    "license": {
        "version": "2026-03-11",
        "sha256": "fa26c2cc00d31f7385f2ff40a75e9a1bce7c7e0b2b646f8db1884d6e239ff954",
    },
}


def auth_enumeration_delay(*, sleeper: Callable[[], None]) -> None:
    sleeper()


def require_legal_acceptance(data: dict[str, object]) -> datetime:
    legal = data.get("legal")
    if not isinstance(legal, dict):
        abort(
            _json_error(
                412,
                error="legal_required",
                message="Legal acceptance required to create an account.",
            )
        )
    legal_dict = cast(dict[str, object], legal)
    checked_at_ms = legal_dict.get("checked_at_ms")
    checked_at = _utc_datetime_from_ms(checked_at_ms, field="legal.checked_at_ms")
    docs = legal_dict.get("docs")
    if not isinstance(docs, list):
        abort(400, description="legal.docs must be an array.")
    docs_list = cast(list[object], docs)
    normalized: list[str] = []
    for doc in docs_list:
        if not isinstance(doc, str):
            abort(400, description="legal.docs must contain strings.")
        normalized.append(doc.strip().lower())
    if set(normalized) != {"tos", "privacy", "license"}:
        abort(400, description="legal.docs must include tos, privacy, and license.")
    return checked_at


def legal_acceptance_columns() -> tuple[
    ColumnElement[str],
    ColumnElement[str],
    ColumnElement[str],
    ColumnElement[str],
]:
    return (
        cast(ColumnElement[str], LegalAcceptance.document),
        cast(ColumnElement[str], LegalAcceptance.version),
        cast(ColumnElement[str], LegalAcceptance.document_hash),
        cast(ColumnElement[str], LegalAcceptance.user_id),
    )


def user_has_current_legal_acceptances(*, user_id: str) -> bool:
    expected_rows = {(doc, meta["version"], meta["sha256"]) for doc, meta in _LEGAL_DOCS.items()}
    document_col, version_col, document_hash_col, user_id_col = legal_acceptance_columns()
    try:
        rows_raw = cast(
            list[tuple[object, object, object]],
            db.session.execute(
                select(
                    document_col,
                    version_col,
                    document_hash_col,
                ).where(
                    user_id_col == user_id,
                    document_col.in_(list(_LEGAL_DOCS.keys())),
                )
            ).all(),
        )
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")
    found: set[tuple[str, str, str]] = set()
    for doc_raw, ver_raw, hash_raw in rows_raw:
        if not isinstance(doc_raw, str) or not isinstance(ver_raw, str):
            continue
        hash_value = hash_raw if isinstance(hash_raw, str) else ""
        found.add((doc_raw, ver_raw, hash_value.strip()))
    return expected_rows.issubset(found)


def ensure_current_legal_acceptances(*, user_id: str, checked_at: datetime) -> None:
    now = _utc_now()
    ip_address = request_ip_address()
    user_agent = request_user_agent()
    document_col, version_col, document_hash_col, user_id_col = legal_acceptance_columns()
    try:
        existing_rows = cast(
            list[tuple[object, object, object]],
            db.session.execute(
                select(
                    document_col,
                    version_col,
                    document_hash_col,
                ).where(
                    user_id_col == user_id,
                    document_col.in_(list(_LEGAL_DOCS.keys())),
                )
            ).all(),
        )
        existing_set: set[tuple[str, str, str]] = set()
        for doc_raw, ver_raw, hash_raw in existing_rows:
            if not isinstance(doc_raw, str) or not isinstance(ver_raw, str):
                continue
            hash_value = hash_raw if isinstance(hash_raw, str) else ""
            existing_set.add((doc_raw, ver_raw, hash_value.strip()))
        for doc, meta in _LEGAL_DOCS.items():
            key = (doc, meta["version"], meta["sha256"])
            if key in existing_set:
                continue
            acceptance = LegalAcceptance()
            acceptance.user_id = user_id
            acceptance.document = doc
            acceptance.version = meta["version"]
            acceptance.document_hash = meta["sha256"]
            acceptance.checked_at = checked_at
            acceptance.submitted_at = now
            acceptance.ip_address = ip_address
            acceptance.user_agent = user_agent
            db.session.add(acceptance)
    except SQLAlchemyError:
        abort(503, description="Auth backend is unavailable right now.")


def record_signon_event(*, user_id: str, provider: str, action: str) -> None:
    if auth_is_mocked():
        return
    ip_address = request_ip_address()
    user_agent = request_user_agent()
    event = AuthSignonEvent()
    event.user_id = user_id
    event.provider = provider
    event.action = action
    event.ip_address = ip_address
    event.user_agent = user_agent
    db.session.add(event)
