"""First-party telemetry routes.

POST /v1/page-views records SPA route changes into the ``page_views`` table
the usage-analytics dashboard reads. Fire-and-forget by design: the endpoint
always returns 204 so a telemetry hiccup can never surface in the UI, and it
is CSRF-exempt (see ``backend.auth.session_runtime.csrf_required``) because
it performs no security-relevant state change and keeping it header-free
lets the client fall back to ``navigator.sendBeacon`` on page unload.

Stored values are display strings the analytics dashboard will render, so
control characters are stripped and the referrer must be an http(s) URL —
but any dashboard rendering ``path``/``referrer`` must still HTML-escape.
"""

from __future__ import annotations

import logging

from flask import Blueprint, Response, g, make_response, request
from sqlalchemy.exc import SQLAlchemyError

from backend.auth.session_runtime import auth_is_mocked
from backend.extensions import db
from backend.models import PageView
from backend.services.usage import request_country

logger = logging.getLogger(__name__)

_MAX_PATH_LENGTH = 512
_MAX_REFERRER_LENGTH = 512


def _clean_display_string(value: str) -> str:
    """Drop control characters (NUL breaks the Postgres insert; the rest are
    log/terminal-injection noise)."""
    return "".join(ch for ch in value if ord(ch) >= 32).strip()


def register_telemetry_routes() -> Blueprint:
    blp = Blueprint("telemetry", __name__)

    @blp.route("/v1/page-views", methods=["POST"])
    def record_page_view() -> Response:
        no_content = make_response("", 204)
        no_content.headers["Cache-Control"] = "no-store"
        if auth_is_mocked():
            return no_content
        payload = request.get_json(silent=True, force=True)
        if not isinstance(payload, dict):
            return no_content
        raw_path = payload.get("path")
        if not isinstance(raw_path, str):
            return no_content
        path = _clean_display_string(raw_path)[:_MAX_PATH_LENGTH]
        if not path.startswith("/") or path.startswith("//"):
            return no_content
        raw_referrer = payload.get("referrer")
        referrer = None
        if isinstance(raw_referrer, str):
            cleaned = _clean_display_string(raw_referrer)[:_MAX_REFERRER_LENGTH]
            if cleaned.startswith("http://") or cleaned.startswith("https://"):
                referrer = cleaned
        ctx = getattr(g, "access_ctx", None)
        user_id = getattr(ctx, "user_id", None)
        view = PageView()
        view.path = path
        view.user_id = user_id if isinstance(user_id, str) and user_id else None
        view.referrer = referrer
        view.country = request_country()
        try:
            db.session.add(view)
            db.session.commit()
        except SQLAlchemyError:
            db.session.rollback()
            logger.warning("page_view_insert_failed", exc_info=True)
        return no_content

    return blp
