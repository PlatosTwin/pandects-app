"""First-party telemetry routes.

POST /v1/page-views records SPA route changes into the ``page_views`` table
the usage-analytics dashboard reads. Fire-and-forget by design: the endpoint
always returns 204 so a telemetry hiccup can never surface in the UI, and it
is CSRF-exempt (see ``backend.auth.session_runtime.csrf_required``) because
it performs no security-relevant state change and the SPA sends it via
``sendBeacon``/keepalive fetch which cannot attach custom headers reliably.
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
        path = raw_path.strip()[:_MAX_PATH_LENGTH]
        if not path.startswith("/") or path.startswith("//"):
            return no_content
        raw_referrer = payload.get("referrer")
        referrer = (
            raw_referrer.strip()[:_MAX_REFERRER_LENGTH]
            if isinstance(raw_referrer, str) and raw_referrer.strip()
            else None
        )
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
