from __future__ import annotations

from typing import cast

from flask import Flask, Response, current_app, jsonify, make_response, request
from marshmallow import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.exceptions import HTTPException, InternalServerError
from werkzeug.http import HTTP_STATUS_CODES

from backend.extensions import db

# Every /v1 error response has the envelope {"error": <code>, "message": <text>}.
# `error` is a stable snake_case machine code clients may branch on; `message`
# is human-readable and may change wording. Routes with a more specific
# condition pass their own code through `json_error` (for example
# `captcha_required`, `legal_required`, `rate_limited`); everything else gets
# the code for its HTTP status. `/mcp` keeps JSON-RPC error objects, and the
# OAuth token and client-registration endpoints use `oauth_error`, whose
# `error` values are the codes those RFCs define.
ERROR_CODES_BY_STATUS: dict[int, str] = {
    400: "bad_request",
    401: "unauthorized",
    403: "forbidden",
    404: "not_found",
    405: "method_not_allowed",
    409: "conflict",
    413: "payload_too_large",
    415: "unsupported_media_type",
    422: "validation_error",
    429: "rate_limited",
    500: "internal_error",
    502: "bad_gateway",
    503: "service_unavailable",
    504: "gateway_timeout",
}


def error_code_for_status(status: int) -> str:
    known = ERROR_CODES_BY_STATUS.get(status)
    if known is not None:
        return known
    reason = HTTP_STATUS_CODES.get(status, "")
    slug = "_".join(reason.lower().replace("-", " ").replace("'", "").split())
    return slug or ("internal_error" if status >= 500 else "bad_request")


def _flatten_validation_messages(messages: object) -> str:
    if isinstance(messages, dict):
        return " ".join(
            _flatten_validation_messages(value)
            for value in cast(dict[object, object], messages).values()
        )
    if isinstance(messages, (list, tuple)):
        return " ".join(str(item) for item in cast(list[object], messages))
    return str(messages)


def handle_validation_error(err: ValidationError):
    if request.path.startswith("/v1/"):
        resp = jsonify(
            {
                "error": error_code_for_status(422),
                "message": _flatten_validation_messages(err.messages),
            }
        )
        resp.status_code = 422
        return resp
    raise err


def _http_exception_message(err: HTTPException) -> str:
    # flask-smorest's request-argument parser aborts 422 with the field errors
    # in `data["messages"]` and a generic description; surface the fields.
    data = cast(object, getattr(err, "data", None))
    if isinstance(data, dict):
        messages = cast(dict[str, object], data).get("messages")
        if isinstance(messages, dict) and messages:
            return _field_validation_messages(cast(dict[object, object], messages))
    return err.description or (err.name or "")


_REQUEST_LOCATIONS = frozenset({"query", "json", "form", "headers", "cookies", "files", "view_args", "path"})


def _field_validation_messages(messages: dict[object, object], *, prefix: str = "") -> str:
    """`{"query": {"page_size": ["Not a valid integer."]}}` -> `page_size: Not a valid integer.`"""
    parts: list[str] = []
    for key, value in messages.items():
        name = str(key)
        if not prefix and name in _REQUEST_LOCATIONS and isinstance(value, dict):
            parts.append(_field_validation_messages(cast(dict[object, object], value)))
            continue
        if not prefix:
            field = name
        elif isinstance(key, int):
            field = f"{prefix}[{name}]"
        else:
            field = f"{prefix}.{name}"
        if isinstance(value, dict):
            parts.append(_field_validation_messages(cast(dict[object, object], value), prefix=field))
        else:
            parts.append(f"{field}: {_flatten_validation_messages(value)}")
    return " ".join(part for part in parts if part)


def handle_http_exception(err: HTTPException):
    if request.path.startswith("/v1/"):
        status = err.code or 500
        resp = jsonify(
            {"error": error_code_for_status(status), "message": _http_exception_message(err)}
        )
        resp.status_code = status
        # Keep headers the exception sets (Retry-After on a 503, Allow on a 405).
        for name, value in err.get_headers():
            if name.lower() != "content-type":
                resp.headers[name] = value
        return resp
    return err


def handle_internal_server_error(err: InternalServerError):
    if request.path.startswith("/v1/"):
        current_app.logger.exception("Unhandled API exception: %s", err)
        resp = jsonify(
            {"error": error_code_for_status(500), "message": "Unexpected server error."}
        )
        resp.status_code = 500
        return resp
    return err


def handle_sqlalchemy_error(err: SQLAlchemyError):
    db.session.rollback()
    if request.path.startswith("/v1/"):
        current_app.logger.exception("Database error: %s", err)
        resp = jsonify(
            {"error": error_code_for_status(503), "message": "Database is unavailable."}
        )
        resp.status_code = 503
        return resp
    if request.path == "/mcp":
        current_app.logger.exception("MCP database error: %s", err)
        raw_payload = request.get_json(silent=True)
        request_id = (
            cast(dict[str, object], raw_payload).get("id")
            if isinstance(raw_payload, dict)
            else None
        )
        resp = jsonify(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32004,
                    "message": "Database is unavailable.",
                    "data": {"category": "database", "status_code": 503},
                },
            }
        )
        resp.status_code = 503
        return resp
    raise err


def register_error_handlers(target_app: Flask) -> None:
    target_app.register_error_handler(HTTPException, handle_http_exception)
    target_app.register_error_handler(InternalServerError, handle_internal_server_error)
    target_app.register_error_handler(SQLAlchemyError, handle_sqlalchemy_error)
    target_app.register_error_handler(ValidationError, handle_validation_error)


def json_error(
    status: int, *, error: str, message: str, headers: dict[str, str] | None = None
) -> Response:
    resp = make_response(jsonify({"error": error, "message": message}), status)
    if headers:
        resp.headers.update(headers)
    return resp


def status_response(status: str, *, code: int = 200) -> Response:
    return make_response(jsonify({"status": status}), code)


def oauth_error(status: int, *, error: str, description: str) -> Response:
    """Error body for the OAuth token (RFC 6749 §5.2) and client-registration
    (RFC 7591 §3.2.2) endpoints.

    `error` is the code the RFC defines (`invalid_grant`, `invalid_client_metadata`,
    ...) and `error_description` its text; `message` repeats the text so the
    body is still a /v1 envelope.
    """
    resp = make_response(
        jsonify({"error": error, "error_description": description, "message": description}),
        status,
    )
    resp.headers["Cache-Control"] = "no-store"
    resp.headers["Pragma"] = "no-cache"
    return resp
