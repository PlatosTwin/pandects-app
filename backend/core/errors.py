from __future__ import annotations

from typing import cast

from flask import Flask, Response, current_app, jsonify, make_response, request
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.exceptions import HTTPException, InternalServerError

from backend.extensions import db


def handle_http_exception(err: HTTPException):
    if request.path.startswith("/v1/"):
        resp = jsonify({"error": err.name, "message": err.description})
        resp.status_code = err.code or 500
        return resp
    return err


def handle_internal_server_error(err: InternalServerError):
    if request.path.startswith("/v1/"):
        current_app.logger.exception("Unhandled API exception: %s", err)
        resp = jsonify(
            {"error": "Internal Server Error", "message": "Unexpected server error."}
        )
        resp.status_code = 500
        return resp
    return err


def handle_sqlalchemy_error(err: SQLAlchemyError):
    db.session.rollback()
    if request.path.startswith("/v1/"):
        current_app.logger.exception("Database error: %s", err)
        resp = jsonify(
            {"error": "Service Unavailable", "message": "Database is unavailable."}
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


def json_error(
    status: int, *, error: str, message: str, headers: dict[str, str] | None = None
) -> Response:
    resp = make_response(jsonify({"error": error, "message": message}), status)
    if headers:
        resp.headers.update(headers)
    return resp


def status_response(status: str, *, code: int = 200) -> Response:
    return make_response(jsonify({"status": status}), code)
