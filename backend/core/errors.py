from __future__ import annotations

from flask import Flask, Response, current_app, jsonify, make_response, request
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.exceptions import HTTPException, InternalServerError


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
    if request.path.startswith("/v1/"):
        current_app.logger.exception("Database error: %s", err)
        resp = jsonify(
            {"error": "Service Unavailable", "message": "Database is unavailable."}
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
