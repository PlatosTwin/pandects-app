from __future__ import annotations

from dataclasses import dataclass
import logging
import time
from typing import Any, cast

from flask import Blueprint, Flask, Response, jsonify, make_response, request
from marshmallow import ValidationError
from werkzeug.exceptions import HTTPException

from backend.auth.mcp_runtime import (
    McpAuthError,
    authenticate_mcp_request,
    mcp_protocol_version,
    mcp_protected_resource_metadata,
    mcp_server_name,
    mcp_server_version,
)
from backend.mcp.tools import McpOutputValidationError, call_tool, tool_definitions
from backend.routes.deps import AgreementsDeps, ReferenceDataDeps, SectionsServiceDeps


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class McpDeps:
    sections_service_deps: SectionsServiceDeps
    agreements_deps: AgreementsDeps
    reference_data_deps: ReferenceDataDeps


def _json_rpc_error(
    *,
    request_id: object,
    code: int,
    message: str,
    data: object | None = None,
    status_code: int = 200,
    www_authenticate: str | None = None,
) -> Response:
    error_payload: dict[str, object] = {"code": code, "message": message}
    if data is not None:
        error_payload["data"] = data
    payload: dict[str, object] = {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": error_payload,
    }
    response = make_response(jsonify(payload), status_code)
    if www_authenticate is not None:
        response.headers["WWW-Authenticate"] = www_authenticate
    return response


def _json_rpc_result(*, request_id: object, result: dict[str, object]) -> Response:
    return make_response(jsonify({"jsonrpc": "2.0", "id": request_id, "result": result}), 200)


def _ensure_object_payload() -> dict[str, object]:
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ValueError("Request body must be a JSON object.")
    return cast(dict[str, object], body)


def _request_id_from_json_body() -> object:
    body = request.get_json(silent=True)
    if isinstance(body, dict):
        return body.get("id")
    return None


def _log_mcp_tool_event(
    *,
    tool_name: str,
    outcome: str,
    latency_ms: int,
    request_id: object,
    scope_count: int,
    error_category: str | None = None,
) -> None:
    logger.info(
        "mcp_tool_call",
        extra={
            "event": "mcp_tool_call",
            "tool_name": tool_name,
            "outcome": outcome,
            "latency_ms": latency_ms,
            "request_id": request_id,
            "scope_count": scope_count,
            "error_category": error_category,
        },
    )


def register_mcp_routes(target_app: Flask, *, deps: McpDeps) -> Blueprint:
    mcp_blp = Blueprint("mcp", "mcp")

    @mcp_blp.get("/.well-known/oauth-protected-resource")
    def protected_resource_metadata() -> Response:
        try:
            payload = mcp_protected_resource_metadata()
        except RuntimeError as exc:
            return make_response(
                jsonify({"error": "Service Unavailable", "message": str(exc)}),
                503,
            )
        return make_response(jsonify(payload), 200)

    @mcp_blp.route("/mcp", methods=["POST"])
    def mcp_endpoint() -> Response:
        request_id = _request_id_from_json_body()
        try:
            principal = authenticate_mcp_request()
        except McpAuthError as exc:
            return _json_rpc_error(
                request_id=request_id,
                code=-32001 if exc.status_code == 401 else -32003,
                message=exc.message,
                data={"category": "authentication", "status_code": exc.status_code},
                status_code=exc.status_code,
                www_authenticate=exc.www_authenticate,
            )

        try:
            payload = _ensure_object_payload()
        except ValueError as exc:
            return _json_rpc_error(request_id=None, code=-32700, message=str(exc))

        request_id = payload.get("id")
        method = payload.get("method")
        params = payload.get("params")
        if not isinstance(method, str) or not method.strip():
            return _json_rpc_error(request_id=request_id, code=-32600, message="Missing JSON-RPC method.")
        params_obj = params if isinstance(params, dict) else {}

        if method == "notifications/initialized" and request_id is None:
            return make_response("", 202)
        if method == "ping":
            return _json_rpc_result(request_id=request_id, result={})
        if method == "initialize":
            return _json_rpc_result(
                request_id=request_id,
                result={
                    "protocolVersion": mcp_protocol_version(),
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {
                        "name": mcp_server_name(),
                        "version": mcp_server_version(),
                    },
                },
            )
        if method == "tools/list":
            return _json_rpc_result(
                request_id=request_id,
                result={"tools": tool_definitions()},
            )
        if method == "tools/call":
            tool_name = params_obj.get("name")
            arguments = params_obj.get("arguments", {})
            if not isinstance(tool_name, str) or not tool_name.strip():
                return _json_rpc_error(
                    request_id=request_id,
                    code=-32602,
                    message="Tool name is required.",
                )
            if not isinstance(arguments, dict):
                return _json_rpc_error(
                    request_id=request_id,
                    code=-32602,
                    message="Tool arguments must be an object.",
                )
            started_at = time.perf_counter()
            try:
                result = call_tool(
                    tool_name,
                    arguments=cast(dict[str, object], arguments),
                    principal=principal,
                    sections_service_deps=deps.sections_service_deps,
                    agreements_deps=deps.agreements_deps,
                    reference_data_deps=deps.reference_data_deps,
                )
            except KeyError:
                _log_mcp_tool_event(
                    tool_name=tool_name,
                    outcome="error",
                    latency_ms=max(0, int((time.perf_counter() - started_at) * 1000)),
                    request_id=request_id,
                    scope_count=len(principal.scopes),
                    error_category="unknown_tool",
                )
                return _json_rpc_error(
                    request_id=request_id,
                    code=-32601,
                    message=f"Unknown tool: {tool_name}",
                )
            except ValidationError as exc:
                _log_mcp_tool_event(
                    tool_name=tool_name,
                    outcome="error",
                    latency_ms=max(0, int((time.perf_counter() - started_at) * 1000)),
                    request_id=request_id,
                    scope_count=len(principal.scopes),
                    error_category="validation",
                )
                return _json_rpc_error(
                    request_id=request_id,
                    code=-32602,
                    message="Invalid tool arguments.",
                    data=exc.messages,
                )
            except PermissionError as exc:
                _log_mcp_tool_event(
                    tool_name=tool_name,
                    outcome="error",
                    latency_ms=max(0, int((time.perf_counter() - started_at) * 1000)),
                    request_id=request_id,
                    scope_count=len(principal.scopes),
                    error_category="authorization",
                )
                return _json_rpc_error(
                    request_id=request_id,
                    code=-32003,
                    message=str(exc),
                    data={"category": "authorization", "status_code": 403},
                    status_code=403,
                    www_authenticate='Bearer realm="pandects-mcp"',
                )
            except McpOutputValidationError as exc:
                logger.exception(
                    "mcp_tool_output_validation_failed",
                    extra={
                        "event": "mcp_tool_output_validation_failed",
                        "tool_name": tool_name,
                        "request_id": request_id,
                        "scope_count": len(principal.scopes),
                    },
                )
                _log_mcp_tool_event(
                    tool_name=tool_name,
                    outcome="error",
                    latency_ms=max(0, int((time.perf_counter() - started_at) * 1000)),
                    request_id=request_id,
                    scope_count=len(principal.scopes),
                    error_category="output_validation",
                )
                return _json_rpc_error(
                    request_id=request_id,
                    code=-32603,
                    message="Tool result violated the advertised output contract.",
                    data=exc.messages,
                )
            except HTTPException as exc:
                _log_mcp_tool_event(
                    tool_name=tool_name,
                    outcome="error",
                    latency_ms=max(0, int((time.perf_counter() - started_at) * 1000)),
                    request_id=request_id,
                    scope_count=len(principal.scopes),
                    error_category="http_exception",
                )
                description = exc.description if isinstance(exc.description, str) else "Request failed."
                return _json_rpc_error(
                    request_id=request_id,
                    code=-32602 if exc.code == 400 else -32004,
                    message=description,
                )
            _log_mcp_tool_event(
                tool_name=tool_name,
                outcome="ok",
                latency_ms=max(0, int((time.perf_counter() - started_at) * 1000)),
                request_id=request_id,
                scope_count=len(principal.scopes),
            )
            return _json_rpc_result(
                request_id=request_id,
                result={
                    "content": [{"type": "text", "text": result.text}],
                    "structuredContent": result.structured_content,
                },
            )
        return _json_rpc_error(
            request_id=request_id,
            code=-32601,
            message=f"Method not found: {method}",
        )

    return mcp_blp


__all__ = ["McpDeps", "register_mcp_routes"]
