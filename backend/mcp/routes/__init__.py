from __future__ import annotations

import logging
import time
from typing import cast
from urllib.parse import urlparse

from flask import Blueprint, Flask, Response, jsonify, make_response, request
from marshmallow import ValidationError
from werkzeug.exceptions import HTTPException

from backend.auth.mcp_oauth_runtime import mcp_oauth_issuer, mcp_oauth_metadata
from backend.auth.mcp_runtime import (
    McpAuthError,
    authenticate_mcp_request,
    mcp_protected_resource_metadata,
    mcp_protocol_version,
    mcp_server_name,
    mcp_server_version,
)
from backend.mcp.routes.deps import McpDeps
from backend.mcp.routes.helpers import (
    AUTHORIZATION_ERROR_MESSAGE,
    BAD_TOOL_REQUEST_MESSAGE,
    INVALID_JSON_RPC_PAYLOAD_MESSAGE,
    PROTECTED_RESOURCE_UNAVAILABLE_MESSAGE,
    _apply_protocol_headers,
    _client_prefers_sse,
    _ensure_object_payload,
    _extract_progress_token,
    _issue_session_id,
    _json_rpc_error,
    _json_rpc_result,
    _log_mcp_tool_event,
    _prompt_definitions,
    _read_resource,
    _render_prompt,
    _request_id_from_json_body,
    _resource_definitions,
    _sse_retry_probe_response,
    _stream_tool_call_response,
    metrics_registry,
)
from backend.mcp.tools import McpOutputValidationError, call_tool, tool_definitions


logger = logging.getLogger(__name__)



def register_mcp_routes(target_app: Flask, *, deps: McpDeps) -> Blueprint:
    mcp_blp = Blueprint("mcp", "mcp")

    def _oauth_authorization_server_metadata_response() -> Response:
        try:
            payload = cast(dict[str, object], dict(mcp_oauth_metadata()))
        except RuntimeError as exc:
            logger.warning("mcp_oauth_metadata_unavailable", exc_info=exc)
            return make_response(
                jsonify(
                    {
                        "error": "Service Unavailable",
                        "message": PROTECTED_RESOURCE_UNAVAILABLE_MESSAGE,
                    }
                ),
                503,
            )
        response = make_response(jsonify(payload), 200)
        response.headers["Cache-Control"] = "no-store"
        return response

    def _issuer_path_suffix() -> str:
        try:
            return urlparse(mcp_oauth_issuer()).path.strip("/")
        except Exception:  # pragma: no cover - defensive
            return ""

    @mcp_blp.get("/.well-known/oauth-authorization-server")
    def oauth_authorization_server_root_metadata() -> Response:
        return _oauth_authorization_server_metadata_response()

    @mcp_blp.get("/.well-known/oauth-authorization-server/<path:suffix>")
    def oauth_authorization_server_pathed_metadata(suffix: str) -> Response:
        expected = _issuer_path_suffix()
        if expected and suffix.strip("/") != expected:
            return make_response(
                jsonify({"error": "Not Found", "message": "Unknown authorization server path."}),
                404,
            )
        return _oauth_authorization_server_metadata_response()

    @mcp_blp.get("/.well-known/openid-configuration")
    def openid_configuration_root_metadata() -> Response:
        return _oauth_authorization_server_metadata_response()

    @mcp_blp.get("/.well-known/oauth-protected-resource")
    def protected_resource_metadata() -> Response:
        try:
            payload = mcp_protected_resource_metadata()
        except RuntimeError as exc:
            logger.warning("mcp_protected_resource_metadata_unavailable", exc_info=exc)
            return make_response(
                jsonify(
                    {
                        "error": "Service Unavailable",
                        "message": PROTECTED_RESOURCE_UNAVAILABLE_MESSAGE,
                    }
                ),
                503,
            )
        return make_response(jsonify(payload), 200)

    @mcp_blp.route("/mcp", methods=["POST"])
    def mcp_endpoint() -> Response:
        request_id = _request_id_from_json_body()
        try:
            principal = authenticate_mcp_request()
        except McpAuthError as exc:
            metrics_registry.record_auth_failure(status_code=exc.status_code)
            return _json_rpc_error(
                request_id=request_id,
                code=-32001 if exc.status_code == 401 else -32003,
                message=exc.message,
                data={
                    "category": "authentication",
                    "status_code": exc.status_code,
                    "reason": exc.reason,
                    "action": exc.action,
                    "client_message": exc.client_message,
                },
                status_code=exc.status_code,
                www_authenticate=exc.www_authenticate,
            )

        try:
            payload = _ensure_object_payload()
        except ValueError:
            return _json_rpc_error(request_id=None, code=-32700, message=INVALID_JSON_RPC_PAYLOAD_MESSAGE)

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
            client_version = params_obj.get("protocolVersion")
            negotiated_version = (
                client_version
                if isinstance(client_version, str) and client_version.strip()
                else mcp_protocol_version()
            )
            return _json_rpc_result(
                request_id=request_id,
                result={
                    "protocolVersion": negotiated_version,
                    "capabilities": {
                        "tools": {"listChanged": False},
                        "resources": {"listChanged": False, "subscribe": False},
                        "prompts": {"listChanged": False},
                        "logging": {},
                    },
                    "serverInfo": {
                        "name": mcp_server_name(),
                        "version": mcp_server_version(),
                    },
                    "instructions": (
                        "Pandects MCP exposes M&A agreement and clause research tools. "
                        "Start with get_server_capabilities for concept guidance and scope mapping, "
                        "then list_filter_options or search_agreements/search_sections for discovery."
                    ),
                },
                extra_headers={"Mcp-Session-Id": _issue_session_id()},
            )
        if method == "tools/list":
            return _json_rpc_result(
                request_id=request_id,
                result={"tools": tool_definitions()},
            )
        if method == "prompts/list":
            return _json_rpc_result(
                request_id=request_id,
                result={"prompts": _prompt_definitions()},
            )
        if method == "prompts/get":
            prompt_name = params_obj.get("name")
            if not isinstance(prompt_name, str) or not prompt_name.strip():
                return _json_rpc_error(
                    request_id=request_id, code=-32602, message="Prompt name is required."
                )
            arguments = params_obj.get("arguments")
            arguments_dict = arguments if isinstance(arguments, dict) else {}
            prompt_payload = _render_prompt(
                prompt_name,
                cast(dict[str, object], arguments_dict),
            )
            if prompt_payload is None:
                return _json_rpc_error(
                    request_id=request_id,
                    code=-32602,
                    message=f"Unknown prompt: {prompt_name}",
                )
            return _json_rpc_result(request_id=request_id, result=prompt_payload)
        if method == "resources/list":
            return _json_rpc_result(
                request_id=request_id,
                result={"resources": _resource_definitions()},
            )
        if method == "resources/read":
            resource_uri = params_obj.get("uri")
            if not isinstance(resource_uri, str) or not resource_uri.strip():
                return _json_rpc_error(
                    request_id=request_id, code=-32602, message="Resource uri is required."
                )
            resource_contents = _read_resource(resource_uri)
            if resource_contents is None:
                return _json_rpc_error(
                    request_id=request_id,
                    code=-32602,
                    message=f"Unknown resource: {resource_uri}",
                )
            return _json_rpc_result(
                request_id=request_id, result={"contents": resource_contents}
            )
        if method == "logging/setLevel":
            return _json_rpc_result(request_id=request_id, result={})
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
            progress_token = _extract_progress_token(params_obj)
            if progress_token is not None and _client_prefers_sse():
                return _stream_tool_call_response(
                    request_id=request_id,
                    tool_name=tool_name,
                    arguments=cast(dict[str, object], arguments),
                    principal=principal,
                    deps=deps,
                    progress_token=progress_token,
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
                logger.warning(
                    "mcp_tool_permission_denied",
                    extra={
                        "event": "mcp_tool_permission_denied",
                        "tool_name": tool_name,
                        "request_id": request_id,
                        "scope_count": len(principal.scopes),
                    },
                    exc_info=exc,
                )
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
                    message=AUTHORIZATION_ERROR_MESSAGE,
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
                return _json_rpc_error(
                    request_id=request_id,
                    code=-32602 if exc.code == 400 else -32004,
                    message=BAD_TOOL_REQUEST_MESSAGE if exc.code == 400 else "Tool request failed.",
                )
            latency_ms = max(0, int((time.perf_counter() - started_at) * 1000))
            if tool_name == "get_server_metrics" and isinstance(result.structured_content, dict):
                _log_mcp_tool_event(
                    tool_name=tool_name,
                    outcome="ok",
                    latency_ms=latency_ms,
                    request_id=request_id,
                    scope_count=len(principal.scopes),
                )
                result = result.__class__(
                    text=result.text,
                    structured_content=metrics_registry.snapshot(),
                )
            else:
                _log_mcp_tool_event(
                    tool_name=tool_name,
                    outcome="ok",
                    latency_ms=latency_ms,
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

    @mcp_blp.route("/mcp", methods=["DELETE"])
    def mcp_session_delete() -> Response:
        try:
            authenticate_mcp_request()
        except McpAuthError as exc:
            metrics_registry.record_auth_failure(status_code=exc.status_code)
            response = _json_rpc_error(
                request_id=None,
                code=-32001 if exc.status_code == 401 else -32003,
                message=exc.message,
                data={
                    "category": "authentication",
                    "status_code": exc.status_code,
                    "reason": exc.reason,
                    "action": exc.action,
                    "client_message": exc.client_message,
                },
                status_code=exc.status_code,
                www_authenticate=exc.www_authenticate,
            )
            return response
        response = make_response("", 204)
        return _apply_protocol_headers(response)

    @mcp_blp.get("/mcp")
    def mcp_sse_endpoint() -> Response:
        try:
            authenticate_mcp_request()
        except McpAuthError as exc:
            metrics_registry.record_auth_failure(status_code=exc.status_code)
            return _json_rpc_error(
                request_id=None,
                code=-32001 if exc.status_code == 401 else -32003,
                message=exc.message,
                data={
                    "category": "authentication",
                    "status_code": exc.status_code,
                    "reason": exc.reason,
                    "action": exc.action,
                    "client_message": exc.client_message,
                },
                status_code=exc.status_code,
                www_authenticate=exc.www_authenticate,
            )

        return _sse_retry_probe_response()

    return mcp_blp


__all__ = ["McpDeps", "register_mcp_routes"]
