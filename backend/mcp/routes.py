from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
import json
import logging
import time
from uuid import uuid4
from typing import Any, cast

from flask import Blueprint, Flask, Response, jsonify, make_response, request, stream_with_context
from marshmallow import ValidationError
from werkzeug.exceptions import HTTPException

from backend.auth.mcp_oauth_runtime import mcp_oauth_issuer, mcp_oauth_metadata
from backend.auth.mcp_runtime import (
    McpAuthError,
    authenticate_mcp_request,
    mcp_protocol_version,
    mcp_protected_resource_metadata,
    mcp_server_name,
    mcp_server_version,
)
from urllib.parse import urlparse
from backend.mcp.metrics import get_mcp_metrics_registry
from backend.mcp.tools import McpOutputValidationError, call_tool, tool_definitions
from backend.routes.deps import AgreementsDeps, ReferenceDataDeps, SectionsServiceDeps


logger = logging.getLogger(__name__)
metrics_registry = get_mcp_metrics_registry()
PROTECTED_RESOURCE_UNAVAILABLE_MESSAGE = "Protected resource metadata is unavailable right now."
INVALID_JSON_RPC_PAYLOAD_MESSAGE = "Request body must be a JSON object."
AUTHORIZATION_ERROR_MESSAGE = "You do not have permission to call this tool."
BAD_TOOL_REQUEST_MESSAGE = "Tool request was invalid."


@dataclass(frozen=True)
class McpDeps:
    sections_service_deps: SectionsServiceDeps
    agreements_deps: AgreementsDeps
    reference_data_deps: ReferenceDataDeps


def _client_prefers_sse() -> bool:
    accept_header = request.headers.get("Accept", "") or ""
    if not accept_header:
        return False
    tokens = [token.strip().lower() for token in accept_header.split(",") if token.strip()]
    has_sse = any(token.startswith("text/event-stream") for token in tokens)
    has_json = any(token.startswith("application/json") or token.startswith("*/*") for token in tokens)
    if not has_sse:
        return False
    if not has_json:
        return True
    for token in tokens:
        media = token.split(";")[0].strip()
        if media == "text/event-stream":
            return True
        if media in ("application/json", "*/*"):
            return False
    return False


def _apply_protocol_headers(response: Response) -> Response:
    try:
        response.headers["MCP-Protocol-Version"] = mcp_protocol_version()
    except Exception:  # pragma: no cover - defensive
        pass
    return response


def _json_compatible(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _json_response(payload: dict[str, object], status_code: int) -> Response:
    return make_response(jsonify(_json_compatible(payload)), status_code)


def _sse_body(payload: dict[str, object]) -> str:
    event_id = uuid4().hex
    try:
        data = json.dumps(_json_compatible(payload), separators=(",", ":"))
    except TypeError:
        logger.exception("mcp_sse_payload_serialization_failed")
        raise
    return f"id: {event_id}\nevent: message\ndata: {data}\n\n"


def _sse_envelope(payload: dict[str, object]) -> Response:
    response = make_response(_sse_body(payload), 200)
    response.headers["Content-Type"] = "text/event-stream; charset=utf-8"
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Accel-Buffering"] = "no"
    return response


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
    if status_code == 200 and _client_prefers_sse():
        response = _sse_envelope(payload)
    else:
        response = _json_response(payload, status_code)
    if www_authenticate is not None:
        response.headers["WWW-Authenticate"] = www_authenticate
    return _apply_protocol_headers(response)


def _json_rpc_result(
    *,
    request_id: object,
    result: dict[str, object],
    extra_headers: dict[str, str] | None = None,
) -> Response:
    payload: dict[str, object] = {"jsonrpc": "2.0", "id": request_id, "result": result}
    if _client_prefers_sse():
        response = _sse_envelope(payload)
    else:
        response = _json_response(payload, 200)
    if extra_headers:
        for key, value in extra_headers.items():
            response.headers[key] = value
    return _apply_protocol_headers(response)


def _sse_retry_probe_response() -> Response:
    event_id = uuid4().hex
    response = make_response(
        f"id: {event_id}\nretry: 1000\ndata:\n\n",
        200,
    )
    response.headers["Content-Type"] = "text/event-stream; charset=utf-8"
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Accel-Buffering"] = "no"
    return _apply_protocol_headers(response)


def _issue_session_id() -> str:
    return uuid4().hex


def _extract_progress_token(params: dict[str, object]) -> object | None:
    meta = params.get("_meta")
    if not isinstance(meta, dict):
        return None
    token = meta.get("progressToken")
    if isinstance(token, (str, int, float)) and str(token).strip():
        return token
    return None


def _encode_sse_event(payload: dict[str, object]) -> str:
    return _sse_body(payload)


def _stream_tool_call_response(
    *,
    request_id: object,
    tool_name: str,
    arguments: dict[str, object],
    principal: Any,
    deps: McpDeps,
    progress_token: object,
) -> Response:
    def _progress_event(progress: float, total: float, message: str) -> dict[str, object]:
        return {
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {
                "progressToken": progress_token,
                "progress": progress,
                "total": total,
                "message": message,
            },
        }

    def generator() -> "Any":
        yield _encode_sse_event(
            _progress_event(0.0, 1.0, f"Starting {tool_name}")
        )
        started_at = time.perf_counter()
        scope_count = len(getattr(principal, "scopes", []) or [])
        try:
            result = call_tool(
                tool_name,
                arguments=arguments,
                principal=cast(Any, principal),
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
                scope_count=scope_count,
                error_category="unknown_tool",
            )
            yield _encode_sse_event(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": f"Unknown tool: {tool_name}"},
                }
            )
            return
        except ValidationError as exc:
            _log_mcp_tool_event(
                tool_name=tool_name,
                outcome="error",
                latency_ms=max(0, int((time.perf_counter() - started_at) * 1000)),
                request_id=request_id,
                scope_count=scope_count,
                error_category="validation",
            )
            yield _encode_sse_event(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": f"Invalid tool arguments: {', '.join(str(k) for k in exc.messages.keys()) if isinstance(exc.messages, dict) else 'unknown fields'}.",
                        "data": exc.messages,
                    },
                }
            )
            return
        except PermissionError:
            _log_mcp_tool_event(
                tool_name=tool_name,
                outcome="error",
                latency_ms=max(0, int((time.perf_counter() - started_at) * 1000)),
                request_id=request_id,
                scope_count=scope_count,
                error_category="authorization",
            )
            yield _encode_sse_event(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32003,
                        "message": AUTHORIZATION_ERROR_MESSAGE,
                        "data": {"category": "authorization", "status_code": 403},
                    },
                }
            )
            return
        except McpOutputValidationError as exc:
            _log_mcp_tool_event(
                tool_name=tool_name,
                outcome="error",
                latency_ms=max(0, int((time.perf_counter() - started_at) * 1000)),
                request_id=request_id,
                scope_count=scope_count,
                error_category="output_validation",
            )
            yield _encode_sse_event(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32603,
                        "message": "Tool result violated the advertised output contract.",
                        "data": exc.messages,
                    },
                }
            )
            return
        except HTTPException as exc:
            _log_mcp_tool_event(
                tool_name=tool_name,
                outcome="error",
                latency_ms=max(0, int((time.perf_counter() - started_at) * 1000)),
                request_id=request_id,
                scope_count=scope_count,
                error_category="http_exception",
            )
            yield _encode_sse_event(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602 if exc.code == 400 else -32004,
                        "message": BAD_TOOL_REQUEST_MESSAGE if exc.code == 400 else "Tool request failed.",
                    },
                }
            )
            return
        latency_ms = max(0, int((time.perf_counter() - started_at) * 1000))
        if tool_name == "get_server_metrics" and isinstance(result.structured_content, dict):
            result = result.__class__(
                text=result.text,
                structured_content=metrics_registry.snapshot(),
            )
        _log_mcp_tool_event(
            tool_name=tool_name,
            outcome="ok",
            latency_ms=latency_ms,
            request_id=request_id,
            scope_count=scope_count,
        )
        yield _encode_sse_event(
            _progress_event(1.0, 1.0, f"{tool_name} complete")
        )
        yield _encode_sse_event(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [{"type": "text", "text": result.text}],
                    "structuredContent": result.structured_content,
                },
            }
        )

    response = Response(stream_with_context(generator()), status=200)
    response.headers["Content-Type"] = "text/event-stream; charset=utf-8"
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Accel-Buffering"] = "no"
    return _apply_protocol_headers(response)


_PROMPT_TEMPLATES: tuple[dict[str, object], ...] = (
    {
        "name": "compare_agreements",
        "title": "Compare two M&A agreements",
        "description": (
            "Side-by-side comparison of two merger agreements on structure, "
            "deal economics, and negotiated risk allocation. Pulls sections, "
            "key clauses, and metadata via the MCP tools."
        ),
        "arguments": [
            {"name": "agreement_a", "description": "Agreement UUID, SEC URL, or target name for deal A.", "required": True},
            {"name": "agreement_b", "description": "Agreement UUID, SEC URL, or target name for deal B.", "required": True},
            {"name": "focus", "description": "Optional focus area (e.g. 'MAE', 'closing conditions', 'termination fees').", "required": False},
        ],
        "template": (
            "You are an M&A research analyst with access to the Pandects MCP server. "
            "Compare the following two agreements and produce a structured comparison.\n\n"
            "Agreement A: {agreement_a}\nAgreement B: {agreement_b}\n"
            "Focus: {focus}\n\n"
            "Workflow:\n"
            "1. Resolve each identifier with `get_agreement` (or `search_agreements` if only a name is given).\n"
            "2. Use `list_agreement_sections` and `get_section` to surface the sections relevant to the focus.\n"
            "3. Call `get_agreement_tax_clauses` or `get_section_tax_clauses` for quantified clause positions.\n"
            "4. Present a table: Deal A vs Deal B across structure, consideration, conditions, covenants, remedies, and the focus area.\n"
            "5. Call out material divergences and explain likely negotiating rationale."
        ),
    },
    {
        "name": "clause_family_survey",
        "title": "Survey a clause family across a filter",
        "description": (
            "Run a corpus-level survey of how a clause concept is drafted across "
            "a filtered subset of agreements (by year, industry, deal type, etc.)."
        ),
        "arguments": [
            {"name": "concept", "description": "Clause concept in plain language (e.g. 'material adverse effect carveouts').", "required": True},
            {"name": "filters", "description": "Free-form filter hints (year range, industry, deal type, counsel).", "required": False},
        ],
        "template": (
            "Task: Survey how '{concept}' is drafted across the filter '{filters}'.\n\n"
            "1. Call `suggest_clause_families` with the concept to locate canonical taxonomy nodes.\n"
            "2. Call `list_filter_options` to confirm which filter values exist for the hinted dimensions.\n"
            "3. Call `search_sections` with the taxonomy match plus filters. Paginate until you have a representative sample.\n"
            "4. Cluster returned sections by drafting pattern. Quote minimal verbatim excerpts via `get_section_snippet`.\n"
            "5. Deliver: (a) the taxonomy nodes used, (b) prevalence of each drafting variant, (c) illustrative excerpts with citations."
        ),
    },
    {
        "name": "deal_trend_brief",
        "title": "Brief on deal trends for a slice",
        "description": (
            "Produce a quantitative brief on deal activity for a given slice "
            "(industry, year range, deal type). Combines summary counts, filters, and trend data."
        ),
        "arguments": [
            {"name": "slice", "description": "Describe the slice (e.g. 'US tech M&A 2018-2024').", "required": True},
        ],
        "template": (
            "Brief the slice: {slice}.\n\n"
            "1. Start with `get_agreements_summary` and `get_agreement_trends` for corpus-wide counts.\n"
            "2. Use `list_filter_options` to discover valid values for the slice's dimensions.\n"
            "3. Use `list_agreements` with the slice filters; paginate for representative sample.\n"
            "4. Call `search_sections` to spot-check drafting shifts for 2-3 clause families relevant to the slice.\n"
            "5. Deliver a one-page brief: deal count, consideration mix, notable counsels, clause-level shifts."
        ),
    },
)


def _prompt_definitions() -> list[dict[str, object]]:
    definitions: list[dict[str, object]] = []
    for prompt in _PROMPT_TEMPLATES:
        definitions.append(
            {
                "name": prompt["name"],
                "title": prompt["title"],
                "description": prompt["description"],
                "arguments": prompt["arguments"],
            }
        )
    return definitions


def _render_prompt(name: str, arguments: dict[str, object]) -> dict[str, object] | None:
    for prompt in _PROMPT_TEMPLATES:
        if prompt["name"] != name:
            continue
        template = cast(str, prompt["template"])
        rendered = template
        for arg_def in cast(list[dict[str, object]], prompt["arguments"]):
            arg_name = cast(str, arg_def["name"])
            value = arguments.get(arg_name, "")
            if not isinstance(value, str):
                value = json.dumps(value)
            rendered = rendered.replace("{" + arg_name + "}", value or "(unspecified)")
        return {
            "description": prompt["description"],
            "messages": [
                {
                    "role": "user",
                    "content": {"type": "text", "text": rendered},
                }
            ],
        }
    return None


_RESOURCE_DEFINITIONS: tuple[dict[str, object], ...] = (
    {
        "uri": "pandects://capabilities",
        "name": "Pandects MCP capabilities",
        "description": (
            "Server capabilities including tool inventory, required scopes, "
            "field inventory, concept notes, and research workflows."
        ),
        "mimeType": "application/json",
    },
    {
        "uri": "pandects://auth-help",
        "name": "Authentication help",
        "description": "How to obtain and refresh bearer tokens for Pandects MCP.",
        "mimeType": "application/json",
    },
    {
        "uri": "pandects://tools-manifest",
        "name": "Tools manifest",
        "description": (
            "Lightweight index of every tool: name, one-line description, required scopes, "
            "and top-level parameter names with their types. Cheaper than tools/list; "
            "read this first to orient before loading individual tool schemas."
        ),
        "mimeType": "application/json",
    },
)


def _resource_definitions() -> list[dict[str, object]]:
    return [dict(resource) for resource in _RESOURCE_DEFINITIONS]


def _tools_manifest_payload() -> list[dict[str, object]]:
    from backend.mcp.tools import _tool_specs

    entries: list[dict[str, object]] = []
    for spec in _tool_specs():
        raw_params = spec.input_schema.get("properties", {})
        params = raw_params if isinstance(raw_params, dict) else {}
        param_summaries: list[dict[str, object]] = [
            {"name": k, "type": v["type"] if isinstance(v, dict) and "type" in v else "any"}
            for k, v in params.items()
        ]
        entries.append(
            {
                "name": spec.name,
                "description": spec.description,
                "scopes": list(spec.scopes),
                "selection_hint": spec.selection_hint,
                "parameters": param_summaries,
                "pagination": spec.pagination,
            }
        )
    return entries


def _read_resource(uri: str) -> list[dict[str, object]] | None:
    from backend.mcp.tools import _server_capabilities_payload

    if uri == "pandects://capabilities":
        payload = _server_capabilities_payload()
        return [
            {
                "uri": uri,
                "mimeType": "application/json",
                "text": json.dumps(payload, separators=(",", ":"), default=str),
            }
        ]
    if uri == "pandects://auth-help":
        payload = _server_capabilities_payload().get("auth_help", {})
        return [
            {
                "uri": uri,
                "mimeType": "application/json",
                "text": json.dumps(payload, separators=(",", ":"), default=str),
            }
        ]
    if uri == "pandects://tools-manifest":
        return [
            {
                "uri": uri,
                "mimeType": "application/json",
                "text": json.dumps(_tools_manifest_payload(), separators=(",", ":"), default=str),
            }
        ]
    return None


def _ensure_object_payload() -> dict[str, object]:
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        raise ValueError(INVALID_JSON_RPC_PAYLOAD_MESSAGE)
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
    metrics_registry.record_tool_call(
        tool_name=tool_name,
        latency_ms=latency_ms,
        outcome=outcome,
        error_category=error_category,
    )


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
