from __future__ import annotations

import json
import re
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request

from flask import abort, current_app

from backend.core.runtime_utils import urlopen_read_bytes as _urlopen_read_bytes

_ZITADEL_ERROR_SUFFIX_RE = re.compile(r"\s+\([A-Za-z0-9-]+\)\s*$")


def _clean_remote_error_message(message: str | None) -> str | None:
    if not isinstance(message, str):
        return None
    cleaned = _ZITADEL_ERROR_SUFFIX_RE.sub("", message).strip()
    return cleaned or None


def oauth_fetch_json(
    url: str,
    *,
    data: dict[str, str] | None = None,
    json_body: dict[str, object] | None = None,
    headers: dict[str, str] | None = None,
    method: str | None = None,
) -> dict[str, object]:
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    body = None
    if data is not None:
        body = urlencode(data).encode("utf-8")
        request_headers["Content-Type"] = "application/x-www-form-urlencoded"
    elif json_body is not None:
        body = json.dumps(json_body).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request_method = method or ("POST" if body is not None else "GET")
    req = Request(url, data=body, headers=request_headers, method=request_method)
    try:
        raw = _urlopen_read_bytes(req, timeout=15).decode("utf-8")
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
        current_app.logger.warning(
            "Remote auth provider HTTP error: url=%s status=%s body=%s",
            url,
            e.code,
            raw,
        )
        try:
            err_parsed_obj = cast(object, json.loads(raw))
        except json.JSONDecodeError:
            err_parsed_obj = None
        err_payload: dict[str, object] | None = (
            cast(dict[str, object], err_parsed_obj) if isinstance(err_parsed_obj, dict) else None
        )
        description = (
            cast(str, err_payload["error_description"])
            if err_payload and isinstance(err_payload.get("error_description"), str)
            else None
        )
        if description is None and err_payload and isinstance(err_payload.get("message"), str):
            description = cast(str, err_payload["message"])
        abort(
            e.code,
            description=_clean_remote_error_message(description)
            or f"Remote auth provider error ({e.code}).",
        )
    except URLError:
        abort(503, description="Remote auth provider is unavailable right now.")

    try:
        parsed_obj = cast(object, json.loads(raw))
    except json.JSONDecodeError:
        abort(502, description="Remote auth provider returned invalid JSON.")
    if not isinstance(parsed_obj, dict):
        abort(502, description="Remote auth provider returned an unexpected payload.")
    return cast(dict[str, object], parsed_obj)
