from __future__ import annotations

import json
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request

from flask import abort

from backend.core.runtime_utils import urlopen_read_bytes as _urlopen_read_bytes


def oauth_fetch_json(url: str, *, data: dict[str, str] | None = None) -> dict[str, object]:
    headers = {"Accept": "application/json"}
    body = None
    if data is not None:
        body = urlencode(data).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    req = Request(url, data=body, headers=headers, method="POST" if data is not None else "GET")
    try:
        raw = _urlopen_read_bytes(req, timeout=15).decode("utf-8")
    except HTTPError as e:
        raw = e.read().decode("utf-8", errors="replace")
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
        abort(e.code, description=description or f"Remote auth provider error ({e.code}).")
    except URLError:
        abort(503, description="Remote auth provider is unavailable right now.")

    try:
        parsed_obj = cast(object, json.loads(raw))
    except json.JSONDecodeError:
        abort(502, description="Remote auth provider returned invalid JSON.")
    if not isinstance(parsed_obj, dict):
        abort(502, description="Remote auth provider returned an unexpected payload.")
    return cast(dict[str, object], parsed_obj)
