"""Changelog summary for the MCP capabilities payload.

Reads the published machine-readable changelog (the R2 artifact that ships
next to the bulk dumps — see bulk/changelog/DESIGN.md) so agents can discover
dataset/schema change history. The fetch is TTL-cached, time-boxed, and
failure-tolerant: capabilities must never 500 or hang because R2 is down, so
on any error the section is returned with null latest-release fields.

Set MCP_CHANGELOG_FETCH=0 to disable the network fetch entirely (tests do
this to stay deterministic).
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
from threading import Lock
from typing import cast

CHANGELOG_PUBLIC_URL = os.environ.get(
    "CHANGELOG_JSON_URL", "https://bulk.pandects.org/dumps/changelog.json"
)
_FETCH_TIMEOUT_SECONDS = 3
_TTL_OK_SECONDS = 900
_TTL_FAIL_SECONDS = 120

_cache: dict[str, object] = {"ts": 0.0, "ok": False, "payload": None}
_cache_lock = Lock()


def _fetch_changelog_json() -> dict[str, object] | None:
    if os.environ.get("MCP_CHANGELOG_FETCH", "1") == "0":
        return None
    now = time.time()
    with _cache_lock:
        age = now - cast(float, _cache["ts"])
        ttl = _TTL_OK_SECONDS if _cache["ok"] else _TTL_FAIL_SECONDS
        if age < ttl:
            return cast("dict[str, object] | None", _cache["payload"])
    payload: dict[str, object] | None = None
    try:
        with urllib.request.urlopen(
            CHANGELOG_PUBLIC_URL, timeout=_FETCH_TIMEOUT_SECONDS
        ) as response:
            parsed = cast(object, json.loads(response.read()))
        if isinstance(parsed, dict):
            payload = cast(dict[str, object], parsed)
    except Exception:
        payload = None
    with _cache_lock:
        _cache["ts"] = time.time()
        _cache["ok"] = payload is not None
        _cache["payload"] = payload
    return payload


def changelog_capabilities_section() -> dict[str, object]:
    payload = _fetch_changelog_json()
    latest_version: object = None
    latest_released: object = None
    breaking: object = None
    if payload is not None:
        latest_version = payload.get("latest_version")
        latest_released = payload.get("latest_released")
        releases = payload.get("releases")
        if isinstance(releases, list) and releases and isinstance(releases[0], dict):
            changes = cast(dict[str, object], releases[0]).get("changes")
            if isinstance(changes, list):
                breaking = any(
                    isinstance(change, dict) and change.get("severity") == "breaking"
                    for change in changes
                )
    return {
        "latest_version": latest_version,
        "latest_released": latest_released,
        "breaking_changes_in_latest": breaking,
        "url": CHANGELOG_PUBLIC_URL,
        "api_route": "/v1/changelog",
        "note": (
            "Dataset, schema, and API change history, one release per published bulk "
            "dump. Correlate a response's X-Pandects-Dump-Hash header (or "
            "dump_version.hash body field) with releases[].dump_sha256 to know exactly "
            "which changes the data you are querying includes. Check it when results "
            "seem to have changed across sessions."
        ),
    }
