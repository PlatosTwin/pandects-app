"""Write (or check) the committed OpenAPI specs.

Two specs come from one app:

- ``frontend/public/openapi.yaml``: the public API (the flask-smorest
  resources). The docs site renders it.
- ``backend/openapi/ui-openapi.json``: the public API plus the web client's
  internal routes (``backend/schemas/ui_api.py``). ``frontend/shared/api.gen.ts``
  is generated from it (``npm run gen:api`` in ``frontend/``).

Usage, from the repository root (no database needed):

    backend/venv/bin/python -m backend.scripts.export_openapi          # write
    backend/venv/bin/python -m backend.scripts.export_openapi --check  # fail on drift
"""

from __future__ import annotations

import argparse
import atexit
import copy
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from flask import Flask

    from backend.schemas.ui_operations import SchemaRef, UiOperation

REPO_ROOT = Path(__file__).resolve().parents[2]
PUBLIC_SPEC_PATH = REPO_ROOT / "frontend" / "public" / "openapi.yaml"
UI_SPEC_PATH = REPO_ROOT / "backend" / "openapi" / "ui-openapi.json"

_UI_TAGS: tuple[dict[str, str], ...] = (
    {"name": "ui-data", "description": "Internal: dataset summaries and filter options for the web client"},
    {"name": "auth", "description": "Internal: web-client authentication and account routes"},
    {"name": "favorites", "description": "Internal: favorites, projects, and tags"},
)
_PATH_PARAM_RE = re.compile(r"{([^}]+)}")


def _set_offline_env() -> None:
    """Let ``backend.app`` import without MariaDB or Postgres (as the tests do)."""
    defaults = {
        "SKIP_MAIN_DB_REFLECTION": "1",
        "MARIADB_USER": "openapi",
        "MARIADB_PASSWORD": "openapi",
        "MARIADB_HOST": "127.0.0.1",
        "MARIADB_DATABASE": "pdx",
        "AUTH_SECRET_KEY": "openapi-export",
        "PUBLIC_API_BASE_URL": "http://localhost:5000",
        "PUBLIC_FRONTEND_BASE_URL": "http://localhost:8080",
    }
    for key, value in defaults.items():
        _ = os.environ.setdefault(key, value)
    if not os.environ.get("AUTH_DATABASE_URI"):
        auth_db = Path(tempfile.gettempdir()) / f"pandects_openapi_auth_{os.getpid()}.sqlite"
        os.environ["AUTH_DATABASE_URI"] = f"sqlite:///{auth_db}"
        _ = atexit.register(auth_db.unlink, missing_ok=True)


def _json_content(schema: SchemaRef) -> dict[str, object]:
    return {"application/json": {"schema": schema}}


def _operation(op: UiOperation) -> dict[str, object]:
    responses: dict[str, object] = {}
    statuses: dict[int, tuple[str, SchemaRef | None]] = {
        op.status: (op.response_description, op.response),
        **op.extra_responses,
    }
    for status, (description, schema) in sorted(statuses.items()):
        response: dict[str, object] = {"description": description}
        if schema is not None:
            response["content"] = _json_content(schema)
        responses[str(status)] = response
    responses["default"] = {"$ref": "#/components/responses/DEFAULT_ERROR"}

    parameters: list[dict[str, object]] = [
        {
            "in": "path",
            "name": name,
            "required": True,
            "schema": {"type": "integer" if name in op.int_path_params else "string"},
        }
        for name in _PATH_PARAM_RE.findall(op.path)
    ]
    if op.query is not None:
        parameters.append({"in": "query", "schema": op.query})

    operation: dict[str, object] = {
        "tags": [op.tag],
        "operationId": op.operation_id,
        "summary": op.summary,
        "responses": responses,
    }
    if op.description:
        operation["description"] = op.description
    if parameters:
        operation["parameters"] = parameters
    if op.request_body is not None:
        operation["requestBody"] = {"required": True, "content": _json_content(op.request_body)}
    return operation


def _ui_only_spec(operations: tuple[UiOperation, ...], *, openapi_version: str) -> dict[str, object]:
    from apispec import APISpec
    from apispec.ext.marshmallow import MarshmallowPlugin

    spec = APISpec(
        title="Pandects UI API",
        version="v1",
        openapi_version=openapi_version,
        plugins=[MarshmallowPlugin()],
    )
    by_path: dict[str, dict[str, object]] = {}
    for op in operations:
        methods = by_path.setdefault(op.path, {})
        if op.method in methods:
            raise RuntimeError(f"Duplicate UI operation {op.method.upper()} {op.path}")
        methods[op.method] = _operation(op)
    for path, methods in by_path.items():
        _ = spec.path(path=path, operations=methods)
    return spec.to_dict()


def _merge_ui_spec(public: dict[str, object], ui_only: dict[str, object]) -> dict[str, object]:
    merged = copy.deepcopy(public)
    paths = cast(dict[str, dict[str, object]], merged.setdefault("paths", {}))
    for path, methods in cast(dict[str, dict[str, object]], ui_only.get("paths", {})).items():
        target = paths.setdefault(path, {})
        for method, operation in methods.items():
            if method in target:
                raise RuntimeError(f"UI operation {method.upper()} {path} is already public")
            target[method] = operation

    components = cast(dict[str, dict[str, object]], merged.setdefault("components", {}))
    schemas = components.setdefault("schemas", {})
    ui_components = cast(dict[str, dict[str, object]], ui_only.get("components", {}))
    for name, schema in ui_components.get("schemas", {}).items():
        existing = schemas.get(name)
        if existing is not None and existing != schema:
            message = f"Schema component {name!r} differs between the public and UI specs; rename the UI schema class."
            raise RuntimeError(message)
        schemas[name] = schema

    tags = cast(list[dict[str, str]], merged.setdefault("tags", []))
    tags.extend(dict(tag) for tag in _UI_TAGS)
    info = cast(dict[str, object], merged.setdefault("info", {}))
    info["title"] = "Pandects UI API"
    info["description"] = (
        "Generated by backend/scripts/export_openapi.py: the public API plus the internal "
        "routes the Pandects web client calls. Internal routes are not a public contract."
    )
    return merged


def render_specs(app: Flask | None = None) -> tuple[str, str]:
    """Return ``(public_yaml, ui_json)`` for ``app`` (default: ``backend.app.app``)."""
    from backend.extensions import api
    from backend.schemas.ui_api import UI_OPERATIONS

    if app is None:
        from backend.app import app as default_app

        app = default_app
    with app.test_request_context():
        spec = api.spec
        if spec is None:
            raise RuntimeError("OpenAPI spec is unavailable.")
        public_yaml = spec.to_yaml()
        public = spec.to_dict()
    ui_only = _ui_only_spec(UI_OPERATIONS, openapi_version=str(public.get("openapi", "3.0.2")))
    ui = _merge_ui_spec(public, ui_only)
    return public_yaml, json.dumps(ui, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write (or check) the committed OpenAPI specs.")
    _ = parser.add_argument(
        "--check", action="store_true", help="Exit 1 when a committed spec is out of date."
    )
    args = parser.parse_args(argv)
    check = bool(cast(object, args.check))

    _set_offline_env()
    public_yaml, ui_json = render_specs()
    stale: list[Path] = []
    for path, content in ((PUBLIC_SPEC_PATH, public_yaml), (UI_SPEC_PATH, ui_json)):
        current = path.read_text(encoding="utf-8") if path.exists() else None
        if current == content:
            continue
        if check:
            stale.append(path)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        _ = path.write_text(content, encoding="utf-8")
        print(f"Wrote {path.relative_to(REPO_ROOT)}")
    if stale:
        for path in stale:
            print(f"Out of date: {path.relative_to(REPO_ROOT)}", file=sys.stderr)
        hint = "Run `python -m backend.scripts.export_openapi`, then `npm run gen:api` in frontend/, and commit."
        print(hint, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
