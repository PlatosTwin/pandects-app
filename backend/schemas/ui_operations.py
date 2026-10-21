"""Registry type for the web client's internal `/v1` operations.

Most UI-only routes (summaries, filter options, auth, favorites) are plain Flask
views, not flask-smorest resources, so ``@blp.response`` cannot document them.
Each one is listed as a ``UiOperation`` instead (see ``ui_api.py``,
``ui_auth.py``, ``ui_favorites.py``); ``backend/scripts/export_openapi.py``
adds them to the UI spec (``backend/openapi/ui-openapi.json``) next to the
public paths. None of this is used at request time.

``backend/tests/test_openapi_spec.py`` fails when a ``/v1`` Flask rule is in
neither the public spec nor this registry (nor ``UNDOCUMENTED_V1_RULES``).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from marshmallow import Schema

SchemaRef = type[Schema] | Schema


@dataclass(frozen=True)
class UiOperation:
    path: str
    """OpenAPI path template, e.g. ``/v1/me/tags/{tag_id}``."""
    method: str
    operation_id: str
    tag: str
    summary: str
    response: SchemaRef | None
    """JSON body for ``status``; ``None`` documents a response with no JSON body."""
    status: int = 200
    response_description: str = "Success"
    description: str | None = None
    query: type[Schema] | None = None
    request_body: SchemaRef | None = None
    int_path_params: tuple[str, ...] = ()
    """Path parameters (read from ``path``) that are integers; the rest are strings."""
    extra_responses: dict[int, tuple[str, SchemaRef | None]] = field(default_factory=dict)
    """Other documented statuses: ``{status: (description, schema or None)}``."""


__all__ = ["SchemaRef", "UiOperation"]
