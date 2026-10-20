"""HTTP-level helpers for the agreements blueprint.

Query logic lives in ``backend.services.agreements_service`` and the shared
filter/eligibility builders in ``backend.filtering``. The underscore aliases
below are kept for modules that still import them from here.
"""

from __future__ import annotations

from collections.abc import Mapping

from flask import Response, jsonify

from backend.filtering import agreement_is_public_eligible_expr as _agreement_is_public_eligible_expr
from backend.services.agreements_service import to_float_or_none as _to_float_or_none


def _cacheable_json_response(
    payload: Mapping[str, object],
    *,
    max_age: int,
) -> tuple[Response, int]:
    resp = jsonify(payload)
    resp.headers["Cache-Control"] = f"public, max-age={max_age}"
    return resp, 200


__all__ = [
    "_agreement_is_public_eligible_expr",
    "_cacheable_json_response",
    "_to_float_or_none",
]
