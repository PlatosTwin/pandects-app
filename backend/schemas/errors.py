"""The `/v1` error envelope (see ``backend/core/errors.py``)."""

from __future__ import annotations

from marshmallow import Schema, fields


class ErrorSchema(Schema):
    error = fields.Str(
        required=True,
        metadata={
            "description": (
                "Stable snake_case machine code, for example `not_found`, "
                "`validation_error`, or `rate_limited`."
            ),
            "example": "not_found",
        },
    )
    message = fields.Str(
        required=True,
        metadata={"description": "Human-readable explanation; wording may change."},
    )


__all__ = ["ErrorSchema"]
