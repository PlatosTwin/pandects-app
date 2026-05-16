from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, cast

from marshmallow import Schema, ValidationError

from backend.auth.mcp_runtime import McpPrincipal


@dataclass(frozen=True)
class McpToolResult:
    text: str
    structured_content: object


class McpOutputValidationError(ValueError):
    def __init__(self, messages: dict[str, object]):
        super().__init__("Tool result did not match the advertised output schema.")
        self.messages = messages


@dataclass(frozen=True)
class McpToolSpec:
    name: str
    description: str
    input_schema: dict[str, object]
    output_schema: dict[str, object]
    examples: tuple[dict[str, object], ...]
    response_examples: tuple[dict[str, object], ...]
    scopes: tuple[str, ...]
    selection_hint: str
    negative_guidance: tuple[str, ...]
    pagination: str
    access_behavior: str
    redaction_behavior: str
    fulltext_scope: str | None
    handler: Callable[..., McpToolResult]


def _require_scope(principal: McpPrincipal, scope: str) -> None:
    if scope in principal.scopes:
        return
    raise PermissionError(f"Missing required scope: {scope}")


def _validate_payload(schema: Schema, payload: dict[str, object]) -> dict[str, object]:
    try:
        loaded = schema.load(payload)
    except ValidationError:
        raise
    return cast(dict[str, object], loaded)


def _matches_schema_type(expected_type: object, value: object) -> bool:
    if expected_type == "object":
        return isinstance(value, dict)
    if expected_type == "array":
        return isinstance(value, list)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def _validate_output_against_schema(
    schema: dict[str, object],
    value: object,
    *,
    path: str = "structuredContent",
) -> dict[str, object]:
    errors: dict[str, object] = {}
    expected_type = schema.get("type")
    if isinstance(expected_type, list):
        if not any(_matches_schema_type(type_name, value) for type_name in expected_type):
            errors[path] = f"Expected one of {expected_type}, got {type(value).__name__}."
            return errors
    elif expected_type is not None and not _matches_schema_type(expected_type, value):
        errors[path] = f"Expected {expected_type}, got {type(value).__name__}."
        return errors

    if isinstance(value, dict):
        properties = cast(dict[str, dict[str, object]], schema.get("properties", {}))
        required = cast(list[str], schema.get("required", []))
        for field_name in required:
            if field_name not in value:
                errors[f"{path}.{field_name}"] = "Missing required field."
        for field_name, field_schema in properties.items():
            if field_name in value:
                errors.update(
                    _validate_output_against_schema(
                        field_schema,
                        value[field_name],
                        path=f"{path}.{field_name}",
                    )
                )
        additional_properties = schema.get("additionalProperties", True)
        extra_keys = [field_name for field_name in value.keys() if field_name not in properties]
        if additional_properties is False:
            for field_name in extra_keys:
                errors[f"{path}.{field_name}"] = "Unexpected field."
        elif isinstance(additional_properties, dict):
            for field_name in extra_keys:
                errors.update(
                    _validate_output_against_schema(
                        additional_properties,
                        value[field_name],
                        path=f"{path}.{field_name}",
                    )
                )
        return errors

    if isinstance(value, list):
        item_schema = cast(dict[str, object], schema.get("items", {}))
        for index, item in enumerate(value):
            errors.update(_validate_output_against_schema(item_schema, item, path=f"{path}[{index}]"))
        return errors

    enum_values = schema.get("enum")
    if isinstance(enum_values, list) and value not in enum_values:
        errors[path] = f"Unexpected enum value: {value!r}."
    return errors
