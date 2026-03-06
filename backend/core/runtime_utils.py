from __future__ import annotations

import base64
import binascii
import json
import math
from collections.abc import Callable, Iterable, Mapping
from datetime import date, datetime, timezone
from typing import Protocol, SupportsInt, cast
from urllib.request import Request, urlopen

from flask import Flask, Response, abort, current_app, request
from marshmallow import EXCLUDE, Schema, ValidationError


class _HttpResponseReader(Protocol):
    def __enter__(self) -> "_HttpResponseReader":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object | None,
    ) -> bool | None:
        return None

    def read(self, amt: int | None = None) -> bytes:
        ...


def utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def utc_today() -> date:
    return utc_now().date()


def urlopen_read_bytes(req: Request, *, timeout: float = 15) -> bytes:
    with cast(_HttpResponseReader, urlopen(req, timeout=timeout)) as resp:
        return resp.read()


def current_app_object() -> Flask:
    app_obj = cast(object, current_app)
    getter = getattr(app_obj, "_get_current_object", None)
    if callable(getter):
        return cast(Flask, getter())
    return cast(Flask, app_obj)


def to_int(value: object, *, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return int(stripped)
        except ValueError:
            try:
                return int(float(stripped))
            except ValueError:
                return default
    if isinstance(value, SupportsInt):
        try:
            return int(value)
        except (TypeError, ValueError, OverflowError):
            return default
    return default


def row_mapping_as_dict(row: object) -> dict[str, object]:
    if isinstance(row, dict):
        return cast(dict[str, object], row)
    if isinstance(row, Mapping):
        mapping_row = cast(Mapping[object, object], row)
        return {str(key): value for key, value in mapping_row.items()}
    mapping_obj = cast(object, getattr(row, "_mapping", None))
    if mapping_obj is None:
        return {}
    items = getattr(mapping_obj, "items", None)
    if not callable(items):
        return {}
    result: dict[str, object] = {}
    for key, value in cast(Iterable[tuple[object, object]], items()):
        if isinstance(key, str):
            result[key] = value
    return result


def require_json() -> dict[str, object]:
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        abort(400, description="Expected JSON object body.")
    return cast(dict[str, object], data)


JsonErrorFn = Callable[..., Response]


def load_json(
    schema: Schema,
    *,
    json_error: JsonErrorFn,
) -> dict[str, object]:
    data = require_json()
    try:
        loaded = cast(object, schema.load(data, unknown=EXCLUDE))
    except ValidationError as exc:
        current_app.logger.debug("Validation error: %s", cast(object, exc.messages))
        abort(json_error(400, error="validation_error", message="Invalid request body."))
    if not isinstance(loaded, dict):
        abort(400, description="Expected JSON object body.")
    return cast(dict[str, object], loaded)


def load_query(
    schema: Schema,
    *,
    json_error: JsonErrorFn,
) -> dict[str, object]:
    try:
        loaded = cast(object, schema.load(request.args, unknown=EXCLUDE))
    except ValidationError as exc:
        current_app.logger.debug("Validation error: %s", cast(object, exc.messages))
        abort(json_error(400, error="validation_error", message="Invalid query parameters."))
    if not isinstance(loaded, dict):
        abort(400, description="Expected query object.")
    return cast(dict[str, object], loaded)


def pagination_metadata(
    *,
    total_count: int,
    page: int,
    page_size: int,
    has_next_override: bool | None = None,
    total_count_is_approximate: bool = False,
) -> dict[str, object]:
    total_pages = math.ceil(total_count / page_size) if total_count else 0
    if total_count and page > total_pages:
        total_pages = page
    if has_next_override and total_pages <= page:
        total_pages = page + 1
    has_prev = page > 1
    has_next = has_next_override if has_next_override is not None else page < total_pages
    prev_num = page - 1 if has_prev else None
    next_num = page + 1 if has_next else None
    return {
        "page": page,
        "page_size": page_size,
        "total_count": total_count,
        "total_count_is_approximate": total_count_is_approximate,
        "total_pages": total_pages,
        "has_next": has_next,
        "has_prev": has_prev,
        "next_num": next_num,
        "prev_num": prev_num,
    }


def dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def encode_agreements_cursor(agreement_uuid: str) -> str:
    payload = json.dumps({"agreement_uuid": agreement_uuid}, separators=(",", ":"))
    token = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")
    return token.rstrip("=")


def decode_agreements_cursor(cursor_raw: str | None) -> str | None:
    if cursor_raw is None:
        return None
    cursor = cursor_raw.strip()
    if not cursor:
        return None
    padded = cursor + ("=" * (-len(cursor) % 4))
    try:
        decoded_bytes = base64.urlsafe_b64decode(padded.encode("ascii"))
        decoded_obj = cast(object, json.loads(decoded_bytes.decode("utf-8")))
    except (binascii.Error, ValueError, UnicodeDecodeError, json.JSONDecodeError):
        abort(400, description="Invalid cursor.")
    if not isinstance(decoded_obj, dict):
        abort(400, description="Invalid cursor.")
    decoded_dict = cast(dict[str, object], decoded_obj)
    agreement_uuid = decoded_dict.get("agreement_uuid")
    if not isinstance(agreement_uuid, str) or not agreement_uuid.strip():
        abort(400, description="Invalid cursor.")
    return agreement_uuid


def utc_datetime_from_ms(value: object, *, field: str) -> datetime:
    if not isinstance(value, int):
        abort(400, description=f"{field} must be an integer (milliseconds since epoch).")
    if value <= 0:
        abort(400, description=f"{field} must be a positive integer.")
    seconds = value / 1000.0
    if not math.isfinite(seconds):
        abort(400, description=f"{field} must be a finite integer.")
    return datetime.fromtimestamp(seconds, tz=timezone.utc).replace(tzinfo=None)


def request_ip_address(*, is_running_on_fly: bool) -> str | None:
    if is_running_on_fly:
        fly_client_ip = request.headers.get("Fly-Client-IP")
        if isinstance(fly_client_ip, str) and fly_client_ip.strip():
            return fly_client_ip.strip()

        forwarded_for = request.headers.get("X-Forwarded-For")
        if isinstance(forwarded_for, str) and forwarded_for.strip():
            first = forwarded_for.split(",", 1)[0].strip()
            return first or None
    remote = request.remote_addr
    return remote.strip() if isinstance(remote, str) and remote.strip() else None


def request_user_agent() -> str | None:
    ua = request.headers.get("User-Agent")
    if not isinstance(ua, str):
        return None
    ua = ua.strip()
    if not ua:
        return None
    return ua[:512]
