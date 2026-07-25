"""Dependency container for the favorites blueprint.

Split from ``__init__`` so helper submodules can import the type without
circularity.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from marshmallow import Schema

from backend.routes.deps import AccessContextProtocol, UserLikeProtocol


@dataclass(frozen=True)
class FavoritesDeps:
    Favorite: type
    FavoriteProject: type
    FavoriteProjectAssignment: type
    FavoriteTag: type
    FavoriteTagAssignment: type
    Sections: type
    Clauses: type
    Agreements: type
    XML: type
    db: object
    _section_latest_xml_join_condition: Callable[[], Any]
    _agreement_latest_xml_join_condition: Callable[[], Any]
    _coalesced_section_standard_ids: Callable[..., Any]
    _parse_section_standard_ids: Callable[[object], list[str]]
    _require_auth_db: Callable[[], None]
    _require_verified_user: Callable[[], tuple[UserLikeProtocol, AccessContextProtocol]]
    _auth_is_mocked: Callable[[], bool]
    _load_json: Callable[[Schema], dict[str, object]]
    _utc_now: Callable[[], datetime]
