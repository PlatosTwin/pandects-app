"""Favorites, favorite projects, and tags: queries and mutations.

Every function takes the explicit ``FavoritesDeps`` and the signed-in user's
id. Request parsing, auth gates, and HTTP shaping stay in
``backend/routes/favorites``.

Error contract: a client-facing failure raises ``FavoritesError`` (the route
turns it into the matching HTTP error); a database failure propagates as
``SQLAlchemyError`` (the route rolls back and answers 503). Enrichment lookups
for response payloads (agreement UUIDs, tags, project ids) degrade to empty
results on a database error instead of failing the request.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, cast

from marshmallow import Schema
from sqlalchemy.exc import SQLAlchemyError

from backend.filtering import agreement_is_public_eligible_expr
from backend.routes.deps import AccessContextProtocol, UserLikeProtocol
from backend.services.agreements_service import to_float_or_none
from backend.services.taxonomy_service import standard_id_labels

DEFAULT_PROJECT_NAME = "Scratchpad"


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


class FavoritesError(Exception):
    """A client-facing failure: HTTP status plus the message to show."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status: int = status
        self.message: str = message


def db_session(deps: FavoritesDeps) -> Any:
    return cast(Any, deps.db).session


# ── Serializers ───────────────────────────────────────────────────────────
def serialize_favorite(
    fav: Any,
    *,
    agreement_uuid: str | None,
    tags: list[dict[str, object]] | None = None,
    project_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "id": fav.id,
        "project_id": fav.project_id,
        "project_ids": project_ids or [fav.project_id],
        "item_type": fav.item_type,
        "item_uuid": fav.item_uuid,
        "agreement_uuid": agreement_uuid,
        "note": fav.note,
        "context": fav.context,
        "tags": tags or [],
        "created_at": fav.created_at.isoformat() if fav.created_at else None,
        "updated_at": fav.updated_at.isoformat() if fav.updated_at else None,
    }


def serialize_favorite_minimal(fav: Any) -> dict[str, object]:
    return {
        "id": fav.id,
        "project_id": fav.project_id,
        "project_ids": [fav.project_id],
        "item_type": fav.item_type,
        "item_uuid": fav.item_uuid,
        "note": fav.note,
        "context": fav.context,
        "created_at": fav.created_at.isoformat() if fav.created_at else None,
        "updated_at": fav.updated_at.isoformat() if fav.updated_at else None,
    }


def serialize_project(project: Any) -> dict[str, object]:
    return {
        "id": project.id,
        "name": project.name,
        "color": project.color,
        "is_default": bool(project.is_default),
        "sort_order": project.sort_order,
        "created_at": project.created_at.isoformat() if project.created_at else None,
    }


def serialize_tag(tag: Any) -> dict[str, object]:
    return {
        "id": tag.id,
        "name": tag.name,
        "color": tag.color,
        "created_at": tag.created_at.isoformat() if tag.created_at else None,
    }


# ── Shared loaders ────────────────────────────────────────────────────────
def ensure_default_project(deps: FavoritesDeps, *, user_id: str) -> Any:
    project = cast(Any, deps.FavoriteProject).query.filter_by(
        user_id=user_id, is_default=True
    ).first()
    if project is not None:
        return project
    project = cast(Any, deps.FavoriteProject)()
    project.id = str(uuid.uuid4())
    project.user_id = user_id
    project.name = DEFAULT_PROJECT_NAME
    project.color = "slate"
    project.is_default = True
    project.sort_order = 0
    db_session(deps).add(project)
    db_session(deps).commit()
    return project


def _owned_project(deps: FavoritesDeps, *, user_id: str, project_id: str) -> Any:
    return cast(Any, deps.FavoriteProject).query.filter_by(id=project_id, user_id=user_id).first()


def _owned_favorite(deps: FavoritesDeps, *, user_id: str, favorite_id: str) -> Any:
    fav = cast(Any, deps.Favorite).query.filter_by(id=favorite_id, user_id=user_id).first()
    if fav is None:
        raise FavoritesError(404, "Favorite not found.")
    return fav


def _owned_favorites_exactly(deps: FavoritesDeps, *, user_id: str, favorite_ids: list[str]) -> list[Any]:
    favorite = cast(Any, deps.Favorite)
    rows = cast(
        list[Any],
        favorite.query.filter(favorite.user_id == user_id, favorite.id.in_(favorite_ids)).all(),
    )
    if {row.id for row in rows} != set(favorite_ids):
        raise FavoritesError(404, "One or more favorites not found.")
    return rows


def _owned_tag_ids_exactly(deps: FavoritesDeps, *, user_id: str, tag_ids: list[str]) -> list[str]:
    tag_model = cast(Any, deps.FavoriteTag)
    tags = cast(
        list[Any],
        tag_model.query.filter(tag_model.user_id == user_id, tag_model.id.in_(tag_ids)).all(),
    )
    valid_tag_ids = [tag.id for tag in tags]
    if len(valid_tag_ids) != len(set(tag_ids)):
        raise FavoritesError(404, "One or more tags not found.")
    return valid_tag_ids


def _next_project_sort_order(deps: FavoritesDeps, *, user_id: str) -> int:
    rows = cast(list[Any], cast(Any, deps.FavoriteProject).query.filter_by(user_id=user_id).all())
    if not rows:
        return 0
    return max(int(row.sort_order or 0) for row in rows) + 1


def load_tags_for_favorites(
    deps: FavoritesDeps, *, favorite_ids: list[str]
) -> dict[str, list[dict[str, object]]]:
    if not favorite_ids:
        return {}
    assignment = cast(Any, deps.FavoriteTagAssignment)
    tag_model = cast(Any, deps.FavoriteTag)
    rows = (
        db_session(deps)
        .query(assignment.favorite_id, tag_model.id, tag_model.name, tag_model.color)
        .join(tag_model, tag_model.id == assignment.tag_id)
        .filter(assignment.favorite_id.in_(favorite_ids))
        .order_by(tag_model.name.asc())
        .all()
    )
    out: dict[str, list[dict[str, object]]] = {}
    for favorite_id, tag_id, name, color in rows:
        out.setdefault(favorite_id, []).append({"id": tag_id, "name": name, "color": color})
    return out


def load_projects_for_favorites(
    deps: FavoritesDeps, *, favorite_ids: list[str]
) -> dict[str, list[str]]:
    if not favorite_ids:
        return {}
    assignment = cast(Any, deps.FavoriteProjectAssignment)
    rows = (
        db_session(deps)
        .query(assignment.favorite_id, assignment.project_id)
        .filter(assignment.favorite_id.in_(favorite_ids))
        .all()
    )
    out: dict[str, list[str]] = {}
    for favorite_id, project_id in rows:
        out.setdefault(favorite_id, []).append(project_id)
    return out


def ensure_favorite_project_assignment(
    deps: FavoritesDeps, *, favorite_id: str, project_id: str
) -> None:
    assignment_model = cast(Any, deps.FavoriteProjectAssignment)
    existing = assignment_model.query.filter_by(
        favorite_id=favorite_id, project_id=project_id
    ).first()
    if existing is not None:
        return
    assignment = assignment_model()
    assignment.favorite_id = favorite_id
    assignment.project_id = project_id
    db_session(deps).add(assignment)


def _backfill_favorite_project_assignments(deps: FavoritesDeps, *, user_id: str) -> None:
    favorites = cast(list[Any], cast(Any, deps.Favorite).query.filter_by(user_id=user_id).all())
    for fav in favorites:
        ensure_favorite_project_assignment(deps, favorite_id=fav.id, project_id=fav.project_id)


def resolve_agreement_uuids(
    deps: FavoritesDeps, *, favorites: list[Any]
) -> dict[tuple[str, str], str]:
    section_uuids: list[str] = []
    clause_uuids: list[str] = []
    agreement_uuids: list[str] = []
    for fav in favorites:
        item_type = cast(str, fav.item_type)
        item_uuid = cast(str, fav.item_uuid)
        if item_type == "section":
            section_uuids.append(item_uuid)
        elif item_type == "tax_clause":
            clause_uuids.append(item_uuid)
        elif item_type == "agreement":
            agreement_uuids.append(item_uuid)

    resolved: dict[tuple[str, str], str] = {}
    session = db_session(deps)
    sections = cast(Any, deps.Sections)
    clauses = cast(Any, deps.Clauses)
    if section_uuids:
        rows = (
            session.query(sections.section_uuid, sections.agreement_uuid)
            .filter(sections.section_uuid.in_(section_uuids))
            .all()
        )
        for section_uuid, agreement_uuid in rows:
            resolved[("section", section_uuid)] = agreement_uuid
    if clause_uuids:
        rows = (
            session.query(clauses.clause_uuid, clauses.agreement_uuid)
            .filter(clauses.clause_uuid.in_(clause_uuids))
            .all()
        )
        for clause_uuid, agreement_uuid in rows:
            resolved[("tax_clause", clause_uuid)] = agreement_uuid
    for agreement_uuid in agreement_uuids:
        resolved[("agreement", agreement_uuid)] = agreement_uuid
    return resolved


def favorite_payloads(deps: FavoritesDeps, favorites: list[Any]) -> list[dict[str, object]]:
    """Full favorite payloads, enriched best-effort with agreement, tags, and projects."""
    favorite_ids = [fav.id for fav in favorites]
    try:
        agreement_map = resolve_agreement_uuids(deps, favorites=favorites)
    except SQLAlchemyError:
        agreement_map = {}
    try:
        tag_map = load_tags_for_favorites(deps, favorite_ids=favorite_ids)
    except SQLAlchemyError:
        tag_map = {}
    try:
        project_map = load_projects_for_favorites(deps, favorite_ids=favorite_ids)
    except SQLAlchemyError:
        project_map = {}
    return [
        serialize_favorite(
            fav,
            agreement_uuid=agreement_map.get((fav.item_type, fav.item_uuid)),
            tags=tag_map.get(fav.id, []),
            project_ids=project_map.get(fav.id, [fav.project_id]),
        )
        for fav in favorites
    ]


def favorite_tags_best_effort(deps: FavoritesDeps, favorite_id: str) -> list[dict[str, object]]:
    try:
        tag_map = load_tags_for_favorites(deps, favorite_ids=[favorite_id])
    except SQLAlchemyError:
        tag_map = {}
    return tag_map.get(favorite_id, [])


def _year_from_filing_date(value: object) -> int | None:
    if isinstance(value, (date, datetime)):
        return value.year
    if isinstance(value, str) and len(value) >= 4 and value[:4].isdigit():
        return int(value[:4])
    return None


def load_agreement_metadata(
    deps: FavoritesDeps, *, agreement_uuids: list[str]
) -> dict[str, dict[str, object]]:
    """Batch-load display metadata for agreements the favorites UI renders.

    Only touches serving-schema tables (prod lacks ETL-only tables) and applies
    the same public-eligibility filter and latest-verified-XML join as
    /v1/agreements/<uuid>, so batch misses match that endpoint's 404s.
    """
    if not agreement_uuids:
        return {}
    agreements = cast(Any, deps.Agreements)
    xml = cast(Any, deps.XML)
    rows = (
        db_session(deps)
        .query(
            agreements.agreement_uuid,
            agreements.filing_date,
            agreements.target,
            agreements.acquirer,
            agreements.transaction_price_total,
        )
        .join(xml, deps._agreement_latest_xml_join_condition())
        .filter(
            agreements.agreement_uuid.in_(agreement_uuids),
            agreement_is_public_eligible_expr(agreements),
        )
        .all()
    )
    out: dict[str, dict[str, object]] = {}
    for agreement_uuid, filing_date, target, acquirer, price_total in rows:
        out[agreement_uuid] = {
            "agreement_uuid": agreement_uuid,
            "year": _year_from_filing_date(filing_date),
            "target": target,
            "acquirer": acquirer,
            "transaction_price_total": to_float_or_none(price_total),
        }
    return out


def load_section_details(
    deps: FavoritesDeps, *, section_uuids: list[str]
) -> dict[str, dict[str, object]]:
    """Batch-load section payloads for section favorites.

    Mirrors GET /v1/sections/<uuid>: serving-schema `sections` joined to the
    latest verified `xml` version, so favorites pointing at stale or unknown
    sections resolve to misses rather than partial rows.
    """
    if not section_uuids:
        return {}
    section_cols = cast(Any, deps.Sections).__table__.c
    rows = (
        db_session(deps)
        .query(
            section_cols["agreement_uuid"],
            section_cols["section_uuid"],
            deps._coalesced_section_standard_ids().label("section_standard_ids"),
            section_cols["xml_content"],
            section_cols["article_title"],
            section_cols["section_title"],
        )
        .join(deps.XML, deps._section_latest_xml_join_condition())
        .filter(section_cols["section_uuid"].in_(section_uuids))
        .all()
    )
    out: dict[str, dict[str, object]] = {}
    for (
        agreement_uuid,
        section_uuid,
        section_standard_ids_raw,
        xml_content,
        article_title,
        section_title,
    ) in rows:
        section_standard_id = deps._parse_section_standard_ids(section_standard_ids_raw)
        out[section_uuid] = {
            "agreement_uuid": agreement_uuid,
            "section_uuid": section_uuid,
            "section_standard_id": section_standard_id,
            "standard_id_labels": standard_id_labels(section_standard_id),
            "xml": xml_content,
            "article_title": article_title,
            "section_title": section_title,
        }
    return out


# ── Projects ──────────────────────────────────────────────────────────────
def list_projects(deps: FavoritesDeps, *, user_id: str) -> list[dict[str, object]]:
    _ = ensure_default_project(deps, user_id=user_id)
    _backfill_favorite_project_assignments(deps, user_id=user_id)
    db_session(deps).commit()
    project_model = cast(Any, deps.FavoriteProject)
    projects = cast(
        list[Any],
        project_model.query.filter_by(user_id=user_id)
        .order_by(
            project_model.is_default.desc(),
            project_model.sort_order.asc(),
            project_model.created_at.asc(),
        )
        .all(),
    )
    return [serialize_project(p) for p in projects]


def create_project(deps: FavoritesDeps, *, user_id: str, name: str, color: str) -> dict[str, object]:
    _ = ensure_default_project(deps, user_id=user_id)
    project_model = cast(Any, deps.FavoriteProject)
    if project_model.query.filter_by(user_id=user_id, name=name).first() is not None:
        raise FavoritesError(409, "A project with that name already exists.")
    project = project_model()
    project.id = str(uuid.uuid4())
    project.user_id = user_id
    project.name = name
    project.color = color
    project.is_default = False
    project.sort_order = _next_project_sort_order(deps, user_id=user_id)
    db_session(deps).add(project)
    db_session(deps).commit()
    return serialize_project(project)


def update_project(
    deps: FavoritesDeps, *, user_id: str, project_id: str, data: dict[str, object]
) -> dict[str, object]:
    project = _owned_project(deps, user_id=user_id, project_id=project_id)
    if project is None:
        raise FavoritesError(404, "Project not found.")
    if "name" in data:
        new_name = cast(str, data["name"]).strip()
        if not new_name:
            raise FavoritesError(400, "Project name is required.")
        if new_name != project.name:
            clash = cast(Any, deps.FavoriteProject).query.filter_by(
                user_id=user_id, name=new_name
            ).first()
            if clash is not None and clash.id != project.id:
                raise FavoritesError(409, "A project with that name already exists.")
            project.name = new_name
    if "color" in data:
        project.color = cast(str, data["color"])
    if "sort_order" in data:
        project.sort_order = cast(int, data["sort_order"])
    db_session(deps).commit()
    return serialize_project(project)


def delete_project(
    deps: FavoritesDeps, *, user_id: str, project_id: str, reassign_project_id: str | None
) -> dict[str, object]:
    """Delete a project, moving its favorites to another project first."""
    project_model = cast(Any, deps.FavoriteProject)
    project = _owned_project(deps, user_id=user_id, project_id=project_id)
    if project is None:
        raise FavoritesError(404, "Project not found.")
    if reassign_project_id:
        target = _owned_project(deps, user_id=user_id, project_id=reassign_project_id)
        if target is None or target.id == project.id:
            raise FavoritesError(400, "Choose a different project to reassign favorites.")
    else:
        default_project = ensure_default_project(deps, user_id=user_id)
        if default_project.id != project.id:
            target = default_project
        else:
            target = (
                project_model.query.filter(
                    project_model.user_id == user_id,
                    project_model.id != project.id,
                )
                .order_by(project_model.sort_order.asc())
                .first()
            )
    if target is None:
        raise FavoritesError(
            400,
            "Create another project or provide reassign_project_id before deleting this project.",
        )
    session = db_session(deps)
    assignment_model = cast(Any, deps.FavoriteProjectAssignment)
    assigned_rows = (
        session.query(assignment_model.favorite_id).filter_by(project_id=project.id).all()
    )
    affected_favorite_ids = [row[0] for row in assigned_rows]
    moved = cast(Any, deps.Favorite).query.filter_by(
        user_id=user_id, project_id=project.id
    ).update({"project_id": target.id}, synchronize_session=False)
    session.query(assignment_model).filter_by(project_id=project.id).delete(
        synchronize_session=False
    )
    for favorite_id in affected_favorite_ids:
        ensure_favorite_project_assignment(deps, favorite_id=favorite_id, project_id=target.id)
    session.delete(project)
    session.commit()
    return {"deleted": True, "reassigned_to_project_id": target.id, "moved": moved}


# ── Favorites ─────────────────────────────────────────────────────────────
def list_favorites(
    deps: FavoritesDeps,
    *,
    user_id: str,
    item_type: str | None,
    project_id: str | None,
    tag_ids: list[str],
) -> list[Any]:
    """Favorites newest first, filtered by type, project, and ALL of ``tag_ids``."""
    favorite = cast(Any, deps.Favorite)
    q = favorite.query.filter_by(user_id=user_id)
    if item_type:
        q = q.filter_by(item_type=item_type)
    if project_id:
        project_assignment = cast(Any, deps.FavoriteProjectAssignment)
        sub = (
            db_session(deps)
            .query(project_assignment.favorite_id)
            .filter(project_assignment.project_id == project_id)
        )
        q = q.filter(favorite.id.in_(sub))
    tag_assignment = cast(Any, deps.FavoriteTagAssignment)
    for tag_id in tag_ids:
        sub = db_session(deps).query(tag_assignment.favorite_id).filter(
            tag_assignment.tag_id == tag_id
        )
        q = q.filter(favorite.id.in_(sub))
    return cast(list[Any], q.order_by(favorite.created_at.desc()).all())


def favorite_ids_by_item_uuid(
    deps: FavoritesDeps, *, user_id: str, item_type: str, item_uuids: list[str]
) -> dict[str, str]:
    favorite = cast(Any, deps.Favorite)
    rows = cast(
        list[Any],
        favorite.query.filter(
            favorite.user_id == user_id,
            favorite.item_type == item_type,
            favorite.item_uuid.in_(item_uuids),
        ).all(),
    )
    return {row.item_uuid: row.id for row in rows}


def upsert_favorite(
    deps: FavoritesDeps,
    *,
    user_id: str,
    item_type: str,
    item_uuid: str,
    note: str | None,
    context: dict[str, object] | None,
    project_id: str | None,
) -> tuple[Any, bool]:
    """Create a favorite, or update the existing one for the same item.

    Returns ``(favorite, created)``. With no ``project_id`` the favorite goes
    to the user's default project.
    """
    if project_id:
        project = _owned_project(deps, user_id=user_id, project_id=project_id)
        if project is None:
            raise FavoritesError(404, "Project not found.")
    else:
        project = ensure_default_project(deps, user_id=user_id)

    existing = cast(Any, deps.Favorite).query.filter_by(
        user_id=user_id, item_type=item_type, item_uuid=item_uuid
    ).first()
    if existing is not None:
        if note is not None:
            existing.note = note
        if context is not None:
            existing.context = context
        existing.project_id = project.id
        existing.updated_at = deps._utc_now()
        ensure_favorite_project_assignment(deps, favorite_id=existing.id, project_id=project.id)
        db_session(deps).commit()
        return existing, False

    fav = cast(Any, deps.Favorite)()
    fav.id = str(uuid.uuid4())
    fav.user_id = user_id
    fav.project_id = project.id
    fav.item_type = item_type
    fav.item_uuid = item_uuid
    fav.note = note
    fav.context = context
    db_session(deps).add(fav)
    ensure_favorite_project_assignment(deps, favorite_id=fav.id, project_id=project.id)
    db_session(deps).commit()
    return fav, True


def update_favorite(
    deps: FavoritesDeps, *, user_id: str, favorite_id: str, data: dict[str, object]
) -> Any:
    fav = _owned_favorite(deps, user_id=user_id, favorite_id=favorite_id)
    if "note" in data:
        note_val = data.get("note")
        fav.note = note_val.strip() if isinstance(note_val, str) and note_val.strip() else None
    if "project_id" in data:
        project_id = data.get("project_id")
        if project_id is None:
            fav.project_id = ensure_default_project(deps, user_id=user_id).id
        else:
            if not isinstance(project_id, str):
                raise FavoritesError(400, "project_id must be a string.")
            project = _owned_project(deps, user_id=user_id, project_id=project_id)
            if project is None:
                raise FavoritesError(404, "Project not found.")
            fav.project_id = project.id
            ensure_favorite_project_assignment(deps, favorite_id=fav.id, project_id=project.id)
    fav.updated_at = deps._utc_now()
    db_session(deps).commit()
    return fav


def bulk_move_favorites(
    deps: FavoritesDeps, *, user_id: str, favorite_ids: list[str], project_id: str
) -> dict[str, object]:
    """Make ``project_id`` the only project of each favorite."""
    project = _owned_project(deps, user_id=user_id, project_id=project_id)
    if project is None:
        raise FavoritesError(404, "Project not found.")
    rows = _owned_favorites_exactly(deps, user_id=user_id, favorite_ids=favorite_ids)
    now = deps._utc_now()
    session = db_session(deps)
    assignment_model = cast(Any, deps.FavoriteProjectAssignment)
    for fav in rows:
        fav.project_id = project.id
        fav.updated_at = now
        session.query(assignment_model).filter_by(favorite_id=fav.id).delete(
            synchronize_session=False
        )
        ensure_favorite_project_assignment(deps, favorite_id=fav.id, project_id=project.id)
    session.commit()
    return {
        "project_id": project.id,
        "favorite_ids": [row.id for row in rows],
        "moved": len(rows),
    }


def bulk_copy_favorites(
    deps: FavoritesDeps, *, user_id: str, favorite_ids: list[str], project_ids: list[str]
) -> dict[str, object]:
    """Add each favorite to every project in ``project_ids`` (keeping existing ones)."""
    project_model = cast(Any, deps.FavoriteProject)
    projects = cast(
        list[Any],
        project_model.query.filter(
            project_model.user_id == user_id,
            project_model.id.in_(project_ids),
        ).all(),
    )
    if {project.id for project in projects} != set(project_ids):
        raise FavoritesError(404, "One or more projects not found.")
    rows = _owned_favorites_exactly(deps, user_id=user_id, favorite_ids=favorite_ids)
    session = db_session(deps)
    for fav in rows:
        for project_id in project_ids:
            ensure_favorite_project_assignment(deps, favorite_id=fav.id, project_id=project_id)
        fav.updated_at = deps._utc_now()
    session.commit()
    return {
        "project_ids": project_ids,
        "favorite_ids": [row.id for row in rows],
        "copied": len(rows),
    }


def bulk_update_favorite_tags(
    deps: FavoritesDeps,
    *,
    user_id: str,
    favorite_ids: list[str],
    tag_ids: list[str],
    action: str,
) -> dict[str, object]:
    """Add (``action == "add"``) or remove ``tag_ids`` on every favorite."""
    rows = _owned_favorites_exactly(deps, user_id=user_id, favorite_ids=favorite_ids)
    valid_tag_ids = _owned_tag_ids_exactly(deps, user_id=user_id, tag_ids=tag_ids)

    session = db_session(deps)
    assignment_model = cast(Any, deps.FavoriteTagAssignment)
    now = deps._utc_now()
    if action == "add":
        existing = {
            (favorite_id, tag_id)
            for favorite_id, tag_id in session.query(
                assignment_model.favorite_id,
                assignment_model.tag_id,
            )
            .filter(assignment_model.favorite_id.in_(favorite_ids))
            .filter(assignment_model.tag_id.in_(valid_tag_ids))
            .all()
        }
        for favorite_id in favorite_ids:
            for tag_id in valid_tag_ids:
                if (favorite_id, tag_id) in existing:
                    continue
                assignment = assignment_model()
                assignment.favorite_id = favorite_id
                assignment.tag_id = tag_id
                session.add(assignment)
    else:
        session.query(assignment_model).filter(
            assignment_model.favorite_id.in_(favorite_ids),
            assignment_model.tag_id.in_(valid_tag_ids),
        ).delete(synchronize_session=False)
    for fav in rows:
        fav.updated_at = now
    session.commit()
    tag_map = load_tags_for_favorites(deps, favorite_ids=[row.id for row in rows])
    return {
        "action": action,
        "tag_ids": valid_tag_ids,
        "favorite_ids": [row.id for row in rows],
        "tags_by_favorite": {row.id: tag_map.get(row.id, []) for row in rows},
        "updated": len(rows),
    }


def delete_favorite(deps: FavoritesDeps, *, user_id: str, favorite_id: str) -> None:
    fav = _owned_favorite(deps, user_id=user_id, favorite_id=favorite_id)
    session = db_session(deps)
    session.query(deps.FavoriteTagAssignment).filter_by(favorite_id=fav.id).delete(
        synchronize_session=False
    )
    session.query(deps.FavoriteProjectAssignment).filter_by(favorite_id=fav.id).delete(
        synchronize_session=False
    )
    session.delete(fav)
    session.commit()


def get_owned_favorite(deps: FavoritesDeps, *, user_id: str, favorite_id: str) -> Any:
    return _owned_favorite(deps, user_id=user_id, favorite_id=favorite_id)


def set_favorite_tags(
    deps: FavoritesDeps, *, user_id: str, favorite_id: str, tag_ids: list[str]
) -> Any:
    """Replace a favorite's tags with ``tag_ids``; returns the favorite."""
    fav = _owned_favorite(deps, user_id=user_id, favorite_id=favorite_id)
    valid_ids: list[str] = []
    if tag_ids:
        valid_ids = _owned_tag_ids_exactly(deps, user_id=user_id, tag_ids=tag_ids)
    session = db_session(deps)
    assignment_model = cast(Any, deps.FavoriteTagAssignment)
    session.query(assignment_model).filter_by(favorite_id=fav.id).delete(
        synchronize_session=False
    )
    for tag_id in valid_ids:
        assignment = assignment_model()
        assignment.favorite_id = fav.id
        assignment.tag_id = tag_id
        session.add(assignment)
    fav.updated_at = deps._utc_now()
    session.commit()
    return fav


def set_favorite_projects(
    deps: FavoritesDeps, *, user_id: str, favorite_id: str, project_ids: list[str]
) -> list[str]:
    """Replace a favorite's projects; its primary project stays if still listed."""
    fav = _owned_favorite(deps, user_id=user_id, favorite_id=favorite_id)
    project_model = cast(Any, deps.FavoriteProject)
    rows = cast(
        list[Any],
        project_model.query.filter(
            project_model.user_id == user_id,
            project_model.id.in_(project_ids),
        ).all(),
    )
    valid_ids = [row.id for row in rows]
    if len(valid_ids) != len(set(project_ids)):
        raise FavoritesError(404, "One or more projects not found.")
    session = db_session(deps)
    session.query(deps.FavoriteProjectAssignment).filter_by(favorite_id=fav.id).delete(
        synchronize_session=False
    )
    for project_id in valid_ids:
        ensure_favorite_project_assignment(deps, favorite_id=fav.id, project_id=project_id)
    if fav.project_id not in valid_ids:
        fav.project_id = valid_ids[0]
    fav.updated_at = deps._utc_now()
    session.commit()
    return valid_ids


# ── Tags ──────────────────────────────────────────────────────────────────
def list_tags(deps: FavoritesDeps, *, user_id: str) -> list[dict[str, object]]:
    tag_model = cast(Any, deps.FavoriteTag)
    tags = cast(
        list[Any],
        tag_model.query.filter_by(user_id=user_id).order_by(tag_model.name.asc()).all(),
    )
    return [serialize_tag(t) for t in tags]


def create_tag(
    deps: FavoritesDeps, *, user_id: str, name: str, color: str
) -> tuple[dict[str, object], bool]:
    """Create a tag, or return the existing one with that name. ``(tag, created)``."""
    tag_model = cast(Any, deps.FavoriteTag)
    existing = tag_model.query.filter_by(user_id=user_id, name=name).first()
    if existing is not None:
        return serialize_tag(existing), False
    tag = tag_model()
    tag.id = str(uuid.uuid4())
    tag.user_id = user_id
    tag.name = name
    tag.color = color
    db_session(deps).add(tag)
    db_session(deps).commit()
    return serialize_tag(tag), True


def update_tag(
    deps: FavoritesDeps, *, user_id: str, tag_id: str, data: dict[str, object]
) -> dict[str, object]:
    tag_model = cast(Any, deps.FavoriteTag)
    tag = tag_model.query.filter_by(id=tag_id, user_id=user_id).first()
    if tag is None:
        raise FavoritesError(404, "Tag not found.")
    if "name" in data:
        new_name = cast(str, data["name"]).strip()
        if not new_name:
            raise FavoritesError(400, "Tag name is required.")
        if new_name != tag.name:
            clash = tag_model.query.filter_by(user_id=user_id, name=new_name).first()
            if clash is not None and clash.id != tag.id:
                raise FavoritesError(409, "A tag with that name already exists.")
            tag.name = new_name
    if "color" in data:
        tag.color = cast(str, data["color"])
    db_session(deps).commit()
    return serialize_tag(tag)


def delete_tag(deps: FavoritesDeps, *, user_id: str, tag_id: str) -> None:
    tag = cast(Any, deps.FavoriteTag).query.filter_by(id=tag_id, user_id=user_id).first()
    if tag is None:
        raise FavoritesError(404, "Tag not found.")
    session = db_session(deps)
    session.query(deps.FavoriteTagAssignment).filter_by(tag_id=tag.id).delete(
        synchronize_session=False
    )
    session.delete(tag)
    session.commit()
