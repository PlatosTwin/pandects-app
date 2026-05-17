"""Module-level helpers for the favorites blueprint.

Serializers, project/tag loaders, and small DB utilities. Each helper takes
the explicit ``FavoritesDeps`` instance rather than capturing it from a
closure, so they live cleanly outside ``register_favorites_routes``.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from flask import jsonify, make_response

from backend.routes.favorites.deps import FavoritesDeps


_DEFAULT_PROJECT_NAME = "Scratchpad"


def _db_session(deps: FavoritesDeps) -> Any:
    return cast(Any, deps.db).session


def _ensure_default_project(deps: FavoritesDeps, *, user_id: str):
    project = deps.FavoriteProject.query.filter_by(
        user_id=user_id, is_default=True
    ).first()
    if project is not None:
        return project
    project = deps.FavoriteProject()  # type: ignore[call-arg]
    project.id = str(uuid.uuid4())
    project.user_id = user_id
    project.name = _DEFAULT_PROJECT_NAME
    project.color = "slate"
    project.is_default = True
    project.sort_order = 0
    _db_session(deps).add(project)
    _db_session(deps).commit()
    return project


def _serialize_favorite(
    fav,
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


def _serialize_favorite_minimal(fav) -> dict[str, object]:
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


def _serialize_project(project) -> dict[str, object]:
    return {
        "id": project.id,
        "name": project.name,
        "color": project.color,
        "is_default": bool(project.is_default),
        "sort_order": project.sort_order,
        "created_at": project.created_at.isoformat() if project.created_at else None,
    }


def _serialize_tag(tag) -> dict[str, object]:
    return {
        "id": tag.id,
        "name": tag.name,
        "color": tag.color,
        "created_at": tag.created_at.isoformat() if tag.created_at else None,
    }


def _next_project_sort_order(deps: FavoritesDeps, *, user_id: str) -> int:
    rows = deps.FavoriteProject.query.filter_by(user_id=user_id).all()
    if not rows:
        return 0
    return max(int(row.sort_order or 0) for row in rows) + 1


def _load_tags_for_favorites(
    deps: FavoritesDeps, *, favorite_ids: list[str]
) -> dict[str, list[dict[str, object]]]:
    if not favorite_ids:
        return {}
    db_obj = cast(Any, deps.db)
    session = db_obj.session
    rows = (
        session.query(
            deps.FavoriteTagAssignment.favorite_id,
            deps.FavoriteTag.id,
            deps.FavoriteTag.name,
            deps.FavoriteTag.color,
        )
        .join(
            deps.FavoriteTag,
            deps.FavoriteTag.id == deps.FavoriteTagAssignment.tag_id,
        )
        .filter(
            deps.FavoriteTagAssignment.favorite_id.in_(favorite_ids)
        )
        .order_by(deps.FavoriteTag.name.asc())
        .all()
    )
    out: dict[str, list[dict[str, object]]] = {}
    for favorite_id, tag_id, name, color in rows:
        out.setdefault(favorite_id, []).append(
            {"id": tag_id, "name": name, "color": color}
        )
    return out


def _load_projects_for_favorites(
    deps: FavoritesDeps, *, favorite_ids: list[str]
) -> dict[str, list[str]]:
    if not favorite_ids:
        return {}
    rows = (
        _db_session(deps)
        .query(
            deps.FavoriteProjectAssignment.favorite_id,
            deps.FavoriteProjectAssignment.project_id,
        )
        .filter(deps.FavoriteProjectAssignment.favorite_id.in_(favorite_ids))
        .all()
    )
    out: dict[str, list[str]] = {}
    for favorite_id, project_id in rows:
        out.setdefault(favorite_id, []).append(project_id)
    return out


def _ensure_favorite_project_assignment(
    deps: FavoritesDeps, *, favorite_id: str, project_id: str
) -> None:
    existing = deps.FavoriteProjectAssignment.query.filter_by(
        favorite_id=favorite_id, project_id=project_id
    ).first()
    if existing is not None:
        return
    assignment = deps.FavoriteProjectAssignment()  # type: ignore[call-arg]
    assignment.favorite_id = favorite_id
    assignment.project_id = project_id
    _db_session(deps).add(assignment)


def _backfill_favorite_project_assignments(deps: FavoritesDeps, *, user_id: str) -> None:
    favorites = deps.Favorite.query.filter_by(user_id=user_id).all()
    for fav in favorites:
        _ensure_favorite_project_assignment(
            deps, favorite_id=fav.id, project_id=fav.project_id
        )


def _resolve_agreement_uuids(
    deps: FavoritesDeps, *, favorites: list[object]
) -> dict[tuple[str, str], str]:
    section_uuids: list[str] = []
    clause_uuids: list[str] = []
    agreement_uuids: list[str] = []
    for fav in favorites:
        item_type = cast(str, getattr(fav, "item_type"))
        item_uuid = cast(str, getattr(fav, "item_uuid"))
        if item_type == "section":
            section_uuids.append(item_uuid)
        elif item_type == "tax_clause":
            clause_uuids.append(item_uuid)
        elif item_type == "agreement":
            agreement_uuids.append(item_uuid)

    resolved: dict[tuple[str, str], str] = {}
    db_obj = cast(Any, deps.db)
    session = db_obj.session

    if section_uuids:
        rows = (
            session.query(deps.Sections.section_uuid, deps.Sections.agreement_uuid)
            .filter(deps.Sections.section_uuid.in_(section_uuids))
            .all()
        )
        for section_uuid, agreement_uuid in rows:
            resolved[("section", section_uuid)] = agreement_uuid
    if clause_uuids:
        rows = (
            session.query(deps.Clauses.clause_uuid, deps.Clauses.agreement_uuid)
            .filter(deps.Clauses.clause_uuid.in_(clause_uuids))
            .all()
        )
        for clause_uuid, agreement_uuid in rows:
            resolved[("tax_clause", clause_uuid)] = agreement_uuid
    for agreement_uuid in agreement_uuids:
        resolved[("agreement", agreement_uuid)] = agreement_uuid
    return resolved


def _no_store(payload: object):
    resp = make_response(jsonify(payload))
    resp.headers["Cache-Control"] = "no-store"
    return resp
