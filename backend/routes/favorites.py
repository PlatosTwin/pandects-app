"""Favorites + projects routes (Phase A).

Surfaces /v1/me/favorites and /v1/me/favorite-projects. All endpoints
require an authenticated, verified user via cookie or bearer session.
API-key access is not supported here — favorites are user-scoped UI
state, not part of the public API.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import cast

from flask import Blueprint, Flask, abort, jsonify, make_response, request
from sqlalchemy.exc import SQLAlchemyError

from backend.routes.deps import AccessContextProtocol, UserLikeProtocol
from backend.schemas.favorites import (
    ITEM_TYPES,
    TAG_COLORS,
    FavoriteCreateSchema,
    FavoriteExistsQuerySchema,
    FavoriteTagsSetSchema,
    FavoriteUpdateSchema,
    TagCreateSchema,
    TagUpdateSchema,
)


@dataclass(frozen=True)
class FavoritesDeps:
    Favorite: type
    FavoriteProject: type
    FavoriteTag: type
    FavoriteTagAssignment: type
    Sections: type
    Clauses: type
    Agreements: type
    db: object
    _require_auth_db: Callable[[], None]
    _require_verified_user: Callable[[], tuple[UserLikeProtocol, AccessContextProtocol]]
    _auth_is_mocked: Callable[[], bool]
    _load_json: Callable[[object], dict[str, object]]
    _utc_now: Callable[[], datetime]


_DEFAULT_PROJECT_NAME = "Scratchpad"


def _ensure_default_project(deps: FavoritesDeps, *, user_id: str):
    project = deps.FavoriteProject.query.filter_by(  # type: ignore[attr-defined]
        user_id=user_id, is_default=True
    ).first()
    if project is not None:
        return project
    project = deps.FavoriteProject()  # type: ignore[call-arg]
    project.id = str(uuid.uuid4())
    project.user_id = user_id
    project.name = _DEFAULT_PROJECT_NAME
    project.is_default = True
    project.sort_order = 0
    cast(object, deps.db).session.add(project)  # type: ignore[attr-defined]
    cast(object, deps.db).session.commit()  # type: ignore[attr-defined]
    return project


def _serialize_favorite(
    fav,
    *,
    agreement_uuid: str | None,
    tags: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    return {
        "id": fav.id,
        "project_id": fav.project_id,
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


def _load_tags_for_favorites(
    deps: FavoritesDeps, *, favorite_ids: list[str]
) -> dict[str, list[dict[str, object]]]:
    if not favorite_ids:
        return {}
    db_obj = cast(object, deps.db)
    session = db_obj.session  # type: ignore[attr-defined]
    rows = (
        session.query(
            deps.FavoriteTagAssignment.favorite_id,  # type: ignore[attr-defined]
            deps.FavoriteTag.id,  # type: ignore[attr-defined]
            deps.FavoriteTag.name,  # type: ignore[attr-defined]
            deps.FavoriteTag.color,  # type: ignore[attr-defined]
        )
        .join(
            deps.FavoriteTag,  # type: ignore[attr-defined]
            deps.FavoriteTag.id == deps.FavoriteTagAssignment.tag_id,  # type: ignore[attr-defined]
        )
        .filter(
            deps.FavoriteTagAssignment.favorite_id.in_(favorite_ids)  # type: ignore[attr-defined]
        )
        .order_by(deps.FavoriteTag.name.asc())  # type: ignore[attr-defined]
        .all()
    )
    out: dict[str, list[dict[str, object]]] = {}
    for favorite_id, tag_id, name, color in rows:
        out.setdefault(favorite_id, []).append(
            {"id": tag_id, "name": name, "color": color}
        )
    return out


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
    db_obj = cast(object, deps.db)
    session = db_obj.session  # type: ignore[attr-defined]

    if section_uuids:
        rows = (
            session.query(deps.Sections.section_uuid, deps.Sections.agreement_uuid)  # type: ignore[attr-defined]
            .filter(deps.Sections.section_uuid.in_(section_uuids))  # type: ignore[attr-defined]
            .all()
        )
        for section_uuid, agreement_uuid in rows:
            resolved[("section", section_uuid)] = agreement_uuid
    if clause_uuids:
        rows = (
            session.query(deps.Clauses.clause_uuid, deps.Clauses.agreement_uuid)  # type: ignore[attr-defined]
            .filter(deps.Clauses.clause_uuid.in_(clause_uuids))  # type: ignore[attr-defined]
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


def register_favorites_routes(target_app: Flask, *, deps: FavoritesDeps) -> Blueprint:
    favorites_blp = Blueprint("favorites", __name__, url_prefix="/v1/me")

    def _guard_not_mocked() -> None:
        if deps._auth_is_mocked():
            abort(503, description="Favorites are unavailable in mock auth mode.")

    @favorites_blp.route("/favorite-projects", methods=["GET"])
    def list_projects():
        _guard_not_mocked()
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        try:
            _ = _ensure_default_project(deps, user_id=user.id)
            projects = (
                deps.FavoriteProject.query.filter_by(user_id=user.id)  # type: ignore[attr-defined]
                .order_by(
                    deps.FavoriteProject.is_default.desc(),  # type: ignore[attr-defined]
                    deps.FavoriteProject.sort_order.asc(),  # type: ignore[attr-defined]
                    deps.FavoriteProject.created_at.asc(),  # type: ignore[attr-defined]
                )
                .all()
            )
        except SQLAlchemyError:
            cast(object, deps.db).session.rollback()  # type: ignore[attr-defined]
            abort(503, description="Favorites backend is unavailable right now.")
        return _no_store({"projects": [_serialize_project(p) for p in projects]})

    @favorites_blp.route("/favorites", methods=["GET"])
    def list_favorites():
        _guard_not_mocked()
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        item_type = request.args.get("item_type")
        if item_type is not None and item_type not in ITEM_TYPES:
            abort(400, description="Invalid item_type filter.")
        raw_tag_ids = request.args.get("tag_ids", "")
        tag_id_filter = [t for t in (s.strip() for s in raw_tag_ids.split(",")) if t]
        try:
            q = deps.Favorite.query.filter_by(user_id=user.id)  # type: ignore[attr-defined]
            if item_type:
                q = q.filter_by(item_type=item_type)
            if tag_id_filter:
                # Match favorites that have ALL of the requested tags.
                for tag_id in tag_id_filter:
                    sub = (
                        cast(object, deps.db).session.query(  # type: ignore[attr-defined]
                            deps.FavoriteTagAssignment.favorite_id  # type: ignore[attr-defined]
                        ).filter(deps.FavoriteTagAssignment.tag_id == tag_id)  # type: ignore[attr-defined]
                    )
                    q = q.filter(deps.Favorite.id.in_(sub))  # type: ignore[attr-defined]
            favorites = q.order_by(deps.Favorite.created_at.desc()).all()  # type: ignore[attr-defined]
        except SQLAlchemyError:
            cast(object, deps.db).session.rollback()  # type: ignore[attr-defined]
            abort(503, description="Favorites backend is unavailable right now.")
        try:
            agreement_map = _resolve_agreement_uuids(deps, favorites=favorites)
        except SQLAlchemyError:
            agreement_map = {}
        try:
            tag_map = _load_tags_for_favorites(
                deps, favorite_ids=[fav.id for fav in favorites]
            )
        except SQLAlchemyError:
            tag_map = {}
        return _no_store(
            {
                "favorites": [
                    _serialize_favorite(
                        fav,
                        agreement_uuid=agreement_map.get((fav.item_type, fav.item_uuid)),
                        tags=tag_map.get(fav.id, []),
                    )
                    for fav in favorites
                ]
            }
        )

    @favorites_blp.route("/favorites/exists", methods=["GET"])
    def exists_favorites():
        _guard_not_mocked()
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        schema = FavoriteExistsQuerySchema()
        errors = schema.validate(request.args)
        if errors:
            abort(400, description="Invalid query parameters.")
        item_type = request.args["item_type"]
        raw_uuids = request.args.get("item_uuids", "")
        uuids = [u for u in (s.strip() for s in raw_uuids.split(",")) if u]
        if not uuids:
            return _no_store({"favorites": {}})
        if len(uuids) > 200:
            abort(400, description="Too many item_uuids (max 200).")
        try:
            rows = (
                deps.Favorite.query.filter(  # type: ignore[attr-defined]
                    deps.Favorite.user_id == user.id,  # type: ignore[attr-defined]
                    deps.Favorite.item_type == item_type,  # type: ignore[attr-defined]
                    deps.Favorite.item_uuid.in_(uuids),  # type: ignore[attr-defined]
                ).all()
            )
        except SQLAlchemyError:
            cast(object, deps.db).session.rollback()  # type: ignore[attr-defined]
            abort(503, description="Favorites backend is unavailable right now.")
        return _no_store(
            {"favorites": {row.item_uuid: row.id for row in rows}}
        )

    @favorites_blp.route("/favorites", methods=["POST"])
    def create_favorite():
        _guard_not_mocked()
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        data = deps._load_json(FavoriteCreateSchema())
        minimal_response = request.args.get("view") == "minimal"
        item_type = cast(str, data["item_type"])
        item_uuid = cast(str, data["item_uuid"]).strip()
        note_val = data.get("note")
        note: str | None = note_val.strip() if isinstance(note_val, str) and note_val.strip() else None
        context_val = data.get("context")
        context = context_val if isinstance(context_val, dict) else None
        project_id = data.get("project_id")
        if project_id is not None and not isinstance(project_id, str):
            abort(400, description="project_id must be a string.")

        try:
            project = None
            if project_id:
                project = deps.FavoriteProject.query.filter_by(  # type: ignore[attr-defined]
                    id=project_id, user_id=user.id
                ).first()
                if project is None:
                    abort(404, description="Project not found.")
            else:
                project = _ensure_default_project(deps, user_id=user.id)

            existing = deps.Favorite.query.filter_by(  # type: ignore[attr-defined]
                user_id=user.id, item_type=item_type, item_uuid=item_uuid
            ).first()
            if existing is not None:
                if note is not None:
                    existing.note = note
                if context is not None:
                    existing.context = context
                existing.project_id = project.id
                existing.updated_at = deps._utc_now()
                cast(object, deps.db).session.commit()  # type: ignore[attr-defined]
                fav = existing
                created = False
            else:
                fav = deps.Favorite()  # type: ignore[call-arg]
                fav.id = str(uuid.uuid4())
                fav.user_id = user.id
                fav.project_id = project.id
                fav.item_type = item_type
                fav.item_uuid = item_uuid
                fav.note = note
                fav.context = context
                cast(object, deps.db).session.add(fav)  # type: ignore[attr-defined]
                cast(object, deps.db).session.commit()  # type: ignore[attr-defined]
                created = True
        except SQLAlchemyError:
            cast(object, deps.db).session.rollback()  # type: ignore[attr-defined]
            abort(503, description="Favorites backend is unavailable right now.")
        if minimal_response:
            resp = _no_store(
                {
                    "favorite": _serialize_favorite_minimal(fav),
                    "created": created,
                }
            )
            resp.status_code = 201 if created else 200
            return resp
        try:
            agreement_map = _resolve_agreement_uuids(deps, favorites=[fav])
        except SQLAlchemyError:
            agreement_map = {}
        try:
            tag_map = _load_tags_for_favorites(deps, favorite_ids=[fav.id])
        except SQLAlchemyError:
            tag_map = {}
        resp = _no_store(
            {
                "favorite": _serialize_favorite(
                    fav,
                    agreement_uuid=agreement_map.get((fav.item_type, fav.item_uuid)),
                    tags=tag_map.get(fav.id, []),
                ),
                "created": created,
            }
        )
        resp.status_code = 201 if created else 200
        return resp

    @favorites_blp.route("/favorites/<string:favorite_id>", methods=["PATCH"])
    def update_favorite(favorite_id: str):
        _guard_not_mocked()
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        data = deps._load_json(FavoriteUpdateSchema())
        try:
            fav = deps.Favorite.query.filter_by(  # type: ignore[attr-defined]
                id=favorite_id, user_id=user.id
            ).first()
            if fav is None:
                abort(404, description="Favorite not found.")
            if "note" in data:
                note_val = data.get("note")
                fav.note = (
                    note_val.strip()
                    if isinstance(note_val, str) and note_val.strip()
                    else None
                )
            if "project_id" in data:
                project_id = data.get("project_id")
                if project_id is None:
                    fav.project_id = _ensure_default_project(deps, user_id=user.id).id
                else:
                    if not isinstance(project_id, str):
                        abort(400, description="project_id must be a string.")
                    project = deps.FavoriteProject.query.filter_by(  # type: ignore[attr-defined]
                        id=project_id, user_id=user.id
                    ).first()
                    if project is None:
                        abort(404, description="Project not found.")
                    fav.project_id = project.id
            fav.updated_at = deps._utc_now()
            cast(object, deps.db).session.commit()  # type: ignore[attr-defined]
        except SQLAlchemyError:
            cast(object, deps.db).session.rollback()  # type: ignore[attr-defined]
            abort(503, description="Favorites backend is unavailable right now.")
        try:
            agreement_map = _resolve_agreement_uuids(deps, favorites=[fav])
        except SQLAlchemyError:
            agreement_map = {}
        try:
            tag_map = _load_tags_for_favorites(deps, favorite_ids=[fav.id])
        except SQLAlchemyError:
            tag_map = {}
        return _no_store(
            {
                "favorite": _serialize_favorite(
                    fav,
                    agreement_uuid=agreement_map.get((fav.item_type, fav.item_uuid)),
                    tags=tag_map.get(fav.id, []),
                )
            }
        )

    @favorites_blp.route("/favorites/<string:favorite_id>", methods=["DELETE"])
    def delete_favorite(favorite_id: str):
        _guard_not_mocked()
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        try:
            fav = deps.Favorite.query.filter_by(  # type: ignore[attr-defined]
                id=favorite_id, user_id=user.id
            ).first()
            if fav is None:
                abort(404, description="Favorite not found.")
            session = cast(object, deps.db).session  # type: ignore[attr-defined]
            session.query(deps.FavoriteTagAssignment).filter_by(  # type: ignore[attr-defined]
                favorite_id=fav.id
            ).delete(synchronize_session=False)
            session.delete(fav)
            session.commit()
        except SQLAlchemyError:
            cast(object, deps.db).session.rollback()  # type: ignore[attr-defined]
            abort(503, description="Favorites backend is unavailable right now.")
        return _no_store({"deleted": True})

    @favorites_blp.route("/tags", methods=["GET"])
    def list_tags():
        _guard_not_mocked()
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        try:
            tags = (
                deps.FavoriteTag.query.filter_by(user_id=user.id)  # type: ignore[attr-defined]
                .order_by(deps.FavoriteTag.name.asc())  # type: ignore[attr-defined]
                .all()
            )
        except SQLAlchemyError:
            cast(object, deps.db).session.rollback()  # type: ignore[attr-defined]
            abort(503, description="Favorites backend is unavailable right now.")
        return _no_store(
            {
                "tags": [_serialize_tag(t) for t in tags],
                "available_colors": list(TAG_COLORS),
            }
        )

    @favorites_blp.route("/tags", methods=["POST"])
    def create_tag():
        _guard_not_mocked()
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        data = deps._load_json(TagCreateSchema())
        name = cast(str, data["name"]).strip()
        if not name:
            abort(400, description="Tag name is required.")
        color = cast(str, data.get("color") or "slate")
        try:
            existing = deps.FavoriteTag.query.filter_by(  # type: ignore[attr-defined]
                user_id=user.id, name=name
            ).first()
            if existing is not None:
                return _no_store({"tag": _serialize_tag(existing), "created": False})
            tag = deps.FavoriteTag()  # type: ignore[call-arg]
            tag.id = str(uuid.uuid4())
            tag.user_id = user.id
            tag.name = name
            tag.color = color
            cast(object, deps.db).session.add(tag)  # type: ignore[attr-defined]
            cast(object, deps.db).session.commit()  # type: ignore[attr-defined]
        except SQLAlchemyError:
            cast(object, deps.db).session.rollback()  # type: ignore[attr-defined]
            abort(503, description="Favorites backend is unavailable right now.")
        resp = _no_store({"tag": _serialize_tag(tag), "created": True})
        resp.status_code = 201
        return resp

    @favorites_blp.route("/tags/<string:tag_id>", methods=["PATCH"])
    def update_tag(tag_id: str):
        _guard_not_mocked()
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        data = deps._load_json(TagUpdateSchema())
        try:
            tag = deps.FavoriteTag.query.filter_by(  # type: ignore[attr-defined]
                id=tag_id, user_id=user.id
            ).first()
            if tag is None:
                abort(404, description="Tag not found.")
            if "name" in data:
                new_name = cast(str, data["name"]).strip()
                if not new_name:
                    abort(400, description="Tag name is required.")
                if new_name != tag.name:
                    clash = deps.FavoriteTag.query.filter_by(  # type: ignore[attr-defined]
                        user_id=user.id, name=new_name
                    ).first()
                    if clash is not None and clash.id != tag.id:
                        abort(409, description="A tag with that name already exists.")
                    tag.name = new_name
            if "color" in data:
                tag.color = cast(str, data["color"])
            cast(object, deps.db).session.commit()  # type: ignore[attr-defined]
        except SQLAlchemyError:
            cast(object, deps.db).session.rollback()  # type: ignore[attr-defined]
            abort(503, description="Favorites backend is unavailable right now.")
        return _no_store({"tag": _serialize_tag(tag)})

    @favorites_blp.route("/tags/<string:tag_id>", methods=["DELETE"])
    def delete_tag(tag_id: str):
        _guard_not_mocked()
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        try:
            tag = deps.FavoriteTag.query.filter_by(  # type: ignore[attr-defined]
                id=tag_id, user_id=user.id
            ).first()
            if tag is None:
                abort(404, description="Tag not found.")
            session = cast(object, deps.db).session  # type: ignore[attr-defined]
            session.query(deps.FavoriteTagAssignment).filter_by(  # type: ignore[attr-defined]
                tag_id=tag.id
            ).delete(synchronize_session=False)
            session.delete(tag)
            session.commit()
        except SQLAlchemyError:
            cast(object, deps.db).session.rollback()  # type: ignore[attr-defined]
            abort(503, description="Favorites backend is unavailable right now.")
        return _no_store({"deleted": True})

    @favorites_blp.route("/favorites/<string:favorite_id>/tags", methods=["PUT"])
    def set_favorite_tags(favorite_id: str):
        _guard_not_mocked()
        deps._require_auth_db()
        user, _ctx = deps._require_verified_user()
        data = deps._load_json(FavoriteTagsSetSchema())
        raw_ids = data.get("tag_ids")
        if not isinstance(raw_ids, list):
            abort(400, description="tag_ids must be a list.")
        tag_ids = [t for t in (str(x).strip() for x in raw_ids) if t]
        try:
            fav = deps.Favorite.query.filter_by(  # type: ignore[attr-defined]
                id=favorite_id, user_id=user.id
            ).first()
            if fav is None:
                abort(404, description="Favorite not found.")
            valid_ids: list[str] = []
            if tag_ids:
                rows = (
                    deps.FavoriteTag.query.filter(  # type: ignore[attr-defined]
                        deps.FavoriteTag.user_id == user.id,  # type: ignore[attr-defined]
                        deps.FavoriteTag.id.in_(tag_ids),  # type: ignore[attr-defined]
                    ).all()
                )
                valid_ids = [row.id for row in rows]
                if len(valid_ids) != len(set(tag_ids)):
                    abort(404, description="One or more tags not found.")
            session = cast(object, deps.db).session  # type: ignore[attr-defined]
            session.query(deps.FavoriteTagAssignment).filter_by(  # type: ignore[attr-defined]
                favorite_id=fav.id
            ).delete(synchronize_session=False)
            for tag_id in valid_ids:
                assignment = deps.FavoriteTagAssignment()  # type: ignore[call-arg]
                assignment.favorite_id = fav.id
                assignment.tag_id = tag_id
                session.add(assignment)
            fav.updated_at = deps._utc_now()
            session.commit()
        except SQLAlchemyError:
            cast(object, deps.db).session.rollback()  # type: ignore[attr-defined]
            abort(503, description="Favorites backend is unavailable right now.")
        try:
            tag_map = _load_tags_for_favorites(deps, favorite_ids=[fav.id])
        except SQLAlchemyError:
            tag_map = {}
        return _no_store({"tags": tag_map.get(fav.id, [])})

    target_app.register_blueprint(favorites_blp)
    return favorites_blp


__all__ = ["FavoritesDeps", "register_favorites_routes"]
