"""Favorites + projects routes (Phase A).

Surfaces /v1/me/favorites and /v1/me/favorite-projects. All endpoints
require an authenticated, verified user via cookie or bearer session.
API-key access is not supported here — favorites are user-scoped UI
state, not part of the public API.
"""

from __future__ import annotations

import uuid
from typing import cast

from flask import Blueprint, Flask, abort, request
from sqlalchemy.exc import SQLAlchemyError

from backend.routes.deps import UserLikeProtocol
from backend.routes.favorites.deps import FavoritesDeps
from backend.routes.favorites.helpers import (
    _backfill_favorite_project_assignments,
    _db_session,
    _ensure_default_project,
    _ensure_favorite_project_assignment,
    _load_projects_for_favorites,
    _load_tags_for_favorites,
    _next_project_sort_order,
    _no_store,
    _resolve_agreement_uuids,
    _serialize_favorite,
    _serialize_favorite_minimal,
    _serialize_project,
    _serialize_tag,
)
from backend.schemas.favorites import (
    ITEM_TYPES,
    TAG_COLORS,
    FavoriteCreateSchema,
    FavoriteExistsQuerySchema,
    FavoriteProjectsSetSchema,
    FavoriteTagsSetSchema,
    FavoritesBulkTagsSchema,
    FavoritesBulkCopySchema,
    FavoritesBulkMoveSchema,
    FavoriteUpdateSchema,
    ProjectCreateSchema,
    ProjectDeleteQuerySchema,
    ProjectUpdateSchema,
    TagCreateSchema,
    TagUpdateSchema,
)

__all__ = ["FavoritesDeps", "register_favorites_routes"]


def register_favorites_routes(target_app: Flask, *, deps: FavoritesDeps) -> Blueprint:
    favorites_blp = Blueprint("favorites", __name__, url_prefix="/v1/me")

    def _guard_not_mocked() -> None:
        if deps._auth_is_mocked():
            abort(503, description="Favorites are unavailable in mock auth mode.")

    def _require_favorites_user() -> UserLikeProtocol:
        user, ctx = deps._require_verified_user()
        if ctx.tier != "user":
            abort(403, description="Favorites require a signed-in user session.")
        return user

    @favorites_blp.route("/favorite-projects", methods=["GET"])
    def list_projects():
        _guard_not_mocked()
        deps._require_auth_db()
        user = _require_favorites_user()
        try:
            _ = _ensure_default_project(deps, user_id=user.id)
            _backfill_favorite_project_assignments(deps, user_id=user.id)
            _db_session(deps).commit()
            projects = (
                deps.FavoriteProject.query.filter_by(user_id=user.id)
                .order_by(
                    deps.FavoriteProject.is_default.desc(),
                    deps.FavoriteProject.sort_order.asc(),
                    deps.FavoriteProject.created_at.asc(),
                )
                .all()
            )
        except SQLAlchemyError:
            _db_session(deps).rollback()
            abort(503, description="Favorites backend is unavailable right now.")
        return _no_store({"projects": [_serialize_project(p) for p in projects]})

    @favorites_blp.route("/favorite-projects", methods=["POST"])
    def create_project():
        _guard_not_mocked()
        deps._require_auth_db()
        user = _require_favorites_user()
        data = deps._load_json(ProjectCreateSchema())
        name = cast(str, data["name"]).strip()
        if not name:
            abort(400, description="Project name is required.")
        color = cast(str, data.get("color") or "slate")
        try:
            _ = _ensure_default_project(deps, user_id=user.id)
            existing = deps.FavoriteProject.query.filter_by(
                user_id=user.id, name=name
            ).first()
            if existing is not None:
                abort(409, description="A project with that name already exists.")
            project = deps.FavoriteProject()  # type: ignore[call-arg]
            project.id = str(uuid.uuid4())
            project.user_id = user.id
            project.name = name
            project.color = color
            project.is_default = False
            project.sort_order = _next_project_sort_order(deps, user_id=user.id)
            _db_session(deps).add(project)
            _db_session(deps).commit()
        except SQLAlchemyError:
            _db_session(deps).rollback()
            abort(503, description="Favorites backend is unavailable right now.")
        resp = _no_store({"project": _serialize_project(project), "created": True})
        resp.status_code = 201
        return resp

    @favorites_blp.route("/favorite-projects/<string:project_id>", methods=["PATCH"])
    def update_project(project_id: str):
        _guard_not_mocked()
        deps._require_auth_db()
        user = _require_favorites_user()
        data = deps._load_json(ProjectUpdateSchema())
        try:
            project = deps.FavoriteProject.query.filter_by(
                id=project_id, user_id=user.id
            ).first()
            if project is None:
                abort(404, description="Project not found.")
            if "name" in data:
                new_name = cast(str, data["name"]).strip()
                if not new_name:
                    abort(400, description="Project name is required.")
                if new_name != project.name:
                    clash = deps.FavoriteProject.query.filter_by(
                        user_id=user.id, name=new_name
                    ).first()
                    if clash is not None and clash.id != project.id:
                        abort(409, description="A project with that name already exists.")
                    project.name = new_name
            if "color" in data:
                project.color = cast(str, data["color"])
            if "sort_order" in data:
                project.sort_order = cast(int, data["sort_order"])
            _db_session(deps).commit()
        except SQLAlchemyError:
            _db_session(deps).rollback()
            abort(503, description="Favorites backend is unavailable right now.")
        return _no_store({"project": _serialize_project(project)})

    @favorites_blp.route("/favorite-projects/<string:project_id>", methods=["DELETE"])
    def delete_project(project_id: str):
        _guard_not_mocked()
        deps._require_auth_db()
        user = _require_favorites_user()
        schema = ProjectDeleteQuerySchema()
        errors = schema.validate(request.args)
        if errors:
            abort(400, description="Invalid query parameters.")
        reassign_project_id = request.args.get("reassign_project_id")
        try:
            project = deps.FavoriteProject.query.filter_by(
                id=project_id, user_id=user.id
            ).first()
            if project is None:
                abort(404, description="Project not found.")
            target = None
            if reassign_project_id:
                target = deps.FavoriteProject.query.filter_by(
                    id=reassign_project_id, user_id=user.id
                ).first()
                if target is None or target.id == project.id:
                    abort(400, description="Choose a different project to reassign favorites.")
            else:
                default_project = _ensure_default_project(deps, user_id=user.id)
                if default_project.id != project.id:
                    target = default_project
                else:
                    target = (
                        deps.FavoriteProject.query.filter(
                            deps.FavoriteProject.user_id == user.id,
                            deps.FavoriteProject.id != project.id,
                        )
                        .order_by(deps.FavoriteProject.sort_order.asc())
                        .first()
                    )
            if target is None:
                abort(
                    400,
                    description="Create another project or provide reassign_project_id before deleting this project.",
                )
            session = _db_session(deps)
            assigned_rows = (
                session.query(deps.FavoriteProjectAssignment.favorite_id)
                .filter_by(project_id=project.id)
                .all()
            )
            affected_favorite_ids = [row[0] for row in assigned_rows]
            moved = (
                deps.Favorite.query.filter_by(
                    user_id=user.id, project_id=project.id
                ).update({"project_id": target.id}, synchronize_session=False)
            )
            session.query(deps.FavoriteProjectAssignment).filter_by(
                project_id=project.id
            ).delete(synchronize_session=False)
            for favorite_id in affected_favorite_ids:
                _ensure_favorite_project_assignment(
                    deps, favorite_id=favorite_id, project_id=target.id
                )
            session.delete(project)
            session.commit()
        except SQLAlchemyError:
            _db_session(deps).rollback()
            abort(503, description="Favorites backend is unavailable right now.")
        return _no_store(
            {"deleted": True, "reassigned_to_project_id": target.id, "moved": moved}
        )

    @favorites_blp.route("/favorites", methods=["GET"])
    def list_favorites():
        _guard_not_mocked()
        deps._require_auth_db()
        user = _require_favorites_user()
        item_type = request.args.get("item_type")
        if item_type is not None and item_type not in ITEM_TYPES:
            abort(400, description="Invalid item_type filter.")
        project_id = request.args.get("project_id")
        raw_tag_ids = request.args.get("tag_ids", "")
        tag_id_filter = [t for t in (s.strip() for s in raw_tag_ids.split(",")) if t]
        try:
            q = deps.Favorite.query.filter_by(user_id=user.id)
            if item_type:
                q = q.filter_by(item_type=item_type)
            if project_id:
                sub = (
                    _db_session(deps)
                    .query(deps.FavoriteProjectAssignment.favorite_id)
                    .filter(deps.FavoriteProjectAssignment.project_id == project_id)
                )
                q = q.filter(deps.Favorite.id.in_(sub))
            if tag_id_filter:
                # Match favorites that have ALL of the requested tags.
                for tag_id in tag_id_filter:
                    sub = (
                        _db_session(deps).query(
                            deps.FavoriteTagAssignment.favorite_id
                        ).filter(deps.FavoriteTagAssignment.tag_id == tag_id)
                    )
                    q = q.filter(deps.Favorite.id.in_(sub))
            favorites = q.order_by(deps.Favorite.created_at.desc()).all()
        except SQLAlchemyError:
            _db_session(deps).rollback()
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
        try:
            project_map = _load_projects_for_favorites(
                deps, favorite_ids=[fav.id for fav in favorites]
            )
        except SQLAlchemyError:
            project_map = {}
        return _no_store(
            {
                "favorites": [
                    _serialize_favorite(
                        fav,
                        agreement_uuid=agreement_map.get((fav.item_type, fav.item_uuid)),
                        tags=tag_map.get(fav.id, []),
                        project_ids=project_map.get(fav.id, [fav.project_id]),
                    )
                    for fav in favorites
                ]
            }
        )

    @favorites_blp.route("/favorites/exists", methods=["GET"])
    def exists_favorites():
        _guard_not_mocked()
        deps._require_auth_db()
        user = _require_favorites_user()
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
                deps.Favorite.query.filter(
                    deps.Favorite.user_id == user.id,
                    deps.Favorite.item_type == item_type,
                    deps.Favorite.item_uuid.in_(uuids),
                ).all()
            )
        except SQLAlchemyError:
            _db_session(deps).rollback()
            abort(503, description="Favorites backend is unavailable right now.")
        return _no_store(
            {"favorites": {row.item_uuid: row.id for row in rows}}
        )

    @favorites_blp.route("/favorites", methods=["POST"])
    def create_favorite():
        _guard_not_mocked()
        deps._require_auth_db()
        user = _require_favorites_user()
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
                project = deps.FavoriteProject.query.filter_by(
                    id=project_id, user_id=user.id
                ).first()
                if project is None:
                    abort(404, description="Project not found.")
            else:
                project = _ensure_default_project(deps, user_id=user.id)

            existing = deps.Favorite.query.filter_by(
                user_id=user.id, item_type=item_type, item_uuid=item_uuid
            ).first()
            if existing is not None:
                if note is not None:
                    existing.note = note
                if context is not None:
                    existing.context = context
                existing.project_id = project.id
                existing.updated_at = deps._utc_now()
                _ensure_favorite_project_assignment(
                    deps, favorite_id=existing.id, project_id=project.id
                )
                _db_session(deps).commit()
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
                _db_session(deps).add(fav)
                _ensure_favorite_project_assignment(
                    deps, favorite_id=fav.id, project_id=project.id
                )
                _db_session(deps).commit()
                created = True
        except SQLAlchemyError:
            _db_session(deps).rollback()
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
        try:
            project_map = _load_projects_for_favorites(deps, favorite_ids=[fav.id])
        except SQLAlchemyError:
            project_map = {}
        resp = _no_store(
            {
                "favorite": _serialize_favorite(
                    fav,
                    agreement_uuid=agreement_map.get((fav.item_type, fav.item_uuid)),
                    tags=tag_map.get(fav.id, []),
                    project_ids=project_map.get(fav.id, [fav.project_id]),
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
        user = _require_favorites_user()
        data = deps._load_json(FavoriteUpdateSchema())
        try:
            fav = deps.Favorite.query.filter_by(
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
                    project = deps.FavoriteProject.query.filter_by(
                        id=project_id, user_id=user.id
                    ).first()
                    if project is None:
                        abort(404, description="Project not found.")
                    fav.project_id = project.id
                    _ensure_favorite_project_assignment(
                        deps, favorite_id=fav.id, project_id=project.id
                    )
            fav.updated_at = deps._utc_now()
            _db_session(deps).commit()
        except SQLAlchemyError:
            _db_session(deps).rollback()
            abort(503, description="Favorites backend is unavailable right now.")
        try:
            agreement_map = _resolve_agreement_uuids(deps, favorites=[fav])
        except SQLAlchemyError:
            agreement_map = {}
        try:
            tag_map = _load_tags_for_favorites(deps, favorite_ids=[fav.id])
        except SQLAlchemyError:
            tag_map = {}
        try:
            project_map = _load_projects_for_favorites(deps, favorite_ids=[fav.id])
        except SQLAlchemyError:
            project_map = {}
        return _no_store(
            {
                "favorite": _serialize_favorite(
                    fav,
                    agreement_uuid=agreement_map.get((fav.item_type, fav.item_uuid)),
                    tags=tag_map.get(fav.id, []),
                    project_ids=project_map.get(fav.id, [fav.project_id]),
                )
            }
        )

    @favorites_blp.route("/favorites/bulk-move", methods=["POST"])
    def bulk_move_favorites():
        _guard_not_mocked()
        deps._require_auth_db()
        user = _require_favorites_user()
        data = deps._load_json(FavoritesBulkMoveSchema())
        raw_favorite_ids = cast(list[object], data["favorite_ids"])
        favorite_ids = [
            value for value in (str(raw).strip() for raw in raw_favorite_ids) if value
        ]
        if not favorite_ids:
            abort(400, description="favorite_ids is required.")
        if len(favorite_ids) > 500:
            abort(400, description="Too many favorite_ids (max 500).")
        project_id = cast(str, data["project_id"]).strip()
        try:
            project = deps.FavoriteProject.query.filter_by(
                id=project_id, user_id=user.id
            ).first()
            if project is None:
                abort(404, description="Project not found.")
            rows = (
                deps.Favorite.query.filter(
                    deps.Favorite.user_id == user.id,
                    deps.Favorite.id.in_(favorite_ids),
                ).all()
            )
            found_ids = {row.id for row in rows}
            if found_ids != set(favorite_ids):
                abort(404, description="One or more favorites not found.")
            now = deps._utc_now()
            session = _db_session(deps)
            for fav in rows:
                fav.project_id = project.id
                fav.updated_at = now
                session.query(deps.FavoriteProjectAssignment).filter_by(
                    favorite_id=fav.id
                ).delete(synchronize_session=False)
                _ensure_favorite_project_assignment(
                    deps, favorite_id=fav.id, project_id=project.id
                )
            session.commit()
        except SQLAlchemyError:
            _db_session(deps).rollback()
            abort(503, description="Favorites backend is unavailable right now.")
        return _no_store(
            {
                "project_id": project.id,
                "favorite_ids": [row.id for row in rows],
                "moved": len(rows),
            }
        )

    @favorites_blp.route("/favorites/bulk-copy", methods=["POST"])
    def bulk_copy_favorites():
        _guard_not_mocked()
        deps._require_auth_db()
        user = _require_favorites_user()
        data = deps._load_json(FavoritesBulkCopySchema())
        raw_favorite_ids = cast(list[object], data["favorite_ids"])
        favorite_ids = [
            value for value in (str(raw).strip() for raw in raw_favorite_ids) if value
        ]
        raw_project_ids = cast(list[object], data["project_ids"])
        project_ids = [
            value for value in (str(raw).strip() for raw in raw_project_ids) if value
        ]
        if not favorite_ids or not project_ids:
            abort(400, description="favorite_ids and project_ids are required.")
        if len(favorite_ids) > 500:
            abort(400, description="Too many favorite_ids (max 500).")
        if len(project_ids) > 50:
            abort(400, description="Too many project_ids (max 50).")
        try:
            projects = (
                deps.FavoriteProject.query.filter(
                    deps.FavoriteProject.user_id == user.id,
                    deps.FavoriteProject.id.in_(project_ids),
                ).all()
            )
            if {project.id for project in projects} != set(project_ids):
                abort(404, description="One or more projects not found.")
            rows = (
                deps.Favorite.query.filter(
                    deps.Favorite.user_id == user.id,
                    deps.Favorite.id.in_(favorite_ids),
                ).all()
            )
            if {row.id for row in rows} != set(favorite_ids):
                abort(404, description="One or more favorites not found.")
            session = _db_session(deps)
            for fav in rows:
                for project_id in project_ids:
                    _ensure_favorite_project_assignment(
                        deps, favorite_id=fav.id, project_id=project_id
                    )
                fav.updated_at = deps._utc_now()
            session.commit()
        except SQLAlchemyError:
            _db_session(deps).rollback()
            abort(503, description="Favorites backend is unavailable right now.")
        return _no_store(
            {
                "project_ids": project_ids,
                "favorite_ids": [row.id for row in rows],
                "copied": len(rows),
            }
        )

    @favorites_blp.route("/favorites/bulk-tags", methods=["POST"])
    def bulk_update_favorite_tags():
        _guard_not_mocked()
        deps._require_auth_db()
        user = _require_favorites_user()
        data = deps._load_json(FavoritesBulkTagsSchema())
        raw_favorite_ids = cast(list[object], data["favorite_ids"])
        favorite_ids = list(
            dict.fromkeys(
                value
                for value in (str(raw).strip() for raw in raw_favorite_ids)
                if value
            )
        )
        raw_tag_ids = cast(list[object], data["tag_ids"])
        tag_ids = list(
            dict.fromkeys(
                value for value in (str(raw).strip() for raw in raw_tag_ids) if value
            )
        )
        action = cast(str, data["action"])
        if not favorite_ids or not tag_ids:
            abort(400, description="favorite_ids and tag_ids are required.")
        if len(favorite_ids) > 500:
            abort(400, description="Too many favorite_ids (max 500).")
        try:
            rows = (
                deps.Favorite.query.filter(
                    deps.Favorite.user_id == user.id,
                    deps.Favorite.id.in_(favorite_ids),
                ).all()
            )
            if {row.id for row in rows} != set(favorite_ids):
                abort(404, description="One or more favorites not found.")
            tags = (
                deps.FavoriteTag.query.filter(
                    deps.FavoriteTag.user_id == user.id,
                    deps.FavoriteTag.id.in_(tag_ids),
                ).all()
            )
            valid_tag_ids = [tag.id for tag in tags]
            if len(valid_tag_ids) != len(set(tag_ids)):
                abort(404, description="One or more tags not found.")

            session = _db_session(deps)
            now = deps._utc_now()
            if action == "add":
                existing = {
                    (favorite_id, tag_id)
                    for favorite_id, tag_id in session.query(
                        deps.FavoriteTagAssignment.favorite_id,
                        deps.FavoriteTagAssignment.tag_id,
                    )
                    .filter(deps.FavoriteTagAssignment.favorite_id.in_(favorite_ids))
                    .filter(deps.FavoriteTagAssignment.tag_id.in_(valid_tag_ids))
                    .all()
                }
                for favorite_id in favorite_ids:
                    for tag_id in valid_tag_ids:
                        if (favorite_id, tag_id) in existing:
                            continue
                        assignment = deps.FavoriteTagAssignment()  # type: ignore[call-arg]
                        assignment.favorite_id = favorite_id
                        assignment.tag_id = tag_id
                        session.add(assignment)
            else:
                session.query(deps.FavoriteTagAssignment).filter(
                    deps.FavoriteTagAssignment.favorite_id.in_(favorite_ids),
                    deps.FavoriteTagAssignment.tag_id.in_(valid_tag_ids),
                ).delete(synchronize_session=False)
            for fav in rows:
                fav.updated_at = now
            session.commit()
            tag_map = _load_tags_for_favorites(
                deps, favorite_ids=[row.id for row in rows]
            )
        except SQLAlchemyError:
            _db_session(deps).rollback()
            abort(503, description="Favorites backend is unavailable right now.")
        return _no_store(
            {
                "action": action,
                "tag_ids": valid_tag_ids,
                "favorite_ids": [row.id for row in rows],
                "tags_by_favorite": {
                    row.id: tag_map.get(row.id, []) for row in rows
                },
                "updated": len(rows),
            }
        )

    @favorites_blp.route("/favorites/<string:favorite_id>", methods=["DELETE"])
    def delete_favorite(favorite_id: str):
        _guard_not_mocked()
        deps._require_auth_db()
        user = _require_favorites_user()
        try:
            fav = deps.Favorite.query.filter_by(
                id=favorite_id, user_id=user.id
            ).first()
            if fav is None:
                abort(404, description="Favorite not found.")
            session = _db_session(deps)
            session.query(deps.FavoriteTagAssignment).filter_by(
                favorite_id=fav.id
            ).delete(synchronize_session=False)
            session.query(deps.FavoriteProjectAssignment).filter_by(
                favorite_id=fav.id
            ).delete(synchronize_session=False)
            session.delete(fav)
            session.commit()
        except SQLAlchemyError:
            _db_session(deps).rollback()
            abort(503, description="Favorites backend is unavailable right now.")
        return _no_store({"deleted": True})

    @favorites_blp.route("/tags", methods=["GET"])
    def list_tags():
        _guard_not_mocked()
        deps._require_auth_db()
        user = _require_favorites_user()
        try:
            tags = (
                deps.FavoriteTag.query.filter_by(user_id=user.id)
                .order_by(deps.FavoriteTag.name.asc())
                .all()
            )
        except SQLAlchemyError:
            _db_session(deps).rollback()
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
        user = _require_favorites_user()
        data = deps._load_json(TagCreateSchema())
        name = cast(str, data["name"]).strip()
        if not name:
            abort(400, description="Tag name is required.")
        color = cast(str, data.get("color") or "slate")
        try:
            existing = deps.FavoriteTag.query.filter_by(
                user_id=user.id, name=name
            ).first()
            if existing is not None:
                return _no_store({"tag": _serialize_tag(existing), "created": False})
            tag = deps.FavoriteTag()  # type: ignore[call-arg]
            tag.id = str(uuid.uuid4())
            tag.user_id = user.id
            tag.name = name
            tag.color = color
            _db_session(deps).add(tag)
            _db_session(deps).commit()
        except SQLAlchemyError:
            _db_session(deps).rollback()
            abort(503, description="Favorites backend is unavailable right now.")
        resp = _no_store({"tag": _serialize_tag(tag), "created": True})
        resp.status_code = 201
        return resp

    @favorites_blp.route("/tags/<string:tag_id>", methods=["PATCH"])
    def update_tag(tag_id: str):
        _guard_not_mocked()
        deps._require_auth_db()
        user = _require_favorites_user()
        data = deps._load_json(TagUpdateSchema())
        try:
            tag = deps.FavoriteTag.query.filter_by(
                id=tag_id, user_id=user.id
            ).first()
            if tag is None:
                abort(404, description="Tag not found.")
            if "name" in data:
                new_name = cast(str, data["name"]).strip()
                if not new_name:
                    abort(400, description="Tag name is required.")
                if new_name != tag.name:
                    clash = deps.FavoriteTag.query.filter_by(
                        user_id=user.id, name=new_name
                    ).first()
                    if clash is not None and clash.id != tag.id:
                        abort(409, description="A tag with that name already exists.")
                    tag.name = new_name
            if "color" in data:
                tag.color = cast(str, data["color"])
            _db_session(deps).commit()
        except SQLAlchemyError:
            _db_session(deps).rollback()
            abort(503, description="Favorites backend is unavailable right now.")
        return _no_store({"tag": _serialize_tag(tag)})

    @favorites_blp.route("/tags/<string:tag_id>", methods=["DELETE"])
    def delete_tag(tag_id: str):
        _guard_not_mocked()
        deps._require_auth_db()
        user = _require_favorites_user()
        try:
            tag = deps.FavoriteTag.query.filter_by(
                id=tag_id, user_id=user.id
            ).first()
            if tag is None:
                abort(404, description="Tag not found.")
            session = _db_session(deps)
            session.query(deps.FavoriteTagAssignment).filter_by(
                tag_id=tag.id
            ).delete(synchronize_session=False)
            session.delete(tag)
            session.commit()
        except SQLAlchemyError:
            _db_session(deps).rollback()
            abort(503, description="Favorites backend is unavailable right now.")
        return _no_store({"deleted": True})

    @favorites_blp.route("/favorites/<string:favorite_id>/tags", methods=["GET"])
    def get_favorite_tags(favorite_id: str):
        _guard_not_mocked()
        deps._require_auth_db()
        user = _require_favorites_user()
        try:
            fav = deps.Favorite.query.filter_by(
                id=favorite_id, user_id=user.id
            ).first()
            if fav is None:
                abort(404, description="Favorite not found.")
        except SQLAlchemyError:
            _db_session(deps).rollback()
            abort(503, description="Favorites backend is unavailable right now.")
        try:
            tag_map = _load_tags_for_favorites(deps, favorite_ids=[fav.id])
        except SQLAlchemyError:
            tag_map = {}
        return _no_store({"tags": tag_map.get(fav.id, [])})

    @favorites_blp.route("/favorites/<string:favorite_id>/tags", methods=["PUT"])
    def set_favorite_tags(favorite_id: str):
        _guard_not_mocked()
        deps._require_auth_db()
        user = _require_favorites_user()
        data = deps._load_json(FavoriteTagsSetSchema())
        raw_ids = data.get("tag_ids")
        if not isinstance(raw_ids, list):
            abort(400, description="tag_ids must be a list.")
        tag_ids = [t for t in (str(x).strip() for x in raw_ids) if t]
        try:
            fav = deps.Favorite.query.filter_by(
                id=favorite_id, user_id=user.id
            ).first()
            if fav is None:
                abort(404, description="Favorite not found.")
            valid_ids: list[str] = []
            if tag_ids:
                rows = (
                    deps.FavoriteTag.query.filter(
                        deps.FavoriteTag.user_id == user.id,
                        deps.FavoriteTag.id.in_(tag_ids),
                    ).all()
                )
                valid_ids = [row.id for row in rows]
                if len(valid_ids) != len(set(tag_ids)):
                    abort(404, description="One or more tags not found.")
            session = _db_session(deps)
            session.query(deps.FavoriteTagAssignment).filter_by(
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
            _db_session(deps).rollback()
            abort(503, description="Favorites backend is unavailable right now.")
        try:
            tag_map = _load_tags_for_favorites(deps, favorite_ids=[fav.id])
        except SQLAlchemyError:
            tag_map = {}
        return _no_store({"tags": tag_map.get(fav.id, [])})

    @favorites_blp.route("/favorites/<string:favorite_id>/projects", methods=["PUT"])
    def set_favorite_projects(favorite_id: str):
        _guard_not_mocked()
        deps._require_auth_db()
        user = _require_favorites_user()
        data = deps._load_json(FavoriteProjectsSetSchema())
        raw_ids = cast(list[object], data["project_ids"])
        project_ids = [value for value in (str(raw).strip() for raw in raw_ids) if value]
        if not project_ids:
            abort(400, description="project_ids must include at least one project.")
        try:
            fav = deps.Favorite.query.filter_by(
                id=favorite_id, user_id=user.id
            ).first()
            if fav is None:
                abort(404, description="Favorite not found.")
            rows = (
                deps.FavoriteProject.query.filter(
                    deps.FavoriteProject.user_id == user.id,
                    deps.FavoriteProject.id.in_(project_ids),
                ).all()
            )
            valid_ids = [row.id for row in rows]
            if len(valid_ids) != len(set(project_ids)):
                abort(404, description="One or more projects not found.")
            session = _db_session(deps)
            session.query(deps.FavoriteProjectAssignment).filter_by(
                favorite_id=fav.id
            ).delete(synchronize_session=False)
            for project_id in valid_ids:
                _ensure_favorite_project_assignment(
                    deps, favorite_id=fav.id, project_id=project_id
                )
            if fav.project_id not in valid_ids:
                fav.project_id = valid_ids[0]
            fav.updated_at = deps._utc_now()
            session.commit()
        except SQLAlchemyError:
            _db_session(deps).rollback()
            abort(503, description="Favorites backend is unavailable right now.")
        return _no_store({"project_ids": valid_ids})

    target_app.register_blueprint(favorites_blp)
    return favorites_blp


__all__ = ["FavoritesDeps", "register_favorites_routes"]
