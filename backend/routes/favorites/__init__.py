"""Favorites + projects routes (Phase A).

Surfaces /v1/me/favorites and /v1/me/favorite-projects. All endpoints
require an authenticated, verified user via cookie or bearer session.
API-key access is not supported here — favorites are user-scoped UI
state, not part of the public API.

Routes parse and bound the request, gate access, and shape responses; the
queries and mutations live in ``backend/services/favorites_service.py``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, cast

from flask import Blueprint, Flask, Response, abort, jsonify, make_response, request
from sqlalchemy.exc import SQLAlchemyError

from backend.routes.deps import UserLikeProtocol
from backend.schemas.favorites import (
    ITEM_TYPES,
    TAG_COLORS,
    FavoriteCreateSchema,
    FavoriteExistsQuerySchema,
    FavoriteProjectsSetSchema,
    FavoriteTagsSetSchema,
    FavoritesAgreementMetadataSchema,
    FavoritesSectionDetailsSchema,
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
from backend.services import favorites_service as svc
from backend.services.favorites_service import FavoritesDeps, FavoritesError

__all__ = ["FavoritesDeps", "register_favorites_routes"]

T = TypeVar("T")

_UNAVAILABLE_MESSAGE = "Favorites backend is unavailable right now."


def _no_store(payload: object, *, status: int = 200) -> Response:
    resp = make_response(jsonify(payload))
    resp.headers["Cache-Control"] = "no-store"
    resp.status_code = status
    return resp


def _clean_ids(raw: list[object], *, dedupe: bool = False) -> list[str]:
    values = [value for value in (str(item).strip() for item in raw) if value]
    return list(dict.fromkeys(values)) if dedupe else values


def _csv_arg(name: str) -> list[str]:
    return [value for value in (part.strip() for part in request.args.get(name, "").split(",")) if value]


def register_favorites_routes(target_app: Flask, *, deps: FavoritesDeps) -> Blueprint:
    favorites_blp = Blueprint("favorites", __name__, url_prefix="/v1/me")

    def _favorites_user() -> UserLikeProtocol:
        """Gate every favorites route: real auth DB, verified user, session (not API key)."""
        if deps._auth_is_mocked():
            abort(503, description="Favorites are unavailable in mock auth mode.")
        deps._require_auth_db()
        user, ctx = deps._require_verified_user()
        if ctx.tier != "user":
            abort(403, description="Favorites require a signed-in user session.")
        return user

    def _run(operation: Callable[[], T], *, unavailable: str = _UNAVAILABLE_MESSAGE) -> T:
        try:
            return operation()
        except FavoritesError as exc:
            abort(exc.status, description=exc.message)
        except SQLAlchemyError:
            svc.db_session(deps).rollback()
            abort(503, description=unavailable)

    # ── Projects ──────────────────────────────────────────────────────────
    @favorites_blp.route("/favorite-projects", methods=["GET"])
    def list_projects():
        user = _favorites_user()
        projects = _run(lambda: svc.list_projects(deps, user_id=user.id))
        return _no_store({"projects": projects})

    @favorites_blp.route("/favorite-projects", methods=["POST"])
    def create_project():
        user = _favorites_user()
        data = deps._load_json(ProjectCreateSchema())
        name = cast(str, data["name"]).strip()
        if not name:
            abort(400, description="Project name is required.")
        color = cast(str, data.get("color") or "slate")
        project = _run(lambda: svc.create_project(deps, user_id=user.id, name=name, color=color))
        return _no_store({"project": project, "created": True}, status=201)

    @favorites_blp.route("/favorite-projects/<string:project_id>", methods=["PATCH"])
    def update_project(project_id: str):
        user = _favorites_user()
        data = deps._load_json(ProjectUpdateSchema())
        project = _run(
            lambda: svc.update_project(deps, user_id=user.id, project_id=project_id, data=data)
        )
        return _no_store({"project": project})

    @favorites_blp.route("/favorite-projects/<string:project_id>", methods=["DELETE"])
    def delete_project(project_id: str):
        user = _favorites_user()
        if ProjectDeleteQuerySchema().validate(request.args):
            abort(400, description="Invalid query parameters.")
        reassign_project_id = request.args.get("reassign_project_id")
        result = _run(
            lambda: svc.delete_project(
                deps,
                user_id=user.id,
                project_id=project_id,
                reassign_project_id=reassign_project_id,
            )
        )
        return _no_store(result)

    # ── Favorites ─────────────────────────────────────────────────────────
    @favorites_blp.route("/favorites", methods=["GET"])
    def list_favorites():
        user = _favorites_user()
        item_type = request.args.get("item_type")
        if item_type is not None and item_type not in ITEM_TYPES:
            abort(400, description="Invalid item_type filter.")
        project_id = request.args.get("project_id")
        tag_ids = _csv_arg("tag_ids")
        favorites = _run(
            lambda: svc.list_favorites(
                deps,
                user_id=user.id,
                item_type=item_type,
                project_id=project_id,
                tag_ids=tag_ids,
            )
        )
        return _no_store({"favorites": svc.favorite_payloads(deps, favorites)})

    @favorites_blp.route("/favorites/exists", methods=["GET"])
    def exists_favorites():
        user = _favorites_user()
        if FavoriteExistsQuerySchema().validate(request.args):
            abort(400, description="Invalid query parameters.")
        item_type = request.args["item_type"]
        uuids = _csv_arg("item_uuids")
        if not uuids:
            return _no_store({"favorites": {}})
        if len(uuids) > 200:
            abort(400, description="Too many item_uuids (max 200).")
        found = _run(
            lambda: svc.favorite_ids_by_item_uuid(
                deps, user_id=user.id, item_type=item_type, item_uuids=uuids
            )
        )
        return _no_store({"favorites": found})

    @favorites_blp.route("/favorites/agreement-metadata", methods=["POST"])
    def favorites_agreement_metadata():
        """Batch metadata lookup so the favorites page avoids one request per row.

        POST (not GET) because large favorites lists would overflow URL limits,
        matching the existing bulk-* endpoints.
        """
        _ = _favorites_user()
        data = deps._load_json(FavoritesAgreementMetadataSchema())
        agreement_uuids = _clean_ids(cast(list[object], data["agreement_uuids"]), dedupe=True)
        if not agreement_uuids:
            abort(400, description="agreement_uuids is required.")
        if len(agreement_uuids) > 500:
            abort(400, description="Too many agreement_uuids (max 500).")
        metadata_map = _run(
            lambda: svc.load_agreement_metadata(deps, agreement_uuids=agreement_uuids),
            unavailable="Agreement metadata is unavailable right now.",
        )
        return _no_store({"agreements": metadata_map})

    @favorites_blp.route("/favorites/section-details", methods=["POST"])
    def favorites_section_details():
        """Batch section lookup so the favorites page avoids one request per
        section favorite. POST (not GET) for the same URL-length reason as
        /favorites/agreement-metadata.
        """
        _ = _favorites_user()
        data = deps._load_json(FavoritesSectionDetailsSchema())
        section_uuids = _clean_ids(cast(list[object], data["section_uuids"]), dedupe=True)
        if not section_uuids:
            abort(400, description="section_uuids is required.")
        if len(section_uuids) > 500:
            abort(400, description="Too many section_uuids (max 500).")
        sections_map = _run(
            lambda: svc.load_section_details(deps, section_uuids=section_uuids),
            unavailable="Section details are unavailable right now.",
        )
        return _no_store({"sections": sections_map})

    @favorites_blp.route("/favorites", methods=["POST"])
    def create_favorite():
        user = _favorites_user()
        data = deps._load_json(FavoriteCreateSchema())
        minimal_response = request.args.get("view") == "minimal"
        item_type = cast(str, data["item_type"])
        item_uuid = cast(str, data["item_uuid"]).strip()
        note_val = data.get("note")
        note = note_val.strip() if isinstance(note_val, str) and note_val.strip() else None
        context_val = data.get("context")
        context = cast(dict[str, object], context_val) if isinstance(context_val, dict) else None
        project_id = data.get("project_id")
        if project_id is not None and not isinstance(project_id, str):
            abort(400, description="project_id must be a string.")

        fav, created = _run(
            lambda: svc.upsert_favorite(
                deps,
                user_id=user.id,
                item_type=item_type,
                item_uuid=item_uuid,
                note=note,
                context=context,
                project_id=project_id,
            )
        )
        status = 201 if created else 200
        if minimal_response:
            return _no_store(
                {"favorite": svc.serialize_favorite_minimal(fav), "created": created},
                status=status,
            )
        return _no_store(
            {"favorite": svc.favorite_payloads(deps, [fav])[0], "created": created},
            status=status,
        )

    @favorites_blp.route("/favorites/<string:favorite_id>", methods=["PATCH"])
    def update_favorite(favorite_id: str):
        user = _favorites_user()
        data = deps._load_json(FavoriteUpdateSchema())
        fav = _run(
            lambda: svc.update_favorite(deps, user_id=user.id, favorite_id=favorite_id, data=data)
        )
        return _no_store({"favorite": svc.favorite_payloads(deps, [fav])[0]})

    @favorites_blp.route("/favorites/bulk-move", methods=["POST"])
    def bulk_move_favorites():
        user = _favorites_user()
        data = deps._load_json(FavoritesBulkMoveSchema())
        favorite_ids = _clean_ids(cast(list[object], data["favorite_ids"]))
        if not favorite_ids:
            abort(400, description="favorite_ids is required.")
        if len(favorite_ids) > 500:
            abort(400, description="Too many favorite_ids (max 500).")
        project_id = cast(str, data["project_id"]).strip()
        result = _run(
            lambda: svc.bulk_move_favorites(
                deps, user_id=user.id, favorite_ids=favorite_ids, project_id=project_id
            )
        )
        return _no_store(result)

    @favorites_blp.route("/favorites/bulk-copy", methods=["POST"])
    def bulk_copy_favorites():
        user = _favorites_user()
        data = deps._load_json(FavoritesBulkCopySchema())
        favorite_ids = _clean_ids(cast(list[object], data["favorite_ids"]))
        project_ids = _clean_ids(cast(list[object], data["project_ids"]))
        if not favorite_ids or not project_ids:
            abort(400, description="favorite_ids and project_ids are required.")
        if len(favorite_ids) > 500:
            abort(400, description="Too many favorite_ids (max 500).")
        if len(project_ids) > 50:
            abort(400, description="Too many project_ids (max 50).")
        result = _run(
            lambda: svc.bulk_copy_favorites(
                deps, user_id=user.id, favorite_ids=favorite_ids, project_ids=project_ids
            )
        )
        return _no_store(result)

    @favorites_blp.route("/favorites/bulk-tags", methods=["POST"])
    def bulk_update_favorite_tags():
        user = _favorites_user()
        data = deps._load_json(FavoritesBulkTagsSchema())
        favorite_ids = _clean_ids(cast(list[object], data["favorite_ids"]), dedupe=True)
        tag_ids = _clean_ids(cast(list[object], data["tag_ids"]), dedupe=True)
        action = cast(str, data["action"])
        if not favorite_ids or not tag_ids:
            abort(400, description="favorite_ids and tag_ids are required.")
        if len(favorite_ids) > 500:
            abort(400, description="Too many favorite_ids (max 500).")
        result = _run(
            lambda: svc.bulk_update_favorite_tags(
                deps,
                user_id=user.id,
                favorite_ids=favorite_ids,
                tag_ids=tag_ids,
                action=action,
            )
        )
        return _no_store(result)

    @favorites_blp.route("/favorites/<string:favorite_id>", methods=["DELETE"])
    def delete_favorite(favorite_id: str):
        user = _favorites_user()
        _run(lambda: svc.delete_favorite(deps, user_id=user.id, favorite_id=favorite_id))
        return _no_store({"deleted": True})

    @favorites_blp.route("/favorites/<string:favorite_id>/tags", methods=["GET"])
    def get_favorite_tags(favorite_id: str):
        user = _favorites_user()
        fav = _run(lambda: svc.get_owned_favorite(deps, user_id=user.id, favorite_id=favorite_id))
        return _no_store({"tags": svc.favorite_tags_best_effort(deps, fav.id)})

    @favorites_blp.route("/favorites/<string:favorite_id>/tags", methods=["PUT"])
    def set_favorite_tags(favorite_id: str):
        user = _favorites_user()
        data = deps._load_json(FavoriteTagsSetSchema())
        raw_ids = data.get("tag_ids")
        if not isinstance(raw_ids, list):
            abort(400, description="tag_ids must be a list.")
        tag_ids = _clean_ids(cast(list[object], raw_ids))
        fav = _run(
            lambda: svc.set_favorite_tags(
                deps, user_id=user.id, favorite_id=favorite_id, tag_ids=tag_ids
            )
        )
        return _no_store({"tags": svc.favorite_tags_best_effort(deps, fav.id)})

    @favorites_blp.route("/favorites/<string:favorite_id>/projects", methods=["PUT"])
    def set_favorite_projects(favorite_id: str):
        user = _favorites_user()
        data = deps._load_json(FavoriteProjectsSetSchema())
        project_ids = _clean_ids(cast(list[object], data["project_ids"]))
        if not project_ids:
            abort(400, description="project_ids must include at least one project.")
        valid_ids = _run(
            lambda: svc.set_favorite_projects(
                deps, user_id=user.id, favorite_id=favorite_id, project_ids=project_ids
            )
        )
        return _no_store({"project_ids": valid_ids})

    # ── Tags ──────────────────────────────────────────────────────────────
    @favorites_blp.route("/tags", methods=["GET"])
    def list_tags():
        user = _favorites_user()
        tags = _run(lambda: svc.list_tags(deps, user_id=user.id))
        return _no_store({"tags": tags, "available_colors": list(TAG_COLORS)})

    @favorites_blp.route("/tags", methods=["POST"])
    def create_tag():
        user = _favorites_user()
        data = deps._load_json(TagCreateSchema())
        name = cast(str, data["name"]).strip()
        if not name:
            abort(400, description="Tag name is required.")
        color = cast(str, data.get("color") or "slate")
        tag, created = _run(lambda: svc.create_tag(deps, user_id=user.id, name=name, color=color))
        return _no_store({"tag": tag, "created": created}, status=201 if created else 200)

    @favorites_blp.route("/tags/<string:tag_id>", methods=["PATCH"])
    def update_tag(tag_id: str):
        user = _favorites_user()
        data = deps._load_json(TagUpdateSchema())
        tag = _run(lambda: svc.update_tag(deps, user_id=user.id, tag_id=tag_id, data=data))
        return _no_store({"tag": tag})

    @favorites_blp.route("/tags/<string:tag_id>", methods=["DELETE"])
    def delete_tag(tag_id: str):
        user = _favorites_user()
        _run(lambda: svc.delete_tag(deps, user_id=user.id, tag_id=tag_id))
        return _no_store({"deleted": True})

    target_app.register_blueprint(favorites_blp)
    return favorites_blp
