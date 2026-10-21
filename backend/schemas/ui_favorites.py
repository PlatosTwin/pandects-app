"""Response schemas and `UiOperation` entries for the web client's `/v1/me/*` routes.

Documentation only: the favorites routes are plain Flask views
(``backend/routes/favorites``) serialized by ``backend/services/favorites_service.py``,
so nothing here runs at request time. Error responses are added by the exporter.
"""

from __future__ import annotations

from marshmallow import Schema, fields

from backend.schemas.favorites import (
    FavoriteCreateSchema,
    FavoriteExistsQuerySchema,
    FavoriteProjectsSetSchema,
    FavoritesAgreementMetadataSchema,
    FavoritesBulkCopySchema,
    FavoritesBulkMoveSchema,
    FavoritesBulkTagsSchema,
    FavoritesSectionDetailsSchema,
    FavoriteTagsSetSchema,
    FavoriteUpdateSchema,
    ProjectCreateSchema,
    ProjectDeleteQuerySchema,
    ProjectUpdateSchema,
    TagCreateSchema,
    TagUpdateSchema,
)
from backend.schemas.ui_operations import UiOperation

_TAG = "favorites"


# ── Entities ──────────────────────────────────────────────────────────────
class FavoriteProjectSchema(Schema):
    id = fields.Str(required=True)
    name = fields.Str(required=True)
    color = fields.Str(required=True)
    is_default = fields.Bool(required=True)
    sort_order = fields.Int(required=True)
    created_at = fields.Str(required=True, allow_none=True)


class FavoriteTagSchema(Schema):
    id = fields.Str(required=True)
    name = fields.Str(required=True)
    color = fields.Str(required=True)
    created_at = fields.Str(required=True, allow_none=True)


class FavoriteTagRefSchema(Schema):
    """A tag as attached to a favorite (no ``created_at``)."""

    id = fields.Str(required=True)
    name = fields.Str(required=True)
    color = fields.Str(required=True)


class FavoriteMinimalSchema(Schema):
    id = fields.Str(required=True)
    project_id = fields.Str(required=True)
    project_ids = fields.List(fields.Str(), required=True)
    item_type = fields.Str(required=True, metadata={"description": "`section`, `agreement`, or `tax_clause`."})
    item_uuid = fields.Str(required=True)
    note = fields.Str(required=True, allow_none=True)
    context = fields.Dict(
        keys=fields.Str(),
        values=fields.Raw(),
        required=True,
        allow_none=True,
        metadata={"description": "Client-supplied context stored with the favorite."},
    )
    created_at = fields.Str(required=True, allow_none=True)
    updated_at = fields.Str(required=True, allow_none=True)


class FavoriteSchema(FavoriteMinimalSchema):
    agreement_uuid = fields.Str(
        required=True,
        allow_none=True,
        metadata={"description": "Agreement the item belongs to; null when it cannot be resolved."},
    )
    tags = fields.List(fields.Nested(FavoriteTagRefSchema), required=True)


# ── Projects ──────────────────────────────────────────────────────────────
class FavoriteProjectsResponseSchema(Schema):
    projects = fields.List(fields.Nested(FavoriteProjectSchema), required=True)


class FavoriteProjectCreateResponseSchema(Schema):
    project = fields.Nested(FavoriteProjectSchema, required=True)
    created = fields.Bool(required=True)


class FavoriteProjectResponseSchema(Schema):
    project = fields.Nested(FavoriteProjectSchema, required=True)


class FavoriteProjectDeleteResponseSchema(Schema):
    deleted = fields.Bool(required=True)
    reassigned_to_project_id = fields.Str(required=True)
    moved = fields.Int(required=True)


# ── Favorites ─────────────────────────────────────────────────────────────
class FavoritesListQuerySchema(Schema):
    item_type = fields.Str(metadata={"description": "`section`, `agreement`, or `tax_clause`."})
    project_id = fields.Str()
    tag_ids = fields.Str(metadata={"description": "Comma-separated tag ids."})


class FavoritesListResponseSchema(Schema):
    favorites = fields.List(fields.Nested(FavoriteSchema), required=True)


class FavoritesExistsResponseSchema(Schema):
    favorites = fields.Dict(
        keys=fields.Str(),
        values=fields.Str(),
        required=True,
        metadata={"description": "Favorite id keyed by `item_uuid`, for the uuids that are favorited."},
    )


class FavoriteAgreementMetadataSchema(Schema):
    agreement_uuid = fields.Str(required=True)
    year = fields.Int(required=True, allow_none=True)
    target = fields.Str(required=True, allow_none=True)
    acquirer = fields.Str(required=True, allow_none=True)
    transaction_price_total = fields.Float(required=True, allow_none=True)


class FavoritesAgreementMetadataResponseSchema(Schema):
    agreements = fields.Dict(
        keys=fields.Str(),
        values=fields.Nested(FavoriteAgreementMetadataSchema),
        required=True,
        metadata={"description": "Keyed by `agreement_uuid`; missing or ineligible agreements are absent."},
    )


class FavoriteSectionDetailSchema(Schema):
    agreement_uuid = fields.Str(required=True)
    section_uuid = fields.Str(required=True)
    section_standard_id = fields.List(fields.Str(), required=True)
    xml = fields.Str(required=True, allow_none=True)
    article_title = fields.Str(required=True, allow_none=True)
    section_title = fields.Str(required=True, allow_none=True)


class FavoritesSectionDetailsResponseSchema(Schema):
    sections = fields.Dict(
        keys=fields.Str(),
        values=fields.Nested(FavoriteSectionDetailSchema),
        required=True,
        metadata={"description": "Keyed by `section_uuid`; unknown or stale sections are absent."},
    )


class FavoriteCreateQuerySchema(Schema):
    view = fields.Str(metadata={"description": "`minimal` returns the favorite without tags or `agreement_uuid`."})


class FavoriteCreatedSchema(FavoriteMinimalSchema):
    """The full favorite, or only the minimal fields with ``view=minimal``."""

    agreement_uuid = fields.Str(
        allow_none=True, metadata={"description": "Omitted with `view=minimal`."}
    )
    tags = fields.List(
        fields.Nested(FavoriteTagRefSchema), metadata={"description": "Omitted with `view=minimal`."}
    )


class FavoriteCreateResponseSchema(Schema):
    favorite = fields.Nested(FavoriteCreatedSchema, required=True)
    created = fields.Bool(required=True)


class FavoriteResponseSchema(Schema):
    favorite = fields.Nested(FavoriteSchema, required=True)


class FavoritesBulkMoveResponseSchema(Schema):
    project_id = fields.Str(required=True)
    favorite_ids = fields.List(fields.Str(), required=True)
    moved = fields.Int(required=True)


class FavoritesBulkCopyResponseSchema(Schema):
    project_ids = fields.List(fields.Str(), required=True)
    favorite_ids = fields.List(fields.Str(), required=True)
    copied = fields.Int(required=True)


class FavoritesBulkTagsResponseSchema(Schema):
    action = fields.Str(required=True, metadata={"description": "`add` or `remove`."})
    tag_ids = fields.List(fields.Str(), required=True)
    favorite_ids = fields.List(fields.Str(), required=True)
    tags_by_favorite = fields.Dict(
        keys=fields.Str(),
        values=fields.List(fields.Nested(FavoriteTagRefSchema)),
        required=True,
        metadata={"description": "Each updated favorite's tags, keyed by favorite id."},
    )
    updated = fields.Int(required=True)


class FavoriteDeletedResponseSchema(Schema):
    deleted = fields.Bool(required=True)


class FavoriteTagsResponseSchema(Schema):
    tags = fields.List(fields.Nested(FavoriteTagRefSchema), required=True)


class FavoriteProjectIdsResponseSchema(Schema):
    project_ids = fields.List(fields.Str(), required=True)


# ── Tags ──────────────────────────────────────────────────────────────────
class FavoriteTagListResponseSchema(Schema):
    tags = fields.List(fields.Nested(FavoriteTagSchema), required=True)
    available_colors = fields.List(fields.Str(), required=True)


class FavoriteTagCreateResponseSchema(Schema):
    tag = fields.Nested(FavoriteTagSchema, required=True)
    created = fields.Bool(required=True)


class FavoriteTagResponseSchema(Schema):
    tag = fields.Nested(FavoriteTagSchema, required=True)


FAVORITES_OPERATIONS: tuple[UiOperation, ...] = (
    # Projects
    UiOperation(
        path="/v1/me/favorite-projects",
        method="get",
        operation_id="listFavoriteProjects",
        tag=_TAG,
        summary="List the user's favorite projects",
        response=FavoriteProjectsResponseSchema,
    ),
    UiOperation(
        path="/v1/me/favorite-projects",
        method="post",
        operation_id="createFavoriteProject",
        tag=_TAG,
        summary="Create a favorite project",
        response=FavoriteProjectCreateResponseSchema,
        status=201,
        response_description="Created",
        request_body=ProjectCreateSchema,
    ),
    UiOperation(
        path="/v1/me/favorite-projects/{project_id}",
        method="patch",
        operation_id="updateFavoriteProject",
        tag=_TAG,
        summary="Rename, recolor, or reorder a project",
        response=FavoriteProjectResponseSchema,
        request_body=ProjectUpdateSchema,
    ),
    UiOperation(
        path="/v1/me/favorite-projects/{project_id}",
        method="delete",
        operation_id="deleteFavoriteProject",
        tag=_TAG,
        summary="Delete a project, moving its favorites to another project",
        response=FavoriteProjectDeleteResponseSchema,
        query=ProjectDeleteQuerySchema,
    ),
    # Favorites
    UiOperation(
        path="/v1/me/favorites",
        method="get",
        operation_id="listFavorites",
        tag=_TAG,
        summary="List the user's favorites",
        response=FavoritesListResponseSchema,
        query=FavoritesListQuerySchema,
    ),
    UiOperation(
        path="/v1/me/favorites",
        method="post",
        operation_id="createFavorite",
        tag=_TAG,
        summary="Favorite an item (idempotent)",
        response=FavoriteCreateResponseSchema,
        status=201,
        response_description="Created",
        query=FavoriteCreateQuerySchema,
        request_body=FavoriteCreateSchema,
        extra_responses={200: ("Already a favorite", FavoriteCreateResponseSchema)},
    ),
    UiOperation(
        path="/v1/me/favorites/exists",
        method="get",
        operation_id="getFavoritesExist",
        tag=_TAG,
        summary="Which of these items are favorited",
        response=FavoritesExistsResponseSchema,
        query=FavoriteExistsQuerySchema,
    ),
    UiOperation(
        path="/v1/me/favorites/agreement-metadata",
        method="post",
        operation_id="getFavoritesAgreementMetadata",
        tag=_TAG,
        summary="Batch display metadata for favorited agreements",
        response=FavoritesAgreementMetadataResponseSchema,
        request_body=FavoritesAgreementMetadataSchema,
    ),
    UiOperation(
        path="/v1/me/favorites/section-details",
        method="post",
        operation_id="getFavoritesSectionDetails",
        tag=_TAG,
        summary="Batch section payloads for favorited sections",
        response=FavoritesSectionDetailsResponseSchema,
        request_body=FavoritesSectionDetailsSchema,
    ),
    UiOperation(
        path="/v1/me/favorites/{favorite_id}",
        method="patch",
        operation_id="updateFavorite",
        tag=_TAG,
        summary="Update a favorite's note or project",
        response=FavoriteResponseSchema,
        request_body=FavoriteUpdateSchema,
    ),
    UiOperation(
        path="/v1/me/favorites/{favorite_id}",
        method="delete",
        operation_id="deleteFavorite",
        tag=_TAG,
        summary="Remove a favorite",
        response=FavoriteDeletedResponseSchema,
    ),
    UiOperation(
        path="/v1/me/favorites/bulk-move",
        method="post",
        operation_id="bulkMoveFavorites",
        tag=_TAG,
        summary="Move favorites to a project",
        response=FavoritesBulkMoveResponseSchema,
        request_body=FavoritesBulkMoveSchema,
    ),
    UiOperation(
        path="/v1/me/favorites/bulk-copy",
        method="post",
        operation_id="bulkCopyFavorites",
        tag=_TAG,
        summary="Add favorites to more projects",
        response=FavoritesBulkCopyResponseSchema,
        request_body=FavoritesBulkCopySchema,
    ),
    UiOperation(
        path="/v1/me/favorites/bulk-tags",
        method="post",
        operation_id="bulkUpdateFavoriteTags",
        tag=_TAG,
        summary="Add or remove tags on many favorites",
        response=FavoritesBulkTagsResponseSchema,
        request_body=FavoritesBulkTagsSchema,
    ),
    UiOperation(
        path="/v1/me/favorites/{favorite_id}/tags",
        method="get",
        operation_id="getFavoriteTags",
        tag=_TAG,
        summary="A favorite's tags",
        response=FavoriteTagsResponseSchema,
    ),
    UiOperation(
        path="/v1/me/favorites/{favorite_id}/tags",
        method="put",
        operation_id="setFavoriteTags",
        tag=_TAG,
        summary="Replace a favorite's tags",
        response=FavoriteTagsResponseSchema,
        request_body=FavoriteTagsSetSchema,
    ),
    UiOperation(
        path="/v1/me/favorites/{favorite_id}/projects",
        method="put",
        operation_id="setFavoriteProjects",
        tag=_TAG,
        summary="Replace the projects a favorite belongs to",
        response=FavoriteProjectIdsResponseSchema,
        request_body=FavoriteProjectsSetSchema,
    ),
    # Tags
    UiOperation(
        path="/v1/me/tags",
        method="get",
        operation_id="listFavoriteTags",
        tag=_TAG,
        summary="List the user's tags",
        response=FavoriteTagListResponseSchema,
    ),
    UiOperation(
        path="/v1/me/tags",
        method="post",
        operation_id="createFavoriteTag",
        tag=_TAG,
        summary="Create a tag (returns the existing tag with that name)",
        response=FavoriteTagCreateResponseSchema,
        status=201,
        response_description="Created",
        request_body=TagCreateSchema,
        extra_responses={200: ("A tag with that name already exists", FavoriteTagCreateResponseSchema)},
    ),
    UiOperation(
        path="/v1/me/tags/{tag_id}",
        method="patch",
        operation_id="updateFavoriteTag",
        tag=_TAG,
        summary="Rename or recolor a tag",
        response=FavoriteTagResponseSchema,
        request_body=TagUpdateSchema,
    ),
    UiOperation(
        path="/v1/me/tags/{tag_id}",
        method="delete",
        operation_id="deleteFavoriteTag",
        tag=_TAG,
        summary="Delete a tag",
        response=FavoriteDeletedResponseSchema,
    ),
)


__all__ = ["FAVORITES_OPERATIONS"]
