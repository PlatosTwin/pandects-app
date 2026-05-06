"""Schemas for favorites/projects/tags."""

from marshmallow import Schema, fields, validate

ITEM_TYPES = ("section", "agreement", "tax_clause")

# Fixed palette: keeps tag visuals coherent across the app and avoids a11y traps
# from arbitrary user-picked hex colors.
TAG_COLORS = (
    "slate",
    "red",
    "orange",
    "amber",
    "green",
    "teal",
    "blue",
    "violet",
)


class FavoriteCreateSchema(Schema):
    item_type = fields.Str(required=True, validate=validate.OneOf(ITEM_TYPES))
    item_uuid = fields.Str(required=True)
    project_id = fields.Str(required=False, allow_none=True)
    note = fields.Str(required=False, allow_none=True)
    context = fields.Dict(required=False, allow_none=True)


class FavoriteUpdateSchema(Schema):
    note = fields.Str(required=False, allow_none=True)
    project_id = fields.Str(required=False, allow_none=True)


class FavoriteExistsQuerySchema(Schema):
    item_type = fields.Str(required=True, validate=validate.OneOf(ITEM_TYPES))
    item_uuids = fields.Str(required=True)


class TagCreateSchema(Schema):
    name = fields.Str(required=True, validate=validate.Length(min=1, max=64))
    color = fields.Str(required=False, validate=validate.OneOf(TAG_COLORS))


class TagUpdateSchema(Schema):
    name = fields.Str(required=False, validate=validate.Length(min=1, max=64))
    color = fields.Str(required=False, validate=validate.OneOf(TAG_COLORS))


class FavoriteTagsSetSchema(Schema):
    tag_ids = fields.List(fields.Str(), required=True)
