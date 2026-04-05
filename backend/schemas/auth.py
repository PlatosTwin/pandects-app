"""Marshmallow schemas for auth request/response validation."""

from marshmallow import Schema, fields


class AuthApiKeySchema(Schema):
    name = fields.Str(required=False, allow_none=True)


class AuthDeleteAccountSchema(Schema):
    confirm = fields.Str(required=True)


class AuthPasswordLoginSchema(Schema):
    email = fields.Email(required=True)
    password = fields.Str(required=True)
    next = fields.Str(required=False, allow_none=True)


class AuthPasswordSignupSchema(Schema):
    email = fields.Email(required=True)
    password = fields.Str(required=True)
    first_name = fields.Str(required=False, allow_none=True)
    last_name = fields.Str(required=False, allow_none=True)
    next = fields.Str(required=False, allow_none=True)


class AuthPasswordResetRequestSchema(Schema):
    email = fields.Email(required=True)


class AuthPasswordResetConfirmSchema(Schema):
    user_id = fields.Str(required=True)
    code = fields.Str(required=True)
    password = fields.Str(required=True)


class AuthExternalSubjectLinkSchema(Schema):
    access_token = fields.Str(required=True)
    provider = fields.Str(required=False, allow_none=True)


class AuthFlagInaccurateSchema(Schema):
    source = fields.Str(required=True)
    agreement_uuid = fields.Str(required=True)
    section_uuid = fields.Str(required=False, allow_none=True)
    message = fields.Str(required=False, allow_none=True)
    request_follow_up = fields.Bool(required=False, allow_none=True)
    issue_types = fields.List(fields.Str(), required=True)
