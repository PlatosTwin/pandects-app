"""Marshmallow schemas for auth request/response validation."""

from marshmallow import Schema, fields


class AuthRegisterSchema(Schema):
    email = fields.Str(required=True)
    password = fields.Str(required=True)
    legal = fields.Dict(required=False, allow_none=True)
    captcha_token = fields.Str(required=False, allow_none=True)


class AuthLoginSchema(Schema):
    email = fields.Str(required=True)
    password = fields.Str(required=True)


class AuthApiKeySchema(Schema):
    name = fields.Str(required=False, allow_none=True)


class AuthDeleteAccountSchema(Schema):
    confirm = fields.Str(required=True)


class AuthGoogleCredentialSchema(Schema):
    credential = fields.Str(required=True)
    legal = fields.Dict(required=False, allow_none=True)


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
