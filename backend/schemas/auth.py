from marshmallow import Schema, fields


class AuthRegisterSchema(Schema):
    email = fields.Str(required=True)
    password = fields.Str(required=True)
    legal = fields.Dict(required=False, allow_none=True)
    captchaToken = fields.Str(required=False, allow_none=True)


class AuthLoginSchema(Schema):
    email = fields.Str(required=True)
    password = fields.Str(required=True)


class AuthEmailSchema(Schema):
    email = fields.Str(required=True)


class AuthPasswordResetSchema(Schema):
    token = fields.Str(required=True)
    password = fields.Str(required=True)


class AuthTokenSchema(Schema):
    token = fields.Str(required=True)


class AuthApiKeySchema(Schema):
    name = fields.Str(required=False, allow_none=True)


class AuthDeleteAccountSchema(Schema):
    confirm = fields.Str(required=True)


class AuthGoogleCredentialSchema(Schema):
    credential = fields.Str(required=True)
    legal = fields.Dict(required=False, allow_none=True)
