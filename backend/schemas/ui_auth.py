"""Response schemas and `UiOperation` entries for the web client's `/v1/auth/*` routes.

Documentation only: the auth routes are plain Flask views that build their JSON
by hand (``backend/routes/auth``), so nothing here runs at request time. Error
responses (the ``{"error", "message"}`` envelope) are added by the exporter.
"""

from __future__ import annotations

from marshmallow import Schema, fields

from backend.schemas.auth import (
    AuthApiKeySchema,
    AuthDeleteAccountSchema,
    AuthExternalSubjectLinkSchema,
    AuthFlagInaccurateSchema,
    AuthPasswordLoginSchema,
    AuthPasswordResetConfirmSchema,
    AuthPasswordResetRequestSchema,
    AuthPasswordSignupSchema,
)
from backend.schemas.ui_operations import UiOperation

_TAG = "auth"


# ── Shared shapes ─────────────────────────────────────────────────────────
class AuthStatusResponseSchema(Schema):
    status = fields.Str(
        required=True,
        metadata={
            "description": (
                "Outcome keyword: `ok`, `unlinked`, `requested`, `updated`, `revoked`, "
                "or `deleted`, depending on the route."
            )
        },
    )


class AuthUserSchema(Schema):
    id = fields.Str(required=True)
    email = fields.Str(required=True)


class AuthMeResponseSchema(Schema):
    user = fields.Nested(AuthUserSchema, required=True)


class AuthFlowResponseSchema(Schema):
    """Result of a website sign-in step (password login/signup, ZITADEL callback, legal finalize)."""

    status = fields.Str(
        required=True,
        metadata={
            "description": (
                "`authenticated` (session issued), `legal_required` (accept the terms via "
                "`/v1/auth/zitadel/finalize`), `verification_required` (check email), or "
                "`mcp_token` (`/v1/auth/zitadel/complete` in the MCP-token flow only)."
            )
        },
    )
    next_path = fields.Str(required=True, metadata={"description": "Where the client goes next."})
    user = fields.Nested(
        AuthUserSchema,
        metadata={"description": "Present for every status except `mcp_token`."},
    )
    session_token = fields.Str(
        metadata={
            "description": "Session token; only with `status=authenticated` under bearer session transport."
        }
    )
    access_token = fields.Str(metadata={"description": "MCP access token (`status=mcp_token`)."})
    token_type = fields.Str(metadata={"description": "`status=mcp_token`, when the IdP returns one."})
    expires_in = fields.Int(metadata={"description": "`status=mcp_token`, when the IdP returns one."})
    scope = fields.Str(metadata={"description": "`status=mcp_token`, when the IdP returns one."})


class AuthAuthorizeUrlResponseSchema(Schema):
    authorize_url = fields.Str(
        required=True, metadata={"description": "Identity-provider URL the browser navigates to."}
    )


# ── Individual routes ─────────────────────────────────────────────────────
class AuthCsrfResponseSchema(Schema):
    status = fields.Str(required=True, metadata={"description": "Always `ok`."})
    csrf_token = fields.Str(
        metadata={"description": "CSRF token; only under cookie session transport."}
    )


class AuthCaptchaDebugSchema(Schema):
    turnstile_enabled = fields.Str(allow_none=True)
    has_site_key = fields.Bool()
    has_secret_key = fields.Bool()


class AuthCaptchaSiteKeyResponseSchema(Schema):
    enabled = fields.Bool(required=True)
    site_key = fields.Str(metadata={"description": "Turnstile site key; present when `enabled`."})
    debug = fields.Nested(
        AuthCaptchaDebugSchema, metadata={"description": "Only when the app runs in debug mode."}
    )


class AuthVerificationResendUserSchema(Schema):
    email = fields.Str(required=True)


class AuthVerificationResendResponseSchema(Schema):
    status = fields.Str(required=True, metadata={"description": "Always `verification_required`."})
    user = fields.Nested(AuthVerificationResendUserSchema, required=True)


class AuthEmailVerifyConfirmBodySchema(Schema):
    user_id = fields.Str(required=True)
    code = fields.Str(required=True)
    next = fields.Str()


class AuthEmailVerifyResendBodySchema(Schema):
    email = fields.Str(required=True)


class AuthZitadelCompleteBodySchema(Schema):
    code = fields.Str(metadata={"description": "OAuth code (with `state`), or send the intent pair."})
    state = fields.Str()
    intent_id = fields.Str()
    intent_token = fields.Str()
    user_id = fields.Str()


class AuthZitadelLinkCompleteBodySchema(Schema):
    code = fields.Str(required=True)
    state = fields.Str(required=True)


class AuthLegalAcceptanceSchema(Schema):
    checked_at_ms = fields.Int(
        required=True, metadata={"description": "When the user checked the box (epoch ms)."}
    )
    docs = fields.List(
        fields.Str(),
        required=True,
        metadata={"description": "Must be exactly `tos`, `privacy`, and `license`."},
    )


class AuthLegalAcceptanceBodySchema(Schema):
    """Body checked by ``backend.auth.legal_runtime.require_legal_acceptance``."""

    legal = fields.Nested(AuthLegalAcceptanceSchema, required=True)


class AuthApiKeySummarySchema(Schema):
    id = fields.Str(required=True)
    name = fields.Str(required=True, allow_none=True)
    prefix = fields.Str(required=True)
    created_at = fields.Str(required=True)
    last_used_at = fields.Str(required=True, allow_none=True)
    revoked_at = fields.Str(required=True, allow_none=True)


class AuthApiKeysResponseSchema(Schema):
    keys = fields.List(fields.Nested(AuthApiKeySummarySchema), required=True)


class AuthCreatedApiKeySchema(Schema):
    id = fields.Str(required=True)
    name = fields.Str(required=True, allow_none=True)
    prefix = fields.Str(required=True)
    created_at = fields.Str(required=True)


class AuthCreateApiKeyResponseSchema(Schema):
    api_key = fields.Nested(AuthCreatedApiKeySchema, required=True)
    api_key_plaintext = fields.Str(
        required=True, metadata={"description": "The full key; shown only in this response."}
    )


class AuthUsageDaySchema(Schema):
    day = fields.Str(required=True, metadata={"description": "ISO date."})
    count = fields.Int(required=True)


class AuthUsageResponseSchema(Schema):
    by_day = fields.List(fields.Nested(AuthUsageDaySchema), required=True)
    total = fields.Int(required=True)


class AuthUsageQuerySchema(Schema):
    period = fields.Str(metadata={"description": "`1w`, `1m` (default), `1y`, or `all`."})
    api_key_id = fields.Str(metadata={"description": "Limit to one of the caller's keys (UUID)."})


class AuthExternalSubjectSchema(Schema):
    id = fields.Int(required=True)
    issuer = fields.Str(required=True)
    subject = fields.Str(required=True)
    created_at = fields.Str(required=True)


class AuthExternalSubjectsResponseSchema(Schema):
    links = fields.List(fields.Nested(AuthExternalSubjectSchema), required=True)


class AuthExternalLinkSchema(Schema):
    id = fields.Int(required=True)
    provider = fields.Str(required=True)
    issuer = fields.Str(required=True)
    subject = fields.Str(required=True)
    created_at = fields.Str(required=True)


class AuthExternalLinkResponseSchema(Schema):
    status = fields.Str(required=True, metadata={"description": "`linked` (201) or `already_linked` (200)."})
    link = fields.Nested(AuthExternalLinkSchema, required=True)


class AuthZitadelLinkCompleteResponseSchema(AuthExternalLinkResponseSchema):
    return_to = fields.Str(required=True)


class AuthNextQuerySchema(Schema):
    next = fields.Str(metadata={"description": "Same-site path to return to (default `/account`)."})


class AuthZitadelStartQuerySchema(AuthNextQuerySchema):
    provider = fields.Str(metadata={"description": "`email` (default) or `google`."})
    prompt = fields.Str(metadata={"description": "`login`, `create`, or `select_account`."})


class AuthOAuthConsentBodySchema(Schema):
    client_id = fields.Str(required=True)
    scope = fields.Str(required=True)


class AuthOAuthConsentResponseSchema(Schema):
    status = fields.Str(required=True, metadata={"description": "Always `granted`."})
    client_id = fields.Str(required=True)
    scope = fields.Str(required=True, metadata={"description": "The full granted scope after merging."})


class AuthOAuthAuthorizeQuerySchema(Schema):
    client_id = fields.Str(required=True)
    redirect_uri = fields.Str(required=True)
    response_type = fields.Str(required=True, metadata={"description": "Must be `code`."})
    scope = fields.Str()
    state = fields.Str()
    code_challenge = fields.Str(required=True)
    code_challenge_method = fields.Str(metadata={"description": "Must be `S256`."})


AUTH_OPERATIONS: tuple[UiOperation, ...] = (
    UiOperation(
        path="/v1/auth/health",
        method="get",
        operation_id="getAuthHealth",
        tag=_TAG,
        summary="Check that the auth backend is reachable",
        response=AuthStatusResponseSchema,
    ),
    UiOperation(
        path="/v1/auth/csrf",
        method="get",
        operation_id="getAuthCsrf",
        tag=_TAG,
        summary="Fetch (and set) the CSRF token",
        response=AuthCsrfResponseSchema,
    ),
    UiOperation(
        path="/v1/auth/captcha/site-key",
        method="get",
        operation_id="getAuthCaptchaSiteKey",
        tag=_TAG,
        summary="Turnstile captcha configuration",
        response=AuthCaptchaSiteKeyResponseSchema,
    ),
    UiOperation(
        path="/v1/auth/me",
        method="get",
        operation_id="getAuthMe",
        tag=_TAG,
        summary="The signed-in, verified user",
        response=AuthMeResponseSchema,
    ),
    UiOperation(
        path="/v1/auth/logout",
        method="post",
        operation_id="postAuthLogout",
        tag=_TAG,
        summary="Revoke the current session",
        response=AuthStatusResponseSchema,
    ),
    UiOperation(
        path="/v1/auth/login/password",
        method="post",
        operation_id="postAuthPasswordLogin",
        tag=_TAG,
        summary="Sign in with email and password",
        response=AuthFlowResponseSchema,
        request_body=AuthPasswordLoginSchema,
        description="Returns `authenticated` or `legal_required`.",
    ),
    UiOperation(
        path="/v1/auth/signup/password",
        method="post",
        operation_id="postAuthPasswordSignup",
        tag=_TAG,
        summary="Create an account with email and password",
        response=AuthFlowResponseSchema,
        request_body=AuthPasswordSignupSchema,
        description="Returns `verification_required` or `legal_required`.",
    ),
    UiOperation(
        path="/v1/auth/password-reset/request",
        method="post",
        operation_id="postAuthPasswordResetRequest",
        tag=_TAG,
        summary="Email a password-reset link",
        response=AuthStatusResponseSchema,
        request_body=AuthPasswordResetRequestSchema,
        description="Same response whether or not the account exists (`status=requested`).",
    ),
    UiOperation(
        path="/v1/auth/password-reset/confirm",
        method="post",
        operation_id="postAuthPasswordResetConfirm",
        tag=_TAG,
        summary="Set a new password with a reset code",
        response=AuthStatusResponseSchema,
        request_body=AuthPasswordResetConfirmSchema,
    ),
    UiOperation(
        path="/v1/auth/email/verify/confirm",
        method="post",
        operation_id="postAuthEmailVerifyConfirm",
        tag=_TAG,
        summary="Confirm an email verification code",
        response=AuthFlowResponseSchema,
        request_body=AuthEmailVerifyConfirmBodySchema,
    ),
    UiOperation(
        path="/v1/auth/email/verify/resend",
        method="post",
        operation_id="postAuthEmailVerifyResend",
        tag=_TAG,
        summary="Resend the verification email",
        response=AuthVerificationResendResponseSchema,
        request_body=AuthEmailVerifyResendBodySchema,
    ),
    UiOperation(
        path="/v1/auth/zitadel/start",
        method="get",
        operation_id="getAuthZitadelStart",
        tag=_TAG,
        summary="Start a ZITADEL website sign-in",
        response=AuthAuthorizeUrlResponseSchema,
        query=AuthZitadelStartQuerySchema,
    ),
    UiOperation(
        path="/v1/auth/zitadel/google/start",
        method="get",
        operation_id="getAuthZitadelGoogleStart",
        tag=_TAG,
        summary="Start a Google sign-in through ZITADEL",
        response=AuthAuthorizeUrlResponseSchema,
        query=AuthNextQuerySchema,
    ),
    UiOperation(
        path="/v1/auth/zitadel/complete",
        method="post",
        operation_id="postAuthZitadelComplete",
        tag=_TAG,
        summary="Finish a ZITADEL website sign-in callback",
        response=AuthFlowResponseSchema,
        request_body=AuthZitadelCompleteBodySchema,
    ),
    UiOperation(
        path="/v1/auth/zitadel/finalize",
        method="post",
        operation_id="postAuthZitadelFinalize",
        tag=_TAG,
        summary="Accept the terms to finish a pending sign-in",
        response=AuthFlowResponseSchema,
        request_body=AuthLegalAcceptanceBodySchema,
        description="Returns `authenticated` or `verification_required`.",
    ),
    UiOperation(
        path="/v1/auth/mcp-token/start",
        method="get",
        operation_id="getAuthMcpTokenStart",
        tag=_TAG,
        summary="Start the flow that mints an MCP access token",
        response=AuthAuthorizeUrlResponseSchema,
        query=AuthNextQuerySchema,
    ),
    UiOperation(
        path="/v1/auth/api-keys",
        method="get",
        operation_id="listAuthApiKeys",
        tag=_TAG,
        summary="List the user's API keys",
        response=AuthApiKeysResponseSchema,
    ),
    UiOperation(
        path="/v1/auth/api-keys",
        method="post",
        operation_id="createAuthApiKey",
        tag=_TAG,
        summary="Create an API key",
        response=AuthCreateApiKeyResponseSchema,
        request_body=AuthApiKeySchema,
    ),
    UiOperation(
        path="/v1/auth/api-keys/{key_id}",
        method="delete",
        operation_id="revokeAuthApiKey",
        tag=_TAG,
        summary="Revoke an API key",
        response=AuthStatusResponseSchema,
    ),
    UiOperation(
        path="/v1/auth/api-keys/{key_id}/permanent",
        method="delete",
        operation_id="deleteAuthApiKey",
        tag=_TAG,
        summary="Permanently delete an API key",
        response=AuthStatusResponseSchema,
    ),
    UiOperation(
        path="/v1/auth/usage",
        method="get",
        operation_id="getAuthUsage",
        tag=_TAG,
        summary="API request counts per day",
        response=AuthUsageResponseSchema,
        query=AuthUsageQuerySchema,
    ),
    UiOperation(
        path="/v1/auth/external-subjects",
        method="get",
        operation_id="listAuthExternalSubjects",
        tag=_TAG,
        summary="List linked external identities",
        response=AuthExternalSubjectsResponseSchema,
    ),
    UiOperation(
        path="/v1/auth/external-subjects",
        method="post",
        operation_id="linkAuthExternalSubject",
        tag=_TAG,
        summary="Link an external identity from an access token",
        response=AuthExternalLinkResponseSchema,
        status=201,
        response_description="Linked",
        request_body=AuthExternalSubjectLinkSchema,
        extra_responses={200: ("Already linked", AuthExternalLinkResponseSchema)},
    ),
    UiOperation(
        path="/v1/auth/external-subjects/{link_id}",
        method="delete",
        operation_id="unlinkAuthExternalSubject",
        tag=_TAG,
        summary="Unlink an external identity",
        response=AuthStatusResponseSchema,
        int_path_params=("link_id",),
    ),
    UiOperation(
        path="/v1/auth/external-subjects/zitadel/start",
        method="get",
        operation_id="getAuthZitadelLinkStart",
        tag=_TAG,
        summary="Start linking a ZITADEL identity",
        response=AuthAuthorizeUrlResponseSchema,
        query=AuthNextQuerySchema,
    ),
    UiOperation(
        path="/v1/auth/external-subjects/zitadel/complete",
        method="post",
        operation_id="postAuthZitadelLinkComplete",
        tag=_TAG,
        summary="Finish linking a ZITADEL identity",
        response=AuthZitadelLinkCompleteResponseSchema,
        status=201,
        response_description="Linked",
        request_body=AuthZitadelLinkCompleteBodySchema,
        extra_responses={200: ("Already linked", AuthZitadelLinkCompleteResponseSchema)},
    ),
    UiOperation(
        path="/v1/auth/flag-inaccurate",
        method="post",
        operation_id="postAuthFlagInaccurate",
        tag=_TAG,
        summary="Report inaccurate agreement or section data",
        response=AuthStatusResponseSchema,
        request_body=AuthFlagInaccurateSchema,
    ),
    UiOperation(
        path="/v1/auth/account/delete",
        method="post",
        operation_id="deleteAuthAccount",
        tag=_TAG,
        summary="Delete the signed-in account",
        response=AuthStatusResponseSchema,
        request_body=AuthDeleteAccountSchema,
    ),
    UiOperation(
        path="/v1/auth/oauth/authorize",
        method="get",
        operation_id="getAuthOAuthAuthorize",
        tag=_TAG,
        summary="MCP OAuth authorization endpoint (browser navigation)",
        response=None,
        status=302,
        response_description=(
            "Redirect to the client's `redirect_uri` (code or error) or to the consent page"
        ),
        query=AuthOAuthAuthorizeQuerySchema,
        extra_responses={
            200: ("HTML page that sends the browser to sign in first", None),
        },
    ),
    UiOperation(
        path="/v1/auth/oauth/consent",
        method="post",
        operation_id="postAuthOAuthConsent",
        tag=_TAG,
        summary="Grant an MCP OAuth client the requested scopes",
        response=AuthOAuthConsentResponseSchema,
        request_body=AuthOAuthConsentBodySchema,
    ),
    UiOperation(
        path="/v1/auth/oauth/browser-session",
        method="post",
        operation_id="postAuthOAuthBrowserSession",
        tag=_TAG,
        summary="Bind the signed-in user to the OAuth browser cookie",
        response=AuthStatusResponseSchema,
    ),
)


__all__ = ["AUTH_OPERATIONS"]
