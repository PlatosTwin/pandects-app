# MCP Setup

This backend exposes a read-only authenticated MCP server at `/mcp` and protected-resource metadata at `/.well-known/oauth-protected-resource`.

The MCP server is an OAuth protected resource. Browser sessions and `X-API-Key` auth do not work on `/mcp`. MCP requests must send `Authorization: Bearer <access-token>`.

For Codex specifically, `codex mcp add ...` only registers the server. The login attempt starts with `codex mcp login <name>`. If Codex reports `Dynamic client registration not supported`, the identity provider backing Pandects MCP is not yet exposing the OAuth dynamic client registration support Codex expects.

## ZITADEL Config

For ZITADEL, configure these backend env vars:

```env
MCP_IDENTITY_PROVIDER=zitadel

MCP_PUBLIC_BASE_URL=https://api.pandects.org
MCP_RESOURCE_URL=https://api.pandects.org/mcp
MCP_SERVER_NAME=pandects-mcp
MCP_SERVER_VERSION=0.1.0

MCP_OIDC_ISSUER=https://<your-zitadel-domain>
MCP_OIDC_AUTHORIZATION_SERVER_URL=https://<your-zitadel-domain>
MCP_OIDC_AUDIENCE=https://api.pandects.org/mcp

# Prefer discovery in production unless you have a strong reason to pin JWKS directly.
MCP_OIDC_DISCOVERY_URL=https://<your-zitadel-domain>/.well-known/openid-configuration
# Optional alternative to discovery:
# MCP_OIDC_JWKS_URL=https://<your-zitadel-domain>/oauth/v2/keys

MCP_SUPPORTED_SCOPES=sections:search,agreements:search,agreements:read,agreements:read_fulltext
```

Notes:

- `MCP_OIDC_AUDIENCE` must match the resource/audience ZITADEL includes in the access token for MCP.
- `MCP_RESOURCE_URL` should be the exact MCP resource URL you want clients to use.
- `MCP_OIDC_ISSUER` and `MCP_OIDC_AUTHORIZATION_SERVER_URL` are usually the same for ZITADEL.
- If access tokens are not signed with the default algorithms already accepted by the backend, set `MCP_OIDC_SIGNING_ALGORITHMS`.

## Website Auth Cutover

Pandects website auth now uses Pandects-owned browser pages backed by the same ZITADEL identity that authenticates MCP. On first successful sign-in:

1. Pandects validates the ZITADEL token server-side.
2. If `issuer + subject` is already linked in `auth_external_subjects`, that local user is reused.
3. Otherwise, Pandects auto-links an existing verified local user by verified email, or creates a new local user and links it immediately.
4. If current Pandects legal acceptance is missing, the frontend is sent through a final legal-acceptance step before the local session is issued.

Website auth endpoints:

- `GET /v1/auth/zitadel/google/start`
- `POST /v1/auth/login/password`
- `POST /v1/auth/signup/password`
- `POST /v1/auth/password-reset/request`
- `POST /v1/auth/password-reset/confirm`
- `POST /v1/auth/zitadel/complete`
- `POST /v1/auth/zitadel/finalize`

Optional backend env for website auth:

```env
# Default callback target is {PUBLIC_FRONTEND_BASE_URL}/auth/zitadel/callback
AUTH_ZITADEL_REDIRECT_URI=

# Optional: if set, "Continue with Google" can jump straight to the upstream
# Google IdP inside ZITADEL.
AUTH_ZITADEL_GOOGLE_IDP_HINT=

# Optional static bearer token for Google start + email/password signup/reset APIs.
AUTH_ZITADEL_API_TOKEN=

# Preferred long-term option: mint the ZITADEL API token from a service-app key.
AUTH_ZITADEL_API_CLIENT_ID=
AUTH_ZITADEL_API_KEY_ID=
AUTH_ZITADEL_API_PRIVATE_KEY=

# Optional explicit Google IdP ID. If omitted, the backend falls back to
# AUTH_ZITADEL_GOOGLE_IDP_HINT for the same value.
AUTH_ZITADEL_GOOGLE_IDP_ID=
```

### Pandects-Rendered Auth Emails

Pandects can send verification and password-reset emails itself while still letting ZITADEL own the underlying auth events and codes.

Configure a ZITADEL **Email HTTP provider** that points to:

- `POST {PUBLIC_API_BASE_URL}/v1/auth/zitadel/notifications/email`

Set these backend env vars:

```env
AUTH_ZITADEL_NOTIFICATION_SIGNING_KEY=
AUTH_ZITADEL_NOTIFICATION_MAX_AGE_SECONDS=300

RESEND_API_KEY=
RESEND_FROM_EMAIL=
RESEND_FROM_NAME=Pandects
```

Implementation notes:

- The backend verifies the `Zitadel-Signature` header using the provider signing key.
- Supported notification URLs currently map to:
  - `/verify-email`
  - `/reset-password/confirm`
- The backend renders Pandects HTML emails from the React Email templates in `emails/emails/` and sends them through Resend.
- The `emails/` workspace dependencies must be installed on the server because backend email rendering uses `emails/scripts/render-template.ts`.

For the frontend website-auth flow, set these frontend env vars:

```env
VITE_ZITADEL_AUTHORITY=https://<your-zitadel-domain>
VITE_ZITADEL_CLIENT_ID=<your-public-zitadel-client-id>
VITE_ZITADEL_REDIRECT_URI=https://app.pandects.org/auth/zitadel/callback
VITE_ZITADEL_SCOPES=openid profile email offline_access sections:search agreements:search agreements:read

# Optional if your ZITADEL setup expects them explicitly
VITE_ZITADEL_RESOURCE=https://api.pandects.org/mcp
VITE_ZITADEL_AUDIENCE=https://api.pandects.org/mcp

# Optional overrides if you do not want the default /oauth/v2 paths
VITE_ZITADEL_AUTHORIZATION_ENDPOINT=
VITE_ZITADEL_TOKEN_ENDPOINT=
```

Pandects now owns the user-facing auth pages:

- `/login`
- `/signup`
- `/reset-password`
- `/reset-password/confirm`

`Continue with Google` uses the custom Google-start route, while email sign-in, account creation, and password reset use Pandects backend endpoints directly. The callback page still completes/finalizes Google sign-in with the backend. There is no separate MCP-specific connect flow in the website UI after cutover.

## Legacy Manual Linking

The low-level linking API still exists for migration or admin use:

- `GET /v1/auth/external-subjects`
- `POST /v1/auth/external-subjects`
- `DELETE /v1/auth/external-subjects/<id>`

Example manual link call:

```bash
curl -X POST https://api.pandects.org/v1/auth/external-subjects \
  -H 'Authorization: Bearer <pandects-session-token>' \
  -H 'Content-Type: application/json' \
  -d '{
    "provider": "zitadel",
    "access_token": "<zitadel-access-token>"
  }'
```

Example MCP initialize call:

```bash
curl -X POST https://api.pandects.org/mcp \
  -H 'Authorization: Bearer <zitadel-access-token>' \
  -H 'Content-Type: application/json' \
  -d '{
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize"
  }'
```

## Current Tool Surface

The current MCP tools are:

- `search_sections`
- `list_agreements`
- `get_agreement`

Scope requirements:

- `sections:search` for `search_sections`
- `agreements:search` for `list_agreements`
- `agreements:read` for `get_agreement`
- `agreements:read_fulltext` to receive unredacted XML from agreement-fetch tools

Without `agreements:read_fulltext`, agreement XML stays redacted.
