# MCP Setup

This backend exposes a read-only authenticated MCP server at `/mcp` and protected-resource metadata at `/.well-known/oauth-protected-resource`.

The MCP server is an OAuth protected resource. Browser sessions and `X-API-Key` auth do not work on `/mcp`. MCP requests must send `Authorization: Bearer <access-token>`.

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

## Link Flow

Valid external OAuth tokens are not enough by themselves. The token subject must also be linked to a verified Pandects account in `auth_external_subjects`.

Current linking API:

- `GET /v1/auth/external-subjects`
- `POST /v1/auth/external-subjects`

`POST /v1/auth/external-subjects` expects:

```json
{
  "provider": "zitadel",
  "access_token": "<zitadel-access-token>"
}
```

The caller must already be signed into Pandects with a verified account. The backend verifies the external access token server-side, normalizes `issuer` + `subject`, and then persists the link.

## Admin Or Frontend Sequence

Recommended flow:

1. User signs into Pandects through the normal app auth flow.
2. Frontend or admin tooling obtains a ZITADEL access token for the same user and the MCP audience/resource.
3. Frontend or admin tooling calls `POST /v1/auth/external-subjects` with the Pandects session and the ZITADEL access token.
4. After the link exists, the same external identity can call `/mcp` directly with `Authorization: Bearer <zitadel-access-token>`.

Example link call:

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
