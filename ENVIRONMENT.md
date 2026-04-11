# Environment Reference

This file explains the environment variables used across the repository and how contributors should think about them.

## Categories

- `public-safe`: can be set by outside contributors without private access
- `optional`: useful overrides, but not required for normal local work
- `maintainer-only`: tied to private infrastructure or operational workflows
- `secret`: must never be committed; usually also maintainer-only

## Backend

### Public-safe

- `PUBLIC_API_BASE_URL`: local-safe default such as `http://localhost:5113`
- `PUBLIC_FRONTEND_BASE_URL`: local-safe default such as `http://localhost:8080`
- `SKIP_MAIN_DB_REFLECTION`: set to `1` for public-safe backend work
- `ENABLE_MAIN_DB_REFLECTION`: leave enabled only when you actually have a valid main DB connection
- `AUTH_SESSION_TRANSPORT`: `bearer` is the normal local-safe choice
- `FILTER_OPTIONS_TTL_SECONDS`
- `DUMPS_CACHE_TTL_SECONDS`
- `DUMPS_MANIFEST_CACHE_TTL_SECONDS`
- `SEARCH_EXPLAIN_ESTIMATE_ENABLED`
- `RATE_LIMIT_MAX_KEYS`
- `RATE_LIMIT_PRUNE_INTERVAL_SECONDS`
- `API_KEY_LAST_USED_TOUCH_SECONDS`
- `API_KEY_LAST_USED_MAX_KEYS`
- `USAGE_SAMPLE_RATE_2XX`
- `USAGE_SAMPLE_RATE_3XX`
- `USAGE_LOG_BUFFER_ENABLED`
- `USAGE_LOG_BUFFER_FLUSH_SECONDS`
- `USAGE_LOG_BUFFER_MAX_EVENTS`
- `ASYNC_SIDE_EFFECTS_ENABLED`
- `ASYNC_SIDE_EFFECTS_MAX_QUEUE`

### Optional

- `MAIN_DATABASE_URI`: alternative explicit DB URI when not using `MARIADB_*`
- `MAIN_DB_SCHEMA`
- `AUTH_DATABASE_URI`: optional override for the auth DB; local work otherwise uses sqlite
- `MCP_IDENTITY_PROVIDER`
- `MCP_PUBLIC_BASE_URL`
- `MCP_SERVER_NAME`
- `MCP_SERVER_VERSION`
- `MCP_SUPPORTED_SCOPES`
- `CORS_ORIGINS`

### Maintainer-only or secret

- `MARIADB_USER`
- `MARIADB_PASSWORD`
- `MARIADB_HOST`
- `MARIADB_DATABASE`
- `AUTH_SECRET_KEY`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `GOOGLE_OAUTH_CLIENT_ID`
- `GOOGLE_OAUTH_CLIENT_SECRET`
- `MCP_OIDC_ISSUER`
- `MCP_OIDC_AUTHORIZATION_SERVER_URL`
- `MCP_OIDC_AUTHORIZATION_ENDPOINT`
- `MCP_OIDC_TOKEN_ENDPOINT`
- `MCP_OIDC_AUDIENCE`
- `MCP_OIDC_JWKS_URL`
- `MCP_OIDC_DISCOVERY_URL`
- `MCP_OIDC_INTROSPECTION_ENDPOINT`
- `MCP_OIDC_INTROSPECTION_CLIENT_ID`
- `MCP_OIDC_INTROSPECTION_CLIENT_SECRET`
- `MCP_OIDC_SIGNING_ALGORITHMS`
- `MCP_ZITADEL_CLIENT_ID`
- `MCP_ZITADEL_REDIRECT_URI`
- `MCP_ZITADEL_SCOPES`
- `MCP_ZITADEL_AUDIENCE`
- `MCP_ZITADEL_RESOURCE`
- `MCP_ZITADEL_ROLE_SCOPE_MAP`
- `AUTH_ZITADEL_REDIRECT_URI`
- `AUTH_ZITADEL_GOOGLE_IDP_HINT`
- `AUTH_ZITADEL_API_TOKEN`
- `AUTH_ZITADEL_API_CLIENT_ID`
- `AUTH_ZITADEL_API_KEY_ID`
- `AUTH_ZITADEL_API_PRIVATE_KEY`
- `AUTH_ZITADEL_GOOGLE_IDP_ID`
- `AUTH_ZITADEL_DEFAULT_ROLE_KEYS`
- `AUTH_ZITADEL_PROJECT_ID`
- `AUTH_ZITADEL_NOTIFICATION_SIGNING_KEY`
- `AUTH_ZITADEL_NOTIFICATION_MAX_AGE_SECONDS`
- `RESEND_API_KEY`
- `RESEND_FROM_EMAIL`
- `RESEND_FROM_NAME`
- `TURNSTILE_ENABLED`
- `TURNSTILE_SITE_KEY`
- `TURNSTILE_SECRET_KEY`

## Frontend

### Public-safe

- `VITE_API_BASE_URL`: normally `http://localhost:5113`
- `VITE_AUTH_SESSION_TRANSPORT`: `bearer` for local-safe work
- `VITE_DISABLE_PANDA_EASTER_EGG`
- `VITE_PANDA_END_STYLE`
- `VITE_DEV_HOST`
- `VITE_DEV_HTTPS_CERT`
- `VITE_DEV_HTTPS_KEY`
- `PUBLIC_ORIGIN`
- `PUBLIC_DOCS_URL`

### Optional maintainer-backed auth integration

- `VITE_ZITADEL_AUTHORITY`
- `VITE_ZITADEL_CLIENT_ID`
- `VITE_ZITADEL_REDIRECT_URI`
- `VITE_ZITADEL_SCOPES`
- `VITE_ZITADEL_RESOURCE`
- `VITE_ZITADEL_AUDIENCE`
- `VITE_ZITADEL_AUTHORIZATION_ENDPOINT`
- `VITE_ZITADEL_TOKEN_ENDPOINT`

Outside contributors usually do not need the Zitadel values unless they are working on that exact auth flow.

## ETL

### Public-safe

- `SKIP_MAIN_DB_REFLECTION` when reusing backend imports in tests or tooling

### Maintainer-only or secret

- `MARIADB_USER`
- `MARIADB_PASSWORD`
- `MARIADB_HOST`
- `MARIADB_DATABASE`
- `MARIADB_PORT`
- `OPENAI_API_KEY`
- `VOYAGE_API_KEY`

Most ETL paths are operationally maintainer-gated because they depend on live data or paid model APIs.

## Emails

### Public-safe

- none required for local preview of the existing templates

### Maintainer-only or secret

- `RESEND_API_KEY`
- `npm_config_template_id` when syncing a specific template

## Bulk

### Maintainer-only or secret

- `MARIADB_HOST`
- `MARIADB_PORT`
- `MARIADB_USER`
- `MARIADB_PASSWORD`
- `MARIADB_DATABASE`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`
- `MYDUMPER_THREADS`
- `MYDUMPER_ROWS`
- `MYLOADER_THREADS`

`bulk/` is open-source code but not part of the public-safe default workflow.

## DB

### Maintainer-only or secret

- `MARIADB_ROOT_PASSWORD`
- `MARIADB_PASSWORD`
- `MARIADB_HOST`
- `R2_ACCESS_KEY_ID`
- `R2_SECRET_ACCESS_KEY`

The `db/` directory is deployment and operations oriented.
