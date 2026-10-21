---
id: error-model
title: Error Models
description: The error envelope, its codes, and how to handle them.
sidebar_position: 3
---

# Error Models

Every non-`2xx` response from a `/v1` endpoint has the same JSON body:

```json
{
  "error": "validation_error",
  "message": "page_size: Not a valid integer."
}
```

- `error` is a stable, snake_case machine code. Branch on it (together with the HTTP status).
- `message` is human-readable text for logs and UIs. Its wording can change; do not parse it.

## Codes

| Status | `error` | Meaning |
| --- | --- | --- |
| `400` | `bad_request` | The request is malformed or a parameter value is not allowed. |
| `400` | `validation_error` | A JSON body or query string failed validation before it reached the endpoint. |
| `401` | `unauthorized` | The endpoint needs a signed-in session or a valid API key. |
| `403` | `forbidden` | The caller is authenticated but may not perform this action. |
| `404` | `not_found` | The resource (for example an agreement or section UUID) does not exist or is not public. |
| `405` | `method_not_allowed` | The path exists but not for this HTTP method. |
| `409` | `conflict` | The request conflicts with existing state (for example a duplicate name). |
| `422` | `validation_error` | At least one query or path parameter failed validation. `message` names the field. |
| `429` | `rate_limited` | Too many requests. Wait for the `Retry-After` header (seconds) before retrying. |
| `500` | `internal_error` | Unexpected server error. |
| `503` | `service_unavailable` | A backing service (usually the database) is temporarily unavailable. |

Any other status uses the snake_case form of its HTTP reason phrase (for example `410` is `gone`).

Some endpoints return a more specific code for a condition the status alone does not describe:

- Sign-in and account endpoints: `captcha_required`, `captcha_failed`, `legal_required`, `auth_failed`.
- The OAuth authorization endpoint for MCP clients redirects with RFC 6749 codes in the redirect query (`invalid_request`, `invalid_scope`, `unsupported_response_type`). The OAuth token and client-registration endpoints use the generic codes above.

The MCP endpoint (`/mcp`) does not use this envelope; it returns JSON-RPC 2.0 error objects (see the MCP technical details).

## Client recommendations

- Validate parameter types before sending requests, and send repeated list parameters using repeated query keys (form style). Retry a `422` only after correcting input.
- Retry `429`, `500`, and `503` with bounded backoff and jitter; honor `Retry-After` on `429`.
- Do not retry other `4xx` responses unchanged.
- Log the endpoint, status, `error`, and `message`.
- Fail closed for downstream writes if agreement or section payloads are incomplete.
