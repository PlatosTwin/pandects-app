---
id: error-model
title: Error Model
description: How to handle validation and server errors.
---

# Error Model

Treat non-`2xx` responses as structured errors and log request IDs when available.

## Validation errors (`422`)

A `422` means at least one query/path parameter failed validation.

Client recommendations:

- Validate parameter types before sending requests.
- Send repeated list parameters using repeated query keys (form style).
- Retry only after correcting input.

## Other errors

Default error responses can indicate transient failures.

Client recommendations:

- Use bounded retries with jitter.
- Surface endpoint, status, and high-level error details in logs.
- Fail closed for downstream writes if agreement or section payloads are incomplete.
