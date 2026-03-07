---
id: getting-started
title: Getting Started
description: Base URL, response format, and first API calls.
sidebar_position: 1
---

# Getting Started

Pandects exposes a public, read-only REST API for searching agreements and reading full agreement or section text.

## Base URL

```bash
https://api.pandects.org
```

All API routes are versioned under `/v1`.

## First request

```bash
curl "https://api.pandects.org/v1/sections?page=1&page_size=10"
```

The API returns JSON and standard HTTP status codes.

## Response conventions

- `200`: request succeeded
- `422`: invalid request shape or unsupported parameters
- `default`: other server-side error envelopes

## Next step

Continue with `Request Patterns` for common query combinations and endpoint workflows.
