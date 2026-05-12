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

## Data provenance

Every response includes the sha256 hash of the public database dump the data was sourced from. This lets you record exactly which snapshot your queries came from.

**Response header** (always present):

```
X-Pandects-Dump-Hash: 3a7f9b2c...
```

**Response body** (present by default on `/v1/sections`, `/v1/agreements`, and `/v1/agreements/search`):

```json
{
  "results": [...],
  "dump_version": {
    "hash": "3a7f9b2c...",
    "dump_ts": "2025-03-01 04:00:00"
  }
}
```

To omit `dump_version` from the JSON body, pass `?include_dump=false`. The header is always set regardless.

## For LLMs and agents

Plaintext, machine-readable mirrors of these docs are published for use by coding assistants and agents:

- [`/llms.txt`](pathname:///llms.txt) — index of guides and endpoints in the [llms.txt](https://llmstxt.org/) format
- [`/llms-full.txt`](pathname:///llms-full.txt) — concatenated full documentation in one file
- [`/openapi.yaml`](pathname:///openapi.yaml) — canonical OpenAPI 3 spec
- [`/llms/pandects/<operationId>.md`](pathname:///llms/pandects/searchAgreements.md) — one plain-markdown page per endpoint

## Next step

Continue with `Request Patterns` for common query combinations and endpoint workflows.
