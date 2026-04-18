---
id: technical-details
title: Technical Details
description: High-level technical reference for The Pandects MCP server and the tool surface it exposes.
sidebar_position: 3
---

# Technical Details

The Pandects MCP is a read-only remote MCP server at `https://api.pandects.org/mcp`.

It is designed for human users working through LLM clients, but this page describes the technical surface available behind the scenes.

## High-Level Capability Areas

The current MCP surface covers:

- Agreement discovery and retrieval
- Section search and section listing within one agreement
- Section-level retrieval
- Concept-to-taxonomy mapping and focused snippet extraction
- Tax clause retrieval
- Filter and taxonomy lookup
- Counsel, NAICS, summary, trend, and server-introspection reference data

## Tool Groups

### Agreement Discovery And Retrieval

- `search_agreements`
- `list_agreements`
- `get_agreement`

Use these when the client needs to find the right agreement first or fetch one agreement directly.

### Section Research

- `search_sections`
- `list_agreement_sections`
- `get_section`
- `get_section_snippet`

Use these when the client needs to search clause language across the corpus, navigate sections inside one agreement, inspect a known section directly, or extract a shorter plain-text excerpt from one section.

### Discovery Helpers

- `suggest_clause_families`

Use this when the client knows the business or legal concept but does not know the right taxonomy `standard_id` yet. The tool returns ranked clause-family candidates with their taxonomy paths and matched terms.
Each match also reports whether it is a canonical fit, a proxy, or a broader semantic match.

### Tax Clause Research

- `get_agreement_tax_clauses`
- `get_section_tax_clauses`

Use these when the task is specifically about extracted tax-module clauses rather than the full agreement or section text.

### Research Bootstrap

- `list_filter_options`
- `get_clause_taxonomy`
- `get_tax_clause_taxonomy`

Use these when the client needs valid structured inputs before searching.

### Reference And Context

- `get_counsel_catalog`
- `get_naics_catalog`
- `get_agreements_summary`
- `get_agreement_trends`
- `get_server_capabilities`
- `get_server_metrics`

Use these when the client needs canonical lookup data, corpus-level context, or MCP introspection metadata.

## Current Tool List

The current MCP tools are:

- `search_agreements`
- `search_sections`
- `list_agreements`
- `list_agreement_sections`
- `get_agreement`
- `get_section`
- `get_section_snippet`
- `get_agreement_tax_clauses`
- `get_section_tax_clauses`
- `list_filter_options`
- `suggest_clause_families`
- `get_server_metrics`
- `get_server_capabilities`
- `get_clause_taxonomy`
- `get_tax_clause_taxonomy`
- `get_counsel_catalog`
- `get_naics_catalog`
- `get_agreements_summary`
- `get_agreement_trends`

## Design Notes

- The server is read-only
- Clients should typically choose the right tools automatically
- `search_agreements` is the discovery-oriented agreement search
- `list_agreements` is the exact-filter, cursor-based agreement listing surface
- `list_agreement_sections` is an MCP convenience tool for within-agreement navigation
- `search_sections` is a clause-language retrieval surface, not a normalized document-facts surface
- `search_sections` exposes `count_mode` and returns `count_metadata` plus `interpretation` so clients can tell when totals are exact versus estimated and when taxonomy is acting as a proxy
- `search_agreements` returns exact totals today and also includes `count_metadata` plus `interpretation`
- `suggest_clause_families` exists to bridge plain-English concepts to taxonomy ids and now reports fit/confidence metadata so clients can distinguish canonical matches from broader proxies
- `get_section_snippet` is a focused reading aid, not a replacement for `get_section` or a canonical extracted-facts surface
- `get_agreement` preserves the current redaction and full-text access behavior
- `get_server_capabilities` is the main machine-readable semantics surface; it includes auth guidance, field inventory, concept notes, and negative guidance about when not to use a tool
- The server exposes a small set of MCP resources (`pandects://capabilities`, `pandects://auth-help`) that mirror `get_server_capabilities` for clients that prefer the `resources/read` primitive over calling a tool
- The server exposes curated MCP prompts (`compare_agreements`, `clause_family_survey`, `deal_trend_brief`) as research templates; they orchestrate the primitive retrieval tools rather than introducing new functionality

## Transport

- `POST /mcp` is the primary JSON-RPC endpoint. It supports content negotiation: clients that advertise `Accept: text/event-stream` receive an SSE-framed response; clients that prefer `application/json` receive a plain JSON body. This matches the Streamable HTTP behaviour required by Claude Code.
- `GET /mcp` returns an SSE retry probe for clients that opportunistically open a server-to-client stream.
- `DELETE /mcp` is accepted as an authenticated session-termination signal and returns `204`.
- `initialize` responses carry an `Mcp-Session-Id` header. The server is stateless, so the id is informational — clients are not required to echo it, but Claude Code does.
- Every response carries an `MCP-Protocol-Version` header echoing the negotiated protocol version.
- Advertised server capabilities: `tools`, `resources` (listChanged=false, subscribe=false), `prompts` (listChanged=false), and `logging` (`logging/setLevel` is accepted as a no-op).

### Progress notifications

When a `tools/call` request includes `params._meta.progressToken` **and** the client advertises `Accept: text/event-stream`, the server returns a multi-event SSE stream:

1. `notifications/progress` with `progress=0`, `total=1`, and a `Starting <tool>` message
2. `notifications/progress` with `progress=1`, `total=1`, and a `<tool> complete` message
3. The final `tools/call` JSON-RPC result (or error)

This keeps intermediary proxies and client UIs aware of in-flight work on long calls. Clients that do not set a progress token, or do not accept SSE, receive the usual single-response behaviour.

### OAuth discovery and Dynamic Client Registration

The server is protected by an embedded OAuth authorization server whose issuer lives under `/v1/auth/oauth`. To make OAuth discovery work with clients that implement RFC 8414 strictly (including Claude Code), authorization-server metadata is exposed at three locations:

- `GET /.well-known/oauth-authorization-server` — host-root fallback
- `GET /.well-known/oauth-authorization-server/v1/auth/oauth` — RFC 8414 host-root + issuer-path form
- `GET /v1/auth/oauth/.well-known/oauth-authorization-server` — issuer-prefixed form (original)

`GET /.well-known/openid-configuration` is also exposed at the host root for OIDC-leaning clients.

The metadata document advertises `registration_endpoint` (`/v1/auth/oauth/register`), so compliant clients can self-register via Dynamic Client Registration (RFC 7591) without a manual out-of-band step. Only public, PKCE (S256), authorization-code clients are supported.

## Authentication

- MCP uses normal Pandects account login
- MCP does not use Pandects API keys
- `codex mcp add` only registers the server; in Codex, `codex mcp login <name>` starts OAuth
- Auth failures return structured remediation metadata so clients can distinguish missing-token, expired-token, unverified-account, and unlinked-subject cases

## Related Pages

- [Using MCP](./using)
- [Setup](./setup)
