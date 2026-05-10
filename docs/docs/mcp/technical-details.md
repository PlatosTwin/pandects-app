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
- `list_agreement_sections_batch`
- `get_section`
- `get_section_snippet`
- `get_section_snippets_batch`
- `get_sections_batch`

Use these when the client needs to search clause language across the corpus, navigate sections inside one agreement, inspect a known section directly, or extract a shorter plain-text excerpt from one section. `list_agreement_sections_batch`, `get_section_snippets_batch`, and `get_sections_batch` accept a list of agreement or section UUIDs and return results in a single call, reducing round-trips for multi-agreement workflows. `get_sections_batch` returns full section XML (capped by default at 10 000 characters per section); `get_section_snippets_batch` returns focused plain-text excerpts and is the right choice when the full XML is not needed.

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
- `list_agreement_sections_batch`
- `get_agreement`
- `get_section`
- `get_section_snippet`
- `get_section_snippets_batch`
- `get_sections_batch`
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
- `search_agreements` is the discovery-oriented agreement search; accepts a `standard_id` list to filter to agreements that contain at least one section tagged with any of the given taxonomy ids
- `list_agreements` is the exact-filter, cursor-based agreement listing surface; also accepts `standard_id` for taxonomy-based agreement filtering
- `list_agreement_sections` is an MCP convenience tool for within-agreement navigation
- `search_sections` is a clause-language retrieval surface, not a normalized document-facts surface
- `search_sections` exposes `count_mode` and returns `count_metadata` plus `interpretation` so clients can tell when totals are exact versus estimated and when taxonomy is acting as a proxy
- `search_agreements` returns exact totals today and also includes `count_metadata` plus `interpretation`
- `suggest_clause_families` exists to bridge plain-English concepts to taxonomy ids and now reports fit/confidence metadata so clients can distinguish canonical matches from broader proxies
- `get_section_snippet` is a focused reading aid, not a replacement for `get_section` or a canonical extracted-facts surface
- `get_section_snippets_batch` and `list_agreement_sections_batch` accept arrays of UUIDs and collapse multiple single-item calls into one round-trip; use them when a workflow would otherwise fan out across many agreements or sections
- `get_sections_batch` fetches full section XML for up to 10 sections in one call; XML is capped at `max_xml_chars` per section (default 10 000, range 500–20 000) to prevent context overload; when a section is truncated the result includes `xml_truncated: true`; pass `max_xml_chars: null` only if uncapped XML is explicitly needed
- `search_sections` results include `filing_date` and `transaction_price_total` inline on every result without needing to request them via `metadata`
- `get_section_snippet`, `get_section_snippets_batch`, and `get_sections_batch` all include a `monetary_values` list — dollar amounts and value expressions extracted from the section text — so clients can surface deal economics without parsing XML
- `search_sections` and `search_agreements` both accept `filed_after` and `filed_before` (ISO 8601 date strings, `YYYY-MM-DD`) for sub-year filing-date precision; `year`/`year_min`/`year_max` filter on the agreement year; `filed_after`/`filed_before` filter on the exact filing date
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
