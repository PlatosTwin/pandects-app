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
- Tax clause retrieval
- Filter and taxonomy lookup
- Counsel, NAICS, summary, and trend reference data

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

Use these when the client needs to search clause language across the corpus, navigate sections inside one agreement, or inspect a known section directly.

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

Use these when the client needs canonical lookup data or corpus-level context.

## Design Notes

- The server is read-only
- Clients should typically choose the right tools automatically
- `search_agreements` is the discovery-oriented agreement search
- `list_agreements` is the exact-filter, cursor-based agreement listing surface
- `list_agreement_sections` is an MCP convenience tool for within-agreement navigation
- `get_agreement` preserves the current redaction and full-text access behavior

## Authentication

- MCP uses normal Pandects account login
- MCP does not use Pandects API keys
- `codex mcp add` only registers the server; in Codex, `codex mcp login <name>` starts OAuth
- Clients that use OAuth for remote MCP may require dynamic client registration support from the Pandects identity provider

## Related Pages

- [Using MCP](./using)
- [Setup](./setup)
