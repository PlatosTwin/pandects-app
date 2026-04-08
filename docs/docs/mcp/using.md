---
id: using
title: Using MCP
description: Understand what Pandects MCP can do and how to work with it effectively in your client.
sidebar_position: 1
---

# Using MCP

Pandects exposes a read-only MCP server at `https://api.pandects.org/mcp`.

Use it inside MCP clients such as Codex or Claude Code to discover agreements, search clauses, inspect sections, fetch agreement text, and pull supporting taxonomy or reference data without leaving your workspace.

## What You Can Do With Pandects MCP

Pandects MCP currently supports:

- Agreement discovery: `search_agreements`, `list_agreements`, `get_agreement`
- Section research: `search_sections`, `list_agreement_sections`, `get_section`
- Tax clause research: `get_agreement_tax_clauses`, `get_section_tax_clauses`
- Research bootstrap: `list_filter_options`, `get_clause_taxonomy`, `get_tax_clause_taxonomy`
- Reference and context: `get_counsel_catalog`, `get_naics_catalog`, `get_agreements_summary`, `get_agreement_trends`

## Typical Workflows

In practice, that means you can ask your client to do things like:

- Find change-of-recommendation sections from 2023 public deals
- Search for sections by clause family, filing year, target, or acquirer
- Find the right agreement from a company name search, then list its sections
- Pull a specific agreement by UUID and inspect the text
- Inspect one section directly from a section UUID
- Fetch extracted tax clauses for an agreement or section
- Load taxonomy and filter catalogs before running a more precise search

## How To Think About The Tools

`search_agreements` is the best starting point when you know a company name, partial name, or year and need to find the right agreement first.

`search_sections` is the best starting point when you are looking for language patterns or a category of clauses across many agreements.

`list_agreements` is useful when you already know the deal-level filters you care about and want a clean list of matching agreements.

`list_agreement_sections` is the follow-up tool when you have one agreement and want to navigate inside it before opening a specific section.

`get_section` is the direct drilldown tool for section-level inspection.

`get_agreement` is the document-level retrieval tool when you want the full agreement.

`list_filter_options` and the taxonomy tools are the bootstrap layer for agents that need valid values before issuing precise searches.

## Recommended Research Loop

For most workflows, use MCP in this order:

1. `search_agreements` or `search_sections`
2. `list_agreement_sections` if you need to navigate within one agreement
3. `get_section` for section-level inspection
4. `get_agreement` if you need full agreement context
5. `get_agreement_tax_clauses` or `get_section_tax_clauses` for tax-focused work

## What MCP Is Best For

Pandects MCP is strongest when you want your coding or research client to:

- Explore a set of agreements interactively
- Gather examples before writing or refining analysis
- Move from high-level search to specific source documents
- Keep research and drafting in the same tool session

## Access Model

Use your normal Pandects account when the client opens the browser sign-in flow.

You should not need to manually copy tokens or pull API keys or anything else from the account page. The MCP server does not use API keys.

## Next Step

To connect Pandects in a supported client, continue to [Setup](./setup).
