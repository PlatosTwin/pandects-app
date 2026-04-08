---
id: using
title: Using MCP
description: Understand what Pandects MCP can do and how to work with it effectively in your client.
sidebar_position: 1
---

# Using MCP

Pandects exposes a read-only MCP server at `https://api.pandects.org/mcp`.

Use it inside MCP clients such as Codex or Claude Code to search sections, browse agreements, and inspect agreement text without leaving your workspace.

## What You Can Do With Pandects MCP

Pandects MCP currently supports three core actions:

- `search_sections`: Search merger agreement sections with structured filters
- `list_agreements`: List agreements with filters and pagination
- `get_agreement`: Fetch a specific agreement by UUID

## Typical Workflows

In practice, that means you can ask your client to do things like:

- Find change-of-recommendation sections from 2023 public deals
- List recent agreements for a target or acquirer
- Pull a specific agreement by UUID and inspect the text
- Search for sections by clause family, filing year, target, or acquirer
- Narrow from a broad section search into one agreement for closer review

## How To Think About The Tools

`search_sections` is the best starting point when you are looking for language patterns or a category of clauses across many agreements.

`list_agreements` is useful when you already know the deal-level filters you care about and want a clean list of matching agreements.

`get_agreement` is the follow-up tool when you want to inspect one agreement directly.

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
