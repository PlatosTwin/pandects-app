---
id: using
title: Using MCP
description: Understand what The Pandects MCP can do and how to work with it effectively in your client.
sidebar_position: 1
---

# Using MCP

The Pandects MCP is available at `https://api.pandects.org/mcp`.

Use it inside MCP clients such as Codex or Claude Code to work with merger agreements in plain language.

You can ask your client to find agreements, compare clause language, inspect specific sections, pull tax-related provisions, and use Pandects reference data while you work.

## What MCP Is Good For

The Pandects MCP is most useful when you want your client to:

- Find the right agreement when you know a company name but not the document you need
- Compare clause language across many agreements
- Open one section or one agreement for closer reading
- Pull tax-related language from an agreement
- Use Pandects taxonomy, counsel, industry, and summary data while researching

## Example Prompts

The Pandects MCP server allows you to interact with Pandects APIs using natural language.

Examples:

- “Find the merger agreement for Target A and show me the termination and fiduciary-out sections.”
- “Compare change-of-recommendation language across 2023 public-target deals.”
- “Find agreements where Skadden represented the acquirer.”
- “Pull the tax clauses from this agreement and summarize who bears transfer taxes.”
- “Show me the section text for this section UUID and explain how it fits into the agreement.”
- “What clause taxonomy categories should I use for fiduciary-out and no-shop searches?”
- “Give me the valid Pandects filter values for target counsel and target industry.”

## What Your Client Can Reach Through MCP

Behind those prompts, The Pandects MCP gives your client access to:

- Agreement search and retrieval
- Section search and section-level inspection
- Tax clause extraction results
- Filter catalogs for companies, counsel, and industries
- Clause taxonomy and tax-clause taxonomy
- Counsel and NAICS reference data
- High-level corpus summaries and trend data

You generally should not need to think about exact tool names or tool order unless you are debugging or building a more structured workflow on top of MCP.

## How to Think About MCP

The Pandects MCP is designed for humans using LLM clients. That means:

- You ask for the research task in normal language
- The client decides which Pandects tools to call
- You review the result, refine the request, and keep going
- You should not need to manually fetch tokens, API keys, or document IDs from the website just to make MCP work

## Next Step

To connect Pandects in a supported client, continue to [Setup](./setup). If you want the technical tool surface, see [Technical Details](./technical-details).
