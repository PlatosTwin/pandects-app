---
id: setup
title: MCP Setup
description: Connect Pandects to an MCP client using your normal Pandects account.
sidebar_position: 1
---

# MCP Setup

Pandects exposes a read-only MCP server at `https://api.pandects.org/mcp`.

The important user-facing point is simple:

- use your normal Pandects account
- sign in when your MCP client opens the browser auth flow
- you do **not** manually copy a bearer token out of the Pandects website

Pandects uses the same ZITADEL-backed identity for both website sign-in and MCP.

## What You Actually Need To Do

1. Make sure you have a normal Pandects account.
2. Add the Pandects MCP server to your MCP client.
3. When the client prompts for authentication, sign into Pandects in the browser.
4. Return to the client and start using the tools.

That is the intended setup flow.

## Do I Need Anything From The Account Page?

Not to manually fetch a raw bearer token.

Today, the account page is mainly for:

- managing `X-API-Key` credentials for the REST API
- viewing usage

Those API keys are **not** used for MCP.

For MCP, the client should handle OAuth and obtain the bearer token for you after you sign in with your Pandects account.

## How MCP Auth Maps To Pandects Auth

Under the hood:

- Pandects website sign-in and MCP both use the same ZITADEL identity
- when you first sign in successfully, Pandects links that external identity to your local Pandects account
- MCP accepts bearer tokens for that linked, verified Pandects user

So from the user perspective, the rule is:

use the same Pandects login you already use on the website.

## Server URL

Use this MCP server URL in your client:

```text
https://api.pandects.org/mcp
```

The protected-resource metadata lives at:

```text
https://api.pandects.org/.well-known/oauth-protected-resource
```

OAuth-aware MCP clients should discover what they need from that metadata automatically.

## Current Tools

The current MCP tool surface is:

- `search_sections`
- `list_agreements`
- `get_agreement`

Required scopes are handled by the OAuth login flow. If your client signs in successfully but a tool still fails, the most likely issue is that the token was minted without the right MCP scopes.

## Codex

Add the server:

```bash
codex mcp add pandects --url https://api.pandects.org/mcp
codex mcp list
```

Then use the MCP auth flow in Codex and sign in with your Pandects account when the browser opens.

## Claude Code

Add the server:

```bash
claude mcp add --transport http pandects https://api.pandects.org/mcp
```

Then run:

```text
/mcp
```

Claude Code should prompt you to authenticate the remote MCP server. Sign in with your Pandects account in the browser flow it opens.

## ChatGPT

ChatGPT is the awkward case right now.

OpenAI currently documents ChatGPT custom connector compatibility around a `search` + `fetch` tool shape. The current Pandects MCP server does not expose that compatibility surface yet, so it should not be documented as a polished ChatGPT connector today.

Right now, Pandects MCP is best treated as:

- ready for clients like Codex and Claude Code
- not yet packaged for first-class ChatGPT connector UX

## Troubleshooting

If MCP login succeeds but requests still fail:

- confirm you signed in with the same Pandects account you expect to use
- confirm that account is verified
- do not use a Pandects API key on `/mcp`
- do not expect an existing website session cookie to act as the MCP token

## Product Gap

The product should eventually make this easier from the account page with something like:

- `Connect to Codex`
- `Connect to Claude Code`
- copy-paste setup snippets
- a short explanation that MCP uses your normal Pandects login, not API keys

That UX is not exposed in the account page yet, even though the underlying auth model already supports the same account identity for both website use and MCP.

## References

- [OpenAI: Docs MCP quickstart for Codex](https://developers.openai.com/learn/docs-mcp)
- [OpenAI: MCP docs](https://platform.openai.com/docs/mcp)
- [OpenAI Help: Connectors in ChatGPT](https://help.openai.com/en/articles/11487775-connectors-in-chatgpt)
- [Anthropic: Connect Claude Code to tools via MCP](https://docs.anthropic.com/en/docs/claude-code/mcp)
