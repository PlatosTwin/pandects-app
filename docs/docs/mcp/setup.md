---
id: setup
title: Setup
description: Connect The Pandects MCP to Codex or Claude Code.
sidebar_position: 2
---

# Setup

The Pandects MCP is available at `https://api.pandects.org/mcp`.

## Before You Start

You need only two things:

1. A Pandects account
2. An MCP client that supports a remote HTTP server and browser sign-in

## Server URL

Add this server in your MCP client:

```text
https://api.pandects.org/mcp
```

## Quick Setup

The normal flow is:

1. Add the Pandects MCP server in your client
2. Start the client's MCP auth flow explicitly
3. When your browser opens, sign in with your Pandects account
4. Return to the client and start using Pandects tools

## Codex

Add the server:

```bash
codex mcp add pandects --url https://api.pandects.org/mcp
```

Then start the OAuth flow explicitly:

```bash
codex mcp login pandects
```

`codex mcp add` only saves the server configuration. It does not itself open the browser sign-in flow.

Optional check:

```bash
codex mcp list
```

If Codex shows `Auth Unsupported`, or `codex mcp login pandects` fails with `Dynamic client registration not supported`, the Pandects identity provider is not yet advertising the OAuth dynamic client registration support Codex expects for remote MCP login. In that state, browser auth will not start successfully in Codex.

Fallback for Codex right now:

1. Sign in to your Pandects account on the website
2. Open Account and generate an MCP bearer token
3. Export that token in your shell
4. Add the MCP server with a bearer-token env var

```bash
export PANDECTS_MCP_BEARER_TOKEN='<token-from-account-page>'
codex mcp add pandects --url https://api.pandects.org/mcp --bearer-token-env-var PANDECTS_MCP_BEARER_TOKEN
```

## Claude Code

Add the server:

```bash
claude mcp add --transport http pandects https://api.pandects.org/mcp
```

Then open MCP inside Claude Code:

```text
/mcp
```

Then authenticate the `pandects` server inside the MCP UI and finish the browser sign-in flow.

## Supported Clients Right Now

Pandects currently documents setup for:

- Codex
- Claude Code

Other MCP clients may work if they support remote HTTP MCP servers with browser-based authentication. Codex additionally expects OAuth dynamic client registration for remote MCP login.

## Troubleshooting

If setup does not complete cleanly:

- Make sure you are signing in with the Pandects account you intend to use
- Make sure you are connecting to `https://api.pandects.org/mcp`
- In Codex, run `codex mcp login pandects` after `codex mcp add ...`
- Use browser sign-in when the client asks for authentication
- Do not use a Pandects API key for MCP
- If the client already has a stale failed connection saved, remove the server and add it again
- If Codex reports `Dynamic client registration not supported`, the issue is on the Pandects auth-server side, not in your local Codex setup

If authentication succeeds but tool calls still fail, reconnect the server and retry the browser sign-in flow once before investigating anything more exotic.

## Related Links

- [OpenAI: Docs MCP quickstart for Codex](https://developers.openai.com/learn/docs-mcp)
- [OpenAI: MCP docs](https://platform.openai.com/docs/mcp)
- [Anthropic: Connect Claude Code to tools via MCP](https://docs.anthropic.com/en/docs/claude-code/mcp)
