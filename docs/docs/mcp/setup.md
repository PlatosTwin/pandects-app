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

:::note Sign-in needs an interactive browser
Authentication is a browser sign-in, not an API key. A non-interactive or headless agent session cannot complete it on its own — a person has to finish the browser step. If you are driving an automated agent, hand off the sign-in to a human, then let the agent use the tools once the server is connected.
:::

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

Recent Codex versions keep MCP tools out of the model's context until its `tool_search` tool loads them. Codex shows the model the Pandects server instructions, which map each research task to the tools for it, so you rarely need to name a tool. If Codex does not pick up the right one, name it in your prompt by its bare name, for example "use search_sections"; Codex's search does not currently match the prefixed `mcp__pandects__search_sections` form.

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

Claude Code loads MCP tools on demand through tool search: at the start of a session Claude sees only the tool names and the server instructions, and loads a tool's full definition when a task needs it. If you use Pandects in nearly every session and want its tools loaded up front, set `"alwaysLoad": true` on the `pandects` entry in your MCP configuration; this costs context in every conversation.

## Supported Clients Right Now

Pandects currently documents setup for:

- Codex
- Claude Code

Other MCP clients may work if they support remote HTTP MCP servers with browser-based authentication.

## Troubleshooting

If setup does not complete cleanly:

- Make sure you are signing in with the Pandects account you intend to use
- Make sure you are connecting to `https://api.pandects.org/mcp`
- In Codex, run `codex mcp login pandects` after `codex mcp add ...`
- Use browser sign-in when the client asks for authentication
- Do not use a Pandects API key for MCP
- If the client already has a stale failed connection saved, remove the server and add it again

If authentication succeeds but tool calls still fail, reconnect the server and retry the browser sign-in flow once before investigating anything more exotic.

## Related Links

- [OpenAI: Docs MCP quickstart for Codex](https://developers.openai.com/learn/docs-mcp)
- [OpenAI: MCP docs](https://platform.openai.com/docs/mcp)
- [Anthropic: Connect Claude Code to tools via MCP](https://docs.anthropic.com/en/docs/claude-code/mcp)
