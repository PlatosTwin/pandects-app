# Proposals not implemented on this branch

Each item below changes what existing users or their AI clients get, or changes legal text, so
it needs the maintainer's decision. Each has a recommendation and, where useful, ready text.

## P1. Make feedback user-initiated

**Problem.** The server `instructions` (`backend/mcp/tools/instructions.py`, last paragraph),
the `submit_feedback` description, the `feedback_note` in `get_server_capabilities`
(`backend/mcp/tools/__init__.py`) and the `submit_feedback` result message all ask the model to
send feedback on its own, during and at the end of a session. Both programs check for this:

- Anthropic rejects descriptions that "tell Claude to behave in ways unrelated to the tool's
  function", and the portal's Compliance step has an acknowledgment on conversation data
  collection.
- OpenAI asks plugins to "respect user intent" and not to collect unnecessary data or
  reconstruct chat logs.

A note that the model writes on its own, from the user's session, to the server owner is the
pattern both reviews look for.

**Recommendation.** Keep the tool. Change the wording so the model offers feedback when it
fits, and sends it when the user agrees or asks. Expect fewer feedback rows.

Ready text for the last paragraph of `SERVER_INSTRUCTIONS`:

> If the user wants to report a problem with these tools or the data, or agrees when you offer,
> send it with submit_feedback; the maintainer reads every report.

Ready text for the first part of the `submit_feedback` description:

> Send a short feedback note about this MCP server to its maintainer: bugs, workflow friction,
> missing capabilities, data-quality issues, documentation gaps, or what worked well. The note
> is stored and read by the maintainer. Send it when the user asks to report something, or
> after the user agrees to your offer to report a problem you hit. The most useful notes are
> concrete: name the tool, the arguments you passed (context), what you expected, and what
> happened.

Selection hint: "Use when the user asks to report friction, wrong or missing data, or good
behavior, or agrees to your offer to report it." Result message: "Feedback recorded. Thank
you." The `feedback_note` in capabilities would match. The snapshot fixtures and the
instructions test change with it.

## P2. Privacy policy section for the MCP server

**Problem.** Both programs require a privacy policy that covers the connector's data. The
current policy (`frontend/public/legal/privacy-policy.md`) covers accounts, server logs and API
usage in general terms but does not name the MCP server, the per-user tool usage records, or
feedback notes.

**Recommendation.** Add a short subsection. Draft (the maintainer must check it against what
the service really does and against the retention practice):

> ### E. MCP server
>
> When you connect an AI client (such as Claude, ChatGPT, Codex or Claude Code) to the
> Pandects MCP server at `https://api.pandects.org/mcp`, you sign in with your Pandects account
> and the client receives an access token. We record, for each hour, which tools your account
> used, through which client registration, whether each call succeeded, how long it took and
> the request size. We use these records to operate the service, prevent abuse and enforce
> rate limits. We do not store the content of your conversations with your AI client. The
> arguments your client sends with a tool call (for example a search phrase) are processed to
> answer the call and are not kept in the usage records. [Maintainer: confirm whether error
> logs or the hosting provider's logs can hold arguments, and for how long.]
>
> If you or your AI client use the `submit_feedback` tool, we store the note (summary, detail,
> suggestions and a short context object) with your account ID, the client registration and
> the token's scopes. Only the maintainer reads feedback notes. Do not include personal data
> in feedback.
>
> We do not sell this information or share it with the AI client provider beyond the tool
> results your client requests. You can disconnect a client at any time in that client, and
> you can ask us to delete your account and its MCP records at the contact address below.

Also add "MCP usage records" to section 6 (Data Retention) with the real retention period.

## P3. Client ID Metadata Document (CIMD) support

Anthropic: "For servers expecting high traffic from the directory, prefer CIMD or
`oauth_anthropic_creds` over DCR," because DCR registers a new client on every fresh
connection. OpenAI also prefers CIMD. The server supports DCR only.

**Recommendation.** Not needed to submit. Revisit if the `oauth clients` table grows fast
after listing. CIMD is an auth change: the server would fetch and validate the client's
metadata URL (`https://claude.ai/oauth/claude-code-client-metadata`,
`https://chatgpt.com/oauth/client.json`) and advertise
`"client_id_metadata_document_supported": true`.

## P4. `destructiveHint` on `submit_feedback`

OpenAI: `true` for "irreversible sends"; `false` only for "additive writes with no destructive
or irreversible effects". The feedback note is additive and harms nothing, but the user cannot
take it back.

**Recommendation.** Keep `false` and explain it in the review form if a finding comes back
("additive note to the developer; it changes no user data"). Setting `true` would make Claude
ask for confirmation on every call. If P1 is not adopted, consider `true` as the safer
answer for review.

## P5. Optional fields in protected resource metadata

OpenAI lists optional `resource_documentation`, `resource_policy_uri` and `resource_tos_uri` in
`/.well-known/oauth-protected-resource`.

**Recommendation.** Low value. Add them only with the next auth change:
`https://docs.pandects.org/docs/mcp/setup`, `https://pandects.org/privacy-policy`,
`https://pandects.org/terms`.

## P6. Pathed protected resource metadata

`GET /.well-known/oauth-protected-resource/mcp` returns 404. Claude reads the
`resource_metadata` pointer in the `401` header first, so this is only a fallback path.

**Recommendation.** No action unless a client reports a discovery failure.
