# Pandects MCP: audit against the marketplace rules

Audit date: 2026-11-17. Server audited: `https://api.pandects.org/mcp` (live) and the code on
branch `claude/marketplace-listings`. Rules were read on 2026-11-17 from the pages listed in
[README.md](README.md#sources). None of those pages shows a "last updated" date.

Status values: **MET**, **NOT MET**, **UNCLEAR** (cannot be checked without the maintainer's
account or a vendor decision). "This PR" means the item was closed on this branch.

## Facts about the server that the audit uses

| Fact | Evidence |
| --- | --- |
| Transport is Streamable HTTP: `POST /mcp` answers JSON-RPC (SSE only when a client asks for progress), `GET /mcp` and `DELETE /mcp` exist, `initialize` returns `Mcp-Session-Id`. | `backend/mcp/routes/__init__.py` (`mcp_endpoint`, `mcp_sse_endpoint`, `mcp_session_delete`) |
| Every MCP request needs an OAuth bearer token. An unauthenticated call gets `401` with `WWW-Authenticate: Bearer realm="pandects-mcp", resource_metadata="https://api.pandects.org/.well-known/oauth-protected-resource", ...`. | Live `POST /mcp` without a token, 2026-11-17 |
| Protected resource metadata: `resource` = `https://api.pandects.org/mcp`, one authorization server `https://api.pandects.org/v1/auth/oauth`. | Live `GET /.well-known/oauth-protected-resource` |
| Authorization server metadata: `registration_endpoint` (DCR), `code_challenge_methods_supported: ["S256"]`, `token_endpoint_auth_methods_supported: ["none"]`, grants `authorization_code` and `refresh_token`. Served at the root, issuer-pathed and OIDC paths. No `client_id_metadata_document_supported`. | Live `GET /.well-known/oauth-authorization-server` (also `/v1/auth/oauth`-pathed and `openid-configuration`); `backend/auth/mcp_oauth_runtime.py:122-143` |
| DCR accepts any HTTPS redirect URI and loopback HTTP on any port. | `backend/routes/auth/mcp_oauth.py:135-160`, `:446-458` |
| Token endpoint reads form-encoded bodies, returns `invalid_grant` for bad refresh tokens, rotates refresh tokens and revokes the family on reuse. | `backend/routes/auth/mcp_oauth.py:796-910` |
| Access tokens carry the MCP resource URL as audience. | `backend/routes/auth/mcp_oauth.py:766` |
| Token exchange fails with "Linked Pandects account is not verified" for an unverified account. | `backend/routes/auth/mcp_oauth.py:830`, `:886` |
| New accounts cannot be created in production. `TEMPORARY_ACCESS_LOCKDOWN` defaults to on, and `backend/fly.toml` does not set it. While it is on, password signup gets 403, and a new Google or OAuth identity is rolled back with 403 "New user creation is temporarily disabled". Existing accounts can still sign in, and the gate on `/v1/` data routes does not cover `/mcp`. | `backend/core/config.py:185-198`; `backend/auth/access.py:219-226`; `backend/routes/auth/website_auth.py:279-288` |
| 25 tools. 24 read only. `submit_feedback` writes a row to the `mcp_feedback` table. | `backend/tests/fixtures/mcp_tools_list_snapshot.json` |
| Every tool has `title` and `annotations` with explicit `readOnlyHint`, `destructiveHint` (always `false`), `idempotentHint`, `openWorldHint` (always `false`). | `backend/mcp/tools/__init__.py` `_tool_list_entry`; snapshot fixture |
| Longest tool name: `list_agreement_sections_batch` (29 characters). | Snapshot fixture |
| Tool results hold corpus data only. No session IDs, trace IDs or telemetry in results. | Live calls to `get_agreements_summary`, `search_agreements`, `search_sections`, `search_tax_clauses`, `get_agreement_trends`, `suggest_clause_families` |
| Server keeps per-user hourly usage rollups (user id, OAuth client id, tool name, outcome, latency, request size). Tool arguments are not logged in that rollup. | `backend/mcp/routes/helpers.py:692-732` |
| `submit_feedback` stores user id, client id, token scopes, category, severity, tool name, summary, detail, suggestions and a small context object. | `backend/models/auth.py:215-246`; `backend/mcp/tools/handlers.py:2134-2214` |
| Server `instructions` and the `submit_feedback` description ask the model to send feedback on its own initiative during a session. | `backend/mcp/tools/instructions.py` (last paragraph); `submit_feedback` description in the snapshot |
| Public pages exist: privacy policy `https://pandects.org/privacy-policy`, terms `https://pandects.org/terms`, support `https://pandects.org/support` (links to GitHub issues), docs `https://docs.pandects.org/docs/mcp/setup`. | `frontend/client/lib/routes.tsx:101,142,147`; `frontend/public/legal/*.md`; `frontend/client/pages/Support.tsx:132-140` |
| The terms make the service 18+ only. | `frontend/public/legal/terms-of-service.md:25` |
| Logo files: `branding/logo/pandects-logo.png` (762x762), 256 and 128 px versions. | `sips` on the files |

## Anthropic: Connectors Directory (MCP connector)

Summary: 15 MET, 4 NOT MET, 3 UNCLEAR, 1 needed only on request (A22). After the P1/P2 PR merges: 18 MET, 3 NOT MET, 1 UNCLEAR.

| # | Requirement | Status | Evidence or note | Who acts |
| --- | --- | --- | --- | --- |
| A1 | Remote server reachable over HTTPS | MET | `https://api.pandects.org/mcp` | |
| A2 | Supported transport (Streamable HTTP) | MET | See facts table | |
| A3 | Auth is OAuth 2.0 for an account-based service | MET | DCR + PKCE S256, public client | |
| A4 | `401` with `WWW-Authenticate` `resource_metadata` pointer | MET | Live probe | |
| A5 | `resource` equals the URL users enter | MET | `https://api.pandects.org/mcp` | |
| A6 | Callback `https://claude.ai/api/mcp/auth_callback` accepted | MET | DCR accepts any HTTPS redirect URI | |
| A7 | Claude Code loopback on any port, `localhost` and `127.0.0.1` | MET | `mcp_oauth.py:146` (RFC 8252 port-agnostic match) | |
| A8 | Token endpoint: form-encoded, `invalid_grant`, refresh rotation | MET | `mcp_oauth.py:796-910` | |
| A9 | Every tool has `title` | MET | Snapshot | |
| A10 | Every tool has `readOnlyHint` or `destructiveHint` | MET | Snapshot | |
| A11 | Tool names 64 characters or fewer | MET | Max 29 | |
| A12 | No catch-all read/write tool | MET | One purpose per tool | |
| A13 | API ownership: server calls its own API on its own domain | MET | `api.pandects.org` | |
| A14 | Descriptions free of prompt-injection patterns (no instructions unrelated to the tool, no steering of other tools) | MET once the P1 PR merges | The server instructions, `submit_feedback` description, selection hint, capabilities note, workflow note and result message now tell the model to send feedback only when the user asks or agrees to an offer. See [proposals.md](proposals.md#p1-make-feedback-user-initiated) | Merge the P1/P2 PR |
| A15 | No collection of conversation data beyond what a tool needs | MET once the P1 PR merges | The server instructions, `submit_feedback` description, selection hint, capabilities note, workflow note and result message now tell the model to send feedback only when the user asks or agrees to an offer. See [proposals.md](proposals.md#p1-make-feedback-user-initiated) | Merge the P1/P2 PR |
| A16 | Privacy policy URL | MET | `https://pandects.org/privacy-policy` | |
| A17 | Privacy policy covers this connector's data (collection, use, storage, sharing, retention, contact) | MET once the P1/P2 PR merges | The privacy policy has a new section 3.E for the MCP server (usage records, arguments, logs, conversations, feedback notes) plus retention and rights entries. The policy version is bumped, so existing users re-accept once. See [proposals.md](proposals.md#p2-privacy-policy-section-for-the-mcp-server) | Merge the P1/P2 PR |
| A18 | Public setup and usage documentation | MET (this PR) | `docs/docs/mcp/setup.md` now covers Claude custom connectors and ChatGPT as well as Codex and Claude Code. Goes live when the PR merges and deploys | Maintainer merges |
| A19 | Tested as a custom connector in Claude, every tool called | NOT MET | Needs a claude.ai account and a Pandects account. Not done here | Maintainer |
| A20 | Listing materials: icon, support contact, docs URL, privacy URL | UNCLEAR | Icon file exists. Support contact: `https://pandects.org/support` points to GitHub issues; the portal also asks for a contact, and only the maintainer can choose the address | Maintainer |
| A21 | Submitter on a paid Claude plan | NOT MET / unknown | Only the maintainer can check | Maintainer |
| A22 | Test account for Verified review (reviewer asks by email) | not required at submission | Needed only if Anthropic escalates to Verified review. Account must be email-verified (`mcp_oauth.py:830`). While signup is locked (A23), the maintainer creates it through the pre-authorization script, not through signup | Maintainer, if asked |
| A23 | A directory user who clicks Connect can get an account and finish sign-in | NOT MET | Signup lockdown (see facts table). Only people with an existing account can connect; everyone else gets 403 at sign-in | Maintainer decides whether to end the lockdown before listing |

High-traffic note from Anthropic's auth page: DCR makes a new client on every fresh connection.
The page recommends CIMD or Anthropic-held credentials for servers that expect directory
traffic. This is a recommendation, not a requirement. See
[proposals.md](proposals.md#p3-client-id-metadata-document-cimd-support).

## Anthropic: Claude plugin bundle (optional second listing)

Anthropic says to submit a remote server as an MCP connector first, and optionally a plugin
bundle that points at the same URL. A plugin bundle is read from a public GitHub repository
folder. This branch adds one at `marketplace/claude-plugin/pandects/`.

| # | Requirement | Status | Evidence or note | Who acts |
| --- | --- | --- | --- | --- |
| B1 | Folder with `.claude-plugin/plugin.json` | MET (this PR) | `marketplace/claude-plugin/pandects/.claude-plugin/plugin.json` | |
| B2 | MCP server reference at the same URL as the connector | MET (this PR) | `.mcp.json` | |
| B3 | README and license | MET (this PR) | `README.md`; license field GPL-3.0 to match the repo | |
| B4 | Repository public before the listing goes live | MET | `PlatosTwin/pandects-app` is public | |
| B5 | GitHub account connected on claude.ai with push access | NOT MET | Maintainer | Maintainer |
| B6 | Passes the portal's Validate step and security scan | UNCLEAR | Run `claude plugin validate ./marketplace/claude-plugin/pandects` and the portal's Validate | Maintainer |

## OpenAI: ChatGPT Directory (plugin with a remote MCP server)

Summary: 17 MET, 7 NOT MET, 3 UNCLEAR. After the P1/P2 PR merges: 19 MET, 6 NOT MET, 2 UNCLEAR.

| # | Requirement | Status | Evidence or note | Who acts |
| --- | --- | --- | --- | --- |
| O1 | Verified individual or business identity in the OpenAI Platform org | NOT MET / unknown | Maintainer | Maintainer |
| O2 | Org owner or member with Apps Management Write | NOT MET / unknown | Maintainer | Maintainer |
| O3 | Plugin ZIP with `plugin.json` and `mcp.json` | MET (this PR) | `marketplace/openai-plugin/pandects/` | |
| O4 | Public, stable HTTPS MCP endpoint (Streamable HTTP) | MET | `https://api.pandects.org/mcp` | |
| O5 | Domain verification token at `/.well-known/openai-apps-challenge` | NOT MET (code ready in this PR) | Route added; serves `OPENAI_APPS_CHALLENGE_TOKEN`. Maintainer copies the token from the portal into the Fly secret after the PR deploys | Maintainer |
| O6 | OAuth: PKCE S256 advertised | MET | Metadata | |
| O7 | OAuth: DCR or CIMD registration | MET | DCR | |
| O8 | OAuth: redirect `https://chatgpt.com/connector/oauth/{callback_id}` accepted | MET | DCR accepts any HTTPS redirect URI. The server does not advertise RFC 9207 `iss`, so ChatGPT uses the callback-ID form | |
| O9 | OAuth: `resource` parameter bound into the token audience | UNCLEAR | Tokens use the fixed MCP resource URL as `aud` (`mcp_oauth.py:766`), which equals the value ChatGPT sends. The code does not compare the request's `resource` value. Works for this single-resource server | |
| O10 | Per-tool `_meta.securitySchemes` | MET (this PR) | `{"type": "oauth2", "scopes": [...]}` on every tool, listing every scope the handler can check (`list_agreements` and `get_agreement` add `agreements:read_fulltext`). OpenAI documents no form for "any authenticated user". The three tools that accept any scope (`get_server_capabilities`, `get_server_metrics`, `submit_feedback`) declare `"scopes": []`, modeled on OpenAI's profile-tool example, which is the only empty-scope form the docs show | |
| O11 | All three annotations explicit on every tool | MET | Snapshot | |
| O12 | `readOnlyHint` correct | MET | 24 reads `true`; `submit_feedback` `false` | |
| O13 | `destructiveHint` correct | UNCLEAR | `submit_feedback` is `false`. OpenAI says `false` fits "additive writes with no destructive or irreversible effects" but `true` fits "irreversible sends". A feedback note is additive and the user cannot recall it. See [proposals.md](proposals.md#p4-destructivehint-on-submit_feedback) | Maintainer decides |
| O14 | `openWorldHint` correct (`false` for a bounded catalog) | MET | Closed corpus | |
| O15 | Tool names human-readable, verb-based, unique | MET | Snapshot | |
| O16 | Descriptions match behavior, do not steer toward or away from other plugins | MET | Snapshot | |
| O17 | Respect user intent; no unnecessary data collection; no reconstruction of chat logs | MET once the P1 PR merges | The server instructions, `submit_feedback` description, selection hint, capabilities note, workflow note and result message now tell the model to send feedback only when the user asks or agrees to an offer. See [proposals.md](proposals.md#p1-make-feedback-user-initiated) | Merge the P1/P2 PR |
| O18 | Response minimization (no IDs, telemetry or timestamps that are not needed) | MET | Live results hold corpus data and paging fields only | |
| O19 | Privacy policy covers data categories, purposes, recipients, retention, user controls | MET once the P1/P2 PR merges | The privacy policy has a new section 3.E for the MCP server (usage records, arguments, logs, conversations, feedback notes) plus retention and rights entries. The policy version is bumped, so existing users re-accept once. See [proposals.md](proposals.md#p2-privacy-policy-section-for-the-mcp-server) | Merge the P1/P2 PR |
| O20 | Website, support, privacy and terms URLs (HTTPS) | MET | `https://pandects.org`, `/support`, `/privacy-policy`, `/terms` | |
| O21 | Name rules (no "MCP" or "Plugin" suffix, not generic) | MET | "Pandects" is a brand name | |
| O22 | No pricing or promotion in descriptions; no commerce | MET | Packet text has none | |
| O23 | Suitable for general audiences including ages 13-17 | MET with note | Content is public SEC filings. The Pandects terms restrict accounts to 18+, which is stricter than OpenAI's rule, not a conflict | |
| O24 | 5 positive and 3 negative test cases | MET (this PR) | In `plugin.json` and [openai-chatgpt.md](openai-chatgpt.md). Positive cases were run against the live server on 2026-11-17 | |
| O25 | Demo video URL | NOT MET | Maintainer records the five positive cases | Maintainer |
| O26 | Reviewer test account: full features, sample data, no MFA, no email codes | NOT MET | While signup is locked (O27), the maintainer creates an email-verified account through the pre-authorization script, not through signup, and enters it only in the portal's Review details form | Maintainer |
| O27 | A ChatGPT user who connects can get an account and finish sign-in | NOT MET | Signup lockdown (see facts table). New users get 403 at sign-in. OpenAI rejects plugins that are not "complete" or that need sign-ups a reviewer cannot finish | Maintainer decides whether to end the lockdown before listing |

Not applicable: UI and iframe rules (no UI resources), commerce, location, file inputs, EU data
residency (the maintainer's OpenAI project must use global residency).
