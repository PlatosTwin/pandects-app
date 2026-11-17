# Anthropic: Claude Connectors Directory submission

Program: the Anthropic directory (shown in Claude as the Connectors Directory; Anthropic also
calls the public catalog "Claude Marketplace" and says there is no separate submission for it).
Status on 2026-11-17: open. "The portal is always open for new submissions."

Where: https://claude.ai/directory/manage, select **Submit new**, then **MCP connector**.
Who: any paid Claude plan (Pro, Max, Ultra, Team, Enterprise). On Team or Enterprise, an Owner
or a member with the **Directory** permission. The listing belongs to the organization you
submit from.

Before you start: finish the items marked "before submitting" in [README.md](README.md#checklist).

`[MAINTAINER: ...]` marks a value only the maintainer can supply.

## Step 1. Connection

- Server URL: `https://api.pandects.org/mcp`
- Users connect to different URLs: **No** (universal URL)

## Step 2. Tools

The portal reads the tools from the server. Expected result: 25 tools, all with titles and
annotations, none flagged. 24 are read-only; `submit_feedback` is a write tool.

| Tool | Title | Read-only |
| --- | --- | --- |
| `search_agreements` | Search agreements | yes |
| `search_sections` | Search clause sections | yes |
| `search_tax_clauses` | Search tax clauses | yes |
| `list_agreements` | List agreements by exact filters | yes |
| `list_agreement_sections` | List an agreement's sections | yes |
| `list_agreement_sections_batch` | List sections for many agreements | yes |
| `get_agreement` | Get agreement | yes |
| `get_agreements_batch` | Get agreements in batch | yes |
| `get_section` | Get section | yes |
| `get_agreement_tax_clauses` | Get an agreement's tax clauses | yes |
| `get_section_tax_clauses` | Get a section's tax clauses | yes |
| `list_filter_options` | List filter values | yes |
| `suggest_clause_families` | Map a concept to clause taxonomy | yes |
| `get_section_snippet` | Get section snippet | yes |
| `get_section_snippets_batch` | Get section snippets in batch | yes |
| `get_sections_batch` | Get sections in batch | yes |
| `get_server_metrics` | Server metrics | yes |
| `get_server_capabilities` | Server reference guide | yes |
| `get_clause_taxonomy` | Clause taxonomy tree | yes |
| `get_tax_clause_taxonomy` | Tax clause taxonomy tree | yes |
| `get_counsel_catalog` | Law firm catalog | yes |
| `get_naics_catalog` | NAICS industry catalog | yes |
| `get_agreements_summary` | Corpus size summary | yes |
| `get_agreement_trends` | Deal trends by year | yes |
| `submit_feedback` | Send feedback to the maintainer | no (adds a note) |

The server also exposes 3 prompts ("Compare two M&A agreements", "Survey a clause family across
a filter", "Brief on deal trends for a slice") and resources.

## Step 3. Listing

- Server name (max 100): `Pandects`
- One-liner (max 200, 139 used):

  > Search and read SEC-filed M&A agreements: deal metadata, law-firm counsel, clause-level text, a clause taxonomy, and extracted tax clauses.

- Description (max 2,000, about 1,250 used):

  > Pandects is an open database of merger and acquisition agreements that companies filed with the SEC on EDGAR. This connector gives Claude read access to that corpus.
  >
  > What you can do:
  > - Find agreements by party name, filing year, deal size range, deal type, consideration, industry (NAICS code), public or private status, private-equity involvement, and target-side or acquirer-side law firm.
  > - Read agreement metadata: parties, counsel, filing and deal dates, price, consideration, and the link to the SEC filing.
  > - Search clause-level section text by phrase or by clause family (for example termination fees, no-shop, material adverse effect, efforts covenants and indemnification), and read the full section.
  > - Search extracted tax clauses by tax concept (for example an intended tax-free reorganization).
  > - Get year-by-year corpus trends: public and private targets, buyer types and industries.
  >
  > Each agreement result links to the SEC filing. The corpus holds agreements that an SEC filer attached as an exhibit, so deals between two private companies are mostly absent, and one deal can appear in several filings.
  >
  > You need a Pandects account with a verified email address. When you connect, Claude opens the Pandects sign-in page.

- Categories (1 to 5): pick the closest from the portal list, in this order of fit:
  Legal, Research, Finance, Data. (The category list is shown only in the portal.)
- Documentation URL: `https://docs.pandects.org/docs/mcp/setup`
- Privacy policy URL: `https://pandects.org/privacy-policy`
- Support contact: `https://pandects.org/support` (GitHub issues) or `[MAINTAINER: support email]`
- Icon: `marketplace/anthropic-icon.png` (762 x 762 PNG, copy of `branding/logo/pandects-logo.png`)
- URL slug (permanent): `pandects`
- Allowed link URIs: none. The server does not use `ui/open-link`.

## Step 4. Use cases

- Primary use cases:
  1. Find M&A agreements and deal terms by party, year, deal size, industry or law firm.
  2. Read and compare how a clause is drafted across deals (termination fees, no-shop, MAE, efforts covenants, indemnification).
  3. Find tax structure language, such as intended tax-free reorganization clauses.
  4. Get year-by-year context on the M&A agreements in the corpus.
- What users need before they connect: a Pandects account (sign up at `https://pandects.org/signup`) with a verified email address. `[MAINTAINER: confirm that no paid plan is needed; the terms name no fees]`
- Reads or writes: **Both**. 24 tools read the corpus. `submit_feedback` adds a feedback note for the Pandects maintainer and changes no user data.

## Step 5. Company

- Company name: `[MAINTAINER: legal name of the publisher, or "Pandects"]`
- Website: `https://pandects.org`
- Primary contact for review updates: `[MAINTAINER: name and email]`

## Step 6. Authentication

- Authentication requirement: **Always required**
- Authentication method: **OAuth sign-in**. The server supports Dynamic Client Registration
  (`https://api.pandects.org/v1/auth/oauth/register`) with PKCE S256 and public clients. Do not
  enter a client ID or secret; let Claude register itself.

## Step 7. Data handling

- Underlying API: **your own** (`api.pandects.org`, operated by Pandects).
- Personal health data: **No**.
- Sponsored content: **No**.

## Step 8. Test & launch

Confirm only after you have done this:

1. In Claude, **Customize > Connectors > Add > Add custom connector**, URL `https://api.pandects.org/mcp`, keep the detected settings, **Add**, **Connect**, sign in.
2. Call every tool once from a conversation. The five prompts in
   [openai-chatgpt.md](openai-chatgpt.md#test-cases) cover the main tools; also ask for
   `get_server_capabilities`, `get_clause_taxonomy`, `get_naics_catalog` and `get_server_metrics`.
3. Optionally run MCP Inspector against the URL.

Do not put credentials in the portal.

## Step 9. Compliance

Seven acknowledgments, all required: directory guidelines, first-party API usage, financial
transactions, AI media generation, prompt injection, conversation data collection, public
documentation. Before you tick "prompt injection" and "conversation data collection", read
[proposals.md P1](proposals.md#p1-make-feedback-user-initiated): the current feedback wording
asks the model to send session notes on its own.

## Step 10. Review and submit

Check the quality warnings the portal shows, then submit. Anthropic scans the submission and by
default lists it as a Community connector. Escalations: `mcp-review@anthropic.com`.

## Optional: plugin bundle

Anthropic recommends submitting the server as a connector first, then a plugin bundle that
points at the same URL. The bundle is at `marketplace/claude-plugin/pandects/` and passes
`claude plugin validate`. To submit: portal, **Submit new**, **Plugin bundle**, Repository
`PlatosTwin/pandects-app`, Plugin path `marketplace/claude-plugin/pandects`, Branch `main`
(after this PR merges), then **Validate**. The portal needs your GitHub account connected on
claude.ai with push access. Data handling answers: reads no personal data, sends data only to
its declared connector, keeps no data, not intended for people under 18.
