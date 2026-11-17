# OpenAI: ChatGPT Directory submission

Program: the ChatGPT Directory. OpenAI now calls listings "plugins"; a plugin can hold a remote
MCP server, skills, or both, and the same listing reaches ChatGPT and Codex. (Earlier names:
"apps in ChatGPT", "app directory", Apps SDK.) Status on 2026-11-17: open to submissions from
verified developers; OpenAI describes the directory and SDK as beta.

Where: https://platform.openai.com/plugins, select **Upload new or existing plugin**.
Who: an organization owner, or a member with **Apps Management Write**. The organization must
have completed individual or business verification, and you pick a verified **Developer
identity** when you create the draft. The project must use global (not EU) data residency.

`[MAINTAINER: ...]` marks a value only the maintainer can supply.

## Step 1. Build the ZIP

The package is `marketplace/openai-plugin/pandects/`:

```text
pandects/
  plugin.json        Agent Plugins manifest with the listing, review cases and release notes
  mcp.json           https://api.pandects.org/mcp, Streamable HTTP, OAuth with DCR
  skills/pandects-research/SKILL.md
  assets/logo.png            256 x 256
  assets/composer-icon.png   128 x 128
```

From the repository root:

```bash
cd marketplace/openai-plugin && zip -r pandects-openai-plugin.zip pandects
```

Before you zip, add your demo video link to `plugin.json` under
`extensions["com.openai"].review.demo_recording_url`. Do not put test credentials in the ZIP;
the portal rejects `test_credentials` and `reviewer_instructions` there.

## Step 2. Upload and fix automated findings

Upload the ZIP. The portal runs metadata and skill checks. If it lists issues, use **Copy
issues**, fix the files, and upload with **Upload plugin to fix issues**.

## Step 3. Connect the MCP server and verify the domain

1. Connect `https://api.pandects.org/mcp` in the dashboard. Authentication: **OAuth**, client
   registration **Dynamic Client Registration**. Do not enter a client secret.
2. The portal shows a domain-verification token. Set it as a Fly secret on the backend app:
   `OPENAI_APPS_CHALLENGE_TOKEN=<token>`. This PR adds the route that serves it at
   `https://api.pandects.org/.well-known/openai-apps-challenge` (404 while unset). Check it
   with `curl https://api.pandects.org/.well-known/openai-apps-challenge`.
3. Wait for the automated tool scan. Expected: 25 tools; every tool has `readOnlyHint`,
   `destructiveHint` and `openWorldHint` set explicitly and `_meta.securitySchemes` of type
   `oauth2`. See [proposals.md P4](proposals.md#p4-destructivehint-on-submit_feedback) if the
   scan questions `submit_feedback`.

## Step 4. Listing fields

All are in `plugin.json` under `extensions["com.openai"].interface`:

| Field | Limit | Value |
| --- | --- | --- |
| displayName | 30 | `Pandects` |
| shortDescription | 30 | `Search SEC M&A agreements` |
| longDescription | 4000 | Same text as the Anthropic description, with "ChatGPT" for "Claude" (1,235 characters) |
| developerName | 80 | `Pandects` `[MAINTAINER: change to your verified developer identity name if it differs]` |
| category | dashboard list | `Research` (change to the closest dashboard category if the upload flags it) |
| capabilities | 20 x 120 | 5 lines, see `plugin.json` |
| websiteURL | | `https://pandects.org` |
| supportURL | | `https://pandects.org/support` |
| privacyPolicyURL | | `https://pandects.org/privacy-policy` |
| termsOfServiceURL | | `https://pandects.org/terms` |
| defaultPrompt | 3 x 128 | The first three positive prompts below |
| logo, composerIcon | square, 48 px or more | `./assets/logo.png`, `./assets/composer-icon.png` |

## Test cases

OpenAI requires 5 positive and 3 negative cases for the first MCP review. All five positive
cases were run against the live server on 2026-11-17 and returned the results named here.
Counts can change as the corpus grows.

| # | Prompt | Tools | Expected result |
| --- | --- | --- | --- |
| P1 | Find the Salesforce-Slack merger agreement and tell me which law firm advised each side. | `search_agreements` | 2020 agreement, salesforce.com, inc. acquires Slack Technologies, Inc., with SEC link. Target counsel Latham & Watkins; acquirer counsel Wachtell, Lipton, Rosen & Katz. |
| P2 | Show reverse termination fee clauses from 2023 deals valued between $1B and $2B. | `suggest_clause_families`, `search_sections` | Clause family `4207f2e8f6698935` (Termination Fee / Reverse Termination Fee; Expense Reimbursement). About 10 sections, for example Section 7.3 Termination Fee, SP Plus Corporation / Metropolis Technologies. |
| P3 | Which 2022 deals had Wachtell, Lipton, Rosen & Katz as target counsel? | `search_agreements` | About 10 agreements, for example Continental Resources / Omega Acquisition and TEGNA / Teton Parent Corp. |
| P4 | Find 2024 merger agreement clauses saying the merger is intended to qualify as a tax-free reorganization. | `suggest_clause_families`, `search_tax_clauses` | Tax node `tax.1.1.1` (Intended Tax-Free Reorganization). About 10 clauses citing Section 368(a), for example Penns Woods Bancorp / Northwest Bancshares. |
| P5 | How did the median deal size for public targets change from 2020 to 2024 in Pandects? | `get_agreement_trends` | Per-year public-target counts and median values for 2020 to 2024, described as corpus counts, not the whole market. |

| # | Prompt | Expected behavior |
| --- | --- | --- |
| N1 | What will the weather be in Boston tomorrow? | No Pandects tool is called. |
| N2 | Buy 100 shares of Salesforce for me. | No Pandects tool is called; Pandects has no trading tools. |
| N3 | Delete the Slack merger agreement from Pandects. | No Pandects tool is called; the model says the tools are read-only. |

## Step 5. Review details (dashboard form, not the ZIP)

- Reviewer test account: `[MAINTAINER: email and password of a dedicated Pandects account]`.
  It must have a verified email, no MFA, no email or SMS codes, and must work from outside your
  network. Sign-in with email and password must work for it (not only Google sign-in).
- Demo video: `[MAINTAINER: URL]`, a walkthrough of P1 to P5 in ChatGPT.
- Commerce: none.

## Step 6. Submit, then publish

Select the draft, **Submit for review**, complete the policy attestations. Feedback arrives by
email; appeal by replying. After approval, open the approved version and select **Publish
plugin**. Later server changes go live after OpenAI's daily scan; package changes need a new
ZIP. The MCP origin (`https://api.pandects.org`) cannot change without a new plugin.
