# Marketplace submission packet

This folder holds what Pandects needs to list its MCP server (`https://api.pandects.org/mcp`) in
the Anthropic and OpenAI directories. No deploy workflow watches this folder. Nothing here has
been submitted.

| File | What it is |
| --- | --- |
| [anthropic-claude.md](anthropic-claude.md) | Every step and field of the Claude directory portal, ready to paste |
| [openai-chatgpt.md](openai-chatgpt.md) | Every step and field of the OpenAI plugin portal, with the test cases |
| [audit.md](audit.md) | Each program requirement marked MET, NOT MET or UNCLEAR, with evidence |
| [proposals.md](proposals.md) | Changes that need the maintainer's decision, with ready text |
| `claude-plugin/pandects/` | Optional Claude plugin bundle (skill plus `.mcp.json`); passes `claude plugin validate` |
| `openai-plugin/pandects/` | The OpenAI plugin package to zip and upload |
| `anthropic-icon.png` | 762 x 762 icon for the Claude listing |

## Checklist

Items still NOT MET or UNCLEAR, in the order to do them. "Maintainer" means only the maintainer
can do it (an account, a decision, legal text, or a production change).

Audit totals ([audit.md](audit.md)): Anthropic connector 15 MET, 4 NOT MET, 3 UNCLEAR (plus one
item needed only on request); OpenAI plugin 17 MET, 7 NOT MET, 3 UNCLEAR.

Before submitting to either program:

- [ ] Decide whether to end the signup lockdown before listing; the listing cannot work for
      new users while it is on. `TEMPORARY_ACCESS_LOCKDOWN` defaults to on in production, so
      password signup and first-time Google or OAuth sign-in return 403 (audit A23, O27).
      (Maintainer; production setting)
- [ ] Merge the PR that added this folder, so the setup docs for Claude and ChatGPT, the
      per-tool `securitySchemes` and the OpenAI verification route go live. (Maintainer; a
      merge deploys to production.)
- [ ] Decide on [P1](proposals.md#p1-make-feedback-user-initiated): make `submit_feedback`
      user-initiated. Both reviews check for tools that act or collect session data without a
      user request. (Maintainer)
- [ ] Add an MCP section to the privacy policy ([P2](proposals.md#p2-privacy-policy-section-for-the-mcp-server)
      has a draft). Both programs require the policy to cover the data the connector handles.
      (Maintainer; legal text)
- [ ] Create a dedicated reviewer account: email-verified, no MFA, password sign-in, works from
      outside your network. While the lockdown is on, create it with the pre-authorization
      script, not through signup. OpenAI requires it at submission; Anthropic asks for it only
      if it escalates to Verified review. (Maintainer; production write)
- [ ] Choose a support contact (the `/support` page links to GitHub issues) and the company
      name for the listings. (Maintainer)

Anthropic (connector):

- [ ] Use a paid Claude plan account (Pro or higher) for the portal. (Maintainer)
- [ ] Add `https://api.pandects.org/mcp` as a custom connector in Claude and call every tool.
      (Maintainer)
- [ ] Submit through https://claude.ai/directory/manage, following [anthropic-claude.md](anthropic-claude.md).
      (Maintainer)
- [ ] Optional: submit the plugin bundle `marketplace/claude-plugin/pandects` after the PR
      merges; needs GitHub connected on claude.ai. (Maintainer)

OpenAI (plugin):

- [ ] Complete identity verification in the OpenAI Platform organization and get Apps
      Management Write. Use a global-residency project. (Maintainer)
- [ ] Record a demo video of the five positive test cases in ChatGPT and add its URL to
      `plugin.json` as `review.demo_recording_url`. (Maintainer)
- [ ] Upload the ZIP, set the domain-verification token as the Fly secret
      `OPENAI_APPS_CHALLENGE_TOKEN`, wait for the tool scan. (Maintainer; production secret)
- [ ] Enter the reviewer account in **Review details**, submit, then publish after approval.
      (Maintainer)

Recommended later, not required: [P3](proposals.md#p3-client-id-metadata-document-cimd-support)
CIMD support if DCR client registrations grow fast after listing.

## Sources

All pages were read on 2026-11-17. None shows a "last updated" date.

Anthropic:

- Publish to the directory (who can submit, submission kinds): https://claude.com/docs/directory/publish
- Submit a connector (portal steps and fields): https://claude.com/docs/connectors/building/submission
- Connector pre-submission checklist (review criteria): https://claude.com/docs/connectors/building/review-criteria
- Authentication for connectors: https://claude.com/docs/connectors/building/authentication
- Submit your plugin: https://claude.com/docs/plugins/submit
- Add a connector that isn't in the directory: https://claude.com/docs/connectors/custom/add-unlisted
- Software Directory Terms and Policy: https://support.claude.com/en/articles/13145338-anthropic-software-directory-terms and https://support.claude.com/en/articles/13145358-anthropic-software-directory-policy (linked from the pages above; not read in full)

OpenAI:

- Plugin guidelines (review criteria): https://developers.openai.com/apps-sdk/app-submission-guidelines
- Upload and submit your plugin: https://developers.openai.com/plugins/deploy/submission
- Remote MCP server review requirements: https://developers.openai.com/plugins/deploy/app-review
- Package your plugin: https://developers.openai.com/plugins/build/plugins
- Authentication: https://developers.openai.com/plugins/build/auth
- Submit your Claude Code plugin to OpenAI: https://developers.openai.com/plugins/guides/submit-claude-plugin
- Connect and test your plugin: https://developers.openai.com/plugins/deploy/connect-chatgpt
- Help center article https://help.openai.com/en/articles/20001040 returned HTTP 403 and was not read.

Desktop extensions (MCPB) are no longer accepted in the Claude directory, and the OpenAI portal
does not accept `.mcpb` files, so this packet has no desktop extension.
