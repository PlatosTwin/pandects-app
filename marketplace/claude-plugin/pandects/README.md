# Pandects plugin

This plugin connects Claude to the Pandects MCP server at `https://api.pandects.org/mcp` and adds
a skill that maps M&A research questions to the right Pandects tools.

Pandects is a database of merger and acquisition agreements filed with the SEC on EDGAR. The
server gives read access to agreement metadata (parties, law-firm counsel, industry, deal size,
consideration, dates), clause-level section text, a clause-family taxonomy, extracted tax
clauses and year-by-year trends.

## Contents

- `.mcp.json`: the remote Pandects MCP server
- `skills/pandects-research/SKILL.md`: which tool to use for which research question

## Requirements

A Pandects account with a verified email address (sign in at https://pandects.org/login). When
the server first connects, your client opens the Pandects sign-in page in your browser.

## Tools

24 tools read data. One tool, `submit_feedback`, saves a note for the Pandects maintainer.
Setup and tool documentation: https://docs.pandects.org/docs/mcp/setup

## Privacy Policy

https://pandects.org/privacy-policy

## Support

https://pandects.org/support

## License

GPL-3.0. See `LICENSE.md` at the root of https://github.com/PlatosTwin/pandects-app.
