# Pandects Dataset Changelog

<!-- GENERATED FILE — do not edit by hand.
     Edit bulk/changelog/changelog.yml and run
     bulk/changelog/render_changelog.py render (push_to_r2.sh does this
     automatically at release time). -->

Change history for the public Pandects database dumps, the REST API, and
the MCP surface. Machine-readable version:
<https://bulk.pandects.org/dumps/changelog.json>.

## 2026-09-09 — released 2026-09-09

- Dump: `dumps/public_2026-09-09_16-08-00.sql.gz`
- SHA-256: `183757f178fb2abc2a50ff76cd42288f21d799a255340567a479bbdf6c3793a9`

- **[data/notable]** Fixed SPAC acquirer misattribution: 135 agreements had the Pubco recorded as acquirer
  - Offline acquirer extraction now disambiguates the SPAC counterparty from the surviving Pubco entity. acquirer name fields on 135 agreements rows were corrected in place; one agreement (Overture) still awaits a re-scrape.
  - Tables: agreements
  - Refs: 0da4871
- **[mcp/notable]** search_sections gains lexical full-text search over current section text via text_query and text_match_mode (phrase, all_terms, any_terms)
  - Text search composes with standard_id and every structured filter. It is served from a private section_text_search table that is part of the production restore snapshot but not the public dump. Queries are capped at 256 characters and 24 terms, and each text-search statement runs under a 20-second database bound; a query that exceeds it returns a validation error asking for a narrower query.
  - Refs: 2d4c315, 7200bdb
- **[data/notable]** Monthly load: 146 agreements added (filings through 2026-08-14) with 3,430 new searchable sections
  - Covers filings from late July through 14 August 2026. agreements 13,980 -> 14,126; latest_sections_search 947,546 -> 950,976. The section text search index tracks latest_sections_search row for row.
  - Tables: agreements, sections, latest_sections_search
- **[api/minor]** Added changelog distribution: dumps/changelog.json is published next to each dump, GET /v1/changelog serves it (filterable by since/dump_sha256), and /v1/dumps entries gain changelog_url
  - Refs: 1eb2010
- **[mcp/minor]** get_server_capabilities gains a changelog section (latest release, breaking flag, URLs) and the initialize instructions point agents at it
  - The section reports the newest published release and whether it carries breaking changes. Its latest_version, latest_released, and breaking_changes_in_latest fields are null when the published changelog is unreachable; the url and api_route fields are always populated.
  - Refs: 1eb2010
- Stats: 30 tables, 2,511,170 total rows (per-table counts in the machine-readable changelog)

## 2026-07-19 — released 2026-07-20

- Dump: `dumps/public_2026-07-19_22-30-04.sql.gz`
- SHA-256: `fd703d72dfadd59a0fdc720399f5ecc27c23a8c6f99b76279c80983bf7fd99a8`

- **[docs/minor]** Retroactive baseline entry for the previously published dump; change history begins with the next release.
  - Fields backfilled from dumps/latest.json and the committed schema docs. Row counts were measured from the source database on 2026-08-05 and may differ slightly from the dump as published on 2026-07-20.
- Stats: 30 tables, 2,490,840 total rows (per-table counts in the machine-readable changelog)
