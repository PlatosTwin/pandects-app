# Pandects Dataset Changelog

<!-- GENERATED FILE — do not edit by hand.
     Edit bulk/changelog/changelog.yml and run
     bulk/changelog/render_changelog.py render (push_to_r2.sh does this
     automatically at release time). -->

Change history for the public Pandects database dumps, the REST API, and
the MCP surface. Machine-readable version:
<https://bulk.pandects.org/dumps/changelog.json>.

## Unreleased

- **[data/notable]** Fixed SPAC acquirer misattribution: 135 agreements had the Pubco recorded as acquirer
  - Offline acquirer extraction now disambiguates the SPAC counterparty from the surviving Pubco entity. acquirer name fields on 135 agreements rows were corrected in place; one agreement (Overture) still awaits a re-scrape.
  - Tables: agreements
  - Refs: 0da4871
- **[api/minor]** Added changelog distribution: dumps/changelog.json is published next to each dump, GET /v1/changelog serves it (filterable by since/dump_sha256), and /v1/dumps entries gain changelog_url
  - Refs: 1eb2010
- **[mcp/minor]** get_server_capabilities gains a changelog section (latest release, breaking flag, URLs) and the initialize instructions point agents at it
  - Refs: 1eb2010

## 2026-07-19 — released 2026-07-20

- Dump: `dumps/public_2026-07-19_22-30-04.sql.gz`
- SHA-256: `fd703d72dfadd59a0fdc720399f5ecc27c23a8c6f99b76279c80983bf7fd99a8`

- **[docs/minor]** Retroactive baseline entry for the previously published dump; change history begins with the next release.
  - Fields backfilled from dumps/latest.json and the committed schema docs. Row counts were measured from the source database on 2026-08-05 and may differ slightly from the dump as published on 2026-07-20.
- Stats: 30 tables, 2,490,840 total rows (per-table counts in the machine-readable changelog)
