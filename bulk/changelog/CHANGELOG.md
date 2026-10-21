# Pandects Dataset Changelog

<!-- GENERATED FILE — do not edit by hand.
     Edit bulk/changelog/changelog.yml and run
     bulk/changelog/render_changelog.py render (push_to_r2.sh does this
     automatically at release time). -->

Change history for the public Pandects database dumps, the REST API, and
the MCP surface. Machine-readable version:
<https://bulk.pandects.org/dumps/changelog.json>.

## Unreleased

- **[api/notable]** Error responses now carry a machine code in `error` instead of the HTTP reason phrase.
  - Every /v1 error body is {"error": <code>, "message": <text>}. Generic errors used to put the HTTP reason phrase in `error` ("Bad Request", "Not Found", "Unprocessable Entity", ...); they now use stable snake_case codes: bad_request (400), unauthorized (401), forbidden (403), not_found (404), method_not_allowed (405), conflict (409), validation_error (422), rate_limited (429), internal_error (500), service_unavailable (503). Other statuses use the snake_case reason phrase. Codes that were already specific (validation_error, rate_limited, captcha_required, legal_required, ...) are unchanged. A 422 from query-parameter parsing now names the field in `message` ("page_size: Not a valid integer.") instead of a generic sentence. Status codes and the /mcp JSON-RPC errors are unchanged.
  - Migration: Clients that compared `error` to a reason phrase should compare it to the new code, or branch on the HTTP status.
- **[mcp/notable]** search_sections phrase queries for common language now return results instead of an error.
  - A phrase common enough to match a large share of the corpus -- "material adverse effect", "reasonable best efforts", "governing law" -- previously exceeded the database time bound and came back as a validation error telling the caller to add filters, which could not help. Phrase mode now picks between two query plans per search, so these queries return normally. Two consumer-visible consequences: the error text for a phrase that still cannot be served now asks for a longer or more distinctive phrase rather than filters, and for the most common phrases total_count is reported as a lower bound with method `filtered_lower_bound` and planning_reliability `low`, because an exact count of those matches is not affordable. Read total_count together with has_next on those searches; count_mode=exact degrades the same way rather than failing. Result rows and their order are unchanged.
- **[api/notable]** /v1/search/agreements now applies text_query, and rejects metadata with a 422.
  - text_query and text_match_mode (phrase, all_terms, any_terms) were accepted on /v1/search/agreements but silently ignored, so a text search returned every agreement matching the other filters. An agreement now matches when at least one of its latest sections matches the text (and, together with standard_id or section_uuid, when one section matches all of them). match_count counts those sections, matched_sections previews the first three in document order with a snippet centered on the match, and each preview carries a new matched_terms list of the matched spans. When an exact count is not affordable, total_count is a lower bound (total_count_is_approximate true, count_metadata.method filtered_lower_bound); read it with has_next. A text query too expensive to serve returns a 422 on text_query. The metadata parameter, previously accepted and ignored, now returns a 422: every result already carries the full agreement metadata. Responses also gain count_metadata, and dump_version on searches without section criteria.
- **[api/notable]** Section search and GET /v1/sections/{section_uuid} exclude gated, unverified agreements.
  - /v1/sections and GET /v1/sections/{section_uuid} now apply the same public-eligibility rule as the agreement endpoints, which already hid agreements that are gated and not yet verified. A section of such an agreement no longer appears in search results and returns 404 when fetched directly. The MCP get_section, get_sections_batch, and get_agreement tools apply the same rule.
- **[api/notable]** An unknown price bucket is a 422 instead of being ignored.
  - transaction_price_total, transaction_price_stock, transaction_price_cash, and transaction_price_assets accept only the canonical bucket labels (0 - 100M, 100M - 250M, 250M - 500M, 500M - 750M, 750M - 1B, 1B - 2B, 2B - 5B, 5B - 10B, 10B - 20B, 20B+). Any other value (for example "1B - 5B") used to be dropped silently, so the request returned results unfiltered by price. /v1/sections, /v1/search/agreements, /v1/agreements, and /v1/tax-clauses now answer 422 validation_error, and the message names the field, the rejected value, and the valid labels. The MCP search_sections, search_agreements, list_agreements, and search_tax_clauses tools return the same detail as a -32602 invalid-arguments error. The OpenAPI spec lists the labels as an enum.
  - Migration: Send one of the listed labels; list_filter_options (MCP) returns them under transaction_price_totals.
- **[data/notable]** Adjacent headings of the same type are no longer merged into one section: 1,370 agreements re-sectioned.
  - The tagger printed two back-to-back headings of the same kind (for example "Section 3.5 [Reserved]." directly followed by "Section 3.6 Material Contracts.") as a single tag, so the XML and the sections table held one section where the agreement has two. The 2,261 affected body pages in 1,370 agreements were re-tagged, adding 2,436 section, 71 article and 114 page boundaries, and their XML and sections were rebuilt. Of those agreements, 1,301 that were verified stay verified and 11 that were invalid now verify; none went from verified to invalid. Three agreements also had typos in the filed section numbers corrected in the source text (cb7e8e4d, 0b3a6ac0, e739fd7e). Rebuilt sections get new section ids, and sections with new ids carry no taxonomy label (standard_id) until the taxonomy is rebuilt, so standard_id coverage is lower in these agreements for now.
  - Tables: xml, sections, latest_sections_search, latest_sections_search_standard_ids
  - Migration: Section ids for the 1,370 agreements changed; re-fetch them rather than joining on ids saved from the previous dump.
  - Refs: 2e4190bd
- **[data/notable]** Page boundaries corrected: signature pages, exhibits and tables of contents removed from agreement body text in 93 agreements.
  - Each filing's pages are labeled front matter, table of contents, body, signature or back matter, and only body pages become XML sections. The page classifier was replaced with a more accurate model (see Sources & Methods), and its labels were rechecked for 1,912 agreements whose stored labels looked least reliable. 668 page labels changed across 228 agreements (113 of them in 7 hand-labeled agreements whose labels were wrong). In 93 agreements the body itself changed: typically a signature page or an attached exhibit (a transition services or escrow agreement, for example) had been parsed as the agreement's last sections, or a table-of-contents page as its first. Their XML and sections were rebuilt; 2 agreements that previously failed XML validation now pass. Changes that moved pages only among the non-body classes do not affect the XML or sections.
  - Tables: xml, sections, latest_sections_search, latest_sections_search_standard_ids
  - Refs: b5a626a2, 54955dea
- **[data/notable]** Deal metadata gaps filled: 976 empty fields across 580 agreements.
  - Fills were written only into empty fields; no existing value changed. Missing values dropped for deal_status (647 -> 405), attitude (300 -> 129), acquirer_industry (318 -> 261), target_industry (232 -> 188) and purpose (113 -> 95).
  - Tables: agreements
- **[data/notable]** Deal metadata corrected on about 1,300 agreements, including consideration type on roughly 870 deals previously marked mixed.
  - An audit of 250 deals found systematic errors, which were then corrected across the dataset. 706 agreements were corrected in a first pass: 71 audit findings confirmed by an independent second check (45 deal price totals, 13 close dates, 10 consideration types), 113 deals whose SPAC acquirer is listed now recorded as having a public acquirer, 213 deals left as pending long after closing resolved to complete or cancelled, and 309 deals marked mixed consideration corrected (292 of them to all-stock with the price cleaned). A second pass made 631 corrections to deals marked mixed that have only one price component; 571 of those changed consideration type. Deals marked mixed fell from 4,919 to 4,046. Deal price now means the equity value at signing, without earnouts or CVRs, and cash paid in lieu of fractional shares no longer makes a deal mixed.
  - Tables: agreements
- **[data/notable]** [N] agreements added from a review of the previously unadmitted EDGAR filing queue (filings 2021-01 to 2026-10).
  - 3,542 EDGAR filings that the automatic title check could not admit were reviewed against a written rubric, with a second, independent read of every admitted filing: an executed, arm's-length, definitive agreement in which a buyer acquires a company or operating business. 2,171 were admitted and 1,371 rejected. [N] of the admitted deals have completed XML and sections in this release, adding [M] searchable sections; the rest will follow. Where an admitted filing is an amended and restated version of an agreement already in the dataset, the newer version replaces the older one ([R] agreements removed). Deal metadata (parties, price, consideration) for the new agreements is mostly empty and will be filled in a later release.
  - Tables: agreements, xml, sections, latest_sections_search
- **[mcp/minor]** MCP OAuth token and client-registration errors now use the codes the OAuth RFCs define.
  - POST /v1/auth/oauth/token errors return {"error", "error_description", "message"} with the RFC 6749 codes invalid_request, invalid_client, invalid_grant, unauthorized_client and unsupported_grant_type. POST /v1/auth/oauth/register errors use the RFC 7591 codes invalid_redirect_uri and invalid_client_metadata. Errors from both endpoints carry Cache-Control: no-store and Pragma: no-cache. Both used to return the generic bad_request code. A code or refresh token whose Pandects account is not verified is now 400 invalid_grant instead of 403. `message` still carries the same text. The /.well-known OAuth discovery endpoints now return the standard error envelope codes (not_found, service_unavailable) instead of HTTP reason phrases.
  - Migration: MCP clients that follow the OAuth RFCs need no change. Code that matched `error == "bad_request"` on these two endpoints should match the RFC code or the status.
- **[docs/minor]** The published OpenAPI spec now matches what the API returns.
  - The spec at /openapi.yaml is regenerated from the API on every change (CI fails when it is stale), so it now documents the endpoints and parameters added since the last manual export, including /v1/taxonomy/counts and /v1/taxonomy/tax-clauses/counts. The `Error` schema now describes the real error body, {"error": <code>, "message": <text>}, instead of flask-smorest's {code, status, message, errors}. Response fields that are always present are marked required, and target_pe / acquirer_pe are documented as the 0/1 integers (or null) the API returns. Responses themselves are unchanged.
- **[api/minor]** REST responses carry industry and taxonomy labels next to the raw codes.
  - Agreement rows on /v1/agreements, /v1/agreements/{uuid}, and /v1/search/agreements gain target_industry_label and acquirer_industry_label: the NAICS description of the code (a code with no catalog entry of its own rolls up to its 2-digit sector; null when it cannot be resolved). The /v1/sections result metadata block gains the same labels for the industry fields that were requested. Section results, /v1/sections/{uuid}, /v1/agreements/{uuid}/sections, and /v1/search/agreements matched_sections gain standard_id_labels, a {standard_id: label} map for the clause-taxonomy ids in standard_id. Tax-clause results on /v1/tax-clauses gain tax_standard_id_labels for tax_standard_ids, and the agreement and section tax-clause lists gain standard_id_labels for standard_ids. /v1/filter-options gains industry_labels ({code: label}) whenever target_industries or acquirer_industries is returned. Existing fields are unchanged; callers no longer need /v1/naics or /v1/taxonomy just to print a label.
- **[api/minor]** New /v1/taxonomy/counts and /v1/taxonomy/tax-clauses/counts endpoints.
  - GET /v1/taxonomy/counts returns {"counts": {standard_id: n}}: how many sections /v1/sections?standard_id=<id> matches for each clause-taxonomy node, a parent counting the distinct sections tagged with it or any descendant. GET /v1/taxonomy/tax-clauses/counts returns the same shape for the tax-clause taxonomy, counting the clauses /v1/tax-clauses?tax_standard_id=<id> matches with default options (rep-and-warranty clauses excluded). Nodes with no matches are omitted.
- **[api/minor]** The clause_types field of /v1/filter-options is deprecated in favor of /v1/taxonomy.
  - clause_types has always returned the same tree as GET /v1/taxonomy; both now come from one builder and one cache. clause_types keeps working for existing callers, but new code should read /v1/taxonomy, which is cacheable on its own. The Pandects web client no longer requests clause_types.
- **[api/minor]** /v1/agreements-summary now reports processed_agreements alongside agreements.
  - The new processed_agreements field counts the agreements that have been processed and can be browsed and searched (the rows behind /v1/agreements-index). The existing agreements field is unchanged and still counts every agreement ingested, including those pending processing or verification.
- **[api/minor]** An out-of-range page_size now returns the largest allowed page instead of 25 rows.
  - On /v1/sections, /v1/tax-clauses, /v1/agreements, /v1/search/agreements, and the MCP search_sections, search_tax_clauses, search_agreements, list_agreements, and list_agreement_sections tools, a page_size above the caller's ceiling (100 for signed-in and API-key callers, 10 for anonymous callers) is now clamped to that ceiling rather than reset to the default of 25. A page_size below 1 still falls back to the default. Requests within range are unchanged.
- **[api/minor]** /v1/agreements accepts year_min, year_max, filed_after, and filed_before.
  - The year and filing-date range filters already accepted by /v1/sections and /v1/search/agreements now work on /v1/agreements too, with the same semantics (filed_after inclusive, filed_before exclusive, YYYY-MM-DD). Every agreement endpoint and MCP tool now shares one metadata filter implementation, so the same arguments select the same agreements everywhere.
- **[api/minor]** Canonical id fields on single reads, sort_direction on the agreements index, dump_version on tax clauses.
  - GET /v1/agreements/{agreement_uuid} and the MCP get_agreement tool now return agreement_uuid. GET /v1/sections/{section_uuid} returns standard_id (the name used everywhere else); section_standard_id stays as a deprecated alias with the same value. /v1/agreements-index accepts sort_direction like every other paginated endpoint; sort_dir keeps working as a deprecated alias, and sort_direction wins when both are sent. /v1/tax-clauses returns dump_version and accepts include_dump=false to omit it, as /v1/sections does.
- **[api/minor]** Search snippets no longer contain page numbers and page ids.
  - A section that crosses a page break carries <page> and <pageUUID> markers in its XML, and their contents leaked into /v1/search/agreements matched_sections[].snippet as text ("... more than 20 69 66af9b9a-20dc-11f1-ac97-be44a5e954c8 Business Days ..."). Snippets are now built from the agreement text only, so a text_query phrase that spans a page break also matches and is reported in matched_terms. The XML returned by the section and agreement endpoints is unchanged.
- **[mcp/minor]** Snippets and monetary_values ignore page numbers and page ids.
  - get_section_snippet, get_section_snippets_batch, get_sections_batch, and search_sections / search_agreements with snippets now drop the <page> and <pageUUID> markers before excerpting, so snippets read as clean text and a focus term that spans a page break is found. monetary_values is extracted from the same cleaned text.
- **[mcp/minor]** MCP industry labels roll an unlisted NAICS code up to its sector, as REST does.
  - target_industry_label / acquirer_industry_label (list_agreements, get_agreement, get_agreements_batch, search_sections metadata) and the list_filter_options industry_labels map used to match the code exactly, so a code with no catalog entry of its own (for example "459", "513", "922") was null in MCP while REST labeled it with its 2-digit sector ("Retail Trade" for "459"). Both surfaces now share one resolver. A non-numeric value or a code whose sector is also missing stays null.
- **[data/minor]** Counsel refreshed for 46 agreements.
  - Tables: agreement_counsel, counsel

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
- **[data/notable]** Monthly load: 146 agreements added (filings through 2026-08-14) with 6,937 new searchable sections
  - Covers filings from late July through 14 August 2026. agreements 13,980 -> 14,126; latest_sections_search 947,546 -> 954,483. The section text search index tracks latest_sections_search row for row.
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
