---
name: pandects-research
description: Research SEC-filed M&A agreements with the Pandects tools. Use when a task involves merger or acquisition agreements, deal terms, clause language or drafting precedent (material adverse effect, no-shop, fiduciary out, termination fees, efforts covenants, indemnification, tax structure), which law firms advised on deals, or M&A deal trends.
---

# Research M&A agreements with Pandects

Pandects is a database of merger and acquisition agreements that companies filed with the SEC
on EDGAR. The Pandects MCP server gives read access to agreement metadata, clause-level section
text, a clause-family taxonomy, extracted tax clauses and year-by-year trends.

## Pick the tool for the question

| Question | Tools, in order |
| --- | --- |
| Find a known deal | `search_agreements` with `query` (a party name prefix), then `get_agreement` |
| Deals with exact filters (year, counsel, size, deal type, industry) | `list_filter_options` for valid values, then `list_agreements` or `search_agreements` |
| How a clause is drafted (MAE, no-shop, termination fee, and so on) | `suggest_clause_families`, then `search_sections` with the returned `standard_id` and `include_snippet=true`, then `get_section` for the full text |
| Tax structure language | `suggest_clause_families` with `taxonomy="tax_clauses"`, then `search_tax_clauses` with the returned `tax_standard_id` |
| Which law firms advised | `get_counsel_catalog` for exact firm names, then `search_agreements` with `target_counsel`, `acquirer_counsel` or `any_counsel` |
| Corpus size or market trends | `get_agreements_summary`, `get_agreement_trends` |

## Read results with care

- A filing count is not a deal count: one deal can appear in several filings.
- Deals between two private companies are mostly absent, because every agreement comes from an
  SEC filer.
- A taxonomy match can be a proxy. Check `coverage` and `fit` from `suggest_clause_families`
  and read the clause text before you state a conclusion.
- Industry filters take NAICS codes, not labels. Price filters take bucket labels such as
  `"1B - 2B"`.
- Give the user the SEC filing `url` for each agreement you cite.

## Feedback

`submit_feedback` sends a note to the Pandects maintainer. Use it only when the user asks to
report a problem or agrees to your offer to report one.
