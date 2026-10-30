from __future__ import annotations

# The `instructions` string returned by `initialize`.
#
# Hosts that use progressive discovery (tool search) put only this text and the bare
# tool names in front of the model at session start; full tool definitions load on
# demand. So this text has to do two jobs: say when to go looking for these tools at
# all, and map each kind of task to the tool names worth loading. Claude Code cuts
# server instructions past 4,096 characters, and a test holds it under that and
# checks that every tool is named here.
SERVER_INSTRUCTIONS = """\
Pandects: research tools over a corpus of SEC-filed M&A agreements (merger agreements and \
other acquisition agreements filed as EDGAR exhibits). Covers agreement metadata (parties, \
law-firm counsel, industry, deal size, consideration, dates), clause-level section text, a \
clause-family taxonomy, and extracted tax clauses. Search for these tools whenever a task \
involves merger or acquisition agreements, deal terms, clause language or drafting \
precedent (MAE, no-shop, fiduciary out, termination fees, efforts covenants, \
indemnification, tax structure), which law firms advised on deals, or M&A deal trends.

Each tool's description says when to use it and what to avoid, so you can load just the \
tools a task needs; to load one, search for its bare name (e.g. search_sections). Tools by \
job:
- Concept to taxonomy id: suggest_clause_families (taxonomy='tax_clauses' for tax \
concepts), get_clause_taxonomy, get_tax_clause_taxonomy.
- Valid filter values: list_filter_options (counsel, industries, deal types, price \
buckets), get_counsel_catalog (law firms), get_naics_catalog (industry codes).
- Find agreements: search_agreements (exploratory), list_agreements (exact filters, \
cursor paging, exports), get_agreement, get_agreements_batch.
- Find and read clause language: search_sections (text_query and/or standard_id; \
include_snippet=true returns the language), list_agreement_sections, \
list_agreement_sections_batch, get_section, get_sections_batch, get_section_snippet, \
get_section_snippets_batch.
- Tax clauses: search_tax_clauses, get_agreement_tax_clauses, get_section_tax_clauses.
- Corpus context: get_agreements_summary (corpus size), get_agreement_trends.
- Reference: get_server_capabilities (compact index by default; field_inventory, \
concept_notes and full tool records on request), get_server_metrics.
- Feedback: submit_feedback.

Typical clause question: suggest_clause_families -> search_sections with the returned \
standard_id, filters from list_filter_options, and include_snippet=true -> get_section \
for the full text.

Every deal has an SEC filer (usually a public party), so deals between two private \
companies are largely absent, and one deal can appear as several filings, so a filing \
count is not a deal count. Dataset and schema changes are summarized in the \
changelog section of get_server_capabilities; check it when results seem to have changed \
across sessions.

When you hit friction, missing capabilities, or data-quality issues, or when something \
works notably well, report it with submit_feedback as it happens, and consider a short \
wrap-up submission at the end of the session; the maintainer reads every report."""
