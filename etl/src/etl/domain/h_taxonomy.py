# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from typing import Any, Dict, Protocol, TypedDict
from typing_extensions import NotRequired


_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")

_TAXONOMY_LLM_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "section_uuid": {"type": "string"},
                    "categories": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["section_uuid", "categories"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["assignments"],
    "additionalProperties": False,
}


class TaxonomyInput(TypedDict):
    article_title: str
    section_title: str
    section_text: str


class TaxonomyPrediction(TypedDict):
    label: str
    alt_probs: NotRequired[list[float]]


class SectionIndex(TypedDict):
    section_uuid: str
    agreement_uuid: str


class TaxonomyPredictor(Protocol):
    def predict(self, rows: list[TaxonomyInput]) -> list[TaxonomyPrediction]: ...


class LoggerProtocol(Protocol):
    def info(self, msg: str) -> None: ...


class ContextProtocol(Protocol):
    log: LoggerProtocol


class TaxonomyRow(TypedDict):
    section_uuid: str
    agreement_uuid: str
    article_title: str | None
    section_title: str | None
    xml_content: str


class TaxonomyLLMRow(TypedDict):
    section_uuid: str
    agreement_uuid: str
    article_title_normed: str | None
    section_title_normed: str | None
    prev_article_title_normed: str | None
    prev_section_title_normed: str | None
    next_article_title_normed: str | None
    next_section_title_normed: str | None


def strip_xml_tags_to_text(xml_fragment: str) -> str:
    """Collapse an XML/HTML fragment to plain text."""
    if not xml_fragment:
        return ""
    no_tags = _TAG_RE.sub(" ", xml_fragment)
    collapsed = _SPACE_RE.sub(" ", no_tags)
    return collapsed.strip()


def serialize_taxonomy_labels(labels: list[str]) -> str:
    deduped_labels: list[str] = []
    seen: set[str] = set()
    for label in labels:
        cleaned = label.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        deduped_labels.append(cleaned)
    return json.dumps(deduped_labels, ensure_ascii=False)


def taxonomy_llm_response_schema() -> dict[str, Any]:
    return _TAXONOMY_LLM_RESPONSE_SCHEMA


def build_taxonomy_llm_instructions(taxonomy_json: list[dict[str, Any]]) -> str:
    taxonomy_json_text = json.dumps(taxonomy_json, ensure_ascii=False)
    return f"""
# Identity
You are an expert mergers and acquisitions (M&A) lawyer skilled at categorizing agreement sections from headings.

# Task
You will receive:
1) A taxonomy JSON with definitive categories nested up to 3 levels, each category having a stable "standard_id".
2) A list of sections. For each section you will receive:
   - section_uuid
   - section_title
   - article_title (the article/heading the section appears under, if any)
   - preceding/following section_title + article_title (if any)

Assign EACH section to one or more taxonomy categories using ONLY the provided titles/context (no full section text).

# Core rules
- Be judicious and discerning: do not treat categories too broadly, but also do not construe categories too narrowly.
- Always assign to the LOWEST available level (L3 if applicable; otherwise L2; never L1). Never return both a parent and its child. If an L3 applies, do not also include its L2.
- Return ONLY standard_id values that appear in the provided taxonomy JSON.
- If none apply or you are not confident, return an empty list [] for that section.

# Mandatory method (applies to EVERY section)
You MUST use this 3-step method for each section:

STEP 0 — Combine title signals (mandatory)
Treat (section_title + article_title + immediate neighbors' titles) as the evidence record.
- The section_title is primary evidence.
- The article_title is strong context for disambiguation (e.g., Conditions vs Covenants vs Termination vs Indemnification vs Reps).
- Neighboring titles are weak tie-breakers only; do not overfit to neighbors.

STEP 1 — Primary classification (1 best category)
Pick the single best-fitting category that captures the dominant subject indicated by the titles.

STEP 2 — Secondary-topic scan (up to 3 additional categories)
Add secondary categories ONLY if the titles clearly indicate distinct additional topics that a lawyer would expect to be searchable as separate headings.
Because you do NOT have the full text, be conservative: do not add secondary tags unless the topic is explicitly signaled in the titles
(e.g., the section title itself is compound like "Fees and Expenses; Brokers" or includes multiple canonical terms like "No-Shop; Fiduciary Out").

STEP 3 — Prune over-specific or speculative tags
Remove any category not directly supported by title language. Before finalizing each tag, ensure the section_title or article_title
contains clear keywords/terms of art that would convince another M&A lawyer the tag fits. If not, drop the tag.

# Evidence standards (hard constraints)
A) No inference from generic modifiers:
Do NOT select a narrow or jurisdiction-specific category based solely on generic words like:
"foreign", "governmental", "regulatory", "applicable law", "authority", "court", "agency", "filing", "approval".
These words can support broad categories by themselves, but they do NOT justify narrow subcategories unless accompanied by a named statute/authority/document/term of art.

B) Narrow-category threshold (titles-only):
Only assign a narrow subcategory when the titles contain one or more of the following:
- a named statute/regulation/rule/regime (e.g., "HSR", "DGCL", "Exchange Act", "ERISA", "CFIUS"),
- a named authority/body (e.g., "SEC", "DOJ", "FTC", "European Commission", "CMA"),
- a named instrument/document type (e.g., "Proxy Statement", "Schedule 14A", "Form S-4"),
- or an unmistakable term of art uniquely tied to that category (e.g., "go-shop", "reverse termination fee", "RWI", "paying agent", "appraisal rights").

C) Article-title disambiguation rules (hard)
Use the article_title to resolve common two-homes:
- If article_title indicates "Conditions" or "Conditions to Closing", prefer condition leaves (bring-down, performance, MAE, approvals, no injunction).
- If article_title indicates "Covenants", prefer covenant leaves (efforts, no-shop, fiduciary out, regulatory covenants, stockholder meeting/proxy covenants).
- If article_title indicates "Representations and Warranties", prefer rep leaves even if the section_title is short/generic (e.g., "Compliance with Law", "Litigation").
- If article_title indicates "Indemnification", prefer indemnity leaves (caps/baskets/survival/procedures/escrow/setoff/exclusive remedy).
- If article_title indicates "Termination", prefer termination leaves (events/outside date/fees/effects).
- If article_title indicates "Dispute Resolution" or "Enforcement", prefer forum/law/jury trial/specific performance/damages waiver leaves.

D) Compound-title rule (titles-only)
If a single section_title clearly contains multiple independent topics separated by ";" "/" "and" (e.g., "No-Shop; Fiduciary Out" or "Paying Agent; Withholding"),
you MAY assign multiple tags, but only when each topic maps cleanly to a distinct leaf in the taxonomy.

E) Disclosure schedule cross-reference language is usually NOT the main topic (titles-only)
Do not tag disclosure schedules/letters unless the section_title or article_title explicitly references "Disclosure Schedules", "Disclosure Letter", or "Interpretation".

F) Specific-over-general (hard)
When a specific leaf clearly matches due to a named statute/authority/document/term of art in the titles, you MUST include it and you MUST NOT replace it with a broader bucket.

G) Access vs. disclosure coordination (disambiguation)
Do NOT use access/inspection for titles that clearly relate to preparing transaction disclosure documents (proxy/prospectus/S-4/14A) or regulatory filings;
reserve access for diligence/inspection-type headings (books and records, premises, personnel, information access generally).

# Output format (STRICT)
Return a single JSON object and nothing else.
- Keys must be the section_uuid strings.
- Values must be arrays of standard_id strings.
- Do not include explanations, comments, or additional keys.
- Every input section_uuid must appear exactly once in the output.

# Formatting examples
- Exclusive Jurisdiction (L3):
  {{"<sectionUUID>": ["340731650c6c9cc1"]}}
- Multiple categories:
  {{"<sectionUUID>": ["9b7acdd48216f9c5", "428ed22cd51400a4"]}}

# Taxonomy
{taxonomy_json_text}

# Sections to map (below)
""".strip()


def build_taxonomy_prompt_payload(row: TaxonomyLLMRow) -> str:
    current_article = (row.get("article_title_normed") or "").strip() or "N/A"
    current_section = (row.get("section_title_normed") or "").strip() or "N/A"
    prev_article = (row.get("prev_article_title_normed") or "").strip() or "N/A"
    prev_section = (row.get("prev_section_title_normed") or "").strip() or "N/A"
    next_article = (row.get("next_article_title_normed") or "").strip() or "N/A"
    next_section = (row.get("next_section_title_normed") or "").strip() or "N/A"
    return (
        f"Section UUID: {row['section_uuid']}.\n"
        f"Section to map >> Article title: {current_article}. Section title: {current_section}.\n"
        f"Preceding section >> Article title: {prev_article}. Section title: {prev_section}.\n"
        f"Following section >> Article title: {next_article}. Section title: {next_section}.\n"
    )


def build_taxonomy_llm_request_body(
    *,
    custom_id: str,
    section_payloads: list[str],
    taxonomy_json: list[dict[str, Any]],
    model: str,
) -> dict[str, Any]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "reasoning": {"effort": "high"},
            "instructions": build_taxonomy_llm_instructions(taxonomy_json),
            "input": [{"role": "user", "content": "\n\n".join(section_payloads)}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "section_category_assignments",
                    "strict": True,
                    "schema": taxonomy_llm_response_schema(),
                }
            },
        },
    }


def parse_taxonomy_llm_response_text(raw_text: str) -> dict[str, list[str]]:
    obj = json.loads(raw_text)
    if not isinstance(obj, dict):
        raise ValueError("Response JSON is not an object.")

    if "assignments" in obj:
        assignments = obj.get("assignments")
        if not isinstance(assignments, list):
            raise TypeError("assignments must be a list.")
        assignments_out: dict[str, list[str]] = {}
        for item in assignments:
            if not isinstance(item, dict):
                raise TypeError("assignment items must be objects.")
            section_uuid = item.get("section_uuid")
            categories = item.get("categories")
            if not isinstance(section_uuid, str):
                raise TypeError("assignment.section_uuid must be a string.")
            if not isinstance(categories, list) or not all(
                isinstance(category, str) for category in categories
            ):
                raise TypeError("assignment.categories must be an array of strings.")
            assignments_out[section_uuid] = [
                category.strip() for category in categories if category.strip()
            ]
        return assignments_out

    out: dict[str, list[str]] = {}
    for section_uuid, categories in obj.items():
        if not isinstance(section_uuid, str):
            raise TypeError("section_uuid keys must be strings.")
        if not isinstance(categories, list) or not all(
            isinstance(category, str) for category in categories
        ):
            raise TypeError("section category values must be arrays of strings.")
        out[section_uuid] = [category.strip() for category in categories if category.strip()]
    return out


def predict_taxonomy(
    rows: list[TaxonomyRow],
    model: TaxonomyPredictor,
    context: ContextProtocol,
) -> tuple[list[SectionIndex], list[TaxonomyPrediction]]:
    """Prepare inputs and run taxonomy prediction for a set of sections."""
    inputs: list[TaxonomyInput] = []
    sec_idx: list[SectionIndex] = []
    for r in rows:
        text_block = strip_xml_tags_to_text(r["xml_content"])
        inputs.append(
            {
                "article_title": r.get("article_title") or "",
                "section_title": r.get("section_title") or "",
                "section_text": text_block,
            }
        )
        sec_idx.append(
            {
                "section_uuid": r["section_uuid"],
                "agreement_uuid": r["agreement_uuid"],
            }
        )

    context.log.info(f"Running taxonomy prediction on {len(inputs)} sections")
    preds = model.predict(inputs)
    return sec_idx, preds


def apply_standard_ids_to_xml(xml_str: str, section_uuid_to_label: Dict[str, str]) -> str:
    """Set standardId on <section> elements matching provided UUIDs."""
    if not section_uuid_to_label:
        return xml_str

    root = ET.fromstring(xml_str)
    for elem in root.iter("section"):
        su = elem.get("uuid")
        if su and su in section_uuid_to_label:
            elem.set("standardId", section_uuid_to_label[su])
    return ET.tostring(root, encoding="unicode")
