# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false
from __future__ import annotations

import json
import re
import uuid
from html import unescape
from typing import Any, TypedDict


_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_TEXT_BLOCK_RE = re.compile(r"<text>(.*?)</text>", re.DOTALL | re.IGNORECASE)
_TOP_LEVEL_ENUMERATOR_RE = re.compile(r"^\(\s*([a-z])\s*\)\s*", re.IGNORECASE)
_REP_CONTEXT_RE = re.compile(r"representations|warranties", re.IGNORECASE)
_TAX_SIGNAL_RE = re.compile(
    r"(?<![a-z0-9])(?:"
    + r"tax|taxes|taxable|taxation|tax-free|"
    + r"withholding|backup\s+withholding|firpta|"
    + r"338|336|754|83\(b\)|382|383|280g|409a|"
    + r"cfc|pfic|gilti|subpart\s+f|"
    + r"transfer\s+tax|transfer\s+taxes|stamp\s+tax|stamp\s+taxes|sales\s+tax|sales\s+taxes|use\s+tax|use\s+taxes|"
    + r"conveyance\s+tax|conveyance\s+taxes|gains\s+tax|gains\s+taxes|vat|gst/?hst|"
    + r"net\s+operating\s+loss|nols?|carryback|carryforward|"
    + r"straddle\s+period|straddle\s+tax\s+period|refunds?|tax\s+credits?|refunds?\s+and\s+credits?|"
    + r"tax\s+contest|contests?\s+related\s+to\s+taxes|audits?\s+and\s+contests?\s+with\s+respect\s+to\s+taxes|"
    + r"reorganization|tax-free\s+reorganization|368\(a\)|section\s+368|"
    + r"purchase\s+price\s+allocation|allocation\s+of\s+purchase\s+price|"
    + r"entity\s+classification\s+elections?|push-out\s+elections?|"
    + r"golden\s+parachute|parachute\s+payments?|cleansing\s+vote|"
    + r"tax\s+sharing|tax-sharing|"
    + r"irs\s+ruling|israeli\s+tax\s+ruling|ruling\s+request|"
    + r"reit|ric|s\s+corporation"
    + r")(?![a-z0-9])",
    re.IGNORECASE,
)
_SECTION_NUMBER_PREFIX_RE = re.compile(
    r"^\s*(?:section|sec\.?)\s+\d+(?:\.\d+)+\.?\s*",
    re.IGNORECASE,
)
_GENERIC_TAX_CONTAINER_TITLES = {
    "tax",
    "taxes",
    "tax matters",
    "certain tax matters",
    "tax covenants",
    "tax provisions",
    "tax agreements",
    "tax matters and tax covenants",
}


class TaxSectionRow(TypedDict):
    agreement_uuid: str
    section_uuid: str
    article_title: str | None
    article_title_normed: str | None
    section_title: str | None
    section_title_normed: str | None
    xml_content: str
    xml_version: int | None
    section_standard_id: object | None
    section_standard_id_gold_label: object | None


class TaxClauseRecord(TypedDict):
    clause_uuid: str
    agreement_uuid: str
    section_uuid: str
    xml_version: int | None
    module: str
    clause_order: int
    anchor_label: str | None
    start_char: int
    end_char: int
    clause_text: str
    source_method: str
    context_type: str


class TaxAssignmentRecord(TypedDict):
    clause_uuid: str
    standard_id: str
    is_gold_label: int
    model_name: str | None


class TaxModuleLLMRow(TypedDict):
    clause_uuid: str
    agreement_uuid: str
    section_uuid: str
    article_title: str | None
    section_title: str | None
    clause_text: str
    anchor_label: str | None
    source_method: str
    context_type: str


def strip_tags_to_text(xml_fragment: str) -> str:
    no_tags = _TAG_RE.sub(" ", xml_fragment)
    return _SPACE_RE.sub(" ", unescape(no_tags)).strip()


def parse_standard_ids(raw_value: object) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]
    if isinstance(raw_value, str):
        stripped = raw_value.strip()
        if stripped == "":
            return []
        parsed = json.loads(stripped)
        if not isinstance(parsed, list):
            raise ValueError("Expected JSON list of standard IDs.")
        return [str(item).strip() for item in parsed if str(item).strip()]
    raise TypeError(f"Unsupported standard-id payload type: {type(raw_value)!r}")


def context_type_for_article(article_title_normed: str | None) -> str:
    normalized = (article_title_normed or "").strip()
    if _REP_CONTEXT_RE.search(normalized):
        return "rep_warranty"
    return "operative"


def has_tax_signal(value: str | None) -> bool:
    return bool(_TAX_SIGNAL_RE.search(value or ""))


def _generic_tax_container_title(value: str | None) -> bool:
    normalized = strip_tags_to_text(value or "").lower()
    normalized = _SECTION_NUMBER_PREFIX_RE.sub("", normalized)
    normalized = re.sub(r"^[\d.]+\s*", "", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip(" .:-\u2013\u2014")
    return normalized in _GENERIC_TAX_CONTAINER_TITLES


def is_topic_specific_tax_section(row: TaxSectionRow) -> bool:
    section_title = row.get("section_title") or row.get("section_title_normed")
    if not has_tax_signal(section_title):
        return False
    return not _generic_tax_container_title(section_title)


def is_tax_related_section(row: TaxSectionRow, *, tax_standard_ids: set[str]) -> bool:
    section_title_normed = (row.get("section_title_normed") or "").strip().lower()
    article_title_normed = (row.get("article_title_normed") or "").strip().lower()
    section_title = row.get("section_title") or section_title_normed
    article_title = row.get("article_title") or article_title_normed
    if (
        "tax" in section_title_normed
        or "tax" in article_title_normed
        or has_tax_signal(section_title)
        or has_tax_signal(article_title)
        or has_tax_signal(row.get("xml_content"))
    ):
        return True

    standard_ids = (
        parse_standard_ids(row.get("section_standard_id_gold_label"))
        or parse_standard_ids(row.get("section_standard_id"))
    )
    return any(standard_id in tax_standard_ids for standard_id in standard_ids)


class _Block(TypedDict):
    start_char: int
    end_char: int
    raw_text: str
    plain_text: str
    enumerator: str | None


class _ClauseCandidate(TypedDict):
    anchor_label: str | None
    start_char: int
    end_char: int
    clause_text: str
    source_method: str


def _top_level_enumerator(plain_text: str) -> str | None:
    match = _TOP_LEVEL_ENUMERATOR_RE.match(plain_text)
    if match is None:
        return None
    return match.group(1).lower()


def _next_letter(current_letter: str) -> str | None:
    if len(current_letter) != 1 or not current_letter.isalpha():
        return None
    lower = current_letter.lower()
    if lower == "z":
        return None
    return chr(ord(lower) + 1)


def _text_blocks(xml_content: str) -> list[_Block]:
    blocks: list[_Block] = []
    for match in _TEXT_BLOCK_RE.finditer(xml_content):
        raw_text = match.group(1)
        plain_text = strip_tags_to_text(raw_text)
        if plain_text == "":
            continue
        blocks.append(
            {
                "start_char": match.start(),
                "end_char": match.end(),
                "raw_text": raw_text,
                "plain_text": plain_text,
                "enumerator": _top_level_enumerator(plain_text),
            }
        )
    return blocks


def _whole_section_clause(row: TaxSectionRow, *, source_method: str) -> list[TaxClauseRecord]:
    xml_content = row["xml_content"]
    clause_text = strip_tags_to_text(xml_content)
    if clause_text == "":
        return []
    return [
        {
            "clause_uuid": _clause_uuid(
                agreement_uuid=row["agreement_uuid"],
                section_uuid=row["section_uuid"],
                xml_version=row.get("xml_version"),
                clause_order=1,
                anchor_label=None,
                clause_text=clause_text,
            ),
            "agreement_uuid": row["agreement_uuid"],
            "section_uuid": row["section_uuid"],
            "xml_version": row.get("xml_version"),
            "module": "tax",
            "clause_order": 1,
            "anchor_label": None,
            "start_char": 0,
            "end_char": len(xml_content),
            "clause_text": clause_text,
            "source_method": source_method,
            "context_type": context_type_for_article(row.get("article_title_normed")),
        }
    ]


def _clause_uuid(
    *,
    agreement_uuid: str,
    section_uuid: str,
    xml_version: int | None,
    clause_order: int,
    anchor_label: str | None,
    clause_text: str,
) -> str:
    seed = (
        f"tax|{agreement_uuid}|{section_uuid}|{xml_version}|"
        f"{clause_order}|{anchor_label or ''}|{clause_text}"
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def _continues_previous_paragraph(previous_text: str, current_text: str) -> bool:
    stripped_previous = previous_text.rstrip()
    stripped_current = current_text.lstrip()
    if stripped_previous == "" or stripped_current == "":
        return False
    if not stripped_previous.endswith((".", "?", "!", '"', "'", "\u201d", "\u2019", ")", "]")):
        return True
    first_char = stripped_current[0]
    return first_char.islower() or first_char in ",;:)]"


def _paragraph_clause_candidates(blocks: list[_Block]) -> list[_ClauseCandidate]:
    grouped_blocks: list[list[_Block]] = []
    current_group: list[_Block] = []
    for block in blocks:
        if not current_group:
            current_group = [block]
            continue
        if _continues_previous_paragraph(current_group[-1]["plain_text"], block["plain_text"]):
            current_group.append(block)
            continue
        grouped_blocks.append(current_group)
        current_group = [block]
    if current_group:
        grouped_blocks.append(current_group)

    candidates: list[_ClauseCandidate] = []
    for group in grouped_blocks:
        clause_text = "\n\n".join(block["plain_text"] for block in group).strip()
        if clause_text:
            candidates.append(
                {
                    "anchor_label": None,
                    "start_char": group[0]["start_char"],
                    "end_char": group[-1]["end_char"],
                    "clause_text": clause_text,
                    "source_method": "paragraph_split",
                }
            )
    return candidates


def _tax_clause_records_from_candidates(
    row: TaxSectionRow,
    candidates: list[_ClauseCandidate],
) -> list[TaxClauseRecord]:
    out: list[TaxClauseRecord] = []
    context_type = context_type_for_article(row.get("article_title_normed"))
    for clause_order, clause in enumerate(candidates, start=1):
        anchor_label = clause["anchor_label"]
        clause_text = clause["clause_text"]
        out.append(
            {
                "clause_uuid": _clause_uuid(
                    agreement_uuid=row["agreement_uuid"],
                    section_uuid=row["section_uuid"],
                    xml_version=row.get("xml_version"),
                    clause_order=clause_order,
                    anchor_label=anchor_label,
                    clause_text=clause_text,
                ),
                "agreement_uuid": row["agreement_uuid"],
                "section_uuid": row["section_uuid"],
                "xml_version": row.get("xml_version"),
                "module": "tax",
                "clause_order": clause_order,
                "anchor_label": anchor_label,
                "start_char": clause["start_char"],
                "end_char": clause["end_char"],
                "clause_text": clause_text,
                "source_method": clause["source_method"],
                "context_type": context_type,
            }
        )
    return out


def extract_tax_clauses(row: TaxSectionRow) -> list[TaxClauseRecord]:
    xml_content = row["xml_content"]
    blocks = _text_blocks(xml_content)

    if not blocks:
        return _whole_section_clause(row, source_method="whole_section_fallback")

    if is_topic_specific_tax_section(row):
        return _whole_section_clause(row, source_method="section_title_match")

    clauses: list[_ClauseCandidate] = []
    pending_blocks: list[_Block] = []
    current_blocks: list[_Block] = []
    current_anchor: str | None = None
    next_expected = "a"

    def flush_current() -> None:
        nonlocal current_blocks, current_anchor
        if not current_blocks:
            return
        clause_text = "\n\n".join(block["plain_text"] for block in current_blocks).strip()
        if clause_text == "":
            current_blocks = []
            current_anchor = None
            return
        clauses.append(
            {
                "anchor_label": None if current_anchor is None else f"({current_anchor})",
                "start_char": current_blocks[0]["start_char"],
                "end_char": current_blocks[-1]["end_char"],
                "clause_text": clause_text,
                "source_method": "enumerated_split",
            }
        )
        current_blocks = []
        current_anchor = None

    saw_top_level = False
    for block in blocks:
        enumerator = block["enumerator"]
        is_expected_top_level = enumerator is not None and enumerator == next_expected
        if is_expected_top_level:
            saw_top_level = True
            flush_current()
            if enumerator is None:
                raise ValueError("Expected top-level enumerator.")
            current_anchor = enumerator
            current_blocks = [*pending_blocks, block]
            pending_blocks = []
            next_expected = _next_letter(enumerator) or next_expected
            continue

        if not saw_top_level:
            pending_blocks.append(block)
            continue

        current_blocks.append(block)

    flush_current()

    if not saw_top_level or not clauses:
        paragraph_candidates = _paragraph_clause_candidates(blocks)
        if len(paragraph_candidates) <= 1:
            return _whole_section_clause(row, source_method="whole_section_fallback")
        return _tax_clause_records_from_candidates(row, paragraph_candidates)

    return _tax_clause_records_from_candidates(row, clauses)


_TAX_MODULE_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "assignments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "clause_uuid": {"type": "string"},
                    "categories": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["clause_uuid", "categories"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["assignments"],
    "additionalProperties": False,
}


def tax_module_response_schema() -> dict[str, Any]:
    return _TAX_MODULE_RESPONSE_SCHEMA


def build_tax_clause_llm_instructions(taxonomy_json: list[dict[str, Any]]) -> str:
    taxonomy_json_text = json.dumps(taxonomy_json, ensure_ascii=False)
    return f"""
# Identity
You are an expert M&A tax lawyer categorizing tax clauses in acquisition agreements.

# Task
You will receive:
1) A tax-clause taxonomy JSON with stable standard IDs nested up to 3 levels.
2) A list of extracted tax clauses. For each clause you will receive:
   - clause_uuid
   - section_title
   - article_title
   - context_type (`operative` or `rep_warranty`)
   - anchor_label
   - source_method (`section_title_match`, `enumerated_split`, `paragraph_split`, or `whole_section_fallback`)
   - clause_text

Assign EACH clause to zero or more taxonomy categories using ONLY the provided taxonomy.

# Core rules
- Always assign the lowest applicable level available. Never return both a parent and its child.
- Return only standard_id values from the provided taxonomy JSON.
- If no taxonomy leaf clearly fits, return an empty list.
- Be conservative with `rep_warranty` clauses. These often describe disclosure state rather than operative tax mechanics.
- Do not ignore `rep_warranty` clauses. Tag them when the clause text clearly fits a tax subject in the taxonomy.
- Favor operative tax mechanics for `operative` clauses, including transfer taxes, withholding, tax sharing, tax refunds, tax elections, tax treatment, tax cooperation, and indemnification-related tax mechanics.
- If source_method is `section_title_match`, treat the section title as strong evidence that the entire extracted text belongs to that tax topic.
- If source_method is `paragraph_split` or `enumerated_split`, classify only the extracted paragraph/subclause, not the surrounding section as a whole.
- Do not penalize a paragraph for lacking a repeated tax term if the section_title or anchor_label clearly identifies the tax topic.

# Output format
Return a single JSON object and nothing else.
- Use the shape {{ "assignments": [{{"clause_uuid": "...", "categories": ["..."]}}] }}.
- Every input clause_uuid must appear exactly once.

# Taxonomy
{taxonomy_json_text}
""".strip()


def build_tax_clause_prompt_payload(row: TaxModuleLLMRow) -> str:
    return (
        f"Clause UUID: {row['clause_uuid']}.\n"
        f"Context type: {row['context_type']}.\n"
        f"Article title: {(row.get('article_title') or 'N/A').strip() or 'N/A'}.\n"
        f"Section title: {(row.get('section_title') or 'N/A').strip() or 'N/A'}.\n"
        f"Anchor label: {(row.get('anchor_label') or 'N/A').strip() or 'N/A'}.\n"
        f"Source method: {row['source_method']}.\n"
        f"Clause text: {row['clause_text']}\n"
    )


def build_tax_clause_llm_request_body(
    *,
    custom_id: str,
    clause_payloads: list[str],
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
            "instructions": build_tax_clause_llm_instructions(taxonomy_json),
            "input": [{"role": "user", "content": "\n\n".join(clause_payloads)}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "tax_clause_assignments",
                    "strict": True,
                    "schema": tax_module_response_schema(),
                }
            },
        },
    }


def parse_tax_clause_llm_response_text(raw_text: str) -> dict[str, list[str]]:
    obj = json.loads(raw_text)
    if not isinstance(obj, dict):
        raise ValueError("Response JSON is not an object.")
    assignments = obj.get("assignments")
    if not isinstance(assignments, list):
        raise ValueError("Missing assignments list.")
    parsed: dict[str, list[str]] = {}
    for assignment in assignments:
        if not isinstance(assignment, dict):
            raise ValueError("Assignment is not an object.")
        clause_uuid = assignment.get("clause_uuid")
        categories = assignment.get("categories")
        if not isinstance(clause_uuid, str):
            raise ValueError("clause_uuid must be a string.")
        if not isinstance(categories, list) or not all(
            isinstance(category, str) for category in categories
        ):
            raise ValueError("categories must be a list of strings.")
        parsed[clause_uuid] = [category.strip() for category in categories if category.strip()]
    return parsed
