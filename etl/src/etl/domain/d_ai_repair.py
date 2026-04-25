"""
Domain logic for AI repair of NER tags:
- Heuristics to choose full-page vs. excerpt windows
- Prompt construction (full/excerpt) + JSON schema
- JSONL line builders for OpenAI Batch
- Strict output parsing (no fallback/massage)
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

# ------------------------------
# Data models
# ------------------------------


@dataclass
class UncertainSpan:
    entity: str
    start_char: int  # inclusive, in ORIGINAL page text
    end_char: int    # exclusive, in ORIGINAL page text
    avg_confidence: float


@dataclass
class RepairDecision:
    """mode: 'full' or 'excerpt'; windows in PAGE char coords; token_map for reference."""

    mode: str
    windows: List[Tuple[int, int]]
    token_map: List[Tuple[int, int]]


# ------------------------------
# Token <-> char mapping
# ------------------------------

_TOKEN_RE = re.compile(r"\S+")


def _build_token_char_map(text: str) -> List[Tuple[int, int]]:
    """Map whitespace tokens (\\S+) to char spans."""
    return [(m.start(), m.end()) for m in _TOKEN_RE.finditer(text)]


def _tokens_to_chars(
    token_map: List[Tuple[int, int]], i: int, j: int
) -> Tuple[int, int]:
    """Convert token span [i, j] inclusive to character span [start, end)."""
    if not token_map:
        raise ValueError("token_map must be non-empty.")
    if i < 0 or j < i or j >= len(token_map):
        raise ValueError("Token span is out of bounds.")
    return (token_map[i][0], token_map[j][1])


def _chars_to_tokens(
    token_map: List[Tuple[int, int]], s_char: int, e_char: int
) -> Tuple[int, int]:
    """Convert character span [s_char, e_char) to inclusive token indices [i, j]."""
    if not token_map:
        raise ValueError("token_map must be non-empty.")
    if s_char < 0 or e_char < s_char:
        raise ValueError("Character span is invalid.")
    n = len(token_map)

    # Find first token whose end > s_char
    i = 0
    for t, (ts, te) in enumerate(token_map):
        if te > s_char:
            i = t
            break
    else:
        i = n - 1

    # Find last token whose start < e_char
    j = n - 1
    for t in range(n - 1, -1, -1):
        ts, te = token_map[t]
        if ts < e_char:
            j = t
            break

    if j < i:
        j = i
    return (i, j)


# ------------------------------
# Heuristics: diffuse vs localized
# ------------------------------


def _merge_token_index_spans(spans_tok: List[Tuple[int, int]], gap: int) -> List[Tuple[int, int]]:
    """Merge token-index spans [i,j] when they are within `gap` tokens."""
    if not spans_tok:
        return []
    segs = sorted(spans_tok)
    merged: List[Tuple[int, int]] = []
    cur_s, cur_e = segs[0]
    for s, e in segs[1:]:
        if s <= cur_e + gap:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _coverage_metrics_from_tokens(
    spans_tok: List[Tuple[int, int]], num_tokens: int
) -> Dict[str, float]:
    if num_tokens <= 0 or not spans_tok:
        return {"coverage": 0.0, "breadth": 0.0}
    total_uncertain = sum((e - s + 1) for s, e in spans_tok)
    coverage = total_uncertain / num_tokens
    min_i = min(s for s, _ in spans_tok)
    max_j = max(e for _, e in spans_tok)
    breadth = (max_j - min_i + 1) / num_tokens
    return {"coverage": coverage, "breadth": breadth}


def decide_repair_windows(
    text: str,
    uncertain_spans: List[UncertainSpan],
    *,
    cluster_gap_tokens: int = 20,
    pre_tokens: int = 50,
    post_tokens: int = 50,
    coverage_threshold: float = 0.15,
    breadth_threshold: float = 0.6,
    cluster_threshold: int = 4,
) -> RepairDecision:
    """
    Decide whether to send the full page or windows around uncertain spans.
    Returns character windows in PAGE coordinates (for excerpt mode).
    """
    token_map = _build_token_char_map(text)
    num_tokens = len(token_map)

    if not uncertain_spans:
        return RepairDecision(mode="excerpt", windows=[], token_map=token_map)
    if num_tokens == 0:
        raise ValueError("Cannot build repair windows with empty token map.")

    # Convert char spans to token-index spans for clustering/metrics
    tok_spans: List[Tuple[int, int]] = []
    for s in uncertain_spans:
        i, j = _chars_to_tokens(token_map, int(s.start_char), int(s.end_char))
        tok_spans.append((i, j))
    merged_tok = _merge_token_index_spans(tok_spans, gap=cluster_gap_tokens)
    metrics = _coverage_metrics_from_tokens(tok_spans, num_tokens)
    n_clusters = len(merged_tok)

    send_full = (
        metrics["coverage"] >= coverage_threshold
        or metrics["breadth"] >= breadth_threshold
        or n_clusters >= cluster_threshold
        or num_tokens < 120
    )
    if send_full:
        return RepairDecision(
            mode="full", windows=[(0, len(text))], token_map=token_map
        )

    # Excerpt windows with pre/post expansion; then merge overlaps in char space
    char_windows: List[Tuple[int, int]] = []
    for i, j in merged_tok:
        i2 = max(0, i - pre_tokens)
        j2 = min(num_tokens - 1, j + post_tokens)
        cs, ce = _tokens_to_chars(token_map, i2, j2)
        char_windows.append((cs, ce))

    char_windows.sort()
    merged_char: List[Tuple[int, int]] = []
    cur_s, cur_e = char_windows[0]
    for s, e in char_windows[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged_char.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged_char.append((cur_s, cur_e))

    return RepairDecision(mode="excerpt", windows=merged_char, token_map=token_map)


# ------------------------------
# Prompts & JSON schema
# ------------------------------


def _json_schema_full_tagged_text() -> Dict[str, Any]:
    """
    Full-page mode: return exact source spans to tag and optional warnings.
    """
    return {
        "type": "object",
        "properties": {
            "spans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_char": {"type": "integer", "minimum": 0},
                        "end_char": {"type": "integer", "minimum": 0},
                        "label": {
                            "type": "string",
                            "enum": ["article", "section", "page"],
                        },
                        "selected_text": {"type": "string"},
                    },
                    "required": ["start_char", "end_char", "label", "selected_text"],
                    "additionalProperties": False,
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["spans", "warnings"],
        "additionalProperties": False,
    }


def _json_schema_excerpt_rulings() -> Dict[str, Any]:
    """
    Excerpt mode: only return rulings for provided candidate spans.
    Offsets are RELATIVE TO THE EXCERPT text.
    """
    return {
        "type": "object",
        "properties": {
            "rulings": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "start_char": {"type": "integer", "minimum": 0},
                        "end_char": {"type": "integer", "minimum": 0},
                        "label": {
                            "type": "string",
                            "enum": ["article", "section", "page", "none"],
                        },
                    },
                    "required": ["start_char", "end_char", "label"],
                    "additionalProperties": False,
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["rulings", "warnings"],
        "additionalProperties": False,
    }


def _json_schema_source_text_verdict() -> Dict[str, Any]:
    """
    Agreement-level source-text verdict for hard XML failures.

    This is intentionally separate from tag-span repair: the model must decide
    whether the source agreement text itself has a numbering/structure defect
    that should bypass hard validation after normal retagging has failed.
    """
    return {
        "type": "object",
        "properties": {
            "verdict": {
                "type": "string",
                "enum": [
                    "source_text_hard_rule_exception",
                    "tagging_error_or_missing_tag",
                    "insufficient_context",
                ],
            },
            "source_text_is_correct": {"type": "boolean"},
            "flagged_pages_are_causal": {"type": "boolean"},
            "saw_full_problem_text": {"type": "boolean"},
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
            "problem_summary": {"type": "string"},
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "page_uuid": {"type": "string"},
                        "quote": {"type": "string"},
                        "interpretation": {"type": "string"},
                    },
                    "required": ["page_uuid", "quote", "interpretation"],
                    "additionalProperties": False,
                },
            },
            "missing_or_duplicate_section_numbers": {
                "type": "array",
                "items": {"type": "string"},
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "verdict",
            "source_text_is_correct",
            "flagged_pages_are_causal",
            "saw_full_problem_text",
            "confidence",
            "problem_summary",
            "evidence",
            "missing_or_duplicate_section_numbers",
            "warnings",
        ],
        "additionalProperties": False,
    }


def _system_prompt_full(toc_context: str | None = None) -> str:
    prompt = '''
    # Identity

    You are a professional data tagger. You will be reviewing and tagging individual pages from Merger & Acquisition agreements.

    # Instructions

    First things first, if the page appears to be: 1) front matter; 2) part of the Table of Contents; 3) part of any signatures block sections; 4) part of a disclosure schedule in the Exhibits section; or 5) otherwise not from the main agreement, do not add tags. This will be the case rarely, if at all. Note that the recitals section IS part of the main agreement and should NOT be skipped; the recitals section often has "WHEREAS" clauses or "W I T N E S E T H" or a general description of the transaction and the parties.

    If the page IS from the main agreement, then your task is to identify the exact character spans that should be wrapped in tags. Use label "section" for section HEADINGS (the number and title), "article" for article HEADINGS (the number and title), and "page" for a page number at the very end of the page when applicable. Articles are higher in the hierarchy than sections, and you are to ignore sub-sections. The spans should encompass only the number AND heading title--NOT the body text of the section or article itself. References to sections and article--rather than headings--should never be tagged. Once again, your task is to tag HEADINGS; you will ignore REFERENCES. Be attentive to context to ensure accuracy.

    Precision matters more than recall. False positives are worse than missed spans. If you are uncertain whether text is truly a heading, do not tag it.

    # Output invariants (MANDATORY)
    1. Use the provided page text as the base string.
    2. Return only JSON matching the schema: a list of spans and warnings.
    3. Every span must use 0-based character offsets into the ORIGINAL page text, with end_char exclusive.
    4. For every span, selected_text must equal the exact substring from the original page text at [start_char:end_char].
    5. Do not add overlapping spans or nested spans.
    6. Do not return prompt or metadata text in any selected_text value.
    7. If no tags apply, return an empty spans list.
    8. Only tag text that is physically present on this page exactly as shown. Do not infer, reconstruct, or borrow headings from prior or subsequent pages.
    9. Before emitting a span, verify that selected_text is a contiguous substring from this exact page text and that the offsets point to that text. If you are unsure about the exact boundaries, omit the span.

    Finally, at the very end of some pages, there may be a number that corresponds to the page of the agreement. If you see a number at the end of a page and that number, in context, looks like it could be a page number, return a span with label "page" for that number.

    Some additional notes:
    1. Articles will almost always be preceded by the word "Article" and may look like "Article I   Representations" or "Article 1  Warranties"
    2. Sections will often but not always be preceded by the word "Section"; sometimes they will be just numbers, like "5.01   Company Representations", and sometimes they will just be numbers (which should be tagged) followed by the section body (which should not be tagged), like "5.01"
    2.1. Bare numeric headings like "10.1" or "12.2" should be tagged only when they appear as the heading at the start of a new block or paragraph.
    2.2. Do not tag bare numbers that merely appear inside a continuing paragraph or sentence.
    3. If you are placing an <article> or <section> tag around long sentences, you're probably doing something wrong, like confusing the section body for the heading title. See #2, above, and the second-to-last example, below.
    4. Sub-sections do not count as sections, thus ignore headings like "9.1.4" or "9.1(a)". Do not splice tags into these; ignore them entirely. Tag only the section heading itself, in this case "9.1 [title text]," which would come at some point before "9.1.4"
    5. Sometimes there will be lots of extra spaces between the word "Section" or "Article" and the heading's title. This is fine and should not affect your decision to tag or not tag.
    6. Sometimes you will encounter long sections of definitions, where terms in double quotes are juxtaposed to section or article references, such as in: “Disposition Actions”\n\nSection 8(d)\n\n.
    6.1. Context should enable you to distinguish these long definitions pages from the Table of Contents; that is, definitions sections are almost always in the main body of agreements and thus should almost never be skipped.
    6.2. Context should also help you avoid mistakenly tagging article or section references in these definitions pages. A crude rule of thumb is that, if you are tagging an article or section and the text before and after it is in double quotes, you're probably doing something wrong.
    6.3. If the candidate text is surrounded by ordinary running prose, quotations, or inline sentence text, it is probably a reference rather than a heading.
    7. Do not hallucinate headings that are not present. Do not rewrite or reflow text; preserve all characters exactly and only insert tags.
    8. All-caps text should not be tagged unless it is clearly functioning as a heading block on this page.

    # Examples

    <page_snippet>
    ... any Ancillary Agreement or the transactions contemplated hereby or thereby. 10.19 Mutual Drafting. This Agreement and the Ancillary Agreements shall be deemed...
    <page_snippet>

    <assistant_response>
    {"spans":[{"start_char":64,"end_char":87,"label":"section","selected_text":"10.19 Mutual Drafting."}],"warnings":[]}
    </assistant_response>

    <page_snippet>
    ... Acceptance Time and all filings and notifications described in Section 5.4(b) will have been made and any waiting periods thereunder will have terminated or expired...
    </page_snippet>

    <assistant_response>
    {"spans":[],"warnings":[]}
    </assistant_response>

    <page_snippet>
    NOW, THEREFORE, the parties hereto agree as follows:    ARTICLE 1    DEFINITIONS    Section 1.01. Definitions. (a) As used herein, the following terms have the following meanings:    “ 1934 Act ” means the Securities Exchange Act of 1934.
    </page_snippet>

    <assistant_response>
    {"spans":[
      {"start_char":52,"end_char":77,"label":"article","selected_text":"ARTICLE 1    DEFINITIONS"},
      {"start_char":81,"end_char":106,"label":"section","selected_text":"Section 1.01. Definitions."}
    ],"warnings":[]}
    </assistant_response>

    <page_snippet>
    ... 19.9 Representations of the Company 19.7.1 The Company hereby warrants and represents that it shall not modify its operations outside of the ordinary course of business during the time...
    </page_snippet>

    <assistant_response>
    {"spans":[{"start_char":4,"end_char":37,"label":"section","selected_text":"19.9 Representations of the Company"}],"warnings":[]}
    </assistant_response>

    <page_snippet>
    ... 10.1 The purchase price for the Purchased Assets and the Shares is (i) One Hundred and Twenty Million U.S. Dollars ($120,000,000),  plus  or  minus , as applicable...
    </page_snippet>

    <assistant_response>
    {"spans":[{"start_char":4,"end_char":8,"label":"section","selected_text":"10.1"}],"warnings":[]}
    </assistant_response>
    '''
    if toc_context is not None and toc_context.strip():
        prompt += (
            "\n\n# TOC context\n"
            f"{toc_context.strip()}\n"
            "Use this only as supporting evidence for numbering patterns. "
            "All offsets and selected_text values must still be based only on the provided page text. "
            "Do not infer or reconstruct missing headings from the TOC. "
            "If the TOC shows that a section number should appear around this page, use that as a cue to inspect the page carefully for a matching standalone heading block or bare numeric heading at the start of a new block. "
            "Pay especially close attention to text near the top of the page, immediately after visible line breaks, and at the start of indented or isolated blocks, because missing headings often appear there as short standalone numbers. "
            "If the exact section number appears on this page as a standalone block heading, even without a title, tag that exact on-page text as a section. "
            "If a candidate number appears alone on its own line or clearly starts a new block and the surrounding text is section body text, that is stronger evidence of a true heading than the same number appearing inline in prose. "
            "Do not tag inline prose references or numbers that are not functioning as headings on this page."
        )
    return prompt


def _system_prompt_excerpt() -> str:
    return (
        '''
        # Identity
        You are an expert legal tagging assistant specializing in M&A agreements.

        # Task
        * Your task is to review candidate spans from an NER model. Each span includes its start/end offsets relative to the excerpt, the actual text in the span, and the NER model's predicted label. The spans represent tokens, so some words may be split across multiple spans; this is normal.
        * Using the below entity identification rules and core principles, return a JSON object with "rulings" (each candidate plus a "label" from ["article","section","page","none"]) and "warnings" if needed.

        # Core principles
        1. Context over isolation: Always use surrounding text to decide the label for each candidate span. The candidate’s boundaries are immutable, but its label must reflect what that span represents in context.
        2. Offsets are authoritative (boundaries only): Do not expand, shrink, or merge spans. You only choose the label.
        3. Defer to the model only when context is inconclusive: If context clearly indicates a heading type, follow the context even if the model disagrees. Use the model’s predicted label as a tie-breaker when evidence is ambiguous.

        # Heading blocks (CRITICAL)
        Agreements often format ARTICLE headings as a multi-line block:
        - Line 1: "ARTICLE <number>."
        - Line 2+: one or more standalone title lines (often ALL CAPS), e.g. "MISCELLANEOUS", "DEFINITIONS".
        All such lines together form the ARTICLE heading block.

        When a candidate span is a standalone title line that is visually part of an ARTICLE heading block,
        label it "article" even if it does NOT include the word "Article" or a number.

        # Entity identification rules
        1. Headings only: Article and section entities consist of only the numbers and titles (not body text). A single logical heading may be split across multiple candidate spans.
        2. Hierarchy: Articles are higher-level than sections. Subsections (e.g., “9.1.4”, “9.1(a)”) are ignored.
        3. Article pattern:
        - The FIRST line of an ARTICLE heading block usually begins with “Article” or "ARTICLE" and a number.
        - Subsequent title lines in the same heading block may omit the word “Article” and the number; these lines are still labeled "article".
        4. Section pattern: Sections may begin with “Section” or just a number, e.g. "Section 5.01 Company Representations." >> "section"
        5. Ending punctuation: A final period (if present) IS part of the heading.
        6. References vs. headings: References within definitions, quotations, or prose are not entities.
        7. Whitespace: Extra spaces or line breaks between heading components do not affect the entity type.  
        Example: "Article II \n\nRepresentations \nand Warranties."
        8. Page numbers: A lone number at the very end of a page may represent a page number. Sometimes the page number will have a hyphen or other punctuation before or after it, in which case it should be included in the page entity.  
        Examples: "- 56 -" >> "page"; "56" >> "page".
        9. Sanity check: If a span reads like a full sentence or paragraph, it’s not a heading >> "none".

        # Article-block detection (use context)
        Label a span as "article" if ALL of the following are true:
        1) Nearby (typically within the previous few lines) there is an explicit "ARTICLE <number>" heading, AND
        2) The span is a standalone line (often ALL CAPS) appearing between that "ARTICLE <number>" line and the first numbered section (e.g., "11.1", "Section 11.1"), AND
        3) The span is not a full sentence or paragraph.

        Otherwise, do not label it "article" unless it itself matches the explicit article pattern.

        # Multiple spans per heading
        It is valid for multiple adjacent candidate spans to be labeled "article" if they are parts of the same ARTICLE heading block.
        Do not force a single span to represent the entire article heading.

        # Examples
        If the excerpt contains:

        "ARTICLE 11."
        "MISCELLANEOUS"
        "DEFINITIONS"
        "11.1 Defined Terms."

        Then:
        - "ARTICLE 11."        >> article
        - "MISCELLANEOUS"       >> article
        - "DEFINITIONS"         >> article
        - "11.1 Defined Terms." >> section
        '''
    )


def _user_prompt_full(page_uuid: str, text: str, toc_context: str | None = None) -> str:
    # Pass only the source page text so the model cannot echo prompt scaffolding
    # (e.g., PAGE_UUID/Task lines) into tagged_text.
    return text


def _user_prompt_excerpt(
    page_uuid: str,
    text_excerpt: str,
    base_offset: int,
    candidates: List[Tuple[int, int, str]],
) -> str:
    """
    candidates: list of (start_char, end_char, predicted_label) relative to the EXCERPT.
    """
    cand_with_text = [
        {"start_char": s, "end_char": e, "text": text_excerpt[s:e], "predicted_label": pl}
        for (s, e, pl) in candidates
    ]
    cand_json = json.dumps(cand_with_text, ensure_ascii=False)
    return (
        f"PAGE_UUID={page_uuid}\n"
        f"EXCERPT_BASE_CHAR_OFFSET={base_offset}\n"
        f"CANDIDATES={cand_json}\n"
        "Task: For each candidate span, in accordance with the entity identification rules and core principles you've been provided, decide if the span is a heading ('article','section','page') or 'none'. "
        "Return JSON with key 'rulings' (mirroring candidates with a 'label') and 'warnings'.\n\n"
        f"{text_excerpt}"
    )


def _system_prompt_source_text_verdict() -> str:
    return '''
    # Identity

    You are an expert legal agreement source-text auditor. You are not repairing tags in this task.

    # Task

    You will receive:
    1. Hard XML validation failures for one agreement.
    2. The source text and current tagged text for the pages believed to cause those failures.
    3. Neighboring pages for context when available.

    Decide whether the agreement failed hard validation because the SOURCE AGREEMENT TEXT itself has an unusual but genuine numbering/structure problem, rather than because the tagger missed or misplaced tags.

    # Verdicts

    Return "source_text_hard_rule_exception" only if all of these are true:
    - The flagged page(s) shown are enough to see the complete local problem.
    - The hard-validation failure is caused by text physically present in the source agreement.
    - The current source text is correct as printed, even though it violates the normal hard XML rule.
    - There is no plausible missing article/section heading on the shown pages that should be tagged to fix the failure.

    Return "tagging_error_or_missing_tag" if the shown source text contains an untagged or mistagged heading that should be repaired, or if the numbering looks normal once headings are correctly interpreted.

    Return "insufficient_context" if the shown pages do not let you confidently see the full local issue.

    # Rules

    - Review SOURCE text first. Tagged text is supporting evidence only.
    - Do not infer missing pages or missing headings from references in prose.
    - Section references inside prose are not section headings.
    - A true source-text exception can include duplicated section numbers, skipped section numbers, or a missing section heading where nearby source pages show the agreement truly jumps over it.
    - Be conservative. False bypasses are worse than leaving an agreement invalid.
    - If confidence is below 0.90, do not use "source_text_hard_rule_exception".

    # Output

    Return only JSON matching the schema.
    '''


def _user_prompt_source_text_verdict(
    *,
    agreement_uuid: str,
    xml_version: int,
    reason_rows: List[Dict[str, Any]],
    page_contexts: List[Dict[str, Any]],
) -> str:
    payload = {
        "agreement_uuid": agreement_uuid,
        "xml_version": xml_version,
        "hard_validation_failures": reason_rows,
        "pages": page_contexts,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


# ------------------------------
# JSONL line builders (OpenAI Batch)
# ------------------------------


def build_jsonl_lines_for_page(
    page_uuid: str,
    text: str,
    decision: RepairDecision,
    *,
    model: str,
    uncertain_spans: List[UncertainSpan],
    xml_version: int | None = None,
    toc_context: str | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      - lines: list of JSONL request objects
      - meta:  list of {request_id, page_uuid, mode, excerpt_start, excerpt_end}
    """
    schema_full = _json_schema_full_tagged_text()
    schema_excerpt = _json_schema_excerpt_rulings()
    lines: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []

    if decision.mode == "full":
        version_token = int(xml_version) if xml_version is not None else 0
        custom_id = f"{page_uuid}::full::{version_token}"
        body = {
            "model": model,
            "reasoning": {"effort": "high"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "full_page_tag_spans",
                    "strict": True,
                    "schema": schema_full,
                }
            },
            "instructions": _system_prompt_full(toc_context=toc_context),
            "input": [
                {
                    "role": "user",
                    "content": _user_prompt_full(page_uuid, text, toc_context=toc_context),
                }
            ],
        }
        lines.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
        )
        metas.append(
            {
                "request_id": custom_id,
                "page_uuid": page_uuid,
                "mode": "full",
                "excerpt_start": 0,
                "excerpt_end": len(text),
            }
        )
        return lines, metas

    # excerpt mode
    for k, (cs, ce) in enumerate(decision.windows):
        excerpt = text[cs:ce]
        custom_id = f"{page_uuid}::excerpt::{k}::{cs}::{ce}"

        # Restrict to candidates intersecting this excerpt and convert to excerpt-relative spans, preserving predicted label
        cand_excerpt: List[Tuple[int, int, str]] = []
        for span in uncertain_spans:
            ps = int(span.start_char)
            pe = int(span.end_char)
            if pe <= cs or ps >= ce:
                continue
            s_rel = max(ps, cs) - cs
            e_rel = min(pe, ce) - cs
            if e_rel > s_rel:
                cand_excerpt.append((s_rel, e_rel, str(span.entity)))

        body = {
            "model": model,
            "reasoning": {"effort": "high"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "excerpt_rulings",
                    "strict": True,
                    "schema": schema_excerpt,
                }
            },
            "instructions": _system_prompt_excerpt(),
            "input": [
                {
                    "role": "user",
                    "content": _user_prompt_excerpt(page_uuid, excerpt, cs, cand_excerpt),
                }
            ],
        }
        lines.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
        )
        metas.append(
            {
                "request_id": custom_id,
                "page_uuid": page_uuid,
                "mode": "excerpt",
                "excerpt_start": cs,
                "excerpt_end": ce,
            }
        )
    return lines, metas


def build_jsonl_line_for_source_text_verdict(
    *,
    agreement_uuid: str,
    xml_version: int,
    anchor_page_uuid: str,
    model: str,
    reason_rows: List[Dict[str, Any]],
    page_contexts: List[Dict[str, Any]],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    custom_id = f"{agreement_uuid}::source::{int(xml_version)}"
    body = {
        "model": model,
        "reasoning": {"effort": "high"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "source_text_hard_rule_verdict",
                "strict": True,
                "schema": _json_schema_source_text_verdict(),
            }
        },
        "instructions": _system_prompt_source_text_verdict(),
        "input": [
            {
                "role": "user",
                "content": _user_prompt_source_text_verdict(
                    agreement_uuid=agreement_uuid,
                    xml_version=xml_version,
                    reason_rows=reason_rows,
                    page_contexts=page_contexts,
                ),
            }
        ],
    }
    return (
        {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        },
        {
            "request_id": custom_id,
            "page_uuid": anchor_page_uuid,
            "mode": "source",
            "excerpt_start": 0,
            "excerpt_end": 0,
        },
    )
