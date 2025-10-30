"""
Domain logic for AI repair of NER tags:
- Heuristics to choose full-page vs. excerpt windows
- Prompt construction (full/excerpt) + JSON schema
- JSONL line builders for OpenAI Batch
- Strict output parsing (no fallback/massage)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

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
        return (0, 0)
    i = max(0, min(i, len(token_map) - 1))
    j = max(0, min(j, len(token_map) - 1))
    return (token_map[i][0], token_map[j][1])


def _chars_to_tokens(
    token_map: List[Tuple[int, int]], s_char: int, e_char: int
) -> Tuple[int, int]:
    """Convert character span [s_char, e_char) to inclusive token indices [i, j].
    Falls back to nearest tokens if boundaries land in whitespace.
    """
    if not token_map:
        return (0, 0)
    n = len(token_map)
    s_char = max(0, s_char)
    e_char = max(s_char, e_char)

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
    coverage = total_uncertain / max(1, num_tokens)
    min_i = min(s for s, _ in spans_tok)
    max_j = max(e for _, e in spans_tok)
    breadth = (max_j - min_i + 1) / max(1, num_tokens)
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

    if not uncertain_spans or num_tokens == 0:
        return RepairDecision(mode="excerpt", windows=[], token_map=token_map)

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
    Full-page mode: return fully tagged text (string) and optional warnings.
    """
    return {
        "type": "object",
        "properties": {
            "tagged_text": {"type": "string"},
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["tagged_text", "warnings"],
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


def _system_prompt_full() -> str:
    return (
        "You are an expert legal tagging assistant. "
        "Return a JSON object with 'tagged_text' containing the ORIGINAL page text with <article>, <section>, and <page> tags inserted around heading spans only. "
        "Do not rewrite or reflow text; preserve all characters exactly and only insert tags. "
        "Do not hallucinate headings that are not present. "
        "Include a 'warnings' array for any notes. "
        "At the very end of some pages, there may be a page number; if appropriate, wrap that number with <page>. "
        "Guidelines: Articles start with 'Article' (e.g., 'Article I', 'Article 1'); Sections often start with 'Section' or numeric like '5.01'; avoid tagging long sentences or subsection markers like '9.1.4' or '9.1(a)'."
    )


def _system_prompt_excerpt() -> str:
    return (
        '''
        # Identity
        You are an expert legal tagging assistant specializing in M&A agreements.
        
        # Task
        * Your task is to review low-confidence candidate spans from an NER model. Each span includes its start/end offsets relative to the excerpt, the actual text in the span, and the NER model's predicted label. The spans represent tokens, so some words may be split across multiple spans; this is normal.
        * Using the below entity identification rules and core principles, return a JSON object with "rulings" (each candidate plus a "label" from ["article","section","page","none"]) and "warnings" if needed.
        
        # Core principles
        1. Context over isolation: Always use surrounding text to decide the label for each candidate span. The candidate’s boundaries are immutable, but its label must reflect what that span represents in context.
        2. Offsets are authoritative (boundaries only): Do not expand, shrink, or merge spans. You only choose the label.
        3. Defer to the model only when context is inconclusive: If context clearly indicates a heading type, follow the context even if the model disagrees. Use the model’s predicted label as a tie-breaker when evidence is ambiguous.

        # Entity identification rules
        1. Headings only: Article and section entities consist of only the numbers and titles (e.g., "Article II Representations and Warranties."), not the body text.
        2. Hierarchy: Articles are higher-level than sections. Subsections (e.g., “9.1.4”, “9.1(a)”) are ignored.
        3. Article pattern: Articles almost always begins with “Article” or "ARTICLE", e.g. "Article II Representations and Warranties." >> "article"
        4. Section pattern: Sections may begin with “Section” or just a number, e.g. "Section 5.01 Company Representations." >> "section"
        5. Ending punctuation: A final period (if present) IS part of the heading.
        6. References vs. headings: References within definitions or quotations are not entities.
        7. Whitespace: Extra spaces between labels and titles do not affect the entity type. E.g., "Article II \n\nRepresentations \nand Warranties."
        8. Page numbers: A lone number at the very end of a page may represent a page number. Sometimes the page number will have a hyphen or other punctuation before or after it, in which case it should be included in the page entity. E.g., "- 56 -" >> "page"; "56" >> "page".
        9. Sanity check: If a span reads like a full sentence or paragraph, it’s not a heading >> "none".
        '''
    )


def _user_prompt_full(page_uuid: str, text: str) -> str:
    return (
        f"PAGE_UUID={page_uuid}\n"
        "Task: Insert <article>, <section>, and <page> tags into the exact original text below."
        " Return JSON with key 'tagged_text' containing the fully tagged text and 'warnings' as an array.\n\n"
        f"{text}"
    )


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
        custom_id = f"{page_uuid}::full::0"
        body = {
            "model": model,
            "reasoning": {"effort": "high"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "full_page_tagged_text",
                    "strict": True,
                    "schema": schema_full,
                }
            },
            "instructions": _system_prompt_full(),
            "input": [{"role": "user", "content": _user_prompt_full(page_uuid, text)}],
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


# ------------------------------
# Output parsing (STRICT)
# ------------------------------


def parse_batch_output_line(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Strictly parse a single JSONL 'raw' line from Responses API.
    Assumes:
      raw = {
        "custom_id": "...",
        "response": {
          "status_code": 200,
          "body": {
            "output": [
              {"type":"reasoning", ...},
              {"type":"message","content":[{"type":"output_text","text":"<JSON STRING>"}], "role":"assistant"}
            ]
          }
        }
      }
    Returns {"request_id": str, "page_uuid": "", "entities": List[Dict]]} or None on deviation.
    """
    try:
        rid = raw["custom_id"]
        resp = raw["response"]
        sc = resp.get("status_code")
        if sc not in (200, 201, 202):
            return None
        body = resp["body"]
        output = body["output"]
        msg_blocks = [o for o in output if o.get("type") == "message"]
        if not msg_blocks:
            return None
        contents = msg_blocks[0]["content"]
        text_items = [c for c in contents if isinstance(c, dict) and "text" in c]
        if not text_items:
            return None
        raw_text = text_items[0]["text"]
        obj = json.loads(raw_text)
        if not isinstance(obj, dict) or "entities" not in obj or "warnings" not in obj:
            return None
        ents = obj["entities"]
        if not isinstance(ents, list):
            return None
        
        return {"request_id": rid, "page_uuid": "", "entities": ents}
    except Exception:
        return None
