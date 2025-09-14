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
    start_token: int
    end_token: int
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


# ------------------------------
# Heuristics: diffuse vs localized
# ------------------------------


def _merge_token_spans(spans: List[UncertainSpan], gap: int) -> List[Tuple[int, int]]:
    """Merge uncertain token spans when they are within `gap` tokens."""
    if not spans:
        return []
    segs = sorted([(s.start_token, s.end_token) for s in spans])
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


def _coverage_metrics(
    uncertain: List[UncertainSpan], num_tokens: int
) -> Dict[str, float]:
    if num_tokens <= 0:
        return {"coverage": 0.0, "breadth": 0.0}
    total_uncertain = sum((s.end_token - s.start_token + 1) for s in uncertain)
    coverage = total_uncertain / max(1, num_tokens)
    min_i = min((s.start_token for s in uncertain), default=0)
    max_j = max((s.end_token for s in uncertain), default=0)
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

    merged_tok = _merge_token_spans(uncertain_spans, gap=cluster_gap_tokens)
    metrics = _coverage_metrics(uncertain_spans, num_tokens)
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


def _json_schema() -> Dict[str, Any]:
    """
    Minimal schema: entities = [{label, start_char, end_char, title}]
    Offsets are RELATIVE TO THE PROVIDED TEXT (full page or excerpt).
    """
    return {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "label": {
                            "type": "string",
                            "enum": ["article", "section", "page"],
                        },
                        "start_char": {"type": "integer", "minimum": 0},
                        "end_char": {"type": "integer", "minimum": 0},
                        "title": {"type": "string"},
                    },
                    "required": ["label", "start_char", "end_char", "title"],
                    "additionalProperties": False,
                },
            },
            "warnings": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["entities", "warnings"],
        "additionalProperties": False,
    }


def _system_prompt() -> str:
    return (
        "You are an expert legal tagging assistant. "
        "Your task is to identify ARTICLE and SECTION headings in pages and other excerpts of merger agreements. "
        "Tag only the heading spans; do not include body text. "
        "Return structured JSON per the schema. "
        "Do not hallucinate headings that are not present. "
        "Preserve input order; no reflow; no rewriting."
        "At the very end of *some* pages, there may be a number that corresponds to the page of the agreement. "
        "If you see a number at the end of a page and that number, in context, looks like it could be a page number, extract the page number section."
        "Some additional notes:\n"
        '1. Articles will always be preceeded by the word "Article" and may look like "Article I   Representations" or "Article 1  Warranties"\n'
        '2. Sections will often but not always be preceeded by the word "Section"; sometimes they will be just numbers, like "5.01   Company Representations", and sometimes they will just be numbers (which should be tagged) followed by the section body (which should not be tagged), like "5.01"\n'
        "3. If you are placing an <article> or <section> tag around long sentences, you're probably doing something wrong, like confusing the section body for the heading title. See #2, above.\n"
        '4. Sub-section do not count as sections, thus ignore headings like "9.1.4" or "9.1(a)". Do not splice tags into these; ignore them entirely. Tag only the section heading itself, in this case "9.1 [title text]," which would come at some point before "9.1.4"\n'
        '5. Sometimes there will be lots of extra spaces between the word "Section" or "Article" and the heading\'s title. This is fine and should not affect your decision to tag or not tag.\n'
    )


def _user_prompt_full(page_uuid: str, text: str) -> str:
    return (
        f"PAGE_UUID={page_uuid}\n"
        "Task: Extract ARTICLE and SECTION headings from the following page. "
        "Return 0-based character offsets relative to THIS TEXT.\n\n"
        f"{text}"
    )


def _user_prompt_excerpt(page_uuid: str, text_excerpt: str, base_offset: int) -> str:
    return (
        f"PAGE_UUID={page_uuid}\n"
        f"EXCERPT_BASE_CHAR_OFFSET={base_offset}\n"
        "Task: Extract ARTICLE and SECTION headings only within this excerpt. "
        "Return 0-based character offsets relative to THIS EXCERPT text (not page). "
        "We will translate to page coordinates using the base offset.\n\n"
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
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Returns:
      - lines: list of JSONL request objects
      - meta:  list of {request_id, page_uuid, mode, excerpt_start, excerpt_end}
    """
    schema = _json_schema()
    lines: List[Dict[str, Any]] = []
    metas: List[Dict[str, Any]] = []

    if decision.mode == "full":
        custom_id = f"{page_uuid}::full::0"
        body = {
            "model": model,
            "reasoning": {"effort": "medium"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "repair_response",
                    "strict": True,
                    "schema": schema,
                }
            },
            "instructions": _system_prompt(),
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
        body = {
            "model": model,
            "reasoning": {"effort": "medium"},
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "repair_response",
                    "strict": True,
                    "schema": schema,
                }
            },
            "instructions": _system_prompt(),
            "input": [
                {
                    "role": "user",
                    "content": _user_prompt_excerpt(page_uuid, excerpt, cs),
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
