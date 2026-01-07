"""
Postprocessing utilities for ARTICLE entities.
"""

from __future__ import annotations

import re


_ARTICLE_PATTERNS = [
    re.compile(
        "^\\s*ARTICLE\\s+[IVXLCDM]+[\\.,:;\\-\\u2014]?\\s*(.*)?$",
        re.IGNORECASE,
    ),
    re.compile("^\\s*ARTICLE\\s+\\d+[\\.,:;\\-\\u2014]?\\s*(.*)?$", re.IGNORECASE),
]


def _line_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    line_start = text.rfind("\n", 0, max(start, 0))
    if line_start == -1:
        line_start = 0
    else:
        line_start += 1
    line_end = text.find("\n", max(end, 0))
    if line_end == -1:
        line_end = len(text)
    return line_start, line_end


def _span_char_bounds(
    span: tuple[int, int, str], token_offsets: list[tuple[int, int]]
) -> tuple[int, int]:
    start_idx, end_idx, _ = span
    if start_idx < 0 or end_idx >= len(token_offsets):
        raise ValueError("Span indices out of bounds for token offsets.")
    start_char = token_offsets[start_idx][0]
    end_char = token_offsets[end_idx][1]
    return start_char, end_char


def _line_text_for_span(
    span: tuple[int, int, str], raw_text: str, token_offsets: list[tuple[int, int]]
) -> tuple[int, int, str]:
    start_char, end_char = _span_char_bounds(span, token_offsets)
    line_start, line_end = _line_bounds(raw_text, start_char, end_char)
    return line_start, line_end, raw_text[line_start:line_end].rstrip("\r")


def apply_article_regex_gating(
    spans: list[tuple[int, int, str]],
    raw_text: str,
    token_offsets: list[tuple[int, int]],
) -> list[tuple[int, int, str]]:
    """
    Drop ARTICLE spans whose line does not match an ARTICLE heading pattern.
    """
    filtered: list[tuple[int, int, str]] = []
    for span in spans:
        if span[2] != "ARTICLE":
            filtered.append(span)
            continue
        _, _, line_text = _line_text_for_span(span, raw_text, token_offsets)
        if any(pat.match(line_text) for pat in _ARTICLE_PATTERNS):
            filtered.append(span)
    return filtered


def apply_article_line_snapping(
    spans: list[tuple[int, int, str]],
    raw_text: str,
    token_offsets: list[tuple[int, int]],
) -> list[tuple[int, int, str]]:
    """
    Expand ARTICLE spans to cover the full line containing the heading.
    """
    snapped: list[tuple[int, int, str]] = []
    for span in spans:
        if span[2] != "ARTICLE":
            snapped.append(span)
            continue
        line_start, line_end, _ = _line_text_for_span(span, raw_text, token_offsets)
        indices = [
            i
            for i, (s, e) in enumerate(token_offsets)
            if s < line_end and e > line_start
        ]
        if not indices:
            snapped.append(span)
            continue
        snapped.append((min(indices), max(indices), span[2]))
    return snapped
