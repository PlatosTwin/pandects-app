"""Plain-text excerpts of section XML, shared by the REST and MCP search surfaces."""

from __future__ import annotations

import re
from html import unescape

from backend.text_search import parse_text_search_terms

_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\s+")
# Page breaks inside section XML: `<page>69</page>` (the printed page number) and
# `<pageUUID>66af9b9a-...</pageUUID>` (the page id). Their content is not agreement
# text, so it is dropped with the element instead of leaking into excerpts as
# "... 69 66af9b9a-20dc-11f1-ac97-be44a5e954c8 ...".
_PAGE_MARKER_RE = re.compile(r"<(pageUUID|page)\b[^>]*?(?:/>|>.*?</\1\s*>)", re.DOTALL)


def _strip_tags(xml: str) -> str:
    return _TAG_RE.sub(" ", _PAGE_MARKER_RE.sub(" ", xml))


def extract_text_from_xml(xml: object) -> str:
    """Strip tags, page markers, and entities from section XML, collapsing whitespace."""
    if not isinstance(xml, str):
        return ""
    text_content = unescape(_strip_tags(xml))
    return _WHITESPACE_RE.sub(" ", text_content).strip()


def strip_xml_snippet(xml_value: object, *, max_chars: int = 220) -> str | None:
    """Leading excerpt of a section, or None when it has no text."""
    if not isinstance(xml_value, str) or not xml_value.strip():
        return None
    text_only = _strip_tags(xml_value)
    normalized = _WHITESPACE_RE.sub(" ", text_only).strip()
    if not normalized:
        return None
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 1].rstrip()}…"


def focused_snippet(
    text_content: str, *, focus_terms: list[str], max_chars: int
) -> tuple[str, list[str]]:
    """Excerpt centered on the earliest focus term, plus the terms that were found.

    Falls back to the leading excerpt when no focus term occurs in the text.
    """
    cleaned_text = _WHITESPACE_RE.sub(" ", text_content).strip()
    if cleaned_text == "":
        return "", []

    lowered_text = cleaned_text.lower()
    matched_terms: list[str] = []
    best_index: int | None = None
    for term in focus_terms:
        normalized_term = term.strip().lower()
        if normalized_term == "":
            continue
        index = lowered_text.find(normalized_term)
        if index >= 0:
            matched_terms.append(term)
            if best_index is None or index < best_index:
                best_index = index

    return _snippet_window(cleaned_text, best_index, max_chars), matched_terms


def _snippet_window(cleaned_text: str, focus_index: int | None, max_chars: int) -> str:
    if focus_index is None:
        snippet = cleaned_text[:max_chars].rstrip()
        if len(cleaned_text) > len(snippet):
            snippet = snippet.rstrip(" .,;:") + "..."
        return snippet

    start = max(focus_index - (max_chars // 3), 0)
    end = min(start + max_chars, len(cleaned_text))
    if end - start < max_chars and start > 0:
        start = max(end - max_chars, 0)
    snippet = cleaned_text[start:end].strip()
    if start > 0:
        snippet = "..." + snippet
    if end < len(cleaned_text):
        snippet = snippet.rstrip(" .,;:") + "..."
    return snippet


def _term_pattern(term: str) -> str:
    if term.endswith("*"):
        return re.escape(term[:-1]) + r"\w*"
    return re.escape(term)


def text_match_patterns(text_query: str, match_mode: str) -> list[re.Pattern[str]]:
    """Regexes locating a text query in extracted text, mirroring the search semantics.

    ``phrase`` yields one ordered pattern; ``all_terms``/``any_terms`` yield one per term.
    A trailing ``*`` matches any word continuation, as in the search itself.
    """
    terms = parse_text_search_terms(text_query)
    if match_mode == "phrase":
        body = r"\W+".join(_term_pattern(term) for term in terms)
        return [re.compile(rf"(?<!\w){body}(?!\w)", re.IGNORECASE)]
    return [
        re.compile(rf"(?<!\w){_term_pattern(term)}(?!\w)", re.IGNORECASE)
        for term in dict.fromkeys(terms)
    ]


def text_match_snippet(
    text_content: str, *, patterns: list[re.Pattern[str]], max_chars: int
) -> tuple[str, list[str]]:
    """Excerpt centered on the earliest text-query match, plus the matched spans.

    ``matched_terms`` holds the distinct spans as they appear in the text (for example
    ``"Termination Fee"``), so a client can highlight them without re-tokenizing.
    """
    cleaned_text = _WHITESPACE_RE.sub(" ", text_content).strip()
    if cleaned_text == "":
        return "", []
    best_index: int | None = None
    matched_terms: list[str] = []
    for pattern in patterns:
        found = pattern.search(cleaned_text)
        if found is None:
            continue
        span = found.group(0)
        if span not in matched_terms:
            matched_terms.append(span)
        if best_index is None or found.start() < best_index:
            best_index = found.start()
    return _snippet_window(cleaned_text, best_index, max_chars), matched_terms


__all__ = [
    "extract_text_from_xml",
    "focused_snippet",
    "strip_xml_snippet",
    "text_match_patterns",
    "text_match_snippet",
]
