"""Strip running page headers and footers from body pages before they are joined.

Body pages are concatenated into one text before sectioning, so a running header
or footer on each page (a "Table of Contents" back-link, a law-firm document ID,
an "EXECUTION VERSION" banner, a bare page number) ends up inside the served text
of whichever section spans that page boundary.

The strip is deliberately narrow. A line is removed only when all of these hold:

- the whole line (after trimming whitespace) matches one of the furniture
  patterns below; a line with any other words in it is never touched;
- it is one of the first or last ``EDGE_LINE_WINDOW`` non-empty lines of its
  page (``<page>`` page-number marker lines do not count toward the window);
- it is not inside an ``<article>`` or ``<section>`` heading tag;
- the same line shape (digits collapsed) sits at a page edge on at least
  ``MIN_REPEAT_PAGES`` body pages of the agreement, so a one-off line is kept.

Removed lines are dropped whole; no other text on the page changes.
"""

import re
from dataclasses import dataclass, field

EDGE_LINE_WINDOW = 3
MIN_REPEAT_PAGES = 3

_PAGE_MARKER_LINE_RE = re.compile(r"<page>.*?</page>")
_DIGITS_RE = re.compile(r"\d+")
_HEADING_OPEN_RE = re.compile(r"<(article|section)>", re.IGNORECASE)
_HEADING_CLOSE_RE = re.compile(r"</(article|section)>", re.IGNORECASE)

# Each pattern must match the whole trimmed line (re.fullmatch).
FURNITURE_PATTERNS: dict[str, re.Pattern[str]] = {
    # HTML filings: the running "Table of Contents" back-link.
    "toc_link": re.compile(
        r"table\s+of\s+contents|(?:back|return)\s+to\s+(?:the\s+)?(?:table\s+of\s+)?contents",
        re.IGNORECASE,
    ),
    # Law-firm document-management IDs.
    "doc_id": re.compile(
        r"\|*\s*(?:"
        # US-DOCS\123456789.12, ACTIVE/123456789.1, DOCS 123456.7
        r"(?:US-DOCS|DOCS|ACTIVE|LEGAL|DMS)\s?[\\/: ]\s?\d{5,}(?:[._]\d{1,4}){0,2}"
        # NY 12345678v3, DC 1234567.2, CHI:1234567 v2, H:836720 2
        r"|[A-Z]{1,6}\s?[: \\/]\s?\d{6,}(?:\s?v\.?\s?\d{1,4}|[._]\d{1,4}(?:\.\d{1,4})?)?(?:\s\d{1,3})?"
        # 33067829.14, 62323062_2, 1002446029v14, 249717839 v15, 51175145.1
        r"|\d{6,}(?:[._]\d{1,4}(?:\.\d{1,4})?|\s?v\.?\s?\d{1,4})(?:\s\d{1,3})?"
        # 041945-0274-Active.21403067.2
        r"|\d{3,}(?:-\d{2,}){1,3}-Active\.\d{5,}(?:\.\d{1,4})?(?:\s\d{1,3})?"
        r")\s*\|*"
    ),
    # Draft-status and confidentiality banners.
    "banner": re.compile(
        r"\**\s*(?:(?:final\s+)?execution(?:\s+(?:copy|version))?"
        r"|(?:strictly\s+)?confidential(?:\s+treatment\s+requested)?)\s*\**",
        re.IGNORECASE,
    ),
    # Page numbers the tagger did not wrap in <page>: 12, - 12 -, -12-, Page 12,
    # Page 12 of 80, 12/80, 12 | 80, A-12.
    "page_number": re.compile(
        r"\d{1,3}"
        r"|[-–—]\s*\d{1,3}\s*[-–—]"
        r"|page\s+\d{1,3}(?:\s+of\s+\d{1,3})?"
        r"|\d{1,3}\s?[/|]\s?\d{1,3}"
        r"|[A-Z]{1,2}-\d{1,3}",
        re.IGNORECASE,
    ),
    # 8-K exhibit label repeated on every page: Exhibit 2.1, EXHIBIT 10.1.
    "exhibit_label": re.compile(r"exhibit\s+\d{1,2}(?:\.\d{1,2})?", re.IGNORECASE),
}


@dataclass
class RemovedFurnitureLine:
    page_index: int
    line: str
    kind: str


@dataclass
class FurnitureStripResult:
    page_texts: list[str]
    removed: list[RemovedFurnitureLine] = field(default_factory=list)


def furniture_kind(line: str) -> str | None:
    """Return the furniture pattern name a whole trimmed line matches, else None."""
    stripped = line.strip()
    if not stripped or "<" in stripped or ">" in stripped:
        return None
    for kind, pattern in FURNITURE_PATTERNS.items():
        if pattern.fullmatch(stripped):
            return kind
    return None


def _line_shape(line: str) -> str:
    return _DIGITS_RE.sub("#", " ".join(line.split()))


def _lines_inside_headings(lines: list[str]) -> set[int]:
    """Indexes of lines that sit inside an open <article>/<section> heading tag."""
    inside: set[int] = set()
    depth = 0
    for index, line in enumerate(lines):
        if depth > 0:
            inside.add(index)
        depth += len(_HEADING_OPEN_RE.findall(line))
        depth -= len(_HEADING_CLOSE_RE.findall(line))
        depth = max(depth, 0)
        if depth > 0:
            inside.add(index)
    return inside


def _edge_candidates(text: str) -> list[tuple[int, str]]:
    """(line index, furniture kind) for edge lines of one page that match a pattern."""
    lines = text.split("\n")
    countable = [
        index
        for index, line in enumerate(lines)
        if line.strip() and not _PAGE_MARKER_LINE_RE.fullmatch(line.strip())
    ]
    if not countable:
        return []
    edge = set(countable[:EDGE_LINE_WINDOW]) | set(countable[-EDGE_LINE_WINDOW:])
    inside = _lines_inside_headings(lines)
    candidates: list[tuple[int, str]] = []
    for index in sorted(edge):
        if index in inside:
            continue
        kind = furniture_kind(lines[index])
        if kind is not None:
            candidates.append((index, kind))
    return candidates


def strip_page_furniture(page_texts: list[str]) -> FurnitureStripResult:
    """Remove repeated running headers/footers from the edges of an agreement's body pages.

    ``page_texts`` is the tagged text of each body page, in page order. Returns the
    texts with furniture lines dropped plus a record of each removed line.
    """
    per_page = [_edge_candidates(text) for text in page_texts]
    pages_by_shape: dict[str, set[int]] = {}
    for page_index, candidates in enumerate(per_page):
        lines = page_texts[page_index].split("\n")
        for line_index, _ in candidates:
            shape = _line_shape(lines[line_index])
            pages_by_shape.setdefault(shape, set()).add(page_index)
    repeated = {shape for shape, pages in pages_by_shape.items() if len(pages) >= MIN_REPEAT_PAGES}

    result = FurnitureStripResult(page_texts=[])
    for page_index, (text, candidates) in enumerate(zip(page_texts, per_page)):
        lines = text.split("\n")
        drop = {
            line_index: kind
            for line_index, kind in candidates
            if _line_shape(lines[line_index]) in repeated
        }
        if not drop:
            result.page_texts.append(text)
            continue
        kept: list[str] = []
        for line_index, line in enumerate(lines):
            if line_index in drop:
                result.removed.append(
                    RemovedFurnitureLine(page_index=page_index, line=line.strip(), kind=drop[line_index])
                )
                continue
            kept.append(line)
        result.page_texts.append("\n".join(kept))
    return result
