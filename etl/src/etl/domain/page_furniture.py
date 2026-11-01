"""Strip running page headers and footers from body pages before they are joined.

Body pages are concatenated into one text before sectioning, so a running header
or footer on each page (a "Table of Contents" back-link, a law-firm document ID,
an "EXECUTION VERSION" banner, a bare page number) ends up inside the served text
of whichever section spans that page boundary.

The strip is deliberately narrow. Every removed line must:

- match one of the furniture patterns below as a whole trimmed line; a line with
  any other words or with a tag in it is never touched;
- be one of the first or last ``EDGE_LINE_WINDOW`` non-empty lines of its page
  (lines that are only a ``<page>`` page-number marker do not count);
- not sit inside an ``<article>`` or ``<section>`` heading tag.

On top of that, each kind needs its own evidence that it is a running line:

- Table of Contents links, exhibit labels and marked document IDs (a letter
  prefix with a separator, a ``v`` version suffix, an underscore): the same
  line shape (digits collapsed) sits at a page edge on ``MIN_REPEAT_PAGES``
  pages. An unmarked ID (``33067829.14``, ``ASIA 34949044``) needs the exact
  same ID text on ``MIN_REPEAT_PAGES`` pages (a trailing page number may vary).
- Page numbers: the number must fit the agreement's page sequence. Candidates
  are chained across pages while the number rises and ``number - page index``
  stays within ``PAGE_OFFSET_TOLERANCE`` of the previous link; only chains of at least
  ``MIN_REPEAT_PAGES`` pages are stripped, at most one number per page edge, and
  never a number whose next non-blank line opens a heading without a number of
  its own (``12`` above ``<section>Miscellaneous</section>`` is a clause number).
- Banners (EXECUTION VERSION, CONFIDENTIAL): the shape repeats on
  ``MIN_REPEAT_PAGES`` pages and every non-empty line between the banner and the
  page edge is itself being stripped, so the banner is the outermost text.

Removed lines are dropped whole; no other text on the page changes.
"""

import re
from dataclasses import dataclass, field

EDGE_LINE_WINDOW = 3
MIN_REPEAT_PAGES = 3
PAGE_OFFSET_TOLERANCE = 2

# A line that is only a tagged page number ("<page>12</page>", "Page <page>12</page>").
_PAGE_MARKER_LINE_RE = re.compile(r"(?:page\s*)?<page>.*?</page>", re.IGNORECASE)
_DIGITS_RE = re.compile(r"\d+")
_HEADING_OPEN_RE = re.compile(r"<(article|section)>", re.IGNORECASE)
_HEADING_CLOSE_RE = re.compile(r"</(article|section)>", re.IGNORECASE)
_HEADING_START_RE = re.compile(r"^\s*<(article|section)>(?P<inner>[^<]*)", re.IGNORECASE)
# A heading carries its own number: digits, a Roman numeral, or a letter enumerator ("B.", "(a)").
_HEADING_HAS_NUMBER_RE = re.compile(r"\d|\b[IVXLC]+\b|^\s*(?:[A-Z]\.|\([A-Za-z]\))")

_CURRENCY_PREFIXES = r"(?!(?:USD|US\$|EUR|GBP|CAD|AUD|NZD|JPY|CHF|CNY|RMB|HKD|SGD|INR|BRL|MXN|SEK|NOK|DKK|ZAR)\b)"

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
        # US-DOCS\123456789.12, ACTIVE/123456789.1, ACTIVE 123456789v1
        r"(?:US-DOCS|DOCS|ACTIVE|LEGAL|DMS)\s?[\\/: ]\s?\d{5,}(?:\s?v\.?\s?\d{1,4}|[._]\d{1,4}){0,2}"
        # EAST\1234567.2, HOU:1234567.1, H:836720 2 (letter prefix with \ / : separator)
        + r"|" + _CURRENCY_PREFIXES + r"[A-Z]{1,6}\s?[\\/:]\s?\d{6,}(?:\s?v\.?\s?\d{1,4}|[._]\d{1,4}(?:\.\d{1,4})?)?(?:\s\d{1,3})?"
        # NY 12345678v3, ATL 1234567v2 (space separator with a version suffix)
        + r"|" + _CURRENCY_PREFIXES + r"[A-Z]{1,6}\s\d{6,}\s?v\.?\s?\d{1,4}(?:\s\d{1,3})?"
        # ASIA 34949044 5 (space separator, no version: the exact ID must repeat)
        + r"|" + _CURRENCY_PREFIXES + r"[A-Z]{2,6}\s\d{6,}(?:\s\d{1,3})?"
        # 1002446029v14, 249717839 v15, 62323062_2 (marked by v or underscore)
        r"|\d{6,}(?:\s?v\.?\s?\d{1,4}|_\d{1,4})(?:\s\d{1,3})?"
        # 33067829.14, 51175145.1 2 (purely numeric: exact text must repeat)
        r"|\d{6,}\.\d{1,4}(?:\.\d{1,4})?(?:\s\d{1,3})?"
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

# Doc IDs with no marker beyond digits must repeat with identical ID text; the
# group is the ID without any trailing page number.
_UNMARKED_DOC_ID_RE = re.compile(
    r"\|*\s*(?P<id>\d{6,}\.\d{1,4}(?:\.\d{1,4})?|[A-Z]{2,6}\s\d{6,})(?:\s\d{1,3})?\s*\|*"
)


@dataclass
class RemovedFurnitureLine:
    page_index: int
    line: str
    kind: str


@dataclass
class FurnitureStripResult:
    page_texts: list[str]
    removed: list[RemovedFurnitureLine] = field(default_factory=list)


@dataclass
class _Candidate:
    page_index: int
    line_index: int
    kind: str
    text: str
    side: str  # "top" or "bottom"
    rank: int  # 0 = outermost countable line on its side


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


def _repeat_key(candidate: _Candidate) -> str:
    if candidate.kind == "doc_id":
        unmarked = _UNMARKED_DOC_ID_RE.fullmatch(candidate.text)
        if unmarked is not None:
            return "exact:" + unmarked.group("id")
    return candidate.kind + ":" + _line_shape(candidate.text)


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


def _countable_indexes(lines: list[str]) -> list[int]:
    return [
        index
        for index, line in enumerate(lines)
        if line.strip() and not _PAGE_MARKER_LINE_RE.fullmatch(line.strip())
    ]


def _next_line_opens_unnumbered_heading(lines: list[str], index: int) -> bool:
    for line in lines[index + 1:]:
        if not line.strip():
            continue
        match = _HEADING_START_RE.match(line)
        if match is None:
            return False
        return _HEADING_HAS_NUMBER_RE.search(match.group("inner")) is None
    return False


def _page_candidates(page_index: int, text: str) -> list[_Candidate]:
    lines = text.split("\n")
    countable = _countable_indexes(lines)
    if not countable:
        return []
    inside = _lines_inside_headings(lines)
    positions: dict[int, tuple[str, int]] = {}
    for rank, index in enumerate(countable[:EDGE_LINE_WINDOW]):
        positions[index] = ("top", rank)
    for rank, index in enumerate(reversed(countable[-EDGE_LINE_WINDOW:])):
        if index not in positions or rank < positions[index][1]:
            positions[index] = ("bottom", rank)
    candidates: list[_Candidate] = []
    for index in sorted(positions):
        if index in inside:
            continue
        kind = furniture_kind(lines[index])
        if kind is None:
            continue
        if kind == "page_number" and _next_line_opens_unnumbered_heading(lines, index):
            continue
        side, rank = positions[index]
        candidates.append(_Candidate(page_index, index, kind, lines[index].strip(), side, rank))
    return candidates


def _repeated(candidates: list[_Candidate]) -> list[_Candidate]:
    pages_by_key: dict[str, set[int]] = {}
    for candidate in candidates:
        pages_by_key.setdefault(_repeat_key(candidate), set()).add(candidate.page_index)
    return [c for c in candidates if len(pages_by_key[_repeat_key(c)]) >= MIN_REPEAT_PAGES]


def _page_number_value(text: str) -> int:
    match = _DIGITS_RE.search(text)
    assert match is not None
    return int(match.group())


def _sequenced_page_numbers(candidates: list[_Candidate]) -> list[_Candidate]:
    """Keep page-number candidates that chain into a page sequence of MIN_REPEAT_PAGES pages."""
    accepted: list[_Candidate] = []
    by_shape: dict[str, list[_Candidate]] = {}
    for candidate in candidates:
        by_shape.setdefault(_line_shape(candidate.text), []).append(candidate)
    for group in by_shape.values():
        by_page: dict[int, list[_Candidate]] = {}
        for candidate in group:
            by_page.setdefault(candidate.page_index, []).append(candidate)
        chains: list[list[_Candidate]] = []
        for page_index in sorted(by_page):
            extended: set[int] = set()
            for candidate in sorted(by_page[page_index], key=lambda c: c.rank):
                offset = _page_number_value(candidate.text) - page_index
                best: int | None = None
                for chain_index, chain in enumerate(chains):
                    if chain_index in extended:
                        continue
                    last = chain[-1]
                    last_offset = _page_number_value(last.text) - last.page_index
                    value_rises = _page_number_value(candidate.text) > _page_number_value(last.text)
                    if value_rises and abs(offset - last_offset) <= PAGE_OFFSET_TOLERANCE and (
                        best is None or len(chain) > len(chains[best])
                    ):
                        best = chain_index
                if best is None:
                    chains.append([candidate])
                    extended.add(len(chains) - 1)
                else:
                    chains[best].append(candidate)
                    extended.add(best)
        for chain in chains:
            if len(chain) >= MIN_REPEAT_PAGES:
                accepted.extend(chain)
    # At most one page number per page edge: keep the outermost.
    best_by_edge: dict[tuple[int, str], _Candidate] = {}
    for candidate in accepted:
        key = (candidate.page_index, candidate.side)
        if key not in best_by_edge or candidate.rank < best_by_edge[key].rank:
            best_by_edge[key] = candidate
    return list(best_by_edge.values())


def _banner_is_outermost(candidate: _Candidate, lines: list[str], removed_lines: set[int]) -> bool:
    countable = _countable_indexes(lines)
    position = countable.index(candidate.line_index)
    between = countable[:position] if candidate.side == "top" else countable[position + 1:]
    return all(index in removed_lines for index in between)


def strip_page_furniture(page_texts: list[str]) -> FurnitureStripResult:
    """Remove running headers/footers from the edges of an agreement's body pages.

    ``page_texts`` is the tagged text of each body page, in page order. Returns the
    texts with furniture lines dropped plus a record of each removed line.
    """
    all_candidates = [c for i, text in enumerate(page_texts) for c in _page_candidates(i, text)]
    accepted = _repeated([c for c in all_candidates if c.kind not in ("page_number", "banner")])
    accepted += _sequenced_page_numbers([c for c in all_candidates if c.kind == "page_number"])

    removed_by_page: dict[int, set[int]] = {}
    for candidate in accepted:
        removed_by_page.setdefault(candidate.page_index, set()).add(candidate.line_index)

    banners = [c for c in all_candidates if c.kind == "banner"]
    banner_lines_by_page: dict[int, set[int]] = {}
    for candidate in banners:
        banner_lines_by_page.setdefault(candidate.page_index, set()).add(candidate.line_index)
    page_lines = [text.split("\n") for text in page_texts]

    def outermost(candidate: _Candidate, extra: dict[int, set[int]]) -> bool:
        allowed = removed_by_page.get(candidate.page_index, set()) | extra.get(candidate.page_index, set())
        return _banner_is_outermost(candidate, page_lines[candidate.page_index], allowed)

    banners = _repeated([c for c in banners if outermost(c, banner_lines_by_page)])
    kept_banner_lines: dict[int, set[int]] = {}
    for candidate in banners:
        kept_banner_lines.setdefault(candidate.page_index, set()).add(candidate.line_index)
    banners = [c for c in banners if outermost(c, kept_banner_lines)]
    accepted += banners

    drop: dict[int, dict[int, str]] = {}
    for candidate in accepted:
        drop.setdefault(candidate.page_index, {})[candidate.line_index] = candidate.kind

    result = FurnitureStripResult(page_texts=[])
    for page_index, text in enumerate(page_texts):
        page_drop = drop.get(page_index)
        if not page_drop:
            result.page_texts.append(text)
            continue
        kept: list[str] = []
        for line_index, line in enumerate(page_lines[page_index]):
            if line_index in page_drop:
                result.removed.append(
                    RemovedFurnitureLine(page_index=page_index, line=line.strip(), kind=page_drop[line_index])
                )
                continue
            kept.append(line)
        result.page_texts.append("\n".join(kept))
    return result
