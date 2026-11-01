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
  ``MIN_REPEAT_PAGES`` pages are stripped, and only when the chain's pages plus
  the pages numbered some other way (a tagged ``<page>`` number, or a short
  split-off header/footer fragment) cover at least ``MIN_CHAIN_COVERAGE`` of the
  pages it spans. On pages that also carry a tagged ``<page>`` number, the bare
  number must sit the same distance from the tagged one on every link of the
  chain (list items 3, 4, 6 on pages tagged 11-13 do not). At most one number
  per page is removed:
  when both edges of a page fit, the agreement's dominant edge wins (a tie
  strips neither). A number off the dominant edge that sits right above a
  heading-like line is kept, and a number whose next non-blank line opens a
  heading without a number of its own is always kept (``12`` above
  ``<section>Miscellaneous</section>`` is a clause number).
- Banners (EXECUTION VERSION, CONFIDENTIAL): the shape repeats on
  ``MIN_REPEAT_PAGES`` pages and every non-empty line between the banner and the
  page edge is itself being stripped, so the banner is the outermost text.

Removed lines are dropped whole; no other text on the page changes.

Known limitations (each needs a coincidence the rules cannot see through):

- A bare number on the dominant edge of a page that lost its own page number is
  stripped when it happens to equal that page's number.
- Clause numbers equal to the page number on 3 or more consecutive tagged pages
  are stripped (they sit a constant distance from the tagged numbers).
- An untagged one-word ``CONFIDENTIAL`` heading on the last line of 3 or more
  pages is stripped as a banner.
- An identical amount (``250000.00``) alone on the edges of 3 or more pages is
  stripped as a numeric document ID.
"""

import re
from dataclasses import dataclass, field

EDGE_LINE_WINDOW = 3
MIN_REPEAT_PAGES = 3
PAGE_OFFSET_TOLERANCE = 2
# Share of the pages a page-number chain spans that must be numbered: by a chain
# link, a tagged <page> number, or as a short split-off header/footer fragment.
MIN_CHAIN_COVERAGE = 0.5

# A line that is only a tagged page number ("<page>12</page>", "Page <page>12</page>").
_PAGE_MARKER_LINE_RE = re.compile(r"(?:page\s*)?<page>.*?</page>", re.IGNORECASE)
_PAGE_TAG_RE = re.compile(r"<page>.*?</page>", re.IGNORECASE)
# A tagged page number: "<page>12</page>", "<page>- 12 -</page>". Not a tag glued to
# other text ("<page>4893-</page>2596-7688v2" is a document ID the tagger split).
_CLEAN_PAGE_TAG_RE = re.compile(
    r"(?<![\w-])<page>\s*(?:page\s*)?"
    r"(?:[-\u2013\u2014]\s*(?P<dashed>\d{1,4})\s*[-\u2013\u2014]|(?P<plain>\d{1,4}))"
    r"\s*</page>(?![\w-])",
    re.IGNORECASE,
)
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
    next_is_heading_like: bool = False  # the next non-blank line looks like a heading


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


def _page_is_numbered_elsewhere(text: str) -> bool:
    """A page that needs no bare page number to count as numbered: it carries a
    tagged <page> number, or it is a short fragment (a split-off header or footer)."""
    return _PAGE_TAG_RE.search(text) is not None or len(_countable_indexes(text.split("\n"))) <= EDGE_LINE_WINDOW


def _next_line_is_heading_like(lines: list[str], index: int) -> bool:
    """True when the next non-blank line opens a tagged heading or is a short all-caps title."""
    for line in lines[index + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if _HEADING_START_RE.match(stripped):
            return True
        letters = [ch for ch in stripped if ch.isalpha()]
        return bool(letters) and all(ch.isupper() for ch in letters) and len(stripped.split()) <= 8
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
        candidates.append(
            _Candidate(
                page_index,
                index,
                kind,
                lines[index].strip(),
                side,
                rank,
                next_is_heading_like=_next_line_is_heading_like(lines, index),
            )
        )
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


def _tagged_page_numbers(page_texts: list[str]) -> dict[int, int]:
    """The tagged <page> number of each page that has one.

    A value repeated on a neighboring page (a document ID the tagger wrapped in
    <page>, say) is not a page number and is left out.
    """
    values: dict[int, int] = {}
    for index, text in enumerate(page_texts):
        match = _CLEAN_PAGE_TAG_RE.search(text)
        if match is not None:
            values[index] = int(match.group("dashed") or match.group("plain"))
    return {
        index: value
        for index, value in values.items()
        if values.get(index - 1) != value and values.get(index + 1) != value
    }


def _tag_gap(candidate: _Candidate, tagged: dict[int, int]) -> int | None:
    tag = tagged.get(candidate.page_index)
    return None if tag is None else _page_number_value(candidate.text) - tag


def _chain_tag_gap(chain: list[_Candidate], tagged: dict[int, int]) -> int | None:
    for link in chain:
        gap = _tag_gap(link, tagged)
        if gap is not None:
            return gap
    return None


def _fits_after(candidate: _Candidate, chain: list[_Candidate], tagged: dict[int, int]) -> bool:
    last = chain[-1]
    value = _page_number_value(candidate.text)
    last_value = _page_number_value(last.text)
    offset = value - candidate.page_index
    last_offset = last_value - last.page_index
    if not (value > last_value and abs(offset - last_offset) <= PAGE_OFFSET_TOLERANCE):
        return False
    # On tagged pages, the bare number must sit a constant distance from the tagged one.
    gap = _tag_gap(candidate, tagged)
    chain_gap = _chain_tag_gap(chain, tagged)
    return gap is None or chain_gap is None or gap == chain_gap


def _chain_page_numbers(
    group: list[_Candidate], dominant: str | None, tagged: dict[int, int]
) -> list[list[_Candidate]]:
    """Link one shape's candidates into rising page sequences.

    When a chain can be extended by candidates on both edges of one page, the
    candidate on the agreement's dominant side wins; with no dominant side,
    neither is used.
    """
    by_page: dict[int, list[_Candidate]] = {}
    for candidate in group:
        by_page.setdefault(candidate.page_index, []).append(candidate)
    chains: list[list[_Candidate]] = []
    for page_index in sorted(by_page):
        page_candidates = sorted(by_page[page_index], key=lambda c: c.rank)
        used: set[int] = set()
        for chain in sorted(chains, key=len, reverse=True):
            fitting = [
                c for c in page_candidates if id(c) not in used and _fits_after(c, chain, tagged)
            ]
            if not fitting:
                continue
            if len({c.side for c in fitting}) == 2:
                if dominant is None:
                    used.update(id(c) for c in fitting)
                    continue
                fitting = [c for c in fitting if c.side == dominant]
            pick = min(fitting, key=lambda c: c.rank)
            chain.append(pick)
            used.add(id(pick))
        for candidate in page_candidates:
            if id(candidate) not in used:
                chains.append([candidate])
                used.add(id(candidate))
    return chains


def _accepted_chain(chain: list[_Candidate], numbered_pages: set[int]) -> bool:
    """A chain needs MIN_REPEAT_PAGES links, and the pages it spans must be numbered
    densely: its links plus pages numbered some other way cover MIN_CHAIN_COVERAGE."""
    if len(chain) < MIN_REPEAT_PAGES:
        return False
    first, last = chain[0].page_index, chain[-1].page_index
    numbered = {c.page_index for c in chain} | {p for p in numbered_pages if first <= p <= last}
    return len(numbered) >= MIN_CHAIN_COVERAGE * (last - first + 1)


def _accepted_page_numbers(
    candidates: list[_Candidate], dominant: str | None, numbered_pages: set[int], tagged: dict[int, int]
) -> list[_Candidate]:
    accepted: list[_Candidate] = []
    by_shape: dict[str, list[_Candidate]] = {}
    for candidate in candidates:
        by_shape.setdefault(_line_shape(candidate.text), []).append(candidate)
    for group in by_shape.values():
        for chain in _chain_page_numbers(group, dominant, tagged):
            if _accepted_chain(chain, numbered_pages):
                accepted.extend(chain)
    return accepted


def _dominant_side(candidates: list[_Candidate]) -> str | None:
    top = sum(1 for c in candidates if c.side == "top")
    bottom = len(candidates) - top
    if top == bottom:
        return None
    return "top" if top > bottom else "bottom"


def _sequenced_page_numbers(
    candidates: list[_Candidate], numbered_pages: set[int], tagged: dict[int, int]
) -> list[_Candidate]:
    """Keep page-number candidates that chain into the agreement's page sequence."""
    dominant = _dominant_side(_accepted_page_numbers(candidates, None, numbered_pages, tagged))
    accepted = _accepted_page_numbers(candidates, dominant, numbered_pages, tagged)
    if dominant is not None:
        # Off the usual edge, a number right above a heading is a clause number.
        accepted = [c for c in accepted if c.side == dominant or not c.next_is_heading_like]
    by_page: dict[int, list[_Candidate]] = {}
    for candidate in accepted:
        by_page.setdefault(candidate.page_index, []).append(candidate)
    result: list[_Candidate] = []
    for page_candidates in by_page.values():
        sides = {c.side for c in page_candidates}
        if len(sides) == 2:
            if dominant is None:
                continue
            page_candidates = [c for c in page_candidates if c.side == dominant]
        # At most one page number per page: keep the outermost.
        result.append(min(page_candidates, key=lambda c: c.rank))
    return result


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
    numbered_pages = {i for i, text in enumerate(page_texts) if _page_is_numbered_elsewhere(text)}
    tagged = _tagged_page_numbers(page_texts)
    accepted += _sequenced_page_numbers(
        [c for c in all_candidates if c.kind == "page_number"], numbered_pages, tagged
    )

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
