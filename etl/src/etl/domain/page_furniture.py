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
- Running titles: the line (digits collapsed, a trailing ``<page>`` tag read as
  its number) sits at a page edge on at least ``MIN_TITLE_PAGE_SHARE`` of the
  body pages and on ``MIN_TITLE_PAGES`` pages, is outermost like a banner, and
  never equals a tagged heading of the agreement. A title with a page number
  ("ASSET PURCHASE AGREEMENT PAGE 12", "Share Sale Agreement - 12",
  "12 Share Sale Agreement") must also fit the page sequence like a bare page
  number; when its number is tagged (``PAGE <page>12</page>``) the ``<page>``
  tag is left in place. A bare title ("Share Sale Agreement", at least two words,
  naming an agreement or plan) must be the agreement's own printed title: the
  first line of the first body page, or text in the opening pages outside the
  running lines. Its copy on the first body page is kept as the cover title, and
  so is any copy whose next line opens a preamble ("THIS", "dated", "by and
  among", "between", "made", "is entered into") or names the title inside longer
  text. At most
  one copy per page edge goes (the outermost), and only on the title's dominant
  edge.

- Page labels: a line that is only a label glued to a tagged page number
  ("Page <page>12</page>") at a page edge, on ``MIN_REPEAT_PAGES`` pages, loses
  the label word; the ``<page>`` tag stays.
- Any other repeated edge line (deal names, firm names, legends, document IDs in
  other formats, split counters such as "Page <page>2</page> of 23"): the exact
  residue, meaning the line with its clean ``<page>`` tags taken out and any other
  ``<page>`` tag unwrapped ("<page>4848-</page>1716-0445" is the ID
  "4848-1716-0445"), sits at a page edge on at least ``MIN_TITLE_PAGE_SHARE`` of
  the body pages and on ``MIN_TITLE_PAGES`` pages. The residue must not equal a
  tagged heading or precede a preamble opener. At most one copy per page edge is
  removed (the outermost), only on the residue's dominant edge, and only when it
  is outermost. Clean page tags on the line are kept.
  - On most of its pages the line must be the outermost line left after the
    other rules, or sit directly under such a stripped running line. Only
    legends, dates and lines carrying a page tag stack deeper (multi-line
    legends); other lines never cascade in once the lines above them go.
  - The residue's first copy stays unless it is on the first body page: an
    embedded schedule's header or table row is real text where it first appears.
  - Left alone: single words other than "page"/"back" (word-per-line text,
    defined terms), lines with no letters or digits (rules, blanks), and lines
    without digits that end like a clause (".", ";", ":", ","), unless they are a
    confidentiality or omission legend. Also page-break boilerplate ("[Signature
    Page Follows]", "[Remainder of page ...]", "intentionally left blank"),
    signature-block lines ("By:", "Name:", "Title:", "/s/"), "(continued)" lines,
    lone enumerators ("(a)", "(iv)", and "(1)" unless beside a page tag), and
    lines that open with Exhibit, Schedule, Annex or Appendix.

Removed lines are dropped whole (except a kept ``<page>`` tag).

Furniture fused into a body line (run after the line rules, on their output):

- A furniture line the line rules confirmed on this agreement (a document ID, a
  repeated line, a running title with a page number) is cut from the very start
  or end of an edge-window line, its page number tagged or bare. A copy that is
  a whole line is removed only when its page number is bare (the line rules
  could not match it); other whole-line copies were kept on purpose. A
  template needs a page-number slot or a document ID: words alone ("Classified
  as Secret") can also end a body sentence.
- A document-management ID token (6+ digits with a version, all-caps letters or
  a \\ / : separator before 6+ digits, or ####-####-####; never "Loan-1234567")
  at the start or end of edge lines on at least half of the body pages and on 5
  pages. The page number beside it goes only when it fits the agreement's page
  sequence and at least half of the ID's copies carry a fitting number on that
  side. The ID also takes the rest of its footer, read away from the body text:
  tokens printed beside it on at least 80% of those copies ("WA", "DB1/",
  "CAN_DMS:", "(c) King & Wood Mallesons Share Sale Agreement") and page
  numbers there that fit the sequence. A word that ends a clause ("Parties.")
  stops it, and the footer words must occur nowhere else in the body pages.
- Both are skipped for the whole agreement when the text also occurs anywhere
  else in the body pages (mid-line, or on a non-edge line), and a confirmed
  line shorter than ``MIN_FUSED_CHARS`` is never cut unless it is a document ID.
  Bare running titles are never cut: "this Agreement and Plan of Merger" ends
  body lines.
- An untagged page number that is the first word of a page's first line (or
  last word of its last line), on a page with no page number of its own, when
  the nearest tagged pages before and after give that exact number. Not when
  the page already shows that number in its edge window (a bare line, or the
  word at the other edge), when a unit follows it at the top ("12 months",
  "12 Business Days", "12 shares", "12%"), or when a reference word precedes it
  at the bottom ("shall not exceed 12", "Section 12", "$ 12").

Only the furniture substring goes; the whitespace it leaves is collapsed.

Known limitations (each needs a coincidence the rules cannot see through):

- A bare number on the dominant edge of a page that lost its own page number is
  stripped when it happens to equal that page's number.
- Clause numbers equal to the page number on 3 or more consecutive tagged pages
  are stripped (they sit a constant distance from the tagged numbers).
- An untagged one-word ``CONFIDENTIAL`` heading on the last line of 3 or more
  pages is stripped as a banner.
- An identical amount (``250000.00``) alone on the edges of 3 or more pages is
  stripped as a numeric document ID.
- A heading or sentence split across lines whose last line, at a page edge,
  equals the agreement's running title is stripped with the running title.
- A title with a page number need not name an agreement, so a repeated firm
  footer such as "DLA Piper | 12" is stripped. Its words must be identical on
  every page (title words carry no digits, so the digit-collapsed shape fixes
  them), and its number must fit the page sequence.
- A body number right before a fused document ID is cut with it when it happens
  to fit the page sequence ("... within 4 NAI-5005005981v11" on printed page 4).
- On a page that lost its own page number, a first or last word that happens to
  equal that number is cut as the page number ("13 directors ...", "... an
  aggregate of 13", "... Rule 13"), unless a unit or reference word guard
  applies.
"""

import re
from collections.abc import Callable
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
# Running titles. Title words: letters and light punctuation, no digits.
_TITLE_WORDS = r"[A-Za-z][A-Za-z&,.'\u2019()/ -]{0,74}[A-Za-z.)]"
_TITLE_WITH_NUMBER_RES = [
    # ASSET PURCHASE AGREEMENT PAGE 12
    re.compile(rf"(?P<title>{_TITLE_WORDS})\s+page\s+(?P<number>\d{{1,3}})", re.IGNORECASE),
    # Share Sale Agreement - 12, Share Sale Agreement | 12
    re.compile(rf"(?P<title>{_TITLE_WORDS})\s*[-\u2013\u2014|]\s*(?P<number>\d{{1,3}})"),
    # 12 Share Sale Agreement, 12 | Share Sale Agreement
    re.compile(rf"(?P<number>\d{{1,3}})\s*[-\u2013\u2014|]?\s+(?P<title>{_TITLE_WORDS})"),
]
_TITLE_LINE_RE = re.compile(_TITLE_WORDS)
# A titled page number whose number the tagger wrapped: "ASSET PURCHASE AGREEMENT PAGE <page>6</page>".
_TRAILING_PAGE_TAG_RE = re.compile(r"^(?P<head>[^<>]*?)\s*(?P<tag><page>\s*(?P<number>\d{1,3})\s*</page>)$", re.IGNORECASE)
_AGREEMENT_WORD_RE = re.compile(r"\b(?:agreement|plan\s+of\s+(?:merger|arrangement|reorganization))\b", re.IGNORECASE)
MAX_TITLE_CHARS = 80
# A running title must sit at a page edge on at least this share of body pages.
MIN_TITLE_PAGE_SHARE = 0.5
MIN_TITLE_PAGES = 5

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
    keep: str | None = None  # text left in place of the line when it is stripped (a <page> tag)


def furniture_kind(line: str) -> str | None:
    """Return the furniture pattern name a whole trimmed line matches, else None."""
    stripped = line.strip()
    if not stripped or "<" in stripped or ">" in stripped:
        return None
    for kind, pattern in FURNITURE_PATTERNS.items():
        if pattern.fullmatch(stripped):
            return kind
    return None


def _untag_trailing_page_number(line: str) -> tuple[str, str | None]:
    """("TITLE PAGE 6", "<page>6</page>") for "TITLE PAGE <page>6</page>"; (line, None) otherwise."""
    stripped = " ".join(line.split())
    match = _TRAILING_PAGE_TAG_RE.match(stripped)
    if match is None or not match.group("head"):
        return stripped, None
    return f"{match.group('head')} {match.group('number')}", match.group("tag")


def running_title_kind(line: str) -> str | None:
    """"titled_page_number" for a title with a page number ("ASSET PURCHASE AGREEMENT
    PAGE 12", also with the number tagged as ``<page>12</page>``), "title" for a bare
    agreement title ("Share Sale Agreement"), else None. Lines that match an
    ordinary furniture pattern are not running titles."""
    stripped, tag = _untag_trailing_page_number(line)
    if not stripped or len(stripped) > MAX_TITLE_CHARS or "<" in stripped or ">" in stripped:
        return None
    if tag is not None:
        if any(pattern.fullmatch(stripped) for pattern in _TITLE_WITH_NUMBER_RES[:2]):
            return "titled_page_number"
        return None
    if furniture_kind(stripped) is not None:
        return None
    for pattern in _TITLE_WITH_NUMBER_RES:
        if pattern.fullmatch(stripped):
            return "titled_page_number"
    if (
        _TITLE_LINE_RE.fullmatch(stripped)
        and len(stripped.split()) >= 2
        and _AGREEMENT_WORD_RE.search(stripped)
    ):
        return "title"
    return None


def _title_number(text: str) -> int:
    untagged, _ = _untag_trailing_page_number(text)
    for pattern in _TITLE_WITH_NUMBER_RES:
        match = pattern.fullmatch(untagged)
        if match is not None:
            return int(match.group("number"))
    raise ValueError(text)


def _line_shape(line: str) -> str:
    return _DIGITS_RE.sub("#", " ".join(line.split()))


def _title_shape(line: str) -> str:
    return _line_shape(_untag_trailing_page_number(line)[0])


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


# A printed table of contents opens with a "Page" column header ("Page", "PAGE ----",
# "<TABLE> <CAPTION> Page") or goes straight into article entries that end in page
# numbers ("ARTICLE IV CAPITAL STOCK 8 4.1 Issuance of Stock 8"). An index of defined
# terms ("Exchange Agent 3.2(a) Exchange Fund 3.2(b)") cites sections, not pages.
_TOC_PAGE_HEADER_RE = re.compile(r"(?:<TABLE>\s*<CAPTION>\s*)?page\b", re.IGNORECASE)
_TOC_ARTICLE_ENTRY_RE = re.compile(r"(?:ARTICLE|Article)\s")
_TOC_ENTRY_PAGE_RE = re.compile(r"\s(?:[A-Z]-)?\d{1,3}\s+(?=ARTICLE\b|Article\b|SECTION\b|Section\b|\d+\.\d+)")


def _opens_toc_listing(lines: list[str], index: int) -> bool:
    """True when the line after ``index`` starts a printed table of contents, so a
    "Table of Contents" line at ``index`` is that table's heading, not the back-link."""
    for line in lines[index + 1:]:
        stripped = line.strip()
        if not stripped:
            continue
        if _TOC_PAGE_HEADER_RE.match(stripped):
            return True
        return bool(_TOC_ARTICLE_ENTRY_RE.match(stripped)) and len(_TOC_ENTRY_PAGE_RE.findall(stripped)) >= 2
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
        kind = furniture_kind(lines[index]) or running_title_kind(lines[index])
        if kind is None:
            continue
        if kind == "page_number" and _next_line_opens_unnumbered_heading(lines, index):
            continue
        side, rank = positions[index]
        keep = _untag_trailing_page_number(lines[index])[1] if kind == "titled_page_number" else None
        candidates.append(
            _Candidate(
                page_index,
                index,
                kind,
                lines[index].strip(),
                side,
                rank,
                next_is_heading_like=_next_line_is_heading_like(lines, index),
                keep=keep,
            )
        )
    return candidates


def _repeated(candidates: list[_Candidate]) -> list[_Candidate]:
    pages_by_key: dict[str, set[int]] = {}
    for candidate in candidates:
        pages_by_key.setdefault(_repeat_key(candidate), set()).add(candidate.page_index)
    return [c for c in candidates if len(pages_by_key[_repeat_key(c)]) >= MIN_REPEAT_PAGES]


def _page_number_value(text: str) -> int:
    if running_title_kind(text) == "titled_page_number":
        return _title_number(text)
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
        by_shape.setdefault(_title_shape(candidate.text), []).append(candidate)
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


def _heading_texts(page_texts: list[str]) -> set[str]:
    """Normalized text of every tagged <article>/<section> heading in the agreement."""
    texts: set[str] = set()
    for text in page_texts:
        for match in re.finditer(r"<(article|section)>(.*?)</\1>", text, re.IGNORECASE | re.DOTALL):
            texts.add(" ".join(match.group(2).split()).lower())
    return texts


def _title_is_printed(title: str, page_texts: list[str], title_lines: set[tuple[int, int]]) -> bool:
    """The title is the agreement's own: it is the first line of the first body page,
    or it appears in the opening body pages outside the running-title lines."""
    if not page_texts:
        return False
    first_lines = [line.strip() for line in page_texts[0].split("\n") if line.strip()]
    if first_lines and " ".join(first_lines[0].split()) == title:
        return True
    needle = title.lower()
    for page_index, text in enumerate(page_texts[:2]):
        for line_index, line in enumerate(text.split("\n")):
            if (page_index, line_index) in title_lines:
                continue
            if needle in " ".join(line.split()).lower():
                return True
    return False


# How a preamble opens. Capitalized forms only for words that also start body
# continuation lines ("this Agreement and ...", "between the parties ...").
_PREAMBLE_START_RE = re.compile(
    r"^\s*(?:"
    r"(?:THIS|This)\b"
    r"|(?:DATED|Dated|dated)\b"
    r"|(?:BY AND|By and|by and)\s+(?:AMONG|BETWEEN|among|between)\b"
    r"|(?:BETWEEN|Between)\b"
    r"|(?:MADE|Made)\b"
    r"|(?:IS|Is)\s+(?:MADE|ENTERED\s+INTO|made|entered\s+into)\b"
    r")"
)


def _next_nonblank(lines: list[str], index: int) -> str | None:
    for line in lines[index + 1:]:
        if line.strip():
            return line.strip()
    return None


def _opens_preamble_or_repeats_title(next_line: str | None, title: str) -> bool:
    """The line after a title copy opens the agreement ("THIS AGREEMENT is dated ...",
    "by and among ...") or names the title inside longer text: that copy is a real title."""
    if next_line is None:
        return False
    following = " ".join(next_line.split()).lower()
    # A line that is exactly the title again is a second copy, not a preamble.
    return bool(_PREAMBLE_START_RE.match(next_line) or (title.lower() in following and following != title.lower()))


def _bare_title_copies(group: list[_Candidate], page_texts: list[str], title: str) -> list[_Candidate]:
    """The copies of a bare running title to strip.

    The first body page's copy (the cover title) and any copy followed by a
    preamble, or by longer text naming the title, are kept. At most one copy per page edge is
    stripped (the outermost), and only on the agreement's dominant edge for this
    title; with no dominant edge, none is stripped.
    """
    outermost: dict[tuple[int, str], _Candidate] = {}
    for candidate in group:
        if candidate.page_index == 0:
            continue
        lines = page_texts[candidate.page_index].split("\n")
        if _opens_preamble_or_repeats_title(_next_nonblank(lines, candidate.line_index), title):
            continue
        key = (candidate.page_index, candidate.side)
        if key not in outermost or candidate.rank < outermost[key].rank:
            outermost[key] = candidate
    dominant = _dominant_side(list(outermost.values()))
    if dominant is None:
        return []
    return [c for c in outermost.values() if c.side == dominant]


def _running_titles(
    candidates: list[_Candidate],
    page_texts: list[str],
    numbered_pages: set[int],
    tagged: dict[int, int],
) -> list[_Candidate]:
    """Running titles repeated on most body pages.

    A shape (digits collapsed) must sit at a page edge on at least
    MIN_TITLE_PAGE_SHARE of the body pages and on MIN_TITLE_PAGES pages. A title
    with a page number must also fit the page sequence like a bare page number; a
    bare title must be the agreement's own printed title, and its copy on the first
    body page (the cover title) is kept. A line that is also a tagged heading
    anywhere in the agreement is never stripped.
    """
    needed = max(MIN_TITLE_PAGES, MIN_TITLE_PAGE_SHARE * len(page_texts))
    headings = _heading_texts(page_texts)
    by_shape: dict[str, list[_Candidate]] = {}
    for candidate in candidates:
        by_shape.setdefault(_title_shape(candidate.text), []).append(candidate)
    accepted: list[_Candidate] = []
    for shape, group in by_shape.items():
        if len({c.page_index for c in group}) < needed:
            continue
        if any(_untag_trailing_page_number(c.text)[0].lower() in headings for c in group):
            continue
        if group[0].kind == "titled_page_number":
            accepted += _sequenced_page_numbers(group, numbered_pages, tagged)
            continue
        title_lines = {(c.page_index, c.line_index) for c in group}
        if not _title_is_printed(shape, page_texts, title_lines):
            continue
        accepted += _bare_title_copies(group, page_texts, shape)
    return accepted


# A whole line that is a page label glued to a tagged page number: "Page <page>12</page>".
_PAGE_LABEL_LINE_RE = re.compile(r"(?:page|Page|PAGE)\s*(?P<tag><page>\s*\d{1,4}\s*</page>)")
_ANY_PAGE_TAG_RE = re.compile(r"<page>(?P<inner>.*?)</page>", re.IGNORECASE | re.DOTALL)
_OTHER_TAG_RE = re.compile(r"</?(?!page>)[a-zA-Z]+>")
# Single words repeat at page edges when text is broken word-per-line, or as a
# defined term; only these two are page furniture.
_SINGLE_WORD_FURNITURE = {"page", "back"}
# Legend words: a legend sentence repeated on most pages is furniture, not text.
_LEGEND_WORD_RE = re.compile(r"confidential|omitted|redacted", re.IGNORECASE)
MAX_REPEATED_LINE_CHARS = 100


def _line_residue(line: str) -> tuple[str, str | None]:
    """The furniture text of a line and the genuine page tags to keep.

    Clean page-number tags (``<page>12</page>``) are taken out and returned to be
    kept; any other <page> tag is unwrapped into the residue, since its content is
    not a page number ("<page>4898-</page>3834-0359.9" is the document ID
    "4898-3834-0359.9")."""
    kept: list[str] = []

    def take(match: re.Match[str]) -> str:
        tag = match.group(0)
        if _CLEAN_PAGE_TAG_RE.fullmatch(tag) and re.fullmatch(r"\s*(?:page\s*)?[-\u2013\u2014]?\s*\d{1,4}\s*[-\u2013\u2014]?\s*", match.group("inner"), re.IGNORECASE):
            kept.append(tag)
            return " "
        return match.group("inner")

    residue = _ANY_PAGE_TAG_RE.sub(take, line)
    return " ".join(residue.split()), (" ".join(kept) or None)


# Lines that repeat on many pages but belong to the agreement's own structure:
# page-break boilerplate, signature blocks, continued-table markers, enumerators
# and exhibit/schedule captions.
_REPEATED_LINE_EXCLUDED_RES = [
    re.compile(r"\[[^\]]*(?:remainder\s+of|signature\s+page|left\s+blank)[^\]]*\]?", re.IGNORECASE),
    re.compile(r".*\bintentionally\s+(?:left\s+)?blank\b.*", re.IGNORECASE),
    re.compile(r"(?:by|name|title|its)\s*:.*|/s/.*", re.IGNORECASE),
    re.compile(r".*\(\s*cont(?:inued|['’]d|\.)?\s*\).*|continued\b.*", re.IGNORECASE),
    re.compile(r"\(?(?:[a-z]{1,2}|[ivxlc]{1,6}|[A-Z]{1,2}|[IVXLC]{1,6})[.)]", re.IGNORECASE),
    re.compile(r"(?:exhibit|schedule|annex|appendix)\b.*", re.IGNORECASE),
]
_DIGIT_ENUMERATOR_RE = re.compile(r"\(?\d{1,3}[.)]")


def _residue_is_eligible(residue: str, keep: str | None = None) -> bool:
    if not residue or len(residue) > MAX_REPEATED_LINE_CHARS or "<" in residue or ">" in residue:
        return False
    if any(pattern.fullmatch(residue) for pattern in _REPEATED_LINE_EXCLUDED_RES):
        return False
    if keep is None and _DIGIT_ENUMERATOR_RE.fullmatch(residue):
        return False  # a lone "(1)" or "2." is an enumerator unless it sits beside a page tag
    if not any(ch.isalnum() for ch in residue):
        return False  # rules and blanks ("-----", "[______]") are left alone
    if " " not in residue and residue.isalpha() and residue.lower() not in _SINGLE_WORD_FURNITURE:
        return False  # a single word: word-per-line body text ("each") or a defined term ("Agreement")
    has_digit = any(ch.isdigit() for ch in residue)
    if not has_digit and residue[-1] in ".;:," and not _LEGEND_WORD_RE.search(residue):
        return False  # reads as a sentence or clause of body text
    return True


_DATE_LINE_RE = re.compile(
    r"(?:\d{1,2}(?:/\d{1,2})?\.?\s+)?(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?"
    r"(?:\s+\d{1,2},?)?\s+\d{4}",
    re.IGNORECASE,
)


def _stacks_deeper(residue: str, copies: list[_Candidate]) -> bool:
    """A line that may sit under two or more stripped running lines."""
    return bool(
        _LEGEND_WORD_RE.search(residue)
        or _DATE_LINE_RE.fullmatch(residue)
        or any(c.keep is not None for c in copies)
    )


def _repeated_edge_lines(
    page_texts: list[str], removed_by_page: dict[int, set[int]]
) -> list[_Candidate]:
    """Any other line repeated, word for word, at the edges of most body pages.

    The exact residue (the line without its genuine page tags) must sit at a page
    edge on at least MIN_TITLE_PAGE_SHARE of the body pages and on MIN_TITLE_PAGES
    pages, never equal a tagged heading, and not be followed by a preamble opener.
    At most one copy per page edge goes (the outermost), only on the residue's
    dominant edge, and only when every line between it and the edge is also
    being stripped. Bare agreement titles and titles with page numbers are left
    to the running-title rule.
    """
    needed = max(MIN_TITLE_PAGES, MIN_TITLE_PAGE_SHARE * len(page_texts))
    headings = _heading_texts(page_texts)
    by_residue: dict[str, list[_Candidate]] = {}
    outer_line: dict[tuple[int, int], int] = {}
    page_lines = [text.split("\n") for text in page_texts]
    for page_index, lines in enumerate(page_lines):
        countable = _countable_indexes(lines)
        if not countable:
            continue
        inside = _lines_inside_headings(lines)
        positions: dict[int, tuple[str, int]] = {}
        for rank, index in enumerate(countable[:EDGE_LINE_WINDOW]):
            positions[index] = ("top", rank)
        for rank, index in enumerate(reversed(countable[-EDGE_LINE_WINDOW:])):
            if index not in positions or rank < positions[index][1]:
                positions[index] = ("bottom", rank)
        for index, (side, rank) in positions.items():
            if index in inside or index in removed_by_page.get(page_index, set()):
                continue
            line = lines[index]
            if _OTHER_TAG_RE.search(line):
                continue
            residue, keep = _line_residue(line)
            if not _residue_is_eligible(residue, keep):
                continue
            # Titles with page numbers and bare agreement titles have their own rule.
            title_kind = running_title_kind(residue)
            if title_kind == "titled_page_number" or (title_kind == "title" and keep is None):
                continue
            # Rank once the lines already stripped by the other rules are gone.
            edge_side = countable if side == "top" else countable[::-1]
            outer = edge_side[: edge_side.index(index)]
            removed = removed_by_page.get(page_index, set())
            effective = [i for i in outer if i not in removed]
            candidate = _Candidate(page_index, index, "repeated_line", line.strip(), side, len(effective), keep=keep)
            by_residue.setdefault(residue, []).append(candidate)
            if effective:
                outer_line[(page_index, index)] = effective[-1]
    copies_by_residue: dict[str, list[_Candidate]] = {}
    for residue, group in by_residue.items():
        if len({c.page_index for c in group}) < needed or residue.lower() in headings:
            continue
        outermost: dict[tuple[int, str], _Candidate] = {}
        for candidate in group:
            following = _next_nonblank(page_lines[candidate.page_index], candidate.line_index)
            if following is not None and _PREAMBLE_START_RE.match(following):
                continue
            key = (candidate.page_index, candidate.side)
            if key not in outermost or candidate.rank < outermost[key].rank:
                outermost[key] = candidate
        dominant = _dominant_side(list(outermost.values()))
        if dominant is None:
            continue
        copies = [c for c in outermost.values() if c.side == dominant]
        # The residue's first copy stays unless it is on the first body page:
        # a schedule header or table row repeated on later pages is real text
        # where it first appears, and a schedule never opens the agreement.
        first_page = min(c.page_index for c in group)
        if first_page > 0:
            copies = [c for c in copies if c.page_index != first_page]
        copies_by_residue[residue] = copies

    def mostly(copies: list[_Candidate], ok: Callable[[_Candidate], bool]) -> list[_Candidate]:
        good = [c for c in copies if ok(c)]
        return good if copies and 2 * len(good) > len(copies) else []

    # The line must be the outermost one left on most of its pages, or sit right
    # under such a running line. Only legends, dates and lines that carry a page
    # tag stack deeper (multi-line legends, "Project X / date / version"
    # headers); other lines never cascade in once the lines above them go.
    accepted: list[_Candidate] = []
    accepted_lines: set[tuple[int, int]] = set()
    pending = dict(copies_by_residue)

    def under_accepted(candidate: _Candidate, max_rank: int) -> bool:
        if candidate.rank == 0:
            return True
        above = outer_line.get((candidate.page_index, candidate.line_index))
        return candidate.rank <= max_rank and (candidate.page_index, above) in accepted_lines

    for max_rank in (0, 1, EDGE_LINE_WINDOW, EDGE_LINE_WINDOW):
        for residue, copies in list(pending.items()):
            if max_rank > 1 and not _stacks_deeper(residue, copies):
                continue
            good = mostly(copies, lambda c: under_accepted(c, max_rank))
            if good:
                accepted += good
                accepted_lines.update((c.page_index, c.line_index) for c in good)
                del pending[residue]
    return accepted


def _page_label_lines(page_texts: list[str]) -> list[_Candidate]:
    """ "Page <page>12</page>" lines at a page edge: the label word goes, the tag stays.

    The line must be among the first or last EDGE_LINE_WINDOW non-empty lines, and
    the agreement must carry such labels on MIN_REPEAT_PAGES pages."""
    found: list[_Candidate] = []
    for page_index, text in enumerate(page_texts):
        lines = text.split("\n")
        nonempty = [i for i, line in enumerate(lines) if line.strip()]
        edge = set(nonempty[:EDGE_LINE_WINDOW]) | set(nonempty[-EDGE_LINE_WINDOW:])
        for index in sorted(edge):
            match = _PAGE_LABEL_LINE_RE.fullmatch(lines[index].strip())
            if match is not None:
                found.append(
                    _Candidate(page_index, index, "page_label", lines[index].strip(), "top", 0, keep=match.group("tag"))
                )
    if len({c.page_index for c in found}) < MIN_REPEAT_PAGES:
        return []
    return found


def strip_page_furniture(page_texts: list[str]) -> FurnitureStripResult:
    """Remove running headers/footers from the edges of an agreement's body pages.

    ``page_texts`` is the tagged text of each body page, in page order. Returns the
    texts with furniture lines dropped plus a record of each removed line.
    """
    all_candidates = [c for i, text in enumerate(page_texts) for c in _page_candidates(i, text)]
    accepted = _repeated([c for c in all_candidates if c.kind in ("toc_link", "doc_id", "exhibit_label")])
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
    for candidate in banners:
        removed_by_page.setdefault(candidate.page_index, set()).add(candidate.line_index)

    titles = _running_titles(
        [c for c in all_candidates if c.kind in ("titled_page_number", "title")],
        page_texts,
        numbered_pages,
        tagged,
    )
    title_lines_by_page: dict[int, set[int]] = {}
    for candidate in titles:
        title_lines_by_page.setdefault(candidate.page_index, set()).add(candidate.line_index)
    titles = [c for c in titles if outermost(c, title_lines_by_page)]
    for candidate in titles:
        candidate.kind = "running_title"
        removed_by_page.setdefault(candidate.page_index, set()).add(candidate.line_index)
    accepted += titles

    repeated = _repeated_edge_lines(page_texts, removed_by_page)
    for _ in range(2):
        repeated_lines_by_page: dict[int, set[int]] = {}
        for candidate in repeated:
            repeated_lines_by_page.setdefault(candidate.page_index, set()).add(candidate.line_index)
        repeated = [c for c in repeated if outermost(c, repeated_lines_by_page)]
    accepted += repeated

    already = {(c.page_index, c.line_index) for c in accepted}
    accepted += [c for c in _page_label_lines(page_texts) if (c.page_index, c.line_index) not in already]

    drop: dict[int, dict[int, _Candidate]] = {}
    for candidate in accepted:
        drop.setdefault(candidate.page_index, {})[candidate.line_index] = candidate

    result = FurnitureStripResult(page_texts=[])
    for page_index, text in enumerate(page_texts):
        page_drop = drop.get(page_index)
        if not page_drop:
            result.page_texts.append(text)
            continue
        kept: list[str] = []
        for line_index, line in enumerate(page_lines[page_index]):
            # A table's own heading stays; it still counted as an edge line for the rules above,
            # so keeping it changes no other decision.
            if line_index in page_drop and not (
                page_drop[line_index].kind == "toc_link" and _opens_toc_listing(page_lines[page_index], line_index)
            ):
                candidate = page_drop[line_index]
                result.removed.append(RemovedFurnitureLine(page_index=page_index, line=line.strip(), kind=candidate.kind))
                if candidate.keep is not None:
                    kept.append(candidate.keep)
                continue
            kept.append(line)
        result.page_texts.append("\n".join(kept))
    _strip_fused_furniture(result, accepted, tagged)
    return result


# --- Furniture fused into a body line -------------------------------------------
#
# Some extractions glue the running header or footer onto the first or last body
# line of a page ("NAI-5005005981v11 69 closing of the Merger ..."), or print the
# page number untagged as the first word of a one-line page. The rules below only
# remove text the line rules already confirmed as furniture on the same
# agreement (or a document ID repeated at the edges of most pages), only at the
# very start or end of an edge-window line, and only when the same text never
# appears anywhere else in the body pages.

MIN_FUSED_CHARS = 8
MIN_FUSED_REST_CHARS = 40
_NUMBER_SLOT = "\x00"
_SLOT_RE = r"(?:<page>\s*\d{1,4}\s*</page>|\d{1,4})"
_LEADING_TAGS_RE = re.compile(r"^(?:\s*<page>\s*\d{1,4}\s*</page>)*\s*", re.IGNORECASE)
_TRAILING_TAGS_RE = re.compile(r"\s*(?:<page>\s*\d{1,4}\s*</page>\s*)*$", re.IGNORECASE)
_KEPT_TAG_RE = re.compile(r"<page>\s*\d{1,4}\s*</page>", re.IGNORECASE)
# A document-management ID as one token: NAI-5005005981v11, 1002446029v14,
# EAST\1234567.2, US-DOCS\137120358.19, 4848-1716-0445.2, 4897-9174-6440.v17.
# Bare digit runs never count.
_FUSED_DOC_ID_RE = re.compile(
    r"(?:[A-Za-z][A-Za-z0-9_-]{0,11}[-\\/:_])?\d{6,}(?:\.?v\.?\d{1,4}|[._]\d{1,4}){1,2}"
    # Letters then 6+ digits and no version: all-caps letters, or a \ / : separator
    # (NAI-5005005981, EAST\1234567), never a word like "Loan-1234567".
    r"|[A-Z][A-Z0-9_-]{0,11}[-_]\d{6,}|[A-Za-z][A-Za-z0-9_-]{0,11}[\\/:]\d{6,}"
    r"|\d{4}-\d{4}-\d{4}(?:\.?v?\.?\d{1,4})?"
)


def _template_from_removed_line(candidate: _Candidate) -> tuple[re.Pattern[str], str, bool] | None:
    """A confirmed furniture line as a pattern whose page number may be tagged or bare."""
    text = " ".join(_CLEAN_PAGE_TAG_RE.sub(f" {_NUMBER_SLOT} ", candidate.text).split())
    tokens = text.split(" ")
    if candidate.kind in ("running_title", "doc_id"):
        tokens = [_NUMBER_SLOT if re.fullmatch(r"\d{1,3}", t) else t for t in tokens]
        if candidate.kind == "running_title" and _NUMBER_SLOT not in tokens:
            return None  # a bare title also reads as body text ("... this Agreement and Plan of Merger")
    literal = " ".join(t for t in tokens if t != _NUMBER_SLOT)
    is_doc_id = candidate.kind == "doc_id" or any(_FUSED_DOC_ID_RE.fullmatch(t) for t in tokens)
    if _NUMBER_SLOT not in tokens and not is_doc_id:
        return None  # words alone ("Classified as Secret") can also end a body sentence
    if len(literal.replace(" ", "")) < MIN_FUSED_CHARS and not is_doc_id:
        return None
    if not any(ch.isalpha() for ch in literal) and not is_doc_id:
        return None
    parts = [_SLOT_RE if token == _NUMBER_SLOT else re.escape(token) for token in tokens]
    pattern = ""
    for index, part in enumerate(parts):
        if index:
            pattern += "[ \t]+" if part != _SLOT_RE and parts[index - 1] != _SLOT_RE else "[ \t]*"
        pattern += part
    return re.compile(pattern), literal, is_doc_id


def _edge_line_indexes(lines: list[str]) -> set[int]:
    countable = _countable_indexes(lines)
    return set(countable[:EDGE_LINE_WINDOW]) | set(countable[-EDGE_LINE_WINDOW:])


def _heading_depths(lines: list[str]) -> list[tuple[int, int]]:
    """(depth before, depth after) of the open <article>/<section> tags per line."""
    depths: list[tuple[int, int]] = []
    depth = 0
    for line in lines:
        before = depth
        depth = max(depth + len(_HEADING_OPEN_RE.findall(line)) - len(_HEADING_CLOSE_RE.findall(line)), 0)
        depths.append((before, depth))
    return depths


def _anchored_spans(pattern: re.Pattern[str], line: str, before: int, after: int) -> list[tuple[int, int]]:
    """Spans where the pattern is the whole line, its start, or its end (outside any heading)."""
    spans: list[tuple[int, int]] = []
    lead = _LEADING_TAGS_RE.match(line)
    start = lead.end() if lead else 0
    trail = _TRAILING_TAGS_RE.search(line)
    end = trail.start() if trail else len(line)
    if start >= end:
        return spans
    # A match may run on over the trailing page tags (its number slot can be one).
    prefix = pattern.match(line, start)
    if prefix is not None and before == 0 and (prefix.end() >= end or line[prefix.end()].isspace()):
        spans.append(prefix.span())
    for match in pattern.finditer(line, start):
        if match.end() < end or after != 0 or (match.start() > start and not line[match.start() - 1].isspace()):
            continue
        if match.span() not in spans:
            spans.append(match.span())
    return spans


def _cut(line: str, spans: list[tuple[int, int]]) -> tuple[str, list[str]]:
    """The line without the spans (their page tags kept in place), and the removed texts."""
    removed: list[str] = []
    out = line
    for start, end in sorted(spans, reverse=True):
        piece = out[start:end]
        removed.append(" ".join(piece.split()))
        tags = " ".join(_KEPT_TAG_RE.findall(piece))
        out = out[:start] + (f" {tags} " if tags else " ") + out[end:]
    return " ".join(out.split()), removed


def _apply_fused(result: FurnitureStripResult, found: dict[tuple[int, int], list[tuple[int, int]]], kind: str) -> None:
    by_page: dict[int, dict[int, list[tuple[int, int]]]] = {}
    for (page_index, line_index), spans in found.items():
        by_page.setdefault(page_index, {})[line_index] = spans
    for page_index, page_found in by_page.items():
        lines = result.page_texts[page_index].split("\n")
        emptied: set[int] = set()
        for line_index, spans in page_found.items():
            new_line, removed = _cut(lines[line_index], spans)
            lines[line_index] = new_line
            if not new_line:
                emptied.add(line_index)  # the whole line was furniture: drop it like the line rules do
            for text in removed:
                result.removed.append(RemovedFurnitureLine(page_index=page_index, line=text, kind=kind))
        result.page_texts[page_index] = "\n".join(line for i, line in enumerate(lines) if i not in emptied)


def _fused_matches(
    page_texts: list[str], pattern: re.Pattern[str]
) -> dict[tuple[int, int], list[tuple[int, int]]] | None:
    """Anchored edge-line matches of the pattern, or None when it also occurs anywhere else."""
    found: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for page_index, text in enumerate(page_texts):
        lines = text.split("\n")
        edges = _edge_line_indexes(lines)
        depths = _heading_depths(lines)
        for line_index, line in enumerate(lines):
            occurrences = [m.span() for m in pattern.finditer(line)]
            if not occurrences:
                continue
            anchored = _anchored_spans(pattern, line, *depths[line_index]) if line_index in edges else []
            if any(span not in anchored for span in occurrences):
                return None  # the text also occurs inside the page or mid-line: it may be body text
            # A whole edge line the line rules left in place (a residue's kept
            # first copy, say) stays, unless its page number is untagged: the
            # line rules cannot match that variant.
            lead = _LEADING_TAGS_RE.match(line)
            trail = _TRAILING_TAGS_RE.search(line)
            core = (lead.end() if lead else 0, trail.start() if trail else len(line))
            anchored = [
                span for span in anchored
                if not (span[0] <= core[0] and span[1] >= core[1])
                or (_SLOT_RE in pattern.pattern and not _KEPT_TAG_RE.search(line[span[0]:span[1]]))
            ]
            if anchored:
                found[(page_index, line_index)] = anchored
    return found


def _strip_fused_furniture(result: FurnitureStripResult, accepted: list[_Candidate], tagged: dict[int, int]) -> None:
    # 1. Residues the line rules confirmed on this agreement, fused into an edge line.
    seen: set[str] = set()
    for candidate in accepted:
        if candidate.kind not in ("doc_id", "repeated_line", "running_title"):
            continue
        template = _template_from_removed_line(candidate)
        if template is None or template[0].pattern in seen:
            continue
        seen.add(template[0].pattern)
        found = _fused_matches(result.page_texts, template[0])
        if found:
            # Fused into body text the rest of the line is long. A short rest
            # ("Membership Interest Purchase and Sale Agreement" before
            # "(Project Dome II)") is a footer variant the line rules declined:
            # cutting part of it would leave the other part.
            found = {
                key: spans for key, spans in found.items()
                if not 0 < len(_cut(result.page_texts[key[0]].split("\n")[key[1]], spans)[0]) < MIN_FUSED_REST_CHARS
            }
        if found:
            _apply_fused(result, found, "fused_residue")

    # 2. A document ID fused with its page number at the start or end of edge lines.
    _strip_fused_doc_ids(result)

    # 3. An untagged page number as the first or last word of a page whose
    #    neighbors carry tagged numbers in the same sequence.
    _strip_fused_page_numbers(result, tagged)


def _strip_fused_doc_ids(result: FurnitureStripResult) -> None:
    page_texts = result.page_texts
    needed = max(MIN_TITLE_PAGES, MIN_TITLE_PAGE_SHARE * len(page_texts))
    pages_by_id: dict[str, set[int]] = {}
    for page_index, text in enumerate(page_texts):
        lines = text.split("\n")
        for line_index in _edge_line_indexes(lines):
            line = lines[line_index]
            lead = _LEADING_TAGS_RE.match(line)
            trail = _TRAILING_TAGS_RE.search(line)
            body = line[lead.end() if lead else 0:trail.start() if trail else len(line)].split()
            for token in (body[:2] + body[-2:]) if body else []:
                if _FUSED_DOC_ID_RE.fullmatch(token):
                    pages_by_id.setdefault(token, set()).add(page_index)
    for doc_id, pages in pages_by_id.items():
        if len(pages) < needed:
            continue
        escaped = re.escape(doc_id)
        pattern = re.compile(rf"(?<!\S)(?:\d{{1,3}}[ \t]+)?{escaped}(?:[ \t]+\d{{1,3}})?(?!\S)")
        found = _fused_matches(page_texts, pattern)
        if not found:
            continue
        # A page number beside the ID goes only when it fits one page sequence,
        # and only on a side where at least half of the ID's copies carry a
        # fitting number ("NAI-... 69", "33 WA ..."); otherwise the ID goes alone.
        copies: list[tuple[tuple[int, int], int, int, int, str, str]] = []
        offsets: dict[int, int] = {}
        for (page_index, line_index), spans in found.items():
            line = page_texts[page_index].split("\n")[line_index]
            for start, end in spans:
                text = line[start:end]
                id_abs = start + text.index(doc_id)
                head, tail = line[start:id_abs].strip(), line[id_abs + len(doc_id):end].strip()
                copies.append(((page_index, line_index), start, end, id_abs, head, tail))
                for number in (head, tail):
                    if number:
                        offsets[int(number) - page_index] = offsets.get(int(number) - page_index, 0) + 1
        best = max(offsets.items(), key=lambda item: item[1])[0] if offsets else None

        def fits(number: str, page_index: int) -> bool:
            return bool(number) and best is not None and int(number) - page_index == best

        # Pages that print their number in a <page> tag need no bare number.
        untagged = [c for c in copies if not _CLEAN_PAGE_TAG_RE.search(page_texts[c[0][0]])]
        share = MIN_TITLE_PAGE_SHARE * len(untagged)
        head_side = bool(untagged) and sum(1 for c in untagged if fits(c[4], c[0][0])) >= share
        tail_side = bool(untagged) and sum(1 for c in untagged if fits(c[5], c[0][0])) >= share
        trimmed: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for key, start, end, id_abs, head, tail in copies:
            new_start = start if head_side and fits(head, key[0]) else id_abs
            new_end = end if tail_side and fits(tail, key[0]) else id_abs + len(doc_id)
            trimmed.setdefault(key, []).append((new_start, new_end))
        _apply_fused(result, _with_footer_prefix(page_texts, trimmed, best), "fused_doc_id")


# A token printed before a document ID in a footer: "WA", "Active", "DB1/", "Dok.nr.".
_FOOTER_TOKEN_RE = re.compile(r"(?=\S*[A-Z0-9/:\\{}&|©–—-])\S{1,24}")
MAX_FOOTER_PREFIX_TOKENS = 12


def _token_before(line: str, start: int) -> tuple[str, int] | None:
    match = re.search(r"(\S+)[ \t]+$", line[:start])
    return (match.group(1), match.start(1)) if match else None


def _token_after(line: str, end: int) -> tuple[str, int] | None:
    match = re.match(r"[ \t]+(\S+)", line[end:])
    return (match.group(1), end + match.end(1)) if match else None


# A page number printed in a footer: 12, -12-, - 12 -.
_FOOTER_NUMBER_RE = re.compile(r"[-–—]?(?P<n>\d{1,3})[-–—]?")


def _with_footer_prefix(
    page_texts: list[str], spans_by_line: dict[tuple[int, int], list[tuple[int, int]]], offset: int | None
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Extend a fused ID over the rest of its footer.

    An ID at the end of a line grows backward and one at the start grows
    forward, token by token: a token goes when the same token sits there on at
    least 80% of those copies ("WA", "Active", "DB1/", "(c) King & Wood
    Mallesons", "CAN_DMS:"), or when the tokens there are page numbers ("12",
    "-12-") that fit one page sequence on at least half of them. Extension stops
    at the first token that is neither, so the body text the footer was glued to
    stays.
    """
    out = {key: list(spans) for key, spans in spans_by_line.items()}
    for direction in ("back", "forward"):
        copies: list[tuple[tuple[int, int], int, str]] = []  # (line key, span index, line)
        for key, spans in out.items():
            line = page_texts[key[0]].split("\n")[key[1]]
            for index, (start, end) in enumerate(spans):
                at_end = _TRAILING_TAGS_RE.fullmatch(line[end:]) is not None
                lead = _LEADING_TAGS_RE.match(line)
                at_start = start <= (lead.end() if lead else 0)
                if (direction == "back" and at_end and not at_start) or (direction == "forward" and at_start and not at_end):
                    copies.append((key, index, line))
        if len(copies) < MIN_TITLE_PAGES:
            continue
        snapshots = [{key: list(spans) for key, spans in out.items()}]
        steps: list[str | None] = []  # the shared tokens taken, None for a page-number step
        active = {(key, index) for key, index, _ in copies}
        for _ in range(MAX_FOOTER_PREFIX_TOKENS):
            tokens: dict[tuple[tuple[int, int], int], tuple[str, int]] = {}
            for key, index, line in copies:
                if (key, index) not in active:
                    continue
                start, end = out[key][index]
                found = _token_before(line, start) if direction == "back" else _token_after(line, end)
                if found is not None:
                    tokens[(key, index)] = found
            counts: dict[str, int] = {}
            offsets: dict[int, int] = {}
            for (key, _), (token, _) in tokens.items():
                counts[token] = counts.get(token, 0) + 1
                number = _FOOTER_NUMBER_RE.fullmatch(token)
                if number is not None:
                    value = int(number.group("n")) - key[0]
                    offsets[value] = offsets.get(value, 0) + 1
            token, count = max(counts.items(), key=lambda item: item[1]) if counts else ("", 0)
            if count >= 0.8 * len(copies) and _is_footer_token(token):
                take = {k for k, (t, _) in tokens.items() if t == token}
                steps.append(token)
            elif offsets and max(offsets.values()) >= MIN_TITLE_PAGE_SHARE * len(copies):
                best = max(offsets.items(), key=lambda item: item[1])[0]
                take = set()
                for k, (t, _) in tokens.items():
                    number = _FOOTER_NUMBER_RE.fullmatch(t)
                    if number is not None and int(number.group("n")) - k[0][0] == best:
                        take.add(k)
                steps.append(None)
            else:
                break
            for key, index in take:
                start, end = out[key][index]
                position = tokens[(key, index)][1]
                out[key][index] = (position, end) if direction == "back" else (start, position)
            active = take
            snapshots.append({key: list(spans) for key, spans in out.items()})
        # Keep the longest extension whose footer words occur nowhere else.
        kept = len(steps)
        while kept and not _footer_text_is_unique(page_texts, snapshots[kept], steps[:kept], direction):
            kept -= 1
        out = snapshots[kept]
    return out


def _is_footer_token(token: str) -> bool:
    """A shared token that may belong to a footer, never a word that ends a clause
    ("Parties.", "Company,"); "Dok.nr.", "CAN_DMS:" and "DB1/" still qualify."""
    if not _FOOTER_TOKEN_RE.fullmatch(token) or _FOOTER_NUMBER_RE.fullmatch(token):
        return False
    return not (token[-1] in ".;," and token[:-1].isalpha())


def _footer_text_is_unique(
    page_texts: list[str],
    spans_by_line: dict[tuple[int, int], list[tuple[int, int]]],
    steps: list[str | None],
    direction: str,
) -> bool:
    """The footer words taken beside the ID (between page numbers) occur nowhere
    in the body pages except as part of a footer: on a line that carries the ID,
    an occurrence counts as footer only when nothing but page numbers and the
    footer's own tokens separate it from the ID (one-line pages hold their whole
    page text on that line)."""
    step_tokens = {step for step in steps if step is not None}

    def beside_the_id(line: str, start: int, end: int, spans: list[tuple[int, int]]) -> bool:
        for span_start, span_end in spans:
            if start < span_end and end > span_start:
                return True
            gap = line[end:span_start] if end <= span_start else line[span_end:start]
            gap = _KEPT_TAG_RE.sub(" ", gap)  # a tagged page number is a page number too
            if all(_FOOTER_NUMBER_RE.fullmatch(token) or token in step_tokens for token in gap.split()):
                return True
        return False

    ordered = steps[::-1] if direction == "back" else steps
    phrases: list[str] = []
    current: list[str] = []
    for step in ordered + [None]:
        if step is None:
            if current:
                phrases.append(" ".join(current))
            current = []
        else:
            current.append(step)
    for phrase in phrases:
        pattern = re.compile(r"(?<!\S)" + r"\s+".join(re.escape(word) for word in phrase.split(" ")) + r"(?!\S)")
        for page_index, text in enumerate(page_texts):
            for line_index, line in enumerate(text.split("\n")):
                spans = spans_by_line.get((page_index, line_index), [])
                for match in pattern.finditer(line):
                    if not beside_the_id(line, match.start(), match.end(), spans):
                        return False
    return True


# A number that opens a page and is followed by one of these words is a quantity
# ("12 months after ..."); one that ends a page after one of these is a reference
# or amount ("... shall not exceed 12", "... Section 12").
_FUSED_NUMBER_UNIT_RE = re.compile(
    r"(?:months?|days?|years?|weeks?|hours?|business|calendar|trading|percent|%|shares?|million|billion)\W*",
    re.IGNORECASE,
)
_FUSED_NUMBER_LEAD_RE = re.compile(r"(?:exceeds?|than|within|least|approximately|sections?|article|\$)", re.IGNORECASE)


def _strip_fused_page_numbers(result: FurnitureStripResult, tagged: dict[int, int]) -> None:
    numbered = {r.page_index for r in result.removed if r.kind in ("page_number", "fused_doc_id")}
    for page_index, text in enumerate(result.page_texts):
        if page_index in tagged or page_index in numbered or _CLEAN_PAGE_TAG_RE.search(text):
            continue
        before = [tagged[i] - i for i in (page_index - 1, page_index - 2) if i in tagged][:1]
        after = [tagged[i] - i for i in (page_index + 1, page_index + 2) if i in tagged][:1]
        if not before or not after or before[0] != after[0]:
            continue
        expected = str(page_index + before[0])
        lines = text.split("\n")
        countable = _countable_indexes(lines)
        if not countable:
            continue
        depths = _heading_depths(lines)
        first, last = countable[0], countable[-1]
        top, bottom = lines[first].split(), lines[last].split()
        # The page's number already printed elsewhere in its edge window (a bare
        # line, or the word at the other edge): the fused word is text.
        if any(lines[i].strip() == expected for i in _edge_line_indexes(lines)):
            continue
        if top and bottom and top[0] == expected and bottom[-1] == expected:
            continue
        top_hit = len(top) > 3 and top[0] == expected and depths[first][0] == 0
        bottom_hit = len(bottom) > 3 and bottom[-1] == expected and depths[last][1] == 0
        if top_hit and not _FUSED_NUMBER_UNIT_RE.fullmatch(top[1]):
            lines[first] = " ".join(top[1:])
        elif bottom_hit and not _FUSED_NUMBER_LEAD_RE.fullmatch(bottom[-2]):
            lines[last] = " ".join(bottom[:-1])
        else:
            continue
        result.page_texts[page_index] = "\n".join(lines)
        result.removed.append(RemovedFurnitureLine(page_index=page_index, line=expected, kind="fused_page_number"))


# --- Signature and back-matter pages ---------------------------------------------
#
# Signature pages and back matter are rendered page by page, not joined into
# sections, so the line rules above never see them. Their printed page numbers
# ("55", "- 97 -", "-40-", "A-3") and the body's confirmed running lines (the
# "Table of Contents" back-link, a document ID) were served as text. The strip
# here is narrower than the body rules: it only takes lines from the very edge of
# a page, and only page numbers that fit a sequence and lines identical to
# furniture the body rules removed. Signer text is never a whole-line page number
# or an exact copy of a body furniture line.

# Body furniture kinds whose exact text is furniture on any page of the agreement:
# a running title is the agreement's own printed title ("Share Purchase
# Agreement"). Other repeated lines are left out: a deal or company name repeated
# at the top of body pages can head a signature block.
NONBODY_LEARNED_KINDS = ("toc_link", "doc_id", "banner", "exhibit_label", "running_title")


def _normalized(line: str) -> str:
    return " ".join(line.split())


def _body_page_number_anchors(
    body_page_texts: list[str], body_page_orders: list[int], body_removed: list[RemovedFurnitureLine]
) -> list[tuple[int, int]]:
    """(page_order, printed page number) for each body page whose number is known."""
    by_index: dict[int, int] = {}
    for removed in body_removed:
        if removed.kind == "page_number":
            by_index[removed.page_index] = _page_number_value(removed.line)
    by_index.update(_tagged_page_numbers(body_page_texts))
    return sorted((body_page_orders[i], value) for i, value in by_index.items() if i < len(body_page_orders))


def _fits_body_sequence(order: int, value: int, anchors: list[tuple[int, int]]) -> bool:
    """The number continues the body's page numbering, read from the nearest numbered body page."""
    if not anchors:
        return False
    anchor_order, anchor_value = min(anchors, key=lambda a: (abs(a[0] - order), -a[0]))
    return abs(value - (anchor_value + order - anchor_order)) <= PAGE_OFFSET_TOLERANCE


def _edge_page_numbers(lines: list[str]) -> dict[str, tuple[int, str, int]]:
    """The outermost countable line on each edge when it is a page number: side -> (line, shape, value)."""
    countable = _countable_indexes(lines)
    found: dict[str, tuple[int, str, int]] = {}
    for side, index in (("top", countable[0] if countable else None), ("bottom", countable[-1] if countable else None)):
        if index is None:
            continue
        if furniture_kind(lines[index]) == "page_number":
            found[side] = (index, _line_shape(lines[index].strip()), _page_number_value(lines[index].strip()))
    return found


def strip_nonbody_page_furniture(
    page_texts: list[str],
    page_orders: list[int],
    *,
    body_page_texts: list[str],
    body_page_orders: list[int],
    body_removed: list[RemovedFurnitureLine],
) -> list[str]:
    """Remove page numbers and confirmed body furniture from the edges of sig and back-matter pages.

    ``page_texts`` and ``page_orders`` are the agreement's signature and
    back-matter pages in page order; ``body_page_texts`` (before the body strip),
    ``body_page_orders`` and ``body_removed`` describe its body pages. A line goes
    only when it is one of the outermost lines of its page (lines removed by this
    rule do not count, up to ``EDGE_LINE_WINDOW`` per edge) and either

    - is a page number ("55", "- 97 -", "Page 39", "A-3") that continues the body's
      numbering from the nearest numbered body page (within
      ``PAGE_OFFSET_TOLERANCE``), or that rises by exactly the page distance
      against the page number on the same edge of the neighboring sig or
      back-matter page, in the same shape ("A-1", "A-2"; "2", "3"); or
    - equals a line the body rules removed as a Table of Contents link, a
      document ID, a banner, an exhibit label or a running title.
    """
    anchors = _body_page_number_anchors(body_page_texts, body_page_orders, body_removed)
    learned = {_normalized(r.line) for r in body_removed if r.kind in NONBODY_LEARNED_KINDS}
    page_lines = [text.split("\n") for text in page_texts]
    edges = [_edge_page_numbers(lines) for lines in page_lines]

    def chained(k: int, side: str) -> bool:
        _, shape, value = edges[k][side]
        for j in (k - 1, k + 1):
            if 0 <= j < len(edges) and side in edges[j]:
                _, other_shape, other_value = edges[j][side]
                if other_shape == shape and other_value - value == page_orders[j] - page_orders[k]:
                    return True
        return False

    result: list[str] = []
    for k, lines in enumerate(page_lines):
        drop: set[int] = set()
        countable = _countable_indexes(lines)
        for side, ordered in (("top", countable), ("bottom", list(reversed(countable)))):
            for rank, index in enumerate(ordered[:EDGE_LINE_WINDOW]):
                if index in drop:
                    break
                line = lines[index].strip()
                removable = _normalized(line) in learned
                if not removable and furniture_kind(line) == "page_number":
                    value = _page_number_value(line)
                    outermost = rank == 0 and edges[k].get(side, (None,))[0] == index
                    removable = _fits_body_sequence(page_orders[k], value, anchors) or (
                        outermost and chained(k, side)
                    )
                if not removable:
                    break
                drop.add(index)
        if len(drop) == len(countable):
            drop = set()  # never empty a page: a lone number is then the page's content
        # A table's own heading stays (it still counted as an edge line above).
        drop = {i for i in drop if not (furniture_kind(lines[i]) == "toc_link" and _opens_toc_listing(lines, i))}
        if not drop:
            result.append(page_texts[k])
            continue
        # The blank lines that separated a dropped edge line go with it.
        result.append("\n".join(line for i, line in enumerate(lines) if i not in drop).strip("\n"))
    return result
