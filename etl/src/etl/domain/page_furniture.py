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

import difflib
import html
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

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


@lru_cache(maxsize=1 << 16)
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


# A whole-line page number ("Page 91", "Page-12", "Page 3 of 40"), not a column header
# followed by an entry number ("Page 5.6.5 Treasurer 15").
_PAGE_NUMBER_WORD_RE = re.compile(r"page\s*[-\u2013]?\s*\d{1,4}(?:\s+of\s+\d{1,4})?", re.IGNORECASE)


def _opens_toc_listing(lines: list[str], index: int, skip: set[int] | frozenset[int] = frozenset()) -> bool:
    """True when the line after ``index`` starts a printed table of contents, so a
    "Table of Contents" line at ``index`` is that table's heading, not the back-link.
    Lines in ``skip`` (already dropped as furniture) are passed over, except another
    "Table of Contents" line, which makes this one the back-link above the heading; a
    page number ("Page 3 of 40") is not a "Page" column header."""
    for offset, line in enumerate(lines[index + 1:], start=index + 1):
        stripped = line.strip()
        if not stripped:
            continue
        if furniture_kind(stripped) == "toc_link":
            return False
        if offset in skip:
            continue
        if furniture_kind(stripped) == "page_number" or _PAGE_NUMBER_WORD_RE.fullmatch(stripped):
            return False
        if _TOC_PAGE_HEADER_RE.match(stripped):
            return True
        return bool(_TOC_ARTICLE_ENTRY_RE.match(stripped)) and len(_TOC_ENTRY_PAGE_RE.findall(stripped)) >= 2
    return False


def _toc_heading_lines(
    page_lines: list[list[str]], drops: list[set[int]], consecutive: Callable[[int], bool]
) -> list[set[int]]:
    """The dropped "Table of Contents" lines to keep as a printed table's heading, per page.

    A table that runs over several pages repeats its "Page" column header, and the
    back-link above it on the continuation pages is still furniture, so a heading is
    kept only on a page whose previous page (``consecutive(k)``) opens no listing.
    """
    opens = [
        {i for i, line in enumerate(lines) if furniture_kind(line) == "toc_link" and _opens_toc_listing(lines, i, drop)}
        for lines, drop in zip(page_lines, drops)
    ]
    return [
        set() if k > 0 and consecutive(k) and opens[k - 1] else opens[k] & drops[k]
        for k in range(len(page_lines))
    ]


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

    toc_headings = _toc_heading_lines(
        page_lines, [set(drop.get(k, {})) for k in range(len(page_texts))], lambda k: True
    )
    result = FurnitureStripResult(page_texts=[])
    for page_index, text in enumerate(page_texts):
        page_drop = drop.get(page_index)
        if not page_drop:
            result.page_texts.append(text)
            continue
        kept: list[str] = []
        for line_index, line in enumerate(page_lines[page_index]):
            # A table's own heading stays; it still counted as an edge line for the line rules above.
            if line_index in page_drop and not (
                page_drop[line_index].kind == "toc_link" and line_index in toc_headings[page_index]
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
# a page, and only page numbers that fit a sequence (the body's numbering, a
# neighbor's number, or the page's place in its exhibit or block of sig pages) and
# lines identical to furniture the body rules removed. Signer text is never a
# whole-line page number or an exact copy of a body furniture line.

# Body furniture kinds whose exact text is furniture on any page of the agreement:
# a running title is the agreement's own printed title ("Share Purchase
# Agreement"). Other repeated lines are left out here: a deal or company name
# repeated at the top of body pages can head a signature block. A repeated line
# that the body prints nowhere else (a law-firm footer) is handled by
# ``_learned_running_lines``.
NONBODY_LEARNED_KINDS = ("toc_link", "doc_id", "banner", "exhibit_label", "running_title")


def _normalized(line: str) -> str:
    return " ".join(line.split())


def _body_page_number_anchors(
    body_page_texts: list[str], body_page_orders: list[int], body_removed: list[RemovedFurnitureLine]
) -> list[list[tuple[int, int]]]:
    """(page_order, printed page number) for the body pages, one list per numbering.

    The tagged ``<page>`` numbers are one numbering (the last tag of a page tagged at
    both edges is another), and each shape of bare number the body rules removed is
    another: a filing can print a sheet counter at the top of each page ("65") and
    the agreement's own number at the bottom ("-60-").
    """
    families: dict[str, dict[int, int]] = {}
    for removed in body_removed:
        if removed.kind == "page_number":
            shape = "bare:" + _line_shape(removed.line.strip())
            families.setdefault(shape, {})[removed.page_index] = _page_number_value(removed.line.strip())
    tagged = _tagged_page_numbers(body_page_texts)
    if tagged:
        families["tagged"] = tagged
    # A page tagged at both edges: the last tag is a numbering of its own.
    last: dict[int, int] = {}
    for index, text in enumerate(body_page_texts):
        matches = list(_CLEAN_PAGE_TAG_RE.finditer(text))
        if len(matches) >= 2:
            last[index] = int(matches[-1].group("dashed") or matches[-1].group("plain"))
    last = {i: v for i, v in last.items() if last.get(i - 1) != v and last.get(i + 1) != v}
    if last:
        families["tagged_last"] = last
    return [
        sorted((body_page_orders[i], value) for i, value in family.items() if i < len(body_page_orders))
        for family in families.values()
    ]


def _fits_body_sequence(order: int, value: int, anchors: list[list[tuple[int, int]]]) -> bool:
    """The number continues one of the body's page numberings, read from its nearest numbered body page."""
    for family in anchors:
        if not family:
            continue
        anchor_order, anchor_value = min(family, key=lambda a: (abs(a[0] - order), -a[0]))
        if abs(value - (anchor_value + order - anchor_order)) <= PAGE_OFFSET_TOLERANCE:
            return True
    return False


# Page-number shapes beyond the body pattern, read only on sig and back-matter pages:
# a label with a space after the hyphen ("A- 2", "EX- 1", "I- 1"), a two-level label
# ("A-3-2"), a dashed label ("- A-2 -", "-A-1-"), and a lowercase roman number
# ("i", "-ii-") on the front pages of an exhibit.
_SPACED_LABEL_RE = re.compile(r"(?P<prefix>[A-Z]{1,3}(?:-\d{1,2})?)\s?[-–—]\s?(?P<n>\d{1,3})")
_DASHED_LABEL_RE = re.compile(r"[-–—]\s?(?P<prefix>[A-Z]{1,3})\s?[-–—]\s?(?P<n>\d{1,3})\s?[-–—]")
_ROMAN_LINE_RE = re.compile(r"(?P<lead>[-–—]\s*)?(?P<roman>[ivxl]{1,7})(?P<trail>\s*[-–—])?")
_ROMAN_DIGITS = {"i": 1, "v": 5, "x": 10, "l": 50}
MAX_ROMAN_PAGE = 40


def _roman_value(text: str, limit: int = MAX_ROMAN_PAGE) -> int | None:
    """The value of a well-formed lowercase roman number up to ``limit``."""
    total = 0
    for index, ch in enumerate(text):
        value = _ROMAN_DIGITS[ch]
        if index + 1 < len(text) and _ROMAN_DIGITS[text[index + 1]] > value:
            total -= value
        else:
            total += value
    if not 1 <= total <= limit or _to_roman(total) != text:
        return None
    return total


def _to_roman(value: int) -> str:
    out = ""
    for number, letters in ((50, "l"), (40, "xl"), (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i")):
        while value >= number:
            out += letters
            value -= number
    return out


@dataclass(frozen=True)
class _NonbodyNumber:
    index: int
    shape: str
    value: int
    roman: bool


def _nonbody_number(line: str) -> tuple[str, int, bool] | None:
    """(shape, value, is_roman) when the whole line is a printed page number or label."""
    stripped = " ".join(line.split())
    match = _SPACED_LABEL_RE.fullmatch(stripped)
    if match is not None:
        return match.group("prefix") + "-#", int(match.group("n")), False
    match = _DASHED_LABEL_RE.fullmatch(stripped)
    if match is not None:
        return "-" + match.group("prefix") + "-#-", int(match.group("n")), False
    if furniture_kind(stripped) == "page_number":
        return _line_shape(stripped), _page_number_value(stripped), False
    match = _ROMAN_LINE_RE.fullmatch(stripped)
    if match is not None:
        value = _roman_value(match.group("roman"))
        if value is not None:
            shape = ("-" if match.group("lead") else "") + "roman" + ("-" if match.group("trail") else "")
            return shape, value, True
    return None


def _is_doc_id_line(line: str) -> bool:
    stripped = line.strip()
    return furniture_kind(stripped) == "doc_id" or _FUSED_DOC_ID_RE.fullmatch(stripped) is not None


# The first line of an exhibit, annex or schedule ("EXHIBIT A", "Annex I", "SCHEDULE C
# FORM OF RESOLUTIONS", "Annexure B", "SCHEDULE “C”", "-SCHEDULE A-", "EXHIBIT A:
# EXAMPLES"), and of a printed table of contents.
_EXHIBIT_HEADING_RE = re.compile(
    r"(?:[-–—\[(]\s*)?(?:exhibit|annex(?:ure)?|schedule|appendix|attachment)\s+"
    r"[“\"‘'(]?[A-Z0-9][A-Za-z0-9.\-]*[”\"’')]?[:.]?(?:\s.{0,90}|\s*[-–—\])])?",
    re.IGNORECASE,
)
_CONTENTS_HEADING_RE = re.compile(r"(?:table\s+of\s+)?contents(?:\s*\((?:continued|cont[’']d)\))?", re.IGNORECASE)


_COVER_NOTE_RE = re.compile(r"[\[(][^\])]{1,60}[\])]")


def _is_cover_line(line: str) -> bool:
    """A cover-sheet line: a heading or title ("EXHIBIT C", "Form of Lock-Up Agreement") or a
    bracketed note ("(See attached)"), never a page number or a sentence."""
    stripped = line.strip()
    if _COVER_NOTE_RE.fullmatch(stripped):
        return True
    return _nonbody_number(stripped) is None and len(stripped) <= 100 and stripped[-1] not in ".;:,"


def _first_text_line(lines: list[str]) -> str | None:
    """The first countable line that is not a page number."""
    for index in _countable_indexes(lines):
        if _nonbody_number(lines[index]) is None:
            return lines[index].strip()
    return None


# A draft-status line printed above an exhibit heading ("FINAL FORM", "EXECUTION VERSION").
_ABOVE_HEADING_BANNER_RE = re.compile(r"(?:final\s+)?form|execution\s+(?:version|copy)|(?:strictly\s+)?confidential", re.IGNORECASE)


def _opens_exhibit(lines: list[str]) -> bool:
    """The page's first text line (or the one under a draft-status banner) is an exhibit heading."""
    texts = [lines[i].strip() for i in _countable_indexes(lines) if _nonbody_number(lines[i]) is None][:2]
    if texts and _ABOVE_HEADING_BANNER_RE.fullmatch(texts[0]):
        texts = texts[1:]
    return bool(texts) and _EXHIBIT_HEADING_RE.fullmatch(texts[0]) is not None


def _nonbody_positions(
    page_lines: list[list[str]], page_orders: list[int], page_types: list[str]
) -> tuple[list[set[int]], list[int | None]]:
    """Per page, the page numbers its own run would print, and its place in a table of contents.

    A run is a block of consecutive signature pages, or an exhibit: back-matter
    pages from one that opens with an exhibit heading ("EXHIBIT B") to the next.
    The n-th page of a run is page n; when the run opens with a cover sheet (three
    lines or fewer, each a heading, a title or a bracketed note such as "(See
    attached)", and no number of its own) the numbering may start after it.
    A table of contents (a page that opens with "Table of Contents") starts its
    own count for roman numbers.
    """
    positions: list[set[int]] = []
    contents: list[int | None] = []
    start: int | None = None
    contents_start: int | None = None
    for k, lines in enumerate(page_lines):
        consecutive = k > 0 and page_orders[k] - page_orders[k - 1] == 1 and page_types[k] == page_types[k - 1]
        first = _first_text_line(lines)
        opens_exhibit = page_types[k] != "sig" and _opens_exhibit(lines)
        if not consecutive or opens_exhibit or start is None:
            start = k
            contents_start = None
        if first is not None and _CONTENTS_HEADING_RE.fullmatch(first) is not None:
            contents_start = k
        position = page_orders[k] - page_orders[start] + 1
        allowed = {position}
        start_lines = page_lines[start]
        cover = len(_countable_indexes(start_lines)) <= EDGE_LINE_WINDOW and all(
            _is_cover_line(start_lines[i]) for i in _countable_indexes(start_lines)
        )
        if cover and position > 1:
            allowed.add(position - 1)
        positions.append(allowed)
        contents.append(None if contents_start is None else page_orders[k] - page_orders[contents_start] + 1)
    return positions, contents


def _learned_running_lines(
    body_page_texts: list[str], body_removed: list[RemovedFurnitureLine]
) -> dict[str, set[str]]:
    """Repeated body edge lines (a law-firm footer) that are safe to take off sig and back pages: side -> lines.

    The body rules removed the line from at least half of the body pages (and
    ``MIN_TITLE_PAGES``), and the body never prints it anywhere else: a party or
    deal name repeated as a running header also appears in the preamble, the
    notices clause or the signature blocks, so it never qualifies.
    """
    copies: dict[str, set[int]] = {}
    for removed in body_removed:
        if removed.kind == "repeated_line":
            copies.setdefault(_normalized(removed.line), set()).add(removed.page_index)
    need = max(MIN_TITLE_PAGES, (len(body_page_texts) + 1) // 2)
    joined = "\n".join(" ".join(text.split()) for text in body_page_texts).lower()
    sides: dict[str, set[str]] = {"top": set(), "bottom": set()}
    for line, pages in copies.items():
        if len(pages) < need or not any(ch.isalpha() for ch in line):
            continue
        if joined.count(line.lower()) != len(pages):
            continue
        for side in ("top", "bottom"):
            on_side = 0
            for index in pages:
                lines = body_page_texts[index].split("\n")
                countable = _countable_indexes(lines)
                half = (len(countable) + 1) // 2
                window = countable[:min(EDGE_LINE_WINDOW, half)] if side == "top" else countable[half:][-EDGE_LINE_WINDOW:]
                on_side += any(_normalized(lines[i]) == line for i in window)
            if on_side * 2 >= len(pages):
                sides[side].add(line)
    return sides


def strip_nonbody_page_furniture(
    page_texts: list[str],
    page_orders: list[int],
    *,
    body_page_texts: list[str],
    body_page_orders: list[int],
    body_removed: list[RemovedFurnitureLine],
    page_types: list[str] | None = None,
) -> list[str]:
    """Remove page numbers and confirmed body furniture from the edges of sig and back-matter pages.

    ``page_texts``, ``page_orders`` and ``page_types`` ("sig" or "back_matter")
    are the agreement's signature and back-matter pages in page order;
    ``body_page_texts`` (before the body strip), ``body_page_orders`` and
    ``body_removed`` describe its body pages. A line goes only when it is one of
    the outermost lines of its page (lines removed by this rule do not count, up to
    ``EDGE_LINE_WINDOW`` per edge) and either

    - equals a line the body rules removed as a Table of Contents link, a
      document ID, a banner, an exhibit label or a running title; or a repeated
      body edge line (a law-firm footer) that the body prints nowhere else, on the
      edge where the body printed it;
    - is a document ID ("4892-0619-9513") repeated at the same edge of
      ``MIN_REPEAT_PAGES`` of these pages; or
    - is a page number ("55", "- 97 -", "Page 39", "A-3", "A- 3", "A-3-2",
      "- A-2 -", or a lowercase roman "ii") that

      - continues one of the body's numberings from its nearest numbered body
        page (within ``PAGE_OFFSET_TOLERANCE``; not roman numbers); or, when it is
        the outermost number on its edge (a document ID beyond it does not count),
      - rises by exactly the page distance against the number on the same edge of
        a page one or two places away, in the same shape ("A-1", "A-2"; "2", "3";
        "i", "ii"); or
      - equals the page's place in its own run: the n-th page of a block of
        signature pages or of an exhibit (from the page that opens with "EXHIBIT
        B" or "Annex I") prints n, or n - 1 after a cover sheet; a roman number
        counts from the page that opens with "Table of Contents" (or the run).
        Only at the bottom edge: a lone number at the top of a page can be a
        footnote marker carried over from the page before.

    A page is never emptied: a lone number is then the page's content.
    """
    types = page_types if page_types is not None else ["back_matter"] * len(page_texts)
    anchors = _body_page_number_anchors(body_page_texts, body_page_orders, body_removed)
    learned = {_normalized(r.line) for r in body_removed if r.kind in NONBODY_LEARNED_KINDS}
    running = _learned_running_lines(body_page_texts, body_removed)
    page_lines = [text.split("\n") for text in page_texts]
    positions, contents = _nonbody_positions(page_lines, page_orders, types)

    def edge_order(lines: list[str], side: str) -> list[int]:
        countable = _countable_indexes(lines)
        return countable if side == "top" else list(reversed(countable))

    def is_learned(line: str, side: str) -> bool:
        normalized = _normalized(line)
        return normalized in learned or normalized in running[side]

    # Document IDs printed on the sig and back pages themselves.
    id_counts: dict[tuple[str, str], int] = {}
    for lines in page_lines:
        for side in ("top", "bottom"):
            for index in edge_order(lines, side)[:EDGE_LINE_WINDOW]:
                if _is_doc_id_line(lines[index]):
                    key = (side, _normalized(lines[index]))
                    id_counts[key] = id_counts.get(key, 0) + 1
    repeated_ids = {key for key, count in id_counts.items() if count >= MIN_REPEAT_PAGES}

    def passable(line: str, side: str) -> bool:
        return is_learned(line, side) or _is_doc_id_line(line)

    def edge_number(lines: list[str], side: str) -> _NonbodyNumber | None:
        """The outermost page number on this edge, passing over learned lines and document IDs."""
        for index in edge_order(lines, side)[:EDGE_LINE_WINDOW]:
            parsed = _nonbody_number(lines[index])
            if parsed is not None:
                return _NonbodyNumber(index, *parsed)
            if not passable(lines[index], side):
                return None
        return None

    edges = [{side: edge_number(lines, side) for side in ("top", "bottom")} for lines in page_lines]

    def chained(k: int, side: str) -> bool:
        number = edges[k][side]
        assert number is not None
        for j in (k - 1, k + 1, k - 2, k + 2):
            other = edges[j][side] if 0 <= j < len(edges) else None
            if other is not None and other.shape == number.shape and other.value - number.value == page_orders[j] - page_orders[k]:
                return True
        return False

    def removable_number(k: int, side: str, index: int, shape: str, value: int, roman: bool) -> bool:
        if not roman and _fits_body_sequence(page_orders[k], value, anchors):
            return True
        number = edges[k][side]
        if number is None or number.index != index:
            return False
        if chained(k, side):
            return True
        if side == "top":
            return False  # a lone "1" or "2" at the top can be a footnote marker carried over
        if roman:
            return value in ({contents[k]} if contents[k] is not None else positions[k])
        return value in positions[k]

    drops: list[set[int]] = []
    for k, lines in enumerate(page_lines):
        drop: set[int] = set()
        countable = _countable_indexes(lines)
        for side in ("top", "bottom"):
            for index in edge_order(lines, side)[:EDGE_LINE_WINDOW]:
                if index in drop:
                    break
                line = lines[index].strip()
                if is_learned(line, side) or (side, _normalized(line)) in repeated_ids:
                    drop.add(index)
                    continue
                parsed = _nonbody_number(line)
                if parsed is not None and removable_number(k, side, index, *parsed):
                    drop.add(index)
                    continue
                if _is_doc_id_line(line):
                    continue  # an ID that stays: the number under it can still go
                break
        if len(drop) == len(countable):
            drop = set()  # never empty a page: a lone number is then the page's content
        drops.append(drop)

    # A table's own heading stays (it still counted as an edge line above).
    toc_headings = _toc_heading_lines(page_lines, drops, lambda k: page_orders[k] - page_orders[k - 1] == 1)
    result: list[str] = []
    for k, lines in enumerate(page_lines):
        drop = drops[k] - toc_headings[k]
        if not drop:
            result.append(page_texts[k])
            continue
        # The blank lines that separated a dropped edge line go with it.
        result.append("\n".join(line for i, line in enumerate(lines) if i not in drop).strip("\n"))
    return result


# --- Furniture confirmed by the page as filed --------------------------------------
#
# The rules above read only the tagged text. The rules below also read the page as
# filed (``pages.raw_page_content``). They run last, on the output of the body and
# sig/back rules, over all of an agreement's pages (front matter, contents, body,
# sig and back matter) in page order. A tagged line goes only when the filing
# prints the same text, as its own line (or run of lines), among the first or last
# ``RAW_EDGE_WINDOW`` lines of the same page, and the tagged line is one of the
# outermost lines of its page (up to ``EDGE_LINE_WINDOW`` per edge, walking in
# from the edge past the lines this pass removes and stopping at the first it
# keeps).
#
# - Page numbers ("-77-", "- 12", "103", "xlv", "41.", ". 50", "7/107", "Page 3
#   of 13", "A-6", "App-2", "- B-2 -", "Appendix A - Page 14", "Annex A-3") must
#   also fit a numbering: they equal the page's own tagged number; or a page one
#   or two places away prints (as filed, on the same edge) the same shape rising
#   by exactly the page distance; or, at the bottom edge only, not roman, and on
#   a page with no tagged number of its own, the number continues the nearest
#   tagged page number exactly. On front-matter and contents pages only the
#   chain counts (a lone number there can be an entry's page). A label that
#   restarts per exhibit ("S-1") needs the chain or is page 1, and counts only at
#   the bottom, never as its page's first line ("EXHIBIT C-1" is a caption). A
#   plain number right above an unnumbered heading stays (a clause number).
# - A tagged page number glued to a stray "." or "|", a total ("/107", "of 92")
#   or its own number again ("<page>41</page>.", ". <page>50</page>", "102
#   <page>102</page>"), or printed with its label ("Page <page>2</page> of 92"),
#   keeps only the tag, and a lone "." or "|" line beside the page number goes.
# - A repeated line (running header or footer, legend, document ID): its shape
#   (digits collapsed) is printed as its own line at the edge of at least
#   ``MIN_RAW_REPEAT_PAGES`` pages and of at least ``MIN_RAW_REPEAT_SHARE`` of the
#   body pages, or of the other pages from its first copy to its last (an
#   exhibit's own footer). Only copies on its dominant edge go, and tagged page
#   numbers on the line stay. Every number in it must be the same on every copy
#   (a document ID, "of 87") or a page number that rises with the pages: a date,
#   an amount or a clause number makes it text. The first copy of a line without
#   digits stays (a cover title, a table's first header). Never a tagged heading,
#   a line inside one, an ARTICLE/Section/Exhibit line, a signature-page legend
#   ("[Signature Page to ...]"), page-break boilerplate, a signature-block line,
#   an enumerator or a single word. A bracketed line, or a legend line shorter
#   than ``MIN_RAW_LEGEND_CHARS`` ("[REDACTED]", "[Intentionally Omitted]", "[Exhibit
#   has been omitted ...]"), is what its page holds unless it is a long legend
#   repeated on the body pages. Off the body and back pages (sig, front,
#   contents), and for a line repeated only off the body, it also needs a
#   document ID, a page number or a legend word: a party name heading each
#   counterpart signature page is signature text.
# - A law-firm document ID ("|US-DOCS\\140904150.6||") printed word for word as
#   its own edge line on two or more pages; and on contents pages a "Page"
#   column header, or the contents title repeated after the first contents page.
# - A filing that prints a page as one run of text glues its header or footer to
#   the text. A run of up to ``MAX_RAW_AFFIX_TOKENS`` words at the start (or end)
#   of the filed page, repeated in shape at the same end of the same share of
#   pages, is cut from the outermost tagged line when it holds a document ID (six
#   or more digits, "4124-8327-5548") or a page number that rises with the pages
#   ("EXECUTION VERSION 2"), under the same number rule. Without a document ID it
#   must be the filing's whole first (or last) line, unless the page is one run:
#   a number the filing prints inside a sentence stays. With one it must be a
#   paragraph of its own where the filing has lines (a file number at the end of
#   a sentence is text). It never takes a caption,
#   a heading word, a bracketed legend or a signature-block word. A page tag the
#   tagger put inside the run stays ("<page>4</page> LEGAL\67828860\11 ..."), and
#   a number the page already carries as a tag nearer the edge is text ("affiliates
#   52" above "<page>52</page>" in an index of terms). Runs are grouped with their
#   long numbers as printed, so each document ID is its own run.
# - A repeated line glued to the start (top edge) or end (bottom edge) of the
#   outermost remaining line is cut when the filing prints it as its own edge
#   line on that page, it has two words or digits, the agreement never prints it
#   more often than it has edge copies (so it is never part of a clause), and the
#   filing prints it as a paragraph of its own: a sentence that wraps (a <br>, a
#   text line) onto a repeated last line, "(See Annex I to the proxy" + "statement/
#   prospectus on Form S-4 ... filed on July 9, 2025)", is not glued.
# - A glued run at either limit (``MAX_RAW_AFFIX_TOKENS`` words,
#   ``MAX_RAW_JOINED_LINES`` filed lines) is not cut when the word after it has
#   the same shape on most of the pages it is cut from: the furniture goes on past
#   the limit, and a partial cut would leave the rest glued to the text.
# - A legend longer than the edge window goes with its last lines
#   (``_drop_legend_continuations``). A legend of ``MIN_RAW_LEGEND_CHARS`` or more
#   printed only on the pages it concerns counts on ``MIN_RAW_REPEAT_PAGES`` pages
#   at any share. A back-matter footer that opens with a heading word ("Schedule 1
#   Definitions", "EXHIBIT 5.19 ...") counts when it is the last filed line of
#   ``MIN_EXHIBIT_FOOTER_PAGES`` pages (``exhibit_footer``). What a cut leaves goes
#   too when it is the page number alone ("iv" after its document ID) or a
#   misprint of a running title ("Refinery Assel Purchase Agreement"). A line
#   whose tag wraps the page total ("56 | <page>76</page>") stays as filed.
#
# A page is never emptied.

RAW_EDGE_WINDOW = 4
MIN_RAW_REPEAT_PAGES = 5
MIN_RAW_REPEAT_SHARE = 0.5
MAX_RAW_REPEATED_CHARS = 160
MAX_RAW_LEGEND_CHARS = 400
# A shorter legend line, or any bracketed line, is a placeholder unless the body repeats it.
MIN_RAW_LEGEND_CHARS = 40
MAX_RAW_JOINED_LINES = 4
MAX_RAW_AFFIX_TOKENS = 12
RAW_CHAIN_DISTANCE = 2
MAX_FOOTER_ID_CHARS = 60
# Body pages can be numbered in romans throughout ("xlv", "lxviii").
MAX_RAW_ROMAN_PAGE = 89


@dataclass
class RawEdgePage:
    """One page for ``strip_raw_edge_furniture``: its text after the other rules and the page as filed."""

    text: str
    raw: str | None
    is_html: bool
    page_order: int
    zone: str  # "front_matter", "toc", "body", "sig" or "back_matter"


@dataclass(frozen=True)
class _RawLine:
    text: str
    whole: bool  # its own line as printed: HTML lines always, text and <pre> lines set off by blank lines
    joined: bool = False  # in the same paragraph as the line before (a <br> or a wrapped text line)


_RAW_PRE_RE = re.compile(r"(<pre\b[^>]*>.*?</pre>)", re.IGNORECASE | re.DOTALL)
_RAW_BLOCK_RE = re.compile(
    r"<\s*/?\s*(?:p|div|tr|td|th|li|h[1-6]|table|center|pre|blockquote|ul|ol|dl|dt|dd|hr|body|html)\b[^>]*>",
    re.IGNORECASE,
)
_RAW_BR_RE = re.compile(r"<\s*br\b[^>]*>", re.IGNORECASE)
_RAW_TAG_RE = re.compile(r"<[^>]*>")
_INVISIBLE_RE = re.compile("[​-‏⁠﻿­]")


def _clean_raw_line(line: str) -> str:
    return " ".join(_INVISIBLE_RE.sub("", line).replace(" ", " ").split())


def _set_off_lines(lines: list[str], whole: bool = False) -> list[_RawLine]:
    cleaned = [_clean_raw_line(line) for line in lines]
    return [
        _RawLine(
            line,
            whole or ((k == 0 or not cleaned[k - 1]) and (k + 1 == len(cleaned) or not cleaned[k + 1])),
            k > 0 and bool(cleaned[k - 1]),
        )
        for k, line in enumerate(cleaned)
        if line
    ]


def raw_page_lines(raw: str | None, is_html: bool) -> list[_RawLine]:
    """The printed lines of a filed page: HTML block and line-break tags end a line
    (source newlines inside a block do not); in text pages and <pre> blocks each
    line is a line, and SGML markup lines ("<PAGE>") are not printed."""
    if not raw:
        return []
    if not is_html:
        return _set_off_lines(["" if _RAW_TAG_RE.fullmatch(line.strip()) else line for line in raw.split("\n")])
    out: list[_RawLine] = []
    for k, part in enumerate(_RAW_PRE_RE.split(raw)):
        if k % 2:
            out += _set_off_lines(html.unescape(_RAW_TAG_RE.sub("", part)).split("\n"))
            continue
        text = _RAW_BLOCK_RE.sub("\n\n", part.replace("\r", " ").replace("\n", " "))
        text = html.unescape(_RAW_TAG_RE.sub("", _RAW_BR_RE.sub("\n", text)))
        out += _set_off_lines(text.split("\n"), whole=True)
    return out


@dataclass(frozen=True)
class _PageToken:
    shape: str  # the token without its residue, digits collapsed ("-#-", "a-#", "page # of #", "roman")
    value: int
    label: bool  # a label that restarts per exhibit ("A-3", "Appendix A - Page 3")
    bare: bool  # a plain number ("12", "12 of 40"): at the top it can be a footnote marker
    roman: bool
    bottom_only: bool = False  # "Appendix A-1": a caption when it opens a page


_TOKEN_RESIDUE_RE = re.compile(r"(?P<lead>[.|]\s*)?(?P<core>.*?)(?P<trail>\s*[.|])?")
_TOKEN_NUMBER_RE = re.compile(
    r"(?P<page>page\s+)?(?P<n>\d{1,3})(?:\s+of\s+\d{1,3}|\s?/\s?\d{1,3}|\(\d{1,3}\))?", re.IGNORECASE
)
_TOKEN_DASHED_RE = re.compile(r"[-–—]\s*(?P<n>\d{1,3})\s*[-–—]?|(?P<m>\d{1,3})\s*[-–—]")
_TOKEN_LABEL_RE = re.compile(
    r"(?:[-–—]\s?)?(?:(?:App|Sch|Exh?)\.?\s?)?(?P<prefix>[A-Z]{1,3}(?:-\d{1,2})?|App|Sch|Exh?)\.?"
    r"\s?[-–—]\s?(?P<n>\d{1,3})(?:\s?[-–—])?"
    r"|(?P<letter>[A-Z])\s(?P<n2>\d{1,3})"
)
# "Appendix A-1", "Annex B-3": read only at the bottom of a page (at the top it is a caption).
# Also "Annex 1-6" and "Annex A -5-".
_TOKEN_WORD_LABEL_RE = re.compile(
    r"(?:appendix|annex(?:ure)?|exhibit|schedule)\s+(?P<prefix>[A-Z]{1,2}|\d{1,2})\s?[-\u2013\u2014]\s?(?P<n>\d{1,3})"
    r"(?:\s?[-\u2013\u2014])?",
    re.IGNORECASE,
)
_TOKEN_LABEL_PAGE_RE = re.compile(
    r"(?:(?:appendix|annex(?:ure)?|exhibit|schedule|attachment)\s+[A-Z0-9][\w.\-]{0,6}|confidential)"
    r"\s*[-–—|,]?\s*page\s+(?P<n>\d{1,3})(?:\s+of\s+\d{1,3})?",
    re.IGNORECASE,
)


MAX_PAGE_TOKEN_CHARS = 64
MAX_PAGE_TOKEN_RAW_CHARS = 400  # a .txt page number can sit behind a long run of spaces


@lru_cache(maxsize=65536)
def _page_token(line: str) -> _PageToken | None:
    """The page number a whole line prints, with an optional stray "." or "|" beside it."""
    if len(line) > MAX_PAGE_TOKEN_RAW_CHARS:
        return None
    normalized = _normalized(line)
    if len(normalized) > MAX_PAGE_TOKEN_CHARS:
        return None  # the longest form, "Attachment 12.3(a) - Page 123 of 456", is under 50
    match = _TOKEN_RESIDUE_RE.fullmatch(normalized)
    assert match is not None
    core = match.group("core").strip()
    if not core:
        return None
    shape = _line_shape(core).lower()
    number = _TOKEN_NUMBER_RE.fullmatch(core)
    if number is not None:
        return _PageToken(shape, int(number.group("n")), False, number.group("page") is None, False)
    dashed = _TOKEN_DASHED_RE.fullmatch(core)
    if dashed is not None:
        return _PageToken(shape, int(dashed.group("n") or dashed.group("m")), False, False, False)
    label = _TOKEN_LABEL_RE.fullmatch(core) or _TOKEN_LABEL_PAGE_RE.fullmatch(core)
    if label is not None:
        groups = label.groupdict()
        return _PageToken(shape, int(groups.get("n") or groups["n2"]), True, False, False)
    word_label = _TOKEN_WORD_LABEL_RE.fullmatch(core)
    if word_label is not None:
        return _PageToken(shape, int(word_label.group("n")), True, False, False, bottom_only=True)
    roman = _ROMAN_LINE_RE.fullmatch(core)
    if roman is not None:
        value = _roman_value(roman.group("roman"), MAX_RAW_ROMAN_PAGE)
        if value is not None:
            shape = ("-" if roman.group("lead") else "") + "roman" + ("-" if roman.group("trail") else "")
            return _PageToken(shape, value, False, False, True)
    return None


def _unwrapped(line: str) -> str:
    """The line as printed: every <page> tag unwrapped, invisible characters dropped."""
    return _clean_raw_line(_ANY_PAGE_TAG_RE.sub(lambda m: " " + m.group("inner") + " ", line))


def _glued_unwrapped(line: str) -> str:
    """Like ``_unwrapped``, but a tag glued to text stays glued ("<page>41</page>." -> "41.")."""
    return _clean_raw_line(_ANY_PAGE_TAG_RE.sub(lambda m: m.group("inner"), line))


def _edge_keys(lines: list[_RawLine]) -> tuple[dict[str, tuple[frozenset[str], bool]], dict[str, tuple[bool, bool, int]]]:
    """Edge lines of a filed page, alone and run together (a legend printed over
    several lines): casefolded text -> (edges it sits on, printed as its own lines),
    and casefolded text -> (it opens a paragraph, it closes one, the most lines it joins)."""
    keys: dict[str, tuple[frozenset[str], bool]] = {}
    bounds: dict[str, tuple[bool, bool, int]] = {}
    for side, offset, window in (("top", 0, lines[:RAW_EDGE_WINDOW]), ("bottom", max(0, len(lines) - RAW_EDGE_WINDOW), lines[-RAW_EDGE_WINDOW:])):
        for start in range(len(window)):
            for end in range(start + 1, min(len(window), start + MAX_RAW_JOINED_LINES) + 1):
                run = window[start:end]
                key = " ".join(raw_line.text for raw_line in run).casefold()
                sides, whole = keys.get(key, (frozenset(), False))
                keys[key] = (sides | {side}, whole or all(raw_line.whole for raw_line in run))
                after = offset + end
                opens, closes, joins = bounds.get(key, (False, False, 0))
                bounds[key] = (opens or not run[0].joined, closes or after >= len(lines) or not lines[after].joined, max(joins, len(run)))
    return keys, bounds


_RESIDUE_LINE_RE = re.compile(r"[.|]")
# What a tagged page number can carry on its line: a stray "." or "|", the total
# ("/ 107", "of 92"), or the same number printed again.
_TAG_COMPANION_RE = re.compile(r"[.|]|(?:/|of)\s*\d{1,3}|\d{1,3}", re.IGNORECASE)
_RAW_LEGEND_WORD_RE = re.compile(r"confidential|omitted|redacted|initials|privileged", re.IGNORECASE)
# A legend that explains a marker: it opens with the asterisks ("**", "[***] =", "“********” =")
# and says what they stand for.
_MARKER_LEGEND_RE = re.compile(
    r"[\[“\"(]?\s*\*+\s*[\]”\")]?.*\b(?:confidential|denot\w*|indicat\w*|marked|portions?|redacted|information|filed\s+separately)\b",
    re.IGNORECASE,
)
_DOC_ID_TOKEN_RE = re.compile(r"\d{6,}|\d{4}-\d{4}-\d{4}")
_PLAIN_NUMBER_RE = re.compile(r"\d{1,3}")
_TOC_COLUMN_HEADER_RE = re.compile(r"page(?:\s+page)?|section\s+page|page\s+no\.?", re.IGNORECASE)
_RAW_HEADING_START_RE = re.compile(
    r"\(?(?:article|section|clause|part|chapter|rule|regulation|paragraph|item|step|schedule|exhibit|annex|appendix)\b",
    re.IGNORECASE,
)
# A glued run never takes a caption, a bracketed legend or a signature-block word with it.
_AFFIX_EXCLUDED_RE = re.compile(
    r"[\[\]]|/s/|\b(?:exhibit|schedule|annex|appendix|attachment|article|section|by|name|title)\b", re.IGNORECASE
)


def _clean_tag_values(text: str) -> list[int]:
    return [int(m.group("dashed") or m.group("plain")) for m in _CLEAN_PAGE_TAG_RE.finditer(text)]


def _tag_companion(line: str) -> tuple[str, str] | None:
    """(tag, companion) for a line holding one clean page tag and a companion."""
    stripped = line.strip()
    tags = list(_ANY_PAGE_TAG_RE.finditer(stripped))
    if len(tags) != 1:
        return None
    values = _clean_tag_values(tags[0].group(0))
    if not values:
        return None
    companion = _normalized(stripped[: tags[0].start()] + " " + stripped[tags[0].end():])
    if not companion or _TAG_COMPANION_RE.fullmatch(companion) is None:
        return None
    if companion.isdigit() and int(companion) != values[0]:
        return None
    return tags[0].group(0), companion


_PAGE_NUMBER_NEIGHBOR_RE = re.compile(r"(?i)page|pg\.?|p\.|of|[-\u2013\u2014|:,]+")


def _page_number_position(tokens: list[str], i: int) -> bool:
    """Whether token ``i`` sits where a running line prints its page number: at
    either end of the line (a document ID printed past it does not count:
    "... Share Sale Agreement 57 30379804_3"), or next to "Page", "of" or a dash."""
    first, last = 0, len(tokens) - 1
    while first < i and _DOC_ID_TOKEN_RE.search(tokens[first]):
        first += 1
    while last > i and _DOC_ID_TOKEN_RE.search(tokens[last]):
        last -= 1
    if i in (first, last):
        return True
    return any(_PAGE_NUMBER_NEIGHBOR_RE.fullmatch(tokens[j]) for j in (i - 1, i + 1))


def _digit_fit(copies: dict[int, list[str]], orders: list[int]) -> tuple[dict[int, bool], bool] | None:
    """Whether the numbers in a repeated line (one copy per page, as tokens) are furniture.

    Every token with digits must be the same on every copy ("DB1/", "of 87", a
    document ID), or a plain number that rises with the pages on at least
    ``MIN_RAW_REPEAT_SHARE`` of the copies (a page number); any other number
    (a date, an amount, a clause number) makes the line text. Returns, per page,
    whether its copy's page numbers fit, and whether the line holds a document
    ID or a page number; None when the line is text."""
    sample = next(iter(copies.values()))
    fits = {k: True for k in copies}
    marked = False
    for i, token in enumerate(sample):
        if not any(ch.isdigit() for ch in token):
            continue
        values = {run[i] for run in copies.values()}
        if len(values) == 1:
            marked = marked or _DOC_ID_TOKEN_RE.search(token) is not None
            continue
        if not all(_PLAIN_NUMBER_RE.fullmatch(value) for value in values):
            return None
        if not _page_number_position(sample, i):
            return None  # "POLARIS SUB 2 INC." names a different signer on each page
        offsets: dict[int, int] = {}
        for k, run in copies.items():
            offsets[int(run[i]) - orders[k]] = offsets.get(int(run[i]) - orders[k], 0) + 1
        best, count = max(offsets.items(), key=lambda item: item[1])
        if count < MIN_RAW_REPEAT_SHARE * len(copies):
            return None
        marked = True
        for k, run in copies.items():
            fits[k] = fits[k] and int(run[i]) - orders[k] == best
    return fits, marked


def _raw_line_is_eligible(printed: str, legend: bool, footer: bool = False) -> bool:
    """A repeated line that can be furniture (see the notes above). A line that
    opens with a heading word counts only as a footer (``footer``): a heading
    heads its page, while "Schedule 1 Definitions" or "EXHIBIT 5.19 ... PAGE 3"
    repeated at the foot of an exhibit's pages is its running footer."""
    if not printed or "<" in printed or ">" in printed:
        return False
    if len(printed) > (MAX_RAW_LEGEND_CHARS if legend else MAX_RAW_REPEATED_CHARS):
        return False
    excluded = _REPEATED_LINE_EXCLUDED_RES[:-1] if footer else _REPEATED_LINE_EXCLUDED_RES  # the last: captions
    if any(pattern.fullmatch(printed) for pattern in excluded) or "/s/" in printed:
        return False
    if _RAW_HEADING_START_RE.match(printed) and not footer:
        return False  # "ARTICLE 3" heads each page of an exhibit's articles
    if _DIGIT_ENUMERATOR_RE.fullmatch(printed) or not any(ch.isalnum() for ch in printed):
        return False
    return " " in printed or not printed.isalpha()


@lru_cache(maxsize=512)  # about one large agreement's pages
def _raw_page_reading(
    raw: str | None, is_html: bool
) -> tuple[tuple[_RawLine, ...], dict[str, tuple[frozenset[str], bool]], dict[str, tuple[bool, bool, int]]]:
    """``raw_page_lines`` and ``_edge_keys`` of a filed page, cached: the builder
    renders an agreement several times (tag repairs, hard-rule checks) from the
    same pages. The dicts are shared; callers only read them."""
    lines = raw_page_lines(raw, is_html)
    keys, bounds = _edge_keys(lines)
    return tuple(lines), keys, bounds


def strip_raw_edge_furniture(pages: list[RawEdgePage]) -> FurnitureStripResult:
    """Remove page numbers, stray page-number punctuation and repeated header and
    footer lines that the filed page prints at its edges (see the notes above).

    ``pages`` are all of the agreement's pages (front matter, contents, body, sig
    and back matter) in page order, each with its text after the other rules.
    Returns their texts plus a record of each removed line (``page_index`` is the
    position in ``pages``).
    """
    result = FurnitureStripResult(page_texts=[page.text for page in pages])
    if not pages:
        return result
    readings = [_raw_page_reading(page.raw, page.is_html) for page in pages]
    raw_lines = [list(reading[0]) for reading in readings]
    raw_edges = [reading[1] for reading in readings]
    raw_bounds = [reading[2] for reading in readings]
    orders = [page.page_order for page in pages]
    page_lines = [page.text.split("\n") for page in pages]
    headings = _heading_texts([page.text for page in pages if page.zone == "body"])
    zone_sizes = {"body": sum(1 for page in pages if page.zone == "body")}

    def group(k: int) -> str:
        return "body" if pages[k].zone == "body" else "other"

    other_positions = {k: n for n, k in enumerate(k for k in range(len(pages)) if group(k) == "other")}

    def repeated_enough(found: set[int] | dict[int, object]) -> tuple[bool, bool]:
        """(repeated on the body pages, repeated on the sig/back pages).

        On the body pages the share is of all body pages. On the sig/back pages it
        is of the pages from its first copy to its last: an exhibit has its own
        footer."""
        in_body = sum(1 for k in found if group(k) == "body")
        others = sorted(other_positions[k] for k in found if group(k) == "other")
        body = in_body >= MIN_RAW_REPEAT_PAGES and in_body >= MIN_RAW_REPEAT_SHARE * zone_sizes["body"]
        other = len(others) >= MIN_RAW_REPEAT_PAGES and len(others) >= MIN_RAW_REPEAT_SHARE * (others[-1] - others[0] + 1)
        return body, other

    # Page-number tokens the filing prints at each page edge: (shape, value, edge) -> pages.
    raw_tokens: dict[tuple[str, int, str], set[int]] = {}
    for k, lines in enumerate(raw_lines):
        for side, window in (("top", lines[:RAW_EDGE_WINDOW]), ("bottom", lines[-RAW_EDGE_WINDOW:])):
            # A line alone, or a few run together ("Confidential" / "Page 3").
            for start in range(len(window)):
                for end in range(start + 1, min(len(window), start + MAX_RAW_JOINED_LINES) + 1):
                    token = _page_token(" ".join(raw_line.text for raw_line in window[start:end]))
                    if token is not None:
                        raw_tokens.setdefault((token.shape, token.value, side), set()).add(k)
    tagged = {k: _clean_tag_values(text) for k, text in enumerate(result.page_texts)}
    tagged_orders = sorted((orders[k], values[-1]) for k, values in tagged.items() if values)

    def chained(k: int, token: _PageToken, side: str) -> bool:
        """A page one or two places away prints the same shape on the same edge, rising by the page distance."""
        for j in range(max(0, k - RAW_CHAIN_DISTANCE), min(len(pages), k + RAW_CHAIN_DISTANCE + 1)):
            distance = orders[j] - orders[k]
            key = (token.shape, token.value + distance, side)
            if j != k and 0 < abs(distance) <= RAW_CHAIN_DISTANCE and j in raw_tokens.get(key, set()):
                return True
        return False

    def continues_tagged(k: int, value: int) -> bool:
        before = [a for a in tagged_orders if a[0] < orders[k]]
        after = [a for a in tagged_orders if a[0] > orders[k]]
        nearest = ([before[-1]] if before else []) + ([after[0]] if after else [])
        return any(value == anchor_value + orders[k] - anchor_order for anchor_order, anchor_value in nearest)

    def is_page_number(k: int, index: int, side: str, text: str | None = None) -> bool:
        line = (page_lines[k][index] if text is None else text).strip()
        if "<" in line or ">" in line:
            return False
        token = _page_token(line)
        if token is None or _normalized(line).casefold() not in raw_edges[k]:
            return False
        if token.bare and side == "top" and _next_line_opens_unnumbered_heading(page_lines[k], index):
            return False
        if token.bottom_only or (token.label and side == "bottom"):
            # A caption opens its page: a label counts at the bottom only when the
            # filing prints it at the bottom alone and it is not the page's first line.
            first = _countable_indexes(page_lines[k])[:1]
            if side != "bottom" or first == [index] or raw_lines[k][0].text.casefold() == _normalized(line).casefold():
                return False
        if pages[k].zone in ("toc", "front_matter"):
            return chained(k, token, side)  # a lone number on a contents page can be an entry's page
        own = tagged[k]
        if token.value in own and not (token.label or token.roman):
            return True
        if chained(k, token, side):
            return True
        if side != "bottom" or token.roman:
            return False
        if token.label:
            return token.value == 1
        return not own and continues_tagged(k, token.value)

    def page_tag_cleanup(k: int, index: int) -> str | None:
        """The tag alone, for a page tag glued to a stray "." / "|", a total, or its own number again."""
        line = page_lines[k][index]
        found = _tag_companion(line)
        if found is None:
            return None
        tag, companion = found
        printed = (_glued_unwrapped(line).casefold(), _unwrapped(line).casefold(), companion.casefold())
        return tag if any(text in raw_edges[k] for text in printed) else None

    def page_label_cleanup(k: int, index: int) -> str | None:
        """The tag alone, for a tagged page number printed with its label ("Page <page>2</page> of 92")."""
        line = page_lines[k][index].strip()
        tags = _CLEAN_PAGE_TAG_RE.findall(line)
        if len(tags) != 1 or _OTHER_TAG_RE.search(line):
            return None
        printed = _unwrapped(line)
        token = _page_token(printed)
        if token is None or token.value not in tagged[k] or printed.casefold() not in raw_edges[k]:
            return None
        return _CLEAN_PAGE_TAG_RE.search(line).group(0)  # type: ignore[union-attr]

    def is_residue_line(k: int, index: int, dropped: set[int]) -> bool:
        line = page_lines[k][index].strip()
        if _RESIDUE_LINE_RE.fullmatch(line) is None:
            return False
        if not any(key == line or (key.startswith(line) or key.endswith(line)) and _page_token(key) is not None for key in raw_edges[k]):
            return False
        lines = page_lines[k]
        return any(
            i in dropped or _CLEAN_PAGE_TAG_RE.fullmatch(lines[i].strip()) is not None
            for i in (index - 1, index + 1, index - 2, index + 2)
            if 0 <= i < len(lines) and lines[i].strip()
        )

    # Repeated lines the filing prints at page edges: shape -> {page: edges}, and the copies' tokens.
    edge_shapes: dict[str, dict[int, frozenset[str]]] = {}
    edge_copies: dict[str, dict[int, list[str]]] = {}
    for k, edges in enumerate(raw_edges):
        for key, (sides, whole) in edges.items():
            # Longer runs are never furniture text (``_raw_line_is_eligible``).
            if whole and len(key) <= MAX_RAW_LEGEND_CHARS and _page_token(key) is None:
                found = edge_shapes.setdefault(_line_shape(key), {})
                found[k] = found.get(k, frozenset()) | sides
                edge_copies.setdefault(_line_shape(key), {}).setdefault(k, key.split())
    digit_fits: dict[str, tuple[dict[int, bool], bool] | None] = {}

    def running(shape: str) -> tuple[str, bool] | None:
        """(dominant edge, repeated on the body pages) for a repeated edge-line shape."""
        found = edge_shapes.get(shape, {})
        in_body, in_other = repeated_enough(found)
        if len(found) < MIN_RAW_REPEAT_PAGES or not (in_body or in_other):
            return None
        top = sum(1 for sides in found.values() if "top" in sides)
        bottom = sum(1 for sides in found.values() if "bottom" in sides)
        if top == bottom:
            return None
        return ("top" if top > bottom else "bottom"), in_body

    def outermost(k: int, key: str, side: str) -> bool:
        """The filing prints ``key`` as the page's last (bottom) or first (top) lines."""
        lines = raw_lines[k]
        for length in range(1, min(len(lines), MAX_RAW_JOINED_LINES) + 1):
            run = lines[-length:] if side == "bottom" else lines[:length]
            if " ".join(raw_line.text for raw_line in run).casefold() == key:
                return True
        return False

    def exhibit_footer(k: int, key: str, shape: str, side: str) -> bool:
        """A back-matter footer with no page number ("Schedule 1 Definitions"): the
        filing prints it as the last line of at least ``MIN_EXHIBIT_FOOTER_PAGES``
        pages, on ``MIN_EXHIBIT_FOOTER_SHARE`` of the pages from its first copy to
        its last, in a few words that do not end a sentence."""
        if pages[k].zone != "back_matter" or side != "bottom" or not outermost(k, key, side):
            return False
        if len(key.split()) > MAX_EXHIBIT_FOOTER_WORDS or _SENTENCE_END_RE.search(key):
            return False
        copies = [j for j in edge_shapes.get(shape, {}) if pages[j].zone == "back_matter" and outermost(j, key, side)]
        if len(copies) < MIN_EXHIBIT_FOOTER_PAGES:
            return False
        span = other_positions[max(copies)] - other_positions[min(copies)] + 1
        if len(copies) < MIN_EXHIBIT_FOOTER_SHARE * span:
            return False
        # The filed line just above it must change from page to page: when it repeats
        # too, the footer is the end of a longer block ("SIGNATURE PAGE TO" / "AMENDED
        # AND RESTATED" / "REGISTRATION RIGHTS AGREEMENT"), and taking its last line
        # alone would split it.
        above: dict[str, int] = {}
        for j in copies:
            end = _raw_run_end(raw_lines[j], key, "bottom", MAX_RAW_JOINED_LINES)
            if end is not None and end > 0:
                text = raw_lines[j][end - 1].text.casefold()
                above[text] = above.get(text, 0) + 1
        return not above or max(above.values()) * 2 < len(copies)

    def is_furniture_text(k: int, printed: str, side: str) -> bool:
        """A repeated edge line (or run of lines) the filing prints on this page's edge."""
        key = printed.casefold()
        if key not in raw_edges[k] or not raw_edges[k][key][1] or side not in raw_edges[k][key][0]:
            return False
        if _page_token(printed) is not None or key in headings:
            return False
        has_digit = _DIGITS_RE.search(printed) is not None
        legend = _RAW_LEGEND_WORD_RE.search(printed) is not None
        if not _raw_line_is_eligible(printed, legend, footer=side == "bottom" and outermost(k, key, side)):
            return False
        shape = _line_shape(key)
        found = running(shape)
        if found is None and legend and len(printed) >= MIN_RAW_LEGEND_CHARS and _MARKER_LEGEND_RE.match(printed):
            # A legend printed only on the pages it concerns ("** The appearance of a
            # double asterisk denotes confidential information ..."): the same words
            # as their own lines at this edge of enough pages, at any share. It must
            # explain a marker: "[Omitted pursuant to Item 601(a)(5) ...]" or "*Omitted
            # pursuant to Item 601(b)(2) ..." is what an omitted exhibit's page holds.
            copies = sum(1 for edges in raw_edges if key in edges and edges[key][1] and side in edges[key][0])
            found = (side, True) if copies >= MIN_RAW_REPEAT_PAGES else None
        if found is None or found[0] != side:
            return False
        if not has_digit and min(edge_shapes[shape]) == k:
            return False  # the first copy: a cover title, or a table's first header
        marked = False
        if has_digit:
            if shape not in digit_fits:
                digit_fits[shape] = _digit_fit(edge_copies[shape], orders)
            fit = digit_fits[shape]
            if fit is None or not fit[0].get(k, False):
                return False
            marked = fit[1]
        if (pages[k].zone not in ("body", "back_matter") or not found[1]) and not (marked or legend or exhibit_footer(k, key, shape, side)):
            return False
        placeholder = printed.startswith("[") or (legend and len(printed) < MIN_RAW_LEGEND_CHARS)
        if placeholder and not marked and not (found[1] and (len(printed) >= MIN_RAW_LEGEND_CHARS or not printed.startswith("["))):
            # "[REDACTED]", "[Exhibit has been omitted ...]": what a page or schedule holds.
            # A short legend repeated on the body pages, unbracketed, is a running line
            # ("Confidential and Proprietary").
            return False
        return True

    # A law-firm document ID ("|US-DOCS\\140904150.6||") printed as its own edge line on two or more pages.
    doc_id_pages: dict[str, set[int]] = {}
    for k, edges in enumerate(raw_edges):
        for key, (_, whole) in edges.items():
            if whole and len(key) <= MAX_FOOTER_ID_CHARS and _DOC_ID_TOKEN_RE.search(key):
                doc_id_pages.setdefault(key, set()).add(k)

    def is_doc_id(k: int, index: int, side: str) -> bool:
        line = page_lines[k][index].strip()
        if "<" in line or ">" in line:
            return False
        key = _normalized(line).casefold()
        if not _is_doc_id_line(_normalized(line)):
            return False
        return len(doc_id_pages.get(key, ())) >= 2 and k in doc_id_pages[key] and side in raw_edges[k][key][0]

    def is_toc_header(k: int, index: int, side: str, dropped: dict[int, tuple[str, str | None]]) -> bool:
        """A "Page" column header ("Page" / "No." on two lines), or the contents
        title repeated on a later contents page."""
        if pages[k].zone != "toc" or side != "top":
            return False
        line = page_lines[k][index].strip()
        key = _normalized(line).casefold()
        if key not in raw_edges[k] or "top" not in raw_edges[k][key][0]:
            return False
        if _TOC_COLUMN_HEADER_RE.fullmatch(key):
            return True
        if key in ("no.", "no") and any(kind == "raw_toc_header" for kind, _ in dropped.values()):
            return True
        first_toc = min(j for j, page in enumerate(pages) if page.zone == "toc")
        return k != first_toc and _CONTENTS_HEADING_RE.fullmatch(key) is not None

    def repeated_line_keep(k: int, index: int, side: str) -> tuple[bool, str | None]:
        line = page_lines[k][index].strip()
        if _OTHER_TAG_RE.search(line) or not is_furniture_text(k, _unwrapped(line), side):
            return False, None
        # Page tags on the line stay, a clean one or one that holds a page number
        # with a stray mark ("<page>19 |</page>"); a document ID the tagger wrapped goes.
        kept = [m.group(0) for m in _ANY_PAGE_TAG_RE.finditer(line) if _page_token(m.group("inner")) is not None]
        if kept:
            # "56 | <page>76</page>": the tagger wrapped the page total, and the
            # printed page number is the number before it; leave the line as filed.
            # ("<page>15</page> | 100" and "... <page>2</page> of 20" print the total after.)
            values = set(_clean_tag_values(line))
            before = line[: _ANY_PAGE_TAG_RE.search(line).start()]
            if any(int(token) not in values for token in before.split() if _PLAIN_NUMBER_RE.fullmatch(token)):
                return False, None
        return True, " ".join(kept) or None

    # Whole lines, walked in from each edge.
    drops: list[dict[int, tuple[str, str | None]]] = []
    walked: list[dict[str, list[int]]] = []  # per page and edge, the lines dropped in walk order
    for k, lines in enumerate(page_lines):
        drop: dict[int, tuple[str, str | None]] = {}
        walked.append({"top": [], "bottom": []})
        countable = _countable_indexes(lines)
        inside = _lines_inside_headings(lines)
        for side in ("top", "bottom"):
            ordered = countable if side == "top" else countable[::-1]
            for index in ordered[:EDGE_LINE_WINDOW]:
                if index in inside:
                    break
                if index in drop:
                    continue
                walked[k][side].append(index)
                if is_page_number(k, index, side):
                    drop[index] = ("raw_page_number", None)
                elif is_residue_line(k, index, set(drop)):
                    drop[index] = ("raw_page_residue", None)
                elif (tag := page_tag_cleanup(k, index) or page_label_cleanup(k, index)) is not None:
                    drop[index] = ("raw_page_residue", tag)
                elif is_doc_id(k, index, side):
                    drop[index] = ("raw_doc_id", None)
                elif is_toc_header(k, index, side, drop):
                    drop[index] = ("raw_toc_header", None)
                else:
                    repeated, keep = repeated_line_keep(k, index, side)
                    if not repeated:
                        walked[k][side].pop()
                        break
                    drop[index] = ("raw_repeated_line", keep)
        drops.append(drop)

    _drop_legend_continuations(pages, raw_lines, page_lines, drops, walked, headings, repeated_enough)
    for k, lines in enumerate(page_lines):
        countable = _countable_indexes(lines)
        if drops[k] and all(index in drops[k] and drops[k][index][1] is None for index in countable):
            drops[k] = {}  # never empty a page

    cuts, capped = _raw_affix_cuts(pages, raw_lines, page_lines, drops, repeated_enough)

    edge_key_pages: dict[str, int] = {}  # filled on first use

    def count_edge_keys() -> dict[str, int]:
        if not edge_key_pages:
            for edges in raw_edges:
                for edge_key in edges:
                    edge_key_pages[edge_key] = edge_key_pages.get(edge_key, 0) + 1
        return edge_key_pages

    def near_running_line(k: int, text: str) -> bool:
        """What a cut leaves is a whole filed edge line of this page that differs by a
        letter or two from a line the filing prints at the edge of
        ``MIN_RAW_REPEAT_PAGES`` or more pages: its running title, misprinted."""
        key = _clean_raw_line(text).casefold()
        words = key.split()
        if not 2 <= len(words) <= MAX_EXHIBIT_FOOTER_WORDS or key not in raw_edges[k] or _SENTENCE_END_RE.search(key):
            return False
        compact = key.replace(" ", "")
        for other, count in count_edge_keys().items():
            if count < MIN_RAW_REPEAT_PAGES or other == key:
                continue
            other_compact = other.replace(" ", "")
            if abs(len(other_compact) - len(compact)) > 2 or abs(len(other.split()) - len(words)) > 1:
                continue
            if difflib.SequenceMatcher(None, compact, other_compact).ratio() >= MIN_TITLE_SIMILARITY:
                return True
        return False

    def continues_into_furniture(k: int, key: str, rest: str, side: str) -> bool:
        """The text left after a glued cut starts (top) or ends (bottom) with the
        next filed line inward, that line holds a legend word, and the filing prints
        that very line at the edge of ``MIN_RAW_REPEAT_PAGES`` or more pages: a
        legend the page's text runs together, which a cut would split."""
        raw = raw_lines[k]
        end = _raw_run_end(raw, key, side, RAW_EDGE_WINDOW)
        if end is None or not 0 <= (end if side == "top" else end - 1) < len(raw):
            return False
        neighbor = raw[end] if side == "top" else raw[end - 1]
        words = neighbor.text.casefold().split()
        rest_words = _clean_raw_line(_unwrapped(rest)).casefold().split()
        span = min(3, len(words), len(rest_words))
        if not span or (rest_words[:span] != words[:span] if side == "top" else rest_words[-span:] != words[-span:]):
            return False
        if not _RAW_LEGEND_WORD_RE.search(neighbor.text):
            return False  # a running header's next line ("Page: 10" / "Debtors: ..."): the cut leaves it as main does
        return count_edge_keys().get(neighbor.text.casefold(), 0) >= MIN_RAW_REPEAT_PAGES
    joined_text: list[str] = []

    def joined() -> str:
        """The agreement's printed text, casefolded (built on first use)."""
        if not joined_text:
            joined_text.append("\n".join(_unwrapped(line) for lines in page_lines for line in lines).casefold())
        return joined_text[0]

    for k, lines in enumerate(page_lines):
        remaining = [i for i in _countable_indexes(lines) if i not in drops[k]]
        if not remaining:
            continue
        inside = _lines_inside_headings(lines)
        for side, index in (("top", remaining[0]), ("bottom", remaining[-1])):
            if index in inside or (k, index, side) in cuts:
                continue
            line = cuts.get((k, index, "top" if side == "bottom" else "bottom"), (lines[index], ""))[0]
            if "<" in line or ">" in line:
                continue
            printed = _clean_raw_line(line)
            folded = printed.casefold()
            if len(folded) != len(printed):
                continue
            # Repeated lines glued to the outermost remaining line, longest first.
            for key in sorted(raw_edges[k], key=len, reverse=True):
                if len(key) < MIN_FUSED_CHARS or len(key) >= len(folded):
                    continue
                if not (folded.startswith(key) if side == "top" else folded.endswith(key)):
                    continue
                if not (any(ch.isdigit() for ch in key) or len(key.split()) >= 2):
                    continue
                if side == "top" and folded.startswith(key + " "):
                    cut, rest = printed[: len(key)], printed[len(key):].strip()
                elif side == "bottom" and folded.endswith(" " + key):
                    cut, rest = printed[len(printed) - len(key):], printed[: len(printed) - len(key)].strip()
                else:
                    continue
                if not raw_bounds[k][key][0 if side == "bottom" else 1]:
                    continue  # the filing runs it on from (or into) the text: a wrapped sentence, not glued
                if continues_into_furniture(k, key, rest, side):
                    continue  # the rest of the line is more of the same legend: leave it whole
                if not (_has_words(rest) or _page_token(rest) is not None) or not is_furniture_text(k, cut, side):
                    continue
                if joined().count(key) > len(edge_shapes.get(_line_shape(key), {})):
                    continue  # the agreement prints it elsewhere: it can end a clause
                cuts[(k, index, side)] = (rest, cut)
                if raw_bounds[k][key][2] >= MAX_RAW_JOINED_LINES:
                    capped.add((k, index, side))
                break

    cuts = _drop_continued_cuts(cuts, capped)
    for k, lines in enumerate(page_lines):
        drop = drops[k]
        page_cuts = {key: value for key, value in cuts.items() if key[0] == k}
        if not drop and not page_cuts:
            continue
        kept: list[str] = []
        left = [i for i in _countable_indexes(lines) if i not in drop]  # never empty the page
        for index, line in enumerate(lines):
            if index in drop:
                kind, keep = drop[index]
                result.removed.append(RemovedFurnitureLine(page_index=k, line=line.strip(), kind=kind))
                if keep is not None:
                    kept.append(keep)
                continue
            for side in ("top", "bottom"):
                if (k, index, side) in page_cuts:
                    cut = page_cuts[(k, index, side)][1]
                    shorter = _cut_affix(line, cut, side)
                    if shorter != line:
                        result.removed.append(RemovedFurnitureLine(page_index=k, line=cut, kind="raw_fused_line"))
                        # What a cut leaves can be the page number alone ("iv" after its document ID).
                        if len(left) > 1 and is_page_number(k, index, side, shorter):
                            result.removed.append(RemovedFurnitureLine(page_index=k, line=shorter.strip(), kind="raw_page_number"))
                            shorter = ""
                        elif len(left) > 1 and near_running_line(k, _ANY_PAGE_TAG_RE.sub(" ", shorter)):
                            # Or a running title the filing misprints on this page ("Refinery Assel ...").
                            title = _clean_raw_line(_ANY_PAGE_TAG_RE.sub(" ", shorter))
                            result.removed.append(RemovedFurnitureLine(page_index=k, line=title, kind="raw_repeated_line"))
                            shorter = " ".join(m.group(0) for m in _ANY_PAGE_TAG_RE.finditer(shorter))
                    line = shorter
            if line.strip():
                kept.append(line)
        result.page_texts[k] = "\n".join(kept)
    return result


MAX_LEGEND_CONTINUATION_LINES = 4
MIN_EXHIBIT_FOOTER_PAGES = 10
MIN_EXHIBIT_FOOTER_SHARE = 0.8
MAX_EXHIBIT_FOOTER_WORDS = 8
MIN_TITLE_SIMILARITY = 0.9
_SENTENCE_END_RE = re.compile(r"[.;:!?][\s\"'\u201d\u2019)\]]*$")


def _raw_run_end(lines: list[_RawLine], key: str, side: str, limit: int) -> int | None:
    """Where the filed lines that print ``key`` (casefolded, possibly several lines
    run together) end, counted from this edge: the index of the first raw line past
    the run (top), or of the run's first line (bottom). Only the first ``limit``
    lines from the edge are searched."""
    ordered = lines[:limit] if side == "top" else lines[::-1][:limit]
    for start in range(len(ordered)):
        for end in range(start + 1, min(len(ordered), start + MAX_RAW_JOINED_LINES) + 1):
            run = ordered[start:end] if side == "top" else ordered[start:end][::-1]
            if " ".join(raw_line.text for raw_line in run).casefold() == key:
                return end if side == "top" else len(lines) - end
    return None


def _drop_legend_continuations(
    pages: list[RawEdgePage],
    raw_lines: list[list[_RawLine]],
    page_lines: list[list[str]],
    drops: list[dict[int, tuple[str, str | None]]],
    walked: list[dict[str, list[int]]],
    headings: set[str],
    repeated_enough: Callable[[set[int]], tuple[bool, bool]],
) -> None:
    """Carry a removed legend's last lines with it.

    The edge window sees ``RAW_EDGE_WINDOW`` filed lines, so a longer legend
    ("... TREATMENT HAS BEEN REQUESTED WITH RESPECT TO THIS OMITTED" / "INFORMATION.")
    loses its first lines and keeps its tail. The next line inward is taken too
    when the filing prints it right after the removed lines, the lines from the
    page edge through it hold a legend word, and the same line follows the removed
    lines on ``MIN_RAW_REPEAT_PAGES`` or more pages, at the share a repeated line
    needs, and the legend's sentence runs on into it (the line before the junction
    does not end a sentence). A one-word line ("AMENDED.") counts. The agreement's
    headings and tagged lines stay."""
    reach = RAW_EDGE_WINDOW + MAX_LEGEND_CONTINUATION_LINES * MAX_RAW_JOINED_LINES
    taken = {k: {"top": 0, "bottom": 0} for k in range(len(pages))}
    while True:
        found: dict[tuple[str, str], dict[int, tuple[int, int]]] = {}
        for k, lines in enumerate(page_lines):
            countable = _countable_indexes(lines)
            inside = _lines_inside_headings(lines)
            for side in ("top", "bottom"):
                run = walked[k][side]
                if not run or taken[k][side] >= MAX_LEGEND_CONTINUATION_LINES:
                    continue
                if drops[k][run[-1]][0] not in ("raw_repeated_line", "raw_legend_continuation"):
                    continue
                ordered = countable if side == "top" else countable[::-1]
                position = ordered.index(run[-1]) + 1
                if position >= len(ordered):
                    continue
                index = ordered[position]
                line = lines[index].strip()
                if index in drops[k] or index in inside or "<" in line or ">" in line:
                    continue
                printed = _clean_raw_line(line)
                key = printed.casefold()
                # (A heading word can open the line: the legend's sentence runs on into it,
                # "... IN ACCORDANCE WITH" / "RULE 24B-2 UNDER ...".)
                if not printed or len(printed) > MAX_RAW_LEGEND_CHARS or key in headings:
                    continue
                raw = raw_lines[k]
                last = _clean_raw_line(_unwrapped(lines[run[-1]])).casefold()
                end = _raw_run_end(raw, last, side, reach)
                if end is None:
                    continue
                for length in range(1, MAX_RAW_JOINED_LINES + 1):
                    piece = raw[end : end + length] if side == "top" else raw[max(0, end - length) : end]
                    if len(piece) < length:
                        break
                    if " ".join(raw_line.text for raw_line in piece).casefold() != key:
                        continue
                    # The legend's sentence runs on across the junction: the upper line ends
                    # in a word or a comma ("... THIS OMITTED" / "INFORMATION."), and both
                    # lines are set alike. "[See attached]" or "[***]" above a legend is what
                    # the page holds.
                    before = raw[end - 1].text if side == "top" else piece[-1].text
                    after = piece[0].text if side == "top" else raw[end].text
                    if _SENTENCE_END_RE.search(before) or not _RUNS_ON_RE.search(before):
                        break
                    if not after[:1].islower() and _all_capitals(before) != _all_capitals(after):
                        break
                    if _OPENS_ITEM_RE.match(after):
                        break  # "**** Confidential ...", "[***] ...": a legend's first line, or a new item
                    edge_part = raw[: end + length] if side == "top" else raw[end - length :]
                    if _RAW_LEGEND_WORD_RE.search(" ".join(raw_line.text for raw_line in edge_part)):
                        found.setdefault((side, _line_shape(key)), {})[k] = (index, length)
                    break
        accepted = False
        for (side, _), copies in found.items():
            if len(copies) < MIN_RAW_REPEAT_PAGES or not any(repeated_enough(set(copies))):
                continue
            for k, (index, _) in copies.items():
                drops[k][index] = ("raw_legend_continuation", None)
                walked[k][side].append(index)
                taken[k][side] += 1
                accepted = True
        if not accepted:
            return


def _drop_continued_cuts(
    cuts: dict[tuple[int, int, str], tuple[str, str]], capped: set[tuple[int, int, str]]
) -> dict[tuple[int, int, str], tuple[str, str]]:
    """Leave glued runs that reach the longest run the rules can see while the
    furniture goes on past it.

    A header or footer longer than ``MAX_RAW_AFFIX_TOKENS`` words (or
    ``MAX_RAW_JOINED_LINES`` filed lines) would be cut partway, and the rest of it
    would stay glued to the text ("February 2025 29 (c) From the time ...").
    ``capped`` are the cuts at that limit. The text a cut leaves behind differs
    from page to page; when the word right after a capped cut has the same shape
    on most of the pages that share the cut, the furniture continues, and none of
    those cuts is made: the whole run stays as filed. Shorter cuts are complete
    runs and stay."""
    groups: dict[tuple[str, str], list[tuple[tuple[int, int, str], str]]] = {}
    for key, (rest, cut) in cuts.items():
        if key not in capped:
            continue
        words = _unwrapped(rest).split()
        if not words:
            continue
        nxt = words[0] if key[2] == "top" else words[-1]
        groups.setdefault((key[2], _line_shape(cut.casefold())), []).append((key, _line_shape(nxt.casefold())))
    dropped: set[tuple[int, int, str]] = set()
    for copies in groups.values():
        if len(copies) < 3:
            continue
        counts: dict[str, int] = {}
        for _, shape in copies:
            counts[shape] = counts.get(shape, 0) + 1
        if max(counts.values()) >= max(3, MIN_RAW_REPEAT_SHARE * len(copies)):
            dropped.update(key for key, _ in copies)
    return {key: value for key, value in cuts.items() if key not in dropped}


def _cut_affix(line: str, cut: str, side: str) -> str:
    """The line without ``cut`` at its start (top) or end (bottom), whitespace-insensitive."""
    pattern = r"\s+".join(rf"(?:<page>\s*{re.escape(token)}\s*</page>|{re.escape(token)})" for token in cut.split())
    if side == "top":
        match = re.match(rf"\s*{pattern}(?:\s+|$)", line, flags=re.IGNORECASE)
        if match is None:
            return line
        rest = line[match.end():]
    else:
        # Search only the line's tail: a page filed as one run can be a very long line.
        start = max(0, len(line) - (len(cut) + 40 * len(cut.split()) + 64))
        match = re.search(rf"(?:^|\s+){pattern}\s*$", line[start:], flags=re.IGNORECASE)
        if match is None or (match.start() == 0 and start > 0 and not line[start].isspace()):
            return line
        rest = line[: start + match.start()]
    # A page tag the tagger put inside the run stays ("<page>4</page> LEGAL\67828860\11 ...").
    tags = " ".join(tag.group(0) for tag in _ANY_PAGE_TAG_RE.finditer(match.group(0)))
    if not tags:
        return rest
    return f"{tags} {rest}" if side == "top" else f"{rest} {tags}"


MAX_RAW_LINE_CHARS = 400


def _opens_own_paragraph(lines: list[_RawLine], length: int, side: str) -> bool:
    """The run of ``length`` words at this end of the filed page is a paragraph of
    its own: it ends (top) or starts (bottom) where a filed line does, and the text
    beyond it does not run on from it. A page filed as one long run has no lines to
    go by."""
    if len(lines) <= 1:
        return True
    ordered = lines if side == "top" else lines[::-1]
    count = 0
    for n, raw_line in enumerate(ordered):
        if len(raw_line.text) > MAX_RAW_LINE_CHARS:
            return True
        count += len(raw_line.text.split())
        if count == length:
            if n + 1 == len(ordered):
                return True
            inner = ordered[n + 1]
            return not (inner.joined if side == "top" else ordered[n].joined)
        if count > length:
            return False
    return True


_WORD_RE = re.compile(r"[^\W\d_]{2,}")
_OPENS_ITEM_RE = re.compile(r"\s*(?:[\[\u201c\"(]\s*)?\*|\s*\[")
_RUNS_ON_RE = re.compile(r"[^\W_]\s*$|,\s*$")


def _all_capitals(text: str) -> bool:
    letters = [ch for ch in text if ch.isalpha()]
    return bool(letters) and all(ch.isupper() for ch in letters)


def _has_words(text: str) -> bool:
    """Whether a cut leaves text: a word of two or more letters ("A05230281/1.15/02" has none)."""
    return _WORD_RE.search(text) is not None


def _end_tokens(lines: list[_RawLine], limit: int) -> tuple[list[str], list[str], int]:
    """The first and last ``limit`` + 1 words of a filed page, and its word count
    (capped at ``limit`` + 1: no run is longer than ``limit``)."""
    head: list[str] = []
    for raw_line in lines:
        head += raw_line.text.split()
        if len(head) > limit:
            break
    if len(head) <= limit:
        return head, head, len(head)
    tail: list[str] = []
    for raw_line in reversed(lines):
        tail = raw_line.text.split() + tail
        if len(tail) > limit:
            break
    return head[: limit + 1], tail[-(limit + 1):], limit + 1


_SHORT_NUMBER_RE = re.compile(r"(?<!\d)\d{1,3}(?!\d)")


@lru_cache(maxsize=1 << 16)
def _run_token_shape(token: str) -> str:
    """A glued run's token with its page-number-sized numbers collapsed. Longer
    numbers stay as printed, so each document ID ("SK 39196 0005 12134921 v17",
    "SK 39196 0005 12138149 v5") makes its own run instead of failing as one."""
    return _SHORT_NUMBER_RE.sub("#", token)


def _raw_affix_cuts(
    pages: list[RawEdgePage],
    raw_lines: list[list[_RawLine]],
    page_lines: list[list[str]],
    drops: list[dict[int, tuple[str, str | None]]],
    repeated_enough: Callable[[set[int]], tuple[bool, bool]],
) -> tuple[dict[tuple[int, int, str], tuple[str, str]], set[tuple[int, int, str]]]:
    """Headers and footers glued to the text of a page filed as one run (see the notes above).

    Returns (page, line, edge) -> (the line without the run, the run), and the
    cuts whose run is ``MAX_RAW_AFFIX_TOKENS`` words long."""
    raw_ends = [_end_tokens(lines, MAX_RAW_AFFIX_TOKENS) for lines in raw_lines]
    orders = [page.page_order for page in pages]
    cuts: dict[tuple[int, int, str], tuple[str, str]] = {}
    capped: set[tuple[int, int, str]] = set()
    for side in ("top", "bottom"):
        # (length, shape) -> {page: the run's tokens}
        runs: dict[tuple[int, str], dict[int, list[str]]] = {}
        for k, (head, tail, count) in enumerate(raw_ends):
            tokens = head if side == "top" else tail
            # A run's shape is its tokens' shapes joined (digits never span a space).
            shapes = [_run_token_shape(token) for token in (tokens if side == "top" else tokens[::-1])[:MAX_RAW_AFFIX_TOKENS]]
            for length in range(1, min(MAX_RAW_AFFIX_TOKENS, count - 1) + 1):
                run = tokens[:length] if side == "top" else tokens[-length:]
                shape = " ".join(shapes[:length] if side == "top" else shapes[:length][::-1])
                runs.setdefault((length, shape), {})[k] = run

        def qualifies(copies: dict[int, list[str]]) -> dict[int, bool] | None:
            """Per page, whether its copy's page numbers fit; None when the run is not furniture."""
            if not any(repeated_enough(set(copies))) or len(copies) < MIN_RAW_REPEAT_PAGES:
                return None
            if _AFFIX_EXCLUDED_RE.search(" ".join(next(iter(copies.values())))):
                return None
            fit = _digit_fit(copies, orders)
            return fit[0] if fit is not None and fit[1] else None

        qualified: dict[int, list[list[str]]] = {}
        for (length, _), copies in sorted(runs.items(), key=lambda item: -item[0][0]):
            fits = qualifies(copies)
            if fits is None:
                continue
            for k, run in copies.items():
                if fits[k]:
                    qualified.setdefault(k, []).append(run)
        for k, candidates in qualified.items():
            lines = page_lines[k]
            remaining = [i for i in _countable_indexes(lines) if i not in drops[k]]
            if not remaining:
                continue
            index = remaining[0] if side == "top" else remaining[-1]
            tokens = lines[index].split()
            # Page numbers the page already carries as a tag between this line and its
            # edge: a run that prints one of them again is text ("affiliates 52" in an
            # index of terms, above "<page>52</page>").
            outer = range(index) if side == "top" else range(index + 1, len(lines))
            elsewhere = {value for i in outer for value in _clean_tag_values(lines[i])}
            for run in candidates:  # longest first
                if any(_PLAIN_NUMBER_RE.fullmatch(token) and int(token) in elsewhere for token in run):
                    continue
                if not any(_DOC_ID_TOKEN_RE.search(token) for token in run):
                    # A page number glued to the text: the filing prints it as its own line.
                    edge_line = raw_lines[k][0] if side == "top" else raw_lines[k][-1]
                    if len(raw_lines[k]) > 1 and edge_line.text != " ".join(run):
                        continue
                elif not _opens_own_paragraph(raw_lines[k], len(run), side):
                    continue
                edge = tokens[: len(run)] if side == "top" else tokens[-len(run):]
                # A token may be a whole page tag the tagger put inside the run; it stays.
                if len(tokens) <= len(run) or any(("<" in t or ">" in t) and _ANY_PAGE_TAG_RE.fullmatch(t) is None for t in edge):
                    continue
                if [_clean_raw_line(_ANY_PAGE_TAG_RE.sub(lambda m: m.group("inner"), t)) for t in edge] != run:
                    continue
                rest = " ".join(tokens[len(run):] if side == "top" else tokens[: -len(run)])
                if not _has_words(_ANY_PAGE_TAG_RE.sub(" ", rest)) and _page_token(rest) is None:
                    break  # nothing but furniture would stay: leave the line as filed
                cuts[(k, index, side)] = (rest, " ".join(run))
                if len(run) == MAX_RAW_AFFIX_TOKENS:
                    capped.add((k, index, side))
                break
    return cuts, capped
