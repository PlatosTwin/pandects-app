# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false
"""Restore word spaces the page text lost, and drop spaces it added, using the filed page.

The HTML page text loses or gains white space at a few kinds of markup:

* A ``<br>``, a block boundary (``</p><p>``, a table cell) or a span that holds
  only white space prints as a space, but the extractor sometimes drops it, so
  words are glued: "Sprint Nextel Corporation2001 Edmund Drive",
  "Tax</font><font> </font>Return" read "TaxReturn".
* An inline-tag boundary inside a hyphenated word prints nothing, but the
  extractor sometimes puts a space there: "<b>nti</b><b>-Corruption</b>"
  reads "Anti -Corruption".

The builder fixes both, one junction at a time, and only where the filed page
proves it:

* The junction is found in the raw page by the letters and digits around it
  (a key of up to 40 on each side that occurs exactly once on the page), and
  the printed characters between those letters must be the same in the page
  text and in the raw page.
* A space is inserted only where the raw page has white space, a ``<br>``, or
  a block or table-cell boundary at that exact spot. Boundaries whose layout
  is unknown (absolutely positioned or inline-block boxes, floats, images,
  hidden elements) are not evidence. No space is put next to a hyphen, a dash
  or a slash, before closing punctuation, or after opening punctuation; nor
  inside a heading number ("ARTICLE X<br>I"), a <page> marker, or a word the
  agreement prints unbroken elsewhere (a hand wrap, "Aktiengese<br>llschaft").
* A space next to a hyphen is removed only where the raw page has nothing at
  all at that spot (no white space and no boundary of any kind), and only when
  the hyphen is joined to a word on its other side, so dash leaders stay.

Private-use glyphs from Symbol and Wingdings bullets ("\\uf0b7") are mapped to
"•"; a private-use logo glyph that is pure decoration is dropped.

The page text is left unchanged when the raw page is missing, is not HTML, or
the junction cannot be aligned.
"""
from __future__ import annotations

import html
import re
from dataclasses import dataclass, field
from collections.abc import Mapping

from bs4 import BeautifulSoup
from bs4.element import Comment, Declaration, Doctype, NavigableString, ProcessingInstruction, Tag

from etl.domain.heading_titles import (
    article_title_number,
    normalize_glued_article_title,
    normalize_glued_section_title,
)
from etl.domain.raw_block_joins import INLINE_TAG_NAMES, PRE_TAG_NAMES

# --- private-use glyphs ---------------------------------------------------------------------------------

# Symbol / Wingdings code points that HTML filings carry in the private-use area
# (U+F000 + the font's code). Bullets and list markers print as a bullet.
_PUA_BULLETS = {
    "\uf0b7": "\u2022",  # Symbol bullet
    "\uf0a7": "\u2022",  # Wingdings small square bullet
    "\uf06c": "\u2022",  # Wingdings filled circle (also the "[\u25cf]" placeholder)
    "\uf06e": "\u2022",  # Wingdings filled square
    "\uf0a8": "\u2610",  # Wingdings box (a checklist box)
    "\uf0a1": "\u2022",  # Wingdings circle bullet
    "\uf0d8": "\u2022",  # Wingdings arrowhead bullet
    "\uf0fc": "\u2713",  # Wingdings check mark
    "\uf076": "\u2022",  # Wingdings diamond bullet
    "\uf075": "\u2022",  # Wingdings diamond bullet
    "\uf098": "\u2022",  # Wingdings ornament (a "[ \u25cf ]" placeholder)
    "\uf0a6": "\u2022",
    "\uf09f": "\u2022",
    "\uf02d": "-",  # Symbol minus, used as a dash list marker
    "\uf03a": ":",  # Symbol colon
}
# Decoration with no text value (a law firm's logo glyph in a running footer).
_PUA_DROP = frozenset({"\uf8e9"})
_PUA_RE = re.compile("[" + "".join(sorted(set(_PUA_BULLETS) | _PUA_DROP)) + "]")
_PUA_DROP_RE = re.compile("([ \\t]?)[" + "".join(sorted(_PUA_DROP)) + "]([ \\t]?)")


def map_private_use_glyphs(text: str) -> str:
    """Map Symbol/Wingdings private-use bullets to "•" and drop decoration glyphs."""
    if not _PUA_RE.search(text):
        return text
    if any(ch in text for ch in _PUA_DROP):
        # A dropped glyph between two spaces leaves one of them.
        text = _PUA_DROP_RE.sub(lambda m: " " if m.group(1) and m.group(2) else "", text)
    return _PUA_RE.sub(lambda m: _PUA_BULLETS.get(m.group(0), ""), text)


# --- the filed page as a character stream --------------------------------------------------------------

NONE, SOFT, HARD = 0, 1, 2

_ROOT_TAGS = frozenset({"html", "body", "[document]"})
_SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "head", "title"})
_HARD_BLOCK_TAGS = frozenset(
    {
        "p", "div", "td", "th", "tr", "li", "ul", "ol", "dl", "dt", "dd", "table", "tbody", "thead",
        "tfoot", "caption", "h1", "h2", "h3", "h4", "h5", "h6", "center", "blockquote", "pre",
        "section", "article", "header", "footer", "address", "hr", "form", "xmp", "listing",
    }
)
_PRE_STYLE_RE = re.compile(r"white-space\s*:\s*pre", re.IGNORECASE)
# Layout the reader cannot be sure of: boxes placed side by side or anywhere.
_SOFT_STYLE_RE = re.compile(
    r"(?:^|;)\s*(?:display\s*:\s*inline-block|position\s*:\s*(?:absolute|fixed)|float\s*:\s*(?:left|right))",
    re.IGNORECASE,
)
_BLOCK_STYLE_RE = re.compile(r"(?:^|;)\s*display\s*:\s*(?:block|flex|grid|table|list-item)", re.IGNORECASE)
_HIDDEN_STYLE_RE = re.compile(
    r"(?:^|;)\s*(?:display\s*:\s*none|visibility\s*:\s*(?:hidden|collapse))", re.IGNORECASE
)
_DROPPED_CHARS_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F​‌‍‎‏‪-‮⁠﻿]")
_WS_RE = re.compile(r"\s+")
# An end tag of a non-inline element: html.parser drops one that has no open
# element, so a marker keeps the place (as an unsure boundary).
_NON_INLINE_END_TAG_RE = re.compile(
    r"</(?!(?:" + "|".join(sorted(INLINE_TAG_NAMES, key=len, reverse=True)) + r")\s*>)[A-Za-z][A-Za-z0-9:-]*\s*>",
    re.IGNORECASE,
)


@dataclass
class RawSpaceStream:
    """The visible text of a filed page: white space collapsed to one space, and
    ``bounds[i]`` the strongest element boundary just before ``text[i]``."""

    text: str
    bounds: dict[int, int]
    alnum: str
    alnum_pos: list[int]
    # positions with a <br> boundary, and positions with any other boundary
    brs: frozenset[int] = frozenset()
    others: frozenset[int] = frozenset()
    # for each character, the table cell (<td>/<th>) it sits in, innermost; -1 outside cells
    cells: tuple[int, ...] = ()


def _style(tag: Tag) -> str:
    value = tag.get("style")
    if isinstance(value, list):
        return " ".join(value)
    return str(value or "")


def _is_hidden(tag: Tag) -> bool:
    if tag.has_attr("hidden"):
        return True
    aria = tag.get("aria-hidden")
    if str(" ".join(aria) if isinstance(aria, list) else aria or "").strip().lower() == "true":
        return True
    return bool(_HIDDEN_STYLE_RE.search(_style(tag)))


class _Builder:
    def __init__(self) -> None:
        self.chars: list[str] = []
        self.bounds: dict[int, int] = {}
        self.brs: set[int] = set()
        self.others: set[int] = set()
        self.cells: list[int] = []
        self.cell = -1
        self.cell_count = 0

    def boundary(self, kind: int, *, br: bool = False) -> None:
        at = len(self.chars)
        (self.brs if br else self.others).add(at)
        if self.bounds.get(at, NONE) < kind:
            self.bounds[at] = kind

    def text(self, value: str) -> None:
        value = value.replace(" ", " ")
        value = _WS_RE.sub(" ", _DROPPED_CHARS_RE.sub("", value))
        self.chars.extend(value)
        self.cells.extend([self.cell] * len(value))

    def walk(self, node: Tag) -> None:
        for child in node.children:
            if isinstance(child, (Comment, Declaration, Doctype, ProcessingInstruction)):
                continue
            if isinstance(child, NavigableString):
                self.text(str(child))
                continue
            if not isinstance(child, Tag):
                continue
            name = (child.name or "").lower()
            if name in _SKIP_TAGS:
                continue
            if _is_hidden(child):
                self.boundary(SOFT)
                continue
            if name == "br":
                self.boundary(HARD, br=True)
                continue
            if name in _ROOT_TAGS:
                self.walk(child)
                continue
            style = _style(child)
            if _SOFT_STYLE_RE.search(style):
                kind = SOFT
            elif name in INLINE_TAG_NAMES and not _BLOCK_STYLE_RE.search(style):
                self.walk(child)
                continue
            elif name in _HARD_BLOCK_TAGS or name in PRE_TAG_NAMES or _BLOCK_STYLE_RE.search(style):
                kind = HARD
            else:  # images, unknown elements, the end-tag marker
                kind = SOFT
            self.boundary(kind)
            outer = self.cell
            if name in ("td", "th"):
                self.cell_count += 1
                self.cell = self.cell_count
            self.walk(child)
            self.cell = outer
            self.boundary(kind)


def raw_space_stream(raw_html: str) -> RawSpaceStream:
    soup = BeautifulSoup(_NON_INLINE_END_TAG_RE.sub(r"\g<0><rsp-break></rsp-break>", raw_html), "html.parser")
    builder = _Builder()
    builder.walk(soup)
    text = "".join(builder.chars)
    alnum_pos = [i for i, ch in enumerate(text) if ch.isalnum()]
    return RawSpaceStream(
        text=text,
        bounds=builder.bounds,
        alnum="".join(text[i] for i in alnum_pos),
        alnum_pos=alnum_pos,
        brs=frozenset(builder.brs),
        others=frozenset(builder.others),
        cells=tuple(builder.cells),
    )


# --- cheap screen ----------------------------------------------------------------------------------------

_CHEAP_SKIP_RE = re.compile(r"<!--.*?-->|<(script|style|noscript|template|head|title)\b.*?</\1\s*>", re.DOTALL | re.IGNORECASE)
_CHEAP_TAG_RE = re.compile(r"<(/?)([A-Za-z][A-Za-z0-9:-]*)[^>]*>|<[^>]*>")


def _cheap_raw_tokens(raw_html: str) -> set[str]:
    """Words of the filed page, inline tags removed without a separator and every
    other tag read as a space: a stand-in for the parsed stream."""

    def tag(m: re.Match[str]) -> str:
        name = m.group(2)
        return "" if name is not None and name.lower() in INLINE_TAG_NAMES else " "

    text = _CHEAP_TAG_RE.sub(tag, _CHEAP_SKIP_RE.sub("", raw_html))
    text = _DROPPED_CHARS_RE.sub("", html.unescape(text).replace(" ", " "))
    return set(text.split())


def _split_points(token: str, words: set[str], max_pieces: int = 4) -> list[int] | None:
    """Offsets that cut ``token`` into 2..max_pieces words of the page, fewest pieces first."""
    n = len(token)
    if n < 2 or n > 120:
        return None
    # best[i] = fewest pieces covering token[:i]
    best: list[tuple[int, int] | None] = [None] * (n + 1)
    best[0] = (0, -1)
    for i in range(1, n + 1):
        for j in range(max(0, i - 60), i):
            prev = best[j]
            if prev is None or prev[0] >= max_pieces:
                continue
            current = best[i]
            if token[j:i] in words and (current is None or prev[0] + 1 < current[0]):
                best[i] = (prev[0] + 1, j)
    last = best[n]
    if last is None or last[0] < 2:
        return None
    cuts: list[int] = []
    i = n
    while i > 0:
        entry = best[i]
        assert entry is not None
        i = entry[1]
        if i > 0:
            cuts.append(i)
    return sorted(cuts)


# --- the page text ---------------------------------------------------------------------------------------

# Markup the tagger and the builder put into page text.
# (A <page> marker's number is a separate value: it is never edited.)
_MARKUP_RE = re.compile(r"<pageUUID>.*?</pageUUID>|<page>.*?</page>|</?(?:article|section)>", re.DOTALL)
_ROMAN_LETTERS = frozenset("IVXLCDM")
_LETTER_RUN_RE = re.compile(r"[^\W\d_]{2,}")
_HYPHENS = "-‐‑"
_NO_SPACE_NEXT_TO = frozenset("-‐‑‒–—―/\\")
_NO_SPACE_BEFORE = frozenset(".,;:!?)]}”’'\"%")
_NO_SPACE_AFTER = frozenset("([{“‘$#")
_KEY_WIDTHS = (40, 80, 160)
_SHRINK_WIDTHS = (24, 16, 10)
_MIN_KEY_TOTAL = 16


@dataclass
class RawSpaceResult:
    text: str
    edits: list[dict[str, object]] = field(default_factory=list)


def _find_unique(raw: RawSpaceStream, left: str, right: str) -> int | None:
    """Raw alnum index of the first ``right`` character, when the key around the
    junction occurs exactly once on the page."""

    def find(width: int) -> tuple[int, int] | None:
        lw, rw = min(width, len(left)), min(width, len(right))
        if lw + rw < _MIN_KEY_TOTAL or not lw or not rw:
            return None
        key = left[len(left) - lw:] + right[:rw]
        first = raw.alnum.find(key)
        if first < 0:
            return 0, -1
        if raw.alnum.find(key, first + 1) >= 0:
            return 2, -1
        return 1, first + lw

    result = find(_KEY_WIDTHS[0])
    if result is None:
        return None
    if result[0] == 1:
        return result[1]
    widths = _KEY_WIDTHS[1:] if result[0] == 2 else _SHRINK_WIDTHS
    for width in widths:
        other = find(width)
        if other is None:
            return None
        if other[0] == 1:
            return other[1]
        if (result[0] == 2 and other[0] == 0) or (result[0] == 0 and other[0] == 2):
            return None
    return None


def _raw_gap(raw: RawSpaceStream, rp1: int, rp2: int, k: int) -> tuple[bool, int, int, int] | None:
    """White space and the strongest boundary in the raw gap after the k-th
    printed character that follows raw position rp1 (k = 0: right after rp1),
    up to the next printed character (before rp2 at the latest), with the raw
    positions of the printed characters on its two sides."""
    a = rp1
    need = k
    pos = rp1 + 1
    while need:
        while pos < rp2 and raw.text[pos] == " ":
            pos += 1
        if pos >= rp2:
            return None
        a = pos
        pos += 1
        need -= 1
    b = a + 1
    while b < rp2 and raw.text[b] == " ":
        b += 1
    space = b > a + 1
    bound = max((raw.bounds.get(x, NONE) for x in range(a + 1, b + 1)), default=NONE)
    return space, bound, a, b


class _PageView:
    """The printed characters of a page text (markup removed) and their offsets."""

    def __init__(self, text: str) -> None:
        chars: list[str] = []
        pos: list[int] = []
        # markup_after[i]: markup sits between visible char i-1 and i
        self.markup_before: set[int] = set()
        # visible indices inside an <article> or <section> heading
        self.heading: set[int] = set()
        in_heading = False
        last = 0
        for m in _MARKUP_RE.finditer(text):
            for i in range(last, m.start()):
                if in_heading:
                    self.heading.add(len(chars))
                chars.append(text[i])
                pos.append(i)
            self.markup_before.add(len(chars))
            tag = m.group(0)
            if tag in ("<article>", "<section>"):
                in_heading = True
            elif tag in ("</article>", "</section>"):
                in_heading = False
            last = m.end()
        for i in range(last, len(text)):
            chars.append(text[i])
            pos.append(i)
        self.vis = "".join(chars)
        self.pos = pos
        alnum_idx = [i for i, ch in enumerate(self.vis) if ch.isalnum()]
        self.alnum = "".join(self.vis[i] for i in alnum_idx)
        self.alnum_idx = alnum_idx
        # count of alnum chars before each visible index
        before = [0] * (len(self.vis) + 1)
        c = 0
        for i, ch in enumerate(self.vis):
            before[i] = c
            if ch.isalnum():
                c += 1
        before[len(self.vis)] = c
        self.alnum_before = before


def _locate(view: _PageView, raw: RawSpaceStream, j: int) -> tuple[int, int, int, int] | None:
    """Align the visible junction before view.vis[j] to the raw page.

    Returns (pp1, pp2, rp1, rp2): the last letter/digit before the junction and
    the first one after it, in the page view and in the raw stream."""
    a = view.alnum_before[j]
    if a == 0 or a >= len(view.alnum):
        return None
    # Keys never reach past the widest key, so a window keeps this linear.
    width = _KEY_WIDTHS[-1]
    r = _find_unique(raw, view.alnum[max(0, a - width):a], view.alnum[a:a + width])
    if r is None or r <= 0 or r >= len(raw.alnum_pos):
        return None
    return view.alnum_idx[a - 1], view.alnum_idx[a], raw.alnum_pos[r - 1], raw.alnum_pos[r]


def _nonspace(text: str) -> str:
    return "".join(text.split())


def _check_junction(view: _PageView, raw: RawSpaceStream, j: int) -> tuple[bool, int, int, int] | None:
    """The raw separator at the visible junction before view.vis[j] (j is the
    first character after the junction, or the space at it)."""
    located = _locate(view, raw, j)
    if located is None:
        return None
    pp1, pp2, rp1, rp2 = located
    page_between = view.vis[pp1 + 1:pp2]
    if _nonspace(page_between) != _nonspace(raw.text[rp1 + 1:rp2]):
        return None
    k = len(_nonspace(view.vis[pp1 + 1:j]))
    return _raw_gap(raw, rp1, rp2, k)


def _token_spans(text: str) -> list[tuple[int, int]]:
    return [m.span() for m in re.finditer(r"\S+", text)]


def _screen(plain: str, words: set[str]) -> bool:
    """Whether the page text has a glued token or a hyphen space worth aligning."""
    tokens = plain.split()
    for k, token in enumerate(tokens):
        if token not in words and _split_points(token, words):
            return True
        if k + 1 < len(tokens):
            nxt = tokens[k + 1]
            if (token[-1] in _HYPHENS or nxt[0] in _HYPHENS) and token + nxt in words:
                return True
    return False


def _candidates(view: _PageView, words: set[str]) -> tuple[list[int], list[int]]:
    """(insert_at, remove_at): visible indices of glued junctions to test (the
    first character after the junction), and of spaces next to a hyphen."""
    inserts: list[int] = []
    removes: list[int] = []
    vis = view.vis
    spans = _token_spans(vis)
    for k, (s, e) in enumerate(spans):
        token = vis[s:e]
        if token not in words:
            for c in _split_points(token, words) or ():
                inserts.append(s + c)
        if k + 1 < len(spans):
            s2, e2 = spans[k + 1]
            if s2 == e + 1 and (vis[e - 1] in _HYPHENS or vis[s2] in _HYPHENS) and token + vis[s2:e2] in words:
                removes.append(e)
    return inserts, removes


def _letter_words(words: set[str]) -> set[str]:
    """The casefolded letter runs of a page's words ("Aktiengesellschaft," -> "aktiengesellschaft")."""
    return {run.casefold() for word in words for run in _LETTER_RUN_RE.findall(word)}


@dataclass
class PrintedWords:
    """What an agreement's filed pages print whole: casefolded letter runs, and raw words."""

    letters: set[str] = field(default_factory=set)
    tokens: set[str] = field(default_factory=set)

    @classmethod
    def of(cls, words: set[str]) -> PrintedWords:
        return cls(_letter_words(words), set(words))


_CJK_RANGES = (
    (0x1100, 0x11FF), (0x2E80, 0x2FDF), (0x3000, 0x30FF), (0x3100, 0x31FF), (0x3200, 0x9FFF),
    (0xA960, 0xA97F), (0xAC00, 0xD7FF), (0xF900, 0xFAFF), (0xFE30, 0xFE4F), (0xFF00, 0xFFEF),
    (0x20000, 0x3FFFF),
)


def _is_cjk(ch: str) -> bool:
    """Han, kana, Hangul and full-width characters."""
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _CJK_RANGES)


# A registration or application number: up to four letters, then a digit, then
# letters, digits and ". - /" ("D879715", "201730550953.X", "2011P07671").
_ID_SHAPE_RE = re.compile(r"[A-Za-z]{0,4}\d[A-Za-z\d.\-/]*")


def _raw_run(raw: RawSpaceStream, pos: int, step: int) -> str:
    """The raw characters from ``pos`` outward (step -1 or 1) up to white space or an element boundary."""
    out: list[str] = []
    i = pos
    while 0 <= i < len(raw.text) and raw.text[i] != " ":
        out.append(raw.text[i])
        if (i if step < 0 else i + 1) in raw.bounds:  # a boundary before i, or before the next character
            break
        i += step
    return "".join(reversed(out)) if step < 0 else "".join(out)


def _is_wrapped_token(raw: RawSpaceStream, a: int, b: int, printed: PrintedWords) -> bool:
    """Whether a lone <br> between raw positions a and b wraps one number in a
    narrow cell ("D87971<br/>5", "201722<br/>535") rather than separating two values.

    Only a <br> with no white space and no other boundary, next to a digit. The
    token is whole when the agreement prints it unbroken elsewhere, or when the
    right piece is a short tail (at most four letters or digits, with a digit,
    shorter than the left piece, which also has a digit) of an id-shaped token.
    Two full values ("15-005003.1000<br/>15-005003.0000", "82.109%<br/>73%",
    "19494-03<br/>DNPI") keep their space. A gap with other block boundaries is
    a wrap only inside one table cell (``_is_cell_block_wrap``)."""
    left, right = raw.text[a], raw.text[b]
    if not (left.isalnum() and right.isalnum() and (left.isdigit() or right.isdigit())):
        return False
    gap = range(a + 1, b + 1)
    if not any(x in raw.brs or x in raw.others for x in gap):
        return False
    lone_br = not any(x in raw.others for x in gap)
    left_run, right_run = _raw_run(raw, a, -1), _raw_run(raw, b, 1)
    if not lone_br:
        return _is_cell_block_wrap(raw, a, b, left_run, right_run, printed)
    if left_run + right_run in printed.tokens:
        return True
    return _is_id_tail(left_run, right_run)


def _is_id_tail(left_run: str, right_run: str) -> bool:
    """The right piece is a short tail of one id-shaped token (see ``_is_wrapped_token``)."""
    left_alnum = [c for c in left_run if c.isalnum()]
    right_alnum = [c for c in right_run if c.isalnum()]
    return (
        _ID_SHAPE_RE.fullmatch(left_run + right_run) is not None
        and len(right_alnum) <= 4
        and len(right_alnum) < len(left_alnum)
        and any(c.isdigit() for c in left_alnum)
        and any(c.isdigit() for c in right_alnum)
    )


def _is_cell_block_wrap(
    raw: RawSpaceStream, a: int, b: int, left_run: str, right_run: str, printed: PrintedWords
) -> bool:
    """Whether block boundaries inside one table cell ("<td><p>0602CA00070</p><p>3</p></td>")
    wrap one id rather than separate two lines.

    Only when both sides sit in the same <td>/<th> and every boundary between
    them is a hard one (a <p>, <div> or <br>, never a cell edge or an unsure box).
    The id is whole when the agreement prints it unbroken elsewhere, or when the
    left piece fills its line (8 or more letters or digits, no space before them
    on the line) and the right piece is at most 2 letters or digits or a short
    id tail, and that tail is its whole line. Two lines ("MAC C7300-107</div><div>1700 Lincoln St",
    "100</div><div>1</div><div>3,500", "O:847-485-4752</div><div>M: 847-284-4495") keep their space."""
    if not raw.cells or raw.cells[a] < 0 or raw.cells[a] != raw.cells[b]:
        return False
    if any(raw.bounds.get(x, NONE) not in (NONE, HARD) for x in range(a + 1, b + 1)):
        return False
    if left_run + right_run in printed.tokens:
        return True
    start, end = a - len(left_run) + 1, b + len(right_run)
    if not (_at_line_edge(raw, start - 1, -1) and _at_line_edge(raw, end, 1)):
        return False  # the left piece must start its line and the right piece must be its whole line
    if sum(c.isalnum() for c in left_run) < 8:
        return False
    return sum(c.isalnum() for c in right_run) <= 2 or _is_id_tail(left_run, right_run)


def _at_line_edge(raw: RawSpaceStream, pos: int, step: int) -> bool:
    """Whether a run that stops just before ``pos`` (stepping by ``step``) meets the edge of its line:
    the page edge or an element boundary, after at most one space."""
    edge = pos + 1 if step < 0 else pos
    if edge <= 0 or edge >= len(raw.text) or edge in raw.bounds:
        return True
    if raw.text[pos] == " ":
        nxt = pos if step < 0 else pos + 1
        return nxt <= 0 or nxt >= len(raw.text) or nxt in raw.bounds
    return False


def restore_raw_spaces(
    page_text: str,
    raw_html: str | None,
    *,
    collect: bool = False,
    printed_words: PrintedWords | None = None,
) -> RawSpaceResult:
    """Insert the spaces a filed HTML page prints between glued words, and remove
    spaces next to hyphens that it does not print (see the module docstring).

    ``printed_words`` is what the agreement's filed pages print whole: a line
    break inside such a word is a hand wrap in a narrow cell
    ("Aktiengese<br/>llschaft", "201722<br/>535"), not a word space."""
    if not page_text or not raw_html:
        return RawSpaceResult(page_text)
    words = _cheap_raw_tokens(raw_html)
    if printed_words is None:
        printed_words = PrintedWords.of(words)
    if not _screen(_MARKUP_RE.sub(" ", page_text), words):
        return RawSpaceResult(page_text)
    view = _PageView(page_text)
    inserts, removes = _candidates(view, words)
    if not inserts and not removes:
        return RawSpaceResult(page_text)
    raw = raw_space_stream(raw_html)
    vis = view.vis
    insert_at: list[int] = []  # page_text offsets: put a space before this char
    remove_at: list[int] = []  # page_text offsets of spaces to delete
    edits: list[dict[str, object]] = []
    for j in sorted(set(inserts)):
        left, right = vis[j - 1], vis[j]
        if left.isspace() or right.isspace() or j in view.markup_before:
            continue
        if view.pos[j] != view.pos[j - 1] + 1:
            continue
        if left in _NO_SPACE_NEXT_TO or right in _NO_SPACE_NEXT_TO:
            continue
        if right in _NO_SPACE_BEFORE or left in _NO_SPACE_AFTER:
            continue
        if _is_cjk(left) and _is_cjk(right):
            # Chinese, Japanese, Korean and full-width text: a line break is never a word space.
            continue
        if j in view.heading and (
            (left.isdigit() and right.isdigit()) or (left in _ROMAN_LETTERS and right in _ROMAN_LETTERS)
        ):
            # A heading number printed across a line break ("ARTICLE X<br>I",
            # "SECTION 1<br>1.14") stays one number.
            continue
        if left.islower() and right.islower():
            a = j - 1
            while a > 0 and vis[a - 1].isalpha():
                a -= 1
            b = j + 1
            while b < len(vis) and vis[b].isalpha():
                b += 1
            # the word may start at the letter run's start or at a capital in it
            starts = [a] + [k for k in range(a + 1, j) if vis[k].isupper()]
            if any(vis[k:b].casefold() in printed_words.letters for k in starts):
                continue
        gap = _check_junction(view, raw, j)
        if gap is None:
            continue
        space, bound, ra, rb = gap
        if not (space or bound == HARD):
            continue
        if not space and _is_wrapped_token(raw, ra, rb, printed_words):
            continue
        insert_at.append(view.pos[j])
        if collect:
            edits.append({"kind": "insert", "at": view.pos[j], "why": "space" if space else "boundary",
                          "context": vis[max(0, j - 40):j] + "|" + vis[j:j + 40]})
    for i in removes:
        if i in view.markup_before or i + 1 in view.markup_before:
            continue
        # Only a hyphen joined to a word on its other side ("Anti -Corruption",
        # "Post- Closing"); never in rules or leaders ("- - - -").
        if vis[i + 1] in _HYPHENS:
            if not (vis[i - 1].isalnum() and i + 2 < len(vis) and vis[i + 2].isalnum()):
                continue
        elif not (vis[i + 1].isalnum() and i >= 2 and vis[i - 2].isalnum()):
            continue
        if view.pos[i + 1] != view.pos[i] + 1 or view.pos[i] != view.pos[i - 1] + 1:
            continue
        gap = _check_junction(view, raw, i)
        if gap is None:
            continue
        space, bound, _, _ = gap
        if space or bound != NONE:
            continue
        remove_at.append(view.pos[i])
        if collect:
            edits.append({"kind": "remove", "at": view.pos[i],
                          "context": vis[max(0, i - 40):i] + "|" + vis[i + 1:i + 41]})
    if not insert_at and not remove_at:
        return RawSpaceResult(page_text, edits)
    out: list[str] = []
    last = 0
    removed = set(remove_at)
    for at in sorted(set(insert_at) | removed):
        out.append(page_text[last:at])
        if at in removed:
            last = at + 1
        else:
            out.append(" ")
            last = at
    out.append(page_text[last:])
    return RawSpaceResult("".join(out), edits)


def _has_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_html_page(row: Mapping[str, object]) -> bool:
    is_html = row.get("source_is_html")
    return is_html is not None and is_html == is_html and bool(is_html) and _has_text(row.get("raw_page_content"))


def restore_page_text(text: str, row: Mapping[str, object], printed_words: PrintedWords | None = None) -> str:
    """The builder's page-text pass: raw-backed spaces for HTML pages, then glyphs."""
    if text and _is_html_page(row):
        text = restore_raw_spaces(text, str(row.get("raw_page_content")), printed_words=printed_words).text
    return map_private_use_glyphs(text) if text else text


def agreement_printed_words(rows: list[Mapping[str, object]]) -> PrintedWords:
    """What every filed HTML page of an agreement prints whole."""
    tokens: set[str] = set()
    for row in rows:
        if _is_html_page(row):
            tokens |= _cheap_raw_tokens(str(row.get("raw_page_content")))
    return PrintedWords.of(tokens)


_HEADING_RE = re.compile(r"<(article|section)>(.*?)</\1>", re.DOTALL)


def heading_id_titles(id_tagged_text: str, matches: list[re.Match[str]]) -> list[str] | None:
    """The heading titles of the text before the space fixes, normalized as the
    builder normalizes titles, for hashing ids; None when its headings do not line
    up with ``matches`` (the headings of the fixed text) or nothing changed."""
    id_matches = list(_HEADING_RE.finditer(id_tagged_text))
    if [m.group(1) for m in id_matches] != [m.group(1) for m in matches]:
        return None
    if all(a.group(2) == b.group(2) for a, b in zip(id_matches, matches)):
        return None
    titles: list[str] = []
    previous: int | None = None
    for m in id_matches:
        title = " ".join(m.group(2).strip().split())
        if m.group(1) == "article":
            title = normalize_glued_article_title(title, previous + 1 if previous is not None else None)
            number = article_title_number(title)
            if number is not None:
                previous = number
        else:
            title = normalize_glued_section_title(title)
        titles.append(title)
    return titles
