# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false
"""Join page-text paragraphs that the raw HTML prints as one block.

HTML page extraction sometimes breaks one printed paragraph into several
lines of page text. Two causes account for most of it:

* The source puts blank lines inside a ``<p>`` or ``<div>``. A browser shows
  them as one space, but the extractor reads a blank line as a paragraph
  break.
* A page split cuts the HTML mid-block, so the page starts with text that has
  no enclosing element. The extractor makes each top-level text node and
  inline tag its own paragraph (``provided in`` / ``Section 2.3(e))`` /
  ``equal to the product``).

Each line becomes its own ``<text>`` element, so the served paragraph is split
mid-sentence. This module rejoins two consecutive lines of one page when the
raw page proves they are one printed block, and it changes whitespace only.

A junction between two lines is joined only when all of these hold:

* Both sides are plain text: the end of the first line is not inside an
  ``<article>``/``<section>`` heading or a ``<page>`` marker, and the second
  line does not start with a tag, a quote, or an all-caps heading. If the
  first line ends a sentence or a clause (``. : ; ! ?``), the second must
  start with a lowercase letter.
* A second line that starts with an enumerator (``(a)``, ``1.``, ``•``)
  joins only an inline run ("by multiplying (A) the number"): the raw has a
  space at the junction and no source line break, and the first line does
  not end a clause. A list item on its own source line is never joined.
* The last 40 letters and digits of the first line plus the first 40 of the
  second occur exactly once in the raw page (the window grows or shrinks
  when that is ambiguous or absent).
* In the raw page, the two sides sit in the same block: no block element
  boundary, no ``<br>``, no table cell, no hidden element, and no
  ``<pre>``/``white-space: pre`` line break between them. A blank line in
  text that has no enclosing element is also kept as a break, since such
  pages may be plain text.
* The printed characters between the two sides in the raw page are exactly
  the characters between them in the page text.

The join is the raw page's own separator: one space when the raw has white
space at the junction, nothing when it has none (``provided`` + ``, that``,
or a word split across two font tags, ``P`` + ``ropriety``). So the printed
characters never change, and no space is ever put inside a word.

Lines are never joined across a page, and the page text is left unchanged
when the raw page is missing or cannot be aligned.
"""
import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup
from bs4.element import Comment, Declaration, Doctype, NavigableString, ProcessingInstruction, Tag

# Tags whose content flows inline: no paragraph break before or after.
_INLINE_TAGS = frozenset(
    {
        "a", "abbr", "acronym", "b", "bdi", "bdo", "big", "cite", "code", "data",
        "del", "dfn", "em", "font", "i", "ins", "kbd", "label", "mark", "nobr",
        "q", "s", "samp", "small", "span", "strike", "strong", "sub", "sup",
        "time", "tt", "u", "var", "wbr",
    }
)
# Containers that do not open a block of their own (the anonymous top level).
_ROOT_TAGS = frozenset({"html", "body"})
_SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "head", "title"})
_PRE_TAGS = frozenset({"pre", "xmp", "listing", "plaintext", "textarea"})
_PRE_STYLE_RE = re.compile(r"white-space\s*:\s*pre", re.IGNORECASE)
_BLOCK_STYLE_RE = re.compile(
    r"(?:^|;)\s*(?:display\s*:\s*(?:block|flex|grid|table|list-item)|position\s*:\s*(?:absolute|fixed)|float\s*:\s*(?:left|right))",
    re.IGNORECASE,
)
_HIDDEN_STYLE_RE = re.compile(
    r"(?:^|;)\s*(?:display\s*:\s*none|visibility\s*:\s*(?:hidden|collapse))", re.IGNORECASE
)
# Characters the page-text extractor drops (normalize_text).
_DROPPED_CHARS_RE = re.compile("[\x00-\x08\x0B\x0C\x0E-\x1F\u200B-\u200F\u202A-\u202E\u2060\uFEFF]")
_BLANK_LINE_RE = re.compile(r"\n[^\S\n]*\n")
_WS_RE = re.compile(r"\s+")

# Markup the tagger and the builder put into page text.
_MARKUP_RE = re.compile(r"</?(?:article|section|page|pageUUID)>")
_OPEN_HEADING_RE = re.compile(r"<(article|section)>|</(article|section)>")
_PAGE_UUID_RE = re.compile(r"<pageUUID>.*?</pageUUID>", re.DOTALL)
_ENUMERATOR_RE = re.compile(
    r"^(?:\(\s*[A-Za-z0-9]{1,6}\s*\)|[A-Za-z0-9]{1,4}\)|\d{1,3}(?:\.\d{1,3})*\.(?:\s|$)|[ivxlcIVXLC]{1,6}\.\s|[A-Z]\.\s|[•·▪◦●○■□\-–—*§¶])"
)
_QUOTE_START_RE = re.compile(r"^[\"'“”‘’«»]")
_CLAUSE_END_RE = re.compile(r"[.:;!?][\"'”’)\]]*$")
_MIN_KEY = 12
_KEY_WIDTHS = (40, 80, 160, 320)
_SHRINK_WIDTHS = (24, 16, _MIN_KEY)


@dataclass
class RawStream:
    """The visible text of a raw HTML page, one character per entry.

    ``text`` holds the characters with white space collapsed to single
    spaces; ``blocks[i]`` is the id of the printed block that ``text[i]``
    belongs to. A new id starts at every block boundary.
    """

    text: str
    blocks: list[int]
    alnum: str
    alnum_pos: list[int]


@dataclass
class BlockJoinResult:
    text: str
    joins: list[dict[str, object]] = field(default_factory=list)


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


def _collapse_ws(text: str) -> str:
    # One character per white-space run: a newline when the run holds a source
    # line break, else a space. Both print as one space.
    return _WS_RE.sub(lambda m: "\n" if "\n" in m.group(0) else " ", text)


class _StreamBuilder:
    def __init__(self) -> None:
        self.chars: list[str] = []
        self.blocks: list[int] = []
        self.block = 0

    def boundary(self) -> None:
        self.block += 1

    def emit(self, text: str) -> None:
        for ch in text:
            self.chars.append(ch)
            self.blocks.append(self.block)

    def text_node(self, text: str, *, pre: bool, in_block: bool) -> None:
        text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\u00a0", " ")
        text = _DROPPED_CHARS_RE.sub("", text)
        if pre:
            for i, line in enumerate(text.split("\n")):
                if i:
                    self.boundary()
                self.emit(line)
            return
        if not in_block:
            # Text with no enclosing element: a blank line may be a printed
            # paragraph break in a plain-text page, so keep it as one.
            parts = _BLANK_LINE_RE.split(text)
            for i, part in enumerate(parts):
                if i:
                    self.boundary()
                self.emit(_collapse_ws(part))
            return
        self.emit(_collapse_ws(text))

    def walk(self, node: Tag, *, pre: bool, in_block: bool) -> None:
        for child in node.children:
            if isinstance(child, (Comment, Declaration, Doctype, ProcessingInstruction)):
                continue
            if isinstance(child, NavigableString):
                self.text_node(str(child), pre=pre, in_block=in_block)
                continue
            if not isinstance(child, Tag):
                continue
            name = (child.name or "").lower()
            if name in _SKIP_TAGS:
                continue
            if _is_hidden(child):
                self.boundary()
                continue
            if name == "br":
                self.boundary()
                continue
            style = _style(child)
            child_pre = pre or name in _PRE_TAGS or bool(_PRE_STYLE_RE.search(style))
            if name in _ROOT_TAGS:
                self.walk(child, pre=child_pre, in_block=in_block)
                continue
            if name in _INLINE_TAGS and not _BLOCK_STYLE_RE.search(style):
                self.walk(child, pre=child_pre, in_block=in_block)
                continue
            # Block elements, tables, images and anything unknown: a break
            # on both sides.
            self.boundary()
            self.walk(child, pre=child_pre, in_block=True)
            self.boundary()


def raw_stream(raw_html: str) -> RawStream:
    soup = BeautifulSoup(raw_html, "html.parser")
    builder = _StreamBuilder()
    builder.walk(soup, pre=False, in_block=False)
    text = "".join(builder.chars)
    alnum_pos = [i for i, ch in enumerate(text) if ch.isalnum()]
    return RawStream(
        text=text,
        blocks=builder.blocks,
        alnum="".join(text[i] for i in alnum_pos),
        alnum_pos=alnum_pos,
    )


def _letters_upper_share(text: str) -> float:
    letters = [ch for ch in text if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for ch in letters if ch.isupper()) / len(letters)


def _looks_like_caps_heading(text: str) -> bool:
    words = text.split()
    return 0 < len(words) <= 12 and _letters_upper_share(text) >= 0.8 and sum(ch.isalpha() for ch in text) >= 4


def _line_tail(line: str, heading_open_at_end: bool) -> str | None:
    """The plain text at the end of a line, or None when it is not joinable."""
    if heading_open_at_end:
        return None
    last = None
    for m in _MARKUP_RE.finditer(line):
        last = m
    tail = line[last.end():] if last is not None else line
    if last is not None and last.group(0) in ("<page>", "<pageUUID>"):
        return None
    if not any(ch.isalnum() for ch in tail):
        return None
    return tail


def _line_head(line: str) -> tuple[str, bool] | None:
    """The plain text at the start of a line and whether it starts with an
    enumerator, or None when it is not joinable."""
    m = _MARKUP_RE.search(line)
    head = line[: m.start()] if m is not None else line
    if m is not None and m.start() == 0:
        return None
    if not any(ch.isalnum() for ch in head):
        return None
    stripped = head.lstrip()
    if _QUOTE_START_RE.match(stripped):
        return None
    enumerator = bool(_ENUMERATOR_RE.match(stripped))
    if _looks_like_caps_heading(line if m is None else head):
        return None
    return head, enumerator


def _find_unique(raw: RawStream, left: str, right: str) -> int | None:
    """Raw alnum index of the first ``right`` character, if the key is unique.

    The key is the end of ``left`` plus the start of ``right``. It widens
    while it is ambiguous and narrows while it is absent (the page text may
    differ from the raw a little way off, e.g. a stripped page number); a
    key that stays ambiguous or absent is no match.
    """

    def find(width: int) -> tuple[int, int] | None:
        lw, rw = min(width, len(left)), min(width, len(right))
        if lw + rw < 2 * _MIN_KEY:
            return None
        key = left[len(left) - lw :] + right[:rw]
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
    if result[0] == 2:
        for width in _KEY_WIDTHS[1:]:
            wider = find(width)
            if wider is None or wider[0] == 0:
                return None
            if wider[0] == 1:
                return wider[1]
        return None
    for width in _SHRINK_WIDTHS:
        narrower = find(width)
        if narrower is None or narrower[0] == 2:
            return None
        if narrower[0] == 1:
            return narrower[1]
    return None


def _after_last_alnum(text: str) -> str:
    for i in range(len(text) - 1, -1, -1):
        if text[i].isalnum():
            return text[i + 1 :]
    return text


def _before_first_alnum(text: str) -> str:
    for i, ch in enumerate(text):
        if ch.isalnum():
            return text[:i]
    return text


def _nonspace(text: str) -> str:
    return _WS_RE.sub("", text)


def _raw_joiner(raw: RawStream, rp1: int, rp2: int, tail_rest: str, head_rest: str) -> tuple[str, bool] | None:
    """The raw separator at the junction and whether it holds a source line
    break, or None when the raw does not match."""
    if raw.blocks[rp1] != raw.blocks[rp2]:
        return None
    seg = raw.text[rp1 + 1 : rp2]
    expected = _nonspace(tail_rest) + _nonspace(head_rest)
    if _nonspace(seg) != expected:
        return None
    # Walk past the first line's trailing characters; what follows, up to the
    # second line's first character, is the junction.
    need = len(_nonspace(tail_rest))
    i = 0
    while need and i < len(seg):
        if not seg[i].isspace():
            need -= 1
        i += 1
    j = i
    while j < len(seg) and seg[j].isspace():
        j += 1
    gap = seg[i:j]
    return (" " if gap else ""), "\n" in gap


def _heading_open_after_each_line(lines: list[str]) -> list[bool]:
    open_tags: list[str] = []
    result: list[bool] = []
    for line in lines:
        for m in _OPEN_HEADING_RE.finditer(line):
            if m.group(1):
                open_tags.append(m.group(1))
            elif open_tags and open_tags[-1] == m.group(2):
                open_tags.pop()
        result.append(bool(open_tags))
    return result


def join_raw_block_splits(page_text: str, raw_html: str | None) -> BlockJoinResult:
    """Join consecutive lines of one page that the raw HTML prints as one block."""
    if not page_text or not raw_html or not raw_html.strip():
        return BlockJoinResult(page_text)
    lines = page_text.split("\n")
    nonempty = [i for i, line in enumerate(lines) if line.strip()]
    if len(nonempty) < 2:
        return BlockJoinResult(page_text)
    open_after = _heading_open_after_each_line(lines)

    candidates: list[tuple[int, int, str, str, bool]] = []
    for a, b in zip(nonempty, nonempty[1:]):
        line_a, line_b = lines[a].rstrip(), lines[b].lstrip()
        tail = _line_tail(line_a, open_after[a])
        head_info = _line_head(line_b)
        if tail is None or head_info is None:
            continue
        head, enumerator = head_info
        if enumerator and _CLAUSE_END_RE.search(tail.rstrip()):
            continue
        if _CLAUSE_END_RE.search(tail.rstrip()) and not head.lstrip()[:1].islower():
            continue
        if _looks_like_caps_heading(tail) and not head.lstrip()[:1].islower():
            continue
        candidates.append((a, b, tail, head, enumerator))
    if not candidates:
        return BlockJoinResult(page_text)

    # Letters and digits of the page text (markup removed), with the line
    # each one came from.
    page_alnum: list[str] = []
    alnum_line: list[int] = []
    for i, line in enumerate(lines):
        visible = _MARKUP_RE.sub("", _PAGE_UUID_RE.sub("", line))
        for ch in visible:
            if ch.isalnum():
                page_alnum.append(ch)
                alnum_line.append(i)
    alnum_text = "".join(page_alnum)
    first_alnum_of_line: dict[int, int] = {}
    last_alnum_of_line: dict[int, int] = {}
    for k, i in enumerate(alnum_line):
        first_alnum_of_line.setdefault(i, k)
        last_alnum_of_line[i] = k

    raw = raw_stream(raw_html)
    if not raw.alnum:
        return BlockJoinResult(page_text)

    joiners: dict[int, str] = {}
    joins: list[dict[str, object]] = []
    for a, b, tail, head, enumerator in candidates:
        ka, kb = last_alnum_of_line.get(a), first_alnum_of_line.get(b)
        if ka is None or kb is None or kb != ka + 1:
            continue
        left, right = alnum_text[: ka + 1], alnum_text[kb:]
        pos = _find_unique(raw, left, right)
        if pos is None or pos <= 0 or pos >= len(raw.alnum_pos):
            continue
        rp1, rp2 = raw.alnum_pos[pos - 1], raw.alnum_pos[pos]
        tail_rest = _after_last_alnum(tail.rstrip())
        head_rest = _before_first_alnum(head.lstrip())
        found = _raw_joiner(raw, rp1, rp2, tail_rest, head_rest)
        if found is None:
            continue
        joiner, line_break = found
        if enumerator and (line_break or not joiner):
            # An enumerator joins only an inline run ("by multiplying (A) the
            # number"), never a list item set on its own source line.
            continue
        joiners[a] = joiner
        joins.append(
            {
                "left": lines[a].rstrip()[-80:],
                "right": lines[b].lstrip()[:80],
                "joiner": joiner,
                "raw": raw.text[max(0, rp1 - 60) : rp2 + 60],
            }
        )
    if not joiners:
        return BlockJoinResult(page_text)

    out: list[str] = []
    i = 0
    while i < len(lines):
        if i in joiners:
            # Merge this line with the next non-empty one, dropping the blank
            # lines between them.
            merged = lines[i].rstrip()
            j = i
            while j in joiners:
                nxt = next(k for k in nonempty if k > j)
                merged = merged + joiners[j] + lines[nxt].strip()
                j = nxt
            out.append(merged)
            i = j + 1
            continue
        out.append(lines[i])
        i += 1
    return BlockJoinResult("\n".join(out), joins)
