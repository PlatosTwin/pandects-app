# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false
"""Join a paragraph cut by a page break.

Each page's text is built on its own, so a sentence that runs over a page
break is served as two ``<text>`` elements with the page markers between them
("any amounts payable by the Purchaser" / <page>5</page> / "to the Sellers").
This pass moves the start of the next page up onto the end of the page before,
so the reader gets one paragraph, and the page markers stay in order, right
after it.

The last line of a body page and the first line of the next body page are
joined only when all of these hold:

* The pages are adjacent in the filing (page orders differ by one).
* The left line is plain text: it is not a heading, it does not sit after a
  page number printed below other text, and it is not a short caption (all
  caps or title case).
* The left line does not end a sentence or a clause (``. : ; ! ?``, also before
  a closing quote or bracket), it does not end a list item (``; and``, ``; or``),
  and it does not end in a dash.
* The left line is not a line of a notice block (an e-mail address, a phone or
  fax number, an "Attention" line).
* The right text runs up to the first heading or page marker on its line. It is
  not an enumerated list item, a "(continued)" or "page N" label, or the next
  line of a notice block ("with a copy to:", "if to Seller,"), and:

  - it starts with a lowercase letter or with closing punctuation, or
  - the left line has at least four words and ends in a word that cannot end a
    sentence ("of the"), and the right text starts with a capital, a digit or a
    quote, without opening on an all-caps word ("NOW, THEREFORE") unless the
    left line is all caps as well.

* Neither line repeats at the same page edge elsewhere in the agreement.
* On HTML pages, the filed page confirms both sides: each is found on its page,
  outside tables, lists and hidden elements, and the two sides have the same
  font size (a footnote is smaller than the sentence it interrupts). Neither is
  in a heading element or a centered block, unless its line is long enough to be
  prose (some filings set whole paragraphs in ``<h5>``).

The join changes white space and nothing else, with one exception. A left line
that ends in a word hyphen ("con-" / "sent") joins with no space. The hyphen is
dropped only when the agreement prints the closed word ("consent") more often
than the hyphenated one. Otherwise the printed hyphen stays ("third-party"), and
a suspended pair ("pre- and post-closing") keeps its space too.

Headings never move, so no article or section id, order or pageUUID changes.
"""
import html
import re
from bisect import bisect_right
from collections import Counter
from dataclasses import dataclass, field

_MARKUP_RE = re.compile(r"</?(?:article|section|page|pageUUID)\b[^>]*>")
_OPEN_HEADING_RE = re.compile(r"<(/?)(?:article|section)>")
_PAGE_MARKER_RE = re.compile(r"<page>[^<]*</page>")
_TAIL_MARKERS_RE = re.compile(r"(?:\s*<page>[^<]*</page>)*\s*$")
_LEADING_MARKERS_RE = re.compile(r"\s*(?:<page>[^<]*</page>\s*)*")
_LIST_ITEM_END_RE = re.compile(r";\s*(?:and|or|and/or)$", re.IGNORECASE)
_LINE_RE = re.compile(r"[^\n]*\S[^\n]*")
_LINE_CONTENT_RE = re.compile(r"[^\W_]")

_ENUMERATOR_RE = re.compile(
    r"^(?:\(\s*[A-Za-z0-9]{1,6}\s*\)|\[\s*[A-Za-z0-9]{1,6}\s*\]|[A-Za-z0-9]{1,4}\)"
    r"|\d{1,3}(?:\.\d{1,3})*\.(?:\s|$)|[ivxlcdmIVXLCDM]{1,6}\.(?:\s|$)|[A-Za-z]\.(?:\s|$)"
    r"|[•·▪◦●○■□\-–—*§¶])"
)
_SENTENCE_END_RE = re.compile(r"[.:;!?…][\"'”’»)\]]*$")
_DASH_END_RE = re.compile(r"[–—―]$|(?:^|\s)-$|--$")
_WORD_HYPHEN_END_RE = re.compile(r"([^\W\d_]+(?:-[^\W\d_]+)*)-$")
_RIGHT_WORD_RE = re.compile(r"[^\W\d_]+")
_WORD_TOKEN_RE = re.compile(r"[^\W\d_]+(?:-[^\W\d_]+)*")
_LABEL_RE = re.compile(r"^\(?(?:continued|cont(?:'|’|\.)d?|page\s+\d+|\d+\s+of\s+\d+)\b", re.IGNORECASE)
_CLOSING_START_RE = re.compile(r"^[,;)\]]")
_CONTACT_LEFT_RE = re.compile(
    r"\S@\S|https?:|www\.|\b(?:e-?mail|fax|facsimile|telecopy|telecopier|tel|telephone|phone|attn|attention)\b\.?\s*(?:no\.?\s*)?:"
    r"|\d{3}[\s.\-)]+\d{3,4}[\s.\-]*\d{0,4}$",
    re.IGNORECASE,
)
_CONTACT_RIGHT_RE = re.compile(r"^(?:with\s+(?:a\s+|an\s+)?(?:required\s+|additional\s+)*cop(?:y|ies)\b|if\s+to\b|and\s+if\s+to\b)", re.IGNORECASE)
_SUSPENDED_WORDS = frozenset({"and", "or", "to", "through"})
# Words that leave a sentence open: a line that ends in one continues on the
# next page even when the next page starts with a capital ("of the" / "Company").
_OPEN_END_WORDS = frozenset(
    """
    a an the of to by for in on at with from into upon under over between among against
    and or nor but that which who whom whose such any all each every no this these those
    its their his her our your is are was were be been being shall will would may might
    must can could should has have had not as than
    """.split()
)
_CAPTION_MAX_WORDS = 10
_MIN_KEY = 8
_KEY_WIDTHS = (32, 16, _MIN_KEY)
_REPEAT_LIMIT = 1
_OPEN_END_MIN_WORDS = 4
_PROSE_WORDS = 12


@dataclass(frozen=True)
class JoinPage:
    text: str
    raw: str | None = None
    is_html: bool = False
    page_order: int | None = None


@dataclass
class PageJoinResult:
    page_texts: list[str]
    joins: list[dict[str, object]] = field(default_factory=list)
    skipped: Counter[str] = field(default_factory=Counter)
    rejected: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class _Edge:
    """The last or first content line of a page."""

    line: str
    start: int  # offset of the line in the page text
    end: int


def _last_line(text: str) -> tuple[_Edge | None, str | None]:
    """The last content line and, when it is not joinable, why not."""
    tail_start = _TAIL_MARKERS_RE.search(text).start()
    body = text[:tail_start]
    last = None
    for m in _LINE_RE.finditer(body):
        last = m
    if last is None:
        return None, "empty"
    line = last.group(0).rstrip()
    edge = _Edge(line, last.start(), last.start() + len(line))
    # A page number with text above it is the foot of the page: what follows it is furniture.
    # One at the top of the page is a running head.
    marker = None
    for m in _PAGE_MARKER_RE.finditer(body):
        marker = m
    if marker is not None and _LINE_CONTENT_RE.search(body[: marker.start()]):
        return edge, "left_after_page_number"
    return edge, None


def _first_line(text: str) -> _Edge | None:
    """The text a page opens with, up to the first heading or marker on its line."""
    for m in _LINE_RE.finditer(text):
        line = m.group(0)
        lead = _LEADING_MARKERS_RE.match(line)
        start = lead.end() if lead else 0
        if start == len(line):
            continue
        markup = _MARKUP_RE.search(line, start)
        end = markup.start() if markup else len(line)
        return _Edge(line[start:end], m.start() + start, m.start() + end)
    return None


def _heading_open(text: str) -> bool:
    last = None
    for m in _OPEN_HEADING_RE.finditer(text):
        last = m
    return last is not None and last.group(1) == ""


def _plain(line: str) -> str:
    return _MARKUP_RE.sub(" ", line).strip()


def _edge_key(line: str) -> str:
    return re.sub(r"[\W\d_]+", " ", _plain(line)).strip().casefold()


def _is_caption(line: str) -> bool:
    """A short line of capitalized words: a caption, not a cut sentence."""
    words = [w for w in re.split(r"\s+", line.strip()) if any(c.isalpha() for c in w)]
    if not words:
        return True
    if all(c.isupper() for w in words for c in w if c.isalpha()) and len(words) <= 2 * _CAPTION_MAX_WORDS:
        return True
    if len(words) > _CAPTION_MAX_WORDS:
        return False
    small = {"of", "the", "and", "or", "to", "for", "in", "on", "by", "with", "a", "an"}
    content = [w for w in words if w.casefold() not in small]
    return bool(content) and all(next(c for c in w if c.isalpha()).isupper() for w in content)


def _left_problem(left: str, heading_open: bool) -> str | None:
    if heading_open:
        return "left_heading"
    tail = left
    last = None
    for m in _MARKUP_RE.finditer(left):
        last = m
    if last is not None:
        tail = left[last.end() :]
        if last.group(0) not in ("</section>", "</article>", "</page>"):
            return "left_markup"
    if not any(c.isalpha() for c in tail):
        return "left_heading" if last is not None and last.group(0) != "</page>" else "left_no_letters"
    tail = tail.strip()
    if _SENTENCE_END_RE.search(tail):
        return "left_ends_sentence"
    if _DASH_END_RE.search(tail):
        return "left_dash"
    if _LIST_ITEM_END_RE.search(tail):
        return "left_list_item_end"
    if _CONTACT_LEFT_RE.search(tail[-80:]):
        return "left_contact"
    if _is_caption(tail) and (last is None or last.group(0) == "</page>"):
        return "left_caption"
    return None


def _is_all_caps_word(word: str) -> bool:
    letters = [c for c in word if c.isalpha()]
    return len(letters) >= 2 and all(c.isupper() for c in letters)


def _open_end(left: str) -> bool:
    words = left.rstrip().rsplit(None, 1)
    return bool(words) and words[-1].casefold() in _OPEN_END_WORDS


def _right_problem(right: str, left: str) -> str | None:
    """Why the right line cannot continue the left one, or None."""
    stripped = right.strip()
    if not stripped:
        return "right_heading"
    if _LABEL_RE.match(stripped):
        return "right_label"
    if _CONTACT_RIGHT_RE.match(stripped):
        return "right_contact"
    if _CLOSING_START_RE.match(stripped):
        return None
    if _ENUMERATOR_RE.match(stripped):
        return "right_enumerator"
    first = next((c for c in stripped if c.isalnum()), "")
    if not first:
        return "right_no_letters"
    if stripped[0].islower():
        return None
    if not _open_end(left) or len(_plain(left).split()) < _OPEN_END_MIN_WORDS:
        return "right_not_continuation"
    if _is_all_caps_word(stripped.split()[0]) and not _is_all_caps_word(left.rsplit(None, 1)[-1]):
        return "right_caps"
    return None


# --- what the filed page prints --------------------------------------------------------------------

_RAW_TOKEN_RE = re.compile(
    r"<!--.*?-->|<(script|style|head|title)\b.*?</\1\s*>|<(/?)([A-Za-z][A-Za-z0-9:-]*)((?:[^>\"']|\"[^\"]*\"|'[^']*')*)>",
    re.DOTALL | re.IGNORECASE,
)
_VOID_TAGS = frozenset({"br", "hr", "img", "meta", "input", "link", "area", "base", "col", "wbr"})
_TABLE_TAGS = frozenset({"table", "tbody", "thead", "tfoot", "tr", "td", "th", "caption"})
_LIST_TAGS = frozenset({"li", "ul", "ol", "dl", "dt", "dd"})
_HEADING_TAGS = frozenset({"h1", "h2", "h3", "h4", "h5", "h6"})
_ALIGN_RE = re.compile(r"text-align\s*:\s*([a-z-]+)|\balign\s*=\s*[\"']?([a-z]+)", re.IGNORECASE)
_HIDDEN_RE = re.compile(r"display\s*:\s*none|visibility\s*:\s*(?:hidden|collapse)|(?:^|\s)hidden(?:\s*=|\s|/?$)", re.IGNORECASE)
_FONT_SIZE_RE = re.compile(r"font-size\s*:\s*([\d.]+\s*[a-z%]*)", re.IGNORECASE)
_FONT_ATTR_RE = re.compile(r"\bsize\s*=\s*[\"']?([+-]?\d+)", re.IGNORECASE)

_TABLE, _LIST, _HEADING, _CENTER, _HIDDEN = 1, 2, 4, 8, 16
_FLAG_NAMES = ((_TABLE, "table"), (_LIST, "list"), (_HEADING, "heading"), (_CENTER, "centered"), (_HIDDEN, "hidden"))


@dataclass
class RawPage:
    """The letters and digits of a filed page, with the layout around each run."""

    alnum: str
    starts: list[int]
    flags: list[int]
    sizes: list[str | None]

    def run_at(self, pos: int) -> int:
        return bisect_right(self.starts, pos) - 1


def _normal_size(value: str) -> str:
    m = re.fullmatch(r"([\d.]+)\s*([a-z%]*)", value.strip().lower())
    return f"{float(m.group(1)):g}{m.group(2)}" if m and m.group(1).count(".") < 2 else value


def read_raw_page(raw: str) -> RawPage:
    chars: list[str] = []
    starts: list[int] = []
    flags: list[int] = []
    sizes: list[str | None] = []
    stack: list[tuple[str, int, str | None]] = []
    total = 0
    cut_flags = 0  # layout a page cut mid-element closes without opening
    cut_until = 0

    def add_text(chunk: str) -> None:
        nonlocal total
        letters = [c for c in html.unescape(chunk) if c.isalnum()]
        if not letters:
            return
        top = stack[-1] if stack else ("", 0, None)
        starts.append(total)
        flags.append(top[1])
        sizes.append(top[2])
        chars.extend(letters)
        total += len(letters)

    pos = 0
    for m in _RAW_TOKEN_RE.finditer(raw):
        add_text(raw[pos : m.start()])
        pos = m.end()
        name = (m.group(3) or "").lower()
        if not name:
            continue
        attrs = m.group(4) or ""
        if m.group(2):
            depth = next((i for i in range(len(stack) - 1, -1, -1) if stack[i][0] == name), -1)
            if depth >= 0:
                del stack[depth:]
            else:
                bit = _TABLE if name in _TABLE_TAGS else _LIST if name in _LIST_TAGS else _HEADING if name in _HEADING_TAGS else 0
                if bit:
                    cut_flags |= bit
                    cut_until = total
            continue
        if name in _VOID_TAGS or attrs.rstrip().endswith("/"):
            continue
        parent_flags, parent_size = (stack[-1][1], stack[-1][2]) if stack else (0, None)
        bit = parent_flags
        if name in _TABLE_TAGS:
            bit |= _TABLE
        elif name in _LIST_TAGS:
            bit |= _LIST
        elif name in _HEADING_TAGS:
            bit |= _HEADING
        elif name == "center":
            bit |= _CENTER
        align = _ALIGN_RE.search(attrs)
        if align:
            # The nearest alignment wins: a justified paragraph inside a centered wrapper is not centered.
            if (align.group(1) or align.group(2)).lower() == "center":
                bit |= _CENTER
            else:
                bit &= ~_CENTER
        if _HIDDEN_RE.search(attrs):
            bit |= _HIDDEN
        size = _FONT_SIZE_RE.search(attrs)
        font = _FONT_ATTR_RE.search(attrs) if name == "font" else None
        value = _normal_size(size.group(1)) if size else "font" + font.group(1) if font else parent_size
        stack.append((name, bit, value))
    add_text(raw[pos:])
    if cut_flags:
        for i, start in enumerate(starts):
            if start < cut_until:
                flags[i] |= cut_flags
    return RawPage("".join(chars), starts, flags, sizes)


def _alnum(text: str) -> str:
    return "".join(c for c in text if c.isalnum())


def _locate(page: RawPage, key: str, *, last: bool) -> int | None:
    """Index in ``page`` of the run that holds the edge of ``key``."""
    for width in _KEY_WIDTHS:
        if len(key) < width and width != _MIN_KEY:
            continue
        piece = key[-width:] if last else key[:width]
        if len(piece) < _MIN_KEY and len(piece) != len(key):
            continue
        at = page.alnum.rfind(piece) if last else page.alnum.find(piece)
        if at >= 0:
            return page.run_at(at + len(piece) - 1 if last else at)
    return None


def _raw_problem(left: str, right: str, left_raw: str, right_raw: str) -> str | None:
    left_page, right_page = read_raw_page(left_raw), read_raw_page(right_raw)
    left_key, right_key = _alnum(_plain(left)), _alnum(_plain(right))
    if len(left_key) < 2 or len(right_key) < 2:
        return "raw_unlocated"
    left_run = _locate(left_page, left_key, last=True)
    right_run = _locate(right_page, right_key, last=False)
    if left_run is None or right_run is None or left_run < 0 or right_run < 0:
        return "raw_unlocated"
    flags = 0
    for line, side_flags in ((left, left_page.flags[left_run]), (right, right_page.flags[right_run])):
        flags |= side_flags & (_TABLE | _LIST | _HIDDEN)
        if len(_plain(line).split()) < _PROSE_WORDS:
            flags |= side_flags & (_HEADING | _CENTER)
    for bit, name in _FLAG_NAMES:
        if flags & bit:
            return f"raw_{name}"
    left_size, right_size = left_page.sizes[left_run], right_page.sizes[right_run]
    if left_size is not None and right_size is not None and left_size != right_size:
        return "raw_font_size"
    return None


# --- the join --------------------------------------------------------------------------------------


def _hyphen_join(left: str, right: str, words: Counter[str]) -> tuple[str, str]:
    """(left text, joiner) for a left line that ends in a word hyphen."""
    m = _WORD_HYPHEN_END_RE.search(left)
    word = _RIGHT_WORD_RE.match(right)
    if word is None:
        return left, ""
    tail = word.group(0)
    if tail.casefold() in _SUSPENDED_WORDS:
        return left, " "
    if m is not None and words[(m.group(1) + tail).casefold()] > words[f"{m.group(1)}-{tail}".casefold()]:
        return left[:-1], ""
    return left, ""


def agreement_words(texts: list[str]) -> Counter[str]:
    """How often the agreement's pages print each word whole, hyphenated ones as one word."""
    words: Counter[str] = Counter()
    for text in texts:
        words.update(w.casefold() for w in _WORD_TOKEN_RE.findall(text))
    return words


def join_across_pages(pages: list[JoinPage], *, collect: bool = False) -> PageJoinResult:
    """Join the paragraphs that page breaks cut, across the pages of one zone in page order.

    ``collect`` also lists each pair of lines that was not joined, with the reason."""
    result = PageJoinResult(page_texts=[p.text for p in pages])

    def skip(reason: str, i: int, left: _Edge | None = None, right: _Edge | None = None) -> None:
        result.skipped[reason] += 1
        if collect:
            result.rejected.append(
                {
                    "reason": reason,
                    "page_index": i,
                    "left": left.line[-120:] if left else "",
                    "right": right.line[:120] if right else "",
                }
            )

    if len(pages) < 2:
        return result
    lasts = [_last_line(p.text) for p in pages]
    firsts = [_first_line(p.text) for p in pages]
    last_keys = Counter(k for k in (_edge_key(e.line) if e else "" for e, _ in lasts) if k)
    first_keys = Counter(k for k in (_edge_key(e.line) if e else "" for e in firsts) if k)
    words: Counter[str] | None = None
    # Decide on the page texts as they stand, then apply from the last page back, so a
    # page that is one running paragraph chains through.
    decided: list[tuple[int, _Edge, _Edge, str, str, str]] = []
    for i in range(len(pages) - 1):
        left_page, right_page = pages[i], pages[i + 1]
        if left_page.page_order is None or right_page.page_order is None:
            skip("pages_unordered", i)
            continue
        if right_page.page_order != left_page.page_order + 1:
            skip("pages_not_adjacent", i)
            continue
        left_edge, why = lasts[i]
        right_edge = firsts[i + 1]
        if left_edge is None or right_edge is None:
            skip("empty", i)
            continue
        if why is not None:
            skip(why, i, left_edge, right_edge)
            continue
        why = _left_problem(left_edge.line, _heading_open(left_page.text[: left_edge.end]))
        if why is not None:
            skip(why, i, left_edge, right_edge)
            continue
        left_plain = left_edge.line.rstrip()
        right_plain = right_edge.line.strip()
        why = _right_problem(right_plain, left_plain)
        if why is not None:
            skip(why, i, left_edge, right_edge)
            continue
        if last_keys[_edge_key(left_edge.line)] > _REPEAT_LIMIT or first_keys[_edge_key(right_edge.line)] > _REPEAT_LIMIT:
            skip("repeated_line", i, left_edge, right_edge)
            continue
        if left_page.is_html and right_page.is_html and left_page.raw and right_page.raw:
            why = _raw_problem(left_edge.line, right_edge.line, left_page.raw, right_page.raw)
            if why is not None:
                skip(why, i, left_edge, right_edge)
                continue
        elif left_page.is_html or right_page.is_html:
            skip("raw_missing", i, left_edge, right_edge)
            continue
        new_left, joiner, kind = left_plain, " ", "plain"
        if left_plain.endswith("-"):
            if not right_plain[0].islower():
                skip("hyphen_upper", i, left_edge, right_edge)
                continue
            if words is None:
                words = agreement_words([p.text for p in pages])
            new_left, joiner = _hyphen_join(left_plain, right_plain, words)
            kind = "hyphen_closed" if new_left != left_plain else "hyphen_kept" if joiner == "" else "hyphen_suspended"
        elif _CLOSING_START_RE.match(right_plain):
            joiner, kind = "", "closing"
        elif not right_plain[0].islower():
            kind = "open_end"
        decided.append((i, left_edge, right_edge, new_left, joiner, kind))

    texts = result.page_texts
    for i, left_edge, right_edge, new_left, joiner, kind in reversed(decided):
        right_text = texts[i + 1]
        # The page after may already have joined onto this line's end.
        end = right_text.find("\n", right_edge.start)
        end = end if end >= 0 else len(right_text)
        markup = _MARKUP_RE.search(right_text, right_edge.start, end)
        cut = markup.start() if markup else end
        moved = right_text[right_edge.start : cut].strip()
        rest = right_text[cut:].lstrip(" \t") if markup else right_text[cut:].lstrip("\n")
        texts[i + 1] = right_text[: right_edge.start] + rest
        left_text = texts[i]
        texts[i] = left_text[: left_edge.start] + new_left + joiner + moved + left_text[left_edge.end :]
        result.joins.append(
            {"page_index": i, "kind": kind, "joiner": joiner, "left": left_edge.line[-120:], "right": right_edge.line[:120]}
        )
    result.joins.reverse()
    return result
