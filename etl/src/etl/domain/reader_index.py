# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false
"""Reader index: outline, defined terms and cross-references for one agreement XML.

Pure module, no DB code: an XML string goes in and a JSON-ready payload comes
out. See docs/plans/smart-reader.md (Section 6) for the design and the anchor
contract. In short:

- An anchor is ``{"s": container, "t": ordinal}``. ``container`` is the uuid of
  the nearest enclosing ``<section>`` or ``<article>``, or the region tag
  (``frontMatter``, ``body``, ``backMatter``, ...) for text outside them.
  ``t`` counts every ``<text>`` element in that container in document order,
  including text inside ``<definition>``.
- Spans are ``[start, end)`` in UTF-16 code units over the entity-decoded text
  of that ``<text>`` element, the string the React reader renders.

Rule of thumb throughout: a wrong link is worse than no link.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any

ENRICHER_VERSION = "0.10.0"
PAYLOAD_SCHEMA = 1

# Regions whose text is indexed for term uses and cross-references. The table of
# contents and signature pages only produce noise (page numbers, party names).
_INDEXED_REGIONS = frozenset({"frontMatter", "body", "backMatter"})
_NON_TEXT_REGIONS = frozenset({"metadata"})
_CONTAINER_TAGS = frozenset({"section", "article"})
# The reader replaces these with a space before rendering; all are one UTF-16 unit.
_NON_BREAKING_RE = re.compile("[   ⁠﻿]")

_ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}


def _roman_to_int(s: str) -> int | None:
    s = s.lower()
    if not s or any(ch not in _ROMAN for ch in s):
        return None
    total, prev = 0, 0
    for ch in reversed(s):
        v = _ROMAN[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


# I-XXXIX: a bare "L" or "C" leading a dotted number is a French code article
# ("Article L.233-3") or a lettered section ("Section C.6"), not a Roman article
_ROMAN_NUM = r"(?:X{0,3}(?:IX|IV|V?I{1,3}|V)|X{1,3})"
_ROMAN_LEAD = re.compile(r"^(" + _ROMAN_NUM + r")(?=\.\d)")


def _deroman(num: str) -> str:
    """'VIII.2(e)' is '8.2(e)': a Roman article number leading a dotted section number."""
    m = _ROMAN_LEAD.match(num)
    n = _roman_to_int(m.group(1)) if m else None
    return f"{n}{num[m.end():]}" if m and n else num


# ---------------------------------------------------------------- text nodes


@dataclass
class _Text:
    s: str  # container: section/article uuid or region tag
    t: int  # ordinal within the container
    region: str
    text: str  # entity-decoded, non-breaking spaces replaced (same length as rendered)
    section_key: str | None  # normalized number of the enclosing section, if any
    section_el: ET.Element | None
    utf16: list[int] | None = None  # code point -> UTF-16 offset table, None when all BMP
    section_label: str | None = None  # the section number as printed ("VIII.2" for key "8.2")

    def off(self, i: int) -> int:
        return self.utf16[i] if self.utf16 is not None else i

    def span(self, a: int, b: int) -> list[int]:
        return [self.off(a), self.off(b)]


def _utf16_table(text: str) -> list[int] | None:
    if all(ord(ch) < 0x10000 for ch in text):
        return None
    table = [0]
    acc = 0
    for ch in text:
        acc += 2 if ord(ch) > 0xFFFF else 1
        table.append(acc)
    return table


def _collect_texts(root: ET.Element) -> tuple[list[_Text], dict[ET.Element, _Text]]:
    texts: list[_Text] = []
    by_el: dict[ET.Element, _Text] = {}
    counters: Counter[str] = Counter()
    roman = _roman_scheme(root)

    def walk(el: ET.Element, region: str, container: str, sec_el: ET.Element | None) -> None:
        for child in el:
            if child.tag == "text":
                raw = child.text or ""
                norm = _NON_BREAKING_RE.sub(" ", raw)
                key = label = None
                if sec_el is not None and (num := _section_num(sec_el.get("title") or "", roman)):
                    key, label = num
                node = _Text(container, counters[container], region, norm, key, sec_el, _utf16_table(norm), label)
                counters[container] += 1
                texts.append(node)
                by_el[child] = node
                continue
            uuid = child.get("uuid")
            if child.tag in _CONTAINER_TAGS and uuid:
                walk(child, region, uuid, child if child.tag == "section" else sec_el)
            else:
                walk(child, region, container, sec_el)

    for region_el in root:
        if region_el.tag in _NON_TEXT_REGIONS:
            continue
        walk(region_el, region_el.tag, region_el.tag, None)
    return texts, by_el


def text_hash(xml: str) -> str:
    """SHA-256 over the document's text nodes, stored so an in-place rewrite of one
    XML version can be detected (attribute-only rewrites such as taxonomy standardId
    leave it alone). The asset does not compare it yet: today only those
    attribute-only rewrites keep a version number."""
    root = ET.fromstring(xml)
    h = hashlib.sha256()
    for el in root.iter("text"):
        h.update((el.text or "").encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


# ---------------------------------------------------------------- outline

# "III. REPRESENTATIONS AND WARRANTIES" is article III too: a Roman number (I-XXXIX), a period and a capital,
# with no "ARTICLE" before it, still goes to group 1
ARTICLE_NUM = re.compile(
    r"^\s*(?:(?:ARTICLE|Article|PART|Part|SECTION|Section)\s+|(?=" + _ROMAN_NUM + r"\.\s+[A-Z]))([IVXLC]+|\d+|[A-Z])\b"
    + r"|^\s*(\d+)\s*[.:]?\s+\D",
)
# "Section VIII.2": some agreements number sections under their Roman article number
SECTION_NUM = re.compile(
    r"^\s*(?:Section|SECTION|Sec\.|§)?\s*(\d+[A-Z]?(?:\.\d+[A-Z]?)+|" + _ROMAN_NUM + r"(?:\.\d+[A-Z]?)+|\d+[A-Z]?)\.?(?=\s|$)",
)
# A paragraph that opens with one or more enumerators: "(a)", "(a)(i)", "(ii) ..."
ENUM_PREFIX = re.compile(r"^\s*((?:\([A-Za-z0-9]{1,6}\)\s*)+)")
ENUM_TOKEN = re.compile(r"\(([A-Za-z0-9]{1,6})\)")
# "5.16.1 ..." or "Section 5.16.1 ...": a paragraph numbered as a dotted subsection
DECIMAL_PARA = re.compile(r"^\s*(?:(?:Section|SECTION|Sec\.|§)\s*)?(\d+(?:\.\d+){2,})\.?\s")
# "A. No Solicitation." opening a paragraph: a lettered group of the section's "(a)" paragraphs,
# cited as "Section 5.3A(a)" or "Section 5.3(A)(a)"
GROUP_PARA = re.compile(r"^\s*([A-Z])\.\s+\S")
# "VIII.2.1 ...": a dotted paragraph under a Roman article number (Roman schemes only)
ROMAN_DECIMAL_PARA = re.compile(r"^\s*(?:(?:Section|SECTION|Sec\.|§)\s*)?(" + _ROMAN_NUM + r"(?:\.\d+){2,})\.?\s")
# "1.1 ..." opening a paragraph: a section the XML parse left untagged
SECTION_PARA = re.compile(r"^\s*\d+\.\d+[.\s]")
ATTACHMENT_HEADING = re.compile(
    r"^(EXHIBIT|Exhibit|ANNEX|Annex|SCHEDULE|Schedule|APPENDIX|Appendix)\s+([A-Z0-9][A-Z0-9.\-]*)\b"
)

# the enumerators of an attachment heading's number: "Annex 2.02(a)", "Schedule 1.1(b)(ii)"
ATTACHMENT_ENUMS = re.compile(r"(?:\([A-Za-z0-9]{1,6}\))+")

_LOWER_ROMANS = frozenset(
    "i ii iii iv v vi vii viii ix x xi xii xiii xiv xv xvi xvii xviii xix xx".split()
)


def _norm_num(num: str) -> str:
    """'7.02' and '7.2' collide; letters like 5A are kept."""
    parts = []
    for p in num.split("."):
        m = re.match(r"0*(\d+)([A-Z]?)$", p)
        parts.append(f"{int(m.group(1))}{m.group(2)}" if m else p)
    return ".".join(parts)


def _roman_scheme(root: ET.Element) -> bool:
    """Whether the body numbers its sections "VIII.2": at least three, and most of its sections. A stray
    "I.1" section (an embedded plan) or lettered articles ("C.1" among "A.1", "B.1") are not a scheme."""
    body = root.find("body")
    titles = [sec.get("title") or "" for sec in body.iter("section")] if body is not None else []
    roman = sum(1 for t in titles if (m := SECTION_NUM.match(t)) and _ROMAN_LEAD.match(m.group(1)))
    return roman >= 3 and roman * 2 > len(titles)


def _section_num(title: str, roman: bool) -> tuple[str, str] | None:
    """(key, label) of a section title's number, or None when it has none. "VIII.2" counts only in a
    Roman scheme."""
    m = SECTION_NUM.match(title)
    if not m or (not roman and _ROMAN_LEAD.match(m.group(1))):
        return None
    return _section_key(m.group(1))


def _section_key(printed: str) -> tuple[str, str]:
    """(key, label): "VIII.2" is keyed "8.2", like the article it sits in, and keeps its printed label."""
    key = _norm_num(_deroman(printed))
    return key, (printed if _ROMAN_LEAD.match(printed) else key)


def _enum_kind(tok: str) -> str:
    if tok.isdigit():
        return "digit"
    if tok.islower():
        return "roman" if tok in _LOWER_ROMANS else "alpha"
    return "ROMAN" if tok.lower() in _LOWER_ROMANS and len(tok) > 1 else "ALPHA"


def _enum_ordinal(tok: str) -> int:
    kind = _enum_kind(tok)
    if kind == "digit":
        return int(tok)
    if kind in ("roman", "ROMAN"):
        return _roman_to_int(tok.upper()) or 0
    return ord(tok.lower()[0]) * 10 + len(tok)  # (a) < (b) < (aa)


def _follows(prev: tuple[str, str], tok: str) -> bool:
    """Whether enumerator `tok` is the next sibling after `prev` (kind, token)."""
    pk, ptok = prev
    if pk == "alpha" and len(ptok) == 1 and len(tok) == 1 and tok.islower():
        return ord(tok) == ord(ptok) + 1  # "(h)" then "(i)" is alpha
    if _enum_kind(tok) != pk:
        return False
    step = 10 if pk in ("alpha", "ALPHA") else 1
    return _enum_ordinal(tok) == _enum_ordinal(ptok) + step


def _classify_tokens(tokens: list[str], prev_by_level: list[tuple[str, str]]) -> list[str]:
    """Pick alpha vs roman for ambiguous single letters (i, v, x) using the
    previous sibling at the same level."""
    kinds = []
    for depth, tok in enumerate(tokens):
        k = _enum_kind(tok)
        if tok in ("i", "v", "x") and depth < len(prev_by_level):
            pk, ptok = prev_by_level[depth]
            if pk == "alpha" and len(ptok) == 1 and ord(tok) == ord(ptok) + 1:
                k = "alpha"
        kinds.append(k)
    return kinds


@dataclass
class _Node:
    id: str
    kind: str  # article | section | sub | attachment
    label: str
    title: str
    anchor: dict[str, Any]
    children: list[_Node] = field(default_factory=list)
    inferred: bool = False  # the number comes from _positional_numbers, not from the printed heading

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "kind": self.kind, "label": self.label, "title": self.title, "anchor": self.anchor}
        if self.inferred:
            out["inferred"] = True  # only then, so the payloads of other agreements do not change
        if self.children:
            out["children"] = [c.to_json() for c in self.children]
        return out


@dataclass
class _Outline:
    roots: list[_Node] = field(default_factory=list)
    articles: dict[str, list[_Node]] = field(default_factory=dict)  # "4" -> nodes
    sections: dict[str, list[_Node]] = field(default_factory=dict)  # "7.2" -> nodes
    subs: dict[str, _Node] = field(default_factory=dict)  # "7.2(b)(ii)" -> node
    # Keys of numbered articles with no <section> inside: a flat agreement's
    # "Section 2." items, tagged as articles. Their "(a)" paragraphs are subsections.
    flat_articles: set[str] = field(default_factory=set)
    # (container, ordinal) of paragraphs that became dotted subsection nodes ("5.16.1 ...")
    numbered_paras: set[tuple[str, int]] = field(default_factory=set)
    attachments: dict[str, _Node] = field(default_factory=dict)  # "exhibit a" -> node
    groups: set[str] = field(default_factory=set)  # lettered group keys: "8.1(A)"
    roman_sections: bool = False  # most sections are numbered "VIII.2" (_roman_scheme): "Section VIII.2(e)" resolves
    para_labels: dict[str, str] = field(default_factory=dict)  # "8.2.1" -> printed "VIII.2.1"
    # the body writes "this Section 8" inside "ARTICLE VIII": its bare "Section N" can mean article N (_self_cites_article)
    self_cited_articles: bool = False
    # container -> (ordinal, depth, key) of its paragraph nodes, built on first use (_nearest_scope)
    para_index: dict[str, list[tuple[int, int, str]]] | None = None
    ids: set[str] = field(default_factory=set)
    unnumbered_sections: int = 0
    positional_sections: int = 0  # sections numbered by position (_positional_numbers)

    def unique_id(self, base: str) -> str:
        if base not in self.ids:
            self.ids.add(base)
            return base
        n = 2
        while f"{base}~{n}" in self.ids:
            n += 1
        self.ids.add(f"{base}~{n}")
        return f"{base}~{n}"


def _lettered_groups(texts: list[str]) -> bool:
    """Whether a section is split into lettered groups "A.", "B.", ... of its own "(a)" paragraphs.

    The groups run A, B, C from the first, at least two of them; "A." comes before any enumerated
    paragraph, and each group's list starts over at (a), (1) or (A), or at (i) unless that (i) only
    continues an (h) or runs on to (j). An "A." list nested inside "(n)" is not a group.
    """
    letters: list[str] = []
    expect_first = False  # the paragraph after a group header, if enumerated, opens a list
    last_alpha = ""  # "(i)" right after "(h)" continues the list rather than starting one
    opened_with_i = False  # a group whose list goes "(i)", "(j)" is an alpha list run on from the last group
    for txt in texts:
        gm = GROUP_PARA.match(txt)
        if gm:
            letters.append(gm.group(1))
            expect_first = True
            continue
        pm = ENUM_PREFIX.match(txt)
        if not pm:
            continue
        if not letters:
            return False
        tok = ENUM_TOKEN.findall(pm.group(1))[0]
        if expect_first and not (tok in ("a", "1", "A") or (tok == "i" and last_alpha != "h")):
            return False
        if opened_with_i and tok == "j":
            return False
        opened_with_i = expect_first and tok == "i"
        expect_first = False
        if len(tok) == 1 and tok.islower():
            last_alpha = tok
    return len(letters) >= 2 and "".join(letters) == "ABCDEFGHIJKLMNOPQRSTUVWXYZ"[: len(letters)]


# "this Section 4.2" inside a section, "this Article IV" inside an article: the text's own statement of the number
THIS_SECTION = re.compile(r"\bthis\s+(?:Section|SECTION)\s+(\d+[A-Z]?\.\d+[A-Z]?)(?![\d.]*\d)")
THIS_ARTICLE = re.compile(r"\bthis\s+(?:Article|ARTICLE)\s+([IVXLC]+|\d+)\b")


@dataclass
class _Positional:
    """Section numbers taken from the text and from position, for headings that print no usable number."""

    sections: dict[ET.Element, str] = field(default_factory=dict)  # section element -> "4.2"
    keys: set[str] = field(default_factory=set)
    # Article element -> its number, only when the article headings cannot be used (then no other article counts)
    articles: dict[ET.Element, str] = field(default_factory=dict)
    printed: set[ET.Element] = field(default_factory=set)  # sections whose printed number stands
    fits: int = 0  # fixed points that sit where counting from the fixed point before them puts them
    misfits: int = 0  # fixed points that do not


def _printed_article_key(title: str, expected: int) -> tuple[str, str] | None:
    """(key, raw) of an article title's printed number."""
    m = ARTICLE_NUM.match(title)
    raw = (m.group(1) or m.group(2)) if m else _glued_roman(title, expected)
    if not raw and (sm := SPELLED_ARTICLE.match(title)):
        # L3: "ARTICLE ONE", "Article Twelve" is article 1, article 12
        return str(_SPELLED.index(sm.group(1).lower()) + 1), sm.group(1)
    if not raw:
        return None
    as_int = _roman_to_int(raw) if not raw.isdigit() else None
    return (str(as_int) if as_int else raw).upper(), raw


def _majority(values: list[str]) -> str | None:
    """The value that more than half of `values` share."""
    if not values:
        return None
    top, n = Counter(values).most_common(1)[0]
    return top if n * 2 > len(values) else None


def _count_between(n: int, fixed: dict[int, int]) -> tuple[dict[int, int], set[int], int, int]:
    """Numbers for items 0..n-1 of which some have a fixed number. A run of other items is counted on from the
    fixed number before it (from 1 at the start); a run with a fixed number after it must end exactly one below
    that number, else the run gets nothing.

    Returns (number per item that got one, the items of the run after the last fixed number, fits, misfits).
    """
    out: dict[int, int] = dict(fixed)
    run: list[int] = []
    prev = fits = misfits = 0
    for i in range(n):
        if i not in fixed:
            run.append(i)
            continue
        if fixed[i] == prev + len(run) + 1:
            fits += 1
            out.update({r: prev + j for j, r in enumerate(run, 1)})
        else:
            misfits += 1
        prev, run = fixed[i], []
    out.update({r: prev + j for j, r in enumerate(run, 1)})
    return out, set(run), fits, misfits


def _positional_numbers(root: ET.Element, by_el: dict[ET.Element, _Text], roman: bool) -> _Positional:
    """Numbers for sections whose headings lost them ("Section i.. Merger.", "Closing.", ".1 Consideration", or
    "Section 1.1" printed in every article).

    Fixed points come first. A printed "N.M" in Article N that no other section prints stands. A section with no
    such heading that says "this Section N.M" about itself (in most of its "this Section" phrases, and as the only
    section to say so) is Section N.M. Between fixed points the sections are counted: the k-th after Section N.M
    is N.(M+k), but only when the count arrives exactly one below the next fixed point. Otherwise a section is
    missing or extra in the XML, and that stretch gets no numbers.

    The stretch after an article's last fixed point (or a whole article with none) has no such check. It is
    numbered only when the agreement shows that counting works in it: at least two self-stated numbers sit where
    the count puts them, and at most one in five does not.

    When the printed article numbers do not get that far (an article prints no usable number, or most printed
    section numbers do not fit the article they sit in), the same is tried with article numbers taken from the
    text: an article is the one its sections name ("this Section 4.2", "this Article IV"), a unique printed number
    stands otherwise, and the articles between are counted like sections. That numbering replaces the printed one
    only if the check above then passes.
    """
    pos = _Positional()
    body = root.find("body")
    if body is None:
        return pos
    arts: list[tuple[ET.Element, list[ET.Element]]] = []

    def collect(el: ET.Element, owner: list[ET.Element] | None) -> None:
        for child in el:
            if child.tag == "article":
                arts.append((child, []))
                collect(child, arts[-1][1])
            elif child.tag == "section":
                if owner is not None:
                    owner.append(child)
            elif child.tag != "text":
                collect(child, owner)

    collect(body, None)
    if not arts:
        return pos
    printed: dict[ET.Element, str] = {}
    count: Counter[str] = Counter()
    said: dict[ET.Element, str | None] = {}
    for _, secs in arts:
        for sec in secs:
            sn = _section_num(sec.get("title") or "", roman)
            if sn:
                printed[sec] = sn[0]
                count[sn[0]] += 1
            said[sec] = _majority(
                [_norm_num(m.group(1)) for t in sec.iter("text") if t in by_el for m in THIS_SECTION.finditer(by_el[t].text)]
            )
    said_count = Counter(v for v in said.values() if v)

    def plan(keys: list[str | None]) -> _Positional:
        out = _Positional()
        tail: list[tuple[ET.Element, str]] = []
        fixed_keys: set[str] = set()
        lone: set[ET.Element] = set()
        per_article: list[tuple[str, list[ET.Element], dict[int, int], set[int]]] = []
        for (_, secs), key in zip(arts, keys):
            if not key:
                continue
            own = re.compile(re.escape(key) + r"\.(\d+)")
            fixed: dict[int, int] = {}
            for i, sec in enumerate(secs):
                num = printed.get(sec)
                if num and count[num] == 1 and (m := own.fullmatch(num)):
                    fixed[i] = int(m.group(1))
                    out.printed.add(sec)
                    fixed_keys.add(num)
            per_article.append((key, secs, fixed, set()))
        for key, secs, fixed, stated in per_article:
            own = re.compile(re.escape(key) + r"\.(\d+)")
            for i, sec in enumerate(secs):
                num = said[sec]
                if i not in fixed and num and said_count[num] == 1 and num not in fixed_keys and (m := own.fullmatch(num)):
                    fixed[i] = int(m.group(1))
                    stated.add(i)
            # A stated number that counting agrees with from nowhere (not from the article's start, not from another
            # fixed point) is either left by an earlier draft ("this Section 3.5" in the sixth section) or follows a
            # section missing from the XML. Its section gets no number, but it still stops the count around it.
            for i in stated:
                if fixed[i] != i + 1 and not any(fixed[j] - fixed[i] == j - i for j in fixed if j != i):
                    lone.add(secs[i])
        for key, _, fixed, _ in per_article:
            fixed_keys.update(f"{key}.{n}" for n in fixed.values())
        for key, secs, fixed, stated in per_article:
            numbers, after_last, fits, misfits = _count_between(len(secs), fixed)
            if fixed and secs[max(fixed)] in lone:
                after_last = set()  # nothing is counted on from a stated number that nothing agrees with
                numbers = {i: n for i, n in numbers.items() if i <= max(fixed)}
            # Printed numbers that follow each other say nothing about counting; a stated or a counted one does
            checked = [i for i in fixed if i in stated or (i > 0 and i - 1 not in fixed)]
            good = sum(1 for i in checked if numbers.get(i - 1, 0) == fixed[i] - 1 or (i == 0 and fixed[i] == 1))
            out.fits += good
            out.misfits += len(checked) - good
            for i, n in numbers.items():
                sec, num = secs[i], f"{key}.{n}"
                if sec in out.printed or sec in lone or (i not in fixed and num in fixed_keys):
                    continue
                if i in after_last:
                    tail.append((sec, num))
                elif num not in out.keys:
                    out.sections[sec] = num
                    out.keys.add(num)
        if out.fits >= 2 and out.misfits * 5 <= out.fits:
            for sec, num in tail:
                if num not in out.keys:
                    out.sections[sec] = num
                    out.keys.add(num)
        return out

    by_print: list[str | None] = []
    seen: set[str] = set()
    last = 0
    for art, _ in arts:
        pk = _printed_article_key(art.get("title") or "", last + 1)
        key = pk[0] if pk and pk[0].isdigit() and pk[0] not in seen else None
        if key:
            seen.add(key)
            last = int(key)
        by_print.append(key)
    pos = plan(by_print)
    if (pos.fits >= 2 and pos.misfits * 5 <= pos.fits) or len(arts) < 3:
        return pos
    # Every article prints its own number and most printed section numbers stand in it: the printed numbers do
    # get that far. A few "this Section 6.6" phrases that count one article less are stale drafting, not lost numbers.
    if all(by_print) and 2 * len(pos.printed) > len(printed):
        return pos

    # Article numbers from the text
    stated_art: list[str | None] = []
    for art, secs in arts:
        named = [n.split(".")[0] for sec in secs if (n := said[sec])]
        for t in art.iter("text"):
            if t in by_el:
                for m in THIS_ARTICLE.finditer(by_el[t].text):
                    as_int = _roman_to_int(m.group(1)) if not m.group(1).isdigit() else int(m.group(1))
                    if as_int:
                        named.append(str(as_int))
        top = _majority(named)
        stated_art.append(top if top and top.isdigit() else None)
    stated_n = Counter(k for k in stated_art if k)
    printed_n = Counter(k for k in by_print if k)
    fixed_art: dict[int, int] = {}
    for i, key in enumerate(stated_art):
        if key and stated_n[key] == 1:
            fixed_art[i] = int(key)
    stated_pts = sorted(fixed_art.items())
    for i, key in enumerate(by_print):
        if i in fixed_art or not key or printed_n[key] != 1 or stated_n[key]:
            continue
        # A printed article number stands only where counting from the stated ones agrees with it ("1 ARTICLE",
        # "3 ARTICLE", "5 ARTICLE" printed for Articles I, II, III does not)
        before = [p for p in stated_pts if p[0] < i]
        after = [p for p in stated_pts if p[0] > i]
        if before and int(key) != before[-1][1] + i - before[-1][0]:
            continue
        if after and int(key) != after[0][1] - (after[0][0] - i):
            continue
        fixed_art[i] = int(key)
    numbers, after_last, _, _ = _count_between(len(arts), fixed_art)
    # As with sections: a stated article number that counting agrees with from nowhere ("this Section 14.1" left in
    # Article XIII by an earlier draft) numbers neither its article nor the articles counted on from it
    for i, _ in stated_pts:
        if fixed_art[i] != i + 1 and not any(fixed_art[j] - fixed_art[i] == j - i for j in fixed_art if j != i):
            numbers.pop(i, None)
            if i == max(fixed_art):
                for j in after_last:
                    numbers.pop(j, None)
    twice = {n for n, c in Counter(numbers.values()).items() if c > 1}
    by_text: list[str | None] = [str(numbers[i]) if i in numbers and numbers[i] not in twice else None for i in range(len(arts))]
    if by_text == by_print:
        return pos
    alt = plan(by_text)
    if alt.fits >= 2 and alt.misfits * 5 <= alt.fits:
        alt.articles = {art: key for (art, _), key in zip(arts, by_text) if key}
        return alt
    return pos


def _build_outline(root: ET.Element, by_el: dict[ET.Element, _Text]) -> _Outline:
    ol = _Outline()
    ol.roman_sections = _roman_scheme(root)
    pos = _positional_numbers(root, by_el, ol.roman_sections)
    ol.positional_sections = len(pos.sections)
    renumbered: dict[ET.Element, tuple[str, str] | None] = {}  # body sections, once any number is inferred

    def add_section(sec: ET.Element, parent: list[_Node]) -> None:
        stitle = sec.get("title") or ""
        uuid = sec.get("uuid")
        inferred = pos.sections.get(sec)
        sn = (inferred, inferred) if inferred else _section_num(stitle, ol.roman_sections)
        # A printed number that an inferred one took, or one printed under articles numbered by position
        if sn and not inferred and (sn[0] in pos.keys or (pos.articles and sec not in pos.printed)):
            sn = None
        if pos.sections:
            renumbered[sec] = sn
        if not sn or not uuid:
            ol.unnumbered_sections += 1
            return
        num, label = sn
        node = _Node(ol.unique_id(f"s{num}"), "section", label, stitle, {"s": uuid}, inferred=bool(inferred))
        parent.append(node)
        ol.sections.setdefault(num, []).append(node)
        if len(ol.sections[num]) > 1:
            return  # duplicate number after a bad parse: no subsection targets either
        add_subs(sec, num, uuid, node)
        printed = {num: label} | {k: v for k, v in ol.para_labels.items() if k.startswith(num + ".")}
        if any(k != v for k, v in printed.items()):
            prefixes = sorted(printed, key=len, reverse=True)
            relabel = list(node.children)
            while relabel:  # "8.2(e)" is shown as printed, "VIII.2(e)"; a printed "8.2.1" paragraph keeps its number
                sub = relabel.pop()
                for p in prefixes:
                    if sub.label == p or sub.label.startswith(p + "("):
                        sub.label = printed[p] + sub.label[len(p) :]
                        break
                relabel.extend(sub.children)

    def add_subs(el: ET.Element, num: str, uuid: str, node: _Node) -> None:
        """Subsection nodes under `node` from the numbered paragraphs inside `el`."""
        stack: list[tuple[str, str]] = []  # (kind, token) per level
        base = num  # the latest dotted paragraph ("5.16.1") owns the "(a)" paragraphs after it
        outer: list[tuple[str, str]] = []  # the section-level stack from before the first dotted paragraph
        foreign = False  # after a dotted paragraph numbered for another section: record nothing
        groups = _lettered_groups([tn.text for t_el in el.iter("text") if (tn := by_el.get(t_el))])
        for t_el in el.iter("text"):
            tn = by_el.get(t_el)
            if tn is None:
                continue
            txt = tn.text
            dm = DECIMAL_PARA.match(txt) or (ROMAN_DECIMAL_PARA.match(txt) if ol.roman_sections else None)
            # "Section 5.12.5 of the Company Disclosure Schedule sets forth ...": a schedule item, not paragraph 5.12.5
            if dm and DISCLOSURE.match(txt[dm.end() :]):
                dm = None
            if dm:
                key = _norm_num(_deroman(dm.group(1)))
                if key != dm.group(1):
                    ol.para_labels.setdefault(key, dm.group(1))
                foreign = not key.startswith(num + ".")
                if foreign:
                    continue
                if key not in ol.subs:
                    sub = _Node(ol.unique_id(f"s{key}"), "sub", key, txt.strip()[:80], {"s": uuid, "t": tn.t})
                    ol.subs[key] = sub
                    node.children.append(sub)
                    ol.numbered_paras.add((uuid, tn.t))
                if base == num:
                    outer = stack
                base, stack = key, []
                continue
            if foreign:
                continue
            gm = GROUP_PARA.match(txt) if groups and base == num else None
            if gm:
                key = f"{num}({gm.group(1)})"
                if key not in ol.subs:
                    sub = _Node(ol.unique_id(f"s{key}"), "sub", key, txt.strip()[:80], {"s": uuid, "t": tn.t})
                    ol.subs[key] = sub
                    node.children.append(sub)
                ol.groups.add(key)
                stack = [("GROUP", gm.group(1))]
                continue
            pm = ENUM_PREFIX.match(txt)
            if not pm:
                continue
            tokens = ENUM_TOKEN.findall(pm.group(1))
            # "(b)" after "(a) ...: 5.16.1 ... 5.16.2 ..." continues the section's own list
            og = 1 if outer and outer[0][0] == "GROUP" else 0
            if (
                base != num
                and len(outer) > og
                and _follows(outer[og], tokens[0])
                and not (stack and _follows(stack[0], tokens[0]))
            ):
                base, stack = num, outer[: og + 1]
            g = 1 if stack and stack[0][0] == "GROUP" else 0  # the group letter stays above every level
            kinds = _classify_tokens(tokens, stack[g:])
            level = next((i for i, (k, _) in enumerate(stack) if i >= g and k == kinds[0]), len(stack))
            stack = stack[:level]
            for k, tok in zip(kinds, tokens):
                stack.append((k, tok))
                key = base + "".join(f"({tk})" for _, tk in stack)
                if key in ol.subs:
                    continue
                parent_key = key[: key.rfind("(")]
                parent_node = ol.subs.get(parent_key, node)
                sub = _Node(ol.unique_id(f"s{key}"), "sub", key, txt.strip()[:80], {"s": uuid, "t": tn.t})
                ol.subs[key] = sub
                parent_node.children.append(sub)

    last_article: list[int] = [0]

    def walk(el: ET.Element, parent: list[_Node]) -> None:
        for child in el:
            if child.tag == "article":
                title = child.get("title") or ""
                uuid = child.get("uuid")
                if pos.articles:
                    pk = (pos.articles[child], pos.articles[child]) if child in pos.articles else None
                else:
                    pk = _printed_article_key(title, last_article[0] + 1)
                if pk and uuid:
                    key, raw = pk
                    if key.isdigit():
                        last_article[0] = int(key)
                    node = _Node(ol.unique_id(f"a{key}"), "article", raw, title, {"s": uuid}, inferred=bool(pos.articles))
                    parent.append(node)
                    ol.articles.setdefault(key, []).append(node)
                    # flat: no <section> inside, and no "1.1 ..." paragraphs a parse left untagged
                    if (
                        key.isdigit()
                        and child.find(".//section") is None
                        and not any(
                            SECTION_PARA.match(by_el[t].text) for t in child.iter("text") if t in by_el
                        )
                    ):
                        ol.flat_articles.add(key)
                        if len(ol.articles[key]) == 1:
                            add_subs(child, key, uuid, node)
                    walk(child, node.children)
                else:
                    walk(child, parent)
            elif child.tag == "section":
                add_section(child, parent)
            elif child.tag != "text":
                walk(child, parent)

    body = root.find("body")
    if body is not None:
        walk(body, ol.roots)
    if renumbered:  # the texts of those sections: "clause (b)" and a term's "in Section ..." go by the new number
        for tn in by_el.values():
            if tn.section_el in renumbered:
                tn.section_key, tn.section_label = renumbered[tn.section_el] or (None, None)

    back = root.find("backMatter")
    if back is not None:
        for t_el in back.iter("text"):
            tn = by_el.get(t_el)
            if tn is None:
                continue
            txt = tn.text.strip()
            m = ATTACHMENT_HEADING.match(txt)
            if not m or len(txt) >= 120:
                continue
            num = m.group(2).lower().rstrip(".")
            key = f"{m.group(1).lower()} {num}"
            # "Annex 2.02(a)" keeps its enumerators, so "Annex 2.02(a)" and "Annex 2.02(b)" are two attachments;
            # a bare "Annex 2.02" still finds the first of them
            em = ATTACHMENT_ENUMS.match(txt, m.end(2)) if not m.group(2).endswith(".") else None
            enums = em.group(0) if em else ""
            full = key + enums.lower()
            if full in ol.attachments:
                continue
            node = _Node(ol.unique_id("x-" + full.replace(" ", "-")), "attachment", num.upper() + enums, txt, {"s": tn.s, "t": tn.t})
            ol.attachments[full] = node
            _ = ol.attachments.setdefault(key, node)
            ol.roots.append(node)
    # Subsections of a duplicated section or flat item can't say which copy they belong to
    dup = [n + "." for n, v in ol.sections.items() if len(v) > 1] + [n + "(" for n, v in ol.sections.items() if len(v) > 1]
    dup += [k + "(" for k in ol.flat_articles if len(ol.articles[k]) > 1]
    if dup:
        prefixes = tuple(dup)
        ol.subs = {k: v for k, v in ol.subs.items() if not k.startswith(prefixes)}
        ol.groups = {k for k in ol.groups if k in ol.subs}
    return ol


GLUED_ARTICLE = re.compile(r"^\s*(?:ARTICLE|Article)\s+([IVXLC]+)[A-Z]")


def _glued_roman(title: str, expected: int) -> str | None:
    """'ARTICLE VIISURVIVAL' (a missing space in the source): accept the roman
    prefix only when it is the next article number in sequence."""
    m = GLUED_ARTICLE.match(title)
    if not m:
        return None
    letters = m.group(1)
    for n in range(len(letters), 0, -1):
        if _roman_to_int(letters[:n]) == expected:
            return letters[:n]
    return None


def _container_label(tn: _Text) -> str:
    """Where a span lives, for the redaction teaser ("Defined in Section 1.1")."""
    if tn.section_key:
        return f"Section {tn.section_label or tn.section_key}"
    return tn.region


# ---------------------------------------------------------------- definitions

_Q_OPEN = "“\""
_Q_CLOSE = "”\""
_TERM = r"[A-Z0-9$][^“”\"]{0,80}?"
# Run declarations: a run of quoted terms ('"Tax" or "Taxes"', '"Liberty Media Affiliate," "Affiliate
# of Liberty Media" and similar phrases') before a keyword declares each of them. Quotes are paired
# first (_quoted_spans), then the run is walked back from the keyword, so matching stays linear.
# "Foo" means / shall mean / is defined as / as used herein, also after a short qualifier
# ('"Affiliate" of any Person means', '"Material Adverse Effect" on the Company means')
DEF_MEANS_KW = re.compile(r"(?:shall\s+)?(?:means?\b|is defined(?!\s+in\b)|as used herein|refers?\s+to\b)")
# ('"Beneficial Owner" and derivatives thereof, with respect to any Shares, mean', '"X" or "Y," or a
# variation thereof, means', '"KNOWLEDGE" with respect to (i) the Company means', '"X" as of any time shall mean')
_MEANS_GAP = re.compile(
    r"\s*(?:\([^)]*\)\s*)?(?:,\s*)?(?:(?:and|or)\s+(?:a\s+)?(?:derivatives?|variations?)\s+thereof\s*,?\s*)?(?:as\s+used\s+(?:in\s+this\s+Agreement|herein)\s*,?\s*)?"
    + r"(?:(?:when\s+used\s+)?(?:of|on|for|in relation to|in respect of|with respect to|on or with respect to|as applied to|as of)\s+(?:[^“”\".;:()]|\([a-z0-9]{1,4}\)){1,60}?,?\s+)?"
)
# '"Expenses" includes ...' opening a paragraph (or a definition item) is a definition
DEF_INCLUDES_KW = re.compile(r"(?:shall\s+)?includes?\b|shall\s+include\b")
_ITEM_START = re.compile(r"^\s*(?:\([A-Za-z0-9]{1,4}\)\s*|[A-Za-z0-9]{1,3}[.)]\s+)?$")
# "Controlled" and "Controlling" have correlative meanings; the terms "X" and "Y" have meanings
# that correspond to the foregoing
DEF_CORRELATIVE_KW = re.compile(
    r"(?:shall\s+)?(?:have|has)\s+(?:the\s+)?(?:correlative|correlated|corresponding|meanings?\s+(?:that\s+)?correspond)"
)
_CORRELATIVE_GAP = re.compile(r"\s*,?\s*")
_RUN_JOIN = re.compile(r"\s*,?\s*(?:(?:or|and|and/or)\s+)?")
_RUN_TAIL = re.compile(r",?\s+(?:and|or)\s+similar\s+(?:phrases|terms|expressions)(?:\s+used\s+in\s+this\s+Agreement)?")
_RUN_LOOKBACK = 120
_REFERENCES_TO = re.compile(r"\breferences?\b[^.;“”\"]{0,60}\bto\s+(?:the\s+)?$")


# "U.S.", "Vandalia Corp.": the final period belongs to the term
_ABBREV_END = re.compile(r"(?:\b[A-Z]\.[A-Z]\.|\b(?:Corp|Inc|Co|Ltd|Bros|L\.P|N\.A|S\.A)\.)$")


def _quoted_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of the inside of each quoted phrase. Curly quotes pair by direction; straight
    quotes pair in order. A trailing comma or period inside the quotes is not part of the phrase."""
    spans: list[tuple[int, int]] = []
    open_at: int | None = None
    for i, ch in enumerate(text):
        # "(c) ”Knowledge” of any person": a closing curly quote used to open, after a space and before a capital
        opens_right = ch == "”" and open_at is None and (i == 0 or text[i - 1] in " (\t") and text[i + 1 : i + 2].isupper()
        if ch == "“" or (ch == '"' and open_at is None) or opens_right:
            open_at = i + 1
        elif (ch == "”" or ch == '"') and open_at is not None:
            end = i
            while end > open_at and text[end - 1] in ",." and not _ABBREV_END.search(text, open_at, end):
                end -= 1
            if 0 < end - open_at <= 80:
                spans.append((open_at, end))
            open_at = None
    return spans


def _run_before(text: str, spans: list[tuple[int, int]], pos: int, gap: re.Pattern[str]) -> list[tuple[int, int]]:
    """The run of quoted spans that ends just before ``pos``, joined by commas, "or" and "and"."""
    i = next((k for k in range(len(spans) - 1, -1, -1) if spans[k][1] < pos), None)
    if i is None:
        return []
    # skip a trailing comma/period and the closing quote
    between = text[spans[i][1] : pos].lstrip(",.")[1:]
    m_tail = _RUN_TAIL.match(between)
    if m_tail:
        between = between[m_tail.end() :]
    if len(between) > _RUN_LOOKBACK or not gap.fullmatch(between):
        return []
    run = [spans[i]]
    while i > 0:
        prev = spans[i - 1]
        # text between the previous closing quote and this opening quote
        trail = text[prev[1] : prev[1] + 1]
        joint = text[prev[1] : spans[i][0] - 1].lstrip(",.")[1:]
        # a period inside the quotes ends the sentence; bare whitespace joins only '"X," "Y"'
        comma = trail == "," or joint.lstrip().startswith(",")
        if trail == "." or not _RUN_JOIN.fullmatch(joint) or (not joint.strip() and not comma):
            break
        run.insert(0, prev)
        i -= 1
    # a letter or digit right before the first opening quote means the quotes are mispaired
    first_open = run[0][0] - 1
    if first_open > 0 and (text[first_open - 1].isalnum()):
        return []
    return run
# (the "Company"), (each, a "Party"), (collectively, the "Sellers"), ("Parent")
DEF_PAREN = re.compile(
    rf"\(\s*(?:(?:together|collectively|individually|each|each of which|any such|such|hereinafter|herein)[^{_Q_OPEN})]*?\s+)?(?:the\s+|a\s+|an\s+|this\s+)?[{_Q_OPEN}]({_TERM})[{_Q_CLOSE}](?:\s*(?:or|and)\s+(?:the\s+|a\s+|an\s+)?[{_Q_OPEN}]({_TERM})[{_Q_CLOSE}])?\s*\)",
)
# (the entity so determined, the "EBITDA Expert"), (jointly, a "Parent Corporate Event")
DEF_PAREN_LEADIN = re.compile(rf"\(\s*[^()“”\"]{{1,80}}?,\s*(?:the|a|an)\s+[{_Q_OPEN}]({_TERM})[{_Q_CLOSE}]\s*\)")
# a quoted term inside a parenthetical that ends with a quoted term: "(the time ... being the
# "Effective Time")", "(the "Cash Consideration" and together with ..., the "Merger Consideration")"
_PAREN_TAIL_TERM = re.compile(rf"[{_Q_OPEN}][^{_Q_OPEN}{_Q_CLOSE}]{{1,80}}[,.]?[{_Q_CLOSE}]\s*$")
_PAREN_TERM = re.compile(
    rf"(?:^|(?<=[\s,(]))(?:(\w+)\s+)?(?:the|a|an|this)\s+[{_Q_OPEN}]({_TERM})[,.]?[{_Q_CLOSE}]|^\s*[{_Q_OPEN}]({_TERM})[,.]?[{_Q_CLOSE}]"
    + rf"|(?<=[\s,(])(?:collectively|together|individually)\s*,?\s+[{_Q_OPEN}]({_TERM})[,.]?[{_Q_CLOSE}]"
)
# '(a "Certificate", it being understood ...)': a parenthetical that opens with a declared term and runs on
DEF_PAREN_RUNON = re.compile(rf"\(\s*(?:the|a|an|each)\s+[{_Q_OPEN}]({_TERM})[,]?[{_Q_CLOSE}]\s*,\s*(?:it\s+being|which|provided|in\s+each\s+case|including|it\s+is)\b")
# "the X" after these words refers to a term ("as defined in the "Credit Agreement"", "together
# with the "Common Exchange Ratio"") rather than declaring it
_PAREN_REF_WORDS = frozenset("of in under to by with from for into on at pursuant per than".split())
_PAREN_MAX = 400
# referred to herein as "Compaq Broad Plans", being referred to herein as the "Merger Effective Time"
DEF_AS = re.compile(rf"\b(?:(?:referred\s+(?:to\s+)?|defined\s+)(?:herein\s+|hereinafter\s+|below\s+|in\s+this\s+Agreement\s+)?as|(?:herein\s+)?called(?:\s+herein)?)\s+(?:the\s+|a\s+|an\s+)?[{_Q_OPEN}]({_TERM})[,.]?[{_Q_CLOSE}]")
# '... referred to herein as the "Balance Sheet" and the date thereof as the "Balance Sheet Date"',
# '... referred to herein as a "Material Contract" and collectively, the "Material Contracts"'
_AS_MORE = re.compile(
    rf"\s*(?:,\s*)?(?:and|or)\s+(?:[^.;“”\"]{{0,80}}?\bas\s+|(?:collectively|together|individually|each)\s*,?\s*)(?:the\s+|a\s+|an\s+)?[{_Q_OPEN}]({_TERM})[,.]?[{_Q_CLOSE}]"
)
# the term "Material Adverse Effect," when used ..., means; not "the use of the term "X" in Section 6.2"
DEF_THE_TERM = re.compile(
    rf"\b(?i:the\s+term)\s+[{_Q_OPEN}]({_TERM})[,.]?[{_Q_CLOSE}](?:\s*or\s+[{_Q_OPEN}][^{_Q_OPEN}{_Q_CLOSE}]{{1,80}}[{_Q_CLOSE}])?\s*,?\s*(?:(?:when|as)\s+used\b[^.;]{{0,160}}?,?\s*)?(?:shall\s+)?(?:means?\b|refers?\s+to\b|includes?\b|include\b)"
)
# 'As used herein, a "Tax Protection Agreement" is an agreement ...', 'The "Parent Measurement Price"
# shall be the ...': a sentence that opens on a quoted term and says what it is
DEF_IS = re.compile(
    rf"(?:^|(?<=[.;:]\s)|(?<=herein,\s)|(?<=Agreement,\s))(?:The|the|A|An|a|an)\s+[{_Q_OPEN}]({_TERM})[,.]?[{_Q_CLOSE}]\s+(?:is|shall\s+be|will\s+be)\s+(?:an?|the|its)\s"
)
# Unquoted glossary entries, "Business Day: any day other than ...": counted only in a container with
# at least _COLON_GLOSSARY_MIN of them, so a stray "Note: ..." is never a term
COLON_ENTRY = re.compile(r"^\s*((?:[A-Z0-9][\w&.’'/\-]*|of|and|or|the|to|for|in|on|by)(?:\s+(?:[A-Z0-9][\w&.’'/\-]*|of|and|or|the|to|for|in|on|by)){0,8}):\s+\S")
_COLON_GLOSSARY_MIN = 5
_DEFS_TITLE = re.compile(r"defin|interpretation|glossary", re.IGNORECASE)
# words that name parts of a document: 'the terms "Article", "Section" and "Exhibit" refer to ...' sets a
# reading rule, and treating them as terms would mark every cross-reference as a term use
_DOC_PART_WORDS = frozenset(
    (
        "article articles section sections subsection subsections clause clauses paragraph paragraphs subparagraph "
        + "schedule schedules exhibit exhibits annex annexes appendix appendices recital recitals part parts preamble"
    ).split()
)
_NOTICE_LABELS = frozenset(
    "attention attn facsimile facsimile no fax fax no telephone telephone no phone tel e-mail email by name title address copy copies with a copy to telecopy telecopier date re note its".split()
    + ["facsimile no", "telephone no", "fax no", "with a copy to", "e-mail address"]
)
# All references to "$" or "dollar" are to U.S. dollars
DEF_REFERENCES = re.compile(
    rf"\breferences?\s+(?:[^“”\".;]{{0,40}}?\s+)?to\s+[{_Q_OPEN}]({_TERM})[,.]?[{_Q_CLOSE}](?:\s*,?\s*(?:or|and)\s+[{_Q_OPEN}]([^{_Q_OPEN}{_Q_CLOSE}]{{1,80}}?)[,.]?[{_Q_CLOSE}])?\s+(?:are|is|shall\s+be|means?)\b"
)
# "Foo" has the meaning set forth in Section 3.2 / is defined in Section 3.2
DEF_POINTER = re.compile(
    rf"[{_Q_OPEN}]({_TERM})[{_Q_CLOSE}]\s*(?:shall\s+)?(?:has|have|is)\s+(?:the\s+meanings?\s+(?:set\s+forth|given|ascribed|assigned|specified)\s+(?:to\s+(?:it|such term|them)\s+)?in|defined\s+in)\s+(Section|Article|Exhibit|Schedule|Annex)\s+([0-9A-Z.()]*[0-9A-Z)])",
)
# "Buyer" shall have the meaning set forth in the introductory paragraph / the Preamble / the Parent
# Charter: a pointer whose target is not a numbered provision
DEF_POINTER_OTHER = re.compile(
    rf"[{_Q_OPEN}]({_TERM})[,.]?[{_Q_CLOSE}]\s*(?:shall\s+)?(?:has|have)\s+the\s+meanings?\s+(?:set\s+forth|given|ascribed|assigned|specified|provided)"
    + r"(?:\s+(?:to\s+(?:it|such\s+term|them)|thereto|therefor))?\s+(?:in|on|by)\s+(?:the\s+)?([^.;“”\"]{1,60}?)\s*(?:hereto|hereof)?\s*[.;]?\s*$"
)


_SO_CALLED = re.compile(r"so[\s-]called\s+[“\"]$", re.IGNORECASE)
_NUMBER_LIKE = re.compile(r"[\d$.,%\s]+")
# a quoted section title after its number, 'Section 2.3 ("Capitalization, Etc.")', is not a declaration
_TITLE_AFTER_NUMBER = re.compile(r"\b(?:Sections?|Articles?)\s+[\dIVXLC][\w.()]*\s*\(\s*[“\"]$")


def _is_term(t: str, doc_part_ok: bool = False) -> bool:
    """A declared term starts with a capital, a digit or a symbol ("$"), or is a lowercase-first
    acronym with an internal capital ("cGMP"). Lowercase words ("dollars", "control") are not terms."""
    if not t or ";" in t or (t != "$" and _NUMBER_LIKE.fullmatch(t)) or (t.casefold() in _DOC_PART_WORDS and not doc_part_ok):
        return False
    if t[0].isupper() or t[0].isdigit() or t[0] == "$":
        return True
    return " " not in t and any(ch.isupper() for ch in t[1:])


def _paren_terms(text: str) -> list[tuple[str, int, int]]:
    """Terms declared inside parentheticals that end with a quoted term."""
    out: list[tuple[str, int, int]] = []
    stack: list[int] = []
    for i, ch in enumerate(text):
        if ch == "(":
            stack.append(i)
        elif ch == ")" and stack:
            start = stack.pop()
            body = text[start + 1 : i]
            if len(body) > _PAREN_MAX or not _PAREN_TAIL_TERM.search(body):
                continue
            for m in _PAREN_TERM.finditer(body):
                g = next(k for k in (2, 3, 4) if m.group(k))
                if m.group(1) and m.group(1).lower() in _PAREN_REF_WORDS:
                    continue
                out.append((m.group(g).strip(), start + 1 + m.start(g), start + 1 + m.end(g)))
    return out
# Glossary layout: 'Acquisition Proposal. "Acquisition Proposal" shall mean ...'. The
# leading heading repeats the term; it is not a use. f_xml.py does not tag these
# as <definition> yet (design decision 11.5), so the enricher catches them here.
GLOSSARY_ENTRY = re.compile(rf"^\s*([A-Z0-9][^“”\".]{{0,80}}?)\.\s+[{_Q_OPEN}]([^{_Q_CLOSE}]+)[{_Q_CLOSE}]")

_KIND_RANK = {"element": 0, "means": 1, "paren": 2, "pointer": 3}

# Single-word terms that are ordinary English words. They get low confidence, no
# sentence-initial matches, and the reader hides them by default (decision 11.7).
COMMON_WORD_TERMS = frozenset(
    """
    action actions affiliate affiliates agreement agreements assets business code contract
    contracts damages day days effect employee employees law laws liabilities liability losses
    material notice order orders party parties person persons plan plans property representatives
    securities shares sub subsidiary subsidiaries tax taxes term transactions year
    """.split()
)


@dataclass
class _Decl:
    term: str
    kind: str
    tn: _Text
    start: int
    end: int
    pointer: str | None = None
    doc_part_ok: bool = False


def _collect_declarations(root: ET.Element, texts: list[_Text], by_el: dict[ET.Element, _Text]) -> list[_Decl]:
    found: list[_Decl] = []
    for d in root.iter("definition"):
        first = d.find("text")
        tn = by_el.get(first) if first is not None else None
        if tn is None:
            continue
        m = re.match(rf"\s*[{_Q_OPEN}]([^{_Q_CLOSE}]+)[{_Q_CLOSE}]", tn.text)
        if m:
            # a glossary pointer wrapped in <definition> ("X" has the meaning set forth in ...) is
            # still a pointer
            pm = DEF_POINTER.match(tn.text.lstrip())
            om = DEF_POINTER_OTHER.match(tn.text.lstrip())
            if pm:
                found.append(_Decl(m.group(1).strip(), "pointer", tn, m.start(1), m.end(1), f"{pm.group(2)} {pm.group(3)}"))
            elif om:
                found.append(_Decl(m.group(1).strip(), "pointer", tn, m.start(1), m.end(1), om.group(2).strip()))
            else:
                found.append(_Decl(m.group(1).strip(), "element", tn, m.start(1), m.end(1)))
    for tn in texts:
        if tn.region not in _INDEXED_REGIONS:
            continue
        pointer_starts: set[int] = set()
        for m in DEF_POINTER.finditer(tn.text):
            pointer_starts.add(m.start())
            found.append(_Decl(m.group(1).strip(), "pointer", tn, m.start(1), m.end(1), f"{m.group(2)} {m.group(3)}"))
        off = len(tn.text) - len(tn.text.lstrip())
        om = DEF_POINTER_OTHER.match(tn.text, off)
        if om:
            pointer_starts.add(off)
            found.append(_Decl(om.group(1).strip(), "pointer", tn, om.start(1), om.end(1), om.group(2).strip()))
        # a run of quoted terms before "means" or "have correlative meanings" declares each of them
        spans = _quoted_spans(tn.text)
        for kw, gap in ((DEF_MEANS_KW, _MEANS_GAP), (DEF_CORRELATIVE_KW, _CORRELATIVE_GAP)):
            for m in kw.finditer(tn.text):
                run = _run_before(tn.text, spans, m.start(), gap)
                if run and run[0][0] - 1 in pointer_starts:
                    continue
                # 'references in this Agreement to the "X" and the "Y" shall refer to ...' is a reading rule
                if run and _REFERENCES_TO.search(tn.text, max(0, run[0][0] - 80), run[0][0] - 1):
                    continue
                for a, b in run:
                    # '"Exhibits" means the exhibits listed ...' defines a document part on purpose
                    own = len(run) == 1 and m.group(0).rstrip().endswith(("mean", "means")) and bool(_ITEM_START.match(tn.text[: a - 1]))
                    found.append(_Decl(tn.text[a:b].strip(), "means", tn, a, b, doc_part_ok=own))
            for m in DEF_INCLUDES_KW.finditer(tn.text):
                run = _run_before(tn.text, spans, m.start(), _MEANS_GAP)
                lead = tn.text[: run[0][0] - 1] if len(run) == 1 else ""
                if len(run) == 1 and (_ITEM_START.match(lead) or re.search(r"[.;:]\s+$", lead)):
                    a, b = run[0]
                    found.append(_Decl(tn.text[a:b].strip(), "means", tn, a, b))
        for m in DEF_PAREN.finditer(tn.text):
            for g in (1, 2):
                if m.group(g):
                    found.append(_Decl(m.group(g).strip(), "paren", tn, m.start(g), m.end(g)))
        for term, a, b in _paren_terms(tn.text):
            found.append(_Decl(term, "paren", tn, a, b))
        for m in DEF_PAREN_RUNON.finditer(tn.text):
            found.append(_Decl(m.group(1).strip(), "paren", tn, m.start(1), m.end(1)))
        for m in DEF_PAREN_LEADIN.finditer(tn.text):
            found.append(_Decl(m.group(1).strip(), "paren", tn, m.start(1), m.end(1)))
        for m in DEF_AS.finditer(tn.text):
            found.append(_Decl(m.group(1).strip(), "paren", tn, m.start(1), m.end(1)))
            pos = m.end()
            while (more := _AS_MORE.match(tn.text, pos)) is not None:
                found.append(_Decl(more.group(1).strip(), "paren", tn, more.start(1), more.end(1)))
                pos = more.end()
        for m in DEF_THE_TERM.finditer(tn.text):
            found.append(_Decl(m.group(1).strip(), "means", tn, m.start(1), m.end(1)))
        for m in DEF_IS.finditer(tn.text):
            found.append(_Decl(m.group(1).strip(), "means", tn, m.start(1), m.end(1)))
        for m in DEF_REFERENCES.finditer(tn.text):
            # only symbols ("$", "US$"): 'references to "Schedules" shall be to ...' and 'each reference
            # to "Chairman of the Board" shall be amended' are not definitions
            for g in (1, 2):
                if m.group(g) and not m.group(g)[0].isalpha():
                    found.append(_Decl(m.group(g).strip(), "means", tn, m.start(g), m.end(g)))
    colon: dict[str, list[tuple[_Text, re.Match[str]]]] = {}
    # only inside a body section titled as definitions: charter articles ("FIRST: The name ..."),
    # signature blocks ("By:", "Name:") and form fields use the same layout
    defs_sections = {el.get("uuid") for el in root.iter("section") if _DEFS_TITLE.search(el.get("title") or "")}
    for tn in texts:
        if tn.region == "body" and tn.s in defs_sections and (m := COLON_ENTRY.match(tn.text)) and m.group(1)[0].isupper():
            colon.setdefault(tn.s, []).append((tn, m))
    for entries in colon.values():
        # notice blocks ("Attention:", "Facsimile No.:", "By:") repeat a few labels with short values
        entries = [(tn, m) for tn, m in entries if m.group(1).strip().rstrip(".").casefold() not in _NOTICE_LABELS and len(tn.text) - m.end(1) >= 30]
        if len({m.group(1).strip() for _, m in entries}) >= _COLON_GLOSSARY_MIN:
            found.extend(_Decl(m.group(1).strip(), "means", tn, m.start(1), m.end(1)) for tn, m in entries)
    # one declaration per (term, text node), the strongest kind wins
    best: dict[tuple[str, str, int], _Decl] = {}
    for d in found:
        if d.kind == "paren" and _TITLE_AFTER_NUMBER.search(d.tn.text, max(0, d.start - 40), d.start):
            continue
        # 'the so-called "Big Four"' names something; it defines nothing
        if _SO_CALLED.search(d.tn.text, max(0, d.start - 12), d.start):
            continue
        # "$" is the one single-character term (rubric v1.2)
        if not (2 <= len(d.term) <= 80 or d.term == "$") or not _is_term(d.term, d.doc_part_ok):
            continue
        k = (d.term, d.tn.s, d.tn.t)
        if k not in best or _KIND_RANK[d.kind] < _KIND_RANK[best[k].kind]:
            best[k] = d
    decls = list(best.values())
    # pointers only stand in for terms with no real declaration
    real = {d.term for d in decls if d.kind != "pointer"}
    return [d for d in decls if d.kind != "pointer" or d.term not in real]


_SMALL_WORDS = frozenset("a an and as at by for in of on or the to with".split())


def _title_variants(term: str) -> list[str]:
    """Older filings declare "COMPANY" or "HSR ACT" and write "the Company" and "the HSR Act"."""
    if not (term.isupper() and len(term) > 3 and any(ch.isalpha() for ch in term)):
        return []
    words = term.split(" ")
    titled = [w if not w.isalpha() else (w.lower() if i and w.lower() in _SMALL_WORDS else w.capitalize()) for i, w in enumerate(words)]
    out = [" ".join(titled)]
    if len(words) > 1 and len(words[0]) <= 4:
        out.append(" ".join([words[0], *titled[1:]]))  # "HSR Act", "SEC Reports"
    return out


def _number_forms(term: str) -> list[str]:
    forms = []
    if term.endswith("y") and len(term) > 2 and term[-2] not in "aeiou":
        forms.append(term[:-1] + "ies")
    # a plural term ("Acquired Employees") is also used in the singular
    if term.endswith("ies") and len(term) > 4:
        forms.append(term[:-3] + "y")
    elif re.search(r"(?:x|ch|sh)es$", term) and len(term) > 5 and term[-3].islower():
        forms.append(term[:-2])  # "Taxes" -> "Tax"
    elif term.endswith("ses") and len(term) > 5 and term[-4].islower():
        forms.extend([term[:-1], term[:-2]])  # "Causes" -> "Cause", "Annual Bonuses" -> "Annual Bonus"
    elif term.endswith("s") and not term.endswith("ss") and len(term) > 3 and term[-2].islower():
        forms.append(term[:-1])
    return forms


def _term_forms(term: str) -> list[str]:
    forms = [term, *_number_forms(term)]
    for v in _title_variants(term):
        forms.extend([v, *_number_forms(v)])
    return forms


def _term_regex(forms: dict[str, int]) -> re.Pattern[str] | None:
    # a declared document part ("Exhibits" means ...) is still not matched: every "Schedule 2.2" would become a use
    alts = sorted((re.escape(f) for f in forms if len(f) >= 2 and f.casefold() not in _DOC_PART_WORDS), key=len, reverse=True)
    if not alts:
        return None
    # longest first; plural, "(s)" and possessive forms; case-sensitive
    return re.compile(
        r"(?<![\w“\"])(" + "|".join(alts) + r")(?:\(s\)|s|es)?(?:’s|'s|’|')?(?![\w”\"])"
    )


PROPER_NAME_ACT = re.compile(r"\s+Act\b")
PROPER_NAME_LAW = re.compile(r"\s+(?:Law|Code)\b")
PROPER_NAME_BEFORE = re.compile(r"\b(?:Trust|Bank|Investment)\s+$")
_CAP_WORD_BEFORE = re.compile(r"\b[A-Z][a-z]+\s+$")


def _in_proper_name(text: str, start: int, end: int) -> bool:
    before = text[max(0, start - 20) : start]
    if PROPER_NAME_ACT.match(text, end) or (" " not in text[start:end] and PROPER_NAME_BEFORE.search(before)):
        return True
    return bool(PROPER_NAME_LAW.match(text, end) and _CAP_WORD_BEFORE.search(before) and not _sentence_initial(text, start - len(before.split()[-1]) - 1))
_ENUM_ONLY = re.compile(r"^\s*(?:\([A-Za-z0-9]{1,6}\)\s*)*$")


def _sentence_initial(text: str, start: int) -> bool:
    prefix = text[:start].rstrip()
    return not prefix or prefix[-1] in ".:;!?" or bool(_ENUM_ONLY.match(prefix))


# ---------------------------------------------------------------- xrefs

STATUTE = re.compile(
    # "Section 4 para 1 and 2 AWG": German statutes cite paragraphs as "para"
    r"^\s*para\.?\s+\d|"
    + r"^\s*(?:\([^)]*\)\s*)*,?\s*(?:of|under|pursuant to)\s+(?:the\s+)?(?:[A-Z]{3,6}\b|Code|DGCL|Exchange Act|Securities Act|HSR Act|ERISA|Treasury Regulations?|Regulation|Rule|Delaware|General Corporation Law|Investment Company Act|Internal Revenue Code|(?:[A-Z][A-Za-z]+\s+){1,4}(?:Act|Law|Code|Statutes?)\b|(?:[A-Z][A-Za-z]+\s+){0,3}(?:1[89]|20)\d\d\s+Act\b|Title)",
)
DISCLOSURE = re.compile(
    # "of the Seller’s Disclosure Letter", "item 1 of the ...", "Disclosure ----- Schedule" across a page break;
    # "of Seller’s Disclosure Schedule": a possessive name with no article (the same opening as OTHER_DOC)
    r"^\s*(?:\([^)]*\)\s*)*(?:item\s+\d+\s+)?(?:of|to|in|under)\s+(?:(?:the|such|any|each)\s+|(?=[A-Z][\w\-]*[’']s\s))(?:[A-Z][\w\-’']*\s+){0,4}(?:(?i:disclosure[\s\-–—_]+(?:schedules?|letter|memorandum|statements?))|Schedules)\b"
    # "of the Company Schedule of Exceptions"; not "this Article VI to such Schedule of Exceptions" (its own article)
    + r"|^\s*(?:\([^)]*\)\s*)*(?:of|in|on)\s+(?:(?:the|such|any|each)\s+|(?=[A-Z][\w\-]*[’']s\s))(?:[A-Z][\w\-’']*\s+){0,4}Schedule\s+of\s+Exceptions\b",
)
DISCLOSURE_BEFORE = re.compile(r"(?i:disclosure)\s+(?:Schedules?|Letter|Memorandum)\s*$")
LOWER_HEAD_OTHER = re.compile(r"^[\d:.()A-Za-z]*\s*(?:\([^)]*\)\s*)*,?\s*(?:of|under|pursuant\s+to)\s+(?:the\s+|such\s+|its\s+|their\s+)?(?!this\b|Agreement\b|hereof\b)[A-Z]")
OCR_COMMA = re.compile(r",(\d{2})(?![\d,])")
# "Section 1. 6(c)", "Section 1 .6": a space typed or scanned into the section number. The second part
# is one whole number on the same line, and what follows must not make it a count ("Section 4. 15 days
# after ...") or a deeper number ("Section 4. 4.2 Covenants", where a heading follows the end of a sentence).
_ENUMS = r"((?:\s?\([A-Za-z0-9]{1,6}\))*)"
_NOT_A_COUNT = (
    r"(?![\dA-Za-z]|\.\d|\s+(?i:(?:business|calendar|trading)\s+)?"
    r"(?i:days?|months?|years?|hours?|weeks?|percent|shares?|million|billion|[ap]\.m\.|or\s+more)(?![A-Za-z])|\s*%)"
)
SPLIT_DOT = re.compile(r"(?:\.[^\S\n]+|[^\S\n]\.[^\S\n]*)(\d{1,3}[A-Z]?)" + _ENUMS + _NOT_A_COUNT)
# "Section 9. 0 3(a)", "Section 2.0 4": the space falls after the zero of 9.03 or 2.04. No section number
# ends in ".0", so a single digit after the space completes it.
ZERO_SPLIT = re.compile(r"[^\S\n](\d)" + _ENUMS + _NOT_A_COUNT)
# "-3" right after "1.368" (no space): the rest of a regulation number. "Sections 2.1-2.5" is still a range.
REG_HYPHEN = re.compile(r"[\-‑]\d+[A-Z]?(?:\s?\([A-Za-z0-9]{1,6}\))*(?![\d\-‑]|\.\d)")
COLON_NUMBER = re.compile(r":\d")
LEFTOVER_ITEMS = re.compile(r"^(?:\s*,?\s*(?:and/or|and|or)?\s*\([A-Za-z0-9]{1,6}\))+")
INCLUSIVE = re.compile(r"^\s*,?\s*inclusive\s*,?(?=\s*(?:of|to|in|under|hereof)\b)")
ATTACHMENT_TAIL = re.compile(r"^\s*(?:\([^)]*\)\s*)*(?:of|in|to)\s+(?:the\s+|this\s+)?(?:Schedules?|Part|Annex|Appendix|Exhibit)\s+[A-Z0-9IVX]")
# L5: "Section 2 of Exhibit B to the Escrow Agreement", "item 3 of Annex A to the Commitment Letter": an attachment
# of another document. "of Schedule 1 to the Company Disclosure Schedule" is the disclosure schedule.
OTHER_ATTACHMENT_TAIL = re.compile(
    r"^\s*(?:\([^)]*\)\s*)*(?:of|in|to)\s+(?:the\s+|this\s+)?(?:Exhibit|Schedule|Annex|Appendix|Part)\s+\S+\s+(?:of|to)\s+(?:the|such|its)\s+"
    + r"(?!(?:Merger\s+)?Agreement\b)(?:[A-Z][\w\-’']*\s+){0,5}(?:Letter|Agreement|Announcement|Plan|Indenture|Schedules?|Memorandum)\b"
)
SELF_DOC = re.compile(r"^\s*(?:\([^)]*\)\s*)*(?:of|to|in|under)\s+(?:the|this)\s+Agreement\b")
SELF_DOC_FROM_EXHIBIT = re.compile(r"^\s*(?:\([^)]*\)\s*)*(?:of|to|in|under)\s+(?:the|this)\s+(?:Merger\s+)?Agreement\b")
OTHER_DOC = re.compile(
    # "of the Company's Amended and Restated Certificate", "of Parent's Credit Agreement": a possessive name
    r"^\s*(?:\([^)]*\)\s*)*(?:of|to|in|under)\s+(?:(?:the|such|any|each)\s+|(?=[A-Z][\w\-]*[’']s\s))(?!Agreement\b)(?:[A-Z][\w\-]*(?:[’']s?)?\s+){0,5}(?:Agreement|Letter|Plan|Certificate|Charter|Bylaws|Indenture|Note|Schedule|Statement|Articles)\b"
    + r"|^\s*(?:\([^)]*\)\s*)*(?:of|to|in)\s+(?:the|its|their|such)\s+(?:[a-z]+\s+){0,3}(?:bylaws|by-laws|charter|certificate of incorporation|articles of incorporation|operating agreement)\b"
    + r"|^\s*(?:\([^)]*\)\s*)*(?:of|to|in)\s+(?:the|its|their|such)\s+(?:[A-Z][\w\-]*[’']s?\s+)?(?:Amended\s+and\s+Restated\s+)?(?:Articles|Certificate)\s+of\s+(?:Incorporation|Association|Organization)\b",
)
# Self markers: "this Section 6.3" and "Section 6.3 hereof" name the document's own section
THIS_BEFORE = re.compile(r"(?i:\bthis)\s+$")
HEREOF_TAIL = re.compile(r"^\s*(?:\([^)]*\)\s*)*,?\s*(?:hereof|herein|hereunder)\b")
# "Section 14a-13 of the Exchange Act": an Act named after the reference. A verb before it means the Act
# belongs to the rest of the sentence ("Section 6.3 shall not apply to filings under the Securities Act").
ACT_LATER = re.compile(r"\bof\s+(?:the\s+)?(?:[A-Z][\w .\-()]{2,60}|(?:1[89]|20)\d\d\s+)\bAct\b")
_VERB = re.compile(r"\b(?:shall|will|is|are|must|may|would|should|was|were|has|have|be|been)\b")


# Tails that keep a reference in this agreement: none at all, "hereof", "of this Agreement", "above", "below"
_SELF_TAIL = re.compile(
    r"^[\s,]*(?:\([^)]*\)\s*)*,?\s*(?:here(?:of|to|in|under)\b|(?:of|to|in|under)\s+(?:this|the)\s+Agreement\b)"
    + r"|(?i:^[\s,]*(?:above|below|hereinabove|hereinbelow|preceding|following)\b)"
)
# Tails that name something else: "of Revenue Procedure 2004-53", "of this Article", "to the Disclosure Letter",
# "thereof"; and OCR continuations of the number ("Section 7.l", "Section 7,05", "Section 1-3")
_OTHER_TAIL = re.compile(
    r"^[\s,]*(?:\([^)]*\)\s*)*,?\s*(?:(?:of|under|pursuant\s+to)\s|there(?:of|to|in|under)\b"
    + r"|(?:to|in)\s+(?:this\s+(?!Agreement\b)\w|(?:the|such|any|each|its|their)\s+[A-Z]|(?:Exhibit|Annex|Schedule|Appendix|Part)\s))"
    + r"|^(?:[.,]{1,2}\s?\d|\.l|-\d)"
    # "9 U.S.C., Section 1 et seq.": a statute; "Section 5f.103-1(c)": a letter glued to the number
    + r"|^[\s,]*et\.?\s+seq\b|^[A-Za-z]\.\d"
)


# "clause (iii) of paragraph 2 of Annex C of the Commitment Letter", "clause (ii) of the Credit Agreement":
# "clause (i) thereto": a clause of something other than this agreement's section
CLAUSE_OF_OTHER = re.compile(
    r"^\s*(?:\([^)]*\)\s*)*,?\s*there(?:of|to|in|under)\b"
    + r"|^[^.;]{0,80}?\b(?:of|under|in)\s+(?:(?:the|such|any|each|its|their)\s+(?!Agreement\b)(?:[A-Z][\w\-’']*\s+){0,5}"
    + r"(?:Agreement|Letter|Plan|Certificate|Charter|Bylaws|Indenture|Note|Schedule|Statement|Articles|Act|Code|Regulations?)\b"
    + r"|(?:paragraph|Annex|Exhibit|Schedule|Appendix)\s)"
)
# "clause (ii) of subparagraph (a) above", "clause (B) of paragraph (b)(i)": the clause sits under a named
# enumerator of the same section. Not when that enumerator belongs to something named after it ("of paragraph (a)
# of Exhibit B", "of clause (a) thereof"): then the clause gets no nearest-scope tie-break either.
_NAMED_ENUM = r"^\s*,?\s*(?:of|in|under)\s+(?:the\s+)?(?:sub-?)?(?:paragraph|clause|section)\s*((?:\([A-Za-z0-9]{1,6}\)\s?)++)"
NAMED_ENUM = re.compile(_NAMED_ENUM)
PARENT_ENUM = re.compile(_NAMED_ENUM + r"(?!\s*,?\s*(?:of|in|under|to|there(?:of|to|in|under))\b)")


def _plain_tail(tail: str) -> bool:
    return bool(_SELF_TAIL.match(tail)) or not _OTHER_TAIL.match(tail)


def _act_later(tail: str) -> bool:
    m = ACT_LATER.search(tail[:70])
    return m is not None and not _VERB.search(tail, 0, m.end())


# "Code Section 409A", "Treasury Regulations Section 1.368-2" (from the kirkland ingester)
AUTHORITY_BEFORE = re.compile(r"(?:\b(?:Code|Regulations?|Reg\.|ERISA|Act|Rules?|Treas\.|Treasury|Statute|Law|IRC|U\.S\.C\.|C\.F\.R\.|CFR|Regs\.)|§)\s*$")
SEC_FORM = re.compile(r"^(?:\d{2}[A-Z](?:-\d+)?|TO|13E-3)$")
# lowercase "section 3.5" and "article VI" are references too (rubric v1.4 lists the head words, not their case)
# "Section 6.4Section 6.3(a)": a head glued to the number before it still starts a reference
REF_HEAD = re.compile(r"(?<![A-Za-z])(Sections?|Articles?|SECTIONS?|ARTICLES?|sections?|articles?|Exhibits?|Schedules?|Annex(?:es)?|Appendix|Appendices|clauses?|paragraphs?)\s+(?=[\dIVXLC(A-Z])")
_SPELLED: tuple[str, ...] = tuple("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())
# "Article Three"
SPELLED_ITEM = re.compile(r"(" + "|".join(w.capitalize() + "|" + w.upper() for w in _SPELLED) + r")\b")
# "ARTICLE ONE": an article title that spells its number
SPELLED_ARTICLE = re.compile(r"^\s*(?:ARTICLE|Article)\s+" + SPELLED_ITEM.pattern)
# a letter suffix ("2.4A", "Section 409Aof") is not the start of a glued head ("6.4Section 6.3(a)")
_SUFFIX = r"(?:[A-Z](?!ections?\b|ECTIONS?\b|rticles?\b|RTICLES?\b|chedules?\b|xhibits?\b|nnex))"
GLUED_HEAD = re.compile(r"(?:Sections?|SECTIONS?|Articles?|ARTICLES?|Schedules?|Exhibits?|Annex)\b")
# L6: "K&S", "D&O", "U.S." after a head are names, not "Section K" or "Schedule U". A Roman item followed by ".X."
# is a reference ("Article IV.B.4", "Section I.E.1 above"), so only "&" stops it.
REF_ITEM = re.compile(r"(\d+" + _SUFFIX + r"?(?:-\d+)?(?:\.\d+" + _SUFFIX + r"?)*(?:\s?\([A-Za-z0-9]{1,6}\))*|[IVXLC]+\b(?!&)|[A-Z]\b(?!&|\.[A-Z]\.)(?:-\d+)?|\([A-Za-z0-9]{1,6}\)(?:\([A-Za-z0-9]{1,6}\))*)")
# "Section VIII.2(e)": tried first for section, article, clause and paragraph heads only ("Exhibit C.2" is Exhibit C)
ROMAN_ITEM = re.compile(r"(" + _ROMAN_NUM + r"(?:\.\d+" + _SUFFIX + r"?)+(?:\s?\([A-Za-z0-9]{1,6}\))*)")
# "Exhibit C-1", "Exhibit I-1": tried first for attachment heads, so the label is not Roman "C" plus a range "-1"
ATTACHMENT_ITEM = re.compile(r"([A-Z]-\d+)\b")
# "(other than Sections 1.1, 6.2 (a)), 6.6 (f)": a list can step out of the parenthetical it started in
RELATIVE_FOLLOW = re.compile(
    r"\s*(?:[,;.)]|$)|\s+(?:and|or|of|through|to|hereof|hereto|herein|thereof|above|below|shall|will|is|are|has|have|was|were|would|may|applies|apply)\b"
)
# "Sections 4.1 (Organization; Subsidiaries), 4.2 (Authority) and 4.7": a caption or other parenthetical
# (not an enumerator) may follow each item
REF_JOIN = re.compile(r"(?:\s*\((?![A-Za-z0-9]{1,6}\))[^()]{2,200}\))?(?:\s*(?:\)?,\s*(?:and|or)?\s*|\s+(?:and|or|through|to|and/or)\s+|\s*[\-–]\s*))")
# "clause (ii) of Section 4.2"; also "clauses (A)-(D) of, the foregoing Section 5.2(a)" and, after a list,
# "clauses (i) through (xiii) or clause (xvi) of Section 4.12(a)"
QUALIFIED = re.compile(
    r"(?:\s*,?\s*(?:and|or|and/or)\s+clauses?\s+\([A-Za-z0-9]{1,6}\)(?:\s*(?:through|to|[\-–])\s*\([A-Za-z0-9]{1,6}\))?)?"
    + r"\s*,?\s+(?:of|in)\s*,?\s+(?:the\s+foregoing\s+|this\s+)?[Ss]ection\s+((?:\d+|" + _ROMAN_NUM + r"(?=\.\d))[A-Z]?(?:\.[^\S\n]?\d+[A-Z]?)*(?:\([A-Za-z0-9]{1,6}\))*)")
SECTION_TARGET = re.compile(r"(\d+[A-Z]?(?:\.\d+[A-Z]?)*)((?:\([A-Za-z0-9]+\))*)$")

_HEAD_NORM = {"annexe": "annex", "appendice": "appendix"}
_IN_DOCUMENT_HEADS = frozenset({"section", "article", "clause", "paragraph"})
# "Sections 8.2 and 9.1 thereof", but not "Section 3.22 (except for paragraph (g) thereof)": there the "thereof"
# sits inside a parenthetical about Section 3.22 itself, so a skipped parenthetical may not hold another "("
THEREOF_AFTER = re.compile(r"^\s*(?:\([^()]*\)\s*)*,?\s*thereof\b")
# The section printed after a clause list the run did not take whole: "clauses (i) through (xxi) of Section 6.3(b)",
# "clauses (D), (E) and (J), of Section 8.2(b)", "clause (x) of the fourth sentence of Section 5.2",
# "clause (A) in the proviso of Section 9.01(b)(i)", "clauses (i) through (vii) above of this Section 12.5(a)"
_E = r"\([A-Za-z0-9]{1,6}\)"
# The item repeat is possessive and holds no inner repeat: "clause (a) (b) (c) ... (k) are paid" failed after
# exponential backtracking (each space between items could go to several \s). The rest starts "of" or "in", never
# "(", so the items never need to give anything back. Each run of spaces is taken possessively by one \s*+, since
# what follows it never starts with a space; three \s* in a row made a long run of spaces cubic.
LATE_QUALIFIED = re.compile(
    r"(?:\s*+(?:,\s*+)?(?:(?:and/or|and|or|through|to|[\-–])\s*+)?(?:clauses?\s+|paragraphs?\s+)?" + _E + r")*+"
    + r"\s*+(?:,\s*+)?(?:(?:above|below)\s+)?(?:in\s+each\s+case\s*+(?:,\s*+)?)?(?:of|in)\s+"
    + r"(?:the\s+(?:(?:first|second|third|fourth|fifth|sixth|last|final|penultimate|preceding|introductory|lead-in)\s+)?"
    + r"(?:sentence|paragraph|proviso|provisos)\s+(?:of|in|to)\s+)?(?:this\s+)?[Ss]ection\s+"
    + r"((?:\d+|" + _ROMAN_NUM + r"(?=\.\d))[A-Z]?(?:\.\d+[A-Z]?)+(?:\([A-Za-z0-9]{1,6}\))*)"
)


def _late_qualifier(text: str, pos: int) -> re.Match[str] | None:
    """The section that a clause run names after the rest of its list, unless that section is another document's."""
    m = LATE_QUALIFIED.match(text, pos)
    if m is None or m.end() - pos > 300:
        return None
    after = text[m.end() : m.end() + 90]
    if STATUTE.match(after) or OTHER_DOC.match(after) or DISCLOSURE.match(after) or ATTACHMENT_TAIL.match(after):
        return None
    return m


_PAREN_PAIR = re.compile(r"\([^()]*\)")
_THEREOF_SECTION = re.compile(r"(?:\d+|" + _ROMAN_NUM + r")\.\d")


def _thereof_section(run: list[_Ref], text: str, start: int, last_sec: list[_Ref] | None) -> list[_Ref] | None:
    """"Section 6.01(a) (other than clauses (iii) and (xx) thereof)", "Section 5.8 shall be enforceable as provided
    in clause (d) thereof": a clause run whose "thereof" means the section of this agreement printed just before.
    The gap between them stays in one sentence, within 160 characters, and names nothing (no capitalized word once
    whole parentheticals such as captions are dropped). Returns the section run a later "thereof" may use."""
    if run[0].head not in ("clause", "paragraph"):
        return run if run[0].head == "section" and not run[0].qualified else None
    sec = last_sec[-1] if last_sec else None
    if sec is None or run[0].detail != "other_document" or sec.status or not _THEREOF_SECTION.match(sec.target):
        return last_sec
    gap = text[sec.end : start]
    bare = _PAREN_PAIR.sub("", _PAREN_PAIR.sub("", gap))
    if len(gap) > 160 or re.search(r"[.;]", bare) or re.search(r"[A-Z][a-z]", bare):
        return last_sec
    for r in run:
        if r.target.startswith("("):
            r.head, r.target, r.status, r.detail = "section", sec.target + r.target, "", ""
            r.base_end, r.qualified = r.start, True
    return last_sec


@dataclass
class _Ref:
    head: str
    target: str
    start: int  # span over the printed reference
    end: int
    base_end: int  # end of the bare number ("6.3" in "6.3(g)"), for section fallback
    relative: bool = False
    status: str = ""
    detail: str = ""
    qualified: bool = False  # "clause (ii) of Section 4.2": the section is printed next to it
    lead: bool = False  # the first item of its run: it prints its own head ("Section 8")
    this_before: bool = False  # "this Section 8"
    plain_tail: bool = False  # no tail naming anything but this agreement, and no OCR continuation (".9", "-3")
    other_tail: bool = False  # "clause (iii) of paragraph 2 of Annex C of the Commitment Letter"
    tail: str = ""  # the text after the run (90 characters, stray items dropped), shared by its members
    self_marker: bool = False  # "this Section 6.3", "Section 6.3 hereof" (body only, rule 10)
    after: str = ""  # the text from the end of this item to 60 characters past the end of its run
    parent_enum: str = ""  # "clause (ii) of subparagraph (a) above": "(a)", the paragraph that holds the clause
    own_para: bool = False  # B1: "this clause (b)" resolved to the paragraph that holds it


def _section_range_end(base: str, hyphen: str, before: str) -> str | None:
    """L8: "7.16" and "-17" make the range end "7.17" when the tail is a larger number with as many digits as the
    base's last part and no authority precedes ("Treasury Regulations Section 1.368-2" stays one number)."""
    m = re.fullmatch(r"(\d{1,2})\.(\d{1,2})", base)
    tm = re.match(r"[\-‑](\d+)(?![A-Z])", hyphen)  # what follows the digits ("(Taxes)", "(a)") stays in the tail
    if not m or not tm or AUTHORITY_BEFORE.search(before):
        return None
    last, end = m.group(2), tm.group(1)
    if len(end) != len(last) or int(end) <= int(last):
        return None
    return f"{m.group(1)}.{end}"


def _find_refs(text: str, in_exhibit: bool, back: bool = False) -> list[_Ref]:
    # back: the text is in back matter, where "this Section 6.3" is an attached document's own section, so the
    # self markers and the verb guard do not apply there (2026-11-20 corpus read: 24 of 26 such links were wrong)
    refs: list[_Ref] = []
    prev_run: list[_Ref] = []
    prev_end: int | None = None
    prev_self = False
    # a paragraph with three or more all-caps heads is a table of contents, often of another document
    caps_toc = sum(1 for m in REF_HEAD.finditer(text) if m.group(1).isupper()) >= 3
    last_sec: list[_Ref] | None = None  # the latest section run, for a clause run's "thereof"
    for hm in REF_HEAD.finditer(text):
        # an all-caps "ARTICLE VII" that opens the paragraph is a heading, not a reference
        if hm.group(1).isupper() and (caps_toc or not text[: hm.start()].strip()):
            continue
        head = hm.group(1).lower().rstrip("s")
        head = _HEAD_NORM.get(head, head)
        pos = hm.end()
        run: list[_Ref] = []
        first = True
        while True:
            item_pos = pos
            # "Section 4,02": a comma typed for the period
            if run and (cm := OCR_COMMA.match(text, pos)) and re.fullmatch(r"\d+", run[-1].target):
                last = run[-1]
                last.target, last.end, last.base_end = f"{last.target}.{cm.group(1)}", cm.end(), cm.end()
                pos = cm.end()
                continue
            # "Section 1.368-3", "Section 1.1502-6(a)": a regulation number, not "1.368" and a list member "3".
            # The whole number fits no section, so it stays unresolved unless its tail makes it external.
            if run and re.fullmatch(r"\d+[A-Z]?(?:\.\d+[A-Z]?)+", run[-1].target) and (rm := REG_HYPHEN.match(text, pos)):
                last = run[-1]
                end = _section_range_end(last.target, rm.group(0), text[max(0, hm.start() - 40) : hm.start()]) if head == "section" else None
                stop = rm.start() + 1 + len(end.split(".")[1]) if end else 0
                # "Section 1.1-5 of the Disclosure Schedule" is one schedule item, numbered "1.1-5"
                if end and not DISCLOSURE.match(LEFTOVER_ITEMS.sub("", text[stop : stop + 90], count=1)):
                    # L8: "Sections 7.16-17 (Taxes)" is a range, 7.16 through 7.17; the second member prints only "17"
                    run.append(_Ref(head, end, rm.start() + 1, stop, stop))
                    pos = stop
                    continue
                last.target, last.end = last.target + re.sub(r"\s", "", rm.group(0)), rm.end()
                pos = rm.end()
                continue
            # "Section 1. 6(c)", "Section 9. 0 3(a)": a space inside the number; the reference is to
            # Section 1.6(c) and Section 9.03(a)
            if run and head == "section" and not run[-1].relative:
                last, sm, sep = run[-1], None, ""
                if re.fullmatch(r"\d+", last.target):
                    sm, sep = SPLIT_DOT.match(text, pos), "."
                elif re.fullmatch(r"\d+\.0", last.target):
                    sm = ZERO_SPLIT.match(text, pos)
                if sm:
                    last.target = f"{last.target}{sep}{sm.group(1)}" + re.sub(r"\s", "", sm.group(2))
                    last.end, last.base_end = sm.end(), sm.end(1)
                    pos = sm.end()
                    continue
            if not first:
                jm = REF_JOIN.match(text, pos)
                if not jm:
                    break
                item_pos = jm.end()
            # only consume a joiner when an item follows it: in "Section 3.14(a) to the Disclosure
            # Schedule" the "to" belongs to the tail, not to a range
            im = (ROMAN_ITEM if head in _IN_DOCUMENT_HEADS else ATTACHMENT_ITEM).match(text, item_pos) or REF_ITEM.match(text, item_pos)
            if not im and head == "article":
                im = SPELLED_ITEM.match(text, item_pos)
            if not im:
                break
            item = re.sub(r"\s", "", im.group(1))  # "6.2 (a)" is "6.2(a)"
            if item[-1].isdigit() and GLUED_HEAD.match(text, im.end()):
                break  # "Section 6.4Section 6.3(a)": the number runs into the next reference
            if not first and item.startswith("("):
                # a relative member "(c)" only continues the run when the previous item ends in
                # an enumerator of the same kind: "414(b), (c)" yes; "7.6(b), (ii) Everest" no
                prev = run[-1].target if run else ""
                pm = re.search(r"\(([A-Za-z0-9]{1,6})\)$", prev)
                tok_m = ENUM_TOKEN.match(item)
                if not pm or tok_m is None or _enum_kind(pm.group(1)) != _enum_kind(tok_m.group(1)):
                    break
                # "Section 3.2(d), or (b) the termination": a list moves forward; (b) after (d) is a new item.
                # Outside a range it moves in small steps: "Section 9.2(b) or (y) to the extent" is an
                # inline (x)/(y) pair, and "Section 5.1(d)(i)), and (ii)" leaves the parenthetical
                gap = _enum_ordinal(tok_m.group(1)) - _enum_ordinal(pm.group(1))
                joiner = text[pos:item_pos]
                if gap <= 0 or (gap > 30 and not re.search(r"through|to|[\-–]", joiner)) or ")" in joiner:
                    break
                # "Section 8.1 (d), or (y) Section 8.1 (c)", "Section 7.08(d)), (e) by taking": an item that
                # opens a phrase is an inline enumerator, not a list member
                if not RELATIVE_FOLLOW.match(text, im.end()):
                    break
                run.append(_Ref(head, prev[: pm.start()] + item, im.start(), im.end(), im.start(), relative=True))
                pos = im.end()
                continue
            bm = re.match(r"(?:\d+|" + _ROMAN_NUM + r"(?=\.\d))[A-Z]?(?:\.\d+[A-Z]?)*", item)
            base_end = im.start() + (bm.end() if bm else len(item))
            run.append(_Ref(head, item, hm.start() if first else im.start(), im.end(), base_end))
            pos = im.end()
            first = False
        if not run:
            continue
        # "Section 1.3.22. of the Company Disclosure Letter": a stray period before the tail
        tail = re.sub(r"^\.(?=\s+(?:of|to|in|under)\b)", "", text[pos : pos + 90])
        # "Section 4.14(b), (c), (m) or (o) of the Code": items the run did not take still lead to the tail
        tail = LEFTOVER_ITEMS.sub("", tail, count=1)
        # "Sections 2.2 to 2.5, inclusive, of the Distribution Agreement": the tail follows the "inclusive"
        tail = INCLUSIVE.sub("", tail, count=1)
        before = text[max(0, hm.start() - 40) : hm.start()]
        lower_head = hm.group(1) in ("section", "sections", "article", "articles")
        self_doc = (SELF_DOC_FROM_EXHIBIT if in_exhibit else SELF_DOC).match(tail)
        # "this Section 6.3", "Section 6.3 hereof": the document's own section, whatever the sentence names later
        self_marker = not back and bool(THIS_BEFORE.search(before) or HEREOF_TAIL.match(tail))
        status = ""
        if COLON_NUMBER.match(text, pos):
            # "articles 7:17 DCC", "section 2:107a": civil-code numbering, never this agreement's
            status = "external"
        elif AUTHORITY_BEFORE.search(before):
            status = "external"
        elif DISCLOSURE.match(tail):
            status = "disclosure_schedule"
        elif lower_head and DISCLOSURE_BEFORE.search(before):
            # "Seller Disclosure Schedule section 3.4(c)"; an index table's "Company Disclosure Schedule Section 2"
            # names a term and where it is defined
            status = "disclosure_schedule"
        elif not self_doc and not self_marker and (
            STATUTE.match(tail)
            or OTHER_DOC.match(tail)
            or (ACT_LATER.search(tail[:70]) is not None if back else _act_later(tail))
            # a lowercase "section 9 of the Decree", "article 15 of the DCC" is how other documents and foreign
            # statutes are cited; this agreement's own sections are capitalized or "of this Agreement"
            or (lower_head and LOWER_HEAD_OTHER.match(tail))
        ):
            status = "external"
        elif THEREOF_AFTER.match(tail):
            # "Sections 8.2 and 9.1 thereof": the sections of a document named just before
            status = "unresolved"
            for r in run:
                r.detail = "other_document"
        elif not self_marker and (om := OTHER_ATTACHMENT_TAIL.match(tail)):
            status = "disclosure_schedule" if re.search(r"(?i:disclosure)|Schedules?$", om.group(0)) else "external"
        elif ATTACHMENT_TAIL.match(tail):
            # "paragraph 3 of Schedule 2", "Section 2 of Exhibit B": the attachment's own numbering
            status = "unresolved"
            for r in run:
                r.detail = "attachment_local"
        elif in_exhibit and not self_doc:
            # an exhibit's bare "Section 2.1" usually means its own paragraphs; those
            # need attachment segmentation (Phase 4), so do not guess
            status = "unresolved"
        for r in run:
            r.status = status
            if status == "unresolved" and not r.detail:
                r.detail = "exhibit_local"
        qm = QUALIFIED.match(tail)
        qm = qm or (_late_qualifier(text, pos) if run[0].head in ("clause", "paragraph") else None)
        if qm and run[0].head in ("clause", "paragraph"):
            for r in run:
                if r.target.startswith("(") and not r.status:
                    # "clause (ii) of Section 4. 2": the section number may carry a stray space
                    r.head, r.target = "section", re.sub(r"\s", "", qm.group(1)) + r.target
                    r.base_end = r.start  # never fall back on a clause-only span
                    r.qualified = True
        last_sec = _thereof_section(run, text, hm.start(), last_sec)
        for r in run:
            if r.head == "schedule" and SEC_FORM.match(r.target):
                r.status = "external"
        # "Section 355 or Section 361 of the Code": a run joined to an external run is external too
        # "Section 3.2(a) or Section 3.2(b) of the Company Schedules": the same for disclosure schedules
        # A "this Section 6.3" run keeps its own status: "this Section 6.3 or Section 13 of the Exchange Act"
        if (
            prev_run
            and prev_end is not None
            and not prev_self
            and run[0].status in ("external", "disclosure_schedule")
            and re.fullmatch(r"\s*(?:,\s*)?(?:and|or|and/or)?\s*", text[prev_end : hm.start()])
        ):
            for r in prev_run:
                r.status = run[0].status
        pm = PARENT_ENUM.match(tail)
        other_tail = CLAUSE_OF_OTHER.match(tail) is not None or (pm is None and NAMED_ENUM.match(tail) is not None)
        for r in run:
            r.other_tail = other_tail
            r.parent_enum = re.sub(r"\s", "", pm.group(1)) if pm else ""
            r.tail, r.self_marker, r.after = tail, self_marker, text[r.end : pos + 60]
        run[0].lead = True
        run[0].this_before = THIS_BEFORE.search(before) is not None
        run[0].plain_tail = _plain_tail(tail)
        prev_run, prev_end, prev_self = run, pos, self_marker
        refs.extend(run)
    return refs


def _in_one_group(ol: _Outline, base: str, subs: str) -> _Node | None:
    """"Section 2.14(t)" in a section split into lettered groups: the one group that has a (t)."""
    hits = [ol.subs[k] for g in ol.groups if g.startswith(base + "(") and (k := g + subs) in ol.subs]
    return hits[0] if len(hits) == 1 else None


# "VI. CONDITIONS" (a Roman article title without the word) is cited "Article VI" too, never a bare "Section 6"
_ARTICLE_WORD = re.compile(r"^\s*(?:ARTICLE|Article|PART|Part)\b|^\s*" + _ROMAN_NUM + r"\.\s")


def _section_style_article(ol: _Outline, key: str) -> bool:
    """Article `key` can be cited as "Section 8": its heading prints "SECTION 8" or "8.", or it is flat."""
    return key in ol.flat_articles or not any(_ARTICLE_WORD.match(a.title) for a in ol.articles[key])


def _para_index(ol: _Outline) -> dict[str, list[tuple[int, int, str]]]:
    """Section uuid -> (paragraph ordinal, depth, key) of its paragraph nodes, sorted; built on first use."""
    if ol.para_index is None:
        ol.para_index = {}
        for key, node in ol.subs.items():
            if "t" in node.anchor:
                ol.para_index.setdefault(node.anchor["s"], []).append((node.anchor["t"], key.count("("), key))
        for v in ol.para_index.values():
            v.sort()
    return ol.para_index


def _own_paragraph(ref: _Ref, ol: _Outline, tn: _Text, t: str) -> _Node | None:
    """B1: "this clause (b)" inside a paragraph that opens with "(b)": the node anchored at that paragraph whose key
    ends in "(b)". None unless the paragraph's own leading enumerators include the item and exactly one node fits."""
    if not ref.lead or not ref.this_before or not re.fullmatch(r"\([A-Za-z0-9]{1,6}\)", t):
        return None
    pm = ENUM_PREFIX.match(tn.text)
    if pm is None or t[1:-1] not in ENUM_TOKEN.findall(pm.group(1)):
        return None
    keys = [key for pt, _, key in _para_index(ol).get(tn.s, []) if pt == tn.t and key.endswith(t) and key not in ol.groups]
    return ol.subs[keys[0]] if len(keys) == 1 else None


# B2: "clauses (v) or (vi) of the definition of Permitted Liens", "clause (i) of Rule 13a-15", "of the Code": the
# clause belongs to a definition, a rule or a statute, not to the paragraph that holds the reference
_DEF_OR_RULE_TAIL = re.compile(
    r"^\s*(?:\([^()]*\)\s*)*,?\s*(?:of|in|under|to)\s+(?:the\s+|such\s+|that\s+|any\s+)?(?:(?:definitions?|defined\s+terms?|terms?|meaning)\b|[\"“]|(?:Rule|Regulations?|Code|Act)\b)"
)


def _definitions_scope_ok(ref: _Ref, ol: _Outline, tn: _Text, target: str, key_by_id: dict[str, str], decl_ts: dict[str, list[int]]) -> bool:
    """B2: a clause resolved in a definitions section keeps its link when the reference sits in a paragraph the outline
    recorded, the target's parent is a paragraph (not the section) in that paragraph's own scope chain, no definition
    paragraph (one that declares a term and opens with no item number) lies between the target and the reference, and
    no tail names a definition, a rule or another document."""
    key = key_by_id.get(target)
    if key is None or ref.other_tail or _DEF_OR_RULE_TAIL.match(ref.tail) or not tn.section_key:
        return False
    if not any(pt == tn.t for pt, _, _ in _para_index(ol).get(tn.s, [])):
        return False
    # a later definition's "(c)" repeats a key the outline already holds, so the node may belong to another definition
    # ("sub-paragraphs (b) or (c) above" went to the "Control" definition's (c), 2026-11-21 read)
    tt = ol.subs[key].anchor.get("t")
    if tt is None or ol.subs[key].anchor.get("s") != tn.s:
        return False
    lo, hi = min(tt, tn.t), max(tt, tn.t)
    if any(lo < d <= hi for d in decl_ts.get(tn.s, ())):
        return False
    parent = key[: key.rfind("(")]
    return parent != tn.section_key and _nearest_scope(ol, tn, tn.section_key, [key]) == key


def _nearest_scope(ol: _Outline, tn: _Text, cur: str, cands: list[str]) -> str | None:
    """"clause (ii)" with several (ii) in the section: the one whose parent is the paragraph that holds the
    reference, or one of that paragraph's ancestors. None unless exactly one candidate is left."""
    here = [key for t, _, key in _para_index(ol).get(tn.s, []) if t <= tn.t]
    chain = {cur}
    k = here[-1] if here else ""
    while k:
        chain.add(k)  # "5.16.1(b)", then "5.16.1"
        k = k[: k.rfind("(")] if "(" in k else ""
    keep = [c for c in cands if c[: c.rfind("(")] in chain]
    return keep[0] if len(keep) == 1 else None


_SELF_SECTION_CITE = re.compile(r"(?i:\bthis\s+section)\s+\d")


def _self_cites_article(texts: list[_Text], ol: _Outline) -> bool:
    """Whether the body writes "this Section 8" inside a section 8.x of an "ARTICLE VIII" (no section 8): such an
    agreement calls its articles "Section N", so its bare "Section N" may link to article N (census: 27 of 30)."""
    for tn in texts:
        if tn.region != "body" or not tn.section_key or not _SELF_SECTION_CITE.search(tn.text):
            continue
        art = _norm_num(tn.section_key.split(".", 1)[0])
        if art not in ol.articles or ol.sections.get(art) or _section_style_article(ol, art):
            continue
        for r in _find_refs(tn.text, in_exhibit=False):
            if r.lead and r.this_before and r.head == "section" and not r.status and re.fullmatch(r"\d+", r.target) and _norm_num(r.target) == art:
                return True
    return False


def _self_cited_article(ref: _Ref, ol: _Outline, tn: _Text) -> bool:
    """A body "Section 8" that prints its head, in an agreement that cites its articles that way, with no tail naming
    another document and no OCR continuation of the number. Back matter never qualifies."""
    return ol.self_cited_articles and tn.region == "body" and ref.head == "section" and ref.lead and ref.plain_tail


_LARGE_BARE = re.compile(r"(\d+)[A-Z]?")
# "Section 409A", "Section 262", "Section 338(h)(10)": a bare number this large with no node of its own is a
# statute or another document (2026-11-20 census: 79 of 79 read). "Section 13" or "Section 16" of the Exchange
# Act stays unresolved: small numbers are too often the agreement's own articles.
_LARGE_BARE_MIN = 40


# L9 guard: "of Section 6.01.(a)", "of this Section 8.4 (b)": the qualifier continues past the number QUALIFIED read
_QUALIFIER_GOES_ON = re.compile(r"[^.;]{0,80}?\b[Ss]ection\s+\d+[A-Z]?(?:\.\d+[A-Z]?)*(?:\.\s?|\s)\(")


def _is_large_bare_number(ref: _Ref, ol: _Outline, base: str, subs: str) -> bool:
    """A bare "Section N", "Section N(a)" or "Section NA" with N >= 40 and no section, subsection or article N."""
    m = _LARGE_BARE.fullmatch(base)
    if ref.head != "section" or m is None or int(m.group(1)) < _LARGE_BARE_MIN:
        return False
    keys = {base, m.group(1)}
    return not any(k in ol.articles or k in ol.sections or k in ol.subs or (k + subs) in ol.subs for k in keys)


# L1: "Section 2.3 of the Escrow Agreement", "Article 5 under the Plan": a tail that names something other than this
# agreement, its own attachments or an article ("Section 9.5 of Article IX" is a charter's or bylaws' own numbering)
NAMED_OTHER_TAIL = re.compile(
    r"^\s*(?:\([^()]*\)\s*)*(?:of|under|pursuant\s+to)\s+(?:the\s+|such\s+|its\s+|their\s+)?"
    + r"(?!this\b|Agreement\b|hereof\b|Merger\s+Agreement\b|(?:Exhibit|Annex|Schedule|Appendix|Part|Article)\b)[A-Z]"
)


def _names_other(ref: _Ref) -> bool:
    """L1: the run's tail names another document or a statute, and no self marker says it is this agreement's."""
    t = ref.tail
    return (
        ref.head in ("section", "article", "clause")
        and not ref.self_marker
        and NAMED_OTHER_TAIL.match(t) is not None
        and not (DISCLOSURE.match(t) or ATTACHMENT_TAIL.match(t) or THEREOF_AFTER.match(t))
    )


# L2: what may follow a bare Roman "Section VIII" that means article VIII. Not "Section IX B", "Section I(1)",
# "Section IX.D"; and nothing in the rest of the clause names another document ("Section II of that certain
# valuation report", "Sections I and II of Exhibit I", "Section IX and X of the Stipulated Order")
_ROMAN_BARE_NEXT = re.compile(r"(?![\-(\w]|\.\S|\s+[A-Z]\b)")
_ROMAN_OTHER_LATER = re.compile(
    r"\b(?:of|under|in|to|pursuant\s+to)\s+(?:that\s+certain\b|(?:the|such|its|their|any)\s+(?!Agreement\b)[A-Z]|(?:Exhibit|Annex|Schedule|Appendix|Part)\b|[A-Z][\w\-]*[’']s\s)"
)


def _roman_section_article(ref: _Ref, ol: _Outline, tn: _Text) -> _Node | None:
    """L2: a body "Section VIII" (bare Roman) is article VIII when that article's heading opens "SECTION" or the
    reference reads "this Section VIII", and nothing after it names another document."""
    if tn.region != "body" or ref.head != "section" or not re.fullmatch(r"[IVXLC]+", ref.target):
        return None
    n = _roman_to_int(ref.target)
    arts = ol.articles.get(str(n)) if n else None
    if not arts or len(arts) != 1 or not _ROMAN_BARE_NEXT.match(ref.after):
        return None
    # the heading prints the Roman number too: "4. Escrow Accounts" under a Roman "Section IV" part is not it
    # (2026-11-20 corpus read: 5 of 30 such links were wrong)
    if not re.fullmatch(r"[IVXLC]+", arts[0].label):
        return None
    if not (re.match(r"\s*(?:SECTION|Section)\b", arts[0].title) or ref.this_before):
        return None
    clause = re.split(r"[;]|\.(?:\s|$)", ref.after, maxsplit=1)[0]
    if not _plain_tail(ref.tail) or _ROMAN_OTHER_LATER.search(clause):
        return None
    return arts[0]


def _resolve(ref: _Ref, ol: _Outline, tn: _Text, sec_text: str) -> tuple[str, str | None, list[str] | None]:
    """Returns (status, target id, ambiguous candidates)."""
    if ref.status == "unresolved" and ref.detail == "exhibit_local" and _names_other(ref):
        # L1 in back matter: an attachment's "Section 2.3 of the Escrow Agreement" is external when the main
        # agreement has no such node either ("Section 2.1 of the Purchase Agreement" may be the main agreement's)
        probe = replace(ref, status="", detail="")
        if _resolve(probe, ol, tn, sec_text)[0] == "external":
            ref.detail = ""
            return "external", None, None
    if ref.status:
        return ref.status, None, None
    t = ref.target
    if ol.roman_sections and ref.head in ("section", "clause", "article"):
        t = _deroman(t)  # "Section VIII.2(e)" is section 8.2(e)
    # UK drafting: "clause 9.4" is a section and "clause 5" a top-level clause.
    # Numbered "paragraph" references usually point into schedules, so they stay unlinked.
    numbered_clause = ref.head == "clause" and t[:1].isdigit()
    # "Article 2.1" in agreements that call their sections articles, when there is no article "2.1"
    dotted_article = ref.head == "article" and re.match(r"\d+\.\d", t) is not None
    if ref.head == "section" or numbered_clause or dotted_article:
        m = SECTION_TARGET.match(t)
        if not m:
            if (art := _roman_section_article(ref, ol, tn)) is not None:
                return "resolved", art.id, None
            # L1 for a number no section can have ("Section 1.368-2(g) of the Treasury regulations")
            return ("external" if _names_other(ref) else "unresolved"), None, None
        base = _norm_num(m.group(1))
        subs = m.group(2)
        nodes = ol.sections.get(base)
        if not nodes:
            if base + subs in ol.subs:
                return "resolved", ol.subs[base + subs].id, None
            if base in ol.subs:  # "Section 5.16.1(c)" where 5.16.1 has no paragraph (c)
                return "section_fallback", ol.subs[base].id, None
            # "Section 8.1A(d)" with no section 8.1A: paragraph (d) of group "A." in Section 8.1
            gm = re.fullmatch(r"(\d+(?:\.\d+)+)([A-Z])", base)
            if gm and len(ol.sections.get(gm.group(1), [])) == 1:
                group = f"{gm.group(1)}({gm.group(2)})"
                if group in ol.groups and group + subs in ol.subs:
                    return "resolved", ol.subs[group + subs].id, None
                if group in ol.groups:
                    return "section_fallback", ol.subs[group].id, None
            # "Section 8" in agreements whose top level is "SECTION 8" (or "8."), or a flat item.
            # Under "ARTICLE VIII" with sections 8.1, 8.2, a bare "Section 8" is a statute, another
            # document or an exhibit's own section, and seldom the article (0.8.0 sample: 3 of 28 right).
            if "." not in base and not subs and base in ol.articles and (_section_style_article(ol, base) or _self_cited_article(ref, ol, tn)):
                arts = ol.articles[base]
                if len(arts) > 1:
                    return "ambiguous", None, [a.id for a in arts]
                return "resolved", arts[0].id, None
            # "Section 2(c)" in a flat agreement whose item 2 has no paragraph (c) of its own
            if "." not in base and subs and base in ol.flat_articles and len(ol.articles[base]) == 1:
                if (one := _in_one_group(ol, base, subs)) is not None:
                    return "resolved", one.id, None
                return "section_fallback", ol.articles[base][0].id, None
            # L4: "Article 3.2.11" with no node falls back to the longest dotted section prefix that exists ("3.2"),
            # when nothing follows the run ("as are granted by Article 5.12" of a foreign code has no such prefix)
            if dotted_article and base.count(".") >= 2 and _plain_tail(ref.tail):
                parts = base.split(".")
                for k in range(len(parts) - 1, 1, -1):
                    pre = ".".join(parts[:k])
                    if pre in ol.subs:
                        return "section_fallback", ol.subs[pre].id, None
                    if len(ol.sections.get(pre, [])) == 1:
                        return "section_fallback", ol.sections[pre][0].id, None
            if _is_large_bare_number(ref, ol, base, subs) or _names_other(ref):
                return "external", None, None
            return "unresolved", None, None
        if len(nodes) > 1:
            return "ambiguous", None, [n.id for n in nodes]
        if not subs:
            return "resolved", nodes[0].id, None
        sub = ol.subs.get(base + subs)
        if sub is not None:
            return "resolved", sub.id, None
        if (one := _in_one_group(ol, base, subs)) is not None:
            return "resolved", one.id, None
        if ref.qualified:
            # L9: "clauses (i) through (xxiii) of this Section 6.4" where (i) sits under 6.4(a): the one descendant
            # of the printed section that ends in the clause
            # Not when the clause is an inline item of the reference's own paragraph ("the foregoing clauses (i)-(iii)
            # in this Section 2.6"), or the printed section goes on past the part read ("Section 6.01.(a)",
            # "Section 8.4 (b) (i)"): 2 of 30 such links in the first corpus read went to the wrong paragraph
            inline = tn.section_key == base and tn.text.count(subs) >= 2
            if re.fullmatch(r"\([A-Za-z0-9]+\)", subs) and not inline and not _QUALIFIER_GOES_ON.match(ref.tail):
                deep = [k for k in ol.subs if k.startswith(base + "(") and k.endswith(subs) and k not in ol.groups]
                if len(deep) == 1:
                    return "resolved", ol.subs[deep[0]].id, None
            # "clauses (i) through (v) of Section 3.4(c)" with no paragraph-level target: link the
            # deepest printed ancestor that exists, which the reader sees named right there
            parts = re.findall(r"\([A-Za-z0-9]+\)", subs)
            for k in range(len(parts) - 1, 0, -1):
                anc = ol.subs.get(base + "".join(parts[:k])) or _in_one_group(ol, base, "".join(parts[:k]))
                if anc is not None:
                    return "section_fallback", anc.id, None
            return "section_fallback", nodes[0].id, None
        if ref.base_end <= ref.start and not ref.relative:
            return "unresolved", None, None
        # "Sections 10.2(a), (b) and (c)" with no subsection targets: the printed section is right there
        return "section_fallback", nodes[0].id, None
    if ref.head == "article":
        key = t.split(".", 1)[0] if _ROMAN_LEAD.match(t) else t  # "Article IV.2" outside a Roman scheme
        key = key.upper()
        as_int = _SPELLED.index(t.lower()) + 1 if t.lower() in _SPELLED else _roman_to_int(key)
        key = str(as_int) if as_int else key
        arts = ol.articles.get(key)
        if not arts:
            return ("external" if _names_other(ref) else "unresolved"), None, None
        if len(arts) > 1:
            return "ambiguous", None, [a.id for a in arts]
        return "resolved", arts[0].id, None
    if ref.head in ("exhibit", "schedule", "annex", "appendix"):
        node = ol.attachments.get(f"{ref.head} {t.lower()}")
        if node is not None:
            return "resolved", node.id, None
        if ref.head == "schedule" and re.fullmatch(r"\d+(?:\.\d+)+(?:\([A-Za-z0-9]+\))*", t):
            return "disclosure_schedule", None, None
        return "unlocated_attachment", None, None
    if ref.head in ("clause", "paragraph"):
        cur = tn.section_key
        if (own := _own_paragraph(ref, ol, tn, t)) is not None:
            # B1: "all personal property described in this clause (b)" inside paragraph (b): its own leading "(b)"
            # made the count below call it an inline item
            ref.own_para = True
            return "resolved", own.id, None
        if t.startswith("(") and tn.text.count(t) >= 2:
            # "clauses (i) and (iii) through (v)" inside a paragraph that enumerates (i)...(v)
            # itself, typical of definitions: the items are inline, not subsections
            ref.detail = "inline"
            return "unresolved", None, None
        if cur and t.startswith("("):
            cands = [k for k in ol.subs if k.startswith(cur + "(") and k.endswith(t) and k not in ol.groups]
            if len(cands) == 1:
                return "resolved", ol.subs[cands[0]].id, None
            if len(cands) > 1:
                if ref.parent_enum:
                    # the tail names the clause's parent: take that parent's clause, never the nearest one
                    hits = [k for k in cands if k[: k.rfind("(")].endswith(ref.parent_enum)]
                    if len(hits) == 1:
                        return "resolved", ol.subs[hits[0]].id, None
                    return "ambiguous", None, [ol.subs[k].id for k in cands]
                if not ref.other_tail and (one := _nearest_scope(ol, tn, cur, cands)) is not None:
                    return "resolved", ol.subs[one].id, None
                return "ambiguous", None, [ol.subs[k].id for k in cands]
        if t.startswith("(") and sec_text.count(t) >= 2:
            # an inline enumeration in the same section ("(i) ... (ii) ... clause (i) above");
            # recorded, but there is no paragraph-level target to link yet
            ref.detail = "inline"
        return "unresolved", None, None
    return "unresolved", None, None


# ---------------------------------------------------------------- payload

# 'Awards. "Awards" ...', '"Awards" ...' and a numbered definition, '1.1 "Accounting Principles" ...'
_DECL_START = re.compile(rf"^\s*(?:[A-Z0-9][^“”\".]{{0,80}}?\.\s+|\d{{1,3}}(?:\.\d{{1,3}})+\.?\s+)?[{_Q_OPEN}][A-Z0-9][^{_Q_CLOSE}]{{0,80}}[{_Q_CLOSE}]")
_GLOSSARY_MIN_ENTRIES = 3
_DEFINITIONS_SECTION_MIN_DECLS = 5
# A paragraph that opens with its own number: "Section 4.1 Title", "4.1 Title", "2. Title", "ARTICLE II", and the
# UK "1 DEFINITIONS". A number before a quoted term ("1.1 “Accounting Principles” means") is a numbered
# definition, so it does not count, and neither does a bare "100 Main Street".
OWN_NUMBER = re.compile(
    r"^\s*(?:(?:Section|SECTION|Sec\.|§)\s*\d{1,3}[A-Z]?(?:\.\d{1,3})*\.?|(?:Article|ARTICLE)\s+(?:\d{1,3}|[IVXL]+)\b\.?"
    + r"|\d{1,3}(?:\.\d{1,3})+\.?|\d{1,3}\.|\d{1,2}(?=\s+[A-Z]{3,}\b))\s+(?=[A-Z(])"
)
_OWN_NUMBERING_MIN = 3
_OWN_NUMBERING_RATIO = 5  # at least one numbered paragraph per five definitions
_DOTTED = re.compile(r"\d+\.\d+")
_ATTACHMENT_HEADS = frozenset({"exhibit", "schedule", "annex", "appendix"})
# An index of defined terms, "Acceptable Confidentiality Agreement Section 4.5(a) Additional Amounts Section 6.3(h)
# Agreement Preamble": the places it cites, and the term before each one
INDEX_PLACE = re.compile(
    r"(?:Sections?|§)\s*\d+(?:\.\d+)*[A-Z]?(?:\s?\([A-Za-z0-9]{1,6}\))*|(?:Article|ARTICLE)\s+[IVXLC\d]+\b|Preamble\b|Recitals?\b"
)
_INDEX_TERM = re.compile(r"[A-Z0-9$“\"][\w’'&/\-.,“”\"$]*(?:\s+[\w’'&/\-.,“”\"$]+){0,9}")
_INDEX_SMALL = frozenset("a an and as at by for in of on or the to under with".split())
_INDEX_MIN_PAIRS = 4


def _index_pairs(text: str) -> int:
    """How many "Term Section N.M" pairs a paragraph holds. The text before each cited place must be a short
    capitalized term: "Closing Date", "Certificate of Merger", but not "as set forth in" or "(Capitalization),"."""
    pairs, prev = 0, 0
    for m in INDEX_PLACE.finditer(text):
        chunk = text[prev : m.start()].strip()
        prev = m.end()
        # a term that ends in a page number ("Definitions 1", "Term D-39") is a table-of-contents line
        if not _INDEX_TERM.fullmatch(chunk) or chunk.endswith(",") or re.search(r"(?:^|\s)(?:[A-Z]-)?\d+$", chunk):
            continue
        words = chunk.split()
        if words[-1].lower() in _INDEX_SMALL or any(not (w[0].isupper() or not w[0].isalpha() or w in _INDEX_SMALL) for w in words):
            continue
        pairs += m.group(0)[0] in "S§"  # "Preamble" and "Article II" separate entries; only sections count
    return pairs


# A heading title after a section number: "Section 9.1 Amendments; Waivers.", "Section 8. Fees and Expenses.",
# "Section 1.1 Definitions 1" (with a page number), or the whole paragraph. "Section 2.1 The Company shall" is not.
_TITLE_WORD = r"[A-Z][\w’'&/\-]*[,;]?"
_TITLE_SMALL = r"(?:of|and|or|the|to|for|in|on|by|with|a|an|upon|from|its|their|not|no)"
_TITLE = rf"{_TITLE_WORD}(?:\s+(?:{_TITLE_WORD}|{_TITLE_SMALL}))*"
BACK_HEADING_TITLE = re.compile(rf"[.:]?\s+(?:[-–—]\s*)?{_TITLE}\s*(?:[.:]|[-–—]|\s(?:[A-Z]{{1,2}}-)?\d{{1,3}}\s*$|$)")
# A mixed-case table-of-contents entry: "Section 1.2 Interpretation 4" before the next entry or the end
BACK_TOC_ENTRY = re.compile(
    rf"(?:Sections?|SECTIONS?|Articles?|ARTICLES?)\s+[\dIVXLC]+(?:\.\d+)*\.?\s+{_TITLE}\.?\s+(?:[A-Z]{{1,2}}-)?\d{{1,3}}(?![\d.])"
    + r"(?=\s+(?:Sections?|SECTIONS?|Articles?|ARTICLES?|Exhibits?|EXHIBITS?|Schedules?|SCHEDULES?|Annex|ANNEX)\b|\s*$)"
)
_BACK_TOC_MIN = 4
# The number at the start of an outline title: "Section 5.4", "ARTICLE VIII", "6.16"
_HEAD_LABEL = re.compile(r"^\s*(?:(?:Section|SECTION|Article|ARTICLE|Sec\.)\s*)?[\dIVXLC]+(?:\.\d+)*\.?\s*")


def _names_main_heading(r: _Ref, ol: _Outline, tn: _Text, sec_text: str, after: str, by_id: dict[str, _Node]) -> bool:
    """Whether a heading-shaped "Section 5.4 Authority; Approval" in back matter prints the title of the agreement's
    own Section 5.4: a list of disclosure schedules or the agreement's table of contents, so a reference after all."""
    if r.status:
        return False
    status, target, _ = _resolve(r, ol, tn, sec_text)
    node = by_id.get(target) if status in ("resolved", "section_fallback") and target else None
    if node is None:
        return False
    a = re.findall(r"[a-z]+", _HEAD_LABEL.sub("", node.title, count=1).lower())
    b = re.findall(r"[a-z]+", after.lower())
    k = min(len(a), len(b), 4)
    return k >= 1 and a[:k] == b[:k] and (k >= 2 or len(a) == len(b) == 1)


# Inside an attachment, "attached hereto as Exhibit A", "Exhibit A hereto", "Annex A of this release" and
# "Schedule 1 of the Company's constitution" name the attachment's own exhibit or another document's
_HERETO_BEFORE = re.compile(r"\bhere(?:to|with)\s+(?:as\s+)?$", re.IGNORECASE)
_HERETO_AFTER = re.compile(r"\s*(?:\([^)]*\)\s*)*,?\s*(?:(?:attached|annexed|appended)\s+)?here(?:to|with)\b", re.IGNORECASE)
_OWN_ATTACHMENT_TAIL = re.compile(r"\s*(?:\([^)]*\)\s*)*(?:of|to)\s+(?:this\s+(?![Aa]greement\b|AGREEMENT\b)[A-Za-z]|the\s+Company['’]s\b)")
_SCHEDULE_HEAD = re.compile(r"(?:SCHEDULE|Schedule)\b")
# A deed or covenant set out in a schedule ("this Deed", "this Tax Covenant") numbers its own clauses
_OWN_DEED = re.compile(r"\b[Tt]his\s+(?:[A-Z][a-z]+\s+)?(?:Deed|deed|Covenant|covenant)\b")
# "clause 2.1 of the Tax Covenant", "clause 4.2 of this Deed": another document; "of this agreement" is not
_OTHER_DOC_TAIL = re.compile(r"\s*(?:\([^)]*\)\s*)*,?\s*(?:of|in|under|to),?\s+(?:this|the)\s+(?![Aa]greement\b|AGREEMENT\b)(?:[A-Z]|deed\b)")
# "clause 2.1 above": the schedule's own paragraph
_ABOVE_BELOW = re.compile(r"\s*(?:\([^)]*\)\s*)*(?:above|below)\b")
# UK drafting cites its own clauses as "clause 9.2"; a US agreement says "Section 9.2" and its schedules' "clause 9.2"
# is something else
_CLAUSE_DOTTED = re.compile(r"\b[Cc]lauses?\s+\d+\.\d")
_SECTION_DOTTED = re.compile(r"\b(?:Sections?|SECTIONS?)\s+\d+\.\d")


@dataclass
class _Segment:
    """A stretch of back matter that starts at an attachment heading (or at the start of back matter)."""

    head: str  # the heading line that opens it, "" before the first one
    glossary: bool = False  # the agreement's own definitions: a bare "Section 5.4" there is the agreement's
    own_numbered: bool = False  # its paragraphs carry their own section numbers: an attached document
    defines_terms: bool = False  # three or more paragraphs open with a quoted term
    deed: bool = False  # it calls itself "this Deed" or "this Covenant": its "clause 2.1" is its own


def _back_segments(texts: list[_Text]) -> dict[int, _Segment]:
    """The segment of each back-matter text, by id().

    Back matter is split at attachment headings. A segment where at least three paragraphs open with a quoted
    term is a glossary for the agreement itself ("Exhibit A: Definitions"), so its bare "Section 5.4" means the
    agreement's Section 5.4, unless at least three of its paragraphs (and one per five definitions) open with their
    own section number. Such a
    segment is an attached document (registration rights, LLC, escrow or voting agreement) that defines its own
    terms, and its bare references are to its own sections. A segment without its own numbering that holds an
    index of defined terms (a paragraph with four or more "Term Section N.M" pairs) is the agreement's as well.
    """
    segments: list[tuple[_Segment, list[_Text]]] = [(_Segment(""), [])]
    for tn in texts:
        if tn.region != "backMatter":
            continue
        txt = tn.text.strip()
        if ATTACHMENT_HEADING.match(txt) and len(txt) < 120:
            segments.append((_Segment(txt), []))
        segments[-1][1].append(tn)
    out: dict[int, _Segment] = {}
    for seg, seg_texts in segments:
        # "Annex 3.7" whose paragraphs are "3.7.1", "3.7.2": the agreement's own Section 3.7 set out at the back,
        # so those numbers are the agreement's, not an attached document's
        hm = ATTACHMENT_HEADING.match(seg.head)
        cont = hm.group(2).rstrip(".") + "." if hm and _DOTTED.fullmatch(hm.group(2).rstrip(".")) else None
        own = sum(1 for tn in seg_texts if OWN_NUMBER.match(tn.text) and not (cont and tn.text.lstrip().startswith(cont)))
        entries = sum(1 for tn in seg_texts if _DECL_START.match(tn.text))
        # the agreement's own glossary can run on into a short unheaded document ("1. Amendment to the Merger
        # Agreement"): a few numbered paragraphs after a hundred definitions do not make it an attached document
        seg.own_numbered = own >= _OWN_NUMBERING_MIN and own * _OWN_NUMBERING_RATIO >= entries
        seg.defines_terms = entries >= _GLOSSARY_MIN_ENTRIES
        seg.deed = any(_OWN_DEED.search(tn.text) for tn in seg_texts)
        # an index of the agreement's defined terms ("Closing Date Section 2.2") cites the agreement too
        index = any(_index_pairs(tn.text) >= _INDEX_MIN_PAIRS for tn in seg_texts)
        seg.glossary = (seg.defines_terms or index) and not seg.own_numbered
        out.update((id(tn), seg) for tn in seg_texts)
    return out


def _printed_after(ol: _Outline, r: _Ref, tn: _Text) -> bool:
    """Whether the attachment an "Exhibit A" names has its heading later in back matter than the reference."""
    node = ol.attachments.get(f"{r.head} {r.target.lower()}")
    return node is not None and node.anchor.get("s") == tn.s and node.anchor.get("t", -1) > tn.t


def _glossary_exhibit_texts(texts: list[_Text]) -> set[int]:
    """Back-matter texts that sit in the agreement's own definitions exhibit (see _back_segments)."""
    return {k for k, seg in _back_segments(texts).items() if seg.glossary}


def build_reader_index(xml: str, *, agreement_uuid: str | None = None, xml_version: int | None = None) -> dict[str, Any]:
    root = ET.fromstring(xml)
    texts, by_el = _collect_texts(root)
    ol = _build_outline(root, by_el)
    ol.self_cited_articles = _self_cites_article(texts, ol)

    # terms
    decls = _collect_declarations(root, texts, by_el)
    order = {id(tn): i for i, tn in enumerate(texts)}
    decls.sort(key=lambda d: (order[id(d.tn)], d.start))
    term_ids: dict[str, int] = {}
    terms_json: list[dict[str, Any]] = []
    folded: dict[str, int] = {}
    for d in decls:
        if d.term not in term_ids:
            k = d.term.casefold()
            same = folded.get(k)
            if same is not None and (d.term.isupper() or terms_json[same]["term"].isupper()):
                # "COMPANY" in an old glossary and (the "Company") in the preamble are one term
                term_ids[d.term] = same
            else:
                term_ids[d.term] = len(terms_json)
                _ = folded.setdefault(k, term_ids[d.term])
                low = " " not in d.term and d.term.lower() in COMMON_WORD_TERMS
                terms_json.append({"id": term_ids[d.term], "term": d.term, "decls": [], "confidence": "low" if low else "high"})
        entry: dict[str, Any] = {"anchor": {"s": d.tn.s, "t": d.tn.t}, "span": d.tn.span(d.start, d.end), "kind": d.kind, "in": _container_label(d.tn)}
        if d.term != terms_json[term_ids[d.term]]["term"]:
            entry["as"] = d.term  # printed spelling, when it differs from the term's
        if d.pointer:
            entry["pointer"] = d.pointer
        terms_json[term_ids[d.term]]["decls"].append(entry)
    # a declared term keeps its own spelling: "Amtrol Companies" is not a form of "Amtrol Company"
    forms: dict[str, int] = dict(term_ids)
    for term, tid in term_ids.items():
        for f in _term_forms(term):
            _ = forms.setdefault(f, tid)
    low_ids = {t["id"] for t in terms_json if t["confidence"] == "low"}
    rx = _term_regex(forms)

    uses: dict[str, list[list[int]]] = {}
    n_uses = 0
    xrefs: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    detail_counts: Counter[str] = Counter()
    body_counts: Counter[str] = Counter()  # in-document heads, body text only
    sec_text_cache: dict[int, str] = {}

    back_segments = _back_segments(texts)
    body_text = " ".join(t.text for t in texts if t.region == "body")
    n_clauses = len(_CLAUSE_DOTTED.findall(body_text))
    uk_clauses = n_clauses > 0 and n_clauses >= len(_SECTION_DOTTED.findall(body_text))  # a body with neither is not UK drafting
    nodes_by_id: dict[str, _Node] = {}
    stack = list(ol.roots)
    while stack:
        n = stack.pop()
        nodes_by_id[n.id] = n
        stack.extend(n.children)
    # sections holding many declarations: their "(a)", "(i)" items belong to single definitions,
    # so a bare "clause (i)" there is never a subsection of the section
    decl_count = Counter(d.tn.s for d in decls)
    definition_containers = {c for c, n in decl_count.items() if n >= _DEFINITIONS_SECTION_MIN_DECLS}
    key_by_id = {n.id: k for k, n in ol.subs.items()}
    # B2: paragraphs that open with a declared term ("“Control” means"), not an item that declares one in passing
    decl_ts: dict[str, list[int]] = {}
    for d in decls:
        if not ENUM_PREFIX.match(d.tn.text):
            decl_ts.setdefault(d.tn.s, []).append(d.tn.t)
    for tn in texts:
        if tn.region not in _INDEXED_REGIONS:
            continue
        key = f"{tn.s}/{tn.t}"
        if rx is not None:
            skip_to = 0
            gm = GLOSSARY_ENTRY.match(tn.text)
            if gm and gm.group(1).strip() == gm.group(2).strip():
                skip_to = gm.end(1) + 1
            row: list[list[int]] = []
            for m in rx.finditer(tn.text):
                if m.start() < skip_to:
                    continue
                tid = forms[m.group(1)]
                if tid in low_ids and _sentence_initial(tn.text, m.start()):
                    continue
                # "Investment Company Act", "General Corporation Law", "Deutsche Bank Trust Company": part of a
                # proper name ("state or local Income Tax Law" still uses "Income Tax")
                if _in_proper_name(tn.text, m.start(), m.end()):
                    continue
                row.append([*tn.span(m.start(), m.end()), tid])
            if row:
                uses[key] = row
                n_uses += len(row)

        if tn.section_el is not None:
            sk = id(tn.section_el)
            if sk not in sec_text_cache:
                sec_text_cache[sk] = " ".join(t.text or "" for t in tn.section_el.iter("text"))
            sec_text = sec_text_cache[sk]
        else:
            sec_text = tn.text
        seg = back_segments.get(id(tn))
        in_exhibit = tn.region == "backMatter" and not (seg is not None and seg.glossary)
        is_heading = ATTACHMENT_HEADING.match(tn.text.strip()) is not None and len(tn.text.strip()) < 120
        lead = len(tn.text) - len(tn.text.lstrip())
        numbered_para = (tn.s, tn.t) in ol.numbered_paras
        back = tn.region == "backMatter"
        found = _find_refs(tn.text, in_exhibit=in_exhibit, back=back)
        if back and len(BACK_TOC_ENTRY.findall(tn.text)) >= _BACK_TOC_MIN:
            # "Section 1.1 Definitions 1 Section 1.2 Interpretation 4 ...": an attachment's table of contents, unless
            # its titles are the agreement's own (the agreement's contents or a list of its disclosure schedules)
            heads = [r for r in found if r.head in _IN_DOCUMENT_HEADS]
            same = sum(_names_main_heading(r, ol, tn, sec_text, tn.text[r.end : r.end + 120], nodes_by_id) for r in heads)
            if not (same >= 2 and same * 2 >= len(heads)):
                continue
        for r in found:
            if (is_heading or numbered_para) and r.start == lead:
                continue  # the heading "Annex 3.7" or "Section 5.16.1 Payments" itself, not a reference to it
            if (
                back and r.start == lead and r.head in _IN_DOCUMENT_HEADS and BACK_HEADING_TITLE.match(tn.text, r.end)
                and not _names_main_heading(r, ol, tn, sec_text, tn.text[r.end : r.end + 120], nodes_by_id)
            ):
                continue  # "Section 9.1 Amendments." opening a back-matter paragraph: an attachment's own heading
            if (
                r.status == "unresolved" and r.detail == "exhibit_local" and r.head in _ATTACHMENT_HEADS and seg is not None and (seg.defines_terms or not seg.own_numbered)
                and not _OWN_ATTACHMENT_TAIL.match(tn.text, r.end)
                and (not (_HERETO_BEFORE.search(tn.text, 0, r.start) or _HERETO_AFTER.match(tn.text, r.end)) or _printed_after(ol, r, tn))
            ):
                # "Exhibit C", "this Annex I": the exhibit-local rule is for section numbers. An attachment name is the
                # agreement's attachment, except in an attached document with its own numbering (its own exhibits);
                # one that defines terms keeps the glossary reading it had before rule 1 ("this Annex 3.7"). A tail that
                # names another document ("Annex A of this release") keeps the exhibit-local reading, and so does
                # "attached hereto as Exhibit A" unless that attachment is printed after the reference (the
                # attachment's own exhibit, printed next to it); before it, it is the agreement's or the attachment itself
                r.status = r.detail = ""
            if (
                r.status == "unresolved" and r.detail == "exhibit_local" and r.head == "clause" and seg is not None and _SCHEDULE_HEAD.match(seg.head) and _DOTTED.match(r.target)
                and not seg.deed and not THIS_BEFORE.search(tn.text, 0, r.start) and not _OTHER_DOC_TAIL.match(tn.text, r.end)
                and not _ABOVE_BELOW.match(tn.text, r.end) and uk_clauses
            ):
                # UK drafting: a schedule numbers its own items as paragraphs and cites the agreement's "clause 9.2".
                # Not "this clause 9.3" (the schedule's own), a deed or covenant set out in the schedule, or a tail
                # that names another document ("clause 2.1 of the Tax Covenant", "clause 3.1 of, the Primary Policy"), or
                # "clause 2.1 above"; and only when the agreement itself cites "clause N.M" at least as often as "Section N.M"
                r.status = r.detail = ""
            status, target, cands = _resolve(r, ol, tn, sec_text)
            if (
                status == "resolved" and r.head in ("clause", "paragraph") and r.target.startswith("(") and tn.s in definition_containers and not r.own_para
                and not (target and _definitions_scope_ok(r, ol, tn, target, key_by_id, decl_ts))
            ):
                status, target, r.detail = "unresolved", None, "inline"
            # a relative "(b)" or a qualified "clause (ii)" has no printed number of its own: keep its span
            end = r.base_end if status == "section_fallback" and not (r.qualified or r.relative) else r.end
            x: dict[str, Any] = {"at": {"s": tn.s, "t": tn.t}, "span": tn.span(r.start, end), "text": tn.text[r.start : end], "status": status}
            if target:
                x["target"] = target
            if cands:
                x["candidates"] = cands
            xrefs.append(x)
            status_counts[status] += 1
            if tn.region == "body" and r.head in _IN_DOCUMENT_HEADS:
                body_counts[status] += 1
            if r.detail:
                detail_counts[r.detail] += 1

    internal = sum(status_counts.values()) - status_counts["external"] - status_counts["disclosure_schedule"]
    linked = status_counts["resolved"] + status_counts["section_fallback"]
    body_internal = sum(body_counts.values()) - body_counts["external"] - body_counts["disclosure_schedule"]
    stats: dict[str, Any] = {
        "n_terms": len(terms_json),
        "n_term_uses": n_uses,
        "n_xrefs": sum(status_counts.values()),
        "n_xrefs_resolved": linked,
        "n_xrefs_unresolved": status_counts["unresolved"] + status_counts["ambiguous"] + status_counts["unlocated_attachment"],
        "xrefs_by_status": dict(sorted(status_counts.items())),
        "xrefs_by_detail": dict(sorted(detail_counts.items())),
        "exact_rate": round(status_counts["resolved"] / internal, 4) if internal else None,
        "section_level_rate": round(linked / internal, 4) if internal else None,
        "body_exact_rate": round(body_counts["resolved"] / body_internal, 4) if body_internal else None,
        "body_section_level_rate": round((body_counts["resolved"] + body_counts["section_fallback"]) / body_internal, 4) if body_internal else None,
        "sections_numbered": sum(len(v) for v in ol.sections.values()),
        "sections_unnumbered": ol.unnumbered_sections,
        "sections_positional": ol.positional_sections,
        "duplicate_section_numbers": sorted(k for k, v in ol.sections.items() if len(v) > 1),
        "subsections": len(ol.subs),
        "attachments_located": len({n.id for n in ol.attachments.values()}),  # aliases ("annex 2.02") are not nodes
    }
    return {
        "schema": PAYLOAD_SCHEMA,
        "enricher": ENRICHER_VERSION,
        "agreement_uuid": agreement_uuid or root.get("uuid"),
        "xml_version": xml_version,
        "outline": [n.to_json() for n in ol.roots],
        "terms": terms_json,
        "uses": uses,
        "xrefs": xrefs,
        "stats": stats,
    }


def reader_index_row(xml: str, *, agreement_uuid: str, xml_version: int) -> dict[str, Any]:
    """One `reader_index` table row. The payload is stored gzip-compressed (about
    40 KB instead of 190 KB per agreement) so the API can serve the bytes as-is
    with Content-Encoding: gzip."""
    payload = build_reader_index(xml, agreement_uuid=agreement_uuid, xml_version=xml_version)
    stats = payload["stats"]
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return {
        "agreement_uuid": agreement_uuid,
        "xml_version": xml_version,
        "enricher_version": ENRICHER_VERSION,
        "text_hash": text_hash(xml),
        "payload_gz": gzip.compress(body, mtime=0),
        "n_terms": stats["n_terms"],
        "n_term_uses": stats["n_term_uses"],
        "n_xrefs": stats["n_xrefs"],
        "n_xrefs_resolved": stats["n_xrefs_resolved"],
        "n_xrefs_unresolved": stats["n_xrefs_unresolved"],
        "body_section_level_rate": stats["body_section_level_rate"],
    }
