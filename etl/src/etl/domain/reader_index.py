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
from dataclasses import dataclass, field
from typing import Any

ENRICHER_VERSION = "0.5.0"
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

    def walk(el: ET.Element, region: str, container: str, sec_el: ET.Element | None) -> None:
        for child in el:
            if child.tag == "text":
                raw = child.text or ""
                norm = _NON_BREAKING_RE.sub(" ", raw)
                key = None
                if sec_el is not None:
                    m = SECTION_NUM.match(sec_el.get("title") or "")
                    key = _norm_num(m.group(1)) if m else None
                node = _Text(container, counters[container], region, norm, key, sec_el, _utf16_table(norm))
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

ARTICLE_NUM = re.compile(
    r"^\s*(?:ARTICLE|Article|PART|Part|SECTION|Section)\s+([IVXLC]+|\d+|[A-Z])\b|^\s*(\d+)\s*[.:]?\s+\D",
)
SECTION_NUM = re.compile(
    r"^\s*(?:Section|SECTION|Sec\.|§)?\s*(\d+[A-Z]?(?:\.\d+[A-Z]?)+|\d+[A-Z]?)\.?(?=\s|$)",
)
# A paragraph that opens with one or more enumerators: "(a)", "(a)(i)", "(ii) ..."
ENUM_PREFIX = re.compile(r"^\s*((?:\([A-Za-z0-9]{1,6}\)\s*)+)")
ENUM_TOKEN = re.compile(r"\(([A-Za-z0-9]{1,6})\)")
# "5.16.1 ..." or "Section 5.16.1 ...": a paragraph numbered as a dotted subsection
DECIMAL_PARA = re.compile(r"^\s*(?:(?:Section|SECTION|Sec\.|§)\s*)?(\d+(?:\.\d+){2,})\.?\s")
ATTACHMENT_HEADING = re.compile(
    r"^(EXHIBIT|Exhibit|ANNEX|Annex|SCHEDULE|Schedule|APPENDIX|Appendix)\s+([A-Z0-9][A-Z0-9.\-]*)\b"
)

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

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "kind": self.kind, "label": self.label, "title": self.title, "anchor": self.anchor}
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
    attachments: dict[str, _Node] = field(default_factory=dict)  # "exhibit a" -> node
    ids: set[str] = field(default_factory=set)
    unnumbered_sections: int = 0

    def unique_id(self, base: str) -> str:
        if base not in self.ids:
            self.ids.add(base)
            return base
        n = 2
        while f"{base}~{n}" in self.ids:
            n += 1
        self.ids.add(f"{base}~{n}")
        return f"{base}~{n}"


def _build_outline(root: ET.Element, by_el: dict[ET.Element, _Text]) -> _Outline:
    ol = _Outline()

    def add_section(sec: ET.Element, parent: list[_Node]) -> None:
        stitle = sec.get("title") or ""
        uuid = sec.get("uuid")
        sm = SECTION_NUM.match(stitle)
        if not sm or not uuid:
            ol.unnumbered_sections += 1
            return
        num = _norm_num(sm.group(1))
        node = _Node(ol.unique_id(f"s{num}"), "section", num, stitle, {"s": uuid})
        parent.append(node)
        ol.sections.setdefault(num, []).append(node)
        if len(ol.sections[num]) > 1:
            return  # duplicate number after a bad parse: no subsection targets either
        add_subs(sec, num, uuid, node)

    def add_subs(el: ET.Element, num: str, uuid: str, node: _Node) -> None:
        """Subsection nodes under `node` from the numbered paragraphs inside `el`."""
        stack: list[tuple[str, str]] = []  # (kind, token) per level
        base = num  # the latest dotted paragraph ("5.16.1") owns the "(a)" paragraphs after it
        for t_el in el.iter("text"):
            tn = by_el.get(t_el)
            if tn is None:
                continue
            txt = tn.text
            dm = DECIMAL_PARA.match(txt)
            if dm and _norm_num(dm.group(1)).startswith(num + "."):
                key = _norm_num(dm.group(1))
                if key not in ol.subs:
                    sub = _Node(ol.unique_id(f"s{key}"), "sub", key, txt.strip()[:80], {"s": uuid, "t": tn.t})
                    ol.subs[key] = sub
                    node.children.append(sub)
                base, stack = key, []
                continue
            pm = ENUM_PREFIX.match(txt)
            if not pm:
                continue
            tokens = ENUM_TOKEN.findall(pm.group(1))
            kinds = _classify_tokens(tokens, stack)
            level = next((i for i, (k, _) in enumerate(stack) if k == kinds[0]), len(stack))
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
                m = ARTICLE_NUM.match(title)
                uuid = child.get("uuid")
                raw = (m.group(1) or m.group(2)) if m else _glued_roman(title, last_article[0] + 1)
                if raw and uuid:
                    as_int = _roman_to_int(raw) if not raw.isdigit() else None
                    key = (str(as_int) if as_int else raw).upper()
                    if key.isdigit():
                        last_article[0] = int(key)
                    node = _Node(ol.unique_id(f"a{key}"), "article", raw, title, {"s": uuid})
                    parent.append(node)
                    ol.articles.setdefault(key, []).append(node)
                    if key.isdigit() and child.find(".//section") is None:
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
            key = f"{m.group(1).lower()} {m.group(2).lower().rstrip('.')}"
            if key in ol.attachments:
                continue
            node = _Node(ol.unique_id("x-" + key.replace(" ", "-")), "attachment", key.split(" ", 1)[1].upper(), txt, {"s": tn.s, "t": tn.t})
            ol.attachments[key] = node
            ol.roots.append(node)
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
        return f"Section {tn.section_key}"
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
    + r"^\s*(?:\([^)]*\)\s*)*,?\s*(?:of|under|pursuant to)\s+(?:the\s+)?(?:[A-Z]{3,6}\b|Code|DGCL|Exchange Act|Securities Act|HSR Act|ERISA|Treasury Regulations?|Regulation|Rule|Delaware|General Corporation Law|Investment Company Act|Internal Revenue Code|(?:[A-Z][A-Za-z]+\s+){1,4}(?:Act|Law|Code|Statutes?)\b|Title)",
)
DISCLOSURE = re.compile(
    # "of the Seller’s Disclosure Letter", "item 1 of the ...", "Disclosure ----- Schedule" across a page break
    r"^\s*(?:\([^)]*\)\s*)*(?:item\s+\d+\s+)?(?:of|to|in|under)\s+(?:the|such|any|each)\s+(?:[A-Z][\w\-’']*\s+){0,4}(?:(?i:disclosure[\s\-–—_]+(?:schedules?|letter))|Schedules)\b",
)
DISCLOSURE_BEFORE = re.compile(r"(?i:disclosure)\s+(?:Schedules?|Letter|Memorandum)\s*$")
LOWER_HEAD_OTHER = re.compile(r"^[\d:.()A-Za-z]*\s*(?:\([^)]*\)\s*)*,?\s*(?:of|under|pursuant\s+to)\s+(?:the\s+|such\s+|its\s+|their\s+)?(?!this\b|Agreement\b|hereof\b)[A-Z]")
OCR_COMMA = re.compile(r",(\d{2})(?![\d,])")
COLON_NUMBER = re.compile(r":\d")
LEFTOVER_ITEMS = re.compile(r"^(?:\s*,?\s*(?:and/or|and|or)?\s*\([A-Za-z0-9]{1,6}\))+")
THEREOF = re.compile(r"^\s*(?:\([^)]*\)\s*)*,?\s*thereof\b")
ATTACHMENT_TAIL = re.compile(r"^\s*(?:\([^)]*\)\s*)*(?:of|in|to)\s+(?:the\s+|this\s+)?(?:Schedules?|Part|Annex|Appendix|Exhibit)\s+[A-Z0-9IVX]")
SELF_DOC = re.compile(r"^\s*(?:\([^)]*\)\s*)*(?:of|to|in|under)\s+(?:the|this)\s+Agreement\b")
SELF_DOC_FROM_EXHIBIT = re.compile(r"^\s*(?:\([^)]*\)\s*)*(?:of|to|in|under)\s+(?:the|this)\s+(?:Merger\s+)?Agreement\b")
OTHER_DOC = re.compile(
    r"^\s*(?:\([^)]*\)\s*)*(?:of|to|in|under)\s+(?:the|such|any|each)\s+(?!Agreement\b)(?:[A-Z][\w\-]*\s+){0,5}(?:Agreement|Letter|Plan|Certificate|Charter|Bylaws|Indenture|Note|Schedule|Statement|Articles)\b"
    + r"|^\s*(?:\([^)]*\)\s*)*(?:of|to|in)\s+(?:the|its|their|such)\s+(?:[a-z]+\s+){0,3}(?:bylaws|by-laws|charter|certificate of incorporation|articles of incorporation|operating agreement)\b"
    + r"|^\s*(?:\([^)]*\)\s*)*(?:of|to|in)\s+(?:the|its|their|such)\s+(?:Amended\s+and\s+Restated\s+)?(?:Articles|Certificate)\s+of\s+(?:Incorporation|Association|Organization)\b",
)
# "Code Section 409A", "Treasury Regulations Section 1.368-2" (from the kirkland ingester)
AUTHORITY_BEFORE = re.compile(r"(?:\b(?:Code|Regulations?|Reg\.|ERISA|Act|Rules?|Treas\.|Treasury|Statute|Law|IRC|U\.S\.C\.|C\.F\.R\.)|§)\s*$")
SEC_FORM = re.compile(r"^(?:\d{2}[A-Z](?:-\d+)?|TO|13E-3)$")
# lowercase "section 3.5" and "article VI" are references too (rubric v1.4 lists the head words, not their case)
REF_HEAD = re.compile(r"\b(Sections?|Articles?|SECTIONS?|ARTICLES?|sections?|articles?|Exhibits?|Schedules?|Annex(?:es)?|Appendix|Appendices|clauses?|paragraphs?)\s+(?=[\dIVXLC(A-Z])")
_SPELLED: tuple[str, ...] = tuple("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())
# "Article Three"
SPELLED_ITEM = re.compile(r"(" + "|".join(w.capitalize() + "|" + w.upper() for w in _SPELLED) + r")\b")
REF_ITEM = re.compile(r"(\d+[A-Z]?(?:-\d+)?(?:\.\d+[A-Z]?)*(?:\s?\([A-Za-z0-9]{1,6}\))*|[IVXLC]+\b|[A-Z]\b(?:-\d+)?|\([A-Za-z0-9]{1,6}\)(?:\([A-Za-z0-9]{1,6}\))*)")
# "(other than Sections 1.1, 6.2 (a)), 6.6 (f)": a list can step out of the parenthetical it started in
RELATIVE_FOLLOW = re.compile(
    r"\s*(?:[,;.)]|$)|\s+(?:and|or|of|through|to|hereof|hereto|herein|thereof|above|below|shall|will|is|are|has|have|was|were|would|may|applies|apply)\b"
)
# "Sections 4.1 (Organization; Subsidiaries), 4.2 (Authority) and 4.7": a caption or other parenthetical
# (not an enumerator) may follow each item
REF_JOIN = re.compile(r"(?:\s*\((?![A-Za-z0-9]{1,6}\))[^()]{2,200}\))?(?:\s*(?:\)?,\s*(?:and|or)?\s*|\s+(?:and|or|through|to|and/or)\s+|\s*[\-–]\s*))")
QUALIFIED = re.compile(r"\s+(?:of|in)\s+(?:this\s+)?[Ss]ection\s+(\d+[A-Z]?(?:\.\d+[A-Z]?)*(?:\([A-Za-z0-9]{1,6}\))*)")
SECTION_TARGET = re.compile(r"(\d+[A-Z]?(?:\.\d+[A-Z]?)*)((?:\([A-Za-z0-9]+\))*)$")

_HEAD_NORM = {"annexe": "annex", "appendice": "appendix"}
_IN_DOCUMENT_HEADS = frozenset({"section", "article", "clause", "paragraph"})


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


def _find_refs(text: str, in_exhibit: bool) -> list[_Ref]:
    refs: list[_Ref] = []
    prev_run: list[_Ref] = []
    prev_end: int | None = None
    # a paragraph with three or more all-caps heads is a table of contents, often of another document
    caps_toc = sum(1 for m in REF_HEAD.finditer(text) if m.group(1).isupper()) >= 3
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
            if not first:
                jm = REF_JOIN.match(text, pos)
                if not jm:
                    break
                item_pos = jm.end()
            # only consume a joiner when an item follows it: in "Section 3.14(a) to the Disclosure
            # Schedule" the "to" belongs to the tail, not to a range
            im = REF_ITEM.match(text, item_pos)
            if not im and head == "article":
                im = SPELLED_ITEM.match(text, item_pos)
            if not im:
                break
            item = re.sub(r"\s", "", im.group(1))  # "6.2 (a)" is "6.2(a)"
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
            bm = re.match(r"\d+[A-Z]?(?:\.\d+[A-Z]?)*", item)
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
        before = text[max(0, hm.start() - 40) : hm.start()]
        lower_head = hm.group(1) in ("section", "sections", "article", "articles")
        self_doc = (SELF_DOC_FROM_EXHIBIT if in_exhibit else SELF_DOC).match(tail)
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
        elif not self_doc and (
            STATUTE.match(tail)
            or OTHER_DOC.match(tail)
            or re.search(r"\bof\s+(?:the\s+)?[A-Z][\w .\-()]{2,60}\bAct\b", tail[:70])
            # a lowercase "section 9 of the Decree", "article 15 of the DCC" is how other documents and foreign
            # statutes are cited; this agreement's own sections are capitalized or "of this Agreement"
            or (lower_head and LOWER_HEAD_OTHER.match(tail))
        ):
            status = "external"
        elif THEREOF.match(tail):
            # "Sections 8.2 and 9.1 thereof": the sections of a document named just before
            status = "unresolved"
            for r in run:
                r.detail = "other_document"
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
        if qm and run[0].head in ("clause", "paragraph"):
            for r in run:
                if r.target.startswith("(") and not r.status:
                    r.head, r.target = "section", qm.group(1) + r.target
                    r.base_end = r.start  # never fall back on a clause-only span
                    r.qualified = True
        for r in run:
            if r.head == "schedule" and SEC_FORM.match(r.target):
                r.status = "external"
        # "Section 355 or Section 361 of the Code": a run joined to an external run is external too
        # "Section 3.2(a) or Section 3.2(b) of the Company Schedules": the same for disclosure schedules
        if (
            prev_run
            and prev_end is not None
            and run[0].status in ("external", "disclosure_schedule")
            and re.fullmatch(r"\s*(?:,\s*)?(?:and|or|and/or)?\s*", text[prev_end : hm.start()])
        ):
            for r in prev_run:
                r.status = run[0].status
        prev_run, prev_end = run, pos
        refs.extend(run)
    return refs


def _resolve(ref: _Ref, ol: _Outline, tn: _Text, sec_text: str) -> tuple[str, str | None, list[str] | None]:
    """Returns (status, target id, ambiguous candidates)."""
    if ref.status:
        return ref.status, None, None
    t = ref.target
    # UK drafting: "clause 9.4" is a section and "clause 5" a top-level clause.
    # Numbered "paragraph" references usually point into schedules, so they stay unlinked.
    numbered_clause = ref.head == "clause" and t[:1].isdigit()
    # "Article 2.1" in agreements that call their sections articles, when there is no article "2.1"
    dotted_article = ref.head == "article" and re.match(r"\d+\.\d", t) is not None
    if ref.head == "section" or numbered_clause or dotted_article:
        m = SECTION_TARGET.match(t)
        if not m:
            return "unresolved", None, None
        base = _norm_num(m.group(1))
        subs = m.group(2)
        nodes = ol.sections.get(base)
        if not nodes:
            if base + subs in ol.subs:
                return "resolved", ol.subs[base + subs].id, None
            if base in ol.subs:  # "Section 5.16.1(c)" where 5.16.1 has no paragraph (c)
                return "section_fallback", ol.subs[base].id, None
            # "Section 8" in agreements whose top level is "SECTION 8"
            if "." not in base and not subs and base in ol.articles:
                arts = ol.articles[base]
                if len(arts) > 1:
                    return "ambiguous", None, [a.id for a in arts]
                return "resolved", arts[0].id, None
            # "Section 2(c)" in a flat agreement whose item 2 has no paragraph (c) of its own
            if "." not in base and subs and base in ol.flat_articles and len(ol.articles[base]) == 1:
                return "section_fallback", ol.articles[base][0].id, None
            return "unresolved", None, None
        if len(nodes) > 1:
            return "ambiguous", None, [n.id for n in nodes]
        if not subs:
            return "resolved", nodes[0].id, None
        sub = ol.subs.get(base + subs)
        if sub is not None:
            return "resolved", sub.id, None
        if ref.qualified:
            # "clauses (i) through (v) of Section 3.4(c)" with no paragraph-level target: link the
            # deepest printed ancestor that exists, which the reader sees named right there
            parts = re.findall(r"\([A-Za-z0-9]+\)", subs)
            for k in range(len(parts) - 1, 0, -1):
                anc = ol.subs.get(base + "".join(parts[:k]))
                if anc is not None:
                    return "section_fallback", anc.id, None
            return "section_fallback", nodes[0].id, None
        if ref.base_end <= ref.start and not ref.relative:
            return "unresolved", None, None
        # "Sections 10.2(a), (b) and (c)" with no subsection targets: the printed section is right there
        return "section_fallback", nodes[0].id, None
    if ref.head == "article":
        key = t.upper()
        as_int = _SPELLED.index(t.lower()) + 1 if t.lower() in _SPELLED else _roman_to_int(key)
        key = str(as_int) if as_int else key
        arts = ol.articles.get(key)
        if not arts:
            return "unresolved", None, None
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
        if t.startswith("(") and tn.text.count(t) >= 2:
            # "clauses (i) and (iii) through (v)" inside a paragraph that enumerates (i)...(v)
            # itself, typical of definitions: the items are inline, not subsections
            ref.detail = "inline"
            return "unresolved", None, None
        if cur and t.startswith("("):
            cands = [k for k in ol.subs if k.startswith(cur + "(") and k.endswith(t)]
            if len(cands) == 1:
                return "resolved", ol.subs[cands[0]].id, None
            if len(cands) > 1:
                return "ambiguous", None, [ol.subs[k].id for k in cands]
        if t.startswith("(") and sec_text.count(t) >= 2:
            # an inline enumeration in the same section ("(i) ... (ii) ... clause (i) above");
            # recorded, but there is no paragraph-level target to link yet
            ref.detail = "inline"
        return "unresolved", None, None
    return "unresolved", None, None


# ---------------------------------------------------------------- payload

_DECL_START = re.compile(rf"^\s*(?:[A-Z0-9][^“”\".]{{0,80}}?\.\s+)?[{_Q_OPEN}][A-Z0-9][^{_Q_CLOSE}]{{0,80}}[{_Q_CLOSE}]")
_GLOSSARY_MIN_ENTRIES = 3
_DEFINITIONS_SECTION_MIN_DECLS = 5


def _glossary_exhibit_texts(texts: list[_Text]) -> set[int]:
    """Back-matter texts that sit in a definitions exhibit ("Exhibit A: Definitions").

    Back matter is split at attachment headings. A segment where at least three
    paragraphs open with a quoted term is a glossary for the agreement itself, so
    its bare "Section 5.4" means the agreement's Section 5.4. Other exhibits keep
    the conservative exhibit-local rule.
    """
    segments: list[list[_Text]] = [[]]
    for tn in texts:
        if tn.region != "backMatter":
            continue
        txt = tn.text.strip()
        if ATTACHMENT_HEADING.match(txt) and len(txt) < 120:
            segments.append([])
        segments[-1].append(tn)
    out: set[int] = set()
    for seg in segments:
        if sum(1 for tn in seg if _DECL_START.match(tn.text)) >= _GLOSSARY_MIN_ENTRIES:
            out.update(id(tn) for tn in seg)
    return out


def build_reader_index(xml: str, *, agreement_uuid: str | None = None, xml_version: int | None = None) -> dict[str, Any]:
    root = ET.fromstring(xml)
    texts, by_el = _collect_texts(root)
    ol = _build_outline(root, by_el)

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

    glossary_texts = _glossary_exhibit_texts(texts)
    # sections holding many declarations: their "(a)", "(i)" items belong to single definitions,
    # so a bare "clause (i)" there is never a subsection of the section
    decl_count = Counter(d.tn.s for d in decls)
    definition_containers = {c for c, n in decl_count.items() if n >= _DEFINITIONS_SECTION_MIN_DECLS}
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
        in_exhibit = tn.region == "backMatter" and id(tn) not in glossary_texts
        is_heading = ATTACHMENT_HEADING.match(tn.text.strip()) is not None and len(tn.text.strip()) < 120
        lead = len(tn.text) - len(tn.text.lstrip())
        numbered_para = DECIMAL_PARA.match(tn.text) is not None
        for r in _find_refs(tn.text, in_exhibit=in_exhibit):
            if (is_heading or numbered_para) and r.start == lead:
                continue  # the heading "Annex 3.7" or "Section 5.16.1 Payments" itself, not a reference to it
            status, target, cands = _resolve(r, ol, tn, sec_text)
            if status == "resolved" and r.head in ("clause", "paragraph") and r.target.startswith("(") and tn.s in definition_containers:
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
        "duplicate_section_numbers": sorted(k for k, v in ol.sections.items() if len(v) > 1),
        "subsections": len(ol.subs),
        "attachments_located": len(ol.attachments),
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
