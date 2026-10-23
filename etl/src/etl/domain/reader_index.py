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

ENRICHER_VERSION = "0.1.0"
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
    """SHA-256 over the document's text nodes. Guards against in-place rewrites of
    one XML version; attribute-only rewrites (taxonomy standardId) leave it alone."""
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
DECIMAL_PARA = re.compile(r"^\s*(\d+(?:\.\d+){2,})\.?\s")
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
        stack: list[tuple[str, str]] = []  # (kind, token) per level
        for t_el in sec.iter("text"):
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
                key = num + "".join(f"({tk})" for _, tk in stack)
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
_TERM = r"[A-Z0-9][^“”\"]{0,80}?"
# "Foo" means / shall mean / is defined as / as used herein
DEF_MEANS = re.compile(
    rf"[{_Q_OPEN}]({_TERM})[{_Q_CLOSE}]\s*(?:\([^)]*\)\s*)?(?:,\s*)?(?:shall\s+)?(?:means?\b|is defined(?!\s+in\b)|as used herein)",
)
# (the "Company"), (each, a "Party"), (collectively, the "Sellers"), ("Parent")
DEF_PAREN = re.compile(
    rf"\(\s*(?:(?:together|collectively|individually|each|each of which|any such|such|hereinafter|herein)[^{_Q_OPEN})]*?\s+)?(?:the\s+|a\s+|an\s+)?[{_Q_OPEN}]({_TERM})[{_Q_CLOSE}](?:\s*(?:or|and)\s+(?:the\s+|a\s+|an\s+)?[{_Q_OPEN}]({_TERM})[{_Q_CLOSE}])?\s*\)",
)
# (the entity so determined, the "EBITDA Expert"), (jointly, a "Parent Corporate Event")
DEF_PAREN_LEADIN = re.compile(rf"\(\s*[^()“”\"]{{1,80}}?,\s*(?:the|a|an)\s+[{_Q_OPEN}]({_TERM})[{_Q_CLOSE}]\s*\)")
# referred to herein as "Compaq Broad Plans", being referred to herein as the "Merger Effective Time"
DEF_AS = re.compile(rf"\b(?:referred\s+to|defined)\s+(?:herein\s+|hereinafter\s+|below\s+)?as\s+(?:the\s+|a\s+|an\s+)?[{_Q_OPEN}]({_TERM})[,.]?[{_Q_CLOSE}]")
# the term "Material Adverse Effect," when used ..., means
DEF_THE_TERM = re.compile(rf"\bthe\s+term\s+[{_Q_OPEN}]({_TERM})[,.]?[{_Q_CLOSE}]")
# "Foo" has the meaning set forth in Section 3.2 / is defined in Section 3.2
DEF_POINTER = re.compile(
    rf"[{_Q_OPEN}]({_TERM})[{_Q_CLOSE}]\s*(?:shall\s+)?(?:has|have|is)\s+(?:the\s+meanings?\s+(?:set\s+forth|given|ascribed|assigned|specified)\s+(?:to\s+(?:it|such term|them)\s+)?in|defined\s+in)\s+(Section|Article|Exhibit|Schedule|Annex)\s+([0-9A-Z.()]*[0-9A-Z)])",
)
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
            if pm:
                found.append(_Decl(m.group(1).strip(), "pointer", tn, m.start(1), m.end(1), f"{pm.group(2)} {pm.group(3)}"))
            else:
                found.append(_Decl(m.group(1).strip(), "element", tn, m.start(1), m.end(1)))
    for tn in texts:
        if tn.region not in _INDEXED_REGIONS:
            continue
        pointer_starts: set[int] = set()
        for m in DEF_POINTER.finditer(tn.text):
            pointer_starts.add(m.start())
            found.append(_Decl(m.group(1).strip(), "pointer", tn, m.start(1), m.end(1), f"{m.group(2)} {m.group(3)}"))
        for m in DEF_MEANS.finditer(tn.text):
            if m.start() not in pointer_starts:
                found.append(_Decl(m.group(1).strip(), "means", tn, m.start(1), m.end(1)))
        for m in DEF_PAREN.finditer(tn.text):
            for g in (1, 2):
                if m.group(g):
                    found.append(_Decl(m.group(g).strip(), "paren", tn, m.start(g), m.end(g)))
        for rx in (DEF_PAREN_LEADIN, DEF_AS):
            for m in rx.finditer(tn.text):
                found.append(_Decl(m.group(1).strip(), "paren", tn, m.start(1), m.end(1)))
        for m in DEF_THE_TERM.finditer(tn.text):
            found.append(_Decl(m.group(1).strip(), "means", tn, m.start(1), m.end(1)))
    # one declaration per (term, text node), the strongest kind wins
    best: dict[tuple[str, str, int], _Decl] = {}
    for d in found:
        if not 2 <= len(d.term) <= 80:
            continue
        k = (d.term, d.tn.s, d.tn.t)
        if k not in best or _KIND_RANK[d.kind] < _KIND_RANK[best[k].kind]:
            best[k] = d
    decls = list(best.values())
    # pointers only stand in for terms with no real declaration
    real = {d.term for d in decls if d.kind != "pointer"}
    return [d for d in decls if d.kind != "pointer" or d.term not in real]


def _term_forms(term: str) -> list[str]:
    forms = [term]
    if term.endswith("y") and len(term) > 2 and term[-2] not in "aeiou":
        forms.append(term[:-1] + "ies")
    # a plural term ("Acquired Employees") is also used in the singular
    if term.endswith("ies") and len(term) > 4:
        forms.append(term[:-3] + "y")
    elif term.endswith("s") and not term.endswith("ss") and len(term) > 3 and term[-2].islower():
        forms.append(term[:-1])
    return forms


def _term_regex(forms: dict[str, int]) -> re.Pattern[str] | None:
    alts = sorted((re.escape(f) for f in forms if len(f) >= 2), key=len, reverse=True)
    if not alts:
        return None
    # longest first; plural, "(s)" and possessive forms; case-sensitive
    return re.compile(
        r"(?<![\w“\"])(" + "|".join(alts) + r")(?:\(s\)|s|es)?(?:’s|'s|’|')?(?![\w”\"])"
    )


_ENUM_ONLY = re.compile(r"^\s*(?:\([A-Za-z0-9]{1,6}\)\s*)*$")


def _sentence_initial(text: str, start: int) -> bool:
    prefix = text[:start].rstrip()
    return not prefix or prefix[-1] in ".:;!?" or bool(_ENUM_ONLY.match(prefix))


# ---------------------------------------------------------------- xrefs

STATUTE = re.compile(
    r"^\s*(?:\([^)]*\)\s*)*,?\s*(?:of|under|pursuant to)\s+(?:the\s+)?(?:[A-Z]{3,6}\b|Code|DGCL|Exchange Act|Securities Act|HSR Act|ERISA|Treasury Regulations?|Regulation|Rule|Delaware|General Corporation Law|Investment Company Act|Internal Revenue Code|(?:[A-Z][A-Za-z]+\s+){1,4}(?:Act|Law|Code|Statutes?)\b|Title)",
)
DISCLOSURE = re.compile(
    r"^\s*(?:\([^)]*\)\s*)*(?:of|to|in|under)\s+(?:the|such|any|each)\s+(?:[A-Z][\w\-]*\s+){0,4}(?:Disclosure\s+(?:Schedules?|Letter)|Schedules)\b",
)
ATTACHMENT_TAIL = re.compile(r"^\s*(?:\([^)]*\)\s*)*(?:of|in|to)\s+(?:the\s+|this\s+)?(?:Schedules?|Part|Annex|Appendix|Exhibit)\s+[A-Z0-9IVX]")
SELF_DOC = re.compile(r"^\s*(?:\([^)]*\)\s*)*(?:of|to|in|under)\s+(?:the|this)\s+Agreement\b")
SELF_DOC_FROM_EXHIBIT = re.compile(r"^\s*(?:\([^)]*\)\s*)*(?:of|to|in|under)\s+(?:the|this)\s+(?:Merger\s+)?Agreement\b")
OTHER_DOC = re.compile(
    r"^\s*(?:\([^)]*\)\s*)*(?:of|to|in|under)\s+(?:the|such|any|each)\s+(?!Agreement\b)(?:[A-Z][\w\-]*\s+){0,5}(?:Agreement|Letter|Plan|Certificate|Charter|Bylaws|Indenture|Note|Schedule|Statement)\b"
    r"|^\s*(?:\([^)]*\)\s*)*(?:of|to|in)\s+(?:the|its|their|such)\s+(?:[a-z]+\s+){0,3}(?:bylaws|by-laws|charter|certificate of incorporation|articles of incorporation|operating agreement)\b",
)
# "Code Section 409A", "Treasury Regulations Section 1.368-2" (from the kirkland ingester)
AUTHORITY_BEFORE = re.compile(r"(?:\b(?:Code|Regulations?|Reg\.|ERISA|Act|Rules?|Treas\.|Treasury|Statute|Law|IRC|U\.S\.C\.|C\.F\.R\.)|§)\s*$")
SEC_FORM = re.compile(r"^(?:\d{2}[A-Z](?:-\d+)?|TO|13E-3)$")
REF_HEAD = re.compile(r"\b(Sections?|Articles?|Exhibits?|Schedules?|Annex(?:es)?|Appendix|Appendices|clauses?|paragraphs?)\s+(?=[\dIVXLC(A-Z])")
REF_ITEM = re.compile(r"(\d+[A-Z]?(?:-\d+)?(?:\.\d+[A-Z]?)*(?:\([A-Za-z0-9]{1,6}\))*|[IVXLC]+\b|[A-Z]\b(?:-\d+)?|\([A-Za-z0-9]{1,6}\)(?:\([A-Za-z0-9]{1,6}\))*)")
REF_JOIN = re.compile(r"\s*(?:,\s*(?:and|or)?\s*|\s+(?:and|or|through|to|and/or)\s+|\s*[\-–]\s*)")
QUALIFIED = re.compile(r"\s+(?:of|in)\s+(?:this\s+)?Section\s+(\d+[A-Z]?(?:\.\d+[A-Z]?)*(?:\([A-Za-z0-9]{1,6}\))*)")
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


def _find_refs(text: str, in_exhibit: bool) -> list[_Ref]:
    refs: list[_Ref] = []
    prev_run: list[_Ref] = []
    prev_end: int | None = None
    for hm in REF_HEAD.finditer(text):
        head = hm.group(1).lower().rstrip("s")
        head = _HEAD_NORM.get(head, head)
        pos = hm.end()
        run: list[_Ref] = []
        first = True
        while True:
            if not first:
                jm = REF_JOIN.match(text, pos)
                if not jm:
                    break
                pos = jm.end()
            im = REF_ITEM.match(text, pos)
            if not im:
                break
            item = im.group(1)
            if not first and item.startswith("("):
                # a relative member "(c)" only continues the run when the previous item ends in
                # an enumerator of the same kind: "414(b), (c)" yes; "7.6(b), (ii) Everest" no
                prev = run[-1].target if run else ""
                pm = re.search(r"\(([A-Za-z0-9]{1,6})\)$", prev)
                tok_m = ENUM_TOKEN.match(item)
                if not pm or tok_m is None or _enum_kind(pm.group(1)) != _enum_kind(tok_m.group(1)):
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
        tail = text[pos : pos + 90]
        before = text[max(0, hm.start() - 40) : hm.start()]
        self_doc = (SELF_DOC_FROM_EXHIBIT if in_exhibit else SELF_DOC).match(tail)
        status = ""
        if AUTHORITY_BEFORE.search(before):
            status = "external"
        elif DISCLOSURE.match(tail):
            status = "disclosure_schedule"
        elif not self_doc and (
            STATUTE.match(tail) or OTHER_DOC.match(tail) or re.search(r"\bof\s+(?:the\s+)?[A-Z][\w .\-]{2,60}\bAct\b", tail[:70])
        ):
            status = "external"
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
    if ref.head == "section" or numbered_clause:
        m = SECTION_TARGET.match(t)
        if not m:
            return "unresolved", None, None
        base = _norm_num(m.group(1))
        subs = m.group(2)
        nodes = ol.sections.get(base)
        if not nodes:
            if base in ol.subs and not subs:
                return "resolved", ol.subs[base].id, None
            # "Section 8" in agreements whose top level is "SECTION 8"
            if "." not in base and not subs and base in ol.articles:
                arts = ol.articles[base]
                if len(arts) > 1:
                    return "ambiguous", None, [a.id for a in arts]
                return "resolved", arts[0].id, None
            return "unresolved", None, None
        if len(nodes) > 1:
            return "ambiguous", None, [n.id for n in nodes]
        if not subs:
            return "resolved", nodes[0].id, None
        sub = ol.subs.get(base + subs)
        if sub is not None:
            return "resolved", sub.id, None
        if ref.relative or ref.base_end <= ref.start:
            return "unresolved", None, None
        return "section_fallback", nodes[0].id, None
    if ref.head == "article":
        key = t.upper()
        as_int = _roman_to_int(key)
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
    for d in decls:
        if d.term not in term_ids:
            term_ids[d.term] = len(terms_json)
            low = " " not in d.term and d.term.lower() in COMMON_WORD_TERMS
            terms_json.append({"id": term_ids[d.term], "term": d.term, "decls": [], "confidence": "low" if low else "high"})
        entry: dict[str, Any] = {"anchor": {"s": d.tn.s, "t": d.tn.t}, "span": d.tn.span(d.start, d.end), "kind": d.kind, "in": _container_label(d.tn)}
        if d.pointer:
            entry["pointer"] = d.pointer
        terms_json[term_ids[d.term]]["decls"].append(entry)
    forms: dict[str, int] = {}
    for term, tid in term_ids.items():
        for f in _term_forms(term):
            forms.setdefault(f, tid)
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
        for r in _find_refs(tn.text, in_exhibit=in_exhibit):
            if is_heading and r.start == len(tn.text) - len(tn.text.lstrip()):
                continue  # the heading "Annex 3.7" itself, not a reference to it
            status, target, cands = _resolve(r, ol, tn, sec_text)
            if status == "resolved" and r.head in ("clause", "paragraph") and r.target.startswith("(") and tn.s in definition_containers:
                status, target, r.detail = "unresolved", None, "inline"
            end = r.base_end if status == "section_fallback" else r.end
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
