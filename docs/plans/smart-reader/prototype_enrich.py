"""Smart-reader prototype: index sections and subsections, defined terms and
cross-references in Pandects agreement XML, and report how many resolve.

Throwaway research code for docs/plans/smart-reader.md. It reads XML files
only; it writes nothing to any database and is not wired into the ETL.

Dump a latest XML locally (HEX avoids the CLI's curly-quote mangling):

    ./bin/mariadb-read "SELECT HEX(xml) FROM xml WHERE latest=1 AND agreement_uuid='<uuid>'" \
      | tail -n +2 | python3 -c "import sys;sys.stdout.buffer.write(bytes.fromhex(sys.stdin.read().strip()))" > <uuid>.xml

Usage: python prototype_enrich.py XML_FILE [XML_FILE ...] [--json OUT] [--samples N]

"Resolved" here means the reference found a target that exists in the index.
It does not mean the target was checked as correct; precision needs the gold
set described in the plan.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from dataclasses import dataclass, field

ROMAN = {"i": 1, "v": 5, "x": 10, "l": 50, "c": 100}


def roman_to_int(s: str) -> int | None:
    s = s.lower()
    if not s or any(ch not in ROMAN for ch in s):
        return None
    total, prev = 0, 0
    for ch in reversed(s):
        v = ROMAN[ch]
        total = total - v if v < prev else total + v
        prev = max(prev, v)
    return total


# ---------------------------------------------------------------- structure

ARTICLE_NUM = re.compile(
    r"^\s*(?:ARTICLE|Article|PART|Part|SECTION|Section)\s+([IVXLC]+|\d+|[A-Z])\b|^\s*(\d+)\s*[.:]?\s+\D",
)
SECTION_NUM = re.compile(
    r"^\s*(?:Section|SECTION|Sec\.|§)?\s*(\d+[A-Z]?(?:\.\d+[A-Z]?)+|\d+[A-Z]?)\.?(?=\s|$)",
)
# A paragraph that opens with one or more enumerators: "(a)", "(a)(i)", "(ii) ..."
ENUM_PREFIX = re.compile(r"^\s*((?:\([A-Za-z0-9]{1,6}\)\s*)+)")
ENUM_TOKEN = re.compile(r"\(([A-Za-z0-9]{1,6})\)")


def norm_num(num: str) -> str:
    """'7.02' and '7.2' should collide; keep letters like 5A."""
    parts = []
    for p in num.split("."):
        m = re.match(r"0*(\d+)([A-Z]?)$", p)
        parts.append(f"{int(m.group(1))}{m.group(2)}" if m else p)
    return ".".join(parts)


LOWER_ROMANS = {"i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x", "xi", "xii", "xiii", "xiv", "xv", "xvi", "xvii", "xviii", "xix", "xx"}


def enum_kind(tok: str) -> str:
    if tok.isdigit():
        return "digit"
    if tok.islower():
        return "roman" if tok in LOWER_ROMANS else "alpha"
    return "ROMAN" if tok.lower() in LOWER_ROMANS and len(tok) > 1 else "ALPHA"


@dataclass
class Node:
    kind: str  # article | section | sub
    label: str  # canonical key, e.g. "7.2" or "7.2(b)(ii)"
    uuid: str | None
    title: str
    parent: str | None = None


@dataclass
class Index:
    articles: dict[str, Node] = field(default_factory=dict)  # "4" -> node (roman normalized to int)
    sections: dict[str, Node] = field(default_factory=dict)  # "7.2" -> node
    subs: dict[str, Node] = field(default_factory=dict)  # "7.2(b)(ii)" -> node
    dup_sections: set[str] = field(default_factory=set)
    unnumbered_sections: int = 0
    attachments: dict[str, str] = field(default_factory=dict)  # "exhibit a" -> heading text


def classify_tokens(tokens: list[str], prev_by_level: list[tuple[str, str]]) -> list[str]:
    """Pick alpha vs roman for ambiguous single letters (i, v, x) using the
    previous sibling at the same level."""
    kinds = []
    for depth, tok in enumerate(tokens):
        k = enum_kind(tok)
        if tok in ("i", "v", "x") and depth < len(prev_by_level):
            pk, ptok = prev_by_level[depth]
            if pk == "alpha" and ord(tok) == ord(ptok) + 1:
                k = "alpha"
        kinds.append(k)
    return kinds


def build_index(root: ET.Element) -> Index:
    idx = Index()
    body = root.find("body")
    if body is None:
        return idx
    for art in body.iter("article"):
        title = art.get("title") or ""
        m = ARTICLE_NUM.match(title)
        if m:
            raw = m.group(1) or m.group(2)
            key = str(roman_to_int(raw)) if roman_to_int(raw) and not raw.isdigit() else raw
            idx.articles[key.upper()] = Node("article", key, art.get("uuid"), title)
        for sec in art.iter("section"):
            stitle = sec.get("title") or ""
            sm = SECTION_NUM.match(stitle)
            if not sm:
                idx.unnumbered_sections += 1
                continue
            num = norm_num(sm.group(1))
            if num in idx.sections:
                idx.dup_sections.add(num)
                continue
            idx.sections[num] = Node("section", num, sec.get("uuid"), stitle)
            # subsection tree from paragraph enumerators
            stack: list[tuple[str, str]] = []  # (kind, token) per level
            for t in sec.iter("text"):
                txt = t.text or ""
                dm = re.match(r"^\s*(\d+(?:\.\d+){2,})\.?\s", txt)
                if dm and norm_num(dm.group(1)).startswith(num + "."):
                    k = norm_num(dm.group(1))
                    idx.subs.setdefault(k, Node("sub", k, sec.get("uuid"), txt[:80], num))
                    continue
                pm = ENUM_PREFIX.match(txt)
                if not pm:
                    continue
                tokens = ENUM_TOKEN.findall(pm.group(1))
                kinds = classify_tokens(tokens, stack)
                # place the first token: find level whose kind matches, else push
                level = next((i for i, (k, _) in enumerate(stack) if k == kinds[0]), len(stack))
                stack = stack[:level]
                for k, tok in zip(kinds, tokens):
                    stack.append((k, tok))
                key = num + "".join(f"({tok})" for _, tok in stack)
                idx.subs.setdefault(key, Node("sub", key, sec.get("uuid"), txt[:80], num))
    # attachments: exhibits/schedules/annexes mentioned as headings in back matter
    back = root.find("backMatter")
    if back is not None:
        for t in back.iter("text"):
            txt = (t.text or "").strip()
            m = re.match(r"^(EXHIBIT|Exhibit|ANNEX|Annex|SCHEDULE|Schedule|APPENDIX|Appendix)\s+([A-Z0-9][A-Z0-9.\-]*)\b", txt)
            if m and len(txt) < 120:
                idx.attachments.setdefault(f"{m.group(1).lower()} {m.group(2).lower().rstrip('.')}", txt)
    return idx


# ---------------------------------------------------------------- definitions

Q_OPEN = "“\""
Q_CLOSE = "”\""
TERM = r"[A-Z0-9][^“”\"]{0,80}?"
# "Foo" means / shall mean / has the meaning / shall have the meaning
DEF_MEANS = re.compile(
    rf"[{Q_OPEN}]({TERM})[{Q_CLOSE}]\s*(?:\([^)]*\)\s*)?(?:,\s*)?(?:shall\s+)?(?:means?|has the meaning|shall have the (?:respective )?meanings?|have the meanings?|is defined|as used herein)",
)
# (the "Company"), (each, a "Party"), (collectively, the "Sellers"), ("Parent")
DEF_PAREN = re.compile(
    rf"\(\s*(?:(?:together|collectively|individually|each|each of which|any such|such|hereinafter|herein)[^{Q_OPEN})]*?\s+)?(?:the\s+|a\s+|an\s+)?[{Q_OPEN}]({TERM})[{Q_CLOSE}](?:\s*(?:or|and)\s+(?:the\s+|a\s+|an\s+)?[{Q_OPEN}]({TERM})[{Q_CLOSE}])?\s*\)",
)
# "Foo" has the meaning set forth in Section 3.2 / defined in Section 3.2
DEF_POINTER = re.compile(
    rf"[{Q_OPEN}]({TERM})[{Q_CLOSE}]\s*(?:shall\s+)?(?:has|have|is)\s+(?:the\s+meanings?\s+(?:set\s+forth|given|ascribed|assigned|specified)\s+(?:to\s+(?:it|such term|them)\s+)?in|defined\s+in)\s+(Section|Article|Exhibit|Schedule|Annex)\s+([0-9A-Z.()]+)",
)


@dataclass
class Definition:
    term: str
    source: str  # element | means | paren | pointer
    where: str  # section key or region
    pointer: str | None = None


def collect_definitions(root: ET.Element, idx: Index) -> list[Definition]:
    defs: list[Definition] = []
    sec_of: dict[ET.Element, str] = {}
    for region in root:
        for sec in region.iter("section"):
            sm = SECTION_NUM.match(sec.get("title") or "")
            for t in sec.iter("text"):
                sec_of[t] = norm_num(sm.group(1)) if sm else "?"
    for region in root:
        rname = region.tag
        for d in region.iter("definition"):
            first = d.find("text")
            txt = (first.text or "") if first is not None else ""
            m = re.match(rf"\s*[{Q_OPEN}]([^{Q_CLOSE}]+)[{Q_CLOSE}]", txt)
            term = m.group(1) if m else (d.get("term") or "")
            defs.append(Definition(term.strip(), "element", sec_of.get(first, rname) if first is not None else rname))
        if rname not in ("body", "backMatter", "frontMatter"):
            continue
        for t in region.iter("text"):
            txt = t.text or ""
            where = sec_of.get(t, rname)
            for m in DEF_POINTER.finditer(txt):
                defs.append(Definition(m.group(1).strip(), "pointer", where, f"{m.group(2)} {m.group(3)}"))
            for m in DEF_MEANS.finditer(txt):
                defs.append(Definition(m.group(1).strip(), "means", where))
            for m in DEF_PAREN.finditer(txt):
                for g in (m.group(1), m.group(2)):
                    if g:
                        defs.append(Definition(g.strip(), "paren", where))
    return defs


def term_regex(terms: list[str]) -> re.Pattern | None:
    if not terms:
        return None
    alts = sorted({re.escape(t) for t in terms if len(t) >= 2}, key=len, reverse=True)
    # longest-first alternation; allow plural and possessive; case-sensitive
    return re.compile(r"(?<![\w“\"])(?:" + "|".join(alts) + r")(?:s|es)?(?:’s|'s)?(?![\w”\"])")


# ---------------------------------------------------------------- xrefs

STATUTE = re.compile(
    r"^\s*(?:\([^)]*\)\s*)*,?\s*(?:of|under|pursuant to)\s+(?:the\s+)?(?:[A-Z]{3,6}\b|Code|DGCL|Exchange Act|Securities Act|HSR Act|ERISA|Treasury Regulations?|Regulation|Rule|Delaware|General Corporation Law|Investment Company Act|Internal Revenue Code|[A-Z][A-Za-z]+ (?:Act|Law|Code|Statutes?)|Title)",
)
OTHER_DOC = re.compile(
    r"^\s*(?:\([^)]*\)\s*)*(?:of|to|in|under)\s+(?:the|such|any|each)\s+(?!Agreement\b)(?:[A-Z][\w\-]*\s+){0,5}(?:Agreement|Letter|Plan|Certificate|Charter|Bylaws|Indenture|Note|Schedule|Statement)\b"
    r"|^\s*(?:\([^)]*\)\s*)*(?:of|to|in)\s+(?:the|its|their|such)\s+(?:[a-z]+\s+){0,3}(?:bylaws|by-laws|charter|certificate of incorporation|articles of incorporation|operating agreement)\b",
)
# borrowed from the kirkland ingester's AUTHORITY_BEFORE_RE: "Code Section 409A", "Treasury Regulations Section 1.368-2"
AUTHORITY_BEFORE = re.compile(r"(?:\b(?:Code|Regulations?|Reg\.|ERISA|Act|Rules?|Treas\.|Treasury|Statute|Law|IRC|U\.S\.C\.|C\.F\.R\.)|§)\s*$")
SEC_FORM = re.compile(r"^(?:\d{2}[A-Z](?:-\d+)?|TO|13E-3)$")
REF_HEAD = re.compile(r"\b(Sections?|Articles?|Exhibits?|Schedules?|Annex(?:es)?|Appendix|Appendices|clauses?|paragraphs?)\s+(?=[\dIVXLC(A-Z])")
REF_ITEM = re.compile(r"(\d+[A-Z]?(?:-\d+)?(?:\.\d+[A-Z]?)*(?:\([A-Za-z0-9]{1,6}\))*|[IVXLC]+\b|[A-Z]\b(?:-\d+)?|\([A-Za-z0-9]{1,6}\)(?:\([A-Za-z0-9]{1,6}\))*)")
REF_JOIN = re.compile(r"\s*(?:,\s*(?:and|or)?\s*|\s+(?:and|or|through|to|and/or)\s+|\s*[\-–]\s*)")


@dataclass
class Ref:
    head: str
    target: str
    is_range_end: bool
    status: str = ""  # section | sub | sub_fallback | article | attachment | external | local_clause | unresolved
    context: str = ""


def find_refs(text: str) -> list[Ref]:
    refs: list[Ref] = []
    runs: list[tuple[list[Ref], int | None]] = []
    for hm in REF_HEAD.finditer(text):
        head = hm.group(1).lower().rstrip("s")
        if head == "annexe":
            head = "annex"
        if head == "appendice":
            head = "appendix"
        pos = hm.end()
        run: list[Ref] = []
        first = True
        range_next = False
        while True:
            if not first:
                jm = REF_JOIN.match(text, pos)
                if not jm:
                    break
                range_next = bool(re.search(r"through|to|[\-–]", jm.group(0)))
                pos = jm.end()
            im = REF_ITEM.match(text, pos)
            if not im:
                break
            if not first and im.group(1).startswith("("):
                # relative member "(c)" only continues the run when the previous item ends in an
                # enumerator of the same kind: "414(b), (c)" yes; "7.6(b), (ii) Everest" / "1.4, (b)" no
                prev = run[-1].target if run else ""
                pm = re.search(r"\(([A-Za-z0-9]{1,6})\)$", prev)
                tok = ENUM_TOKEN.match(im.group(1)).group(1)
                if not pm or enum_kind(pm.group(1)) != enum_kind(tok):
                    break
                run.append(Ref(head, prev[: pm.start()] + im.group(1), range_next, context=text[max(0, hm.start() - 40): im.end() + 60]))
                pos = im.end()
                continue
            run.append(Ref(head, im.group(1), range_next, context=text[max(0, hm.start() - 40): im.end() + 60]))
            pos = im.end()
            first = False
        # a statute/other-document tail applies to the whole run ("Sections 354 and 361 of the Code")
        tail = text[pos: pos + 90]
        before = text[max(0, hm.start() - 40): hm.start()]
        if AUTHORITY_BEFORE.search(before):
            for r in run:
                r.status = "external"
        if STATUTE.match(tail) or OTHER_DOC.match(tail) or re.search(r"\bof\s+(?:the\s+)?[A-Z][\w \-]{2,60}\bAct\b", tail[:70]):
            for r in run:
                r.status = "external"
        qm = re.match(r"\s+(?:of|in)\s+(?:this\s+)?Section\s+(\d+[A-Z]?(?:\.\d+[A-Z]?)*(?:\([A-Za-z0-9]{1,6}\))*)", tail)
        if qm and run and run[0].head in ("clause", "paragraph"):
            for r in run:
                if r.target.startswith("(") and not r.status:
                    r.head, r.target = "section", qm.group(1) + r.target
        for r in run:
            if r.head == "schedule" and SEC_FORM.match(r.target):
                r.status = "external"
        # "Section 355 or Section 361 of the Code": a run joined to an external run is external too
        if runs and runs[-1][1] is not None and re.fullmatch(r"\s*(?:,\s*)?(?:and|or|and/or)?\s*", text[runs[-1][1]: hm.start()]) and run and run[0].status == "external":
            for r in runs[-1][0]:
                r.status = "external"
        runs.append((run, pos))
        refs.extend(run)
    return refs


def resolve(ref: Ref, idx: Index, cur_section: str | None, sec_text: str = "") -> str:
    if ref.status:
        return ref.status
    t = ref.target
    if ref.head == "section":
        m = re.match(r"(\d+[A-Z]?(?:\.\d+[A-Z]?)*)((?:\([A-Za-z0-9]+\))*)$", t)
        if not m:
            return "unresolved"
        base = norm_num(m.group(1))
        subs = m.group(2)
        if base not in idx.sections:
            # "Section 8" in agreements whose top level is "SECTION 8" / Article 8
            if "." not in base and not subs and base in idx.articles:
                return "article"
            return "unresolved"
        if not subs:
            return "section"
        return "sub" if base + subs in idx.subs else "sub_fallback"
    if ref.head == "article":
        key = t.upper()
        key = str(roman_to_int(key)) if roman_to_int(key) else key
        return "article" if key in idx.articles else "unresolved"
    if ref.head in ("exhibit", "schedule", "annex", "appendix"):
        return "attachment" if f"{ref.head} {t.lower()}" in idx.attachments else "attachment_unlocated"
    if ref.head in ("clause", "paragraph"):
        if cur_section and t.startswith("("):
            if any(k.startswith(cur_section) and k.endswith(t) for k in idx.subs):
                return "local_clause"
        if t.startswith("(") and sec_text.count(t) >= 2:
            # inline enumeration in the same section: "(i) ... (ii) ... clause (i) above"
            return "local_inline"
        return "unresolved"
    return "unresolved"


# ---------------------------------------------------------------- report


def analyze(path: str) -> dict:
    root = ET.parse(path).getroot()
    idx = build_index(root)
    defs = collect_definitions(root, idx)
    terms = sorted({d.term for d in defs if d.source != "pointer" and 2 <= len(d.term) <= 80})
    # a term declared in two different places (true redefinition, or a party-specific definition)
    places: dict[str, set[str]] = defaultdict(set)
    for d in defs:
        if d.source != "pointer":
            places[d.term].add(d.where)
    dup_terms = [t for t, w in places.items() if len(w) > 1]
    rx = term_regex(terms)

    uses = 0
    used_terms: set[str] = set()
    ref_status: Counter = Counter()
    head_status: dict[str, Counter] = defaultdict(Counter)
    unresolved_samples: list[str] = []
    body = root.find("body")
    texts: list[tuple[str | None, str, str]] = []
    if body is not None:
        for sec in body.iter("section"):
            sm = SECTION_NUM.match(sec.get("title") or "")
            cur = norm_num(sm.group(1)) if sm else None
            sec_full = " ".join(t.text or "" for t in sec.iter("text"))
            for t in sec.iter("text"):
                texts.append((cur, t.text or "", sec_full))
    for cur, txt, sec_full in texts:
        if rx:
            for m in rx.finditer(txt):
                uses += 1
                used_terms.add(m.group(0))
        for r in find_refs(txt):
            st = resolve(r, idx, cur, sec_full)
            ref_status[st] += 1
            head_status[r.head][st] += 1
            if st in ("unresolved", "sub_fallback", "attachment_unlocated") and len(unresolved_samples) < 400:
                unresolved_samples.append(f"[{st}] {r.head} {r.target} :: {r.context.strip()}")

    total_refs = sum(ref_status.values())
    internal = total_refs - ref_status["external"]
    ok = ref_status["section"] + ref_status["sub"] + ref_status["article"] + ref_status["attachment"] + ref_status["local_clause"] + ref_status["local_inline"]
    return {
        "agreement": path.rsplit("/", 1)[-1].removesuffix(".xml"),
        "articles": len(idx.articles),
        "sections_numbered": len(idx.sections),
        "sections_unnumbered": idx.unnumbered_sections,
        "duplicate_section_numbers": sorted(idx.dup_sections),
        "subsections": len(idx.subs),
        "attachments_located": len(idx.attachments),
        "definitions_by_source": dict(Counter(d.source for d in defs)),
        "distinct_terms": len(terms),
        "terms_defined_twice": len(dup_terms),
        "term_uses_in_body": uses,
        "xrefs_total": total_refs,
        "xrefs_by_status": dict(ref_status),
        "xrefs_by_head": {h: dict(c) for h, c in head_status.items()},
        "internal_resolution_rate": round(ok / internal, 3) if internal else None,
        "section_level_rate": round((ok + ref_status["sub_fallback"]) / internal, 3) if internal else None,
        "unresolved_samples": unresolved_samples,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--json")
    ap.add_argument("--samples", type=int, default=0)
    a = ap.parse_args()
    results = [analyze(f) for f in a.files]
    cols = ["agreement", "sections_numbered", "subsections", "distinct_terms", "term_uses_in_body", "xrefs_total", "internal_resolution_rate", "section_level_rate"]
    print("\t".join(cols))
    for r in results:
        print("\t".join(str(r[c])[:12] for c in cols))
    agg: Counter = Counter()
    for r in results:
        agg.update(r["xrefs_by_status"])
    print("aggregate xref status:", dict(agg))
    internal = sum(agg.values()) - agg["external"]
    ok = sum(agg[k] for k in ("section", "sub", "article", "attachment", "local_clause", "local_inline"))
    print(f"aggregate internal exact={ok/internal:.3f} section-level={(ok+agg['sub_fallback'])/internal:.3f} n={internal}")
    # in-document targets only (sections, articles, clauses); attachments reported apart
    heads: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        for h, c in r["xrefs_by_head"].items():
            heads[h].update(c)
    body_c: Counter = Counter()
    for h in ("section", "article", "clause", "paragraph"):
        body_c.update(heads.get(h, Counter()))
    body_int = sum(body_c.values()) - body_c["external"]
    body_ok = sum(body_c[k] for k in ("section", "sub", "article", "local_clause", "local_inline"))
    print(f"in-document refs: exact={body_ok/body_int:.3f} section-level={(body_ok+body_c['sub_fallback'])/body_int:.3f} n={body_int} external={body_c['external']}")
    att: Counter = Counter()
    for h in ("exhibit", "schedule", "annex", "appendix"):
        att.update(heads.get(h, Counter()))
    print("attachment refs:", dict(att))
    if a.samples:
        for r in results:
            for s in r["unresolved_samples"][: a.samples]:
                print(r["agreement"][:8], s[:220])
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(results, fh, indent=1)


if __name__ == "__main__":
    sys.exit(main())
