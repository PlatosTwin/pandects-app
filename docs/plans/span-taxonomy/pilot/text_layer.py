"""Canonical text layer for the span-taxonomy pilot (docs/plans/span-taxonomy.md, section 6).

The layer is the agreement body (decision Q3), plus defined-terms exhibits when
project(exhibits=True) is used (M2 holdout, plan section 7.5): every <text> leaf,
in document order, followed by "\n". <definition> is a container of <text> leaves
(it never holds direct text), so it is not a leaf; a leaf inside one is tagged
kind="definition" with the defined term.
Leaf text is not stripped, so per-leaf offsets match the smart reader's node text
(PR #186). <page> and <pageUUID> markers are skipped. Each leaf records its section uuid, the
article/section titles, and its [start, end) range in the layer.

Throwaway pilot code: it reads dumps and writes JSON, never the database.
"""
import base64, csv, hashlib, re, sys
import xml.etree.ElementTree as ET

csv.field_size_limit(10**9)
UUID_LINE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\t")


def read_dump(path):
    """Yield rows of a mariadb-read TSV whose last column is TO_BASE64(xml)."""
    cur = None
    with open(path) as f:
        next(f)
        for line in f:
            line = line.rstrip("\n")
            if UUID_LINE.match(line):
                if cur:
                    yield cur
                cur = line.split("\t")
            else:
                cur[-1] += line
    if cur:
        yield cur


def decode(b64):
    return base64.b64decode(b64.replace("\\n", "")).decode()


EXHIBIT_HEAD = re.compile(r"^\s*(EXHIBIT|Exhibit|ANNEX|Annex|APPENDIX|Appendix|SCHEDULE|Schedule)\s+[A-Z0-9][\w.\-]*\s*$")
DEFINES = re.compile(r"^\s*(\([a-z0-9]{1,4}\)\s*)?([^\"“]{0,80}?\.\s+)?[\"“]([^\"”]{2,80})[\"”]"
                     r"(\s+of any \w+)?\s*(shall\s+)?(means?|has the meaning|shall have the meaning|is defined)")
DEFS_HEAD = re.compile(r"^\s*((exhibit|annex|appendix|schedule)\s+[\w.\-]+\s*[-–:]?\s*)?(certain\s+)?(definitions|defined terms)\b", re.I)
DEFS_TITLE = re.compile(r"defin|interpretation", re.I)
LOOSE_HEAD = re.compile(r"^\s*(Section|SECTION)?\s*\d+(\.\d+)+\.?\s+\S")


def project(xml, exhibits=False):
    """Return {"text", "leaves": [...], "sections": [...], "layer_hash"}.

    exhibits=False is the body-only layer the M1 and M2 pilots ran on. exhibits=True
    (M2 holdout) also admits defined-terms exhibits from <backMatter>, and splits body
    text that sits outside any <section> into synthetic sections at numbered headings.
    """
    root = ET.fromstring(xml)
    body = root.find("body")
    parts, leaves, sections, pos = [], [], [], 0

    def add_leaf(t, sec, term):
        nonlocal pos
        leaves.append({"kind": "definition" if term else "text", "term": term,
                       "start": pos, "end": pos + len(t), "section": sec["uuid"] if sec else None})
        parts.append(t + "\n")
        pos += len(t) + 1

    loose = {"sec": None, "n": 0}

    def close_loose():
        if loose["sec"] is not None:
            loose["sec"]["end"] = pos
            sections.append(loose["sec"])
            loose["sec"] = None

    def loose_section(t, article):
        # body text outside any <section>: start a synthetic section at each numbered heading
        cur = loose["sec"]
        if cur is None or cur["article"] != article or LOOSE_HEAD.match(t):
            close_loose()
            loose["n"] += 1
            cur = loose["sec"] = {"uuid": f"loose:{loose['n']}", "title": t[:80], "article": article, "start": pos}
        return cur

    def walk(el, article, sec, term):
        if el.tag == "definition":
            term = el.get("term")
        if el.tag == "article":
            article = el.get("title")
        if el.tag == "section":
            close_loose()
            sec = {"uuid": el.get("uuid"), "title": el.get("title"), "article": article, "start": pos}
        if el.tag == "text":
            t = el.text or ""
            if t.strip():
                add_leaf(t, sec if sec or not exhibits else loose_section(t, article), term)
        for ch in el:
            walk(ch, article, sec, term)
        if el.tag == "section":
            sec["end"] = pos
            sections.append(sec)

    if body is not None:
        walk(body, None, None, None)
    close_loose()
    back = root.find("backMatter") if exhibits else None
    if back is not None:
        for i, block in enumerate(exhibit_blocks(back)):
            if not is_definitions_block(block):
                continue
            sec = {"uuid": f"exhibit:{i}", "title": " ".join(block[:2])[:120], "article": "EXHIBIT", "start": pos}
            term = None
            for t in block:
                # a definition's continuation lines (carve-out clauses) belong to its term
                m = DEFINES.match(t)
                term = m.group(3) if m else term
                add_leaf(t, sec, term)
            sec["end"] = pos
            sections.append(sec)
    text = "".join(parts)
    return {"text": text, "leaves": leaves, "sections": sections,
            "layer_hash": hashlib.sha256(text.encode()).hexdigest()[:16]}


def exhibit_blocks(back):
    """Split <backMatter> text lines into blocks, one per EXHIBIT/ANNEX/APPENDIX/SCHEDULE heading."""
    blocks = [[]]
    for el in back.iter("text"):
        t = el.text or ""
        if not t.strip():
            continue
        if EXHIBIT_HEAD.match(t) and blocks[-1]:
            blocks.append([])
        blocks[-1].append(t)
    return [b for b in blocks if b]


def is_definitions_block(block):
    """A defined-terms exhibit: titled as definitions near its top, or mostly "X" means ... lines."""
    if any(DEFS_HEAD.match(t) and len(t) < 80 for t in block[:4]):
        return True
    defs = sum(1 for t in block if DEFINES.match(t))
    return defs >= 5 and defs >= 0.3 * len(block)


def norm(s):
    return re.sub(r"\s+", " ", s).strip().lower()


def repeated_leaves(layer, min_len=80):
    """(long leaves, long leaves whose normalized text repeats an earlier one)."""
    seen, dups, n = set(), 0, 0
    for lf in layer["leaves"]:
        t = norm(layer["text"][lf["start"]:lf["end"]])
        if len(t) < min_len:
            continue
        n += 1
        if t[:150] in seen:
            dups += 1
        seen.add(t[:150])
    return n, dups
