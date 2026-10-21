"""Canonical text layer for the span-taxonomy pilot (docs/plans/span-taxonomy.md, section 6).

The layer is the agreement body only (no exhibits, decision Q3): every <text> leaf,
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


def project(xml):
    """Return {"text", "leaves": [...], "sections": [...], "layer_hash"}."""
    root = ET.fromstring(xml)
    body = root.find("body")
    parts, leaves, sections, pos = [], [], [], 0

    def walk(el, article, sec, term):
        nonlocal pos
        if el.tag == "definition":
            term = el.get("term")
        if el.tag == "article":
            article = el.get("title")
        if el.tag == "section":
            sec = {"uuid": el.get("uuid"), "title": el.get("title"), "article": article, "start": pos}
        if el.tag == "text":
            t = el.text or ""
            if t.strip():
                leaves.append({"kind": "definition" if term else "text", "term": term,
                               "start": pos, "end": pos + len(t),
                               "section": sec["uuid"] if sec else None})
                parts.append(t + "\n")
                pos += len(t) + 1
        for ch in el:
            walk(ch, article, sec, term)
        if el.tag == "section":
            sec["end"] = pos
            sections.append(sec)

    if body is not None:
        walk(body, None, None, None)
    text = "".join(parts)
    return {"text": text, "leaves": leaves, "sections": sections,
            "layer_hash": hashlib.sha256(text.encode()).hexdigest()[:16]}


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
