"""Span re-anchoring pilot for docs/plans/span-taxonomy.md.

Projects each agreement XML version onto the canonical text layer (body <text> and
<definition> leaves, one per line, page markers skipped), samples sentence-length
spans on each older version, and measures how many land on the same text in the
latest version by raw offset, by section-relative offset, and by quote + context
re-anchoring. Throwaway research code: it reads a dump and writes nothing.

Dump (read-only; base64 sidesteps the CLI's escaping of XML text):

    ./bin/mariadb-read "SELECT x.agreement_uuid, x.version, x.latest, x.status, TO_BASE64(x.xml)
      FROM xml x JOIN (SELECT a.agreement_uuid FROM xml a JOIN xml b
        ON b.agreement_uuid=a.agreement_uuid AND b.latest=1 AND b.status='verified'
        GROUP BY a.agreement_uuid HAVING COUNT(*)>=2 ORDER BY RAND(7) LIMIT 40) t
      ON t.agreement_uuid=x.agreement_uuid" > multi.tsv

Usage: python3 anchor_pilot.py multi.tsv
"""
import base64, csv, sys, re, random, difflib, json
from collections import defaultdict
import xml.etree.ElementTree as ET
csv.field_size_limit(10**9)
LEAF = {"text", "definition"}

def project(xml):
    root = ET.fromstring(xml)
    body = root.find("body")
    parts, pos, secs = [], 0, {}
    def walk(el, sec):
        nonlocal pos
        if el.tag == "section":
            sec = el.get("uuid")
            start = pos
        if el.tag in LEAF:
            t = "".join(el.itertext()).strip()
            if t:
                parts.append(t + "\n"); pos += len(t) + 1
        for ch in el:
            walk(ch, sec)
        if el.tag == "section":
            secs[sec] = (start, pos, el.get("title"))
    if body is not None:
        walk(body, None)
    return "".join(parts), secs

rows = defaultdict(dict)
recs, cur = [], None
with open(sys.argv[1]) as f:
    next(f)
    for line in f:
        line = line.rstrip("\n")
        if re.match(r"^[0-9a-f-]{36}\t", line):
            if cur: recs.append(cur)
            cur = line.split("\t")
        else:
            cur[-1] += line
    recs.append(cur)
for a, v, latest, status, b64 in recs:
    rows[a][int(v)] = (int(latest), status, base64.b64decode(b64.replace("\\n", "")).decode())

random.seed(1)
stats = defaultdict(int)
per = []
for a, vers in rows.items():
    vs = sorted(vers)
    latest_v = max(v for v in vs if vers[v][0] == 1) if any(vers[v][0] for v in vs) else vs[-1]
    try:
        new_text, new_secs = project(vers[latest_v][2])
    except ET.ParseError:
        stats["parse_error"] += 1; continue
    for v in vs:
        if v == latest_v: continue
        try:
            old_text, old_secs = project(vers[v][2])
        except ET.ParseError:
            stats["parse_error"] += 1; continue
        stats["pairs"] += 1
        same = old_text == new_text
        stats["identical_text"] += same
        sec_kept = sum(1 for s in old_secs if s in new_secs)
        stats["old_secs"] += len(old_secs); stats["secs_uuid_kept"] += sec_kept
        # sample spans: 40 random sentence-ish spans of 80-400 chars from old text
        cands = [m for m in re.finditer(r"[^.\n]{80,400}[.;]", old_text)]
        sample = random.sample(cands, min(40, len(cands)))
        r = defaultdict(int)
        for m in sample:
            s, e = m.start(), m.end(); quote = old_text[s:e]
            r["n"] += 1
            r["offset_ok"] += new_text[s:e] == quote
            # section-relative offset
            sec = next((u for u, (ss, ee, _) in old_secs.items() if ss <= s < ee), None)
            if sec and sec in new_secs:
                d = s - old_secs[sec][0]; ns = new_secs[sec][0] + d
                r["secrel_ok"] += new_text[ns:ns + len(quote)] == quote
            hits = [h.start() for h in re.finditer(re.escape(quote), new_text)]
            if len(hits) == 1: r["quote_unique"] += 1
            elif len(hits) > 1:
                pre = old_text[max(0, s - 64):s]; suf = old_text[e:e + 64]
                ctx = [h for h in hits if new_text[max(0, h - 64):h] == pre or new_text[h + len(quote):h + len(quote) + 64] == suf]
                if len(ctx) == 1: r["quote_ctx_unique"] += 1
                else:
                    # tiebreak: hit nearest the old offset scaled by length ratio
                    exp = s * len(new_text) / max(1, len(old_text))
                    best = min(hits, key=lambda h: abs(h - exp))
                    r["quote_nearest_correct" if best == s + (len(new_text) - len(old_text)) or new_text[max(0,best-64):best] == pre else "quote_nearest_unsure"] += 1
            else:
                # fuzzy: normalize whitespace/case
                norm = lambda x: re.sub(r"\s+", " ", x).lower()
                r["fuzzy_norm_hit" if norm(quote) in norm(new_text) else "lost"] += 1
        for k, val in r.items(): stats["span_" + k] += val
        per.append({"a": a, "v": v, "latest": latest_v, "identical": same, "len_old": len(old_text), "len_new": len(new_text),
                    "ratio": round(difflib.SequenceMatcher(None, old_text[:20000], new_text[:20000], autojunk=False).quick_ratio(), 3),
                    "secs_old": len(old_secs), "secs_new": len(new_secs), "secs_kept": sec_kept, **r})
print(json.dumps(dict(stats), indent=1))
for p in per: print(json.dumps(p))
