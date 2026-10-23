# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false
"""Gold-set tooling for the reader index (docs/plans/smart-reader.md, Section 9).

``build_packet`` turns one agreement XML into the labeling packet described in
docs/plans/smart-reader/gold_rubric.md, plus a map from paragraph ids to
anchors. ``score`` compares an enricher payload with adjudicated gold labels.

    python -m etl.utils.reader_gold packet XML_FILE OUT_DIR
    python -m etl.utils.reader_gold score XML_FILE GOLD_JSON
"""

from __future__ import annotations

import json
import random
import re
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from etl.domain.reader_index import (
    ARTICLE_NUM,
    SECTION_NUM,
    _collect_texts,  # pyright: ignore[reportPrivateUsage]
    _glossary_exhibit_texts,  # pyright: ignore[reportPrivateUsage]
    _norm_num,  # pyright: ignore[reportPrivateUsage]
    _roman_to_int,  # pyright: ignore[reportPrivateUsage]
    _Text,  # pyright: ignore[reportPrivateUsage]
    build_reader_index,
)

SAMPLED_SECTIONS = 4
DEFINITIONS_MAX_PARAGRAPHS = 40
SECTION_MAX_CHARS = 8000
_ATTACHMENT_LINE = re.compile(r"^\s*(exhibit|annex|schedule|appendix)\s+[A-Z0-9]", re.IGNORECASE)


def _article_id(title: str) -> str:
    m = ARTICLE_NUM.match(title)
    if not m:
        return "?"
    raw = m.group(1) or m.group(2)
    as_int = _roman_to_int(raw) if not raw.isdigit() else None
    return "a" + (str(as_int) if as_int else raw).upper()


def _section_id(title: str) -> str:
    m = SECTION_NUM.match(title)
    return "s" + _norm_num(m.group(1)) if m else "?"


def build_packet(xml: str, agreement_uuid: str) -> tuple[str, dict[str, dict[str, Any]]]:
    root = ET.fromstring(xml)
    texts, _ = _collect_texts(root)
    by_container: dict[str, list[_Text]] = {}
    for tn in texts:
        by_container.setdefault(tn.s, []).append(tn)

    outline: list[str] = []
    sections: list[tuple[str, str, ET.Element]] = []  # (id, title, element)
    body = root.find("body")
    if body is not None:
        for el in body.iter():
            if el.tag == "article":
                outline.append(f"{_article_id(el.get('title') or '')} | {el.get('title') or ''}")
            elif el.tag == "section" and el.get("uuid"):
                sid = _section_id(el.get("title") or "")
                outline.append(f"  {sid} | {el.get('title') or ''}")
                sections.append((sid, el.get("title") or "", el))
    for tn in texts:
        if tn.region == "backMatter" and len(tn.text.strip()) < 120 and _ATTACHMENT_LINE.match(tn.text):
            outline.append(f"(back matter) {tn.text.strip()}")

    def sec_texts(el: ET.Element) -> list[_Text]:
        return by_container.get(el.get("uuid") or "", [])

    defs_title = ""
    defs_paras: list[_Text] = []
    defs_el: ET.Element | None = None
    for _sid, title, el in sections:
        if re.search(r"defin", title, re.IGNORECASE) and len(sec_texts(el)) >= 3:
            defs_el, defs_title, defs_paras = el, title, sec_texts(el)
            break
    if not defs_paras:
        glossary = _glossary_exhibit_texts(texts)
        defs_paras = [tn for tn in texts if id(tn) in glossary]
        defs_title = "(back-matter glossary)"
    if not defs_paras and sections:
        # no definitions article or glossary (older filings define terms inline): use the
        # section with the most quoted phrases
        def quoted(el: ET.Element) -> int:
            return sum(len(re.findall(r"[“\"][A-Z][^”\"]{0,80}[”\"]", tn.text)) for tn in sec_texts(el))

        _sid, defs_title, defs_el = max(sections, key=lambda x: quoted(x[2]))
        defs_paras = sec_texts(defs_el)
        defs_title = f"(most quoted terms) {defs_title}"
    defs_paras = defs_paras[:DEFINITIONS_MAX_PARAGRAPHS]

    eligible = [
        (sid, title, el)
        for sid, title, el in sections
        if el is not defs_el and sum(len(tn.text) for tn in sec_texts(el)) >= 400
    ]
    rng = random.Random(agreement_uuid)
    sampled = sorted(rng.sample(eligible, min(SAMPLED_SECTIONS, len(eligible))), key=lambda x: sections.index(x))

    para_map: dict[str, dict[str, Any]] = {}
    lines = [f"# Packet for {agreement_uuid}", "", "## Outline", "", *outline, ""]

    def emit(label: str, paras: list[_Text], cap: int | None, excerpt: str) -> None:
        lines.extend([f"## {label}", ""])
        used = 0
        for tn in paras:
            if cap is not None and used + len(tn.text) > cap and used > 0:
                lines.append("[... section truncated ...]")
                break
            pid = f"P{len(para_map) + 1}"
            para_map[pid] = {"s": tn.s, "t": tn.t, "excerpt": excerpt}
            lines.append(f"[{pid}] {tn.text.strip()}")
            used += len(tn.text)
        lines.append("")

    emit(f"Definitions excerpt: {defs_title}", defs_paras, None, "definitions")
    for sid, title, el in sampled:
        emit(f"Sampled section {sid}: {title}", sec_texts(el), SECTION_MAX_CHARS, "sampled")
    return "\n".join(lines), para_map


# ---------------------------------------------------------------- scoring


def _locate(text: str, context: str, mention: str) -> int | None:
    i = text.find(context)
    if i < 0:
        return None
    # the mention as a whole token: "A" in "Annex A" is not the "A" of "Annex", and "1" in
    # "Schedule 1" is not the "1" of "1.1"
    m = re.search(rf"(?<![\w.]){re.escape(mention)}(?!\w|\.\d)", context)
    return None if m is None else i + m.start()


def _id_section(target: str) -> str:
    """'s7.2(b)(ii)' -> 's7.2'; other ids unchanged."""
    m = re.match(r"(s[0-9A-Z.]+)", target)
    return m.group(1) if m else target


def score(xml: str, gold: dict[str, Any], para_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload = build_reader_index(xml)
    root = ET.fromstring(xml)
    texts, _ = _collect_texts(root)
    text_of = {(tn.s, tn.t): tn.text for tn in texts}
    linked = {"resolved", "section_fallback"}

    # index enricher xrefs by (anchor, end offset of the printed number); spans are UTF-16,
    # so packets with non-BMP text would need conversion (none in the pilot)
    sys_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    for x in payload["xrefs"]:
        a = x["at"]
        sys_by_key[(a["s"], a["t"], x["span"][1])] = x

    c: Counter[str] = Counter()
    misses: list[dict[str, Any]] = []
    matched_sys: set[tuple[str, int, int]] = set()
    for g in gold.get("xrefs", []):
        anchor = para_map[g["para"]]
        txt = text_of[(anchor["s"], anchor["t"])]
        pos = _locate(txt, g["context"], g["mention"])
        if pos is None:
            c["gold_unlocated"] += 1
            continue
        end = pos + len(g["mention"])
        internal = not g["target"] in ("external", "disclosure_schedule", "attachment_local", "inline", "not_in_filing")
        c["gold_internal" if internal else "gold_no_target"] += 1
        # the enricher's span may stop early on section fallback: accept any span ending inside the mention
        hit = next(
            (k for k in ((anchor["s"], anchor["t"], e) for e in range(pos + 1, end + 1)) if k in sys_by_key),
            None,
        )
        x = sys_by_key.get(hit) if hit else None
        if hit:
            matched_sys.add(hit)
        is_linked = x is not None and x["status"] in linked
        if is_linked and x is not None:
            c["sys_linked"] += 1
            if internal and x.get("target") == g["target"]:
                c["link_exact"] += 1
            elif internal and _id_section(x.get("target", "")) == _id_section(g["target"]):
                c["link_section"] += 1
            else:
                c["link_wrong"] += 1
                misses.append({"kind": "wrong_link", "gold": g, "sys": x})
        elif internal:
            misses.append({"kind": "missed", "gold": g, "sys_status": x["status"] if x else None})
    # enricher links on mentions the gold does not list at all (only inside labeled paragraphs)
    labeled = {(para_map[p]["s"], para_map[p]["t"]) for p in para_map}
    for g in gold.get("excluded_xrefs", []):
        if g.get("para") not in para_map:
            continue
        anchor = para_map[g["para"]]
        pos = _locate(text_of[(anchor["s"], anchor["t"])], g.get("context", ""), g.get("mention", ""))
        if pos is not None:
            matched_sys.update((anchor["s"], anchor["t"], e) for e in range(pos + 1, pos + len(g["mention"]) + 1))
    for k, x in sys_by_key.items():
        if (k[0], k[1]) in labeled and k not in matched_sys and x["status"] in linked:
            c["sys_linked_unlabeled"] += 1
            misses.append({"kind": "unlabeled_link", "sys": x})

    linked_n = c["sys_linked"] + c["sys_linked_unlabeled"]
    out: dict[str, Any] = {
        "counts": dict(c),
        "xref_precision_exact_or_section": round((c["link_exact"] + c["link_section"]) / linked_n, 4) if linked_n else None,
        "xref_precision_exact": round(c["link_exact"] / linked_n, 4) if linked_n else None,
        "xref_recall_internal": round((c["link_exact"] + c["link_section"]) / c["gold_internal"], 4) if c["gold_internal"] else None,
        "errors": misses,
    }

    # declarations: real ones (means/paren/element) as (term, anchor) sets. Pointers ("has the
    # meaning set forth in Section 2.3") are scored as coverage: the enricher drops a pointer when the
    # term has a real declaration elsewhere, so the question is only whether it knows the term.
    sys_real = {
        (t["term"], d["anchor"]["s"], d["anchor"]["t"]) for t in payload["terms"] for d in t["decls"] if d["kind"] != "pointer"
    }
    gold_decls = gold.get("declarations", [])
    gold_real = {(d["term"], para_map[d["para"]]["s"], para_map[d["para"]]["t"]) for d in gold_decls if d["kind"] != "pointer"}
    sys_in_scope = {d for d in sys_real if (d[1], d[2]) in labeled}
    tp = len(gold_real & sys_in_scope)
    out["decl_precision"] = round(tp / len(sys_in_scope), 4) if sys_in_scope else None
    out["decl_recall"] = round(tp / len(gold_real), 4) if gold_real else None
    out["decl_missed"] = sorted(t for t, *_ in gold_real - sys_in_scope)
    out["decl_extra"] = sorted(t for t, *_ in sys_in_scope - gold_real)
    known_terms = {t["term"] for t in payload["terms"]}
    gold_pointers = {d["term"] for d in gold_decls if d["kind"] == "pointer"}
    out["pointer_term_coverage"] = round(len(gold_pointers & known_terms) / len(gold_pointers), 4) if gold_pointers else None
    out["pointer_terms_unknown"] = sorted(gold_pointers - known_terms)

    # term uses: counts per (term, paragraph)
    term_id = {t["term"]: t["id"] for t in payload["terms"]}
    tu_tp = tu_sys = tu_gold = 0
    gold_terms = {u["term"] for u in gold.get("term_uses", [])}
    gold_counts = {(u["term"], para_map[u["para"]]["s"], para_map[u["para"]]["t"]): int(u["count"]) for u in gold.get("term_uses", [])}
    sampled = {(a["s"], a["t"]) for a in para_map.values() if a.get("excerpt", "sampled") == "sampled"}
    for term in gold_terms:
        tid = term_id.get(term)
        for (s, t) in sampled:
            sys_n = sum(1 for u in payload["uses"].get(f"{s}/{t}", []) if u[2] == tid) if tid is not None else 0
            g_n = gold_counts.get((term, s, t), 0)
            tu_tp += min(sys_n, g_n)
            tu_sys += sys_n
            tu_gold += g_n
    # term uses are labeled only in the sampled sections (rubric 3)
    out["term_use_precision"] = round(tu_tp / tu_sys, 4) if tu_sys else None
    out["term_use_recall"] = round(tu_tp / tu_gold, 4) if tu_gold else None
    return out


def adjudicate(draft: dict[str, Any], check: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Agreed items become gold; every disagreement, removal or addition is a dispute for the
    user to rule on. Disputed xrefs are listed in ``excluded_xrefs`` so scoring ignores links there."""
    gold: dict[str, Any] = {"agreement_uuid": draft.get("agreement_uuid"), "rubric": draft.get("rubric", "v1")}
    disputes: list[dict[str, Any]] = []
    flagged: dict[tuple[str, int], dict[str, Any]] = {}
    for d in check.get("disagree", []):
        flagged[(d["section"], int(d["index"]))] = {"type": "disagree", **d}
    for d in check.get("remove", []):
        flagged[(d["section"], int(d["index"]))] = {"type": "remove", **d}
    excluded: list[dict[str, Any]] = []
    for section in ("declarations", "xrefs", "term_uses"):
        kept = []
        for i, item in enumerate(draft.get(section, [])):
            f = flagged.get((section, i))
            if f is None:
                kept.append(item)
                continue
            if f["type"] == "disagree" and set(f["fix"]) <= {"context"}:
                # a quote-length fix does not change the answer: apply it, no ruling needed
                kept.append({**item, **f["fix"]})
                continue
            proposed = {**item, **f["fix"]} if f["type"] == "disagree" else None
            disputes.append({"section": section, "draft": item, "checker": proposed, "type": f["type"], "reason": f.get("reason", "")})
            if section == "xrefs":
                excluded.append(item)
        gold[section] = kept
    for m in check.get("missing", []):
        disputes.append({"section": m["section"], "draft": None, "checker": m["item"], "type": "missing", "reason": m.get("reason", "")})
        if m["section"] == "xrefs":
            excluded.append(m["item"])
    gold["excluded_xrefs"] = excluded
    return gold, disputes


def main(argv: list[str]) -> None:
    cmd = argv[0]
    xml = Path(argv[1]).read_text(encoding="utf-8")
    uuid = ET.fromstring(xml).get("uuid") or Path(argv[1]).stem
    if cmd == "packet":
        out = Path(argv[2])
        text, para_map = build_packet(xml, uuid)
        _ = (out / f"{uuid}.packet.md").write_text(text, encoding="utf-8")
        _ = (out / f"{uuid}.paramap.json").write_text(json.dumps(para_map), encoding="utf-8")
        print(f"{uuid}: {len(para_map)} paragraphs, {len(text)} chars")
    elif cmd == "score":
        gold = json.loads(Path(argv[2]).read_text(encoding="utf-8"))
        print(json.dumps(score(xml, gold, gold["para_map"]), indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
