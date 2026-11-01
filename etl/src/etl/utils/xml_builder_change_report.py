"""Read-only report: which served agreements would get different XML from a builder change.

The XML is rebuilt in memory from the current pages / tagged outputs with the
same path the pipeline uses (safe tag repairs, then ``generate_xml``, then the
hard rules). Nothing is written to the database.

Run it once per code version, then compare the two dumps:

    # old code (an exported copy of origin/main's etl/src) and new code
    PYTHONPATH=<old etl/src> python -m etl.utils.xml_builder_change_report dump --out old.jsonl
    PYTHONPATH=<new etl/src> python -m etl.utils.xml_builder_change_report dump --out new.jsonl
    python -m etl.utils.xml_builder_change_report compare old.jsonl new.jsonl --out affected.tsv

The MARIADB_* variables from backend/.env must be set. "Served" means the
agreement has rows in latest_sections_search.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import multiprocessing as mp
import os
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from typing import Any

PAGE_SQL = """
SELECT p.agreement_uuid, p.page_uuid, p.page_order, p.raw_page_content,
  coalesce(p.gold_label, p.source_page_type) AS source_page_type,
  CASE WHEN coalesce(p.gold_label, p.source_page_type) = 'body'
       THEN coalesce(tgo.tagged_text_gold, tgo.tagged_text_corrected, tgo.tagged_text, p.processed_page_content)
       ELSE p.processed_page_content END AS tagged_output,
  a.url, a.filing_date, p.source_is_txt, p.source_is_html
FROM pages p JOIN agreements a ON p.agreement_uuid = a.agreement_uuid
LEFT JOIN tagged_outputs tgo ON p.page_uuid = tgo.page_uuid
WHERE p.agreement_uuid = :u ORDER BY p.page_order
"""
SERVED_SQL = "SELECT DISTINCT agreement_uuid FROM latest_sections_search ORDER BY agreement_uuid"

_ENGINE: Any = None


def _engine() -> Any:
    global _ENGINE
    if _ENGINE is None:
        from etl.defs.resources import DBResource

        _ENGINE = DBResource(
            host=os.environ["MARIADB_HOST"],
            port=os.environ.get("MARIADB_PORT", "3306"),
            user=os.environ["MARIADB_USER"],
            password=os.environ["MARIADB_PASSWORD"],
            database=os.environ["MARIADB_DATABASE"],
        ).get_engine()
    return _ENGINE


def _text_len(elem: ET.Element) -> int:
    total = 0
    for child in elem:
        if child.tag in {"text", "definition"} and child.text:
            total += len(child.text)
    return total


def build_record(agreement_uuid: str) -> dict[str, Any]:
    import pandas as pd
    from sqlalchemy import text

    from etl.defs.f_xml_asset import _apply_safe_xml_tag_repairs_to_df, find_hard_rule_violations
    from etl.domain.f_xml import generate_xml

    with _engine().connect() as conn:
        rows = conn.execute(text(PAGE_SQL), {"u": agreement_uuid}).mappings().fetchall()
    df = pd.DataFrame(rows)
    record: dict[str, Any] = {"uuid": agreement_uuid}
    if df.empty:
        record["error"] = "no pages"
        return record
    try:
        repaired = _apply_safe_xml_tag_repairs_to_df(df.copy(), context=None, log_prefix="report")
        xml_rows, failures = generate_xml(repaired, {})
    except Exception as exc:  # noqa: BLE001 - report and keep going
        record["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return record
    if failures:
        record["error"] = "; ".join(f.error for f in failures)[:300]
        return record
    if not xml_rows:
        record["error"] = "no XML generated"
        return record
    xml = xml_rows[0].xml
    root = ET.fromstring(xml)
    headings: list[list[Any]] = []
    body = root.find("body")
    if body is not None:
        for elem in body.iter():
            if elem.tag in {"article", "section"}:
                headings.append([elem.tag, elem.attrib.get("title", ""), _text_len(elem), elem.attrib.get("uuid", "")])
    record["sha1"] = hashlib.sha1(xml if isinstance(xml, bytes) else xml.encode("utf-8")).hexdigest()
    record["headings"] = headings
    record["violations"] = sorted({f"{v.reason_code}: {v.reason_detail}" for v in find_hard_rule_violations(root)})
    return record


def _safe_build(agreement_uuid: str) -> dict[str, Any]:
    try:
        return build_record(agreement_uuid)
    except Exception as exc:  # noqa: BLE001
        return {"uuid": agreement_uuid, "error": f"{type(exc).__name__}: {exc}"[:300]}


def dump(out_path: str, workers: int, limit: int | None, uuids_file: str | None) -> None:
    from sqlalchemy import text

    if uuids_file:
        with open(uuids_file) as fh:
            uuids = [line.strip() for line in fh if line.strip()]
    else:
        with _engine().connect() as conn:
            uuids = [row[0] for row in conn.execute(text(SERVED_SQL))]
    if limit:
        uuids = uuids[:limit]
    done: set[str] = set()
    if os.path.exists(out_path):
        with open(out_path) as fh:
            done = {json.loads(line)["uuid"] for line in fh if line.strip()}
    todo = [u for u in uuids if u not in done]
    print(f"{len(uuids)} agreements, {len(done)} already dumped, {len(todo)} to build", file=sys.stderr)
    # Import by module name so spawned workers can unpickle the function.
    from etl.utils.xml_builder_change_report import _safe_build as build

    with open(out_path, "a") as out, mp.get_context("spawn").Pool(workers) as pool:
        for i, record in enumerate(pool.imap_unordered(build, todo, chunksize=4), 1):
            out.write(json.dumps(record) + "\n")
            if i % 250 == 0:
                out.flush()
                print(f"  {i}/{len(todo)}", file=sys.stderr)


def _load(path: str) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                record = json.loads(line)
                records[record["uuid"]] = record
    return records


def _short(title: str, width: int = 70) -> str:
    title = " ".join(title.split())
    return title if len(title) <= width else title[: width - 3] + "..."


def _only_spaces_added(old: str, new: str) -> bool:
    return old != new and old.replace(" ", "") == new.replace(" ", "")


def classify(old: dict[str, Any], new: dict[str, Any]) -> tuple[list[str], str]:
    """Return (change kinds, one-line summary) for two records of one agreement."""
    if "error" in old or "error" in new:
        return ["build_status"], f"old={old.get('error', 'ok')} new={new.get('error', 'ok')}"
    old_h = old["headings"]
    new_h = new["headings"]
    kinds: set[str] = set()
    parts: list[str] = []
    old_keys = [(tag, " ".join(title.split()).replace(" ", "")) for tag, title, _n, _u in old_h]
    new_keys = [(tag, " ".join(title.split()).replace(" ", "")) for tag, title, _n, _u in new_h]
    matcher = difflib.SequenceMatcher(a=old_keys, b=new_keys, autojunk=False)
    glued = Counter()
    glued_examples: list[str] = []
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal":
            for a, b in zip(old_h[i1:i2], new_h[j1:j2]):
                if a[1] != b[1] and _only_spaces_added(a[1], b[1]):
                    glued[a[0]] += 1
                    if len(glued_examples) < 2:
                        glued_examples.append(f"{_short(a[1], 40)!r}->{_short(b[1], 40)!r}")
                elif a[1] != b[1]:
                    kinds.add("title_changed")
                    parts.append(f"title {_short(a[1])!r}->{_short(b[1])!r}")
                elif a[2] != b[2]:
                    kinds.add("text_moved")
            continue
        removed = old_h[i1:i2]
        added = new_h[j1:j2]
        for item in removed:
            long_title = len(item[1]) > 250
            kinds.add("section_removed_long_title" if long_title else f"{item[0]}_removed")
            parts.append(f"-{item[0]} {_short(item[1])!r} ({len(item[1])} chars)")
        for item in added:
            kinds.add(f"{item[0]}_added")
            parts.append(f"+{item[0]} {_short(item[1])!r} ({len(item[1])} chars)")
    if glued:
        for tag, count in sorted(glued.items()):
            kinds.add(f"glued_{tag}_title")
        parts.insert(
            0,
            "spaces restored in "
            + ", ".join(f"{count} {tag} titles" for tag, count in sorted(glued.items()))
            + (": " + "; ".join(glued_examples) if glued_examples else ""),
        )
    if not kinds and old["sha1"] != new["sha1"]:
        kinds.add("uuid_or_text_only")
        old_uuids = {u for *_rest, u in old_h}
        new_uuids = {u for *_rest, u in new_h}
        parts.append(f"{len(old_uuids - new_uuids)} heading uuids changed")
    if kinds and not kinds <= {"glued_article_title", "glued_section_title", "uuid_or_text_only"}:
        changed_uuids = len({u for *_r, u in old_h if u} - {u for *_r, u in new_h if u})
        parts.append(f"{changed_uuids} old heading uuids not in new")
    return sorted(kinds), " | ".join(parts)


def compare(old_path: str, new_path: str, out_path: str) -> None:
    old = _load(old_path)
    new = _load(new_path)
    kinds_count: Counter[str] = Counter()
    rows = []
    new_violation_rows = 0
    for uuid in sorted(set(old) & set(new)):
        a, b = old[uuid], new[uuid]
        if a.get("sha1") == b.get("sha1") and a.get("error") == b.get("error"):
            continue
        kinds, summary = classify(a, b)
        added_violations = sorted(set(b.get("violations", [])) - set(a.get("violations", [])))
        if added_violations:
            new_violation_rows += 1
        for kind in kinds:
            kinds_count[kind] += 1
        rows.append((uuid, ",".join(kinds), summary, " ; ".join(added_violations)))
    with open(out_path, "w") as fh:
        fh.write("uuid\tchange_kind\tsummary\tnew_hard_rule_violations\n")
        for row in rows:
            fh.write("\t".join(col.replace("\t", " ").replace("\n", " ") for col in row) + "\n")
    print(f"compared {len(set(old) & set(new))} agreements; {len(rows)} changed; {new_violation_rows} with new violations")
    for kind, count in kinds_count.most_common():
        print(f"  {kind}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_dump = sub.add_parser("dump")
    p_dump.add_argument("--out", required=True)
    p_dump.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p_dump.add_argument("--limit", type=int)
    p_dump.add_argument("--uuids-file")
    p_cmp = sub.add_parser("compare")
    p_cmp.add_argument("old")
    p_cmp.add_argument("new")
    p_cmp.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.cmd == "dump":
        dump(args.out, args.workers, args.limit, args.uuids_file)
    else:
        compare(args.old, args.new, args.out)


if __name__ == "__main__":
    main()
