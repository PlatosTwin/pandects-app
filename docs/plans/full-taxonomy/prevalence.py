"""
Measure how common each M&A concept is in the corpus, for docs/plans/full-taxonomy.md (section 4.4).

It gives two counts of agreements per concept:
  title  the concept is the heading of a section of its own (the rules in title_coverage.py).
         This is a lower bound: a clause inside a larger section is not counted.
  text   a pattern for the concept matches the body of some section (prevalence_patterns.tsv).
         The patterns are hand-written and unreviewed. Several match too much.
The true share lies between the two. This is evidence for the granularity decision. It is not a labeler.

Export the two inputs from the repo root (read-only):
  ./bin/mariadb-read "SELECT LEFT(agreement_uuid,8) a, deal_type, target_type, \
      article_title, section_title, section_standard_ids \
      FROM latest_sections_search" > sections.tsv
  ./bin/mariadb-read "SELECT LEFT(t.agreement_uuid,8) a, t.normalized_text \
      FROM section_text_search t JOIN latest_sections_search l \
      ON l.section_uuid = t.section_uuid" > text.tsv        # about 3 GB

Run (needs ripgrep, `rg`, on the PATH; about 10 minutes):
  etl/.venv/bin/python docs/plans/full-taxonomy/prevalence.py sections.tsv text.tsv [--per-concept]

Each long step prints PROGRESS n/total.
"""

from __future__ import annotations

import argparse
import collections
import csv
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import title_coverage as tc  # noqa: E402  (also puts etl/src on the path)
from etl.domain import taxonomy_pack  # noqa: E402

csv.field_size_limit(10**9)

# A text pattern can be limited to sections that match one of these.
FILTERS = {
    "FILTER_CONDUCT": r"between the date (of this agreement|hereof) and|from the date (of this agreement|hereof) (until|through|to)|prior to the (closing|effective time)|pending the (closing|merger)|during the (interim|pre-closing) period",
    "FILTER_MAE": r"material adverse (effect|change)[^.]{0,40}(means|shall mean)",
    "FILTER_TERM": r"this agreement may be terminated|may terminate this agreement|agreement may be terminated",
    "FILTER_TERM_EFFECT": r"effect of termination|in the event of (the |any )?termination|event of termination|(upon|following) (any |the )?termination of this agreement",
    "FILTER_COND": r"conditions? (to|of|precedent)",
}
BUCKETS = ("50% or more", "10% to 50%", "1% to 10%", "under 1%", "not measured")


def bucket(agreements: int | None, total: int) -> str:
    if agreements is None:
        return "not measured"
    share = agreements / total
    if share >= 0.5:
        return "50% or more"
    if share >= 0.1:
        return "10% to 50%"
    return "1% to 10%" if share >= 0.01 else "under 1%"


def title_reach(sections: Path) -> tuple[dict[str, set[str]], int]:
    """Agreements in which a concept is the target of a title rule, and the number of agreements."""
    reach: dict[str, set[str]] = collections.defaultdict(set)
    agreements: set[str] = set()
    with sections.open(newline="", encoding="utf-8", errors="replace") as handle:
        reader = csv.reader(handle, delimiter="\t", quoting=csv.QUOTE_NONE)
        next(reader)
        for row in reader:
            if len(row) < 5:
                continue
            agreements.add(row[0])
            title = tc.norm_title(row[4])
            if not title:
                continue
            ctx = tc.contexts(row[3])
            for part in re.split(r"\s*;\s*", title):
                target = tc.map_part(part.strip(), ctx) if part.strip() else None
                if target:
                    reach[target].add(row[0])
    return reach, len(agreements)


def text_reach(text: Path, patterns: list[tuple[str, str, str | None]]) -> dict[str, set[str]]:
    """Agreements in which a concept's text pattern matches some section."""
    rg = shutil.which("rg")
    if rg is None:
        raise SystemExit("ripgrep (rg) is not on the PATH")
    ids: list[bytes] = []
    with text.open("rb") as handle:
        for line in handle:
            ids.append(line[: line.find(b"\t")])

    def lines_for(pattern: str) -> set[int]:
        # Byte mode: the text holds bytes that are not valid UTF-8, and in Unicode mode `.` stops at them.
        cmd = [rg, "--no-unicode", "-n", "-o", "-r", "", "--no-filename", "-e", pattern, str(text)]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if proc.returncode not in (0, 1):
            raise RuntimeError(f"rg failed on {pattern[:60]}: {proc.stderr[:200]}")
        return {int(line.split(":", 1)[0]) for line in proc.stdout.split("\n") if line}

    filters = {name: lines_for(pattern) for name, pattern in FILTERS.items()}
    done = 0

    def work(row: tuple[str, str, str | None]) -> tuple[str, set[bytes]]:
        nonlocal done
        concept, pattern, flt = row
        hit = lines_for(pattern)
        if flt:
            hit &= filters[flt]
        done += 1
        print(f"PROGRESS {done}/{len(patterns)} text patterns", file=sys.stderr, flush=True)
        return concept, {ids[n - 1] for n in hit if 1 < n <= len(ids)}  # line 1 is the header

    reach: dict[str, set[str]] = collections.defaultdict(set)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for concept, found in pool.map(work, patterns):
            reach[concept] |= {item.decode("ascii", "replace") for item in found}
    return reach


def read_patterns(path: Path, known: set[str]) -> list[tuple[str, str, str | None]]:
    rows: list[tuple[str, str, str | None]] = []
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        if parts[0] not in known:
            raise SystemExit(f"{path.name}: unknown concept {parts[0]}")
        flt = parts[2] if len(parts) > 2 and parts[2] else None
        if flt is not None and flt not in FILTERS:
            raise SystemExit(f"{path.name}: unknown filter {flt}")
        rows.append((parts[0], parts[1], flt))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument("sections", type=Path, help="section titles, exported as in the docstring")
    parser.add_argument("text", type=Path, help="section text, exported as in the docstring")
    parser.add_argument("--per-concept", action="store_true", help="print one line per concept")
    args = parser.parse_args()

    taxonomy = taxonomy_pack.load()
    concepts = [
        node
        for node in taxonomy.nodes.values()
        if node.pack == "ma" and node.is_concept and node.id not in tc.CATCH_ALL
    ]
    patterns = read_patterns(HERE / "prevalence_patterns.tsv", {node.id for node in concepts})

    print("PROGRESS 1/2 title rules", file=sys.stderr, flush=True)
    by_title, total = title_reach(args.sections)
    print("PROGRESS 2/2 text patterns", file=sys.stderr, flush=True)
    by_text = text_reach(args.text, patterns)

    rows = []
    for node in concepts:
        title = len(by_title[node.id]) if node.id in by_title else None
        text = len(by_text[node.id]) if node.id in by_text else None
        best = max(value for value in (title, text, 0) if value is not None)
        rows.append((node, title, text, best if (title is not None or text is not None) else None))

    print(f"agreements: {total:,}   M&A concepts measured: {len(rows)} (catch-alls left out)")
    print(f"{'share of agreements':<22}{'title only':>12}{'title or text':>15}{'wave 1':>8}{'wave 2':>8}")
    for name in BUCKETS:
        title_n = sum(1 for _, title, _, _ in rows if bucket(title, total) == name)
        best_rows = [node for node, _, _, best in rows if bucket(best, total) == name]
        wave_1 = sum(1 for node in best_rows if node.data.get("wave") == 1)
        print(f"{name:<22}{title_n:>12}{len(best_rows):>15}{wave_1:>8}{len(best_rows) - wave_1:>8}")
    rare = sorted((best, node.id) for node, _, _, best in rows if best is not None and best < 100)
    print("under 100 agreements (title or text):", ", ".join(f"{cid} ({n})" for n, cid in rare) or "none")
    section_level = sum(1 for node, title, _, _ in rows if node.kind == "provision" and (title or 0) >= total / 100)
    print(f"provisions that head a section of their own in at least 1% of agreements: {section_level}")

    if args.per_concept:
        print("\nconcept\tkind\twave\ttitle\ttext")
        for node, title, text, best in sorted(rows, key=lambda row: row[3] or 0):
            print(f"{node.id}\t{node.kind}\t{node.data.get('wave')}\t{'' if title is None else title}\t{'' if text is None else text}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
