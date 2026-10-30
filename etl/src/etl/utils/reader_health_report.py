"""Read-only health report over `reader_index` (docs/plans/smart-reader.md, decision 11.6).

The reader index doubles as an XML-quality signal: an agreement whose sections are
unnumbered, numbered twice or missing cannot resolve its own cross-references. This
report lists those agreements for the XML validation queue. It never changes a
status (11.6: a report, not a verify rule).

A low resolution rate alone is not an XML problem, so it is not a queue entry. On
2026-10-30 most structurally sound agreements below 50% were UK or Australian
drafting ("clause 17", "paragraph 2.1"), or US agreements that number subsections
"Section 5.16.1", which the enricher does not resolve yet. Those go to a separate
file of enricher gaps.

Usage (from etl/):
    uv run python -m etl.utils.reader_health_report --out ../ma_validation_review/reader_health_YYYYMMDD
"""
# pyright: reportAny=false

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import text
from sqlalchemy.engine import Engine

from etl.defs.resources import DBResource
from etl.domain.reader_index import ENRICHER_VERSION
from etl.utils.toc_validation_audit import _build_db_resource, _load_env  # pyright: ignore[reportPrivateUsage]

REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ENV_PATH = REPO_ROOT / "backend" / ".env"

#: This many duplicated section numbers, in an agreement that also resolves
#: poorly, is a parse problem rather than a stray typo or a quoted amendment.
DUPLICATE_SECTIONS_MIN = 3
DUPLICATE_SECTIONS_RATE_MAX = 0.8
#: Below this body section-level rate an agreement is looked at more closely.
LOW_RATE = 0.5

_US_STATES = (
    "Alabama|Alaska|Arizona|Arkansas|California|Colorado|Connecticut|Delaware|Florida|Georgia|Hawaii|Idaho|"
    "Illinois|Indiana|Iowa|Kansas|Kentucky|Louisiana|Maine|Maryland|Massachusetts|Michigan|Minnesota|"
    "Mississippi|Missouri|Montana|Nebraska|Nevada|New Hampshire|New Jersey|New Mexico|New York|North Carolina|"
    "North Dakota|Ohio|Oklahoma|Oregon|Pennsylvania|Rhode Island|South Carolina|South Dakota|Tennessee|Texas|"
    "Utah|Vermont|Virginia|Washington|West Virginia|Wisconsin|Wyoming"
)
# A US agreement is governed by a state's law; 380 of a random 400 indexed
# agreements say "laws of the State of <state>" somewhere.
_US_LAW = re.compile(rf"laws? of the (?:State|Commonwealth) of (?:{_US_STATES})\b", re.I)
# Without US governing law, the markers name the likely jurisdiction.
_JURISDICTION_MARKERS: dict[str, re.Pattern[str]] = {
    "uk": re.compile(
        r"Companies Act 2006|England and Wales|Takeover Code|City Code on Takeovers|Court Meeting|scheme of arrangement",
        re.I,
    ),
    "au": re.compile(r"Corporations Act|Pty\.? Ltd|Australian Securities|Takeovers Panel|\bASX\b", re.I),
    "ca": re.compile(r"Business Corporations Act \((?:British Columbia|Ontario|Alberta)|\bCBCA\b|Canada Business Corporations Act", re.I),
}

# Reference shapes the enricher does not resolve yet (enricher gaps, not XML faults).
_GAP_SHAPES: dict[str, re.Pattern[str]] = {
    "dotted_subsection": re.compile(r"(?i)^sections?\s+\d+\.\d+\.\d+"),
    "article_as_section": re.compile(r"(?i)^articles?\s+\d+\.\d+"),
    "letter_suffix_section": re.compile(r"(?i)^sections?\s+\d+\.\d+[A-Z]\b"),
}


def jurisdiction(xml: str) -> str:
    """`us` when a US state's law appears, else `uk`, `au` or `ca` by markers, else `other`."""
    if _US_LAW.search(xml):
        return "us"
    hits = {name: len(pattern.findall(xml)) for name, pattern in _JURISDICTION_MARKERS.items()}
    best = max(hits, key=lambda name: hits[name])
    return best if hits[best] >= 1 else "other"


# A flat agreement ("Section 1. Defined Terms", "2. Purchase and Sale") whose
# numbered top-level items are tagged as articles: the XML is a fair reading of
# the document, but the enricher only resolves "Section 2(a)" against sections.
_NUMBERED_ARTICLE_TITLE = re.compile(r'<article\b[^>]*\btitle="(?:Section\s+)?\d+\.?\s', re.I)
_ARTICLE_TAG = re.compile(r"<article\b")


def sections_tagged_as_articles(xml: str) -> bool:
    """True when most articles are titled like numbered sections."""
    articles = len(_ARTICLE_TAG.findall(xml))
    return articles >= 3 and len(_NUMBERED_ARTICLE_TITLE.findall(xml)) * 2 > articles


def structure_problems(stats: dict[str, Any], rate: float | None = None) -> list[str]:
    """XML structure faults visible in a reader-index payload's stats. `rate` is
    the body section-level rate; duplicates count only when it is low or unknown."""
    numbered = int(stats.get("sections_numbered") or 0)
    unnumbered = int(stats.get("sections_unnumbered") or 0)
    duplicates = len(stats.get("duplicate_section_numbers") or [])
    problems: list[str] = []
    if numbered == 0:
        problems.append("no_numbered_sections")
    elif unnumbered > numbered:
        problems.append("mostly_unnumbered_sections")
    if duplicates >= DUPLICATE_SECTIONS_MIN and (rate is None or rate < DUPLICATE_SECTIONS_RATE_MAX):
        problems.append("duplicate_section_numbers")
    return problems


def gap_shapes(xrefs: list[dict[str, Any]]) -> Counter[str]:
    """Unresolved references by the enricher-gap shape they match."""
    counts: Counter[str] = Counter()
    for xref in xrefs:
        if xref.get("status") != "unresolved":
            continue
        for name, pattern in _GAP_SHAPES.items():
            if pattern.match(xref.get("text", "")):
                counts[name] += 1
                break
    return counts


@dataclass
class Row:
    agreement_uuid: str
    xml_version: int
    target: str
    acquirer: str
    filing_date: str
    url: str
    body_section_level_rate: float | None
    sections_numbered: int
    sections_unnumbered: int
    duplicate_section_numbers: int
    n_xrefs: int
    problems: list[str] = field(default_factory=list)
    gaps: Counter[str] = field(default_factory=Counter)
    jurisdiction: str = ""


def _engine(env_path: Path) -> Engine:
    _load_env(env_path)
    db: DBResource = _build_db_resource()
    return db.get_engine()


def collect(engine: Engine) -> tuple[list[Row], int]:
    """Rows worth a look, and the number of agreements scanned."""
    flagged: list[Row] = []
    scanned = 0
    with engine.connect().execution_options(stream_results=True) as conn:
        result = conn.execute(
            text(
                """
                SELECT r.agreement_uuid, r.xml_version, r.payload_gz, r.body_section_level_rate,
                       a.target, a.acquirer, a.filing_date, a.url
                FROM reader_index r
                JOIN xml x ON x.agreement_uuid = r.agreement_uuid AND x.version = r.xml_version
                 AND x.latest = 1 AND x.status = 'verified'
                JOIN agreements a ON a.agreement_uuid = r.agreement_uuid
                WHERE r.enricher_version = :enricher_version
                """
            ),
            {"enricher_version": ENRICHER_VERSION},
        )
        for rec in result.mappings():
            scanned += 1
            payload = json.loads(gzip.decompress(rec["payload_gz"]))
            stats = payload["stats"]
            rate = None if rec["body_section_level_rate"] is None else float(rec["body_section_level_rate"])
            problems = structure_problems(stats, rate)
            low = rate is not None and rate < LOW_RATE
            if not problems and not low:
                continue
            flagged.append(
                Row(
                    agreement_uuid=rec["agreement_uuid"],
                    xml_version=int(rec["xml_version"]),
                    target=rec["target"] or "",
                    acquirer=rec["acquirer"] or "",
                    filing_date=str(rec["filing_date"] or ""),
                    url=rec["url"] or "",
                    body_section_level_rate=rate,
                    sections_numbered=int(stats.get("sections_numbered") or 0),
                    sections_unnumbered=int(stats.get("sections_unnumbered") or 0),
                    duplicate_section_numbers=len(stats.get("duplicate_section_numbers") or []),
                    n_xrefs=int(stats.get("n_xrefs") or 0),
                    problems=problems,
                    gaps=gap_shapes(payload["xrefs"]),
                )
            )
        for row in flagged:
            xml = conn.execute(
                text("SELECT xml FROM xml WHERE agreement_uuid = :u AND version = :v"),
                {"u": row.agreement_uuid, "v": row.xml_version},
            ).scalar()
            row.jurisdiction = jurisdiction(str(xml or ""))
            if "no_numbered_sections" in row.problems and sections_tagged_as_articles(str(xml or "")):
                row.problems.remove("no_numbered_sections")
                row.gaps["sections_tagged_as_articles"] += 1
    return flagged, scanned


_CSV_FIELDS = (
    "agreement_uuid",
    "xml_version",
    "target",
    "acquirer",
    "filing_date",
    "body_section_level_rate",
    "sections_numbered",
    "sections_unnumbered",
    "duplicate_section_numbers",
    "n_xrefs",
    "reason",
    "url",
)


def _write_csv(path: Path, rows: list[Row], reason: str | None = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{name: getattr(row, name) for name in _CSV_FIELDS if hasattr(row, name)},
                    "reason": reason
                    or ";".join(row.problems)
                    or ";".join(f"{k}={v}" for k, v in row.gaps.most_common()),
                }
            )


def write_report(out: Path, flagged: list[Row], scanned: int) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    us = [r for r in flagged if r.jurisdiction == "us"]
    structure = sorted(
        (r for r in us if r.problems),
        key=lambda r: (r.body_section_level_rate if r.body_section_level_rate is not None else 1.0, r.agreement_uuid),
    )
    gaps = sorted(
        (r for r in us if not r.problems and sum(r.gaps.values()) > 0),
        key=lambda r: (-int("sections_tagged_as_articles" in r.gaps), -sum(r.gaps.values())),
    )
    unexplained = [r for r in us if not r.problems and sum(r.gaps.values()) == 0]
    non_us = Counter(r.jurisdiction for r in flagged if r.jurisdiction != "us")
    _write_csv(out / "xml_structure.csv", structure)
    _write_csv(out / "enricher_gaps.csv", gaps)
    _write_csv(out / "low_rate_unexplained.csv", unexplained, reason="low_rate")

    problem_counts = Counter(p for r in structure for p in r.problems)
    gap_counts: Counter[str] = Counter()
    for r in gaps:
        gap_counts.update(r.gaps)
    summary = {
        "scanned": scanned,
        "enricher_version": ENRICHER_VERSION,
        "xml_structure": len(structure),
        "xml_structure_by_problem": dict(problem_counts),
        "enricher_gaps": len(gaps),
        "enricher_gap_refs": dict(gap_counts),
        "low_rate_unexplained": len(unexplained),
        "non_us_excluded": dict(non_us),
    }
    lines = [
        f"# Reader-index health report, {date.today().isoformat()}",
        "",
        f"Scanned {scanned:,} agreements with a reader index at enricher {ENRICHER_VERSION}"
        " (latest verified XML only). Read-only: nothing in the database was changed.",
        "",
        "| File | Agreements | What it is |",
        "|---|---|---|",
        f"| `xml_structure.csv` | {len(structure)} | US agreements whose XML structure is broken. **For the validation queue.** |",
        f"| `enricher_gaps.csv` | {len(gaps)} | US agreements the enricher reads poorly though their XML is sound: references in shapes it does not resolve yet, or flat agreements whose numbered sections are tagged as articles. Not for the validation queue. |",
        f"| `low_rate_unexplained.csv` | {len(unexplained)} | US agreements below {LOW_RATE:.0%} with sound structure and no known gap shape. Often very few references. |",
        "",
        "**XML structure problems** (an agreement can have more than one):",
        "",
        *[f"- `{name}`: {count}" for name, count in problem_counts.most_common()],
        "",
        "**Enricher gaps** (unresolved references by shape; `sections_tagged_as_articles` counts agreements):",
        "",
        *[f"- `{name}`: {count}" for name, count in gap_counts.most_common()],
        "",
        "**Left out as non-US** (no US state's governing law; named by markers such as the Companies Act 2006 or the Corporations Act): "
        + (", ".join(f"{k} {v}" for k, v in non_us.most_common()) or "none")
        + ".",
        "",
    ]
    (out / "README.md").write_text("\n".join(lines), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    _ = parser.add_argument("--out", type=Path, required=True, help="Directory for the CSVs and README.md")
    _ = parser.add_argument("--env", type=Path, default=BACKEND_ENV_PATH, help="Env file with MARIADB_* settings")
    args = parser.parse_args()
    flagged, scanned = collect(_engine(args.env))
    print(json.dumps(write_report(args.out, flagged, scanned), indent=1))


if __name__ == "__main__":
    main()
