"""Read-only audit utility for TOC-aware XML validation.

Use cases:
- audit the four Agreement Index tooltip examples
- sample current invalid agreements with section_non_sequential reasons
- compare persisted latest XML reasons against recomputed TOC-aware hard-rule violations
"""
# pyright: reportAny=false, reportPrivateUsage=false

from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

from sqlalchemy import bindparam, text

from etl.defs.f_xml_asset import (
    extract_body_section_sequences,
    extract_toc_section_sequences,
    find_hard_rule_violations,
    toc_consistent_article_numbers,
)
from etl.defs.resources import DBResource


REPO_ROOT = Path(__file__).resolve().parents[4]
BACKEND_ENV_PATH = REPO_ROOT / "backend" / ".env"
TOOLTIP_URLS = (
    "https://www.sec.gov/Archives/edgar/data/862861/000095017023013095/jan-ex10_95.htm",
    "https://www.sec.gov/Archives/edgar/data/1863181/000114036123012443/ny20008306x2_ex2-1.htm",
    "https://www.sec.gov/Archives/edgar/data/1078799/000107997321000583/ex2x1.htm",
    "https://www.sec.gov/Archives/edgar/data/1820143/000119312521053301/d102219dex21.htm",
)


@dataclass(frozen=True)
class AgreementRecord:
    agreement_uuid: str
    url: str
    latest_xml_version: int
    latest_xml_status: str
    xml_text: str


@dataclass(frozen=True)
class AgreementAuditResult:
    agreement_uuid: str
    url: str
    latest_xml_version: int
    latest_xml_status: str
    persisted_reason_codes: list[str]
    recomputed_reason_codes: list[str]
    suppressed_section_non_sequential: bool
    miss_category: str | None
    toc_consistent_articles: list[int]
    consistent_non_sequential_articles: list[int]
    toc_conflict_articles: list[int]
    toc_sequences: dict[str, list[int]]
    body_sequences: dict[str, list[int]]
    parse_error: str | None


def _is_sequential_sequence(sequence: list[int]) -> bool:
    return sequence == list(range(1, len(sequence) + 1))


def _consistent_non_sequential_articles(
    toc_sequences: dict[int, list[int]],
    body_sequences: dict[int, list[int]],
) -> list[int]:
    matching_articles: list[int] = []
    for article_num, body_sequence in body_sequences.items():
        toc_sequence = toc_sequences.get(article_num)
        if toc_sequence != body_sequence:
            continue
        if _is_sequential_sequence(body_sequence):
            continue
        matching_articles.append(article_num)
    return sorted(matching_articles)


def _toc_conflict_articles(
    toc_sequences: dict[int, list[int]],
    body_sequences: dict[int, list[int]],
) -> list[int]:
    conflicting_articles: list[int] = []
    for article_num, body_sequence in body_sequences.items():
        toc_sequence = toc_sequences.get(article_num)
        if toc_sequence is None:
            continue
        if toc_sequence == body_sequence:
            continue
        if _is_sequential_sequence(body_sequence) and _is_sequential_sequence(toc_sequence):
            continue
        conflicting_articles.append(article_num)
    return sorted(conflicting_articles)


def _categorize_section_non_sequential_miss(
    *,
    persisted_reason_codes: list[str],
    recomputed_reason_codes: list[str],
    toc_sequences: dict[int, list[int]],
    body_sequences: dict[int, list[int]],
    toc_conflict_articles: list[int],
) -> str | None:
    if "section_non_sequential" not in persisted_reason_codes:
        return None
    if "section_non_sequential" not in recomputed_reason_codes:
        return None
    if not toc_sequences:
        return "missing_toc"
    if _consistent_non_sequential_articles(toc_sequences, body_sequences):
        return "other_article_still_invalid"
    if toc_conflict_articles:
        return "toc_conflict"
    other_reason_codes = {
        reason_code
        for reason_code in recomputed_reason_codes
        if reason_code != "section_non_sequential"
    }
    if other_reason_codes:
        return "structural_invalid_mixed"
    return "uncategorized"


def _load_env(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"Expected env file at {path}")
    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key:
                os.environ[key] = value


def _build_db_resource() -> DBResource:
    return DBResource(
        user=os.environ["MARIADB_USER"],
        password=os.environ["MARIADB_PASSWORD"],
        host=os.environ["MARIADB_HOST"],
        port=os.environ.get("MARIADB_PORT", "3306"),
        database=os.environ["MARIADB_DATABASE"],
    )


def _fetch_records_by_agreement_uuids(db: DBResource, agreement_uuids: list[str]) -> list[AgreementRecord]:
    if not agreement_uuids:
        return []
    engine = db.get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT
                    a.agreement_uuid,
                    a.url,
                    x.version AS latest_xml_version,
                    x.status AS latest_xml_status,
                    x.xml AS xml_text
                FROM {db.database}.agreements a
                JOIN {db.database}.xml x
                    ON x.agreement_uuid = a.agreement_uuid
                   AND x.latest = 1
                WHERE a.agreement_uuid IN :agreement_uuids
                ORDER BY a.agreement_uuid
                """
            ).bindparams(bindparam("agreement_uuids", expanding=True)),
            {"agreement_uuids": agreement_uuids},
        ).mappings().fetchall()
    return [
        AgreementRecord(
            agreement_uuid=str(row["agreement_uuid"]),
            url=str(row["url"]),
            latest_xml_version=int(row["latest_xml_version"]),
            latest_xml_status=str(row["latest_xml_status"] or ""),
            xml_text=str(row["xml_text"] or ""),
        )
        for row in rows
    ]


def _resolve_tooltip_agreement_uuids(db: DBResource) -> list[str]:
    engine = db.get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT agreement_uuid
                FROM {db.database}.agreements
                WHERE url IN :urls
                   OR secondary_filing_url IN :urls
                ORDER BY agreement_uuid
                """
            ).bindparams(bindparam("urls", expanding=True)),
            {"urls": list(TOOLTIP_URLS)},
        ).scalars().all()
    return [str(row) for row in rows]


def _sample_invalid_gap_agreement_uuids(db: DBResource, limit: int) -> list[str]:
    engine = db.get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT DISTINCT x.agreement_uuid
                FROM {db.database}.xml x
                JOIN {db.database}.xml_status_reasons r
                    ON r.agreement_uuid = x.agreement_uuid
                   AND r.xml_version = x.version
                WHERE x.latest = 1
                  AND x.status = 'invalid'
                  AND r.reason_code = 'section_non_sequential'
                ORDER BY x.agreement_uuid
                LIMIT :limit_n
                """
            ),
            {"limit_n": int(limit)},
        ).scalars().all()
    return [str(row) for row in rows]


def _fetch_persisted_reason_codes(
    db: DBResource,
    *,
    agreement_uuid: str,
    xml_version: int,
) -> list[str]:
    engine = db.get_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT DISTINCT reason_code
                FROM {db.database}.xml_status_reasons
                WHERE agreement_uuid = :agreement_uuid
                  AND xml_version = :xml_version
                ORDER BY reason_code
                """
            ),
            {
                "agreement_uuid": agreement_uuid,
                "xml_version": int(xml_version),
            },
        ).scalars().all()
    return [str(row) for row in rows]


def audit_agreement_record(db: DBResource, record: AgreementRecord) -> AgreementAuditResult:
    persisted_reason_codes = _fetch_persisted_reason_codes(
        db,
        agreement_uuid=record.agreement_uuid,
        xml_version=record.latest_xml_version,
    )
    parse_error: str | None = None
    recomputed_reason_codes: list[str] = []
    consistent_articles: list[int] = []
    consistent_non_sequential_articles: list[int] = []
    toc_conflict_articles: list[int] = []
    toc_sequences: dict[str, list[int]] = {}
    body_sequences: dict[str, list[int]] = {}
    toc_sequences_raw: dict[int, list[int]] = {}
    body_sequences_raw: dict[int, list[int]] = {}
    try:
        root = ET.fromstring(record.xml_text)
        recomputed_reason_codes = sorted(
            {violation.reason_code for violation in find_hard_rule_violations(root)}
        )
        consistent_articles = sorted(toc_consistent_article_numbers(root))
        toc_sequences_raw = extract_toc_section_sequences(root)
        body_sequences_raw = extract_body_section_sequences(root)
        consistent_non_sequential_articles = _consistent_non_sequential_articles(
            toc_sequences_raw,
            body_sequences_raw,
        )
        toc_conflict_articles = _toc_conflict_articles(
            toc_sequences_raw,
            body_sequences_raw,
        )
        toc_sequences = {
            str(article_num): sequence
            for article_num, sequence in toc_sequences_raw.items()
        }
        body_sequences = {
            str(article_num): sequence
            for article_num, sequence in body_sequences_raw.items()
        }
    except ET.ParseError as exc:
        parse_error = str(exc)
    suppressed_gap = (
        "section_non_sequential" in persisted_reason_codes
        and "section_non_sequential" not in recomputed_reason_codes
    )
    miss_category = _categorize_section_non_sequential_miss(
        persisted_reason_codes=persisted_reason_codes,
        recomputed_reason_codes=recomputed_reason_codes,
        toc_sequences=toc_sequences_raw,
        body_sequences=body_sequences_raw,
        toc_conflict_articles=toc_conflict_articles,
    )
    return AgreementAuditResult(
        agreement_uuid=record.agreement_uuid,
        url=record.url,
        latest_xml_version=record.latest_xml_version,
        latest_xml_status=record.latest_xml_status,
        persisted_reason_codes=persisted_reason_codes,
        recomputed_reason_codes=recomputed_reason_codes,
        suppressed_section_non_sequential=suppressed_gap,
        miss_category=miss_category,
        toc_consistent_articles=consistent_articles,
        consistent_non_sequential_articles=consistent_non_sequential_articles,
        toc_conflict_articles=toc_conflict_articles,
        toc_sequences=toc_sequences,
        body_sequences=body_sequences,
        parse_error=parse_error,
    )


def _print_human_report(results: list[AgreementAuditResult]) -> None:
    total = len(results)
    persisted_gap = sum("section_non_sequential" in row.persisted_reason_codes for row in results)
    suppressed_gap = sum(row.suppressed_section_non_sequential for row in results)
    with_toc = sum(bool(row.toc_sequences) for row in results)
    miss_categories = {
        "missing_toc": sum(row.miss_category == "missing_toc" for row in results),
        "other_article_still_invalid": sum(row.miss_category == "other_article_still_invalid" for row in results),
        "toc_conflict": sum(row.miss_category == "toc_conflict" for row in results),
        "structural_invalid_mixed": sum(row.miss_category == "structural_invalid_mixed" for row in results),
        "uncategorized": sum(row.miss_category == "uncategorized" for row in results),
    }
    print(f"Audited agreements: {total}")
    print(f"With TOC entries: {with_toc}")
    print(f"Persisted section_non_sequential: {persisted_gap}")
    print(f"Suppressed by recompute: {suppressed_gap}")
    summary_line = (
        "Unresolved miss categories: "
        + f"missing_toc={miss_categories['missing_toc']}, "
        + f"other_article_still_invalid={miss_categories['other_article_still_invalid']}, "
        + f"toc_conflict={miss_categories['toc_conflict']}, "
        + f"structural_invalid_mixed={miss_categories['structural_invalid_mixed']}, "
        + f"uncategorized={miss_categories['uncategorized']}"
    )
    print(summary_line)
    for row in results:
        print()
        print(f"{row.agreement_uuid} | v{row.latest_xml_version} | {row.url}")
        print(f"  latest_xml_status: {row.latest_xml_status}")
        print(f"  persisted_reason_codes: {', '.join(row.persisted_reason_codes) or 'none'}")
        print(f"  recomputed_reason_codes: {', '.join(row.recomputed_reason_codes) or 'none'}")
        print(f"  suppressed_section_non_sequential: {row.suppressed_section_non_sequential}")
        print(f"  miss_category: {row.miss_category or 'n/a'}")
        print(f"  toc_consistent_articles: {row.toc_consistent_articles}")
        print(f"  consistent_non_sequential_articles: {row.consistent_non_sequential_articles}")
        print(f"  toc_conflict_articles: {row.toc_conflict_articles}")
        if row.parse_error is not None:
            print(f"  parse_error: {row.parse_error}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit TOC-aware XML validation against live agreement XML.")
    _ = parser.add_argument("--tooltip-examples", action="store_true", help="Audit the four Agreement Index tooltip agreements.")
    _ = parser.add_argument("--agreement-uuid", action="append", default=[], help="Agreement UUID to audit. May be repeated.")
    _ = parser.add_argument("--sample-invalid-gaps", type=int, default=0, help="Audit the first N latest invalid agreements with section_non_sequential.")
    _ = parser.add_argument("--json", action="store_true", help="Emit JSON instead of human-readable output.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _load_env(BACKEND_ENV_PATH)
    db = _build_db_resource()

    target_uuids: list[str] = []
    if args.tooltip_examples:
        target_uuids.extend(_resolve_tooltip_agreement_uuids(db))
    target_uuids.extend(str(value) for value in args.agreement_uuid)
    if args.sample_invalid_gaps > 0:
        target_uuids.extend(_sample_invalid_gap_agreement_uuids(db, args.sample_invalid_gaps))
    target_uuids = sorted(dict.fromkeys(target_uuids))
    if not target_uuids:
        print("No targets selected. Use --tooltip-examples, --agreement-uuid, or --sample-invalid-gaps.", file=sys.stderr)
        return 2

    records = _fetch_records_by_agreement_uuids(db, target_uuids)
    results = [audit_agreement_record(db, record) for record in records]
    if args.json:
        print(json.dumps([asdict(row) for row in results], indent=2, sort_keys=True))
    else:
        _print_human_report(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
