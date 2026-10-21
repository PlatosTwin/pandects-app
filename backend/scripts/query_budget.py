"""Check the hot read endpoints against a real MariaDB: statements and pages read.

Production runs on a 1-vCPU MariaDB with a 192MB buffer pool, so a query that
touches a few million InnoDB pages answers in a second on a laptop with the
whole corpus cached and times out in production. Wall time on a warm local
database hides that; the number of buffer-pool pages a request touches
(``Innodb_buffer_pool_read_requests``, logical reads) does not. Logical reads
are a proxy: on a cold 192MB pool most of them become disk reads.
This script calls each hot endpoint in-process and fails when its first call
(before the in-process count and reference caches hold it) issues more
statements or reads more pages than its budget; a second call is reported for
reference.

Usage, from the repository root, with ``backend/.env`` pointing at a MariaDB
holding a production-sized ``pdx`` (run it with nothing else using that
database; page reads are server-wide counters):

    backend/venv/bin/python -m backend.scripts.query_budget
    backend/venv/bin/python -m backend.scripts.query_budget --only sections
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass

# Page budgets sit at roughly 5x what these requests read on the 2026-10 corpus
# (1.1M latest sections) after the clause-search fix. Before it, the broadest
# clause type read 3.4M pages for a section search and 1.3M for a deal search.
_PAGE_BUDGET_SMALL = 50_000
_PAGE_BUDGET_CLAUSE = 600_000


@dataclass(frozen=True)
class Probe:
    path: str
    max_statements: int
    max_pages: int


def _probes(clause_ids: list[str], agreement_uuid: str) -> list[Probe]:
    clause = "&".join(f"standard_id={value}" for value in clause_ids)
    return [
        # Statement budgets count a first call: it also pays one-time lookups
        # (taxonomy expansion, the ineligible-agreement probe) that later calls skip.
        Probe("/v1/sections?page=1&page_size=25", 7, _PAGE_BUDGET_SMALL),
        Probe(f"/v1/sections?page=1&page_size=25&{clause}", 8, _PAGE_BUDGET_CLAUSE),
        Probe("/v1/search/agreements?page=1&page_size=25", 2, _PAGE_BUDGET_SMALL),
        Probe(f"/v1/search/agreements?page=1&page_size=25&{clause}", 4, _PAGE_BUDGET_CLAUSE),
        Probe("/v1/agreements-index?page=1&page_size=25", 3, _PAGE_BUDGET_SMALL),
        Probe(f"/v1/agreements/{agreement_uuid}", 1, _PAGE_BUDGET_SMALL),
        Probe(f"/v1/agreements/{agreement_uuid}/sections", 2, _PAGE_BUDGET_SMALL),
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check hot read endpoints against MariaDB statement and page budgets."
    )
    _ = parser.add_argument("--only", help="run only probes whose path contains this text")
    args = parser.parse_args(argv)

    os.environ["TEMPORARY_ACCESS_LOCKDOWN"] = "0"
    from sqlalchemy import event, text
    from sqlalchemy.engine import Engine

    from backend.app import app
    from backend.extensions import db
    from backend.services import taxonomy_service

    statements: list[str] = []

    def _record(_conn, _cursor, statement, _params, _context, _executemany) -> None:
        statements.append(statement)

    def _pages_read() -> int:
        with db.engine.connect() as conn:
            row = conn.execute(
                text("SHOW GLOBAL STATUS LIKE 'Innodb_buffer_pool_read_requests'")
            ).one()
        return int(row[1])

    with app.app_context():
        # The broadest top-level clause type, as a user picking it in Search sends it.
        counts = taxonomy_service.taxonomy_section_counts()
        top_level = [str(v) for (v,) in db.session.execute(text("SELECT standard_id FROM taxonomy_l1"))]
        clause_ids = sorted(top_level, key=lambda value: counts.get(value, 0), reverse=True)[:1]
        agreement_uuid = db.session.execute(
            text(
                "SELECT agreement_uuid FROM latest_sections_search "
                "GROUP BY agreement_uuid ORDER BY COUNT(*) DESC LIMIT 1"
            )
        ).scalar_one()
        db.session.remove()
        probes = _probes(clause_ids, str(agreement_uuid))

    if args.only:
        probes = [probe for probe in probes if args.only in probe.path]
    client = app.test_client()
    # One-time process work (dump version, table reflection) lands here, not on a probe.
    _ = client.get("/v1/agreements?page_size=1")
    failures = 0
    event.listen(Engine, "before_cursor_execute", _record)
    try:
        for probe in probes:
            for label in ("first", "again"):
                with app.app_context():
                    before = _pages_read()
                statements.clear()
                started = time.perf_counter()
                response = client.get(probe.path)
                elapsed_ms = (time.perf_counter() - started) * 1000
                issued = len(statements)
                with app.app_context():
                    pages = _pages_read() - before
                over: list[str] = []
                if response.status_code != 200:
                    over.append(f"status {response.status_code}")
                elif label == "first":
                    if issued > probe.max_statements:
                        over.append(f"statements {issued} > {probe.max_statements}")
                    if pages > probe.max_pages:
                        over.append(f"pages {pages:,} > {probe.max_pages:,}")
                failures += bool(over)
                verdict = "FAIL " + "; ".join(over) if over else "ok"
                print(
                    f"{label:5s} {elapsed_ms:7.0f}ms statements={issued:2d} "
                    f"pages={pages:>10,d}  {verdict:30s} {probe.path[:110]}"
                )
    finally:
        event.remove(Engine, "before_cursor_execute", _record)
    print(f"{failures} probe(s) over budget" if failures else "all probes within budget")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
