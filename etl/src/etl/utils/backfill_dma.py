"""Backfill DMA filing metadata from SEC index pages."""

from __future__ import annotations

import argparse
import logging
import os
import re
import time
from urllib.parse import unquote
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, cast

from bs4 import BeautifulSoup, Tag
from requests.sessions import Session
from sqlalchemy import text
from sqlalchemy.engine import Connection

from etl.defs.resources import DBResource
from etl.utils.sec_utils import SEC_USER_AGENT


_SEC_URL_RE = re.compile(
    r"^https?://www\.sec\.gov/Archives/edgar/data/(\d+)/(\d{18})/([^/?#]+)"
)
_EXHIBIT_TYPE_RE = re.compile(r"(?i)\bEX[-\s]*(10|2|99)(?:\.[\w-]+)?\b")
_FORM_TYPE_RE = re.compile(r"\bType:\s*([A-Za-z0-9-]+)")
_CIK_RE = re.compile(r"CIK:\s*(\d{1,10})")
_FILING_DATE_RE = re.compile(r"\bFiling Date\s+(\d{4}-\d{2}-\d{2})\b")

_REQUEST_TIMEOUT_SECONDS = 10.0
_COMMIT_EVERY = 100
_RATE_LIMIT_MAX_REQUESTS = 10
_RATE_LIMIT_WINDOW_SECONDS = 1.0
_ENV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.env"))


class _ResponseLike(Protocol):
    text: str

    def raise_for_status(self) -> None: ...


class _SessionLike(Protocol):
    def get(self, url: str, **kwargs: object) -> _ResponseLike: ...


@dataclass(frozen=True)
class SecUrlParts:
    cik: str
    accession_no_dashes: str
    filename: str

    @property
    def accession_with_dashes(self) -> str:
        no_dashes = self.accession_no_dashes
        return f"{no_dashes[:10]}-{no_dashes[10:12]}-{no_dashes[12:]}"

    @property
    def index_url(self) -> str:
        return (
            "https://www.sec.gov/Archives/edgar/data/"
            f"{self.cik}/{self.accession_no_dashes}/{self.accession_with_dashes}-index.html"
        )


@dataclass
class FilingMetadata:
    filing_date: date | None
    filing_company_name: str | None
    filing_company_cik: str | None
    form_type: str | None
    exhibit_type: str | None


@dataclass
class BackfillStats:
    total_rows: int = 0
    updated: int = 0
    skipped_invalid_url: int = 0
    fetch_failed: int = 0
    no_metadata: int = 0
    row_errors: int = 0


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: float) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: list[float] = []

    def acquire(self) -> None:
        if self.max_requests <= 0 or self.window_seconds <= 0:
            return
        now = time.monotonic()
        self._timestamps = [
            ts for ts in self._timestamps if now - ts < self.window_seconds
        ]
        if len(self._timestamps) >= self.max_requests:
            sleep_for = self.window_seconds - (now - self._timestamps[0])
            if sleep_for > 0:
                time.sleep(sleep_for)
        self._timestamps.append(time.monotonic())


def parse_sec_url(url: str) -> SecUrlParts | None:
    match = _SEC_URL_RE.match(url)
    if not match:
        return None
    cik, accession_no_dashes, filename = match.groups()
    return SecUrlParts(cik=cik, accession_no_dashes=accession_no_dashes, filename=filename)


def fetch_index_page(
    session: Session,
    url_parts: SecUrlParts,
    rate_limiter: RateLimiter,
) -> str | None:
    headers = {"User-Agent": SEC_USER_AGENT}
    typed_session = cast(_SessionLike, cast(object, session))
    try:
        rate_limiter.acquire()
        response = typed_session.get(
            url_parts.index_url, headers=headers, timeout=_REQUEST_TIMEOUT_SECONDS
        )
        _ = response.raise_for_status()
    except Exception as exc:
        logging.warning("Failed to fetch %s: %s", url_parts.index_url, exc)
        return None
    return response.text


def _find_label_value(soup: BeautifulSoup, label: str) -> str | None:
    label_re = re.compile(rf"^{re.escape(label)}$", re.IGNORECASE)
    for node in soup.find_all(string=label_re):
        parent = node.parent
        if parent is None:
            continue
        sibling = parent.find_next_sibling()
        if sibling is not None:
            value = sibling.get_text(" ", strip=True)
            if value:
                return value
        row = parent.find_parent("tr")
        if row is None:
            continue
        cells = row.find_all(["td", "th"])
        if len(cells) >= 2:
            return cells[1].get_text(" ", strip=True) or None
    return None


def _parse_filing_date(raw_value: str | None) -> date | None:
    if raw_value is None:
        return None
    try:
        return datetime.strptime(raw_value.strip(), "%Y-%m-%d").date()
    except ValueError:
        return None


def _extract_company_info(soup: BeautifulSoup) -> tuple[str | None, str | None]:
    company_tags = soup.find_all("span", class_="companyName")
    company_text = None
    for tag in company_tags:
        text = tag.get_text(" ", strip=True)
        if "Filer" in text:
            company_text = text
            break
    if company_text is None and company_tags:
        company_text = company_tags[0].get_text(" ", strip=True)

    company_name = None
    if company_text:
        company_name = company_text.split(" (")[0].strip() or None

    cik = None
    if company_text:
        cik_match = _CIK_RE.search(company_text)
        if cik_match:
            cik = cik_match.group(1)

    if cik is None:
        cik_match = _CIK_RE.search(soup.get_text(" ", strip=True))
        if cik_match:
            cik = cik_match.group(1)
    if cik is None:
        cik_link = soup.select_one("span.companyName a[href*='CIK=']")
        if cik_link is not None:
            href = cik_link.get("href")
            if isinstance(href, str):
                cik_match = re.search(r"CIK=(\d+)", href)
                if cik_match:
                    cik = cik_match.group(1)

    return company_name, cik


def _extract_form_type(soup: BeautifulSoup) -> str | None:
    form_type = _find_label_value(soup, "Form Type")
    if form_type:
        return form_type
    match = _FORM_TYPE_RE.search(soup.get_text(" ", strip=True))
    if match:
        return match.group(1)
    return None


def _normalize_filename(value: str) -> str:
    return unquote(value).split("/")[-1].strip().lower()


def _row_matches_filename(row: Tag, target: str) -> bool:
    cols = row.find_all("td")
    if len(cols) < 3:
        return False
    link_tag = cols[2].find("a")
    if not isinstance(link_tag, Tag):
        return False
    doc_name = link_tag.get_text(strip=True)
    href = link_tag.get("href")
    candidates: list[str] = []
    if doc_name:
        candidates.append(doc_name)
    if isinstance(href, str):
        candidates.append(href)
    return any(_normalize_filename(candidate) == target for candidate in candidates)


def _extract_exhibit_type(soup: BeautifulSoup, filename: str) -> str | None:
    document_table: Tag | None = None
    tables = soup.find_all("table", class_="tableFile")
    for table in tables:
        first_row = table.find("tr")
        if first_row is None:
            continue
        if "Document Format Files" in str(table) or "Seq" in str(first_row):
            document_table = table
            break
    if document_table is None:
        return None

    target = _normalize_filename(filename)
    for row in document_table.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) < 4:
            continue
        if not _row_matches_filename(row, target):
            continue
        doc_type = cols[3].get_text(" ", strip=True)
        match = _EXHIBIT_TYPE_RE.search(doc_type)
        if match:
            return match.group(1)
        description = cols[1].get_text(" ", strip=True)
        match = _EXHIBIT_TYPE_RE.search(description)
        if match:
            return match.group(1)
        return None
    return None


def _extract_metadata(html: str, filename: str) -> FilingMetadata:
    soup = BeautifulSoup(html, "html.parser")
    filing_date = _parse_filing_date(_find_label_value(soup, "Filing Date"))
    if filing_date is None:
        match = _FILING_DATE_RE.search(soup.get_text(" ", strip=True))
        if match:
            filing_date = _parse_filing_date(match.group(1))
    form_type = _extract_form_type(soup)
    filing_company_name, filing_company_cik = _extract_company_info(soup)
    exhibit_type = _extract_exhibit_type(soup, filename)
    return FilingMetadata(
        filing_date=filing_date,
        filing_company_name=filing_company_name,
        filing_company_cik=filing_company_cik,
        form_type=form_type,
        exhibit_type=exhibit_type,
    )


def _build_db_resource() -> DBResource:
    return DBResource(
        user=os.environ["MARIADB_USER"],
        password=os.environ["MARIADB_PASSWORD"],
        host=os.environ["MARIADB_HOST"],
        port=os.environ["MARIADB_PORT"],
        database=os.environ["MARIADB_DATABASE"],
    )


def _load_env(path: str) -> None:
    if not os.path.exists(path):
        logging.warning("Env file not found at %s; using existing environment.", path)
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def _fetch_rows(conn: Connection, schema: str) -> list[Mapping[str, object]]:
    query = text(
        f"""
        SELECT agreement_uuid, url
        FROM {schema}.agreements
        WHERE url LIKE 'https://www.sec.gov/Archives/edgar/data/%'
          AND (
            filing_date IS NULL
            OR filing_company_name IS NULL
            OR filing_company_cik IS NULL
            OR form_type IS NULL
            OR exhibit_type IS NULL
          )
        """
    )
    result = conn.execute(query)
    return cast(list[Mapping[str, object]], result.mappings().all())


def _update_row(
    conn: Connection,
    schema: str,
    agreement_uuid: str,
    metadata: FilingMetadata,
) -> None:
    update_sql = text(
        f"""
        UPDATE {schema}.agreements
        SET
          filing_date = COALESCE(:filing_date, filing_date),
          filing_company_name = COALESCE(:filing_company_name, filing_company_name),
          filing_company_cik = COALESCE(:filing_company_cik, filing_company_cik),
          form_type = COALESCE(:form_type, form_type),
          exhibit_type = COALESCE(:exhibit_type, exhibit_type)
        WHERE agreement_uuid = :agreement_uuid
          AND (
            NOT (filing_date <=> COALESCE(:filing_date, filing_date))
            OR NOT (filing_company_name <=> COALESCE(:filing_company_name, filing_company_name))
            OR NOT (filing_company_cik <=> COALESCE(:filing_company_cik, filing_company_cik))
            OR NOT (form_type <=> COALESCE(:form_type, form_type))
            OR NOT (exhibit_type <=> COALESCE(:exhibit_type, exhibit_type))
          )
        """
    )
    _ = conn.execute(
        update_sql,
        {
            "filing_date": metadata.filing_date,
            "filing_company_name": metadata.filing_company_name,
            "filing_company_cik": metadata.filing_company_cik,
            "form_type": metadata.form_type,
            "exhibit_type": metadata.exhibit_type,
            "agreement_uuid": agreement_uuid,
        },
    )


def backfill_dma() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s backfill_dma: %(message)s",
    )
    _load_env(_ENV_PATH)
    db = _build_db_resource()
    schema = db.database
    engine = db.get_engine()
    rate_limiter = RateLimiter(
        max_requests=_RATE_LIMIT_MAX_REQUESTS,
        window_seconds=_RATE_LIMIT_WINDOW_SECONDS,
    )

    with engine.begin() as conn:
        rows = list(_fetch_rows(conn, schema))
    stats = BackfillStats(total_rows=len(rows))

    if not rows:
        logging.info("No rows found for backfill.")
        return

    session = Session()
    updated = 0
    with engine.connect() as conn:
        transaction = conn.begin()
        try:
            for idx, row in enumerate(rows, start=1):
                try:
                    url = str(row["url"])
                    url_parts = parse_sec_url(url)
                    if url_parts is None:
                        logging.info("Skipping unsupported SEC URL: %s", url)
                        stats.skipped_invalid_url += 1
                        continue

                    html = fetch_index_page(session, url_parts, rate_limiter)
                    if html is None:
                        stats.fetch_failed += 1
                        continue

                    metadata = _extract_metadata(html, url_parts.filename)
                    if all(
                        value is None
                        for value in (
                            metadata.filing_date,
                            metadata.filing_company_name,
                            metadata.filing_company_cik,
                            metadata.form_type,
                            metadata.exhibit_type,
                        )
                    ):
                        logging.info("No metadata found for %s", url_parts.index_url)
                        stats.no_metadata += 1
                        continue

                    _update_row(conn, schema, str(row["agreement_uuid"]), metadata)
                    updated += 1
                    stats.updated += 1
                except Exception as exc:
                    logging.exception(
                        "Row failed (agreement_uuid=%s): %s",
                        row.get("agreement_uuid"),
                        exc,
                    )
                    stats.row_errors += 1

                if idx % _COMMIT_EVERY == 0:
                    transaction.commit()
                    transaction = conn.begin()
                    logging.info("Committed after %s rows.", idx)

                if idx % 25 == 0:
                    logging.info("Processed %s/%s rows...", idx, len(rows))
        except Exception:
            transaction.rollback()
            raise
        else:
            transaction.commit()
        finally:
            session.close()

        logging.info(
            (
                "Backfill complete. total=%s updated=%s skipped_invalid_url=%s "
                "fetch_failed=%s no_metadata=%s row_errors=%s"
            ),
            stats.total_rows,
            stats.updated,
            stats.skipped_invalid_url,
            stats.fetch_failed,
            stats.no_metadata,
            stats.row_errors,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill DMA filing metadata from SEC index pages."
    )
    return parser.parse_args()


def _run_cli() -> None:
    _ = _parse_args()
    backfill_dma()


if __name__ == "__main__":
    _run_cli()
