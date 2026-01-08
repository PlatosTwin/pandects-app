# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
# pyright: reportMissingTypeStubs=false
import argparse
import datetime
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from collections import deque
from collections.abc import Callable, Iterable
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Protocol, TypedDict, cast
from urllib.parse import urlparse

from datasketch import MinHash, MinHashLSH

import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4.element import Tag

from etl.domain.b_pre_processing import format_content, split_to_pages
from etl.defs.resources import PipelineConfig, ProcessingScope
from etl.models.exhibit_classifier.exhibit_classifier import ExhibitClassifier
from etl.utils.sec_utils import SEC_USER_AGENT


@dataclass
class FilingMetadata:
    """Metadata for a filing document.
    
    Supports two schemas:
    - DMA corpus: has target, acquirer, filing_date as datetime
    - SEC index: has prob_filing, filing_company_name, filing_company_cik, form_type, filing_date as str
    """
    agreement_uuid: str
    url: str
    filing_date: datetime.datetime | str  # datetime for DMA corpus, str (YYYYMMDD) for SEC index
    # DMA corpus fields (optional for SEC index flow)
    target: str | None = None
    acquirer: str | None = None
    # SEC index fields (optional for DMA corpus flow)
    prob_filing: float | None = None
    filing_company_name: str | None = None
    filing_company_cik: str | None = None
    form_type: str | None = None
    exhibit_type: str | None = None  # "2" or "10"
    secondary_filing_url: str | None = None  # URL of duplicate filing if detected


def get_uuid(x: str) -> str:
    """Generate a UUID5 hash from the input string."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, x))


class ExhibitClassifierProtocol(Protocol):
    def predict(self, text: str, threshold: float = 0.5) -> bool: ...

    def predict_proba(self, text: str) -> float: ...

    def predict_proba_batch(self, texts: list[str]) -> list[float]: ...


class _Logger(Protocol):
    def info(self, msg: str) -> None: ...


class _Context(Protocol):
    @property
    def log(self) -> _Logger: ...


@dataclass
class ExhibitCandidate:
    """An exhibit URL with its associated filing metadata from the idx file."""

    exhibit_url: str
    form_type: str
    filing_company_name: str
    filing_company_cik: str
    filing_date: str  # YYYYMMDD format from idx
    exhibit_type: str  # "2" or "10"


@dataclass
class AgreementCandidateResult:
    """Classification result for a candidate agreement filing."""

    candidate_url: str
    is_ma_agreement: bool
    ma_probability: float
    form_type: str
    filing_company_name: str
    filing_company_cik: str
    filing_date: str  # YYYYMMDD format from idx
    exhibit_type: str  # "2" or "10"
    minhash: MinHash  # For near-duplicate detection via LSH


class IndexFiling(TypedDict):
    form_type: str
    company_name: str
    cik: str
    date_filed: str
    file_name: str


def fetch_new_filings_dma_corpus(since: str) -> list[FilingMetadata]:
    """
    Fetch new filings from the DMA corpus (legacy/testing flow).

    This function uses a local DMA corpus CSV file for testing purposes.
    For production use, use fetch_new_filings_sec_index instead.

    Args:
        since: Date string to filter filings from.

    Returns:
        List of FilingMetadata objects.
    """
    df = pd.read_csv(
        "/Users/nikitabogdanov/PycharmProjects/merger_agreements/dma_corpus/dma_corpus_metadata_250_sample.csv",
        usecols=cast(Any, ["target", "acquirer", "date_announcement", "url", "filename"]),
        parse_dates=cast(Any, ["date_announcement"]),
    )

    # Drop duplicate filings by filename
    df.drop_duplicates(subset="filename", inplace=True)

    # Keep only filings announced after `since`
    cutoff = pd.to_datetime(since)
    df = df[df["date_announcement"] > cutoff]

    # # Sort oldest first and take only the 10 oldest new filings
    # df.sort_values("date_announcement", ascending=True, inplace=True)
    # # df = df.head(10)
    # df = df.sample(frac=0.25)

    # Build our results list via a memory‑light iterator
    results: list[FilingMetadata] = []
    for row in df.itertuples(index=False):
        # Cast to Any to work around pandas-stubs typing limitations with itertuples
        r = cast(Any, row)
        date_ann = r.date_announcement
        if isinstance(date_ann, pd.Timestamp):
            filing_date = date_ann.to_pydatetime()
        elif isinstance(date_ann, datetime.datetime):
            filing_date = date_ann
        else:
            raise TypeError(
                f"Unexpected date_announcement type: {type(date_ann).__name__}"
            )
        results.append(
            FilingMetadata(
                agreement_uuid=get_uuid(str(r.filename)),
                url=str(r.url),
                filing_date=filing_date,
                target=str(r.target),
                acquirer=str(r.acquirer),
            )
        )

    return results


# MinHash parameters for near-duplicate detection
_MINHASH_NUM_PERM = 128  # Number of permutations (higher = more accurate, slower)
_MINHASH_SHINGLE_SIZE = 5  # Size of word shingles


def _compute_minhash(content: str, is_txt: bool, is_html: bool) -> MinHash:
    """Compute a MinHash signature for near-duplicate detection.
    
    Uses only the first 20,000 characters to focus on the main agreement content
    (roughly the first N pages equivalent) and ignore exhibit sections that may 
    differ between filings.
    Uses word-level shingles (n-grams) for robustness against minor text differences.
    For very short texts (fewer words than shingle size), falls back to individual words.
    """
    # Format the entire content into text
    text = _render_agreement_text(content, is_txt=is_txt, is_html=is_html)
    
    # Use only the first 20,000 characters (roughly first N pages equivalent)
    text = text[:20000]
    
    mh = MinHash(num_perm=_MINHASH_NUM_PERM)
    words = text.lower().split()
    
    if len(words) < _MINHASH_SHINGLE_SIZE:
        # Fallback for very short documents: use individual words
        for word in words:
            mh.update(word.encode("utf-8"))
    else:
        # Create shingles (word n-grams)
        for i in range(len(words) - _MINHASH_SHINGLE_SIZE + 1):
            shingle = " ".join(words[i:i + _MINHASH_SHINGLE_SIZE])
            mh.update(shingle.encode("utf-8"))
    return mh


def classify_exhibit_candidates(
    exhibit_classifier: ExhibitClassifierProtocol,
    context: _Context,
    start_date: str,
    pipeline_config: PipelineConfig,
    *,
    days_override: int | None = None,
) -> list[AgreementCandidateResult]:
    """
    Fetch and classify all exhibit candidates from SEC indexes.
    
    Returns ALL candidates with their classification results (not filtered).
    Includes MinHash for near-duplicate detection.
    
    Args:
        days_override: If provided, overrides pipeline_config.staging_days_to_fetch.
                       Useful for day-by-day processing.
    """
    if not pipeline_config.is_batched():
        raise RuntimeError("classify_exhibit_candidates requires batched mode.")

    days_to_fetch = days_override if days_override is not None else pipeline_config.staging_days_to_fetch
    exhibit_candidates = fetch_material_exhibit_links(
        start_date=start_date,
        days=days_to_fetch,
        user_agent=SEC_USER_AGENT,
        context=context,
        rate_limit_max_requests=pipeline_config.staging_rate_limit_max_requests,
        rate_limit_window_seconds=pipeline_config.staging_rate_limit_window_seconds,
        max_workers=pipeline_config.staging_max_workers,
        use_keyword_filter=pipeline_config.staging_use_keyword_filter,
    )
    
    # Fetch content and filter out unsupported file types
    valid_candidates: list[ExhibitCandidate] = []
    agreement_texts: list[str] = []
    minhashes: list[MinHash] = []

    for candidate in exhibit_candidates:
        try:
            fetch_result = _fetch_exhibit_content(
                candidate.exhibit_url, user_agent=SEC_USER_AGENT
            )
        except Exception as e:
            # Log and skip this candidate rather than crashing the whole batch
            context.log.info(f"Failed to fetch {candidate.exhibit_url} after retries: {e}")
            continue
            
        if fetch_result is None:
            context.log.info(f"Skipping unsupported file type: {candidate.exhibit_url}")
            continue
        content, is_txt, is_html = fetch_result
        agreement_text = _render_agreement_text(content, is_txt=is_txt, is_html=is_html)
        agreement_texts.append(agreement_text)
        minhashes.append(_compute_minhash(content, is_txt=is_txt, is_html=is_html))
        valid_candidates.append(candidate)

    # Early return if no valid candidates (all fetches failed/skipped)
    if not valid_candidates:
        context.log.info("No valid exhibit candidates to classify after fetching.")
        return []

    probabilities = exhibit_classifier.predict_proba_batch(agreement_texts)

    results: list[AgreementCandidateResult] = []
    for idx, candidate in enumerate(valid_candidates):
        probability = probabilities[idx]
        is_ma = probability >= 0.5
        context.log.info(
            f"Exhibit classifier candidate {idx + 1}/{len(valid_candidates)}: ma_probability={probability:.3f} is_ma={is_ma}"
        )
        results.append(
            AgreementCandidateResult(
                candidate_url=candidate.exhibit_url,
                is_ma_agreement=is_ma,
                ma_probability=probability,
                form_type=candidate.form_type,
                filing_company_name=candidate.filing_company_name,
                filing_company_cik=candidate.filing_company_cik,
                filing_date=candidate.filing_date,
                exhibit_type=candidate.exhibit_type,
                minhash=minhashes[idx],
            )
        )

    return results


# LSH threshold for near-duplicate detection (Jaccard similarity)
# 0.85 means documents with ~85% similarity are considered duplicates
# Provides headroom for title/signature page diffs and minor formatting variations
_LSH_THRESHOLD = 0.85


def fetch_new_filings_sec_index(
    exhibit_classifier: ExhibitClassifierProtocol,
    context: _Context,
    start_date: str,
    pipeline_config: PipelineConfig,
    *,
    days_override: int | None = None,
) -> list[FilingMetadata]:
    """
    Fetch agreement candidates from SEC indexes and filter using the exhibit classifier.
    
    Returns only filings classified as M&A agreements (probability >= 0.5).
    De-duplicates near-duplicate filings using MinHash LSH (same agreement filed by 
    target and acquirer, possibly with minor differences in title/signature pages).
    
    Args:
        days_override: If provided, overrides pipeline_config.staging_days_to_fetch.
                       Useful for day-by-day processing with incremental commits.
    """
    all_candidates = classify_exhibit_candidates(
        exhibit_classifier=exhibit_classifier,
        context=context,
        start_date=start_date,
        pipeline_config=pipeline_config,
        days_override=days_override,
    )

    # Filter to M&A candidates only
    ma_candidates = [c for c in all_candidates if c.is_ma_agreement]

    # Use LSH for near-duplicate detection
    # Each candidate carries its original index for stable ordering (reflects .idx file order)
    lsh = MinHashLSH(threshold=_LSH_THRESHOLD, num_perm=_MINHASH_NUM_PERM)
    
    # Track which indices belong to which duplicate group
    # union-find structure: parent[i] = parent index, or self if root
    parent: list[int] = list(range(len(ma_candidates)))
    
    def find(x: int) -> int:
        if parent[x] != x:
            parent[x] = find(parent[x])  # path compression
        return parent[x]
    
    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py:
            # Always merge to lower index (earlier in queue)
            if px < py:
                parent[py] = px
            else:
                parent[px] = py
    
    # Insert each candidate's MinHash into LSH and check for near-duplicates
    for idx, candidate in enumerate(ma_candidates):
        # Query for similar items before inserting
        similar_keys = lsh.query(candidate.minhash)
        for key in similar_keys:
            # Keys are string representations of indices
            union(idx, int(cast(str, key)))
        # Insert with index as key
        lsh.insert(str(idx), candidate.minhash)
    
    # Group by root parent
    groups: dict[int, list[tuple[int, AgreementCandidateResult]]] = {}
    for idx, candidate in enumerate(ma_candidates):
        root = find(idx)
        if root not in groups:
            groups[root] = []
        groups[root].append((idx, candidate))

    results: list[FilingMetadata] = []
    for _, group in groups.items():
        # Sort by filing_date (earliest first), then by original index (idx order = .idx file order)
        group.sort(key=lambda x: (x[1].filing_date, x[0]))
        _, primary = group[0]
        secondary_url: str | None = None

        if len(group) > 1:
            _, secondary = group[1]
            secondary_url = secondary.candidate_url
            context.log.info(
                f"De-dup: {primary.candidate_url} (primary) has near-duplicate {secondary_url}"
            )

        results.append(
            FilingMetadata(
                agreement_uuid=get_uuid(primary.candidate_url),
                url=primary.candidate_url,
                filing_date=primary.filing_date,
                prob_filing=primary.ma_probability,
                filing_company_name=primary.filing_company_name,
                filing_company_cik=primary.filing_company_cik,
                form_type=primary.form_type,
                exhibit_type=primary.exhibit_type,
                secondary_filing_url=secondary_url,
            )
        )

    return results


# Retry configuration for transient HTTP errors (503, 429, etc.)
_FETCH_MAX_RETRIES = 3
_FETCH_RETRY_BACKOFF_BASE = 2.0  # Exponential backoff: 2s, 4s, 8s


def _fetch_exhibit_content(
    exhibit_url: str, user_agent: str, timeout: float = 25.0
) -> tuple[str, bool, bool] | None:
    """Fetch exhibit content from URL with retry on transient errors.
    
    Returns None for unsupported file types (only .txt, .htm, .html are supported).
    Retries up to 3 times with exponential backoff for 5xx and 429 errors.
    """
    # Check suffix before fetching to skip unsupported file types
    path = urlparse(exhibit_url).path
    suffix = path.lower().rsplit(".", 1)[-1] if "." in path else ""
    if suffix not in {"txt", "htm", "html"}:
        return None

    headers = {"User-Agent": user_agent}
    last_exception: Exception | None = None
    
    for attempt in range(_FETCH_MAX_RETRIES):
        try:
            response = requests.get(exhibit_url, headers=headers, timeout=timeout)
            response.raise_for_status()
            content = response.text

            if suffix in {"htm", "html"}:
                return content, False, True
            if suffix == "txt":
                return content, True, False
            # Fallback to Content-Type header (shouldn't reach here given suffix check above)
            content_type = response.headers.get("Content-Type", "").lower()
            if "html" in content_type:
                return content, False, True
            return content, True, False
            
        except requests.exceptions.HTTPError as e:
            # Retry on 5xx server errors and 429 rate limiting
            if e.response is not None and e.response.status_code in {429, 500, 502, 503, 504}:
                last_exception = e
                if attempt < _FETCH_MAX_RETRIES - 1:
                    wait_time = _FETCH_RETRY_BACKOFF_BASE ** (attempt + 1)
                    print(f"Retry {attempt + 1}/{_FETCH_MAX_RETRIES} for {exhibit_url}: {e.response.status_code}, waiting {wait_time}s")
                    _ = time.sleep(wait_time)
                    continue
            raise  # Non-retryable HTTP error
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            # Retry on timeout and connection errors
            last_exception = e
            if attempt < _FETCH_MAX_RETRIES - 1:
                wait_time = _FETCH_RETRY_BACKOFF_BASE ** (attempt + 1)
                print(f"Retry {attempt + 1}/{_FETCH_MAX_RETRIES} for {exhibit_url}: {type(e).__name__}, waiting {wait_time}s")
                _ = time.sleep(wait_time)
                continue
            raise
    
    # Should not reach here, but satisfy type checker
    if last_exception:
        raise last_exception
    return None


def _render_agreement_text(content: str, is_txt: bool, is_html: bool) -> str:
    pages = split_to_pages(content, is_txt=is_txt, is_html=is_html)
    rendered_pages = [
        format_content(page["content"], is_txt=is_txt, is_html=is_html).strip()
        for page in pages
    ]
    return "\n\n".join(page for page in rendered_pages if page)


def get_sec_index_urls(start_date_str: str, days_to_fetch: int) -> list[str]:
    """Generate daily index URLs for the next N days."""
    base_url = "https://www.sec.gov/Archives/edgar/daily-index"
    links: list[str] = []

    try:
        current_date = datetime.datetime.strptime(start_date_str, "%Y-%m-%d")
    except ValueError:
        raise ValueError(f"Invalid date format: {start_date_str}")

    for i in range(1, days_to_fetch + 1):
        next_date = current_date + datetime.timedelta(days=i)
        year = next_date.year
        month = next_date.month
        quarter = (month - 1) // 3 + 1
        file_date_str = next_date.strftime("%Y%m%d")
        url = f"{base_url}/{year}/QTR{quarter}/form.{file_date_str}.idx"
        links.append(url)

    return links


def parse_index_file(
    index_url: str,
    user_agent: str,
    context: _Context,
    rate_limited_get: Callable[..., requests.Response],
) -> list[IndexFiling]:
    """Download and parse a daily index file for specific form types."""
    target_forms = {
        "S-1",
        "S-3",
        "SF-1",
        "SF-3",
        "S-4",
        "S-11",
        "F-1",
        "F-3",
        "F-4",
        "10",
        "8-K",
        "10-D",
        "10-Q",
        "10-K",
    }

    headers = {"User-Agent": user_agent}
    filings: list[IndexFiling] = []

    try:
        response = rate_limited_get(index_url, headers=headers, timeout=10.0)
        response.raise_for_status()
    except Exception as exc:
        context.log.info(f"Failed to fetch index {index_url}: {exc}")
        return filings

    lines = response.text.splitlines()
    found_separator = False

    for line in lines:
        if line.startswith("---"):
            found_separator = True
            continue
        if not found_separator:
            continue

        parts = line.split()
        if len(parts) < 5:
            continue

        form_type = parts[0]
        if form_type in target_forms:
            filing_entry: IndexFiling = {
                "form_type": form_type,
                "company_name": " ".join(parts[1:-3]),
                "cik": parts[-3],
                "date_filed": parts[-2],
                "file_name": parts[-1],
            }
            filings.append(filing_entry)

    return filings


def check_filing_for_keywords(
    file_relative_path: str,
    user_agent: str,
    context: _Context,
    rate_limited_get: Callable[..., requests.Response],
) -> bool:
    """Scan a filing stream for material definitive agreement keywords."""
    base_url = "https://www.sec.gov/Archives/"
    full_url = base_url + file_relative_path
    headers = {"User-Agent": user_agent}

    item_pattern = re.compile(
        r"ITEM INFORMATION:.*Entry into a Material Definitive Agreement",
        re.IGNORECASE,
    )
    desc_pattern = re.compile(
        r"<DESCRIPTION>\s*(EX|Exhibit|EX\.)[\s\-]*(10|2)[\.\s]",
        re.IGNORECASE,
    )

    response: requests.Response | None = None
    try:
        response = rate_limited_get(
            full_url, headers=headers, stream=True, timeout=10.0
        )
        response.raise_for_status()
        for line in cast(Iterable[str], response.iter_lines(decode_unicode=True)):
            if line and (item_pattern.search(line) or desc_pattern.search(line)):
                return True
    except Exception as exc:
        context.log.info(f"Error reading filing {file_relative_path}: {exc}")
    finally:
        if response is not None:
            response.close()

    return False


def get_exhibit_links_from_index_page(
    file_path: str,
    user_agent: str,
    context: _Context,
    rate_limited_get: Callable[..., requests.Response],
) -> list[tuple[str, str]]:
    """Visit the filing's index page and scrape Exhibit 10.* and 2.* links.
    
    Returns list of (url, exhibit_type) tuples where exhibit_type is "2" or "10".
    """
    base_sec_url = "https://www.sec.gov/Archives/"

    try:
        filename = file_path.split("/")[-1]
        accession_number = filename.replace(".txt", "")
        accession_no_dashes = accession_number.replace("-", "")
        path_parts = file_path.split("/")
        data_index = path_parts.index("data")
        cik = path_parts[data_index + 1]
    except Exception as exc:
        context.log.info(f"Could not parse path {file_path}: {exc}")
        return []

    index_page_url = (
        f"{base_sec_url}edgar/data/{cik}/{accession_no_dashes}/{accession_number}-index.html"
    )
    file_base_url = f"{base_sec_url}edgar/data/{cik}/{accession_no_dashes}/"

    headers = {"User-Agent": user_agent}
    final_links: list[tuple[str, str]] = []

    try:
        response = rate_limited_get(index_page_url, headers=headers, timeout=10.0)
        response.raise_for_status()
    except Exception as exc:
        context.log.info(f"Error parsing index page {index_page_url}: {exc}")
        return final_links

    soup = BeautifulSoup(response.content, "html.parser")
    tables = soup.find_all("table", class_="tableFile")
    document_table: Tag | None = None
    for table in tables:
        first_row = table.find("tr")
        if first_row is None:
            continue
        if "Document Format Files" in str(table) or "Seq" in str(first_row):
            document_table = table
            break

    if not document_table:
        return final_links

    type_pattern = re.compile(r"(?i)(EX|Exhibit)[\s\-]*(10|2)(\..*)?$")

    for row in document_table.find_all("tr"):
        cols = row.find_all("td")
        if len(cols) < 4:
            continue

        doc_type = cols[3].get_text().strip()
        match = type_pattern.match(doc_type)
        if not match:
            continue
        exhibit_type = match.group(2)  # "2" or "10"

        link_cell = cols[2]
        link_tag = link_cell.find("a")
        if isinstance(link_tag, Tag):
            rel_link = link_tag.get("href")
            if isinstance(rel_link, str):
                if rel_link.startswith("/Archives/"):
                    final_links.append(("https://www.sec.gov" + rel_link, exhibit_type))
                else:
                    final_links.append((file_base_url + rel_link, exhibit_type))

    return final_links


def fetch_material_exhibit_links(
    start_date: str,
    days: int,
    user_agent: str,
    context: _Context,
    rate_limit_max_requests: int = 10,
    rate_limit_window_seconds: float = 1.025,
    max_workers: int = 4,
    use_keyword_filter: bool = True,
) -> list[ExhibitCandidate]:
    """Fetch exhibit links from filings likely to contain definitive agreements."""
    all_exhibit_candidates: list[ExhibitCandidate] = []
    index_urls = get_sec_index_urls(start_date, days)
    context.log.info(
        f"Generated {len(index_urls)} daily index URLs for {days} days after {start_date}"
    )

    request_times: deque[float] = deque()
    rate_lock = threading.Lock()
    thread_local = threading.local()
    sessions: list[requests.Session] = []
    sessions_lock = threading.Lock()

    def acquire_rate_slot() -> None:
        if rate_limit_max_requests <= 0 or rate_limit_window_seconds <= 0:
            return
        while True:
            with rate_lock:
                now = time.monotonic()
                while request_times and now - request_times[0] >= rate_limit_window_seconds:
                    _ = request_times.popleft()
                if len(request_times) < rate_limit_max_requests:
                    request_times.append(now)
                    return
                wait_seconds = rate_limit_window_seconds - (now - request_times[0])
            time.sleep(wait_seconds)

    def get_session() -> requests.Session:
        session = getattr(thread_local, "session", None)
        if session is None:
            session = requests.Session()
            thread_local.session = session
            with sessions_lock:
                sessions.append(session)
        return session

    def make_rate_limited_get(session: requests.Session) -> Callable[..., requests.Response]:
        def rate_limited_get(url: str, **kwargs: Any) -> requests.Response:
            acquire_rate_slot()
            last_exception: Exception | None = None
            for attempt in range(_FETCH_MAX_RETRIES):
                try:
                    response = session.get(url, **kwargs)
                    response.raise_for_status()
                    return response
                except requests.exceptions.HTTPError as e:
                    # Retry on 5xx server errors and 429 rate limiting
                    if e.response is not None and e.response.status_code in {429, 500, 502, 503, 504}:
                        last_exception = e
                        if attempt < _FETCH_MAX_RETRIES - 1:
                            wait_time = _FETCH_RETRY_BACKOFF_BASE ** (attempt + 1)
                            context.log.info(f"Retry {attempt + 1}/{_FETCH_MAX_RETRIES} for {url}: {e.response.status_code}, waiting {wait_time}s")
                            _ = time.sleep(wait_time)
                            continue
                    raise
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    last_exception = e
                    if attempt < _FETCH_MAX_RETRIES - 1:
                        wait_time = _FETCH_RETRY_BACKOFF_BASE ** (attempt + 1)
                        context.log.info(f"Retry {attempt + 1}/{_FETCH_MAX_RETRIES} for {url}: {type(e).__name__}, waiting {wait_time}s")
                        _ = time.sleep(wait_time)
                        continue
                    raise
            # Should not reach here
            if last_exception:
                raise last_exception
            raise RuntimeError("Unexpected: retry loop exited without result")

        return rate_limited_get

    max_workers = max(1, max_workers)
    try:
        for idx_url in index_urls:
            context.log.info(f"Processing index {idx_url}")
            index_start = time.perf_counter()
            index_session = get_session()
            index_rate_limited_get = make_rate_limited_get(index_session)
            filings = parse_index_file(
                idx_url, user_agent, context, index_rate_limited_get
            )
            index_elapsed = time.perf_counter() - index_start
            context.log.info(
                f"Timing: parse_index_file {idx_url} in {index_elapsed:.2f}s"
            )
            context.log.info(
                f"Found {len(filings)} candidate filings matching target form types."
            )

            if not use_keyword_filter:
                context.log.info(
                    "Skipping keyword scan; scraping exhibit links for all filings."
                )
                relevant_filings = filings
            else:
                def keyword_task(filing: IndexFiling) -> tuple[IndexFiling, bool]:
                    file_path = filing["file_name"]
                    keyword_start = time.perf_counter()
                    session = get_session()
                    rate_limited_get = make_rate_limited_get(session)
                    is_relevant = check_filing_for_keywords(
                        file_path, user_agent, context, rate_limited_get
                    )
                    keyword_elapsed = time.perf_counter() - keyword_start
                    context.log.info(
                        f"Timing: check_filing_for_keywords {file_path} in {keyword_elapsed:.2f}s"
                    )
                    return filing, is_relevant

                relevant_filings: list[IndexFiling] = []
                if max_workers == 1:
                    for i, filing in enumerate(filings, start=1):
                        if i % 10 == 0:
                            context.log.info(
                                f"Scanning filing {i}/{len(filings)} for agreement keywords."
                            )
                        filing, is_relevant = keyword_task(filing)
                        if is_relevant:
                            relevant_filings.append(filing)
                else:
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = {
                            executor.submit(keyword_task, filing): filing
                            for filing in filings
                        }
                        for i, future in enumerate(as_completed(futures), start=1):
                            if i % 10 == 0:
                                context.log.info(
                                    f"Scanning filing {i}/{len(filings)} for agreement keywords."
                                )
                            filing, is_relevant = future.result()
                            if is_relevant:
                                relevant_filings.append(filing)

            def exhibits_task(filing: IndexFiling) -> tuple[IndexFiling, list[tuple[str, str]]]:
                file_path = filing["file_name"]
                exhibit_start = time.perf_counter()
                session = get_session()
                rate_limited_get = make_rate_limited_get(session)
                exhibits = get_exhibit_links_from_index_page(
                    file_path, user_agent, context, rate_limited_get
                )
                exhibit_elapsed = time.perf_counter() - exhibit_start
                context.log.info(
                    f"Timing: get_exhibit_links_from_index_page {file_path} in {exhibit_elapsed:.2f}s"
                )
                return filing, exhibits

            def _make_candidates(filing: IndexFiling, exhibits: list[tuple[str, str]]) -> list[ExhibitCandidate]:
                return [
                    ExhibitCandidate(
                        exhibit_url=url,
                        form_type=filing["form_type"],
                        filing_company_name=filing["company_name"],
                        filing_company_cik=filing["cik"],
                        filing_date=filing["date_filed"],
                        exhibit_type=exhibit_type,
                    )
                    for url, exhibit_type in exhibits
                ]

            if max_workers == 1:
                for filing in relevant_filings:
                    company_name = filing["company_name"]
                    form_type = filing["form_type"]
                    context.log.info(f"Match: {company_name} ({form_type})")
                    _, exhibits = exhibits_task(filing)
                    if exhibits:
                        context.log.info(
                            f"Found {len(exhibits)} exhibit(s) for {company_name}."
                        )
                        all_exhibit_candidates.extend(_make_candidates(filing, exhibits))
                    else:
                        context.log.info(
                            f"No Exhibit 10/2 links found for {company_name}."
                        )
            else:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(exhibits_task, filing): filing
                        for filing in relevant_filings
                    }
                    for future in as_completed(futures):
                        filing, exhibits = future.result()
                        company_name = filing["company_name"]
                        form_type = filing["form_type"]
                        context.log.info(f"Match: {company_name} ({form_type})")
                        if exhibits:
                            context.log.info(
                                f"Found {len(exhibits)} exhibit(s) for {company_name}."
                            )
                            all_exhibit_candidates.extend(_make_candidates(filing, exhibits))
                        else:
                            context.log.info(
                                f"No Exhibit 10/2 links found for {company_name}."
                            )
    finally:
        for session in sessions:
            session.close()

    return all_exhibit_candidates


class _CliLogger:
    def info(self, msg: str) -> None:
        print(msg)


class _CliContext:
    _log: _Logger = _CliLogger()

    @property
    def log(self) -> _Logger:
        return self._log


def _default_model_path() -> Path:
    base_dir = Path(__file__).resolve().parents[1]
    return base_dir / "models" / "exhibit_classifier" / "model_files" / "exhibit-classifier.joblib"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run staging exhibit discovery and classification."
    )
    _ = parser.add_argument(
        "--start-date",
        required=True,
        help="Start date in YYYY-MM-DD format.",
    )
    _ = parser.add_argument(
        "--days",
        type=int,
        required=True,
        help="Number of days after start-date to scan.",
    )
    _ = parser.add_argument(
        "--model-path",
        default=str(_default_model_path()),
        help="Path to the trained exhibit classifier model.",
    )
    return parser.parse_args()


def _run_cli() -> None:
    args = _parse_args()
    model_path = Path(cast(str, args.model_path))
    start_date = cast(str, args.start_date)
    days = cast(int, args.days)
    classifier = ExhibitClassifier.load(model_path)
    pipeline_config = PipelineConfig(
        scope=ProcessingScope.BATCHED,
        staging_days_to_fetch=days,
    )
    # Use classify_exhibit_candidates to get ALL candidates for logging
    results = classify_exhibit_candidates(
        exhibit_classifier=classifier,
        context=_CliContext(),
        start_date=start_date,
        pipeline_config=pipeline_config,
    )

    # Write ALL results to output file (not just M&A filings)
    output_dir = Path(__file__).resolve().parents[1] / "models" / "exhibit_classifier" / "data"
    output_file = output_dir / f"staging_data_{start_date}_{days}.txt"
    with open(output_file, "w") as f:
        _ = f.write(f"Found {len(results)} classified exhibit(s).\n")
        for result in results:
            _ = f.write(
                f"{result.candidate_url} prob={result.ma_probability:.3f} is_ma={result.is_ma_agreement}\n"
            )
    print(f"Results written to {output_file}")


if __name__ == "__main__":
    _run_cli()
