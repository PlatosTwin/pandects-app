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

import requests
from bs4 import BeautifulSoup
from bs4.element import Tag
from requests.api import get as requests_get
from requests.models import Response
from requests.sessions import Session

from etl.domain.b_pre_processing import format_content, split_to_pages
from etl.domain.dedupe_signatures import (
    CoverIdentity,
    DocumentSignature,
    DuplicateAction,
    LSH_INDEX_THRESHOLD,
    SurvivorCandidate,
    compute_content_fingerprint,
    compute_document_signature,
    decide_duplicate,
    extract_cover_identity,
    pick_survivor,
)
from etl.defs.resources import PipelineConfig
from etl.models.exhibit_classifier.exhibit_classifier import ExhibitClassifier
from etl.utils.sec_utils import SEC_USER_AGENT


@dataclass
class FilingMetadata:
    """Metadata for a filing document.
    
    Supports two schemas:
    - DMA corpus: has target, acquirer, announce_date, filing_date as None (no SEC filing date available)
    - SEC index: has prob_filing, filing_company_name, filing_company_cik, form_type, filing_date as str
    """
    agreement_uuid: str
    url: str
    filing_date: datetime.datetime | str | None = None  # str (YYYYMMDD) for SEC index, None for DMA corpus
    # DMA corpus fields (optional for SEC index flow)
    target: str | None = None
    acquirer: str | None = None
    announce_date: datetime.datetime | None = None  # Deal announcement date (from DMA corpus)
    # SEC index fields (optional for DMA corpus flow)
    prob_filing: float | None = None
    filing_company_name: str | None = None
    filing_company_cik: str | None = None
    form_type: str | None = None
    exhibit_type: str | None = None  # "2" or "10"
    secondary_filing_url: str | None = None  # URL of duplicate filing if detected
    auto_status_verified: bool = False


def get_uuid(x: str) -> str:
    """Generate a UUID5 hash from the input string."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, x))


class ExhibitClassifierProtocol(Protocol):
    decision_threshold: float
    min_chars_hard_negative: int

    def predict(self, text: str, threshold: float | None = None) -> bool: ...

    def predict_proba(self, text: str) -> float: ...

    def predict_proba_batch(self, texts: list[str]) -> list[float]: ...


class _Logger(Protocol):
    def info(self, msg: str) -> None: ...

    def warning(self, msg: str) -> None: ...


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
    page_count: int
    auto_status_verified: bool
    # Full-text content signature and cover identity for duplicate detection.
    # None when computation failed; such candidates are never silently merged
    # and get recorded in pending-reconciliation state by the staging asset.
    signature: DocumentSignature | None
    cover_identity: CoverIdentity | None


@dataclass(frozen=True)
class StagedFiling:
    """A staged filing plus the content signature persisted alongside it."""

    metadata: FilingMetadata
    signature: DocumentSignature | None
    cover_identity: CoverIdentity | None


@dataclass(frozen=True)
class ExhibitSignature:
    page_count: int
    auto_status_verified: bool
    content_fingerprint: str
    minhash: MinHash


class SecDailyIndexUnavailable(Exception):
    """Raised when a requested SEC daily index is unavailable to the ingest job."""

    def __init__(self, index_url: str) -> None:
        self.index_url = index_url
        super().__init__(f"SEC daily index unavailable: {index_url}")


class IndexFiling(TypedDict):
    form_type: str
    company_name: str
    cik: str
    date_filed: str
    file_name: str


# MinHash parameters for near-duplicate detection
_MINHASH_NUM_PERM = 128  # Number of permutations (higher = more accurate, slower)
_MINHASH_SHINGLE_SIZE = 5  # Size of word shingles
_AUTO_VERIFY_FIRST_CHARS = 500
_AUTO_VERIFY_MIN_PAGES = 15
_AUTO_VERIFY_INCLUDE_PHRASES = (
    "agreement and plan of merger",
    "business combination",
    "membership interest purchase",
)
_AUTO_VERIFY_EXCLUDE_PHRASES = ("amend", "restate")


def _compute_minhash(rendered_text: str) -> MinHash:
    """Compute a MinHash signature for near-duplicate detection.
    
    Uses only the first 20,000 characters to focus on the main agreement content
    (roughly the first N pages equivalent) and ignore exhibit sections that may 
    differ between filings.
    Uses word-level shingles (n-grams) for robustness against minor text differences.
    For very short texts (fewer words than shingle size), falls back to individual words.
    """
    # Use only the first 20,000 characters (roughly first N pages equivalent)
    text = rendered_text[:20000]
    
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


def _compute_content_fingerprint(rendered_text: str) -> str:
    """Compute a stable fingerprint for exact duplicate detection.

    Delegates to dedupe_signatures.compute_content_fingerprint (kept as a
    module-level name for the legacy ExhibitSignature path and tests).
    """
    return compute_content_fingerprint(rendered_text)


def fetch_document_signature(
    exhibit_url: str, *, user_agent: str = SEC_USER_AGENT
) -> tuple[DocumentSignature, CoverIdentity] | None:
    """Fetch a document and compute its full-text signature and cover identity.

    Returns None for unsupported file types. Raises on fetch failures so
    callers can record the candidate for retry instead of silently skipping.
    """
    fetch_result = _fetch_exhibit_content(exhibit_url, user_agent=user_agent)
    if fetch_result is None:
        return None
    content, is_txt, is_html = fetch_result
    agreement_text, page_count = _render_agreement_text_and_page_count(
        content, is_txt=is_txt, is_html=is_html
    )
    return (
        compute_document_signature(agreement_text, page_count=page_count),
        extract_cover_identity(agreement_text),
    )


def fetch_exhibit_signature(exhibit_url: str, *, user_agent: str = SEC_USER_AGENT) -> ExhibitSignature | None:
    fetch_result = _fetch_exhibit_content(exhibit_url, user_agent=user_agent)
    if fetch_result is None:
        return None
    content, is_txt, is_html = fetch_result
    agreement_text, page_count = _render_agreement_text_and_page_count(
        content, is_txt=is_txt, is_html=is_html
    )
    return ExhibitSignature(
        page_count=page_count,
        auto_status_verified=should_auto_verify_agreement(agreement_text, page_count),
        content_fingerprint=_compute_content_fingerprint(agreement_text),
        minhash=_compute_minhash(agreement_text),
    )


def should_auto_verify_agreement(rendered_text: str, page_count: int) -> bool:
    """Return whether the rendered agreement qualifies for staging auto-verify."""
    if page_count < _AUTO_VERIFY_MIN_PAGES:
        return False

    normalized_text = rendered_text.casefold()
    first_window = normalized_text[:_AUTO_VERIFY_FIRST_CHARS]

    if not any(phrase in first_window for phrase in _AUTO_VERIFY_INCLUDE_PHRASES):
        return False
    if any(phrase in first_window for phrase in _AUTO_VERIFY_EXCLUDE_PHRASES):
        return False
    return True


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
        days_override: Number of days to fetch starting from start_date.
                       Useful for day-by-day processing.
    """
    if days_override is None:
        raise ValueError("days_override is required for classify_exhibit_candidates.")
    days_to_fetch = days_override
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
    signatures: list[DocumentSignature | None] = []
    cover_identities: list[CoverIdentity | None] = []
    page_counts: list[int] = []
    auto_statuses_verified: list[bool] = []
    cached_exhibits: dict[
        str, tuple[str, int, DocumentSignature | None, CoverIdentity | None, bool] | None
    ] = {}

    for candidate in exhibit_candidates:
        cached = cached_exhibits.get(candidate.exhibit_url)
        if cached is None and candidate.exhibit_url in cached_exhibits:
            context.log.info(f"Skipping unsupported file type: {candidate.exhibit_url}")
            continue
        if cached is None:
            try:
                fetch_result = _fetch_exhibit_content(
                    candidate.exhibit_url, user_agent=SEC_USER_AGENT
                )
            except Exception as e:
                # Log and skip this candidate rather than crashing the whole batch
                context.log.info(f"Failed to fetch {candidate.exhibit_url} after retries: {e}")
                continue

            if fetch_result is None:
                cached_exhibits[candidate.exhibit_url] = None
                context.log.info(f"Skipping unsupported file type: {candidate.exhibit_url}")
                continue
            content, is_txt, is_html = fetch_result
            agreement_text, page_count = _render_agreement_text_and_page_count(
                content, is_txt=is_txt, is_html=is_html
            )
            auto_status_verified = should_auto_verify_agreement(
                agreement_text,
                page_count,
            )
            signature: DocumentSignature | None
            cover_identity: CoverIdentity | None
            try:
                signature = compute_document_signature(agreement_text, page_count=page_count)
                cover_identity = extract_cover_identity(agreement_text)
            except Exception as e:
                # Never silently drop the candidate: it is still ingested, and
                # the staging asset records it for pending reconciliation.
                context.log.warning(
                    f"Signature computation failed for {candidate.exhibit_url}: {e}"
                )
                signature = None
                cover_identity = None
            cached_exhibits[candidate.exhibit_url] = (
                agreement_text,
                page_count,
                signature,
                cover_identity,
                auto_status_verified,
            )
        else:
            agreement_text, page_count, signature, cover_identity, auto_status_verified = cached

        agreement_texts.append(agreement_text)
        signatures.append(signature)
        cover_identities.append(cover_identity)
        page_counts.append(page_count)
        auto_statuses_verified.append(auto_status_verified)
        valid_candidates.append(candidate)

    # Early return if no valid candidates (all fetches failed/skipped)
    if not valid_candidates:
        context.log.info("No valid exhibit candidates to classify after fetching.")
        return []

    probabilities = exhibit_classifier.predict_proba_batch(agreement_texts)
    threshold = exhibit_classifier.decision_threshold
    min_chars_hard_negative = exhibit_classifier.min_chars_hard_negative
    is_ma_predictions = [
        len(str(text)) >= min_chars_hard_negative and probability >= threshold
        for text, probability in zip(agreement_texts, probabilities)
    ]

    results: list[AgreementCandidateResult] = []
    for idx, candidate in enumerate(valid_candidates):
        probability = probabilities[idx]
        is_ma = is_ma_predictions[idx]
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
                page_count=page_counts[idx],
                auto_status_verified=auto_statuses_verified[idx],
                signature=signatures[idx],
                cover_identity=cover_identities[idx],
            )
        )

    return results


def _parse_idx_filing_date(filing_date: str) -> datetime.date:
    return datetime.datetime.strptime(filing_date, "%Y%m%d").date()


def _candidate_survivor_input(idx: int, candidate: AgreementCandidateResult) -> SurvivorCandidate:
    # Encode batch order in ingested_date so recency tie-breaks follow .idx
    # file order deterministically (later in the index = more recent).
    return SurvivorCandidate(
        agreement_uuid=get_uuid(candidate.candidate_url),
        url=candidate.candidate_url,
        filing_date=_parse_idx_filing_date(candidate.filing_date),
        ingested_date=datetime.datetime.min + datetime.timedelta(seconds=idx),
        page_count=candidate.page_count,
    )


def _should_merge_batch_pair(
    left_idx: int,
    left: AgreementCandidateResult,
    right_idx: int,
    right: AgreementCandidateResult,
) -> bool:
    """Decide whether two same-batch candidates are copies of the same agreement.

    Only auto-merges on an exact fingerprint match or a content hit confirmed by
    cover identity (matching dated-as-of, or an amends-and-restates newer copy).
    Unconfirmed content hits are left un-merged on purpose: both copies get
    ingested and the corpus-wide reconciliation flags the pair for review
    instead of silently dropping one.
    """
    if left.signature is None or right.signature is None:
        return False
    left_cover = left.cover_identity
    right_cover = right.cover_identity
    if left_cover is None or right_cover is None:
        return left.signature.content_fingerprint == right.signature.content_fingerprint
    left_key = (left.filing_date, left_idx)
    right_key = (right.filing_date, right_idx)
    newer_cover = left_cover if left_key >= right_key else right_cover
    decision = decide_duplicate(
        left.signature,
        left_cover,
        right.signature,
        right_cover,
        newer_amends_and_restates=newer_cover.amends_and_restates,
    )
    return decision.action is DuplicateAction.AUTO_DEDUPE


def fetch_new_filings_sec_index(
    exhibit_classifier: ExhibitClassifierProtocol,
    context: _Context,
    start_date: str,
    pipeline_config: PipelineConfig,
    *,
    days_override: int | None = None,
) -> list[StagedFiling]:
    """
    Fetch agreement candidates from SEC indexes and filter using the exhibit classifier.

    Returns only filings classified as M&A agreements under the model's
    decision threshold and hard-negative guards, each paired with its full-text
    content signature for persistence.

    De-duplicates same-batch copies of the same agreement (same agreement filed
    by target and acquirer, possibly with heavy wrapper/formatting differences)
    using full-text MinHash LSH at a low index threshold plus Jaccard/containment
    verification, gated on cover-identity confirmation.

    Args:
        days_override: Number of days to fetch starting from start_date.
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

    # LSH over FULL-TEXT minhashes at a low threshold: a copy embedded in a
    # larger wrapper can have low Jaccard despite near-total containment, so
    # the index only surfaces candidates and _should_merge_batch_pair decides.
    lsh = MinHashLSH(threshold=LSH_INDEX_THRESHOLD, num_perm=_MINHASH_NUM_PERM)
    fingerprint_first_seen: dict[str, int] = {}

    for idx, candidate in enumerate(ma_candidates):
        if candidate.signature is None:
            continue
        first_seen_idx = fingerprint_first_seen.get(candidate.signature.content_fingerprint)
        if first_seen_idx is None:
            fingerprint_first_seen[candidate.signature.content_fingerprint] = idx
        else:
            union(idx, first_seen_idx)

        # Query for similar items before inserting
        similar_keys = lsh.query(candidate.signature.minhash)
        for key in similar_keys:
            other_idx = int(cast(str, key))
            if _should_merge_batch_pair(
                other_idx, ma_candidates[other_idx], idx, candidate
            ):
                union(idx, other_idx)
        # Insert with index as key
        lsh.insert(str(idx), candidate.signature.minhash)

    # Group by root parent
    groups: dict[int, list[tuple[int, AgreementCandidateResult]]] = {}
    for idx, candidate in enumerate(ma_candidates):
        root = find(idx)
        if root not in groups:
            groups[root] = []
        groups[root].append((idx, candidate))

    results: list[StagedFiling] = []
    for _, group in groups.items():
        # Stable order: filing_date, then original index (.idx file order)
        group.sort(key=lambda x: (x[1].filing_date, x[0]))

        secondary_url: str | None = None
        primary_idx, primary = group[0]

        if len(group) > 1:
            # Survivor rule: keep the most recent copy, unless it has
            # drastically less parsed content than a complete one.
            survivor_input = _candidate_survivor_input(primary_idx, primary)
            by_url = {candidate.candidate_url: (idx, candidate) for idx, candidate in group}
            for idx, candidate in group[1:]:
                survivor_input, _ = pick_survivor(
                    survivor_input, _candidate_survivor_input(idx, candidate)
                )
            primary_idx, primary = by_url[survivor_input.url]
            other_urls = [
                c.candidate_url for _, c in group if c.candidate_url != primary.candidate_url
            ]
            secondary_url = other_urls[0] if other_urls else None
            context.log.info(
                f"De-dup: {primary.candidate_url} (primary, survivor rule) has near-duplicate(s): {', '.join(other_urls)}"
            )

        results.append(
            StagedFiling(
                metadata=FilingMetadata(
                    agreement_uuid=get_uuid(primary.candidate_url),
                    url=primary.candidate_url,
                    filing_date=primary.filing_date,
                    prob_filing=primary.ma_probability,
                    filing_company_name=primary.filing_company_name,
                    filing_company_cik=primary.filing_company_cik,
                    form_type=primary.form_type,
                    exhibit_type=primary.exhibit_type,
                    secondary_filing_url=secondary_url,
                    auto_status_verified=primary.auto_status_verified,
                ),
                signature=primary.signature,
                cover_identity=primary.cover_identity,
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
            response = requests_get(exhibit_url, headers=headers, timeout=timeout)
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


def _render_agreement_text_and_page_count(
    content: str, is_txt: bool, is_html: bool
) -> tuple[str, int]:
    pages = split_to_pages(content, is_txt=is_txt, is_html=is_html)
    rendered_pages = [
        format_content(page["content"], is_txt=is_txt, is_html=is_html).strip()
        for page in pages
    ]
    return "\n\n".join(page for page in rendered_pages if page), len(pages)


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
    rate_limited_get: Callable[..., Response],
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
    except requests.exceptions.HTTPError as exc:
        status_code = exc.response.status_code if exc.response is not None else None
        if status_code in {403, 404}:
            context.log.info(
                f"SEC daily index unavailable ({status_code}) for ingest: {index_url}"
            )
            raise SecDailyIndexUnavailable(index_url) from exc
        raise

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
    rate_limited_get: Callable[..., Response],
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

    response: Response | None = None
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
    rate_limited_get: Callable[..., Response],
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

    soup = BeautifulSoup(response.text, "html.parser")
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
    sessions: list[Session] = []
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

    def get_session() -> Session:
        session = getattr(thread_local, "session", None)
        if session is None:
            session = Session()
            thread_local.session = session
            with sessions_lock:
                sessions.append(session)
        return session

    def make_rate_limited_get(session: Session) -> Callable[..., Response]:
        def rate_limited_get(url: str, **kwargs: Any) -> Response:
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

    def warning(self, msg: str) -> None:
        print(f"WARNING: {msg}")


class _CliContext:
    _log: _Logger = _CliLogger()

    @property
    def log(self) -> _Logger:
        return self._log


def _default_model_path() -> Path:
    base_dir = Path(__file__).resolve().parents[1]
    return base_dir / "models" / "exhibit_classifier" / "model_files" / "exhibit-classifier.joblib"


# CLI for debug/troubleshoot (no Dagster, no DB): from repo root:
#   cd etl && uv run python -m etl.domain.a_staging --start-date YYYY-MM-DD --days N
# Optional: --model-path /path/to/exhibit-classifier.joblib
# Writes results to etl/src/etl/models/exhibit_classifier/data/staging_data_<start>_<days>.txt


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
    pipeline_config = PipelineConfig()
    # Use classify_exhibit_candidates to get ALL candidates for logging
    results = classify_exhibit_candidates(
        exhibit_classifier=classifier,
        context=_CliContext(),
        start_date=start_date,
        pipeline_config=pipeline_config,
        days_override=days,
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
