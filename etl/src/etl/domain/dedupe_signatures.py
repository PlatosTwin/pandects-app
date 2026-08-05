# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportMissingTypeStubs=false
"""Content signatures and duplicate-detection logic for agreement documents.

Pure domain logic (no DB, no network). Used by the staging pipeline to detect
the same signed agreement ingested at different EDGAR URLs, regardless of which
filer's copy it is or how far apart the filings are.

Design notes (from the 2026-08-05 duplicate audit of 176 known dupe pairs):

- The first-20k-char MinHash caught only 24% of known duplicate pairs at 0.85
  Jaccard, because cover page/TOC formatting is where EDGAR prep vendors differ
  most. Full-text shingles catch 87.5%. Signatures here hash the FULL text.
- Digits and punctuation are stripped from shingles so TOC page numbers and
  section numbering artifacts do not dilute similarity.
- One filer's copy can embed the agreement in a much larger wrapper (a measured
  case had 3.3x the characters), so plain Jaccard under-scores true duplicates.
  We additionally estimate CONTAINMENT (intersection over the smaller shingle
  set), which stays high for a document embedded in a larger wrapper.
- Party identity alone must NEVER auto-dedupe: the same target/acquirer pair
  can sign genuinely different agreements (terminated-then-re-signed deals,
  the same target courting two different SPACs). Party match only forces a
  content comparison.
- The near-perfect identity signal is the cover page's "dated as of <date>".
  Two copies of the same signed agreement always share it. Content hit plus
  matching dated-as-of auto-dedupes; a content hit without it is flagged for
  review instead of auto-deleted, unless the newer document explicitly amends
  and restates (in which case the most recent copy is kept).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Final

from datasketch import MinHash

MINHASH_NUM_PERM: Final[int] = 128
SHINGLE_SIZE: Final[int] = 5

# LSH index threshold is deliberately low: a true duplicate embedded in a 3.3x
# wrapper has Jaccard ~0.3 even though containment is ~1.0, and datasketch's
# banding at 0.3 only surfaces j~0.39 pairs about half the time (a measured
# true-dupe pair was missed). At 0.2 such pairs surface reliably; the extra
# candidate noise is filtered by cheap in-memory verification, and the actual
# dedupe decision is made by decide_duplicate below.
LSH_INDEX_THRESHOLD: Final[float] = 0.2

# Validated on the labeled pairs in dedupe_plan_20260805 (171 true-dupe pairs,
# 5 non-dupe pairs including 4 distinct-deal traps) via
# etl/src/etl/utils/eval_dedupe_similarity.py.
#
# Two tiers:
# - "content hit" (the lower pair): the pair is copies-or-near-copies and at
#   minimum must be flagged for review.
# - "strong" (the higher pair): high enough that, combined with a matching
#   dated-as-of, auto-dedupe is safe. The strong tier exists because an
#   AMENDED agreement keeps the original cover date and can score j~0.89
#   against the original (measured: KushCo/Greenlane pair, j=0.891 c=0.942,
#   labeled non-dupe) -- moderate similarity plus a matching date must only
#   ever reach review, never auto-delete.
JACCARD_DUPLICATE_THRESHOLD: Final[float] = 0.80
CONTAINMENT_DUPLICATE_THRESHOLD: Final[float] = 0.85
STRONG_JACCARD_THRESHOLD: Final[float] = 0.94
STRONG_CONTAINMENT_THRESHOLD: Final[float] = 0.97

# Survivor rule: keep the most recent copy, unless it has drastically less
# parsed content than the other copy (a truncated/partial re-filing must not
# displace a complete document).
DRASTIC_CONTENT_RATIO: Final[float] = 0.5

# Window of leading text scanned for cover-page identity signals. The date
# window is wide because "dated as of" can sit in the recitals after the TOC;
# party names live on the cover/preamble, so their window is narrow to avoid
# harvesting TOC headings as entity names.
COVER_WINDOW_CHARS: Final[int] = 8000
PARTY_WINDOW_CHARS: Final[int] = 2500

_SHINGLE_TOKEN_RE: Final[re.Pattern[str]] = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class DocumentSignature:
    """Content identity of one rendered agreement document."""

    content_fingerprint: str
    minhash: MinHash
    shingle_count: int
    char_count: int
    page_count: int


@dataclass(frozen=True)
class CoverIdentity:
    """High-precision identity signals extracted from the cover pages."""

    dated_as_of: date | None
    party_tokens: frozenset[str]
    amends_and_restates: bool


@dataclass(frozen=True)
class SimilarityScores:
    jaccard: float
    containment: float


class DuplicateAction(str, Enum):
    NONE = "none"
    AUTO_DEDUPE = "auto_dedupe"
    REVIEW = "review"


@dataclass(frozen=True)
class DuplicateDecision:
    action: DuplicateAction
    reason: str
    fingerprint_match: bool
    scores: SimilarityScores


def compute_content_fingerprint(rendered_text: str) -> str:
    """Stable fingerprint for exact duplicate detection.

    Normalizes case and collapses whitespace so filings that differ only in SEC
    wrapper formatting still collapse deterministically.
    """
    normalized = re.sub(r"\s+", " ", rendered_text.lower()).strip()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def normalized_shingles(rendered_text: str) -> set[str]:
    """Word shingles over alphabetic tokens only.

    Stripping digits/punctuation kills TOC page-number noise and pagination
    artifacts that differ between two filers' copies of the same agreement.
    """
    tokens = _SHINGLE_TOKEN_RE.findall(rendered_text.lower())
    if not tokens:
        return set()
    if len(tokens) < SHINGLE_SIZE:
        return set(tokens)
    return {
        " ".join(tokens[i : i + SHINGLE_SIZE])
        for i in range(len(tokens) - SHINGLE_SIZE + 1)
    }


def compute_document_signature(rendered_text: str, *, page_count: int) -> DocumentSignature:
    """Compute the persistent content signature for one document (full text)."""
    shingles = normalized_shingles(rendered_text)
    minhash = MinHash(num_perm=MINHASH_NUM_PERM)
    if shingles:
        minhash.update_batch([shingle.encode("utf-8") for shingle in shingles])
    return DocumentSignature(
        content_fingerprint=compute_content_fingerprint(rendered_text),
        minhash=minhash,
        shingle_count=len(shingles),
        char_count=len(rendered_text),
        page_count=page_count,
    )


def similarity_scores(left: DocumentSignature, right: DocumentSignature) -> SimilarityScores:
    """Estimate Jaccard and containment (intersection over the smaller set).

    Containment is derived from the MinHash Jaccard estimate and the exact
    shingle counts: |A n B| ~= J / (1 + J) * (|A| + |B|).
    """
    jaccard = float(left.minhash.jaccard(right.minhash))
    smaller = min(left.shingle_count, right.shingle_count)
    if smaller <= 0:
        return SimilarityScores(jaccard=jaccard, containment=0.0)
    estimated_intersection = jaccard / (1.0 + jaccard) * (left.shingle_count + right.shingle_count)
    containment = min(1.0, estimated_intersection / smaller)
    return SimilarityScores(jaccard=jaccard, containment=containment)


def is_content_duplicate(left: DocumentSignature, right: DocumentSignature) -> bool:
    if left.content_fingerprint == right.content_fingerprint:
        return True
    scores = similarity_scores(left, right)
    return (
        scores.jaccard >= JACCARD_DUPLICATE_THRESHOLD
        or scores.containment >= CONTAINMENT_DUPLICATE_THRESHOLD
    )


# --- Cover identity extraction -------------------------------------------------

_MONTHS: Final[dict[str, int]] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}

_MONTH_PATTERN: Final[str] = "|".join(
    sorted(_MONTHS.keys(), key=len, reverse=True)
)

# Date bodies, most specific first. Named groups: day / month / year.
_DATE_BODIES: Final[tuple[str, ...]] = (
    # "the 21st day of June, 2021"
    rf"(?:the\s+)?(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+day\s+of\s+(?P<month>{_MONTH_PATTERN})\.?\s*,?\s*(?P<year>\d{{4}})",
    # "June 21, 2021"
    rf"(?P<month>{_MONTH_PATTERN})\.?\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s*,?\s*(?P<year>\d{{4}})",
    # "21 June 2021"
    rf"(?P<day>\d{{1,2}})(?:st|nd|rd|th)?\s+(?P<month>{_MONTH_PATTERN})\.?\s*,?\s*(?P<year>\d{{4}})",
)

_DATE_ANCHORS: Final[tuple[str, ...]] = (
    # "dated as of", "is made and entered into as of", "entered into as of"
    r"(?:dated|made|entered\s+into)[a-z,\s]{0,40}?\bas\s+of\s+(?:the\s+)?",
    # "Dated: June 21, 2021" / "dated June 21, 2021"
    r"dated\s*:?\s+(?:the\s+)?",
)

_DATED_AS_OF_RES: Final[tuple[re.Pattern[str], ...]] = tuple(
    re.compile(anchor + body, re.IGNORECASE)
    for anchor in _DATE_ANCHORS
    for body in _DATE_BODIES
)

# Requires "agreement"/"merger" within a few words after the phrase so that
# boilerplate like "Amended and Restated Certificate of Incorporation" (present
# in ordinary, non-A&R agreements) does not trigger the flag. Matches both the
# title form ("AMENDED AND RESTATED AGREEMENT AND PLAN OF MERGER") and the
# recital form ("amends and restates in its entirety the Agreement ...").
_AMENDS_AND_RESTATES_RE: Final[re.Pattern[str]] = re.compile(
    r"amend(?:s|ed)?\s+and\s+restat(?:es|ed|e)\b(?:\W+\w+){0,6}?\W+(?:agreement|merger)",
    re.IGNORECASE,
)

# OCR/rendering artifacts split digit runs ("202 5" for 2025). Joining adjacent
# digit runs inside the cover window is safe: real dates separate numbers with
# punctuation or month names.
_SPLIT_DIGITS_RE: Final[re.Pattern[str]] = re.compile(r"(?<=\d)\s+(?=\d)")

_CORP_SUFFIX_WORDS: Final[tuple[str, ...]] = (
    "Incorporated",
    "Corporation",
    "Company",
    "Limited",
    "Inc",
    "Corp",
    "LLC",
    "L.L.C",
    "Ltd",
    "L.P",
    "LP",
    "LLP",
    "Co",
    "N.V",
    "B.V",
    "S.A",
    "A.G",
    "GmbH",
    "plc",
    "PLC",
)

_CORP_SUFFIX_PATTERN: Final[str] = "|".join(
    # Match Title-case and ALL-CAPS cover styling, but not lowercase prose like
    # "a Delaware corporation".
    f"(?:{re.escape(word)}|{re.escape(word.upper())})"
    for word in sorted(_CORP_SUFFIX_WORDS, key=len, reverse=True)
)

_PARTY_RE: Final[re.Pattern[str]] = re.compile(
    r"\b([A-Z][A-Za-z0-9&'’\-\.]*(?:\s+(?:[A-Z][A-Za-z0-9&'’\-\.]*|&|and|of|the)){0,6}?)"
    rf",?\s+({_CORP_SUFFIX_PATTERN})\.?(?=[\s,.;:)”\"]|$)"
)

_SUFFIX_TOKENS: Final[frozenset[str]] = frozenset(
    word.upper().replace(".", "") for word in _CORP_SUFFIX_WORDS
)

_PARTY_STOP_TOKENS: Final[frozenset[str]] = frozenset({"THE", "AND", "OF", "A", "AN"})


def normalize_party_name(name: str) -> str:
    """Normalize an entity name for identity comparison.

    Uppercases, strips punctuation, and drops legal-suffix tokens so
    "BlackSky Holdings, Inc." and "BLACKSKY HOLDINGS INC" compare equal.
    """
    cleaned = re.sub(r"[^A-Za-z0-9&\s]", " ", name.upper())
    tokens = [token for token in cleaned.split() if token]
    while tokens and tokens[-1] in _SUFFIX_TOKENS:
        _ = tokens.pop()
    tokens = [token for token in tokens if token not in _PARTY_STOP_TOKENS]
    return " ".join(tokens)


def _parse_extracted_date(day_raw: str, month_raw: str, year_raw: str) -> date | None:
    month = _MONTHS.get(month_raw.lower())
    if month is None:
        return None
    try:
        day = int(day_raw)
        year = int(year_raw)
    except ValueError:
        return None
    if not 1990 <= year <= 2100:
        return None
    try:
        return date(year, month, day)
    except ValueError:
        return None


def extract_dated_as_of(rendered_text: str, *, window_chars: int = COVER_WINDOW_CHARS) -> date | None:
    """Extract the cover page's "dated as of <date>" (and variants).

    Handles "dated as of June 21, 2021", "Dated: June 21, 2021",
    "dated as of the 21st day of June, 2021", "21 June 2021", and OCR
    artifacts that split digit runs ("202 1").
    """
    window = rendered_text[:window_chars]
    window = re.sub(r"\s+", " ", window)
    window = _SPLIT_DIGITS_RE.sub("", window)

    best: tuple[int, date] | None = None
    for pattern in _DATED_AS_OF_RES:
        match = pattern.search(window)
        if match is None:
            continue
        parsed = _parse_extracted_date(
            match.group("day"), match.group("month"), match.group("year")
        )
        if parsed is None:
            continue
        if best is None or match.start() < best[0]:
            best = (match.start(), parsed)
    return None if best is None else best[1]


def extract_party_tokens(
    rendered_text: str, *, window_chars: int = PARTY_WINDOW_CHARS
) -> frozenset[str]:
    """Extract normalized corporate entity names from the cover pages."""
    window = re.sub(r"\s+", " ", rendered_text[:window_chars])
    tokens: set[str] = set()
    for match in _PARTY_RE.finditer(window):
        normalized = normalize_party_name(match.group(1))
        if len(normalized) >= 3:
            tokens.add(normalized)
    return frozenset(tokens)


def extract_cover_identity(
    rendered_text: str, *, window_chars: int = COVER_WINDOW_CHARS
) -> CoverIdentity:
    window = rendered_text[:window_chars]
    return CoverIdentity(
        dated_as_of=extract_dated_as_of(rendered_text, window_chars=window_chars),
        party_tokens=extract_party_tokens(
            rendered_text, window_chars=min(window_chars, PARTY_WINDOW_CHARS)
        ),
        amends_and_restates=_AMENDS_AND_RESTATES_RE.search(window) is not None,
    )


def party_tokens_overlap(left: CoverIdentity, right: CoverIdentity) -> bool:
    return bool(left.party_tokens & right.party_tokens)


# --- Duplicate decision --------------------------------------------------------


def decide_duplicate(
    left_signature: DocumentSignature,
    left_cover: CoverIdentity,
    right_signature: DocumentSignature,
    right_cover: CoverIdentity,
    *,
    newer_amends_and_restates: bool,
    party_match: bool = False,
) -> DuplicateDecision:
    """Decide whether two documents are duplicates and how to act.

    `party_match=True` means the pair was surfaced by a party-identity trigger;
    it NEVER suffices on its own (the same parties can sign genuinely different
    agreements). `newer_amends_and_restates` is the A&R flag of whichever
    document the caller determined to be the more recent filing.
    """
    fingerprint_match = left_signature.content_fingerprint == right_signature.content_fingerprint
    scores = similarity_scores(left_signature, right_signature)
    content_hit = fingerprint_match or (
        scores.jaccard >= JACCARD_DUPLICATE_THRESHOLD
        or scores.containment >= CONTAINMENT_DUPLICATE_THRESHOLD
    )
    if not content_hit:
        reason = "party_match_without_content_match" if party_match else "below_threshold"
        return DuplicateDecision(
            action=DuplicateAction.NONE,
            reason=reason,
            fingerprint_match=fingerprint_match,
            scores=scores,
        )
    if fingerprint_match:
        return DuplicateDecision(
            action=DuplicateAction.AUTO_DEDUPE,
            reason="exact_fingerprint",
            fingerprint_match=True,
            scores=scores,
        )
    strong_hit = (
        scores.jaccard >= STRONG_JACCARD_THRESHOLD
        or scores.containment >= STRONG_CONTAINMENT_THRESHOLD
    )
    if left_cover.dated_as_of is not None and right_cover.dated_as_of is not None:
        if left_cover.dated_as_of == right_cover.dated_as_of:
            if strong_hit:
                return DuplicateDecision(
                    action=DuplicateAction.AUTO_DEDUPE,
                    reason="content_and_dated_as_of",
                    fingerprint_match=False,
                    scores=scores,
                )
            # An amended agreement keeps the original cover date but differs in
            # body content, so a matching date with only moderate similarity
            # must be reviewed, never auto-deleted.
            return DuplicateDecision(
                action=DuplicateAction.REVIEW,
                reason="dated_as_of_match_but_moderate_similarity",
                fingerprint_match=False,
                scores=scores,
            )
        if newer_amends_and_restates:
            # Amended & restated: different dated-as-of but the same live deal.
            # Keep the most recent per the operative-document principle.
            return DuplicateDecision(
                action=DuplicateAction.AUTO_DEDUPE,
                reason="amends_and_restates_keep_most_recent",
                fingerprint_match=False,
                scores=scores,
            )
        return DuplicateDecision(
            action=DuplicateAction.REVIEW,
            reason="dated_as_of_mismatch",
            fingerprint_match=False,
            scores=scores,
        )
    if newer_amends_and_restates:
        return DuplicateDecision(
            action=DuplicateAction.AUTO_DEDUPE,
            reason="amends_and_restates_keep_most_recent",
            fingerprint_match=False,
            scores=scores,
        )
    return DuplicateDecision(
        action=DuplicateAction.REVIEW,
        reason="content_match_without_cover_identity",
        fingerprint_match=False,
        scores=scores,
    )


# --- Survivor selection --------------------------------------------------------


@dataclass(frozen=True)
class SurvivorCandidate:
    agreement_uuid: str
    url: str
    filing_date: date
    ingested_date: datetime | None
    page_count: int | None


def _recency_key(candidate: SurvivorCandidate) -> tuple[date, datetime, str]:
    return (
        candidate.filing_date,
        candidate.ingested_date or datetime.min,
        candidate.url,
    )


def pick_survivor(
    left: SurvivorCandidate, right: SurvivorCandidate
) -> tuple[SurvivorCandidate, SurvivorCandidate]:
    """Operative-document survivor rule.

    Keep the most recent copy (filing_date, then ingested_date), except a copy
    with drastically less parsed content must not survive over a complete one.
    Returns (survivor, loser).
    """
    if _recency_key(left) >= _recency_key(right):
        newer, older = left, right
    else:
        newer, older = right, left
    newer_pages = newer.page_count or 0
    older_pages = older.page_count or 0
    if older_pages > 0 and newer_pages < DRASTIC_CONTENT_RATIO * older_pages:
        return older, newer
    return newer, older
