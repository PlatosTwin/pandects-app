# pyright: reportPrivateUsage=false
import itertools
import string
import unittest
from datetime import date
from unittest.mock import Mock, patch

from etl.defs.resources import PipelineConfig
from etl.domain.a_staging import (
    AgreementCandidateResult,
    _compute_content_fingerprint,
    fetch_new_filings_sec_index,
    should_auto_verify_agreement,
)
from etl.domain.dedupe_signatures import (
    CoverIdentity,
    compute_document_signature,
    extract_cover_identity,
)

_WORDS = ["".join(letters) for letters in itertools.product(string.ascii_lowercase, repeat=3)]


def _text(start: int, count: int) -> str:
    return " ".join(_WORDS[start : start + count])


class _Logger:
    def info(self, msg: str) -> None:
        _ = msg

    def warning(self, msg: str) -> None:
        _ = msg


class _Context:
    log = _Logger()


def _candidate(
    url: str,
    text: str,
    *,
    filing_date: str,
    page_count: int,
    auto_status_verified: bool = False,
    ma_probability: float = 0.95,
    dated_as_of: date | None = None,
    filing_company_name: str = "Filer",
    cik: str = "1",
    form_type: str = "8-K",
) -> AgreementCandidateResult:
    cover = extract_cover_identity(text)
    if dated_as_of is not None:
        cover = CoverIdentity(
            dated_as_of=dated_as_of,
            party_tokens=cover.party_tokens,
            amends_and_restates=cover.amends_and_restates,
            ar_reference_dates=cover.ar_reference_dates,
        )
    return AgreementCandidateResult(
        candidate_url=url,
        is_ma_agreement=True,
        ma_probability=ma_probability,
        form_type=form_type,
        filing_company_name=filing_company_name,
        filing_company_cik=cik,
        filing_date=filing_date,
        exhibit_type="2",
        page_count=page_count,
        auto_status_verified=auto_status_verified,
        signature=compute_document_signature(text, page_count=page_count),
        cover_identity=cover,
    )


class StagingDedupTests(unittest.TestCase):
    def test_should_auto_verify_accepts_mixed_case_phrase_in_first_window(self) -> None:
        text = ("x" * 50) + "AgReEmEnT aNd PlAn Of MeRgEr" + (" y" * 600)
        self.assertTrue(should_auto_verify_agreement(text, 15))

    def test_should_auto_verify_rejects_when_phrase_after_first_500_chars(self) -> None:
        text = ("x" * 501) + " business combination"
        self.assertFalse(should_auto_verify_agreement(text, 20))

    def test_should_auto_verify_rejects_amend_and_restate_case_insensitively_in_first_window(self) -> None:
        self.assertFalse(
            should_auto_verify_agreement(
                "membership interest purchase " + ("z " * 20) + "AmEnD",
                18,
            )
        )
        self.assertFalse(
            should_auto_verify_agreement(
                "business combination " + ("z " * 20) + "ReStAtE",
                18,
            )
        )

    def test_should_auto_verify_allows_amend_and_restate_after_first_window(self) -> None:
        text = "Agreement and Plan of Merger" + ("x" * 520) + "AmEnD ReStAtE"
        self.assertTrue(should_auto_verify_agreement(text, 18))

    def test_should_auto_verify_requires_at_least_15_pages(self) -> None:
        text = "Agreement and Plan of Merger"
        self.assertFalse(should_auto_verify_agreement(text, 14))
        self.assertTrue(should_auto_verify_agreement(text, 15))

    def test_content_fingerprint_collapses_whitespace_only_variants(self) -> None:
        left = "Agreement and Plan of Merger\n\nSection 1.1   The Merger"
        right = "agreement and plan of merger Section 1.1 the merger"
        self.assertEqual(
            _compute_content_fingerprint(left),
            _compute_content_fingerprint(right),
        )

    def test_fetch_new_filings_sec_index_dedupes_exact_matches_keeping_most_recent(self) -> None:
        pipeline_config = PipelineConfig()
        context = _Context()
        classifier = Mock()
        shared_text = _text(0, 2000)
        candidates = [
            _candidate(
                "https://example.com/earlier.htm",
                shared_text,
                filing_date="20210101",
                page_count=80,
                filing_company_name="Target",
            ),
            _candidate(
                "https://example.com/later.htm",
                shared_text,
                filing_date="20210102",
                page_count=120,
                auto_status_verified=True,
                filing_company_name="Acquirer",
                cik="2",
                form_type="S-4",
            ),
        ]

        with patch("etl.domain.a_staging.classify_exhibit_candidates", return_value=candidates):
            staged = fetch_new_filings_sec_index(
                exhibit_classifier=classifier,
                context=context,
                start_date="2021-01-01",
                pipeline_config=pipeline_config,
            )

        self.assertEqual(len(staged), 1)
        # Survivor rule: most recent filing wins.
        self.assertEqual(staged[0].metadata.url, "https://example.com/later.htm")
        self.assertEqual(
            staged[0].metadata.secondary_filing_url,
            "https://example.com/earlier.htm",
        )
        self.assertTrue(staged[0].metadata.auto_status_verified)
        self.assertIsNotNone(staged[0].signature)

    def test_fetch_new_filings_sec_index_merges_wrapper_asymmetric_copy_with_matching_date(self) -> None:
        # One filer's copy embedded in a much larger wrapper (low Jaccard,
        # high containment) with the same cover date must still merge.
        pipeline_config = PipelineConfig()
        context = _Context()
        classifier = Mock()
        inner = _text(0, 2000)
        wrapped = " ".join([inner, _text(4000, 4000)])
        dated = date(2021, 6, 21)
        candidates = [
            _candidate(
                "https://example.com/plain.htm",
                inner,
                filing_date="20210101",
                page_count=60,
                dated_as_of=dated,
            ),
            _candidate(
                "https://example.com/wrapped.htm",
                wrapped,
                filing_date="20210102",
                page_count=70,
                dated_as_of=dated,
                cik="2",
            ),
        ]

        with patch("etl.domain.a_staging.classify_exhibit_candidates", return_value=candidates):
            staged = fetch_new_filings_sec_index(
                exhibit_classifier=classifier,
                context=context,
                start_date="2021-01-01",
                pipeline_config=pipeline_config,
            )

        self.assertEqual(len(staged), 1)
        self.assertEqual(staged[0].metadata.url, "https://example.com/wrapped.htm")
        self.assertEqual(
            staged[0].metadata.secondary_filing_url,
            "https://example.com/plain.htm",
        )

    def test_fetch_new_filings_sec_index_keeps_both_when_dated_as_of_differs(self) -> None:
        # Near-identical content but different cover dates (e.g. original vs
        # re-signed deal) must NOT silently merge in the batch; both are staged
        # and the corpus reconciliation flags the pair for review.
        pipeline_config = PipelineConfig()
        context = _Context()
        classifier = Mock()
        base = _text(0, 2000)
        candidates = [
            _candidate(
                "https://example.com/march.htm",
                base,
                filing_date="20210101",
                page_count=60,
                dated_as_of=date(2021, 3, 21),
            ),
            _candidate(
                "https://example.com/september.htm",
                base + " " + _text(10000, 10),
                filing_date="20210102",
                page_count=60,
                dated_as_of=date(2021, 9, 15),
                cik="2",
            ),
        ]

        with patch("etl.domain.a_staging.classify_exhibit_candidates", return_value=candidates):
            staged = fetch_new_filings_sec_index(
                exhibit_classifier=classifier,
                context=context,
                start_date="2021-01-01",
                pipeline_config=pipeline_config,
            )

        self.assertEqual(len(staged), 2)

    def test_fetch_new_filings_sec_index_drastic_content_deficit_does_not_survive(self) -> None:
        pipeline_config = PipelineConfig()
        context = _Context()
        classifier = Mock()
        shared_text = _text(0, 2000)
        candidates = [
            _candidate(
                "https://example.com/complete.htm",
                shared_text,
                filing_date="20210101",
                page_count=120,
            ),
            _candidate(
                "https://example.com/truncated.htm",
                shared_text,
                filing_date="20210102",
                page_count=8,
                cik="2",
            ),
        ]

        with patch("etl.domain.a_staging.classify_exhibit_candidates", return_value=candidates):
            staged = fetch_new_filings_sec_index(
                exhibit_classifier=classifier,
                context=context,
                start_date="2021-01-01",
                pipeline_config=pipeline_config,
            )

        self.assertEqual(len(staged), 1)
        self.assertEqual(staged[0].metadata.url, "https://example.com/complete.htm")

    def test_fetch_new_filings_sec_index_never_merges_same_accession_siblings(self) -> None:
        # Sibling exhibits of one filing (ex2-1 vs ex2-2) are different
        # agreements even with near-identical boilerplate and matching dates.
        pipeline_config = PipelineConfig()
        context = _Context()
        classifier = Mock()
        shared_text = _text(0, 2000)
        accession = "https://www.sec.gov/Archives/edgar/data/1799983/000121390021020720"
        candidates = [
            _candidate(
                f"{accession}/ea139171ex2-1_riceacq.htm",
                shared_text,
                filing_date="20210407",
                page_count=90,
                dated_as_of=date(2021, 4, 7),
            ),
            _candidate(
                f"{accession}/ea139171ex2-2_riceacq.htm",
                shared_text,
                filing_date="20210407",
                page_count=95,
                dated_as_of=date(2021, 4, 7),
                cik="2",
            ),
        ]

        with patch("etl.domain.a_staging.classify_exhibit_candidates", return_value=candidates):
            staged = fetch_new_filings_sec_index(
                exhibit_classifier=classifier,
                context=context,
                start_date="2021-04-06",
                pipeline_config=pipeline_config,
            )

        self.assertEqual(len(staged), 2)


if __name__ == "__main__":
    _ = unittest.main()
