# pyright: reportPrivateUsage=false
import unittest
from unittest.mock import Mock, patch

from etl.defs.resources import PipelineConfig
from etl.domain.a_staging import (
    AgreementCandidateResult,
    _compute_content_fingerprint,
    _compute_minhash,
    fetch_new_filings_sec_index,
    should_auto_verify_agreement,
)


class _Logger:
    def info(self, msg: str) -> None:
        _ = msg


class _Context:
    log = _Logger()


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

    def test_fetch_new_filings_sec_index_dedupes_exact_matches_before_lsh(self) -> None:
        pipeline_config = PipelineConfig()
        context = _Context()
        classifier = Mock()
        shared_text = "agreement and plan of merger " * 500
        fingerprint = _compute_content_fingerprint(shared_text)
        candidates = [
            AgreementCandidateResult(
                candidate_url="https://example.com/earlier-no-pagination.htm",
                is_ma_agreement=True,
                ma_probability=0.99,
                form_type="8-K",
                filing_company_name="Target",
                filing_company_cik="1",
                filing_date="20210101",
                exhibit_type="2",
                page_count=8,
                auto_status_verified=False,
                content_fingerprint=fingerprint,
                minhash=_compute_minhash("completely different text " * 300),
            ),
            AgreementCandidateResult(
                candidate_url="https://example.com/later-with-pagination.htm",
                is_ma_agreement=True,
                ma_probability=0.98,
                form_type="S-4",
                filing_company_name="Acquirer",
                filing_company_cik="2",
                filing_date="20210102",
                exhibit_type="2",
                page_count=120,
                auto_status_verified=True,
                content_fingerprint=fingerprint,
                minhash=_compute_minhash("another distinct text block " * 300),
            ),
        ]

        with patch("etl.domain.a_staging.classify_exhibit_candidates", return_value=candidates):
            filings = fetch_new_filings_sec_index(
                exhibit_classifier=classifier,
                context=context,
                start_date="2021-01-01",
                pipeline_config=pipeline_config,
            )

        self.assertEqual(len(filings), 1)
        self.assertEqual(filings[0].url, "https://example.com/later-with-pagination.htm")
        self.assertEqual(
            filings[0].secondary_filing_url,
            "https://example.com/earlier-no-pagination.htm",
        )
        self.assertTrue(filings[0].auto_status_verified)

    def test_fetch_new_filings_sec_index_keeps_near_duplicate_lsh_behavior(self) -> None:
        pipeline_config = PipelineConfig()
        context = _Context()
        classifier = Mock()
        shared_minhash = _compute_minhash("agreement plan merger closing " * 20)

        candidates = [
            AgreementCandidateResult(
                candidate_url="https://example.com/a.htm",
                is_ma_agreement=True,
                ma_probability=0.95,
                form_type="8-K",
                filing_company_name="Target",
                filing_company_cik="1",
                filing_date="20210101",
                exhibit_type="2",
                page_count=50,
                auto_status_verified=True,
                content_fingerprint=_compute_content_fingerprint("agreement version a"),
                minhash=shared_minhash,
            ),
            AgreementCandidateResult(
                candidate_url="https://example.com/b.htm",
                is_ma_agreement=True,
                ma_probability=0.94,
                form_type="S-4",
                filing_company_name="Acquirer",
                filing_company_cik="2",
                filing_date="20210102",
                exhibit_type="2",
                page_count=60,
                auto_status_verified=False,
                content_fingerprint=_compute_content_fingerprint("agreement version b"),
                minhash=shared_minhash,
            ),
        ]

        with patch("etl.domain.a_staging.classify_exhibit_candidates", return_value=candidates):
            filings = fetch_new_filings_sec_index(
                exhibit_classifier=classifier,
                context=context,
                start_date="2021-01-01",
                pipeline_config=pipeline_config,
            )

        self.assertEqual(len(filings), 1)
        self.assertEqual(filings[0].url, "https://example.com/a.htm")
        self.assertEqual(filings[0].secondary_filing_url, "https://example.com/b.htm")
        self.assertTrue(filings[0].auto_status_verified)


if __name__ == "__main__":
    _ = unittest.main()
