# pyright: reportPrivateUsage=false
import itertools
import string
import unittest
from datetime import date, datetime

from etl.domain.dedupe_signatures import (
    CoverIdentity,
    DuplicateAction,
    SurvivorCandidate,
    compute_document_signature,
    decide_duplicate,
    extract_cover_identity,
    extract_dated_as_of,
    extract_party_tokens,
    is_content_duplicate,
    normalize_party_name,
    normalized_shingles,
    pick_survivor,
    similarity_scores,
)

_WORDS = ["".join(letters) for letters in itertools.product(string.ascii_lowercase, repeat=3)]


def _text(start: int, count: int) -> str:
    return " ".join(_WORDS[start : start + count])


def _sig(text: str, page_count: int = 100):
    return compute_document_signature(text, page_count=page_count)


_NO_COVER = CoverIdentity(dated_as_of=None, party_tokens=frozenset(), amends_and_restates=False)


def _cover(
    dated: date | None,
    *,
    amends: bool = False,
    parties: frozenset[str] = frozenset(),
) -> CoverIdentity:
    return CoverIdentity(dated_as_of=dated, party_tokens=parties, amends_and_restates=amends)


class SimilarityTests(unittest.TestCase):
    def test_identical_documents_score_full_similarity(self) -> None:
        text = _text(0, 2000)
        scores = similarity_scores(_sig(text), _sig(text))
        self.assertGreater(scores.jaccard, 0.99)
        self.assertGreater(scores.containment, 0.99)

    def test_document_embedded_in_larger_wrapper_has_high_containment(self) -> None:
        # A true copy embedded in a 4x wrapper: low Jaccard, near-total containment.
        inner = _text(0, 2000)
        wrapper = " ".join([inner, _text(4000, 6000)])
        scores = similarity_scores(_sig(inner), _sig(wrapper))
        self.assertLess(scores.jaccard, 0.5)
        self.assertGreater(scores.containment, 0.9)
        self.assertTrue(is_content_duplicate(_sig(inner), _sig(wrapper)))

    def test_distinct_documents_are_not_duplicates(self) -> None:
        left = _sig(_text(0, 2000))
        right = _sig(_text(3000, 2000))
        scores = similarity_scores(left, right)
        self.assertLess(scores.jaccard, 0.05)
        self.assertLess(scores.containment, 0.1)
        self.assertFalse(is_content_duplicate(left, right))

    def test_shingles_ignore_page_numbers_and_punctuation(self) -> None:
        base_words = _text(0, 200)
        with_noise = " 17 ".join(base_words.split(" wor ")) if " wor " in base_words else base_words
        noisy = " ".join(
            word if index % 10 else f"{word} 42 ..." for index, word in enumerate(base_words.split())
        )
        self.assertEqual(normalized_shingles(base_words), normalized_shingles(noisy))
        _ = with_noise


class CoverIdentityTests(unittest.TestCase):
    def test_dated_as_of_plain_form(self) -> None:
        text = "AGREEMENT AND PLAN OF MERGER dated as of June 21, 2021 among Alpha Corp."
        self.assertEqual(extract_dated_as_of(text), date(2021, 6, 21))

    def test_dated_as_of_day_of_variant(self) -> None:
        text = "This Agreement, dated as of the 21st day of June, 2021, is entered into"
        self.assertEqual(extract_dated_as_of(text), date(2021, 6, 21))

    def test_dated_colon_form(self) -> None:
        text = "AGREEMENT AND PLAN OF MERGER\n\nDated: June 21, 2021\n\namong the parties"
        self.assertEqual(extract_dated_as_of(text), date(2021, 6, 21))

    def test_entered_into_as_of_form(self) -> None:
        text = "This Agreement is made and entered into as of March 3, 2022 by and among"
        self.assertEqual(extract_dated_as_of(text), date(2022, 3, 3))

    def test_ocr_split_year_artifact(self) -> None:
        text = "AGREEMENT AND PLAN OF MERGER dated as of June 21, 202 1 among the parties"
        self.assertEqual(extract_dated_as_of(text), date(2021, 6, 21))

    def test_ocr_split_day_artifact(self) -> None:
        text = "dated as of the 2 1st day of June, 2021"
        self.assertEqual(extract_dated_as_of(text), date(2021, 6, 21))

    def test_day_month_year_form(self) -> None:
        text = "MERGER AGREEMENT dated as of 21 June 2021 between the parties"
        self.assertEqual(extract_dated_as_of(text), date(2021, 6, 21))

    def test_no_date_returns_none(self) -> None:
        self.assertIsNone(extract_dated_as_of("AGREEMENT AND PLAN OF MERGER among parties"))

    def test_invalid_calendar_date_is_rejected(self) -> None:
        self.assertIsNone(extract_dated_as_of("dated as of February 30, 2021"))

    def test_party_tokens_extracted_and_normalized(self) -> None:
        text = (
            "AGREEMENT AND PLAN OF MERGER by and among Osprey Technology Acquisition Corp., "
            "a Delaware corporation, Osprey Technology Merger Sub, Inc., and "
            "BlackSky Holdings, Inc."
        )
        tokens = extract_party_tokens(text)
        self.assertIn("OSPREY TECHNOLOGY ACQUISITION", tokens)
        self.assertIn("BLACKSKY HOLDINGS", tokens)

    def test_party_tokens_match_across_casing_and_punctuation(self) -> None:
        self.assertEqual(
            normalize_party_name("BlackSky Holdings, Inc."),
            normalize_party_name("BLACKSKY HOLDINGS INC"),
        )

    def test_amends_and_restates_detected_for_agreement_title(self) -> None:
        text = "AMENDED AND RESTATED AGREEMENT AND PLAN OF MERGER dated as of April 1, 2021"
        self.assertTrue(extract_cover_identity(text).amends_and_restates)

    def test_amends_and_restates_recital_form_detected(self) -> None:
        text = "This Agreement amends and restates in its entirety the Agreement and Plan of Merger"
        self.assertTrue(extract_cover_identity(text).amends_and_restates)

    def test_amended_certificate_boilerplate_does_not_trigger(self) -> None:
        text = (
            "AGREEMENT AND PLAN OF MERGER dated as of March 31, 2021. The Amended and "
            "Restated Certificate of Incorporation of the Company shall remain in effect."
        )
        self.assertFalse(extract_cover_identity(text).amends_and_restates)


class DecideDuplicateTests(unittest.TestCase):
    def setUp(self) -> None:
        base = _text(0, 3000)
        near_copy = base + " " + _text(10000, 30)
        moderate_copy = " ".join([_text(0, 2700), _text(10000, 300)])
        self.base_sig = _sig(base)
        self.near_sig = _sig(near_copy)
        self.moderate_sig = _sig(moderate_copy)
        self.dated = date(2021, 6, 21)

    def test_exact_fingerprint_auto_dedupes(self) -> None:
        decision = decide_duplicate(
            self.base_sig, _NO_COVER, self.base_sig, _NO_COVER,
            newer_amends_and_restates=False,
        )
        self.assertIs(decision.action, DuplicateAction.AUTO_DEDUPE)
        self.assertEqual(decision.reason, "exact_fingerprint")

    def test_strong_content_with_matching_dated_as_of_auto_dedupes(self) -> None:
        decision = decide_duplicate(
            self.base_sig, _cover(self.dated), self.near_sig, _cover(self.dated),
            newer_amends_and_restates=False,
        )
        self.assertIs(decision.action, DuplicateAction.AUTO_DEDUPE)
        self.assertEqual(decision.reason, "content_and_dated_as_of")

    def test_moderate_content_with_matching_dated_as_of_goes_to_review(self) -> None:
        # The amended-agreement trap: original cover date retained but body
        # differs (measured KushCo/Greenlane at j=0.891) -- must never auto-delete.
        scores = similarity_scores(self.base_sig, self.moderate_sig)
        self.assertGreaterEqual(scores.jaccard, 0.6)
        self.assertLess(scores.jaccard, 0.94)
        decision = decide_duplicate(
            self.base_sig, _cover(self.dated), self.moderate_sig, _cover(self.dated),
            newer_amends_and_restates=False,
        )
        self.assertIs(decision.action, DuplicateAction.REVIEW)
        self.assertEqual(decision.reason, "dated_as_of_match_but_moderate_similarity")

    def test_content_hit_with_dated_mismatch_goes_to_review(self) -> None:
        decision = decide_duplicate(
            self.base_sig, _cover(self.dated), self.near_sig, _cover(date(2021, 9, 15)),
            newer_amends_and_restates=False,
        )
        self.assertIs(decision.action, DuplicateAction.REVIEW)
        self.assertEqual(decision.reason, "dated_as_of_mismatch")

    def test_content_hit_with_dated_mismatch_and_newer_amends_and_restates_auto_dedupes(self) -> None:
        decision = decide_duplicate(
            self.base_sig, _cover(self.dated), self.near_sig, _cover(date(2021, 9, 15)),
            newer_amends_and_restates=True,
        )
        self.assertIs(decision.action, DuplicateAction.AUTO_DEDUPE)
        self.assertEqual(decision.reason, "amends_and_restates_keep_most_recent")

    def test_content_hit_without_cover_identity_goes_to_review(self) -> None:
        decision = decide_duplicate(
            self.base_sig, _NO_COVER, self.near_sig, _NO_COVER,
            newer_amends_and_restates=False,
        )
        self.assertIs(decision.action, DuplicateAction.REVIEW)
        self.assertEqual(decision.reason, "content_match_without_cover_identity")

    def test_party_match_alone_never_dedupes(self) -> None:
        parties = frozenset({"BLACKSKY HOLDINGS", "OSPREY TECHNOLOGY ACQUISITION"})
        left_sig = _sig(_text(0, 2000))
        right_sig = _sig(_text(3000, 2000))
        decision = decide_duplicate(
            left_sig, _cover(self.dated, parties=parties),
            right_sig, _cover(self.dated, parties=parties),
            newer_amends_and_restates=False,
            party_match=True,
        )
        self.assertIs(decision.action, DuplicateAction.NONE)
        self.assertEqual(decision.reason, "party_match_without_content_match")


class SurvivorTests(unittest.TestCase):
    @staticmethod
    def _candidate(
        uuid: str,
        filing: date,
        *,
        ingested: datetime | None = None,
        pages: int | None = 100,
    ) -> SurvivorCandidate:
        return SurvivorCandidate(
            agreement_uuid=uuid,
            url=f"https://example.com/{uuid}.htm",
            filing_date=filing,
            ingested_date=ingested,
            page_count=pages,
        )

    def test_most_recent_filing_survives(self) -> None:
        early = self._candidate("early", date(2021, 4, 1))
        late = self._candidate("late", date(2021, 7, 9))
        survivor, loser = pick_survivor(early, late)
        self.assertEqual(survivor.agreement_uuid, "late")
        self.assertEqual(loser.agreement_uuid, "early")

    def test_ingested_date_breaks_filing_date_tie(self) -> None:
        first = self._candidate("first", date(2021, 4, 1), ingested=datetime(2026, 1, 1, 9, 0))
        second = self._candidate("second", date(2021, 4, 1), ingested=datetime(2026, 1, 1, 10, 0))
        survivor, _ = pick_survivor(first, second)
        self.assertEqual(survivor.agreement_uuid, "second")

    def test_drastically_less_content_never_survives(self) -> None:
        complete = self._candidate("complete", date(2021, 4, 1), pages=120)
        truncated = self._candidate("truncated", date(2021, 7, 9), pages=12)
        survivor, loser = pick_survivor(complete, truncated)
        self.assertEqual(survivor.agreement_uuid, "complete")
        self.assertEqual(loser.agreement_uuid, "truncated")

    def test_comparable_content_keeps_most_recent(self) -> None:
        complete = self._candidate("older", date(2021, 4, 1), pages=120)
        newer = self._candidate("newer", date(2021, 7, 9), pages=80)
        survivor, _ = pick_survivor(complete, newer)
        self.assertEqual(survivor.agreement_uuid, "newer")


if __name__ == "__main__":
    _ = unittest.main()
