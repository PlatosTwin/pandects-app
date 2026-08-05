# pyright: reportPrivateUsage=false
"""Tests for the corpus-wide duplicate reconciliation planner in the staging asset."""
import itertools
import string
import unittest
from datetime import date, datetime

from etl.defs.a_staging_asset import (
    _CorpusAgreement,
    _CorpusItem,
    _plan_corpus_dedupe,
)
from etl.domain.dedupe_signatures import (
    CoverIdentity,
    compute_document_signature,
)
from etl.utils.agreement_signature_store import StoredSignature

_WORDS = ["".join(letters) for letters in itertools.product(string.ascii_lowercase, repeat=3)]


def _text(start: int, count: int) -> str:
    return " ".join(_WORDS[start : start + count])


def _item(
    uuid: str,
    text: str,
    *,
    filing: date,
    dated_as_of: date | None = None,
    parties: frozenset[str] = frozenset(),
    amends: bool = False,
    ar_refs: frozenset[date] = frozenset(),
    target: str | None = None,
    acquirer: str | None = None,
    page_count: int = 100,
    ingested: datetime | None = None,
    url: str | None = None,
) -> _CorpusItem:
    resolved_url = url or f"https://example.com/{uuid}-dir/{uuid}.htm"
    return _CorpusItem(
        agreement=_CorpusAgreement(
            agreement_uuid=uuid,
            url=resolved_url,
            filing_date=filing,
            ingested_date=ingested,
            target=target,
            acquirer=acquirer,
        ),
        stored=StoredSignature(
            agreement_uuid=uuid,
            url=resolved_url,
            signature=compute_document_signature(text, page_count=page_count),
            cover=CoverIdentity(
                dated_as_of=dated_as_of,
                party_tokens=parties,
                amends_and_restates=amends,
                ar_reference_dates=ar_refs,
            ),
        ),
    )


class PlanCorpusDedupeTests(unittest.TestCase):
    def test_cross_day_duplicate_far_beyond_thirty_days_is_caught(self) -> None:
        # Re-filed exhibit copies months later were structurally invisible to
        # the old +/-30-day window; the corpus-wide plan must catch them.
        text = _text(0, 3000)
        dated = date(2021, 6, 21)
        old = _item("old-uuid", text, filing=date(2021, 6, 22), dated_as_of=dated)
        refiled = _item(
            "new-uuid", text + " " + _text(10000, 20),
            filing=date(2022, 3, 1), dated_as_of=dated,
        )
        resolutions, reviews = _plan_corpus_dedupe({"new-uuid"}, [old, refiled])
        self.assertEqual(reviews, [])
        self.assertEqual(len(resolutions), 1)
        # Survivor rule: keep the most recent copy.
        self.assertEqual(resolutions[0].survivor_uuid, "new-uuid")
        self.assertEqual(resolutions[0].loser_uuid, "old-uuid")

    def test_party_match_alone_never_deletes(self) -> None:
        # Same target/acquirer pair, genuinely different agreement content
        # (terminated-then-re-signed deal): the party trigger must force a
        # comparison but never delete.
        old = _item(
            "old-uuid", _text(0, 3000),
            filing=date(2021, 3, 21), dated_as_of=date(2021, 3, 21),
            target="Kansas City Southern", acquirer="Canadian Pacific Railway Limited",
        )
        resigned = _item(
            "new-uuid", _text(5000, 3000),
            filing=date(2021, 9, 15), dated_as_of=date(2021, 9, 15),
            target="Kansas City Southern", acquirer="Canadian Pacific Railway Ltd",
        )
        resolutions, reviews = _plan_corpus_dedupe({"new-uuid"}, [old, resigned])
        self.assertEqual(resolutions, [])
        self.assertEqual(reviews, [])

    def test_content_hit_without_cover_identity_is_flagged_for_review(self) -> None:
        text = _text(0, 3000)
        old = _item("old-uuid", text, filing=date(2021, 6, 22))
        near = _item("new-uuid", text + " " + _text(10000, 20), filing=date(2021, 8, 1))
        resolutions, reviews = _plan_corpus_dedupe({"new-uuid"}, [old, near])
        self.assertEqual(resolutions, [])
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0].new_agreement_uuid, "new-uuid")
        self.assertEqual(reviews[0].existing_agreement_uuid, "old-uuid")
        self.assertEqual(reviews[0].reason, "content_match_without_cover_identity")

    def test_confirmed_amends_and_restates_dedupes_keeping_most_recent(self) -> None:
        text = _text(0, 3000)
        parties = frozenset({"TV AMMO", "BREEZE HOLDINGS ACQUISITION"})
        original = _item(
            "old-uuid", text,
            filing=date(2022, 11, 1), dated_as_of=date(2022, 10, 31), parties=parties,
        )
        restated = _item(
            "new-uuid", text + " " + _text(10000, 20),
            filing=date(2024, 2, 21), dated_as_of=date(2024, 2, 14),
            amends=True, ar_refs=frozenset({date(2022, 10, 31)}), parties=parties,
        )
        resolutions, reviews = _plan_corpus_dedupe({"new-uuid"}, [original, restated])
        self.assertEqual(reviews, [])
        self.assertEqual(len(resolutions), 1)
        self.assertEqual(resolutions[0].survivor_uuid, "new-uuid")

    def test_ar_flag_without_recital_reference_is_review_not_delete(self) -> None:
        # Audit regression: SPAC boilerplate made unrelated deals look like
        # A&R supersessions; without the recital date citation, only review.
        text = _text(0, 3000)
        original = _item(
            "old-uuid", text,
            filing=date(2021, 3, 31), dated_as_of=date(2021, 3, 31),
            parties=frozenset({"VG ACQUISITION"}),
        )
        lookalike = _item(
            "new-uuid", text + " " + _text(10000, 20),
            filing=date(2021, 7, 9), dated_as_of=date(2021, 7, 9), amends=True,
            parties=frozenset({"DUDDELL STREET ACQUISITION"}),
        )
        resolutions, reviews = _plan_corpus_dedupe({"new-uuid"}, [original, lookalike])
        self.assertEqual(resolutions, [])
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0].reason, "dated_as_of_mismatch")

    def test_same_accession_sibling_exhibits_never_dedupe(self) -> None:
        # ex2-1 and ex2-2 of one filing are different agreements even when the
        # boilerplate is near-identical and the cover dates match.
        text = _text(0, 3000)
        dated = date(2021, 4, 7)
        accession = "https://www.sec.gov/Archives/edgar/data/1799983/000121390021020720"
        first = _item(
            "old-uuid", text, filing=date(2021, 4, 8), dated_as_of=dated,
            url=f"{accession}/ea139171ex2-1_riceacq.htm",
        )
        second = _item(
            "new-uuid", text + " " + _text(10000, 20),
            filing=date(2021, 4, 8), dated_as_of=dated,
            url=f"{accession}/ea139171ex2-2_riceacq.htm",
        )
        resolutions, reviews = _plan_corpus_dedupe({"new-uuid"}, [first, second])
        self.assertEqual(resolutions, [])
        self.assertEqual(reviews, [])

    def test_party_metadata_disagreement_forces_review(self) -> None:
        # Same-day sibling template deals: strong content + matching date but
        # disjoint curated targets must reach review, never auto-delete.
        text = _text(0, 3000)
        dated = date(2021, 4, 7)
        first = _item(
            "old-uuid", text, filing=date(2021, 4, 8), dated_as_of=dated,
            target="23andMe, Inc.", acquirer="VG Acquisition Corp.",
        )
        second = _item(
            "new-uuid", text + " " + _text(10000, 20),
            filing=date(2021, 4, 9), dated_as_of=dated,
            target="FiscalNote, Inc.", acquirer="Duddell Street Acquisition Corp.",
        )
        resolutions, reviews = _plan_corpus_dedupe({"new-uuid"}, [first, second])
        self.assertEqual(resolutions, [])
        self.assertEqual(len(reviews), 1)
        self.assertEqual(reviews[0].reason, "party_metadata_conflict")

    def test_drastically_less_parsed_content_never_survives(self) -> None:
        text = _text(0, 3000)
        dated = date(2021, 6, 21)
        complete = _item(
            "old-uuid", text, filing=date(2021, 6, 22), dated_as_of=dated, page_count=120
        )
        truncated = _item(
            "new-uuid", text + " " + _text(10000, 20),
            filing=date(2022, 1, 5), dated_as_of=dated, page_count=15,
        )
        resolutions, _ = _plan_corpus_dedupe({"new-uuid"}, [complete, truncated])
        self.assertEqual(len(resolutions), 1)
        self.assertEqual(resolutions[0].survivor_uuid, "old-uuid")
        self.assertEqual(resolutions[0].loser_uuid, "new-uuid")

    def test_unrelated_agreements_produce_no_actions(self) -> None:
        left = _item("old-uuid", _text(0, 2000), filing=date(2021, 6, 22))
        right = _item("new-uuid", _text(4000, 2000), filing=date(2021, 8, 1))
        resolutions, reviews = _plan_corpus_dedupe({"new-uuid"}, [left, right])
        self.assertEqual(resolutions, [])
        self.assertEqual(reviews, [])


if __name__ == "__main__":
    _ = unittest.main()
