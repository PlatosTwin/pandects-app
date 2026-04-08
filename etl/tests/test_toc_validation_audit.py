# pyright: reportAny=false, reportPrivateUsage=false
import unittest

from etl.utils.toc_validation_audit import (
    _categorize_section_non_sequential_miss,
    _consistent_non_sequential_articles,
    _toc_conflict_articles,
)


class TocValidationAuditTests(unittest.TestCase):
    def test_categorize_miss_as_missing_toc(self) -> None:
        self.assertEqual(
            _categorize_section_non_sequential_miss(
                persisted_reason_codes=["section_non_sequential"],
                recomputed_reason_codes=["section_non_sequential"],
                toc_sequences={},
                body_sequences={4: [1, 2, 4]},
                toc_conflict_articles=[],
            ),
            "missing_toc",
        )

    def test_categorize_miss_as_other_article_still_invalid(self) -> None:
        self.assertEqual(
            _categorize_section_non_sequential_miss(
                persisted_reason_codes=["section_non_sequential"],
                recomputed_reason_codes=["section_non_sequential"],
                toc_sequences={8: [1, 2, 3, 5], 13: [1, 2, 4]},
                body_sequences={8: [1, 2, 3, 5], 13: [1, 2, 3]},
                toc_conflict_articles=[13],
            ),
            "other_article_still_invalid",
        )

    def test_categorize_miss_as_toc_conflict(self) -> None:
        self.assertEqual(
            _categorize_section_non_sequential_miss(
                persisted_reason_codes=["section_non_sequential"],
                recomputed_reason_codes=["section_non_sequential"],
                toc_sequences={4: [1, 2, 3, 4]},
                body_sequences={4: [1, 2, 4]},
                toc_conflict_articles=[4],
            ),
            "toc_conflict",
        )

    def test_categorize_miss_as_structural_invalid_mixed_when_no_conflict_articles(self) -> None:
        self.assertEqual(
            _categorize_section_non_sequential_miss(
                persisted_reason_codes=["section_non_sequential"],
                recomputed_reason_codes=[
                    "body_starts_non_article",
                    "section_non_sequential",
                    "section_title_invalid_numbering",
                ],
                toc_sequences={},
                body_sequences={},
                toc_conflict_articles=[],
            ),
            "missing_toc",
        )
        self.assertEqual(
            _categorize_section_non_sequential_miss(
                persisted_reason_codes=["section_non_sequential"],
                recomputed_reason_codes=[
                    "body_starts_non_article",
                    "section_non_sequential",
                    "section_title_invalid_numbering",
                ],
                toc_sequences={4: [1, 2, 3]},
                body_sequences={4: [1, 2, 3]},
                toc_conflict_articles=[],
            ),
            "structural_invalid_mixed",
        )

    def test_consistent_non_sequential_articles_tracks_only_matching_gaps(self) -> None:
        self.assertEqual(
            _consistent_non_sequential_articles(
                toc_sequences={4: [1, 2, 4], 5: [1, 2, 3], 6: [1, 3]},
                body_sequences={4: [1, 2, 4], 5: [1, 2, 3], 6: [1, 2]},
            ),
            [4],
        )

    def test_toc_conflict_articles_ignores_matching_and_fully_sequential_articles(self) -> None:
        self.assertEqual(
            _toc_conflict_articles(
                toc_sequences={4: [1, 2, 3, 4], 5: [1, 2, 4], 6: [1, 2, 3]},
                body_sequences={4: [1, 2, 4], 5: [1, 2, 4], 6: [1, 2, 3]},
            ),
            [4],
        )


if __name__ == "__main__":
    _ = unittest.main()
