# pyright: reportAny=false, reportPrivateUsage=false
import unittest
from collections.abc import Mapping, Sequence
from typing import cast
from sqlalchemy.engine import Connection

from etl.defs.d_ai_repair_asset import (
    _fetch_candidates,
    _repair_model_for_attempted,
    _repair_model_for_candidate,
    _apply_full_page_tag_spans,
    _validate_full_page_tag_spans,
)


class _FakeResult:
    def __init__(self, rows: list[dict[str, object]] | None = None) -> None:
        self._rows = rows or []

    def mappings(self) -> "_FakeResult":
        return self

    def fetchall(self) -> list[dict[str, object]]:
        return self._rows

    class _Scalars:
        def __init__(self, rows: list[dict[str, object]]) -> None:
            self._rows = rows

        def all(self) -> list[object]:
            return [next(iter(row.values())) for row in self._rows]

    def scalars(self) -> "_FakeResult._Scalars":
        return _FakeResult._Scalars(self._rows)


class _FakeConn:
    def __init__(
        self,
        invalid_rows: Sequence[Mapping[str, object]],
        existing_request_rows: Sequence[Mapping[str, object]],
        completed_request_rows: Sequence[Mapping[str, object]],
        page_rows: Sequence[Mapping[str, object]],
    ) -> None:
        self._invalid_rows = [dict(row) for row in invalid_rows]
        self._existing_request_rows = [dict(row) for row in existing_request_rows]
        self._completed_request_rows = [dict(row) for row in completed_request_rows]
        self._page_rows = [dict(row) for row in page_rows]

    def execute(self, statement: object, _params: dict[str, object] | None = None) -> _FakeResult:
        sql = str(statement)
        if "JOIN pdx.xml_status_reasons r" in sql:
            return _FakeResult(self._invalid_rows)
        if "FROM pdx.ai_repair_requests" in sql and "status = 'completed'" in sql:
            return _FakeResult(self._completed_request_rows)
        if "FROM pdx.ai_repair_requests" in sql and "request_id IN" in sql:
            return _FakeResult(self._existing_request_rows)
        if "FROM pdx.pages p" in sql and "p.page_uuid IN" in sql:
            return _FakeResult(self._page_rows)
        raise AssertionError(f"Unexpected SQL in test: {sql}")


class AiRepairTargetingTests(unittest.TestCase):
    def test_repair_model_routing_by_attempt_state(self) -> None:
        self.assertEqual(_repair_model_for_attempted(0), "gpt-5.4-mini")
        self.assertEqual(_repair_model_for_attempted(1), "gpt-5.4-mini")
        with self.assertRaises(ValueError):
            _repair_model_for_attempted(2)
        self.assertEqual(
            _repair_model_for_candidate(1, has_completed_requests=False),
            "gpt-5.4-mini",
        )
        self.assertEqual(
            _repair_model_for_candidate(1, has_completed_requests=True),
            "gpt-5.4",
        )

    def test_fetch_candidates_uses_reason_page_targets_and_agreement_ranking(self) -> None:
        invalid_rows = [
            {"agreement_uuid": "agreement-a", "xml_version": 5, "ai_repair_attempted": 0, "reason_code": "section_non_sequential", "page_uuid": "page-a1"},
            {"agreement_uuid": "agreement-a", "xml_version": 5, "ai_repair_attempted": 0, "reason_code": "section_article_mismatch", "page_uuid": "page-a2"},
            {"agreement_uuid": "agreement-b", "xml_version": 3, "ai_repair_attempted": 0, "reason_code": "first_article_not_one", "page_uuid": "page-b1"},
            {"agreement_uuid": "agreement-c", "xml_version": 8, "ai_repair_attempted": 0, "reason_code": "section_non_sequential", "page_uuid": "page-c1"},
            {"agreement_uuid": "agreement-c", "xml_version": 8, "ai_repair_attempted": 0, "reason_code": "section_non_sequential", "page_uuid": "page-c2"},
            {"agreement_uuid": "agreement-c", "xml_version": 8, "ai_repair_attempted": 0, "reason_code": "section_non_sequential", "page_uuid": "page-c3"},
        ]
        page_rows = [
            {"page_uuid": "page-a1", "agreement_uuid": "agreement-a", "page_order": 1, "text": "a1"},
            {"page_uuid": "page-a2", "agreement_uuid": "agreement-a", "page_order": 2, "text": "a2"},
            {"page_uuid": "page-b1", "agreement_uuid": "agreement-b", "page_order": 1, "text": "b1"},
            {"page_uuid": "page-c1", "agreement_uuid": "agreement-c", "page_order": 1, "text": "c1"},
            {"page_uuid": "page-c2", "agreement_uuid": "agreement-c", "page_order": 2, "text": "c2"},
            {"page_uuid": "page-c3", "agreement_uuid": "agreement-c", "page_order": 3, "text": "c3"},
        ]
        conn = _FakeConn(
            invalid_rows,
            existing_request_rows=[],
            completed_request_rows=[],
            page_rows=page_rows,
        )

        candidates = _fetch_candidates(cast(Connection, cast(object, conn)), "pdx", agreement_limit=2)

        self.assertEqual(
            {str(row["agreement_uuid"]) for row in candidates},
            {"agreement-a", "agreement-b"},
        )
        request_ids = {
            f"{row['page_uuid']}::full::{row['xml_version']}" for row in candidates
        }
        self.assertEqual(request_ids, {"page-a1::full::5", "page-a2::full::5", "page-b1::full::3"})

    def test_fetch_candidates_skips_previously_processed_requests(self) -> None:
        invalid_rows = [
            {"agreement_uuid": "agreement-z", "xml_version": 4, "ai_repair_attempted": 0, "reason_code": "section_non_sequential", "page_uuid": "page-z1"},
            {"agreement_uuid": "agreement-z", "xml_version": 4, "ai_repair_attempted": 0, "reason_code": "section_non_sequential", "page_uuid": "page-z2"},
        ]
        existing_request_rows = [{"request_id": "page-z1::full::4"}]
        page_rows = [
            {"page_uuid": "page-z1", "agreement_uuid": "agreement-z", "page_order": 1, "text": "z1"},
            {"page_uuid": "page-z2", "agreement_uuid": "agreement-z", "page_order": 2, "text": "z2"},
        ]
        conn = _FakeConn(
            invalid_rows,
            existing_request_rows=existing_request_rows,
            completed_request_rows=[],
            page_rows=page_rows,
        )

        candidates = _fetch_candidates(cast(Connection, cast(object, conn)), "pdx", agreement_limit=10)

        self.assertEqual(len(candidates), 1)
        self.assertEqual(str(candidates[0]["page_uuid"]), "page-z2")
        self.assertEqual(int(candidates[0]["xml_version"]), 4)

    def test_fetch_candidates_prioritizes_unattempted_over_attempted(self) -> None:
        invalid_rows = [
            {"agreement_uuid": "agreement-new", "xml_version": 10, "ai_repair_attempted": 0, "reason_code": "section_non_sequential", "page_uuid": "page-new-1"},
            {"agreement_uuid": "agreement-new", "xml_version": 10, "ai_repair_attempted": 0, "reason_code": "section_non_sequential", "page_uuid": "page-new-2"},
            {"agreement_uuid": "agreement-retry", "xml_version": 11, "ai_repair_attempted": 1, "reason_code": "section_non_sequential", "page_uuid": "page-retry-1"},
        ]
        page_rows = [
            {"page_uuid": "page-new-1", "agreement_uuid": "agreement-new", "page_order": 1, "text": "n1"},
            {"page_uuid": "page-new-2", "agreement_uuid": "agreement-new", "page_order": 2, "text": "n2"},
            {"page_uuid": "page-retry-1", "agreement_uuid": "agreement-retry", "page_order": 1, "text": "r1"},
        ]
        conn = _FakeConn(
            invalid_rows,
            existing_request_rows=[],
            completed_request_rows=[],
            page_rows=page_rows,
        )

        candidates = _fetch_candidates(cast(Connection, cast(object, conn)), "pdx", agreement_limit=1)

        self.assertEqual({str(row["agreement_uuid"]) for row in candidates}, {"agreement-new"})

    def test_fetch_candidates_sets_completed_history_flag(self) -> None:
        invalid_rows = [
            {"agreement_uuid": "agreement-retry", "xml_version": 11, "ai_repair_attempted": 1, "reason_code": "section_non_sequential", "page_uuid": "page-retry-1"},
            {"agreement_uuid": "agreement-retry", "xml_version": 11, "ai_repair_attempted": 1, "reason_code": "section_non_sequential", "page_uuid": "page-retry-2"},
        ]
        completed_request_rows = [{"request_id": "page-retry-1::full::11"}]
        page_rows = [
            {"page_uuid": "page-retry-1", "agreement_uuid": "agreement-retry", "page_order": 1, "text": "r1"},
            {"page_uuid": "page-retry-2", "agreement_uuid": "agreement-retry", "page_order": 2, "text": "r2"},
        ]
        conn = _FakeConn(
            invalid_rows,
            existing_request_rows=[],
            completed_request_rows=completed_request_rows,
            page_rows=page_rows,
        )

        candidates = _fetch_candidates(cast(Connection, cast(object, conn)), "pdx", agreement_limit=10)

        self.assertEqual(len(candidates), 2)
        self.assertTrue(all(int(c["has_completed_requests"]) == 1 for c in candidates))

    def test_full_page_span_validation_accepts_matching_non_overlapping_spans(self) -> None:
        source = "Section 1.01 text."
        spans = [
            {
                "start_char": 0,
                "end_char": 12,
                "label": "section",
                "selected_text": "Section 1.01",
            }
        ]
        validated = _validate_full_page_tag_spans(source, spans)
        self.assertEqual(validated, spans)

    def test_full_page_span_validation_rejects_mismatched_selected_text(self) -> None:
        source = "Section 1.01 text."
        contaminated = [
            {
                "start_char": 0,
                "end_char": 12,
                "label": "section",
                "selected_text": "PAGE_UUID=12",
            }
        ]
        with self.assertRaises(ValueError):
            _validate_full_page_tag_spans(source, contaminated)

    def test_full_page_span_validation_realigns_nearby_exact_match(self) -> None:
        source = "\nSection 6.03. Termination."
        shifted = [
            {
                "start_char": 0,
                "end_char": 26,
                "label": "section",
                "selected_text": "Section 6.03. Termination.",
            }
        ]

        validated = _validate_full_page_tag_spans(source, shifted)

        self.assertEqual(
            validated,
            [
                {
                    "start_char": 1,
                    "end_char": 27,
                    "label": "section",
                    "selected_text": "Section 6.03. Termination.",
                }
            ],
        )

    def test_full_page_span_validation_realigns_whitespace_normalized_match(self) -> None:
        source = "ARTICLE III \n\nREPRESENTATIONS AND WARRANTIES OF THE SELLER \n\nAND IOS CAPITAL"
        normalized = [
            {
                "start_char": 0,
                "end_char": len(source) - 2,
                "label": "article",
                "selected_text": "ARTICLE III\n\nREPRESENTATIONS AND WARRANTIES OF THE SELLER\n\nAND IOS CAPITAL",
            }
        ]

        validated = _validate_full_page_tag_spans(source, normalized)

        self.assertEqual(
            validated,
            [
                {
                    "start_char": 0,
                    "end_char": len(source),
                    "label": "article",
                    "selected_text": source,
                }
            ],
        )

    def test_full_page_span_validation_realigns_out_of_bounds_page_number(self) -> None:
        source = "Body text\n- 32 -"
        shifted = [
            {
                "start_char": len(source) + 2,
                "end_char": len(source) + 8,
                "label": "page",
                "selected_text": "- 32 -",
            }
        ]

        validated = _validate_full_page_tag_spans(source, shifted)

        self.assertEqual(
            validated,
            [
                {
                    "start_char": len("Body text\n"),
                    "end_char": len(source),
                    "label": "page",
                    "selected_text": "- 32 -",
                }
            ],
        )

    def test_full_page_span_validation_rejects_far_away_exact_match(self) -> None:
        source = "Intro text. " + ("x" * 200) + "Section 9.12 Financing Sources."
        far_off = [
            {
                "start_char": 10,
                "end_char": 33,
                "label": "section",
                "selected_text": "Section 9.12 Financing Sources.",
            }
        ]

        with self.assertRaises(ValueError):
            _validate_full_page_tag_spans(source, far_off)

    def test_full_page_span_validation_rejects_ambiguous_nearby_exact_matches(self) -> None:
        source = "TAG" + ("x" * 17) + "TAG"
        ambiguous = [
            {
                "start_char": 10,
                "end_char": 13,
                "label": "section",
                "selected_text": "TAG",
            }
        ]

        with self.assertRaises(ValueError):
            _validate_full_page_tag_spans(source, ambiguous)

    def test_full_page_span_validation_rejects_overlapping_spans(self) -> None:
        source = "ARTICLE I\nSection 1.01"
        spans = [
            {
                "start_char": 0,
                "end_char": 9,
                "label": "article",
                "selected_text": "ARTICLE I",
            },
            {
                "start_char": 8,
                "end_char": 22,
                "label": "section",
                "selected_text": "I\nSection 1.01",
            },
        ]
        with self.assertRaises(ValueError):
            _validate_full_page_tag_spans(source, spans)

    def test_apply_full_page_tag_spans_inserts_tags_deterministically(self) -> None:
        source = "ARTICLE I\nSection 1.01 text.\n-56-"
        spans = [
            {
                "start_char": 0,
                "end_char": 9,
                "label": "article",
                "selected_text": "ARTICLE I",
            },
            {
                "start_char": 10,
                "end_char": 22,
                "label": "section",
                "selected_text": "Section 1.01",
            },
            {
                "start_char": 29,
                "end_char": 33,
                "label": "page",
                "selected_text": "-56-",
            },
        ]
        tagged = _apply_full_page_tag_spans(source, spans)
        self.assertEqual(
            tagged,
            "<article>ARTICLE I</article>\n<section>Section 1.01</section> text.\n<page>-56-</page>",
        )


if __name__ == "__main__":
    unittest.main()
