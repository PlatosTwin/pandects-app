# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportAny=false
import unittest
from typing import cast
from unittest.mock import patch

import requests

from etl.utils.backfill_auto_verify_edgar import (
    AgreementCandidate,
    ScanStats,
    apply_status_updates,
    collect_matching_agreement_uuids,
)


class _FakeResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


class _FakeConnection:
    def __init__(self) -> None:
        self.executions: list[tuple[str, dict[str, object]]] = []

    def execute(self, statement: object, params: dict[str, object]) -> _FakeResult:
        self.executions.append((str(statement), params))
        uuids = cast(list[str], params["agreement_uuids"])
        return _FakeResult(len(uuids))


class BackfillAutoVerifyEdgarTests(unittest.TestCase):
    def test_collect_matching_agreement_uuids_is_case_insensitive(self) -> None:
        candidates = [
            AgreementCandidate("agreement-1", "https://example.com/1.htm"),
            AgreementCandidate("agreement-2", "https://example.com/2.htm"),
        ]

        with patch(
            "etl.utils.backfill_auto_verify_edgar.agreement_matches_auto_verify_rule",
            side_effect=[True, False],
        ):
            outcome = collect_matching_agreement_uuids(
                candidates,
                max_workers=2,
                rate_limit_max_requests=10,
                rate_limit_window_seconds=1.025,
            )

        self.assertEqual(outcome.matched_uuids, ["agreement-1"])
        self.assertEqual(outcome.checked, 2)
        self.assertEqual(outcome.failed, 0)

    def test_collect_matching_agreement_uuids_skips_fetch_failures_without_matching(self) -> None:
        candidates = [AgreementCandidate("agreement-1", "https://example.com/1.htm")]

        with patch(
            "etl.utils.backfill_auto_verify_edgar.agreement_matches_auto_verify_rule",
            side_effect=requests.exceptions.Timeout("boom"),
        ):
            outcome = collect_matching_agreement_uuids(
                candidates,
                max_workers=1,
                rate_limit_max_requests=10,
                rate_limit_window_seconds=1.025,
                stats=ScanStats(),
            )

        self.assertEqual(outcome.matched_uuids, [])
        self.assertEqual(outcome.checked, 1)
        self.assertEqual(outcome.failed, 1)

    def test_apply_status_updates_batches_and_updates_only_input_ids(self) -> None:
        conn = _FakeConnection()
        uuids = [f"agreement-{index}" for index in range(205)]

        updated = apply_status_updates(cast(object, conn), "pdx", uuids)  # type: ignore[arg-type]

        self.assertEqual(updated, 205)
        self.assertEqual(len(conn.executions), 2)
        self.assertIn("SET status = 'verified'", conn.executions[0][0])
        self.assertEqual(len(cast(list[str], conn.executions[0][1]["agreement_uuids"])), 200)
        self.assertEqual(len(cast(list[str], conn.executions[1][1]["agreement_uuids"])), 5)


if __name__ == "__main__":
    _ = unittest.main()
