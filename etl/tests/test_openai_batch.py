# pyright: reportAny=false, reportUnknownMemberType=false, reportUnknownArgumentType=false
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from dagster import AssetExecutionContext
from openai import OpenAI

from etl.utils.openai_batch import poll_batch_until_terminal


class _FakeLog:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def info(self, msg: str) -> None:
        self.messages.append(msg)


class _FakeContext:
    def __init__(self) -> None:
        self.log = _FakeLog()


class _FakeBatches:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self._responses = responses
        self._idx = 0

    def retrieve(self, _batch_id: str) -> SimpleNamespace:
        response = self._responses[self._idx]
        self._idx += 1
        return response


class _FakeClient:
    def __init__(self, responses: list[SimpleNamespace]) -> None:
        self.batches = _FakeBatches(responses)


class OpenAIBatchUtilsTests(unittest.TestCase):
    def test_poll_batch_logs_progress_counts_when_available(self) -> None:
        context = _FakeContext()
        client = _FakeClient(
            [
                SimpleNamespace(
                    id="batch-1",
                    status="in_progress",
                    request_counts=SimpleNamespace(total=10, completed=2, failed=1),
                ),
                SimpleNamespace(
                    id="batch-1",
                    status="completed",
                    request_counts=SimpleNamespace(total=10, completed=8, failed=2),
                ),
            ]
        )

        with patch("etl.utils.openai_batch.time.sleep", return_value=None):
            final = poll_batch_until_terminal(
                context=cast(
                    AssetExecutionContext,
                    cast(object, SimpleNamespace(log=context.log)),
                ),
                client=cast(OpenAI, cast(object, SimpleNamespace(batches=client.batches))),
                batch_id="batch-1",
                log_prefix="xml_verify_asset",
            )

        self.assertEqual(final.status, "completed")
        self.assertTrue(
            any(
                "status=in_progress; progress=3/10 (30%), completed=2, failed=1; sleeping 5s"
                in msg
                for msg in context.log.messages
            )
        )
        self.assertTrue(
            any(
                "terminal status=completed; progress=10/10 (100%), completed=8, failed=2"
                in msg
                for msg in context.log.messages
            )
        )

    def test_poll_batch_logs_terminal_without_request_counts(self) -> None:
        context = _FakeContext()
        client = _FakeClient(
            [
                SimpleNamespace(id="batch-2", status="in_progress"),
                SimpleNamespace(id="batch-2", status="failed"),
            ]
        )

        with patch("etl.utils.openai_batch.time.sleep", return_value=None):
            final = poll_batch_until_terminal(
                context=cast(
                    AssetExecutionContext,
                    cast(object, SimpleNamespace(log=context.log)),
                ),
                client=cast(OpenAI, cast(object, SimpleNamespace(batches=client.batches))),
                batch_id="batch-2",
                log_prefix="tx_metadata_asset (offline)",
            )

        self.assertEqual(final.status, "failed")
        self.assertTrue(
            any("status=in_progress; sleeping 5s" in msg for msg in context.log.messages)
        )
        self.assertTrue(
            any("terminal status=failed" in msg for msg in context.log.messages)
        )


if __name__ == "__main__":
    _ = unittest.main()
