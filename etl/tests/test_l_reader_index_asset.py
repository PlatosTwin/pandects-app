# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
import gzip
import json
import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any, Iterator, cast
from unittest.mock import patch

from dagster import AssetExecutionContext, build_op_context

from etl.defs import l_reader_index_asset
from etl.defs.l_reader_index_asset import _run_reader_index, reader_index_queue_sql
from etl.defs.resources import DBResource, PipelineConfig, QueueRunMode
from etl.domain.reader_index import ENRICHER_VERSION, reader_index_row

XML = (
    '<document uuid="a-1"><body><article uuid="art1" title="ARTICLE I GENERAL">'
    '<section uuid="s1" title="Section 1.1 Defs"><text>“Buyer” means Acme. See Section 1.2.</text></section>'
    '<section uuid="s2" title="Section 1.2 Other"><text>The Buyer agrees.</text></section>'
    "</article></body></document>"
)


class _Result:
    def __init__(self, rows: list[Any]) -> None:
        self._rows = rows

    def scalars(self) -> "_Result":
        return self

    def mappings(self) -> "_Result":
        return self

    def all(self) -> list[Any]:
        return self._rows

    def fetchall(self) -> list[Any]:
        return self._rows


class _FakeConn:
    def __init__(self, engine: "_FakeEngine") -> None:
        self.engine = engine

    def execute(self, stmt: Any, params: Any = None) -> _Result:
        sql = str(stmt)
        if "information_schema.tables" in sql:
            return _Result(list(self.engine.tables))
        if "LEFT JOIN" in sql:
            self.engine.queue_params.append(params)
            pending = [u for u in self.engine.queue if u > params["last_uuid"] and u not in self.engine.written]
            return _Result(pending[: params["lim"]])
        if sql.lstrip().startswith("SELECT x.agreement_uuid, x.version"):
            return _Result([{"agreement_uuid": u, "xml_version": 4, "xml": self.engine.xml[u]} for u in params["auuids"]])
        if "INSERT INTO" in sql:
            for row in params:
                self.engine.written[row["agreement_uuid"]] = row
            return _Result([])
        raise AssertionError(f"unexpected SQL: {sql}")


class _FakeEngine:
    def __init__(self, xml: dict[str, str], tables: tuple[str, ...] = ("xml", "reader_index")) -> None:
        self.xml = xml
        self.queue = sorted(xml)
        self.tables = tables
        self.written: dict[str, dict[str, Any]] = {}
        self.queue_params: list[dict[str, Any]] = []

    @contextmanager
    def begin(self) -> Iterator[_FakeConn]:
        yield _FakeConn(self)


def _run(engine: _FakeEngine, batch_size: int = 1) -> list[str]:
    db = SimpleNamespace(get_engine=lambda: engine, database="pdx")
    cfg = SimpleNamespace(xml_agreement_batch_size=batch_size, queue_run_mode=QueueRunMode.DRAIN)
    with build_op_context() as context:
        return _run_reader_index(
            cast(AssetExecutionContext, cast(object, context)),
            cast(DBResource, cast(object, db)),
            cast(PipelineConfig, cast(object, cfg)),
            log_prefix="test",
        )


class ReaderIndexRowTests(unittest.TestCase):
    def test_row_counts_and_gzip_payload(self) -> None:
        row = reader_index_row(XML, agreement_uuid="a-1", xml_version=4)
        self.assertEqual(row["enricher_version"], ENRICHER_VERSION)
        self.assertEqual(len(row["text_hash"]), 64)
        payload = json.loads(gzip.decompress(row["payload_gz"]))
        self.assertEqual(payload["xml_version"], 4)
        self.assertEqual(row["n_terms"], 1)
        self.assertEqual(row["n_term_uses"], 1)
        self.assertEqual((row["n_xrefs"], row["n_xrefs_resolved"]), (1, 1))
        # gzip with mtime=0 keeps rows byte-identical across runs
        self.assertEqual(row["payload_gz"], reader_index_row(XML, agreement_uuid="a-1", xml_version=4)["payload_gz"])


class ReaderIndexAssetTests(unittest.TestCase):
    def test_queue_targets_latest_verified_xml_missing_current_enricher_row(self) -> None:
        sql = reader_index_queue_sql("pdx")
        for fragment in ("x.latest = 1", "x.status = 'verified'", "r.enricher_version = :enricher_version", "r.agreement_uuid IS NULL"):
            self.assertIn(fragment, sql)

    def test_pages_through_queue_and_skips_unparseable_xml(self) -> None:
        engine = _FakeEngine({"a-1": XML, "a-2": "<document><body>", "a-3": XML.replace("a-1", "a-3")})
        processed = _run(engine)
        self.assertEqual(processed, ["a-1", "a-3"])
        self.assertEqual(sorted(engine.written), ["a-1", "a-3"])
        self.assertTrue(all(p["enricher_version"] == ENRICHER_VERSION for p in engine.queue_params))

    def test_an_enricher_error_skips_only_that_agreement(self) -> None:
        engine = _FakeEngine({"a-1": XML, "a-2": XML.replace("a-1", "a-2"), "a-3": XML.replace("a-1", "a-3")})

        def row_or_fail(xml: str, *, agreement_uuid: str, xml_version: int) -> dict[str, Any]:
            if agreement_uuid == "a-2":
                raise IndexError("enricher bug")
            return reader_index_row(xml, agreement_uuid=agreement_uuid, xml_version=xml_version)

        with patch.object(l_reader_index_asset, "reader_index_row", row_or_fail):
            processed = _run(engine, batch_size=3)
        self.assertEqual(processed, ["a-1", "a-3"])

    def test_fails_fast_without_the_table(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "reader_index"):
            _ = _run(_FakeEngine({"a-1": XML}, tables=("xml",)))


if __name__ == "__main__":
    _ = unittest.main()
