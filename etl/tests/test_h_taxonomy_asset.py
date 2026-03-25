# pyright: reportAny=false
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from dagster import AssetExecutionContext

from etl.defs.h_taxonomy_asset import taxonomy_asset
from etl.defs.resources import QueueRunMode, TaxonomyMode


class _FakeResult:
    def __init__(self, *, rowcount: int = 0, rows: list[dict[str, object]] | None = None) -> None:
        self.rowcount = rowcount
        self._rows = rows or []

    def mappings(self) -> "_FakeResult":
        return self

    def fetchall(self) -> list[dict[str, object]]:
        return self._rows


class _FakeBeginContext:
    def __init__(self, conn: object) -> None:
        self._conn = conn

    def __enter__(self) -> object:
        return self._conn

    def __exit__(self, *_exc: object) -> None:
        return None


class _FakeConn:
    def __init__(self, sec_rows: list[dict[str, object]]) -> None:
        self._sec_rows = sec_rows
        self.executed_sql: list[str] = []
        self.update_payloads: list[dict[str, object]] = []

    def execute(self, statement: object, params: object | None = None) -> _FakeResult:
        sql = str(statement)
        self.executed_sql.append(sql)
        if "SELECT DISTINCT" in sql and "FROM pdx.sections" in sql:
            return _FakeResult(rows=[{"agreement_uuid": "agreement-1"}])
        if "section_uuid" in sql and "FROM pdx.sections" in sql:
            return _FakeResult(rows=self._sec_rows)
        if "UPDATE pdx.sections" in sql:
            if isinstance(params, list):
                for payload in cast(list[object], params):
                    if isinstance(payload, dict):
                        self.update_payloads.append(
                            cast(dict[str, object], payload)
                        )
            return _FakeResult(rowcount=1)
        if "SELECT m.agreement_uuid, m.xml, m.version" in sql:
            return _FakeResult(
                rows=[
                    {
                        "agreement_uuid": "agreement-1",
                        "xml": "<document />",
                        "version": 2,
                    }
                ]
            )
        raise AssertionError(f"Unexpected SQL in test: {sql}")


class _FakeEngine:
    def __init__(self, conn: object) -> None:
        self._conn = conn

    def begin(self) -> _FakeBeginContext:
        return _FakeBeginContext(self._conn)


class _FakeLog:
    def info(self, *args: object, **kwargs: object) -> None:
        _ = args
        _ = kwargs


class TaxonomyAssetProjectionRefreshTests(unittest.TestCase):
    def test_taxonomy_asset_refreshes_latest_sections_search(self) -> None:
        conn = _FakeConn(
            sec_rows=[
                {
                    "section_uuid": "section-1",
                    "agreement_uuid": "agreement-1",
                    "article_title": "ARTICLE I",
                    "section_title": "Section 1",
                    "xml_content": "<section>Body</section>",
                    "section_standard_id_gold_label": None,
                }
            ]
        )
        context = SimpleNamespace(log=_FakeLog())
        db = SimpleNamespace(database="pdx", get_engine=lambda: _FakeEngine(conn))
        taxonomy_model = SimpleNamespace(model=lambda: object())
        pipeline_config = SimpleNamespace(
            taxonomy_agreement_batch_size=10,
            taxonomy_mode=TaxonomyMode.INFERENCE,
            queue_run_mode=QueueRunMode.SINGLE_BATCH,
        )

        with (
            patch(
                "etl.defs.h_taxonomy_asset.predict_taxonomy",
                return_value=(
                    [{"section_uuid": "section-1", "agreement_uuid": "agreement-1"}],
                    [{"label": "governing_law", "alt_probs": [0.9, 0.05, 0.05]}],
                ),
            ),
            patch(
                "etl.defs.h_taxonomy_asset.apply_standard_ids_to_xml",
                return_value="<document standardId=\"governing_law\" />",
            ),
            patch("etl.defs.h_taxonomy_asset.upsert_xml") as upsert_xml,
            patch("etl.defs.h_taxonomy_asset.refresh_latest_sections_search") as refresh,
            patch("etl.defs.h_taxonomy_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, taxonomy_asset.op.compute_fn), "decorated_fn")
            _ = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                db=cast(object, db),
                taxonomy_model=cast(object, taxonomy_model),
                pipeline_config=cast(object, pipeline_config),
            )

        upsert_xml.assert_called_once()
        refresh.assert_called_once_with(conn, "pdx", ["agreement-1"])
        self.assertTrue(
            any(
                "x.version = s.xml_version" in sql
                and "x.latest = 1" in sql
                and "x.status = 'verified'" in sql
                for sql in conn.executed_sql
            )
        )
        self.assertTrue(
            any(
                "FROM pdx.xml m" in sql
                and "m.latest = 1" in sql
                and "m.status = 'verified'" in sql
                for sql in conn.executed_sql
            )
        )

    def test_inference_mode_prefers_gold_label_for_xml(self) -> None:
        conn = _FakeConn(
            sec_rows=[
                {
                    "section_uuid": "section-1",
                    "agreement_uuid": "agreement-1",
                    "article_title": "ARTICLE I",
                    "section_title": "Section 1",
                    "xml_content": "<section>Body</section>",
                    "section_standard_id_gold_label": "gold_governing_law",
                }
            ]
        )
        context = SimpleNamespace(log=_FakeLog())
        db = SimpleNamespace(database="pdx", get_engine=lambda: _FakeEngine(conn))
        taxonomy_model = SimpleNamespace(model=lambda: object())
        pipeline_config = SimpleNamespace(
            taxonomy_agreement_batch_size=10,
            taxonomy_mode=TaxonomyMode.INFERENCE,
            queue_run_mode=QueueRunMode.SINGLE_BATCH,
        )

        with (
            patch(
                "etl.defs.h_taxonomy_asset.predict_taxonomy",
                return_value=(
                    [{"section_uuid": "section-1", "agreement_uuid": "agreement-1"}],
                    [{"label": "inferred_governing_law", "alt_probs": [0.8, 0.1, 0.1]}],
                ),
            ),
            patch(
                "etl.defs.h_taxonomy_asset.apply_standard_ids_to_xml",
                return_value="<document standardId=\"gold_governing_law\" />",
            ) as apply_standard_ids_to_xml,
            patch("etl.defs.h_taxonomy_asset.upsert_xml"),
            patch("etl.defs.h_taxonomy_asset.refresh_latest_sections_search"),
            patch("etl.defs.h_taxonomy_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, taxonomy_asset.op.compute_fn), "decorated_fn")
            _ = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                db=cast(object, db),
                taxonomy_model=cast(object, taxonomy_model),
                pipeline_config=cast(object, pipeline_config),
            )

        apply_standard_ids_to_xml.assert_called_once_with(
            "<document />",
            {"section-1": "gold_governing_law"},
        )
        self.assertEqual(conn.update_payloads[0]["label"], "inferred_governing_law")
        self.assertTrue(
            any(
                "x.version = s.xml_version" in sql
                and "x.latest = 1" in sql
                and "x.status = 'verified'" in sql
                for sql in conn.executed_sql
            )
        )
        self.assertTrue(
            any(
                "FROM pdx.xml m" in sql
                and "m.latest = 1" in sql
                and "m.status = 'verified'" in sql
                for sql in conn.executed_sql
            )
        )

    def test_inference_mode_ignores_empty_array_gold_label_for_xml(self) -> None:
        conn = _FakeConn(
            sec_rows=[
                {
                    "section_uuid": "section-1",
                    "agreement_uuid": "agreement-1",
                    "article_title": "ARTICLE I",
                    "section_title": "Section 1",
                    "xml_content": "<section>Body</section>",
                    "section_standard_id_gold_label": "[]",
                }
            ]
        )
        context = SimpleNamespace(log=_FakeLog())
        db = SimpleNamespace(database="pdx", get_engine=lambda: _FakeEngine(conn))
        taxonomy_model = SimpleNamespace(model=lambda: object())
        pipeline_config = SimpleNamespace(
            taxonomy_agreement_batch_size=10,
            taxonomy_mode=TaxonomyMode.INFERENCE,
            queue_run_mode=QueueRunMode.SINGLE_BATCH,
        )

        with (
            patch(
                "etl.defs.h_taxonomy_asset.predict_taxonomy",
                return_value=(
                    [{"section_uuid": "section-1", "agreement_uuid": "agreement-1"}],
                    [{"label": "inferred_governing_law", "alt_probs": [0.8, 0.1, 0.1]}],
                ),
            ),
            patch(
                "etl.defs.h_taxonomy_asset.apply_standard_ids_to_xml",
                return_value="<document standardId=\"inferred_governing_law\" />",
            ) as apply_standard_ids_to_xml,
            patch("etl.defs.h_taxonomy_asset.upsert_xml"),
            patch("etl.defs.h_taxonomy_asset.refresh_latest_sections_search"),
            patch("etl.defs.h_taxonomy_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, taxonomy_asset.op.compute_fn), "decorated_fn")
            _ = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                db=cast(object, db),
                taxonomy_model=cast(object, taxonomy_model),
                pipeline_config=cast(object, pipeline_config),
            )

        apply_standard_ids_to_xml.assert_called_once_with(
            "<document />",
            {"section-1": "inferred_governing_law"},
        )

    def test_gold_backfill_mode_does_not_run_inference(self) -> None:
        conn = _FakeConn(
            sec_rows=[
                {
                    "section_uuid": "section-1",
                    "agreement_uuid": "agreement-1",
                    "section_standard_id_gold_label": "gold_governing_law",
                }
            ]
        )
        context = SimpleNamespace(log=_FakeLog())
        db = SimpleNamespace(database="pdx", get_engine=lambda: _FakeEngine(conn))
        taxonomy_model = SimpleNamespace(
            model=lambda: (_ for _ in ()).throw(
                AssertionError("taxonomy model should not be loaded in gold_backfill mode")
            )
        )
        pipeline_config = SimpleNamespace(
            taxonomy_agreement_batch_size=10,
            taxonomy_mode=TaxonomyMode.GOLD_BACKFILL,
            queue_run_mode=QueueRunMode.SINGLE_BATCH,
        )

        with (
            patch(
                "etl.defs.h_taxonomy_asset.predict_taxonomy",
                side_effect=AssertionError(
                    "predict_taxonomy should not run in gold_backfill mode"
                ),
            ),
            patch(
                "etl.defs.h_taxonomy_asset.apply_standard_ids_to_xml",
                return_value="<document standardId=\"gold_governing_law\" />",
            ) as apply_standard_ids_to_xml,
            patch("etl.defs.h_taxonomy_asset.upsert_xml") as upsert_xml,
            patch("etl.defs.h_taxonomy_asset.refresh_latest_sections_search") as refresh,
            patch("etl.defs.h_taxonomy_asset.run_post_asset_refresh", return_value=None),
        ):
            decorated_fn = getattr(cast(object, taxonomy_asset.op.compute_fn), "decorated_fn")
            _ = decorated_fn(
                cast(AssetExecutionContext, cast(object, context)),
                db=cast(object, db),
                taxonomy_model=cast(object, taxonomy_model),
                pipeline_config=cast(object, pipeline_config),
            )

        apply_standard_ids_to_xml.assert_called_once_with(
            "<document />",
            {"section-1": "gold_governing_law"},
        )
        upsert_xml.assert_called_once()
        refresh.assert_called_once_with(conn, "pdx", ["agreement-1"])
        self.assertFalse(any("UPDATE pdx.sections" in sql for sql in conn.executed_sql))
        self.assertTrue(
            any(
                "x.version = s.xml_version" in sql
                and "x.latest = 1" in sql
                and "x.status = 'verified'" in sql
                for sql in conn.executed_sql
            )
        )
        self.assertTrue(
            any(
                "FROM pdx.xml m" in sql
                and "m.latest = 1" in sql
                and "m.status = 'verified'" in sql
                for sql in conn.executed_sql
            )
        )
        self.assertTrue(
            any("TRIM(s.section_standard_id_gold_label) <> '[]'" in sql for sql in conn.executed_sql)
        )


if __name__ == "__main__":
    _ = unittest.main()
