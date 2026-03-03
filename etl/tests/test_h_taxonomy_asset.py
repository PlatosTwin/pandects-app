# pyright: reportAny=false
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from dagster import AssetExecutionContext

from etl.defs.h_taxonomy_asset import taxonomy_asset


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
    def __init__(self) -> None:
        self.executed_sql: list[str] = []

    def execute(self, statement: object, _params: object | None = None) -> _FakeResult:
        sql = str(statement)
        self.executed_sql.append(sql)
        if "SELECT DISTINCT agreement_uuid" in sql:
            return _FakeResult(rows=[{"agreement_uuid": "agreement-1"}])
        if "SELECT section_uuid," in sql and "FROM pdx.sections" in sql:
            return _FakeResult(
                rows=[
                    {
                        "section_uuid": "section-1",
                        "agreement_uuid": "agreement-1",
                        "article_title": "ARTICLE I",
                        "section_title": "Section 1",
                        "xml_content": "<section>Body</section>",
                    }
                ]
            )
        if "UPDATE pdx.sections" in sql:
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
    def __init__(self, conn: _FakeConn) -> None:
        self._conn = conn

    def begin(self) -> _FakeBeginContext:
        return _FakeBeginContext(self._conn)


class _FakeLog:
    def info(self, *args: object, **kwargs: object) -> None:
        _ = args
        _ = kwargs


class TaxonomyAssetProjectionRefreshTests(unittest.TestCase):
    def test_taxonomy_asset_refreshes_latest_sections_search(self) -> None:
        conn = _FakeConn()
        context = SimpleNamespace(log=_FakeLog())
        db = SimpleNamespace(database="pdx", get_engine=lambda: _FakeEngine(conn))
        taxonomy_model = SimpleNamespace(model=lambda: object())
        pipeline_config = SimpleNamespace(taxonomy_agreement_batch_size=10)

        with (
            patch("etl.defs.h_taxonomy_asset.is_batched", return_value=True),
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


if __name__ == "__main__":
    _ = unittest.main()
