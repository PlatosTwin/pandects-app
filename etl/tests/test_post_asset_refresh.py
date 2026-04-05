# pyright: reportAny=false
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from dagster import AssetExecutionContext

from etl.defs.resources import DBResource, PipelineConfig
from etl.utils.post_asset_refresh import run_post_asset_refresh


class PostAssetRefreshTests(unittest.TestCase):
    def test_refresh_disabled_is_no_op(self) -> None:
        context = SimpleNamespace(asset_key=SimpleNamespace(path=["3_tagging_asset"]))
        db = SimpleNamespace(database="pdx")
        pipeline_config = SimpleNamespace(refresh=False)

        with (
            patch("etl.utils.post_asset_refresh.apply_gating") as apply_gating,
            patch("etl.utils.post_asset_refresh.refresh_summary_data") as refresh_summary_data,
        ):
            run_post_asset_refresh(
                cast(AssetExecutionContext, cast(object, context)),
                cast(DBResource, cast(object, db)),
                cast(PipelineConfig, cast(object, pipeline_config)),
            )
            apply_gating.assert_not_called()
            refresh_summary_data.assert_not_called()

    def test_refresh_enabled_runs_for_taxonomy_metadata_and_embed_assets(self) -> None:
        db = SimpleNamespace(database="pdx", get_engine=lambda: SimpleNamespace(begin=lambda: None))
        pipeline_config = SimpleNamespace(refresh=True)

        class _FakeBegin:
            def __enter__(self) -> object:
                return object()

            def __exit__(self, *_exc: object) -> None:
                return None

        db = SimpleNamespace(database="pdx", get_engine=lambda: SimpleNamespace(begin=lambda: _FakeBegin()))

        for asset_name in ("7_taxonomy_asset", "8_tx_metadata_asset", "9_embed_sections"):
            context = SimpleNamespace(asset_key=SimpleNamespace(path=[asset_name]))
            with (
                patch("etl.utils.post_asset_refresh.apply_gating") as apply_gating,
                patch("etl.utils.post_asset_refresh.refresh_summary_data") as refresh_summary_data,
            ):
                run_post_asset_refresh(
                    cast(AssetExecutionContext, cast(object, context)),
                    cast(DBResource, cast(object, db)),
                    cast(PipelineConfig, cast(object, pipeline_config)),
                )
                apply_gating.assert_called_once()
                refresh_summary_data.assert_called_once()


if __name__ == "__main__":
    _ = unittest.main()
