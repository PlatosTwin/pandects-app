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
        context = SimpleNamespace(asset_key=SimpleNamespace(path=["03_tagging_asset"]))
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

    def test_refresh_enabled_runs_only_for_terminal_assets(self) -> None:
        db = SimpleNamespace(database="pdx", get_engine=lambda: SimpleNamespace(begin=lambda: None))
        pipeline_config = SimpleNamespace(refresh=True)

        class _FakeBegin:
            def __enter__(self) -> object:
                return object()

            def __exit__(self, *_exc: object) -> None:
                return None

        db = SimpleNamespace(database="pdx", get_engine=lambda: SimpleNamespace(begin=lambda: _FakeBegin()))

        for asset_name in (
            "07_taxonomy_asset",
            "10_tx_metadata_asset",
            "10-02_regular_ingest_tx_metadata_web_search_asset",
            "10-04_ingestion_cleanup_a_tx_metadata_web_search_asset",
            "10-06_ingestion_cleanup_b_tx_metadata_web_search_asset",
            "10-08_ingestion_cleanup_c_tx_metadata_web_search_asset",
            "10-10_ingestion_cleanup_d_tx_metadata_web_search_asset",
            "11_embed_sections",
        ):
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

    def test_refresh_enabled_skips_non_terminal_assets(self) -> None:
        class _FakeBegin:
            def __enter__(self) -> object:
                return object()

            def __exit__(self, *_exc: object) -> None:
                return None

        db = SimpleNamespace(database="pdx", get_engine=lambda: SimpleNamespace(begin=lambda: _FakeBegin()))
        pipeline_config = SimpleNamespace(refresh=True)

        for asset_name in (
            "03_tagging_asset",
            "10-01_regular_ingest_tx_metadata_offline_asset",
            "10-03_ingestion_cleanup_a_tx_metadata_offline_asset",
            "10-05_ingestion_cleanup_b_tx_metadata_offline_asset",
            "10-07_ingestion_cleanup_c_tx_metadata_offline_asset",
            "10-09_ingestion_cleanup_d_tx_metadata_offline_asset",
        ):
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
                apply_gating.assert_not_called()
                refresh_summary_data.assert_not_called()


if __name__ == "__main__":
    _ = unittest.main()
