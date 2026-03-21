# pyright: reportAny=false
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

from dagster import AssetExecutionContext, build_op_context

from etl.defs.g_sections_asset import sections_from_fresh_xml_asset
from etl.defs.resources import DBResource, PipelineConfig


class FreshSectionsAssetTests(unittest.TestCase):
    def test_sections_from_fresh_xml_asset_falls_back_to_queue_when_upstream_empty(self) -> None:
        context = build_op_context()
        db = SimpleNamespace()
        pipeline_config = SimpleNamespace()

        with patch("etl.defs.g_sections_asset._run_sections_for_agreements", return_value=[]) as runner:
            sections_from_fresh_xml_asset(
                context=cast(AssetExecutionContext, cast(object, context)),
                db=cast(DBResource, cast(object, db)),
                pipeline_config=cast(PipelineConfig, cast(object, pipeline_config)),
                verified_fresh_agreement_uuids=[],
            )

        runner.assert_called_once_with(
            cast(AssetExecutionContext, cast(object, context)),
            cast(DBResource, cast(object, db)),
            cast(PipelineConfig, cast(object, pipeline_config)),
            target_agreement_uuids=None,
            log_prefix="sections_from_fresh_xml_asset",
        )

    def test_sections_from_fresh_xml_asset_preserves_run_scope_when_upstream_present(self) -> None:
        context = build_op_context()
        db = SimpleNamespace()
        pipeline_config = SimpleNamespace()
        target_agreement_uuids = ["agreement-1"]

        with patch("etl.defs.g_sections_asset._run_sections_for_agreements", return_value=[]) as runner:
            sections_from_fresh_xml_asset(
                context=cast(AssetExecutionContext, cast(object, context)),
                db=cast(DBResource, cast(object, db)),
                pipeline_config=cast(PipelineConfig, cast(object, pipeline_config)),
                verified_fresh_agreement_uuids=target_agreement_uuids,
            )

        runner.assert_called_once_with(
            cast(AssetExecutionContext, cast(object, context)),
            cast(DBResource, cast(object, db)),
            cast(PipelineConfig, cast(object, pipeline_config)),
            target_agreement_uuids=target_agreement_uuids,
            log_prefix="sections_from_fresh_xml_asset",
        )


if __name__ == "__main__":
    unittest.main()
