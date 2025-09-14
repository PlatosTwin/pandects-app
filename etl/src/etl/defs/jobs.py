import dagster as dg
from etl.defs.staging_asset import staging_asset
from etl.defs.pre_processing_asset import pre_processing_asset
from etl.defs.tagging_asset import tagging_asset
from etl.defs.xml_asset import xml_asset
from etl.defs.resources import get_resources

base_resources = get_resources()

etl_pipeline = dg.define_asset_job(
    name="etl_pipeline",
    selection=dg.AssetSelection.assets(
        staging_asset, pre_processing_asset, tagging_asset, xml_asset
    ),
    tags={"pipeline_mode": "from_scratch"},
)

cleanup_pipeline = dg.define_asset_job(
    name="cleanup_pipeline",
    selection=dg.AssetSelection.assets(
        staging_asset, pre_processing_asset, tagging_asset
    ),
    tags={"pipeline_mode": "cleanup"},
)

defs = dg.Definitions(
    assets=[staging_asset, pre_processing_asset, tagging_asset, xml_asset],
    jobs=[etl_pipeline, cleanup_pipeline],
    resources=base_resources,
)
