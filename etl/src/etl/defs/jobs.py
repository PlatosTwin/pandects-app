from dagster import job
from etl.defs.staging_asset import staging_asset
from etl.defs.pre_processing_asset import pre_processing_asset
from etl.defs.tagging_asset import tagging_asset


@job
def etl_pipeline():
    # 1) run staging
    staged = staging_asset()

    # 2) feed that into pre-processing
    pre_processed = pre_processing_asset(staged)

    # 3) finally tagging
    tagging_asset(pre_processed)
