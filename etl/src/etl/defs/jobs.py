"""Dagster job definitions for the ETL pipeline.

This module defines the main ETL pipeline jobs that orchestrate the asset execution.
"""

from dagster import job
from etl.defs.staging_asset import staging_asset
from etl.defs.pre_processing_asset import pre_processing_asset
from etl.defs.tagging_asset import tagging_asset
from etl.defs.xml_asset import xml_asset
from etl.defs.resources import PipelineMode, PipelineConfig, get_resources

# Get the base resources dict
base_resources = get_resources()


@job(resource_defs={**base_resources, "pipeline_config": PipelineConfig(mode=PipelineMode.FROM_SCRATCH)})
def etl_pipeline():
    """Main ETL pipeline that processes agreements from scratch.
    
    This job runs the complete ETL pipeline in FROM_SCRATCH mode,
    processing new filings through all stages: staging, pre-processing,
    tagging, and XML generation.
    """
    staged = staging_asset()
    pre_processed = pre_processing_asset(staged)
    tagged = tagging_asset(pre_processed)
    xml_asset(tagged)


@job(resource_defs={**base_resources, "pipeline_config": PipelineConfig(mode=PipelineMode.CLEANUP)})
def cleanup_pipeline():
    """Cleanup pipeline that reprocesses existing unprocessed data.
    
    This job runs the ETL pipeline in CLEANUP mode, processing only
    existing unprocessed agreements through all stages.
    """
    staged = staging_asset()
    pre_processed = pre_processing_asset(staged)
    tagged = tagging_asset(pre_processed)
    xml_asset(tagged)