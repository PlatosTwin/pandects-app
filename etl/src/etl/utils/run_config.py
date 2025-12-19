from __future__ import annotations

import dagster as dg

from etl.defs.resources import PipelineConfig, PipelineMode, ProcessingScope


def get_pipeline_mode(
    context: dg.AssetExecutionContext, pipeline_config: PipelineConfig
) -> PipelineMode:
    return pipeline_config.mode


def is_cleanup_mode(
    context: dg.AssetExecutionContext, pipeline_config: PipelineConfig
) -> bool:
    return get_pipeline_mode(context, pipeline_config) == PipelineMode.CLEANUP


def get_processing_scope(
    context: dg.AssetExecutionContext, pipeline_config: PipelineConfig
) -> ProcessingScope:
    return pipeline_config.scope


def is_batched(
    context: dg.AssetExecutionContext, pipeline_config: PipelineConfig
) -> bool:
    return (
        get_processing_scope(context, pipeline_config) == ProcessingScope.BATCHED
    )
