from __future__ import annotations

import dagster as dg

from etl.defs.resources import PipelineConfig, PreProcessingMode, ProcessingScope


def get_pre_processing_mode(
    _context: dg.AssetExecutionContext, pipeline_config: PipelineConfig
) -> PreProcessingMode:
    return pipeline_config.pre_processing_mode


def is_pre_processing_cleanup_mode(
    _context: dg.AssetExecutionContext, pipeline_config: PipelineConfig
) -> bool:
    return (
        get_pre_processing_mode(_context, pipeline_config)
        == PreProcessingMode.CLEANUP
    )


def get_processing_scope(
    _context: dg.AssetExecutionContext, pipeline_config: PipelineConfig
) -> ProcessingScope:
    return pipeline_config.scope


def is_batched(
    _context: dg.AssetExecutionContext, pipeline_config: PipelineConfig
) -> bool:
    return (
        get_processing_scope(_context, pipeline_config) == ProcessingScope.BATCHED
    )
