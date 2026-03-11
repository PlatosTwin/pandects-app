from __future__ import annotations

import dagster as dg

from etl.defs.resources import PipelineConfig, PreProcessingMode, QueueRunMode


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


def get_queue_run_mode(
    _context: dg.AssetExecutionContext, pipeline_config: PipelineConfig
) -> QueueRunMode:
    return pipeline_config.queue_run_mode


def runs_single_batch(
    _context: dg.AssetExecutionContext, pipeline_config: PipelineConfig
) -> bool:
    return (
        get_queue_run_mode(_context, pipeline_config) == QueueRunMode.SINGLE_BATCH
    )


def ensure_single_batch_run(
    context: dg.AssetExecutionContext,
    pipeline_config: PipelineConfig,
    *,
    asset_name: str,
) -> None:
    if not runs_single_batch(context, pipeline_config):
        raise ValueError(
            f"{asset_name} requires pipeline_config.queue_run_mode='single_batch'."
        )
