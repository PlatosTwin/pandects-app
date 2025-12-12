from __future__ import annotations

import dagster as dg

from etl.defs.resources import PipelineConfig, PipelineMode, ProcessingScope


def get_pipeline_mode(
    context: dg.AssetExecutionContext, pipeline_config: PipelineConfig
) -> PipelineMode:
    tag_val = context.run.tags.get("pipeline_mode")
    if tag_val is None:
        return pipeline_config.mode
    try:
        return PipelineMode(tag_val)
    except ValueError as e:
        raise ValueError(f"Invalid `pipeline_mode` tag: {tag_val!r}") from e


def is_cleanup_mode(
    context: dg.AssetExecutionContext, pipeline_config: PipelineConfig
) -> bool:
    return get_pipeline_mode(context, pipeline_config) == PipelineMode.CLEANUP


def get_processing_scope(
    context: dg.AssetExecutionContext, pipeline_config: PipelineConfig
) -> ProcessingScope:
    tag_val = context.run.tags.get("run_scope")
    if tag_val is None:
        return pipeline_config.scope
    try:
        return ProcessingScope(tag_val)
    except ValueError as e:
        raise ValueError(f"Invalid `run_scope` tag: {tag_val!r}") from e


def is_batched(
    context: dg.AssetExecutionContext, pipeline_config: PipelineConfig
) -> bool:
    return (
        get_processing_scope(context, pipeline_config) == ProcessingScope.BATCHED
    )


def get_int_tag(context: dg.AssetExecutionContext, tag_name: str) -> int | None:
    val = context.run.tags.get(tag_name)
    if val is None:
        return None
    try:
        return int(val)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid `{tag_name}` tag (expected int): {val!r}") from e

