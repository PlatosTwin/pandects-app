"""Dagster resource definitions for the ETL pipeline.

This module defines the configurable resources used throughout the ETL pipeline,
including database connections, ML models, and pipeline configuration.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

from enum import Enum
from pathlib import Path
from typing import Any

import dagster as dg
import yaml
from pydantic import PrivateAttr
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from etl.models.page_classifier.classifier import ClassifierInference
from etl.models.ner.ner import NERInference
from etl.models.ner.ner_constants import NER_CKPT_PATH, NER_LABEL_LIST
from etl.models.page_classifier.page_classifier_constants import (
    CLASSIFIER_CKPT_PATH,
    CLASSIFIER_XGB_PATH,
)
from etl.models.taxonomy.taxonomy_constants import TAXONOMY_LABEL_LIST


class PipelineMode(Enum):
    """Pipeline execution modes."""

    FROM_SCRATCH = "from_scratch"
    CLEANUP = "cleanup"


class ProcessingScope(Enum):
    """Scope of processing for a single run."""

    BATCHED = "batched"
    FULL = "full"


class AiRepairEntityFocus(Enum):
    """Entity focus for AI repair filtering."""

    ARTICLE = "article"
    SECTION = "section"
    PAGE = "page"
    O = "o"


class AiRepairMode(Enum):
    """Mode for AI repair enqueue behavior."""

    EXCERPT = "excerpt"
    ALL = "all"


class TxMetadataMode(Enum):
    """Transaction metadata enrichment mode: offline (document-only) or web_search."""

    OFFLINE = "offline"
    WEB_SEARCH = "web_search"


class PipelineConfig(dg.ConfigurableResource[object]):
    """Configuration for pipeline execution mode and batching behavior."""

    mode: PipelineMode = PipelineMode.FROM_SCRATCH
    scope: ProcessingScope = ProcessingScope.BATCHED
    tagging_agreement_batch_size: int = 500  # used in tagging_asset
    pre_processing_agreement_batch_size: int = 5  # used in pre_processing_asset
    xml_agreement_batch_size: int = 10  # used in xml_asset
    sections_agreement_batch_size: int = 10  # used in sections_asset
    taxonomy_agreement_batch_size: int = 50  # used in taxonomy_asset
    ai_repair_agreement_batch_size: int = 150  # used in ai_repair_enqueue_asset
    ai_repair_mode: AiRepairMode = AiRepairMode.EXCERPT  # excerpt | all
    ai_repair_entity_focus: AiRepairEntityFocus = AiRepairEntityFocus.O
    ai_repair_confidence_threshold: float = 1.0
    reconcile_tags_agreement_batch_size: int = 500  # used in reconcile_tags asset
    tx_metadata_agreement_batch_size: int = 10  # used in tx_metadata_asset
    tx_metadata_mode: TxMetadataMode = TxMetadataMode.OFFLINE  # offline | web_search
    staging_days_to_fetch: int = 2  # used in staging_asset alt flow
    staging_rate_limit_max_requests: int = 10  # used in staging_asset alt flow
    staging_rate_limit_window_seconds: float = 1.025  # used in staging_asset alt flow
    staging_max_workers: int = 8  # used in staging_asset alt flow
    staging_use_keyword_filter: bool = True  # used in staging_asset alt flow

    def is_cleanup_mode(self) -> bool:
        """Check if the pipeline is running in cleanup mode."""
        return self.mode == PipelineMode.CLEANUP

    def is_batched(self) -> bool:
        """Check if the pipeline should run a single batch per asset invocation."""
        return self.scope == ProcessingScope.BATCHED


class DBResource(dg.ConfigurableResource[object]):
    """Database connection resource."""

    host: str
    port: str
    user: str
    password: str
    database: str

    def get_engine(self) -> Engine:
        """Create a SQLAlchemy engine for the configured database."""
        url = (
            f"mariadb+mysqldb://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )
        return create_engine(url)


class ClassifierModel(dg.ConfigurableResource[object]):
    """Resource for the page classification model."""

    _inf: ClassifierInference | None = PrivateAttr(default=None)

    def model(self):
        if self._inf is None:
            self._inf = ClassifierInference(
                ckpt_path=CLASSIFIER_CKPT_PATH,
                xgb_path=CLASSIFIER_XGB_PATH,
                num_workers=7,
            )
        return self._inf


class TaggingModel(dg.ConfigurableResource[object]):
    """Resource for the NER tagging model."""

    def model(self) -> NERInference:
        """Load and return the NER tagging model."""
        model = NERInference(
            ckpt_path=NER_CKPT_PATH, label_list=NER_LABEL_LIST, review_threshold=0.80
        )
        return model


class TaxonomyModel(dg.ConfigurableResource[object]):
    """Placeholder resource for section taxonomy classification.

    The real model will return a primary label and probabilities for the next
    three most likely labels. For now, we return a deterministic dummy label
    ("other") with zeroed alternate probabilities.
    """

    def model(self):
        class _DummyTaxonomyModel:
            def __init__(self, label_list: list[str]):
                self.label_list = label_list

            def predict(self, rows: list[dict[str, object]]) -> list[dict[str, object]]:
                # rows items may contain article_title, section_title, section_text
                out: list[dict[str, object]] = []
                primary = (
                    "other"
                    if "other" in self.label_list
                    else (self.label_list[0] if self.label_list else "other")
                )
                for _ in rows:
                    out.append(
                        {
                            "label": primary,
                            "alt_probs": [0.0, 0.0, 0.0],
                        }
                    )
                return out

        return _DummyTaxonomyModel(TAXONOMY_LABEL_LIST)


def _load_yaml_config() -> dict[str, Any]:
    """Load pipeline configuration from YAML file.

    Returns:
        Dictionary with config values from YAML, or empty dict if file not found.
    """
    config_path = Path(__file__).parent.parent.parent.parent / "configs" / "etl_pipeline.yaml"
    if not config_path.exists():
        return {}
    
    try:
        with config_path.open() as f:
            yaml_data: Any = yaml.safe_load(f)
            if yaml_data and "resources" in yaml_data:
                pipeline_config = yaml_data["resources"].get("pipeline_config", {})
                return pipeline_config.get("config", {})
    except Exception:
        # If YAML loading fails, return empty dict to fall back to defaults
        return {}
    
    return {}


def get_resources() -> dict[str, object]:
    """Get the base resource configuration for the pipeline.

    Loads defaults from etl/configs/etl_pipeline.yaml if available.
    Values can still be overridden via Dagster's -c flag at runtime.

    Returns:
        Dictionary containing all resource definitions.
    """
    yaml_config = _load_yaml_config()
    
    # Convert YAML values to appropriate types, with fallback to current defaults
    pipeline_config_kwargs: dict[str, Any] = {}
    
    if "mode" in yaml_config:
        mode_str = str(yaml_config["mode"]).lower()
        pipeline_config_kwargs["mode"] = PipelineMode(mode_str)
    
    if "scope" in yaml_config:
        scope_str = str(yaml_config["scope"]).lower()
        pipeline_config_kwargs["scope"] = ProcessingScope(scope_str)
    
    if "tagging_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["tagging_agreement_batch_size"] = int(yaml_config["tagging_agreement_batch_size"])

    if "pre_processing_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["pre_processing_agreement_batch_size"] = int(yaml_config["pre_processing_agreement_batch_size"])
    
    if "xml_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["xml_agreement_batch_size"] = int(yaml_config["xml_agreement_batch_size"])
    
    if "sections_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["sections_agreement_batch_size"] = int(yaml_config["sections_agreement_batch_size"])
    
    if "taxonomy_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["taxonomy_agreement_batch_size"] = int(yaml_config["taxonomy_agreement_batch_size"])
    
    if "ai_repair_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["ai_repair_agreement_batch_size"] = int(yaml_config["ai_repair_agreement_batch_size"])

    if "ai_repair_mode" in yaml_config:
        mode_str = str(yaml_config["ai_repair_mode"]).lower()
        pipeline_config_kwargs["ai_repair_mode"] = AiRepairMode(mode_str)
    
    if "ai_repair_entity_focus" in yaml_config:
        focus_str = str(yaml_config["ai_repair_entity_focus"]).lower()
        pipeline_config_kwargs["ai_repair_entity_focus"] = AiRepairEntityFocus(focus_str)
    
    if "ai_repair_confidence_threshold" in yaml_config:
        pipeline_config_kwargs["ai_repair_confidence_threshold"] = float(yaml_config["ai_repair_confidence_threshold"])
    
    if "reconcile_tags_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["reconcile_tags_agreement_batch_size"] = int(yaml_config["reconcile_tags_agreement_batch_size"])
    
    if "tx_metadata_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["tx_metadata_agreement_batch_size"] = int(yaml_config["tx_metadata_agreement_batch_size"])
    
    if "tx_metadata_mode" in yaml_config:
        mode_str = str(yaml_config["tx_metadata_mode"]).lower()
        pipeline_config_kwargs["tx_metadata_mode"] = TxMetadataMode(mode_str)
    
    if "staging_days_to_fetch" in yaml_config:
        pipeline_config_kwargs["staging_days_to_fetch"] = int(yaml_config["staging_days_to_fetch"])
    
    if "staging_rate_limit_max_requests" in yaml_config:
        pipeline_config_kwargs["staging_rate_limit_max_requests"] = int(yaml_config["staging_rate_limit_max_requests"])
    
    if "staging_rate_limit_window_seconds" in yaml_config:
        pipeline_config_kwargs["staging_rate_limit_window_seconds"] = float(yaml_config["staging_rate_limit_window_seconds"])
    
    if "staging_max_workers" in yaml_config:
        pipeline_config_kwargs["staging_max_workers"] = int(yaml_config["staging_max_workers"])
    
    if "staging_use_keyword_filter" in yaml_config:
        pipeline_config_kwargs["staging_use_keyword_filter"] = bool(yaml_config["staging_use_keyword_filter"])
    
    return {
        "db": DBResource(
            user=dg.EnvVar("MARIADB_USER"),
            password=dg.EnvVar("MARIADB_PASSWORD"),
            host=dg.EnvVar("MARIADB_HOST"),
            port=dg.EnvVar("MARIADB_PORT"),
            database=dg.EnvVar("MARIADB_DATABASE"),
        ),
        "classifier_model": ClassifierModel(),
        "tagging_model": TaggingModel(),
        "taxonomy_model": TaxonomyModel(),
        "pipeline_config": PipelineConfig(**pipeline_config_kwargs),  # type: ignore
    }
