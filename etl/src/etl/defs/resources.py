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

PIPELINE_CONFIG_FILENAME = "pipeline_config.yaml"


class PreProcessingMode(Enum):
    """Pre-processing execution modes."""

    FROM_SCRATCH = "from_scratch"
    CLEANUP = "cleanup"


class ProcessingScope(Enum):
    """Scope of processing for a single run."""

    BATCHED = "batched"
    FULL = "full"


class TxMetadataMode(Enum):
    """Transaction metadata enrichment mode: offline (document-only) or web_search."""

    OFFLINE = "offline"
    WEB_SEARCH = "web_search"


class PipelineConfig(dg.ConfigurableResource[object]):
    """Configuration for pre-processing mode and batching behavior."""

    pre_processing_mode: PreProcessingMode = PreProcessingMode.FROM_SCRATCH
    scope: ProcessingScope = ProcessingScope.BATCHED
    refresh: bool = False  # run end-of-asset gating + summary refresh
    resume_open_batches: bool = True  # resume matching in-flight LLM batches when possible
    tagging_agreement_batch_size: int = 500  # used in tagging_asset
    pre_processing_agreement_batch_size: int = 5  # used in pre_processing_asset
    xml_agreement_batch_size: int = 10  # used across XML + AI-repair cycle assets
    taxonomy_agreement_batch_size: int = 50  # used in taxonomy_asset
    tx_metadata_agreement_batch_size: int = 10  # used in tx_metadata_asset
    tx_metadata_mode: TxMetadataMode = TxMetadataMode.OFFLINE  # offline | web_search
    staging_days_to_fetch: int = 2  # used in staging_asset alt flow
    staging_rate_limit_max_requests: int = 10  # used in staging_asset alt flow
    staging_rate_limit_window_seconds: float = 1.025  # used in staging_asset alt flow
    staging_max_workers: int = 8  # used in staging_asset alt flow
    staging_use_keyword_filter: bool = True  # used in staging_asset alt flow

    def is_pre_processing_cleanup_mode(self) -> bool:
        """Check if pre-processing should run in cleanup mode."""
        return self.pre_processing_mode == PreProcessingMode.CLEANUP

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


def _parse_bool(value: Any, *, field_name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return bool(value)
        raise ValueError(f"{field_name} must be boolean-like (0/1), got {value!r}.")
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    raise ValueError(f"{field_name} must be a boolean value, got {value!r}.")


def _load_yaml_config() -> dict[str, Any]:
    """Load pipeline configuration from YAML file.

    Returns:
        Dictionary with config values from YAML, or empty dict if file not found.
    """
    config_path = (
        Path(__file__).parent.parent.parent.parent
        / "configs"
        / PIPELINE_CONFIG_FILENAME
    )
    if not config_path.exists():
        return {}
    
    try:
        with config_path.open() as f:
            yaml_data: Any = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"Failed to load ETL pipeline config at {config_path}: {exc}") from exc

    if yaml_data is None:
        return {}
    if not isinstance(yaml_data, dict):
        raise ValueError(f"{PIPELINE_CONFIG_FILENAME} root must be a mapping.")

    resources = yaml_data.get("resources")
    if resources is None:
        return {}
    if not isinstance(resources, dict):
        raise ValueError(
            f"{PIPELINE_CONFIG_FILENAME} field 'resources' must be a mapping."
        )

    pipeline_config = resources.get("pipeline_config")
    if pipeline_config is None:
        return {}
    if not isinstance(pipeline_config, dict):
        raise ValueError(
            f"{PIPELINE_CONFIG_FILENAME} field 'resources.pipeline_config' must be a mapping."
        )

    config = pipeline_config.get("config")
    if config is None:
        return {}
    if not isinstance(config, dict):
        raise ValueError(
            f"{PIPELINE_CONFIG_FILENAME} field 'resources.pipeline_config.config' must be a mapping."
        )

    return config


def get_resources() -> dict[str, object]:
    """Get the base resource configuration for the pipeline.

    Loads defaults from etl/configs/pipeline_config.yaml if available.
    Values can still be overridden via Dagster's -c flag at runtime.

    Returns:
        Dictionary containing all resource definitions.
    """
    yaml_config = _load_yaml_config()
    
    # Convert YAML values to appropriate types, with fallback to current defaults
    allowed_pipeline_config_keys = {
        "pre_processing_mode",
        "scope",
        "refresh",
        "resume_open_batches",
        "tagging_agreement_batch_size",
        "pre_processing_agreement_batch_size",
        "xml_agreement_batch_size",
        "taxonomy_agreement_batch_size",
        "tx_metadata_agreement_batch_size",
        "tx_metadata_mode",
        "staging_days_to_fetch",
        "staging_rate_limit_max_requests",
        "staging_rate_limit_window_seconds",
        "staging_max_workers",
        "staging_use_keyword_filter",
    }
    unknown_keys = sorted(set(yaml_config) - allowed_pipeline_config_keys)
    if unknown_keys:
        raise ValueError(
            "Unknown keys in resources.pipeline_config.config: "
            + ", ".join(unknown_keys)
        )

    pipeline_config_kwargs: dict[str, Any] = {}
    
    if "pre_processing_mode" in yaml_config:
        mode_str = str(yaml_config["pre_processing_mode"]).lower()
        pipeline_config_kwargs["pre_processing_mode"] = PreProcessingMode(mode_str)
    
    if "scope" in yaml_config:
        scope_str = str(yaml_config["scope"]).lower()
        pipeline_config_kwargs["scope"] = ProcessingScope(scope_str)

    if "refresh" in yaml_config:
        pipeline_config_kwargs["refresh"] = _parse_bool(
            yaml_config["refresh"],
            field_name="refresh",
        )

    if "resume_open_batches" in yaml_config:
        pipeline_config_kwargs["resume_open_batches"] = _parse_bool(
            yaml_config["resume_open_batches"],
            field_name="resume_open_batches",
        )
    
    if "tagging_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["tagging_agreement_batch_size"] = int(yaml_config["tagging_agreement_batch_size"])

    if "pre_processing_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["pre_processing_agreement_batch_size"] = int(yaml_config["pre_processing_agreement_batch_size"])
    
    if "xml_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["xml_agreement_batch_size"] = int(yaml_config["xml_agreement_batch_size"])

    if "taxonomy_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["taxonomy_agreement_batch_size"] = int(yaml_config["taxonomy_agreement_batch_size"])
    
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
        pipeline_config_kwargs["staging_use_keyword_filter"] = _parse_bool(
            yaml_config["staging_use_keyword_filter"],
            field_name="staging_use_keyword_filter",
        )
    
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
