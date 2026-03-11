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

from etl.models.page_classifier_revamp.inference import ClassifierInference
from etl.models.page_classifier_revamp.review_model import ReviewModelInference
from etl.models.ner.ner import NERInference
from etl.models.ner.ner_constants import NER_CKPT_PATH, NER_LABEL_LIST
from etl.models.page_classifier_revamp.page_classifier_constants import (
    CLASSIFIER_CRF_PATH,
    CLASSIFIER_REVIEW_MODEL_PATH,
)
from etl.models.taxonomy.taxonomy import TaxonomyInference
from etl.models.taxonomy.taxonomy_constants import (
    TAXONOMY_CKPT_PATH,
    TAXONOMY_TITLE_RULES_PATH,
    TAXONOMY_VECTORIZER_PATH,
)

PIPELINE_CONFIG_FILENAME = "pipeline_config.yaml"


class PreProcessingMode(Enum):
    """Pre-processing execution modes."""

    FROM_SCRATCH = "from_scratch"
    CLEANUP = "cleanup"


class QueueRunMode(Enum):
    """Execution mode for queue-draining assets."""

    SINGLE_BATCH = "single_batch"
    DRAIN = "drain"


class TxMetadataMode(Enum):
    """Transaction metadata enrichment mode: offline (document-only) or web_search."""

    OFFLINE = "offline"
    WEB_SEARCH = "web_search"


class TaxonomyMode(Enum):
    """Taxonomy asset execution mode."""

    INFERENCE = "inference"
    GOLD_BACKFILL = "gold_backfill"


class EmbedTarget(Enum):
    """Section embedding target: agreement batch or specific section-standard-id batch."""

    AGREEMENT = "agreement"
    SECTION = "section"


class PipelineConfig(dg.ConfigurableResource[object]):
    """Configuration for pre-processing mode and batching behavior."""

    pre_processing_mode: PreProcessingMode = PreProcessingMode.FROM_SCRATCH
    queue_run_mode: QueueRunMode = QueueRunMode.SINGLE_BATCH
    refresh: bool = False  # run end-of-asset gating + summary refresh
    resume_openai_batches: bool = True  # resume matching in-flight OpenAI batches when possible
    tagging_agreement_batch_size: int = 500  # used in tagging_asset
    pre_processing_agreement_batch_size: int = 5  # used in pre_processing_asset
    pre_processing_validate_agreement_batch_size: int = 10  # used in 2-2_validate_pre_processing_asset
    pre_processing_validate_candidate_pages_per_agreement: int = 24  # high-cap candidate page review
    pre_processing_validate_candidate_min_risk: float = 0.10  # minimum calibrated risk to prioritize a page
    pre_processing_validate_ungate_max_remaining_risk: float = 0.08  # conservative release threshold
    pre_processing_validate_min_llm_confidence: float = 0.85  # minimum LLM confidence to apply label change
    pre_processing_validate_min_model_support: float = 0.08  # minimum CRF class probability for relabel support
    pre_processing_validate_model: str = "gpt-5-mini"  # responses/batch model for page validation
    pre_processing_validate_completion_window: str = "24h"  # OpenAI batch completion window
    pre_processing_validate_snippet_chars: int = 1200  # max chars for current page snippet
    xml_agreement_batch_size: int = 10  # used across XML + AI-repair cycle assets
    taxonomy_agreement_batch_size: int = 50  # used in taxonomy_asset
    taxonomy_mode: TaxonomyMode = TaxonomyMode.GOLD_BACKFILL  # inference | gold_backfill
    tx_metadata_agreement_batch_size: int = 10  # used in tx_metadata_asset
    tx_metadata_mode: TxMetadataMode = TxMetadataMode.OFFLINE  # offline | web_search
    embed_agreement_batch_size: int = 10  # used in 9_embed_sections when embed_target=agreement
    embed_focus_section: str = ""  # section_standard_id value when embed_target=section
    embed_focus_section_batch_size: int = 100  # used in 9_embed_sections when embed_target=section
    embed_target: EmbedTarget = EmbedTarget.SECTION  # agreement | section
    staging_days_to_fetch: int = 2  # used in staging_asset alt flow
    staging_rate_limit_max_requests: int = 10  # used in staging_asset alt flow
    staging_rate_limit_window_seconds: float = 1.025  # used in staging_asset alt flow
    staging_max_workers: int = 8  # used in staging_asset alt flow
    staging_use_keyword_filter: bool = True  # used in staging_asset alt flow

    def is_pre_processing_cleanup_mode(self) -> bool:
        """Check if pre-processing should run in cleanup mode."""
        return self.pre_processing_mode == PreProcessingMode.CLEANUP

    def runs_single_batch(self) -> bool:
        """Check if queue-draining assets should stop after one batch."""
        return self.queue_run_mode == QueueRunMode.SINGLE_BATCH


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
        return create_engine(
            url,
            pool_pre_ping=True,
            pool_recycle=3600,
        )


class ClassifierModel(dg.ConfigurableResource[object]):
    """Resource for the page classification model."""

    _inf: ClassifierInference | None = PrivateAttr(default=None)

    def model(self):
        if self._inf is None:
            self._inf = ClassifierInference(
                model_path=CLASSIFIER_CRF_PATH,
            )
        return self._inf


class ReviewModel(dg.ConfigurableResource[object]):
    """Resource for the agreement review model."""

    _inf: ReviewModelInference | None = PrivateAttr(default=None)

    def model(
        self,
        *,
        page_classifier: ClassifierInference | None = None,
    ) -> ReviewModelInference:
        if self._inf is None:
            self._inf = ReviewModelInference(
                model_path=CLASSIFIER_REVIEW_MODEL_PATH,
                page_classifier=page_classifier,
            )
        elif page_classifier is not None and self._inf.page_classifier is None:
            self._inf.page_classifier = page_classifier
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
    """Resource for section taxonomy classification."""

    _inf: TaxonomyInference | None = PrivateAttr(default=None)

    def model(self) -> TaxonomyInference:
        if self._inf is None:
            ckpt_path = Path(TAXONOMY_CKPT_PATH)
            if not ckpt_path.exists():
                raise RuntimeError(
                    f"Taxonomy checkpoint not found at {ckpt_path}."
                )
            self._inf = TaxonomyInference(
                ckpt_path=TAXONOMY_CKPT_PATH,
                label_list=None,
                mode=None,
                vectorizer_path=TAXONOMY_VECTORIZER_PATH,
                title_rules_path=TAXONOMY_TITLE_RULES_PATH,
            )
        return self._inf


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
        "queue_run_mode",
        "refresh",
        "resume_openai_batches",
        "tagging_agreement_batch_size",
        "pre_processing_agreement_batch_size",
        "pre_processing_validate_agreement_batch_size",
        "pre_processing_validate_candidate_pages_per_agreement",
        "pre_processing_validate_candidate_min_risk",
        "pre_processing_validate_ungate_max_remaining_risk",
        "pre_processing_validate_min_llm_confidence",
        "pre_processing_validate_min_model_support",
        "pre_processing_validate_model",
        "pre_processing_validate_completion_window",
        "pre_processing_validate_snippet_chars",
        "xml_agreement_batch_size",
        "taxonomy_agreement_batch_size",
        "taxonomy_mode",
        "tx_metadata_agreement_batch_size",
        "tx_metadata_mode",
        "embed_agreement_batch_size",
        "embed_focus_section",
        "embed_focus_section_batch_size",
        "embed_target",
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
    
    if "queue_run_mode" in yaml_config:
        run_mode_str = str(yaml_config["queue_run_mode"]).lower()
        pipeline_config_kwargs["queue_run_mode"] = QueueRunMode(run_mode_str)

    if "refresh" in yaml_config:
        pipeline_config_kwargs["refresh"] = _parse_bool(
            yaml_config["refresh"],
            field_name="refresh",
        )

    if "resume_openai_batches" in yaml_config:
        pipeline_config_kwargs["resume_openai_batches"] = _parse_bool(
            yaml_config["resume_openai_batches"],
            field_name="resume_openai_batches",
        )
    
    if "tagging_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["tagging_agreement_batch_size"] = int(yaml_config["tagging_agreement_batch_size"])

    if "pre_processing_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["pre_processing_agreement_batch_size"] = int(yaml_config["pre_processing_agreement_batch_size"])

    if "pre_processing_validate_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["pre_processing_validate_agreement_batch_size"] = int(
            yaml_config["pre_processing_validate_agreement_batch_size"]
        )

    if "pre_processing_validate_candidate_pages_per_agreement" in yaml_config:
        pipeline_config_kwargs["pre_processing_validate_candidate_pages_per_agreement"] = int(
            yaml_config["pre_processing_validate_candidate_pages_per_agreement"]
        )

    if "pre_processing_validate_candidate_min_risk" in yaml_config:
        pipeline_config_kwargs["pre_processing_validate_candidate_min_risk"] = float(
            yaml_config["pre_processing_validate_candidate_min_risk"]
        )

    if "pre_processing_validate_ungate_max_remaining_risk" in yaml_config:
        pipeline_config_kwargs["pre_processing_validate_ungate_max_remaining_risk"] = float(
            yaml_config["pre_processing_validate_ungate_max_remaining_risk"]
        )

    if "pre_processing_validate_min_llm_confidence" in yaml_config:
        pipeline_config_kwargs["pre_processing_validate_min_llm_confidence"] = float(
            yaml_config["pre_processing_validate_min_llm_confidence"]
        )

    if "pre_processing_validate_min_model_support" in yaml_config:
        pipeline_config_kwargs["pre_processing_validate_min_model_support"] = float(
            yaml_config["pre_processing_validate_min_model_support"]
        )

    if "pre_processing_validate_model" in yaml_config:
        pipeline_config_kwargs["pre_processing_validate_model"] = str(
            yaml_config["pre_processing_validate_model"]
        ).strip()

    if "pre_processing_validate_completion_window" in yaml_config:
        pipeline_config_kwargs["pre_processing_validate_completion_window"] = str(
            yaml_config["pre_processing_validate_completion_window"]
        ).strip()

    if "pre_processing_validate_snippet_chars" in yaml_config:
        pipeline_config_kwargs["pre_processing_validate_snippet_chars"] = int(
            yaml_config["pre_processing_validate_snippet_chars"]
        )
    
    if "xml_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["xml_agreement_batch_size"] = int(yaml_config["xml_agreement_batch_size"])

    if "taxonomy_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["taxonomy_agreement_batch_size"] = int(yaml_config["taxonomy_agreement_batch_size"])

    if "taxonomy_mode" in yaml_config:
        mode_str = str(yaml_config["taxonomy_mode"]).lower()
        pipeline_config_kwargs["taxonomy_mode"] = TaxonomyMode(mode_str)
    
    if "tx_metadata_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["tx_metadata_agreement_batch_size"] = int(yaml_config["tx_metadata_agreement_batch_size"])
    
    if "tx_metadata_mode" in yaml_config:
        mode_str = str(yaml_config["tx_metadata_mode"]).lower()
        pipeline_config_kwargs["tx_metadata_mode"] = TxMetadataMode(mode_str)

    if "embed_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["embed_agreement_batch_size"] = int(yaml_config["embed_agreement_batch_size"])

    if "embed_focus_section" in yaml_config:
        pipeline_config_kwargs["embed_focus_section"] = str(yaml_config["embed_focus_section"]).strip()

    if "embed_focus_section_batch_size" in yaml_config:
        pipeline_config_kwargs["embed_focus_section_batch_size"] = int(yaml_config["embed_focus_section_batch_size"])

    if "embed_target" in yaml_config:
        target_str = str(yaml_config["embed_target"]).lower()
        pipeline_config_kwargs["embed_target"] = EmbedTarget(target_str)
    
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
        "review_model": ReviewModel(),
        "tagging_model": TaggingModel(),
        "taxonomy_model": TaxonomyModel(),
        "pipeline_config": PipelineConfig(**pipeline_config_kwargs),  # type: ignore
    }
