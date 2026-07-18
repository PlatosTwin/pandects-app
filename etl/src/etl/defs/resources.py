"""Dagster resource definitions for the ETL pipeline.

This module defines the configurable resources used throughout the ETL pipeline,
including database connections, ML models, and pipeline configuration.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

from enum import Enum
from pathlib import Path
from types import UnionType
from typing import Any, Union, get_args, get_origin

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
from etl.models.taxonomy.ml.taxonomy import TaxonomyInference
from etl.models.taxonomy.ml.taxonomy_constants import (
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

    LLM = "llm"
    ML = "ml"
    GOLD_BACKFILL = "gold_backfill"


class EmbedTarget(Enum):
    """Section embedding target: agreement batch or specific section-standard-id batch."""

    AGREEMENT = "agreement"
    SECTION = "section"


class AIRepairAttemptPriority(Enum):
    """Ordering for AI-repair agreement selection."""

    NOT_ATTEMPTED_FIRST = "not_attempted_first"
    ATTEMPTED_FIRST = "attempted_first"


class PipelineConfig(dg.ConfigurableResource[object]):
    """Configuration for pre-processing mode and batching behavior."""

    pre_processing_mode: PreProcessingMode = PreProcessingMode.FROM_SCRATCH
    queue_run_mode: QueueRunMode = QueueRunMode.SINGLE_BATCH
    refresh: bool = True  # run end-of-job gating + summary refresh for terminal assets
    resume_openai_batches: bool = True  # resume matching in-flight OpenAI batches when possible
    xml_enable_llm_verification: bool = False  # gate XML LLM verification route; hard rules still run when disabled
    resume_logical_runs: bool = True  # resume unfinished logical runs for managed ingest jobs
    force_new_logical_run: bool = False  # bypass unfinished logical run and start a fresh managed ingest run
    tagging_agreement_batch_size: int = 500  # used in tagging_asset
    pre_processing_agreement_batch_size: int = 5  # used in pre_processing_asset
    xml_agreement_batch_size: int = 10  # used across XML + AI-repair cycle assets
    ai_repair_page_budget: int = 0  # when > 0, ai_repair_enqueue selects by page budget instead of agreement count
    ai_repair_attempt_priority: AIRepairAttemptPriority = AIRepairAttemptPriority.NOT_ATTEMPTED_FIRST
    taxonomy_agreement_batch_size: int = 50  # used in taxonomy_asset
    enable_section_taxonomy: bool = True  # managed ingest: run section taxonomy LLM stage when true
    taxonomy_mode: TaxonomyMode = TaxonomyMode.LLM  # llm | ml | gold_backfill
    taxonomy_section_title_regex: str | None = None  # optional REGEXP filter for taxonomy prediction modes
    taxonomy_llm_model: str = "gpt-5.4-mini"  # used in taxonomy_asset llm mode
    taxonomy_llm_sections_per_request: int = 5  # sections bundled into each LLM request within a batch
    tax_module_agreement_batch_size: int = 25  # used in tax_module assets
    enable_tax_taxonomy: bool = False  # managed ingest: run tax taxonomy stage when true
    tax_module_llm_model: str = "gpt-5.4-mini"  # used in tax_module_asset
    tax_module_llm_clauses_per_request: int = 5  # clauses bundled into each LLM request within a batch
    tx_metadata_agreement_batch_size: int = 10  # used in tx_metadata_asset
    tx_metadata_mode: TxMetadataMode = TxMetadataMode.OFFLINE  # offline | web_search
    tx_metadata_web_search_verify_recent_pending: bool = False  # regular_ingest: end-of-job web-search sweep over recent pending deals (expensive)
    embed_agreement_batch_size: int = 10  # used in 11_embed_sections when embed_target=agreement
    embed_focus_section: str = ""  # section_standard_id value when embed_target=section
    embed_focus_section_batch_size: int = 100  # used in 11_embed_sections when embed_target=section
    embed_target: EmbedTarget = EmbedTarget.SECTION  # agreement | section
    staging_target_date: str = ""  # yyyy-mm-dd: staging asset ingests through and including this date
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

    _inf: NERInference | None = PrivateAttr(default=None)

    def model(self) -> NERInference:
        """Load and return the NER tagging model."""
        if self._inf is None:
            self._inf = NERInference(
                ckpt_path=NER_CKPT_PATH, label_list=NER_LABEL_LIST, review_threshold=0.80
            )
        return self._inf


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


def _coerce_pipeline_config_value(*, field_name: str, annotation: Any, raw: Any) -> Any:
    """Coerce a YAML value to the type declared on the PipelineConfig field."""
    origin = get_origin(annotation)
    if origin is Union or origin is UnionType:
        non_none_args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if non_none_args == [str]:
            return None if raw is None else str(raw).strip() or None
        raise ValueError(
            f"PipelineConfig field {field_name!r} has unsupported union type {annotation!r}; extend _coerce_pipeline_config_value."
        )
    if annotation is bool:
        return _parse_bool(raw, field_name=field_name)
    if annotation is int:
        return int(raw)
    if annotation is float:
        return float(raw)
    if annotation is str:
        return str(raw).strip()
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation(str(raw).lower())
    raise ValueError(
        f"PipelineConfig field {field_name!r} has unsupported type {annotation!r}; extend _coerce_pipeline_config_value."
    )


def get_resources() -> dict[str, object]:
    """Get the base resource configuration for the pipeline.

    Loads defaults from etl/configs/pipeline_config.yaml if available.
    Values can still be overridden via Dagster's -c flag at runtime.

    Returns:
        Dictionary containing all resource definitions.
    """
    yaml_config = _load_yaml_config()

    # The allowlist and per-field parsing both derive from PipelineConfig's own
    # field definitions, so adding a field to the class is the whole change.
    pipeline_config_fields = PipelineConfig.model_fields
    unknown_keys = sorted(set(yaml_config) - set(pipeline_config_fields))
    if unknown_keys:
        raise ValueError(
            "Unknown keys in resources.pipeline_config.config: "
            + ", ".join(unknown_keys)
        )

    pipeline_config_kwargs: dict[str, Any] = {
        key: _coerce_pipeline_config_value(
            field_name=key,
            annotation=pipeline_config_fields[key].annotation,
            raw=raw,
        )
        for key, raw in yaml_config.items()
    }

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
