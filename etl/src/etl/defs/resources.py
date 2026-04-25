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
    taxonomy_mode: TaxonomyMode = TaxonomyMode.LLM  # llm | ml | gold_backfill
    taxonomy_section_title_regex: str | None = None  # optional REGEXP filter for taxonomy prediction modes
    taxonomy_llm_model: str = "gpt-5.4-mini"  # used in taxonomy_asset llm mode
    taxonomy_llm_sections_per_request: int = 5  # sections bundled into each LLM request within a batch
    tax_module_agreement_batch_size: int = 25  # used in tax_module assets
    tax_module_llm_model: str = "gpt-5.4-mini"  # used in tax_module_asset
    tax_module_llm_clauses_per_request: int = 5  # clauses bundled into each LLM request within a batch
    tx_metadata_agreement_batch_size: int = 10  # used in tx_metadata_asset
    tx_metadata_mode: TxMetadataMode = TxMetadataMode.OFFLINE  # offline | web_search
    embed_agreement_batch_size: int = 10  # used in 11_embed_sections when embed_target=agreement
    embed_focus_section: str = ""  # section_standard_id value when embed_target=section
    embed_focus_section_batch_size: int = 100  # used in 11_embed_sections when embed_target=section
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
        "xml_enable_llm_verification",
        "resume_logical_runs",
        "force_new_logical_run",
        "tagging_agreement_batch_size",
        "pre_processing_agreement_batch_size",
        "xml_agreement_batch_size",
        "ai_repair_page_budget",
        "ai_repair_attempt_priority",
        "taxonomy_agreement_batch_size",
        "taxonomy_mode",
        "taxonomy_section_title_regex",
        "taxonomy_llm_model",
        "taxonomy_llm_sections_per_request",
        "tax_module_agreement_batch_size",
        "tax_module_llm_model",
        "tax_module_llm_clauses_per_request",
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

    if "xml_enable_llm_verification" in yaml_config:
        pipeline_config_kwargs["xml_enable_llm_verification"] = _parse_bool(
            yaml_config["xml_enable_llm_verification"],
            field_name="xml_enable_llm_verification",
        )

    if "resume_logical_runs" in yaml_config:
        pipeline_config_kwargs["resume_logical_runs"] = _parse_bool(
            yaml_config["resume_logical_runs"],
            field_name="resume_logical_runs",
        )

    if "force_new_logical_run" in yaml_config:
        pipeline_config_kwargs["force_new_logical_run"] = _parse_bool(
            yaml_config["force_new_logical_run"],
            field_name="force_new_logical_run",
        )
    
    if "tagging_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["tagging_agreement_batch_size"] = int(yaml_config["tagging_agreement_batch_size"])

    if "pre_processing_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["pre_processing_agreement_batch_size"] = int(yaml_config["pre_processing_agreement_batch_size"])
    
    if "xml_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["xml_agreement_batch_size"] = int(yaml_config["xml_agreement_batch_size"])

    if "ai_repair_page_budget" in yaml_config:
        pipeline_config_kwargs["ai_repair_page_budget"] = int(yaml_config["ai_repair_page_budget"])

    if "ai_repair_attempt_priority" in yaml_config:
        priority_str = str(yaml_config["ai_repair_attempt_priority"]).lower()
        pipeline_config_kwargs["ai_repair_attempt_priority"] = AIRepairAttemptPriority(priority_str)

    if "taxonomy_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["taxonomy_agreement_batch_size"] = int(yaml_config["taxonomy_agreement_batch_size"])

    if "taxonomy_mode" in yaml_config:
        mode_str = str(yaml_config["taxonomy_mode"]).lower()
        pipeline_config_kwargs["taxonomy_mode"] = TaxonomyMode(mode_str)

    if "taxonomy_section_title_regex" in yaml_config:
        raw_regex = yaml_config["taxonomy_section_title_regex"]
        pipeline_config_kwargs["taxonomy_section_title_regex"] = (
            None if raw_regex is None else str(raw_regex).strip() or None
        )

    if "taxonomy_llm_model" in yaml_config:
        pipeline_config_kwargs["taxonomy_llm_model"] = str(
            yaml_config["taxonomy_llm_model"]
        ).strip()

    if "taxonomy_llm_sections_per_request" in yaml_config:
        pipeline_config_kwargs["taxonomy_llm_sections_per_request"] = int(
            yaml_config["taxonomy_llm_sections_per_request"]
        )

    if "tax_module_agreement_batch_size" in yaml_config:
        pipeline_config_kwargs["tax_module_agreement_batch_size"] = int(
            yaml_config["tax_module_agreement_batch_size"]
        )

    if "tax_module_llm_model" in yaml_config:
        pipeline_config_kwargs["tax_module_llm_model"] = str(
            yaml_config["tax_module_llm_model"]
        ).strip()

    if "tax_module_llm_clauses_per_request" in yaml_config:
        pipeline_config_kwargs["tax_module_llm_clauses_per_request"] = int(
            yaml_config["tax_module_llm_clauses_per_request"]
        )
    
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
