"""Dagster resource definitions for the ETL pipeline.

This module defines the configurable resources used throughout the ETL pipeline,
including database connections, ML models, and pipeline configuration.
"""
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false

from enum import Enum

import dagster as dg
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


class PipelineConfig(dg.ConfigurableResource[object]):
    """Configuration for pipeline execution mode and batching behavior."""

    mode: PipelineMode = PipelineMode.FROM_SCRATCH
    scope: ProcessingScope = ProcessingScope.BATCHED
    tagging_agreement_batch_size: int = 500  # used in tagging_asset
    xml_agreement_batch_size: int = 10  # used in xml_asset
    taxonomy_agreement_batch_size: int = 50  # used in taxonomy_asset
    ai_repair_agreement_batch_size: int = 150  # used in ai_repair_enqueue_asset
    tx_metadata_agreement_batch_size: int = 10  # used in tx_metadata_asset
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


def get_resources() -> dict[str, object]:
    """Get the base resource configuration for the pipeline.

    Returns:
        Dictionary containing all resource definitions.
    """
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
        "pipeline_config": PipelineConfig(),
    }
