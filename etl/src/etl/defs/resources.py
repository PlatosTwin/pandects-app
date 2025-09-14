"""Dagster resource definitions for the ETL pipeline.

This module defines the configurable resources used throughout the ETL pipeline,
including database connections, ML models, and pipeline configuration.
"""

import dagster as dg
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from etl.models.code.classifier import ClassifierInference
from etl.models.code.ner import NERInference
from pathlib import Path
from etl.models.code.shared_constants import (
    NER_LABEL_LIST,
    NER_CKPT_PATH,
    CLASSIFIER_CKPT_PATH,
    CLASSIFIER_XGB_PATH,
)
from enum import Enum
from typing import Dict, Any, Optional
from pydantic import PrivateAttr


class PipelineMode(Enum):
    """Pipeline execution modes."""
    FROM_SCRATCH = "from_scratch"
    CLEANUP = "cleanup"


class PipelineConfig(dg.ConfigurableResource):
    """Configuration for pipeline execution mode."""
    
    mode: PipelineMode = PipelineMode.CLEANUP

    def is_cleanup_mode(self) -> bool:
        """Check if the pipeline is running in cleanup mode."""
        return self.mode == PipelineMode.CLEANUP


class DBResource(dg.ConfigurableResource):
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


class ClassifierModel(dg.ConfigurableResource):
    """Resource for the page classification model."""

    _inf: Optional[ClassifierInference] = PrivateAttr(default=None)

    def model(self):
        if self._inf is None:
            self._inf = ClassifierInference(
                ckpt_path=CLASSIFIER_CKPT_PATH,
                xgb_path=CLASSIFIER_XGB_PATH,
                num_workers=7
            )
        return self._inf


class TaggingModel(dg.ConfigurableResource):
    """Resource for the NER tagging model."""

    def model(self) -> NERInference:
        """Load and return the NER tagging model."""
        model = NERInference(ckpt_path=NER_CKPT_PATH, label_list=NER_LABEL_LIST)
        return model


def get_resources() -> Dict[str, Any]:
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
        "pipeline_config": PipelineConfig(),
    }
