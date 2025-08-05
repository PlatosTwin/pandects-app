import dagster as dg
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from etl.models.code.classifier import ClassifierInference
from etl.models.code.ner import NERInference
from pathlib import Path
from etl.models.code.constants import (
    NER_LABEL_LIST,
    NER_CKPT_PATH,
    CLASSIFIER_CKPT_PATH,
)


class DBResource(dg.ConfigurableResource):
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

    def model(self) -> ClassifierInference:
        """Load and return the PageClassifier model on the selected device."""
        model = ClassifierInference(ckpt_path=CLASSIFIER_CKPT_PATH)
        return model


class TaggingModel(dg.ConfigurableResource):

    def model(self) -> NERInference:
        """Load and return the PageClassifier model on the selected device."""
        model = NERInference(ckpt_path=NER_CKPT_PATH, label_list=NER_LABEL_LIST)
        return model


@dg.definitions
def resources() -> dg.Definitions:
    """
    Return Dagster Definitions for resources.
    """
    return dg.Definitions(
        resources={
            "db": DBResource(
                user=dg.EnvVar("MARIADB_USER"),
                password=dg.EnvVar("MARIADB_PASSWORD"),
                host=dg.EnvVar("MARIADB_HOST"),
                port=dg.EnvVar("MARIADB_PORT"),
                database=dg.EnvVar("MARIADB_DATABASE"),
            ),
            "classifier_model": ClassifierModel(),
            "tagging_model": TaggingModel(),
        }
    )
