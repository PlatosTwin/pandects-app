import dagster as dg
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
import torch
from etl.src.etl.domain.classifier_model import PageClassifier
from pathlib import Path

ckpt = ""
ckpt_path = (
    Path("/Users/nikitabogdanov/PycharmProjects/merger_agreements/scripts/models")
    / ckpt
)


class DBResource(dg.ConfigurableResource):
    host: str
    port: str
    user: str
    password: str
    database: str

    def get_engine(self) -> Engine:
        url = (
            f"mariadb+mysqldb://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )
        return create_engine(url)


class ClassifierModel(dg.ConfigurableResource):

    def __init__(self):
        if torch.backends.mps.is_available():
            self.DEVICE = "mps"
        elif torch.cuda.is_available():
            self.DEVICE = "cuda"
        else:
            self.DEVICE = "cpu"

    def model(self) -> PageClassifier:
        model = PageClassifier.load_from_checkpoint(
            ckpt_path,
        )
        model.eval()
        model.to(self.DEVICE)

        return model
    
    
class TaggingModel(dg.ConfigurableResource):

    def __init__(self):
        if torch.backends.mps.is_available():
            self.DEVICE = "mps"
        elif torch.cuda.is_available():
            self.DEVICE = "cuda"
        else:
            self.DEVICE = "cpu"

    def model(self) -> PageClassifier:
        model = PageClassifier.load_from_checkpoint(
            ckpt_path,
        )
        model.eval()
        model.to(self.DEVICE)

        return model


@dg.definitions
def resources() -> dg.Definitions:
    return dg.Definitions(
        resources={
            "db": DBResource(
                user=dg.EnvVar("MARIADB_USER"),
                password=dg.EnvVar("MARIADB_PASSWORD"),
                host=dg.EnvVar("MARIADB_HOST"),
                port=dg.EnvVar("MARIADB_PORT"),
                database=dg.EnvVar("MARIADB_DATABASE"),
            ),
            "classified_model": ClassifierModel(),
        }
    )
