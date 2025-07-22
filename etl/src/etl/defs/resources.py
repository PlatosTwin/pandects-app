import dagster as dg
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


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
            )
        }
    )
