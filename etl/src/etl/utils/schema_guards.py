# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import bindparam, text
from sqlalchemy.engine import Connection


def assert_tables_exist(conn: Connection, *, schema: str, table_names: Sequence[str]) -> None:
    """Fail fast when required runtime tables are missing.

    Schema objects are expected to be managed by migrations, not created by assets.
    """
    unique_names = sorted({name for name in table_names if name})
    if not unique_names:
        return

    query = text(
        """
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = :schema
          AND table_name IN :table_names
        """
    ).bindparams(bindparam("table_names", expanding=True))

    existing = set(
        conn.execute(
            query,
            {
                "schema": schema,
                "table_names": unique_names,
            },
        ).scalars().all()
    )

    missing = [name for name in unique_names if name not in existing]
    if missing:
        missing_csv = ", ".join(missing)
        raise RuntimeError(
            f"Missing required table(s) in schema '{schema}': {missing_csv}. "
            + "Create these tables via migrations before running this asset."
        )
