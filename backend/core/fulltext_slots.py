"""A server-wide cap on concurrent FULLTEXT searches.

Each InnoDB FULLTEXT query may hold up to ``innodb_ft_result_cache_limit``
(128MB in production) while it runs, and a common phrase ("Material Adverse
Effect means") fills it. Production MariaDB has a 768MB VM with a 192MB buffer
pool, so a few of these at once got mariadbd OOM-killed (2026-10-21 17:07).
Every API machine, worker and thread can start one, and the server keeps
running a search after the browser gives up on it, so an in-process limit
cannot bound them.

``fulltext_slot()`` takes one of ``FULLTEXT_SEARCH_SLOTS`` MariaDB named locks
(``GET_LOCK``) on a dedicated connection and holds it for the whole text search.
A named lock belongs to its connection, so it is released when the block exits
or, if the process dies, when the server drops the connection. Other databases
(SQLite in tests) have no named locks and are not limited.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from flask import current_app, has_app_context
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from werkzeug.exceptions import ServiceUnavailable

FULLTEXT_SEARCH_SLOTS = max(1, int(os.environ.get("FULLTEXT_SEARCH_SLOTS", "2")))
# How long a search waits for a slot before answering 503. A slot is held for the
# whole request; its statements are each capped (2s, 4s and 20s budgets, so up to
# about 46s for an agreement search), so this bounds the queue, not the search.
FULLTEXT_SLOT_WAIT_SECONDS = max(0, int(os.environ.get("FULLTEXT_SLOT_WAIT_SECONDS", "8")))
_SLOT_POLL_SECONDS = 0.25
FULLTEXT_BUSY_MESSAGE = (
    "Keyword search is busy right now. Please try again in a few seconds."
)


class FulltextSearchBusy(ServiceUnavailable):
    """503: every FULLTEXT slot stayed taken for the whole wait."""

    description = FULLTEXT_BUSY_MESSAGE
    # MCP passes the message through and tells the agent to retry.
    retryable = True

    def get_headers(self, environ: Any = None, scope: Any = None) -> list[tuple[str, str]]:
        return [*super().get_headers(environ, scope), ("Retry-After", "5")]


def _slot_names(schema: str) -> list[str]:
    # Named locks are server-wide; the schema keeps two databases on one server apart.
    return [f"pandects:{schema}:fulltext:{index}" for index in range(FULLTEXT_SEARCH_SLOTS)]


def _try_any_slot(conn: Any, names: list[str]) -> str | None:
    # CASE evaluates its branches in order and stops at the first lock it gets,
    # so this is one round trip that takes at most one slot.
    branches = " ".join(
        f"WHEN GET_LOCK(:n{index}, 0) = 1 THEN :n{index}" for index in range(len(names))
    )
    params = {f"n{index}": name for index, name in enumerate(names)}
    acquired = conn.execute(text(f"SELECT CASE {branches} END"), params).scalar()
    return acquired if isinstance(acquired, str) else None


@contextmanager
def fulltext_slot(engine: Any) -> Iterator[None]:
    """Hold one server-wide FULLTEXT slot for the duration of the block."""
    if engine.dialect.name not in {"mysql", "mariadb"}:
        yield
        return
    schema = current_app.config.get("MAIN_DB_SCHEMA") if has_app_context() else None
    names = _slot_names(schema.strip() if isinstance(schema, str) and schema.strip() else "default")
    with engine.connect() as conn:
        # Poll every slot rather than block on one: a blocking GET_LOCK on a single
        # name would sit out its whole wait while another slot was already free.
        deadline = time.monotonic() + FULLTEXT_SLOT_WAIT_SECONDS
        acquired = _try_any_slot(conn, names)
        while acquired is None and time.monotonic() < deadline:
            time.sleep(_SLOT_POLL_SECONDS)
            acquired = _try_any_slot(conn, names)
        if acquired is None:
            raise FulltextSearchBusy()
        try:
            yield
        finally:
            try:
                _ = conn.execute(text("SELECT RELEASE_LOCK(:n)"), {"n": acquired})
            except SQLAlchemyError:
                # The lock may still be held on this connection; never hand it back
                # to the pool, or the slot stays taken until the connection recycles.
                conn.invalidate()


__all__ = [
    "FULLTEXT_BUSY_MESSAGE",
    "FULLTEXT_SEARCH_SLOTS",
    "FulltextSearchBusy",
    "fulltext_slot",
]
