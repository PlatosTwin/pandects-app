"""Deterministic keys for run-scoped agreement batches."""

import hashlib
from collections.abc import Iterable


def agreement_batch_key(agreement_uuids: Iterable[str]) -> str:
    """
    Return a stable hash key for an unordered set of agreement UUIDs.

    Empty input is invalid because batch_key is only meaningful for non-empty batches.
    """
    normalized = sorted({str(u).strip() for u in agreement_uuids if str(u).strip()})
    if not normalized:
        raise ValueError("agreement_batch_key requires at least one agreement UUID.")
    raw = "|".join(normalized).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()
