"""Human-readable labels the REST responses carry next to raw codes.

Agreement rows store NAICS codes in ``target_industry``/``acquirer_industry`` and
sections store taxonomy ``standard_id``s. Clients used to download the whole
NAICS catalog and taxonomy tree just to print a label; the response schemas now
add ``<field>_label`` / ``standard_id_labels`` siblings from these cached maps.
"""

from __future__ import annotations

from typing import cast

from backend.extensions import db
from backend.models.main_db import NaicsSector, NaicsSubSector
from backend.services.taxonomy_service import cached_reference_value

INDUSTRY_FIELDS: tuple[str, ...] = ("target_industry", "acquirer_industry")


def naics_label_by_code() -> dict[str, str]:
    """Sector (2-digit) and subsector (3-digit) descriptions keyed by code."""

    def _build() -> dict[str, str]:
        label_by_code: dict[str, str] = {}
        for code, desc in cast(
            list[tuple[object, object]],
            db.session.query(NaicsSector.sector_code, NaicsSector.sector_desc).all(),
        ) + cast(
            list[tuple[object, object]],
            db.session.query(NaicsSubSector.sub_sector_code, NaicsSubSector.sub_sector_desc).all(),
        ):
            if code is not None and isinstance(desc, str) and desc:
                label_by_code[str(code)] = desc
        return label_by_code

    return cached_reference_value("naics_label_by_code", _build)


def naics_industry_label(value: object) -> str | None:
    """The NAICS description for an industry code, or null when it cannot be resolved.

    A code without its own entry (a 4-digit industry group, or a subsector missing
    from the catalog) rolls up to its 2-digit sector, as the web client always did.
    Non-numeric values are not codes and get no label.
    """
    if not isinstance(value, str):
        return None
    code = value.strip()
    if not code.isdigit():
        return None
    label_by_code = naics_label_by_code()
    label = label_by_code.get(code)
    if label is None and len(code) > 2:
        label = label_by_code.get(code[:2])
    return label


def with_industry_labels(payload: dict[str, object]) -> dict[str, object]:
    """Copy of ``payload`` with a ``<field>_label`` for each industry field it has."""
    if not any(field in payload for field in INDUSTRY_FIELDS):
        return payload
    labeled = dict(payload)
    for field in INDUSTRY_FIELDS:
        if field in payload:
            labeled[f"{field}_label"] = naics_industry_label(payload.get(field))
    return labeled


def industry_labels_for(values: list[str]) -> dict[str, str]:
    """``{code: label}`` for the resolvable codes in ``values``."""
    labels: dict[str, str] = {}
    for value in values:
        label = naics_industry_label(value)
        if label is not None:
            labels[value] = label
    return labels


__all__ = [
    "INDUSTRY_FIELDS",
    "industry_labels_for",
    "naics_industry_label",
    "naics_label_by_code",
    "with_industry_labels",
]
