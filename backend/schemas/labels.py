"""Response-schema mixins that add human-readable labels next to raw codes.

REST-only: the MCP tools build their own payloads (and already carry
``*_industry_label`` and a response-level ``standard_id_labels``), so labels are
added while REST responses are serialized rather than in the shared services.
"""

from __future__ import annotations

from typing import cast

from marshmallow import Schema, fields, pre_dump

_INDUSTRY_LABEL_DESCRIPTION = (
    "NAICS description of `{field}`. A code with no catalog entry of its own rolls up "
    "to its 2-digit sector; null when the value cannot be resolved."
)


def _labels_dict_field(description: str) -> fields.Dict:
    return fields.Dict(
        keys=fields.Str(),
        values=fields.Str(),
        metadata={"description": description},
    )


class IndustryLabelsSchema(Schema):
    target_industry_label = fields.Str(
        allow_none=True,
        metadata={"description": _INDUSTRY_LABEL_DESCRIPTION.format(field="target_industry")},
    )
    acquirer_industry_label = fields.Str(
        allow_none=True,
        metadata={"description": _INDUSTRY_LABEL_DESCRIPTION.format(field="acquirer_industry")},
    )

    @pre_dump
    def _add_industry_labels(self, data: object, **_kwargs: object) -> object:
        if not isinstance(data, dict):
            return data
        from backend.services.reference_labels import with_industry_labels

        return with_industry_labels(cast(dict[str, object], data))


class StandardIdLabelsSchema(Schema):
    """``standard_id_labels`` for a clause-taxonomy ``standard_id`` list."""

    standard_id_labels = _labels_dict_field(
        "Clause-taxonomy label for each id in `standard_id`, keyed by id."
    )

    @pre_dump
    def _add_standard_id_labels(self, data: object, **_kwargs: object) -> object:
        if not isinstance(data, dict):
            return data
        from backend.services.taxonomy_service import standard_id_labels

        payload = cast(dict[str, object], data)
        return {**payload, "standard_id_labels": standard_id_labels(payload.get("standard_id"))}


def _with_tax_labels(data: object, *, source_field: str, target_field: str) -> object:
    if not isinstance(data, dict):
        return data
    from backend.services.taxonomy_service import standard_id_labels

    payload = cast(dict[str, object], data)
    return {
        **payload,
        target_field: standard_id_labels(payload.get(source_field), kind="tax_clauses"),
    }


class TaxStandardIdLabelsSchema(Schema):
    """``tax_standard_id_labels`` for a tax-clause search result's ``tax_standard_ids``."""

    tax_standard_id_labels = _labels_dict_field(
        "Tax-clause-taxonomy label for each id in `tax_standard_ids`, keyed by id."
    )

    @pre_dump
    def _add_tax_standard_id_labels(self, data: object, **_kwargs: object) -> object:
        return _with_tax_labels(
            data, source_field="tax_standard_ids", target_field="tax_standard_id_labels"
        )


class TaxClauseStandardIdLabelsSchema(Schema):
    """``standard_id_labels`` for an agreement/section tax clause's ``standard_ids``."""

    standard_id_labels = _labels_dict_field(
        "Tax-clause-taxonomy label for each id in `standard_ids`, keyed by id."
    )

    @pre_dump
    def _add_standard_id_labels(self, data: object, **_kwargs: object) -> object:
        return _with_tax_labels(data, source_field="standard_ids", target_field="standard_id_labels")


__all__ = [
    "IndustryLabelsSchema",
    "StandardIdLabelsSchema",
    "TaxClauseStandardIdLabelsSchema",
    "TaxStandardIdLabelsSchema",
]
