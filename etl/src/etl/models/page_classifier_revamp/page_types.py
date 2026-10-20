"""Page labels and document types shared by the CRF pipeline and its feature sources."""

from __future__ import annotations

from dataclasses import dataclass

PAGE_LABELS = ["front_matter", "toc", "body", "sig", "back_matter"]

FeatureValue = float | bool
FeatureDict = dict[str, FeatureValue]


@dataclass(frozen=True)
class AgreementDocument:
    agreement_uuid: str
    page_texts: list[str]
    labels: list[str]
