# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
import re
import xml.etree.ElementTree as ET
from typing import Dict, NotRequired, Protocol, TypedDict


_TAG_RE = re.compile(r"<[^>]+>")


class TaxonomyInput(TypedDict):
    article_title: str
    section_title: str
    section_text: str


class TaxonomyPrediction(TypedDict):
    label: str
    alt_probs: NotRequired[list[float]]


class SectionIndex(TypedDict):
    section_uuid: str
    agreement_uuid: str


class TaxonomyPredictor(Protocol):
    def predict(self, rows: list[TaxonomyInput]) -> list[TaxonomyPrediction]: ...


class LoggerProtocol(Protocol):
    def info(self, msg: str) -> None: ...


class ContextProtocol(Protocol):
    log: LoggerProtocol


class TaxonomyRow(TypedDict):
    section_uuid: str
    agreement_uuid: str
    article_title: str | None
    section_title: str | None
    xml_content: str


def strip_xml_tags_to_text(xml_fragment: str) -> str:
    """Collapse an XML/HTML fragment to plain text.

    Replaces tags with spaces and collapses whitespace.
    """
    if not xml_fragment:
        return ""
    no_tags = _TAG_RE.sub(" ", xml_fragment)
    collapsed = re.sub(r"\s+", " ", no_tags)
    return collapsed.strip()


def predict_taxonomy(
    rows: list[TaxonomyRow],
    model: TaxonomyPredictor,
    context: ContextProtocol,
) -> tuple[list[SectionIndex], list[TaxonomyPrediction]]:
    """Prepare inputs and run taxonomy prediction for a set of sections."""
    inputs: list[TaxonomyInput] = []
    sec_idx: list[SectionIndex] = []
    for r in rows:
        text_block = strip_xml_tags_to_text(r["xml_content"])
        inputs.append(
            {
                "article_title": r.get("article_title") or "",
                "section_title": r.get("section_title") or "",
                "section_text": text_block,
            }
        )
        sec_idx.append(
            {
                "section_uuid": r["section_uuid"],
                "agreement_uuid": r["agreement_uuid"],
            }
        )

    context.log.info(f"Running taxonomy prediction on {len(inputs)} sections")
    preds = model.predict(inputs)
    return sec_idx, preds


def apply_standard_ids_to_xml(xml_str: str, section_uuid_to_label: Dict[str, str]) -> str:
    """Set standardId on <section> elements matching provided UUIDs.

    Args:
        xml_str: Full agreement XML document string.
        section_uuid_to_label: Mapping of section UUID -> standardId label.

    Returns:
        Updated XML string.
    """
    if not section_uuid_to_label:
        return xml_str

    root = ET.fromstring(xml_str)
    for elem in root.iter("section"):
        su = elem.get("uuid")
        if su and su in section_uuid_to_label:
            elem.set("standardId", section_uuid_to_label[su])
    return ET.tostring(root, encoding="unicode")
