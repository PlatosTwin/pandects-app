import re
import xml.etree.ElementTree as ET
from typing import Dict


_TAG_RE = re.compile(r"<[^>]+>")


def strip_xml_tags_to_text(xml_fragment: str) -> str:
    """Collapse an XML/HTML fragment to plain text.

    Replaces tags with spaces and collapses whitespace.
    """
    if not xml_fragment:
        return ""
    no_tags = _TAG_RE.sub(" ", xml_fragment)
    collapsed = re.sub(r"\s+", " ", no_tags)
    return collapsed.strip()


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


