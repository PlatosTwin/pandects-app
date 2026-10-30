# Parse with defusedxml to harden against billion-laughs / quadratic-blowup
# even though today's input is trusted (own DB). Serialize with stdlib ET —
# defusedxml does not provide a tostring/SubElement constructor.
import copy
import xml.etree.ElementTree as ET
from typing import Any

import defusedxml.ElementTree as DET


def _local_name(tag: str) -> str:
    if "}" in tag:
        return tag.rsplit("}", 1)[-1]
    return tag


def _allowed_section_uuids(
    root: ET.Element, *, focus_section_uuid: str | None, neighbor_sections: int
) -> set[str]:
    """The focus section and its neighbors in document order: the sections an
    anonymous caller may read."""
    if neighbor_sections < 0 or neighbor_sections > 5:
        raise ValueError("neighborSections must be between 0 and 5.")
    ordered_uuids: list[str] = [
        el.attrib["uuid"]
        for el in root.iter()
        if _local_name(el.tag).lower() == "section" and "uuid" in el.attrib
    ]
    if not focus_section_uuid:
        return set()
    try:
        focus_index = ordered_uuids.index(focus_section_uuid)
    except ValueError:
        return set()
    lo = max(0, focus_index - neighbor_sections)
    hi = min(len(ordered_uuids) - 1, focus_index + neighbor_sections)
    return set(ordered_uuids[lo : hi + 1])


def redact_agreement_xml(
    xml_content: str, *, focus_section_uuid: str | None, neighbor_sections: int
) -> str:
    root = DET.fromstring(xml_content)
    allowed = _allowed_section_uuids(
        root, focus_section_uuid=focus_section_uuid, neighbor_sections=neighbor_sections
    )

    def _is_section(el: ET.Element) -> bool:
        return _local_name(el.tag).lower() == "section"

    def _is_article(el: ET.Element) -> bool:
        return _local_name(el.tag).lower() == "article"

    def _section_is_allowed(section: ET.Element) -> bool:
        node_uuid = section.attrib.get("uuid")
        return isinstance(node_uuid, str) and node_uuid in allowed

    def _redact_section_body(section: ET.Element) -> None:
        section.text = None
        section.tail = None
        for child in list(section):
            section.remove(child)
        placeholder = ET.SubElement(section, "text")
        placeholder.set("redacted", "true")
        placeholder.text = "[REDACTED]"

    def _redact_article_preamble(article: ET.Element) -> None:
        had_text = bool(article.text and article.text.strip())
        article.text = None
        article.tail = None
        removed_any = False
        for child in list(article):
            if _is_section(child):
                child.tail = None
                continue
            article.remove(child)
            removed_any = True
        if removed_any or had_text:
            placeholder = ET.Element("text", {"redacted": "true"})
            placeholder.text = "[REDACTED]"
            article.insert(0, placeholder)

    def _walk(el: ET.Element, *, inside_allowed_section: bool) -> None:
        if _is_section(el):
            keep_contents = inside_allowed_section or _section_is_allowed(el)
            if not keep_contents:
                _redact_section_body(el)
                return
            for child in list(el):
                _walk(child, inside_allowed_section=True)
            return

        if inside_allowed_section:
            for child in list(el):
                _walk(child, inside_allowed_section=True)
            return

        if _is_article(el):
            if not inside_allowed_section:
                _redact_article_preamble(el)
            for child in list(el):
                _walk(child, inside_allowed_section=False)
            return

        for child in list(el):
            _walk(child, inside_allowed_section=False)

    _walk(root, inside_allowed_section=False)
    return ET.tostring(root, encoding="unicode")


def _hidden_containers(root: ET.Element, allowed: set[str]) -> set[str]:
    """Reader-index containers (section and article uuids) whose text
    `redact_agreement_xml` removes for the same caller: sections outside the
    allowed set (unless nested in an allowed one) and every article preamble."""
    hidden: set[str] = set()

    def _walk(el: ET.Element, *, inside_allowed_section: bool) -> None:
        name = _local_name(el.tag).lower()
        node_uuid = el.attrib.get("uuid")
        if name == "section":
            if inside_allowed_section or node_uuid in allowed:
                for child in el:
                    _walk(child, inside_allowed_section=True)
            else:
                # the whole body goes, nested sections included
                hidden.update(d.attrib["uuid"] for d in el.iter() if "uuid" in d.attrib)
            return
        if name == "article" and not inside_allowed_section and node_uuid:
            hidden.add(node_uuid)
        for child in el:
            _walk(child, inside_allowed_section=inside_allowed_section)

    _walk(root, inside_allowed_section=False)
    return hidden


def redact_reader_index(
    payload: dict[str, Any],
    xml_content: str,
    *,
    focus_section_uuid: str | None,
    neighbor_sections: int,
) -> dict[str, Any]:
    """The reader index an anonymous caller may see next to the XML that
    `redact_agreement_xml` gives them (docs/plans/smart-reader.md, Section 6.4).

    Term uses and cross-references inside hidden text are dropped. A declaration
    inside hidden text becomes ``{"redacted": true, "label": "Section 1.1"}``, so the
    reader can say where a term is defined without showing the definition.
    Subsection titles in hidden sections are blanked, since they quote the text."""
    root = DET.fromstring(xml_content)
    allowed = _allowed_section_uuids(
        root, focus_section_uuid=focus_section_uuid, neighbor_sections=neighbor_sections
    )
    hidden = _hidden_containers(root, allowed)
    out = copy.deepcopy(payload)

    def _is_hidden(anchor: object) -> bool:
        return isinstance(anchor, dict) and anchor.get("s") in hidden

    for term in out.get("terms", []):
        term["decls"] = [
            {"redacted": True, "label": d.get("in")} if _is_hidden(d.get("anchor")) else d
            for d in term.get("decls", [])
        ]
    out["uses"] = {
        key: row
        for key, row in out.get("uses", {}).items()
        if key.rsplit("/", 1)[0] not in hidden
    }
    out["xrefs"] = [x for x in out.get("xrefs", []) if not _is_hidden(x.get("at"))]

    def _blank_subs(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            if node.get("kind") == "sub" and _is_hidden(node.get("anchor")):
                node["title"] = ""
            _blank_subs(node.get("children", []))

    _blank_subs(out.get("outline", []))
    out["is_redacted"] = True
    return out
