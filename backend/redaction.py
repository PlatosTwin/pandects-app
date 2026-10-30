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


# Bump when the reader-index redaction rules change, so clients holding an older
# anonymous payload refetch it instead of revalidating it with a 304.
READER_INDEX_REDACTION_VERSION = 2


def _visible_text_anchors(root: ET.Element, allowed: set[str]) -> set[str]:
    """``"<container>/<t>"`` keys of the `<text>` elements that survive
    `redact_agreement_xml` for the same caller.

    Containers and ordinals follow the enricher's walk exactly
    (etl/src/etl/domain/reader_index.py, `_collect_texts`): a section or article with
    a uuid opens a container, anything else keeps its parent's, and each container
    counts its `<text>` elements in document order. Visibility follows
    `redact_agreement_xml`: a section's body survives only inside an allowed section,
    an article keeps nothing but its sections, and region text outside articles
    stays. Callers keep only anchors found here, so a shape either walk misreads is
    hidden rather than shown."""
    visible: set[str] = set()
    counters: dict[str, int] = {}

    def _walk(el: ET.Element, container: str, mode: str) -> None:
        # mode: "open" (outside articles), "allowed" (inside an allowed section),
        # "article" (an article's own content, outside its sections), "hidden"
        for child in el:
            if child.tag == "text":
                t = counters.get(container, 0)
                counters[container] = t + 1
                if mode in ("open", "allowed"):
                    visible.add(f"{container}/{t}")
                continue
            uuid = child.attrib.get("uuid")
            name = _local_name(child.tag).lower()
            if mode == "allowed" or mode == "hidden":
                child_mode = mode
            elif name == "section":
                child_mode = "allowed" if uuid in allowed else "hidden"
            elif mode == "article":
                # an article's non-section children are removed whole
                child_mode = "hidden"
            elif name == "article":
                child_mode = "article"
            else:
                child_mode = "open"
            child_container = uuid if child.tag in ("section", "article") and uuid else container
            _walk(child, child_container, child_mode)

    for region in root:
        if region.tag == "metadata":
            continue
        _walk(region, region.tag, "open")
    return visible


def redact_reader_index(
    payload: dict[str, Any],
    xml_content: str,
    *,
    focus_section_uuid: str | None,
    neighbor_sections: int,
) -> dict[str, Any]:
    """The reader index an anonymous caller may see next to the XML that
    `redact_agreement_xml` gives them (docs/plans/smart-reader.md, Section 6.4).

    Term uses and cross-references outside the visible text are dropped. A
    declaration outside it becomes ``{"redacted": true, "label": "Section 1.1"}``, so
    the reader can say where a term is defined without showing the definition.
    Outline titles that quote hidden text (subsections, attachment headings) are
    blanked; section and article titles stay, as they do in the redacted XML."""
    root = DET.fromstring(xml_content)
    allowed = _allowed_section_uuids(
        root, focus_section_uuid=focus_section_uuid, neighbor_sections=neighbor_sections
    )
    visible = _visible_text_anchors(root, allowed)
    out = copy.deepcopy(payload)

    def _is_visible(anchor: object) -> bool:
        return isinstance(anchor, dict) and f"{anchor.get('s')}/{anchor.get('t')}" in visible

    for term in out.get("terms", []):
        term["decls"] = [
            d if _is_visible(d.get("anchor")) else {"redacted": True, "label": d.get("in")}
            for d in term.get("decls", [])
        ]
    out["uses"] = {key: row for key, row in out.get("uses", {}).items() if key in visible}
    out["xrefs"] = [x for x in out.get("xrefs", []) if _is_visible(x.get("at"))]

    def _blank_titles(nodes: list[dict[str, Any]]) -> None:
        for node in nodes:
            anchor = node.get("anchor")
            quotes_text = isinstance(anchor, dict) and "t" in anchor
            if quotes_text and not _is_visible(anchor):
                node["title"] = ""
            _blank_titles(node.get("children", []))

    _blank_titles(out.get("outline", []))
    out["is_redacted"] = True
    return out
