import xml.etree.ElementTree as ET


def redact_agreement_xml(
    xml_content: str, *, focus_section_uuid: str | None, neighbor_sections: int
) -> str:
    if neighbor_sections < 0 or neighbor_sections > 5:
        raise ValueError("neighborSections must be between 0 and 5.")

    def _local_name(tag: str) -> str:
        if "}" in tag:
            return tag.rsplit("}", 1)[-1]
        return tag

    root = ET.fromstring(xml_content)
    sections: list[ET.Element] = [
        el for el in root.iter() if _local_name(el.tag).lower() == "section"
    ]
    ordered_uuids: list[str] = [
        el.attrib["uuid"] for el in sections if "uuid" in el.attrib
    ]

    if not focus_section_uuid:
        allowed: set[str] = set()
    else:
        try:
            focus_index = ordered_uuids.index(focus_section_uuid)
        except ValueError:
            allowed = set()
        else:
            lo = max(0, focus_index - neighbor_sections)
            hi = min(len(ordered_uuids) - 1, focus_index + neighbor_sections)
            allowed = set(ordered_uuids[lo : hi + 1])

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

