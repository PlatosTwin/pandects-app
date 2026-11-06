# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
from __future__ import annotations

import re
import unicodedata
import uuid
from typing import Any, Dict, List
import xml.etree.ElementTree as ET


_NUM_WORDS: List[str] = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
]
_NUM_PATTERN = "|".join(_NUM_WORDS)
_WHITESPACE_RE = re.compile(r"\s+")
_NON_CONTENT_TAGS = frozenset({"page", "pageuuid"})
# A printed definitions block before Article 1 gets an article-level row once it
# holds this many definition paragraphs; a few inline terms in a preamble do not.
_PRE_ARTICLE_MIN_DEFINITIONS = 10
_PRE_ARTICLE_DEFINITION_TEXT_RE = re.compile(
    r"^[\u201C\u201D\"][^\"\u201C\u201D]{1,120}[\u201C\u201D\"]\s*[,:]?\s*"
    r"(?:means?|shall\s+mean|shall\s+have|has\s+the|have\s+the|is\s+defined|are\s+defined)\b",
    re.IGNORECASE,
)
# The lines printed right above the first definition: "DEFINITIONS", "Definitions: For all
# purposes of this Agreement, ...", "... the following terms shall have ...".
_PRE_ARTICLE_LEAD_IN_RE = re.compile(
    r"\bdefinitions?\b|\bdefined\s+terms\b|\bterms\s+defined\b|\bfollowing\s+(?:\w+\s+){0,3}(?:terms|meanings?)\b",
    re.IGNORECASE,
)
# A heading printed as a run-in label: "Definitions: For all purposes of this Agreement, ...".
_PRE_ARTICLE_RUN_IN_HEADING_RE = re.compile(r"^(?:certain\s+)?definitions?(?=\s*:)", re.IGNORECASE)
_PRE_ARTICLE_NOT_LEAD_IN_RE = re.compile(r"^\s*(?:WHEREAS|NOW,?\s+THEREFORE|THIS\b)", re.IGNORECASE)
_PRE_ARTICLE_MAX_LEAD_INS = 2
_PRE_ARTICLE_HEADING_MAX_WORDS = 5
# The agreement's own opening, which some filings print after the definitions. The XML nests
# it in the last definition: preamble, recitals and the consideration clause.
_PRE_ARTICLE_AGREEMENT_OPENING_RE = re.compile(
    r"^(?:WHEREAS\b|RECITALS?\b|BACKGROUND\s*:?$|NOW,?\s+THEREFORE\b|THIS\s"
    r"|This\s[^.]{0,80}\bAgreement\b.{0,120}\b(?:made|entered|executed|dated|by\s+and|between|among)\b)"
)


def _local_tag_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].casefold()


def _visible_text_chunks(element: ET.Element) -> List[str]:
    if _local_tag_name(element.tag) in _NON_CONTENT_TAGS:
        return []

    chunks: List[str] = []
    if element.text:
        chunks.append(element.text)
    for child in element:
        chunks.extend(_visible_text_chunks(child))
        if child.tail:
            chunks.append(child.tail)
    return chunks


def section_plain_text_from_xml(xml_fragment: str) -> str:
    """Convert a stored section XML fragment to canonical searchable text.

    Section fragments commonly contain several adjacent ``<text>`` elements,
    so chunks are separated before whitespace is collapsed. Page numbers and
    internal page UUID markers are structural metadata rather than agreement
    language and are deliberately omitted.
    """
    if not xml_fragment:
        return ""

    try:
        root = ET.fromstring(f"<section-search-root>{xml_fragment}</section-search-root>")
    except ET.ParseError as exc:
        raise ValueError("Stored section XML is not well-formed.") from exc

    chunks = [chunk for chunk in _visible_text_chunks(root) if chunk.strip()]
    plain_text = " ".join(chunks)
    return _WHITESPACE_RE.sub(" ", unicodedata.normalize("NFC", plain_text)).strip()


def normalize_section_search_text(plain_text: str) -> str:
    """Return the stable case-insensitive representation used for indexing."""
    normalized = unicodedata.normalize("NFKC", plain_text).casefold()
    return _WHITESPACE_RE.sub(" ", normalized).strip()


def build_section_search_text(xml_fragment: str) -> tuple[str, str]:
    plain_text = section_plain_text_from_xml(xml_fragment)
    return plain_text, normalize_section_search_text(plain_text)


def clean_article_title(title: str) -> str:
    pattern = rf"(?i)^article\s+(?:[IVXLCDM]+|\d+|{_NUM_PATTERN})\s*"
    t = re.sub(pattern, "", title)
    t = t.lower().strip()
    # Drop trailing periods/spaces
    t = re.sub(r"[\s\.]+$", "", t)
    # Remove punctuation except hyphen, parentheses, and slash
    t = re.sub(r"[^\w\s\-()/]", "", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def clean_section_title(title: str) -> str:
    # Remove leading "Section <digits[.digits]>." prefix
    t = re.sub(r"(?i)^\s*section\s+[0-9]+(?:\.[0-9]+)*(?:\.)?\s*", "", title)
    t = t.lower().strip()
    # Drop trailing periods/spaces
    t = re.sub(r"[\s\.]+$", "", t)
    # Remove punctuation except hyphen, parentheses, and slash
    t = re.sub(r"[^\w\s\-()/]", "", t)
    # Remove any leading standalone digits left after cleaning
    t = re.sub(r"^\d+\s*", "", t)
    # Collapse whitespace
    t = re.sub(r"\s+", " ", t).strip()
    return t


def get_inner_xml(elem: ET.Element) -> str:
    parts: List[str] = []
    if elem.text:
        parts.append(elem.text)
    for child in elem:
        parts.append(ET.tostring(child, encoding="unicode"))
    return "".join(parts)


def get_article_level_xml_excluding_sections(article: ET.Element) -> str:
    parts: List[str] = []
    if article.text:
        parts.append(article.text)
    for child in article:
        if child.tag != "section":
            parts.append(ET.tostring(child, encoding="unicode"))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


def has_meaningful_xml_text(xml_fragment: str) -> bool:
    wrapped = ET.fromstring(f"<root>{xml_fragment}</root>")
    return any(text.strip() for text in wrapped.itertext())


def _element_text(element: ET.Element) -> str:
    return _WHITESPACE_RE.sub(" ", " ".join(_visible_text_chunks(element))).strip()


def _is_agreement_opening(element: ET.Element) -> bool:
    return element.tag == "text" and _PRE_ARTICLE_AGREEMENT_OPENING_RE.match(_element_text(element)) is not None


def _is_stray_caps_line(element: ET.Element) -> bool:
    """An all-caps line with no quoted term: an agreement title or "THE MERGER", never a definition."""
    text = _element_text(element)
    return element.tag == "text" and text == text.upper() and not re.search(r"[\"\u201C\u201D]", text)


def _without_agreement_opening(element: ET.Element) -> ET.Element:
    """The last definition without what the XML nests after it: the agreement's opening and title."""
    children = list(element)
    end = next((i for i, child in enumerate(children) if _is_agreement_opening(child)), len(children))
    while end > 1 and (children[end - 1].tag in ("page", "pageUUID") or _is_stray_caps_line(children[end - 1])):
        end -= 1
    if end == len(children):
        return element
    trimmed = ET.Element(element.tag, element.attrib)
    trimmed.text = element.text
    trimmed.extend(children[:end])
    return trimmed


def _pre_article_definitions_row(root: ET.Element, body: ET.Element) -> Dict[str, Any] | None:
    """The article-level row for a definitions block printed before Article 1, or None.

    Some agreements print their definitions between the recitals and Article 1.
    The XML serves them as loose text in <body>, so without this row they reach
    no section row and no section search. The row runs from the block's printed
    lead-in to the end of the definitions: to the first article, or to the
    agreement's own opening where it is printed after them. The preamble and
    recitals stay out either way, and the XML is not touched. Its article title is the block's printed heading, or
    blank if the block prints none; no heading is ever made up.
    """
    agreement_uuid = root.get("uuid")
    children = list(body)
    first_article = next((i for i, child in enumerate(children) if child.tag == "article"), None)
    if not agreement_uuid or first_article is None:
        return None
    if any(child.tag == "section" for child in children[:first_article]):
        return None
    definitions = [
        i
        for i, child in enumerate(children[:first_article])
        if child.tag == "definition"
        or (child.tag == "text" and _PRE_ARTICLE_DEFINITION_TEXT_RE.match(_element_text(child)))
    ]
    if len(definitions) < _PRE_ARTICLE_MIN_DEFINITIONS:
        return None

    start = definitions[0]
    heading = ""
    lead_ins = 0
    for i in range(start - 1, -1, -1):
        child = children[i]
        if child.tag != "text":
            if child.tag in ("page", "pageUUID"):
                continue
            break
        line = _element_text(child)
        if (
            lead_ins == _PRE_ARTICLE_MAX_LEAD_INS
            or _PRE_ARTICLE_NOT_LEAD_IN_RE.match(line)
            or not _PRE_ARTICLE_LEAD_IN_RE.search(line)
        ):
            break
        start = i
        lead_ins += 1
        run_in = _PRE_ARTICLE_RUN_IN_HEADING_RE.match(line)
        if run_in is not None:
            heading = run_in.group(0)
            break
        if len(line.split()) <= _PRE_ARTICLE_HEADING_MAX_WORDS:
            heading = line.rstrip(" :.")
            break

    last = definitions[-1]
    end = next((i for i in range(last + 1, first_article) if _is_agreement_opening(children[i])), first_article)
    block = [*children[start:last], _without_agreement_opening(children[last]), *children[last + 1 : end]]
    return {
        # No XML element carries this row, so its id derives from the agreement alone.
        "section_uuid": str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{agreement_uuid}\x1fpre-article-definitions")),
        "article_title": heading,
        "article_title_normed": clean_article_title(heading),
        "article_order": None,
        "section_title": "",
        "section_title_normed": "",
        "section_order": None,
        "xml_content": "".join(ET.tostring(child, encoding="unicode") for child in block),
    }


def extract_sections_from_xml(xml_str: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_str)

    out: List[Dict[str, Any]] = []

    body = root.find(".//body")
    if body is None:
        return out

    pre_article_definitions = _pre_article_definitions_row(root, body)
    if pre_article_definitions is not None:
        out.append(pre_article_definitions)

    # Sections nested within articles
    for article in body.findall(".//article"):
        article_title = article.get("title", "")
        article_title_normed = clean_article_title(article_title)
        article_uuid = article.get("uuid")
        article_order_raw = article.get("order")
        try:
            article_order = int(article_order_raw) if article_order_raw is not None else None
        except Exception:
            article_order = None

        article_sections = article.findall(".//section")
        article_level_xml = get_article_level_xml_excluding_sections(article)
        if article_uuid and article_level_xml.strip() and has_meaningful_xml_text(article_level_xml):
            out.append(
                {
                    "section_uuid": article_uuid,
                    "article_title": article_title,
                    "article_title_normed": article_title_normed,
                    "article_order": article_order,
                    "section_title": "",
                    "section_title_normed": "",
                    "section_order": None,
                    "xml_content": article_level_xml,
                }
            )

        for section in article_sections:
            section_uuid = section.get("uuid")
            section_title = section.get("title", "")
            section_title_normed = clean_section_title(section_title)
            xml_content = get_inner_xml(section)
            section_order_raw = section.get("order")
            try:
                section_order = int(section_order_raw) if section_order_raw is not None else None
            except Exception:
                section_order = None

            out.append(
                {
                    "section_uuid": section_uuid,
                    "article_title": article_title,
                    "article_title_normed": article_title_normed,
                    "article_order": article_order,
                    "section_title": section_title,
                    "section_title_normed": section_title_normed,
                    "section_order": section_order,
                    "xml_content": xml_content,
                }
            )

    # Sections directly under body (if any)
    for section in body.findall("./section"):
        section_uuid = section.get("uuid")
        section_title = section.get("title", "")
        section_title_normed = clean_section_title(section_title)
        xml_content = get_inner_xml(section)
        section_order_raw = section.get("order")
        try:
            section_order = int(section_order_raw) if section_order_raw is not None else None
        except Exception:
            section_order = None
        out.append(
            {
                "section_uuid": section_uuid,
                "article_title": "",
                "article_title_normed": "",
                "article_order": None,
                "section_title": section_title,
                "section_title_normed": section_title_normed,
                "section_order": section_order,
                "xml_content": xml_content,
            }
        )

    return out
