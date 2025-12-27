# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
from __future__ import annotations

import re
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


def extract_sections_from_xml(xml_str: str) -> List[Dict[str, Any]]:
    root = ET.fromstring(xml_str)

    out: List[Dict[str, Any]] = []

    body = root.find(".//body")
    if body is None:
        return out

    # Sections nested within articles
    for article in body.findall(".//article"):
        article_title = article.get("title", "")
        article_title_normed = clean_article_title(article_title)
        article_order_raw = article.get("order")
        try:
            article_order = int(article_order_raw) if article_order_raw is not None else None
        except Exception:
            article_order = None

        for section in article.findall(".//section"):
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


