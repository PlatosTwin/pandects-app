"""
Shared text utilities for taxonomy classification.
"""

import re

_NUM_WORDS: list[str] = [
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
    text = re.sub(pattern, "", title)
    text = text.lower().strip()
    text = re.sub(r"[\s\.]+$", "", text)
    text = re.sub(r"[^\w\s\-()/]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_section_title(title: str) -> str:
    text = re.sub(
        r"(?i)^\s*section\s+[0-9]+(?:\.[0-9]+)*(?:\.)?\s*",
        "",
        title,
    )
    text = text.lower().strip()
    text = re.sub(r"[\s\.]+$", "", text)
    text = re.sub(r"[^\w\s\-()/]", "", text)
    text = re.sub(r"^\d+\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def combine_taxonomy_text(
    article_title: str,
    section_title: str,
    section_text: str,
) -> str:
    article_raw = article_title or ""
    section_raw = section_title or ""
    text_raw = section_text or ""
    article_norm = clean_article_title(article_raw)
    section_norm = clean_section_title(section_raw)

    # Repetition intentionally increases TF-IDF signal from title features.
    title_focus = " ".join(
        token
        for token in (
            section_norm,
            section_norm,
            section_norm,
            article_norm,
            article_norm,
        )
        if token
    )

    return (
        f"[ARTICLE_RAW] {article_raw}\n"
        f"[ARTICLE_NORM] {article_norm}\n"
        f"[SECTION_RAW] {section_raw}\n"
        f"[SECTION_NORM] {section_norm}\n"
        f"[TITLE_FOCUS] {title_focus}\n"
        f"[TEXT] {text_raw}"
    )
