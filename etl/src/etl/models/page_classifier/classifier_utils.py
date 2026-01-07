"""
Feature extraction utilities for page classification.

This module provides functions to extract features from text and HTML content
for use in page classification models.
"""

import re
import string

import numpy as np


def extract_features(text: str, html: str, order: float) -> np.ndarray:
    """
    Extract features from text and HTML content for page classification.
    
    Args:
        text: Raw text content of the page
        html: HTML content of the page
        order: Page order/sequence number
        
    Returns:
        numpy.ndarray: Feature vector for classification
    """
    # Handle NaN values
    text = str(text) if text == text else ""
    html = str(html) if html == html else ""

    # Basic text statistics
    num_chars = len(text)
    words = text.split()
    num_words = len(words)
    avg_chars_per_word = num_chars / num_words if num_words > 0 else 0.0
    prop_spaces = text.count(" ") / num_chars if num_chars > 0 else 0.0
    prop_digits = sum(c.isdigit() for c in text) / num_chars if num_chars > 0 else 0.0
    prop_newlines = text.count("\\n") / num_chars if num_chars > 0 else 0.0
    prop_punct = (
        sum(c in string.punctuation for c in text) / num_chars if num_chars > 0 else 0.0
    )

    # Line-structure stats
    lines = text.split("\n")
    num_lines = len(lines)
    line_lengths = [len(line.strip()) for line in lines]
    avg_line_len = sum(line_lengths) / num_lines if num_lines > 0 else 0.0
    very_short_lines = sum(1 for length in line_lengths if length <= 3)
    empty_lines = sum(1 for length in line_lengths if length == 0)
    frac_short_lines = very_short_lines / num_lines if num_lines > 0 else 0.0
    frac_empty_lines = empty_lines / num_lines if num_lines > 0 else 0.0

    # Signature block line cues
    sig_line_re = re.compile(r"^(by|name|title)\s*:", re.IGNORECASE)
    sig_line_count = sum(1 for line in lines if sig_line_re.match(line.strip()))

    # Title/heading density
    non_empty_lines = [line for line in lines if line.strip()]
    non_empty_count = len(non_empty_lines)

    def _is_all_caps(line: str) -> bool:
        letters = [c for c in line if c.isalpha()]
        if not letters:
            return False
        return all(c.isupper() for c in letters)

    def _is_title_case(line: str) -> bool:
        tokens = [tok for tok in re.split(r"\s+", line.strip()) if tok]
        if not tokens:
            return False
        alpha_tokens = [tok for tok in tokens if any(c.isalpha() for c in tok)]
        if not alpha_tokens:
            return False
        for tok in alpha_tokens:
            letters = [c for c in tok if c.isalpha()]
            if not letters:
                continue
            if not letters[0].isupper():
                return False
            if any(c.isupper() for c in letters[1:]):
                return False
        return True

    all_caps_count = sum(1 for line in non_empty_lines if _is_all_caps(line))
    title_case_count = sum(1 for line in non_empty_lines if _is_title_case(line))
    frac_heading_lines = (
        (all_caps_count + title_case_count) / non_empty_count
        if non_empty_count > 0
        else 0.0
    )

    first_10_lines = lines[:10]
    count_article_10 = sum(
        len(re.findall(r"\\barticle\\b", line, flags=re.IGNORECASE))
        for line in first_10_lines
    )
    count_section_10 = sum(
        len(re.findall(r"\\bsection\\b", line, flags=re.IGNORECASE))
        for line in first_10_lines
    )

    # Page number detection
    _DIGIT_RE = re.compile(r"^[\-\s—]*(\d+)[\-\s—]*$")
    s = text.rsplit("\\n", 1)[-1]
    m = _DIGIT_RE.match(s)
    if m:
        num = m.group(1)
        flag_is_all_digits = 1
        flag_is_less_than_order = int(int(num) < order)
    else:
        flag_is_all_digits = 0
        flag_is_less_than_order = 0

    # Legal document specific features
    count_section = text.lower().count("section")
    count_article = text.lower().count("article")
    num_all_caps = sum(1 for w in words if w.isalpha() and w.isupper())
    prop_word_cap = (
        sum(1 for w in words if w[:1].isupper()) / num_words if num_words > 0 else 0.0
    )

    # N-gram features
    bigrams = [" ".join(bg) for bg in zip(words, words[1:])]
    num_bigrams = len(bigrams)
    unique_bigrams = len(set(bigrams))
    prop_unique_bigrams = unique_bigrams / num_bigrams if num_bigrams > 0 else 0.0
    trigrams = [" ".join(tg) for tg in zip(words, words[1:], words[2:])]
    num_trigrams = len(trigrams)
    unique_trigrams = len(set(trigrams))
    prop_unique_trigrams = unique_trigrams / num_trigrams if num_trigrams > 0 else 0.0

    # HTML structure features
    num_tags = html.count("<")
    tag_to_text_ratio = num_tags / num_chars if num_chars > 0 else 0.0
    link_count = html.lower().count("<a ")
    img_count = html.lower().count("<img ")
    heading_tags = sum(html.lower().count(f"<h{i}") for i in range(1, 7))
    list_count = html.lower().count("<li")

    # Document structure features
    bullet_count = sum(1 for line in text.split("\\n") if line.strip().startswith("-"))

    # Legal boilerplate terms
    boilerplate_terms = ["hereto", "herein", "hereby", "thereof", "wherein"]
    boilerplate_counts = [text.lower().count(t) for t in boilerplate_terms]

    # Legal document keywords
    legal_keywords = [
        "table of contents",
        "execution version",
        "in witness whereof",
        "exhibit",
        "signature",
        "list of exhibits",
        "schedule",
        "list of schedules",
        "index of",
        "recitals",
        "whereas",
        "now, therefore",
        "signed",
        "execution date",
        "effective",
        "dated as of",
        "entered into by and among",
        "[signature",
        "w i t n e s e t h",
        "/s/",
        "intentionally blank",
        "page follows]",
        "page follows.]",
        "by:",
    ]
    keyword_flags = [1.0 if kw in text.lower() else 0.0 for kw in legal_keywords]

    # Signature indicators
    sig_indicators = "by" in text.lower() and "title" in text.lower()

    # Punctuation analysis
    num_colon = text.count(":")
    num_period = text.count(".")
    num_consecutive_periods = max(
        (m.group().count(".") for m in re.finditer(r"(?: *\.)+", text)), default=0
    )
    num_comma = text.count(",")
    total_punct = sum(c in string.punctuation for c in text)
    prop_colon = num_colon / total_punct if total_punct > 0 else 0.0
    prop_period = num_period / total_punct if total_punct > 0 else 0.0
    prop_comma = num_comma / total_punct if total_punct > 0 else 0.0

    # HTML structure analysis
    has_table = 1.0 if re.search(r"</?(table|tr|td)", html.lower()) else 0.0
    count_p = html.lower().count("<p")
    count_div = html.lower().count("<div")

    # --- Extra cues for exhibits/back matter & signature blocks ---
    lower = text.lower()
    first_line = text.split("\n", 1)[0].strip()
    first_lower = first_line.lower()

    # Headings like "EXHIBIT A", "ANNEX I", "SCHEDULE 1.1(a)"
    exhibit_heading = 1.0 if re.search(r"^\s*(exhibit|annex|schedule)\s+([a-z0-9][a-z0-9\.\-\(\)]*)", first_lower, re.I) else 0.0
    # Common back-matter/exhibit phrases
    has_counterparts = 1.0 if ("counterpart" in lower or "counterparts" in lower) else 0.0
    has_form_of = 1.0 if ("form of" in lower and "agreement" in lower) else 0.0
    signature_page_to = 1.0 if ("signature page to" in lower) else 0.0
    # Signature field cues
    sig_field_cues = sum(kw in lower for kw in ["name:", "title:", "its:", "date:"])
    # Long underscore runs (signature lines)
    underscore_runs = len(re.findall(r"_{5,}", text))
    
    # Compile feature vector
    features = [
        num_words,
        num_chars,
        avg_chars_per_word,
        prop_spaces,
        prop_digits,
        prop_newlines,
        prop_punct,
        num_lines,
        avg_line_len,
        frac_short_lines,
        frac_empty_lines,
        sig_line_count,
        frac_heading_lines,
        count_article_10,
        count_section_10,
        flag_is_all_digits,
        flag_is_less_than_order,
        count_section,
        count_article,
        num_all_caps,
        prop_word_cap,
        num_bigrams,
        unique_bigrams,
        prop_unique_bigrams,
        num_trigrams,
        unique_trigrams,
        prop_unique_trigrams,
        num_tags,
        tag_to_text_ratio,
        link_count,
        img_count,
        heading_tags,
        list_count,
        bullet_count,
        *boilerplate_counts,
        *keyword_flags,
        sig_indicators,
        prop_colon,
        prop_period,
        num_consecutive_periods,
        prop_comma,
        has_table,
        count_p,
        count_div,
        order,
        exhibit_heading,
        has_counterparts,
        has_form_of,
        signature_page_to,
        float(sig_field_cues),
        float(underscore_runs),
    ]
    return np.array(features, dtype=float)
