"""Deterministic tag-boundary repairs for XML generation.

These helpers only move or add XML heading tags around existing text. They do
not rewrite heading text.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from collections import Counter
from dataclasses import dataclass, field


SECTION_TAG_RE = re.compile(r"<section>(.*?)</section>", re.IGNORECASE | re.DOTALL)
ARTICLE_TAG_RE = re.compile(r"<article>(.*?)</article>", re.IGNORECASE | re.DOTALL)
STRUCTURAL_OPEN_RE = re.compile(r"(?=<section>|<article>|<page>|<pageUUID>)", re.IGNORECASE)
OMITTED_HEADING_FRAGMENT_RE = re.compile(
    r"(?:\[?\s*(?:intentionally\s+)?(?:omitted|deleted|reserved)\.?\s*\]?|\*{3,})",
    re.IGNORECASE,
)
SECTION_HEADING_SCAN_RE = re.compile(
    (
        r"(?<![\d.])(?:Section\s+)?(?P<article>\d+)\s*\.\s*(?P<section>\d+)"
        + r"\s*\.?\s+(?=[A-Z\[\(\x22\x27\u201c\u201d\u2018\u2019])"
    ),
    re.IGNORECASE,
)
SECTION_DECIMAL_TITLE_RE = re.compile(
    r"^\s*(?:SECTION\s+|Section\s+)?(?P<article>\d+)\s*\.\s*(?P<section>\d+)",
    re.IGNORECASE,
)
BARE_NUMBER_SECTION_TAG_RE = re.compile(
    r"<section>(?P<inner>\s*\d{1,3}\.?\s*)</section>",
    re.IGNORECASE,
)
STANDALONE_SECTION_LABEL_TAG_RE = re.compile(
    r"<section>(?P<inner>\s*(?:section|§)\s*)</section>",
    re.IGNORECASE,
)
NO_SPACE_SECTION_PREFIX_TAG_RE = re.compile(
    r"<section>(?P<prefix>\s*Section)(?P<number>\d{1,2}\.\d{1,3}\b)",
    re.IGNORECASE,
)
LEADING_PAGE_NUMBER_SECTION_TAG_RE = re.compile(
    r"<section>\s*(?P<page>\d{1,3})\s+(?P<title>(?:Section\s+)?\d{1,2}\.\d{1,3}\b.*?)</section>",
    re.IGNORECASE | re.DOTALL,
)
INLINE_SECTION_REFERENCE_TAG_RE = re.compile(
    (
        r"<section>(?P<inner>\s*(?:Section\s+)?\d{1,2}\.\d{1,3}\s*,"
        + r".*?)(?P<trailing>\s*)</section>"
    ),
    re.IGNORECASE | re.DOTALL,
)
ARTICLE_HEADING_SCAN_RE = re.compile(
    r"(?<![A-Za-z0-9])ARTICLE\s+(?P<number>[IVXLCDM]+|\d+)\s*\.?\s+",
    re.IGNORECASE,
)
WHOLE_NUMBER_HEADING_RE = re.compile(
    r"^\s*(?:(?:SECTION|Section)\s+)?(?P<article>\d+)\s*[\.):]?\s+(?P<rest>.+)$",
    re.DOTALL,
)
ROMAN_VALUES = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}


@dataclass(frozen=True)
class SectionGap:
    article_num: int
    expected: int
    found: int


@dataclass
class XMLTagRepairStats:
    attempts: Counter[str] = field(default_factory=Counter)
    applied: Counter[str] = field(default_factory=Counter)

    def update(self, other: "XMLTagRepairStats") -> None:
        self.attempts.update(other.attempts)
        self.applied.update(other.applied)


def _section_heading_pattern(article_num: int, section_num: int) -> re.Pattern[str]:
    pattern = (
        rf"(?<![\d.])(?:Section\s+)?{article_num}\s*\.\s*0*{section_num}"
        r"(?!\s*\.\s*\d)(?=\b|[\s.;:,\)\]\-])"
    )
    return re.compile(
        pattern,
        re.IGNORECASE,
    )


def _roman_to_int(value: str) -> int | None:
    roman = value.strip().upper()
    if not roman:
        return None
    total = 0
    prev = 0
    for char in reversed(roman):
        if char not in ROMAN_VALUES:
            return None
        current = ROMAN_VALUES[char]
        if current < prev:
            total -= current
        else:
            total += current
            prev = current
    return total


def _article_heading_number(match: re.Match[str]) -> int | None:
    token = match.group("number")
    if token.isdigit():
        return int(token)
    return _roman_to_int(token)


def _heading_starts_with_decimal(rest: str, article_num: str) -> bool:
    return re.match(rf"^\s*{re.escape(article_num)}\s*\.\s*\d+\b", rest) is not None


def _find_embedded_first_section(rest: str, article_num: str) -> re.Match[str] | None:
    pattern = re.compile(
        rf"(?<![\d.]){re.escape(article_num)}\s*\.\s*1\s*\.?(?:\s+(?=[A-Z\[\(])|$)",
        re.IGNORECASE,
    )
    return pattern.search(rest)


def _title_starts_with_same_article_decimal(title: str, article_num: str) -> bool:
    return (
        re.match(
            rf"^\s*(?:SECTION\s+|Section\s+)?{re.escape(article_num)}\s*\.\s*\d+\s*\.?(?:\s+|$)",
            title,
            re.IGNORECASE | re.DOTALL,
        )
        is not None
    )


def section_title_is_whole_number_article_heading(title: str) -> str | None:
    parsed = WHOLE_NUMBER_HEADING_RE.match(title.strip())
    if not parsed:
        return None
    article_num = parsed.group("article")
    rest = parsed.group("rest")
    if _heading_starts_with_decimal(rest, article_num):
        return None
    if _find_embedded_first_section(rest, article_num) is not None:
        return None
    return article_num


def section_title_starts_with_same_article_decimal(title: str, article_num: str) -> bool:
    return _title_starts_with_same_article_decimal(title, article_num)


def _untagged_article_heading_before_section_pattern(article_num: str) -> re.Pattern[str]:
    pattern = (
        rf"(?P<heading>(?<![\d.])(?:SECTION\s+|Section\s+)?"
        rf"{re.escape(article_num)}\.\s+(?!\d)[^\n<]{{2,220}}?)\s*$"
    )
    return re.compile(pattern, re.IGNORECASE)


def _split_tag_inner_at_positions(
    *,
    tag_name: str,
    inner: str,
    split_positions: list[int],
) -> str | None:
    if not split_positions:
        return None
    pieces: list[str] = []
    starts = [0, *sorted(set(split_positions))]
    ends = [*starts[1:], len(inner)]
    for start, end in zip(starts, ends):
        piece = inner[start:end].strip()
        if not piece:
            return None
        pieces.append(f"<{tag_name}>{piece}</{tag_name}>")
    return "\n\n".join(pieces)


def _omitted_section_split_positions(inner: str) -> list[int]:
    matches = list(SECTION_HEADING_SCAN_RE.finditer(inner))
    if len(matches) < 2 or inner[: matches[0].start()].strip():
        return []
    split_positions: list[int] = []
    for previous, current in zip(matches, matches[1:]):
        previous_article = int(previous.group("article"))
        current_article = int(current.group("article"))
        previous_section = int(previous.group("section"))
        current_section = int(current.group("section"))
        between = inner[previous.end() : current.start()]
        if previous_article != current_article:
            continue
        if current_section != previous_section + 1:
            continue
        if OMITTED_HEADING_FRAGMENT_RE.search(between):
            split_positions.append(current.start())
    return split_positions


def _omitted_article_split_positions(inner: str) -> list[int]:
    matches = list(ARTICLE_HEADING_SCAN_RE.finditer(inner))
    if len(matches) < 2 or inner[: matches[0].start()].strip():
        return []
    split_positions: list[int] = []
    for previous, current in zip(matches, matches[1:]):
        previous_number = _article_heading_number(previous)
        current_number = _article_heading_number(current)
        between = inner[previous.end() : current.start()]
        if previous_number is None or current_number is None:
            continue
        if current_number != previous_number + 1:
            continue
        if OMITTED_HEADING_FRAGMENT_RE.search(between):
            split_positions.append(current.start())
    return split_positions


def split_combined_omitted_heading_tags(tagged_text: str) -> tuple[str, XMLTagRepairStats]:
    stats = XMLTagRepairStats()

    def split_sections(match: re.Match[str]) -> str:
        inner = match.group(1)
        stats.attempts["split_combined_omitted_section_tags"] += 1
        replacement = _split_tag_inner_at_positions(
            tag_name="section",
            inner=inner,
            split_positions=_omitted_section_split_positions(inner),
        )
        if replacement is None:
            return match.group(0)
        stats.applied["split_combined_omitted_section_tags"] += 1
        return replacement

    def split_articles(match: re.Match[str]) -> str:
        inner = match.group(1)
        stats.attempts["split_combined_omitted_article_tags"] += 1
        replacement = _split_tag_inner_at_positions(
            tag_name="article",
            inner=inner,
            split_positions=_omitted_article_split_positions(inner),
        )
        if replacement is None:
            return match.group(0)
        stats.applied["split_combined_omitted_article_tags"] += 1
        return replacement

    repaired = SECTION_TAG_RE.sub(split_sections, tagged_text)
    repaired = ARTICLE_TAG_RE.sub(split_articles, repaired)
    return repaired, stats


def split_combined_missing_section_tags(
    tagged_text: str,
    gaps: Iterable[SectionGap],
) -> tuple[str, XMLTagRepairStats]:
    stats = XMLTagRepairStats()
    gap_numbers = {
        (gap.article_num, gap.expected)
        for gap in gaps
        if gap.found > gap.expected
    }
    if not gap_numbers:
        return tagged_text, stats

    def repl(match: re.Match[str]) -> str:
        inner = match.group(1)
        candidates: list[tuple[int, int, int]] = []
        for article_num, section_num in gap_numbers:
            stats.attempts["split_combined_missing_section_tags"] += 1
            for heading in _section_heading_pattern(article_num, section_num).finditer(inner):
                if heading.start() == 0:
                    continue
                candidates.append((heading.start(), article_num, section_num))
        if not candidates:
            return match.group(0)
        first_pos, _article_num, _section_num = min(candidates)
        left = inner[:first_pos].rstrip()
        right = inner[first_pos:].lstrip()
        if not left or not right:
            return match.group(0)
        stats.applied["split_combined_missing_section_tags"] += 1
        return f"<section>{left}</section>\n\n<section>{right}</section>"

    return SECTION_TAG_RE.sub(repl, tagged_text), stats


def split_embedded_article_section_headings(tagged_text: str) -> tuple[str, XMLTagRepairStats]:
    stats = XMLTagRepairStats()

    def repl(match: re.Match[str]) -> str:
        inner = match.group(1).strip()
        parsed = WHOLE_NUMBER_HEADING_RE.match(inner)
        stats.attempts["split_embedded_article_section_headings"] += 1
        if not parsed:
            return match.group(0)
        article_num = parsed.group("article")
        rest = parsed.group("rest")
        if _heading_starts_with_decimal(rest, article_num):
            return match.group(0)
        section_match = _find_embedded_first_section(rest, article_num)
        if section_match is None:
            return match.group(0)
        article_heading = inner[: parsed.start("rest") + section_match.start()].strip()
        section_heading = rest[section_match.start() :].strip()
        if not article_heading or not section_heading:
            return match.group(0)
        stats.applied["split_embedded_article_section_headings"] += 1
        return f"<article>{article_heading}</article>\n\n<section>{section_heading}</section>"

    return SECTION_TAG_RE.sub(repl, tagged_text), stats


def promote_whole_number_sections_to_articles(tagged_text: str) -> tuple[str, XMLTagRepairStats]:
    stats = XMLTagRepairStats()
    matches = list(SECTION_TAG_RE.finditer(tagged_text))
    replacements: dict[tuple[int, int], str] = {}

    for idx, match in enumerate(matches):
        inner = match.group(1).strip()
        stats.attempts["promote_whole_number_sections_to_articles"] += 1
        article_num = section_title_is_whole_number_article_heading(inner)
        if article_num is None:
            continue
        next_match = matches[idx + 1] if idx + 1 < len(matches) else None
        next_title = next_match.group(1).strip() if next_match else ""
        if not section_title_starts_with_same_article_decimal(next_title, article_num):
            continue
        stats.applied["promote_whole_number_sections_to_articles"] += 1
        replacements[match.span()] = f"<article>{inner}</article>"

    if not replacements:
        return tagged_text, stats

    pieces: list[str] = []
    last = 0
    for (start, end), replacement in sorted(replacements.items()):
        pieces.append(tagged_text[last:start])
        pieces.append(replacement)
        last = end
    pieces.append(tagged_text[last:])
    return "".join(pieces), stats


def apply_body_start_tag_repairs(tagged_text: str) -> tuple[str, XMLTagRepairStats]:
    stats = XMLTagRepairStats()
    repaired, split_stats = split_embedded_article_section_headings(tagged_text)
    stats.update(split_stats)
    repaired, promote_stats = promote_whole_number_sections_to_articles(repaired)
    stats.update(promote_stats)
    return repaired, stats


def unwrap_bare_number_section_tags(tagged_text: str) -> tuple[str, XMLTagRepairStats]:
    stats = XMLTagRepairStats()

    def repl(match: re.Match[str]) -> str:
        stats.attempts["unwrap_bare_number_section_tags"] += 1
        stats.applied["unwrap_bare_number_section_tags"] += 1
        return match.group("inner")

    return BARE_NUMBER_SECTION_TAG_RE.sub(repl, tagged_text), stats


def unwrap_standalone_section_label_tags(tagged_text: str) -> tuple[str, XMLTagRepairStats]:
    stats = XMLTagRepairStats()

    def repl(match: re.Match[str]) -> str:
        stats.attempts["unwrap_standalone_section_label_tags"] += 1
        stats.applied["unwrap_standalone_section_label_tags"] += 1
        return match.group("inner")

    return STANDALONE_SECTION_LABEL_TAG_RE.sub(repl, tagged_text), stats


def normalize_no_space_section_prefixes(tagged_text: str) -> tuple[str, XMLTagRepairStats]:
    stats = XMLTagRepairStats()

    def repl(match: re.Match[str]) -> str:
        stats.attempts["normalize_no_space_section_prefixes"] += 1
        stats.applied["normalize_no_space_section_prefixes"] += 1
        return f"<section>{match.group('prefix')} {match.group('number')}"

    return NO_SPACE_SECTION_PREFIX_TAG_RE.sub(repl, tagged_text), stats


def split_leading_page_number_section_tags(tagged_text: str) -> tuple[str, XMLTagRepairStats]:
    stats = XMLTagRepairStats()

    def repl(match: re.Match[str]) -> str:
        stats.attempts["split_leading_page_number_section_tags"] += 1
        stats.applied["split_leading_page_number_section_tags"] += 1
        return f"{match.group('page')} <section>{match.group('title').strip()}</section>"

    return LEADING_PAGE_NUMBER_SECTION_TAG_RE.sub(repl, tagged_text), stats


def unwrap_inline_section_reference_tags(tagged_text: str) -> tuple[str, XMLTagRepairStats]:
    stats = XMLTagRepairStats()

    def repl(match: re.Match[str]) -> str:
        stats.attempts["unwrap_inline_section_reference_tags"] += 1
        stats.applied["unwrap_inline_section_reference_tags"] += 1
        return f"{match.group('inner')}{match.group('trailing')}"

    return INLINE_SECTION_REFERENCE_TAG_RE.sub(repl, tagged_text), stats


def insert_missing_section_heading_tags(
    tagged_text: str,
    gaps: Iterable[SectionGap],
) -> tuple[str, XMLTagRepairStats]:
    stats = XMLTagRepairStats()
    candidates: list[tuple[int, int]] = []
    for gap in gaps:
        if gap.found <= gap.expected:
            continue
        pattern = _section_heading_pattern(gap.article_num, gap.expected)
        for match in pattern.finditer(tagged_text):
            stats.attempts["insert_missing_section_heading_tags"] += 1
            before = tagged_text[: match.start()]
            if before.lower().rfind("<section>") > before.lower().rfind("</section>"):
                continue
            if before.lower().rfind("<article>") > before.lower().rfind("</article>"):
                continue
            tail = tagged_text[match.start() :]
            close_match = STRUCTURAL_OPEN_RE.search(tail, pos=max(1, match.end() - match.start()))
            if close_match is None or close_match.start() <= 0:
                continue
            candidates.append((match.start(), match.start() + close_match.start()))

    if not candidates:
        return tagged_text, stats

    merged: list[tuple[int, int]] = []
    for start, end in sorted(set(candidates)):
        if merged and start < merged[-1][1]:
            continue
        merged.append((start, end))

    pieces: list[str] = []
    last = 0
    for start, end in merged:
        pieces.append(tagged_text[last:start])
        pieces.append("<section>")
        pieces.append(tagged_text[start:end].rstrip())
        pieces.append("</section>\n\n")
        last = end
        stats.applied["insert_missing_section_heading_tags"] += 1
    pieces.append(tagged_text[last:])
    return "".join(pieces), stats


def wrap_untagged_article_headings_before_first_sections(
    tagged_text: str,
) -> tuple[str, XMLTagRepairStats]:
    stats = XMLTagRepairStats()
    replacements: dict[tuple[int, int], str] = {}
    for section_match in SECTION_TAG_RE.finditer(tagged_text):
        section_title = section_match.group(1).strip()
        section_num_match = SECTION_DECIMAL_TITLE_RE.match(section_title)
        stats.attempts["wrap_untagged_article_headings_before_first_sections"] += 1
        if not section_num_match:
            continue
        if int(section_num_match.group("section")) != 1:
            continue
        article_num = section_num_match.group("article")
        prefix = tagged_text[: section_match.start()]
        heading_match = _untagged_article_heading_before_section_pattern(article_num).search(prefix)
        if heading_match is None:
            continue
        heading = heading_match.group("heading").strip()
        if not heading:
            continue
        replacements[(heading_match.start("heading"), heading_match.end("heading"))] = (
            f"<article>{heading}</article>"
        )

    if not replacements:
        return tagged_text, stats

    pieces: list[str] = []
    last = 0
    for (start, end), replacement in sorted(replacements.items()):
        pieces.append(tagged_text[last:start])
        pieces.append(replacement)
        last = end
        stats.applied["wrap_untagged_article_headings_before_first_sections"] += 1
    pieces.append(tagged_text[last:])
    return "".join(pieces), stats
