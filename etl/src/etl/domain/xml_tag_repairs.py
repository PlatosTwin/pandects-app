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
ORPHAN_SECTION_LABEL_FRAGMENT_RE = re.compile(
    r"^\s*(?:section|sections|secton|sectons|§)\s*\.?:?\s*$",
    re.IGNORECASE,
)
NO_SPACE_SECTION_PREFIX_TAG_RE = re.compile(
    r"<section>(?P<prefix>\s*Section)(?P<number>\d{1,2}\.\d{1,3}\b)",
    re.IGNORECASE,
)
# The tagger closes a heading at a word-piece boundary, and its pre-tokenizer splits
# digit runs from letter runs, so a heading can end inside a word: "Section 280</section>G",
# "WARRANTIES OF H</article>2EG". The tail is moved back inside only where the tag splits a
# digit from a letter (or the reverse) and the tail is not a word that could start the
# next sentence ("Section 5.1</section>A Party", "Section 5</section>IN WITNESS").
DIGIT_LETTER_SPLIT_TAIL_RE = re.compile(
    r"(?<=[0-9])</(?P<tag>section|article)>(?P<tail>[A-Z]{1,3})(?=\s|$)"
)
LETTER_DIGIT_SPLIT_TAIL_RE = re.compile(
    r"(?<=[A-Za-z])</(?P<tag>section|article)>(?P<tail>[0-9]+[A-Za-z][A-Za-z0-9]*)(?=\s|$)"
)
SPLIT_TAIL_SENTENCE_WORDS = frozenset(
    """
    A I AN AS AT BE BY DO IF IN IS IT NO OF ON OR SO TO UP WE
    ALL AND ANY ARE BUT EACH FOR HAS ITS MAY NOR NOT THE WAS
    """.split()
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
STRAY_DECIMAL_INNER = r"\s*(?:Section\s+)?\d{1,3}\s*\.\s*\d{1,3}[\s.(]*"
STRAY_DECIMAL_AFTER_SECTION_TAG_RE = re.compile(
    (
        r"<section>(?P<prev_title>[^<]*)(?P<prev></section>\s*)"
        + rf"<section>(?P<inner>{STRAY_DECIMAL_INNER})</section>(?!\s*<section>)"
        + r"(?P<after>\s*\w*)"
    ),
    re.IGNORECASE,
)
STRAY_DECIMAL_FRAGMENT_RE = re.compile(r"\d\s+\.|\.\s*\(")
NUMBER_ONLY_SECTION_BEFORE_TITLE_RE = re.compile(
    (
        rf"<section>(?P<number>{STRAY_DECIMAL_INNER})</section>"
        + r"(?P<gap>\s*)<section>(?P<title>[^<]*?)</section>"
    ),
    re.IGNORECASE,
)
SECTION_OF_REFERENCE_TAG_RE = re.compile(
    r"<section>(?P<inner>\s*(?:Section\s+)?\d{1,2}\s*\.\s*\d{1,3}\s+of\s[^<]*?)</section>",
    re.IGNORECASE,
)
ADJACENT_SAME_TYPE_HEADING_PAIR_RE = re.compile(
    (
        r"<(?P<tag>section|article)>(?P<first>[^<]*?)</(?P=tag)>"
        + r"(?P<gap>\s*)<(?P=tag)>(?P<second>[^<]*?)</(?P=tag)>"
    ),
    re.IGNORECASE,
)
SAME_LINE_SECTION_PAIR_RE = re.compile(
    r"(?P<first><section>[^<]*?</section>)(?P<gap>[ \t]+)<section>(?P<second>[^<]*?)</section>",
    re.IGNORECASE,
)
BARE_ARTICLE_TAG_BEFORE_SECTION_RE = re.compile(
    (
        r"<article>(?P<inner>\s*ARTICLE\s+(?P<number>[IVXLCDM]+|\d+)\s*\.?\s*)</article>"
        + r"(?P<gap>\s*)(?=<section>(?P<next>[^<]*))"
    ),
    re.IGNORECASE,
)
ARTICLE_TITLE_NUMBER_RE = re.compile(
    r"^\s*ARTICLE\s+(?P<number>[IVXLCDM]+|\d+)\b",
    re.IGNORECASE,
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
    """Match a printed section number that can start a heading.

    The number must be followed by the end of the line, a tag, or (after an
    optional period or dash) a heading-like token: an uppercase letter, a
    quote, a bracket, a number after a space ("1.2 2014 Option Rights"), or a
    spaced clause label followed by an uppercase word ("1.2 (b) Northwest
    shall"). Cross-reference followers such as ``,``, ``(a)``, ``)``, ``;`` or
    a lowercase word (``of``, ``and``, ``date hereof``) never match.
    """
    pattern = (
        rf"(?<![\d.])(?:(?:Section|SECTION|section)\s+)?{article_num}\s*\.\s*0*{section_num}"
        r"(?!\s*\.\s*\d)(?!\d)"
        # [^\S\r\n] is any horizontal space (nbsp, en/em/thin spaces too), never a newline.
        r"(?=[^\S\r\n]*[.:]?[^\S\r\n]*(?:[-\u2013\u2014][^\S\r\n]*)?"
        r"(?:$|<|[A-Z\[\x22\x27\u201c\u201d\u2018\u2019])|[^\S\r\n]+\d"
        r"|[^\S\r\n]+\([A-Za-z0-9]{1,4}\)[^\S\r\n]+[A-Z])"
    )
    # Case-sensitive on purpose: the follower check needs a real uppercase
    # letter. MULTILINE lets "$" accept a number alone at the end of a line.
    return re.compile(pattern, re.MULTILINE)


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
        if ORPHAN_SECTION_LABEL_FRAGMENT_RE.fullmatch(left):
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


def move_split_word_tails_into_headings(tagged_text: str) -> tuple[str, XMLTagRepairStats]:
    """Move a word tail glued to a closing heading tag back inside the heading."""
    stats = XMLTagRepairStats()

    def repl(match: re.Match[str]) -> str:
        stats.attempts["move_split_word_tails_into_headings"] += 1
        if match.group("tail").upper() in SPLIT_TAIL_SENTENCE_WORDS:
            return match.group(0)
        stats.applied["move_split_word_tails_into_headings"] += 1
        return f"{match.group('tail')}</{match.group('tag')}>"

    repaired = DIGIT_LETTER_SPLIT_TAIL_RE.sub(repl, tagged_text)
    return LETTER_DIGIT_SPLIT_TAIL_RE.sub(repl, repaired), stats


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


# Text allowed between the start of a line and an inserted section heading:
# whitespace, page markers, heading tags, and a bare page number ("12", "- 12 -").
# Text extracted from HTML often runs a whole article onto one line, so the end
# of a sentence (". ", "; ", ": ") or of a heading tag also counts as a line start.
MISSING_HEADING_SENTENCE_START_RE = re.compile(
    r"(?:[.;:]|</(?:section|article)>)[ \t\u00a0]+$",
    re.IGNORECASE,
)
MISSING_HEADING_LINE_PREFIX_RE = re.compile(
    r"^(?:\s|<pageUUID>[^<]*</pageUUID>|<page>[^<]*</page>|</?(?:section|article)>)*"
    r"(?:[-\u2013\u2014]?[ \t]*\d{1,3}[ \t]*[-\u2013\u2014]?[ \t]+)?$",
    re.IGNORECASE,
)
# End of a heading's title sentence on a line that also carries body text:
# a period (or colon) followed by whitespace and an uppercase-led sentence. A
# single capital letter before the period ("U.S.") and common abbreviations do
# not end the title.
MISSING_HEADING_TITLE_END_RE = re.compile(
    r"(?<![\s.][A-Z])(?<!\bInc)(?<!\bNo)(?<!\bCo)(?<!\bLtd)(?<!\bCorp)[.:]"
    r"(?=[ \t\u00a0]+[A-Z\x22\u201c(])"
)
MISSING_HEADING_TITLE_MAX_CHARS = 250
# A title sentence longer than this is body text, not a heading, unless it is
# Title Case and no longer than MISSING_HEADING_TITLE_CASE_SENTENCE_MAX_CHARS.
MISSING_HEADING_SENTENCE_MAX_CHARS = 120
MISSING_HEADING_TITLE_CASE_SENTENCE_MAX_CHARS = 200
_QUOTED_TERM = r"[\x22\u201c\u201d][^\x22\u201c\u201d\n<]{1,120}[\x22\u201c\u201d]"
_HSPACE = r"[^\S\r\n]"
# Quoted defined term(s) ("1.70 “UCC” means ...", "“Party” and “Parties” shall
# have ...", "“U.S. GAAP” or “GAAP” means ..."): the title is the term(s), with
# a period or colon printed right after them ("“Knowledge”. As used ...").
MISSING_HEADING_DEFINED_TERM_RE = re.compile(
    _QUOTED_TERM
    + rf"(?:{_HSPACE}*(?:,{_HSPACE}*(?:(?:or|and){_HSPACE}+)?|(?:and/or|or|and){_HSPACE}+){_QUOTED_TERM})*"
    + rf"(?:{_HSPACE}?[.:](?!\d))?"
)
_DEFINING_VERB_RE = re.compile(
    rf"{_HSPACE}*[.:]?{_HSPACE}+(?:means|has the meaning|shall have the meaning|shall mean)\b"
)
# Unquoted defined term ("1.27 IT Systems has the meaning set out in ...",
# "1.2 2014 Option Rights has the meaning ..."): the title is the Title Case
# term before the defining verb.
MISSING_HEADING_UNQUOTED_TERM_RE = re.compile(
    r"(?P<term>[A-Z0-9][^\n<.;:]{0,78}?)"
    r"(?=[ \t\u00a0]+(?:means|has the meaning|shall have the meaning|shall mean)\b)"
)
_TITLE_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\u2019-]*")
_TITLE_SMALL_WORDS = frozenset(
    "a an and as at by for from in of on or the to under upon with".split()
)


MISSING_HEADING_PROSE_OPENING_RE = re.compile(r"In this\b")
# Words that open a body sentence; a heading's Title Case run ends right before
# one of them ("Number and Gender In this Agreement ...").
_SENTENCE_OPENERS = frozenset(
    """
    a after all an any as at before during each except for from if in it no
    notwithstanding on prior save such subject the there this unless upon where
    whereas when with without
    """.split()
)
_RUN_WORD_RE = re.compile(r"\S+")
_RUN_MAX_WORDS = 12
_RUN_MAX_CHARS = 80


def _strip_word(word: str) -> str:
    return word.strip("\x22\x27\u201c\u201d\u2018\u2019()[],;:.")


def _leading_title_run_end(tagged_text: str, title_from: int, line_end: int) -> int | None:
    """End of a heading printed without a period before its body text, or None.

    "Merger Without a Stockholders’ Meeting As promptly as ..." -> the run before
    "As"; "Coeur’s indemnity Coeur agrees ..." -> the run before the repeated
    "Coeur". The run ends right before a capitalized sentence opener (or a
    repeat of a run word) that is followed by a lowercase word. Anything
    ambiguous returns None (the caller then uses a number-only title).
    """
    words = list(_RUN_WORD_RE.finditer(tagged_text, title_from, line_end))
    if len(words) < 3:
        return None
    first = _strip_word(words[0].group())
    if not first or not first[0].isupper() or first.lower() in _SENTENCE_OPENERS:
        return None
    for i in range(1, min(len(words) - 1, _RUN_MAX_WORDS + 1)):
        word = _strip_word(words[i].group())
        following = _strip_word(words[i + 1].group())
        if not word or not word[0].isupper() or not following or not following[0].islower():
            continue
        if following.lower() in _TITLE_SMALL_WORDS:
            continue
        run_words = [_strip_word(w.group()) for w in words[:i]]
        stems = {w.lower().removesuffix("\u2019s").removesuffix("'s") for w in run_words}
        if word.lower() not in _SENTENCE_OPENERS and word.lower() not in stems:
            return None  # "Kabel BW Holdings is ...": a proper-noun subject, ambiguous
        run = tagged_text[title_from : words[i - 1].end()]
        lowercase = [w for w in run_words if w and w[0].islower() and w.lower() not in _TITLE_SMALL_WORDS]
        if (
            len(lowercase) > 1
            or run_words[-1].lower() in _TITLE_SMALL_WORDS
            or len(run) > _RUN_MAX_CHARS
            or re.search(r"[.;,]", run.rstrip(".:"))
        ):
            return None
        return words[i - 1].end()
    return None


def _looks_like_prose(title: str) -> bool:
    """True for the start of a body sentence rather than a heading.

    Headings are short or Title Case ("Merger Consideration; Conversion of
    Shares"); an untitled clause reads as a sentence ("All references in this
    Agreement to ...").
    """
    stripped = title.strip()
    if MISSING_HEADING_PROSE_OPENING_RE.match(stripped):
        return True  # "In this Agreement, unless the context otherwise requires"
    if stripped[-1:] in {":", "-", "\u2013", "\u2014"}:
        return True  # lead-in to a list: "If to Buyer:", "... requires –"
    if len(title) <= 60:
        return False
    letters = [c for c in title if c.isalpha()]
    if len(title) > 80 and letters and sum(c.isupper() for c in letters) / len(letters) > 0.9:
        return True  # a long all-caps line is a clause ("EACH OF THE PARTIES HERETO WAIVES ...")
    words = [w for w in _TITLE_WORD_RE.findall(title) if w.lower() not in _TITLE_SMALL_WORDS]
    if not words:
        return False
    capitalized = sum(1 for w in words if w[0].isupper())
    return capitalized / len(words) < 0.5


def _is_title_case(title: str) -> bool:
    words = [w for w in _TITLE_WORD_RE.findall(title) if w.lower() not in _TITLE_SMALL_WORDS]
    if not words:
        return False
    capitalized = sum(1 for w in words if w[0].isupper())
    return capitalized / len(words) >= 0.8


# Mid-line, a heading may follow a table cell or a reference ("... Section
# 10.4(d) 1.2 Other Capitalized Terms. For all ...") but never a lowercase word
# ("aware on the 1.40 date", "thereunder 1.40 IFRS").
MISSING_HEADING_AFTER_LOWERCASE_WORD_RE = re.compile(r"[a-z][\w'\u2019]*[,]?[ \t\u00a0]+$")
MISSING_HEADING_MID_LINE_TITLE_MAX_CHARS = 100


def _unquoted_defined_term_end(tagged_text: str, title_from: int, line_end: int) -> int | None:
    """End of a Title Case defined term followed by "means" / "has the meaning", or None."""
    match = MISSING_HEADING_UNQUOTED_TERM_RE.match(tagged_text, title_from, line_end)
    if match is None or not _is_title_case(match.group("term")):
        return None
    return match.end()


def _defines_term_at(tagged_text: str, number_end: int) -> bool:
    """True when the section number is followed by a defined term and "means"."""
    after = re.compile(r"[ \t\u00a0]*\.?[ \t\u00a0]*").match(tagged_text, number_end)
    start = after.end() if after else number_end
    line_end = tagged_text.find("\n", start)
    tag = tagged_text.find("<", start)
    limit = min(x for x in (line_end, tag, len(tagged_text)) if x >= 0)
    quoted = MISSING_HEADING_DEFINED_TERM_RE.match(tagged_text, start, limit)
    if quoted is not None:
        # The match may have absorbed a trailing ":" ("“X”: shall mean").
        verb_from = quoted.end() - 1 if tagged_text[quoted.end() - 1] in ".:" else quoted.end()
        return _DEFINING_VERB_RE.match(tagged_text[verb_from:limit]) is not None
    return _unquoted_defined_term_end(tagged_text, start, limit) is not None


# Words that make a following number a cross-reference, not a heading.
MISSING_HEADING_XREF_PREFIX_RE = re.compile(
    r"(?:\b(?:[Ss]ections?|SECTIONS?|[Cc]lauses?|[Aa]rticles?|[Ss]chedules?|[Pp]aragraphs?|"
    r"pursuant to|of|in|under|to|and|or|with|by|see|per|than|at|on|from|this|that|such|said)"
    r"|[\u00a7(,])[^\S\r\n]*$"
)
_CAPS_TERM_RE = re.compile(r"(?P<term>[A-Z][A-Z0-9&\-]{1,15})(?=[^\S\r\n]+[A-Z][a-z])")
_DEFINITION_FOLLOWER_RE = re.compile(
    r"[^\S\r\n]*[.:]?[^\S\r\n]*(?:means|has the meaning|shall have the meaning|shall mean)\b"
)


def _caps_term_end(tagged_text: str, number_end: int) -> int | None:
    """End of an all-caps term right after the number ("1.40 IFRS International ...")."""
    after = re.compile(rf"{_HSPACE}+").match(tagged_text, number_end)
    if after is None:
        return None
    match = _CAPS_TERM_RE.match(tagged_text, after.end())
    return match.end() if match else None


def _in_definitions_run(tagged_text: str, pos: int) -> bool:
    """True when the nearest heading before pos is a definition ("<section>1.39 X</section> means")."""
    open_at = tagged_text.rfind("<section>", 0, pos)
    if open_at < 0:
        return False
    close_at = tagged_text.find("</section>", open_at, pos)
    if close_at < 0:
        return False
    inner = tagged_text[open_at + len("<section>") : close_at]
    if re.search(r"[\x22\u201c\u201d]", inner):
        return True
    return _DEFINITION_FOLLOWER_RE.match(tagged_text, close_at + len("</section>"), pos) is not None


def _missing_heading_title_end(tagged_text: str, start: int, number_end: int) -> tuple[int, str] | None:
    """Return where an inserted section heading tag should close, or None.

    The title is the heading line: from the number to the end of its line (or
    the next tag); a number printed alone on its line takes the next non-empty
    line. Then:
    - a defined term ("1.70 “UCC” means ...") ends the title after the term;
    - a short first sentence ("2.2 Merger Consideration. At the ...") ends it;
    - a line that reads as prose, or is longer than
      MISSING_HEADING_TITLE_MAX_CHARS, keeps a leading Title Case run printed
      without a period ("Number and Gender In this ..."); otherwise it is an
      untitled clause: the title is the number alone and the line stays body
      text.
    Returns (end, kind) with kind "term", "sentence", "line", "run" or "number".
    """
    limit = tagged_text.find("<", number_end)
    if limit < 0:
        limit = len(tagged_text)
    after_number = re.compile(r"[ \t\u00a0]*[.:]?[ \t\u00a0]*(?:[-\u2013\u2014][ \t\u00a0]*)?").match(
        tagged_text, number_end
    )
    title_from = after_number.end() if after_number else number_end
    number_only_end = title_from
    while number_only_end > number_end and tagged_text[number_only_end - 1] in " \t\u00a0-\u2013\u2014":
        number_only_end -= 1
    line_end = tagged_text.find("\n", title_from, limit)
    if line_end < 0:
        line_end = limit
    if not tagged_text[title_from:line_end].strip():
        # Number printed alone: the title is the next non-empty line.
        next_line = re.compile(r"\s*").match(tagged_text, line_end, limit)
        title_from = next_line.end() if next_line else line_end
        if title_from >= limit:
            return None
        line_end = tagged_text.find("\n", title_from, limit)
        if line_end < 0:
            line_end = limit
    term = MISSING_HEADING_DEFINED_TERM_RE.match(tagged_text, title_from, line_end)
    if term is not None:
        return term.end(), "term"
    unquoted_term = _unquoted_defined_term_end(tagged_text, title_from, line_end)
    if unquoted_term is not None:
        return unquoted_term, "term"
    kind = "line"
    end = line_end
    sentence_end = MISSING_HEADING_TITLE_END_RE.search(tagged_text, title_from, line_end)
    if sentence_end is not None:
        sentence_chars = sentence_end.end() - title_from
        if sentence_chars <= MISSING_HEADING_SENTENCE_MAX_CHARS or (
            sentence_chars <= MISSING_HEADING_TITLE_CASE_SENTENCE_MAX_CHARS
            and _is_title_case(tagged_text[title_from : sentence_end.end()])
        ):
            end = sentence_end.end()
            kind = "sentence"
    while end > start and tagged_text[end - 1].isspace():
        end -= 1
    title = tagged_text[title_from:end]
    if end - start > MISSING_HEADING_TITLE_MAX_CHARS or _looks_like_prose(title):
        run_end = _leading_title_run_end(tagged_text, title_from, line_end)
        if run_end is not None:
            return run_end, "run"
        return number_only_end, "number"
    return end, kind


def insert_missing_section_heading_tags(
    tagged_text: str,
    gaps: Iterable[SectionGap],
) -> tuple[str, XMLTagRepairStats]:
    """Tag an untagged section heading whose number fills a numbering gap.

    A heading qualifies at the start of a line (after whitespace, page markers
    or a bare page number) or right after the end of a sentence; elsewhere on a
    line only a short Title Case heading sentence that does not follow a
    lowercase word qualifies ("... 10.4(d) 1.2 Other Capitalized Terms. For").
    The number must also be followed by a heading-like token (see
    _section_heading_pattern), so mid-sentence
    cross-references ("Subject to Section 2.2, ...") are never promoted. The
    inserted tag closes at the end of the heading's title (see
    _missing_heading_title_end); the rest stays body text.
    """
    stats = XMLTagRepairStats()
    candidates: list[tuple[int, int]] = []
    lowered = tagged_text.lower()
    for gap in gaps:
        if gap.found <= gap.expected:
            continue
        pattern = _section_heading_pattern(gap.article_num, gap.expected)
        for match in pattern.finditer(tagged_text):
            stats.attempts["insert_missing_section_heading_tags"] += 1
            before = lowered[: match.start()]
            if before.rfind("<section>") > before.rfind("</section>"):
                continue
            if before.rfind("<article>") > before.rfind("</article>"):
                continue
            line_start = tagged_text.rfind("\n", 0, match.start()) + 1
            line_prefix = tagged_text[line_start : match.start()]
            at_real_line_start = MISSING_HEADING_LINE_PREFIX_RE.match(line_prefix) is not None
            at_line_start = (
                at_real_line_start or MISSING_HEADING_SENTENCE_START_RE.search(line_prefix) is not None
            )
            if not at_real_line_start and MISSING_HEADING_XREF_PREFIX_RE.search(line_prefix):
                continue  # "pursuant to Section 5.11 ...", "in 2.2 ...", "(see 4.3 ...)"
            # A defined term ("1.42 Existing Debt Facilities means ...") is a heading
            # wherever it sits in flattened definitions text.
            is_definition = _defines_term_at(tagged_text, match.end())
            if (
                not at_line_start
                and not is_definition
                and MISSING_HEADING_AFTER_LOWERCASE_WORD_RE.search(line_prefix)
            ):
                # Inside a run of definitions, an all-caps term opens a definition
                # even after a lowercase word: "... promulgated thereunder 1.40 IFRS
                # International Financial Reporting Standards, as in effect ...".
                caps_term_end = _caps_term_end(tagged_text, match.end())
                if caps_term_end is None or not _in_definitions_run(tagged_text, match.start()):
                    continue
                candidates.append((match.start(), caps_term_end))
                continue
            result = _missing_heading_title_end(tagged_text, match.start(), match.end())
            if result is None:
                continue
            end, kind = result
            if end < match.end():
                continue
            title = tagged_text[match.end() : end].strip(" \t .:-–—")
            if not at_real_line_start and kind in {"sentence", "line"} and not _is_title_case(title):
                # Away from a real line start only a Title Case heading counts:
                # "... this Agreement. Section 5.11 Disclosure Schedule sets forth ..." does not.
                continue
            if not at_line_start and not is_definition:
                # Mid-line: only a short Title Case heading sentence qualifies.
                if (
                    not tagged_text[:end].endswith(".")
                    or not title
                    or end - match.start() > MISSING_HEADING_MID_LINE_TITLE_MAX_CHARS
                    or not _is_title_case(title)
                ):
                    continue
            candidates.append((match.start(), end))

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
        pieces.append(tagged_text[start:end])
        pieces.append("</section>")
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


def _section_title_decimal(title: str) -> tuple[int, int] | None:
    match = SECTION_DECIMAL_TITLE_RE.match(title)
    if match is None:
        return None
    return int(match.group("article")), int(match.group("section"))


def _article_title_number(title: str) -> int | None:
    match = ARTICLE_TITLE_NUMBER_RE.match(title)
    if match is None:
        return None
    return _article_heading_number(match)


def _heading_number(tag_name: str, title: str) -> tuple[int, ...] | None:
    if tag_name.lower() == "article":
        number = _article_title_number(title)
        return None if number is None else (number,)
    return _section_title_decimal(title)


def unwrap_section_of_reference_tags(tagged_text: str) -> tuple[str, XMLTagRepairStats]:
    """Unwrap cross-references such as "Section 4.21 of the Disclosure Schedule"."""
    stats = XMLTagRepairStats()

    def repl(match: re.Match[str]) -> str:
        stats.attempts["unwrap_section_of_reference_tags"] += 1
        stats.applied["unwrap_section_of_reference_tags"] += 1
        return match.group("inner")

    return SECTION_OF_REFERENCE_TAG_RE.sub(repl, tagged_text), stats


def merge_number_only_section_titles(tagged_text: str) -> tuple[str, XMLTagRepairStats]:
    """Merge a number-only section tag into the next tag when that tag is its title.

    The next tag counts as a title when it does not open with a section number of
    the same article, e.g. "Section 10.6" + "Section 56.4 Election." or
    "6.8" + "Section 16(b) Exemption.".
    """
    stats = XMLTagRepairStats()

    def repl(match: re.Match[str]) -> str:
        stats.attempts["merge_number_only_section_titles"] += 1
        number = _section_title_decimal(match.group("number"))
        title_number = _section_title_decimal(match.group("title"))
        if number is None or (title_number is not None and title_number[0] == number[0]):
            return match.group(0)
        stats.applied["merge_number_only_section_titles"] += 1
        return f"<section>{match.group('number')}{match.group('gap')}{match.group('title')}</section>"

    return NUMBER_ONLY_SECTION_BEFORE_TITLE_RE.sub(repl, tagged_text), stats


def unwrap_stray_decimal_section_tags(tagged_text: str) -> tuple[str, XMLTagRepairStats]:
    """Unwrap a bare section number ("4.6 .", "2.1", "4.1.(") that follows a section tag.

    The number is unwrapped only when it is a cross-reference ("Section 2.24 of
    the ..."), a fragment ("4.6 ."), or cannot be the next clause after the
    previous section, so a real number-only clause such as "4 Completion" +
    "4.1" or "2.1 ..." + "2.2" stays.
    """
    stats = XMLTagRepairStats()

    def repl(match: re.Match[str]) -> str:
        number = _section_title_decimal(match.group("inner"))
        previous = _section_title_decimal(match.group("prev_title"))
        if number is None or previous is None:
            return match.group(0)
        stats.attempts["unwrap_stray_decimal_section_tags"] += 1
        is_reference = match.group("after").strip().lower() == "of"
        # "4.6 ." or "4.1.(" is a split-off fragment, never a clause number of its own.
        is_fragment = STRAY_DECIMAL_FRAGMENT_RE.search(match.group("inner")) is not None
        next_numbers = {(previous[0], previous[1] + 1), (previous[0] + 1, 1)}
        if not (is_reference or is_fragment) and number in next_numbers:
            return match.group(0)
        stats.applied["unwrap_stray_decimal_section_tags"] += 1
        return (
            f"<section>{match.group('prev_title')}{match.group('prev')}"
            + f"{match.group('inner')}{match.group('after')}"
        )

    return STRAY_DECIMAL_AFTER_SECTION_TAG_RE.sub(repl, tagged_text), stats


def dedupe_adjacent_same_number_headings(tagged_text: str) -> tuple[str, XMLTagRepairStats]:
    """Unwrap the first of two adjacent same-type headings that carry the same number.

    Covers headings the source prints twice ("ARTICLE VIII" + "ARTICLE VIII
    TERMINATION", or an inline copy followed by an underlined copy).
    """
    stats = XMLTagRepairStats()

    def repl(match: re.Match[str]) -> str:
        tag_name = match.group("tag")
        first_number = _heading_number(tag_name, match.group("first"))
        if first_number is None:
            return match.group(0)
        stats.attempts["dedupe_adjacent_same_number_headings"] += 1
        if _heading_number(tag_name, match.group("second")) != first_number:
            return match.group(0)
        stats.applied["dedupe_adjacent_same_number_headings"] += 1
        return f"{match.group('first')}{match.group('gap')}<{tag_name}>{match.group('second')}</{tag_name}>"

    return ADJACENT_SAME_TYPE_HEADING_PAIR_RE.sub(repl, tagged_text), stats


def unwrap_same_line_cross_article_section_tags(tagged_text: str) -> tuple[str, XMLTagRepairStats]:
    """Unwrap a section tag on the same line as the previous one when it belongs to another article."""
    stats = XMLTagRepairStats()

    def repl(match: re.Match[str]) -> str:
        first_number = _section_title_decimal(SECTION_TAG_RE.sub(r"\1", match.group("first")))
        second_number = _section_title_decimal(match.group("second"))
        if first_number is None or second_number is None:
            return match.group(0)
        stats.attempts["unwrap_same_line_cross_article_section_tags"] += 1
        if first_number[0] == second_number[0]:
            return match.group(0)
        stats.applied["unwrap_same_line_cross_article_section_tags"] += 1
        return f"{match.group('first')}{match.group('gap')}{match.group('second')}"

    return SAME_LINE_SECTION_PAIR_RE.sub(repl, tagged_text), stats


def unwrap_bare_article_tags_before_other_article_sections(
    tagged_text: str,
) -> tuple[str, XMLTagRepairStats]:
    """Unwrap a bare "ARTICLE N" tag that sits directly before a section of another article."""
    stats = XMLTagRepairStats()

    def repl(match: re.Match[str]) -> str:
        article_number = _article_heading_number(match)
        next_number = _section_title_decimal(match.group("next"))
        if article_number is None or next_number is None:
            return match.group(0)
        stats.attempts["unwrap_bare_article_tags_before_other_article_sections"] += 1
        if next_number[0] == article_number:
            return match.group(0)
        stats.applied["unwrap_bare_article_tags_before_other_article_sections"] += 1
        return f"{match.group('inner')}{match.group('gap')}"

    return BARE_ARTICLE_TAG_BEFORE_SECTION_RE.sub(repl, tagged_text), stats
