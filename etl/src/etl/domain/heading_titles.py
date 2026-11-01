"""Space restoration for headings whose number is glued to the title.

Some filings print (or our text extraction yields) article and section headings
with no space between the number and the title: "ARTICLE IVREPRESENTATIONS AND
WARRANTIES", "ARTICLE 4DEFINITIONS", "ARTICLEVII TERMINATION", "Section
5.1Closing". The number then cannot be parsed, so article-order checks skip the
heading and the served title looks broken.

These helpers insert the missing space and change nothing else: the result is
the printed text with one space added between the keyword and the number
and/or between the number and the title.
"""

from __future__ import annotations

import re

# Strict Roman numerals from 1 to 39. Article numbers never go higher in this
# corpus, and the cap keeps "XCLOSING" from reading as XC (90) + "LOSING".
_STRICT_ROMAN_RE = re.compile(r"^(?=[IVX])X{0,3}(?:IX|IV|V?I{0,3})$")
_ROMAN_VALUES = {"I": 1, "V": 5, "X": 10}

# Title words that begin with a Roman-numeral letter. When the letters after
# ARTICLE run together ("VIIINDEMNIFICATION"), the split that leaves one of
# these words wins ("VII" + "INDEMNIFICATION", not "VIII" + "NDEMNIFICATION").
# Taken from the first words of correctly spaced served article titles.
_ROMAN_INITIAL_TITLE_WORDS = frozenset(
    """
    I IMPLEMENTATION IMMIGRATION IN INCOME INCORPORATION INDEBTEDNESS INDEMNIFICATION
    INDEMNIFICATIONS INDEMNIFIED INDEMNITIES INDEMNITY INDEPENDENT INDIVIDUAL INFORMATION
    INITIAL INSOLVENCY INSPECTION INSPECTIONS INSTRUMENTS INSURANCE INTELLECTUAL
    INTENTIONALLY INTERIM INTERNAL INTERPRETATION INTERPRETATIONS INTRODUCTION INVALIDITY
    INVENTORIES INVENTORY INVESTIGATION INVESTMENT INVESTMENTS IPO ISSUANCE ISSUANCES
    ISSUE ISSUES ITEMS IDENTIFICATION ILLEGALITY INTEGRATION INTERPRETIVE INTEREST
    INTERESTS INVESTOR INVESTORS ISSUER ISSUERS INCENTIVE INCENTIVES INDENTURE INTENT
    INTERCREDITOR IMPORTANT INSTRUCTIONS INTERPRETIVE
    VALUATION VALUE VARIATIONS VARIOUS VAT VENDOR VENDORS VOTING VESTING VALIDITY
    VOLUNTARY
    """.split()
)

# Title words that begin with A or B, letters that can also be an article-number
# suffix ("ARTICLE 4A", "ARTICLE IIIB"). "ARTICLE 4ADDITIONAL" splits as
# 4 + ADDITIONAL only when the word is known; "ARTICLE 4ATAXES" stays as printed.
_SUFFIX_INITIAL_TITLE_WORDS = frozenset(
    """
    ACCESS ACCOUNTING ACQUISITION ACTIONS ADDITIONAL ADJUSTMENT ADJUSTMENTS ADMINISTRATIVE
    AGREEMENT AGREEMENTS ALLOCATION AMENDMENT AMENDMENTS ANCILLARY ANTITRUST APPRAISAL
    APPROVALS ARBITRATION ASSETS ASSIGNMENT ASSUMED ASSUMPTION AUTHORITY
    BENEFIT BENEFITS BOARD BOOKS BREAKUP BROKERS BUSINESS BUYER BUYERS BASIC
    """.split()
)

_ARTICLE_GLUE_RE = re.compile(
    r"^(?P<lead>\s*)(?P<kw>ARTICLE|Article)(?P<gap>[ \t ]*)"
    r"(?P<num>\d+|[IVXLCDM]+)(?P<rest>.*)$",
    re.DOTALL,
)
_SECTION_GLUE_RE = re.compile(
    r"^(?P<prefix>\s*(?:(?:Section|SECTION)[ \t ]*)?\d{1,3}\.\d{1,3})"
    r"(?=[A-Z][A-Za-z])"
)
_LEADING_WORD_RE = re.compile(r"[A-Za-z]+")


def _roman_value(token: str) -> int | None:
    if not _STRICT_ROMAN_RE.match(token):
        return None
    total = 0
    prev = 0
    for char in reversed(token):
        value = _ROMAN_VALUES[char]
        if value < prev:
            total -= value
        else:
            total += value
            prev = value
    return total


def _first_word(text: str) -> str:
    match = _LEADING_WORD_RE.match(text)
    return match.group(0) if match else ""


def _split_glued_roman(
    letters: str,
    rest: str,
    expected_number: int | None,
) -> tuple[str, str] | None:
    """Split "IV" + "REPRESENTATIONS..." out of letters + rest.

    `letters` is the run of Roman-numeral letters right after ARTICLE and
    `rest` is everything after it. Returns (numeral, title) or None when the
    heading is not a glued numeral + title.
    """
    glued = letters + rest
    # "IIIA REPRESENTATIONS", "IVB": a lettered article number, not glued.
    if re.match(r"^[A-Z](?![A-Za-z])", rest):
        return None
    # A heading that is a plain word ("ARTICLE INDEMNIFICATION") is not glued.
    if _first_word(glued).upper() in _ROMAN_INITIAL_TITLE_WORDS:
        return None
    candidates: list[tuple[int, str, str, int]] = []
    for cut in range(len(letters), 0, -1):
        numeral = letters[:cut]
        value = _roman_value(numeral)
        if value is None:
            continue
        title = glued[cut:]
        word = _first_word(title)
        # The title must begin with an uppercase word of 3+ letters.
        if len(word) < 3 or not word[0].isupper():
            continue
        if word[0] in "AB" and word.upper() not in _SUFFIX_INITIAL_TITLE_WORDS:
            continue  # "IIIATAXES" may be article IIIA: leave it as printed
        if word[1:].islower():
            if word[0] in "IVX" and word.upper() not in _ROMAN_INITIAL_TITLE_WORDS:
                # "IXdefinitions" is IX + "definitions", not I + "Xdefinitions".
                continue
            rank = 0  # "VIIIIndemnification": capitalized word starts here
        elif word.upper() in _ROMAN_INITIAL_TITLE_WORDS:
            rank = 1  # "VIIINDEMNIFICATION" -> VII + INDEMNIFICATION
        elif word[0] in "IVX":
            # Title would start with a numeral letter but is not a known word
            # ("IIIDENTIFICATION" could be III + DENTIFICATION or II +
            # IDENTIFICATION): leave the heading as printed.
            continue
        elif not word.isupper() and not word[1:].islower():
            continue
        else:
            rank = 2 if value == expected_number else 3
        candidates.append((rank, numeral, title, -cut))
    if not candidates:
        return None
    candidates.sort(key=lambda item: (item[0], item[3]))
    _rank, numeral, title, _cut = candidates[0]
    return numeral, title


def normalize_glued_article_title(title: str, expected_number: int | None = None) -> str:
    """Insert the missing space(s) in a glued article heading.

    "ARTICLE IVREPRESENTATIONS" -> "ARTICLE IV REPRESENTATIONS",
    "ARTICLE 4DEFINITIONS" -> "ARTICLE 4 DEFINITIONS",
    "ARTICLEVII TERMINATION" -> "ARTICLE VII TERMINATION".
    `expected_number` (previous article number + 1) breaks ties between
    otherwise equal Roman splits. Anything else is returned unchanged.
    """
    match = _ARTICLE_GLUE_RE.match(title)
    if match is None:
        return title
    lead, kw, gap, num, rest = (match.group(name) for name in ("lead", "kw", "gap", "num", "rest"))
    rest_is_separate = not rest or not rest[0].isalpha()
    if num.isdigit():
        numeral, title_part = num, rest
        word = _first_word(rest)
        glued_title = len(word) >= 3 and word[0].isupper()
        if glued_title and word[0] in "AB" and word.upper() not in _SUFFIX_INITIAL_TITLE_WORDS:
            # "4ATAXES" may be article 4A: leave it as printed.
            return title
        if not glued_title and not rest_is_separate:
            return title
    elif _roman_value(num) is not None and rest_is_separate:
        numeral, title_part, glued_title = num, rest, False
    else:
        split = _split_glued_roman(num, rest, expected_number)
        if split is None:
            return title
        numeral, title_part = split
        glued_title = True
    if gap and not glued_title:
        return title
    new_gap = gap if gap else " "
    separator = " " if glued_title else ""
    return f"{lead}{kw}{new_gap}{numeral}{separator}{title_part}"


def normalize_glued_section_title(title: str) -> str:
    """Insert the missing space in "Section 5.1Closing" -> "Section 5.1 Closing"."""
    match = _SECTION_GLUE_RE.match(title)
    if match is None:
        return title
    return f"{match.group('prefix')} {title[match.end():]}"


def article_title_number(title: str) -> int | None:
    """Parse the number of a normalized "ARTICLE <n>" heading, if any."""
    match = re.match(r"^\s*ARTICLE\s+(\d+|[IVX]+)\b", title, re.IGNORECASE)
    if match is None:
        return None
    token = match.group(1)
    if token.isdigit():
        return int(token)
    return _roman_value(token.upper())
