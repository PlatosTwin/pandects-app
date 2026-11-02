"""Rejoin ordinals and possessives that an inline-tag boundary split with a space.

The HTML page text joins inline tags with a space, so "15<sup>th</sup>" reads
"15 th" and "Broker<u>’</u><u>s Fees</u>" reads "Broker’ s Fees". The XML builder
rejoins such a split only when the agreement's own HTML confirms it: the joined
form ("15th", "Broker’s") is printed in the raw body pages and the spaced form
("15 th", "Broker’ s", "Broker ’s") is printed nowhere in them. Inline tags are
removed from the raw text without a separator for this check, block tags and
<br> become a space, so a space the reader sees in the source always counts as
printed.
"""
from __future__ import annotations

import html
import re
from collections.abc import Iterable

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_SCRIPT_RE = re.compile(r"<(script|style)\b.*?</\1\s*>", re.DOTALL | re.IGNORECASE)
_BLOCK_TAG_RE = re.compile(
    r"</?(?:p|div|td|th|tr|li|ul|ol|dl|dt|dd|table|tbody|thead|tfoot|caption|h[1-6]|center|"
    r"blockquote|pre|section|article|header|footer|address|form|hr|img|body|html|br)\b[^>]*>",
    re.IGNORECASE,
)
_ANY_TAG_RE = re.compile(r"<[^>]*>")
_INVISIBLE_RE = re.compile(r"[​-‏‪-‮⁠﻿]")
_QUOTE_TRANSLATION = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "′": "'"})

# "15 th", "December 31 st": a number, the space, and an ordinal suffix that ends the word.
_SPLIT_ORDINAL_RE = re.compile(r"(?<![0-9.,])(?P<number>\d{1,4})[ \t]+(?P<suffix>st|nd|rd|th|ST|ND|RD|TH)(?![A-Za-z0-9])")
# "Broker’ s", "BUYER’ S": a word, the apostrophe, the space, and an "s" that ends the word.
_SPLIT_AFTER_QUOTE_RE = re.compile(r"(?P<word>[A-Za-z0-9]+)(?P<quote>[’'])[ \t]+(?P<s>[sS])(?![A-Za-z0-9])")
# "Company ’s": a word, the space, then the apostrophe and an "s" that ends the word.
_SPLIT_BEFORE_QUOTE_RE = re.compile(r"(?P<word>[A-Za-z0-9]+)[ \t]+(?P<quote>[’'])(?P<s>[sS])(?![A-Za-z0-9])")


def raw_inline_plain(raw_html: str) -> str:
    """The visible text of an HTML page with inline tags removed without a separator."""
    text = _COMMENT_RE.sub("", raw_html)
    text = _SCRIPT_RE.sub("", text)
    text = _BLOCK_TAG_RE.sub(" ", text)
    text = _ANY_TAG_RE.sub("", text)
    text = html.unescape(text).replace(" ", " ")
    text = _INVISIBLE_RE.sub("", text)
    text = text.translate(_QUOTE_TRANSLATION)
    return re.sub(r"\s+", " ", text)


def _ordinal_suffix(number: str) -> str:
    value = int(number)
    if 11 <= value % 100 <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")


class InlineSplitEvidence:
    """The raw HTML of an agreement's body pages, used to confirm a rejoin."""

    def __init__(self, raw_pages: Iterable[str]) -> None:
        self._plain = " ".join(raw_inline_plain(raw) for raw in raw_pages if raw)
        self._cache: dict[tuple[str, str], bool] = {}

    def _confirms(self, joined_re: str, spaced_re: str) -> bool:
        key = (joined_re, spaced_re)
        cached = self._cache.get(key)
        if cached is None:
            cached = (
                re.search(joined_re, self._plain) is not None
                and re.search(spaced_re, self._plain) is None
            )
            self._cache[key] = cached
        return cached

    def _join_ordinal(self, match: re.Match[str]) -> str:
        number, suffix = match.group("number"), match.group("suffix")
        if suffix.lower() != _ordinal_suffix(number):
            return match.group(0)
        n, s = re.escape(number), re.escape(suffix)
        if self._confirms(
            rf"(?<![0-9]){n}{s}(?![A-Za-z0-9])",
            rf"(?<![0-9]){n}\s+{s}(?![A-Za-z0-9])",
        ):
            return number + suffix
        return match.group(0)

    def _join_possessive(self, match: re.Match[str]) -> str:
        word, s = re.escape(match.group("word")), re.escape(match.group("s"))
        if self._confirms(
            rf"(?<![A-Za-z0-9]){word}'{s}(?![A-Za-z0-9])",
            rf"(?<![A-Za-z0-9]){word}(?:\s+'\s*|'\s+){s}(?![A-Za-z0-9])",
        ):
            return match.group("word") + match.group("quote") + match.group("s")
        return match.group(0)

    def join(self, text: str) -> str:
        """Rejoin the split ordinals and possessives in text that the raw HTML confirms."""
        if not self._plain or (" " not in text and "\t" not in text):
            return text
        if any(ch.isdigit() for ch in text):
            text = _SPLIT_ORDINAL_RE.sub(self._join_ordinal, text)
        if "’" in text or "'" in text:
            text = _SPLIT_AFTER_QUOTE_RE.sub(self._join_possessive, text)
            text = _SPLIT_BEFORE_QUOTE_RE.sub(self._join_possessive, text)
        return text
