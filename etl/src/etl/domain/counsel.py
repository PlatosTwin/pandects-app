from __future__ import annotations

import re


_MULTI_COUNSEL_SPLIT_RE = re.compile(r"\s*;\s*|\s*\n+\s*|\s+/\s+")
_AND_SUFFIX_CONNECTOR_RE = re.compile(
    r"(?i)\b(?:llp|l\.l\.p\.|pc|p\.c\.|professional corporation|pllc|plc|llc|ltd|lp|l\.p\.|n\.v\.|a\.p\.c\.|corp\.|corporation)\b\s+and\s+"
)
_TRAILING_GEO_PAREN_RE = re.compile(r"\s*\((?:us|u\.s\.|usa|ny|delaware|uk)\)\s*$", re.I)
_TRAILING_SUFFIX_RE = re.compile(
    "".join(
        (
            r"(?i)(?:",
            r"(?:,|\s)+",
            r"(?:",
            r"llp|l\.l\.p\.|limited liability partnership|",
            r"pc|p\.c\.|professional corporation|",
            r"pllc|plc|llc|ltd|",
            r"lp|l\.p\.|",
            r"corp\.?|corporation|",
            r"inc\.?|",
            r"n\.v\.|",
            r"a\.p\.c\.|",
            r"s\.c\.",
            r")",
            r")+\.?\s*$",
        )
    )
)
_CANONICAL_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_CANONICAL_SPACE_RE = re.compile(r"\s+")
_DISPLAY_SPACE_RE = re.compile(r"\s+")
_CANONICAL_STOPWORDS = {
    "llp",
    "pc",
    "professional",
    "corporation",
    "pllc",
    "plc",
    "llc",
    "lp",
    "corp",
    "inc",
    "ltd",
    "nv",
    "apc",
    "sc",
}


def _clean_display_candidate(raw_name: str) -> str:
    cleaned = raw_name.strip().strip(";,")
    if not cleaned:
        return ""
    while True:
        next_cleaned = _TRAILING_GEO_PAREN_RE.sub("", cleaned).strip()
        next_cleaned = _TRAILING_SUFFIX_RE.sub("", next_cleaned).strip().strip(",;")
        if next_cleaned == cleaned:
            break
        cleaned = next_cleaned
    cleaned = re.sub(r"\s+and\s+", " & ", cleaned, flags=re.I)
    cleaned = _DISPLAY_SPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def _split_after_legal_suffix_connector(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for match in _AND_SUFFIX_CONNECTOR_RE.finditer(value):
        split_at = value[: match.end()].lower().rfind(" and ")
        if split_at <= start:
            continue
        parts.append(value[start:split_at])
        start = match.end()
    parts.append(value[start:])
    return parts


def split_counsel_names(value: object | None) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, str):
        raise TypeError("Counsel value must be a string or null.")
    cleaned = value.strip()
    if not cleaned:
        return []

    segments: list[str] = []
    for chunk in _MULTI_COUNSEL_SPLIT_RE.split(cleaned):
        chunk = chunk.strip().strip(";,")
        if not chunk:
            continue
        for subchunk in _split_after_legal_suffix_connector(chunk):
            normalized = subchunk.strip().strip(";,")
            if normalized:
                segments.append(normalized)
    return segments


def canonicalize_counsel_name(value: object | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Counsel value must be a string or null.")
    display_candidate = _clean_display_candidate(value)
    if not display_candidate:
        return None

    canonical = display_candidate.lower()
    canonical = canonical.replace("&", " and ")
    canonical = _CANONICAL_PUNCT_RE.sub(" ", canonical)
    canonical = canonical.replace(" and ", " ")
    canonical = _CANONICAL_SPACE_RE.sub(" ", canonical).strip()
    tokens = [token for token in canonical.split(" ") if token and token not in _CANONICAL_STOPWORDS]
    if not tokens:
        return None
    return " ".join(tokens)


def format_counsel_display_name(value: object | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Counsel value must be a string or null.")
    display_candidate = _clean_display_candidate(value)
    return display_candidate or None
