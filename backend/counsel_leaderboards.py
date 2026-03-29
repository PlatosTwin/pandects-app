from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import re
from typing import Any


_MULTI_COUNSEL_SPLIT_RE = re.compile(r"\s*;\s*|\s*\n+\s*|\s+/\s+")
_AND_SUFFIX_CONNECTOR_RE = re.compile(
    r"(?i)\b(?:llp|l\.l\.p\.|pc|p\.c\.|professional corporation|pllc|plc|llc|ltd|lp|l\.p\.|n\.v\.|a\.p\.c\.|corp\.|corporation)\b\s+and\s+"
)
_TRAILING_GEO_PAREN_RE = re.compile(r"\s*\((?:us|u\.s\.|usa|ny|delaware|uk)\)\s*$", re.I)
_TRAILING_SUFFIX_RE = re.compile(
    r"(?i)(?:"
    r"(?:,|\s)+"
    r"(?:"
    r"llp|l\.l\.p\.|limited liability partnership|"
    r"pc|p\.c\.|professional corporation|"
    r"pllc|plc|llc|ltd|"
    r"lp|l\.p\.|"
    r"corp\.?|corporation|"
    r"inc\.?|"
    r"n\.v\.|"
    r"a\.p\.c\.|"
    r"s\.c\."
    r")"
    r")+\.?\s*$"
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
    "corporation",
    "inc",
    "ltd",
    "nv",
    "apc",
    "sc",
}


@dataclass
class _FirmAggregate:
    display_candidates: Counter[str] = field(default_factory=Counter)
    deal_count: int = 0
    total_transaction_value: Decimal = field(default_factory=lambda: Decimal("0"))
    yearly: dict[int, dict[str, Decimal | int]] = field(default_factory=dict)


def _decimal_from_value(value: object | None) -> Decimal:
    if value is None:
        return Decimal("0")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, (int, float, str)):
        try:
            return Decimal(str(value))
        except (InvalidOperation, ValueError):
            return Decimal("0")
    return Decimal("0")


def _year_from_value(value: object | None) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, datetime):
        return value.year
    if isinstance(value, date):
        return value.year
    if isinstance(value, str):
        stripped = value.strip()
        if len(stripped) >= 4 and stripped[:4].isdigit():
            return int(stripped[:4])
    return None


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
        subchunks = _split_after_legal_suffix_connector(chunk)
        for subchunk in subchunks:
            normalized = subchunk.strip().strip(";,")
            if normalized:
                segments.append(normalized)
    return segments


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


def build_counsel_leaderboards(
    records: list[dict[str, Any]],
    *,
    limit: int = 15,
) -> dict[str, object]:
    assignments: list[dict[str, Any]] = []
    for record in records:
        year = _year_from_value(record.get("filing_date"))
        if year is None:
            continue
        transaction_value = _decimal_from_value(record.get("transaction_price_total"))
        side_to_value = {
            "buy_side": record.get("acquirer_counsel") or record.get("acquirer_counsel_normalized"),
            "sell_side": record.get("target_counsel") or record.get("target_counsel_normalized"),
        }

        for side, counsel_value in side_to_value.items():
            canonical_to_display: dict[str, str] = {}
            for raw_name in split_counsel_names(counsel_value):
                canonical_name = canonicalize_counsel_name(raw_name)
                display_name = format_counsel_display_name(raw_name)
                if canonical_name is None or display_name is None:
                    continue
                canonical_to_display.setdefault(canonical_name, display_name)

            for canonical_name, display_name in canonical_to_display.items():
                assignments.append(
                    {
                        "side": side,
                        "counsel_key": canonical_name,
                        "counsel": display_name,
                        "filing_date": year,
                        "transaction_price_total": transaction_value,
                    }
                )

    return build_counsel_leaderboards_from_assignments(assignments, limit=limit)


def build_counsel_leaderboards_from_assignments(
    assignments: list[dict[str, Any]],
    *,
    limit: int = 15,
) -> dict[str, object]:
    sides: dict[str, dict[str, _FirmAggregate]] = {
        "buy_side": {},
        "sell_side": {},
    }

    for assignment in assignments:
        side = str(assignment.get("side") or "").strip()
        if side not in sides:
            continue
        year = _year_from_value(assignment.get("filing_date"))
        if year is None:
            continue
        counsel_key = str(assignment.get("counsel_key") or assignment.get("counsel") or "").strip()
        counsel_name = str(assignment.get("counsel") or "").strip()
        if not counsel_key or not counsel_name:
            continue
        transaction_value = _decimal_from_value(assignment.get("transaction_price_total"))
        aggregate = sides[side].setdefault(counsel_key, _FirmAggregate())
        aggregate.display_candidates[counsel_name] += 1
        aggregate.deal_count += 1
        aggregate.total_transaction_value += transaction_value
        year_bucket = aggregate.yearly.setdefault(
            year,
            {
                "deal_count": 0,
                "total_transaction_value": Decimal("0"),
            },
        )
        year_bucket["deal_count"] = int(year_bucket["deal_count"]) + 1
        year_bucket["total_transaction_value"] = (
            _decimal_from_value(year_bucket["total_transaction_value"])
            + transaction_value
        )

    def to_payload(side_aggregates: dict[str, _FirmAggregate]) -> dict[str, object]:
        rows: list[dict[str, object]] = []
        for aggregate in side_aggregates.values():
            display_name = min(
                aggregate.display_candidates.items(),
                key=lambda item: (-item[1], len(item[0]), item[0].lower()),
            )[0]
            years = [
                {
                    "year": year,
                    "deal_count": int(bucket["deal_count"]),
                    "total_transaction_value": float(
                        _decimal_from_value(bucket["total_transaction_value"])
                    ),
                }
                for year, bucket in sorted(aggregate.yearly.items())
            ]
            rows.append(
                {
                    "counsel": display_name,
                    "deal_count": aggregate.deal_count,
                    "total_transaction_value": float(aggregate.total_transaction_value),
                    "years": years,
                }
            )

        top_by_count = sorted(
            rows,
            key=lambda item: (
                -int(item["deal_count"]),
                -float(item["total_transaction_value"]),
                str(item["counsel"]).lower(),
            ),
        )[:limit]
        top_by_value = sorted(
            rows,
            key=lambda item: (
                -float(item["total_transaction_value"]),
                -int(item["deal_count"]),
                str(item["counsel"]).lower(),
            ),
        )[:limit]
        return {
            "top_by_count": top_by_count,
            "top_by_value": top_by_value,
        }

    return {
        "buy_side": to_payload(sides["buy_side"]),
        "sell_side": to_payload(sides["sell_side"]),
    }
