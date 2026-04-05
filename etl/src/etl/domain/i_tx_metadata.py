# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
from __future__ import annotations

import copy
import json
import re
from decimal import Decimal
from datetime import date, datetime
from typing import Any, Dict, Optional

DEAL_TYPE_ALLOWED = (
    "merger",
    "stock_acquisition",
    "asset_acquisition",
    "tender_offer",
    "membership_interest_purchase",
)

WEB_SEARCH_CITABLE_FIELDS = (
    "consideration_type",
    "purchase_price.cash",
    "purchase_price.stock",
    "purchase_price.assets",
    "target_public",
    "acquirer_public",
    "target_pe",
    "acquirer_pe",
    "target_industry",
    "acquirer_industry",
    "announce_date",
    "close_date",
    "deal_status",
    "attitude",
    "purpose",
)

CITABLE_FIELDS_WITH_DEAL_TYPE = WEB_SEARCH_CITABLE_FIELDS + ("deal_type",)

NON_CORE_CITATION_OPTIONAL_FIELDS = (
    "announce_date",
    "close_date",
    "target_pe",
    "acquirer_pe",
    "target_industry",
    "acquirer_industry",
    "deal_status",
    "attitude",
    "purpose",
)

TOKEN_USAGE_REQUIRED_FIELDS = ("input_tokens", "output_tokens", "total_tokens")
RETRY_FOCUS_FIELD_ORDER = (
    "consideration_type",
    "purchase_price.cash",
    "purchase_price.stock",
    "purchase_price.assets",
    "target_public",
    "acquirer_public",
    "announce_date",
    "close_date",
    "deal_status",
    "target_pe",
    "acquirer_pe",
    "target_industry",
    "acquirer_industry",
    "attitude",
    "purpose",
)


def json_schema_transaction_metadata() -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "consideration_type": {
                "type": "string",
                "enum": ["all_cash", "all_stock", "mixed", "unknown"],
            },
            "purchase_price": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    # USD amounts. Use 0 only when confidently absent; use null when unknown.
                    "cash": {"type": ["number", "null"], "minimum": 0},
                    "stock": {"type": ["number", "null"], "minimum": 0},
                    "assets": {"type": ["number", "null"], "minimum": 0},
                },
                "required": ["cash", "stock", "assets"],
            },
            "target_public": {"type": ["boolean", "null"]},
            "acquirer_public": {"type": ["boolean", "null"]},
            "target_pe": {"type": ["boolean", "null"]},
            "acquirer_pe": {"type": ["boolean", "null"]},
            "target_industry": {"type": ["string", "null"]},
            "acquirer_industry": {"type": ["string", "null"]},
            # Deal timeline
            "announce_date": {
                "type": ["string", "null"],
                "pattern": r"^\d{4}-\d{2}-\d{2}$",
            },
            "close_date": {
                "type": ["string", "null"],
                "pattern": r"^\d{4}-\d{2}-\d{2}$",
            },
            "deal_status": {
                "anyOf": [
                    {"type": "string", "enum": ["pending", "complete", "cancelled", "unknown"]},
                    {"type": "null"},
                ]
            },
            # Deal characteristics
            "attitude": {
                "anyOf": [
                    {"type": "string", "enum": ["friendly", "hostile", "unsolicited"]},
                    {"type": "null"},
                ]
            },
            "deal_type": {
                "anyOf": [
                    {
                        "type": "string",
                        "enum": list(DEAL_TYPE_ALLOWED),
                    },
                    {"type": "null"},
                ]
            },
            "purpose": {
                "anyOf": [
                    {"type": "string", "enum": ["strategic", "financial"]},
                    {"type": "null"},
                ]
            },
            "metadata_sources": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "field": {
                                    "type": "string",
                                    "enum": list(CITABLE_FIELDS_WITH_DEAL_TYPE),
                                },
                                "url": {"type": "string"},
                                "source_type": {"type": "string", "minLength": 1},
                                "published_at": {
                                    "type": ["string", "null"],
                                    "pattern": r"^\d{4}-\d{2}-\d{2}$",
                                },
                                "locator": {"type": ["string", "null"]},
                                "excerpt": {"type": "string", "minLength": 1},
                            },
                            "required": ["field", "url", "source_type", "published_at", "locator", "excerpt"],
                        },
                    },
                    "notes": {"type": ["string", "null"]},
                },
                "required": ["citations", "notes"],
            },
        },
        "required": [
            "consideration_type",
            "purchase_price",
            "target_public",
            "acquirer_public",
            "target_pe",
            "acquirer_pe",
            "target_industry",
            "acquirer_industry",
            "announce_date",
            "close_date",
            "deal_status",
            "attitude",
            "deal_type",
            "purpose",
            "metadata_sources",
        ],
    }


def json_schema_transaction_metadata_web_search_only() -> Dict[str, Any]:
    """Schema for web-search mode: same as full but omit deal_type (collected offline)."""
    full = copy.deepcopy(json_schema_transaction_metadata())
    props = {k: v for k, v in full["properties"].items() if k not in ("deal_type",)}
    required = [k for k in full["required"] if k != "deal_type"]
    # Remove deal_type from metadata_sources.citations.field enum if present
    if "metadata_sources" in props and "properties" in props["metadata_sources"]:
        cites = props["metadata_sources"]["properties"].get("citations", {})
        if "items" in cites and "properties" in cites["items"]:
            field_prop = cites["items"]["properties"].get("field", {})
            if "enum" in field_prop:
                field_prop["enum"] = [x for x in field_prop["enum"] if x != "deal_type"]
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": props,
        "required": required,
    }


TX_METADATA_INSTRUCTIONS_WEB_SEARCH = (
    "You are an expert M&A research analyst. For the provided transaction details, "
    "and using trusted sources only (via the websearch tool), find: "
    "1) consideration_type using EXACTLY one of: 'all_cash', 'all_stock', 'mixed', 'unknown'. "
    "If any asset/non-cash/non-stock component is present, classify as 'mixed'. "
    "2) the purchase price components (USD), without accounting for any assumed debt, notes, etc. "
    "(i.e., equity value / consideration paid to sellers, not enterprise value). "
    "Return full USD amounts (e.g., $4.18 billion -> 4180000000), not shorthand units; "
    "Never convert non-USD prices into USD yourself. "
    "If the deal value is stated only in a non-USD currency, set all purchase_price fields to null and explain that briefly in `metadata_sources.notes`. "
    "If sources only provide enterprise value / transaction value including assumed debt and you cannot isolate equity consideration, use null for purchase price. "
    "3) whether the target was public or private at announcement. Public means the target itself is publicly traded (listed or OTC) or is an SEC reporting issuer; "
    "if the target is a privately held subsidiary of a public parent, treat the target as private unless the target itself is publicly traded/SEC-reporting. "
    "If you cannot determine, use null. "
    "4) whether the acquirer was public or private at announcement. Public means the acquirer itself is publicly traded (listed or OTC) or is an SEC reporting issuer; "
    "if the acquirer is a privately held acquisition vehicle controlled by a public/PE parent, treat the acquirer as private unless it is itself publicly traded/SEC-reporting. "
    "If you cannot determine, use null. "
    "5) whether the target was owned by a private equity shop; "
    "6) whether the acquirer was a private equity shop; "
    "7) the target industry using the NAICS subsector code as digits only (prefer 3-digit subsector; if only 2-digit sector is available, return that; otherwise null); "
    "8) the acquirer industry using the NAICS subsector code as digits only (prefer 3-digit subsector; if only 2-digit sector is available, return that; otherwise null); "
    "9) announce_date: the date the deal was publicly announced (YYYY-MM-DD); "
    "10) close_date: the date the deal closed (YYYY-MM-DD), or null if not completed; "
    "11) deal_status: 'pending', 'complete', or 'cancelled' (use 'unknown' if unclear); "
    "12) attitude: 'friendly', 'hostile', or 'unsolicited' (use null if unclear); "
    "13) purpose: 'strategic' or 'financial' (use null if unclear). "
    "Do not extract or return target, acquirer, or deal_type (those are provided separately). "
    "For fields where you don't know the answer, use null. "
    "Do a sanity check before completing, and dig deeper if need be—e.g., "
    "if you think total consideration is $19, that is probably wrong, as it's too small; "
    "if you think consideration_type is mixed, at least one purchase_price component should be non-zero unless price is unknown; "
    "etc. "
    "Always start from the provided SEC filing URL when available, then expand web search as needed. "
    "Prefer primary sources in this order: SEC filing pages/documents, company/investor press releases, "
    "then reputable financial wires; avoid tertiary aggregators when a primary source is available. "
    "Use no more than 12 web searches for this transaction. "
    "Stop searching once every non-null output field has citation coverage and no unresolved source conflicts remain. "
    "Avoid redundant lookups; do not re-run materially identical searches unless needed to resolve a conflict. "
    "For sources, populate `metadata_sources.citations` with one evidence record per supported field. "
    "Each evidence record must include the field name, URL, source_type, published_at when available, locator when available, and a short supporting excerpt. "
    "Every non-null output field must appear in at least one citation record's `field`. "
    "When sources conflict, prefer SEC/company-primary sources and explain conflict resolution in notes. "
    "If you couldn't find a field, do not guess; say why briefly in `metadata_sources.notes`."
)

TX_METADATA_RETRY_INSTRUCTIONS_WEB_SEARCH = (
    "This is a retry for partially-complete metadata. "
    "Prioritize the explicitly listed focus fields first. "
    "Treat the listed known trusted metadata as locked context and do not spend search budget re-finding it. "
    "If you find stronger contradictory evidence for a locked field, you may revise it, but only if you cite that stronger evidence. "
    "If you do not find stronger contradictory evidence, echo the locked value back unchanged in the JSON output. "
    "Do not return null for a locked field unless you have stronger evidence that the prior value is wrong and you explain that in notes."
)


def _filing_date_to_str(filing_date: Any) -> str:
    if filing_date is None:
        return "unknown"
    if isinstance(filing_date, (datetime, date)):
        return filing_date.isoformat()
    if isinstance(filing_date, str):
        return filing_date
    raise TypeError(f"Unexpected filing_date type: {type(filing_date).__name__}")


def _db_consideration_to_prompt(value: object | None) -> Optional[str]:
    if value is None:
        return None
    if value == "cash":
        return "all_cash"
    if value == "stock":
        return "all_stock"
    if value == "mixed":
        return "mixed"
    raise TypeError(f"Unexpected stored consideration value: {value!r}")


def _db_type_to_public_flag(value: object | None) -> Optional[bool]:
    if value is None:
        return None
    if value == "public":
        return True
    if value == "private":
        return False
    raise TypeError(f"Unexpected stored entity type: {value!r}")


def _string_or_none(value: object | None) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if not isinstance(value, str):
        raise TypeError(f"Expected string or null, got {type(value).__name__}.")
    stripped = value.strip()
    return stripped or None


def _boolish_or_none(value: object | None) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value in (0, 1):
            return bool(value)
    raise TypeError(f"Expected boolean-like value or null, got {type(value).__name__}.")


def _numeric_or_none(value: object | None) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("Expected numeric value or null, got bool.")
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    raise TypeError(f"Expected numeric value or null, got {type(value).__name__}.")


def _parse_metadata_sources_blob(raw_value: object | None) -> Dict[str, Any]:
    empty_payload = {"metadata_sources": {"citations": [], "notes": None}}
    if raw_value is None:
        return empty_payload
    if isinstance(raw_value, str):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError:
            return empty_payload
    elif isinstance(raw_value, dict):
        parsed = copy.deepcopy(raw_value)
    else:
        return empty_payload
    if not isinstance(parsed, dict):
        return empty_payload
    metadata_sources = parsed.get("metadata_sources")
    if metadata_sources is None:
        # Backward compatibility for legacy flat payloads shaped like
        # {"citations": [...], "notes": "..."}.
        if "citations" in parsed or "notes" in parsed:
            metadata_sources = {
                "citations": parsed.get("citations"),
                "notes": parsed.get("notes"),
            }
        else:
            metadata_sources = {"citations": [], "notes": None}
    if not isinstance(metadata_sources, dict):
        return empty_payload
    citations = metadata_sources.get("citations")
    if citations is None or not isinstance(citations, list):
        metadata_sources["citations"] = []
    if "notes" not in metadata_sources or (metadata_sources.get("notes") is not None and not isinstance(metadata_sources.get("notes"), str)):
        metadata_sources["notes"] = None
    return {
        "metadata_sources": metadata_sources,
        "metadata_run_stats": parsed.get("metadata_run_stats"),
    }


def _parse_metadata_uncited_fields(raw_value: object | None) -> set[str]:
    if raw_value is None:
        return set()
    if isinstance(raw_value, str):
        parsed = json.loads(raw_value)
    elif isinstance(raw_value, list):
        parsed = raw_value
    else:
        raise TypeError("metadata_uncited_fields must be a JSON string, list, or null.")
    if not isinstance(parsed, list):
        raise TypeError("metadata_uncited_fields must decode to a list.")
    return {str(value) for value in parsed if isinstance(value, str)}


def _existing_web_search_metadata_obj(agreement: Dict[str, Any]) -> Dict[str, Any]:
    metadata_payload = _parse_metadata_sources_blob(agreement.get("metadata_sources"))
    return {
        "consideration_type": _db_consideration_to_prompt(agreement.get("transaction_consideration")),
        "purchase_price": {
            "cash": _numeric_or_none(agreement.get("transaction_price_cash")),
            "stock": _numeric_or_none(agreement.get("transaction_price_stock")),
            "assets": _numeric_or_none(agreement.get("transaction_price_assets")),
        },
        "target_public": _db_type_to_public_flag(agreement.get("target_type")),
        "acquirer_public": _db_type_to_public_flag(agreement.get("acquirer_type")),
        "target_pe": _boolish_or_none(agreement.get("target_pe")),
        "acquirer_pe": _boolish_or_none(agreement.get("acquirer_pe")),
        "target_industry": _string_or_none(agreement.get("target_industry")),
        "acquirer_industry": _string_or_none(agreement.get("acquirer_industry")),
        "announce_date": _string_or_none(agreement.get("announce_date")),
        "close_date": _string_or_none(agreement.get("close_date")),
        "deal_status": _string_or_none(agreement.get("deal_status")),
        "attitude": _string_or_none(agreement.get("attitude")),
        "purpose": _string_or_none(agreement.get("purpose")),
        "metadata_sources": metadata_payload["metadata_sources"],
        "metadata_run_stats": metadata_payload.get("metadata_run_stats"),
    }


def _get_output_field_value(tx_metadata_obj: Dict[str, Any], field_name: str) -> Any:
    if field_name.startswith("purchase_price."):
        price = tx_metadata_obj.get("purchase_price")
        if not isinstance(price, dict):
            return None
        return price.get(field_name.split(".", 1)[1])
    return tx_metadata_obj.get(field_name)


def _set_output_field_value(tx_metadata_obj: Dict[str, Any], field_name: str, value: Any) -> None:
    if field_name.startswith("purchase_price."):
        price = tx_metadata_obj.setdefault("purchase_price", {})
        if not isinstance(price, dict):
            raise TypeError("purchase_price must be an object.")
        price[field_name.split(".", 1)[1]] = value
        return
    tx_metadata_obj[field_name] = value


def _supported_retry_focus_fields(agreement: Dict[str, Any]) -> list[str]:
    focus_fields: set[str] = set()
    uncited_fields = _parse_metadata_uncited_fields(agreement.get("metadata_uncited_fields"))

    consideration = _string_or_none(agreement.get("transaction_consideration"))
    if consideration is None:
        focus_fields.add("consideration_type")
    if _string_or_none(agreement.get("target_type")) is None:
        focus_fields.add("target_public")
    if _string_or_none(agreement.get("acquirer_type")) is None:
        focus_fields.add("acquirer_public")

    cash = agreement.get("transaction_price_cash")
    stock = agreement.get("transaction_price_stock")
    assets = agreement.get("transaction_price_assets")
    total = agreement.get("transaction_price_total")
    present_components = sum(value is not None for value in (cash, stock, assets))
    if consideration == "cash" and (cash is None or total is None):
        focus_fields.add("purchase_price.cash")
    elif consideration == "stock" and (stock is None or total is None):
        focus_fields.add("purchase_price.stock")
    elif consideration == "mixed" and (present_components < 2 or total is None):
        if cash is None:
            focus_fields.add("purchase_price.cash")
        if stock is None:
            focus_fields.add("purchase_price.stock")
        if assets is None:
            focus_fields.add("purchase_price.assets")

    announce_date = _string_or_none(agreement.get("announce_date"))
    close_date = _string_or_none(agreement.get("close_date"))
    deal_status = _string_or_none(agreement.get("deal_status"))
    if announce_date is None or "announce_date" in uncited_fields:
        focus_fields.add("announce_date")
    if close_date is None and deal_status == "complete":
        focus_fields.update({"close_date", "deal_status"})
    elif close_date is not None and deal_status is None:
        focus_fields.add("deal_status")
    for field_name in uncited_fields:
        if field_name in RETRY_FOCUS_FIELD_ORDER:
            focus_fields.add(field_name)

    return [field for field in RETRY_FOCUS_FIELD_ORDER if field in focus_fields]


def build_web_search_retry_context(agreement: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if int(agreement.get("initial_metadata_pass", 1) or 0) == 1:
        return None
    existing_obj = _existing_web_search_metadata_obj(agreement)
    known_values = {
        field_name: _get_output_field_value(existing_obj, field_name)
        for field_name in RETRY_FOCUS_FIELD_ORDER
        if _get_output_field_value(existing_obj, field_name) is not None
    }
    focus_fields = _supported_retry_focus_fields(agreement)
    return {
        "known_values": known_values,
        "focus_fields": focus_fields,
        "revisable_fields": list(known_values.keys()),
        "retry_reason": "Agreement was previously enriched but still has unresolved web metadata fields.",
    }


def _format_retry_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def _dedupe_citations(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[object, object, object, object, object, object]] = set()
    for citation in citations:
        key = (
            citation.get("field"),
            citation.get("url"),
            citation.get("source_type"),
            citation.get("published_at"),
            citation.get("locator"),
            citation.get("excerpt"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(citation)
    return deduped


def build_tx_metadata_request_body_web_search_only(
    agreement: Dict[str, Any], *, model: str, retry_context: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build web-search request body for fields sourced online (deal_type handled offline)."""
    schema = json_schema_transaction_metadata_web_search_only()
    target_raw = agreement.get("target")
    acquirer_raw = agreement.get("acquirer")
    target = target_raw.strip() if isinstance(target_raw, str) else ""
    acquirer = acquirer_raw.strip() if isinstance(acquirer_raw, str) else ""
    sec_url_raw = agreement.get("url")
    if sec_url_raw is None:
        sec_url = ""
    elif isinstance(sec_url_raw, str):
        sec_url = sec_url_raw.strip()
    else:
        raise TypeError("agreement.url must be a string or null.")
    filing_date_str = _filing_date_to_str(agreement.get("filing_date"))
    has_url = bool(sec_url)

    context_lines = ["Known transaction context:"]
    if acquirer:
        context_lines.append(f"- acquirer: {acquirer}")
    if target:
        context_lines.append(f"- target: {target}")
    context_lines.append(f"- sec_filing_date: {filing_date_str}")
    if has_url:
        context_lines.append(f"- sec_filing_url: {sec_url}")
    instructions = TX_METADATA_INSTRUCTIONS_WEB_SEARCH
    if retry_context is not None:
        instructions = f"{TX_METADATA_INSTRUCTIONS_WEB_SEARCH} {TX_METADATA_RETRY_INSTRUCTIONS_WEB_SEARCH}"
        focus_fields = retry_context.get("focus_fields") or []
        known_values = retry_context.get("known_values") or {}
        revisable_fields = retry_context.get("revisable_fields") or []
        retry_reason = retry_context.get("retry_reason")
        context_lines.append("")
        context_lines.append("Retry context:")
        if isinstance(retry_reason, str) and retry_reason:
            context_lines.append(f"- retry_reason: {retry_reason}")
        if focus_fields:
            context_lines.append("- focus_fields: " + ", ".join(str(field) for field in focus_fields))
        if revisable_fields:
            context_lines.append("- revisable_fields_if_stronger_evidence: " + ", ".join(str(field) for field in revisable_fields))
        if known_values:
            context_lines.append("- known_trusted_metadata:")
            for field_name in RETRY_FOCUS_FIELD_ORDER:
                if field_name not in known_values:
                    continue
                context_lines.append(f"  - {field_name}: {_format_retry_value(known_values[field_name])}")
    input_text = "\n".join(context_lines)

    return {
        "model": model,
        "tools": [{"type": "web_search"}],
        "instructions": instructions,
        "input": input_text,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "transaction_metadata_web_search",
                "strict": True,
                "schema": schema,
            }
        },
    }


REQUIRED_KEYS_WEB_SEARCH = {
    "consideration_type",
    "purchase_price",
    "target_public",
    "acquirer_public",
    "target_pe",
    "acquirer_pe",
    "target_industry",
    "acquirer_industry",
    "announce_date",
    "close_date",
    "deal_status",
    "attitude",
    "purpose",
    "metadata_sources",
}


def parse_tx_metadata_response_text_web_search(raw_text: str) -> Dict[str, Any]:
    """Parse web-search response (schema omits deal_type)."""
    obj = json.loads(raw_text)
    if not isinstance(obj, dict):
        raise ValueError("Response JSON is not an object.")
    if not REQUIRED_KEYS_WEB_SEARCH.issubset(obj.keys()):
        raise ValueError("Missing required keys in response JSON.")
    return obj


def build_web_search_runtime_metadata(
    *,
    response_usage: Dict[str, Any],
    search_count: int = 0,
) -> Dict[str, Any]:
    token_usage: Dict[str, int] = {}
    for key in TOKEN_USAGE_REQUIRED_FIELDS:
        value = response_usage.get(key)
        if not isinstance(value, int):
            raise TypeError(f"response_usage.{key} must be an integer.")
        if value < 0:
            raise ValueError(f"response_usage.{key} must be >= 0.")
        token_usage[key] = value
    if search_count < 0:
        raise ValueError("search_count must be >= 0.")
    return {"token_usage": token_usage, "search_count": search_count}


def merge_retry_web_search_response(
    *,
    agreement: Dict[str, Any],
    tx_metadata_obj: Dict[str, Any],
) -> Dict[str, Any]:
    retry_context = build_web_search_retry_context(agreement)
    if retry_context is None:
        return tx_metadata_obj

    existing_obj = _existing_web_search_metadata_obj(agreement)
    merged_obj = copy.deepcopy(tx_metadata_obj)
    existing_sources = existing_obj.get("metadata_sources")
    new_sources = merged_obj.get("metadata_sources")
    if not isinstance(existing_sources, dict):
        existing_sources = {"citations": [], "notes": None}
    if not isinstance(new_sources, dict):
        raise TypeError("metadata_sources must be an object.")
    existing_citations = existing_sources.get("citations")
    new_citations = new_sources.get("citations")
    if not isinstance(existing_citations, list):
        existing_citations = []
    if not isinstance(new_citations, list):
        raise TypeError("metadata_sources.citations must be an array.")

    new_cited_fields = {
        str(citation.get("field"))
        for citation in new_citations
        if isinstance(citation, dict) and isinstance(citation.get("field"), str)
    }
    focus_fields = set(retry_context.get("focus_fields") or [])
    preserved_fields: set[str] = set()

    for field_name in RETRY_FOCUS_FIELD_ORDER:
        existing_value = _get_output_field_value(existing_obj, field_name)
        incoming_value = _get_output_field_value(merged_obj, field_name)
        if existing_value is None:
            continue
        if field_name in focus_fields:
            if incoming_value is None or field_name not in new_cited_fields:
                _set_output_field_value(merged_obj, field_name, existing_value)
                preserved_fields.add(field_name)
            continue
        if incoming_value is None or incoming_value == existing_value or field_name not in new_cited_fields:
            _set_output_field_value(merged_obj, field_name, existing_value)
            preserved_fields.add(field_name)

    merged_citations = list(new_citations)
    for citation in existing_citations:
        if not isinstance(citation, dict):
            continue
        field_name = citation.get("field")
        if field_name not in preserved_fields:
            continue
        merged_citations.append(copy.deepcopy(citation))

    merged_notes = new_sources.get("notes")
    if (merged_notes is None or (isinstance(merged_notes, str) and not merged_notes.strip())) and isinstance(existing_sources.get("notes"), str):
        merged_notes = existing_sources.get("notes")
    merged_obj["metadata_sources"] = {
        "citations": _dedupe_citations(merged_citations),
        "notes": merged_notes,
    }
    return merged_obj


def _parse_optional_date_like(value: object | None, *, field_name: str) -> Optional[date]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        s = value.strip()
        m = re.match(r"^(\d{4}-\d{2}-\d{2})$", s)
        if not m:
            raise TypeError(f"{field_name} must be a YYYY-MM-DD string/date/datetime or null.")
        try:
            return datetime.strptime(m.group(1), "%Y-%m-%d").date()
        except ValueError as exc:
            raise TypeError(
                f"{field_name} must be a valid calendar date in YYYY-MM-DD format."
            ) from exc
    raise TypeError(f"{field_name} must be a YYYY-MM-DD string/date/datetime or null.")


def _date_years_ago(anchor: date, years: int) -> date:
    if years < 0:
        raise ValueError("years must be non-negative.")
    try:
        return anchor.replace(year=anchor.year - years)
    except ValueError:
        # Handle leap day edge case.
        return anchor.replace(year=anchor.year - years, month=2, day=28)


def _sanitize_web_search_metadata_obj(
    tx_metadata_obj: Dict[str, Any],
    *,
    filing_date: object | None,
    pending_max_age_years: int,
) -> Dict[str, Any]:
    sanitized = copy.deepcopy(tx_metadata_obj)
    metadata_sources = sanitized.get("metadata_sources")
    if isinstance(metadata_sources, dict):
        citations = metadata_sources.get("citations")
        if isinstance(citations, list):
            for citation in citations:
                if not isinstance(citation, dict):
                    continue
                published_at = citation.get("published_at")
                if published_at is None:
                    continue
                try:
                    _ = _parse_optional_date_like(
                        published_at,
                        field_name="metadata_sources.citations[].published_at",
                    )
                except (TypeError, ValueError):
                    citation["published_at"] = None

    def _safe_date_string(field_name: str) -> Optional[str]:
        raw_value = sanitized.get(field_name)
        if raw_value is None:
            return None
        try:
            parsed = _parse_optional_date_like(raw_value, field_name=field_name)
        except (TypeError, ValueError):
            sanitized[field_name] = None
            return None
        return parsed.isoformat() if parsed is not None else None

    announce_date = _safe_date_string("announce_date")
    close_date = _safe_date_string("close_date")
    deal_status = sanitized.get("deal_status")
    if not isinstance(deal_status, str):
        return sanitized

    if announce_date is not None and close_date is not None and close_date < announce_date:
        sanitized["close_date"] = None
        close_date = None
    if deal_status == "complete" and close_date is None:
        sanitized["deal_status"] = None
        return sanitized
    if deal_status == "pending" and close_date is not None:
        sanitized["close_date"] = None
        return sanitized
    if deal_status == "pending":
        try:
            filing_date_obj = _parse_optional_date_like(filing_date, field_name="filing_date")
        except (TypeError, ValueError):
            filing_date_obj = None
        if filing_date_obj is not None:
            cutoff = _date_years_ago(date.today(), pending_max_age_years)
            if filing_date_obj <= cutoff:
                sanitized["deal_status"] = None
    return sanitized


# --- Offline mode: target, acquirer, deal_type from document only ---

def json_schema_offline_metadata() -> Dict[str, Any]:
    """JSON schema for offline extraction: target, acquirer, deal_type only."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target": {"type": ["string", "null"]},
            "acquirer": {"type": ["string", "null"]},
            "deal_type": {
                "anyOf": [
                    {"type": "string", "enum": list(DEAL_TYPE_ALLOWED)},
                    {"type": "null"},
                ]
            },
        },
        "required": ["target", "acquirer", "deal_type"],
    }


TX_METADATA_OFFLINE_INSTRUCTIONS = (
    "You are an expert at reading M&A agreement documents. "
    "Using ONLY the provided document text (no external knowledge or web search), extract: "
    "1) target: the full legal or customary name of the company being acquired (the target). Use null if not clearly stated. "
    "2) acquirer: the full legal or customary name of the company acquiring (the acquirer/buyer). Use null if not clearly stated. "
    "3) deal_type: exactly one of 'merger', 'stock_acquisition', 'asset_acquisition', 'tender_offer', 'membership_interest_purchase', or null if unclear. "
    "Be precise: use names as they appear in the document (e.g. in the introductory recitals or header). "
    "Return only valid JSON matching the schema; no commentary."
)


def build_offline_tx_metadata_request_body(
    agreement_uuid: str,
    concatenated_page_text: str,
    *,
    model: str = "gpt-5-mini",
) -> Dict[str, Any]:
    """Build request body for offline extraction (one agreement). No tools; JSON output."""
    schema = json_schema_offline_metadata()
    return {
        "custom_id": agreement_uuid,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "instructions": TX_METADATA_OFFLINE_INSTRUCTIONS,
            "input": [{"role": "user", "content": concatenated_page_text}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "offline_tx_metadata",
                    "strict": True,
                    "schema": schema,
                }
            },
        },
    }


def parse_offline_tx_metadata_response_text(raw_text: str) -> Dict[str, Any]:
    obj = json.loads(raw_text)
    if not isinstance(obj, dict):
        raise ValueError("Response JSON is not an object.")
    for key in ("target", "acquirer", "deal_type"):
        if key not in obj:
            raise ValueError(f"Missing required key in response JSON: {key!r}.")
    target = obj.get("target")
    acquirer = obj.get("acquirer")
    deal_type_raw = obj.get("deal_type")
    if target is not None and not isinstance(target, str):
        raise TypeError("target must be a string or null.")
    if acquirer is not None and not isinstance(acquirer, str):
        raise TypeError("acquirer must be a string or null.")
    return {
        "target": target if (isinstance(target, str) and target.strip()) else None,
        "acquirer": acquirer if (isinstance(acquirer, str) and acquirer.strip()) else None,
        "deal_type": _validate_nullable_enum_standalone(
            deal_type_raw, field_name="deal_type", allowed=DEAL_TYPE_ALLOWED
        ),
    }


def _validate_nullable_enum_standalone(
    v: object | None, *, field_name: str, allowed: tuple[str, ...]
) -> Optional[str]:
    if v is None:
        return None
    if not isinstance(v, str):
        raise TypeError(f"{field_name} must be a string or null.")
    if v not in allowed:
        raise TypeError(f"{field_name} must be one of {allowed} or null.")
    return v


def build_offline_update_params(
    *, agreement_uuid: str, parsed: Dict[str, Any]
) -> Dict[str, Any]:
    """Params for UPDATE agreements SET target, acquirer, deal_type only."""
    target = parsed.get("target")
    acquirer = parsed.get("acquirer")
    deal_type = parsed.get("deal_type")
    if target is not None and not isinstance(target, str):
        raise TypeError("target must be a string or null.")
    if acquirer is not None and not isinstance(acquirer, str):
        raise TypeError("acquirer must be a string or null.")
    deal_type = _validate_nullable_enum_standalone(
        deal_type, field_name="deal_type", allowed=DEAL_TYPE_ALLOWED
    )
    return {
        "uuid": agreement_uuid,
        "target": target or None,
        "acquirer": acquirer or None,
        "deal_type": deal_type,
    }


def json_schema_offline_counsel() -> Dict[str, Any]:
    """JSON schema for offline counsel extraction."""
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "target_counsel": {"type": ["string", "null"]},
            "acquirer_counsel": {"type": ["string", "null"]},
        },
        "required": ["target_counsel", "acquirer_counsel"],
    }


TX_METADATA_OFFLINE_COUNSEL_INSTRUCTIONS = (
    "You are an expert at reading M&A agreement documents. "
    "Using ONLY the provided section text plus the provided target and acquirer party names, "
    "extract the law firm serving as counsel to each side. "
    "Return two fields: "
    "1) target_counsel: the law firm representing the target/seller/company being acquired. "
    "2) acquirer_counsel: the law firm representing the acquirer/buyer/parent. "
    "If either side is unclear, return null for that field. "
    "Return the firm name only, not individual lawyers, addresses, or extra commentary. "
    "Return only valid JSON matching the schema; no commentary."
)


def build_offline_counsel_request_body(
    agreement_uuid: str,
    *,
    section_text: str,
    target_name: str,
    acquirer_name: str,
    model: str = "gpt-5-mini",
) -> Dict[str, Any]:
    """Build request body for counsel extraction from the counsel section."""
    schema = json_schema_offline_counsel()
    prompt = (
        f"Target name: {target_name}\n"
        f"Acquirer name: {acquirer_name}\n\n"
        "Counsel section text:\n"
        f"{section_text}"
    )
    return {
        "custom_id": agreement_uuid,
        "method": "POST",
        "url": "/v1/responses",
        "body": {
            "model": model,
            "instructions": TX_METADATA_OFFLINE_COUNSEL_INSTRUCTIONS,
            "input": [{"role": "user", "content": prompt}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "offline_tx_counsel",
                    "strict": True,
                    "schema": schema,
                }
            },
        },
    }


def parse_offline_counsel_response_text(raw_text: str) -> Dict[str, Any]:
    obj = json.loads(raw_text)
    if not isinstance(obj, dict):
        raise ValueError("Response JSON is not an object.")
    for key in ("target_counsel", "acquirer_counsel"):
        if key not in obj:
            raise ValueError(f"Missing required key in response JSON: {key!r}.")
        if obj[key] is not None and not isinstance(obj[key], str):
            raise TypeError(f"{key} must be a string or null.")
    target_counsel_raw = obj.get("target_counsel")
    acquirer_counsel_raw = obj.get("acquirer_counsel")
    target_counsel = (
        target_counsel_raw.strip() if isinstance(target_counsel_raw, str) and target_counsel_raw.strip() else None
    )
    acquirer_counsel = (
        acquirer_counsel_raw.strip()
        if isinstance(acquirer_counsel_raw, str) and acquirer_counsel_raw.strip()
        else None
    )
    return {
        "target_counsel": target_counsel,
        "acquirer_counsel": acquirer_counsel,
    }


def build_offline_counsel_update_params(
    *,
    agreement_uuid: str,
    parsed: Dict[str, Any],
) -> Dict[str, Any]:
    target_counsel = parsed.get("target_counsel")
    acquirer_counsel = parsed.get("acquirer_counsel")
    if target_counsel is not None and not isinstance(target_counsel, str):
        raise TypeError("target_counsel must be a string or null.")
    if acquirer_counsel is not None and not isinstance(acquirer_counsel, str):
        raise TypeError("acquirer_counsel must be a string or null.")
    return {
        "uuid": agreement_uuid,
        "target_counsel": target_counsel or None,
        "acquirer_counsel": acquirer_counsel or None,
    }


def map_consideration_type_to_db(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    if value == "all_cash":
        return "cash"
    if value == "all_stock":
        return "stock"
    if value == "mixed":
        return "mixed"
    if value == "unknown":
        return None
    raise ValueError(f"Unexpected consideration_type: {value}")


def map_public_flag_to_type(value: object | None) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise TypeError("Public flag must be a boolean or null.")
    return "public" if value else "private"


def build_tx_metadata_update_params(
    *,
    agreement_uuid: str,
    tx_metadata_obj: Dict[str, Any],
    filing_date: object | None = None,
    pending_max_age_years: int = 3,
) -> Dict[str, Any]:
    if pending_max_age_years < 0:
        raise ValueError("pending_max_age_years must be >= 0.")

    consideration = map_consideration_type_to_db(tx_metadata_obj.get("consideration_type"))
    price = tx_metadata_obj.get("purchase_price")
    if not isinstance(price, dict):
        raise TypeError("purchase_price must be an object.")

    def _to_float_or_none(v: Any) -> Optional[float]:
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return float(v)
        raise TypeError("Price component must be a number or null.")

    price_cash = _to_float_or_none(price.get("cash"))
    price_stock = _to_float_or_none(price.get("stock"))
    price_assets = _to_float_or_none(price.get("assets"))
    if consideration == "cash" and ((price_stock or 0.0) > 0.0 or (price_assets or 0.0) > 0.0):
        raise ValueError(
            "consideration_type='all_cash' is inconsistent with non-zero stock/assets values."
        )
    if consideration == "stock" and ((price_cash or 0.0) > 0.0 or (price_assets or 0.0) > 0.0):
        raise ValueError(
            "consideration_type='all_stock' is inconsistent with non-zero cash/assets values."
        )

    price_total = None
    raw_components = {
        "cash": price_cash,
        "stock": price_stock,
        "assets": price_assets,
    }
    if all(v is None for v in raw_components.values()):
        price_total = None
    elif consideration == "cash":
        price_total = price_cash
    elif consideration == "stock":
        price_total = price_stock
    else:
        non_null_components = [value for value in raw_components.values() if value is not None]
        if len(non_null_components) >= 2:
            price_total = float(sum(non_null_components))
        else:
            # Mixed deals need at least two populated components before the total is reliable.
            price_total = None

    target_type = map_public_flag_to_type(tx_metadata_obj.get("target_public"))
    acquirer_type = map_public_flag_to_type(tx_metadata_obj.get("acquirer_public"))

    target_pe = tx_metadata_obj.get("target_pe")
    if target_pe is not None and not isinstance(target_pe, bool):
        raise TypeError("target_pe must be a boolean or null.")
    acquirer_pe = tx_metadata_obj.get("acquirer_pe")
    if acquirer_pe is not None and not isinstance(acquirer_pe, bool):
        raise TypeError("acquirer_pe must be a boolean or null.")

    sources_obj = tx_metadata_obj.get("metadata_sources")
    if not isinstance(sources_obj, dict):
        raise TypeError("metadata_sources must be an object.")
    citations = sources_obj.get("citations")
    notes = sources_obj.get("notes")
    if not isinstance(citations, list):
        raise TypeError("metadata_sources.citations must be an array.")
    if notes is not None and not isinstance(notes, str):
        raise TypeError("metadata_sources.notes must be a string or null.")
    citation_fields_supported = set(CITABLE_FIELDS_WITH_DEAL_TYPE)
    cited_fields: set[str] = set()
    for c in citations:
        if not isinstance(c, dict):
            raise TypeError("metadata_sources.citations items must be objects.")
        field_name = c.get("field")
        url = c.get("url")
        source_type = c.get("source_type")
        published_at = c.get("published_at")
        locator = c.get("locator")
        excerpt = c.get("excerpt")
        if not isinstance(field_name, str) or field_name not in citation_fields_supported:
            raise TypeError("metadata_sources.citations[].field contains an unsupported value.")
        if not isinstance(url, str) or not url:
            raise TypeError("metadata_sources.citations[].url must be a non-empty string.")
        if not re.match(r"^https?://", url):
            raise TypeError("metadata_sources.citations[].url must start with http:// or https://.")
        if not isinstance(source_type, str) or not source_type.strip():
            raise TypeError("metadata_sources.citations[].source_type must be a non-empty string.")
        if published_at is not None:
            _ = _parse_optional_date_like(published_at, field_name="metadata_sources.citations[].published_at")
        if locator is not None and not isinstance(locator, str):
            raise TypeError("metadata_sources.citations[].locator must be a string or null.")
        if not isinstance(excerpt, str) or not excerpt.strip():
            raise TypeError("metadata_sources.citations[].excerpt must be a non-empty string.")
        cited_fields.add(field_name)
    def _validate_naics(code: object | None, *, field_name: str) -> Optional[str]:
        if code is None:
            return None
        if not isinstance(code, str):
            raise TypeError(f"{field_name} must be a string or null.")
        c = code.strip()
        if not re.fullmatch(r"\d{2}|\d{3}", c):
            raise TypeError(f"{field_name} must be a 2- or 3-digit NAICS code string or null.")
        return c

    target_industry = _validate_naics(
        tx_metadata_obj.get("target_industry"), field_name="target_industry"
    )
    acquirer_industry = _validate_naics(
        tx_metadata_obj.get("acquirer_industry"), field_name="acquirer_industry"
    )

    def _validate_date_str(v: object | None, *, field_name: str) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise TypeError(f"{field_name} must be a string or null.")
        s = v.strip()
        m = re.match(r"^(\d{4}-\d{2}-\d{2})$", s)
        if not m:
            raise TypeError(f"{field_name} must be a YYYY-MM-DD string or null.")
        try:
            _ = datetime.strptime(m.group(1), "%Y-%m-%d")
        except ValueError as exc:
            raise TypeError(f"{field_name} must be a valid calendar date in YYYY-MM-DD format.") from exc
        return m.group(1)

    def _validate_nullable_enum(
        v: object | None, *, field_name: str, allowed: tuple[str, ...]
    ) -> Optional[str]:
        if v is None:
            return None
        if not isinstance(v, str):
            raise TypeError(f"{field_name} must be a string or null.")
        if v not in allowed:
            raise TypeError(f"{field_name} must be one of {allowed} or null.")
        return v

    announce_date = _validate_date_str(tx_metadata_obj.get("announce_date"), field_name="announce_date")
    close_date = _validate_date_str(tx_metadata_obj.get("close_date"), field_name="close_date")

    deal_status_raw = tx_metadata_obj.get("deal_status")
    deal_status = None
    if deal_status_raw is not None:
        if not isinstance(deal_status_raw, str):
            raise TypeError("deal_status must be a string or null.")
        if deal_status_raw == "unknown":
            deal_status = None
        else:
            deal_status = _validate_nullable_enum(
                deal_status_raw,
                field_name="deal_status",
                allowed=("pending", "complete", "cancelled"),
            )
    if announce_date is not None and close_date is not None and close_date < announce_date:
        raise ValueError("close_date cannot be earlier than announce_date.")
    if deal_status == "complete" and close_date is None:
        raise ValueError("deal_status='complete' requires a non-null close_date.")
    if deal_status == "pending" and close_date is not None:
        raise ValueError("deal_status='pending' requires close_date to be null.")
    filing_date_obj = _parse_optional_date_like(filing_date, field_name="filing_date")
    if deal_status == "pending" and filing_date_obj is not None:
        cutoff = _date_years_ago(date.today(), pending_max_age_years)
        if filing_date_obj <= cutoff:
            deal_status = None

    attitude = _validate_nullable_enum(
        tx_metadata_obj.get("attitude"),
        field_name="attitude",
        allowed=("friendly", "hostile", "unsolicited"),
    )
    deal_type = _validate_nullable_enum(
        tx_metadata_obj.get("deal_type"),
        field_name="deal_type",
        allowed=DEAL_TYPE_ALLOWED,
    )
    purpose = _validate_nullable_enum(
        tx_metadata_obj.get("purpose"),
        field_name="purpose",
        allowed=("strategic", "financial"),
    )

    required_cited_fields: set[str] = set()

    def _require(field_name: str, value: object | None) -> None:
        if value is not None:
            required_cited_fields.add(field_name)

    _require("consideration_type", tx_metadata_obj.get("consideration_type"))
    _require("purchase_price.cash", price_cash)
    _require("purchase_price.stock", price_stock)
    _require("purchase_price.assets", price_assets)
    _require("target_public", tx_metadata_obj.get("target_public"))
    _require("acquirer_public", tx_metadata_obj.get("acquirer_public"))
    _require("target_pe", target_pe)
    _require("acquirer_pe", acquirer_pe)
    _require("target_industry", target_industry)
    _require("acquirer_industry", acquirer_industry)
    _require("announce_date", announce_date)
    _require("close_date", close_date)
    if deal_status is not None:
        required_cited_fields.add("deal_status")
    _require("attitude", attitude)
    _require("deal_type", deal_type)
    _require("purpose", purpose)

    missing_citations = sorted(required_cited_fields - cited_fields)
    blocking_missing_citations = sorted(
        field_name for field_name in missing_citations if field_name not in NON_CORE_CITATION_OPTIONAL_FIELDS
    )
    if blocking_missing_citations:
        raise ValueError(
            "metadata_sources.citations must cover every non-null output field. Missing: "
            + ", ".join(blocking_missing_citations)
        )
    uncited_non_core_fields = sorted(
        field_name for field_name in missing_citations if field_name in NON_CORE_CITATION_OPTIONAL_FIELDS
    )
    metadata_uncited_fields = json.dumps(uncited_non_core_fields, ensure_ascii=False, separators=(",", ":"))

    runtime_metadata = tx_metadata_obj.get("metadata_run_stats")
    if runtime_metadata is not None:
        if not isinstance(runtime_metadata, dict):
            raise TypeError("metadata_run_stats must be an object or null.")
        token_usage_obj = runtime_metadata.get("token_usage")
        if not isinstance(token_usage_obj, dict):
            raise TypeError("metadata_run_stats.token_usage must be an object.")
        raw_search_count = runtime_metadata.get("search_count", 0)
        if not isinstance(raw_search_count, int):
            raise TypeError("metadata_run_stats.search_count must be an integer.")
        runtime_metadata = build_web_search_runtime_metadata(
            response_usage=token_usage_obj,
            search_count=raw_search_count,
        )

    notes_normalized = notes.strip().lower() if isinstance(notes, str) else ""
    has_non_usd_note = "not stated in usd" in notes_normalized or "non-usd" in notes_normalized or "non usd" in notes_normalized
    if has_non_usd_note and any(value is not None for value in (price_cash, price_stock, price_assets)):
        raise ValueError("Non-USD price notes require all purchase_price fields to be null.")

    metadata_payload = {
        "metadata_sources": sources_obj,
        "metadata_run_stats": runtime_metadata,
    }
    metadata_sources = json.dumps(metadata_payload, ensure_ascii=False, separators=(",", ":"))

    return {
        "consideration": consideration,
        "price_cash": price_cash,
        "price_stock": price_stock,
        "price_assets": price_assets,
        "price_total": price_total,
        "target_type": target_type,
        "acquirer_type": acquirer_type,
        "target_pe": target_pe,
        "acquirer_pe": acquirer_pe,
        "target_industry": target_industry,
        "acquirer_industry": acquirer_industry,
        "announce_date": announce_date,
        "close_date": close_date,
        "deal_status": deal_status,
        "attitude": attitude,
        "deal_type": deal_type,
        "purpose": purpose,
        "metadata_sources": metadata_sources,
        "metadata_uncited_fields": metadata_uncited_fields,
        "uuid": agreement_uuid,
    }


def build_tx_metadata_update_params_web_search_only(
    *,
    agreement_uuid: str,
    tx_metadata_obj: Dict[str, Any],
    response_usage: Dict[str, Any],
    search_count: int = 0,
    filing_date: object | None = None,
    pending_max_age_years: int = 3,
) -> Dict[str, Any]:
    """Build UPDATE params for web-search mode: same as full but omit target, acquirer, deal_type."""
    sanitized_obj = _sanitize_web_search_metadata_obj(
        tx_metadata_obj,
        filing_date=filing_date,
        pending_max_age_years=pending_max_age_years,
    )
    obj_with_deal_type = {
        **sanitized_obj,
        "deal_type": tx_metadata_obj.get("deal_type", None),
        "metadata_run_stats": build_web_search_runtime_metadata(
            response_usage=response_usage,
            search_count=search_count,
        ),
    }
    params = build_tx_metadata_update_params(
        agreement_uuid=agreement_uuid,
        tx_metadata_obj=obj_with_deal_type,
        filing_date=filing_date,
        pending_max_age_years=pending_max_age_years,
    )
    return {k: v for k, v in params.items() if k not in ("target", "acquirer", "deal_type")}
