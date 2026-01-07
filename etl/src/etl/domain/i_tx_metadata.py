# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any, Dict, Optional


def json_schema_transaction_metadata():
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
                        "enum": [
                            "merger",
                            "stock_acquisition",
                            "asset_acquisition",
                            "tender_offer",
                        ],
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
            "sources": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "citations": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "url": {"type": "string"},
                                "fields": {
                                    "type": "array",
                                    "items": {
                                        "type": "string",
                                        "enum": [
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
                                            "deal_type",
                                            "purpose",
                                        ],
                                    },
                                },
                            },
                            "required": ["url", "fields"],
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
            "sources",
        ],
    }


TX_METADATA_INSTRUCTIONS = (
    "You are an expert M&A research analyst. For the provided transaction details, "
    "and using trusted sources only (via the websearch tool), find: "
    "1) the type of consideration (all stock, all cash, all asset swap, mixed); "
    "2) the purchase price components (USD), without accounting for any assumed debt, notes, etc. "
    "(i.e., equity value / consideration paid to sellers, not enterprise value). "
    "Return full USD amounts (e.g., $4.18 billion -> 4180000000), not shorthand units; "
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
    "13) deal_type: 'merger', 'stock_acquisition', 'asset_acquisition', or 'tender_offer' (use null if unclear); "
    "14) purpose: 'strategic' or 'financial' (use null if unclear). "
    "For fields where you don't know the answer, use null. "
    "Do a sanity check before completing, and dig deeper if need be—e.g., "
    "if you think total consideration is $19, that is probably wrong, as it's too small; "
    "if you think the consideration is mixed, at least two consideration type columns should be non-zero; "
    "etc."
    "For sources, populate `sources.citations` with URLs and explicitly list which fields each URL supports. "
    "If you couldn't find a field, do not guess; say why briefly in `sources.notes`."
)


def _filing_date_to_str(filing_date: Any) -> str:
    if filing_date is None:
        return "unknown"
    if isinstance(filing_date, (datetime, date)):
        return filing_date.isoformat()
    if isinstance(filing_date, str):
        return filing_date
    raise TypeError(f"Unexpected filing_date type: {type(filing_date).__name__}")


def build_tx_metadata_request_body(
    agreement: Dict[str, Any], *, model: str
) -> Dict[str, Any]:
    schema = json_schema_transaction_metadata()
    target: str = agreement.get("target") or ""
    acquirer: str = agreement.get("acquirer") or ""
    filing_date_str = _filing_date_to_str(agreement.get("filing_date"))

    return {
        "model": model,
        "tools": [{"type": "web_search"}],
        "instructions": TX_METADATA_INSTRUCTIONS,
        "input": f"Transaction: {acquirer} acquired {target}, with an SEC filing date of {filing_date_str}.",
        "text": {
            "format": {
                "type": "json_schema",
                "name": "transaction_metadata",
                "strict": True,
                "schema": schema,
            }
        },
    }


def parse_tx_metadata_response_text(raw_text: str) -> Dict[str, Any]:
    obj = json.loads(raw_text)
    if not isinstance(obj, dict):
        raise ValueError("Response JSON is not an object.")
    required_keys = {
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
        "sources",
    }
    if not required_keys.issubset(obj.keys()):
        raise ValueError("Missing required keys in response JSON.")
    return obj


def parse_tx_metadata_response(resp: Any) -> Dict[str, Any]:
    if not hasattr(resp, "output_text"):
        raise TypeError("Response object is missing output_text.")
    raw_text = resp.output_text  # type: ignore[attr-defined]
    if not isinstance(raw_text, str):
        raise TypeError("Response output_text is not a string.")
    return parse_tx_metadata_response_text(raw_text)


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
    *, agreement_uuid: str, tx_metadata_obj: Dict[str, Any]
) -> Dict[str, Any]:
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
    price_total = None
    components = (price_cash, price_stock, price_assets)
    if all(v is None for v in components):
        price_total = None
    elif any(v is None for v in components):
        # Avoid misleading totals when any component is unknown.
        price_total = None
    else:
        assert price_cash is not None and price_stock is not None and price_assets is not None
        price_total = price_cash + price_stock + price_assets

    target_type = map_public_flag_to_type(tx_metadata_obj.get("target_public"))
    acquirer_type = map_public_flag_to_type(tx_metadata_obj.get("acquirer_public"))

    target_pe = tx_metadata_obj.get("target_pe")
    if target_pe is not None and not isinstance(target_pe, bool):
        raise TypeError("target_pe must be a boolean or null.")
    acquirer_pe = tx_metadata_obj.get("acquirer_pe")
    if acquirer_pe is not None and not isinstance(acquirer_pe, bool):
        raise TypeError("acquirer_pe must be a boolean or null.")

    sources_obj = tx_metadata_obj.get("sources")
    if not isinstance(sources_obj, dict):
        raise TypeError("sources must be an object.")
    citations = sources_obj.get("citations")
    notes = sources_obj.get("notes")
    if not isinstance(citations, list):
        raise TypeError("sources.citations must be an array.")
    if notes is not None and not isinstance(notes, str):
        raise TypeError("sources.notes must be a string or null.")
    for c in citations:
        if not isinstance(c, dict):
            raise TypeError("sources.citations items must be objects.")
        url = c.get("url")
        fields = c.get("fields")
        if not isinstance(url, str) or not url:
            raise TypeError("sources.citations[].url must be a non-empty string.")
        if not isinstance(fields, list) or not all(isinstance(f, str) for f in fields):
            raise TypeError("sources.citations[].fields must be an array of strings.")
    sources = json.dumps(sources_obj, ensure_ascii=False, separators=(",", ":"))

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

    attitude = _validate_nullable_enum(
        tx_metadata_obj.get("attitude"),
        field_name="attitude",
        allowed=("friendly", "hostile", "unsolicited"),
    )
    deal_type = _validate_nullable_enum(
        tx_metadata_obj.get("deal_type"),
        field_name="deal_type",
        allowed=("merger", "stock_acquisition", "asset_acquisition", "tender_offer"),
    )
    purpose = _validate_nullable_enum(
        tx_metadata_obj.get("purpose"),
        field_name="purpose",
        allowed=("strategic", "financial"),
    )

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
        "sources": sources,
        "uuid": agreement_uuid,
    }
