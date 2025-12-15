from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple


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
                    "cash": {"type": ["integer", "null"], "minimum": 0},
                    "stock": {"type": ["integer", "null"], "minimum": 0},
                    "assets": {"type": ["integer", "null"], "minimum": 0},
                },
                "required": ["cash", "stock", "assets"],
            },
            "target_public": {"type": ["boolean", "null"]},
            "acquirer_public": {"type": ["boolean", "null"]},
            "target_pe": {"type": ["boolean", "null"]},
            "acquirer_pe": {"type": ["boolean", "null"]},
            "sources": {"type": "string"},
        },
        "required": [
            "consideration_type",
            "purchase_price",
            "target_public",
            "acquirer_public",
            "target_pe",
            "acquirer_pe",
            "sources",
        ],
    }


TX_METADATA_INSTRUCTIONS = (
    "You are an expert M&A research analyst. For the provided transaction details, "
    "and using trusted sources only (via the websearch tool), find: "
    "1) the type of consideration (all stock, all cash, all asset swap, mixed); "
    "2) the purchase price (USD), without accounting for any assumed debt, notes, etc.; "
    "3) whether the target was public or private; "
    "4) whether the acquirer was public or private; "
    "5) whether the target was owned by a private equity shop; "
    "6) whether the acquirer was a private equity shop. "
    "For booleans and integers where you don't know the answer, use null."
)


def _filing_date_to_str(filing_date: Any) -> str:
    if filing_date is None:
        return "unknown"
    if isinstance(filing_date, (datetime, date)):
        return filing_date.isoformat()
    if isinstance(filing_date, str):
        return filing_date
    raise TypeError(f"Unexpected filing_date type: {type(filing_date).__name__}")


def build_jsonl_lines_for_agreements(
    agreements: List[Dict[str, Any]], *, model: str
) -> List[Dict[str, Any]]:
    lines: List[Dict[str, Any]] = []
    for row in agreements:
        agreement_uuid: str = row["agreement_uuid"]
        body = build_tx_metadata_request_body(row, model=model)

        line = {
            "custom_id": agreement_uuid,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }
        lines.append(line)

    return lines


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


def _extract_message_text(body: Dict[str, Any]) -> str:
    output = body["output"]
    msg_blocks = [o for o in output if o.get("type") == "message"]
    if not msg_blocks:
        raise ValueError("No assistant message block in output.")
    contents = msg_blocks[0]["content"]
    text_items = [c for c in contents if isinstance(c, dict) and "text" in c]
    if not text_items:
        raise ValueError("Assistant message has no text content.")
    raw_text = text_items[0]["text"]
    return raw_text


def parse_metadata_line(raw: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    rid = raw["custom_id"]
    resp = raw.get("response") or {}
    sc = resp.get("status_code")
    if sc not in (200, 201, 202):
        raise ValueError(f"Non-success status_code: {sc}")
    body = resp["body"]
    raw_text = _extract_message_text(body)
    obj = parse_tx_metadata_response_text(raw_text)
    return rid, obj


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


def map_public_flag_to_type(value: Optional[bool]) -> Optional[str]:
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
    if price_cash is not None or price_stock is not None or price_assets is not None:
        price_total = (price_cash or 0.0) + (price_stock or 0.0) + (price_assets or 0.0)

    target_type = map_public_flag_to_type(tx_metadata_obj.get("target_public"))
    acquirer_type = map_public_flag_to_type(tx_metadata_obj.get("acquirer_public"))

    target_pe = tx_metadata_obj.get("target_pe")
    if target_pe is not None and not isinstance(target_pe, bool):
        raise TypeError("target_pe must be a boolean or null.")
    acquirer_pe = tx_metadata_obj.get("acquirer_pe")
    if acquirer_pe is not None and not isinstance(acquirer_pe, bool):
        raise TypeError("acquirer_pe must be a boolean or null.")

    sources = tx_metadata_obj.get("sources")
    if not isinstance(sources, str):
        raise TypeError("sources must be a string.")

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
        "sources": sources,
        "uuid": agreement_uuid,
    }


