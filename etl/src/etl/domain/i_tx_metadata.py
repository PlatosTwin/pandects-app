from __future__ import annotations

import json
from datetime import date, datetime
from typing import Any, Dict, List, Tuple


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
                    "cash": {"type": ["integer", "null"], "minimum": 0},
                    "stock": {"type": ["integer", "null"], "minimum": 0},
                },
                "required": ["cash", "stock"],
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
    schema = json_schema_transaction_metadata()
    for row in agreements:
        agreement_uuid: str = row["agreement_uuid"]
        target: str = row.get("target") or ""
        acquirer: str = row.get("acquirer") or ""
        filing_date_str = _filing_date_to_str(row.get("filing_date"))

        body = {
            "model": model,
            "tools": [{"type": "web_search"}],
            "instructions": (
                "You are an expert M&A research analyst. For the provided transaction details, "
                "and using trusted sources only (via the websearch tool), find: "
                "1) the type of consideration (all stock, all cash, mixed); "
                "2) the purchase price (USD), without accounting for debt; "
                "3) whether the target was public or private; "
                "4) whether the acquirer was public or private; "
                "5) whether the target was owned by a private equity shop; "
                "6) whether the acquirer was a private equity shop. "
                "For booleans and integers where you don't know the answer, use null."
            ),
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

        line = {
            "custom_id": agreement_uuid,
            "method": "POST",
            "url": "/v1/responses",
            "body": body,
        }
        lines.append(line)

    return lines


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
    obj = json.loads(raw_text)
    if not isinstance(obj, dict):
        raise ValueError("Response JSON is not an object.")
    # Basic required keys presence
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
    return rid, obj




