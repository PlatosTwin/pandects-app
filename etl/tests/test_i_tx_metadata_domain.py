# pyright: reportAny=false
import json
import unittest

from etl.domain.i_tx_metadata import (
    build_tx_metadata_update_params_web_search_only,
    json_schema_transaction_metadata_web_search_only,
    parse_tx_metadata_response_text_web_search,
)


class TxMetadataDomainTests(unittest.TestCase):
    def _valid_web_search_obj(self) -> dict[str, object]:
        return {
            "consideration_type": "all_cash",
            "purchase_price": {"cash": 100.0, "stock": 0.0, "assets": 25.0},
            "target_public": True,
            "acquirer_public": False,
            "target_pe": None,
            "acquirer_pe": True,
            "target_industry": "311",
            "acquirer_industry": "52",
            "announce_date": "2024-01-01",
            "close_date": None,
            "deal_status": "pending",
            "attitude": "friendly",
            "purpose": "strategic",
            "metadata_sources": {
                "citations": [
                    {
                        "url": "https://example.com/deal",
                        "fields": ["consideration_type", "purchase_price.cash"],
                    }
                ],
                "notes": None,
            },
        }

    def test_schema_web_search_only_excludes_deal_type(self) -> None:
        schema = json_schema_transaction_metadata_web_search_only()
        self.assertNotIn("deal_type", schema["properties"])
        self.assertNotIn("deal_type", schema["required"])
        fields_enum = schema["properties"]["metadata_sources"]["properties"]["citations"]["items"][
            "properties"
        ]["fields"]["items"]["enum"]
        self.assertNotIn("deal_type", fields_enum)

    def test_parse_web_search_response_requires_all_keys(self) -> None:
        invalid_payload = {"consideration_type": "all_cash"}
        with self.assertRaisesRegex(ValueError, "Missing required keys"):
            _ = parse_tx_metadata_response_text_web_search(json.dumps(invalid_payload))

    def test_build_update_params_web_search_only_maps_valid_payload(self) -> None:
        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=self._valid_web_search_obj(),
        )
        self.assertEqual(params["uuid"], "agreement-1")
        self.assertEqual(params["consideration"], "cash")
        self.assertEqual(params["target_type"], "public")
        self.assertEqual(params["acquirer_type"], "private")
        self.assertEqual(params["price_total"], 125.0)
        self.assertIsInstance(params["metadata_sources"], str)
        self.assertNotIn("deal_type", params)


if __name__ == "__main__":
    _ = unittest.main()
