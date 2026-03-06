# pyright: reportAny=false
import json
import unittest
from datetime import date

from etl.domain.i_tx_metadata import (
    build_tx_metadata_request_body_web_search_only,
    build_tx_metadata_update_params_web_search_only,
    json_schema_transaction_metadata_web_search_only,
    parse_tx_metadata_response_text_web_search,
)


class TxMetadataDomainTests(unittest.TestCase):
    def _valid_web_search_obj(self) -> dict[str, object]:
        return {
            "consideration_type": "all_cash",
            "purchase_price": {"cash": 100.0, "stock": 0.0, "assets": 0.0},
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
                        "fields": [
                            "consideration_type",
                            "purchase_price.cash",
                            "purchase_price.stock",
                            "purchase_price.assets",
                            "target_public",
                            "acquirer_public",
                            "acquirer_pe",
                            "target_industry",
                            "acquirer_industry",
                            "announce_date",
                            "deal_status",
                            "attitude",
                            "purpose",
                        ],
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
        self.assertEqual(params["price_total"], 100.0)
        self.assertIsInstance(params["metadata_sources"], str)
        self.assertEqual(params["metadata_uncited_fields"], "[]")
        self.assertNotIn("deal_type", params)

    def test_build_update_params_web_search_only_totals_all_cash_when_other_components_null(self) -> None:
        payload = self._valid_web_search_obj()
        payload["purchase_price"] = {"cash": 562_500_000.0, "stock": None, "assets": None}
        payload["consideration_type"] = "all_cash"

        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=payload,
        )

        self.assertEqual(params["price_total"], 562_500_000.0)

    def test_build_update_params_web_search_only_keeps_total_null_when_mixed_is_incomplete(self) -> None:
        payload = self._valid_web_search_obj()
        payload["purchase_price"] = {"cash": 10.0, "stock": None, "assets": 5.0}
        payload["consideration_type"] = "mixed"

        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=payload,
        )

        self.assertIsNone(params["price_total"])

    def test_build_tx_metadata_request_body_web_search_only_includes_sec_url(self) -> None:
        request_body = build_tx_metadata_request_body_web_search_only(
            {
                "target": "Target A",
                "acquirer": "Acquirer A",
                "filing_date": "2024-01-01",
                "url": "https://www.sec.gov/Archives/edgar/data/123/abc.htm",
            },
            model="gpt-5.1",
        )
        input_text = request_body.get("input")
        self.assertIsInstance(input_text, str)
        assert isinstance(input_text, str)
        self.assertIn("acquirer: Acquirer A", input_text)
        self.assertIn("target: Target A", input_text)
        self.assertIn("sec_filing_url: https://www.sec.gov/Archives/edgar/data/123/abc.htm", input_text)
        instructions = request_body.get("instructions")
        self.assertIsInstance(instructions, str)
        assert isinstance(instructions, str)
        self.assertIn("no more than 12 web searches", instructions)

    def test_build_tx_metadata_request_body_web_search_only_names_only_mode(self) -> None:
        request_body = build_tx_metadata_request_body_web_search_only(
            {
                "target": "Target A",
                "acquirer": "Acquirer A",
                "filing_date": "2024-01-01",
                "url": None,
            },
            model="gpt-5.1",
        )
        input_text = request_body.get("input")
        self.assertIsInstance(input_text, str)
        assert isinstance(input_text, str)
        self.assertIn("acquirer: Acquirer A", input_text)
        self.assertIn("target: Target A", input_text)
        self.assertIn("sec_filing_date: 2024-01-01", input_text)
        self.assertNotIn("sec_filing_url:", input_text)

    def test_build_tx_metadata_request_body_web_search_only_url_only_mode(self) -> None:
        request_body = build_tx_metadata_request_body_web_search_only(
            {
                "target": None,
                "acquirer": "Acquirer A",
                "filing_date": "2024-01-01",
                "url": "https://www.sec.gov/Archives/edgar/data/123/abc.htm",
            },
            model="gpt-5.1",
        )
        input_text = request_body.get("input")
        self.assertIsInstance(input_text, str)
        assert isinstance(input_text, str)
        self.assertIn("sec_filing_url: https://www.sec.gov/Archives/edgar/data/123/abc.htm", input_text)
        self.assertNotIn("acquirer:", input_text)
        self.assertNotIn("target:", input_text)
        self.assertNotIn("sec_filing_date:", input_text)

    def test_build_update_params_web_search_only_requires_citation_coverage_for_non_null_fields(self) -> None:
        payload = self._valid_web_search_obj()
        payload["metadata_sources"] = {
            "citations": [{"url": "https://example.com/deal", "fields": ["consideration_type"]}],
            "notes": None,
        }

        with self.assertRaisesRegex(ValueError, "must cover every non-null output field"):
            _ = build_tx_metadata_update_params_web_search_only(
                agreement_uuid="agreement-1",
                tx_metadata_obj=payload,
            )

    def test_build_update_params_web_search_only_allows_missing_non_core_citations(self) -> None:
        payload = self._valid_web_search_obj()
        payload["target_pe"] = True
        payload["acquirer_pe"] = False
        payload["deal_status"] = "pending"
        payload["attitude"] = "friendly"
        payload["purpose"] = "strategic"
        payload["target_industry"] = "311"
        payload["acquirer_industry"] = "52"
        payload["metadata_sources"] = {
            "citations": [
                {
                    "url": "https://example.com/deal",
                    "fields": [
                        "consideration_type",
                        "purchase_price.cash",
                        "purchase_price.stock",
                        "purchase_price.assets",
                        "target_public",
                        "acquirer_public",
                        "announce_date",
                    ],
                }
            ],
            "notes": None,
        }

        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=payload,
        )
        self.assertEqual(
            params["metadata_uncited_fields"],
            json.dumps(
                [
                    "acquirer_industry",
                    "acquirer_pe",
                    "attitude",
                    "deal_status",
                    "purpose",
                    "target_industry",
                    "target_pe",
                ],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
        self.assertEqual(params["deal_status"], "pending")

    def test_build_update_params_web_search_only_allows_missing_date_citations(self) -> None:
        payload = self._valid_web_search_obj()
        payload["announce_date"] = "2024-01-01"
        payload["close_date"] = "2024-02-01"
        payload["deal_status"] = "complete"
        payload["target_pe"] = None
        payload["acquirer_pe"] = None
        payload["target_industry"] = None
        payload["acquirer_industry"] = None
        payload["attitude"] = None
        payload["purpose"] = None
        payload["metadata_sources"] = {
            "citations": [
                {
                    "url": "https://example.com/deal",
                    "fields": [
                        "consideration_type",
                        "purchase_price.cash",
                        "purchase_price.stock",
                        "purchase_price.assets",
                        "target_public",
                        "acquirer_public",
                    ],
                }
            ],
            "notes": None,
        }

        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=payload,
        )
        self.assertEqual(
            params["metadata_uncited_fields"],
            json.dumps(
                ["announce_date", "close_date", "deal_status"],
                ensure_ascii=False,
                separators=(",", ":"),
            ),
        )
        self.assertEqual(params["deal_status"], "complete")

    def test_build_update_params_web_search_only_nulls_pending_for_old_filing(self) -> None:
        payload = self._valid_web_search_obj()
        payload["deal_status"] = "pending"

        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=payload,
            filing_date="2000-01-01",
            pending_max_age_years=3,
        )
        self.assertIsNone(params["deal_status"])

    def test_build_update_params_web_search_only_rejects_pending_with_close_date(self) -> None:
        payload = self._valid_web_search_obj()
        payload["deal_status"] = "pending"
        payload["close_date"] = "2024-02-01"
        payload["metadata_sources"] = {
            "citations": [
                {
                    "url": "https://example.com/deal",
                    "fields": [
                        "consideration_type",
                        "purchase_price.cash",
                        "purchase_price.stock",
                        "purchase_price.assets",
                        "target_public",
                        "acquirer_public",
                        "acquirer_pe",
                        "target_industry",
                        "acquirer_industry",
                        "announce_date",
                        "close_date",
                        "deal_status",
                        "attitude",
                        "purpose",
                    ],
                }
            ],
            "notes": None,
        }

        with self.assertRaisesRegex(ValueError, "pending' requires close_date to be null"):
            _ = build_tx_metadata_update_params_web_search_only(
                agreement_uuid="agreement-1",
                tx_metadata_obj=payload,
                filing_date=date.today().isoformat(),
                pending_max_age_years=3,
            )

    def test_build_update_params_web_search_only_allows_pending_for_recent_filing(self) -> None:
        payload = self._valid_web_search_obj()
        payload["deal_status"] = "pending"
        today_iso = date.today().isoformat()

        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=payload,
            filing_date=today_iso,
            pending_max_age_years=3,
        )
        self.assertEqual(params["deal_status"], "pending")


if __name__ == "__main__":
    _ = unittest.main()
