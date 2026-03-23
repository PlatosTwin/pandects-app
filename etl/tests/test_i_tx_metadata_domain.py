# pyright: reportAny=false
import json
import unittest
from datetime import date
from typing import cast

from etl.domain.i_tx_metadata import (
    build_tx_metadata_request_body_web_search_only,
    build_tx_metadata_update_params_web_search_only,
    build_web_search_runtime_metadata,
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
                    self._citation("consideration_type"),
                    self._citation("purchase_price.cash"),
                    self._citation("purchase_price.stock"),
                    self._citation("purchase_price.assets"),
                    self._citation("target_public"),
                    self._citation("acquirer_public"),
                    self._citation("acquirer_pe"),
                    self._citation("target_industry"),
                    self._citation("acquirer_industry"),
                    self._citation("announce_date"),
                    self._citation("deal_status"),
                    self._citation("attitude"),
                    self._citation("purpose"),
                ],
                "notes": None,
            },
        }

    def _citation(self, field: str) -> dict[str, object]:
        return {
            "field": field,
            "url": "https://example.com/deal",
            "source_type": "company_press_release",
            "published_at": "2024-01-01",
            "locator": "press release",
            "excerpt": f"Support for {field}.",
        }

    def _usage(self) -> dict[str, int]:
        return {"input_tokens": 111, "output_tokens": 22, "total_tokens": 133}

    def test_schema_web_search_only_excludes_deal_type(self) -> None:
        schema = json_schema_transaction_metadata_web_search_only()
        self.assertNotIn("deal_type", schema["properties"])
        self.assertNotIn("deal_type", schema["required"])
        self.assertNotIn("metadata_run_stats", schema["properties"])
        field_enum = schema["properties"]["metadata_sources"]["properties"]["citations"]["items"][
            "properties"
        ]["field"]["enum"]
        self.assertNotIn("deal_type", field_enum)

    def test_parse_web_search_response_requires_all_keys(self) -> None:
        invalid_payload = {"consideration_type": "all_cash"}
        with self.assertRaisesRegex(ValueError, "Missing required keys"):
            _ = parse_tx_metadata_response_text_web_search(json.dumps(invalid_payload))

    def test_build_update_params_web_search_only_maps_valid_payload(self) -> None:
        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=self._valid_web_search_obj(),
            response_usage=self._usage(),
        )
        self.assertEqual(params["uuid"], "agreement-1")
        self.assertEqual(params["consideration"], "cash")
        self.assertEqual(params["target_type"], "public")
        self.assertEqual(params["acquirer_type"], "private")
        self.assertEqual(params["price_total"], 100.0)
        self.assertIsInstance(params["metadata_sources"], str)
        self.assertEqual(params["metadata_uncited_fields"], "[]")
        self.assertNotIn("deal_type", params)
        metadata_payload = json.loads(params["metadata_sources"])
        self.assertEqual(
            metadata_payload["metadata_run_stats"],
            {"token_usage": self._usage()},
        )

    def test_build_update_params_web_search_only_totals_all_cash_when_other_components_null(self) -> None:
        payload = self._valid_web_search_obj()
        payload["purchase_price"] = {"cash": 562_500_000.0, "stock": None, "assets": None}
        payload["consideration_type"] = "all_cash"

        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=payload,
            response_usage=self._usage(),
        )

        self.assertEqual(params["price_total"], 562_500_000.0)

    def test_build_update_params_web_search_only_keeps_total_null_when_mixed_is_incomplete(self) -> None:
        payload = self._valid_web_search_obj()
        payload["purchase_price"] = {"cash": 10.0, "stock": None, "assets": 5.0}
        payload["consideration_type"] = "mixed"

        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=payload,
            response_usage=self._usage(),
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
        self.assertIn("acquirer: Acquirer A", input_text)
        self.assertIn("sec_filing_date: 2024-01-01", input_text)
        self.assertNotIn("target:", input_text)

    def test_build_update_params_web_search_only_requires_citation_coverage_for_non_null_fields(self) -> None:
        payload = self._valid_web_search_obj()
        payload["metadata_sources"] = {
            "citations": [self._citation("consideration_type")],
            "notes": None,
        }

        with self.assertRaisesRegex(ValueError, "must cover every non-null output field"):
            _ = build_tx_metadata_update_params_web_search_only(
                agreement_uuid="agreement-1",
                tx_metadata_obj=payload,
                response_usage=self._usage(),
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
                self._citation("consideration_type"),
                self._citation("purchase_price.cash"),
                self._citation("purchase_price.stock"),
                self._citation("purchase_price.assets"),
                self._citation("target_public"),
                self._citation("acquirer_public"),
                self._citation("announce_date"),
            ],
            "notes": None,
        }

        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=payload,
            response_usage=self._usage(),
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
                self._citation("consideration_type"),
                self._citation("purchase_price.cash"),
                self._citation("purchase_price.stock"),
                self._citation("purchase_price.assets"),
                self._citation("target_public"),
                self._citation("acquirer_public"),
            ],
            "notes": None,
        }

        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=payload,
            response_usage=self._usage(),
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
            response_usage=self._usage(),
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
                self._citation("consideration_type"),
                self._citation("purchase_price.cash"),
                self._citation("purchase_price.stock"),
                self._citation("purchase_price.assets"),
                self._citation("target_public"),
                self._citation("acquirer_public"),
                self._citation("acquirer_pe"),
                self._citation("target_industry"),
                self._citation("acquirer_industry"),
                self._citation("announce_date"),
                self._citation("close_date"),
                self._citation("deal_status"),
                self._citation("attitude"),
                self._citation("purpose"),
            ],
            "notes": None,
        }

        with self.assertRaisesRegex(ValueError, "pending' requires close_date to be null"):
            _ = build_tx_metadata_update_params_web_search_only(
                agreement_uuid="agreement-1",
                tx_metadata_obj=payload,
                response_usage=self._usage(),
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
            response_usage=self._usage(),
            filing_date=today_iso,
            pending_max_age_years=3,
        )
        self.assertEqual(params["deal_status"], "pending")

    def test_build_update_params_web_search_only_rejects_legacy_citation_shape(self) -> None:
        payload = self._valid_web_search_obj()
        payload["metadata_sources"] = {
            "citations": [{"url": "https://example.com/deal", "fields": ["consideration_type"]}],
            "notes": None,
        }
        with self.assertRaisesRegex(TypeError, "citations\\[\\]\\.field"):
            _ = build_tx_metadata_update_params_web_search_only(
                agreement_uuid="agreement-1",
                tx_metadata_obj=payload,
                response_usage=self._usage(),
            )

    def test_build_update_params_web_search_only_rejects_non_usd_note_with_prices(self) -> None:
        payload = self._valid_web_search_obj()
        metadata_sources = payload["metadata_sources"]
        self.assertIsInstance(metadata_sources, dict)
        payload["metadata_sources"] = {
            "citations": cast(dict[str, object], metadata_sources)["citations"],
            "notes": "Price omitted because consideration is not stated in USD.",
        }
        with self.assertRaisesRegex(ValueError, "Non-USD price notes require"):
            _ = build_tx_metadata_update_params_web_search_only(
                agreement_uuid="agreement-1",
                tx_metadata_obj=payload,
                response_usage=self._usage(),
            )

    def test_build_update_params_web_search_only_allows_non_usd_note_with_null_prices(self) -> None:
        payload = self._valid_web_search_obj()
        payload["purchase_price"] = {"cash": None, "stock": None, "assets": None}
        payload["metadata_sources"] = {
            "citations": [
                self._citation("consideration_type"),
                self._citation("target_public"),
                self._citation("acquirer_public"),
                self._citation("acquirer_pe"),
                self._citation("target_industry"),
                self._citation("acquirer_industry"),
                self._citation("announce_date"),
                self._citation("deal_status"),
                self._citation("attitude"),
                self._citation("purpose"),
            ],
            "notes": "Price omitted because consideration is not stated in USD.",
        }
        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=payload,
            response_usage=self._usage(),
        )
        self.assertIsNone(params["price_cash"])
        self.assertIsNone(params["price_total"])

    def test_build_web_search_runtime_metadata_validates_usage(self) -> None:
        self.assertEqual(
            build_web_search_runtime_metadata(response_usage=self._usage()),
            {"token_usage": self._usage()},
        )
        with self.assertRaisesRegex(TypeError, "response_usage.input_tokens"):
            _ = build_web_search_runtime_metadata(
                response_usage={"input_tokens": "bad", "output_tokens": 2, "total_tokens": 3}
            )


if __name__ == "__main__":
    _ = unittest.main()
