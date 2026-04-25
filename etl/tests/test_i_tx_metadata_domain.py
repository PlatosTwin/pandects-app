# pyright: reportAny=false
import json
import unittest
from datetime import date
from decimal import Decimal
from typing import cast

from etl.domain.i_tx_metadata import (
    build_tx_metadata_request_body_web_search_only,
    build_web_search_retry_context,
    build_tx_metadata_update_params_web_search_only,
    build_web_search_runtime_metadata,
    json_schema_transaction_metadata_web_search_only,
    merge_retry_web_search_response,
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

    def _requeue_agreement_row(self, **overrides: object) -> dict[str, object]:
        row: dict[str, object] = {
            "agreement_uuid": "agreement-1",
            "target": "Target A",
            "acquirer": "Acquirer A",
            "filing_date": "2024-01-01",
            "url": "https://www.sec.gov/Archives/edgar/data/123/abc.htm",
            "transaction_consideration": "cash",
            "transaction_price_cash": 100.0,
            "transaction_price_stock": None,
            "transaction_price_assets": None,
            "transaction_price_total": 100.0,
            "target_type": "public",
            "acquirer_type": "private",
            "target_pe": None,
            "acquirer_pe": True,
            "target_industry": "311",
            "acquirer_industry": "52",
            "announce_date": "2024-01-01",
            "close_date": None,
            "deal_status": "complete",
            "attitude": "friendly",
            "purpose": "strategic",
            "metadata_sources": json.dumps(
                {
                    "metadata_sources": {
                        "citations": [
                            self._citation("consideration_type"),
                            self._citation("purchase_price.cash"),
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
                        "notes": "Existing trusted metadata.",
                    },
                    "metadata_run_stats": {"token_usage": self._usage(), "search_count": 1},
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            "metadata_uncited_fields": json.dumps(["close_date"], ensure_ascii=False, separators=(",", ":")),
            "initial_metadata_pass": 0,
        }
        row.update(overrides)
        return row

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
            {"token_usage": self._usage(), "search_count": 0},
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
        payload["purchase_price"] = {"cash": 10.0, "stock": None, "assets": None}
        payload["consideration_type"] = "mixed"

        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=payload,
            response_usage=self._usage(),
        )

        self.assertIsNone(params["price_total"])

    def test_build_update_params_web_search_only_totals_mixed_cash_and_stock(self) -> None:
        payload = self._valid_web_search_obj()
        payload["purchase_price"] = {"cash": 152_400_000.0, "stock": 493_637_000.0, "assets": None}
        payload["consideration_type"] = "mixed"

        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=payload,
            response_usage=self._usage(),
        )

        self.assertEqual(params["price_total"], 646_037_000.0)

    def test_build_update_params_web_search_only_totals_mixed_cash_and_assets(self) -> None:
        payload = self._valid_web_search_obj()
        payload["purchase_price"] = {"cash": 10.0, "stock": None, "assets": 5.0}
        payload["consideration_type"] = "mixed"

        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=payload,
            response_usage=self._usage(),
        )

        self.assertEqual(params["price_total"], 15.0)

    def test_build_update_params_web_search_only_totals_mixed_stock_and_assets(self) -> None:
        payload = self._valid_web_search_obj()
        payload["purchase_price"] = {"cash": None, "stock": 10.0, "assets": 5.0}
        payload["consideration_type"] = "mixed"

        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=payload,
            response_usage=self._usage(),
        )

        self.assertEqual(params["price_total"], 15.0)

    def test_build_tx_metadata_request_body_web_search_only_includes_sec_url(self) -> None:
        request_body = build_tx_metadata_request_body_web_search_only(
            {
                "target": "Target A",
                "acquirer": "Acquirer A",
                "filing_date": "2024-01-01",
                "url": "https://www.sec.gov/Archives/edgar/data/123/abc.htm",
            },
            model="gpt-5.4",
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
            model="gpt-5.4",
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
            model="gpt-5.4",
        )
        input_text = request_body.get("input")
        self.assertIsInstance(input_text, str)
        assert isinstance(input_text, str)
        self.assertIn("sec_filing_url: https://www.sec.gov/Archives/edgar/data/123/abc.htm", input_text)
        self.assertIn("acquirer: Acquirer A", input_text)
        self.assertIn("sec_filing_date: 2024-01-01", input_text)
        self.assertNotIn("target:", input_text)

    def test_build_web_search_retry_context_targets_close_date_and_deal_status(self) -> None:
        retry_context = build_web_search_retry_context(self._requeue_agreement_row())

        assert retry_context is not None
        self.assertEqual(retry_context["focus_fields"], ["close_date", "deal_status"])
        self.assertEqual(retry_context["known_values"]["consideration_type"], "all_cash")
        self.assertEqual(retry_context["known_values"]["target_public"], True)

    def test_build_web_search_retry_context_targets_cash_price_not_stock_for_cash_deal(self) -> None:
        retry_context = build_web_search_retry_context(
            self._requeue_agreement_row(
                transaction_price_cash=None,
                transaction_price_total=None,
                metadata_uncited_fields="[]",
            )
        )

        assert retry_context is not None
        self.assertEqual(retry_context["focus_fields"], ["purchase_price.cash", "close_date", "deal_status"])

    def test_build_web_search_retry_context_accepts_date_objects_from_db(self) -> None:
        retry_context = build_web_search_retry_context(
            self._requeue_agreement_row(
                announce_date=date(2024, 1, 1),
                close_date=date(2024, 2, 1),
                metadata_uncited_fields="[]",
            )
        )

        assert retry_context is not None
        self.assertEqual(retry_context["known_values"]["announce_date"], "2024-01-01")
        self.assertEqual(retry_context["known_values"]["close_date"], "2024-02-01")

    def test_build_web_search_retry_context_accepts_legacy_flat_metadata_sources_shape(self) -> None:
        retry_context = build_web_search_retry_context(
            self._requeue_agreement_row(
                metadata_sources=json.dumps(
                    {
                        "citations": [
                            self._citation("consideration_type"),
                            self._citation("purchase_price.cash"),
                            self._citation("target_public"),
                        ],
                        "notes": "Legacy flat metadata_sources payload.",
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )
        )

        assert retry_context is not None
        self.assertEqual(retry_context["known_values"]["consideration_type"], "all_cash")

    def test_build_web_search_retry_context_tolerates_invalid_metadata_sources_json(self) -> None:
        retry_context = build_web_search_retry_context(
            self._requeue_agreement_row(
                metadata_sources="{not valid json",
            )
        )

        assert retry_context is not None
        self.assertEqual(retry_context["known_values"]["consideration_type"], "all_cash")

    def test_build_web_search_retry_context_normalizes_db_native_bool_and_decimal_types(self) -> None:
        retry_context = build_web_search_retry_context(
            self._requeue_agreement_row(
                transaction_price_cash=Decimal("194000000.00"),
                transaction_price_total=Decimal("194000000.00"),
                target_pe=1,
                acquirer_pe=0,
                metadata_uncited_fields="[]",
            )
        )

        assert retry_context is not None
        self.assertEqual(retry_context["known_values"]["purchase_price.cash"], 194000000.0)
        self.assertEqual(retry_context["known_values"]["target_pe"], True)
        self.assertEqual(retry_context["known_values"]["acquirer_pe"], False)

    def test_build_tx_metadata_request_body_web_search_only_includes_retry_focus_and_locked_fields(self) -> None:
        retry_context = build_web_search_retry_context(self._requeue_agreement_row())
        request_body = build_tx_metadata_request_body_web_search_only(
            self._requeue_agreement_row(),
            model="gpt-5.4",
            retry_context=retry_context,
        )

        instructions = request_body.get("instructions")
        self.assertIsInstance(instructions, str)
        assert isinstance(instructions, str)
        self.assertIn("Treat the listed known trusted metadata as locked context", instructions)
        self.assertIn("stronger contradictory evidence", instructions)

        input_text = request_body.get("input")
        self.assertIsInstance(input_text, str)
        assert isinstance(input_text, str)
        self.assertIn("focus_fields: close_date, deal_status", input_text)
        self.assertIn("known_trusted_metadata:", input_text)
        self.assertIn("consideration_type: all_cash", input_text)
        self.assertIn("target_public: true", input_text)

    def test_merge_retry_web_search_response_preserves_locked_fields_on_null_retry(self) -> None:
        agreement = self._requeue_agreement_row()
        retry_payload = self._valid_web_search_obj()
        retry_payload["target_public"] = None
        retry_payload["acquirer_public"] = None
        retry_payload["purchase_price"] = {"cash": None, "stock": None, "assets": None}
        retry_payload["metadata_sources"] = {
            "citations": [
                self._citation("close_date"),
                self._citation("deal_status"),
            ],
            "notes": "Retry only found timeline evidence.",
        }

        merged = merge_retry_web_search_response(
            agreement=agreement,
            tx_metadata_obj=retry_payload,
        )

        self.assertEqual(merged["target_public"], True)
        self.assertEqual(merged["acquirer_public"], False)
        self.assertEqual(merged["purchase_price"]["cash"], 100.0)
        merged_fields = {
            citation["field"]
            for citation in cast(list[dict[str, object]], cast(dict[str, object], merged["metadata_sources"])["citations"])
            if isinstance(citation.get("field"), str)
        }
        self.assertIn("target_public", merged_fields)
        self.assertIn("purchase_price.cash", merged_fields)

    def test_merge_retry_web_search_response_accepts_cited_revision_for_locked_field(self) -> None:
        agreement = self._requeue_agreement_row()
        retry_payload = self._valid_web_search_obj()
        retry_payload["target_public"] = False
        retry_payload["metadata_sources"] = {
            "citations": [
                self._citation("consideration_type"),
                self._citation("purchase_price.cash"),
                self._citation("purchase_price.stock"),
                self._citation("purchase_price.assets"),
                self._citation("target_public"),
                self._citation("acquirer_public"),
                self._citation("target_industry"),
                self._citation("acquirer_industry"),
                self._citation("announce_date"),
                self._citation("deal_status"),
                self._citation("attitude"),
                self._citation("purpose"),
            ],
            "notes": "Found stronger contradictory evidence on target listing status.",
        }

        merged = merge_retry_web_search_response(
            agreement=agreement,
            tx_metadata_obj=retry_payload,
        )

        self.assertEqual(merged["target_public"], False)

    def test_merge_retry_web_search_response_dedupes_preserved_citations(self) -> None:
        agreement = self._requeue_agreement_row()
        retry_payload = self._valid_web_search_obj()
        retry_payload["target_public"] = None
        retry_payload["metadata_sources"] = {
            "citations": [
                self._citation("target_public"),
                self._citation("close_date"),
                self._citation("deal_status"),
            ],
            "notes": "Retry only found timeline evidence.",
        }

        merged = merge_retry_web_search_response(
            agreement=agreement,
            tx_metadata_obj=retry_payload,
        )

        merged_citations = cast(list[dict[str, object]], cast(dict[str, object], merged["metadata_sources"])["citations"])
        target_public_citations = [
            citation
            for citation in merged_citations
            if citation.get("field") == "target_public"
        ]
        self.assertEqual(len(target_public_citations), 1)

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

        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=payload,
            response_usage=self._usage(),
            filing_date=date.today().isoformat(),
            pending_max_age_years=3,
        )
        self.assertEqual(params["deal_status"], "pending")
        self.assertIsNone(params["close_date"])

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

    def test_build_update_params_web_search_only_nulls_invalid_citation_published_at(self) -> None:
        payload = self._valid_web_search_obj()
        metadata_sources = cast(dict[str, object], payload["metadata_sources"])
        citations = cast(list[dict[str, object]], metadata_sources["citations"])
        citations[0]["published_at"] = "2024-02-30"

        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=payload,
            response_usage=self._usage(),
        )

        metadata_payload = json.loads(params["metadata_sources"])
        self.assertIsNone(metadata_payload["metadata_sources"]["citations"][0]["published_at"])

    def test_build_update_params_web_search_only_nulls_complete_status_without_close_date(self) -> None:
        payload = self._valid_web_search_obj()
        payload["deal_status"] = "complete"
        payload["close_date"] = None
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
                self._citation("deal_status"),
                self._citation("attitude"),
                self._citation("purpose"),
            ],
            "notes": None,
        }

        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=payload,
            response_usage=self._usage(),
        )

        self.assertIsNone(params["deal_status"])
        self.assertIsNone(params["close_date"])

    def test_build_update_params_web_search_only_nulls_close_date_when_earlier_than_announce_date(self) -> None:
        payload = self._valid_web_search_obj()
        payload["deal_status"] = "complete"
        payload["announce_date"] = "2024-03-01"
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

        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=payload,
            response_usage=self._usage(),
        )

        self.assertEqual(params["announce_date"], "2024-03-01")
        self.assertIsNone(params["close_date"])
        self.assertIsNone(params["deal_status"])

    def test_build_update_params_web_search_only_allows_close_date_before_announce_for_private_target(self) -> None:
        payload = self._valid_web_search_obj()
        payload["target_public"] = False
        payload["deal_status"] = "complete"
        payload["announce_date"] = "2024-03-01"
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

        params = build_tx_metadata_update_params_web_search_only(
            agreement_uuid="agreement-1",
            tx_metadata_obj=payload,
            response_usage=self._usage(),
        )

        self.assertEqual(params["target_type"], "private")
        self.assertEqual(params["announce_date"], "2024-03-01")
        self.assertEqual(params["close_date"], "2024-02-01")
        self.assertEqual(params["deal_status"], "complete")

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
            build_web_search_runtime_metadata(
                response_usage=self._usage(),
                search_count=3,
            ),
            {"token_usage": self._usage(), "search_count": 3},
        )
        with self.assertRaisesRegex(TypeError, "response_usage.input_tokens"):
            _ = build_web_search_runtime_metadata(
                response_usage={"input_tokens": "bad", "output_tokens": 2, "total_tokens": 3},
                search_count=0,
            )
        with self.assertRaisesRegex(ValueError, "search_count must be >= 0"):
            _ = build_web_search_runtime_metadata(
                response_usage=self._usage(),
                search_count=-1,
            )


if __name__ == "__main__":
    _ = unittest.main()
