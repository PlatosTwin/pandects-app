import os
import json
import threading
import tempfile
import unittest
from typing import TYPE_CHECKING, cast
from datetime import date, datetime, timezone
from decimal import Decimal
from urllib.request import Request, urlopen
from unittest.mock import patch

import jwt
from sqlalchemy import text
from werkzeug.exceptions import BadRequest
from werkzeug.serving import BaseWSGIServer, make_server

from backend.mcp.metrics import get_mcp_metrics_registry

if TYPE_CHECKING:
    from backend.auth.mcp_runtime import ExternalIdentity


class McpClientHarness:
    def __init__(self, test_case: "McpTests", *, scope: str = "sections:search agreements:search agreements:read") -> None:
        self._test_case = test_case
        self._scope = scope
        self._client = test_case.app.test_client()
        self._tools: dict[str, dict[str, object]] = {}

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        response = self._client.post(
            "/mcp",
            headers={"Authorization": self._test_case._bearer(scope=self._scope)},
            json=payload,
        )
        self._test_case.assertEqual(response.status_code, 200)
        body = response.get_json()
        self._test_case.assertIsInstance(body, dict)
        self._test_case.assertEqual(body["jsonrpc"], "2.0")
        return cast(dict[str, object], body)

    def initialize(self) -> dict[str, object]:
        return self._post({"jsonrpc": "2.0", "id": 1, "method": "initialize"})

    def list_tools(self) -> list[dict[str, object]]:
        body = self._post({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = cast(list[dict[str, object]], cast(dict[str, object], body["result"])["tools"])
        self._tools = {cast(str, tool["name"]): tool for tool in tools}
        return tools

    def call_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        body = self._post(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        self._test_case.assertIn("result", body)
        tool = self._tools[name]
        import backend.mcp.tools as tools_module

        structured_content = cast(dict[str, object], cast(dict[str, object], body["result"])["structuredContent"])
        output_errors = tools_module._validate_output_against_schema(
            cast(dict[str, object], tool["outputSchema"]),
            structured_content,
        )
        self._test_case.assertEqual(output_errors, {})
        return structured_content


class LiveMcpHttpClientHarness:
    def __init__(self, test_case: "McpTests", *, scope: str = "sections:search agreements:search agreements:read") -> None:
        self._test_case = test_case
        self._scope = scope
        self._server: BaseWSGIServer | None = None
        self._thread: threading.Thread | None = None
        self._base_url = ""
        self._tools: dict[str, dict[str, object]] = {}

    def __enter__(self) -> "LiveMcpHttpClientHarness":
        transport_required = os.environ.get("MCP_ENABLE_LIVE_TRANSPORT_TESTS") == "1"
        try:
            self._server = make_server("127.0.0.1", 0, self._test_case.app, threaded=True)
        except (OSError, PermissionError, SystemExit):
            if transport_required:
                raise
            self._test_case.skipTest("Local loopback socket binding is not permitted in this environment.")
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        self._base_url = f"http://127.0.0.1:{self._server.server_port}"
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _post(self, payload: dict[str, object]) -> dict[str, object]:
        request = Request(
            f"{self._base_url}/mcp",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": self._test_case._bearer(scope=self._scope),
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urlopen(request, timeout=5) as response:
            self._test_case.assertEqual(response.status, 200)
            body = json.loads(response.read().decode("utf-8"))
        self._test_case.assertIsInstance(body, dict)
        self._test_case.assertEqual(body["jsonrpc"], "2.0")
        return cast(dict[str, object], body)

    def initialize(self) -> dict[str, object]:
        return self._post({"jsonrpc": "2.0", "id": 1, "method": "initialize"})

    def list_tools(self) -> list[dict[str, object]]:
        body = self._post({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = cast(list[dict[str, object]], cast(dict[str, object], body["result"])["tools"])
        self._tools = {cast(str, tool["name"]): tool for tool in tools}
        return tools

    def call_tool(self, name: str, arguments: dict[str, object]) -> dict[str, object]:
        body = self._post(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        )
        tool = self._tools[name]
        import backend.mcp.tools as tools_module

        structured_content = cast(dict[str, object], cast(dict[str, object], body["result"])["structuredContent"])
        output_errors = tools_module._validate_output_against_schema(
            cast(dict[str, object], tool["outputSchema"]),
            structured_content,
        )
        self._test_case.assertEqual(output_errors, {})
        return structured_content


def _normalized_tools_snapshot(tools: list[dict[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "name": tool["name"],
            "description": tool["description"],
            "annotations": tool["annotations"],
            "inputSchema": tool["inputSchema"],
            "outputSchema": tool["outputSchema"],
        }
        for tool in tools
    ]


def _normalized_capabilities_snapshot(payload: dict[str, object]) -> dict[str, object]:
    tools = cast(list[dict[str, object]], payload["tools"])
    return {
        "server": payload["server"],
        "auth_help": payload["auth_help"],
        "field_inventory": payload["field_inventory"],
        "concept_notes": payload["concept_notes"],
        "tool_limitations": payload["tool_limitations"],
        "tools": [
            {
                "name": tool["name"],
                "required_scopes": tool["required_scopes"],
                "pagination": tool["pagination"],
                "selection_hint": tool["selection_hint"],
                "negative_guidance": tool["negative_guidance"],
                "response_examples": tool["response_examples"],
                "access": tool["access"],
                "limits": tool["limits"],
                "input_schema": tool["input_schema"],
                "output_schema": tool["output_schema"],
            }
            for tool in tools
        ],
        "workflows": payload["workflows"],
    }


def _set_default_env(main_db_uri: str, auth_db_uri: str) -> None:
    os.environ["SKIP_MAIN_DB_REFLECTION"] = "1"
    os.environ["MAIN_DATABASE_URI"] = main_db_uri
    os.environ["MAIN_DB_SCHEMA"] = ""
    os.environ.setdefault("MARIADB_USER", "root")
    os.environ.setdefault("MARIADB_PASSWORD", "password")
    os.environ.setdefault("MARIADB_HOST", "127.0.0.1")
    os.environ.setdefault("MARIADB_DATABASE", "pdx")
    os.environ.setdefault("AUTH_SECRET_KEY", "test-auth-secret")
    os.environ["PUBLIC_API_BASE_URL"] = "http://localhost:5000"
    os.environ["PUBLIC_FRONTEND_BASE_URL"] = "http://localhost:8080"
    os.environ["AUTH_SESSION_TRANSPORT"] = "bearer"
    os.environ["AUTH_DATABASE_URI"] = auth_db_uri
    os.environ["MCP_OIDC_ISSUER"] = "https://issuer.example.com"
    os.environ["MCP_OIDC_AUDIENCE"] = "pandects-mcp"
    os.environ["MCP_OIDC_JWKS_URL"] = "https://issuer.example.com/jwks"
    os.environ["MCP_OIDC_SIGNING_ALGORITHMS"] = "HS256"
    os.environ["MCP_OIDC_AUTHORIZATION_SERVER_URL"] = "https://issuer.example.com"


class McpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        main_db = tempfile.NamedTemporaryFile(prefix="pandects_mcp_main_", suffix=".sqlite", delete=False)
        main_db.close()
        auth_db = tempfile.NamedTemporaryFile(prefix="pandects_mcp_auth_", suffix=".sqlite", delete=False)
        auth_db.close()
        _set_default_env(f"sqlite:///{main_db.name}", f"sqlite:///{auth_db.name}")

        import backend.app as app_module
        import backend.auth.mcp_runtime as mcp_runtime

        cls.app_module = app_module
        cls.mcp_runtime = mcp_runtime
        sqlite_uri = f"sqlite:///{main_db.name}"
        cls.app = cls.app_module.create_test_app(
            config_overrides={
                "MAIN_DB_SCHEMA": "",
                "SQLALCHEMY_DATABASE_URI": sqlite_uri,
                "SQLALCHEMY_BINDS": {"auth": f"sqlite:///{auth_db.name}"},
            }
        )
        with cls.app.app_context():
            engine = cls.app_module.db.engine
            cls.app_module.metadata.create_all(engine)
            cls.app_module.db.create_all(bind_key="auth")
            with engine.begin() as conn:
                conn.execute(
                    text(
                        "INSERT INTO agreements ("
                        "agreement_uuid, filing_date, target, acquirer, verified, url, "
                        "transaction_consideration, target_type, acquirer_type, target_industry, acquirer_industry, "
                        "deal_status, attitude, deal_type, purpose, target_pe, acquirer_pe"
                        ") VALUES ("
                        "'a1', '2020-01-01', 'Target A', 'Acquirer A', 1, 'http://example.com/a1', "
                        "'cash', 'public', 'public', 'tech', 'tech', 'complete', 'friendly', 'merger', 'strategic', 0, 0"
                        ")"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreements ("
                        "agreement_uuid, filing_date, target, acquirer, verified, url, "
                        "transaction_consideration, target_type, acquirer_type, target_industry, acquirer_industry, "
                        "deal_status, attitude, deal_type, purpose, target_pe, acquirer_pe"
                        ") VALUES ("
                        "'a2', '2021-06-15', 'Company B', 'Buyer B', 1, 'http://example.com/a2', "
                        "'cash', 'public', 'public', 'tech', 'tech', 'complete', 'friendly', 'merger', 'strategic', 0, 0"
                        ")"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO xml (agreement_uuid, xml, version, status, latest) VALUES "
                        "('a1', '<document><article>"
                        "<section uuid=\"00000000-0000-0000-0000-000000000001\"><text>KEEP</text></section>"
                        "<section uuid=\"00000000-0000-0000-0000-000000000002\"><text>HIDE</text></section>"
                        "<section uuid=\"00000000-0000-0000-0000-000000000003\"><text>Material Adverse Effect means any effect except disproportionate effects on the Company relative to other participants in the industry.</text></section>"
                        "</article></document>', 1, 'verified', 1)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO xml (agreement_uuid, xml, version, status, latest) VALUES "
                        "('a2', '<document><article>"
                        "<section uuid=\"00000000-0000-0000-0000-000000000004\"><text>Material Adverse Effect excludes changes affecting the industry generally, except to the extent they have a disproportionate effect on the Company.</text></section>"
                        "</article></document>', 1, 'verified', 1)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO sections (agreement_uuid, section_uuid, article_title, section_title, "
                        "xml_content, section_standard_id, xml_version) VALUES "
                        "('a1', '00000000-0000-0000-0000-000000000001', "
                        "'ARTICLE I', 'Section 1', '<section>TEXT</section>', '[\"s1\"]', 1)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO sections (agreement_uuid, section_uuid, article_title, section_title, "
                        "xml_content, section_standard_id, xml_version) VALUES "
                        "('a1', '00000000-0000-0000-0000-000000000002', "
                        "'ARTICLE II', 'Section 2', '<section>MORE</section>', '[\"s2\"]', 1)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO sections (agreement_uuid, section_uuid, article_title, section_title, "
                        "xml_content, section_standard_id, xml_version) VALUES "
                        "('a1', '00000000-0000-0000-0000-000000000003', "
                        "'ARTICLE III', 'Material Adverse Effect', "
                        "'<section><text>Material Adverse Effect means any effect except disproportionate effects on the Company relative to other participants in the industry.</text></section>', "
                        "'[\"2.1\", \"2.1.1\"]', 1)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO sections (agreement_uuid, section_uuid, article_title, section_title, "
                        "xml_content, section_standard_id, xml_version) VALUES "
                        "('a2', '00000000-0000-0000-0000-000000000004', "
                        "'ARTICLE I', 'Material Adverse Effect', "
                        "'<section><text>Material Adverse Effect excludes changes affecting the industry generally, except to the extent they have a disproportionate effect on the Company.</text></section>', "
                        "'[\"2.1\", \"2.1.1\"]', 1)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO latest_sections_search ("
                        "section_uuid, agreement_uuid, filing_date, prob_filing, filing_company_name, "
                        "filing_company_cik, form_type, exhibit_type, target, acquirer, "
                        "transaction_price_total, transaction_price_stock, transaction_price_cash, "
                        "transaction_price_assets, transaction_consideration, target_type, acquirer_type, "
                        "target_industry, acquirer_industry, announce_date, close_date, deal_status, "
                        "attitude, deal_type, purpose, target_pe, acquirer_pe, verified, url, "
                        "section_standard_ids, article_title, section_title"
                        ") VALUES ("
                        "'00000000-0000-0000-0000-000000000001', 'a1', '2020-01-01', NULL, NULL, NULL, "
                        "NULL, NULL, 'Target A', 'Acquirer A', NULL, NULL, NULL, NULL, 'cash', 'public', "
                        "'public', 'tech', 'tech', NULL, NULL, 'complete', 'friendly', 'merger', "
                        "'strategic', 0, 0, 1, 'http://example.com/a1', '[\"s1\"]', 'ARTICLE I', 'Section 1'"
                        ")"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO latest_sections_search ("
                        "section_uuid, agreement_uuid, filing_date, prob_filing, filing_company_name, "
                        "filing_company_cik, form_type, exhibit_type, target, acquirer, "
                        "transaction_price_total, transaction_price_stock, transaction_price_cash, "
                        "transaction_price_assets, transaction_consideration, target_type, acquirer_type, "
                        "target_industry, acquirer_industry, announce_date, close_date, deal_status, "
                        "attitude, deal_type, purpose, target_pe, acquirer_pe, verified, url, "
                        "section_standard_ids, article_title, section_title"
                        ") VALUES ("
                        "'00000000-0000-0000-0000-000000000002', 'a1', '2020-01-01', NULL, NULL, NULL, "
                        "NULL, NULL, 'Target A', 'Acquirer A', NULL, NULL, NULL, NULL, 'cash', 'public', "
                        "'public', 'tech', 'tech', NULL, NULL, 'complete', 'friendly', 'merger', "
                        "'strategic', 0, 0, 1, 'http://example.com/a1', '[\"s2\"]', 'ARTICLE II', 'Section 2'"
                        ")"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO latest_sections_search ("
                        "section_uuid, agreement_uuid, filing_date, prob_filing, filing_company_name, "
                        "filing_company_cik, form_type, exhibit_type, target, acquirer, "
                        "transaction_price_total, transaction_price_stock, transaction_price_cash, "
                        "transaction_price_assets, transaction_consideration, target_type, acquirer_type, "
                        "target_industry, acquirer_industry, announce_date, close_date, deal_status, "
                        "attitude, deal_type, purpose, target_pe, acquirer_pe, verified, url, "
                        "section_standard_ids, article_title, section_title"
                        ") VALUES ("
                        "'00000000-0000-0000-0000-000000000003', 'a1', '2020-01-01', NULL, NULL, NULL, "
                        "NULL, NULL, 'Target A', 'Acquirer A', NULL, NULL, NULL, NULL, 'cash', 'public', "
                        "'public', 'tech', 'tech', NULL, NULL, 'complete', 'friendly', 'merger', "
                        "'strategic', 0, 0, 1, 'http://example.com/a1', '[\"2.1\", \"2.1.1\"]', 'ARTICLE III', 'Material Adverse Effect'"
                        ")"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO latest_sections_search ("
                        "section_uuid, agreement_uuid, filing_date, prob_filing, filing_company_name, "
                        "filing_company_cik, form_type, exhibit_type, target, acquirer, "
                        "transaction_price_total, transaction_price_stock, transaction_price_cash, "
                        "transaction_price_assets, transaction_consideration, target_type, acquirer_type, "
                        "target_industry, acquirer_industry, announce_date, close_date, deal_status, "
                        "attitude, deal_type, purpose, target_pe, acquirer_pe, verified, url, "
                        "section_standard_ids, article_title, section_title"
                        ") VALUES ("
                        "'00000000-0000-0000-0000-000000000004', 'a2', '2021-06-15', NULL, NULL, NULL, "
                        "NULL, NULL, 'Company B', 'Buyer B', NULL, NULL, NULL, NULL, 'cash', 'public', "
                        "'public', 'tech', 'tech', NULL, NULL, 'complete', 'friendly', 'merger', "
                        "'strategic', 0, 0, 1, 'http://example.com/a2', '[\"2.1\", \"2.1.1\"]', 'ARTICLE I', 'Material Adverse Effect'"
                        ")"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO latest_sections_search_standard_ids (standard_id, section_uuid, agreement_uuid) "
                        "VALUES ('s1', '00000000-0000-0000-0000-000000000001', 'a1')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO latest_sections_search_standard_ids (standard_id, section_uuid, agreement_uuid) "
                        "VALUES ('s2', '00000000-0000-0000-0000-000000000002', 'a1')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO latest_sections_search_standard_ids (standard_id, section_uuid, agreement_uuid) "
                        "VALUES ('2.1', '00000000-0000-0000-0000-000000000003', 'a1'), "
                        "('2.1.1', '00000000-0000-0000-0000-000000000003', 'a1'), "
                        "('2.1', '00000000-0000-0000-0000-000000000004', 'a2'), "
                        "('2.1.1', '00000000-0000-0000-0000-000000000004', 'a2')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO clauses (clause_uuid, agreement_uuid, section_uuid, xml_version, module, clause_order, "
                        "anchor_label, start_char, end_char, clause_text, source_method, context_type) VALUES "
                        "('clause-a1-1', 'a1', '00000000-0000-0000-0000-000000000001', 1, 'tax', 1, "
                        "'(a)', 0, 20, 'Parent shall bear all transfer taxes.', 'enumerated_split', 'operative'), "
                        "('clause-a1-2', 'a1', '00000000-0000-0000-0000-000000000001', 1, 'tax', 2, "
                        "'(b)', 21, 60, 'The merger is intended to qualify as tax-free.', 'enumerated_split', 'rep_warranty')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO tax_clause_assignments (clause_uuid, standard_id, is_gold_label, model_name, assigned_at) VALUES "
                        "('clause-a1-1', 'tax_transfer', 1, 'gpt-5-mini', '2026-04-02T00:00:00Z'), "
                        "('clause-a1-2', 'tax_treatment', 1, 'gpt-5-mini', '2026-04-02T00:00:00Z')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO counsel (counsel_id, canonical_name, canonical_name_normalized) VALUES "
                        "(1, 'Wachtell, Lipton, Rosen & Katz', 'wachtell lipton rosen and katz'), "
                        "(2, 'Skadden, Arps, Slate, Meagher & Flom LLP', 'skadden arps slate meagher flom llp')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreement_counsel (agreement_uuid, side, position, raw_name, counsel_id) VALUES "
                        "('a1', 'target', 1, 'Wachtell', 1), "
                        "('a1', 'acquirer', 1, 'Skadden', 2), "
                        "('a2', 'target', 1, 'Skadden', 2), "
                        "('a2', 'acquirer', 1, 'Wachtell', 1)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO taxonomy_l1 (standard_id, label) VALUES ('1', 'Deal Protection')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO taxonomy_l1 (standard_id, label) VALUES ('2', 'Definitions')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO taxonomy_l2 (standard_id, label, parent_id) VALUES ('1.1', 'Fiduciary Out', '1')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO taxonomy_l2 (standard_id, label, parent_id) VALUES ('2.1', 'Material Adverse Effect', '2')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO taxonomy_l3 (standard_id, label, parent_id) VALUES ('1.1.1', 'Change Of Recommendation', '1.1')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO taxonomy_l3 (standard_id, label, parent_id) VALUES ('2.1.1', 'Disproportionate Effects', '2.1')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO tax_clause_taxonomy_l1 (standard_id, label) VALUES ('tax', 'Tax')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO tax_clause_taxonomy_l2 (standard_id, label, parent_id) VALUES ('tax.1', 'Treatment', 'tax')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO tax_clause_taxonomy_l3 (standard_id, label, parent_id) VALUES ('tax.1.1', 'Tax-Free Reorg', 'tax.1')"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO naics_sectors (super_sector, sector_group, sector_desc, sector_code) VALUES "
                        "('Goods-Producing Industries', 'Natural Resources and Mining', "
                        "'Agriculture, Forestry, Fishing and Hunting', 11)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO naics_sub_sectors (sub_sector_desc, sub_sector_code, sector_code) VALUES "
                        "('Crop Production', 111, 11)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS summary_data ("
                        "count_agreements INTEGER NOT NULL, count_sections INTEGER NOT NULL, count_pages INTEGER NOT NULL)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO summary_data (count_agreements, count_sections, count_pages) VALUES (1, 2, 5)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS agreement_ownership_mix_summary ("
                        "year INTEGER NOT NULL, target_bucket TEXT NOT NULL, deal_count INTEGER NOT NULL, total_transaction_value REAL NOT NULL)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS agreement_ownership_deal_size_summary ("
                        "year INTEGER NOT NULL, target_bucket TEXT NOT NULL, deal_count INTEGER NOT NULL, "
                        "p25_transaction_value REAL NULL, median_transaction_value REAL NULL, p75_transaction_value REAL NULL)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS agreement_buyer_type_matrix_summary ("
                        "target_bucket TEXT NOT NULL, buyer_bucket TEXT NOT NULL, deal_count INTEGER NOT NULL, median_transaction_value REAL NULL)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS agreement_target_industry_summary ("
                        "year INTEGER NOT NULL, industry TEXT NOT NULL, deal_count INTEGER NOT NULL, total_transaction_value REAL NOT NULL)"
                    )
                )
                conn.execute(
                    text(
                        "CREATE TABLE IF NOT EXISTS agreement_industry_pairing_summary ("
                        "target_industry TEXT NOT NULL, acquirer_industry TEXT NOT NULL, deal_count INTEGER NOT NULL, total_transaction_value REAL NOT NULL)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreement_ownership_mix_summary (year, target_bucket, deal_count, total_transaction_value) VALUES "
                        "(2020, 'public', 1, 50000000)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreement_ownership_deal_size_summary "
                        "(year, target_bucket, deal_count, p25_transaction_value, median_transaction_value, p75_transaction_value) VALUES "
                        "(2020, 'public', 1, 50000000, 50000000, 50000000)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreement_buyer_type_matrix_summary "
                        "(target_bucket, buyer_bucket, deal_count, median_transaction_value) VALUES "
                        "('public', 'public_buyer', 1, 50000000)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreement_target_industry_summary "
                        "(year, industry, deal_count, total_transaction_value) VALUES "
                        "(2020, '111', 1, 50000000)"
                    )
                )
                conn.execute(
                    text(
                        "INSERT INTO agreement_industry_pairing_summary "
                        "(target_industry, acquirer_industry, deal_count, total_transaction_value) VALUES "
                        "('111', '111', 1, 50000000)"
                    )
                )

            user = cls.app_module.AuthUser()
            user.id = "00000000-0000-0000-0000-0000000000f1"
            user.email = "mcp@example.com"
            user.password_hash = None
            user.email_verified_at = datetime.now(timezone.utc).replace(tzinfo=None)
            cls.app_module.db.session.add(user)
            cls.app_module.db.session.flush()

            subject = cls.app_module.AuthExternalSubject()
            subject.user_id = user.id
            subject.issuer = "https://issuer.example.com"
            subject.subject = "sub-123"
            cls.app_module.db.session.add(subject)
            cls.app_module.db.session.commit()

        class DummyJwkClient:
            def get_signing_key_from_jwt(self, _token: str):
                return type("SigningKey", (), {"key": "test-signing-secret"})()

        cls.mcp_runtime._mcp_jwk_client = DummyJwkClient()
        cls.mcp_runtime._mcp_identity_provider = None

    def setUp(self) -> None:
        get_mcp_metrics_registry().reset()

    def _bearer(self, scope: str = "sections:search agreements:search agreements:read") -> str:
        return self._bearer_for_subject(subject="sub-123", scope=scope)

    def _bearer_for_subject(
        self,
        *,
        subject: str,
        scope: str = "sections:search agreements:search agreements:read",
    ) -> str:
        token = jwt.encode(
            {
                "iss": "https://issuer.example.com",
                "sub": subject,
                "aud": "pandects-mcp",
                "scope": scope,
            },
            "test-signing-secret",
            algorithm="HS256",
        )
        return f"Bearer {token}"

    def _call_tool(self, name: str, arguments: dict[str, object], *, scope: str | None = None):
        client = self.app.test_client()
        return client.post(
            "/mcp",
            headers={"Authorization": self._bearer(scope=scope or "sections:search agreements:search agreements:read")},
            json={
                "jsonrpc": "2.0",
                "id": 999,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            },
        )

    def test_protected_resource_metadata(self):
        client = self.app.test_client()
        res = client.get("/.well-known/oauth-protected-resource")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["resource"], self.mcp_runtime.mcp_resource_url())
        self.assertIn("http://localhost:5000/v1/auth/oauth", payload["authorization_servers"])

    def test_protected_resource_metadata_hides_runtime_details(self):
        client = self.app.test_client()
        with patch(
            "backend.mcp.routes.mcp_protected_resource_metadata",
            side_effect=RuntimeError("OIDC discovery document missing jwks_uri."),
        ):
            with self.assertLogs("backend.mcp.routes", level="WARNING") as log_context:
                res = client.get("/.well-known/oauth-protected-resource")
        self.assertEqual(res.status_code, 503)
        payload = res.get_json()
        self.assertEqual(payload["error"], "Service Unavailable")
        self.assertEqual(
            payload["message"],
            "Protected resource metadata is unavailable right now.",
        )
        self.assertNotIn("jwks_uri", payload["message"])
        self.assertTrue(
            any("mcp_protected_resource_metadata_unavailable" in line for line in log_context.output)
        )

    def test_permission_error_hides_runtime_details(self):
        client = self.app.test_client()
        with patch(
            "backend.mcp.routes.call_tool",
            side_effect=PermissionError("missing agreements:read scope for private tool"),
        ):
            with self.assertLogs("backend.mcp.routes", level="WARNING") as log_context:
                res = client.post(
                    "/mcp",
                    headers={"Authorization": self._bearer()},
                    json={
                        "jsonrpc": "2.0",
                        "id": 80,
                        "method": "tools/call",
                        "params": {"name": "search_agreements", "arguments": {"query": "Target"}},
                    },
                )
        self.assertEqual(res.status_code, 403)
        body = res.get_json()
        self.assertEqual(body["error"]["message"], "You do not have permission to call this tool.")
        self.assertEqual(body["error"]["data"]["category"], "authorization")
        self.assertTrue(any("mcp_tool_permission_denied" in line for line in log_context.output))

    def test_http_exception_hides_runtime_details(self):
        client = self.app.test_client()
        with patch(
            "backend.mcp.routes.call_tool",
            side_effect=BadRequest(description="raw backend details"),
        ):
            res = client.post(
                "/mcp",
                headers={"Authorization": self._bearer()},
                json={
                    "jsonrpc": "2.0",
                    "id": 81,
                    "method": "tools/call",
                    "params": {"name": "search_agreements", "arguments": {"query": "Target"}},
                },
            )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body["error"]["message"], "Tool request was invalid.")

    def test_mcp_requires_bearer_token(self):
        client = self.app.test_client()
        res = client.post("/mcp", json={"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        self.assertEqual(res.status_code, 401)
        self.assertIn("WWW-Authenticate", res.headers)
        payload = res.get_json()
        self.assertEqual(payload["jsonrpc"], "2.0")
        self.assertEqual(payload["id"], 1)
        self.assertEqual(payload["error"]["data"]["category"], "authentication")
        self.assertEqual(payload["error"]["data"]["reason"], "missing_token")
        self.assertEqual(payload["error"]["data"]["action"], "login")
        self.assertIn("Sign in", payload["error"]["data"]["client_message"])

    def test_mcp_sse_probe_requires_bearer_token(self):
        client = self.app.test_client()
        res = client.get("/mcp", headers={"Accept": "text/event-stream"})
        self.assertEqual(res.status_code, 401)
        self.assertIn("WWW-Authenticate", res.headers)
        payload = res.get_json()
        self.assertEqual(payload["jsonrpc"], "2.0")
        self.assertIsNone(payload["id"])
        self.assertEqual(payload["error"]["data"]["category"], "authentication")
        self.assertEqual(payload["error"]["data"]["reason"], "missing_token")
        self.assertEqual(payload["error"]["data"]["action"], "login")

    def test_mcp_sse_probe_returns_stream_for_authenticated_clients(self):
        client = self.app.test_client()
        res = client.get(
            "/mcp",
            headers={
                "Authorization": self._bearer(),
                "Accept": "text/event-stream",
            },
        )
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers["Cache-Control"], "no-store")
        self.assertIn("text/event-stream", res.headers["Content-Type"])
        body = res.get_data(as_text=True)
        self.assertIn("id: ", body)
        self.assertIn("retry: 1000", body)
        self.assertIn("data:", body)

    def test_initialize_and_tools_list(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer()},
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        )
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["result"]["serverInfo"]["name"], "pandects-mcp")

        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer()},
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )
        self.assertEqual(res.status_code, 200)
        tools = res.get_json()["result"]["tools"]
        self.assertEqual(
            [tool["name"] for tool in tools],
            [
                "search_agreements",
                "search_sections",
                "list_agreements",
                "list_agreement_sections",
                "list_agreement_sections_batch",
                "get_agreement",
                "get_section",
                "get_agreement_tax_clauses",
                "get_section_tax_clauses",
                "list_filter_options",
                "suggest_clause_families",
                "get_section_snippet",
                "get_server_metrics",
                "get_server_capabilities",
                "get_clause_taxonomy",
                "get_tax_clause_taxonomy",
                "get_counsel_catalog",
                "get_naics_catalog",
                "get_agreements_summary",
                "get_agreement_trends",
            ],
        )
        self.assertEqual(payload["jsonrpc"], "2.0")
        self.assertEqual(res.get_json()["jsonrpc"], "2.0")

    def test_authorization_server_metadata_served_at_host_root(self):
        client = self.app.test_client()
        res = client.get("/.well-known/oauth-authorization-server")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertIn("issuer", payload)
        self.assertIn("registration_endpoint", payload)
        self.assertIn("authorization_endpoint", payload)
        self.assertIn("token_endpoint", payload)
        self.assertIn("S256", payload["code_challenge_methods_supported"])

    def test_authorization_server_metadata_served_at_rfc8414_path(self):
        client = self.app.test_client()
        res = client.get("/.well-known/oauth-authorization-server/v1/auth/oauth")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertIn("registration_endpoint", payload)

    def test_authorization_server_metadata_rejects_unknown_issuer_path(self):
        client = self.app.test_client()
        res = client.get("/.well-known/oauth-authorization-server/bogus/path")
        self.assertEqual(res.status_code, 404)

    def test_openid_configuration_served_at_host_root(self):
        client = self.app.test_client()
        res = client.get("/.well-known/openid-configuration")
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertIn("issuer", payload)

    def test_initialize_issues_session_id_and_protocol_version_header(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer()},
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("Mcp-Session-Id", res.headers)
        self.assertTrue(res.headers["Mcp-Session-Id"].strip())
        self.assertIn("MCP-Protocol-Version", res.headers)

    def test_post_returns_sse_when_client_prefers_event_stream(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={
                "Authorization": self._bearer(),
                "Accept": "text/event-stream",
            },
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/event-stream", res.headers["Content-Type"])
        body = res.get_data(as_text=True)
        self.assertIn("event: message", body)
        self.assertIn('"jsonrpc":"2.0"', body)
        self.assertIn('"result"', body)

    def test_post_returns_json_when_client_prefers_application_json(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={
                "Authorization": self._bearer(),
                "Accept": "application/json, text/event-stream;q=0.9",
            },
            json={"jsonrpc": "2.0", "id": 1, "method": "ping"},
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("application/json", res.headers["Content-Type"])

    def test_progress_token_streams_progress_notifications_then_result(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={
                "Authorization": self._bearer(),
                "Accept": "text/event-stream",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "get_agreements_summary",
                    "arguments": {},
                    "_meta": {"progressToken": "progress-token-abc"},
                },
            },
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("text/event-stream", res.headers["Content-Type"])
        body = res.get_data(as_text=True)
        events = [block for block in body.split("\n\n") if block.strip()]
        # Expect: start progress, complete progress, final result.
        self.assertEqual(len(events), 3, body)
        self.assertIn("notifications/progress", events[0])
        self.assertIn("progress-token-abc", events[0])
        self.assertIn("notifications/progress", events[1])
        self.assertIn("progress-token-abc", events[1])
        self.assertIn('"result"', events[2])
        self.assertIn('"structuredContent"', events[2])

    def test_progress_token_without_sse_preference_uses_regular_json(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={
                "Authorization": self._bearer(),
                "Accept": "application/json",
            },
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "get_agreements_summary",
                    "arguments": {},
                    "_meta": {"progressToken": "progress-token-xyz"},
                },
            },
        )
        self.assertEqual(res.status_code, 200)
        self.assertIn("application/json", res.headers["Content-Type"])
        self.assertIn("result", res.get_json())

    def test_delete_mcp_requires_auth_and_returns_204(self):
        client = self.app.test_client()
        res = client.delete("/mcp")
        self.assertEqual(res.status_code, 401)
        res = client.delete("/mcp", headers={"Authorization": self._bearer()})
        self.assertEqual(res.status_code, 204)

    def test_prompts_list_and_get(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer()},
            json={"jsonrpc": "2.0", "id": 1, "method": "prompts/list"},
        )
        self.assertEqual(res.status_code, 200)
        prompts = res.get_json()["result"]["prompts"]
        self.assertTrue(any(prompt["name"] == "compare_agreements" for prompt in prompts))

        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer()},
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "prompts/get",
                "params": {
                    "name": "compare_agreements",
                    "arguments": {
                        "agreement_a": "deal-a",
                        "agreement_b": "deal-b",
                        "focus": "MAE",
                    },
                },
            },
        )
        self.assertEqual(res.status_code, 200)
        messages = res.get_json()["result"]["messages"]
        text = messages[0]["content"]["text"]
        self.assertIn("deal-a", text)
        self.assertIn("deal-b", text)
        self.assertIn("MAE", text)

    def test_resources_list_and_read(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer()},
            json={"jsonrpc": "2.0", "id": 1, "method": "resources/list"},
        )
        self.assertEqual(res.status_code, 200)
        uris = {resource["uri"] for resource in res.get_json()["result"]["resources"]}
        self.assertIn("pandects://capabilities", uris)

        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer()},
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "resources/read",
                "params": {"uri": "pandects://capabilities"},
            },
        )
        self.assertEqual(res.status_code, 200)
        contents = res.get_json()["result"]["contents"]
        self.assertEqual(contents[0]["uri"], "pandects://capabilities")
        self.assertIn("application/json", contents[0]["mimeType"])

    def test_tool_call_returns_jsonrpc_result_contract(self):
        client = self.app.test_client()
        with self.assertLogs("backend.mcp.routes", level="INFO") as log_context:
            res = client.post(
                "/mcp",
                headers={"Authorization": self._bearer()},
                json={
                    "jsonrpc": "2.0",
                    "id": 77,
                    "method": "tools/call",
                    "params": {"name": "search_agreements", "arguments": {"query": "Target"}},
                },
            )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body["jsonrpc"], "2.0")
        self.assertEqual(body["id"], 77)
        self.assertIn("result", body)
        self.assertEqual(body["result"]["content"][0]["type"], "text")
        self.assertIsInstance(body["result"]["content"][0]["text"], str)
        structured = body["result"]["structuredContent"]
        self.assertEqual(structured["results"][0]["agreement_uuid"], "a1")
        self.assertTrue(any("mcp_tool_call" in line for line in log_context.output))

    def test_output_contract_violation_returns_jsonrpc_server_error(self):
        import backend.mcp.tools as tools_module

        client = self.app.test_client()
        with patch(
            "backend.mcp.routes.call_tool",
            side_effect=tools_module.McpOutputValidationError({"structuredContent.page": "Missing required field."}),
        ):
            res = client.post(
                "/mcp",
                headers={"Authorization": self._bearer()},
                json={
                    "jsonrpc": "2.0",
                    "id": 79,
                    "method": "tools/call",
                    "params": {"name": "search_agreements", "arguments": {"query": "Target"}},
                },
            )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body["jsonrpc"], "2.0")
        self.assertEqual(body["id"], 79)
        self.assertEqual(body["error"]["code"], -32603)
        self.assertEqual(body["error"]["message"], "Tool result violated the advertised output contract.")
        self.assertIn("structuredContent.page", body["error"]["data"])

    def test_invalid_tool_arguments_return_jsonrpc_error_contract(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer()},
            json={
                "jsonrpc": "2.0",
                "id": 78,
                "method": "tools/call",
                "params": {"name": "search_sections", "arguments": {"target_counsels": ["Wachtell"]}},
            },
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body["jsonrpc"], "2.0")
        self.assertEqual(body["id"], 78)
        self.assertEqual(body["error"]["code"], -32602)
        self.assertEqual(body["error"]["message"], "Invalid tool arguments.")
        self.assertIn("target_counsels", body["error"]["data"])

    def test_tools_list_advertises_structured_counsel_filters(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer()},
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        )
        self.assertEqual(res.status_code, 200)
        tools = {
            tool["name"]: tool["inputSchema"]
            for tool in res.get_json()["result"]["tools"]
        }
        for schema in tools.values():
            self.assertEqual(schema["type"], "object")
            self.assertFalse(schema["additionalProperties"])

        search_agreements_schema = tools["search_agreements"]
        self.assertIn("target_counsel", search_agreements_schema["properties"])
        self.assertIn("acquirer_counsel", search_agreements_schema["properties"])
        self.assertEqual(
            search_agreements_schema["properties"]["sort_by"]["enum"],
            ["year", "target", "acquirer"],
        )
        self.assertIn(
            "100M - 250M",
            search_agreements_schema["properties"]["transaction_price_total"]["items"]["enum"],
        )
        search_agreements_tool = next(tool for tool in res.get_json()["result"]["tools"] if tool["name"] == "search_agreements")
        self.assertIn("examples", search_agreements_tool)
        self.assertIn("outputSchema", search_agreements_tool)
        self.assertIn("returned_count", search_agreements_tool["outputSchema"]["properties"])
        self.assertIn("count_metadata", search_agreements_tool["outputSchema"]["properties"])
        self.assertIn("interpretation", search_agreements_tool["outputSchema"]["properties"])
        self.assertEqual(search_agreements_tool["annotations"]["pagination"], "page")
        self.assertIn("agreements:search", search_agreements_tool["annotations"]["requiredScopes"])

        search_sections_schema = tools["search_sections"]
        self.assertIn("target_counsel", search_sections_schema["properties"])
        self.assertIn("acquirer_counsel", search_sections_schema["properties"])
        self.assertEqual(search_sections_schema["properties"]["count_mode"]["enum"], ["auto", "exact"])
        self.assertNotIn("target_counsels", search_sections_schema["properties"])
        self.assertNotIn("acquirer_counsels", search_sections_schema["properties"])
        self.assertIn("deal_type", search_sections_schema["properties"]["metadata"]["items"]["enum"])
        self.assertEqual(search_sections_schema["properties"]["sort_direction"]["enum"], ["asc", "desc"])
        self.assertIn("count_metadata", next(tool for tool in res.get_json()["result"]["tools"] if tool["name"] == "search_sections")["outputSchema"]["properties"])

        list_agreements_schema = tools["list_agreements"]
        self.assertIn("target_counsel", list_agreements_schema["properties"])
        self.assertIn("acquirer_counsel", list_agreements_schema["properties"])

        list_filter_options_schema = tools["list_filter_options"]
        self.assertIn("deal_types", list_filter_options_schema["properties"]["fields"]["items"]["enum"])
        self.assertIn("transaction_price_totals", list_filter_options_schema["properties"]["fields"]["items"]["enum"])

        concept_schema = tools["suggest_clause_families"]
        self.assertEqual(concept_schema["properties"]["taxonomy"]["enum"], ["clauses", "tax_clauses"])

        snippet_schema = tools["get_section_snippet"]
        self.assertIn("focus_terms", snippet_schema["properties"])

        capabilities_schema = tools["get_server_capabilities"]
        self.assertEqual(capabilities_schema["properties"], {})

        metrics_schema = tools["get_server_metrics"]
        self.assertEqual(metrics_schema["properties"], {})

        agreement_tax_schema = tools["get_agreement_tax_clauses"]
        self.assertEqual(agreement_tax_schema["required"], ["agreement_uuid"])
        self.assertNotIn("focus_section_uuid", agreement_tax_schema["properties"])

    def test_end_to_end_jsonrpc_workflow_from_initialize_to_retrieval(self):
        client = self.app.test_client()
        init_res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer()},
            json={"jsonrpc": "2.0", "id": 90, "method": "initialize"},
        )
        self.assertEqual(init_res.status_code, 200)
        self.assertEqual(init_res.get_json()["result"]["serverInfo"]["name"], "pandects-mcp")

        tools_res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer()},
            json={"jsonrpc": "2.0", "id": 91, "method": "tools/list"},
        )
        self.assertEqual(tools_res.status_code, 200)
        tool_names = [tool["name"] for tool in tools_res.get_json()["result"]["tools"]]
        self.assertIn("list_filter_options", tool_names)
        self.assertIn("search_agreements", tool_names)
        self.assertIn("get_server_metrics", tool_names)
        self.assertIn("get_server_capabilities", tool_names)

        filter_res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer()},
            json={
                "jsonrpc": "2.0",
                "id": 92,
                "method": "tools/call",
                "params": {"name": "list_filter_options", "arguments": {"fields": ["target_counsels"]}},
            },
        )
        self.assertEqual(filter_res.status_code, 200)
        filter_body = filter_res.get_json()["result"]["structuredContent"]
        retrieval_param = filter_body["retrieval_parameter_map"]["target_counsels"]
        canonical_name = "Wachtell, Lipton, Rosen & Katz"
        self.assertIn(canonical_name, filter_body["target_counsels"])

        search_res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer()},
            json={
                "jsonrpc": "2.0",
                "id": 93,
                "method": "tools/call",
                "params": {
                    "name": "search_agreements",
                    "arguments": {retrieval_param: [canonical_name]},
                },
            },
        )
        self.assertEqual(search_res.status_code, 200)
        search_body = search_res.get_json()["result"]["structuredContent"]
        agreement_uuid = search_body["results"][0]["agreement_uuid"]
        self.assertEqual(agreement_uuid, "a1")

        agreement_res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer(scope="agreements:read")},
            json={
                "jsonrpc": "2.0",
                "id": 94,
                "method": "tools/call",
                "params": {"name": "get_agreement", "arguments": {"agreement_uuid": agreement_uuid}},
            },
        )
        self.assertEqual(agreement_res.status_code, 200)
        agreement_body = agreement_res.get_json()["result"]["structuredContent"]
        self.assertEqual(agreement_body["target"], "Target A")

    def test_search_agreements_tool(self):
        res = self._call_tool("search_agreements", {"query": "Target"})
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()["result"]["structuredContent"]
        self.assertEqual(payload["page"], 1)
        self.assertEqual(payload["returned_count"], 1)
        self.assertEqual(payload["results"][0]["agreement_uuid"], "a1")
        self.assertEqual(payload["count_metadata"]["mode"], "exact")
        self.assertEqual(payload["count_metadata"]["method"], "query_count")
        self.assertEqual(payload["interpretation"]["heuristics_used"], ["prefix_name_match"])

    def test_search_agreements_interpretation_marks_exact_filters(self):
        res = self._call_tool(
            "search_agreements",
            {"target_counsel": ["Wachtell, Lipton, Rosen & Katz"]},
        )
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()["result"]["structuredContent"]
        applied_filters = payload["interpretation"]["applied_filters"]
        self.assertIn(
            {
                "field": "target_counsel",
                "representation": "first_class_agreement_field",
                "match_kind": "exact_metadata_filter",
            },
            applied_filters,
        )

    def test_search_agreements_supports_counsel_filters(self):
        res = self._call_tool(
            "search_agreements",
            {"target_counsel": ["Wachtell, Lipton, Rosen & Katz"]},
        )
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()["result"]["structuredContent"]
        self.assertEqual(payload["returned_count"], 1)
        self.assertEqual(payload["results"][0]["agreement_uuid"], "a1")

    def test_get_section_tool(self):
        res = self._call_tool(
            "get_section",
            {"section_uuid": "00000000-0000-0000-0000-000000000001"},
        )
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()["result"]["structuredContent"]
        self.assertEqual(payload["agreement_uuid"], "a1")
        self.assertEqual(payload["standard_id"], ["s1"])

    def test_list_agreement_sections_tool(self):
        res = self._call_tool("list_agreement_sections", {"agreement_uuid": "a1"})
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()["result"]["structuredContent"]
        self.assertEqual(payload["agreement_uuid"], "a1")
        self.assertEqual(payload["returned_count"], 3)
        self.assertEqual(payload["results"][0]["section_uuid"], "00000000-0000-0000-0000-000000000001")

    def test_tax_clause_tools(self):
        agreement_res = self._call_tool(
            "get_agreement_tax_clauses",
            {"agreement_uuid": "a1"},
            scope="agreements:read",
        )
        self.assertEqual(agreement_res.status_code, 200)
        agreement_payload = agreement_res.get_json()["result"]["structuredContent"]
        self.assertEqual(agreement_payload["returned_count"], 2)
        self.assertEqual(agreement_payload["clauses"][0]["standard_ids"], ["tax_transfer"])

        section_res = self._call_tool(
            "get_section_tax_clauses",
            {"section_uuid": "00000000-0000-0000-0000-000000000001"},
            scope="agreements:read",
        )
        self.assertEqual(section_res.status_code, 200)
        section_payload = section_res.get_json()["result"]["structuredContent"]
        self.assertEqual(section_payload["returned_count"], 2)

    def test_filter_and_catalog_tools(self):
        filter_res = self._call_tool(
            "list_filter_options",
            {"fields": ["targets", "target_counsels", "transaction_price_totals", "deal_types", "target_pes"]},
        )
        self.assertEqual(filter_res.status_code, 200)
        filter_payload = filter_res.get_json()["result"]["structuredContent"]
        self.assertEqual(filter_payload["targets"], ["Company B", "Target A"])
        self.assertEqual(
            filter_payload["target_counsels"],
            [
                "Skadden, Arps, Slate, Meagher & Flom LLP",
                "Wachtell, Lipton, Rosen & Katz",
            ],
        )
        self.assertIn("0 - 100M", filter_payload["transaction_price_totals"])
        self.assertEqual(filter_payload["deal_types"], ["merger"])
        self.assertEqual(filter_payload["target_pes"], ["true", "false"])
        self.assertEqual(filter_payload["retrieval_parameter_map"]["targets"], "target")
        self.assertEqual(filter_payload["retrieval_parameter_map"]["target_counsels"], "target_counsel")
        self.assertEqual(
            filter_payload["retrieval_parameter_map"]["transaction_price_totals"],
            "transaction_price_total",
        )
        self.assertEqual(filter_payload["retrieval_parameter_map"]["deal_types"], "deal_type")
        self.assertEqual(
            filter_payload["field_metadata"]["transaction_price_totals"]["value_kind"],
            "bucket",
        )
        self.assertEqual(
            filter_payload["field_metadata"]["target_pes"]["allowed_values"],
            ["true", "false"],
        )

        taxonomy_res = self._call_tool("get_clause_taxonomy", {})
        self.assertEqual(taxonomy_res.status_code, 200)
        taxonomy_payload = taxonomy_res.get_json()["result"]["structuredContent"]
        self.assertIn("Deal Protection", taxonomy_payload)

        tax_taxonomy_res = self._call_tool("get_tax_clause_taxonomy", {})
        self.assertEqual(tax_taxonomy_res.status_code, 200)
        tax_taxonomy_payload = tax_taxonomy_res.get_json()["result"]["structuredContent"]
        self.assertIn("Tax", tax_taxonomy_payload)

        counsel_res = self._call_tool("get_counsel_catalog", {})
        self.assertEqual(counsel_res.status_code, 200)
        counsel_payload = counsel_res.get_json()["result"]["structuredContent"]
        self.assertEqual(counsel_payload["counsel"][0]["canonical_name"], "Skadden, Arps, Slate, Meagher & Flom LLP")

        naics_res = self._call_tool("get_naics_catalog", {})
        self.assertEqual(naics_res.status_code, 200)
        naics_payload = naics_res.get_json()["result"]["structuredContent"]
        self.assertEqual(naics_payload["sectors"][0]["sector_code"], "11")

    def test_concept_mapping_and_snippet_tools(self):
        suggest_res = self._call_tool("suggest_clause_families", {"concept": "MAE carveouts", "top_k": 3})
        self.assertEqual(suggest_res.status_code, 200)
        suggest_payload = suggest_res.get_json()["result"]["structuredContent"]
        self.assertEqual(suggest_payload["matches"][0]["standard_id"], "2.1")
        self.assertIn("Material Adverse Effect", suggest_payload["matches"][0]["path"])
        self.assertIn(suggest_payload["matches"][0]["fit"], ["canonical", "proxy", "broad_match"])
        self.assertIn(suggest_payload["matches"][0]["confidence"], ["high", "medium", "low"])
        self.assertIsInstance(suggest_payload["matches"][0]["scope_note"], str)

        snippet_res = self._call_tool(
            "get_section_snippet",
            {
                "section_uuid": "00000000-0000-0000-0000-000000000003",
                "focus_terms": ["disproportionate effects"],
                "max_chars": 220,
            },
        )
        self.assertEqual(snippet_res.status_code, 200)
        snippet_payload = snippet_res.get_json()["result"]["structuredContent"]
        self.assertIn("disproportionate effects", snippet_payload["snippet"].lower())
        self.assertEqual(snippet_payload["matched_terms"], ["disproportionate effects"])

    def test_server_capabilities_tool(self):
        res = self._call_tool("get_server_capabilities", {})
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()["result"]["structuredContent"]
        self.assertEqual(payload["server"]["introspection_tool"], "get_server_capabilities")
        self.assertEqual(payload["server"]["metrics_tool"], "get_server_metrics")
        self.assertEqual(payload["server"]["transport"], "http_jsonrpc")
        self.assertFalse(payload["server"]["resources_supported"])
        self.assertFalse(payload["server"]["resource_templates_supported"])
        self.assertTrue(payload["auth_help"]["login_required"])
        self.assertEqual(payload["auth_help"]["fulltext_scope"], "agreements:read_fulltext")
        self.assertTrue(payload["field_inventory"]["agreement_fields"])
        self.assertTrue(payload["concept_notes"])
        self.assertTrue(payload["tool_limitations"])
        search_agreements_tool = next(tool for tool in payload["tools"] if tool["name"] == "search_agreements")
        self.assertEqual(search_agreements_tool["pagination"], "page")
        self.assertEqual(search_agreements_tool["limits"]["default_page_size"], 25)
        self.assertEqual(search_agreements_tool["limits"]["max_page_size"], 100)
        self.assertEqual(search_agreements_tool["access"]["scope_behavior"], "strict_scope_required")
        self.assertTrue(search_agreements_tool["response_examples"])
        self.assertIn("agreements:search", search_agreements_tool["required_scopes"])
        self.assertTrue(search_agreements_tool["examples"])
        self.assertTrue(search_agreements_tool["negative_guidance"])
        get_agreement_tool = next(tool for tool in payload["tools"] if tool["name"] == "get_agreement")
        self.assertEqual(get_agreement_tool["access"]["redaction"], "redacted_without_fulltext_scope")
        self.assertEqual(get_agreement_tool["access"]["fulltext_scope"], "agreements:read_fulltext")
        workflow_names = [workflow["name"] for workflow in payload["workflows"]]
        self.assertIn("discover agreements by counsel", workflow_names)
        self.assertIn("map a plain-English concept to clause samples", workflow_names)

    def test_server_metrics_tool(self):
        self._call_tool("search_agreements", {"query": "Target"})
        self._call_tool("list_filter_options", {"fields": ["targets"]})

        res = self._call_tool("get_server_metrics", {})
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()["result"]["structuredContent"]
        self.assertEqual(payload["latency_bucket_bounds_ms"], [50, 100, 250, 500, 1000, 2500, 5000])
        self.assertIn("search_agreements", payload["tool_calls"])
        self.assertGreaterEqual(payload["tool_calls"]["search_agreements"]["calls"], 1)
        self.assertIn("get_server_metrics", payload["tool_calls"])

    def test_json_compatible_value_normalizes_dates_and_decimals(self):
        import backend.mcp.tools as tools_module

        self.assertEqual(tools_module._json_compatible_value(date(2020, 1, 1)), "2020-01-01")
        self.assertEqual(tools_module._json_compatible_value(Decimal("12.50")), 12.5)

    def test_client_harness_compatibility_flow(self):
        client = McpClientHarness(self)
        init_payload = cast(dict[str, object], client.initialize()["result"])
        self.assertEqual(cast(dict[str, object], init_payload["serverInfo"])["name"], "pandects-mcp")

        tools = client.list_tools()
        self.assertIn("search_agreements", [tool["name"] for tool in tools])

        agreements_payload = client.call_tool("search_agreements", {"query": "Target"})
        agreement_results = cast(list[dict[str, object]], agreements_payload["results"])
        self.assertEqual(agreement_results[0]["agreement_uuid"], "a1")

        metrics_payload = client.call_tool("get_server_metrics", {})
        self.assertIn("search_agreements", cast(dict[str, object], metrics_payload["tool_calls"]))

    def test_live_http_transport_client_compatibility_flow(self):
        with LiveMcpHttpClientHarness(self) as client:
            init_payload = cast(dict[str, object], client.initialize()["result"])
            self.assertEqual(cast(dict[str, object], init_payload["serverInfo"])["name"], "pandects-mcp")

            tools = client.list_tools()
            self.assertIn("get_server_capabilities", [tool["name"] for tool in tools])

            capabilities_payload = client.call_tool("get_server_capabilities", {})
            self.assertEqual(cast(dict[str, object], capabilities_payload["server"])["transport"], "http_jsonrpc")

            agreements_payload = client.call_tool("search_agreements", {"query": "Target"})
            agreement_results = cast(list[dict[str, object]], agreements_payload["results"])
            self.assertEqual(agreement_results[0]["agreement_uuid"], "a1")

    def test_live_http_transport_filter_to_retrieval_workflow(self):
        with LiveMcpHttpClientHarness(self) as client:
            client.initialize()
            client.list_tools()

            filter_payload = client.call_tool("list_filter_options", {"fields": ["target_counsels"]})
            catalog_names = cast(list[str], filter_payload["target_counsels"])
            catalog_name = "Wachtell, Lipton, Rosen & Katz"
            self.assertIn(catalog_name, catalog_names)
            retrieval_param = cast(dict[str, str], filter_payload["retrieval_parameter_map"])["target_counsels"]

            agreements_payload = client.call_tool("search_agreements", {retrieval_param: [catalog_name]})
            agreement_results = cast(list[dict[str, object]], agreements_payload["results"])
            self.assertEqual(agreement_results[0]["agreement_uuid"], "a1")

    def test_live_http_transport_concept_to_snippet_workflow(self):
        with LiveMcpHttpClientHarness(self) as client:
            client.initialize()
            client.list_tools()

            suggestions = client.call_tool("suggest_clause_families", {"concept": "MAE carveouts", "top_k": 2})
            self.assertEqual(cast(list[dict[str, object]], suggestions["matches"])[0]["standard_id"], "2.1")
            snippet = client.call_tool(
                "get_section_snippet",
                {
                    "section_uuid": "00000000-0000-0000-0000-000000000003",
                    "focus_terms": ["disproportionate effects"],
                    "max_chars": 220,
                },
            )
            self.assertEqual(cast(str, snippet["section_uuid"]), "00000000-0000-0000-0000-000000000003")

    def test_live_http_transport_redaction_and_fulltext_workflow(self):
        with LiveMcpHttpClientHarness(self) as client:
            client.initialize()
            client.list_tools()

            redacted_payload = client.call_tool("get_agreement", {"agreement_uuid": "a1"})
            self.assertTrue(cast(bool, redacted_payload["is_redacted"]))

        with LiveMcpHttpClientHarness(
            self,
            scope="sections:search agreements:search agreements:read agreements:read_fulltext",
        ) as fulltext_client:
            fulltext_client.initialize()
            fulltext_client.list_tools()
            fulltext_payload = fulltext_client.call_tool("get_agreement", {"agreement_uuid": "a1"})
            self.assertFalse(cast(bool, fulltext_payload["is_redacted"]))

    def test_tools_list_contract_snapshot(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer()},
            json={"jsonrpc": "2.0", "id": 401, "method": "tools/list"},
        )
        self.assertEqual(res.status_code, 200)
        tools = _normalized_tools_snapshot(res.get_json()["result"]["tools"])
        snapshot_path = os.path.join(os.path.dirname(__file__), "fixtures", "mcp_tools_list_snapshot.json")
        with open(snapshot_path, "r", encoding="utf-8") as snapshot_file:
            expected = json.load(snapshot_file)
        self.assertEqual(tools, expected)

    def test_server_capabilities_snapshot(self):
        res = self._call_tool("get_server_capabilities", {})
        self.assertEqual(res.status_code, 200)
        payload = _normalized_capabilities_snapshot(res.get_json()["result"]["structuredContent"])
        snapshot_path = os.path.join(os.path.dirname(__file__), "fixtures", "mcp_server_capabilities_snapshot.json")
        with open(snapshot_path, "r", encoding="utf-8") as snapshot_file:
            expected = json.load(snapshot_file)
        self.assertEqual(payload, expected)

    def test_summary_and_trends_tools(self):
        summary_res = self._call_tool("get_agreements_summary", {})
        self.assertEqual(summary_res.status_code, 200)
        summary_payload = summary_res.get_json()["result"]["structuredContent"]
        self.assertEqual(summary_payload["agreements"], 1)
        self.assertEqual(summary_payload["sections"], 2)

        trends_res = self._call_tool("get_agreement_trends", {})
        self.assertEqual(trends_res.status_code, 200)
        trends_payload = trends_res.get_json()["result"]["structuredContent"]
        self.assertEqual(trends_payload["ownership"]["mix_by_year"][0]["public_deal_count"], 1)
        self.assertEqual(trends_payload["industries"]["target_industries_by_year"][0]["industry"], "Crop Production")

    def test_list_filter_options_rejects_unknown_fields(self):
        res = self._call_tool("list_filter_options", {"fields": ["nope"]})
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()
        self.assertEqual(payload["error"]["code"], -32602)

    def test_chain_workflow_from_agreement_search_to_section(self):
        search_res = self._call_tool("search_agreements", {"query": "Target"})
        search_payload = search_res.get_json()["result"]["structuredContent"]
        agreement_uuid = search_payload["results"][0]["agreement_uuid"]

        list_res = self._call_tool("list_agreement_sections", {"agreement_uuid": agreement_uuid})
        list_payload = list_res.get_json()["result"]["structuredContent"]
        section_uuid = list_payload["results"][0]["section_uuid"]

        section_res = self._call_tool("get_section", {"section_uuid": section_uuid})
        self.assertEqual(section_res.status_code, 200)
        section_payload = section_res.get_json()["result"]["structuredContent"]
        self.assertEqual(section_payload["agreement_uuid"], agreement_uuid)

    def test_workflow_maps_filter_option_catalog_to_agreement_search_parameter(self):
        filter_res = self._call_tool("list_filter_options", {"fields": ["target_counsels"]})
        self.assertEqual(filter_res.status_code, 200)
        filter_payload = filter_res.get_json()["result"]["structuredContent"]

        catalog_name = "Wachtell, Lipton, Rosen & Katz"
        self.assertIn(catalog_name, filter_payload["target_counsels"])
        retrieval_param = filter_payload["retrieval_parameter_map"]["target_counsels"]
        self.assertEqual(retrieval_param, "target_counsel")

        search_res = self._call_tool("search_agreements", {retrieval_param: [catalog_name]})
        self.assertEqual(search_res.status_code, 200)
        search_payload = search_res.get_json()["result"]["structuredContent"]
        self.assertEqual(search_payload["returned_count"], 1)
        self.assertEqual(search_payload["results"][0]["agreement_uuid"], "a1")

    def test_workflow_uses_counsel_catalog_for_section_search(self):
        counsel_res = self._call_tool("get_counsel_catalog", {})
        self.assertEqual(counsel_res.status_code, 200)
        counsel_payload = counsel_res.get_json()["result"]["structuredContent"]
        canonical_names = [row["canonical_name"] for row in counsel_payload["counsel"]]
        self.assertIn("Wachtell, Lipton, Rosen & Katz", canonical_names)

        search_res = self._call_tool(
            "search_sections",
            {"target_counsel": ["Wachtell, Lipton, Rosen & Katz"], "page_size": 10},
        )
        self.assertEqual(search_res.status_code, 200)
        search_payload = search_res.get_json()["result"]["structuredContent"]
        self.assertEqual(search_payload["results"][0]["agreement_uuid"], "a1")

    def test_retrieval_tools_reject_pluralized_catalog_keys(self):
        search_res = self._call_tool("search_sections", {"target_counsels": ["Wachtell, Lipton, Rosen & Katz"]})
        self.assertEqual(search_res.status_code, 200)
        search_payload = search_res.get_json()
        self.assertEqual(search_payload["error"]["code"], -32602)

        agreement_res = self._call_tool("search_agreements", {"target_counsels": ["Wachtell, Lipton, Rosen & Katz"]})
        self.assertEqual(agreement_res.status_code, 200)
        agreement_payload = agreement_res.get_json()
        self.assertEqual(agreement_payload["error"]["code"], -32602)

    def test_chain_workflow_from_section_search_to_agreement(self):
        search_res = self._call_tool("search_sections", {"page_size": 10})
        search_payload = search_res.get_json()["result"]["structuredContent"]
        section_uuid = search_payload["results"][0]["section_uuid"]
        agreement_uuid = search_payload["results"][0]["agreement_uuid"]

        section_res = self._call_tool("get_section", {"section_uuid": section_uuid})
        self.assertEqual(section_res.status_code, 200)

        agreement_res = self._call_tool(
            "get_agreement",
            {"agreement_uuid": agreement_uuid},
            scope="agreements:read",
        )
        self.assertEqual(agreement_res.status_code, 200)

    def test_chain_workflow_from_concept_to_snippet(self):
        suggest_res = self._call_tool("suggest_clause_families", {"concept": "MAE carveouts"})
        suggest_payload = suggest_res.get_json()["result"]["structuredContent"]
        standard_id = suggest_payload["matches"][0]["standard_id"]

        search_res = self._call_tool("search_sections", {"standard_id": [standard_id], "page_size": 10})
        search_payload = search_res.get_json()["result"]["structuredContent"]
        section_uuid = search_payload["results"][0]["section_uuid"]

        snippet_res = self._call_tool(
            "get_section_snippet",
            {"section_uuid": section_uuid, "focus_terms": ["disproportionate effect"], "max_chars": 220},
        )
        self.assertEqual(snippet_res.status_code, 200)
        snippet_payload = snippet_res.get_json()["result"]["structuredContent"]
        self.assertEqual(snippet_payload["section_uuid"], section_uuid)

    def test_search_sections_tool(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer(scope="sections:search")},
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "search_sections", "arguments": {"page_size": 10}},
            },
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        structured = body["result"]["structuredContent"]
        self.assertEqual(len(structured["results"]), 4)
        self.assertEqual(structured["access"]["tier"], "mcp")
        self.assertEqual(structured["count_metadata"]["mode"], "exact")
        self.assertEqual(structured["count_metadata"]["method"], "query_count")
        self.assertEqual(structured["count_metadata"]["planning_reliability"], "high")
        self.assertEqual(structured["interpretation"]["taxonomy_filters"], [])

    def test_search_sections_exact_count_mode_returns_exact_metadata(self):
        res = self._call_tool(
            "search_sections",
            {"standard_id": ["2.1"], "count_mode": "exact", "page_size": 10},
            scope="sections:search",
        )
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()["result"]["structuredContent"]
        self.assertEqual(payload["total_count"], 2)
        self.assertFalse(payload["total_count_is_approximate"])
        self.assertEqual(payload["count_metadata"]["mode"], "exact")
        self.assertEqual(payload["count_metadata"]["method"], "query_count")
        self.assertEqual(payload["count_metadata"]["planning_reliability"], "high")
        self.assertTrue(payload["count_metadata"]["exact_count_requested"])
        self.assertEqual(
            payload["interpretation"]["taxonomy_filters"],
            [{"standard_id": "2.1", "match_mode": "expanded_descendants"}],
        )

    def test_search_sections_default_count_mode_and_interpretation(self):
        res = self._call_tool(
            "search_sections",
            {"target_counsel": ["Wachtell, Lipton, Rosen & Katz"], "page_size": 10},
            scope="sections:search",
        )
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()["result"]["structuredContent"]
        self.assertEqual(payload["count_metadata"]["exact_count_requested"], False)
        self.assertIn(
            {
                "field": "target_counsel",
                "representation": "first_class_agreement_field",
                "match_kind": "exact_metadata_filter",
            },
            payload["interpretation"]["applied_filters"],
        )
        self.assertEqual(payload["interpretation"]["heuristics_used"], [])

    def test_search_sections_auto_count_can_return_estimate_for_paginated_filtered_search(self):
        res = self._call_tool(
            "search_sections",
            {"target_counsel": ["Wachtell, Lipton, Rosen & Katz"], "page": 2, "page_size": 1},
            scope="sections:search",
        )
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()["result"]["structuredContent"]
        self.assertTrue(payload["total_count_is_approximate"])
        self.assertEqual(payload["count_metadata"]["mode"], "estimated")
        self.assertEqual(payload["count_metadata"]["method"], "filtered_lower_bound")
        self.assertEqual(payload["count_metadata"]["planning_reliability"], "low")

    def test_get_agreement_redacts_without_fulltext_scope(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer(scope="agreements:read")},
            json={
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "get_agreement",
                    "arguments": {
                        "agreement_uuid": "a1",
                        "focus_section_uuid": "00000000-0000-0000-0000-000000000001",
                        "neighbor_sections": 0,
                    },
                },
            },
        )
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()["result"]["structuredContent"]
        self.assertTrue(payload["is_redacted"])
        self.assertIn("[REDACTED]", payload["xml"])

    def test_get_agreement_fulltext_requires_scope(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer(scope="agreements:read agreements:read_fulltext")},
            json={
                "jsonrpc": "2.0",
                "id": 5,
                "method": "tools/call",
                "params": {
                    "name": "get_agreement",
                    "arguments": {"agreement_uuid": "a1"},
                },
            },
        )
        self.assertEqual(res.status_code, 200)
        payload = res.get_json()["result"]["structuredContent"]
        self.assertFalse(payload["is_redacted"])
        self.assertIn("<text>KEEP</text>", payload["xml"])

    def test_missing_scope_is_403(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer(scope="sections:search")},
            json={
                "jsonrpc": "2.0",
                "id": 6,
                "method": "tools/call",
                "params": {
                    "name": "get_agreement",
                    "arguments": {"agreement_uuid": "a1"},
                },
            },
        )
        self.assertEqual(res.status_code, 403)
        payload = res.get_json()
        self.assertEqual(payload["jsonrpc"], "2.0")
        self.assertEqual(payload["id"], 6)
        self.assertEqual(payload["error"]["data"]["category"], "authorization")
        self.assertEqual(payload["error"]["message"], "You do not have permission to call this tool.")

    def test_zitadel_provider_falls_back_to_introspection_for_opaque_tokens(self):
        original_provider = os.environ.get("MCP_IDENTITY_PROVIDER")
        original_decode = self.mcp_runtime._decode_access_token
        original_introspect = self.mcp_runtime._introspect_access_token
        self.mcp_runtime._mcp_identity_provider = None
        os.environ["MCP_IDENTITY_PROVIDER"] = "zitadel"
        try:
            def _fail_decode(_token: str):
                raise RuntimeError("not a jwt")

            def _fake_introspect(token: str):
                self.assertEqual(token, "opaque-zitadel-token")
                return {
                    "active": True,
                    "iss": "https://issuer.example.com",
                    "sub": "sub-123",
                    "aud": ["pandects-mcp"],
                    "scope": "agreements:read",
                }

            self.mcp_runtime._decode_access_token = _fail_decode
            self.mcp_runtime._introspect_access_token = _fake_introspect
            identity = self.mcp_runtime.authenticate_external_identity(
                access_token="opaque-zitadel-token",
                provider_name="zitadel",
            )
            self.assertEqual(identity.issuer, "https://issuer.example.com")
            self.assertEqual(identity.subject, "sub-123")
            self.assertEqual(identity.scopes, frozenset({"agreements:read"}))
        finally:
            self.mcp_runtime._decode_access_token = original_decode
            self.mcp_runtime._introspect_access_token = original_introspect
            self.mcp_runtime._mcp_identity_provider = None
            if original_provider is None:
                os.environ.pop("MCP_IDENTITY_PROVIDER", None)
            else:
                os.environ["MCP_IDENTITY_PROVIDER"] = original_provider

    def test_unlinked_subject_is_401(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={"Authorization": self._bearer_for_subject(subject="sub-missing")},
            json={"jsonrpc": "2.0", "id": 7, "method": "initialize"},
        )
        self.assertEqual(res.status_code, 401)
        self.assertIn("WWW-Authenticate", res.headers)
        payload = res.get_json()
        self.assertEqual(payload["jsonrpc"], "2.0")
        self.assertEqual(payload["id"], 7)
        self.assertEqual(payload["error"]["data"]["category"], "authentication")
        self.assertEqual(payload["error"]["data"]["reason"], "unlinked_subject")
        self.assertEqual(payload["error"]["data"]["action"], "contact_support")

    def test_invalid_token_error_includes_relogin_guidance(self):
        client = self.app.test_client()
        res = client.post(
            "/mcp",
            headers={"Authorization": "Bearer definitely-not-a-valid-token"},
            json={"jsonrpc": "2.0", "id": 8, "method": "initialize"},
        )
        self.assertEqual(res.status_code, 401)
        payload = res.get_json()
        self.assertEqual(payload["error"]["data"]["category"], "authentication")
        self.assertEqual(payload["error"]["data"]["reason"], "expired_token")
        self.assertEqual(payload["error"]["data"]["action"], "relogin")
        self.assertIn("Sign in again", payload["error"]["data"]["client_message"])

    def test_normalized_external_identity_supports_scope_and_scp_claims(self):
        normalized_from_scope = self.mcp_runtime._normalize_external_identity(
            {
                "iss": "https://issuer.example.com/",
                "sub": "sub-123",
                "aud": "pandects-mcp",
                "scope": "agreements:read agreements:read_fulltext",
            }
        )
        self.assertEqual(normalized_from_scope.issuer, "https://issuer.example.com")
        self.assertEqual(
            normalized_from_scope.scopes,
            frozenset({"agreements:read", "agreements:read_fulltext"}),
        )
        self.assertEqual(normalized_from_scope.audiences, frozenset({"pandects-mcp"}))

        normalized_from_scp = self.mcp_runtime._normalize_external_identity(
            {
                "iss": "https://issuer.example.com",
                "sub": "sub-123",
                "aud": ["pandects-mcp", "other-audience"],
                "scp": ["sections:search", "agreements:search"],
            }
        )
        self.assertEqual(
            normalized_from_scp.scopes,
            frozenset({"sections:search", "agreements:search"}),
        )
        self.assertEqual(
            normalized_from_scp.audiences,
            frozenset({"pandects-mcp", "other-audience"}),
        )

    def test_normalized_external_identity_maps_zitadel_role_claims_to_mcp_scopes(self):
        normalized_from_roles = self.mcp_runtime._normalize_external_identity(
            {
                "iss": "https://issuer.example.com",
                "sub": "sub-123",
                "aud": "pandects-mcp",
                "urn:zitadel:iam:org:project:365876335077286382:roles": {
                    "sections_search": {"365876094760509934": "pandects"},
                    "agreements_read": {"365876094760509934": "pandects"},
                },
            }
        )
        self.assertEqual(
            normalized_from_roles.scopes,
            frozenset({"sections:search", "agreements:read"}),
        )

    def test_identity_provider_name_defaults_to_oidc(self):
        previous = os.environ.pop("MCP_IDENTITY_PROVIDER", None)
        try:
            self.assertEqual(self.mcp_runtime.mcp_identity_provider_name(), "oidc")
        finally:
            if previous is not None:
                os.environ["MCP_IDENTITY_PROVIDER"] = previous

    def test_identity_provider_name_is_normalized(self):
        previous = os.environ.get("MCP_IDENTITY_PROVIDER")
        os.environ["MCP_IDENTITY_PROVIDER"] = " ZITADEL "
        try:
            self.assertEqual(self.mcp_runtime.mcp_identity_provider_name(), "zitadel")
        finally:
            if previous is None:
                os.environ.pop("MCP_IDENTITY_PROVIDER", None)
            else:
                os.environ["MCP_IDENTITY_PROVIDER"] = previous

    def test_identity_provider_can_be_selected_by_env(self):
        runtime = self.mcp_runtime
        previous = os.environ.get("MCP_IDENTITY_PROVIDER")

        class StubProvider(runtime.McpIdentityProvider):
            seen_token: str | None = None

            def authenticate_access_token(self, token: str) -> "ExternalIdentity":
                self.seen_token = token
                return runtime.ExternalIdentity(
                    issuer="https://stub-issuer.example.com",
                    subject="stub-subject",
                    scopes=frozenset({"agreements:read"}),
                    audiences=frozenset({"pandects-mcp"}),
                    claims={"sub": "stub-subject"},
                )

        runtime.register_mcp_identity_provider("stub", StubProvider)
        runtime._mcp_identity_provider = None
        os.environ["MCP_IDENTITY_PROVIDER"] = "stub"
        try:
            provider = runtime._identity_provider()
            self.assertIsInstance(provider, StubProvider)
        finally:
            runtime._mcp_identity_provider = None
            if previous is None:
                os.environ.pop("MCP_IDENTITY_PROVIDER", None)
            else:
                os.environ["MCP_IDENTITY_PROVIDER"] = previous

    def test_zitadel_identity_provider_can_be_selected_by_env(self):
        runtime = self.mcp_runtime
        previous = os.environ.get("MCP_IDENTITY_PROVIDER")
        runtime._mcp_identity_provider = None
        os.environ["MCP_IDENTITY_PROVIDER"] = "zitadel"
        try:
            provider = runtime._identity_provider()
            self.assertIsInstance(provider, runtime.ZitadelMcpIdentityProvider)
        finally:
            runtime._mcp_identity_provider = None
            if previous is None:
                os.environ.pop("MCP_IDENTITY_PROVIDER", None)
            else:
                os.environ["MCP_IDENTITY_PROVIDER"] = previous

    def test_unknown_identity_provider_raises_runtime_error(self):
        runtime = self.mcp_runtime
        previous = os.environ.get("MCP_IDENTITY_PROVIDER")
        runtime._mcp_identity_provider = None
        os.environ["MCP_IDENTITY_PROVIDER"] = "missing-provider"
        try:
            with self.assertRaises(RuntimeError):
                runtime._identity_provider()
        finally:
            runtime._mcp_identity_provider = None
            if previous is None:
                os.environ.pop("MCP_IDENTITY_PROVIDER", None)
            else:
                os.environ["MCP_IDENTITY_PROVIDER"] = previous


if __name__ == "__main__":
    unittest.main()
