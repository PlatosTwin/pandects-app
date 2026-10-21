"""One taxonomy builder, cached labels, and the corpus-count endpoints."""

import os
import tempfile
import unittest

from sqlalchemy import text

from backend.tests.test_main_routes import _set_default_env


class TaxonomyServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        main_db = tempfile.NamedTemporaryFile(prefix="pandects_taxonomy_", suffix=".sqlite", delete=False)
        main_db.close()
        auth_db = tempfile.NamedTemporaryFile(prefix="pandects_taxonomy_auth_", suffix=".sqlite", delete=False)
        auth_db.close()
        cls._db_paths = (main_db.name, auth_db.name)
        _set_default_env(f"sqlite:///{main_db.name}", f"sqlite:///{auth_db.name}")

        import backend.app as app_module

        cls.app_module = app_module
        cls.app = app_module.create_test_app(
            config_overrides={
                "MAIN_DB_SCHEMA": "",
                "SQLALCHEMY_DATABASE_URI": f"sqlite:///{main_db.name}",
                "SQLALCHEMY_BINDS": {"auth": f"sqlite:///{auth_db.name}"},
            }
        )
        with cls.app.app_context():
            engine = app_module.db.engine
            app_module.metadata.create_all(engine)
            app_module.db.create_all(bind_key="auth")
            statements = [
                "INSERT INTO taxonomy_l1 (standard_id, label) VALUES ('t1', 'Termination'), ('d1', 'Deal Protection')",
                "INSERT INTO taxonomy_l2 (standard_id, label, parent_id) VALUES "
                "('t1.fees', 'Termination Fees', 't1'), ('d1.ns', 'No-Shop', 'd1')",
                "INSERT INTO taxonomy_l3 (standard_id, label, parent_id) VALUES "
                "('t1.fees.co', 'Company Termination Fee', 't1.fees'), "
                "('t1.fees.rev', 'Reverse Termination Fee', 't1.fees'), "
                "('d1.ns.fo', 'Fiduciary Out', 'd1.ns')",
                "INSERT INTO tax_clause_taxonomy_l1 (standard_id, label) VALUES ('tax', 'Tax')",
                "INSERT INTO tax_clause_taxonomy_l2 (standard_id, label, parent_id) VALUES ('tax.op', 'Operational', 'tax')",
                "INSERT INTO tax_clause_taxonomy_l3 (standard_id, label, parent_id) VALUES "
                "('tax.op.transfer', 'Transfer Taxes', 'tax.op'), ('tax.op.allocation', 'Allocation', 'tax.op')",
                "INSERT INTO naics_sectors (super_sector, sector_group, sector_desc, sector_code) VALUES "
                "('Goods-Producing Industries', 'Manufacturing', 'Manufacturing', 31), "
                "('Service-Providing Industries', 'Information', 'Information', 51)",
                "INSERT INTO naics_sub_sectors (sub_sector_desc, sub_sector_code, sector_code) VALUES "
                "('Food Manufacturing', 311, 31)",
                "INSERT INTO agreements (agreement_uuid, filing_date, target, acquirer, verified, gated, url) VALUES "
                "('a1', '2020-01-01', 'T1', 'A1', 1, 0, 'http://example.com/a1'), "
                "('a2', '2021-01-01', 'T2', 'A2', 0, NULL, 'http://example.com/a2'), "
                "('a3', '2022-01-01', 'T3', 'A3', 0, 1, 'http://example.com/a3')",
                "INSERT INTO xml (agreement_uuid, xml, version, status, latest) VALUES "
                "('a1', '<document/>', 1, 'verified', 1), ('a2', '<document/>', 1, NULL, 1), "
                "('a3', '<document/>', 1, NULL, 1)",
                # s1 carries two leaves under one L2: the L2/L1 counts must not double it.
                "INSERT INTO latest_sections_search_standard_ids (standard_id, section_uuid, agreement_uuid) VALUES "
                "('t1.fees.co', 's1', 'a1'), ('t1.fees.rev', 's1', 'a1'), "
                "('t1.fees.co', 's2', 'a2'), ('t1.fees', 's3', 'a2'), "
                "('d1.ns.fo', 's4', 'a2'), "
                # a3 is gated and unverified: search hides it, so counts must too.
                "('t1.fees.co', 's5', 'a3'), ('d1.ns.fo', 's5', 'a3')",
                "INSERT INTO clauses (clause_uuid, agreement_uuid, section_uuid, xml_version, module, clause_order, "
                "start_char, end_char, clause_text, source_method, context_type) VALUES "
                "('c1', 'a1', 's1', 1, 'tax', 1, 0, 5, 'x', 'm', 'operative'), "
                "('c2', 'a2', 's2', 1, 'tax', 1, 0, 5, 'x', 'm', 'operative'), "
                "('c3', 'a2', 's2', 1, 'tax', 2, 0, 5, 'x', 'm', 'rep_warranty'), "
                "('c4', 'a3', 's5', 1, 'tax', 1, 0, 5, 'x', 'm', 'operative'), "
                "('c5', 'a1', 's1', 1, 'other', 2, 0, 5, 'x', 'm', 'operative')",
                "CREATE TABLE IF NOT EXISTS agreement_filter_option_summary ("
                "field_name TEXT NOT NULL, option_value TEXT NOT NULL, agreement_count INTEGER NOT NULL)",
                "INSERT INTO agreement_filter_option_summary (field_name, option_value, agreement_count) VALUES "
                "('target_industries', '311', 1), ('target_industries', 'tech', 1), "
                "('acquirer_industries', '51', 1)",
                "INSERT INTO tax_clause_assignments (clause_uuid, standard_id, is_gold_label) VALUES "
                "('c1', 'tax.op.transfer', 1), ('c1', 'tax.op.allocation', 1), "
                "('c2', 'tax.op.transfer', 1), ('c3', 'tax.op.transfer', 1), "
                "('c4', 'tax.op.transfer', 1), ('c5', 'tax.op.transfer', 1)",
            ]
            with engine.begin() as conn:
                for statement in statements:
                    conn.execute(text(statement))

    @classmethod
    def tearDownClass(cls) -> None:
        for path in cls._db_paths:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass

    def setUp(self) -> None:
        from backend.services.taxonomy_service import clear_reference_caches

        clear_reference_caches()
        self.app_module._filter_options_cache["payload"] = None
        self.app_module._filter_options_cache["ts"] = 0

    def test_rest_mcp_and_filter_options_share_one_tree(self):
        from backend.mcp.tools.shared import _build_taxonomy_tree

        client = self.app.test_client()
        rest_tree = client.get("/v1/taxonomy").get_json()
        self.assertEqual(
            rest_tree,
            {
                "Deal Protection": {
                    "id": "d1",
                    "children": {
                        "No-Shop": {"id": "d1.ns", "children": {"Fiduciary Out": {"id": "d1.ns.fo"}}}
                    },
                },
                "Termination": {
                    "id": "t1",
                    "children": {
                        "Termination Fees": {
                            "id": "t1.fees",
                            "children": {
                                "Company Termination Fee": {"id": "t1.fees.co"},
                                "Reverse Termination Fee": {"id": "t1.fees.rev"},
                            },
                        }
                    },
                },
            },
        )
        filter_options = client.get("/v1/filter-options?fields=clause_types").get_json()
        self.assertEqual(filter_options["clause_types"], rest_tree)
        with self.app.app_context():
            deps = self.app_module._build_route_deps()[2]
            mcp_tree = _build_taxonomy_tree(
                l1_model=deps.TaxonomyL1, l2_model=deps.TaxonomyL2, l3_model=deps.TaxonomyL3, deps=deps
            )
        self.assertEqual(mcp_tree, rest_tree)

    def test_section_counts_roll_up_distinct_sections_and_skip_gated(self):
        res = self.app.test_client().get("/v1/taxonomy/counts")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(
            res.get_json()["counts"],
            {
                "t1.fees.co": 2,  # s1, s2 (s5 is gated)
                "t1.fees.rev": 1,
                "t1.fees": 3,  # s1 once, s2, and s3 tagged on the L2 itself
                "t1": 3,
                "d1.ns.fo": 1,
                "d1.ns": 1,
                "d1": 1,
            },
        )

    def test_tax_clause_counts_follow_default_search_gates(self):
        res = self.app.test_client().get("/v1/taxonomy/tax-clauses/counts")
        self.assertEqual(res.status_code, 200)
        # c3 is rep_warranty, c4 gated, c5 not the tax module.
        self.assertEqual(
            res.get_json()["counts"],
            {"tax.op.transfer": 2, "tax.op.allocation": 1, "tax.op": 2, "tax": 2},
        )

    def test_naics_industry_label_rolls_up_to_sector(self):
        from backend.services.reference_labels import naics_industry_label

        with self.app.app_context():
            self.assertEqual(naics_industry_label("311"), "Food Manufacturing")
            self.assertEqual(naics_industry_label("31"), "Manufacturing")
            self.assertEqual(naics_industry_label("3121"), "Manufacturing")
            self.assertEqual(naics_industry_label("519"), "Information")
            self.assertIsNone(naics_industry_label("99"))
            self.assertIsNone(naics_industry_label("tech"))
            self.assertIsNone(naics_industry_label(None))

    def test_response_schemas_add_labels(self):
        from backend.schemas.public_api import (
            AgreementResponseSchema,
            AgreementSearchResultSchema,
            TaxClauseSchema,
        )
        from backend.schemas.sections import SectionItemSchema
        from backend.schemas.tax_clauses import TaxClauseItemSchema

        with self.app.app_context():
            agreement = AgreementResponseSchema().dump(
                {"agreement_uuid": "a1", "target_industry": "311", "acquirer_industry": None}
            )
            self.assertEqual(agreement["target_industry_label"], "Food Manufacturing")
            self.assertIsNone(agreement["acquirer_industry_label"])

            search_row = AgreementSearchResultSchema().dump(
                {
                    "agreement_uuid": "a1",
                    "target_industry": "51",
                    "acquirer_industry": "tech",
                    "match_count": 1,
                    "matched_sections": [
                        {"section_uuid": "s1", "standard_id": ["t1.fees.co", "unknown"]}
                    ],
                }
            )
            self.assertEqual(search_row["target_industry_label"], "Information")
            self.assertIsNone(search_row["acquirer_industry_label"])
            self.assertEqual(
                search_row["matched_sections"][0]["standard_id_labels"],
                {"t1.fees.co": "Company Termination Fee"},
            )

            section = SectionItemSchema().dump(
                {
                    "section_uuid": "s1",
                    "standard_id": ["t1.fees.co", "t1.fees"],
                    "metadata": {"target_industry": "311"},
                }
            )
            self.assertEqual(
                section["standard_id_labels"],
                {"t1.fees.co": "Company Termination Fee", "t1.fees": "Termination Fees"},
            )
            # Only the requested metadata fields get a label.
            self.assertEqual(
                section["metadata"],
                {"target_industry": "311", "target_industry_label": "Food Manufacturing"},
            )

            tax_result = TaxClauseItemSchema().dump(
                {"clause_uuid": "c1", "tax_standard_ids": ["tax.op.transfer"]}
            )
            self.assertEqual(tax_result["tax_standard_id_labels"], {"tax.op.transfer": "Transfer Taxes"})
            agreement_tax_clause = TaxClauseSchema().dump(
                {"clause_uuid": "c1", "standard_ids": ["tax.op.allocation"]}
            )
            self.assertEqual(agreement_tax_clause["standard_id_labels"], {"tax.op.allocation": "Allocation"})

    def test_filter_options_label_requested_industries(self):
        res = self.app.test_client().get(
            "/v1/filter-options?fields=target_industries&fields=acquirer_industries"
        )
        self.assertEqual(res.status_code, 200)
        body = res.get_json()
        self.assertEqual(body["target_industries"], ["311", "tech"])
        self.assertEqual(
            body["industry_labels"], {"311": "Food Manufacturing", "51": "Information"}
        )


if __name__ == "__main__":
    unittest.main()
