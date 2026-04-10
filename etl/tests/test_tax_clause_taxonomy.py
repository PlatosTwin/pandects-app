# pyright: reportAny=false
import unittest

from etl.models.taxonomy.tax_clause_taxonomy import (
    TAX_CLAUSE_TAXONOMY,
    iter_tax_clause_taxonomy_rows,
    render_tax_clause_taxonomy_seed_sql,
)


class TaxClauseTaxonomyTests(unittest.TestCase):
    def test_taxonomy_shape_and_ids_are_stable(self) -> None:
        l1_rows, l2_rows, l3_rows = iter_tax_clause_taxonomy_rows()

        self.assertEqual(len(TAX_CLAUSE_TAXONOMY), 7)
        self.assertEqual(len(l1_rows), 7)
        self.assertEqual(len(l2_rows), 26)
        self.assertEqual(len(l3_rows), 85)
        self.assertEqual(l1_rows[0], ("tax.1", "Transaction Tax Status and Intended Treatment"))
        self.assertIn(
            ("tax.5.3.1", "Tax Indemnity / Reimbursement Obligation", "tax.5.3"),
            l3_rows,
        )
        self.assertIn(
            ("tax.7.4.2", "FIRPTA Status", "tax.7.4"),
            l3_rows,
        )

        all_ids = [standard_id for standard_id, _label in l1_rows]
        all_ids.extend(standard_id for standard_id, _label, _parent_id in l2_rows)
        all_ids.extend(standard_id for standard_id, _label, _parent_id in l3_rows)
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_seed_sql_contains_expected_tables_and_labels(self) -> None:
        sql = render_tax_clause_taxonomy_seed_sql()

        self.assertIn("DELETE FROM pdx.tax_clause_taxonomy_l3;", sql)
        self.assertIn(
            "INSERT INTO pdx.tax_clause_taxonomy_l1 (standard_id, label)",
            sql,
        )
        self.assertIn("Transaction Tax Status and Intended Treatment", sql)
        self.assertIn("Gross-Up Obligation", sql)
        self.assertIn("Reorganization / Intended Tax Treatment Support", sql)


if __name__ == "__main__":
    _ = unittest.main()
