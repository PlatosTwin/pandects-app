from __future__ import annotations

import re
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PUBLIC_TABLES_PATH = _REPO_ROOT / "bulk" / "public_tables.txt"
_PRIVATE_RESTORE_TABLES_PATH = _REPO_ROOT / "bulk" / "private_restore_tables.txt"
_EXPECTED_PRIVATE_API_TABLES = {"section_text_search"}
_MAIN_DB_MODELS_PATH = _REPO_ROOT / "backend" / "models" / "main_db.py"
_PUBLIC_API_SOURCE_PATHS = (
    _REPO_ROOT / "backend" / "routes" / "agreements" / "__init__.py",
    _REPO_ROOT / "backend" / "routes" / "agreements" / "helpers.py",
    _REPO_ROOT / "backend" / "routes" / "reference_data.py",
    _REPO_ROOT / "backend" / "services" / "sections_service.py",
)
_ORM_DEP_TO_TABLE = {
    "AgreementCounsel": "agreement_counsel",
    "Agreements": "agreements",
    "Clauses": "clauses",
    "Counsel": "counsel",
    "LatestSectionsSearch": "latest_sections_search",
    "NaicsSector": "naics_sectors",
    "NaicsSubSector": "naics_sub_sectors",
    "Sections": "sections",
    "SectionTextSearch": "section_text_search",
    "TaxClauseAssignment": "tax_clause_assignments",
    "TaxClauseTaxonomyL1": "tax_clause_taxonomy_l1",
    "TaxClauseTaxonomyL2": "tax_clause_taxonomy_l2",
    "TaxClauseTaxonomyL3": "tax_clause_taxonomy_l3",
    "TaxonomyL1": "taxonomy_l1",
    "TaxonomyL2": "taxonomy_l2",
    "TaxonomyL3": "taxonomy_l3",
    "XML": "xml",
}
_ORM_DEP_PATTERN = re.compile(
    r"deps\.(" + "|".join(sorted(_ORM_DEP_TO_TABLE.keys(), key=len, reverse=True)) + r")\b"
)
_SQL_TABLE_PATTERNS = (
    re.compile(r"\{deps\._schema_prefix\(\)\}([a-z_]+)"),
    re.compile(r"\{schema_prefix\}([a-z_]+)"),
    re.compile(r"TABLE_NAME = '([a-z_]+)'"),
)


def _parse_table_file(path: Path) -> list[str]:
    tables: list[str] = []
    for raw_line in path.read_text().splitlines():
        entry = raw_line.split("#", 1)[0].strip()
        if entry:
            tables.append(entry)
    return tables


def _derive_expected_api_tables() -> set[str]:
    tables: set[str] = set()
    sources = {path: path.read_text() for path in _PUBLIC_API_SOURCE_PATHS}
    for path, source in sources.items():
        for dep_name in _ORM_DEP_PATTERN.findall(source):
            tables.add(_ORM_DEP_TO_TABLE[dep_name])
        for pattern in _SQL_TABLE_PATTERNS:
            tables.update(pattern.findall(source))
    sections_service_source = sources[_REPO_ROOT / "backend" / "services" / "sections_service.py"]
    if "standard_id_filter_expr(" in sections_service_source:
        model_source = _MAIN_DB_MODELS_PATH.read_text()
        if (
            "def standard_id_filter_expr" in model_source
            and "LatestSectionsSearchStandardId" in model_source
        ):
            tables.add("latest_sections_search_standard_ids")
    return tables


class R2AllowlistTests(unittest.TestCase):
    def test_section_text_search_canonical_ddl_is_lean_and_indexed(self) -> None:
        ddl = (_REPO_ROOT / "db" / "section_text_search.sql").read_text()
        self.assertIn("source_xml_sha256 BINARY(32) NOT NULL", ddl)
        self.assertIn(
            "FULLTEXT KEY ft_section_text_search_normalized_text (normalized_text)",
            ddl,
        )
        self.assertNotIn("plain_text", ddl)

    def test_restore_allowlist_matches_api_route_table_set(self) -> None:
        public_tables = _parse_table_file(_PUBLIC_TABLES_PATH)
        private_restore_tables = _parse_table_file(_PRIVATE_RESTORE_TABLES_PATH)
        self.assertTrue(public_tables)
        expected_api_tables = _derive_expected_api_tables()
        self.assertEqual(set(private_restore_tables), _EXPECTED_PRIVATE_API_TABLES)
        self.assertEqual(
            set(public_tables),
            expected_api_tables - _EXPECTED_PRIVATE_API_TABLES,
        )

    def test_private_serving_tables_are_not_in_public_dump(self) -> None:
        public_tables = set(_parse_table_file(_PUBLIC_TABLES_PATH))
        private_restore_tables = set(_parse_table_file(_PRIVATE_RESTORE_TABLES_PATH))
        self.assertTrue(private_restore_tables)
        self.assertTrue(public_tables.isdisjoint(private_restore_tables))

    def test_push_to_r2_allowlists_have_no_duplicates(self) -> None:
        for path in (_PUBLIC_TABLES_PATH, _PRIVATE_RESTORE_TABLES_PATH):
            tables = _parse_table_file(path)
            self.assertEqual(len(tables), len(set(tables)), path)


if __name__ == "__main__":
    unittest.main()
