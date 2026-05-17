from __future__ import annotations

import re
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PUSH_TO_R2_PATH = _REPO_ROOT / "bulk" / "push_to_r2.sh"
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


def _parse_api_tables() -> list[str]:
    script = _PUSH_TO_R2_PATH.read_text()
    match = re.search(r"API_TABLES=\((.*?)\)", script, re.DOTALL)
    if match is None:
        raise AssertionError("push_to_r2.sh is missing API_TABLES.")
    return [
        line.strip()
        for line in match.group(1).splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _derive_expected_public_api_tables() -> set[str]:
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
    def test_push_to_r2_allowlist_matches_public_api_route_table_set(self) -> None:
        api_tables = _parse_api_tables()
        self.assertEqual(set(api_tables), _derive_expected_public_api_tables())

    def test_push_to_r2_allowlist_has_no_duplicates(self) -> None:
        api_tables = _parse_api_tables()
        self.assertEqual(len(api_tables), len(set(api_tables)))


if __name__ == "__main__":
    unittest.main()
