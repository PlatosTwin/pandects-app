from __future__ import annotations

import re
import unittest
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]
_PUSH_TO_R2_PATH = _REPO_ROOT / "bulk" / "push_to_r2.sh"
_EXPECTED_PUBLIC_API_TABLES = {
    "agreement_deal_type_summary",
    "agreement_status_summary",
    "agreements",
    "latest_sections_search",
    "latest_sections_search_standard_ids",
    "naics_sectors",
    "naics_sub_sectors",
    "sections",
    "summary_data",
    "taxonomy_l1",
    "taxonomy_l2",
    "taxonomy_l3",
    "xml",
}


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


class R2AllowlistTests(unittest.TestCase):
    def test_push_to_r2_allowlist_matches_public_api_table_set(self) -> None:
        api_tables = _parse_api_tables()
        self.assertEqual(set(api_tables), _EXPECTED_PUBLIC_API_TABLES)

    def test_push_to_r2_allowlist_has_no_duplicates(self) -> None:
        api_tables = _parse_api_tables()
        self.assertEqual(len(api_tables), len(set(api_tables)))


if __name__ == "__main__":
    unittest.main()
