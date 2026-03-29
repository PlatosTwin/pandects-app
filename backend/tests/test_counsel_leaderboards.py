import unittest

from backend.counsel_leaderboards import (
    build_counsel_leaderboards,
    canonicalize_counsel_name,
    format_counsel_display_name,
    split_counsel_names,
)


class CounselLeaderboardHelpersTests(unittest.TestCase):
    def test_split_counsel_names_handles_semicolons_and_suffix_delimited_and(self) -> None:
        self.assertEqual(
            split_counsel_names("Wilson Sonsini Goodrich & Rosati, P.C.; Goodwin Procter LLP"),
            [
                "Wilson Sonsini Goodrich & Rosati, P.C.",
                "Goodwin Procter LLP",
            ],
        )
        self.assertEqual(
            split_counsel_names(
                "Simpson Thacher & Bartlett LLP and Wilson Sonsini Goodrich & Rosati"
            ),
            [
                "Simpson Thacher & Bartlett LLP",
                "Wilson Sonsini Goodrich & Rosati",
            ],
        )
        self.assertEqual(
            split_counsel_names("Wiggin and Dana LLP"),
            ["Wiggin and Dana LLP"],
        )

    def test_canonicalization_merges_suffix_variants(self) -> None:
        self.assertEqual(
            canonicalize_counsel_name("Wilson Sonsini Goodrich & Rosati Professional Corporation"),
            canonicalize_counsel_name("Wilson Sonsini Goodrich & Rosati, P.C."),
        )
        self.assertEqual(
            canonicalize_counsel_name("Wiggin & Dana, LLP"),
            canonicalize_counsel_name("Wiggin and Dana"),
        )
        self.assertEqual(
            format_counsel_display_name("Wiggin and Dana LLP"),
            "Wiggin & Dana",
        )

    def test_build_counsel_leaderboards_fully_attributes_multi_firm_entries(self) -> None:
        payload = build_counsel_leaderboards(
            [
                {
                    "filing_date": "2020-01-01",
                    "transaction_price_total": "50000000",
                    "target_counsel": "Wilson Sonsini Goodrich & Rosati, P.C.; Goodwin Procter LLP",
                    "acquirer_counsel": "Wiggin and Dana LLP",
                },
                {
                    "filing_date": "2021-01-01",
                    "transaction_price_total": "150000000",
                    "target_counsel": "Wilson Sonsini Goodrich & Rosati Professional Corporation",
                    "acquirer_counsel": "Wiggin & Dana, LLP",
                },
                {
                    "filing_date": "2022-01-01",
                    "transaction_price_total": "300000000",
                    "target_counsel": "Wachtell, Lipton, Rosen & Katz",
                    "acquirer_counsel": "Skadden, Arps, Slate, Meagher & Flom LLP",
                },
            ]
        )

        self.assertEqual(
            payload["buy_side"]["top_by_count"][0]["counsel"],
            "Wiggin & Dana",
        )
        self.assertEqual(
            payload["buy_side"]["top_by_count"][0]["deal_count"],
            2,
        )
        self.assertEqual(
            payload["buy_side"]["top_by_value"][0]["counsel"],
            "Skadden, Arps, Slate, Meagher & Flom",
        )
        self.assertEqual(
            payload["sell_side"]["top_by_count"][0]["counsel"],
            "Wilson Sonsini Goodrich & Rosati",
        )
        self.assertTrue(
            any(
                row["counsel"] == "Goodwin Procter"
                and row["deal_count"] == 1
                and row["total_transaction_value"] == 50000000.0
                for row in payload["sell_side"]["top_by_count"]
            )
        )


if __name__ == "__main__":
    unittest.main()
