import unittest
from typing import cast

from backend.counsel_leaderboards import (
    build_counsel_leaderboards,
    canonicalize_counsel_name,
    format_counsel_display_name,
    split_counsel_names,
)


def _dict_value(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload[key]
    if not isinstance(value, dict):
        raise AssertionError(f"Expected {key} to be an object.")
    return cast(dict[str, object], value)


def _list_of_dicts(payload: dict[str, object], key: str) -> list[dict[str, object]]:
    value = payload[key]
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise AssertionError(f"Expected {key} to be a list of objects.")
    return cast(list[dict[str, object]], value)


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

    def test_canonicalization_merges_known_aliases_and_typos(self) -> None:
        self.assertEqual(
            format_counsel_display_name("JonesDay"),
            "Jones Day",
        )
        self.assertEqual(
            canonicalize_counsel_name("Skadden, Arps, Slate, Meager & Flom"),
            canonicalize_counsel_name("Skadden, Arps, Slate, Meagher & Flom LLP & Affiliates"),
        )
        self.assertEqual(
            format_counsel_display_name("McGuire Woods"),
            "McGuireWoods",
        )
        self.assertEqual(
            format_counsel_display_name("Davis Polk & Wardwell London"),
            "Davis Polk & Wardwell",
        )
        self.assertEqual(
            format_counsel_display_name("Blake, Cassels & Graydon"),
            "Blakes, Cassels & Graydon",
        )
        self.assertEqual(
            format_counsel_display_name("Clifford Chance US"),
            "Clifford Chance",
        )
        self.assertEqual(
            format_counsel_display_name("Dentons Canada"),
            "Dentons",
        )
        self.assertEqual(
            format_counsel_display_name("Eaton & VanWinkle"),
            "Eaton & Van Winkle",
        )
        self.assertEqual(
            format_counsel_display_name("Goodwin Proctor"),
            "Goodwin Procter",
        )
        self.assertEqual(
            format_counsel_display_name("Greenberg, Traurig, PA"),
            "Greenberg Traurig",
        )
        self.assertEqual(
            format_counsel_display_name("Greenberg Traurig P.A."),
            "Greenberg Traurig",
        )
        self.assertEqual(
            format_counsel_display_name("Herzog, Fox & Neeman"),
            "Herzog, Fox & Ne'eman",
        )
        self.assertEqual(
            format_counsel_display_name("Hill, Ward & Henderson, P.A."),
            "Hill Ward Henderson",
        )
        self.assertEqual(
            format_counsel_display_name("Hogan Lovells US"),
            "Hogan Lovells",
        )
        self.assertEqual(
            format_counsel_display_name("JunHe Law Offices"),
            "Jun He Law Offices",
        )
        self.assertEqual(
            format_counsel_display_name("Ledgewood Law Firm"),
            "Ledgewood",
        )
        self.assertEqual(
            format_counsel_display_name("LEWIS, RICE & FINGERSH, L.C."),
            "Lewis, Rice & Fingersh",
        )
        self.assertEqual(
            format_counsel_display_name("Manatt, Phelps & Philips"),
            "Manatt, Phelps & Phillips",
        )
        self.assertEqual(
            format_counsel_display_name("Rimôn"),
            "Rimon",
        )
        self.assertEqual(
            format_counsel_display_name("Watchell, Lipton, Rosen & Katz"),
            "Wachtell, Lipton, Rosen & Katz",
        )

    def test_split_counsel_names_handles_known_multi_firm_ampersand_entries(self) -> None:
        self.assertEqual(
            split_counsel_names("Davis Polk & Wardwell & Osler, Hoskin & Harcourt"),
            ["Davis Polk & Wardwell", "Osler, Hoskin & Harcourt"],
        )
        self.assertEqual(
            split_counsel_names("Meitar | Law Offices & Greenberg Traurig"),
            ["Meitar | Law Offices", "Greenberg Traurig"],
        )
        self.assertEqual(
            split_counsel_names("Sidley & Austin, Orrick, Herrington & Sutcliffe"),
            ["Sidley Austin", "Orrick, Herrington & Sutcliffe"],
        )
        self.assertEqual(
            split_counsel_names("Arthur Cox & Simpson Thacher & Bartlett"),
            ["Arthur Cox", "Simpson Thacher & Bartlett"],
        )
        self.assertEqual(
            split_counsel_names(
                "Baker, Donelson, Bearman, Caldwell & Berkowitz, P.C. & Akin Gump Strauss Hauer & Feld"
            ),
            [
                "Baker, Donelson, Bearman, Caldwell & Berkowitz",
                "Akin Gump Strauss Hauer & Feld",
            ],
        )
        self.assertEqual(
            split_counsel_names("Eversheds Sutherland (US) LLP, Simpson Thacher & Bartlett"),
            ["Eversheds Sutherland", "Simpson Thacher & Bartlett"],
        )
        self.assertEqual(
            split_counsel_names(
                "Faegre Drinker Biddle & Reath LLP, Goodmans LLP, Kirkland & Ellis"
            ),
            ["Faegre Drinker Biddle & Reath", "Goodmans", "Kirkland & Ellis"],
        )
        self.assertEqual(
            split_counsel_names("Hinkle Elkouri Law Firm L.L.C. & Thompson & Knight"),
            ["Hinkle Elkouri Law Firm", "Thompson & Knight"],
        )
        self.assertEqual(
            split_counsel_names("Ropes & Gray LLP, Hogan Lovells US"),
            ["Ropes & Gray", "Hogan Lovells"],
        )
        self.assertEqual(
            split_counsel_names("Vinson & Elkins L.L.P. & Clifford Chance"),
            ["Vinson & Elkins", "Clifford Chance"],
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

        buy_side = _dict_value(payload, "buy_side")
        sell_side = _dict_value(payload, "sell_side")
        buy_top_by_count = _list_of_dicts(buy_side, "top_by_count")
        buy_top_by_value = _list_of_dicts(buy_side, "top_by_value")
        sell_top_by_count = _list_of_dicts(sell_side, "top_by_count")
        buy_annual = _list_of_dicts(buy_side, "annual")

        self.assertEqual(
            buy_top_by_count[0]["counsel"],
            "Wiggin & Dana",
        )
        self.assertEqual(
            buy_top_by_count[0]["deal_count"],
            2,
        )
        self.assertEqual(
            buy_top_by_value[0]["counsel"],
            "Skadden, Arps, Slate, Meagher & Flom",
        )
        self.assertEqual(
            sell_top_by_count[0]["counsel"],
            "Wilson Sonsini Goodrich & Rosati",
        )
        self.assertEqual(
            buy_annual[0],
            {
                "year": 2022,
                "top_by_count": [
                    {
                        "counsel": "Skadden, Arps, Slate, Meagher & Flom",
                        "deal_count": 1,
                        "total_transaction_value": 300000000.0,
                        "years": [
                            {
                                "year": 2022,
                                "deal_count": 1,
                                "total_transaction_value": 300000000.0,
                            }
                        ],
                    }
                ],
                "top_by_value": [
                    {
                        "counsel": "Skadden, Arps, Slate, Meagher & Flom",
                        "deal_count": 1,
                        "total_transaction_value": 300000000.0,
                        "years": [
                            {
                                "year": 2022,
                                "deal_count": 1,
                                "total_transaction_value": 300000000.0,
                            }
                        ],
                    }
                ],
            },
        )
        self.assertTrue(
            any(
                row["counsel"] == "Goodwin Procter"
                and row["deal_count"] == 1
                and row["total_transaction_value"] == 50000000.0
                for row in sell_top_by_count
            )
        )


if __name__ == "__main__":
    unittest.main()
