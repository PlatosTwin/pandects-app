import pytest

from etl.domain.counsel import format_counsel_display_name, split_counsel_names


def _split(value: str) -> list[str]:
    return [format_counsel_display_name(name) for name in split_counsel_names(value)]


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("RC Law LLP, Loeb & Loeb LLP", ["RC Law", "Loeb & Loeb"]),
        ("Liskow & Lewis, Thompson & Knight LLP", ["Liskow & Lewis", "Thompson & Knight"]),
        (
            "Morris, Nichols, Arsht & Tunnell and Baker & Hostetler LLP",
            ["Morris, Nichols, Arsht & Tunnell", "Baker & Hostetler"],
        ),
        ("Levett Rockwood P.C. and Edwards Wildman Palmer LLP", ["Levett Rockwood", "Edwards Wildman Palmer"]),
        (
            "McCann FitzGerald Solicitors and Skadden, Arps, Slate, Meagher & Flom LLP",
            ["McCann FitzGerald", "Skadden, Arps, Slate, Meagher & Flom"],
        ),
        ("Meitar Law Offices and McDermott Will & Emery LLP", ["Meitar Law Offices", "McDermott Will & Emery"]),
        (
            "Erdinast, Ben Nathan, Toledano & Co. with Hamburger Evron",
            ["Erdinast, Ben Nathan, Toledano & Co.", "Hamburger Evron"],
        ),
        ("Studio Legale Associato in association with Clifford Chance", ["Clifford Chance"]),
        ("Smith Mackinnon, PA & Smith Gambrell & Russell LLP", ["Smith Mackinnon, PA", "Smith Gambrell & Russell"]),
    ],
)
def test_joined_firms_are_split(value: str, expected: list[str]) -> None:
    assert _split(value) == expected


@pytest.mark.parametrize(
    "value",
    [
        "Weil, Gotshal & Manges LLP",
        "Skadden, Arps, Slate, Meagher & Flom LLP",
        "Wachtell, Lipton, Rosen & Katz",
        "Paul, Weiss, Rifkind, Wharton & Garrison LLP",
        "Jones, Walker, Waechter, Poitevent, Carrère & Denègre, L.L.P.",
        "Mitchell, Williams, Selig, Gates & Woodyard, P.L.L.C.",
        "Koley Jessen P.C., L.L.O.",
        "Hogan & Hartson L.L.P., Washington, D.C.",
        "Fenwick & West LLP, Mountain View",
        "Ward and Smith, P.A.",
        "Haynes and Boone, LLP",
        "Skadden, Arps, Slate, Meagher & Flom LLP & Affiliates",
    ],
)
def test_single_firm_names_are_not_split(value: str) -> None:
    assert len(split_counsel_names(value)) == 1
