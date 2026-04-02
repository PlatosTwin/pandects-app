from __future__ import annotations

import re


_MULTI_COUNSEL_SPLIT_RE = re.compile(r"\s*;\s*|\s*\n+\s*|\s+/\s+")
_AND_SUFFIX_CONNECTOR_RE = re.compile(
    r"(?i)\b(?:llp|l\.l\.p\.|pc|p\.c\.|professional corporation|pllc|plc|llc|ltd|lp|l\.p\.|n\.v\.|a\.p\.c\.|corp\.|corporation)\b\s+and\s+"
)
_TRAILING_GEO_PAREN_RE = re.compile(r"\s*\((?:us|u\.s\.|usa|ny|delaware|uk)\)\s*$", re.I)
_TRAILING_SUFFIX_RE = re.compile(
    "".join(
        (
            r"(?i)(?:",
            r"(?:,|\s)+",
            r"(?:",
            r"llp|l\.l\.p\.|limited liability partnership|",
            r"pc|p\.c\.|professional corporation|",
            r"pllc|plc|llc|ltd|",
            r"lp|l\.p\.|",
            r"corp\.?|corporation|",
            r"inc\.?|",
            r"n\.v\.|",
            r"a\.p\.c\.|",
            r"s\.c\.",
            r")",
            r")+\.?\s*$",
        )
    )
)
_CANONICAL_PUNCT_RE = re.compile(r"[^a-z0-9 ]+")
_CANONICAL_SPACE_RE = re.compile(r"\s+")
_DISPLAY_SPACE_RE = re.compile(r"\s+")
_CANONICAL_STOPWORDS = {
    "llp",
    "pc",
    "professional",
    "corporation",
    "pllc",
    "plc",
    "llc",
    "lp",
    "corp",
    "inc",
    "ltd",
    "nv",
    "apc",
    "sc",
}
_EXACT_DISPLAY_OVERRIDES = {
    "Advokatfirma DLA Piper Norway DA": "DLA Piper",
    "Becker & Poliakoff, P.A.": "Becker & Poliakoff",
    "Eaton & VanWinkle": "Eaton & Van Winkle",
    "Fox Rothschild": "Fox Rothchild",
    "Freshfields Bruckhaus Deringer US": "Freshfields Bruckhaus Deringer",
    "Fried, Frank, Harris, Shiver & Jacobson": "Fried, Frank, Harris, Shriver & Jacobson",
    "Fulbright & Jaworski L.L.P": "Fulbright & Jaworski",
    "Goodwin Proctor": "Goodwin Procter",
    "Greenberg Traurig, P.A.": "Greenberg Traurig",
    "Greenberg Traurig P.A.": "Greenberg Traurig",
    "Greenberg, Traurig, PA": "Greenberg Traurig",
    "Gunderson Dettmer Stough Villeneuve Franklin & Hachigian": "Gunderson Dettmer Stough Villenueve Franklin & Hachigian",
    "Akerman Senterfitt, P.A.": "Akerman Senterfitt",
    "Allen & Overy SCS": "Allen & Overy",
    "Baker, Donelson, Bearman, Caldwell & Berkowitz, a": "Baker, Donelson, Bearman, Caldwell & Berkowitz",
    "Blake, Cassels & Graydon": "Blakes, Cassels & Graydon",
    "Bryan Cave HRO": "Bryan Cave",
    "Buchalter, a": "Buchalter",
    "Clifford Chance Prague": "Clifford Chance",
    "Clifford Chance US": "Clifford Chance",
    "Cooley HK": "Cooley",
    "Crowe & Dunlevy, A": "Crowe & Dunlevy",
    "Dentons Canada": "Dentons",
    "Dentons US": "Dentons",
    "JonesDay": "Jones Day",
    "Kirkland & Ellis International": "Kirkland & Ellis",
    "Latham & Watkins (London)": "Latham & Watkins",
    "Skadden, Arps, Slate, Meager & Flom": "Skadden, Arps, Slate, Meagher & Flom",
    "Skadden, Arps, Slate, Meager & Flom (Illinois)": "Skadden, Arps, Slate, Meagher & Flom",
    "Skadden, Arps, Slate, Meagher & Flom (Illinois)": "Skadden, Arps, Slate, Meagher & Flom",
    "Skadden, Arps, Slate, Meagher & Flom LLP & Affiliates": "Skadden, Arps, Slate, Meagher & Flom",
    "Wilmer Culter Pickering Hale & Dorr": "Wilmer Cutler Pickering Hale & Dorr",
    "Sidley Austin Brown & Wood": "Sidley Austin",
    "Simpson Thacher & Barlett": "Simpson Thacher & Bartlett",
    "Simpson Thacher & Bartlettt": "Simpson Thacher & Bartlett",
    "Simpson Thatcher & Bartlett": "Simpson Thacher & Bartlett",
    "Pillsbury Winthrop Shaw Pitman": "Pillsbury Winthrop Shaw Pittman",
    "Parr Brown Gee & Loveless, P.C": "Parr Brown Gee & Loveless",
    "Norton Rose Canada": "Norton Rose",
    "Norton Rose Fulbright (Asia)": "Norton Rose Fulbright",
    "Norton Rose Fulbright Canada": "Norton Rose Fulbright",
    "Norton Rose Fulbright US": "Norton Rose Fulbright",
    "Neuberger Quinn Gielen Rubin Gibber P.A.": "Neuberger Quinn Gielen Rubin & Gibber",
    "Neuberger, Quinn, Gielen, Rubin & Gibber, PA": "Neuberger Quinn Gielen Rubin & Gibber",
    "Morrison & Forrester": "Morrison & Foerster",
    "Mintz, Levin, Cohn, Ferris, Glosky & Popeo": "Mintz, Levin, Cohn, Ferris, Glovsky & Popeo",
    "Mintz, Levin, Cohn, Ferris, Glovsky & Pope": "Mintz, Levin, Cohn, Ferris, Glovsky & Popeo",
    "Mintz, Levin, Cohn, Ferris, Glovsky & Popeo, P.C": "Mintz, Levin, Cohn, Ferris, Glovsky & Popeo",
    "Mintz, Levin, Cohn, Ferris, Glovsky, & Popeo": "Mintz, Levin, Cohn, Ferris, Glovsky & Popeo",
    "Milbank (Hong Kong)": "Milbank",
    "Meitar Liquornik Geva Leshem Tal, Law Offices": "Meitar Liquornik Geva Leshem Tal",
    "McGuire Woods": "McGuireWoods",
    "McGrath North Mullin & Kratz, PC LLO": "McGrath, North, Mullin & Kratz",
    "McCarthy Tétrault": "McCarthy Tetrault",
    "Luse Gorman Pomerenck & Schick": "Luse Gorman Pomerenk & Schick",
    "Lindquist & Vennum P.L.L.P.": "Lindquist & Vennum",
    "Lindquist & Vennum, PLLP": "Lindquist & Vennum",
    "Lindquist + Vennum": "Lindquist & Vennum",
    "Lewis, Rice & Fingersh, L.C.": "Lewis, Rice & Fingersh",
    "Lewis, Rice & Fingersh, LC": "Lewis, Rice & Fingersh",
    "Leonard, Street & Deinard P.A.": "Leonard, Street & Deinard",
    "Leonard, Street & Deinard PA": "Leonard, Street & Deinard",
    "Leonard, Street & Deinard Professional Association": "Leonard, Street & Deinard",
    "LeClair Ryan, A": "LeClairRyan",
    "Kohrman Jackson & Krantz P.L.L.": "Kohrman Jackson & Krantz",
    "Kohrman Jackson & Krantz, PLL": "Kohrman Jackson & Krantz",
    "DLA Piper (Canada)": "DLA Piper",
    "DLA Piper Rudnick Gray Cary": "DLA Piper",
    "DLA Piper Rudnick Gray Cary US": "DLA Piper",
    "DLA Piper Singapore Pte.": "DLA Piper",
    "DLA Piper UK": "DLA Piper",
    "DLA Piper US": "DLA Piper",
    "Davis Polk & Wardell": "Davis Polk & Wardwell",
    "Davis Polk & Wardwell London": "Davis Polk & Wardwell",
}
_EXACT_MULTI_FIRM_SPLITS = {
    "Eversheds Sutherland (US) LLP, Simpson Thacher & Bartlett": [
        "Eversheds Sutherland",
        "Simpson Thacher & Bartlett",
    ],
    "Faegre Drinker Biddle & Reath LLP, Goodmans LLP, Kirkland & Ellis": [
        "Faegre Drinker Biddle & Reath",
        "Goodmans",
        "Kirkland & Ellis",
    ],
    "Akerman Senterfitt & Berger Singerman, P.A.": [
        "Akerman Senterfitt",
        "Berger Singerman",
    ],
    "Arthur Cox & Simpson Thacher & Bartlett": [
        "Arthur Cox",
        "Simpson Thacher & Bartlett",
    ],
    "Baker, Donelson, Bearman, Caldwell & Berkowitz, P.C. & Akin Gump Strauss Hauer & Feld": [
        "Baker, Donelson, Bearman, Caldwell & Berkowitz",
        "Akin Gump Strauss Hauer & Feld",
    ],
    "Jones Day & Caraza y Morayta": ["Jones Day", "Caraza y Morayta"],
    "Kirkland & Ellis LLP, Wyrick Robbins Yates & Ponton": [
        "Kirkland & Ellis",
        "Wyrick Robbins Yates & Ponton",
    ],
    "Latham & Watkins LLP, Stikeman Elliott": ["Latham & Watkins", "Stikeman Elliott"],
    "Slaughter & May, Skadden, Arps, Slate, Meagher & Flom": [
        "Slaughter & May",
        "Skadden, Arps, Slate, Meagher & Flom",
    ],
    "Sidley & Austin, Orrick, Herrington & Sutcliffe": [
        "Sidley Austin",
        "Orrick, Herrington & Sutcliffe",
    ],
    "Meitar | Law Offices & Goodwin Procter": ["Meitar | Law Offices", "Goodwin Procter"],
    "Meitar | Law Offices & Greenberg Traurig": ["Meitar | Law Offices", "Greenberg Traurig"],
    "DLA Piper LLP (US) & Hogan Lovells US": ["DLA Piper", "Hogan Lovells"],
    "Davis Polk & Wardwell & Andrews Kurth": ["Davis Polk & Wardwell", "Andrews Kurth"],
    "Davis Polk & Wardwell & Munger, Tolles & Olson": [
        "Davis Polk & Wardwell",
        "Munger, Tolles & Olson",
    ],
    "Davis Polk & Wardwell & Osler, Hoskin & Harcourt": [
        "Davis Polk & Wardwell",
        "Osler, Hoskin & Harcourt",
    ],
}
_PREFERRED_DISPLAY_BY_NORMALIZED = {
    "advokatfirma dla piper norway da": "DLA Piper",
    "akerman senterfitt": "Akerman Senterfitt",
    "becker poliakoff": "Becker & Poliakoff",
    "eaton vanwinkle": "Eaton & Van Winkle",
    "fox rothschild": "Fox Rothchild",
    "freshfields bruckhaus deringer us": "Freshfields Bruckhaus Deringer",
    "fried frank harris shiver jacobson": "Fried, Frank, Harris, Shriver & Jacobson",
    "fulbright jaworski l l p": "Fulbright & Jaworski",
    "goodwin proctor": "Goodwin Procter",
    "greenberg traurig p a": "Greenberg Traurig",
    "greenberg traurig pa": "Greenberg Traurig",
    "gunderson dettmer stough villeneuve franklin hachigian": "Gunderson Dettmer Stough Villenueve Franklin & Hachigian",
    "allen overy scs": "Allen & Overy",
    "baker donelson bearman caldwell berkowitz a": "Baker, Donelson, Bearman, Caldwell & Berkowitz",
    "blake cassels graydon": "Blakes, Cassels & Graydon",
    "bryan cave hro": "Bryan Cave",
    "buchalter a": "Buchalter",
    "clifford chance prague": "Clifford Chance",
    "clifford chance us": "Clifford Chance",
    "cooley hk": "Cooley",
    "crowe dunlevy a": "Crowe & Dunlevy",
    "dentons canada": "Dentons",
    "dentons us": "Dentons",
    "jonesday": "Jones Day",
    "kirkland ellis international": "Kirkland & Ellis",
    "latham watkins london": "Latham & Watkins",
    "skadden arps slate meager flom": "Skadden, Arps, Slate, Meagher & Flom",
    "skadden arps slate meager flom illinois": "Skadden, Arps, Slate, Meagher & Flom",
    "skadden arps slate meagher flom illinois": "Skadden, Arps, Slate, Meagher & Flom",
    "skadden arps slate meagher flom affiliates": "Skadden, Arps, Slate, Meagher & Flom",
    "wilmer culter pickering hale dorr": "Wilmer Cutler Pickering Hale & Dorr",
    "sidley austin brown wood": "Sidley Austin",
    "simpson thacher barlett": "Simpson Thacher & Bartlett",
    "simpson thacher bartlettt": "Simpson Thacher & Bartlett",
    "simpson thatcher bartlett": "Simpson Thacher & Bartlett",
    "pillsbury winthrop shaw pitman": "Pillsbury Winthrop Shaw Pittman",
    "parr brown gee loveless": "Parr Brown Gee & Loveless",
    "norton rose canada": "Norton Rose",
    "norton rose fulbright asia": "Norton Rose Fulbright",
    "norton rose fulbright canada": "Norton Rose Fulbright",
    "norton rose fulbright us": "Norton Rose Fulbright",
    "neuberger quinn gielen rubin gibber": "Neuberger Quinn Gielen Rubin & Gibber",
    "morrison forrester": "Morrison & Foerster",
    "mintz levin cohn ferris glosky popeo": "Mintz, Levin, Cohn, Ferris, Glovsky & Popeo",
    "mintz levin cohn ferris glovsky pope": "Mintz, Levin, Cohn, Ferris, Glovsky & Popeo",
    "mintz levin cohn ferris glovsky popeo": "Mintz, Levin, Cohn, Ferris, Glovsky & Popeo",
    "milbank hong kong": "Milbank",
    "meitar liquornik geva leshem tal law offices": "Meitar Liquornik Geva Leshem Tal",
    "mcguire woods": "McGuireWoods",
    "mcgrath north mullin kratz pc llo": "McGrath, North, Mullin & Kratz",
    "mccarthy tetrault": "McCarthy Tetrault",
    "luse gorman pomerenck schick": "Luse Gorman Pomerenk & Schick",
    "lindquist vennum": "Lindquist & Vennum",
    "lewis rice fingersh": "Lewis, Rice & Fingersh",
    "leonard street deinard": "Leonard, Street & Deinard",
    "leclair ryan a": "LeClairRyan",
    "leclairryan": "LeClairRyan",
    "kohrman jackson krantz": "Kohrman Jackson & Krantz",
    "dla piper canada": "DLA Piper",
    "dla piper rudnick gray cary": "DLA Piper",
    "dla piper rudnick gray cary us": "DLA Piper",
    "dla piper singapore pte": "DLA Piper",
    "dla piper uk": "DLA Piper",
    "dla piper us": "DLA Piper",
    "davis polk wardell": "Davis Polk & Wardwell",
    "davis polk wardwell london": "Davis Polk & Wardwell",
}
_MULTI_FIRM_SPLITS_BY_NORMALIZED = {
    "eversheds sutherland us simpson thacher bartlett": [
        "Eversheds Sutherland",
        "Simpson Thacher & Bartlett",
    ],
    "faegre drinker biddle reath goodmans kirkland ellis": [
        "Faegre Drinker Biddle & Reath",
        "Goodmans",
        "Kirkland & Ellis",
    ],
    "akerman senterfitt berger singerman": [
        "Akerman Senterfitt",
        "Berger Singerman",
    ],
    "arthur cox simpson thacher bartlett": [
        "Arthur Cox",
        "Simpson Thacher & Bartlett",
    ],
    "baker donelson bearman caldwell berkowitz akin gump strauss hauer feld": [
        "Baker, Donelson, Bearman, Caldwell & Berkowitz",
        "Akin Gump Strauss Hauer & Feld",
    ],
    "jones day caraza y morayta": ["Jones Day", "Caraza y Morayta"],
    "kirkland ellis wyrick robbins yates ponton": [
        "Kirkland & Ellis",
        "Wyrick Robbins Yates & Ponton",
    ],
    "latham watkins stikeman elliott": ["Latham & Watkins", "Stikeman Elliott"],
    "slaughter may skadden arps slate meagher flom": [
        "Slaughter & May",
        "Skadden, Arps, Slate, Meagher & Flom",
    ],
    "sidley austin orrick herrington sutcliffe": [
        "Sidley Austin",
        "Orrick, Herrington & Sutcliffe",
    ],
    "meitar law offices goodwin procter": ["Meitar | Law Offices", "Goodwin Procter"],
    "meitar law offices greenberg traurig": ["Meitar | Law Offices", "Greenberg Traurig"],
    "dla piper hogan lovells": ["DLA Piper", "Hogan Lovells"],
    "davis polk wardwell andrews kurth": ["Davis Polk & Wardwell", "Andrews Kurth"],
    "davis polk wardwell munger tolles olson": [
        "Davis Polk & Wardwell",
        "Munger, Tolles & Olson",
    ],
    "davis polk wardwell osler hoskin harcourt": [
        "Davis Polk & Wardwell",
        "Osler, Hoskin & Harcourt",
    ],
}


def _normalized_counsel_key(value: str) -> str:
    normalized = value.lower()
    normalized = normalized.replace("&", " and ")
    normalized = _CANONICAL_PUNCT_RE.sub(" ", normalized)
    normalized = normalized.replace(" and ", " ")
    normalized = _CANONICAL_SPACE_RE.sub(" ", normalized).strip()
    tokens = [
        token for token in normalized.split(" ") if token and token not in _CANONICAL_STOPWORDS
    ]
    return " ".join(tokens)


def _clean_display_candidate(raw_name: str) -> str:
    cleaned = raw_name.strip().strip(";,")
    if not cleaned:
        return ""
    while True:
        next_cleaned = _TRAILING_GEO_PAREN_RE.sub("", cleaned).strip()
        next_cleaned = _TRAILING_SUFFIX_RE.sub("", next_cleaned).strip().strip(",;")
        if next_cleaned == cleaned:
            break
        cleaned = next_cleaned
    cleaned = re.sub(r"\s+and\s+", " & ", cleaned, flags=re.I)
    cleaned = _DISPLAY_SPACE_RE.sub(" ", cleaned).strip()
    return cleaned


def _split_after_legal_suffix_connector(value: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for match in _AND_SUFFIX_CONNECTOR_RE.finditer(value):
        split_at = value[: match.end()].lower().rfind(" and ")
        if split_at <= start:
            continue
        parts.append(value[start:split_at])
        start = match.end()
    parts.append(value[start:])
    return parts


def split_counsel_names(value: object | None) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, str):
        raise TypeError("Counsel value must be a string or null.")
    cleaned = value.strip()
    if not cleaned:
        return []

    segments: list[str] = []
    for chunk in _MULTI_COUNSEL_SPLIT_RE.split(cleaned):
        chunk = chunk.strip().strip(";,")
        if not chunk:
            continue
        for subchunk in _split_after_legal_suffix_connector(chunk):
            normalized = subchunk.strip().strip(";,")
            if normalized:
                cleaned = _clean_display_candidate(normalized)
                if not cleaned:
                    continue
                exact_split = _EXACT_MULTI_FIRM_SPLITS.get(cleaned)
                if exact_split is not None:
                    segments.extend(exact_split)
                    continue
                normalized_key = _normalized_counsel_key(cleaned)
                split_names = _MULTI_FIRM_SPLITS_BY_NORMALIZED.get(normalized_key)
                if split_names is None:
                    segments.append(normalized)
                    continue
                segments.extend(split_names)
    return segments


def canonicalize_counsel_name(value: object | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Counsel value must be a string or null.")
    display_candidate = _clean_display_candidate(value)
    if not display_candidate:
        return None
    exact_override = _EXACT_DISPLAY_OVERRIDES.get(display_candidate)
    if exact_override is not None:
        return _normalized_counsel_key(exact_override)
    normalized_key = _normalized_counsel_key(display_candidate)
    if not normalized_key:
        return None
    preferred_display = _PREFERRED_DISPLAY_BY_NORMALIZED.get(normalized_key)
    if preferred_display is not None:
        return _normalized_counsel_key(preferred_display)
    return normalized_key


def format_counsel_display_name(value: object | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError("Counsel value must be a string or null.")
    display_candidate = _clean_display_candidate(value)
    if not display_candidate:
        return None
    exact_override = _EXACT_DISPLAY_OVERRIDES.get(display_candidate)
    if exact_override is not None:
        return exact_override
    normalized_key = _normalized_counsel_key(display_candidate)
    return _PREFERRED_DISPLAY_BY_NORMALIZED.get(normalized_key, display_candidate)
