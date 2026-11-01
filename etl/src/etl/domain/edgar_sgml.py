"""Strip EDGAR SGML markup from pages that came from plain-text (.txt) filings.

Old EDGAR filings were submitted as SGML-wrapped plain text. Their page text
still carries the wrapper and the text-table markup: a document header
(``<DOCUMENT> <TYPE>EX-2.1 <SEQUENCE>2 <FILENAME>ex2-1.txt <DESCRIPTION>...
<TEXT>``), the closing ``</TEXT> </DOCUMENT>``, table markup (``<TABLE>``,
``<CAPTION>``, ``<S>``, ``<C>``, ``<FN>``), page breaks (``<PAGE>``) and, in a
few filings, the SEC header block (``<SEC-HEADER>...</SEC-HEADER>``). The XML
builder used to pass all of it through into served section text.

The strip is deliberately narrow:

* Only tags whose names are on an EDGAR whitelist are removed. There is no
  generic ``<...>`` pattern, so angle-bracket text that is real content
  (``<1%``, ``<john@example.com>``, ``<21,178,538>``) is left alone.
* The builder's own markup is never touched: ``<article>``, ``<section>`` and
  paired ``<page>N</page>`` markers survive. A lowercase ``<page>`` is removed
  only when it is an orphan EDGAR page break (no ``</page>`` follows before
  the next tag).
* Whole lines made only of ``-`` or ``=`` rule characters (at least 10 of
  them, spaces allowed between runs) are removed when they are decoration or
  text-table rules. A rule line that is a fill-in blank or a signature line is
  kept: see ``_keep_rule_line``. Underscore runs are never removed.
* A line that held nothing but SGML markup is removed; every other line keeps
  its text, with the tags taken out.

Callers apply this only to pages whose source is a .txt filing.
"""
import re
from dataclasses import dataclass, field

# Tags removed wherever they appear (case-insensitive). Each name is EDGAR
# SGML: the document wrapper, text-table markup, and the SEC header block.
EDGAR_TAG_NAMES: tuple[str, ...] = (
    # Document wrapper (the header unit below also takes the values).
    "DOCUMENT",
    "TYPE",
    "SEQUENCE",
    "FILENAME",
    "DESCRIPTION",
    # Text-table and preformatted-text markup.
    "TABLE",
    "CAPTION",
    "S",
    "C",
    "FN",
    "PRE",
    # SEC header block.
    "SEC-DOCUMENT",
    "SEC-HEADER",
    "IMS-DOCUMENT",
    "IMS-HEADER",
    "FILER",
    "COMPANY-DATA",
    "FILING-VALUES",
    "BUSINESS-ADDRESS",
    "MAIL-ADDRESS",
    "FORMER-COMPANY",
    "SUBJECT-COMPANY",
    "FILED-BY",
)

# Placeholder for removed markup. NUL never survives into page text: the
# builder strips XML-invalid control characters first.
_MARK = "\x00"

# The SEC header block is filing metadata (accession number, filer addresses);
# when a page holds the whole block it goes as one unit.
_SEC_HEADER_BLOCK_RE = re.compile(r"<SEC-HEADER>.*?</SEC-HEADER>", re.IGNORECASE | re.DOTALL)
# An EDGAR page break. The builder's own <page>N</page> marker always has its
# closing tag before any other tag, so an opening tag followed by one is kept.
_PAGE_TAG_RE = re.compile(r"<page>(?![^<]*</page>)", re.IGNORECASE)
# "<DOCUMENT> <TYPE>EX-2.1 <SEQUENCE>2 <FILENAME>ex.txt <DESCRIPTION>... <TEXT>":
# the header values are filing metadata, not agreement text, so the whole unit
# goes. Each value stops at the next "<" or the end of the line.
_DOCUMENT_HEADER_RE = re.compile(
    r"<DOCUMENT>[ \t]*"
    r"(?:<TYPE>[^<\n]*)?"
    r"(?:<SEQUENCE>[^<\n]*)?"
    r"(?:<FILENAME>[^<\n]*)?"
    r"(?:<DESCRIPTION>[^<\n]*)?"
    r"<TEXT>",
    re.IGNORECASE,
)
_WHITELIST_TAG_RE = re.compile(
    r"</?(?:" + "|".join(re.escape(name) for name in EDGAR_TAG_NAMES) + r")>",
    re.IGNORECASE,
)
# <TEXT> is matched in upper case only: a lowercase <text> is an XML builder
# element name and must never be touched.
_TEXT_TAG_RE = re.compile(r"</?TEXT>")
# Whole-text passes (they may span lines), then per-line passes.
_TEXT_PATTERNS = (_SEC_HEADER_BLOCK_RE, _PAGE_TAG_RE)
_LINE_PATTERNS = (_DOCUMENT_HEADER_RE, _WHITELIST_TAG_RE, _TEXT_TAG_RE)

_TABLE_OPEN_RE = re.compile(r"<TABLE>", re.IGNORECASE)
_TABLE_CLOSE_RE = re.compile(r"</TABLE>", re.IGNORECASE)

# One flat character class: a nested quantifier here backtracks exponentially
# on a long dash run that ends in another character.
_RULE_LINE_RE = re.compile(r"[-= \t]+")
_MIN_RULE_CHARS = 10

# Neighbors that mark a rule line as a fill-in blank or a signature line.
_FORM_LABEL_RE = re.compile(
    r"^(?:By|Name|Title|Its|Address|Attention|Attn|Telephone|Fax|Witness|Signature|Print|Date)\b"
)
# "Seller's Initials ____ Purchaser's Initials ____" style labels.
_FORM_WORD_RE = re.compile(r"\b(?:Initials|Signature)\b")
# A neighboring blank: a dash or underscore line. A "=" line is decoration
# (title and "Dated as of" borders), so it never makes its neighbor a blank.
_BLANKISH_LINE_RE = re.compile(r"[-_ ]{10,}")
_UNDERSCORE_BLANK_RE = re.compile(r"_{4,}")
_SIGNATORY_MAX_CHARS = 60


@dataclass
class EdgarSgmlStripResult:
    text: str
    tags_removed: list[str] = field(default_factory=list)
    lines_removed: list[str] = field(default_factory=list)
    # Rule lines kept as fill-in blanks or signature lines, as (line, reason).
    rule_lines_kept: list[tuple[str, str]] = field(default_factory=list)


def _is_rule_line(line: str) -> bool:
    if not _RULE_LINE_RE.fullmatch(line):
        return False
    return sum(1 for ch in line if ch in "-=") >= _MIN_RULE_CHARS


def _is_short_all_caps_name(line: str) -> bool:
    if not line or len(line) > _SIGNATORY_MAX_CHARS:
        return False
    letters = [ch for ch in line if ch.isalpha()]
    return len(letters) >= 2 and all(ch.isupper() for ch in letters)


def _keep_rule_line(
    line: str, prev_line: str | None, next_line: str | None, *, page_order: int | None
) -> str | None:
    """Return why a rule line outside a text table is a blank to keep, else None."""
    if prev_line is not None and prev_line.endswith((":", ",")):
        return "after_colon_or_comma"
    for neighbor in (prev_line, next_line):
        if neighbor is None:
            continue
        if _FORM_LABEL_RE.match(neighbor) or _FORM_WORD_RE.search(neighbor):
            return "form_label"
        # A stack of dash lines is a column of blanks (witness addresses); a "="
        # line is decoration, so it never counts as a blank itself.
        if "=" not in line and (
            _BLANKISH_LINE_RE.fullmatch(neighbor) or _UNDERSCORE_BLANK_RE.search(neighbor)
        ):
            return "next_to_blank"
    if page_order is not None and page_order > 2:
        for neighbor in (prev_line, next_line):
            if neighbor is not None and _is_short_all_caps_name(neighbor):
                return "signatory_name"
    return None


def strip_edgar_sgml_with_report(
    text: str, *, page_type: str | None = None, page_order: int | None = None
) -> EdgarSgmlStripResult:
    """Strip EDGAR SGML markup and decorative rule lines; report what went.

    ``page_type`` is the page's label (``sig`` pages keep every rule line) and
    ``page_order`` its position in the agreement.
    """
    result = EdgarSgmlStripResult(text=text)
    if not text:
        return result

    work = text
    for pattern in _TEXT_PATTERNS:
        matches = [m.group(0) for m in pattern.finditer(work)]
        if not matches:
            continue
        if pattern is _SEC_HEADER_BLOCK_RE:
            result.tags_removed.extend("<SEC-HEADER>...</SEC-HEADER>" for _ in matches)
            # The removed block keeps the lines around it apart.
            work = pattern.sub("\n" + _MARK + "\n", work)
        else:
            result.tags_removed.extend(matches)
            work = pattern.sub(_MARK, work)

    lines: list[str] = []
    in_table: list[bool] = []
    table_open = False
    for line in work.split("\n"):
        # A rule line belongs to a text table when it sits inside <TABLE>..</TABLE>
        # or on a line that opens or closes one.
        opens = bool(_TABLE_OPEN_RE.search(line))
        closes = bool(_TABLE_CLOSE_RE.search(line))
        line_in_table = table_open or opens or closes
        if opens and not closes:
            table_open = True
        elif closes and not opens:
            table_open = False
        elif opens and closes:
            table_open = line.lower().rfind("<table>") > line.lower().rfind("</table>")

        new_line = line
        for pattern in _LINE_PATTERNS:
            matches = [m.group(0) for m in pattern.finditer(new_line)]
            if matches:
                result.tags_removed.extend(matches)
                new_line = pattern.sub(_MARK, new_line)
        if _MARK in new_line:
            # A run of markup and the spaces around it collapse to one space, so
            # "Exhibit 99.1 <PAGE> SECTION 3.14" reads "Exhibit 99.1 SECTION 3.14".
            new_line = " ".join(part.strip() for part in new_line.split(_MARK) if part.strip())
            if not new_line:
                # The line held nothing but SGML markup.
                continue
        lines.append(new_line)
        in_table.append(line_in_table)

    nonblank = [i for i, line in enumerate(lines) if line.strip()]
    position = {line_index: k for k, line_index in enumerate(nonblank)}
    kept: list[str] = []
    for i, line in enumerate(lines):
        if not _is_rule_line(line):
            kept.append(line)
            continue
        stripped = line.strip()
        if page_type == "sig":
            result.rule_lines_kept.append((stripped, "sig_page"))
            kept.append(line)
            continue
        if not in_table[i]:
            k = position[i]
            prev_line = lines[nonblank[k - 1]].strip() if k > 0 else None
            next_line = lines[nonblank[k + 1]].strip() if k + 1 < len(nonblank) else None
            reason = _keep_rule_line(stripped, prev_line, next_line, page_order=page_order)
            if reason is not None:
                result.rule_lines_kept.append((stripped, reason))
                kept.append(line)
                continue
        result.lines_removed.append(stripped)
    result.text = "\n".join(kept)
    return result


def strip_edgar_sgml(
    text: str, *, page_type: str | None = None, page_order: int | None = None
) -> str:
    """Return ``text`` with EDGAR SGML markup and decorative rule lines removed."""
    return strip_edgar_sgml_with_report(text, page_type=page_type, page_order=page_order).text
