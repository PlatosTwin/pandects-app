"""Strip EDGAR SGML markup from pages that came from plain-text (.txt) filings.

Old EDGAR filings were submitted as SGML-wrapped plain text. Their page text
still carries the wrapper and the text-table markup: a document header
(``<DOCUMENT> <TYPE>EX-2.1 <SEQUENCE>2 <FILENAME>ex2-1.txt <DESCRIPTION>...
<TEXT>``), the closing ``</TEXT> </DOCUMENT>``, table markup (``<TABLE>``,
``<CAPTION>``, ``<S>``, ``<C>``, ``<FN>``), page breaks (``<PAGE>``) and, in a
few filings, the SEC header block (``<SEC-HEADER>``, ``</FILER>``...). The XML
builder used to pass all of it through into served section text.

The strip is deliberately narrow:

* Only tags whose names are on an EDGAR whitelist are removed. There is no
  generic ``<...>`` pattern, so angle-bracket text that is real content
  (``<1%``, ``<john@example.com>``, ``<21,178,538>``) is left alone.
* The builder's own markup is never touched: ``<article>``, ``<section>`` and
  paired ``<page>N</page>`` markers survive. A lowercase ``<page>`` is removed
  only when it is an orphan EDGAR page break (no ``</page>`` follows on its
  line).
* Whole lines made only of ``-`` or ``=`` rule characters (at least 10 of
  them, spaces allowed between runs) are removed: they are text-table rules.
  Underscore runs are never removed; they are blanks and signature lines.
* A line that held nothing but SGML tags is removed; every other line keeps
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
# An EDGAR page break. The builder's own <page>N</page> marker always has its
# closing tag on the same line, so an opening tag followed by one is kept.
_PAGE_TAG_RE = re.compile(r"<page>(?![^<\n]*</page>)", re.IGNORECASE)
_TAG_PATTERNS = tuple(
    re.compile(r"[ \t]*(?:" + pattern.pattern + r")[ \t]*", pattern.flags)
    for pattern in (_DOCUMENT_HEADER_RE, _WHITELIST_TAG_RE, _TEXT_TAG_RE, _PAGE_TAG_RE)
)
# Placeholder for a removed tag (NUL never survives into page text: the
# builder strips XML-invalid control characters first).
_MARK = "\x00"
_MARK_RUN_RE = re.compile(r"[ \t]*(?:\x00[ \t]*)+")
# One flat character class: a nested quantifier here backtracks exponentially
# on a long dash run that ends in another character.
_RULE_LINE_RE = re.compile(r"[-= \t]+")
_MIN_RULE_CHARS = 10


@dataclass
class EdgarSgmlStripResult:
    text: str
    tags_removed: list[str] = field(default_factory=list)
    lines_removed: list[str] = field(default_factory=list)


def _is_rule_line(line: str) -> bool:
    if not _RULE_LINE_RE.fullmatch(line):
        return False
    return sum(1 for ch in line if ch in "-=") >= _MIN_RULE_CHARS


def strip_edgar_sgml_with_report(text: str) -> EdgarSgmlStripResult:
    """Strip EDGAR SGML tags and text-table rule lines; report what went."""
    result = EdgarSgmlStripResult(text=text)
    if not text:
        return result
    kept: list[str] = []
    for line in text.split("\n"):
        new_line = line
        for pattern in _TAG_PATTERNS:
            matches = [m.group(0).strip() for m in pattern.finditer(new_line)]
            if matches:
                result.tags_removed.extend(matches)
                new_line = pattern.sub(_MARK, new_line)
        if new_line != line:
            # A run of tags and the spaces around it collapse to one space, so
            # "Exhibit 99.1 <PAGE> SECTION 3.14" reads "Exhibit 99.1 SECTION 3.14".
            new_line = _MARK_RUN_RE.sub(" ", new_line).strip()
            if not new_line:
                # The line held nothing but SGML markup.
                continue
        if _is_rule_line(new_line):
            result.lines_removed.append(new_line.strip())
            continue
        kept.append(new_line)
    result.text = "\n".join(kept)
    return result


def strip_edgar_sgml(text: str) -> str:
    """Return ``text`` with EDGAR SGML tags and text-table rule lines removed."""
    return strip_edgar_sgml_with_report(text).text
