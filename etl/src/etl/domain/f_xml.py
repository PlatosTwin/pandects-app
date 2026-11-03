# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
import copy
import functools
import re
import unicodedata
import xml.dom.minidom
import xml.etree.ElementTree as ET
import uuid
from dataclasses import dataclass
from typing import Any
from datetime import date
from xml.parsers.expat import ExpatError

from bs4 import BeautifulSoup
from bs4.element import Comment, NavigableString, PageElement, PreformattedString, Tag

from etl.domain.edgar_sgml import strip_edgar_sgml
from etl.domain.g_sections import clean_article_title
from etl.domain.g_sections import clean_section_title
from etl.domain.heading_titles import article_title_number
from etl.domain.heading_titles import normalize_glued_article_title
from etl.domain.heading_titles import normalize_glued_section_title
from etl.domain.inline_tag_splits import InlineSplitEvidence
from etl.domain.page_furniture import (
    furniture_kind,
    running_title_kind,
    strip_nonbody_page_furniture,
    strip_page_furniture,
)
from etl.domain.raw_block_joins import join_raw_block_splits
from etl.domain.txt_underlines import strip_inline_underlines


@dataclass
class XMLData:
    """Data structure for XML output."""

    agreement_uuid: str
    xml: str
    version: int


@dataclass
class XMLGenerationFailure:
    """A recoverable XML generation error for one agreement."""

    agreement_uuid: str
    error: str


def get_uuid(x: str) -> str:
    """Generate a UUID5 hash from the input string."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, x))


def make_section_uuid(
    agreement_uuid: str,
    article_title_normed: str | None,
    section_title_normed: str | None,
    article_order: int | None,
    section_order: int | None,
) -> str:
    parts = [
        agreement_uuid,
        article_title_normed or "",
        section_title_normed or "",
        str(-1 if article_order is None else article_order),
        str(-1 if section_order is None else section_order),
    ]
    return get_uuid("\x1f".join(parts))


def add_metadata_nodes(
    root: ET.Element,
    agreement_uuid: str,
    filing_date: date,
    url: str,
    source_format: str,
) -> ET.Element:
    metadata = ET.SubElement(root, "metadata")
    ET.SubElement(metadata, "agreementUuid").text = agreement_uuid
    ET.SubElement(metadata, "filingDate").text = filing_date.strftime("%Y-%m-%d")
    ET.SubElement(metadata, "url").text = url
    ET.SubElement(metadata, "sourceFormat").text = source_format
    return metadata


SECTION_TAG_RE = re.compile(r"<section>(.*?)</section>", re.DOTALL)
# A heading whose own number is three or more levels deep ("11.1.4 Escrow",
# "Section 2.3.1."), anchored at the start of the title so a section that only
# mentions one ("Section 11.11. Disbursements from Section 11.1.4 Escrow
# Account") keeps its tag.
SUBSECTION_NUMBER_RE = re.compile(
    r"^\s*(?:(?:Section|SECTION|section|Sec\.|SEC\.|§)\s*)?\d+\.\d+\.\d+(?:\.\d+)*(?!\d)"
)
ARTICLE_TAG_RE = re.compile(r"<article\b")
INLINE_MARKER_RE = re.compile(r"(<pageUUID>.*?</pageUUID>|<page>.*?</page>)", re.DOTALL)
SIGNATURE_FIELD_LABEL_RE = re.compile(r"^(?P<label>By|Name|Title|Its):?$", re.IGNORECASE)
# Joins a label cell printed without a colon ("By" | "/s/ Jane Doe") to its value
# cell, so the row still reads as a field; it is served as a space.
_SIGNATURE_LABEL_CELL_JOIN = "\ue000"
SIGNATURE_FIELD_WITH_VALUE_RE = re.compile(
    r"^(?P<label>By|Name|Title|Its)(?P<sep>:\s*|\ue000)(?P<value>.+)$",
    re.IGNORECASE,
)
SIGNATURE_FIELD_NON_VALUE_RE = re.compile(
    (
        r"^(?:\[|\(|signature page\b|address:|facsimile:|email:|attention:|ein:|"
        r"aggregate\b|name in which\b|you must\b)"
    ),
    re.IGNORECASE,
)
SIGNATURE_VALUE_RE = re.compile(r"^(?:/s/|s/|_+)", re.IGNORECASE)
SIGNATURE_ENTITY_SUFFIX_RE = re.compile(
    (
        r"\b(?:INC\.?|INCORPORATED|CORP\.?|CORPORATION|LLC|L\.L\.C\.|LTD\.?|"
        r"LIMITED|COMPANY|CO\.|LP|L\.P\.|PTE\.?\s+LTD\.?|AG|S\.A\.|PLC|"
        r"HOLDINGS?)\.?,?$"
    ),
    re.IGNORECASE,
)
SIGNATURE_HTML_STRUCTURAL_TAGS = {"div", "p", "table", "tr", "td", "th"}
# Inline elements: EDGAR writes small caps as "I<small>N</small> W<small>ITNESS</small>"
# and splits words across font runs ("Na<font>me:</font>"), so an inline boundary
# between two letters is no word gap.
SIGNATURE_HTML_INLINE_TAGS = {
    "a", "abbr", "b", "big", "cite", "code", "del", "em", "font", "i", "ins",
    "kbd", "mark", "q", "s", "samp", "small", "span", "strike", "strong", "sub",
    "sup", "tt", "u", "var",
}
# An inline element styled to take its own room ("padding-right:107pt",
# "display:inline-block;width:36pt") is laid out like a separate cell. "float: none"
# and "display: inline" leave the element in the text flow.
SIGNATURE_HTML_SPACED_STYLE_RE = re.compile(
    (
        r"(?:^|;)\s*(?:padding|margin|width|position|float\s*:\s*(?:left|right)|"
        r"display\s*:\s*(?:block|inline-block|table|flex))"
    ),
    re.IGNORECASE,
)
# An entity name in caps ("CAMERON CAPITAL CORPORATION", "ACME PARTNERS, INC., as
# general partner ...") opens a signer block, so it is no field's value.
SIGNATURE_ENTITY_TOKEN_RE = re.compile(
    (
        r"(?<!\bTHE )\b(?:INC|INCORPORATED|CORP|CORPORATION|LLC|L\.L\.C|LTD|LIMITED|COMPANY|"
        r"LP|L\.P|LLP|L\.L\.P|PLC|N\.V|B\.V|S\.A|GMBH|HOLDINGS)(?![\w-])\.?"
    )
)
SIGNATURE_SENTENCE_MIN_WORDS = 12
# A value that opens with an office ("Executive Vice President, ...", "GENERAL
# PARTNER OF ACME HOLDINGS LP") is a title, not a sentence or an entity.
SIGNATURE_TITLE_LEAD_RE = re.compile(
    (
        r"^(?:(?:senior|executive|group|first|second)\s+)*(?:vice\s+)?"
        r"(?:president|chief\b|ceo\b|cfo\b|coo\b|manager\b|"
        r"managing\s+(?:member|director|partner)|general\s+partner|general\s+counsel|"
        r"sole\s+member|member\b|director\b|secretary|treasurer|trustee|authori[sz]ed\b|"
        r"attorney|chairman|chair\b|officer|partner\b|principal\b|controller)(?![\w'’])"
    ),
    re.IGNORECASE,
)
# "in his capacity as ...", "solely in its capacity as ...", "as Sellers' Representative".
SIGNATURE_CAPACITY_LEAD_RE = re.compile(
    r"^(?:(?:solely\s+)?in\s+(?:his|her|its|their)\s+capacity|as\s)",
    re.IGNORECASE,
)
# "ACME FUND GP, LLC, ITS GENERAL PARTNER": an intermediate signer is a By: value.
SIGNATURE_INTERMEDIATE_SIGNER_RE = re.compile(
    (
        r"\b(?:its|as)\s+(?:sole\s+|general\s+|managing\s+)?"
        r"(?:general\s+partner|manager|managing\s+member|sole\s+member|member|trustee|attorney)"
    ),
    re.IGNORECASE,
)
# A party-role header ("PURCHASER:", "SELLERS:") or a bare page number ("51").
SIGNATURE_NON_FIELD_VALUE_RE = re.compile(r"[A-Z][A-Z'’&,. ()-]{1,40}:|\d{1,3}")
_SIGNATURE_SOFT_BREAK = "\x00"
# A soft break between two printed characters (whitespace on either side already
# makes the gap).
_SIGNATURE_INLINE_JOIN_RE = re.compile(r"(?<=(\S))\x00+(?=(\S))")
_SIGNATURE_CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯豈-﫿]")


def _signature_letters_join(left: str, right: str) -> str:
    """Join two letters of the same script; keep the gap at a Latin/CJK change."""
    left_cjk = _SIGNATURE_CJK_RE.match(left) is not None
    right_cjk = _SIGNATURE_CJK_RE.match(right) is not None
    return "" if left_cjk == right_cjk else _SIGNATURE_SOFT_BREAK


def _signature_inline_join(match: re.Match[str]) -> str:
    """Join text across an inline-element boundary the way a browser renders it.

    An inline boundary adds no space: "/<small>S</small>/" reads "/S/",
    "BRISTOL<font>-MYERS</font>" reads "BRISTOL-MYERS", "STOCKHOLDERS<font>’</font>"
    reads "STOCKHOLDERS’", "10.<font>9</font>" reads "10.9". Letters in any script
    ("Fran<font>çois</font>") join, except at a Latin/CJK change. Punctuation joins
    its neighbor, except that a label's colon keeps its gap ("Name:<font>Jane
    Doe</font>") and an underscore blank stays apart from the word beside it
    ("DiModica<u>____</u>"). A digit beside a letter or digit keeps the gap, and so
    does a conformed "/s/" before the signer's name ("/s/<u>Jane Doe</u>"), which the
    page text prints as "/s/ Jane Doe".
    """
    left, right = match.group(1), match.group(2)
    if left == "/" and right.isalpha():
        before = match.string[max(match.start(1) - 12, 0):match.start(1) + 1].replace(_SIGNATURE_SOFT_BREAK, "")
        if before[-3:].lower() == "/s/" and not before[-4:-3].isalnum():
            return _SIGNATURE_SOFT_BREAK
    if left.isalpha() and right.isalpha():
        return _signature_letters_join(left, right)
    if left == ":":
        return _SIGNATURE_SOFT_BREAK
    if (left == "_" and right.isalnum()) or (right == "_" and left.isalnum()):
        return _SIGNATURE_SOFT_BREAK
    if unicodedata.category(left)[0] in "PS" or unicodedata.category(right)[0] in "PS":
        return ""
    return _SIGNATURE_SOFT_BREAK


# EDGAR navigation links such as <a href="#toc">BACK</a> at the top of a page.
SIGNATURE_HTML_NAV_LINK_TEXT_RE = re.compile(r"(?:back|return\s+to\s+top)", re.IGNORECASE)
XML_INVALID_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def strip_subsection_section_tags(tagged_text: str) -> str:
    """
    Remove <section> tags from subsection headings like 1.2.3 or 1.2.3.4.

    Only a title that starts with such a number is unwrapped; a section title
    that merely mentions one ("Section 11.11. Disbursements from Section
    11.1.4 Escrow Account") keeps its tag.

    Args:
        tagged_text: Tagged text that may include <section> headings.

    Returns:
        Tagged text with subsection <section> tags removed.
    """

    def replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        if SUBSECTION_NUMBER_RE.match(inner):
            return inner
        return match.group(0)

    return SECTION_TAG_RE.sub(replace, tagged_text)


def count_article_tags(tagged_text: str) -> int:
    """Count <article> tags in tagged text."""
    return len(ARTICLE_TAG_RE.findall(tagged_text))


def _iter_line_fragments(text_block: str) -> list[str]:
    fragments: list[str] = []
    for line in text_block.splitlines():
        stripped_line = line.strip()
        if not stripped_line:
            continue
        parts = INLINE_MARKER_RE.split(stripped_line)
        for part in parts:
            if part and part.strip():
                fragments.append(part.strip())
    return fragments


def _strip_xml_invalid_control_chars(text: str) -> str:
    return XML_INVALID_CONTROL_CHAR_RE.sub("", text)


def _normalize_toc_cell_text(text: str) -> str:
    text = _strip_xml_invalid_control_chars(text)
    text = text.replace("\u00a0", " ").replace("\xa0", " ")
    text = re.sub(r"[\u200B\u200C\u200D\u200E\u200F\u202A-\u202E\u2060\uFEFF]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _toc_visible_text(tag: Tag) -> str:
    return _normalize_toc_cell_text(tag.get_text(separator=" ", strip=False))


def _signature_is_inline(tag: Tag) -> bool:
    if tag.name not in SIGNATURE_HTML_INLINE_TAGS:
        return False
    return SIGNATURE_HTML_SPACED_STYLE_RE.search(str(tag.get("style") or "")) is None


def _signature_raw_text(node: PageElement, parts: list[str]) -> None:
    """Append a node's text, with a gap around block elements and line breaks
    and a soft break around inline elements."""
    if isinstance(node, NavigableString):
        # Comments, doctypes, CDATA and processing instructions are not visible.
        if not isinstance(node, PreformattedString):
            parts.append(str(node))
        return
    if not isinstance(node, Tag):
        return
    boundary = _SIGNATURE_SOFT_BREAK if _signature_is_inline(node) else " "
    parts.append(boundary)
    for child in node.children:
        _signature_raw_text(child, parts)
    parts.append(boundary)


def _signature_nodes_text(nodes: list[PageElement]) -> str:
    """Visible text of nodes; a soft break between letters joins them and is a space elsewhere."""
    parts: list[str] = []
    for node in nodes:
        _signature_raw_text(node, parts)
    text = _SIGNATURE_INLINE_JOIN_RE.sub(_signature_inline_join, "".join(parts))
    return _normalize_toc_cell_text(text.replace(_SIGNATURE_SOFT_BREAK, " "))


def _signature_visible_text(tag: Tag) -> str:
    return _signature_nodes_text([tag])


def _split_toc_page_label(text: str) -> list[str]:
    normalized = _normalize_toc_cell_text(text)
    if not normalized:
        return []

    prefix_match = re.match(
        r"^(?P<head>(?:TABLE OF CONTENTS|Table of Contents|CONTENTS|Contents)(?:\s+\([^)]+\))?)\s+Page\s+(?P<rest>.+)$",
        normalized,
        flags=re.IGNORECASE,
    )
    if prefix_match:
        return [
            prefix_match.group("head").strip(),
            "Page",
            prefix_match.group("rest").strip(),
        ]

    if not normalized.endswith(" Page"):
        return [normalized]

    heading = normalized[: -len(" Page")].strip()
    if not heading:
        return [normalized]

    heading_like = (
        re.search(r"\bTABLE OF CONTENTS\b", heading, flags=re.IGNORECASE)
        or re.fullmatch(r"CONTENTS(?:\s+\(CONTINUED\))?", heading, flags=re.IGNORECASE)
        or re.fullmatch(
            r"TABLE OF CONTENTS(?:\s+\(CONTINUED\))?",
            heading,
            flags=re.IGNORECASE,
        )
    )
    if heading_like:
        return [heading, "Page"]

    return [normalized]


def _append_toc_text_block(blocks: list[str], text: str) -> None:
    blocks.extend(_split_toc_page_label(text))


def _remove_toc_nonvisible_nodes(soup: BeautifulSoup) -> None:
    for comment in soup.find_all(string=lambda node: isinstance(node, Comment)):
        _ = comment.extract()
    for tag in soup.find_all(["script", "style", "noscript", "template"]):
        tag.decompose()
    for tag in list(soup.find_all(True)):
        if tag.decomposed:
            continue
        style = str(tag.get("style") or "").lower().replace(" ", "")
        hidden = (
            tag.has_attr("hidden")
            or str(tag.get("aria-hidden") or "").lower() == "true"
            or "display:none" in style
            or "visibility:hidden" in style
        )
        if hidden:
            tag.decompose()


def _format_toc_flat_row(text: str, *, line_width: int) -> str:
    text = _normalize_toc_cell_text(text)
    match = re.match(r"^(.*?)(?:\s+)(\d+|[ivxlcdm]+)$", text, flags=re.IGNORECASE)
    if not match:
        return text
    left = match.group(1).strip()
    page = match.group(2).strip()
    left = re.sub(
        r"^(Section\s+\d+(?:\.\d+)*)\s+",
        lambda m: m.group(1).ljust(14),
        left,
    )
    if len(left) + 2 + len(page) >= line_width:
        return f"{left}  {page}"
    return f"{left}{' ' * (line_width - len(left) - len(page))}{page}"


def _format_toc_table_row(cells: list[Tag], *, line_width: int) -> str | None:
    parts = [_toc_visible_text(cell) for cell in cells]
    parts = [part for part in parts if part]
    if not parts:
        return None

    if len(parts) == 1:
        return parts[0]

    page = parts[-1] if re.fullmatch(r"[ivxlcdmIVXLCDM]+|\d+[A-Za-z]?", parts[-1]) else ""
    left_parts = parts[:-1] if page else parts

    if len(left_parts) >= 2 and re.fullmatch(r"\d+(?:\.\d+)*[A-Za-z]?", left_parts[0]):
        prefix = left_parts[0].ljust(8)
        title = " ".join(left_parts[1:])
    else:
        prefix = ""
        title = " ".join(left_parts)

    left = f"{prefix}{title}".rstrip()
    if not page:
        return left

    if len(left) + 2 + len(page) >= line_width:
        return f"{left}  {page}"
    return f"{left}{' ' * (line_width - len(left) - len(page))}{page}"


def _toc_direct_block_texts(cell: Tag) -> list[str]:
    blocks: list[str] = []
    for child in cell.find_all(["div", "p"], recursive=False):
        text = _toc_visible_text(child)
        if text:
            blocks.append(text)
    if blocks:
        return blocks
    text = _toc_visible_text(cell)
    return [text] if text else []


def _format_toc_parallel_cell_rows(cells: list[Tag], *, line_width: int) -> list[str]:
    if len(cells) < 2:
        return []

    page_blocks = _toc_direct_block_texts(cells[-1])
    if len(page_blocks) <= 1:
        return []
    if not page_blocks or not all(
        re.fullmatch(r"\d+|[ivxlcdm]+", page, flags=re.IGNORECASE)
        for page in page_blocks
    ):
        return []

    left_blocks: list[str] = []
    for cell in cells[:-1]:
        left_blocks.extend(_toc_direct_block_texts(cell))

    if len(left_blocks) <= len(page_blocks):
        return []

    pages = [""] * (len(left_blocks) - len(page_blocks)) + page_blocks
    lines: list[str] = []
    for left, page in zip(left_blocks, pages):
        if page:
            lines.append(_format_toc_flat_row(f"{left} {page}", line_width=line_width))
        else:
            lines.append(left)
    return lines


def _without_nested_table_rows(cell: Tag, table: Tag) -> Tag:
    """A cell without the rows of its own table that are nested inside it.

    html.parser does not close an open <td> when the next <tr> starts, so the
    rows after an unclosed cell become its descendants. The table loops render
    those rows on their own, so the cell's text must leave them out.
    """
    if not any(tr.find_parent("table") is table for tr in cell.find_all("tr")):
        return cell
    trimmed = copy.copy(cell)
    for tr in trimmed.find_all("tr"):
        # The copy is detached, so a row of the cell's own table has no table above it.
        if not tr.decomposed and tr.find_parent("table") is None:
            tr.decompose()
    return trimmed


def _table_row_cells(tr: Tag, table: Tag) -> list[Tag]:
    return [
        _without_nested_table_rows(cell, table)
        for cell in tr.find_all(["td", "th"], recursive=False)
    ]


def _toc_table_to_lines(table: Tag, *, line_width: int) -> list[str]:
    lines: list[str] = []
    for tr in table.find_all("tr"):
        cells = _table_row_cells(tr, table)
        if not cells:
            continue
        parallel_lines = _format_toc_parallel_cell_rows(cells, line_width=line_width)
        if parallel_lines:
            lines.extend(parallel_lines)
            continue
        row = _format_toc_table_row(cells, line_width=line_width)
        if row:
            lines.append(row)
    return lines


def _toc_style_map(tag: Tag) -> dict[str, str]:
    parsed: dict[str, str] = {}
    style = str(tag.get("style") or "")
    for part in style.split(";"):
        if ":" not in part:
            continue
        key, value = part.split(":", 1)
        parsed[key.strip().lower()] = value.strip().lower()
    return parsed


def _looks_like_positioned_toc_row(tag: Tag) -> bool:
    if tag.name != "div":
        return False
    styles = _toc_style_map(tag)
    if styles.get("overflow") != "hidden":
        return False
    if styles.get("position") != "relative":
        return False
    text = _toc_visible_text(tag)
    if not re.search(r"\s(?:\d+|[ivxlcdm]+)$", text, flags=re.IGNORECASE):
        return False
    return bool(
        re.match(
            r"^(ARTICLE|Section|\d+(?:\.\d+)*)\b",
            text,
            flags=re.IGNORECASE,
        )
    )


def _positioned_toc_rows(soup: BeautifulSoup, *, line_width: int) -> list[tuple[Tag, list[str]]]:
    """Each positioned TOC row with its line; a repeated row keeps no line of its own."""
    rows: list[tuple[Tag, list[str]]] = []
    seen: set[str] = set()
    for row in soup.find_all(_looks_like_positioned_toc_row):
        text = _toc_visible_text(row)
        if text in seen:
            rows.append((row, []))
            continue
        seen.add(text)
        rows.append((row, [_format_toc_flat_row(text, line_width=line_width)]))
    return rows


def _toc_heading_blocks(soup: BeautifulSoup) -> list[str]:
    blocks: list[str] = []
    for p in soup.find_all("p"):
        text = _toc_visible_text(p)
        if not text:
            continue
        split_text = _split_toc_page_label(text)
        if split_text and re.fullmatch(
            r"Table of Contents|TABLE OF CONTENTS|Contents|CONTENTS",
            split_text[0],
        ):
            blocks.extend(split_text[:2])
        elif text == "Page":
            _append_toc_text_block(blocks, text)
        if len(blocks) >= 2:
            break
    return blocks


def _content_toc_tables(soup: BeautifulSoup, *, line_width: int) -> list[tuple[Tag, list[str]]]:
    """The tables that hold TOC entries, each with its lines."""
    content_tables: list[tuple[Tag, list[str]]] = []
    for table in soup.find_all("table"):
        lines = _toc_table_to_lines(table, line_width=line_width)
        if not lines:
            continue
        entry_lines = [
            line
            for line in lines
            if re.search(
                r"\b(?:ARTICLE|Section|SECTION|\d+(?:\.\d+)+)\b",
                line,
                flags=re.IGNORECASE,
            )
        ]
        reference_lines = [
            line
            for line in lines
            if re.search(
                r"\s(?:\d+|Preamble|Recitals|Section\s+\d|Article\s+\d)$",
                line,
                flags=re.IGNORECASE,
            )
        ]
        if len(entry_lines) >= 3 or (len(lines) >= 5 and len(reference_lines) >= 3):
            content_tables.append((table, lines))
    return content_tables


# Blocks that hold printed TOC text outside the entry tables or rows: the exhibit
# and schedule lists and index headings printed after the last entry, and article
# headings printed between entry tables.
_TOC_TEXT_BLOCK_TAGS = (
    "p", "div", "li", "h1", "h2", "h3", "h4", "h5", "h6", "center", "blockquote", "pre", "dt", "dd",
)
_TOC_ENTRY_START_RE = re.compile(r"(?:ARTICLE|Article|SECTION|Section)\s+(?:\d+|[IVXLC]+)\b")
# Roman page numbers: "iii", "- iv -", "ii.", "II".
_TOC_ROMAN_PAGE_RE = re.compile(r"[-\u2013\u2014(\s]*(?:[ivxlc]{1,7}|[IVXLC]{1,7})\s*\.?[-\u2013\u2014)\s]*")
_TOC_LABEL_RE = re.compile(
    r"(?:(?:table\s+of\s+)?contents\s*)?(?:\(?continued\)?\s*)?(?:page|pages|page\s+no\.?)?",
    re.IGNORECASE,
)
# Column headings alone: "Section Description", "Exhibit Title Page".
_TOC_COLUMN_HEADER_RE = re.compile(
    r"(?:(?:sections?|articles?|exhibits?|schedules?|annex|description|title|page|no\.?|number|document|tab)\s*){2,4}",
    re.IGNORECASE,
)
# A page label at the end of a footer: "Ref: SLRB 1136588 Page 3", "3.5.2022page (ii)",
# "Page | ii", "Page ii of iv"; "(CONTINUED ON NEXT PAGE)" and "[Remainder of page intentionally left blank]".
_TOC_PAGE_LABEL_END_RE = re.compile(
    r"(?<![A-Za-z])page\s*[|:\-\u2013]*\s*\(?(?:\d{1,3}|[ivxlc]{1,6})\)?(?:\s+of\s+(?:\d{1,3}|[ivxlc]{1,6}))?\s*$"
    r"|^\(?continued\s+on\s+(?:the\s+)?next\s+page\)?$"
    r"|^\[?\s*(?:the\s+)?remainder\s+of\s+(?:this\s+)?page\b",
    re.IGNORECASE,
)
_TOC_LONG_NUMBER_RE = re.compile(r"\d{5,}")
# Doc IDs inside a line ("#4854-0942-8064v43", "SL1 3716757v12") and Word field codes.
_TOC_DOC_ID_RE = re.compile(
    r"\d{4}-\d{4}-\d{4}|\d{5,}\s?v\.?\s?\d{1,4}\b|DOCVARIABLE|DOCPROPERTY|SWDocID|MERGEFORMAT"
)
# Exhibit page labels ("A-ii", "E-B1") and lines without a letter ("()", "________", "2 84").
_TOC_LABELED_PAGE_RE = re.compile(r"[A-Z]{1,2}-(?:[ivxlcIVXLC]{1,6}|[A-Z]?\d{1,3})|[^A-Za-z]+")
# The note that the TOC is not part of the agreement ("1 The Table of Contents is not a part of this Agreement.").
_TOC_EXECUTION_BANNER_RE = re.compile(r"\bexecution\s+(?:version|copy)\b", re.IGNORECASE)
# A trailing page label, left off when a running header or footer is compared across pages
# or read as a running title: " iii", " 12", " -ii-", " (iv)", " A-3", "-i".
_TOC_TRAILING_PAGE_RE = re.compile(
    r"(?:\s+|(?<=[A-Za-z])-)[-\u2013\u2014(]*\s*(?:[ivxlc]{1,7}|[IVXLC]{1,7}|\d{1,3}|[A-Z]{1,2}-\d{1,3})\s*[-\u2013\u2014).]*$"
)
# A TOC page label: "ToC-i", "Contents - 3", "Index - iv", "A- ii", "2 of 42", "P AGE 3", "i v".
_TOC_PAGE_TAG_RE = re.compile(
    r"(?:toc|contents?|index|[A-Z]{1,2})?\s*[-\u2013]\s*(?:[ivxlc]{1,7}|\d{1,3})"
    r"|(?:toc|contents?)\s*(?:[ivxlc]{1,7}|\d{1,3})"
    r"|(?:page\s+)?(?:[ivxlc]{1,7}|\d{1,3})\s+of\s+(?:[ivxlc]{1,7}|\d{1,3})"
    r"|p\s?a\s?g\s?e\s*[-:|]?\s*(?:[ivxlc]{1,7}|\d{1,3})"
    r"|(?:[ivxlc]\s){1,6}[ivxlc]",
    re.IGNORECASE,
)
# A page label that marks a running title: " iii", " -ii-", " (iv)", "-i". A plain arabic number
# or an "A-3" label is not one: "Credit Agreement 75" and "Merger Agreement A-3" are index rows.
_TOC_TITLE_PAGE_RE = re.compile(
    r"(?:\s+|(?<=[A-Za-z])-)(?:[-\u2013\u2014(]+\s*(?:[ivxlc]{1,7}|[IVXLC]{1,7}|\d{1,3})\s*[-\u2013\u2014).]*"
    r"|[ivxlc]{1,7}\.?)$"
)
# A leading page label: "(i) SHARE PURCHASE AGREEMENT".
_TOC_LEADING_PAGE_RE = re.compile(r"^\((?:[ivxlc]{1,7}|[IVXLC]{1,7}|\d{1,3})\)\s+")
# An agreement preamble that starts on a TOC-labeled page; the page's TOC ends before it.
# Not applied to a line that opens with a list label ("Exhibit B Form of ..., by and among ...").
_TOC_PREAMBLE_RE = re.compile(
    r"^\s*THIS\b.{0,160}\bAGREEMENT\b|\bby\s+and\s+(?:among|between)\b|\bis\s+(?:made|entered\s+into)\s+(?:as\s+of|by|on)\b",
)
_TOC_NOT_PART_RE = re.compile(r"(?:table\s+of\s+contents|headings).{0,40}\bnot\s+(?:a\s+)?part\s+of\s+this\s+agreement", re.IGNORECASE)
# A paragraph of prose, not a TOC line; the page's TOC ends before it.
_TOC_PROSE_CHARS = 300
# A word, for telling a doc ID from a line of text: letters only, not a roman numeral or "page".
_TOC_WORD_TOKEN_RE = re.compile(r"[A-Za-z]{3,}[.,;:]?")
_TOC_ROMAN_TOKEN_RE = re.compile(r"[ivxlcIVXLC]+[.,;:]?")


def _toc_word_count(text: str) -> int:
    return sum(
        1
        for token in text.split()
        if _TOC_WORD_TOKEN_RE.fullmatch(token)
        and _TOC_ROMAN_TOKEN_RE.fullmatch(token) is None
        and token.rstrip(".,;:").lower() != "page"
    )
# SEC filing legends and confidential-treatment notes; the lines after one on the page continue it.
_TOC_LEGEND_RE = re.compile(
    r"\bconfidential\b|\bomitted\b|\bomissions?\b|rule\s+24b-2|regulation\s+s-k|item\s+601"
    r"|securities\s+and\s+exchange\s+commission|\[\s*\*+\s*\]|^\*|\bcopyright\b",
    re.IGNORECASE,
)
# A listed exhibit, schedule or annex: "Exhibit A ...", "Annex B ...", "A-1 Form of ...".
_TOC_LIST_ITEM_RE = re.compile(
    r"(?:exhibits?|schedules?|annex(?:es)?|appendix|appendices|attachments?)\b|[A-Z](?:-\d{1,2})?\s",
    re.IGNORECASE,
)


def _toc_extra_is_furniture(text: str) -> bool:
    """A page number, a doc ID, a banner, a running title, or a "TABLE OF CONTENTS"/"Page" label."""
    if (
        furniture_kind(text) is not None
        or _TOC_ROMAN_PAGE_RE.fullmatch(text) is not None
        or _TOC_LABEL_RE.fullmatch(text) is not None
        or _TOC_COLUMN_HEADER_RE.fullmatch(text) is not None
        or _TOC_PAGE_LABEL_END_RE.search(text) is not None
        or _TOC_LABELED_PAGE_RE.fullmatch(text) is not None
        or _TOC_DOC_ID_RE.search(text) is not None
        or _TOC_NOT_PART_RE.search(text) is not None
        or _TOC_EXECUTION_BANNER_RE.search(text) is not None
        or _TOC_PAGE_TAG_RE.fullmatch(text) is not None
    ):
        return True
    if _TOC_LONG_NUMBER_RE.search(text) is not None and _toc_word_count(text) < 2:
        return True
    if (
        _TOC_LIST_ITEM_RE.match(text) is not None
        or _TOC_ENTRY_START_RE.match(text) is not None
        or "form of" in text.lower()
    ):
        return False
    # A running title, with or without a page label ("... Agreement -ii-", "... AGREEMENT iii"),
    # or as one part of a piped footer ("Share sale agreement | Slinger Bag, Inc.").
    if running_title_kind(text) == "title":
        return True
    titled = _TOC_TITLE_PAGE_RE.sub("", text).strip()
    if titled != text and _toc_word_count(titled) >= 2 and running_title_kind(titled) == "title":
        return True
    # A footer that names the deal before a dash ("Project Grease - share purchase agreement 1").
    tail = re.split(r"\s[-\u2013\u2014]\s", _TOC_TRAILING_PAGE_RE.sub("", text).strip())
    if len(tail) > 1 and running_title_kind(tail[-1].strip()) == "title":
        return True
    return "|" in text and any(running_title_kind(part.strip()) == "title" for part in text.split("|"))


def _toc_strip_page_label(line: str) -> str:
    """The line without a leading "(i)" or trailing page label, when at least two words are left."""
    stripped = _TOC_TRAILING_PAGE_RE.sub("", _TOC_LEADING_PAGE_RE.sub("", line)).strip()
    return stripped if _toc_word_count(stripped) >= 2 else line


def toc_repeat_key(line: str) -> str:
    """A TOC line without its trailing page label, for finding running headers and footers."""
    return _toc_strip_page_label(_normalize_toc_cell_text(line)).casefold()


def _toc_candidate_blocks(
    soup: BeautifulSoup,
    entry_elements: list[Tag],
    *,
    line_width: int,
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """The text blocks around the TOC entry elements, in document order, unfiltered.

    Returns (position, lines) pairs, where position is the number of entry
    elements that come before the block. Only leaf text blocks and tables
    outside every table and entry element count.
    """
    entry_ids = {id(el) for el in entry_elements}
    candidates = soup.find_all([*_TOC_TEXT_BLOCK_TAGS, "table"])
    # Elements inside an entry or a table; elements that hold an entry or a block.
    inside: set[int] = set()
    for el in (*entry_elements, *soup.find_all("table")):
        inside.update(id(d) for d in el.descendants if isinstance(d, Tag))
    holds_entry: set[int] = set()
    for el in entry_elements:
        holds_entry.update(id(parent) for parent in el.parents)
    holds_block: set[int] = set()
    for el in candidates:
        for parent in el.parents:
            if id(parent) in holds_block:
                break
            holds_block.add(id(parent))
    blocks: list[tuple[int, tuple[str, ...]]] = []
    position = 0
    for el in candidates:
        if id(el) in entry_ids:
            position += 1
            continue
        if id(el) in inside or id(el) in holds_entry:
            continue
        if el.name == "table":
            lines = _toc_table_to_lines(el, line_width=line_width)
        else:
            if id(el) in holds_block:
                continue
            text = _toc_visible_text(el)
            lines = [text] if text else []
        if lines:
            blocks.append((position, tuple(lines)))
    return tuple(blocks)


def _select_toc_extras(
    candidates: tuple[tuple[int, tuple[str, ...]], ...],
    drop_keys: frozenset[str] = frozenset(),
) -> list[tuple[int, list[str]]]:
    """The printed extras among the candidate blocks.

    Furniture lines and lines whose repeat key is in drop_keys are left out. A
    filing legend, a paragraph of prose or an agreement preamble after the first
    entry element ends the extras of the page (the lines after a legend continue
    it), and before the first entry element a block counts only from the first
    one that opens with an ARTICLE or Section heading.
    """
    extras: list[tuple[int, list[str]]] = []
    leading_open = False
    for position, block_lines in candidates:
        lines = [
            line
            for line in block_lines
            if line.strip()
            and not _toc_extra_is_furniture(line.strip())
            and toc_repeat_key(line) not in drop_keys
        ]
        if not lines:
            continue
        if any(
            _TOC_LEGEND_RE.search(line) is not None
            or len(line) > _TOC_PROSE_CHARS
            or (_TOC_PREAMBLE_RE.search(line) is not None and _TOC_LIST_ITEM_RE.match(line) is None)
            for line in lines
        ):
            if position == 0:
                continue
            break
        if position == 0 and not leading_open:
            if _TOC_ENTRY_START_RE.match(lines[0]) is None:
                continue
            leading_open = True
        extras.append((position, lines))
    return extras


def _merge_toc_entries_and_extras(
    entries: list[list[str]],
    extras: list[tuple[int, list[str]]],
    *,
    separate_entries: bool,
) -> list[str]:
    """Entry lines and printed extras in document order."""
    merged: list[str] = []
    extra_index = 0
    for entry_index, lines in enumerate(entries + [[]]):
        while extra_index < len(extras) and extras[extra_index][0] <= entry_index:
            merged.extend(extras[extra_index][1])
            extra_index += 1
        if entry_index == len(entries):
            break
        if separate_entries and entry_index and lines:
            merged.append("")
        merged.extend(lines)
    return merged


def _wrap_toc_long_lines(text: str, *, line_width: int) -> str:
    import textwrap

    wrapped: list[str] = []
    for line in text.splitlines():
        item_splits = re.split(
            r"\s+(?=(?:Exhibit|Schedule)\s+[A-Z0-9])",
            line,
            flags=re.IGNORECASE,
        )
        if len(item_splits) >= 3:
            for item in item_splits:
                wrapped.extend(
                    textwrap.wrap(
                        item,
                        width=line_width,
                        subsequent_indent="    ",
                        break_long_words=False,
                        break_on_hyphens=False,
                    )
                )
            continue
        if len(line) <= line_width:
            wrapped.append(line)
            continue
        if len(line) <= 180 and re.search(
            r"\s(?:\d+|[ivxlcdm]+)$", line, flags=re.IGNORECASE
        ):
            wrapped.append(line)
            continue
        wrapped.extend(
            textwrap.wrap(
                line,
                width=line_width,
                subsequent_indent="    ",
                break_long_words=False,
                break_on_hyphens=False,
            )
        )
    return "\n".join(wrapped)


@dataclass(frozen=True)
class _TocPage:
    """A TOC page parsed once: its entries and the candidate blocks around them,
    or, for a page without entries, its finished text."""

    entries: tuple[tuple[str, ...], ...]
    separate_entries: bool
    heading_blocks: tuple[str, ...]
    candidates: tuple[tuple[int, tuple[str, ...]], ...]
    text: str | None


@functools.lru_cache(maxsize=128)
def _parse_toc_page(raw_html: str, line_width: int) -> _TocPage:
    soup = BeautifulSoup(raw_html, "html.parser")
    _remove_toc_nonvisible_nodes(soup)

    positioned_rows = _positioned_toc_rows(soup, line_width=line_width)
    if any(lines for _, lines in positioned_rows):
        return _TocPage(
            entries=tuple(tuple(lines) for _, lines in positioned_rows),
            separate_entries=False,
            heading_blocks=tuple(_toc_heading_blocks(soup)),
            candidates=_toc_candidate_blocks(soup, [row for row, _ in positioned_rows], line_width=line_width),
            text=None,
        )

    content_tables = _content_toc_tables(soup, line_width=line_width)
    if content_tables:
        return _TocPage(
            entries=tuple(tuple(lines) for _, lines in content_tables),
            separate_entries=True,
            heading_blocks=tuple(_toc_heading_blocks(soup)),
            candidates=_toc_candidate_blocks(soup, [table for table, _ in content_tables], line_width=line_width),
            text=None,
        )

    return _TocPage(entries=(), separate_entries=False, heading_blocks=(), candidates=(),
                    text=_toc_fallback_text(soup, line_width=line_width))


def toc_page_extra_lines(raw_html: str, *, line_width: int = 120) -> list[str]:
    """The printed extras format_toc_html_like_screen would add to this TOC page."""
    page = _parse_toc_page(raw_html, line_width)
    return [line for _, lines in _select_toc_extras(page.candidates) for line in lines]


def repeated_toc_extra_keys(raw_pages: list[str]) -> frozenset[str]:
    """Keys of extras printed on two or more TOC pages: running headers and footers."""
    counts: dict[str, int] = {}
    for raw_html in raw_pages:
        for key in {toc_repeat_key(line) for line in toc_page_extra_lines(raw_html)}:
            counts[key] = counts.get(key, 0) + 1
    return frozenset(key for key, count in counts.items() if count >= 2 and key)


def format_toc_html_like_screen(
    raw_html: str, *, line_width: int = 120, drop_keys: frozenset[str] = frozenset()
) -> str:
    page = _parse_toc_page(raw_html, line_width)
    if page.text is not None:
        return page.text
    lines = _merge_toc_entries_and_extras(
        [list(lines) for lines in page.entries],
        _select_toc_extras(page.candidates, drop_keys),
        separate_entries=page.separate_entries,
    )
    blocks = [*page.heading_blocks, "\n".join(lines)]
    return _wrap_toc_long_lines("\n\n".join(blocks).strip(), line_width=line_width)


def _toc_fallback_text(soup: BeautifulSoup, *, line_width: int) -> str:
    blocks: list[str] = []
    for child in (soup.body or soup).find_all(recursive=False):
        if child.name == "table":
            table_lines = _toc_table_to_lines(child, line_width=line_width)
            if table_lines:
                blocks.append("\n".join(table_lines))
            continue
        if child.find("table"):
            for descendant in child.find_all(["div", "p", "table"], recursive=False):
                if descendant.name == "table":
                    table_lines = _toc_table_to_lines(descendant, line_width=line_width)
                    if table_lines:
                        blocks.append("\n".join(table_lines))
                else:
                    text = _toc_visible_text(descendant)
                    if text:
                        _append_toc_text_block(blocks, text)
            continue
        text = _toc_visible_text(child)
        if text:
            _append_toc_text_block(blocks, text)

    if not blocks:
        text = soup.get_text(separator="\n", strip=False)
        text = "\n".join(_normalize_toc_cell_text(line) for line in text.splitlines())
        return _wrap_toc_long_lines(
            re.sub(r"\n{3,}", "\n\n", text).strip(),
            line_width=line_width,
        )

    return _wrap_toc_long_lines("\n\n".join(blocks).strip(), line_width=line_width)


def _signature_text_lines(text: str) -> list[str]:
    text = _strip_xml_invalid_control_chars(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200B\u200C\u200D\u200E\u200F\u202A-\u202E\u2060\uFEFF]", "", text)
    return [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.splitlines()]


def _normalize_signature_punctuation_spacing(text: str) -> str:
    text = re.sub(r"[^\S\r\n]+([,.;:)\]])", r"\1", text)
    return re.sub(r"([(\[])[^\S\r\n]+", r"\1", text)


def _signature_table_row_text(tr: Tag, table: Tag) -> str | None:
    cells = _table_row_cells(tr, table)
    if not cells:
        return None

    parts = [_signature_visible_text(cell) for cell in cells]
    parts = [part for part in parts if part]
    if not parts:
        return ""

    # A label cell ("By", "Name:") keeps its printed case and colon. One printed
    # without a colon is joined so the row still reads as a field.
    label_match = SIGNATURE_FIELD_LABEL_RE.fullmatch(parts[0])
    if label_match is not None and len(parts) > 1 and not parts[0].endswith(":"):
        return f"{parts[0]}{_SIGNATURE_LABEL_CELL_JOIN}{' '.join(parts[1:])}"
    return " ".join(parts)


def _signature_table_to_lines(table: Tag) -> list[str]:
    lines: list[str] = []
    for tr in table.find_all("tr"):
        row_text = _signature_table_row_text(tr, table)
        if row_text is None:
            continue
        lines.append(row_text)
    return lines


def _signature_html_blocks(node: Tag) -> list[str]:
    """Text blocks of a node, in order.

    Loose text and inline elements that sit directly in a container next to its
    tables or paragraphs form a block of their own. EDGAR pages often leave a
    <p> unclosed before a <table>, so html.parser nests the table inside the
    paragraph, and the opening "IN WITNESS WHEREOF" sentence is loose text in
    that <p>.
    """
    blocks: list[str] = []
    inline_run: list[PageElement] = []

    def flush_inline_run() -> None:
        if inline_run:
            text = _signature_nodes_text(inline_run)
            if text:
                blocks.append(text)
            inline_run.clear()

    for child in node.children:
        if not isinstance(child, Tag):
            if isinstance(child, NavigableString) and not isinstance(child, PreformattedString):
                inline_run.append(child)
            continue

        if child.name == "table":
            flush_inline_run()
            table_lines = _signature_table_to_lines(child)
            if table_lines:
                blocks.append("\n".join(table_lines))
            continue

        has_structural_children = any(
            grandchild.name in SIGNATURE_HTML_STRUCTURAL_TAGS
            for grandchild in child.find_all(recursive=False)
        )
        if _signature_is_inline(child) and not has_structural_children:
            inline_run.append(child)
            continue

        flush_inline_run()
        if has_structural_children:
            blocks.extend(_signature_html_blocks(child))
            continue

        text = _signature_visible_text(child)
        if text:
            blocks.append(text)
    flush_inline_run()
    return blocks


def _remove_signature_nav_links(soup: BeautifulSoup) -> None:
    """Drop internal navigation links whose only text is BACK, Back or Return to top."""
    for link in list(soup.find_all("a")):
        if link.decomposed:
            continue
        href = str(link.get("href") or "").strip()
        if not href.startswith("#"):
            continue
        text = _normalize_toc_cell_text(link.get_text(separator=" ", strip=False))
        if SIGNATURE_HTML_NAV_LINK_TEXT_RE.fullmatch(text):
            link.decompose()


def format_signature_html_like_screen(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    _remove_toc_nonvisible_nodes(soup)
    _remove_signature_nav_links(soup)

    blocks = _signature_html_blocks(soup.body or soup)
    if not blocks:
        text = soup.get_text(separator="\n", strip=False)
        text = "\n".join(_normalize_toc_cell_text(line) for line in text.splitlines())
        normalized_text = _normalize_signature_punctuation_spacing(
            re.sub(r"\n{3,}", "\n\n", text).strip()
        )
        return format_signature_text_like_screen(normalized_text)

    normalized_blocks = _normalize_signature_punctuation_spacing(
        "\n\n".join(blocks).strip()
    )
    return format_signature_text_like_screen(normalized_blocks)


def _looks_like_signature_entity_fragment(line: str) -> bool:
    if (
        not line
        or SIGNATURE_FIELD_LABEL_RE.fullmatch(line)
        or SIGNATURE_FIELD_WITH_VALUE_RE.match(line)
    ):
        return False
    letters = [ch for ch in line if ch.isalpha()]
    if not letters:
        return False
    uppercase_ratio = sum(1 for ch in letters if ch.isupper()) / len(letters)
    return uppercase_ratio >= 0.65


def _should_join_signature_entity_lines(left: str, right: str) -> bool:
    if not _looks_like_signature_entity_fragment(left):
        return False
    if not _looks_like_signature_entity_fragment(right):
        return False
    if SIGNATURE_ENTITY_SUFFIX_RE.search(left):
        return False
    return bool(SIGNATURE_ENTITY_SUFFIX_RE.search(right))


def _signature_starts_entity(line: str) -> bool:
    """True for a line that opens with an entity name in caps."""
    # An office lead is a title only with "of" or "for" ("GENERAL PARTNER OF ACME LP");
    # "PRINCIPAL FINANCIAL GROUP, INC." is an entity.
    if SIGNATURE_TITLE_LEAD_RE.match(line) and re.search(r"\b(?:of|for)\b", line, re.IGNORECASE):
        return False
    words = line.split()
    head = " ".join(words[:8])
    letters = [ch for ch in head if ch.isalpha()]
    if len(letters) < 4:
        return False
    if sum(1 for ch in letters if ch.isupper()) / len(letters) < 0.8:
        return False
    return SIGNATURE_ENTITY_TOKEN_RE.search(head) is not None


def _signature_looks_like_sentence(line: str) -> bool:
    if re.match(r"in\s+witness\b", line, re.IGNORECASE):
        return True
    if (
        line[:1].islower()
        or SIGNATURE_CAPACITY_LEAD_RE.match(line)
        or SIGNATURE_TITLE_LEAD_RE.match(line)
    ):
        return False
    words = line.split()
    if len(words) < SIGNATURE_SENTENCE_MIN_WORDS:
        return False
    return sum(1 for word in words if word[:1].islower()) >= 4


def _is_signature_by_label(line: str) -> bool:
    match = SIGNATURE_FIELD_LABEL_RE.fullmatch(line) or SIGNATURE_FIELD_WITH_VALUE_RE.match(line)
    return bool(match) and match.group("label").lower() == "by"


def _signature_next_line(lines: list[str], start_index: int) -> tuple[str, int] | None:
    """The next non-empty line, unless it is a bare label or a known non-value."""
    index = start_index
    while index < len(lines) and not lines[index]:
        index += 1
    if index >= len(lines):
        return None

    value = lines[index]
    if (
        not value
        or SIGNATURE_FIELD_LABEL_RE.fullmatch(value)
        or SIGNATURE_FIELD_NON_VALUE_RE.search(value)
    ):
        return None
    return value, index


def _signature_line_starts_next_block(
    lines: list[str], index: int, label: str | None
) -> bool:
    """True when lines[index] is not the value of an empty field label.

    That is another field with its value, a party-role header, a page number,
    a sentence, or an entity name in caps that starts the next signer block.
    """
    value = lines[index]
    if (
        SIGNATURE_FIELD_WITH_VALUE_RE.match(value)
        or SIGNATURE_NON_FIELD_VALUE_RE.fullmatch(value)
        or _signature_looks_like_sentence(value)
    ):
        return True
    if not _signature_starts_entity(value):
        return False
    if label is None or label.lower() != "by":
        return True
    # "By:" may name an intermediate signer ("By: ACME GP LLC, its general
    # partner"), unless the entity is followed by its own "By:" line.
    if SIGNATURE_INTERMEDIATE_SIGNER_RE.search(value):
        return False
    next_index = index + 1
    while next_index < len(lines) and not lines[next_index]:
        next_index += 1
    return next_index < len(lines) and _is_signature_by_label(lines[next_index])


def _signature_continuation_value(
    lines: list[str], start_index: int, label: str | None = None
) -> tuple[str, int] | None:
    """The value on the next non-empty line for an empty field label, or None."""
    candidate = _signature_next_line(lines, start_index)
    if candidate is None or _signature_line_starts_next_block(lines, candidate[1], label):
        return None
    return candidate


def format_signature_text_like_screen(text: str) -> str:
    lines = _signature_text_lines(text)
    formatted: list[str] = []
    index = 0

    while index < len(lines):
        line = lines[index]
        if not line:
            if formatted and formatted[-1]:
                formatted.append("")
            index += 1
            continue

        if (
            index + 1 < len(lines)
            and _should_join_signature_entity_lines(line, lines[index + 1])
        ):
            line = f"{line} {lines[index + 1]}"
            index += 1

        # Labels keep their printed case and colon: the source's "By", "by:" or
        # "NAME:" is served as printed, never normalized to "By:" / "Name:".
        label_match = SIGNATURE_FIELD_LABEL_RE.fullmatch(line)
        if label_match:
            label = line
            candidate = _signature_next_line(lines, index + 1)
            if candidate is not None:
                value, value_index = candidate
                if _signature_line_starts_next_block(
                    lines, value_index, label_match.group("label")
                ):
                    # The field is empty; the next line opens something else.
                    formatted.append(label)
                    index += 1
                    continue
                formatted.append(f"{label} {value}")
                index = value_index + 1
                continue

        field_value_match = SIGNATURE_FIELD_WITH_VALUE_RE.match(line)
        if field_value_match:
            label = field_value_match.group("label")
            if field_value_match.group("sep").startswith(":"):
                label += ":"
            formatted.append(f"{label} {field_value_match.group('value').strip()}")
            index += 1
            continue

        if line.lower() == "by":
            continuation = _signature_continuation_value(lines, index + 1, "by")
            if continuation is not None:
                value, value_index = continuation
                if SIGNATURE_VALUE_RE.search(value):
                    formatted.append(f"{line} {value}")
                    index = value_index + 1
                    continue

        formatted.append(line)
        index += 1

    text = "\n".join(formatted).replace(_SIGNATURE_LABEL_CELL_JOIN, " ")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# The period that ends a printed heading, left behind when the tagger closed the
# heading tag before it ("<section>7.02 Access to Information</section>." or the
# period alone on the next line, also after a page break when the heading ends a
# page). Not an ellipsis and not a decimal (".5%").
_LEADING_PAGE_MARKERS_RE = re.compile(r"(?:\s*<page>[^<]*</page>|\s*<pageUUID>[^<]*</pageUUID>)*\s*")
# The period may also be glued to the first enumerator (".(a) The Company ...")
# or to the first word (".In the event ..."), never to a digit (".5%").
_ORPHAN_HEADING_PERIOD_RE = re.compile(
    r"\.(?!\.)(?=[ \t]*(?:\n|<|$)|[ \t]+\S|\([a-zA-Z0-9]{1,4}\)\s|[A-Z][a-z])"
)

# A space the page text put before punctuation at an inline-tag boundary
# ("<u>Counterparts</u><font></font>." reads "Counterparts ."). The mark must
# follow a printed character on the same line and end a word: a space, a
# closing quote or bracket, or the end of the text comes next, so a mark that
# opens a number (" .5%", "1 ,000", "10 : 30") or a word (" .NET") stays. An
# ellipsis (" . . .") stays, and so does a mark after another mark of the set
# or at the start of a line (a lone "." enumerator), and so does a spaced mark
# between two numbers ("10 : 30", "Section 3 . 2") or after a letter-spaced
# word ("W I T N E S S E T H :").
_SPACE_BEFORE_PUNCTUATION_RE = re.compile(
    r"(?<=([^\s.,;:]))[ \t]+([.,;:])(?=\s|$|[)\]\"”’])(?![ \t]*\.)"
)
_NEXT_PRINTED_CHAR_RE = re.compile(r"\s*(\S)")
_LETTER_SPACED_TAIL_RE = re.compile(r"(?<!\S)(?:\S ){2,}\S$")


def _tighten_punctuation(match: re.Match[str]) -> str:
    start = match.start()
    if _LETTER_SPACED_TAIL_RE.search(match.string, max(start - 80, 0), start) is not None:
        return match.group(0)
    if match.group(1).isdigit():
        following = _NEXT_PRINTED_CHAR_RE.match(match.string, match.end())
        if following is not None and following.group(1).isdigit():
            return match.group(0)
    return match.group(2)


def tighten_space_before_punctuation(text: str) -> str:
    """Remove the space an inline-tag boundary left before "." "," ";" or ":"."""
    if " " not in text and "\t" not in text:
        return text
    return _SPACE_BEFORE_PUNCTUATION_RE.sub(_tighten_punctuation, text)


def _drop_orphan_heading_period(content: str) -> str:
    """Drop a lone "." that opens the text right after an article or section heading.

    A section's own text never starts with a bare period; it is the heading's
    terminator. Page markers before it stay; the rest of the content is unchanged.
    """
    if "." not in content[:400]:
        return content
    markers = _LEADING_PAGE_MARKERS_RE.match(content)
    start = markers.end() if markers is not None else 0
    if _ORPHAN_HEADING_PERIOD_RE.match(content, start) is None:
        return content
    return content[:start] + content[start + 1:].lstrip(" \t")


def convert_to_xml(
    tagged_text: str,
    agreement_uuid: str,
    filing_date: date,
    url: str,
    source_format: str,
    inline_splits: InlineSplitEvidence | None = None,
) -> str:
    """
    Convert text with <article>...</article> and <section>...</section> headings
    into a proper XML hierarchy.

    Process:
    - Wraps any leading text (before the first <article>) in <recitals>/<text>/<definition>/<page> blocks
    - Wraps all <article>…</article> elements inside a top-level <body> element
    - Standalone <page>…</page> lines become <page> elements

    Args:
        tagged_text: Text containing article and section tags.
        agreement_uuid: UUID of the agreement.
        filing_date: Date of the filing.
        url: URL of the source document.
        source_format: Format of the source document.
        inline_splits: The agreement's raw body HTML; when given, ordinals and
            possessives an inline-tag boundary split ("15 th", "Broker’ s") are
            rejoined where the raw confirms it.

    Returns:
        XML string representation of the document.
    """
    # Find all <article> or <section> headings and their positions.
    # We infer each heading's source page UUID from the first page marker that
    # appears after the heading start in the flattened body text.
    pattern = re.compile(r"<(article|section)>(.*?)</\1>", re.DOTALL)
    page_uuid_pattern = re.compile(r"<pageUUID>(.*?)</pageUUID>", re.DOTALL)
    matches = list(pattern.finditer(tagged_text))
    page_uuid_markers = [
        (m.start(), m.group(1).strip())
        for m in page_uuid_pattern.finditer(tagged_text)
        if m.group(1).strip()
    ]

    def heading_page_uuid(heading_start: int) -> str | None:
        if not page_uuid_markers:
            return None
        for marker_pos, marker_page_uuid in page_uuid_markers:
            if heading_start < marker_pos:
                return marker_page_uuid
        return page_uuid_markers[-1][1]

    root = ET.Element("document", uuid=agreement_uuid)

    # Add metadata
    _ = add_metadata_nodes(root, agreement_uuid, filing_date, url, source_format)

    # Helper to add <text>, <definition>, <pageUUID> or <page> children
    def add_text_nodes(parent: ET.Element, text_block: str) -> None:
        """
        Add text nodes to the parent element based on content patterns.

        Args:
            parent: Parent XML element.
            text_block: Text content to process.
        """
        # Definition: starts with "…some text…" means …
        definition_re_a = re.compile(
            r'^[\u201C\u201D"]'  # opening curly or straight quote
            + r'[^"\u201C\u201D]+'  # the term itself
            + r'[\u201C\u201D"]\s+'  # closing quote + space
            + r"(?:mean|means|shall have the meaning|shall mean)\b",
            re.IGNORECASE,
        )
        term_re = re.compile(r'^[\u201C\u201D"]([^"\u201C\u201D]+)[\u201C\u201D"]')

        definition_re_b = re.compile(
            r"""(?xi)                       # case‐insensitive, verbose
            (?:                             # two alternatives:
              [\u201C\u201D"]               #   opening curly or straight quote
              ([^"\u201C\u201D]+)           #   term1
              [\u201C\u201D"]               #   closing quote
              \s+or\s+                      
              [\u201C\u201D"]               #   opening quote for term2
              ([^"\u201C\u201D]+)           #   term2
              [\u201C\u201D"]               #   closing quote
              \s+
              (?:mean|means|shall\ mean|shall\ have\ the\ meaning)
            |
              [\u201C\u201D"]               #   opening quote
              ([^"\u201C\u201D]+)           #   term
              [\u201C\u201D"]               #   closing quote
              (?:\s+\S+){0,5}               #   up to 4 words
              \s+
              (?:mean|means|shall\ mean|shall\ have\ the\ meaning)
            )
            \b""",
            re.IGNORECASE | re.VERBOSE,
        )

        for stripped in _iter_line_fragments(text_block):
            # The element kind is decided on the text before split ordinals and
            # possessives are rejoined, so the rejoin never turns a <text> into
            # a <definition>; the element keeps the rejoined text.
            joined = stripped
            if not stripped.startswith("<page"):
                stripped = tighten_space_before_punctuation(stripped)
                joined = inline_splits.join(stripped) if inline_splits is not None else stripped

            # 1) <pageUUID>
            if re.fullmatch(r"<pageUUID>.*?</pageUUID>", stripped, re.DOTALL):
                m_uuid = re.search(r"<pageUUID>(.*?)</pageUUID>", stripped, re.DOTALL)
                pu = ET.SubElement(parent, "pageUUID")
                pu.text = m_uuid.group(1).strip() if m_uuid else stripped

            # 2) <page>…</page>
            elif re.fullmatch(r"<page>.*?</page>", stripped, re.DOTALL):
                m_page = re.search(r"<page>(.*?)</page>", stripped, re.DOTALL)
                p = ET.SubElement(parent, "page")
                p.text = m_page.group(1).strip() if m_page else stripped

            # 3) <definition> if it starts with "…" means …
            elif definition_re_a.match(stripped) or definition_re_b.match(stripped):
                # Extract the first quoted term, lowercase it
                m_term = term_re.match(joined)
                term_val = m_term.group(1).lower() if m_term else ""

                # Now include it as an attribute
                d = ET.SubElement(
                    parent, "definition", standardID="<placeholder>", term=term_val
                )
                d.text = joined

            # 4) Otherwise normal <text>
            else:
                t = ET.SubElement(parent, "text")
                t.text = joined

    # 1) Leading text → <frontMatter>
    first_pos = matches[0].start() if matches else len(tagged_text)
    leading = tagged_text[:first_pos].strip()
    if leading:
        rec = ET.SubElement(root, "frontMatter")
        add_text_nodes(rec, leading)

    # 2) Create <body> wrapper and then process articles/sections into it
    body = ET.SubElement(root, "body")
    current_article = None
    article_uuid_title = ""
    section_count = 0
    article_count = 0
    previous_article_number: int | None = None

    for i, m in enumerate(matches):
        tag = m.group(1)
        raw_title = m.group(2).strip()
        title = " ".join(raw_title.split())
        # Restore the space a glued heading lost ("ARTICLE IVREPRESENTATIONS",
        # "Section 5.1Closing"); the title is otherwise the printed text.
        if tag == "article":
            title = normalize_glued_article_title(
                title,
                previous_article_number + 1 if previous_article_number is not None else None,
            )
            parsed_number = article_title_number(title)
            if parsed_number is not None:
                previous_article_number = parsed_number
        else:
            title = normalize_glued_section_title(title)
        # Ids hash the titles as they were before the punctuation is tightened
        # and split ordinals and possessives are rejoined, so neither fix moves
        # served ids.
        uuid_title = title
        title = tighten_space_before_punctuation(title)
        if inline_splits is not None:
            title = inline_splits.join(title)
        source_page_uuid = heading_page_uuid(m.start())

        _, end = m.span()
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(tagged_text)
        content = _drop_orphan_heading_period(tagged_text[end:next_start].strip())

        if tag == "article":
            article_count += 1
            article_title_normed = clean_article_title(title)
            article_uuid_title = uuid_title

            article_attrs = {
                "title": title,
                "uuid": get_uuid(agreement_uuid + uuid_title),
                "order": str(article_count),
                "standardId": "<placeholder>",
            }
            if source_page_uuid is not None:
                article_attrs["pageUUID"] = source_page_uuid
            current_article = ET.SubElement(body, "article", attrib=article_attrs)

            section_count = 0
            if content:
                add_text_nodes(current_article, content)

        else:  # section
            # If no current article, attach section directly under body.
            container = current_article if current_article is not None else body

            section_count += 1
            article_title_normed = (
                clean_article_title(article_uuid_title)
                if current_article is not None
                else ""
            )
            section_title_normed = clean_section_title(uuid_title)
            section_attrs = {
                "title": title,
                "uuid": make_section_uuid(
                    agreement_uuid,
                    article_title_normed,
                    section_title_normed,
                    article_count if current_article is not None else None,
                    section_count,
                ),
                "order": str(section_count),
                "standardId": "<placeholder>",
            }
            if source_page_uuid is not None:
                section_attrs["pageUUID"] = source_page_uuid
            sec = ET.SubElement(container, "section", attrib=section_attrs)
            if content:
                add_text_nodes(sec, content)

    # Note: trailing text after the final heading remains within the last article/section's content.

    # Pretty-print with encoding in header
    rough = ET.tostring(root, "utf-8")
    return rough


def collapse_text_into_definitions(xml_str: str) -> str:
    """
    Move <text> elements into preceding <definition> elements and ensure proper structure.

    Process:
    1. Moves <text> (and the relevant <page>/<pageUUID>) into the preceding <definition>.
    2. Ensures that any free text directly inside a <definition> is wrapped in its own <text> tag,
       so <definition> contains only <text>, <page>, and <pageUUID> children.

    Args:
        xml_str: XML string to process.

    Returns:
        Processed XML string.
    """
    root = ET.fromstring(xml_str)

    def process_container(parent: ET.Element) -> None:
        """Process a container element to move text into definitions."""
        children = list(parent)
        for idx, child in enumerate(children):
            if child.tag != "definition":
                continue

            def_elem = child
            # Locate next <definition> or end
            next_def_idx = next(
                (
                    k
                    for k in range(idx + 1, len(children))
                    if children[k].tag == "definition"
                ),
                len(children),
            )
            segment = children[idx + 1 : next_def_idx]

            # Find any <text> in the segment
            text_indices = [i for i, el in enumerate(segment) if el.tag == "text"]
            if not text_indices:
                continue

            last_text_idx = max(text_indices)
            # Choose elements to move: all <text>, plus any <page>/<pageUUID> at or before last text
            to_move = [
                el
                for i, el in enumerate(segment)
                if el.tag == "text"
                or (el.tag in ("page", "pageUUID") and i <= last_text_idx)
            ]

            # Move them under the definition
            for el in to_move:
                parent.remove(el)
                def_elem.append(el)

    # First pass: collapse relevant <text>/<page>/<pageUUID> into definitions
    for elem in root.iter():
        process_container(elem)

    # Second pass: wrap any free-floating text inside <definition> into its own <text> tag
    for def_elem in root.iter("definition"):
        # Wrap leading text
        if def_elem.text and def_elem.text.strip():
            txt = def_elem.text
            new = ET.Element("text")
            new.text = txt
            def_elem.insert(0, new)
        def_elem.text = None

        # Wrap tails after children
        for child in list(def_elem):
            if child.tail and child.tail.strip():
                txt = child.tail
                new = ET.Element("text")
                new.text = txt
                idx = list(def_elem).index(child)
                def_elem.insert(idx + 1, new)
            child.tail = None

    return ET.tostring(root, encoding="unicode")


def _has_text_value(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


_BLOCK_JOIN_PAGE_TYPES = frozenset({"body", "front_matter", "back_matter"})


def _xml_page_text(row: Any, *, toc_drop_keys: frozenset[str] = frozenset()) -> str:
    tagged_output = row.get("tagged_output")
    gold_label = row.get("gold_label")
    source_page_type = (
        gold_label if _has_text_value(gold_label) else row.get("source_page_type")
    )
    if (
        source_page_type == "toc"
        and bool(row.get("source_is_html"))
        and _has_text_value(row.get("raw_page_content"))
    ):
        formatted_toc = format_toc_html_like_screen(
            str(row.get("raw_page_content")), drop_keys=toc_drop_keys
        )
        if formatted_toc.strip():
            return _strip_xml_invalid_control_chars(formatted_toc)
    if (
        source_page_type == "sig"
        and bool(row.get("source_is_html"))
        and _has_text_value(row.get("raw_page_content"))
    ):
        formatted_signature_html = format_signature_html_like_screen(
            str(row.get("raw_page_content"))
        )
        if formatted_signature_html.strip():
            return _strip_xml_invalid_control_chars(formatted_signature_html)
    if (
        source_page_type == "sig"
        and bool(row.get("source_is_html"))
        and _has_text_value(tagged_output)
    ):
        formatted_signature = format_signature_text_like_screen(str(tagged_output))
        if formatted_signature.strip():
            return _strip_xml_invalid_control_chars(formatted_signature)
    page_text = (
        _strip_xml_invalid_control_chars(str(tagged_output))
        if _has_text_value(tagged_output)
        else ""
    )
    source_is_html = row.get("source_is_html")
    raw_page_content = row.get("raw_page_content")
    if (
        page_text
        and source_page_type in _BLOCK_JOIN_PAGE_TYPES
        and source_is_html is not None
        and source_is_html == source_is_html
        and bool(source_is_html)
        and _has_text_value(raw_page_content)
    ):
        # Lines the HTML extraction split inside one printed block (blank
        # lines inside a <p>, top-level text of a page cut mid-block).
        page_text = join_raw_block_splits(page_text, str(raw_page_content)).text
    source_is_txt = row.get("source_is_txt")
    # NULL can arrive as None or NaN (truthy), so test the value, not bool(NaN).
    if page_text and source_is_txt is not None and source_is_txt == source_is_txt and bool(source_is_txt):
        # Underline dash lines that the reflow fused into sentences. Sig pages
        # keep theirs: a dash line there is a signature line.
        raw_page_content = row.get("raw_page_content")
        if source_page_type != "sig" and _has_text_value(raw_page_content):
            page_text = strip_inline_underlines(page_text, str(raw_page_content)).text
        # Pages from old EDGAR .txt filings carry SGML markup (<TABLE>, <S>,
        # <C>, <PAGE>, the <DOCUMENT> header) and text-table rule lines.
        page_order = row.get("page_order")
        page_text = strip_edgar_sgml(
            page_text,
            page_type=str(source_page_type) if _has_text_value(source_page_type) else None,
            page_order=int(page_order) if page_order is not None and page_order == page_order else None,
        )
    return page_text


def _html_raw_pages(rows: Any) -> list[str]:
    """The raw HTML of the rows that come from HTML sources."""
    if "raw_page_content" not in rows.columns:
        return []
    return [
        str(r["raw_page_content"])
        for _, r in rows.iterrows()
        # NULL can arrive as None or NaN (truthy), so test the value, not bool(NaN).
        if r.get("source_is_html") is not None
        and r.get("source_is_html") == r.get("source_is_html")
        and bool(r.get("source_is_html"))
        and _has_text_value(r.get("raw_page_content"))
    ]


def _toc_running_extra_keys(toc_rows: Any) -> frozenset[str]:
    """Extras printed on two or more HTML TOC pages (running headers and footers) stay out."""
    raw_pages = _html_raw_pages(toc_rows)
    return repeated_toc_extra_keys(raw_pages) if len(raw_pages) >= 2 else frozenset()


def _body_inline_split_evidence(body_rows: Any) -> InlineSplitEvidence | None:
    """The raw HTML of the HTML body pages, or None when there is none."""
    raw_pages = _html_raw_pages(body_rows)
    return InlineSplitEvidence(raw_pages) if raw_pages else None


def generate_xml(
    df: Any, version_map: dict[str, int] | None = None
) -> tuple[list[XMLData], list[XMLGenerationFailure]]:
    """
    Generate XML data from a DataFrame.

    Args:
        df: DataFrame containing agreement data.
        version_map: Optional dict mapping agreement_uuid to version number.
                    If not provided, defaults to version 1 for all.

    Returns:
        A tuple:
        - list of successfully generated XMLData objects
        - list of recoverable per-agreement XML generation failures
    """
    staged_xml = []
    failures = []
    if version_map is None:
        version_map = {}

    # Helper: add simple nodes (text/definition/page/pageUUID) to container
    def add_text_nodes_simple(parent: ET.Element, text_block: str) -> None:
        definition_re_a = re.compile(
            r'^[\u201C\u201D\"][^"\u201C\u201D]+[\u201C\u201D\"]\s+(?:mean|means|shall have the meaning|shall mean)\b',
            re.IGNORECASE,
        )
        term_re = re.compile(r'^[\u201C\u201D\"]([^"\u201C\u201D]+)[\u201C\u201D\"]')
        definition_re_b = re.compile(
            r"""(?xi)
            (?:
              [\u201C\u201D\"][^"\u201C\u201D]+[\u201C\u201D\"]\s+or\s+[\u201C\u201D\"][^"\u201C\u201D]+[\u201C\u201D\"]\s+(?:mean|means|shall\ mean|shall\ have\ the\ meaning)
            |
              [\u201C\u201D\"][^"\u201C\u201D]+[\u201C\u201D\"](?:\s+\S+){0,5}\s+(?:mean|means|shall\ mean|shall\ have\ the\ meaning)
            )\b
            """,
            re.IGNORECASE | re.VERBOSE,
        )

        for stripped in _iter_line_fragments(text_block):
            if re.fullmatch(r"<pageUUID>.*?</pageUUID>", stripped, re.DOTALL):
                m_uuid = re.search(r"<pageUUID>(.*?)</pageUUID>", stripped, re.DOTALL)
                pu = ET.SubElement(parent, "pageUUID")
                pu.text = m_uuid.group(1).strip() if m_uuid else stripped
            elif re.fullmatch(r"<page>.*?</page>", stripped, re.DOTALL):
                m_page = re.search(r"<page>(.*?)</page>", stripped, re.DOTALL)
                p = ET.SubElement(parent, "page")
                p.text = m_page.group(1).strip() if m_page else stripped
            elif definition_re_a.match(stripped) or definition_re_b.match(stripped):
                m_term = term_re.match(stripped)
                term_val = m_term.group(1).lower() if m_term else ""
                d = ET.SubElement(parent, "definition", standardID="<placeholder>", term=term_val)
                d.text = stripped
            else:
                t = ET.SubElement(parent, "text")
                t.text = stripped

    agreement_uuids = df["agreement_uuid"].unique().tolist()
    for agreement_uuid in agreement_uuids:
        try:
            temp = df[df["agreement_uuid"] == agreement_uuid].copy()
            # Preserve order
            if "page_order" in temp.columns:
                temp = temp.sort_values(by=["page_order", "page_uuid"], kind="stable")
            else:
                temp = temp.sort_values(by=["page_uuid"], kind="stable")
            temp["_xml_page_type"] = temp["source_page_type"]
            if "gold_label" in temp.columns:
                gold_label_mask = temp["gold_label"].map(_has_text_value)
                temp.loc[gold_label_mask, "_xml_page_type"] = temp.loc[
                    gold_label_mask, "gold_label"
                ]

            url = temp["url"].to_list()[0]
            announcement_date = temp["filing_date"].to_list()[0]
            source_format = "html" if temp["source_is_html"].to_list()[0] else "txt"

            root = ET.Element("document", uuid=agreement_uuid)
            _ = add_metadata_nodes(root, agreement_uuid, announcement_date, url, source_format)

            # Containers by page type
            # frontMatter
            fm_rows = temp[temp["_xml_page_type"] == "front_matter"]
            if not fm_rows.empty:
                fm_el = ET.SubElement(root, "frontMatter")
                text_block = "\n".join(
                    (f"{_xml_page_text(r)}<pageUUID>{r['page_uuid']}</pageUUID>") for _, r in fm_rows.iterrows()
                )
                add_text_nodes_simple(fm_el, text_block)

            # tableOfContents
            toc_rows = temp[temp["_xml_page_type"] == "toc"]
            if not toc_rows.empty:
                toc_el = ET.SubElement(root, "tableOfContents")
                toc_drop_keys = _toc_running_extra_keys(toc_rows)
                text_block = "\n".join(
                    (f"{_xml_page_text(r, toc_drop_keys=toc_drop_keys)}<pageUUID>{r['page_uuid']}</pageUUID>")
                    for _, r in toc_rows.iterrows()
                )
                add_text_nodes_simple(toc_el, text_block)

            # body (preserve page order; parse headings across all body pages)
            body_rows = temp[temp["_xml_page_type"] == "body"]
            body_source_texts = [_xml_page_text(r) for _, r in body_rows.iterrows()]
            body_strip = strip_page_furniture(body_source_texts)
            if not body_rows.empty:
                body_el = ET.SubElement(root, "body")
                body_page_texts = body_strip.page_texts
                body_text = "\n".join(
                    f"{page_text}<pageUUID>{page_uuid}</pageUUID>"
                    for page_text, page_uuid in zip(body_page_texts, body_rows["page_uuid"].to_list())
                )
                body_text = strip_subsection_section_tags(body_text)
                tmp_xml = convert_to_xml(
                    body_text,
                    agreement_uuid,
                    announcement_date,
                    url,
                    source_format,
                    inline_splits=_body_inline_split_evidence(body_rows),
                )
                tmp_root = ET.fromstring(tmp_xml)
                # Include any leading content that appeared before the first heading within body pages
                fm_tmp = tmp_root.find("frontMatter")
                if fm_tmp is not None:
                    for child in list(fm_tmp):
                        body_el.append(child)
                # Merge parsed body (articles/sections spanning pages)
                body_tmp = tmp_root.find("body")
                if body_tmp is not None:
                    for child in list(body_tmp):
                        body_el.append(child)

            # sigPages and backMatter: page numbers and confirmed body furniture
            # come off the page edges (they are rendered page by page).
            nonbody_rows = temp[temp["_xml_page_type"].isin(["sig", "back_matter"])]
            nonbody_texts = strip_nonbody_page_furniture(
                [_xml_page_text(r) for _, r in nonbody_rows.iterrows()],
                [int(o) for o in nonbody_rows["page_order"].to_list()] if "page_order" in nonbody_rows.columns
                else list(range(len(nonbody_rows))),
                body_page_texts=body_source_texts,
                body_page_orders=[int(o) for o in body_rows["page_order"].to_list()] if "page_order" in body_rows.columns
                else list(range(len(body_rows))),
                body_removed=body_strip.removed,
                page_types=nonbody_rows["_xml_page_type"].to_list(),
            )
            nonbody_text_by_uuid = dict(zip(nonbody_rows["page_uuid"].to_list(), nonbody_texts))

            for page_type, container_name in (("sig", "sigPages"), ("back_matter", "backMatter")):
                rows = temp[temp["_xml_page_type"] == page_type]
                if rows.empty:
                    continue
                container_el = ET.SubElement(root, container_name)
                text_block = "\n".join(
                    f"{nonbody_text_by_uuid[r['page_uuid']]}<pageUUID>{r['page_uuid']}</pageUUID>"
                    for _, r in rows.iterrows()
                )
                add_text_nodes_simple(container_el, text_block)

            xml_str = ET.tostring(root, encoding="unicode")
            xml_str = collapse_text_into_definitions(xml_str)
            xml_str = xml.dom.minidom.parseString(xml_str).toprettyxml(indent="  ")
        except (ET.ParseError, ExpatError) as error:
            failures.append(
                XMLGenerationFailure(agreement_uuid=agreement_uuid, error=str(error))
            )
            continue

        # Get version from map or default to 1
        version = version_map.get(agreement_uuid, 1)

        staged_xml.append(
            XMLData(
                agreement_uuid=agreement_uuid,
                xml=xml_str,
                version=version,
            )
        )

    return staged_xml, failures
