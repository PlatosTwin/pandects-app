# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
import re
import xml.dom.minidom
import xml.etree.ElementTree as ET
import uuid
from dataclasses import dataclass
from typing import Any
from datetime import date
from xml.parsers.expat import ExpatError

from bs4 import BeautifulSoup
from bs4.element import Comment, Tag

from etl.domain.g_sections import clean_article_title
from etl.domain.g_sections import clean_section_title


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
SUBSECTION_NUMBER_RE = re.compile(r"\b\d+\.\d+\.\d+(?:\.\d+)*\b")
ARTICLE_TAG_RE = re.compile(r"<article\b")
INLINE_MARKER_RE = re.compile(r"(<pageUUID>.*?</pageUUID>|<page>.*?</page>)", re.DOTALL)
SIGNATURE_FIELD_LABEL_RE = re.compile(r"^(?P<label>By|Name|Title|Its):?$", re.IGNORECASE)
SIGNATURE_FIELD_WITH_VALUE_RE = re.compile(
    r"^(?P<label>By|Name|Title|Its):\s*(?P<value>.+)$",
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


def strip_subsection_section_tags(tagged_text: str) -> str:
    """
    Remove <section> tags from subsection headings like 1.2.3 or 1.2.3.4.

    Args:
        tagged_text: Tagged text that may include <section> headings.

    Returns:
        Tagged text with subsection <section> tags removed.
    """

    def replace(match: re.Match[str]) -> str:
        inner = match.group(1)
        if SUBSECTION_NUMBER_RE.search(inner):
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


def _normalize_toc_cell_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\xa0", " ")
    text = re.sub(r"[\u200B\u200C\u200D\u200E\u200F\u202A-\u202E\u2060\uFEFF]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _toc_visible_text(tag: Tag) -> str:
    return _normalize_toc_cell_text(tag.get_text(separator=" ", strip=False))


def _signature_visible_text(tag: Tag) -> str:
    return _normalize_toc_cell_text(tag.get_text(separator=" ", strip=False))


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


def _toc_table_to_lines(table: Tag, *, line_width: int) -> list[str]:
    lines: list[str] = []
    for tr in table.find_all("tr"):
        cells = tr.find_all(["td", "th"], recursive=False)
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


def _positioned_toc_lines(soup: BeautifulSoup, *, line_width: int) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for row in soup.find_all(_looks_like_positioned_toc_row):
        text = _toc_visible_text(row)
        if text in seen:
            continue
        seen.add(text)
        lines.append(_format_toc_flat_row(text, line_width=line_width))
    return lines


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


def _content_toc_table_lines(soup: BeautifulSoup, *, line_width: int) -> list[str]:
    content_tables: list[list[str]] = []
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
            content_tables.append(lines)

    if not content_tables:
        return []

    merged: list[str] = []
    for table_idx, lines in enumerate(content_tables):
        if table_idx:
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


def format_toc_html_like_screen(raw_html: str, *, line_width: int = 120) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    _remove_toc_nonvisible_nodes(soup)

    positioned_lines = _positioned_toc_lines(soup, line_width=line_width)
    if positioned_lines:
        blocks = _toc_heading_blocks(soup)
        blocks.append("\n".join(positioned_lines))
        return _wrap_toc_long_lines("\n\n".join(blocks).strip(), line_width=line_width)

    table_lines = _content_toc_table_lines(soup, line_width=line_width)
    if table_lines:
        blocks = _toc_heading_blocks(soup)
        blocks.append("\n".join(table_lines))
        return _wrap_toc_long_lines("\n\n".join(blocks).strip(), line_width=line_width)

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
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\u200B\u200C\u200D\u200E\u200F\u202A-\u202E\u2060\uFEFF]", "", text)
    return [re.sub(r"[^\S\n]+", " ", line).strip() for line in text.splitlines()]


def _normalize_signature_punctuation_spacing(text: str) -> str:
    text = re.sub(r"[^\S\r\n]+([,.;:)\]])", r"\1", text)
    return re.sub(r"([(\[])[^\S\r\n]+", r"\1", text)


def _format_signature_label(label: str) -> str:
    return label.strip().rstrip(":").title() + ":"


def _signature_table_row_text(tr: Tag) -> str | None:
    cells = tr.find_all(["td", "th"], recursive=False)
    if not cells:
        return None

    parts = [_signature_visible_text(cell) for cell in cells]
    parts = [part for part in parts if part]
    if not parts:
        return ""

    if len(parts) == 1:
        return parts[0]

    label_match = SIGNATURE_FIELD_LABEL_RE.fullmatch(parts[0])
    if label_match is not None:
        return f"{_format_signature_label(label_match.group('label'))} {' '.join(parts[1:])}"

    return " ".join(parts)


def _signature_table_to_lines(table: Tag) -> list[str]:
    lines: list[str] = []
    for tr in table.find_all("tr"):
        row_text = _signature_table_row_text(tr)
        if row_text is None:
            continue
        lines.append(row_text)
    return lines


def _signature_html_blocks(node: Tag) -> list[str]:
    blocks: list[str] = []
    for child in node.find_all(recursive=False):
        if child.name == "table":
            table_lines = _signature_table_to_lines(child)
            if table_lines:
                blocks.append("\n".join(table_lines))
            continue

        if any(
            grandchild.name in SIGNATURE_HTML_STRUCTURAL_TAGS
            for grandchild in child.find_all(recursive=False)
        ):
            blocks.extend(_signature_html_blocks(child))
            continue

        text = _signature_visible_text(child)
        if text:
            blocks.append(text)
    return blocks


def format_signature_html_like_screen(raw_html: str) -> str:
    soup = BeautifulSoup(raw_html, "html.parser")
    _remove_toc_nonvisible_nodes(soup)

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


def _signature_continuation_value(lines: list[str], start_index: int) -> tuple[str, int] | None:
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

        label_match = SIGNATURE_FIELD_LABEL_RE.fullmatch(line)
        if label_match:
            label = _format_signature_label(label_match.group("label"))
            continuation = _signature_continuation_value(lines, index + 1)
            if continuation is not None:
                value, value_index = continuation
                formatted.append(f"{label} {value}")
                index = value_index + 1
                continue

        field_value_match = SIGNATURE_FIELD_WITH_VALUE_RE.match(line)
        if field_value_match:
            label = _format_signature_label(field_value_match.group("label"))
            value = field_value_match.group("value").strip()
            formatted.append(f"{label} {value}")
            index += 1
            continue

        if line.lower() == "by":
            continuation = _signature_continuation_value(lines, index + 1)
            if continuation is not None:
                value, value_index = continuation
                if SIGNATURE_VALUE_RE.search(value):
                    formatted.append(f"by: {value}")
                    index = value_index + 1
                    continue

        formatted.append(line)
        index += 1

    return re.sub(r"\n{3,}", "\n\n", "\n".join(formatted)).strip()


def convert_to_xml(
    tagged_text: str,
    agreement_uuid: str,
    filing_date: date,
    url: str,
    source_format: str,
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
                m_term = term_re.match(stripped)
                term_val = m_term.group(1).lower() if m_term else ""

                # Now include it as an attribute
                d = ET.SubElement(
                    parent, "definition", standardID="<placeholder>", term=term_val
                )
                d.text = stripped

            # 4) Otherwise normal <text>
            else:
                t = ET.SubElement(parent, "text")
                t.text = stripped

    # 1) Leading text → <frontMatter>
    first_pos = matches[0].start() if matches else len(tagged_text)
    leading = tagged_text[:first_pos].strip()
    if leading:
        rec = ET.SubElement(root, "frontMatter")
        add_text_nodes(rec, leading)

    # 2) Create <body> wrapper and then process articles/sections into it
    body = ET.SubElement(root, "body")
    current_article = None
    section_count = 0
    article_count = 0

    for i, m in enumerate(matches):
        tag = m.group(1)
        raw_title = m.group(2).strip()
        title = " ".join(raw_title.split())
        source_page_uuid = heading_page_uuid(m.start())

        _, end = m.span()
        next_start = matches[i + 1].start() if i + 1 < len(matches) else len(tagged_text)
        content = tagged_text[end:next_start].strip()

        if tag == "article":
            article_count += 1
            article_title_normed = clean_article_title(title)

            article_attrs = {
                "title": title,
                "uuid": get_uuid(agreement_uuid + title),
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
                clean_article_title(current_article.attrib.get("title", ""))
                if current_article is not None
                else ""
            )
            section_title_normed = clean_section_title(title)
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


def _xml_page_text(row: Any) -> str:
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
        formatted_toc = format_toc_html_like_screen(str(row.get("raw_page_content")))
        if formatted_toc.strip():
            return formatted_toc
    if (
        source_page_type == "sig"
        and bool(row.get("source_is_html"))
        and _has_text_value(row.get("raw_page_content"))
    ):
        formatted_signature_html = format_signature_html_like_screen(
            str(row.get("raw_page_content"))
        )
        if formatted_signature_html.strip():
            return formatted_signature_html
    if (
        source_page_type == "sig"
        and bool(row.get("source_is_html"))
        and _has_text_value(tagged_output)
    ):
        formatted_signature = format_signature_text_like_screen(str(tagged_output))
        if formatted_signature.strip():
            return formatted_signature
    return tagged_output if _has_text_value(tagged_output) else ""


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
                text_block = "\n".join(
                    (f"{_xml_page_text(r)}<pageUUID>{r['page_uuid']}</pageUUID>") for _, r in toc_rows.iterrows()
                )
                add_text_nodes_simple(toc_el, text_block)

            # body (preserve page order; parse headings across all body pages)
            body_rows = temp[temp["_xml_page_type"] == "body"]
            if not body_rows.empty:
                body_el = ET.SubElement(root, "body")
                body_text = "\n".join(
                    (f"{_xml_page_text(r)}<pageUUID>{r['page_uuid']}</pageUUID>") for _, r in body_rows.iterrows()
                )
                body_text = strip_subsection_section_tags(body_text)
                tmp_xml = convert_to_xml(
                    body_text,
                    agreement_uuid,
                    announcement_date,
                    url,
                    source_format,
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

            # sigPages
            sig_rows = temp[temp["_xml_page_type"] == "sig"]
            if not sig_rows.empty:
                sig_el = ET.SubElement(root, "sigPages")
                text_block = "\n".join(
                    (f"{_xml_page_text(r)}<pageUUID>{r['page_uuid']}</pageUUID>") for _, r in sig_rows.iterrows()
                )
                add_text_nodes_simple(sig_el, text_block)

            # backMatter
            bm_rows = temp[temp["_xml_page_type"] == "back_matter"]
            if not bm_rows.empty:
                bm_el = ET.SubElement(root, "backMatter")
                text_block = "\n".join(
                    (f"{_xml_page_text(r)}<pageUUID>{r['page_uuid']}</pageUUID>") for _, r in bm_rows.iterrows()
                )
                add_text_nodes_simple(bm_el, text_block)

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
