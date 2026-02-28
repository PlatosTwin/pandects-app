# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
# Standard library
import os
import re
import threading
import time
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol, TypedDict, cast
from urllib.parse import urlparse

# Third-party libraries
import pandas as pd
from bs4 import BeautifulSoup
from bs4.element import Comment, NavigableString, PageElement, Tag
from requests.api import get as requests_get
from typing_extensions import NotRequired

# Rate limiting configuration
_request_times: deque[float] = deque()
_request_lock = threading.Lock()
_RATE_LIMIT = 10  # requests per period
_RATE_PERIOD = 1.025  # seconds
_PADDED_QUOTED_TERM_RE = re.compile(
    r'(?<![A-Za-z0-9])([“"])\s+([^"”\n]{1,220}?)\s+([”"])(?![A-Za-z0-9])'
)


class ClassifierProbs(TypedDict):
    front_matter: float
    toc: float
    body: float
    sig: float
    back_matter: float


class ClassifierPrediction(TypedDict):
    pred_class: str
    pred_probs: ClassifierProbs
    postprocess_modified: bool


ClassifierPredRaw = dict[str, object]
ClassifierPredsRaw = Sequence[ClassifierPredRaw] | Sequence[Sequence[ClassifierPredRaw]]


class ClassifierModelProtocol(Protocol):
    def classify(self, df: pd.DataFrame) -> ClassifierPredsRaw: ...


class LoggerProtocol(Protocol):
    def info(self, msg: str) -> None: ...


class ContextProtocol(Protocol):
    log: LoggerProtocol


class AgreementRow(TypedDict):
    url: str
    agreement_uuid: str


class CleanupRow(TypedDict):
    agreement_uuid: str
    is_txt: bool
    is_html: bool
    page_order: int
    content: str
    page_uuid: str | None


class PageFragment(TypedDict):
    content: str
    order: NotRequired[int]
    page_uuid: NotRequired[str | None]


@dataclass
class PageMetadata:
    """Metadata for a single page of an agreement."""

    agreement_uuid: str | None = None
    page_order: int | None = None
    raw_page_content: str | None = None
    processed_page_content: str | None = None
    source_is_txt: bool | None = None
    source_is_html: bool | None = None
    source_page_type: str | None = None
    page_type_prob_front_matter: float | None = None
    page_type_prob_toc: float | None = None
    page_type_prob_body: float | None = None
    page_type_prob_sig: float | None = -1
    page_type_prob_back_matter: float | None = None
    page_uuid: str | None = None
    postprocess_modified: bool | None = None


def _is_semantically_hidden_tag(tag: Tag) -> bool:
    if getattr(tag, "attrs", None) is None:
        return False
    if tag.has_attr("hidden"):
        return True
    aria_attr = tag.get("aria-hidden")
    if isinstance(aria_attr, list):
        aria_val = " ".join(aria_attr)
    else:
        aria_val = str(aria_attr or "")
    return aria_val.strip().lower() == "true"


def pull_agreement_content(url: str, timeout: float = 10.0) -> str:
    """
    Fetch the HTML content at the given URL with rate limiting.
    Used in FROM_SCRATCH mode only.

    Args:
        url: The URL to pull.
        timeout: Seconds to wait before timing out.

    Returns:
        The page's HTML content.

    Raises:
        requests.HTTPError: On bad HTTP status codes.
        requests.RequestException: On connection issues, timeouts, etc.
    """
    with _request_lock:
        now = time.time()
        # Drop timestamps older than RATE_PERIOD
        while _request_times and now - _request_times[0] >= _RATE_PERIOD:
            _ = _request_times.popleft()

        if len(_request_times) >= _RATE_LIMIT:
            # Wait until the oldest request moves outside the window
            sleep_for = _RATE_PERIOD - (now - _request_times[0])
            time.sleep(sleep_for)

        _request_times.append(time.time())

    headers = {
        "User-Agent": "New York University School of Law nmb9729@nyu.edu",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov",
        "Connection": "keep-alive",
    }
    resp = requests_get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def classify(
    classifier_model: ClassifierModelProtocol, data: list[PageMetadata]
) -> ClassifierPredsRaw:
    """
    Classify pages using the provided classifier model.

    Args:
        classifier_model: The classifier model to use for classification.
        data: List of PageMetadata objects to classify.

    Returns:
        List of classification results for each page.
    """
    agreement_uuids = [pm.agreement_uuid for pm in data]
    texts = [pm.processed_page_content for pm in data]
    orders = [pm.page_order for pm in data]
    htmls = [
        pm.raw_page_content
        if (pm.source_is_html is True and pm.source_is_txt is not True)
        else None
        for pm in data
    ]

    df = pd.DataFrame(
        {"agreement_uuid": agreement_uuids, "text": texts, "html": htmls, "order": orders},
        copy=False,
    )

    return classifier_model.classify(df)


def normalize_text(text: str) -> str:
    """
    Normalize text by handling whitespace and newlines consistently.

    Process:
    1. Replace non-breaking spaces with regular spaces.
    2. Temporarily collapse any cluster of two or more newlines
       (even if separated by spaces/tabs) into a placeholder.
    3. Replace all remaining single newlines with a space.
    4. Restore each placeholder back to a single newline.
    5. Collapse runs of two or more spaces into a single space.

    Args:
        text: The text to normalize.

    Returns:
        The normalized text.
    """
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", text)

    # Collapse multi-newline clusters into a placeholder
    placeholder = "__NL__"
    text = re.sub(r"\n[ \t]*(?:\n[ \t]*)+", placeholder, text)
    # Convert any leftover single newlines to spaces
    text = text.replace("\n", " ")
    # Restore placeholders to single newlines
    text = text.replace(placeholder, "\n\n")
    # Collapse multiple spaces to one
    text = re.sub(r" {2,}", " ", text)

    return text.strip()


def normalize_padded_quoted_terms(text: str) -> str:
    def repl(match: re.Match[str]) -> str:
        inner = match.group(2)
        if re.search(r"[A-Za-z0-9]", inner) is None:
            return match.group(0)
        return f"{match.group(1)}{inner.strip()}{match.group(3)}"

    return _PADDED_QUOTED_TERM_RE.sub(repl, text)


def _normalize_padded_quoted_terms(text: str) -> str:
    return normalize_padded_quoted_terms(text)


def strip_formatting_tags(
    soup: BeautifulSoup, remove_tags: list[str] | None = None
) -> BeautifulSoup:
    """
    Remove font-like tags and clean up formatting.

    Process:
    1. Strip style attributes
    2. Unwrap formatting tags (but leave all their whitespace intact)
    3. Normalize NBSPs to spaces
    4. Leave newlines alone

    Args:
        soup: BeautifulSoup object to process.
        remove_tags: List of tag names to remove. If None, uses default list.

    Returns:
        The cleaned BeautifulSoup object.
    """
    # Remove HTML comments
    for c in soup.find_all(string=lambda text: isinstance(text, Comment)):
        _ = c.extract()

    def _is_hidden_self(tag: Tag) -> bool:
        # Keep CSS-hidden text. Some SEC filings place agreement body in a display:none
        # container that still represents the canonical document text.
        return _is_semantically_hidden_tag(tag)

    def _has_hidden_ancestor(tag: Tag) -> bool:
        if _is_hidden_self(tag):
            return True
        for parent in tag.parents:
            if _is_hidden_self(parent):
                return True
        return False

    for tag in list(soup.find_all(True)):
        if _has_hidden_ancestor(tag):
            tag.decompose()

    if remove_tags is None:
        remove_tags = [
            "font",
            "span",
            "b",
            "strong",
            "i",
            "em",
            "u",
            "small",
            "big",
            "sup",
            "sub",
            "tt",
            "s",
            "strike",
            "ins",
            "del",
            "a",
        ]

    # Remove all style attributes
    for tag in soup.find_all(True):
        _ = tag.attrs.pop("style", None)

    no_space_before_chars = set(".,;:!?)]}%")
    no_space_after_chars = set("([{")
    opening_quote_chars = {"“", "‘", "«", "‹", '"', "'"}
    closing_quote_chars = {"”", "’", "»", "›", '"', "'"}
    formatting_tag_names = set(remove_tags)

    def _node_text(node: object) -> str:
        if isinstance(node, NavigableString):
            return str(node)
        if isinstance(node, Tag):
            return node.get_text()
        return ""

    def _previous_nonempty_sibling(node: Tag) -> object | None:
        sibling = node.previous_sibling
        while sibling is not None and _node_text(sibling) == "":
            sibling = sibling.previous_sibling
        return sibling

    def _next_nonempty_sibling(node: Tag) -> object | None:
        sibling = node.next_sibling
        while sibling is not None and _node_text(sibling) == "":
            sibling = sibling.next_sibling
        return sibling

    def _previous_substantive_sibling(node: object) -> object | None:
        sibling = getattr(node, "previous_sibling", None)
        while sibling is not None and not _node_text(sibling).strip():
            sibling = sibling.previous_sibling
        return sibling

    def _next_substantive_sibling(node: object) -> object | None:
        sibling = getattr(node, "next_sibling", None)
        while sibling is not None and not _node_text(sibling).strip():
            sibling = sibling.next_sibling
        return sibling

    def _sibling_has_content(sib: object | None) -> bool:
        return bool(_node_text(sib).strip())

    def _sibling_ends_space(sib: object | None) -> bool:
        text = _node_text(sib)
        return bool(text) and text.rstrip() != text

    def _sibling_starts_space(sib: object | None) -> bool:
        text = _node_text(sib)
        return bool(text) and text.lstrip() != text

    def _is_formatting_or_whitespace_node(node: object | None) -> bool:
        if node is None:
            return False
        if isinstance(node, NavigableString):
            return not str(node).strip()
        return isinstance(node, Tag) and node.name in formatting_tag_names

    def _first_nonspace_char(text: str) -> str:
        stripped = text.lstrip()
        return stripped[0] if stripped else ""

    def _last_nonspace_char(text: str) -> str:
        stripped = text.rstrip()
        return stripped[-1] if stripped else ""

    def _quote_only_tag_text(text: str) -> str:
        stripped = text.strip()
        if len(stripped) == 1 and (
            stripped in opening_quote_chars or stripped in closing_quote_chars
        ):
            return stripped
        return text

    def _trim_text_node_after_opening_quote(text: str) -> str:
        return re.sub(r'([“"‘«‹])\s+$', r"\1", text)

    def _trim_text_node_before_closing_quote(text: str) -> str:
        return re.sub(r'^\s+([”"’»›])', r"\1", text)

    def _trim_tag_text_at_quote_edges(
        text: str,
        prev_text: str,
        next_text: str,
        within_quote_run: bool,
    ) -> str:
        if not within_quote_run or not text.strip():
            return text
        trimmed = text
        prev_last_char = _last_nonspace_char(prev_text)
        next_first_char = _first_nonspace_char(next_text)
        first_char = _first_nonspace_char(trimmed)
        last_char = _last_nonspace_char(trimmed)
        if prev_last_char in opening_quote_chars or first_char in opening_quote_chars:
            trimmed = trimmed.lstrip()
        if next_first_char in closing_quote_chars or last_char in closing_quote_chars:
            trimmed = trimmed.rstrip()
        return trimmed

    def _tag_is_within_quoted_format_run(tag: Tag) -> bool:
        left_edge: object = tag
        while _is_formatting_or_whitespace_node(left_edge.previous_sibling):
            sibling = left_edge.previous_sibling
            if sibling is None:
                break
            left_edge = sibling

        right_edge: object = tag
        while _is_formatting_or_whitespace_node(right_edge.next_sibling):
            sibling = right_edge.next_sibling
            if sibling is None:
                break
            right_edge = sibling

        run_parts: list[str] = []
        current: object | None = left_edge
        while current is not None:
            run_parts.append(_node_text(current))
            if current is right_edge:
                break
            current = current.next_sibling
        run_text = "".join(run_parts)
        run_first_char = _first_nonspace_char(run_text)
        run_last_char = _last_nonspace_char(run_text)

        prev_outer = _previous_substantive_sibling(left_edge)
        next_outer = _next_substantive_sibling(right_edge)
        prev_outer_char = _last_nonspace_char(_node_text(prev_outer))
        next_outer_char = _first_nonspace_char(_node_text(next_outer))

        opening_char = (
            run_first_char if run_first_char in opening_quote_chars else prev_outer_char
        )
        closing_char = (
            run_last_char if run_last_char in closing_quote_chars else next_outer_char
        )
        return opening_char in opening_quote_chars and closing_char in closing_quote_chars

    # Unwrap every formatting tag (preserve whitespace around tags)
    for tag_name in remove_tags:
        replacements: dict[int, tuple[PageElement, str]] = {}
        for tag in list(soup.find_all(tag_name)):
            # Check adjacent siblings to determine if we need to add spaces
            prev_sibling = _previous_nonempty_sibling(tag)
            next_sibling = _next_nonempty_sibling(tag)
            prev_text = _node_text(prev_sibling)
            next_text = _node_text(next_sibling)
            within_quote_run = _tag_is_within_quoted_format_run(tag)

            if within_quote_run and isinstance(prev_sibling, NavigableString):
                prev_replacement = _trim_text_node_after_opening_quote(str(prev_sibling))
                if prev_replacement != str(prev_sibling):
                    replacements[id(prev_sibling)] = (prev_sibling, prev_replacement)
                    prev_text = prev_replacement
            if within_quote_run and isinstance(next_sibling, NavigableString):
                next_replacement = _trim_text_node_before_closing_quote(str(next_sibling))
                if next_replacement != str(next_sibling):
                    replacements[id(next_sibling)] = (next_sibling, next_replacement)
                    next_text = next_replacement

            # Get the text content of the tag
            tag_text = tag.get_text()
            tag_text = _quote_only_tag_text(tag_text)
            tag_text = _trim_tag_text_at_quote_edges(
                tag_text,
                prev_text,
                next_text,
                within_quote_run,
            )

            # If the tag is empty/whitespace-only but sits between two content nodes,
            # preserve a single space to avoid concatenation (e.g. "Section 1.1" + NBSP font
            # + "Defined Terms" -> "Section 1.1 Defined Terms").
            if not tag_text.strip():
                if (
                    not within_quote_run
                    and
                    _sibling_has_content(prev_sibling)
                    and _sibling_has_content(next_sibling)
                    and not _sibling_ends_space(prev_sibling)
                    and not _sibling_starts_space(next_sibling)
                ):
                    replacements[id(tag)] = (tag, " ")
                    continue
                replacements[id(tag)] = (tag, "")
                continue
            
            # Check if we need a space before the tag content
            # Only add space if previous sibling is text that doesn't end with whitespace
            # and tag text doesn't start with whitespace
            needs_space_before = False
            if prev_text.strip():  # Previous sibling has non-whitespace content
                prev_ends_space = prev_text.rstrip() != prev_text  # Ends with whitespace
                tag_starts_space = tag_text and tag_text.lstrip() != tag_text  # Starts with whitespace
                prev_last_char = prev_text.rstrip()[-1] if prev_text.rstrip() else ""
                tag_first_char = tag_text.lstrip()[0] if tag_text.lstrip() else ""
                wrapped_by_quotes = within_quote_run
                # Avoid splitting a single word that is visually styled across tags,
                # e.g. "D<small>EFINITIONS</small>" -> "DEFINITIONS".
                split_word_boundary = bool(prev_last_char.isalpha() and tag_first_char.isalpha())
                starts_with_closing_punct = bool(tag_first_char in no_space_before_chars)
                is_closing_quote_tag = bool(tag_text.strip() in closing_quote_chars)
                starts_with_closing_quote = bool(tag_first_char in closing_quote_chars)
                wrapped_by_quotes_before = wrapped_by_quotes and not bool(
                    tag_text.strip() in opening_quote_chars
                )
                if (
                    not prev_ends_space
                    and not tag_starts_space
                    and not split_word_boundary
                    and prev_last_char not in opening_quote_chars
                    and not starts_with_closing_punct
                    and not starts_with_closing_quote
                    and not is_closing_quote_tag
                    and not wrapped_by_quotes_before
                ):
                    needs_space_before = True
            
            # Check if we need a space after the tag content
            # Only add space if next sibling is text that doesn't start with whitespace
            # and tag text doesn't end with whitespace
            needs_space_after = False
            if next_text.strip():  # Next sibling has non-whitespace content
                next_starts_space = next_text.lstrip() != next_text  # Starts with whitespace
                tag_ends_space = tag_text and tag_text.rstrip() != tag_text  # Ends with whitespace
                tag_last_char = tag_text.rstrip()[-1] if tag_text.rstrip() else ""
                next_first_char = next_text.lstrip()[0] if next_text.lstrip() else ""
                wrapped_by_quotes = within_quote_run
                split_word_boundary = bool(tag_last_char.isalpha() and next_first_char.isalpha())
                starts_with_closing_punct = bool(next_first_char in no_space_before_chars)
                ends_with_opening_punct = bool(tag_last_char in no_space_after_chars)
                is_opening_quote_tag = bool(tag_text.strip() in opening_quote_chars)
                ends_with_opening_quote = bool(tag_last_char in opening_quote_chars)
                wrapped_by_quotes_after = wrapped_by_quotes and not bool(
                    tag_text.strip() in closing_quote_chars
                )
                if (
                    not next_starts_space
                    and not tag_ends_space
                    and not split_word_boundary
                    and next_first_char not in closing_quote_chars
                    and not starts_with_closing_punct
                    and not ends_with_opening_punct
                    and not ends_with_opening_quote
                    and not is_opening_quote_tag
                    and not wrapped_by_quotes_after
                ):
                    needs_space_after = True
            
            # Add spaces if needed
            if needs_space_before:
                tag_text = ' ' + tag_text
            if needs_space_after:
                tag_text = tag_text + ' '

            replacements[id(tag)] = (tag, tag_text)

        for node, replacement_text in replacements.values():
            _ = node.replace_with(NavigableString(replacement_text))

    # Normalize any non-breaking spaces into real spaces
    for node in soup.find_all(string=True):
        text = str(node).replace("\u00a0", " ").replace("\xa0", " ")
        _ = node.replace_with(NavigableString(text))

    return soup


def block_level_soup(
    soup: BeautifulSoup,
    block_tags: Iterable[str] = (
        "p",
        "div",
        "li",
        "section",
        "article",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
    ),
) -> BeautifulSoup:
    """
    Create a new BeautifulSoup containing only top-level block tags.

    This allows for clean text extraction that breaks only between block elements,
    never inside one.

    Args:
        soup: BeautifulSoup object to process.
        block_tags: Iterable of tag names to consider as block-level elements.

    Returns:
        New BeautifulSoup object with only top-level block elements.
    """
    new_soup = BeautifulSoup("", "html.parser")
    root = soup.body if soup.body else soup

    for child in root.contents:
        if isinstance(child, NavigableString) and child.strip():
            p = new_soup.new_tag("p")
            p.string = str(child)
            _ = new_soup.append(p)
        elif isinstance(child, Tag) and child.name not in block_tags:
            # Avoid capturing non-block wrappers that already contain block tags
            if child.find(block_tags):
                block_descendants = child.find_all(block_tags)
                inline_texts: list[str] = []
                for text_node in child.find_all(string=True):
                    if not text_node.strip():
                        continue
                    if any(parent in block_descendants for parent in text_node.parents):
                        continue
                    inline_texts.append(str(text_node))
                if inline_texts:
                    p = new_soup.new_tag("p")
                    p.string = " ".join(inline_texts)
                    _ = new_soup.append(p)
                continue
            if child.get_text(strip=True):
                p = new_soup.new_tag("p")
                fragment = BeautifulSoup(str(child), "html.parser")
                for node in fragment.contents:
                    _ = p.append(node)
                _ = new_soup.append(p)
    for tag in soup.find_all(block_tags):
        # Skip any that live inside another block_tag
        if tag.find_parent(block_tags):
            continue
        # Clone via string round-trip so we don't detach from the original
        fragment = BeautifulSoup(str(tag), "html.parser")
        # If fragment has multiple roots, append each; otherwise append the single root
        for child in fragment.contents:
            _ = new_soup.append(child)
    return new_soup


def collapse_tables(soup: BeautifulSoup) -> BeautifulSoup:
    """
    Replace each <table> with a <p> containing the table's text content.

    Process:
    - Each row = cells' text joined by single spaces
    - Rows separated by newlines
    - Uses get_text(separator='', strip=True) so inline tags don't add spaces.

    Args:
        soup: BeautifulSoup object to process.

    Returns:
        BeautifulSoup object with tables converted to paragraphs.
    """
    for table in soup.find_all("table"):
        rows: list[str] = []
        for tr in table.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            texts: list[str] = []
            for cell in cells:
                # Formatting-tag spacing has already been normalized upstream, so using an
                # empty separator here avoids reintroducing spaces inside quoted terms when
                # table cells contain text split across inline tags.
                raw = cell.get_text(separator="", strip=False)
                # Collapse multiple whitespace to single space
                clean = re.sub(r"\s+", " ", raw).strip()
                if clean:
                    texts.append(clean)
            if texts:
                rows.append(" ".join(texts))
        new_p = soup.new_tag("p")
        new_p.string = "\n".join(rows)
        _ = table.replace_with(new_p)
    return soup


def _extract_relaxed_html_text(content: str) -> str:
    soup = BeautifulSoup(content, "html.parser")
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        _ = c.extract()
    for tag in soup.find_all(["script", "style", "noscript", "template"]):
        tag.decompose()
    for tag in list(soup.find_all(True)):
        if _is_semantically_hidden_tag(tag):
            tag.decompose()
    text = soup.get_text(separator="\n\n", strip=False).strip()
    return normalize_text(text)


def _should_use_relaxed_html_fallback(primary_text: str, relaxed_text: str) -> bool:
    primary_len = len(primary_text)
    relaxed_len = len(relaxed_text)
    return (
        primary_len <= 2000
        and relaxed_len >= 20000
        and primary_len * 10 < relaxed_len
    )


def format_content(content: str, is_txt: bool, is_html: bool) -> str:
    """
    Format content based on its source type.

    Args:
        content: The raw content to format.
        is_txt: Whether the content is from a text file.
        is_html: Whether the content is from an HTML file.

    Returns:
        The formatted content.

    Raises:
        RuntimeError: If the source type is unknown.
    """
    if is_txt:
        return normalize_text(content)
    elif is_html:
        html = BeautifulSoup(content, "html.parser")
        cleaned = strip_formatting_tags(html)
        cleaned = collapse_tables(cleaned)
        primary_text = _normalize_padded_quoted_terms(normalize_text(
            block_level_soup(cleaned).get_text(separator="\n\n", strip=False).strip()
        ))
        if len(primary_text) <= 2000:
            relaxed_text = _normalize_padded_quoted_terms(_extract_relaxed_html_text(content))
            if _should_use_relaxed_html_fallback(primary_text, relaxed_text):
                return relaxed_text
        return primary_text
    else:
        raise RuntimeError("Unknown page source type.")


def split_to_pages(content: str, is_txt: bool, is_html: bool) -> list[PageFragment]:
    """
    Split content into individual pages.

    Args:
        content: The content to split.
        is_txt: Whether the content is from a text file.
        is_html: Whether the content is from an HTML file.

    Returns:
        List of dictionaries containing page content and metadata.

    Raises:
        RuntimeError: If the source type is unknown.
    """
    if is_txt:
        fragments = re.split(r"<PAGE>", content)
        # Filter out entirely empty fragments but preserve original order indices
        txt_pages: list[PageFragment] = [
            {"content": page, "order": i}
            for i, page in enumerate(fragments)
            if page.strip()
        ]
        return txt_pages
    elif is_html:
        soup = BeautifulSoup(content, "html.parser")

        # Remove HTML comments
        for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
            _ = c.extract()

        # Convert div-based page-breaks to <hr data-page-break>
        def _parse_style(style: str) -> dict[str, str]:
            parsed: dict[str, str] = {}
            for part in style.split(";"):
                if ":" not in part:
                    continue
                key, value = part.split(":", 1)
                key = key.strip().lower()
                value = value.strip().lower()
                if key:
                    parsed[key] = value
            return parsed

        def _is_break_value(value: str) -> bool:
            return value in {"page", "always", "left", "right", "recto", "verso"}

        def is_page_break_div(tag: Tag) -> bool:
            if tag.name != "div":
                return False
            style_attr = tag.get("style")
            if isinstance(style_attr, list):
                style = " ".join(style_attr)
            else:
                style = str(style_attr or "")
            styles = _parse_style(style)
            for key in ("page-break-before", "page-break-after", "break-before", "break-after"):
                value = styles.get(key)
                if value and _is_break_value(value):
                    return True
            return False

        for div in soup.find_all(is_page_break_div):
            style_attr = div.get("style")
            if isinstance(style_attr, list):
                style = " ".join(style_attr).lower()
            else:
                style = str(style_attr or "").lower()
            styles = _parse_style(style)
            has_hr = div.find("hr") is not None
            if not has_hr:
                before_val = styles.get("page-break-before") or styles.get("break-before")
                if before_val and _is_break_value(before_val):
                    hr = soup.new_tag("hr")
                    hr["data-page-break"] = "true"
                    _ = div.insert_before(hr)
            if not has_hr:
                after_val = styles.get("page-break-after") or styles.get("break-after")
                if after_val and _is_break_value(after_val):
                    hr = soup.new_tag("hr")
                    hr["data-page-break"] = "true"
                    _ = div.insert_after(hr)

        # Mark every other <hr> outside of tables
        for hr in soup.find_all("hr"):
            attr_val = hr.get("data-page-break")
            if isinstance(attr_val, list):
                data_break = " ".join(attr_val)
            else:
                data_break = str(attr_val or "")
            if data_break != "true" and hr.find_parent("table") is None:
                hr["data-page-break"] = "true"

        # Regex-split on those data-page-break HRs (catches nested ones)
        normalized = str(soup)
        fragments = re.split(
            r'(?i)<hr\b[^>]*\bdata-page-break\s*=\s*"true"[^>]*>', normalized
        )

        # Filter out entirely empty fragments (from consecutive page breaks or page breaks at start/end)
        html_pages: list[PageFragment] = [
            {"content": frag}
            for frag in fragments
            if frag.strip()
        ]
        return html_pages
    else:
        raise RuntimeError("Unknown page source type.")


def _format_page(page: PageFragment, is_txt: bool, is_html: bool) -> str:
    return format_content(page["content"], is_txt, is_html)


def _count_alpha_tokens(text: str) -> int:
    return sum(1 for token in text.split() if re.search(r"[A-Za-z]", token))


def _attach_preds_to_pages(
    page_objs: list[PageMetadata], preds: ClassifierPredsRaw
) -> None:
    # Flatten preds (supports list or list-of-lists) into a single list
    if preds and isinstance(preds[0], dict):
        flat_preds = list(cast(Sequence[ClassifierPredRaw], preds))
    else:
        flat_preds = [
            item
            for sublist in cast(Sequence[Sequence[ClassifierPredRaw]], preds)
            for item in sublist
        ]
    for po, pc in zip(page_objs, flat_preds):
        pred = cast(ClassifierPrediction, cast(object, pc))
        po.source_page_type = pred["pred_class"]
        pp = pred["pred_probs"]
        po.page_type_prob_front_matter = pp["front_matter"]
        po.page_type_prob_toc = pp["toc"]
        po.page_type_prob_body = pp["body"]
        po.page_type_prob_sig = pp["sig"]
        po.page_type_prob_back_matter = pp["back_matter"]
        po.postprocess_modified = pred["postprocess_modified"]


def pre_process(
    context: ContextProtocol | None,
    rows: list[AgreementRow],
    classifier_model: ClassifierModelProtocol,
) -> tuple[list[PageMetadata], dict[str, bool]]:
    """
    Split agreements into pages, classify page type, and process HTML into formatted text.

    Args:
        context: Optional context for logging.
        rows: List of agreement data dictionaries.
        classifier_model: Model to use for page classification.

    Returns:
        Tuple of processed PageMetadata objects and a mapping of agreement UUIDs
        to whether they appear paginated.
    """
    staged_pages: list[PageMetadata] = []
    all_page_objs: list[PageMetadata] = []
    pagination_statuses: dict[str, bool] = {}

    for agreement in rows:
        raw_url = agreement["url"]
        if raw_url.startswith("view-source:"):
            raw_url = raw_url.split(":", 1)[1]
        path = urlparse(raw_url).path
        filename = os.path.basename(path)
        _, ext = os.path.splitext(filename)

        if ext.lower() in [".html", ".htm"]:
            is_html = True
            is_txt = False
        elif ext.lower() in [".txt"]:
            is_html = False
            is_txt = True
        else:
            raise RuntimeError(f"Unknown filing extension: {ext}")

        # Pull down content from EDGAR
        content = pull_agreement_content(raw_url)

        # Split into individual pages
        pages = split_to_pages(content, is_txt, is_html)

        is_paginated = len(pages) > 10
        pagination_statuses[agreement["agreement_uuid"]] = is_paginated

        if not is_paginated:
            if context:
                context.log.info(
                    f"Agreement {agreement['agreement_uuid']} likely is not paginated. Skipping page upload."
                )
            else:
                print(
                    f"Agreement {agreement['agreement_uuid']} likely is not paginated. Skipping page upload."
                )
            continue

        # Format and classify
        page_objs: list[PageMetadata] = []
        page_order = 0
        for page in pages:
            formatted = _format_page(page, is_txt, is_html)
            if not formatted:
                continue
            if _count_alpha_tokens(formatted) > 2000:
                if context:
                    context.log.info(
                        f"Agreement {agreement['agreement_uuid']} has a long page. Skipping page upload."
                    )
                else:
                    print(
                        f"Agreement {agreement['agreement_uuid']} has a long page. Skipping page upload."
                    )
                pagination_statuses[agreement["agreement_uuid"]] = False
                page_objs = []
                break
            page_objs.append(
                PageMetadata(
                    agreement_uuid=agreement["agreement_uuid"],
                    page_order=page_order,
                    raw_page_content=page["content"],
                    processed_page_content=formatted,
                    source_is_txt=is_txt,
                    source_is_html=is_html,
                    page_uuid=page.get("page_uuid"),
                )
            )
            page_order += 1
        if page_objs:
            all_page_objs.extend(page_objs)

    if not all_page_objs:
        return staged_pages, pagination_statuses

    preds = classify(classifier_model, all_page_objs)
    _attach_preds_to_pages(all_page_objs, preds)
    staged_pages.extend(all_page_objs)
    return staged_pages, pagination_statuses


def cleanup(
    rows: list[CleanupRow],
    classifier_model: ClassifierModelProtocol,
    context: ContextProtocol | None = None,
) -> list[PageMetadata]:
    """
    Clean up and reprocess existing page data.

    Args:
        rows: List of page data dictionaries.
        classifier_model: Model to use for page classification.
        context: Optional context for logging.

    Returns:
        List of reprocessed PageMetadata objects.
    """
    _ = context
    if not rows:
        return []

    all_page_objs: list[PageMetadata] = []
    for agreement in set(r["agreement_uuid"] for r in rows):
        pages = [r for r in rows if r["agreement_uuid"] == agreement]
        is_txt = pages[0]["is_txt"]
        is_html = pages[0]["is_html"]
        page_order = 0
        for p in sorted(pages, key=lambda x: x["page_order"]):
            formatted = format_content(p["content"], is_txt, is_html)
            if not formatted:
                continue

            all_page_objs.append(
                PageMetadata(
                    agreement_uuid=agreement,
                    page_order=page_order,
                    raw_page_content=p["content"],
                    processed_page_content=formatted,
                    source_is_txt=is_txt,
                    source_is_html=is_html,
                    page_uuid=p["page_uuid"],
                )
            )
            page_order += 1

    if not all_page_objs:
        return []

    # context.pdb.set_trace()
    preds = classify(classifier_model, all_page_objs)
    _attach_preds_to_pages(all_page_objs, preds)
    
    return all_page_objs


if __name__ == "__main__":
    from etl.models.page_classifier_revamp.inference import ClassifierInference
    from etl.models.page_classifier_revamp.page_classifier_constants import (
        CLASSIFIER_CRF_PATH,
    )

    clf_mdl = ClassifierInference(model_path=CLASSIFIER_CRF_PATH)

    _, _ = pre_process(
        None,
        [
            {
                "url": "https://www.sec.gov/Archives/edgar/data/1035884/000110465907050454/a07-17577_1ex2d1.htm",
                "agreement_uuid": "foo",
            }
        ],
        clf_mdl,
    )

    print()
