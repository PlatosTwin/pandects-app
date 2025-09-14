# Standard library
import os
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Union
from urllib.parse import urlparse

# Third-party libraries
import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment, NavigableString

# Rate limiting configuration
_request_times: deque = deque()
_request_lock = threading.Lock()
_RATE_LIMIT = 10  # requests per period
_RATE_PERIOD = 1.025  # seconds


def get_uuid(x: str) -> str:
    """Generate a UUID5 hash from the input string."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, x))


@dataclass
class PageMetadata:
    """Metadata for a single page of an agreement."""

    agreement_uuid: Optional[str] = None
    page_order: Optional[int] = None
    raw_page_content: Optional[str] = None
    processed_page_content: Optional[str] = None
    source_is_txt: Optional[bool] = None
    source_is_html: Optional[bool] = None
    source_page_type: Optional[str] = None
    page_type_prob_front_matter: Optional[float] = None
    page_type_prob_toc: Optional[float] = None
    page_type_prob_body: Optional[float] = None
    page_type_prob_sig: Optional[float] = -1
    page_type_prob_back_matter: Optional[float] = None
    page_uuid: Optional[str] = None
    postprocess_modified: Optional[bool] = None


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
            _request_times.popleft()

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
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def classify(classifier_model: Any, data: List[PageMetadata]) -> List[Dict[str, Any]]:
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
        pm.raw_page_content if (pm.source_is_html and not pm.source_is_txt) else None
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


def strip_formatting_tags(
    soup: BeautifulSoup, remove_tags: Optional[List[str]] = None
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
        c.extract()

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
        tag.attrs.pop("style", None)

    # Unwrap every formatting tag (don't strip its inner whitespace)
    for tag_name in remove_tags:
        for tag in soup.find_all(tag_name):
            tag.unwrap()

    # Normalize any non-breaking spaces into real spaces
    for node in soup.find_all(string=True):
        if isinstance(node, NavigableString):
            text = str(node).replace("\u00a0", " ").replace("\xa0", " ")
            node.replace_with(text)

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
    for tag in soup.find_all(block_tags):
        # Skip any that live inside another block_tag
        if tag.find_parent(block_tags):
            continue
        # Clone via string round-trip so we don't detach from the original
        fragment = BeautifulSoup(str(tag), "html.parser")
        # If fragment has multiple roots, append each; otherwise append the single root
        for child in fragment.contents:
            new_soup.append(child)
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
        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            texts = []
            for cell in cells:
                # Pull all text without auto-spaces, then collapse whitespace
                raw = cell.get_text(separator="", strip=True)
                clean = re.sub(r"\s+", " ", raw)
                if clean:
                    texts.append(clean)
            if texts:
                rows.append(" ".join(texts))
        new_p = soup.new_tag("p")
        new_p.string = "\n".join(rows)
        table.replace_with(new_p)
    return soup


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
        text = block_level_soup(cleaned).get_text(separator="\n\n", strip=False).strip()
        return normalize_text(text)
    else:
        raise RuntimeError("Unknown page source type.")


def split_to_pages(
    content: str, is_txt: bool, is_html: bool
) -> List[Dict[str, Union[str, int]]]:
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
        return [{"content": page, "order": i} for i, page in enumerate(fragments)]
    elif is_html:
        soup = BeautifulSoup(content, "html.parser")

        # Remove HTML comments
        for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
            c.extract()

        # Convert div-based page-breaks to <hr data-page-break>
        def is_page_break_div(tag):
            if tag.name != "div":
                return False
            style = tag.get("style", "")
            return (
                "page-break-before" in style.lower()
                or "page-break-after" in style.lower()
            )

        for div in soup.find_all(is_page_break_div):
            hr = soup.new_tag("hr")
            hr["data-page-break"] = "true"
            div.replace_with(hr)

        # Mark every other <hr> outside of tables
        for hr in soup.find_all("hr"):
            if hr.get("data-page-break") != "true" and hr.find_parent("table") is None:
                hr["data-page-break"] = "true"

        # Regex-split on those data-page-break HRs (catches nested ones)
        normalized = str(soup)
        fragments = re.split(
            r'(?i)<hr\b[^>]*\bdata-page-break\s*=\s*"true"[^>]*>', normalized
        )

        return [{"content": frag} for frag in fragments]
    else:
        raise RuntimeError("Unknown page source type.")


def _format_page(page, is_txt, is_html):
    return format_content(page["content"], is_txt, is_html)


def _attach_preds_to_pages(
    page_objs: List[PageMetadata], preds: List[List[Dict[str, Any]]]
) -> None:
    # Flatten preds (list of lists) into a single list
    flat_preds = [item for sublist in preds for item in sublist]
    for po, pc in zip(page_objs, flat_preds):
        po.source_page_type = pc["pred_class"]
        pp = pc["pred_probs"]
        po.page_type_prob_front_matter = pp["front_matter"]
        po.page_type_prob_toc = pp["toc"]
        po.page_type_prob_body = pp["body"]
        po.page_type_prob_sig = pp["sig"]
        po.page_type_prob_back_matter = pp["back_matter"]
        po.postprocess_modified = pc["postprocess_modified"]


def pre_process(
    rows: List[Dict[str, Any]], classifier_model: Any
) -> Optional[List[PageMetadata]]:
    """
    Split agreements into pages, classify page type, and process HTML into formatted text.

    Args:
        rows: List of agreement data dictionaries.
        classifier_model: Model to use for page classification.

    Returns:
        List of processed PageMetadata objects, or None if processing should be skipped.
    """
    staged_pages: List[PageMetadata] = []
    all_page_objs: List[PageMetadata] = []

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

        if len(pages) <= 10:
            print("Agreement likely is not paginated. Skipping page upload.")
            return staged_pages

        # Format and classify
        page_order = 0
        for page in pages:
            formatted = _format_page(page, is_txt, is_html)
            if not formatted:
                continue
            all_page_objs.append(
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

    if not all_page_objs:
        return staged_pages

    preds = classify(classifier_model, all_page_objs)
    _attach_preds_to_pages(all_page_objs, preds)
    staged_pages.extend(all_page_objs)
    return staged_pages


def cleanup(
    rows: List[Dict[str, Any]], classifier_model: Any, context: Any
) -> Optional[List[PageMetadata]]:
    """
    Clean up and reprocess existing page data.

    Args:
        rows: List of page data dictionaries.
        classifier_model: Model to use for page classification.

    Returns:
        List of reprocessed PageMetadata objects.
    """
    if not rows:
        return []

    all_page_objs: List[PageMetadata] = []
    for agreement in set(r["agreement_uuid"] for r in rows):
        pages = [r for r in rows if r["agreement_uuid"] == agreement]
        is_txt = pages[0]["is_txt"]
        is_html = pages[0]["is_html"]
        page_order = 0
        for p in sorted(pages, key=lambda x: x["page_uuid"]):
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
    from etl.models.code.classifier import ClassifierInference
    from etl.models.code.shared_constants import CLASSIFIER_CKPT_PATH

    clf_mdl = ClassifierInference(ckpt_path=CLASSIFIER_CKPT_PATH)

    pre_process(
        [
            {
                "url": "https://www.sec.gov/Archives/edgar/data/1035884/000110465907050454/a07-17577_1ex2d1.htm",
                "agreement_uuid": "foo",
            }
        ],
        clf_mdl,
    )

    print()
