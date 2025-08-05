# Standard library
import os
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from typing import Iterable, List, Optional
from urllib.parse import urlparse
import pprint

# Third-party libraries
import pandas as pd
import requests
from bs4 import BeautifulSoup
from bs4.element import Comment, NavigableString

_request_times = deque()
_request_lock = threading.Lock()
_RATE_LIMIT = 10  # requests
_RATE_PERIOD = 1.025  # seconds


def get_uuid(x):
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, x))


@dataclass
class PageMetadata:
    agreement_uuid: str
    page_order: int
    raw_page_content: str
    processed_page_content: str
    source_is_txt: bool
    source_is_html: bool
    source_page_type: Optional[str] = None
    page_type_prob_front_matter: Optional[float] = None
    page_type_prob_toc: Optional[float] = None
    page_type_prob_body: Optional[float] = None
    page_type_prob_sig: Optional[float] = None
    page_type_prob_back_matter: Optional[float] = None


def pull_agreement_content(url: str, timeout: float = 10.0) -> str:
    """
    Fetch the HTML content at the given URL and return it as a string,
    but send no more than 10 requests per 1 second.

    Args:
        url: The URL to pull.
        timeout: Seconds to wait before timing out.

    Returns:
        The page’s HTML.

    Raises:
        requests.HTTPError: on bad HTTP status codes.
        requests.RequestException: on connection issues, timeouts, etc.
    """
    with _request_lock:
        now = time.time()
        # drop timestamps older than RATE_PERIOD
        while _request_times and now - _request_times[0] >= _RATE_PERIOD:
            _request_times.popleft()

        if len(_request_times) >= _RATE_LIMIT:
            # wait until the oldest request moves outside the window
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


def classify(classifier_model, data):
    df = pd.DataFrame([asdict(pm) for pm in data])

    df["html"] = df[["raw_page_content", "source_is_txt"]].apply(
        lambda x: pd.NA if x["source_is_txt"] else x["raw_page_content"], axis=1
    )
    df.rename(
        {"processed_page_content": "text", "page_order": "order"}, axis=1, inplace=True
    )

    model_output = classifier_model.classify(df)[0]

    return model_output


def normalize_text(text: str) -> str:
    """
    1. Replace non-breaking spaces with regular spaces.
    2. Temporarily collapse any cluster of two or more newlines
       (even if separated by spaces/tabs) into a placeholder.
    3. Replace all remaining single newlines with a space.
    4. Restore each placeholder back to a single newline.
    5. Collapse runs of two or more spaces into a single space.
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

    text = text.strip()

    return text


def strip_formatting_tags(soup: BeautifulSoup, remove_tags=None) -> BeautifulSoup:
    """
    Remove font-like tags by:
      1) stripping style attributes,
      2) unwrapping them (but leaving all their whitespace intact),
      3) normalizing NBSPs to spaces,
      4) (leaving newlines alone).
    """
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

    # 1) Remove all style attributes
    for tag in soup.find_all(True):
        tag.attrs.pop("style", None)

    # 2) Unwrap every formatting tag (don't strip its inner whitespace)
    for tag_name in remove_tags:
        for tag in soup.find_all(tag_name):
            tag.unwrap()

    # 3) Normalize any non-breaking spaces into real spaces
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
    Returns a new BeautifulSoup containing only the top-level block_tags
    (i.e. those not nested inside another block_tag).  You can then do

        new_soup = block_level_soup(old_soup)
        text = new_soup.get_text(separator='\n\n')

    and it will break only between those block elements, never inside one.
    """
    new_soup = BeautifulSoup("", "html.parser")
    for tag in soup.find_all(block_tags):
        # skip any that live inside another block_tag
        if tag.find_parent(block_tags):
            continue
        # clone via string round-trip so we don't detach from the original
        fragment = BeautifulSoup(str(tag), "html.parser")
        # if fragment has multiple roots, append each; otherwise append the single root
        for child in fragment.contents:
            new_soup.append(child)
    return new_soup


def collapse_tables(soup: BeautifulSoup) -> BeautifulSoup:
    """
    Replace each <table> with a <p> whose text is:
      • each row = cells’ text joined by single spaces
      • rows separated by newlines
    Uses get_text(separator='', strip=True) so inline tags don’t add spaces.
    """
    for table in soup.find_all("table"):
        rows = []
        for tr in table.find_all("tr"):
            cells = tr.find_all(["td", "th"])
            texts = []
            for cell in cells:
                # pull all text without auto-spaces, then collapse whitespace
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


def format_content(content, is_txt, is_html):
    if is_txt:
        return normalize_text(content)

    elif is_html:
        html = BeautifulSoup(content, "html.parser")
        cleaned = strip_formatting_tags(html)
        cleaned = collapse_tables(cleaned)
        text = block_level_soup(cleaned).get_text(separator="\n\n", strip=False).strip()
        normed = normalize_text(text)

        return normed

    else:
        raise RuntimeError("Unknown page source type.")


def split_to_pages(content, is_txt, is_html):
    if is_txt:
        fragments = re.split(r"<PAGE>", content)
        return [{"content": page, "order": i} for i, page in enumerate(fragments)]

    elif is_html:
        soup = BeautifulSoup(content, "html.parser")

        # 1) strip out HTML comments
        for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
            c.extract()

        # 2) convert div-based page-breaks to <hr data-page-break>
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

        # 3) mark every other <hr> outside of tables
        for hr in soup.find_all("hr"):
            if hr.get("data-page-break") != "true" and hr.find_parent("table") is None:
                hr["data-page-break"] = "true"

        # 4) regex-split on those data-page-break HRs (catches nested ones)
        normalized = str(soup)
        fragments = re.split(
            r'(?i)<hr\b[^>]*\bdata-page-break\s*=\s*"true"[^>]*>', normalized
        )

        return [
            {
                "content": frag,
            }
            for frag in fragments
        ]

    else:
        raise RuntimeError(f"Unknown page source type.")


def pre_process(rows, classifier_model) -> List[PageMetadata] | None:
    """
    Splits agreements into pages, classifies page type,
    and processes HTML into formatted text.
    """
    staged_pages: List[PageMetadata] = []
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

        # 1. pull down content from EDGAR
        content = pull_agreement_content(agreement["url"])

        # 2. split into individual pages
        pages = split_to_pages(content, is_txt, is_html)

        if len(pages) <= 10:
            print("Agreement likely is not paginated. Skipping page upload.")
            continue

        # 3. loop through pages to format
        temp_pages = []
        page_order = 0
        for page in pages:

            page_formatted_content = format_content(page["content"], is_txt, is_html)

            if not page_formatted_content:
                continue

            temp_pages.append(
                PageMetadata(
                    agreement_uuid=agreement["agreement_uuid"],
                    page_order=page_order,
                    raw_page_content=page["content"],
                    processed_page_content=page_formatted_content,
                    source_is_txt=is_txt,
                    source_is_html=is_html,
                )
            )

            page_order += 1

        # 4. classify pages
        page_classes = classify(classifier_model, temp_pages)

        for page_object, page_class in zip(temp_pages, page_classes):
            page_object.source_page_type = page_class["pred_class"]
            page_object.page_type_prob_front_matter = page_class["front_matter"]
            page_object.page_type_prob_toc = page_class["toc"]
            page_object.page_type_prob_body = page_class["body"]
            page_object.page_type_prob_sig = page_class["sig"]
            page_object.page_type_prob_back_matter = page_class["back_matter"]

        staged_pages.extend(temp_pages)

    return staged_pages


if __name__ == "__main__":
    from etl.models.code.classifier import ClassifierInference
    from etl.models.code.constants import CLASSIFIER_CKPT_PATH

    classifier_model = ClassifierInference(ckpt_path=CLASSIFIER_CKPT_PATH)

    pre_process(
        [
            {
                "url": "https://www.sec.gov/Archives/edgar/data/1035884/000110465907050454/a07-17577_1ex2d1.htm",
                "agreement_uuid": "foo",
            }
        ],
        classifier_model,
    )
