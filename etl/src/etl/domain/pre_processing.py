from typing import List, Iterable
from dataclasses import dataclass
import uuid
import os
import re
from bs4 import BeautifulSoup
import torch
import requests
from urllib.parse import urlparse
import os

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
    source_page_type: str
    page_type_prob_front_matter: float
    page_type_prob_toc: float
    page_type_prob_body: float
    page_type_prob_sig: float


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
    return text


def strip_formatting_tags(soup: BeautifulSoup, remove_tags=None) -> BeautifulSoup:
    """
    Remove font-like tags by:
      1) stripping style attributes,
      2) unwrapping them (but leaving all their whitespace intact),
      3) normalizing NBSPs to spaces,
      4) (leaving newlines alone).
    """
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
    block_tags: Iterable[str] = ("p", "div", "li", "section", "article"),
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


def pull_agreement_content(url: str, timeout: float = 10.0) -> str:
    """
    Fetch the HTML content at the given URL and return it as a string.

    Args:
        url: The URL to pull.
        timeout: Seconds to wait before timing out.

    Returns:
        The page’s HTML.

    Raises:
        requests.HTTPError: on bad HTTP status codes.
        requests.RequestException: on connection issues, timeouts, etc.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; pull_agreement_content/1.0)"
    }
    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def split_to_pages(content, is_txt, is_html):

    split_pages = []
    if is_txt:

        fragments = re.split(r"<PAGE>", content)

        split_pages = [
            {"content": page, "order": i} for i, page in enumerate(fragments)
        ]

    elif is_html:

        soup = BeautifulSoup(content, "html.parser")

        for div in soup.find_all(
            lambda tag: tag.name == "div"
            and tag.has_attr("style")
            and "page-break-before" in tag["style"].lower()
        ):
            # Replace the entire DIV (and its contents) with a simple <hr>
            div.replace_with(soup.new_tag("hr"))

        # 2. Serialize back to string
        normalized_html = str(soup)

        # Split on any <hr> tag (case‐insensitive, with optional attributes)
        fragments = re.split(r"(?i)<hr\b[^>]*>", normalized_html)

        split_pages = [
            {"content": page, "order": i} for i, page in enumerate(fragments)
        ]

    return split_pages


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


def classify(classifier_model, content, formatted_content):
    with torch.no_grad():
        logits = classifier_model(content)  # TODO: pass in appropriate vector
        probs = torch.softmax(logits, dim=1)  # probabilities
        pred_class = torch.argmax(probs, dim=1).item()

    # 0 = front matter
    # 1 = main body
    # 2 = TOC
    # 3 = sig page

    if pred_class == 0:
        page_type = "front_matter"
    elif pred_class == 3:
        page_type = "sig"
    elif pred_class == 2:
        page_type = "toc"
    elif pred_class == 1:
        page_type = "body"
    else:
        raise RuntimeError("Unknown prediction class.")

    class_dict = {
        "class": page_type,
        "prob_front_matter": probs[0, 0].item(),
        "prob_toc": probs[0, 2].item(),
        "prob_body": probs[0, 1].item(),
        "prob_sig": probs[0, 3].item(),
    }

    return class_dict


def pre_process(rows, classifier_model) -> List[PageMetadata]:
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
        base_name, ext = os.path.splitext(filename)

        if ext.lower() in [".html", ".htm"]:
            is_html = True
            is_txt = False
        elif ext.lower() in [".txt"]:
            is_html = False
            is_txt = True
        else:
            raise RuntimeError(f"Unknown filing extension: {ext}")

        # step 1: pull down content from EDGAR
        content = pull_agreement_content(agreement["url"])

        # step 2: split into individual pages
        pages = split_to_pages(content, is_txt, is_html)

        # step 3: loop through pages to classify and format
        for page in pages:
            page_formatted_content = format_content(
                page["content"], is_txt, is_html
            )  # format
            page_class = classify(
                classifier_model, page["content"], page_formatted_content
            )  # classify

            staged_pages.append(
                PageMetadata(
                    agreement_uuid=agreement["agreement_uuid"],
                    page_order=page["order"],
                    raw_page_content=page["content"],
                    processed_page_content=page_formatted_content,
                    source_is_txt=is_txt,
                    source_is_html=is_html,
                    source_page_type=page_class["class"],
                    page_type_prob_front_matter=page_class["prob_front_matter"],
                    page_type_prob_toc=page_class["prob_toc"],
                    page_type_prob_body=page_class["prob_body"],
                    page_type_prob_sig=page_class["prob_sig"],
                )
            )

    return staged_pages
