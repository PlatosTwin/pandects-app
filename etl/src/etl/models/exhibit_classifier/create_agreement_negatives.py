"""Create negative agreement examples from EDGAR exhibit links."""

import argparse
from pathlib import Path
from collections.abc import Iterable
from typing import cast
from urllib.parse import urlparse

import pandas as pd
import requests

from etl.domain.b_pre_processing import format_content, split_to_pages
from etl.utils.sec_utils import SEC_USER_AGENT


def _fetch_exhibit_content(
    exhibit_url: str, user_agent: str, timeout: float = 10.0
) -> tuple[str, bool, bool]:
    headers = {"User-Agent": user_agent}
    response = requests.get(exhibit_url, headers=headers, timeout=timeout)
    response.raise_for_status()
    content = response.text

    path = urlparse(exhibit_url).path
    suffix = path.lower().rsplit(".", 1)[-1] if "." in path else ""
    if suffix in {"htm", "html"}:
        return content, False, True
    if suffix == "txt":
        return content, True, False
    content_type = response.headers.get("Content-Type", "").lower()
    if "html" in content_type:
        return content, False, True
    if "text/plain" in content_type:
        return content, True, False
    raise ValueError(f"Unknown exhibit content type for {exhibit_url}")


def _render_agreement_text(content: str, is_txt: bool, is_html: bool) -> str:
    pages = split_to_pages(content, is_txt=is_txt, is_html=is_html)
    rendered_pages = [
        format_content(page["content"], is_txt=is_txt, is_html=is_html).strip()
        for page in pages
    ]
    return "\n\n".join(page for page in rendered_pages if page)


def _load_links(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def _append_rows(output_path: Path, texts: Iterable[str]) -> None:
    text_list = list(texts)
    df = pd.DataFrame({"text": text_list, "label": [0] * len(text_list)})
    write_header = not output_path.exists()
    df.to_csv(output_path, mode="a", index=False, header=write_header)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create negative agreement examples from EDGAR links."
    )
    _ = parser.add_argument(
        "--links-file",
        required=True,
        help="Path to a newline-delimited file of EDGAR exhibit links.",
    )
    _ = parser.add_argument(
        "--output-path",
        default="etl/src/etl/models/exhibit_classifier/data/exhibit-negatives.csv",
        help="Output CSV path (appends if file exists).",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    links_path = Path(cast(str, args.links_file))
    output_path = Path(cast(str, args.output_path))

    links = _load_links(links_path)
    if not links:
        raise RuntimeError("No links found to process.")

    texts: list[str] = []
    for idx, link in enumerate(links, start=1):
        content, is_txt, is_html = _fetch_exhibit_content(link, SEC_USER_AGENT)
        agreement_text = _render_agreement_text(content, is_txt=is_txt, is_html=is_html)
        if agreement_text.strip():
            texts.append(agreement_text)
        print(f"Processed {idx}/{len(links)} links.")

    _append_rows(output_path, texts)
    print(f"Wrote {len(texts)} rows to {output_path}.")


if __name__ == "__main__":
    main()
