"""Create agreement training data from EDGAR exhibit links."""

import argparse
from pathlib import Path
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
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    return [line.strip() for line in lines if line.strip()]


def _fetch_texts_from_links(links: list[str], label_name: str) -> list[str]:
    texts: list[str] = []
    for idx, link in enumerate(links, start=1):
        try:
            content, is_txt, is_html = _fetch_exhibit_content(link, SEC_USER_AGENT)
            agreement_text = _render_agreement_text(content, is_txt=is_txt, is_html=is_html)
            if agreement_text.strip():
                texts.append(agreement_text)
        except Exception as e:
            print(f"Failed to fetch {label_name} link {idx}/{len(links)}: {e}")
            continue
        print(f"Processed {label_name} {idx}/{len(links)} links.")
    return texts


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create agreement training data from EDGAR links."
    )
    default_data_dir = Path(__file__).parent / "data"
    _ = parser.add_argument(
        "--positives-file",
        default=str(default_data_dir / "exhibit-positives.txt"),
        help="Path to newline-delimited file of positive EDGAR exhibit links.",
    )
    _ = parser.add_argument(
        "--negatives-file",
        default=str(default_data_dir / "exhibit-negatives.txt"),
        help="Path to newline-delimited file of negative EDGAR exhibit links.",
    )
    _ = parser.add_argument(
        "--output-path",
        default=str(default_data_dir / "exhibit-data.parquet"),
        help="Output parquet path.",
    )
    group = parser.add_mutually_exclusive_group()
    _ = group.add_argument(
        "--positives-only",
        action="store_true",
        help="Only re-read positives file; preserve existing negatives from parquet.",
    )
    _ = group.add_argument(
        "--negatives-only",
        action="store_true",
        help="Only re-read negatives file; preserve existing positives from parquet.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    positives_path = Path(cast(str, args.positives_file))
    negatives_path = Path(cast(str, args.negatives_file))
    output_path = Path(cast(str, args.output_path))
    positives_only = cast(bool, args.positives_only)
    negatives_only = cast(bool, args.negatives_only)

    # Load existing data if doing partial update
    existing_df: pd.DataFrame | None = None
    if (positives_only or negatives_only) and output_path.exists():
        existing_df = pd.read_parquet(output_path)  # pyright: ignore[reportUnknownMemberType]

    if positives_only:
        positive_links = _load_links(positives_path)
        if not positive_links:
            raise RuntimeError("No positive links found to process.")
        positive_texts = _fetch_texts_from_links(positive_links, "positive")
        positive_df = pd.DataFrame({"text": positive_texts, "label": [1] * len(positive_texts)})
        if existing_df is not None:
            negative_df = existing_df[existing_df["label"] == 0]  # pyright: ignore[reportUnknownVariableType]
        else:
            negative_df = pd.DataFrame(columns=["text", "label"])  # pyright: ignore[reportArgumentType]
        combined_df = pd.concat([positive_df, negative_df], ignore_index=True)  # pyright: ignore[reportUnknownArgumentType]
        _ = combined_df.to_parquet(output_path, index=False)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        print(
            f"Replaced positives: {len(positive_texts)} positives, {len(negative_df)} negatives " +  # pyright: ignore[reportUnknownArgumentType]
            f"({len(combined_df)} total) to {output_path}."
        )
    elif negatives_only:
        negative_links = _load_links(negatives_path)
        if not negative_links:
            raise RuntimeError("No negative links found to process.")
        negative_texts = _fetch_texts_from_links(negative_links, "negative")
        negative_df = pd.DataFrame({"text": negative_texts, "label": [0] * len(negative_texts)})
        if existing_df is not None:
            positive_df = existing_df[existing_df["label"] == 1]  # pyright: ignore[reportUnknownVariableType]
        else:
            positive_df = pd.DataFrame(columns=["text", "label"])  # pyright: ignore[reportArgumentType]
        combined_df = pd.concat([positive_df, negative_df], ignore_index=True)  # pyright: ignore[reportUnknownArgumentType]
        _ = combined_df.to_parquet(output_path, index=False)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        print(
            f"Replaced negatives: {len(positive_df)} positives, {len(negative_texts)} negatives " +  # pyright: ignore[reportUnknownArgumentType]
            f"({len(combined_df)} total) to {output_path}."
        )
    else:
        positive_links = _load_links(positives_path)
        negative_links = _load_links(negatives_path)
        if not positive_links and not negative_links:
            raise RuntimeError("No links found to process.")
        positive_texts = _fetch_texts_from_links(positive_links, "positive")
        negative_texts = _fetch_texts_from_links(negative_links, "negative")
        positive_df = pd.DataFrame({"text": positive_texts, "label": [1] * len(positive_texts)})
        negative_df = pd.DataFrame({"text": negative_texts, "label": [0] * len(negative_texts)})
        combined_df = pd.concat([positive_df, negative_df], ignore_index=True)
        _ = combined_df.to_parquet(output_path, index=False)  # pyright: ignore[reportUnknownMemberType]
        print(
            f"Wrote {len(positive_texts)} positives and {len(negative_texts)} negatives " +
            f"({len(combined_df)} total) to {output_path}."
        )


if __name__ == "__main__":
    main()

