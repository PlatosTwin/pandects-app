from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class TagData:
    """Data structure for tagged text output."""
    page_uuid: str
    tagged_text: str
    low_count: int
    spans: list
    tokens: dict


def tag(rows: List[Dict[str, Any]], tagging_model: Any, context: Any) -> List[TagData]:
    """
    Tag text content using the provided tagging model.

    Args:
        rows: List of dictionaries containing page data.
        tagging_model: Model to use for tagging.

    Returns:
        List of TagData objects containing tagged text and metadata.
    """
    texts = []
    page_uuids = []
    for r in rows:
        if len(r["processed_page_content"].split()) > 4096:
            context.log.info(f"Skipping page {r['page_uuid']} because it is too long")
            continue

        texts.append(r["processed_page_content"])
        page_uuids.append(r["page_uuid"])

    tagged_texts = tagging_model.label(texts)

    staged_tagged = []
    for page_uuid, tagged in zip(page_uuids, tagged_texts):
        staged_tagged.append(
            TagData(
                page_uuid=page_uuid,
                tagged_text=tagged["tagged"],
                low_count=tagged["low_count"],
                spans=tagged["spans"],
                tokens=tagged["tokens"],
            )
        )
    
    return staged_tagged
