from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class TagData:
    """Data structure for tagged text output."""
    page_uuid: str
    tagged_text: str
    low_count: int
    spans: list
    chars: dict


def tag(rows: List[Dict[str, Any]], tagging_model: Any) -> List[TagData]:
    """
    Tag text content using the provided tagging model.

    Args:
        rows: List of dictionaries containing page data.
        tagging_model: Model to use for tagging.

    Returns:
        List of TagData objects containing tagged text and metadata.
    """
    texts = [r["processed_page_content"] for r in rows]
    page_uuids = [r["page_uuid"] for r in rows]

    tagged_texts = tagging_model.label(texts)

    staged_tagged = []
    for page_uuid, tagged in zip(page_uuids, tagged_texts):
        staged_tagged.append(
            TagData(
                page_uuid=page_uuid,
                tagged_text=tagged["tagged"],
                low_count=tagged["low_count"],
                spans=tagged["spans"],
                chars=tagged["chars"],
            )
        )
    
    return staged_tagged
