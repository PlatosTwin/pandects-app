# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
from dataclasses import dataclass
from typing import Protocol, TypedDict


@dataclass
class TagData:
    """Data structure for tagged text output."""
    page_uuid: str
    tagged_text: str
    low_count: int
    spans: list[dict[str, object]]
    tokens: list[dict[str, object]]


class TaggingRow(TypedDict):
    page_uuid: str
    processed_page_content: str


class TaggedOutput(TypedDict):
    tagged: str
    low_count: int
    spans: list[dict[str, object]]
    tokens: list[dict[str, object]]


class TaggingModelProtocol(Protocol):
    def label(self, texts: list[str]) -> list[TaggedOutput]: ...


class LoggerProtocol(Protocol):
    def info(self, msg: str) -> None: ...


class ContextProtocol(Protocol):
    log: LoggerProtocol


def tag(
    rows: list[TaggingRow],
    tagging_model: TaggingModelProtocol,
    context: ContextProtocol,
) -> list[TagData]:
    """
    Tag text content using the provided tagging model.

    Args:
        rows: List of dictionaries containing page data.
        tagging_model: Model to use for tagging.

    Returns:
        List of TagData objects containing tagged text and metadata.
    """
    texts: list[str] = []
    page_uuids: list[str] = []
    for r in rows:
        if len(r["processed_page_content"].split()) > 4096:
            context.log.info(f"Skipping page {r['page_uuid']} because it is too long")
            continue

        texts.append(r["processed_page_content"])
        page_uuids.append(r["page_uuid"])

    tagged_texts = tagging_model.label(texts)

    staged_tagged: list[TagData] = []
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
