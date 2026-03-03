"""
Utilities for exporting problematic entity cases from a trained checkpoint.

This module is entity-generic. ARTICLE- and PAGE-specific helpers remain as thin
wrappers around the core entity audit builder for convenience.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict, cast

if TYPE_CHECKING:
    from .ner_classes import tags_to_spans
else:
    try:
        from .ner_classes import tags_to_spans
    except ImportError:  # pragma: no cover - supports running as a script
        from ner_classes import tags_to_spans


class EntityFailureRecord(TypedDict):
    doc_id: int
    entity_type: str
    failure_type: str
    gold_token_start: int | None
    gold_token_end: int | None
    gold_char_start: int | None
    gold_char_end: int | None
    gold_text: str | None
    gold_line_text: str | None
    gold_context: str | None
    pred_token_start: int | None
    pred_token_end: int | None
    pred_char_start: int | None
    pred_char_end: int | None
    pred_text: str | None
    pred_line_text: str | None
    pred_context: str | None
    overlap_token_count: int
    start_token_delta: int | None
    end_token_delta: int | None


ArticleFailureRecord = EntityFailureRecord
PageFailureRecord = EntityFailureRecord


def _overlaps(a: tuple[int, int, str], b: tuple[int, int, str]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _token_overlap(a: tuple[int, int, str], b: tuple[int, int, str]) -> int:
    return max(0, min(a[1], b[1]) - max(a[0], b[0]) + 1)


def _line_bounds(text: str, start: int, end: int) -> tuple[int, int]:
    line_start = text.rfind("\n", 0, max(start, 0))
    if line_start == -1:
        line_start = 0
    else:
        line_start += 1
    line_end = text.find("\n", max(end, 0))
    if line_end == -1:
        line_end = len(text)
    return line_start, line_end


def _context_bounds(text: str, start: int, end: int, radius: int) -> tuple[int, int]:
    return max(0, start - radius), min(len(text), end + radius)


def _span_details(
    span: tuple[int, int, str] | None,
    raw_text: str,
    token_offsets: list[tuple[int, int]],
    context_chars: int,
    prefix: str,
) -> dict[str, int | str | None]:
    if span is None:
        return {
            f"{prefix}_token_start": None,
            f"{prefix}_token_end": None,
            f"{prefix}_char_start": None,
            f"{prefix}_char_end": None,
            f"{prefix}_text": None,
            f"{prefix}_line_text": None,
            f"{prefix}_context": None,
        }
    start_idx, end_idx, _ = span
    if start_idx < 0 or end_idx >= len(token_offsets):
        raise ValueError("Span indices are out of bounds for token offsets.")
    char_start = token_offsets[start_idx][0]
    char_end = token_offsets[end_idx][1]
    line_start, line_end = _line_bounds(raw_text, char_start, char_end)
    context_start, context_end = _context_bounds(
        raw_text, char_start, char_end, context_chars
    )
    return {
        f"{prefix}_token_start": start_idx,
        f"{prefix}_token_end": end_idx,
        f"{prefix}_char_start": char_start,
        f"{prefix}_char_end": char_end,
        f"{prefix}_text": raw_text[char_start:char_end],
        f"{prefix}_line_text": raw_text[line_start:line_end],
        f"{prefix}_context": raw_text[context_start:context_end],
    }


def build_entity_failure_records(
    *,
    entity_type: str,
    doc_id: int,
    pred_tags: list[str],
    gold_tags: list[str],
    raw_text: str,
    token_offsets: list[tuple[int, int]],
    context_chars: int = 160,
) -> list[EntityFailureRecord]:
    entity_type = entity_type.upper()
    pred_spans = [span for span in tags_to_spans(pred_tags) if span[2] == entity_type]
    gold_spans = [span for span in tags_to_spans(gold_tags) if span[2] == entity_type]

    matched_pred: set[int] = set()
    matched_gold: set[int] = set()
    records: list[EntityFailureRecord] = []

    for pred_idx, pred_span in enumerate(pred_spans):
        for gold_idx, gold_span in enumerate(gold_spans):
            if gold_idx in matched_gold or pred_idx in matched_pred:
                continue
            if pred_span == gold_span:
                matched_pred.add(pred_idx)
                matched_gold.add(gold_idx)
                break

    for pred_idx, pred_span in enumerate(pred_spans):
        if pred_idx in matched_pred:
            continue
        best_gold_idx: int | None = None
        best_overlap = 0
        for gold_idx, gold_span in enumerate(gold_spans):
            if gold_idx in matched_gold:
                continue
            if not _overlaps(pred_span, gold_span):
                continue
            overlap = _token_overlap(pred_span, gold_span)
            if overlap > best_overlap:
                best_overlap = overlap
                best_gold_idx = gold_idx
        if best_gold_idx is None:
            continue
        gold_span = gold_spans[best_gold_idx]
        matched_pred.add(pred_idx)
        matched_gold.add(best_gold_idx)
        records.append(
            cast(
                EntityFailureRecord,
                cast(
                    object,
                    {
                    "doc_id": doc_id,
                    "entity_type": entity_type,
                    "failure_type": "boundary_mismatch",
                    **_span_details(
                        gold_span, raw_text, token_offsets, context_chars, "gold"
                    ),
                    **_span_details(
                        pred_span, raw_text, token_offsets, context_chars, "pred"
                    ),
                    "overlap_token_count": best_overlap,
                    "start_token_delta": pred_span[0] - gold_span[0],
                    "end_token_delta": pred_span[1] - gold_span[1],
                    },
                ),
            )
        )

    for gold_idx, gold_span in enumerate(gold_spans):
        if gold_idx in matched_gold:
            continue
        records.append(
            cast(
                EntityFailureRecord,
                cast(
                    object,
                    {
                    "doc_id": doc_id,
                    "entity_type": entity_type,
                    "failure_type": "false_negative",
                    **_span_details(
                        gold_span, raw_text, token_offsets, context_chars, "gold"
                    ),
                    **_span_details(
                        None, raw_text, token_offsets, context_chars, "pred"
                    ),
                    "overlap_token_count": 0,
                    "start_token_delta": None,
                    "end_token_delta": None,
                    },
                ),
            )
        )

    for pred_idx, pred_span in enumerate(pred_spans):
        if pred_idx in matched_pred:
            continue
        records.append(
            cast(
                EntityFailureRecord,
                cast(
                    object,
                    {
                    "doc_id": doc_id,
                    "entity_type": entity_type,
                    "failure_type": "false_positive",
                    **_span_details(
                        None, raw_text, token_offsets, context_chars, "gold"
                    ),
                    **_span_details(
                        pred_span, raw_text, token_offsets, context_chars, "pred"
                    ),
                    "overlap_token_count": 0,
                    "start_token_delta": None,
                    "end_token_delta": None,
                    },
                ),
            )
        )

    records.sort(
        key=lambda record: (
            0
            if record["failure_type"] == "boundary_mismatch"
            else 1
            if record["failure_type"] == "false_negative"
            else 2,
            int(record["doc_id"]),
            int(record["gold_token_start"] or record["pred_token_start"] or -1),
        )
    )
    return records


def build_article_failure_records(
    *,
    doc_id: int,
    pred_tags: list[str],
    gold_tags: list[str],
    raw_text: str,
    token_offsets: list[tuple[int, int]],
    context_chars: int = 160,
) -> list[ArticleFailureRecord]:
    return build_entity_failure_records(
        entity_type="ARTICLE",
        doc_id=doc_id,
        pred_tags=pred_tags,
        gold_tags=gold_tags,
        raw_text=raw_text,
        token_offsets=token_offsets,
        context_chars=context_chars,
    )


def build_page_failure_records(
    *,
    doc_id: int,
    pred_tags: list[str],
    gold_tags: list[str],
    raw_text: str,
    token_offsets: list[tuple[int, int]],
    context_chars: int = 160,
) -> list[PageFailureRecord]:
    return build_entity_failure_records(
        entity_type="PAGE",
        doc_id=doc_id,
        pred_tags=pred_tags,
        gold_tags=gold_tags,
        raw_text=raw_text,
        token_offsets=token_offsets,
        context_chars=context_chars,
    )
