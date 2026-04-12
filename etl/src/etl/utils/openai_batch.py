# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from dagster import AssetExecutionContext

if TYPE_CHECKING:
    from openai import OpenAI

TERMINAL_BATCH_STATUSES = ("completed", "failed", "cancelled", "expired")


def extract_output_text_from_batch_body(body: dict[str, Any]) -> str:
    """Pull assistant message text from a Responses API body payload."""
    output = body.get("output")
    if not isinstance(output, list):
        raise ValueError(f"Expected body.output to be a list, got {type(output).__name__}")

    msg_blocks = [o for o in output if o.get("type") == "message"]
    if not msg_blocks:
        raise ValueError("No assistant message block in output.")

    contents = msg_blocks[0].get("content")
    if not isinstance(contents, list):
        raise ValueError(
            f"Expected message content to be a list, got {type(contents).__name__}"
        )

    text_items = [c for c in contents if isinstance(c, dict) and "text" in c]
    if not text_items:
        raise ValueError("Assistant message has no text content.")

    raw_text = text_items[0]["text"]
    if not isinstance(raw_text, str):
        raise ValueError(f"Expected text to be a string, got {type(raw_text).__name__}")
    return raw_text


def read_openai_file_text(resp: Any) -> str:
    """Read OpenAI file content from SDK response objects across SDK variants."""
    text_attr = getattr(resp, "text", None)
    if callable(text_attr):
        out_text = text_attr()
    elif isinstance(text_attr, str):
        out_text = text_attr
    else:
        content_attr = getattr(resp, "content", None)
        if isinstance(content_attr, bytes):
            out_text = content_attr.decode("utf-8")
        else:
            read_attr = getattr(resp, "read", None)
            if not callable(read_attr):
                raise TypeError("Batch output content has no text/content/read interface.")
            raw_bytes = read_attr()
            if not isinstance(raw_bytes, bytes):
                raise TypeError("Batch output read() did not return bytes.")
            out_text = raw_bytes.decode("utf-8")

    if not isinstance(out_text, str):
        raise TypeError("Batch output text is not a string.")
    return out_text


def poll_batch_until_terminal(
    context: AssetExecutionContext,
    client: "OpenAI",
    batch_id: str,
    *,
    log_prefix: str,
) -> Any:
    """Poll a batch with exponential backoff until a terminal status."""
    base_sleep_seconds = 5
    backoff_level = 0
    no_update_polls = 0
    last_progress_snapshot: tuple[Any, ...] | None = None
    max_sleep_seconds = 30 * 60

    while True:
        batch = client.batches.retrieve(batch_id)
        rc = getattr(batch, "request_counts", None)
        progress_msg: str | None = None
        if rc is not None:
            total = int(getattr(rc, "total", 0) or 0)
            completed = getattr(rc, "completed", 0) or 0
            failed = getattr(rc, "failed", 0) or 0
            done = int(completed) + int(failed)
            pct = int((done / total) * 100) if total else 0
            progress_snapshot = (batch.status, int(completed), int(failed), total)
            progress_msg = (
                f"progress={done}/{total} ({pct}%), completed={int(completed)}, failed={int(failed)}"
            )
        else:
            progress_snapshot = (batch.status,)

        if batch.status in TERMINAL_BATCH_STATUSES:
            if progress_msg is None:
                context.log.info(f"{log_prefix}: batch {batch_id} terminal status={batch.status}")
            else:
                context.log.info(
                    f"{log_prefix}: batch {batch_id} terminal status={batch.status}; {progress_msg}"
                )
            return batch

        if progress_snapshot == last_progress_snapshot:
            no_update_polls += 1
        else:
            if backoff_level > 0:
                prev_sleep = min(base_sleep_seconds * (2**backoff_level), max_sleep_seconds)
                context.log.info(
                    f"{log_prefix}: backoff reset: interval {prev_sleep}s -> {base_sleep_seconds}s"
                )
            no_update_polls = 0
            backoff_level = 0
            last_progress_snapshot = progress_snapshot

        if no_update_polls >= 10:
            prev_sleep = min(base_sleep_seconds * (2**backoff_level), max_sleep_seconds)
            backoff_level += 1
            no_update_polls = 0
            new_sleep = min(base_sleep_seconds * (2**backoff_level), max_sleep_seconds)
            if new_sleep > prev_sleep:
                context.log.info(
                    f"{log_prefix}: backoff increased: interval {prev_sleep}s -> {new_sleep}s"
                )

        sleep_seconds = min(base_sleep_seconds * (2**backoff_level), max_sleep_seconds)
        if progress_msg is None:
            context.log.info(
                f"{log_prefix}: batch {batch_id} status={batch.status}; sleeping {sleep_seconds}s"
            )
        else:
            context.log.info(
                f"{log_prefix}: batch {batch_id} status={batch.status}; {progress_msg}; sleeping {sleep_seconds}s"
            )
        time.sleep(sleep_seconds)
