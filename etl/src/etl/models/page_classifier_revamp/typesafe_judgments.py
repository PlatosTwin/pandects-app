"""TypeSafe page judgments used as extra CRF features.

Each page gets one TypeSafe System One request with three questions: which part of the
filing the page is in (Choice), whether it is a signature page of the main agreement (Noul),
and whether it begins material attached after the main agreement (Noul). The CRF consumes the
answers as features; code keeps owning sequence structure.
"""
# pyright: reportAny=false

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TypedDict, cast

import requests

from .page_types import PAGE_LABELS, AgreementDocument, FeatureDict

TYPESAFE_API_URL = "https://api.typesafe.ai/v1/systemone"
TYPESAFE_MODEL = "jev-1.13.0"
TYPESAFE_PROMPT_VERSION = "v3"
TYPESAFE_MAX_PAGE_CHARS = 12000
TYPESAFE_NEIGHBOR_EXCERPT_CHARS = 500
TYPESAFE_CONCURRENCY = 16
TYPESAFE_MAX_ATTEMPTS = 8
TYPESAFE_RETRY_STATUSES = frozenset({429, 500, 502, 503, 529})
MAIN_SIG_SEEN_THRESHOLD = 0.8

PART_QUESTION: dict[str, object] = {
    "type": "choice",
    "instructions": (
        "`page` is one page of an M&A agreement as filed with the SEC (merger agreement, stock or asset "
        "purchase agreement, and similar). `position` says where the page falls in the filed document, and "
        "`previous_page_end` / `next_page_start` are excerpts of the neighboring pages. Which part of the "
        "filed document is `page` in? The parts always appear in this order: cover, table of contents, "
        "main agreement text, signature pages of the main agreement, then attachments. Judge `page` by its "
        "content, not by running headers or footers repeated on every page (for example 'Table of "
        "Contents', the agreement's name, or 'Execution Version' printed at the top or bottom of each page)."
    ),
    "criteria": {
        "front_matter": (
            "Cover or title page before the table of contents and before the agreement text begins: "
            "agreement title, parties, date, filing legends such as 'Exhibit 2.1' or 'Execution Version'."
        ),
        "toc": (
            "Table of contents or index at the front of the document, before the main agreement text "
            "starts: lists of articles and sections with page numbers, lists of exhibits and schedules, or "
            "an index of defined terms that appears before the agreement text."
        ),
        "body": (
            "Operative text of the main agreement, from its preamble ('This Agreement ... dated as of ... by "
            "and among') through its last numbered section, including recitals, definitions articles, and "
            "any 'Intentionally omitted' pages within the agreement."
        ),
        "sig": (
            "Signature page of the main agreement: 'IN WITNESS WHEREOF', 'By: / Name: / Title:' blocks, "
            "'[Signature Page to Agreement and Plan of Merger]'. Signature pages that belong to an attached "
            "exhibit or ancillary agreement are not this part."
        ),
        "back_matter": (
            "Anything attached after the main agreement's signature pages: exhibits, schedules, annexes, "
            "appendices, forms of ancillary agreements (even if they contain full agreement text or their "
            "own signature blocks), disclosure letters, and an index of defined terms placed at the end."
        ),
    },
}
MAIN_SIG_QUESTION: dict[str, object] = {
    "type": "noul",
    "instructions": (
        "Is `page` a signature page of the main agreement in this filing, where the parties to that "
        "agreement sign it ('IN WITNESS WHEREOF' followed by 'By: / Name: / Title:' blocks, or '[Signature "
        "Page to ...]' naming the main agreement), or a continuation page of those same signature blocks? "
        "Answer no for signature blocks that belong to an exhibit, a form of ancillary agreement, or any "
        "other attached document."
    ),
}
ATTACHMENT_START_QUESTION: dict[str, object] = {
    "type": "noul",
    "instructions": (
        "Does `page` begin material that is attached after the main agreement has ended, such as the first "
        "page of an exhibit, schedule, annex, appendix, a form of another agreement, a list of omitted "
        "schedules, a placeholder saying an exhibit was filed separately or redacted, or an index of defined "
        "terms at the end? Answer no if `page` is the main agreement's cover, table of contents, operative "
        "text, or signature page, and no if `page` continues an attachment that began on an earlier page. "
        "A filing legend such as 'Exhibit 2.1' or 'Exhibit 10.2' on the main agreement's cover page marks "
        "the whole filing, not an attachment, so answer no for that cover page and for a table of contents "
        "that lists exhibits or schedules."
    ),
}
QUESTIONS: dict[str, dict[str, object]] = {
    "part": PART_QUESTION,
    "main_sig": MAIN_SIG_QUESTION,
    "attachment_start": ATTACHMENT_START_QUESTION,
}


class PageJudgment(TypedDict):
    part_probs: dict[str, float]
    main_sig: float
    attachment_start: float


class TypeSafeError(RuntimeError):
    """Raised when TypeSafe judgments cannot be obtained."""


def build_page_state(page_texts: list[str], page_index: int) -> dict[str, str]:
    page_count = len(page_texts)
    position = page_index + 1
    return {
        "position": (
            f"page {position} of {page_count} "
            + f"({round(100 * position / page_count)}% of the way through the document)"
        ),
        "page": page_texts[page_index][:TYPESAFE_MAX_PAGE_CHARS],
        "previous_page_end": (
            page_texts[page_index - 1][-TYPESAFE_NEIGHBOR_EXCERPT_CHARS:]
            if page_index > 0
            else "(none: this is the first page)"
        ),
        "next_page_start": (
            page_texts[page_index + 1][:TYPESAFE_NEIGHBOR_EXCERPT_CHARS]
            if page_index + 1 < page_count
            else "(none: this is the last page)"
        ),
    }


def _cache_key(state: dict[str, str]) -> str:
    payload = json.dumps(
        {"model": TYPESAFE_MODEL, "prompt_version": TYPESAFE_PROMPT_VERSION, "state": state},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _parse_judgment(response_json: dict[str, object]) -> PageJudgment:
    answers = cast(dict[str, dict[str, object]], response_json["answers"])
    probabilities = cast(dict[str, float], answers["part"]["probabilities"])
    return {
        "part_probs": {label: float(probabilities.get(label, 0.0)) for label in PAGE_LABELS},
        "main_sig": float(cast(float, answers["main_sig"]["noul"])),
        "attachment_start": float(cast(float, answers["attachment_start"]["noul"])),
    }


class JudgmentCache:
    """Append-only JSONL cache of judgments keyed by model, prompt version and page state."""

    def __init__(self, path: Path | None) -> None:
        self.path: Path | None = path
        self._entries: dict[str, PageJudgment] = {}
        if path is not None and path.exists():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    record = cast(dict[str, object], json.loads(line))
                    self._entries[str(record["key"])] = cast(PageJudgment, record["judgment"])

    def get(self, key: str) -> PageJudgment | None:
        return self._entries.get(key)

    def put_many(self, items: list[tuple[str, PageJudgment]]) -> None:
        for key, judgment in items:
            self._entries[key] = judgment
        if self.path is None or not items:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for key, judgment in items:
                _ = handle.write(json.dumps({"key": key, "judgment": judgment}) + "\n")


def _request_judgment(session: requests.Session, api_key: str, state: dict[str, str]) -> PageJudgment:
    body = {"model": TYPESAFE_MODEL, "state": state, "questions": QUESTIONS}
    last_error = "no attempt made"
    for attempt in range(TYPESAFE_MAX_ATTEMPTS):
        try:
            response = session.post(
                TYPESAFE_API_URL,
                json=body,
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=120,
            )
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
        else:
            if response.status_code == 200:
                return _parse_judgment(cast(dict[str, object], response.json()))
            last_error = f"HTTP {response.status_code}: {response.text[:300]}"
            if response.status_code not in TYPESAFE_RETRY_STATUSES:
                raise TypeSafeError(f"TypeSafe request failed: {last_error}")
        time.sleep(min(60.0, 2.0**attempt + random.random()))
    raise TypeSafeError(f"TypeSafe request failed after {TYPESAFE_MAX_ATTEMPTS} attempts: {last_error}")


def fetch_page_judgments(
    documents: list[AgreementDocument],
    *,
    cache_path: Path | None = None,
    api_key: str | None = None,
) -> list[list[PageJudgment]]:
    """Return one judgment per page, calling TypeSafe only for pages missing from the cache."""
    cache = JudgmentCache(cache_path)
    keyed_states: list[list[tuple[str, dict[str, str]]]] = []
    missing: dict[str, dict[str, str]] = {}
    for document in documents:
        document_states: list[tuple[str, dict[str, str]]] = []
        for page_index in range(len(document.page_texts)):
            state = build_page_state(document.page_texts, page_index)
            key = _cache_key(state)
            document_states.append((key, state))
            if cache.get(key) is None:
                missing[key] = state
        keyed_states.append(document_states)

    if missing:
        resolved_api_key = api_key or os.getenv("TYPESAFE_API_KEY")
        if not resolved_api_key:
            raise ValueError("TYPESAFE_API_KEY is required for TypeSafe page judgments.")
        keys = list(missing)
        with requests.Session() as session, ThreadPoolExecutor(max_workers=TYPESAFE_CONCURRENCY) as pool:

            def request(key: str) -> PageJudgment:
                return _request_judgment(session, resolved_api_key, missing[key])

            judgments = list(pool.map(request, keys))
        cache.put_many(list(zip(keys, judgments, strict=True)))

    return [
        [cast(PageJudgment, cache.get(key)) for key, _ in document_states]
        for document_states in keyed_states
    ]


def append_typesafe_features(
    sequences: list[list[FeatureDict]],
    judgments: list[list[PageJudgment]],
) -> list[list[FeatureDict]]:
    """Add TypeSafe judgment features to each page's CRF feature dict, in place."""
    for sequence, document_judgments in zip(sequences, judgments, strict=True):
        if len(sequence) != len(document_judgments):
            raise ValueError("Feature sequence and TypeSafe judgments differ in page count.")
        main_sig_seen = 0.0
        for page_index, (feature_dict, judgment) in enumerate(
            zip(sequence, document_judgments, strict=True)
        ):
            for label in PAGE_LABELS:
                feature_dict[f"ts_p_{label}"] = float(judgment["part_probs"].get(label, 0.0))
            feature_dict["ts_main_sig"] = float(judgment["main_sig"])
            feature_dict["ts_att_start"] = float(judgment["attachment_start"])
            feature_dict["ts_prev_main_sig"] = (
                float(document_judgments[page_index - 1]["main_sig"]) if page_index > 0 else 0.0
            )
            feature_dict["ts_next_att_start"] = (
                float(document_judgments[page_index + 1]["attachment_start"])
                if page_index + 1 < len(document_judgments)
                else 0.0
            )
            if judgment["main_sig"] >= MAIN_SIG_SEEN_THRESHOLD:
                main_sig_seen = 1.0
            feature_dict["ts_seen_main_sig"] = main_sig_seen
    return sequences


def typesafe_artifact_metadata() -> dict[str, str]:
    return {"model": TYPESAFE_MODEL, "prompt_version": TYPESAFE_PROMPT_VERSION}
