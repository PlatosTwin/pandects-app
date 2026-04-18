"""
NER (Named Entity Recognition) models and datasets.

This module contains PyTorch Lightning modules and datasets for NER tasks
using transformer models with BIO tagging scheme.
"""

# pyright: reportUnknownVariableType=false, reportUnknownMemberType=false, reportUnknownArgumentType=false, reportAny=false, reportUnnecessaryComparison=false

# Standard library
import os
import re
from typing import Protocol, TYPE_CHECKING, TypedDict, cast
import yaml
from torch.optim import Optimizer
from lightning.pytorch.utilities.types import LRSchedulerConfig
import numpy as np

# Environment config
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ML frameworks & utilities
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchmetrics import F1Score as F1

torch.set_float32_matmul_precision("high")

# Transformers
from transformers import AutoModelForTokenClassification, AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollatorForTokenClassification
from transformers.optimization import get_linear_schedule_with_warmup

if TYPE_CHECKING:
    from .ner_constants import SPECIAL_TOKENS_TO_ADD
else:
    try:
        from .ner_constants import SPECIAL_TOKENS_TO_ADD
    except ImportError:  # pragma: no cover - supports running as a script
        from ner_constants import SPECIAL_TOKENS_TO_ADD


# ASCII-only, length-preserving lowercase (A-Z -> a-z)
_ASCII_LOWER_TBL = str.maketrans({chr(i): chr(i + 32) for i in range(65, 91)})


class _LogitsOutput(Protocol):
    logits: torch.Tensor
    hidden_states: tuple[torch.Tensor, ...] | None


class EvalDocRecord(TypedDict):
    doc_id: int
    raw_text: str
    token_offsets: list[tuple[int, int]]
    gold_tags: list[str]
    pred_tags_raw: list[str]


def ascii_lower(s: str) -> str:
    return s.translate(_ASCII_LOWER_TBL)


def _upgrade_token_head(
    model: nn.Module, num_labels: int, p_drop: float = 0.1, hidden_mult: float = 1.0
) -> nn.Module:
    # Works for most HF token classifiers (including ModernBERT variants)
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Linear):
        in_dim = model.classifier.in_features
        mid = int(in_dim * hidden_mult)
        model.classifier = nn.Sequential(
            nn.Dropout(p_drop),
            nn.Linear(in_dim, mid),
            nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(mid, num_labels),
        )
    return model


def _process_document(
    raw: str,
    tokenizer: PreTrainedTokenizerBase,
    label2id: dict[str, int],
    preserve_case: bool = False,
) -> dict[str, object]:
    tag_pattern = re.compile(r"<(section|article|page)>(.*?)</\1>", re.DOTALL)

    # 1) Build cleaned_text while tracking spans in cleaned-text coordinates
    parts: list[str] = []
    spans: list[tuple[int, int, str]] = []  # (start_in_cleaned, end_in_cleaned, tag)
    src_pos = 0
    out_len = 0

    for m in tag_pattern.finditer(raw):
        # text before tag
        pre = raw[src_pos : m.start()]
        parts.append(pre)
        out_len += len(pre)

        tag = m.group(1)
        content = m.group(2)

        # record span in cleaned-text coords
        span_start = out_len
        parts.append(content)
        out_len += len(content)
        span_end = out_len
        spans.append((span_start, span_end, tag))

        src_pos = m.end()

    # tail after last tag
    parts.append(raw[src_pos:])
    cleaned_text = "".join(parts)

    # 2) Tokenize without specials using ASCII-only lowercase mirror; get offsets & word_ids
    norm = cleaned_text if preserve_case else ascii_lower(cleaned_text)
    encoding = tokenizer(
        norm,
        return_offsets_mapping=True,
        truncation=False,
        add_special_tokens=False,
    )
    offsets = cast(list[tuple[int, int]], encoding["offset_mapping"])
    word_ids = encoding.word_ids()
    if word_ids is None:
        raise RuntimeError(
            "Tokenizer did not return word_ids; ensure a fast tokenizer is used."
        )

    encoding_dict = cast(dict[str, object], cast(object, encoding))

    # 3) Locate first token of each word (for word-level labeling & later span logic)
    first_token_idx: list[int] = []
    seen_words = set()
    for i, wid in enumerate(word_ids):
        if wid is None:
            continue
        if wid not in seen_words:
            first_token_idx.append(i)
            seen_words.add(wid)

    # 4) Initialize labels to 'O'
    o_id = label2id["O"]
    labels = [o_id] * len(offsets)

    # 5) Apply BIOES to the FIRST token of each word overlapping each char span
    for c0, c1, tag in spans:
        wid_to_first_tok: dict[int, int] = {}
        for i, wid in enumerate(word_ids):
            if wid is None:
                continue
            t0, t1 = offsets[i]
            if max(t0, c0) < min(t1, c1):  # overlap
                if wid not in wid_to_first_tok:
                    wid_to_first_tok[wid] = i

        if not wid_to_first_tok:
            continue

        ordered_first_toks = sorted(wid_to_first_tok.values())
        if len(ordered_first_toks) == 1:
            if tag in {"section", "article"}:
                raise ValueError(
                    f"Single-token {tag} span found; S-{tag.upper()} is not supported."
                )
            labels[ordered_first_toks[0]] = label2id[f"S-{tag.upper()}"]
        else:
            labels[ordered_first_toks[0]] = label2id[f"B-{tag.upper()}"]
            labels[ordered_first_toks[-1]] = label2id[f"E-{tag.upper()}"]
            for idx in ordered_first_toks[1:-1]:
                labels[idx] = label2id[f"I-{tag.upper()}"]

    # 6) Mask non-first-subword tokens as -100; keep track of first-token indices
    final_labels: list[int] = []
    last_wid = None
    for i, wid in enumerate(word_ids):
        if wid is None:
            final_labels.append(-100)
        elif wid != last_wid:
            final_labels.append(labels[i])
        else:
            final_labels.append(-100)
        last_wid = wid

    encoding_dict["labels"] = final_labels
    encoding_dict["cleaned_text"] = cleaned_text
    encoding_dict["first_token_idx"] = first_token_idx
    return encoding_dict


def _singleton_tag(typ: str) -> str:
    # We intentionally do NOT support S-ARTICLE / S-SECTION
    if typ in {"ARTICLE", "SECTION"}:
        return f"E-{typ}"  # close immediately
    return f"S-{typ}"


def repair_bioes(tags: list[str]) -> list[str]:
    """
    Make a best-effort repair of illegal BIOES sequences.
    - Lone I-* without a preceding B-*: -> B-*
    - E-* without open entity: -> S-*, unless Article or Section,
        then E (ideally never happens)
    - B-* followed directly by O or B-*: -> S-* (close immediately),
        unless Article or Section, then E (ideally never happens)
    - I-* followed by O/B-*: -> E-* (close)
    """
    repaired = tags[:]
    open_type = None
    for i, t in enumerate(repaired):
        if t == "O":
            if open_type is not None:
                # close previous entity
                repaired[i - 1] = f"E-{open_type}"
                open_type = None
            continue

        pref = t[0]  # B/I/E/S
        typ = t[2:] if len(t) > 2 else ""
        if pref == "B":
            if open_type is not None:
                repaired[i - 1] = f"E-{open_type}"
            # peek next
            nxt = repaired[i + 1] if i + 1 < len(repaired) else "O"
            if nxt.startswith(("O", "B", "S")):
                repaired[i] = _singleton_tag(typ)
                open_type = None
            else:
                open_type = typ

        elif pref == "I":
            if open_type is None or typ != open_type:
                # convert to B
                repaired[i] = f"B-{typ}"
                # same logic as B for immediate close
                nxt = repaired[i + 1] if i + 1 < len(repaired) else "O"
                if nxt.startswith(("O", "B", "S")):
                    repaired[i] = _singleton_tag(typ)
                    open_type = None
                else:
                    open_type = typ
            else:
                # potentially close if next breaks
                nxt = repaired[i + 1] if i + 1 < len(repaired) else "O"
                if nxt.startswith(("O", "B", "S")):
                    repaired[i] = f"E-{typ}"
                    open_type = None

        elif pref == "E":
            if open_type is None or typ != open_type:
                repaired[i] = _singleton_tag(typ)
                open_type = None
            else:
                open_type = None

        elif pref == "S":
            if open_type is not None:
                repaired[i - 1] = f"E-{open_type}"
                open_type = None
            # S-* is already a closed single, nothing to do
        else:
            repaired[i] = "O"

    if open_type is not None:
        # close trailing open entity
        repaired[-1] = f"E-{open_type}"
    return repaired


def tags_to_spans(tags: list[str]) -> list[tuple[int, int, str]]:
    """
    Convert BIOES tags to spans as (start_idx, end_idx, type), inclusive.
    Assumes tags have been repaired (legal BIOES).
    """
    spans = []
    cur_start, cur_type = None, None
    for i, t in enumerate(tags):
        if t == "O":
            continue
        pref = t[0]
        typ = t[2:]
        if pref == "S":
            spans.append((i, i, typ))
        elif pref == "B":
            cur_start, cur_type = i, typ
        elif pref == "I":
            # continuing
            pass
        elif pref == "E":
            if cur_start is None or cur_type != typ:
                # fallback: treat as single
                spans.append((i, i, typ))
            else:
                spans.append((cur_start, i, typ))
            cur_start, cur_type = None, None
    return spans


def spans_to_tags(spans: list[tuple[int, int, str]], length: int) -> list[str]:
    """
    Convert spans back into BIOES tags of the requested length.
    """
    tags = ["O"] * length
    for start, end, typ in spans:
        if start < 0 or end >= length or end < start:
            raise ValueError("Span bounds are invalid for tag conversion.")
        if start == end:
            tags[start] = _singleton_tag(typ)
            continue
        tags[start] = f"B-{typ}"
        for i in range(start + 1, end):
            tags[i] = f"I-{typ}"
        tags[end] = f"E-{typ}"
    return tags


def prf1_from_spans(
    pred_spans: list[tuple[int, int, str]],
    gold_spans: list[tuple[int, int, str]],
) -> tuple[int, int, int]:
    """
    Exact-match micro: returns (tp, fp, fn).
    """
    pred_set = set(pred_spans)
    gold_set = set(gold_spans)
    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def prf1_from_spans_lenient(
    pred_spans: list[tuple[int, int, str]],
    gold_spans: list[tuple[int, int, str]],
) -> tuple[int, int, int]:
    """
    Lenient micro: overlap match within same entity type, greedy 1-to-1.
    """
    pred_by_type: dict[str, list[tuple[int, int]]] = {}
    gold_by_type: dict[str, list[tuple[int, int]]] = {}
    for s, e, t in pred_spans:
        pred_by_type.setdefault(t, []).append((s, e))
    for s, e, t in gold_spans:
        gold_by_type.setdefault(t, []).append((s, e))

    def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return not (a[1] < b[0] or b[1] < a[0])

    tp = fp = fn = 0
    for ent_type in set(pred_by_type) | set(gold_by_type):
        preds = pred_by_type.get(ent_type, [])
        golds = gold_by_type.get(ent_type, [])
        unmatched = golds[:]
        for p in preds:
            match_idx = None
            for i, g in enumerate(unmatched):
                if _overlaps(p, g):
                    match_idx = i
                    break
            if match_idx is None:
                fp += 1
            else:
                tp += 1
                _ = unmatched.pop(match_idx)
        fn += len(unmatched)
    return tp, fp, fn


def _bootstrap_entity_f1_ci(
    doc_spans: list[tuple[list[tuple[int, int, str]], list[tuple[int, int, str]]]],
    n_bootstrap: int = 2000,
    ci_level: float = 0.95,
    seed: int = 0,
) -> dict[str, object]:
    """Bootstrap CI for entity-level F1 by resampling docs with replacement."""
    if len(doc_spans) < 2:
        return {}
    rng = np.random.default_rng(seed)
    n = len(doc_spans)

    all_types: set[str] = set()
    for pred_s, gold_s in doc_spans:
        for _, _, t in pred_s:
            all_types.add(t)
        for _, _, t in gold_s:
            all_types.add(t)

    def _f1(tp_v: int, fp_v: int, fn_v: int) -> float:
        p = tp_v / (tp_v + fp_v) if tp_v + fp_v else 0.0
        r = tp_v / (tp_v + fn_v) if tp_v + fn_v else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    micro_f1s: list[float] = []
    type_f1s: dict[str, list[float]] = {t: [] for t in all_types}

    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        tp = fp = fn = 0
        tpc: dict[str, int] = {}
        fpc: dict[str, int] = {}
        fnc: dict[str, int] = {}
        for idx in indices:
            pred_set = set(doc_spans[int(idx)][0])
            gold_set = set(doc_spans[int(idx)][1])
            matched = pred_set & gold_set
            false_pos = pred_set - gold_set
            false_neg = gold_set - pred_set
            tp += len(matched)
            fp += len(false_pos)
            fn += len(false_neg)
            for _, _, t in matched:
                tpc[t] = tpc.get(t, 0) + 1
            for _, _, t in false_pos:
                fpc[t] = fpc.get(t, 0) + 1
            for _, _, t in false_neg:
                fnc[t] = fnc.get(t, 0) + 1
        micro_f1s.append(_f1(tp, fp, fn))
        for t in all_types:
            type_f1s[t].append(_f1(tpc.get(t, 0), fpc.get(t, 0), fnc.get(t, 0)))

    alpha = 1.0 - ci_level
    lo_p = 100.0 * alpha / 2
    hi_p = 100.0 * (1.0 - alpha / 2)
    return {
        "n_docs": n,
        "n_bootstrap": n_bootstrap,
        "ci_level": ci_level,
        "micro": {
            "f1_lo": float(np.percentile(micro_f1s, lo_p)),
            "f1_hi": float(np.percentile(micro_f1s, hi_p)),
        },
        "per_type": {
            t: {
                "f1_lo": float(np.percentile(type_f1s[t], lo_p)),
                "f1_hi": float(np.percentile(type_f1s[t], hi_p)),
            }
            for t in sorted(all_types)
        },
    }


def _parse_bioes(tag: str) -> tuple[str, str | None]:
    # "O" -> ("O", None), "B-FOO" -> ("B","FOO"), etc.
    if tag == "O":
        return "O", None
    if "-" not in tag:
        return "O", None
    p, t = tag.split("-", 1)
    return p, t


def _legal_bioes(
    prev_tag: str, prev_type: str | None, curr_tag: str, curr_type: str | None
) -> bool:
    # BIOES legality with type agreement for inside segments
    if prev_tag == "O":
        return curr_tag in ("O", "B", "S")
    if prev_tag == "B":
        return (curr_tag, curr_type) in (("I", prev_type), ("E", prev_type))
    if prev_tag == "I":
        return (curr_tag, curr_type) in (("I", prev_type), ("E", prev_type))
    if prev_tag == "E":
        return curr_tag in ("O", "B", "S")
    if prev_tag == "S":
        return curr_tag in ("O", "B", "S")
    return curr_tag in ("O", "B", "S")  # fallback


def build_bioes_constraints(
    id2label: dict[int, str],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (trans, start, end):
      trans: [C,C] add to emissions[t] when transitioning prev->curr
      start: [C]   add to emissions[0]
      end:   [C]   add at the last step
    Legal = 0.0, illegal = large negative (not -inf to avoid NaNs).
    """
    C = len(id2label)
    NEG = -1e4
    trans = torch.full((C, C), NEG)
    start = torch.full((C,), NEG)
    end = torch.full((C,), NEG)

    # pre-parse
    parts = {i: _parse_bioes(lbl) for i, lbl in id2label.items()}

    # transitions
    for i in range(C):
        pi, ti = parts[i]
        for j in range(C):
            pj, tj = parts[j]
            if _legal_bioes(pi, ti, pj, tj):
                trans[i, j] = 0.0

    # start: cannot start with I/E
    for j in range(C):
        pj, _ = parts[j]
        if pj in ("O", "B", "S"):
            start[j] = 0.0

    # end: should not end on B/I (open entity)
    for i in range(C):
        pi, _ = parts[i]
        if pi in ("O", "E", "S"):
            end[i] = 0.0

    return trans, start, end


_ARTICLE_HEADING_PATTERNS = [
    re.compile(r"^\s*ARTICLE\s+[IVXLCDM]+[\.,:;\-\u2014]?\s*(.*)?$", re.IGNORECASE),
    re.compile(r"^\s*ARTICLE\s+\d+[\.,:;\-\u2014]?\s*(.*)?$", re.IGNORECASE),
]
_SECTION_HEADING_PATTERNS = [
    re.compile(r"^\s*SECTION\s+\d+(?:\.\d+)*[\.,:;\-\u2014]?\s*(.*)?$", re.IGNORECASE),
    re.compile(r"^\s*SECTION\s+[A-Z][\.,:;\-\u2014]?\s*(.*)?$", re.IGNORECASE),
]


def _window_bounds(center: int, window: int, num_tokens: int) -> tuple[int, int]:
    if num_tokens <= window:
        return 0, num_tokens
    half = window // 2
    start = max(0, min(center - half, num_tokens - window))
    end = min(start + window, num_tokens)
    return start, end


def _sample_candidates(
    rng: np.random.Generator,
    candidates: list[tuple[int, int]],
    count: int,
) -> list[tuple[int, int]]:
    if count <= 0 or not candidates:
        return []
    if len(candidates) <= count:
        return candidates[:]
    idx = rng.choice(len(candidates), size=count, replace=False)
    return [candidates[int(i)] for i in idx]


def _line_text_matches_heading(line_text: str) -> bool:
    return any(p.match(line_text) for p in _ARTICLE_HEADING_PATTERNS) or any(
        p.match(line_text) for p in _SECTION_HEADING_PATTERNS
    )


class TrainDataset(Dataset[dict[str, object]]):
    """
    Training dataset for NER with token-based sub-sampling strategies.
    This version includes the fix for the off-by-one error in span collection.
    """

    def __init__(
        self,
        data: list[str],
        tokenizer: PreTrainedTokenizerBase,
        label2id: dict[str, int],
        subsample_window: int,
        sampling_mode: str = "boundary_mix",
        seed: int = 42,
        preserve_case: bool = False,
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.id2label = {idx: label for label, idx in label2id.items()}
        self.subsample_window = subsample_window
        self.sampling_mode = sampling_mode
        self.seed = seed
        self.preserve_case = preserve_case
        self.examples: list[dict[str, object]] = []

        for doc_idx, raw in enumerate(data):
            processed_doc = _process_document(
                raw,
                self.tokenizer,
                self.label2id,
                preserve_case=self.preserve_case,
            )
            rng = np.random.default_rng(seed + doc_idx)
            self._create_samples_from_doc(processed_doc, rng)

    def _entity_spans_for_doc(
        self,
        labels: list[int],
        num_tokens: int,
    ) -> list[tuple[int, int]]:
        o_id = self.label2id["O"]
        first_idxs = [i for i, lab in enumerate(labels) if lab != -100]
        is_ent_word = [labels[i] != o_id for i in first_idxs]
        entity_spans: list[tuple[int, int]] = []
        open_k: int | None = None
        for k, flag in enumerate(is_ent_word):
            if flag and open_k is None:
                open_k = k
            elif (not flag) and (open_k is not None):
                last_k = k - 1
                start_tok = first_idxs[open_k]
                end_tok = (
                    (first_idxs[last_k + 1] - 1)
                    if (last_k + 1 < len(first_idxs))
                    else (num_tokens - 1)
                )
                entity_spans.append((start_tok, end_tok))
                open_k = None
        if open_k is not None:
            entity_spans.append((first_idxs[open_k], num_tokens - 1))
        return entity_spans

    def _boundary_candidates(
        self,
        labels: list[int],
        num_tokens: int,
        rng: np.random.Generator,
    ) -> list[tuple[int, int]]:
        candidates: list[tuple[int, int]] = []
        jitter_max = 64
        target_prefixes = {
            "B-ARTICLE",
            "E-ARTICLE",
            "B-SECTION",
            "E-SECTION",
        }
        for idx, lab_id in enumerate(labels):
            if lab_id == -100:
                continue
            label = self.id2label[lab_id]
            if label not in target_prefixes:
                continue
            jitter = int(rng.integers(-jitter_max, jitter_max + 1))
            start, end = _window_bounds(idx + jitter, self.subsample_window, num_tokens)
            candidates.append((start, end))
        return candidates

    def _hard_negative_candidates(
        self,
        processed_doc: dict[str, object],
        labels: list[int],
        num_tokens: int,
    ) -> list[tuple[int, int]]:
        offsets = cast(list[tuple[int, int]], processed_doc["offset_mapping"])
        cleaned_text = cast(str, processed_doc["cleaned_text"])
        candidates: list[tuple[int, int]] = []
        search_start = 0
        for line in cleaned_text.splitlines():
            if not _line_text_matches_heading(line):
                search_start += len(line) + 1
                continue
            line_start = cleaned_text.find(line, search_start)
            if line_start == -1:
                search_start += len(line) + 1
                continue
            line_end = line_start + len(line)
            search_start = line_end + 1
            overlapping = [
                i
                for i, (start, end) in enumerate(offsets)
                if start < line_end and end > line_start
            ]
            if not overlapping:
                continue
            center = overlapping[len(overlapping) // 2]
            start, end = _window_bounds(center, self.subsample_window, num_tokens)
            if any(label != self.label2id["O"] for label in labels[start:end] if label != -100):
                continue
            candidates.append((start, end))
        return candidates

    def _random_candidates(
        self,
        num_tokens: int,
        rng: np.random.Generator,
        count: int,
    ) -> list[tuple[int, int]]:
        if num_tokens <= 0 or count <= 0:
            return []
        max_start = max(1, num_tokens - self.subsample_window + 1)
        starts = [int(rng.integers(0, max_start)) for _ in range(count)]
        return [(start, min(start + self.subsample_window, num_tokens)) for start in starts]

    def _store_unique_windows(
        self,
        input_ids: list[int],
        labels: list[int],
        windows: list[tuple[int, int]],
    ) -> None:
        seen: set[tuple[int, int]] = set()
        for start, end in windows:
            key = (max(0, start), min(end, len(input_ids)))
            if key in seen or key[0] >= key[1]:
                continue
            seen.add(key)
            self._store_chunk(input_ids[key[0] : key[1]], labels[key[0] : key[1]])

    def _create_boundary_mix_samples(
        self,
        processed_doc: dict[str, object],
        input_ids: list[int],
        labels: list[int],
        entity_spans: list[tuple[int, int]],
        rng: np.random.Generator,
    ) -> None:
        num_tokens = len(input_ids)
        entity_windows: list[tuple[int, int]] = []
        for start, end in entity_spans:
            center = (start + end) // 2
            entity_windows.append(_window_bounds(center, self.subsample_window, num_tokens))

        boundary_windows = self._boundary_candidates(labels, num_tokens, rng)
        hard_negative_windows = self._hard_negative_candidates(processed_doc, labels, num_tokens)
        random_windows = self._random_candidates(num_tokens, rng, count=2)

        base_count = max(len(entity_windows), 1)
        total_target = max(4, int(np.ceil(base_count / 0.4)))
        entity_target = min(len(entity_windows), max(1, int(round(total_target * 0.4))))
        boundary_target = max(1, int(round(total_target * 0.3)))
        hard_negative_target = max(1, int(round(total_target * 0.2)))
        random_target = max(1, total_target - entity_target - boundary_target - hard_negative_target)

        windows = []
        windows.extend(_sample_candidates(rng, entity_windows, entity_target))
        windows.extend(_sample_candidates(rng, boundary_windows, boundary_target))
        windows.extend(_sample_candidates(rng, hard_negative_windows, hard_negative_target))
        windows.extend(_sample_candidates(rng, random_windows, random_target))

        if not windows:
            windows.extend(self._random_candidates(num_tokens, rng, count=2))

        self._store_unique_windows(input_ids, labels, windows)

    def _create_samples_from_doc(
        self, processed_doc: dict[str, object], rng: np.random.Generator
    ) -> None:
        input_ids = cast(list[int], processed_doc["input_ids"])
        labels = cast(list[int], processed_doc["labels"])
        num_tokens = len(input_ids)
        entity_spans = self._entity_spans_for_doc(labels, num_tokens)

        if self.sampling_mode != "boundary_mix":
            raise ValueError(f"Unsupported sampling_mode {self.sampling_mode!r}.")
        self._create_boundary_mix_samples(
            processed_doc, input_ids, labels, entity_spans, rng
        )

    def _store_chunk(self, id_chunk: list[int], label_chunk: list[int]) -> None:
        """Adds special tokens and stores a training example."""
        if self.tokenizer.cls_token_id is None or self.tokenizer.sep_token_id is None:
            raise RuntimeError("Tokenizer is missing CLS/SEP token ids.")
        input_ids = (
            [self.tokenizer.cls_token_id] + id_chunk + [self.tokenizer.sep_token_id]
        )
        labels = [-100] + label_chunk + [-100]
        attention_mask = [1] * len(input_ids)

        self.examples.append(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
            }
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.examples[idx]


class ValWindowedDataset(Dataset[dict[str, object]]):
    """
    Sliding-window NER dataset for validation, operating on tokens.
    """

    def __init__(
        self,
        data: list[str],
        tokenizer: PreTrainedTokenizerBase,
        label2id: dict[str, int],
        data_collator: DataCollatorForTokenClassification,
        window: int = 510,
        stride: int = 256,
        preserve_case: bool = False,
    ):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.collator = data_collator
        self.window = window
        self.stride = stride
        self.preserve_case = preserve_case
        self.examples: list[dict[str, object]] = []

        for doc_id, raw in enumerate(data):
            processed_doc = _process_document(
                raw,
                self.tokenizer,
                self.label2id,
                preserve_case=self.preserve_case,
            )

            input_ids = cast(list[int], processed_doc["input_ids"])
            labels = cast(list[int], processed_doc["labels"])
            offset_mapping = cast(
                list[tuple[int, int]], processed_doc["offset_mapping"]
            )
            cleaned_text = cast(str, processed_doc["cleaned_text"])
            num_tokens = len(input_ids)

            for start in range(0, num_tokens, self.stride):
                end = min(start + self.window, num_tokens)

                id_chunk = input_ids[start:end]
                label_chunk = labels[start:end]
                offset_chunk = offset_mapping[start:end]

                if (
                    self.tokenizer.cls_token_id is None
                    or self.tokenizer.sep_token_id is None
                ):
                    raise RuntimeError("Tokenizer is missing CLS/SEP token ids.")
                final_ids = (
                    [self.tokenizer.cls_token_id]
                    + id_chunk
                    + [self.tokenizer.sep_token_id]
                )
                final_labels = [-100] + label_chunk + [-100]
                final_offsets = [(0, 0)] + offset_chunk + [(0, 0)]

                self.examples.append(
                    {
                        "doc_id": doc_id,
                        "window_start": start,
                        "input_ids": final_ids,
                        "attention_mask": [1] * len(final_ids),
                        "labels": final_labels,
                        "offset_mapping": final_offsets,
                        "raw": cleaned_text,
                    }
                )

                if end == num_tokens:
                    break

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return dict(self.examples[idx])

    def collate_fn(self, features: list[dict[str, object]]) -> dict[str, object]:
        doc_ids = [f.pop("doc_id") for f in features]
        starts = [f.pop("window_start") for f in features]
        offsets = [f.pop("offset_mapping") for f in features]
        raw = [f.pop("raw") for f in features]

        batch = self.collator(features)

        batch["doc_id"] = torch.tensor(doc_ids, dtype=torch.long)
        batch["window_start"] = torch.tensor(starts, dtype=torch.long)
        batch["offset_mapping"] = offsets
        batch["raw"] = raw

        return batch


class NERDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for NER task.

    Handles train/val dataset creation and dataloaders.
    """

    def __init__(
        self,
        train_data: list[str],
        val_data: list[str],
        test_data: list[str] | None,
        tokenizer_name: str,
        label_list: list[str],
        batch_size: int,
        train_subsample_window: int,
        num_workers: int,
        sampling_mode: str = "boundary_mix",
        seed: int = 42,
        val_window: int = 510,
        val_stride: int = 256,
        preserve_case: bool = False,
    ):
        """
        Initialize the NER data module.

        Args:
            train_data: Training data strings
            val_data: Validation data strings
            tokenizer_name: HuggingFace tokenizer name
            label_list: List of label names
            batch_size: Batch size for training
            train_subsample_window: Window size for training sub-sampling
            num_workers: Number of data loading workers
            val_window: Window size for validation
            val_stride: Stride size for validation
        """
        super().__init__()
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_dataset: TrainDataset | None = None
        self.val_dataset: ValWindowedDataset | None = None
        self.test_dataset: ValWindowedDataset | None = None
        self.pin_memory = torch.cuda.is_available()
        self.persistent_workers = num_workers > 0

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS_TO_ADD}
        )

        self.train_subsample_window = train_subsample_window
        self.sampling_mode = sampling_mode
        self.seed = seed
        self.val_window = val_window
        self.val_stride = val_stride
        self.preserve_case = preserve_case

        self.label2id = {label: idx for idx, label in enumerate(label_list)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}

        self.data_collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer, label_pad_token_id=-100
        )

    class _LoaderKwargs(TypedDict, total=False):
        num_workers: int
        pin_memory: bool
        persistent_workers: bool
        prefetch_factor: int

    def _loader_kwargs(self) -> _LoaderKwargs:
        kwargs: NERDataModule._LoaderKwargs = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
        }
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = 4 if self.num_workers >= 4 else 2
        return kwargs

    def setup(self, stage: str | None = None) -> None:
        """
        Set up datasets for training and validation.

        Args:
            stage: Lightning stage
        """
        self.train_dataset = TrainDataset(
            data=self.train_data,
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            subsample_window=self.train_subsample_window,
            sampling_mode=self.sampling_mode,
            seed=self.seed,
            preserve_case=self.preserve_case,
        )
        self.val_dataset = ValWindowedDataset(
            data=self.val_data,
            tokenizer=self.tokenizer,
            label2id=self.label2id,
            data_collator=self.data_collator,
            window=self.val_window,
            stride=self.val_stride,
            preserve_case=self.preserve_case,
        )
        if self.test_data is not None:
            self.test_dataset = ValWindowedDataset(
                data=self.test_data,
                tokenizer=self.tokenizer,
                label2id=self.label2id,
                data_collator=self.data_collator,
                window=self.val_window,
                stride=self.val_stride,
                preserve_case=self.preserve_case,
            )

    def train_dataloader(self) -> DataLoader[dict[str, object]]:
        """Returns DataLoader for training set."""
        if self.train_dataset is None:
            raise RuntimeError("Train dataset not initialized. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.data_collator,
            **self._loader_kwargs(),
        )

    def val_dataloader(self) -> DataLoader[dict[str, object]]:
        """Returns DataLoader for validation set."""
        if self.val_dataset is None:
            raise RuntimeError(
                "Validation dataset not initialized. Call setup() first."
            )
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.val_dataset.collate_fn,
            **self._loader_kwargs(),
        )

    def test_dataloader(self) -> DataLoader[dict[str, object]]:
        """Returns DataLoader for test set."""
        if self.test_dataset is None:
            raise RuntimeError("Test dataset not initialized. Call setup() first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.test_dataset.collate_fn,
            **self._loader_kwargs(),
        )


class FocalLoss(torch.nn.Module):
    """
    Focal loss for class imbalance in NER.

    Addresses class imbalance by down-weighting easy examples.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        ignore_index: int = -100,
        class_weights: torch.Tensor | None = None,
    ):
        """
        Initialize focal loss.

        Args:
            gamma: Focusing parameter
            ignore_index: Index to ignore in loss computation
            class_weights: Optional per-class weights aligned to label ids
        """
        super().__init__()
        self.gamma = gamma
        self.ignore = ignore_index
        self.ce = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=ignore_index)
        self.class_weights: torch.Tensor | None
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            logits: Model logits (B, T, C) or (N, C)
            labels: Target labels (B, T) or (N,)

        Returns:
            Scalar loss value
        """
        if logits.ndim == 3:
            logits = logits.reshape(-1, logits.shape[-1])
            labels = labels.reshape(-1)

        active = labels != self.ignore
        if not active.any():
            return logits.sum() * 0.0

        logits_active = logits[active].float()
        labels_active = labels[active]

        ce_loss = F.cross_entropy(
            logits_active,
            labels_active,
            reduction="none",
        )
        pt = torch.softmax(logits_active, dim=-1).gather(
            dim=1, index=labels_active.unsqueeze(1)
        ).squeeze(1)

        if self.class_weights is not None:
            weights = self.class_weights
            if weights.device != labels_active.device:
                weights = weights.to(labels_active.device)
            ce_loss = ce_loss * weights[labels_active]

        focal = ((1.0 - pt).clamp(min=0.0) ** self.gamma) * ce_loss
        return focal.mean()


class ConstrainedLinearChainCRF(nn.Module):
    """
    Linear-chain CRF with hard BIOES legality masks.
    """

    def __init__(
        self,
        transition_mask: torch.Tensor,
        start_mask: torch.Tensor,
        end_mask: torch.Tensor,
    ) -> None:
        super().__init__()
        self.transitions = nn.Parameter(torch.zeros_like(transition_mask))
        self.start_transitions = nn.Parameter(torch.zeros_like(start_mask))
        self.end_transitions = nn.Parameter(torch.zeros_like(end_mask))
        self.transition_mask: torch.Tensor
        self.start_mask: torch.Tensor
        self.end_mask: torch.Tensor
        self.register_buffer("transition_mask", transition_mask)
        self.register_buffer("start_mask", start_mask)
        self.register_buffer("end_mask", end_mask)

    def _masked_transition_scores(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.transitions + self.transition_mask,
            self.start_transitions + self.start_mask,
            self.end_transitions + self.end_mask,
        )

    def neg_log_likelihood(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if emissions.ndim != 3:
            raise ValueError("emissions must have shape [B, T, C].")
        if tags.ndim != 2 or mask.ndim != 2:
            raise ValueError("tags and mask must have shape [B, T].")
        if emissions.size(0) == 0:
            return emissions.sum() * 0.0

        transitions, start_scores, end_scores = self._masked_transition_scores()
        batch_size, seq_len, _ = emissions.shape
        device = emissions.device
        lengths = mask.long().sum(dim=1)
        if (lengths == 0).any():
            raise ValueError("CRF sequences must contain at least one active token.")

        batch_idx = torch.arange(batch_size, device=device)
        score = start_scores[tags[:, 0]] + emissions[batch_idx, 0, tags[:, 0]]
        for t in range(1, seq_len):
            active = mask[:, t]
            if not active.any():
                continue
            prev_tags = tags[:, t - 1]
            curr_tags = tags[:, t]
            trans_score = transitions[prev_tags, curr_tags]
            emit_score = emissions[batch_idx, t, curr_tags]
            score = score + (trans_score + emit_score) * active

        last_positions = lengths - 1
        last_tags = tags[batch_idx, last_positions]
        score = score + end_scores[last_tags]

        alpha = start_scores.unsqueeze(0) + emissions[:, 0]
        for t in range(1, seq_len):
            step_scores = (
                alpha.unsqueeze(2)
                + transitions.unsqueeze(0)
                + emissions[:, t].unsqueeze(1)
            )
            next_alpha = torch.logsumexp(step_scores, dim=1)
            alpha = torch.where(mask[:, t].unsqueeze(1), next_alpha, alpha)

        log_z = torch.logsumexp(alpha + end_scores.unsqueeze(0), dim=1)
        return (log_z - score).mean()

    def decode(self, emissions: torch.Tensor) -> torch.Tensor:
        if emissions.ndim != 2:
            raise ValueError("emissions must have shape [T, C] for decode.")
        transitions, start_scores, end_scores = self._masked_transition_scores()
        T, C = emissions.shape
        score = start_scores + emissions[0]
        backp = torch.zeros((T, C), dtype=torch.long, device=emissions.device)

        for t in range(1, T):
            prev = score.unsqueeze(1) + transitions
            best_prev, best_idx = prev.max(dim=0)
            score = best_prev + emissions[t]
            backp[t] = best_idx

        score = score + end_scores
        last = int(torch.argmax(score))
        path = torch.empty((T,), dtype=torch.long, device=emissions.device)
        path[-1] = last
        for t in range(T - 1, 0, -1):
            path[t - 1] = backp[t, path[t]]
        return path


class NERTagger(pl.LightningModule):
    """
    LightningModule for NER with token-level validation stitching (BIOES-aware).
    """

    def __init__(
        self,
        model_name: str,
        num_labels: int,
        id2label: dict[int, str],
        learning_rate: float,
        weight_decay: float,
        warmup_steps_pct: float,
        default_class_weight: float = 1.0,
        decoder_mode: str = "independent",
        token_loss_mode: str = "focal",
        token_loss_weight: float = 1.0,
        crf_loss_weight: float = 0.0,
        boundary_head: bool = False,
        boundary_loss_weight: float = 0.0,
        label_smoothing: float = 0.0,
        preserve_case: bool = False,
        metrics_output_dir: str | None = None,
        metrics_output_name: str = "ner_test_metrics.yaml",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.warmup_steps_pct = float(warmup_steps_pct)
        self.num_labels = num_labels
        self.id2label = id2label
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.decoder_mode = decoder_mode
        self.token_loss_mode = token_loss_mode
        self.token_loss_weight = float(token_loss_weight)
        self.crf_loss_weight = float(crf_loss_weight)
        self.boundary_head_enabled = bool(boundary_head)
        self.boundary_loss_weight = float(boundary_loss_weight)
        self.label_smoothing = float(label_smoothing)
        self.preserve_case = bool(preserve_case)
        self.use_crf = self.decoder_mode == "crf"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS_TO_ADD}
        )

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
        )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.train()
        _ = _upgrade_token_head(
            self.model, self.num_labels, p_drop=0.1, hidden_mult=1.0
        )

        self.ignore_index = -100
        class_weight_map: dict[str, float] = {}
        class_weights: list[float] = []
        for idx in range(self.num_labels):
            label_name = self.id2label[idx]
            if self.token_loss_mode == "ce":
                if label_name in {"B-ARTICLE", "E-ARTICLE"}:
                    weight = 4.0 * float(default_class_weight)
                elif label_name == "I-ARTICLE":
                    weight = 2.0 * float(default_class_weight)
                elif label_name in {"B-SECTION", "E-SECTION"}:
                    weight = 2.0 * float(default_class_weight)
                elif label_name == "I-SECTION":
                    weight = 1.25 * float(default_class_weight)
                else:
                    weight = float(default_class_weight)
            else:
                weight = float(default_class_weight)
            class_weight_map[label_name] = weight
            class_weights.append(weight)
        class_weight_tensor = torch.tensor(class_weights, dtype=torch.float32)
        print(
            "NER class weights: "
            + ", ".join(
                f"{label} -> {weight:g}" for label, weight in class_weight_map.items()
            )
        )
        self.class_weight_map = class_weight_map
        self.focal_loss_fn = FocalLoss(
            gamma=2.0,
            ignore_index=self.ignore_index,
            class_weights=class_weight_tensor,
        )
        self.token_class_weights: torch.Tensor
        self.register_buffer("token_class_weights", class_weight_tensor)

        # CRF-style constraints (buffers, move with .to(device), saved in state_dict)
        _trans, _start, _end = build_bioes_constraints(self.id2label)
        self._crf_trans: torch.Tensor
        self._crf_start: torch.Tensor
        self._crf_end: torch.Tensor
        self.register_buffer("_crf_trans", _trans)  # [C,C]
        self.register_buffer("_crf_start", _start)  # [C]
        self.register_buffer("_crf_end", _end)  # [C]
        self.crf: ConstrainedLinearChainCRF | None = None
        if self.use_crf:
            self.crf = ConstrainedLinearChainCRF(_trans, _start, _end)

        hidden_size = int(getattr(self.model.config, "hidden_size"))
        self.boundary_classifier: nn.Linear | None = None
        if self.boundary_head_enabled:
            self.boundary_classifier = nn.Linear(hidden_size, 4)
        boundary_pos_weight = torch.tensor([6.0, 6.0, 3.0, 3.0], dtype=torch.float32)
        self.boundary_pos_weight: torch.Tensor
        self.register_buffer("boundary_pos_weight", boundary_pos_weight)
        self._boundary_label_map = {
            "B-ARTICLE": 0,
            "E-ARTICLE": 1,
            "B-SECTION": 2,
            "E-SECTION": 3,
        }

        # --- Metrics (micro) ---
        o_id = self.label2id["O"]

        # Training
        self.train_f1 = F1(
            task="multiclass",
            num_classes=num_labels,
            average="micro",
        )
        self.train_f1_no_o = F1(
            task="multiclass",
            num_classes=num_labels,
            average="micro",
            ignore_index=o_id,
        )

        # Sample-level validation
        self.val_f1_no_o = F1(
            task="multiclass",
            num_classes=num_labels,
            average="micro",
            ignore_index=o_id,
        )

        # >>> Token-level stitching buffers (per doc)
        self._tok_sum: dict[int, torch.Tensor] = {}  # [T_doc, C]
        self._tok_cnt: dict[int, torch.Tensor] = {}  # [T_doc]
        self._tok_gold: dict[int, torch.Tensor] = (
            {}
        )  # [T_doc] (label ids, -100 for unknown)
        self._doc_raw: dict[int, str] = {}  # Optional: original/clean text
        self._tok_offsets: dict[int, list[tuple[int, int]]] = {}
        self.test_metrics: dict[str, object] | None = None

    def _reset_eval_buffers(self) -> None:
        self._tok_sum.clear()
        self._tok_cnt.clear()
        self._tok_gold.clear()
        self._doc_raw.clear()
        self._tok_offsets.clear()

    def _pack_active_tokens(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        hidden_states: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        active_mask = labels != self.ignore_index
        batch_size = logits.size(0)
        lengths = active_mask.long().sum(dim=1)
        if (lengths == 0).any():
            raise ValueError("Each training sample must contain at least one active label.")
        max_len = int(lengths.max().item())
        packed_logits = logits.new_zeros((batch_size, max_len, self.num_labels))
        packed_labels = labels.new_zeros((batch_size, max_len))
        packed_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=labels.device)
        packed_hidden: torch.Tensor | None = None
        if hidden_states is not None:
            packed_hidden = hidden_states.new_zeros(
                (batch_size, max_len, hidden_states.size(-1))
            )

        for row_idx in range(batch_size):
            row_mask = active_mask[row_idx]
            row_len = int(lengths[row_idx].item())
            packed_logits[row_idx, :row_len] = logits[row_idx][row_mask]
            packed_labels[row_idx, :row_len] = labels[row_idx][row_mask]
            packed_mask[row_idx, :row_len] = True
            if packed_hidden is not None and hidden_states is not None:
                packed_hidden[row_idx, :row_len] = hidden_states[row_idx][row_mask]

        return packed_logits, packed_labels, packed_mask, packed_hidden

    def _compute_token_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        packed_logits: torch.Tensor,
        packed_labels: torch.Tensor,
        packed_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.token_loss_mode == "focal":
            return self.focal_loss_fn(logits, labels)

        active_logits = packed_logits[packed_mask]
        active_labels = packed_labels[packed_mask]
        weights = self.token_class_weights.to(active_logits.device)
        return F.cross_entropy(
            active_logits,
            active_labels,
            weight=weights,
            label_smoothing=self.label_smoothing,
        )

    def _build_boundary_targets(self, packed_labels: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = packed_labels.shape
        targets = torch.zeros((batch_size, seq_len, 4), device=packed_labels.device)
        for label_name, target_idx in self._boundary_label_map.items():
            label_id = self.label2id.get(label_name)
            if label_id is None:
                continue
            targets[:, :, target_idx] = (packed_labels == label_id).float()
        return targets

    def _compute_boundary_loss(
        self,
        packed_hidden: torch.Tensor | None,
        packed_labels: torch.Tensor,
        packed_mask: torch.Tensor,
    ) -> torch.Tensor:
        if not self.boundary_head_enabled or self.boundary_classifier is None:
            return packed_labels.new_tensor(0.0, dtype=torch.float32)
        if packed_hidden is None:
            raise RuntimeError("Boundary supervision requires hidden states.")
        boundary_logits = self.boundary_classifier(packed_hidden)
        boundary_targets = self._build_boundary_targets(packed_labels)
        active_logits = boundary_logits[packed_mask]
        active_targets = boundary_targets[packed_mask]
        return F.binary_cross_entropy_with_logits(
            active_logits,
            active_targets,
            pos_weight=self.boundary_pos_weight.to(active_logits.device),
        )

    def _compute_losses(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        hidden_states: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        packed_logits, packed_labels, packed_mask, packed_hidden = self._pack_active_tokens(
            logits, labels, hidden_states
        )
        token_loss = self._compute_token_loss(
            logits, labels, packed_logits, packed_labels, packed_mask
        )
        crf_loss = logits.new_tensor(0.0)
        if self.use_crf and self.crf is not None:
            crf_loss = self.crf.neg_log_likelihood(
                packed_logits, packed_labels, packed_mask
            )
        boundary_loss = self._compute_boundary_loss(
            packed_hidden, packed_labels, packed_mask
        )
        total_loss = (
            self.token_loss_weight * token_loss
            + self.crf_loss_weight * crf_loss
            + self.boundary_loss_weight * boundary_loss
        )
        return total_loss, token_loss, crf_loss, boundary_loss

    def _accumulate_windows(
        self, batch: dict[str, object], logits: torch.Tensor
    ) -> None:
        doc_ids = cast(torch.Tensor, batch["doc_id"])
        window_starts = cast(torch.Tensor, batch["window_start"])
        attention_mask = cast(torch.Tensor, batch["attention_mask"])
        labels = cast(torch.Tensor, batch["labels"])
        offset_mapping = cast(list[list[tuple[int, int]]], batch["offset_mapping"])
        raw = cast(list[str], batch["raw"])

        B = logits.size(0)
        for i in range(B):
            doc_id = int(doc_ids[i])
            window_start = int(window_starts[i])
            offsets = offset_mapping[i]
            attn = attention_mask[i]
            logits_win = logits[i]
            labels_win = labels[i]

            if doc_id not in self._doc_raw:
                self._doc_raw[doc_id] = raw[i]

            non_special_tok_count = 0
            for (o0, o1), a in zip(offsets, attn):
                if a == 0:
                    continue
                if o0 == 0 and o1 == 0:
                    continue
                non_special_tok_count += 1

            required_len = window_start + non_special_tok_count
            if doc_id not in self._tok_sum:
                device = logits_win.device
                self._tok_sum[doc_id] = torch.zeros(
                    (required_len, self.num_labels), device=device
                )
                self._tok_cnt[doc_id] = torch.zeros(required_len, device=device)
                self._tok_gold[doc_id] = torch.full(
                    (required_len,), self.ignore_index, dtype=torch.long, device=device
                )
                self._tok_offsets[doc_id] = [(0, 0)] * required_len
            elif required_len > self._tok_sum[doc_id].size(0):
                device = logits_win.device
                old = self._tok_sum[doc_id].size(0)
                pad = required_len - old
                self._tok_sum[doc_id] = torch.cat(
                    [
                        self._tok_sum[doc_id],
                        torch.zeros((pad, self.num_labels), device=device),
                    ],
                    dim=0,
                )
                self._tok_cnt[doc_id] = torch.cat(
                    [self._tok_cnt[doc_id], torch.zeros(pad, device=device)], dim=0
                )
                self._tok_gold[doc_id] = torch.cat(
                    [
                        self._tok_gold[doc_id],
                        torch.full(
                            (pad,), self.ignore_index, dtype=torch.long, device=device
                        ),
                    ],
                    dim=0,
                )
                self._tok_offsets[doc_id].extend([(0, 0)] * pad)

            abs_tok = window_start
            for t_idx, ((o0, o1), a) in enumerate(zip(offsets, attn)):
                if a == 0:
                    continue
                if o0 == 0 and o1 == 0:
                    continue

                self._tok_sum[doc_id][abs_tok] += logits_win[t_idx]
                self._tok_cnt[doc_id][abs_tok] += 1

                lab = int(labels_win[t_idx].item())
                if (
                    lab != self.ignore_index
                    and self._tok_gold[doc_id][abs_tok].item() == self.ignore_index
                ):
                    self._tok_gold[doc_id][abs_tok] = lab

                if self._tok_offsets[doc_id][abs_tok] == (0, 0) and (o0, o1) != (0, 0):
                    self._tok_offsets[doc_id][abs_tok] = (int(o0), int(o1))

                abs_tok += 1

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> _LogitsOutput:
        return cast(
            _LogitsOutput,
            self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=self.boundary_head_enabled,
            ),
        )

    # ----------------- TRAIN -----------------
    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        outputs = self.forward(batch["input_ids"], batch["attention_mask"])
        logits = outputs.logits
        hidden_states = (
            outputs.hidden_states[-1]
            if self.boundary_head_enabled and outputs.hidden_states is not None
            else None
        )
        loss, token_loss, crf_loss, boundary_loss = self._compute_losses(
            logits, batch["labels"], hidden_states
        )

        labels = batch["labels"]
        preds = torch.argmax(logits, dim=-1)
        mask = labels != self.ignore_index

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=batch["input_ids"].shape[0],
        )
        self.log(
            "train/token_loss",
            token_loss,
            prog_bar=False,
            on_step=True,
            on_epoch=True,
            batch_size=batch["input_ids"].shape[0],
        )
        if self.use_crf:
            self.log(
                "train/crf_loss",
                crf_loss,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=batch["input_ids"].shape[0],
            )
        if self.boundary_head_enabled:
            self.log(
                "train/boundary_loss",
                boundary_loss,
                prog_bar=False,
                on_step=True,
                on_epoch=True,
                batch_size=batch["input_ids"].shape[0],
            )
        self._log_lr()

        if mask.any():
            self.train_f1(preds[mask], labels[mask])
            self.train_f1_no_o(preds[mask], labels[mask])

        return loss

    def on_train_epoch_end(self) -> None:
        self.train_f1.reset()
        self.train_f1_no_o.reset()

    def _viterbi_constrained_doc(self, emissions: torch.Tensor) -> torch.Tensor:
        """
        emissions: [T, C] (already stitched/averaged), full doc tokens only
        returns:  [T] best label path enforcing BIOES legality
        """
        if self.use_crf and self.crf is not None:
            return self.crf.decode(emissions)
        T, C = emissions.shape
        trans = self._crf_trans.to(emissions.device)
        start = self._crf_start.to(emissions.device)
        end = self._crf_end.to(emissions.device)

        score = start + emissions[0]  # [C]
        backp = torch.zeros((T, C), dtype=torch.long, device=emissions.device)

        for t in range(1, T):
            prev = score.unsqueeze(1) + trans  # [C_prev, C_curr]
            best_prev, best_idx = prev.max(dim=0)  # [C], [C]
            score = best_prev + emissions[t]  # [C]
            backp[t] = best_idx

        score = score + end
        last = int(torch.argmax(score))
        path = torch.empty((T,), dtype=torch.long, device=emissions.device)
        path[-1] = last
        for t in range(T - 1, 0, -1):
            path[t - 1] = backp[t, path[t]]
        return path

    def decode_constrained_doc(self, emissions: torch.Tensor) -> torch.Tensor:
        return self._viterbi_constrained_doc(emissions)

    # ----------------- VAL -----------------
    def on_validation_epoch_start(self) -> None:
        self._reset_eval_buffers()

    def validation_step(self, batch: dict[str, object], batch_idx: int) -> torch.Tensor:
        input_ids = cast(torch.Tensor, batch["input_ids"])
        attention_mask = cast(torch.Tensor, batch["attention_mask"])
        labels = cast(torch.Tensor, batch["labels"])

        outputs = self.forward(input_ids, attention_mask)
        logits = outputs.logits
        hidden_states = (
            outputs.hidden_states[-1]
            if self.boundary_head_enabled and outputs.hidden_states is not None
            else None
        )
        loss, token_loss, crf_loss, boundary_loss = self._compute_losses(
            logits, labels, hidden_states
        )
        preds = torch.argmax(logits, dim=-1)
        mask = labels != self.ignore_index

        # Window-level metrics (micro)
        if mask.any():
            self.val_f1_no_o(preds[mask], labels[mask])

        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=input_ids.shape[0],
        )
        self.log(
            "val/token_loss",
            token_loss,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
            batch_size=input_ids.shape[0],
        )
        if self.use_crf:
            self.log(
                "val/crf_loss",
                crf_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=input_ids.shape[0],
            )
        if self.boundary_head_enabled:
            self.log(
                "val/boundary_loss",
                boundary_loss,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
                batch_size=input_ids.shape[0],
            )

        self._accumulate_windows(batch, logits)

        return loss

    def on_validation_epoch_end(self) -> None:
        self.val_f1_no_o.reset()

        metrics_raw = self._compute_variant_metrics()
        if metrics_raw is None:
            self.log(
                "val_ent_f1",
                0.0,
                logger=False,
                prog_bar=False,
                on_step=False,
                on_epoch=True,
            )
            return
        val_ent_f1 = self._log_headline_metrics("val", metrics_raw)
        self.log(
            "val_ent_f1",
            val_ent_f1,
            logger=False,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

    def on_test_epoch_start(self) -> None:
        self._reset_eval_buffers()

    def test_step(self, batch: dict[str, object], batch_idx: int) -> torch.Tensor:
        input_ids = cast(torch.Tensor, batch["input_ids"])
        attention_mask = cast(torch.Tensor, batch["attention_mask"])
        labels = cast(torch.Tensor, batch["labels"])

        outputs = self.forward(input_ids, attention_mask)
        logits = outputs.logits
        hidden_states = (
            outputs.hidden_states[-1]
            if self.boundary_head_enabled and outputs.hidden_states is not None
            else None
        )
        loss, _token_loss, _crf_loss, _boundary_loss = self._compute_losses(
            logits, labels, hidden_states
        )

        self._accumulate_windows(batch, logits)
        return loss

    def _compute_token_metrics(
        self, pred_ids: list[int], gold_ids: list[int]
    ) -> dict[str, object]:
        labels = [self.id2label[i] for i in range(self.num_labels)]
        counts = {label: {"tp": 0, "fp": 0, "fn": 0, "support": 0} for label in labels}
        correct = 0
        total = 0
        for p, g in zip(pred_ids, gold_ids):
            g_label = self.id2label[g]
            p_label = self.id2label[p]
            total += 1
            if p == g:
                correct += 1
                counts[g_label]["tp"] += 1
            else:
                counts[p_label]["fp"] += 1
                counts[g_label]["fn"] += 1
            counts[g_label]["support"] += 1

        def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            return prec, rec, f1

        per_label = {}
        for label, c in counts.items():
            prec, rec, f1 = _prf(c["tp"], c["fp"], c["fn"])
            per_label[label] = {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": c["support"],
            }

        def _aggregate(label_names: list[str]) -> dict[str, float]:
            tp = sum(counts[l]["tp"] for l in label_names)
            fp = sum(counts[l]["fp"] for l in label_names)
            fn = sum(counts[l]["fn"] for l in label_names)
            prec, rec, f1 = _prf(tp, fp, fn)
            return {"precision": prec, "recall": rec, "f1": f1}

        def _macro(label_names: list[str]) -> dict[str, float]:
            vals = [
                per_label[l]["f1"] for l in label_names if per_label[l]["support"] > 0
            ]
            if not vals:
                return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            precs = [
                per_label[l]["precision"]
                for l in label_names
                if per_label[l]["support"] > 0
            ]
            recs = [
                per_label[l]["recall"]
                for l in label_names
                if per_label[l]["support"] > 0
            ]
            return {
                "precision": sum(precs) / len(precs),
                "recall": sum(recs) / len(recs),
                "f1": sum(vals) / len(vals),
            }

        no_o_labels = [l for l in labels if l != "O"]
        metrics = {
            "accuracy": (correct / total) if total else 0.0,
            "micro": _aggregate(labels),
            "macro": _macro(labels),
            "micro_no_o": _aggregate(no_o_labels),
            "macro_no_o": _macro(no_o_labels),
            "per_label": per_label,
            "total_tokens": total,
        }
        return cast(dict[str, object], metrics)

    def _compute_entity_metrics(
        self,
        pred_tags: list[str],
        gold_tags: list[str],
        counts: dict[str, dict[str, int]],
    ) -> None:
        pred_spans = tags_to_spans(pred_tags)
        gold_spans = tags_to_spans(gold_tags)

        pred_by_type: dict[str, set[tuple[int, int]]] = {}
        gold_by_type: dict[str, set[tuple[int, int]]] = {}
        for s, e, t in pred_spans:
            pred_by_type.setdefault(t, set()).add((s, e))
        for s, e, t in gold_spans:
            gold_by_type.setdefault(t, set()).add((s, e))

        types = set(pred_by_type) | set(gold_by_type)
        for t in types:
            pred_set = pred_by_type.get(t, set())
            gold_set = gold_by_type.get(t, set())
            tp = len(pred_set & gold_set)
            fp = len(pred_set - gold_set)
            fn = len(gold_set - pred_set)
            _ = counts.setdefault(t, {"tp": 0, "fp": 0, "fn": 0, "support": 0})
            counts[t]["tp"] += tp
            counts[t]["fp"] += fp
            counts[t]["fn"] += fn
            counts[t]["support"] += len(gold_set)

    def _compute_entity_metrics_lenient(
        self,
        pred_tags: list[str],
        gold_tags: list[str],
        counts: dict[str, dict[str, int]],
    ) -> None:
        pred_spans = tags_to_spans(pred_tags)
        gold_spans = tags_to_spans(gold_tags)

        pred_by_type: dict[str, list[tuple[int, int]]] = {}
        gold_by_type: dict[str, list[tuple[int, int]]] = {}
        for s, e, t in pred_spans:
            pred_by_type.setdefault(t, []).append((s, e))
        for s, e, t in gold_spans:
            gold_by_type.setdefault(t, []).append((s, e))

        def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
            return not (a[1] < b[0] or b[1] < a[0])

        types = set(pred_by_type) | set(gold_by_type)
        for t in types:
            preds = pred_by_type.get(t, [])
            golds = gold_by_type.get(t, [])
            unmatched = golds[:]
            tp = fp = 0
            for p in preds:
                match_idx = None
                for i, g in enumerate(unmatched):
                    if _overlaps(p, g):
                        match_idx = i
                        break
                if match_idx is None:
                    fp += 1
                else:
                    tp += 1
                    _ = unmatched.pop(match_idx)
            fn = len(unmatched)
            _ = counts.setdefault(t, {"tp": 0, "fp": 0, "fn": 0, "support": 0})
            counts[t]["tp"] += tp
            counts[t]["fp"] += fp
            counts[t]["fn"] += fn
            counts[t]["support"] += len(golds)

    def _compute_eval_metrics(
        self,
        pred_tags_by_doc: dict[int, list[str]],
        gold_tags_by_doc: dict[int, list[str]],
    ) -> dict[str, object]:
        """
        Compute token and entity metrics for a set of document-level tags.
        """
        all_pred_ids: list[int] = []
        all_gold_ids: list[int] = []
        ent_counts: dict[str, dict[str, int]] = {}
        ent_counts_lenient: dict[str, dict[str, int]] = {}
        doc_count = 0
        doc_with_entities = 0
        tp_nonempty = fp_nonempty = fn_nonempty = 0
        tp_nonempty_len = fp_nonempty_len = fn_nonempty_len = 0
        doc_spans_all: list[tuple[list[tuple[int, int, str]], list[tuple[int, int, str]]]] = []

        for doc_id, gold_tags in gold_tags_by_doc.items():
            pred_tags = pred_tags_by_doc[doc_id]
            if len(pred_tags) != len(gold_tags):
                raise ValueError("Predicted and gold tag lengths do not match.")
            doc_count += 1

            pred_spans = tags_to_spans(pred_tags)
            gold_spans = tags_to_spans(gold_tags)
            doc_spans_all.append((pred_spans, gold_spans))
            self._compute_entity_metrics(pred_tags, gold_tags, ent_counts)
            self._compute_entity_metrics_lenient(
                pred_tags, gold_tags, ent_counts_lenient
            )

            if gold_spans:
                doc_with_entities += 1
                tpi, fpi, fni = prf1_from_spans(pred_spans, gold_spans)
                tp_nonempty += tpi
                fp_nonempty += fpi
                fn_nonempty += fni

                tpi, fpi, fni = prf1_from_spans_lenient(pred_spans, gold_spans)
                tp_nonempty_len += tpi
                fp_nonempty_len += fpi
                fn_nonempty_len += fni

            for tag in pred_tags:
                all_pred_ids.append(self.label2id[tag])
            for tag in gold_tags:
                all_gold_ids.append(self.label2id[tag])

        if not all_gold_ids:
            return {}

        token_metrics = self._compute_token_metrics(all_pred_ids, all_gold_ids)

        def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            return prec, rec, f1

        per_type = {}
        types_with_support = []
        tp_total = fp_total = fn_total = 0
        for ent_type, c in ent_counts.items():
            prec, rec, f1 = _prf(c["tp"], c["fp"], c["fn"])
            per_type[ent_type] = {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": c["support"],
            }
            tp_total += c["tp"]
            fp_total += c["fp"]
            fn_total += c["fn"]
            if c["support"] > 0:
                types_with_support.append(ent_type)

        micro_prec, micro_rec, micro_f1 = _prf(tp_total, fp_total, fn_total)
        if types_with_support:
            macro_prec = sum(
                per_type[t]["precision"] for t in types_with_support
            ) / len(types_with_support)
            macro_rec = sum(per_type[t]["recall"] for t in types_with_support) / len(
                types_with_support
            )
            macro_f1 = sum(per_type[t]["f1"] for t in types_with_support) / len(
                types_with_support
            )
        else:
            macro_prec = macro_rec = macro_f1 = 0.0

        entity_metrics: dict[str, object] = {
            "micro": {"precision": micro_prec, "recall": micro_rec, "f1": micro_f1},
            "macro": {"precision": macro_prec, "recall": macro_rec, "f1": macro_f1},
            "per_type": per_type,
            "total_entities": sum(c["support"] for c in ent_counts.values()),
        }

        per_type_len = {}
        types_with_support_len = []
        tp_total_len = fp_total_len = fn_total_len = 0
        for ent_type, c in ent_counts_lenient.items():
            prec, rec, f1 = _prf(c["tp"], c["fp"], c["fn"])
            per_type_len[ent_type] = {
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "support": c["support"],
            }
            tp_total_len += c["tp"]
            fp_total_len += c["fp"]
            fn_total_len += c["fn"]
            if c["support"] > 0:
                types_with_support_len.append(ent_type)

        micro_prec_len, micro_rec_len, micro_f1_len = _prf(
            tp_total_len, fp_total_len, fn_total_len
        )
        if types_with_support_len:
            macro_prec_len = sum(
                per_type_len[t]["precision"] for t in types_with_support_len
            ) / len(types_with_support_len)
            macro_rec_len = sum(
                per_type_len[t]["recall"] for t in types_with_support_len
            ) / len(types_with_support_len)
            macro_f1_len = sum(
                per_type_len[t]["f1"] for t in types_with_support_len
            ) / len(types_with_support_len)
        else:
            macro_prec_len = macro_rec_len = macro_f1_len = 0.0

        entity_metrics["lenient"] = {
            "micro": {
                "precision": micro_prec_len,
                "recall": micro_rec_len,
                "f1": micro_f1_len,
            },
            "macro": {
                "precision": macro_prec_len,
                "recall": macro_rec_len,
                "f1": macro_f1_len,
            },
            "per_type": per_type_len,
            "total_entities": sum(c["support"] for c in ent_counts_lenient.values()),
        }

        empty_docs = doc_count - doc_with_entities
        empty_pct = (empty_docs / doc_count) if doc_count else 0.0
        entity_metrics["docs"] = {
            "total": doc_count,
            "with_entities": doc_with_entities,
            "empty": empty_docs,
            "empty_pct": empty_pct,
        }

        nonempty_prec, nonempty_rec, nonempty_f1 = _prf(
            tp_nonempty, fp_nonempty, fn_nonempty
        )
        nonempty_prec_len, nonempty_rec_len, nonempty_f1_len = _prf(
            tp_nonempty_len, fp_nonempty_len, fn_nonempty_len
        )
        entity_metrics["nonempty_micro"] = {
            "precision": nonempty_prec,
            "recall": nonempty_rec,
            "f1": nonempty_f1,
        }
        entity_metrics["nonempty_micro_lenient"] = {
            "precision": nonempty_prec_len,
            "recall": nonempty_rec_len,
            "f1": nonempty_f1_len,
        }

        bootstrap_ci = _bootstrap_entity_f1_ci(doc_spans_all)
        if bootstrap_ci:
            entity_metrics["bootstrap_ci"] = bootstrap_ci

        return {"token_level": token_metrics, "entity_level": entity_metrics}

    def on_test_epoch_end(self) -> None:
        metrics_raw = self._compute_variant_metrics()
        if metrics_raw is None:
            return
        prefix = cast(str, getattr(self, "eval_log_prefix", "test"))
        _ = self._log_headline_metrics(prefix, metrics_raw)
        variants = {"raw": metrics_raw}
        metrics = {
            "variants": variants,
            "primary_variant": "raw",
            "primary": metrics_raw,
            "class_weights": {
                "default": float(getattr(self.hparams, "default_class_weight", 1.0)),
                "by_label": self.class_weight_map,
            },
        }

        metrics_dir = cast(
            str | None, getattr(self.hparams, "metrics_output_dir", None)
        )
        if metrics_dir is None:
            return
        if metrics_dir == "":
            metrics_dir = "."
        metrics_name = cast(
            str, getattr(self.hparams, "metrics_output_name", "ner_test_metrics.yaml")
        )
        path = os.path.join(metrics_dir, metrics_name)
        os.makedirs(metrics_dir, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            _ = yaml.safe_dump(metrics, f, sort_keys=False)
        self.test_metrics = cast(dict[str, object], metrics)

    def _log_lr(self) -> None:
        if not self.trainer or not self.trainer.optimizers:
            return
        lr = self.trainer.optimizers[0].param_groups[0].get("lr")
        if lr is None:
            return
        self.log("lr", lr, prog_bar=False, on_step=True, on_epoch=False)

    def collect_eval_docs(self) -> list[EvalDocRecord]:
        records: list[EvalDocRecord] = []
        for doc_id, sum_logits in self._tok_sum.items():
            cnt = self._tok_cnt[doc_id].clamp(min=1.0).unsqueeze(-1)
            avg_logits = sum_logits / cnt
            pred_ids = self._viterbi_constrained_doc(avg_logits)
            gold_ids = self._tok_gold[doc_id]
            mask = gold_ids != self.ignore_index
            if not mask.any():
                continue

            pred_ids = pred_ids[mask].tolist()
            gold_ids = gold_ids[mask].tolist()

            pred_tags = [self.id2label[i] for i in pred_ids]
            gold_tags = [self.id2label[i] for i in gold_ids]
            pred_tags = repair_bioes(pred_tags)

            offsets_full = self._tok_offsets.get(doc_id)
            if offsets_full is None:
                raise RuntimeError("Missing token offsets for evaluation.")
            mask_list = cast(list[bool], mask.tolist())
            offsets_masked = [off for off, keep in zip(offsets_full, mask_list) if keep]
            if len(offsets_masked) != len(pred_tags):
                raise RuntimeError("Token offsets do not align with tag sequence.")

            raw_text = self._doc_raw.get(doc_id)
            if raw_text is None:
                raise RuntimeError("Missing raw text for evaluation.")

            records.append(
                {
                    "doc_id": doc_id,
                    "raw_text": raw_text,
                    "token_offsets": offsets_masked,
                    "gold_tags": gold_tags,
                    "pred_tags_raw": pred_tags,
                }
            )

        return records

    def _compute_variant_metrics(self) -> dict[str, object] | None:
        pred_tags_by_doc_raw: dict[int, list[str]] = {}
        gold_tags_by_doc: dict[int, list[str]] = {}

        for record in self.collect_eval_docs():
            doc_id = int(record["doc_id"])
            pred_tags_by_doc_raw[doc_id] = record["pred_tags_raw"]
            gold_tags_by_doc[doc_id] = record["gold_tags"]

        metrics_raw = self._compute_eval_metrics(pred_tags_by_doc_raw, gold_tags_by_doc)
        if not metrics_raw:
            return None
        return metrics_raw

    def _log_headline_metrics(
        self,
        prefix: str,
        metrics_raw: dict[str, object],
    ) -> float:
        headline: dict[str, float] = {}
        entity_level = cast(dict[str, object], metrics_raw["entity_level"])
        micro = cast(dict[str, float], entity_level["micro"])
        per_type = cast(dict[str, dict[str, float]], entity_level["per_type"])
        article_metrics = per_type.get("ARTICLE", {"f1": 0.0, "recall": 0.0})
        headline[f"{prefix}/entity_strict_f1_raw"] = float(micro["f1"])
        headline[f"{prefix}/article_strict_f1_raw"] = float(article_metrics["f1"])
        headline[f"{prefix}/article_strict_recall_raw"] = float(
            article_metrics["recall"]
        )

        for name, value in headline.items():
            self.log(name, value, prog_bar=True, on_epoch=True)

        return float(headline[f"{prefix}/entity_strict_f1_raw"])

    # ----------------- OPTIM -----------------
    def configure_optimizers(self) -> tuple[list[Optimizer], list[LRSchedulerConfig]]:
        no_decay = ["bias", "LayerNorm.weight"]
        params = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": getattr(self, "weight_decay", 0.01),
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer: Optimizer = torch.optim.AdamW(
            params, lr=getattr(self, "learning_rate", 3e-5)
        )
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(getattr(self, "warmup_steps_pct", 0.1) * total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        scheduler_dict = cast(
            LRSchedulerConfig,
            cast(object, {"scheduler": scheduler, "interval": "step"}),
        )
        return [optimizer], [scheduler_dict]
