"""
Offline batch taxonomy inference using sglang's in-process Engine.

No server, no ports: loads DeepSeek-V4-Pro into the GPU(s) via `sgl.Engine`,
sends ALL prompts to `engine.generate(...)` in one shot, lets sglang batch
them internally, and writes one output line per section with its assigned
taxonomy standard_ids.

Each input JSONL row is a ready-to-ingest record produced by
generate_batches.py:
  {"sections": [{"section_uuid": ..., "agreement_uuid": ...}, ...],
   "messages": [{system}, {user}]}
A row may carry one or more sections (set by --sections-per-request at
generation time). This script only renders the chat template and dispatches,
then writes one output line per section_uuid.

Designed to run inside the sglang apptainer/runtime image on HPC.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any


def _build_response_schema(
    section_uuids: list[str],
    allowed_category_ids: list[str] | None = None,
    want_rationale: bool = False,
) -> dict[str, Any]:
    """Per-row schema pinning section_uuid to the row's actual UUIDs.

    Without the enum, grammar-constrained decoding accepts any string for
    section_uuid, and the model hallucinates UUIDs that don't match the
    request (observed ~50% miss rate at 2 sections/row). The enum forces
    the decoder to emit exactly one of the requested UUIDs. minItems ==
    maxItems == N additionally pins the assignments array length.

    Retrieval-limited batches additionally pass `allowed_category_ids` (the
    union of that row's candidate standard_ids); pinning `categories` to that
    enum makes an out-of-shortlist / hallucinated id structurally impossible.
    Old full-taxonomy batches omit it and keep the free-string behavior.

    `want_rationale=True` (A/B variant B) prepends a bounded `rationale`
    string as the FIRST property of each assignment. Grammar-constrained
    decoding emits object properties in schema order, so the model is forced
    to produce its reasoning before it commits to `categories` — a
    structured-output substitute for native thinking (thinking_mode stays
    'chat'); parsing ignores the extra field.
    """
    cat_item: dict[str, Any] = {"type": "string"}
    if allowed_category_ids:
        cat_item = {"type": "string", "enum": list(allowed_category_ids)}

    item_props: dict[str, Any] = {}
    required: list[str] = []
    if want_rationale:
        item_props["rationale"] = {"type": "string", "maxLength": 600}
        required.append("rationale")
    item_props["section_uuid"] = {"type": "string", "enum": section_uuids}
    item_props["categories"] = {"type": "array", "items": cat_item}
    required += ["section_uuid", "categories"]

    return {
        "type": "object",
        "properties": {
            "assignments": {
                "type": "array",
                "minItems": len(section_uuids),
                "maxItems": len(section_uuids),
                "items": {
                    "type": "object",
                    "properties": item_props,
                    "required": required,
                },
            }
        },
        "required": ["assignments"],
    }


def _parse_assignments(content: str) -> dict[str, list[str]]:
    data = json.loads(content)
    if not isinstance(data, dict) or "assignments" not in data:
        raise ValueError("Response missing 'assignments' object")
    out: dict[str, list[str]] = {}
    for item in data["assignments"]:
        sid = item.get("section_uuid")
        cats = item.get("categories", [])
        if not isinstance(sid, str) or not isinstance(cats, list):
            raise ValueError(f"Bad assignment shape: {item!r}")
        out[sid] = [str(c).strip() for c in cats if str(c).strip()]
    return out


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _build_prompts(rows: list[dict[str, Any]], encode_messages: Any) -> list[str]:
    """Render each row's pre-built messages into a chat-templated prompt string.

    DeepSeek-V4-Pro's tokenizer ships without a chat_template. We use the
    official `encode_messages` encoder published alongside the model
    (encoding_dsv4.py), which handles BOS, role markers, and the trailing
    `<｜Assistant｜></think>` transition required for chat-mode generation.
    """
    prompts: list[str] = []
    for row in rows:
        messages = row["messages"]
        prompt = encode_messages(messages, thinking_mode="chat")
        prompts.append(prompt)
    return prompts


def _log_generation_stats(
    outputs: list[Any], gen_elapsed: float, log: Any
) -> None:
    """Summarize per-prompt token throughput and finish-reason distribution.

    `engine.generate` is one blocking call so there is no per-iteration
    cadence to report; instead we surface what sglang attaches to each
    output's meta_info (completion_tokens, finish_reason).
    """
    completion_tokens: list[int] = []
    prompt_tokens: list[int] = []
    finish_reasons: dict[str, int] = {}
    for output in outputs:
        meta = output.get("meta_info", {}) if isinstance(output, dict) else {}
        ct = meta.get("completion_tokens")
        pt = meta.get("prompt_tokens")
        if isinstance(ct, int):
            completion_tokens.append(ct)
        if isinstance(pt, int):
            prompt_tokens.append(pt)
        fr = meta.get("finish_reason")
        if isinstance(fr, dict):
            fr_type = str(fr.get("type") or "unknown")
        elif isinstance(fr, str):
            fr_type = fr
        else:
            fr_type = "unknown"
        finish_reasons[fr_type] = finish_reasons.get(fr_type, 0) + 1

    prompts_per_s = len(outputs) / gen_elapsed if gen_elapsed > 0 else 0.0
    log(
        "engine.generate returned %d outputs in %.1fs (%.2f prompts/s, mean %.2fs/prompt)"
        % (len(outputs), gen_elapsed, prompts_per_s,
           gen_elapsed / len(outputs) if outputs else 0.0)
    )
    if completion_tokens:
        ct_sorted = sorted(completion_tokens)
        total_ct = sum(completion_tokens)
        log(
            "completion tokens: total=%d mean=%.1f min=%d p50=%d p95=%d max=%d  decode=%.1f tok/s"
            % (total_ct,
               total_ct / len(completion_tokens),
               ct_sorted[0],
               ct_sorted[len(ct_sorted) // 2],
               ct_sorted[int(len(ct_sorted) * 0.95)],
               ct_sorted[-1],
               total_ct / gen_elapsed if gen_elapsed > 0 else 0.0)
        )
    if prompt_tokens:
        pt_sorted = sorted(prompt_tokens)
        log(
            "prompt tokens: total=%d mean=%.1f min=%d p50=%d p95=%d max=%d"
            % (sum(prompt_tokens),
               sum(prompt_tokens) / len(prompt_tokens),
               pt_sorted[0],
               pt_sorted[len(pt_sorted) // 2],
               pt_sorted[int(len(pt_sorted) * 0.95)],
               pt_sorted[-1])
        )
    log("finish reasons: %s" % dict(sorted(finish_reasons.items())))


def _write_results(
    rows: list[dict[str, Any]],
    outputs: list[Any],
    out_path: Path,
    log: Any,
) -> tuple[int, int, int]:
    """Write one output line per section_uuid. Returns (ok, failed, parse_errors).

    Side effect only: this is the place where row-level parse failures are
    attributed back to each section_uuid, and where the per-section JSONL
    output file is written.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    completed_sections = 0
    failed_sections = 0
    parse_errors = 0
    with out_path.open("w", encoding="utf-8") as out_fh:
        for row_idx, (row, output) in enumerate(zip(rows, outputs)):
            sections = row["sections"]
            text_out = output["text"] if isinstance(output, dict) else str(output)
            try:
                assignments = _parse_assignments(text_out)
                row_parse_error = False
            except (ValueError, json.JSONDecodeError) as exc:
                parse_errors += 1
                row_parse_error = True
                log("ERROR row %d (%d sections) parse error: %s :: head=%r"
                    % (row_idx, len(sections), exc, text_out[:200]))
                assignments = {}

            for section in sections:
                sid = section["section_uuid"]
                if sid in assignments:
                    completed_sections += 1
                    err: str | None = None
                    categories = assignments[sid]
                else:
                    failed_sections += 1
                    err = "parse_error" if row_parse_error else "missing_in_response"
                    categories = []
                out_fh.write(
                    json.dumps(
                        {
                            "section_uuid": sid,
                            "agreement_uuid": section.get("agreement_uuid"),
                            "categories": categories,
                            "error": err,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if (row_idx + 1) % 100 == 0 or row_idx + 1 == len(rows):
                log("progress: rows=%d/%d ok=%d failed=%d parse_errors=%d"
                    % (row_idx + 1, len(rows),
                       completed_sections, failed_sections, parse_errors))

    return completed_sections, failed_sections, parse_errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Input JSONL of ready-to-ingest rows from generate_batches.py")
    parser.add_argument("--output", type=Path, required=True, help="Output JSONL with categories")
    parser.add_argument("--model", default="deepseek-ai/DeepSeek-V4-Pro")
    parser.add_argument("--tp-size", type=int, default=8)
    parser.add_argument("--mem-fraction-static", type=float, default=0.88)
    parser.add_argument("--max-running-requests", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Tight cap frees KV cache to batch more requests concurrently. "
                             "Default: auto, scaled to max sections/row in the input "
                             "(50 base + 150 per section).")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--moe-runner-backend", default="marlin")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    # sglang's Engine init clobbers Python's `logging` config (both stdout
    # reassignment and root-handler replacement). Bypass logging entirely:
    # open a plain file and write timestamped lines directly. Nothing sglang
    # does can touch a file descriptor we already hold.
    log_path = args.output.with_suffix(args.output.suffix + ".pylog")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_fh = log_path.open("w", buffering=1, encoding="utf-8")  # line-buffered

    def log(msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} :: {msg}\n"
        log_fh.write(line)
        log_fh.flush()
        try:
            sys.stderr.write(line)
            sys.stderr.flush()
        except Exception:
            pass

    log("python log file: %s" % log_path)

    log("reading rows from %s" % args.input)
    rows = _load_jsonl(args.input)
    log("loaded %d rows" % len(rows))

    max_sections_per_row = max((len(r["sections"]) for r in rows), default=1)
    if args.max_new_tokens is None:
        base = 50
        per_section = 150
        effective_max_new_tokens = base + per_section * max_sections_per_row
        log("auto max_new_tokens=%d (max %d sections/row × %d + %d base)"
            % (effective_max_new_tokens, max_sections_per_row, per_section, base))
    else:
        effective_max_new_tokens = args.max_new_tokens
        log("max_new_tokens=%d (overridden; max %d sections/row in input)"
            % (effective_max_new_tokens, max_sections_per_row))

    # Imports deferred so --help works without sglang installed.
    log("loading sglang Engine: model=%s tp=%d mem_fraction=%.2f"
        % (args.model, args.tp_size, args.mem_fraction_static))
    t_load = time.monotonic()
    import sglang as sgl  # type: ignore  # pyright: ignore[reportMissingImports]
    # encoding_dsv4.py is the official DeepSeek-V4 prompt encoder; it must
    # sit next to this script (downloaded from the model's HF repo).
    from encoding_dsv4 import encode_messages  # type: ignore
    engine_kwargs: dict[str, Any] = {
        "model_path": args.model,
        "trust_remote_code": args.trust_remote_code,
        "tp_size": args.tp_size,
        "mem_fraction_static": args.mem_fraction_static,
        "max_running_requests": args.max_running_requests,
        "moe_runner_backend": args.moe_runner_backend,
    }
    log("engine kwargs: %s" % {k: v for k, v in engine_kwargs.items() if k != "model_path"})
    engine = sgl.Engine(**engine_kwargs)
    log("engine ready in %.1fs" % (time.monotonic() - t_load))

    prompts = _build_prompts(rows, encode_messages)
    sampling_params = [
        {
            "temperature": args.temperature,
            "max_new_tokens": effective_max_new_tokens,
            "json_schema": json.dumps(
                _build_response_schema(
                    [s["section_uuid"] for s in row["sections"]],
                    row.get("allowed_category_ids"),
                    bool(row.get("want_rationale", False)),
                )
            ),
        }
        for row in rows
    ]
    prompt_char_lens = [len(p) for p in prompts]
    log("dispatching %d prompts to engine.generate (prompt chars: min=%d p50=%d p95=%d max=%d)"
        % (len(prompts),
           min(prompt_char_lens),
           sorted(prompt_char_lens)[len(prompt_char_lens) // 2],
           sorted(prompt_char_lens)[int(len(prompt_char_lens) * 0.95)],
           max(prompt_char_lens)))

    t_gen = time.monotonic()
    try:
        outputs = engine.generate(prompts, sampling_params)
        if len(outputs) != len(prompts):
            raise RuntimeError(
                f"engine.generate returned {len(outputs)} outputs for "
                f"{len(prompts)} prompts; aborting before partial write"
            )
        gen_elapsed = time.monotonic() - t_gen

        _log_generation_stats(outputs, gen_elapsed, log)

        completed_sections, failed_sections, parse_errors = _write_results(
            rows, outputs, args.output, log
        )
    finally:
        engine.shutdown()

    log("done: sections_ok=%d sections_failed=%d parse_errors=%d total_elapsed=%.1fs"
        % (completed_sections, failed_sections, parse_errors,
           time.monotonic() - t_gen))
    log_fh.close()
    return 0 if failed_sections == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
