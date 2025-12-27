# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
from __future__ import annotations

import re
from typing import List, Tuple

import dagster as dg
from sqlalchemy import text

from etl.defs.d_ai_repair_asset import ai_repair_poll_asset
from etl.defs.resources import DBResource, PipelineConfig


def _strip_tags_and_spans(tagged_text: str) -> Tuple[str, List[Tuple[int, int, str]]]:
    """
    Convert tagged text with <article>/<section>/<page> into raw text and spans
    with page-level character offsets.
    Returns (raw_text, spans), where spans = [(start, end, label)].
    """
    open_tag_re = re.compile(r"<(article|section|page)>")
    close_tag_re = re.compile(r"</(article|section|page)>")

    raw_chars: List[str] = []
    spans: List[Tuple[int, int, str]] = []
    stack: List[Tuple[str, int]] = []  # (label, start_pos_in_raw)

    i = 0
    while i < len(tagged_text):
        if tagged_text[i] == "<":
            # Try opening
            m = open_tag_re.match(tagged_text, i)
            if m:
                label = m.group(1)
                stack.append((label, len(raw_chars)))
                i = m.end()
                continue
            # Try closing
            m = close_tag_re.match(tagged_text, i)
            if m:
                label = m.group(1)
                if stack and stack[-1][0] == label:
                    _, start_pos = stack.pop()
                    spans.append((start_pos, len(raw_chars), label))
                i = m.end()
                continue
        # Plain char
        raw_chars.append(tagged_text[i])
        i += 1

    return ("".join(raw_chars), spans)


def _render_tags(raw_text: str, spans: List[Tuple[int, int, str]]) -> str:
    """
    Render tags over raw_text given non-overlapping sorted spans.
    Adjacent spans with the same label are merged.
    """
    if not spans:
        return raw_text
    spans_sorted = sorted(spans, key=lambda x: (x[0], x[1]))
    merged: List[Tuple[int, int, str]] = []
    for s, e, lab in spans_sorted:
        if not merged:
            merged.append((s, e, lab))
            continue
        ms, me, mlab = merged[-1]
        if lab == mlab and s <= me:
            merged[-1] = (ms, max(me, e), mlab)
        else:
            merged.append((s, e, lab))
    out: List[str] = []
    pos = 0
    for s, e, lab in merged:
        if s < pos:
            # skip malformed/overlap silently
            continue
        if pos < s:
            out.append(raw_text[pos:s])
        out.append(f"<{lab}>")
        out.append(raw_text[s:e])
        out.append(f"</{lab}>")
        pos = e
    if pos < len(raw_text):
        out.append(raw_text[pos:])
    return "".join(out)


def _merge_with_rulings(
    base_raw_text: str,
    base_spans: List[Tuple[int, int, str]],
    rulings: List[Tuple[int, int, str]],
) -> str:
    """
    Strictly reconcile LLM rulings with existing NER spans without mutating
    existing spans except when the ruling is entirely outside any tag.

    Rules per requirements:
      - Ruling identical to a base span -> ruling wins
      - Ruling fully within exactly one base span:
          * if labels match -> keep original (base wins)
          * else -> conflict
      - Ruling entirely outside any base span:
          * if label=='none' -> no change
          * else -> insert new span
      - Ruling that partially overlaps any base span or overlaps multiple
        base spans -> conflict

    Conflicts surface as ValueError for the caller to handle.
    """

    # Sort base spans by start for efficient overlap checks
    base_sorted = sorted(base_spans, key=lambda x: (x[0], x[1]))

    # Normalize rulings: sort and merge only when they share the same context
    # relative to base spans (outside, or inside the same base span).
    r_sorted = sorted(rulings, key=lambda x: (x[0], x[1]))

    def overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        return not (a[1] <= b[0] or a[0] >= b[1])

    def classify_segment(s: int, e: int) -> Tuple[str, int | None]:
        """
        Return one of:
          ('outside', None)
          ('inside', base_index)
          ('partial', base_index)
          ('multi', None)
        """
        overlaps_idx: List[int] = []
        for idx, (bs, be, _) in enumerate(base_sorted):
            if be <= s:
                continue
            if bs >= e:
                break
            overlaps_idx.append(idx)
        if not overlaps_idx:
            return ("outside", None)
        if len(overlaps_idx) > 1:
            return ("multi", None)
        idx = overlaps_idx[0]
        bs, be, _ = base_sorted[idx]
        if bs <= s and e <= be:
            return ("inside", idx)
        return ("partial", idx)

    norm_with_ctx: List[Tuple[int, int, str, Tuple[str, int | None]]] = []
    for rs, re, rlab in r_sorted:
        ctx = classify_segment(rs, re)
        if not norm_with_ctx:
            norm_with_ctx.append((rs, re, rlab, ctx))
            continue
        ls, le, llab, lctx = norm_with_ctx[-1]
        # Decide if we can merge with previous
        is_adjacent = le == rs
        is_overlap = overlaps((ls, le), (rs, re))
        if is_overlap and llab != rlab:
            raise ValueError(
                f"Overlapping rulings with different labels: ({ls},{le},{llab}) vs ({rs},{re},{rlab})"
            )
        if (llab == rlab) and (is_overlap or is_adjacent):
            merged_s = min(ls, rs)
            merged_e = max(le, re)
            mctx = classify_segment(merged_s, merged_e)
            can_merge = False
            # Merge only if merged context remains consistent with base spans
            if mctx[0] == "outside" and lctx[0] == "outside" and ctx[0] == "outside":
                can_merge = True
            elif (
                mctx[0] == "inside"
                and lctx[0] == "inside"
                and ctx[0] == "inside"
                and mctx[1] == lctx[1] == ctx[1]
            ):
                can_merge = True
            if can_merge:
                norm_with_ctx[-1] = (merged_s, merged_e, rlab, mctx)
            else:
                norm_with_ctx.append((rs, re, rlab, ctx))
        else:
            norm_with_ctx.append((rs, re, rlab, ctx))

    norm_rulings: List[Tuple[int, int, str]] = [(s, e, lab) for (s, e, lab, _) in norm_with_ctx]

    to_insert: List[Tuple[int, int, str]] = []
    replacements: dict[int, Tuple[int, int, str] | None] = {}

    # For each normalized ruling, evaluate against base spans
    i = 0  # pointer into base_sorted
    n = len(base_sorted)
    for rs, re, rlab in norm_rulings:
        # gather overlaps with base spans
        overlaps_idx: List[int] = []
        # advance i to the first span that could overlap
        while i < n and base_sorted[i][1] <= rs:
            i += 1
        j = i
        while j < n and base_sorted[j][0] < re:
            bs, be, blab = base_sorted[j]
            if not (be <= rs or bs >= re):
                overlaps_idx.append(j)
            j += 1

        if not overlaps_idx:
            # Entirely outside all tags
            if rlab == "none":
                continue
            to_insert.append((rs, re, rlab))
            continue

        if len(overlaps_idx) > 1:
            raise ValueError(
                f"Ruling ({rs},{re},{rlab}) overlaps multiple base spans"
            )

        # Exactly one overlap
        bs, be, blab = base_sorted[overlaps_idx[0]]
        coextensive = rs == bs and re == be
        fully_inside = bs <= rs and re <= be
        if not fully_inside:
            raise ValueError(
                f"Ruling ({rs},{re},{rlab}) partially overlaps base span ({bs},{be},{blab})"
            )
        # fully enclosed
        if coextensive:
            # Ruling wins: replace or remove the base span
            if rlab == "none":
                replacements[overlaps_idx[0]] = None
            else:
                replacements[overlaps_idx[0]] = (rs, re, rlab)
        else:
            # strictly inside but not coextensive: labels must match
            if rlab != blab:
                raise ValueError(
                    f"Ruling label {rlab} disagrees with base label {blab} within same span"
                )
            # labels match → keep original, no change

    # Combine: apply replacements to base spans, then add insertions
    combined: List[Tuple[int, int, str]] = []
    for idx, span in enumerate(base_sorted):
        if idx in replacements:
            rep = replacements[idx]
            if rep is not None:
                combined.append(rep)
        else:
            combined.append(span)
    final_spans = sorted(combined + to_insert, key=lambda x: (x[0], x[1]))
    return _render_tags(base_raw_text, final_spans)


@dg.asset(deps=[ai_repair_poll_asset], name="5_reconcile_tags")
def reconcile_tags(
    context: dg.AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> None:
    """
    Merge LLM outputs into corrected tagged text per page and update
    pdx.tagged_outputs.tagged_text_corrected.
    """
    engine = db.get_engine()

    # batching controls
    batch_size = pipeline_config.tagging_agreement_batch_size
    is_batched = pipeline_config.is_batched()

    last_uuid = ""
    ran_batches = 0
    while True:
        with engine.begin() as conn:
            # Identify pages to reconcile: those with rulings or full-page outputs
            rows = (
                conn.execute(
                    text(
                        """
                        SELECT DISTINCT p.page_uuid
                        FROM pdx.pages p
                        LEFT JOIN pdx.ai_repair_full_pages f USING(page_uuid)
                        LEFT JOIN pdx.ai_repair_rulings r USING(page_uuid)
                        LEFT JOIN pdx.tagged_outputs t USING(page_uuid)
                        WHERE p.page_uuid > :last
                          AND (f.request_id IS NOT NULL OR r.request_id IS NOT NULL)
                          AND (t.tagged_text_corrected IS NULL OR t.tagged_text_corrected = '')
                        ORDER BY p.page_uuid
                        LIMIT :lim
                        """
                    ),
                    {"last": last_uuid, "lim": batch_size},
                )
                .scalars()
                .all()
            )

            if not rows:
                break

            # Prepare statements
            sel_page = text(
                """
                SELECT p.page_uuid, p.processed_page_content AS text,
                       t.tagged_text, t.tagged_text_corrected
                FROM pdx.pages p
                LEFT JOIN pdx.tagged_outputs t USING(page_uuid)
                WHERE p.page_uuid = :pid
                """
            )
            sel_full = text(
                "SELECT tagged_text FROM pdx.ai_repair_full_pages WHERE page_uuid = :pid LIMIT 1"
            )
            sel_rulings = text(
                """
                SELECT start_char, end_char, label
                FROM pdx.ai_repair_rulings WHERE page_uuid = :pid
                ORDER BY start_char, end_char
                """
            )
            update_corrected = text(
                """
                UPDATE pdx.tagged_outputs
                SET tagged_text_corrected = :txt
                WHERE page_uuid = :pid
                """
            )
            update_label_error = text(
                """
                UPDATE pdx.tagged_outputs
                SET label_error = 1
                WHERE page_uuid = :pid
                """
            )

            # per-batch reconciliation counters
            rulings_success_count = 0
            rulings_fail_count = 0
            pages_success_count = 0
            pages_fail_count = 0

            for pid in rows:
                meta = conn.execute(sel_page, {"pid": pid}).mappings().first()
                if not meta:
                    continue
                raw_text = meta["text"] or ""
                tagged_text_orig = meta.get("tagged_text") or ""

                # Full-page output wins
                full = conn.execute(sel_full, {"pid": pid}).scalar()
                if full:
                    _ = conn.execute(update_corrected, {"pid": pid, "txt": full})
                    continue

                # Else overlay excerpt rulings
                rrows = conn.execute(sel_rulings, {"pid": pid}).mappings().fetchall()
                rulings = [
                    (int(r["start_char"]), int(r["end_char"]), str(r["label"]))
                    for r in rrows
                ]
                if not rulings:
                    # nothing to change
                    continue
                rulings_count_for_page = len(rulings)

                if tagged_text_orig:
                    base_raw, base_spans = _strip_tags_and_spans(tagged_text_orig)
                    # Sanity: if base_raw deviates, fall back to raw page text
                    if base_raw != raw_text:
                        base_raw = raw_text
                        base_spans = []
                else:
                    base_raw = raw_text
                    base_spans = []

                try:
                    corrected = _merge_with_rulings(base_raw, base_spans, rulings)
                except Exception:
                    # context.log.error(f"Reconciliation conflict for page {pid}: {e}")
                    # Flag the page as having a label error, but do not abort the run
                    try:
                        _ = conn.execute(update_label_error, {"pid": pid})
                    except Exception as db_err:
                        context.log.warning(
                            f"Failed to set label_error for page {pid}: {db_err}"
                        )
                    # Skip writing corrected text for this page
                    rulings_fail_count += rulings_count_for_page
                    pages_fail_count += 1
                    continue
                _ = conn.execute(update_corrected, {"pid": pid, "txt": corrected})
                rulings_success_count += rulings_count_for_page
                pages_success_count += 1

            last_uuid = rows[-1]
            total_rulings = rulings_success_count + rulings_fail_count
            total_pages = pages_success_count + pages_fail_count
            if total_rulings > 0 or total_pages > 0:
                page_err_pct = int((pages_fail_count / total_pages) * 100) if total_pages else 0
                rul_err_pct = int((rulings_fail_count / total_rulings) * 100) if total_rulings else 0
                context.log.info(
                    f"Batch statistics: pages success={pages_success_count}, failed={pages_fail_count}, total={total_pages}, error rate={page_err_pct}% ; rulings success={rulings_success_count}, failed={rulings_fail_count}, total={total_rulings}, error rate={rul_err_pct}%"
                )
        ran_batches += 1
        if is_batched:
            break
