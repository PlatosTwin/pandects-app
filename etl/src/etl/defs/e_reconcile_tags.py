# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false
import re
from collections import defaultdict
from typing import List, Tuple

import dagster as dg
from dagster import AssetExecutionContext
from sqlalchemy import bindparam, text

from etl.defs.d_ai_repair_asset import ai_repair_poll_asset
from etl.defs.resources import DBResource, PipelineConfig
from etl.utils.post_asset_refresh import run_post_asset_refresh


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

    def is_whitespace_segment(s: int, e: int) -> bool:
        return base_raw_text[s:e].strip() == ""

    def is_punct_tail_segment(s: int, e: int) -> bool:
        segment = base_raw_text[s:e].strip()
        if not segment:
            return False
        # Allow whitespace between punctuation (e.g., ". -----------------")
        return all(ch in (".", "-", " ", "\n", "\r", "\u00A0") for ch in segment) and any(ch in (".", "-") for ch in segment)

    def is_mid_whitespace_segment(s: int, e: int) -> bool:
        segment = base_raw_text[s:e]
        if not segment or any(ch not in ("\n", "\r", " ", "\u00A0") for ch in segment):
            return False
        if sum(1 for ch in segment if ch in (" ", "\u00A0")) > 4:
            return False
        left = base_raw_text[:s]
        right = base_raw_text[e:]
        left_alnum = sum(1 for ch in left if ch.isalnum())
        right_alnum = sum(1 for ch in right if ch.isalnum())
        return (
            left_alnum >= 2
            and right_alnum >= 2
        )

    def is_whitespace_before_punct(s: int, e: int) -> bool:
        segment = base_raw_text[s:e]
        if not segment or any(ch not in ("\n", "\r", " ", "\u00A0") for ch in segment):
            return False
        remainder = base_raw_text[e:].lstrip(" \u00A0\r\n")
        return remainder.startswith(".") or remainder.startswith("-")

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
    trims: dict[int, Tuple[int, int]] = {}

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
            if (
                rlab == "none"
                and blab in ("section", "article")
                and re == be
                and is_punct_tail_segment(rs, re)
            ):
                continue
            if (
                rlab == "none"
                and blab == "article"
                and is_mid_whitespace_segment(rs, re)
            ):
                continue
            if (
                rlab == "none"
                and blab == "section"
                and is_mid_whitespace_segment(rs, re)
            ):
                continue
            if (
                rlab == "none"
                and blab == "section"
                and is_whitespace_before_punct(rs, re)
            ):
                continue
            if (
                rlab == "none"
                and is_whitespace_segment(rs, re)
                and (rs == bs or re == be)
            ):
                cur_s, cur_e = trims.get(overlaps_idx[0], (bs, be))
                if rs == bs:
                    cur_s = max(cur_s, re)
                if re == be:
                    cur_e = min(cur_e, rs)
                trims[overlaps_idx[0]] = (cur_s, cur_e)
                continue
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
        elif idx in trims:
            ts, te = trims[idx]
            if ts < te:
                _, _, lab = span
                combined.append((ts, te, lab))
        else:
            combined.append(span)
    final_spans = sorted(combined + to_insert, key=lambda x: (x[0], x[1]))
    return _render_tags(base_raw_text, final_spans)


@dg.asset(deps=[ai_repair_poll_asset], name="5-3_reconcile_tags")
def reconcile_tags(
    context: AssetExecutionContext,
    db: DBResource,
    pipeline_config: PipelineConfig,
) -> None:
    """
    Merge LLM outputs into corrected tagged text per page and update
    pdx.tagged_outputs.tagged_text_corrected.
    """
    engine = db.get_engine()
    schema = db.database
    pages_table = f"{schema}.pages"
    tagged_outputs_table = f"{schema}.tagged_outputs"
    ai_repair_full_pages_table = f"{schema}.ai_repair_full_pages"
    ai_repair_rulings_table = f"{schema}.ai_repair_rulings"

    # batching controls
    batch_size = pipeline_config.reconcile_tags_agreement_batch_size
    is_batched = pipeline_config.is_batched()

    last_uuid = ""
    while True:
        with engine.begin() as conn:
            # Identify agreements to reconcile: those with rulings or full-page outputs
            agreement_rows = (
                conn.execute(
                    text(
                        f"""
                        SELECT DISTINCT p.agreement_uuid
                        FROM {pages_table} p
                        LEFT JOIN {ai_repair_full_pages_table} f USING(page_uuid)
                        LEFT JOIN {ai_repair_rulings_table} r USING(page_uuid)
                        LEFT JOIN {tagged_outputs_table} t USING(page_uuid)
                        WHERE p.agreement_uuid > :last
                          AND (f.request_id IS NOT NULL OR r.request_id IS NOT NULL)
                          AND (t.tagged_text_corrected IS NULL OR t.tagged_text_corrected = '')
                          AND (t.label_error IS NULL OR t.label_error = 0)
                        ORDER BY p.agreement_uuid
                        LIMIT :lim
                        """
                    ),
                    {"last": last_uuid, "lim": batch_size},
                )
                .scalars()
                .all()
            )

            if not agreement_rows:
                break
            agreement_uuids = list(agreement_rows)

            # Fetch pages to reconcile for the selected agreements
            page_rows = (
                conn.execute(
                    text(
                        f"""
                        SELECT DISTINCT p.page_uuid
                        FROM {pages_table} p
                        LEFT JOIN {ai_repair_full_pages_table} f USING(page_uuid)
                        LEFT JOIN {ai_repair_rulings_table} r USING(page_uuid)
                        LEFT JOIN {tagged_outputs_table} t USING(page_uuid)
                        WHERE p.agreement_uuid IN :auuids
                          AND (f.request_id IS NOT NULL OR r.request_id IS NOT NULL)
                          AND (t.tagged_text_corrected IS NULL OR t.tagged_text_corrected = '')
                          AND (t.label_error IS NULL OR t.label_error = 0)
                        ORDER BY p.agreement_uuid, p.page_uuid
                        """
                    ).bindparams(bindparam("auuids", expanding=True)),
                    {"auuids": tuple(agreement_uuids)},
                )
                .scalars()
                .all()
            )
            page_uuid_list = list(page_rows)

            if not page_uuid_list:
                last_uuid = agreement_uuids[-1]
                if is_batched:
                    break
                continue

            # Batch-fetch page metadata, full-page text, and rulings (avoid N+1)
            meta_rows = conn.execute(
                text(
                    f"""
                    SELECT p.page_uuid, p.processed_page_content AS text,
                           t.tagged_text, t.tagged_text_corrected
                    FROM {pages_table} p
                    LEFT JOIN {tagged_outputs_table} t ON t.page_uuid = p.page_uuid
                    WHERE p.page_uuid IN :pids
                    """
                ).bindparams(bindparam("pids", expanding=True)),
                {"pids": page_uuid_list},
            ).mappings().fetchall()
            page_meta = {r["page_uuid"]: r for r in meta_rows}

            full_rows = conn.execute(
                text(
                    f"SELECT page_uuid, tagged_text FROM {ai_repair_full_pages_table} WHERE page_uuid IN :pids"
                ).bindparams(bindparam("pids", expanding=True)),
                {"pids": page_uuid_list},
            ).mappings().fetchall()
            full_by_page = {r["page_uuid"]: r["tagged_text"] for r in full_rows}

            rulings_rows = conn.execute(
                text(
                    f"""
                    SELECT page_uuid, start_char, end_char, label
                    FROM {ai_repair_rulings_table} WHERE page_uuid IN :pids
                    ORDER BY page_uuid, start_char, end_char
                    """
                ).bindparams(bindparam("pids", expanding=True)),
                {"pids": page_uuid_list},
            ).mappings().fetchall()
            rulings_by_page: dict[str, list[Tuple[int, int, str]]] = defaultdict(list)
            for r in rulings_rows:
                rulings_by_page[r["page_uuid"]].append(
                    (int(r["start_char"]), int(r["end_char"]), str(r["label"]))
                )

            update_corrected = text(
                f"""
                UPDATE {tagged_outputs_table}
                SET tagged_text_corrected = :txt
                WHERE page_uuid = :pid
                  AND NOT (tagged_text_corrected <=> :txt)
                """
            )
            update_label_error = text(
                f"""
                UPDATE {tagged_outputs_table}
                SET label_error = 1
                WHERE page_uuid = :pid
                  AND NOT (label_error <=> 1)
                """
            )

            # per-batch reconciliation counters
            rulings_success_count = 0
            rulings_fail_count = 0
            pages_success_count = 0
            pages_fail_count = 0

            for pid in page_uuid_list:
                meta = page_meta.get(pid)
                if not meta:
                    continue
                raw_text = meta.get("text") or ""
                tagged_text_orig = meta.get("tagged_text") or ""

                # Full-page output wins
                full = full_by_page.get(pid)
                if full:
                    _ = conn.execute(update_corrected, {"pid": pid, "txt": full})
                    continue

                # Else overlay excerpt rulings
                rulings = rulings_by_page.get(pid, [])
                if not rulings:
                    # nothing to change
                    continue
                rulings_count_for_page = len(rulings)

                if tagged_text_orig:
                    base_raw, base_spans = _strip_tags_and_spans(tagged_text_orig)
                else:
                    base_raw = raw_text
                    base_spans = []

                try:
                    corrected = _merge_with_rulings(base_raw, base_spans, rulings)
                except ValueError as e:
                    # Expected: reconciliation conflicts (overlapping rulings, label mismatches)
                    context.log.warning(f"Reconciliation conflict for page {pid}: {e}")
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
                except Exception as e:
                    # Unexpected errors: log with full traceback
                    context.log.error(
                        f"Unexpected error reconciling page {pid}: {e}",
                        exc_info=True,
                    )
                    # Still flag as error and skip
                    try:
                        _ = conn.execute(update_label_error, {"pid": pid})
                    except Exception as db_err:
                        context.log.warning(
                            f"Failed to set label_error for page {pid}: {db_err}"
                        )
                    rulings_fail_count += rulings_count_for_page
                    pages_fail_count += 1
                    continue
                _ = conn.execute(update_corrected, {"pid": pid, "txt": corrected})
                rulings_success_count += rulings_count_for_page
                pages_success_count += 1

            last_uuid = agreement_uuids[-1]
            total_rulings = rulings_success_count + rulings_fail_count
            total_pages = pages_success_count + pages_fail_count
            if total_rulings > 0 or total_pages > 0:
                page_err_pct = int((pages_fail_count / total_pages) * 100) if total_pages else 0
                rul_err_pct = int((rulings_fail_count / total_rulings) * 100) if total_rulings else 0
                context.log.info(
                    f"Batch statistics: pages success={pages_success_count}, failed={pages_fail_count}, total={total_pages}, error rate={page_err_pct}% ; rulings success={rulings_success_count}, failed={rulings_fail_count}, total={total_rulings}, error rate={rul_err_pct}%"
                )
        if is_batched:
            break

    run_post_asset_refresh(context, db, pipeline_config)
