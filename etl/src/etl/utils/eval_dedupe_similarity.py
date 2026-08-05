# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false, reportMissingTypeStubs=false, reportPrivateUsage=false
"""Evaluate the dedupe similarity metric on the labeled 2026-08-05 pairs.

Ground truth lives in pdx.dedupe_plan_20260805 (auto_tier=1: true duplicate
pairs; auto_tier=0: non-duplicate pairs, including the 4 distinct-deal traps).
Survivor page text comes from pdx.pages; loser page text from
pdx.dedupe_bak_20260805_pages (falling back to pdx.pages for undeleted
auto_tier=0 losers).

Compares:
- OLD metric: Jaccard of the first-20k-char MinHash (etl.domain.a_staging._compute_minhash)
  at the historical 0.85 threshold.
- NEW metric: full-text digit/punctuation-stripped shingles; duplicate when
  Jaccard >= JACCARD_DUPLICATE_THRESHOLD or containment >= CONTAINMENT_DUPLICATE_THRESHOLD.
Also reports LSH retrievability at LSH_INDEX_THRESHOLD (a duplicate the LSH
index cannot surface would never reach verification) and cover-identity
(dated-as-of) agreement per pair.

Usage (from repo root):
    caffeinate -i etl/.venv/bin/python -m etl.utils.eval_dedupe_similarity
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from datasketch import MinHashLSH
from sqlalchemy import text
from sqlalchemy.engine import Connection

if __package__ in {None, ""}:
    src_root = Path(__file__).resolve().parents[2]
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

from etl.domain.a_staging import _compute_minhash
from etl.domain.dedupe_signatures import (
    CONTAINMENT_DUPLICATE_THRESHOLD,
    JACCARD_DUPLICATE_THRESHOLD,
    LSH_INDEX_THRESHOLD,
    MINHASH_NUM_PERM,
    compute_document_signature,
    decide_duplicate,
    extract_cover_identity,
    similarity_scores,
)
from etl.utils.backfill_agreement_signatures import load_agreement_text_from_pages
from etl.utils.reset_stuck_agreements import build_engine_from_env

_SCHEMA = "pdx"
_PLAN_TABLE = "dedupe_plan_20260805"
_BAK_PAGES_TABLE = "dedupe_bak_20260805_pages"
_OLD_THRESHOLD = 0.85


@dataclass(frozen=True)
class PairScore:
    loser_uuid: str
    survivor_uuid: str
    auto_tier: int
    old_jaccard: float
    new_jaccard: float
    new_containment: float
    lsh_retrievable: bool
    dated_as_of_match: bool | None  # None when either side lacks a date
    decision_action: str
    decision_reason: str


def _load_pairs(conn: Connection) -> list[tuple[str, str, int]]:
    rows = conn.execute(
        text(
            f"""
            SELECT loser_uuid, survivor_uuid, auto_tier
            FROM {_SCHEMA}.{_PLAN_TABLE}
            ORDER BY auto_tier DESC, loser_uuid
            """
        )
    ).fetchall()
    return [(str(row[0]), str(row[1]), int(row[2])) for row in rows]


def _load_text(conn: Connection, agreement_uuid: str, *, prefer_backup: bool) -> tuple[str, int]:
    tables = [_BAK_PAGES_TABLE, "pages"] if prefer_backup else ["pages", _BAK_PAGES_TABLE]
    for table in tables:
        rendered, page_count = load_agreement_text_from_pages(
            conn, _SCHEMA, agreement_uuid, pages_table=table
        )
        if rendered:
            return rendered, page_count
    raise RuntimeError(f"No page text found for {agreement_uuid}")


def _load_filing_dates(conn: Connection) -> dict[str, object]:
    rows = conn.execute(
        text(
            f"""
            SELECT agreement_uuid, filing_date FROM {_SCHEMA}.agreements
            UNION
            SELECT agreement_uuid, filing_date FROM {_SCHEMA}.dedupe_bak_20260805_agreements
            """
        )
    ).fetchall()
    return {str(row[0]): row[1] for row in rows}


def _score_pairs(conn: Connection) -> list[PairScore]:
    pairs = _load_pairs(conn)
    filing_dates = _load_filing_dates(conn)
    scores: list[PairScore] = []
    for index, (loser_uuid, survivor_uuid, auto_tier) in enumerate(pairs, start=1):
        loser_text, loser_pages = _load_text(conn, loser_uuid, prefer_backup=True)
        survivor_text, survivor_pages = _load_text(conn, survivor_uuid, prefer_backup=False)

        old_jaccard = float(
            _compute_minhash(loser_text).jaccard(_compute_minhash(survivor_text))
        )
        loser_sig = compute_document_signature(loser_text, page_count=loser_pages)
        survivor_sig = compute_document_signature(survivor_text, page_count=survivor_pages)
        new_scores = similarity_scores(loser_sig, survivor_sig)

        # Realistic LSH banding check: would the index surface this pair?
        lsh = MinHashLSH(threshold=LSH_INDEX_THRESHOLD, num_perm=MINHASH_NUM_PERM)
        lsh.insert("survivor", survivor_sig.minhash)
        lsh_retrievable = bool(lsh.query(loser_sig.minhash))

        loser_cover = extract_cover_identity(loser_text)
        survivor_cover = extract_cover_identity(survivor_text)
        dated_match: bool | None = None
        if loser_cover.dated_as_of is not None and survivor_cover.dated_as_of is not None:
            dated_match = loser_cover.dated_as_of == survivor_cover.dated_as_of

        loser_filed = filing_dates.get(loser_uuid)
        survivor_filed = filing_dates.get(survivor_uuid)
        if loser_filed is not None and survivor_filed is not None:
            newer_cover = loser_cover if str(loser_filed) >= str(survivor_filed) else survivor_cover
        else:
            newer_cover = survivor_cover
        decision = decide_duplicate(
            loser_sig,
            loser_cover,
            survivor_sig,
            survivor_cover,
            newer_amends_and_restates=newer_cover.amends_and_restates,
        )

        scores.append(
            PairScore(
                loser_uuid=loser_uuid,
                survivor_uuid=survivor_uuid,
                auto_tier=auto_tier,
                old_jaccard=old_jaccard,
                new_jaccard=new_scores.jaccard,
                new_containment=new_scores.containment,
                lsh_retrievable=lsh_retrievable,
                dated_as_of_match=dated_match,
                decision_action=decision.action.value,
                decision_reason=decision.reason,
            )
        )
        if index % 25 == 0:
            print(f"Scored {index}/{len(pairs)} pairs...", file=sys.stderr)
    return scores


def _new_metric_flags(score: PairScore) -> bool:
    return (
        score.new_jaccard >= JACCARD_DUPLICATE_THRESHOLD
        or score.new_containment >= CONTAINMENT_DUPLICATE_THRESHOLD
    )


def _report(scores: list[PairScore]) -> None:
    positives = [score for score in scores if score.auto_tier == 1]
    negatives = [score for score in scores if score.auto_tier == 0]

    def _stats(name: str, flag_fn: Callable[[PairScore], bool]) -> None:
        true_pos = sum(1 for score in positives if flag_fn(score))
        false_pos = sum(1 for score in negatives if flag_fn(score))
        flagged = true_pos + false_pos
        precision = true_pos / flagged if flagged else 1.0
        recall = true_pos / len(positives) if positives else 0.0
        print(
            f"{name}: recall={recall:.3f} ({true_pos}/{len(positives)}) "
            f"precision={precision:.3f} false_positives={false_pos}/{len(negatives)}"
        )

    print(f"\nLabeled pairs: {len(positives)} true-dupe, {len(negatives)} non-dupe")
    _stats(f"OLD (first-20k jaccard >= {_OLD_THRESHOLD})", lambda s: s.old_jaccard >= _OLD_THRESHOLD)
    _stats(
        f"NEW (jaccard >= {JACCARD_DUPLICATE_THRESHOLD} or containment >= {CONTAINMENT_DUPLICATE_THRESHOLD})",
        _new_metric_flags,
    )
    _stats(
        f"NEW LSH stage (index threshold {LSH_INDEX_THRESHOLD}, banding)",
        lambda s: s.lsh_retrievable,
    )

    dated_known = [s for s in positives if s.dated_as_of_match is not None]
    dated_agree = sum(1 for s in dated_known if s.dated_as_of_match)
    print(
        f"Cover identity on true dupes: dated-as-of extracted on both sides for "
        f"{len(dated_known)}/{len(positives)}; matching for {dated_agree}/{len(dated_known) or 1}"
    )

    def _decision_counts(rows: list[PairScore]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            key = f"{row.decision_action}:{row.decision_reason}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    print("\nEnd-to-end decisions on true dupes (auto_tier=1):")
    for key, count in sorted(_decision_counts(positives).items()):
        print(f"  {key}: {count}")
    print("End-to-end decisions on non-dupes (auto_tier=0) -- none may be auto_dedupe:")
    for key, count in sorted(_decision_counts(negatives).items()):
        print(f"  {key}: {count}")
    false_autos = [s for s in negatives if s.decision_action == "auto_dedupe"]
    print(f"FALSE AUTO-DEDUPES ON NON-DUPES: {len(false_autos)}")

    print("\nNon-dupe (auto_tier=0) pair scores -- none may reach auto_dedupe:")
    for score in negatives:
        print(
            f"  loser={score.loser_uuid} survivor={score.survivor_uuid} "
            f"old_j={score.old_jaccard:.3f} new_j={score.new_jaccard:.3f} "
            f"new_c={score.new_containment:.3f} dated_match={score.dated_as_of_match} "
            f"decision={score.decision_action}:{score.decision_reason}"
        )

    missed = [s for s in positives if not _new_metric_flags(s)]
    if missed:
        print(f"\nTrue-dupe pairs missed by NEW metric ({len(missed)}):")
        for score in missed:
            print(
                f"  loser={score.loser_uuid} survivor={score.survivor_uuid} "
                f"old_j={score.old_jaccard:.3f} new_j={score.new_jaccard:.3f} "
                f"new_c={score.new_containment:.3f} lsh={score.lsh_retrievable} "
                f"dated_match={score.dated_as_of_match}"
            )


def main() -> None:
    db = build_engine_from_env()
    with db.get_engine().begin() as conn:
        scores = _score_pairs(conn)
    _report(scores)


if __name__ == "__main__":
    main()
