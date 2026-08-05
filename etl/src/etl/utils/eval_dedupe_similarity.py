# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false, reportMissingTypeStubs=false, reportPrivateUsage=false
"""Evaluate the dedupe similarity metric and decision policy on labeled pairs.

Ground truth is the committed dataset etl/data/dedupe_eval/labels.json:
187 true-dupe and 28 non-dupe pairs across four audited 2026-08-05 batches
(A: dedupe-plan tier flags, B: confirmed dupes plus audit-confirmed hard
negatives, C: one re-adjudicated batch A pair, D: review-audited negatives
including the Wood Sage pair -- the hardest known negative). Each pair carries
an orientation-insensitive pair id and its adjudication provenance.

Document text comes exclusively from the local pdx.dedupe_eval_documents
corpus, materialized by etl.utils.build_dedupe_eval_corpus and pinned by the
committed etl/data/dedupe_eval/corpus_manifest.json; a missing or drifted
corpus fails loudly before any scoring. The pdx.dedupe_plan_20260805{,b,c}
and pdx.dedupe_bak_20260805* tables are pure disaster-recovery backups now --
dropping them does not affect this eval (only re-materializing the corpus
would need the *_pages backups).

Requirement: ZERO auto-dedupe decisions across all 28 negatives; report recall
on all 187 positives.

Compares:
- OLD metric: Jaccard of the first-20k-char MinHash at the historical 0.85
  threshold.
- NEW metric: full-text digit/punctuation-stripped shingles (Jaccard +
  containment), plus the full decision policy (cover identity, A&R
  supersession, party-metadata conflict, same-accession hard block).

Usage (from repo root):
    caffeinate -i etl/.venv/bin/python -m etl.utils.eval_dedupe_similarity
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from datasketch import MinHashLSH
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
    party_metadata_conflict,
    same_edgar_accession,
    similarity_scores,
)
from etl.utils.db_env import build_engine_from_env
from etl.utils.dedupe_eval_dataset import (
    EvalPair,
    load_corpus_manifest,
    load_document,
    load_labels,
    missing_from_manifest,
    verify_corpus,
)

_SCHEMA = "pdx"
_OLD_THRESHOLD = 0.85


@dataclass(frozen=True)
class PairScore:
    pair: EvalPair
    old_jaccard: float
    new_jaccard: float
    new_containment: float
    lsh_retrievable: bool
    dated_as_of_match: bool | None  # None when either side lacks a date
    decision_action: str
    decision_reason: str


def _score_pairs(
    conn: Connection, pairs: list[EvalPair], manifest: Mapping[str, str]
) -> list[PairScore]:
    scores: list[PairScore] = []
    for index, pair in enumerate(pairs, start=1):
        loser = pair.loser
        survivor = pair.survivor
        loser_text, loser_pages = load_document(
            conn, _SCHEMA, loser.agreement_uuid, manifest[loser.agreement_uuid]
        )
        survivor_text, survivor_pages = load_document(
            conn, _SCHEMA, survivor.agreement_uuid, manifest[survivor.agreement_uuid]
        )

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

        loser_key = (
            loser.filing_date if loser.filing_date else date.min,
            loser.agreement_uuid,
        )
        survivor_key = (
            survivor.filing_date if survivor.filing_date else date.min,
            survivor.agreement_uuid,
        )
        if loser_key >= survivor_key:
            newer_sig, newer_cover = loser_sig, loser_cover
            older_sig, older_cover = survivor_sig, survivor_cover
        else:
            newer_sig, newer_cover = survivor_sig, survivor_cover
            older_sig, older_cover = loser_sig, loser_cover
        decision = decide_duplicate(
            newer_sig,
            newer_cover,
            older_sig,
            older_cover,
            party_conflict=party_metadata_conflict(
                loser.target,
                loser.acquirer,
                survivor.target,
                survivor.acquirer,
            ),
            same_accession=same_edgar_accession(loser.url, survivor.url),
        )

        scores.append(
            PairScore(
                pair=pair,
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
    positives = [score for score in scores if score.pair.is_dupe]
    negatives = [score for score in scores if not score.pair.is_dupe]

    def _stats(name: str, flag_fn: Callable[[PairScore], bool]) -> None:
        true_pos = sum(1 for score in positives if flag_fn(score))
        false_pos = sum(1 for score in negatives if flag_fn(score))
        flagged = true_pos + false_pos
        precision = true_pos / flagged if flagged else 1.0
        recall = true_pos / len(positives) if positives else 0.0
        print(
            f"{name}: recall={recall:.3f} ({true_pos}/{len(positives)})"
            + f" precision={precision:.3f} false_positives={false_pos}/{len(negatives)}"
        )

    def _batch_breakdown(rows: list[PairScore]) -> str:
        counts: dict[str, int] = {}
        for row in rows:
            counts[row.pair.batch] = counts.get(row.pair.batch, 0) + 1
        return " + ".join(f"{count} batch {batch}" for batch, count in sorted(counts.items()))

    print(
        f"\nLabeled pairs: {len(positives)} true-dupe ({_batch_breakdown(positives)}),"
        + f" {len(negatives)} non-dupe ({_batch_breakdown(negatives)})"
    )
    _stats(f"OLD (first-20k jaccard >= {_OLD_THRESHOLD})", lambda s: s.old_jaccard >= _OLD_THRESHOLD)
    _stats(
        f"NEW content hit (jaccard >= {JACCARD_DUPLICATE_THRESHOLD} or containment >= {CONTAINMENT_DUPLICATE_THRESHOLD})",
        _new_metric_flags,
    )
    _stats(
        f"NEW LSH stage (index threshold {LSH_INDEX_THRESHOLD}, banding)",
        lambda s: s.lsh_retrievable,
    )
    _stats("NEW end-to-end AUTO_DEDUPE", lambda s: s.decision_action == "auto_dedupe")
    _stats(
        "NEW end-to-end AUTO or REVIEW (guard visibility)",
        lambda s: s.decision_action in {"auto_dedupe", "review"},
    )

    def _decision_counts(rows: list[PairScore]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for row in rows:
            key = f"{row.decision_action}:{row.decision_reason}"
            counts[key] = counts.get(key, 0) + 1
        return counts

    print("\nEnd-to-end decisions on true dupes:")
    for key, count in sorted(_decision_counts(positives).items()):
        print(f"  {key}: {count}")
    print("End-to-end decisions on non-dupes -- none may be auto_dedupe:")
    for key, count in sorted(_decision_counts(negatives).items()):
        print(f"  {key}: {count}")
    false_autos = [s for s in negatives if s.decision_action == "auto_dedupe"]
    print(f"FALSE AUTO-DEDUPES ON THE {len(negatives)} KNOWN NEGATIVES: {len(false_autos)}")
    for score in false_autos:
        print(
            f"  !! loser={score.pair.loser.agreement_uuid} survivor={score.pair.survivor.agreement_uuid}"
            + f" {score.decision_action}:{score.decision_reason}"
        )

    print("\nNon-dupe pair scores:")
    for score in negatives:
        print(
            f"  [{score.pair.batch}] loser={score.pair.loser.agreement_uuid} survivor={score.pair.survivor.agreement_uuid}"
            + f" old_j={score.old_jaccard:.3f} new_j={score.new_jaccard:.3f}"
            + f" new_c={score.new_containment:.3f} dated_match={score.dated_as_of_match}"
            + f" decision={score.decision_action}:{score.decision_reason}"
        )

    missed = [s for s in positives if s.decision_action == "none"]
    if missed:
        print(f"\nTrue-dupe pairs invisible to the guard ({len(missed)}):")
        for score in missed:
            print(
                f"  [{score.pair.batch}] loser={score.pair.loser.agreement_uuid} survivor={score.pair.survivor.agreement_uuid}"
                + f" old_j={score.old_jaccard:.3f} new_j={score.new_jaccard:.3f}"
                + f" new_c={score.new_containment:.3f} lsh={score.lsh_retrievable}"
                + f" dated_match={score.dated_as_of_match} reason={score.decision_reason}"
            )


def main() -> None:
    pairs = load_labels()
    manifest = load_corpus_manifest()
    unlisted = missing_from_manifest(pairs, manifest)
    if unlisted:
        raise RuntimeError(
            "Labeled documents missing from corpus_manifest.json (regenerate"
            + " with build_dedupe_eval_corpus --write-manifest and commit): "
            + ", ".join(unlisted)
        )
    db = build_engine_from_env()
    with db.get_engine().begin() as conn:
        verify_corpus(conn, _SCHEMA, manifest)
        scores = _score_pairs(conn, pairs, manifest)
    _report(scores)


if __name__ == "__main__":
    main()
