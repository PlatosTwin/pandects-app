# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportDeprecated=false, reportExplicitAny=false, reportMissingTypeStubs=false, reportPrivateUsage=false
"""Evaluate the dedupe similarity metric and decision policy on labeled pairs.

Ground truth (two audited batches):
- Batch A: pdx.dedupe_plan_20260805 — auto_tier=1: 171 true-dupe pairs
  (loser text in pdx.dedupe_bak_20260805_pages); auto_tier=0: 5 non-dupe
  pairs, including 4 distinct-deal traps.
- Batch B: pdx.dedupe_plan_20260805b — 15 confirmed-dupe pairs (loser text in
  pdx.dedupe_bak_20260805b_pages), plus 13 audit-confirmed NON-dupe pairs
  (both sides still live in pdx.pages) hardcoded below — the strongest
  negatives available (SPAC boilerplate and same-accession sibling exhibits
  that defeated the first-cut A&R rule).

Requirement: ZERO auto-dedupe decisions across all 18 negatives; report recall
on all 186 positives.

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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Final

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
    party_metadata_conflict,
    same_edgar_accession,
    similarity_scores,
)
from etl.utils.backfill_agreement_signatures import load_agreement_text_from_pages
from etl.utils.reset_stuck_agreements import build_engine_from_env

_SCHEMA = "pdx"
_OLD_THRESHOLD = 0.85
_PAGE_TABLES = ("pages", "dedupe_bak_20260805_pages", "dedupe_bak_20260805b_pages")
_AGREEMENT_TABLES = (
    "agreements",
    "dedupe_bak_20260805_agreements",
    "dedupe_bak_20260805b_agreements",
)

# Batch B negatives: 13 pairs the 2026-08-05 audits confirmed as NOT duplicates
# even though the first-cut guard auto-flagged them (10 via the loose A&R rule,
# 3 via content+dated on same-accession sibling exhibits). (loser, survivor)
# uuids as recorded in the residual-dupes export.
BATCH_B_NEGATIVE_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("8b54570c-101c-563d-adf4-2f16c73a7598", "0ffd3729-55d4-5f9d-8290-ff972cd423ac"),
    ("e1217f37-91cc-5f60-8304-44f7093e2029", "1818ad1c-d292-5128-b698-b9ea20982b6b"),
    ("00508685-e189-5314-84ef-d275b70faff4", "1b76ed40-ca07-5acf-954e-e8aafcb6621c"),
    ("8c431e2d-7c3f-5201-ba29-588f8e4227a7", "395016d5-eea1-52c1-84d5-3ebc925bfb19"),
    ("154fd281-cd3f-594a-afe9-9f30aadea943", "45483aee-f45d-50d9-a0d9-b807017fd0ca"),
    ("93ca6bef-f2cd-5b0c-8bc2-1bfedcab0bf1", "76072bed-6446-5c2e-8d53-ad379be5c3bb"),
    ("1003a6f4-ca70-5101-a77c-2824ed79495c", "b252607c-b53a-56c9-8329-ac50e8bc957e"),
    ("d90c8847-9979-59e5-b6da-ef4478f492b9", "c78b8d42-2639-5672-926f-6a32f43317ee"),
    ("c44799bf-b495-5dde-94b8-0e47d87dda23", "cb6d4d12-e081-5726-8bb2-a86f6b61356d"),
    ("d11d5abd-e3d1-5147-82f7-3961167dff0b", "d5148d6d-ee5f-5570-9b95-e4440e38fa92"),
    ("c918c3e5-0613-5089-a8a6-b0954f4f2da7", "5d868b8b-739e-5c91-aa2d-4bf5327ead0a"),
    ("e33d8023-083a-529b-b540-ceed29bfccd3", "8ceacaf4-face-5fbe-b606-bebaf8f80be7"),
    ("98b2f389-b86f-5b29-ae79-ab9020121a75", "c8365dc6-c530-599c-be83-071700465512"),
)


@dataclass(frozen=True)
class LabeledPair:
    loser_uuid: str
    survivor_uuid: str
    is_dupe: bool
    batch: str


@dataclass(frozen=True)
class AgreementMeta:
    url: str
    filing_date: date | None
    target: str | None
    acquirer: str | None


@dataclass(frozen=True)
class PairScore:
    pair: LabeledPair
    old_jaccard: float
    new_jaccard: float
    new_containment: float
    lsh_retrievable: bool
    dated_as_of_match: bool | None  # None when either side lacks a date
    decision_action: str
    decision_reason: str


def _load_pairs(conn: Connection) -> list[LabeledPair]:
    pairs: list[LabeledPair] = []
    rows = conn.execute(
        text(
            f"""
            SELECT loser_uuid, survivor_uuid, auto_tier
            FROM {_SCHEMA}.dedupe_plan_20260805
            ORDER BY auto_tier DESC, loser_uuid
            """
        )
    ).fetchall()
    for row in rows:
        pairs.append(
            LabeledPair(
                loser_uuid=str(row[0]),
                survivor_uuid=str(row[1]),
                is_dupe=int(row[2]) == 1,
                batch="A",
            )
        )
    rows = conn.execute(
        text(
            f"""
            SELECT loser_uuid, survivor_uuid
            FROM {_SCHEMA}.dedupe_plan_20260805b
            ORDER BY loser_uuid
            """
        )
    ).fetchall()
    for row in rows:
        pairs.append(
            LabeledPair(
                loser_uuid=str(row[0]),
                survivor_uuid=str(row[1]),
                is_dupe=True,
                batch="B",
            )
        )
    for loser_uuid, survivor_uuid in BATCH_B_NEGATIVE_PAIRS:
        pairs.append(
            LabeledPair(
                loser_uuid=loser_uuid,
                survivor_uuid=survivor_uuid,
                is_dupe=False,
                batch="B",
            )
        )
    return pairs


def _load_agreement_meta(conn: Connection) -> dict[str, AgreementMeta]:
    meta: dict[str, AgreementMeta] = {}
    # Live table last so live rows win over backups for undeleted agreements.
    for table in reversed(_AGREEMENT_TABLES):
        rows = conn.execute(
            text(
                f"""
                SELECT agreement_uuid, url, filing_date, target, acquirer
                FROM {_SCHEMA}.{table}
                """
            )
        ).fetchall()
        for row in rows:
            filing_raw = row[2]
            filing = filing_raw if isinstance(filing_raw, date) else None
            meta[str(row[0])] = AgreementMeta(
                url=str(row[1]),
                filing_date=filing,
                target=None if row[3] is None else str(row[3]),
                acquirer=None if row[4] is None else str(row[4]),
            )
    return meta


def _load_text(conn: Connection, agreement_uuid: str, *, prefer_backup: bool) -> tuple[str, int]:
    tables = list(_PAGE_TABLES)
    if prefer_backup:
        tables = tables[1:] + tables[:1]
    for table in tables:
        rendered, page_count = load_agreement_text_from_pages(
            conn, _SCHEMA, agreement_uuid, pages_table=table
        )
        if rendered:
            return rendered, page_count
    raise RuntimeError(f"No page text found for {agreement_uuid}")


def _score_pairs(conn: Connection) -> list[PairScore]:
    pairs = _load_pairs(conn)
    meta = _load_agreement_meta(conn)
    scores: list[PairScore] = []
    for index, pair in enumerate(pairs, start=1):
        loser_text, loser_pages = _load_text(conn, pair.loser_uuid, prefer_backup=pair.is_dupe)
        survivor_text, survivor_pages = _load_text(conn, pair.survivor_uuid, prefer_backup=False)

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

        loser_meta = meta.get(pair.loser_uuid)
        survivor_meta = meta.get(pair.survivor_uuid)
        loser_key = (
            loser_meta.filing_date if loser_meta and loser_meta.filing_date else date.min,
            pair.loser_uuid,
        )
        survivor_key = (
            survivor_meta.filing_date if survivor_meta and survivor_meta.filing_date else date.min,
            pair.survivor_uuid,
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
                loser_meta.target if loser_meta else None,
                loser_meta.acquirer if loser_meta else None,
                survivor_meta.target if survivor_meta else None,
                survivor_meta.acquirer if survivor_meta else None,
            ),
            same_accession=(
                same_edgar_accession(loser_meta.url, survivor_meta.url)
                if loser_meta and survivor_meta
                else False
            ),
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
            f"{name}: recall={recall:.3f} ({true_pos}/{len(positives)}) "
            f"precision={precision:.3f} false_positives={false_pos}/{len(negatives)}"
        )

    print(
        f"\nLabeled pairs: {len(positives)} true-dupe "
        f"({sum(1 for s in positives if s.pair.batch == 'A')} batch A + "
        f"{sum(1 for s in positives if s.pair.batch == 'B')} batch B), "
        f"{len(negatives)} non-dupe "
        f"({sum(1 for s in negatives if s.pair.batch == 'A')} batch A + "
        f"{sum(1 for s in negatives if s.pair.batch == 'B')} batch B)"
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
            f"  !! loser={score.pair.loser_uuid} survivor={score.pair.survivor_uuid} "
            f"{score.decision_action}:{score.decision_reason}"
        )

    print("\nNon-dupe pair scores:")
    for score in negatives:
        print(
            f"  [{score.pair.batch}] loser={score.pair.loser_uuid} survivor={score.pair.survivor_uuid} "
            f"old_j={score.old_jaccard:.3f} new_j={score.new_jaccard:.3f} "
            f"new_c={score.new_containment:.3f} dated_match={score.dated_as_of_match} "
            f"decision={score.decision_action}:{score.decision_reason}"
        )

    missed = [s for s in positives if s.decision_action == "none"]
    if missed:
        print(f"\nTrue-dupe pairs invisible to the guard ({len(missed)}):")
        for score in missed:
            print(
                f"  [{score.pair.batch}] loser={score.pair.loser_uuid} survivor={score.pair.survivor_uuid} "
                f"old_j={score.old_jaccard:.3f} new_j={score.new_jaccard:.3f} "
                f"new_c={score.new_containment:.3f} lsh={score.lsh_retrievable} "
                f"dated_match={score.dated_as_of_match} reason={score.decision_reason}"
            )


def main() -> None:
    db = build_engine_from_env()
    with db.get_engine().begin() as conn:
        scores = _score_pairs(conn)
    _report(scores)


if __name__ == "__main__":
    main()
