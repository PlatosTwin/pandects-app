"""
Retrieval-limited DeepSeek batch generator.

Instead of baking the full 140-leaf taxonomy into every prompt, this runs the
hybrid retriever per section and puts ONLY that section's top-K enriched
candidates (plus their parent-L2 backoffs) in the system prompt. The output
JSONL is byte-compatible with batch_inference.py / run_inference.sbatch (one
ready-to-ingest row per section: {"sections":[...], "messages":[sys,user]}).

Because each section has its own candidate set, there is exactly one section
per request (spr is forced to 1).

A sidecar data/<name>.candidates.jsonl records the retrieved set + per-signal
scores per section — the seed data for the future reranker stage.

Section source:
  --from-gold PATH   use the exact sections in a gold file (default; lets you
                     compare DeepSeek output 1:1 against gold_generate.py)
  --num-sections N   otherwise, fresh random selection (section_select rules)

Run from repo root (needs DB + VOYAGE_API_KEY):
  caffeinate -i etl/.venv/bin/python \
      etl/src/etl/models/taxonomy/deepseek/generate_batches_hybrid.py \
      --from-gold etl/src/etl/models/taxonomy/deepseek/data/gold_new_taxonomy.out.jsonl \
      --name gold_hybrid_2026-05-17
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import bindparam, text

from enrichment_lib import build_engine
from generate_batches import _build_user_payload
from retrieval import HybridRetriever, Node, RetrievalConfig, Section
from section_select import clean_body, select_sections

DEEPSEEK_DIR = Path(__file__).resolve().parent
DATA_DIR = DEEPSEEK_DIR / "data"


_SYSTEM_TEMPLATE = """# Identity
You are an expert mergers and acquisitions (M&A) lawyer who classifies a
single agreement section against a SHORTLIST of candidate taxonomy categories.

{tx_context}# Inputs
1) A shortlist of candidate categories, pre-selected for THIS section by a
   retrieval system. Each entry shows its `standard_id`, level (L3 leaf or L2
   group), full path, scope, key terms, representative language, and — where
   given — "Not this if" disambiguators.
2) One section: `section_uuid`, `article_title`, `section_title`,
   `section_text` (full body).

# Task
Assign zero or more categories to the section, choosing ONLY from the
candidate `standard_id` values listed below.

# Rules
- Choose ONLY `standard_id` values that appear verbatim in the candidate list.
  Never invent an id; never use one not listed.
- Assign at the LOWEST applicable level: an L3 leaf if one fits; otherwise the
  L2 group; never return both a parent and its child.
- Prefer a single best-fitting category. Add more only when the section text
  clearly covers distinct operative topics.
- Return `[]` if none of the candidates truly fits. The shortlist is
  retrieval-based and may not contain a correct category — do not force a
  weak or merely-adjacent match. `[]` is the correct answer for boilerplate
  or residual text with no fitting candidate.
- Respect the "Not this if" disambiguators: if a candidate's exclusion
  applies, do not choose it.
- Choose the CATEGORY from the operative text, not from titles or
  cross-references alone; use `article_title`/`section_title` only to
  disambiguate. (This governs category choice only — the representation-side
  rule above is unaffected: the representing party is fixed by `article_title`
  and overrides the section text for the Buyer-rep vs Company-rep choice.){rationale_rule}

# Output (STRICT)
{output_block}

# Candidate Categories (choose only from these)
{candidates}
"""

_RATIONALE_RULE = (
    "\n- Reason first: in `rationale` (one or two sentences) state the "
    "section's operative legal function and the single best-matching "
    "candidate (or that none fits); your `categories` must be consistent "
    "with that rationale."
)

_OUTPUT_A = (
    'Return one JSON object:\n'
    '{"assignments": [{"section_uuid": "<uuid>", '
    '"categories": ["<standard_id>", ...]}]}\n'
    "- Exactly one entry, for the input section_uuid.\n"
    "- No prose, no extra keys."
)

_OUTPUT_B = (
    'Return one JSON object:\n'
    '{"assignments": [{"rationale": "<one or two sentences>", '
    '"section_uuid": "<uuid>", "categories": ["<standard_id>", ...]}]}\n'
    "- Exactly one entry, for the input section_uuid.\n"
    "- Emit `rationale` first, then `section_uuid`, then `categories`. "
    "No other keys; no prose outside the JSON."
)


def _format_candidates(
    cand_ids: list[tuple[str, bool, float]], node_by_id: dict[str, Node]
) -> str:
    """cand_ids: list of (standard_id, is_backoff, score) already ordered
    (primary L3 by score, then L2 backoffs)."""
    primary: list[str] = []
    backoff: list[str] = []
    for sid, is_backoff, _score in cand_ids:
        n = node_by_id[sid]
        block = [f"[{sid}] ({n.level.upper()}) {n.enriched_text}"]
        if n.exclusion_cues:
            block.append("  Not this if: " + " | ".join(n.exclusion_cues))
        (backoff if is_backoff else primary).append("\n".join(block))
    out = ["## Specific leaves (prefer one of these)"]
    out.extend(primary)
    if backoff:
        out.append(
            "\n## Broader groups (use ONLY if no specific leaf above fits)"
        )
        out.extend(backoff)
    return "\n\n".join(out)


def _gold_section_ids(path: Path) -> list[tuple[str, str]]:
    """(section_uuid, agreement_uuid) for every non-errored gold row,
    including the model's []-abstentions (tests DeepSeek's discipline too)."""
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("error"):
                continue
            sid = str(r["section_uuid"])
            if sid in seen:
                continue
            seen.add(sid)
            out.append((sid, str(r.get("agreement_uuid") or "")))
    return out


def _fetch_parties(agreement_uuids: list[str]) -> dict[str, dict[str, str]]:
    """agreement_uuid -> {acquirer, acquirer_type, target, target_type}.

    Only rows with BOTH acquirer and target non-blank are returned; callers
    treat a missing agreement as 'no transaction context' (never guess a
    role — a wrong role would induce the very mirror mis-siding we are
    trying to remove)."""
    if not agreement_uuids:
        return {}
    engine, schema = build_engine()
    out: dict[str, dict[str, str]] = {}
    uniq = list(dict.fromkeys(agreement_uuids))
    with engine.begin() as conn:
        for i in range(0, len(uniq), 1000):
            chunk = uniq[i : i + 1000]
            rows = conn.execute(
                text(
                    f"""
                    SELECT agreement_uuid, acquirer, acquirer_type,
                           target, target_type
                    FROM {schema}.agreements
                    WHERE agreement_uuid IN :a
                    """
                ).bindparams(bindparam("a", expanding=True)),
                {"a": tuple(chunk)},
            ).mappings().all()
            for r in rows:
                acq = str(r["acquirer"] or "").strip()
                tgt = str(r["target"] or "").strip()
                if acq and tgt:
                    out[str(r["agreement_uuid"])] = {
                        "acquirer": acq,
                        "acquirer_type": str(r["acquirer_type"] or "").strip(),
                        "target": tgt,
                        "target_type": str(r["target_type"] or "").strip(),
                    }
    return out


def _render_tx_context(parties: dict[str, str] | None) -> str:
    """The '# Transaction context' block, or '' when metadata is absent."""
    if not parties:
        return ""
    acq, tgt = parties["acquirer"], parties["target"]
    at = f' ({parties["acquirer_type"]})' if parties.get("acquirer_type") else ""
    tt = f' ({parties["target_type"]})' if parties.get("target_type") else ""
    return (
        "# Transaction context\n"
        f"Acquirer / Parent / Buyer side: {acq}{at}.\n"
        f"Target / Company / Seller side: {tgt}{tt}.\n"
        "For any representation or warranty, the REPRESENTING PARTY is fixed "
        "by the `article_title`, not by the prose. Resolve it in this order: "
        "(1) if `article_title` names a side generically (\"...OF BUYER\", "
        "\"...OF PARENT\", \"...OF THE COMPANY\", \"...OF SELLER\"), that side "
        "is the representing party; (2) else if it names an entity, map it via "
        "the parties above — the Acquirer/Parent/Buyer side means the Buyer, "
        "the Target/Company/Seller side means the Company. Merger Sub, Holdco, "
        "and acquisition vehicles are the Buyer side; a target business, its "
        "subsidiaries, or selling principals are the Company side. Then choose "
        "the candidate whose suffix matches: '(Buyer rep)' if the representing "
        "party is the Buyer/Parent side, '(Company rep)' if it is the "
        "Target/Company/Seller side. This article-derived side OVERRIDES any "
        "contrary inference from the section text.\n"
        "Disclaimer / non-reliance / \"no other representations\" clauses are "
        "written reciprocally — they describe representations made (or not "
        "made) by BOTH parties. Do NOT infer the side from that reciprocal "
        "prose; the side is still the representing party per the rule above "
        "(e.g., a non-reliance clause inside \"REPRESENTATIONS AND WARRANTIES "
        "OF BUYER\" is the '(Buyer rep)' leaf, even though its text disclaims "
        "the Company's reps).\n\n"
    )


def _fetch_section_text(
    section_uuids: list[str],
) -> dict[str, dict[str, str]]:
    engine, schema = build_engine()
    out: dict[str, dict[str, str]] = {}
    with engine.begin() as conn:
        for i in range(0, len(section_uuids), 1000):
            chunk = section_uuids[i : i + 1000]
            rows = conn.execute(
                text(
                    f"""
                    SELECT section_uuid, agreement_uuid,
                           article_title, section_title, xml_content
                    FROM {schema}.sections
                    WHERE section_uuid IN :uuids
                    """
                ).bindparams(bindparam("uuids", expanding=True)),
                {"uuids": tuple(chunk)},
            ).mappings().all()
            for r in rows:
                out[str(r["section_uuid"])] = {
                    "agreement_uuid": str(r["agreement_uuid"] or ""),
                    "article_title": str(r["article_title"] or ""),
                    "section_title": str(r["section_title"] or ""),
                    "body": clean_body(str(r["xml_content"] or "")),
                }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-gold", help="Gold JSONL; use its exact sections.")
    src.add_argument("--num-sections", type=int,
                     help="Fresh random selection instead of a gold file.")
    parser.add_argument("--name", required=True,
                        help="Output stem -> data/<name>.jsonl (+ .candidates.jsonl)")
    parser.add_argument("--variant", choices=["A", "B"], default="A",
                        help="A: direct answer. B: forced `rationale` field "
                             "decoded before `categories` (reason-then-commit).")
    parser.add_argument("--k", type=int, default=50,
                        help="Top-K L3 candidates per section.")
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--pool-agreements", type=int, default=250)
    parser.add_argument("--min-words", type=int, default=10)
    parser.add_argument("--max-words", type=int, default=3000)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s :: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("generate_batches_hybrid")

    if args.from_gold:
        ids = _gold_section_ids(Path(args.from_gold))
        log.info("from-gold: %d unique sections", len(ids))
        meta = _fetch_section_text([s for s, _ in ids])
        sections: list[Section] = []
        agg: dict[str, str] = {}
        for sid, ag in ids:
            m = meta.get(sid)
            if not m:
                log.warning("section %s not found in DB; skipping", sid)
                continue
            sections.append(Section(m["article_title"], m["section_title"],
                                    m["body"], sid))
            agg[sid] = ag or m["agreement_uuid"]
    else:
        sel = select_sections(
            num_sections=args.num_sections, seed=args.seed,
            pool_agreements=args.pool_agreements,
            min_words=args.min_words, max_words=args.max_words, log=log,
        )
        sections = [Section(s.article_title, s.section_title, s.body,
                            s.section_uuid) for s in sel]
        agg = {s.section_uuid: s.agreement_uuid for s in sel}

    if not sections:
        log.error("no sections to process")
        return 1

    retriever = HybridRetriever.from_db(
        RetrievalConfig(k=args.k, include_parent_l2=True)
    )
    node_by_id = {n.standard_id: n for n in retriever.nodes}

    log.info("retrieving candidates for %d sections (k=%d) ...",
             len(sections), args.k)
    all_cands = retriever.retrieve_batch(sections)

    want_rationale = args.variant == "B"
    rationale_rule = _RATIONALE_RULE if want_rationale else ""
    output_block = _OUTPUT_B if want_rationale else _OUTPUT_A

    parties = _fetch_parties([agg.get(s.section_uuid or "", "") for s in sections])
    n_with_tx = sum(
        1 for s in sections if agg.get(s.section_uuid or "", "") in parties
    )
    log.info("variant=%s (want_rationale=%s, categories enum-pinned per row); "
             "transaction context on %d/%d sections (%.1f%%)",
             args.variant, want_rationale, n_with_tx, len(sections),
             100.0 * n_with_tx / len(sections) if sections else 0.0)

    out_path = DATA_DIR / f"{args.name}.jsonl"
    cand_path = DATA_DIR / f"{args.name}.candidates.jsonl"
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as bf, \
         cand_path.open("w", encoding="utf-8") as cf:
        for sec, cands in zip(sections, all_cands):
            sid = sec.section_uuid or ""
            ordered = [(c.standard_id, c.is_backoff, c.score) for c in cands]
            system = _SYSTEM_TEMPLATE.format(
                tx_context=_render_tx_context(parties.get(agg.get(sid, ""))),
                rationale_rule=rationale_rule,
                output_block=output_block,
                candidates=_format_candidates(ordered, node_by_id),
            )
            user = _build_user_payload([{
                "section_uuid": sid,
                "agreement_uuid": agg.get(sid, ""),
                "article_title": sec.article_title,
                "section_title": sec.section_title,
                "_body": sec.section_text,
            }])
            bf.write(json.dumps({
                "sections": [{"section_uuid": sid,
                              "agreement_uuid": agg.get(sid, "")}],
                "allowed_category_ids": [c.standard_id for c in cands],
                "want_rationale": want_rationale,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            }, ensure_ascii=False) + "\n")
            cf.write(json.dumps({
                "section_uuid": sid,
                "agreement_uuid": agg.get(sid, ""),
                "candidates": [
                    {
                        "standard_id": c.standard_id,
                        "level": c.level,
                        "label": c.label,
                        "l1_label": c.l1_label,
                        "l2_label": c.l2_label,
                        "score": round(c.score, 6),
                        "dense": round(c.dense, 6),
                        "sparse": round(c.sparse, 6),
                        "keyword": round(c.keyword, 6),
                        "is_backoff": c.is_backoff,
                    }
                    for c in cands
                ],
            }, ensure_ascii=False) + "\n")

    n_cand = [len(c) for c in all_cands]
    log.info(
        "wrote %s (%d rows, 1 section/row) and %s\n"
        "candidates/section: min=%d max=%d mean=%.1f",
        out_path, len(sections), cand_path,
        min(n_cand), max(n_cand), sum(n_cand) / len(n_cand),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
