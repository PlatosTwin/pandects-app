"""
Generate a ready-to-ingest batch of sections for DeepSeek taxonomy inference on HPC.

Pulls N agreements' worth of sections from the local MariaDB instance,
strips XML to plain text, fetches a taxonomy snapshot, and writes one
self-contained prompt per section to:

  data/<name>.jsonl   one JSON object per line, with `section_uuid`,
                      `agreement_uuid`, and a fully-built `messages` list
                      (system instructions with taxonomy baked in + user
                      payload with the section). batch_inference.py only
                      needs to apply the model's chat template per row.

Eligibility: section is attached to latest+verified XML, has no gold label
yet, and its body text (XML tags stripped, inner text preserved) contains
between --min-words and --max-words words. No title filters.

Run from repo root, e.g.:
  caffeinate -i etl/.venv/bin/python \
      etl/src/etl/models/taxonomy/deepseek/generate_batches.py \
      --num-agreements 5 --name smoke_2026-04-25
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from sqlalchemy import bindparam, create_engine, text


_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")

REPO_ROOT = Path(__file__).resolve().parents[6]
DEEPSEEK_DIR = Path(__file__).resolve().parent
DATA_DIR = DEEPSEEK_DIR / "data"
BACKEND_ENV = REPO_ROOT / "backend" / ".env"


def _clean_body(fragment: str) -> str:
    """Strip XML tag markup (e.g. <text>, <page>, <pageUUID>), keep inner text."""
    if not fragment:
        return ""
    return _SPACE_RE.sub(" ", _TAG_RE.sub(" ", fragment)).strip()


def _word_count(body: str) -> int:
    return len(body.split()) if body else 0


def _build_engine() -> Any:
    load_dotenv(BACKEND_ENV)
    user = os.environ["MARIADB_USER"]
    password = os.environ["MARIADB_PASSWORD"]
    host = os.environ["MARIADB_HOST"]
    database = os.environ["MARIADB_DATABASE"]
    url = f"mysql+mysqldb://{user}:{password}@{host}/{database}?charset=utf8mb4"
    return create_engine(url, pool_pre_ping=True), database


def _fetch_taxonomy(conn: Any, schema: str) -> list[dict[str, Any]]:
    rows = conn.execute(
        text(
            f"""
            SELECT
                l1.standard_id AS l1_standard_id, l1.label AS l1_label,
                l2.standard_id AS l2_standard_id, l2.label AS l2_label,
                l3.standard_id AS l3_standard_id, l3.label AS l3_label
            FROM {schema}.taxonomy_l1 l1
            LEFT JOIN {schema}.taxonomy_l2 l2 ON l1.standard_id = l2.parent_id
            LEFT JOIN {schema}.taxonomy_l3 l3 ON l2.standard_id = l3.parent_id
            ORDER BY l1.standard_id, l2.standard_id, l3.standard_id
            """
        )
    ).mappings()
    return [dict(row) for row in rows]


def _select_candidate_agreements(conn: Any, schema: str, limit: int) -> list[str]:
    """Agreement UUIDs (ordered) that have at least one section on latest+verified XML.

    Returns an over-selected pool; the final N is chosen after Python-side
    word-count filtering."""
    rows = conn.execute(
        text(
            f"""
            SELECT DISTINCT s.agreement_uuid
            FROM {schema}.sections s
            JOIN {schema}.xml x
              ON x.agreement_uuid = s.agreement_uuid AND x.version = s.xml_version
            WHERE x.latest = 1 AND x.status = 'verified'
              AND (s.section_standard_id_gold_label IS NULL
                   OR TRIM(s.section_standard_id_gold_label) = '')
            ORDER BY s.agreement_uuid
            LIMIT :lim
            """
        ),
        {"lim": limit},
    ).mappings().fetchall()
    return [str(r["agreement_uuid"]) for r in rows]


def _fetch_sections(
    conn: Any, schema: str, agreement_uuids: list[str]
) -> list[dict[str, Any]]:
    if not agreement_uuids:
        return []
    query = text(
        f"""
        SELECT
            s.section_uuid, s.agreement_uuid,
            s.article_title, s.section_title,
            s.article_order, s.section_order,
            s.xml_content
        FROM {schema}.sections s
        JOIN {schema}.xml x
          ON x.agreement_uuid = s.agreement_uuid AND x.version = s.xml_version
        WHERE s.agreement_uuid IN :agreements
          AND x.latest = 1 AND x.status = 'verified'
          AND s.section_order IS NOT NULL
          AND (s.section_standard_id_gold_label IS NULL
               OR TRIM(s.section_standard_id_gold_label) = '')
        ORDER BY s.agreement_uuid,
                 COALESCE(s.article_order, -1),
                 COALESCE(s.section_order, -1),
                 s.section_uuid
        """
    ).bindparams(bindparam("agreements", expanding=True))
    rows = conn.execute(query, {"agreements": tuple(agreement_uuids)}).mappings().fetchall()
    return [dict(r) for r in rows]


def _build_system_instructions(taxonomy_json: list[dict[str, Any]]) -> str:
    taxonomy_text = json.dumps(taxonomy_json, ensure_ascii=False)
    return f"""# Identity
You are an expert mergers and acquisitions (M&A) lawyer who categorizes agreement sections against a fixed taxonomy.

# Inputs
1) A taxonomy JSON: a flat list of rows with `l1_standard_id`/`l1_label`, optional `l2_standard_id`/`l2_label`, optional `l3_standard_id`/`l3_label`. Each non-null `standard_id` is a valid leaf at its level.
2) One or more sections, each with: `section_uuid`, `article_title`, `section_title`, `section_text` (full body).

# Task
For EACH input section, assign zero or more taxonomy categories based on its title and full text.

# Rules
- Assign at the LOWEST applicable level: L3 if it fits; otherwise L2; never L1. Never return both a parent and its child.
- Use ONLY `standard_id` values that appear verbatim in the taxonomy JSON.
- Return `[]` if no category clearly applies.
- Prefer a single best-fitting category. Add more only when the section text clearly covers distinct topics (e.g., a section that addresses both No-Shops and Fiduciary Outs).
- Use `article_title` to disambiguate two-homed concepts (Conditions vs Covenants vs Reps vs Termination vs Indemnification vs Dispute Resolution).
- For named statutes / authorities / instruments / terms-of-art (HSR, DGCL, ERISA, CFIUS, SEC, DOJ, FTC, Proxy Statement, S-4, "go-shop", "RWI", "appraisal rights", etc.), pick the specific leaf when the text supports it.
- Be judicious: do not stretch categories, do not construe them too narrowly.

# Classification Discipline
- Prefer `[]` over a weak semantic neighbor. Return a category only when the section's operative text clearly contains that category's core legal concept.
- Do not use "Other / Unclassified" as a fallback. Use it only when the section is substantively a residual or unclassified provision and an explicit residual label is more informative than `[]`.
- Use `section_title` and `article_title` for disambiguation, but do not assign a category based on title or cross-references alone.
- Add multiple categories only when the section independently contains multiple operative legal topics. Do not add a second category merely because a related word or cross-reference appears.
- Respect legal context: representations, covenants, conditions, termination, indemnity, dispute resolution, financing, and transaction mechanics are distinct taxonomy surfaces. Prefer the leaf matching the section's actual legal function.
- For clauses that are adjacent to a taxonomy concept but not actually about that concept, return `[]` rather than stretching the nearest category.

# Silent Check
Before finalizing each assignment, silently check:
1) What is the section's legal function?
2) Which operative sentence supports each selected category?
3) Is any selected category only a loose neighbor, title-only match, or cross-reference-only match?
4) If no category is clearly supported, return `[]`.

# Output (STRICT)
Return one JSON object of the form:
{{"assignments": [{{"section_uuid": "<uuid>", "categories": ["<standard_id>", ...]}}, ...]}}
- Include EXACTLY one entry per input `section_uuid`, in the same order as the input.
- No additional keys, prose, or commentary.

# Taxonomy
{taxonomy_text}
""".strip()


def _build_user_payload(sections: list[dict[str, Any]]) -> str:
    n = len(sections)
    header = f"Categorize the following {n} section{'s' if n != 1 else ''}."
    parts: list[str] = [header]
    for idx, section in enumerate(sections, start=1):
        article = (section.get("article_title") or "N/A").strip() or "N/A"
        title = (section.get("section_title") or "N/A").strip() or "N/A"
        parts.append(
            f"=== Section {idx} ===\n"
            f"section_uuid: {section['section_uuid']}\n"
            f"article_title: {article}\n"
            f"section_title: {title}\n"
            f"section_text:\n{section['_body']}"
        )
    return "\n\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--num-agreements",
        type=int,
        required=True,
        help="Number of agreements' worth of eligible sections to include.",
    )
    parser.add_argument(
        "--name",
        default="batch",
        help="Output filename stem (e.g. 'smoke_2026-04-25' -> data/smoke_2026-04-25.jsonl)",
    )
    parser.add_argument("--min-words", type=int, default=10,
                        help="Minimum word count (inclusive) for the cleaned section body.")
    parser.add_argument("--max-words", type=int, default=1000,
                        help="Maximum word count (inclusive) for the cleaned section body.")
    parser.add_argument("--candidate-multiplier", type=int, default=3,
                        help="Over-select agreements by this factor to leave room for "
                             "Python-side word-count filtering.")
    parser.add_argument("--sections-per-request", type=int, default=1,
                        help="Group this many eligible sections into a single prompt/request. "
                             "Higher N amortizes the taxonomy prefix across more sections; "
                             "max_new_tokens on the inference side must scale roughly linearly.")
    parser.add_argument("--max-sections", type=int, default=None,
                        help="Hard cap on the total number of eligible sections written "
                             "(applied after agreement selection). Useful to size the run "
                             "exactly: --max-sections 500 --sections-per-request 2 -> 250 rows.")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s :: %(message)s",
        stream=sys.stdout,
    )
    log = logging.getLogger("generate_batches")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DATA_DIR / f"{args.name}.jsonl"

    target_n = args.num_agreements
    candidate_limit = max(target_n * args.candidate_multiplier, target_n + 50)

    engine, schema = _build_engine()
    with engine.begin() as conn:
        log.info("selecting up to %d candidate agreements (target %d)",
                 candidate_limit, target_n)
        candidate_uuids = _select_candidate_agreements(conn, schema, candidate_limit)
        log.info("got %d candidate agreements", len(candidate_uuids))
        if not candidate_uuids:
            log.warning("no candidate agreements found; nothing to write")
            return 0

        log.info("fetching sections for candidate agreements")
        raw_sections = _fetch_sections(conn, schema, candidate_uuids)
        log.info("fetched %d raw sections", len(raw_sections))

        log.info("fetching taxonomy snapshot")
        taxonomy_json = _fetch_taxonomy(conn, schema)
        log.info("fetched %d taxonomy rows", len(taxonomy_json))

    eligible: list[dict[str, Any]] = []
    skipped_too_short = 0
    skipped_too_long = 0
    for row in raw_sections:
        body = _clean_body(str(row.get("xml_content") or ""))
        wc = _word_count(body)
        if wc < args.min_words:
            skipped_too_short += 1
            continue
        if wc > args.max_words:
            skipped_too_long += 1
            continue
        row["_body"] = body
        eligible.append(row)

    log.info(
        "word-count filter: kept %d / %d sections (skipped %d <%d words, %d >%d words)",
        len(eligible), len(raw_sections),
        skipped_too_short, args.min_words,
        skipped_too_long, args.max_words,
    )

    selected: list[dict[str, Any]] = []
    agreements_in_order: list[str] = []
    seen_agreements: set[str] = set()
    for row in eligible:
        ag = str(row["agreement_uuid"])
        if ag not in seen_agreements:
            if len(seen_agreements) >= target_n:
                break
            seen_agreements.add(ag)
            agreements_in_order.append(ag)
        selected.append(row)
        if args.max_sections is not None and len(selected) >= args.max_sections:
            break

    if len(agreements_in_order) < target_n:
        log.warning(
            "only %d agreements had eligible sections in candidate pool (target %d); "
            "consider raising --candidate-multiplier",
            len(agreements_in_order), target_n,
        )

    system_instructions = _build_system_instructions(taxonomy_json)
    spr = max(1, args.sections_per_request)
    chunks: list[list[dict[str, Any]]] = [
        selected[i : i + spr] for i in range(0, len(selected), spr)
    ]

    with out_path.open("w", encoding="utf-8") as fh:
        for chunk in chunks:
            user_payload = _build_user_payload(chunk)
            payload = {
                "sections": [
                    {
                        "section_uuid": str(s["section_uuid"]),
                        "agreement_uuid": str(s["agreement_uuid"]),
                    }
                    for s in chunk
                ],
                "messages": [
                    {"role": "system", "content": system_instructions},
                    {"role": "user", "content": user_payload},
                ],
            }
            fh.write(json.dumps(payload, ensure_ascii=False) + "\n")

    log.info(
        "wrote %s (%d sections across %d agreements, %d rows @ %d sections/request, "
        "%d taxonomy entries baked in)",
        out_path,
        len(selected),
        len(agreements_in_order),
        len(chunks),
        spr,
        len(taxonomy_json),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
