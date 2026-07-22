"""
Random eligible-section sampler (shared by gold generation and the hybrid
batch generator).

Eligibility (per the maintainer's spec):
  - section is attached to latest + verified XML
  - gold-label presence is IGNORED (the old labels are stale)
  - random agreements, then random sections within them
  - cleaned body word count in [min_words, max_words] (default 10..3000)
  - EXCLUDE if section_title in {Definitions, Certain Definitions,
    Defined Terms}; ELSE IF section_title is null/blank and article_title in
    {Definitions, Defined Terms} -> EXCLUDE
  - section_order IS NULL is eligible ONLY if it is the sole section in its
    parent article (no sibling sections); a non-null section_order is fine

Deterministic for a given --seed (MariaDB RAND(seed) for the agreement pool,
Python random for the final section shuffle).
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from dataclasses import dataclass
from typing import Any

from sqlalchemy import bindparam, text

from enrichment_lib import build_engine

_TAG_RE = re.compile(r"<[^>]+>")
_SPACE_RE = re.compile(r"\s+")
_LEAD_RE = re.compile(r"(?i)^\s*(section|article)\s+[0-9ivxlcdm.]+\s*[.\-]?\s*")

_SECTION_TITLE_EXCL = {"definitions", "certain definitions", "defined terms"}
_ARTICLE_TITLE_EXCL = {"definitions", "defined terms"}


def clean_body(fragment: str) -> str:
    if not fragment:
        return ""
    return _SPACE_RE.sub(" ", _TAG_RE.sub(" ", fragment)).strip()


def _title_forms(title: str | None) -> set[str]:
    """Trimmed-lowercase title plus a leading-'Section X'/'Article X'-stripped
    variant, so 'Section 1.1 Definitions' / 'ARTICLE I DEFINITIONS' match."""
    if not title:
        return set()
    raw = title.strip().lower()
    stripped = _LEAD_RE.sub("", title).strip().lower()
    return {f for f in (raw, stripped) if f}


@dataclass(frozen=True)
class SelectedSection:
    section_uuid: str
    agreement_uuid: str
    article_title: str
    section_title: str
    body: str
    article_order: int | None
    section_order: int | None


def _is_definitions_excluded(section_title: str | None, article_title: str | None) -> bool:
    st = (section_title or "").strip()
    if st:
        if _title_forms(section_title) & _SECTION_TITLE_EXCL:
            return True
        return False
    # section_title is null/blank -> fall back to the article-title rule
    return bool(_title_forms(article_title) & _ARTICLE_TITLE_EXCL)


def _article_key(
    agreement_uuid: str, article_order: Any, article_title: Any
) -> tuple[Any, ...]:
    if article_order is not None:
        return (agreement_uuid, "O", int(article_order))
    return (agreement_uuid, "T", (article_title or "").strip().lower())


def select_sections(
    num_sections: int = 750,
    seed: int = 20260517,
    pool_agreements: int = 250,
    min_words: int = 10,
    max_words: int = 3000,
    log: logging.Logger | None = None,
) -> list[SelectedSection]:
    log = log or logging.getLogger("section_select")
    engine, schema = build_engine()

    with engine.begin() as conn:
        # MariaDB RAND(seed) is deterministic; this returns a random pool of
        # agreements that have at least one latest+verified section.
        agreement_uuids = [
            str(r[0])
            for r in conn.execute(
                text(
                    f"""
                    SELECT agreement_uuid FROM (
                        SELECT DISTINCT s.agreement_uuid AS agreement_uuid
                        FROM {schema}.sections s
                        JOIN {schema}.xml x
                          ON x.agreement_uuid = s.agreement_uuid
                         AND x.version = s.xml_version
                        WHERE x.latest = 1 AND x.status = 'verified'
                    ) t
                    ORDER BY RAND(:seed)
                    LIMIT :pool
                    """
                ),
                {"seed": seed, "pool": pool_agreements},
            ).all()
        ]
        if not agreement_uuids:
            log.warning("no candidate agreements")
            return []

        rows = conn.execute(
            text(
                f"""
                SELECT s.section_uuid, s.agreement_uuid,
                       s.article_title, s.section_title,
                       s.article_order, s.section_order, s.xml_content
                FROM {schema}.sections s
                JOIN {schema}.xml x
                  ON x.agreement_uuid = s.agreement_uuid
                 AND x.version = s.xml_version
                WHERE x.latest = 1 AND x.status = 'verified'
                  AND s.agreement_uuid IN :agreements
                """
            ).bindparams(bindparam("agreements", expanding=True)),
            {"agreements": tuple(agreement_uuids)},
        ).mappings().all()

    # Article sibling counts over the FULL latest+verified section set of each
    # pooled agreement (so "sole section in its parent article" is exact).
    article_counts: dict[tuple[Any, ...], int] = {}
    for r in rows:
        k = _article_key(r["agreement_uuid"], r["article_order"], r["article_title"])
        article_counts[k] = article_counts.get(k, 0) + 1

    eligible: list[SelectedSection] = []
    drop_def = drop_words = drop_nullorder = 0
    for r in rows:
        if _is_definitions_excluded(r["section_title"], r["article_title"]):
            drop_def += 1
            continue
        if r["section_order"] is None:
            k = _article_key(r["agreement_uuid"], r["article_order"], r["article_title"])
            if article_counts.get(k, 0) != 1:
                drop_nullorder += 1
                continue
        body = clean_body(str(r["xml_content"] or ""))
        wc = len(body.split())
        if wc < min_words or wc > max_words:
            drop_words += 1
            continue
        eligible.append(
            SelectedSection(
                section_uuid=str(r["section_uuid"]),
                agreement_uuid=str(r["agreement_uuid"]),
                article_title=str(r["article_title"] or ""),
                section_title=str(r["section_title"] or ""),
                body=body,
                article_order=(None if r["article_order"] is None else int(r["article_order"])),
                section_order=(None if r["section_order"] is None else int(r["section_order"])),
            )
        )

    rng = random.Random(seed)
    rng.shuffle(eligible)
    chosen = eligible[:num_sections]

    log.info(
        "pool=%d agreements, %d sections fetched; dropped def=%d nullorder=%d "
        "words=%d; eligible=%d; selected=%d",
        len(agreement_uuids), len(rows), drop_def, drop_nullorder,
        drop_words, len(eligible), len(chosen),
    )
    if len(chosen) < num_sections:
        log.warning(
            "only %d eligible (< requested %d); raise --pool-agreements",
            len(chosen), num_sections,
        )
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-sections", type=int, default=750)
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--pool-agreements", type=int, default=250)
    parser.add_argument("--min-words", type=int, default=10)
    parser.add_argument("--max-words", type=int, default=3000)
    parser.add_argument("--out", default=None,
                        help="Optional JSONL dump (section_uuid, agreement_uuid, titles).")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=args.log_level.upper(),
        format="%(asctime)s %(levelname)s :: %(message)s",
        stream=sys.stdout,
    )
    sel = select_sections(
        num_sections=args.num_sections,
        seed=args.seed,
        pool_agreements=args.pool_agreements,
        min_words=args.min_words,
        max_words=args.max_words,
    )
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for s in sel:
                fh.write(json.dumps({
                    "section_uuid": s.section_uuid,
                    "agreement_uuid": s.agreement_uuid,
                    "article_title": s.article_title,
                    "section_title": s.section_title,
                    "article_order": s.article_order,
                    "section_order": s.section_order,
                    "words": len(s.body.split()),
                }, ensure_ascii=False) + "\n")
        logging.getLogger("section_select").info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
