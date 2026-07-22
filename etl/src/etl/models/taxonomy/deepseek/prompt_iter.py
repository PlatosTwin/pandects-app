"""
Prompt-iteration harness: replay specific sections through the FP8 API with an
optional system-prompt patch, and compare baseline vs patched vs gold.

The A batch input carries the fully-rendered system+user messages per section.
This tool can inject an extra rule block into the system message (default:
immediately before the "# Output (STRICT)" header) so a candidate prompt edit
can be tested on a few hand-picked sections before it is folded into
generate_batches_hybrid.py and re-run at scale.

Run from repo root, e.g.:
  caffeinate -i etl/.venv/bin/python \
      etl/src/etl/models/taxonomy/deepseek/prompt_iter.py \
      --sids 39542708,d96b2cf7,fb74af71 --patch patches/rep_side_v1.txt
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from sqlalchemy import text

from enrichment_lib import build_engine

DEEPSEEK_DIR = Path(__file__).resolve().parent
DATA_DIR = DEEPSEEK_DIR / "data"
API_KEY = os.environ["DEEPSEEK_API_KEY"]
BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-v4-pro"
ANCHOR = "# Output (STRICT)"


def _load(path: Path) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        if r.get("error"):
            continue
        sid = r.get("section_uuid")
        if isinstance(sid, str):
            out[sid] = {str(c).strip() for c in (r.get("categories") or [])
                        if str(c).strip()}
    return out


def _batch(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        secs = r.get("sections") or []
        if not secs:
            continue
        sid = secs[0].get("section_uuid")
        if isinstance(sid, str):
            out[sid] = {"messages": r["messages"],
                        "allowed": set(r.get("allowed_category_ids") or [])}
    return out


def _tax() -> dict[str, str]:
    engine, schema = build_engine()
    with engine.begin() as conn:
        rows = conn.execute(text(
            f"SELECT standard_id, label, l2_label "
            f"FROM {schema}.taxonomy_node_meta")).mappings().all()
    return {str(r["standard_id"]): f"{r['l2_label']} > {r['label']}"
            for r in rows}


def _fmt(ids: set[str], tax: dict[str, str]) -> str:
    return " | ".join(tax.get(c, f"?{c}") for c in sorted(ids)) or "[]"


def _patch_messages(messages: list[dict[str, str]], patch: str
                    ) -> list[dict[str, str]]:
    if not patch:
        return messages
    sys_m = messages[0]["content"]
    if ANCHOR not in sys_m:
        raise SystemExit(f"anchor {ANCHOR!r} not found in system message")
    new_sys = sys_m.replace(ANCHOR, patch.rstrip() + "\n\n" + ANCHOR, 1)
    return [{"role": "system", "content": new_sys}, messages[1]]


def _call(client: OpenAI, messages: list[dict[str, str]], sid: str,
          tries: int = 3) -> tuple[set[str], str | None]:
    """Call FP8; retry on empty content (a known bare-API quirk)."""
    last = "no attempt"
    for _ in range(tries):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=messages,  # pyright: ignore[reportArgumentType]
                temperature=0.0, max_tokens=2000,
                response_format={"type": "json_object"})
        except Exception as e:  # noqa: BLE001
            last = f"api_error: {e}"
            continue
        raw = resp.choices[0].message.content or ""
        if not raw.strip():
            last = "empty_content (retrying)"
            continue
        try:
            asg = (json.loads(raw).get("assignments") or [])
            if not asg:
                last = f"no-assignments :: {raw[:160]!r}"
                continue
            a = asg[0]
            return {str(c).strip() for c in (a.get("categories") or [])
                    if str(c).strip()}, None
        except (json.JSONDecodeError, AttributeError, TypeError) as e:
            last = f"parse_error: {e} :: {raw[:160]!r}"
    return set(), last


def _article(messages: list[dict[str, str]]) -> str:
    m = re.search(r"article_title:\s*(.*)", messages[1]["content"])
    return m.group(1).strip() if m else "?"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sids", required=True,
                   help="comma-separated 8-char prefixes or full uuids")
    p.add_argument("--batch", default=str(DATA_DIR / "gold_hybrid_A_2026-05-17.jsonl"))
    p.add_argument("--pred", default=str(DATA_DIR / "gold_hybrid_A_2026-05-17.out.jsonl"))
    p.add_argument("--gold", default=str(DATA_DIR / "gold_new_taxonomy.out.jsonl"))
    p.add_argument("--patch", default="",
                   help="path to a text file with the rule block to inject")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    logging.basicConfig(level="INFO", stream=sys.stdout,
                        format="%(asctime)s %(levelname)s :: %(message)s")
    log = logging.getLogger("prompt_iter")

    gold = _load(Path(args.gold))
    fp4 = _load(Path(args.pred))
    batch = _batch(Path(args.batch))
    tax = _tax()
    patch = Path(args.patch).read_text(encoding="utf-8") if args.patch else ""

    want = [s.strip() for s in args.sids.split(",") if s.strip()]
    sids: list[str] = []
    for w in want:
        match = [k for k in batch if k == w or k.startswith(w)]
        if not match:
            log.warning("no batch row for %s", w)
        sids.extend(match)

    log.info("patch=%s  sections=%d  model=%s",
             args.patch or "(none)", len(sids), MODEL)
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    def _job(sid: str) -> tuple[str, set[str], str | None]:
        msgs = _patch_messages(batch[sid]["messages"], patch)
        cats, err = _call(client, msgs, sid)
        return sid, cats, err

    res: dict[str, tuple[set[str], str | None]] = {}
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_job, s): s for s in sids}
        for fut in as_completed(futs):
            sid, cats, err = fut.result()
            res[sid] = (cats, err)

    fixed = broke = same = 0
    log.info("=" * 80)
    for sid in sids:
        g, f4 = gold.get(sid, set()), fp4.get(sid, set())
        new, err = res[sid]
        allowed = batch[sid]["allowed"]
        bad = {c for c in new if c not in allowed}
        base_ok, new_ok = (f4 == g), (new == g)
        if err:
            tag = f"API-ERR {err[:70]}"
        elif new_ok and not base_ok:
            tag, _ = "✓ FIXED (now == gold)", (fixed := fixed + 1)
        elif not new_ok and base_ok:
            tag, _ = "✗ REGRESSED (was right)", (broke := broke + 1)
        elif new_ok and base_ok:
            tag, _ = "= still correct", (same := same + 1)
        else:
            tag, _ = "· still wrong", (same := same + 1)
        log.info("%s  %s", sid[:8], tag)
        log.info("   article : %s", _article(batch[sid]["messages"]))
        log.info("   gold    : %s", _fmt(g, tax))
        log.info("   baseline: %s", _fmt(f4, tax))
        log.info("   patched : %s%s", _fmt(new, tax),
                 f"  [INVALID {bad}]" if bad else "")
    log.info("=" * 80)
    log.info("FIXED=%d  REGRESSED=%d  UNCHANGED-verdict=%d", fixed, broke, same)
    return 0


if __name__ == "__main__":
    sys.exit(main())
