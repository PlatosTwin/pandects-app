# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportAny=false, reportExplicitAny=false
"""
Hybrid taxonomy retriever.

For a contract section, returns the top-K candidate L3 leaves (plus their
parent L2 as a backoff) so the downstream LLM sees ~50 enriched candidates
instead of the full 140-leaf taxonomy.

Three signals, fused per query:
  - dense   : Voyage query embedding vs. the 202 node embeddings (cosine)
  - sparse  : in-repo Okapi BM25 over each node's enriched_text
  - keyword : bounded boost when a node's canonical_terms occur in the section

exclusion_cues are intentionally NOT a retrieval signal: as authored they are
routing guidance ("X is the Y leaf"), not contract phrases, so they would not
substring-match real section text. They are carried on each candidate as
metadata for the downstream prompt / reranker.

Pure logic (tokenizer, BM25, fusion, keyword, selection) imports and runs
without a DB or Voyage so it is unit-testable; DB + Voyage live behind
HybridRetriever.from_db().

CLI smoke test (needs DB + VOYAGE_API_KEY):
  caffeinate -i etl/.venv/bin/python \
      etl/src/etl/models/taxonomy/deepseek/retrieval.py --self-test
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from enrichment_lib import EMBED_MODEL, build_engine, load_envs

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RetrievalConfig:
    k: int = 50                      # number of L3 candidates to return
    dense_w: float = 0.55
    sparse_w: float = 0.30
    keyword_w: float = 0.15
    title_repeat: int = 3            # title token upweight for the sparse query
    max_body_words: int = 800        # body words fed to query (dense + sparse)
    bm25_k1: float = 1.5
    bm25_b: float = 0.75
    keyword_term_cap: int = 5        # max canonical-term hits counted per node
    include_parent_l2: bool = True   # attach each selected L3's parent L2
    voyage_query_batch: int = 128


# ---------------------------------------------------------------------------
# Text / tokenization
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"\[[^\]]*\]")          # [•], [*], [address], ...
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_SECTION_LEAD_RE = re.compile(r"(?i)^\s*(section|article)\s+[0-9ivxlcdm.]+\s*[.\-]?\s*")


def tokenize(text: str) -> list[str]:
    """Lowercase, drop bracketed placeholders, split to alphanumeric tokens.

    Placeholder stripping removes the pervasive '[•]' clause-template noise
    (the deferred review item) so it does not pollute BM25."""
    if not text:
        return []
    text = _PLACEHOLDER_RE.sub(" ", text.lower())
    return [t for t in _TOKEN_RE.findall(text) if len(t) > 1]


def _clean_title(title: str) -> str:
    if not title:
        return ""
    return _SECTION_LEAD_RE.sub("", title).strip()


def build_queries(
    article_title: str,
    section_title: str,
    section_text: str,
    cfg: RetrievalConfig,
) -> tuple[str, list[str], str]:
    """Return (dense_text, sparse_tokens, keyword_haystack) for one section."""
    art = _clean_title(article_title)
    sec = _clean_title(section_title)
    body = " ".join((section_text or "").split()[: cfg.max_body_words])

    dense_text = f"{art} > {sec}\n{body}".strip()

    title_tokens = tokenize(f"{sec} {art}")
    sparse_tokens = title_tokens * cfg.title_repeat + tokenize(body)

    keyword_haystack = f"{article_title} {section_title} {section_text}".lower()
    return dense_text, sparse_tokens, keyword_haystack


# ---------------------------------------------------------------------------
# Okapi BM25 (in-repo; no rank-bm25 dependency)
# ---------------------------------------------------------------------------


class BM25:
    """Standard Okapi BM25 over a fixed corpus of pre-tokenized documents."""

    def __init__(
        self, corpus_tokens: Sequence[Sequence[str]], k1: float, b: float
    ) -> None:
        self.k1 = k1
        self.b = b
        self.n_docs = len(corpus_tokens)
        self.doc_freqs: list[Counter[str]] = [Counter(d) for d in corpus_tokens]
        self.doc_len = np.array(
            [float(len(d)) for d in corpus_tokens], dtype=np.float64
        )
        self.avgdl = float(self.doc_len.mean()) if self.n_docs else 0.0
        df: Counter[str] = Counter()
        for d in self.doc_freqs:
            df.update(d.keys())
        self.idf: dict[str, float] = {
            term: math.log(1.0 + (self.n_docs - n + 0.5) / (n + 0.5))
            for term, n in df.items()
        }

    def scores(self, query_tokens: Sequence[str]) -> np.ndarray:
        out = np.zeros(self.n_docs, dtype=np.float64)
        if self.avgdl == 0.0:
            return out
        for term in set(query_tokens):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for i, freq in enumerate(self.doc_freqs):
                f = freq.get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (
                    1.0 - self.b + self.b * self.doc_len[i] / self.avgdl
                )
                out[i] += idf * (f * (self.k1 + 1.0)) / denom
        return out


# ---------------------------------------------------------------------------
# Fusion
# ---------------------------------------------------------------------------


def _minmax(x: np.ndarray) -> np.ndarray:
    lo = float(x.min())
    hi = float(x.max())
    if hi - lo < 1e-12:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def fuse(
    dense: np.ndarray,
    sparse: np.ndarray,
    keyword: np.ndarray,
    cfg: RetrievalConfig,
) -> np.ndarray:
    return (
        cfg.dense_w * _minmax(dense)
        + cfg.sparse_w * _minmax(sparse)
        + cfg.keyword_w * _minmax(keyword)
    )


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Node:
    standard_id: str
    level: str            # 'l2' | 'l3'
    label: str
    l1_label: str
    l2_label: str
    enriched_text: str
    canonical_terms: list[str]
    exclusion_cues: list[str]


@dataclass(frozen=True)
class Section:
    article_title: str
    section_title: str
    section_text: str
    section_uuid: str | None = None


@dataclass(frozen=True)
class Candidate:
    standard_id: str
    level: str
    label: str
    l1_label: str
    l2_label: str
    score: float
    dense: float
    sparse: float
    keyword: float
    is_backoff: bool
    exclusion_cues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Retriever
# ---------------------------------------------------------------------------

QueryEmbedder = Callable[[list[str]], np.ndarray]


class HybridRetriever:
    def __init__(
        self,
        nodes: list[Node],
        node_embeddings: np.ndarray,
        query_embedder: QueryEmbedder,
        cfg: RetrievalConfig | None = None,
    ) -> None:
        if len(nodes) != node_embeddings.shape[0]:
            raise ValueError("nodes and node_embeddings length mismatch")
        self.cfg = cfg or RetrievalConfig()
        self.nodes = nodes
        self.query_embedder = query_embedder
        self._emb = self._l2_normalize(node_embeddings.astype(np.float64))
        self._bm25 = BM25(
            [tokenize(n.enriched_text) for n in nodes],
            self.cfg.bm25_k1,
            self.cfg.bm25_b,
        )
        self._terms = [
            [t.lower() for t in n.canonical_terms if t.strip()] for n in nodes
        ]
        self._l3_idx = [i for i, n in enumerate(nodes) if n.level == "l3"]
        # (l1_label, l2_label) -> index of the L2 umbrella node
        self._l2_by_path: dict[tuple[str, str], int] = {
            (n.l1_label, n.l2_label): i
            for i, n in enumerate(nodes)
            if n.level == "l2"
        }

    @staticmethod
    def _l2_normalize(m: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(m, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return m / norms

    def _keyword_scores(self, haystack: str) -> np.ndarray:
        out = np.zeros(len(self.nodes), dtype=np.float64)
        cap = self.cfg.keyword_term_cap
        for i, terms in enumerate(self._terms):
            hits = 0
            for term in terms:
                if term and term in haystack:
                    hits += 1
                    if hits >= cap:
                        break
            out[i] = float(hits)
        return out

    def retrieve_batch(self, sections: Sequence[Section]) -> list[list[Candidate]]:
        if not sections:
            return []
        dense_texts: list[str] = []
        sparse_list: list[list[str]] = []
        haystacks: list[str] = []
        for s in sections:
            d, sp, hk = build_queries(
                s.article_title, s.section_title, s.section_text, self.cfg
            )
            dense_texts.append(d)
            sparse_list.append(sp)
            haystacks.append(hk)

        q = self._l2_normalize(
            np.asarray(self.query_embedder(dense_texts), dtype=np.float64)
        )
        if q.shape[0] != len(sections):
            raise RuntimeError("query embedder returned wrong row count")
        dense_all = q @ self._emb.T  # (n_sections, n_nodes)

        results: list[list[Candidate]] = []
        for si in range(len(sections)):
            dense = dense_all[si]
            sparse = self._bm25.scores(sparse_list[si])
            keyword = self._keyword_scores(haystacks[si])
            fused = fuse(dense, sparse, keyword, self.cfg)
            results.append(
                self._select(dense, sparse, keyword, fused)
            )
        return results

    def retrieve(self, section: Section) -> list[Candidate]:
        return self.retrieve_batch([section])[0]

    def _select(
        self,
        dense: np.ndarray,
        sparse: np.ndarray,
        keyword: np.ndarray,
        fused: np.ndarray,
    ) -> list[Candidate]:
        # Rank L3 only; deterministic tie-break by standard_id.
        ranked = sorted(
            self._l3_idx,
            key=lambda i: (-float(fused[i]), self.nodes[i].standard_id),
        )
        chosen = ranked[: self.cfg.k]

        def mk(i: int, backoff: bool) -> Candidate:
            n = self.nodes[i]
            return Candidate(
                standard_id=n.standard_id,
                level=n.level,
                label=n.label,
                l1_label=n.l1_label,
                l2_label=n.l2_label,
                score=float(fused[i]),
                dense=float(dense[i]),
                sparse=float(sparse[i]),
                keyword=float(keyword[i]),
                is_backoff=backoff,
                exclusion_cues=list(n.exclusion_cues),
            )

        out = [mk(i, False) for i in chosen]
        if self.cfg.include_parent_l2:
            seen = {self.nodes[i].standard_id for i in chosen}
            for i in chosen:
                n = self.nodes[i]
                l2i = self._l2_by_path.get((n.l1_label, n.l2_label))
                if l2i is not None:
                    sid = self.nodes[l2i].standard_id
                    if sid not in seen:
                        seen.add(sid)
                        out.append(mk(l2i, True))
        return out

    # ----- DB / Voyage wiring -------------------------------------------

    @classmethod
    def from_db(cls, cfg: RetrievalConfig | None = None) -> "HybridRetriever":
        nodes, emb = _load_sidecar()
        return cls(nodes, emb, _voyage_query_embedder(cfg or RetrievalConfig()), cfg)


def _load_sidecar() -> tuple[list[Node], np.ndarray]:
    from sqlalchemy import text

    engine, schema = build_engine()
    with engine.begin() as conn:
        rows = conn.execute(
            text(
                f"""
                SELECT m.standard_id, m.level, m.label, m.l1_label, m.l2_label,
                       m.canonical_terms, m.exclusion_cues, m.enriched_text,
                       VEC_ToText(e.embedding) AS emb
                FROM {schema}.taxonomy_node_meta m
                JOIN {schema}.taxonomy_node_embedding e
                  ON e.standard_id = m.standard_id
                ORDER BY m.standard_id
                """
            )
        ).mappings().all()
    if not rows:
        raise RuntimeError(
            "taxonomy sidecar is empty; run enrichment_load.py first"
        )
    nodes: list[Node] = []
    vecs: list[list[float]] = []
    for r in rows:
        nodes.append(
            Node(
                standard_id=str(r["standard_id"]),
                level=str(r["level"]),
                label=str(r["label"]),
                l1_label=str(r["l1_label"]),
                l2_label=str(r["l2_label"]),
                enriched_text=str(r["enriched_text"]),
                canonical_terms=json.loads(r["canonical_terms"]),
                exclusion_cues=json.loads(r["exclusion_cues"]),
            )
        )
        vecs.append(json.loads(r["emb"]))
    return nodes, np.asarray(vecs, dtype=np.float64)


def _voyage_query_embedder(cfg: RetrievalConfig) -> QueryEmbedder:
    import importlib
    import os

    load_envs()
    api_key = os.getenv("VOYAGE_API_KEY")
    if not api_key:
        raise ValueError("VOYAGE_API_KEY is required for query embedding.")
    client = importlib.import_module("voyageai").Client(api_key=api_key)

    def embed(texts: list[str]) -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), cfg.voyage_query_batch):
            batch = texts[i : i + cfg.voyage_query_batch]
            resp = client.embed(batch, model=EMBED_MODEL, input_type="query")
            out.extend(resp.embeddings)
        return np.asarray(out, dtype=np.float64)

    return embed


def _self_test() -> int:
    r = HybridRetriever.from_db()
    section = Section(
        article_title="Article VI Covenants",
        section_title="Regulatory Approvals; HSR",
        section_text=(
            "Each party shall make its filing under the HSR Act within ten "
            "Business Days and use reasonable best efforts to obtain "
            "expiration of the antitrust waiting period."
        ),
    )
    cands = r.retrieve(section)
    print(f"returned {len(cands)} candidates; top 8:")
    for c in cands[:8]:
        tag = " [L2 backoff]" if c.is_backoff else ""
        print(f"  {c.score:.3f}  {c.l1_label} > {c.l2_label} > {c.label}{tag}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true",
                        help="Load the sidecar and retrieve for one sample section.")
    args = parser.parse_args()
    if args.self_test:
        return _self_test()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
