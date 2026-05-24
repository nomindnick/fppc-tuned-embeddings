"""
Local-model variant of search-lab's CitationScoreFusion (Experiment 009).

The 009 engine uses OpenAI text-embedding-3-small at query time and a
precomputed OpenAI pickle for the corpus. This wrapper swaps both sides for
a local SentenceTransformer + the .npy + .json caches already produced by
Sprint 3's `_eval_lib.encode_corpus()`. Everything else (BM25, citation
index, min-max score fusion, circuit-breaker, query routing) is identical
to the production engine.

Tunable knobs are exposed via constructor args:
- cb_threshold (default 1.3): BM25 top1/top2 ratio that triggers BM25-only.
- w_bm25, w_sem (defaults 0.4 / 0.6): normalized-score fusion weights.
- semantic_only_path (default False): if True, route every query through
  the fusion path (i.e. semantic also affects non-citation queries).

Indexes live in `fppc-opinions-search-lab/indexes/` (BM25 + citation,
shared) and `fppc-tuned-embeddings/data/indexes/` (semantic, per model).
"""

from __future__ import annotations

import os
import pickle
import re
import sys
import json
from typing import Any

import numpy as np

SEARCH_LAB_ROOT = "/home/nick/Projects/fppc-opinions-search-lab"
TUNED_ROOT = "/home/nick/Projects/fppc-tuned-embeddings"


# ---------------------------------------------------------------------------
# Tokenizer + citation parser
#
# Copied from fppc-opinions-search-lab/src/engines/{bm25_full_text.py,
# bm25_citation_boost.py}. The original repo's modules can't be imported here
# without colliding with fppc-opinions-eval's `src.scorer` (both repos expose a
# top-level `src` package). These two functions are pure-stdlib + regex with
# no external state, so duplication is safer than module-system contortions.
# Keep in sync if the search-lab regex set changes.
# ---------------------------------------------------------------------------

_STOPWORDS = frozenset([
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but", "by",
    "can", "could", "did", "do", "does", "doing", "done", "down", "during",
    "each", "few", "for", "from", "further", "get", "got", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him",
    "himself", "his", "how", "i", "if", "in", "into", "is", "it", "its",
    "itself", "just", "let", "may", "me", "might", "more", "most", "much",
    "must", "my", "myself", "nor", "of", "off", "on", "once", "only", "or",
    "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shall", "she", "should", "so", "some", "such", "than", "that",
    "the", "their", "theirs", "them", "themselves", "then", "there", "these",
    "they", "this", "those", "through", "to", "too", "under", "until", "up",
    "upon", "us", "very", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "whom", "why", "will", "with", "would", "yet",
    "you", "your", "yours", "yourself", "yourselves",
    "about", "above", "after", "again", "against", "all", "am", "any",
    "because", "before", "below", "between", "both", "also",
])

_PAREN_SUB = re.compile(r"(\d+)\(([a-zA-Z0-9])\)")
_NON_ALNUM = re.compile(r"[^a-z0-9\-]+")


def tokenize(text: str) -> list[str]:
    """BM25 tokenizer (matches search-lab bm25_full_text.tokenize)."""
    text = text.lower()
    text = _PAREN_SUB.sub(r"\1\2", text)
    text = _NON_ALNUM.sub(" ", text)
    tokens = text.split()
    return [t for t in tokens if t not in _STOPWORDS]


_RE_PREFIXED_STATUTE = re.compile(
    r"(?:Section|Gov(?:ernment)?\.?\s*Code)\s+(\d{3,5})(\([a-zA-Z0-9]\))?",
    re.IGNORECASE,
)
_RE_PREFIXED_REG = re.compile(
    r"(?:Reg(?:ulation)?\.?)\s+(\d{4,5}(?:\.\d+)?)", re.IGNORECASE,
)
_RE_BARE_STATUTE = re.compile(
    r"\b(8[1-9]\d{3}|90\d{3}|91014|109[0-7])(?:\(([a-zA-Z0-9])\))?\b"
)
_RE_BARE_REG = re.compile(r"\b(18\d{3}(?:\.\d+)?)\b")


def parse_query_citations(query: str) -> dict:
    """Statute/regulation extraction (matches search-lab parse_query_citations)."""
    gov_code = []
    regulations = []
    seen_gc: set[str] = set()
    seen_reg: set[str] = set()

    for m in _RE_PREFIXED_STATUTE.finditer(query):
        base = m.group(1)
        sub = m.group(2) or ""
        raw = base + sub
        if raw not in seen_gc:
            seen_gc.add(raw)
            gov_code.append({"raw": raw, "base": base, "subsection": sub})

    for m in _RE_PREFIXED_REG.finditer(query):
        full = m.group(1)
        base = full.split(".")[0]
        sub = "." + full.split(".", 1)[1] if "." in full else ""
        if full not in seen_reg:
            seen_reg.add(full)
            regulations.append({"raw": full, "base": base, "subsection": sub})

    for m in _RE_BARE_STATUTE.finditer(query):
        base = m.group(1)
        sub_letter = m.group(2) or ""
        sub = f"({sub_letter})" if sub_letter else ""
        raw = base + sub
        if raw not in seen_gc:
            seen_gc.add(raw)
            gov_code.append({"raw": raw, "base": base, "subsection": sub})

    for m in _RE_BARE_REG.finditer(query):
        full = m.group(1)
        base = full.split(".")[0]
        sub = "." + full.split(".", 1)[1] if "." in full else ""
        if full not in seen_reg:
            seen_reg.add(full)
            regulations.append({"raw": full, "base": base, "subsection": sub})

    return {"gov_code": gov_code, "regulations": regulations}

BM25_INDEX = os.path.join(SEARCH_LAB_ROOT, "indexes", "BM25FullText_index.pkl")
CITATION_INDEX = os.path.join(
    SEARCH_LAB_ROOT, "indexes", "BM25CitationBoost_citation_index.pkl"
)

_BM25_POOL = 100  # BM25 top-N to union into the candidate pool


def _min_max(pool: dict[str, float]) -> dict[str, float]:
    if not pool:
        return pool
    vals = list(pool.values())
    lo, hi = min(vals), max(vals)
    rng = hi - lo
    if rng == 0:
        return {k: 1.0 for k in pool}
    return {k: (v - lo) / rng for k, v in pool.items()}


class LocalCitationScoreFusion:
    """009-equivalent fusion using a local SentenceTransformer."""

    def __init__(
        self,
        model_dir: str,
        sem_npy: str,
        sem_ids_json: str,
        query_prefix: str = "query: ",
        cb_threshold: float = 1.3,
        w_bm25: float = 0.4,
        w_sem: float = 0.6,
        semantic_only_path: bool = False,
        max_seq_length: int | None = 512,
        w_sem_by_type: dict[str, float] | None = None,
        adaptive_confidence_rule: Any | None = None,
        # --- Variant B: simple confidence-gated semantic boost ---
        # If `confidence_top1_threshold` is set and the top-1 semantic
        # cosine within the candidate pool meets/exceeds it, override the
        # default `w_sem` with `confidence_boost_w_sem` for this query.
        # Optional secondary gate: require `(top1 - mean) >= gap_threshold`.
        confidence_top1_threshold: float | None = None,
        confidence_gap_threshold: float | None = None,
        confidence_boost_w_sem: float = 0.7,
    ):
        """
        Args:
          w_sem_by_type: if provided, maps `query_type` (keyword /
            natural_language / fact_pattern) to a `w_sem`. When the caller
            supplies `query_type` to `.search()`, that type's weight is
            used in place of the default `w_sem`. `w_bm25` is computed as
            `1 - w_sem` per query.
          adaptive_confidence_rule: optional callable
            `fn(top1_sem, mean_pool_sem, default_w_sem) -> w_sem` that
            adjusts `w_sem` per query based on d3's semantic confidence
            in the candidate pool. Returns the effective `w_sem`; `w_bm25`
            is `1 - w_sem`.
        """
        self._cb_threshold = cb_threshold
        self._w_bm25 = w_bm25
        self._w_sem = w_sem
        self._query_prefix = query_prefix
        self._semantic_only_path = semantic_only_path
        self._w_sem_by_type = w_sem_by_type or {}
        self._adaptive_confidence_rule = adaptive_confidence_rule
        self._confidence_top1_threshold = confidence_top1_threshold
        self._confidence_gap_threshold = confidence_gap_threshold
        self._confidence_boost_w_sem = confidence_boost_w_sem

        # Lazy import so non-engine consumers don't pay the SBERT cost
        from sentence_transformers import SentenceTransformer

        print(f"Loading model from {model_dir}…", file=sys.stderr)
        self._model = SentenceTransformer(model_dir)
        if max_seq_length is not None:
            self._model.max_seq_length = max_seq_length

        print(f"Loading BM25 index from {BM25_INDEX}…", file=sys.stderr)
        with open(BM25_INDEX, "rb") as f:
            bm = pickle.load(f)
        self._bm25_ids = bm["opinion_ids"]
        self._bm25 = bm["bm25"]
        self._bm25_id_to_idx = {oid: i for i, oid in enumerate(self._bm25_ids)}

        print(f"Loading semantic index from {sem_npy}…", file=sys.stderr)
        self._sem_vecs = np.load(sem_npy).astype(np.float32)
        with open(sem_ids_json) as f:
            self._sem_ids = json.load(f)
        self._sem_id_to_idx = {oid: i for i, oid in enumerate(self._sem_ids)}
        # Sanity: assume L2-normalized so dot product = cosine.
        n = np.linalg.norm(self._sem_vecs[0])
        if not (0.99 < n < 1.01):
            raise RuntimeError(
                f"Semantic vectors not L2-normalized (||v[0]||={n:.4f}). "
                f"Re-encode with normalize_embeddings=True."
            )

        print(f"Loading citation index from {CITATION_INDEX}…", file=sys.stderr)
        with open(CITATION_INDEX, "rb") as f:
            self._cite_index = pickle.load(f)

    def _embed_query(self, query: str) -> np.ndarray:
        text = self._query_prefix + query
        v = self._model.encode(
            [text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0].astype(np.float32)
        return v

    def _bm25_only(self, bm25_scores: np.ndarray, top_k: int) -> list[str]:
        top_indices = bm25_scores.argsort()[::-1][:top_k]
        return [self._bm25_ids[i] for i in top_indices if bm25_scores[i] > 0]

    def search(
        self,
        query: str,
        top_k: int = 20,
        query_type: str | None = None,
    ) -> tuple[list[str], dict[str, Any]]:
        """Return (top_k opinion_ids, diagnostics).

        Diagnostics record which path the query took (`bm25_only` /
        `cb_fired` / `fusion`), so we can attribute lifts in the rollup.

        `query_type` is optional metadata used by per-type adaptive
        weights (`w_sem_by_type`). When supplied and the type has an
        entry, that type's weight overrides the default `w_sem` for
        this query. Logged into diagnostics.
        """
        diag: dict[str, Any] = {}

        tokens = tokenize(query)
        if not tokens:
            diag["path"] = "empty_query"
            return [], diag
        bm25_scores = self._bm25.get_scores(tokens)

        parsed = parse_query_citations(query)
        has_citations = bool(parsed["gov_code"] or parsed["regulations"])
        diag["has_citations"] = has_citations

        if not has_citations and not self._semantic_only_path:
            diag["path"] = "bm25_only_noncite"
            return self._bm25_only(bm25_scores, top_k), diag

        # Build candidate pool. With semantic_only_path=True and no citations,
        # we have no statute-pool — pool = BM25 top-N.
        pool: set[str] = set()
        if has_citations:
            gc_exact = self._cite_index["gc_exact"]
            gc_base = self._cite_index["gc_base"]
            reg_exact = self._cite_index["reg_exact"]
            for cite in parsed["gov_code"]:
                pool |= gc_exact.get(cite["raw"], set())
                pool |= gc_base.get(cite["base"], set())
            for cite in parsed["regulations"]:
                pool |= reg_exact.get(cite["raw"], set())
                if cite["subsection"]:
                    pool |= reg_exact.get(cite["base"], set())

        bm25_top = {
            self._bm25_ids[i]
            for i in bm25_scores.argsort()[::-1][:_BM25_POOL]
            if bm25_scores[i] > 0
        }
        candidate_pool = pool | bm25_top
        diag["pool_size"] = len(candidate_pool)
        diag["citation_pool_size"] = len(pool)

        if not candidate_pool:
            diag["path"] = "empty_pool"
            return [], diag

        bm25_pool = {}
        for oid in candidate_pool:
            idx = self._bm25_id_to_idx.get(oid)
            bm25_pool[oid] = float(bm25_scores[idx]) if idx is not None else 0.0

        sorted_scores = sorted(bm25_pool.values(), reverse=True)
        if len(sorted_scores) <= 1:
            ratio = float("inf")
        else:
            ratio = (sorted_scores[0] / sorted_scores[1]
                     if sorted_scores[1] > 0 else float("inf"))
        diag["bm25_top12_ratio"] = ratio if ratio != float("inf") else None

        if ratio >= self._cb_threshold:
            diag["path"] = "cb_fired"
            return sorted(bm25_pool, key=bm25_pool.get, reverse=True)[:top_k], diag

        q_vec = self._embed_query(query)
        cos_all = self._sem_vecs @ q_vec

        sem_pool = {}
        for oid in candidate_pool:
            idx = self._sem_id_to_idx.get(oid)
            sem_pool[oid] = float(cos_all[idx]) if idx is not None else 0.0

        norm_bm25 = _min_max(bm25_pool)
        norm_sem = _min_max(sem_pool)

        # Effective fusion weights — per-type override and/or adaptive
        # confidence rule. Otherwise default to engine-level w_sem/w_bm25.
        eff_w_sem = self._w_sem
        if query_type is not None and query_type in self._w_sem_by_type:
            eff_w_sem = self._w_sem_by_type[query_type]
            diag["w_sem_source"] = f"by_type[{query_type}]"
        if self._adaptive_confidence_rule is not None and sem_pool:
            sem_vals = list(sem_pool.values())
            top1 = max(sem_vals)
            mean_pool = sum(sem_vals) / len(sem_vals)
            eff_w_sem = self._adaptive_confidence_rule(top1, mean_pool, eff_w_sem)
            diag["sem_top1"] = top1
            diag["sem_mean"] = mean_pool
        # Variant B — simple thresholded confidence boost
        if self._confidence_top1_threshold is not None and sem_pool:
            sem_vals = list(sem_pool.values())
            top1 = max(sem_vals)
            mean_pool = sum(sem_vals) / len(sem_vals)
            diag["sem_top1"] = top1
            diag["sem_mean"] = mean_pool
            diag["sem_gap"] = top1 - mean_pool
            gate_top1 = top1 >= self._confidence_top1_threshold
            gate_gap = (
                self._confidence_gap_threshold is None
                or (top1 - mean_pool) >= self._confidence_gap_threshold
            )
            if gate_top1 and gate_gap:
                eff_w_sem = self._confidence_boost_w_sem
                diag["w_sem_source"] = "confidence_boost"
        eff_w_bm25 = 1.0 - eff_w_sem
        diag["eff_w_sem"] = eff_w_sem
        diag["eff_w_bm25"] = eff_w_bm25

        combined = {
            oid: eff_w_bm25 * norm_bm25.get(oid, 0.0)
            + eff_w_sem * norm_sem.get(oid, 0.0)
            for oid in candidate_pool
        }
        diag["path"] = "fusion"
        return sorted(combined, key=combined.get, reverse=True)[:top_k], diag

    def name(self) -> str:
        return "LocalCitationScoreFusion"
