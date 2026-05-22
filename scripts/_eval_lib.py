"""
Shared encode + retrieve + score logic for the 65-query graded eval.

Used by both:
- scripts/check2_open_baselines.py (off-the-shelf baselines)
- scripts/score_finetuned.py        (Sprint 3+ fine-tuned models)

Goal: bit-identical numbers between baseline and tuned-model scoring, so the
"tuned > baseline" comparison is apples-to-apples.

Caching: corpus embeddings can be cached to disk under INDEX_CACHE_DIR keyed
by a caller-provided cache_key. Pass cache_key=None to skip caching (the
default for one-shot fine-tune scoring).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

# Reuse the eval harness scoring logic to get apples-to-apples numbers.
sys.path.insert(0, "/home/nick/Projects/fppc-opinions-eval")
from src.scorer import (  # type: ignore[import-not-found]  # noqa: E402
    aggregate_metrics,
    compute_mrr,
    compute_ndcg,
    compute_precision,
    compute_recall,
)

EVAL_PATH = "/home/nick/Projects/fppc-opinions-eval/eval/dataset.json"
CORPUS_DIR = "/home/nick/Projects/fppc-opinions-corpus/data/extracted"
INDEX_CACHE_DIR = "/home/nick/Projects/fppc-tuned-embeddings/data/indexes"


# ---------------------------------------------------------------------------
# Corpus + eval loading
# ---------------------------------------------------------------------------

def load_corpus_texts() -> tuple[list[str], list[str]]:
    """Return (opinion_ids, qa_texts) walked in the same order every run.

    Uses `embedding.qa_text`, falling back to `content.full_text` when qa_text
    is missing or trivially short — same convention as the search-lab semantic
    baseline.
    """
    ids: list[str] = []
    texts: list[str] = []
    for year_dir in sorted(os.listdir(CORPUS_DIR)):
        year_path = os.path.join(CORPUS_DIR, year_dir)
        if not os.path.isdir(year_path):
            continue
        for filename in sorted(os.listdir(year_path)):
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(year_path, filename)) as f:
                op = json.load(f)
            oid = op.get("id", filename.replace(".json", ""))
            qa = (op.get("embedding") or {}).get("qa_text") or ""
            if len(qa.strip()) < 20:
                qa = (op.get("content") or {}).get("full_text") or ""
            ids.append(oid)
            texts.append(qa.strip() or " ")
    return ids, texts


def load_eval() -> list[dict]:
    with open(EVAL_PATH) as f:
        return json.load(f)["queries"]


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_corpus(
    model: "SentenceTransformer",
    ids: list[str],
    texts: list[str],
    doc_prefix: str,
    cache_key: str | None = None,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode corpus docs with L2 normalization. Optionally cache to disk.

    If `cache_key` is provided and a cached vector file exists whose
    accompanying ID list matches `ids` exactly, the cached vectors are
    returned without re-encoding. Otherwise the model encodes from scratch
    and (if cache_key is set) writes a new cache.
    """
    if cache_key is not None:
        os.makedirs(INDEX_CACHE_DIR, exist_ok=True)
        cache_path = os.path.join(
            INDEX_CACHE_DIR, f"corpus_embeddings_{cache_key}_qa_text.npy"
        )
        ids_path = os.path.join(
            INDEX_CACHE_DIR, f"corpus_embeddings_{cache_key}_ids.json"
        )
        if os.path.exists(cache_path) and os.path.exists(ids_path):
            cached_ids = json.load(open(ids_path))
            if cached_ids == ids:
                print(f"  [cache] Loaded {cache_key} corpus embeddings from disk")
                return np.load(cache_path)

    inputs = [doc_prefix + t for t in texts]
    print(
        f"  Embedding {len(inputs)} docs "
        f"(max_seq_length={model.max_seq_length}, batch_size={batch_size})…"
    )
    t0 = time.time()
    vecs = model.encode(
        inputs,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"  Done in {time.time() - t0:.1f}s. Shape: {vecs.shape}")
    vecs = vecs.astype(np.float32)

    if cache_key is not None:
        np.save(cache_path, vecs)
        with open(ids_path, "w") as f:
            json.dump(ids, f)

    return vecs


def encode_queries(
    model: "SentenceTransformer",
    query_texts: list[str],
    query_prefix: str,
    batch_size: int = 32,
) -> np.ndarray:
    """Encode queries with L2 normalization."""
    inputs = [query_prefix + q for q in query_texts]
    vecs = model.encode(
        inputs,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# Retrieval + scoring
# ---------------------------------------------------------------------------

def retrieve_top_k(
    query_vecs: np.ndarray,
    doc_vecs: np.ndarray,
    ids: list[str],
    k: int = 20,
) -> list[list[str]]:
    """For each query row, return the top-k opinion IDs by cosine similarity.

    Assumes inputs are L2-normalized — dot product equals cosine.
    """
    sim = query_vecs @ doc_vecs.T  # (n_queries, n_docs)
    top_k_idx = np.argpartition(-sim, kth=k, axis=1)[:, :k]
    sorted_top: list[list[str]] = []
    for row_i, idxs in enumerate(top_k_idx):
        scores_row = sim[row_i, idxs]
        order = np.argsort(-scores_row)
        sorted_top.append([ids[idxs[j]] for j in order])
    return sorted_top


def evaluate_query(query: dict, results: list[str]) -> dict:
    judgments = {j["opinion_id"]: j["score"] for j in query["relevance_judgments"]}
    metrics = {
        "mrr": compute_mrr(results, judgments),
        "ndcg@5": compute_ndcg(results, judgments, 5),
        "ndcg@10": compute_ndcg(results, judgments, 10),
        "precision@5": compute_precision(results, judgments, 5),
        "precision@10": compute_precision(results, judgments, 10),
        "recall@10": compute_recall(results, judgments, 10),
        "recall@20": compute_recall(results, judgments, 20),
    }
    return {
        "query_id": query["id"],
        "query_text": query["text"],
        "query_type": query.get("type", "unknown"),
        "query_topic": query.get("topic", "unknown"),
        "num_results": len(results),
        "results": results,
        "metrics": metrics,
    }


def aggregate_by(per_query: list[dict], key: str) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = {}
    for qr in per_query:
        buckets.setdefault(qr[key], []).append(qr)
    return {k: aggregate_metrics(v) for k, v in buckets.items()}


# ---------------------------------------------------------------------------
# Top-level: run full eval for a model
# ---------------------------------------------------------------------------

def score_model_on_eval(
    model: "SentenceTransformer",
    engine_name: str,
    hf_id: str,
    query_prefix: str,
    doc_prefix: str,
    max_seq_length: int | None,
    notes: str = "",
    cache_key: str | None = None,
    batch_size: int = 32,
    top_k: int = 20,
    extra: dict | None = None,
) -> dict:
    """Run the full encode + retrieve + score pipeline.

    Returns a dict in the same shape as results/check2_*.json. Pass `extra`
    to attach Sprint 3+ run metadata (run_id, loss, positive_column, etc.).
    """
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length

    print(f"\n=== {engine_name} ===")
    ids, texts = load_corpus_texts()
    print(f"Corpus: {len(ids)} opinions")

    queries = load_eval()
    print(f"Eval queries: {len(queries)}")

    doc_vecs = encode_corpus(
        model, ids, texts,
        doc_prefix=doc_prefix,
        cache_key=cache_key,
        batch_size=batch_size,
    )
    query_vecs = encode_queries(
        model, [q["text"] for q in queries],
        query_prefix=query_prefix,
        batch_size=batch_size,
    )

    print(f"Retrieving top-{top_k} for each query…")
    sorted_top = retrieve_top_k(query_vecs, doc_vecs, ids, k=top_k)

    per_query = [evaluate_query(q, sorted_top[i]) for i, q in enumerate(queries)]

    result = {
        "engine_name": engine_name,
        "hf_id": hf_id,
        "notes": notes,
        "max_seq_length": max_seq_length,
        "query_prefix": query_prefix,
        "doc_prefix": doc_prefix,
        "n_queries": len(queries),
        "overall": aggregate_metrics(per_query),
        "by_type": aggregate_by(per_query, "query_type"),
        "by_topic": aggregate_by(per_query, "query_topic"),
        "per_query": per_query,
    }
    if extra:
        result.update(extra)
    return result
