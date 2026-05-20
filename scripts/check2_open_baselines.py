"""
Check 2: Run open-weights embedding models against the existing eval harness
and compare to the OpenAI text-embedding-3-small baseline (0.262 nDCG@5,
0.485 MRR).

For each model:
  1. Embed every opinion's `embedding.qa_text` (same field OpenAI baseline used)
  2. Embed the 65 eval queries (with the model's recommended query prefix)
  3. Retrieve top-20 opinion IDs per query by cosine similarity
  4. Score using the eval harness's metric functions
  5. Persist results JSON and a row in the comparison table

Designed to run on CPU; uses sentence-transformers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import time
from dataclasses import dataclass

import numpy as np

# Reuse the eval harness scoring logic to get apples-to-apples numbers
sys.path.insert(0, "/home/nick/Projects/fppc-opinions-eval")
from src.scorer import (  # type: ignore[import-not-found]
    aggregate_metrics,
    compute_mrr,
    compute_ndcg,
    compute_precision,
    compute_recall,
)

EVAL_PATH = "/home/nick/Projects/fppc-opinions-eval/eval/dataset.json"
CORPUS_DIR = "/home/nick/Projects/fppc-opinions-corpus/data/extracted"
RESULTS_DIR = "/home/nick/Projects/fppc-tuned-embeddings/results"
INDEX_CACHE_DIR = "/home/nick/Projects/fppc-tuned-embeddings/data/indexes"


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    name: str           # short id used for filenames
    hf_id: str          # huggingface model id
    query_prefix: str   # prepended to queries before embedding
    doc_prefix: str     # prepended to documents before embedding
    max_seq_length: int | None  # override default if set
    trust_remote_code: bool
    notes: str


MODELS: dict[str, ModelConfig] = {
    "bge-base": ModelConfig(
        name="bge-base-en-v1.5",
        hf_id="BAAI/bge-base-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
        doc_prefix="",
        max_seq_length=512,
        trust_remote_code=False,
        notes="BGE recommended prefix on queries only.",
    ),
    "bge-large": ModelConfig(
        name="bge-large-en-v1.5",
        hf_id="BAAI/bge-large-en-v1.5",
        query_prefix="Represent this sentence for searching relevant passages: ",
        doc_prefix="",
        max_seq_length=512,
        trust_remote_code=False,
        notes="Larger BGE; same prefix convention.",
    ),
    "nomic-v1.5": ModelConfig(
        name="nomic-embed-text-v1.5",
        hf_id="nomic-ai/nomic-embed-text-v1.5",
        query_prefix="search_query: ",
        doc_prefix="search_document: ",
        # Native cap is 8192 but qa_text p90 ≈ 649 tokens, p99 ≈ 2296.
        # 1024 covers ~98% of docs without truncation and keeps CPU runtime
        # tractable. The 2-3% of longer docs get truncated; we accept that
        # for the baseline pass.
        max_seq_length=1024,
        trust_remote_code=True,
        notes="Matryoshka; max_seq capped at 1024 for CPU runtime (covers ~98% of docs).",
    ),
}


# ---------------------------------------------------------------------------
# Corpus + query loading
# ---------------------------------------------------------------------------

def load_corpus_texts() -> tuple[list[str], list[str]]:
    """Return (opinion_ids, qa_texts) walked in the same order every run."""
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
                # Same fallback the search-lab semantic baseline uses
                qa = (op.get("content") or {}).get("full_text") or ""
            ids.append(oid)
            texts.append(qa.strip() or " ")
    return ids, texts


def load_eval() -> list[dict]:
    with open(EVAL_PATH) as f:
        return json.load(f)["queries"]


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed_corpus(cfg: ModelConfig, ids: list[str], texts: list[str]) -> np.ndarray:
    """Embed all corpus docs with a single load of the model. L2-normalized."""
    from sentence_transformers import SentenceTransformer

    cache_path = os.path.join(
        INDEX_CACHE_DIR, f"corpus_embeddings_{cfg.name}_qa_text.npy"
    )
    ids_path = os.path.join(
        INDEX_CACHE_DIR, f"corpus_embeddings_{cfg.name}_ids.json"
    )
    os.makedirs(INDEX_CACHE_DIR, exist_ok=True)

    if os.path.exists(cache_path) and os.path.exists(ids_path):
        cached_ids = json.load(open(ids_path))
        if cached_ids == ids:
            print(f"  [cache] Loaded {cfg.name} corpus embeddings from disk")
            return np.load(cache_path)

    print(f"  Loading model: {cfg.hf_id}")
    model = SentenceTransformer(cfg.hf_id, trust_remote_code=cfg.trust_remote_code)
    if cfg.max_seq_length is not None:
        model.max_seq_length = cfg.max_seq_length

    inputs = [cfg.doc_prefix + t for t in texts]
    print(f"  Embedding {len(inputs)} docs (max_seq_length={model.max_seq_length})…")
    t0 = time.time()
    vecs = model.encode(
        inputs,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    print(f"  Done in {time.time() - t0:.1f}s. Shape: {vecs.shape}")

    np.save(cache_path, vecs.astype(np.float32))
    with open(ids_path, "w") as f:
        json.dump(ids, f)
    return vecs.astype(np.float32)


def embed_queries(cfg: ModelConfig, query_texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(cfg.hf_id, trust_remote_code=cfg.trust_remote_code)
    if cfg.max_seq_length is not None:
        model.max_seq_length = cfg.max_seq_length
    inputs = [cfg.query_prefix + q for q in query_texts]
    vecs = model.encode(
        inputs,
        batch_size=32,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


# ---------------------------------------------------------------------------
# Retrieval + scoring
# ---------------------------------------------------------------------------

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


def run_for_model(cfg: ModelConfig) -> dict:
    print(f"\n=== {cfg.name} ===")
    ids, texts = load_corpus_texts()
    print(f"Corpus: {len(ids)} opinions")

    queries = load_eval()
    print(f"Eval queries: {len(queries)}")

    doc_vecs = embed_corpus(cfg, ids, texts)
    query_vecs = embed_queries(cfg, [q["text"] for q in queries])

    # Retrieval — cosine via dot product on L2-normalized vectors
    # Score matrix (Q x D) — small enough at 65 x 14k to materialize
    print("Retrieving top-20 for each query…")
    sim = query_vecs @ doc_vecs.T  # (n_queries, n_docs)
    top_k_idx = np.argpartition(-sim, kth=20, axis=1)[:, :20]
    # Sort the top-20 indices by score descending per row
    sorted_top = []
    for row_i, idxs in enumerate(top_k_idx):
        scores_row = sim[row_i, idxs]
        order = np.argsort(-scores_row)
        sorted_top.append([ids[idxs[j]] for j in order])

    per_query = [evaluate_query(q, sorted_top[i]) for i, q in enumerate(queries)]

    overall = aggregate_metrics(per_query)
    by_type = _agg_by(per_query, "query_type")
    by_topic = _agg_by(per_query, "query_topic")

    return {
        "engine_name": cfg.name,
        "hf_id": cfg.hf_id,
        "notes": cfg.notes,
        "max_seq_length": cfg.max_seq_length,
        "query_prefix": cfg.query_prefix,
        "doc_prefix": cfg.doc_prefix,
        "n_queries": len(queries),
        "overall": overall,
        "by_type": by_type,
        "by_topic": by_topic,
        "per_query": per_query,
    }


def _agg_by(per_query: list[dict], key: str) -> dict[str, dict]:
    buckets: dict[str, list[dict]] = {}
    for qr in per_query:
        buckets.setdefault(qr[key], []).append(qr)
    return {k: aggregate_metrics(v) for k, v in buckets.items()}


# ---------------------------------------------------------------------------
# OpenAI baseline (reuse precomputed index from search-lab; only query
# embeddings are computed live, which still needs an API key).
# ---------------------------------------------------------------------------

def evaluate_openai_baseline_from_cached() -> dict | None:
    """Re-evaluate the OpenAI baseline using the precomputed search-lab index.

    Only embeds 65 query strings via the API call (cheap), then runs the same
    cosine retrieval + scoring path so the comparison is methodologically
    identical to the open-model runs.
    """
    index_path = (
        "/home/nick/Projects/fppc-opinions-search-lab/indexes/"
        "embeddings_text-embedding-3-small_qa_text.pkl"
    )
    env_path = "/home/nick/Projects/fppc-opinions-search-lab/.env"
    if not os.path.exists(index_path):
        print("[openai] no precomputed index, skipping")
        return None

    api_key = None
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if line.startswith("OPENAI_API_KEY="):
                    api_key = line.strip().split("=", 1)[1].strip("'\"")
                    break
    api_key = api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("[openai] no API key found, skipping live baseline replay")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        print("[openai] openai package not installed; skipping")
        return None

    print("\n=== OpenAI text-embedding-3-small (baseline replay) ===")
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    ids = data["opinion_ids"]
    doc_vecs = data["embeddings"]  # already L2-normalized per search-lab code

    queries = load_eval()
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[q["text"] for q in queries],
    )
    q_vecs = np.array([r.embedding for r in resp.data], dtype=np.float32)
    # Normalize
    norms = np.linalg.norm(q_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    q_vecs /= norms

    sim = q_vecs @ doc_vecs.T
    top_k_idx = np.argpartition(-sim, kth=20, axis=1)[:, :20]
    sorted_top = []
    for row_i, idxs in enumerate(top_k_idx):
        scores_row = sim[row_i, idxs]
        order = np.argsort(-scores_row)
        sorted_top.append([ids[idxs[j]] for j in order])

    per_query = [evaluate_query(q, sorted_top[i]) for i, q in enumerate(queries)]
    return {
        "engine_name": "text-embedding-3-small",
        "hf_id": "openai/text-embedding-3-small",
        "notes": "Baseline replay using precomputed search-lab index.",
        "max_seq_length": 8191,
        "query_prefix": "",
        "doc_prefix": "",
        "n_queries": len(queries),
        "overall": aggregate_metrics(per_query),
        "by_type": _agg_by(per_query, "query_type"),
        "by_topic": _agg_by(per_query, "query_topic"),
        "per_query": per_query,
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_compare(rows: list[dict]) -> None:
    keys = ["mrr", "ndcg@5", "ndcg@10", "precision@5", "precision@10", "recall@10", "recall@20"]
    header = ["model"] + keys
    widths = [28] + [9] * len(keys)
    print()
    print(" | ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        line = [r["engine_name"]] + [f"{r['overall'].get(k, 0.0):.3f}" for k in keys]
        print(" | ".join(c.ljust(w) for c, w in zip(line, widths)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(MODELS.keys()),
        help="Subset of model keys to run",
    )
    parser.add_argument(
        "--skip-openai", action="store_true", help="Skip OpenAI baseline replay"
    )
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for key in args.models:
        if key not in MODELS:
            print(f"Unknown model key: {key}")
            continue
        cfg = MODELS[key]
        result = run_for_model(cfg)
        out_path = os.path.join(RESULTS_DIR, f"check2_{cfg.name}.json")
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Wrote {out_path}")
        rows.append(result)

    if not args.skip_openai:
        openai_result = evaluate_openai_baseline_from_cached()
        if openai_result is not None:
            out_path = os.path.join(RESULTS_DIR, "check2_openai_baseline.json")
            with open(out_path, "w") as f:
                json.dump(openai_result, f, indent=2)
            print(f"Wrote {out_path}")
            rows.append(openai_result)

    print_compare(rows)

    summary = {
        "comparison": [
            {"engine_name": r["engine_name"], "overall": r["overall"]} for r in rows
        ]
    }
    with open(os.path.join(RESULTS_DIR, "check2_comparison.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
