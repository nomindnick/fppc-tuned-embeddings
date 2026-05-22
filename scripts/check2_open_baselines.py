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

Core encode/retrieve/score logic lives in `scripts/_eval_lib.py` so Sprint 3
fine-tune scoring uses the exact same code path.

Designed to run on CPU; uses sentence-transformers.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from dataclasses import dataclass

import numpy as np

import _eval_lib as evl

RESULTS_DIR = "/home/nick/Projects/fppc-tuned-embeddings/results"


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
        max_seq_length=1024,
        trust_remote_code=True,
        notes="Matryoshka; max_seq capped at 1024 for CPU runtime (covers ~98% of docs).",
    ),
    "gte-modernbert-base": ModelConfig(
        name="gte-modernbert-base",
        hf_id="Alibaba-NLP/gte-modernbert-base",
        query_prefix="",
        doc_prefix="",
        max_seq_length=1024,
        trust_remote_code=False,
        notes="ModernBERT encoder, 8192 native context, no prompt prefix.",
    ),
    "qwen3-embed-0.6b": ModelConfig(
        name="qwen3-embed-0.6b",
        hf_id="Qwen/Qwen3-Embedding-0.6B",
        query_prefix=(
            "Instruct: Given a legal question, retrieve the FPPC advisory "
            "opinion that addresses it.\nQuery: "
        ),
        doc_prefix="",
        max_seq_length=1024,
        trust_remote_code=False,
        notes="Decoder-as-encoder with last-token pooling; instruction-aware.",
    ),
    "snowflake-arctic-l-v2": ModelConfig(
        name="snowflake-arctic-embed-l-v2.0",
        hf_id="Snowflake/snowflake-arctic-embed-l-v2.0",
        query_prefix="query: ",
        doc_prefix="",
        max_seq_length=1024,
        trust_remote_code=False,
        notes="XLM-RoBERTa base, 8192 context; query: prefix on queries only.",
    ),
}


# ---------------------------------------------------------------------------
# Run one open-model baseline via the shared eval lib
# ---------------------------------------------------------------------------

def run_for_model(cfg: ModelConfig) -> dict:
    from sentence_transformers import SentenceTransformer

    print(f"  Loading model: {cfg.hf_id}")
    model = SentenceTransformer(cfg.hf_id, trust_remote_code=cfg.trust_remote_code)

    return evl.score_model_on_eval(
        model=model,
        engine_name=cfg.name,
        hf_id=cfg.hf_id,
        query_prefix=cfg.query_prefix,
        doc_prefix=cfg.doc_prefix,
        max_seq_length=cfg.max_seq_length,
        notes=cfg.notes,
        cache_key=cfg.name,  # cache corpus embeddings by short model name
        batch_size=32,
        top_k=20,
    )


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

    queries = evl.load_eval()
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[q["text"] for q in queries],
    )
    q_vecs = np.array([r.embedding for r in resp.data], dtype=np.float32)
    norms = np.linalg.norm(q_vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    q_vecs /= norms

    sorted_top = evl.retrieve_top_k(q_vecs, doc_vecs, ids, k=20)
    per_query = [evl.evaluate_query(q, sorted_top[i]) for i, q in enumerate(queries)]

    from src.scorer import aggregate_metrics  # type: ignore[import-not-found]
    return {
        "engine_name": "text-embedding-3-small",
        "hf_id": "openai/text-embedding-3-small",
        "notes": "Baseline replay using precomputed search-lab index.",
        "max_seq_length": 8191,
        "query_prefix": "",
        "doc_prefix": "",
        "n_queries": len(queries),
        "overall": aggregate_metrics(per_query),
        "by_type": evl.aggregate_by(per_query, "query_type"),
        "by_topic": evl.aggregate_by(per_query, "query_topic"),
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
