"""
Embed the FPPC corpus with a SentenceTransformer model and run the
65-query semantic-only eval. Caches embeddings in the format
`run_sweep.py` expects.

Used to score the MLM-only Snowflake (no d3 contrastive on top) and
the d3-FPPC model in Sprint 4.

Usage:
  python scripts/embed_and_score.py \
    --model-dir /path/to/model \
    --cache-key sprint4_mlmonly \
    --label "MLM-only Snowflake" \
    --out experiments/sprint4_mlmonly/eval_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _eval_lib as evl  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--cache-key", required=True,
                        help="e.g. 'sprint4_mlmonly' → "
                             "corpus_embeddings_<cache_key>_qa_text.npy")
    parser.add_argument("--label", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--query-prefix", default="query: ")
    parser.add_argument("--doc-prefix", default="")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    print(f"Loading SentenceTransformer from {args.model_dir}…")
    model = SentenceTransformer(args.model_dir)

    result = evl.score_model_on_eval(
        model=model,
        engine_name=args.label or args.cache_key,
        hf_id=args.model_dir,
        query_prefix=args.query_prefix,
        doc_prefix=args.doc_prefix,
        max_seq_length=args.max_seq_length,
        notes=f"Semantic-only eval ({args.label or args.cache_key})",
        cache_key=args.cache_key,
        batch_size=args.batch_size,
        top_k=20,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {args.out}")
    overall = result["overall"]
    print("\nHEADLINE (semantic-only):")
    for k in ["mrr", "ndcg@5", "ndcg@10", "precision@5", "recall@10"]:
        print(f"  {k:14s} = {overall.get(k, 0.0):.4f}")
    coi = result["by_topic"].get("conflicts_of_interest", {})
    if coi:
        print(f"  COI nDCG@5    = {coi.get('ndcg@5', 0.0):.4f}")


if __name__ == "__main__":
    main()
