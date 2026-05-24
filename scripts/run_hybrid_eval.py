"""
Drive a hybrid-fusion engine against the 65-query graded eval.

Usage:
    python scripts/run_hybrid_eval.py experiments/<run_id>/config.json

The config schema mirrors Sprint 3's, with hybrid-fusion fields added:

    {
      "run_id": "sprint5_d3_009stock",
      "fusion_engine": "LocalCitationScoreFusion",
      "model_dir": "/abs/path/to/model",
      "sem_npy": "/abs/path/to/corpus_embeddings_<tag>_qa_text.npy",
      "sem_ids_json": "/abs/path/to/corpus_embeddings_<tag>_ids.json",
      "query_prefix": "query: ",
      "doc_prefix": "",
      "max_seq_length": 512,
      "cb_threshold": 1.3,
      "w_bm25": 0.4,
      "w_sem": 0.6,
      "semantic_only_path": false,
      "experiment_dir": "/abs/path/to/experiments/<run_id>",
      "notes": "..."
    }

Writes `<experiment_dir>/eval_metrics.json` with the same overall/by_type/
by_topic/per_query shape as `results/check2_*.json`. Adds a `diagnostics`
field per query (path taken: bm25_only_noncite / cb_fired / fusion /
empty_pool) so we can break down the headline number by routing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Reuse Sprint 3's aggregators + scoring imports (which themselves pull
# from fppc-opinions-eval) so the numbers are bit-identical to Check 2.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _eval_lib as evl  # noqa: E402
from _hybrid_engine import LocalCitationScoreFusion  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    with open(args.config_path) as f:
        cfg = json.load(f)

    run_id = cfg["run_id"]
    out_dir = cfg["experiment_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== {run_id} ===")
    print(f"  fusion_engine = {cfg['fusion_engine']}")
    print(f"  model_dir = {cfg['model_dir']}")
    print(f"  sem_npy = {cfg['sem_npy']}")
    print(f"  cb_threshold={cfg['cb_threshold']} "
          f"w_bm25={cfg['w_bm25']} w_sem={cfg['w_sem']} "
          f"semantic_only_path={cfg.get('semantic_only_path', False)}")

    engine = LocalCitationScoreFusion(
        model_dir=cfg["model_dir"],
        sem_npy=cfg["sem_npy"],
        sem_ids_json=cfg["sem_ids_json"],
        query_prefix=cfg["query_prefix"],
        cb_threshold=cfg["cb_threshold"],
        w_bm25=cfg["w_bm25"],
        w_sem=cfg["w_sem"],
        semantic_only_path=cfg.get("semantic_only_path", False),
        max_seq_length=cfg.get("max_seq_length", 512),
    )

    queries = evl.load_eval()
    print(f"Running {len(queries)} queries…")

    per_query = []
    t0 = time.time()
    for i, q in enumerate(queries):
        results, diag = engine.search(q["text"], top_k=args.top_k)
        row = evl.evaluate_query(q, results)
        row["diagnostics"] = diag
        per_query.append(row)
        if (i + 1) % 10 == 0 or i == len(queries) - 1:
            print(f"  [{i + 1}/{len(queries)}] "
                  f"t={time.time() - t0:.1f}s last_path={diag.get('path')}")

    # Same aggregator path as Sprint 3 / Check 2 — bit-identical metrics.
    from src.scorer import aggregate_metrics  # type: ignore
    result = {
        "engine_name": run_id,
        "fusion_engine": cfg["fusion_engine"],
        "model_dir": cfg["model_dir"],
        "sem_npy": cfg["sem_npy"],
        "query_prefix": cfg["query_prefix"],
        "doc_prefix": cfg.get("doc_prefix", ""),
        "max_seq_length": cfg.get("max_seq_length"),
        "cb_threshold": cfg["cb_threshold"],
        "w_bm25": cfg["w_bm25"],
        "w_sem": cfg["w_sem"],
        "semantic_only_path": cfg.get("semantic_only_path", False),
        "notes": cfg.get("notes", ""),
        "n_queries": len(queries),
        "overall": aggregate_metrics(per_query),
        "by_type": evl.aggregate_by(per_query, "query_type"),
        "by_topic": evl.aggregate_by(per_query, "query_topic"),
        "per_query": per_query,
        "path_breakdown": {},
    }
    # Path counts so we can attribute lift to routing decisions.
    paths: dict[str, int] = {}
    for q in per_query:
        p = q["diagnostics"].get("path", "unknown")
        paths[p] = paths.get(p, 0) + 1
    result["path_breakdown"] = paths

    out_path = os.path.join(out_dir, "eval_metrics.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")

    overall = result["overall"]
    print("\nHEADLINE:")
    for k in ["mrr", "ndcg@5", "ndcg@10", "precision@5", "recall@10"]:
        print(f"  {k:14s} = {overall.get(k, 0.0):.4f}")
    coi = result["by_topic"].get("conflicts_of_interest", {})
    if coi:
        print(f"  COI nDCG@5    = {coi.get('ndcg@5', 0.0):.4f}  (SPEC gate ≥ 0.20)")
    print(f"  Path counts   = {paths}")


if __name__ == "__main__":
    main()
