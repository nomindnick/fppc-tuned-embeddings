"""
Score a Sprint 3 fine-tuned model on the 65-query graded eval set.

Reuses scripts/_eval_lib.py so results are bit-identical-methodology with
the off-the-shelf baselines in results/check2_*.json.

Usage:
  python scripts/score_finetuned.py experiments/<run_id>/config.json

Writes experiments/<run_id>/eval_metrics.json with the same JSON schema as
results/check2_*.json, plus Sprint 3 metadata (run_id, base_model, loss,
positive_column).
"""

from __future__ import annotations

import argparse
import json
import os

from sentence_transformers import SentenceTransformer

import _eval_lib as evl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the run config JSON")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable corpus-embedding cache for this run")
    args = parser.parse_args()

    with open(args.config_path) as f:
        cfg = json.load(f)

    model_dir = cfg["model_save_dir"]
    run_id = cfg["run_id"]

    print(f"=== Scoring {run_id} ===")
    print(f"  model_dir = {model_dir}")
    model = SentenceTransformer(model_dir)

    extra = {
        "run_id": run_id,
        "base_model": cfg["base_model"],
        "loss": cfg["loss"],
        "positive_column": cfg["positive_column"],
        "use_hard_negatives": cfg["use_hard_negatives"],
        "n_hard_negatives_per_row": cfg.get("n_hard_negatives_per_row", 0),
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "mini_batch_size": cfg.get("mini_batch_size"),
        "learning_rate": cfg["learning_rate"],
        "warmup_ratio": cfg["warmup_ratio"],
        "seed": cfg["seed"],
        "bf16": cfg.get("bf16", False),
    }

    result = evl.score_model_on_eval(
        model=model,
        engine_name=run_id,
        hf_id=cfg["base_model"],
        query_prefix=cfg["query_prefix"],
        doc_prefix=cfg["doc_prefix"],
        max_seq_length=cfg["max_seq_length"],
        notes=f"Sprint 3 fine-tuned run: {cfg['loss']} on {cfg['positive_column']}",
        cache_key=None if args.no_cache else f"sprint3_{run_id}",
        batch_size=32,
        top_k=20,
        extra=extra,
    )

    out_path = os.path.join(cfg["experiment_dir"], "eval_metrics.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {out_path}")

    overall = result["overall"]
    print("\nHEADLINE:")
    for k in ["mrr", "ndcg@5", "ndcg@10", "precision@5", "recall@10"]:
        print(f"  {k:14s} = {overall.get(k, 0.0):.4f}")
    coi = result["by_topic"].get("conflicts_of_interest", {})
    if coi:
        print(f"  COI nDCG@5    = {coi.get('ndcg@5', 0.0):.4f}  "
              f"(SPEC gate ≥ 0.20)")


if __name__ == "__main__":
    main()
