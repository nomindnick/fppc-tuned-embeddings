"""
Knob sweep over LocalCitationScoreFusion: load one engine, vary `w_bm25`,
`w_sem`, and `cb_threshold` across N trials, and produce a single summary
JSON of all trials' metrics.

This avoids the model-load cost (~5s) per trial, which would otherwise
dominate runtime on a 15-trial sweep.

Usage:
    python scripts/run_sweep.py experiments/<run_id>/sweep_config.json

The sweep config schema:

    {
      "run_id": "sprint5b_d3_sweep",
      "model_dir": "...",
      "sem_npy": "...",
      "sem_ids_json": "...",
      "query_prefix": "query: ",
      "max_seq_length": 512,
      "trials": [
        {"label": "w0.30_cb1.3", "w_bm25": 0.70, "w_sem": 0.30, "cb_threshold": 1.3},
        ...
      ],
      "experiment_dir": "/abs/path/to/experiments/<run_id>"
    }

Writes:
- `<experiment_dir>/sweep_summary.json` — list of {label, params, overall, by_type, by_topic, path_breakdown}
- `<experiment_dir>/trial_<label>/eval_metrics.json` — per-trial full output (per-query)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _eval_lib as evl  # noqa: E402
from _hybrid_engine import LocalCitationScoreFusion  # noqa: E402


def run_trial(engine, queries, top_k: int) -> tuple[list[dict], dict]:
    per_query = []
    paths: dict[str, int] = {}
    for q in queries:
        results, diag = engine.search(
            q["text"], top_k=top_k, query_type=q.get("type")
        )
        row = evl.evaluate_query(q, results)
        row["diagnostics"] = diag
        per_query.append(row)
        p = diag.get("path", "unknown")
        paths[p] = paths.get(p, 0) + 1
    return per_query, paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path")
    parser.add_argument("--top-k", type=int, default=20)
    args = parser.parse_args()

    with open(args.config_path) as f:
        cfg = json.load(f)

    out_dir = cfg["experiment_dir"]
    os.makedirs(out_dir, exist_ok=True)

    print(f"=== {cfg['run_id']} ===")
    print(f"  model_dir = {cfg['model_dir']}")
    print(f"  sem_npy   = {cfg['sem_npy']}")
    print(f"  trials    = {len(cfg['trials'])}")

    # Build engine once with placeholder knobs; mutate per trial.
    engine = LocalCitationScoreFusion(
        model_dir=cfg["model_dir"],
        sem_npy=cfg["sem_npy"],
        sem_ids_json=cfg["sem_ids_json"],
        query_prefix=cfg["query_prefix"],
        cb_threshold=1.3,
        w_bm25=0.4,
        w_sem=0.6,
        semantic_only_path=cfg.get("semantic_only_path", False),
        max_seq_length=cfg.get("max_seq_length", 512),
    )

    queries = evl.load_eval()
    print(f"Loaded {len(queries)} eval queries.")

    from src.scorer import aggregate_metrics  # type: ignore

    summary = {
        "run_id": cfg["run_id"],
        "model_dir": cfg["model_dir"],
        "sem_npy": cfg["sem_npy"],
        "query_prefix": cfg["query_prefix"],
        "n_queries": len(queries),
        "trials": [],
    }

    for trial in cfg["trials"]:
        label = trial["label"]
        engine._w_bm25 = trial["w_bm25"]
        engine._w_sem = trial["w_sem"]
        engine._cb_threshold = trial["cb_threshold"]
        # Per-type adaptive weights (Variant A). Optional in trial config.
        engine._w_sem_by_type = trial.get("w_sem_by_type") or {}
        # Also let allpath be a per-trial knob (overrides the
        # config-level setting if present).
        if "semantic_only_path" in trial:
            engine._semantic_only_path = trial["semantic_only_path"]
        # Variant B — confidence-thresholded semantic boost
        engine._confidence_top1_threshold = trial.get("confidence_top1_threshold")
        engine._confidence_gap_threshold = trial.get("confidence_gap_threshold")
        engine._confidence_boost_w_sem = trial.get("confidence_boost_w_sem", 0.7)

        t0 = time.time()
        per_query, paths = run_trial(engine, queries, args.top_k)
        elapsed = time.time() - t0

        overall = aggregate_metrics(per_query)
        by_type = evl.aggregate_by(per_query, "query_type")
        by_topic = evl.aggregate_by(per_query, "query_topic")

        # Write per-trial JSON
        trial_dir = os.path.join(out_dir, f"trial_{label}")
        os.makedirs(trial_dir, exist_ok=True)
        trial_record = {
            "label": label,
            "w_bm25": trial["w_bm25"],
            "w_sem": trial["w_sem"],
            "cb_threshold": trial["cb_threshold"],
            "w_sem_by_type": trial.get("w_sem_by_type"),
            "semantic_only_path": engine._semantic_only_path,
            "n_queries": len(queries),
            "elapsed_seconds": elapsed,
            "overall": overall,
            "by_type": by_type,
            "by_topic": by_topic,
            "path_breakdown": paths,
            "per_query": per_query,
        }
        with open(os.path.join(trial_dir, "eval_metrics.json"), "w") as f:
            json.dump(trial_record, f, indent=2)

        summary["trials"].append({
            k: trial_record[k]
            for k in ("label", "w_bm25", "w_sem", "cb_threshold",
                      "w_sem_by_type", "semantic_only_path",
                      "elapsed_seconds", "overall", "by_type", "by_topic",
                      "path_breakdown")
        })

        coi = by_topic.get("conflicts_of_interest", {}).get("ndcg@5", 0.0)
        kw = by_type.get("keyword", {}).get("ndcg@5", 0.0)
        fp = by_type.get("fact_pattern", {}).get("ndcg@5", 0.0)
        print(
            f"  [{label:>20s}] "
            f"nDCG@5={overall['ndcg@5']:.4f}  "
            f"MRR={overall['mrr']:.4f}  "
            f"COI={coi:.4f}  "
            f"KW={kw:.4f}  "
            f"FP={fp:.4f}  "
            f"({elapsed:.1f}s)"
        )

    out_path = os.path.join(out_dir, "sweep_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
