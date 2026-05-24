"""
Roll up Sprint 4 results across the full pipeline:
- baselines (OpenAI/009, Snow-base, d3 cite-only & allpath best)
- Lever 1a (MLM-only + d3-FPPC)
- Lever 1b (MLM-only + d3-FPPC, conservative)
- q001 / q018 gold-rank mechanism check

Run after every Lever-1 variant's downstream eval lands.
"""

from __future__ import annotations

import json
import os
import sys
from glob import glob

SL_009 = "/home/nick/Projects/fppc-opinions-search-lab/results/009-new-corpus.json"
TUNED = "/home/nick/Projects/fppc-tuned-embeddings"


def load(p: str) -> dict:
    with open(p) as f:
        return json.load(f)


def safe_load(p: str) -> dict | None:
    try:
        return load(p)
    except FileNotFoundError:
        return None


def best_trial(sweep_summary: dict | None, key: str = "ndcg@5") -> dict | None:
    if sweep_summary is None:
        return None
    best = None
    best_score = -1
    for t in sweep_summary["trials"]:
        s = t["overall"].get(key, 0.0)
        if s > best_score:
            best_score = s
            best = t
    return best


def fmt(t: dict | None) -> str:
    if not t:
        return "—"
    ov = t["overall"]
    return (
        f"{ov['ndcg@5']:.4f}/{ov['mrr']:.4f} "
        f"(w_sem={t['w_sem']:.2f},cb={t['cb_threshold']:.1f}"
        f"{',allpath' if t.get('semantic_only_path') else ''})"
    )


def main():
    rows = []

    # Production reference
    oa = load(SL_009)
    rows.append((
        "OpenAI / 009 (production)",
        oa["overall"]["ndcg@5"],
        oa["overall"]["mrr"],
        oa.get("by_topic", {}).get("conflicts_of_interest", {}).get("ndcg@5", 0),
        "stock w=0.60,cb=1.3",
    ))

    # Sprint 5 ship candidate (d3 best, cite-only)
    sw = load(f"{TUNED}/experiments/sprint5b_d3_finegrain/sweep_summary.json")
    best = next(t for t in sw["trials"] if t["label"] == "w0.40_cb1.5")
    rows.append((
        "d3 (Sprint 3) @ best cite-only [SHIP CANDIDATE]",
        best["overall"]["ndcg@5"],
        best["overall"]["mrr"],
        best["by_topic"].get("conflicts_of_interest", {}).get("ndcg@5", 0),
        "w=0.40,cb=1.5",
    ))

    # d3 allpath
    best = next(t for t in sw["trials"] if t["label"] == "w0.40_cb2.0")
    rows.append((
        "d3 (Sprint 3) @ best allpath",
        best["overall"]["ndcg@5"],
        best["overall"]["mrr"],
        best["by_topic"].get("conflicts_of_interest", {}).get("ndcg@5", 0),
        "w=0.40,cb=2.0,allpath",
    ))

    # Snowflake-base at best knobs (from sprint5b_snow_cb15)
    sw = load(f"{TUNED}/experiments/sprint5b_snow_cb15/sweep_summary.json")
    b = best_trial(sw)
    rows.append((
        "Snowflake-base (no FT) @ best",
        b["overall"]["ndcg@5"],
        b["overall"]["mrr"],
        b["by_topic"].get("conflicts_of_interest", {}).get("ndcg@5", 0),
        f"w={b['w_sem']:.2f},cb={b['cb_threshold']:.1f}",
    ))

    # Lever 1a: MLM-only + d3-FPPC
    sw = safe_load(f"{TUNED}/experiments/sprint4_mlmonly_sweep/sweep_summary.json")
    b = best_trial(sw)
    if b:
        rows.append((
            "Lever 1a — MLM-only Snowflake (LR=1e-5, 2ep)",
            b["overall"]["ndcg@5"],
            b["overall"]["mrr"],
            b["by_topic"].get("conflicts_of_interest", {}).get("ndcg@5", 0),
            f"w={b['w_sem']:.2f},cb={b['cb_threshold']:.1f}",
        ))
    sw = safe_load(f"{TUNED}/experiments/sprint4_d3fppc_sweep/sweep_summary.json")
    b = best_trial(sw)
    if b:
        rows.append((
            "Lever 1a — d3-FPPC (contrastive on top)",
            b["overall"]["ndcg@5"],
            b["overall"]["mrr"],
            b["by_topic"].get("conflicts_of_interest", {}).get("ndcg@5", 0),
            f"w={b['w_sem']:.2f},cb={b['cb_threshold']:.1f}"
            + (",allpath" if b.get("semantic_only_path") else ""),
        ))

    # Lever 1b
    sw = safe_load(f"{TUNED}/experiments/sprint4b_mlmonly_sweep/sweep_summary.json")
    b = best_trial(sw)
    if b:
        rows.append((
            "Lever 1b — MLM-only Snowflake (LR=5e-6, 1ep, freeze500)",
            b["overall"]["ndcg@5"],
            b["overall"]["mrr"],
            b["by_topic"].get("conflicts_of_interest", {}).get("ndcg@5", 0),
            f"w={b['w_sem']:.2f},cb={b['cb_threshold']:.1f}",
        ))
    sw = safe_load(f"{TUNED}/experiments/sprint4b_d3fppc_sweep/sweep_summary.json")
    b = best_trial(sw)
    if b:
        rows.append((
            "Lever 1b — d3-FPPC-b (contrastive on top)",
            b["overall"]["ndcg@5"],
            b["overall"]["mrr"],
            b["by_topic"].get("conflicts_of_interest", {}).get("ndcg@5", 0),
            f"w={b['w_sem']:.2f},cb={b['cb_threshold']:.1f}"
            + (",allpath" if b.get("semantic_only_path") else ""),
        ))

    print(f"{'Engine':<55s} {'nDCG@5':>8s} {'MRR':>8s} {'COI':>8s}  Config")
    print("-" * 110)
    for name, n, m, c, cfg in rows:
        print(f"{name:<55s} {n:>8.4f} {m:>8.4f} {c:>8.4f}  {cfg}")


if __name__ == "__main__":
    main()
