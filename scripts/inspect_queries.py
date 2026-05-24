"""
Deep-dive on specific queries where d3 underperforms OpenAI: show the
query text, gold judgments, and top-10 results from each engine
side-by-side. Helps isolate "fine-tune broke this query" from "Snowflake
architecture can't handle this query."

Usage:
    python scripts/inspect_queries.py q001 q018 q012 q046 q048 q041 q060 q054
"""

from __future__ import annotations

import argparse
import json
import os

SL_009 = "/home/nick/Projects/fppc-opinions-search-lab/results/009-new-corpus.json"
EVAL = "/home/nick/Projects/fppc-opinions-eval/eval/dataset.json"
TUNED = "/home/nick/Projects/fppc-tuned-embeddings"

D3_CITE = f"{TUNED}/experiments/sprint5b_d3_finegrain/trial_w0.40_cb1.5/eval_metrics.json"
F2_CITE = f"{TUNED}/experiments/sprint5b_f2_cb15/trial_w0.40_cb1.5/eval_metrics.json"
SNOW_CITE = f"{TUNED}/experiments/sprint5b_snow_cb15/trial_w0.30_cb1.5/eval_metrics.json"


def load(p: str) -> dict:
    with open(p) as f:
        return json.load(f)


def per_q(run: dict) -> dict[str, dict]:
    return {q["query_id"]: q for q in run["per_query"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("qids", nargs="+", help="Query IDs to inspect (e.g. q001)")
    args = parser.parse_args()

    eval_ds = load(EVAL)["queries"]
    eval_by_id = {q["id"]: q for q in eval_ds}

    oa = per_q(load(SL_009))
    d3 = per_q(load(D3_CITE))
    f2 = per_q(load(F2_CITE))
    snow = per_q(load(SNOW_CITE))

    for qid in args.qids:
        if qid not in eval_by_id:
            print(f"!! {qid} not in eval dataset")
            continue
        q = eval_by_id[qid]
        oaq, d3q, f2q, sq = oa.get(qid), d3.get(qid), f2.get(qid), snow.get(qid)
        if not all([oaq, d3q, f2q, sq]):
            print(f"!! {qid} missing in one of the engine outputs")
            continue

        print(f"\n{'=' * 90}")
        print(f"  {qid}  [{q.get('type')}/{q.get('topic')}]")
        print(f"  {q['text']}")
        print(f"{'=' * 90}")

        # Gold judgments
        golds = q["relevance_judgments"]
        score_2 = sorted([g["opinion_id"] for g in golds if g["score"] == 2])
        score_1 = sorted([g["opinion_id"] for g in golds if g["score"] == 1])
        print(f"\n  GOLD (score=2, n={len(score_2)}): {score_2}")
        print(f"  GOLD (score=1, n={len(score_1)}): {score_1}")
        gold_2 = set(score_2)
        gold_1 = set(score_1)

        # Metric summary
        def mline(name, qq):
            m = qq["metrics"]
            return (
                f"  {name:>15s}  nDCG@5={m['ndcg@5']:.3f}  MRR={m['mrr']:.3f}  "
                f"path={(qq.get('diagnostics') or {}).get('path', '?')}"
            )
        print()
        print(mline("OpenAI/009", oaq))
        print(mline("Snow-base", sq))
        print(mline("f2 (LoRA)", f2q))
        print(mline("d3", d3q))

        # Top-10 side-by-side
        def annotated(rid: str) -> str:
            if rid in gold_2:
                return f"**{rid}**(2)"
            if rid in gold_1:
                return f"{rid}(1)"
            return rid

        oa_top = oaq["results"][:10]
        sn_top = sq["results"][:10]
        f2_top = f2q["results"][:10]
        d3_top = d3q["results"][:10]

        print()
        print("  Top-10 (** = score=2 gold, (1) = score=1 gold):")
        print(f"  {'rank':>4s}  {'OpenAI/009':<22s}  {'Snow-base':<22s}  "
              f"{'f2':<22s}  {'d3':<22s}")
        for i in range(10):
            oa_r = annotated(oa_top[i]) if i < len(oa_top) else ""
            sn_r = annotated(sn_top[i]) if i < len(sn_top) else ""
            f2_r = annotated(f2_top[i]) if i < len(f2_top) else ""
            d3_r = annotated(d3_top[i]) if i < len(d3_top) else ""
            print(f"  {i+1:>4d}  {oa_r:<22s}  {sn_r:<22s}  {f2_r:<22s}  {d3_r:<22s}")


if __name__ == "__main__":
    main()
