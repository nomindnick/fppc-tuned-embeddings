"""
Per-query delta analysis: compare engines on the 65-query eval at the
individual-query level. Identify queries where d3 wins vs loses vs the
production OpenAI/009, look for clusters by topic/type/routing-path, and
flag queries where d3 and f2 disagree (ensemble candidates).

Outputs a single markdown report to stdout (redirect to file as wanted).

Usage:
    python scripts/analyze_per_query.py > notes/sprint5_per_query_analysis.md
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict

SL_009 = "/home/nick/Projects/fppc-opinions-search-lab/results/009-new-corpus.json"
EVAL = "/home/nick/Projects/fppc-opinions-eval/eval/dataset.json"
TUNED = "/home/nick/Projects/fppc-tuned-embeddings"

D3_CITE = f"{TUNED}/experiments/sprint5b_d3_finegrain/trial_w0.40_cb1.5/eval_metrics.json"
D3_ALL  = f"{TUNED}/experiments/sprint5c_d3_allpath/trial_w0.40_cb2.0_allpath/eval_metrics.json"
F2_CITE = f"{TUNED}/experiments/sprint5b_f2_cb15/trial_w0.40_cb1.5/eval_metrics.json"
SNOW_CITE = f"{TUNED}/experiments/sprint5b_snow_cb15/trial_w0.30_cb1.5/eval_metrics.json"
D3_STOCK = f"{TUNED}/experiments/sprint5_d3_009stock/eval_metrics.json"


def load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def per_q_map(run: dict) -> dict[str, dict]:
    return {q["query_id"]: q for q in run["per_query"]}


def fmt(x: float) -> str:
    return f"{x:+.3f}" if x != 0 else "  0.000"


def main() -> None:
    eval_ds = load(EVAL)["queries"]
    eval_by_id = {q["id"]: q for q in eval_ds}

    oa = load(SL_009)
    d3c = load(D3_CITE)
    d3a = load(D3_ALL)
    f2 = load(F2_CITE)
    snow = load(SNOW_CITE)
    d3s = load(D3_STOCK)

    oam = per_q_map(oa)
    d3cm = per_q_map(d3c)
    d3am = per_q_map(d3a)
    f2m = per_q_map(f2)
    snowm = per_q_map(snow)
    d3sm = per_q_map(d3s)

    # ----- header -----
    print("# Sprint 5 — Per-Query Delta Analysis\n")
    print("Engines compared (best configurations from Sprint 5B/C):\n")
    print("| Tag | Engine | nDCG@5 | MRR |")
    print("|---|---|---:|---:|")
    print(f"| `oa`      | OpenAI / 009 stock (production)      | "
          f"{oa['overall']['ndcg@5']:.4f} | {oa['overall']['mrr']:.4f} |")
    print(f"| `snow`    | Snow-base @ w=0.30 cb=1.5 (cite-only) | "
          f"{snow['overall']['ndcg@5']:.4f} | {snow['overall']['mrr']:.4f} |")
    print(f"| `f2`      | f2 @ w=0.40 cb=1.5 (cite-only)        | "
          f"{f2['overall']['ndcg@5']:.4f} | {f2['overall']['mrr']:.4f} |")
    print(f"| `d3-cite` | d3 @ w=0.40 cb=1.5 (cite-only)        | "
          f"{d3c['overall']['ndcg@5']:.4f} | {d3c['overall']['mrr']:.4f} |")
    print(f"| `d3-all`  | d3 @ w=0.40 cb=2.0 (allpath)          | "
          f"{d3a['overall']['ndcg@5']:.4f} | {d3a['overall']['mrr']:.4f} |")
    print(f"| `d3-stk`  | d3 @ w=0.60 cb=1.3 (stock 009 knobs)  | "
          f"{d3s['overall']['ndcg@5']:.4f} | {d3s['overall']['mrr']:.4f} |\n")

    # ----- d3-cite vs OpenAI: per-query deltas (sorted) -----
    deltas = []
    for qid in sorted(oam):
        if qid not in d3cm:
            continue
        oq, dq = oam[qid], d3cm[qid]
        d_ndcg5 = dq["metrics"]["ndcg@5"] - oq["metrics"]["ndcg@5"]
        d_mrr = dq["metrics"]["mrr"] - oq["metrics"]["mrr"]
        path = (dq.get("diagnostics") or {}).get("path", "?")
        deltas.append({
            "qid": qid,
            "type": oq["query_type"],
            "topic": oq["query_topic"],
            "text": oq["query_text"],
            "oa_ndcg5": oq["metrics"]["ndcg@5"],
            "d3_ndcg5": dq["metrics"]["ndcg@5"],
            "d_ndcg5": d_ndcg5,
            "oa_mrr": oq["metrics"]["mrr"],
            "d3_mrr": dq["metrics"]["mrr"],
            "d_mrr": d_mrr,
            "path": path,
        })

    # Headline counts
    nontrivial = [d for d in deltas
                  if abs(d["d_ndcg5"]) >= 0.05 or abs(d["d_mrr"]) >= 0.05]
    wins = [d for d in deltas if d["d_ndcg5"] >= 0.05]
    losses = [d for d in deltas if d["d_ndcg5"] <= -0.05]
    flat = [d for d in deltas if abs(d["d_ndcg5"]) < 0.05]
    print(f"## d3-cite vs OpenAI / 009 — nDCG@5 deltas\n")
    print(f"- 65 queries total")
    print(f"- {len(wins):>3d} queries d3 wins by ≥ 0.05 nDCG@5")
    print(f"- {len(losses):>3d} queries d3 loses by ≥ 0.05 nDCG@5")
    print(f"- {len(flat):>3d} queries within ±0.05 (effectively tied or 0/0)\n")

    # Wins table
    wins_sorted = sorted(wins, key=lambda d: -d["d_ndcg5"])
    print("### Where d3-cite *wins* by ≥ 0.05 nDCG@5\n")
    print("| qid | type | topic | path | OA | d3 | Δ nDCG@5 | Δ MRR | query |")
    print("|---|---|---|---|---:|---:|---:|---:|---|")
    for d in wins_sorted:
        text = d["text"][:60].replace("|", "\\|")
        print(f"| {d['qid']} | {d['type'][:4]} | {d['topic'][:6]} | "
              f"{d['path'][:8]} | {d['oa_ndcg5']:.3f} | {d['d3_ndcg5']:.3f} | "
              f"**{fmt(d['d_ndcg5'])}** | {fmt(d['d_mrr'])} | {text} |")
    print()

    # Losses table
    losses_sorted = sorted(losses, key=lambda d: d["d_ndcg5"])
    print("### Where d3-cite *loses* by ≥ 0.05 nDCG@5\n")
    print("| qid | type | topic | path | OA | d3 | Δ nDCG@5 | Δ MRR | query |")
    print("|---|---|---|---|---:|---:|---:|---:|---|")
    for d in losses_sorted:
        text = d["text"][:60].replace("|", "\\|")
        print(f"| {d['qid']} | {d['type'][:4]} | {d['topic'][:6]} | "
              f"{d['path'][:8]} | {d['oa_ndcg5']:.3f} | {d['d3_ndcg5']:.3f} | "
              f"**{fmt(d['d_ndcg5'])}** | {fmt(d['d_mrr'])} | {text} |")
    print()

    # ----- Win/loss breakdown by topic and type -----
    print("### Win/loss breakdown by topic and type\n")
    def bucket_table(keyfn, label):
        buckets = defaultdict(lambda: {"win": 0, "loss": 0, "flat": 0, "d_sum": 0.0, "n": 0})
        for d in deltas:
            k = keyfn(d)
            if d["d_ndcg5"] >= 0.05:
                buckets[k]["win"] += 1
            elif d["d_ndcg5"] <= -0.05:
                buckets[k]["loss"] += 1
            else:
                buckets[k]["flat"] += 1
            buckets[k]["d_sum"] += d["d_ndcg5"]
            buckets[k]["n"] += 1
        print(f"**By {label}**:\n")
        print(f"| {label} | n | wins | losses | flat | mean Δ nDCG@5 |")
        print("|---|---:|---:|---:|---:|---:|")
        for k in sorted(buckets):
            b = buckets[k]
            mean_d = b["d_sum"] / b["n"] if b["n"] else 0.0
            print(f"| {k} | {b['n']} | {b['win']} | {b['loss']} | {b['flat']} | "
                  f"{mean_d:+.4f} |")
        print()
    bucket_table(lambda d: d["topic"], "topic")
    bucket_table(lambda d: d["type"], "type")
    bucket_table(lambda d: d["path"], "routing path")

    # ----- COI deep dive -----
    print("## COI deep-dive — where the remaining gap to OpenAI lives\n")
    coi_deltas = [d for d in deltas if d["topic"] == "conflicts_of_interest"]
    coi_d_sum = sum(d["d_ndcg5"] for d in coi_deltas)
    coi_n = len(coi_deltas)
    print(f"- {coi_n} COI queries; OpenAI mean nDCG@5 = "
          f"{sum(d['oa_ndcg5'] for d in coi_deltas)/coi_n:.4f}, "
          f"d3-cite mean = {sum(d['d3_ndcg5'] for d in coi_deltas)/coi_n:.4f}, "
          f"mean Δ = {coi_d_sum/coi_n:+.4f}")

    # Per-COI-query table sorted by Δ ascending (worst losses first)
    print(f"\n### All {coi_n} COI queries — d3-cite vs OpenAI")
    print("| qid | type | path | OA | d3 | Δ | query |")
    print("|---|---|---|---:|---:|---:|---|")
    for d in sorted(coi_deltas, key=lambda x: x["d_ndcg5"]):
        text = d["text"][:60].replace("|", "\\|")
        marker = ""
        if d["d_ndcg5"] <= -0.05:
            marker = " 🔻"
        elif d["d_ndcg5"] >= 0.05:
            marker = " ✅"
        print(f"| {d['qid']} | {d['type'][:4]} | {d['path'][:8]} | "
              f"{d['oa_ndcg5']:.3f} | {d['d3_ndcg5']:.3f} | "
              f"**{fmt(d['d_ndcg5'])}**{marker} | {text} |")
    print()

    # ----- d3-cite vs d3-allpath: where does allpath help/hurt? -----
    print("## d3-cite vs d3-allpath — what the routing extension changes\n")
    routing_changes = []
    for qid in sorted(d3cm):
        if qid not in d3am:
            continue
        cq, aq = d3cm[qid], d3am[qid]
        d_ndcg5 = aq["metrics"]["ndcg@5"] - cq["metrics"]["ndcg@5"]
        d_mrr = aq["metrics"]["mrr"] - cq["metrics"]["mrr"]
        if abs(d_ndcg5) < 0.01 and abs(d_mrr) < 0.01:
            continue
        routing_changes.append({
            "qid": qid,
            "type": cq["query_type"],
            "topic": cq["query_topic"],
            "text": cq["query_text"],
            "cite_ndcg5": cq["metrics"]["ndcg@5"],
            "all_ndcg5": aq["metrics"]["ndcg@5"],
            "d_ndcg5": d_ndcg5,
            "cite_mrr": cq["metrics"]["mrr"],
            "all_mrr": aq["metrics"]["mrr"],
            "d_mrr": d_mrr,
            "cite_path": (cq.get("diagnostics") or {}).get("path", "?"),
            "all_path": (aq.get("diagnostics") or {}).get("path", "?"),
        })
    routing_changes.sort(key=lambda d: -d["d_ndcg5"])
    print(f"{len(routing_changes)} queries where allpath changes the result "
          f"by ≥ 0.01 on nDCG@5 or MRR.\n")
    if routing_changes:
        print("| qid | type | topic | cite-path → all-path | Δ nDCG@5 | Δ MRR | query |")
        print("|---|---|---|---|---:|---:|---|")
        for d in routing_changes:
            text = d["text"][:50].replace("|", "\\|")
            paths = f"{d['cite_path'][:8]} → {d['all_path'][:8]}"
            print(f"| {d['qid']} | {d['type'][:4]} | {d['topic'][:6]} | "
                  f"{paths} | **{fmt(d['d_ndcg5'])}** | {fmt(d['d_mrr'])} | {text} |")
        print()

    # ----- Ensemble candidate: d3 vs f2 per-query -----
    print("## d3 vs f2 (cite-only best) — ensemble candidate analysis\n")
    print("Per-query nDCG@5 for the two fine-tunes at their respective best configs.\n")
    ens_rows = []
    for qid in sorted(d3cm):
        if qid not in f2m:
            continue
        dq, fq = d3cm[qid], f2m[qid]
        d_score = dq["metrics"]["ndcg@5"]
        f_score = fq["metrics"]["ndcg@5"]
        ens_rows.append({
            "qid": qid,
            "type": dq["query_type"],
            "topic": dq["query_topic"],
            "d3": d_score,
            "f2": f_score,
            "max": max(d_score, f_score),
            "min": min(d_score, f_score),
            "oa": oam[qid]["metrics"]["ndcg@5"],
        })

    sum_d3 = sum(r["d3"] for r in ens_rows) / len(ens_rows)
    sum_f2 = sum(r["f2"] for r in ens_rows) / len(ens_rows)
    sum_max = sum(r["max"] for r in ens_rows) / len(ens_rows)
    sum_oa = sum(r["oa"] for r in ens_rows) / len(ens_rows)
    print(f"- mean nDCG@5 (d3 best):  {sum_d3:.4f}")
    print(f"- mean nDCG@5 (f2 best):  {sum_f2:.4f}")
    print(f"- mean nDCG@5 (per-query MAX of d3, f2): **{sum_max:.4f}** "
          f"(oracle upper bound for a query-router ensemble)")
    print(f"- mean nDCG@5 (OpenAI):   {sum_oa:.4f}\n")

    # Queries where d3 and f2 disagree by ≥ 0.10
    disagree = sorted(
        [r for r in ens_rows if abs(r["d3"] - r["f2"]) >= 0.10],
        key=lambda r: -abs(r["d3"] - r["f2"]),
    )
    print(f"{len(disagree)} queries where d3 and f2 differ by ≥ 0.10 nDCG@5 "
          "(ensemble would have most to gain on these).\n")
    if disagree:
        print("| qid | type | topic | d3 | f2 | Δ |")
        print("|---|---|---|---:|---:|---:|")
        for r in disagree[:20]:
            print(f"| {r['qid']} | {r['type'][:4]} | {r['topic'][:6]} | "
                  f"{r['d3']:.3f} | {r['f2']:.3f} | "
                  f"{r['f2'] - r['d3']:+.3f} |")
        print()


if __name__ == "__main__":
    main()
