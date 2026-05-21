"""Print per-overall, per-topic and per-type tables across all four baselines."""
from __future__ import annotations

import json

FILES = {
    "OpenAI text-embedding-3-small": "results/check2_openai_baseline.json",
    "BGE-base-en-v1.5": "results/check2_bge-base-en-v1.5.json",
    "BGE-large-en-v1.5": "results/check2_bge-large-en-v1.5.json",
    "Nomic-embed-text-v1.5": "results/check2_nomic-embed-text-v1.5.json",
    "gte-modernbert-base": "results/check2_gte-modernbert-base.json",
    "Qwen3-Embedding-0.6B": "results/check2_qwen3-embed-0.6b.json",
    "Snowflake-arctic-l-v2": "results/check2_snowflake-arctic-embed-l-v2.0.json",
}


def fmt_cell(v):
    return f"{v:.3f}" if v is not None else "  -  "


def main() -> None:
    data = {name: json.load(open(p)) for name, p in FILES.items()}

    print("=== OVERALL ===")
    keys = ["mrr", "ndcg@5", "ndcg@10", "precision@5", "precision@10", "recall@10", "recall@20"]
    header = "model".ljust(32) + " | " + " | ".join(k.rjust(9) for k in keys)
    print(header)
    for name, d in data.items():
        o = d["overall"]
        row = name.ljust(32) + " | " + " | ".join(fmt_cell(o.get(k)).rjust(9) for k in keys)
        print(row)

    print()
    print("=== BY TYPE (nDCG@5 / n_queries) ===")
    all_types = sorted({t for d in data.values() for t in d.get("by_type", {}).keys()})
    header_cells = [n.ljust(24) for n in data]
    print("type".ljust(20) + " | n_q | " + " | ".join(header_cells))
    for t in all_types:
        n_q = None
        cells = []
        for n, d in data.items():
            bucket = d.get("by_type", {}).get(t, {})
            if n_q is None and "n_queries" in bucket:
                n_q = bucket["n_queries"]
            cells.append(fmt_cell(bucket.get("ndcg@5")).ljust(24))
        nq_str = str(n_q if n_q is not None else "")
        print(t.ljust(20) + " | " + nq_str.rjust(3) + " | " + " | ".join(cells))

    print()
    print("=== BY TOPIC (nDCG@5 / n_queries) ===")
    all_topics = sorted({t for d in data.values() for t in d.get("by_topic", {}).keys()})
    print("topic".ljust(28) + " | n_q | " + " | ".join(header_cells))
    for t in all_topics:
        n_q = None
        cells = []
        for n, d in data.items():
            bucket = d.get("by_topic", {}).get(t, {})
            if n_q is None and "n_queries" in bucket:
                n_q = bucket["n_queries"]
            cells.append(fmt_cell(bucket.get("ndcg@5")).ljust(24))
        nq_str = str(n_q if n_q is not None else "")
        print(t.ljust(28) + " | " + nq_str.rjust(3) + " | " + " | ".join(cells))


if __name__ == "__main__":
    main()
