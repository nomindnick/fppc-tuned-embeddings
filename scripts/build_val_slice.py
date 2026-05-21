"""
Carve a validation slice out of the training pairs.

Reads data/training/pairs.jsonl, samples ~5% of rows stratified by year, and
writes them to data/training/val_slice.jsonl with the same row schema. The
trainer is expected to read pairs.jsonl and skip any opinion_id that appears
in val_slice.jsonl.

The slice is reproducible: deterministic seed (RANDOM_SEED below).
"""

from __future__ import annotations

import json
import os
import random
import sys
from collections import Counter, defaultdict

PAIRS_PATH = "/home/nick/Projects/fppc-tuned-embeddings/data/training/pairs.jsonl"
OUT_VAL = "/home/nick/Projects/fppc-tuned-embeddings/data/training/val_slice.jsonl"
OUT_COVERAGE = "/home/nick/Projects/fppc-tuned-embeddings/data/training/val_slice_coverage.json"

TARGET_FRACTION = 0.05
RANDOM_SEED = 20260521  # date the slice was first constructed


def main() -> int:
    if not os.path.exists(PAIRS_PATH):
        print(f"ERROR: {PAIRS_PATH} not found. Run scripts/build_training_pairs.py first.", file=sys.stderr)
        return 1

    rows: list[dict] = []
    with open(PAIRS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    print(f"Loaded {len(rows)} training pairs.")

    by_year: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by_year[r["year"]].append(i)

    rng = random.Random(RANDOM_SEED)
    sampled_indices: list[int] = []
    sampled_by_year: dict[int, int] = {}
    for year in sorted(by_year):
        idxs = by_year[year]
        n = round(len(idxs) * TARGET_FRACTION)
        if n <= 0:
            sampled_by_year[year] = 0
            continue
        chosen = rng.sample(idxs, n)
        sampled_indices.extend(chosen)
        sampled_by_year[year] = n

    sampled_indices.sort()
    print(f"Sampled {len(sampled_indices)} pairs for validation "
          f"({100 * len(sampled_indices) / len(rows):.2f}% of pairs).")

    with open(OUT_VAL, "w") as f:
        for i in sampled_indices:
            f.write(json.dumps(rows[i]) + "\n")

    sampled = [rows[i] for i in sampled_indices]
    qs_counts = Counter(r["question_source"] for r in sampled)
    body_complete = sum(1 for r in sampled if r.get("body_complete"))
    topic_counts = Counter(r.get("topic_primary") or "(unclassified)" for r in sampled)

    pop_qs = Counter(r["question_source"] for r in rows)
    pop_body_complete = sum(1 for r in rows if r.get("body_complete"))
    pop_topic = Counter(r.get("topic_primary") or "(unclassified)" for r in rows)

    def _pct(num: int, denom: int) -> float:
        return 0.0 if denom == 0 else round(100 * num / denom, 2)

    coverage = {
        "n_pairs_total": len(rows),
        "n_val_sampled": len(sampled_indices),
        "fraction_target": TARGET_FRACTION,
        "fraction_actual": round(len(sampled_indices) / len(rows), 4),
        "random_seed": RANDOM_SEED,
        "stratification": "year",
        "per_year_sample_counts": {str(k): sampled_by_year[k] for k in sorted(sampled_by_year)},
        "question_source": {
            "val": dict(qs_counts),
            "val_pct": {k: _pct(v, len(sampled_indices)) for k, v in qs_counts.items()},
            "population_pct": {k: _pct(v, len(rows)) for k, v in pop_qs.items()},
        },
        "body_complete": {
            "val_n": body_complete,
            "val_pct": _pct(body_complete, len(sampled_indices)),
            "population_pct": _pct(pop_body_complete, len(rows)),
        },
        "topic_primary": {
            "val_pct": {k: _pct(v, len(sampled_indices)) for k, v in topic_counts.most_common()},
            "population_pct": {k: _pct(v, len(rows)) for k, v in pop_topic.most_common()},
        },
    }
    with open(OUT_COVERAGE, "w") as f:
        json.dump(coverage, f, indent=2)

    print()
    print(f"Val slice written:  {OUT_VAL}")
    print(f"Coverage report:    {OUT_COVERAGE}")
    print()
    print("Distribution check (val vs population, %):")
    print("  question_source:")
    for k in sorted(set(qs_counts) | set(pop_qs)):
        v = _pct(qs_counts.get(k, 0), len(sampled_indices))
        p = _pct(pop_qs.get(k, 0), len(rows))
        print(f"    {k:<12} val={v:>5.2f}  pop={p:>5.2f}  delta={v - p:+.2f}")
    print(f"  body_complete: val={_pct(body_complete, len(sampled_indices)):.2f}  "
          f"pop={_pct(pop_body_complete, len(rows)):.2f}")
    print("  topic_primary (top 6):")
    for k, _ in pop_topic.most_common(6):
        v = _pct(topic_counts.get(k, 0), len(sampled_indices))
        p = _pct(pop_topic.get(k, 0), len(rows))
        print(f"    {k:<24} val={v:>5.2f}  pop={p:>5.2f}  delta={v - p:+.2f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
