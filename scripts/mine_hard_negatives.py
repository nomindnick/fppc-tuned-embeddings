"""
Mine hard negatives for the Sprint 2 training pool.

For each opinion in data/training/pairs.jsonl that is NOT in val_slice.jsonl,
emit a list of 5-10 hard-negative opinion IDs combined from two sources:

  1. BM25 top-k over content.full_text, querying with the training question.
     Reuses the search-lab's BM25FullText engine so negatives mirror the
     production retrieval distribution.
  2. Same-statute neighbours — other corpus opinions whose government_code set
     overlaps with the positive's. These are the hardest legal distractors
     (same statute, different facts/outcome). Ranked by shared-code count.

Exclusions (applied to every negative candidate pool):
  - The positive opinion itself.
  - The 624 eval-referenced opinions (held-out).
  - The val-slice opinion IDs.

Output:
  data/training/hard_negatives.jsonl   — one line per training opinion
  data/training/hard_negatives_coverage.json  — summary stats

Schema (per JSONL line):
  {
    "opinion_id": "<positive id>",
    "negatives": [
      {"opinion_id": "...", "source": "bm25",          "rank": 1, "score": 12.34},
      {"opinion_id": "...", "source": "same_statute",  "shared_codes": ["1090", "87100"]},
      {"opinion_id": "...", "source": "bm25+same_statute",
       "rank": 3, "score": 8.21, "shared_codes": ["1090"]},
      ...
    ]
  }

The script is CPU-only (no GPU needed). The cached BM25 index at
../fppc-opinions-search-lab/indexes/BM25FullText_index.pkl should already exist
from prior search-lab runs; if missing, the engine rebuilds it on first call
(~2 min).
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

# Reuse the search-lab BM25 engine. It expects to be importable from its repo
# root (its modules import `from src.interface import SearchEngine`).
SEARCH_LAB_ROOT = "/home/nick/Projects/fppc-opinions-search-lab"
sys.path.insert(0, SEARCH_LAB_ROOT)

# These imports depend on SEARCH_LAB_ROOT being on sys.path first.
from src.engines.bm25_full_text import BM25FullText, tokenize  # noqa: E402

EVAL_PATH = "/home/nick/Projects/fppc-opinions-eval/eval/dataset.json"
CORPUS_DIR = "/home/nick/Projects/fppc-opinions-corpus/data/extracted"
TRAIN_DIR = "/home/nick/Projects/fppc-tuned-embeddings/data/training"
PAIRS_PATH = os.path.join(TRAIN_DIR, "pairs.jsonl")
VAL_PATH = os.path.join(TRAIN_DIR, "val_slice.jsonl")
OUT_PATH = os.path.join(TRAIN_DIR, "hard_negatives.jsonl")
OUT_COVERAGE = os.path.join(TRAIN_DIR, "hard_negatives_coverage.json")

# How many candidates to over-fetch per source before exclusions, and how many
# to ultimately keep per source. Over-fetch gives headroom for exclusions
# (the positive, held-out, val-slice) without falling below the per-source target.
BM25_OVERFETCH = 30
BM25_TARGET = 5
STATUTE_TARGET = 5

# Strip parenthetical subsections so "87103(a)" and "87103(b)(1)" both
# normalize to "87103". This makes "same-statute" mean "same base statute,"
# which is the level lawyers actually reason at.
_PAREN_RE = re.compile(r"\([^)]*\)")


def base_code(code: str) -> str:
    return _PAREN_RE.sub("", code).strip()


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def load_held_out() -> set[str]:
    """Same logic as build_training_pairs.load_held_out."""
    with open(EVAL_PATH) as f:
        eval_data = json.load(f)
    held: set[str] = set()
    for q in eval_data["queries"]:
        for j in q.get("relevance_judgments", []):
            held.add(j["opinion_id"])
    return held


def load_val_ids() -> set[str]:
    ids: set[str] = set()
    with open(VAL_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids.add(json.loads(line)["opinion_id"])
    return ids


def load_pairs() -> list[dict]:
    rows: list[dict] = []
    with open(PAIRS_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_statute_index() -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    """Walk the corpus once. Return (by_code, opinion_to_codes)."""
    by_code: dict[str, set[str]] = defaultdict(set)
    opinion_to_codes: dict[str, set[str]] = {}
    seen: set[str] = set()
    for year in sorted(os.listdir(CORPUS_DIR)):
        ydir = os.path.join(CORPUS_DIR, year)
        if not os.path.isdir(ydir):
            continue
        for fname in sorted(os.listdir(ydir)):
            if not fname.endswith(".json"):
                continue
            with open(os.path.join(ydir, fname)) as f:
                op = json.load(f)
            oid = op.get("id")
            if not oid or oid in seen:
                continue
            seen.add(oid)
            codes = (op.get("citations") or {}).get("government_code") or []
            bases = {base_code(c) for c in codes if c}
            bases.discard("")
            opinion_to_codes[oid] = bases
            for b in bases:
                by_code[b].add(oid)
    return by_code, opinion_to_codes


# ---------------------------------------------------------------------------
# Miner
# ---------------------------------------------------------------------------

@dataclass
class NegRecord:
    opinion_id: str
    source: str
    rank: int | None = None
    score: float | None = None
    shared_codes: list[str] | None = None

    def to_dict(self) -> dict:
        d: dict = {"opinion_id": self.opinion_id, "source": self.source}
        if self.rank is not None:
            d["rank"] = self.rank
        if self.score is not None:
            d["score"] = round(self.score, 4)
        if self.shared_codes is not None:
            d["shared_codes"] = self.shared_codes
        return d


def bm25_negatives(
    engine: BM25FullText,
    query: str,
    positive_id: str,
    excluded: set[str],
    target: int,
    overfetch: int,
) -> list[NegRecord]:
    tokens = tokenize(query)
    if not tokens:
        return []
    scores = engine._bm25.get_scores(tokens)
    # argsort ascending → reverse for descending
    top_ix = scores.argsort()[::-1][:overfetch]
    out: list[NegRecord] = []
    rank = 0
    for idx in top_ix:
        s = float(scores[idx])
        if s <= 0:
            break
        oid = engine._opinion_ids[idx]
        if oid == positive_id or oid in excluded:
            continue
        rank += 1
        out.append(NegRecord(opinion_id=oid, source="bm25", rank=rank, score=s))
        if len(out) >= target:
            break
    return out


def statute_negatives(
    positive_id: str,
    positive_codes: set[str],
    by_code: dict[str, set[str]],
    opinion_to_codes: dict[str, set[str]],
    excluded: set[str],
    target: int,
) -> list[NegRecord]:
    if not positive_codes:
        return []
    candidate_overlap: dict[str, set[str]] = {}
    for c in positive_codes:
        for other in by_code.get(c, ()):
            if other == positive_id or other in excluded:
                continue
            candidate_overlap.setdefault(other, set()).add(c)
    if not candidate_overlap:
        return []
    # Rank by (shared count desc, then opinion_id asc) for deterministic order.
    ranked = sorted(
        candidate_overlap.items(),
        key=lambda kv: (-len(kv[1]), kv[0]),
    )[:target]
    return [
        NegRecord(
            opinion_id=oid,
            source="same_statute",
            shared_codes=sorted(shared),
        )
        for oid, shared in ranked
    ]


def merge(bm25: list[NegRecord], statute: list[NegRecord]) -> list[NegRecord]:
    """Dedup by opinion_id. If both sources hit the same id, merge into one
    record with source 'bm25+same_statute' carrying both metadata fields."""
    by_id: dict[str, NegRecord] = {}
    order: list[str] = []
    for r in bm25:
        if r.opinion_id not in by_id:
            order.append(r.opinion_id)
        by_id[r.opinion_id] = r
    for r in statute:
        if r.opinion_id in by_id:
            existing = by_id[r.opinion_id]
            existing.source = "bm25+same_statute"
            existing.shared_codes = r.shared_codes
        else:
            order.append(r.opinion_id)
            by_id[r.opinion_id] = r
    return [by_id[oid] for oid in order]


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    if not os.path.exists(PAIRS_PATH):
        print(f"ERROR: {PAIRS_PATH} not found; run scripts/build_training_pairs.py first.",
              file=sys.stderr)
        return 1
    if not os.path.exists(VAL_PATH):
        print(f"ERROR: {VAL_PATH} not found; run scripts/build_val_slice.py first.",
              file=sys.stderr)
        return 1

    print("[setup] loading held-out and val IDs...")
    held = load_held_out()
    val_ids = load_val_ids()
    excluded = held | val_ids
    print(f"  held-out: {len(held)}  val-slice: {len(val_ids)}  union: {len(excluded)}")

    print("[setup] loading training pairs...")
    pairs = load_pairs()
    print(f"  loaded {len(pairs)} pairs")
    # Train pool = pairs minus val-slice
    train_pool = [p for p in pairs if p["opinion_id"] not in val_ids]
    print(f"  train pool: {len(train_pool)} (= pairs minus val-slice)")

    print("[setup] building statute index from corpus...")
    t0 = time.time()
    by_code, opinion_to_codes = load_statute_index()
    print(f"  indexed {len(opinion_to_codes)} opinions across {len(by_code)} base codes "
          f"in {time.time() - t0:.1f}s")

    print("[setup] loading BM25 engine (cached index)...")
    t0 = time.time()
    engine = BM25FullText()
    print(f"  engine ready in {time.time() - t0:.1f}s")

    # Stats accumulators
    n_negs_per_row: list[int] = []
    source_class_counts = Counter()  # 'bm25_only', 'statute_only', 'overlap', 'empty'
    bm25_kept = 0
    statute_kept = 0
    overlap_kept = 0
    no_bm25_rows = 0
    no_statute_rows = 0
    empty_rows = 0
    bm25_score_samples: list[float] = []
    statute_share_samples: list[int] = []

    print(f"[mine] writing → {OUT_PATH}")
    os.makedirs(TRAIN_DIR, exist_ok=True)
    t0 = time.time()
    last_log = t0
    with open(OUT_PATH, "w") as out_f:
        for i, row in enumerate(train_pool):
            pos_id = row["opinion_id"]
            question = row["question"]
            pos_codes = opinion_to_codes.get(pos_id, set())

            bm25 = bm25_negatives(
                engine, question, pos_id, excluded,
                target=BM25_TARGET, overfetch=BM25_OVERFETCH,
            )
            statute = statute_negatives(
                pos_id, pos_codes, by_code, opinion_to_codes,
                excluded, target=STATUTE_TARGET,
            )
            merged = merge(bm25, statute)

            out_f.write(json.dumps({
                "opinion_id": pos_id,
                "negatives": [r.to_dict() for r in merged],
            }) + "\n")

            # Track stats
            n_negs_per_row.append(len(merged))
            if not bm25:
                no_bm25_rows += 1
            if not statute:
                no_statute_rows += 1
            if not merged:
                empty_rows += 1
                source_class_counts["empty"] += 1
            else:
                has_bm25 = any(r.source.startswith("bm25") for r in merged)
                has_stat = any("same_statute" in r.source for r in merged)
                if has_bm25 and has_stat:
                    source_class_counts["mixed"] += 1
                elif has_bm25:
                    source_class_counts["bm25_only"] += 1
                else:
                    source_class_counts["statute_only"] += 1
            for r in merged:
                if r.source == "bm25":
                    bm25_kept += 1
                    if r.score is not None:
                        bm25_score_samples.append(r.score)
                elif r.source == "same_statute":
                    statute_kept += 1
                    if r.shared_codes is not None:
                        statute_share_samples.append(len(r.shared_codes))
                else:  # bm25+same_statute
                    overlap_kept += 1
                    if r.score is not None:
                        bm25_score_samples.append(r.score)
                    if r.shared_codes is not None:
                        statute_share_samples.append(len(r.shared_codes))

            now = time.time()
            if now - last_log > 10 or i + 1 == len(train_pool):
                elapsed = now - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(train_pool) - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1:>5}/{len(train_pool)}] {rate:5.1f} rows/s  "
                      f"elapsed={elapsed:5.0f}s  eta={eta:5.0f}s")
                last_log = now

    def _pct(num: int, denom: int) -> float:
        return 0.0 if denom == 0 else round(100 * num / denom, 2)

    def _stats(xs: list[float], name: str) -> dict:
        if not xs:
            return {"name": name, "n": 0}
        xs_sorted = sorted(xs)
        return {
            "name": name,
            "n": len(xs_sorted),
            "p10": xs_sorted[len(xs_sorted) // 10],
            "p50": xs_sorted[len(xs_sorted) // 2],
            "p90": xs_sorted[int(len(xs_sorted) * 0.9)],
            "p99": xs_sorted[int(len(xs_sorted) * 0.99)],
            "min": xs_sorted[0],
            "max": xs_sorted[-1],
            "mean": round(sum(xs_sorted) / len(xs_sorted), 4),
        }

    negs_counter = Counter(n_negs_per_row)
    report = {
        "n_train_pool": len(train_pool),
        "n_with_at_least_one_neg": len(train_pool) - empty_rows,
        "n_with_zero_negs": empty_rows,
        "n_rows_with_no_bm25": no_bm25_rows,
        "n_rows_with_no_statute": no_statute_rows,
        "source_classification": dict(source_class_counts),
        "kept_negative_records": {
            "bm25_only": bm25_kept,
            "same_statute_only": statute_kept,
            "bm25_and_same_statute": overlap_kept,
            "total": bm25_kept + statute_kept + overlap_kept,
        },
        "negatives_per_row_histogram": {
            str(k): negs_counter[k] for k in sorted(negs_counter)
        },
        "negatives_per_row_mean": round(
            sum(n_negs_per_row) / max(1, len(n_negs_per_row)), 3
        ),
        "bm25_score_stats": _stats(bm25_score_samples, "bm25_score"),
        "shared_codes_count_stats": _stats(
            [float(x) for x in statute_share_samples], "shared_codes_count"
        ),
        "config": {
            "bm25_overfetch": BM25_OVERFETCH,
            "bm25_target": BM25_TARGET,
            "statute_target": STATUTE_TARGET,
            "excludes_held_out": True,
            "excludes_val_slice": True,
        },
    }
    with open(OUT_COVERAGE, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print(f"Hard-negatives file: {OUT_PATH}")
    print(f"Coverage report:     {OUT_COVERAGE}")
    print()
    print(f"Train pool:                 {len(train_pool)}")
    print(f"  with ≥1 negative:         {len(train_pool) - empty_rows} "
          f"({_pct(len(train_pool) - empty_rows, len(train_pool))}%)")
    print(f"  with 0 negatives:         {empty_rows}")
    print(f"  no BM25 hits:             {no_bm25_rows}")
    print(f"  no same-statute hits:     {no_statute_rows}")
    print(f"Source breakdown (rows):    {dict(source_class_counts)}")
    print(f"Negative records kept:")
    print(f"  bm25 only:                {bm25_kept}")
    print(f"  same_statute only:        {statute_kept}")
    print(f"  bm25+same_statute:        {overlap_kept}")
    print(f"  total:                    {bm25_kept + statute_kept + overlap_kept}")
    print(f"Mean negatives/row:         {report['negatives_per_row_mean']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
