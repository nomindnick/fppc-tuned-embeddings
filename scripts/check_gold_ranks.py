"""
For specific queries, report where each gold-relevant opinion ranks
in d3's semantic-only ranking over the full 14k corpus.

If gold answers rank within top-100, reranking can plausibly recover
them. If they rank deep (1000+), the bi-encoder architecture isn't
surfacing them as candidates and the fix must be at the semantic level.

Usage:
    python scripts/check_gold_ranks.py q001 q018 q012 q046
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

EVAL = "/home/nick/Projects/fppc-opinions-eval/eval/dataset.json"
TUNED = "/home/nick/Projects/fppc-tuned-embeddings"


def load(p: str) -> dict | list:
    with open(p) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("qids", nargs="+")
    parser.add_argument("--model-tag", default="sprint3_s3-d3-mnrl-conclusion-lr1e6",
                        help="Cache key (matches the npy filename suffix).")
    parser.add_argument("--model-dir", default=None,
                        help="Explicit model dir. Overrides the implied path from "
                             "--model-tag.")
    args = parser.parse_args()

    npy = f"{TUNED}/data/indexes/corpus_embeddings_{args.model_tag}_qa_text.npy"
    ids_path = f"{TUNED}/data/indexes/corpus_embeddings_{args.model_tag}_ids.json"
    doc_vecs = np.load(npy).astype(np.float32)
    with open(ids_path) as f:
        sem_ids = json.load(f)
    id_to_idx = {oid: i for i, oid in enumerate(sem_ids)}

    eval_ds = load(EVAL)["queries"]
    eval_by_id = {q["id"]: q for q in eval_ds}

    # Encode queries
    print(f"Loading model for {args.model_tag}…")
    from sentence_transformers import SentenceTransformer
    if args.model_dir:
        model_dir = args.model_dir
    elif args.model_tag.startswith("sprint3_"):
        model_dir = f"{TUNED}/data/models/{args.model_tag.replace('sprint3_', '')}"
    else:
        model_dir = args.model_tag
    model = SentenceTransformer(model_dir)
    model.max_seq_length = 512

    for qid in args.qids:
        q = eval_by_id[qid]
        text = "query: " + q["text"]
        qv = model.encode([text], normalize_embeddings=True, convert_to_numpy=True,
                          show_progress_bar=False)[0].astype(np.float32)
        scores = doc_vecs @ qv  # cosine similarity (assumes both L2-normalized)
        order = np.argsort(-scores)
        rank_of = {sem_ids[idx]: r + 1 for r, idx in enumerate(order)}

        golds_2 = sorted([g["opinion_id"] for g in q["relevance_judgments"] if g["score"] == 2])
        golds_1 = sorted([g["opinion_id"] for g in q["relevance_judgments"] if g["score"] == 1])

        print(f"\n{'=' * 80}")
        print(f"  {qid} [{q.get('type')}/{q.get('topic')}]: {q['text'][:90]}…")
        print(f"{'=' * 80}")

        print(f"\n  Score-2 gold ({len(golds_2)}):")
        for oid in golds_2:
            r = rank_of.get(oid, "MISSING")
            print(f"    {oid:>14s}  rank = {r}")
        print(f"\n  Score-1 gold ({len(golds_1)}):")
        for oid in golds_1:
            r = rank_of.get(oid, "MISSING")
            print(f"    {oid:>14s}  rank = {r}")

        # Summary
        ranks_2 = [rank_of.get(oid, 99999) for oid in golds_2]
        ranks_1 = [rank_of.get(oid, 99999) for oid in golds_1]
        all_ranks = ranks_2 + ranks_1
        print(f"\n  Best score-2 rank:        {min(ranks_2) if ranks_2 else 'n/a'}")
        print(f"  Best score-1 rank:        {min(ranks_1) if ranks_1 else 'n/a'}")
        print(f"  # gold-2 in top-100:      {sum(1 for r in ranks_2 if r <= 100)} / {len(ranks_2)}")
        print(f"  # gold in top-100 (any):  {sum(1 for r in all_ranks if r <= 100)} / {len(all_ranks)}")
        print(f"  # gold in top-1000 (any): {sum(1 for r in all_ranks if r <= 1000)} / {len(all_ranks)}")


if __name__ == "__main__":
    main()
