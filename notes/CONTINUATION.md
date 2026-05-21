# Continuation — Hard-Negative Mining (Sprint 2 final task)

Snapshot of state at session end on 2026-05-21. Sprint 2's last deliverable
is the hard-negative miner; everything else in Sprint 2 is complete.

## Where we are

- **Sprint 0**: ✅ done. Leakage analysis and OpenAI/BGE baselines.
- **Sprint 1**: ✅ done. Seven open-model baselines on Strix Halo.
  Fine-tune target = **`Snowflake/snowflake-arctic-embed-l-v2.0`**
  (1024d, native 8192-token context, `query: ` prefix only).
  Untuned baseline: nDCG@5 = 0.296, MRR = 0.522. See
  `notes/check2_open_baselines_report.md`.
- **Sprint 2**: in progress.
  - ✅ Held-out exclusion (624 opinions referenced by any eval judgment).
  - ✅ `data/training/pairs.jsonl` — 10,806 rows. 4 positive-doc columns:
    `pos_qa_text`, `pos_body`, `pos_qa_plus_body`, `pos_full_text`.
    Schema documented in `notes/training_data_design.md`.
  - ✅ `data/training/val_slice.jsonl` — 543 rows (5%), year-stratified,
    seed = `20260521`.
  - ⏸️ **`data/training/hard_negatives.jsonl` — to be built next.**

## What's pending: hard-negative mining

Implement `scripts/mine_hard_negatives.py` to produce one record per
training-pool opinion (everything in `pairs.jsonl` minus `val_slice.jsonl`
IDs), listing 5–10 hard-negative opinion IDs per positive.

### Two negative sources to combine

1. **BM25 top-k** against the corpus, using the *training question* text as
   the query. Reuse the search-lab BM25 engine at
   `/home/nick/Projects/fppc-opinions-search-lab/src/engines/bm25_full_text.py`
   — class `BM25FullText`, method `.search(query, top_k)`. This gives
   negatives that look like real production retrieval candidates.

2. **Same-statute different opinion** — any other corpus opinion whose
   `citations.government_code` set overlaps with the positive's. These are
   the hardest legal distractors (same statute, different facts/outcome).

### Exclusions from negative candidate pools

For each positive opinion, the negative pool MUST exclude:

- The positive opinion itself.
- All 624 held-out opinions (use the same `load_held_out()` logic as
  `scripts/build_training_pairs.py`; the function derives the set directly
  from `/home/nick/Projects/fppc-opinions-eval/eval/dataset.json`).
- All val-slice opinion IDs (read from `data/training/val_slice.jsonl`).

The first two exclusions are correctness-critical (eval leakage). The
third prevents the trainer from seeing val opinions as negatives — mild
leakage, but worth avoiding.

### Output schema (proposed)

One JSON object per line, keyed by `opinion_id`:

```json
{
  "opinion_id": "A-19-008",
  "negatives": [
    {"opinion_id": "16-117", "source": "bm25", "rank": 1, "score": 12.34},
    {"opinion_id": "11-002", "source": "same_statute", "shared_codes": ["1090", "87100"]},
    ...
  ]
}
```

Target ~5 BM25 + ~3-5 same-statute, deduplicate by `opinion_id`. If the
same opinion appears in both sources, keep both source markers in a single
record (e.g., `"source": "bm25+same_statute"`).

### Practical notes

- **Cost**: BM25 over 14k docs is fast; the bottleneck will be the ~10k
  queries (one per training-pool opinion). Reuse a single `BM25FullText`
  instance and call `.search()` in a loop. Expect a few minutes total.
- **Cache the same-statute graph**: build a `Dict[gov_code, set[opinion_id]]`
  once from the corpus, then look up neighbors per positive. Don't walk the
  corpus per query.
- **No GPU needed** for this script. Don't bother with
  `HSA_OVERRIDE_GFX_VERSION` — BM25 is CPU-only.
- **Determinism**: BM25 ranking is deterministic given fixed index;
  same-statute ordering should be sorted for reproducibility.

### Possible gotchas

- The search-lab BM25 engine is built around `SearchEngine` protocol from
  `src/engines/base.py`. May need a minimal corpus-loader shim to initialize
  it. Inspect `src/engines/bm25_full_text.py` `__init__` first to see how
  it loads the corpus.
- The search-lab repo has its own `.venv/` and its imports might collide
  with ours. Safer approach: `sys.path.insert(0, "/home/nick/Projects/fppc-opinions-search-lab")`
  before importing, similar to what `scripts/check2_open_baselines.py`
  does for the eval harness.

## Hardware reminder

Strix Halo (Framework Desktop). For any GPU work (not needed for the
hard-negative miner specifically, but for everything downstream): prepend
`HSA_OVERRIDE_GFX_VERSION=11.0.0` to GPU commands. See
`memory/strix-halo-rocm.md` for the full explanation.

## Files to reference quickly

- `scripts/build_training_pairs.py` — pair builder; reference for the
  held-out loader and pair-file schema.
- `scripts/build_val_slice.py` — val-slice carver; reference for reading
  the JSONL and stratified sampling.
- `notes/training_data_design.md` — Sprint 2 design decisions, schema,
  hard-negative-section specifies the constraints listed above.
- `data/training/pairs.jsonl` — input for the miner (gitignored; regenerate
  with `scripts/build_training_pairs.py` if missing).
- `data/training/val_slice.jsonl` — gives the val IDs to exclude.
- `/home/nick/Projects/fppc-opinions-search-lab/src/engines/bm25_full_text.py`
  — BM25 implementation to reuse.

## Once hard-negative mining is done

- Update `notes/training_data_design.md` with the actual coverage stats
  (negatives per positive distribution, BM25 vs same-statute proportions,
  topic distribution of negatives).
- Mark Sprint 2 complete in `IMPLEMENTATION_PLAN.md`.
- Review Sprint 3 plan in light of what we learned in Sprint 2 (any
  surprises in the pair distribution that should shape the pilot ablations?).
