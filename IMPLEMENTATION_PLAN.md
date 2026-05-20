# Implementation Plan (draft)

> **Status:** Draft sprints. Sprint 0 ("Pre-planning checks") is partially
> complete. Sprint 1 finalizes Check 2 on the Strix Halo. Later sprints are
> outlined but should be reviewed once we have the full baseline picture.

---

## Sprint 0 — Pre-planning checks ✅ (mostly done)

**Goal**: ground the project plan in measured numbers before designing
training data.

| Step | Status | Artifact |
|---|---|---|
| Check 1: Eval-set leakage analysis | ✅ done | `notes/check1_leakage_report.md`, `results/check1_leakage.json` |
| Check 2a: Reproduce OpenAI baseline on this scoring path | ✅ done | `results/check2_openai_baseline.json` |
| Check 2b: bge-base-en-v1.5 baseline | ✅ done | `results/check2_bge-base-en-v1.5.json` |
| Check 2c: nomic-embed-text-v1.5 baseline | ⏸️ pending — paused for hardware switch | `results/check2_nomic-embed-text-v1.5.json` (TBD) |
| Check 2d: bge-large-en-v1.5 baseline (optional scale check) | ⏸️ pending | `results/check2_bge-large-en-v1.5.json` (TBD) |

**Headline so far**:
- BGE-base ≈ OpenAI on this corpus (nDCG@5 0.268 vs 0.263). Open-model
  parity confirmed for at least one fine-tunable option.
- Leakage is real but contained — hold out 596 eval-referenced opinions.

---

## Sprint 1 — Finalize Check 2 on Strix Halo

**Goal**: complete the open-model baseline picture so we can pick a
fine-tune target with confidence.

1. Set up the project on Strix Halo (see `notes/CONTINUATION.md`).
2. Run nomic-v1.5 baseline at max_seq_length=1024. Should be ~5–10× faster
   than on laptop CPU.
3. Run bge-large-en-v1.5 baseline (scale-vs-base check).
4. Write `notes/check2_open_baselines_report.md` summarizing all four
   numbers (OpenAI, BGE-base, BGE-large, Nomic-v1.5) with per-topic
   breakdowns and a final recommendation on the fine-tune target.
5. Finalize SPEC.md success criteria with concrete numbers.

---

## Sprint 2 — Training data construction

**Goal**: produce a versioned training dataset and a held-out validation
slice; document every decision.

1. Verify the held-out 596 opinions are excluded from any training pool
   we construct.
2. Decide: include `question_synthetic` pairs? Run a small ablation on a
   tiny pilot fine-tune to test.
3. Build the base `(question, opinion_qa_text)` positive set.
4. Mine hard negatives:
   - BM25 top-k that aren't in any gold judgment for that "query".
   - Same-statute-different-opinion (using `citations.government_code`).
5. (Optional) Augment queries with LLM paraphrases in multiple registers
   (formal sentence, keyword bag, fact pattern). Gate by an ablation.
6. Carve a validation slice (~5% sampled stratified by year) for in-loop
   feedback during training. Eval set remains untouched until the end.

Artifacts:
- `data/training/pairs.jsonl`
- `data/training/hard_negatives.jsonl`
- `data/training/val_slice.jsonl`
- `notes/training_data_design.md`

---

## Sprint 3 — Initial fine-tuning sweep

**Goal**: small grid; identify which base × loss config is worth pushing on.

- Base models: top 1–2 picks from Sprint 1.
- Losses: `MultipleNegativesRankingLoss`, `CachedMultipleNegativesRankingLoss`
  (large in-batch negatives), `TripletLoss` with mined hard negatives.
- Short runs (1–3 epochs), default hyperparams.
- Score every checkpoint on validation + final on eval.
- Track runs in `experiments/<run_id>/`.

---

## Sprint 4 — Winner tuning

- Hyperparameter sweep on the Sprint 3 winner (LR, warmup, batch, epochs).
- Loss-wrapper variations (Matryoshka, hard-negative weight).
- Best checkpoint scored on the full eval.

---

## Sprint 5 — Hybrid integration & honest assessment

- Drop the winning model into Experiment 009's CitationScoreFusion code.
- Re-run the production-equivalent eval.
- Compare:
  - Semantic-only (tuned) vs semantic-only (OpenAI/BGE baseline).
  - Hybrid (tuned semantic) vs hybrid (baseline semantic, the published 0.387).
- Per-topic breakdown, especially `conflicts_of_interest`.
- Decide whether to recommend handoff to the app repo.

Artifact: `notes/final_assessment.md`.
