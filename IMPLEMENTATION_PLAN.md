# Implementation Plan (draft)

> **Status:** Sprints 0 and 1 complete. Fine-tune target selected
> (BGE-large-en-v1.5). Sprint 2 (training data construction) is next; later
> sprints are outlined but should be reviewed against the Sprint 2 results
> before committing.

---

## Sprint 0 — Pre-planning checks ✅ done

**Goal**: ground the project plan in measured numbers before designing
training data.

| Step | Status | Artifact |
|---|---|---|
| Check 1: Eval-set leakage analysis | ✅ done | `notes/check1_leakage_report.md`, `results/check1_leakage.json` |
| Check 2a: Reproduce OpenAI baseline on this scoring path | ✅ done | `results/check2_openai_baseline.json` |
| Check 2b: bge-base-en-v1.5 baseline | ✅ done | `results/check2_bge-base-en-v1.5.json` |
| Check 2c: nomic-embed-text-v1.5 baseline | ✅ done | `results/check2_nomic-embed-text-v1.5.json` |
| Check 2d: bge-large-en-v1.5 baseline | ✅ done | `results/check2_bge-large-en-v1.5.json` |

**Headline** (after second-pass benchmark of 3 additional candidates):
- **Snowflake-arctic-embed-l-v2.0 is the best open-weights baseline**
  (nDCG@5 = 0.296, MRR = 0.522). Beats BGE-large (0.282 / 0.495) and OpenAI
  (0.263 / 0.494) on every overall metric. Native 8192-token context
  unlocks full-text positive docs for Sprint 2.
- gte-modernbert-base wins lobbying outright; Qwen3-Embedding-0.6B wins
  keyword queries outright. Both could be Sprint 4 ensembling levers.
- Leakage is real but contained — hold out 596 eval-referenced opinions.
- `conflicts_of_interest` (45% of eval) scores ≈0.09–0.13 nDCG@5 on **every**
  one of 7 models tested. Universally broken; fine-tuning has to attack it
  directly.

---

## Sprint 1 — Finalize Check 2 on Strix Halo ✅ done

**Goal**: complete the open-model baseline picture and pick a fine-tune target.

1. ✅ Set up the project on Strix Halo. PyTorch ROCm 6.2 wheels run on the
   gfx1151 iGPU under `HSA_OVERRIDE_GFX_VERSION=11.0.0` (falls back to
   gfx1100 kernels).
2. ✅ Nomic-v1.5 at max_seq=1024 (~4 min on GPU vs ~2.5h estimated on laptop).
3. ✅ BGE-large baseline (~8 min on GPU).
4. ✅ `notes/check2_open_baselines_report.md` summarizes all four models
   with per-topic and per-type breakdowns.
5. ✅ SPEC.md "Target model" and "Success criteria" sections updated with
   concrete numbers.

**Fine-tune target**: `Snowflake/snowflake-arctic-embed-l-v2.0` (see SPEC for
justification). Updated 2026-05-21 after second-pass benchmark of three
additional candidates (gte-modernbert-base, Qwen3-Embedding-0.6B,
snowflake-arctic-l-v2) flushed out Snowflake as the clear winner.

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
