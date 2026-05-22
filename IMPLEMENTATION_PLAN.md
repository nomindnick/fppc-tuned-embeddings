# Implementation Plan (draft)

> **Status:** Sprints 0, 1, and 2 complete. Fine-tune target selected
> (Snowflake-arctic-embed-l-v2.0). Sprint 3 (initial fine-tuning sweep) is
> next; later sprints are outlined but should be reviewed against the
> Sprint 2 results before committing.

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

## Sprint 2 — Training data construction ✅ done

**Goal**: produce a versioned training dataset and a held-out validation
slice; document every decision. Design captured in
`notes/training_data_design.md`.

| Step | Status | Artifact |
|---|---|---|
| Held-out exclusion (624 eval-referenced opinions) | ✅ done | enforced inside the pair builder |
| Pair file with 4 positive-doc columns + question_source marker | ✅ done | `data/training/pairs.jsonl` (10,806 rows, gitignored) |
| Validation slice, 5% stratified by year, seed=20260521 | ✅ done | `data/training/val_slice.jsonl` (543 rows, gitignored) |
| Hard-negative mining (BM25 top-k + same-statute) | ✅ done | `data/training/hard_negatives.jsonl` (10,263 rows, 95,665 negs, gitignored) |
| Design doc | ✅ done | `notes/training_data_design.md` |
| Optional query augmentation (LLM paraphrases) | ⏸️ deferred to Sprint 3 ablation | — |

The pair builder uses a strictly conservative held-out (all 624 judged
opinions, including score=0) rather than the 596-positive-only count from
the leakage report. Training distribution is naturally COI-heavy (56.5%) so
no explicit topic oversampling is applied.

Hard-negative coverage: 100% of the training pool gets at least one
negative; 80% get the full 10 (5 BM25 + 5 same-statute, deduplicated).
The 1,150 rows with no statute hits (~11.2%) are opinions whose
`citations.government_code` list is empty — they fall back to BM25-only
with 5 negatives. BM25 and same-statute pools are largely complementary
(only 1.2% of records overlap), so the two sources surface different
distractors as intended.

---

## Sprint 3 — Initial fine-tuning sweep  *(2026-05-22: viable recipe found — see [experiment log](notes/sprint3_experiment_log.md))*

**Goal**: small grid; identify which base × loss config is worth pushing on.

**Outcome**: After two diagnostic detours (Stage A's catastrophic
regressions and the BGE-large control swap), s3-d3 produced the first
fine-tune that beats the base on more metrics than it loses on.

| | Snowflake base | **s3-d3 (winner)** | Δ |
|---|---:|---:|---:|
| nDCG@5 | 0.296 | 0.290 | −0.006 |
| MRR | 0.522 | **0.538** | **+0.016** |
| COI nDCG@5 | 0.106 | **0.124** | **+0.018** |
| natural_language nDCG@5 | 0.322 | **0.330** | **+0.008** |
| fact_pattern nDCG@5 | 0.353 | **0.390** | **+0.037** |
| keyword nDCG@5 | 0.237 | 0.190 | −0.047 |

**Two mechanisms identified and fixed**:
1. **Lexical leakage** — `pos_qa_text` contained the verbatim query;
   model learned identity-matching, not retrieval. Fixed by stripping
   the QUESTION block (`pos_conclusion_only` derived column).
2. **LR too hot** — 2e-5 (sentence-transformers default) over-wrote the
   strong pretrained directions on a saturated 568M model. Fixed by
   dropping to 1e-6.

**Winning recipe**: Snowflake-arctic-l-v2 + MNRL in-batch + `pos_conclusion_only`
+ LR=1e-6 + bs=16 + max_seq=512 + bf16 + 1 epoch. **SPEC threshold (0.33
nDCG@5) not yet hit**; Sprint 4 levers (LoRA, more epochs, paraphrase
augmentation for keyword queries, hard-negative reintroduction at low LR)
remain to push toward it. Stage B (positive-column ablation) is no longer
relevant — `pos_conclusion_only` is the de facto positive choice.

The full record of what was tried, observed numbers, and active hypotheses
lives in `notes/sprint3_experiment_log.md`.

Original plan (kept for reference):

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
