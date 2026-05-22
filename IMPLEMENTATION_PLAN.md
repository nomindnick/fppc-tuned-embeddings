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

## Sprint 3 — Initial fine-tuning sweep ✅ done  *(2026-05-22 — see [experiment log](notes/sprint3_experiment_log.md))*

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

**Winning recipe (s3-d3)**: Snowflake-arctic-l-v2 + MNRL in-batch +
`pos_conclusion_only` + LR=1e-6 + bs=16 + max_seq=512 + bf16 + 1 epoch.

**Lever sweep after d3**: explored four candidate improvements (more
epochs, LoRA, paraphrase augmentation, hard-negatives-at-low-LR). None
produced a model that beats d3 on headline nDCG@5 + MRR. Key findings:

| Lever | Verdict |
|---|---|
| More epochs (s3-e1, 3 ep) | Refuted — mild overtraining |
| Hard negatives at LR=1e-6 (s3-g1) | Catastrophically refuted — false-negative pollution (nDCG@5 0.059) |
| Paraphrase augmentation (s3-h1, 1k×2 via gemma4:e4b) | Tied within noise; keyword target slice did not move |
| LoRA r=16 Q/V LR=1e-5 (s3-f2) | Different model shape, not better — gifts/lobbying preserved, fact_pattern/campaign_finance lost. Ensemble candidate. |

**SPEC threshold (≥ 0.33 nDCG@5) not hit**; Sprint 4 levers worth
investment are now: (a) **ensemble d3 + f2**, (b) **cross-encoder-filtered
hard negatives**, (c) **MarginMSE distillation from a reranker**,
(d) **larger-scale paraphrase augmentation with a bigger LLM**. The
full record lives in `notes/sprint3_experiment_log.md`.

Original plan (kept for reference):

- Base models: top 1–2 picks from Sprint 1.
- Losses: `MultipleNegativesRankingLoss`, `CachedMultipleNegativesRankingLoss`
  (large in-batch negatives), `TripletLoss` with mined hard negatives.
- Short runs (1–3 epochs), default hyperparams.
- Score every checkpoint on validation + final on eval.
- Track runs in `experiments/<run_id>/`.

---

## Sprint 4 — Winner tuning  *(revised post Sprint 3 lever sweep)*

Original plan was a hyperparameter sweep on the Sprint 3 winner.
The Sprint 3 lever sweep changed that picture: the obvious knobs (LR,
epochs, LoRA, hard negatives) are all either already-optimized or
structurally toxic. The remaining-value levers are larger structural
changes, not hyperparameter tuning:

1. **Ensemble d3 + f2 (low cost, likely high value)**. Two valid
   fine-tunes with different topic strengths. Score fusion or per-query
   model selection on the 65-query eval.
2. **Cross-encoder-filtered hard-negative mining**. Use a BGE reranker
   (or similar) to drop "negatives" that are actually relevant before
   training. Likely unlocks hard negatives without the s3-g1 disaster.
3. **MarginMSE distillation**. Continuous relevance scores from a
   reranker teacher; replaces MNRL's binary objective. The clean fix
   for the false-negative-pollution mechanism.
4. **Larger-scale paraphrase augmentation**. All 9k training rows × 2-3
   paraphrases via a 26B+ LLM (vs s3-h1's 1k × 2 via gemma4:e4b). Gives
   H4 a fair test at the right scale.

Original hyperparameter levers (LR, warmup, batch, epochs, Matryoshka)
are deprioritized — Sprint 3 found d3's recipe to be at or near the
optimum for full FT on this corpus.

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
