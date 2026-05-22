# Sprint 3 Experiment Log

Running record of every fine-tuning experiment in Sprint 3, the hypothesis
behind each, and what we learned. Updated as we go.

**Baselines for comparison** (from `results/check2_*.json`, full 65-query
graded eval, semantic-only retrieval against the full 14k corpus):

| Model | nDCG@5 | MRR | nDCG@10 | COI nDCG@5 |
|---|---:|---:|---:|---:|
| OpenAI text-embedding-3-small (production) | 0.263 | 0.494 | — | — |
| BGE-large-en-v1.5 | 0.282 | 0.495 | — | — |
| **Snowflake-arctic-l-v2 (fine-tune target)** | **0.296** | **0.522** | 0.266 | **0.106** |

**SPEC success criteria**:
- Threshold (must hit): semantic-only nDCG@5 ≥ 0.33
- Goal: ≥ 0.36 (matches BM25-only)
- Stretch: hybrid nDCG@5 ≥ 0.42, MRR ≥ 0.72
- Subgroup gate: COI nDCG@5 ≥ 0.20

---

## Headline results

All runs scored on the 65-query graded eval against the full 14,094-doc
corpus, using the same scoring code as Check 2 baselines
(`scripts/_eval_lib.py`, verified bit-identical 2026-05-21).

| Run | Loss | LR | Steps/Ep | nDCG@5 | MRR | nDCG@10 | COI@5 | Δ vs Snowflake |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Snowflake base (untuned) | — | — | — | **0.296** | 0.522 | 0.266 | 0.106 | — |
| s3-a1-mnrl-qa | MNRL | 2e-5 | 642/1 | 0.200 | 0.404 | 0.174 | 0.085 | **−0.097** |
| s3-a2-cached-qa | CachedMNRL bs=128 | 2e-5 | 81/1 | 0.184 | 0.381 | 0.160 | 0.080 | **−0.112** |
| s3-a3-mnrl-hn-qa | MNRL + 1 hard neg | 2e-5 | 642/1 | 0.085 | 0.198 | 0.078 | 0.050 | **−0.212** |
| **BGE-large base (untuned)** | — | — | — | **0.282** | 0.495 | — | 0.090 | — |
| s3-c1-bge-mnrl-qa | MNRL | 2e-5 | 642/1 | 0.181 | 0.396 | 0.158 | 0.099 | **−0.101** (vs BGE) |

---

## Setup

- **Base model**: `Snowflake/snowflake-arctic-embed-l-v2.0` unless noted
- **Training data**: `data/training/pairs.jsonl` (10,806 rows, gitignored).
  Held-out: 624 eval-referenced opinions + 543 val-slice opinions.
  Effective training pool: 10,263 rows for `pos_qa_text` (all rows have it).
- **Hard negatives**: `data/training/hard_negatives.jsonl` (95,665 records,
  BM25 top-5 + same-statute top-5, deduped). 100% coverage, 80% with 10 negs.
- **Query prefix**: `query: ` (Snowflake convention; queries only).
- **Tokenization**: `max_seq_length=512`, bf16 precision.
- **Seed**: 20260521.
- **In-loop val**: 543 val opinions as both queries and candidates — found to
  be **saturated** for all checkpoints (accuracy@1 ≈ 0.998 even on the base
  model). Useless as a comparison signal; we use end-of-run 65-query eval.
- **Hardware**: Strix Halo iGPU (Navi31, 93.75 GB VRAM). ROCm 6.2 wheels.

Run artifacts live under `experiments/<run_id>/` (configs + metrics, committed)
and `data/models/<run_id>/` (weights, gitignored).

---

## Experiments

### s3-a1-mnrl-qa — plain MNRL on pos_qa_text  *(2026-05-21)*

**Hypothesis**: The simplest contrastive loss — in-batch random negatives —
establishes a floor for the bake-off. Standard sentence-transformers recipe.

**Config**: `MultipleNegativesRankingLoss`, batch_size=16, 1 epoch (642
steps), LR=2e-5, warmup_ratio=0.10, `pos_qa_text` as positive.

**Result**: nDCG@5 = **0.200** (−0.097 vs baseline), MRR = 0.404, COI = 0.085.

**Observations**:
- Training loss collapsed from ~1.4 → ~3e-6 within ~600 steps — model
  found a near-zero in-batch loss.
- Lobbying went *up* (0.492 vs 0.660 base; still down) but COI fell.
- Per-topic, every topic regressed.

**Interpretation**: Classic MNRL collapse on a strong pretrained model.
In-batch random negatives are too easy in a topic-imbalanced corpus
(COI = 56% of training) — the model can separate anchor-positive from
random batch-mates without learning the within-topic distinctions that
matter at inference time.

**Wall time**: 43 min train + 9 min score.

---

### s3-a2-cached-qa — CachedMNRL with global batch=128  *(2026-05-21)*

**Hypothesis**: 128 in-batch negatives (vs 15 in s3-a1) should give the
model a finer-grained discrimination signal, reducing the chance of finding
a trivial separation.

**Config**: `CachedMultipleNegativesRankingLoss`, global batch=128,
mini_batch=8, 1 epoch (81 optimizer steps), LR=2e-5, warmup_ratio=0.10.

**Result**: nDCG@5 = **0.184** (−0.112 vs baseline) — *worse* than s3-a1.

**Observations**:
- 8x more in-batch negatives → strictly worse result.
- Only 81 optimizer steps (vs 642) due to the larger global batch.

**Interpretation**: Random in-batch negatives at any batch size are the
wrong signal for this corpus. More negatives → stronger "push away from
random docs" signal → faster collapse of the cosine geometry the base
model already had. The fact that LESS frequent optimizer steps still
hurt suggests it's not just an "over-training" issue.

**Wall time**: 73 min train + 9 min score.

---

### s3-a3-mnrl-hn-qa — MNRL with 1 mined hard negative  *(2026-05-21)*

**Hypothesis**: Explicit hard negatives — BM25 top-K or same-statute
opinions — should force the model to learn the within-domain distinctions
that random negatives don't.

**Config**: `MultipleNegativesRankingLoss` with 1 explicit hard negative
column (first negative from `hard_negatives.jsonl` per row), batch_size=16,
1 epoch (642 steps), LR=2e-5, warmup_ratio=0.10. Negative document text
uses corpus `qa_text` regardless of positive-column choice.

**Result**: nDCG@5 = **0.085** (−0.212 vs baseline) — *dramatically*
worse, and the worst of the three.

**Observations**:
- Hard negatives accelerated the collapse, not corrected it.
- MRR dropped from 0.522 → 0.198 (~60% relative drop).

**Interpretation**: The model was learning to push the *correct* answer
away alongside the hard negative — likely because the hard negative is
*also* a topically relevant opinion that a good encoder would still want
near the query. At LR=2e-5 + 1 epoch, the gradient signal "push neg away
from query" overrides "pull pos toward query" for nearby-in-topic neg/pos
pairs.

**Wall time**: 65 min train + 9 min score.

---

### s3-c1-bge-mnrl-qa — plain MNRL on BGE-large  *(2026-05-21)*

**Hypothesis (H5)**: BGE-large is less collapse-prone than Snowflake-arctic-l-v2
under naive MNRL — smaller model, more headroom, recipe matches BGE's own
training methodology better.

**Config**: identical to s3-a1 in every dimension except `base_model =
BAAI/bge-large-en-v1.5` and `query_prefix = "Represent this sentence for
searching relevant passages: "` (BGE convention).

**Result**: nDCG@5 = **0.181** (−0.101 vs BGE-large baseline of 0.282),
MRR = 0.396, COI = 0.099.

**Observations**:
- Delta from BGE baseline (−0.101) is essentially identical to s3-a1's
  delta from Snowflake baseline (−0.096). The collapse magnitude doesn't
  care which base model we picked.
- One small bright spot: COI subgroup ticked *up* (0.090 → 0.099), the
  only positive movement we've seen in any cell of any tuned model. Could
  be noise at Δ=+0.009, but worth noting.
- Per-topic, lobbying took the biggest hit (0.605 → 0.340).

**Interpretation**: **H5 is refuted.** The training recipe is the
controlling variable; the base model is incidental. Whatever's wrong with
LR=2e-5 + 1-epoch + MNRL + this training data wrecks both bases equally.

This shifts probability mass strongly toward H1/H2 (LR/steps overtrain
the model) and H4 (training/eval distribution mismatch); H3 (hard-neg
selection) was a Stage A explanation that's now downstream of an
unresolved recipe problem.

**Wall time**: 41 min train + 7 min score.

---

## What Stage A taught us

1. **All three losses degrade the base model.** Not "one wins by a small
   margin" — every variant is significantly worse than the untuned model on
   every metric and every topic.
2. **The trend is monotonic with negative difficulty**: random in-batch
   (easiest) → bigger random batch → mined hard negs. Each step "harder"
   makes the model *more* damaged, not less.
3. **The val-slice IR evaluator gave no warning** — it was saturated from
   the first checkpoint.
4. **Training loss collapse is the symptom** — by step ~600 the model has
   zero in-batch loss, which is just the trivial degenerate clustering
   solution at low LR.

So the recipe at LR=2e-5 / 1 epoch / Snowflake-arctic-l-v2 is broken
*before* the loss choice matters. **Stage B (positive-column ablation) is
deferred** until we identify a recipe that produces models that beat
baseline; comparing four flavors of "broken" tells us nothing.

---

## Active hypotheses / candidate next experiments

(Numbered for easy reference; not yet run.)

### H1 — LR is wrecking the pretrained weights

The standard 2e-5 fine-tuning LR is well-suited to weaker base models.
Snowflake-arctic-l-v2 is already saturated on general retrieval and
high-quality. Lower LRs (1e-6 to 1e-5) should let it specialize without
overwriting general structure.

**Proposed runs (Sprint 3.1)**:
- `s3.1-lr5e-6` — Same config as s3-a3 (MNRL+1hn) but LR=5e-6.
- `s3.1-lr1e-6` — Same but LR=1e-6.

### H2 — Even 1 epoch is too long

Training loss hit zero by step 600 of 642. The useful learning may all
happen in the first 100–200 steps; everything after is memorization.

**Proposed approach**: checkpoint every 50 steps and score each on
65-query eval (not val_slice). Stop when 65-query nDCG@5 stops improving.

### H3 — Hard negative selection is too aggressive

`hard_negatives.jsonl` mines BM25 top-K and same-statute opinions. For
COI cases, the "hard negative" might literally be a near-duplicate opinion
on the same legal question — pushing it away from the query is bad
training signal.

**Proposed**: filter hard negatives by some similarity ceiling; or use the
later ranks (BM25 rank 6–10) instead of top-5.

### H4 — Training/eval distribution mismatch

Training questions are FPPC opinion-author paraphrases of conclusions
(e.g. "May a school board member..."), drawn from the corpus's own
structure. Eval queries are real user search phrasings ("conflict of
interest rules for school board members"). The model may be learning to
match *paraphrases* to opinions, which is a different task than
matching *user search intent*.

**Proposed**: LLM-paraphrase a random 1–2k training questions into
search-style queries; mix with the original training questions.

### H5 — Switch base model  *(REFUTED 2026-05-21 by s3-c1)*

~~BGE-large-en-v1.5 is #2 in our baselines (0.282 nDCG@5). Weaker base =
more room for fine-tuning to help, less collapse-prone (less to lose).~~

**Outcome**: s3-c1 (BGE + MNRL, same recipe as s3-a1) regressed by
−0.101 nDCG@5 vs BGE baseline — essentially identical magnitude to s3-a1's
−0.096 vs Snowflake. **The base model is not the controlling variable.**
We did not run c2/c3 since the diagnosis applies to the recipe, not the
loss × base interaction. Probability mass shifts to H1/H2 (recipe is too
aggressive) and H4 (training/eval distribution mismatch).

### H6 — Document the negative result and stop

If H1–H5 don't pan out within a reasonable budget, this becomes the
Sprint 3 outcome: "naive contrastive fine-tuning of strong pretrained
embedding models on this corpus degrades performance." Still a
publishable negative result.

---

## Decision log

- **2026-05-21**: Refactored `check2_open_baselines.py` into
  `scripts/_eval_lib.py`. Verified bit-identical Snowflake JSON output.
- **2026-05-21**: Set max_seq_length=512 + bf16 across Sprint 3 (vs the
  originally-planned 1024 fp32) after Strix Halo feasibility check showed
  the latter only fit batch_size=8 with 80+ GB peak. bf16+512 fits
  batch_size=16 comfortably.
- **2026-05-21**: Val pool = val-slice-only with 543 distractors. Confirmed
  saturated; in future use 65-query eval for in-loop signal.
- **2026-05-21**: **Halted Stage B** after all 3 Stage A runs degraded the
  base. Awaiting user direction on which of H1–H6 to pursue.
- **2026-05-21**: Picked **H5 (switch base model to BGE-large)** as the first
  follow-up. Stage C runs mirror Stage A's loss bake-off but on BGE-large.
  Start with s3-c1 alone; chain c2/c3 only if c1 shows non-broken direction.
- **2026-05-21**: s3-c1 regressed identically to s3-a1 (−0.101 vs BGE base
  vs −0.096 vs Snowflake base). **H5 refuted, c2/c3 not run.** Diagnosis
  shifts to recipe (H1/H2/H4), not base model. Awaiting user call on which
  hypothesis to pursue next.

---

## Files used / produced this sprint

Inputs (built in Sprint 2):
- `data/training/pairs.jsonl` (10,806 rows)
- `data/training/val_slice.jsonl` (543 rows)
- `data/training/hard_negatives.jsonl` (95,665 records)

Scripts:
- `scripts/_eval_lib.py` — shared encode + retrieve + score
- `scripts/feasibility_check.py` — HW batch-size probe
- `scripts/train_sprint3.py` — single-config trainer
- `scripts/score_finetuned.py` — 65-query eval for a saved model

Run artifacts (per run_id under `experiments/`):
- `config.json` — exact training config (committed)
- `train.log`, `score.log` — stdout (committed)
- `val_metrics.jsonl` — in-loop val (saturated; kept for completeness)
- `eval_metrics.json` — final 65-query eval result (committed)
- `build_report.json` — dataset construction counts (committed)
- `model_pointer.txt` — path to `data/models/<run_id>/` (committed)

Weights: `data/models/<run_id>/` (gitignored, ~2 GB each in safetensors).
