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
| s3-d1-mnrl-conclusion-qa | MNRL on `pos_conclusion_only` | 2e-5 | 560/1 | 0.230 | 0.426 | 0.223 | 0.083 | **−0.066** (vs Snowflake) |
| **s3-d2-mnrl-conclusion-lr5e6** | MNRL on `pos_conclusion_only`, LR↓ | **5e-6** | 560/1 | **0.275** | **0.504** | 0.253 | 0.102 | **−0.021** (vs Snowflake) |

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

### s3-d1-mnrl-conclusion-qa — MNRL with the question stripped from the positive  *(2026-05-22)*

**Hypothesis (H7, leakage)**: Stage A's collapse is driven by the verbatim
query appearing as a substring of `pos_qa_text` (`"QUESTION: <query>...
CONCLUSION: ..."`). The training task degenerates to "find the doc that
contains this exact token sequence" — trivial for a 568M model, found in
<200 steps, leaving 400+ more steps to memorize the shortcut and overwrite
pretrained semantic capability. Predicted symptoms if true: val-slice IR
evaluator no longer saturates at 1.0; the catastrophic regressions on
topics the base was strongest at (gifts, lobbying) partially recover.

**Config**: identical to s3-a1 in every dimension except
`positive_column = "pos_conclusion_only"` — a derived column built by
splitting `pos_qa_text` on `"\n\nCONCLUSION:"` and keeping the right
half (the conclusion text only, no QUESTION block). 9,431 of 10,806 pairs
(87.3%) had the delimiter; 8,949 training rows after val-overlap filter
(vs s3-a1's 10,263).

**Result**: nDCG@5 = **0.230** (−0.066 vs Snowflake baseline 0.296),
MRR = 0.426, COI = 0.083.

| Topic | n | Snowflake base | s3-a1 (leaked) | **s3-d1 (clean)** | Δ vs base |
|---|---:|---:|---:|---:|---:|
| conflicts_of_interest | 29 | 0.106 | 0.085 | 0.083 | −0.023 |
| campaign_finance | 14 | 0.397 | 0.257 | 0.301 | −0.096 |
| gifts_honoraria | 7 | 0.638 | 0.308 | **0.490** | −0.148 |
| lobbying | 5 | 0.660 | 0.492 | 0.507 | −0.153 |
| other | 10 | 0.283 | 0.233 | 0.238 | −0.045 |

| Type | n | Snowflake base | s3-a1 (leaked) | **s3-d1 (clean)** |
|---|---:|---:|---:|---:|
| keyword | 26 | 0.237 | 0.168 | 0.173 |
| natural_language | 22 | 0.322 | 0.201 | 0.247 |
| fact_pattern | 17 | 0.353 | 0.248 | 0.297 |

**Val-slice IR evaluator** (the one that was useless in Stage A,
saturated at acc@1=0.998): now reads **acc@1 0.85 → 0.87**, **nDCG@5
0.89 → 0.91** across checkpoints (step 100 → 560). The shortcut is gone;
the evaluator gives real signal again.

**Interpretation — H7 partially confirmed, not the whole story.**

*What changed (consistent with leakage being real):*
- Val-slice acc@1 dropped from 0.998 (s3-a1) to 0.85 (s3-d1). The model
  is no longer identity-matching the question token sequence.
- `gifts_honoraria` recovered from 0.308 → 0.490 (+0.182). The topics
  the base was strongest at — and that s3-a1 destroyed — are partially
  restored.
- Every query type moved upward vs s3-a1: keyword +0.005,
  natural_language +0.046, fact_pattern +0.049.

*What didn't change (consistent with a second mechanism still active):*
- Still −0.066 vs base overall. Removing the lexical shortcut
  recovered ~0.03 of the ~0.10 nDCG@5 gap; ~0.07 remains.
- COI didn't move (0.085 → 0.083). Notably, COI is the topic the base
  is *weakest* at — if catastrophic forgetting were the only mechanism,
  COI would be the least vulnerable.
- Gifts/lobbying still hurt by ~0.15 despite recovering substantially.
  Something is still washing out the strongest pretrained capabilities.

**Diagnosis after s3-d1**: there are (at least) two stacked mechanisms.
Leakage caused the most violent damage (worst on natural-language and
fact-pattern queries, which have no verbatim eval-time overlap to ride
the shortcut on). With leakage removed, what remains looks like
**LR-driven catastrophic forgetting** — a "hot" 2e-5 LR rewrites the
strongest pretrained directions (gifts, lobbying) even when the loss
signal isn't degenerate. The remaining gap is much smaller and topic-
shaped in the way an LR-induced forgetting story predicts.

**Wall time**: 30.6 min train + 8 min score.

---

### s3-d2-mnrl-conclusion-lr5e6 — same as s3-d1 but LR=5e-6  *(2026-05-22)*

**Hypothesis (H1, residual)**: With the lexical-leakage shortcut removed
by s3-d1, the remaining −0.066 nDCG@5 gap is consistent with LR-driven
catastrophic forgetting (gifts/lobbying still hurt by ~0.15, COI flat).
Dropping LR from 2e-5 to 5e-6 should let the model adapt to FPPC retrieval
without overwriting its strongest pretrained directions.

**Config**: identical to s3-d1 in every dimension except
`learning_rate = 5e-6` (vs 2e-5). Same data, same loss, same epochs, same
prefix, same seed.

**Result**: nDCG@5 = **0.275** (−0.021 vs Snowflake base 0.296),
MRR = 0.504, COI = 0.102.

| Topic | n | Base | s3-d1 | **s3-d2** | Δ vs base |
|---|---:|---:|---:|---:|---:|
| conflicts_of_interest | 29 | 0.106 | 0.083 | 0.102 | −0.004 |
| campaign_finance | 14 | 0.397 | 0.301 | 0.387 | −0.010 |
| gifts_honoraria | 7 | 0.638 | 0.490 | 0.554 | −0.084 |
| lobbying | 5 | 0.660 | 0.507 | 0.618 | −0.042 |
| other | 10 | 0.283 | 0.238 | 0.252 | −0.031 |

| Type | n | Base | s3-d1 | **s3-d2** | Δ vs base |
|---|---:|---:|---:|---:|---:|
| keyword | 26 | 0.237 | 0.173 | 0.196 | −0.041 |
| natural_language | 22 | 0.322 | 0.247 | 0.304 | −0.018 |
| **fact_pattern** | 17 | 0.353 | 0.297 | **0.358** | **+0.005** |

**Val-slice IR evaluator** during training (482 queries / 482 docs):

| step | acc@1 | nDCG@5 |
|---:|---:|---:|
| 100 | 0.859 | 0.898 |
| 200 | 0.871 | 0.907 |
| **300** | **0.890** | **0.918** |
| 400 | 0.882 | 0.915 |
| 500 | 0.876 | 0.911 |
| 560 | 0.880 | 0.914 |

Peak at step 300 with a mild softening through step 560 — hint of mild
overtraining in the back half. Without intermediate checkpoints saved
we can't 65-query-eval the step-300 model, but a future run could.

**Interpretation — H1 is the residual mechanism (mostly).**

Recovery vs s3-d1 is **+0.045 nDCG@5** with the LR drop as the only
variable. Every topic recovered. Every query type recovered. The shape:

1. **fact_pattern beats base** (+0.005, n=17). First slice in any tuned
   model to clear baseline. Fact patterns are multi-sentence hypothetical
   queries — the closest distributional match to our training questions
   (FPPC-author paraphrases of conclusions). Fine-tuning *helps* exactly
   where the training distribution looks most like the eval distribution.
2. **natural_language nearly recovers** (−0.018). Same distribution
   intuition: NL queries are sentence-shaped, partially overlap with
   training questions.
3. **keyword queries remain the deficit** (−0.041). Keyword bags
   ("Section 87103(a) disqualification...") look nothing like training
   questions. This is the H4 (distribution mismatch) signature: the
   model is now good at one input shape but worse at the other.
4. **gifts_honoraria still −0.084**, though much better than s3-d1's
   −0.148. Only 7 queries; noise floor is meaningful but the topic was
   genuinely the strongest base capability and the hardest to preserve.

**Diagnosis**: At LR=5e-6, ~80% of the gap closes. The recipe is now
*nearly* viable — the remaining ~0.02 nDCG@5 gap is roughly split between:
- Residual catastrophic forgetting on the strongest base topics
  (gifts/lobbying), which lower LR / shorter training / LoRA could
  address.
- Query-distribution mismatch (H4) on keyword queries, which paraphrase
  augmentation could address.

**Wall time**: 30.6 min train + 8 min score.

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

### H1 — LR is wrecking the pretrained weights  *(CONFIRMED AS RESIDUAL MECHANISM 2026-05-22 by s3-d2)*

The standard 2e-5 fine-tuning LR is well-suited to weaker base models.
Snowflake-arctic-l-v2 is already saturated on general retrieval and
high-quality. Lower LRs (1e-6 to 1e-5) should let it specialize without
overwriting general structure.

**Outcome**: s3-d2 (LR=5e-6 on the leakage-free recipe) recovered +0.045
nDCG@5 vs s3-d1 (LR=2e-5 on same data). Gap to base shrank from −0.066 to
−0.021. fact_pattern queries (n=17) beat base by +0.005 — first slice of
any tuned model to clear baseline. So **LR was the second mechanism**;
the question is now how to close the residual ~0.02 gap.

**Candidate continuations** (in rough order of cost):
- `s3-d3-lr1e-6` — drop LR one more step. If gap closes further, LR is
  still the binding constraint. If it plateaus, we've found the LR limit
  and need a different lever.
- `s3-d4-checkpoint-sweep` — re-run s3-d2 with checkpoints every 100
  steps + 65-query eval at each. Val-slice peaked at step 300 then
  softened; the step-300 checkpoint may already beat base.
- Reintroduce hard negatives at LR=5e-6 (test if false-negative
  pollution was real on top of leakage+LR, or if it was just downstream).
- LoRA on Snowflake (rank=16 or 32 on attention layers; protects base
  capabilities by construction).

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

### H7 — Lexical leakage from question into positive doc  *(PARTIAL CONFIRM 2026-05-22 by s3-d1)*

The query text appeared as a verbatim substring of `pos_qa_text`
(`"QUESTION: <query>... CONCLUSION: ..."`), letting the model learn
identity-matching as a shortcut rather than retrieval. s3-d1 stripped
the QUESTION block; the val-slice IR evaluator stopped saturating
(0.998 → 0.85), gifts_honoraria partially recovered (0.308 → 0.490),
and natural-language queries improved (0.201 → 0.247).

**But the run still regressed −0.066 vs base.** Leakage was a real
mechanism but not the only one. A second mechanism — almost certainly
LR-driven catastrophic forgetting of the base's strongest topic
representations — is still active.

**Next experiment** (proposed s3-d2): same recipe as s3-d1
(conclusion-only positive, MNRL, no hard negs) but **LR = 5e-6**. A
single-variable test of the residual LR hypothesis on the leakage-free
recipe. If the −0.066 gap closes, LR is the second mechanism. If it
doesn't, something else is in play (training-data distribution H4,
maybe).

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
- **2026-05-22**: External LLM second-opinion review identified the
  question-in-positive-doc lexical leakage as a likely root cause
  (formalized as H7). User selected H7 as the next test. Built
  `pos_conclusion_only` as a derived column inside `train_sprint3.py`
  (split `pos_qa_text` on `\n\nCONCLUSION:`, keep right half); ran s3-d1
  as a strict single-variable ablation of s3-a1.
- **2026-05-22**: s3-d1 result: nDCG@5 0.230 (−0.066 vs Snowflake base,
  vs s3-a1's −0.097). **H7 partially confirmed**: val-slice IR
  desaturated, gifts_honoraria +0.18, natural_language +0.05; but
  −0.066 residual gap and unchanged COI suggest a second mechanism
  remains. Next test: hold the conclusion-only data, drop LR 2e-5 → 5e-6
  to probe the residual H1 (LR) story.
- **2026-05-22**: s3-d2 result (same as s3-d1 but LR=5e-6): nDCG@5
  **0.275** (−0.021 vs base, vs s3-d1's −0.066). +0.045 nDCG@5 recovered
  from a single-knob LR drop. **H1 confirmed as the residual mechanism.**
  fact_pattern queries beat base (+0.005, n=17) — first slice of any
  tuned model to clear baseline. Awaiting user call on next experiment:
  s3-d3 LR=1e-6 (extend the LR ramp), or pivot to LoRA / checkpoint
  sweep / H4 paraphrase augmentation to close the residual ~0.02 gap.

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
