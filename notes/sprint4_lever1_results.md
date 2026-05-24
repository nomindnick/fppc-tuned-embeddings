# Sprint 4 Lever 1 — Continued MLM Pretraining of Snowflake on FPPC Corpus

**Date**: 2026-05-23. **Status**: Mechanism validated, implementation
needs refinement. Overall headline metrics regress vs the Sprint 5B/C
ship candidate; the falsifiable prediction was supported in shape but
the MLM-induced geometric damage outweighed the gain.

## Goal

Sprint 5's per-query analysis pinned d3's remaining −0.011 hybrid
deficit vs OpenAI/009 to a single query: **q001** ("Section 87103(a)
disqualification business entity investment financial interest"),
where **no Snowflake variant — base, f2, or d3 — surfaced any of
the 22 gold-relevant opinions in its top-100 semantic candidates.**
Best gold-2 rank under d3 was 137 over the full 14,096-doc corpus.

Hypothesis: this is a domain-pretraining-exposure gap. OpenAI's
embedding has seen vastly more legal text and apparently learned that
"Section 87103(a)" is semantically close to opinions analyzing it.
Continued MLM pretraining of Snowflake on the FPPC corpus should
narrow that gap.

Falsifiable prediction: after MLM, q001's gold-2 opinions should
move from rank 137+ into d3's top-100 semantic candidates.

## Setup

**MLM run** (`experiments/sprint4_mlm/`):
- Base: Snowflake-arctic-embed-l-v2.0 → `AutoModelForMaskedLM` (the
  Snowflake checkpoint dropped `lm_head` during its retrieval
  fine-tuning; it's freshly initialized on load)
- Training corpus: 13,469 non-held-out opinions → 56,528 chunks at
  512 tokens (max 10 chunks/opinion to prevent long-opinion dominance)
- Hyperparameters: bs=16, LR=1e-5, warmup_ratio=0.10, weight_decay=0.01,
  mlm_probability=0.15, **2 epochs**, bf16
- Total: 7,066 steps over 6.4 hours wall-clock
- Final MLM loss: 0.97 (down from initial ~14)
- Encoder transplanted back into SentenceTransformer wrapper; saved
  to `data/models/snowflake_fppc_mlm/`

**Downstream contrastive** (`experiments/sprint4_d3fppc/`):
- Sprint 3's winning recipe applied to the FPPC-MLM-adapted base
- MNRL on `pos_conclusion_only`, LR=1e-6, 1 epoch, bs=16, bf16,
  warmup_ratio=0.10
- 30 minutes wall-clock; saved to `data/models/sprint4_d3fppc/`

## Falsifiable prediction — supported in shape

Best gold-2 rank for q001 over the full 14,096-doc corpus:

| Model | q001 best score-2 rank | gold-2 in top-100 | gold-any in top-100 |
|---|---:|---:|---:|
| Snowflake-base | not measured | (presumably similar to d3) | — |
| **d3 (baseline)** | **137** | **0/12** | **0/22** |
| MLM-only Snowflake | **60** | 1/12 | 2/22 |
| **d3-FPPC** | **42** | **1/12** | 2/22 |

**Both MLM-only and d3-FPPC moved q001's best gold-2 from rank 137 to
~50.** One gold-2 (`10-198`) entered the top-100 candidate pool.
The mechanism is real — domain-adaptive MLM does push FPPC-relevant
opinions toward statute-keyword queries in Snowflake's latent space.

q018, the other COI deficit query, shows the cost of the same
intervention:

| Model | q018 best score-2 rank |
|---|---:|
| d3 (baseline) | **1** |
| MLM-only Snowflake | 74 |
| d3-FPPC | 5 |

d3's previous strength on q018 (00-035 at semantic rank 1) was
disrupted by MLM (rank 74) and partially recovered by the d3 contrastive
(rank 5). The contrastive can recover but not fully — the encoder
geometry is permanently shifted.

## Headline numbers — regression

### Semantic-only nDCG@5

| Model | nDCG@5 | MRR | COI | KW | FP |
|---|---:|---:|---:|---:|---:|
| Snowflake-base | 0.296 | 0.522 | 0.106 | 0.237 | 0.353 |
| **d3 (Sprint 3 winner)** | **0.290** | **0.538** | **0.124** | 0.190 | **0.390** |
| MLM-only Snowflake | 0.158 | 0.328 | 0.038 | 0.118 | 0.202 |
| **d3-FPPC** | 0.232 | 0.472 | 0.087 | 0.122 | 0.344 |

MLM alone caused **−0.138 nDCG@5** vs Snowflake-base — broad
geometric damage across all topics/types, not a targeted regression.
The contrastive step recovered ~0.07 of that but couldn't reach the
original d3 baseline (still −0.058 below d3).

### Hybrid nDCG@5 (best fusion settings per model)

| Engine | best fusion | nDCG@5 | MRR | COI | KW | FP |
|---|---|---:|---:|---:|---:|---:|
| OpenAI / 009 (production) | stock | 0.3752 | **0.6721** | **0.189** | 0.324 | 0.407 |
| MLM-only Snowflake | w=0.40 cb=1.5 | 0.3672 | 0.6686 | 0.175 | 0.303 | 0.437 |
| **d3-FPPC** | w=0.30 cb=1.5 | 0.3707 | 0.6704 | 0.170 | 0.307 | 0.439 |
| Snow-base @ best | w=0.30 cb=1.5 | 0.3824 | 0.6727 | 0.168 | 0.316 | 0.446 |
| **d3 (Sprint 3 winner) @ best cite** | w=0.40 cb=1.5 | **0.3911** | **0.6726** | 0.178 | 0.333 | 0.454 |
| **d3 @ best allpath** | w=0.40 cb=2.0 | **0.3975** | 0.6655 | 0.177 | 0.333 | 0.461 |

**d3-FPPC's best hybrid (0.371) loses to OpenAI/009 by −0.005 and to
the d3 ship candidate by −0.020.** This is a Sprint 4 regression — the
MLM intervention hurt the deployable product. The semantic-axis
damage carried into the fusion despite BM25's averaging.

## Interpretation

The hypothesis (MLM as the cure for q001-class statute-keyword
failures) is **mechanistically validated** but **operationally
counterproductive at these hyperparameters**:

- MLM moved q001's gold from rank 137 to 42 → the bi-encoder *can* learn
  FPPC-specific statute semantics.
- MLM also disrupted dozens of other queries where Snowflake was already
  doing well → the geometric damage outweighs the targeted gain.
- The downstream d3 contrastive on FPPC-MLM partially repairs but
  cannot fully restore the geometry in a single epoch.

Likely root causes of the over-damage:
1. **Freshly-initialized lm_head**: the saved Snowflake checkpoint
   doesn't include `lm_head` weights. The random head produced large
   initial gradients that propagated into the encoder despite warmup.
2. **2 epochs is too many** for an already-saturated encoder. Standard
   domain-adaptive MLM recipes often use 1 epoch at the corpus size
   we have (~13k docs).
3. **LR=1e-5 was too hot** for protecting the encoder. Sprint 3 found
   that even 1e-5 over-wrote Snowflake's contrastive directions; the
   same may apply to MLM.

## Options for what next

### Option A — Retry MLM with conservative hyperparameters (Lever 1b)

Concrete changes:
- LR: **5e-6** (half of the failed run)
- Epochs: **1** (half)
- Optional: **freeze the encoder for the first 500 steps**, let the
  lm_head converge against frozen embeddings, then unfreeze. This is
  the canonical recipe for adding a new head to a saturated encoder.
- Optional: warmup_ratio=0.20 (longer)
- Same chunking + masking otherwise

Cost: ~3-4 hours MLM + 30 min contrastive + 30 min eval = ~5 hours.

Falsifiable prediction: q001's best gold-2 should still move
(target: ≤ 50), AND q018's best gold-2 should stay ≤ 3, AND
semantic-only nDCG@5 should be **≥ 0.28** (within 0.01 of d3 baseline).
If all three hold, the hybrid should beat 0.39 and possibly 0.40.

### Option B — Cross-encoder reranker on d3-FPPC's top-100 (Lever 2)

The MLM-shifted candidate pool for q001 now contains a gold-2
(`10-198`) at rank 42. A cross-encoder reranker over the top-100
could promote it.

Cost: ~half a day to wire BGE-reranker-large in as a post-fusion
stage. ~2,000 reranker calls per eval run (~1 min on Strix Halo iGPU).

Concrete plan: pull d3-FPPC's fusion output (top-100 instead of
top-20), cross-encoder-rerank to top-10, score.

Less appealing now because d3-FPPC's hybrid is already worse than
d3-baseline's, so we'd be reranking a weaker pool. Better to apply
Lever 2 to **d3 (no MLM)** if we go this route.

### Option C — Ship d3 + retune now, document Lever 1 as inconclusive

We already have a deployable improvement: d3 + w=0.40 + cb=1.5 (cite)
beats OpenAI/009 by +0.016 nDCG@5 with MRR tied. The remaining
−0.011 COI gap is small enough to be a publishable finding rather
than a blocker.

Cost: just the writeup + production handoff to the app repo.

## Recommendation

**Option A (Lever 1b — conservative MLM retry)** is the highest-value
move. The mechanism is validated (q001 moved exactly as predicted);
we just need to limit the collateral damage. The conservative recipe
is the canonical fix for "MLM over-damaged the encoder" and the
expected outcome is a hybrid score in the 0.39–0.42 range — potentially
hitting the SPEC stretch goal.

If Lever 1b also regresses, then we have strong evidence that
domain-adaptive MLM is fundamentally incompatible with the saturated
Snowflake encoder for this corpus, and Option B (reranker on
the original d3) becomes the next move.

## Files produced

- `scripts/train_mlm.py` — MLM trainer
- `scripts/embed_and_score.py` — embed + 65-query semantic-only eval
- `experiments/sprint4_mlm/` — MLM run config, train log, build report
- `experiments/sprint4_mlmonly_sweep/` — MLM-only hybrid sweep
- `experiments/sprint4_d3fppc/` — downstream contrastive run
- `experiments/sprint4_d3fppc_sweep/` — d3-FPPC hybrid sweep
- `data/models/snowflake_fppc_mlm/` — MLM-adapted Snowflake (2.3GB)
- `data/models/sprint4_d3fppc/` — d3 contrastive on FPPC-MLM base (2.3GB)
