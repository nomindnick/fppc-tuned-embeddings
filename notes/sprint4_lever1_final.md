# Sprint 4 Lever 1 — Final findings (2026-05-23)

Both Lever 1a (aggressive MLM) and Lever 1b (conservative MLM with
encoder freeze) failed to produce a hybrid that beats the Sprint 5
ship candidate. But the data reveals a deeper structural issue worth
recording.

## Headline comparison

| Engine | Best hybrid nDCG@5 | MRR | COI | Config |
|---|---:|---:|---:|---|
| OpenAI / 009 (production) | 0.3752 | **0.6721** | **0.189** | stock |
| **d3 (Sprint 3) @ best [SHIP CANDIDATE]** | **0.3911** | **0.6726** | 0.178 | w=0.40, cb=1.5 |
| d3 allpath best | **0.3975** | 0.6655 | 0.177 | w=0.40, cb=2.0, allpath |
| Snowflake-base @ best | 0.3824 | 0.6727 | 0.168 | w=0.30, cb=1.5 |
| Lever 1a MLM-only | 0.3676 | 0.6582 | 0.174 | w=0.30, cb=1.5 |
| Lever 1a d3-FPPC | 0.3707 | 0.6704 | 0.170 | w=0.30, cb=1.5 |
| Lever 1b MLM-only (conservative) | 0.3687 | 0.6589 | 0.173 | w=0.30, cb=1.5 |
| Lever 1b d3-FPPC-b | 0.3766 | 0.6513 | 0.165 | w=0.45, cb=2.0, allpath |

Best Lever 1 variant: **d3-FPPC-b at 0.3766** — beats OpenAI by +0.001
(noise), loses to ship candidate by −0.015.

## The structural finding: MLM and d3 contrastive push in opposite directions

The most interesting Sprint 4 finding is in the gold-rank tracking for
q001 ("Section 87103(a) disqualification …"), the single query that
accounted for most of d3's COI deficit vs OpenAI:

| Stage | q001 best score-2 rank | gold-2 in top-100 |
|---|---:|---:|
| d3 (no MLM) baseline | **137** | **0 / 12** |
| Lever 1a MLM-only | 60 | 1 / 12 |
| Lever 1a d3-FPPC (after contrastive) | 42 | 1 / 12 |
| Lever 1b MLM-only (conservative) | **42** | **4 / 12** |
| Lever 1b d3-FPPC-b (after contrastive) | **120** | **0 / 12** |

Lever 1b's MLM-only was the strongest q001 mechanism we've measured:
**4 of 12 gold-2 opinions in the top-100 semantic candidates** (vs 1
in Lever 1a, 0 in d3 baseline). The freeze-then-unfreeze recipe
allowed the encoder to absorb deeper FPPC-specific structure.

But the d3 contrastive step on top **wiped that structure out**:
q001's best gold-2 went from rank 42 (MLM-only) back to rank 120
(d3-FPPC-b) — even *worse* than the original d3 baseline at 137.

q018, the other COI deficit query, shows the opposite pattern:

| Stage | q018 best score-2 rank |
|---|---:|
| d3 (no MLM) | 1 |
| Lever 1b MLM-only | 50 |
| **Lever 1b d3-FPPC-b** | **1** (fully recovered) |

The contrastive perfectly rescued q018. For queries where Snowflake
already had good geometry (q018), d3 contrastive pulls it back to
that geometry. For queries where MLM *added* useful structure (q001),
the same contrastive pull *erases* it.

**Mechanistically**: the d3 training data is `(opinion_question,
opinion_conclusion)` pairs. There are no statute-keyword anchors. The
contrastive objective directly rewards pulling `question` and `conclusion`
embeddings toward each other and pushing them away from in-batch
distractors. None of those gradients reinforce the
"statute_keyword → opinion-citing-statute" structure MLM added. The
contrastive's softmax over in-batch negatives systematically *unwinds*
any encoder geometry that doesn't help (Q, C) pair retrieval.

This is a fundamental limitation, not a hyperparameter issue. To
preserve MLM's q001 gains we'd need either:
1. A different training data mix that explicitly contains
   (statute_keyword, opinion-citing-statute) anchors.
2. A different loss that has some encoder-preservation regularization
   (e.g. KL-divergence to the MLM-base outputs).
3. A two-encoder architecture where d3 stays the retrieval-axis and
   a separate FPPC-MLM encoder is queried for statute-keyword cases
   via routing.

All three are real Sprint 4-shape investments (multi-day each).

## Conclusion on Lever 1

**Mechanism validated** (MLM moves q001 reliably), **but currently
not deployable**. The two-stage MLM → contrastive pipeline cannot
preserve MLM's domain structure through the contrastive step at this
recipe. Even the strongest MLM-only model (Lever 1b, q001 4/12 in
top-100) lost that structure entirely after 1 epoch of d3 contrastive.

## Where this leaves us

The Sprint 5 ship candidate is still our deployable improvement:
- **d3 @ w=0.40, cb=1.5 (cite-routing)**: +0.016 nDCG@5 over OpenAI/009,
  MRR tied at 0.673.
- **d3 @ w=0.40, cb=2.0 (allpath)**: +0.022 nDCG@5, −0.007 MRR.

The remaining −0.011 COI gap (mostly q001's −0.339) requires either:

1. **Lever 2 — Cross-encoder reranker on d3's top-K**. Bypass the
   bi-encoder limitation entirely; the cross-encoder can compute
   query↔doc statute-keyword similarity at search time. ~½ day to
   wire up BGE-reranker-large. The catch: q001's gold isn't in d3's
   top-100 (best score-2 rank = 137), so we'd need to enlarge the
   reranker pool to top-200 or top-300 — about double the inference
   cost per query.

2. **Lever 3 — MarginMSE distillation**. Use a cross-encoder teacher
   to produce continuous relevance scores for (query, candidate)
   pairs, then train d3 with MSE loss against those scores. This is
   the cleanest structural fix because the teacher's signal IS the
   statute-keyword relevance signal, and the loss directly encodes
   it into the encoder. Multi-day project.

3. **Ship d3 + retune now**. We have a +0.016 deployable improvement.
   Move on.

## My recommendation

**Ship d3 + retune now** (Sprint 5 candidate) and **defer Lever 2/3
to a follow-on project**. Reasoning:

- The current ship beats production OpenAI/009 by a meaningful margin
  on the headline metric, MRR tied.
- Lever 2 has uncertain payoff because q001's gold may not even be in
  the reranker's input pool at any practical depth.
- Lever 3 is the right structural fix but requires writing a new
  training pipeline and integrating a teacher model — at least a
  week of work for what could be a modest additional lift.
- Two of the three remaining levers we'd test in a hypothetical Sprint 5
  (more epochs, more LR, more LoRA capacity) are already known to
  regress per Sprint 3's lever sweep.

If you disagree and want to push further, **Lever 2** is the
shortest test. It can be deployed in production behind a feature
flag, lets us measure real-user wins, and keeps the door open to
Lever 3 if the rerank pool turns out to be too shallow.

## Files produced (Sprint 4)

- `scripts/train_mlm.py` — MLM trainer with optional encoder freeze
- `scripts/embed_and_score.py` — semantic-only embed + score helper
- `scripts/sprint4_rollup.py` — multi-experiment rollup
- Lever 1a: `experiments/sprint4_mlm/`, `experiments/sprint4_d3fppc/`,
  `experiments/sprint4_mlmonly_sweep/`, `experiments/sprint4_d3fppc_sweep/`
- Lever 1b: `experiments/sprint4b_mlm/`, `experiments/sprint4b_d3fppc/`,
  `experiments/sprint4b_mlmonly_sweep/`, `experiments/sprint4b_d3fppc_sweep/`
- Models: `data/models/snowflake_fppc_mlm/`,
  `data/models/snowflake_fppc_mlm_b/`,
  `data/models/sprint4_d3fppc/`,
  `data/models/sprint4b_d3fppc/`
- Notes: `notes/sprint4_lever1_results.md` (1a alone),
  `notes/sprint4_lever1_final.md` (this file — full picture)
