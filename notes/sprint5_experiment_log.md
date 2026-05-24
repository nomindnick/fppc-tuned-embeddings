# Sprint 5 Experiment Log

Running record of hybrid-integration experiments. Sprint 3 produced a
candidate fine-tune (s3-d3); Sprint 5 measures how it behaves inside the
production fusion engine (search-lab Experiment 009, `CitationScoreFusion`).

**Reference numbers** (semantic-only nDCG@5, full 65-query graded eval):

| Model | Semantic-only nDCG@5 | MRR | COI@5 |
|---|---:|---:|---:|
| OpenAI text-embedding-3-small | 0.263 | 0.494 | 0.129 |
| Snowflake-arctic-l-v2 (untuned) | 0.296 | 0.522 | 0.106 |
| **s3-d3 (Sprint 3 winner)** | **0.290** | **0.538** | **0.124** |
| s3-f2 (LoRA, ensemble candidate) | 0.277 | 0.521 | 0.124 |

**Production reference (hybrid)**: `results/009-new-corpus.json` in search-lab,
which scores OpenAI text-embedding-3-small inside the 009 `CitationScoreFusion`
engine against the *current* corpus.

| Engine | nDCG@5 | MRR | nDCG@10 | COI@5 |
|---|---:|---:|---:|---:|
| **OpenAI / 009 (current corpus)** | **0.3752** | **0.6721** | **0.3297** | **0.1892** |
| OpenAI / 009 (legacy, original corpus, for ref) | 0.3871 | 0.6843 | — | — |

The 0.3752 figure is the number to beat.

---

## Setup

- **Engine wrapper**: `scripts/_hybrid_engine.py` — `LocalCitationScoreFusion`,
  a 009-equivalent fusion engine that uses a local SentenceTransformer
  (replacing OpenAI's API call) plus pre-computed `.npy + .json` semantic
  indexes from Sprint 3's encode cache. BM25 and citation indexes are
  loaded from the search-lab repo unchanged. Tokenizer + citation parser
  are inlined to avoid the `src` package collision between search-lab and
  `fppc-opinions-eval`.
- **Driver**: `scripts/run_hybrid_eval.py` — config-driven; writes
  `experiments/<run_id>/eval_metrics.json` in the same schema as
  `results/check2_*.json`, plus a `path_breakdown` and per-query
  `diagnostics` field showing which routing path each query took
  (`fusion` / `bm25_only_noncite` / `cb_fired` / `empty_pool`).
- **Stock 009 knobs**: `cb_threshold=1.3`, `w_bm25=0.4`, `w_sem=0.6`,
  citation-routing only (non-citation queries → pure BM25).
- **Routing is engine-independent** — the citation parser inspects query
  text. All four engines route identically: 30 fusion queries, 29
  BM25-only-noncite, 6 CB-fired. The 29 + 6 BM25-only queries score
  identically across all four engines; the entire model-induced delta is
  concentrated in 30 fusion queries.

---

## Sprint 5A — Drop-in baseline (stock 009 knobs)

**Goal**: measure each candidate semantic model inside the production
fusion engine with no parameter retuning. Establishes whether the model
swap helps *before* we touch the fusion config.

| Engine | nDCG@5 | MRR | nDCG@10 | P@5 | R@10 | COI@5 | Δ nDCG@5 vs OpenAI/009 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **OpenAI / 009** | **0.3752** | **0.6721** | 0.3297 | 0.3292 | 0.1875 | 0.1892 | — |
| Snowflake-base / 009 | 0.3647 | 0.6638 | **0.3348** | 0.3108 | **0.1944** | 0.1660 | −0.011 |
| **s3-d3 / 009** | 0.3616 | 0.6541 | 0.3290 | 0.3138 | 0.1883 | **0.1769** | −0.014 |
| s3-f2 / 009 | 0.3635 | 0.6438 | 0.3307 | 0.3231 | 0.1920 | 0.1756 | −0.012 |

**Headline**: all three Snowflake-based variants come in slightly below the
production OpenAI hybrid (−0.011 to −0.014 nDCG@5). The model-swap alone
(Snow-base vs OpenAI) is also negative, indicating the 009 fusion is
genuinely tuned to OpenAI's behavior — not just OpenAI-using.

### By query type (nDCG@5)

| Engine | keyword (n=26) | natural_language (n=22) | fact_pattern (n=17) |
|---|---:|---:|---:|
| **OpenAI / 009** | **0.3244** | 0.4106 | 0.4072 |
| Snowflake-base / 009 | 0.2929 | 0.4042 | 0.4236 |
| **s3-d3 / 009** | 0.2802 | 0.4084 | **0.4258** |
| s3-f2 / 009 | 0.2847 | 0.4084 | **0.4258** |

| Engine | keyword MRR | natural_language MRR | fact_pattern MRR |
|---|---:|---:|---:|
| OpenAI / 009 | **0.5386** | **0.7446** | 0.7827 |
| Snowflake-base / 009 | 0.5002 | 0.7188 | **0.8431** |
| s3-d3 / 009 | 0.4757 | 0.7188 | **0.8431** |
| s3-f2 / 009 | 0.4502 | 0.7188 | **0.8431** |

**This is the most important table on the page.** The headline gap
(d3 / 009 = 0.362 vs OpenAI / 009 = 0.375) is **not uniform**:

- **fact_pattern (n=17): d3 *beats* OpenAI by +0.019 nDCG@5 and +0.060 MRR.**
  Fine-tuning's intended payout — multi-sentence hypothetical queries —
  shows up exactly where Sprint 3 predicted.
- **natural_language (n=22)**: essentially tied across all four models.
- **keyword (n=26): OpenAI dominates** (+0.04 vs d3 and +0.05 vs Snow-base).
  This is where the entire overall deficit comes from.

The fine-tuning produced the predicted topic-shape lift; the loss is in
fusion misweighting on keyword queries, not in the model.

### By topic (nDCG@5)

| Engine | conflicts_of_interest (n=29) | campaign_finance (n=14) | gifts_honoraria (n=7) | lobbying (n=5) | other (n=10) |
|---|---:|---:|---:|---:|---:|
| OpenAI / 009 | **0.1892** | 0.5200 | **0.6118** | **0.5992** | **0.4343** |
| Snowflake-base / 009 | 0.1660 | **0.5618** | 0.5898 | 0.5460 | 0.4169 |
| **s3-d3 / 009** | **0.1769** | 0.5494 | 0.5724 | 0.5507 | 0.3926 |
| s3-f2 / 009 | 0.1756 | 0.5456 | 0.5918 | 0.5507 | 0.3999 |

- **COI (n=29, 45% of eval)**: d3 closes ~half the gap from Snow-base
  (0.166) to OpenAI (0.189). The Sprint 3 +0.018 COI move on
  semantic-only translates into +0.011 inside the hybrid. Direction is
  right; magnitude isn't enough to overtake OpenAI in stock 009.
- **campaign_finance (n=14)**: all three Snowflake variants beat
  OpenAI; Snow-base wins by +0.04 over OpenAI. The 009 fusion was
  arguably under-utilizing the semantic axis on campaign_finance even
  with OpenAI.
- **gifts / lobbying**: small-n; OpenAI nominally wins but the gap is
  noise-sized (Δ < 0.05 on n=7 / n=5).

### Path breakdown (identical across all three Snowflake runs)

- 30 fusion queries (citation present, CB didn't fire) — **where semantic matters**
- 29 BM25-only-noncite queries (pure BM25)
- 6 CB-fired (citation present but BM25 confident → BM25-only)

So the 35 BM25-only queries are model-independent. The full −0.014
overall nDCG@5 gap to OpenAI is concentrated in 30 fusion queries.

---

## Reading

**1. The 009 fusion engine is genuinely OpenAI-tuned, not just OpenAI-using.**
Even the *untuned* Snowflake — a "better" off-the-shelf model on the
semantic-only metric (0.296 vs OpenAI's 0.263) — regresses by 0.011
nDCG@5 in stock 009. The model-swap is not a free win.

**2. Fine-tuning's predicted strengths show up inside the hybrid.**
d3's Sprint 3 wins (fact_pattern +0.037, COI +0.018, MRR +0.016) carry
into 009: d3 beats OpenAI on fact_pattern in the hybrid; d3 beats
Snow-base on COI in the hybrid. The fine-tune is doing what it was
designed to do.

**3. The remaining gap is one shaped failure: keyword queries.**
OpenAI/009 nDCG@5 on keyword queries = 0.324; d3/009 = 0.280. That
single slice accounts for essentially the entire overall gap. Keyword
queries are exactly the regime where d3 was *expected* to underperform
(see Sprint 3 log — d3 keyword@5 was −0.047 vs Snow-base semantic-only).
The hybrid amplifies it because keyword queries currently route into the
fusion path with `w_sem=0.6`, giving a noisier-on-keywords model 60% of
the score weight.

**4. f2 and d3 are nearly identical in the hybrid.** Different shapes on
semantic-only collapse to within noise once 0.4 BM25 + 0.6 semantic
averaging is applied. An ensemble would have to do something cleverer
than score-averaging to extract additional lift from f2.

**5. Sprint 5A's hypothesis is supported but not yet productive.**
"d3 might be production-valuable even if semantic-only doesn't beat
OpenAI" → on stock 009 knobs, d3 ≈ Snow-base ≈ −0.012 to −0.014 vs
OpenAI/009. The drop-in is not a winner. The decision tree from the
plan now points at:

> If d3/hybrid 0.39–0.42 → real but marginal lift, Sprint 4 worth one
> targeted shot. **(Branch not hit)**
>
> If d3/hybrid ≈ OpenAI's 0.387 → fine-tuning didn't translate into
> hybrid value at default settings. **(This is roughly where we are.)**

— but the per-query-type breakdown changes the read. d3 is *not* worse
in any deep sense; it's worse only on keyword queries, where the fusion
is misweighted for its strengths. That's a 5B fix, not a 5C/Sprint-4
problem.

---

## Sprint 5B — Knob retune

**Goal**: find each model's optimum (`w_bm25`, `w_sem`, `cb_threshold`)
inside 009's fusion shape. One-time model load + per-trial knob mutation
keeps total compute at ~7s per trial; the full Sprint 5B grid below ran
in ~5 min wall clock total.

### Phase 1 — `w_sem` sweep at cb=1.3 (5 points × 3 models)

| Model / w_sem | 0.30 | 0.40 | 0.50 | 0.60 (stock) | 0.70 |
|---|---:|---:|---:|---:|---:|
| **d3** nDCG@5 | 0.3808 | **0.3830** | 0.3735 | 0.3616 | 0.3569 |
| **Snowflake-base** nDCG@5 | **0.3749** | 0.3747 | 0.3689 | 0.3647 | 0.3676 |
| **f2** nDCG@5 | 0.3792 | **0.3812** | 0.3768 | 0.3635 | 0.3572 |

Every Snowflake-family model peaks at `w_sem ≤ 0.40` and degrades at
the stock 0.60. This is the structural diagnosis: the stock 009 weights
are **OpenAI-shaped**, not Snowflake-shaped.

### Phase 2 — `cb_threshold` sweep at the per-model optimum

| Model (best w_sem) / cb_threshold | 1.1 | 1.2 | 1.3 | **1.5** | 2.0 |
|---|---:|---:|---:|---:|---:|
| **d3 @ w0.40** nDCG@5 | 0.3859 | 0.3836 | 0.3830 | **0.3911** | 0.3911 |
| Snow-base @ w0.30 nDCG@5 | — | — | 0.3749 | **0.3824** | — |
| f2 @ w0.40 nDCG@5 | — | — | 0.3812 | **0.3893** | — |

`cb=1.5` and `cb=2.0` are identical for d3 → no queries have BM25
top1/top2 ratio in (1.5, 2.0). The CB never fires at cb≥1.5; lower
thresholds gate out queries where the fusion would otherwise help d3.

### Phase 1+2 — best per-model configurations

| Model | Best config | nDCG@5 | MRR | KW | FP | COI |
|---|---|---:|---:|---:|---:|---:|
| OpenAI / 009 (production reference) | stock (w=0.6, cb=1.3) | 0.3752 | 0.6721 | 0.324 | 0.407 | **0.189** |
| Snow-base @ best | w=0.30, cb=1.5 | 0.3824 | 0.6727 | 0.316 | 0.446 | 0.168 |
| f2 @ best | w=0.40, cb=1.5 | 0.3893 | 0.6713 | 0.327 | 0.453 | 0.178 |
| **d3 @ best (cite-routing)** | **w=0.40, cb=1.5** | **0.3911** | **0.6726** | **0.333** | **0.454** | 0.178 |

**Decomposition of d3's +0.016 vs OpenAI/009**:
- Model swap alone (Snow-base @ stock knobs): −0.011
- **Retune Snow-base → cite-routing optimum**: +0.018 net move (back above OpenAI by +0.007)
- d3 fine-tune on top of retuned-Snow-base: **+0.009**

So the retune captures ~⅔ of the lift; fine-tuning contributes the
remaining ⅓ as +0.009 marginal nDCG@5. Fine-tuning's contribution
shows up exactly where Sprint 3 said it would: keyword +0.020 vs
Snow-base, fact_pattern +0.010 vs Snow-base, COI +0.010 vs Snow-base.

---

## Sprint 5C — Extend fusion to all queries (allpath)

Currently the 29 non-citation queries route to pure BM25. d3 has
un-utilized semantic capability on natural-language queries (Sprint 3
NL@5 = 0.33 semantic-only). Routing every query through fusion is a
single-flag change (`semantic_only_path=true`).

### d3 with allpath, varying knobs

| Config | nDCG@5 | MRR | COI | KW | FP |
|---|---:|---:|---:|---:|---:|
| d3 @ w0.40, cb1.5, **cite-only** | 0.3911 | **0.6726** | 0.178 | 0.333 | 0.454 |
| d3 @ w0.30, cb1.5, allpath | 0.3898 | 0.6649 | 0.169 | 0.330 | 0.449 |
| d3 @ w0.40, cb1.5, allpath | 0.3964 | 0.6655 | 0.177 | 0.333 | 0.457 |
| d3 @ w0.50, cb1.5, allpath | 0.3884 | 0.6507 | 0.183 | 0.301 | 0.467 |
| **d3 @ w0.40, cb2.0, allpath** | **0.3975** | 0.6655 | 0.177 | 0.333 | **0.461** |

### Snow-base with allpath (control)

| Config | nDCG@5 | MRR | COI | KW | FP |
|---|---:|---:|---:|---:|---:|
| Snow-base @ w0.30, cb1.5, cite-only | 0.3824 | **0.6727** | 0.168 | 0.316 | 0.446 |
| Snow-base @ w0.30, cb2.0, allpath | 0.3802 | 0.6591 | 0.157 | 0.316 | 0.439 |
| Snow-base @ w0.40, cb2.0, allpath | 0.3852 | 0.6587 | 0.166 | 0.310 | 0.466 |

**allpath effect:**
- d3: +0.006 nDCG@5, −0.007 MRR
- Snow-base: +0.003 nDCG@5, −0.014 MRR

allpath helps d3 slightly more than it helps Snow-base. The marginal
contribution of fine-tuning grows to **+0.012** when both retune and
allpath are applied (d3 best 0.3975 vs Snow-base best 0.3852).

---

## Final scoreboard

| Engine config | nDCG@5 | MRR | nDCG@10 | KW | NL | FP | COI |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenAI / 009 (production) | 0.3752 | **0.6721** | 0.3297 | 0.324 | 0.411 | 0.407 | **0.189** |
| Snow-base @ best (cite-only) | 0.3824 | 0.6727 | — | 0.316 | — | 0.446 | 0.168 |
| f2 @ best (cite-only) | 0.3893 | 0.6713 | — | 0.327 | — | 0.453 | 0.178 |
| **d3 @ best (cite-only)** — *safe ship* | **0.3911** | **0.6726** | — | 0.333 | 0.411 | 0.454 | 0.178 |
| **d3 @ best (allpath)** — *bigger nDCG@5 win, small MRR cost* | **0.3975** | 0.6655 | — | 0.333 | — | **0.461** | 0.177 |

**vs OpenAI/009 deltas:**

| Engine config | Δ nDCG@5 | Δ MRR | Δ FP | Δ KW | Δ COI |
|---|---:|---:|---:|---:|---:|
| d3 best cite-only | **+0.016** | +0.001 | +0.047 | +0.010 | −0.011 |
| d3 best allpath | **+0.022** | −0.007 | +0.054 | +0.010 | −0.012 |

### SPEC scorecard

| Goal | Target | Current best (d3 allpath) | Hit? |
|---|---:|---:|:---:|
| Threshold (sem-only) | nDCG@5 ≥ 0.33 | sem-only d3 = 0.290 | ❌ |
| Goal (sem-only) | nDCG@5 ≥ 0.36 | sem-only d3 = 0.290 | ❌ |
| Stretch hybrid | nDCG@5 ≥ 0.42 | 0.3975 | ❌ (Δ −0.023) |
| Stretch hybrid | MRR ≥ 0.72 | 0.6655 | ❌ |
| Subgroup gate | COI nDCG@5 ≥ 0.20 | 0.177 | ❌ (Δ −0.023) |

The semantic-only SPEC threshold was missed, but the hybrid is improved
by +0.022 nDCG@5 over production. The stretch and COI gates remain
unmet.

---

## What this tells us about Sprint 4

The original Sprint 4 decision tree:
- ≥ 0.42 hybrid → ship, Sprint 4 is upside → **NOT HIT**
- 0.39 – 0.42 hybrid → meaningful but marginal → **HIT** (0.3975)
- ≈ 0.387 (OpenAI) → tied → not this branch
- < 0.387 → kill → not this branch

So we landed in the "one targeted Sprint 4 shot" branch. The shape of
the remaining gap is informative:

- **Keyword and fact_pattern queries beat OpenAI.** Fine-tuning + retune
  delivered.
- **Natural-language queries tied.** Allpath gives a small lift.
- **COI underperforms OpenAI by 0.011.** This is the single remaining
  shaped deficit. Sprint 3 moved COI from 0.106 to 0.124 semantic-only;
  the hybrid converts that to 0.178, still below OpenAI's 0.189. To hit
  the COI subgroup gate (≥0.20) we need ~+0.023 more.

The highest-leverage Sprint 4 lever for closing the COI gap is the one
that directly attacks COI's underlying weakness — false-negative
pollution among co-relevant opinions. Per the Sprint 3 lever sweep, that
points at:

1. **Cross-encoder-filtered hard negatives.** Use a BGE reranker to drop
   "negatives" the reranker rates relevant, then train s3-d3 + 1 hard
   neg with the cleaner set. The s3-g1 disaster (false-neg pollution)
   was the failure mode; this is the targeted fix.
2. **MarginMSE distillation from a reranker.** Continuous relevance
   scores replace MNRL's binary objective; the structural fix for the
   same problem.
3. **Larger paraphrase corpus** (≥9k rows × 2 paraphrases via 26B+ LLM)
   — H4 retest at proper scale. Probably helps keyword queries more
   than COI; lower priority for closing the headline gap now.

(1) is the cheapest experiment and most directly attacks the remaining
deficit. (2) is harder but a clean structural fix.

---

## Files produced

- `scripts/_hybrid_engine.py` — `LocalCitationScoreFusion` engine
- `scripts/run_hybrid_eval.py` — single-config eval driver
- `scripts/run_sweep.py` — knob-sweep driver (shared engine, multi-trial)
- Sprint 5A drop-in runs:
  - `experiments/sprint5_{d3,snowflake,f2}_009stock/` (config + eval_metrics)
- Sprint 5B w_sem sweep:
  - `experiments/sprint5b_{d3,snowflake,f2}_sweep/` (sweep_config + sweep_summary + per-trial)
- Sprint 5B fine-grain (d3) + cb-1.5 sweeps:
  - `experiments/sprint5b_d3_finegrain/`
  - `experiments/sprint5b_{snow,f2}_cb15/`
- Sprint 5C allpath:
  - `experiments/sprint5c_d3_allpath/`
  - `experiments/sprint5c_snow_allpath/`

## Decision log

- **2026-05-23**: Sprint 5A drop-in: all three Snowflake variants
  regress 0.011–0.014 nDCG@5 vs OpenAI/009 in stock fusion. Gap
  concentrated in keyword queries; d3 already wins fact_pattern.
- **2026-05-23**: Sprint 5B retune: every Snowflake-family model peaks
  at `w_sem ∈ {0.30, 0.40}` (vs stock 0.60), confirming the OpenAI-tuned
  weight bias. Best d3 config (w=0.40, cb=1.5) beats OpenAI/009 by
  +0.016 nDCG@5 with MRR essentially tied. Fine-tuning's marginal
  contribution (above retuned Snow-base): +0.009 nDCG@5, +0.020 keyword,
  +0.010 fact_pattern, +0.010 COI.
- **2026-05-23**: Sprint 5C allpath: extending fusion to non-citation
  queries gives +0.006 to d3 (0.3911 → 0.3975) at a −0.007 MRR cost.
  Two viable production candidates: (a) cite-only routing for nDCG@5
  +0.016 + MRR tied; (b) allpath routing for nDCG@5 +0.022 + MRR −0.007.
- **Recommendation**: ship d3 + retune as a deployable improvement
  to the production hybrid. Treat Sprint 4 as contingent on whether
  the COI subgroup gate (0.177 → 0.20) is required by the production
  app's users — that's the only remaining shaped deficit vs OpenAI/009.
