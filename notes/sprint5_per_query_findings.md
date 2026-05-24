# Sprint 5 — Per-Query Findings & Sprint 4 Implications

Companion analysis to `sprint5_per_query_analysis.md` (the raw table dump
from `scripts/analyze_per_query.py`). This file interprets the findings
and maps them to candidate Sprint 4 levers.

---

## Methodology footnote: BM25-only-path noise

Comparing my engine's per-query output against the published
`009-new-corpus.json` revealed **24 of 29 non-citation queries have
different top-5 results**, even though both engines run pure BM25
on the same `BM25FullText_index.pkl`. The differences are mostly adjacent-
rank swaps (positions 3↔4 or 4↔5), consistent with numpy argsort
tie-breaking on equal BM25 scores.

The published `009-new-corpus.json` was scored before the
April-2026 BM25 index rebuild, while my engine loads the current pickle.
The corpus walk and index are bit-identical now, but tied-score
opinions land in slightly different positions due to the underlying
score ordering when the index was built.

**Implication**: per-query deltas on the 29 BM25-only-path queries are
noise. Real semantic differences live in the 30 fusion-path queries.
The headline metrics (averaged over all 65 queries) are largely
unaffected because the tie-break swaps don't systematically favor
either engine, but per-query attribution must filter to fusion-path
only.

Affected example: q021 was nominally a +0.208 d3 "win" — but the
routing path is `bm25_only_noncite`, so d3 didn't contribute. It's a
tie-break artifact in BM25's ordering. Same for q029 (+0.197).

---

## Real per-query findings (fusion-path queries only)

The 30 fusion-path queries are where the embedding model genuinely
contributes. d3-cite vs OpenAI/009 on those:

### Real wins (d3 beats OpenAI on a fusion-path query by ≥ 0.05)

| qid | type | topic | OA → d3 | Δ | query (truncated) |
|---|---|---|---|---:|---|
| q038 | keyw | campaign_finance | 0.000 → 0.485 | +0.485 | 85304 transfer ban candidate controlled committee slate mailer |
| q056 | keyw | other | 0.170 → 0.515 | +0.345 | section 89001 government newsletter mass mailing public expense |
| q049 | keyw | gifts_honoraria | 0.470 → 0.718 | +0.248 | 89518 campaign funds candidate self-compensation consulting |
| q043 | natu | campaign_finance | 0.509 → 0.699 | +0.190 | broad based political committee 85102(d) contribution limits |
| q036 | fact | campaign_finance | 0.553 → 0.723 | +0.170 | engineering firm subcontractor 84308 |
| q055 | fact | lobbying | 0.773 → 0.930 | +0.156 | lobbying firm school district SB100 |
| q044 | keyw | gifts_honoraria | 0.424 → 0.555 | +0.131 | 89501 89502 honorarium ban designated employee op-ed |
| q040 | fact | campaign_finance | 0.619 → 0.699 | +0.080 | central committee 85312 member communications |
| q039 | natu | campaign_finance | 0.844 → 0.922 | +0.078 | candidate-controlled committee Reed v FPPC 85501 |
| q061 | keyw | other | 0.485 → 0.553 | +0.068 | 82041 local government agency Siegel four-factor |

**Pattern**: 10 fusion-path wins, mostly on **campaign_finance** (6 of
10), with one each on lobbying, gifts_honoraria, and "other". Both
keyword and natural-language types win cleanly. **No COI wins on the
fusion path.**

### Real losses (d3 loses on a fusion-path query by ≥ 0.05)

| qid | type | topic | OA → d3 | Δ | query (truncated) |
|---|---|---|---|---:|---|
| **q001** | keyw | conflicts_of_interest | 0.339 → 0.000 | **−0.339** | 87103(a) disqualification business entity investment |
| **q018** | keyw | conflicts_of_interest | 0.339 → 0.000 | **−0.339** | post-employment revolving door 87400 former official |
| q046 | keyw | gifts_honoraria | 0.740 → 0.544 | −0.196 | 89503 gift limit exemption part-time governing board |
| q048 | natu | gifts_honoraria | 1.000 → 0.830 | −0.170 | 89506(a)(2) travel payments foreign government |
| q041 | keyw | campaign_finance | 0.956 → 0.815 | −0.140 | 84224 behested payment $5000 threshold |
| q012 | keyw | conflicts_of_interest | 0.229 → 0.140 | −0.089 | 87103(e) personal financial effect |
| q054 | keyw | lobbying | 0.670 → 0.597 | −0.073 | placement agent contingency fee 86205(f) |

**Pattern**: 7 fusion-path losses, **dominated by keyword queries** (6
of 7). The two −0.339 losses (q001, q018) are catastrophic and account
for **0.678 / 30 ≈ 0.023 of the fusion-path nDCG@5 deficit** — nearly
all of d3's remaining gap to OpenAI is in these two queries.

---

## The key diagnostic: q001 and q018 are not fine-tune regressions

Looking at the top-10 results side-by-side on q001 (`Section 87103(a)
disqualification business entity investment`):

| Rank | OpenAI/009 | Snow-base | f2 | d3 |
|---:|---|---|---|---|
| 1 | **92-657** (gold-2) | 82A167 | 82A167 | 82A167 |
| 2 | 93-175 | 95-317 | 95-317 | 78ADV-78-303 |
| 3 | 78ADV-78-303 | 76ADV-214 | 78ADV-78-303 | 95-317 |
| 4 | 94-138 | 05-039 | 83A144 | 05-039 |
| 5 | 92-620 | 08-149 | 05-039 | 76196 |
| 6 | 96-031 | 93-286 | 76ADV-214 | 89-409 |
| 7 | 95-317 | 94-038 | 78ADV-78-304 | 93-286 |
| 8 | 78ADV-78-304 | 83A144 | 76126 | 08-149 |
| 9 | 05-039 | 09-031 | 93-286 | 76ADV-214 |
| 10 | 92-123(a) | 89-386 | 08-149 | 83A144 |

**None of the 22 gold-relevant opinions for q001 appear in any
Snowflake-family top-10.** Not f2's, not d3's, not the untuned
Snow-base's. OpenAI finds `92-657` at rank 1 — Snowflake doesn't
find it at all.

q018 (`post-employment revolving door … Section 87400 former official`)
tells a similar story: OpenAI puts `00-035` (gold-2) at rank 1; d3
and f2 surface it at rank 8 (just out of top-5, hence nDCG@5 = 0);
Snow-base doesn't find it in top-10 at all.

This is a **Snowflake-vs-OpenAI architectural difference on
statute-keyword queries**, not a fine-tune regression. Whatever lets
OpenAI's embedding rank `92-657` and `00-035` highly for these queries
(plausibly: better BPE-level handling of `87103(a)`-style tokens,
broader pretraining corpus exposure to legal/statute text) is not
something d3's current fine-tune recipe can recover.

The smaller fusion-path losses (q046, q048, q041, q012, q054) are
real but modest — every Snowflake variant ranks the correct opinion
1–3 positions lower than OpenAI does, costing a few hundredths of
nDCG@5 each.

---

## Ensemble (d3 + f2) is not a meaningful lever

- mean nDCG@5 (d3 best, cite-only): 0.3911
- mean nDCG@5 (f2 best, cite-only): 0.3893
- mean nDCG@5 (oracle per-query MAX of d3 and f2): **0.3938**

The oracle upper bound for any d3+f2 ensemble is +0.003 above d3 alone.
Only 2 queries differ by ≥ 0.10 nDCG@5 between the two models. The
LoRA's "different shape" we saw on semantic-only (gifts +0.039 vs d3)
mostly washed out in the hybrid because the fusion + 60% BM25 weight
dominates. **Ensemble not worth pursuing**.

---

## Sprint 4 — concrete lever options

The remaining gap to OpenAI is **almost entirely q001 + q018** —
two statute-keyword COI queries where Snowflake's bi-encoder
architecture simply doesn't surface the relevant opinions, and the
fusion engine can't repair what's not in the candidate pool.

Three Sprint 4 levers are plausible. Each fixes a different piece of
the diagnosis:

### Lever 1 — Continued MLM pretraining of Snowflake on the FPPC corpus

**Goal**: teach Snowflake the FPPC-specific statute-token semantics it
doesn't have. The bet: OpenAI's advantage on q001/q018 comes from its
much-larger pretraining corpus seeing more legal text; closing that
gap via domain-adaptive MLM should let Snowflake learn that
"Section 87103(a)" co-occurs with "92-657" et al.

**Cost**: 1–2 days. Standard MLM on `content.full_text` of the
training-pool opinions (excluding held-outs). Likely 1–3 epochs at
LR ~1e-5. Produces a Snowflake-FPPC checkpoint we then run the
Sprint 3 winning recipe (MNRL, `pos_conclusion_only`, LR=1e-6,
1 epoch) on top of.

**Falsifiable prediction**: q001 and q018 should put at least one
score-2 opinion in their top-10 if MLM works. If they don't move, the
issue isn't pretraining-corpus exposure.

**Risk**: 568M MLM on Strix Halo iGPU is non-trivial; we may need to
adjust batch size / sequence length. Sprint 3 showed bf16 + seq_len=512
fits batch=16, so MLM should fit similarly.

### Lever 2 — Cross-encoder reranker as a final stage

**Goal**: post-process the fusion's top-20 with a strong cross-encoder
(BGE-reranker-large or similar). The bet: cross-attention between
query and candidate text is the only architecture that reliably handles
statute-specific keyword queries; bi-encoders can't.

**Cost**: half a day to wire up. ~20 (query, doc) pairs per query × 65
queries = 1,300 cross-encoder calls. Sub-second per query on the
Strix Halo.

**Falsifiable prediction**: reranker should rescue q001 and q018 if
its top-10 candidate set (from d3's fusion) contains the gold answer.
Need to verify the gold opinions are in d3's top-20.

**Risk**: if d3's top-20 for q001 doesn't contain `92-657`, the
reranker can't surface it. Need to enlarge the candidate pool first
(e.g., top-100 → reranker → top-10).

### Lever 3 — MarginMSE distillation from a cross-encoder teacher

**Goal**: distill cross-encoder relevance into d3 itself via continuous
relevance scores (MSE loss on teacher scores), replacing MNRL's binary
positive/negative. The structural fix for the false-negative-pollution
mechanism Sprint 3 identified.

**Cost**: 2–3 days. Need (a) a teacher (BGE-reranker-large) that
produces continuous scores for ~10k (query, candidate) pairs; (b) a
new training script with MarginMSE loss; (c) retrain d3.

**Falsifiable prediction**: should help on COI queries where
near-relevant opinions cluster (the s3-g1 disaster scenario, but
with continuous scores rather than push-away).

**Risk**: higher complexity. The teacher's calibration matters; bad
scores → bad student.

---

## Recommendation

Run **Lever 1 (continued MLM pretraining)** first. It's the
lowest-cost lever and directly tests the leading hypothesis — that
OpenAI's lead on q001/q018 is a pretraining-corpus exposure
difference. If it doesn't move those two queries, the bi-encoder vs
cross-encoder gap is the real issue and we should move to Lever 2.

Lever 3 (MarginMSE) is the highest-leverage but also highest-cost
move. Worth pursuing only if Lever 1 partially works (signals the
mechanism is right but needs sharper contrastive signal).

### Pre-Lever-1 check: where are the gold answers in d3's semantic ranking?

Ran the rank check (`scripts/check_gold_ranks.py`). For each of d3's
biggest fusion-path losses, here's where the gold answers actually sit
in d3's full-corpus semantic ranking (over 14,096 opinions):

| qid | best gold-2 rank | gold-2 in top-100 | gold-any in top-100 | gold-any in top-1000 | Diagnosis |
|---|---:|---:|---:|---:|---|
| **q001** | **137** | 0/12 | 0/22 | 9/22 | **Architecturally broken** — no gold reaches the fusion pool. |
| **q018** | **1** | 2/6 | 3/12 | 12/12 | **Fusion-misweighted** — gold is rank-1 semantically, but BM25 leg buries it to hybrid-rank-8. |
| q012 | 73 | 1/4 | 1/11 | 3/11 | Marginal — gold at boundary of top-100. |
| q046 | 1 | 3/4 | 6/11 | 10/11 | **Fusion-misweighted** — d3 finds 3 of 4 gold-2 in top-25 semantically; hybrid pushes one to rank 3. |
| q048 | 3 | 7/7 | 12/16 | 16/16 | **Fusion-misweighted** — all 7 gold-2 in top-10 semantically; hybrid scrambles ordering. |
| q041 | 6 | 4/4 | 10/13 | 11/13 | **Fusion-misweighted** — strong semantic ranking, modest hybrid drift. |
| q054 | 1 | 3/5 | 8/11 | 9/11 | **Fusion-misweighted** — d3 ranks one score-2 at #1, hybrid keeps it but loses ordering on others. |

**This is the single most important finding of Sprint 5.**

Only **q001** is a true architectural failure: d3's semantic doesn't
even have any gold answer in its top-100 candidates over the full
corpus. No reranker, no fusion-knob change, no routing rule can recover
q001 — the bi-encoder isn't surfacing the relevant opinions at all.

**Every other loss is a fusion-misweighting**, not a model failure.
d3 already ranks the gold answers highly in pure semantic — q018 has
the gold-2 at semantic rank 1, q046 at rank 1, q048 at rank 3, q041
at rank 6 — but the fusion math at `w_sem=0.40` pushes them down
because BM25 in the citation pool ranks other (non-gold but
high-statute-overlap) opinions higher.

### Revised Sprint 4 lever read

**Lever 0 — per-query or per-topic adaptive w_sem** (newly visible
from the gold-rank check):

For queries where d3's semantic confidently ranks the gold at top-K,
we want to lean *more* on semantic, not less. Conversely, when BM25 is
the discriminator (e.g., q036, q055 wins), we want the current weight.
A heuristic that detects "high semantic confidence" (e.g., the gap
between top-1 cosine and the in-pool mean) and raises `w_sem` for
those queries could pick up several of the q018-class losses.

**Cost**: a few hours to implement and sweep.
**Expected lift**: up to ~+0.01 nDCG@5 if we recover q018 (+0.339)
and q046 (+0.196). q001 is unaffected — still a hard architecture
problem.

**Lever 1 — Continued MLM pretraining of Snowflake on FPPC corpus**

Now revealed as the *only* lever that could fix q001 specifically.
Cost-benefit:
- Cost: 1–2 days.
- Expected lift: +0.339 on q001 alone if MLM moves any gold into
  top-100 semantically; this single query is worth +0.011 of overall
  nDCG@5.
- Combined with retune, plausibly takes us past the SPEC stretch
  target of 0.42.

**Lever 2 — Cross-encoder reranker** is now down-weighted: it requires
the gold to be in the candidate pool, and q001's golds are at ranks
137+. We'd need to deepen the fusion pool from top-100 to top-200+,
which doubles reranker cost per query but is still cheap (1,300 → 2,600
calls). However, Lever 0 fixes most of what's currently fixable
without any architecture change.

**Lever 3 — MarginMSE distillation** is the highest-cost, highest-
leverage lever. Could close the q001 gap if the teacher (a cross-
encoder) ranks `92-657` highly for q001 and the distillation transfers
that to d3.

### Recommendation (updated)

1. **Test Lever 0 first** (a few hours). It's the cheapest experiment
   and addresses ~6 of the 7 fusion-path losses. Concrete plan:
   - Implement query-type-conditional weights (`w_sem_keyword=0.30,
     w_sem_natlang=0.50, w_sem_factpattern=0.60`), OR
   - Implement a per-query semantic-confidence heuristic that raises
     `w_sem` when d3's top-1 cosine is far above pool mean.
   - Sweep, see which queries move where.

2. **If Lever 0 plateaus below 0.42, run Lever 1** (continued MLM).
   It's the only way to crack q001. ~1–2 days of work.

3. **Skip Lever 2** unless after Lever 1 we still have q001-style
   failures with golds present in (a deeper) top-K. The deep-pool
   reranker is more architectural change than the lift currently
   justifies.

4. **Defer Lever 3 (MarginMSE)** to a follow-on project. The
   complexity isn't worth it unless we're going to ship a v2 model.

---

## Lever 0 results (2026-05-23)

Both Variant A (per-query-type weights) and Variant B
(semantic-confidence boost) **failed to improve on flat w_sem=0.40
cite-only at 0.3911 nDCG@5**. Best cite-only variant: 0.3902 (within
noise). Best allpath variant: 0.3975 (tied with the no-boost allpath
baseline, meaning the boost didn't change anything).

### Variant A — per-type weights (n=10 trials, `sprint5d_d3_pertype`)

Sweeping `(w_sem_keyword, w_sem_natlang, w_sem_factpattern)` over
representative grid; best variants:

| Variant | nDCG@5 | MRR | Notes |
|---|---:|---:|---|
| flat040 baseline | **0.3911** | 0.6726 | (control) |
| kw40_nl40_fp70 | 0.3902 | 0.6726 | Boost only fact_pattern — within noise |
| kw40_nl55_fp60 | 0.3869 | 0.6726 | Mild lift on NL+FP — slight regression |
| kw30_nl60_fp70 | 0.3845 | 0.6702 | More aggressive on NL+FP — worse |
| kw50_nl40_fp40 | 0.3785 | 0.6566 | Reverse: lean BM25 on keyword — worst |
| fp70_else40 + allpath | 0.3954 | 0.6548 | Allpath variant — below baseline allpath (0.3975) |

**Conclusion**: per-type is too coarse. Within each type, queries
disagree about which weight they want. q018 (keyword, needs more
semantic) and q036 (keyword, BM25-lead win) require opposite
treatment. Type doesn't separate them.

### Variant B — semantic confidence boost (n=11 trials, `sprint5d_d3_confboost`)

Implemented as "if top-1 semantic cosine in the candidate pool
exceeds T1 (optionally with `top1 - mean ≥ T2`), set `w_sem = boost`."
Sweeping (T1 ∈ {0.65, 0.70, 0.75}, T2 ∈ {None, 0.10, 0.15},
boost ∈ {0.6, 0.7, 0.8}). All cite-only variants land at 0.3854–0.3911.

The smoking gun is in the per-query diagnostics:

| qid | top1 cos | gap to mean | gold rank in d3 sem | what we want |
|---|---:|---:|---:|---|
| **q001** | 0.594 | **0.532** | 137 (none in top-100) | DO NOT boost — semantic is confidently *wrong* |
| **q018** | 0.596 | 0.279 | 1 (right answer) | DO boost — semantic is confidently *right* |
| q012 | 0.512 | 0.462 | 73 | mixed |
| q046 | 0.553 | 0.390 | 1 (right answer) | DO boost |
| q048 | 0.673 | 0.291 | 3 (right answer) | DO boost (only one that triggered) |

**q001 has the largest semantic gap of any COI query but is
confidently wrong.** No scalar function of `(top1, gap, query_type)`
discriminates "confidently right" from "confidently wrong" for these
two queries — they look indistinguishable by any of these signals.

The "confident-right" queries (q018, q046, q048) have lower gaps than
the "confident-wrong" queries (q001). So a "lift when gap is high"
rule would actively misfire on q001.

### Why Lever 0 fails

The flat fusion at w_sem=0.40 turns out to be the right answer because
it gives ENOUGH weight to d3's strong queries (q018-class) to make
their semantic top-3-or-better land in the hybrid top-5, while
NOT giving enough weight to d3's wrong-confidence queries (q001) to
let them swamp BM25's contribution.

Any single-knob lift hurts q001-class (and similar) by the same amount
it helps q018-class. The net is approximately zero.

**The remaining COI deficit is genuinely architectural, not
fusion-tunable.** Closing it requires either (a) fixing d3's semantic
on q001-class queries (Lever 1: continued MLM), or (b) introducing a
new signal that distinguishes right from wrong on these queries
(Lever 2: cross-encoder reranker, or Lever 3: MarginMSE distillation).

### Updated recommendation

**Skip further fusion-knob work. Proceed to Lever 1 (continued MLM
pretraining of Snowflake on the FPPC corpus).** This is the
lowest-cost lever that can change d3's behavior on q001-class queries.
Estimated 1–2 days of work; falsifiable prediction is that q001's
gold-2 opinions move from rank 137+ into d3's top-100 semantic.

Current ship candidate stays at **d3 + w=0.40 + cb=1.5 (cite-routing)
or + cb=2.0 (allpath)** depending on whether the +0.022 nDCG@5
(allpath) is worth the −0.007 MRR.
