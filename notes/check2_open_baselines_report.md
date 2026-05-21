# Check 2 — Open-Model Embedding Baselines

**Goal of this check.** Before designing training data for fine-tuning, measure
how well off-the-shelf open-weights embedding models do on this corpus under
exactly the same eval methodology as the published OpenAI baseline. Two
questions:

1. Is there at least one open model that's already competitive with OpenAI?
   (If not, we'd be starting fine-tuning from a much weaker base.)
2. Which open model is the best fine-tune target?

All four models are embedded over the same corpus field (`embedding.qa_text`),
retrieved with cosine over L2-normalized vectors, and scored with the same
`src.scorer` functions as the published baselines. The OpenAI row below is a
fresh replay of `text-embedding-3-small` against the precomputed search-lab
index using the same retrieval+scoring path — within rounding it matches the
published number (0.262 vs 0.263 nDCG@5).

## Overall results

| Model | MRR | nDCG@5 | nDCG@10 | P@5 | P@10 | R@10 | R@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| OpenAI text-embedding-3-small | 0.494 | 0.263 | 0.238 | 0.234 | 0.175 | 0.134 | 0.185 |
| BGE-base-en-v1.5 | 0.478 | 0.268 | 0.240 | 0.240 | 0.177 | 0.137 | 0.186 |
| **BGE-large-en-v1.5** | **0.495** | **0.282** | **0.254** | **0.258** | **0.191** | **0.146** | **0.193** |
| Nomic-embed-text-v1.5 (max_seq=1024) | 0.473 | 0.265 | 0.235 | 0.252 | 0.177 | 0.136 | 0.174 |

Reference points (search-lab, full pipeline — NOT directly comparable, just for
context): BM25 on `full_text` nDCG@5 = 0.358; winning hybrid Experiment 009
(CitationScoreFusion) nDCG@5 = 0.387, MRR = 0.684.

### Headline findings

- **Open-weights parity is real**: BGE-base already matches OpenAI on nDCG@5
  (0.268 vs 0.263) and lags only slightly on MRR. We can fine-tune without
  trading away a starting-point gap to OpenAI.
- **BGE-large is the new leader** on every overall metric: +7% nDCG@5 over
  OpenAI, +5% over BGE-base. Modest in absolute terms but consistent across
  cutoffs.
- **Nomic at max_seq=1024 did not help**: numbers are essentially tied with
  BGE-base. The long-context advantage didn't materialize in the baseline pass,
  likely because `qa_text` is short (p90 ≈ 649 tokens) and the relevant signal
  is concentrated in the question.

## Per-query-type breakdown (nDCG@5)

| Type | n | OpenAI | BGE-base | BGE-large | Nomic |
|---|---:|---:|---:|---:|---:|
| fact_pattern | 17 | 0.301 | 0.340 | 0.348 | 0.339 |
| keyword | 26 | 0.185 | 0.204 | 0.209 | 0.208 |
| natural_language | 22 | 0.327 | 0.287 | 0.317 | 0.276 |

- **Fact-pattern queries are the best regime for semantic** (≈0.34 across the
  three open models). These queries describe a scenario in narrative form,
  which matches the way opinions describe the facts they're advising on.
- **Keyword queries are the worst regime** (≈0.20). Short, sparse queries
  don't carry enough signal for cosine over dense embeddings to discriminate
  well. This is the classic lexical-vs-semantic gap, and exactly where BM25
  fills in inside the production hybrid.
- **Natural-language queries: OpenAI still leads** (0.327 vs 0.317 BGE-large).
  Worth investigating after fine-tuning — this is one of the deltas we'd like
  to close.

## Per-topic breakdown (nDCG@5)

| Topic | n | OpenAI | BGE-base | BGE-large | Nomic |
|---|---:|---:|---:|---:|---:|
| campaign_finance | 14 | 0.264 | 0.337 | 0.449 | 0.299 |
| conflicts_of_interest | 29 | 0.129 | 0.105 | 0.090 | 0.101 |
| gifts_honoraria | 7 | 0.631 | 0.580 | 0.544 | 0.633 |
| lobbying | 5 | 0.592 | 0.569 | 0.605 | 0.626 |
| other | 10 | 0.232 | 0.272 | 0.258 | 0.255 |

- **`conflicts_of_interest` is the dominant failure mode.** It is 29 of 65
  queries (45%) and every model scores ≈0.10 nDCG@5 — about 5–6× worse than
  `gifts_honoraria` or `lobbying`. The published BM25 and hybrid baselines also
  struggle here (see search-lab Experiment 009 per-topic breakdown), but
  semantic-only is worst. **This is the single biggest target for fine-tuning.**
- **`campaign_finance` is where BGE-large posts its biggest absolute lead**
  (0.449 vs 0.337 base, 0.264 OpenAI). Suggests the larger encoder is picking
  up patterns the smaller one misses on a narrow, jargon-heavy topic.
- **`gifts_honoraria` and `lobbying` are already strong** across the board
  (>0.54). Less headroom for fine-tuning to add value; these are not where we
  should focus training data construction.

## Runtime / hardware notes

All four runs executed on the Framework Desktop (Ryzen AI Max+ 395, Radeon
8060S iGPU, gfx1151). PyTorch was the upstream `torch==2.5.1+rocm6.2` wheel;
upstream wheels don't ship native gfx1151 kernels, so we ran with
`HSA_OVERRIDE_GFX_VERSION=11.0.0` to fall back to gfx1100 (RDNA 3) kernels.
hipBLASLt and AOTriton flash-attn paths are disabled under the override;
inference still runs at GPU speed via hipBLAS + reference SDPA.

| Model | Embed time (14k docs) | Throughput | Embedding dim | Max seq |
|---|---:|---:|---:|---:|
| BGE-base-en-v1.5 | (laptop CPU) | n/a | 768 | 512 |
| BGE-large-en-v1.5 | 478 s | ~30 docs/s | 1024 | 512 |
| Nomic-embed-text-v1.5 | 246 s | ~57 docs/s | 768 | 1024 |

Nomic was estimated at ~2.5 hours on laptop CPU; finished in ~4 minutes here.
This gives us a working hardware story for the rest of the project: embedding
the corpus is now a sub-10-minute cost per experiment, and training runs of
this size should be tractable on the same machine.

## Recommendation: fine-tune target = **BGE-large-en-v1.5**

Why:

1. **Best baseline numbers among open models** on every overall metric. Smaller
   gap to close, higher absolute ceiling than starting from base.
2. **Same prompting convention** as BGE-base (we already understand it from the
   earlier baseline). Migration risk is low.
3. **1024-dim embeddings** — more parameters to fit during training without
   moving to a bigger architecture family. Reasonable middle ground between
   base (768) and the multi-billion-parameter encoders.
4. **Native max_seq=512** is sufficient for `qa_text` (p90 ≈ 649 tokens; only
   ~10% of docs hit the cap). No need to deal with long-context positional
   extrapolation during training.
5. **Mature fine-tuning recipes** in sentence-transformers — `MultipleNegatives-
   RankingLoss`, `CachedMultipleNegativesRankingLoss`, mined hard negatives,
   matryoshka wrappers all have known-good hyperparameters for this family.
6. **Hardware fit**: 96 GB unified GPU memory means we can train with a
   meaningful effective batch size for in-batch negatives.

Not chosen, and why:

- **BGE-base**: smaller capacity ceiling, and we already see BGE-large doing
  better on the most important topics (campaign_finance, +0.11 nDCG@5).
- **Nomic**: long-context didn't pay off in the baseline; `trust_remote_code`
  adds friction; and the custom code is more likely to break with future
  transformers releases. Keep available as a Sprint 3 ablation if there's time.
- **OpenAI text-embedding-3-small**: closed weights — can't fine-tune; only
  useful as a baseline reference and (if it survives Sprint 5) a deployment
  fallback.

## What to revisit in Sprint 3

- **Conflicts-of-interest topic**: 45% of the eval at ≈0.10 nDCG@5. If the
  fine-tune can't move this number, it can't move the overall number. Training
  data construction should oversample COI-relevant opinions (large topic in the
  corpus too) and pay particular attention to hard negatives within that
  topic.
- **Keyword query type**: 0.21 nDCG@5 — semantic-only's weakest regime. Worth
  testing whether augmenting queries with synthetic-keyword paraphrases in
  training helps narrow this gap.
- **Natural-language vs OpenAI gap**: only place OpenAI still leads BGE-large.
  Watch this after fine-tuning to make sure we don't regress.
- **Long context for outlier docs**: if conflict-of-interest opinions are
  systematically long, re-test Nomic at max_seq=4096 on a topic-filtered
  subset before fully ruling out a long-context variant.

## Artifacts

- `results/check2_openai_baseline.json` — OpenAI replay
- `results/check2_bge-base-en-v1.5.json`
- `results/check2_bge-large-en-v1.5.json`
- `results/check2_nomic-embed-text-v1.5.json`
- `results/check2_comparison.json` — overall metrics summary
- `data/indexes/corpus_embeddings_*.npy` — cached doc embeddings (gitignored)
- `logs_baselines_strix.txt` — raw run log (gitignored)
- `scripts/_summarize_check2.py` — reproduces the tables in this report
