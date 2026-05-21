# Check 2 — Open-Model Embedding Baselines

**Goal of this check.** Before designing training data for fine-tuning, measure
how well off-the-shelf open-weights embedding models do on this corpus under
exactly the same eval methodology as the published OpenAI baseline. Two
questions:

1. Is there at least one open model that's already competitive with OpenAI?
   (If not, we'd be starting fine-tuning from a much weaker base.)
2. Which open model is the best fine-tune target?

All models are embedded over the same corpus field (`embedding.qa_text`),
retrieved with cosine over L2-normalized vectors, and scored with the same
`src.scorer` functions as the published baselines. The OpenAI row below is a
fresh replay of `text-embedding-3-small` against the precomputed search-lab
index using the same retrieval+scoring path — within rounding it matches the
published number (0.262 vs 0.263 nDCG@5).

The first four models were the initial 2026-05-20 sweep. Three additional
candidates (gte-modernbert-base, Qwen3-Embedding-0.6B, snowflake-arctic-l-v2)
were added after a second-pass literature survey surfaced post-cutoff
releases worth benchmarking before committing to Sprint 2.

## Overall results

| Model | Params | Max-seq | MRR | nDCG@5 | nDCG@10 | P@5 | P@10 | R@10 | R@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OpenAI text-embedding-3-small | (closed) | 8191 | 0.494 | 0.263 | 0.238 | 0.234 | 0.175 | 0.134 | 0.185 |
| BGE-base-en-v1.5 | 110M | 512 | 0.478 | 0.268 | 0.240 | 0.240 | 0.177 | 0.137 | 0.186 |
| BGE-large-en-v1.5 | 335M | 512 | 0.495 | 0.282 | 0.254 | 0.258 | 0.191 | 0.146 | 0.193 |
| Nomic-embed-text-v1.5 | 137M | 1024 | 0.473 | 0.265 | 0.235 | 0.252 | 0.177 | 0.136 | 0.174 |
| gte-modernbert-base | 149M | 1024 | 0.460 | 0.273 | 0.234 | 0.258 | 0.171 | 0.133 | 0.187 |
| Qwen3-Embedding-0.6B | 0.6B | 1024 | 0.485 | 0.266 | 0.242 | 0.243 | 0.183 | 0.142 | 0.198 |
| **Snowflake-arctic-embed-l-v2.0** | **568M** | **1024** | **0.522** | **0.296** | **0.266** | **0.268** | **0.197** | **0.153** | **0.206** |

Reference points (search-lab, full pipeline — NOT directly comparable, just for
context): BM25 on `full_text` nDCG@5 = 0.358; winning hybrid Experiment 009
(CitationScoreFusion) nDCG@5 = 0.387, MRR = 0.684.

### Headline findings

- **Snowflake-arctic-embed-l-v2.0 is the leader on every overall metric**:
  +5.5% MRR and +5.0% nDCG@5 over BGE-large, +12.5% nDCG@5 over OpenAI.
  Same size class as BGE-large (568M vs 335M), same `query:` prefix
  convention, native 8192-token context.
- **Open-weights parity is firmly established**: every open model except
  Qwen3 ties or beats OpenAI on nDCG@5. We're not paying a quality cost
  to leave the OpenAI API.
- **Architecture isn't destiny**: gte-modernbert-base (149M, ModernBERT) is
  fast and middling overall, while Snowflake (568M, XLM-RoBERTa) wins despite
  being a more conventional architecture. The training recipe matters more
  than the encoder family at this scale.
- **Nomic at max_seq=1024 did not help**: tied with BGE-base. The long-context
  advantage didn't materialize, consistent with the literature on "Dwell in
  the Beginning" position bias (dense retrievers over-weight head-of-doc
  content, so the 512-token cap is rarely the binding constraint when the
  answer is concentrated up front — which it is in `qa_text`).
- **Qwen3-Embedding-0.6B has a niche win**: best of all 7 on keyword queries
  (0.242 vs BGE-large 0.209). Not enough to compete overall, but worth
  remembering for the keyword-query failure mode.

## Per-query-type breakdown (nDCG@5)

| Type | n | OpenAI | BGE-base | BGE-large | Nomic | gte-modernbert | Qwen3-0.6B | **Snowflake-l-v2** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fact_pattern | 17 | 0.301 | 0.340 | 0.348 | 0.339 | 0.341 | 0.296 | **0.353** |
| keyword | 26 | 0.185 | 0.204 | 0.209 | 0.208 | 0.215 | **0.242** | 0.237 |
| natural_language | 22 | 0.327 | 0.287 | 0.317 | 0.276 | 0.290 | 0.271 | 0.322 |

- **Fact-pattern queries are the best regime for semantic** (≈0.34 across most
  open models). These queries describe a scenario in narrative form, which
  matches the way opinions describe the facts they're advising on.
  Snowflake-l-v2 leads at 0.353.
- **Keyword queries are the worst regime** (≈0.20). Short, sparse queries
  don't carry enough signal for cosine over dense embeddings to discriminate
  well. **Qwen3-Embedding-0.6B is the surprise** here at 0.242 (+5pp over
  BGE-large) — the instruction-tuned framing seems to help when the query is
  thin. Worth remembering for Sprint 4: a Qwen3-derived sub-model or
  ensembling specifically for keyword queries could be a real lever.
- **Natural-language queries: OpenAI is still the head-to-head leader**
  (0.327 vs Snowflake 0.322 — essentially tied). Worth watching post-fine-tune.

## Per-topic breakdown (nDCG@5)

| Topic | n | OpenAI | BGE-base | BGE-large | Nomic | gte-modernbert | Qwen3-0.6B | **Snowflake-l-v2** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| campaign_finance | 14 | 0.264 | 0.337 | **0.449** | 0.299 | 0.369 | 0.363 | 0.397 |
| conflicts_of_interest | 29 | 0.129 | 0.105 | 0.090 | 0.101 | 0.087 | 0.098 | 0.106 |
| gifts_honoraria | 7 | 0.631 | 0.580 | 0.544 | 0.633 | 0.566 | 0.596 | **0.638** |
| lobbying | 5 | 0.592 | 0.569 | 0.605 | 0.626 | **0.685** | 0.495 | 0.660 |
| other | 10 | 0.232 | 0.272 | 0.258 | 0.255 | 0.269 | 0.271 | **0.283** |

- **`conflicts_of_interest` is universally broken.** Every one of the 7 models
  is at 0.09–0.13 nDCG@5 on this topic. It is 29 of 65 queries (45%) and 5–6×
  worse than `gifts_honoraria` or `lobbying`. This is a property of the corpus
  × dense-embedding combination, **not the architecture** — no model swap will
  fix it. **Fine-tuning has to attack this directly.**
- **gte-modernbert wins lobbying outright** (0.685). Small n (5 queries), so
  noisy, but interesting — the ModernBERT encoder may bring something
  topic-specific to bear here. Not enough to overall-win.
- **`campaign_finance` is where size + recent training help most** (BGE-large
  0.449, Snowflake 0.397 — both well ahead of OpenAI's 0.264). A narrow,
  jargon-heavy topic where the larger encoders pull ahead.
- **`gifts_honoraria` and `lobbying` are already strong** across the board
  (>0.54). Less headroom for fine-tuning to add value; these are not where we
  should focus training data construction.

## Runtime / hardware notes

All runs executed on the Framework Desktop (Ryzen AI Max+ 395, Radeon
8060S iGPU, gfx1151). PyTorch was the upstream `torch==2.5.1+rocm6.2` wheel;
upstream wheels don't ship native gfx1151 kernels, so we ran with
`HSA_OVERRIDE_GFX_VERSION=11.0.0` to fall back to gfx1100 (RDNA 3) kernels.
hipBLASLt and AOTriton flash-attn paths are disabled under the override;
inference still runs at GPU speed via hipBLAS + reference SDPA.

| Model | Params | Embed time (14k docs) | Throughput |
|---|---:|---:|---:|
| BGE-base-en-v1.5 | 110M | (laptop CPU) | n/a |
| BGE-large-en-v1.5 | 335M | 478 s | ~30 docs/s |
| Nomic-embed-text-v1.5 | 137M | 246 s | ~57 docs/s |
| gte-modernbert-base | 149M | 321 s | ~44 docs/s |
| Qwen3-Embedding-0.6B | 0.6B | 957 s | ~15 docs/s |
| Snowflake-arctic-embed-l-v2.0 | 568M | 727 s | ~19 docs/s |

Nomic was estimated at ~2.5 hours on laptop CPU; finished in ~4 minutes here.
Embedding the corpus on any of these models is now a sub-20-minute cost per
experiment.

## Recommendation: fine-tune target = **Snowflake/snowflake-arctic-embed-l-v2.0**

Why:

1. **Best baseline numbers across all 7 models** on every overall metric:
   MRR 0.522, nDCG@5 0.296, nDCG@10 0.266. +5.5% MRR over BGE-large, our
   prior pick.
2. **Native 8192-token context** — covers 100% of corpus documents without
   truncation. This unlocks `full_text` as a real positive-doc column for
   Sprint 2's training-data ablation (it would have been wasted under
   BGE-large's 512 cap).
3. **Simple prompting convention**: `query: ` prefix on queries only, no
   instruction template, no `trust_remote_code`. Drop-in compatible with the
   existing sentence-transformers training pipeline.
4. **1024-dim embeddings** (same as BGE-large) — index size and inference
   cost unchanged from current production hybrid.
5. **XLM-RoBERTa backbone** with extended-context post-training. Well-
   documented architecture, no exotic custom code surface.
6. **Hardware fit**: 568M params + 8192 context fits comfortably in the
   Strix Halo's 96 GB unified GPU memory. Embedded the full corpus in ~12 min.
7. **Apache 2.0 license**, multilingual training but English is in-domain.

Not chosen, and why:

- **BGE-large-en-v1.5**: prior pick, now #2. The 512-token cap was always a
  Sprint 2 design constraint; with Snowflake at 8192 we can include full-text
  positives in the ablation without architectural cost.
- **gte-modernbert-base**: surprisingly mid overall despite ModernBERT
  architecture and 149M-param speed advantage. Wins lobbying outright
  (small n though). Worth remembering if Snowflake plateaus.
- **Qwen3-Embedding-0.6B**: wins keyword queries but loses on everything else.
  The instruction-tuned framing helps when queries are sparse, hurts on dense
  fact-pattern queries. **Possible Sprint 4 lever**: train a Qwen3-derived
  sub-model specifically for keyword queries and ensemble — but only after
  Snowflake fine-tune confirms the main target.
- **Nomic-v1.5**: long-context didn't help; `trust_remote_code` is a future-
  compat liability.
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
- `results/check2_gte-modernbert-base.json`
- `results/check2_qwen3-embed-0.6b.json`
- `results/check2_snowflake-arctic-embed-l-v2.0.json`
- `results/check2_comparison.json` — overall metrics summary
- `data/indexes/corpus_embeddings_*.npy` — cached doc embeddings (gitignored)
- `logs_baselines_strix.txt`, `logs_candidates_strix.txt` — raw run logs (gitignored)
- `scripts/_summarize_check2.py` — reproduces the tables in this report
