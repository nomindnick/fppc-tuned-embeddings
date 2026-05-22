# Phase 4 — Second-Opinion Request

**Audience:** A reviewing LLM with no prior context on this project.
**Goal:** We are stuck. Every fine-tuning run we have attempted so far has
*degraded* a strong off-the-shelf embedding model. We want an outside opinion
on (a) the most likely root cause, (b) which of our queued hypotheses is the
right one to chase first, and (c) any approach we have not yet considered.

This document is self-contained. Sections 1–5 give you the project context;
Section 6 is the experimental record where things have gone wrong; Section 7
states explicitly what we want feedback on.

---

## 1. Project purpose

### The corpus and the production search problem

The California Fair Political Practices Commission (FPPC) has issued
**14,096 advisory and formal opinions** since 1975. These opinions
interpret the Political Reform Act and Government Code §§ 1090 et seq.
(conflicts of interest, campaign finance, gifts/honoraria, lobbying, and
related ethics rules) for California public officials. Lawyers, ethics
officers, and agency staff search these opinions when reasoning about
real-world conflicts and disclosure questions.

The corpus has four sibling repositories:

| Repo | Role |
|---|---|
| `fppc-opinions-corpus` | The 14,096 OCR'd opinions, one JSON per opinion. |
| `fppc-opinions-eval` | Eval harness: 65 graded queries, 877 relevance judgments. |
| `fppc-opinions-search-lab` | Search experiments; produced the published baselines. |
| `fppc-opinions-app` | Production app (Experiment 009 hybrid). |
| `fppc-tuned-embeddings` | **This repo** — fine-tuning an embedder for it. |

The **production search engine** today is a hybrid:

- **BM25 on `full_text`** alone scores **nDCG@5 = 0.358** on the 65-query eval.
- **Pure semantic** (OpenAI `text-embedding-3-small`, on the `qa_text`
  field) scores **nDCG@5 = 0.262, MRR = 0.485**.
- The fused hybrid (Experiment 009, CitationScoreFusion) scores
  **nDCG@5 = 0.387, MRR = 0.684**.

The semantic axis is the weaker partner. **The hypothesis driving this
project**: the corpus contains roughly 10,000+ naturally-occurring
`(question, opinion)` pairs — the FPPC writes its opinions in
question/conclusion/analysis form, with the question text at the head
of the document. That is *prima facie* very clean training data for
contrastive embedding fine-tuning. If we can lift the semantic axis from
~0.26 to ~0.36 nDCG@5, the hybrid should also move.

### Success criteria (from `SPEC.md`)

All measured on the 65-query eval set, scored with the same `src.scorer`
functions used for the published baselines.

- **Threshold**: semantic-only nDCG@5 ≥ 0.33 (above every off-the-shelf
  baseline).
- **Goal**: semantic-only nDCG@5 ≥ 0.36 (matches BM25-only).
- **Stretch**: hybrid nDCG@5 ≥ 0.42, MRR ≥ 0.72.
- **Subgroup gate**: `conflicts_of_interest` topic nDCG@5 ≥ 0.20. This
  topic is 29 of 65 queries (45%) and scores **0.09–0.13 across all 7
  off-the-shelf models we benchmarked**. If we cannot move it,
  overall numbers will not move.

---

## 2. Corpus derivation and JSON structure

### Provenance

The FPPC publishes opinions as PDFs. The corpus repo downloaded all of
them, OCR'd them, and ran an LLM-assisted extractor to identify the
"Question / Facts / Analysis / Conclusion" sections that modern opinions
follow. For older opinions where extraction failed, the corpus author
generated a **synthetic question** with an LLM (gpt-style, marked as
`question_synthetic`).

Coverage:

- 14,096 opinion JSON files → 14,094 unique IDs (one duplicate across
  year folders).
- 10,218 opinions have a real extracted `sections.question`.
- 1,207 more have only `sections.question_synthetic`.
- 2,670 have neither (cannot be used to form a training pair).

### Per-opinion JSON schema (the fields we actually use)

```json
{
  "id": "17-004",
  "year": 2017,
  "sections": {
    "question": "1. Can you advise if there is any other way that a business can legally give more than $99.99 without having a checking account?\n\n2. Could an individual, serving as an intermediary, ...",
    "conclusion": "...",
    "facts": "...",
    "analysis": "...",
    "question_synthetic": null,
    "conclusion_synthetic": null
  },
  "embedding": {
    "qa_text": "QUESTION: 1. Can you advise...\n\nCONCLUSION: There is not...",
    "qa_source": "extracted | synthetic | mixed"
  },
  "citations": {
    "government_code": ["87103", "1090", ...],
    "regulations": ["18702.2", ...],
    "prior_opinions": [...],
    "cited_by": [...]
  },
  "classification": {
    "topic_primary": "conflicts_of_interest | campaign_finance | gifts_honoraria | lobbying | other",
    "confidence": 0.5,
    "classification_method": "heuristic:citation_based"
  },
  "content": { "full_text": "..." }
}
```

The single most important field for fine-tuning is `embedding.qa_text` —
the concatenation of `QUESTION:` and `CONCLUSION:` sections. This is
the field the production index encodes today.

### Eval dataset structure

`fppc-opinions-eval/eval/dataset.json` has 65 queries with metadata:

```json
{
  "id": "q001",
  "text": "Section 87103(a) disqualification business entity investment financial interest",
  "type": "keyword | natural_language | fact_pattern",
  "topic": "conflicts_of_interest | campaign_finance | gifts_honoraria | lobbying | other",
  "issue": "business_entity_interest",
  "relevance_judgments": [
    {"opinion_id": "82A155", "score": 2, "rationale": "..."},
    {"opinion_id": "02-232", "score": 2, "rationale": "..."},
    ...
  ]
}
```

877 total judgments (`score ∈ {0,1,2}`) span **624 unique opinions**
(4.43% of the corpus). Scoring uses MRR and nDCG@5/10/P/R with the
graded judgments. The `src.scorer` module is the canonical scoring
path; we import it directly so numbers are apples-to-apples with the
published baselines.

### Eval leakage analysis (Check 1)

Critically, **three eval queries are nearly verbatim paraphrases of
specific opinion questions** (q037↔09-253A, q033↔90-596, q039↔14-148).
The eval author used real opinion questions as inspiration for some
natural-language queries. We confirmed this with token Jaccard / char
5-gram Jaccard / asymmetric query-token overlap analyses
(median symmetric Jaccard = 0.15, max = 0.50). The conservative response:
**hold out all 624 eval-referenced opinions from training entirely** —
text, real question, synthetic question, any derived field. 4.43% of
training data lost; leakage cleanly eliminated.

---

## 3. How we chose the embedding model

### The 7-model bake-off (Check 2)

We benchmarked every plausible open-weights candidate plus the OpenAI
baseline on the exact eval methodology (cosine over L2-normalized
embeddings of `qa_text`, scored with `src.scorer`).

| Model | Params | Max-seq | MRR | nDCG@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| OpenAI text-embedding-3-small | (closed) | 8191 | 0.494 | 0.263 | 0.238 |
| BGE-base-en-v1.5 | 110M | 512 | 0.478 | 0.268 | 0.240 |
| BGE-large-en-v1.5 | 335M | 512 | 0.495 | 0.282 | 0.254 |
| Nomic-embed-text-v1.5 | 137M | 1024 | 0.473 | 0.265 | 0.235 |
| gte-modernbert-base | 149M | 1024 | 0.460 | 0.273 | 0.234 |
| Qwen3-Embedding-0.6B | 0.6B | 1024 | 0.485 | 0.266 | 0.242 |
| **Snowflake-arctic-embed-l-v2.0** | **568M** | **8192** | **0.522** | **0.296** | **0.266** |

**Per-topic nDCG@5** (the part that matters):

| Topic | n | OpenAI | BGE-base | BGE-large | Nomic | gte-MB | Qwen3 | **Snowflake** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| campaign_finance | 14 | 0.264 | 0.337 | **0.449** | 0.299 | 0.369 | 0.363 | 0.397 |
| conflicts_of_interest | 29 | 0.129 | 0.105 | 0.090 | 0.101 | 0.087 | 0.098 | 0.106 |
| gifts_honoraria | 7 | 0.631 | 0.580 | 0.544 | 0.633 | 0.566 | 0.596 | **0.638** |
| lobbying | 5 | 0.592 | 0.569 | 0.605 | 0.626 | **0.685** | 0.495 | 0.660 |
| other | 10 | 0.232 | 0.272 | 0.258 | 0.255 | 0.269 | 0.271 | **0.283** |

**Selected model: `Snowflake/snowflake-arctic-embed-l-v2.0`**

Reasons:

1. Best baseline numbers across all 7 models on every overall metric.
2. Native **8192-token context** — covers 100% of corpus opinions
   without truncation. Lets us ablate `qa_text` vs `body` vs `full_text`
   as positive-doc choices without architectural constraint.
3. Simple prompting: `query: ` prefix on queries only; no instruction
   template; no `trust_remote_code`.
4. 1024-dim output — same as BGE-large and the production OpenAI index,
   so a future deployment swap doesn't change index size.
5. XLM-RoBERTa backbone, Apache 2.0, mature in sentence-transformers.
6. Fits comfortably on our hardware (Strix Halo iGPU, 96 GB unified).

**Important observation across all 7 models**: `conflicts_of_interest`
is universally broken (0.09–0.13 nDCG@5). This is a corpus × dense-
embedding interaction, not an architectural failure. Fine-tuning *has*
to attack this topic specifically.

---

## 4. Training data construction

### Held-out enforcement

All **624 eval-referenced opinions** (anything a human judged, including
score=0 negatives) are excluded from training. The pair builder enforces
this at row-emit time.

### Pair file schema (`pairs.jsonl`, 10,806 rows)

One row per `(question, opinion)` pair where the opinion is not held
out and has at least one usable question text.

| Field | Meaning |
|---|---|
| `opinion_id`, `year` | Identity |
| `question` | Normalized query text (strip leading `1.`, `QUESTION:`, etc.; drop <15-char results) |
| `question_source` | `"real"` or `"synthetic"` |
| `topic_primary` | Corpus-assigned topic |
| `pos_qa_text` | `embedding.qa_text` — always present |
| `pos_body` | `facts + analysis + conclusion`, joined — null when analysis missing (~31.5% of rows) |
| `pos_qa_plus_body` | `qa_text + body` concat — null when `pos_body` null |
| `pos_full_text` | `content.full_text` — always present |
| `has_real_question`, `has_synth_question`, ... | Coverage flags |

Four positive-doc columns were built so Sprint 3 could ablate
`(query → positive_doc)` representations. **Sprint 3 has so far only
used `pos_qa_text` because all runs degraded; we never reached the
ablation stage.**

### Question source preference

When both real and synthetic exist (1,684 opinions), we use the real
one. ~89% of training questions are real; ~11% synthetic.

### Validation slice (`val_slice.jsonl`, 543 rows)

5% of pairs, stratified by year, seed=20260521. Full-schema rows so the
trainer can validate on the same positive-doc column it trains on.
Topic distribution within 3pp of the population.

### Hard negatives (`hard_negatives.jsonl`, 10,263 rows / 95,665 negatives)

For each training-pool opinion we mine two complementary sources:

1. **BM25 top-k**: run the training question through the search-lab's
   cached BM25 index. Take top-5 non-positive corpus opinions.
2. **Same-statute different opinion**: any other corpus opinion whose
   `citations.government_code` set overlaps the positive's, normalized
   (`87103(a)` → `87103`). Rank by shared-code count. Take top-5.

100% of training rows get ≥1 negative; 80% get the full 10. The two
sources only overlap on 1.2% of records — they surface different
distractors. BM25 captures keyword similarity; same-statute captures
co-citation. Median same-statute negative shares 4 base codes with the
positive (strong overlap signal).

### Training pool topic distribution

| Topic | Pairs | % |
|---|---:|---:|
| conflicts_of_interest | 6,104 | **56.5%** |
| campaign_finance | 1,999 | 18.5% |
| other | 792 | 7.3% |
| gifts_honoraria | 586 | 5.4% |
| lobbying | 117 | 1.1% |
| (unclassified) | 1,208 | 11.2% |

COI is over-represented vs the eval (45%) — naturally COI-heavy, no
explicit oversampling applied.

---

## 5. Phase 3 vs Phase 4 separation

We split fine-tuning into two distinct sprints:

### Phase 3 — Initial fine-tuning sweep

**Question:** *Which (base model × loss × positive-column) configuration
is even worth pushing on?* The deliverable is a winner config that
**beats the relevant baseline** (Snowflake-arctic-l-v2 at 0.296 nDCG@5).
A regression in Phase 3 means we have not found a viable recipe yet;
hyperparameter tuning has nothing to optimize.

The grid (as planned):

- **Base models**: top 1–2 from Check 2 (Snowflake, optionally BGE-large).
- **Losses**: `MultipleNegativesRankingLoss`, `CachedMultipleNegativesRankingLoss`
  (larger global batch via gradient caching), `MNRL + 1 mined hard negative`.
- **Positive columns**: ablate `pos_qa_text` vs `pos_body` vs
  `pos_qa_plus_body` vs `pos_full_text` (Stage B, deferred until Stage A
  produced a non-broken recipe).
- Short runs (1 epoch, ~640 steps), default hyperparameters (LR=2e-5,
  warmup_ratio=0.10, bf16, max_seq=512).
- Score on full 65-query eval; also track an in-loop IR evaluator on
  the val slice.

### Phase 4 — Winner tuning

**Question (only reached if Phase 3 produced a viable recipe):**
*How far can we push it?* Hyperparameter sweep on the Phase 3 winner
(LR, warmup, batch, epochs), loss-wrapper variations (Matryoshka,
hard-negative weight, denoising), longer training, best-checkpoint
selection.

**Why separate them?** Two reasons:

1. **Cost discipline.** Phase 4 hyperparameter sweeps are expensive
   (multiple LRs × multiple epoch counts × multiple seeds). Running
   them on a config that is already worse than baseline burns hardware
   to confirm the obvious.
2. **Diagnosis vs optimization.** Phase 3 answers "is this approach
   even directionally correct?" Phase 4 answers "how good can we make
   the directionally correct approach?" Mixing them obscures
   which knob mattered. If we hand-tune LR while also swapping losses
   we cannot tell what fixed the regression.

**Current status:** Phase 3 has produced 4 fine-tuned models. All 4
*regressed* the base. We have not even reached the Stage B
positive-column ablation, let alone Phase 4.

---

## 6. Experiments to date — Phase 3

**Hardware:** Strix Halo (Ryzen AI Max+ 395, 96 GB unified GPU memory,
gfx1151 iGPU). PyTorch 2.5.1+rocm6.2 with `HSA_OVERRIDE_GFX_VERSION=11.0.0`
to fall back to gfx1100 (RDNA 3) kernels. Inference + training work;
hipBLASLt and AOTriton flash-attn paths are disabled under the override.

**Trainer:** `sentence-transformers` `SentenceTransformerTrainer` with
the standard contrastive losses. Single config JSON per run; trainer is
~400 lines (`scripts/train_sprint3.py`).

**Baselines for comparison** (semantic-only, 65-query eval, full
14k-doc corpus):

| Model | nDCG@5 | MRR | nDCG@10 | COI nDCG@5 |
|---|---:|---:|---:|---:|
| OpenAI text-embedding-3-small (production) | 0.263 | 0.494 | — | — |
| BGE-large-en-v1.5 | 0.282 | 0.495 | — | 0.090 |
| **Snowflake-arctic-l-v2 (fine-tune target)** | **0.296** | **0.522** | 0.266 | **0.106** |

### Stage A results (all on Snowflake-arctic-l-v2)

| Run | Loss | Negatives | LR | Steps | nDCG@5 | MRR | nDCG@10 | COI@5 | Δ vs base |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| s3-a1-mnrl-qa | MNRL | in-batch (bs=16) | 2e-5 | 642/1 | 0.200 | 0.404 | 0.174 | 0.085 | **−0.097** |
| s3-a2-cached-qa | CachedMNRL | in-batch (global bs=128, mini=8) | 2e-5 | 81/1 | 0.184 | 0.381 | 0.160 | 0.080 | **−0.112** |
| s3-a3-mnrl-hn-qa | MNRL + 1 hn | in-batch + 1 mined hn | 2e-5 | 642/1 | 0.085 | 0.198 | 0.078 | 0.050 | **−0.212** |

### Stage C — base-model swap as control

| Run | Base | Loss | nDCG@5 | MRR | COI@5 | Δ vs that base |
|---|---|---|---:|---:|---:|---:|
| s3-c1-bge-mnrl-qa | BGE-large-en-v1.5 | MNRL (same as s3-a1) | 0.181 | 0.396 | 0.099 | **−0.101** vs BGE |

Identical recipe to s3-a1 except `base_model = BAAI/bge-large-en-v1.5`
and `query_prefix = "Represent this sentence for searching relevant
passages: "` (BGE convention).

### What we observed across runs

1. **All runs degrade the base.** Not "one wins by a small margin" —
   every variant is strictly worse on every overall metric and every topic.

2. **Trend is monotonic with negative difficulty.** Random in-batch
   (easiest) → bigger random batch (s3-a2, more in-batch negatives) →
   mined hard negs (hardest). Each step "harder" makes the damage
   *worse*, not better. s3-a3 with one mined hard negative was the
   worst run by a huge margin.

3. **Training loss collapses to ~0.** For s3-a1, training loss dropped
   from ~1.4 to ~3e-6 within ~600 of 642 steps. The model finds a
   near-zero in-batch loss long before training ends.

4. **The val-slice IR evaluator gave no warning signal.** The val
   slice uses the 543 held-in val opinions as both queries and the
   corpus. Accuracy@1 was 0.998 on the very first checkpoint
   (step 100, before significant training). nDCG@5 stayed at 0.999+
   across all eval points. The val slice is **fully saturated** — it
   cannot distinguish a strong base model from a broken fine-tune.

5. **Base-model swap (Stage C) didn't help.** BGE-large + MNRL
   regressed by −0.101 vs BGE baseline; nearly identical magnitude to
   Snowflake + MNRL's −0.097 vs Snowflake baseline. **The recipe is
   the controlling variable, not the base model.**

6. **One small bright spot in s3-c1:** COI subgroup ticked up
   (0.090 → 0.099). The only positive movement in any cell of any
   tuned model. Could be n=29 noise (Δ=+0.009), or could signal that
   the COI distractor structure is just hard enough that even a broken
   recipe scratches it.

### Per-query-type breakdown of the failure (s3-a1, the cleanest run)

| Type | n | Snowflake base | s3-a1 fine-tune | Δ |
|---|---:|---:|---:|---:|
| keyword | 26 | 0.237 | 0.168 | −0.069 |
| natural_language | 22 | 0.322 | 0.201 | −0.121 |
| fact_pattern | 17 | 0.353 | 0.248 | −0.105 |

The damage is largest on natural-language queries (which we expected
fine-tuning to help most, since those resemble the training questions).
Keyword queries — already the worst regime — degrade the least in
absolute terms.

### Per-topic breakdown (s3-a1)

| Topic | n | Snowflake base | s3-a1 | Δ |
|---|---:|---:|---:|---:|
| conflicts_of_interest | 29 | 0.106 | 0.085 | −0.021 |
| campaign_finance | 14 | 0.397 | 0.257 | **−0.140** |
| gifts_honoraria | 7 | 0.638 | 0.308 | **−0.330** |
| lobbying | 5 | 0.660 | 0.492 | −0.168 |
| other | 10 | 0.283 | 0.233 | −0.050 |

`gifts_honoraria` and `lobbying` — the topics the base model was
*already good at* — get destroyed. COI is roughly preserved. This is
suggestive: the strongest base capabilities are washed out the fastest.

### Configs in full (s3-a1, representative)

```json
{
  "base_model": "Snowflake/snowflake-arctic-embed-l-v2.0",
  "loss": "MultipleNegativesRankingLoss",
  "use_hard_negatives": false,
  "positive_column": "pos_qa_text",
  "epochs": 1,
  "batch_size": 16,
  "learning_rate": 2e-5,
  "warmup_ratio": 0.10,
  "max_seq_length": 512,
  "bf16": true,
  "query_prefix": "query: ",
  "doc_prefix": "",
  "seed": 20260521
}
```

s3-a2 differs by `loss=CachedMNRL, batch_size=128, mini_batch_size=8`.
s3-a3 differs by `use_hard_negatives=true, n_hard_negatives_per_row=1`.
s3-c1 differs only by `base_model` and `query_prefix`.

### Active hypotheses (none yet tested)

We have stopped Phase 3 Stage B (positive-column ablation) because
comparing four flavors of "broken" tells us nothing. Queued hypotheses,
in the order we currently weight them:

- **H1 — LR is wrecking the pretrained weights.** 2e-5 is the
  sentence-transformers default. Snowflake-arctic-l-v2 is already a
  strong, saturated general retriever; the standard LR may be 10-100×
  too large for *adapting* it. Proposed: rerun s3-a3 (MNRL+1hn) at
  5e-6 and 1e-6.
- **H2 — Even 1 epoch is too long.** Training loss hit zero by step
  ~600 of 642. The useful learning may all happen in the first 100–200
  steps; the rest is memorization / collapse. Proposed: checkpoint
  every 50 steps and score each on 65-query eval (val slice is
  useless). Pick best checkpoint.
- **H3 — Hard negatives are too aggressive.** Our hard negatives are
  BM25 top-5 + same-statute top-5. For COI especially, the "hard
  negative" may literally be a near-duplicate opinion that a good
  retriever *would* want near the query — pushing it away is a bad
  training signal. Proposed: use BM25 ranks 6–10 instead of 1–5, or
  filter by some similarity ceiling.
- **H4 — Training/eval distribution mismatch.** Training questions
  are FPPC opinion-author paraphrases of conclusions ("May a school
  board member..."). Eval queries include keyword bags, fact patterns,
  and lawyer-style natural-language queries. The model may be learning
  to match a narrow paraphrase style rather than search intent.
  Proposed: LLM-paraphrase 1–2k training questions into search-style
  queries; mix with the original training questions.
- **H5 — Switch base model.** **Refuted by s3-c1.** BGE-large + same
  recipe regressed by an equivalent magnitude.
- **H6 — Document the negative result and stop.** "Naive contrastive
  fine-tuning of strong pretrained embedding models on this corpus
  degrades performance." A publishable negative; we'd rather not.

---

## 7. What we want a second opinion on

We are confident in *some* parts of the diagnosis:

- The val-slice IR evaluator is useless (saturated). We need to score
  on the 65-query eval for in-loop signal.
- The base model is not the controlling variable (Stage C refuted H5).
- The training loss is collapsing to a degenerate optimum.

We are *less* confident about:

1. **Root cause ranking.** Our best guess is that LR (H1) and steps
   (H2) interact: at LR=2e-5 the model can find a trivial
   batch-separating geometry in <200 steps and then over-fits to it
   for the remaining 400+. But this doesn't fully explain the
   monotonic damage trend with negative difficulty (random < big-random
   < mined hard). Why does adding *better* negatives make things
   strictly worse if the LR/steps story is the whole story?

2. **Whether to chase H1, H2, H3, or H4 first.** They are not
   independent. The right ablation order matters a lot for cost; each
   training run is ~50 minutes + ~9 minutes scoring.

3. **Whether our training-data design is itself unsound.** The
   pair-building decisions (use real question when both exist; treat
   `pos_qa_text` as the default positive; mine BM25 top-5 + same-
   statute top-5; held-out the full 624 judged opinions) were defended
   in `notes/training_data_design.md` but never validated empirically.
   It's possible the data is fine and only the recipe is broken; it's
   also possible the data itself drives the collapse (e.g. positive =
   `qa_text` already contains the question text, so the model can
   learn an identity mapping rather than a question→document mapping —
   this is the one we worry about most).

4. **Whether there is an approach we have not considered.** Things we
   have *not* tried:
   - Removing the question text from the positive document before
     training (so positive is `conclusion + body` only — the model
     must learn from semantics, not surface overlap).
   - Distillation from a stronger teacher (e.g., a reranker's scores
     on (query, doc) pairs).
   - GenQ / generative-pseudo-label style data augmentation.
   - Curriculum learning (random negatives first, then hard).
   - Matryoshka loss, denoising objectives, or contrastive masked
     reconstruction.
   - LoRA / partial-parameter fine-tuning instead of full fine-tuning.
   - Continued pretraining on the corpus before contrastive
     fine-tuning.

**Specifically we'd like the reviewer's view on:**

- (a) What is the most likely root cause of the monotonic-with-negative-
  difficulty degradation we are seeing? Is "LR + steps" sufficient to
  explain it, or is there a second mechanism (likely related to
  positive-text leakage or hard-negative pollution) at work?
- (b) Of H1, H2, H3, H4 — or any new hypothesis you propose — what is
  the cheapest, highest-information next experiment? Be concrete:
  exact knob change, expected directional outcome, what would
  falsify it.
- (c) Is there a structural problem with our training data
  construction (positive-doc choice, question normalization, hard-
  negative mining) that we should fix before any more recipe
  experiments?
- (d) Are there published recipes for fine-tuning *strong* off-the-
  shelf embedders (vs the typical "fine-tune BERT into a retriever"
  literature, which starts from a weaker base) that we should be
  mimicking? E.g., specific LR schedules, gradient-cache settings,
  loss temperatures, or in-batch-negative masking strategies.

Brutal feedback is welcome. We'd rather discover the project is
infeasible cheaply than burn another month on doomed sweeps.

---

## Appendix A — Reference files in this repo

- `SPEC.md` — full project spec and success criteria.
- `IMPLEMENTATION_PLAN.md` — sprint-level plan.
- `notes/check1_leakage_report.md` — eval-leakage analysis that
  justifies the 624-opinion held-out.
- `notes/check2_open_baselines_report.md` — 7-model bake-off and
  model-selection rationale.
- `notes/training_data_design.md` — every decision behind
  `pairs.jsonl`, `val_slice.jsonl`, and `hard_negatives.jsonl`.
- `notes/sprint3_experiment_log.md` — running record of Phase 3 runs
  with hypotheses and interpretations.
- `scripts/train_sprint3.py` — single-config trainer
  (`sentence-transformers` based, ~400 lines).
- `scripts/_eval_lib.py` — shared encode/retrieve/score (used by both
  baselines and tuned-model scoring; verified bit-identical to Check 2).
- `results/check2_*.json` — per-baseline scoring results.
- `experiments/s3-*/eval_metrics.json` — per-tuned-model results.

## Appendix B — Sample training row

```json
{
  "opinion_id": "17-004",
  "year": 2017,
  "question": "Can you advise if there is any other way that a business can legally give more than $99.99 without having a checking account? Could an individual, serving as an intermediary, use their own personal checking account to give a donation for a business?",
  "question_source": "real",
  "topic_primary": "campaign_finance",
  "pos_qa_text": "QUESTION: 1. Can you advise if there is any other way that a business can legally give more than $99.99 without having a checking account?\n\n2. Could an individual, serving as an intermediary, use their own personal checking account to give a donation for a business?\n\nCONCLUSION: There is not [a legal way for a business to give more than $99.99 without a checking account]...",
  "pos_body": "FACTS: ...\n\nANALYSIS: ...\n\nCONCLUSION: ...",
  "pos_full_text": "(full opinion text)",
  "has_real_question": true,
  "has_synth_question": false
}
```

Note that the **positive document `pos_qa_text` contains the question
text verbatim** (with a `QUESTION:` prefix and a `CONCLUSION:` suffix).
This is one of our active worries: the training task may collapse to
"match the question token-prefix" rather than "match the conclusion's
legal reasoning." See H4 + open question (c) above.

## Appendix C — Sample eval queries (one per type)

- **keyword (q001):** `"Section 87103(a) disqualification business entity investment financial interest"` — topic: conflicts_of_interest.
- **natural_language (q005):** `"Can a city council member vote on a street widening project if their residence is 370 feet from the project boundary?"` — topic: conflicts_of_interest.
- **fact_pattern (q003):** `"A city council member owns a home within a landscaping and lighting assessment district. The council must vote on the annual levy for the district, which funds maintenance of streetlights, medians, and parks in the area. The assessment would apply to all properties in the district, including the council member's home."` — topic: conflicts_of_interest.

These illustrate the surface-form gap between training questions
(authored-by-FPPC paraphrases of conclusions, often with `QUESTION:`-
prefix structure) and eval queries (keyword bags, hypotheticals,
multi-sentence fact patterns). H4 is the formal statement of this
worry.
