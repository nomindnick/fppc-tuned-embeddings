# Check 1 — Eval-Set Leakage Analysis

**Question:** If we train an embedding model on `(opinion_question → opinion_text)`
pairs, will the eval set be effectively memorized rather than generalized to?

**TL;DR:** There is *some* leakage, concentrated in the natural-language and
fact-pattern eval queries. The simple, safe response — **hold the 596 opinions
referenced by the eval (4.23% of the corpus) out of training entirely** — costs
us almost nothing and cleanly eliminates the concern.

---

## What we measured

For each of the 65 eval queries, we looked at the opinions judged "highly
relevant" (score=2) and compared the eval query text to those opinions'
`sections.question` field (the real extracted question, or the LLM-synthesized
fallback when missing) along three axes:

| Metric | What it captures |
|---|---|
| **Token Jaccard** (symmetric) | Are the two strings near-paraphrases? Sensitive to length parity. |
| **Char 5-gram Jaccard** (symmetric) | Substring-level overlap; catches paraphrases with synonym substitution. |
| **Query-token overlap** (asymmetric) | Of the query's tokens, what fraction appear in the opinion's question? Useful when the query is a short keyword bag and the opinion question is a long sentence. |

Symmetric metrics test "did the eval query come from this opinion's question?"
The asymmetric one tells us "does this opinion's question cover the eval
query's vocabulary?" — which is often just topical relevance, not leakage.

## Headline numbers

| Metric | Median | Mean | p75 | Max |
|---|---|---|---|---|
| Query-token overlap | 0.42 | 0.43 | 0.55 | 1.00 |
| Token Jaccard | 0.15 | 0.15 | 0.19 | 0.50 |
| Char 5-gram Jaccard | 0.12 | 0.12 | 0.15 | 0.37 |

**The symmetric metrics are low.** The eval queries are *not* verbatim copies
or close paraphrases of any single opinion question in the median case. Even
the maximum token Jaccard is only 0.50.

The high asymmetric query-token overlap is mostly an artifact of the *keyword*
query type: a 6-token bag like `"Section 87103(a) disqualification business
entity investment financial interest"` trivially overlaps any on-topic
opinion's much longer question text — that's topical relevance, not leakage.

Broken down by query type:

| Type | n | Median query-overlap | Comment |
|---|---|---|---|
| `keyword` | 26 | 0.50 | High by construction; keywords appear in any relevant doc. Not leakage. |
| `natural_language` | 22 | 0.35 | The real risk area — full sentences. |
| `fact_pattern` | 17 | 0.33 | Long fact stories; overlap by topic vocabulary, not paraphrase. |

## The clear leakage cases

Three eval queries are unambiguously paraphrased from a specific opinion's
question:

- **q037** vs opinion **09-253A**
  - Query: *"Can a county establish a single central filing agency for campaign statements covering both city and county jurisdictions…"*
  - Opinion Q: *"May local jurisdictions establish a single central filing agency to review and process campaign statements for both city and county jurisdictions?"*

- **q033** vs opinion **90-596**
  - Query: *"Can a state senator use campaign funds to pay for a conference on the lobbying process for local public officials in his district…"*
  - Opinion Q: *"May Senator Davis use his campaign funds to pay for the costs of a conference, sponsored by the Senator, on the lobbying process [of] public officials in his district?"*

- **q039** vs opinion **14-148**
  - Query: *"May a candidate-controlled committee make independent expenditures to support other candidates after the court in Reed v. FPPC struck down…"*
  - Opinion Q: *"May a candidate-controlled committee make independent expenditures to support or oppose other candidates given the recent…Charles R. 'Chuck' Reed…decision…"*

These are decisive. Whoever authored the eval set used real opinion questions
as anchors for some natural-language queries. If we train on `(opinion_question
→ opinion_text)` for these three opinions, we hand the model the answer key
for those eval queries.

The same is likely true (less strikingly) for a longer tail of
`natural_language` and `fact_pattern` queries. The conservative move is to
treat *all* eval-referenced opinions as potentially compromised, not just the
ones with very high lexical overlap.

## Other facts worth noting

- **No missing opinions.** Every opinion ID referenced in the eval exists in the
  corpus.
- **596 unique opinions are judged relevant** (score >= 1) across the 65
  queries. That's **4.23% of the 14,095-opinion corpus**.
- **Corpus question coverage**: 10,218 opinions have a real extracted
  `sections.question`; 1,207 more have only a synthetic LLM-generated question;
  2,670 have neither.

## Recommended split rule (for the planning docs)

1. **Held-out (forbidden for training):** the **596 eval-referenced opinions**
   — never use their text, real question, synthetic question, or any derived
   field as a training sample.
2. **Training pool:** the remaining ~13,499 opinions.
3. **Positive-pair source:** of the training pool, prefer the ~9,800-ish that
   have a real `sections.question` (subtracting the intersection with the
   held-out set). Treat synthetic questions as a secondary source — they're
   useful for breadth but they were themselves generated by an LLM and may
   carry stylistic artifacts that hurt generalization to lawyer-typed queries.
4. **Validation slice:** carve a held-in validation set from the training pool
   (e.g., 5% sampled stratified by year) so we have a fast feedback signal
   during training without touching the 65-query eval until the end of each
   run.

## Implications beyond the split

- **Augmentation isn't optional.** Since the symmetric overlap is low, real
  eval queries differ substantially in surface form from the opinion questions
  they're based on. To train a model that handles the gap, we should plan to
  LLM-paraphrase opinion questions into multiple registers (formal, colloquial,
  keyword-bag, fact-pattern) during training data construction.
- **Watch the natural_language slice.** That's where the eval is most
  paraphrased-from-opinions. If post-training metrics jump disproportionately
  on the natural_language subset relative to keyword and fact_pattern, suspect
  residual leakage and investigate.
- **Near-duplicate opinions are a lower-order risk** — even with the 596 held
  out, the model could learn the answer to an eval query by training on a
  *different* opinion that addresses the same fact pattern. We should sanity-check
  by mining nearest-neighbor opinions for each held-out opinion during data
  construction and considering whether to extend the hold-out.

## Artifacts

- `results/check1_leakage.json` — full per-query records (every judgment, every
  similarity score) for further inspection.
- `scripts/check1_leakage.py` — script that produced the report; re-runs in
  ~30 seconds.
