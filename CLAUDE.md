# Project: fppc-tuned-embeddings

Experimental project to fine-tune an embedding model for retrieval over the
California FPPC advisory-opinion corpus, with the hypothesis that the existing
(question, opinion) structure in the corpus is unusually clean training data
for contrastive learning.

## Sibling repos (assumed peer paths)

The four upstream repos this project depends on:

| Repo | Role |
|---|---|
| `../fppc-opinions-corpus` | The 14,096 OCR'd opinions, one JSON per opinion under `data/extracted/<year>/<id>.json`. |
| `../fppc-opinions-eval`   | Eval harness: 65 graded queries in `eval/dataset.json`, scorer in `src/scorer.py`. |
| `../fppc-opinions-search-lab` | Search experiments; produced the published baselines and the OpenAI embedding index at `indexes/embeddings_text-embedding-3-small_qa_text.pkl`. |
| `../fppc-opinions-app` | Production app; runs the search-lab's winning Experiment 009. |

Scripts in this repo expect these to live at sibling paths under
`/home/<user>/Projects/`. Adjust the path constants at the top of each script
if cloning elsewhere.

## Key facts to remember

- **Corpus**: 14,096 opinions, each with `embedding.qa_text` (question +
  conclusion combined). ~10,218 have a real extracted `sections.question`;
  1,207 more have only `sections.question_synthetic` (LLM-generated);
  2,670 have neither.
- **Eval set**: 65 queries, 877 graded judgments (score 0/1/2). Scored with
  MRR, nDCG@5/10, Precision@5/10, Recall@10/20.
- **Baselines (published, search-lab)**:
  - Pure semantic (OpenAI `text-embedding-3-small`, qa_text): nDCG@5=0.262, MRR=0.485
  - BM25 on full_text: nDCG@5=0.358
  - Winning hybrid (Experiment 009, CitationScoreFusion): nDCG@5=0.387, MRR=0.684
- **Eval leakage**: 596 opinions (4.23% of corpus) are referenced by eval
  judgments. **Held-out rule**: never use their text or questions for
  training. See `notes/check1_leakage_report.md` for the full analysis.

## Working conventions

- Python venv lives at `.venv/`; not committed.
- Generated artifacts (`data/indexes/`, `logs_*.txt`, `.venv/`) are
  gitignored; only scripts, results JSONs, and notes are versioned.
- Use the eval harness's metric functions directly
  (`src.scorer.compute_mrr` etc., importing from `../fppc-opinions-eval`)
  so numbers stay apples-to-apples with the published baselines.
- For each open embedding model, respect its recommended query/doc prefix
  (see `MODELS` in `scripts/check2_open_baselines.py`).

## Repo layout

```
scripts/             one-off and pipeline scripts
results/             metric JSONs (committed; small)
data/indexes/        cached embedding matrices (gitignored; large)
notes/               human-readable writeups
SPEC.md              what we're building and why
IMPLEMENTATION_PLAN.md   sprint-level plan
```
