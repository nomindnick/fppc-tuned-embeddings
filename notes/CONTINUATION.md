# Continuation — Resuming on Strix Halo

Snapshot of state when work was paused for hardware switch.

## Where we are

- **Check 1 (leakage)** — ✅ complete. Report: `notes/check1_leakage_report.md`.
  TL;DR: hold out 596 eval-referenced opinions (4.23% of corpus); training
  on the rest is safe.
- **Check 2 (open-model baselines)** — partial.
  - ✅ OpenAI `text-embedding-3-small` replay → `results/check2_openai_baseline.json`
    (nDCG@5 = 0.263, MRR = 0.494 — matches published 0.262 / 0.485 within noise).
  - ✅ `BAAI/bge-base-en-v1.5` (max_seq=512) → `results/check2_bge-base-en-v1.5.json`
    (nDCG@5 = 0.268, MRR = 0.478 — ≈ OpenAI parity).
  - ⏸️ `nomic-ai/nomic-embed-text-v1.5` — killed mid-run; CPU on laptop took
    ~5 min per 32-doc batch at max_seq=1024.
  - ⏸️ `BAAI/bge-large-en-v1.5` — never started; deferred as a scale check.

## Why we paused

Laptop CPU runtime estimate for nomic at max_seq=1024 was ~2.5 hours. Strix
Halo (Ryzen AI Max+ 395, 128GB unified) should run all remaining models in
a fraction of that time — especially if PyTorch can hit the iGPU via ROCm.

## What's in the repo to bring across

Everything under `/home/<user>/Projects/fppc-tuned-embeddings/` except:
- `.venv/` — recreate; environment is machine-specific.
- `data/indexes/` — gitignored; bge-base embedding cache (~43 MB). Optional
  to copy; will regenerate quickly on faster hardware if missing.
- `logs_*.txt` — gitignored throwaway logs.

The four sibling repos under `/home/<user>/Projects/` must also be present:
- `fppc-opinions-corpus`
- `fppc-opinions-eval`
- `fppc-opinions-search-lab`

(The `fppc-opinions-app` repo isn't needed until Sprint 5.)

## Resume steps on Strix Halo

1. Confirm sibling repos exist at expected paths (see `CLAUDE.md`). If your
   username differs, update path constants at the top of
   `scripts/check2_open_baselines.py` (`EVAL_PATH`, `CORPUS_DIR`, and the
   OpenAI index path inside `evaluate_openai_baseline_from_cached`).

2. Create a fresh venv. The laptop used PyTorch CPU build; on Strix Halo
   try ROCm wheels first since the Radeon 8060S iGPU should be supported:

   ```bash
   cd /home/<user>/Projects/fppc-tuned-embeddings
   python3 -m venv .venv
   .venv/bin/pip install --upgrade pip
   # Try ROCm wheel first (verify the index URL against current pytorch.org docs)
   .venv/bin/pip install --index-url https://download.pytorch.org/whl/rocm6.2 torch
   # Fall back to CPU wheel if ROCm install or device test fails:
   #   .venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch
   .venv/bin/pip install sentence-transformers tiktoken einops openai python-dotenv
   ```

   Sanity check device:
   ```bash
   .venv/bin/python -c "import torch; print('cuda?', torch.cuda.is_available()); print('hip?', torch.version.hip)"
   ```

3. If ROCm is live, set the device used by sentence-transformers explicitly
   inside `scripts/check2_open_baselines.py`. The script currently lets ST
   choose, which should pick CUDA/ROCm automatically when present. If not,
   pass `device="cuda"` (yes, even for ROCm — that's how PyTorch surfaces it)
   to `SentenceTransformer(...)`.

4. Run nomic + bge-large:

   ```bash
   .venv/bin/python scripts/check2_open_baselines.py \
     --models nomic-v1.5 bge-large --skip-openai
   ```

   Results land in `results/check2_<model>.json` and the comparison table
   prints at the end.

5. Optionally re-run bge-base for a clean apples-to-apples on the same
   hardware (the existing JSON was produced on laptop CPU — numbers should
   be identical to within rounding because we're using deterministic
   embedding, but the runtime stats won't match):

   ```bash
   .venv/bin/python scripts/check2_open_baselines.py --models bge-base --skip-openai
   ```

6. Write `notes/check2_open_baselines_report.md` summarizing all four
   models' overall metrics, per-topic breakdowns, and a final recommendation
   on the fine-tune target. Update SPEC.md's "Target model" section to
   reflect the choice with concrete numbers.

7. Mark Task #3 ("Sprint 1: Finalize Check 2 on Strix Halo") complete and
   move to Sprint 2 planning.

## Open questions for the Strix Halo session

- Does PyTorch's ROCm support work cleanly on this hardware for sentence-
  transformers inference? If yes, training in Sprint 3 is also viable on the
  same stack. If no, plan: do training data prep + small-model sanity runs
  on Strix Halo CPU, rent an hour of A100 on Modal/RunPod for the actual
  fine-tune sweeps.
- Long-tail token lengths: the corpus has a few opinions >8K tokens. We've
  capped nomic at 1024 for the baseline pass since p99 is 2296. If we want
  to test nomic's long-context advantage more honestly, re-run with
  max_seq_length=4096 on a 1–2% subset and see if it changes the topic
  breakdown.
- Whether to add a fourth baseline (e.g., `Snowflake/snowflake-arctic-embed-l-v2.0`
  or `mxbai-embed-large-v1`) before Sprint 1 closes.

## Files to reference quickly

- `scripts/check1_leakage.py` — leakage analyzer (re-runs in ~30s).
- `scripts/check2_open_baselines.py` — embedding + eval pipeline.
- `results/check2_comparison.json` — current short-form comparison table.
- `notes/check1_leakage_report.md` — full leakage analysis writeup.
- `CLAUDE.md`, `SPEC.md`, `IMPLEMENTATION_PLAN.md` — planning docs.
