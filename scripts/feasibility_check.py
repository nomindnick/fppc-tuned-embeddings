"""
Probe max batch sizes for Sprint 3 training on the Strix Halo iGPU.

For each (loss, n_hard_negatives) combination we want to use in Stage A,
run a single training step with progressively larger batch sizes until we
OOM. Reports the largest batch_size that fit, so Sprint 3 configs use
measured (not guessed) numbers.

Tests at max_seq_length=1024 with the actual Snowflake-arctic-l-v2 model.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback

import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.losses import (
    CachedMultipleNegativesRankingLoss,
    MultipleNegativesRankingLoss,
)

MODEL_ID = "Snowflake/snowflake-arctic-embed-l-v2.0"
DEFAULT_MAX_SEQ = 1024
N_ROWS_PER_PROBE = 64  # enough to fill several batches at any tested batch size

# A text chunk that tokenizes to comfortably more than MAX_SEQ tokens with
# the XLM-RoBERTa BPE tokenizer. Real corpus content is preferable to
# repeated tokens (real text exercises attention patterns properly).
LONG_TEXT_TEMPLATE = (
    "The California Fair Political Practices Commission has issued an "
    "advisory opinion concerning the application of Government Code "
    "section 87100 to the question of whether a public official may "
    "participate in a governmental decision in which she has a financial "
    "interest. Under the Political Reform Act, an official is disqualified "
    "from making, participating in making, or in any way attempting to use "
    "her official position to influence a governmental decision in which "
    "she knows or has reason to know that she has a financial interest. "
    "The analysis proceeds in eight steps. First, identify the public "
    "official. Second, identify the governmental decision in which the "
    "official is participating. Third, identify the official's economic "
    "interests. Fourth, determine whether the official's economic interests "
    "are directly or indirectly involved in the governmental decision. "
    "Fifth, determine whether the financial effect is material under the "
    "applicable materiality standard. Sixth, consider whether the public "
    "generally exception applies. Seventh, consider the legally required "
    "participation exception. Eighth, document the conclusion and any "
    "recusal taken. The conclusion in this opinion is that the official is "
    "disqualified because the decision will have a reasonably foreseeable "
    "material financial effect on her economic interests. "
) * 4  # ~4x repetition to guarantee >1024 tokens after BPE


def fresh_model(max_seq: int) -> SentenceTransformer:
    """Load a clean model, freshly placed on GPU, with seq_len fixed."""
    model = SentenceTransformer(MODEL_ID)
    model.max_seq_length = max_seq
    return model


def make_dataset(n_rows: int, n_hard_negatives: int) -> Dataset:
    rows = []
    for _ in range(n_rows):
        row = {"anchor": LONG_TEXT_TEMPLATE, "positive": LONG_TEXT_TEMPLATE}
        for i in range(n_hard_negatives):
            row[f"negative_{i + 1}"] = LONG_TEXT_TEMPLATE
        rows.append(row)
    return Dataset.from_list(rows)


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def probe(
    loss_kind: str,
    batch_size: int,
    n_hard_negatives: int,
    max_seq: int,
    bf16: bool,
    mini_batch_size: int | None = None,
) -> tuple[bool, str]:
    """Try one training step at (batch_size, n_hard_negatives, mini_batch).

    Returns (ok, reason). Always reloads the model so a prior OOM doesn't
    leak corrupted state into the next probe.
    """
    cleanup()
    try:
        model = fresh_model(max_seq)
        if loss_kind == "mnrl":
            loss = MultipleNegativesRankingLoss(model)
        elif loss_kind == "cached_mnrl":
            assert mini_batch_size is not None
            loss = CachedMultipleNegativesRankingLoss(
                model, mini_batch_size=mini_batch_size
            )
        else:
            raise ValueError(loss_kind)

        ds = make_dataset(n_rows=max(N_ROWS_PER_PROBE, batch_size * 2),
                          n_hard_negatives=n_hard_negatives)
        args = SentenceTransformerTrainingArguments(
            output_dir="/tmp/sprint3_probe",
            per_device_train_batch_size=batch_size,
            num_train_epochs=1,
            max_steps=1,
            logging_steps=1,
            save_strategy="no",
            eval_strategy="no",
            report_to=[],
            dataloader_num_workers=0,
            seed=20260521,
            bf16=bf16,
        )
        trainer = SentenceTransformerTrainer(
            model=model,
            args=args,
            train_dataset=ds,
            loss=loss,
        )
        trainer.train()
        # Touch peak memory before returning
        peak_gb = 0.0
        if torch.cuda.is_available():
            peak_gb = torch.cuda.max_memory_allocated() / 1024**3
            torch.cuda.reset_peak_memory_stats()
        del trainer, loss, model, ds
        cleanup()
        return True, f"ok (peak={peak_gb:.2f} GB)"
    except torch.cuda.OutOfMemoryError as e:
        cleanup()
        return False, f"OOM: {str(e)[:120]}"
    except RuntimeError as e:
        cleanup()
        msg = str(e)
        if "out of memory" in msg.lower() or "hipErrorOutOfMemory" in msg:
            return False, f"OOM (RuntimeError): {msg[:120]}"
        traceback.print_exc()
        return False, f"RuntimeError: {msg[:200]}"


def probe_plain_mnrl(max_seq: int, bf16: bool) -> dict:
    results: dict[str, str] = {}
    last_ok = None
    for bs in [4, 8, 16, 32, 48, 64, 96]:
        print(f"\n[MNRL]    batch_size={bs} …", flush=True)
        ok, reason = probe("mnrl", batch_size=bs, n_hard_negatives=0,
                           max_seq=max_seq, bf16=bf16)
        print(f"  → {reason}")
        results[f"bs={bs}"] = reason
        if ok:
            last_ok = bs
        else:
            break
    return {"loss": "MultipleNegativesRankingLoss", "max_batch_size": last_ok, "trace": results}


def probe_mnrl_with_hard_negs(max_seq: int, bf16: bool, n_hard: int = 1) -> dict:
    results: dict[str, str] = {}
    last_ok = None
    for bs in [4, 8, 16, 24, 32, 48]:
        print(f"\n[MNRL+{n_hard}hn] batch_size={bs} …", flush=True)
        ok, reason = probe("mnrl", batch_size=bs, n_hard_negatives=n_hard,
                           max_seq=max_seq, bf16=bf16)
        print(f"  → {reason}")
        results[f"bs={bs}"] = reason
        if ok:
            last_ok = bs
        else:
            break
    return {
        "loss": f"MultipleNegativesRankingLoss+{n_hard}hn",
        "max_batch_size": last_ok,
        "trace": results,
    }


def probe_cached_mnrl(max_seq: int, bf16: bool) -> dict:
    """Probe largest mini_batch_size for CachedMNRL.

    Mini-batch determines memory; global batch can be anything in multiples.
    We test with global batch = 128 (a typical Sprint 3 target).
    """
    results: dict[str, str] = {}
    last_ok = None
    for mbs in [4, 8, 16, 24, 32]:
        global_bs = max(128, mbs * 2)
        print(f"\n[CachedMNRL] global={global_bs} mini={mbs} …", flush=True)
        ok, reason = probe(
            "cached_mnrl",
            batch_size=global_bs,
            n_hard_negatives=0,
            max_seq=max_seq,
            bf16=bf16,
            mini_batch_size=mbs,
        )
        print(f"  → {reason}")
        results[f"mini_bs={mbs}/global={global_bs}"] = reason
        if ok:
            last_ok = mbs
        else:
            break
    return {
        "loss": "CachedMultipleNegativesRankingLoss",
        "max_mini_batch_size": last_ok,
        "trace": results,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="/home/nick/Projects/fppc-tuned-embeddings/results/sprint3_feasibility.json",
    )
    parser.add_argument("--max-seq", type=int, default=DEFAULT_MAX_SEQ)
    parser.add_argument("--bf16", action="store_true",
                        help="Train in bfloat16 (halves activation memory)")
    parser.add_argument("--skip-cached", action="store_true",
                        help="Skip CachedMNRL probe (slowest)")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — running on CPU, OOM probes will be misleading.")

    t0 = time.time()
    summary = {
        "model": MODEL_ID,
        "max_seq_length": args.max_seq,
        "bf16": args.bf16,
        "torch_version": torch.__version__,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "mnrl": probe_plain_mnrl(args.max_seq, args.bf16),
        "mnrl_with_1hn": probe_mnrl_with_hard_negs(args.max_seq, args.bf16, n_hard=1),
        "cached_mnrl": (
            None if args.skip_cached
            else probe_cached_mnrl(args.max_seq, args.bf16)
        ),
        "wall_time_s": None,
    }
    summary["wall_time_s"] = round(time.time() - t0, 1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + "=" * 60)
    print(f"Wrote {args.out}")
    print(f"Wall time: {summary['wall_time_s']}s")
    print("\nHEADLINE:")
    print(f"  max_seq={args.max_seq}  bf16={args.bf16}")
    print(f"  MNRL max batch_size           = {summary['mnrl']['max_batch_size']}")
    print(f"  MNRL+1hn max batch_size       = {summary['mnrl_with_1hn']['max_batch_size']}")
    if summary['cached_mnrl'] is not None:
        print(f"  CachedMNRL max mini_batch_size = {summary['cached_mnrl']['max_mini_batch_size']}")


if __name__ == "__main__":
    main()
