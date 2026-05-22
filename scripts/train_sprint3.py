"""
Sprint 3 trainer: runs one fine-tuning job from one JSON config.

Usage:
  python scripts/train_sprint3.py experiments/<run_id>/config.json
  python scripts/train_sprint3.py experiments/<run_id>/config.json --smoke

  --smoke runs on only 100 training rows / 10 max steps to validate the pipeline.

Config fields (see notes/training_data_design.md and SPEC.md for context):
  run_id, base_model, loss, use_hard_negatives, n_hard_negatives_per_row,
  positive_column, epochs, batch_size, mini_batch_size, learning_rate,
  warmup_ratio, max_seq_length, query_prefix, doc_prefix, seed,
  val_every_n_steps, val_pool, pairs_path, val_path, hard_negatives_path,
  model_save_dir, experiment_dir, bf16.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.losses import (
    CachedMultipleNegativesRankingLoss,
    MultipleNegativesRankingLoss,
)
from transformers import TrainerCallback

CORPUS_DIR = "/home/nick/Projects/fppc-opinions-corpus/data/extracted"


# ---------------------------------------------------------------------------
# IO + filtering
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_hard_negatives(path: str) -> dict[str, list[dict]]:
    """Returns {opinion_id: [negative_record, ...]} preserving file order."""
    by_id: dict[str, list[dict]] = {}
    for row in load_jsonl(path):
        by_id[row["opinion_id"]] = row["negatives"]
    return by_id


def derive_pos_conclusion_only(qa_text: str | None) -> str | None:
    """Return everything from the CONCLUSION: marker onward; None if not found.

    Used by the pos_conclusion_only column, which tests whether the lexical
    leakage of the verbatim question into pos_qa_text is the dominant cause of
    Stage A's collapse. Stripping the QUESTION section leaves only the
    conclusion (and label) so the model cannot identity-match the query.
    """
    if not qa_text:
        return None
    delim = "\n\nCONCLUSION:"
    idx = qa_text.find(delim)
    if idx < 0:
        return None
    return qa_text[idx + 2:]


def add_derived_pos_columns(rows: list[dict], positive_column: str) -> dict:
    """Mutate rows to add the derived positive column if requested.

    Returns a small report with derivation stats.
    """
    if positive_column != "pos_conclusion_only":
        return {"derived": False}
    n_total = 0
    n_filled = 0
    n_missing = 0
    for r in rows:
        n_total += 1
        derived = derive_pos_conclusion_only(r.get("pos_qa_text"))
        r["pos_conclusion_only"] = derived
        if derived is None:
            n_missing += 1
        else:
            n_filled += 1
    return {
        "derived": True,
        "derived_column": positive_column,
        "rows_total": n_total,
        "rows_with_value": n_filled,
        "rows_missing_delim": n_missing,
    }


def load_corpus_qa_text() -> dict[str, str]:
    """Build {opinion_id: qa_text} for the whole corpus.

    Used to look up negative-document text since some negatives are corpus
    opinions that were filtered out of pairs.jsonl (e.g. no usable question).
    qa_text is always present per the corpus schema (with full_text fallback
    matching _eval_lib.py).
    """
    out: dict[str, str] = {}
    for year_dir in sorted(os.listdir(CORPUS_DIR)):
        year_path = os.path.join(CORPUS_DIR, year_dir)
        if not os.path.isdir(year_path):
            continue
        for filename in sorted(os.listdir(year_path)):
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(year_path, filename)) as f:
                op = json.load(f)
            oid = op.get("id", filename.replace(".json", ""))
            if oid in out:
                continue  # one duplicate ID across years; keep first
            qa = (op.get("embedding") or {}).get("qa_text") or ""
            if len(qa.strip()) < 20:
                qa = (op.get("content") or {}).get("full_text") or ""
            out[oid] = qa.strip() or " "
    return out


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def build_train_dataset(
    pairs: list[dict],
    val_ids: set[str],
    positive_column: str,
    query_prefix: str,
    doc_prefix: str,
    use_hard_negatives: bool,
    n_hard_negatives_per_row: int,
    hard_negatives: dict[str, list[dict]] | None,
    corpus_qa: dict[str, str] | None,
    rng: random.Random,
) -> tuple[Dataset, dict]:
    """Returns (HF Dataset, build_report)."""

    # Filter: drop val opinions; drop rows missing the positive column.
    rows: list[dict] = []
    n_val_dropped = 0
    n_poscol_null = 0
    for r in pairs:
        if r["opinion_id"] in val_ids:
            n_val_dropped += 1
            continue
        if not r.get(positive_column):
            n_poscol_null += 1
            continue
        rows.append(r)

    # Attach hard negatives if requested.
    n_neg_dropped = 0
    if use_hard_negatives:
        assert hard_negatives is not None and corpus_qa is not None
        kept_rows: list[dict] = []
        for r in rows:
            negs = hard_negatives.get(r["opinion_id"], [])
            # Take the first n; each negative's doc text is its qa_text from corpus.
            picked: list[str] = []
            for n in negs:
                if len(picked) >= n_hard_negatives_per_row:
                    break
                txt = corpus_qa.get(n["opinion_id"])
                if not txt:
                    continue
                picked.append(txt)
            if len(picked) < n_hard_negatives_per_row:
                n_neg_dropped += 1
                continue  # not enough hard negatives → drop row
            for i, txt in enumerate(picked):
                r[f"_neg_{i + 1}"] = txt
            kept_rows.append(r)
        rows = kept_rows

    # Build the HF Dataset.
    items: list[dict] = []
    for r in rows:
        item = {
            "anchor": query_prefix + r["question"],
            "positive": doc_prefix + r[positive_column],
        }
        for i in range(n_hard_negatives_per_row if use_hard_negatives else 0):
            item[f"negative_{i + 1}"] = doc_prefix + r[f"_neg_{i + 1}"]
        items.append(item)
    rng.shuffle(items)

    ds = Dataset.from_list(items)
    report = {
        "input_pairs": len(pairs),
        "dropped_val_overlap": n_val_dropped,
        "dropped_poscol_null": n_poscol_null,
        "dropped_insufficient_hard_negs": n_neg_dropped,
        "training_rows": len(items),
    }
    return ds, report


def build_val_evaluator(
    val_rows: list[dict],
    positive_column: str,
    query_prefix: str,
    doc_prefix: str,
    name: str,
) -> tuple[InformationRetrievalEvaluator | None, dict]:
    """Build val-slice-only IR evaluator. Drops val rows where pos col is null."""
    queries: dict[str, str] = {}
    corpus: dict[str, str] = {}
    relevant_docs: dict[str, set[str]] = {}
    n_poscol_null = 0
    for r in val_rows:
        if not r.get(positive_column):
            n_poscol_null += 1
            continue
        oid = r["opinion_id"]
        queries[oid] = query_prefix + r["question"]
        corpus[oid] = doc_prefix + r[positive_column]
        relevant_docs[oid] = {oid}

    if not queries:
        return None, {"val_rows": 0, "val_dropped_poscol_null": n_poscol_null}

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        mrr_at_k=[5, 10],
        ndcg_at_k=[5, 10],
        accuracy_at_k=[1, 3, 5],
        precision_recall_at_k=[5, 10],
        map_at_k=[100],
        show_progress_bar=False,
        batch_size=32,
        name=name,
    )
    report = {
        "val_rows": len(queries),
        "val_dropped_poscol_null": n_poscol_null,
    }
    return evaluator, report


# ---------------------------------------------------------------------------
# Per-step metric logging
# ---------------------------------------------------------------------------

class ValMetricsCallback(TrainerCallback):
    """Append IR evaluator metrics to val_metrics.jsonl after each evaluate."""

    def __init__(self, out_path: str):
        self.out_path = out_path
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        # Truncate prior contents (each run owns its own file).
        Path(out_path).write_text("")

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics:
            return
        record = {"step": state.global_step, "epoch": state.epoch, **metrics}
        with open(self.out_path, "a") as f:
            f.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Loss factory
# ---------------------------------------------------------------------------

def build_loss(model: SentenceTransformer, cfg: dict):
    name = cfg["loss"]
    if name == "MultipleNegativesRankingLoss":
        return MultipleNegativesRankingLoss(model)
    if name == "CachedMultipleNegativesRankingLoss":
        mbs = cfg.get("mini_batch_size")
        assert mbs is not None, "CachedMNRL requires mini_batch_size"
        return CachedMultipleNegativesRankingLoss(model, mini_batch_size=mbs)
    raise ValueError(f"Unknown loss: {name}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to the run config JSON")
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny run: 100 train rows / 10 max steps. Validates pipeline.")
    args = parser.parse_args()

    with open(args.config_path) as f:
        cfg = json.load(f)

    Path(cfg["experiment_dir"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["model_save_dir"]).mkdir(parents=True, exist_ok=True)

    set_seeds(cfg["seed"])

    print(f"=== Sprint 3 run: {cfg['run_id']} ===")
    print(f"  base_model         = {cfg['base_model']}")
    print(f"  loss               = {cfg['loss']}")
    print(f"  positive_column    = {cfg['positive_column']}")
    print(f"  batch_size         = {cfg['batch_size']}"
          + (f"  (mini={cfg['mini_batch_size']})" if cfg.get('mini_batch_size') else ""))
    print(f"  use_hard_negatives = {cfg['use_hard_negatives']}"
          + (f"  (n={cfg['n_hard_negatives_per_row']})" if cfg['use_hard_negatives'] else ""))
    print(f"  max_seq_length     = {cfg['max_seq_length']}")
    print(f"  bf16               = {cfg.get('bf16', False)}")
    print(f"  smoke              = {args.smoke}")

    print("\nLoading data…")
    t0 = time.time()
    pairs = load_jsonl(cfg["pairs_path"])
    n_base = len(pairs)
    n_extra_total = 0
    for extra_path in cfg.get("extra_pairs_paths", []) or []:
        extra_rows = load_jsonl(extra_path)
        pairs.extend(extra_rows)
        n_extra_total += len(extra_rows)
        print(f"  + extra pairs: {extra_path}  ({len(extra_rows)} rows)")
    val_rows = load_jsonl(cfg["val_path"])
    val_ids = {r["opinion_id"] for r in val_rows}
    print(f"  pairs: {n_base} base + {n_extra_total} extra = {len(pairs)}  "
          f"val: {len(val_rows)}  ({time.time() - t0:.1f}s)")

    deriv_pairs = add_derived_pos_columns(pairs, cfg["positive_column"])
    deriv_val = add_derived_pos_columns(val_rows, cfg["positive_column"])
    if deriv_pairs.get("derived"):
        print(f"  derived positive column: {json.dumps(deriv_pairs)}")
        print(f"  derived val column:      {json.dumps(deriv_val)}")

    hard_negatives = None
    corpus_qa = None
    if cfg["use_hard_negatives"]:
        print("  loading hard negatives + corpus qa_text…")
        hard_negatives = load_hard_negatives(cfg["hard_negatives_path"])
        corpus_qa = load_corpus_qa_text()

    print("\nBuilding training dataset…")
    rng = random.Random(cfg["seed"])
    train_ds, build_report = build_train_dataset(
        pairs=pairs,
        val_ids=val_ids,
        positive_column=cfg["positive_column"],
        query_prefix=cfg["query_prefix"],
        doc_prefix=cfg["doc_prefix"],
        use_hard_negatives=cfg["use_hard_negatives"],
        n_hard_negatives_per_row=cfg["n_hard_negatives_per_row"],
        hard_negatives=hard_negatives,
        corpus_qa=corpus_qa,
        rng=rng,
    )
    print(f"  {json.dumps(build_report)}")

    if args.smoke:
        train_ds = train_ds.select(range(min(100, len(train_ds))))
        print(f"  [smoke] truncated to {len(train_ds)} rows")

    print("\nBuilding val evaluator…")
    evaluator, val_report = build_val_evaluator(
        val_rows=val_rows,
        positive_column=cfg["positive_column"],
        query_prefix=cfg["query_prefix"],
        doc_prefix=cfg["doc_prefix"],
        name=cfg["run_id"],
    )
    print(f"  {json.dumps(val_report)}")

    print("\nLoading model…")
    model = SentenceTransformer(cfg["base_model"])
    model.max_seq_length = cfg["max_seq_length"]
    print(f"  model loaded; max_seq_length={model.max_seq_length}")

    if cfg.get("use_lora", False):
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=cfg["lora_rank"],
            lora_alpha=cfg["lora_alpha"],
            target_modules=cfg["lora_target_modules"],
            lora_dropout=cfg.get("lora_dropout", 0.05),
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        # In sentence-transformers v5, the underlying transformer is the
        # `model` submodule (auto_model is a read-only property). Wrap it
        # with peft so we get a real PeftModel (supports merge_and_unload),
        # then put it back via the _modules dict.
        peft_wrapped = get_peft_model(model[0].model, lora_config)
        model[0]._modules["model"] = peft_wrapped
        n_total = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  LoRA enabled: trainable {n_trainable:,} / {n_total:,} "
              f"({100 * n_trainable / max(n_total, 1):.3f}%)")

    loss = build_loss(model, cfg)
    print(f"  loss: {type(loss).__name__}")

    train_args = SentenceTransformerTrainingArguments(
        output_dir=cfg["model_save_dir"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        eval_strategy="steps" if not args.smoke else "no",
        eval_steps=cfg.get("val_every_n_steps", 50),
        save_strategy="no",
        logging_steps=10,
        report_to=[],
        seed=cfg["seed"],
        dataloader_num_workers=0,
        bf16=bool(cfg.get("bf16", False)),
        max_steps=10 if args.smoke else -1,
    )

    callbacks = [ValMetricsCallback(
        os.path.join(cfg["experiment_dir"], "val_metrics.jsonl")
    )]

    trainer = SentenceTransformerTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        loss=loss,
        evaluator=evaluator if not args.smoke else None,
        callbacks=callbacks,
    )

    print("\nStarting training…")
    t0 = time.time()
    trainer.train()
    wall_s = time.time() - t0
    print(f"\nTraining done in {wall_s:.1f}s")

    # Final eval pass before saving.
    if evaluator is not None and not args.smoke:
        print("Running final val evaluator…")
        final_metrics = evaluator(model)
        with open(os.path.join(cfg["experiment_dir"], "val_metrics_final.json"), "w") as f:
            json.dump(final_metrics, f, indent=2)
        print(f"  final val: {json.dumps({k: v for k, v in final_metrics.items() if 'ndcg' in k.lower() or 'mrr' in k.lower()})}")

    print(f"\nSaving model to {cfg['model_save_dir']}…")
    if cfg.get("use_lora", False):
        # Merge the LoRA adapter back into the base weights so the saved
        # checkpoint is a plain sentence-transformers model loadable by
        # score_finetuned.py without any LoRA-specific handling. PeftModel
        # exposes merge_and_unload which returns the merged base model.
        merged_base = model[0].model.merge_and_unload()
        model[0]._modules["model"] = merged_base
        print("  LoRA adapter merged into base before save")
    model.save(cfg["model_save_dir"])
    with open(os.path.join(cfg["experiment_dir"], "model_pointer.txt"), "w") as f:
        f.write(os.path.abspath(cfg["model_save_dir"]) + "\n")

    # Persist build/val reports for reproducibility.
    with open(os.path.join(cfg["experiment_dir"], "build_report.json"), "w") as f:
        json.dump({
            "build_report": build_report,
            "val_report": val_report,
            "wall_time_s": round(wall_s, 1),
            "smoke": args.smoke,
        }, f, indent=2)

    print(f"\nDone. Artifacts in {cfg['experiment_dir']}/")


if __name__ == "__main__":
    main()
