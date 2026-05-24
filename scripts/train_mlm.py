"""
Sprint 4 Lever 1: continued masked-LM pretraining of
Snowflake-arctic-embed-l-v2.0 on the FPPC corpus.

Goal: adapt the encoder to FPPC-specific token semantics (statute
numbers, regulatory citations, legal phrasings). The Sprint 5
per-query analysis showed that for q001-class queries
("Section 87103(a) disqualification …"), Snowflake's bi-encoder does
not surface any gold-relevant opinions in its top-100 — the relevant
opinions don't cluster near these queries in Snowflake's latent space.
Domain-adaptive MLM is the cheapest way to test whether more FPPC
text exposure reshapes that geometry.

The output is a SentenceTransformer-compatible directory that
`train_sprint3.py` can load via `base_model` to apply the d3 recipe
on top.

Usage:
  python scripts/train_mlm.py experiments/<run_id>/mlm_config.json
  python scripts/train_mlm.py experiments/<run_id>/mlm_config.json --smoke

  --smoke runs on 200 chunks / 20 steps to validate the pipeline.

Config schema:
  run_id, base_model, max_seq_length, batch_size, learning_rate,
  warmup_ratio, weight_decay, mlm_probability, epochs, max_chunks_per_opinion,
  bf16, seed, model_save_dir, experiment_dir.
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
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

class EncoderFreezeCallback(TrainerCallback):
    """Freeze encoder params for the first `freeze_steps`, then unfreeze.

    During the freeze phase only `lm_head.dense`, `lm_head.layer_norm`,
    and `lm_head.bias` train (the decoder weight is tied to the frozen
    embeddings). This lets the freshly-initialized head converge against
    the original encoder geometry before any gradient flows back into
    the encoder, limiting the disruption Snowflake's first MLM run saw.
    """

    def __init__(self, freeze_steps: int, encoder_params: list):
        self._freeze_steps = freeze_steps
        self._encoder_params = encoder_params
        self._frozen = False

    def on_train_begin(self, args, state, control, **kwargs):
        if self._freeze_steps <= 0:
            return
        for p in self._encoder_params:
            p.requires_grad = False
        self._frozen = True
        n_frozen = sum(p.numel() for p in self._encoder_params)
        print(
            f"[freeze] Encoder frozen for the first {self._freeze_steps} steps "
            f"({n_frozen:,} params)"
        )

    def on_step_end(self, args, state, control, **kwargs):
        if self._frozen and state.global_step >= self._freeze_steps:
            for p in self._encoder_params:
                p.requires_grad = True
            self._frozen = False
            print(f"[freeze] Encoder unfrozen at step {state.global_step}")

CORPUS_DIR = "/home/nick/Projects/fppc-opinions-corpus/data/extracted"
EVAL_DATASET = "/home/nick/Projects/fppc-opinions-eval/eval/dataset.json"


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_held_out_ids() -> set[str]:
    """All opinion_ids referenced by any eval judgment (any score)."""
    with open(EVAL_DATASET) as f:
        ds = json.load(f)["queries"]
    ids: set[str] = set()
    for q in ds:
        for j in q.get("relevance_judgments", []):
            ids.add(j["opinion_id"])
    return ids


def load_training_texts() -> tuple[list[str], int]:
    """Walk corpus, return (full_text strings, count_excluded)."""
    held = load_held_out_ids()
    texts: list[str] = []
    n_held = 0
    n_short = 0
    seen_ids: set[str] = set()
    for year_dir in sorted(os.listdir(CORPUS_DIR)):
        ypath = os.path.join(CORPUS_DIR, year_dir)
        if not os.path.isdir(ypath):
            continue
        for filename in sorted(os.listdir(ypath)):
            if not filename.endswith(".json"):
                continue
            with open(os.path.join(ypath, filename)) as f:
                op = json.load(f)
            oid = op.get("id", filename.replace(".json", ""))
            if oid in seen_ids:
                continue  # corpus has one duplicate ID across years
            seen_ids.add(oid)
            if oid in held:
                n_held += 1
                continue
            txt = (op.get("content") or {}).get("full_text") or ""
            if len(txt.strip()) < 100:
                n_short += 1
                continue
            texts.append(txt.strip())
    print(
        f"Corpus walk: kept {len(texts)} for MLM "
        f"(held-out excluded: {n_held}; too-short skipped: {n_short})"
    )
    return texts, len(held)


def build_chunked_dataset(
    texts: list[str],
    tokenizer,
    max_seq_length: int,
    max_chunks_per_opinion: int,
    seed: int,
) -> Dataset:
    """Tokenize each text, slice into non-overlapping 512-token windows.

    Caps at `max_chunks_per_opinion` to prevent long opinions from
    dominating the training distribution. Each chunk wraps with
    [CLS] and [SEP].
    """
    rng = random.Random(seed)
    print(f"Tokenizing {len(texts)} opinions…")
    t0 = time.time()
    all_chunks: list[list[int]] = []
    content_len = max_seq_length - 2  # reserve [CLS] + [SEP]
    per_opinion_chunk_counts: list[int] = []
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        n_chunks = max(1, (len(ids) + content_len - 1) // content_len)
        n_chunks = min(n_chunks, max_chunks_per_opinion)
        # Take the first n_chunks non-overlapping windows starting at random offset
        max_start = max(0, len(ids) - n_chunks * content_len)
        start = rng.randint(0, max_start)
        for i in range(n_chunks):
            window = ids[start + i * content_len : start + (i + 1) * content_len]
            if len(window) < content_len // 4:
                break  # skip near-empty tail
            chunk = [tokenizer.cls_token_id] + window + [tokenizer.sep_token_id]
            all_chunks.append(chunk)
        per_opinion_chunk_counts.append(n_chunks)
    rng.shuffle(all_chunks)
    print(
        f"  -> {len(all_chunks)} chunks "
        f"(mean chunks/opinion: {sum(per_opinion_chunk_counts)/max(len(per_opinion_chunk_counts),1):.2f}) "
        f"in {time.time() - t0:.1f}s"
    )
    return Dataset.from_dict({"input_ids": all_chunks})


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
    parser.add_argument("config_path")
    parser.add_argument("--smoke", action="store_true",
                        help="200 chunks / 20 steps; pipeline validation only.")
    args = parser.parse_args()

    with open(args.config_path) as f:
        cfg = json.load(f)

    out_dir = cfg["model_save_dir"]
    exp_dir = cfg["experiment_dir"]
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(exp_dir).mkdir(parents=True, exist_ok=True)

    set_seeds(cfg["seed"])

    print(f"=== MLM run: {cfg['run_id']} ===")
    print(f"  base_model              = {cfg['base_model']}")
    print(f"  max_seq_length          = {cfg['max_seq_length']}")
    print(f"  batch_size              = {cfg['batch_size']}")
    print(f"  learning_rate           = {cfg['learning_rate']}")
    print(f"  warmup_ratio            = {cfg['warmup_ratio']}")
    print(f"  weight_decay            = {cfg.get('weight_decay', 0.01)}")
    print(f"  mlm_probability         = {cfg['mlm_probability']}")
    print(f"  epochs                  = {cfg['epochs']}")
    print(f"  max_chunks_per_opinion  = {cfg.get('max_chunks_per_opinion', 10)}")
    print(f"  bf16                    = {cfg.get('bf16', True)}")
    print(f"  freeze_encoder_steps    = {cfg.get('freeze_encoder_steps', 0)}")
    print(f"  smoke                   = {args.smoke}")

    print("\nLoading SentenceTransformer wrapper (for tokenizer + later transplant)…")
    st_model = SentenceTransformer(cfg["base_model"])
    st_model.max_seq_length = cfg["max_seq_length"]
    tokenizer = st_model.tokenizer
    print(f"  tokenizer: {type(tokenizer).__name__}  vocab_size={tokenizer.vocab_size}")

    print("\nLoading AutoModelForMaskedLM…")
    mlm_model = AutoModelForMaskedLM.from_pretrained(cfg["base_model"])
    print(f"  type: {type(mlm_model).__name__}  encoder: {type(mlm_model.roberta).__name__}")

    print("\nLoading training texts…")
    texts, _ = load_training_texts()
    if args.smoke:
        texts = texts[:50]

    train_ds = build_chunked_dataset(
        texts=texts,
        tokenizer=tokenizer,
        max_seq_length=cfg["max_seq_length"],
        max_chunks_per_opinion=cfg.get("max_chunks_per_opinion", 10),
        seed=cfg["seed"],
    )
    if args.smoke:
        train_ds = train_ds.select(range(min(200, len(train_ds))))

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=cfg["mlm_probability"],
    )

    train_args = TrainingArguments(
        output_dir=os.path.join(exp_dir, "trainer_out"),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        warmup_ratio=cfg["warmup_ratio"],
        weight_decay=cfg.get("weight_decay", 0.01),
        bf16=bool(cfg.get("bf16", True)),
        logging_steps=50,
        save_strategy="no",
        report_to=[],
        seed=cfg["seed"],
        dataloader_num_workers=0,
        max_steps=20 if args.smoke else -1,
    )

    callbacks = []
    freeze_steps = int(cfg.get("freeze_encoder_steps", 0))
    if freeze_steps > 0:
        # Encoder = everything under mlm_model.roberta (embeddings +
        # transformer layers). Excludes the lm_head, which keeps training.
        encoder_params = list(mlm_model.roberta.parameters())
        callbacks.append(EncoderFreezeCallback(freeze_steps, encoder_params))

    trainer = Trainer(
        model=mlm_model,
        args=train_args,
        data_collator=collator,
        train_dataset=train_ds,
        callbacks=callbacks,
    )

    print(f"\nStarting MLM training on {len(train_ds)} chunks…")
    t0 = time.time()
    train_result = trainer.train()
    wall_s = time.time() - t0
    print(f"\nTraining done in {wall_s:.1f}s")
    print(f"  final train loss: {train_result.training_loss:.4f}")

    print("\nTransplanting MLM-trained encoder into the SentenceTransformer wrapper…")
    encoder_state = mlm_model.roberta.state_dict()
    missing, unexpected = st_model[0].model.load_state_dict(encoder_state, strict=False)
    if missing:
        print(f"  load_state_dict MISSING (left untrained in ST wrapper): {len(missing)}")
        for k in missing[:5]:
            print(f"    {k}")
    if unexpected:
        print(f"  load_state_dict UNEXPECTED (ignored): {len(unexpected)}")
        for k in unexpected[:5]:
            print(f"    {k}")

    print(f"\nSaving SentenceTransformer to {out_dir}…")
    st_model.save(out_dir)
    with open(os.path.join(exp_dir, "model_pointer.txt"), "w") as f:
        f.write(os.path.abspath(out_dir) + "\n")
    with open(os.path.join(exp_dir, "build_report.json"), "w") as f:
        json.dump({
            "wall_time_s": round(wall_s, 1),
            "n_texts": len(texts),
            "n_chunks": len(train_ds),
            "final_train_loss": round(float(train_result.training_loss), 4),
            "smoke": args.smoke,
        }, f, indent=2)

    print(f"\nDone. Artifacts in {exp_dir}/")


if __name__ == "__main__":
    main()
