"""
Build query paraphrases for Sprint 3 lever 3 (H4 — distribution mismatch).

Reads pairs.jsonl, filters out val_slice opinion_ids, samples N rows stratified
by topic_primary, then asks a local Ollama LLM to produce two paraphrases of
each question:
  - "keyword" — a 4-10 word keyword bag style (statute numbers + topical terms)
  - "fact_pattern" — a 2-4 sentence hypothetical scenario

Writes paraphrased_pairs.jsonl with two new rows per source row (each carrying
the source row's positive doc(s) but with `question` replaced by the paraphrase
and `question_source` set to "synthetic_keyword" or "synthetic_fact_pattern").

Concurrent Ollama calls via thread pool; idempotent (skips rows already in
the output if --resume is passed).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "gemma4:e4b"

PROMPT_TEMPLATE = """You paraphrase legal questions for a search index. Output ONLY valid JSON, nothing else.

Original question: {question}

Answer the opinion gives (CONCLUSION): {conclusion}

Generate two distinct search-query paraphrases that should retrieve the same opinion:

1. A "keyword" paraphrase: a 4-10 word keyword bag style, statute numbers and topical terms only (no sentence structure, no verbs like "may" or "can"). Example style: "Section 87103(a) disqualification business entity investment financial interest".

2. A "fact_pattern" paraphrase: a 2-4 sentence hypothetical scenario describing a concrete situation, possibly ending with an implicit question. Example style: "A city council member owns a home in an assessment district. The council must vote on the annual levy for the district which funds maintenance. The assessment applies to all properties in the district including the council member's home."

Output EXACTLY this JSON structure, no other text, no markdown fences:
{{"keyword": "<keyword bag>", "fact_pattern": "<fact pattern>"}}"""


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def append_jsonl(rows: list[dict], path: Path) -> None:
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def stratified_sample(
    pairs: list[dict], val_ids: set[str], n: int, seed: int
) -> list[dict]:
    """Sample n rows stratified by topic_primary, excluding val_ids."""
    eligible = [p for p in pairs if p["opinion_id"] not in val_ids]
    by_topic: dict[str, list[dict]] = {}
    for p in eligible:
        by_topic.setdefault(p.get("topic_primary") or "unclassified", []).append(p)

    rng = random.Random(seed)
    # Allocate proportionally
    total_pool = sum(len(v) for v in by_topic.values())
    out: list[dict] = []
    for topic, rows in sorted(by_topic.items()):
        k = max(1, round(n * len(rows) / total_pool))
        rng.shuffle(rows)
        out.extend(rows[:k])
    # If we overshot, trim; if undershot, fill from leftovers
    rng.shuffle(out)
    if len(out) > n:
        out = out[:n]
    return out


def extract_conclusion(qa_text: str | None) -> str:
    """Pull just the CONCLUSION text from pos_qa_text for prompting context."""
    if not qa_text:
        return ""
    m = re.search(r"\n\nCONCLUSION:(.*)", qa_text, flags=re.DOTALL)
    if not m:
        return ""
    return m.group(1).strip()[:1500]


def call_ollama(
    url: str,
    model: str,
    prompt: str,
    timeout_s: float,
) -> str | None:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 800, "num_ctx": 4096},
    }
    try:
        r = requests.post(url, json=payload, timeout=timeout_s)
        r.raise_for_status()
        return r.json().get("response", "")
    except Exception as e:
        return f"__ERROR__:{type(e).__name__}:{e}"


def parse_response(text: str) -> dict | None:
    """Extract {keyword, fact_pattern} from model output. Tolerates whitespace
    and markdown fences. Returns None on parse failure."""
    if not text:
        return None
    # Strip markdown fences if present
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE | re.MULTILINE)
    # Find first {...} block
    m = re.search(r"\{[\s\S]*\}", cleaned)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    kw = obj.get("keyword")
    fp = obj.get("fact_pattern")
    if not (isinstance(kw, str) and isinstance(fp, str) and kw.strip() and fp.strip()):
        return None
    return {"keyword": kw.strip(), "fact_pattern": fp.strip()}


def process_one(args_tuple):
    url, model, row = args_tuple
    question = row.get("question") or ""
    conclusion = extract_conclusion(row.get("pos_qa_text"))
    prompt = PROMPT_TEMPLATE.format(question=question, conclusion=conclusion or "(not available)")
    text = call_ollama(url, model, prompt, timeout_s=60.0)
    if text and text.startswith("__ERROR__"):
        return row["opinion_id"], None, text
    parsed = parse_response(text or "")
    return row["opinion_id"], parsed, text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", default="data/training/pairs.jsonl")
    parser.add_argument("--val", default="data/training/val_slice.jsonl")
    parser.add_argument("--out", default="data/training/paraphrased_pairs.jsonl")
    parser.add_argument("--raw", default="data/training/paraphrases_raw.jsonl",
                        help="Sidecar with raw model responses (for debug)")
    parser.add_argument("--n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260521)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--url", default=DEFAULT_OLLAMA_URL)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true",
                        help="Skip opinion_ids already in --raw file")
    args = parser.parse_args()

    out_path = Path(args.out)
    raw_path = Path(args.raw)

    print(f"Loading pairs from {args.pairs}…", flush=True)
    pairs = load_jsonl(Path(args.pairs))
    val_ids = {r["opinion_id"] for r in load_jsonl(Path(args.val))}
    print(f"  pairs={len(pairs)}  val={len(val_ids)}", flush=True)

    sample = stratified_sample(pairs, val_ids, args.n, args.seed)
    print(f"Stratified sample: {len(sample)} rows", flush=True)

    already_done: set[str] = set()
    if args.resume and raw_path.exists():
        for r in load_jsonl(raw_path):
            already_done.add(r["opinion_id"])
        print(f"  resuming: {len(already_done)} already complete", flush=True)
        sample = [s for s in sample if s["opinion_id"] not in already_done]
        print(f"  remaining: {len(sample)}", flush=True)

    if not sample:
        print("Nothing to do.")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_parse_fail = 0
    n_api_err = 0
    new_pairs: list[dict] = []
    t0 = time.time()

    # Build by_id lookup so we can quickly attach positives by opinion_id
    by_id = {p["opinion_id"]: p for p in sample}

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_one, (args.url, args.model, row)): row["opinion_id"]
            for row in sample
        }
        for i, fut in enumerate(as_completed(futures)):
            oid, parsed, raw_text = fut.result()
            with open(raw_path, "a") as f:
                f.write(json.dumps({"opinion_id": oid, "raw": raw_text, "parsed_ok": parsed is not None}) + "\n")

            if raw_text and raw_text.startswith("__ERROR__"):
                n_api_err += 1
            elif parsed is None:
                n_parse_fail += 1
            else:
                n_ok += 1
                src = by_id[oid]
                base = {k: v for k, v in src.items() if k != "question" and k != "question_source"}
                # Emit two new rows: one keyword, one fact_pattern
                kw_row = {**base, "question": parsed["keyword"], "question_source": "synthetic_keyword"}
                fp_row = {**base, "question": parsed["fact_pattern"], "question_source": "synthetic_fact_pattern"}
                new_pairs.append(kw_row)
                new_pairs.append(fp_row)

            if (i + 1) % 25 == 0 or (i + 1) == len(futures):
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(futures) - (i + 1)) / rate if rate > 0 else 0
                print(f"  [{i+1:4d}/{len(futures)}] ok={n_ok} parse_fail={n_parse_fail} "
                      f"api_err={n_api_err}  rate={rate:.2f}/s  eta={int(eta)}s",
                      flush=True)

    append_jsonl(new_pairs, out_path)
    print(f"\nDone. Wrote {len(new_pairs)} paraphrased rows to {out_path}", flush=True)
    print(f"  parse_ok={n_ok}  parse_fail={n_parse_fail}  api_err={n_api_err}", flush=True)


if __name__ == "__main__":
    main()
