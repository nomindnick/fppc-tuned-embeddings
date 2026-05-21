"""
Build the Sprint 2 training-pair file: one row per (question, opinion) where the
opinion is not in the held-out set and has at least one usable question
(real or synthetic).

Each row carries four positive-doc representations so the trainer can pick at
run time which one to embed as the positive document:

  pos_qa_text          — embedding.qa_text (executive summary, what the
                         current production index uses)
  pos_body             — sections.facts + analysis + conclusion (the IRAC
                         reasoning, including statutory citations; excludes
                         the executive-summary question text)
  pos_qa_plus_body     — qa_text + body  (head-biased toward exec summary,
                         body provides statutory grounding)
  pos_full_text        — content.full_text (the entire opinion)

For each column we also emit a coverage flag so the trainer can skip rows
where a column would have fallen back to noise.

Output:
  data/training/pairs.jsonl
  data/training/pairs_coverage.json   (summary report)

Held-out rule: derive the held-out set directly from
../fppc-opinions-eval/eval/dataset.json (all opinions referenced by any
relevance judgment, including score=0). This is more conservative than the
596-positive-only count in check1_leakage.json by ~28 opinions; cost is
negligible.
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from dataclasses import dataclass, asdict

EVAL_PATH = "/home/nick/Projects/fppc-opinions-eval/eval/dataset.json"
CORPUS_DIR = "/home/nick/Projects/fppc-opinions-corpus/data/extracted"
OUT_DIR = "/home/nick/Projects/fppc-tuned-embeddings/data/training"
OUT_PAIRS = os.path.join(OUT_DIR, "pairs.jsonl")
OUT_COVERAGE = os.path.join(OUT_DIR, "pairs_coverage.json")


# ---------------------------------------------------------------------------
# Held-out set
# ---------------------------------------------------------------------------

def load_held_out() -> set[str]:
    with open(EVAL_PATH) as f:
        eval_data = json.load(f)
    held: set[str] = set()
    for q in eval_data["queries"]:
        for j in q.get("relevance_judgments", []):
            held.add(j["opinion_id"])
    return held


# ---------------------------------------------------------------------------
# Question normalization
# ---------------------------------------------------------------------------

_LEAD_NUM_RE = re.compile(r"^\s*(?:\d+\s*[.)]\s*)+")
_LEAD_LABEL_RE = re.compile(r"^\s*(?:QUESTION|QUESTIONS?\s*PRESENTED|ISSUE)\s*[:.\-]\s*", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")


def normalize_question(q: str) -> str:
    """Strip common OCR artifacts from extracted questions and collapse whitespace.

    Training queries should look like eval queries (clean sentences), so we
    remove leading "1." / "QUESTION:" / "Issue —" prefixes and normalize
    whitespace. Repeated application is idempotent.
    """
    if not q:
        return ""
    s = q
    # Strip up to two leading label/number runs in case the extraction picked
    # up "QUESTION: 1. ..." or "1. QUESTION: ..."
    for _ in range(2):
        new = _LEAD_LABEL_RE.sub("", s)
        new = _LEAD_NUM_RE.sub("", new)
        if new == s:
            break
        s = new
    s = _WS_RE.sub(" ", s).strip()
    return s


# ---------------------------------------------------------------------------
# Per-opinion field assembly
# ---------------------------------------------------------------------------

@dataclass
class PairRow:
    opinion_id: str
    year: int
    question: str
    question_source: str   # "real" | "synthetic"
    qa_source: str         # from embedding.qa_source (extracted | synthetic | mixed)
    topic_primary: str | None

    pos_qa_text: str
    pos_body: str | None
    pos_qa_plus_body: str | None
    pos_full_text: str

    # Coverage / provenance flags — what *real content* backs each column
    has_real_question: bool
    has_synth_question: bool
    has_facts: bool
    has_analysis: bool
    has_conclusion: bool
    body_complete: bool    # all three of facts/analysis/conclusion present


def build_row(op: dict) -> PairRow | None:
    """Return a PairRow if this opinion is usable for training; else None."""
    sections = op.get("sections") or {}
    embedding = op.get("embedding") or {}
    content = op.get("content") or {}
    classification = op.get("classification") or {}

    real_q = (sections.get("question") or "").strip()
    synth_q = (sections.get("question_synthetic") or "").strip()

    if real_q:
        question_raw, question_source = real_q, "real"
    elif synth_q:
        question_raw, question_source = synth_q, "synthetic"
    else:
        return None  # no usable question

    question = normalize_question(question_raw)
    if len(question) < 15:
        # Pathological extraction (e.g., just "?" or a single word).
        return None

    facts = (sections.get("facts") or "").strip()
    analysis = (sections.get("analysis") or "").strip()
    conclusion = (sections.get("conclusion") or "").strip()
    # Fall back to synthetic conclusion if real is missing; we don't fall
    # back for analysis/facts since the corpus doesn't synthesize those.
    if not conclusion:
        conclusion = (sections.get("conclusion_synthetic") or "").strip()

    qa_text = (embedding.get("qa_text") or "").strip()
    full_text = (content.get("full_text") or "").strip()

    if not qa_text or not full_text:
        # Every opinion should have at least one of these; if neither, skip.
        return None

    # Assemble body: facts + analysis + conclusion. If analysis is missing
    # entirely the body column degrades to noise; mark as null so the trainer
    # can skip these rows for body-trained runs.
    body_parts = [p for p in (facts, analysis, conclusion) if p]
    body = "\n\n".join(body_parts) if (analysis and (facts or conclusion)) else None
    # Above: require analysis AND at least one of facts/conclusion to call the
    # body "complete enough" to be a meaningful column value. This is stricter
    # than "any of the three exists."

    qa_plus_body = f"{qa_text}\n\n{body}" if body else None

    return PairRow(
        opinion_id=op["id"],
        year=int(op.get("year", 0)),
        question=question,
        question_source=question_source,
        qa_source=str(embedding.get("qa_source", "")),
        topic_primary=classification.get("topic_primary"),
        pos_qa_text=qa_text,
        pos_body=body,
        pos_qa_plus_body=qa_plus_body,
        pos_full_text=full_text,
        has_real_question=bool(real_q),
        has_synth_question=bool(synth_q),
        has_facts=bool(facts),
        has_analysis=bool(analysis),
        has_conclusion=bool(conclusion),
        body_complete=body is not None,
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main() -> int:
    held = load_held_out()
    print(f"[hold-out] {len(held)} eval-referenced opinions will be excluded.")

    os.makedirs(OUT_DIR, exist_ok=True)

    n_seen = 0
    n_held_out = 0
    n_no_question = 0
    n_short_question = 0
    n_emitted = 0
    seen_ids: set[str] = set()

    coverage = {
        "by_question_source": Counter(),
        "by_qa_source": Counter(),
        "by_topic": Counter(),
        "body_complete": 0,
        "body_missing": 0,
        "real_question_pairs": 0,
        "synthetic_question_pairs": 0,
        "year_distribution": Counter(),
    }
    qa_text_lens: list[int] = []
    body_lens: list[int] = []
    full_text_lens: list[int] = []

    with open(OUT_PAIRS, "w") as out:
        for year in sorted(os.listdir(CORPUS_DIR)):
            ydir = os.path.join(CORPUS_DIR, year)
            if not os.path.isdir(ydir):
                continue
            for fname in sorted(os.listdir(ydir)):
                if not fname.endswith(".json"):
                    continue
                with open(os.path.join(ydir, fname)) as f:
                    op = json.load(f)
                n_seen += 1
                oid = op.get("id")
                if oid is None or oid in seen_ids:
                    continue  # silently dedup the one known duplicate ID
                seen_ids.add(oid)

                if oid in held:
                    n_held_out += 1
                    continue

                row = build_row(op)
                if row is None:
                    if not (op.get("sections") or {}).get("question") and not (
                        op.get("sections") or {}
                    ).get("question_synthetic"):
                        n_no_question += 1
                    else:
                        n_short_question += 1
                    continue

                out.write(json.dumps(asdict(row)) + "\n")
                n_emitted += 1
                coverage["by_question_source"][row.question_source] += 1
                coverage["by_qa_source"][row.qa_source] += 1
                if row.topic_primary:
                    coverage["by_topic"][row.topic_primary] += 1
                if row.body_complete:
                    coverage["body_complete"] += 1
                else:
                    coverage["body_missing"] += 1
                if row.question_source == "real":
                    coverage["real_question_pairs"] += 1
                else:
                    coverage["synthetic_question_pairs"] += 1
                coverage["year_distribution"][row.year] += 1
                qa_text_lens.append(len(row.pos_qa_text))
                if row.pos_body:
                    body_lens.append(len(row.pos_body))
                full_text_lens.append(len(row.pos_full_text))

    def _stats(xs: list[int], name: str) -> dict:
        if not xs:
            return {"name": name, "n": 0}
        xs.sort()
        return {
            "name": name,
            "n": len(xs),
            "p10": xs[len(xs) // 10],
            "p50": xs[len(xs) // 2],
            "p90": xs[int(len(xs) * 0.9)],
            "p99": xs[int(len(xs) * 0.99)],
            "max": xs[-1],
            "mean": int(sum(xs) / len(xs)),
        }

    report = {
        "n_corpus_files_seen": n_seen,
        "n_unique_opinions_seen": len(seen_ids),
        "n_held_out_excluded": n_held_out,
        "n_no_question_excluded": n_no_question,
        "n_short_question_excluded": n_short_question,
        "n_pairs_emitted": n_emitted,
        "held_out_total": len(held),
        "by_question_source": dict(coverage["by_question_source"]),
        "by_qa_source": dict(coverage["by_qa_source"]),
        "real_question_pairs": coverage["real_question_pairs"],
        "synthetic_question_pairs": coverage["synthetic_question_pairs"],
        "body_complete": coverage["body_complete"],
        "body_missing": coverage["body_missing"],
        "by_topic_top10": dict(coverage["by_topic"].most_common(10)),
        "year_min": min(coverage["year_distribution"]) if coverage["year_distribution"] else None,
        "year_max": max(coverage["year_distribution"]) if coverage["year_distribution"] else None,
        "char_length_stats": [
            _stats(qa_text_lens, "pos_qa_text"),
            _stats(body_lens, "pos_body"),
            _stats(full_text_lens, "pos_full_text"),
        ],
    }
    with open(OUT_COVERAGE, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print(f"Pairs file written: {OUT_PAIRS}")
    print(f"Coverage report:    {OUT_COVERAGE}")
    print()
    print(f"Corpus files seen:           {n_seen}")
    print(f"Unique opinions:             {len(seen_ids)}")
    print(f"  Held out (eval-referenced):  {n_held_out}")
    print(f"  No question (real or synth): {n_no_question}")
    print(f"  Question too short:          {n_short_question}")
    print(f"  Pairs emitted:               {n_emitted}")
    print()
    print(f"By question_source: {dict(coverage['by_question_source'])}")
    print(f"  body_complete:    {coverage['body_complete']}")
    print(f"  body_missing:     {coverage['body_missing']}")
    print(f"By qa_source: {dict(coverage['by_qa_source'])}")
    print(f"Top topics: {dict(coverage['by_topic'].most_common(8))}")
    print()
    print("Length stats (chars):")
    for s in report["char_length_stats"]:
        print(f"  {s['name']:<18} n={s['n']:>5}  p10={s.get('p10','-'):>5}  "
              f"p50={s.get('p50','-'):>6}  p90={s.get('p90','-'):>6}  "
              f"p99={s.get('p99','-'):>6}  max={s.get('max','-'):>6}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
