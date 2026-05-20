"""
Check 1: Measure overlap between eval queries and the sections.question
text of opinions they are judged relevant for.

Question this answers:
  If we use (opinion_question -> opinion_text) as training pairs for an
  embedding fine-tune, would the model effectively memorize the eval set?

Output: results/check1_leakage.json + a human-readable summary printed to stdout.
"""

from __future__ import annotations

import json
import os
import re
import statistics
from collections import Counter, defaultdict

EVAL_PATH = "/home/nick/Projects/fppc-opinions-eval/eval/dataset.json"
CORPUS_DIR = "/home/nick/Projects/fppc-opinions-corpus/data/extracted"
OUT_PATH = "/home/nick/Projects/fppc-tuned-embeddings/results/check1_leakage.json"


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_eval_queries() -> list[dict]:
    with open(EVAL_PATH) as f:
        return json.load(f)["queries"]


def load_corpus_index() -> dict[str, dict]:
    """Map opinion_id -> {question, question_synthetic, qa_text, year}."""
    idx: dict[str, dict] = {}
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
            sections = op.get("sections", {}) or {}
            idx[oid] = {
                "year": op.get("year"),
                "question": sections.get("question") or "",
                "question_synthetic": sections.get("question_synthetic") or "",
                "qa_text": (op.get("embedding") or {}).get("qa_text") or "",
                "has_standard_format": sections.get("has_standard_format"),
            }
    return idx


# ---------------------------------------------------------------------------
# Similarity primitives (lexical only — semantic comes from the embedding pass
# later if needed; for leakage detection lexical is usually decisive)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[’']s)?")
_STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "and", "or",
    "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "these", "those", "it", "its", "as", "by", "with", "from", "but",
    "if", "do", "does", "did", "may", "must", "can", "could", "would",
    "should", "shall", "will", "have", "has", "had", "any", "all",
    "not", "no", "than", "then", "there", "their", "they", "we", "you",
    "he", "she", "his", "her", "him", "i", "my", "me", "us", "our",
    "section", "subsection",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def jaccard(a: list[str], b: list[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def char_ngrams(text: str, n: int = 5) -> set[str]:
    s = re.sub(r"\s+", " ", text.lower()).strip()
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


def char_jaccard(a: str, b: str, n: int = 5) -> float:
    sa, sb = char_ngrams(a, n), char_ngrams(b, n)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def overlap_against_target(query: str, target: str) -> float:
    """Fraction of query content tokens that appear in target (one-sided).

    Useful when the eval query is a short keyword string and the opinion
    question is a long sentence — Jaccard would be diluted by length.
    """
    qa = set(tokenize(query))
    if not qa:
        return 0.0
    tb = set(tokenize(target))
    if not tb:
        return 0.0
    return len(qa & tb) / len(qa)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze() -> dict:
    queries = load_eval_queries()
    corpus = load_corpus_index()

    print(f"Loaded {len(queries)} eval queries, {len(corpus)} opinions")

    # For each query, look at relevant opinions and measure best similarity
    # against the question (and synthetic question, as fallback).
    per_query_records = []
    missing_opinions = []
    score2_seen = 0
    score1_seen = 0

    for q in queries:
        qid = q["id"]
        qtext = q["text"]
        qtype = q.get("type", "unknown")
        qtopic = q.get("topic", "unknown")

        judgments = q.get("relevance_judgments", [])
        per_judgment = []
        for j in judgments:
            oid = j["opinion_id"]
            score = j["score"]
            if score == 2:
                score2_seen += 1
            elif score == 1:
                score1_seen += 1

            op = corpus.get(oid)
            if op is None:
                missing_opinions.append(oid)
                continue

            real_q = op["question"]
            syn_q = op["question_synthetic"]
            # Use whichever exists; flag which one we ended up scoring against
            target_q = real_q if real_q.strip() else syn_q
            q_source = "real" if real_q.strip() else ("synthetic" if syn_q.strip() else "none")

            if not target_q.strip():
                per_judgment.append({
                    "opinion_id": oid,
                    "score": score,
                    "q_source": q_source,
                    "token_jaccard": None,
                    "char_jaccard_5": None,
                    "query_token_overlap": None,
                })
                continue

            per_judgment.append({
                "opinion_id": oid,
                "score": score,
                "q_source": q_source,
                "token_jaccard": round(jaccard(tokenize(qtext), tokenize(target_q)), 3),
                "char_jaccard_5": round(char_jaccard(qtext, target_q, 5), 3),
                "query_token_overlap": round(overlap_against_target(qtext, target_q), 3),
            })

        # Best-match against the top-scored relevant opinions
        def best(metric_key: str, score_filter: int) -> float | None:
            vals = [
                pj[metric_key]
                for pj in per_judgment
                if pj.get(metric_key) is not None and pj["score"] >= score_filter
            ]
            return max(vals) if vals else None

        per_query_records.append({
            "query_id": qid,
            "query_text": qtext,
            "query_type": qtype,
            "query_topic": qtopic,
            "n_judgments": len(judgments),
            "best_token_jaccard_score2": best("token_jaccard", 2),
            "best_char_jaccard_score2": best("char_jaccard_5", 2),
            "best_query_overlap_score2": best("query_token_overlap", 2),
            "best_token_jaccard_score1plus": best("token_jaccard", 1),
            "best_char_jaccard_score1plus": best("char_jaccard_5", 1),
            "best_query_overlap_score1plus": best("query_token_overlap", 1),
            "per_judgment": per_judgment,
        })

    # ----- Aggregate -----
    def summarize(values: list[float]) -> dict:
        values = [v for v in values if v is not None]
        if not values:
            return {"n": 0}
        return {
            "n": len(values),
            "mean": round(statistics.mean(values), 3),
            "median": round(statistics.median(values), 3),
            "p25": round(statistics.quantiles(values, n=4)[0], 3) if len(values) >= 4 else None,
            "p75": round(statistics.quantiles(values, n=4)[2], 3) if len(values) >= 4 else None,
            "max": round(max(values), 3),
            "min": round(min(values), 3),
        }

    overall_stats = {
        "best_token_jaccard_score2": summarize([r["best_token_jaccard_score2"] for r in per_query_records]),
        "best_char_jaccard_score2": summarize([r["best_char_jaccard_score2"] for r in per_query_records]),
        "best_query_overlap_score2": summarize([r["best_query_overlap_score2"] for r in per_query_records]),
        "best_token_jaccard_score1plus": summarize([r["best_token_jaccard_score1plus"] for r in per_query_records]),
        "best_char_jaccard_score1plus": summarize([r["best_char_jaccard_score1plus"] for r in per_query_records]),
        "best_query_overlap_score1plus": summarize([r["best_query_overlap_score1plus"] for r in per_query_records]),
    }

    # Buckets: count queries whose best score=2 match exceeds thresholds
    thresholds = [0.3, 0.5, 0.7, 0.85, 0.95]
    bucket_counts = {}
    for metric_key in ["best_token_jaccard_score2", "best_char_jaccard_score2", "best_query_overlap_score2"]:
        vals = [r[metric_key] for r in per_query_records if r[metric_key] is not None]
        bucket_counts[metric_key] = {
            f">={t}": sum(1 for v in vals if v >= t) for t in thresholds
        }
        bucket_counts[metric_key]["total_queries_with_values"] = len(vals)

    # By query type
    by_type = defaultdict(list)
    for r in per_query_records:
        by_type[r["query_type"]].append(r["best_query_overlap_score2"])
    by_type_summary = {k: summarize(v) for k, v in by_type.items()}

    # Set of opinions that are judged relevant in any eval query (score >= 1)
    eval_referenced_opinions: set[str] = set()
    for q in queries:
        for j in q.get("relevance_judgments", []):
            if j["score"] >= 1:
                eval_referenced_opinions.add(j["opinion_id"])

    # Coverage of the corpus's question fields
    n_with_real_q = sum(1 for v in corpus.values() if v["question"].strip())
    n_with_syn_q = sum(1 for v in corpus.values() if v["question_synthetic"].strip() and not v["question"].strip())
    n_with_neither = sum(1 for v in corpus.values() if not v["question"].strip() and not v["question_synthetic"].strip())

    # Top examples of high overlap — to eyeball
    top_examples = sorted(
        per_query_records,
        key=lambda r: (r["best_query_overlap_score2"] or 0),
        reverse=True,
    )[:10]
    top_examples_lite = [
        {
            "query_id": r["query_id"],
            "query_text": r["query_text"][:140],
            "query_type": r["query_type"],
            "best_query_overlap_score2": r["best_query_overlap_score2"],
            "best_token_jaccard_score2": r["best_token_jaccard_score2"],
            "best_char_jaccard_score2": r["best_char_jaccard_score2"],
            "top_match": _best_match_excerpt(r, corpus),
        }
        for r in top_examples
    ]

    return {
        "n_queries": len(queries),
        "n_opinions": len(corpus),
        "n_score2_judgments": score2_seen,
        "n_score1_judgments": score1_seen,
        "n_missing_opinions": len(missing_opinions),
        "missing_opinions_sample": missing_opinions[:10],
        "n_eval_referenced_opinions": len(eval_referenced_opinions),
        "pct_corpus_referenced": round(100 * len(eval_referenced_opinions) / len(corpus), 2),
        "corpus_q_coverage": {
            "with_real_question": n_with_real_q,
            "with_only_synthetic_question": n_with_syn_q,
            "with_neither": n_with_neither,
        },
        "overall_stats": overall_stats,
        "leakage_buckets_score2": bucket_counts,
        "by_query_type_query_overlap_score2": by_type_summary,
        "top_overlap_examples": top_examples_lite,
        "per_query_records": per_query_records,
    }


def _best_match_excerpt(record: dict, corpus: dict[str, dict]) -> dict | None:
    """Return the highest-overlap judgment record with excerpt of the opinion question."""
    score2_only = [pj for pj in record["per_judgment"] if pj["score"] == 2 and pj.get("query_token_overlap") is not None]
    if not score2_only:
        return None
    best = max(score2_only, key=lambda pj: pj["query_token_overlap"])
    op = corpus.get(best["opinion_id"], {})
    target_q = op.get("question") or op.get("question_synthetic", "")
    return {
        "opinion_id": best["opinion_id"],
        "q_source": best["q_source"],
        "query_token_overlap": best["query_token_overlap"],
        "token_jaccard": best["token_jaccard"],
        "char_jaccard_5": best["char_jaccard_5"],
        "opinion_question_excerpt": target_q[:300],
    }


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_summary(report: dict) -> None:
    print()
    print("=" * 78)
    print("Check 1: Eval-set leakage analysis")
    print("=" * 78)
    print(f"Queries: {report['n_queries']}    "
          f"Opinions: {report['n_opinions']}    "
          f"Score=2 judgments: {report['n_score2_judgments']}    "
          f"Score=1 judgments: {report['n_score1_judgments']}")
    print(f"Missing opinions referenced in eval: {report['n_missing_opinions']}")
    print(f"Eval-referenced opinions: {report['n_eval_referenced_opinions']}"
          f"  ({report['pct_corpus_referenced']}% of corpus)")
    cov = report["corpus_q_coverage"]
    print(f"Corpus question coverage: real={cov['with_real_question']}  "
          f"synthetic-only={cov['with_only_synthetic_question']}  "
          f"neither={cov['with_neither']}")

    print()
    print("Best-match similarity to the top-relevant (score=2) opinion's question")
    print("(per query, then aggregated across the 65 queries):")
    print()
    for metric, label in [
        ("best_query_overlap_score2", "query-token overlap (asymmetric)"),
        ("best_token_jaccard_score2", "token Jaccard"),
        ("best_char_jaccard_score2", "char 5-gram Jaccard"),
    ]:
        s = report["overall_stats"][metric]
        if s.get("n"):
            print(f"  {label:<36}  "
                  f"median={s['median']:.2f}  mean={s['mean']:.2f}  "
                  f"p75={s['p75']}  max={s['max']}")

    print()
    print("Leakage buckets (queries with best score=2 match above threshold):")
    for metric_key, label in [
        ("best_query_overlap_score2", "query-token overlap"),
        ("best_token_jaccard_score2", "token Jaccard"),
        ("best_char_jaccard_score2", "char 5-gram Jaccard"),
    ]:
        buckets = report["leakage_buckets_score2"][metric_key]
        total = buckets.pop("total_queries_with_values")
        line = "  " + label.ljust(22) + " "
        line += "  ".join(f"{thr} -> {n}/{total}" for thr, n in buckets.items())
        print(line)

    print()
    print("By query type (query-token overlap vs best score=2 opinion question):")
    for qtype, s in report["by_query_type_query_overlap_score2"].items():
        if s.get("n"):
            print(f"  {qtype:<24}  n={s['n']:>3}  "
                  f"median={s['median']:.2f}  mean={s['mean']:.2f}  "
                  f"max={s['max']}")

    print()
    print("Top 10 highest-overlap examples (the most suspicious for leakage):")
    for ex in report["top_overlap_examples"]:
        m = ex["top_match"]
        if not m:
            continue
        print(f"  - [{ex['query_id']}  type={ex['query_type']:<16}]  "
              f"overlap={ex['best_query_overlap_score2']:.2f}  "
              f"tok_jac={ex['best_token_jaccard_score2']:.2f}  "
              f"char_jac={ex['best_char_jaccard_score2']:.2f}")
        print(f"      query    : {ex['query_text']}")
        print(f"      match  ({m['opinion_id']}, {m['q_source']}): "
              f"{m['opinion_question_excerpt'][:200]}")
        print()


def main():
    report = analyze()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print_summary(report)
    print(f"Wrote full report to {OUT_PATH}")


if __name__ == "__main__":
    main()
