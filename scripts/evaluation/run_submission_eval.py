"""The 1,000-question offline evaluation, end to end, with no provider call.

    .venv/bin/python scripts/evaluation/run_submission_eval.py

One pass over `data/evaluation/synthetic_1000.jsonl`, through the retrieval path
the application actually ships — intent gate, spelling correction, candidate
generation, reranking — recording for every question what came back, where the
target landed, and whether the evidence was enough to answer from.

Three things this deliberately does not do
------------------------------------------

**It does not call Groq, or any provider.** Answer generation is a separate
measurement on a separate sample (`--answers` in `run_synthetic_benchmark.py`,
and Part F of the submission brief). Nothing here produces or judges prose.

**It does not reimplement retrieval.** The classes are the ones `app/pipeline.py`
constructs, at the depths `app/pipeline.py` uses. A benchmark that builds its own
retriever measures the benchmark.

**It does not read the cache.** The validated-answer cache exists to make the
demo fast; treating it as an evaluation substrate would measure what was already
known rather than what the pipeline does.

What it records
---------------

Per question: the ranked ids at 1/5/10, the rank of the target, hit flags, MRR,
retrieval and rerank latency, pool size, and the evidence-sufficiency verdict at
5 and at 10 (see `evidence_sufficiency.py`). The 40 EXPECTED_ABSTENTION questions
have no retrieval target and are scored on a different axis entirely — whether
the corpus offers them anything — and are never mixed into the recall
denominators.

Checkpointing
-------------

Rows are flushed every `BATCH` questions to the output file and the run can be
resumed: an existing output file is read first and its completed `question_id`s
are skipped. A failed run costs the batch it was in, not the hour before it.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import statistics
import sys
import time
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.reranking.features import FeatureReranker  # noqa: E402
from ai.retrieval.base import load_chunks  # noqa: E402
from ai.retrieval.bm25 import BM25Retriever  # noqa: E402
from ai.retrieval.candidates import CandidateGenerator  # noqa: E402
from ai.retrieval.dense import DenseRetriever  # noqa: E402
from ai.retrieval.identifiers import BM25IdentifierBoostRetriever  # noqa: E402
from ai.text_integrity import assess as assess_integrity  # noqa: E402
from app.intent import classify as classify_intent  # noqa: E402
from app.spelling import CorpusSpelling  # noqa: E402

from evaluation.evidence_sufficiency import (  # noqa: E402
    INSUFFICIENT, SUFFICIENT, UNCERTAIN, assess as assess_sufficiency,
    sufficiency_config)
from evaluation.metrics import METRIC_KEYS, aggregate, first_hit_rank, score_question  # noqa: E402
from evaluation.target_equivalence import EquivalenceIndex, equivalence_config  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "data", "evaluation", "synthetic_1000.jsonl")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation", "results", "submission")

BATCH = 100
TOP_K = 10

CLAIM_SAFETY = (
    "AI-generated corpus-grounded evaluation set, scored deterministically "
    "offline. Questions and expected answers are derived from the fixed corpus "
    "by template over verbatim spans; they are not human-verified legal ground "
    "truth. Evidence sufficiency is a property of the retrieved evidence, not a "
    "measure of legal accuracy."
)


def load_questions(path: str = QUESTIONS_PATH) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentiles(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"median": 0.0, "mean": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)

    def at(fraction: float) -> float:
        return ordered[min(len(ordered) - 1, int(fraction * len(ordered)))]

    return {"median": round(statistics.median(ordered), 2),
            "mean": round(statistics.mean(ordered), 2),
            "p90": round(at(0.90), 2), "p95": round(at(0.95), 2),
            "max": round(ordered[-1], 2)}


# ------------------------------------------------------------ the retrieval --

class Retrieval:
    """The shipped path, built once."""

    def __init__(self) -> None:
        self.chunks = load_chunks()
        self.by_id = {chunk.chunk_id: chunk for chunk in self.chunks}
        self.bm25 = BM25Retriever(self.chunks)
        self.lexical = BM25IdentifierBoostRetriever(self.bm25)
        self.dense = DenseRetriever(self.chunks)
        self.dense.embed_query("warm")
        self.generator = CandidateGenerator(
            dense=self.dense, lexical=self.lexical, mode="expanded",
            order="interleaved", dense_depth=50, lexical_depth=50,
            aliases=self.lexical.aliases)
        self.reranker = FeatureReranker(idf=self.bm25.idf, aliases=self.lexical.aliases)
        self.speller = CorpusSpelling(chunk.text for chunk in self.chunks)
        self.equivalence = EquivalenceIndex.build(self.chunks)

    def run(self, question: str) -> dict[str, Any]:
        """Intent gate, spelling, candidates, rerank — in the shipped order."""
        social = classify_intent(question, self.lexical.aliases)
        if social is not None:
            return {"social": social.to_dict(), "ranked": [], "pool_size": 0,
                    "retrieval_ms": 0.0, "rerank_ms": 0.0,
                    "search_query": question, "spelling_applied": False}
        correction = self.speller.correct(question)
        search_query = correction.corrected
        t0 = time.perf_counter()
        pool = self.generator.generate(search_query)
        t1 = time.perf_counter()
        ranked = self.reranker.rerank(search_query, pool.candidates, TOP_K)
        t2 = time.perf_counter()
        return {
            "social": None,
            "ranked": ranked,
            "pool": pool,
            "pool_size": len(pool.candidates),
            "retrieval_ms": (t1 - t0) * 1000,
            "rerank_ms": (t2 - t1) * 1000,
            "search_query": search_query,
            "spelling_applied": correction.applied,
            "spelling": correction.to_dict(),
        }

    def evidence(self, ranked: Sequence[Any]) -> list[dict[str, Any]]:
        out = []
        for item in ranked:
            chunk = self.by_id.get(item.chunk_id)
            out.append({
                "chunk_id": item.chunk_id,
                "document_id": item.document_id,
                "text": chunk.text if chunk else "",
                "locator": dict(chunk.locator) if chunk else {},
                "citation": chunk.citation if chunk else "",
                "heading": (chunk.heading if chunk else None),
                "pages": list(chunk.physical_pages) if chunk else [],
                "temporal_status": chunk.temporal_status if chunk else None,
                "evidence_level": chunk.source_evidence_level if chunk else None,
                "rerank_score": round(item.rerank_score, 6),
                "components": {k: round(float(v), 6)
                               for k, v in dict(item.components).items()},
            })
        return out


# --------------------------------------------------------------- the run ----

def completed(path: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    rows: dict[str, dict[str, Any]] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows[row["question_id"]] = row
    return rows


def evaluate(engine: Retrieval, questions: Sequence[dict[str, Any]],
             rows_path: str, *, resume: bool) -> list[dict[str, Any]]:
    done = completed(rows_path) if resume else {}
    if done:
        print(f"  resuming: {len(done)} questions already recorded", flush=True)
    pending = [q for q in questions if q["question_id"] not in done]
    handle = open(rows_path, "a" if done else "w", encoding="utf-8")
    try:
        for index, record in enumerate(pending, start=1):
            row = evaluate_one(engine, record)
            done[row["question_id"]] = row
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if index % BATCH == 0:
                handle.flush()
                os.fsync(handle.fileno())
                print(f"  {index}/{len(pending)} (checkpointed)", flush=True)
    finally:
        handle.flush()
        handle.close()
    order = {q["question_id"]: i for i, q in enumerate(questions)}
    return sorted(done.values(), key=lambda r: order.get(r["question_id"], 1 << 30))


def evaluate_one(engine: Retrieval, record: dict[str, Any]) -> dict[str, Any]:
    result = engine.run(record["question"])
    ranked = result["ranked"]
    ids = [r.chunk_id for r in ranked]
    documents = [r.document_id for r in ranked]
    evidence = engine.evidence(ranked)
    answerable = record["label_kind"] == "ANSWERABLE"

    targets = list(record.get("target_chunk_ids") or [])
    accepted: list[str] = targets
    rules: dict[str, int] = {}
    if answerable and targets:
        widened, rules = engine.equivalence.acceptable(record["question"], targets)
        accepted = sorted(widened)

    metrics = score_question(ids, targets, documents, [record["document_id"]]) \
        if answerable and targets else {}
    adjusted = score_question(ids, accepted, documents, [record["document_id"]]) \
        if answerable and targets else {}

    target_locators = [dict(engine.by_id[cid].locator)
                       for cid in targets if cid in engine.by_id]
    spans = record.get("expected_answer_spans") or []
    suff5 = suff10 = None
    if answerable:
        common = dict(expected_answer=record["expected_answer"], expected_spans=spans,
                      retrieved=evidence, target_chunk_ids=targets,
                      acceptable_chunk_ids=accepted,
                      target_document_id=record.get("document_id"),
                      target_locators=target_locators, idf=engine.bm25.idf)
        suff5 = assess_sufficiency(**common, depth=5).to_dict()
        suff10 = assess_sufficiency(**common, depth=10).to_dict()

    rank = first_hit_rank(ids, targets) if targets else None
    damaged = [e["chunk_id"] for e in evidence[:5]
               if assess_integrity(e["text"]).status != "CLEAN"]

    return {
        "question_id": record["question_id"],
        "question": record["question"],
        "expected_answer": record["expected_answer"],
        "label_kind": record["label_kind"],
        "answer_type": record["answer_type"],
        "difficulty": record["difficulty"],
        "register": record["register"],
        "domain": record["domain"],
        "target_document_id": record.get("document_id"),
        "target_chunk_ids": targets,
        "acceptable_chunk_ids": accepted,
        "equivalence_rules": rules,
        "locator": record.get("locator"),
        "temporal_status": record.get("temporal_status"),
        "search_query": result["search_query"],
        "spelling_applied": result["spelling_applied"],
        "social_intent": (result["social"] or {}).get("intent"),
        "rank_of_target": rank,
        "rank_of_acceptable": first_hit_rank(ids, accepted) if accepted else None,
        "top1_chunk_id": ids[0] if ids else None,
        "top5_chunk_ids": ids[:5],
        "top10_chunk_ids": ids[:10],
        "retrieval_hit_at_1": bool(metrics.get("recall@1")) if metrics else None,
        "retrieval_hit_at_5": bool(metrics.get("recall@5")) if metrics else None,
        "retrieval_hit_at_10": bool(metrics.get("recall@10")) if metrics else None,
        "mrr": round(metrics.get("mrr", 0.0), 6) if metrics else None,
        "metrics": {k: round(v, 6) for k, v in metrics.items()},
        "metrics_equivalence_adjusted": {k: round(v, 6) for k, v in adjusted.items()},
        "retrieval_latency_ms": round(result["retrieval_ms"], 2),
        "rerank_latency_ms": round(result["rerank_ms"], 2),
        "total_latency_ms": round(result["retrieval_ms"] + result["rerank_ms"], 2),
        "candidate_count": result["pool_size"],
        "reranked_count": len(ids),
        "answerable_from_top5": (suff5 or {}).get("verdict"),
        "answerable_from_top10": (suff10 or {}).get("verdict"),
        "sufficiency_at_5": suff5,
        "sufficiency_at_10": suff10,
        "damaged_chunks_in_top5": damaged,
        "evidence": [{k: v for k, v in e.items() if k != "text"} for e in evidence],
    }


# ------------------------------------------------------------ the summary ----

def summarise(rows: Sequence[dict[str, Any]], questions: Sequence[dict[str, Any]],
              wall_seconds: float) -> dict[str, Any]:
    answerable = [r for r in rows if r["label_kind"] == "ANSWERABLE"]
    abstention = [r for r in rows if r["label_kind"] != "ANSWERABLE"]
    per_q = [r["metrics"] for r in answerable]
    per_adj = [r["metrics_equivalence_adjusted"] for r in answerable]
    overall = aggregate(per_q, METRIC_KEYS)
    overall_adj = aggregate(per_adj, METRIC_KEYS)

    def counts(scores: dict[str, float]) -> dict[str, int]:
        return {k: int(round(v * len(answerable))) for k, v in scores.items()
                if k.startswith("recall")}

    def sufficiency(depth: str) -> dict[str, Any]:
        tally = collections.Counter(r[f"answerable_from_top{depth}"] for r in answerable)
        target_in = sum(1 for r in answerable
                        if r[f"sufficiency_at_{depth}"]["acceptable_included"])
        return {
            "denominator": len(answerable),
            "sufficient": tally[SUFFICIENT],
            "insufficient": tally[INSUFFICIENT],
            "uncertain": tally[UNCERTAIN],
            "rate_sufficient": round(tally[SUFFICIENT] / len(answerable), 4),
            "target_retrieved_but_insufficient": sum(
                1 for r in answerable
                if r[f"sufficiency_at_{depth}"]["acceptable_included"]
                and r[f"answerable_from_top{depth}"] != SUFFICIENT),
            "target_not_retrieved": len(answerable) - target_in,
            "sufficient_without_exact_target": sum(
                1 for r in answerable
                if r[f"answerable_from_top{depth}"] == SUFFICIENT
                and not r[f"sufficiency_at_{depth}"]["target_included"]),
        }

    def segment(key: str) -> dict[str, Any]:
        groups: dict[str, list[dict[str, float]]] = collections.defaultdict(list)
        for row in answerable:
            groups[str(row[key])].append(row["metrics"])
        return {name: {"n": len(group),
                       **{k: round(v, 4) for k, v in aggregate(group, METRIC_KEYS).items()}}
                for name, group in sorted(groups.items())}

    buckets: collections.Counter = collections.Counter()
    for row in answerable:
        rank = row["rank_of_target"]
        if rank == 1:
            buckets["rank 1"] += 1
        elif rank and rank <= 5:
            buckets["rank 2-5"] += 1
        elif rank and rank <= 10:
            buckets["rank 6-10"] += 1
        else:
            buckets[">10 / missing"] += 1

    widened = sum(1 for r in answerable
                  if len(r["acceptable_chunk_ids"]) > len(r["target_chunk_ids"]))
    by_rule: collections.Counter = collections.Counter()
    for row in answerable:
        for rule, count in (row.get("equivalence_rules") or {}).items():
            by_rule[rule] += count

    return {
        "metric": "offline_retrieval_and_evidence_sufficiency",
        "deterministic": True,
        "provider_calls": 0,
        "questions_in_set": len(questions),
        "questions_evaluated": len(rows),
        "answerable": len(answerable),
        "expected_abstention": len(abstention),
        "wall_seconds": round(wall_seconds, 1),
        "denominator_note": (
            f"every retrieval rate below is out of {len(answerable)} answerable "
            f"questions; the {len(abstention)} EXPECTED_ABSTENTION questions have "
            f"no retrieval target and are excluded from the recall denominators"),
        "retrieval": {
            "aggregate": {k: round(v, 4) for k, v in overall.items()},
            "counts": counts(overall),
            "equivalence_adjusted": {
                "aggregate": {k: round(v, 4) for k, v in overall_adj.items()},
                "counts": counts(overall_adj),
                "questions_widened": widened,
                "chunks_added_by_rule": dict(by_rule),
                "config": equivalence_config(),
            },
            "rank_buckets": dict(buckets),
            "target_in_top5_not_top1": buckets["rank 2-5"],
        },
        "evidence_sufficiency": {
            "at_5": sufficiency("5"),
            "at_10": sufficiency("10"),
            "config": sufficiency_config(),
        },
        "latency_ms": {
            "retrieval": percentiles([r["retrieval_latency_ms"] for r in answerable]),
            "reranking": percentiles([r["rerank_latency_ms"] for r in answerable]),
            "retrieval_plus_reranking": percentiles(
                [r["total_latency_ms"] for r in answerable]),
        },
        "pool": {
            "median_candidates": int(statistics.median(
                [r["candidate_count"] for r in answerable])) if answerable else 0,
            "reranked_evidence_count": TOP_K,
        },
        "spelling": {
            "questions_altered": sum(1 for r in rows if r["spelling_applied"]),
            "denominator": len(rows),
        },
        "intent_gate": {
            "questions_captured": sum(1 for r in rows if r["social_intent"]),
            "denominator": len(rows),
        },
        "source_integrity": {
            "questions_with_damaged_chunk_in_top5": sum(
                1 for r in rows if r["damaged_chunks_in_top5"]),
            "denominator": len(rows),
        },
        "by_difficulty": segment("difficulty"),
        "by_answer_type": segment("answer_type"),
        "by_domain": segment("domain"),
        "by_document": segment("target_document_id"),
        "claim_safety": CLAIM_SAFETY,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", default=QUESTIONS_PATH)
    parser.add_argument("--out", default=RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fresh", action="store_true",
                        help="ignore any existing rows and start over")
    args = parser.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    rows_path = os.path.join(args.out, "retrieval_1000.jsonl")
    summary_path = os.path.join(args.out, "retrieval_1000.summary.json")

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[:args.limit]
    print(f"loaded {len(questions)} questions", flush=True)

    started = time.perf_counter()
    engine = Retrieval()
    print(f"retrieval built in {time.perf_counter() - started:.1f}s "
          f"({len(engine.chunks)} chunks)", flush=True)

    run_started = time.perf_counter()
    rows = evaluate(engine, questions, rows_path, resume=not args.fresh)
    wall = time.perf_counter() - run_started

    summary = summarise(rows, questions, wall)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    agg = summary["retrieval"]["aggregate"]
    cnt = summary["retrieval"]["counts"]
    n = summary["answerable"]
    print(f"\nR@1 {cnt['recall@1']}/{n} = {agg['recall@1']:.4f}   "
          f"R@5 {cnt['recall@5']}/{n} = {agg['recall@5']:.4f}   "
          f"R@10 {cnt['recall@10']}/{n} = {agg['recall@10']:.4f}   "
          f"MRR {agg['mrr']:.4f}")
    s5 = summary["evidence_sufficiency"]["at_5"]
    s10 = summary["evidence_sufficiency"]["at_10"]
    print(f"Sufficiency@5 {s5['sufficient']}/{n}   Sufficiency@10 {s10['sufficient']}/{n}")
    print(f"wall {wall:.1f}s -> {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
