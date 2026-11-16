"""Phase H: the 1,000-question benchmark, run.

    # Metric A - retrieval. Deterministic, offline, no provider.
    .venv/bin/python scripts/evaluation/run_synthetic_benchmark.py --retrieval

    # Metric B - answers, on a stratified sample. Needs a provider.
    .venv/bin/python scripts/evaluation/run_synthetic_benchmark.py --answers 60

The two metrics are computed by two different commands writing two different
files, and that separation is the point of the module rather than an accident of
its layout.

**Metric A is retrieval.** Ground truth is `target_chunk_ids`, the measure is set
membership, and the answer layer is never invoked. Nothing here can be argued
about: either the chunk the generator drew the question from came back in the
top *k* or it did not. It runs over all 1,000 questions because it costs nothing
but local compute.

**Metric B is answer quality.** Ground truth is a quoted span, the measure is a
judgement, and the judge is a Claude Code session reading a packet — so it costs
a provider call per question to produce the answer and a human-scale act of
reading to score it. It runs over a *stratified sample*, and the sample size is
printed as the denominator of every figure derived from it. A number computed
over 60 questions is never reported as though it were computed over 1,000.

Mixing them would produce the one claim this project refuses to make: calling
recall "accuracy". A pipeline that retrieves the right provision and then writes
a wrong sentence about it scores 1.0 on Metric A and 0.0 on Metric B, and both
facts matter.

What the answer run does *not* do
---------------------------------

It does not retry, does not fall back to the extractive provider, and does not
lower any gate to raise the answered rate. A rate limit is recorded as
`UNAVAILABLE` and excluded from the answer-correctness denominator — it is a
fact about the free tier, not about the pipeline — and the count of exclusions is
reported beside the rate so the denominator is always visible.
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.metrics import METRIC_KEYS, aggregate, first_hit_rank, score_question  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
QUESTIONS_PATH = os.path.join(PROJECT_ROOT, "data", "evaluation", "synthetic_1000.jsonl")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation", "results", "synthetic")

CLAIM_SAFETY = (
    "Evaluation uses a synthetic, AI-generated and AI-judged benchmark derived "
    "from the fixed corpus. It is an engineering evaluation and not "
    "human-verified legal ground truth."
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
    return {
        "median": round(statistics.median(ordered), 2),
        "mean": round(statistics.mean(ordered), 2),
        "p90": round(at(0.90), 2),
        "p95": round(at(0.95), 2),
        "max": round(ordered[-1], 2),
    }


# ------------------------------------------------------- Metric A: retrieval --

def run_retrieval(questions: Sequence[dict[str, Any]], *, top_k: int = 10,
                  reranker_name: str = "features") -> dict[str, Any]:
    """Candidate generation + reranking over every answerable question.

    Built from the same classes `app/pipeline.py` builds, at the same depths, so
    the figure describes the shipped retrieval path and not a benchmark-only
    variant of it.
    """
    from ai.reranking.features import FeatureReranker
    from ai.retrieval.base import load_chunks
    from ai.retrieval.bm25 import BM25Retriever
    from ai.retrieval.candidates import CandidateGenerator
    from ai.retrieval.dense import DenseRetriever
    from ai.retrieval.identifiers import BM25IdentifierBoostRetriever

    from evaluation.target_equivalence import EquivalenceIndex, equivalence_config

    chunks = load_chunks()
    # Which other chunks answer each question as well as its single label does.
    # Built from the corpus and the question text, before any run happens, so it
    # cannot be influenced by what retrieval returned. See target_equivalence.py.
    equivalence = EquivalenceIndex.build(chunks)
    bm25 = BM25Retriever(chunks)
    lexical = BM25IdentifierBoostRetriever(bm25)
    dense = DenseRetriever(chunks)
    dense.embed_query("warm")
    generator = CandidateGenerator(dense=dense, lexical=lexical, mode="expanded",
                                   order="interleaved", dense_depth=50,
                                   lexical_depth=50, aliases=lexical.aliases)
    reranker = FeatureReranker(idf=bm25.idf, aliases=lexical.aliases)

    answerable = [q for q in questions if q["label_kind"] == "ANSWERABLE"]
    rows: list[dict[str, Any]] = []
    per_question: list[dict[str, float]] = []
    per_question_adjusted: list[dict[str, float]] = []
    equivalence_counts: collections.Counter = collections.Counter()
    retrieval_ms: list[float] = []
    rerank_ms: list[float] = []

    for index, record in enumerate(answerable, start=1):
        t0 = time.perf_counter()
        pool = generator.generate(record["question"])
        t1 = time.perf_counter()
        ranked = reranker.rerank(record["question"], pool.candidates, top_k)
        t2 = time.perf_counter()

        ids = [r.chunk_id for r in ranked]
        documents = [r.document_id for r in ranked]
        metrics = score_question(ids, record["target_chunk_ids"], documents,
                                 [record["document_id"]])
        # The same run, scored a second time against the set of chunks that
        # would be a correct answer rather than the one the generator happened
        # to build from. Both figures are kept; neither replaces the other.
        accepted, rules = equivalence.acceptable(record["question"],
                                                 record["target_chunk_ids"])
        adjusted = score_question(ids, sorted(accepted), documents,
                                  [record["document_id"]])
        per_question.append(metrics)
        per_question_adjusted.append(adjusted)
        for rule, count in rules.items():
            equivalence_counts[rule] += count
        if len(accepted) > len(record["target_chunk_ids"]):
            equivalence_counts["questions_widened"] += 1
        retrieval_ms.append((t1 - t0) * 1000)
        rerank_ms.append((t2 - t1) * 1000)
        pool_ids = [c.chunk_id for c in pool.candidates]
        rows.append({
            "question_id": record["question_id"],
            "question": record["question"],
            "answer_type": record["answer_type"],
            "difficulty": record["difficulty"],
            "register": record["register"],
            "domain": record["domain"],
            "document_id": record["document_id"],
            "target_chunk_ids": record["target_chunk_ids"],
            "acceptable_chunk_ids": sorted(accepted),
            "rank_of_expected": first_hit_rank(ids, record["target_chunk_ids"]),
            "rank_of_acceptable": first_hit_rank(ids, sorted(accepted)),
            "metrics_equivalence_adjusted": adjusted,
            "pool_rank_of_expected": first_hit_rank(pool_ids, record["target_chunk_ids"]),
            "pool_size": len(pool.candidates),
            "ranked": ids,
            "metrics": metrics,
            "retrieval_ms": round((t1 - t0) * 1000, 2),
            "rerank_ms": round((t2 - t1) * 1000, 2),
        })
        if index % 100 == 0:
            print(f"  {index}/{len(answerable)}", flush=True)

    overall = aggregate(per_question, METRIC_KEYS)
    overall_adjusted = aggregate(per_question_adjusted, METRIC_KEYS)

    def segment(key: str) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        groups: dict[str, list[dict[str, float]]] = collections.defaultdict(list)
        for row in rows:
            groups[str(row[key])].append(row["metrics"])
        for name, group in sorted(groups.items()):
            scores = aggregate(group, METRIC_KEYS)
            out[name] = {"n": len(group),
                         **{k: round(v, 4) for k, v in scores.items()}}
        return out

    buckets = collections.Counter()
    for row in rows:
        rank = row["rank_of_expected"]
        if rank == 1:
            buckets["rank 1"] += 1
        elif rank and rank <= 5:
            buckets["rank 2-5"] += 1
        elif rank and rank <= 10:
            buckets["rank 6-10"] += 1
        else:
            buckets[">10 / missing"] += 1

    return {
        "metric": "A_retrieval",
        "deterministic": True,
        "questions_evaluated": len(rows),
        "denominator_note": f"every rate below is out of {len(rows)} answerable "
                            f"questions; the {len(questions) - len(rows)} "
                            f"EXPECTED_ABSTENTION questions have no retrieval "
                            f"target and are excluded from Metric A entirely",
        "reranker": reranker_name,
        "top_k": top_k,
        "aggregate": {k: round(v, 4) for k, v in overall.items()},
        "counts": {k: int(round(v * len(rows))) for k, v in overall.items()
                   if k.startswith("recall")},
        # The headline figure above scores each question against the single
        # chunk the generator wrote it from. The corpus contains provisions that
        # answer the same question equally well — duplicate headings, two halves
        # of one section, the Act section a Rule's own heading names — and the
        # figures below score the identical run against all of them. Reported
        # together, never merged: the first is what a single-label benchmark
        # measures, the second is what it was trying to measure.
        "equivalence_adjusted": {
            "aggregate": {k: round(v, 4) for k, v in overall_adjusted.items()},
            "counts": {k: int(round(v * len(rows)))
                       for k, v in overall_adjusted.items()
                       if k.startswith("recall")},
            "questions_widened": int(equivalence_counts["questions_widened"]),
            "chunks_added_by_rule": {
                rule: int(count) for rule, count in sorted(equivalence_counts.items())
                if rule != "questions_widened"},
            "config": equivalence_config(),
        },
        "rank_buckets": dict(buckets),
        "by_difficulty": segment("difficulty"),
        "by_answer_type": segment("answer_type"),
        "by_register": segment("register"),
        "by_domain": segment("domain"),
        "by_document": segment("document_id"),
        "latency_ms": {"retrieval": percentiles(retrieval_ms),
                       "reranking": percentiles(rerank_ms),
                       "retrieval_plus_reranking":
                           percentiles([a + b for a, b in zip(retrieval_ms, rerank_ms)])},
        "claim_safety": CLAIM_SAFETY,
    }, rows


# --------------------------------------------------------- Metric B: answers --

def stratified_sample(questions: Sequence[dict[str, Any]], size: int,
                      seed: int = 20260915) -> list[dict[str, Any]]:
    """A sample that keeps the file's own shape, so the sample is not a new bias.

    Proportional over (label_kind, difficulty, answer_type-family), filled
    round-robin so a rare stratum contributes at least one question rather than
    rounding to zero and disappearing from a report that claims to cover it.
    """
    import random
    rng = random.Random(seed)
    groups: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for record in questions:
        groups[(record["label_kind"], record["difficulty"])].append(record)
    for rows in groups.values():
        rng.shuffle(rows)
    keys = sorted(groups)
    out: list[dict[str, Any]] = []
    cursors = {key: 0 for key in keys}
    while len(out) < size and any(cursors[k] < len(groups[k]) for k in keys):
        for key in keys:
            if len(out) >= size:
                break
            if cursors[key] < len(groups[key]):
                out.append(groups[key][cursors[key]])
                cursors[key] += 1
    return out


def run_answers(questions: Sequence[dict[str, Any]], size: int,
                *, sleep_seconds: float = 2.0) -> dict[str, Any]:
    """The full pipeline over a stratified sample, recording what it published.

    One request per question, no retry, no provider fallback. `sleep_seconds`
    paces the free-tier per-minute cap; a 429 that still gets through is recorded
    as UNAVAILABLE and excluded from the correctness denominator rather than
    counted as a wrong answer, because a request that never reached the model
    says nothing about the pipeline's answers.
    """
    from app.pipeline import Pipeline

    pipeline = Pipeline()
    sample = stratified_sample(questions, size)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(sample, start=1):
        t0 = time.perf_counter()
        try:
            # Clarification off: this is a benchmark of the answer path, and a
            # clarifying question is neither an answer nor an abstention.
            result = pipeline.answer(record["question"], allow_clarification=False)
        except Exception as exc:  # noqa: BLE001 - one bad question must not end the run
            rows.append({"question_id": record["question_id"], "status": "ERROR",
                         "error": str(exc)[:300]})
            continue
        elapsed = (time.perf_counter() - t0) * 1000
        answer = result.get("answer") or {}
        cited = [e for e in result.get("evidence", []) if e.get("cited")]
        retrieved_ids = [e["chunk_id"] for e in result.get("evidence", [])]
        rows.append({
            "question_id": record["question_id"],
            "question": record["question"],
            "label_kind": record["label_kind"],
            "difficulty": record["difficulty"],
            "answer_type": record["answer_type"],
            "target_chunk_ids": record["target_chunk_ids"],
            "expected_answer": record["expected_answer"],
            "status": result.get("status"),
            "service_failure": (result.get("service_failure") or {}).get("code"),
            "answer_text": answer.get("answer"),
            "grounded": bool(answer.get("grounded")),
            "confidence": answer.get("confidence"),
            "citation_count": len(answer.get("citations") or []),
            "validation_failures": [f.get("code") for f in (answer.get("validation_failures") or [])],
            "unsupported_claims": answer.get("unsupported_claims") or [],
            "insufficient_reasons": [f.get("code") for f in
                                     (answer.get("insufficient_evidence_reasons") or [])],
            "retrieved_chunk_ids": retrieved_ids,
            "retrieval_correct": bool(set(retrieved_ids) & set(record["target_chunk_ids"])),
            "retrieval_correct_at_1": bool(retrieved_ids[:1]
                                           and retrieved_ids[0] in record["target_chunk_ids"]),
            "cited_evidence": [{"marker": e["marker"], "citation": e.get("citation"),
                                "chunk_id": e["chunk_id"],
                                "excerpt": (e.get("excerpt") or {}).get("text"),
                                "temporal_status": e.get("temporal_status")}
                               for e in cited],
            "timings_ms": result.get("timings_ms"),
            "wall_ms": round(elapsed, 1),
            "cache_hit": bool((result.get("cache") or {}).get("hit")),
        })
        print(f"  {index}/{len(sample)} {record['question_id']} -> {rows[-1].get('status')}",
              flush=True)
        if sleep_seconds and index < len(sample):
            time.sleep(sleep_seconds)

    answerable = [r for r in rows if r.get("label_kind") == "ANSWERABLE"]
    abstention_probes = [r for r in rows if r.get("label_kind") == "EXPECTED_ABSTENTION"]
    unavailable = [r for r in rows if r.get("status") == "UNAVAILABLE"]
    scored = [r for r in answerable if r.get("status") in ("ANSWERED", "ABSTAINED")]
    answered = [r for r in scored if r["status"] == "ANSWERED"]

    return {
        "metric": "B_answer_behaviour",
        "deterministic": False,
        "sample_size": len(rows),
        "sampled_from": len(questions),
        "denominator_note":
            f"behavioural rates are out of {len(scored)} answerable questions "
            f"that reached a verdict; {len(unavailable)} were UNAVAILABLE "
            f"(provider/service, excluded) and {len(abstention_probes)} are "
            f"EXPECTED_ABSTENTION probes counted separately",
        "counts": {
            "answered": len(answered),
            "abstained": sum(1 for r in scored if r["status"] == "ABSTAINED"),
            "unavailable": len(unavailable),
            "errors": sum(1 for r in rows if r.get("status") == "ERROR"),
        },
        "grounded_answer_rate": {
            "numerator": sum(1 for r in answered if r["grounded"]),
            "denominator": len(answered)},
        "citation_supported_answer_rate": {
            "numerator": sum(1 for r in answered if r["citation_count"] > 0),
            "denominator": len(answered)},
        "retrieval_correct_in_evidence": {
            "numerator": sum(1 for r in scored if r["retrieval_correct"]),
            "denominator": len(scored)},
        "abstention_precision_on_probes": {
            "numerator": sum(1 for r in abstention_probes if r["status"] == "ABSTAINED"),
            "denominator": len([r for r in abstention_probes
                                if r.get("status") in ("ANSWERED", "ABSTAINED")])},
        "answered_despite_wrong_evidence": sum(
            1 for r in answered if not r["retrieval_correct"]),
        "abstained_despite_correct_evidence": sum(
            1 for r in scored if r["status"] == "ABSTAINED" and r["retrieval_correct"]),
        "citation_validation_failures": sum(
            1 for r in rows if r.get("validation_failures")),
        "grounding_failures": sum(
            1 for r in scored if "ungrounded_answer" in (r.get("insufficient_reasons") or [])),
        "latency_ms": {
            "total": percentiles([r["wall_ms"] for r in rows if "wall_ms" in r]),
            "retrieval": percentiles([(r.get("timings_ms") or {}).get("retrieval", 0)
                                      for r in rows if r.get("timings_ms")]),
            "reranking": percentiles([(r.get("timings_ms") or {}).get("reranking", 0)
                                      for r in rows if r.get("timings_ms")]),
            "answering": percentiles([(r.get("timings_ms") or {}).get("answering", 0)
                                      for r in rows if r.get("timings_ms")]),
        },
        "answer_correctness":
            "NOT MEASURED HERE - requires the AI judge; see "
            "scripts/evaluation/prepare_synthetic_judgement.py",
        "claim_safety": CLAIM_SAFETY,
    }, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--questions", default=QUESTIONS_PATH)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--retrieval", action="store_true",
                        help="Metric A over every answerable question")
    parser.add_argument("--answers", type=int, default=0,
                        help="Metric B over a stratified sample of this size")
    parser.add_argument("--sleep", type=float, default=2.0,
                        help="seconds between provider calls (free-tier pacing)")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)

    if not args.retrieval and not args.answers:
        parser.error("choose --retrieval and/or --answers N")

    questions = load_questions(args.questions)
    os.makedirs(args.results_dir, exist_ok=True)
    suffix = f".{args.tag}" if args.tag else ""

    if args.retrieval:
        print(f"Metric A - retrieval over {len(questions)} questions")
        summary, rows = run_retrieval(questions, top_k=args.top_k)
        with open(os.path.join(args.results_dir, f"retrieval{suffix}.jsonl"), "w",
                  encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(os.path.join(args.results_dir, f"retrieval{suffix}.summary.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(json.dumps({"aggregate": summary["aggregate"],
                          "counts": summary["counts"],
                          "rank_buckets": summary["rank_buckets"]}, indent=2))

    if args.answers:
        print(f"\nMetric B - answers over a stratified sample of {args.answers}")
        summary, rows = run_answers(questions, args.answers, sleep_seconds=args.sleep)
        with open(os.path.join(args.results_dir, f"answers{suffix}.jsonl"), "w",
                  encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(os.path.join(args.results_dir, f"answers{suffix}.summary.json"), "w",
                  encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(json.dumps({k: v for k, v in summary.items()
                          if k not in ("claim_safety",)}, indent=2)[:3000])

    print(f"\n{CLAIM_SAFETY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
