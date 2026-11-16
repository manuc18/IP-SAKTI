"""Phase 6 — benchmark a reranking layer over a high-recall candidate pool.

    .venv/bin/python scripts/evaluation/benchmark_reranking.py \
      --candidate-mode expanded --reranker identity
    .venv/bin/python scripts/evaluation/benchmark_reranking.py \
      --candidate-mode expanded --reranker features
    .venv/bin/python scripts/evaluation/benchmark_reranking.py \
      --candidate-mode expanded --reranker cross_encoder --cross-encoder bge-m3

The question this answers is narrower than "is retrieval good". Recall@10 is
already 0.964 routed and 0.982 fused; what is not settled is whether the right
provision reaches **rank 1**. So every experiment reports two things that are
usually collapsed into one:

*Candidate recall* — was the gold evidence in the pool at all? A reranker cannot
promote what it was never shown, and a headline metric that mixes the two blames
the model for the retriever's misses.

*Conditional MRR* — mean reciprocal rank over exactly the queries whose gold
evidence **was** in the pool. That is the reranking quality figure. The
unconditional MRR is reported alongside because it is what a user experiences,
but it moves when candidate generation changes and says nothing about the model.

Leakage is prevented structurally, as in Phase 4. A :class:`BenchmarkQuery` is
built from fields a user could have typed, before the expected answer is read;
candidate generation and reranking take that object and nothing else. On top of
the structure, :func:`ai.reranking.format.leakage_report` audits the strings the
model actually saw, per query, and the audit result is stored in every summary
rather than asserted in prose.

Rerankers share one set of candidate pools. That is not only cheaper — it is the
only way the comparison is about reranking, since two runs that each regenerated
their own pool would differ by whatever the pool differed by. Candidate latency
is therefore measured once, on a cold cache, and reranking latency once per
reranker.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.reranking.base import IdentityReranker, Reranker  # noqa: E402
from ai.reranking.features import AuthorityPolicyReranker, FeatureReranker  # noqa: E402
from ai.reranking.format import format_candidate, leakage_report  # noqa: E402
from ai.retrieval.base import load_chunks  # noqa: E402
from ai.retrieval.bm25 import BM25Retriever  # noqa: E402
from ai.retrieval.candidates import (  # noqa: E402
    DEFAULT_DENSE_DEPTH,
    DEFAULT_LEXICAL_DEPTH,
    CandidateGenerator,
    CandidateSet,
)
from ai.retrieval.dense import corpus_fingerprint, corpus_version  # noqa: E402
from ai.retrieval.identifiers import BM25IdentifierBoostRetriever  # noqa: E402
from ai.retrieval.router import classify  # noqa: E402
from evaluation.benchmark_retrieval import BenchmarkQuery, declared_filters  # noqa: E402
from evaluation.metrics import (  # noqa: E402
    METRIC_KEYS,
    aggregate,
    first_hit_rank,
    recall_at_k,
    score_question,
)
from evaluation.rerank_errors import classify_failure, summarise_failures  # noqa: E402
from evaluation.validate_golden_dataset import GOLDEN_PATH, load_golden  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation", "results", "reranking")

#: Phase 6 §2. The depths candidate recall is reported at.
CANDIDATE_DEPTHS = (10, 20, 30, 50)

#: Experiment labels from Phase 6 §12. `F2` is parameterised by model, so the
#: stored label carries the model key with it.
EXPERIMENT_LABELS = {
    "identity": "F0",
    "features": "F1",
    "cross_encoder": "F2",
    "features+authority": "F3-features",
    "cross_encoder+authority": "F3",
}

RERANKERS = tuple(EXPERIMENT_LABELS)


def build_reranker(name: str, *, bm25: BM25Retriever, aliases, args) -> Reranker:
    """Construct one reranker. Every knob comes from the CLI, never from data."""
    from ai.reranking.cross_encoder import CrossEncoderReranker

    def features() -> FeatureReranker:
        kwargs = {}
        if args.heading_bonus is not None:
            kwargs["heading_bonus"] = args.heading_bonus
        if args.structural_penalty is not None:
            kwargs["structural_penalty"] = args.structural_penalty
        return FeatureReranker(idf=bm25.idf, aliases=aliases,
                               normalization=args.normalization, **kwargs)

    def cross_encoder() -> CrossEncoderReranker:
        return CrossEncoderReranker(args.cross_encoder,
                                    batch_size=args.batch_size,
                                    max_length=args.max_length)

    if name == "identity":
        return IdentityReranker()
    if name == "features":
        return features()
    if name == "cross_encoder":
        return cross_encoder()
    if name == "features+authority":
        return AuthorityPolicyReranker(features(), weight=args.authority_lambda)
    if name == "cross_encoder+authority":
        return AuthorityPolicyReranker(cross_encoder(), weight=args.authority_lambda)
    raise SystemExit(f"unknown reranker: {name}")


def generate_pools(generator: CandidateGenerator, records: Sequence[dict[str, Any]], *,
                   filter_mode: str) -> list[tuple[BenchmarkQuery, CandidateSet]]:
    """One candidate pool per question, built before any answer is read."""
    pools: list[tuple[BenchmarkQuery, CandidateSet]] = []
    for record in records:
        # Nothing below this line may read record["expected_*"] before generate().
        filters = declared_filters(record["query"]) if filter_mode == "declared" else None
        request = BenchmarkQuery(record["question_id"], record["query"], filters)
        pools.append((request, generator.generate(request.query, request.filters)))
    return pools


def candidate_metrics(pool: CandidateSet, expected: Sequence[str]) -> dict[str, Any]:
    """Was the gold evidence available to the reranker, and how deep?"""
    ids = pool.ids()
    rank = first_hit_rank(ids, expected)
    metrics = {f"candidate_recall@{depth}": recall_at_k(ids, expected, depth)
               for depth in CANDIDATE_DEPTHS}
    metrics["candidate_recall@pool"] = recall_at_k(ids, expected, len(ids))
    metrics["gold_in_pool"] = rank is not None
    metrics["gold_pool_rank"] = rank
    metrics["pool_size"] = len(ids)
    return metrics


def percentiles(values: Sequence[float]) -> dict[str, float]:
    """Median, mean and p95 — Phase 6 §19 asks for all three."""
    if not values:
        return {"median_ms": 0.0, "mean_ms": 0.0, "p95_ms": 0.0, "n": 0}
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
    return {
        "median_ms": round(statistics.median(ordered), 2),
        "mean_ms": round(statistics.fmean(ordered), 2),
        "p95_ms": round(ordered[index], 2),
        "n": len(ordered),
    }


def run(reranker: Reranker, pools: Sequence[tuple[BenchmarkQuery, CandidateSet]],
        records: Sequence[dict[str, Any]], *, top_k: int,
        text_budget: int | None) -> tuple[list[dict[str, Any]], list[dict[str, Any]],
                                          list[dict[str, Any]], list[float]]:
    by_id = {record["question_id"]: record for record in records}
    rows: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    latencies: list[float] = []

    for request, pool in pools:
        started = time.perf_counter()
        reranked = reranker.rerank(request.query, pool.candidates, top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        latencies.append(elapsed_ms)

        record = by_id[request.question_id]
        rendered = [format_candidate(c, text_budget=text_budget) for c in pool.candidates]
        audit = leakage_report(
            request.query, pool.candidates, rendered=rendered,
            forbidden_strings={
                "expected_chunks": record["expected_chunks"],
                "expected_documents": record["expected_documents"],
                "notes": [record.get("notes") or ""],
            })
        audit["question_id"] = request.question_id
        audits.append(audit)

        ranked_chunks = [r.chunk_id for r in reranked]
        ranked_documents = [r.document_id for r in reranked]
        question_scores = score_question(
            ranked_chunks, record["expected_chunks"],
            ranked_documents, record["expected_documents"])
        question_scores.update(candidate_metrics(pool, record["expected_chunks"]))
        question_scores["query_family"] = record["query_family"]
        question_scores["router_family"] = classify(request.query).query_type
        question_scores["question_id"] = request.question_id
        scores.append(question_scores)

        rows.append({
            "question_id": request.question_id,
            "query": request.query,
            "reranker": reranker.name,
            "candidate_mode": pool.mode,
            "filters": (request.filters.describe() if request.filters else {}),
            "candidates": pool.summary(),
            "candidate_latency_ms": round(pool.latency_ms, 2),
            "rerank_latency_ms": round(elapsed_ms, 2),
            "total_latency_ms": round(pool.latency_ms + elapsed_ms, 2),
            "expected_chunks": record["expected_chunks"],
            "expected_documents": record["expected_documents"],
            "expected_locators": record["expected_locators"],
            "query_family": record["query_family"],
            "router_family": question_scores["router_family"],
            "gold_in_pool": question_scores["gold_in_pool"],
            "gold_pool_rank": question_scores["gold_pool_rank"],
            "rank_of_expected": first_hit_rank(ranked_chunks, record["expected_chunks"]),
            "metrics": {k: question_scores[k] for k in METRIC_KEYS},
            "candidate_metrics": {f"candidate_recall@{d}": question_scores[
                f"candidate_recall@{d}"] for d in CANDIDATE_DEPTHS},
            "leakage_clean": audit["clean"],
            "results": [
                {
                    "rerank_rank": r.rerank_rank,
                    "rerank_score": round(float(r.rerank_score), 6),
                    "candidate_order": r.candidate.order,
                    "chunk_id": r.chunk_id,
                    "document_id": r.document_id,
                    "locator": r.locator,
                    "citable": r.citable,
                    "physical_pages": r.metadata.get("physical_pages"),
                    "temporal_status": r.metadata.get("temporal_status"),
                    "source_evidence_level": r.metadata.get("source_evidence_level"),
                    "dense_rank": r.candidate.dense_rank,
                    "lexical_rank": r.candidate.lexical_rank,
                    "identifier_boost": r.candidate.identifier_boost,
                    "sources": list(r.candidate.sources),
                    "components": {k: round(float(v), 6)
                                   for k, v in r.components.items()},
                }
                for r in reranked
            ],
        })
    return rows, scores, audits, latencies


CANDIDATE_KEYS = tuple(f"candidate_recall@{d}" for d in CANDIDATE_DEPTHS) + (
    "candidate_recall@pool",)


def conditional_mrr(scores: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """MRR over the queries whose gold evidence actually reached the pool.

    Phase 6 §13 asks for this separately, and it is the figure that isolates
    reranking quality: a query whose evidence was never retrieved contributes 0
    to the unconditional MRR no matter how good the reranker is, so mixing the
    two measures candidate generation and calls it model quality.
    """
    eligible = [s for s in scores if s["gold_in_pool"]]
    if not eligible:
        return {"questions": 0, "mrr": 0.0, "recall@1": 0.0, "recall@5": 0.0}
    return {
        "questions": len(eligible),
        "mrr": round(sum(s["mrr"] for s in eligible) / len(eligible), 4),
        "recall@1": round(sum(s["recall@1"] for s in eligible) / len(eligible), 4),
        "recall@5": round(sum(s["recall@5"] for s in eligible) / len(eligible), 4),
    }


def summarise(reranker: Reranker, generator: CandidateGenerator,
              scores: Sequence[dict[str, Any]], audits: Sequence[dict[str, Any]],
              latencies: Sequence[float],
              pools: Sequence[tuple[BenchmarkQuery, CandidateSet]], *,
              records: Sequence[dict[str, Any]], chunks, top_k: int,
              filter_mode: str, args) -> dict[str, Any]:
    families = sorted({s["query_family"] for s in scores})
    router_families = sorted({s["router_family"] for s in scores})
    candidate_ms = [p.latency_ms for _, p in pools]
    total_ms = [c + r for c, r in zip(candidate_ms, latencies)]
    leaks = [a for a in audits if not a["clean"]]

    key = reranker.name.split(":")[0].replace("+authority", "") + (
        "+authority" if "+authority" in reranker.name else "")
    return {
        "experiment": EXPERIMENT_LABELS.get(key, "?"),
        "reranker": reranker.name,
        "reranker_config": reranker.config(),
        "candidate_generator_config": generator.config(),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus_version": corpus_version(chunks),
        "chunking_version": corpus_fingerprint(chunks),
        "chunker_version": chunks[0].raw.get("chunking_version") if chunks else None,
        "chunks_indexed": len(chunks),
        "questions": len(scores),
        "questions_human_verified": sum(1 for r in records if r["verified"]),
        "top_k": top_k,
        "filter_mode": filter_mode,
        "python": platform.python_version(),
        "candidate_recall": {k: round(v, 4) for k, v in
                             aggregate(scores, CANDIDATE_KEYS).items()},
        "pool_size": {
            "mean": round(statistics.fmean([len(p) for _, p in pools]), 1) if pools else 0,
            "min": min((len(p) for _, p in pools), default=0),
            "max": max((len(p) for _, p in pools), default=0),
        },
        "overall": {k: round(v, 4) for k, v in aggregate(
            scores, METRIC_KEYS + ("document_recall@1", "document_recall@5",
                                   "document_recall@10")).items()},
        "conditional": conditional_mrr(scores),
        "by_family": {
            family: {
                **{k: round(v, 4) for k, v in aggregate(
                    [s for s in scores if s["query_family"] == family],
                    METRIC_KEYS + CANDIDATE_KEYS).items()},
                "questions": sum(1 for s in scores if s["query_family"] == family),
                "conditional": conditional_mrr(
                    [s for s in scores if s["query_family"] == family]),
            }
            for family in families
        },
        "by_router_family": {
            family: {
                **{k: round(v, 4) for k, v in aggregate(
                    [s for s in scores if s["router_family"] == family],
                    METRIC_KEYS + CANDIDATE_KEYS).items()},
                "questions": sum(1 for s in scores if s["router_family"] == family),
                "conditional": conditional_mrr(
                    [s for s in scores if s["router_family"] == family]),
            }
            for family in router_families
        },
        "latency": {
            "candidate_generation": percentiles(candidate_ms),
            "reranking": percentiles(list(latencies)),
            "total_pipeline": percentiles(total_ms),
            "measurement_note":
                "one process, fresh retriever and model instances, warmed with a "
                "throwaway query before the loop; each benchmark question is seen "
                "once, so no dense query-embedding cache hit is timed. Candidate "
                "pools are shared across rerankers in a run, so candidate "
                "latency is measured once and repeated in each summary.",
        },
        "leakage_audit": {
            "questions_audited": len(audits),
            "clean": not leaks,
            "failing_questions": [a["question_id"] for a in leaks],
            "findings": [f for a in leaks for f in a["findings"]][:50],
            "checks": ["chunk_id never rendered", "forbidden metadata never rendered",
                       "no label outside Document/Locator/Heading/Text",
                       "expected chunk ids absent from query and candidates",
                       "expected document ids absent from query and candidates",
                       "golden `notes` absent from query and candidates"],
        },
        "cli": {
            "candidate_mode": args.candidate_mode,
            "candidate_order": args.candidate_order,
            "dense_depth": args.dense_depth,
            "lexical_depth": args.lexical_depth,
            "normalization": args.normalization,
            "cross_encoder": args.cross_encoder,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "authority_lambda": args.authority_lambda,
            "heading_bonus": args.heading_bonus,
            "structural_penalty": args.structural_penalty,
            "text_budget": args.text_budget,
        },
    }


def table(summaries: Sequence[dict[str, Any]]) -> str:
    header = ("| Exp | Reranker | R@1 | R@5 | R@10 | MRR | nDCG@5 | nDCG@10 | "
              "cond. MRR |\n| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines = [header]
    for summary in summaries:
        o, c = summary["overall"], summary["conditional"]
        lines.append(
            f"| {summary['experiment']} | {summary['reranker']} | {o['recall@1']:.3f} | "
            f"{o['recall@5']:.3f} | {o['recall@10']:.3f} | {o['mrr']:.3f} | "
            f"{o['ndcg@5']:.3f} | {o['ndcg@10']:.3f} | {c['mrr']:.3f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidate-mode", choices=("expanded", "routed"),
                        default="expanded")
    parser.add_argument("--candidate-order", choices=("interleaved", "rrf"),
                        default="interleaved")
    parser.add_argument("--reranker", default="identity",
                        help="comma-separated: " + " | ".join(RERANKERS) + " | all")
    parser.add_argument("--cross-encoder", default="bge-m3",
                        help="minilm | bge-base | bge-m3")
    parser.add_argument("--dense-depth", type=int, default=DEFAULT_DENSE_DEPTH)
    parser.add_argument("--lexical-depth", type=int, default=DEFAULT_LEXICAL_DEPTH)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--normalization", choices=("min_max", "rank_percentile"),
                        default="min_max",
                        help="feature normalization; min-max is the selected "
                             "default (Phase 6 section 7), rank percentile is "
                             "kept as the reported sensitivity check")
    parser.add_argument("--authority-lambda", type=float, default=0.05)
    parser.add_argument("--heading-bonus", type=float, default=None,
                        help="override FeatureReranker.heading_bonus; "
                             "0.0 disables the heading feature entirely")
    parser.add_argument("--structural-penalty", type=float, default=None,
                        help="override FeatureReranker.structural_penalty")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--text-budget", type=int, default=None,
                        help="characters of chunk text shown to the reranker "
                             "(default: no truncation; the model's tokenizer "
                             "truncates and the count is reported)")
    parser.add_argument("--filters", choices=("none", "declared"), default="none")
    parser.add_argument("--golden", default=GOLDEN_PATH)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--tag", default="", help="suffix for the output filenames")
    parser.add_argument("--verified-only", action="store_true")
    args = parser.parse_args(argv)

    records = load_golden(args.golden)
    if args.verified_only:
        records = [r for r in records if r["verified"]]
        if not records:
            print("No human-verified questions in the golden dataset; nothing to "
                  "evaluate. `verified` stays false until a person checks the "
                  "provision, so this is the expected result today.")
            return 2

    names = list(RERANKERS) if args.reranker == "all" else [
        n.strip() for n in args.reranker.split(",") if n.strip()]
    for name in names:
        if name not in RERANKERS:
            raise SystemExit(f"unknown reranker: {name}; choose from {list(RERANKERS)}")

    chunks = load_chunks()
    bm25 = BM25Retriever(chunks)
    boost = BM25IdentifierBoostRetriever(bm25)
    from ai.retrieval.dense import DenseRetriever

    dense = DenseRetriever(chunks)
    # Warm both engines on a query that is not in the benchmark, so the first
    # measured question does not pay for lazy model loading.
    dense.search("warm up the encoder", None, 1)
    boost.search("warm up the lexical index", None, 1)

    generator = CandidateGenerator(
        dense=dense, lexical=boost, mode=args.candidate_mode,
        order=args.candidate_order, dense_depth=args.dense_depth,
        lexical_depth=args.lexical_depth, aliases=boost.aliases)

    pools = generate_pools(generator, records, filter_mode=args.filters)

    os.makedirs(args.results_dir, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for name in names:
        reranker = build_reranker(name, bm25=bm25, aliases=boost.aliases, args=args)
        warm = getattr(reranker, "warm", None) or getattr(
            getattr(reranker, "base", None), "warm", None)
        if warm:
            warm()
        rows, scores, audits, latencies = run(
            reranker, pools, records, top_k=args.top_k, text_budget=args.text_budget)
        summary = summarise(reranker, generator, scores, audits, latencies, pools,
                            records=records, chunks=chunks, top_k=args.top_k,
                            filter_mode=args.filters, args=args)
        failures = [classify_failure(row, scores_by_id={s["question_id"]: s
                                                        for s in scores}, at=5)
                    for row in rows]
        failures = [f for f in failures if f]
        summary["failure_analysis"] = summarise_failures(failures, len(rows))

        stem_parts = [name.replace("+", "-")]
        if name.startswith("cross_encoder"):
            stem_parts.append(args.cross_encoder)
        if args.candidate_mode != "expanded":
            stem_parts.append(args.candidate_mode)
        if args.tag:
            stem_parts.append(args.tag)
        stem = os.path.join(args.results_dir, ".".join(stem_parts))

        with open(stem + ".jsonl", "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(stem + ".summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        with open(stem + ".errors.json", "w", encoding="utf-8") as fh:
            json.dump({"reranker": reranker.name,
                       "at": 5,
                       "summary": summary["failure_analysis"],
                       "failures": failures}, fh, indent=2)
        summaries.append(summary)

        o, c = summary["overall"], summary["conditional"]
        print(f"{reranker.name:28s} r@1={o['recall@1']:.3f} r@5={o['recall@5']:.3f} "
              f"mrr={o['mrr']:.3f} cond.mrr={c['mrr']:.3f}  → {stem}.jsonl")

    print()
    print(f"candidate pool ({args.candidate_mode}/{args.candidate_order}): "
          + "  ".join(f"recall@{d}={summaries[0]['candidate_recall'][f'candidate_recall@{d}']:.3f}"
                      for d in CANDIDATE_DEPTHS))
    print()
    print(table(summaries))
    if not all(s["leakage_audit"]["clean"] for s in summaries):
        print("\nLEAKAGE AUDIT FAILED — see the summary files.")
        return 1
    print("\nleakage audit: clean on every question and every experiment")
    if not any(r["verified"] for r in records):
        print("PROVISIONAL: no question in this run has been verified by a human. "
              "Report these as figures on an unverified benchmark. They are not "
              "a measurement of legal accuracy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
