"""Phase 6 §19 — time each stage of the pipeline, on its own, with nothing else running.

    .venv/bin/python scripts/evaluation/measure_latency.py
    .venv/bin/python scripts/evaluation/measure_latency.py --queries 30 --stage cross_encoder

The per-row `latency_ms` in a benchmark results file is not a clean measurement
and must not be quoted as one. Two hazards, both observed in this project:

* **Concurrency.** Phase 6's experiments were run as background jobs and some of
  them overlapped. Two processes contending for the same Metal device inflate
  each other's numbers — the stored p95 for `bge-reranker-v2-m3` is roughly
  seven times its median, which is contention, not tail behaviour.
* **Caching.** Phase 5 recorded the same trap one level down: `DenseRetriever`
  memoises query embeddings, so a retriever running after another retriever that
  shares its encoder times cache hits rather than work.

This script exists so there is one measurement that is free of both. It builds
every component fresh, warms each on throwaway queries that are not in the
benchmark, runs one stage at a time, and is meant to be run when the machine is
otherwise idle. Everything it reports is median, mean and p95 over the same set
of queries.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.reranking.base import IdentityReranker  # noqa: E402
from ai.reranking.features import FeatureReranker  # noqa: E402
from ai.retrieval.base import load_chunks  # noqa: E402
from ai.retrieval.bm25 import BM25Retriever  # noqa: E402
from ai.retrieval.candidates import CandidateGenerator  # noqa: E402
from ai.retrieval.identifiers import BM25IdentifierBoostRetriever  # noqa: E402
from evaluation.benchmark_reranking import RESULTS_DIR, percentiles  # noqa: E402
from evaluation.validate_golden_dataset import GOLDEN_PATH, load_golden  # noqa: E402

#: Warm-up queries, deliberately not from the benchmark, so a warm-up can never
#: be a measured query's cache entry.
WARMUP = ("warm up the encoder", "warm up the lexical index",
          "warm up the reranking model")

STAGES = ("candidates", "identity", "features", "cross_encoder")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries", type=int, default=30,
                        help="how many benchmark queries to time (the first N)")
    parser.add_argument("--stage", action="append", default=[],
                        help=f"restrict to these stages: {', '.join(STAGES)}")
    parser.add_argument("--cross-encoder", action="append", default=[],
                        help="cross-encoder model keys to time (default: all three)")
    parser.add_argument("--dense-depth", type=int, default=50)
    parser.add_argument("--lexical-depth", type=int, default=50)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--golden", default=GOLDEN_PATH)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    args = parser.parse_args(argv)

    stages = args.stage or list(STAGES)
    models = args.cross_encoder or ["minilm", "bge-base", "bge-m3"]
    queries = [r["query"] for r in load_golden(args.golden)][:args.queries]

    chunks = load_chunks()
    bm25 = BM25Retriever(chunks)
    boost = BM25IdentifierBoostRetriever(bm25)
    from ai.retrieval.dense import DenseRetriever

    dense = DenseRetriever(chunks)
    for query in WARMUP:
        dense.search(query, None, 1)
        boost.search(query, None, 1)

    generator = CandidateGenerator(dense=dense, lexical=boost,
                                   dense_depth=args.dense_depth,
                                   lexical_depth=args.lexical_depth,
                                   aliases=boost.aliases)

    report: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "queries_timed": len(queries),
        "dense_depth": args.dense_depth,
        "lexical_depth": args.lexical_depth,
        "top_k": args.top_k,
        "conditions": "fresh instances, warmed on queries outside the benchmark, "
                      "one stage at a time, machine otherwise idle",
        "stages": {},
    }

    pools = []
    if "candidates" in stages or any(s in stages for s in ("identity", "features",
                                                           "cross_encoder")):
        timings = []
        for query in queries:
            started = time.perf_counter()
            pools.append(generator.generate(query))
            timings.append((time.perf_counter() - started) * 1000.0)
        report["stages"]["candidate_generation"] = percentiles(timings)
        report["stages"]["candidate_generation"]["mean_pool_size"] = round(
            sum(len(p) for p in pools) / len(pools), 1) if pools else 0

    def time_reranker(label, reranker):
        timings = []
        for query, pool in zip(queries, pools):
            started = time.perf_counter()
            reranker.rerank(query, pool.candidates, args.top_k)
            timings.append((time.perf_counter() - started) * 1000.0)
        stats = percentiles(timings)
        stats["reranker"] = reranker.name
        report["stages"][label] = stats
        print(f"{label:28s} median={stats['median_ms']:9.1f} ms  "
              f"mean={stats['mean_ms']:9.1f}  p95={stats['p95_ms']:9.1f}")

    if "identity" in stages:
        time_reranker("rerank_identity", IdentityReranker())
    if "features" in stages:
        time_reranker("rerank_features",
                      FeatureReranker(idf=bm25.idf, aliases=boost.aliases))
    if "cross_encoder" in stages:
        from ai.reranking.cross_encoder import CrossEncoderReranker, resource_report

        for key in models:
            reranker = CrossEncoderReranker(key)
            for query in WARMUP:
                reranker.rerank(query, pools[0].candidates[:8], 5)
            time_reranker(f"rerank_cross_encoder_{key}", reranker)
            report["stages"][f"rerank_cross_encoder_{key}"]["resources"] = (
                resource_report(reranker))
            report["stages"][f"rerank_cross_encoder_{key}"]["truncation"] = (
                reranker.truncation.to_dict())

    candidate = report["stages"].get("candidate_generation", {})
    for label, stats in report["stages"].items():
        if label.startswith("rerank_"):
            stats["total_pipeline_median_ms"] = round(
                candidate.get("median_ms", 0.0) + stats["median_ms"], 2)

    os.makedirs(args.results_dir, exist_ok=True)
    path = os.path.join(args.results_dir, "latency.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\ncandidate generation: median={candidate.get('median_ms')} ms  "
          f"p95={candidate.get('p95_ms')} ms  "
          f"(pool {candidate.get('mean_pool_size')} candidates)")
    print(f"→ {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
