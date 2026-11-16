"""Experiment E — routed retrieval against the Phase 4 baselines.

    .venv/bin/python scripts/evaluation/benchmark_routing.py
    .venv/bin/python scripts/evaluation/benchmark_routing.py --top-k 10 --filters declared

Runs A-E over the same golden benchmark, unfiltered over the whole corpus, and
reports each retriever twice: once overall, and once split by query family. The
split is the point. Phase 4's headline averages hid the only fact that mattered
— that the identifier boost owned cited provisions and BGE-M3 owned everything
else — and a router exists precisely to exploit a difference that an average
erases. Reporting E without the split would repeat the mistake it was built to
correct.

Every retriever here is evaluated on the same corpus build, which is why A-D are
re-run rather than quoted from Phase 4: those numbers were measured on chunking
0.1.0, before the treaty Articles and Schedule entries were repaired. Comparing
E on a corrected corpus against A-D on a broken one would credit routing with
the corpus fix. The Phase 4 figures are kept, unchanged, under
``data/evaluation/results/phase4-chunking-0.1.0/``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.retrieval.base import load_chunks  # noqa: E402
from ai.retrieval.identifiers import BM25IdentifierBoostRetriever  # noqa: E402
from ai.retrieval.router import QUERY_TYPES, RoutedRetriever, classify  # noqa: E402
from evaluation.benchmark_retrieval import (  # noqa: E402
    EXPERIMENTS,
    RESULTS_DIR,
    build_retrievers,
    run_experiment,
    summarise,
)
from evaluation.validate_golden_dataset import GOLDEN_PATH, load_golden  # noqa: E402

#: A-E in the order the report reads them.
LADDER = ("bm25", "dense", "bm25_boost", "hybrid", "hybrid_boost", "routed")
METRICS = ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@5", "ndcg@10")


def router_census(records: Sequence[dict[str, Any]],
                  aliases: dict[str, tuple[str, ...]]) -> dict[str, Any]:
    """How the router classified the benchmark, and how that lines up with the
    families the golden dataset declares.

    The two vocabularies are not the same thing and are not scored against each
    other: `query_family` was assigned when the question was written, by whether
    the question text cites a provision, while the router reads the query at run
    time and also separates CONCEPTUAL from GENERAL. The cross-tab is here so a
    disagreement is visible rather than averaged away — a golden `identifier`
    question the router calls CONCEPTUAL is a parser gap, and it should be
    findable in one place.
    """
    counts: Counter[str] = Counter()
    cross: Counter[tuple[str, str]] = Counter()
    disagreements: list[dict[str, str]] = []
    for record in records:
        decision = classify(record["query"], aliases)
        counts[decision.query_type] += 1
        cross[(record["query_family"], decision.query_type)] += 1
        stated = record["query_family"]
        routed_identifier = decision.query_type == "IDENTIFIER"
        if (stated == "identifier") != routed_identifier:
            disagreements.append({
                "question_id": record["question_id"],
                "query": record["query"],
                "golden_family": stated,
                "router_query_type": decision.query_type,
                "reason": decision.reason,
            })
    return {
        "by_query_type": {t: counts.get(t, 0) for t in QUERY_TYPES},
        "golden_family_vs_router": {f"{a} -> {b}": n for (a, b), n in sorted(cross.items())},
        "identifier_disagreements": disagreements,
    }


def path_census(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Which retrieval path actually answered, fallbacks included."""
    return dict(Counter(row.get("retrieval_path") or "unrouted" for row in rows))


def comparison_table(summaries: Sequence[dict[str, Any]], *,
                     family: str | None = None) -> str:
    head = "| Exp | Retriever | " + " | ".join(
        m.replace("recall@", "Recall@").replace("ndcg@", "nDCG@").replace("mrr", "MRR")
        for m in METRICS) + " |"
    rule = "| --- | --- | " + " | ".join("---:" for _ in METRICS) + " |"
    lines = [head, rule]
    for summary in summaries:
        block = summary["overall"] if family is None else summary["by_family"].get(family)
        if not block:
            continue
        cells = " | ".join(f"{block[m]:.3f}" for m in METRICS)
        lines.append(f"| {summary['experiment']} | {summary['retriever']} | {cells} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", default=GOLDEN_PATH)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--filters", choices=("none", "declared"), default="none",
                        help="Phase 5 §9: the primary routed benchmark is unfiltered")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--retrievers", default=",".join(LADDER))
    args = parser.parse_args(argv)

    records = load_golden(args.golden)
    chunks = load_chunks()
    names = [n.strip() for n in args.retrievers.split(",") if n.strip()]
    retrievers = build_retrievers(names, chunks)

    os.makedirs(args.results_dir, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    paths: dict[str, dict[str, int]] = {}
    suffix = "" if args.filters == "none" else f".{args.filters}"

    for name in names:
        retriever = retrievers[name]
        rows, scores = run_experiment(retriever, records,
                                      top_k=args.top_k, filter_mode=args.filters)
        summary = summarise(retriever, scores, records=records, chunks=chunks,
                            top_k=args.top_k, filter_mode=args.filters)
        summaries.append(summary)
        paths[name] = path_census(rows)
        stem = os.path.join(args.results_dir, f"{name}{suffix}")
        with open(stem + ".jsonl", "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(stem + ".summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"{EXPERIMENTS.get(name, '?'):>7s}  {name:13s} "
              f"recall@1={summary['overall']['recall@1']:.3f} "
              f"mrr={summary['overall']['mrr']:.3f}", flush=True)

    aliases = BM25IdentifierBoostRetriever(retrievers["bm25"]).aliases \
        if "bm25" in retrievers else {}
    if not aliases and any(isinstance(r, RoutedRetriever) for r in retrievers.values()):
        aliases = next(r.aliases for r in retrievers.values()
                       if isinstance(r, RoutedRetriever))
    census = router_census(records, aliases)

    families = sorted({r["query_family"] for r in records})
    print("\n### Overall\n")
    print(comparison_table(summaries))
    for family in families:
        n = sum(1 for r in records if r["query_family"] == family)
        print(f"\n### {family} queries (n={n})\n")
        print(comparison_table(summaries, family=family))

    print("\n### Router\n")
    print(f"query types: {census['by_query_type']}")
    print(f"golden family vs router: {census['golden_family_vs_router']}")
    print(f"retrieval paths taken: {paths.get('routed', {})}")
    if census["identifier_disagreements"]:
        print(f"\n{len(census['identifier_disagreements'])} question(s) where the golden "
              f"family and the router disagree on IDENTIFIER:")
        for item in census["identifier_disagreements"]:
            print(f"  {item['question_id']} golden={item['golden_family']} "
                  f"router={item['router_query_type']}  {item['query']}")

    report = {
        "experiments": {s["retriever"]: s for s in summaries},
        "router": census,
        "retrieval_paths": paths,
        "questions": len(records),
        "questions_human_verified": sum(1 for r in records if r["verified"]),
        "filter_mode": args.filters,
        "top_k": args.top_k,
    }
    out = os.path.join(args.results_dir, f"routing_comparison{suffix}.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    print(f"\nwrote {out}")

    if not any(r["verified"] for r in records):
        print("PROVISIONAL: no question in this run has been verified by a human. "
              "Report these as figures on an unverified benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
