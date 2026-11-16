"""Run the golden benchmark against one or more retrievers.

    .venv/bin/python scripts/evaluation/benchmark_retrieval.py --retriever bm25
    .venv/bin/python scripts/evaluation/benchmark_retrieval.py --retriever dense
    .venv/bin/python scripts/evaluation/benchmark_retrieval.py --retriever hybrid
    .venv/bin/python scripts/evaluation/benchmark_retrieval.py --retriever all

Leakage is prevented structurally rather than by care. A retriever is handed a
:class:`BenchmarkQuery` — question id, query text, filters — and that object is
built before the expected answer is ever read. There is no code path by which an
expected chunk id, document or locator can reach `search()`, which is why
`Filters` has no `document_id` field to begin with.

Filters get the same treatment. The default run is unfiltered, because that is
the only condition under which the numbers describe search over the whole
corpus. `--filters declared` restricts by domain or jurisdiction *only* where
the query text itself names one, and it derives that from the query alone — the
record's `domain` field is never consulted, since that field is part of the
answer key.

Every results file records the corpus version, the chunking version, the
retriever configuration and a timestamp, so a table in a report can always be
traced back to the run that produced it.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.retrieval.base import Chunk, Filters, Retriever, load_chunks  # noqa: E402
from ai.retrieval.bm25 import BM25Retriever  # noqa: E402
from ai.retrieval.dense import corpus_fingerprint, corpus_version  # noqa: E402
from ai.retrieval.hybrid import HybridRRFRetriever  # noqa: E402
from ai.retrieval.identifiers import BM25IdentifierBoostRetriever  # noqa: E402
from ai.retrieval.router import RoutedRetriever  # noqa: E402
from evaluation.metrics import METRIC_KEYS, aggregate, first_hit_rank, score_question  # noqa: E402
from evaluation.validate_golden_dataset import GOLDEN_PATH, load_golden  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation", "results")

#: Experiments A-D are the four Phase 4 asks. D-prime is not one of them: it
#: fuses the *boosted* lexical list instead of plain BM25, and is run because the
#: A-D numbers point straight at it. It is reported as an extra, never in place
#: of D.
EXPERIMENTS = {
    "bm25": "A",
    "dense": "B",
    "bm25_boost": "C",
    "hybrid": "D",
    "hybrid_boost": "D-prime",
    "routed": "E",
}

#: Words that let a query declare its own domain. Used only by
#: `--filters declared`, and read from the query text, never from the answer key.
DOMAIN_WORDS = {
    "patents": ("patent", "patentee", "patentab", "compulsory licence"),
    "trademarks": ("trade mark", "trademark"),
    "geographical_indications": ("geographical indication", "gi act", "gi rules"),
    "designs": ("registered design", "design act", "designs act", "design registration"),
    "copyright": ("copyright",),
    "biodiversity_abs": ("biological resource", "biodiversity", "biological diversity",
                         "benefit sharing", "nagoya", "cbd"),
    "plant_varieties": ("plant variety", "plant varieties", "breeder", "farm-saved",
                        "farmers' rights", "gene fund"),
    "drug_regulation": ("drug", "ayurvedic", "siddha", "unani", "cosmetic", "homoeopathic"),
    "food_safety_ayurveda": ("ayurveda aahara", "fssai"),
    "international_ip": ("trips",),
    "ip_policy": ("ipr policy",),
}
#: Likewise for jurisdiction. The chunk vocabulary is "IN" / "international".
JURISDICTION_WORDS = {
    "IN": ("in india", "indian", "under indian law"),
    "international": ("trips", "wto", "nagoya", "cbd", "convention on biological diversity"),
}


@dataclass(frozen=True)
class BenchmarkQuery:
    """Everything a retriever is allowed to see about a benchmark question."""

    question_id: str
    query: str
    filters: Filters | None


def declared_filters(query: str) -> Filters:
    """Filters the query states for itself. Derived from the query text alone."""
    lowered = query.lower()
    domains = sorted(d for d, words in DOMAIN_WORDS.items()
                     if any(word in lowered for word in words))
    jurisdictions = sorted(j for j, words in JURISDICTION_WORDS.items()
                           if any(word in lowered for word in words))
    return Filters(domain=domains or None, jurisdiction=jurisdictions or None)


def build_retrievers(names: Sequence[str], chunks: Sequence[Chunk]) -> dict[str, Retriever]:
    """Construct only what was asked for; the dense index is expensive to load."""
    built: dict[str, Retriever] = {}
    bm25: BM25Retriever | None = None
    dense = None

    def get_bm25() -> BM25Retriever:
        nonlocal bm25
        if bm25 is None:
            bm25 = BM25Retriever(chunks)
        return bm25

    def get_dense():
        nonlocal dense
        if dense is None:
            from ai.retrieval.dense import DenseRetriever
            dense = DenseRetriever(chunks)
        return dense

    for name in names:
        if name == "bm25":
            built[name] = get_bm25()
        elif name == "bm25_boost":
            built[name] = BM25IdentifierBoostRetriever(get_bm25())
        elif name == "dense":
            built[name] = get_dense()
        elif name == "hybrid":
            built[name] = HybridRRFRetriever([get_bm25(), get_dense()])
        elif name == "hybrid_boost":
            hybrid = HybridRRFRetriever(
                [BM25IdentifierBoostRetriever(get_bm25()), get_dense()])
            hybrid.name = "hybrid_boost"
            built[name] = hybrid
        elif name == "routed":
            boost = BM25IdentifierBoostRetriever(get_bm25())
            built[name] = RoutedRetriever(identifier=boost, dense=get_dense(),
                                          lexical=get_bm25(), aliases=boost.aliases)
        else:
            raise SystemExit(f"unknown retriever: {name}")
    return built


def run_experiment(retriever: Retriever, records: Sequence[dict[str, Any]], *,
                   top_k: int, filter_mode: str) -> tuple[list[dict[str, Any]],
                                                          list[dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    scores: list[dict[str, float]] = []
    for record in records:
        # The query object is built here, from fields a user would have typed.
        # Nothing below this line may read record["expected_*"] before search().
        filters = declared_filters(record["query"]) if filter_mode == "declared" else None
        request = BenchmarkQuery(record["question_id"], record["query"], filters)

        started = time.perf_counter()
        results = retriever.search(request.query, request.filters, top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000.0

        routing = getattr(retriever, "route", None)
        decision = routing(request.query).to_dict() if routing else None

        ranked_chunks = [r.chunk_id for r in results]
        ranked_documents = [r.document_id for r in results]
        question_scores = score_question(
            ranked_chunks, record["expected_chunks"],
            ranked_documents, record["expected_documents"])
        question_scores["query_family"] = record["query_family"]
        scores.append(question_scores)

        rows.append({
            "question_id": record["question_id"],
            "query": record["query"],
            "retriever": retriever.name,
            "filters": (filters.describe() if filters else {}),
            "routing": decision,
            "retrieval_path": (results[0].metadata.get("retrieval_path")
                               if results else None),
            "latency_ms": round(elapsed_ms, 2),
            "expected_chunks": record["expected_chunks"],
            "expected_documents": record["expected_documents"],
            "expected_locators": record["expected_locators"],
            "rank_of_expected": first_hit_rank(ranked_chunks, record["expected_chunks"]),
            "metrics": {k: question_scores[k] for k in METRIC_KEYS},
            "results": [
                {
                    "rank": r.rank,
                    "chunk_id": r.chunk_id,
                    "document_id": r.document_id,
                    "score": round(r.score, 6),
                    "locator": r.locator,
                    "citable": r.citable,
                    "physical_pages": r.metadata["physical_pages"],
                    "temporal_status": r.metadata["temporal_status"],
                    "source_evidence_level": r.metadata["source_evidence_level"],
                    "identifier_boost": r.metadata.get("identifier_boost"),
                    "retrieval_path": r.metadata.get("retrieval_path"),
                }
                for r in results
            ],
        })
    return rows, scores


def summarise(retriever: Retriever, scores: Sequence[dict[str, float]], *,
              records: Sequence[dict[str, Any]], chunks: Sequence[Chunk],
              top_k: int, filter_mode: str) -> dict[str, Any]:
    families = sorted({s["query_family"] for s in scores})
    return {
        "experiment": EXPERIMENTS.get(retriever.name, "?"),
        "retriever": retriever.name,
        "retriever_config": retriever.config(),
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
        "overall": {k: round(v, 4) for k, v in
                    aggregate(scores, METRIC_KEYS + ("document_recall@1",
                                                     "document_recall@5",
                                                     "document_recall@10")).items()},
        "by_family": {
            family: {k: round(v, 4) for k, v in aggregate(
                [s for s in scores if s["query_family"] == family], METRIC_KEYS).items()}
            for family in families
        },
    }


def table(summaries: Sequence[dict[str, Any]]) -> str:
    header = ("| Retriever | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@5 | nDCG@10 |\n"
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines = [header]
    for summary in summaries:
        o = summary["overall"]
        lines.append(
            f"| {summary['retriever']} | {o['recall@1']:.3f} | {o['recall@5']:.3f} | "
            f"{o['recall@10']:.3f} | {o['mrr']:.3f} | {o['ndcg@5']:.3f} | {o['ndcg@10']:.3f} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retriever", default="bm25",
                        help="bm25 | bm25_boost | dense | hybrid | all")
    parser.add_argument("--golden", default=GOLDEN_PATH)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--filters", choices=("none", "declared"), default="none",
                        help="'declared' applies only the filters the query states itself")
    parser.add_argument("--verified-only", action="store_true",
                        help="restrict to human-verified questions (Phase 4 §9)")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    args = parser.parse_args(argv)

    records = load_golden(args.golden)
    if args.verified_only:
        records = [r for r in records if r["verified"]]
        if not records:
            print("No human-verified questions in the golden dataset; nothing to evaluate.\n"
                  "`verified` stays false until a person checks the provision, so this is "
                  "the expected result today. Drop --verified-only to run the provisional "
                  "benchmark, whose numbers must be reported as unverified.")
            return 2

    names = list(EXPERIMENTS) if args.retriever == "all" else [args.retriever]
    chunks = load_chunks()
    retrievers = build_retrievers(names, chunks)

    os.makedirs(args.results_dir, exist_ok=True)
    summaries = []
    for name in names:
        retriever = retrievers[name]
        rows, scores = run_experiment(retriever, records,
                                      top_k=args.top_k, filter_mode=args.filters)
        summary = summarise(retriever, scores, records=records, chunks=chunks,
                            top_k=args.top_k, filter_mode=args.filters)
        suffix = "" if args.filters == "none" else f".{args.filters}"
        stem = os.path.join(args.results_dir, f"{name}{suffix}")
        with open(stem + ".jsonl", "w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(stem + ".summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        summaries.append(summary)
        print(f"{name:11s} recall@1={summary['overall']['recall@1']:.3f} "
              f"recall@5={summary['overall']['recall@5']:.3f} "
              f"recall@10={summary['overall']['recall@10']:.3f} "
              f"mrr={summary['overall']['mrr']:.3f}  → {stem}.jsonl")

    print()
    print(table(summaries))
    print(f"\nquestions={len(records)}  human-verified="
          f"{sum(1 for r in records if r['verified'])}  filters={args.filters}")
    if not any(r["verified"] for r in records):
        print("PROVISIONAL: no question in this run has been verified by a human. "
              "Report these as figures on an unverified benchmark.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
