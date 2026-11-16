"""Phase 6 §10 — does a reranker keep the *cited* provision above its neighbours?

    .venv/bin/python scripts/evaluation/identifier_probe.py
    .venv/bin/python scripts/evaluation/identifier_probe.py --reranker cross_encoder \
        --cross-encoder bge-m3

The worry §10 states is specific and worth taking literally: asked "what does
Section 3(p) say", a cross-encoder may rank a *discussion* of section 3 above
section 3(p) itself, because a passage arguing about section 3 at length looks
more like the query than the terse provision does. Recall@1 over 18 identifier
questions cannot see that — it only says whether the labelled chunk won.

So this probe ignores the golden labels for its primary measurement and
partitions the candidate pool by the chunk's own locator metadata:

``exact``       the chunk *is* the provision named, including its sub-unit
``parent``      the query named a sub-unit; this chunk is the whole provision
``sibling``     the same provision, but a different sub-unit than the one named
``referenced``  a different provision whose text cites the named one
``unrelated``   everything else

Then it reports where each class lands. The number that answers §10 is
`exact_outranked_by`: how often a chunk from a weaker class was placed above
every exact one. That is a property of the ranking, not of the answer key, so it
stays meaningful while the benchmark is unverified.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.reranking.base import IdentityReranker  # noqa: E402
from ai.reranking.features import FeatureReranker, locator_scores  # noqa: E402
from ai.retrieval.base import load_chunks  # noqa: E402
from ai.retrieval.bm25 import BM25Retriever  # noqa: E402
from ai.retrieval.candidates import CandidateGenerator  # noqa: E402
from ai.retrieval.identifiers import (  # noqa: E402
    BM25IdentifierBoostRetriever,
    parse_identifiers,
)
from evaluation.benchmark_reranking import RESULTS_DIR  # noqa: E402
from evaluation.validate_golden_dataset import GOLDEN_PATH, load_golden  # noqa: E402

CLASSES = ("exact", "parent", "sibling", "referenced", "unrelated")

#: A class outranking `exact` is a §10 failure only if it is a *weaker* claim on
#: the citation. `unrelated` is included because a wholly unrelated chunk beating
#: the cited provision is the worst of the four.
WEAKER_THAN_EXACT = ("parent", "sibling", "referenced", "unrelated")


def classify_candidate(refs, candidate) -> str:
    locator = candidate.metadata.get("locator_fields") or {}
    exact, referenced = locator_scores(refs, locator, candidate.text)
    if exact >= 1.0:
        return "exact"
    if exact == 0.6:
        return "parent"
    if exact == 0.3:
        return "sibling"
    return "referenced" if referenced else "unrelated"


def probe(reranker, generator, records: Sequence[dict[str, Any]], *,
          top_k: int) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for record in records:
        refs = parse_identifiers(record["query"])
        if not refs:
            continue
        pool = generator.generate(record["query"])
        classes = {c.chunk_id: classify_candidate(refs, c) for c in pool.candidates}
        ranked = reranker.rerank(record["query"], pool.candidates, top_k=len(pool))

        positions: dict[str, list[int]] = {name: [] for name in CLASSES}
        for result in ranked:
            positions[classes[result.chunk_id]].append(result.rerank_rank)

        best_exact = min(positions["exact"], default=None)
        outranked = [
            {"class": name, "rank": rank}
            for name in WEAKER_THAN_EXACT
            for rank in positions[name]
            if best_exact is not None and rank < best_exact
        ]
        rows.append({
            "question_id": record["question_id"],
            "query": record["query"],
            "cited": [f"{r.kind} {r.number}"
                      + "".join(f"({s})" for s in r.subdivisions) for r in refs],
            "pool_counts": {name: len(positions[name]) for name in CLASSES},
            "best_rank": {name: min(positions[name], default=None) for name in CLASSES},
            "rank_1_class": classes[ranked[0].chunk_id] if ranked else None,
            "exact_present": best_exact is not None,
            "exact_best_rank": best_exact,
            "exact_at_rank_1": best_exact == 1,
            "exact_in_top_k": best_exact is not None and best_exact <= top_k,
            "exact_outranked_by": outranked,
        })

    with_exact = [r for r in rows if r["exact_present"]]
    outranked_rows = [r for r in with_exact if r["exact_outranked_by"]]
    return {
        "reranker": reranker.name,
        "reranker_config": reranker.config(),
        "questions": len(rows),
        "questions_with_an_exact_chunk_in_pool": len(with_exact),
        "exact_at_rank_1": sum(1 for r in with_exact if r["exact_at_rank_1"]),
        "exact_at_rank_1_rate": round(
            sum(1 for r in with_exact if r["exact_at_rank_1"]) / len(with_exact), 4)
        if with_exact else 0.0,
        "exact_in_top_k": sum(1 for r in with_exact if r["exact_in_top_k"]),
        "median_exact_rank": (statistics.median([r["exact_best_rank"] for r in with_exact])
                              if with_exact else None),
        "questions_where_a_weaker_class_outranked_exact": len(outranked_rows),
        "outranking_classes": dict(Counter(
            item["class"] for r in outranked_rows for item in r["exact_outranked_by"])),
        "rank_1_class_census": dict(Counter(
            r["rank_1_class"] for r in rows if r["rank_1_class"])),
        "per_question": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reranker", default="all",
                        help="identity | features | cross_encoder | all")
    parser.add_argument("--cross-encoder", default="bge-m3")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--golden", default=GOLDEN_PATH)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    args = parser.parse_args(argv)

    records = load_golden(args.golden)
    chunks = load_chunks()
    bm25 = BM25Retriever(chunks)
    boost = BM25IdentifierBoostRetriever(bm25)
    from ai.retrieval.dense import DenseRetriever

    dense = DenseRetriever(chunks)
    generator = CandidateGenerator(dense=dense, lexical=boost, aliases=boost.aliases)

    names = (["identity", "features", "cross_encoder"] if args.reranker == "all"
             else [args.reranker])
    reports = []
    for name in names:
        if name == "identity":
            reranker = IdentityReranker()
        elif name == "features":
            reranker = FeatureReranker(idf=bm25.idf, aliases=boost.aliases)
        else:
            from ai.reranking.cross_encoder import CrossEncoderReranker

            reranker = CrossEncoderReranker(args.cross_encoder)
            reranker.warm()
        reports.append(probe(reranker, generator, records, top_k=args.top_k))

    os.makedirs(args.results_dir, exist_ok=True)
    path = os.path.join(args.results_dir, "identifier_probe.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"classes": list(CLASSES), "reports": reports}, fh, indent=2)

    print("| Reranker | queries | exact@1 | exact in top-k | median exact rank | "
          "queries where a weaker class outranked exact |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for report in reports:
        print(f"| {report['reranker']} | {report['questions_with_an_exact_chunk_in_pool']} | "
              f"{report['exact_at_rank_1']} | {report['exact_in_top_k']} | "
              f"{report['median_exact_rank']} | "
              f"{report['questions_where_a_weaker_class_outranked_exact']} |")
    print(f"\n→ {path}")
    print("PROVISIONAL: the partition is read from locator metadata, not from the "
          "golden labels, but the queries are still the unverified benchmark's.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
