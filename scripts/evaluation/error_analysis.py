"""Why the benchmark's failures failed — with the evidence, or not at all.

    .venv/bin/python scripts/evaluation/error_analysis.py --retriever bm25
    .venv/bin/python scripts/evaluation/error_analysis.py --retriever all --at 5

Each failure is classified only when a rule can point at something in the run
that establishes the category: a retrieved chunk carrying the expected chunk's
own locator is evidence of a chunking split; an expected chunk excluded by the
run's filters is evidence of a filtering problem. Where no rule fires, the
failure is reported as `unclassified` with its top-10 attached, because a
plausible-sounding label on a failure nobody examined is worse than no label —
it ends up in a report as if it were a finding.

Two observations are recorded as *flags* rather than causes: that the expected
evidence comes from a source Phase 3 found stale, and that a non-citable chunk
outranked it. Both matter, neither explains a ranking on its own.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.retrieval.base import Filters, load_chunks  # noqa: E402
from ai.retrieval.identifiers import LOCATOR_FIELD, parse_identifiers  # noqa: E402
from evaluation.benchmark_retrieval import RESULTS_DIR  # noqa: E402

#: Locator fields that identify a provision rather than a structural container.
#: Two chunks agreeing on `chapter` say nothing; agreeing on `section` says they
#: are two pieces of the same provision.
PROVISION_FIELDS = ("section", "rule", "article", "regulation", "paragraph",
                    "form", "schedule", "annexure")

CATEGORIES = (
    "metadata-filtering problem",
    "corpus incompleteness",
    "chunking problem",
    "duplicate evidence",
    "exact-term failure",
    "document mismatch",
    "semantic mismatch",
    "unclassified",
)

#: Phase 5 §11. Routed retrieval fails in ways the Phase 4 vocabulary cannot
#: name, because a routed failure has one more moving part: the choice of
#: engine. "semantic mismatch" is the wrong label for a query that BGE-M3 would
#: have answered and the router sent to BM25 instead.
ROUTED_CATEGORIES = (
    "corpus evidence is missing",
    "identifier parser failed",
    "identifier metadata is wrong",
    "router chose wrong family",
    "chunking failed",
    "corpus is stale",
    "dense retrieval failed",
    "BM25 failed",
    "unclassified",
)

#: Which retriever each routed path delegates to, and which one it did not use.
#: The unused engine is the counterfactual: if it found the answer, the routing
#: decision is what lost it, and nothing else needs to be inferred.
PATH_ALTERNATIVE = {
    "identifier": ("bm25_boost", "dense"),
    "identifier_fallback_dense": ("dense", "bm25_boost"),
    "conceptual_dense": ("dense", "bm25_boost"),
    "general_dense": ("dense", "bm25_boost"),
    "conceptual_fallback_lexical": ("bm25", "dense"),
    "general_fallback_lexical": ("bm25", "dense"),
}


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def classify(row: dict[str, Any], chunks_by_id: dict[str, Any],
             *, at: int) -> tuple[str, str]:
    """Return ``(category, evidence)``. `unclassified` when no rule fires."""
    expected_ids = set(row["expected_chunks"])
    expected = [chunks_by_id[cid] for cid in row["expected_chunks"] if cid in chunks_by_id]
    top = row["results"][:at]
    retrieved_ids = [r["chunk_id"] for r in top]
    retrieved_documents = {r["document_id"] for r in top}

    # 1. The filters used in this run exclude the answer. Definitive.
    if row.get("filters"):
        filters = Filters(**{k: v for k, v in row["filters"].items()})
        excluded = [c.chunk_id for c in expected if not filters.matches(c)]
        if excluded:
            return ("metadata-filtering problem",
                    f"filters {row['filters']} exclude {excluded}")

    # 2. The evidence itself has no locator: the corpus, not the ranking, is thin.
    if expected and all(c.locator_unavailable for c in expected):
        return ("corpus incompleteness",
                f"every expected chunk is unlocated: {sorted(expected_ids)}")

    # 3. A sibling chunk of the same provision was retrieved instead. Only
    #    provision-level agreement counts: two chunks sharing a chapter are not
    #    a chunking failure, they are merely neighbours in the same statute.
    for result in top:
        if result["chunk_id"] in expected_ids:
            continue
        sibling = chunks_by_id.get(result["chunk_id"])
        if sibling is None:
            continue
        for target in expected:
            if sibling.document_id != target.document_id:
                continue
            shared = [f for f in PROVISION_FIELDS
                      if target.locator.get(f) is not None
                      and sibling.locator.get(f) == target.locator[f]]
            if shared:
                return ("chunking problem",
                        f"rank {result['rank']} {sibling.chunk_id} shares "
                        f"{shared} with the expected chunk")

    # 4. The same words, under a different id.
    for result in top:
        other = chunks_by_id.get(result["chunk_id"])
        if other and result["chunk_id"] not in expected_ids:
            if any(_normalise(other.text) == _normalise(c.text) for c in expected):
                return ("duplicate evidence",
                        f"rank {result['rank']} {other.chunk_id} repeats the expected text")

    # 5. The query cited the provision by name and we still missed it.
    refs = parse_identifiers(row["query"])
    if refs and expected:
        return ("exact-term failure",
                f"query names {[f'{r.kind} {r.number}' for r in refs]}; "
                f"expected {row['expected_locators']} not in top {at}")

    # 6. Right domain, wrong statute — or the right statute, wrong provision.
    if retrieved_documents and not (retrieved_documents & set(row["expected_documents"])):
        return ("document mismatch",
                f"top {at} came from {sorted(retrieved_documents)}, "
                f"expected {row['expected_documents']}")
    if retrieved_documents & set(row["expected_documents"]):
        return ("semantic mismatch",
                f"the expected document is in the top {at} but not the expected provision; "
                f"retrieved {retrieved_ids[:3]}")

    return ("unclassified", f"top {at}: {retrieved_ids}")


def load_ranks(results_dir: str, name: str) -> dict[str, int | None]:
    """``question_id -> rank of the expected chunk`` from a stored run, if any."""
    path = os.path.join(results_dir, f"{name}.jsonl")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return {row["question_id"]: row["rank_of_expected"]
                for row in (json.loads(line) for line in fh if line.strip())}


def _number_tokens(text: str) -> set[str]:
    return set(re.findall(r"\d+[A-Za-z]{0,4}", text.upper()))


def classify_routed(row: dict[str, Any], chunks_by_id: dict[str, Any],
                    alternatives: dict[str, dict[str, int | None]],
                    *, at: int) -> tuple[str, str]:
    """Attribute a routed failure. ``unclassified`` when nothing establishes a cause.

    The rules are ordered so that a cause is only assigned once the causes
    beneath it have been ruled out by evidence rather than by preference. The
    counterfactual rule — "the engine the router did not pick would have found
    it" — is the one that earns routing its own error class, and it is decided
    from that engine's own stored run, not from a guess about what it would have
    done.
    """
    expected_ids = set(row["expected_chunks"])
    expected = [chunks_by_id[cid] for cid in row["expected_chunks"] if cid in chunks_by_id]
    top = row["results"][:at]
    path = row.get("retrieval_path") or "unrouted"
    routing = row.get("routing") or {}
    refs = parse_identifiers(row["query"])

    # 1. There is nothing to rank. The evidence exists but carries no locator,
    #    so the corpus, not the router, is what fell short.
    if expected and all(c.locator_unavailable for c in expected):
        return ("corpus evidence is missing",
                f"every expected chunk is unlocated: {sorted(expected_ids)}")

    # 2. The answer key cites a provision whose number is sitting in the query,
    #    and the parser did not see it — so the query never reached the engine
    #    that handles citations.
    if not refs and row["expected_locators"]:
        query_numbers = _number_tokens(row["query"])
        cited = {n for locator in row["expected_locators"]
                 for n in _number_tokens(str(locator))}
        overlap = query_numbers & cited
        if overlap:
            return ("identifier parser failed",
                    f"the query contains {sorted(overlap)} and the expected locator is "
                    f"{row['expected_locators']}, but parse_identifiers() returned "
                    f"nothing, so the query routed as {routing.get('query_type')}")

    # 3. The citation was parsed and matched — against the wrong chunk. The
    #    locator the query named is carried by something that is not the answer,
    #    which is a defect in the chunk's metadata, not in the ranking.
    if refs:
        wrong = []
        for result in top:
            if result["chunk_id"] in expected_ids:
                continue
            candidate = chunks_by_id.get(result["chunk_id"])
            if candidate is None:
                continue
            for ref in refs:
                field = LOCATOR_FIELD.get(ref.kind, (None, ()))[0]
                if field is None:
                    continue
                value = candidate.locator.get(field)
                held_by_expected = any(
                    str(c.locator.get(field) or "").upper() == ref.number for c in expected)
                if value is not None and str(value).upper() == ref.number \
                        and not held_by_expected:
                    wrong.append(f"rank {result['rank']} {candidate.chunk_id} carries "
                                 f"{field} {value}")
        if wrong:
            return ("identifier metadata is wrong", "; ".join(wrong[:3]))

    # 4. The engine the router did not choose found the answer. This is the
    #    routing decision itself failing, and it is read off that engine's run.
    chosen, other = PATH_ALTERNATIVE.get(path, (None, None))
    if other:
        other_rank = alternatives.get(other, {}).get(row["question_id"])
        if other_rank is not None and other_rank <= at:
            return ("router chose wrong family",
                    f"routed to {chosen} via '{path}' and missed; {other} ranks the "
                    f"expected chunk {other_rank}")

    # 5. A sibling piece of the same provision was retrieved instead.
    for result in top:
        if result["chunk_id"] in expected_ids:
            continue
        sibling = chunks_by_id.get(result["chunk_id"])
        if sibling is None:
            continue
        for target in expected:
            if sibling.document_id != target.document_id:
                continue
            shared = [f for f in PROVISION_FIELDS
                      if target.locator.get(f) is not None
                      and sibling.locator.get(f) == target.locator[f]]
            if shared:
                return ("chunking failed",
                        f"rank {result['rank']} {sibling.chunk_id} shares {shared} "
                        f"with the expected chunk")

    # 6. Staleness is only a cause when a live instrument actually displaced the
    #    stale one. On its own it is a flag, and it stays one.
    if expected and any(c.temporal_status == "STALE" for c in expected):
        displaced = [
            f"rank {r['rank']} {r['chunk_id']}" for r in top
            if (c := chunks_by_id.get(r["chunk_id"])) is not None
            and c.temporal_status != "STALE"
            and c.document_id != expected[0].document_id
            and any(c.locator.get(f) is not None
                    and c.locator.get(f) == expected[0].locator.get(f)
                    for f in PROVISION_FIELDS)
        ]
        if displaced:
            return ("corpus is stale",
                    f"the expected chunk comes from a source Phase 3 marked STALE and "
                    f"a current instrument carrying the same provision outranked it: "
                    f"{displaced[:2]}")

    # 7. Both engines missed. Name the one that was actually asked.
    if chosen and other:
        other_rank = alternatives.get(other, {}).get(row["question_id"])
        detail = (f"{chosen} missed at {at} and {other} "
                  f"{'also missed' if other_rank is None else f'ranked it {other_rank}'}")
        if chosen == "dense":
            return ("dense retrieval failed", detail)
        return ("BM25 failed", detail)

    return ("unclassified", f"top {at}: {[r['chunk_id'] for r in top]}")


def flags(row: dict[str, Any], chunks_by_id: dict[str, Any], *, at: int) -> list[str]:
    """Observations worth recording that do not by themselves explain a ranking."""
    found = []
    expected = [chunks_by_id[cid] for cid in row["expected_chunks"] if cid in chunks_by_id]
    if any(c.temporal_status == "STALE" for c in expected):
        found.append("expected evidence comes from a stale source")
    if any(not r["citable"] for r in row["results"][:at]):
        found.append("a non-citable chunk appears in the evaluated window")
    if row["results"] and not row["results"][0]["citable"]:
        found.append("rank 1 is not citable")
    return found


def analyse(rows: Sequence[dict[str, Any]], chunks_by_id: dict[str, Any],
            *, at: int,
            alternatives: dict[str, dict[str, int | None]] | None = None
            ) -> list[dict[str, Any]]:
    failures = []
    for row in rows:
        rank = row["rank_of_expected"]
        if rank is not None and rank <= at:
            continue
        if alternatives is not None:
            category, evidence = classify_routed(row, chunks_by_id, alternatives, at=at)
        else:
            category, evidence = classify(row, chunks_by_id, at=at)
        failures.append({
            "question_id": row["question_id"],
            "query": row["query"],
            "retriever": row["retriever"],
            "expected_documents": row["expected_documents"],
            "expected_chunks": row["expected_chunks"],
            "expected_locators": row["expected_locators"],
            "rank_of_expected": rank,
            "retrieval_path": row.get("retrieval_path"),
            "routing": row.get("routing"),
            "document_mismatch": not ({r["document_id"] for r in row["results"][:at]}
                                      & set(row["expected_documents"])),
            "locator_mismatch": [
                {"rank": r["rank"], "locator": r["locator"], "document_id": r["document_id"]}
                for r in row["results"][:at]
                if r["chunk_id"] not in set(row["expected_chunks"])
            ][:5],
            "category": category,
            "evidence": evidence,
            "flags": flags(row, chunks_by_id, at=at),
            "top_10": [
                {"rank": r["rank"], "chunk_id": r["chunk_id"], "locator": r["locator"],
                 "score": r["score"], "citable": r["citable"]}
                for r in row["results"][:10]
            ],
        })
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--retriever", default="bm25",
                        help="bm25 | bm25_boost | dense | hybrid | all")
    parser.add_argument("--at", type=int, default=5,
                        help="the k whose failures are analysed (default 5)")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--show", type=int, default=8, help="failures to print in full")
    args = parser.parse_args(argv)

    names = (["bm25", "dense", "bm25_boost", "hybrid", "hybrid_boost", "routed"]
             if args.retriever == "all" else [args.retriever])
    chunks_by_id = {c.chunk_id: c for c in load_chunks()}
    # Loaded once: the routed classifier needs to know where the engine it did
    # *not* pick ranked the same expected chunk.
    alternatives = {name: load_ranks(args.results_dir, name)
                    for name in ("bm25", "bm25_boost", "dense")}

    for name in names:
        path = os.path.join(args.results_dir, f"{name}.jsonl")
        if not os.path.exists(path):
            print(f"no results for {name}; run benchmark_retrieval.py --retriever {name}")
            continue
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        routed = name == "routed"
        failures = analyse(rows, chunks_by_id, at=args.at,
                           alternatives=alternatives if routed else None)
        counts = Counter(f["category"] for f in failures)

        print(f"\n=== {name}: {len(failures)} of {len(rows)} questions "
              f"fail Recall@{args.at} ===")
        for category, count in counts.most_common():
            print(f"  {count:3d}  {category}")
        for failure in failures[:args.show]:
            path = f" via {failure['retrieval_path']}" if failure.get("retrieval_path") else ""
            print(f"\n  {failure['question_id']} [{failure['category']}]{path} "
                  f"{failure['query']}")
            print(f"    expected  {failure['expected_locators']} "
                  f"{failure['expected_chunks']}")
            print(f"    rank      {failure['rank_of_expected']}")
            print(f"    evidence  {failure['evidence']}")
            for flag in failure["flags"]:
                print(f"    flag      {flag}")
            for hit in failure["top_10"][:5]:
                print(f"      {hit['rank']}. {hit['score']:8.3f} {hit['locator']}"
                      f"{'' if hit['citable'] else '  [not citable]'}")

        out = os.path.join(args.results_dir, f"{name}.errors.json")
        with open(out, "w", encoding="utf-8") as fh:
            json.dump({"retriever": name, "at": args.at,
                       "failures": len(failures), "questions": len(rows),
                       "categories": list(ROUTED_CATEGORIES if routed else CATEGORIES),
                       "by_category": dict(counts.most_common()),
                       "cases": failures}, fh, indent=2)
        print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
