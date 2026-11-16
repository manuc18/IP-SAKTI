"""Why a reranked query failed — and above all, *which module* failed.

Phase 6 §16 asks for four types, and the first distinction is the one the whole
phase turns on:

``A`` **candidate-generation failure.** The gold evidence was never in the pool.
     No reranker can recover it, and no reranking metric should be read as
     evidence about the model on these queries.
``B`` **reranking failure.** The gold evidence was in the pool and the reranker
     left it below the cut. This is the model's own miss.
``C`` **competing evidence.** A reranking failure with a specific, visible
     cause: another chunk carrying the *expected chunk's own locator* was ranked
     above it. That is usually a provision split across chunks, or the same
     provision reproduced in a Schedule — a chunking or corpus-redundancy
     observation rather than a ranking one, so it is separated from plain B.
``D`` **the label itself looks wrong.** Flagged for a human, never acted on.

Type D is deliberately *not* exclusive with A, B or C, and pretending otherwise
would misreport it. "The gold label is questionable" is a statement about the
benchmark; "the pool missed it" is a statement about the retriever. A query can
be both, and the D flags are carried alongside the primary type so that neither
count silently absorbs the other. `summarise_failures` reports the A/B/C counts
as a partition and the D count as an overlay, and says so in its own output.

Nothing here alters the golden dataset. A flag is a request for human review
(`scripts/evaluation/review_reranking.py`), and Phase 6 §17 is explicit that no
automated process may mark a question verified.
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai.retrieval.identifiers import parse_identifiers  # noqa: E402

TYPES = {
    "A": "gold evidence absent from the candidate pool (candidate generation)",
    "B": "gold evidence in the pool but ranked too low (reranking)",
    "C": "competing chunks carrying the expected locator outranked the gold",
    "D": "the gold label itself looks questionable — flag for human review",
}


def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()


def competing_chunks(row: Mapping[str, Any], cutoff: int) -> list[dict[str, Any]]:
    """Chunks ranked above the cutoff that carry an expected locator but are not gold.

    The evidence is the chunk's own citation string, which Phase 2 built from its
    locator metadata — not a number appearing somewhere in its text. A chunk that
    merely mentions "section 3" is not competing with section 3.
    """
    expected_chunks = set(row["expected_chunks"])
    expected_documents = set(row["expected_documents"])
    wanted = [_normalise(loc) for loc in row["expected_locators"] if loc]
    out = []
    for result in row["results"]:
        if result["rerank_rank"] > cutoff or result["chunk_id"] in expected_chunks:
            continue
        citation = _normalise(result["locator"])
        if result["document_id"] in expected_documents and any(
                w and w in citation for w in wanted):
            out.append({"rerank_rank": result["rerank_rank"],
                        "chunk_id": result["chunk_id"],
                        "locator": result["locator"],
                        "citable": result["citable"]})
    return out


def label_review_flags(row: Mapping[str, Any]) -> list[dict[str, str]]:
    """Type D signals. Each names the evidence; none changes anything."""
    flags: list[dict[str, str]] = []

    if row.get("router_family") and row.get("query_family"):
        golden = str(row["query_family"]).upper()
        routed = str(row["router_family"]).upper()
        # CONCEPTUAL and GENERAL are one family as far as the golden dataset's
        # two-way label is concerned, so only a disagreement about IDENTIFIER is
        # a real disagreement.
        if (golden == "IDENTIFIER") != (routed == "IDENTIFIER"):
            flags.append({
                "flag": "query_family_disagreement",
                "detail": f"the golden dataset labels this {row['query_family']!r} "
                          f"but the deterministic router reads it as {routed}; one "
                          f"of the two is wrong about whether the query names a "
                          f"provision",
            })

    refs = parse_identifiers(row["query"])
    if refs:
        locators = " ".join(_normalise(loc) for loc in row["expected_locators"])
        unmatched = [f"{r.kind} {r.number}" for r in refs
                     if f"{r.kind} {r.number.lower()}" not in locators]
        if unmatched and len(unmatched) == len(refs):
            flags.append({
                "flag": "query_cites_a_provision_the_label_does_not",
                "detail": f"the query names {', '.join(unmatched)}, and no expected "
                          f"locator ({row['expected_locators']}) refers to it",
            })

    return flags


def classify_failure(row: Mapping[str, Any], *,
                     scores_by_id: Mapping[str, Mapping[str, Any]],
                     at: int = 5) -> dict[str, Any] | None:
    """Classify one query, or return ``None`` if it succeeded at ``at``.

    Success is "an expected chunk appears in the top `at`", the same definition
    `metrics.recall_at_k` uses, so the failure list and the reported Recall@k can
    never disagree about which queries are which.
    """
    rank = row["rank_of_expected"]
    if rank is not None and rank <= at:
        return None

    flags = label_review_flags(row)
    if not row["gold_in_pool"]:
        primary = "A"
        detail = ("no expected chunk appears anywhere in the candidate pool "
                  f"({row['candidates']['pool_size']} candidates from "
                  f"{'+'.join(row['candidates']['engines'])}); the reranker was "
                  "never shown the evidence")
        competing: list[dict[str, Any]] = []
    else:
        competing = competing_chunks(row, at)
        if competing:
            primary = "C"
            detail = (f"the gold chunk sat at candidate position "
                      f"{row['gold_pool_rank']} and was outranked by "
                      f"{len(competing)} chunk(s) carrying the same expected "
                      f"locator — evidence is split or duplicated, not mis-ranked "
                      f"on relevance alone")
        else:
            primary = "B"
            placed = str(rank) if rank is not None else f"outside the top {len(row['results'])}"
            detail = (f"the gold chunk was in the pool at candidate position "
                      f"{row['gold_pool_rank']} and the reranker placed it at "
                      f"{placed}")

    scores = scores_by_id.get(row["question_id"], {})
    return {
        "question_id": row["question_id"],
        "query": row["query"],
        "type": primary,
        "type_description": TYPES[primary],
        "detail": detail,
        "label_review_flags": flags,
        "query_family": row.get("query_family"),
        "router_family": row.get("router_family"),
        "expected_chunks": list(row["expected_chunks"]),
        "expected_locators": list(row["expected_locators"]),
        "gold_in_pool": row["gold_in_pool"],
        "gold_pool_rank": row["gold_pool_rank"],
        "rank_of_expected": rank,
        "candidate_recall": {k: v for k, v in scores.items()
                             if k.startswith("candidate_recall@")},
        "competing_chunks": competing,
        "top_5": [
            {"rerank_rank": r["rerank_rank"], "chunk_id": r["chunk_id"],
             "locator": r["locator"], "citable": r["citable"],
             "dense_rank": r.get("dense_rank"), "lexical_rank": r.get("lexical_rank")}
            for r in row["results"][:5]
        ],
    }


def summarise_failures(failures: Sequence[Mapping[str, Any]],
                       questions: int) -> dict[str, Any]:
    """Counts, with A/B/C as a partition and D as an overlay."""
    by_type = Counter(f["type"] for f in failures)
    flagged = [f for f in failures if f["label_review_flags"]]
    generation = by_type.get("A", 0)
    reranking = by_type.get("B", 0) + by_type.get("C", 0)
    total = len(failures)
    return {
        "at": 5,
        "questions": questions,
        "failures": total,
        "types": {t: by_type.get(t, 0) for t in ("A", "B", "C")},
        "type_descriptions": TYPES,
        "candidate_generation_failures": generation,
        "reranking_failures": reranking,
        "candidate_generation_share": round(generation / total, 4) if total else 0.0,
        "reranking_share": round(reranking / total, 4) if total else 0.0,
        "type_D_flagged": len(flagged),
        "type_D_note": "Type D is an overlay, not a fourth bucket: a questionable "
                       "label can coexist with either a pool miss or a ranking "
                       "miss, and collapsing it into the partition would hide "
                       "one of the two. Nothing is altered in the golden dataset; "
                       "these are queued for human review.",
        "type_D_questions": [
            {"question_id": f["question_id"],
             "flags": [flag["flag"] for flag in f["label_review_flags"]]}
            for f in flagged
        ],
    }
