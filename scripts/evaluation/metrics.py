"""Retrieval metrics, with the multi-evidence formulation written down.

Most golden questions have exactly one right chunk, but not all: a definition
section split across three chunks is genuinely three pieces of correct evidence.
That makes the relevance formulation a choice, so it is stated here rather than
left implicit in the code.

**Relevance is binary.** A chunk is either in the question's `expected_chunks`
or it is not. Graded relevance would mean ranking one true provision above
another, and nothing in this corpus supports that judgement.

**Recall@k is hit-rate, not coverage.** A question counts as recalled at k when
*at least one* expected chunk appears in the top k. This is the right question
to ask of a legal answer engine: it needs to put the reader in front of the
provision, not enumerate every chunk the provision was split into.

**nDCG@k uses the ideal ranking for that question.** IDCG sums the discounts of
the first `min(len(expected), k)` positions, so a question with three expected
chunks is not penalised for the fact that only one of them can be rank 1. Where
a question has a single expected chunk, nDCG@k reduces to `1/log2(rank+1)`.

**MRR uses the first expected chunk to appear**, and contributes 0 when none
appears within the evaluated depth — which makes MRR depth-dependent, so the
depth is recorded alongside every reported figure.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence


def first_hit_rank(ranked: Sequence[str], expected: Iterable[str]) -> int | None:
    """1-based rank of the first expected id, or None if it never appears."""
    wanted = set(expected)
    for rank, identifier in enumerate(ranked, start=1):
        if identifier in wanted:
            return rank
    return None


def recall_at_k(ranked: Sequence[str], expected: Iterable[str], k: int) -> float:
    """1.0 when any expected id is in the top k."""
    wanted = set(expected)
    return 1.0 if any(identifier in wanted for identifier in ranked[:k]) else 0.0


def reciprocal_rank(ranked: Sequence[str], expected: Iterable[str]) -> float:
    rank = first_hit_rank(ranked, expected)
    return 1.0 / rank if rank else 0.0


def ndcg_at_k(ranked: Sequence[str], expected: Iterable[str], k: int) -> float:
    wanted = set(expected)
    if not wanted:
        return 0.0
    dcg = sum(1.0 / math.log2(rank + 1)
              for rank, identifier in enumerate(ranked[:k], start=1)
              if identifier in wanted)
    ideal = sum(1.0 / math.log2(rank + 1)
                for rank in range(1, min(len(wanted), k) + 1))
    return dcg / ideal if ideal else 0.0


def aggregate(per_question: Sequence[dict[str, float]], keys: Sequence[str]) -> dict[str, float]:
    """Macro-average over questions; an empty set averages to 0.0, not NaN."""
    if not per_question:
        return {key: 0.0 for key in keys}
    return {key: sum(row[key] for row in per_question) / len(per_question) for key in keys}


METRIC_KEYS = ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@5", "ndcg@10")


def score_question(ranked_chunk_ids: Sequence[str], expected_chunks: Sequence[str],
                   ranked_document_ids: Sequence[str],
                   expected_documents: Sequence[str]) -> dict[str, float]:
    """Every metric for one question, at chunk level and at document level.

    Document-level figures are reported alongside because they separate two very
    different failures: retrieving the wrong statute, and retrieving the right
    statute at the wrong provision. Only the first is a search failure.
    """
    scores = {
        "recall@1": recall_at_k(ranked_chunk_ids, expected_chunks, 1),
        "recall@5": recall_at_k(ranked_chunk_ids, expected_chunks, 5),
        "recall@10": recall_at_k(ranked_chunk_ids, expected_chunks, 10),
        "mrr": reciprocal_rank(ranked_chunk_ids, expected_chunks),
        "ndcg@5": ndcg_at_k(ranked_chunk_ids, expected_chunks, 5),
        "ndcg@10": ndcg_at_k(ranked_chunk_ids, expected_chunks, 10),
    }
    scores.update({
        "document_recall@1": recall_at_k(ranked_document_ids, expected_documents, 1),
        "document_recall@5": recall_at_k(ranked_document_ids, expected_documents, 5),
        "document_recall@10": recall_at_k(ranked_document_ids, expected_documents, 10),
    })
    return scores
