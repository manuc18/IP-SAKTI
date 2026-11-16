"""Reciprocal Rank Fusion over independently ranked lists.

RRF is the right first fusion for this corpus precisely because it ignores
scores. A BM25 score of 24 and a cosine similarity of 0.71 are not on any shared
scale, and inventing one — min-max normalisation, z-scores — would bake a
calibration assumption into the baseline before anything has been measured. RRF
asks each retriever only for an ordering, which is the one thing both are
entitled to assert.

Nothing here is tuned. `k = 60` is the constant from Cormack et al. (2009) and is
left alone deliberately: the benchmark's job at this stage is to say whether
fusion helps at all, and a `k` fitted to these queries would answer a different,
much less useful question.
"""
from __future__ import annotations

from typing import Any, Sequence

from ai.retrieval.base import Filters, RetrievalResult, Retriever

RRF_K = 60
#: How deep each retriever is asked to rank before fusion. Deeper than the
#: requested top_k, so a chunk one list ranks 30th can still surface when the
#: other ranks it 2nd — which is the only reason to fuse in the first place.
FUSION_DEPTH = 50


def reciprocal_rank_fusion(rankings: Sequence[Sequence[str]], *, k: int = RRF_K,
                           weights: Sequence[float] | None = None) -> list[tuple[str, float]]:
    """Fuse ranked id lists into one, best first. Ties break on id."""
    if weights is None:
        weights = [1.0] * len(rankings)
    if len(weights) != len(rankings):
        raise ValueError("weights must match the number of rankings")
    fused: dict[str, float] = {}
    for weight, ranking in zip(weights, rankings):
        for rank, identifier in enumerate(ranking, start=1):
            fused[identifier] = fused.get(identifier, 0.0) + weight / (k + rank)
    return sorted(fused.items(), key=lambda item: (-item[1], item[0]))


class HybridRRFRetriever(Retriever):
    """Fuses any number of retrievers; used here for BM25 + BGE-M3."""

    name = "hybrid"
    version = "0.1.0"

    def __init__(self, retrievers: Sequence[Retriever], *, k: int = RRF_K,
                 depth: int = FUSION_DEPTH, weights: Sequence[float] | None = None):
        if not retrievers:
            raise ValueError("hybrid retrieval needs at least one retriever")
        self.retrievers = list(retrievers)
        self.k = k
        self.depth = depth
        self.weights = list(weights) if weights else [1.0] * len(self.retrievers)

    def search(self, query: str, filters: Filters | None = None,
               top_k: int = 10) -> list[RetrievalResult]:
        depth = max(self.depth, top_k)
        per_retriever = [r.search(query, filters, depth) for r in self.retrievers]
        contributions: dict[str, dict[str, int]] = {}
        rankings: list[list[str]] = []
        for retriever, results in zip(self.retrievers, per_retriever):
            rankings.append([r.chunk_id for r in results])
            for result in results:
                contributions.setdefault(result.chunk_id, {})[retriever.name] = result.rank

        fused = reciprocal_rank_fusion(rankings, k=self.k, weights=self.weights)[:top_k]
        # Rebuild each hit from its source chunk so provenance is never inferred
        # from whichever retriever happened to rank it first.
        by_id = {r.chunk_id: r for results in per_retriever for r in results}
        results: list[RetrievalResult] = []
        for rank, (chunk_id, score) in enumerate(fused, start=1):
            source = by_id[chunk_id]
            metadata = dict(source.metadata)
            metadata["contributing_ranks"] = contributions[chunk_id]
            results.append(RetrievalResult(
                chunk_id=source.chunk_id,
                document_id=source.document_id,
                score=float(score),
                rank=rank,
                text=source.text,
                locator=source.locator,
                citable=source.citable,
                metadata=metadata,
            ))
        return results

    def config(self) -> dict[str, Any]:
        return {
            "retriever": self.name,
            "version": self.version,
            "fusion": "reciprocal_rank_fusion",
            "k": self.k,
            "depth": self.depth,
            "weights": self.weights,
            "components": [r.config() for r in self.retrievers],
        }
