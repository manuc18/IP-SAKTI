"""The reranking contract: reorder a candidate pool, change nothing else.

A reranker receives a query and a pool of candidates and returns the same
candidates in a different order. That is the entire permitted effect. It may not
add a chunk, drop provenance, edit a locator, or decide citability — the Phase 3
invariant that `citable` is copied and never inferred survives here exactly as
it survived Phase 4 retrieval and Phase 5 routing.

Two consequences are worth stating because they are easy to violate by accident.

**Truncation to `top_k` is a presentation cut, not a relevance judgement.** The
candidates that fall outside it were still scored, and `rerank()` returns the
scored prefix rather than pretending the tail did not exist. Callers that need
the full scored list ask for `top_k=len(candidates)`.

**A reranker that cannot see a chunk cannot rescue it.** Recall at the pool
boundary is set by :mod:`ai.retrieval.candidates` and is measured separately
(Phase 6 §2 and §16). Every failure analysed later is first sorted into "the
evidence was never in the pool" and "the evidence was in the pool and ranked too
low", because those two are different bugs in different modules and conflating
them has cost every retrieval project that ever conflated them.

The identity reranker in this module is not a placeholder. Without a run that
preserves the candidate generator's own ordering, no later number can be
attributed to reranking rather than to the wider pool the reranking happens to
sit on top of.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ai.retrieval.base import RetrievalResult
from ai.retrieval.candidates import Candidate

#: Bumped whenever a change here would alter a stored reranking.
RERANKING_VERSION = "0.1.0"


@dataclass(frozen=True)
class RerankedResult:
    """One reranked candidate: the original, plus where this reranker put it.

    `components` is the reranker's own working — the individual feature values,
    or the raw model logit — kept so a ranking can be explained after the fact
    rather than only observed. It is diagnostic output and nothing downstream is
    entitled to treat it as provenance.
    """

    candidate: Candidate
    rerank_score: float
    rerank_rank: int
    components: Mapping[str, float] = field(default_factory=dict)

    # -- passthroughs. The original fields survive, by construction. -----------

    @property
    def chunk_id(self) -> str:
        return self.candidate.chunk_id

    @property
    def document_id(self) -> str:
        return self.candidate.document_id

    @property
    def citable(self) -> bool:
        return self.candidate.citable

    @property
    def locator(self) -> str:
        return self.candidate.locator

    @property
    def text(self) -> str:
        return self.candidate.text

    @property
    def metadata(self) -> Mapping[str, Any]:
        return self.candidate.metadata

    def to_dict(self) -> dict[str, Any]:
        """Phase 6 §4's shape, with the candidate's own record kept alongside."""
        record = self.candidate.to_dict()
        record.update({
            "chunk_id": self.chunk_id,
            "rerank_score": round(float(self.rerank_score), 6),
            "rerank_rank": self.rerank_rank,
            "candidate_order": self.candidate.order,
            "components": {k: round(float(v), 6) for k, v in self.components.items()},
        })
        return record

    def to_retrieval_result(self) -> RetrievalResult:
        """A Phase 4 result, so a reranked list is drop-in for a retrieved one."""
        return self.candidate.to_result(
            score=self.rerank_score,
            rank=self.rerank_rank,
            extra={"rerank_score": float(self.rerank_score),
                   "rerank_rank": self.rerank_rank,
                   "candidate_order": self.candidate.order},
        )


def order_by_score(scored: Sequence[tuple[Candidate, float, Mapping[str, float]]],
                   top_k: int) -> list[RerankedResult]:
    """Sort by score, break ties by chunk_id, number from 1.

    Ties break on `chunk_id` rather than on the incoming order, for the same
    reason BM25 does: a ranking should be a function of the query and the pool,
    not of which engine happened to be asked first. Where the incoming order is
    itself the signal — the identity baseline — the score already encodes it.
    """
    ordered = sorted(scored, key=lambda item: (-item[1], item[0].chunk_id))
    return [
        RerankedResult(candidate=candidate, rerank_score=float(score),
                       rerank_rank=rank, components=dict(components))
        for rank, (candidate, score, components) in enumerate(ordered[:top_k], start=1)
    ]


class Reranker(ABC):
    """Reorders a candidate pool for a query."""

    #: Stable identifier used in results files and reports.
    name: str = "reranker"
    version: str = RERANKING_VERSION

    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[Candidate],
               top_k: int = 10) -> list[RerankedResult]:
        """Return at most ``top_k`` candidates, best first, ranks from 1."""

    def config(self) -> dict[str, Any]:
        return {"reranker": self.name, "version": self.version}


class IdentityReranker(Reranker):
    """Keeps the candidate generator's ordering. The control condition.

    The score is ``1 / order`` so that it decreases monotonically and can be
    compared with any other reranker's output by the same machinery. It is not a
    relevance estimate and must never be read as one — it is a restatement of
    the position the candidate arrived in.
    """

    name = "identity"
    version = RERANKING_VERSION

    def rerank(self, query: str, candidates: Sequence[Candidate],
               top_k: int = 10) -> list[RerankedResult]:
        return [
            RerankedResult(candidate=candidate,
                           rerank_score=1.0 / candidate.order,
                           rerank_rank=rank,
                           components={"candidate_order": float(candidate.order)})
            for rank, candidate in enumerate(candidates[:top_k], start=1)
        ]

    def config(self) -> dict[str, Any]:
        return {
            "reranker": self.name,
            "version": self.version,
            "scoring": "1 / candidate_order — a restatement of the incoming "
                       "order, not a relevance estimate",
        }
