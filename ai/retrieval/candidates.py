"""Candidate generation — a recall stage, deliberately not a ranking stage.

Phase 5 ended with a router that beats every fused configuration at rank 1 and
loses to one of them at rank 10 (routed Recall@10 0.964 vs boosted-hybrid 0.982).
That gap is the whole reason this module exists. A router picks one engine, and
nothing downstream can recover a chunk the chosen engine never returned; a
reranker cannot promote what it was never shown. So the pipeline splits in two:
gather widely here, order carefully later.

**This stage asserts no final ordering.** The candidates come back in a
round-robin interleave of the contributing engines' own rankings — dense #1,
lexical #1, dense #2, lexical #2, and so on, keeping the first appearance of each
chunk. That is a merge, not a fusion: it mixes no scores, invents no shared
scale, and claims only that both engines' opinions deserve to be seen. RRF is
available (`order="rrf"`) as a reference ordering for the identity baseline,
because Phase 4 and 5 already measured it and a reader is entitled to the
comparison — but it is not the default, and Phase 6 §1 is explicit that fusion
is not what this stage is for.

**Two modes, and the primary one is `expanded`.**

``expanded``  BGE-M3 to depth 50, BM25+identifier-boost to depth 50, unioned by
              chunk_id. This is the mode the reranking experiments run on.
``routed``    The Phase 5 router picks one engine and it is asked for the full
              depth. Kept because the router is a measured result, not a draft,
              and the reranking numbers mean more when the routed pool is there
              to show what the expansion actually bought.

**Provenance survives, exactly as in Phase 4.** A :class:`Candidate` carries the
chunk's locator, citability, temporal status, evidence level, physical pages and
document id through unchanged, copied from the underlying
:class:`~ai.retrieval.base.RetrievalResult`. Candidate generation never sets or
clears `citable`, and it never filters on authority: Phase 6 §11 wants to know
whether Level B, Level C and stale chunks are retrieved *because they are
relevant*, and excluding them here would destroy the evidence for that question.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ai.retrieval.base import Filters, RetrievalResult, Retriever
from ai.retrieval.hybrid import RRF_K, reciprocal_rank_fusion
from ai.retrieval.router import RoutingDecision, classify

#: Phase 6 §1. Deep enough that the candidate-recall experiment has something to
#: measure at every depth it asks about (10/20/30/50), shallow enough that a
#: cross-encoder scoring the whole pool stays practical on this machine.
DEFAULT_DENSE_DEPTH = 50
DEFAULT_LEXICAL_DEPTH = 50

MODES = ("expanded", "routed")
ORDERS = ("interleaved", "rrf")

#: Engine labels used in `Candidate.sources` and in every stored record.
DENSE = "dense"
LEXICAL = "bm25_boost"


@dataclass(frozen=True)
class Candidate:
    """One chunk offered to the reranker, with every engine's view of it.

    `dense_rank` / `lexical_rank` are ``None`` when that engine did not return
    the chunk within its depth — which is a fact worth keeping, not a zero. A
    feature that treats "ranked 51st" and "not ranked" as the same number throws
    away the difference between an engine that disagreed and an engine that was
    never asked.
    """

    chunk_id: str
    document_id: str
    text: str
    locator: str
    citable: bool
    metadata: Mapping[str, Any]
    order: int
    sources: tuple[str, ...] = ()
    dense_score: float | None = None
    dense_rank: int | None = None
    lexical_score: float | None = None
    lexical_rank: int | None = None
    identifier_boost: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "locator": self.locator,
            "citable": self.citable,
            "order": self.order,
            "sources": list(self.sources),
            "dense_score": self.dense_score,
            "dense_rank": self.dense_rank,
            "lexical_score": self.lexical_score,
            "lexical_rank": self.lexical_rank,
            "identifier_boost": self.identifier_boost,
            "temporal_status": self.metadata.get("temporal_status"),
            "source_evidence_level": self.metadata.get("source_evidence_level"),
            "physical_pages": self.metadata.get("physical_pages"),
        }

    def to_result(self, *, score: float, rank: int,
                  extra: Mapping[str, Any] | None = None) -> RetrievalResult:
        """Re-express this candidate as a Phase 4 result, provenance intact."""
        metadata = dict(self.metadata)
        if extra:
            metadata.update(extra)
        return RetrievalResult(
            chunk_id=self.chunk_id,
            document_id=self.document_id,
            score=float(score),
            rank=rank,
            text=self.text,
            locator=self.locator,
            citable=self.citable,
            metadata=metadata,
        )


@dataclass(frozen=True)
class CandidateSet:
    """The pool for one query, plus how it was assembled."""

    query: str
    mode: str
    order: str
    candidates: tuple[Candidate, ...]
    depths: Mapping[str, int]
    engines: tuple[str, ...]
    routing: RoutingDecision | None = None
    latency_ms: float = 0.0
    per_engine_latency_ms: Mapping[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.candidates)

    def ids(self) -> list[str]:
        return [c.chunk_id for c in self.candidates]

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "order": self.order,
            "engines": list(self.engines),
            "depths": dict(self.depths),
            "pool_size": len(self.candidates),
            "from_dense_only": sum(1 for c in self.candidates if c.sources == (DENSE,)),
            "from_lexical_only": sum(1 for c in self.candidates if c.sources == (LEXICAL,)),
            "from_both": sum(1 for c in self.candidates if len(c.sources) > 1),
            "latency_ms": round(self.latency_ms, 2),
            "per_engine_latency_ms": {k: round(v, 2)
                                      for k, v in self.per_engine_latency_ms.items()},
            "routing": self.routing.to_dict() if self.routing else None,
        }


def _interleave(rankings: Sequence[Sequence[str]]) -> list[str]:
    """Round-robin merge, keeping each id's first appearance.

    Position *i* of every list is taken before position *i+1* of any list, so no
    engine's opinion is systematically deferred and no score is compared across
    engines. Ties within a round follow the order the engines were given, which
    is fixed by the mode, so the merge is a pure function of its inputs.
    """
    merged: list[str] = []
    seen: set[str] = set()
    depth = max((len(r) for r in rankings), default=0)
    for position in range(depth):
        for ranking in rankings:
            if position < len(ranking):
                identifier = ranking[position]
                if identifier not in seen:
                    seen.add(identifier)
                    merged.append(identifier)
    return merged


class CandidateGenerator:
    """Builds the pool a reranker sees. Ranks nothing on its own account."""

    version = "0.1.0"

    def __init__(self, *, dense: Retriever, lexical: Retriever,
                 mode: str = "expanded", order: str = "interleaved",
                 dense_depth: int = DEFAULT_DENSE_DEPTH,
                 lexical_depth: int = DEFAULT_LEXICAL_DEPTH,
                 aliases: Mapping[str, Sequence[str]] | None = None,
                 rrf_k: int = RRF_K):
        if mode not in MODES:
            raise ValueError(f"unknown candidate mode: {mode}")
        if order not in ORDERS:
            raise ValueError(f"unknown candidate order: {order}")
        self.dense = dense
        self.lexical = lexical
        self.mode = mode
        self.order = order
        self.dense_depth = dense_depth
        self.lexical_depth = lexical_depth
        self.aliases = dict(aliases or {})
        self.rrf_k = rrf_k

    # ------------------------------------------------------------- gathering --

    def _engines(self, query: str) -> tuple[list[tuple[str, Retriever, int]],
                                            RoutingDecision | None]:
        """Which engines run for this query, and to what depth."""
        if self.mode == "expanded":
            return ([(DENSE, self.dense, self.dense_depth),
                     (LEXICAL, self.lexical, self.lexical_depth)], None)

        decision = classify(query, self.aliases)
        if decision.query_type == "IDENTIFIER":
            return ([(LEXICAL, self.lexical, self.lexical_depth)], decision)
        return ([(DENSE, self.dense, self.dense_depth)], decision)

    def generate(self, query: str, filters: Filters | None = None) -> CandidateSet:
        plan, decision = self._engines(query)
        started = time.perf_counter()

        rankings: list[list[str]] = []
        by_id: dict[str, dict[str, Any]] = {}
        per_engine_ms: dict[str, float] = {}

        for label, retriever, depth in plan:
            engine_started = time.perf_counter()
            results = retriever.search(query, filters, depth)
            per_engine_ms[label] = (time.perf_counter() - engine_started) * 1000.0
            rankings.append([r.chunk_id for r in results])
            for result in results:
                entry = by_id.setdefault(result.chunk_id, {"result": result, "sources": []})
                entry["sources"].append(label)
                # The first engine to return a chunk supplies its text and
                # provenance. Both engines read the same chunk row, so this is a
                # choice of copy, not of content.
                if label == DENSE:
                    entry["dense_score"] = result.score
                    entry["dense_rank"] = result.rank
                else:
                    entry["lexical_score"] = result.score
                    entry["lexical_rank"] = result.rank
                    entry["identifier_boost"] = float(
                        result.metadata.get("identifier_boost") or 0.0)

        if self.order == "rrf":
            merged = [cid for cid, _ in reciprocal_rank_fusion(rankings, k=self.rrf_k)]
        else:
            merged = _interleave(rankings)

        candidates = tuple(
            Candidate(
                chunk_id=chunk_id,
                document_id=by_id[chunk_id]["result"].document_id,
                text=by_id[chunk_id]["result"].text,
                locator=by_id[chunk_id]["result"].locator,
                citable=by_id[chunk_id]["result"].citable,
                metadata=dict(by_id[chunk_id]["result"].metadata),
                order=position,
                sources=tuple(by_id[chunk_id]["sources"]),
                dense_score=by_id[chunk_id].get("dense_score"),
                dense_rank=by_id[chunk_id].get("dense_rank"),
                lexical_score=by_id[chunk_id].get("lexical_score"),
                lexical_rank=by_id[chunk_id].get("lexical_rank"),
                identifier_boost=by_id[chunk_id].get("identifier_boost", 0.0),
            )
            for position, chunk_id in enumerate(merged, start=1)
        )

        return CandidateSet(
            query=query,
            mode=self.mode,
            order=self.order,
            candidates=candidates,
            depths={label: depth for label, _, depth in plan},
            engines=tuple(label for label, _, _ in plan),
            routing=decision,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            per_engine_latency_ms=per_engine_ms,
        )

    def config(self) -> dict[str, Any]:
        return {
            "candidate_generator": "union",
            "version": self.version,
            "mode": self.mode,
            "order": self.order,
            "dense_depth": self.dense_depth,
            "lexical_depth": self.lexical_depth,
            "rrf_k": self.rrf_k if self.order == "rrf" else None,
            "deduplication": "by chunk_id, first appearance wins",
            "components": {
                "dense": self.dense.config(),
                "lexical": self.lexical.config(),
            },
        }
