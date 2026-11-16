"""The objects that cross the answer layer's boundaries.

Three of them carry the weight of Phase 7, and each exists to close a specific
hole.

**`Evidence` is a numbered passport, not a chunk.** It carries a `marker` — the
``[1]`` a reader sees — and every provider cites by marker and by nothing else.
That is deliberate and it is the whole anti-hallucination design: a provider
that can only emit small integers cannot invent a chunk id, a section number, a
page or a URL and have it accepted, because it never writes those strings. The
worst it can do is cite `[9]` when four items were supplied, which is caught by
a dictionary lookup. Compare the alternative, where a model emits
``"IN-PATENTS-ACT-1970::section-3"`` and something downstream has to decide
whether that string is real.

**`EvidenceSet` is the registry the customs officer checks against.** It is
built once, from retrieval output, before the provider runs, and it fingerprints
itself. A citation is admissible if and only if it resolves inside the set that
was handed to the engine — `fingerprint` is what makes "the same set" checkable
rather than assumed.

**`Answer` is never model output.** The provider returns prose; every field of
the final object is assembled here from evidence records the corpus produced.
The provider's text can be rejected, but it cannot contribute metadata.

Nothing in this module imports a model, a tensor library or a retriever. Phase 7
§25: the answering package stays importable in a process that will never load
BGE-M3.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

#: Bumped whenever a change here would alter a stored answer object.
ANSWERING_VERSION = "0.1.0"

# -- Vocabulary, read off the corpus rather than invented (Phase 7 §8). --------

#: `temporal_status` values the corpus actually uses. UNKNOWN is not present in
#: chunking 0.2.0 (1,356 CURRENT and 1,717 STALE, and nothing else) but
#: `Chunk.from_row` defaults to it for a row that omits the field, so the answer
#: layer must have a defined behaviour for it. Phase 7 §8 forbids inventing new
#: categories, so this list is exactly the three that can reach us.
CURRENT = "CURRENT"
STALE = "STALE"
UNKNOWN_TEMPORAL = "UNKNOWN"
TEMPORAL_STATUSES = (CURRENT, STALE, UNKNOWN_TEMPORAL)

#: Source evidence levels, from `data/metadata/provenance_registry.json`. The
#: corpus contains A, B and C; D is defined by the registry and reachable.
EVIDENCE_LEVELS = ("A", "B", "C", "D")

#: Ordinal confidence. Not a probability, and deliberately not a number: nothing
#: in this pipeline estimates the likelihood that an answer is legally correct,
#: and a decimal would invite exactly that reading.
HIGH = "HIGH"
MODERATE = "MODERATE"
LOW = "LOW"
NONE = "NONE"
CONFIDENCE_LEVELS = (HIGH, MODERATE, LOW, NONE)


@dataclass(frozen=True)
class Evidence:
    """One passage offered to the answer engine, with its provenance intact.

    Built from a Phase 4 :class:`~ai.retrieval.base.RetrievalResult`, which is
    also what a Phase 6 :class:`~ai.reranking.base.RerankedResult` converts to.
    The answer layer therefore consumes the published retrieval contract and
    never the reranker's internals — Phase 7 §2's separation is enforced by the
    type, not by convention.
    """

    marker: int
    chunk_id: str
    document_id: str
    document_name: str
    citation: str
    text: str
    citable: bool
    locator_unavailable: bool
    locator_fields: Mapping[str, Any] = field(default_factory=dict)
    temporal_status: str = UNKNOWN_TEMPORAL
    source_evidence_level: str = "D"
    source_status: str | None = None
    physical_pages: tuple[int, ...] = ()
    official_source_url: str | None = None
    source_filename: str | None = None
    source_sha256: str | None = None
    source_version_id: str | None = None
    consolidated_as_of: str | None = None
    heading: str | None = None
    unit_type: str | None = None
    domain: str | None = None
    retrieval_rank: int | None = None
    retrieval_score: float | None = None

    @classmethod
    def from_result(cls, result: Any, marker: int) -> "Evidence":
        """Adapt a `RetrievalResult`. Copies provenance; decides nothing."""
        metadata: Mapping[str, Any] = getattr(result, "metadata", {}) or {}
        pages = metadata.get("physical_pages") or []
        return cls(
            marker=marker,
            chunk_id=result.chunk_id,
            document_id=result.document_id,
            document_name=str(metadata.get("short_title")
                              or metadata.get("title") or result.document_id),
            citation=result.locator,
            text=result.text,
            # Copied, never inferred. Phase 3's invariant survives Phase 7 for
            # the same reason it survived Phases 4-6.
            citable=bool(result.citable),
            locator_unavailable=bool(metadata.get("locator_unavailable", False)),
            locator_fields=dict(metadata.get("locator_fields") or {}),
            temporal_status=str(metadata.get("temporal_status") or UNKNOWN_TEMPORAL),
            source_evidence_level=str(metadata.get("source_evidence_level") or "D"),
            source_status=metadata.get("source_status"),
            physical_pages=tuple(int(p) for p in pages),
            official_source_url=metadata.get("official_source_url"),
            source_filename=metadata.get("source_filename"),
            source_sha256=metadata.get("source_sha256"),
            source_version_id=metadata.get("source_version_id"),
            consolidated_as_of=metadata.get("consolidated_as_of"),
            heading=metadata.get("heading"),
            unit_type=metadata.get("unit_type"),
            domain=metadata.get("domain"),
            retrieval_rank=getattr(result, "rank", None),
            retrieval_score=getattr(result, "score", None),
        )

    @property
    def locator_text(self) -> str:
        """The citation minus the document name it already repeats."""
        citation = (self.citation or "").strip()
        head, separator, tail = citation.partition(" — ")
        if separator:
            citation = tail
        elif citation.startswith(self.document_name):
            citation = citation[len(self.document_name):]
        return citation.lstrip(" —–-,").strip()

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "document_name": self.document_name,
            "citation": self.citation,
            "citable": self.citable,
            "locator_unavailable": self.locator_unavailable,
            "temporal_status": self.temporal_status,
            "source_evidence_level": self.source_evidence_level,
            "physical_pages": list(self.physical_pages),
            "retrieval_rank": self.retrieval_rank,
        }


@dataclass(frozen=True)
class EvidenceSet:
    """The ordered evidence handed to one answer engine call.

    This is the *only* registry a citation may resolve against. Phase 7 §11: no
    passport, no entry.
    """

    items: tuple[Evidence, ...] = ()

    @classmethod
    def from_results(cls, results: Sequence[Any]) -> "EvidenceSet":
        """Number results from 1 in the order given. Order is the ranking."""
        return cls(tuple(Evidence.from_result(result, marker)
                         for marker, result in enumerate(results, start=1)))

    def __len__(self) -> int:
        return len(self.items)

    def __iter__(self) -> Iterable[Evidence]:
        return iter(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)

    def by_marker(self, marker: int) -> Evidence | None:
        for item in self.items:
            if item.marker == marker:
                return item
        return None

    def by_chunk_id(self, chunk_id: str) -> Evidence | None:
        for item in self.items:
            if item.chunk_id == chunk_id:
                return item
        return None

    @property
    def markers(self) -> tuple[int, ...]:
        return tuple(item.marker for item in self.items)

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(item.chunk_id for item in self.items)

    @property
    def fingerprint(self) -> str:
        """Identity of this exact set, in this exact order.

        Stored on the answer so "the citation came from the evidence that was
        supplied" is a checkable claim afterwards rather than a trusted one.
        """
        digest = hashlib.sha256()
        for item in self.items:
            digest.update(f"{item.marker}\x1f{item.chunk_id}\x1e".encode("utf-8"))
        return digest.hexdigest()[:16]


@dataclass(frozen=True)
class SourceDocument:
    """Machine-resolvable identity of the PDF a citation came out of.

    Every field is copied from the chunk. Phase 7 §9: a URL that the provenance
    registry established is preserved; one that it did not is absent, never
    guessed from the document's name.
    """

    document_id: str
    filename: str | None = None
    sha256: str | None = None
    official_source_url: str | None = None
    source_version_id: str | None = None
    consolidated_as_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"document_id": self.document_id, "filename": self.filename,
                "sha256": self.sha256,
                "official_source_url": self.official_source_url,
                "source_version_id": self.source_version_id,
                "consolidated_as_of": self.consolidated_as_of}


@dataclass(frozen=True)
class Citation:
    """A validated citation. Constructed from evidence, never from prose."""

    marker: int
    chunk_id: str
    document_id: str
    locator: str
    pages: tuple[int, ...]
    citation_text: str
    source_evidence_level: str
    temporal_status: str
    source_document: SourceDocument
    locator_unavailable: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "locator": self.locator,
            "pages": list(self.pages),
            "citation_text": self.citation_text,
            "source_evidence_level": self.source_evidence_level,
            "temporal_status": self.temporal_status,
            "locator_unavailable": self.locator_unavailable,
            "source_document": self.source_document.to_dict(),
        }


@dataclass(frozen=True)
class Failure:
    """One structured reason something was rejected.

    Phase 7 §10: a validator that repairs what it finds teaches the caller that
    its output can be trusted without looking. This carries a stable `code` so a
    caller can branch, and a `detail` so a person can read it.
    """

    code: str
    detail: str
    marker: int | None = None
    chunk_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "detail": self.detail,
                "marker": self.marker, "chunk_id": self.chunk_id}


@dataclass(frozen=True)
class Answer:
    """The engine's output. Phase 7 §5's shape, plus the working behind it.

    `grounded` is a deterministic property of the text and the evidence — every
    citation resolved and every claim carried one — and says nothing about
    whether the answer is legally correct. Nothing in Phase 7 measures that.
    """

    answer: str
    citations: tuple[Citation, ...] = ()
    confidence: str = NONE
    #: The two axes behind `confidence` and every factor that set it, so a
    #: reader is told *why* it says what it says. See `ai/answering/confidence.py`.
    #: Empty on an abstention, which has no confidence to explain.
    confidence_report: Mapping[str, Any] = field(default_factory=dict)
    grounded: bool = False
    insufficient_evidence: bool = True
    insufficient_evidence_reasons: tuple[Failure, ...] = ()
    conflict_detected: bool = False
    conflict_chunks: tuple[str, ...] = ()
    cautions: tuple[str, ...] = ()
    validation_failures: tuple[Failure, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    uncited_evidence: tuple[str, ...] = ()
    evidence_fingerprint: str = ""
    provider: str = ""
    policy: Mapping[str, Any] = field(default_factory=dict)
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    version: str = ANSWERING_VERSION

    def to_dict(self) -> dict[str, Any]:
        """The Phase 7 §5 contract, and the shape §24's future `POST /answer`
        returns. Diagnostics are included but are explicitly not part of it."""
        return {
            "answer": self.answer,
            "citations": [c.to_dict() for c in self.citations],
            "confidence": self.confidence,
            "confidence_report": dict(self.confidence_report),
            "grounded": self.grounded,
            "insufficient_evidence": self.insufficient_evidence,
            "insufficient_evidence_reasons": [
                f.to_dict() for f in self.insufficient_evidence_reasons],
            "conflict_detected": self.conflict_detected,
            "conflict_chunks": list(self.conflict_chunks),
            "cautions": list(self.cautions),
            "validation_failures": [f.to_dict() for f in self.validation_failures],
            "unsupported_claims": list(self.unsupported_claims),
            "uncited_evidence": list(self.uncited_evidence),
            "evidence_fingerprint": self.evidence_fingerprint,
            "provider": self.provider,
            "policy": dict(self.policy),
            "diagnostics": dict(self.diagnostics),
            "version": self.version,
        }
