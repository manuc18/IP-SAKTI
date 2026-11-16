"""The retrieval contract every IP-SAKTI retriever answers to.

Two things are non-negotiable here, and both exist to stop a downstream answer
engine from making a promise the corpus cannot keep.

**Provenance survives retrieval.** A result is not a string with a score. It
carries the chunk's document, locator, physical pages, source evidence level and
temporal status, because the moment a passage leaves the index those are the
only facts that let a reader check it. A retriever that returned bare text would
force the answer layer to look them up again, and a lookup that can be skipped
eventually is.

**Citability is reported, never decided.** `citable` is copied from the chunk
exactly as Phase 3 set it. No retriever may set, clear, or infer it. A chunk
whose locator could not be established stays retrievable — hiding statutory text
from search helps nobody — but it arrives labelled `citable: false` so the caller
can refuse to quote it. See `scripts/ingest/provenance.py` for why the two are
separate axes.

Filtering is defined once, in :class:`Filters`, rather than inside any one
retriever. BM25 and a dense index disagree about almost everything; they must not
also disagree about what `domain="patents"` selects.
"""
from __future__ import annotations

import glob
import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHUNKS_DIR = os.path.join(PROJECT_ROOT, "data", "processed", "chunks")

#: Bumped whenever a change here would alter a stored ranking. Recorded in every
#: benchmark run so a results file can never be silently misattributed.
RETRIEVAL_VERSION = "0.1.0"


@dataclass(frozen=True)
class Chunk:
    """One retrieval unit, as produced by the Phase 2 chunker."""

    chunk_id: str
    document_id: str
    text: str
    citation: str
    locator: dict[str, Any]
    locator_unavailable: bool
    citable: bool
    retrievable: bool
    domain: str
    jurisdiction: str
    document_type: str
    temporal_status: str
    source_evidence_level: str
    source_status: str
    physical_pages: list[int]
    title: str
    short_title: str | None
    heading: str | None
    unit_type: str
    structural_path: str
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "Chunk":
        return cls(
            chunk_id=row["chunk_id"],
            document_id=row["document_id"],
            text=row["text"],
            citation=row["citation"],
            locator=row["locator"],
            locator_unavailable=row["locator_unavailable"],
            citable=row["citable"],
            retrievable=row["retrievable"],
            domain=row["domain"],
            jurisdiction=row["jurisdiction"],
            document_type=row["document_type"],
            temporal_status=row.get("temporal_status", "UNKNOWN"),
            source_evidence_level=row["source_evidence_level"],
            source_status=row["source_status"],
            physical_pages=row["physical_pages"],
            title=row["title"],
            short_title=row.get("short_title"),
            heading=row.get("heading"),
            unit_type=row["unit_type"],
            structural_path=row.get("structural_path", ""),
            raw=row,
        )


@dataclass(frozen=True)
class RetrievalResult:
    """One ranked hit. The shape of :meth:`to_dict` is the published contract."""

    chunk_id: str
    document_id: str
    score: float
    rank: int
    text: str
    locator: str
    citable: bool
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "score": self.score,
            "rank": self.rank,
            "text": self.text,
            "locator": self.locator,
            "citable": self.citable,
            "metadata": self.metadata,
        }


def result_from_chunk(chunk: Chunk, *, score: float, rank: int,
                      extra: dict[str, Any] | None = None) -> RetrievalResult:
    """Build a result from a chunk, copying `citable` rather than deciding it."""
    metadata = {
        "domain": chunk.domain,
        "jurisdiction": chunk.jurisdiction,
        "document_type": chunk.document_type,
        "temporal_status": chunk.temporal_status,
        "source_evidence_level": chunk.source_evidence_level,
        "source_status": chunk.source_status,
        "citability_reason": chunk.raw.get("citability_reason"),
        "locator_unavailable": chunk.locator_unavailable,
        # The structured locator, alongside the rendered citation string in
        # `RetrievalResult.locator`. Phase 6 reranking needs the fields, not the
        # prose: deciding whether a candidate is the exact provision a query
        # named, its parent, or a sibling sub-unit is a comparison of locator
        # keys, and re-parsing them out of "Patents Act, 1970 - Section 3(p)"
        # would be inventing structure that was already established in Phase 2.
        "locator_fields": dict(chunk.locator),
        "physical_pages": list(chunk.physical_pages),
        "title": chunk.title,
        "short_title": chunk.short_title,
        "heading": chunk.heading,
        "unit_type": chunk.unit_type,
        "structural_path": chunk.structural_path,
        "official_source_url": chunk.raw.get("official_source_url"),
        "consolidated_as_of": chunk.raw.get("consolidated_as_of"),
        "source_version_id": chunk.raw.get("source_version_id"),
        # The identity of the PDF this text came out of. Phase 7 §9 requires a
        # citation to be machine-resolvable back to a source document and page,
        # and a filename that has to be looked up again somewhere else is a
        # lookup that will eventually be skipped — the same argument this
        # module's docstring makes about provenance generally.
        "source_filename": chunk.raw.get("source_filename"),
        "source_sha256": chunk.raw.get("source_sha256"),
    }
    if extra:
        metadata.update(extra)
    return RetrievalResult(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        score=float(score),
        rank=rank,
        text=chunk.text,
        locator=chunk.citation,
        citable=chunk.citable,
        metadata=metadata,
    )


def _as_set(value: str | Iterable[str] | None) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return frozenset({value})
    return frozenset(value)


@dataclass(frozen=True)
class Filters:
    """Metadata restrictions, applied identically by every retriever.

    Each field is ``None`` (no restriction), one value, or several. `citable`
    is a tri-state: ``None`` admits everything, ``True`` restricts to chunks a
    caller may quote, ``False`` isolates the ones it may not.

    There is deliberately no `document_id` field. Filtering an evaluation query
    down to the document the answer lives in is not retrieval, it is the answer;
    the benchmark's leakage rules (Phase 4 §12) depend on that door staying shut.
    """

    domain: str | Iterable[str] | None = None
    jurisdiction: str | Iterable[str] | None = None
    document_type: str | Iterable[str] | None = None
    temporal_status: str | Iterable[str] | None = None
    citable: bool | None = None

    def is_empty(self) -> bool:
        return all(getattr(self, f) is None for f in
                   ("domain", "jurisdiction", "document_type", "temporal_status", "citable"))

    def matches(self, chunk: Chunk) -> bool:
        for field_name in ("domain", "jurisdiction", "document_type", "temporal_status"):
            allowed = _as_set(getattr(self, field_name))
            if allowed is not None and getattr(chunk, field_name) not in allowed:
                return False
        if self.citable is not None and chunk.citable is not self.citable:
            return False
        return True

    def describe(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in
                ("domain", "jurisdiction", "document_type", "temporal_status", "citable")
                if getattr(self, f) is not None}


def apply_filters(chunks: Sequence[Chunk], filters: Filters | None) -> list[Chunk]:
    if filters is None or filters.is_empty():
        return list(chunks)
    return [chunk for chunk in chunks if filters.matches(chunk)]


def allowed_ids(chunks: Sequence[Chunk], filters: Filters | None) -> set[str] | None:
    """The chunk ids a filter admits, or ``None`` when nothing is restricted."""
    if filters is None or filters.is_empty():
        return None
    return {chunk.chunk_id for chunk in chunks if filters.matches(chunk)}


def load_chunks(chunks_dir: str = CHUNKS_DIR) -> list[Chunk]:
    """Every retrieval chunk, ordered by chunk_id so indexes are reproducible."""
    rows: list[dict[str, Any]] = []
    for path in sorted(glob.glob(os.path.join(chunks_dir, "*.jsonl"))):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    rows.sort(key=lambda row: row["chunk_id"])
    return [Chunk.from_row(row) for row in rows]


class Retriever(ABC):
    """A ranked-search backend over the Phase 2 chunks."""

    #: Stable identifier used in results files and reports.
    name: str = "retriever"
    #: Bumped when the backend's ranking behaviour changes.
    version: str = RETRIEVAL_VERSION

    @abstractmethod
    def search(self, query: str, filters: Filters | None = None,
               top_k: int = 10) -> list[RetrievalResult]:
        """Return at most ``top_k`` results, best first, ranks starting at 1."""

    def config(self) -> dict[str, Any]:
        """Everything needed to reproduce this retriever's rankings."""
        return {"retriever": self.name, "version": self.version}
