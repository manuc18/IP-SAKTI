"""Okapi BM25 over the Phase 2 retrieval chunks — the lexical baseline.

Deliberately plain. The point of a baseline is to be the thing every later
result is measured against, so there is nothing clever in it: standard Okapi
weighting, the legal-aware tokenizer from `tokenizer.py`, and no stemming.

Two choices are worth defending.

*Only `text` is indexed.* Not `raw_text` (it still holds the corrupt-Hindi lines
Phase 2 excluded), not the heading, not the citation. Folding the citation into
the indexed field would quietly let "Patents Act" match every one of its 197
chunks equally, which flatters BM25 on exactly the queries the identifier boost
is supposed to be measured on. Keeping the field pure keeps the comparison in
`identifiers.py` honest.

*IDF is computed over the whole corpus, never over the filtered subset.* A term
is rare, or is not, as a fact about the corpus; recomputing it per filter would
make a document's score depend on which other documents the caller happened to
allow, and two runs with different filters would stop being comparable.
"""
from __future__ import annotations

import math
from collections import Counter
from typing import Any, Sequence

from ai.retrieval.base import (
    Chunk,
    Filters,
    RetrievalResult,
    Retriever,
    allowed_ids,
    load_chunks,
    result_from_chunk,
)
from ai.retrieval.tokenizer import tokenize

K1 = 1.2
B = 0.75


class BM25Retriever(Retriever):
    name = "bm25"
    version = "0.1.0"

    def __init__(self, chunks: Sequence[Chunk] | None = None, *, k1: float = K1, b: float = B):
        self.chunks: list[Chunk] = list(chunks) if chunks is not None else load_chunks()
        self.k1 = k1
        self.b = b
        self._build()

    def _build(self) -> None:
        self.index: dict[str, int] = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        self.doc_tokens: list[Counter[str]] = []
        self.doc_len: list[int] = []
        postings: dict[str, list[tuple[int, int]]] = {}
        for position, chunk in enumerate(self.chunks):
            counts = Counter(tokenize(chunk.text))
            self.doc_tokens.append(counts)
            self.doc_len.append(sum(counts.values()))
            for term, frequency in counts.items():
                postings.setdefault(term, []).append((position, frequency))
        self.postings = postings
        total = len(self.chunks)
        self.avg_len = (sum(self.doc_len) / total) if total else 0.0
        # Robertson/Sparck-Jones IDF with the +1 that keeps it non-negative.
        self.idf = {
            term: math.log(1.0 + (total - len(plist) + 0.5) / (len(plist) + 0.5))
            for term, plist in postings.items()
        }

    def score_all(self, query: str) -> dict[str, float]:
        """Raw BM25 score for every chunk the query touches, keyed by chunk_id."""
        scores: dict[int, float] = {}
        for term, query_frequency in Counter(tokenize(query)).items():
            plist = self.postings.get(term)
            if not plist:
                continue
            idf = self.idf[term]
            for position, frequency in plist:
                norm = 1.0 - self.b + self.b * (self.doc_len[position] / self.avg_len
                                                if self.avg_len else 0.0)
                contribution = idf * (frequency * (self.k1 + 1.0)) / (frequency + self.k1 * norm)
                scores[position] = scores.get(position, 0.0) + contribution * query_frequency
        return {self.chunks[position].chunk_id: score for position, score in scores.items()}

    def rank(self, query: str, filters: Filters | None = None,
             top_k: int = 10) -> list[tuple[Chunk, float]]:
        scores = self.score_all(query)
        permitted = allowed_ids(self.chunks, filters)
        if permitted is not None:
            scores = {cid: s for cid, s in scores.items() if cid in permitted}
        # chunk_id breaks ties so a ranking is a function of the query alone.
        ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]
        return [(self.chunks[self.index[cid]], score) for cid, score in ordered]

    def search(self, query: str, filters: Filters | None = None,
               top_k: int = 10) -> list[RetrievalResult]:
        return [
            result_from_chunk(chunk, score=score, rank=rank)
            for rank, (chunk, score) in enumerate(self.rank(query, filters, top_k), start=1)
        ]

    def config(self) -> dict[str, Any]:
        return {
            "retriever": self.name,
            "version": self.version,
            "k1": self.k1,
            "b": self.b,
            "indexed_field": "text",
            "stemming": False,
            "stopwords": False,
            "documents_indexed": len(self.chunks),
            "vocabulary": len(self.postings),
            "mean_document_length_tokens": round(self.avg_len, 2),
        }
