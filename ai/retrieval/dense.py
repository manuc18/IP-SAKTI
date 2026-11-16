"""Dense retrieval with BGE-M3, run locally and cached against the corpus.

Everything is local: the model is loaded from the Hugging Face cache and run on
this machine. No embedding API sees the corpus.

The cache is keyed by a fingerprint of what was actually embedded — every
chunk_id paired with a digest of its text, plus the model name, normalisation
setting and sequence limit. Change the chunker and the fingerprint changes, so a
stale index can never be silently reused; change nothing and the 3,031 forward
passes never run twice. That is stricter than a version string, which someone
eventually forgets to bump.

**Citability does not touch the vectors.** A chunk whose locator Phase 2 could
not establish is embedded exactly like any other. Whether a passage may be
quoted is a property of our evidence about it, not of what it says, and
distorting the geometry to encode that would corrupt the one thing the dense
index is for. The flag rides along in the result, where the caller can act on it.

Storage is a plain `.npy` matrix and a JSON sidecar. pgvector is not configured
in this project, and introducing a database to hold 3,031 vectors would be
infrastructure work masquerading as progress. The retriever's contract is the
same one pgvector would satisfy, so moving later is a change of backend.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Sequence

from ai.retrieval.base import (
    PROJECT_ROOT,
    Chunk,
    Filters,
    RetrievalResult,
    Retriever,
    allowed_ids,
    load_chunks,
    result_from_chunk,
)

MODEL_NAME = "BAAI/bge-m3"
EMBEDDING_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation", "embeddings")

#: BGE-M3 accepts 8192 tokens; the longest chunk here is under 5,000 characters,
#: so 1024 truncates nothing and costs a fraction of the compute. Recorded in the
#: sidecar because changing it changes the vectors.
MAX_SEQ_LENGTH = 1024
BATCH_SIZE = 8
NORMALIZE = True


def _device() -> str:
    override = os.environ.get("IPSAKTI_DENSE_DEVICE")
    if override:
        return override
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # pragma: no cover - torch absent
        pass
    return "cpu"


def corpus_fingerprint(chunks: Sequence[Chunk]) -> str:
    """Identity of the exact text being embedded."""
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(hashlib.sha256(chunk.text.encode("utf-8")).digest())
    return digest.hexdigest()[:16]


def corpus_version(chunks: Sequence[Chunk]) -> str:
    """Identity of the source PDFs behind those chunks."""
    sources = sorted({(c.document_id, c.raw.get("source_sha256", "")) for c in chunks})
    digest = hashlib.sha256()
    for document_id, sha in sources:
        digest.update(f"{document_id}:{sha}".encode("utf-8"))
    return digest.hexdigest()[:16]


class DenseRetriever(Retriever):
    name = "dense"
    version = "0.1.0"

    def __init__(self, chunks: Sequence[Chunk] | None = None, *,
                 model_name: str = MODEL_NAME,
                 max_seq_length: int = MAX_SEQ_LENGTH,
                 batch_size: int = BATCH_SIZE,
                 normalize: bool = NORMALIZE,
                 embedding_dir: str = EMBEDDING_DIR):
        self.chunks: list[Chunk] = list(chunks) if chunks is not None else load_chunks()
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.normalize = normalize
        self.embedding_dir = embedding_dir
        self.index = {c.chunk_id: i for i, c in enumerate(self.chunks)}
        self._model = None
        self._query_cache: dict[str, Any] = {}
        self.fingerprint = corpus_fingerprint(self.chunks)
        self.embeddings, self.meta = self._load_or_build()

    # ------------------------------------------------------------ embedding ---

    def _paths(self) -> tuple[str, str]:
        stem = f"{self.model_name.split('/')[-1]}-{self.fingerprint}"
        return (os.path.join(self.embedding_dir, stem + ".npy"),
                os.path.join(self.embedding_dir, stem + ".json"))

    def model(self):
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            torch.manual_seed(0)
            model = SentenceTransformer(self.model_name, device=_device())
            model.max_seq_length = self.max_seq_length
            model.eval()
            self._model = model
        return self._model

    def _encode(self, texts: Sequence[str]):
        import torch

        with torch.inference_mode():
            return self.model().encode(
                list(texts),
                batch_size=self.batch_size,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
                show_progress_bar=False,
            ).astype("float32")

    def _load_or_build(self):
        import numpy as np

        vectors_path, meta_path = self._paths()
        if os.path.exists(vectors_path) and os.path.exists(meta_path):
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
            if meta.get("chunk_ids") == [c.chunk_id for c in self.chunks]:
                return np.load(vectors_path), meta

        os.makedirs(self.embedding_dir, exist_ok=True)
        vectors = self._encode([c.text for c in self.chunks])
        meta = {
            "model_name": self.model_name,
            "model_version": self._model_revision(),
            "normalize_embeddings": self.normalize,
            "embedding_dimension": int(vectors.shape[1]),
            "max_seq_length": self.max_seq_length,
            "batch_size": self.batch_size,
            "device": _device(),
            "dtype": "float32",
            "corpus_version": corpus_version(self.chunks),
            "chunking_version": self.fingerprint,
            "chunk_count": len(self.chunks),
            "chunk_ids": [c.chunk_id for c in self.chunks],
        }
        np.save(vectors_path, vectors)
        with open(meta_path, "w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2)
        return vectors, meta

    def _model_revision(self) -> str | None:
        """The model's commit hash, when the local cache records one."""
        try:
            config = self.model()[0].auto_model.config
            return getattr(config, "_commit_hash", None)
        except Exception:
            return None

    def embed_query(self, query: str):
        if query not in self._query_cache:
            self._query_cache[query] = self._encode([query])[0]
        return self._query_cache[query]

    # -------------------------------------------------------------- search ---

    def rank(self, query: str, filters: Filters | None = None,
             top_k: int = 10) -> list[tuple[Chunk, float]]:
        import numpy as np

        # Vectors are L2-normalised, so the dot product is cosine similarity.
        similarities = self.embeddings @ self.embed_query(query)
        permitted = allowed_ids(self.chunks, filters)
        scored = [
            (chunk.chunk_id, float(similarities[position]))
            for position, chunk in enumerate(self.chunks)
            if permitted is None or chunk.chunk_id in permitted
        ]
        ordered = sorted(scored, key=lambda item: (-item[1], item[0]))[:top_k]
        return [(self.chunks[self.index[cid]], score) for cid, score in ordered]

    def search(self, query: str, filters: Filters | None = None,
               top_k: int = 10) -> list[RetrievalResult]:
        return [
            result_from_chunk(chunk, score=score, rank=rank)
            for rank, (chunk, score) in enumerate(self.rank(query, filters, top_k), start=1)
        ]

    def config(self) -> dict[str, Any]:
        config = {"retriever": self.name, "version": self.version}
        config.update({k: v for k, v in self.meta.items() if k != "chunk_ids"})
        return config
