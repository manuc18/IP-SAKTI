"""Cross-encoder reranking, run locally, with the model's identity recorded.

A cross-encoder reads the query and the passage together and scores the pair, so
it can attend to the words of one against the words of the other. That is the
capability a bi-encoder gives up in exchange for being able to precompute an
index, and it is why reranking a shortlist is worth doing at all.

**No model is used without its identity being written down.** Phase 6 §8 asks
for the model name, its revision, the device, the sequence length, the batch
size and the truncation strategy, and :meth:`CrossEncoderReranker.config`
records every one of them plus the resolved commit hash from the local cache and
the on-disk size. A benchmark that says "cross-encoder: 0.81" and cannot say
which weights produced it is not reproducible, and legal evaluation is not a
place to find that out later.

**Why these candidates, and why not the popular ones.** :data:`MODELS` carries
the shortlist with the reason each entry is on it. Two rejections are recorded
there rather than left silent: `jinaai/jina-reranker-v2-base-multilingual` is
CC-BY-NC, which is not a licence this project can build on, and
`Alibaba-NLP/gte-multilingual-reranker-base` requires `trust_remote_code=True`,
which means executing code fetched from a model repository on every load. Both
are competitive models. Neither is excluded for being bad.

**Truncation is measured, not assumed.** The candidate text is not shortened
before tokenization; the tokenizer truncates, and this module counts how often
it did and by how much, because a reranker that silently drops the operative
half of a long provision would score it on its preamble and nothing in the
metrics would say so.
"""
from __future__ import annotations

import os

from dataclasses import dataclass
from typing import Any, Sequence

from ai.reranking.base import Reranker, RerankedResult, order_by_score
from ai.reranking.format import format_candidate, format_config
from ai.retrieval.candidates import Candidate


@dataclass(frozen=True)
class ModelChoice:
    """A shortlisted cross-encoder, with why it is on the list."""

    key: str
    model_name: str
    license: str
    parameters: str
    languages: str
    max_length: int
    note: str


#: The shortlist actually benchmarked. Ordered small to large, which is also the
#: order of increasing cost per query.
MODELS: dict[str, ModelChoice] = {
    "minilm": ModelChoice(
        key="minilm",
        model_name="cross-encoder/ms-marco-MiniLM-L-6-v2",
        license="apache-2.0",
        parameters="22.7M",
        languages="English",
        max_length=512,
        note="The cheap end of the trade-off, and the honest control on model "
             "size: 22M parameters trained on MS MARCO web passages. If a model "
             "this small closes most of the gap, a 568M model is hard to justify "
             "for a local deployment.",
    ),
    "bge-base": ModelChoice(
        key="bge-base",
        model_name="BAAI/bge-reranker-base",
        license="mit",
        parameters="278M",
        languages="multilingual (XLM-RoBERTa base)",
        max_length=512,
        note="The middle of the range, and multilingual, which matters for a "
             "corpus whose questions will eventually arrive in Hindi.",
    ),
    "bge-m3": ModelChoice(
        key="bge-m3",
        model_name="BAAI/bge-reranker-v2-m3",
        license="apache-2.0",
        parameters="568M",
        languages="multilingual (XLM-RoBERTa large)",
        max_length=512,
        note="The natural pair for the BGE-M3 retriever already in use: same "
             "family, same tokenizer lineage, trained for exactly this reranking "
             "step. Included because of that relationship, not its popularity.",
    ),
}

#: Considered and rejected, with the reason. Kept in the code so the shortlist
#: cannot quietly become "whatever was tried first".
REJECTED: tuple[dict[str, str], ...] = (
    {"model": "jinaai/jina-reranker-v2-base-multilingual",
     "reason": "CC-BY-NC-4.0. Non-commercial licences are not a base this "
               "project can build on, independent of how the model performs."},
    {"model": "Alibaba-NLP/gte-multilingual-reranker-base",
     "reason": "Requires trust_remote_code=True, which executes code fetched "
               "from a model repository at load time. Not acceptable in a "
               "pipeline that handles a provenance-controlled corpus."},
    {"model": "any hosted reranking API (Cohere Rerank, Voyage, Jina API)",
     "reason": "Phase 6 forbids paid external APIs, and sending the corpus to a "
               "third party would breach the same rule that keeps embedding "
               "local."},
)

DEFAULT_MODEL = "bge-m3"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 16
#: `longest_first` truncates whichever member of the (query, passage) pair is
#: longer. The passage is longer in essentially every case here — queries run to
#: about 15 tokens — so in practice it is the tail of the provision that goes.
TRUNCATION = "longest_first"


def _device() -> str:
    override = os.environ.get("IPSAKTI_RERANK_DEVICE")
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


def _cache_size_bytes(model_name: str) -> int | None:
    """On-disk size of the model in the local Hugging Face cache."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except Exception:  # pragma: no cover
        return None
    folder = os.path.join(HF_HUB_CACHE, "models--" + model_name.replace("/", "--"))
    if not os.path.isdir(folder):
        return None
    total = 0
    for root, _, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            # The cache stores each file once under `blobs/` and links to it
            # from every `snapshots/<revision>/`. Following those links counts
            # the same bytes twice: this reported 4,586.5 MB for
            # `bge-reranker-v2-m3`, whose blobs are 2,293.3 MB.
            if os.path.islink(path):
                continue
            try:
                total += os.stat(path, follow_symlinks=False).st_size
            except OSError:
                continue
    return total


@dataclass
class TruncationStats:
    """How much of the corpus the model was actually allowed to read."""

    pairs: int = 0
    truncated: int = 0
    max_tokens_seen: int = 0
    dropped_tokens: int = 0

    def record(self, length: int, limit: int) -> None:
        self.pairs += 1
        self.max_tokens_seen = max(self.max_tokens_seen, length)
        if length > limit:
            self.truncated += 1
            self.dropped_tokens += length - limit

    def to_dict(self) -> dict[str, Any]:
        return {
            "pairs_scored": self.pairs,
            "pairs_truncated": self.truncated,
            "truncated_fraction": round(self.truncated / self.pairs, 4) if self.pairs else 0.0,
            "longest_pair_tokens": self.max_tokens_seen,
            "tokens_dropped_total": self.dropped_tokens,
        }


class CrossEncoderReranker(Reranker):
    """A local sequence-classification cross-encoder over (query, passage) pairs.

    Loaded through `transformers` rather than a wrapper so that the tokenizer is
    reachable: the truncation counts in :class:`TruncationStats` need the
    untruncated token length, which a `.predict()` convenience API does not
    expose.
    """

    version = "0.1.0"

    def __init__(self, model_key: str = DEFAULT_MODEL, *,
                 max_length: int | None = None,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 device: str | None = None,
                 text_budget: int | None = None):
        if model_key not in MODELS:
            raise ValueError(f"unknown cross-encoder: {model_key}; "
                             f"choose from {sorted(MODELS)}")
        self.choice = MODELS[model_key]
        self.name = f"cross_encoder:{model_key}"
        self.model_key = model_key
        self.model_name = self.choice.model_name
        self.max_length = max_length or self.choice.max_length
        self.batch_size = batch_size
        self.device = device or _device()
        self.text_budget = text_budget
        self.truncation = TruncationStats()
        self._model = None
        self._tokenizer = None

    # --------------------------------------------------------------- model ---

    def load(self):
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            torch.manual_seed(0)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            model.eval()
            model.to(self.device)
            self._model = model
        return self._model, self._tokenizer

    def warm(self) -> None:
        """One throwaway pair, so a latency measurement is not a load time."""
        self.score_pairs(["warm-up"], ["Document: warm-up\nLocator: n/a\nText: warm-up"])

    @property
    def revision(self) -> str | None:
        model, _ = self.load()
        return getattr(model.config, "_commit_hash", None)

    # -------------------------------------------------------------- scoring ---

    def score_pairs(self, queries: Sequence[str], passages: Sequence[str]) -> list[float]:
        """Relevance logits for aligned (query, passage) pairs."""
        import torch

        model, tokenizer = self.load()
        scores: list[float] = []
        for start in range(0, len(passages), self.batch_size):
            batch_q = list(queries[start:start + self.batch_size])
            batch_p = list(passages[start:start + self.batch_size])
            # Measured before truncation, so the count is of what was lost.
            for query, passage in zip(batch_q, batch_p):
                full = tokenizer(query, passage, truncation=False)["input_ids"]
                self.truncation.record(len(full), self.max_length)
            encoded = tokenizer(batch_q, batch_p, padding=True, truncation=TRUNCATION,
                                max_length=self.max_length, return_tensors="pt")
            encoded = {k: v.to(self.device) for k, v in encoded.items()}
            with torch.inference_mode():
                logits = model(**encoded).logits.float().cpu()
            # Single-logit rerankers emit one relevance score; a two-way
            # classifier's relevant class is index 1.
            column = logits[:, 0] if logits.shape[-1] == 1 else logits[:, -1]
            scores.extend(float(value) for value in column)
        return scores

    def rerank(self, query: str, candidates: Sequence[Candidate],
               top_k: int = 10) -> list[RerankedResult]:
        if not candidates:
            return []
        passages = [format_candidate(c, text_budget=self.text_budget) for c in candidates]
        scores = self.score_pairs([query] * len(passages), passages)
        payload = [
            (candidate, score, {"cross_encoder_logit": score,
                                "candidate_order": float(candidate.order)})
            for candidate, score in zip(candidates, scores)
        ]
        return order_by_score(payload, top_k)

    def config(self) -> dict[str, Any]:
        size = _cache_size_bytes(self.model_name)
        return {
            "reranker": self.name,
            "version": self.version,
            "trained_by_us": False,
            "model_name": self.model_name,
            "model_revision": self.revision,
            "license": self.choice.license,
            "parameters": self.choice.parameters,
            "languages": self.choice.languages,
            "selection_note": self.choice.note,
            "device": self.device,
            "max_seq_length": self.max_length,
            "batch_size": self.batch_size,
            "truncation_strategy": TRUNCATION,
            "truncation_note": "the pair is truncated, longest member first; the "
                               "passage is longer than the query in every "
                               "observed case, so the provision's tail is what "
                               "is dropped",
            "activation": "none — raw relevance logit; ranking is invariant to a "
                          "monotone squashing, so none is applied",
            "candidate_format": format_config(self.text_budget),
            "model_disk_bytes": size,
            "model_disk_mb": round(size / 1e6, 1) if size else None,
            "rejected_alternatives": [dict(item) for item in REJECTED],
            "truncation_observed": self.truncation.to_dict(),
        }


def resource_report(reranker: CrossEncoderReranker) -> dict[str, Any]:
    """Phase 6 §20: what this model costs to keep and to run."""
    import torch

    model, _ = reranker.load()
    parameters = sum(p.numel() for p in model.parameters())
    bytes_in_memory = sum(p.numel() * p.element_size() for p in model.parameters())
    report = {
        "model_name": reranker.model_name,
        "parameters_counted": parameters,
        "weights_bytes": bytes_in_memory,
        "weights_mb": round(bytes_in_memory / 1e6, 1),
        "dtype": str(next(model.parameters()).dtype),
        "device": reranker.device,
        "batch_size": reranker.batch_size,
        "max_seq_length": reranker.max_length,
        "model_disk_mb": (lambda s: round(s / 1e6, 1) if s else None)(
            _cache_size_bytes(reranker.model_name)),
    }
    if reranker.device == "mps" and hasattr(torch, "mps"):
        try:
            report["mps_current_allocated_mb"] = round(
                torch.mps.current_allocated_memory() / 1e6, 1)
        except Exception:
            pass
    return report
