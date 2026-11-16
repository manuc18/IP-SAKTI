"""Answer providers: the one place a model could ever be plugged in.

Phase 7 §3 and §4 set the shape. A provider receives a query, the selected
passages and the answer instructions, and returns prose. It receives nothing
else — not the corpus, not the retriever, not the golden dataset — and it
returns nothing else: no metadata, no page numbers, no document names that
become citations. Everything a citation is made of comes from the evidence
records, assembled in :mod:`ai.answering.citation_validator`.

That boundary is what makes the ₹0 requirement survivable. A provider is
replaceable — a local model, a hosted API, or the deterministic one below —
without touching retrieval, policy, validation or rendering, and the system runs
today with no external service, no API key and no network call.

The mock provider is not a placeholder. It is extractive: it quotes sentences
that the corpus already contains and marks each with the passage it came from.
That makes it a useful control for exactly the reason Phase 6's identity
reranker was one — every downstream number it produces is attributable to the
pipeline rather than to a model's fluency — and it makes the validator's tests
meaningful, because an extractive provider is *supposed* to be perfectly
grounded and any failure is a real bug rather than a hallucination.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ai.answering.grounding import split_sentences
from ai.answering.models import Evidence, EvidenceSet
from ai.answering.policy import AnswerPolicy, PolicyReport, content_terms

PROVIDER_VERSION = "0.1.0"

#: Versioned, because a prompt is part of a result's identity in exactly the way
#: a model revision is. Phase 7 §22; the prose lives in
#: docs/ANSWER_GENERATION_CONTRACT.md and this is the machine-readable copy.
#:
#: 0.2.0 — Phase 7.4. Rule 4 was "cite every claim", which the model read as
#: "cite the sentences that carry the specifics". Rule 4 was made specific about
#: which sentence kinds count, and a worked example was added.
#:
#: 0.3.0 — Phase 7.5 tested 0.2.0 on Q027 and it did not work. The model cited
#: six of nine sentences and left uncited exactly the three kinds rule 4 had
#: just finished enumerating: an opening topic sentence, a "Thus, Rule 23
#: specifies…" summary, and a comparative aside. Naming the categories did not
#: change the behaviour, so 0.3.0 stops asking for more citations and asks for
#: fewer sentences instead. An answer of one to four evidence-bearing sentences
#: has no room for a topic sentence or a summary, which removes the failure
#: rather than forbidding it — the model has to *want* to write the uncited
#: sentence, and a four-sentence budget spent on the answer leaves no room to.
#:
#: The alternative was to teach `check_grounding` that opening and closing
#: sentences do not need citations. That is the permissive fix: it would publish
#: the same answer while knowing less about it, and every future uncited claim
#: that happened to sit first or last would pass. The gate is correct. The
#: output shape was wrong.
PROMPT_VERSION = "0.3.0"

ANSWER_INSTRUCTIONS = (
    "1. Answer only from the supplied evidence. Do not use outside knowledge, "
    "including your own knowledge of the law.",
    "2. Answer directly. Write no introduction, no framing sentence, no "
    "conclusion, no summary and no preamble such as \"Based on the supplied "
    "evidence:\". Begin with the answer and stop when it is finished.",
    "3. Write 1 to 4 sentences. Use more only if the question genuinely "
    "requires them.",
    "4. Every sentence must state something the supplied evidence directly "
    "supports, and every sentence must carry a citation marker written as [1] "
    "or [2]. Do not write a sentence you cannot support from the evidence you "
    "were given.",
    "5. Put the marker immediately before the sentence's final punctuation, and "
    "repeat it on every sentence it supports rather than citing once for a "
    "paragraph.",
    "6. GOOD: \"Rule 23 governs the filing of trademark applications [1]. It "
    "requires the prescribed TM-A form and the applicant's signature [1].\"",
    "7. BAD: \"Rule 23 is an important provision concerning trademark "
    "applications. The rule governs filing. It requires TM-A [1].\" The first "
    "two sentences carry no marker, so the whole answer is rejected and nothing "
    "is shown to the reader.",
    "8. If the supplied evidence does not answer the question, say so and stop. "
    "Do not fill the gap.",
    "9. Distinguish current evidence from evidence marked STALE, and "
    "authoritative evidence from evidence with weak provenance. Do not silently "
    "reconcile conflicting provisions; state that they differ.",
    "10. Never invent a citation, and never write a page number, locator, URL "
    "or document name of your own; the citation layer supplies all of them from "
    "the evidence record.",
)


@dataclass(frozen=True)
class Passage:
    """One passage as a provider sees it.

    Note what is absent: `chunk_id`. A provider that never receives a chunk id
    cannot emit one, which removes an entire class of fabricated citation before
    any validation has to catch it. The marker is the only handle it gets.
    """

    marker: int
    document: str
    locator: str
    temporal_status: str
    source_evidence_level: str
    citable: bool
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {"marker": self.marker, "document": self.document,
                "locator": self.locator, "temporal_status": self.temporal_status,
                "source_evidence_level": self.source_evidence_level,
                "citable": self.citable, "text": self.text}


@dataclass(frozen=True)
class ProviderRequest:
    """Everything a provider is permitted to see."""

    query: str
    passages: tuple[Passage, ...]
    instructions: tuple[str, ...] = ANSWER_INSTRUCTIONS
    prompt_version: str = PROMPT_VERSION
    policy: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query,
                "passages": [p.to_dict() for p in self.passages],
                "instructions": list(self.instructions),
                "prompt_version": self.prompt_version,
                "policy": dict(self.policy)}


@dataclass(frozen=True)
class ProviderDraft:
    """What a provider returns: prose, and the name of who wrote it."""

    text: str
    provider: str
    provider_version: str = PROVIDER_VERSION
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


def build_request(query: str, evidence_set: EvidenceSet,
                  policy: AnswerPolicy,
                  report: PolicyReport | None = None) -> ProviderRequest:
    """Assemble the provider's view of the evidence, and audit it.

    The audit is cheap and worth keeping: it asserts that no chunk id reached
    the payload. Phase 6 learned that a leakage rule which is only structural
    gets violated the first time someone adds a convenience field, so this is
    checked on every call rather than only in a test.
    """
    admissible = {a.marker for a in (report.assessments if report else ())
                  if a.addresses_query} if report else None
    passages: list[Passage] = []
    for item in evidence_set:
        if admissible is not None and item.marker not in admissible:
            continue
        passages.append(Passage(
            marker=item.marker,
            document=item.document_name,
            locator=item.locator_text or "(no locator established)",
            temporal_status=item.temporal_status,
            source_evidence_level=item.source_evidence_level,
            citable=item.citable,
            text=item.text,
        ))

    request = ProviderRequest(query=query, passages=tuple(passages),
                              policy=policy.to_dict())
    rendered = str(request.to_dict())
    for item in evidence_set:
        if item.chunk_id in rendered:
            raise ValueError(
                f"provider payload leaked a chunk id ({item.chunk_id}); a "
                f"provider must be able to cite only by marker")
    return request


class AnswerProvider(ABC):
    """Turns a query and selected passages into prose. Nothing else."""

    name: str = "provider"
    version: str = PROVIDER_VERSION

    @abstractmethod
    def generate(self, request: ProviderRequest) -> ProviderDraft:
        """Return prose citing supplied markers, or empty text to abstain."""

    def config(self) -> dict[str, Any]:
        return {"provider": self.name, "version": self.version,
                "prompt_version": PROMPT_VERSION, "requires_network": False}


class ExtractiveMockProvider(AnswerProvider):
    """Deterministic, offline, and grounded by construction.

    Picks the sentence from each passage that shares most content terms with the
    query, keeps the best few, and marks each with its passage. It composes no
    prose of its own — which is the point. It cannot hallucinate because it never
    writes a sentence the corpus did not already contain.

    It is a baseline, not an answer generator: it does not synthesise across
    provisions, resolve a question the corpus answers only implicitly, or phrase
    anything for a reader. Those are the jobs a real provider will do, and the
    reason §21 describes one.
    """

    name = "extractive_mock"
    version = PROVIDER_VERSION

    def __init__(self, *, max_sentences: int = 3, prefer_citable: bool = True):
        self.max_sentences = max_sentences
        self.prefer_citable = prefer_citable

    def generate(self, request: ProviderRequest) -> ProviderDraft:
        query_terms = content_terms(request.query)
        scored: list[tuple[int, int, int, str]] = []
        for passage in request.passages:
            if self.prefer_citable and not passage.citable:
                continue
            best_sentence, best_overlap = "", 0
            for sentence in split_sentences(passage.text):
                overlap = len(query_terms & content_terms(sentence))
                if overlap > best_overlap:
                    best_sentence, best_overlap = sentence, overlap
            if best_overlap:
                scored.append((-best_overlap, passage.marker, passage.marker,
                               best_sentence))

        scored.sort()
        chosen = scored[:self.max_sentences]
        if not chosen:
            return ProviderDraft(text="", provider=self.name,
                                 provider_version=self.version,
                                 diagnostics={"reason": "no passage shared a "
                                                        "content term with the query"})

        # The marker goes on every sentence of the quoted span, not just the
        # last. A statutory sentence often survives `split_sentences` as one
        # unit and then splits into several once it is joined into an answer —
        # and a fragment that lost its marker on the way is reported as an
        # uncited claim, correctly, by the grounding check. Marking each
        # sentence keeps an extractive provider grounded by construction, which
        # is the only property that makes it useful as a control.
        parts: list[str] = []
        for _, marker, _, sentence in chosen:
            for unit in split_sentences(_tidy(sentence)) or [_tidy(sentence)]:
                parts.append(_mark(unit, marker))
        return ProviderDraft(
            text=" ".join(parts), provider=self.name, provider_version=self.version,
            diagnostics={"sentences": len(chosen),
                         "markers": [marker for _, marker, _, _ in chosen]})

    def config(self) -> dict[str, Any]:
        config = super().config()
        config.update({"strategy": "extractive: quotes corpus sentences verbatim",
                       "max_sentences": self.max_sentences,
                       "prefer_citable": self.prefer_citable})
        return config


class ExternalLLMProvider(AnswerProvider):
    """The declared integration point for a future model. Deliberately inert.

    Phase 7 §3, §4 and §21: no external API is wired up, nothing is sent
    anywhere, and the project's ₹0 requirement means none may be assumed
    available. A subclass implementing :meth:`generate` is the entire surface a
    future integration touches — retrieval, policy, validation, grounding and
    rendering do not change.

    A conforming implementation must send only ``request.query``,
    ``request.passages`` and ``request.instructions``, must never send the
    corpus or any chunk id, and must return prose whose only citations are
    ``[n]`` markers drawn from the passages it was given. Everything it returns
    is then validated exactly as the mock provider's output is; a provider is
    not trusted because of what it is.
    """

    name = "external_llm"
    version = PROVIDER_VERSION

    def __init__(self, *, model: str | None = None):
        self.model = model

    def generate(self, request: ProviderRequest) -> ProviderDraft:
        raise NotImplementedError(
            "No external provider is configured. Phase 7 ships with no network "
            "dependency and no paid API; see docs/ANSWER_GENERATION_CONTRACT.md "
            "for the contract an implementation must satisfy.")

    def config(self) -> dict[str, Any]:
        config = super().config()
        config.update({"model": self.model, "implemented": False,
                       "requires_network": True})
        return config


def _tidy(sentence: str) -> str:
    """Collapse the whitespace a PDF text layer scatters through a sentence."""
    return re.sub(r"\s+", " ", sentence).strip()


def _mark(sentence: str, marker: int) -> str:
    """Attach a citation marker *inside* the sentence it belongs to.

    Written after the full stop — "…laws. [1]" — the marker detaches: the
    grounding check splits sentences on a terminal period followed by
    whitespace, so the marker becomes the opening token of the next sentence and
    the sentence it was meant to support is reported as an uncited claim. Before
    the stop it stays put, and it is the conventional placement anyway.
    """
    stripped = sentence.rstrip()
    if stripped.endswith((".", "!", "?")):
        return f"{stripped[:-1].rstrip()} [{marker}]{stripped[-1]}"
    return f"{stripped} [{marker}]"
