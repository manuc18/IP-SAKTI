"""Phase 7: grounded answer generation over evidence retrieval already found.

`RERANKED EVIDENCE → GROUNDED ANSWER OBJECT → VALIDATED CITATIONS`, and nothing
further. This package performs no retrieval, loads no model and opens no
network connection; see `docs/ANSWER_ENGINE.md`.
"""
from ai.answering.base import AnswerEngine, GroundedAnswerEngine, as_evidence_set
from ai.answering.citation_validator import ValidationOutcome, validate_citations
from ai.answering.grounding import GroundingReport, check_grounding
from ai.answering.groq import GroqConfig, GroqError, GroqProvider
from ai.answering.models import (
    ANSWERING_VERSION,
    Answer,
    Citation,
    Evidence,
    EvidenceSet,
    Failure,
    SourceDocument,
)
from ai.answering.policy import AnswerPolicy, PolicyReport, apply_policy
from ai.answering.providers import (
    AnswerProvider,
    ExternalLLMProvider,
    ExtractiveMockProvider,
    ProviderDraft,
    ProviderRequest,
)
from ai.answering.render import SYSTEM_DISCLAIMER, render_answer

__all__ = [
    "ANSWERING_VERSION", "Answer", "AnswerEngine", "AnswerPolicy",
    "AnswerProvider", "Citation", "Evidence", "EvidenceSet",
    "ExternalLLMProvider", "ExtractiveMockProvider", "Failure",
    "GroqConfig", "GroqError", "GroqProvider",
    "GroundedAnswerEngine", "GroundingReport", "PolicyReport", "ProviderDraft",
    "ProviderRequest", "SYSTEM_DISCLAIMER", "SourceDocument",
    "ValidationOutcome", "apply_policy", "as_evidence_set", "check_grounding",
    "render_answer", "validate_citations",
]
