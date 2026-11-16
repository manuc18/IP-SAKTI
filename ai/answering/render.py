"""Rendering, kept apart from everything that decides anything.

Phase 7 §13. The renderer reads a validated :class:`~ai.answering.models.Answer`
and produces text. It cannot add a citation, promote a caution, or change a
confidence, because it receives an object that is already final — which is what
makes it safe to write a second renderer later for a UI without any of Phase 7's
guarantees having to be restated there.

**On disclaimers (§16).** A paragraph of legal boilerplate under every answer is
not caution, it is wallpaper: readers stop seeing it within a day, and it is
then absent exactly when it matters. So the standing position is a single
system-level line, shown once by whatever surrounds this — see
:data:`SYSTEM_DISCLAIMER` — and per-answer cautions appear only when the
evidence earned one: stale text, conflicting provisions, weak provenance, or no
sufficient evidence at all.
"""
from __future__ import annotations

from typing import Any

from ai.answering.models import Answer, Citation

RENDER_VERSION = "0.1.0"

#: Shown once by the surrounding application, not appended to every answer.
SYSTEM_DISCLAIMER = (
    "IP-SAKTI is an information and research tool that answers from a fixed "
    "corpus of published legal texts. It is not legal advice and is not a "
    "substitute for a qualified practitioner."
)

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "I couldn't find sufficient authoritative evidence in the current corpus to "
    "answer this reliably."
)


def render_citation(citation: Citation) -> str:
    """One source line: `Document, Locator, p. 12`.

    Pages render as an inclusive range when they are contiguous, which they are
    for almost every chunk, and as a list when a provision straddles a page
    break in a way a range would misstate.
    """
    parts = [citation.citation_text.strip() or citation.document_id]
    pages = list(citation.pages)
    if pages:
        if len(pages) == 1:
            parts.append(f"p. {pages[0]}")
        elif pages == list(range(pages[0], pages[-1] + 1)):
            parts.append(f"pp. {pages[0]}–{pages[-1]}")
        else:
            parts.append("pp. " + ", ".join(str(page) for page in pages))
    if citation.temporal_status != "CURRENT":
        parts.append(f"[{citation.temporal_status}]")
    return f"[{citation.marker}] " + ", ".join(parts)


def render_answer(answer: Answer, *, show_confidence: bool = False) -> str:
    """The canonical text form of a finished answer (Phase 7 §13)."""
    lines: list[str] = ["Answer", ""]

    if answer.insufficient_evidence:
        lines.append(answer.answer or INSUFFICIENT_EVIDENCE_MESSAGE)
        reasons = [failure.detail for failure in answer.insufficient_evidence_reasons]
        reasons += [f"{failure.code}: {failure.detail}"
                    for failure in answer.validation_failures]
        if reasons:
            lines.extend(["", "Why:"])
            lines.extend(f"- {reason}" for reason in reasons)
        return "\n".join(lines).rstrip() + "\n"

    lines.append(answer.answer.strip())

    if answer.citations:
        lines.extend(["", "Source(s)", ""])
        lines.extend(render_citation(citation) for citation in answer.citations)

    if answer.conflict_detected:
        lines.extend(["", "Conflicting evidence", ""])
        lines.append("These sources were flagged as conflicting or "
                     "indistinguishable from one another. This system reports "
                     "the disagreement and does not decide between them:")
        lines.extend(f"- {chunk_id}" for chunk_id in answer.conflict_chunks)

    if answer.cautions:
        lines.extend(["", "Caution", ""])
        lines.extend(f"- {caution}" for caution in answer.cautions)

    if show_confidence:
        lines.extend(["", f"Confidence: {answer.confidence} "
                          f"(ordinal, derived from evidence provenance — not a "
                          f"probability and not a measure of legal correctness)"])

    return "\n".join(lines).rstrip() + "\n"


def render_config() -> dict[str, Any]:
    return {"renderer": "text", "version": RENDER_VERSION,
            "disclaimer_policy": "system-level line shown once; per-answer "
                                 "cautions only when the evidence earns one"}
