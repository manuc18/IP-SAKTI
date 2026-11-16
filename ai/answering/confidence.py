"""Two questions the old single label was answering at once, now asked separately.

`_confidence` used to read one thing: the provenance of the documents the answer
cited. Evidence level, temporal status, conflict. Nothing about the answer.

Measured on the 60-question synthetic sample, that produced a label which was
monotone but almost meaningless. Of 41 published answers, 15 were labelled LOW —
and 13 of those 15 were LOW for exactly one reason, that every instrument they
cited is marked STALE. Put to the AI judge, **12 of those 15 LOW answers were
graded correct or partially correct (80.0%)**, against 85.0% for MODERATE
(17/20) and 100% for HIGH (6/6). A tier that is right four times in five is not
telling a reader what "low confidence" is supposed to tell them. It was
answering a different question — "might this Act have been amended since our
copy?" — and printing the answer under the wrong heading.

So the label is split along the seam it was already broken on:

**Source confidence** is what the old rule computed: how far the cited
*documents* can be trusted and how current they are. Nothing here is relaxed.
Level B never reaches HIGH — `NOT_COMPARED` means the publisher's site was
unreachable when the registry was built and the local copy was never compared,
and promoting an unverified copy because promotion would look better is the
precise move this project exists not to make. STALE never reaches HIGH either.

**Support confidence** is the question that was missing: given those documents,
how well do the passages actually cited hold the answer up? Grounding strength,
how many passages carry it, and where retrieval ranked them.

The published `confidence` is the lower of the two, so a weak answer from an
impeccable source and a strong answer from a stale one are both capped, for
stated and different reasons. Both axes and every factor are on the response, so
a card can say *why* it says LOW instead of just saying LOW — which was the
actual user-visible complaint, and is not fixable by moving a threshold.

On the support thresholds and how far to trust them
---------------------------------------------------

Best-cited-rank and citation count were checked against the same 41 judged
answers. Answers whose best-ranked cited passage was rank 1 were graded
correct-or-partial 33/37 (89.2%); those citing nothing better than rank 2 were
2/4 (50.0%). Answers with two or more citations were 14/15 (93.3%) against 21/26
(80.8%) for a single citation.

Those splits point the right way, and n=4 and n=15 do not establish a threshold.
They are reported because the alternative is choosing constants and not saying
where they came from. The rules below are a-priori defensible on their own
terms — a passage retrieval ranked first is better evidence than one it ranked
sixth — and the sample is consistent with them, which is a weaker claim than
calibration and is the only claim the data supports. The judgements are
AI-generated on a synthetic benchmark and are not human-verified ground truth.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from ai.answering.models import (
    CURRENT,
    HIGH,
    LOW,
    MODERATE,
    NONE,
    STALE,
    Citation,
)

CONFIDENCE_VERSION = "0.3.0"

#: Ordinal, low to high. `min` over this order is how the two axes combine.
_ORDER = (NONE, LOW, MODERATE, HIGH)
_RANK = {level: index for index, level in enumerate(_ORDER)}

#: The best retrieval rank among the cited passages, at or above which support
#: is unreduced. One: the passage the answer leans on should be the passage
#: retrieval put first. Below that the answer is resting on something the
#: retriever itself ranked second-best or worse, which is a real if mild
#: reservation and is recorded as one.
BEST_CITED_RANK_FOR_FULL_SUPPORT = 1


@dataclass(frozen=True)
class Factor:
    """One reason the level is what it is."""

    code: str
    axis: str
    caps_at: str
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "axis": self.axis,
                "caps_at": self.caps_at, "detail": self.detail}


@dataclass(frozen=True)
class ConfidenceReport:
    """The published level, the two axes behind it, and every factor."""

    level: str
    source_confidence: str
    support_confidence: str
    factors: tuple[Factor, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "source_confidence": self.source_confidence,
            "support_confidence": self.support_confidence,
            "factors": [f.to_dict() for f in self.factors],
            "version": CONFIDENCE_VERSION,
            "is_probability": False,
            "note": "an ordinal read of evidence provenance and of how well the "
                    "cited passages support the answer; not an estimate of "
                    "legal correctness",
        }


def _lower(a: str, b: str) -> str:
    return a if _RANK[a] <= _RANK[b] else b


def source_confidence(citations: Sequence[Citation],
                      *, conflicts: bool) -> tuple[str, list[Factor]]:
    """How far the cited documents can be trusted, and how current they are."""
    factors: list[Factor] = []
    if not citations:
        return LOW, [Factor("no_citations", "source", LOW,
                            "the answer cites no passage")]

    levels = {c.source_evidence_level for c in citations}
    statuses = {c.temporal_status for c in citations}
    total = len(citations)
    stale = [c for c in citations if c.temporal_status == STALE]
    weak = sorted(levels & {"C", "D"})

    level = HIGH

    if conflicts:
        level = _lower(level, LOW)
        factors.append(Factor(
            "conflicting_provisions", "source", LOW,
            "the retrieved passages disagree, and the answer does not resolve "
            "which governs"))

    if weak:
        level = _lower(level, LOW)
        factors.append(Factor(
            "weak_provenance", "source", LOW,
            f"a cited source is at evidence level {'/'.join(weak)}: the "
            f"publisher of the underlying PDF could not be established"))

    if stale and len(stale) == total:
        level = _lower(level, LOW)
        factors.append(Factor(
            "all_sources_stale", "source", LOW,
            f"every cited instrument ({total}) has been amended since the "
            f"consolidation held in this corpus"))
    elif stale:
        # Previously this case fell through to MODERATE while all-stale dropped
        # to LOW, so an answer resting partly on an amended instrument scored
        # *better* than one resting wholly on it — although the amended part
        # carries the same risk either way. It is now capped, and named.
        level = _lower(level, MODERATE)
        factors.append(Factor(
            "some_sources_stale", "source", MODERATE,
            f"{len(stale)} of {total} cited instruments have been amended since "
            f"the consolidation held in this corpus"))

    if "B" in levels:
        level = _lower(level, MODERATE)
        factors.append(Factor(
            "local_copy_unverified", "source", MODERATE,
            "a cited source is at evidence level B: its identity is established "
            "but the local copy was never compared against the publisher's"))

    if level == HIGH:
        factors.append(Factor(
            "verified_current_sources", "source", HIGH,
            f"every cited source is level A and marked {CURRENT}"))
    return level, factors


def support_confidence(grounding: Any,
                       citations: Sequence[Citation],
                       best_cited_rank: int | None) -> tuple[str, list[Factor]]:
    """How well the passages actually cited hold the answer up."""
    factors: list[Factor] = []
    if not citations:
        return LOW, [Factor("no_citations", "support", LOW,
                            "the answer cites no passage")]

    level = HIGH

    if not getattr(grounding, "grounded", False):
        level = _lower(level, LOW)
        factors.append(Factor(
            "ungrounded", "support", LOW,
            "at least one claim carries no citation"))

    weakly = tuple(getattr(grounding, "weakly_grounded", ()) or ())
    if weakly:
        level = _lower(level, MODERATE)
        factors.append(Factor(
            "weak_textual_support", "support", MODERATE,
            f"{len(weakly)} sentence(s) cite a passage they share little "
            f"wording with"))

    if best_cited_rank is not None and best_cited_rank > BEST_CITED_RANK_FOR_FULL_SUPPORT:
        level = _lower(level, MODERATE)
        factors.append(Factor(
            "low_ranked_evidence", "support", MODERATE,
            f"the best-ranked passage the answer cites was retrieved at rank "
            f"{best_cited_rank}, not first"))

    if level == HIGH:
        factors.append(Factor(
            "fully_grounded_top_ranked", "support", HIGH,
            "every claim cites a passage, the wording matches, and the answer "
            "rests on the passage retrieval ranked first"))
    return level, factors


def assess(citations: Sequence[Citation], grounding: Any, *,
           conflicts: bool, best_cited_rank: int | None = None
           ) -> ConfidenceReport:
    """The published confidence, as the lower of two separately stated axes."""
    source, source_factors = source_confidence(citations, conflicts=conflicts)
    support, support_factors = support_confidence(grounding, citations,
                                                  best_cited_rank)
    return ConfidenceReport(
        level=_lower(source, support),
        source_confidence=source,
        support_confidence=support,
        factors=tuple(source_factors + support_factors),
    )


def confidence_config() -> dict[str, Any]:
    return {
        "version": CONFIDENCE_VERSION,
        "levels": list(reversed(_ORDER)),
        "axes": ("source_confidence", "support_confidence"),
        "combination": "the published level is the lower of the two axes",
        "best_cited_rank_for_full_support": BEST_CITED_RANK_FOR_FULL_SUPPORT,
        "is_probability": False,
        "promotes_stale_or_unverified_sources": False,
        "guarantee": "no source-side factor can raise a level; HIGH requires "
                     "every cited source to be evidence level A and CURRENT, "
                     "with full grounding on a rank-1 passage",
    }
