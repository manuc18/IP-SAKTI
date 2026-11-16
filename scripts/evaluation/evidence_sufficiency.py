"""Does the retrieved evidence contain what the answer needs? Decided offline.

Recall@k answers a narrow question: did the one chunk the generator wrote the
question from come back. It is exact and it is deterministic, and it is also
unfair in one direction and generous in the other.

*Unfair*, because the corpus contains other chunks that answer the same question
just as completely — the second half of a section that got split, the Act
provision a Rule's heading names. Recall@1 scores those zero.

*Generous*, because a chunk id matching proves nothing about whether a reader
holding that chunk could actually answer. A 40-word heading stub carries the
right id and none of the substance.

So this module computes a second metric over the same run. It asks whether the
top-*k* evidence **contains the material the expected answer is made of**, and
it answers with SUFFICIENT / INSUFFICIENT / UNCERTAIN.

Deliberately not called accuracy
--------------------------------

This is not a judgement about legal correctness and it is not a judgement about
any generated answer — no answer is generated here. It is a statement about the
evidence set, computed by string and term comparison against an expected answer
that was itself extracted verbatim from the corpus. It is called EVIDENCE
SUFFICIENCY for that reason and the name should not be relaxed in reporting.

How it decides
--------------

Five signals, all deterministic, all computed from the question, the expected
answer and the retrieved text:

``span_containment``
    An expected-answer span, normalised for whitespace only, appears verbatim
    inside some retrieved chunk. This is the strongest signal available: the
    spans are verbatim corpus text, so a hit means the reader is looking at the
    exact words the expected answer was drawn from. It is *not* the same as
    target inclusion — overlapping chunks and duplicated provisions mean a span
    can arrive on a chunk whose id was never the label.

``key_term_coverage``
    IDF-weighted fraction of the expected answer's content terms present in the
    retrieved text. Weighted because "the" appearing in both proves nothing and
    "circumvents" appearing in both proves a great deal. The IDF table is the
    corpus's own, taken from the BM25 retriever, so no stopword list is invented
    here.

``target_included`` / ``acceptable_included``
    The label, and the equivalence-widened label set from ``target_equivalence``.

``locator_agreement``
    Some retrieved chunk carries the same provision locator as the target.

The rule is declared rather than fitted:

* SUFFICIENT when a verbatim span is present, **or** when an acceptable target
  is present *and* key-term coverage clears ``STRONG_COVERAGE``.
* INSUFFICIENT when coverage is below ``WEAK_COVERAGE`` and no acceptable target
  and no locator agreement — nothing in the evidence set resembles the answer.
* UNCERTAIN otherwise, and UNCERTAIN is reported as its own bucket rather than
  being folded into either side. A metric that has no middle is a metric that is
  hiding one.

The thresholds are declared constants, not tuned against the result: they were
set once from the property each is trying to express and are reported in the
config block beside every figure so a reader can disagree with them explicitly.
"""
from __future__ import annotations


import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

SUFFICIENCY_VERSION = "0.1.0"

SUFFICIENT = "sufficient"
INSUFFICIENT = "insufficient"
UNCERTAIN = "uncertain"

#: Coverage at or above this, with an acceptable target present, is sufficient.
STRONG_COVERAGE = 0.60
#: Coverage below this, with nothing else present, is insufficient.
WEAK_COVERAGE = 0.30
#: Terms shorter than this carry no discriminating power and are not counted.
MIN_TERM_LENGTH = 4
_WORD = re.compile(r"[a-z0-9]+")
_WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    """Whitespace-only normalisation. Legal wording is never altered here."""
    return _WHITESPACE.sub(" ", text).strip()


def terms(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if len(t) >= MIN_TERM_LENGTH]


def key_term_coverage(expected: str, retrieved: str,
                      idf: Mapping[str, float]) -> float:
    """IDF-weighted share of the expected answer's terms found in the evidence.

    A term absent from the IDF table is a term the corpus has never seen, which
    for an expected answer extracted verbatim from the corpus means it came from
    the question template rather than the provision. It is given the median
    weight rather than dropped, so template words neither dominate nor vanish.
    """
    wanted = terms(expected)
    if not wanted:
        return 0.0
    present = set(terms(retrieved))
    default = _median_idf(idf)
    total = 0.0
    hit = 0.0
    for term in set(wanted):
        weight = idf.get(term, default)
        if weight <= 0.0:
            weight = default
        total += weight
        if term in present:
            hit += weight
    return hit / total if total else 0.0


_MEDIAN_CACHE: dict[int, float] = {}


def _median_idf(idf: Mapping[str, float]) -> float:
    key = id(idf)
    if key not in _MEDIAN_CACHE:
        values = sorted(v for v in idf.values() if v > 0.0)
        _MEDIAN_CACHE[key] = values[len(values) // 2] if values else 1.0
    return _MEDIAN_CACHE[key]


def span_containment(spans: Sequence[str], retrieved_texts: Sequence[str]) -> bool:
    """True when any expected span appears verbatim in any retrieved chunk."""
    haystacks = [normalise(text) for text in retrieved_texts]
    for span in spans:
        needle = normalise(span)
        if len(needle) < 40:
            continue
        if any(needle in hay for hay in haystacks):
            return True
    return False


def _locator_key(locator: Mapping[str, Any] | None) -> str:
    if not locator:
        return ""
    parts = [f"{k}={locator[k]}" for k in sorted(locator)
             if locator[k] not in (None, "", [])]
    return "|".join(str(p).lower() for p in parts)


@dataclass(frozen=True)
class Sufficiency:
    """One question's evidence-sufficiency verdict and the signals behind it."""

    verdict: str
    coverage: float
    span_present: bool
    target_included: bool
    acceptable_included: bool
    locator_agreement: bool
    document_agreement: bool
    depth: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "key_term_coverage": round(self.coverage, 4),
            "span_present": self.span_present,
            "target_included": self.target_included,
            "acceptable_included": self.acceptable_included,
            "locator_agreement": self.locator_agreement,
            "document_agreement": self.document_agreement,
            "depth": self.depth,
            "version": SUFFICIENCY_VERSION,
            "is_legal_accuracy": False,
        }


def assess(*, expected_answer: str, expected_spans: Sequence[str],
           retrieved: Sequence[Mapping[str, Any]], target_chunk_ids: Sequence[str],
           acceptable_chunk_ids: Sequence[str], target_document_id: str | None,
           target_locators: Sequence[Mapping[str, Any]],
           idf: Mapping[str, float], depth: int) -> Sufficiency:
    """Verdict over the first `depth` retrieved items.

    `retrieved` items need only expose `chunk_id`, `document_id`, `text` and
    `locator`; that is the intersection of what every stage of the pipeline
    already carries, so nothing has to be re-derived to score a run.
    """
    window = list(retrieved[:depth])
    texts = [str(item.get("text") or "") for item in window]
    ids = {str(item.get("chunk_id")) for item in window}
    documents = {str(item.get("document_id")) for item in window}
    joined = "\n".join(texts)

    coverage = key_term_coverage(expected_answer, joined, idf)
    span = span_containment(expected_spans or [expected_answer], texts)
    target_in = bool(ids & set(target_chunk_ids))
    acceptable_in = bool(ids & set(acceptable_chunk_ids or target_chunk_ids))
    wanted_locators = {_locator_key(loc) for loc in target_locators if loc}
    wanted_locators.discard("")
    locator_in = any(_locator_key(item.get("locator")) in wanted_locators
                     for item in window) if wanted_locators else False
    document_in = bool(target_document_id) and target_document_id in documents

    if span or (acceptable_in and coverage >= STRONG_COVERAGE):
        verdict = SUFFICIENT
    elif coverage < WEAK_COVERAGE and not acceptable_in and not locator_in:
        verdict = INSUFFICIENT
    else:
        verdict = UNCERTAIN

    return Sufficiency(verdict=verdict, coverage=coverage, span_present=span,
                       target_included=target_in, acceptable_included=acceptable_in,
                       locator_agreement=locator_in, document_agreement=document_in,
                       depth=depth)


def sufficiency_config() -> dict[str, Any]:
    return {
        "version": SUFFICIENCY_VERSION,
        "deterministic": True,
        "uses_model": False,
        "provider": None,
        "verdicts": (SUFFICIENT, INSUFFICIENT, UNCERTAIN),
        "thresholds": {
            "strong_coverage": STRONG_COVERAGE,
            "weak_coverage": WEAK_COVERAGE,
            "min_term_length": MIN_TERM_LENGTH,
            "min_span_length_chars": 40,
        },
        "rule": ("SUFFICIENT when a verbatim expected-answer span is present in "
                 "the retrieved text, or when an acceptable target chunk is "
                 "present and IDF-weighted key-term coverage is at least "
                 f"{STRONG_COVERAGE}; INSUFFICIENT when coverage is below "
                 f"{WEAK_COVERAGE} with no acceptable target and no locator "
                 "agreement; UNCERTAIN otherwise"),
        "claim_safety": ("EVIDENCE SUFFICIENCY is a statement about the "
                         "retrieved evidence set, not about legal correctness "
                         "and not about any generated answer."),
    }
