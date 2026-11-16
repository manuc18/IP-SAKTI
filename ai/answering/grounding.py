"""Deterministic grounding checks over an answer and the evidence it was given.

Phase 7 §12 is precise about the ambition: this does **not** try to establish
that an answer is legally true. It establishes four mechanical properties, all
of which are checkable without a model, and none of which is a truth claim.

* **Citation support** — every ``[n]`` in the prose resolves to supplied evidence.
* **Evidence coverage** — every sentence that asserts something carries a marker.
* **Unsupported claims** — the assertions that do not.
* **Citation completeness** — evidence that was supplied and then never used.

A fifth check is advisory. A sentence that cites `[2]` and shares no content
term with `[2]`'s text is *weakly grounded*: the citation resolves, so it is not
a hallucination, but the association between claim and source is thin. That is a
lexical heuristic and is reported as one — Phase 7 §12 forbids adding an
embedding model or a second cross-encoder to do better, and a heuristic labelled
as a heuristic is more useful than a number that pretends otherwise.

**Sentence splitting errs towards merging.** Legal prose is dense with "s. 3",
"r. 65(2)" and "No. 332", and a splitter that breaks on those would manufacture
fragments that assert nothing and then report them as unsupported claims. The
cost of the choice is stated rather than hidden: a merged sentence containing
one cited assertion and one uncited assertion is counted as supported, so
unsupported-claim detection under-reports rather than over-reports.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from ai.answering.models import Evidence, EvidenceSet
from ai.answering.policy import content_terms

#: 0.2.0 — Phase 7.4: marker aliases are normalised before any check reads
#: them, so a report produced under 0.1.0 and one produced now can disagree
#: about the same answer text. That is exactly what a version is for.
GROUNDING_VERSION = "0.2.0"

#: The only citation syntax the pipeline accepts. A provider writes `[2]`; it
#: never writes a chunk id, a section number or a URL that becomes a citation.
CITATION_MARKER = re.compile(r"\[(\d+)\]")

#: Marker spellings that *mean* `[n]` and are rewritten to it before anything
#: reads them. Exactly one alias, and it is here because a model actually wrote
#: it: in Phase 7.3 Q001 the answer cited 【1】 — full-width brackets —, which this
#: module's ASCII-only pattern could not see, so the citation was invisible to
#: the validator *and* to the grounding check and a correctly-sourced sentence
#: was reported as an uncited claim.
#:
#: An enumerated table rather than a pattern, and that is the whole design. A
#: rule like "any bracket-ish pair around a digit" would start admitting
#: punctuation nobody agreed to, and marker syntax is a contract with the
#: provider, not a guess about it. Adding a spelling is a one-line change here,
#: made deliberately. Note also that the alias is as strict as the ASCII form:
#: no internal whitespace, because `[ 1 ]` is not accepted either and a
#: normaliser that was more permissive than the thing it normalises to would be
#: widening the contract while claiming to translate it.
MARKER_ALIASES: tuple[tuple[str, str], ...] = (('\u3010', '['), ('\u3011', ']'))

_ALIASED_MARKER = re.compile(r"\u3010(\d+)\u3011")


def normalise_markers(text: str) -> str:
    """Rewrite recognised equivalent marker spellings as `[n]`.

    Applied once, by the engine, to the text that is then validated, grounded
    *and published* — so a reader never sees a marker in the prose that the
    citation list does not use. Normalising only inside the checks would leave
    the two disagreeing, which is a smaller bug than the one being fixed but
    still one this layer would have put there on purpose.
    """
    return _ALIASED_MARKER.sub(lambda match: f"[{match.group(1)}]", text or "")


#: A sentence ends at `.`, `!` or `?` — but not at a period that is one of
#: several in a row. An ellipsis is punctuation *inside* a sentence, and the
#: consequence of reading it as a terminator is not a cosmetic mis-split: in the
#: first live Groq run the model wrote `…does not result in an enhanced... [1]`,
#: the break detached `[1]` into a sentence of its own, and the claim it was
#: supporting was then reported as uncited and the whole answer refused. The
#: lookbehind is deliberately narrow — two periods, no more grammar than that.
#: `…` needs no handling: it is not in the terminator class, so it never split.
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])(?<!\.\.)\s+")

#: Tokens that end in a full stop without ending a sentence.
_ABBREVIATIONS = frozenset("""
s ss r rr cl art arts sch no nos cf para paras sec secs reg regs vs viz i.e e.g
etc pt ch cap ors anr govt
""".split())

#: Sentences that are framing rather than assertion, so they are not expected to
#: carry a citation. Matched as a prefix, case-folded.
_NON_CLAIM_PREFIXES = (
    "i couldn't find", "i could not find", "the supplied evidence",
    "no supplied", "this system does not", "the corpus does not",
    "sources:", "source(s):", "answer", "note:", "caution:",
)


def split_sentences(text: str) -> list[str]:
    """Split prose into sentences, keeping legal abbreviations intact."""
    parts = _SENTENCE_BREAK.split((text or "").strip())
    merged: list[str] = []
    for part in parts:
        if merged and _ends_with_abbreviation(merged[-1]):
            merged[-1] = f"{merged[-1]} {part}"
        else:
            merged.append(part)
    return [part.strip() for part in merged if part.strip()]


def _ends_with_abbreviation(sentence: str) -> bool:
    match = re.search(r"(\S+)\.\s*$", sentence)
    if not match:
        return False
    tail = match.group(1).lower().rstrip(".")
    return tail in _ABBREVIATIONS or tail.isdigit() or len(tail) <= 2


def extract_markers(text: str) -> list[int]:
    """Every citation marker in the text, in order, deduplicated."""
    seen: list[int] = []
    for match in CITATION_MARKER.finditer(text or ""):
        marker = int(match.group(1))
        if marker not in seen:
            seen.append(marker)
    return seen


def is_claim(sentence: str) -> bool:
    """Does this sentence assert something that needs a source?"""
    stripped = sentence.strip()
    if not stripped:
        return False
    lowered = stripped.lower()
    if any(lowered.startswith(prefix) for prefix in _NON_CLAIM_PREFIXES):
        return False
    # A line that is only a citation marker, or only a heading, asserts nothing.
    without_markers = CITATION_MARKER.sub("", stripped).strip(" .,;:—-")
    return len(without_markers.split()) >= 3


@dataclass(frozen=True)
class SentenceGrounding:
    """One sentence and the evidence it points at."""

    sentence: str
    markers: tuple[int, ...]
    is_claim: bool
    supported: bool
    weak: bool
    shared_terms: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"sentence": self.sentence, "markers": list(self.markers),
                "is_claim": self.is_claim, "supported": self.supported,
                "weak": self.weak, "shared_terms": list(self.shared_terms)}


@dataclass(frozen=True)
class GroundingReport:
    """Phase 7 §12's four checks, plus the advisory fifth."""

    sentences: tuple[SentenceGrounding, ...] = ()
    unresolved_markers: tuple[int, ...] = ()
    unsupported_claims: tuple[str, ...] = ()
    uncited_evidence: tuple[str, ...] = ()
    weakly_grounded: tuple[str, ...] = ()
    claims: int = 0
    supported_claims: int = 0
    version: str = GROUNDING_VERSION

    @property
    def citation_support(self) -> bool:
        """Every marker in the prose resolved to supplied evidence."""
        return not self.unresolved_markers

    @property
    def supported_claim_rate(self) -> float:
        return round(self.supported_claims / self.claims, 4) if self.claims else 0.0

    @property
    def grounded(self) -> bool:
        """Every marker resolved and every claim carried one.

        Weak grounding does not clear this flag: the citation is real, and
        downgrading a real citation on a lexical heuristic would be the heuristic
        making a decision it is not good enough to make.
        """
        return self.citation_support and not self.unsupported_claims and self.claims > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "grounded": self.grounded,
            "citation_support": self.citation_support,
            "claims": self.claims,
            "supported_claims": self.supported_claims,
            "supported_claim_rate": self.supported_claim_rate,
            "unresolved_markers": list(self.unresolved_markers),
            "unsupported_claims": list(self.unsupported_claims),
            "uncited_evidence": list(self.uncited_evidence),
            "weakly_grounded": list(self.weakly_grounded),
            "sentences": [s.to_dict() for s in self.sentences],
            "version": self.version,
        }


def check_grounding(text: str, evidence_set: EvidenceSet) -> GroundingReport:
    """Run Phase 7 §12's checks over an answer and the evidence it was given."""
    sentences: list[SentenceGrounding] = []
    unresolved: list[int] = []
    unsupported: list[str] = []
    weak: list[str] = []
    used: set[int] = set()
    claims = 0
    supported_claims = 0

    for sentence in split_sentences(text):
        markers = tuple(extract_markers(sentence))
        claim = is_claim(sentence)
        resolved: list[Evidence] = []
        for marker in markers:
            item = evidence_set.by_marker(marker)
            if item is None:
                if marker not in unresolved:
                    unresolved.append(marker)
            else:
                resolved.append(item)
                used.add(marker)

        supported = bool(resolved)
        shared: tuple[str, ...] = ()
        if resolved:
            sentence_terms = content_terms(CITATION_MARKER.sub("", sentence))
            evidence_terms: set[str] = set()
            for item in resolved:
                evidence_terms |= set(content_terms(item.text))
            shared = tuple(sorted(sentence_terms & evidence_terms))

        is_weak = bool(resolved) and not shared
        if claim:
            claims += 1
            if supported:
                supported_claims += 1
            else:
                unsupported.append(sentence)
        if is_weak:
            weak.append(sentence)

        sentences.append(SentenceGrounding(
            sentence=sentence, markers=markers, is_claim=claim,
            supported=supported, weak=is_weak, shared_terms=shared))

    uncited = tuple(item.chunk_id for item in evidence_set
                    if item.marker not in used)

    return GroundingReport(
        sentences=tuple(sentences),
        unresolved_markers=tuple(unresolved),
        unsupported_claims=tuple(unsupported),
        uncited_evidence=uncited,
        weakly_grounded=tuple(weak),
        claims=claims,
        supported_claims=supported_claims,
    )
