"""The customs officer. No passport, no entry.

Phase 7 §10 and §11. Every citation that reaches a reader must resolve against
the evidence set the answer engine was given, and the citation's *contents* —
document, locator, pages, source PDF identity, evidence level, temporal status —
are copied out of that evidence record. The provider's prose contributes a
marker and nothing more. Fabricating a citation is therefore not something this
module catches; it is something the data flow makes impossible, and what this
module catches is everything else.

Two rules are worth stating because they are the ones a well-meaning refactor
breaks.

**Failure is not repaired.** Phase 7 §10 is explicit, and the reason is that a
validator which quietly drops the one bad citation and returns the rest teaches
every caller downstream that its output can be trusted without being read. When
validation fails this returns no citations at all and a list of structured
failures. The engine's response to that is to abstain, not to publish four
citations out of five.

**The prose is scanned too.** A provider can cite `[1]` correctly and still
write "under section 45 of the Patents Act" in the sentence, where no section 45
was supplied. That is a fabricated citation wearing prose clothes, so every
provision named anywhere in the answer must trace back to the evidence, and any
URL at all is a failure — the pipeline never asks a provider for a link.

"Trace back" has three arms, and none of them is a loophole. A provision is
supported if it *is* one of the supplied passages; or if it is a subdivision of
a supplied passage **whose marker is actually present in that passage's text**
(Phase 7.4 — Section 3's chunk supports "section 3(p)" because clause (p) is in
it, and supports nothing it does not contain); or if some supplied passage's own
text names it. Statutes cross-reference constantly — Rule 65 of the Drugs
and Cosmetics Rules cannot be quoted without naming Form 20 — so a rule that
required every named provision to be separately supplied would reject correct
quotations of the corpus for containing the corpus. That is the same trap Phase
6 §14 recorded about anchor terms, and it is avoided the same way: the test is
whether the string came from the evidence, not whether it appears in an index.
What remains caught is the thing worth catching — a provision that appears in
neither the supplied locators nor the supplied text, which is precisely a
provision the provider invented.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Sequence

from ai.answering.grounding import extract_markers
from ai.answering.models import (
    Citation,
    Evidence,
    EvidenceSet,
    Failure,
    SourceDocument,
)
from ai.answering.policy import (
    AnswerPolicy,
    meets_evidence_floor,
    ref_matches_locator,
    ref_matches_subdivision_of,
)
from ai.retrieval.identifiers import LOCATOR_FIELD, parse_identifiers

#: 0.2.0 — Phase 7.4: a clause of a supplied provision is admissible in
#: prose when the clause is present in the supplied text, and marker
#: aliases reach here already normalised.
VALIDATOR_VERSION = "0.2.0"

_URL = re.compile(r"\b(?:https?://|www\.)\S+", re.IGNORECASE)


@dataclass(frozen=True)
class ValidationOutcome:
    """Either a set of citations, or the reasons there are none."""

    ok: bool
    citations: tuple[Citation, ...] = ()
    failures: tuple[Failure, ...] = ()
    version: str = VALIDATOR_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok,
                "citations": [c.to_dict() for c in self.citations],
                "failures": [f.to_dict() for f in self.failures],
                "version": self.version}


def build_citation(evidence: Evidence) -> Citation:
    """Assemble a citation from an evidence record. Copies; never composes.

    `citation_text` is the chunk's own Phase 2 citation string, which is where
    the document name and locator were established. Rebuilding it here from
    parts would be a second opinion about how a provision is cited, and two
    opinions is one too many.
    """
    return Citation(
        marker=evidence.marker,
        chunk_id=evidence.chunk_id,
        document_id=evidence.document_id,
        locator=evidence.locator_text,
        pages=tuple(evidence.physical_pages),
        citation_text=evidence.citation,
        source_evidence_level=evidence.source_evidence_level,
        temporal_status=evidence.temporal_status,
        locator_unavailable=evidence.locator_unavailable,
        source_document=SourceDocument(
            document_id=evidence.document_id,
            filename=evidence.source_filename,
            sha256=evidence.source_sha256,
            # Preserved when the provenance registry established one, absent
            # otherwise. Phase 7 §9: never guessed from a document's name.
            official_source_url=evidence.official_source_url,
            source_version_id=evidence.source_version_id,
            consolidated_as_of=evidence.consolidated_as_of,
        ),
    )


def _check_evidence(evidence: Evidence, policy: AnswerPolicy) -> list[Failure]:
    """Rules 3, 4, 5 and 7 for one cited passage."""
    failures: list[Failure] = []
    where = {"marker": evidence.marker, "chunk_id": evidence.chunk_id}

    # 5. Non-citable evidence may be read and shown internally, never cited.
    if not evidence.citable:
        failures.append(Failure(
            code="non_citable_evidence",
            detail=f"chunk {evidence.chunk_id} is marked citable=false and cannot "
                   f"be presented as an authoritative citation", **where))

    # 3. Document identity.
    if not evidence.document_id or not str(evidence.document_name).strip():
        failures.append(Failure(
            code="missing_document_identity",
            detail="citation has no resolvable document identity", **where))

    # 4. A locator, or the repository's explicit locator_unavailable state — and
    #    the two must agree. A chunk claiming both a locator and its absence is a
    #    corrupt record, and guessing which half is right is exactly the silent
    #    repair this module refuses.
    has_locator = bool(evidence.locator_text.strip())
    if evidence.locator_unavailable and has_locator:
        failures.append(Failure(
            code="inconsistent_locator_state",
            detail=f"chunk {evidence.chunk_id} is flagged locator_unavailable but "
                   f"renders a locator ({evidence.locator_text!r})", **where))
    elif not evidence.locator_unavailable and not has_locator:
        failures.append(Failure(
            code="missing_locator",
            detail=f"chunk {evidence.chunk_id} has no locator and is not flagged "
                   f"locator_unavailable", **where))

    # 7. Pages must resolve to somewhere in the source PDF.
    pages = evidence.physical_pages
    if not pages or any((not isinstance(p, int)) or p < 1 for p in pages):
        failures.append(Failure(
            code="unresolvable_pages",
            detail=f"chunk {evidence.chunk_id} carries no resolvable physical page "
                   f"({pages!r})", **where))

    # The authority floor, which is policy rather than integrity, but a citation
    # below it is still not admissible as one.
    if not meets_evidence_floor(evidence.source_evidence_level,
                                policy.minimum_evidence_level):
        failures.append(Failure(
            code="below_evidence_floor",
            detail=f"source evidence level {evidence.source_evidence_level} is "
                   f"below the policy floor {policy.minimum_evidence_level}",
            **where))
    return failures


def _scan_prose(text: str, evidence_set: EvidenceSet) -> list[Failure]:
    """Rules 6 and 8 against the sentences, not just the markers (§11)."""
    failures: list[Failure] = []

    for match in _URL.finditer(text or ""):
        failures.append(Failure(
            code="fabricated_url",
            detail=f"the answer text contains a URL ({match.group(0)!r}); a "
                   f"provider never supplies links, they come from the "
                   f"provenance registry via the citation record"))

    # Every provision named by any supplied passage's own text. Computed once:
    # this is the corpus's own cross-referencing, and quoting it is not invention.
    cross_referenced = {
        (ref.kind, ref.number, ref.subdivisions)
        for item in evidence_set for ref in parse_identifiers(item.text)
    }

    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    for ref in parse_identifiers(text or ""):
        if ref.kind not in LOCATOR_FIELD:
            continue
        key = (ref.kind, ref.number, ref.subdivisions)
        if key in seen:
            continue
        seen.add(key)
        if any(ref_matches_locator(ref, item) for item in evidence_set):
            continue
        # A clause of a supplied provision, where the clause is genuinely in the
        # supplied text. Phase 7.3 Q001 refused an answer for citing "section
        # 3(p)" against the Section 3 chunk that contains clause (p); this is
        # the third arm that admits it. Narrow by construction — see
        # `ref_matches_subdivision_of`, which will not admit 3(zz).
        if any(ref_matches_subdivision_of(ref, item) for item in evidence_set):
            continue
        if key in cross_referenced:
            continue
        rendered = f"{ref.kind} {ref.number}" + "".join(
            f"({sub})" for sub in ref.subdivisions)
        failures.append(Failure(
            code="unsupported_locator_in_prose",
            detail=f"the answer names {rendered}, which is neither a supplied "
                   f"provision nor named by any supplied passage"))
    return failures


def validate_citations(text: str, evidence_set: EvidenceSet,
                       policy: AnswerPolicy | None = None,
                       expected_fingerprint: str | None = None) -> ValidationOutcome:
    """Validate every citation in `text` against `evidence_set`.

    `expected_fingerprint` is the identity of the evidence set as it was when the
    engine handed it to the provider. Checking it makes "these citations came
    from the evidence that was supplied" a verified statement rather than a
    trusted one — Phase 7 §10 rule 6.
    """
    policy = policy or AnswerPolicy()
    failures: list[Failure] = []

    if expected_fingerprint is not None and expected_fingerprint != evidence_set.fingerprint:
        failures.append(Failure(
            code="evidence_not_supplied",
            detail=f"the evidence set validated ({evidence_set.fingerprint}) is not "
                   f"the one supplied to the engine ({expected_fingerprint})"))
        return ValidationOutcome(ok=False, failures=tuple(failures))

    citations: list[Citation] = []
    for marker in extract_markers(text):
        # 1 and 2. The marker must resolve inside the supplied set. There is no
        # other registry to fall back to, by design.
        evidence = evidence_set.by_marker(marker)
        if evidence is None:
            failures.append(Failure(
                code="unknown_marker", marker=marker,
                detail=f"citation [{marker}] does not resolve to any supplied "
                       f"evidence (markers supplied: "
                       f"{list(evidence_set.markers) or 'none'})"))
            continue
        item_failures = _check_evidence(evidence, policy)
        failures.extend(item_failures)
        if not item_failures:
            citations.append(build_citation(evidence))

    failures.extend(_scan_prose(text, evidence_set))

    if failures:
        # No partial acceptance: see the module docstring.
        return ValidationOutcome(ok=False, failures=tuple(failures))
    return ValidationOutcome(ok=True, citations=tuple(citations))
