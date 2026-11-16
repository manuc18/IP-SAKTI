"""Evidence policy: four separate questions about one passage.

Phase 7 §6 insists these stay apart, and the corpus shows why. *Relevant* is
about the query. *Authoritative* is about who published the PDF. *Current* is
about whether the text still states the law. *Citable* is about whether a
locator was ever established. They are independent, and every project that
collapses them into one "quality" score ends up unable to say why it dropped
something.

**This module labels; it does not discard.** Phase 7 §6 forbids throwing away
Level B/C or stale evidence, and the corpus makes that instruction load-bearing:
1,717 of 3,073 chunks (55.9%) are STALE, including all 920 chunks of the Drugs
and Cosmetics Rules, 1945 — the single most Ayurveda-relevant instrument in the
corpus. A policy that refused stale evidence would refuse most of the questions
the project exists to answer. So staleness produces a caution and a label on the
citation, never a silent deletion, and `stale_only_is_insufficient` exists so the
opposite policy is one flag away rather than a rewrite.

**Conflict means the same provision, not merely different provisions.** Phase 7
§15 asks for divergent temporal statuses and versions. Firing on any evidence
set that mixes CURRENT and STALE would fire on most of them, and a flag that is
always true carries no information. So a conflict requires two passages that are
*about the same thing* — the same document and the same locator — or an explicit
textual override naming another supplied provision. A set that merely mixes
statuses across different provisions gets a caution instead.

Nothing here resolves a conflict. Phase 7 §15 is explicit that legal reasoning
about which provision prevails is out of scope, and a deterministic module is
the wrong place to acquire that opinion by accident.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ai.answering.models import (
    CURRENT,
    EVIDENCE_LEVELS,
    STALE,
    Evidence,
    EvidenceSet,
    Failure,
)
from ai.retrieval.identifiers import LOCATOR_FIELD, parse_identifiers
from ai.retrieval.tokenizer import tokenize

POLICY_VERSION = "0.3.0"

#: Words carried by almost every legal question, so their presence says nothing
#: about whether a passage addresses it. Deliberately short: this is a screen
#: for evidence that shares *nothing* with the query, not a relevance model, and
#: a long list would start making relevance judgements it cannot support.
STOPWORDS = frozenset("""
a an and any are as at be been being by can do does for from has have how in
into is it its may must my no not of on or shall should such that the their
then there these this those to under upon was were what when where which who
whom whose will with would you your
""".split())

#: Ordered worst-to-best, so a floor can be expressed as a comparison.
_LEVEL_ORDER = {level: index for index, level in enumerate(reversed(EVIDENCE_LEVELS))}

#: Phrases by which one provision announces that it has displaced another. Used
#: only in combination with a locator that resolves to another *supplied* item,
#: never on their own — "omitted" appears in plenty of prose that overrides
#: nothing.
OVERRIDE_MARKERS = (
    "omitted by", "substituted by", "inserted by", "repealed by", "repealed and",
    "shall cease to have effect", "stands repealed", "shall be omitted",
    "shall be substituted", "in supersession of", "superseded by",
)


@dataclass(frozen=True)
class AnswerPolicy:
    """The rules the answer layer applies, stated rather than assumed.

    Every default here is a judgement that can be argued with, which is exactly
    why each is a named field recorded on the answer object instead of a
    constant buried in a branch.
    """

    #: Lowest source evidence level admissible as a citation. "C" admits the
    #: whole corpus (A 1,727 · B 1,155 · C 191) and excludes only level D, which
    #: no chunk currently carries. Raising it to "A" is a real restriction:
    #: it would drop the Patents Act and the Trade Marks Act, both level B.
    minimum_evidence_level: str = "C"

    #: Whether an answer needs at least one citable passage before it may assert
    #: anything. With this false the engine would answer from non-citable text
    #: and offer no source, which is the failure mode Phase 3 was built to stop.
    require_citable_authority: bool = True

    #: Whether evidence must share content terms with the query to be admitted.
    require_relevance: bool = True

    #: How many content terms must overlap. One is deliberately permissive: this
    #: screen exists to catch evidence with no lexical connection at all, and a
    #: higher threshold would start rejecting correct provisions that happen to
    #: use different words — which is the entire reason the pipeline has a dense
    #: retriever and a cross-encoder in front of it.
    relevance_min_overlap: int = 1

    #: Phase 7 §14 lists "only stale evidence for a current-law question" as an
    #: insufficiency condition. Set false by default, because 55.9% of this
    #: corpus is STALE and abstaining on all of it would silence the Drugs and
    #: Cosmetics Rules entirely. False answers with a prominent staleness
    #: caution and a STALE label on every citation; true abstains instead.
    #: See docs/ANSWER_ENGINE.md — this is the phase's most consequential switch.
    stale_only_is_insufficient: bool = False

    #: Phase 7 §15 says expose the conflict and make the answer acknowledge it,
    #: not refuse. True turns a detected conflict into an abstention.
    abstain_on_conflict: bool = False

    #: Whether a claim sentence carrying no citation blocks the answer. True is
    #: the strict reading of Phase 7 §12 and costs nothing under the extractive
    #: provider, which cites every sentence it emits by construction. It becomes
    #: load-bearing the moment a real model writes connective prose.
    require_full_grounding: bool = True

    #: Ceiling on citations in one answer, spent by the provider rather than
    #: enforced by the validator: trimming a validated citation list would be
    #: the silent repair citation_validator.py refuses to perform.
    max_citations: int = 5

    version: str = POLICY_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_version": self.version,
            "minimum_evidence_level": self.minimum_evidence_level,
            "require_citable_authority": self.require_citable_authority,
            "require_relevance": self.require_relevance,
            "relevance_min_overlap": self.relevance_min_overlap,
            "stale_only_is_insufficient": self.stale_only_is_insufficient,
            "abstain_on_conflict": self.abstain_on_conflict,
            "require_full_grounding": self.require_full_grounding,
            "max_citations": self.max_citations,
        }


@dataclass(frozen=True)
class EvidenceAssessment:
    """The four axes, kept apart, for one passage."""

    marker: int
    chunk_id: str
    addresses_query: bool
    overlap_terms: tuple[str, ...]
    authoritative: bool
    source_evidence_level: str
    current: bool
    temporal_status: str
    citable: bool
    admissible_as_citation: bool
    role: str
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "marker": self.marker,
            "chunk_id": self.chunk_id,
            "addresses_query": self.addresses_query,
            "overlap_terms": list(self.overlap_terms),
            "authoritative": self.authoritative,
            "source_evidence_level": self.source_evidence_level,
            "current": self.current,
            "temporal_status": self.temporal_status,
            "citable": self.citable,
            "admissible_as_citation": self.admissible_as_citation,
            "role": self.role,
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class Conflict:
    """Two or more passages that disagree about the same provision."""

    kind: str
    chunk_ids: tuple[str, ...]
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "chunk_ids": list(self.chunk_ids),
                "detail": self.detail}


@dataclass(frozen=True)
class PolicyReport:
    """What the policy concluded about one evidence set, and why."""

    assessments: tuple[EvidenceAssessment, ...] = ()
    conflicts: tuple[Conflict, ...] = ()
    cautions: tuple[str, ...] = ()
    insufficient_reasons: tuple[Failure, ...] = ()

    @property
    def citable_markers(self) -> tuple[int, ...]:
        return tuple(a.marker for a in self.assessments if a.admissible_as_citation)

    @property
    def relevant_markers(self) -> tuple[int, ...]:
        return tuple(a.marker for a in self.assessments if a.addresses_query)

    @property
    def conflict_chunks(self) -> tuple[str, ...]:
        seen: list[str] = []
        for conflict in self.conflicts:
            for chunk_id in conflict.chunk_ids:
                if chunk_id not in seen:
                    seen.append(chunk_id)
        return tuple(seen)

    @property
    def insufficient(self) -> bool:
        return bool(self.insufficient_reasons)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessments": [a.to_dict() for a in self.assessments],
            "conflicts": [c.to_dict() for c in self.conflicts],
            "cautions": list(self.cautions),
            "insufficient_reasons": [f.to_dict() for f in self.insufficient_reasons],
        }


# -- the four axes ------------------------------------------------------------


def content_terms(text: str) -> frozenset[str]:
    """Query/evidence terms that carry topic. Identifiers survive intact."""
    return frozenset(token for token in tokenize(text)
                     if token not in STOPWORDS and (len(token) > 2 or token.isdigit()
                                                    or not token.isalpha()))


def meets_evidence_floor(level: str, floor: str) -> bool:
    """Is `level` at least as strong as `floor`? Unknown levels are weakest."""
    return _LEVEL_ORDER.get(level, -1) >= _LEVEL_ORDER.get(floor, 0)


def locator_signature(evidence: Evidence) -> tuple[tuple[str, str], ...]:
    """The provision this passage *is*, as comparable keys.

    Empty when no locator field is set, which is how a `locator_unavailable`
    chunk is kept out of every same-provision comparison rather than colliding
    with every other locatorless chunk in the set.
    """
    return tuple(
        (key, str(value).strip().upper())
        for key, value in sorted(evidence.locator_fields.items())
        if value not in (None, "", [])
    )


def ref_matches_locator(ref: Any, target: Evidence) -> bool:
    """Is `ref` — a Phase 5 :class:`LocatorRef` — the provision `target` *is*?"""
    signature = dict(locator_signature(target))
    if not signature:
        return False
    mapping = LOCATOR_FIELD.get(ref.kind)
    if not mapping:
        return False
    field_name, sub_fields = mapping
    if signature.get(field_name) != ref.number:
        return False
    if not ref.subdivisions:
        return True
    subs = [signature.get(name) for name in sub_fields]
    return list(ref.subdivisions) == [s for s in subs if s][:len(ref.subdivisions)]


def ref_matches_subdivision_of(ref: Any, target: Evidence) -> bool:
    """Is `ref` a subdivision of `target`, *and one `target` actually contains*?

    Phase 7.3 Q001: the evidence supplied was the whole of Section 3 of the
    Patents Act; the model wrote "section 3(p)", which is a real clause of that
    very passage; and `_scan_prose` reported it as a provision named by nothing
    supplied, because `ref_matches_locator` compares subdivisions exactly and
    the chunk's locator carries none. The answer was refused for correctly
    citing its own evidence.

    The rule is hierarchical but not permissive, and the second half is the
    whole of it: **a parent supports a child only when the child's marker is
    present in the parent's text.** Section 3 does not support Section 3(zz)
    merely by being Section 3. What makes this safe is that the containment is
    read off the corpus text rather than assumed from the numbering — the same
    move `_scan_prose`'s cross-reference arm already makes, and for the same
    reason: the test is whether the string came from the evidence.

    Two further restrictions keep this from becoming a descendant rule:

    * `target` must be the *whole* provision. A chunk that is itself Section
      3(p) is a sibling of Section 3(q), not its container, so a chunk carrying
      any subdivision of its own is never a parent here.
    * every subdivision the ref names must be present, not merely the first.

    Nothing about the citation changes. `build_citation` still copies the
    evidence record, so the reader is shown the corpus's own locator — Section
    3 — and never the model's `3(p)`. This function admits prose; it does not
    author a locator.
    """
    signature = dict(locator_signature(target))
    mapping = LOCATOR_FIELD.get(ref.kind)
    if not signature or not mapping or not ref.subdivisions:
        return False
    field_name, sub_fields = mapping
    if signature.get(field_name) != ref.number:
        return False
    if any(signature.get(name) for name in sub_fields):
        return False
    return all(subdivision_in_text(sub, target.text) for sub in ref.subdivisions)


def subdivision_in_text(subdivision: str, text: str) -> bool:
    """Does `text` carry the parenthesised marker `(p)` as its own token?

    Deliberately literal. Legal instruments print their clauses exactly this
    way — "(p) an invention which…" — so an exact parenthesised match is not a
    heuristic standing in for something better; it is the notation itself.
    Case-insensitive only because the identifier parser upper-cases what it
    reads, and `(p)` and `(P)` are the same clause.
    """
    return re.search(rf"\(\s*{re.escape(subdivision)}\s*\)", text or "",
                     re.IGNORECASE) is not None


def references_locator(text: str, target: Evidence) -> bool:
    """Does `text` name the provision `target` is?

    Uses the Phase 5 identifier parser rather than a second one, so "section
    3(p)", "sec. 3(p)" and "3(p) of the Act" are recognised here exactly as the
    boosted retriever recognises them.
    """
    return any(ref_matches_locator(ref, target) for ref in parse_identifiers(text))


def assess_evidence(query: str, evidence: Evidence,
                    policy: AnswerPolicy) -> EvidenceAssessment:
    """Answer the four questions about one passage. Decides nothing else."""
    query_terms = content_terms(query)
    overlap = tuple(sorted(query_terms & content_terms(evidence.text)))
    addresses = (not policy.require_relevance
                 or len(overlap) >= policy.relevance_min_overlap)

    authoritative = meets_evidence_floor(evidence.source_evidence_level,
                                         policy.minimum_evidence_level)
    current = evidence.temporal_status == CURRENT

    notes: list[str] = []
    if not evidence.citable:
        # Phase 3's invariant, restated at the point where it bites: retrievable
        # and readable, but never a citation.
        notes.append("not citable: " + ("locator unavailable"
                                        if evidence.locator_unavailable
                                        else "citability withheld upstream"))
    if not authoritative:
        notes.append(f"source evidence level {evidence.source_evidence_level} is "
                     f"below the floor {policy.minimum_evidence_level}")
    if evidence.temporal_status == STALE:
        notes.append("temporal status STALE: this text may have been amended")
    elif not current:
        notes.append(f"temporal status {evidence.temporal_status}")
    if not addresses:
        notes.append("shares no content terms with the query")

    admissible = bool(evidence.citable and authoritative and addresses)
    if admissible:
        role = "citable_authority"
    elif addresses:
        role = "context_only"
    else:
        role = "off_topic"

    return EvidenceAssessment(
        marker=evidence.marker,
        chunk_id=evidence.chunk_id,
        addresses_query=addresses,
        overlap_terms=overlap,
        authoritative=authoritative,
        source_evidence_level=evidence.source_evidence_level,
        current=current,
        temporal_status=evidence.temporal_status,
        citable=evidence.citable,
        admissible_as_citation=admissible,
        role=role,
        notes=tuple(notes),
    )


# -- conflict detection (Phase 7 §15) -----------------------------------------


def detect_conflicts(evidence_set: EvidenceSet) -> list[Conflict]:
    """Find passages that disagree about the same provision.

    Four deterministic detectors, and no legal reasoning. What is *not* covered
    is stated plainly in docs/ANSWER_ENGINE.md: two provisions that contradict
    each other in substance while sharing no locator and announcing no override
    are invisible here, and finding them would require the legal reasoning §15
    defers.
    """
    conflicts: list[Conflict] = []
    items = list(evidence_set)

    by_provision: dict[tuple[str, tuple[tuple[str, str], ...]], list[Evidence]] = {}
    for item in items:
        signature = locator_signature(item)
        if signature:
            by_provision.setdefault((item.document_id, signature), []).append(item)

    for (document_id, signature), group in sorted(by_provision.items()):
        if len(group) < 2:
            continue
        chunk_ids = tuple(sorted(e.chunk_id for e in group))
        rendered = ", ".join(f"{key}={value}" for key, value in signature)
        statuses = {e.temporal_status for e in group}
        versions = {e.source_version_id for e in group}
        as_of = {e.consolidated_as_of for e in group}
        if len(statuses) > 1:
            conflicts.append(Conflict(
                kind="temporal_divergence", chunk_ids=chunk_ids,
                detail=f"{document_id} {rendered} appears with different temporal "
                       f"statuses: {', '.join(sorted(str(s) for s in statuses))}"))
        if len(versions) > 1 or len(as_of) > 1:
            conflicts.append(Conflict(
                kind="version_divergence", chunk_ids=chunk_ids,
                detail=f"{document_id} {rendered} appears at different consolidated "
                       f"versions: {', '.join(sorted(str(v) for v in versions))}"))
        if len(statuses) == 1 and len(versions) <= 1 and len(as_of) <= 1:
            conflicts.append(Conflict(
                kind="same_provision_divergence", chunk_ids=chunk_ids,
                detail=f"{len(group)} separate passages claim to be "
                       f"{document_id} {rendered}"))

    # Two passages that render the same citation string are indistinguishable to
    # a reader even when the corpus knows they are different provisions. This is
    # not hypothetical: the Drugs and Cosmetics Rules, 1945 contain both Part VI
    # Rule 65 (conditions of licence) and Part II Rule 65 (a schedule entry), and
    # both cite as "Drugs and Cosmetics Rules, 1945 — Rule 65". The citation
    # object still resolves — it carries chunk_id and pages — but a footnote
    # does not, so the answer has to say so.
    by_citation: dict[str, list[Evidence]] = {}
    for item in items:
        by_citation.setdefault(item.citation.strip(), []).append(item)
    for citation, group in sorted(by_citation.items()):
        if len(group) > 1 and citation:
            conflicts.append(Conflict(
                kind="ambiguous_citation",
                chunk_ids=tuple(sorted(e.chunk_id for e in group)),
                detail=f"{len(group)} different passages both cite as "
                       f"{citation!r}; a reader cannot tell them apart from the "
                       f"citation alone"))

    for item in items:
        lowered = item.text.lower()
        markers = [marker for marker in OVERRIDE_MARKERS if marker in lowered]
        if not markers:
            continue
        for other in items:
            if other.chunk_id == item.chunk_id:
                continue
            if references_locator(item.text, other):
                conflicts.append(Conflict(
                    kind="textual_override",
                    chunk_ids=tuple(sorted((item.chunk_id, other.chunk_id))),
                    detail=f"{item.chunk_id} says '{markers[0]}' of the provision "
                           f"{other.chunk_id} is"))
    return conflicts


# -- the whole set ------------------------------------------------------------


def citation_cautions(citations: Sequence[Any]) -> tuple[str, ...]:
    """Cautions about the sources an answer *actually cites*.

    Deliberately not computable inside :func:`apply_policy`: policy runs before
    the provider, and which passages get cited is a fact that does not exist
    until the citation validator has produced them. Phase 7.7 published its
    first answer over one citation drawn from five supplied passages and told
    the reader "4 of 5 cited sources are marked STALE", because the count was
    the only one available at the time it was worded.

    Sources are deduplicated by `chunk_id`: two markers resolving to the same
    passage are one source, and counting them twice would overstate both halves
    of "n of m". Deterministic, and no model is consulted about it.
    """
    sources: dict[str, Any] = {}
    for citation in citations or ():
        sources.setdefault(citation.chunk_id, citation)
    cited = list(sources.values())
    if not cited:
        return ()

    cautions: list[str] = []
    total = len(cited)
    stale = [c for c in cited if c.temporal_status == STALE]
    if stale:
        if total == 1:
            head = "The cited source is marked stale."
        elif len(stale) == total == 2:
            head = "Both cited sources are marked stale."
        elif len(stale) == total:
            head = f"All {total} cited sources are marked stale."
        elif len(stale) == 1:
            head = f"1 of {total} cited sources is marked stale."
        else:
            head = f"{len(stale)} of {total} cited sources are marked stale."
        cautions.append(
            f"{head} The text may have been amended since the consolidated "
            f"version held in the corpus; verify against the current official "
            f"text.")

    weak = [c for c in cited if c.source_evidence_level in ("C", "D")]
    if weak:
        levels = "/".join(sorted({c.source_evidence_level for c in weak}))
        if total == 1:
            head = f"The cited source is at evidence level {levels}."
        elif len(weak) == total:
            head = f"All {total} cited sources are at evidence level {levels}."
        elif len(weak) == 1:
            head = f"1 of {total} cited sources is at evidence level {levels}."
        else:
            head = f"{len(weak)} of {total} cited sources are at evidence level {levels}."
        cautions.append(
            f"{head} The publisher of the underlying PDF could not be "
            f"established.")
    return tuple(cautions)


def apply_policy(query: str, evidence_set: EvidenceSet,
                 policy: AnswerPolicy | None = None) -> PolicyReport:
    """Assess every passage, detect conflicts, and decide about sufficiency."""
    policy = policy or AnswerPolicy()
    assessments = tuple(assess_evidence(query, item, policy) for item in evidence_set)
    conflicts = tuple(detect_conflicts(evidence_set))

    cautions: list[str] = []
    reasons: list[Failure] = []

    if not evidence_set:
        reasons.append(Failure(
            code="no_evidence",
            detail="no evidence was supplied to the answer engine"))
        return PolicyReport(assessments=assessments, conflicts=conflicts,
                            cautions=tuple(cautions),
                            insufficient_reasons=tuple(reasons))

    relevant = [a for a in assessments if a.addresses_query]
    citable = [a for a in assessments if a.admissible_as_citation]

    if not relevant:
        reasons.append(Failure(
            code="no_relevant_evidence",
            detail="no supplied passage shares any content term with the query"))

    if policy.require_citable_authority and not citable:
        blocked_by_citability = [a for a in assessments
                                 if a.addresses_query and not a.citable]
        blocked_by_level = [a for a in assessments
                            if a.addresses_query and a.citable and not a.authoritative]
        if blocked_by_citability and not blocked_by_level:
            detail = (f"{len(blocked_by_citability)} relevant passage(s) were "
                      f"supplied but none is citable; a non-citable chunk may be "
                      f"read but never presented as authority")
        elif blocked_by_level:
            detail = (f"no relevant passage meets the source evidence floor "
                      f"{policy.minimum_evidence_level}")
        else:
            detail = "no citable authoritative evidence was supplied"
        reasons.append(Failure(code="no_citable_authority", detail=detail))

    if citable:
        stale = [a for a in citable if a.temporal_status == STALE]
        if len(stale) == len(citable) and policy.stale_only_is_insufficient:
            reasons.append(Failure(
                code="only_stale_evidence",
                detail="every citable passage is marked STALE and the policy "
                       "requires current authority"))
        # The staleness and provenance cautions used to be raised here, over the
        # *citable supplied* passages, and worded as though they were about the
        # cited ones. Nothing caught it for four phases because every live answer
        # abstained and no caution was ever rendered; the first published answer,
        # Phase 7.7 Q027, said "4 of 5 cited sources are marked STALE" under a
        # single citation. This function runs before the provider does and cannot
        # know what will be cited, so the honest fix is that it does not say. See
        # `citation_cautions`, which the engine calls once the citations exist.

    statuses = {a.temporal_status for a in assessments if a.addresses_query}
    if len(statuses) > 1 and not conflicts:
        cautions.append(
            "The supplied evidence mixes current and superseded instruments; they "
            "address different provisions, so this is not flagged as a conflict.")

    if conflicts:
        kinds = ", ".join(sorted({c.kind for c in conflicts}))
        cautions.append(
            f"Conflicting evidence was detected ({kinds}). This system does not "
            f"decide which provision prevails; both are shown.")
        if policy.abstain_on_conflict:
            reasons.append(Failure(
                code="conflicting_evidence",
                detail=f"conflicting evidence detected ({kinds}) and the policy "
                       f"requires abstention"))

    return PolicyReport(assessments=assessments, conflicts=conflicts,
                        cautions=tuple(cautions), insufficient_reasons=tuple(reasons))
