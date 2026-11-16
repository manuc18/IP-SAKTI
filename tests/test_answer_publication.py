"""Issues 2 and 4: what actually reaches the page, and what must not.

Deterministic and offline. The provider is `ExtractiveMockProvider`, which
quotes corpus sentences verbatim and cannot invent one, so every assertion here
is about the *gates* — policy, citation validation, grounding, confidence —
rather than about a model's prose.

Two questions are asked of the same machinery:

**Does a conflict elsewhere in the evidence set demote a clean answer?** It used
to. `apply_policy` detects conflicts over every supplied passage, and two
chunks of an unrelated instrument that render the same citation string are a
conflict — so an answer citing one clean provision was published LOW because of
a pair it never mentioned.

**Does "how to trademark my own firm" abstain when it should not?** Only if the
retrieved evidence really does carry the procedure. The pair of tests below
supplies each case explicitly, because the interesting claim is not that the
system answers but that it answers *exactly when the evidence is there*.
"""
from __future__ import annotations

import pytest

from ai.answering.base import GroundedAnswerEngine, as_evidence_set
from ai.answering.models import HIGH, LOW, MODERATE, EvidenceSet
from ai.answering.policy import AnswerPolicy, apply_policy, detect_conflicts
from ai.answering.providers import ExtractiveMockProvider
from ai.retrieval.base import RetrievalResult

TM_PROCEDURE = (
    "18. Application for registration.—(1) Any person claiming to be the "
    "proprietor of a trade mark used or proposed to be used by him, who is "
    "desirous of registering it, shall apply in writing to the Registrar in "
    "the prescribed manner for the registration of his trade mark. (2) A "
    "single application may be made for registration of a trade mark for "
    "different classes of goods and services and fee payable therefor shall "
    "be in respect of each such class of goods or services."
)

TM_FORM = (
    "23. Form and signing of application.—(1) An application to the Registrar "
    "for registration of a trade mark shall be signed by the applicant or his "
    "agent. (2) An application for registration of a trade mark shall be filed "
    "in the office of the Trade Marks Registry within whose territorial limits "
    "the principal place of business in India of the applicant is situated."
)

#: Trade-mark evidence that answers a different question entirely. Retrieval
#: really does return this for "how to trademark my own firm": the fee schedule
#: and the rule on altering a registered mark both talk about trade marks and
#: neither one tells anybody how to register one.
TM_IRRELEVANT = (
    "102. Alteration of a registered trade mark.—An application for alteration "
    "of a registered trade mark under section 59 shall be made in Form TM-P "
    "and shall be accompanied by the prescribed fee."
)


def result(chunk_id, document_id, text, locator, rank, *, section=None,
           rule=None, article=None, citable=True, level="B", status="CURRENT",
           version="v1", as_of="2024-01-01") -> RetrievalResult:
    fields = {key: None for key in (
        "part", "chapter", "section", "subsection", "clause", "rule", "sub_rule",
        "regulation", "article", "paragraph", "subparagraph", "schedule", "form",
        "annexure", "item")}
    fields.update({"section": section, "rule": rule, "article": article})
    return RetrievalResult(
        chunk_id=chunk_id, document_id=document_id, score=1.0 - rank * 0.01,
        rank=rank, text=text, locator=locator, citable=citable,
        metadata={"domain": "trademarks", "jurisdiction": "IN",
                  "document_type": "act", "temporal_status": status,
                  "source_evidence_level": level,
                  "source_status": "SOURCE_UNAVAILABLE",
                  "citability_reason": None, "locator_unavailable": False,
                  "locator_fields": fields, "physical_pages": [1],
                  "title": document_id, "short_title": document_id,
                  "heading": locator, "unit_type": "section",
                  "structural_path": locator, "official_source_url": None,
                  "consolidated_as_of": as_of, "source_version_id": version,
                  "source_filename": "x.pdf", "source_sha256": "0" * 64})


def engine(**policy_kwargs) -> GroundedAnswerEngine:
    return GroundedAnswerEngine(provider=ExtractiveMockProvider(),
                                policy=AnswerPolicy(**policy_kwargs))


QUESTION = "How do I apply for registration of a trade mark for my firm?"


# -- issue 4: answer when the evidence carries the procedure ------------------

def test_the_registration_procedure_is_published_with_citations():
    """Evidence that really answers the question must not abstain.

    Not social, policy-sufficient, generation-eligible, citations resolve,
    grounding passes — asserted in that order, so a failure names its own stage.
    """
    from app.intent import classify

    assert classify(QUESTION) is None, "a real legal question must not be social"

    evidence = as_evidence_set([
        result("TM::s18", "IN-TM-ACT-1999", TM_PROCEDURE,
               "Trade Marks Act, 1999 — Section 18", 1, section="18"),
        result("TM::r23", "IN-TM-RULES-2017", TM_FORM,
               "Trade Marks Rules, 2017 — Rule 23", 2, rule="23"),
    ])

    report = apply_policy(QUESTION, evidence, AnswerPolicy())
    assert report.insufficient_reasons == (), "the evidence is sufficient"
    assert any(a.addresses_query for a in report.assessments)
    assert any(a.admissible_as_citation for a in report.assessments)

    answer = engine().answer(QUESTION, evidence)
    assert answer.insufficient_evidence is False
    assert answer.grounded is True
    assert answer.citations, "a published answer must carry at least one citation"
    assert answer.unsupported_claims == ()
    assert all(c.chunk_id in {"TM::s18", "TM::r23"} for c in answer.citations)


def test_trade_mark_evidence_that_does_not_carry_the_procedure_still_abstains():
    """The negative half. Abstention is the correct answer here and must stay.

    The passage is about trade marks, is citable, is current and is level B —
    everything except responsive. Nothing in the fix may turn this into an
    answer.
    """
    evidence = as_evidence_set([
        result("TM::r102", "IN-TM-RULES-2017", TM_IRRELEVANT,
               "Trade Marks Rules, 2017 — Rule 102", 1, rule="102"),
    ])
    answer = engine().answer(
        "What are the grounds for revocation of a patent for non-working?",
        evidence)
    assert answer.insufficient_evidence is True
    assert answer.citations == ()
    assert answer.confidence not in (HIGH, MODERATE)


def test_an_empty_evidence_set_abstains_rather_than_answering():
    answer = engine().answer(QUESTION, EvidenceSet.from_results([]))
    assert answer.insufficient_evidence is True
    assert answer.citations == ()


# -- issue 2: the conflict that belongs to somebody else ----------------------

def _set_with_a_foreign_conflict() -> EvidenceSet:
    """The cited provision, plus an unrelated pair sharing a citation string.

    The two TRIPS articles carry no word of the question, so the extractive
    provider cannot quote them and the answer cannot cite them — which is
    exactly the situation the bug produced in production.
    """
    return as_evidence_set([
        result("TM::s18", "IN-TM-ACT-1999", TM_PROCEDURE,
               "Trade Marks Act, 1999 — Section 18", 1, section="18"),
        result("TRIPS::a22", "WTO-TRIPS",
               "Members shall provide the legal means for interested parties to "
               "prevent designation of a geographical origin.",
               "TRIPS — Article 22", 2, article="22"),
        result("TRIPS::a23", "WTO-TRIPS",
               "Each Member shall provide the legal means for interested parties "
               "to prevent use of a geographical origin for wines.",
               "TRIPS — Article 22", 3, article="22"),
    ])


def test_the_foreign_pair_really_is_detected_as_a_conflict():
    """Without this the next test would pass for the wrong reason."""
    conflicts = detect_conflicts(_set_with_a_foreign_conflict())
    assert conflicts, "the two articles share a citation string"
    involved = set().union(*[set(c.chunk_ids) for c in conflicts])
    assert "TM::s18" not in involved


def test_a_conflict_touching_no_cited_chunk_does_not_demote_the_answer():
    answer = engine().answer(QUESTION, _set_with_a_foreign_conflict())
    assert answer.insufficient_evidence is False
    assert {c.chunk_id for c in answer.citations} == {"TM::s18"}
    # Detected, and said so — the caution is not suppressed.
    assert answer.conflict_detected is True
    assert any("Conflicting evidence" in c for c in answer.cautions)
    # …but the level reflects the cited source, which is level B and CURRENT.
    assert answer.confidence == MODERATE
    assert not any(f["code"] == "conflicting_provisions"
                   for f in answer.confidence_report["factors"])


def test_a_conflict_that_does_touch_a_cited_chunk_still_forces_low():
    """The safety half. A conflict about the passage the answer rests on stays.

    Both chunks carry the same citation string *and* the same wording as the
    question, so the provider quotes one of them and the conflict is no longer
    somebody else's.
    """
    evidence = as_evidence_set([
        result("TM::s18a", "IN-TM-ACT-1999", TM_PROCEDURE,
               "Trade Marks Act, 1999 — Section 18", 1, section="18"),
        result("TM::s18b", "IN-TM-ACT-1999",
               TM_PROCEDURE.replace("in writing", "in the prescribed form"),
               "Trade Marks Act, 1999 — Section 18", 2, section="18"),
    ])
    conflicts = detect_conflicts(evidence)
    involved = set().union(*[set(c.chunk_ids) for c in conflicts])
    assert {"TM::s18a", "TM::s18b"} <= involved

    answer = engine().answer(QUESTION, evidence)
    assert answer.insufficient_evidence is False
    assert answer.citations
    assert answer.confidence == LOW
    assert any(f["code"] == "conflicting_provisions"
               for f in answer.confidence_report["factors"])


def test_stale_and_weak_provenance_still_lower_confidence_after_the_fix():
    """Nothing in the conflict scoping may reach the other source-side rules."""
    stale = as_evidence_set([
        result("TM::s18", "IN-TM-ACT-1999", TM_PROCEDURE,
               "Trade Marks Act, 1999 — Section 18", 1, section="18",
               status="STALE")])
    answer = engine(stale_only_is_insufficient=False).answer(QUESTION, stale)
    if not answer.insufficient_evidence:
        assert answer.confidence == LOW

    weak = as_evidence_set([
        result("TM::s18", "IN-TM-ACT-1999", TM_PROCEDURE,
               "Trade Marks Act, 1999 — Section 18", 1, section="18", level="C")])
    answer = engine(minimum_evidence_level="C").answer(QUESTION, weak)
    if not answer.insufficient_evidence:
        assert answer.confidence == LOW
