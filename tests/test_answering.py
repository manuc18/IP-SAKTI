"""Phase 7 — the answer engine, its policy, and the citation customs officer.

Every test here is deterministic and offline. No model is loaded, no index is
opened, no network call is made: the fixtures are seven JSON files of real
corpus passages and everything else is arithmetic over them. That is a property
of the architecture rather than of the tests, and one test asserts it directly
(`test_answering_package_loads_no_ml_dependency`).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from ai.answering.base import GroundedAnswerEngine, as_evidence_set
from ai.answering.citation_validator import build_citation, validate_citations
from ai.answering.grounding import (
    check_grounding,
    extract_markers,
    is_claim,
    split_sentences,
)
from ai.answering.models import (
    Answer,
    Citation,
    Evidence,
    EvidenceSet,
    SourceDocument,
)
from ai.answering.policy import (
    AnswerPolicy,
    apply_policy,
    assess_evidence,
    content_terms,
    detect_conflicts,
    locator_signature,
    meets_evidence_floor,
)
from ai.answering.providers import (
    ANSWER_INSTRUCTIONS,
    AnswerProvider,
    ExternalLLMProvider,
    ExtractiveMockProvider,
    ProviderDraft,
    build_request,
)
from ai.answering.render import (
    INSUFFICIENT_EVIDENCE_MESSAGE,
    render_answer,
    render_citation,
)
from ai.retrieval.base import RetrievalResult

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXTURE_DIR = os.path.join(PROJECT_ROOT, "tests", "fixtures", "answering")


# -- fixture plumbing ---------------------------------------------------------


def load_fixture(name: str) -> dict:
    with open(os.path.join(FIXTURE_DIR, f"{name}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def as_results(rows: list[dict]) -> list[RetrievalResult]:
    return [RetrievalResult(chunk_id=row["chunk_id"], document_id=row["document_id"],
                            score=row["score"], rank=row["rank"], text=row["text"],
                            locator=row["locator"], citable=row["citable"],
                            metadata=row["metadata"]) for row in rows]


def evidence_for(name: str) -> EvidenceSet:
    return EvidenceSet.from_results(as_results(load_fixture(name)["results"]))


def query_for(name: str) -> str:
    return load_fixture(name)["query"]


def make_evidence(marker: int = 1, **overrides) -> Evidence:
    """A minimal well-formed evidence record, for the cases fixtures don't cover."""
    base = dict(
        marker=marker, chunk_id=f"DOC::unit-{marker}", document_id="DOC",
        document_name="Test Act, 2000", citation="Test Act, 2000 — Section 5",
        text="A registered proprietor may assign the mark to another person.",
        citable=True, locator_unavailable=False,
        locator_fields={"section": "5"}, temporal_status="CURRENT",
        source_evidence_level="A", physical_pages=(7,),
        source_filename="test.pdf", source_sha256="abc123",
        official_source_url="https://example.invalid/test.pdf",
    )
    base.update(overrides)
    return Evidence(**base)


class StubProvider(AnswerProvider):
    """Returns whatever text the test wants, so bad drafts can be exercised."""

    name = "stub"

    def __init__(self, text: str):
        self.text = text

    def generate(self, request):
        return ProviderDraft(text=self.text, provider=self.name)


# -- 1. evidence and answer object validation ---------------------------------


def test_evidence_set_numbers_from_one_in_retrieval_order():
    evidence = evidence_for("multiple_citations")
    assert evidence.markers == (1, 2, 3)
    assert evidence.items[0].retrieval_rank == 1


def test_evidence_copies_provenance_and_decides_nothing():
    item = evidence_for("citable_statutory").items[0]
    assert item.chunk_id == "IN-PATENTS-ACT-1970::chapter-ii-section-3"
    assert item.citable is True
    assert item.temporal_status == "CURRENT"
    assert item.source_evidence_level == "B"
    assert item.physical_pages == (11, 12)
    assert item.source_filename and item.source_sha256


def test_evidence_fingerprint_is_order_sensitive():
    evidence = evidence_for("multiple_citations")
    reversed_set = EvidenceSet(tuple(reversed(evidence.items)))
    assert evidence.fingerprint != reversed_set.fingerprint


def test_empty_evidence_set_is_falsey_and_has_a_stable_fingerprint():
    empty = evidence_for("empty")
    assert not empty and len(empty) == 0
    assert empty.fingerprint == EvidenceSet(()).fingerprint


def test_answer_to_dict_carries_the_phase7_contract():
    record = GroundedAnswerEngine().answer(
        query_for("citable_statutory"), evidence_for("citable_statutory")).to_dict()
    for key in ("answer", "citations", "confidence", "grounded",
                "insufficient_evidence"):
        assert key in record
    assert isinstance(record["citations"], list)


def test_locator_text_strips_the_repeated_document_name():
    assert make_evidence().locator_text == "Section 5"


# -- 2. citation resolution and provenance ------------------------------------


def test_citation_is_built_from_evidence_not_from_prose():
    citation = build_citation(make_evidence())
    assert citation.chunk_id == "DOC::unit-1"
    assert citation.pages == (7,)
    assert citation.source_document.sha256 == "abc123"
    assert citation.source_document.filename == "test.pdf"


def test_citation_retains_everything_a_ui_needs_to_open_the_source():
    citation = build_citation(evidence_for("citable_statutory").items[0])
    record = citation.to_dict()
    assert record["chunk_id"] and record["document_id"] and record["locator"]
    assert record["pages"] == [11, 12]
    assert record["source_document"]["filename"]
    assert record["source_document"]["sha256"]
    assert record["source_evidence_level"] == "B"
    assert record["temporal_status"] == "CURRENT"


def test_absent_source_url_stays_absent_and_is_never_guessed():
    citation = build_citation(make_evidence(official_source_url=None))
    assert citation.source_document.official_source_url is None


def test_marker_resolves_to_supplied_evidence():
    evidence = EvidenceSet((make_evidence(1),))
    outcome = validate_citations("A proprietor may assign the mark [1].", evidence)
    assert outcome.ok
    assert [c.chunk_id for c in outcome.citations] == ["DOC::unit-1"]


# -- 3. fabricated and unsupported citations (§10, §11) -----------------------


def test_marker_outside_the_supplied_set_is_rejected():
    evidence = EvidenceSet((make_evidence(1),))
    outcome = validate_citations("The mark may be assigned [4].", evidence)
    assert not outcome.ok
    assert [f.code for f in outcome.failures] == ["unknown_marker"]
    assert outcome.citations == ()


def test_validation_failure_yields_no_citations_at_all():
    """No partial acceptance: a failed validation is not silently repaired."""
    evidence = EvidenceSet((make_evidence(1),))
    outcome = validate_citations("Assignment is permitted [1] and so is this [9].",
                                 evidence)
    assert not outcome.ok
    assert outcome.citations == ()


def test_invented_provision_in_prose_is_caught():
    evidence = EvidenceSet((make_evidence(1),))
    outcome = validate_citations(
        "Section 45 of the Act bars this [1].", evidence)
    assert not outcome.ok
    assert "unsupported_locator_in_prose" in {f.code for f in outcome.failures}


def test_cross_reference_inside_supplied_text_is_not_an_invention():
    """The corpus cites itself constantly; quoting that is not fabrication."""
    evidence = EvidenceSet((make_evidence(
        1, text="A person may apply under section 3(2) of the Act in Form 7."),))
    outcome = validate_citations(
        "A person may apply under section 3(2) of the Act in Form 7 [1].", evidence)
    assert outcome.ok, [f.to_dict() for f in outcome.failures]


def test_supplied_provisions_own_locator_is_supported_in_prose():
    evidence = EvidenceSet((make_evidence(1),))
    outcome = validate_citations("Section 5 permits assignment [1].", evidence)
    assert outcome.ok, [f.to_dict() for f in outcome.failures]


def test_url_in_the_answer_text_is_a_failure():
    evidence = EvidenceSet((make_evidence(1),))
    outcome = validate_citations(
        "See https://example.com/patents for the text [1].", evidence)
    assert not outcome.ok
    assert "fabricated_url" in {f.code for f in outcome.failures}


def test_citations_cannot_refer_to_evidence_outside_the_engine_input():
    supplied = EvidenceSet((make_evidence(1),))
    swapped = EvidenceSet((make_evidence(1, chunk_id="OTHER::unit-1"),))
    outcome = validate_citations("Assignment is permitted [1].", swapped,
                                 expected_fingerprint=supplied.fingerprint)
    assert not outcome.ok
    assert [f.code for f in outcome.failures] == ["evidence_not_supplied"]


# -- 4. citable enforcement (§7) ----------------------------------------------


def test_non_citable_evidence_can_never_become_a_citation():
    evidence = EvidenceSet((make_evidence(1, citable=False,
                                          locator_unavailable=True,
                                          locator_fields={}, citation="unlocated"),))
    outcome = validate_citations("The rules were published in the Gazette [1].",
                                 evidence)
    assert not outcome.ok
    assert "non_citable_evidence" in {f.code for f in outcome.failures}


def test_non_citable_only_evidence_makes_the_engine_abstain():
    answer = GroundedAnswerEngine().answer(query_for("non_citable_only"),
                                           evidence_for("non_citable_only"))
    assert answer.insufficient_evidence
    assert answer.citations == ()
    assert "no_citable_authority" in {f.code
                                      for f in answer.insufficient_evidence_reasons}


def test_non_citable_evidence_is_still_retrievable_and_readable():
    """Phase 3's invariant is 'not quotable', not 'not visible'."""
    item = evidence_for("non_citable_only").items[0]
    assert item.citable is False
    assert item.text.strip()


# -- 5. locator handling (§10 rule 4) -----------------------------------------


def test_locator_unavailable_and_a_rendered_locator_is_a_corrupt_record():
    answer = GroundedAnswerEngine().answer(query_for("locator_inconsistent"),
                                           evidence_for("locator_inconsistent"))
    assert answer.insufficient_evidence
    assert "inconsistent_locator_state" in {f.code
                                            for f in answer.validation_failures}


def test_missing_locator_without_the_flag_is_rejected():
    evidence = EvidenceSet((make_evidence(1, citation="Test Act, 2000",
                                          locator_fields={}),))
    outcome = validate_citations("Assignment is permitted [1].", evidence)
    assert not outcome.ok
    assert "missing_locator" in {f.code for f in outcome.failures}


def test_locator_signature_is_empty_when_no_locator_field_is_set():
    assert locator_signature(make_evidence(locator_fields={})) == ()


# -- 6. page handling (§10 rule 7, §17) ---------------------------------------


def test_citation_without_resolvable_pages_is_rejected():
    evidence = EvidenceSet((make_evidence(1, physical_pages=()),))
    outcome = validate_citations("Assignment is permitted [1].", evidence)
    assert not outcome.ok
    assert "unresolvable_pages" in {f.code for f in outcome.failures}


def test_negative_page_number_is_rejected():
    evidence = EvidenceSet((make_evidence(1, physical_pages=(0,)),))
    outcome = validate_citations("Assignment is permitted [1].", evidence)
    assert "unresolvable_pages" in {f.code for f in outcome.failures}


def test_contiguous_pages_render_as_a_range():
    citation = build_citation(make_evidence(physical_pages=(11, 12, 13)))
    assert render_citation(citation).endswith("pp. 11–13")


def test_non_contiguous_pages_render_as_a_list():
    citation = build_citation(make_evidence(physical_pages=(11, 40)))
    assert render_citation(citation).endswith("pp. 11, 40")


# -- 7. the four policy axes (§6) ---------------------------------------------


def test_the_four_axes_are_assessed_separately():
    item = make_evidence(citable=True, source_evidence_level="C",
                         temporal_status="STALE")
    assessment = assess_evidence("assign the mark", item, AnswerPolicy())
    assert assessment.citable is True
    assert assessment.authoritative is True     # C meets the default floor
    assert assessment.current is False
    assert assessment.addresses_query is True


def test_authority_floor_is_ordinal():
    assert meets_evidence_floor("A", "C")
    assert meets_evidence_floor("C", "C")
    assert not meets_evidence_floor("D", "C")
    assert not meets_evidence_floor("B", "A")


def test_level_b_and_c_evidence_is_labelled_not_discarded():
    """Phase 7 §6 forbids dropping weak-provenance evidence outright."""
    report = apply_policy("assign the mark",
                          EvidenceSet((make_evidence(1, source_evidence_level="C"),)),
                          AnswerPolicy())
    assert not report.insufficient
    assert report.citable_markers == (1,)
    # The label itself moved in Phase 7.8: `apply_policy` runs before the
    # provider and cannot know which passages get cited, so the caution that
    # says "cited source" is now written by `citation_cautions` once they
    # exist. What this test is about — level C evidence is labelled rather than
    # discarded — is unchanged, and both halves are asserted.
    assert not any("cited source" in caution.lower()
                   for caution in report.cautions)
    from ai.answering.citation_validator import build_citation
    from ai.answering.policy import citation_cautions

    labelled = citation_cautions([build_citation(
        make_evidence(1, source_evidence_level="C"))])
    assert any("evidence level" in caution for caution in labelled)


def test_evidence_below_the_floor_is_context_only_not_a_citation():
    report = apply_policy("assign the mark",
                          EvidenceSet((make_evidence(1, source_evidence_level="D"),)),
                          AnswerPolicy(minimum_evidence_level="C"))
    assert report.assessments[0].role == "context_only"
    assert report.citable_markers == ()


def test_off_topic_evidence_is_screened_out():
    item = make_evidence(1, text="Zebras graze quietly on the veldt.")
    assessment = assess_evidence("patent revocation grounds", item, AnswerPolicy())
    assert assessment.addresses_query is False
    assert assessment.role == "off_topic"


def test_content_terms_keep_legal_identifiers_intact():
    assert "3(p)" in content_terms("what does section 3(p) exclude")
    assert "the" not in content_terms("what does section 3(p) exclude")


# -- 8. stale evidence (§8) ---------------------------------------------------


def test_stale_only_evidence_answers_with_a_caution_by_default():
    answer = GroundedAnswerEngine().answer(query_for("stale_only"),
                                           evidence_for("stale_only"))
    assert not answer.insufficient_evidence
    assert answer.citations
    # Phase 7.8 rewrote this caution to count the sources actually cited; the
    # word is now lower case, and the count is 1 rather than the size of the
    # supplied set.
    assert any("stale" in caution.lower() for caution in answer.cautions)
    assert answer.confidence == "LOW"


def test_stale_citation_is_labelled_stale_when_rendered():
    answer = GroundedAnswerEngine().answer(query_for("stale_only"),
                                           evidence_for("stale_only"))
    assert "[STALE]" in render_answer(answer)


def test_stale_only_can_be_made_insufficient_by_policy():
    engine = GroundedAnswerEngine(policy=AnswerPolicy(stale_only_is_insufficient=True))
    answer = engine.answer(query_for("stale_only"), evidence_for("stale_only"))
    assert answer.insufficient_evidence
    assert "only_stale_evidence" in {f.code
                                     for f in answer.insufficient_evidence_reasons}


def test_no_temporal_category_is_invented():
    from ai.answering.models import TEMPORAL_STATUSES

    assert set(TEMPORAL_STATUSES) == {"CURRENT", "STALE", "UNKNOWN"}


# -- 9. insufficient evidence (§14) -------------------------------------------


def test_no_evidence_abstains_and_never_answers_from_model_knowledge():
    answer = GroundedAnswerEngine().answer(query_for("empty"), evidence_for("empty"))
    assert answer.insufficient_evidence
    assert answer.answer == INSUFFICIENT_EVIDENCE_MESSAGE
    assert answer.citations == ()
    assert answer.confidence == "NONE"


def test_no_relevant_evidence_abstains():
    evidence = EvidenceSet((make_evidence(1, text="Zebras graze on the veldt."),))
    answer = GroundedAnswerEngine().answer("patent revocation grounds", evidence)
    assert answer.insufficient_evidence
    assert "no_relevant_evidence" in {f.code
                                      for f in answer.insufficient_evidence_reasons}


def test_provider_returning_nothing_abstains():
    engine = GroundedAnswerEngine(provider=StubProvider(""))
    answer = engine.answer(query_for("citable_statutory"),
                           evidence_for("citable_statutory"))
    assert answer.insufficient_evidence
    assert "provider_abstained" in {f.code
                                    for f in answer.insufficient_evidence_reasons}


def test_citation_validation_failure_abstains_rather_than_publishing():
    engine = GroundedAnswerEngine(
        provider=StubProvider("Section 45 of the Act bars this [1]."))
    answer = engine.answer(query_for("citable_statutory"),
                           evidence_for("citable_statutory"))
    assert answer.insufficient_evidence
    assert "citation_validation_failed" in {
        f.code for f in answer.insufficient_evidence_reasons}
    assert answer.citations == ()


def test_uncited_claim_abstains_under_the_default_policy():
    engine = GroundedAnswerEngine(
        provider=StubProvider("Inventions may be excluded from patentability."))
    answer = engine.answer(query_for("citable_statutory"),
                           evidence_for("citable_statutory"))
    assert answer.insufficient_evidence
    assert "ungrounded_answer" in {f.code
                                   for f in answer.insufficient_evidence_reasons}


def test_insufficient_answers_explain_themselves_when_rendered():
    answer = GroundedAnswerEngine().answer(query_for("empty"), evidence_for("empty"))
    rendered = render_answer(answer)
    assert INSUFFICIENT_EVIDENCE_MESSAGE in rendered
    assert "Why:" in rendered
    assert "Source(s)" not in rendered


# -- 10. conflicting evidence (§15) -------------------------------------------


def test_identical_citation_strings_are_flagged_as_ambiguous():
    answer = GroundedAnswerEngine().answer(query_for("conflicting"),
                                           evidence_for("conflicting"))
    assert answer.conflict_detected
    assert len(answer.conflict_chunks) == 2
    assert "ambiguous_citation" in json.dumps(answer.diagnostics)


def test_conflict_is_exposed_not_resolved():
    answer = GroundedAnswerEngine().answer(query_for("conflicting"),
                                           evidence_for("conflicting"))
    assert not answer.insufficient_evidence
    assert len(answer.citations) == 2       # both sides shown, neither preferred
    assert any("does not decide" in caution for caution in answer.cautions)


def test_same_provision_with_different_temporal_status_is_a_conflict():
    conflicts = detect_conflicts(EvidenceSet((
        make_evidence(1, chunk_id="DOC::a", temporal_status="CURRENT"),
        make_evidence(2, chunk_id="DOC::b", temporal_status="STALE",
                      citation="Test Act, 2000 — Section 5 (2019)"),
    )))
    assert "temporal_divergence" in {c.kind for c in conflicts}


def test_textual_override_between_supplied_passages_is_a_conflict():
    conflicts = detect_conflicts(EvidenceSet((
        make_evidence(1, chunk_id="DOC::a", locator_fields={"section": "67"},
                      citation="Test Act, 2000 — Section 67",
                      text="Section 67 was omitted by the Amendment Act, 2026."),
        make_evidence(2, chunk_id="DOC::b", locator_fields={"section": "67"},
                      citation="Test Act, 2000 — Section 67 (as enacted)",
                      text="67. Any person who contravenes this Act shall be liable."),
    )))
    assert {"textual_override", "same_provision_divergence"} & {c.kind
                                                                for c in conflicts}


def test_different_provisions_with_different_statuses_are_not_a_conflict():
    """A flag that fires on every mixed set carries no information."""
    report = apply_policy("assignment of a mark", EvidenceSet((
        make_evidence(1, chunk_id="DOC::a", locator_fields={"section": "5"},
                      temporal_status="CURRENT"),
        make_evidence(2, chunk_id="DOC::b", locator_fields={"section": "6"},
                      citation="Test Act, 2000 — Section 6",
                      temporal_status="STALE"),
    )))
    assert report.conflicts == ()
    assert any("mixes current and superseded" in c for c in report.cautions)


def test_conflict_can_be_made_an_abstention_by_policy():
    engine = GroundedAnswerEngine(policy=AnswerPolicy(abstain_on_conflict=True))
    answer = engine.answer(query_for("conflicting"), evidence_for("conflicting"))
    assert answer.insufficient_evidence
    assert "conflicting_evidence" in {f.code
                                      for f in answer.insufficient_evidence_reasons}


# -- 11. grounding (§12) ------------------------------------------------------


def test_every_claim_carrying_a_marker_is_grounded():
    evidence = EvidenceSet((make_evidence(1),))
    report = check_grounding("A proprietor may assign the mark [1].", evidence)
    assert report.grounded
    assert report.citation_support
    assert report.supported_claim_rate == 1.0


def test_unsupported_claim_is_detected():
    evidence = EvidenceSet((make_evidence(1),))
    report = check_grounding(
        "A proprietor may assign the mark [1]. Assignment requires a stamp duty "
        "of five hundred rupees.", evidence)
    assert not report.grounded
    assert len(report.unsupported_claims) == 1


def test_uncited_evidence_is_reported():
    evidence = EvidenceSet((make_evidence(1), make_evidence(2, chunk_id="DOC::two")))
    report = check_grounding("A proprietor may assign the mark [1].", evidence)
    assert report.uncited_evidence == ("DOC::two",)


def test_unresolved_marker_breaks_citation_support():
    evidence = EvidenceSet((make_evidence(1),))
    report = check_grounding("The mark may be assigned [7].", evidence)
    assert not report.citation_support
    assert report.unresolved_markers == (7,)


def test_weak_grounding_is_advisory_not_disqualifying():
    evidence = EvidenceSet((make_evidence(1, text="Zebras graze on the veldt."),))
    report = check_grounding("Kangaroos hop across plains [1].", evidence)
    assert report.weakly_grounded
    assert report.grounded          # the citation resolves; the heuristic advises


def test_sentence_splitting_keeps_legal_abbreviations_intact():
    assert split_sentences("See s. 3 of the Act. It applies.") == [
        "See s. 3 of the Act.", "It applies."]


def test_markers_are_extracted_in_order_and_deduplicated():
    assert extract_markers("a [2] b [1] c [2]") == [2, 1]


def test_framing_sentences_are_not_treated_as_claims():
    assert not is_claim("The supplied evidence does not address this.")
    assert is_claim("A proprietor may assign the registered mark.")


# -- 12. providers (§3, §21) --------------------------------------------------


def test_mock_provider_is_deterministic():
    query, evidence = query_for("stale_only"), evidence_for("stale_only")
    policy = AnswerPolicy()
    request = build_request(query, evidence, policy)
    first = ExtractiveMockProvider().generate(request).text
    second = ExtractiveMockProvider().generate(request).text
    assert first == second and first


def test_provider_payload_never_contains_a_chunk_id():
    evidence = evidence_for("multiple_citations")
    request = build_request(query_for("multiple_citations"), evidence, AnswerPolicy())
    rendered = json.dumps(request.to_dict())
    for item in evidence:
        assert item.chunk_id not in rendered


def test_provider_payload_carries_the_labels_the_contract_requires():
    request = build_request(query_for("stale_only"), evidence_for("stale_only"),
                            AnswerPolicy())
    passage = request.passages[0]
    assert passage.temporal_status == "STALE"
    assert passage.source_evidence_level == "A"
    assert passage.citable is True
    assert request.instructions == ANSWER_INSTRUCTIONS


def test_mock_provider_grounds_every_sentence_it_emits():
    query, evidence = query_for("multiple_citations"), evidence_for("multiple_citations")
    draft = ExtractiveMockProvider().generate(
        build_request(query, evidence, AnswerPolicy()))
    assert check_grounding(draft.text, evidence).grounded


def test_mock_provider_quotes_only_the_corpus():
    evidence = evidence_for("citable_statutory")
    draft = ExtractiveMockProvider().generate(
        build_request(query_for("citable_statutory"), evidence, AnswerPolicy()))
    stripped = draft.text.replace(" [1]", "").rstrip(".")
    assert stripped.split(".")[0] in " ".join(evidence.items[0].text.split())


def test_external_provider_is_declared_but_inert():
    provider = ExternalLLMProvider(model="some-model")
    assert provider.config()["implemented"] is False
    with pytest.raises(NotImplementedError):
        provider.generate(build_request("q", evidence_for("empty"), AnswerPolicy()))


def test_engine_is_provider_agnostic():
    engine = GroundedAnswerEngine(provider=StubProvider(
        "The following are not inventions within the meaning of this Act [1]."))
    answer = engine.answer(query_for("citable_statutory"),
                           evidence_for("citable_statutory"))
    assert not answer.insufficient_evidence
    assert answer.provider == "stub"


# -- 13. the engine boundary (§1, §2, §25) ------------------------------------


def test_engine_accepts_retrieval_results_directly():
    results = as_results(load_fixture("citable_statutory")["results"])
    answer = GroundedAnswerEngine().answer(query_for("citable_statutory"), results)
    assert not answer.insufficient_evidence


def test_engine_performs_no_retrieval():
    assert GroundedAnswerEngine().config()["retrieval_performed_here"] is False
    source = open(os.path.join(PROJECT_ROOT, "ai", "answering", "base.py"),
                  encoding="utf-8").read()
    for forbidden in ("load_chunks", "BM25Retriever", "DenseRetriever", "CandidateGenerator"):
        assert forbidden not in source


def test_answer_signature_admits_no_labels():
    import inspect

    parameters = list(inspect.signature(GroundedAnswerEngine.answer).parameters)
    assert parameters == ["self", "query", "evidence"]


def test_answering_package_loads_no_ml_dependency():
    """Phase 7 §25: the default path must not drag in torch or an encoder."""
    code = ("import sys, ai.answering, ai.answering.base;"
            "print([m for m in ('torch','sentence_transformers','transformers')"
            " if m in sys.modules])")
    output = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT,
                            capture_output=True, text=True, check=True)
    assert output.stdout.strip() == "[]"


def test_evidence_fingerprint_is_recorded_on_the_answer():
    evidence = evidence_for("citable_statutory")
    answer = GroundedAnswerEngine().answer(query_for("citable_statutory"), evidence)
    assert answer.evidence_fingerprint == evidence.fingerprint


# -- 14. rendering (§13, §16) -------------------------------------------------


def test_renderer_produces_the_canonical_shape():
    answer = GroundedAnswerEngine().answer(query_for("citable_statutory"),
                                           evidence_for("citable_statutory"))
    rendered = render_answer(answer)
    assert rendered.startswith("Answer\n")
    assert "Source(s)" in rendered
    assert "[1] Patents Act, 1970 — Section 3, pp. 11–12" in rendered


def test_renderer_omits_a_disclaimer_from_a_clean_answer():
    answer = GroundedAnswerEngine().answer(query_for("citable_statutory"),
                                           evidence_for("citable_statutory"))
    rendered = render_answer(answer)
    assert "Caution" not in rendered
    assert "not legal advice" not in rendered


def test_renderer_surfaces_caution_only_when_earned():
    answer = GroundedAnswerEngine().answer(query_for("stale_only"),
                                           evidence_for("stale_only"))
    assert "Caution" in render_answer(answer)


def test_renderer_cannot_add_a_citation():
    answer = Answer(answer="Something happened [1].", citations=(),
                    insufficient_evidence=False, grounded=True)
    assert "Source(s)" not in render_answer(answer)


def test_confidence_is_ordinal_and_labelled_as_such():
    answer = GroundedAnswerEngine().answer(query_for("citable_statutory"),
                                           evidence_for("citable_statutory"))
    rendered = render_answer(answer, show_confidence=True)
    assert answer.confidence in ("HIGH", "MODERATE", "LOW", "NONE")
    assert "not a probability" in rendered
