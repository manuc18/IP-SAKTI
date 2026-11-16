"""Phase 7.4 — the three failures the first live run actually produced.

Phase 7.3 ran five real questions through the pipeline. Two reached the model
and both were refused, for three separate reasons that had nothing to do with
the model inventing anything:

* **F1** — Q027. The model wrote an opening sentence stating what Rule 23
  governs, cited nothing on it, and cited ``[1]`` on the sentences after. The
  grounding check counted that opening sentence as an uncited legal
  proposition, which it is. The prompt was not specific about which sentences
  need a marker.
* **F2** — Q001. The evidence was the whole of Section 3; the model wrote
  "section 3(p)", a clause of that very passage, and the prose scan reported it
  as a provision named by nothing supplied.
* **F3** — Q001. The model wrote the marker as ``【1】``, which the ASCII-only
  pattern could not see at all.

Every fix here has to make the system *more accurate*, not merely more
accepting, so each group below pairs the case that should now pass with the
neighbouring case that must still fail. Offline, CPU-only, no network, no model.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.answering.base import GroundedAnswerEngine  # noqa: E402
from ai.answering.citation_validator import validate_citations  # noqa: E402
from ai.answering.grounding import (  # noqa: E402
    check_grounding,
    extract_markers,
    normalise_markers,
)
from ai.answering.models import Evidence, EvidenceSet  # noqa: E402
from ai.answering.policy import (  # noqa: E402
    ref_matches_locator,
    ref_matches_subdivision_of,
)
from ai.answering.providers import (  # noqa: E402
    ANSWER_INSTRUCTIONS,
    PROMPT_VERSION,
    AnswerProvider,
    ProviderDraft,
    build_request,
)
from ai.retrieval.identifiers import parse_identifiers  # noqa: E402

# Section 3 of the Patents Act as the corpus stores it: one chunk, the whole
# provision, clauses lettered inside its text and *not* in its locator. This is
# the exact shape that produced F2.
SECTION_3_TEXT = (
    "3. What are not inventions. The following are not inventions within the "
    "meaning of this Act, namely, (d) the mere discovery of a new form of a "
    "known substance which does not result in the enhancement of the known "
    "efficacy of that substance, (p) an invention which in effect is "
    "traditional knowledge or which is an aggregation or duplication of known "
    "properties of traditionally known component or components."
)

RULE_23_TEXT = (
    "23. Filing of application. (1) An application for the registration of a "
    "trade mark shall be made in Form TM-A and shall be signed by the "
    "applicant or his agent."
)


def evidence(marker=1, *, text=SECTION_3_TEXT, locator_fields=None,
             citation="Patents Act, 1970 — Section 3", citable=True,
             locator_unavailable=False, pages=(11,)):
    return Evidence(
        marker=marker,
        chunk_id=f"IN-PATENTS-ACT-1970::c{marker}",
        document_id="IN-PATENTS-ACT-1970",
        document_name="Patents Act, 1970",
        citation=citation,
        text=text,
        citable=citable,
        locator_unavailable=locator_unavailable,
        locator_fields=dict(locator_fields or {"section": "3"}),
        temporal_status="CURRENT",
        source_evidence_level="A",
        physical_pages=tuple(pages),
    )


def evidence_set(*items):
    return EvidenceSet(items or (evidence(),))


def ref(text):
    """The single provision `text` names, through the Phase 5 parser."""
    refs = parse_identifiers(text)
    assert len(refs) == 1, f"expected one identifier in {text!r}, got {refs}"
    return refs[0]


# -- F1: the prompt contract ------------------------------------------------


class TestF1PromptRequiresEveryProposition:
    """The instruction, not the gate. Grounding is unchanged and must stay so."""

    def test_prompt_version_was_bumped(self):
        # An answer produced under the old prompt and one produced under this
        # one are not comparable, and the version is what says so. 0.2.0 is the
        # version this file was written for; Phase 7.5 tested it live, it did
        # not work, and Phase 7.6 replaced it — see test_answer_contract_76.py.
        assert PROMPT_VERSION == "0.3.0"

    def test_the_contract_still_addresses_every_kind_of_sentence(self):
        # 0.2.0 answered F1 by *naming* the sentence kinds that need a marker:
        # opening, framing, explanatory, paraphrase, conclusion, summary. Phase
        # 7.5 showed the model went on omitting markers from exactly those, so
        # 0.3.0 forbids writing them at all. The concern is the same one and it
        # is still covered; what changed is which side of it the contract acts
        # on. Rewriting this to check for the old wording would be asserting
        # that a superseded fix is still in place.
        joined = " ".join(ANSWER_INSTRUCTIONS).lower()
        for forbidden in ("introduction", "framing", "conclusion", "summary",
                          "preamble"):
            assert forbidden in joined, f"the contract does not forbid a {forbidden}"

    def test_the_contract_carries_the_worked_example(self):
        joined = " ".join(ANSWER_INSTRUCTIONS)
        assert "BAD:" in joined and "GOOD:" in joined

    def test_the_good_example_passes_our_own_grounding_check(self):
        # The point of the whole fix. An example we hand a model is worthless if
        # our own gate would refuse it, so the gate judges it here.
        good = ("Rule 23 governs how trademark applications are filed [1]. "
                "It requires form TM-A [1].")
        report = check_grounding(good, evidence_set(evidence(text=RULE_23_TEXT)))
        assert report.grounded
        assert report.unsupported_claims == ()

    def test_the_bad_example_still_fails_it(self):
        # If this ever passes, grounding has been weakened and F1 was "fixed" by
        # lowering the bar instead of raising the instruction.
        bad = ("Rule 23 governs how trademark applications are filed. "
               "It requires form TM-A [1].")
        report = check_grounding(bad, evidence_set(evidence(text=RULE_23_TEXT)))
        assert not report.grounded
        assert len(report.unsupported_claims) == 1
        assert report.unsupported_claims[0].startswith("Rule 23 governs")

    def test_the_q027_shape_is_what_the_bad_example_reproduces(self):
        # The literal draft from Phase 7.3 Q027, still correctly refused.
        drafted = ("Rule 23 of the Trade Marks Rules, 2017 governs how a "
                   "trademark application must be filed. It requires that the "
                   "application be made on the prescribed form TM-A [1].")
        item = evidence(text=RULE_23_TEXT,
                        citation="Trade Marks Rules, 2017 — Rule 23",
                        locator_fields={"rule": "23"})
        report = check_grounding(drafted, evidence_set(item))
        assert not report.grounded

    def test_the_instructions_reach_the_provider_payload(self):
        request = build_request(ref_free_query(), evidence_set(), policy())
        assert request.instructions == ANSWER_INSTRUCTIONS
        assert request.prompt_version == PROMPT_VERSION


def ref_free_query():
    return "what is excluded from patentability"


def policy():
    from ai.answering.policy import AnswerPolicy
    return AnswerPolicy()


# -- F2: a parent locator supporting a clause it actually contains ----------


class TestF2SubdivisionOfSuppliedProvision:
    def test_section_3_supports_section_3p_when_clause_p_is_in_the_text(self):
        assert ref_matches_subdivision_of(ref("section 3(p)"), evidence())

    def test_section_3_rejects_section_3zz_which_is_not_in_the_text(self):
        # The safety half of the rule. Being Section 3 is not enough.
        assert not ref_matches_subdivision_of(ref("section 3(zz)"), evidence())

    def test_section_3p_exact_match_is_unchanged(self):
        item = evidence(locator_fields={"section": "3", "clause": "p"},
                        citation="Patents Act, 1970 — Section 3(p)")
        assert ref_matches_locator(ref("section 3(p)"), item)
        # And the new rule does not also fire: a chunk that *is* the clause is
        # not the clause's parent.
        assert not ref_matches_subdivision_of(ref("section 3(p)"), item)

    def test_rule_23_supports_sub_rule_1_only_when_1_is_in_the_text(self):
        present = evidence(text=RULE_23_TEXT, locator_fields={"rule": "23"},
                           citation="Trade Marks Rules, 2017 — Rule 23")
        absent = evidence(text="23. Filing of application. An application "
                               "shall be made in Form TM-A.",
                          locator_fields={"rule": "23"},
                          citation="Trade Marks Rules, 2017 — Rule 23")
        assert ref_matches_subdivision_of(ref("rule 23(1)"), present)
        assert not ref_matches_subdivision_of(ref("rule 23(1)"), absent)

    def test_an_unrelated_subdivision_of_a_sibling_is_rejected(self):
        # Evidence is Section 3(p); the prose names Section 3(q). Siblings, not
        # parent and child, and neither arm may admit it.
        item = evidence(locator_fields={"section": "3", "clause": "p"},
                        text="(p) an invention which in effect is traditional "
                             "knowledge.")
        assert not ref_matches_locator(ref("section 3(q)"), item)
        assert not ref_matches_subdivision_of(ref("section 3(q)"), item)

    def test_prose_naming_a_contained_clause_now_validates(self):
        # The end-to-end shape of Q001, through the validator that refused it.
        text = ("Section 3(p) excludes traditional knowledge from "
                "patentability [1].")
        outcome = validate_citations(text, evidence_set())
        assert outcome.ok, [f.to_dict() for f in outcome.failures]
        # The citation still carries the corpus's own locator, not the model's.
        assert outcome.citations[0].locator == "Section 3"

    def test_prose_naming_a_clause_that_is_not_there_still_fails(self):
        text = "Section 3(zz) excludes something else entirely [1]."
        outcome = validate_citations(text, evidence_set())
        assert not outcome.ok
        assert [f.code for f in outcome.failures] == ["unsupported_locator_in_prose"]

    def test_the_citable_invariant_is_unchanged(self):
        # A clause that resolves under the new rule is still not citable if the
        # chunk it lives in was never citable.
        item = evidence(citable=False)
        outcome = validate_citations("Section 3(p) excludes this [1].",
                                     EvidenceSet((item,)))
        assert not outcome.ok
        assert "non_citable_evidence" in [f.code for f in outcome.failures]

    def test_the_locator_unavailable_invariant_is_unchanged(self):
        item = evidence(locator_unavailable=True)
        outcome = validate_citations("Section 3(p) excludes this [1].",
                                     EvidenceSet((item,)))
        assert not outcome.ok
        assert "inconsistent_locator_state" in [f.code for f in outcome.failures]


# -- F3: marker spellings ---------------------------------------------------


class TestF3MarkerNormalisation:
    def test_ascii_marker_is_untouched(self):
        assert normalise_markers("a claim [1].") == "a claim [1]."
        assert extract_markers("a claim [1].") == [1]

    def test_full_width_marker_becomes_ascii(self):
        assert normalise_markers("a claim 【1】.") == "a claim [1]."

    def test_multiple_ascii_markers(self):
        assert extract_markers(normalise_markers("[1] and [2] and [1]")) == [1, 2]

    def test_multiple_full_width_markers(self):
        assert normalise_markers("【1】【2】") == "[1][2]"
        assert extract_markers(normalise_markers("【1】【2】")) == [1, 2]

    def test_mixed_spellings_in_one_answer(self):
        assert normalise_markers("first [1]. second 【2】.") == "first [1]. second [2]."

    def test_malformed_markers_are_not_repaired(self):
        # Not a marker syntax, and normalisation is not a repair shop. Each of
        # these stays exactly as written and resolves to nothing.
        for malformed in ("【a】", "【1]", "[1】", "【 1 】", "(1)", "[1", "1]"):
            assert normalise_markers(malformed) == malformed
            assert extract_markers(normalise_markers(malformed)) == []

    def test_an_out_of_range_marker_still_fails_however_it_is_spelled(self):
        for spelling in ("[9]", "【9】"):
            text = f"Section 3 excludes traditional knowledge {spelling}."
            outcome = validate_citations(normalise_markers(text), evidence_set())
            assert not outcome.ok, spelling
            assert "unknown_marker" in [f.code for f in outcome.failures]

    def test_the_engine_publishes_the_normalised_text(self):
        # The reason normalisation lives in the engine: the prose a reader sees
        # and the citation list beneath it must use the same syntax.
        answer = engine_answering("Section 3 excludes traditional knowledge 【1】."
                                  ).answer
        assert "【1】" not in answer
        assert "[1]" in answer

    def test_the_engine_records_that_it_normalised(self):
        normalised = engine_answering("Section 3 excludes traditional knowledge 【1】.")
        assert normalised.diagnostics["marker_normalisation_applied"] is True
        untouched = engine_answering("Section 3 excludes traditional knowledge [1].")
        assert untouched.diagnostics["marker_normalisation_applied"] is False

    def test_a_full_width_marker_no_longer_reads_as_an_uncited_claim(self):
        # F3 as it actually bit: the marker was invisible, so a correctly
        # sourced sentence was reported as carrying no citation at all.
        answered = engine_answering("Section 3 excludes traditional knowledge 【1】.")
        assert not answered.insufficient_evidence
        assert answered.grounded


class FixedDraftProvider(AnswerProvider):
    """Returns one prepared draft. Stands in for a model, deterministically."""

    name = "fixed_draft"

    def __init__(self, text):
        self.text = text

    def generate(self, request):
        return ProviderDraft(text=self.text, provider=self.name)


#: Shares content terms with SECTION_3_TEXT, so policy admits the passage and
#: the draft actually reaches the citation and grounding gates. A query that
#: shared nothing would abstain before the provider ran and these tests would
#: pass for the wrong reason.
ENGINE_QUERY = "which inventions are excluded as traditional knowledge"


def engine_answering(text):
    engine = GroundedAnswerEngine(provider=FixedDraftProvider(text))
    return engine.answer(ENGINE_QUERY, evidence_set())
