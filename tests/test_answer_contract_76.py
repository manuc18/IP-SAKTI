"""Phase 7.6 — the three deterministic defects Q027 exposed, and the one it did not.

Phase 7.5 spent one live request on Q027 and it was refused. What refused it was
not a hallucination:

* **Issue 1** — the answer quoted the supplied chunk's own ``sub-rule (2) of
  rule 20`` using U+2011 NON-BREAKING HYPHEN. The chunk parsed as
  ``('rule','20',('2',))`` and the quotation as ``('rule','20',())``, so the
  citation validator's cross-reference arm missed and reported the corpus's own
  words as an invented locator.
* **Issue 2** — ``Geographical Indications Rules 2002`` parsed as a rule
  numbered 2002. An instrument's year was read as a provision number.
* **Issue 3** — six of nine sentences carried a marker. The three that did not
  were an opening topic sentence, a summary and an aside: exactly the kinds
  prompt 0.2.0 had just finished enumerating as things that must be cited.

Issues 1 and 2 are parser bugs and are fixed here. Issue 3 is not a parser bug
and is not fixed by asking harder, so 0.3.0 asks for *fewer sentences* instead.
Whether that works is a question only the reserved live request can answer;
these tests establish what the contract now says, not that a model will obey it.

Every test below is offline and CPU-only. No network, no model, no torch.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.answering.groq import (  # noqa: E402
    DEFAULT_MAX_COMPLETION_TOKENS,
    GroqConfig,
    _completion_meta,
)
from ai.answering.grounding import check_grounding, split_sentences  # noqa: E402
from ai.answering.providers import ANSWER_INSTRUCTIONS, PROMPT_VERSION  # noqa: E402
from ai.retrieval.identifiers import (  # noqa: E402
    HYPHEN_EQUIVALENTS,
    normalise_identifier_text,
    parse_identifiers,
)

from test_answer_contract_74 import evidence, evidence_set  # noqa: E402

#: The reference at the centre of Issue 1, in each spelling that must mean the
#: same thing. The first is the corpus's; the second is what the model wrote.
SUB_RULE_SPELLINGS = (
    "under sub-rule (2) of rule 20",          # U+002D  HYPHEN-MINUS (corpus)
    "under sub‑rule (2) of rule 20",     # U+2011  NON-BREAKING HYPHEN (model)
    "under sub‐rule (2) of rule 20",     # U+2010  HYPHEN
    "under sub–rule (2) of rule 20",     # U+2013  EN DASH (corpus, 4 places)
)


def keys(text):
    return [(r.kind, r.number, r.subdivisions) for r in parse_identifiers(text)]


# -- Issue 1: hyphen spellings ----------------------------------------------


class TestIssue1HyphenNormalisation:
    """Four spellings of one reference must produce one key."""

    def test_the_corpus_spelling_parses(self):
        assert keys(SUB_RULE_SPELLINGS[0]) == [("rule", "20", ("2",))]

    def test_the_model_spelling_parses_identically(self):
        # The literal Phase 7.5 Q027 failure. Before this fix the non-breaking
        # hyphen produced ('rule','20',()) and the answer was refused.
        assert keys(SUB_RULE_SPELLINGS[1]) == [("rule", "20", ("2",))]

    def test_every_supported_spelling_produces_the_same_key(self):
        parsed = {tuple(keys(text)) for text in SUB_RULE_SPELLINGS}
        assert len(parsed) == 1, f"spellings disagree: {parsed}"

    def test_normalisation_is_narrow_and_idempotent(self):
        once = normalise_identifier_text("sub‑rule–a‐b")
        assert once == "sub-rule-a-b"
        assert normalise_identifier_text(once) == once

    def test_em_dash_and_minus_are_not_normalised(self):
        # Measured, not assumed: U+2014 appears 1905 times in the corpus and is
        # the legislative heading separator ("namely:—"), not a hyphen. Folding
        # it in would be normalising all punctuation, which is the failure mode
        # this table exists to avoid.
        assert normalise_identifier_text("a—b−c") == "a—b−c"
        assert "—" not in dict(HYPHEN_EQUIVALENTS)
        assert "−" not in dict(HYPHEN_EQUIVALENTS)

    def test_punctuation_alone_does_not_become_a_provision(self):
        for text in ("sub-rule", "-- (2) --", "rule -", "20 (2)", "a-b-c"):
            assert keys(text) == [], f"{text!r} produced an identifier"

    def test_existing_parsing_is_unchanged(self):
        assert keys("section 3(p)") == [("section", "3", ("P",))]
        assert keys("sub-section (1) of section 39") == [("section", "39", ("1",))]
        assert keys("Article 27.1") == [("article", "27", ("1",))]
        assert keys("how long does a patent last in India") == []


# -- Issue 2: an instrument's year is not a provision number ------------------


class TestIssue2InstrumentTitles:
    """A plural legislative term plus a bare year names an instrument."""

    def test_the_literal_q027_false_positive_is_gone(self):
        assert keys("Geographical Indications Rules 2002") == []

    def test_it_generalises_across_the_corpus_instruments(self):
        for title in ("Patents Rules 2003", "Trade Marks Rules 2017",
                      "Designs Rules 2001", "Copyright Rules 2013",
                      "PPVFR Rules 2003", "Biological Diversity Rules 2024",
                      "Drugs and Cosmetics Rules 1945"):
            assert keys(title) == [], f"{title!r} parsed as a provision"

    def test_the_comma_spelling_is_still_inert(self):
        # It always was: `_DIRECT` never matched across the comma. The fix is
        # for the spelling *without* it, which is the one a model writes.
        assert keys("Patents Rules, 2003") == []

    def test_a_singular_term_still_names_a_provision(self):
        # The distinction is grammatical, not numeric. This module has no
        # business ruling out a four-digit provision it has never seen.
        assert keys("Rule 2003") == [("rule", "2003", ())]

    def test_rules_2003_does_not_parse_as_rule_2003(self):
        assert keys("Rules 2003") == []
        assert ("rule", "2003", ()) not in keys("the Patents Rules 2003")

    def test_rule_23_still_parses(self):
        assert keys("Rule 23") == [("rule", "23", ())]

    def test_rule_23_1_still_parses(self):
        assert keys("Rule 23(1)") == [("rule", "23", ("1",))]

    def test_a_plural_term_with_an_ordinary_number_still_parses(self):
        assert keys("rules 23 and 24") == [("rule", "23", ())]
        assert keys("sections 3 and 4") == [("section", "3", ())]

    def test_a_year_shaped_number_with_a_subdivision_still_parses(self):
        # The guard requires a *bare* year. A parenthesised tail is a citation's
        # syntax and no instrument title carries one.
        assert keys("Rules 2003(2)") == [("rule", "2003", ("2",))]

    def test_the_whole_q027_sentence_now_resolves(self):
        sentence = ("the names of goods and services must correspond to those "
                    "published by the Registrar under sub‑rule (2) of Rule "
                    "20, as in the Geographical Indications Rules 2002")
        assert keys(sentence) == [("rule", "20", ("2",))]


# -- Issue 3: the answer contract ---------------------------------------------


GOOD = ("Rule 23 governs the filing of trademark applications [1]. It requires "
        "the prescribed TM-A form and the applicant's signature [1].")

BAD = ("Rule 23 is an important provision concerning trademark applications. "
       "The rule governs filing. It requires TM-A [1].")

#: The completion Phase 7.5 actually got, shortened to its shape: an uncited
#: opening, cited middles, an uncited summary.
Q027_SHAPE = ("Rule 23 of the Trade Marks Rules, 2017 governs the form and "
              "signing of a trademark application. An application must be made "
              "in Form TM-A and signed by the applicant [1]. Thus, Rule 23 "
              "specifies that the prescribed form is TM-A.")


class TestIssue3AnswerContract:
    """Fewer sentences, every one of them evidence-bearing."""

    def test_prompt_version_records_the_change(self):
        assert PROMPT_VERSION == "0.3.0"

    def test_the_contract_forbids_an_introduction(self):
        joined = " ".join(ANSWER_INSTRUCTIONS).lower()
        assert "no introduction" in joined and "framing" in joined

    def test_the_contract_forbids_a_conclusion_or_summary(self):
        joined = " ".join(ANSWER_INSTRUCTIONS).lower()
        assert "conclusion" in joined and "summary" in joined

    def test_the_contract_forbids_a_conversational_preamble(self):
        joined = " ".join(ANSWER_INSTRUCTIONS).lower()
        assert "preamble" in joined
        assert "based on the supplied evidence" in joined

    def test_the_contract_asks_for_one_to_four_sentences(self):
        joined = " ".join(ANSWER_INSTRUCTIONS).lower()
        assert "1 to 4 sentences" in joined

    def test_the_contract_requires_a_marker_on_every_sentence(self):
        joined = " ".join(ANSWER_INSTRUCTIONS).lower()
        assert "every sentence must carry a citation marker" in joined

    def test_the_contract_states_where_the_marker_goes(self):
        joined = " ".join(ANSWER_INSTRUCTIONS).lower()
        assert "immediately before the sentence's final punctuation" in joined

    def test_the_contract_requires_markers_to_be_repeated(self):
        joined = " ".join(ANSWER_INSTRUCTIONS).lower()
        assert "repeat it on every sentence" in joined

    def test_the_contract_prohibits_an_unsupportable_sentence(self):
        joined = " ".join(ANSWER_INSTRUCTIONS).lower()
        assert "do not write a sentence you cannot support" in joined

    def test_the_contract_carries_both_worked_examples(self):
        joined = " ".join(ANSWER_INSTRUCTIONS)
        assert f"GOOD: \"{GOOD}\"" in joined
        assert f"BAD: \"{BAD}\"" in joined

    def test_the_good_example_passes_our_own_grounding_check(self):
        # An example handed to a model is worthless if our own gate refuses it.
        report = check_grounding(GOOD, evidence_set(evidence(1)))
        assert report.grounded
        assert len(split_sentences(GOOD)) == 2

    def test_the_bad_example_still_fails_grounding(self):
        # The anti-weakening guard. Issue 3 was answered by changing the output
        # shape, and a future "fix" that instead taught `check_grounding` to
        # ignore opening and closing sentences would publish this.
        report = check_grounding(BAD, evidence_set(evidence(1)))
        assert not report.grounded
        assert len(report.unsupported_claims) == 2

    def test_the_q027_shape_still_fails_grounding(self):
        # The literal Phase 7.5 failure, in miniature: uncited opening, uncited
        # summary. Grounding must go on refusing it under 0.3.0.
        report = check_grounding(Q027_SHAPE, evidence_set(evidence(1)))
        assert not report.grounded
        assert len(report.unsupported_claims) == 2

    def test_grounding_itself_was_not_relaxed(self):
        # Not one sentence gained an exemption in Phase 7.6.
        from ai.answering.grounding import GROUNDING_VERSION, _NON_CLAIM_PREFIXES

        assert GROUNDING_VERSION == "0.2.0"
        assert "rule 23" not in " ".join(_NON_CLAIM_PREFIXES)
        assert "thus" not in " ".join(_NON_CLAIM_PREFIXES)


# -- Issue 4 and 5: the completion budget, and no retries ---------------------


class TestCompletionConfiguration:
    """800 was measured and kept, and nothing regenerates."""

    def test_max_completion_tokens_is_eight_hundred(self):
        assert DEFAULT_MAX_COMPLETION_TOKENS == 800
        assert GroqConfig().max_completion_tokens == 800
        assert GroqConfig.from_environment(
            load_dotenv=False).max_completion_tokens == 800

    def test_the_environment_cannot_raise_it(self):
        # There is no GROQ_MAX_COMPLETION_TOKENS. The ceiling is a code
        # decision with a recorded reason, not a configuration knob.
        assert "max_completion_tokens" not in os.environ
        assert GroqConfig().to_dict()["max_completion_tokens"] == 800

    def test_the_config_still_reports_zero_retries(self):
        assert GroqConfig().to_dict()["retries"] == 0

    def test_truncation_is_now_visible_rather_than_inferred(self):
        import json

        body = json.dumps({
            "choices": [{"finish_reason": "length",
                         "message": {"content": "x"}}],
            "usage": {"prompt_tokens": 2231, "completion_tokens": 800,
                      "total_tokens": 3031}}).encode("utf-8")
        meta = _completion_meta(body)
        assert meta["finish_reason"] == "length"
        assert meta["completion_tokens"] == 800

    def test_reading_the_metadata_can_never_break_a_completion(self):
        assert _completion_meta(b"not json at all") == {}
        assert _completion_meta(b"{}") == {}


class TestIssue5NoRetriesWereAdded:
    """A failed answer stays an abstention."""

    def test_the_provider_module_names_no_retry_machinery(self):
        import ai.answering.groq as groq

        source = open(groq.__file__, encoding="utf-8").read().lower()
        for banned in ("for attempt", "while attempt", "time.sleep",
                       "backoff", "fallback_provider", "max_retries"):
            assert banned not in source, f"{banned!r} appeared in the provider"

    def test_the_engine_calls_the_provider_exactly_once(self):
        import ai.answering.base as base

        source = open(base.__file__, encoding="utf-8").read()
        assert source.count(".generate(request)") == 1
