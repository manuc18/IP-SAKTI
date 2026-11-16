"""End to end, over the real corpus: every case the fix sprint had to answer for.

One pipeline, built once, with the offline extractive provider so no test in
this file depends on a network, a key or a rate limit. What is under test is the
*routing* — which of SOCIAL, NEEDS_CLARIFICATION, ANSWERED and ABSTAINED a given
input produces, and what travels with it — not the wording of any answer.
"""
from __future__ import annotations

import os

import pytest


@pytest.fixture(scope="module")
def pipeline():
    os.environ["IPSAKTI_DENSE"] = "0"
    from app.pipeline import Pipeline

    return Pipeline(use_dense=False, provider="extractive", cache=False,
                    log=lambda message: None)


# -- SOCIAL -------------------------------------------------------------------

@pytest.mark.parametrize("query", ["hi", "hello", "how are you?", "thanks",
                                   "good morning", "bye", "ok"])
def test_social_input_is_answered_without_touching_retrieval(pipeline, query):
    result = pipeline.answer(query)
    assert result["status"] == "SOCIAL"
    assert result["answer"] is None
    assert result["evidence"] == []
    assert result["rendered"]
    assert result["pipeline"]["candidates"] == 0
    assert result["social"]["retrieval_performed"] is False


def test_a_social_reply_is_neither_answered_nor_abstained(pipeline):
    """So it cannot move the grounded-answer rate or the abstention rate."""
    assert pipeline.answer("hi")["status"] not in ("ANSWERED", "ABSTAINED",
                                                   "UNAVAILABLE")


def test_a_greeting_attached_to_a_real_question_is_answered(pipeline):
    result = pipeline.answer("hi, how must an application for registration of "
                             "a trade mark be signed?")
    assert result["status"] == "ANSWERED"
    assert result["evidence"]


# -- LEGAL --------------------------------------------------------------------

def test_a_plain_legal_question_is_answered_with_evidence(pipeline):
    result = pipeline.answer(
        "How must an application for registration of a trade mark be signed?")
    assert result["status"] == "ANSWERED"
    assert result["answer"]["grounded"] is True
    assert result["answer"]["citations"]
    assert result["spelling"]["applied"] is False


def test_an_identifier_question_keeps_its_identifier(pipeline):
    result = pipeline.answer("What does section 3(p) of the Patents Act exclude?")
    assert result["spelling"]["search_query"] == result["spelling"]["original_query"]
    assert result["status"] in ("ANSWERED", "ABSTAINED")


# -- TYPO ---------------------------------------------------------------------

@pytest.mark.parametrize("query,expected", [
    ("How do I register a tradmark?", "trademark"),
    ("What are the requirements for a patant?", "patent"),
    ("what is a geographcal indication", "geographical"),
])
def test_a_misspelled_question_is_searched_for_correctly(pipeline, query, expected):
    result = pipeline.answer(query)
    assert result["spelling"]["applied"] is True
    assert expected in result["spelling"]["search_query"]
    # The question the user asked is what is echoed back and what the generator
    # was asked; only retrieval saw the corrected string.
    assert result["question"] == query
    assert result["status"] == "ANSWERED"


def test_a_typo_no_longer_triggers_a_clarifying_question(pipeline):
    """The interaction that made the correction have to run first.

    "tradmark" anchors on no regime, so the clarifier used to intercept and ask
    which area of law was meant — of a question that names one.
    """
    assert pipeline.answer("How do I register a tradmark?")["status"] == "ANSWERED"


# -- AMBIGUOUS / OFF-TOPIC ----------------------------------------------------

def test_a_genuinely_unanchored_question_gets_one_clarification(pipeline):
    result = pipeline.answer("How can I protect my idea?")
    assert result["status"] == "NEEDS_CLARIFICATION"
    assert result["clarification"]["options"]


def test_the_clarification_is_asked_at_most_once(pipeline):
    """The loop guard. The second attempt runs the pipeline whatever it says."""
    again = pipeline.answer("How can I protect my idea?", allow_clarification=False)
    assert again["status"] != "NEEDS_CLARIFICATION"


def test_an_off_topic_question_is_not_swallowed_by_the_social_gate(pipeline):
    result = pipeline.answer("What is the best pizza topping in Naples?")
    assert result["status"] != "SOCIAL"
    assert result["spelling"]["applied"] is False, (
        "an off-topic question must not be spell-corrected into corpus words")


# -- EVIDENCE AND SOURCE CARDS ------------------------------------------------

def test_every_evidence_item_carries_an_integrity_verdict(pipeline):
    result = pipeline.answer(
        "What must a licence to manufacture state under the Drugs and "
        "Cosmetics Rules, 1945?")
    assert result["evidence"]
    for item in result["evidence"]:
        assert item["text_integrity"]["status"] in ("CLEAN", "EXTRACTION_DAMAGED")
        assert item["text_integrity"]["repairable"] is False
        if item.get("excerpt"):
            assert item["excerpt"]["integrity"]["status"] in (
                "CLEAN", "EXTRACTION_DAMAGED")


def test_the_excerpt_is_always_a_substring_of_the_passage_beside_it(pipeline):
    from ai.answering.excerpt import display_text

    for query in ("What is the term of a registered design?",
                  "What must a licence to manufacture state under the Drugs "
                  "and Cosmetics Rules, 1945?"):
        for item in pipeline.answer(query)["evidence"]:
            excerpt = item.get("excerpt")
            if not excerpt:
                continue
            assert excerpt["text"] in display_text(item["text"])


def test_a_published_answer_explains_its_confidence(pipeline):
    result = pipeline.answer(
        "How must an application for registration of a trade mark be signed?")
    report = result["answer"]["confidence_report"]
    assert report["level"] == result["answer"]["confidence"]
    assert report["source_confidence"] and report["support_confidence"]
    assert report["factors"], "a confidence level with no stated reason"
    assert report["is_probability"] is False


def test_the_response_declares_every_deterministic_gate_it_ran(pipeline):
    config = pipeline.answer("What is the term of a registered design?")["pipeline"]
    for gate in ("intent", "spelling", "text_integrity", "excerpt",
                 "clarification"):
        assert gate in config
    assert config["spelling"]["uses_model"] is False
    assert config["intent"]["uses_model"] is False
    assert config["text_integrity"]["repairs_text"] is False


# -- the API contract for the two non-answer statuses -------------------------

def test_the_api_carries_the_social_reply_outside_the_answer_object(pipeline):
    """`answer` stays null so no client renders a greeting as a legal answer."""
    from app.schemas import serialize_result

    body, code = serialize_result(pipeline.answer("hi"), "req-test")
    assert code == 200
    assert body["status"] == "SOCIAL"
    assert body["answer"] is None
    assert body["citations"] == [] and body["sources"] == []
    assert body["reply"]
    assert body["meta"]["retrieval_performed"] is False


def test_the_api_carries_a_clarification_outside_the_answer_object(pipeline):
    from app.schemas import serialize_result

    body, code = serialize_result(
        pipeline.answer("How can I protect my idea?"), "req-test")
    assert code == 200
    assert body["status"] == "NEEDS_CLARIFICATION"
    assert body["answer"] is None
    assert body["clarification"]["question"]
    assert body["meta"]["retrieval_performed"] is False


def test_neither_non_answer_status_reports_itself_as_grounded(pipeline):
    from app.schemas import serialize_result

    for question in ("hi", "How can I protect my idea?"):
        body, _ = serialize_result(pipeline.answer(question), "req-test")
        assert body["meta"]["grounded"] is False
        assert body["cautions"] == []
