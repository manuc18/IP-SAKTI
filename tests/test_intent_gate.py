"""Issue 1: social inputs must not enter legal retrieval — and legal questions must.

The asymmetry matters more than the feature. Missing a greeting costs a slightly
odd response to "hi". Catching a legal question costs the user their answer and
makes the system look like it cannot read. So the tests below spend most of
their effort on the second failure.
"""
from __future__ import annotations

import pytest

from app.intent import (
    ACKNOWLEDGEMENT,
    CAPABILITY,
    FAREWELL,
    GREETING,
    PHRASES,
    REPLIES,
    SMALL_TALK,
    THANKS,
    classify,
    intent_config,
    normalise,
)

SOCIAL = [
    ("hi", GREETING), ("hello", GREETING), ("hey", GREETING), ("yo", GREETING),
    ("Good morning", GREETING), ("hi there", GREETING), ("namaste", GREETING),
    ("how are you?", SMALL_TALK), ("How are you doing?", SMALL_TALK),
    ("what's up", SMALL_TALK), ("hows it going", SMALL_TALK),
    ("thanks", THANKS), ("Thank you", THANKS), ("thanks a lot", THANKS),
    ("thx", THANKS), ("much appreciated", THANKS),
    ("ok", ACKNOWLEDGEMENT), ("yay", ACKNOWLEDGEMENT), ("cool", ACKNOWLEDGEMENT),
    ("got it", ACKNOWLEDGEMENT),
    ("bye", FAREWELL), ("goodbye", FAREWELL), ("see you", FAREWELL),
    ("help", CAPABILITY), ("Help!", CAPABILITY), ("help me", CAPABILITY),
    ("how can you help", CAPABILITY), ("How can you help me?", CAPABILITY),
    ("what can you do", CAPABILITY), ("What can you do?", CAPABILITY),
    ("who are you", CAPABILITY), ("what do you do", CAPABILITY),
    ("can you help", CAPABILITY), ("how does this work", CAPABILITY),
]

#: Real questions, off-topic questions and near-misses. Every one of these must
#: reach the pipeline. The list is deliberately stocked with inputs that share
#: words with the phrase table.
NOT_SOCIAL = [
    "How must an application for registration of a trade mark be signed?",
    "What is the term of a registered design?",
    "section 85B",
    "What does section 3(p) of the Patents Act exclude?",
    "How can I protect my idea?",
    "What is the best pizza topping in Naples?",
    "hi, how do I register a trade mark?",
    "Hello, what is the fee for filing Form 1?",
    "thanks, but what does Rule 23 require?",
    "Is there a good ground for rectification of the register?",
    "What is a fine under the Drugs and Cosmetics Act?",
    "good faith",
    "What is the term of copyright?",
    "ok so what happens if the applicant does not respond",
    "How do I register a tradmark?",
    # The capability table must not reach past its own exact keys. Each of
    # these contains one of them as a substring.
    "help me understand the opposition procedure",
    "What can you tell me about compulsory licensing?",
    "who are the persons entitled to apply for a patent?",
    "how does this work for a geographical indication?",
]


@pytest.mark.parametrize("query,intent", SOCIAL)
def test_social_inputs_are_classified_and_answered_without_retrieval(query, intent):
    result = classify(query)
    assert result is not None, f"{query!r} should have been social"
    assert result.intent == intent
    assert result.reply == REPLIES[intent]
    assert result.to_dict()["retrieval_performed"] is False


@pytest.mark.parametrize("query", NOT_SOCIAL)
def test_everything_that_is_not_a_greeting_reaches_the_pipeline(query):
    assert classify(query) is None, f"{query!r} was wrongly treated as social"


def test_a_greeting_followed_by_a_real_question_is_not_social():
    """The single most damaging false positive, so it gets its own test."""
    assert classify("hi") is not None
    assert classify("hi, how do I register a trade mark?") is None
    assert classify("hello there, what does Rule 23 say?") is None


def test_two_social_segments_are_still_social_and_take_the_last_intent():
    result = classify("hi, thanks")
    assert result is not None and result.intent == THANKS
    assert result.matched == ("hi", "thanks")


def test_an_anchored_query_is_never_social_even_if_it_reads_like_one():
    """Guard one, tested on its own: naming any provision disables the table."""
    assert classify("hi, section 3") is None
    assert classify("thanks, Patents Act") is None


def test_stretched_and_punctuated_spellings_normalise_to_the_same_key():
    assert normalise("Hiii!!!") == "hi"
    assert normalise("HEYYYY") == "hey"
    assert normalise("what's up?") == "whats up"
    assert classify("Hiii!!!").intent == GREETING
    assert classify("heyyy :)").intent == GREETING


def test_empty_and_whitespace_input_is_not_social():
    assert classify("") is None
    assert classify("   ") is None
    assert classify(None) is None


def test_no_phrase_in_the_table_is_a_word_with_legal_meaning_here():
    """`fine` and `good` were removed after review; keep them out.

    A phrase that also names a legal concept is a phrase that can swallow a real
    query, and the anchor guard does not catch a bare content word.
    """
    for forbidden in ("fine", "good", "act", "mark", "term", "file", "claim",
                      "right", "grant", "design", "licence", "license"):
        assert forbidden not in PHRASES


def test_the_config_states_what_it_guarantees():
    config = intent_config()
    assert config["uses_model"] is False
    assert config["deterministic"] is True
    assert config["phrase_count"] == len(PHRASES)


# -- capability questions (issue 1, second half) ------------------------------

CAPABILITY_INPUTS = ["help", "Help", "help me", "i need help", "can you help",
                     "how can you help", "how can you help me",
                     "what can you do", "what do you do", "what can i ask",
                     "who are you", "what are you", "how does this work",
                     "How does it work?", "how do i use this", "how to use this"]


@pytest.mark.parametrize("query", CAPABILITY_INPUTS)
def test_a_question_about_the_assistant_never_reaches_retrieval(query):
    """The screenshot bug: "help" abstained instead of saying what this is.

    It carries the same SOCIAL status as a greeting, so the answer, evidence
    and confidence fields stay empty and nothing counts it as a legal answer.
    """
    result = classify(query)
    assert result is not None, f"{query!r} should have been a capability question"
    assert result.intent == CAPABILITY
    assert result.to_dict()["retrieval_performed"] is False


def test_the_capability_reply_says_what_the_corpus_is_and_how_answers_are_made():
    reply = REPLIES[CAPABILITY]
    assert "22" in reply and "corpus" in reply
    assert "citations" in reply
    # It must offer a next step rather than only describing itself.
    assert "Ask me" in reply
    # Short enough to read at a glance: it is a reply to "help", not a manual.
    assert len(reply) <= 320


def test_the_greeting_reply_hands_the_turn_back_to_a_legal_question():
    assert "IP-SAKTI" in REPLIES[GREETING]
    assert "citations" in REPLIES[GREETING]


def test_no_reply_claims_a_confidence_level_or_cites_anything():
    """A social reply must not look like a published legal claim."""
    for intent, reply in REPLIES.items():
        assert "[1]" not in reply, intent
        for word in ("HIGH", "MODERATE", "LOW"):
            assert word not in reply, intent


ORDINARY_LEGAL_QUESTIONS = [
    "How must an application for registration of a trade mark be signed?",
    "What is the term of protection of a registered design?",
    "What does section 3(p) of the Patents Act exclude from patentability?",
    "How do I oppose a trade mark application?",
    "What are the grounds for revocation of a patent?",
]


@pytest.mark.parametrize("query", ORDINARY_LEGAL_QUESTIONS)
def test_ordinary_legal_questions_still_bypass_the_social_layer(query):
    assert classify(query) is None, f"{query!r} was wrongly treated as social"
