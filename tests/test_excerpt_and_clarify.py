"""Phase E and Phase F: the excerpt selector and the clarification bound.

Two properties carry the weight here and both are safety properties rather than
quality ones. An excerpt must be a *substring of the passage it claims to quote*
— if that ever stops holding, the system is putting quotation marks around words
the statute does not contain. And a clarification must never change a status —
if it can, "ask the user a question" becomes a route around the abstention gate.

Everything in this file is offline: no model, no network, no corpus load beyond
the chunk fixtures built inline.
"""
from __future__ import annotations

import pytest

from ai.answering.excerpt import (
    MAX_EXCERPT_CHARS,
    Excerpt,
    _display_text,
    excerpt_config,
    select_excerpt,
)
from app.clarify import (
    Clarification,
    anchors,
    clarify_config,
    post_abstention,
    pre_retrieval,
)

RULE_23 = (
    "23. Form and signing of application.— (1) An application for the "
    "registration of a trade mark shall be signed by the applicant or his agent. "
    "(2) An application for the registration of a trademark, for goods or "
    "services shall explain with sufficient precision a description by words of "
    "the trademark. (3) The application shall be filed in Form TM-A and shall be "
    "accompanied by the prescribed fee as specified in the First Schedule. "
    "(4) Where the application is made by a partnership firm, it shall be signed "
    "by at least one partner competent to sign on behalf of the firm."
)


# ---------------------------------------------------------------- excerpts ---

class TestExcerptIsVerbatim:
    """The one property that must never break."""

    @pytest.mark.parametrize("query", [
        "how must an application be signed",
        "what form and fee accompany a trade mark application",
        "partnership firm signature",
        "",
        "completely unrelated words about marine biology",
    ])
    def test_excerpt_is_a_substring_of_the_displayed_chunk(self, query):
        excerpt = select_excerpt(RULE_23, query, max_chars=200)
        assert excerpt is not None
        assert excerpt.text in _display_text(RULE_23)

    @pytest.mark.parametrize("query", [
        "how must an application be signed",
        "what form and fee accompany a trade mark application",
        "partnership firm signature",
    ])
    def test_offsets_address_the_span_they_claim(self, query):
        excerpt = select_excerpt(RULE_23, query, max_chars=200)
        assert _display_text(RULE_23)[excerpt.start:excerpt.end].strip() == excerpt.text

    def test_no_character_of_the_legal_wording_is_rewritten(self):
        excerpt = select_excerpt(RULE_23, "form TM-A and prescribed fee",
                                 max_chars=200)
        # Not lower-cased, not re-punctuated, not ellipsised inside the span.
        assert "..." not in excerpt.text
        assert excerpt.text in _display_text(RULE_23)

    def test_a_short_chunk_is_returned_whole_and_says_so(self):
        excerpt = select_excerpt("Rule 5. Fees shall be as prescribed.", "fees")
        assert excerpt.strategy == "whole_chunk"
        assert excerpt.truncated is False

    def test_empty_text_yields_no_excerpt(self):
        assert select_excerpt("", "anything") is None
        assert select_excerpt("   \n  ", "anything") is None


class TestExcerptSelection:

    def test_it_picks_the_span_the_question_is_about(self):
        excerpt = select_excerpt(RULE_23, "which form must accompany the fee",
                                 max_chars=200)
        assert "Form TM-A" in excerpt.text

    def test_the_answers_own_wording_breaks_a_tie(self):
        excerpt = select_excerpt(
            RULE_23, "application requirements",
            answer_text="A partnership firm's application is signed by a "
                        "competent partner [2].",
            marker=2, max_chars=200)
        assert "partnership firm" in excerpt.text

    def test_only_sentences_citing_this_marker_steer_the_choice(self):
        # The marker-3 sentence talks about fees; this passage is marker 2, so
        # the fee sentence must not pull its excerpt.
        excerpt = select_excerpt(
            RULE_23, "who signs",
            answer_text="Applications are signed by the applicant [2]. "
                        "The prescribed fee is in the First Schedule [3].",
            marker=2, max_chars=200)
        assert "signed" in excerpt.text.lower()

    def test_a_single_long_sentence_is_never_cut_mid_clause(self):
        sentence = ("No person shall manufacture for sale any drug except in "
                    "accordance with a licence, and " + "provided that " * 40
                    + "this subsection shall not apply to a government analyst.")
        excerpt = select_excerpt(sentence + " A second short sentence here.",
                                 "who may manufacture a drug", max_chars=200)
        assert excerpt.text in _display_text(sentence + " A second short sentence here.")
        assert len(excerpt.text) > 200  # returned whole rather than truncated

    def test_window_fallback_for_text_with_no_sentence_breaks(self):
        blob = "PART II " + ("licence fee schedule entry " * 60)
        excerpt = select_excerpt(blob, "licence fee", max_chars=200)
        assert excerpt.strategy == "window"
        assert excerpt.text in _display_text(blob)

    def test_config_declares_the_guarantee(self):
        config = excerpt_config()
        assert config["max_excerpt_chars"] == MAX_EXCERPT_CHARS
        assert "verbatim substring" in config["guarantee"]


# ----------------------------------------------------------- clarification ---

class TestClarificationIsBounded:

    def test_a_query_naming_a_regime_is_never_interrupted(self):
        # The Phase E worked example: inferable, so it must be answered.
        assert pre_retrieval("How can I trademark my own firm?", None) is None

    @pytest.mark.parametrize("query", [
        "How do I register a trade mark in India?",
        "What is the term of copyright in a literary work?",
        "patent fees",
        "Is a new form of a known substance patentable?",
    ])
    def test_anchored_queries_go_straight_to_retrieval(self, query):
        assert pre_retrieval(query, None) is None

    @pytest.mark.parametrize("query", [
        "How can I protect my idea?",
        "What are my rights?",
        "help",
    ])
    def test_only_a_query_anchored_on_nothing_is_stopped(self, query):
        clarification = pre_retrieval(query, None)
        assert isinstance(clarification, Clarification)
        assert clarification.stage == "pre_retrieval"
        assert clarification.question.count("?") == 1

    def test_an_identifier_alone_is_an_anchor(self):
        assert anchors("What is section 84?", None)["anchored"] is True
        assert pre_retrieval("What is section 84?", None) is None

    def test_config_states_the_one_question_bound(self):
        config = clarify_config()
        assert config["max_clarifications_per_request"] == 1
        assert config["uses_model"] is False
        assert config["rewrites_query"] is False


class TestClarificationNeverChangesAStatus:
    """The safety property: this module may add a key, never edit one."""

    def test_an_answered_result_is_never_given_a_clarification(self):
        result = {"status": "ANSWERED"}
        assert post_abstention("How can I protect my idea?", result, None) is None

    def test_an_unavailable_result_is_never_given_a_clarification(self):
        result = {"status": "UNAVAILABLE"}
        assert post_abstention("How can I trademark my firm?", result, None) is None

    def test_an_abstention_keeps_its_status_and_gains_a_suggestion(self):
        result = {"status": "ABSTAINED"}
        suggestion = post_abstention("How can I trademark my own firm?",
                                     result, None)
        assert suggestion is not None
        assert suggestion.stage == "post_abstention"
        assert suggestion.regimes == ("trademarks",)
        assert result["status"] == "ABSTAINED"  # untouched

    def test_a_multi_regime_abstention_asks_which(self):
        suggestion = post_abstention(
            "Is a plant variety protected by copyright or by breeders' rights?",
            {"status": "ABSTAINED"}, None)
        assert suggestion is not None
        assert len(suggestion.regimes) > 1

    def test_the_suggestion_serialises_with_its_bound(self):
        suggestion = post_abstention("How can I trademark my own firm?",
                                     {"status": "ABSTAINED"}, None)
        assert suggestion.to_dict()["max_per_request"] == 1
