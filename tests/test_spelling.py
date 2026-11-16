"""Issue 3: typo tolerance that cannot damage legal terminology or identifiers.

The vocabulary under test is the real corpus, because the module's entire
correctness argument rests on the corpus being its dictionary. A fixture
vocabulary would test the arithmetic and none of the claims.
"""
from __future__ import annotations

import pytest

from ai.retrieval.base import load_chunks
from app.spelling import (
    MIN_TARGET_FREQUENCY_FOR_DISTANCE,
    MIN_WORD_LENGTH,
    CorpusSpelling,
    skeleton,
)


@pytest.fixture(scope="module")
def speller() -> CorpusSpelling:
    return CorpusSpelling(chunk.text for chunk in load_chunks())


#: The misspellings named in the brief, plus the families they stand for.
TYPOS = [
    ("trademrk", "trademark"), ("tradmark", "trademark"),
    ("trade markk", "trade mark"), ("patant", "patent"),
    ("copywrite", "copyright"), ("copywright", "copyright"),
    ("geographcal", "geographical"), ("registraton", "registration"),
    ("applcation", "application"), ("infrigement", "infringement"),
    ("oppositon", "opposition"), ("specfication", "specification"),
    ("certifcate", "certificate"), ("biodiversty", "biodiversity"),
    ("inventon", "invention"), ("desgin", "design"),
    ("complaince", "compliance"), ("paetnt", "patent"),
]

#: Words from other subjects entirely. None of these is a misspelling of
#: anything, and correcting one would put a word the user never typed into the
#: audit trail — and, worse, into retrieval.
OUT_OF_DOMAIN = ["pizza", "topping", "toppings", "naples", "mala", "fides",
                 "football", "cricket", "sushi", "ferrari", "bitcoin",
                 "restaurant", "weather", "recipe", "guitar", "volcano",
                 "lasagna", "espresso", "tsunami", "karaoke", "burrito"]


@pytest.mark.parametrize("typed,expected", TYPOS)
def test_a_misspelling_of_a_corpus_word_is_repaired(speller, typed, expected):
    assert speller.correct(typed).corrected == expected


@pytest.mark.parametrize("word", OUT_OF_DOMAIN)
def test_a_word_from_another_subject_is_left_exactly_as_typed(speller, word):
    correction = speller.correct(word)
    assert correction.corrected == word, (
        f"{word!r} was rewritten to {correction.corrected!r}; it is not a "
        f"misspelling of anything in this corpus")
    assert not correction.applied


def test_the_whole_off_topic_question_survives_untouched(speller):
    question = "What is the best pizza topping in Naples?"
    assert speller.correct(question).corrected == question


@pytest.mark.parametrize("query", [
    "section 85B",
    "Section 85B of the Trade Marks Act",
    "section 3(p)",
    "rule 31bis",
    "Article 6ter of the Paris Convention",
    "Form 24A",
    "What does section 25(2) provide?",
])
def test_identifiers_are_never_rewritten(speller, query):
    assert speller.correct(query).corrected == query


def test_a_word_the_corpus_uses_is_never_corrected(speller):
    """The protection for legal terminology, stated as the rule that provides it."""
    for word in ("bis", "ter", "licence", "licensing", "proprietor", "opposition",
                 "assignee", "prescribed", "aforesaid", "hereinafter", "novelty"):
        if word in speller.vocabulary:
            assert speller.correct(word).corrected == word


def test_short_words_are_out_of_scope(speller):
    for word in ("abc", "xyz", "qqq"):
        assert len(word) < MIN_WORD_LENGTH
        assert speller.correct(word).corrected == word


def test_the_original_query_is_kept_for_audit(speller):
    correction = speller.correct("How do I register a tradmark?")
    assert correction.original == "How do I register a tradmark?"
    assert correction.corrected == "How do I register a trademark?"
    assert correction.applied
    payload = correction.to_dict()
    assert payload["original_query"] == correction.original
    assert payload["search_query"] == correction.corrected
    assert payload["replacements"] == [
        {"from": "tradmark", "to": "trademark", "rule": "skeleton_distance_1",
         "document_frequency": payload["replacements"][0]["document_frequency"]}]


def test_case_is_preserved_through_a_correction(speller):
    assert speller.correct("Tradmark").corrected == "Trademark"
    assert speller.correct("TRADMARK").corrected == "TRADEMARK"
    assert speller.correct("tradmark").corrected == "trademark"


def test_a_distance_correction_must_point_at_a_common_word(speller):
    """The constant that separates a typo from a foreign word, tested as such."""
    for word in OUT_OF_DOMAIN:
        replacement = speller._correct_word(word)
        assert replacement is None or (
            replacement.document_frequency >= MIN_TARGET_FREQUENCY_FOR_DISTANCE)


def test_skeleton_folds_only_spellings_that_sound_alike():
    assert skeleton("copywrite") == skeleton("copyright")
    assert skeleton("markk") == skeleton("mark")
    assert skeleton("phase") == skeleton("fase")
    # And does not fold words that merely look similar.
    assert skeleton("grant") != skeleton("grand")
    assert skeleton("patent") != skeleton("patient")
    assert skeleton("licence") != skeleton("licensee")


def test_correction_is_idempotent(speller):
    once = speller.correct("How do I register a tradmark?").corrected
    assert speller.correct(once).corrected == once


def test_an_empty_query_is_handled(speller):
    assert speller.correct("").corrected == ""
    assert speller.correct(None).corrected == ""


# -- issue 3: the corrected string is what retrieval actually sees ------------
#
# The module was already repairing every misspelling the brief names. What no
# test pinned was the half that matters to the user: that the repaired term
# reaches the retriever, and that the pipeline searches for it rather than for
# what was typed. `app/pipeline.answer` computes `search_query` from
# `speller.correct(question)` and passes *that* to candidate generation,
# reranking and excerpt selection, while the generator is asked the original.

BRIEF_TYPOS = [
    ("trademrk", "trademark"),
    ("copyrght", "copyright"),
    ("patnt", "patent"),
    ("regstration", "registration"),
    ("geographcal indication", "geographical indication"),
]


@pytest.mark.parametrize("typed,term", BRIEF_TYPOS)
def test_the_corrected_query_carries_the_intended_legal_term(speller, typed, term):
    assert speller.correct(typed).corrected == term


@pytest.mark.parametrize("typed,term", BRIEF_TYPOS)
def test_the_term_survives_inside_a_whole_question(speller, typed, term):
    """A correction that only works on a bare word is not a correction."""
    question = f"What is the procedure for {typed} in India?"
    corrected = speller.correct(question).corrected
    assert term in corrected
    assert typed not in corrected
    assert corrected.startswith("What is the procedure for ")
    assert corrected.endswith(" in India?")


@pytest.mark.parametrize("typed,term", BRIEF_TYPOS)
def test_the_correction_is_reported_so_the_transcript_shows_both(speller, typed, term):
    correction = speller.correct(f"how do I file for {typed}")
    assert correction.original != correction.corrected
    assert correction.replacements, "a silent rewrite is not auditable"
    assert any(r.corrected in term.split() for r in correction.replacements)


def test_the_tokenizer_already_folds_trademark_onto_trade_mark():
    """Why "trademrk" -> "trademark" is enough to reach a "trade mark" corpus.

    The Act writes "trade mark" and the Rules write "trademark", so the lexical
    index opens the closed compound on both sides. Spelling repair does not have
    to know about the split, and must not try to.
    """
    from ai.retrieval.tokenizer import tokenize

    assert tokenize("trademark") == tokenize("trade mark") == ["trade", "mark"]
    assert tokenize("trade-mark") == ["trade", "mark"]


def test_a_correction_never_rewrites_a_provision_identifier(speller):
    """Restated against the brief's own list, because this is the unsafe one."""
    for query in ("Section 3(p) of the Patents Act", "Rule 23", "Article 31bis",
                  "section 85B", "sub-section (2) of section 25",
                  "Rule 31bis and Form TM-A", "the 1999 Act"):
        assert speller.correct(query).corrected == query


def test_the_pipeline_searches_for_the_correction_and_asks_about_the_original():
    """The seam itself, read from the source rather than asserted about it.

    Running the pipeline here would load BGE-M3 for a claim that is structural,
    so the ordering is checked where it is written.
    """
    import inspect

    from app import pipeline as pipeline_module

    source = inspect.getsource(pipeline_module.Pipeline.answer)
    assert "correction = self.speller.correct(question)" in source
    assert "search_query = correction.corrected" in source
    assert "self.generator.generate(search_query)" in source
    assert "self.reranker.rerank(search_query" in source
    # The provider is asked what the user typed, not what was searched for.
    assert "self.engine.answer(question, evidence)" in source
