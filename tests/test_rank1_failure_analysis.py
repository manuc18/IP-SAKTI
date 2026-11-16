"""The cluster table is only useful if it cannot mislabel a non-failure.

The ordering of the rules is the whole design: `target_equivalence` and
`corpus_extraction` are checked before any ranking explanation, because calling
a question "boilerplate heading" when the rank-1 chunk was in fact a correct
answer turns a labelling artefact into a false ranking defect — and a false
ranking defect is what would justify changing ranking code that does not need
changing. Those two precedence rules are pinned here.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from evaluation.rank1_failure_analysis import build, classify, is_boilerplate  # noqa: E402


def item(chunk_id, *, citation="C1", heading="A heading", dense=0.5, lexical=0.5,
         exact=0.0, referenced=0.0, score=0.5):
    return {"chunk_id": chunk_id, "citation": citation, "heading": heading,
            "rerank_score": score,
            "components": {"dense_percentile": dense, "lexical_percentile": lexical,
                           "locator_exactness": exact, "locator_referenced": referenced,
                           "idf_coverage": 0.1, "phrase_overlap": 0.1,
                           "document_match": 1.0, "heading_bonus": 0.0,
                           "structural_penalty": 0.0}}


def row(**kwargs):
    base = {"acceptable_chunk_ids": ["T"], "damaged_chunks_in_top5": []}
    base.update(kwargs)
    return base


# ------------------------------------------------------------- boilerplate --

def test_structural_headings_are_recognised():
    assert is_boilerplate("Short title and commencement")
    assert is_boilerplate("Definitions")
    assert is_boilerplate("Power to make rules")


def test_extraction_damage_inside_a_heading_does_not_hide_it():
    """"Sho rt title" is still Short Title.

    The match is on markers rather than on the whole string, so a heading the
    PDF broke mid-word is still recognised. It has to be: those headings are
    exactly the ones that reach rank 1 on questions they do not answer.
    """
    assert is_boilerplate("Sho rt title, extent and commencement")


def test_subject_headings_are_not_boilerplate():
    assert not is_boilerplate("Replacement of national registration")
    assert not is_boilerplate("Notice of opposition to registration")
    assert not is_boilerplate(None)


# ------------------------------------------------------------- precedence ---

def test_a_rank1_that_is_itself_acceptable_is_never_called_a_ranking_failure():
    """The rule that stops the table inventing defects.

    Rank 1 here has a boilerplate heading and would match four later rules, but
    it is in the acceptable set — it answered the question.
    """
    winner = item("W", heading="Definitions", citation="C9")
    target = item("T", citation="C1")
    got = classify(row(acceptable_chunk_ids=["T", "W"]), winner, target)
    assert got == "target_equivalence"


def test_extraction_damage_outranks_every_ranking_explanation():
    winner = item("W", heading="Definitions", citation="C9")
    target = item("T", citation="C1")
    got = classify(row(damaged_chunks_in_top5=["W"]), winner, target)
    assert got == "corpus_extraction"


def test_damage_on_the_target_counts_too():
    winner = item("W", heading="Notice of opposition", citation="C9")
    target = item("T", citation="C1")
    assert classify(row(damaged_chunks_in_top5=["T"]), winner, target) == "corpus_extraction"


# ---------------------------------------------------------------- clusters --

def test_two_halves_of_one_provision_are_a_split_not_a_ranking_failure():
    winner = item("W", citation="Patents Act, 1970 — Section 3")
    target = item("T", citation="Patents Act, 1970 — Section 3")
    assert classify(row(), winner, target) == "same_provision_split"


def test_boilerplate_is_reported_when_nothing_earlier_applies():
    winner = item("W", heading="Short title and commencement", citation="C9")
    target = item("T", citation="C1")
    assert classify(row(), winner, target) == "boilerplate_heading"


def test_two_chunks_claiming_the_same_provision_are_identifier_ambiguity():
    winner = item("W", heading="Opposition", citation="C9", exact=1.0)
    target = item("T", citation="C1", exact=1.0)
    assert classify(row(), winner, target) == "identifier_ambiguity"


def test_mentioning_a_provision_without_being_it_is_a_cross_reference():
    winner = item("W", heading="Opposition", citation="C9", exact=0.0, referenced=1.0)
    target = item("T", citation="C1", exact=0.0)
    assert classify(row(), winner, target) == "cross_reference"


def test_winning_on_lexical_and_losing_on_dense_is_broad_lexical():
    winner = item("W", heading="Opposition", citation="C9", lexical=0.9, dense=0.2)
    target = item("T", citation="C1", lexical=0.3, dense=0.8)
    assert classify(row(), winner, target) == "broad_lexical"


def test_winning_on_dense_and_losing_on_lexical_is_dense_confusion():
    winner = item("W", heading="Opposition", citation="C9", lexical=0.2, dense=0.9)
    target = item("T", citation="C1", lexical=0.8, dense=0.3)
    assert classify(row(), winner, target) == "dense_confusion"


def test_winning_on_both_axes_is_other_rather_than_a_guess():
    winner = item("W", heading="Opposition", citation="C9", lexical=0.9, dense=0.9)
    target = item("T", citation="C1", lexical=0.3, dense=0.3)
    assert classify(row(), winner, target) == "other"


# -------------------------------------------------------------- selection ---

def _full_row(qid, rank, evidence):
    return {"question_id": qid, "question": "q", "label_kind": "ANSWERABLE",
            "rank_of_target": rank, "target_chunk_ids": ["T"],
            "acceptable_chunk_ids": ["T"], "damaged_chunks_in_top5": [],
            "evidence": evidence}


def test_only_rank_2_to_5_questions_are_selected():
    evidence = [item("W", citation="C9", heading="Opposition"), item("T", citation="C1")]
    rows = [_full_row("Q1", 1, evidence), _full_row("Q2", 3, evidence),
            _full_row("Q3", 7, evidence), _full_row("Q4", None, evidence)]
    report = build(rows)
    assert report["selected"] == 1
    assert report["table"][0]["question_id"] == "Q2"


def test_the_shares_are_reported_against_the_selected_count():
    evidence = [item("W", citation="C9", heading="Opposition"), item("T", citation="C1")]
    report = build([_full_row(f"Q{i}", 2, evidence) for i in range(4)])
    assert report["selected"] == 4
    assert sum(report["cluster_share"].values()) == 1.0
