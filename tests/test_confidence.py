"""Issue 2: the confidence label must describe the evidence, not decorate the UI.

Every test here is a claim about what a level *means*. The ones that matter most
are the negative ones: that nothing stale, weak or conflicting can be promoted,
whatever the answer looks like.
"""
from __future__ import annotations

import pytest

from ai.answering.confidence import (
    assess,
    confidence_config,
    source_confidence,
    support_confidence,
)
from ai.answering.models import CURRENT, HIGH, LOW, MODERATE, STALE, Citation, SourceDocument


def _citation(marker=1, *, level="A", status=CURRENT, chunk="c1") -> Citation:
    return Citation(
        marker=marker, chunk_id=chunk, document_id="DOC", locator="Section 1",
        pages=(1,), citation_text="Act — Section 1",
        source_evidence_level=level, temporal_status=status,
        source_document=SourceDocument(document_id="DOC"))


class _Grounding:
    def __init__(self, grounded=True, weakly=()):
        self.grounded = grounded
        self.weakly_grounded = tuple(weakly)


# -- the cases the brief names -----------------------------------------------

def test_strong_answer_from_a_verified_current_source_is_high():
    report = assess([_citation()], _Grounding(), conflicts=False, best_cited_rank=1)
    assert report.level == HIGH
    assert report.source_confidence == HIGH
    assert report.support_confidence == HIGH


def test_weak_provenance_is_low():
    for level in ("C", "D"):
        report = assess([_citation(level=level)], _Grounding(),
                        conflicts=False, best_cited_rank=1)
        assert report.level == LOW
        assert any(f.code == "weak_provenance" for f in report.factors)


def test_conflicting_provisions_are_low_and_the_conflict_is_named():
    report = assess([_citation()], _Grounding(), conflicts=True, best_cited_rank=1)
    assert report.level == LOW
    assert any(f.code == "conflicting_provisions" for f in report.factors)


def test_stale_evidence_is_never_high():
    """Both the all-stale and the partly-stale case, which used to disagree."""
    all_stale = assess([_citation(status=STALE)], _Grounding(),
                       conflicts=False, best_cited_rank=1)
    assert all_stale.level == LOW
    assert any(f.code == "all_sources_stale" for f in all_stale.factors)

    mixed = assess([_citation(1, status=CURRENT, chunk="c1"),
                    _citation(2, status=STALE, chunk="c2")],
                   _Grounding(), conflicts=False, best_cited_rank=1)
    assert mixed.level == MODERATE
    assert any(f.code == "some_sources_stale" for f in mixed.factors)


def test_an_abstention_carries_no_misleading_confidence():
    """No citations means no claim about evidence, and the level says so."""
    report = assess([], _Grounding(grounded=False), conflicts=False,
                    best_cited_rank=None)
    assert report.level == LOW
    assert all(f.code == "no_citations" for f in report.factors)


# -- the two axes, and that they can only ever cap ---------------------------

def test_an_unverified_local_copy_caps_at_moderate_however_good_the_answer_is():
    report = assess([_citation(level="B")], _Grounding(),
                    conflicts=False, best_cited_rank=1)
    assert report.level == MODERATE
    assert report.support_confidence == HIGH
    assert any(f.code == "local_copy_unverified" for f in report.factors)


def test_weak_textual_support_demotes_an_impeccable_source():
    report = assess([_citation()], _Grounding(weakly=("one sentence",)),
                    conflicts=False, best_cited_rank=1)
    assert report.source_confidence == HIGH
    assert report.support_confidence == MODERATE
    assert report.level == MODERATE


def test_resting_on_a_passage_retrieval_ranked_low_demotes_the_answer():
    report = assess([_citation()], _Grounding(), conflicts=False, best_cited_rank=4)
    assert report.source_confidence == HIGH
    assert report.support_confidence == MODERATE
    assert report.level == MODERATE
    assert any(f.code == "low_ranked_evidence" for f in report.factors)


def test_the_published_level_is_the_lower_of_the_two_axes():
    report = assess([_citation(level="B")], _Grounding(weakly=("x",)),
                    conflicts=False, best_cited_rank=9)
    assert report.level == MODERATE
    worst = assess([_citation(level="C")], _Grounding(grounded=False),
                   conflicts=True, best_cited_rank=9)
    assert worst.level == LOW


@pytest.mark.parametrize("level,status", [
    ("B", CURRENT), ("C", CURRENT), ("D", CURRENT),
    ("A", STALE), ("B", STALE), ("C", STALE),
])
def test_only_level_a_and_current_can_ever_reach_high(level, status):
    """The promotion guard, exhaustively.

    There is no combination of a perfect answer and an imperfect source that
    produces HIGH. This is the test that fails if someone later decides the
    demo would look better with more green badges.
    """
    report = assess([_citation(level=level, status=status)], _Grounding(),
                    conflicts=False, best_cited_rank=1)
    assert report.level != HIGH
    assert report.source_confidence != HIGH


def test_a_source_side_factor_can_never_raise_a_level():
    plain, _ = source_confidence([_citation(level="C", status=STALE)],
                                 conflicts=True)
    assert plain == LOW


def test_every_level_carries_at_least_one_stated_reason():
    for citations, grounding, conflicts, rank in (
            ([_citation()], _Grounding(), False, 1),
            ([_citation(level="B")], _Grounding(), False, 1),
            ([_citation(status=STALE)], _Grounding(), False, 1),
            ([_citation()], _Grounding(weakly=("x",)), False, 3),
    ):
        report = assess(citations, grounding, conflicts=conflicts,
                        best_cited_rank=rank)
        assert report.factors, "a level with no stated reason is the old bug"
        payload = report.to_dict()
        assert payload["factors"] and all(
            f["code"] and f["detail"] for f in payload["factors"])


def test_confidence_is_not_presented_as_a_probability():
    assert confidence_config()["is_probability"] is False
    assert confidence_config()["promotes_stale_or_unverified_sources"] is False
    assert assess([_citation()], _Grounding(), conflicts=False,
                  best_cited_rank=1).to_dict()["is_probability"] is False


def test_support_axis_alone_never_reports_high_without_citations():
    level, factors = support_confidence(_Grounding(), [], None)
    assert level == LOW
    assert factors


# -- issue 2: a conflict between passages the answer never cited --------------
#
# `apply_policy` detects conflicts across every supplied passage, and the engine
# used to hand that whole flag to `assess`. On the 60-question sample it was the
# only cause of 3 of the 15 LOW answers, and in each one the conflicting pair
# belonged to a different instrument from the one cited: an answer resting on GI
# Act s.26 alone was published LOW because two TRIPS articles elsewhere in the
# evidence set render the same citation string. The engine now scopes the flag
# to conflicts that touch a cited chunk; these tests pin both directions.

def test_a_conflict_among_passages_the_answer_cites_still_forces_low():
    """The safety half. Unchanged behaviour, stated as a test."""
    report = assess([_citation(chunk="c1")], _Grounding(),
                    conflicts=True, best_cited_rank=1)
    assert report.level == LOW
    assert any(f.code == "conflicting_provisions" for f in report.factors)


def test_a_clean_citation_is_not_demoted_by_a_conflict_elsewhere_in_the_set():
    """The bug half, at the level `assess` sees it: no conflict, no penalty."""
    report = assess([_citation(chunk="c1", level="B")], _Grounding(),
                    conflicts=False, best_cited_rank=1)
    assert report.level == MODERATE
    assert not any(f.code == "conflicting_provisions" for f in report.factors)


def test_the_scoping_itself_is_tested_against_the_real_engine():
    """Where the scoping lives, and where it is actually exercised.

    `assess` only ever sees a boolean, so it cannot tell a foreign conflict from
    a cited one — deciding which is which is the engine's job. That decision is
    pinned end to end in `tests/test_answer_publication.py`, which drives
    `GroundedAnswerEngine` over an evidence set carrying a conflict the answer
    does not cite; this line records the seam so the two files are not read
    apart.
    """
    import tests.test_answer_publication as publication

    assert hasattr(publication,
                   "test_a_conflict_touching_no_cited_chunk_does_not_demote_the_answer")
    assert hasattr(publication,
                   "test_a_conflict_that_does_touch_a_cited_chunk_still_forces_low")
