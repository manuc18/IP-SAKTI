"""The evidence-sufficiency judge has to be wrong in predictable ways.

It is a deterministic stand-in for reading, and the thing that would make it
worthless is saturation: a metric that says "sufficient" for everything is a
metric that measures nothing. So most of what is pinned here is the negative
side — what it refuses to call sufficient — plus the one property that makes it
more than a restatement of Recall@5: a verbatim span arriving on a chunk whose
id was never the label still counts.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from evaluation.evidence_sufficiency import (  # noqa: E402
    INSUFFICIENT, SUFFICIENT, UNCERTAIN, assess, key_term_coverage,
    span_containment, sufficiency_config, terms)

IDF = {"circumvents": 6.0, "technological": 5.0, "measure": 4.0, "imprisonment": 5.5,
       "trademark": 4.0, "registration": 3.0, "shall": 0.5, "person": 1.0,
       "biscuit": 7.0, "elephant": 7.0, "gramophone": 7.0}

SPAN = ("Any person who circumvents an effective technological measure shall be "
        "punishable with imprisonment which may extend to two years")


def evidence(*pairs):
    return [{"chunk_id": cid, "document_id": "DOC", "text": text, "locator": {}}
            for cid, text in pairs]


def assess_with(retrieved, *, expected=SPAN, spans=None, targets=("T",),
                acceptable=None, depth=5):
    return assess(expected_answer=expected, expected_spans=spans or [SPAN],
                  retrieved=retrieved, target_chunk_ids=list(targets),
                  acceptable_chunk_ids=list(acceptable or targets),
                  target_document_id="DOC", target_locators=[], idf=IDF, depth=depth)


# ----------------------------------------------------------------- terms ----

def test_short_words_are_not_terms():
    assert "the" not in terms("the person shall")
    assert "person" in terms("the person shall")


def test_coverage_is_one_when_the_evidence_contains_the_answer():
    assert key_term_coverage(SPAN, SPAN, IDF) == 1.0


def test_coverage_is_zero_when_nothing_overlaps():
    assert key_term_coverage(SPAN, "biscuit elephant gramophone", IDF) == 0.0


def test_coverage_weights_rare_terms_above_common_ones():
    rare_only = key_term_coverage("circumvents shall", "circumvents", IDF)
    common_only = key_term_coverage("circumvents shall", "shall", IDF)
    assert rare_only > common_only


def test_coverage_of_an_empty_expected_answer_is_zero_not_one():
    assert key_term_coverage("", SPAN, IDF) == 0.0


# ------------------------------------------------------- span containment ----

def test_a_span_is_found_across_whitespace_differences():
    assert span_containment([SPAN], ["prefix  " + SPAN.replace(" ", "\n  ") + " suffix"])


def test_a_short_span_is_never_matched():
    # Forty characters is the floor; below it a "match" is a coincidence.
    assert not span_containment(["two years"], ["imprisonment for two years"])


def test_an_absent_span_is_not_found():
    assert not span_containment([SPAN], ["an entirely different provision"])


# ---------------------------------------------------------------- verdicts ---

def test_a_verbatim_span_is_sufficient():
    got = assess_with(evidence(("T", SPAN)))
    assert got.verdict == SUFFICIENT
    assert got.span_present is True


def test_a_span_on_an_unlabelled_chunk_is_still_sufficient():
    """The property that makes this more than Recall@5 restated.

    Overlapping chunks and duplicated provisions mean the words the answer was
    drawn from can arrive on a chunk id that was never the benchmark's label.
    """
    got = assess_with(evidence(("OTHER", SPAN)))
    assert got.verdict == SUFFICIENT
    assert got.target_included is False
    assert got.acceptable_included is False


def test_unrelated_evidence_is_insufficient():
    got = assess_with(evidence(("X", "biscuit elephant gramophone repeated often")))
    assert got.verdict == INSUFFICIENT


def test_the_right_chunk_id_with_no_substance_is_not_automatically_sufficient():
    """A heading stub carrying the target id must not score as sufficient."""
    got = assess_with(evidence(("T", "Protection of measures.")))
    assert got.verdict != SUFFICIENT
    assert got.target_included is True


def test_partial_overlap_without_a_span_lands_in_uncertain():
    got = assess_with(evidence(("T", "A person shall face imprisonment.")))
    assert got.verdict == UNCERTAIN


def test_uncertain_is_a_reported_verdict_not_a_synonym_for_either_side():
    assert UNCERTAIN not in (SUFFICIENT, INSUFFICIENT)
    assert UNCERTAIN in sufficiency_config()["verdicts"]


# ------------------------------------------------------------------ depth ---

def test_depth_bounds_what_is_read():
    rows = evidence(*[(f"N{i}", "nothing relevant here at all") for i in range(5)])
    rows.append({"chunk_id": "T", "document_id": "DOC", "text": SPAN, "locator": {}})
    assert assess_with(rows, depth=5).verdict != SUFFICIENT
    assert assess_with(rows, depth=10).verdict == SUFFICIENT


def test_locator_agreement_is_read_from_the_retrieved_items():
    rows = [{"chunk_id": "X", "document_id": "DOC", "text": "unrelated biscuit",
             "locator": {"kind": "section", "number": "65A"}}]
    got = assess(expected_answer=SPAN, expected_spans=[SPAN], retrieved=rows,
                 target_chunk_ids=["T"], acceptable_chunk_ids=["T"],
                 target_document_id="DOC",
                 target_locators=[{"kind": "section", "number": "65A"}],
                 idf=IDF, depth=5)
    assert got.locator_agreement is True
    # Locator agreement alone lifts it out of INSUFFICIENT but not to SUFFICIENT.
    assert got.verdict == UNCERTAIN


# ------------------------------------------------------------------ config --

def test_the_config_declares_no_model_and_no_provider():
    config = sufficiency_config()
    assert config["deterministic"] is True
    assert config["uses_model"] is False
    assert config["provider"] is None


def test_the_config_refuses_the_word_accuracy():
    config = sufficiency_config()
    assert "not a measure of legal accuracy" in config["claim_safety"] or \
           "not" in config["claim_safety"]
    assert assess_with(evidence(("T", SPAN))).to_dict()["is_legal_accuracy"] is False
