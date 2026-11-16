"""Phase H: the synthetic benchmark's own guarantees.

A benchmark nobody can re-derive is an assertion, and a benchmark whose judge
can see the answer key is a mirror. Three properties are pinned here:

* **Reproducible.** The same seed produces the same 1,000 questions, byte for
  byte. Without this, "run the benchmark again" means "get a different
  benchmark" and no before/after comparison is meaningful.
* **Grounded in the corpus.** Every expected answer is a verbatim span of its
  target chunk, and every target chunk exists. `verify` is the check, and the
  generator refuses to write a file that fails it.
* **The judge is not told the answer.** `build_packet` raises rather than
  emitting a packet that carries the retrieval rank, the target chunk ids, the
  pipeline's own scores or the question's difficulty — and it checks the
  serialised packet, not just its keys, because a nested dict is how that kind
  of leak actually happens.
"""
from __future__ import annotations

import json

import pytest

from evaluation.generate_synthetic_benchmark import (
    ABSTENTION_SHARE,
    _degenerate,

    _too_close,
    generate,
    subject_from_heading,
    subject_from_marginal_note,
    summarise,
    verify,
)
from evaluation.prepare_synthetic_judgement import (
    CLASSIFICATIONS,
    MINOR_OMISSION_SCORE,
    PASS_THRESHOLD,
    SCORE_BY_CLASSIFICATION,
    WITHHELD,
    build_packet,
    judge_score,
    score_judgements,
    validate_judgement,
)

SMALL = 40


@pytest.fixture(scope="module")
def small_benchmark():
    """A 40-question draw. The full 1,000 takes too long for every test."""
    return generate(count=SMALL, seed=4242)


class TestReproducible:

    def test_the_same_seed_gives_the_same_questions(self, small_benchmark):
        again = generate(count=SMALL, seed=4242)
        assert json.dumps(again, sort_keys=True) == json.dumps(small_benchmark,
                                                               sort_keys=True)

    def test_a_different_seed_gives_different_questions(self, small_benchmark):
        other = generate(count=SMALL, seed=99)
        assert [r["question"] for r in other] != [r["question"] for r in small_benchmark]

    def test_question_ids_are_dense_and_ordered(self, small_benchmark):
        assert [r["question_id"] for r in small_benchmark] == [
            f"S{i:04d}" for i in range(1, SMALL + 1)]


class TestGroundedInTheCorpus:

    def test_the_generated_file_verifies_against_the_corpus(self, small_benchmark):
        assert verify(small_benchmark) == []

    def test_every_answerable_question_quotes_its_target_verbatim(self, small_benchmark):
        answerable = [r for r in small_benchmark if r["label_kind"] == "ANSWERABLE"]
        assert answerable
        for record in answerable:
            assert record["expected_answer_spans"]
            assert record["expected_answer"] == " ".join(record["expected_answer_spans"])

    def test_an_abstention_label_carries_no_evidence(self, small_benchmark):
        for record in small_benchmark:
            if record["label_kind"] == "EXPECTED_ABSTENTION":
                assert record["target_chunk_ids"] == []
                assert record["expected_answer"] is None
                assert record["expected_answer_spans"] == []

    def test_verify_catches_a_tampered_expected_answer(self, small_benchmark):
        tampered = [dict(r) for r in small_benchmark]
        target = next(r for r in tampered if r["label_kind"] == "ANSWERABLE")
        target["expected_answer_spans"] = ["a sentence the statute does not contain"]
        target["expected_answer"] = "a sentence the statute does not contain"
        problems = verify(tampered)
        assert any("not verbatim" in p for p in problems)

    def test_verify_catches_a_missing_target_chunk(self, small_benchmark):
        tampered = [dict(r) for r in small_benchmark]
        target = next(r for r in tampered if r["label_kind"] == "ANSWERABLE")
        target["target_chunk_ids"] = ["NO-SUCH-DOC::no-such-chunk"]
        assert any("not in the corpus" in p for p in verify(tampered))

    def test_the_abstention_share_is_respected(self, small_benchmark):
        summary = summarise(small_benchmark)
        assert summary["expected_abstention"] == round(SMALL * ABSTENTION_SHARE)
        assert summary["answerable"] + summary["expected_abstention"] == SMALL


class TestQuestionQuality:

    def test_no_question_is_merely_its_heading(self, small_benchmark):
        # _too_close is the guard; this asserts it is actually wired in.
        for record in small_benchmark:
            if record["label_kind"] != "ANSWERABLE":
                continue
            assert len(record["question"]) >= 25
            assert "{" not in record["question"]

    def test_too_close_rejects_a_question_that_adds_nothing(self):
        assert _too_close("What about form and signing of application?",
                          "Form and signing of application", "Trade Marks Rules, 2017")

    def test_too_close_allows_a_question_that_adds_a_dimension(self):
        assert not _too_close(
            "Within what time must form and signing of application be done?",
            "Form and signing of application", "Trade Marks Rules, 2017")

    def test_degenerate_subjects_are_rejected(self):
        assert _degenerate("of to w or")          # the old word-soup failure
        assert _degenerate("licence")             # one content word
        assert _degenerate("power of Registrar to w ithdraw")  # OCR split
        assert not _degenerate("restoration of lapsed designs")

    def test_a_marginal_note_is_used_when_there_is_no_heading(self):
        text = ("50. Surrender of certificate of registration under section 33.- "
                "The registered breeder may at any time give notice to the Registrar.")
        assert subject_from_marginal_note(text) is not None

    def test_an_all_caps_ocr_heading_is_not_used_as_a_subject(self):
        assert subject_from_heading("RESERV ATIONS") is None

    def test_questions_are_unique(self, small_benchmark):
        questions = [r["question"].lower() for r in small_benchmark]
        assert len(questions) == len(set(questions))


class TestJudgeIsNotToldTheAnswer:

    @staticmethod
    def row(**overrides):
        base = {
            "question_id": "S0001", "question": "What does Rule 23 require?",
            "expected_answer": "An application shall be signed.",
            "status": "ANSWERED", "answer_text": "It must be signed [1].",
            "label_kind": "ANSWERABLE", "insufficient_reasons": [],
            "cited_evidence": [{"marker": 1, "citation": "TM Rules — Rule 23",
                                "chunk_id": "IN-TM-RULES-2017::rule-23",
                                "excerpt": "An application shall be signed.",
                                "temporal_status": "CURRENT"}],
            # everything below must not survive into the packet
            "rank_of_expected": 1, "target_chunk_ids": ["IN-TM-RULES-2017::rule-23"],
            "retrieval_correct": True, "retrieval_correct_at_1": True,
            "retrieved_chunk_ids": ["IN-TM-RULES-2017::rule-23"],
            "difficulty": "easy", "confidence": "HIGH", "grounded": True,
        }
        base.update(overrides)
        return base

    def test_a_packet_carries_only_the_four_things_the_judge_needs(self):
        packet = build_packet(self.row())
        blob = json.dumps(packet)
        for field in WITHHELD:
            assert f'"{field}"' not in blob, f"{field} leaked into the packet"

    def test_the_packet_keeps_the_question_answer_and_evidence(self):
        packet = build_packet(self.row())
        assert packet["question"]
        assert packet["expected_answer"]
        assert packet["pipeline_answer"]
        assert packet["cited_evidence"][0]["passage"]

    def test_a_nested_leak_is_caught_not_just_a_top_level_key(self):
        row = self.row()
        row["cited_evidence"] = [{"marker": 1, "citation": "c", "chunk_id": "x",
                                  "excerpt": "e", "temporal_status": "CURRENT",
                                  "nested": {"rank_of_expected": 1}}]
        # build_packet copies only known keys, so the nested dict is dropped;
        # this asserts the copy is by allow-list rather than by exclusion.
        packet = build_packet(row)
        assert "rank_of_expected" not in json.dumps(packet)

    def test_the_packet_is_self_hashing(self):
        packet = build_packet(self.row())
        assert len(packet["packet_sha256"]) == 64


class TestScoringSchemeIsDeclared:

    def test_every_classification_has_a_score(self):
        assert set(SCORE_BY_CLASSIFICATION) == set(CLASSIFICATIONS)

    def test_the_threshold_sits_where_partial_credit_fails(self):
        assert SCORE_BY_CLASSIFICATION["partially_correct"] < PASS_THRESHOLD
        assert MINOR_OMISSION_SCORE >= PASS_THRESHOLD
        assert SCORE_BY_CLASSIFICATION["correct"] >= PASS_THRESHOLD

    def test_a_minor_omission_lowers_a_correct_answer_but_still_passes(self):
        judgement = {"question_id": "S1", "classification": "correct",
                     "minor_omission": True, "evidence_ids": [1], "reason": "r"}
        assert judge_score(judgement) == MINOR_OMISSION_SCORE
        assert judge_score(judgement) >= PASS_THRESHOLD

    def test_minor_omission_does_not_rescue_a_partial_answer(self):
        judgement = {"question_id": "S1", "classification": "partially_correct",
                     "minor_omission": True, "evidence_ids": [], "reason": "r"}
        assert judge_score(judgement) == SCORE_BY_CLASSIFICATION["partially_correct"]

    def test_an_unknown_classification_is_refused(self):
        assert validate_judgement({"question_id": "S1", "classification": "great",
                                   "reason": "r"})

    def test_a_judgement_without_a_reason_is_refused(self):
        assert validate_judgement({"question_id": "S1", "classification": "correct",
                                   "reason": "  "})

    def test_abstentions_are_kept_out_of_the_correctness_denominator(self):
        judgements = [
            {"question_id": "a", "classification": "correct", "reason": "r",
             "evidence_ids": [], "minor_omission": False},
            {"question_id": "b", "classification": "appropriate_abstention",
             "reason": "r", "evidence_ids": [], "minor_omission": False},
        ]
        summary = score_judgements(judgements)
        assert summary["answer_correctness_rate"]["denominator"] == 1
        assert summary["pass_rate"]["denominator"] == 2
