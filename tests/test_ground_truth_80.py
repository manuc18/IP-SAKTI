"""Phase 8: the ground-truth verification schema and its two guarantees.

The two guarantees, stated plainly, because everything else here serves them:

1. **Nothing automated can mark a question VERIFIED.** Mechanical PASS means
   the references resolve. It does not mean a person read the provision.
2. **`golden_questions.jsonl` is never written to.** Corrections are recorded
   beside the original label, never over it.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import re
import os

import pytest

from ai.retrieval.base import load_chunks
from scripts.evaluation import prepare_ground_truth as prep
from scripts.evaluation import review_ground_truth as review
from scripts.evaluation.validate_golden_dataset import load_golden

GOLDEN_PATH = prep.GOLDEN_PATH


@pytest.fixture(scope="module")
def chunks():
    return load_chunks()


@pytest.fixture(scope="module")
def golden():
    return load_golden()


@pytest.fixture(scope="module")
def prepared(golden, chunks):
    """The verification records as the script actually produces them."""
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    mechanical = prep.mechanical_review(golden, chunks)
    fingerprint = prep.corpus_fingerprint(chunks)
    return [prep.new_record(record, mechanical[record["question_id"]], by_id,
                            fingerprint, "2026-08-31T00:00:00+00:00")
            for record in golden]


def find(records, question_id):
    return next(r for r in records if r["question_id"] == question_id)


# -- schema -------------------------------------------------------------------

class TestSchema:
    """Part 6: every record carries the required fields, and null means null."""

    REQUIRED = ("question_id", "verification_status", "reviewer", "reviewed_at",
                "question_valid", "document_correct", "locator_correct",
                "anchors_present", "evidence_sufficient", "temporal_status",
                "required_chunk_ids", "primary_chunk_id", "notes")

    def test_every_record_carries_the_required_fields(self, prepared):
        for record in prepared:
            missing = [f for f in self.REQUIRED if f not in record]
            assert not missing, f"{record['question_id']} missing {missing}"

    def test_one_record_per_question_and_ids_are_unique(self, prepared, golden):
        ids = [r["question_id"] for r in prepared]
        assert len(ids) == len(set(ids)) == len(golden) == 111

    def test_human_judgement_fields_start_null(self, prepared):
        # These four cannot be established without reading the provision, so
        # nothing may pre-fill them - not even a plausible guess.
        for record in prepared:
            for field in ("question_valid", "document_correct", "locator_correct",
                          "evidence_sufficient", "scope_within_corpus"):
                assert record[field] is None, f"{record['question_id']}.{field}"
            assert record["reviewer"] is None
            assert record["reviewed_at"] is None
            assert record["required_chunk_ids"] is None
            assert record["primary_chunk_id"] is None

    def test_temporal_status_is_one_of_the_declared_values(self, prepared):
        allowed = {"CURRENT", "STALE", "UNKNOWN", "MIXED"}
        assert {r["temporal_status"] for r in prepared} <= allowed

    def test_the_golden_claim_is_preserved_alongside_not_instead(self, prepared, golden):
        for record, original in zip(prepared, golden):
            assert record["golden"]["expected_chunks"] == original["expected_chunks"]
            assert record["golden"]["expected_locators"] == original["expected_locators"]
            assert record["golden"]["query"] == original["query"]


# -- guarantee 1: nothing automated may verify --------------------------------

class TestNothingAutomatedCanVerify:

    def test_every_question_starts_unreviewed(self, prepared):
        assert {r["verification_status"] for r in prepared} == {"UNREVIEWED"}
        assert len(prepared) == 111

    def test_the_preparation_script_never_assigns_a_reviewed_state(self):
        # A grep, deliberately. The guarantee is about the whole module, not
        # about the one function a unit test happens to call.
        source = inspect.getsource(prep)
        assignments = re.findall(r'"verification_status":\s*"([A-Z_]+)"', source)
        assignments += re.findall(r'verification_status"\]\s*=\s*"([A-Z_]+)"', source)
        assert assignments == ["UNREVIEWED"], assignments
        for state in prep.REVIEWED_STATES:
            assert state in prep.VERIFICATION_STATES

    def test_mechanical_pass_is_not_verification(self, prepared):
        passing = [r for r in prepared if r["mechanical_status"] == "PASS"]
        assert passing, "the fixture is uninformative if nothing passes"
        for record in passing:
            assert record["verification_status"] == "UNREVIEWED"
            assert record["reviewer"] is None

    def test_mechanical_status_is_a_separate_field_from_verification_status(self, prepared):
        for record in prepared:
            assert record["mechanical_status"] in prep.MECHANICAL_STATES
            assert record["verification_status"] in prep.VERIFICATION_STATES
            assert record["mechanical_status"] != record["verification_status"]

    def test_a_rerun_does_not_overwrite_a_human_decision(self, prepared):
        reviewed = dict(prepared[0])
        reviewed.update(verification_status="VERIFIED", reviewer="a person",
                        reviewed_at="2026-08-31T09:00:00+00:00",
                        question_valid=True, document_correct=True,
                        locator_correct=True, evidence_sufficient=True)
        merged = prep.merge([reviewed], prepared)
        kept = find(merged, reviewed["question_id"])
        assert kept["verification_status"] == "VERIFIED"
        assert kept["reviewer"] == "a person"
        assert kept["evidence_sufficient"] is True
        # ... while the mechanical half is refreshed against the current corpus.
        assert kept["mechanical_status"] == prepared[0]["mechanical_status"]

    def test_a_rerun_does_refresh_an_unreviewed_record(self, prepared):
        stale = dict(prepared[0], mechanical_status="FAIL",
                     mechanical_findings=[{"code": "gone", "severity": "FAIL",
                                           "detail": "from a previous corpus"}])
        merged = prep.merge([stale], prepared)
        assert find(merged, stale["question_id"])["mechanical_status"] != "FAIL"


# -- the mechanical checks ----------------------------------------------------

class TestMechanicalChecks:

    def test_a_missing_chunk_is_a_failure_not_a_warning(self, golden, chunks):
        broken = dict(golden[0], question_id="QX01",
                      expected_chunks=["IN-PATENTS-ACT-1970::no-such-chunk"])
        result = prep.mechanical_review([broken], chunks)["QX01"]
        assert result["mechanical_status"] == "FAIL"
        assert any(f["code"] == "unknown_chunk" for f in result["mechanical_findings"])

    def test_a_missing_anchor_is_a_failure(self, golden, chunks):
        broken = dict(golden[0], question_id="QX02",
                      anchor_terms=["a phrase that occurs in no Indian statute"])
        result = prep.mechanical_review([broken], chunks)["QX02"]
        assert result["mechanical_status"] == "FAIL"
        assert any(f["code"] == "anchor_absent" for f in result["mechanical_findings"])

    def test_an_invalid_locator_is_a_failure(self, golden, chunks):
        # Q001's chunk is section 3; asserting section 99 must not pass.
        broken = dict(golden[0], question_id="QX03", expected_locators=["Section 99"])
        result = prep.mechanical_review([broken], chunks)["QX03"]
        assert result["mechanical_status"] == "FAIL"
        assert any(f["code"] == "locator_mismatch" for f in result["mechanical_findings"])

    def test_a_stale_source_is_a_warning_and_never_a_failure(self, prepared):
        stale = [r for r in prepared
                 if any(f["code"] == "stale_source" for f in r["mechanical_findings"])]
        assert stale, "the corpus contains stale chunks; some question must reach one"
        for record in stale:
            assert record["mechanical_status"] == "WARNING"
            assert record["temporal_status"] in ("STALE", "MIXED")

    def test_a_multi_chunk_target_gets_its_own_review_band(self, golden, chunks, prepared):
        multi = [r for r in golden if len(r["expected_chunks"]) > 1]
        assert multi, "the benchmark has multi-chunk targets"
        by_id = {c.chunk_id: c for c in chunks}
        by_qid = {r["question_id"]: r for r in prepared}
        for record in multi:
            # Never silently collapsed to a single chunk.
            assert len(by_qid[record["question_id"]]["golden"]["expected_chunks"]) > 1
            # And if it carries no warning of its own it sorts to band 2.
            clean = dict(by_qid[record["question_id"]],
                         mechanical_status="PASS", mechanical_findings=[])
            assert prep.priority_for(record, clean, by_id)[0] == 2

    def test_a_weak_anchor_is_flagged_against_the_retrievers_own_depth(self, prepared):
        weak = [r for r in prepared
                if any(f["code"] == "weak_anchor" for f in r["mechanical_findings"])]
        assert weak
        assert prep.WEAK_ANCHOR_THRESHOLD == 50

    def test_the_whole_benchmark_has_no_mechanical_failure(self, prepared):
        failing = [r["question_id"] for r in prepared if r["mechanical_status"] == "FAIL"]
        assert failing == []


# -- the review queue ---------------------------------------------------------

class TestReviewQueue:

    def test_every_question_is_queued_exactly_once(self, golden, chunks, prepared):
        by_id = {c.chunk_id: c for c in chunks}
        queue = prep.build_queue(golden, {r["question_id"]: r for r in prepared}, by_id)
        assert len(queue) == 111
        assert len({row["question_id"] for row in queue}) == 111

    def test_the_queue_is_sorted_hardest_first(self, golden, chunks, prepared):
        by_id = {c.chunk_id: c for c in chunks}
        queue = prep.build_queue(golden, {r["question_id"]: r for r in prepared}, by_id)
        priorities = [row["priority"] for row in queue]
        assert priorities == sorted(priorities)
        assert priorities[0] == 1

    def test_warnings_sort_ahead_of_clean_questions(self, golden, chunks, prepared):
        by_id = {c.chunk_id: c for c in chunks}
        queue = prep.build_queue(golden, {r["question_id"]: r for r in prepared}, by_id)
        first_clean = next(i for i, row in enumerate(queue) if row["priority"] > 1)
        assert all(row["mechanical_status"] != "PASS" for row in queue[:first_clean])


# -- guarantee 2: corrections never overwrite ---------------------------------

class Prompts:
    """A scripted reviewer, so the terminal loop is testable without a terminal."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.asked: list[str] = []

    def __call__(self, question: str) -> str:
        self.asked.append(question)
        assert self.replies, f"the tool asked something unscripted: {question!r}"
        return self.replies.pop(0)


class TestCorrections:

    def test_a_correction_requires_all_four_replacement_fields(self, prepared):
        record = dict(prepared[0])
        record["distinctions"] = dict(record["distinctions"])
        prompts = Prompts("IN-TM-ACT-1999", "Section 9", "IN-TM-ACT-1999::chapter-ii-section-9",
                          "the label names the wrong instrument", "CURRENT")
        status = review.apply_decision(record, "c", "a reviewer", prompts)
        assert status == "CORRECTED"
        correction = record["correction"]
        assert correction["corrected_document"] == "IN-TM-ACT-1999"
        assert correction["corrected_locator"] == "Section 9"
        assert correction["corrected_chunk_id"] == "IN-TM-ACT-1999::chapter-ii-section-9"
        assert correction["reason"]

    def test_a_correction_that_omits_a_field_is_asked_again(self, prepared):
        record = dict(prepared[0])
        record["distinctions"] = dict(record["distinctions"])
        # An empty reply for the document id must not be accepted as an answer.
        prompts = Prompts("", "   ", "IN-TM-ACT-1999", "Section 9",
                          "IN-TM-ACT-1999::chapter-ii-section-9", "reason", "CURRENT")
        review.apply_decision(record, "c", "a reviewer", prompts)
        assert record["correction"]["corrected_document"] == "IN-TM-ACT-1999"
        assert sum(1 for q in prompts.asked if "corrected document" in q) == 3

    def test_a_correction_keeps_the_original_label(self, prepared):
        record = dict(prepared[0])
        record["distinctions"] = dict(record["distinctions"])
        original = list(record["golden"]["expected_chunks"])
        prompts = Prompts("IN-TM-ACT-1999", "Section 9", "IN-TM-ACT-1999::chapter-ii-section-9",
                          "reason", "CURRENT")
        review.apply_decision(record, "c", "a reviewer", prompts)
        assert record["golden"]["expected_chunks"] == original
        assert record["correction"]["original_chunks"] == original

    def test_verified_requires_every_criterion(self, prepared):
        record = dict(prepared[0])
        record["distinctions"] = dict(record["distinctions"])
        # A, B, C, E confirmed; H confirmed; D not. That is not a verification.
        prompts = Prompts("y", "y", "y", "y", "y", "n", "CURRENT",
                          "n", "the chunk lacks the definition")
        status = review.apply_decision(record, "v", "a reviewer", prompts)
        assert status == "NEEDS_REVIEW"
        assert record["verification_status"] == "NEEDS_REVIEW"
        assert record["evidence_sufficient"] is False

    def test_verified_when_every_criterion_holds(self, prepared):
        record = dict(prepared[0])
        record["distinctions"] = dict(record["distinctions"])
        prompts = Prompts("y", "y", "y", "y", "y", "y", "CURRENT", "")
        status = review.apply_decision(record, "v", "a reviewer", prompts)
        assert status == "VERIFIED"
        assert record["reviewer"] == "a reviewer"
        assert record["reviewed_at"]
        # The corpus a verification was performed against is recorded with it.
        assert record["verified_against_corpus"] == record["corpus_fingerprint"]

    def test_a_stale_source_may_still_be_verified(self, prepared):
        stale = next(r for r in prepared if r["temporal_status"] == "STALE")
        record = dict(stale)
        record["distinctions"] = dict(record["distinctions"])
        prompts = Prompts("y", "y", "y", "y", "y", "y", "STALE", "historical question")
        assert review.apply_decision(record, "v", "a reviewer", prompts) == "VERIFIED"
        assert record["temporal_status"] == "STALE"

    def test_skip_records_nothing(self, prepared):
        record = dict(prepared[0])
        assert review.apply_decision(record, "s", "a reviewer", Prompts()) is None
        assert record["verification_status"] == "UNREVIEWED"
        assert record["reviewer"] is None


# -- guarantee 2, the file itself ---------------------------------------------

class TestGoldenDatasetIsNeverWritten:

    def test_every_write_in_both_scripts_targets_a_verification_file(self):
        # Not "the golden path does not appear" - it appears in prose in both
        # modules. What must hold is that no *write* is aimed at it.
        for module in (prep, review):
            source = inspect.getsource(module)
            targets = re.findall(r"(?<!def )write_jsonl\(\s*([^,]+),", source)
            assert set(targets) <= {"args.verification", "args.queue"}, targets
            handles = re.findall(r"""open\(\s*([^,]+),\s*["']w""", source)
            assert set(handles) <= {"path"}, handles

    def test_the_review_tool_writes_only_the_verification_file(self):
        source = inspect.getsource(review)
        assert source.count("write_jsonl(") == 2
        assert "write_jsonl(args.verification, records)" in source
        assert "GOLDEN_PATH" not in source

    def test_preparing_leaves_the_golden_file_byte_identical(self, tmp_path):
        before = hashlib.sha256(open(GOLDEN_PATH, "rb").read()).hexdigest()
        verification = tmp_path / "verification.jsonl"
        queue = tmp_path / "queue.jsonl"
        assert prep.main(["--verification", str(verification),
                          "--queue", str(queue)]) == 0
        after = hashlib.sha256(open(GOLDEN_PATH, "rb").read()).hexdigest()
        assert after == before
        records = [json.loads(line) for line in
                   verification.read_text(encoding="utf-8").splitlines() if line.strip()]
        assert len(records) == 111
        assert {r["verification_status"] for r in records} == {"UNREVIEWED"}

    def test_the_committed_verification_file_has_verified_nothing(self):
        path = prep.VERIFICATION_PATH
        if not os.path.exists(path):
            pytest.skip("prepare_ground_truth.py has not been run")
        records = prep.read_jsonl(path)
        assert len(records) == 111
        assert {r["verification_status"] for r in records} == {"UNREVIEWED"}
        assert sum(1 for r in records if r["verification_status"] == "VERIFIED") == 0
