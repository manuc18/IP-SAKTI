"""Phase 8.3: the frozen provenance manifest, and the first judged batch.

The manifest is the seatbelt. `preflight` already asked whether a packet hashed
to its own recorded hash and whether the corpus and golden file on disk were the
ones it named — but a packet file and a corpus that moved *together* would pass
all of that. The manifest is written once and never rewritten, so it is the only
check that can catch it, and it is what makes "judged against exactly this
evidence, from exactly this corpus and exactly this benchmark version" a
verifiable claim rather than one the code makes about itself.
"""
from __future__ import annotations

import copy
import json
import os

import pytest

from scripts.evaluation import prepare_ai_judgement as prep
from scripts.evaluation import record_claude_code_judgement as cc
from scripts.evaluation import run_ai_judge as judge

GOLDEN_SHA256 = "23024a3dbbf92e3e32086c762fd3a99ab9ae60bcdf08dda26ca05dea558fe2f1"
CORPUS_FINGERPRINT = "be9d648d9cc1c0b0"
BATCH = tuple(f"Q{n:03d}" for n in range(1, 11))


@pytest.fixture(scope="module")
def manifest():
    with open(prep.MANIFEST_PATH, encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def packets():
    return {p["question_id"]: p for p in judge.read_jsonl(prep.PACKETS_PATH)}


@pytest.fixture(scope="module")
def judged():
    return {r["question_id"]: r for r in judge.read_jsonl(judge.JUDGEMENTS_PATH)}


def a_judgement(chunk="IN-PATENTS-ACT-1970::chapter-ii-section-3", **overrides):
    base = {
        "judge_status": "AI_VERIFIED", "question_valid": True,
        "document_correct": True, "locator_correct": True,
        "evidence_sufficient": True, "anchors_supported": True,
        "scope_within_corpus": True,
        "retrieval_target_chunk_ids": [chunk], "answer_source_chunk_ids": [chunk],
        "authoritative_document_id": "IN-PATENTS-ACT-1970",
        "current_source_available": True, "temporal_status": "CURRENT",
        "correction": None, "judge_reasoning": "Tied to the supplied chunk ids.",
        "judge_confidence": "HIGH",
    }
    return dict(base, **overrides)


def check(packet, manifest, **overrides):
    kwargs = dict(golden_hash=GOLDEN_SHA256, fingerprint=CORPUS_FINGERPRINT,
                  force=False, manifest=manifest,
                  prompt_sha=cc.FROZEN_PROMPT_SHA256)
    kwargs.update(overrides)
    return cc.preflight(packet, a_judgement(), [], **kwargs)


# -- the manifest itself ------------------------------------------------------

class TestManifest:

    def test_it_exists_and_covers_every_packet(self, manifest, packets):
        assert manifest["manifest_version"] == "1.0.0"
        assert manifest["packet_count"] == len(packets) == 111
        assert set(manifest["packet_sha256"]) == set(packets)

    def test_it_pins_the_corpus_the_golden_file_and_the_packet_version(self, manifest):
        assert manifest["corpus_fingerprint"] == CORPUS_FINGERPRINT
        assert manifest["golden_dataset_sha256"] == GOLDEN_SHA256
        assert manifest["packet_version"] == prep.PACKET_VERSION

    def test_every_frozen_hash_matches_the_packet_on_disk(self, manifest, packets):
        for question_id, frozen in manifest["packet_sha256"].items():
            assert packets[question_id]["packet_sha256"] == frozen

    def test_rebuilding_does_not_silently_refreeze(self, tmp_path):
        # A manifest that re-freezes whatever is on disk today is a rubber stamp.
        # Rebuilding against a manifest that disagrees must refuse, not rewrite.
        manifest_path = tmp_path / "manifest.json"
        manifest_path.write_text(json.dumps({"manifest_version": "1.0.0",
                                             "packet_count": 1}), encoding="utf-8")
        assert prep.main(["--packets", str(tmp_path / "p.jsonl"),
                          "--manifest", str(manifest_path)]) == 1
        assert json.loads(manifest_path.read_text(encoding="utf-8"))["packet_count"] == 1


# -- the seatbelt fires -------------------------------------------------------

class TestSeatbelt:

    def test_a_matching_packet_passes(self, packets, manifest):
        assert check(packets["Q001"], manifest) == []

    def test_a_packet_hash_that_left_the_manifest_is_refused(self, packets, manifest):
        tampered = copy.deepcopy(manifest)
        tampered["packet_sha256"]["Q001"] = "0" * 64
        problems = check(packets["Q001"], tampered)
        assert any("does not match the frozen manifest" in p for p in problems)

    def test_a_moved_corpus_is_refused(self, packets, manifest):
        tampered = dict(manifest, corpus_fingerprint="ffffffffffffffff")
        assert any("corpus fingerprint" in p for p in check(packets["Q001"], tampered))

    def test_a_moved_golden_dataset_is_refused(self, packets, manifest):
        tampered = dict(manifest, golden_dataset_sha256="deadbeef")
        assert any("golden_questions" in p for p in check(packets["Q001"], tampered))

    def test_a_packet_version_change_is_refused(self, packets, manifest):
        tampered = dict(manifest, packet_version="2.0.0")
        assert any("packet version" in p for p in check(packets["Q001"], tampered))

    def test_a_question_absent_from_the_manifest_is_refused(self, packets, manifest):
        tampered = copy.deepcopy(manifest)
        del tampered["packet_sha256"]["Q001"]
        assert any("not in the frozen manifest" in p
                   for p in check(packets["Q001"], tampered))

    def test_a_prompt_that_is_not_the_frozen_one_is_refused(self, packets, manifest):
        assert any("is not the frozen v1.0.0 prompt" in p
                   for p in check(packets["Q001"], manifest, prompt_sha="0" * 64))

    def test_the_frozen_prompt_hash_is_the_prompt_on_disk(self):
        version, _, sha = judge.load_instructions()
        assert version == cc.FROZEN_PROMPT_VERSION
        assert sha == cc.FROZEN_PROMPT_SHA256

    def test_recording_without_a_manifest_refuses_and_writes_nothing(self, tmp_path):
        judgement = tmp_path / "j.json"
        judgement.write_text(json.dumps(a_judgement()), encoding="utf-8")
        out = tmp_path / "out.jsonl"
        assert cc.main(["--question", "Q001", "--judgement", str(judgement),
                        "--manifest", str(tmp_path / "absent.json"),
                        "--judgements", str(out)]) == 2
        assert not out.exists()


# -- the batch ----------------------------------------------------------------

class TestTheBatch:

    def test_every_question_is_judged_exactly_once(self, judged, packets):
        # Phase 8.3 opened with the ten of BATCH plus the Q027 pilot and was
        # then authorised to complete the benchmark. What must hold now is
        # coverage: one judgement per packet, no duplicates, none invented.
        assert set(judged) == set(packets)
        assert len(judged) == 111
        assert set(BATCH) | {"Q027"} <= set(judged)

    def test_every_record_is_complete_and_valid(self, judged):
        for question_id, record in judged.items():
            assert judge.missing_record_fields(record) == [], question_id
            assert record["judge_status"] in judge.JUDGE_STATUSES
            assert record["judge_confidence"] in judge.CONFIDENCES
            assert record["ai_ground_truth_schema"] == "1.0.0"

    def test_every_record_carries_the_frozen_provenance(self, judged, manifest):
        for question_id, record in judged.items():
            assert record["packet_sha256"] == manifest["packet_sha256"][question_id]
            assert record["corpus_fingerprint"] == CORPUS_FINGERPRINT
            assert record["golden_dataset_sha256"] == GOLDEN_SHA256
            assert record["judge_prompt_version"] == "1.0.0"
            assert record["judge_prompt_sha256"] == cc.FROZEN_PROMPT_SHA256

    def test_every_record_is_the_session_and_not_an_api_call(self, judged):
        for record in judged.values():
            assert record["judge_provider"] == "claude_code_session"
            assert record["judge_temperature"] is None
            assert record["diagnostics"]["api_request_made"] is False
            assert record["diagnostics"]["requests"] == 0
            assert record["diagnostics"]["passes"] == 1

    def test_no_verdict_is_called_verified(self, judged):
        for record in judged.values():
            assert record["judge_status"].startswith("AI_")
            assert "verified" not in record
            assert "verification_status" not in record

    def test_every_reasoning_cites_only_supplied_chunk_ids(self, judged, packets):
        import re
        for question_id, record in judged.items():
            supplied = {e["chunk_id"] for e in packets[question_id]["evidence"]}
            cited = set(re.findall(r"[A-Z0-9-]+::[a-z0-9-]+", record["judge_reasoning"]))
            assert cited, question_id
            assert cited <= supplied, (question_id, cited - supplied)

    def test_a_verified_record_carries_no_correction(self, judged):
        for record in judged.values():
            if record["judge_status"] != "AI_CORRECTED":
                assert record["correction"] is None

    def test_the_golden_dataset_is_still_byte_identical(self):
        import hashlib
        assert hashlib.sha256(
            open(prep.GOLDEN_PATH, "rb").read()).hexdigest() == GOLDEN_SHA256

    def test_the_phase_8_human_review_files_are_untouched(self):
        from scripts.evaluation.prepare_ground_truth import VERIFICATION_PATH
        records = prep.read_jsonl(VERIFICATION_PATH)
        assert len(records) == 111
        assert {r["verification_status"] for r in records} == {"UNREVIEWED"}
