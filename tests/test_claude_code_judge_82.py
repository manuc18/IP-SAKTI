"""Phase 8.2: the Claude Code execution path for the AI judge.

The judge here is a Claude Code session, not an API client. What is testable is
everything around the judgement: that the evidence it saw is pinned by hash,
that its verdict is checked by the same validator the API path uses, that no
credential is needed and no socket is opened, and that a stored verdict cannot
be silently replaced.

The hashes below are pinned deliberately. A test that recomputes what it checks
proves only that a function is consistent with itself.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json

import pytest

from scripts.evaluation import prepare_ai_judgement as prep
from scripts.evaluation import record_claude_code_judgement as cc
from scripts.evaluation import run_ai_judge as judge

#: Frozen in Phase 8.1 and unchanged since. If one of these moves, a judgement
#: recorded under it is a judgement of something else.
GOLDEN_SHA256 = "23024a3dbbf92e3e32086c762fd3a99ab9ae60bcdf08dda26ca05dea558fe2f1"
PROMPT_SHA256 = "aa15a9c6d9b484403387c3b89dd8648fab0f8ed3bd1ec4f136203cc315399301"
Q027_PACKET_SHA256 = "77d7d9d2d0e6c0de1a82a54363b0c4baec88d73a35ae868052317fed7da42ae9"
CORPUS_FINGERPRINT = "be9d648d9cc1c0b0"


@pytest.fixture(scope="module")
def packets():
    return {p["question_id"]: p for p in judge.read_jsonl(prep.PACKETS_PATH)}


@pytest.fixture(scope="module")
def q027(packets):
    return packets["Q027"]


def a_judgement(**overrides):
    base = {
        "judge_status": "AI_VERIFIED",
        "question_valid": True,
        "document_correct": True,
        "locator_correct": True,
        "evidence_sufficient": True,
        "anchors_supported": True,
        "scope_within_corpus": True,
        "retrieval_target_chunk_ids": ["IN-TM-RULES-2017::part-i-chapter-ii-rule-23"],
        "answer_source_chunk_ids": ["IN-TM-RULES-2017::part-i-chapter-ii-rule-23"],
        "authoritative_document_id": "IN-TM-RULES-2017",
        "current_source_available": True,
        "temporal_status": "CURRENT",
        "correction": None,
        "judge_reasoning": "Reasoning tied to the supplied chunk ids.",
        "judge_confidence": "HIGH",
    }
    return dict(base, **overrides)


def write(tmp_path, payload, name="j.json"):
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return str(path)


# -- what the judge saw is pinned ---------------------------------------------

class TestInputsAreByteIdentical:

    def test_the_golden_dataset_is_unchanged(self):
        actual = hashlib.sha256(open(prep.GOLDEN_PATH, "rb").read()).hexdigest()
        assert actual == GOLDEN_SHA256

    def test_the_frozen_prompt_is_unchanged(self):
        version, _, sha = judge.load_instructions()
        assert version == "1.0.0"
        assert sha == PROMPT_SHA256

    def test_the_q027_packet_is_unchanged(self, q027):
        assert q027["packet_sha256"] == Q027_PACKET_SHA256
        body = {k: v for k, v in q027.items() if k != "packet_sha256"}
        assert prep.canonical_sha256(body) == Q027_PACKET_SHA256

    def test_the_packet_pins_the_corpus_and_the_golden_file(self, q027):
        assert q027["corpus_fingerprint"] == CORPUS_FINGERPRINT
        assert q027["golden_dataset_sha256"] == GOLDEN_SHA256

    def test_the_recorder_never_writes_the_golden_file(self):
        source = inspect.getsource(cc)
        targets = [t for t in
                   __import__("re").findall(r"(?<!def )write_jsonl\(\s*([^,]+),", source)]
        assert targets == ["args.judgements"]
        # The recorder opens exactly two files, both read-only: the judgement
        # the session wrote, and the frozen manifest it must match.
        opened = __import__("re").findall(r"open\(\s*([^,)]+)", source)
        assert set(opened) == {"args.judgement", "args.manifest"}, opened
        assert 'open(' not in source.replace('open(args.judgement, encoding="utf-8")', '')\
            .replace('open(args.manifest, encoding="utf-8")', '')


# -- no API, no credential, no socket -----------------------------------------

class TestNoApiInvolved:

    def test_the_recorder_imports_no_http_client(self):
        tree = ast.parse(inspect.getsource(cc))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        for banned in ("urllib", "urllib.request", "http", "httpx", "requests",
                       "anthropic", "openai", "socket"):
            assert not any(name == banned or name.startswith(banned + ".")
                           for name in imported), banned

    def test_the_recorder_names_no_endpoint_and_no_key(self):
        source = inspect.getsource(cc)
        for leak in ("api.anthropic.com", "x-api-key", "ANTHROPIC_API_KEY",
                     "GROQ_API_KEY", "Authorization"):
            assert leak not in source, leak

    def test_recording_works_with_every_socket_booby_trapped(self, monkeypatch, tmp_path):
        def explode(*_a, **_k):
            raise AssertionError("the recorder opened a socket")
        monkeypatch.setattr(judge, "_post", explode)
        monkeypatch.setattr(judge.urllib.request, "urlopen", explode)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        assert cc.main(["--question", "Q027",
                        "--judgement", write(tmp_path, a_judgement()),
                        "--judgements", str(tmp_path / "out.jsonl")]) == 0

    def test_no_api_measurement_is_manufactured(self, q027):
        record = judge.build_record(q027, a_judgement(), model="claude-opus-5",
                                    prompt_version="1.0.0", prompt_sha=PROMPT_SHA256,
                                    provider=cc.JUDGE_PROVIDER,
                                    temperature=cc.JUDGE_TEMPERATURE,
                                    diagnostics={"execution_mode": "claude_code_session",
                                                 "api_request_made": False})
        assert record["judge_temperature"] is None
        for invented in ("latency_ms", "input_tokens", "output_tokens",
                         "stop_reason"):
            assert invented not in record["diagnostics"], invented


# -- provenance ---------------------------------------------------------------

class TestProvenance:

    def test_the_provider_is_the_session_not_the_api(self):
        assert cc.JUDGE_PROVIDER == "claude_code_session"
        assert cc.JUDGE_PROVIDER != judge.JUDGE_PROVIDER
        assert cc.JUDGE_TEMPERATURE is None

    def test_temperature_zero_is_never_claimed_for_this_path(self):
        source = inspect.getsource(cc)
        assert "JUDGE_TEMPERATURE = None" in source
        assert "temperature = 0" not in source.lower()

    def test_an_unexposed_model_identity_is_unknown_not_invented(self):
        assert cc.DEFAULT_JUDGE_MODEL == "unknown"

    def test_the_model_identity_is_recorded_as_self_reported(self, q027):
        record = judge.build_record(q027, a_judgement(), model="claude-opus-5",
                                    prompt_version="1.0.0", prompt_sha=PROMPT_SHA256,
                                    provider=cc.JUDGE_PROVIDER,
                                    temperature=cc.JUDGE_TEMPERATURE,
                                    diagnostics={"judge_model_source": cc.MODEL_SOURCE})
        assert record["judge_model"] == "claude-opus-5"
        assert record["diagnostics"]["judge_model_source"] == "session_self_report"

    def test_every_part_2_field_is_populated(self, q027):
        record = judge.build_record(q027, a_judgement(), model="claude-opus-5",
                                    prompt_version="1.0.0", prompt_sha=PROMPT_SHA256,
                                    provider=cc.JUDGE_PROVIDER,
                                    temperature=cc.JUDGE_TEMPERATURE, diagnostics={})
        assert judge.missing_record_fields(record) == []
        assert record["packet_sha256"] == Q027_PACKET_SHA256
        assert record["corpus_fingerprint"] == CORPUS_FINGERPRINT
        assert record["golden_dataset_sha256"] == GOLDEN_SHA256
        assert record["judge_prompt_version"] == "1.0.0"
        assert record["judge_prompt_sha256"] == PROMPT_SHA256
        assert record["original_document_id"] == "IN-TM-RULES-2017"
        assert record["original_locator"] == "Rule 23"

    def test_the_api_path_keeps_its_own_provenance(self, q027):
        record = judge.build_record(q027, a_judgement(), model="m",
                                    prompt_version="1.0.0", prompt_sha="s",
                                    diagnostics={})
        assert record["judge_provider"] == "anthropic"
        assert record["judge_temperature"] == 0.0


# -- the same validator, not a second copy ------------------------------------

class TestValidation:

    def test_the_recorder_reuses_the_api_paths_validator(self):
        assert cc.validate_judgement is judge.validate_judgement
        assert cc.build_record is judge.build_record

    def test_only_the_four_ai_statuses_are_accepted(self, q027, tmp_path):
        for bad in ("VERIFIED", "AI_OK", "PASS", None):
            problems = cc.preflight(q027, a_judgement(judge_status=bad), [],
                                    golden_hash=GOLDEN_SHA256,
                                    fingerprint=CORPUS_FINGERPRINT, force=False)
            assert problems, bad

    def test_ai_corrected_requires_its_replacement_fields(self, q027):
        problems = cc.preflight(q027, a_judgement(judge_status="AI_CORRECTED",
                                                  correction=None), [],
                                golden_hash=GOLDEN_SHA256,
                                fingerprint=CORPUS_FINGERPRINT, force=False)
        assert any("correction" in p for p in problems)

        complete = {f: "x" for f in judge.CORRECTION_FIELDS}
        assert cc.preflight(q027, a_judgement(judge_status="AI_CORRECTED",
                                              correction=complete), [],
                            golden_hash=GOLDEN_SHA256,
                            fingerprint=CORPUS_FINGERPRINT, force=False) == []

    def test_needs_review_may_leave_every_judgement_field_null(self, q027):
        undecided = a_judgement(
            judge_status="AI_NEEDS_REVIEW", question_valid=None,
            document_correct=None, locator_correct=None, evidence_sufficient=None,
            anchors_supported=None, scope_within_corpus=None,
            retrieval_target_chunk_ids=None, answer_source_chunk_ids=None,
            authoritative_document_id=None, current_source_available=None,
            temporal_status="UNKNOWN", judge_confidence="LOW")
        assert cc.preflight(q027, undecided, [], golden_hash=GOLDEN_SHA256,
                            fingerprint=CORPUS_FINGERPRINT, force=False) == []

    def test_a_judgement_cannot_attach_to_tampered_evidence(self, q027):
        tampered = json.loads(json.dumps(q027))
        tampered["evidence"][0]["text"] += " (inserted)"
        problems = cc.preflight(tampered, a_judgement(), [],
                                golden_hash=GOLDEN_SHA256,
                                fingerprint=CORPUS_FINGERPRINT, force=False)
        assert any("packet hash" in p for p in problems)

    def test_a_moved_corpus_or_golden_file_blocks_recording(self, q027):
        assert any("corpus" in p for p in cc.preflight(
            q027, a_judgement(), [], golden_hash=GOLDEN_SHA256,
            fingerprint="0000000000000000", force=False))
        assert any("golden_questions" in p for p in cc.preflight(
            q027, a_judgement(), [], golden_hash="deadbeef",
            fingerprint=CORPUS_FINGERPRINT, force=False))


# -- no silent replacement ----------------------------------------------------

class TestNoOverwrite:

    def test_a_terminal_judgement_is_not_replaced(self, q027):
        existing = [{"question_id": "Q027", "judge_status": "AI_VERIFIED"}]
        problems = cc.preflight(q027, a_judgement(), existing,
                                golden_hash=GOLDEN_SHA256,
                                fingerprint=CORPUS_FINGERPRINT, force=False)
        assert any("already carries a terminal judgement" in p for p in problems)

    def test_force_replaces_it_deliberately(self, q027):
        existing = [{"question_id": "Q027", "judge_status": "AI_REJECTED"}]
        assert cc.preflight(q027, a_judgement(), existing,
                            golden_hash=GOLDEN_SHA256,
                            fingerprint=CORPUS_FINGERPRINT, force=True) == []

    def test_another_questions_judgement_is_no_obstacle(self, q027):
        existing = [{"question_id": "Q001", "judge_status": "AI_VERIFIED"}]
        assert cc.preflight(q027, a_judgement(), existing,
                            golden_hash=GOLDEN_SHA256,
                            fingerprint=CORPUS_FINGERPRINT, force=False) == []

    def test_recording_refuses_and_writes_nothing(self, tmp_path):
        out = tmp_path / "out.jsonl"
        out.write_text(json.dumps({"question_id": "Q027",
                                   "judge_status": "AI_VERIFIED",
                                   "judge_reasoning": "the first one"}) + "\n",
                       encoding="utf-8")
        before = out.read_text(encoding="utf-8")
        assert cc.main(["--question", "Q027",
                        "--judgement", write(tmp_path, a_judgement()),
                        "--judgements", str(out)]) == 2
        assert out.read_text(encoding="utf-8") == before


# -- the stored Q027 judgement ------------------------------------------------

class TestTheStoredPilot:

    def test_the_pilot_record_is_present_and_well_formed(self):
        # Phase 8.3 added an authorised batch to this file. What this class
        # asserts is unchanged: the pilot record itself, still as it was
        # recorded, and not disturbed by anything written after it.
        records = {r["question_id"]: r for r in judge.read_jsonl(judge.JUDGEMENTS_PATH)}
        assert "Q027" in records
        record = records["Q027"]
        assert record["judge_status"] in judge.JUDGE_STATUSES
        assert judge.missing_record_fields(record) == []
        assert record["judge_provider"] == "claude_code_session"
        assert record["judge_temperature"] is None
        assert record["packet_sha256"] == Q027_PACKET_SHA256
        assert record["judge_prompt_sha256"] == PROMPT_SHA256
        assert record["golden_dataset_sha256"] == GOLDEN_SHA256
        assert record["diagnostics"]["api_request_made"] is False
        assert record["diagnostics"]["requests"] == 0
        assert record["diagnostics"]["passes"] == 1

    def test_the_reasoning_cites_only_supplied_chunk_ids(self, q027):
        record = next(r for r in judge.read_jsonl(judge.JUDGEMENTS_PATH)
                      if r["question_id"] == "Q027")
        supplied = {e["chunk_id"] for e in q027["evidence"]}
        cited = set(__import__("re").findall(r"[A-Z0-9-]+::[a-z0-9-]+",
                                             record["judge_reasoning"]))
        assert cited
        assert cited <= supplied, cited - supplied

    def test_the_pilot_verdict_survived_the_later_batch(self):
        record = next(r for r in judge.read_jsonl(judge.JUDGEMENTS_PATH)
                      if r["question_id"] == "Q027")
        assert record["judge_status"] == "AI_VERIFIED"
        assert record["judge_confidence"] == "HIGH"
        assert record["packet_sha256"] == Q027_PACKET_SHA256
        assert record["original_locator"] == "Rule 23"
