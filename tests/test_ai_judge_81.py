"""Phase 8.1: the AI judge, offline.

Nothing here makes a network request, and one test exists specifically to prove
the module cannot make one without `--live`.

Three guarantees:

1. `golden_questions.jsonl` stays byte-identical.
2. Packets are deterministic, and each carries a hash of its own content.
3. Nothing reaches the judge that would tell it what the label's authors thought
   of the label.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re

import pytest

from ai.retrieval.base import load_chunks
from scripts.evaluation import prepare_ai_judgement as prep
from scripts.evaluation import run_ai_judge as judge
from scripts.evaluation.validate_golden_dataset import load_golden

GOLDEN_PATH = prep.GOLDEN_PATH


@pytest.fixture(scope="module")
def chunks():
    return load_chunks()


@pytest.fixture(scope="module")
def golden():
    return load_golden()


@pytest.fixture(scope="module")
def packets(golden, chunks):
    return prep.build_all(golden, chunks, prep.golden_sha256())


@pytest.fixture(scope="module")
def instructions():
    return judge.load_instructions()


def packet(packets, question_id):
    return next(p for p in packets if p["question_id"] == question_id)


def a_valid_judgement(**overrides):
    return dict(judge.FIXTURE_RESPONSE, **overrides)


# -- guarantee 1 --------------------------------------------------------------

class TestGoldenFileUnchanged:

    def test_the_golden_file_is_byte_identical_after_preparing(self, tmp_path):
        before = hashlib.sha256(open(GOLDEN_PATH, "rb").read()).hexdigest()
        assert prep.main(["--packets", str(tmp_path / "packets.jsonl")]) == 0
        assert hashlib.sha256(open(GOLDEN_PATH, "rb").read()).hexdigest() == before

    def test_the_recorded_golden_hash_is_the_file_on_disk(self, packets):
        expected = hashlib.sha256(open(GOLDEN_PATH, "rb").read()).hexdigest()
        assert {p["golden_dataset_sha256"] for p in packets} == {expected}

    def test_no_judge_write_can_reach_the_golden_file(self):
        # Not "the path is not mentioned" - it is mentioned, in prose. What must
        # hold is that no write is aimed at it.
        for module in (prep, judge):
            source = inspect.getsource(module)
            targets = re.findall(r"(?<!def )write_jsonl\(\s*([^,]+),", source)
            assert set(targets) <= {"args.packets", "args.judgements"}, targets
            handles = re.findall(r"""open\(\s*([^,]+),\s*["']w""", source)
            # The manifest is the only file either script opens for writing
            # outside write_jsonl; it is a sidecar of the packets, never the
            # golden dataset.
            assert set(handles) <= {"args.manifest"}, handles

    def test_the_judge_never_opens_the_golden_file_at_all(self):
        source = inspect.getsource(judge)
        assert "golden_questions" not in source.replace(
            "golden_questions.jsonl was not written to.", "")


# -- guarantee 2: determinism -------------------------------------------------

class TestPacketDeterminism:

    def test_building_twice_gives_identical_packets(self, golden, chunks):
        first = prep.build_all(golden, chunks, "constant-hash")
        second = prep.build_all(golden, chunks, "constant-hash")
        assert first == second

    def test_a_packet_carries_no_timestamp(self, packets):
        blob = json.dumps(packets)
        for word in ("generated_at", "timestamp", "judged_at", "created_at"):
            assert word not in blob

    def test_the_packet_hash_covers_the_packet(self, packets):
        for item in packets:
            body = {k: v for k, v in item.items() if k != "packet_sha256"}
            assert item["packet_sha256"] == prep.canonical_sha256(body)

    def test_the_packet_hash_moves_when_the_evidence_moves(self, packets):
        original = packet(packets, "Q027")
        body = {k: v for k, v in original.items() if k != "packet_sha256"}
        tampered = json.loads(json.dumps(body))
        tampered["evidence"][0]["text"] += " (one extra word)"
        assert prep.canonical_sha256(tampered) != original["packet_sha256"]

    def test_every_question_gets_exactly_one_packet(self, packets, golden):
        assert len(packets) == len(golden) == 111
        assert len({p["question_id"] for p in packets}) == 111

    def test_the_rendered_prompt_is_a_function_of_the_packet_alone(self, packets):
        one = packet(packets, "Q027")
        assert judge.render_user_message(one) == judge.render_user_message(one)

    def test_the_corpus_fingerprint_is_recorded_on_every_packet(self, packets, chunks):
        from scripts.evaluation.prepare_ground_truth import corpus_fingerprint
        assert {p["corpus_fingerprint"] for p in packets} == {corpus_fingerprint(chunks)}


# -- guarantee 3: independence ------------------------------------------------

class TestTheJudgeIsNotToldTheAnswer:

    def test_the_labels_own_provenance_never_reaches_a_packet(self, packets):
        blob = json.dumps(packets, ensure_ascii=False)
        for field in prep.WITHHELD_FROM_JUDGE:
            assert f'"{field}"' not in blob, field

    def test_the_benchmark_notes_never_reach_the_rendered_prompt(self, packets, golden):
        # Q001's note argues for its own label: "s.3(p) is the traditional-
        # knowledge exclusion; it sits inside the single s.3 chunk." Handing the
        # auditor the defence's argument is the circularity Part 8 forbids.
        note = next(r for r in golden if r["question_id"] == "Q001")["notes"]
        assert note
        assert note not in judge.render_user_message(packet(packets, "Q001"))

    def test_the_claim_is_presented_as_a_claim(self, packets):
        message = judge.render_user_message(packet(packets, "Q027"))
        assert "this is the claim under audit, not a fact" in message

    def test_no_retrieval_output_is_used(self, packets):
        blob = json.dumps(packets)
        for word in ("rerank_score", "dense_score", "lexical_score", "rank"):
            assert f'"{word}"' not in blob
        assert {p["evidence_source"] for p in packets} == {
            "the designated chunks plus their structural neighbours in the same "
            "instrument, taken in document order; no retrieval or reranking "
            "output was used"}

    def test_the_judge_does_not_import_the_answer_layer(self):
        # Parsed, not grepped: both modules discuss `ai.answering` in prose, and
        # the guarantee is about imports.
        for module in (prep, judge):
            tree = ast.parse(inspect.getsource(module))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module)
            assert not any(name.startswith("ai.answering") for name in imported), imported

    def test_neighbours_come_from_document_order_within_one_instrument(self, packets):
        one = packet(packets, "Q027")
        document = one["benchmark_claim"]["document_id"]
        neighbours = [e for e in one["evidence"] if e["role"] == "neighbour"]
        assert neighbours
        assert all(e["document_id"] == document for e in neighbours)


# -- the prompt ---------------------------------------------------------------

class TestPrompt:

    def test_the_prompt_is_read_from_the_versioned_document(self, instructions):
        version, text, sha = instructions
        assert version == judge.JUDGE_PROMPT_VERSION == "1.0.0"
        assert sha == prep.canonical_sha256(text)

    def test_the_script_keeps_no_second_copy_of_the_instructions(self, instructions):
        _, text, _ = instructions
        first_rule = "Do not assume the existing target is correct."
        assert first_rule in text
        assert first_rule not in inspect.getsource(judge)

    def test_the_prompt_states_the_five_distinctions(self, instructions):
        _, text, _ = instructions
        assert "STALE is not, by that fact, incorrect" in text
        assert "does not prove the target is correct" in text
        assert "sufficient to answer the question" in text
        assert "Do not assume the existing target is correct" in text
        assert "Do not use outside knowledge" in text

    def test_the_prompt_asks_about_the_designation_not_the_answer(self, instructions):
        _, text, _ = instructions
        assert "justifies the benchmark's designation" in text
        assert "NOT being asked whether an answer is correct" in text

    def test_uncertainty_is_routed_to_needs_review_not_to_a_guess(self, instructions):
        _, text, _ = instructions
        assert "return AI_NEEDS_REVIEW. Do not guess." in text
        assert "that is AI_NEEDS_REVIEW, not AI_REJECTED" in text

    def test_a_version_mismatch_is_refused(self, tmp_path):
        path = tmp_path / "prompt.md"
        path.write_text("<!-- BEGIN JUDGE INSTRUCTIONS v9.9.9 -->\nx\n"
                        "<!-- END JUDGE INSTRUCTIONS v9.9.9 -->\n", encoding="utf-8")
        with pytest.raises(ValueError, match="v9.9.9"):
            judge.load_instructions(str(path))


# -- the judge's output schema ------------------------------------------------

class TestJudgementValidation:

    def test_a_well_formed_judgement_passes(self):
        assert judge.validate_judgement(a_valid_judgement()) == []

    def test_only_the_four_ai_statuses_are_allowed(self):
        assert judge.JUDGE_STATUSES == (
            "AI_VERIFIED", "AI_CORRECTED", "AI_REJECTED", "AI_NEEDS_REVIEW")
        for status in judge.JUDGE_STATUSES:
            assert judge.validate_judgement(
                a_valid_judgement(judge_status=status,
                                  correction=_correction(status))) == []
        for bad in ("VERIFIED", "verified", "PASS", "AI_OK", None):
            assert judge.validate_judgement(a_valid_judgement(judge_status=bad))

    def test_a_correction_requires_every_replacement_field(self):
        for missing in judge.CORRECTION_FIELDS:
            correction = {f: "x" for f in judge.CORRECTION_FIELDS}
            correction[missing] = ""
            problems = judge.validate_judgement(
                a_valid_judgement(judge_status="AI_CORRECTED", correction=correction))
            assert any(missing in p for p in problems), missing

    def test_ai_corrected_without_a_correction_is_refused(self):
        assert judge.validate_judgement(
            a_valid_judgement(judge_status="AI_CORRECTED", correction=None))

    def test_a_non_correction_may_not_smuggle_a_correction(self):
        assert judge.validate_judgement(
            a_valid_judgement(judge_status="AI_VERIFIED",
                              correction={f: "x" for f in judge.CORRECTION_FIELDS}))

    def test_a_missing_field_is_named(self):
        payload = a_valid_judgement()
        del payload["evidence_sufficient"]
        assert "missing field: evidence_sufficient" in judge.validate_judgement(payload)

    def test_null_is_accepted_and_a_manufactured_value_is_not(self):
        assert judge.validate_judgement(a_valid_judgement(locator_correct=None)) == []
        assert judge.validate_judgement(a_valid_judgement(locator_correct="probably"))

    def test_empty_reasoning_is_refused(self):
        assert judge.validate_judgement(a_valid_judgement(judge_reasoning="  "))

    def test_the_prefilled_brace_is_put_back(self):
        body = json.dumps(a_valid_judgement())[1:]
        parsed, error = judge.parse_response(body)
        assert error is None and parsed["judge_status"] == "AI_NEEDS_REVIEW"

    def test_a_malformed_response_is_recorded_not_repaired(self):
        parsed, error = judge.parse_response("this is not JSON at all")
        assert parsed is None and "not valid JSON" in error


def _correction(status):
    if status == "AI_CORRECTED":
        return {f: "x" for f in judge.CORRECTION_FIELDS}
    return None


# -- stale is not rejected ----------------------------------------------------

class TestStaleIsNotIncorrect:

    def test_a_stale_source_may_be_ai_verified(self):
        assert judge.validate_judgement(
            a_valid_judgement(judge_status="AI_VERIFIED", temporal_status="STALE",
                              correction=None)) == []

    def test_the_prompt_says_so_explicitly(self, instructions):
        _, text, _ = instructions
        assert "A source marked STALE is not, by that fact, incorrect" in text
        assert "Distinguish a stale source from an incorrect source" in text

    def test_stale_packets_are_presented_with_their_status(self, packets):
        stale = [p for p in packets
                 if any(e.get("temporal_status") == "STALE" for e in p["evidence"])]
        assert stale
        assert "temporal: STALE" in judge.render_user_message(stale[0])


# -- missing evidence ---------------------------------------------------------

class TestMissingEvidence:

    def test_a_target_absent_from_the_corpus_is_shown_as_absent(self, golden, chunks):
        broken = dict(golden[0], question_id="QX01",
                      expected_chunks=["IN-PATENTS-ACT-1970::no-such-chunk"])
        built = prep.build_all([broken], chunks, "h")[0]
        assert built["evidence"][0]["missing_from_corpus"] is True
        message = judge.render_user_message(built)
        assert "THIS CHUNK IS NOT PRESENT IN THE CORPUS." in message

    def test_the_prompt_routes_insufficient_evidence_to_needs_review(self, instructions):
        _, text, _ = instructions
        assert ("If the supplied evidence is insufficient to decide, return "
                "AI_NEEDS_REVIEW") in text

    def test_needs_review_is_a_first_class_outcome(self):
        assert judge.FIXTURE_RESPONSE["judge_status"] == "AI_NEEDS_REVIEW"
        assert judge.validate_judgement(
            a_valid_judgement(judge_status="AI_NEEDS_REVIEW")) == []


# -- no network without --live ------------------------------------------------

class TestNoNetworkWithoutLive:

    def test_the_dry_run_never_reaches_the_socket(self, monkeypatch, tmp_path):
        def explode(*_args, **_kwargs):
            raise AssertionError("the dry run opened a socket")
        monkeypatch.setattr(judge, "_post", explode)
        monkeypatch.setattr(judge.urllib.request, "urlopen", explode)
        assert judge.main(["--question", "Q027",
                           "--judgements", str(tmp_path / "j.jsonl")]) == 0

    def test_the_dry_run_writes_no_judgement_file(self, tmp_path):
        out = tmp_path / "j.jsonl"
        assert judge.main(["--question", "Q027", "--judgements", str(out)]) == 0
        assert not out.exists()

    def test_the_socket_is_reachable_from_exactly_one_place(self):
        source = inspect.getsource(judge)
        assert source.count("urllib.request.urlopen(") == 1
        assert source.count("_post(") == 2          # the def, and one call site
        assert "if not args.live:" in source

    def test_a_live_run_without_a_key_makes_no_request(self, monkeypatch, tmp_path):
        def explode(*_args, **_kwargs):
            raise AssertionError("a request was made without a key")
        monkeypatch.setattr(judge, "_post", explode)
        monkeypatch.setattr(judge, "load_env_file", lambda *a, **k: None)
        monkeypatch.delenv(judge.API_KEY_ENV, raising=False)
        assert judge.main(["--question", "Q027", "--live",
                           "--judgements", str(tmp_path / "j.jsonl")]) == 2

    def test_the_request_budget_defaults_to_one(self):
        parser_default = [line for line in inspect.getsource(judge).splitlines()
                          if "--max-requests" in line]
        assert parser_default
        assert "default=1" in inspect.getsource(judge)


# -- checkpoint and resume ----------------------------------------------------

class TestResume:

    def test_a_completed_question_is_not_selected_again(self, packets):
        done = [{"question_id": "Q027", "judge_status": "AI_VERIFIED"}]
        remaining = judge.pending(packets, done, question_id=None, start_from=None,
                                  limit=None, only_unjudged=True)
        assert "Q027" not in {p["question_id"] for p in remaining}
        assert len(remaining) == 110

    def test_only_status_all_reselects_everything(self, packets):
        done = [{"question_id": "Q027", "judge_status": "AI_VERIFIED"}]
        assert len(judge.pending(packets, done, question_id=None, start_from=None,
                                 limit=None, only_unjudged=False)) == 111

    def test_start_from_and_limit_bound_the_run(self, packets):
        chosen = judge.pending(packets, [], question_id=None, start_from="Q050",
                               limit=3, only_unjudged=True)
        assert [p["question_id"] for p in chosen] == ["Q050", "Q051", "Q052"]

    def test_selection_is_in_question_order(self, packets):
        chosen = judge.pending(packets, [], question_id=None, start_from=None,
                               limit=None, only_unjudged=True)
        ids = [p["question_id"] for p in chosen]
        assert ids == sorted(ids)

    def test_the_live_loop_refuses_to_overwrite_a_completed_record(self):
        # `pending` filters them, and the loop checks again before requesting.
        source = inspect.getsource(judge.main)
        assert 'if by_qid.get(qid, {}).get("judge_status") in TERMINAL:' in source
        assert "already judged, skipped" in source

    def test_the_checkpoint_is_written_inside_the_loop(self):
        source = inspect.getsource(judge.main)
        loop = source.index("for packet in todo:")
        assert source.index("write_jsonl(args.judgements") > loop


# -- the stored record --------------------------------------------------------

class TestRecord:

    def test_every_part_2_field_is_present(self, packets):
        record = judge.build_record(
            packet(packets, "Q027"), a_valid_judgement(), model="claude-opus-5",
            prompt_version="1.0.0", prompt_sha="abc", diagnostics={})
        assert judge.missing_record_fields(record) == []

    def test_the_prompt_version_and_hash_are_recorded(self, packets, instructions):
        version, _, sha = instructions
        record = judge.build_record(
            packet(packets, "Q027"), a_valid_judgement(), model="m",
            prompt_version=version, prompt_sha=sha, diagnostics={})
        assert record["judge_prompt_version"] == "1.0.0"
        assert record["judge_prompt_sha256"] == sha
        assert record["judge_temperature"] == 0.0
        assert record["judge_provider"] == "anthropic"

    def test_the_corpus_fingerprint_and_golden_hash_are_recorded(self, packets, chunks):
        from scripts.evaluation.prepare_ground_truth import corpus_fingerprint
        one = packet(packets, "Q027")
        record = judge.build_record(one, a_valid_judgement(), model="m",
                                    prompt_version="1.0.0", prompt_sha="s",
                                    diagnostics={})
        assert record["corpus_fingerprint"] == corpus_fingerprint(chunks)
        assert record["golden_dataset_sha256"] == prep.golden_sha256()
        assert record["packet_sha256"] == one["packet_sha256"]

    def test_the_original_label_is_recorded_beside_the_verdict(self, packets):
        one = packet(packets, "Q027")
        record = judge.build_record(one, a_valid_judgement(), model="m",
                                    prompt_version="1.0.0", prompt_sha="s",
                                    diagnostics={})
        assert record["original_document_id"] == "IN-TM-RULES-2017"
        assert record["original_locator"] == "Rule 23"
        assert record["original_chunk_id"] == "IN-TM-RULES-2017::part-i-chapter-ii-rule-23"

    def test_the_verdict_is_never_called_verified(self, packets):
        record = judge.build_record(
            packet(packets, "Q027"), a_valid_judgement(judge_status="AI_VERIFIED"),
            model="m", prompt_version="1.0.0", prompt_sha="s", diagnostics={})
        assert record["judge_status"].startswith("AI_")
        assert "verified" not in record
        assert "verification_status" not in record


# -- credentials --------------------------------------------------------------

class TestCredentials:

    def test_a_key_is_scrubbed_from_any_diagnostic(self):
        assert judge.scrub("failed: key sk-secret-123 rejected", "sk-secret-123") == \
            "failed: key *** rejected"

    def test_the_key_is_read_from_the_environment_and_sent_in_one_header(self):
        source = inspect.getsource(judge)
        assert source.count('"x-api-key"') == 1
        assert 'os.environ.get(API_KEY_ENV)' in source
        for leak in ("print(api_key", "f\"{api_key}", "str(api_key)"):
            assert leak not in source
