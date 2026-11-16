"""Phase 7.2 — the Groq provider, exercised entirely offline.

Every test here injects a fake transport. **No test in this file opens a
socket**, and one of them asserts that by refusing a transport entirely and
checking the provider abstains on the missing key before it would have dialled.

The provider is the first thing in IP-SAKTI that can send corpus text off the
machine, so most of these tests are about what does *not* go into the payload.
"""
from __future__ import annotations

import json
import os

import pytest

from ai.answering.base import GroundedAnswerEngine
from ai.answering.groq import (
    DEFAULT_MAX_COMPLETION_TOKENS,
    DEFAULT_MODEL,
    ENDPOINT,
    MAX_TIMEOUT_SECONDS,
    GroqConfig,
    GroqProvider,
    build_payload,
    load_env_file,
    render_evidence,
)
from ai.answering.models import Evidence, EvidenceSet
from ai.answering.policy import AnswerPolicy
from ai.answering.providers import ANSWER_INSTRUCTIONS, PROMPT_VERSION, build_request

#: Deliberately not shaped like a real Groq credential. A fake that looks
#: genuine trips secret scanners and costs somebody an hour proving it is not.
KEY = "test-key-not-a-credential"


# -- fixtures -----------------------------------------------------------------


def evidence_set() -> EvidenceSet:
    return EvidenceSet((
        Evidence(marker=1, chunk_id="IN-PATENTS-ACT-1970::chapter-ii-section-3",
                 document_id="IN-PATENTS-ACT-1970", document_name="Patents Act, 1970",
                 citation="Patents Act, 1970 — Section 3",
                 text="The following are not inventions within the meaning of this Act.",
                 citable=True, locator_unavailable=False,
                 locator_fields={"section": "3"}, temporal_status="CURRENT",
                 source_evidence_level="B", physical_pages=(11, 12),
                 source_filename="patents.pdf", source_sha256="0badc0ffee",
                 official_source_url="https://ipindia.invalid/patents.pdf",
                 retrieval_rank=1, retrieval_score=0.91),
        Evidence(marker=2, chunk_id="IN-DC-RULES-1945::part-vi-rule-65-c1",
                 document_id="IN-DC-RULES-1945",
                 document_name="Drugs and Cosmetics Rules, 1945",
                 citation="Drugs and Cosmetics Rules, 1945 — Part VI, Rule 65",
                 text="Licences shall be subject to the conditions stated therein.",
                 citable=True, locator_unavailable=False,
                 locator_fields={"part": "VI", "rule": "65"},
                 temporal_status="STALE", source_evidence_level="A",
                 physical_pages=(71, 72), source_filename="dc.pdf",
                 source_sha256="deadbeef", retrieval_rank=2, retrieval_score=0.88),
    ))


def a_request():
    return build_request("What are not inventions under the Patents Act?",
                         evidence_set(), AnswerPolicy())


def transport(status=200, body=None, *, record=None, raises=None):
    """A fake HTTP layer. Records the call instead of making one."""
    payload = body if body is not None else json.dumps(
        {"choices": [{"message": {"content":
                                  "The Act lists what are not inventions [1]."}}]}
    ).encode("utf-8")

    def _call(url, data, headers, timeout):
        if record is not None:
            record.update({"url": url, "body": json.loads(data.decode("utf-8")),
                           "headers": headers, "timeout": timeout,
                           "calls": record.get("calls", 0) + 1})
        if raises is not None:
            raise raises
        return status, payload
    return _call


def provider(**kwargs):
    kwargs.setdefault("api_key", KEY)
    kwargs.setdefault("config", GroqConfig())
    kwargs.setdefault("transport", transport())
    return GroqProvider(**kwargs)


# -- configuration ------------------------------------------------------------


class TestConfiguration:
    def test_default_model_is_the_declared_one(self):
        assert GroqConfig().model == DEFAULT_MODEL == "openai/gpt-oss-120b"

    def test_model_is_configurable_through_the_environment(self, monkeypatch):
        monkeypatch.setenv("GROQ_MODEL", "openai/gpt-oss-20b")
        assert GroqConfig.from_environment(load_dotenv=False).model == "openai/gpt-oss-20b"

    def test_timeout_defaults_to_the_ceiling_and_never_exceeds_it(self, monkeypatch):
        monkeypatch.setenv("GROQ_TIMEOUT_SECONDS", "600")
        config = GroqConfig.from_environment(load_dotenv=False)
        assert config.timeout_seconds == MAX_TIMEOUT_SECONDS == 15.0

    def test_a_lower_timeout_is_honoured(self, monkeypatch):
        monkeypatch.setenv("GROQ_TIMEOUT_SECONDS", "5")
        assert GroqConfig.from_environment(load_dotenv=False).timeout_seconds == 5.0

    def test_an_invalid_timeout_falls_back_rather_than_crashing(self, monkeypatch):
        monkeypatch.setenv("GROQ_TIMEOUT_SECONDS", "soon")
        assert GroqConfig.from_environment(load_dotenv=False).timeout_seconds == 15.0

    def test_config_never_holds_the_api_key(self):
        assert "api_key" not in GroqConfig().to_dict()
        assert KEY not in json.dumps(GroqConfig().to_dict())

    def test_env_file_does_not_override_an_exported_variable(self, monkeypatch, tmp_path):
        env = tmp_path / ".env"
        env.write_text("GROQ_MODEL=from-file\n", encoding="utf-8")
        monkeypatch.setenv("GROQ_MODEL", "from-shell")
        load_env_file(str(env))
        assert os.environ["GROQ_MODEL"] == "from-shell"

    def test_env_file_supplies_a_variable_that_is_unset(self, monkeypatch, tmp_path):
        env = tmp_path / ".env"
        env.write_text("# comment\nGROQ_MODEL=from-file\n\n", encoding="utf-8")
        monkeypatch.delenv("GROQ_MODEL", raising=False)
        load_env_file(str(env))
        assert os.environ["GROQ_MODEL"] == "from-file"

    def test_a_missing_env_file_is_not_an_error(self, tmp_path):
        load_env_file(str(tmp_path / "absent.env"))


# -- request construction -----------------------------------------------------


class TestRequestConstruction:
    def test_endpoint_and_method_are_the_openai_compatible_ones(self):
        record = {}
        provider(transport=transport(record=record)).generate(a_request())
        assert record["url"] == ENDPOINT
        assert record["url"].startswith("https://")

    def test_payload_carries_only_the_declared_fields(self):
        payload = build_payload(a_request(), GroqConfig())
        assert set(payload) == {"model", "messages", "temperature", "top_p",
                                "max_completion_tokens", "stream"}

    def test_generation_is_deterministic_and_bounded_and_unstreamed(self):
        payload = build_payload(a_request(), GroqConfig())
        assert payload["temperature"] == 0.0
        assert payload["stream"] is False
        assert payload["max_completion_tokens"] == DEFAULT_MAX_COMPLETION_TOKENS

    def test_the_system_prompt_is_the_existing_versioned_contract(self):
        system = build_payload(a_request(), GroqConfig())["messages"][0]["content"]
        for instruction in ANSWER_INSTRUCTIONS:
            assert instruction in system

    def test_prompt_version_is_preserved_not_forked(self):
        draft = provider().generate(a_request())
        assert draft.diagnostics["prompt_version"] == PROMPT_VERSION

    def test_model_name_is_recorded_on_the_draft(self):
        draft = provider().generate(a_request())
        assert draft.diagnostics["model"] == "openai/gpt-oss-120b"

    def test_timeout_is_passed_to_the_transport(self):
        record = {}
        GroqProvider(api_key=KEY, config=GroqConfig(timeout_seconds=9.0),
                     transport=transport(record=record)).generate(a_request())
        assert record["timeout"] == 9.0
        assert record["timeout"] <= MAX_TIMEOUT_SECONDS

    def test_exactly_one_request_is_made(self):
        record = {}
        provider(transport=transport(record=record)).generate(a_request())
        assert record["calls"] == 1

    def test_the_key_travels_only_in_the_authorization_header(self):
        record = {}
        provider(transport=transport(record=record)).generate(a_request())
        assert record["headers"]["Authorization"] == f"Bearer {KEY}"
        assert KEY not in json.dumps(record["body"])


# -- what must not leave the machine ------------------------------------------


class TestPayloadPrivacy:
    def _sent(self):
        record = {}
        provider(transport=transport(record=record)).generate(a_request())
        return json.dumps(record["body"])

    def test_no_chunk_id_leaves_the_machine(self):
        sent = self._sent()
        for item in evidence_set():
            assert item.chunk_id not in sent

    def test_no_document_id_leaves_the_machine(self):
        sent = self._sent()
        assert "IN-PATENTS-ACT-1970" not in sent
        assert "IN-DC-RULES-1945" not in sent

    def test_no_source_filename_or_hash_leaves_the_machine(self):
        sent = self._sent()
        for forbidden in ("patents.pdf", "dc.pdf", "0badc0ffee", "deadbeef"):
            assert forbidden not in sent

    def test_no_url_leaves_the_machine(self):
        assert "ipindia.invalid" not in self._sent()

    def test_no_page_numbers_leave_the_machine(self):
        body = json.loads(self._sent())
        user = body["messages"][1]["content"]
        assert "physical_pages" not in user
        assert "p. 11" not in user and "pp. 71" not in user

    def test_no_retrieval_or_reranker_score_leaves_the_machine(self):
        sent = self._sent()
        for forbidden in ("0.91", "0.88", "retrieval_score", "rerank_score",
                          "dense_score", "lexical_score"):
            assert forbidden not in sent

    def test_no_corpus_path_leaves_the_machine(self):
        sent = self._sent()
        for forbidden in ("data/processed", "data/evaluation", "golden_questions",
                          "expected_chunks", ".jsonl"):
            assert forbidden not in sent

    def test_only_query_instructions_and_passage_text_are_sent(self):
        body = json.loads(self._sent())
        assert len(body["messages"]) == 2
        user = body["messages"][1]["content"]
        assert "What are not inventions under the Patents Act?" in user
        for item in evidence_set():
            assert item.text in user

    def test_markers_are_preserved_so_the_model_can_cite(self):
        user = render_evidence(a_request())
        assert "[1]" in user and "[2]" in user

    def test_temporal_status_and_evidence_level_are_supplied(self):
        """Instructions 6 and 7 are impossible to follow without them."""
        user = render_evidence(a_request())
        assert "CURRENT" in user and "STALE" in user
        assert "source evidence level A" in user


# -- failure paths ------------------------------------------------------------


class TestFailureHandling:
    def test_missing_api_key_abstains_without_dialling(self, monkeypatch):
        monkeypatch.delenv("GROQ_API_KEY", raising=False)
        called = {"n": 0}

        def explode(*args, **kwargs):
            called["n"] += 1
            raise AssertionError("the provider dialled without a key")

        draft = GroqProvider(api_key=None, config=GroqConfig(),
                             transport=explode).generate(a_request())
        assert draft.text == ""
        assert draft.diagnostics["error"] == "missing_api_key"
        assert called["n"] == 0

    @pytest.mark.parametrize("status,code", [(401, "unauthorized"),
                                             (403, "forbidden"),
                                             (429, "rate_limited"),
                                             (500, "server_error"),
                                             (503, "server_error"),
                                             (418, "http_error")])
    def test_http_errors_abstain_with_a_stable_code(self, status, code):
        draft = provider(transport=transport(status=status)).generate(a_request())
        assert draft.text == ""
        assert draft.diagnostics["error"] == code

    def test_a_timeout_abstains(self):
        draft = provider(transport=transport(raises=TimeoutError())).generate(a_request())
        assert draft.text == ""
        assert draft.diagnostics["error"] == "unexpected_error"

    def test_malformed_json_abstains(self):
        draft = provider(transport=transport(body=b"<html>gateway</html>")).generate(
            a_request())
        assert draft.diagnostics["error"] == "malformed_response"

    def test_a_response_missing_choices_abstains(self):
        draft = provider(transport=transport(body=b'{"ok": true}')).generate(a_request())
        assert draft.diagnostics["error"] == "malformed_response"

    def test_an_empty_completion_abstains(self):
        body = json.dumps({"choices": [{"message": {"content": ""}}]}).encode()
        draft = provider(transport=transport(body=body)).generate(a_request())
        assert draft.diagnostics["error"] == "empty_completion"

    def test_no_retry_ever_happens(self):
        record = {}
        provider(transport=transport(status=429, record=record)).generate(a_request())
        assert record["calls"] == 1

    def test_failures_never_raise_into_the_engine(self):
        draft = provider(transport=transport(status=500)).generate(a_request())
        assert isinstance(draft.text, str)
        assert draft.diagnostics["retries"] == 0

    def test_no_diagnostic_ever_contains_the_key(self):
        for status in (401, 429, 500):
            draft = provider(transport=transport(status=status)).generate(a_request())
            assert KEY not in json.dumps(dict(draft.diagnostics))


# -- integration with the existing engine -------------------------------------


class TestEngineIntegration:
    def test_a_valid_response_flows_through_to_a_validated_answer(self):
        engine = GroundedAnswerEngine(provider=provider())
        answer = engine.answer("What are not inventions under the Patents Act?",
                               evidence_set())
        assert not answer.insufficient_evidence
        assert answer.grounded
        assert [c.chunk_id for c in answer.citations] == [
            "IN-PATENTS-ACT-1970::chapter-ii-section-3"]
        assert answer.provider == "groq"

    def test_provider_failure_becomes_the_existing_abstention(self):
        engine = GroundedAnswerEngine(provider=provider(
            transport=transport(status=429)))
        answer = engine.answer("What are not inventions under the Patents Act?",
                               evidence_set())
        assert answer.insufficient_evidence
        assert "provider_abstained" in {f.code
                                        for f in answer.insufficient_evidence_reasons}
        assert answer.citations == ()

    def test_a_hallucinated_provision_is_still_refused(self):
        """The citation layer stays authoritative over the model."""
        body = json.dumps({"choices": [{"message": {"content":
            "Section 45 of the Act bars this [1]."}}]}).encode()
        engine = GroundedAnswerEngine(provider=provider(transport=transport(body=body)))
        answer = engine.answer("What are not inventions under the Patents Act?",
                               evidence_set())
        assert answer.insufficient_evidence
        assert "unsupported_locator_in_prose" in {f.code
                                                  for f in answer.validation_failures}

    def test_a_hallucinated_marker_is_still_refused(self):
        body = json.dumps({"choices": [{"message": {"content":
            "The Act excludes several categories [7]."}}]}).encode()
        engine = GroundedAnswerEngine(provider=provider(transport=transport(body=body)))
        answer = engine.answer("What are not inventions under the Patents Act?",
                               evidence_set())
        assert answer.insufficient_evidence
        assert "unknown_marker" in {f.code for f in answer.validation_failures}

    def test_a_model_written_url_is_still_refused(self):
        body = json.dumps({"choices": [{"message": {"content":
            "See https://ipindia.gov.in for the text [1]."}}]}).encode()
        engine = GroundedAnswerEngine(provider=provider(transport=transport(body=body)))
        answer = engine.answer("What are not inventions under the Patents Act?",
                               evidence_set())
        assert answer.insufficient_evidence
        assert "fabricated_url" in {f.code for f in answer.validation_failures}

    def test_the_engine_records_the_model_and_never_the_key(self):
        engine = GroundedAnswerEngine(provider=provider())
        answer = engine.answer("What are not inventions under the Patents Act?",
                               evidence_set())
        serialised = json.dumps(answer.to_dict())
        assert "openai/gpt-oss-120b" in serialised
        assert KEY not in serialised
        assert "GROQ_API_KEY" not in serialised

    def test_provider_config_reports_network_use_honestly(self):
        config = provider().config()
        assert config["requires_network"] is True
        assert config["implemented"] is True
        assert config["retries"] == 0
        assert config["timeout_seconds"] <= 15.0
        assert "api_key" not in config
