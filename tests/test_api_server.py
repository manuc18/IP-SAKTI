"""Tests for the HTTP API layer — transport only.

None of these touch the engine. The pipeline is replaced by a stub returning a
canned result, which is the point: this file tests that the API *carries* what
the engine decided, and nothing here can tell whether an answer is right. That
is measured elsewhere, and by nothing that runs in 30 milliseconds.

No network, no model, no corpus read, no live Groq request.
"""
from __future__ import annotations

import dataclasses
import enum
import json

import pytest

from app.config import Config
from app.api_server import create_app
from app.schemas import serialize_result, to_jsonable

DISCLAIMER = "Not legal advice. Verify against the official text."


def evidence_item(marker: int, chunk_id: str, *, cited: bool = False,
                  temporal_status: str = "CURRENT", level: str = "B"):
    return {
        "marker": marker,
        "chunk_id": chunk_id,
        "document_id": "IN-TM-RULES-2017",
        "document_name": "Trade Marks Rules, 2017",
        "citation": "Trade Marks Rules, 2017 — Rule 23",
        "citable": True,
        "locator_unavailable": False,
        "temporal_status": temporal_status,
        "source_evidence_level": level,
        "physical_pages": [59],
        "retrieval_rank": marker,
        "text": f"Evidence passage {marker}.",
        "locator_text": "Rule 23",
        "heading": "Form and signing of application",
        "unit_type": "rule",
        "domain": "trade_marks",
        "source_status": "consolidated",
        "official_source_url": "https://example.invalid/tm-rules.pdf",
        "retrieval_score": 0.91,
        "cited": cited,
    }


def answered_result(question="How must a trade mark application be signed?"):
    return {
        "question": question,
        "status": "ANSWERED",
        "service_failure": None,
        "answer": {
            "answer": "An application must be signed by the applicant or agent [1].",
            "citations": [{
                "marker": 1,
                "chunk_id": "chunk-a",
                "document_id": "IN-TM-RULES-2017",
                "locator": "Rule 23",
                "pages": [59],
                "citation_text": "Trade Marks Rules, 2017 — Rule 23",
                "source_evidence_level": "B",
                "temporal_status": "CURRENT",
                "locator_unavailable": False,
                "source_document": {"document_id": "IN-TM-RULES-2017",
                                    "filename": "tm-rules-2017.pdf",
                                    "sha256": "abc123",
                                    "official_source_url": None,
                                    "source_version_id": None,
                                    "consolidated_as_of": None},
            }],
            "confidence": "MODERATE",
            "grounded": True,
            "insufficient_evidence": False,
            "insufficient_evidence_reasons": [],
            "conflict_detected": False,
            "conflict_chunks": [],
            "cautions": ["The cited source is marked stale. The text may have been "
                         "amended since the consolidated version held in the corpus; "
                         "verify against the current official text."],
            "validation_failures": [],
            "unsupported_claims": [],
            "uncited_evidence": ["chunk-b"],
            "evidence_fingerprint": "0f1e2d3c4b5a6978",
            "provider": "groq",
            "policy": {"policy_version": "0.3.0"},
            "diagnostics": {},
            "version": "0.1.0",
        },
        "rendered": "An application must be signed by the applicant or agent [1].",
        "disclaimer": DISCLAIMER,
        "evidence": [evidence_item(1, "chunk-a", cited=True),
                     evidence_item(2, "chunk-b")],
        "timings_ms": {"retrieval": 120.0, "reranking": 8.0, "answering": 1800.0,
                       "total": 1928.0},
        "pipeline": {"candidates": 88, "evidence_count": 2,
                     "retrievers": "dense(BGE-M3) + BM25+identifier-boost",
                     "reranker": "features", "provider": "groq · openai/gpt-oss-120b"},
    }


def abstained_result(question="What is the best pizza topping in Naples?"):
    result = answered_result(question)
    result["status"] = "ABSTAINED"
    result["answer"].update({
        "answer": "",
        "citations": [],
        "confidence": "NONE",
        "grounded": False,
        "insufficient_evidence": True,
        "insufficient_evidence_reasons": [
            {"code": "no_relevant_evidence", "detail":
             "no supplied passage addresses the question", "marker": None,
             "chunk_id": None}],
        "cautions": [],
    })
    result["evidence"] = [evidence_item(1, "chunk-a"), evidence_item(2, "chunk-b")]
    return result


def unavailable_result(question="How must a trade mark application be signed?"):
    result = abstained_result(question)
    result["status"] = "UNAVAILABLE"
    result["service_failure"] = {
        "code": "rate_limited",
        "message": "The language provider is rate-limited right now.",
        "detail": "Groq rate limit reached (429)",
    }
    result["answer"]["insufficient_evidence_reasons"] = [
        {"code": "provider_abstained", "detail": "the provider returned no answer",
         "marker": None, "chunk_id": None}]
    return result


class StubPipeline:
    """Stands in for `app.pipeline.Pipeline`. Same two methods, no engine."""

    provider_name = "groq · openai/gpt-oss-120b"

    def __init__(self, result=None, error=None):
        self._result = result
        self._error = error
        self.questions: list[str] = []

    def answer(self, question):
        self.questions.append(question)
        if self._error is not None:
            raise self._error
        return self._result

    def health(self):
        return {"status": "ok", "corpus": {"chunks": 3073, "documents": 22}}


def client_for(pipeline):
    app = create_app(Config(cors_origins=("*",)), pipeline)
    app.config["TESTING"] = True
    return app.test_client()


def post(client, payload, *, raw=None, content_type="application/json"):
    data = raw if raw is not None else json.dumps(payload)
    return client.post("/api/v1/query", data=data, content_type=content_type)


# -- health and root ----------------------------------------------------------


def test_health_returns_200_without_touching_the_engine():
    pipeline = StubPipeline(answered_result())
    response = client_for(pipeline).get("/health")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ok"
    assert body["service"] == "ip-sakti-api"
    assert body["engine"]["corpus"]["chunks"] == 3073
    assert pipeline.questions == []  # no retrieval, no answer generation


def test_root_names_the_service_and_its_endpoints():
    body = client_for(StubPipeline(answered_result())).get("/").get_json()
    assert body["service"] == "IP-SAKTI API"
    assert body["endpoints"]["query"] == "POST /api/v1/query"


# -- ANSWERED -----------------------------------------------------------------


def test_answered_carries_answer_citations_and_sources():
    pipeline = StubPipeline(answered_result())
    response = post(client_for(pipeline), {"question": "  How is it signed?  "})
    assert response.status_code == 200
    body = response.get_json()

    assert body["success"] is True
    assert body["status"] == "ANSWERED"
    assert body["answer"]["text"].endswith("[1].")
    assert body["answer"]["confidence"] == "MODERATE"
    assert body["answer"]["grounded"] is True
    assert body["disclaimer"] == DISCLAIMER
    assert pipeline.questions == ["How is it signed?"]  # whitespace stripped

    (citation,) = body["citations"]
    assert citation["marker"] == "[1]"
    assert citation["chunk_id"] == "chunk-a"
    assert citation["document"] == "Trade Marks Rules, 2017"
    assert citation["locator"] == "Rule 23"
    assert citation["page"] == 59
    assert citation["provenance_level"] == "B"
    assert citation["temporal_status"] == "CURRENT"

    assert [s["chunk_id"] for s in body["sources"]] == ["chunk-a", "chunk-b"]
    assert [s["cited"] for s in body["sources"]] == [True, False]
    assert body["sources"][0]["text"] == "Evidence passage 1."


def test_citation_markers_resolve_to_a_returned_source():
    """The click target the frontend needs: every [n] maps to a source card."""
    body = post(client_for(StubPipeline(answered_result())),
                {"question": "q"}).get_json()
    source_ids = {s["chunk_id"] for s in body["sources"]}
    assert all(c["chunk_id"] in source_ids for c in body["citations"])


def test_cautions_are_labelled_but_never_reworded():
    body = post(client_for(StubPipeline(answered_result())),
                {"question": "q"}).get_json()
    (caution,) = body["cautions"]
    assert caution["type"] == "STALE_SOURCE"
    assert caution["message"] == answered_result()["answer"]["cautions"][0]


def test_request_id_is_generated_and_echoed():
    body = post(client_for(StubPipeline(answered_result())),
                {"question": "q"}).get_json()
    assert body["request_id"]

    client_id = "client-abc-123"
    body = post(client_for(StubPipeline(answered_result())),
                {"question": "q", "request_id": client_id}).get_json()
    assert body["request_id"] == client_id


# -- ABSTAINED ----------------------------------------------------------------


def test_abstention_is_a_200_with_no_answer():
    response = post(client_for(StubPipeline(abstained_result())),
                    {"question": "What is the best pizza topping in Naples?"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["success"] is True
    assert body["status"] == "ABSTAINED"
    assert body["answer"] is None
    assert body["citations"] == []
    assert body["sources"] == []
    assert body["reason"]["code"] == "NO_RELEVANT_EVIDENCE"
    assert body["reason"]["message"]
    # The evidence existed even though none is published; the count says so.
    assert body["meta"]["evidence_retrieved"] == 2
    assert "error" not in body


# -- UNAVAILABLE --------------------------------------------------------------


def test_provider_failure_is_503_and_is_not_an_abstention():
    response = post(client_for(StubPipeline(unavailable_result())),
                    {"question": "How must it be signed?"})
    assert response.status_code == 503
    body = response.get_json()
    assert body["success"] is False
    assert body["status"] == "UNAVAILABLE"
    assert body["answer"] is None
    assert body["error"]["code"] == "PROVIDER_UNAVAILABLE"
    assert body["error"]["provider_error"]["code"] == "rate_limited"
    assert "reason" not in body  # a service failure makes no claim about evidence


# -- invalid requests ---------------------------------------------------------


@pytest.mark.parametrize("payload,code", [
    ({}, "MISSING_QUESTION"),
    ({"question": ""}, "EMPTY_QUESTION"),
    ({"question": "   \n\t "}, "EMPTY_QUESTION"),
    ({"question": 42}, "INVALID_QUESTION_TYPE"),
    ({"question": None}, "INVALID_QUESTION_TYPE"),
    ({"question": "x" * 501}, "QUESTION_TOO_LONG"),
    ({"question": "q", "options": "groq"}, "INVALID_OPTIONS"),
    ({"question": "q", "options": {"provider": "openai"}}, "UNSUPPORTED_OPTION"),
    ([], "INVALID_BODY"),
])
def test_invalid_requests_are_400_with_a_code(payload, code):
    pipeline = StubPipeline(answered_result())
    response = post(client_for(pipeline), payload)
    assert response.status_code == 400
    body = response.get_json()
    assert body["success"] is False
    assert body["status"] == "INVALID_REQUEST"
    assert body["error"]["code"] == code
    assert pipeline.questions == []  # rejected before the engine was called


def test_malformed_json_is_400():
    response = post(client_for(StubPipeline(answered_result())), None,
                    raw="{not json")
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "INVALID_JSON"


def test_non_json_content_type_is_400():
    response = post(client_for(StubPipeline(answered_result())), None,
                    raw="question=hello", content_type="text/plain")
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "INVALID_CONTENT_TYPE"


def test_provider_option_matching_the_running_provider_is_accepted():
    response = post(client_for(StubPipeline(answered_result())),
                    {"question": "q", "options": {"provider": "default"}})
    assert response.status_code == 200


def test_unknown_route_is_404_json():
    response = client_for(StubPipeline(answered_result())).get("/api/v1/nope")
    assert response.status_code == 404
    assert response.get_json()["error"]["code"] == "NOT_FOUND"


def test_wrong_method_is_405_json():
    response = client_for(StubPipeline(answered_result())).get("/api/v1/query")
    assert response.status_code == 405
    assert response.get_json()["error"]["code"] == "METHOD_NOT_ALLOWED"


def test_oversized_body_is_413_json():
    response = post(client_for(StubPipeline(answered_result())), None,
                    raw=json.dumps({"question": "x" * 70_000}))
    assert response.status_code == 413
    assert response.get_json()["error"]["code"] == "REQUEST_TOO_LARGE"


# -- unexpected failures ------------------------------------------------------


def test_unexpected_exception_is_500_and_leaks_nothing():
    boom = RuntimeError("index /secret/path/to/corpus.jsonl exploded")
    response = post(client_for(StubPipeline(error=boom)), {"question": "q"})
    assert response.status_code == 500
    body = response.get_json()
    assert body["success"] is False
    assert body["error"]["code"] == "INTERNAL_ERROR"
    serialised = json.dumps(body)
    assert "Traceback" not in serialised
    assert "/secret/path" not in serialised
    assert "RuntimeError" not in serialised


def test_pipeline_value_error_is_400():
    response = post(client_for(StubPipeline(error=ValueError("Question is too long."))),
                    {"question": "q"})
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "INVALID_QUESTION"


# -- serialisation ------------------------------------------------------------


class _Level(enum.Enum):
    B = "B"


@dataclasses.dataclass(frozen=True)
class _Nested:
    document_id: str
    pages: tuple = (1, 2)


def test_to_jsonable_handles_engine_shapes():
    assert to_jsonable((1, 2)) == [1, 2]
    assert to_jsonable(_Level.B) == "B"
    assert to_jsonable(_Nested("doc")) == {"document_id": "doc", "pages": [1, 2]}
    assert json.dumps(to_jsonable({"a": {_Level.B}, "b": _Nested("doc")}))


def test_serialized_result_is_json_serialisable_with_nested_engine_objects():
    result = answered_result()
    result["answer"]["citations"][0]["pages"] = (59, 60)
    result["answer"]["citations"][0]["source_document"] = _Nested("IN-TM-RULES-2017")
    result["evidence"][0]["physical_pages"] = (59,)
    body, status = serialize_result(result, "rid")
    assert status == 200
    assert json.dumps(body)  # no TypeError
    assert body["citations"][0]["pages"] == [59, 60]


# -- boundary -----------------------------------------------------------------


def test_api_layer_does_not_reimplement_the_engine():
    """The route calls the pipeline. It does not contain a second one."""
    forbidden = ("ai.retrieval", "ai.reranking", "ai.answering",
                 "CandidateGenerator", "GroundedAnswerEngine", "as_evidence_set",
                 "validate_citations", "check_grounding", "AnswerPolicy")
    for path in ("app/api_server.py", "app/schemas.py"):
        with open(path, encoding="utf-8") as fh:
            source = fh.read()
        for name in forbidden:
            assert name not in source, f"{path} references {name}"


# -- cache metadata (additive; nothing a client renders depends on it) --------


def test_a_live_result_is_labelled_as_coming_from_the_pipeline():
    body = post(client_for(StubPipeline(answered_result())),
                {"question": "How is it signed?"}).get_json()
    assert body["meta"]["cache"] == {"hit": False, "source": None,
                                     "cached_at": None}
    assert body["meta"]["response_source"] == "live_pipeline"


def test_a_cached_result_serialises_identically_apart_from_the_label():
    live = answered_result()
    cached = answered_result()
    cached["cache"] = {"hit": True, "stored": True, "source": "seed",
                       "cached_at": "2026-09-07T10:00:00Z",
                       "cached_question": cached["question"],
                       "original_timings_ms": cached["timings_ms"]}
    cached["timings_ms"] = {"retrieval": 0.0, "reranking": 0.0,
                            "answering": 0.0, "total": 0.4}

    live_body = post(client_for(StubPipeline(live)), {"question": "Q?"}).get_json()
    cached_body = post(client_for(StubPipeline(cached)),
                       {"question": "Q?"}).get_json()

    assert cached_body["meta"]["response_source"] == "seed"
    assert cached_body["meta"]["cache"]["hit"] is True
    # Everything the user sees is byte-identical.
    for field in ("status", "answer", "citations", "sources", "cautions",
                  "disclaimer"):
        assert cached_body[field] == live_body[field]
    assert cached_body["meta"]["grounded"] == live_body["meta"]["grounded"]
