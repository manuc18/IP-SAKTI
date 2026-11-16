"""Tests for the validated-answer cache.

No network, no model, no corpus read. The pipeline is a stub that counts how
many times it was asked to generate, because the property under test is exactly
that: a cached question does not reach the provider.

The safety half of this file matters more than the speed half. A cache that
serves a fast wrong answer is worse than a rate limit, so most of what follows
checks what does *not* get stored: abstentions, provider failures, ungrounded
drafts, answers whose citations did not resolve, and entries belonging to a
different corpus or prompt version.
"""
from __future__ import annotations

import copy
import json

import pytest

from app.cache import (
    CACHE_FORMAT,
    ValidatedAnswerCache,
    is_publishable,
    normalize_question,
)

QUESTION = "How must an application for registration of a trade mark be signed?"


def published_result(question=QUESTION):
    """A result the pipeline would have published: ANSWERED, grounded, cited."""
    return {
        "question": question,
        "status": "ANSWERED",
        "service_failure": None,
        "answer": {
            "answer": "An application must be made in Form TM-A and signed by "
                      "the applicant or his agent [1].",
            "citations": [{"marker": 1, "chunk_id": "chunk-a",
                           "document_id": "IN-TM-RULES-2017",
                           "locator": "Rule 23", "pages": [59],
                           "citation_text": "Trade Marks Rules, 2017 — Rule 23",
                           "source_evidence_level": "B",
                           "temporal_status": "CURRENT"}],
            "confidence": "MODERATE",
            "grounded": True,
            "insufficient_evidence": False,
            "insufficient_evidence_reasons": [],
            "validation_failures": [],
            "unsupported_claims": [],
            "cautions": [],
            "evidence_fingerprint": "fp-1",
            "version": "0.1.0",
        },
        "rendered": "...",
        "disclaimer": "Not legal advice.",
        "evidence": [{"marker": 1, "chunk_id": "chunk-a", "cited": True,
                      "text": "Rule 23 ..."}],
        "timings_ms": {"retrieval": 40.0, "reranking": 5.0,
                       "answering": 3100.0, "total": 3145.0},
        "pipeline": {"candidates": 50, "evidence_count": 5,
                     "provider": "groq · openai/gpt-oss-120b"},
    }


def cache(tmp_path, namespace="ns-1"):
    return ValidatedAnswerCache(namespace=namespace,
                                path=str(tmp_path / "answers.json"))


# -- the admission gate -------------------------------------------------------


def test_a_published_result_is_publishable():
    assert is_publishable(published_result()) is True


@pytest.mark.parametrize("mutate,reason", [
    (lambda r: r.update(status="ABSTAINED"), "abstention"),
    (lambda r: r.update(status="UNAVAILABLE"), "provider unavailable"),
    (lambda r: r.update(service_failure={"code": "rate_limited"}), "rate limit"),
    (lambda r: r["answer"].update(grounded=False), "grounding failed"),
    (lambda r: r["answer"].update(
        validation_failures=[{"code": "unresolved_citation"}]),
     "citation validation failed"),
    (lambda r: r["answer"].update(unsupported_claims=["a claim"]),
     "unsupported claim"),
    (lambda r: r["answer"].update(insufficient_evidence=True),
     "insufficient evidence"),
    (lambda r: r["answer"].update(citations=[]), "no citation"),
    (lambda r: r["answer"].update(answer="   "), "no answer text"),
    (lambda r: r.pop("answer"), "no answer object"),
])
def test_unpublished_results_are_never_publishable(mutate, reason):
    result = published_result()
    mutate(result)
    assert is_publishable(result) is False, reason


def test_non_mappings_are_not_publishable():
    assert is_publishable(None) is False
    assert is_publishable("ANSWERED") is False
    assert is_publishable([published_result()]) is False


# -- keys ---------------------------------------------------------------------


def test_normalization_folds_case_punctuation_and_the_trademark_spelling():
    assert normalize_question("  Trade Marks  Act, 1999?  ") == \
           normalize_question("trademarks act 1999")
    assert normalize_question("How is a trademark signed?") == \
           normalize_question("How is a trade-mark signed")
    # Singular and plural stay apart; the tokenizer does not stem.
    assert normalize_question("trade mark") != normalize_question("trade marks")


def test_normalization_keeps_different_questions_apart():
    assert normalize_question("What is the term of copyright?") != \
           normalize_question("What is the term of a patent?")
    # Word order is content, not noise.
    assert normalize_question("patent of a trade mark") != \
           normalize_question("trade mark of a patent")


def test_a_question_of_pure_punctuation_still_gets_a_key(tmp_path):
    store = cache(tmp_path)
    assert store.key("???") != store.key("!!!")


def test_the_namespace_is_inside_the_key(tmp_path):
    assert cache(tmp_path, "ns-1").key(QUESTION) != \
           cache(tmp_path, "ns-2").key(QUESTION)


# -- store and serve ----------------------------------------------------------


def test_a_published_answer_is_stored_and_served(tmp_path):
    store = cache(tmp_path)
    assert store.put(QUESTION, published_result()) is True

    entry = store.get(QUESTION)
    assert entry is not None
    assert entry["result"]["answer"]["answer"].startswith("An application")
    assert entry["result"]["answer"]["citations"][0]["locator"] == "Rule 23"
    assert store.hits == 1


def test_a_normalized_equivalent_question_hits_the_same_entry(tmp_path):
    store = cache(tmp_path)
    store.put(QUESTION, published_result())
    assert store.get("how must an application for registration of a "
                     "trademark be signed") is not None


def test_a_different_question_misses(tmp_path):
    store = cache(tmp_path)
    store.put(QUESTION, published_result())
    assert store.get("What is the term of a patent in India?") is None
    assert store.misses == 1


def test_the_served_payload_is_a_copy(tmp_path):
    store = cache(tmp_path)
    store.put(QUESTION, published_result())
    store.get(QUESTION)["result"]["answer"]["answer"] = "tampered"
    assert store.get(QUESTION)["result"]["answer"]["answer"].startswith(
        "An application")


@pytest.mark.parametrize("mutate", [
    lambda r: r.update(status="ABSTAINED"),
    lambda r: r.update(status="UNAVAILABLE",
                       service_failure={"code": "rate_limited"}),
    lambda r: r["answer"].update(grounded=False),
    lambda r: r["answer"].update(validation_failures=[{"code": "x"}]),
])
def test_unpublished_results_are_not_stored(tmp_path, mutate):
    store = cache(tmp_path)
    result = published_result()
    mutate(result)
    assert store.put(QUESTION, result) is False
    assert store.get(QUESTION) is None
    assert store.questions() == []


# -- persistence and invalidation ---------------------------------------------


def test_entries_survive_a_restart(tmp_path):
    cache(tmp_path).put(QUESTION, published_result())
    assert cache(tmp_path).get(QUESTION) is not None


def test_a_new_namespace_ignores_old_entries(tmp_path):
    cache(tmp_path, "corpus-v1").put(QUESTION, published_result())
    reopened = cache(tmp_path, "corpus-v2")
    assert reopened.get(QUESTION) is None
    assert reopened.questions() == []
    assert reopened.warnings


def test_entries_from_another_namespace_are_kept_but_never_served(tmp_path):
    """Rehearsing offline must not erase a cache that cost provider requests."""
    cache(tmp_path, "groq").put(QUESTION, published_result())

    offline = cache(tmp_path, "extractive")
    assert offline.get(QUESTION) is None            # not served
    offline.put("What is the term of a patent?", published_result())

    back = cache(tmp_path, "groq")
    assert back.get(QUESTION) is not None           # still there
    assert back.questions() == [QUESTION]


def test_a_malformed_file_is_ignored_not_fatal(tmp_path):
    path = tmp_path / "answers.json"
    path.write_text("{ this is not json", encoding="utf-8")
    store = ValidatedAnswerCache(namespace="ns-1", path=str(path))
    assert store.get(QUESTION) is None
    assert store.warnings
    # And it recovers: a later write replaces the bad file.
    assert store.put(QUESTION, published_result()) is True
    assert json.loads(path.read_text())["format"] == CACHE_FORMAT


def test_an_unknown_file_format_is_ignored(tmp_path):
    path = tmp_path / "answers.json"
    path.write_text(json.dumps({"format": 99, "entries": {}}), encoding="utf-8")
    assert ValidatedAnswerCache(namespace="ns-1", path=str(path)).warnings


def test_a_hand_edited_entry_cannot_smuggle_an_answer_back(tmp_path):
    """The gate runs again on read, so editing the file is not a way in."""
    path = tmp_path / "answers.json"
    store = ValidatedAnswerCache(namespace="ns-1", path=str(path))
    store.put(QUESTION, published_result())

    document = json.loads(path.read_text())
    entry = next(iter(document["entries"].values()))
    entry["result"]["answer"]["grounded"] = False
    entry["result"]["answer"]["answer"] = "A fabricated answer with no support."
    path.write_text(json.dumps(document), encoding="utf-8")

    # Rejected on load...
    assert ValidatedAnswerCache(namespace="ns-1", path=str(path)).get(QUESTION) is None
    # ...and rejected on read, if it were placed directly into a live cache.
    live = ValidatedAnswerCache(namespace="ns-1", path=None)
    live._entries[live.key(QUESTION)] = entry
    assert live.get(QUESTION) is None
    assert live.rejections == 1


def test_an_unwritable_path_keeps_working_in_memory(tmp_path):
    blocked = tmp_path / "file-not-a-dir"
    blocked.write_text("x", encoding="utf-8")
    store = ValidatedAnswerCache(namespace="ns-1",
                                 path=str(blocked / "answers.json"))
    assert store.put(QUESTION, published_result()) is True
    assert store.get(QUESTION) is not None
    assert store.warnings


def test_the_cache_is_bounded(tmp_path):
    store = ValidatedAnswerCache(namespace="ns-1",
                                 path=str(tmp_path / "answers.json"),
                                 max_entries=2)
    for word in ("alpha", "beta", "gamma"):
        store.put(f"What is a {word} trade mark?", published_result())
    assert len(store.questions()) == 2


# -- the pipeline boundary ----------------------------------------------------


class CountingPipeline:
    """The real `Pipeline.answer` body, reduced to what the cache wraps."""

    def __init__(self, store, result):
        self.cache = store
        self._result = result
        self.generations = 0

    def answer(self, question):
        from app.pipeline import _from_cache

        question = (question or "").strip()
        if not question:
            raise ValueError("Ask a question first.")
        cached = self.cache.get(question) if self.cache else None
        if cached is not None:
            return _from_cache(cached, question, 0.4)
        self.generations += 1                     # the provider call
        result = copy.deepcopy(self._result)
        result["question"] = question
        stored = bool(self.cache and self.cache.put(question, result))
        result["cache"] = {"hit": False, "stored": stored}
        return result


def test_a_cache_hit_does_not_reach_the_provider(tmp_path):
    pipeline = CountingPipeline(cache(tmp_path), published_result())

    first = pipeline.answer(QUESTION)
    assert first["cache"] == {"hit": False, "stored": True}
    assert pipeline.generations == 1

    second = pipeline.answer(QUESTION)
    assert second["cache"]["hit"] is True
    assert pipeline.generations == 1              # no second generation
    assert second["answer"] == first["answer"]
    assert second["evidence"] == first["evidence"]


def test_a_cache_miss_runs_the_pipeline(tmp_path):
    pipeline = CountingPipeline(cache(tmp_path), published_result())
    pipeline.answer(QUESTION)
    pipeline.answer("Who may apply for a geographical indication?")
    assert pipeline.generations == 2


def test_a_rate_limited_result_is_not_cached_and_is_not_retried(tmp_path):
    unavailable = published_result()
    unavailable["status"] = "UNAVAILABLE"
    unavailable["service_failure"] = {"code": "rate_limited",
                                      "message": "rate limited"}
    pipeline = CountingPipeline(cache(tmp_path), unavailable)

    first = pipeline.answer(QUESTION)
    assert first["status"] == "UNAVAILABLE"
    assert first["cache"] == {"hit": False, "stored": False}
    assert pipeline.generations == 1              # one request, not two

    second = pipeline.answer(QUESTION)            # a fresh user-driven attempt
    assert second["status"] == "UNAVAILABLE"
    assert pipeline.generations == 2              # still one request each


def test_a_hit_reports_its_own_timings_and_the_original(tmp_path):
    pipeline = CountingPipeline(cache(tmp_path), published_result())
    pipeline.answer(QUESTION)
    hit = pipeline.answer(QUESTION)

    assert hit["timings_ms"]["answering"] == 0.0
    assert hit["timings_ms"]["total"] < 1.0
    assert hit["cache"]["original_timings_ms"]["answering"] == 3100.0
    assert hit["cache"]["cached_question"] == QUESTION


def test_a_hit_echoes_the_question_that_was_asked(tmp_path):
    pipeline = CountingPipeline(cache(tmp_path), published_result())
    pipeline.answer(QUESTION)
    asked = "How must an application for registration of a TRADEMARK be signed?"
    hit = pipeline.answer(asked)
    assert hit["cache"]["hit"] is True
    assert hit["question"] == asked
    assert hit["cache"]["cached_question"] == QUESTION


def test_disabling_the_cache_leaves_the_pipeline_unchanged(tmp_path):
    pipeline = CountingPipeline(None, published_result())
    pipeline.answer(QUESTION)
    pipeline.answer(QUESTION)
    assert pipeline.generations == 2


# -- surviving a provider outage ---------------------------------------------
#
# These build the real pipeline once and run the real engine, citation
# validator and grounding check. Only the HTTP transport is replaced, so what
# is under test is the property the demo depends on: a question already
# answered stays answerable while the provider is down, and a question that is
# not stays honestly unavailable.


@pytest.fixture(scope="module")
def corpus_pipeline():
    import os

    os.environ["IPSAKTI_DENSE"] = "0"
    from app.pipeline import Pipeline

    return Pipeline(use_dense=False, provider="extractive", cache=False,
                    log=lambda message: None)


def failing_engine(pipeline, status, body=b"{}"):
    """The real engine and the real Groq provider, over a dead transport."""
    from ai.answering import GroundedAnswerEngine
    from ai.answering.groq import (GroqConfig, GroqError, GroqProvider,
                                   _transport_detail)

    calls = []

    def transport(url, payload, headers, timeout):
        """Stands in for `_urllib_transport`, and honours its contract: a
        network failure is raised as a `GroqError`, an HTTP status is returned.
        Raising a bare OSError here would test the engine's last-resort handler
        instead of the transport failure the demo actually sees."""
        calls.append(url)
        if status == "transport":
            raise GroqError("transport_error", _transport_detail(
                OSError(61, "Connection refused"), timeout))
        return 429, body

    provider = GroqProvider(config=GroqConfig(), transport=transport,
                            api_key="sk-not-a-real-key")
    return GroundedAnswerEngine(provider=provider, policy=pipeline.policy), calls


@pytest.mark.parametrize("failure,expected", [
    ("transport", "transport_error"),
    ("429", "rate_limited"),
])
def test_a_cached_question_answers_while_the_provider_is_down(
        corpus_pipeline, tmp_path, failure, expected):
    pipeline = corpus_pipeline
    healthy = pipeline.engine
    pipeline.cache = ValidatedAnswerCache(namespace="outage",
                                          path=str(tmp_path / f"{failure}.json"))
    try:
        warm = pipeline.answer(QUESTION)
        assert warm["status"] == "ANSWERED"
        assert warm["cache"]["stored"] is True

        pipeline.engine, calls = failing_engine(pipeline, failure)

        # The cached question still answers, and the provider is never reached.
        served = pipeline.answer(QUESTION)
        assert served["status"] == "ANSWERED"
        assert served["cache"]["hit"] is True
        assert served["answer"]["citations"] == warm["answer"]["citations"]
        assert served["answer"]["grounded"] is True
        assert served["evidence"] == warm["evidence"]
        assert calls == []

        # An uncached question stays honestly unavailable: no fabricated
        # answer, no abstention standing in for a service failure, no retry.
        cold = pipeline.answer("What is the term of a patent in India?")
        assert cold["status"] == "UNAVAILABLE"
        assert cold["service_failure"]["code"] == expected
        assert cold["cache"] == {"hit": False, "stored": False}
        assert len(calls) == 1
        assert pipeline.cache.questions() == [QUESTION]
    finally:
        pipeline.engine = healthy
        pipeline.cache = None


def test_the_api_serves_a_cached_answer_during_an_outage(corpus_pipeline, tmp_path):
    """End to end over HTTP: the contract is identical, the provider is dead."""
    from app.api_server import create_app
    from app.config import Config

    pipeline = corpus_pipeline
    healthy = pipeline.engine
    pipeline.cache = ValidatedAnswerCache(namespace="api-outage",
                                          path=str(tmp_path / "api.json"))
    try:
        pipeline.answer(QUESTION)
        pipeline.engine, calls = failing_engine(pipeline, "transport")

        client = create_app(Config(cors_origins=("*",)), pipeline).test_client()
        response = client.post("/api/v1/query", json={"question": QUESTION})
        body = response.get_json()

        assert response.status_code == 200
        assert body["status"] == "ANSWERED"
        assert body["answer"]["text"]
        assert body["answer"]["grounded"] is True
        assert body["citations"] and body["sources"]
        assert body["meta"]["response_source"] == "runtime"
        assert calls == []

        # And an uncached question is still a 503 that says why.
        miss = client.post("/api/v1/query", json={"question": "What is a patent?"})
        assert miss.status_code == 503
        assert miss.get_json()["error"]["provider_error"]["code"] == "transport_error"
    finally:
        pipeline.engine = healthy
        pipeline.cache = None


def test_transport_detail_names_the_cause_not_just_the_class():
    """"URLError" was the recorded detail for a timeout, a DNS failure and a
    refused connection alike. Each has a different fix."""
    from ai.answering.groq import _transport_detail

    assert "timeout" in _transport_detail(TimeoutError("timed out"), 15.0)
    assert "ConnectionRefusedError" in _transport_detail(
        OSError(61, "Connection refused"), 15.0) or "Connection refused" in \
        _transport_detail(OSError(61, "Connection refused"), 15.0)


def test_the_completion_ceiling_is_clamped_in_both_directions(monkeypatch):
    from ai.answering.groq import (DEFAULT_MAX_COMPLETION_TOKENS,
                                   MIN_MAX_COMPLETION_TOKENS, GroqConfig)

    def budget(value):
        monkeypatch.setenv("GROQ_MAX_COMPLETION_TOKENS", value)
        return GroqConfig.from_environment(load_dotenv=False).max_completion_tokens

    assert budget("512") == 512
    assert budget("4000") == DEFAULT_MAX_COMPLETION_TOKENS   # cannot be raised
    assert budget("1") == MIN_MAX_COMPLETION_TOKENS          # cannot starve
    assert budget("not-a-number") == DEFAULT_MAX_COMPLETION_TOKENS
    monkeypatch.delenv("GROQ_MAX_COMPLETION_TOKENS")
    assert GroqConfig.from_environment(load_dotenv=False).max_completion_tokens == \
        DEFAULT_MAX_COMPLETION_TOKENS


# -- the seed layer -----------------------------------------------------------
#
# The seed ships inside the deployment artifact so the demo questions survive
# Render's ephemeral filesystem. It is read-only and gets no more trust than a
# file the server wrote itself.


def seeded(tmp_path, entries, namespace="ns-1"):
    """Write a seed file directly, the way the generator would."""
    keyer = ValidatedAnswerCache(namespace=namespace, path=None, seed_path=None)
    path = tmp_path / "seed.json"
    path.write_text(json.dumps({
        "format": CACHE_FORMAT, "namespace": namespace,
        "entries": {keyer.key(q): {"namespace": namespace, "question": q,
                                   "normalized": normalize_question(q),
                                   "cached_at": "2026-09-07T00:00:00Z",
                                   "result": r}
                    for q, r in entries.items()},
    }), encoding="utf-8")
    return str(path)


def test_the_seed_loads_at_startup_with_no_runtime_cache(tmp_path):
    path = seeded(tmp_path, {QUESTION: published_result()})
    store = ValidatedAnswerCache(namespace="ns-1",
                                 path=str(tmp_path / "absent.json"),
                                 seed_path=path)
    stats = store.stats()
    assert stats["seed_entries"] == 1
    assert stats["runtime_entries"] == 0

    entry = store.get(QUESTION)
    assert entry is not None
    assert entry["source"] == "seed"
    assert entry["result"]["answer"]["citations"][0]["locator"] == "Rule 23"
    assert store.seed_hits == 1


def test_a_seeded_question_survives_a_wiped_runtime_cache(tmp_path):
    """What a Render redeploy does: the writable file is gone, the seed is not."""
    path = seeded(tmp_path, {QUESTION: published_result()})
    runtime = tmp_path / "runtime.json"

    first = ValidatedAnswerCache(namespace="ns-1", path=str(runtime),
                                 seed_path=path)
    first.put("Who may apply for a geographical indication?", published_result())
    assert runtime.exists()

    runtime.unlink()                                   # the redeploy

    after = ValidatedAnswerCache(namespace="ns-1", path=str(runtime),
                                 seed_path=path)
    assert after.get(QUESTION) is not None             # seeded: still served
    assert after.get("Who may apply for a geographical indication?") is None


def test_the_runtime_layer_wins_over_the_seed(tmp_path):
    stale = published_result()
    stale["answer"]["answer"] = "The shipped answer [1]."
    fresh = published_result()
    fresh["answer"]["answer"] = "The answer this deployment produced [1]."

    store = ValidatedAnswerCache(namespace="ns-1", path=str(tmp_path / "r.json"),
                                 seed_path=seeded(tmp_path, {QUESTION: stale}))
    assert store.get(QUESTION)["result"]["answer"]["answer"] == \
        "The shipped answer [1]."

    store.put(QUESTION, fresh)
    served = store.get(QUESTION)
    assert served["source"] == "runtime"
    assert served["result"]["answer"]["answer"] == \
        "The answer this deployment produced [1]."


def test_an_invalid_seed_entry_is_rejected(tmp_path):
    ungrounded = published_result()
    ungrounded["answer"]["grounded"] = False
    abstained = published_result()
    abstained["status"] = "ABSTAINED"

    store = ValidatedAnswerCache(
        namespace="ns-1", path=None,
        seed_path=seeded(tmp_path, {QUESTION: ungrounded,
                                    "Who may apply for a GI?": abstained}))
    assert store.stats()["seed_entries"] == 0
    assert store.get(QUESTION) is None
    assert any("failed validation" in w for w in store.warnings)


def test_a_seed_for_another_configuration_is_reported_not_silently_empty(tmp_path):
    path = seeded(tmp_path, {QUESTION: published_result()}, namespace="groq-dense")
    store = ValidatedAnswerCache(namespace="groq-lexical", path=None,
                                 seed_path=path)
    assert store.stats()["seed_entries"] == 0
    assert store.get(QUESTION) is None
    assert any("different corpus, prompt or provider" in w
               for w in store.warnings)


def test_a_malformed_seed_does_not_stop_the_server(tmp_path):
    path = tmp_path / "seed.json"
    path.write_text("not json at all", encoding="utf-8")
    store = ValidatedAnswerCache(namespace="ns-1", path=None, seed_path=str(path))
    assert store.stats()["seed_entries"] == 0
    assert store.warnings
    assert store.put(QUESTION, published_result()) is True   # still usable


def test_the_seed_is_never_written_to(tmp_path):
    path = seeded(tmp_path, {QUESTION: published_result()})
    before = open(path, encoding="utf-8").read()
    store = ValidatedAnswerCache(namespace="ns-1", path=str(tmp_path / "r.json"),
                                 seed_path=path)
    store.put("What is the term of a patent?", published_result())
    store.get(QUESTION)
    assert open(path, encoding="utf-8").read() == before


def test_a_seeded_question_answers_during_a_provider_outage(corpus_pipeline,
                                                            tmp_path):
    """The property the whole seed exists for, through the real engine."""
    pipeline = corpus_pipeline
    healthy = pipeline.engine
    pipeline.cache = ValidatedAnswerCache(namespace="seed-outage",
                                          path=str(tmp_path / "r.json"),
                                          seed_path=None)
    try:
        warm = pipeline.answer(QUESTION)
        assert warm["status"] == "ANSWERED"

        # Ship that answer as a seed, throw the runtime cache away, kill Groq.
        seed_path = seeded(tmp_path, {QUESTION: {k: v for k, v in warm.items()
                                                 if k != "cache"}},
                           namespace="seed-outage")
        pipeline.cache = ValidatedAnswerCache(namespace="seed-outage",
                                              path=str(tmp_path / "gone.json"),
                                              seed_path=seed_path)
        pipeline.engine, calls = failing_engine(pipeline, "transport")

        served = pipeline.answer(QUESTION)
        assert served["status"] == "ANSWERED"
        assert served["cache"]["source"] == "seed"
        assert served["answer"]["citations"] == warm["answer"]["citations"]
        assert served["answer"]["grounded"] is True
        assert served["evidence"] == warm["evidence"]
        assert calls == []

        # An unseeded question still reaches the real pipeline and fails honestly.
        cold = pipeline.answer("What is the term of a patent in India?")
        assert cold["status"] == "UNAVAILABLE"
        assert cold["service_failure"]["code"] == "transport_error"
        assert len(calls) == 1
    finally:
        pipeline.engine = healthy
        pipeline.cache = None


def test_the_api_reports_the_seed_as_the_response_source(corpus_pipeline, tmp_path):
    from app.api_server import create_app
    from app.config import Config

    pipeline = corpus_pipeline
    pipeline.cache = ValidatedAnswerCache(namespace="seed-api", path=None,
                                          seed_path=None)
    try:
        warm = pipeline.answer(QUESTION)
        pipeline.cache = ValidatedAnswerCache(
            namespace="seed-api", path=None,
            seed_path=seeded(tmp_path, {QUESTION: {k: v for k, v in warm.items()
                                                   if k != "cache"}},
                             namespace="seed-api"))
        client = create_app(Config(cors_origins=("*",)), pipeline).test_client()
        body = client.post("/api/v1/query", json={"question": QUESTION}).get_json()

        assert body["status"] == "ANSWERED"
        assert body["meta"]["response_source"] == "seed"
        assert body["meta"]["cache"] == {"hit": True, "source": "seed",
                                         "cached_at": "2026-09-07T00:00:00Z"}
        assert body["citations"] and body["sources"]
        assert body["answer"]["grounded"] is True
    finally:
        pipeline.cache = None


# -- the page must not spend tokens on mount ----------------------------------


def _page_source():
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "apps", "web", "src", "app", "page.jsx"),
              encoding="utf-8") as handle:
        return handle.read()


def test_page_load_does_not_ask_a_question():
    """A source guard, not a rendering test — apps/web has no JS runner and
    tonight is not the night to add one. It still catches the regression that
    matters: an auto-ask on mount spent a completion on every page load and
    every refresh, before anyone had clicked anything."""
    source = _page_source()
    # The mount effect is the one that runs the health check; the UI rewrite
    # moved it below the component's other effects.
    mount = source[source.index("useEffect(() => {\n    fetchHealth();"):]
    mount = mount[:mount.index("}, []);")]
    assert "fetchHealth()" in mount
    assert "handleSendMessage" not in mount and "handleAsk" not in mount


def test_sample_chips_still_submit_on_an_explicit_click():
    source = _page_source()
    chips = source[source.index("{activeQuestions.slice(0, 4).map("):]
    chips = chips[:chips.index("</button>")]
    assert "onClick" in chips
    assert "setInputText(s)" in chips and "handleSendMessage(s)" in chips
