"""The demo pipeline: built once, answered many times.

`QUESTION → CANDIDATES → RERANK → EVIDENCE → ANSWER → VALIDATED CITATIONS`,
which is the path `scripts/evaluation/phase73_five_questions.py` established.
Two deliberate differences from that script, both stated rather than buried:

* **The dense arm is enabled.** Phase 7.3 disabled it because that phase
  forbade loading a local model. The prototype has no such constraint, and
  lexical-only retrieval would demonstrate the system at Recall@1 0.451 instead
  of the routed 0.766 the project actually achieves.
* **The reranker is the deterministic feature reranker**, not the cross-encoder.
  `bge-reranker-v2-m3` scores higher (0.874 vs 0.784 Recall@1 on the unverified
  benchmark) but its latency was never measured — `docs/RERANKING_BENCHMARK.md`
  lists that as an unrun experiment — and an unmeasured per-query cost is not
  something to discover during a demo. Set `IPSAKTI_RERANKER=cross_encoder` to
  switch; nothing else changes.

The provider is chosen once, at startup, from the environment. That is a
configuration choice, not a fallback chain: a request that fails is an
abstention, and nothing here retries, regenerates or switches models.

One question answered twice does not run twice. `app/cache.py` holds results
this pipeline already published — `ANSWERED`, citations resolved, grounding
passed — and `answer()` consults it first. A hit skips retrieval, reranking and
the provider entirely, which is the only reason a demo can ask the same six
questions repeatedly on a free-tier token budget. A miss runs the code below
unchanged. The cache stores what the pipeline produced; it cannot produce
anything, and it never stores an abstention or a provider failure.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.answering import (  # noqa: E402
    AnswerPolicy,
    ExtractiveMockProvider,
    GroundedAnswerEngine,
    SYSTEM_DISCLAIMER,
    as_evidence_set,
    render_answer,
)
from ai.answering.excerpt import excerpt_config, excerpt_for_evidence  # noqa: E402
from ai.text_integrity import assess as assess_integrity, integrity_config  # noqa: E402
from ai.answering.groq import GroqConfig, GroqProvider, load_env_file  # noqa: E402
from ai.retrieval.base import load_chunks  # noqa: E402
from ai.retrieval.bm25 import BM25Retriever  # noqa: E402
from ai.retrieval.candidates import CandidateGenerator  # noqa: E402
from ai.retrieval.identifiers import BM25IdentifierBoostRetriever  # noqa: E402
from ai.reranking.features import FeatureReranker  # noqa: E402
from app.clarify import clarify_config, post_abstention, pre_retrieval  # noqa: E402
from app.intent import classify as classify_intent, intent_config  # noqa: E402
from app.spelling import CorpusSpelling, spelling_config  # noqa: E402
from app.cache import (  # noqa: E402
    DEFAULT_CACHE_PATH,
    DEFAULT_SEED_PATH,
    ValidatedAnswerCache,
)

APP_VERSION = "0.1.0"

DENSE_DEPTH = 50
LEXICAL_DEPTH = 50
#: Passages handed to the answer provider. Five is what Phase 7.3 used and what
#: the 800-token completion budget was measured against.
TOP_K = 5

MAX_QUESTION_CHARS = 500

#: Provider failures that say nothing about the evidence. `ai/answering/groq.py`
#: turns every one of these into an abstention, which is the right behaviour —
#: no answer is published — but it is the wrong *explanation* to show a user.
#: "I could not support an answer from this evidence" and "the API refused the
#: request" are different claims, and a demo that renders a rate limit as a
#: safety decision is taking credit it has not earned. Reported apart; handled
#: identically. `empty_completion` is deliberately absent: a model that saw the
#: evidence and wrote nothing has made a judgement, not failed to be reached.
SERVICE_ERROR_CODES = frozenset({
    "rate_limited", "transport_error", "unauthorized", "forbidden",
    "server_error", "http_error", "missing_api_key", "malformed_response",
    "payload_audit_failed", "unexpected_error",
})

SERVICE_ERROR_MESSAGES = {
    "rate_limited": "The language provider is rate-limited right now (free-tier "
                    "per-minute cap). No answer was generated. This is a service "
                    "limit, not a finding about the evidence — wait a moment and ask again.",
    "missing_api_key": "No language-provider credential is configured, so no answer "
                       "was generated. Restart with --provider extractive to run "
                       "fully offline.",
    "unauthorized": "The language provider rejected the credential. No answer was generated.",
    "forbidden": "The language provider refused the request. No answer was generated.",
    "transport_error": "The language provider could not be reached. No answer was "
                       "generated. Check the network, or restart with "
                       "--provider extractive to run fully offline.",
}
DEFAULT_SERVICE_MESSAGE = ("The language provider did not complete the request, so "
                           "no answer was generated. Nothing unverified is shown.")


def _service_failure(answer) -> dict[str, Any] | None:
    """Distinguish 'the provider could not be reached' from 'the evidence
    did not support an answer'. Both publish nothing; only one is our claim."""
    if not answer.insufficient_evidence:
        return None
    diagnostics = answer.diagnostics.get("provider_diagnostics") or {}
    code = diagnostics.get("error")
    if code not in SERVICE_ERROR_CODES:
        return None
    return {"code": code,
            "message": SERVICE_ERROR_MESSAGES.get(code, DEFAULT_SERVICE_MESSAGE),
            "detail": str(diagnostics.get("detail") or "")[:400]}


def _flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


class Pipeline:
    """Everything loaded once, guarded by one lock.

    The lock serialises `answer()`. A demo has one user, the dense retriever
    keeps a mutable query cache, and torch is not obliged to enjoy concurrent
    calls — so the cheapest correct thing is to answer one question at a time.
    """

    def __init__(self, *, use_dense: bool = True, reranker: str = "features",
                 provider: str = "auto", cache: bool = True,
                 cache_path: str | None = DEFAULT_CACHE_PATH,
                 seed_cache_path: str | None = DEFAULT_SEED_PATH, log=print):
        self._lock = threading.Lock()
        self.log = log
        self.warnings: list[str] = []
        started = time.perf_counter()

        load_env_file()

        self.chunks = load_chunks()
        log(f"  corpus              {len(self.chunks)} chunks")

        bm25 = BM25Retriever(self.chunks)
        self.lexical = BM25IdentifierBoostRetriever(bm25)
        log(f"  lexical index       ready ({self.lexical.name})")

        # The corpus is its own dictionary; see `app/spelling.py`. Built from
        # the same chunk text BM25 indexes, so a word is "known" here exactly
        # when it is reachable there.
        self.speller = CorpusSpelling(chunk.text for chunk in self.chunks)
        log(f"  spelling            {len(self.speller.targets)} correction "
            f"targets from {len(self.speller.vocabulary)} corpus words")

        self.dense = None
        self.dense_status = "disabled"
        if use_dense:
            self.dense, self.dense_status = self._build_dense()

        self.generator = CandidateGenerator(
            dense=self.dense or _NoDense(), lexical=self.lexical,
            mode="expanded", order="interleaved",
            dense_depth=DENSE_DEPTH, lexical_depth=LEXICAL_DEPTH,
            aliases=self.lexical.aliases)

        self.reranker = self._build_reranker(reranker, bm25)
        log(f"  reranker            {self.reranker.name}")

        self.policy = AnswerPolicy()
        self.provider_name, self.provider = self._build_provider(provider)
        self.engine = GroundedAnswerEngine(provider=self.provider, policy=self.policy)
        log(f"  answer provider     {self.provider_name}")

        self.cache = self._build_cache(cache, cache_path, seed_cache_path)
        if self.cache is None:
            log("  answer cache        disabled")
        else:
            stats = self.cache.stats()
            log(f"  seed answers        {stats['seed_entries']} shipped with this "
                f"build [{stats['namespace']}]")
            log(f"  answer cache        {stats['entries']} validated answer(s) "
                f"({stats['runtime_entries']} from this deployment)")
            self.warnings.extend(self.cache.warnings)

        self.startup_seconds = round(time.perf_counter() - started, 2)

    # -- construction ---------------------------------------------------------

    @staticmethod
    def _prefer_cached_model() -> None:
        """Never let a hub lookup stand between the demo and a cached model.

        `SentenceTransformer` contacts huggingface.co on load to check for a
        newer revision. On a good network that costs a second; on a conference
        network it can hang, and the model is already on disk. So when the
        snapshot is present, the hub is pinned offline. When it is not, the
        variable is left alone and the normal download path still works.
        `HF_HUB_OFFLINE` set by the caller always wins.
        """
        cached = os.path.expanduser("~/.cache/huggingface/hub/models--BAAI--bge-m3")
        if os.path.isdir(cached):
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

    def _build_dense(self):
        """Load BGE-M3, or degrade to lexical-only and say so.

        A missing model or a cold embedding cache must not take the demo down
        with it. Retrieval quality drops and `/api/health` reports it; the
        answer path is unaffected, because nothing downstream of retrieval
        knows which engines produced the candidates.
        """
        try:
            self._prefer_cached_model()
            from ai.retrieval.dense import DenseRetriever

            t = time.perf_counter()
            dense = DenseRetriever(self.chunks)
            self.log(f"  dense index         {dense.embeddings.shape[0]} vectors "
                     f"({time.perf_counter() - t:.1f}s)")
            t = time.perf_counter()
            dense.embed_query("warm up the encoder so the first user does not")
            self.log(f"  BGE-M3 warm         {time.perf_counter() - t:.1f}s")
            return dense, "ready"
        except Exception as exc:  # noqa: BLE001 - degrade, never crash
            message = (f"dense retrieval unavailable ({type(exc).__name__}: {exc}); "
                       f"running lexical-only, which lowers retrieval quality")
            self.warnings.append(message)
            self.log(f"  dense index         UNAVAILABLE — {exc}")
            return None, "unavailable"

    def _build_reranker(self, choice: str, bm25):
        if choice == "cross_encoder":
            try:
                from ai.reranking.cross_encoder import CrossEncoderReranker

                return CrossEncoderReranker()
            except Exception as exc:  # noqa: BLE001
                self.warnings.append(
                    f"cross-encoder reranker unavailable ({exc}); using features")
        return FeatureReranker(idf=bm25.idf, aliases=self.lexical.aliases)

    def _build_provider(self, choice: str):
        """`auto` means Groq when a key exists, the offline provider otherwise.

        The extractive provider is not a degraded Groq. It is the project's
        default provider: deterministic, offline, quotes the corpus verbatim,
        and subject to exactly the same citation and grounding gates. It is what
        makes the demo survive a dead venue network.
        """
        if choice == "extractive":
            return "extractive (offline, deterministic)", ExtractiveMockProvider(
                max_sentences=self.policy.max_citations)
        if not os.environ.get("GROQ_API_KEY"):
            if choice == "groq":
                self.warnings.append(
                    "GROQ_API_KEY is not set; using the offline extractive provider")
            return "extractive (offline, deterministic)", ExtractiveMockProvider(
                max_sentences=self.policy.max_citations)
        config = GroqConfig.from_environment()
        return f"groq · {config.model}", GroqProvider(config=config)

    def _cache_namespace(self) -> str:
        """Everything that would change the answer, folded into one digest.

        A cached answer is only reusable while the run that produced it would
        still be reproduced. So the corpus fingerprint (chunk ids paired with
        digests of their text — the same identity `ai/retrieval/dense.py` keys
        its embeddings by), the retrieval and reranking configuration, and the
        answering versions from `engine.config()` all go in. Re-chunk, change
        the prompt, switch provider, or start with `--no-dense`, and the
        namespace changes, so yesterday's entries stop matching instead of
        being served against evidence they were never generated from.

        Only the digest is kept. `engine.config()` names an endpoint and reports
        whether a credential is present; hashing means no part of it is written
        to disk, and there is no room for a secret to arrive here by accident.
        """
        from ai.retrieval.dense import corpus_fingerprint

        identity = {
            "app_version": APP_VERSION,
            "corpus": corpus_fingerprint(self.chunks),
            "dense": self.dense_status,
            "reranker": self.reranker.name,
            "lexical": self.lexical.name,
            "depths": [DENSE_DEPTH, LEXICAL_DEPTH, TOP_K],
            "provider": self.provider_name,
            "engine": self.engine.config(),
            # A corrected query retrieves different evidence, so an entry
            # published before correction existed is not reusable after it.
            "spelling": spelling_config(self.speller),
        }
        material = json.dumps(identity, sort_keys=True, default=str)
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    def _build_cache(self, enabled: bool, path: str | None, seed_path: str | None):
        """Never let the cache stop the pipeline from starting."""
        if not enabled:
            return None
        try:
            return ValidatedAnswerCache(namespace=self._cache_namespace(),
                                        path=path, seed_path=seed_path)
        except Exception as exc:  # noqa: BLE001 - degrade, never crash
            self.warnings.append(
                f"answer cache unavailable ({type(exc).__name__}: {exc}); "
                f"every question will run the full pipeline")
            return None

    # -- the request path -----------------------------------------------------

    def answer(self, question: str, *,
               allow_clarification: bool = True) -> dict[str, Any]:
        """One question, at most one answer-generation call, no retries.

        *At most*, because a question this pipeline has already answered is
        served from `app/cache.py` without contacting the provider at all. The
        input guards run first either way: an invalid question is rejected
        before anything is looked up, so the cache cannot become a second route
        past validation.

        `allow_clarification=False` is how a caller says "this is the user's
        second attempt — do not ask again". It is the only loop guard Phase E
        has, and it is the caller's to set, because only the caller knows
        whether it already showed one. See `app/clarify.py`.
        """
        question = (question or "").strip()
        if not question:
            raise ValueError("Ask a question first.")
        if len(question) > MAX_QUESTION_CHARS:
            raise ValueError(
                f"Question is too long ({len(question)} characters; "
                f"the limit is {MAX_QUESTION_CHARS}).")

        # Before anything else, including the clarifier. "hi" is not an
        # under-specified legal question and must not be answered with one.
        # `classify` returns None for everything that is not a verbatim entry
        # in its phrase table, so a legal question cannot reach this branch.
        social = classify_intent(question, self.lexical.aliases)
        if social is not None:
            return _social(question, social)

        # Spelling is repaired next, and before the clarifier, because the
        # clarifier's whole test is whether the query names a regime — and
        # "How do I register a tradmark?" names one, in the only sense the user
        # cares about. Running the clarifier first made a typo indistinguishable
        # from a vague question and answered it with "which area of IP law did
        # you mean?", which is the worst of both: the system knew the answer and
        # asked the question anyway.
        #
        # What is searched for and what is asked are kept apart from here on.
        # `search_query` reaches BM25, BGE-M3, the reranker and excerpt
        # selection, because they match strings and a misspelling is a string
        # they cannot match. `question` reaches the answer generator, because
        # the user asked that. Both appear in the response.
        correction = self.speller.correct(question)
        search_query = correction.corrected

        if allow_clarification:
            # Only for a question that anchors on no regime, no instrument and
            # no provision. Nothing else short-circuits the pipeline, and this
            # branch never touches the abstention logic.
            upfront = pre_retrieval(search_query, self.lexical.aliases)
            if upfront is not None:
                return _needs_clarification(question, upfront, correction)

        with self._lock:
            t0 = time.perf_counter()
            cached = self.cache.get(question) if self.cache else None
            if cached is not None:
                return _from_cache(cached, question,
                                   round((time.perf_counter() - t0) * 1000, 1))

            t0 = time.perf_counter()   # the lookup is not retrieval time
            pool = self.generator.generate(search_query)
            t1 = time.perf_counter()
            reranked = self.reranker.rerank(search_query, pool.candidates, TOP_K)
            evidence = as_evidence_set([r.to_retrieval_result() for r in reranked])
            t2 = time.perf_counter()
            answer = self.engine.answer(question, evidence)
            t3 = time.perf_counter()

        cited = {c.chunk_id for c in answer.citations}
        service_failure = _service_failure(answer)
        if service_failure:
            status = "UNAVAILABLE"
            self._log_service_failure(question, service_failure)
        elif answer.insufficient_evidence:
            status = "ABSTAINED"
        else:
            status = "ANSWERED"
        result = {
            "question": question,
            "status": status,
            "spelling": correction.to_dict(),
            "service_failure": service_failure,
            "answer": answer.to_dict(),
            "rendered": render_answer(answer, show_confidence=True),
            "disclaimer": SYSTEM_DISCLAIMER,
            "evidence": [
                dict(item.to_dict(),
                     text=item.text,
                     locator_text=item.locator_text,
                     heading=item.heading,
                     unit_type=item.unit_type,
                     domain=item.domain,
                     source_status=item.source_status,
                     official_source_url=item.official_source_url,
                     retrieval_score=item.retrieval_score,
                     # Phase F: the short verbatim span a source card shows
                     # first. `text` stays beside it for "View full passage" —
                     # the excerpt narrows what is shown, never what is kept.
                     excerpt=excerpt_for_evidence(item, search_query,
                                                  answer_text=answer.answer),
                     # Whether PDF extraction damaged this passage. Reported so
                     # a card can say "the source text did not extract cleanly"
                     # instead of presenting debris as a verbatim quotation.
                     # Nothing here rewrites the text: see ai/text_integrity.py.
                     text_integrity=assess_integrity(item.text).to_dict(),
                     cited=item.chunk_id in cited)
                for item in evidence],
            "timings_ms": {
                "retrieval": round((t1 - t0) * 1000, 1),
                "reranking": round((t2 - t1) * 1000, 1),
                "answering": round((t3 - t2) * 1000, 1),
                "total": round((t3 - t0) * 1000, 1),
            },
            "pipeline": {
                "candidates": len(pool.candidates),
                "evidence_count": len(evidence),
                "retrievers": "dense(BGE-M3) + BM25+identifier-boost"
                              if self.dense else "BM25+identifier-boost (lexical only)",
                "reranker": self.reranker.name,
                "provider": self.provider_name,
                "excerpt": excerpt_config(),
                "clarification": clarify_config(),
                "intent": intent_config(),
                "spelling": spelling_config(self.speller),
                "text_integrity": integrity_config(),
            },
        }
        # Phase E, second and last place a clarification can appear: beside an
        # abstention the pipeline has already decided on. `status` is not read
        # back or changed by this — an ABSTAINED response stays ABSTAINED, with
        # its reasons intact, and gains a suggestion for the next attempt.
        if allow_clarification:
            suggestion = post_abstention(search_query, result, self.lexical.aliases)
            if suggestion is not None:
                result["clarification"] = suggestion.to_dict()

        # Offered to the cache, which stores it only if it was published:
        # ANSWERED, grounded, with citations that resolved. An abstention and a
        # rate limit both reach this line and both leave unstored.
        stored = bool(self.cache and self.cache.put(question, result))
        result["cache"] = {"hit": False, "stored": stored}
        return result

    def _log_service_failure(self, question: str, failure: dict[str, Any]) -> None:
        """One structured line per provider failure, for a deployed log.

        The screenshots from Render said `TRANSPORT_ERROR` and nothing more,
        which is not enough to act on: the same code covers a timeout, a DNS
        failure and a refused connection, and whether the request was even a
        live one is invisible from the outside. So the code, the unwrapped
        detail, the configured timeout and the fact that this was a live call
        rather than a cache hit go out together — a failure only reachable on
        a cache miss, which is why `cache_hit` is always false here and is
        still stated rather than assumed by a reader.

        No credential can appear: `detail` is scrubbed by the provider before
        it reaches this object, and nothing here reads the environment.
        """
        settings = getattr(self.provider, "settings", None)
        self.log(
            "provider_failure "
            f"code={failure.get('code')} "
            f"timeout_s={getattr(settings, 'timeout_seconds', 'n/a')} "
            f"max_completion_tokens={getattr(settings, 'max_completion_tokens', 'n/a')} "
            f"provider={self.provider_name!r} "
            f"cache_hit=false "
            f"question={question[:80]!r} "
            f"detail={failure.get('detail') or '(none)'!r}")

    # -- health ---------------------------------------------------------------

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "app_version": APP_VERSION,
            "startup_seconds": self.startup_seconds,
            "corpus": {"chunks": len(self.chunks),
                       "documents": len({c.document_id for c in self.chunks})},
            "retrieval": {"dense": self.dense_status,
                          "lexical": "ready",
                          "reranker": self.reranker.name},
            "answering": {"provider": self.provider_name,
                          "versions": self.engine.config()},
            "cache": self.cache.stats() if self.cache else {"enabled": False},
            "warnings": self.warnings,
        }


def _social(question: str, social: Any) -> dict[str, Any]:
    """The response to a greeting: courteous, and visibly not a legal answer.

    Same keys as every other response so a client renders it through the path
    it already has, and three of them chosen so that nothing can mistake this
    for something the corpus said. `status` is `SOCIAL`, which is neither
    `ANSWERED` nor `ABSTAINED` — the evidence was not consulted, so counting
    this as either would corrupt both rates. `answer` is null, so no citation,
    confidence or grounding flag exists to be read. `evidence` is empty,
    because nothing was retrieved.

    The reply text is a constant from `app/intent.py`. No provider is called,
    nothing is cached and no clarification is attached: a greeting has nothing
    to clarify.
    """
    return {
        "question": question,
        "status": "SOCIAL",
        "social": social.to_dict(),
        "service_failure": None,
        "answer": None,
        "rendered": social.reply,
        "disclaimer": SYSTEM_DISCLAIMER,
        "evidence": [],
        "timings_ms": {"retrieval": 0.0, "reranking": 0.0,
                       "answering": 0.0, "total": 0.0},
        "pipeline": {"candidates": 0, "evidence_count": 0,
                     "stage": "intent_gate",
                     "intent": intent_config()},
        "cache": {"hit": False, "stored": False},
    }


def _needs_clarification(question: str,
                         clarification: Any,
                         correction: Any = None) -> dict[str, Any]:
    """The response for a question that named no body of law at all.

    Deliberately shaped like every other response — same keys, same disclaimer,
    an empty `evidence` list — so a frontend renders it through the code path it
    already has rather than needing a second one. `status` is its own value and
    not `ABSTAINED`, because the two are different events: an abstention means
    the system looked and the evidence did not support an answer, and this means
    it has not looked yet and is asking what to look for. Reporting the second
    as the first would inflate the abstention count with questions the evidence
    was never consulted about.

    No provider call is made, nothing is cached, and the next request — which
    the caller sends with `clarified: true` — runs the full pipeline whatever
    the user typed. That is what bounds Phase E to one question.
    """
    return {
        "question": question,
        "status": "NEEDS_CLARIFICATION",
        "spelling": correction.to_dict() if correction is not None else None,
        "service_failure": None,
        "answer": None,
        "rendered": "",
        "disclaimer": SYSTEM_DISCLAIMER,
        "evidence": [],
        "clarification": clarification.to_dict(),
        "timings_ms": {"retrieval": 0.0, "reranking": 0.0,
                       "answering": 0.0, "total": 0.0},
        "pipeline": {"candidates": 0, "evidence_count": 0,
                     "stage": "pre_retrieval",
                     "clarification": clarify_config()},
        "cache": {"hit": False, "stored": False},
    }


def _from_cache(entry: dict[str, Any], question: str,
                lookup_ms: float) -> dict[str, Any]:
    """Shape a stored entry as this request's result.

    The answer, citations, cautions, confidence, grounding and evidence are
    returned exactly as they were published — that is the point, and altering
    any of them here would make a cached response a different response.

    Three things are this request's rather than the stored one's. The question
    is the one actually asked, so the response echoes the caller's words and
    not the normalised-equivalent phrasing that produced the entry. The timings
    are the real ones, which is why a hit reports the lookup instead of
    claiming the provider latency it did not spend. And `cache` says plainly
    where the result came from, including the question that generated it.
    """
    result = entry["result"]
    original_timings = result.get("timings_ms")
    result["question"] = question
    result["timings_ms"] = {"retrieval": 0.0, "reranking": 0.0,
                            "answering": 0.0, "total": lookup_ms}
    result["cache"] = {
        "hit": True,
        "stored": True,
        # "seed" shipped with the build; "runtime" was produced by this
        # deployment. Both went through the same gate — the distinction is
        # operational, so a log line can say whether a demo is running on
        # answers the deployment carried in or answers it earned.
        "source": entry.get("source", "runtime"),
        "cached_at": entry.get("cached_at"),
        "cached_question": entry.get("question"),
        "original_timings_ms": original_timings,
    }
    return result


class _NoDense:
    """Stands in for the dense arm when it could not be loaded.

    Named for what it is. `CandidateGenerator` merges whatever each engine
    returns, so one contributing engine makes the round-robin the identity
    function over the lexical ranking — the same mechanism Phase 7.3 used.
    """

    name = "dense_unavailable"
    version = "0.0.0"

    def search(self, query, filters=None, top_k=10):
        return []

    def config(self):
        return {"retriever": self.name, "disabled_reason": "model unavailable"}


def build(log=print) -> Pipeline:
    return Pipeline(
        use_dense=_flag("IPSAKTI_DENSE", True),
        reranker=os.environ.get("IPSAKTI_RERANKER", "features"),
        provider=os.environ.get("IPSAKTI_PROVIDER", "auto"),
        cache=_flag("IPSAKTI_CACHE", True),
        cache_path=os.environ.get("IPSAKTI_CACHE_PATH") or DEFAULT_CACHE_PATH,
        seed_cache_path=os.environ.get("IPSAKTI_SEED_CACHE_PATH") or DEFAULT_SEED_PATH,
        log=log)
