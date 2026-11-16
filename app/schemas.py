"""Request validation and response serialisation for `POST /api/v1/query`.

This module is the whole translation layer. It decides nothing about an answer:
retrieval, reranking, citation validation, grounding, confidence and abstention
all happen inside `ai/` and reach here as a finished result, which is then
reshaped into a stable JSON contract and handed to the client.

Two rules govern the reshaping.

**Nothing is invented.** Every field below is copied out of what the engine
produced. Where a citation needs a human-readable document name or heading, it
is joined to the evidence passage carrying the same `chunk_id` — a lookup, not
a guess. A caution's `message` is the engine's own string, byte for byte; only
the `type` label beside it is added, by exact substring match against the four
caution families `ai/answering/policy.py` constructs.

**Nothing is upgraded.** An abstention stays an abstention, a provider failure
stays a provider failure, and no branch here can turn either into an answer.
"""
from __future__ import annotations

import dataclasses
import enum
import uuid
from typing import Any, Mapping

from app.config import MAX_REQUEST_ID_CHARS

#: Distinguishing substrings of the caution families in
#: `ai/answering/policy.py`. First match wins; anything unrecognised keeps its
#: text and is labelled generically rather than being reworded to fit a bucket.
CAUTION_TYPES = (
    ("marked stale", "STALE_SOURCE"),
    ("evidence level", "WEAK_PROVENANCE"),
    ("mixes current and superseded", "MIXED_TEMPORAL_STATUS"),
    ("Conflicting evidence was detected", "CONFLICTING_EVIDENCE"),
)
DEFAULT_CAUTION_TYPE = "CAUTION"

PROVIDER_UNAVAILABLE = "PROVIDER_UNAVAILABLE"


class RequestError(Exception):
    """A malformed request. Carries the code the client will see."""

    def __init__(self, code: str, message: str, http_status: int = 400):
        super().__init__(message)
        self.code = code
        self.message = message
        self.http_status = http_status


# -- serialisation primitives -------------------------------------------------


def to_jsonable(value: Any) -> Any:
    """Make an engine value safe for `json.dumps` without flattening meaning.

    The pipeline returns plain dictionaries today, but it returns them by
    calling `to_dict()` on frozen dataclasses that hold tuples — so tuples,
    dataclasses and enums are the three shapes that can plausibly appear if a
    future engine field is exposed directly. Each has one obvious JSON form.
    Anything else becomes its `str()`, which keeps the response serialisable
    instead of returning a 500 for a field nobody reads.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, enum.Enum):
        return to_jsonable(value.value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {k: to_jsonable(v) for k, v in dataclasses.asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(v) for v in value]
    return str(value)


# -- request ------------------------------------------------------------------


def new_request_id(client_value: Any = None) -> str:
    """Echo the client's id when it is usable; otherwise mint one.

    A client id is only accepted if it is a short single-line string. It ends up
    in server logs and in the response, and neither is a place for a newline.
    """
    if isinstance(client_value, str):
        candidate = client_value.strip()
        if (candidate and len(candidate) <= MAX_REQUEST_ID_CHARS
                and candidate.isprintable()):
            return candidate
    return uuid.uuid4().hex


def validate_query_request(payload: Any, *, max_question_length: int,
                           active_provider: str) -> tuple[str, str]:
    """Return `(question, request_id)` or raise :class:`RequestError`."""
    if not isinstance(payload, dict):
        raise RequestError("INVALID_BODY", "Request body must be a JSON object.")

    if "question" not in payload:
        raise RequestError("MISSING_QUESTION",
                           "Request must include a 'question' field.")
    question = payload["question"]
    if not isinstance(question, str):
        raise RequestError("INVALID_QUESTION_TYPE", "Question must be a string.")
    question = question.strip()
    if not question:
        raise RequestError("EMPTY_QUESTION", "Question must not be empty.")
    if len(question) > max_question_length:
        raise RequestError(
            "QUESTION_TOO_LONG",
            f"Question is too long ({len(question)} characters; the limit is "
            f"{max_question_length}).")

    options = payload.get("options")
    if options is not None:
        if not isinstance(options, dict):
            raise RequestError("INVALID_OPTIONS", "'options' must be a JSON object.")
        _check_provider_option(options.get("provider"), active_provider)

    return question, new_request_id(payload.get("request_id"))


def _check_provider_option(requested: Any, active_provider: str) -> None:
    """Accept `default`, or the provider already running. Refuse the rest.

    The provider is chosen once, at startup. Honouring a per-request override
    would mean either building a second provider mid-request or quietly ignoring
    the field — and a request that silently did not get what it asked for is
    worse than one that was told so.
    """
    if requested is None or requested == "default":
        return
    if not isinstance(requested, str):
        raise RequestError("INVALID_OPTIONS", "'options.provider' must be a string.")
    if requested == active_provider or active_provider.startswith(requested + " "):
        return
    raise RequestError(
        "UNSUPPORTED_OPTION",
        f"The answer provider is fixed at server startup (currently "
        f"'{active_provider}') and cannot be changed per request. Omit "
        f"'options.provider', or restart the server with --provider.")


# -- response -----------------------------------------------------------------


def error_response(code: str, message: str, *, status: str = "ERROR",
                   request_id: str | None = None,
                   question: str | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {
        "success": False,
        "status": status,
        "error": {"code": code, "message": message},
    }
    if request_id is not None:
        body["request_id"] = request_id
    if question is not None:
        body["question"] = question
    return body


def _caution(text: str) -> dict[str, str]:
    for needle, label in CAUTION_TYPES:
        if needle in text:
            return {"type": label, "message": text}
    return {"type": DEFAULT_CAUTION_TYPE, "message": text}


def _source(item: Mapping[str, Any]) -> dict[str, Any]:
    """One evidence passage, as a source card the frontend can render."""
    pages = [to_jsonable(p) for p in (item.get("physical_pages") or ())]
    return {
        "marker": f"[{item.get('marker')}]",
        "marker_number": to_jsonable(item.get("marker")),
        "chunk_id": to_jsonable(item.get("chunk_id")),
        "document": to_jsonable(item.get("document_name")),
        "document_id": to_jsonable(item.get("document_id")),
        "locator": to_jsonable(item.get("locator_text") or item.get("citation")),
        "citation": to_jsonable(item.get("citation")),
        "heading": to_jsonable(item.get("heading")),
        "page": pages[0] if pages else None,
        "pages": pages,
        "provenance_level": to_jsonable(item.get("source_evidence_level")),
        "temporal_status": to_jsonable(item.get("temporal_status")),
        "citable": to_jsonable(item.get("citable")),
        "locator_unavailable": to_jsonable(item.get("locator_unavailable")),
        "official_source_url": to_jsonable(item.get("official_source_url")),
        "cited": bool(item.get("cited")),
        "retrieval_rank": to_jsonable(item.get("retrieval_rank")),
        "text": to_jsonable(item.get("text")),
    }


def _citation(citation: Mapping[str, Any],
              evidence_by_chunk: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """One validated citation, joined to its evidence passage for display.

    The citation object owns the locator, pages, provenance level and temporal
    status; the evidence passage owns the document's readable name and heading.
    The join key is `chunk_id`, which the citation validator has already
    resolved inside the evidence set — this cannot introduce a source that was
    not supplied, because a citation that failed to resolve never got here.
    """
    source = evidence_by_chunk.get(citation.get("chunk_id"), {})
    pages = [to_jsonable(p) for p in (citation.get("pages") or ())]
    return {
        "marker": f"[{citation.get('marker')}]",
        "marker_number": to_jsonable(citation.get("marker")),
        "chunk_id": to_jsonable(citation.get("chunk_id")),
        "document": to_jsonable(source.get("document_name")
                                or citation.get("document_id")),
        "document_id": to_jsonable(citation.get("document_id")),
        "locator": to_jsonable(citation.get("locator")),
        "heading": to_jsonable(source.get("heading")),
        "page": pages[0] if pages else None,
        "pages": pages,
        "provenance_level": to_jsonable(citation.get("source_evidence_level")),
        "temporal_status": to_jsonable(citation.get("temporal_status")),
        "citable": to_jsonable(source.get("citable")),
        "citation_text": to_jsonable(citation.get("citation_text")),
        "locator_unavailable": to_jsonable(citation.get("locator_unavailable")),
        "source_document": to_jsonable(citation.get("source_document")),
    }


def serialize_result(result: Mapping[str, Any],
                     request_id: str) -> tuple[dict[str, Any], int]:
    """Map one `Pipeline.answer()` result to `(response body, HTTP status)`.

    The three states are not interchangeable and the HTTP status says so:
    ANSWERED and ABSTAINED are both 200, because the pipeline ran to completion
    and reached a decision; UNAVAILABLE is 503, because it did not.
    """
    engine = result.get("answer") or {}
    evidence = list(result.get("evidence") or ())
    status = result.get("status")
    # Additive, and deliberately outside the answer: a cached response is the
    # same published result served again, so nothing a client renders may
    # depend on this. It is here so the dev team can tell a 2 ms hit from a
    # 3 s generation without inferring it from the timings.
    cache = result.get("cache") or {}
    cache_hit = bool(cache.get("hit"))

    body: dict[str, Any] = {
        "success": status != "UNAVAILABLE",
        "status": status,
        "request_id": request_id,
        "question": to_jsonable(result.get("question")),
        "answer": None,
        "citations": [],
        "sources": [],
        "cautions": [_caution(c) for c in (engine.get("cautions") or ())],
        "disclaimer": to_jsonable(result.get("disclaimer")),
        "meta": {
            "result_type": status,
            # Evidence *was* retrieved even where none is returned below. An
            # abstention over five passages and an abstention over none are
            # different events, and a frontend that wants to say which can.
            "evidence_retrieved": len(evidence),
            "evidence_fingerprint": to_jsonable(engine.get("evidence_fingerprint")),
            "grounded": bool(engine.get("grounded")),
            "conflict_detected": bool(engine.get("conflict_detected")),
            "timings_ms": to_jsonable(result.get("timings_ms")),
            "pipeline": to_jsonable(result.get("pipeline")),
            "engine_version": to_jsonable(engine.get("version")),
            "cache": {"hit": cache_hit,
                      "source": to_jsonable(cache.get("source")) if cache_hit
                                else None,
                      "cached_at": to_jsonable(cache.get("cached_at"))},
            # "seed" (shipped with the build), "runtime" (produced by this
            # deployment) or "live_pipeline" (generated for this request).
            "response_source": (to_jsonable(cache.get("source")) or "runtime")
                               if cache_hit else "live_pipeline",
        },
    }

    if status == "UNAVAILABLE":
        failure = result.get("service_failure") or {}
        body["error"] = {
            "code": PROVIDER_UNAVAILABLE,
            "message": to_jsonable(failure.get("message"))
                       or "The answer generation service is currently unavailable.",
            # The engine's own code — `rate_limited`, `transport_error`,
            # `unauthorized` … — kept for the dev team's logs. It is a
            # transport diagnostic, never a statement about the evidence.
            "provider_error": {"code": to_jsonable(failure.get("code")),
                               "detail": to_jsonable(failure.get("detail"))},
        }
        return body, 503

    if status == "SOCIAL":
        # A greeting. The reply is a fixed string from `app/intent.py`, and it
        # goes in `reply` rather than in `answer` so no client can render it
        # through the path that expects citations and a confidence level. It has
        # neither, because nothing was retrieved.
        social = result.get("social") or {}
        body["reply"] = to_jsonable(result.get("rendered"))
        body["intent"] = to_jsonable(social.get("intent"))
        body["meta"]["retrieval_performed"] = False
        return body, 200

    if status == "NEEDS_CLARIFICATION":
        # The pipeline has not looked yet and is asking what to look for. Also
        # not an answer, and — like SOCIAL — kept out of `answer` so that an
        # empty `answer.text` can never reach a client as though it were one.
        clarification = result.get("clarification") or {}
        body["clarification"] = to_jsonable(clarification)
        body["meta"]["retrieval_performed"] = False
        return body, 200

    if status == "ABSTAINED":
        reasons = [
            {"code": str(r.get("code", "")).upper(),
             "message": to_jsonable(r.get("detail")),
             "marker": to_jsonable(r.get("marker")),
             "chunk_id": to_jsonable(r.get("chunk_id"))}
            for r in (engine.get("insufficient_evidence_reasons") or ())]
        first = reasons[0] if reasons else None
        body["reason"] = {
            "code": first["code"] if first else "NO_RELEVANT_EVIDENCE",
            "message": (first["message"] if first else
                        "The available legal evidence does not safely support an "
                        "answer to this question."),
        }
        body["reasons"] = reasons
        return body, 200

    body["answer"] = {
        "text": to_jsonable(engine.get("answer")),
        "confidence": to_jsonable(engine.get("confidence")),
        "grounded": bool(engine.get("grounded")),
    }
    evidence_by_chunk = {item.get("chunk_id"): item for item in evidence}
    body["citations"] = [_citation(c, evidence_by_chunk)
                         for c in (engine.get("citations") or ())]
    body["sources"] = [_source(item) for item in evidence]
    return body, 200
