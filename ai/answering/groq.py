"""The Groq answer provider — the first real model behind the Phase 7 interface.

This is the only module in the project that opens a network connection during
answering, and it is deliberately small. Everything that makes an answer
trustworthy already exists upstream and downstream of it: the evidence policy
decides what may be cited, the citation validator decides what actually is, and
the grounding check decides whether the prose earned it. A model is a writer of
sentences here, not an authority, and this module's job is to fetch sentences
and hand them to machinery that does not trust them.

**What leaves the machine.** Three things and nothing else: the user's query,
the text of the passages retrieval already selected, and the versioned answer
instructions. No chunk id, no document id, no page number, no filename, no
SHA-256, no URL, no retrieval or reranker score, no benchmark metadata, no
corpus path, no PDF, no embedding. The `Passage` objects a provider receives
carry no such field — Phase 7 built them that way — so this is a property of the
type rather than of this module's discretion. `_audit_payload` re-checks the
scaffolding anyway, on every call.

**One request, no retries, no fallbacks.** A 429 is a 429: it returns the
existing abstention path with a reason, and the engine says it could not answer.
Retrying would turn one rate-limit into several, and silently switching models
would make a result unattributable to the model that produced it — the identity
problem Phase 6 §8 refused to have.

**The key is never printed.** It is read from the environment, placed in one
header, and scrubbed from every diagnostic string this module can emit.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from ai.answering.providers import (
    ANSWER_INSTRUCTIONS,
    PROMPT_VERSION,
    ExternalLLMProvider,
    ProviderDraft,
    ProviderRequest,
)

GROQ_VERSION = "0.1.0"

#: Groq's OpenAI-compatible chat completions endpoint.
ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

#: Every HTTP client is expected to say what it is, and this one did not: with
#: no header set, urllib sends `Python-urllib/3.x`, a signature Cloudflare bans
#: outright in front of api.groq.com. The result was a 403 carrying `error code:
#: 1010` and no JSON body — indistinguishable, from inside the provider, from
#: "this key may not use this model", and the reason Phase 7.3's first live test
#: was misdiagnosed as an account problem. Groq's own SDK sends a User-Agent for
#: the same reason; naming the application is the honest form of this header,
#: not a disguise as some other client.
USER_AGENT = f"ip-sakti/{GROQ_VERSION} (grounded-answer-engine; python-urllib)"

#: Configurable through GROQ_MODEL. Recorded on every draft, because an answer
#: produced by an unrecorded model is not a reproducible result.
DEFAULT_MODEL = "openai/gpt-oss-120b"

#: Phase 7.2 §5 caps this at 15 seconds. GROQ_TIMEOUT_SECONDS may lower it and
#: is clamped rather than trusted: a config file cannot raise the ceiling.
MAX_TIMEOUT_SECONDS = 15.0
DEFAULT_TIMEOUT_SECONDS = 15.0

#: Deterministic generation. Temperature 0 does not make a model reproducible —
#: batching and kernel non-determinism remain — but it removes the one source of
#: variance we control, which matters when the same query is re-run to compare
#: policies rather than to compare samples.
TEMPERATURE = 0.0
TOP_P = 1.0

#: An answer that cites its evidence does not need to be long. This is a cost
#: ceiling as much as a style one: the free plan is a fixed token budget.
#:
#: Phase 7.6 measured this rather than adjusting it. Q027's completion was
#: truncated mid-word, which is the kind of observation that usually ends in a
#: raised limit, so the budget was taken apart first:
#:
#: * prompt        ~2231 tokens (system instructions ~454 + five passages ~1777)
#: * completion     ~577 tokens of visible prose across 9 sentences
#: * per sentence    ~64 tokens
#: * unaccounted    ~223 tokens — this model reasons, and Groq charges reasoning
#:                  against `max_completion_tokens` alongside the visible text
#:
#: Truncation was a consequence of nine sentences, not of a tight ceiling. Under
#: the 1-4 sentence contract of prompt 0.3.0 the visible budget is ~256 tokens,
#: leaving ~540 for reasoning — more than twice what Q027 used. So 800 stays.
#:
#: Raising it to 1000 would have cost something real in both directions: ~200
#: more tokens per request against a free-tier TPM ceiling that already returned
#: three 429s in Phase 7.3, and a larger ceiling is an invitation to write the
#: long answer whose extra sentences are precisely what goes uncited. The limit
#: that truncated Q027 was not the problem; the nine sentences were.
DEFAULT_MAX_COMPLETION_TOKENS = 800

#: GROQ_MAX_COMPLETION_TOKENS may lower the ceiling and is clamped rather than
#: trusted, in both directions: a config file cannot raise the cost per request,
#: and it cannot drop the budget below what a cited four-sentence answer plus
#: this model's reasoning has actually been observed to need.
#:
#: The default is unchanged, and deliberately so. Two live runs are on record —
#: 210 and 328 completion tokens, of which 174 and 243 were reasoning, both
#: finishing on `stop` — so nothing measured has come close to 800. But a
#: ceiling is not a charge: a successful request bills what it used, so lowering
#: this cannot reduce the tokens a successful answer costs. What it can do is
#: truncate a longer answer mid-sentence, which fails grounding and publishes
#: nothing. The knob exists so a rate-limited venue can trade that risk
#: deliberately; the default does not take the trade on anyone's behalf.
MIN_MAX_COMPLETION_TOKENS = 384

#: Substrings that must never appear in the scaffolding this module builds. The
#: passage bodies are corpus text and are exempt — the same distinction Phase 6
#: drew when its leakage audit scanned candidate headers rather than statutory
#: prose, which contains anything you care to forbid.
FORBIDDEN_IN_SCAFFOLDING = ("chunk_id", "document_id", "physical_pages", "sha256",
                            "source_filename", "official_source_url",
                            "rerank_score", "dense_score", "lexical_score",
                            "expected_chunks", "data/processed", "data/evaluation")


def load_env_file(path: str | None = None) -> None:
    """Read `.env` into the environment without overriding what is already set.

    Ten lines instead of a dependency, and it never overwrites a variable the
    caller exported deliberately. The file is gitignored; nothing in this
    repository writes a credential anywhere else.
    """
    if path is None:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        path = os.path.join(root, ".env")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    if "SSL_CERT_FILE" not in os.environ:
        try:
            import certifi
            os.environ["SSL_CERT_FILE"] = certifi.where()
        except ImportError:
            pass


@dataclass(frozen=True)
class GroqConfig:
    """Everything the provider needs, and nothing secret.

    The API key is deliberately **not** a field. It is read at call time and
    lives only in the request header, so no config object, repr, log line or
    serialised diagnostic can carry it by accident.
    """

    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    max_completion_tokens: int = DEFAULT_MAX_COMPLETION_TOKENS
    temperature: float = TEMPERATURE
    endpoint: str = ENDPOINT

    @classmethod
    def from_environment(cls, *, load_dotenv: bool = True) -> "GroqConfig":
        if load_dotenv:
            load_env_file()
        try:
            timeout = float(os.environ.get("GROQ_TIMEOUT_SECONDS",
                                           DEFAULT_TIMEOUT_SECONDS))
        except ValueError:
            timeout = DEFAULT_TIMEOUT_SECONDS
        if timeout <= 0:
            timeout = DEFAULT_TIMEOUT_SECONDS
        try:
            budget = int(os.environ.get("GROQ_MAX_COMPLETION_TOKENS",
                                        DEFAULT_MAX_COMPLETION_TOKENS))
        except ValueError:
            budget = DEFAULT_MAX_COMPLETION_TOKENS
        return cls(model=os.environ.get("GROQ_MODEL") or DEFAULT_MODEL,
                   timeout_seconds=min(timeout, MAX_TIMEOUT_SECONDS),
                   max_completion_tokens=min(max(budget, MIN_MAX_COMPLETION_TOKENS),
                                             DEFAULT_MAX_COMPLETION_TOKENS))

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "timeout_seconds": self.timeout_seconds,
                "max_completion_tokens": self.max_completion_tokens,
                "temperature": self.temperature, "endpoint": self.endpoint,
                "stream": False, "retries": 0}


class GroqError(RuntimeError):
    """A request that did not produce usable prose. Carries a stable code."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


def render_evidence(request: ProviderRequest) -> str:
    """The user message: the question, then the numbered passages.

    Each passage is labelled with the marker the model must cite, the instrument
    and locator so it knows what it is reading, and the temporal status and
    evidence level — without those last two, instructions 6 and 7 are impossible
    to follow. There is nothing else to include; a `Passage` holds nothing else.
    """
    lines = [f"Question: {request.query}", "", "Evidence passages:"]
    for passage in request.passages:
        status = passage.temporal_status
        note = "" if passage.citable else ", not citable"
        lines.append("")
        lines.append(f"[{passage.marker}] {passage.document} — {passage.locator} "
                     f"({status}, source evidence level "
                     f"{passage.source_evidence_level}{note})")
        lines.append(passage.text)
    if not request.passages:
        lines.append("")
        lines.append("(none)")
    return "\n".join(lines)


def build_payload(request: ProviderRequest, config: GroqConfig) -> dict[str, Any]:
    """The exact JSON body sent to Groq. Audited before it goes."""
    system = "\n".join([
        "You answer questions about Indian intellectual property, biodiversity "
        "and regulatory law strictly from evidence supplied to you.",
        "",
        *ANSWER_INSTRUCTIONS,
        "",
        "Write plain prose with [n] markers. Do not write document identifiers, "
        "page numbers, URLs, filenames, JSON, confidence scores, or a source "
        "list — the citation layer builds all of those from its own records.",
    ])
    payload = {
        "model": config.model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": render_evidence(request)}],
        "temperature": config.temperature,
        "top_p": TOP_P,
        "max_completion_tokens": config.max_completion_tokens,
        "stream": False,
    }
    _audit_payload(payload, request)
    return payload


def _audit_payload(payload: Mapping[str, Any], request: ProviderRequest) -> None:
    """Check the scaffolding we built, not the corpus text we were given."""
    scaffolding = [payload["messages"][0]["content"]]
    user = payload["messages"][1]["content"]
    bodies = {passage.text for passage in request.passages}
    for line in user.split("\n"):
        if line not in bodies:
            scaffolding.append(line)
    joined = "\n".join(scaffolding)
    for forbidden in FORBIDDEN_IN_SCAFFOLDING:
        if forbidden in joined:
            raise GroqError("payload_audit_failed",
                            f"the request scaffolding contained {forbidden!r}")


def _urllib_transport(url: str, body: bytes, headers: Mapping[str, str],
                      timeout: float) -> tuple[int, bytes]:
    """One POST. No retry, no redirect handling, no session, no streaming."""
    request = urllib.request.Request(url, data=body, headers=dict(headers),
                                     method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:          # 4xx / 5xx with a body
        return exc.code, exc.read()
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise GroqError("transport_error", _transport_detail(exc, timeout)) from None


def _transport_detail(exc: BaseException, timeout: float) -> str:
    """Name what actually failed, not just the class that carried it.

    `urllib` wraps a read timeout, a DNS failure, a refused connection and a TLS
    error in the same `URLError`, so recording `type(exc).__name__` produced the
    string "URLError" for all four. That is the same mistake the 403 handling
    made before `_reason()` existed: a diagnostic that names the category and
    not the cause forces a second live request to learn what the first was
    already told — and on a deployed server there may be no second chance to
    reproduce it.

    So the wrapped `reason` is unwrapped, and a timeout is stated as one
    together with the limit it hit, because "did Groq take longer than 15
    seconds" and "could the host be reached at all" have entirely different
    fixes. This reads no gate and changes no outcome: the failure is still
    `transport_error`, still one request, still no retry. The caller scrubs the
    credential from whatever comes back.
    """
    reason = getattr(exc, "reason", None)
    if isinstance(reason, BaseException):
        inner = reason
    else:
        inner = exc
    if isinstance(inner, TimeoutError) or "timed out" in str(inner).lower():
        return (f"the request exceeded the {timeout:g}s timeout "
                f"({type(inner).__name__}); raise GROQ_TIMEOUT_SECONDS or "
                f"retry when the provider is less loaded")
    detail = str(reason if reason is not None else inner).strip()
    return f"{type(inner).__name__}: {detail[:200]}" if detail else type(inner).__name__


def _completion_meta(body: bytes) -> dict[str, Any]:
    """`finish_reason` and token counts, for the diagnostics. Never raises.

    Phase 7.5 had to infer that Q027 was truncated from the fact that it stopped
    mid-word, and then reason about the token budget from character counts.
    Groq already reports both in the response; not reading them meant a decision
    about `max_completion_tokens` rested on an estimate when a measurement was
    sitting in the body. This reads no gate and changes no outcome — a
    completion is accepted or refused identically whether or not this succeeds,
    which is why every failure here is swallowed rather than raised.
    """
    meta: dict[str, Any] = {}
    try:
        document = json.loads(body.decode("utf-8"))
        choice = document["choices"][0]
        meta["finish_reason"] = choice.get("finish_reason")
        usage = document.get("usage") or {}
        for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
            if field in usage:
                meta[field] = usage[field]
        # This model reasons, and Groq charges reasoning against
        # `max_completion_tokens` alongside the visible text. Phase 7.6 had to
        # derive the split by subtracting an estimate of the prose from the
        # ceiling; if the provider reports it, record it.
        details = usage.get("completion_tokens_details") or {}
        if "reasoning_tokens" in details:
            meta["reasoning_tokens"] = details["reasoning_tokens"]
    except Exception:                               # noqa: BLE001 - diagnostics only
        return {}
    return meta


def _reason(body: bytes) -> str:
    """Groq's own explanation of a rejection, as a parenthetical.

    A status code names the category and not the cause: a 403 is equally "this
    key may not use this model", "the terms for this model are unaccepted" and
    "this organisation is restricted", which have three different fixes. Dropping
    the body forced a second live request to learn something the first one had
    already been told. Best-effort by design — an error body is not a contract,
    so an unparseable one degrades to the bare status rather than raising a
    second failure while reporting the first. The caller scrubs the credential
    from whatever comes back, and a request body is never echoed here.
    """
    try:
        document = json.loads(body.decode("utf-8"))
        message = document.get("error", {}).get("message") or document.get("message")
    except (ValueError, AttributeError, UnicodeDecodeError):
        return ""
    return f": {str(message).strip()[:300]}" if message else ""

class GroqProvider(ExternalLLMProvider):
    """Groq's OpenAI-compatible chat completions API, one request per call.

    Subclasses the integration point Phase 7 declared rather than going around
    it, so the promise in `docs/ANSWER_GENERATION_CONTRACT.md` §5 — that a
    provider is the entire surface a model touches — stays literally true.
    Retrieval, policy, citation validation, grounding and rendering are
    unchanged, and this provider's output is validated exactly as the
    deterministic one's is. A provider is not trusted because of what it is.
    """

    name = "groq"
    version = GROQ_VERSION

    def __init__(self, *, config: GroqConfig | None = None,
                 transport: Callable[..., tuple[int, bytes]] | None = None,
                 api_key: str | None = None):
        # Named `settings`, not `config`: `AnswerProvider.config()` is a method
        # on the interface, and an attribute of the same name would shadow it —
        # silently, until the engine called it.
        self.settings = config or GroqConfig.from_environment()
        self._transport = transport or _urllib_transport
        self._api_key = api_key
        super().__init__(model=self.settings.model)

    # -- credentials ---------------------------------------------------------

    def _key(self) -> str:
        key = self._api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise GroqError("missing_api_key",
                            "GROQ_API_KEY is not set; put it in .env or export it")
        return key

    def _scrub(self, text: str) -> str:
        """Last line of defence: no diagnostic may echo the credential."""
        key = self._api_key or os.environ.get("GROQ_API_KEY")
        return text.replace(key, "***") if key else text

    # -- the request ---------------------------------------------------------

    def generate(self, request: ProviderRequest) -> ProviderDraft:
        """One request. Any failure returns the abstention path, never raises."""
        try:
            key = self._key()
            payload = build_payload(request, self.settings)
            status, body = self._transport(
                self.settings.endpoint,
                json.dumps(payload).encode("utf-8"),
                {"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "User-Agent": USER_AGENT},
                self.settings.timeout_seconds)
            text = self._read_completion(status, body)
            meta = _completion_meta(body)
        except GroqError as exc:
            return self._abstain(exc.code, exc.detail)
        except Exception as exc:                    # noqa: BLE001 - see below
            # A provider must not be able to take the engine down. Anything
            # unforeseen becomes an abstention with its type recorded, which is
            # the same outcome as a 500 and is handled by the same code path.
            return self._abstain("unexpected_error", type(exc).__name__)

        if not text.strip():
            return self._abstain("empty_completion",
                                 "the model returned no text")
        return ProviderDraft(text=text.strip(), provider=self.name,
                             provider_version=self.version,
                             diagnostics={"model": self.settings.model,
                                          "prompt_version": PROMPT_VERSION,
                                          "requests": 1, "retries": 0, **meta})

    def _read_completion(self, status: int, body: bytes) -> str:
        if status == 401:
            raise GroqError("unauthorized",
                            f"Groq rejected the credential (401){_reason(body)}")
        if status == 403:
            raise GroqError("forbidden",
                            f"Groq refused the request (403){_reason(body)}")
        if status == 429:
            raise GroqError("rate_limited",
                            "Groq rate limit reached (429); the free plan has "
                            "per-minute and per-day caps and this provider does "
                            f"not retry{_reason(body)}")
        if status >= 500:
            raise GroqError("server_error", f"Groq returned {status}{_reason(body)}")
        if status != 200:
            raise GroqError("http_error", f"Groq returned {status}{_reason(body)}")
        try:
            document = json.loads(body.decode("utf-8"))
            return document["choices"][0]["message"]["content"] or ""
        except (ValueError, KeyError, IndexError, TypeError, UnicodeDecodeError) as exc:
            raise GroqError("malformed_response",
                            f"could not read a completion ({type(exc).__name__})") from None

    def _abstain(self, code: str, detail: str) -> ProviderDraft:
        """Empty text is the engine's `provider_abstained` signal (Phase 7 §14)."""
        return ProviderDraft(text="", provider=self.name,
                             provider_version=self.version,
                             diagnostics={"error": code,
                                          "detail": self._scrub(detail),
                                          "model": self.settings.model,
                                          "prompt_version": PROMPT_VERSION,
                                          "requests": 1, "retries": 0})

    def config(self) -> dict[str, Any]:
        """No credential appears here, and none ever may: this is serialised
        into every answer object's diagnostics."""
        return {"provider": self.name, "version": self.version,
                "prompt_version": PROMPT_VERSION, "requires_network": True,
                "implemented": True,
                "api_key_configured": bool(self._api_key
                                           or os.environ.get("GROQ_API_KEY")),
                **self.settings.to_dict()}
