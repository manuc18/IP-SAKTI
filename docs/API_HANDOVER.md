# IP-SAKTI API — handover for the development team

One endpoint answers questions. Everything else on this page describes what
comes back from it.

```
POST /api/v1/query   { "question": "..." }   →   ANSWERED | ABSTAINED | UNAVAILABLE
```

The ML engine decides; the API transports. No field below is computed by the
API layer — each is copied from the pipeline that `ai/` already produced and
`tests/` already covers.

---

## 1. Start it

```bash
.venv/bin/python -m app.api_server
```

```
IP-SAKTI API running
http://127.0.0.1:8000

POST /api/v1/query
GET  /health
```

Startup takes about 13 seconds: the corpus, both indexes and the BGE-M3 encoder
load **before** the port opens, so the first request is as fast as the
hundredth. If the banner printed, the engine is up.

| flag | meaning |
|---|---|
| `--port 8001` | use when the demo UI server already holds 8000 (`python -m app.server`) |
| `--host 0.0.0.0` | expose on the LAN. Default is loopback only |
| `--provider extractive` | offline, deterministic answers — no network, no API key |
| `--provider groq` | live model. Needs `GROQ_API_KEY`; falls back to extractive if it is missing |
| `--no-dense` | skip BGE-M3. Starts in <1s, retrieval quality drops. Useful for frontend work |
| `--reranker cross_encoder` | higher-scoring reranker, unmeasured latency |
| `--debug` | verbose Flask errors. The reloader stays off — reloading rebuilds the engine |

**Frontend developers who only need the JSON shape** should start with
`--provider extractive --no-dense`: under a second to boot, no credential, no
network, and the response contract is byte-for-byte identical.

### Environment variables

| variable | default | notes |
|---|---|---|
| `IPSAKTI_API_HOST` / `HOST` | `127.0.0.1` | |
| `IPSAKTI_API_PORT` / `PORT` | `8000` | |
| `IPSAKTI_PROVIDER` | `auto` | `auto` \| `groq` \| `extractive` |
| `IPSAKTI_RERANKER` | `features` | `features` \| `cross_encoder` |
| `IPSAKTI_DENSE` | `1` | `0` disables BGE-M3 |
| `IPSAKTI_CORS_ORIGINS` | `*` | comma-separated allow-list; empty string sends no CORS headers |
| `IPSAKTI_DEBUG` | `0` | |
| `GROQ_API_KEY` | — | read by `ai/answering/groq.py`, also read from `.env`. Never returned by any endpoint |

CORS is wide open by default, which is why the server binds to loopback.
Narrow `IPSAKTI_CORS_ORIGINS` before putting it on a network.

---

## 2. `GET /health`

Liveness only. No retrieval, no model call, no network request — safe to poll.

```json
{
  "status": "ok",
  "service": "ip-sakti-api",
  "api_version": "v1",
  "engine": {
    "status": "ok",
    "app_version": "0.1.0",
    "startup_seconds": 13.41,
    "corpus": { "chunks": 3073, "documents": 22 },
    "retrieval": { "dense": "ready", "lexical": "ready", "reranker": "features" },
    "answering": { "provider": "groq · openai/gpt-oss-120b", "versions": { … } },
    "warnings": []
  }
}
```

`engine.warnings` is worth surfacing somewhere in an ops view: it is how the
server says "dense retrieval failed to load, I am running lexical-only".

`GET /` returns the service name, version and endpoint list. Nothing else.

---

## 3. `POST /api/v1/query`

```http
POST /api/v1/query
Content-Type: application/json
```

```json
{ "question": "How must an application for registration of a trade mark be signed?" }
```

That is the whole required contract. Two optional fields:

```json
{
  "question": "...",
  "request_id": "your-correlation-id",
  "options": { "provider": "default" }
}
```

* `request_id` — echoed back verbatim if it is a printable single-line string
  of ≤64 characters. Otherwise the server mints a UUID. Either way, every
  response carries one.
* `options.provider` — accepts `"default"` or the provider already running.
  Anything else is a `400 UNSUPPORTED_OPTION`. The provider is fixed at
  startup by design: the engine never switches models mid-flight, and a request
  that silently did not get what it asked for would be worse than a refusal.

Limits: question ≤ 500 characters, body ≤ 64 KiB.

---

## 4. The three result states

These are not interchangeable, and the frontend must render them differently.

| state | HTTP | `success` | means |
|---|---|---|---|
| `ANSWERED` | 200 | `true` | An answer passed citation validation and grounding |
| `ABSTAINED` | 200 | `true` | The pipeline ran and **decided the evidence does not support an answer** |
| `UNAVAILABLE` | 503 | `false` | The answer provider could not be reached. No decision was made |

`ABSTAINED` is a feature, not a failure — it is the property that separates
this system from a chatbot. `UNAVAILABLE` is an outage. Never render them the
same way, and never describe a rate limit as "insufficient evidence".

### A. `ANSWERED`

```json
{
  "success": true,
  "status": "ANSWERED",
  "request_id": "cdf5e9fa2e9b4ab5be007ccc62711311",
  "question": "How must an application for registration of a trade mark be signed?",

  "answer": {
    "text": "An application shall be signed by the applicant or his agent [1].",
    "confidence": "MODERATE",
    "grounded": true
  },

  "citations": [
    {
      "marker": "[1]",
      "marker_number": 1,
      "chunk_id": "IN-TM-RULES-2017::part-i-chapter-ii-rule-23",
      "document": "Trade Marks Rules, 2017",
      "document_id": "IN-TM-RULES-2017",
      "locator": "Rule 23",
      "heading": "Form and signing of application",
      "page": 59,
      "pages": [59],
      "provenance_level": "B",
      "temporal_status": "CURRENT",
      "citable": true,
      "citation_text": "Trade Marks Rules, 2017 — Rule 23",
      "locator_unavailable": false,
      "source_document": {
        "document_id": "IN-TM-RULES-2017",
        "filename": "Trade Marks Rules, 2017.pdf",
        "sha256": "3ee682af…",
        "official_source_url": "https://ipindia.gov.in/trade-marks-resources-rules",
        "source_version_id": "IN-TM-RULES-2017@2026-06-01",
        "consolidated_as_of": "2026-06-01"
      }
    }
  ],

  "sources": [ /* see §5 */ ],
  "cautions": [ /* see §6 */ ],
  "disclaimer": "IP-SAKTI is an information and research tool …",

  "meta": {
    "result_type": "ANSWERED",
    "evidence_retrieved": 5,
    "evidence_fingerprint": "7ef8d2e48b3391f7",
    "grounded": true,
    "conflict_detected": false,
    "timings_ms": { "retrieval": 32.8, "reranking": 22.6, "answering": 1804.1, "total": 1859.5 },
    "pipeline": { "candidates": 90, "evidence_count": 5, "retrievers": "…", "reranker": "features", "provider": "…" },
    "engine_version": "0.1.0"
  }
}
```

`confidence` is one of `HIGH` `MODERATE` `LOW` `NONE`. It is **ordinal, not a
probability**, and deliberately not a number — nothing in the pipeline estimates
the likelihood that an answer is legally correct.

`grounded` means every claim in the text carried a citation that resolved
against the supplied evidence. It is a structural property, not a statement
about legal accuracy.

`disclaimer` must be displayed. It is not decorative.

### B. `ABSTAINED`

```json
{
  "success": true,
  "status": "ABSTAINED",
  "request_id": "…",
  "question": "What is the best pizza topping in Naples?",
  "answer": null,
  "citations": [],
  "sources": [],
  "reason": {
    "code": "NO_RELEVANT_EVIDENCE",
    "message": "no supplied passage shares any content term with the query"
  },
  "reasons": [ { "code": "NO_RELEVANT_EVIDENCE", "message": "…", "marker": null, "chunk_id": null } ],
  "cautions": [],
  "disclaimer": "…",
  "meta": { "result_type": "ABSTAINED", "evidence_retrieved": 5, … }
}
```

`reason.code` is the engine's own abstention code, upper-cased. The set you can
expect: `NO_EVIDENCE`, `NO_RELEVANT_EVIDENCE`, `NO_CITABLE_AUTHORITY`,
`ONLY_STALE_EVIDENCE`, `CONFLICTING_EVIDENCE`, `PROVIDER_ABSTAINED`,
`CITATION_VALIDATION_FAILED`, `UNGROUNDED_ANSWER`. Treat the list as open —
branch on the ones you want to word specially and fall back to
`reason.message` for the rest.

`meta.evidence_retrieved` is the count of passages the retriever found even
though none are published. An abstention over five passages and one over zero
are different events, and you can say which.

### C. `UNAVAILABLE`

```json
{
  "success": false,
  "status": "UNAVAILABLE",
  "request_id": "…",
  "question": "How must a trade mark application be signed?",
  "answer": null,
  "citations": [],
  "sources": [],
  "error": {
    "code": "PROVIDER_UNAVAILABLE",
    "message": "The language provider is rate-limited right now (free-tier per-minute cap). No answer was generated.",
    "provider_error": { "code": "rate_limited", "detail": "Groq rate limit reached (429)" }
  },
  "cautions": [],
  "disclaimer": "…",
  "meta": { "result_type": "UNAVAILABLE", "evidence_retrieved": 5, … }
}
```

HTTP **503**. `error.message` is safe to show a user. `error.provider_error` is
for your logs: `rate_limited`, `transport_error`, `unauthorized`, `forbidden`,
`server_error`, `missing_api_key`, and so on.

The free Groq tier has a per-minute token cap that back-to-back requests will
hit. **Do not add a client-side retry loop** — the engine deliberately has no
retry, and hammering the cap keeps it closed. Show the message and let the user
ask again.

### D. `INVALID_REQUEST` and other errors

```json
{
  "success": false,
  "status": "INVALID_REQUEST",
  "request_id": "…",
  "error": { "code": "EMPTY_QUESTION", "message": "Question must not be empty." }
}
```

| code | HTTP | cause |
|---|---|---|
| `INVALID_CONTENT_TYPE` | 400 | missing `Content-Type: application/json` |
| `INVALID_JSON` | 400 | body is not valid JSON |
| `INVALID_BODY` | 400 | body is not a JSON object |
| `MISSING_QUESTION` | 400 | no `question` field |
| `INVALID_QUESTION_TYPE` | 400 | `question` is not a string |
| `EMPTY_QUESTION` | 400 | empty or whitespace-only |
| `QUESTION_TOO_LONG` | 400 | over 500 characters |
| `INVALID_OPTIONS` / `UNSUPPORTED_OPTION` | 400 | bad `options` |
| `NOT_FOUND` | 404 | unknown route |
| `METHOD_NOT_ALLOWED` | 405 | e.g. `GET /api/v1/query` |
| `REQUEST_TOO_LARGE` | 413 | body over 64 KiB |
| `INTERNAL_ERROR` | 500 | unexpected failure — details go to the server log only |

Every error, including 404 and 405, returns this same JSON envelope. No HTML,
no stack traces, no filesystem paths.

---

## 5. The source object

`sources` is the ranked evidence the answer was written from — the source cards
to render. Present on `ANSWERED` only.

| field | type | meaning |
|---|---|---|
| `marker` | `"[1]"` | matches the marker in `answer.text` |
| `marker_number` | int | the same thing, unwrapped |
| `chunk_id` | string | stable id. **Use this as the click target** |
| `document` | string | readable name, e.g. `"Trade Marks Rules, 2017"` |
| `document_id` | string | e.g. `"IN-TM-RULES-2017"` |
| `locator` | string | `"Rule 23"`, `"Section 71"` — the provision |
| `citation` | string | full citation line, document + locator |
| `heading` | string \| null | the provision's own heading |
| `page` / `pages` | int \| null / int[] | physical PDF page(s) |
| `provenance_level` | `A`–`D` | source authority. `C`/`D` = publisher unestablished |
| `temporal_status` | `CURRENT` \| `STALE` \| `UNKNOWN` | `STALE` = may have been amended |
| `citable` | bool | whether it may be cited as authority at all |
| `locator_unavailable` | bool | true when the provision could not be pinpointed |
| `official_source_url` | string \| null | present only where provenance established one |
| `cited` | bool | whether the answer actually cited it |
| `retrieval_rank` | int | rank before the answer was written |
| `text` | string | the full evidence passage |

`citations` carries the same identity fields plus `citation_text` and
`source_document` (filename, SHA-256, version id, consolidation date).

### Wiring `[n]` to a card

`answer.text` contains bare markers: `"… signed by the applicant or his agent [1]."`

1. Regex `\[(\d+)\]` over the answer text.
2. Look the number up in `citations` by `marker_number`.
3. Scroll to the `sources` entry with the matching `chunk_id`.

Every citation returned resolves to a returned source — the engine validated
that before publishing. A marker that does not resolve is a bug worth
reporting, not a case to render.

---

## 6. Cautions

```json
"cautions": [
  { "type": "STALE_SOURCE",
    "message": "The cited source is marked stale. The text may have been amended since the consolidated version held in the corpus; verify against the current official text." }
]
```

`message` is the engine's string, verbatim — display it as-is. `type` is a
label the API adds so you can pick an icon: `STALE_SOURCE`,
`WEAK_PROVENANCE`, `MIXED_TEMPORAL_STATUS`, `CONFLICTING_EVIDENCE`, or
`CAUTION` for anything new.

Cautions appear alongside answers. Show them; they are the difference between
a citation and a *trustworthy* citation.

---

## 7. Rendering guidance

**ANSWERED** — answer text → citations → source cards → cautions → disclaimer.
Show the confidence level as a label, never as a percentage.

**ABSTAINED** —

> IP-SAKTI could not find sufficient legal evidence to safely answer this question.

Not a server error, not a model failure, and never padded out with a guess.

**UNAVAILABLE** —

> Answer generation is temporarily unavailable. Please try again in a moment.

Visually distinct from an abstention. This one is worth a retry button; an
abstention is not.

**While waiting** — a live Groq answer typically takes 1.5–3 seconds end to end
(`meta.timings_ms` breaks it down). Retrieval is ~30 ms of that; the model is
the rest.

---

## 8. Boundaries

The engine owns: retrieval, reranking, evidence selection, answer generation,
citation validation, grounding, abstention, confidence, source metadata.

You own: UI, chat surface, loading states, source cards, citation clicks, error
display, auth, deployment.

The API layer between them validates, calls, classifies, serialises. It holds
no legal logic and no thresholds, and it cannot turn an abstention into an
answer.

---

## 9. Before you deploy this

* `app.run()` is Flask's development server. Put a WSGI server in front
  (`gunicorn -w 1 --timeout 120 'app.api_server:create_app()'`, or `waitress`)
  — and note **`-w 1`**: each worker loads a full copy of the corpus and the
  encoder, and the pipeline answers under a single lock.
* There is no authentication and no rate limiting. Add both before this is
  reachable from anywhere but localhost.
* Nothing is persisted. No database, no request log beyond stdout.
* **No claim of legal accuracy and no claim of production readiness.** The
  corpus is 22 documents; no answer in this project has been reviewed by a
  lawyer. See `PROJECT.md` §4 for what the numbers do and do not mean.

---

## 10. Quick check

```bash
curl -s localhost:8000/health

curl -s -X POST localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"How must an application for registration of a trade mark be signed?"}'

# a safe abstention
curl -s -X POST localhost:8000/api/v1/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the best pizza topping in Naples?"}'
```

API tests: `.venv/bin/python -m pytest tests/test_api_server.py -q` (28 tests,
no network, no model). Full suite: `.venv/bin/python -m pytest -q`.
