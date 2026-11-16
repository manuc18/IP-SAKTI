# Answer provider — Groq

`ANSWER_PROVIDER_STATUS: PROVISIONAL` · provider `0.1.0` · prompt `0.2.0` · Phase 7.2, revised 7.4

The first real model behind the Phase 7 `AnswerProvider` interface. **Seven
live requests have been made**: the `--live` smoke test in Phase 7.2, the five
of Phase 7.3 — two completions, three rejected with HTTP 429 by the free tier —
and one of the two Phase 7.5 permitted. **No completion has ever been
published**; every one was refused by the citation or grounding gate. Nothing
about this model's answer quality has been measured.

Nothing here is a claim of legal accuracy, and nothing here is production-ready.

---

## 1. What it is

| | |
| --- | --- |
| Provider | Groq, OpenAI-compatible chat completions |
| Endpoint | `https://api.groq.com/openai/v1/chat/completions` |
| Model | `openai/gpt-oss-120b` (configurable via `GROQ_MODEL`) |
| Transport | `urllib.request` — Python standard library |
| Dependencies added | **none** |
| Module | `ai/answering/groq.py` — the only file in the project that opens a network connection during answering |

`GroqProvider` subclasses `ExternalLLMProvider`, the integration point Phase 7
declared, rather than going around it. Retrieval, evidence policy, citation
validation, grounding and rendering are unchanged; the model's output is
validated exactly as the deterministic provider's is.

**No SDK.** The `groq` package would add a dependency, a transitive HTTP stack
and a retry policy this design does not want, to send one JSON body to one URL.
The repository already fetches over `urllib` in `scripts/provenance/verify_source.py`.

---

## 2. Configuration

| Variable | Default | Notes |
| --- | --- | --- |
| `GROQ_API_KEY` | *(none)* | Required. Read from `.env` or the environment. Absent ⇒ the provider abstains without dialling. |
| `GROQ_MODEL` | `openai/gpt-oss-120b` | |
| `GROQ_TIMEOUT_SECONDS` | `15` | Clamped to a **15 s ceiling**; a config file cannot raise it. |

`.env` is gitignored, mode `600`, and read by `load_env_file()` — ten lines of
standard library instead of a dependency. It never overrides a variable already
exported, so a shell export wins.

**The key is never a field on any object.** It is not on `GroqConfig`, not on
the provider's `config()`, not in any diagnostic, and it is scrubbed from error
detail strings before they are returned. It exists in one place at runtime: the
`Authorization` header of a single request.

---

## 3. Exactly what leaves the machine

Two chat messages and five scalars:

```json
{
  "model": "openai/gpt-oss-120b",
  "messages": [
    { "role": "system", "content": "<the ten instructions from ANSWER_GENERATION_CONTRACT.md>" },
    { "role": "user",   "content": "Question: <user query>\n\nEvidence passages:\n\n[1] <document> — <locator> (CURRENT, source evidence level B)\n<passage text>" }
  ],
  "temperature": 0.0,
  "top_p": 1.0,
  "max_completion_tokens": 800,
  "stream": false
}
```

So: **the user's query, the text of the passages retrieval already selected, and
the versioned instructions.** Plus, per passage, the instrument name, its
locator, its temporal status and its evidence level — without the last two,
instructions 6 and 7 (distinguish stale, distinguish weak provenance) cannot be
followed.

### Excluded, and why the exclusion holds

| Never sent | Enforced by |
| --- | --- |
| `chunk_id` | `Passage` has no such field. Phase 7 built it that way so a model cannot emit an identifier it never saw. |
| `document_id` | Not a `Passage` field; the human-readable short title is sent instead. |
| Page numbers, `physical_pages` | Not a `Passage` field. |
| Source filename, SHA-256, URLs | Not `Passage` fields. |
| Retrieval scores, reranker scores, candidate order | Not `Passage` fields. |
| Golden labels, expected chunks, benchmark metadata | No parameter reaches the provider carrying them. |
| Corpus paths, `data/` anything | Never constructed into a payload. |
| **The corpus itself** | Only the ≤10 passages one query selected are ever serialised. |
| PDFs, embeddings, the retrieval index, the golden dataset | Nothing reads them in this path. |
| The API key | Header only; scrubbed from every diagnostic. |

`_audit_payload()` re-checks the scaffolding on **every call**, raising rather
than sending if a forbidden token appears. It exempts the passage bodies, for
the reason Phase 6 §14 recorded about anchor terms: statutory prose contains
anything you care to forbid, and scanning it would forbid the corpus from
containing itself.

Fifteen tests assert these exclusions against a real payload.

### Retained locally, never transmitted

Every citation the reader sees — document, locator, pages, source filename,
SHA-256, `official_source_url`, evidence level, temporal status — is built by
`citation_validator.build_citation()` from the local evidence record, after the
model has replied. The model contributes a `[n]` marker and nothing else.

---

## 4. The prompt

The existing versioned contract, not a competing one:
`ANSWER_INSTRUCTIONS` in `ai/answering/providers.py`, `PROMPT_VERSION = "0.2.0"`,
documented in `docs/ANSWER_GENERATION_CONTRACT.md`. The system message is those
ten instructions plus one sentence telling the model not to write identifiers,
pages, URLs, filenames, JSON or a source list, because the citation layer builds
all of them.

**The rules do not depend on the model obeying them.** A model that ignores every
instruction still cannot publish a bad citation: an unresolvable `[n]` fails
validation, a provision named in prose that appears in neither the supplied
locators nor the supplied text fails validation, any URL fails validation, and a
claim with no marker fails the grounding check. Each failure produces an
abstention, not a repaired answer. Four tests drive exactly these cases through
a fake Groq response.

---

## 5. Request safety

| | |
| --- | --- |
| Timeout | ≤ 15 s, clamped |
| Retries | **0** |
| Requests per invocation | **1** |
| Streaming | off |
| Temperature | 0.0 (`top_p` 1.0) |
| Max completion tokens | 800 |
| Fallback provider | none |
| Automatic model switching | none |

### Error handling

Every failure returns the existing structured abstention — `ProviderDraft` with
empty text — which the engine turns into `provider_abstained` with
`insufficient_evidence: true` and no citations.

| Condition | Code |
| --- | --- |
| `GROQ_API_KEY` unset | `missing_api_key` *(no request is made)* |
| 401 | `unauthorized` |
| 403 | `forbidden` |
| 429 | `rate_limited` |
| 5xx | `server_error` |
| Other non-200 | `http_error` |
| Timeout / socket error | `transport_error`, `unexpected_error` |
| Non-JSON or unexpected shape | `malformed_response` |
| Empty completion | `empty_completion` |

Nothing is retried. A 429 is reported as a 429; retrying would turn one
rate-limit into several. No exception escapes into the engine.

---

## 6. Free-tier limitation

This project runs at **₹0 / $0**, and this provider must not change that.

* Groq's **free plan** applies. It has per-minute and per-day request and token
  caps that are set by Groq, change without notice, and are not documented here
  because a copied number would go stale and be believed.
* **Do not assume unlimited requests.** A 429 is an expected outcome, not a bug,
  which is why it has a stable code and an abstention path rather than a retry.
* No billing is enabled, no payment details are stored, and nothing in the code
  requires a paid tier.
* `max_completion_tokens` is a cost ceiling as much as a style one.
* The system remains fully runnable **with no provider at all**:
  `ExtractiveMockProvider` is offline, deterministic, and still the default. If
  Groq's free tier disappears tomorrow, IP-SAKTI keeps working.

---

## 7. Live smoke test

Run once, in Phase 7.2. The command:

```
.venv/bin/python scripts/answering/smoke_test.py --live
```

`--live` is one of the two paths in the repository that make a network request;
the other is `scripts/evaluation/phase73_five_questions.py --run`. Without the
flag the smoke test constructs no Groq provider at all. With it:

* one tiny fixture passage (`tests/fixtures/answering/citable_statutory.json`),
* the question *"What does Section 3 of the supplied Patents Act evidence say?"*,
* **exactly one** request, ≤15 s, no retry,
* no retrieval, no reranking, no corpus load, no golden dataset, no model
  downloaded and nothing loaded locally,
* the full payload printed **before** it is sent, so what leaves the machine is
  read rather than trusted,
* then the ordinary pipeline: `ProviderRequest → Groq → ProviderDraft →
  GroundedAnswerEngine → citation validation → grounding → Answer → rendered`.

That run returned a completion whose opening sentence carried no citation
marker, so the engine abstained. Phase 7.3 reproduced the same shape on real
corpus evidence (Q027), which is the observation behind prompt rules 4–6.

Everything else is offline, and is what runs by default:

```
.venv/bin/python -m pytest tests/test_groq_provider.py -q     # 49 tests, no sockets
.venv/bin/python scripts/answering/smoke_test.py              # no network at all
```

---

## 8. Limits

* **Answer quality is unmeasured.** Six live requests have been made and two
  produced completions; neither was published, and no completion has been
  scored for legal correctness. Every claim above is about the plumbing.
* **No evaluation.** `benchmark_answers.py` has not been run against Groq, and
  the 111 golden questions remain 0 of 111 human-verified.
* **English only.** No multilingual answering, no translation, no voice — later
  phases.
* **No streaming, no conversation memory, no tool use, no agentic loop.**
* **Not production-ready**, and no answer this produces is legal advice or a
  measurement of legal accuracy.
