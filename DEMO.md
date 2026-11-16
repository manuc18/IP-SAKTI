# IP-SAKTI — prototype demo guide

Everything needed to start the prototype, check it is healthy, and run the
rehearsed demo. Read §6 before presenting.

---

## 1. Prerequisites

Already satisfied on the development machine; listed for a fresh one.

* Python 3.14 with the project virtualenv at `.venv` (`pip install -r requirements.txt`).
* The processed corpus at `data/processed/chunks/` — **3,073 chunks**, committed.
* The BGE-M3 embedding cache at `data/evaluation/embeddings/` — gitignored, and
  regenerated deterministically on first run if absent.
* `BAAI/bge-m3` in the local Hugging Face cache. When it is present the server
  pins `HF_HUB_OFFLINE=1` automatically, so startup never waits on the network.
* `.env` with `GROQ_API_KEY` (gitignored). **Optional** — without it the server
  runs the offline provider instead of failing.

No Tesseract, no corpus PDFs and no re-ingestion are needed to run the demo.

## 2. Start it

```bash
.venv/bin/python -m app.server
```

One command, one process, one port. Then open **http://127.0.0.1:8000**.

Startup takes about **13 seconds**, almost all of it loading BGE-M3, and prints
what it loaded:

```
IP-SAKTI — starting
  corpus              3073 chunks
  lexical index       ready (bm25_boost)
  dense index         3073 vectors (0.1s)
  BGE-M3 warm         12.7s
  reranker            features
  answer provider     groq · openai/gpt-oss-120b
  ready in 13.4s

  serving on http://127.0.0.1:8000   (ctrl-c to stop)
```

The model is warmed during startup on purpose, so the first question a judge
asks is as fast as the fifth.

### Options

| flag | when to use it |
|---|---|
| `--port 8080` | 8000 is taken |
| `--provider extractive` | no network at the venue — see §5 |
| `--no-dense` | 1-second startup, lexical-only retrieval, weaker results |
| `--reranker cross_encoder` | the stronger reranker; **not rehearsed, latency unmeasured** |

## 3. Health check

```bash
.venv/bin/python scripts/demo/verify.py --health
```

```
HEALTH 200
  corpus      3073 chunks, 22 documents
  retrieval   dense=ready lexical=ready reranker=features
  provider    groq · openai/gpt-oss-120b
  startup     13.41s
```

`GET /api/health` returns the same as JSON. `dense=ready` is the line to check:
`unavailable` means the server degraded to lexical-only and retrieval is weaker.

## 3A. Multilingual voice interface

The voice-first hackathon UI runs through the existing Next.js workspace. It
uses Google Cloud Translation on the Next.js server and the browser Web Speech
APIs for microphone input and answer playback. No speech text is sent to a paid
speech service.

Enable the Cloud Translation API for the service account, then set its local
file as Application Default Credentials. Do not copy the JSON into the repo:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
npm install
npm run dev
```

Keep `.venv/bin/python -m app.server` running in the other terminal, then open
**http://localhost:3001**. The Next.js UI uses the existing API gateway at
`http://localhost:5001`, which forwards questions to the Python service at
`http://127.0.0.1:8000`.

Quick demo check:

1. Choose Hindi and speak or type a question; edit the live transcript if
   needed, then submit it.
2. Confirm the explanatory answer is Hindi while citation markers, statute
   names, rule/section identifiers, confidence, source metadata, links and the
   English legal disclaimer are unchanged.
3. Click a citation and expand its source text; confirm the retrieved legal text
   is still the original English text.
4. Click **Listen to answer**, then stop playback.
   The answer heading and button name the playback language. If available,
   choose a pleasant locale-specific voice from **Voice persona** first.
5. Deny microphone access once to confirm the typed-input fallback message.
6. Start without `GOOGLE_APPLICATION_CREDENTIALS` to confirm the UI clearly
   returns to English instead of failing or sending a non-English query to RAG.

Chrome or Edge is recommended because Web Speech recognition support varies by
browser. Browser speech synthesis support and installed voices can also vary by
operating system.

### Deployment credentials

`GOOGLE_APPLICATION_CREDENTIALS` points to a file on one computer, so it is a
local-development setting only. On Vercel or another serverless host, add these
as encrypted/sensitive environment variables for the deployed Next.js service:

```bash
GOOGLE_CLOUD_PROJECT_ID=your-project-id
GOOGLE_CLOUD_CLIENT_EMAIL=translator@your-project.iam.gserviceaccount.com
GOOGLE_CLOUD_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
```

Copy those fields manually from the service-account JSON, do not upload or
commit the file, and never expose them through a `NEXT_PUBLIC_` variable. Apply
them to the intended deployment environments and redeploy. The translation
adapter uses these server-only values in production and otherwise falls back to
Application Default Credentials for local development.

## 4. Rehearse the demo questions

```bash
.venv/bin/python scripts/demo/verify.py
```

Runs all six scenarios through the same HTTP endpoint the browser uses, 25
seconds apart, and prints the answer, citations, cautions and latency for each.
`--delay` controls the spacing; `--out run.json` records the whole run.

**Do this once before presenting.** It confirms the network, the credential and
the pipeline in about three minutes.

## 5. If the network dies

```bash
.venv/bin/python -m app.server --provider extractive
```

The offline provider is deterministic, makes no network request, and is subject
to exactly the same citation and grounding validation. Its answers are
**verbatim quotations of the retrieved provisions**, not prose — it quotes, it
does not synthesise. Retrieval, reranking, citations, cautions and source
inspection all still work, so the evidence-first story survives intact even with
no internet. Use it as a fallback, not as the main demo.

## 6. Known limitations — say these, don't hide them

* **No claim of legal accuracy, and no claim of production readiness.** Nothing
  the system produces is legal advice.
* **0 of 111 benchmark questions are human-verified.** Every retrieval metric in
  `docs/` is agreement with an AI-audited, corpus-derived label.
* **The free Groq tier caps tokens per minute.** Several questions fired
  back-to-back can exhaust it. When that happens the UI shows **"Answer
  unavailable"** with the provider error — deliberately *not* the safe-abstention
  screen, because a rate limit is a service problem and abstention is a claim
  about evidence. Wait about a minute and ask again. There is no retry, by design.
* **The corpus is fixed and partly dated.** 39 of 111 benchmark targets sit in
  instruments marked `STALE` (Patents Rules 2003, Copyright Act 1957, TRIPS and
  others). The UI marks those sources `STALE` and attaches a caution.
* **Retrieval is English-only.** Hindi text is excluded from retrieval because
  extraction produces silently-wrong Unicode.
* **Answers are 1–4 sentences by contract**, and every sentence must carry a
  citation marker. Long expository answers are out of scope.

## 7. Demo questions — rehearsed and working

| # | Question | Shows |
|---|---|---|
| 1 | How must an application for registration of a trade mark be signed? | Clean single-provision answer, one validated citation (TM Rules 2017, Rule 23) |
| 2 | Is the mere discovery of a new form of a known substance patentable in India? | Second domain, and the answer cites **`[4]`** — the engine used the rank-4 passage, not rank 1 |
| 3 | What is the duration of the registration of a trade mark and when must it be renewed? | Multi-sentence answer, **two citations across an Act and its Rules**, marker on every sentence |
| 4 | What is the term of copyright in a literary work published in the lifetime of the author? | `STALE` source + provenance level C → **two cautions**, confidence drops to LOW |
| 5 | Who may apply for the registration of a geographical indication? | Corpus breadth — a fifth IP domain |
| 6 | What is the best pizza topping in Naples? | **Safe abstention**: `no_relevant_evidence`, `no_citable_authority` — nothing is invented |

Ask these. Do not improvise on stage: an unrehearsed question can hit an
instrument the corpus does not cover, and the honest result is an abstention.

## 8. What to say while it runs

1. **Ask question 1.** Point at the `[1]` marker — click it, the source card
   highlights and opens. "Every sentence is traceable to a provision."
2. **Ask question 2.** Point at `[4]`. "Retrieval put a different passage first.
   The answer engine cited the one that actually supports the claim."
3. **Ask question 4.** Point at the `STALE` badge and the cautions. "The system
   knows this text is a dated consolidation and says so rather than hiding it."
4. **Ask question 6.** "This is the part that matters. A generic chatbot answers
   this. IP-SAKTI retrieved evidence, found nothing in the corpus that supports
   an answer, and refused to publish one."

The line to land: **the answer must survive citation and grounding validation
against retrieved legal evidence, or it is not published.**

## 9. API, if anyone asks

| endpoint | |
|---|---|
| `GET /api/health` | corpus, index, provider and version status |
| `POST /api/ask` | `{"question": "..."}` → answer, citations, evidence, timings |

`POST /api/ask` returns one of three verdicts: `ANSWERED`, `ABSTAINED`
(the evidence did not support an answer) or `UNAVAILABLE` (the provider could
not be reached). The answer body is `Answer.to_dict()` from
`ai/answering/models.py` — the shape Phase 7 §24 specified for this endpoint,
unchanged.

## 10. Architecture, in one screen

```
Question
  → CandidateGenerator      BGE-M3 dense top-50 ∪ BM25+identifier-boost top-50
  → FeatureReranker         deterministic, ~20 ms → top 5
  → EvidenceSet             relevant · authoritative · current · citable, kept apart
  → GroundedAnswerEngine    one provider call, no retries, no fallback chain
  → validate_citations      a marker with no passage behind it fails the answer
  → check_grounding         a sentence with no citation fails the answer
  → published, or abstained
```

`app/` adds a server and a page. It contains no retrieval logic, no policy and
no validation of its own — those live in `ai/` and were not modified to build
this prototype.
