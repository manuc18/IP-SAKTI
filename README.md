# IP-SAKTI

A citation-grounded question-answering system over Indian intellectual-property
and biodiversity law. Every answer is generated only from retrieved statutory
text, and no sentence is published unless a validator can tie it back to a
provision in the corpus.

**Status: research pipeline, phases 1–8 complete. No server, no UI, no
human-verified benchmark.** Nothing here is legal advice, and no number in this
repository is a measurement of legal accuracy. See `PROJECT.md` for what is
built, what is measured and what remains.

---

## 1. What it does today

```
22 logical documents (21 PDFs, 1,782 pages)
  → ingestion        extract · OCR · clean · structure · chunk · validate  → 3,073 chunks
  → provenance       publisher-of-record verification, evidence levels A–D
  → retrieval        BGE-M3 dense ∪ BM25 + identifier boost, deterministic router
  → reranking        bge-reranker-v2-m3 over the fused pool → top 10
  → evidence policy  relevant · authoritative · current · citable, kept apart
  → answering        prose with [n] markers, from evidence only
  → validation       citation validator + grounding gate; a failure is an abstention
```

Everything from the evidence policy down loads no model and opens no network
connection. The default answer provider is deterministic and offline; the Groq
provider is constructed only when a caller asks for it explicitly.

---

## 2. Quick start

Requires Python 3.14 (developed on 3.14.3) and, for ingestion only, Tesseract 5
with the `eng` and `hin` language data.

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# ingestion only — not needed to run retrieval, reranking or answering
brew install tesseract tesseract-lang          # macOS
# apt-get install tesseract-ocr tesseract-ocr-eng tesseract-ocr-hin   # Debian
```

Run the test suite (699 tests; the retrieval and reranking tests load models and
are the slow ones):

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m pytest tests/test_answering.py -q     # offline, ~0.15 s
```

The fastest end-to-end proof that the answer path works, with no model load and
no network:

```bash
.venv/bin/python scripts/answering/smoke_test.py
```

### The corpus

The source PDFs are **not** in this repository. Phase 2 reads them read-only
from `~/Downloads/IP-SAKTI Corpus`, and `scripts/ingest/extract.py` re-hashes
every file against the digest recorded in `data/metadata/corpus_manifest.json`,
aborting if one has changed. To stage the corpus locally instead, put the files
under `data/raw/` named `<sha256>.pdf` and point `IPSAKTI_CORPUS_ROOT` at that
directory. See `data/raw/README.md`.

### Secrets

`.env` is gitignored and holds `GROQ_API_KEY`, `GROQ_MODEL` and
`GROQ_TIMEOUT_SECONDS`. Only `ai/answering/groq.py` reads it. No other file in
the project opens a network connection during answering.

### Multilingual voice demo

The Next.js demo adds browser speech input/output and a server-only Google Cloud
Translation adapter around the existing English RAG call. It does not modify
the RAG result: only the explanatory answer text is translated for display;
citations, statute names, legal identifiers, source text, URLs, confidence,
jurisdiction and the disclaimer remain as returned.

Set Application Default Credentials in the shell that starts Next.js. Keep the
JSON outside this repository:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=/absolute/path/to/service-account.json
```

For the full local interface, run the existing Python service and the web
workspace in separate terminals:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
npm install

.venv/bin/python -m app.server --provider extractive --no-dense
npm run dev
```

Open **http://localhost:3001**. The Python demo/API remains available at
**http://127.0.0.1:8000**. Chrome or Edge is recommended for browser speech
recognition. If Google Translation credentials or the API are unavailable, the
interface stays usable in English and shows a non-technical status message.
The answer heading and playback button show the language actually displayed.
Speech playback automatically prefers a locale-matching, female-associated
voice when the browser provides one; the **Voice persona** menu lets the user
choose any other installed voice for that language.

For a deployed Next.js app, the local JSON file does not travel with the code.
Configure these values as encrypted/sensitive, server-only environment
variables in the hosting dashboard, then redeploy:

```bash
GOOGLE_CLOUD_PROJECT_ID=your-project-id
GOOGLE_CLOUD_CLIENT_EMAIL=translator@your-project.iam.gserviceaccount.com
GOOGLE_CLOUD_PRIVATE_KEY="-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n"
```

Copy the three values yourself from the service-account JSON without adding the
file to the repository. Do not use a `NEXT_PUBLIC_` prefix. Locally,
`GOOGLE_APPLICATION_CREDENTIALS` remains the recommended option; if the three
deployment variables are absent, the adapter falls back to Application Default
Credentials.

---

## 3. Repository layout

```
ai/                          the library. No scripts, no I/O beyond loading indexes.
  retrieval/
    base.py                  Retriever interface, Filters, RETRIEVAL_VERSION
    bm25.py  dense.py        lexical (BM25) and dense (BGE-M3) retrievers
    hybrid.py                RRF fusion
    identifiers.py           parse_identifiers() — decides IDENTIFIER vs CONCEPTUAL
    router.py                deterministic query router (Phase 5)
    candidates.py            expanded / routed candidate pools (Phase 6)
    tokenizer.py
  reranking/
    base.py                  Reranker interface
    cross_encoder.py         bge-reranker-v2-m3 and the two smaller baselines
    features.py              deterministic feature reranker (the cheap fallback)
    format.py
  answering/                 loads no model; a test asserts torch is absent after import
    models.py                Evidence, Answer, ANSWERING_VERSION
    policy.py                evidence policy — relevant/authoritative/current/citable
    providers.py             ANSWER_INSTRUCTIONS + PROMPT_VERSION + extractive provider
    groq.py                  the only file that opens a socket while answering
    citation_validator.py    no passport, no entry
    grounding.py             every claim must carry a source
    render.py                Answer object → text, SYSTEM_DISCLAIMER

scripts/                     everything that runs, reads or writes
  ingest/                    Phase 2: extract, ocr, clean, structure, chunk, validate
    run_pipeline.py          the entry point; stages are independent and idempotent
    parsers/                 statutory / treaty structural parsers
  provenance/                Phase 3: build_registry.py, verify_source.py
  evaluation/                Phases 4–8: benchmarks, error analysis, the AI judge
  answering/smoke_test.py    Phase 7 end-to-end, offline in ~0.1 s

data/
  raw/                       intentionally empty (gitignored) — see its README
  source_updates/            content-addressed staging for acquired publisher copies
  processed/                 extracted → cleaned → structured → chunks (pipeline output)
  metadata/                  corpus_manifest, provenance_registry, source_verifications
  evaluation/
    golden_questions.jsonl   111 questions, 0 human-verified
    ai_judged_ground_truth.jsonl   111 AI judgements (Phase 8.3)
    ai_judge_packets.jsonl   the evidence each judgement was made from
    ai_judge_manifest.json   frozen provenance manifest — packet hashes, corpus fingerprint
    embeddings/              BGE-M3 cache (gitignored, regenerated deterministically)
    results/                 every benchmark run, by phase

app/                         the runnable prototype. No retrieval or policy logic.
  pipeline.py                builds the engine once; QUESTION → ANSWER in one call
  server.py                  stdlib demo server + the single-page UI (app/static/)
  api_server.py              Flask HTTP API — POST /api/v1/query, GET /health
  schemas.py                 request validation and response serialisation
  config.py                  every environment variable, in one dataclass

docs/                        one document per phase; see §6
tests/                       699 tests; fixtures under tests/fixtures/
```

---

## 4. Running each layer

```bash
# Phase 2 — ingestion (needs Tesseract and the corpus)
.venv/bin/python scripts/ingest/run_pipeline.py --all

# Phase 3 — provenance registry
.venv/bin/python scripts/provenance/build_registry.py

# Phase 4/5 — retrieval benchmarks
.venv/bin/python scripts/evaluation/benchmark_retrieval.py --retriever all
.venv/bin/python scripts/evaluation/benchmark_routing.py

# Phase 6 — reranking
.venv/bin/python scripts/evaluation/benchmark_reranking.py \
    --candidate-mode expanded --reranker cross_encoder --cross-encoder bge-m3

# Phase 7 — answering
.venv/bin/python scripts/answering/smoke_test.py               # offline
.venv/bin/python scripts/evaluation/benchmark_answers.py       # fixtures only
.venv/bin/python scripts/answering/smoke_test.py --live        # ONE Groq request

# Phase 8 — AI ground-truth judge (dry run is the default)
.venv/bin/python scripts/evaluation/prepare_ai_judgement.py
.venv/bin/python scripts/evaluation/run_ai_judge.py --question Q027   # dry run
```

Every script that can reach the network requires an explicit flag to do so
(`--live`, `--run`, `--question`). `run_ai_judge.py` defaults `--max-requests`
to 1: spending 111 requests requires asking for 111.

---

## 5. House rules worth knowing before you change anything

These are conventions the existing code and docs already enforce; breaking one
will fail a test or invalidate a recorded result.

* **Every component carries a version constant** — `RETRIEVAL_VERSION`,
  `ANSWERING_VERSION`, `POLICY_VERSION`, `VALIDATOR_VERSION`,
  `GROUNDING_VERSION`, `PROMPT_VERSION`. A result recorded under an unversioned
  prompt or model revision is not comparable with anything.
* **Prompts live in documents, not in strings.** The judge prompt is the block
  between the markers in `docs/AI_GROUND_TRUTH_JUDGE_PROMPT.md`; the runner
  extracts those bytes, keeps no copy, and records their SHA-256 on every
  judgement.
* **Frozen artifacts are read-only.** `docs/PHASE7_FREEZE.md` and the Phase 7
  result files under `data/evaluation/results/answering/` are not edited. A
  later phase that disagrees writes a new file and says why.
* **The corpus is immutable.** Nothing renames, replaces or deletes a file in
  the corpus directory. Acquired replacements land in `data/source_updates/`
  and supersede a source only after an explicit approval step.
* **Never `verified`.** AI judgements use `AI_VERIFIED`, `AI_CORRECTED`,
  `AI_REJECTED`, `AI_NEEDS_REVIEW`. A test asserts no stored judgement record
  contains a field called `verified`, because human verification is a different
  measurement and it stands at 0 of 111.
* **A failure is an abstention.** There is no retry, no regeneration and no
  fallback provider in the answer path. An answer that fails citation
  validation or grounding is not published.
* **Benchmarks cannot see their own answers.** A retriever receives a
  `BenchmarkQuery` built before the expected answer is read; `Filters` has no
  `document_id` field, which is how leakage is prevented structurally rather
  than by care.

---

## 6. Documentation index

| document | phase | what it records |
|---|---|---|
| `docs/CORPUS_AUDIT.md` | 1 | 21 PDFs audited; OCR need, Hindi extraction risk |
| `docs/INGESTION_VALIDATION.md` | 2 | 3,073 chunks, OCR confidence, validation results |
| `docs/PROVENANCE_REPORT.md` | 3 | publisher verification, evidence levels A–D |
| `docs/RETRIEVAL_BENCHMARK.md` | 4 | BM25 vs BGE-M3 vs fusion on 111 questions |
| `docs/RETRIEVAL_ROUTING.md` | 5 | the deterministic router; locator repairs |
| `docs/RERANKING_BENCHMARK.md` | 6 | three cross-encoders; +0.162 Recall@1 |
| `docs/ANSWER_ENGINE.md` | 7 | evidence policy, citation validation, grounding |
| `docs/ANSWER_GENERATION_CONTRACT.md` | 7 | the ten rules a provider must satisfy |
| `docs/ANSWER_PROVIDER.md` | 7.2 | Groq provider, every live request accounted for |
| `docs/PHASE7_FREEZE.md` | 8 | frozen record of the answer layer (read-only) |
| `docs/GROUND_TRUTH_VERIFICATION.md` | 8 | the human review workflow, unused |
| `docs/AI_GROUND_TRUTH_JUDGING.md` | 8.1–8.3 | 111/111 AI-judged; anti-circularity design |
| `docs/AI_GROUND_TRUTH_JUDGE_PROMPT.md` | 8.1 | the judge prompt itself, v1.0.0 |
| `DEMO.md` | — | how to start the prototype, and the rehearsed demo questions |
| `docs/API_HANDOVER.md` | — | the HTTP API contract the frontend integrates against |
| `PROJECT.md` | — | implemented vs remaining, against the project's aim |

---

## 7. Limits

* **0 of 111 golden questions have been verified by a human.** Every recall, MRR
  and nDCG figure in this repository is agreement with an unverified label.
* **No claim of legal accuracy. No claim of production readiness.** Citation
  validity and grounding are structural properties, not statements about whether
  an answer is right.
* answers are not legal advice.
