# IP-SAKTI — project status

What has been built, what it measures, and what stands between here and the
system this is meant to become. Companion to `README.md`, which covers setup and
layout.

Written 2026-09-07, at `60c4d2c` on branch `rag`.
Updated 2026-09-16 for the final quality sprint — §2, §3 (Phase 9) and §5.3 —
and again for the UX and data-integrity sprint (Phase 10): §2, §3, §5.7 and a
new §5.8 recording what the source PDFs themselves are missing.
`docs/METRICS_FULL_EVIDENCE.md` remains the single source of truth for every
number, with the full working behind each one. `docs/SIH_METRICS_FINAL.md` is
the one-page extract for the SIH panel and `docs/SUBMISSION_METRICS_FINAL.md`
the compact submission table; neither introduces a figure the evidence document
does not carry.

---

## 1. The aim

**Assumption, stated because it is nowhere written down as a single sentence.**
The aim below is reconstructed from what the phase documents commit to and from
what they explicitly record as not built. If it is wrong, this section is the
one to correct first — everything in §5 is scoped against it.

> A citation-grounded question-answering system over Indian intellectual-property
> and biodiversity law, usable in English and Hindi, that answers only from
> verifiable statutory text, refuses to answer when the evidence does not support
> one, and runs at zero operating cost.

Four commitments follow from that, and each is visible in the code:

* **Grounded or silent.** No sentence is published unless a validator can tie it
  to a supplied provision. A failure is an abstention, not a retry.
* **Provenance is a first-class property.** Every chunk carries its source hash,
  page, locator, temporal status, evidence level and citability, and the answer
  layer keeps *relevant*, *authoritative*, *current* and *citable* apart rather
  than collapsing them into one score.
* **₹0 operating cost.** No vector database, no paid API on the default path.
  The dense index is a `.npy` matrix with a JSON sidecar; the LLM provider is
  optional and free-tier.
* **Nothing is claimed that is not measured.** Statuses are `PROVISIONAL` until
  earned, frozen records are not edited, and human verification is reported
  separately from AI audit.

---

## 2. Where the project stands

| Phase | Scope | Status |
|---|---|---|
| 1 | Corpus audit | **Complete** — 21 PDFs, 1,782 pages, manifest written |
| 2 | Ingestion and structural preprocessing | **Complete** — 3,073 chunks, 0 validation failures |
| 3 | Source provenance | **Complete** — 22 documents registered, evidence levels A–D |
| 4 | Retrieval laboratory | **Complete** — BM25 / BGE-M3 / RRF benchmarked |
| 5 | Query routing | **Complete** — router beats both retrievers at rank 1 |
| 6 | Reranking | **Provisional** — `bge-m3` wins; F3 and latency unmeasured |
| 7 | Grounded answer engine | **Provisional** — 2 answers published, of 9 live requests |
| 8 | Ground-truth validation | **AI-complete** — 111/111 AI-judged, 0/111 human-verified |
| 9 | Final quality sprint | **Complete** — R@1 0.802 → 0.847, R@10 1.000, 1,000-question synthetic benchmark, bounded clarification, excerpt-first citations |
| 10 | UX and data integrity | **Complete** — social-intent gate, two-axis confidence, corpus-grounded spelling repair, extraction-damage detection; retrieval unchanged |

---

## 3. What is implemented

### Phase 1 — corpus audit
`docs/CORPUS_AUDIT.md`, `data/metadata/corpus_manifest.json`.

21 PDFs / 1,782 pages / 47.4 MB, 4.37M characters recovered without OCR. Three
findings drove everything downstream: the FSSAI Ayurveda Aahara regulations are
170 of 172 pages of page images; the Patents Rules lose their Forms from page 56
on; and Hindi extraction is *silently wrong* in every bilingual document — valid
Unicode from defective font mappings, which passes every automated check while
being unreadable.

### Phase 2 — ingestion
`scripts/ingest/`, `docs/INGESTION_VALIDATION.md`.

Six independent, idempotent stages: extract → OCR → clean → structure → chunk →
validate. 22 logical documents, 1,565 pages extracted natively and 217 OCR'd
(`eng+hin`, mean confidence 89.6 / 81.9), 3,418 structural units, **3,073
chunks**. 4,847 Hindi lines excluded from retrieval text rather than embedded
wrong. Zero validation failures: every chunk resolves chunk → page → source PDF,
every source still hashes to its Phase 1 digest, no `(cid:N)` artefacts, no
U+FFFD, no Devanagari in a retrieval chunk.

### Phase 3 — provenance
`scripts/provenance/`, `docs/PROVENANCE_REPORT.md`.

Sixteen publisher-of-record copies were fetched and compared page by page on
three axes that survive re-encoding: text layer, page geometry, and the SHA-256
of embedded image XObjects. Evidence levels A–D describe *source provenance
only*. The central finding: the corpus appears complete and is not — one file
occupies the slot of a document it does not contain, and that document is
missing.

### Phase 4 — retrieval
`ai/retrieval/`, `docs/RETRIEVAL_BENCHMARK.md`.

BM25, BGE-M3 dense (a `.npy` matrix, no vector store), RRF hybrid, and an
identifier boost, against 111 golden questions. The finding that shaped Phase 5:
the two families fail in opposite directions — dense owns conceptual queries,
lexical + identifier boost owns cited provisions, and every rank-only fusion
lands between them.

### Phase 5 — routing
`ai/retrieval/router.py`, `docs/RETRIEVAL_ROUTING.md`.

A deterministic router — a pure function of the query string, nothing learned,
nothing tuned — sends identifier queries to the boosted lexical retriever and
everything else to BGE-M3. Also repaired two classes of wrong citation locator,
rebuilding the corpus from 3,031 to 3,073 chunks (Phase 4 numbers are therefore
not comparable with Phase 5's).

| Retriever | Recall@1 | Recall@10 | MRR |
|---|---:|---:|---:|
| bm25 | 0.342 | 0.820 | 0.496 |
| dense (BGE-M3) | 0.712 | 0.964 | 0.807 |
| hybrid_boost (RRF) | 0.631 | **0.982** | 0.755 |
| **routed** | **0.766** | 0.964 | **0.843** |

Median latency 31 ms routed. 0 fallbacks taken over 111 queries.

### Phase 6 — reranking
`ai/reranking/`, `docs/RERANKING_BENCHMARK.md`.

Three cross-encoders and a deterministic feature reranker over an expanded
50 + 50 candidate pool (mean 77.3 candidates, candidate recall 1.000).

| Configuration | R@1 | R@10 | MRR |
|---|---:|---:|---:|
| expanded + identity (no reranking) | 0.712 | 0.964 | 0.818 |
| expanded + features | 0.784 | 0.973 | 0.852 |
| expanded + `bge-reranker-base` | 0.766 | 0.991 | 0.842 |
| **expanded + `bge-reranker-v2-m3`** | **0.874** | **0.991** | **0.926** |

**+0.162 Recall@1 from reranking alone** on an identical candidate set, and
+0.108 over Phase 5's router. Model size mattered more than family: MiniLM (22M)
barely moved anything, 278M matched the router, only 568M produced the step
change.

**Shipped configuration, after the operative-text demotion**
(`features.phase7-operative.summary.json`; candidate generation unchanged):

| Configuration | R@1 | R@5 | R@10 | MRR |
|---|---:|---:|---:|---:|
| expanded + features (Phase 6) | 0.784 | 0.946 | 0.973 | 0.852 |
| **expanded + features + operative-text demotion** | **0.802** | **0.964** | 0.973 | **0.863** |

A title page, a chapter heading and a preamble carry no operative rule, so they
are demoted by a bounded 0.25 of the relevance range rather than filtered out.
Four questions improved, none regressed. Identifier queries reach R@1 1.000.

### Phase 7 — grounded answer engine
`ai/answering/`, `docs/ANSWER_ENGINE.md`, `docs/ANSWER_GENERATION_CONTRACT.md`,
`docs/ANSWER_PROVIDER.md`.

Reranked evidence → evidence policy → generation → citation validation →
grounding → rendered answer. The package imports no retriever, no index and no
tensor library; it loads in 0.075 s and a test asserts `torch`,
`transformers` and `sentence-transformers` are absent from `sys.modules` after
importing it.

* **Evidence policy** keeps relevance, authority, currency and citability apart
  as four separate properties.
* **Citation validation** — a marker with no supplied passage behind it, or a
  provision named in prose that appears in neither the supplied locators nor the
  supplied text, refuses the answer.
* **Grounding** requires every sentence to carry a marker and to be supported by
  the passage it cites.
* **Conflict detection** is four deterministic detectors over the *same*
  provision, never merely different ones.
* **Cautions are earned**, not boilerplate: one system-level disclaimer, plus a
  per-answer caution only when the evidence produced one.

Two providers: an offline deterministic extractive provider (the default), and
`GroqProvider` over `openai/gpt-oss-120b` on stdlib `urllib`, no SDK, no new
dependency, no retries. **Nine live requests have been made in the project's
history and all nine are itemised in `docs/PHASE7_FREEZE.md`** — three were 429
rate limits, four were refused by the citation or grounding gate, and **two were
published**, both under prompt `0.3.0`, both validation PASS with grounding 1.0.

### Phase 8 — ground-truth validation
`docs/GROUND_TRUTH_VERIFICATION.md`, `docs/AI_GROUND_TRUTH_JUDGING.md`,
`docs/AI_GROUND_TRUTH_JUDGE_PROMPT.md`.

The human-review workflow was built, is correct, and was unusable: verifying a
question means deciding whether a provision is the right law for it, which needs
a lawyer this project does not have. Rather than leave the benchmark at 0 of 111
indefinitely, Phase 8 added an AI judge asking a deliberately different
question — not "is this the correct answer?" but "does the supplied evidence
justify the benchmark's *designation* of this document, locator and chunk as
ground truth?"

| judge status | questions |
|---|---:|
| `AI_VERIFIED` | 105 |
| `AI_NEEDS_REVIEW` | 6 |
| `AI_CORRECTED` / `AI_REJECTED` | 0 |
| **total** | **111** |

Anti-circularity is implemented rather than asserted: different question,
different model and provider from the answer generator, different prompt,
no import edge (asserted by parsing ASTs, not grepping), and no retrieval output
in the evidence packets. The `verified` flag, `ground_truth_method` and the
label author's own `notes` are withheld from the judge, each asserted absent from
all 111 packets.  Packets are byte-stable and hash-pinned in a frozen manifest.
**Network requests across all of Phase 8: 0.**

### Phase 10 — UX and data integrity
Four deterministic layers, none of which touches retrieval, reranking, citation
validation or the grounding gate, and each of which is off by construction when
its condition does not hold.

* `app/intent.py` — a closed table of 92 conversational phrases, matched against
  the whole input and only when it anchors on no regime, document or provision.
  A greeting returns `status: SOCIAL` with no answer and no evidence, so it is
  counted in no answer rate. 0/111 benchmark questions are captured.
* `app/spelling.py` — the corpus is the dictionary. Only out-of-vocabulary words
  of four or more letters are examined and only runs of ASCII letters at all, so
  identifiers are never offered to it. Matching is on a silent-letter skeleton
  with Damerau distance 1. Retrieval sees the corrected string; the generator and
  the response see the question as typed. **0/111 clean benchmark questions
  altered.**
* `ai/answering/confidence.py` — the single ordinal label split into
  `source_confidence` and `support_confidence`, published as the lower of the
  two with every contributing factor named. Nothing was promoted: level B and
  STALE still cannot reach HIGH, asserted exhaustively.
* `ai/text_integrity.py` — detection only. 211/3,073 chunks carry PDF-extraction
  damage from two causes that are both upstream of this repository, and a damaged
  passage is shown verbatim behind a disclosure rather than repaired.

### Cross-cutting
1,017 tests. Fixture-driven offline paths for every layer that can reach a model
or a network. `docs/PHASE7_FREEZE.md` freezes the Phase 7 record so no later
phase can improve its own numbers by editing an earlier one.

---

## 4. What the numbers do and do not mean

Read this before quoting anything above.

* **0 of 111 golden questions are human-verified.** Every recall, MRR and nDCG
  figure measures agreement with a corpus-derived label. Whether the label is
  right is exactly what has never been checked by a person.
* **AI-audited ≠ verified.** 111/111 `AI_*` statuses narrow the unknown; they do
  not establish legal accuracy, and the judge shares the corpus's blind spots —
  an extraction error is invisible to an auditor whose only evidence is that
  extraction.
* **Citation validity and grounding are structural.** They say a sentence is
  traceable to a supplied passage. They say nothing about whether it is correct
  law.
* **Two published answers is two observations, not a rate.** Both questions were
  chosen for structural properties, neither is a random draw from the 111.
* **39 of 111 targets are STALE** — whole instruments (Patents Rules 2003, GI
  Rules 2002, Designs Rules 2001, Copyright Act 1957, D&C Act 1940 and Rules
  1945, TRIPS). That is a corpus-vintage property, recorded rather than treated
  as a defect.

---

## 5. What is left

Ordered by what unblocks the most. Items 1–3 are correctness; 4–7 are the gap
between a pipeline and a system.

### 1. Verification of the benchmark — blocks every number downstream
Nothing else on this list is worth doing first. 111 questions need a qualified
reader; `scripts/evaluation/review_ground_truth.py` and
`review_reranking.py` already implement the workflow, and the 111 AI judgements
are a triage order for it. Priority within it: the 6 `AI_NEEDS_REVIEW`
questions, then the 6 identity and 2 `bge-m3` reranker failures (a benchmark
whose winner misses two questions is decided by whether those two labels are
right), then the rest.

### 2. An end-to-end answer evaluation that has actually been run
`scripts/evaluation/benchmark_answers.py` exists and implements the framework.
**It has never been run over the 111 golden questions** — only over fixtures.
Until it is, the answer layer's quality is two observations. Running it needs a
rate-limit strategy: the free tier's TPM cap silently converted three Phase 7.3
questions into abstentions, and the layer has no retry by design.

### 3. The measurements that were cut
* **F3** — `bge-m3` with the authority policy. The run was killed ten minutes in
  and wrote no output; the table row is empty rather than guessed.
* ~~**Reranker latency.** Unmeasured~~ — **resolved 2026-09-16.** It was in the
  Phase 6 artifact the whole time: the 568M cross-encoder reranks at a **median
  of 14,145 ms and a p95 of 98,203 ms**, against **49 ms / 78 ms** for the
  feature reranker. It buys +2.7 pp Recall@1 and *loses* at Recall@5 and
  Recall@10. The feature reranker is now the evidence-backed choice rather than
  the convenient one. See `docs/METRICS_FULL_EVIDENCE.md` §7.4.
* **Mode A × cross-encoder** — "expanded beats routed" is measured for identity
  and features, assumed for the cross-encoder.
* **The 512-token truncation.** 35.8% of scored pairs were truncated and both
  remaining `bge-m3` failures are on chunks over the limit. A longer window is
  the first accuracy experiment to run, ahead of any new model.

### 4. Corpus gaps
* The missing document (Phase 3 §2) is still missing, and the file occupying its
  slot is still the wrong document.
* The Patents Rules copy is consolidated only to 23 June 2017.
* Chunks that begin or end mid-sentence; section 2 of the Biological Diversity
  Act reduced to a single 308-character clause; amendment footnotes stored as
  retrievable chunks; OCR damage in headings (`RESERV ATIONS`, `tradem ark`);
  marginal notes interleaved into PPVFR Act text.
* Publisher copies staged in `data/source_updates/` have not superseded anything
  — that requires an explicit approval step and a fresh ingestion run.

### 5. Benchmark coverage
Every golden question has exactly one expected chunk, and real legal questions
frequently have more than one correct provision; R@1 against a single-chunk
label punishes a system that surfaces an equally valid sibling. Also needed:
more than 18 identifier questions (one question is worth 0.056 there, and it
carries the whole feature-reranker argument), non-English queries (two of the
three shortlisted cross-encoders are multilingual and none was tested that way),
and questions whose correct answer is non-citable or Level B/C evidence.

### 6. The Hindi half of the aim
The corpus is bilingual; the system is not. Hindi lines are excluded from
retrieval text because extraction produces silently-wrong Unicode, no query has
been run in Hindi, and there is no multilingual generation. Closing this needs a
different Hindi text path — re-OCR or a corrected font mapping — before anything
downstream can use it.

### 7. Everything user-facing
~~No server, no API, no UI~~ — a stdlib server, a Flask API and a Next.js UI now
exist (`app/server.py`, `app/api_server.py`, `apps/web/`). Still absent: voice,
and any Hindi path. As of the final sprint every cited source shows a verbatim
excerpt rather than the whole 2–3 page chunk, and a question naming no body of
law gets one deterministic clarifying question instead of a shrug.

**Removed in the final sprint:** `apps/server/src/services/pipelineService.js`
carried a table of hardcoded answers with fabricated citations and
`grounded: true`, served whenever the Python backend was unreachable. They had
been through no retrieval, no citation validation and no grounding gate, and
were indistinguishable from real output. An unreachable backend now returns
`UNAVAILABLE`.

**Fixed in the UX sprint:** the same file dropped the `clarified` flag on its way
to the Python backend. The web app sends it and `app/pipeline.py` reads it, and
it is the entire loop guard on the one clarifying question the system may ask —
so a user who answered the clarification was asked it again.

### 8. What the source PDFs themselves are missing
**211 of 3,073 chunks (6.9 %) did not survive PDF extraction**, and the damage is
upstream of every line of code here — the corrupted passage is byte-identical at
extraction, cleaning, structuring and chunking. Two causes:

* `2016DrugsandCosmeticsAct1940Rules1945.pdf` and `Trademarks rules 2017.pdf`
  position glyphs individually — 1,604 text-positioning operators for 3,481
  characters on one page — so any extractor must guess word boundaries from
  horizontal gaps. poppler's `pdftotext` reproduces the breakage exactly, so
  **changing extraction library will not fix this.**
* The Second Schedule forms of the Patents Rules, 2003 are image scans whose
  dotted fill lines OCR into runs of `e`, `c` and `o`.

`ai/text_integrity.py` detects both and repairs neither. The real fix is a
cleaner source copy for two PDFs, which is a corpus acquisition task and not an
engineering one. Until then the affected passages are disclosed as damaged.

### Also unresolved
* **Rate limiting is unmodelled.** Any future batch run hits the TPM cap again.
* **The relevance screen is lexical** — it cannot tell that a passage is about
  the right regime but the wrong question.
* **Contradiction detection is partial** — two provisions that contradict in
  substance, sharing no locator and announcing no override, are invisible.
* **Prompt 0.3.0's sentence-marker discipline is untested beyond one sentence.**
  Both published answers were a single sentence; whether a three- or
  four-sentence answer still carries a marker on every sentence is unknown.

---

## 6. Standing statement

**No claim of legal accuracy. No claim of production readiness.** No answer this
system produces is legal advice.
