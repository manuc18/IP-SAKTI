# Phase 7.3 — five-question end-to-end test

Status:

`ANSWER_E2E_STATUS: PARTIAL`

`EXPERIMENT_MODE: LEXICAL_ONLY`

> **Not legal accuracy.** Everything below is an engineering property of the
> pipeline. Nothing here evaluates whether an answer is legally correct, and
> this system is not production-ready.

> **Retrieval caveat.** The dense (BGE-M3) arm was disabled for machine
> safety, so candidate generation was BM25 + identifier-boost only. The
> retrieval figures are `lexical_candidate_recall` and
> `reranked_evidence_recall`; they are **not** IP-SAKTI's overall retrieval
> recall and are **not** comparable with the Phase 6 hybrid results.

Model `openai/gpt-oss-120b` · temperature 0.0 · max_completion_tokens 800 · timeout 15.0s · retries 0 · reranker `features` · top_k 5

---

## Q001 — simple_statutory_provision

**Query:**  
What does section 3(p) of the Patents Act, 1970 exclude from patentability?

*Dataset family:* `identifier` · *domain:* `patents` · *jurisdiction:* `india`  
*Selected because:* first identifier question in file order; a single-provision Section lookup

**Retrieved / reranked:** 50 lexical candidates → 5 evidence passages (fingerprint `43209b458db575c4`)

| # | citation | citable | temporal | level |
|---|---|---|---|---|
| [1] | Patents Act, 1970 — Section 3 | True | CURRENT | B |
| [2] | Geographical Indications of Goods (Registration and Protection) Act, 1999 — Section 3 | True | CURRENT | B |
| [3] | Designs Act, 2000 — Section 3 | True | CURRENT | B |
| [4] | Trade Marks Act, 1999 — Section 3 | True | CURRENT | B |
| [5] | Biological Diversity Act, 2002 — Section 3 | True | CURRENT | C |

**Provider status:** completion returned.

**Answer (published):**

```
I couldn't find sufficient authoritative evidence in the current corpus to answer this reliably.
```

**Rejected draft** — diagnostic only, never published as an answer:

```
Section 3(p) of the Patents Act, 1970 bars from patentability any invention that is, in effect, **traditional knowledge** or that merely **aggregates or duplicates the known properties of components that are already traditionally known**【1】.
```

**Citations:** 0 — FAILED

- validation failure `unsupported_locator_in_prose`: the answer names section 3(P), which is neither a supplied provision nor named by any supplied passage

**Grounding:** FAIL

**Confidence:** NONE

**Golden target:** IN-PATENTS-ACT-1970 — Section 3  
`lexical_candidate_recall` True (rank 1) · `reranked_evidence_recall` True (rank 1)

**Assessment: NEEDS_REVIEW**  
The model reached the right evidence and wrote a substantively on-evidence answer, then failed the citation contract twice: it wrote the marker as full-width `【1】` instead of `[1]`, so no marker was extractable, and it cited `section 3(p)` — a subdivision of the supplied Section 3 chunk — which the validator rejected as an unsupported locator. The engine refused, which is the correct outcome given zero resolvable citations, but part of the refusal is our own false positive (see failure pattern F2).

---

## Q078 — conceptual_synthesis

**Query:**  
What rights does a breeder get on registration of a plant variety?

*Dataset family:* `conceptual` · *domain:* `plant_varieties` · *jurisdiction:* `india`  
*Selected because:* the only Indian conceptual question whose golden target spans two chunks, so it cannot be answered by returning one provision

**Retrieved / reranked:** 50 lexical candidates → 5 evidence passages (fingerprint `f50810e033df8644`)

| # | citation | citable | temporal | level |
|---|---|---|---|---|
| [1] | Protection of Plant Varieties and Farmers' Rights Rules, 2003 — Second Schedule, item 9 | True | CURRENT | A |
| [2] | Protection of Plant Varieties and Farmers' Rights Act, 2001 — Section 32 | True | CURRENT | A |
| [3] | Protection of Plant Varieties and Farmers' Rights Rules, 2003 — Second Schedule, item 9 | True | CURRENT | A |
| [4] | Protection of Plant Varieties and Farmers' Rights Rules, 2003 — Second Schedule, item 9 | True | CURRENT | A |
| [5] | Protection of Plant Varieties and Farmers' Rights Rules, 2003 — Form 0 | True | CURRENT | A |

**Provider status:** `rate_limited` — no completion was generated.

**Answer (published):**

```
I couldn't find sufficient authoritative evidence in the current corpus to answer this reliably.
```

**Citations:** 0 — n/a (no citations produced)


**Grounding:** FAIL

**Confidence:** NONE

**Golden target:** IN-PPVFR-ACT-2001 — Section 28  
`lexical_candidate_recall` True (rank 26) · `reranked_evidence_recall` False (rank None)

**Assessment: FAIL**  
No model output: the request was refused with HTTP 429 before generation. Independently, this is the one question where lexical-only retrieval lost the golden target — it entered the candidate pool at rank 26 and did not survive reranking into the top 5. Both causes are infrastructural; neither says anything about the model.

---

## Q027 — identifier

**Query:**  
Rule 23 of the Trade Marks Rules, 2017 — form and signing of an application

*Dataset family:* `identifier` · *domain:* `trademarks` · *jurisdiction:* `india`  
*Selected because:* first Rule-identifier question in file order whose document was not already selected

**Retrieved / reranked:** 50 lexical candidates → 5 evidence passages (fingerprint `02813c7fcc2d8371`)

| # | citation | citable | temporal | level |
|---|---|---|---|---|
| [1] | Trade Marks Rules, 2017 — Rule 23 | True | CURRENT | B |
| [2] | Geographical Indications of Goods (Registration and Protection) Rules, 2002 — Rule 23 | True | STALE | A |
| [3] | Designs Rules, 2001 — Rule 23 | True | STALE | A |
| [4] | Drugs and Cosmetics Rules, 1945 — Rule 23 | True | STALE | A |
| [5] | Patents Rules, 2003 — Rule 23 | True | STALE | B |

**Provider status:** completion returned.

**Answer (published):**

```
I couldn't find sufficient authoritative evidence in the current corpus to answer this reliably.
```

**Rejected draft** — diagnostic only, never published as an answer:

```
Rule 23 of the Trade Marks Rules, 2017 governs how a trademark application must be filed. It requires that the application be made on the prescribed form TM‑A and that it be signed by the applicant or by the applicant’s authorised agent [1]. The rule also sets out the content requirements for the application (precise description, graphical representation, and statements where the mark is three‑dimensional or a colour combination) and the procedural steps for amendments, class specifications and the Registrar’s power to refuse or correct the filing [1].
```

**Citations:** 0 — n/a (no citations produced)

- uncited claim: “Rule 23 of the Trade Marks Rules, 2017 governs how a trademark application must be filed.”

**Grounding:** FAIL

**Confidence:** NONE

**Golden target:** IN-TM-RULES-2017 — Rule 23  
`lexical_candidate_recall` True (rank 1) · `reranked_evidence_recall` True (rank 1)

**Assessment: NEEDS_REVIEW**  
The clearest result of the run. The model answered correctly from the supplied rule and cited `[1]` on two of three sentences; the opening topic sentence carried no marker, so the grounding gate refused the answer. This is the under-citation pattern from the smoke test, reproduced on real corpus evidence — the specific question this phase set out to answer.

---

## Q104 — treaty_international

**Query:**  
What capacity building does the Nagoya Protocol require of its parties?

*Dataset family:* `conceptual` · *domain:* `biodiversity_abs` · *jurisdiction:* `international`  
*Selected because:* first international-jurisdiction question over the Nagoya Protocol

**Retrieved / reranked:** 50 lexical candidates → 5 evidence passages (fingerprint `486a8f7490b9a868`)

| # | citation | citable | temporal | level |
|---|---|---|---|---|
| [1] | Nagoya Protocol — Article 22 | True | CURRENT | A |
| [2] | Nagoya Protocol — Article 25 | True | CURRENT | A |
| [3] | Nagoya Protocol — Article 18 | True | CURRENT | A |
| [4] | Nagoya Protocol — unlocated text (physical p. 1) | False | CURRENT | A |
| [5] | Nagoya Protocol — Article 32 | True | CURRENT | A |

**Provider status:** `rate_limited` — no completion was generated.

**Answer (published):**

```
I couldn't find sufficient authoritative evidence in the current corpus to answer this reliably.
```

**Citations:** 0 — n/a (no citations produced)


**Grounding:** FAIL

**Confidence:** NONE

**Golden target:** UN-NAGOYA-PROTOCOL-2010 — Article 22  
`lexical_candidate_recall` True (rank 1) · `reranked_evidence_recall` True (rank 1)

**Assessment: FAIL**  
No model output: HTTP 429 before generation. Retrieval and policy were sound — the golden target was the top-ranked evidence passage.

---

## Q085 — stale_evidence

**Query:**  
How does Indian drug law define an Ayurvedic, Siddha or Unani drug?

*Dataset family:* `conceptual` · *domain:* `drug_regulation` · *jurisdiction:* `india`  
*Selected because:* first drug-regulation question; every chunk of that instrument carries temporal_status STALE

**Retrieved / reranked:** 50 lexical candidates → 5 evidence passages (fingerprint `a167a840c70e38d1`)

| # | citation | citable | temporal | level |
|---|---|---|---|---|
| [1] | Drugs and Cosmetics Rules, 1945 — Rule 162A | True | STALE | A |
| [2] | Drugs and Cosmetics Rules, 1945 — Rule 167 | True | STALE | A |
| [3] | Drugs and Cosmetics Act, 1940 — Section 33EEC | True | STALE | A |
| [4] | Drugs and Cosmetics Act, 1940 — Section 3 | True | STALE | A |
| [5] | Drugs and Cosmetics Rules, 1945 — Rule 164 | True | STALE | A |

**Provider status:** `rate_limited` — no completion was generated.

**Answer (published):**

```
I couldn't find sufficient authoritative evidence in the current corpus to answer this reliably.
```

**Citations:** 0 — n/a (no citations produced)


**Grounding:** FAIL

**Confidence:** NONE

**Golden target:** IN-DC-ACT-1940 — Section 3  
`lexical_candidate_recall` True (rank 6) · `reranked_evidence_recall` True (rank 4)

**Assessment: FAIL**  
No model output: HTTP 429 before generation. Retrieval was sound (golden target at evidence rank 4) and the policy correctly admitted the STALE evidence rather than silently dropping it, but with no draft there is nothing to judge on stale-awareness.

---

## Aggregate

| | |
|---|---|
| Requests attempted | 5 |
| Reached the model | 2 |
| Refused before generation (HTTP 429) | 3 |
| Retries | 0 |
| Answers published | 0 |
| Grounded answers | 0 |
| Answers refused by a gate | 2 of 2 that reached the model |
| Citation validation failures | 1 |
| `lexical_candidate_recall` | 5/5 |
| `reranked_evidence_recall` | 4/5 |
| GOOD | 0 |
| NEEDS_REVIEW | 2 (Q001, Q027) |
| FAIL | 3 (Q078, Q104, Q085 — all rate-limited, not model failures) |

### Latency

| question | ms | note |
|---|---|---|
| Q001 | 1504.5 | full generation |
| Q078 | 1372.8 | rejection latency (429), not answer latency |
| Q027 | 1135.6 | full generation |
| Q104 | 289.3 | rejection latency (429), not answer latency |
| Q085 | 373.0 | rejection latency (429), not answer latency |

Mean over the 2 requests that actually generated: **1320.0 ms**. The three rejected requests are excluded — a 429 rejection latency is not an answer latency and averaging them together would understate generation cost by roughly a factor of three.

### Failure patterns observed

**F1 — under-citation on real corpus evidence (Q027). Confirmed.** The model opens with an uncited topic sentence and cites the sentences that follow. This is the same shape seen in the smoke test, so it is a stable behaviour of this prompt contract rather than an artefact of one fixture. Prompt rule 4 asks for a marker on every claim; the model treats the opening sentence as framing rather than as a claim, and the grounding gate does not.

**F2 — subdivision citations are rejected (Q001). Our defect, not the model's.** The answer cited `section 3(p)`; the supplied evidence is the Section 3 chunk, which contains clause (p). `ref_matches_locator` compares subdivisions exactly, and the supplied locator carries no subdivision, so a citation to a genuine sub-provision of supplied evidence reads as unsupported. Recorded, not fixed.

**F3 — non-ASCII citation markers (Q001).** The model wrote `【1】` rather than `[1]`. `CITATION_MARKER` matches ASCII brackets only, so the marker was invisible to both the validator and the grounding check. Recorded, not fixed.

**F4 — free-tier token throttling (Q078, Q104, Q085).** Groq's free plan caps this organisation at 8,000 tokens per minute for this model; each request consumed roughly 2,300–4,900. Five sequential requests exceeded the per-minute budget after the second. The provider reported each 429 and did not retry, which is the designed behaviour. It is a pacing constraint, not a defect.

**Not observed:** invented provisions, invented URLs, invented document names, markers pointing at passages that were never supplied, or answers drawn from outside the supplied evidence. On the two drafts that exist, the model stayed inside its evidence.

### What did not happen

No answer was published. Every draft that reached a gate was refused, and the user-facing text in all five cases was the insufficient-evidence message. The citation and grounding layers behaved as designed; the failures above are about citation *density* and *marker syntax*, not about fabrication.

