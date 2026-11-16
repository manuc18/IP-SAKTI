# Phase 7.7 — one controlled live request, Q027

`ANSWER_E2E_STATUS: PASS`

`F1_STATUS: IMPROVED`

`EXPERIMENT_MODE: LEXICAL_ONLY`

`requests_used: 1` · retries 0

> **Not legal accuracy.** Everything below is an engineering property of the
> pipeline. Nothing here evaluates whether the answer is legally correct, and
> this system is not production-ready.

> **Retrieval caveat.** The dense (BGE-M3) arm was disabled for machine safety,
> so candidate generation was BM25 + identifier-boost only. Retrieval figures
> are `lexical_candidate_recall` and are not comparable with Phase 6.

**This is the first answer the pipeline has ever published.** Three prior live
completions (Phase 7.2 smoke, Phase 7.3 Q001 and Q027, Phase 7.5 Q027) were all
refused by the citation or grounding gate.

---

## Q027

**Query:** Rule 23 of the Trade Marks Rules, 2017 — form and signing of an application

*Dataset family:* `identifier` · *domain:* `trademarks` · *jurisdiction:* `india`

**Evidence fingerprint:** `02813c7fcc2d8371` — asserted equal to the
Phase 7.5 baseline **before** the request was sent. Same five passages, same
order. The prompt is the only thing that changed.

| # | citation | citable | temporal | level |
|---|---|---|---|---|
| [1] | Trade Marks Rules, 2017 — Rule 23 | True | CURRENT | B |
| [2] | Geographical Indications of Goods (Registration and Protection) Rules, 2002 — Rule 23 | True | STALE | A |
| [3] | Designs Rules, 2001 — Rule 23 | True | STALE | A |
| [4] | Drugs and Cosmetics Rules, 1945 — Rule 23 | True | STALE | A |
| [5] | Patents Rules, 2003 — Rule 23 | True | STALE | B |

**Retrieved:** 50 lexical candidates → 5 evidence passages ·
golden target `IN-TM-RULES-2017` Rule 23 at **candidate rank 1, reranked rank 1**

| | |
|---|---|
| model | `openai/gpt-oss-120b` (temperature 0.0, top_p 1.0) |
| prompt version | `0.3.0` |
| policy version | `0.2.0` |
| validator version | `0.2.0` |
| grounding version | `0.2.0` |
| provider status | completion returned |
| finish_reason | **`stop`** — not truncated |
| prompt tokens | 1980 |
| completion tokens | 210 of a 800 ceiling (26%) |
| reasoning tokens | 174 |
| visible tokens | 36 |
| latency | 1286.2 ms |
| requests / retries | 1 / 0 |

### Raw model draft

```
An application for the registration of a trademark must be made in Form TM‑A and signed by the applicant or his agent [1].
```

### Normalised published text

```
An application for the registration of a trademark must be made in Form TM‑A and signed by the applicant or his agent [1].
```

`marker_normalisation_applied` = **False**. The model
wrote an ASCII `[1]`, so the Phase 7.4 marker normaliser had nothing to do —
`【1】` did not recur. Note the draft *does* contain U+2011 (`TM‑A`), which the
Phase 7.6 identifier normaliser now handles; had the model written a
`sub‑rule` cross-reference as it did in Phase 7.5, it would have resolved.

### Citations

| # | citation | pages | level | temporal |
|---|---|---|---|---|
| [1] | Trade Marks Rules, 2017 — Rule 23 | p. 59 | B | CURRENT |

Every field copied from the evidence record. The model contributed the marker
and nothing else.

### Validation

**PASS** — 0 failures. No `unsupported_locator_in_prose`, no `fabricated_url`,
no `unknown_marker`. The answer named one provision, Rule 23, which is a
supplied passage.

### Grounding

**PASS** — `grounded = True` · claims 1 · supported 1 ·
`supported_claim_rate` **1.0** · unsupported claims 0 ·
weakly grounded 0 · uncited evidence 4 of 5 passages.

### Published

**YES.** Confidence `MODERATE` (ordinal, from evidence provenance —
not a probability and not a measure of legal correctness).

```
Answer

An application for the registration of a trademark must be made in Form TM‑A and signed by the applicant or his agent [1].

Source(s)

[1] Trade Marks Rules, 2017 — Rule 23, p. 59

Caution

- 4 of 5 cited sources are marked STALE and may have been amended.
- The supplied evidence mixes current and superseded instruments; they address different provisions, so this is not flagged as a conflict.

Confidence: MODERATE (ordinal, derived from evidence provenance — not a probability and not a measure of legal correctness)

```

---

## A/B: Phase 7.5 vs Phase 7.7

Identical question, identical evidence fingerprint `02813c7fcc2d8371`, identical
reranker, identical policy. Prompt `0.2.0` → `0.3.0` is the only variable.

| Metric | 7.5 (prompt 0.2.0) | 7.7 (prompt 0.3.0) |
| --- | ---: | ---: |
| Sentences | 9 | **1** |
| Total propositions | 9 | **1** |
| Cited propositions | 6 | **1** |
| Citation completeness | 6/9 = 67% | **1/1 = 100%** |
| Opening/framing sentence | present, uncited | **none written** |
| Summary sentence | present, uncited | **none written** |
| Comparative aside | present, uncited | **none written** |
| Citation validation | FAIL | **PASS** |
| Grounding | not reached | **PASS (1.0)** |
| Published | NO | **YES** |
| Truncated | YES | **NO** (`finish_reason: stop`) |
| Completion tokens | 800 (ceiling, truncated) | **210** |
| Latency | 2471.3 ms | **1286.2 ms** |

### What actually changed

Phase 7.4 answered F1 by naming the sentence kinds that must be cited. The model
went on writing them uncited. Phase 7.6 stopped asking for more citations and
asked for fewer sentences, and the model wrote **one sentence, which it cited**.

The uncited sentences did not become cited. **They stopped being written.** That
is the mechanism the 1–4 sentence contract was designed around, and this is one
observation of it working.

### What this does not establish

* **One request.** `F1_STATUS: IMPROVED`, not solved. One question, one
  sampling, temperature 0 — which removes sampling variance but not the
  possibility that a different question produces a different shape. A
  three-sentence answer has three chances to omit a marker; this one had one.
* **The answer is much shorter, and that is a real trade.** Phase 7.5's rejected
  draft covered content requirements, amendment on Form TM-M, class corrections
  and the Registrar's discretion. This answer covers the form and the signature
  — which is what the question asked — and drops the rest. Whether 1–4 sentences
  is right for questions that genuinely need more is untested; rule 3 permits
  more, and no question has yet exercised that permission.
* **4 of 5 supplied passages went uncited**, which is expected under a one
  sentence answer and is not a defect. It does mean the answer rests on a single
  source.
* **Nothing here is legal accuracy.** No human has verified that this sentence
  states Rule 23 correctly. The golden dataset remains 0 of 111 human-verified.

### The token decision, now measured rather than estimated

Phase 7.6 kept `max_completion_tokens` at 800 by estimating the budget from
character counts. The measurements:

| | 7.6 estimate | 7.7 measured |
| --- | ---: | ---: |
| prompt tokens | ~2231 | **1980** |
| reasoning tokens | ~223 (inferred by subtraction) | **174** |
| completion tokens | 800 (truncated) | **210** |

The estimate was ~13% high on the prompt, and the reasoning figure was in the
right range. **The decision holds with room to spare:** this completion used
26% of the ceiling, and total consumption was
2190 tokens against a free-tier TPM limit that returned three 429s in
Phase 7.3. Raising the ceiling to 1000 would have bought nothing.

---

## A defect this answer exposed

The rendered caution reads:

> 4 of 5 **cited** sources are marked STALE and may have been amended.

Only **one** source was cited. The caution is counting *supplied* evidence and
calling it cited, which was invisible for as long as every answer abstained and
nothing was rendered. It is a user-facing wording error in the caution, not a
failure of any gate, and it is recorded here rather than fixed — no code was
changed in this phase after the request.

---

## Machine safety

| | |
|---|---|
| local model loaded | NO |
| BGE-M3 loaded | NO |
| torch | NO |
| sentence-transformers | NO |
| GPU / MPS / CUDA | NO |
| multiprocessing | NO |
| background process | NO |
| model download | NO |
| corpus re-ingestion | NO |
| benchmark | NO |
| full pytest | NO |
| network requests | **exactly 1** |
| Groq requests | **exactly 1** |
| retries | **0** |

No credential, Authorization header or API key appears in this file or in
`phase7.7_q027.json`.
