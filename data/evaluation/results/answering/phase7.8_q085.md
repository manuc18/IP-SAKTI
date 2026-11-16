# Phase 7.8 — caution fix, and one live request on the stale-evidence question

`ANSWER_E2E_STATUS: PASS`

`EXPERIMENT_MODE: LEXICAL_ONLY`

`requests_used: 1` · retries 0

> **Not legal accuracy.** Everything below is an engineering property of the
> pipeline. No human has verified that this answer states Indian drug law
> correctly, and this system is not production-ready.

> **Retrieval caveat.** The dense (BGE-M3) arm was disabled for machine safety,
> so candidate generation was BM25 + identifier-boost only. Retrieval figures
> are `lexical_candidate_recall` and are not comparable with Phase 6.

---

## Part A — the caution counted the wrong thing

Phase 7.7 published this project's first answer. It cited one source and told
the reader:

> 4 of 5 **cited** sources are marked STALE and may have been amended.

Four sources were *supplied*. One was cited. The count came from `apply_policy`,
which runs **before** the provider and therefore cannot know what will be cited
— so it counted the citable supplied passages and wrote "cited" over the top.
Nothing caught it for four phases because every earlier live answer abstained
and no caution was ever rendered.

The fix is not better wording in `apply_policy`. It is that `apply_policy` does
not have the fact and stops claiming to. `citation_cautions(citations)` is
called by the engine after the citation validator has produced them:

| cited | stale | wording |
| --- | --- | --- |
| 0 | – | *(no caution)* |
| 1 | 0 | *(no caution)* |
| 1 | 1 | The cited source is marked stale. |
| 2 | 2 | Both cited sources are marked stale. |
| 2 | 1 | 1 of 2 cited sources **is** marked stale. |
| 3 | 2 | 2 of 3 cited sources are marked stale. |
| 3 | 3 | All 3 cited sources are marked stale. |

Sources are deduplicated by `chunk_id`: two markers resolving to one passage are
one source, and counting them twice would overstate both halves of "n of m".
The adjacent weak-provenance caution carried the identical defect and is fixed
with it. `POLICY_VERSION` → `0.3.0`. Deterministic; no model is consulted.

---

## Part D — Q085

**Query:** How does Indian drug law define an Ayurvedic, Siddha or Unani drug?

*Dataset family:* `conceptual` · *domain:* `drug_regulation` · *jurisdiction:* `india`

Selected in Phase 7.3 as the stale-evidence slot and **never answered** — its
7.3 request came back 429 before the model saw it. All five supplied passages
are `STALE` at evidence level A, which makes it the only question in the
selection that exercises the temporal caution end to end.

**Evidence fingerprint:** `a167a840c70e38d1` — identical to Phase 7.3's,
so the Phase 7.6 parser changes moved nothing here either.

| # | citation | citable | temporal | level |
|---|---|---|---|---|
| [1] | Drugs and Cosmetics Rules, 1945 — Rule 162A | True | STALE | A |
| [2] | Drugs and Cosmetics Rules, 1945 — Rule 167 | True | STALE | A |
| [3] | Drugs and Cosmetics Act, 1940 — Section 33EEC | True | STALE | A |
| [4] | Drugs and Cosmetics Act, 1940 — Section 3 | True | STALE | A |
| [5] | Drugs and Cosmetics Rules, 1945 — Rule 164 | True | STALE | A |

| | |
|---|---|
| candidates | 50 |
| supplied evidence | 5 |
| candidate rank of golden target | 6 |
| reranked rank of golden target | 4 |
| **cited sources** | **1** |
| model | `openai/gpt-oss-120b`, temperature 0.0 |
| prompt / policy / validator / grounding | `0.3.0` / `0.3.0` / `0.2.0` / `0.2.0` |
| provider status | completion returned |
| finish_reason | **`stop`** — not truncated |
| prompt tokens | 1725 |
| reasoning tokens | 243 |
| completion tokens | 328 of a 800 ceiling (41%) |
| visible tokens | 85 |
| total tokens | 2053 |
| latency | 1546.3 ms |
| requests / retries | 1 / 0 |

### Raw draft

```
An “Ayurvedic, Siddha or Unani drug” means any medicine intended for internal or external use for diagnosis, treatment, mitigation or prevention of disease or disorder in humans or animals, and manufactured exclusively in accordance with the formulae described in the authoritative books of the Ayurvedic, Siddha and Unani Tibb systems of medicine specified in the First Schedule [4].
```

### Normalised published text

Identical — `marker_normalisation_applied` = **False**. The model
wrote an ASCII `[4]`.

### Citations

| # | citation | pages | level | temporal |
|---|---|---|---|---|
| [4] | Drugs and Cosmetics Act, 1940 — Section 3 | p. 6 | A | **STALE** |

**The model cited `[4]`, not `[1]`.** Five passages were supplied and it chose
the fourth — the Drugs and Cosmetics Act Section 3 chunk that actually carries
the definition, which the reranker had placed at rank 4. It did not default to
the top-ranked passage. Golden target: `IN-DC-ACT-1940` — `Section 3`.

### Validation

**PASS** — 0 failures. No `unsupported_locator_in_prose` despite the answer
naming "the First Schedule": that is a cross-reference present in the supplied
passage's own text, which the validator's third arm admits.

### Grounding

**PASS** — `grounded = True` · claims 1 · supported 1 ·
`supported_claim_rate` **1.0** · unsupported 0 ·
weakly grounded 0 · uncited evidence 4 of 5.

### Stale-source caution

Cited temporal statuses: `['STALE']`. Supplied: `['STALE']`.

```
The cited source is marked stale. The text may have been amended since the consolidated version held in the corpus; verify against the current official text.
```

**This is the exact case the old code got wrong.** One source cited, of five
supplied, all stale. Before Part A it would have read *"Every source cited here
is marked STALE"* — a claim about five sources under an answer resting on one.
The `[STALE]` label on the citation line comes from the citation record and is
unchanged.

### Published

**YES.** Confidence `LOW` — correctly LOW, because the one source the
answer rests on is stale.

```
Answer

An “Ayurvedic, Siddha or Unani drug” means any medicine intended for internal or external use for diagnosis, treatment, mitigation or prevention of disease or disorder in humans or animals, and manufactured exclusively in accordance with the formulae described in the authoritative books of the Ayurvedic, Siddha and Unani Tibb systems of medicine specified in the First Schedule [4].

Source(s)

[4] Drugs and Cosmetics Act, 1940 — Section 3, p. 6, [STALE]

Caution

- The cited source is marked stale. The text may have been amended since the consolidated version held in the corpus; verify against the current official text.

Confidence: LOW (ordinal, derived from evidence provenance — not a probability and not a measure of legal correctness)

```

---

## Two questions, two published answers

| | 7.7 Q027 | 7.8 Q085 |
| --- | ---: | ---: |
| prompt | 0.3.0 | 0.3.0 |
| sentences | 1 | 1 |
| propositions / cited | 1 / 1 | 1 / 1 |
| citation completeness | 100% | **100%** |
| validation | PASS | **PASS** |
| grounding | PASS (1.0) | **PASS (1.0)** |
| published | YES | **YES** |
| truncated | NO | **NO** |
| cited source | rank 1 of 5 | **rank 4 of 5** |
| completion tokens | 210 | 328 |
| latency | 1286.2 ms | 1546.3 ms |

Two for two under prompt `0.3.0`, on questions of different shape — an
identifier lookup and a definitional question — with the second selecting a
non-top-ranked source. That is two observations, not a rate. `F1_STATUS` stays
`IMPROVED`, not solved.

**Both answers are one sentence.** Rule 3 permits up to four and neither
question has needed them. Whether a question that genuinely requires three or
four sentences still gets a marker on each is untested, and it is the obvious
next thing to find out — a three-sentence answer has three chances to omit one.

---

## Machine safety

| | |
|---|---|
| local model loaded | NO |
| BGE-M3 | NO |
| torch | NO |
| sentence-transformers | NO |
| GPU / MPS / CUDA | NO |
| multiprocessing | NO |
| background process | NO |
| model download | NO |
| corpus re-ingestion | NO |
| retrieval benchmark | NO |
| reranking benchmark | NO |
| full pytest | NO |
| network requests | **exactly 1** |
| Groq requests | **exactly 1** |
| retries / fallback / regeneration / model switching | **0 / none / none / none** |

No credential, Authorization header or API key appears in this file or in
`phase7.8_q085.json`.
