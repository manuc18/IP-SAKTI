# Phase 7.5 — two-request controlled validation

Status:

`ANSWER_LAYER_STATUS: FAIL`

`EXPERIMENT_MODE: LEXICAL_ONLY`

`requests_used: 1 of 2`

> **Not legal accuracy.** Everything below is an engineering property of the
> pipeline. Nothing here evaluates whether an answer is legally correct, and
> this system is not production-ready.

> **Retrieval caveat.** The dense (BGE-M3) arm was disabled for machine safety,
> so candidate generation was BM25 + identifier-boost only. Retrieval figures
> are `lexical_candidate_recall` and `reranked_evidence_recall`; they are not
> IP-SAKTI's overall retrieval recall and are not comparable with Phase 6.

Model `openai/gpt-oss-120b` · temperature 0.0 · max_completion_tokens
800 · timeout 15.0s · retries 0 ·
prompt version `0.2.0` · reranker `features` · top_k 5

**Q027 failed. Under the phase's stopping rule the run stopped there and Q001
was never requested**, so F2 and F3 remain unvalidated against a live model.

---

## Documentation consistency (Step 0)

Three statements were false as written and were corrected. No code, policy,
threshold, prompt or validator was touched, and no request was made in this step.

| file | said | corrected to |
|---|---|---|
| `ANSWER_PROVIDER.md` §head | "**No live request has been made.**" | six requests: the 7.2 smoke test and the five of 7.3, two completions, three 429s, none published |
| `ANSWER_PROVIDER.md` §7 | "Live smoke test — **Not run.**" | run once in Phase 7.2; what it returned, and that `--live` is one of *two* network paths (the other is `phase73_five_questions.py --run`) |
| `ANSWER_PROVIDER.md` §8 | "**No live request has been made**, so nothing is known about answer quality" | answer quality is unmeasured *because nothing was scored*, not because nothing was sent |
| `ANSWER_ENGINE.md` §10 | "An LLM provider exists but **has never been called**." | called six times; two completions; none published |

`ANSWER_GENERATION_CONTRACT.md` was already accurate ("six live requests have
now been made — one smoke test and the five of Phase 7.3") and needed no
correction. All three now carry the Phase 7.5 count.

---

## Q027 — validates F1 (under-citation)

**Query:** Rule 23 of the Trade Marks Rules, 2017 — form and signing of an application

*Dataset family:* `identifier` · *domain:* `trademarks` · *jurisdiction:* `india`

**Retrieved / reranked:** 50 lexical candidates → 5 evidence passages
(fingerprint `02813c7fcc2d8371` — **identical to Phase 7.3**, so the
model saw the same evidence as before and the comparison is like for like)

| # | citation | citable | temporal | level |
|---|---|---|---|---|
| [1] | Trade Marks Rules, 2017 — Rule 23 | True | CURRENT | B |
| [2] | Geographical Indications of Goods (Registration and Protection) Rules, 2002 — Rule 23 | True | STALE | A |
| [3] | Designs Rules, 2001 — Rule 23 | True | STALE | A |
| [4] | Drugs and Cosmetics Rules, 1945 — Rule 23 | True | STALE | A |
| [5] | Patents Rules, 2003 — Rule 23 | True | STALE | B |

**Provider:** completion returned, 1 request, 0 retries, 2471.3 ms.

**Published answer:** none. `I couldn't find sufficient authoritative evidence in the current corpus to answer this reliably.`

**Verdict: ABSTAINED at citation validation.**

- `unsupported_locator_in_prose`: the answer names rule 20, which is neither a supplied provision nor named by any supplied passage
- `unsupported_locator_in_prose`: the answer names rule 2002, which is neither a supplied provision nor named by any supplied passage

### Answer the phase asked for

| question | answer |
|---|---|
| Did the opening/topic sentence receive a citation? | **No.** |
| Did every factual/legal proposition receive a marker? | **No** — 3 of 9 sentences are uncited claims. |
| Did citation validation pass? | **No** — 2 failures. |
| Did grounding pass? | **It never ran.** Validation refused first, and the engine abstains at the first gate that refuses. Offline, on the recorded text, it would also have failed. |
| Was an answer published? | **No.** |
| What markers did the model emit? | `[1]` only, ASCII, on 6 of 9 sentences. No `【n】` appeared, so F3 was not exercised. |
| Faithful to the supplied evidence? | The cited body is on-evidence and traceable to the supplied Rule 23 text. The final sentence reaches for a *different* instrument (the GI Rules 2002) and is cut off mid-word by the 800-token ceiling. |

### Sentence by sentence

| # | claim? | markers | sentence |
|---|---|---|---|
| 1 | yes | **none** | Rule 23 of the Trade Marks Rules, 2017 governs the form and signing of a trademark appli… |
| 2 | yes | [1] | An application for registration of a trademark in respect of a specification of goods or… |
| 3 | yes | [1] | The application must (a) contain a precise description by words of the trademark where n… |
| 4 | yes | [1] | If an amendment is required to divide an application under the proviso to section 22, th… |
| 5 | yes | [1] | A non‑series application must relate to one trademark only, although it may cover as man… |
| 6 | yes | [1] | Where an application seeks registration of all goods or services in a class, or a large … |
| 7 | yes | [1] | If a single application is filed and the Registrar determines that the goods or services… |
| 8 | yes | **none** | Thus, Rule 23 specifies that the prescribed form is TM‑A, the signature must be that of … |
| 9 | yes | **none** | (For comparison, Rule 23 in the Geographical Indications Rules 2002 also requires a pres… |

### What actually failed, and why it is not what we were testing

**F1 was not fixed.** Prompt 0.2.0 enumerates opening sentences, framing
sentences and summary sentences as propositions that must be cited. The model
cited the six sentences carrying specifics and left the opening topic sentence,
the "Thus, Rule 23 specifies…" summary and the truncated comparative aside
uncited — the same two categories rule 4 now names explicitly, and the same
behaviour as Phase 7.3. A more specific instruction did not change it.

**Two new false positives appeared, both in the identifier parser.**

1. **`rule 20` — a real quotation of the corpus, refused.** The supplied Rule 23
   chunk literally reads *"published by the registrar under sub-rule (2) of rule
   20"*, and the answer quotes it. The chunk parses as `('rule','20',('2',))`.
   The answer wrote **`sub‑rule`** with U+2011 NON-BREAKING HYPHEN, which
   `parse_identifiers` does not recognise as `sub-rule`, so the same reference
   parsed as `('rule','20',())`. The two keys differ, the cross-reference arm
   missed, and the validator reported the corpus's own words as an invented
   locator. This is the *same class* of defect as F3 — a Unicode spelling the
   pipeline does not normalise — in a different character and a different module.

2. **`rule 2002` — an instrument's year read as a rule number.** *"Geographical
   Indications Rules 2002"* parsed as a rule numbered 2002. No answer citing any
   Indian instrument by its short title can survive this.

Neither is a hallucination, and neither is what Phase 7.4 set out to fix. The
citation contract refused an answer for quoting the evidence it was given.

**The completion was truncated** at `max_completion_tokens = 800`, mid-word.
That is a configuration limit, not a model failure, but it produced the ninth
sentence — an uncited, half-finished aside about a different instrument — which
contributed one of the two locator failures.

---

## Q001 — not run

`requests_used: 1`. Q027 failed, and the phase's stopping rule is explicit:
record the failure and stop, make no further request, fix nothing automatically.
Q001 was the only test of **F2** (safe hierarchical locators) and **F3**
(marker normalisation) against a live model, so both remain validated only by
`tests/test_answer_contract_74.py` — 26 offline tests, no model.

---

## Verdict

`ANSWER_LAYER_STATUS: FAIL`

The answer layer still fails on the first controlled validation. Three things
are now known that were not known before this request:

* **F1 is not fixed by instruction alone.** The model under-cites framing and
  summary sentences under prompt 0.2.0 exactly as it did under 0.1.0.
* **The identifier parser is the next failure surface**, not the citation
  policy. It rejected a verbatim quotation of the supplied text over a
  non-breaking hyphen, and reads an instrument's year as a rule number.
* **F2 and F3 are untested live.** They were designed against Phase 7.3's
  observations and pass offline; whether the model still emits `【1】` or
  `section 3(p)` is unknown.

Nothing here measures legal accuracy, nothing here is production-ready, and the
111 golden questions remain 0 of 111 human-verified.
