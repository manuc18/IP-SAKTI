# Answer engine

`ANSWER_LAYER_STATUS: PROVISIONAL` · answering `0.1.0` · policy/validator/grounding `0.2.0` · prompt `0.3.0` · chunking 0.2.0 · Phase 7, revised 7.6

Phase 7 ends at `RERANKED EVIDENCE → GROUNDED ANSWER OBJECT → VALIDATED
CITATIONS`. There is no server, no UI, no LLM call, and no evaluation run over
the 111 golden questions.

---

## 1. Pipeline

```
USER QUERY
  ↓
RETRIEVAL          ai/retrieval/       BGE-M3 top-50 ∪ BM25+identifier-boost top-50
  ↓
RERANKING          ai/reranking/       bge-reranker-v2-m3 → top-10        (Phase 6)
  ↓  ── the Phase 7 boundary. Everything below loads no model. ──
EVIDENCE POLICY    ai/answering/policy.py              relevant · authoritative ·
  ↓                                                    current · citable, kept apart
ANSWER GENERATION  ai/answering/providers.py           prose + [n] markers only
  ↓
CITATION VALIDATION ai/answering/citation_validator.py  no passport, no entry
  ↓
GROUNDING          ai/answering/grounding.py           every claim carries a source
  ↓
FINAL ANSWER       ai/answering/render.py              Answer object → text
```

The answering package imports no retriever, no index and no tensor library. It
loads in 0.075 s and a test asserts that `torch`, `transformers` and
`sentence-transformers` are absent from `sys.modules` after importing it.

---

## 2. The three objects that carry the design

**`Evidence` is a numbered passport, not a chunk.** Every provider cites by
marker — `[1]`, `[2]` — and by nothing else. A provider that can only emit small
integers cannot invent a chunk id, a section number, a page or a URL and have it
accepted. The worst it can do is cite `[9]` when four passages were supplied,
which is a dictionary lookup away from being caught. Compare the alternative,
where a model emits `IN-PATENTS-ACT-1970::section-3` and something downstream
must decide whether that string is real.

**`EvidenceSet` is the registry the customs officer checks against.** Built once
from retrieval output, before the provider runs, and it fingerprints itself
(SHA-256 over marker + chunk id, order-sensitive). A citation is admissible if
and only if it resolves inside the set that was handed to the engine, and the
fingerprint makes "the same set" checkable rather than assumed.

**`Answer` is never model output.** The provider returns prose. Every field of
the final object is assembled from evidence records the corpus produced. The
provider's text can be rejected; it cannot contribute metadata.

---

## 3. Evidence policy — four axes, kept apart

Phase 7 §6, and the corpus shows why they must not collapse into one score.

| Axis | Question | Source |
| --- | --- | --- |
| Relevant | Does this address the query? | content-term overlap (a screen, see below) |
| Authoritative | Who published the PDF? | `source_evidence_level` A/B/C/D |
| Current | Does it still state the law? | `temporal_status` |
| Citable | May it be quoted as authority? | `citable`, copied from Phase 3 |

Each passage gets a role: `citable_authority` (may be cited), `context_only`
(may be read, never cited), `off_topic`.

**Nothing is discarded for being weak.** Phase 7 §6 forbids it, and the corpus
makes the instruction load-bearing:

| Fact | Count | Consequence |
| --- | ---: | --- |
| Chunks | 3,073 | |
| `temporal_status: STALE` | 1,717 (55.9%) | including all 920 Drugs and Cosmetics Rules chunks |
| `temporal_status: CURRENT` | 1,356 | |
| Evidence level A / B / C | 1,727 / 1,155 / 191 | the Patents Act and Trade Marks Act are **B** |
| `citable: false` | 57 | all for `locator_unavailable` |

A policy that refused stale evidence would refuse most of the questions this
project exists to answer, and a level-A floor would refuse the Patents Act. So
the defaults admit them and label them, and the opposite policy is one flag away.

### The relevance screen is a screen, not a model

Content-term overlap with a short stopword list, threshold 1. It catches
evidence that shares *nothing* with the query and little else. A higher
threshold would start rejecting correct provisions that use different words —
which is the entire reason the pipeline has a dense retriever and a
cross-encoder in front of it. Phase 7 §12 forbids adding another embedding model
here, and this is honestly labelled rather than dressed up.

### `stale_only_is_insufficient` — the phase's most consequential switch

Phase 7 §14 lists "only stale evidence for a current-law question" as an
insufficiency condition. Taken literally against this corpus that silences the
Drugs and Cosmetics Rules entirely — the single most Ayurveda-relevant
instrument present.

**Default: `False`.** The engine answers, marks every citation `[STALE]`,
attaches a prominent caution, and drops confidence to `LOW`. Set it `True` and
the engine abstains instead. Both behaviours are tested. This is a policy
decision that deserves an explicit human ruling rather than a default chosen by
whoever wrote the module — it is flagged, not settled.

---

## 4. Citation validation — no passport, no entry

`ai/answering/citation_validator.py`. Eight rules, Phase 7 §10:

| # | Rule | Failure code |
| --- | --- | --- |
| 1–2 | The marker resolves inside the supplied evidence set | `unknown_marker` |
| 3 | The citation has a document identity | `missing_document_identity` |
| 4 | A locator, or an explicit `locator_unavailable`, and the two must agree | `missing_locator`, `inconsistent_locator_state` |
| 5 | A non-citable chunk can never become a citation | `non_citable_evidence` |
| 6 | Citations cannot refer outside the engine's input | `evidence_not_supplied` (fingerprint mismatch) |
| 7 | Pages resolve to a positive page of the source PDF | `unresolvable_pages` |
| 8 | Citation metadata is not fabricated | structural — see below |
| §11 | No invented provision in the prose | `unsupported_locator_in_prose` |
| §11 | …read through one identifier spelling (Phase 7.6) | see below |
| §11 | No URL written by the provider | `fabricated_url` |
| — | Below the policy's evidence floor | `below_evidence_floor` |

**Rule 8 is enforced by construction, not by checking.** `build_citation()`
copies document, locator, pages, evidence level, temporal status, filename and
SHA-256 out of the evidence record. `citation_text` is the chunk's own Phase 2
citation string — rebuilding it from parts here would be a second opinion about
how a provision is cited, and two opinions is one too many.

**Failure is never repaired.** A validator that quietly drops the one bad
citation and returns the rest teaches every caller downstream that its output
can be trusted without being read. When validation fails it returns *no*
citations and a list of structured failures, and the engine abstains.

**The prose scan has three arms, and none is a loophole.** A named provision is
supported if it *is* a supplied passage; **or** if it is a subdivision of a
supplied passage whose marker is present in that passage's text; **or** if some
supplied passage's own text names it. Statutes cross-reference constantly — Rule 65 of
the Drugs and Cosmetics Rules cannot be quoted without naming Form 20 — so a
rule requiring every named provision to be separately supplied would reject
correct quotations of the corpus for containing the corpus. That is the trap
Phase 6 §14 recorded about anchor terms, avoided the same way: the test is
whether the string came from the evidence, not whether it appears in an index.
What remains caught is a provision appearing in neither the supplied locators
nor the supplied text — precisely a provision the provider invented.

### The second arm: a clause of the passage you were given (Phase 7.4)

Phase 7.3 Q001 supplied the whole of Section 3 of the Patents Act as one chunk.
The model wrote "section 3(p)" — a real clause of that very passage — and the
scan reported `unsupported_locator_in_prose`, because `ref_matches_locator`
compares subdivisions exactly and the chunk's locator carries none. A correct
citation of the supplied evidence was refused.

`ref_matches_subdivision_of` adds the rule, and the second half is the whole of
it: **a parent supports a child only when the child's marker is actually present
in the parent's text.** Section 3's chunk supports `section 3(p)` because `(p)`
is in it, and supports `section 3(zz)` never. Two restrictions keep it from
becoming a descendant rule — the target must be the *whole* provision (a chunk
that is itself Section 3(p) is a sibling of Section 3(q), not its container),
and every subdivision named must be present, not merely the first.

The citation is unaffected: `build_citation` still copies the evidence record,
so a reader sees the corpus's own `Section 3` and never the model's `3(p)`. No
stored corpus locator is altered. The `citable` and `locator_unavailable`
invariants are untouched.

---

### One spelling per reference (Phase 7.6)

Phase 7.5's Q027 was refused for naming a provision that was sitting in the
evidence it cited. The supplied chunk reads *"under sub-rule (2) of rule 20"*;
the answer quoted it with U+2011 NON-BREAKING HYPHEN, the parser did not read
`sub‑rule` as `sub-rule`, and the same reference produced two different keys —
`('rule','20',('2',))` from the chunk, `('rule','20',())` from the quotation —
so the cross-reference arm above never matched.

`parse_identifiers` now normalises three hyphen spellings (U+2010, U+2011,
U+2013 → `-`) before reading, so every caller — the router, the reranker, the
policy and this validator — derives the same key from the same reference. The
em dash and the minus sign are deliberately excluded; both are legislative
heading separators in this corpus, not hyphens. The same phase stopped
`_DIRECT` reading an instrument's year as a provision number, so *"Geographical
Indications Rules 2002"* no longer parses as rule 2002.

Both changes make the validator refuse *less*, which is the direction that
warrants proof rather than assertion: across all 111 golden queries the parser's
output is byte-identical before and after, and across the 3073 processed chunks
8 changed, every one of them dropping a false positive. See
`docs/ANSWER_GENERATION_CONTRACT.md` §10 and §11.

---

## 5. Grounding

`check_grounding` reports four mechanical properties and one advisory:

* **Citation support** — every `[n]` resolves.
* **Evidence coverage** — `supported_claim_rate`.
* **Unsupported claims** — claim sentences with no marker.
* **Citation completeness** — evidence supplied and never cited.
* **Weakly grounded** *(advisory)* — a sentence citing a passage it shares no
  content term with. The citation resolves, so it is not a hallucination; the
  association is thin. It does not clear `grounded`, because downgrading a real
  citation on a lexical heuristic would be the heuristic making a decision it is
  not good enough to make.

**Sentence splitting errs towards merging**, because legal prose is full of
"s. 3", "r. 65(2)" and "No. 332" and a splitter that broke on those would
manufacture fragments and report them as unsupported claims. The cost is stated
rather than hidden: a merged sentence containing one cited and one uncited
assertion counts as supported, so unsupported-claim detection **under-reports**.

**Markers are normalised before anything reads them (Phase 7.4).** Phase 7.3
Q001 wrote its marker as `【1】`; `CITATION_MARKER` is ASCII-only, so the marker
was invisible to the validator *and* to this check, and a correctly sourced
sentence was reported as carrying no citation. `normalise_markers` rewrites
recognised equivalent spellings — today exactly `【n】` — to `[n]`. It is an
enumerated table rather than a pattern, because marker syntax is a contract with
the provider and "any bracket-ish pair around a digit" would admit punctuation
nobody agreed to. The alias is as strict as the ASCII form: `【 1 】` is rejected
because `[ 1 ]` is. Malformed and mismatched forms (`【a】`, `【1]`, `(1)`) are
left untouched and resolve to nothing.

Normalisation runs **once, in the engine**, and the normalised text is what is
validated, grounded *and published*, so the prose and the citation list beneath
it always use the same syntax. `draft.text` keeps what the model actually wrote;
`diagnostics["marker_normalisation_applied"]` records whether the two differ.

**Grounding has never been relaxed, and Phase 7.6 is where that cost
something.** Phase 7.3 saw Q027 open with an uncited topic sentence. Phase 7.4
answered it by making the prompt specific about which sentence kinds need a
marker. Phase 7.5 spent a live request re-running the same question and the
model did the same thing: nine sentences, six markers, and the three uncited
ones were an opening topic sentence, a summary and an aside — exactly the
categories the prompt had just enumerated. **F1 is persistent model behaviour,
not a wording gap.**

The obvious response at that point is to teach `is_claim` that a first or last
sentence is framing. That is the permissive fix, and it is the one this layer
will not take: it would publish the same answer while knowing less about it, and
every future uncited claim that happened to fall first or last would pass with
it. Phase 7.6 changed the *output shape* instead — prompt `0.3.0` asks for one
to four sentences, each carrying evidence, so there is no introduction or
summary left to go uncited. Not one line of `check_grounding` changed;
`GROUNDING_VERSION` is still `0.2.0`, and
`tests/test_answer_contract_76.py::TestIssue3AnswerContract` asserts that the
BAD example and the shape of the actual Phase 7.5 completion both still fail
this gate.

Whether the model complies with `0.3.0` is unknown. One live request is
reserved to find out.

---

## 6. Insufficient evidence

The engine abstains, with structured reasons, on any of:

| Code | Condition |
| --- | --- |
| `no_evidence` | nothing was supplied |
| `no_relevant_evidence` | no passage shares a content term with the query |
| `no_citable_authority` | relevant passages exist but none may be cited |
| `only_stale_evidence` | every citable passage is STALE **and** the policy requires current authority |
| `conflicting_evidence` | a conflict was detected **and** `abstain_on_conflict` |
| `provider_abstained` | the provider returned nothing |
| `citation_validation_failed` | any citation rule failed |
| `ungrounded_answer` | a claim carries no citation and `require_full_grounding` |

The message is fixed: *"I couldn't find sufficient authoritative evidence in the
current corpus to answer this reliably."* The gap is never filled with model
knowledge, and never filled by relaxing a rule that just refused something.

---

## 7. Conflicting evidence

Four deterministic detectors. **Conflict means the same provision, not merely
different provisions** — firing on any set that mixes CURRENT and STALE would
fire on most of them, and a flag that is always true carries no information. A
mixed set of *different* provisions gets a caution instead.

| Kind | Fires when |
| --- | --- |
| `temporal_divergence` | same document + same locator, different `temporal_status` |
| `version_divergence` | same document + same locator, different `source_version_id` / `consolidated_as_of` |
| `same_provision_divergence` | same document + same locator, different chunks |
| `ambiguous_citation` | two passages render the identical citation string |
| `textual_override` | one passage says "omitted by" / "substituted by" of a provision another passage *is* |

Nothing resolves a conflict. Phase 7 §15 is explicit that deciding which
provision prevails is legal reasoning and out of scope, and a deterministic
module is the wrong place to acquire that opinion by accident. Both sides are
cited, the caution says the system does not choose, and confidence drops to
`LOW`.

### A real finding, from building the fixtures

The Drugs and Cosmetics Rules, 1945 contain **both Part VI rule 65** (conditions
of licence, pp. 71–72) **and Part II rule 65** (a schedule entry,
"Oxytetracycline Hydrochloride Ointment", p. 533). The corpus knows they are
different — their locators differ on `part` — but both rendered as
`Drugs and Cosmetics Rules, 1945 — Rule 65`. The citation *objects* resolved
correctly by `chunk_id` and page; the rendered footnotes did not distinguish
them. `ambiguous_citation` exists because of this case.

**Fixed at the source in Phase 7.1**, not papered over here.
`ingest.citation.disambiguate_citations()` runs once per document after its
chunks are built, finds citation strings shared by chunks sitting in *different*
structural containers, and promotes the separating field into the string:
`Part VI, Rule 65` and `Part II, Rule 65`. A provision merely split across
several chunks has one locator, so no field separates it and nothing is touched.
Over the current corpus the pass would change 127 citations across 4 documents
(Copyright Rules 2013, Drugs and Cosmetics Rules 1945, PPV&FR Rules 2003,
TRIPS), leaving only same-provision repeats behind.

**The stored chunks still carry the old strings.** The fix is in the generation
layer; `data/processed/chunks/` is regenerated only by a re-ingest, which
Phase 7.1 was told not to run. Until then `ambiguous_citation` continues to fire
at runtime on the Rule 65 pair — correctly, because at runtime the collision is
still there.

---

## 8. Confidence

Ordinal — `HIGH` / `MODERATE` / `LOW` / `NONE` — and deliberately not a number.
Nothing in this pipeline estimates the probability that an answer is legally
correct, and a decimal would be read as though something did. What it ranks is
the provenance of the sources the answer actually cites:

* `NONE` — abstained.
* `LOW` — conflict detected, or every citation STALE, or any citation at level C/D.
* `HIGH` — grounded, every citation level A and CURRENT, no weak grounding.
* `MODERATE` — otherwise.

Under this ladder a correct answer from the Patents Act is `MODERATE`, because
the Act's PDF is evidence level B. That is the intended reading.

---

## 9. Disclaimer policy

A paragraph of boilerplate under every answer is not caution, it is wallpaper:
readers stop seeing it within a day, and it is then absent exactly when it
matters. So there is one system-level line (`render.SYSTEM_DISCLAIMER`), shown
once by whatever surrounds this, and per-answer cautions appear only when the
evidence earned one — stale text, conflicting provisions, weak provenance, or
insufficient evidence.

---

## 10. What is not built, and what is not measured

* **An LLM provider exists and has been called seven times.** Phase 7.2 added
  `GroqProvider` (`openai/gpt-oss-120b`, stdlib `urllib`, no new dependency, one
  request, no retries, ≤15 s). One smoke test in Phase 7.2, five requests in
  Phase 7.3 and one in Phase 7.5 have been made; three returned completions,
  three were rejected with 429, and **no completion has been published** — each
  was refused by the citation or grounding gate. The default provider is still
  the offline extractive one, and nothing reaches the network unless a caller
  constructs Groq explicitly. See `docs/ANSWER_PROVIDER.md`.
* **No retries, no regeneration, no fallback provider.** A failed answer is an
  abstention. The free tier has already shown a TPM ceiling, and an automatic
  retry turns one refusal into two requests. Phase 7.6 kept
  `max_completion_tokens` at 800 after measuring the budget rather than raising
  it because a nine-sentence answer ran out of room; see the contract §12.
* **No server, no UI, no voice, no multilingual generation.**
* **No evaluation run.** `scripts/evaluation/benchmark_answers.py` implements the
  framework and a fixture path. It has **not** been run over the 111 golden
  questions.
* **The relevance screen is lexical.** It cannot tell that a passage is about the
  right regime but the wrong question.
* **Contradiction detection is partial.** Two provisions that contradict each
  other in substance, while sharing no locator and announcing no override, are
  invisible. Finding them requires the legal reasoning §15 defers.
* **The extractive provider does not synthesise.** It quotes; it does not answer
  a question the corpus answers only implicitly, or phrase anything for a reader.
* **Nothing here is a legal-accuracy measurement**, and the golden dataset
  remains 0 of 111 human-verified. No code path in Phase 7 writes
  `verified: true`.

---

## 11. Running it

```
.venv/bin/python scripts/answering/smoke_test.py                     # ~0.1 s
.venv/bin/python scripts/answering/smoke_test.py --stale-insufficient
.venv/bin/python scripts/evaluation/benchmark_answers.py             # fixtures only
.venv/bin/python -m pytest tests/test_answering.py -q                # 74 tests, ~0.15 s
```

None of these loads a model, opens a network connection, or starts a background
process.
