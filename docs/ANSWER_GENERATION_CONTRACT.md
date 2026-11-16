# Answer generation contract

`ANSWER_LAYER_STATUS: PROVISIONAL` · prompt version `0.3.0` · Phase 7, revised 7.6

This document is the contract a provider must satisfy to generate answers in
IP-SAKTI. Two providers exist. `ExtractiveMockProvider` is the default:
deterministic, offline, quotes the corpus verbatim, and still the only thing
that runs unless a caller asks otherwise. `GroqProvider` (Phase 7.2) implements
this contract against `openai/gpt-oss-120b` over Groq's OpenAI-compatible API —
see `docs/ANSWER_PROVIDER.md`. Seven live requests have now been made — one
smoke test, the five of Phase 7.3 and one of the two Phase 7.5 was allowed —
and no code path reaches the network unless a caller constructs `GroqProvider`
explicitly, passes `--live` to the smoke test, or runs one of the two phase
scripts with `--run`/`--question`. No completion has yet been published as an
answer.

The machine-readable copy of the rules below is `ANSWER_INSTRUCTIONS` in
`ai/answering/providers.py`, and `PROMPT_VERSION` versions it. A prompt is part
of a result's identity in the same way a model revision is: Phase 6 refused to
report a benchmark number without `model_revision`, and an answer produced under
an unrecorded prompt has the same problem.

---

## 1. The ten rules

1. **Answer only from the supplied evidence.** Do not use outside knowledge,
   including your own knowledge of the law.
2. **Answer directly.** Write no introduction, no framing sentence, no
   conclusion, no summary and no preamble such as "Based on the supplied
   evidence:". Begin with the answer and stop when it is finished.
3. **Write 1 to 4 sentences.** Use more only if the question genuinely requires
   them.
4. **Every sentence must state something the supplied evidence directly
   supports, and every sentence must carry a citation marker** written as `[1]`
   or `[2]`. Do not write a sentence you cannot support from the evidence you
   were given.
5. **Put the marker immediately before the sentence's final punctuation**, and
   repeat it on every sentence it supports rather than citing once per
   paragraph.

   > GOOD: "Rule 23 governs the filing of trademark applications [1]. It
   > requires the prescribed TM-A form and the applicant's signature [1]."
   >
   > BAD: "Rule 23 is an important provision concerning trademark applications.
   > The rule governs filing. It requires TM-A [1]." — the first two sentences
   > carry no marker, so the whole answer is rejected and nothing is shown.

6. **If the supplied evidence does not answer the question, say so and stop.**
   Do not fill the gap.
7. **Distinguish current evidence from evidence marked `STALE`**, and
   authoritative evidence from evidence with weak provenance.
8. **Do not silently reconcile conflicting provisions**; state that they differ.
9. **Never invent a citation.**
10. **Never write a page number, locator, URL or document name of your own.**
    The citation layer supplies all of them from the evidence record.

Rules 9 and 10 are not enforced by asking. See §4.

### Where these rules came from, and what replaced what

The contract was rewritten twice, both times because a live request showed the
previous version did not work.

**Phase 7.4 (prompt `0.2.0`) made rule 4 specific.** Phase 7.3 Q027 produced
*"Rule 23 of the Trade Marks Rules, 2017 governs how a trademark application
must be filed."* with no marker, then cited `[1]` on the two sentences that
followed. `check_grounding` counted the opening sentence as an uncited legal
proposition — correctly — and the whole answer was abstained. Rule 4 had read
only "cite every claim", so it was expanded to enumerate the sentence kinds that
count: opening and topic sentences, framing statements, explanatory sentences,
paraphrases, conclusions and summary statements.

**Phase 7.5 tested that on the same question, and it did not work.** Under
`0.2.0` the model wrote nine sentences and cited six. The three it left uncited
were an opening topic sentence, a *"Thus, Rule 23 specifies…"* summary, and a
comparative aside — precisely the categories rule 4 had just finished naming.
Enumerating them changed nothing. **F1 is a persistent behaviour of this model
under an instruction to cite more, not a gap in the wording of that
instruction.**

**Phase 7.6 (prompt `0.3.0`) stopped asking for more citations and asked for
fewer sentences.** Rules 2 and 3 are the fix: an answer of one to four
evidence-bearing sentences has no room for a topic sentence or a summary, so the
uncited sentence is removed rather than forbidden. The model now has to *want*
to write it, and a four-sentence budget spent on the answer leaves nowhere to
put it.

The alternative was to teach `check_grounding` that opening and closing
sentences need no citation. That is the permissive fix: it would publish the
same answer while knowing less about it, and every future uncited claim that
happened to fall first or last would pass. **The gate is correct; the output
shape was wrong.** `tests/test_answer_contract_76.py` asserts that the BAD
example above, and the shape of the actual Phase 7.5 completion, both still fail
grounding — so a future "fix" that relaxed the gate instead would break a test.

Whether `0.3.0` works is unknown. One live request is reserved to find out.

---

## 2. What a provider receives

Exactly one `ProviderRequest`:

```python
ProviderRequest(
    query: str,
    passages: tuple[Passage, ...],   # marker, document, locator, temporal_status,
                                     # source_evidence_level, citable, text
    instructions: tuple[str, ...],   # the twelve rules above
    prompt_version: str,
    policy: Mapping[str, Any],       # the AnswerPolicy that will judge the output
)
```

**What it never receives**, and why each absence matters:

| Absent | Why |
| --- | --- |
| `chunk_id` | A provider that never sees a chunk id cannot emit one. The marker is the only handle it gets, so a fabricated citation has no syntax to be written in. `build_request` raises if a chunk id reaches the payload, checked on every call rather than only in a test. |
| The corpus | Only the passages retrieval selected. Phase 7 §21: privacy, cost and prompt size. |
| Golden labels, expected chunk ids, benchmark metadata | `AnswerEngine.answer(self, query, evidence)` has no parameter one could arrive through — the same structural argument Phase 6 §9 made about the reranker. |
| A retriever | The answer layer performs no retrieval. `ai/answering/base.py` imports no index, no chunk loader and no model. |

`temporal_status`, `source_evidence_level` and `citable` **are** supplied,
because rules 8, 9 and 10 are impossible without them.

---

## 3. What a provider returns

```python
ProviderDraft(text: str, provider: str, provider_version: str, diagnostics: Mapping)
```

Prose, and the name of who wrote it. Nothing else — no metadata, no citation
objects, no pages. Empty text is a valid return and means *I cannot answer from
this*; the engine turns it into an abstention with reason `provider_abstained`.

Citations in the prose are `[n]` markers and nothing else.

---

## 4. Why the rules do not depend on the provider obeying them

A prompt is a request, not a guarantee, and Phase 7 is built so that a provider
which ignores every rule above still cannot publish a bad citation.

| Rule | What actually enforces it |
| --- | --- |
| 9, 10 — no invented citations | Citation objects are built by `citation_validator.build_citation()` from the evidence record. The provider's text contributes a marker; every other field is copied from the corpus. Fabrication is not detected, it is structurally impossible. |
| 9 — no invented provisions in prose | `_scan_prose` parses every provision named anywhere in the answer with the Phase 5 identifier parser. Each must either *be* a supplied passage, be a subdivision of one whose marker is present in that passage's text (§8), or be named by a supplied passage's own text. Anything else fails. |
| 10 — no URLs | Any URL in the answer text is a validation failure. Links come from the provenance registry via the citation record. |
| 2, 3, 4, 5 — every sentence cited | `check_grounding` splits the answer into sentences and reports every claim sentence with no marker. Under the default policy an ungrounded answer is not published. |
| 1, 6 — evidence only | A marker outside the supplied set fails validation; validation failure means the engine abstains rather than publishing a repaired subset. |
| 7 | Rendered independently of the provider: `[STALE]` is attached by the renderer from the citation record, and cautions come from the policy. |
| 8 | Conflicts are detected by `detect_conflicts` before the provider runs, and the caution is attached whatever the provider wrote. |

A provider is not trusted because of what it is. The mock provider's output is
validated exactly as a future model's will be.

---

## 5. Implementing a real provider

Subclass `AnswerProvider` and implement `generate`. That is the entire surface —
retrieval, policy, citation validation, grounding and rendering do not change.

```python
class MyProvider(AnswerProvider):
    name = "my-provider"

    def generate(self, request: ProviderRequest) -> ProviderDraft:
        ...
        return ProviderDraft(text=prose, provider=self.name)
```

A conforming implementation:

* sends only `request.query`, `request.passages` and `request.instructions`;
* never sends the corpus, a chunk id, or any file from `data/`;
* returns prose whose only citations are `[n]` markers drawn from the passages
  it was given;
* records its model identity and revision, as Phase 6 §8 requires of any model
  whose output is reported.

**Cost.** The project's requirement is ₹0 / $0 operating cost, so no paid
dependency may be introduced and no free tier may be assumed permanent. The
system must remain runnable with no external service at all — which today it is,
via the mock provider. An external provider is an optional accelerator behind
this interface, never a load-bearing part of it.

---

## 6. Future service boundary (Phase 7 §24 — documented, not built)

No server exists. No FastAPI dependency has been added. When one is built, the
boundary is:

```
POST /answer
{ "query": "..." }

200
{
  "answer": "...",
  "citations": [
    { "marker": 1, "chunk_id": "...", "document_id": "...", "locator": "...",
      "pages": [12], "citation_text": "...", "source_evidence_level": "A",
      "temporal_status": "CURRENT",
      "source_document": { "filename": "...", "sha256": "...",
                           "official_source_url": null } }
  ],
  "grounded": true,
  "insufficient_evidence": false,
  "confidence": "MODERATE",
  "conflict_detected": false,
  "cautions": []
}
```

This is `Answer.to_dict()` exactly, which is why the shape is already fixed
without a server existing. The handler's whole job is
`retrieve → rerank → GroundedAnswerEngine.answer → to_dict`.

---

## 7. Prompt versioning

`PROMPT_VERSION = "0.3.0"`, recorded on every `ProviderRequest` and in every
benchmark summary. Bump it whenever a rule changes, and never compare answers
produced under different prompt versions without saying so.

| version | phase | what changed |
| --- | --- | --- |
| `0.1.0` | 7.2 | the original twelve rules |
| `0.2.0` | 7.4 | rule 4 enumerated the sentence kinds that must be cited |
| `0.3.0` | 7.6 | ten rules; 1–4 sentences, every one evidence-bearing, no introduction or summary to leave uncited |

Phase 7.3's results were produced under `0.1.0` and Phase 7.5's under `0.2.0`;
neither is comparable to anything produced from here on. `POLICY_VERSION`,
`VALIDATOR_VERSION` and `GROUNDING_VERSION` moved to `0.2.0` in Phase 7.4 and
are unchanged in 7.6 — **no gate was touched in this phase**, which is the whole
point of it.

---

## 8. Safe hierarchical locators (Phase 7.4)

Phase 7.3 Q001 supplied the whole of Section 3 of the Patents Act as one chunk.
The model wrote "section 3(p)" — a real clause **of that very passage** — and
`_scan_prose` reported `unsupported_locator_in_prose`, because
`ref_matches_locator` compares subdivisions exactly and the chunk's locator
carries none. The answer was refused for correctly citing its own evidence.

`ref_matches_subdivision_of` (`ai/answering/policy.py`) adds one arm to the
prose scan:

> A parent locator supports a child subdivision **only if that subdivision is
> actually present in the supplied evidence text.**

Section 3's chunk supports `section 3(p)` because the string `(p)` is in it. It
supports `section 3(zz)` never, because that string is not. The containment is
read off the corpus rather than inferred from the numbering — the same move the
cross-reference arm already makes, and for the same reason: the test is whether
the string came from the evidence.

Two restrictions keep this from becoming an arbitrary-descendant rule:

* **The target must be the whole provision.** A chunk that *is* Section 3(p) is
  a sibling of Section 3(q), not its container, so a chunk carrying any
  subdivision of its own is never treated as a parent here.
* **Every subdivision named must be present**, not merely the first.

**Nothing about the citation changes.** `build_citation` still copies the
evidence record, so the reader is shown the corpus's own locator — `Section 3` —
and never the model's `3(p)`. This rule admits prose; it does not author a
locator, and no stored corpus locator is altered.

The `citable` and `locator_unavailable` invariants are untouched: a clause that
resolves under this rule is still refused if its chunk was never citable, or if
its locator state is inconsistent.

---

## 9. Citation marker normalisation (Phase 7.4)

Phase 7.3 Q001 also wrote its marker as `【1】` rather than `[1]`. `CITATION_MARKER`
is ASCII-only, so the marker was invisible to the validator *and* to the
grounding check, and a correctly sourced sentence was reported as carrying no
citation at all.

`normalise_markers` (`ai/answering/grounding.py`) rewrites recognised equivalent
spellings to `[n]` before anything reads them:

| Written | Read as |
| --- | --- |
| `[1]` | `[1]` |
| `【1】` | `[1]` |

An enumerated table, not a pattern. A rule like "any bracket-ish pair around a
digit" would start admitting punctuation nobody agreed to, and marker syntax is
a contract with the provider rather than a guess about it. Adding a spelling is
a one-line change, made deliberately. The alias is exactly as strict as the
ASCII form — `【 1 】` is rejected because `[ 1 ]` is rejected — since a normaliser
more permissive than its target would be widening the contract while claiming
to translate it. Malformed and mismatched forms (`【a】`, `【1]`, `(1)`) are left
untouched and resolve to nothing.

Normalisation happens **once, in the engine**, and the normalised text is what
is validated, grounded *and published* — so the prose a reader sees and the
citation list beneath it always use the same syntax. `draft.text` is left as the
model wrote it, and `diagnostics["marker_normalisation_applied"]` records
whether the two differ.

---

## 10. Identifier spellings: hyphens (Phase 7.6)

Phase 7.5 Q027 quoted the supplied chunk's own words — *"published by the
registrar under sub-rule (2) of rule 20"* — and was refused for naming a
provision nothing supplied. The chunk parses as `('rule','20',('2',))`. The
model wrote `sub‑rule` with **U+2011 NON-BREAKING HYPHEN**, which
`parse_identifiers` does not read as `sub-rule`, so the same reference parsed as
`('rule','20',())`, the keys differed, and the cross-reference arm of §4 missed.

`normalise_identifier_text()` rewrites three hyphen spellings as ASCII `-`
before either pattern reads the text. It is an enumerated table, and every row
was measured in this project rather than assumed:

| codepoint | name | why it is here |
| --- | --- | --- |
| U+2010 | HYPHEN | the model's typography; the sibling of U+2011 |
| U+2011 | NON-BREAKING HYPHEN | what Q027 actually wrote, 10 times |
| U+2013 | EN DASH | the **corpus's** own spelling — `sub–rule` occurs 4 times in the processed chunks |

Deliberately excluded, also by measurement:

* **U+2014 EM DASH** — 1905 occurrences in the corpus, 41 of them between two
  letters. It is the heading separator of Indian legislative drafting
  ("namely:—"), not a hyphen.
* **U+2212 MINUS SIGN** — 95 occurrences, 2 intra-word; the same separator role
  arriving from a different PDF text layer.
* **Everything else.** "Normalise the dashes" is a rule that keeps growing; a
  table with a reason per row does not.

U+202F NARROW NO-BREAK SPACE needed no entry, and that was checked rather than
assumed: the model emitted 14 of them, and Python's `\s` already matches it.

The one place a hyphen is load-bearing in the parser is
`sub[\s\-]?(?:section|rule|regulation|paragraph)`, so this normalisation is
narrow in consequence as well as in construction.

---

## 11. Instrument titles are not provisions (Phase 7.6)

The same answer wrote *"Geographical Indications Rules 2002"*, which
`_DIRECT` read as a rule numbered 2002.

The distinction drawn is the smallest general one available: **a plural
legislative term, followed by a bare four-digit year, with no parenthesised
tail, is an instrument title.** `Rules 2003` is a title; `Rule 2003` is a
provision; `Rules 2003(2)` is a provision; `rules 23 and 24` is a provision.

Two things it deliberately does not do:

* **It does not decide by number.** A singular term still names a provision
  whatever its number, because a singular term is how a provision is cited and
  this module has no business ruling out a four-digit provision it has never
  seen.
* **It does not look for instrument names.** "Patents", "Trade Marks" and
  "Biological Diversity" are not a closed set, and a rule that recognised them
  would be a document-specific exception wearing a general shape.

The corpus's own convention already escaped this — "Patents Rules, 2003" carries
a comma, which `_DIRECT`'s `\s*` never matched. What is handled now is the
spelling without the comma, which is the one a model writes.

**Measured, not assumed.** Across all 111 golden queries the parser's output is
byte-identical before and after: routing, identifier boost and reranking cannot
move. Across the 3073 processed chunks, 8 changed, and every change removed a
false positive — `rule 1945` (×6, from the Drugs and Cosmetics Rules, 1945),
`rule 2013` and `rule 2003`. Nothing genuine was lost and nothing new appeared.

---

## 12. Completion budget, and why there are still no retries (Phase 7.6)

Q027 was truncated mid-word at `max_completion_tokens = 800`, which is the kind
of observation that usually ends in a raised limit. The budget was taken apart
first:

| | tokens |
| --- | --- |
| prompt — system instructions | ~454 |
| prompt — five evidence passages | ~1777 |
| **prompt total** | **~2231** |
| completion — visible prose, 9 sentences | ~577 |
| completion — per sentence | ~64 |
| completion — unaccounted (this model reasons, and Groq charges reasoning against the same ceiling) | ~223 |

Truncation was a consequence of nine sentences, not of a tight ceiling. Under
the 1–4 sentence contract the visible budget is ~256 tokens, leaving ~540 for
reasoning — more than twice what Q027 used. **800 stays.**

Raising it to 1000 would cost something real in both directions: ~200 more
tokens per request against a free-tier TPM ceiling that already returned three
429s in Phase 7.3, and a larger ceiling is an invitation to write the long
answer whose extra sentences are exactly what goes uncited.

`_completion_meta()` now records `finish_reason` and the token counts Groq
already reports, so the next such decision rests on a measurement instead of on
inferring truncation from a sentence that stopped mid-word. It reads no gate and
changes no outcome; every failure inside it is swallowed.

**No retries, no regeneration, no model switching, no fallback provider, no
multi-pass generation.** A failed answer stays an abstention. This is not
conservatism for its own sake: the free tier has already demonstrated a TPM
ceiling, an automatic retry turns one refusal into two requests, and until the
stricter contract has been observed once there is nothing to retry *toward*.
`tests/test_answer_contract_76.py` asserts the provider module names no retry
machinery and that the engine calls `generate` exactly once.

---

## 13. What these changes do not establish

Phases 7.4 and 7.6 removed five mechanical reasons a correct answer was refused.
They say nothing about whether any answer is legally accurate.

Phase 7.6 in particular must not be read as a solution to F1. It changes what
the model is asked for; whether the model complies is unknown, and one live
request on Q027 is reserved to find out. Nothing in this phase was measured
against the golden dataset, and no request was made in it.
