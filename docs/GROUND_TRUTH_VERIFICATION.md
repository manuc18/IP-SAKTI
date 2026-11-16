# Ground-Truth Verification

Phase 8. The benchmark has 111 questions and **0 of them have been verified by a
human being.** Everything below is preparation for changing that; none of it
changes it.

> **Retrieval metrics remain provisional until the relevant golden questions are
> human-verified.**

That sentence applies to every recall, MRR and nDCG figure in
`docs/RERANKING_BENCHMARK.md`, `docs/RETRIEVAL_BENCHMARK.md` and the Phase 5, 6
and 7 artifacts. Those numbers measure agreement with a label. Whether the label
is right is exactly what has never been checked.

---

## 1. Counts

| verification state | questions |
|---|---:|
| **UNREVIEWED** | **111** |
| VERIFIED | 0 |
| CORRECTED | 0 |
| REJECTED | 0 |
| NEEDS_REVIEW | 0 |
| **total** | **111** |

| mechanical status | questions |
|---|---:|
| PASS | 39 |
| WARNING | 72 |
| FAIL | 0 |

**Mechanical PASS is not verification.** It says the document exists, the chunk
exists, the locator agrees with the chunk's own metadata and the anchor phrase
occurs in the text. It says nothing about whether that provision is the right
law for that question, and 39 questions being mechanically clean is not 39
questions being right.

---

## 2. What the mechanical checks found

Zero hard failures — the benchmark's references all resolve against the current
corpus. The 72 warnings are structural weaknesses a reviewer has to weigh, not
findings that a label is wrong. Counted by question (a question may carry
several):

| finding | questions | what it means |
|---|---:|---|
| `stale_source` | 39 | the target chunk is `temporal_status: STALE`. A stale source can be correct ground truth for a historical question — the reviewer must say so explicitly rather than let it pass unremarked. |
| `weak_anchor` | 35 | the anchor phrase occurs in more than 50 of the 3073 chunks — more than the retriever's own candidate depth — so it cannot be what distinguishes this chunk from the pool. `'application'` occurs in 965. |
| `weak_provenance` | 16 | the target is evidence level C or D. |
| `short_evidence` | 7 | the target chunk is under 300 characters. Not a verdict: CBD Article 41 is 143 characters and answers "who is the depositary" completely. It is a prompt to confirm rather than assume. |
| `query_identifier_unmatched` | 1 | Q082 names "section 26" but the target carries only Rule 41 — because s.26 is part of the rule's *title*. Innocent here, which is why it is a warning for a person and not a failure. |
| `non_citable_evidence_allowed` | 1 | Q110 waives the citability requirement. |

The weak-anchor threshold is `DEFAULT_LEXICAL_DEPTH`, imported rather than typed
in, so if the retriever's candidate depth changes the threshold follows it.

---

## 3. Review queue

| priority | criterion | questions |
|---:|---|---:|
| 1 | mechanical failure or warning | 72 |
| 2 | target spans multiple chunks | 0 |
| 3 | treaty question | 2 |
| 4 | identifier question | 4 |
| 5 | simple single-chunk question | 33 |

**Band 2 is empty, and that is a finding rather than a bug.** Both multi-chunk
questions — Q023 (three chunks of TM Act s.2) and Q078 (two chunks of PPVFR Act
s.28) — already carry a weak-anchor warning, so they sort into band 1. The band
is kept because the sorting rule is a property of the queue, not of today's
data; `test_a_multi_chunk_target_gets_its_own_review_band` asserts a multi-chunk
question with no warnings lands in band 2.

The queue puts 72 of 111 questions first. That is the intent: the structurally
weakest labels get read while the reviewer is freshest, and if the benchmark has
a systematic problem it shows up in the first afternoon rather than the last.

---

## 4. The verification record

`data/evaluation/golden_verification.jsonl`, one record per question, schema
`1.0.0`. `data/evaluation/golden_questions.jsonl` is **not** modified — it is
read, and its claim is copied into each record's `golden` block so the original
and any correction sit side by side.

Three groups of fields:

**Human judgement — null until a person fills them in.** `question_valid` (A),
`document_correct` (B), `locator_correct` (C), `evidence_sufficient` (D and F),
`scope_within_corpus` (H), `required_chunk_ids`, `primary_chunk_id`, `reviewer`,
`reviewed_at`, `notes`. Every one of these needs the provision read. Nothing
pre-fills them, not even a plausible guess.

**Mechanically established.** `anchors_present` (E) is string containment, which
is all criterion E asks. `temporal_status` (G) is copied from the corpus.
`mechanical_status` and `mechanical_findings` are this run's checks.
`corpus_fingerprint` identifies the chunk set the checks ran against.

**Part 5's distinctions**, all null, because collapsing them is how a benchmark
acquires labels that are subtly wrong:

```
distinctions.retrieval_target_chunk_ids   what retrieval should surface
distinctions.answer_source_chunk_ids      what actually carries the answer
distinctions.authoritative_document_id    the correct instrument
distinctions.current_source_available     whether a current source exists at all
```

Phase 7.8's Q085 is the live demonstration that these differ: the model cited
the **rank-4** passage, because that is where the definition was, while the
rank-1 passage was the retrieval target. A single `expected_chunks` list cannot
express that.

`verified_against_corpus` records the fingerprint a decision was made against.
Human verification is only valid for the corpus it was performed on — re-chunk
and a reviewer's "yes, this answers it" refers to a chunk that may no longer
exist. Recording it is what will let a later phase detect that instead of
silently inheriting approvals.

---

## 5. Verification criteria

A question may be marked `VERIFIED` only if a reviewer confirms **all** of:

| | criterion |
|---|---|
| A | the question is meaningful and answerable |
| B | the designated document is the correct authoritative instrument |
| C | the designated locator is the correct provision |
| D | the cited chunk contains sufficient evidence to answer |
| E | every anchor phrase genuinely occurs in the designated evidence |
| F | the chunk is sufficient, or the record identifies all required chunks |
| G | the temporal status is recorded as CURRENT, STALE or UNKNOWN |
| H | the question is answerable from the corpus, or the gap is recorded |

D and F are asked as one question, because a reviewer answering "is this
enough?" is already deciding "or what else is needed?".

G is recorded, never enforced. **A stale source is not a rejection.** It can be
correct ground truth for a historical question; what is forbidden is letting it
through unremarked.

`review_ground_truth.py` enforces this. Pressing `v` opens the checklist; if any
criterion is answered no, the record cannot become VERIFIED and the reviewer is
asked to file it as CORRECTED, REJECTED or NEEDS_REVIEW instead. A one-key
"looks fine" is how a benchmark acquires 111 green ticks nobody checked.

---

## 6. Tools

```
scripts/evaluation/prepare_ground_truth.py     # build the records and the queue
scripts/evaluation/review_ground_truth.py      # the reviewer's terminal
```

`prepare_ground_truth.py` delegates its hard checks to
`validate_golden_dataset.validate()` rather than reimplementing them, so the two
cannot drift into disagreeing about what a broken reference is. Re-running it is
safe: a record carrying a human decision keeps it, and only its `mechanical_*`
fields are refreshed.

`review_ground_truth.py` is local only — no web, no API, no model, nothing
outside the existing environment. It shows the question, the instrument, the
locator, the full chunk text, the anchors, the provenance line, the mechanical
findings and the neighbouring chunk ids, and writes after every decision so a
reviewer who closes the terminal on question 40 keeps 39 decisions.

```
.venv/bin/python scripts/evaluation/review_ground_truth.py --reviewer "name" --priority 1
```

---

## 7. What Phase 8 did not do

* It did not verify anything. 0 of 111.
* It did not change a single golden label. The file is byte-identical
  (`sha256 23024a3d…558fe2f1` before and after), and a test asserts it.
* It did not adjudicate legal correctness. No LLM was asked whether a provision
  is right, no external legal source was consulted, and no network request was
  made. The system may find a reference mechanically inconsistent; it may not
  decide that a label is legally wrong.
* It did not recompute any retrieval metric.

**No claim of legal accuracy. No claim of production readiness.**
