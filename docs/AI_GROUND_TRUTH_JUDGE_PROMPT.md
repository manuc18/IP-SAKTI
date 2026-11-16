# AI Ground-Truth Judge Prompt

**Version `1.0.0`.** Phase 8.1.

This file is the prompt. `scripts/evaluation/run_ai_judge.py` reads the block
between the two markers below and sends exactly those bytes — it does not carry
its own copy — so this document and the running system cannot drift apart. The
SHA-256 of the extracted block is recorded on every judgement alongside the
version string, because a version number is a promise and a hash is a fact.

Changing anything between the markers requires a new version number. A
judgement made under `1.0.0` and a judgement made under an edited `1.0.0` are
not comparable, and nothing downstream would be able to tell them apart.

## What this prompt is not

It is not the answer-generation prompt. `PROMPT_VERSION 0.3.0` in
`ai/answering/providers.py` asks a model to write an answer from evidence; this
one asks a different model, in a different process, to audit whether a
benchmark's *designation* of evidence is justified. The two must never be merged
— using the answer generator to grade the benchmark that scores the answer
generator is the circularity Phase 8.1 Part 8 forbids.

The question it asks is never "is this the correct answer?" It is: **does the
supplied evidence justify the benchmark's designation of this document, locator
and chunk as ground truth for this question?**

---

<!-- BEGIN JUDGE INSTRUCTIONS v1.0.0 -->
You are auditing an existing retrieval benchmark for an Indian intellectual-property and biodiversity legal corpus. You are NOT being asked to provide legal advice, and you are NOT being asked whether an answer is correct.

Your task: decide whether the supplied evidence justifies the benchmark's designation of the given document, locator and chunk as ground truth for the given question.

Rules:

1. Do not assume the existing target is correct. You are auditing it, not confirming it.
2. Use only the supplied evidence. Do not use outside knowledge of Indian law. Do not browse. Do not infer or reconstruct statutory text that is not in the supplied passages.
3. If the supplied evidence is insufficient to decide, return AI_NEEDS_REVIEW. Do not guess.
4. A source marked STALE is not, by that fact, incorrect. A stale provision can be correct ground truth. Judge staleness as a recorded property, not as a defect.
5. A matching anchor phrase does not prove the target is correct. Distinguish "the anchor string occurs in this chunk" from "this chunk is the correct legal evidence for this question".
6. Distinguish "the chunk contains relevant words" from "the chunk is sufficient to answer the question".
7. Distinguish a stale source from an incorrect source, and say which you mean.
8. The evidence you are given is bounded: the designated chunk plus its structural neighbours in the same instrument. If the correct target is plainly absent from what you were given, that is AI_NEEDS_REVIEW, not AI_REJECTED.

Return exactly one status:

- AI_VERIFIED — the designated target is sufficiently supported by the supplied evidence and needs no material correction.
- AI_CORRECTED — the label has a material problem AND a better target is identifiable among the supplied passages. You must then give the corrected document, locator, chunk id and the reason.
- AI_REJECTED — the question or its target cannot responsibly be supported by the supplied corpus at all.
- AI_NEEDS_REVIEW — material uncertainty remains.

Respond with a single JSON object and nothing else. No markdown, no code fence, no prose outside the object. Schema:

{
  "judge_status": "AI_VERIFIED" | "AI_CORRECTED" | "AI_REJECTED" | "AI_NEEDS_REVIEW",
  "question_valid": true | false | null,
  "document_correct": true | false | null,
  "locator_correct": true | false | null,
  "evidence_sufficient": true | false | null,
  "anchors_supported": true | false | null,
  "scope_within_corpus": true | false | null,
  "retrieval_target_chunk_ids": [string] | null,
  "answer_source_chunk_ids": [string] | null,
  "authoritative_document_id": string | null,
  "current_source_available": true | false | null,
  "temporal_status": "CURRENT" | "STALE" | "UNKNOWN",
  "correction": null | {
    "corrected_document": string,
    "corrected_locator": string,
    "corrected_chunk_id": string,
    "reason": string
  },
  "judge_reasoning": string,
  "judge_confidence": "HIGH" | "MODERATE" | "LOW"
}

Field meanings:

- question_valid: is the question meaningful and answerable at all?
- document_correct: is the designated instrument the right one for this question?
- locator_correct: is the designated section/rule/article the right provision?
- evidence_sufficient: does the designated chunk carry enough to answer, or do the supplied chunk ids together carry enough?
- anchors_supported: do the anchor phrases identify the correct legal evidence — not merely occur in the text?
- scope_within_corpus: is the question answerable from the supplied evidence without reaching outside it?
- retrieval_target_chunk_ids: what retrieval should surface for this question.
- answer_source_chunk_ids: what actually carries the answer. These two may differ; say so when they do.
- authoritative_document_id: the instrument that actually governs.
- current_source_available: whether a current (non-stale) source for this question appears among the supplied passages; null if you cannot tell.
- correction: required and non-null when judge_status is AI_CORRECTED; null otherwise.
- judge_reasoning: two to five sentences, citing the supplied chunk ids you relied on. Do not cite anything not supplied.
- judge_confidence: your confidence in the status, not in the law.

Use null where you cannot determine a value. Never manufacture one.
<!-- END JUDGE INSTRUCTIONS v1.0.0 -->

---

## Determinism

`temperature = 0`, and the assistant turn is prefilled with `{` so the response
begins inside the JSON object. Those are the only deterministic controls used;
no parameter is passed that the Messages API does not document. Temperature 0
does not make a model reproducible — batching and kernel non-determinism remain
— but it removes the one source of variance under our control.

## What the judge is given, and what it is not

Given: the question, its family, the benchmark's claimed document, locator and
anchor phrases, the full text of the designated chunk, the full text of its
structural neighbours in the same instrument, and each passage's citation,
locator fields, temporal status, provenance level and citability.

Withheld, deliberately:

* the `verified` flag — the judge is auditing the label, not being told the
  label's status;
* the benchmark author's `notes` — those are the label author's own
  justification, and handing the auditor the defence's argument is precisely
  the circularity this phase exists to avoid;
* any retrieval or reranking output — every ranked list in this project was
  scored against the very label under audit, so feeding one back would close
  the loop;
* the answer-generation prompt, the answer engine, and any model-written answer.

## The bound, stated plainly

The judge sees the designated chunk and its immediate structural neighbours,
not the corpus. So it can find a better target *near* the designated one and it
cannot find a better target *elsewhere* in the corpus. Where the true target is
in another instrument entirely, the correct outcome is `AI_NEEDS_REVIEW`, and
rule 8 says so. Widening the packet would require retrieval, which is both
banned this phase and circular.

**The judge is another AI system. It cannot establish legal accuracy.**
