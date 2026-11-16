# AI-Judged Ground-Truth Validation

Phase 8.1 built this; Phase 8.2 ran the pilot; Phase 8.3 completed the
benchmark. **`AI_JUDGE_STATUS: COMPLETE`** — **111 of 111 questions judged**,
by the Claude Code session rather than by the Anthropic API.
**Zero network requests have been made.**

**AI-audited coverage is 111/111. Human-verified coverage is 0/111.**
Those are different measurements and the second one has not moved.

Phase 8.1 ended at `BLOCKED` because the API path needs a separate Anthropic
credential that does not exist on this machine. Rather than acquire one, Phase
8.2 added a second execution path — see **Claude Code execution mode** below.
The API runner is unchanged and remains available.

> **AI verification provides a deterministic second-pass audit of the benchmark
> labels.** It does not prove the golden dataset is correct.

**The judge is another AI system.** It cannot establish legal accuracy
independently of human or legal review, and no number of `AI_VERIFIED` labels
adds up to one verified by a lawyer. The benchmark remains subject to legal
review if this project ever gets a qualified expert.

---

## 1. Why human verification was replaced

Phase 8 built a human-review workflow for 111 questions across 22 Indian
statutes, rules and treaties. Verifying one question means reading a provision
and deciding whether it is the right law for the question — work only a lawyer
can do, and this project has no access to one. The workflow was correct and
unusable, which is the worst combination: it would have sat at 0 of 111
indefinitely while every retrieval metric downstream carried on quoting
unverified labels.

An AI judge is a weaker instrument and an available one. It replaces "nobody has
checked" with "a second model, given only the corpus, was asked whether the
label holds up" — a real narrowing of the unknown, and not the same thing as
verification.

**Phase 8 is preserved, not deleted.** `prepare_ground_truth.py`,
`review_ground_truth.py`, `golden_verification.jsonl` and
`golden_review_queue.jsonl` are untouched and still work. If an expert ever
becomes available, the human workflow is there, and the AI judgements will be a
useful triage order for it.

---

## 2. Limitations

1. **The judge is an AI.** Everything below is an audit of internal consistency
   between a question, a label and a passage. None of it is a finding of law.
2. **The judge sees a bounded packet**, not the corpus: the designated chunk
   plus its structural neighbours in the same instrument, 4–6 passages. It can
   find a better target beside the designated one; it cannot find one in a
   different instrument. Prompt rule 8 routes that case to `AI_NEEDS_REVIEW`.
3. **Widening the packet would require retrieval**, which is banned this phase
   and would be circular — every ranked list this project can produce was scored
   against the label under audit.
4. **Temperature 0 is not reproducibility.** Batching and kernel
   non-determinism remain. It removes the one source of variance we control.
5. **The corpus is the ceiling.** A label wrong in the same direction as the
   corpus — a mis-chunked provision, an OCR error — is invisible to a judge whose
   only evidence is that corpus.
6. **All 111 are judged, none are verified.** `AI_VERIFIED` records that an AI
   auditor, given the corpus, found a label internally supported. Five of the
   111 (Q001, Q027, Q078, Q085, Q104) are questions this session had previously
   seen through the answer layer — see the independence caveat under
   **Claude Code execution mode**.
7. **The judge and the corpus share a blind spot.** Both the label and the
   auditor derive from the same extracted text, so an error in extraction —
   a truncated section, a dropped clause — is invisible to this audit. Q093
   is the case where it became visible only because the missing content was
   the answer itself.

---

## 3. Status vocabulary

Never `verified`. The four statuses carry the `AI_` prefix in the schema, in the
prompt, in the validator and in the stored records, and a test asserts a stored
record contains no field called `verified`.

| status | meaning |
|---|---|
| `AI_VERIFIED` | the designated target is sufficiently supported by the supplied evidence; no material correction needed |
| `AI_CORRECTED` | material problem **and** a better target identifiable among the supplied passages; requires corrected document, locator, chunk id and reason |
| `AI_REJECTED` | the question or its target cannot responsibly be supported by the supplied corpus |
| `AI_NEEDS_REVIEW` | material uncertainty remains |

`AI_NEEDS_REVIEW` is a first-class outcome, not a failure mode. The offline
fixture that exercises the parser returns it, prompt rules 3 and 8 route
uncertainty into it, and `validate_judgement` accepts it with every judgement
field null. Forcing uncertain cases into `AI_VERIFIED` is the single most
likely way this phase could produce a worse benchmark than it started with.

---

## 4. Schema

`ai_ground_truth_schema = 1.0.0`. One record per question in
`data/evaluation/ai_judged_ground_truth.jsonl`.

The judge returns the verdict fields; the script adds the provenance fields —
the judge cannot write its own attribution.

**From the judge:** `judge_status`, `question_valid`, `document_correct`,
`locator_correct`, `evidence_sufficient`, `anchors_supported`,
`scope_within_corpus`, `retrieval_target_chunk_ids`, `answer_source_chunk_ids`,
`authoritative_document_id`, `current_source_available`, `temporal_status`,
`correction`, `judge_reasoning`, `judge_confidence`.

**Added by the script:** `question_id`, `original_document_id`,
`original_locator`, `original_chunk_id`, `original_chunk_ids`,
`corpus_fingerprint`, `golden_dataset_sha256`, `packet_sha256`,
`judge_provider`, `judge_model`, `judge_temperature`, `judge_prompt_version`,
`judge_prompt_sha256`, `judged_at`, `diagnostics`.

`null` is a valid value everywhere and means the judge could not determine it.
The prompt's closing line is "Use null where you cannot determine a value. Never
manufacture one," and `validate_judgement` rejects a string where a tri-state
belongs, so `"probably"` is a parse failure rather than a stored answer.

`retrieval_target_chunk_ids` and `answer_source_chunk_ids` are separate fields
because they are separate things. Phase 7.8's Q085 is the demonstration: the
answer came from the rank-4 passage while rank 1 was the retrieval target.

---

## 5. The prompt

`docs/AI_GROUND_TRUTH_JUDGE_PROMPT.md`, version **`1.0.0`**, SHA-256 of the
instruction block **`aa15a9c6d9b4…`**.

That document *is* the prompt. `run_ai_judge.py` extracts the bytes between
`<!-- BEGIN JUDGE INSTRUCTIONS v1.0.0 -->` and the matching end marker and sends
exactly those; it keeps no second copy, refuses to run if the document's version
disagrees with its own, and records the hash on every judgement. A version
number is a promise; a hash is a fact.

---

## 6. Anti-circularity

Part 8's rule, implemented rather than asserted:

* **Different question.** Never "is this the correct answer?" Always "does the
  supplied evidence justify the benchmark's *designation* of this document,
  locator and chunk as ground truth?"
* **Different model and provider.** Anthropic `claude-opus-5`, against Groq
  `openai/gpt-oss-120b` for answering.
* **Different prompt.** Judge `1.0.0`; answer contract `0.3.0`. Neither file
  reads the other.
* **No import edge.** Neither judge script imports `ai.answering` — not even for
  the eight-line dotenv reader, which is duplicated instead. A test parses the
  ASTs rather than grepping, because both modules discuss the answer layer in
  prose.
* **No retrieval output.** Neighbouring passages come from document order, never
  from a ranked list. Every ranked list in this project was scored against the
  label under audit.

**Withheld from the judge**, and a test asserts each is absent from all 111
packets: the `verified` flag, `ground_truth_method`, and the benchmark author's
`notes`. That last one is the label author's own argument for the label — Q001's
note reads "s.3(p) is the traditional-knowledge exclusion; it sits inside the
single s.3 chunk" — and handing the auditor the defence's case is exactly the
circularity this phase exists to avoid.

---

## 7. Packet design

`scripts/evaluation/prepare_ai_judgement.py` → `data/evaluation/ai_judge_packets.jsonl`.
Packet version `1.0.0`, 111 packets, offline.

| | |
|---|---|
| passages per packet | 4–6, mean 5.0 |
| packet size | 4.7–20.7 KB, mean 10.4 KB |
| neighbour span | ±2 chunks, same instrument, document order |
| golden SHA-256 | `23024a3d…558fe2f1` |
| corpus fingerprint | `be9d648d9cc1c0b0` (3073 chunks) |
| missing targets | 0 |

Each passage carries its chunk id, citation, set locator fields, unit type,
heading, temporal status, provenance level, source status, citability, pages and
full text, tagged `target` or `neighbour`.

**Packets contain no timestamp and no random draw**, so the file is byte-stable
across runs — verified by regenerating and diffing. Each packet carries
`packet_sha256` over its own canonical JSON, which is what makes "the judge saw
exactly this evidence" checkable rather than merely asserted.

A target absent from the corpus is written into the packet as
`missing_from_corpus: true` and rendered to the judge as "THIS CHUNK IS NOT
PRESENT IN THE CORPUS." Dropping it would have shown a target-less question as
though the corpus simply had nothing to say.

---

## 8. Determinism

This section describes the **API path only**. The Claude Code path records
`judge_temperature: null`, because a session is not invoked through a
temperature parameter and claiming 0 would be false; see **Claude Code
execution mode**.

`temperature = 0`, and the assistant turn is prefilled with `{` so the response
begins inside the JSON object. Those are the only controls used — both
documented Messages API behaviour, no invented parameter.

`max_tokens = 1500`. Timeout default 90 s, clamped to 120 s; the environment may
lower it and may not raise it. `parse_response` puts the prefilled brace back and
calls `json.loads` once: **there is no repair pass**, because a response that
needs coaxing into shape is not one we can call deterministic.

The recorded `judge_model` is the identifier the API reports back, not the one
requested. A judgement attributed to a model that did not produce it is worse
than an unattributed one.

---

## 9. Checkpoint and resume

`scripts/evaluation/run_ai_judge.py`. **Dry run is the default; `--live` is the
only way past it.** `_post` is the one function that opens a socket, it has one
call site, and it is unreachable on the offline path — asserted by a test that
monkeypatches both `_post` and `urlopen` to raise and then runs the dry path.

* one question per request, sequential, no batching
* no retry, no fallback model — a failure stops the run and says why
* `--max-requests` defaults to **1**: spending 111 requests requires asking for 111
* `--limit N`, `--start-from QID`, `--question QID`, `--only-status UNJUDGED|ALL`
* the judgement file is rewritten after **every** successful response
* resume skips any question already carrying one of the four statuses, and the
  live loop checks again immediately before requesting

Without a key, `--live` prints that the key is not set and exits 2 having made
no request. That is the path this phase actually took.

---

## 10. Q027 pilot

**Judged by the Claude Code session** (Phase 8.2). `AI_VERIFIED`, confidence
`HIGH`. The API path was not used: `ANTHROPIC_API_KEY` is absent from the
environment and from `.env`, which holds only `GROQ_API_KEY`, `GROQ_MODEL` and
`GROQ_TIMEOUT_SECONDS` — and the Groq key is not usable here, because the judge
must not be the answer generator's model.

Everything Part 11 required before judging was done first:

| pre-flight check | result |
|---|---|
| schema validation | 57 focused tests pass |
| packet generation | 111 packets, 0 refusals |
| prompt rendering | all 111 render; Q027 is 8238 chars |
| JSON parsing test | prefill restored, malformed input recorded not repaired |
| deterministic fixture test | fixture → valid judgement → complete record |
| checkpoint/resume test | completed questions are not reselected |
| no-network test | dry run with `_post` and `urlopen` booby-trapped passes |
| byte-stability | regenerate-and-diff: identical |

**Result:** `AI_VERIFIED`, confidence `HIGH`.

| | |
|---|---|
| packet | `77d7d9d2d0e6c0de…7da42ae9` |
| corpus fingerprint | `be9d648d9cc1c0b0` |
| golden SHA-256 | `23024a3d…558fe2f1` |
| prompt | v1.0.0, `aa15a9c6d9b4…` |
| provider / model | `claude_code_session` / `claude-opus-5` (self-reported) |
| temperature | `null` |
| network requests | 0 |

The target `IN-TM-RULES-2017::part-i-chapter-ii-rule-23` carries locator rule 23
and the heading "Form and signing of application", and its sub-rule (1) states
both halves of the question: the application shall be made in TM-A and shall be
signed by the applicant or his agent. The designation is corroborated from
inside the packet rather than by the anchor alone — neighbour rule 24(3) refers
to "an application ... filed under rule 23(1)". The anchor `'signed'` is a
common word, but here it falls in the operative signing clause and occurs in
none of the four supplied neighbours. Two caveats recorded and judged not to
bear on the designation: provenance level B
(`OFFICIAL_SOURCE_FOUND_LOCAL_COPY_NOT_MATCHED`) and OCR spacing damage in the
extracted text.

The API path remains available and unrun; the command, if a key ever exists:

```
ANTHROPIC_API_KEY=... .venv/bin/python scripts/evaluation/run_ai_judge.py \
    --question Q027 --live --max-requests 1
```

**Exact network requests made in Phase 8.1 and 8.2: 0.**

---

## Claude Code execution mode

Phase 8.2. The judge is the Claude Code session that is doing the work, not an
API client.

**No separate Anthropic credential was acquired, and none is needed.** That was
a deliberate choice, not a workaround for a locked door: the audit is a
reasoning task over a 10 KB packet, and the thing already reading the repository
can do it. `scripts/evaluation/record_claude_code_judgement.py` imports no HTTP
client, names no endpoint, reads no key, and a test asserts all three by parsing
its imports rather than grepping its prose.

**This is not an API call, and the record says so.**

| field | value | why |
|---|---|---|
| `judge_provider` | `claude_code_session` | never `anthropic` — different provenance, and a record that blurred them would be unreproducible with nothing downstream able to detect it |
| `judge_temperature` | `null` | a session is not invoked through a temperature parameter; writing `0` would assert a determinism control that was never applied |
| `judge_model` | `claude-opus-5` | the session's own identity, recorded because it is exposed. `--judge-model` defaults to `"unknown"`, which is the correct value where nothing is exposed |
| `diagnostics.judge_model_source` | `session_self_report` | the identity comes from the session's context, not from a server response that could corroborate it |
| `diagnostics.claude_code_elapsed_time` | `null` | no honest measurement of the session's reasoning time is available. It is never called model latency |
| `diagnostics.api_request_made` | `false` | |
| `diagnostics.requests` | `0` | |
| `diagnostics.passes` | `1` | |

**No token counts, no latency, no `stop_reason`.** Those are properties of an
API call that did not happen; a test asserts none of them appears on a record
built through this path. The Phase 8.1 defaults are untouched, so a judgement
made through the API runner still records `anthropic` and `0.0`.

**One packet, one judgement, one pass.** The session read the Q027 packet and
the frozen v1.0.0 instructions once and wrote one JSON object. No candidate was
generated and discarded, no judgement was critiqued and rewritten, and no second
opinion was sought — any of which would have made this an undocumented
multi-pass judge whose recorded provenance described something simpler than what
actually happened.

**The packet is the evidence boundary.** The judgement cites four supplied chunk
ids and nothing else; a test extracts every chunk id from `judge_reasoning` and
asserts the set is a subset of the packet's own. No retrieval was run, no
ranking consulted, no Phase 7 answer read, no error analysis opened, no author
note sought, and no outside knowledge of the Trade Marks Rules used — the
corroborating cross-reference in the reasoning comes from rule 24(3) *inside the
packet*.

**Anti-circularity holds across both paths.** The answer generator is Groq
`openai/gpt-oss-120b` under prompt `0.3.0`; the judge is a Claude session under
judge prompt `1.0.0`. Different system, different prompt, different question —
"does the evidence justify this *designation*", never "is this answer correct".

### The independence caveat, stated plainly

An API judge starts every question with no memory. This session does not. It
carried Q027 through Phases 7.3, 7.5 and 7.7 end-to-end, so of the 111
questions, Q027 is the **worst** one for the independence of a session-as-judge
— it is the question this session knows most about from outside the packet.

The reasoning recorded is drawn from the packet and is checkable against it: the
locator fields, the heading, the operative text of rule 23(1), the absence of
the anchor from all four neighbours, and the cross-reference in rule 24(3). But
"I did not use what I already knew" is not something a test can assert, and it
should not be reported as though it were. Prior exposure is a real weakening of
this particular pilot, and it does not apply to the other 110 questions, of
which this session has seen only Q001, Q078, Q085 and Q104 through the answer
layer.

### What the pilot does and does not establish

It establishes that the execution path works end to end: a packet is read, the
frozen prompt is applied, a structured judgement is produced, validated by the
same validator the API path uses, checked against three pinned hashes, and
checkpointed. It establishes that Q027's label survives an audit conducted
against its own evidence.

It establishes nothing about the other 110 questions, and nothing about Indian
trade mark law.

---

## Frozen provenance manifest

Phase 8.3. `data/evaluation/ai_judge_manifest.json`, written once beside the
packets and never rewritten: golden SHA-256, corpus fingerprint, packet version,
and the SHA-256 of all 111 packets.

`preflight` already asked whether a packet hashed to its own recorded hash and
whether the corpus and golden file on disk were the ones it named. A packet file
and a corpus that moved **together** would have passed all of that. The manifest
is the only check that catches it, and it is what turns "judged against exactly
this evidence, from exactly this corpus and exactly this benchmark version" into
a verifiable claim rather than one the code makes about itself. The judge prompt
hash is pinned the same way, as a literal in the recorder.

Six refusals, each with its own test: a packet hash that left the manifest, a
moved corpus, a moved golden dataset, a changed packet version, a question
absent from the manifest, and a prompt that is not the frozen v1.0.0. Rebuilding
the packets against a manifest that disagrees **refuses** rather than
re-freezing — a manifest that re-stamps whatever is on disk today is not a
seatbelt.

## First batch — Q001 to Q010

Ten questions, judged one packet at a time, checkpointed after each. All ten are
Patents Act, 1970.

| | |
|---|---:|
| AI_VERIFIED | 10 |
| AI_CORRECTED | 0 |
| AI_REJECTED | 0 |
| AI_NEEDS_REVIEW | 0 |
| confidence HIGH / MODERATE / LOW | 8 / 2 / 0 |

`question_valid`, `document_correct`, `locator_correct`, `evidence_sufficient`,
`scope_within_corpus` and `current_source_available`: true on all ten.
`anchors_supported`: true on nine, **false on one**. Temporal status CURRENT on
all ten. No corrections, so no clustering by instrument to report.

**Q004 — the anchor that names a topic instead of identifying evidence.**
Section 84 is the right target and the label is right, but `'compulsory licence'`
occurs twice in the designated chunk *and* twice in the neighbour section 85, so
within the packet the anchor does not discriminate. `anchors_supported: false`
under `AI_VERIFIED` is the distinction prompt rule 5 asks for, and it is the
signal that would have been lost by recording a bare status.

**Q007 — relevant words that are not the right evidence.** The neighbour section
105 is headed "Power of court to make declaration as to non-infringement" and so
matches the question's wording more literally than the target does, but it
governs who may seek a declaration rather than who bears the burden of proof.
Section 104A stands.

**The two MODERATE confidences are both about the packet bound, not the label.**
Q004 is a broad "when may" question against a five-passage window, and section
84 visibly continues past the designated chunk. Q010's exclusion is defined by an
onward reference to section 20(1) of the Atomic Energy Act, 1962, which is not
among the supplied passages — the yes/no is supported, the precise class of
excluded inventions is not checkable.

**Zero `AI_NEEDS_REVIEW`, and that is a weak result, not a strong one.** These
ten are the easiest questions in the benchmark: one instrument, all CURRENT, all
mechanically clean, most with a heading that restates the question. The bound
was not tested, so nothing here shows whether the packet is too restrictive.

**What this does not show.** Nothing about the other 21 instruments; nothing
about the 39 stale-source or 35 weak-anchor questions the Phase 8 mechanical
pass flagged; nothing about Indian patent law. Ten agreements between an AI
auditor and a label, both derived from the same corpus.

---

## Full-run results — all 111 questions

| result | count |
|---|---:|
| `AI_VERIFIED` | **105** |
| `AI_CORRECTED` | 0 |
| `AI_REJECTED` | 0 |
| `AI_NEEDS_REVIEW` | **6** |

Confidence: HIGH 96, MODERATE 15, LOW 0.

Criteria across all 111: `question_valid` and `document_correct` true
everywhere; `locator_correct` true 110, null 1; `evidence_sufficient` true 105,
false 6; `anchors_supported` true 92, **false 19**; `scope_within_corpus` true
104, false 6, null 1; `current_source_available` true 72, false 39.

### The six AI_NEEDS_REVIEW

| id | instrument | locator | why |
|---|---|---|---|
| Q012 | Patents Rules 2003 | Rule 7 | fee amounts are in the First Schedule, not the chunk |
| Q028 | TM Rules 2017 | Rule 10 | fee amounts are in the First Schedule |
| Q059 | Copyright Rules 2013 | Rule 83 | fee amounts are in the Second Schedule |
| Q072 | ABS Regulations 2025 | Regulation 6 | fee is in model guidelines, a separate instrument |
| **Q093** | **D&C Rules 1945** | **Rule 2** | **the word "Inspector" does not occur anywhere in the packet** |
| Q105 | Nagoya Protocol | Article 33 | the article states the entry-into-force rule; the question asks for a date that no instrument states |

**Five of the six are one pattern: a question asking for a number the provision
delegates elsewhere.** Four are fee questions across four different instruments
— Patents, Trade Marks, Copyright and ABS — where the designated rule is
correctly the fees rule and says the fee "shall be as specified in the
Schedule". The label is right and the evidence cannot answer the question. Q105
is the same shape with a historical date instead of a figure.

**Q093 is the one substantive label finding.** The question asks how the Drugs
and Cosmetics Rules define an *Inspector*; the designated chunk is rule 2
(Definitions) and defines the Act, the Central Licence Approving Authority, the
Director, Form, Homoeopathic medicines, the Laboratory, a registered
Homoeopathic practitioner, Phytopharmaceutical drug and Registered medical
practitioner — and then breaks off mid-clause. The word "Inspector" appears
nowhere in the packet. The anchor phrase is the single word **`'means'`**,
which occurs seven times in the chunk and identifies nothing. That is exactly
how a mechanical containment check passed this label, and exactly the
distinction prompt rule 5 exists to draw. Returned as `AI_NEEDS_REVIEW` rather
than `AI_CORRECTED` because no better target is identifiable among the supplied
passages.

### Anchors: 19 of 111 do not identify their evidence

`anchors_supported: false` under `AI_VERIFIED` in 18 cases and under
`AI_NEEDS_REVIEW` in one. Measured, not impressionistic: the anchor's occurrence
count in the target was compared with its count in each supplied neighbour, and
recorded false where it did not favour the target.

The failures cluster where a chapter is entirely about one subject, so a topic
word cannot discriminate: `'compulsory licence'` twice in Patents s.84 and
twice in s.85 (Q004); `'examination'` eleven times in Patents Rule 24B and
**twelve** in Rule 24C (Q014); `'drawings'` nine and eight (Q016);
`'application'` nine and nine in GI Act s.11 and s.13 (Q031); `'opposition'` ten
in TM Rule 42 and **sixteen** in Rule 43 (Q025); `'licence'` ten in D&C Rule 61
and **sixteen** in Rule 62A (Q091); `'trade mark'` sixteen in TM Act s.2 and
**forty-three** in the neighbouring s.11 (Q023).

By instrument: GI Act 3, and 2 each in Patents Rules, Copyright Act, PPVFR
Rules, D&C Act and D&C Rules. No instrument is clean of it and none dominates.

### Retrieval target ≠ answer source: 2 cases

| id | designated | carries the answer |
|---|---|---|
| Q023 | TM Act s.2, three chunks | `chapter-i-section-2-c2` only — clause (zb) is in c2; c1 and c3 do not define "trade mark" |
| Q078 | PPVFR Act s.28, two chunks | `chapter-iv-section-28-c1` only — c2 is sub-section (9) on varying and cancelling registration |

Both are the benchmark's only multi-chunk targets, and in both the extra chunks
are correct as *section* members and wrong as *answer* sources. A recall metric
counting any designated chunk as a hit will credit a retriever that returned
only the non-answering chunk. Recorded as a measurement concern, not a label
defect.

### Locator granularity

No wrong locators. Three under-specified ones, all recorded rather than
corrected:

* **Q108** — the label says `Article 2`, but the chunk is **Annex II** Article 2
  (arbitration). Article 2 of the Convention itself is Use of Terms, an entirely
  different provision. The chunk's own locator fields carry `annexure: II`; the
  label's locator string does not.
* **Q001, Q095** — the question names s.3(p) and TRIPS 27.1, the label says
  "Section 3" and "Article 27". The sub-clause sits inside the designated chunk;
  this is the corpus's chunking granularity, not an error.
* **Q111** — "Paragraph 1" for a policy document whose own numbering in that
  chunk reads "2.1", and the chunk begins mid-sentence.

### Stale sources

39 of 111 targets are STALE. **37 were `AI_VERIFIED` and 2 reached
`AI_NEEDS_REVIEW`, neither for staleness** (Q012 fees, Q093 Inspector). Prompt
rule 4 held: staleness was recorded on every affected judgement and never
treated as a defect. Whole instruments are stale as a block — Patents Rules
2003, GI Rules 2002, Designs Rules 2001, Copyright Act 1957, D&C Act 1940, D&C
Rules 1945 and TRIPS — so this is a corpus-vintage property, not a per-label one.

### Where the packet bound was actually felt

Six `AI_NEEDS_REVIEW` plus `scope_within_corpus: false` on Q062 (the question
says "foreign company"; the provision says "person referred to in section 3(2)",
and section 3 is not in the packet). So the ±2-neighbour bound was the binding
constraint on **7 of 111**, and in five of those the missing evidence was a
Schedule that a wider packet could have supplied.

### Corpus observations recorded in passing

Chunks that begin or end mid-sentence (Q111, Q093); section 2 of the
Biological Diversity Act reduced to a single 308-character clause (Q067);
amendment footnotes stored as retrievable chunks (Q085, Q067, Q093); OCR damage
in headings (`RESERV ATIONS`, `DePositary`, `tradem ark`); marginal notes
interleaved into PPVFR Act text (Q074, Q081); garbled sub-rule numbering
(Q083); and one chunk that is four lines of answer followed by a bibliography
(Q110, correctly marked non-citable).

---

## 11. Machine safety

Judged: 111 of 111. Network requests across Phases 8.1, 8.2 and 8.3: **0**.

Anthropic API NO · `ANTHROPIC_API_KEY` NO · Groq NO · OpenAI NO · web NO ·
web search NO · external legal databases NO · BGE-M3 NO · torch NO ·
sentence-transformers NO · GPU/MPS/CUDA NO · local ML model NO · retrieval NO ·
reranking NO · answer generation NO · re-ingestion NO · corpus PDFs unmodified ·
processed chunks unmodified · Phase 8 human-review files unmodified ·
multiprocessing NO · background workers NO · full pytest NO.
Network requests: **0**, across Phase 8.1 and 8.2 together.

One transparency note: importing the recorder pulls `urllib` and `socket` into
`sys.modules`, because it imports `run_ai_judge` for the validator and record
builder that Part 11 asks it to reuse. Nothing calls them — a test runs the
recorder with both `_post` and `urlopen` monkeypatched to raise, and it
passes.

---

## 12. The golden dataset

`data/evaluation/golden_questions.jsonl` is **untouched**, before and after:

```
sha256  23024a3dbbf92e3e32086c762fd3a99ab9ae60bcdf08dda26ca05dea558fe2f1
```

Neither Phase 8.1 script can write to it: every `write_jsonl` call site in both
targets `--packets` or `--judgements`, `run_ai_judge.py` never opens the golden
path at all, and three tests assert it — one of them by regenerating every
packet and re-hashing the file.

**No claim of legal accuracy. No claim of production readiness.**
