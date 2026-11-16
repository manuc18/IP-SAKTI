# Phase 7 Freeze

**Read-only.** This file records the state of the answer layer at the moment
Phase 8 began. It rewrites nothing. No Phase 7 result file, artifact, version
string or golden label was altered to produce it, and none may be altered to
make a later phase's numbers look better — a frozen record that gets edited when
it becomes inconvenient is not a record.

Written: 2026-08-31. Phase 8 Part 1.

---

## 1. Git state

```
HEAD    621a8ebcf3df02500f9f44ed88cd7340f15901c5  "Retrival and Reranking layer"
branch  rag
```

Nothing was committed to produce this freeze. The whole answer layer is still
working-tree state, and that is deliberate: the freeze describes what exists,
not what has been blessed.

### Modified tracked files

| file | phase that changed it |
|---|---|
| `ai/reranking/cross_encoder.py` | Phase 6 |
| `ai/retrieval/base.py` | Phase 6 |
| `ai/retrieval/identifiers.py` | **Phase 7.6** — hyphen normalisation + instrument-title fix |
| `docs/RERANKING_BENCHMARK.md` | Phase 6 |
| `requirements.txt` | Phase 6/7 |
| `scripts/ingest/chunk.py` | Phase 2/6 |
| `scripts/ingest/citation.py` | Phase 2/6 |

`ai/retrieval/identifiers.py` is the only tracked file Phase 7 modified.

### Untracked files belonging to Phase 7

**Answer layer** — `ai/answering/`: `__init__.py`, `base.py`,
`citation_validator.py`, `groq.py`, `grounding.py`, `models.py`, `policy.py`,
`providers.py`, `render.py`.

**Scripts** — `scripts/answering/__init__.py`, `scripts/answering/smoke_test.py`,
`scripts/evaluation/benchmark_answers.py`,
`scripts/evaluation/phase73_five_questions.py`,
`scripts/evaluation/phase75_two_requests.py`,
`scripts/evaluation/phase77_q027.py`,
`scripts/evaluation/phase78_q085.py`.

**Tests** — `tests/test_answering.py`, `tests/test_answer_contract_74.py`,
`tests/test_answer_contract_76.py`, `tests/test_caution_78.py`,
`tests/test_citation_collision.py`, `tests/test_groq_provider.py`,
`tests/test_grounding_ellipsis.py`, and the seven fixtures in
`tests/fixtures/answering/`.

**Documentation** — `docs/ANSWER_ENGINE.md`, `docs/ANSWER_GENERATION_CONTRACT.md`,
`docs/ANSWER_PROVIDER.md`.

### Phase 7 artifacts

| file | what it records |
|---|---|
| `answers.fixtures.jsonl` / `.summary.json` | offline fixture run, no network |
| `phase7.3_five_questions.dry_run.json` | offline dry run |
| `phase7.3_five_questions.json` / `.md` | 5 live requests |
| `phase7.5_two_requests.json` / `.md` | 1 live request (budget 2, stopped at 1) |
| `phase7.7_q027.json` / `.md` | 1 live request |
| `phase7.8_q085.json` / `.md` | 1 live request |

All under `data/evaluation/results/answering/`. **Frozen.**

---

## 2. Component versions

| component | version | source |
|---|---|---|
| answering layer | `0.1.0` | `ai/answering/models.py: ANSWERING_VERSION` |
| prompt / answer contract | `0.3.0` | `ai/answering/providers.py: PROMPT_VERSION` |
| policy | `0.3.0` | `ai/answering/policy.py: POLICY_VERSION` |
| citation validator | `0.2.0` | `ai/answering/citation_validator.py: VALIDATOR_VERSION` |
| grounding | `0.2.0` | `ai/answering/grounding.py: GROUNDING_VERSION` |
| retrieval | `0.1.0` | `ai/retrieval/base.py: RETRIEVAL_VERSION` |

`GROUNDING_VERSION` has not moved since Phase 7.4. That is the point: every
Phase 7 failure was answered by changing what the model was asked to write, and
never by changing what the gate would accept.

---

## 3. Latest answer-engine status

`ANSWER_LAYER_STATUS: PROVISIONAL` (set in Phase 7.6, unchanged).
`ANSWER_E2E_STATUS: PASS` (Phase 7.8, on a sample of one question).
`F1_STATUS: IMPROVED` — not proven solved.

---

## 4. Live requests and outcomes

**Nine live Groq requests have been made in the project's history.** Every one
is accounted for below. No retries have ever been issued; no fallback provider
exists.

| # | phase | question | prompt | outcome |
|---|---|---|---|---|
| 1 | 7.2 | `--live` smoke test | 0.1.0 | connectivity confirmed |
| 2 | 7.3 | Q001 | 0.1.0 | not published — `citation_validation_failed` |
| 3 | 7.3 | Q078 | 0.1.0 | not published — provider abstained (**429 rate limit**) |
| 4 | 7.3 | Q027 | 0.1.0 | not published — `ungrounded_answer` (F1) |
| 5 | 7.3 | Q104 | 0.1.0 | not published — provider abstained (**429 rate limit**) |
| 6 | 7.3 | Q085 | 0.1.0 | not published — provider abstained (**429 rate limit**) |
| 7 | 7.5 | Q027 | 0.2.0 | not published — `citation_validation_failed`; run stopped, Q001 never requested |
| 8 | 7.7 | Q027 | **0.3.0** | **published**, 1 sentence, validation PASS, grounding 1.0 |
| 9 | 7.8 | Q085 | **0.3.0** | **published**, 1 sentence, validation PASS, grounding 1.0, cited rank 4 |

Three of the five Phase 7.3 "abstentions" were the free-tier token-per-minute
cap, not model behaviour. They are not evidence about the prompt.

**Two of two under prompt 0.3.0.** Two observations, not a rate.

---

## 5. Known unresolved issues

1. **No question is human-verified.** 0 of 111. Nothing in the project measures
   legal accuracy, and no published answer has been read by a lawyer. This is
   what Phase 8 exists to begin fixing.
2. **F1 is improved, not solved.** Two single-sentence answers complied with the
   1–4 sentence contract. Rule 3 permits four sentences and no question has yet
   needed more, so whether a three- or four-sentence answer still carries a
   marker on every sentence is **untested**.
3. **The sample is two questions.** Both were selected for specific structural
   properties (identifier lookup; all-stale evidence). Neither is a random draw
   from the 111.
4. **Rate limiting is unmodelled.** The free tier's 8000 TPM cap silently
   converted three Phase 7.3 questions into abstentions. Any future batch run
   will hit it again, and the layer has no retry, by design.
5. **Retrieval and reranking metrics rest on unverified labels.** Every recall
   and MRR figure in `docs/RERANKING_BENCHMARK.md` and the Phase 5/6 artifacts
   is computed against ground truth no human has confirmed. Those numbers are
   provisional.
6. **Answer quality is unmeasured.** Citation validity and grounding are
   structural properties. Neither is a claim about whether the answer is right.

---

## 6. Statement

No Phase 7 result is being rewritten. The artifacts listed in §1 are read-only
for the remainder of the project; a later phase that disagrees with one of them
writes a new file and says why, rather than editing the old one.

**No claim of legal accuracy. No claim of production readiness.**
