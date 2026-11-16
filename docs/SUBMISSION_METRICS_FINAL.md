# IP-SAKTI — Submission Metrics

Measured 2026-09-16 from this repository. Every figure carries its numerator and
denominator; the machine-readable copy is
`data/evaluation/results/submission_metrics.json`, assembled by
`scripts/evaluation/build_submission_metrics.py` from the run artifacts rather
than typed in.

**What these numbers are.** Retrieval figures are **evidence recall** — did the
provision come back — not answer accuracy and not legal accuracy. Evidence
sufficiency is a property of the retrieved evidence set. The judged sample is an
**AI audit**, not independent legal verification. **No benchmark in this project
is human-verified legal ground truth** (curated set: 0/111 human-verified).

---

## One table

| | Metric | Value | Denominator |
|---|---|---|---|
| **Corpus** | Source PDFs | 21 | — |
| | Legal instruments | 22 | — |
| | Pages in source PDFs | 1,782 | — |
| | Pages carrying retrievable text | 1,585 | of 1,782 |
| | Evidence chunks | 3,073 | — |
| | Provenance level A / B / C | 1,727 / 1,155 / 191 | of 3,073 |
| | Temporal CURRENT / STALE | 1,356 / 1,717 | of 3,073 |
| | Chunks with extraction damage | 211 (6.87%) | of 3,073 |
| | Instruments with zero damaged chunks | 12 | of 22 |
| **Evaluation** | Curated questions (hand-written) | 111 | — |
| | AI-generated corpus-grounded questions | 1,000 | — |
| | — answerable | 960 | of 1,000 |
| | — expected-abstention probes | 40 | of 1,000 |
| | Distinct target chunks | 975 | of 3,073 |
| | Human-verified questions | 0 | of 1,111 |
| **Retrieval — curated 111** | Recall@1 | 94 (84.68%) | of 111 |
| | Recall@5 | 110 (99.10%) | of 111 |
| | Recall@10 | 111 (100%) | of 111 |
| | MRR | 0.8995 | — |
| **Retrieval — AI-generated 960** | Recall@1, strict | 673 (70.10%) | of 960 |
| | Recall@1, equivalence-adjusted | 767 (79.90%) | of 960 |
| | Recall@5, strict | 839 (87.40%) | of 960 |
| | Recall@5, equivalence-adjusted | 897 (93.44%) | of 960 |
| | Recall@10, strict | 903 (94.06%) | of 960 |
| | Recall@10, equivalence-adjusted | 920 (95.83%) | of 960 |
| | MRR, strict / adjusted | 0.7784 / 0.8552 | — |
| | Median candidate pool | 80 | per question |
| | Reranked evidence returned | 10 | per question |
| | Median retrieval + rerank latency | 56.14 ms | of 960 |
| **Evidence sufficiency** | Sufficient @5 | 853 (88.85%) | of 960 |
| | Sufficient @10 | 911 (94.90%) | of 960 |
| | Insufficient @5 / @10 | 21 / 5 | of 960 |
| | Uncertain @5 / @10 | 86 / 44 | of 960 |
| | Target retrieved but insufficient @5 | 44 | of 960 |
| | Target not retrieved @5 | 63 | of 960 |
| | Sufficient without the exact target @5 | 14 | of 960 |
| | Target in top-5 but not top-1 | 166 | of 960 |
| | — of those, rank 1 was already an equivalent answer | 50 (30.1%) | of 166 |
| **AI-judged sample** | Expected answer supported | 87 (88.8%) | of 98 |
| | — partial / not supported | 1 / 10 | of 98 |
| | Evidence judged sufficient | 69 (70.4%) | of 98 |
| | Rank-1 chunk carried the support | 72 (73.5%) | of 98 |
| | Contradictions found | 0 | of 98 |
| | Judge provider | `claude_code_session` | — |
| | Rank hidden from judge | yes | — |
| **Live generation — 60-question run** | Answered | 41 | of 60 |
| | Abstained | 19 | of 60 |
| | Unavailable | 0 | of 60 |
| | Citation validation passed | 41 | of 41 |
| | Grounding passed | 41 | of 41 |
| | Served from validated-answer cache | 10 | of 60 |
| | Median end-to-end latency | 2,203.8 ms | of 60 |
| **Live generation — cache-disabled run** | Requests issued | 30 | — |
| | Reached the model | 6 | of 30 |
| | Rate-limited (free-tier daily cap) | 24 | of 30 |
| | Answered / abstained | 4 / 2 | of 6 |
| | Citation validation passed | 4 | of 4 |
| | Grounding passed | 4 | of 4 |
| | Unsupported claims published | 0 | of 4 |
| **Safety** | Out-of-domain probes correctly abstained | 15 | of 15 |
| | Invalid citation published | 0 | of 41 |
| | Unsupported claim published | 0 | of 41 |
| | Corrupted source handling | labelled, never repaired | — |
| **Tests** | Passing | 1,050 | of 1,050 |

---

## The five things a reader should not misread

**1. Recall@1 on the AI-generated set is reported twice, and both are true.**
The strict figure (673/960) scores each question against the single chunk the
generator wrote it from. The adjusted figure (767/960) scores the identical run
against every chunk in the corpus that answers the question as completely —
the other half of a split section, a duplicated heading, the Act provision a
Rule's own heading names. The gap is the benchmark's single-label design, not a
change in behaviour. Neither figure replaces the other.

**2. The curated 111 and the AI-generated 960 measure different things.**
850 of the 1,000 AI-generated questions were written *from the target chunk's
heading*, so a question and its answer share vocabulary by construction. The
curated 111 were written by hand. Where the two disagree, the curated set is the
better estimate of behaviour on a real user's question — see the Part G finding
below.

**3. Evidence sufficiency is not accuracy.** It asks whether the top-*k* chunks
contain the material the expected answer is made of, decided by verbatim span
containment and IDF-weighted term coverage against thresholds declared in
`scripts/evaluation/evidence_sufficiency.py`. It makes no claim about any
generated sentence and no claim about the law.

**4. The live generation sample is small and says so.** The cache-disabled run
reached the model 6 times out of 30 before the Groq free tier's 200,000
tokens-per-day cap closed; the remaining 24 are recorded as UNAVAILABLE and
excluded from every rate. The larger 60-question run is reported beside it, with
its 10 cache hits declared. Neither is large enough to carry a tight interval,
and no rate from either is stated as though it were measured over 1,000.

**5. 211 damaged chunks are a property of two source PDFs, not of the
pipeline.** They are detected, labelled and surfaced to the user as a
source-integrity warning. They are never repaired, and no replacement legal text
is ever generated. 12 of the 22 instruments carry no damage at all.

---

## Reproducing this

```bash
# retrieval + evidence sufficiency over all 1,000 questions (about 60 s, no provider)
.venv/bin/python scripts/evaluation/run_submission_eval.py --fresh

# why the target lost rank 1, on the 166 questions where it was inside the top 5
.venv/bin/python scripts/evaluation/rank1_failure_analysis.py

# judge packets, judged batches, and the folded-back metrics
.venv/bin/python scripts/evaluation/build_judge_packets.py --sample 100
.venv/bin/python scripts/evaluation/record_judgements.py --split 14
#   ... each batch_NN.txt is judged and its batch_NN.jsonl written ...
.venv/bin/python scripts/evaluation/record_judgements.py --collect

# the only run that calls a provider
.venv/bin/python scripts/evaluation/live_generation_sample.py --size 30

# assemble
.venv/bin/python scripts/evaluation/build_submission_metrics.py
.venv/bin/python -m pytest -q
```

Full working behind every figure: `docs/METRICS_FULL_EVIDENCE.md`.
One-page extract for the panel: `docs/SIH_METRICS_FINAL.md`.
