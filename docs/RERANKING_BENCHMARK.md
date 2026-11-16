# IP-SAKTI — Phase 6 reranking benchmark

**These are figures on an unverified benchmark.** 111 golden questions exist;
**none has been verified by a human**, and every record carries `verified:
false`. Nothing below is a measurement of legal accuracy. The correct way to
quote any number here is "Recall@1 on the current unverified benchmark".

RERANKING_STATUS: PROVISIONAL — 111 questions, 0 human-verified, chunking 0.2.0.

---

## 1. Objective

Phase 5 left recall high and rank 1 unsettled: routed retrieval reached
Recall@10 0.964 and boosted-hybrid fusion 0.982, but Recall@1 was 0.766. An
answer engine reads from the top, so the question for Phase 6 is not "is the
provision somewhere in the list" but:

> Given a candidate pool that contains the relevant evidence, does the correct
> authoritative provision end up at rank 1?

The primary metric is therefore **MRR**, with **Recall@1** second. Recall@10 is
watched only to confirm nothing is given up for the gain at the top.

Phase 6 also separates two failures that Phase 5 could not tell apart. A query
can fail because the retriever never returned the evidence, or because the
reranker was shown it and put it eleventh. Those are bugs in different modules,
and every number below is reported in a way that keeps them apart.

---

## 2. Candidate generation

`ai/retrieval/candidates.py`. Two modes; the primary experiments run on
`expanded`.

| Mode | Engines | Depth | Mean pool |
| --- | --- | ---: | ---: |
| `expanded` (primary) | BGE-M3 **and** BM25 + identifier boost | 50 each | 77.3 |
| `routed` | whichever engine the Phase 5 router picks | 50 | 50.0 |

Candidates are unioned by `chunk_id`. The pool's native order is a **round-robin
interleave** of the contributing rankings — dense #1, lexical #1, dense #2,
lexical #2 — keeping each chunk's first appearance. That is a merge, not a
fusion: it compares no scores across engines and asserts no shared scale. Phase 6
§1 forbids RRF as the *final* ranking and this stage does not produce one; RRF
remains available as `--candidate-order rrf` for reference.

Of the 77.3 candidates in an average expanded pool, 27.3 come from BGE-M3 alone,
27.3 from the boosted lexical retriever alone, and 22.7 from both. The two
engines disagree about roughly 70% of the pool, which is the same fact Phase 5
reported as a routing dilemma, seen from the other side.

**Nothing is filtered on authority.** Level B, Level C and STALE chunks stay in
the pool (Phase 6 §11), because the question of whether they are retrieved *for
relevance* cannot be answered on a pool they were excluded from. Every candidate
carries `citable`, `temporal_status`, `source_evidence_level`, `document_id`,
`locator`, `locator_fields` and `physical_pages` through unchanged.

---

## 3. Candidate recall — is the evidence available at all?

111 questions, expanded pool, unfiltered.

| Depth | Candidate Recall@K (expanded) | (routed) |
| --- | ---: | ---: |
| 10 | 0.964 | 0.964 |
| 20 | 0.991 | 0.982 |
| 30 | 0.991 | 0.991 |
| 50 | **1.000** | 0.991 |
| full pool | **1.000** | 0.991 |

**At depth 50 the expanded pool contains gold evidence for every one of the 111
questions.** That single fact reorganises the rest of the phase: with candidate
recall at 1.000, *every* remaining failure is a reranking failure, and the
conditional metrics (§7) coincide exactly with the unconditional ones.

By query family, the shortfall at shallow depth is entirely conceptual:

| Family | n | @10 | @20 | @50 |
| --- | ---: | ---: | ---: | ---: |
| identifier (golden label) | 18 | 1.000 | 1.000 | 1.000 |
| conceptual (golden label) | 93 | 0.957 | 0.989 | 1.000 |
| IDENTIFIER (router) | 17 | 1.000 | 1.000 | 1.000 |
| CONCEPTUAL (router) | 74 | 0.973 | 1.000 | 1.000 |
| GENERAL (router) | 20 | 0.900 | 0.950 | 1.000 |

The queries that need pool depth are the ones with no legal anchor — exactly the
family Phase 5 predicted would need it.

**The routed pool is measurably worse**, and by the specific amount Phase 5
predicted. One question (Q066, "What penalty follows a contravention of the
Biological Diversity Act?") has no gold chunk anywhere in its routed pool: the
router sent it to BGE-M3, which never returned section 55 at any depth, while
the boosted lexical retriever ranks it 3rd. In the expanded pool it is there.
That is the concrete case Phase 6 §3 exists for — a routing decision cannot be
repaired downstream, and an expanded pool can.

---

## 4. Reranker models

`ai/reranking/`. All local, all free, no API sees the corpus.

### Shortlisted and benchmarked

| Key | Model | Licence | Params | Disk | Max len |
| --- | --- | --- | ---: | ---: | ---: |
| `minilm` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Apache-2.0 | 22.7M | 92 MB | 512 |
| `bge-base` | `BAAI/bge-reranker-base` | MIT | 278M | 1,134 MB | 512 |
| `bge-m3` | `BAAI/bge-reranker-v2-m3` | Apache-2.0 | 568M | 2,293 MB | 512 |

Resolved commit hashes are recorded in every results file
(`reranker_config.model_revision`): `233902d25c44` for MiniLM, `2cfc18c9415c`
for bge-reranker-base, `953dc6f6f85a` for bge-reranker-v2-m3. A model whose
weights cannot be named is not a reproducible result.

Selection was not by popularity. MiniLM is the honest control on model size — if
22M parameters closes most of the gap, 568M is hard to justify locally.
bge-reranker-v2-m3 is on the list because it is the trained reranking
counterpart of the BGE-M3 retriever already in use, not because it is the
best-known name.

### Considered and rejected

| Model | Reason |
| --- | --- |
| `jinaai/jina-reranker-v2-base-multilingual` | CC-BY-NC-4.0. A non-commercial licence is not a base this project can build on, independent of how the model performs. |
| `Alibaba-NLP/gte-multilingual-reranker-base` | Requires `trust_remote_code=True` — executing code fetched from a model repository at load time, in a pipeline handling a provenance-controlled corpus. |
| Cohere Rerank, Voyage, Jina API | Paid external APIs are forbidden by Phase 6, and sending the corpus to a third party breaks the same rule that keeps embedding local. |

Both rejections are recorded in `ai/reranking/cross_encoder.py::REJECTED` so the
shortlist cannot quietly become "whatever was tried first".

---

## 5. Feature definitions and normalization

`ai/reranking/features.py`. Deterministic, explainable, **untrained** — Phase 6
§25 forbids fitting anything to 111 unverified questions, and nothing here is
fitted.

### Three scores, kept apart

`relevance_score` (query-to-text and query-to-locator evidence only) ·
`authority_score` (source evidence level and temporal status) · `citability` (a
boolean, copied, never scored). A citable Level A provision that does not address
the question is not a better answer than a stale Level C one that does; it is a
worse answer with better paperwork. Authority is applied afterwards as an
explicit policy (F3) so its effect can be measured rather than assumed.

### Features

| Feature | Definition |
| --- | --- |
| `dense_percentile` | BGE-M3 evidence, normalised within the pool; 0.0 if the engine never returned the chunk |
| `lexical_percentile` | BM25+boost evidence, likewise |
| `locator_exactness` | 1.0 the cited provision (incl. named sub-unit) · 0.6 the parent provision · 0.3 a different sub-unit of it · 0.0 otherwise |
| `locator_referenced` | 1.0 when the chunk's *prose* cites the named provision but its own locator is different. **Weight 0** — a §10 diagnostic, not yet a ranking signal |
| `document_match` | the query names this instrument |
| `idf_coverage` | share of the query's IDF mass present in the chunk. IDF weighting is what removes the need for a stopword list |
| `phrase_overlap` | longest run of ≥2 consecutive query tokens the chunk repeats, over query length |

### Weights, declared a priori

From the Phase 5 per-family Recall@1 tables (identifier: boost 0.944 vs BGE-M3
0.611; conceptual: 0.355 vs 0.731), coarsely rounded, written down before any
Phase 6 run and unchanged since. Each table sums to 1.0.

| Feature | IDENTIFIER | CONCEPTUAL / GENERAL |
| --- | ---: | ---: |
| `locator_exactness` | 0.50 | 0.025 |
| `lexical_percentile` | 0.25 | 0.15 |
| `dense_percentile` | 0.10 | 0.60 |
| `document_match` | 0.10 | 0.025 |
| `phrase_overlap` | 0.05 | 0.10 |
| `idf_coverage` | 0.00 | 0.10 |

The family is decided by the Phase 5 deterministic router, so the reranker
inherits an explainable classification rather than adding an opaque one.

### Normalization — the one empirical choice

Two normalisations were written down in advance, both bounded to [0, 1] within
the pool: min-max, and rank percentile `1 - (rank-1)/n`.

The a-priori argument favoured **rank percentile**: min-max is set by two order
statistics, and on an identifier query the boost adds a full best-BM25-score to
one chunk, so the maximum is roughly double the runner-up and everything else is
compressed. Phase 6 §7 says to choose on empirical behaviour, and the measurement
went the other way:

| Normalization | Recall@1 | MRR | conceptual R@1 | identifier R@1 |
| --- | ---: | ---: | ---: | ---: |
| **min-max** (selected) | **0.784** | **0.852** | **0.753** | 0.944 |
| rank percentile | 0.712 | 0.814 | 0.656 | **1.000** |

The reason is about this corpus, not about the constant. Rank percentile discards
the *size* of the gaps, and inside a 50-deep pool the gaps carry most of the
signal: dense rank 1 and rank 12 differ by 0.22 in percentile, which at weight
0.60 is 0.13 — less than the 0.15 that a rank-1 lexical hit contributes on its
own. Under rank percentile, one strong BM25 match outvotes eleven positions of
dense evidence. Min-max keeps the cosine margin. The outlier concern was real but
lands on the lexical feature, which carries 0.15 of the weight, not the dense one
that carries 0.60.

Two caveats, because this is the only choice in the phase made on measurement:
it is a **single global choice between two options declared in advance**, not a
per-question adjustment; and it was made on a benchmark **no human has
verified**. `--normalization rank_percentile` remains available and its full
results are kept as `features.rankpct.*`. z-score was considered and not run —
it is unbounded and assumes a symmetric spread, and BM25 over 3,073 chunks is
heavily right-skewed.

---

## 6. Experimental setup

### Candidate representation (Phase 6 §9)

One canonical form, defined once in `ai/reranking/format.py`, used by every
reranker so that two models are never compared on two different inputs:

```
Document: Patents Act, 1970
Locator: Section 3
Heading: What are not inventions
Text: The following are not inventions within the meaning of this Act, …
```

`Heading` is omitted when the chunk has none. `Document` and `Locator` are
included deliberately: they are the first two things a lawyer reads, and a model
asked "what does Section 3(p) say" cannot prefer the right provision if the
provision number is absent — the *text* of section 3 writes "3." on one line and
"(p)" thirty lines later.

Excluded, and tested to be absent: `chunk_id` (the join key the answer key is
stored under, not a fact about the document), `citable`, `source_evidence_level`,
`temporal_status`, `source_status`, retrieval scores, retrieval path, and every
field of the golden record. Authority metadata is excluded from the text
specifically so that relevance and authority stay separable at the first layer;
it travels in the candidate's metadata where an explicit policy can act on it.

### Reproducing

```bash
.venv/bin/python scripts/evaluation/benchmark_reranking.py --candidate-mode expanded --reranker identity
.venv/bin/python scripts/evaluation/benchmark_reranking.py --candidate-mode expanded --reranker features
.venv/bin/python scripts/evaluation/benchmark_reranking.py --candidate-mode expanded --reranker cross_encoder --cross-encoder bge-m3
.venv/bin/python scripts/evaluation/benchmark_reranking.py --candidate-mode expanded --reranker cross_encoder+authority --cross-encoder bge-m3
.venv/bin/python scripts/evaluation/benchmark_reranking.py --candidate-mode routed  --reranker identity,features
.venv/bin/python scripts/evaluation/benchmark_reranking.py --reranker features --normalization rank_percentile --tag rankpct
.venv/bin/python scripts/evaluation/identifier_probe.py --reranker all --cross-encoder bge-m3
.venv/bin/python scripts/evaluation/measure_latency.py
.venv/bin/python -m pytest tests/ -q
```

Outputs land in `data/evaluation/results/reranking/` — `.jsonl` per-query
records, `.summary.json` aggregates, `.errors.json` failure analysis, plus
`identifier_probe.json` and `latency.json`.

---

## 7. Metrics

Same definitions as Phase 4 (`scripts/evaluation/metrics.py`): binary relevance,
Recall@k as hit-rate, nDCG@k against the ideal ranking for that question, MRR
over the first expected chunk to appear.

Phase 6 adds two:

**Candidate Recall@K** — was the gold evidence in the pool's first K? Reported
separately from every ranking metric, because a reranker cannot promote what it
was never shown.

**Conditional MRR** — MRR over exactly the queries whose gold evidence reached
the pool. This is the reranking-quality figure. On the expanded pool it equals
the unconditional MRR, because candidate recall is 1.000; on the routed pool it
does not, and the difference is the candidate-generation deficit.

---

## 8. Results

111 questions, top-k 10, unfiltered over the whole corpus, chunking 0.2.0,
expanded candidate pool (50 + 50). **Zero human-verified.**

| Exp | Configuration | R@1 | R@5 | R@10 | MRR | nDCG@5 | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| — | Phase 5 `dense` (BGE-M3 alone) | 0.712 | 0.910 | 0.964 | 0.807 | 0.823 | 0.842 |
| — | Phase 5 `routed` | 0.766 | 0.928 | 0.964 | 0.843 | 0.857 | 0.869 |
| **F0** | expanded + identity | 0.712 | 0.946 | 0.964 | 0.818 | 0.847 | 0.852 |
| **F1** | expanded + features | 0.784 | 0.946 | 0.973 | 0.852 | 0.869 | 0.878 |
| **F2a** | expanded + `minilm` | 0.721 | 0.937 | 0.955 | 0.812 | 0.841 | 0.846 |
| **F2b** | expanded + `bge-base` | 0.766 | 0.955 | 0.991 | 0.842 | 0.866 | 0.877 |
| **F2c** | expanded + **`bge-m3`** | **0.874** | **0.982** | **0.991** | **0.926** | **0.936** | **0.939** |
| **F3** | expanded + `bge-m3` + authority policy | — | — | — | — | — | — |
| F3′ | expanded + features + authority policy | 0.784 | 0.946 | 0.973 | 0.852 | 0.869 | 0.878 |
| A | routed + identity | 0.766 | 0.928 | 0.964 | 0.843 | 0.857 | 0.869 |
| A′ | routed + features | 0.730 | 0.946 | 0.973 | 0.829 | 0.852 | 0.861 |
| — | expanded + features (rank percentile) | 0.712 | 0.946 | 0.973 | 0.814 | 0.842 | 0.851 |

**F3 was not measured.** The run was killed roughly ten minutes in when Phase 6
compute was halted, and it wrote no output. The row is left empty rather than
filled by inference: the authority policy is a bounded tiebreak on min-max
normalised base scores, so a plausible guess is that it changes little, but a
guess is not a measurement and this table carries none. What *is* measured is
F3′, the same policy over the feature reranker, and there it changed nothing at
all — every figure identical to F1 to four decimals, because λ = 0.05 is smaller
than the score gaps it would have to close. That is evidence about the policy,
not about `bge-m3`.

Conditional MRR equals MRR for every expanded-pool row (candidate recall 1.000).
For the routed pool, identity's conditional MRR is 0.850 against an unconditional
0.843, and features' 0.836 against 0.829 — the gap is the one question whose
evidence the routed pool never contained.

**Three things this table settles.**

*Reranking is worth doing.* `bge-m3` over the expanded pool reaches Recall@1
0.874 and MRR 0.926, against 0.712 / 0.818 for the same pool left in its native
order. That is **+0.162 Recall@1 and +0.108 MRR from reranking alone**, on an
identical candidate set — the comparison F0 exists to make.

*Against the previous best system,* Phase 5's router, the gain is +0.108
Recall@1 and +0.083 MRR, and it costs nothing at depth: Recall@10 rises from
0.964 to 0.991 and Recall@5 from 0.928 to 0.982.

*Model size matters more than model family here.* MiniLM (22M) barely moves
anything, `bge-reranker-base` (278M) matches Phase 5's router, and only
`bge-reranker-v2-m3` (568M) produces the step change. There was no way to know
that without running all three, which is why all three were run.

**F0 is the row that makes the rest legible.** The identity baseline over the
expanded pool scores exactly BGE-M3's Recall@1 (0.712) — inevitably, since the
interleave puts the dense first hit at position 1 — while gaining Recall@5
(0.946 vs 0.910) from the lexical list mixed in behind it. Any gain above 0.712
in the F1/F2 rows is reranking, not the wider pool.

---

## 9. Query-family results

Golden-dataset labels (`identifier` 18, `conceptual` 93):

| Configuration | conceptual R@1 | conceptual MRR | identifier R@1 | identifier MRR |
| --- | ---: | ---: | ---: | ---: |
| F0 identity | 0.731 | 0.822 | 0.611 | 0.796 |
| F1 features | 0.753 | 0.829 | 0.944 | 0.972 |
| F1 features (rank pct) | 0.656 | 0.778 | **1.000** | **1.000** |
| F2a minilm | 0.699 | 0.802 | 0.833 | 0.861 |
| F2b bge-base | 0.720 | 0.812 | **1.000** | **1.000** |
| F2c **bge-m3** | **0.860** | **0.917** | 0.944 | 0.972 |

Router families, for `bge-m3` (the split with no golden label behind it):

| Router family | n | R@1 | R@5 | MRR | nDCG@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| IDENTIFIER | 17 | 0.941 | 1.000 | 0.971 | 0.978 |
| CONCEPTUAL | 74 | 0.865 | 0.987 | 0.923 | 0.933 |
| GENERAL | 20 | 0.850 | 0.950 | 0.900 | 0.913 |

**Which families benefit.** Conceptual queries benefit enormously from the
cross-encoder: 0.731 → 0.860 Recall@1, the family that is 93 of the 111
questions and the one BGE-M3 alone was already best at. Identifier queries
benefit from the *feature* reranker rather than the model — the rank-percentile
feature configuration gets 18 of 18 at rank 1, as does `bge-reranker-base`.

**Which regress.** The identity baseline regresses badly on identifier queries
(0.611, below Phase 5's routed 0.944), because the interleave hands rank 1 to the
dense engine on every query including the ones the boost was built for. That is
a property of the pool ordering, not of any reranker, and it is exactly why F0
exists as a control rather than as a candidate architecture.

**The one real trade-off in the table.** `bge-m3` loses one identifier question
against the feature reranker's rank-percentile configuration (17/18 vs 18/18)
while gaining twelve conceptual ones. With 18 identifier questions, one question
is worth 0.056, so the identifier column should be read as "both work" rather
than as a ranking.

---

## 10. Identifier handling (Phase 6 §10)

The specific worry §10 raises is that a cross-encoder will rank a *discussion* of
section 3 above section 3(p) itself, because an argumentative passage looks more
like a question than a terse provision does. Recall@1 over 18 questions cannot
see that, so `scripts/evaluation/identifier_probe.py` measures it directly.

For the 17 questions that name a provision, every candidate in the pool is
classified by its **own locator metadata** — not by the golden label:

`exact` the provision named (incl. its sub-unit) · `parent` the whole provision
when a sub-unit was named · `sibling` a different sub-unit of it · `referenced` a
different provision whose prose cites the named one · `unrelated` everything else.

An average pool holds 8.9 exact, 0.9 parent, 3.0 referenced and 70.4 unrelated
candidates.

| Reranker | queries with an exact chunk in pool | exact at rank 1 | queries where a weaker class outranked every exact chunk |
| --- | ---: | ---: | ---: |
| identity (pool order) | 15 | 10 | **5** |
| features | 15 | **15** | **0** |
| `bge-m3` | 15 | **15** | **0** |

**The §10 failure mode is real, and both rerankers fix it.** In the raw
interleaved order, five queries put an `unrelated` chunk above the cited
provision — Q023, Q044, Q067 (all "section 2" definition queries), Q086 ("section
33EEB") and Q102 ("article 7") — in every case the dense engine's first hit
displacing an exact locator match sitting at position 2. Both rerankers place an
exact chunk at rank 1 in all 15 queries where one exists.

A `referenced` chunk never reached rank 1 under either reranker, which is the
narrower claim §10 asks about: the systems distinguish "this chunk *is* section
3" from "this chunk *mentions* section 3".

Two of the 17 queries (Q001, Q095) have **no** exact chunk in the corpus at all,
because the query names a clause the chunker did not split out — Q001 asks about
section 3(p) and the corpus holds section 3 as one chunk. The feature reranker
puts the `parent` chunk at rank 1 for both, which is the correct answer available;
`bge-m3` gets one of the two.

---

## 11. Error analysis

`scripts/evaluation/rerank_errors.py`, at rank 5, over the expanded pool.

| | F0 identity | F1 features | F2c bge-m3 |
| --- | ---: | ---: | ---: |
| failures | 6 | 6 | **2** |
| **A** — evidence absent from the pool | 0 | 0 | 0 |
| **B** — in the pool, ranked too low | 6 | 6 | 2 |
| **C** — outranked by chunks carrying the same locator | 0 | 0 | 0 |
| **D** — label looks questionable (overlay) | 0 | 0 | 0 |

**100% of expanded-pool failures are reranking failures; 0% are
candidate-generation failures.** That follows from candidate recall being 1.000
and is the cleanest result in the phase: candidate generation, at depth 50 over
two engines, is solved for this benchmark.

Type D is reported as an **overlay, not a fourth bucket**. "The gold label is
questionable" is a statement about the benchmark; "the pool missed it" is a
statement about the retriever; a query can be both, and collapsing them would
hide one. Across the phase, one question was flagged — Q029, where the golden
dataset says `identifier` and the deterministic router says `CONCEPTUAL`. Phase 5
reached the same conclusion about the same question by a different route. It is
queued for human review and **nothing in the golden dataset was altered**.

The routed pool produces the one Type A failure in the phase (Q066), discussed in
§3.

### The two remaining `bge-m3` failures, and what they have in common

> **Q062** "What approval must a foreign company obtain before accessing Indian
> biological resources?" Gold is BDA section 19, pool position 11, reranked to
> 10. Above it sit BDA Chapter II, sections 4 and 6, and ABS Regulation 8 — all
> genuinely about the same approval regime.
>
> **Q093** "How does the Drugs and Cosmetics Rules define an Inspector?" Gold is
> rule 2 (the definitions rule), pool position 32, not reranked into the top 10.

Both gold chunks are **longer than the model's 512-token window**: 603 tokens for
BDA s.19 and 532 for D&C rule 2. The reranker scored a truncated version of each.
Across the whole run, 3,069 of 8,577 pairs (35.8%) were truncated and 1.08M
tokens were dropped; the longest pair was 1,662 tokens. That does not prove
truncation caused these two misses, but it is the one property both share and it
is the first thing to test next (§15).

---

## 12. Latency

Measured by `scripts/evaluation/measure_latency.py`: fresh instances, warmed on
queries outside the benchmark, one stage at a time, machine otherwise idle.

**Not measured. There is no `latency.json`.**

The clean pass was never run: Phase 6 compute was halted after roughly two hours
of sustained load on the machine, and this measurement was still queued. No
number is substituted from the per-row timings in the results files — the note
below explains why those are not a measurement, and the honest consequence of
that note is a blank section rather than a laundered one.

To take the measurement, on an otherwise idle machine, one model per invocation:

```
.venv/bin/python scripts/evaluation/measure_latency.py --queries 30 --stage candidates --stage identity --stage features
.venv/bin/python scripts/evaluation/measure_latency.py --queries 30 --stage cross_encoder --cross-encoder minilm
.venv/bin/python scripts/evaluation/measure_latency.py --queries 30 --stage cross_encoder --cross-encoder bge-base
.venv/bin/python scripts/evaluation/measure_latency.py --queries 30 --stage cross_encoder --cross-encoder bge-m3
```

Each writes `latency.json`, so the runs must be taken one at a time and the file
kept between them; running all three cross-encoders in a single invocation is
supported but leaves the earlier models' weights resident on the Metal device
while the later ones are timed, which is the same contention hazard one process
down.

Until that has run, the only defensible statements about cost are ordinal, and
they come from the architecture rather than from any measurement: candidate
generation is one dense encode plus one BM25 scan over 3,073 chunks; `identity`
is a sort; `features` is arithmetic over metadata already present in the pool;
and a cross-encoder is a forward pass over every one of the ~77 (query, passage)
pairs, which is the dominant term. §13's parameter counts bound that term's
shape. They do not measure it, and nothing in this document should be read as
though they did.

> **A measurement hazard, recorded like Phase 5's.** The per-row `latency_ms` in
> the results files is **not** a clean measurement and must not be quoted. Some
> Phase 6 experiments ran as concurrent background jobs, and two processes
> contending for the same Metal device inflate each other: the stored p95 for
> `bge-m3` is roughly seven times its stored median, which is contention rather
> than tail behaviour. Phase 5 recorded the same trap one level down, where a
> shared `DenseRetriever` query cache made a later retriever's latency into cache
> hits. Quote §12, not the JSONL.

---

## 13. Model and resource requirements

| Model | Params | Weights in memory (fp32) | Disk (HF cache) | Device |
| --- | ---: | ---: | ---: | --- |
| `minilm` | 22.7M | 90.9 MB | 92 MB | MPS |
| `bge-base` | 278M | 1,112 MB | 1,134 MB | MPS |
| `bge-m3` | 568M | 2,272 MB | 2,293 MB | MPS |

Batch size 16, max sequence length 512, fp32, truncation `longest_first` on the
(query, passage) pair — the passage is longer than the query in every observed
case, so the provision's tail is what is dropped.

> **Correction.** The disk column above was wrong in the first draft, and the
> stored `model_disk_mb` in every results file written before 2026-08-30 still
> carries the wrong figure. `_cache_size_bytes()` walked the Hugging Face cache
> with `follow_symlinks=True`, counting each file once under `blobs/` and again
> through the `snapshots/<revision>/` symlink that points at it — exactly double.
> The recorded values were 183.6 / 2,268.8 / 4,586.5 MB; the true sizes are
> 91.8 / 1,134.4 / 2,293.3 MB. The function now skips symlinks. Nothing else
> read that number, so no ranking or latency result depends on it.

`bge-reranker-v2-m3` is the accuracy winner and it is **expensive on this
machine**: 2.3 GB on disk, 2.3 GB of weights resident, and seconds rather than
milliseconds per query at pool depth ~77. §15 says what to do about that. Nothing
here has been quantised or converted; both are open follow-ups with their own
accuracy questions, and doing either silently would break the model-identity
recording §4 depends on.

---

## 14. Leakage controls

Three layers, all in force on every run.

**Structural.** A `BenchmarkQuery` — question id, query text, filters — is built
before the expected answer is read, and candidate generation and reranking take
that object and nothing else. `Reranker.rerank` has the signature
`(self, query, candidates, top_k)`; there is no parameter through which a golden
label could arrive, and `Filters` still has no `document_id` field.

**Representational.** The candidate format (§6) is the only thing a model reads.
It carries no `chunk_id`, no authority metadata, and no field from the golden
record.

**Audited.** `leakage_report()` runs on **every question of every experiment**,
not only in tests, and its result is stored in each summary as `leakage_audit`.
It checks that no rendered candidate contains its own `chunk_id`; that no
forbidden metadata field is rendered; that no label outside
`Document/Locator/Heading/Text` appears in the header; and that no expected chunk
id, expected document id, or golden `notes` string appears in the query or in any
rendered candidate. **Every run in this document audited clean on all 111
questions.**

Two things are deliberately *not* forbidden, and the reason matters. An expected
**locator** may appear — Phase 6 §18 permits it "if it is literally present in
the user query", and the chunk's own citation is part of the canonical
representation for the reason §6 gives. **Anchor terms** are validated by
`validate_golden_dataset.py` to occur in the expected chunk's text, so forbidding
them would forbid the corpus from containing itself; they are evidence the ground
truth was *derived* from the corpus, not a label about it.

---

## 15. Human-verification limitations

**No question in this benchmark has been verified by a human.** Every number
above rests on ground truth derived from the corpus by rule — locator agreement
and anchor-phrase occurrence — and read by no lawyer. `--verified-only` still
evaluates nothing, by design.

That matters more at Phase 6 than it did at Phase 5, for a specific reason: the
headline gain here is large (+0.162 Recall@1), and a large gain on unverified
ground truth is exactly the shape a systematic labelling error produces. The
honest reading is that `bge-reranker-v2-m3` agrees with the corpus-derived labels
much more often than the alternatives — not that it is more legally correct.

`scripts/evaluation/review_reranking.py` exists to make the check cheap. It shows
one screen per question: the query, the expected chunk with its citation,
citability, evidence level, temporal status and pages, the anchor phrases, and
the ten ranked candidates with their text and scores. A reviewer records
`correct`, `incorrect`, `multiple`, `ambiguous` or `remove`, with a note and their
name. **It cannot mark anything verified**: there is no code path in it that
writes `verified: true`, the golden dataset is opened read-only, decisions go to
a separate review file, and each record states `grants_verified_status: false`.
Promoting reviews into the golden dataset stays a deliberate human act.

### Other limitations

**18 identifier questions is a small sample.** One question is worth 0.056.

**The A/B/C/D counts are cheap because candidate recall is 1.000.** On a harder
corpus or a shallower pool the A/B split would carry real weight; here it
degenerates to "everything is B", and that should not be mistaken for a
well-tested classifier.

**One global choice was made on measurement** — the feature normalisation (§5).
It is a single bit selected between two pre-declared options, but it was selected
on unverified ground truth.

**The cross-encoders were not evaluated on non-English queries**, though two of
the three are multilingual and that is part of why they were shortlisted.

**35.8% of scored pairs were truncated at 512 tokens.** Both remaining `bge-m3`
failures are on chunks over that limit. The benchmark does not currently measure
performance at a longer window.

---

## 16. Recommended architecture

```
query
  ↓
candidate generation — BGE-M3 top-50  ∪  BM25+identifier-boost top-50
  ↓   (union by chunk_id, round-robin interleave; ~77 candidates; recall 1.000)
cross-encoder rerank — BAAI/bge-reranker-v2-m3, max_len 512, batch 16
  ↓
top-10 reranked evidence, each carrying document_id, locator, citable,
temporal_status, source_evidence_level, physical_pages, retrieval path
```

The authority policy is **not** in the recommended path. The reason is F3′ in
§8, not F3: over the feature reranker the policy changed no metric at all, so
there is measured evidence that it does not help and none that it does. F3 —
the same policy over `bge-m3` — was never run, so this is a decision taken
without its most relevant experiment, and §17 records it as provisional on that
basis.
Citability and temporal status ride along, labelled, for the answer layer to
enforce; Phase 6 §11 keeps them out of retrieval and out of relevance.

The open engineering question is cost, not accuracy: see §12 and §13. The
candidate stage is cheap and the reranker is not, so the levers are pool depth,
sequence length, batch size, and quantisation — each measurable, none guessed.

Phase 6 ends here: `QUERY → CANDIDATES → RERANKED EVIDENCE`. No answer
generation, no citations in prose, no API, no UI.

---

## 17. Decision report

Phase 6 §26 asks fourteen questions. Each is answered below from completed runs
only. Three answers are **"not measured"** — F3, Mode A × cross-encoder, and the
latency pass were cut when compute was halted, and nothing here is interpolated
across those gaps. Every number rests on 111 corpus-derived questions, **none of
which a human has verified**; read §15 before quoting any of them.

**1. What candidate-pool depth gives sufficient recall?**
Depth 50 per engine. Candidate recall on the expanded pool is 0.964 at 10, 0.991
at 20, 0.991 at 30 and **1.000 at 50**, over a pool averaging 77.3 candidates
(58–93). Depth 20 would cost two questions and depth 10 four. The routed pool at
the same depth reaches 0.991, not 1.000. Caveat: 1.000 is a ceiling measured on
the same 111 questions the rerankers are then scored on, so it establishes that
the pool is not the binding constraint *for this set* — not that 50 is sufficient
for questions nobody has asked yet.

**2. Does the feature reranker beat identity?**
Yes. R@1 0.712 → **0.784** (+0.072, eight questions), MRR 0.818 → **0.852**
(+0.034), R@10 0.964 → 0.973, R@5 unchanged at 0.946. The gain is almost entirely
one family: identifier R@1 0.611 → 0.944, conceptual only 0.731 → 0.753. The
identifier probe says the same thing from the other side — the count of questions
where a locator-`unrelated` chunk outranked the exact cited provision goes from
5 to 0, and exact-at-rank-1 from 10/15 to 15/15. That is the behaviour the
locator features were declared a priori to produce, and it is where they produced
it.

**3. Does the cross-encoder beat identity?**
`bge-m3` and `bge-base` yes, `minilm` essentially not. R@1: identity 0.712,
`minilm` 0.721, `bge-base` 0.766, **`bge-m3` 0.874**. MRR: 0.818 / 0.812 /
0.842 / **0.926**. `minilm` gains 0.009 R@1 while *losing* MRR, R@5 and R@10 — at
22.7M parameters it is the honest control on model size, and it says size is
doing real work here rather than the reranking framing alone.

**4. Does the cross-encoder beat the feature model?**
`bge-m3` yes, decisively: R@1 0.784 → **0.874** (+0.090, ten questions), MRR
0.852 → **0.926** (+0.074), R@5 0.946 → 0.982, R@10 0.973 → 0.991. `bge-base`
does not (R@1 0.766 vs 0.784) and `minilm` does not (0.721). The gap is on
conceptual questions — 0.753 → 0.860 — which is precisely where a hand-declared
feature vector has nothing to offer beyond the dense score it is already
weighting. On identifier questions the two are level at 0.944.

**5. How much does reranking improve Recall@1?**
**+0.162 absolute**, 0.712 → 0.874, best configuration against identity on the
same pool — eighteen of 111 questions moved into rank 1. Against Phase 5's
published baselines: +0.162 over dense-alone (0.712) and +0.108 over the routed
hybrid (0.766). Identity-on-the-expanded-pool coincides with dense-alone at 0.712
by construction, not coincidence: the round-robin interleave places the dense
top hit at pool position 1, so an identity reranker cannot differ from BGE-M3 at
rank 1.

**6. How much does it improve MRR?**
**+0.108**, 0.818 → 0.926 against identity; +0.119 over Phase 5 dense-alone
(0.807) and +0.083 over routed (0.843). Conditional MRR is identical to MRR on
every expanded-pool row, because candidate recall is 1.000 and the two
definitions coincide.

**7. Does it hurt Recall@5/10?**
Not for the recommended configuration: `bge-m3` improves both (R@5 0.946 →
0.982, R@10 0.964 → 0.991). It does for `minilm`, which drops R@5 to 0.937 and
R@10 to 0.955 — a weak reranker can and here does push correct evidence *out* of
a window it was already inside. `features` is neutral at 5 and +0.009 at 10. The
§26 concern is therefore real and model-dependent, and it is the reason R@5/R@10
are reported on every row rather than only the headline metric.

**8. Which query families benefit?**
Both, from different rerankers. By dataset family: identifier (18 q) benefits
most and earliest — 0.611 → 0.944 from features alone, with the cross-encoder
adding nothing on top. Conceptual (93 q) benefits almost entirely from `bge-m3`
— 0.731 → 0.753 from features, → 0.860 from the cross-encoder. By router family:
CONCEPTUAL (74 q) 0.743 → 0.865, GENERAL (20 q) 0.650 → 0.850, IDENTIFIER (17 q)
0.647 → 0.941. GENERAL, the family the Phase 5 router serves worst, gains the
most in relative terms.

**9. Which query families regress?**
One, and only between the two best systems: router-family IDENTIFIER goes
1.000 (features) → 0.941 (`bge-m3`), a single question out of 17. Nothing
regresses against the identity baseline for `bge-m3`. For `minilm`, conceptual
R@5 and R@10 both regress against identity. The IDENTIFIER regression is one
question on a 17-question sample — worth **0.059** — and should not be treated as
an established effect; it is, however, the specific thing a hybrid would be built
to fix, and §15's small-sample warning applies to it more than to anything else
in this document.

**10. What percentage of failures are candidate-generation failures?**
**0%** on the expanded pool, for every reranker, at k=5. Not one Type A failure:
candidate recall at pool depth is 1.000, so the evidence was always present. On
the routed pool it is 12.5% for identity (1 of 8) and 16.7% for features (1 of
6) — the same single question, Q066, whose gold chunk BGE-M3 never returns at any
depth and which only the boosted lexical arm recovers.

**11. What percentage are reranking failures?**
**100%** on the expanded pool — 6 of 6 for identity and features, 2 of 2 for
`bge-m3`, all Type B. `minilm` is the one system with a Type C failure (a
competing chunk carrying the expected locator outranked the gold). The honest
qualification, already in §15: with candidate recall at 1.000 the A/B partition
degenerates, so "100% reranking failures" is a restatement of the pool's
completeness, not an independently informative classification.

**12. Is the latency acceptable?**
**Not measured — this question is not answered.** `latency.json` does not exist;
see §12. The per-row timings in the results files were taken from overlapping
background jobs contending for one Metal device and are not a measurement of
anything. Refusing to answer is the correct outcome here rather than a gap to be
apologised for: an acceptability judgement built on contended numbers would be
wrong in an unknown direction and would then be quoted as though it were not.
What is known without measurement is structural — the cross-encoder is a forward
pass over ~77 pairs per query and dominates a candidate stage that is one encode
plus one BM25 scan — and §13's 568M parameters bound its shape.

**13. Which model/configuration should become the Phase 7 default?**
On accuracy, unambiguously: **expanded pool (50 + 50, interleaved) →
`BAAI/bge-reranker-v2-m3` @ revision `953dc6f6f85a`, max_len 512, batch 16,
fp32, MPS, no authority policy**, exactly as §16 draws it. It wins every metric
in §8 and it wins on both families.

This recommendation is **provisional in three specific ways**, and each is a
measurement that was cut, not an unknowable:
* Latency is unmeasured (Q12). If the clean pass shows seconds per query, the
  decision reopens as a cost decision — not because `bge-m3` became less
  accurate, but because the alternatives become admissible. `features` costs
  arithmetic and delivers +0.072 R@1; that is the fallback the trade would name.
* F3 is unmeasured. The authority policy is excluded on F3′'s evidence
  (no effect over the feature reranker), not on its own.
* Mode A × cross-encoder is unmeasured, so "expanded beats routed" is
  established for identity and for features, and assumed for the cross-encoder.
  It is a reasonable assumption — the routed pool's recall ceiling is 0.991
  against 1.000 — but it is an assumption.
* Separately, 35.8% of scored pairs were truncated at 512 tokens and both
  remaining `bge-m3` failures are on chunks over that limit. A longer window is
  the first accuracy experiment to run next, ahead of any new model.

**14. What additional human-labelled benchmark data is required?**
The largest single gap is not more data — it is **verification of the data that
exists**. Zero of 111 questions have been checked by a person, and the headline
result is a +0.162 gain on labels derived from the corpus by rule, which is the
exact shape a systematic labelling error produces. In priority order:
1. **Human verification of all 111 existing questions**, via
   `scripts/evaluation/review_reranking.py`. Nothing else on this list is worth
   doing first, and no metric in this document should be described as accuracy
   until it is done.
2. **Adjudication of the 6 identity / 2 `bge-m3` failures specifically**, since
   a benchmark where the winner misses two questions is decided by whether those
   two labels are right.
3. **Questions with more than one correct provision.** Every question here has a
   single expected chunk; real legal questions frequently do not, and R@1 against
   a single-chunk label systematically punishes a system that surfaces an equally
   valid sibling. `review_reranking.py` already records a `multiple` decision for
   exactly this.
4. **More identifier questions.** 18 is a sample in which one question is worth
   0.056, and it currently carries the whole feature-reranker argument.
5. **Non-English queries.** Two of the three cross-encoders are multilingual and
   that is part of why they were shortlisted; none was tested that way.
6. **Questions whose correct answer is non-citable or Level B/C evidence**, to
   test §11's separation of relevance from authority against something other
   than the Level A statutory text that dominates the current set.

---

## Status

`RERANKING_STATUS: PROVISIONAL`

Provisional, not PASS, for reasons that are recorded rather than hedged: no
question in this benchmark has been verified by a human; three planned
experiments (F3, Mode A × cross-encoder, the clean latency pass) were not run;
and the ground truth is corpus-derived. The measured claim is that
`bge-reranker-v2-m3` agrees with corpus-derived labels far more often than the
alternatives do. **No claim is made here about legal accuracy.**
