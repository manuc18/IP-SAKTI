# IP-SAKTI — Phase 5 retrieval routing

**These are figures on an unverified benchmark.** 111 golden questions exist;
**none has been verified by a human**, and every record carries `verified:
false`. Nothing below is a measurement of legal accuracy. The correct way to
quote any number here is "Recall@1 on the current unverified benchmark".

Phase 5 did three things: repaired two classes of wrong citation locator in the
corpus, added a deterministic query router above the existing retrievers, and
re-measured everything on the repaired corpus. No reranker, no answer generation.

---

## 1. Why route at all

Phase 4 measured two retrievers that fail in opposite directions.

| On the Phase 4 corpus | Conceptual Recall@1 | Identifier Recall@1 |
| --- | ---: | ---: |
| BGE-M3 | **0.731** | 0.611 |
| BM25 + identifier boost | 0.355 | **0.889** |

Neither dominates. Every fusion tried in Phase 4 — plain RRF, and RRF over the
boosted lexical list — landed *between* them at rank 1, because rank-only fusion
spends one family's strength paying for the other's weakness. The cheapest way
to have both is to stop mixing and start choosing.

Choosing is possible here in a way it usually is not, because the discriminating
feature is not a matter of degree. A query either names a provision or it does
not, and `parse_identifiers()` already decides that exactly. There is nothing to
learn and nothing to tune.

---

## 2. Query families

The router assigns exactly one of three types. The classification is a pure
function of the query string: same input, same output, always.

**`IDENTIFIER`** — the query names a provision. Decided by
`parse_identifiers()`, which reads section/rule/article/regulation citations
including inverted forms ("sub-section (1) of section 39") and dotted treaty
paragraphs ("Article 27.1").

> "What does Section 3(p) of the Patents Act say?" · "What is Article 27.1 of
> TRIPS?" · "Explain Rule 13(2) of the Patents Rules."

**`CONCEPTUAL`** — no provision named, but the query names the body of law it is
asking about, so retrieval starts with the regime already settled.

> "Can a new form of a known substance be patented?" · "What rights arise from
> access to biological resources?"

**`GENERAL`** — an open practical question that names neither a provision nor a
regime. Retrieval has to discover the body of law itself.

> "How do I protect an Ayurvedic formulation?"

### What the router returns

```json
{
  "query_type": "IDENTIFIER",
  "confidence": "deterministic",
  "matched_identifiers": ["section 3(P)"],
  "reason": "the query names section 3(P) of IN-PATENTS-ACT-1970; an exact locator match is decidable from chunk metadata and does not need semantic similarity",
  "matched_documents": ["IN-PATENTS-ACT-1970"],
  "matched_regimes": ["patents"]
}
```

`confidence` is the literal string `"deterministic"`, never a number. There is
no probability model behind a regular expression, and a float here would be an
invented quantity that downstream code would eventually threshold on.

---

## 3. Routing rules

The rules are applied in this order, and the order *is* the policy.

| # | Test | Result |
| --- | --- | --- |
| 1 | `parse_identifiers(query)` finds a provision | `IDENTIFIER` |
| 2 | The query opens with a procedural interrogative | `GENERAL` |
| 3 | The query contains a regime term or names an instrument | `CONCEPTUAL` |
| 4 | Otherwise | `GENERAL` |

**A named provision wins outright.** It is the least ambiguous thing a legal
query can contain, and no amount of surrounding prose makes it less so.

**The procedural test runs before the regime lexicon**, deliberately. "How do I
register a trade mark?" names a regime but is still a request for a procedure
rather than a question about doctrine, and it belongs with the other open
questions.

**The regime lexicon holds bodies of law, not subject matter.** "Ayurvedic"
names something that patents, GI and food-safety regimes all regulate, so on its
own it settles nothing and is deliberately absent from the lexicon. "Patentable"
and "benefit sharing" each name one regime, and are in it.

### Policy

| Query type | Retriever | Why |
| --- | --- | --- |
| `IDENTIFIER` | `bm25_boost` | The locator is in chunk metadata; matching it is exact, not semantic |
| `CONCEPTUAL` | `dense` (BGE-M3) | The regime is known; the remaining problem is meaning |
| `GENERAL` | `dense` (BGE-M3) | Same, with the regime still to be found |

RRF is deliberately **not** in this table. Phase 4 found no fusion beating
BGE-M3 at rank 1, so fusing here would give up the thing routing exists to
protect.

---

## 4. Fallback rules

Fallbacks exist because a routing decision can be right about the query and
wrong about the corpus.

| Condition | Falls back to | Path recorded |
| --- | --- | --- |
| `IDENTIFIER`, but the boost matched no chunk | `dense` | `identifier_fallback_dense` |
| `IDENTIFIER`, but nothing came back at all | `dense` | `identifier_fallback_dense` |
| Dense returned nothing | `bm25` | `conceptual_fallback_lexical` / `general_fallback_lexical` |

**"No reasonable candidates" is read, not guessed.** For an identifier query it
means every returned hit has `identifier_boost == 0` — no chunk in the corpus
carries the locator the query named, so what came back is plain BM25 over a
query whose most informative token matched nothing. That is a fact in the
retriever's own output, not an inference from the scores.

**Every result records the path it arrived by**, in `metadata.retrieval_path`
and in each stored benchmark row:

```
identifier · identifier_fallback_dense · conceptual_dense
conceptual_fallback_lexical · general_dense · general_fallback_lexical
```

A fallback that is not visible is a silent change of engine, and the failure it
causes gets blamed on the retriever the caller thought was running.

**Neither fallback fired on this benchmark.** All 111 questions took a primary
path (17 `identifier`, 80 `conceptual_dense`, 14 `general_dense`). The fallbacks
are covered by unit tests rather than by the benchmark, and the honest reading is
that they are untested against real corpus behaviour.

---

## 5. Routing preserves provenance

Routing chooses an engine. It never edits a chunk.

A routed result carries the chosen retriever's `locator`, `citable`,
`temporal_status`, `source_evidence_level`, `document_id` and physical pages
through unchanged; the router only *adds* `retrieval_path`, `query_type`,
`routed_to`, `routing_reason` and `matched_identifiers`. In particular the
router cannot make a non-citable chunk citable — the Phase 3 hard invariant
survives Phase 5 untouched, and `tests/test_routing.py` asserts it directly.

Temporal filtering is **not** applied automatically (Phase 5 §10). Every result
continues to expose `temporal_status` so that legal-applicability filtering can
be designed on evidence later.

---

## 6. Results

111 questions, top-k 10, unfiltered over the whole corpus, chunking version
0.2.0. **Zero human-verified.**

### Overall

| Exp | Retriever | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@5 | nDCG@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | bm25 | 0.342 | 0.676 | 0.820 | 0.496 | 0.527 | 0.574 |
| B | dense | 0.712 | 0.910 | 0.964 | 0.807 | 0.823 | 0.842 |
| C | bm25_boost | 0.451 | 0.766 | 0.883 | 0.593 | 0.625 | 0.663 |
| D | hybrid (RRF) | 0.577 | 0.865 | 0.946 | 0.700 | 0.731 | 0.756 |
| D′ | hybrid_boost | 0.631 | 0.910 | **0.982** | 0.755 | 0.787 | 0.810 |
| **E** | **routed** | **0.766** | **0.928** | 0.964 | **0.843** | **0.857** | **0.869** |

### Conceptual queries (n = 93)

| Exp | Retriever | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | bm25 | 0.333 | 0.699 | 0.850 | 0.500 | 0.535 |
| B | dense | 0.731 | 0.914 | 0.957 | 0.818 | 0.833 |
| C | bm25_boost | 0.355 | 0.720 | 0.860 | 0.521 | 0.557 |
| D | hybrid | 0.581 | 0.882 | 0.979 | 0.714 | 0.747 |
| D′ | hybrid_boost | 0.591 | 0.892 | 0.979 | 0.726 | 0.759 |
| E | routed | 0.731 | 0.914 | 0.957 | 0.818 | 0.833 |

### Identifier queries (n = 18)

| Exp | Retriever | Recall@1 | Recall@5 | Recall@10 | MRR | nDCG@5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | bm25 | 0.389 | 0.556 | 0.667 | 0.477 | 0.483 |
| B | dense | 0.611 | 0.889 | 1.000 | 0.756 | 0.771 |
| C | bm25_boost | 0.944 | 1.000 | 1.000 | 0.963 | 0.972 |
| D | hybrid | 0.556 | 0.778 | 0.778 | 0.625 | 0.650 |
| D′ | hybrid_boost | 0.833 | 1.000 | 1.000 | 0.907 | 0.931 |
| **E** | **routed** | **0.944** | **1.000** | **1.000** | **0.972** | **0.980** |

**Routing is the first configuration to beat BGE-M3 at rank 1.** +0.054 Recall@1
and +0.036 MRR over B, and it does it by conceding nothing: E is identical to B
on conceptual queries by construction, and identical to C on identifier queries.
The gain is entirely the difference between the two families, which is exactly
what a router is for.

**Routing does not beat D′ at Recall@10** (0.964 vs 0.982). Nothing routing does
can recover a chunk the single chosen engine never returned; fusion still sees a
wider slice of the corpus. That is the trade, and it argues for a reranker over a
deeper fused candidate list — not for abandoning routing.

### Router census

| | |
| --- | --- |
| `IDENTIFIER` | 17 |
| `CONCEPTUAL` | 80 |
| `GENERAL` | 14 |
| Paths taken | `identifier` 17 · `conceptual_dense` 80 · `general_dense` 14 |
| Fallbacks taken | 0 |

### Latency

Median over 111 queries, on a **fresh retriever per measurement** with the model
warmed by one throwaway query.

| Retriever | Median | Mean |
| --- | ---: | ---: |
| bm25_boost | 2.7 ms | 2.8 ms |
| dense | 30.9 ms | 32.3 ms |
| hybrid_boost | 34.3 ms | 38.3 ms |
| routed | 31.1 ms | 30.9 ms |

Routing costs essentially nothing: classification is a handful of regex matches,
and 94 of 111 queries still pay the dense encoder. The 17 identifier queries get
answered in ~3 ms.

> **A measurement hazard worth recording.** `DenseRetriever` memoises query
> embeddings, and the benchmark ladder shares one dense instance across
> experiments. Because `dense` runs before `routed`, the latency stored in
> `routed.jsonl` (1.1 ms median) is entirely cache hits and is **not** a valid
> figure. The table above was re-measured on fresh instances. The stored
> per-row `latency_ms` should not be quoted for any retriever that runs after
> another retriever sharing its encoder.

### Filtered variant — a separate experiment

Phase 5 §9 keeps the unfiltered whole-corpus run as the primary benchmark.
`--filters declared` remains available and remains worse, as in Phase 4:

| Retriever | Recall@1 unfiltered | Recall@1 declared |
| --- | ---: | ---: |
| dense | 0.712 | 0.703 |
| routed | **0.766** | 0.757 |

The keyword domain classifier still pins cross-domain questions to one domain
and excludes the right document. Quote the unfiltered run.

---

## 7. Failure modes

Eight of 111 questions miss Recall@5 under routing, down from 10 for dense
alone. Every one is classified from evidence in the run; nothing is labelled by
resemblance.

| Category | Count |
| --- | ---: |
| dense retrieval failed | 5 |
| router chose wrong family | 3 |

**`router chose wrong family` (3).** The engine the router did *not* pick found
the answer, read from that engine's own stored run — a counterfactual, not a
guess. All three are the same shape: a conceptual-looking query whose answer is
best found lexically.

> **Q002** "Is the mere discovery of a new form of a known substance patentable
> in India?" → routed `conceptual_dense`, missed; `bm25_boost` ranks s.3 **1st**.
> The query paraphrases the *text* of section 3(d) almost verbatim, which is
> BM25's best case and BGE-M3's worst — the embedding spreads across every
> patentability provision.
>
> **Q066** "What penalty follows a contravention of the Biological Diversity
> Act?" → dense returns s.56 first, s.55 not in top 5; `bm25_boost` ranks it 3rd.
>
> **Q094** "How is a licence to manufacture homoeopathic medicines applied for?"
> → routed `general_dense`; dense ranks rule 85B 6th, `bm25_boost` ranks it 2nd.

This is the router's real limitation and it is a **recall** limitation, not a
classification error: the router classified all three correctly under its own
rules. There is no surface feature separating them from the 91 conceptual
queries dense wins.

**`dense retrieval failed` (5).** Both engines missed; routing had no better
option available. Q107 ("Who is the depositary of the CBD?") is worth noting: the
top four hits are all non-citable unlocated preamble text — correctly labelled,
where in Phase 4 two of them were confidently mislabelled as CBD Articles.

### Router classification disagreements

One question where the golden family and the router disagree:

> **Q029** "How does the Geographical Indications Act, 1999 define a geographical
> indication?" — the golden dataset labels it `identifier`; the router says
> `CONCEPTUAL`. **The router is right.** The query names an *instrument*, not a
> provision, so there is no locator to match and nothing for the boost to fire
> on. The golden label is the thing to revisit.

### Safety observations

| | routed | dense |
| --- | ---: | ---: |
| Rank-1 hit not citable | 6 / 111 | 8 / 111 |
| Non-citable hits in top 10 | 45 / 1110 | 49 / 1110 |
| Hits from a STALE source | 416 / 1110 | — |

Every one of these is *labelled*. A non-citable chunk reaching rank 1 is not a
defect in retrieval — hiding statutory text from search helps nobody — but it is
a hard constraint on the answer layer, which must refuse to quote them.

---

## 8. Limitations

**No question has been verified by a human.** Everything above is provisional.
`--verified-only` still evaluates nothing, by design.

**18 identifier questions is a small sample.** Recall@1 0.944 means 17 of 18. One
question is worth 0.056, so the identifier column should be read as "the boost
reliably works", not as a precise rate.

**The `CONCEPTUAL`/`GENERAL` split has no retrieval consequence.** Both route to
BGE-M3, so nothing in the ranking depends on it. It is reported because a router
that cannot say how often it met a question with no legal anchor cannot tell us
what to build next — but it is a surface-form test (a procedural opener, then a
lexicon), not a semantic judgment, and it will misfile queries that ask for
doctrine in procedural clothing.

**The regime lexicon is hand-written and English-only.** It has no coverage for
Hindi queries, and a query using vocabulary outside it falls to `GENERAL`, which
is the safe direction but not a correct classification.

**Neither fallback path fired on the benchmark.** They are tested in isolation,
not in anger.

**The router cannot recover what its chosen engine never returned.** Three of
the eight failures are exactly this. A router is a routing fix, not a recall fix.

---

## 9. Reproducing

```bash
.venv/bin/python scripts/ingest/run_pipeline.py --clean --structure --chunk --validate
.venv/bin/python scripts/evaluation/validate_golden_dataset.py
.venv/bin/python scripts/evaluation/benchmark_routing.py
.venv/bin/python scripts/evaluation/benchmark_routing.py --filters declared
.venv/bin/python scripts/evaluation/error_analysis.py --retriever routed --at 5
.venv/bin/python -m pytest tests/ -q
```

Phase 4's results are kept unchanged under
`data/evaluation/results/phase4-chunking-0.1.0/`. They were measured on 3,031
chunks; Phase 5 measures 3,073, so the two sets of numbers are not comparable
and A–D were re-run rather than quoted.

ROUTING_STATUS: PROVISIONAL — 111 questions, 0 human-verified, chunking 0.2.0.
