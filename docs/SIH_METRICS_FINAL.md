# IP-SAKTI — SIH Metrics

Grounded question answering over Indian intellectual-property law. Every number
below was measured from this repository on 2026-09-16. Full working:
`docs/METRICS_FULL_EVIDENCE.md`. Machine-readable:
`data/evaluation/results/submission_metrics.json`.

---

## Corpus

| | |
|---|---|
| **22 legal instruments** | Acts, Rules, Regulations and treaties, in full |
| **21 source PDFs · 1,782 pages** | ingested once, never edited |
| **3,073 evidence chunks** | every one addressable by section, rule or article |

## Retrieval — hand-written benchmark (111 questions)

| | |
|---|---|
| **84.7% Recall@1** — 94/111 | the right provision came back first |
| **99.1% Recall@5** — 110/111 | the right provision is on the first screen |
| **100% Recall@10** — 111/111 | it is never missed within ten results |
| **MRR 0.8995** | mean reciprocal rank of the right provision |
| **56 ms median** | retrieval + reranking, on a laptop CPU |

## Retrieval — AI-generated corpus-grounded set (1,000 questions)

| | |
|---|---|
| **1,000 AI-generated corpus-grounded evaluation queries** | 960 answerable + 40 out-of-corpus probes |
| **79.9% Recall@1** — 767/960 | scored against every chunk that answers the question |
| **93.4% Recall@5** — 897/960 | same basis |
| **95.8% Recall@10** — 920/960 | same basis |
| *strict single-label basis* | 70.1% / 87.4% / 94.1% — reported alongside, never instead |

## Evidence quality

| | |
|---|---|
| **88.9% Evidence Sufficiency@5** — 853/960 | the top 5 chunks contain the material the answer needs |
| **94.9% Evidence Sufficiency@10** — 911/960 | same, at depth 10 |
| **88.8% of expected answers supported** — 87/98 | AI-audited sample, rank hidden from the judge |
| **0 contradictions found** — 0/98 | no retrieved chunk contradicted its expected answer |

## Safety — what the system refuses to do

| | |
|---|---|
| **100% out-of-domain abstention** — 15/15 | questions the corpus cannot answer are refused, not guessed |
| **0 invalid citations published** — 0/41 | every published citation resolves to a real retrieved chunk |
| **0 unsupported claims published** — 0/41 | every published sentence traces to quoted evidence |
| **100% grounding pass** — 41/41 | ungrounded drafts are withheld, not shown with a caveat |
| *live-sample caveat* | 41 answers over one 60-question run, 10 of them served from the validated-answer cache |
| **211 damaged source chunks labelled** — 6.87% of 3,073 | flagged to the user; never repaired, never rewritten |

## Engineering

| | |
|---|---|
| **1,050 automated tests, all passing** | 1,050/1,050 |
| **Deterministic retrieval** | BM25 + BGE-M3 + a 7-feature untrained reranker; no fitted weights |
| **Zero provider calls** in the 1,000-question evaluation | the whole set runs offline in 60 seconds |

---

### What we do not claim

- These are **not accuracy** figures. Recall measures whether the right
  provision was retrieved, not whether an answer about it is legally correct.
- **No benchmark here is human-verified legal ground truth.** The 111 curated
  questions are 0/111 lawyer-reviewed; the 1,000-question set is AI-generated
  from the corpus and AI-judged.
- **Not legal advice.** IP-SAKTI answers from a fixed corpus of published legal
  texts and says so on every answer.
- 1,717 of 3,073 chunks come from instruments with a **newer official version**
  than the copy in the corpus, and answers resting on them are labelled STALE
  and capped below HIGH confidence.
