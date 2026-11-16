"""Does spelling correction recover retrieval a typo destroys — and at what cost?

The brief asks two questions about `app/spelling.py` and they have to be
answered together, because one of them is the risk the other creates:

1. On a question with a typo in it, does correction recover the retrieval the
   typo lost?
2. On a question with no typo in it, does correction change anything at all?

So this runs the 111-question benchmark three times over the same candidate
pools and the same reranker, varying only the query string:

* **clean** — the questions as written. The published baseline.
* **typo, uncorrected** — one seeded, realistic misspelling per question, sent
  to retrieval as typed. This is the damage.
* **typo, corrected** — the same misspelled questions, through
  `CorpusSpelling.correct`. This is the repair.

A fourth run, **clean, corrected**, is the one that matters most for safety: it
must reproduce the clean baseline exactly. If correction moves a single question
that had nothing wrong with it, it is not conservative and the numbers above are
not worth having.

One effect the four runs above cannot see is counted separately. Recall@k is
measured against retrieval, and retrieval is not where a typo does most of its
damage: a misspelled regime name — `tradmark`, `patant` — matches no document
alias and no regime term, so `app/clarify.py` finds nothing to anchor on and the
request never reaches retrieval at all. The user gets "which area of IP law did
you mean?" for a question that named one. `clarifier_interceptions` counts how
many of the 111 questions land there before and after correction, which is a
plain count of requests that were answered instead of being asked back.

Typos are injected deterministically. One content word of at least six letters
per question — the longest, so the choice cannot drift — with one of three edits
chosen by a seeded generator: drop a letter, swap two adjacent letters, or double
a letter. Those are what hands actually do. Identifiers, numbers and short words
are never touched, so no question loses its section number to the injector.

    .venv/bin/python scripts/evaluation/benchmark_typos.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.reranking.features import FeatureReranker  # noqa: E402
from ai.retrieval.base import load_chunks  # noqa: E402
from ai.retrieval.bm25 import BM25Retriever  # noqa: E402
from ai.retrieval.candidates import CandidateGenerator  # noqa: E402
from ai.retrieval.dense import DenseRetriever  # noqa: E402
from ai.retrieval.identifiers import BM25IdentifierBoostRetriever  # noqa: E402
from app.clarify import pre_retrieval  # noqa: E402
from app.spelling import CorpusSpelling  # noqa: E402
from evaluation.metrics import METRIC_KEYS, aggregate, score_question  # noqa: E402
from evaluation.validate_golden_dataset import GOLDEN_PATH, load_golden  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_PATH = os.path.join(PROJECT_ROOT, "data", "evaluation", "results",
                            "typo_tolerance.json")

SEED = 20260916
MIN_INJECTION_LENGTH = 6

#: Words never chosen for injection: they carry the instrument's identity, and
#: mangling one would measure document routing rather than spelling.
_PROTECTED = frozenset({"section", "sections", "rule", "rules", "article",
                        "articles", "regulation", "regulations", "schedule",
                        "form", "chapter", "part", "clause", "paragraph"})

_WORD = re.compile(r"[A-Za-z]+")


def inject(question: str, rng: random.Random) -> tuple[str, str | None]:
    """One realistic misspelling, or the question unchanged if none fits."""
    words = [(m.group(0), m.start(), m.end()) for m in _WORD.finditer(question)
             if len(m.group(0)) >= MIN_INJECTION_LENGTH
             and m.group(0).lower() not in _PROTECTED]
    if not words:
        return question, None
    word, start, end = max(words, key=lambda w: (len(w[0]), -w[1]))
    edit = rng.choice(("drop", "swap", "double"))
    position = rng.randrange(1, len(word) - 1)
    if edit == "drop":
        broken = word[:position] + word[position + 1:]
    elif edit == "swap":
        broken = word[:position] + word[position + 1] + word[position] + word[position + 2:]
    else:
        broken = word[:position] + word[position] + word[position:]
    return question[:start] + broken + question[end:], f"{word}->{broken}"


def evaluate(records: Sequence[dict[str, Any]], queries: Sequence[str],
             generator: CandidateGenerator, reranker: FeatureReranker,
             top_k: int) -> tuple[dict[str, float], list[str | None]]:
    per_question: list[dict[str, float]] = []
    first: list[str | None] = []
    for record, query in zip(records, queries):
        pool = generator.generate(query)
        ranked = reranker.rerank(query, pool.candidates, top_k)
        ids = [r.chunk_id for r in ranked]
        documents = [r.document_id for r in ranked]
        per_question.append(score_question(
            ids, record["expected_chunks"], documents,
            record.get("expected_documents") or []))
        first.append(ids[0] if ids else None)
    return aggregate(per_question, METRIC_KEYS), first


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default=GOLDEN_PATH)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", default=RESULTS_PATH)
    args = parser.parse_args(argv)

    records = load_golden(args.golden)
    chunks = load_chunks()
    bm25 = BM25Retriever(chunks)
    lexical = BM25IdentifierBoostRetriever(bm25)
    dense = DenseRetriever(chunks)
    dense.embed_query("warm")
    generator = CandidateGenerator(dense=dense, lexical=lexical, mode="expanded",
                                   order="interleaved", dense_depth=50,
                                   lexical_depth=50, aliases=lexical.aliases)
    reranker = FeatureReranker(idf=bm25.idf, aliases=lexical.aliases)
    speller = CorpusSpelling(chunk.text for chunk in chunks)

    rng = random.Random(SEED)
    clean = [record["query"] for record in records]
    injected: list[str] = []
    edits: list[str | None] = []
    for question in clean:
        broken, edit = inject(question, rng)
        injected.append(broken)
        edits.append(edit)

    repaired = [speller.correct(q).corrected for q in injected]
    clean_corrected = [speller.correct(q).corrected for q in clean]

    recovered = sum(1 for broken, fixed, original in zip(injected, repaired, clean)
                    if broken != original and fixed == original)
    touched_clean = sum(1 for a, b in zip(clean, clean_corrected) if a != b)
    injected_count = sum(1 for edit in edits if edit)

    def intercepted(queries: Sequence[str]) -> int:
        return sum(1 for query in queries
                   if pre_retrieval(query, lexical.aliases) is not None)

    interceptions = {label: intercepted(queries) for label, queries in
                     (("clean", clean), ("typo_uncorrected", injected),
                      ("typo_corrected", repaired))}

    runs: dict[str, dict[str, Any]] = {}
    for label, queries in (("clean", clean),
                           ("clean_corrected", clean_corrected),
                           ("typo_uncorrected", injected),
                           ("typo_corrected", repaired)):
        scores, first = evaluate(records, queries, generator, reranker, args.top_k)
        runs[label] = {"aggregate": {k: round(v, 4) for k, v in scores.items()},
                       "counts": {k: int(round(v * len(records)))
                                  for k, v in scores.items() if k.startswith("recall")},
                       "rank1": first}
        print(f"  {label:18s} r@1={scores['recall@1']:.4f} "
              f"r@5={scores['recall@5']:.4f} r@10={scores['recall@10']:.4f} "
              f"mrr={scores['mrr']:.4f}")

    identical = sum(1 for a, b in zip(runs["clean"]["rank1"],
                                      runs["clean_corrected"]["rank1"]) if a == b)
    summary = {
        "check": "typo_tolerance",
        "questions": len(records),
        "seed": SEED,
        "denominator_note":
            f"every rate is out of {len(records)} benchmark questions; "
            f"{injected_count} of them received an injected misspelling and "
            f"{len(records) - injected_count} had no word long enough to break",
        "injection": {
            "questions_injected": injected_count,
            "edits": [edit for edit in edits if edit][:20],
            "policy": "one edit (drop/swap/double) in the longest content word "
                      "of at least six letters; identifiers and structural "
                      "words are never touched",
        },
        "repair": {
            "restored_to_original": recovered,
            "of_injected": injected_count,
            "clean_questions_altered_by_correction": touched_clean,
            "clean_rank1_unchanged": f"{identical}/{len(records)}",
        },
        "clarifier_interceptions": {
            **interceptions,
            "denominator": len(records),
            "note": "questions app/clarify.py would answer with a clarifying "
                    "question instead of running the pipeline; a misspelled "
                    "regime name anchors on nothing, which Recall@k cannot see "
                    "because the benchmark harness does not run the clarifier",
        },
        "runs": {label: {k: v for k, v in run.items() if k != "rank1"}
                 for label, run in runs.items()},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"\n  repaired {recovered}/{injected_count} injected misspellings back "
          f"to the original wording")
    print(f"  correction altered {touched_clean}/{len(records)} clean questions; "
          f"rank-1 unchanged on {identical}/{len(records)}")
    print(f"  clarifier interceptions: clean {interceptions['clean']}, "
          f"typo {interceptions['typo_uncorrected']}, "
          f"typo+corrected {interceptions['typo_corrected']} "
          f"(of {len(records)})")
    print(f"  -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
