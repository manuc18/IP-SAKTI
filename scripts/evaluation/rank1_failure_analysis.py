"""Why did something else win rank 1? One table, nine causes, no sweep.

    .venv/bin/python scripts/evaluation/rank1_failure_analysis.py

Reads the rows written by `run_submission_eval.py` and looks only at the
questions where the target chunk came back **inside the top 5 but not at rank
1**. Those are the questions where candidate generation did its job and ranking
did not, and they are the entire remaining gap between Recall@1 and Recall@5.

For each one it prints what won, what should have won, and the feature values
behind both — the reranker already reports its own working per candidate
(`FeatureReranker.score`), so nothing is re-derived or guessed here.

The clustering
--------------

Nine causes, assigned by the first rule that matches, in this order:

1. ``target_equivalence`` — the rank-1 chunk is in the question's acceptable set.
   Nothing went wrong; the benchmark's single label is narrower than the corpus.
   This is checked first because every other label would be a misdiagnosis.
2. ``corpus_extraction`` — the rank-1 chunk or the target carries extraction
   damage. A ranking argument about garbled text is not a ranking argument.
3. ``same_provision_split`` — both chunks carry the same citation. The chunker
   split one provision and the reranker picked the other half.
4. ``boilerplate_heading`` — the rank-1 chunk's heading is one of the recurring
   structural headings every instrument carries ("Short title and
   commencement", "Definitions", "Extent"). These match many questions weakly
   and specifically.
5. ``identifier_ambiguity`` — both chunks score above zero on locator exactness,
   so the query named a provision that more than one chunk claims.
6. ``cross_reference`` — the rank-1 chunk *mentions* the provision rather than
   *being* it: ``locator_referenced`` positive, ``locator_exactness`` zero.
7. ``broad_lexical`` — rank-1 beat the target on the lexical axis and lost on the
   dense one.
8. ``dense_confusion`` — the mirror image: rank-1 beat the target on the dense
   axis and lost on the lexical one.
9. ``other`` — everything the eight rules above do not explain, reported as its
   own count rather than absorbed into the nearest neighbour.

The point of the order is that it puts the two *non-ranking* causes first. A
cluster table that reports "boilerplate heading" for a question whose rank-1
answer was simply correct is worse than no table.

Nothing here changes ranking. It produces the evidence for deciding whether a
single intervention is worth making, and the acceptance bar for that decision
(Recall@1 up, Recall@5 down by no more than 0.5 points) is in the brief, not in
this file.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROWS_PATH = os.path.join(PROJECT_ROOT, "data", "evaluation", "results", "submission",
                         "retrieval_1000.jsonl")
OUT_PATH = os.path.join(PROJECT_ROOT, "data", "evaluation", "results",
                        "submission_failure_analysis.json")

#: Headings that recur in nearly every instrument and carry no subject matter.
#: Declared, not learned: each is a structural fixture of Indian legislative
#: drafting, and the list is short enough to read.
_BOILERPLATE = (
    "short title", "commencement", "extent", "definitions", "interpretation",
    "repeal", "savings", "power to make rules", "power of central government",
    "application", "preliminary", "act to have overriding effect",
)
_NORMALISE = re.compile(r"[^a-z0-9 ]+")


def is_boilerplate(heading: str | None) -> bool:
    if not heading:
        return False
    text = _NORMALISE.sub(" ", heading.lower())
    text = " ".join(text.split())
    return any(marker in text for marker in _BOILERPLATE)


def load_rows(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def classify(row: dict[str, Any], winner: dict[str, Any],
             target: dict[str, Any]) -> str:
    wc = winner.get("components") or {}
    tc = target.get("components") or {}
    acceptable = set(row.get("acceptable_chunk_ids") or [])
    damaged = set(row.get("damaged_chunks_in_top5") or [])

    if winner["chunk_id"] in acceptable:
        return "target_equivalence"
    if winner["chunk_id"] in damaged or target["chunk_id"] in damaged:
        return "corpus_extraction"
    if winner.get("citation") and winner.get("citation") == target.get("citation"):
        return "same_provision_split"
    if is_boilerplate(winner.get("heading")):
        return "boilerplate_heading"
    if wc.get("locator_exactness", 0.0) > 0.0 and tc.get("locator_exactness", 0.0) > 0.0:
        return "identifier_ambiguity"
    if wc.get("locator_referenced", 0.0) > 0.0 and wc.get("locator_exactness", 0.0) == 0.0:
        return "cross_reference"
    lexical_edge = wc.get("lexical_percentile", 0.0) - tc.get("lexical_percentile", 0.0)
    dense_edge = wc.get("dense_percentile", 0.0) - tc.get("dense_percentile", 0.0)
    if lexical_edge > 0 and dense_edge <= 0:
        return "broad_lexical"
    if dense_edge > 0 and lexical_edge <= 0:
        return "dense_confusion"
    return "other"


def explain(winner: dict[str, Any], target: dict[str, Any]) -> str:
    """The single largest weighted-score gap, named."""
    wc = winner.get("components") or {}
    tc = target.get("components") or {}
    gaps = {name: round(wc.get(name, 0.0) - tc.get(name, 0.0), 4)
            for name in ("dense_percentile", "lexical_percentile", "locator_exactness",
                         "idf_coverage", "phrase_overlap", "document_match",
                         "heading_bonus", "structural_penalty")}
    driver = max(gaps, key=lambda k: abs(gaps[k]))
    return f"{driver} {gaps[driver]:+.4f}"


def build(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    answerable = [r for r in rows if r["label_kind"] == "ANSWERABLE"]
    selected = [r for r in answerable
                if r["rank_of_target"] and 2 <= r["rank_of_target"] <= 5]

    table: list[dict[str, Any]] = []
    clusters: collections.Counter = collections.Counter()
    for row in selected:
        evidence = {e["chunk_id"]: e for e in row["evidence"]}
        winner = row["evidence"][0] if row["evidence"] else {}
        target_id = next((c for c in row["target_chunk_ids"] if c in evidence), None)
        target = evidence.get(target_id, {})
        if not winner or not target:
            continue
        cluster = classify(row, winner, target)
        clusters[cluster] += 1
        wc = winner.get("components") or {}
        tc = target.get("components") or {}
        table.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "target_rank": row["rank_of_target"],
            "cluster": cluster,
            "rank1_chunk": winner["chunk_id"],
            "target_chunk": target["chunk_id"],
            "rank1_locator": winner.get("citation"),
            "target_locator": target.get("citation"),
            "rank1_heading": winner.get("heading"),
            "target_heading": target.get("heading"),
            "scores": {
                "rank1": {"dense": wc.get("dense_percentile"),
                          "lexical": wc.get("lexical_percentile"),
                          "heading_bonus": wc.get("heading_bonus"),
                          "locator_exactness": wc.get("locator_exactness"),
                          "idf_coverage": wc.get("idf_coverage"),
                          "structural_penalty": wc.get("structural_penalty"),
                          "authority": wc.get("authority_score"),
                          "final": winner.get("rerank_score")},
                "target": {"dense": tc.get("dense_percentile"),
                           "lexical": tc.get("lexical_percentile"),
                           "heading_bonus": tc.get("heading_bonus"),
                           "locator_exactness": tc.get("locator_exactness"),
                           "idf_coverage": tc.get("idf_coverage"),
                           "structural_penalty": tc.get("structural_penalty"),
                           "authority": tc.get("authority_score"),
                           "final": target.get("rerank_score")},
            },
            "score_gap": round((winner.get("rerank_score") or 0.0)
                               - (target.get("rerank_score") or 0.0), 6),
            "why_rank1_won": explain(winner, target),
        })

    gaps = sorted(r["score_gap"] for r in table)
    return {
        "source_rows": len(rows),
        "answerable": len(answerable),
        "selected": len(table),
        "selection": "target chunk retrieved at rank 2-5, i.e. inside Recall@5 "
                     "and outside Recall@1",
        "clusters": dict(clusters.most_common()),
        "cluster_share": {name: round(count / len(table), 4)
                          for name, count in clusters.most_common()} if table else {},
        "score_gap": {
            "min": round(gaps[0], 6) if gaps else 0.0,
            "median": round(gaps[len(gaps) // 2], 6) if gaps else 0.0,
            "max": round(gaps[-1], 6) if gaps else 0.0,
            "under_0_02": sum(1 for g in gaps if g < 0.02),
            "note": "the margin rank 1 won by; a small margin means the ranking "
                    "was nearly right, not that it was arbitrary",
        },
        "why_rank1_won_top_drivers": dict(
            collections.Counter(r["why_rank1_won"].split()[0] for r in table).most_common()),
        "table": table,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default=ROWS_PATH)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args(argv)

    report = build(load_rows(args.rows))
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"{report['selected']} questions with the target at rank 2-5\n")
    for name, count in report["clusters"].items():
        print(f"  {count:4d}  {report['cluster_share'][name]:6.1%}  {name}")
    print(f"\nscore gap: median {report['score_gap']['median']:.4f}, "
          f"{report['score_gap']['under_0_02']} decided by under 0.02")
    print(f"drivers: {report['why_rank1_won_top_drivers']}")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
