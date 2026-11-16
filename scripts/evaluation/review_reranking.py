"""Put a reranking run in front of a human. Phase 6 §17.

    .venv/bin/python scripts/evaluation/review_reranking.py --run cross_encoder.bge-m3
    .venv/bin/python scripts/evaluation/review_reranking.py --run features --only-failures
    .venv/bin/python scripts/evaluation/review_reranking.py --run identity \
        --interactive --reviewer "A. Reviewer"

The benchmark has 111 questions and none of them has been checked by a person.
Every number this project has published rests on ground truth that was derived
from the corpus by rule and never read by a lawyer. This tool exists to make
that check cheap enough to actually happen: one screen per question, with the
expected provision, its anchor phrases, and the ten chunks the system put in
front of the reader, each with its citation, its citability, its evidence level
and its temporal status.

**What this tool cannot do, by construction.** It cannot mark a question
verified. There is no code path here that writes `verified: true`, and the
golden dataset is opened read-only. A reviewer's decision is recorded in a
separate review file with their name and the time; promoting those decisions
into the golden dataset is a deliberate, separate act by a person who has read
them. Phase 6 §17 requires exactly this separation, and it is the same rule
Phase 3 applied to provenance: an automated process may gather evidence, and may
never award the conclusion.

The five decisions a reviewer can record are the ones the spec names:

``correct``     the expected evidence is the right provision for this question
``incorrect``   it is not; the question points at the wrong chunk
``multiple``    several chunks are acceptable answers, not just the labelled one
``ambiguous``   the question itself admits more than one reading
``remove``      the question should not be in the benchmark

plus ``skip``, which records nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import textwrap
from datetime import datetime, timezone
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.retrieval.base import load_chunks  # noqa: E402
from evaluation.validate_golden_dataset import GOLDEN_PATH, load_golden  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation", "results", "reranking")
REVIEW_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation", "review")
REVIEW_PATH = os.path.join(REVIEW_DIR, "reranking_review.jsonl")

DECISIONS = {
    "1": ("correct", "the expected evidence is the right provision"),
    "2": ("incorrect", "the expected evidence is the wrong provision"),
    "3": ("multiple", "several chunks are acceptable, not only the labelled one"),
    "4": ("ambiguous", "the question admits more than one reading"),
    "5": ("remove", "the question should not be in the benchmark"),
    "s": ("skip", "no decision recorded"),
}

RULE = "─" * 78


def load_run(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def wrap(text: str, width: int = 74, indent: str = "      ") -> str:
    body = " ".join(text.split())
    return textwrap.fill(body, width=width, initial_indent=indent,
                         subsequent_indent=indent)


def render(row: dict[str, Any], record: dict[str, Any], chunks: dict[str, Any], *,
           excerpt: int) -> str:
    """One question's full review screen."""
    lines = [RULE, f"{row['question_id']}  [{record.get('query_family')}]"
                   f"  ({row.get('reranker')} over {row['candidates']['pool_size']} "
                   f"candidates)", RULE,
             "  QUERY", wrap(row["query"]), "",
             "  EXPECTED EVIDENCE (corpus-derived, NOT human-verified)"]
    for chunk_id, locator in zip(record["expected_chunks"],
                                 record["expected_locators"] + [""] * len(
                                     record["expected_chunks"])):
        chunk = chunks.get(chunk_id)
        lines.append(f"      {chunk_id}")
        lines.append(f"        document : {', '.join(record['expected_documents'])}")
        lines.append(f"        locator  : {locator or '(none stated)'}")
        if chunk is not None:
            lines.append(f"        citation : {chunk.citation}")
            lines.append(f"        citable  : {chunk.citable}   "
                         f"evidence: {chunk.source_evidence_level}   "
                         f"temporal: {chunk.temporal_status}   "
                         f"pages: {chunk.physical_pages}")
            lines.append(wrap(chunk.text[:excerpt], indent="        "))
        else:
            lines.append("        !! this chunk id is not in the current corpus")
    if record.get("anchor_terms"):
        lines.append("")
        lines.append("  ANCHOR PHRASES (each verified to occur in the expected text)")
        for term in record["anchor_terms"]:
            lines.append(f"      · {term}")
    if record.get("notes"):
        lines.append("")
        lines.append("  GROUND-TRUTH NOTE")
        lines.append(wrap(record["notes"]))

    lines.append("")
    lines.append(f"  RANK OF EXPECTED EVIDENCE: "
                 f"{row['rank_of_expected'] if row['rank_of_expected'] else 'not in top-k'}"
                 f"   (candidate pool position: {row['gold_pool_rank']})")
    lines.append("")
    lines.append("  TOP CANDIDATES AS RANKED")
    for result in row["results"]:
        chunk = chunks.get(result["chunk_id"])
        mark = "★" if result["chunk_id"] in record["expected_chunks"] else " "
        lines.append(
            f"   {mark}{result['rerank_rank']:>2}. {result['locator']}")
        lines.append(
            f"        score {result['rerank_score']:+.4f}   "
            f"citable: {result['citable']}   "
            f"evidence: {result['source_evidence_level']}   "
            f"temporal: {result['temporal_status']}   "
            f"pages: {result['physical_pages']}")
        lines.append(
            f"        dense rank: {result.get('dense_rank')}   "
            f"lexical rank: {result.get('lexical_rank')}   "
            f"identifier boost: {result.get('identifier_boost')}   "
            f"candidate #{result['candidate_order']}")
        if chunk is not None:
            lines.append(wrap(chunk.text[:excerpt], indent="        "))
        lines.append("")
    return "\n".join(lines)


def record_decision(path: str, *, question_id: str, reviewer: str, decision: str,
                    note: str, run: str) -> None:
    """Append one reviewer decision. Never touches the golden dataset."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "question_id": question_id,
            "reviewer": reviewer,
            "decision": decision,
            "note": note,
            "run": run,
            "reviewed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            # Stated in the record itself so a later reader of this file cannot
            # mistake a review for a verification. Promoting a review into the
            # golden dataset is a separate, deliberate act.
            "grants_verified_status": False,
        }, ensure_ascii=False) + "\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="cross_encoder.bge-m3",
                        help="results stem under data/evaluation/results/reranking/")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--golden", default=GOLDEN_PATH)
    parser.add_argument("--question", action="append", default=[],
                        help="review only these question ids")
    parser.add_argument("--only-failures", action="store_true",
                        help="only questions whose expected evidence missed the top 5")
    parser.add_argument("--only-flagged", action="store_true",
                        help="only questions the error analysis flagged for review")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--excerpt", type=int, default=600,
                        help="characters of chunk text to display")
    parser.add_argument("--interactive", action="store_true",
                        help="prompt for a decision after each question")
    parser.add_argument("--reviewer", default="",
                        help="the person reviewing; required with --interactive")
    parser.add_argument("--review-path", default=REVIEW_PATH)
    args = parser.parse_args(argv)

    if args.interactive and not args.reviewer.strip():
        raise SystemExit("--interactive requires --reviewer: an unattributed "
                         "review is not a review.")

    run_path = os.path.join(args.results_dir, f"{args.run}.jsonl")
    if not os.path.exists(run_path):
        available = sorted(f[:-6] for f in os.listdir(args.results_dir)
                           if f.endswith(".jsonl")) if os.path.isdir(args.results_dir) else []
        raise SystemExit(f"no such run: {run_path}\navailable: {', '.join(available) or 'none'}")

    rows = load_run(run_path)
    records = {r["question_id"]: r for r in load_golden(args.golden)}
    chunks = {c.chunk_id: c for c in load_chunks()}

    flagged: set[str] = set()
    errors_path = os.path.join(args.results_dir, f"{args.run}.errors.json")
    if os.path.exists(errors_path):
        with open(errors_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        flagged = {f["question_id"] for f in payload.get("failures", [])
                   if f.get("label_review_flags")}

    selected = rows
    if args.question:
        wanted = set(args.question)
        selected = [r for r in selected if r["question_id"] in wanted]
    if args.only_failures:
        selected = [r for r in selected
                    if not (r["rank_of_expected"] and r["rank_of_expected"] <= 5)]
    if args.only_flagged:
        selected = [r for r in selected if r["question_id"] in flagged]
    if args.limit:
        selected = selected[:args.limit]

    if not selected:
        print("nothing selected")
        return 0

    print(f"reviewing {len(selected)} of {len(rows)} questions from {args.run}")
    print("This tool records reviewer decisions. It does not and cannot mark any "
          "question `verified` — that stays false until a person promotes a "
          "review into the golden dataset deliberately.\n")

    for row in selected:
        record = records.get(row["question_id"])
        if record is None:
            print(f"{row['question_id']}: not in the golden dataset any more; skipping")
            continue
        print(render(row, record, chunks, excerpt=args.excerpt))
        if row["question_id"] in flagged:
            print("  ⚑ flagged by the error analysis as a possible label problem.")
        if not args.interactive:
            continue
        print("  Decision:")
        for key, (name, description) in DECISIONS.items():
            print(f"    [{key}] {name:<10s} {description}")
        try:
            choice = input("  > ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nstopping; decisions recorded so far are kept.")
            break
        if choice not in DECISIONS:
            print("  unrecognised; recorded nothing.")
            continue
        decision = DECISIONS[choice][0]
        if decision == "skip":
            continue
        try:
            note = input("  note (optional): ").strip()
        except (EOFError, KeyboardInterrupt):
            note = ""
        record_decision(args.review_path, question_id=row["question_id"],
                        reviewer=args.reviewer.strip(), decision=decision,
                        note=note, run=args.run)
        print(f"  recorded: {decision}")

    if args.interactive:
        print(f"\ndecisions appended to {args.review_path}")
        print("`verified` in the golden dataset is unchanged and remains false.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
