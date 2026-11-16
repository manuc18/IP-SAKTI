"""The human review terminal for the golden benchmark. Phase 8, Parts 4 and 9.

    .venv/bin/python scripts/evaluation/review_ground_truth.py --reviewer "your name"

Entirely local. No web, no API, no model, no network, and nothing imported that
is not already in the project environment. It reads the corpus chunks and
`golden_verification.jsonl`, shows a reviewer one question at a time in review-
queue order, and records what they decide.

**It never writes to `golden_questions.jsonl`.** A correction is stored as a
correction — the original label stays exactly where it is, next to it, so the
two can be compared later. Overwriting the label would destroy the only evidence
of what the benchmark used to claim.

`VERIFIED` is gated. It is not a keystroke: pressing `v` opens the eight
criteria from Phase 8 Part 4 and the record is only marked VERIFIED if the
reviewer answers yes to all of them. Anything less lands in `NEEDS_REVIEW`,
`CORRECTED` or `REJECTED`. That is deliberate — a one-key "looks fine" is how a
benchmark acquires 111 green ticks that nobody actually checked.

Keys: `v` verified · `c` corrected · `r` rejected · `n` needs review ·
`s` skip · `q` save and quit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai.retrieval.base import Chunk, load_chunks  # noqa: E402
from scripts.evaluation.prepare_ground_truth import (  # noqa: E402
    QUEUE_PATH,
    VERIFICATION_PATH,
    read_jsonl,
    write_jsonl,
)

#: Phase 8 Part 4, in the order a reviewer can actually answer them: the cheap
#: structural questions before the one that needs the provision read closely.
CRITERIA: tuple[tuple[str, str, str], ...] = (
    ("A", "question_valid",
     "Is the question meaningful and answerable?"),
    ("B", "document_correct",
     "Is the designated document the correct authoritative instrument?"),
    ("C", "locator_correct",
     "Is the designated section/rule/article the correct legal locator?"),
    ("E", "anchors_present",
     "Does every anchor phrase genuinely occur in the designated evidence?"),
    ("H", "scope_within_corpus",
     "Is the question answerable from the corpus alone (or is that recorded)?"),
    ("D", "evidence_sufficient",
     "Does the cited chunk contain sufficient evidence to answer, or does the "
     "record identify all required chunks?"),
)

TEMPORAL_CHOICES = ("CURRENT", "STALE", "UNKNOWN")


# -- presentation -------------------------------------------------------------

RULE = "=" * 78


def neighbours(chunk_id: str, ordered_ids: list[str], span: int = 2) -> list[str]:
    """The chunk ids either side of this one within the same document.

    A reviewer deciding criterion F — is this chunk sufficient, or are more
    needed — is almost always asking about the provision immediately before or
    after. Showing the ids saves them going to look.
    """
    if chunk_id not in ordered_ids:
        return []
    document = chunk_id.split("::", 1)[0]
    index = ordered_ids.index(chunk_id)
    low = max(0, index - span)
    window = ordered_ids[low:index] + ordered_ids[index + 1:index + 1 + span]
    return [cid for cid in window if cid.split("::", 1)[0] == document]


def render(record: dict[str, Any], by_id: dict[str, Chunk], ordered_ids: list[str],
           position: str) -> str:
    golden = record["golden"]
    lines = [RULE, f"{record['question_id']}  ({position})", RULE,
             f"QUESTION   {golden['query']}",
             f"family     {golden['query_family']}   domain {golden['domain']}   "
             f"jurisdiction {golden['jurisdiction']}",
             f"DOCUMENT   {', '.join(golden['expected_documents'])}",
             f"LOCATOR    {', '.join(golden['expected_locators'])}",
             f"ANCHORS    {', '.join(repr(a) for a in golden['anchor_terms'])}",
             f"TEMPORAL   {record['temporal_status']}",
             f"MECHANICAL {record['mechanical_status']}"]
    for finding in record["mechanical_findings"]:
        lines.append(f"           [{finding['severity']}] {finding['code']}: "
                     f"{finding['detail']}")
    if golden.get("notes"):
        lines.append(f"NOTES      {golden['notes']}")

    for chunk_id in golden["expected_chunks"]:
        lines.append("-" * 78)
        chunk = by_id.get(chunk_id)
        if chunk is None:
            lines.append(f"CHUNK      {chunk_id}  ** NOT IN THE CORPUS **")
            continue
        lines.append(f"CHUNK      {chunk_id}")
        lines.append(f"citation   {chunk.citation}")
        lines.append(f"provenance level {chunk.source_evidence_level} · "
                     f"{chunk.temporal_status} · {chunk.source_status} · "
                     f"citable={chunk.citable} · pages={chunk.physical_pages}")
        near = neighbours(chunk_id, ordered_ids)
        if near:
            lines.append(f"neighbours {', '.join(near)}")
        lines.append("")
        lines.append(chunk.text.strip())
    lines.append(RULE)
    return "\n".join(lines)


# -- the decision -------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def ask_criteria(prompt: Callable[[str], str]) -> tuple[dict[str, bool], bool]:
    """Walk Part 4's checklist. Returns the answers and whether all held."""
    answers: dict[str, bool] = {}
    for letter, field, question in CRITERIA:
        while True:
            reply = prompt(f"  [{letter}] {question} (y/n) ").strip().lower()
            if reply in ("y", "yes"):
                answers[field] = True
                break
            if reply in ("n", "no"):
                answers[field] = False
                break
    return answers, all(answers.values())


def ask_temporal(prompt: Callable[[str], str], current: str) -> str:
    """Criterion G. Stale is not a rejection; it is a fact to record."""
    while True:
        reply = prompt(f"  [G] temporal status "
                       f"{'/'.join(TEMPORAL_CHOICES)} [{current}] ").strip().upper()
        if not reply:
            return current
        if reply in TEMPORAL_CHOICES:
            return reply


def ask_required(prompt: Callable[[str], str], label: str) -> str:
    """A field a correction cannot be recorded without."""
    while True:
        reply = prompt(f"  {label}: ").strip()
        if reply:
            return reply


def apply_decision(record: dict[str, Any], key: str, reviewer: str,
                   prompt: Callable[[str], str]) -> str | None:
    """Mutate `record` in place. Returns the new status, or None if skipped.

    Nothing here can reach `golden_questions.jsonl`, and nothing here can set
    VERIFIED without the full checklist having been answered yes.
    """
    golden = record["golden"]

    if key == "v":
        answers, all_held = ask_criteria(prompt)
        record.update(answers)
        record["temporal_status"] = ask_temporal(prompt, record["temporal_status"])
        if not all_held:
            failed = [letter for letter, field, _ in CRITERIA if not answers[field]]
            print(f"  criteria {', '.join(failed)} were not confirmed - this "
                  f"cannot be VERIFIED.")
            reply = prompt("  record as [c]orrected / [r]ejected / "
                           "[n]eeds review? ").strip().lower()
            if reply not in ("c", "r", "n"):
                reply = "n"
            return apply_decision(record, reply, reviewer, prompt)
        record["required_chunk_ids"] = list(golden["expected_chunks"])
        record["primary_chunk_id"] = golden["expected_chunks"][0]
        record["distinctions"]["retrieval_target_chunk_ids"] = list(golden["expected_chunks"])
        record["distinctions"]["answer_source_chunk_ids"] = list(golden["expected_chunks"])
        record["distinctions"]["authoritative_document_id"] = golden["expected_documents"][0]
        record["notes"] = prompt("  note (optional): ").strip() or None
        status = "VERIFIED"

    elif key == "c":
        # Part 9: a correction is not accepted without all four fields. The
        # original label is untouched - it stays in record["golden"].
        record["correction"] = {
            "corrected_document": ask_required(prompt, "corrected document id"),
            "corrected_locator": ask_required(prompt, "corrected locator"),
            "corrected_chunk_id": ask_required(prompt, "corrected chunk id"),
            "reason": ask_required(prompt, "reason"),
            "original_documents": list(golden["expected_documents"]),
            "original_locators": list(golden["expected_locators"]),
            "original_chunks": list(golden["expected_chunks"]),
        }
        record["required_chunk_ids"] = [record["correction"]["corrected_chunk_id"]]
        record["primary_chunk_id"] = record["correction"]["corrected_chunk_id"]
        record["distinctions"]["authoritative_document_id"] = \
            record["correction"]["corrected_document"]
        record["temporal_status"] = ask_temporal(prompt, record["temporal_status"])
        status = "CORRECTED"

    elif key == "r":
        record["notes"] = ask_required(prompt, "reason for rejection")
        status = "REJECTED"

    elif key == "n":
        record["notes"] = ask_required(prompt, "what needs a second look")
        status = "NEEDS_REVIEW"

    else:
        return None

    record["verification_status"] = status
    record["reviewer"] = reviewer
    record["reviewed_at"] = _now()
    record["verified_against_corpus"] = record["corpus_fingerprint"]
    return status


# -- the loop -----------------------------------------------------------------

def select(records: list[dict[str, Any]], queue: list[dict[str, Any]],
           priority: int | None, question_id: str | None,
           include_reviewed: bool) -> list[dict[str, Any]]:
    by_qid = {r["question_id"]: r for r in records}
    order = [row["question_id"] for row in queue] or sorted(by_qid)
    chosen = []
    for qid in order:
        record = by_qid.get(qid)
        if record is None:
            continue
        if question_id and qid != question_id:
            continue
        if priority is not None:
            row = next((r for r in queue if r["question_id"] == qid), None)
            if row is None or row["priority"] != priority:
                continue
        if not include_reviewed and record["verification_status"] != "UNREVIEWED":
            continue
        chosen.append(record)
    return chosen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reviewer", required=True,
                        help="the person doing the review; recorded on every decision")
    parser.add_argument("--verification", default=VERIFICATION_PATH)
    parser.add_argument("--queue", default=QUEUE_PATH)
    parser.add_argument("--priority", type=int, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--question", help="review one question by id")
    parser.add_argument("--include-reviewed", action="store_true",
                        help="also show questions that already carry a decision")
    args = parser.parse_args(argv)

    records = read_jsonl(args.verification)
    if not records:
        print(f"no verification records at {args.verification}; run "
              f"prepare_ground_truth.py first")
        return 1
    queue = read_jsonl(args.queue)
    chunks = load_chunks()
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    ordered_ids = [chunk.chunk_id for chunk in chunks]

    pending = select(records, queue, args.priority, args.question, args.include_reviewed)
    if not pending:
        print("nothing to review with those filters")
        return 0

    print(f"{len(pending)} question(s) to review as {args.reviewer!r}.")
    print("keys: v verified · c corrected · r rejected · n needs review · "
          "s skip · q save and quit\n")

    decided = 0
    for index, record in enumerate(pending, start=1):
        print(render(record, by_id, ordered_ids, f"{index} of {len(pending)}"))
        while True:
            key = input("decision [v/c/r/n/s/q]: ").strip().lower()
            if key in ("v", "c", "r", "n", "s", "q"):
                break
        if key == "q":
            break
        if key == "s":
            print("  skipped\n")
            continue
        status = apply_decision(record, key, args.reviewer, input)
        if status:
            decided += 1
            # Written after every decision, not at the end: a reviewer who
            # closes the terminal on question 40 keeps 39 decisions.
            write_jsonl(args.verification, records)
            print(f"  -> {status}\n")

    write_jsonl(args.verification, records)
    states: dict[str, int] = {}
    for record in records:
        states[record["verification_status"]] = states.get(record["verification_status"], 0) + 1
    print(f"\n{decided} decision(s) recorded this session.")
    for state, count in sorted(states.items()):
        print(f"  {state:14s} {count:4d}")
    print(f"\ngolden_questions.jsonl was not written to.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
