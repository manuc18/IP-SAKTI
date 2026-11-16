"""Prepare the golden benchmark for human review. Phase 8, Parts 3, 6, 7 and 8.

    .venv/bin/python scripts/evaluation/prepare_ground_truth.py

Reads `data/evaluation/golden_questions.jsonl` and never writes to it. Produces
two files beside it:

* `golden_verification.jsonl` — one verification record per question, every one
  of them `UNREVIEWED`;
* `golden_review_queue.jsonl` — the same questions in the order a reviewer
  should take them, hardest structural problems first.

**This script cannot mark anything VERIFIED.** It computes a `mechanical_status`
of PASS, WARNING or FAIL, and that is a statement about whether the *references
resolve*, not about whether the law is right. A question whose document, chunk,
locator and anchor all check out is still `UNREVIEWED`, because none of those
checks read the provision and decided it answers the question. The distinction
is the whole point of Phase 8; collapsing it would give us 111 green ticks and
no ground truth. `verification_status` is only ever written by
`review_ground_truth.py`, under a named reviewer.

Re-running is safe. Records already carrying a human decision keep it: their
`mechanical_*` fields are refreshed against the current corpus and everything
else is left exactly as the reviewer left it.

The hard checks are delegated to `validate_golden_dataset.validate()` rather
than reimplemented, so the two scripts cannot drift into disagreeing about what
a broken reference is.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai.retrieval.base import Chunk, load_chunks  # noqa: E402
from ai.retrieval.candidates import DEFAULT_LEXICAL_DEPTH  # noqa: E402
from ai.retrieval.identifiers import LOCATOR_FIELD, parse_identifiers  # noqa: E402
from scripts.evaluation.validate_golden_dataset import (  # noqa: E402
    GOLDEN_PATH,
    load_golden,
    validate,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVALUATION_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation")
VERIFICATION_PATH = os.path.join(EVALUATION_DIR, "golden_verification.jsonl")
QUEUE_PATH = os.path.join(EVALUATION_DIR, "golden_review_queue.jsonl")

SCHEMA_VERSION = "1.0.0"

#: Part 3's state machine. `UNREVIEWED` is the only state this script may write.
VERIFICATION_STATES = ("UNREVIEWED", "VERIFIED", "CORRECTED", "REJECTED", "NEEDS_REVIEW")
REVIEWED_STATES = tuple(s for s in VERIFICATION_STATES if s != "UNREVIEWED")

MECHANICAL_STATES = ("PASS", "WARNING", "FAIL")

#: An anchor phrase occurring in more chunks than the retriever's own candidate
#: depth cannot, on its own, be why we believe this chunk is the target: it does
#: not distinguish the target from the pool the retriever already returns. Tied
#: to the pipeline's depth rather than to a round number, so that if the depth
#: changes the threshold follows it.
WEAK_ANCHOR_THRESHOLD = DEFAULT_LEXICAL_DEPTH

#: Below this, a chunk is short enough that a reviewer should positively confirm
#: it carries enough context to answer, rather than assume it. Not a verdict —
#: a 67-character article can answer a yes/no question completely.
SHORT_CHUNK_CHARS = 300


# -- mechanical checks --------------------------------------------------------

def corpus_fingerprint(chunks: list[Chunk]) -> str:
    """Identifies the chunk set a review was performed against.

    Human verification is only valid for the corpus it was done on. Re-chunk and
    a reviewer's "yes, this chunk answers it" refers to a chunk that may no
    longer exist. Recording the fingerprint is what lets a later phase detect
    that instead of inheriting stale approvals.
    """
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk.chunk_id.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def anchor_frequency(chunks: list[Chunk]) -> dict:
    """Case-folded chunk text, once, so anchor counting is one pass not 111."""
    return [(chunk.chunk_id, chunk.text.lower()) for chunk in chunks]


def _count_anchor(anchor: str, lowered) -> int:
    needle = anchor.lower()
    return sum(1 for _, text in lowered if needle in text)


def warnings_for(record: dict[str, Any], by_id: dict[str, Chunk],
                 lowered) -> list[dict[str, str]]:
    """Structural weaknesses that need a human's attention but are not breaks.

    Each one is a reason a reviewer might reasonably reach a different
    conclusion from the label. None of them is a finding that the label is
    wrong — this script is not allowed to have an opinion about that.
    """
    findings: list[dict[str, str]] = []
    chunks = [by_id[cid] for cid in record["expected_chunks"] if cid in by_id]

    # A multi-chunk target is not warned about here. The review queue gives it
    # its own priority band, and raising it as a warning too would sweep those
    # questions into priority 1 and leave band 2 permanently empty — the same
    # signal counted twice, with the ordering it was meant to produce destroyed.

    for chunk in chunks:
        if chunk.temporal_status == "STALE":
            findings.append({
                "code": "stale_source",
                "detail": f"{chunk.chunk_id} is marked STALE; a stale source can "
                          f"still be correct ground truth, but the reviewer must "
                          f"say so explicitly"})
        elif chunk.temporal_status not in ("CURRENT", "STALE"):
            findings.append({
                "code": "temporal_status_unknown",
                "detail": f"{chunk.chunk_id} carries temporal_status "
                          f"{chunk.temporal_status!r}"})
        if chunk.source_evidence_level in ("C", "D"):
            findings.append({
                "code": "weak_provenance",
                "detail": f"{chunk.chunk_id} is evidence level "
                          f"{chunk.source_evidence_level}"})
        if len(chunk.text) < SHORT_CHUNK_CHARS:
            findings.append({
                "code": "short_evidence",
                "detail": f"{chunk.chunk_id} is {len(chunk.text)} characters; "
                          f"confirm it carries enough context to answer"})

    for anchor in record.get("anchor_terms", []):
        count = _count_anchor(anchor, lowered)
        if count > WEAK_ANCHOR_THRESHOLD:
            findings.append({
                "code": "weak_anchor",
                "detail": f"{anchor!r} occurs in {count} chunks, more than the "
                          f"retriever's candidate depth ({WEAK_ANCHOR_THRESHOLD}); "
                          f"it does not distinguish the target"})

    # An identifier named in the question but absent from the target's locator.
    # Often innocent — the identifier may be part of a provision's title, as in
    # "Rule 41 ... benefit sharing claim under section 26" — which is exactly
    # why it is a warning for a person to read rather than a failure.
    for ref in parse_identifiers(record["query"]):
        field, _ = LOCATOR_FIELD.get(ref.kind, (None, ()))
        if field is None:
            continue
        matched = any(chunk.locator.get(field) is not None
                      and str(chunk.locator[field]).upper() == ref.number
                      for chunk in chunks)
        if not matched:
            findings.append({
                "code": "query_identifier_unmatched",
                "detail": f"the question names {ref.kind} {ref.number}, which no "
                          f"expected chunk carries as a locator"})

    if record.get("allow_non_citable"):
        findings.append({
            "code": "non_citable_evidence_allowed",
            "detail": "the record waives the citability requirement"})

    return findings


def mechanical_review(records: list[dict[str, Any]], chunks: list[Chunk]) -> dict[str, dict]:
    """`{question_id: {status, findings}}`. Never returns a verification state."""
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    lowered = anchor_frequency(chunks)

    failures: dict[str, list[dict[str, str]]] = {}
    for question_id, check, detail in validate(records, chunks):
        failures.setdefault(question_id, []).append({"code": check, "detail": detail})

    out: dict[str, dict] = {}
    for record in records:
        question_id = record["question_id"]
        hard = failures.get(question_id, [])
        soft = warnings_for(record, by_id, lowered)
        findings = ([dict(f, severity="FAIL") for f in hard]
                    + [dict(f, severity="WARNING") for f in soft])
        status = "FAIL" if hard else ("WARNING" if soft else "PASS")
        out[question_id] = {"mechanical_status": status, "mechanical_findings": findings}
    return out


# -- verification records -----------------------------------------------------

def new_record(record: dict[str, Any], mechanical: dict, by_id: dict[str, Chunk],
               fingerprint: str, checked_at: str) -> dict[str, Any]:
    """One UNREVIEWED verification record.

    Human-judgement fields are null. Two fields are not, and both are mechanical
    facts rather than judgements: `anchors_present` is string containment, which
    is all criterion E asks; `temporal_status` is copied from the corpus, which
    is what criterion G asks the record to carry. Everything that requires
    reading the provision — is the document the right instrument, is the locator
    the right provision, is the evidence sufficient — stays null until a person
    fills it in.
    """
    chunks = [by_id[cid] for cid in record["expected_chunks"] if cid in by_id]
    anchors = record.get("anchor_terms", [])
    anchors_present: bool | None = None
    if chunks and anchors:
        anchors_present = all(
            any(anchor.lower() in chunk.text.lower() for chunk in chunks)
            for anchor in anchors)

    statuses = {chunk.temporal_status for chunk in chunks}
    if not statuses:
        temporal = "UNKNOWN"
    elif len(statuses) == 1:
        temporal = statuses.pop()
        if temporal not in ("CURRENT", "STALE"):
            temporal = "UNKNOWN"
    else:
        temporal = "MIXED"

    return {
        "question_id": record["question_id"],
        "schema_version": SCHEMA_VERSION,

        # -- the state machine (Part 3). Only review_ground_truth.py moves this.
        "verification_status": "UNREVIEWED",
        "reviewer": None,
        "reviewed_at": None,

        # -- criteria A-H (Part 4). Null means "no human has decided".
        "question_valid": None,          # A
        "document_correct": None,        # B
        "locator_correct": None,         # C
        "evidence_sufficient": None,     # D and F
        "anchors_present": anchors_present,   # E - mechanical containment only
        "temporal_status": temporal,     # G - copied from the corpus
        "scope_within_corpus": None,     # H

        # -- the golden claim, restated (Part 5). These are the *claims* under
        # review, not confirmations of them, so they stay separate from the
        # reviewer's own answers below.
        "required_chunk_ids": None,
        "primary_chunk_id": None,
        "notes": None,

        # Part 5: four things the label collapses into one. A reviewer may find
        # the retrieval target and the answer source are different chunks, or
        # that the authoritative instrument is not the one cited, or that no
        # current source exists in the corpus at all.
        "distinctions": {
            "retrieval_target_chunk_ids": None,
            "answer_source_chunk_ids": None,
            "authoritative_document_id": None,
            "current_source_available": None,
        },
        "correction": None,

        "golden": {
            "query": record["query"],
            "query_family": record.get("query_family"),
            "domain": record.get("domain"),
            "jurisdiction": record.get("jurisdiction"),
            "expected_documents": list(record["expected_documents"]),
            "expected_chunks": list(record["expected_chunks"]),
            "expected_locators": list(record["expected_locators"]),
            "anchor_terms": list(anchors),
            "requires_citation": record.get("requires_citation"),
            "allow_non_citable": record.get("allow_non_citable"),
            "ground_truth_method": record.get("ground_truth_method"),
            "notes": record.get("notes", ""),
        },

        "mechanical_status": mechanical["mechanical_status"],
        "mechanical_findings": mechanical["mechanical_findings"],
        "mechanical_checked_at": checked_at,
        "corpus_fingerprint": fingerprint,
        "verified_against_corpus": None,
    }


def merge(existing: list[dict[str, Any]], fresh: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Refresh the mechanical fields; never touch a human decision.

    A reviewer's afternoon is not something a script gets to discard because it
    was run again.
    """
    prior = {r["question_id"]: r for r in existing}
    out: list[dict[str, Any]] = []
    for record in fresh:
        old = prior.get(record["question_id"])
        if old is None or old.get("verification_status") == "UNREVIEWED":
            out.append(record)
            continue
        kept = dict(old)
        for field in ("mechanical_status", "mechanical_findings",
                      "mechanical_checked_at", "corpus_fingerprint", "golden"):
            kept[field] = record[field]
        out.append(kept)
    return out


# -- the review queue (Part 8) ------------------------------------------------

PRIORITY_LABELS = {
    1: "mechanical failure or warning",
    2: "target spans multiple chunks",
    3: "treaty question",
    4: "identifier question",
    5: "simple single-chunk question",
}


def priority_for(record: dict[str, Any], verification: dict[str, Any],
                 by_id: dict[str, Chunk]) -> tuple[int, str]:
    """Hardest structural problems first, so weaknesses surface early."""
    if verification["mechanical_status"] in ("FAIL", "WARNING"):
        codes = sorted({f["code"] for f in verification["mechanical_findings"]})
        return 1, f"{verification['mechanical_status']}: {', '.join(codes)}"
    if len(record["expected_chunks"]) > 1:
        return 2, PRIORITY_LABELS[2]
    types = {by_id[cid].document_type for cid in record["expected_chunks"] if cid in by_id}
    if "treaty" in types:
        return 3, PRIORITY_LABELS[3]
    if record.get("query_family") == "identifier":
        return 4, PRIORITY_LABELS[4]
    return 5, PRIORITY_LABELS[5]


def build_queue(records: list[dict[str, Any]], verifications: dict[str, dict],
                by_id: dict[str, Chunk]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        verification = verifications[record["question_id"]]
        priority, reason = priority_for(record, verification, by_id)
        rows.append({
            "question_id": record["question_id"],
            "priority": priority,
            "priority_label": PRIORITY_LABELS[priority],
            "priority_reason": reason,
            "mechanical_status": verification["mechanical_status"],
            "verification_status": verification["verification_status"],
            "query": record["query"],
        })
    rows.sort(key=lambda row: (row["priority"], row["question_id"]))
    return rows


# -- entry point --------------------------------------------------------------

def write_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def summarise(verifications: list[dict[str, Any]],
              queue: list[dict[str, Any]]) -> dict[str, Any]:
    states = {state: 0 for state in VERIFICATION_STATES}
    mechanical = {state: 0 for state in MECHANICAL_STATES}
    findings: dict[str, int] = {}
    for record in verifications:
        states[record["verification_status"]] += 1
        mechanical[record["mechanical_status"]] += 1
        for finding in record["mechanical_findings"]:
            findings[finding["code"]] = findings.get(finding["code"], 0) + 1
    priorities: dict[int, int] = {}
    for row in queue:
        priorities[row["priority"]] = priorities.get(row["priority"], 0) + 1
    return {
        "total_questions": len(verifications),
        "verification_states": states,
        "mechanical_status": mechanical,
        "findings_by_code": dict(sorted(findings.items(), key=lambda kv: -kv[1])),
        "review_priorities": dict(sorted(priorities.items())),
        "human_verified": states["VERIFIED"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", default=GOLDEN_PATH)
    parser.add_argument("--verification", default=VERIFICATION_PATH)
    parser.add_argument("--queue", default=QUEUE_PATH)
    parser.add_argument("--dry-run", action="store_true",
                        help="compute and report, write nothing")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    records = load_golden(args.golden)
    chunks = load_chunks()
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    fingerprint = corpus_fingerprint(chunks)
    checked_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    mechanical = mechanical_review(records, chunks)
    fresh = [new_record(r, mechanical[r["question_id"]], by_id, fingerprint, checked_at)
             for r in records]
    verifications = merge(read_jsonl(args.verification), fresh)

    by_qid = {r["question_id"]: r for r in verifications}
    queue = build_queue(records, by_qid, by_id)
    summary = summarise(verifications, queue)
    summary["corpus_fingerprint"] = fingerprint
    summary["corpus_chunks"] = len(chunks)

    if not args.dry_run:
        write_jsonl(args.verification, verifications)
        write_jsonl(args.queue, queue)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(f"questions            {summary['total_questions']}")
        print(f"corpus               {summary['corpus_chunks']} chunks, "
              f"fingerprint {fingerprint}")
        print("\nverification states")
        for state, count in summary["verification_states"].items():
            print(f"  {state:14s} {count:4d}")
        print("\nmechanical status (NOT human verification)")
        for state, count in summary["mechanical_status"].items():
            print(f"  {state:14s} {count:4d}")
        print("\nfindings")
        for code, count in summary["findings_by_code"].items():
            print(f"  {code:32s} {count:4d}")
        print("\nreview queue")
        for priority, count in summary["review_priorities"].items():
            print(f"  P{priority} {PRIORITY_LABELS[priority]:38s} {count:4d}")
        print(f"\nHUMAN-VERIFIED: {summary['human_verified']} of "
              f"{summary['total_questions']}")
        if args.dry_run:
            print("(dry run - nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
