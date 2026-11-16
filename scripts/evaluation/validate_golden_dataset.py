"""Validate the golden benchmark against the corpus it claims to describe.

A benchmark is only worth the ground truth under it, and ground truth rots
silently: re-chunk the corpus and a question that used to point at section 3 now
points at nothing, while the metrics carry on reporting a number. Every check
here exists to turn that silence into a failure.

    .venv/bin/python scripts/evaluation/validate_golden_dataset.py

The substantive checks are:

* every expected chunk and document exists in the current corpus;
* every expected locator agrees with the chunk's own locator metadata — a
  question asserting "Section 22" must not resolve to a chunk the pipeline
  labelled Rule 22;
* every anchor phrase actually occurs in the expected text, which is what makes
  the ground truth *derived* from the corpus rather than asserted about it;
* a question that requires authoritative citation may only expect citable
  chunks — and an unlocated chunk may only be expected where the record says so
  outright, via `allow_non_citable`.

`verified` is not something this script can grant. It stays false until a person
reads the provision and says so; nothing automated, here or elsewhere, may flip
it. What this script certifies is narrower and worth stating precisely: that the
references resolve and the anchors hold.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai.retrieval.base import load_chunks  # noqa: E402
from ai.retrieval.identifiers import LOCATOR_FIELD, parse_identifiers  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GOLDEN_PATH = os.path.join(PROJECT_ROOT, "data", "evaluation", "golden_questions.jsonl")

REQUIRED_FIELDS = (
    "question_id", "query", "domain", "jurisdiction", "expected_documents",
    "expected_chunks", "expected_locators", "notes", "verified",
)
QUERY_FAMILIES = {"identifier", "conceptual"}
GROUND_TRUTH_METHODS = {"corpus_derived", "human_authored"}


def load_golden(path: str = GOLDEN_PATH) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def validate(records: list[dict[str, Any]], chunks) -> list[tuple[str, str, str]]:
    """Return ``(question_id, check, detail)`` for every failure found."""
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    documents = {chunk.document_id for chunk in chunks}
    failures: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    for record in records:
        qid = record.get("question_id", "<no id>")

        def fail(check: str, detail: str) -> None:
            failures.append((qid, check, detail))

        for field in REQUIRED_FIELDS:
            if field not in record:
                fail("schema", f"missing field: {field}")
        if any(field not in record for field in REQUIRED_FIELDS):
            continue

        if qid in seen:
            fail("duplicate_question_id", qid)
        seen.add(qid)

        if not str(record["query"]).strip():
            fail("empty_query", "")
        if not isinstance(record["verified"], bool):
            fail("verified_type", repr(record["verified"]))
        if record.get("query_family") not in QUERY_FAMILIES:
            fail("query_family", str(record.get("query_family")))
        if record.get("ground_truth_method") not in GROUND_TRUTH_METHODS:
            fail("ground_truth_method", str(record.get("ground_truth_method")))
        if not record["expected_chunks"]:
            fail("no_expected_chunks", "")
        if not record["expected_documents"]:
            fail("no_expected_documents", "")

        for document_id in record["expected_documents"]:
            if document_id not in documents:
                fail("unknown_document", document_id)

        expected = [by_id[cid] for cid in record["expected_chunks"] if cid in by_id]
        for chunk_id in record["expected_chunks"]:
            if chunk_id not in by_id:
                fail("unknown_chunk", chunk_id)

        for chunk in expected:
            if chunk.document_id not in record["expected_documents"]:
                fail("chunk_outside_expected_documents",
                     f"{chunk.chunk_id} belongs to {chunk.document_id}")

        # Locator agreement: the asserted citation must be the chunk's own.
        for stated in record["expected_locators"]:
            refs = parse_identifiers(stated)
            if not refs:
                continue  # free-text locators (e.g. "Annexure II") are not checked here
            for ref in refs:
                field, _ = LOCATOR_FIELD.get(ref.kind, (None, ()))
                matched = any(
                    field and chunk.locator.get(field) is not None
                    and str(chunk.locator[field]).upper() == ref.number
                    for chunk in expected
                )
                if not matched:
                    fail("locator_mismatch",
                         f"'{stated}' matches no expected chunk's {ref.kind} locator")

        # Grounding: the anchors are why we believe the chunk answers the question.
        for anchor in record.get("anchor_terms", []):
            if not any(anchor.lower() in chunk.text.lower() for chunk in expected):
                fail("anchor_absent", f"{anchor!r} occurs in no expected chunk")

        # The citability invariant, restated as a property of the benchmark.
        allow_non_citable = bool(record.get("allow_non_citable", False))
        requires_citation = bool(record.get("requires_citation", True))
        for chunk in expected:
            if chunk.locator_unavailable and not allow_non_citable:
                fail("unmarked_missing_locator",
                     f"{chunk.chunk_id} has no locator; set allow_non_citable")
            if requires_citation and not chunk.citable:
                fail("non_citable_evidence",
                     f"{chunk.chunk_id} is not citable but the question requires citation")
        if requires_citation and allow_non_citable:
            fail("contradictory_citation_policy",
                 "requires_citation and allow_non_citable are both true")

    return failures


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    families: dict[str, int] = {}
    domains: dict[str, int] = {}
    for record in records:
        families[record.get("query_family", "?")] = families.get(record.get("query_family", "?"), 0) + 1
        domains[record["domain"]] = domains.get(record["domain"], 0) + 1
    return {
        "questions": len(records),
        "verified": sum(1 for r in records if r["verified"]),
        "unverified": sum(1 for r in records if not r["verified"]),
        "documents_covered": len({d for r in records for d in r["expected_documents"]}),
        "expecting_non_citable_evidence": sum(
            1 for r in records if r.get("allow_non_citable")),
        "by_family": dict(sorted(families.items())),
        "by_domain": dict(sorted(domains.items())),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", default=GOLDEN_PATH)
    parser.add_argument("--json", action="store_true", help="emit the summary as JSON")
    args = parser.parse_args(argv)

    records = load_golden(args.golden)
    failures = validate(records, load_chunks())
    summary = summarise(records)
    summary["failures"] = len(failures)

    if args.json:
        print(json.dumps({"summary": summary,
                          "failures": [dict(zip(("question_id", "check", "detail"), f))
                                       for f in failures]}, indent=2))
    else:
        for question_id, check, detail in failures:
            print(f"FAIL {question_id} {check}: {detail}")
        print(f"\nquestions            {summary['questions']}")
        print(f"human-verified       {summary['verified']}")
        print(f"unverified           {summary['unverified']}")
        print(f"documents covered    {summary['documents_covered']} / 22")
        print(f"by family            {summary['by_family']}")
        print(f"non-citable evidence {summary['expecting_non_citable_evidence']}")
        print(f"\nGOLDEN_DATASET: {'PASS' if not failures else 'FAIL'}"
              f"  ({len(failures)} failing checks)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
