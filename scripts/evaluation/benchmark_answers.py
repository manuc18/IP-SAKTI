"""Phase 7 §23 — the framework for evaluating answers. Deliberately not run yet.

    .venv/bin/python scripts/evaluation/benchmark_answers.py                    # fixtures
    .venv/bin/python scripts/evaluation/benchmark_answers.py --rerank-run PATH  # 111 questions

The default mode runs the seven answering fixtures and finishes in well under a
second. It exists so the framework is exercised without touching the golden
dataset.

The second mode replays a Phase 6 results file: it takes the reranked chunk ids
that run already produced, rejoins them to the chunk store for their text and
provenance, and answers each question. **It loads no model.** Retrieval and
reranking are not repeated — they already happened, and their output is on disk.
That is the whole reason Phase 7 kept answering separate from retrieval: the
expensive half of the pipeline runs once and the cheap half can be re-evaluated
as often as its policy changes.

**What these metrics are not.** Every number below measures agreement between a
deterministic answer engine and corpus-derived labels that **no human has
verified**. None of them is legal accuracy, and none may be described as such —
the same rule Phase 6 §16 fixed for retrieval, for the same reason. What they
measure is whether citations resolve, whether claims carry them, and whether the
engine abstained when the evidence it was given did not contain the answer.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if __package__ in (None, ""):
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from ai.answering import (  # noqa: E402
    AnswerPolicy,
    EvidenceSet,
    ExtractiveMockProvider,
    GroundedAnswerEngine,
)
from ai.retrieval.base import RetrievalResult, load_chunks, result_from_chunk  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation", "results", "answering")
FIXTURE_DIR = os.path.join(PROJECT_ROOT, "tests", "fixtures", "answering")

FIXTURES = ("citable_statutory", "multiple_citations", "stale_only", "conflicting",
            "non_citable_only", "locator_inconsistent", "empty")

#: Stated on every summary, so a number cannot travel without its caveat.
DISCLAIMER = ("These metrics measure citation integrity and abstention behaviour "
              "against corpus-derived, human-unverified labels. They are not a "
              "measure of legal accuracy and must not be described as one.")


# -- evidence sources ---------------------------------------------------------


def evidence_from_fixture(name: str) -> tuple[str, str, EvidenceSet, list[str]]:
    with open(os.path.join(FIXTURE_DIR, f"{name}.json"), encoding="utf-8") as fh:
        fixture = json.load(fh)
    results = [RetrievalResult(chunk_id=row["chunk_id"],
                               document_id=row["document_id"], score=row["score"],
                               rank=row["rank"], text=row["text"],
                               locator=row["locator"], citable=row["citable"],
                               metadata=row["metadata"])
               for row in fixture["results"]]
    return name, fixture["query"], EvidenceSet.from_results(results), []


def evidence_from_rerank_run(path: str, top_k: int):
    """Replay a Phase 6 results file, rejoining chunk ids to the chunk store.

    The golden labels on each row (`expected_chunks`) are read here and handed
    back *alongside* the evidence, never inside it. Nothing that reaches the
    answer engine carries them — the same structural separation Phase 6 §9
    imposed on the reranker.
    """
    chunks = {chunk.chunk_id: chunk for chunk in load_chunks()}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            results = []
            for entry in row["results"][:top_k]:
                chunk = chunks.get(entry["chunk_id"])
                if chunk is None:
                    continue
                results.append(result_from_chunk(
                    chunk, score=entry.get("rerank_score", 0.0),
                    rank=entry.get("rerank_rank", len(results) + 1)))
            yield (row["question_id"], row["query"],
                   EvidenceSet.from_results(results),
                   list(row.get("expected_chunks") or []))


# -- metrics (Phase 7 §23) ----------------------------------------------------


def summarise(records: Sequence[dict[str, Any]], *, labelled: bool) -> dict[str, Any]:
    total = len(records)
    answered = [r for r in records if not r["insufficient_evidence"]]
    validated = [r for r in records if r["reached_validation"]]
    citations = [c for r in answered for c in r["citations"]]

    resolvable = [c for c in citations
                  if c["source_document"].get("filename") and c["pages"]]
    supported_rates = [r["supported_claim_rate"] for r in answered]

    completeness = []
    for record in answered:
        admissible = record["citable_evidence"]
        if admissible:
            completeness.append(len(record["cited_chunks"]) / admissible)

    summary: dict[str, Any] = {
        "questions": total,
        "answered": len(answered),
        "abstained": total - len(answered),
        "abstention_rate": _rate(total - len(answered), total),
        # Of the attempts that produced a draft, how many passed every citation
        # rule. An abstention before the provider ran is not a validation event.
        "citation_validity_rate": _rate(
            sum(1 for r in validated if r["validation_ok"]), len(validated)),
        "citations_per_answer": round(len(citations) / len(answered), 3) if answered else 0.0,
        # Every citation resolves to a source PDF and at least one page, which is
        # what Phase 7 §17's "click [1], open the page" depends on.
        "source_resolution_rate": _rate(len(resolvable), len(citations)),
        "evidence_supported_claim_rate": round(
            sum(supported_rates) / len(supported_rates), 4) if supported_rates else 0.0,
        "citation_completeness": round(
            sum(completeness) / len(completeness), 4) if completeness else 0.0,
        "conflicts_detected": sum(1 for r in records if r["conflict_detected"]),
        "not_a_legal_accuracy_measurement": DISCLAIMER,
    }
    summary["unsupported_claim_rate"] = round(
        1.0 - summary["evidence_supported_claim_rate"], 4)

    if labelled:
        # Abstention is correct when the evidence the engine was given did not
        # contain the labelled provision, and answering is correct when it did.
        # This is agreement with an unverified label, and nothing more.
        correct = sum(1 for r in records
                      if r["gold_in_evidence"] is not None
                      and r["gold_in_evidence"] != r["insufficient_evidence"])
        judged = sum(1 for r in records if r["gold_in_evidence"] is not None)
        summary["abstention_correctness"] = _rate(correct, judged)
        summary["abstention_correctness_note"] = (
            "agreement with corpus-derived labels on whether evidence for the "
            "labelled provision was present; 0 of the golden questions are "
            "human-verified")
    return summary


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


# -- run ----------------------------------------------------------------------


def run(pairs, policy: AnswerPolicy) -> list[dict[str, Any]]:
    engine = GroundedAnswerEngine(provider=ExtractiveMockProvider(), policy=policy)
    records: list[dict[str, Any]] = []
    for question_id, query, evidence, expected in pairs:
        answer = engine.answer(query, evidence)
        grounding = answer.diagnostics.get("grounding", {})
        validation = answer.diagnostics.get("validation")
        report = answer.diagnostics.get("policy_report", {})
        citable = sum(1 for a in report.get("assessments", [])
                      if a.get("admissible_as_citation"))
        records.append({
            "question_id": question_id,
            "query": query,
            "evidence_size": len(evidence),
            "evidence_fingerprint": evidence.fingerprint,
            "citable_evidence": citable,
            "insufficient_evidence": answer.insufficient_evidence,
            "insufficient_reasons": [f.code
                                     for f in answer.insufficient_evidence_reasons],
            "reached_validation": validation is not None,
            "validation_ok": bool(validation and validation.get("ok")),
            "validation_failures": [f.to_dict() for f in answer.validation_failures],
            "grounded": answer.grounded,
            "supported_claim_rate": grounding.get("supported_claim_rate", 0.0),
            "citations": [c.to_dict() for c in answer.citations],
            "cited_chunks": [c.chunk_id for c in answer.citations],
            "confidence": answer.confidence,
            "conflict_detected": answer.conflict_detected,
            "cautions": list(answer.cautions),
            # Labels live beside the record, never inside the engine's input.
            "gold_in_evidence": (bool(set(expected) & set(evidence.chunk_ids))
                                 if expected else None),
        })
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rerank-run", default=None,
                        help="a Phase 6 results .jsonl to replay; omit to run the "
                             "deterministic fixtures")
    parser.add_argument("--top-k", type=int, default=10,
                        help="evidence passages per question (default: 10)")
    parser.add_argument("--minimum-evidence-level", default="C", choices=list("ABCD"))
    parser.add_argument("--stale-insufficient", action="store_true")
    parser.add_argument("--abstain-on-conflict", action="store_true")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--tag", default="")
    args = parser.parse_args(argv)

    policy = AnswerPolicy(minimum_evidence_level=args.minimum_evidence_level,
                          stale_only_is_insufficient=args.stale_insufficient,
                          abstain_on_conflict=args.abstain_on_conflict)

    if args.rerank_run:
        pairs = evidence_from_rerank_run(args.rerank_run, args.top_k)
        source, labelled = os.path.basename(args.rerank_run), True
    else:
        pairs = (evidence_from_fixture(name) for name in FIXTURES)
        source, labelled = "fixtures", False

    records = run(pairs, policy)
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "evidence_source": source,
        "top_k": args.top_k,
        "engine": GroundedAnswerEngine(provider=ExtractiveMockProvider(),
                                       policy=policy).config(),
        "metrics": summarise(records, labelled=labelled),
    }

    os.makedirs(args.results_dir, exist_ok=True)
    stem = os.path.join(args.results_dir,
                        ".".join(filter(None, ["answers", os.path.splitext(source)[0],
                                               args.tag])))
    with open(stem + ".jsonl", "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    with open(stem + ".summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    metrics = summary["metrics"]
    print(f"evidence source: {source}   questions: {metrics['questions']}")
    for key in ("answered", "abstained", "citation_validity_rate",
                "source_resolution_rate", "evidence_supported_claim_rate",
                "unsupported_claim_rate", "citation_completeness",
                "citations_per_answer", "conflicts_detected",
                "abstention_correctness"):
        if key in metrics:
            print(f"  {key:32s} {metrics[key]}")
    print(f"\n{DISCLAIMER}")
    print(f"→ {stem}.summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
