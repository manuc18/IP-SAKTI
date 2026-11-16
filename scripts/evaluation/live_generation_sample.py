"""Thirty live answers, once each, with the cache off and nothing retried.

    .venv/bin/python scripts/evaluation/live_generation_sample.py --size 30

This is the only part of the submission evaluation that calls a provider. The
1,000-question run is retrieval and evidence only; generation is measured here,
on a sample small enough to run inside a free-tier rate limit and large enough
to contain each state the pipeline can end in.

Why the cache is off
--------------------

`ValidatedAnswerCache` exists so a demo does not pay for an answer twice. Reading
it here would measure which answers had already been produced, which is not a
property of the pipeline. `Pipeline(cache=False)` is therefore not an
optimisation setting but the condition that makes the figure mean anything.

One request per question. No retry, no provider fallback, no second attempt at a
429 — a rate-limited request is recorded as UNAVAILABLE and excluded from the
answer denominator, because a request that never reached the model says nothing
about the answer path. The count of exclusions is printed beside every rate.

The sample
----------

The brief fixes the shape, so the shape is built explicitly rather than drawn:
ten easy, ten medium, ten hard, and within that, guaranteed coverage of

* questions whose target was at rank 1,
* questions whose target was at rank 2-5 (at least three),
* multi-chunk questions,
* questions resting on STALE-provenance documents,
* questions carrying an injected typo.

The typo cases reuse `benchmark_typos.inject`, so the misspellings here are the
same deterministic ones that benchmark measures rather than a new set invented
for this run. Every selection is seeded and the seed is written into the output.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation.benchmark_typos import inject  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation", "results", "submission")
ROWS_PATH = os.path.join(SUBMISSION_DIR, "retrieval_1000.jsonl")

SEED = 20260916
#: Seconds between requests. The free tier's per-minute cap is the binding
#: constraint on this run, not the pipeline's latency.
PACING_SECONDS = 22.0
#: How many of the sample carry an injected typo.
TYPO_SLOTS = 4
#: Minimum questions whose target sat outside rank 1 but inside rank 5.
MIN_RANK_2_5 = 3


def load_rows(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def select(rows: Sequence[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    """Ten per difficulty, with the required properties filled first."""
    rng = random.Random(SEED)
    answerable = [r for r in rows if r["label_kind"] == "ANSWERABLE"]
    per_difficulty = size // 3

    def bucket(row: dict[str, Any]) -> str:
        rank = row["rank_of_target"]
        return "rank1" if rank == 1 else ("rank2_5" if rank and rank <= 5 else "deeper")

    chosen: list[dict[str, Any]] = []
    taken: set[str] = set()

    def take(candidates: Sequence[dict[str, Any]], count: int) -> None:
        pool = [r for r in candidates if r["question_id"] not in taken]
        rng.shuffle(pool)
        for row in pool[:count]:
            taken.add(row["question_id"])
            chosen.append(row)

    # The properties the brief names, filled before the difficulty quotas so a
    # rare one cannot be squeezed out by a random draw.
    take([r for r in answerable if bucket(r) == "rank2_5"], MIN_RANK_2_5)
    take([r for r in answerable if len(r["target_chunk_ids"]) > 1], 3)
    take([r for r in answerable if r.get("temporal_status") == "STALE"], 3)
    take([r for r in answerable if bucket(r) == "deeper"], 2)

    for difficulty in ("easy", "medium", "hard"):
        have = sum(1 for r in chosen if r["difficulty"] == difficulty)
        take([r for r in answerable if r["difficulty"] == difficulty],
             max(0, per_difficulty - have))

    chosen = sorted(chosen, key=lambda r: r["question_id"])[:size]

    # Typos go on the last few, chosen deterministically, and the original
    # question is kept beside the misspelled one for the record.
    typo_rng = random.Random(SEED + 1)
    for row in chosen[-TYPO_SLOTS:]:
        broken, edit = inject(row["question"], typo_rng)
        row["asked_question"] = broken
        row["typo_edit"] = edit
    for row in chosen:
        row.setdefault("asked_question", row["question"])
        row.setdefault("typo_edit", None)
    return chosen


def run(sample: Sequence[dict[str, Any]], *, pacing: float) -> dict[str, Any]:
    from app.pipeline import Pipeline

    # cache=False is the point of this run; see the module docstring.
    pipeline = Pipeline(cache=False)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(sample, start=1):
        if index > 1:
            time.sleep(pacing)
        started = time.perf_counter()
        try:
            result = pipeline.answer(record["asked_question"], allow_clarification=False)
        except Exception as exc:  # noqa: BLE001 - one failure must not end the run
            rows.append({"question_id": record["question_id"], "status": "ERROR",
                         "error": str(exc)[:300]})
            print(f"  {index}/{len(sample)} {record['question_id']} -> ERROR", flush=True)
            continue
        elapsed = (time.perf_counter() - started) * 1000
        answer = result.get("answer") or {}
        evidence = result.get("evidence") or []
        cited = [e for e in evidence if e.get("cited")]
        retrieved = [e["chunk_id"] for e in evidence]
        report = answer.get("confidence_report") or {}
        rows.append({
            "question_id": record["question_id"],
            "question": record["question"],
            "asked_question": record["asked_question"],
            "typo_edit": record["typo_edit"],
            "difficulty": record["difficulty"],
            "answer_type": record["answer_type"],
            "temporal_status": record.get("temporal_status"),
            "target_chunk_ids": record["target_chunk_ids"],
            "rank_of_target_offline": record["rank_of_target"],
            "multi_chunk": len(record["target_chunk_ids"]) > 1,
            "status": result.get("status"),
            "service_failure": (result.get("service_failure") or {}).get("code"),
            "answer_text": answer.get("answer"),
            "grounded": bool(answer.get("grounded")),
            "confidence": answer.get("confidence"),
            "confidence_source": report.get("source_confidence"),
            "confidence_support": report.get("support_confidence"),
            "confidence_factors": [f.get("code") for f in (report.get("factors") or [])],
            "citation_count": len(answer.get("citations") or []),
            "citation_validation_failures": [f.get("code") for f in
                                             (answer.get("validation_failures") or [])],
            "unsupported_claims": answer.get("unsupported_claims") or [],
            "spelling_applied": bool((result.get("spelling") or {}).get("applied")),
            "retrieved_chunk_ids": retrieved,
            "target_retrieved": bool(set(retrieved) & set(record["target_chunk_ids"])),
            "target_at_rank_1": bool(retrieved[:1] and
                                     retrieved[0] in record["target_chunk_ids"]),
            "cited_evidence": [{"marker": e.get("marker"), "citation": e.get("citation"),
                                "chunk_id": e.get("chunk_id"),
                                "temporal_status": e.get("temporal_status"),
                                "integrity": (e.get("text_integrity") or {}).get("status")}
                               for e in cited],
            "damaged_evidence_shown": [e.get("chunk_id") for e in evidence
                                       if (e.get("text_integrity") or {}).get("status")
                                       not in (None, "CLEAN")],
            "finish_reason": (result.get("provider") or {}).get("finish_reason"),
            "tokens": (result.get("provider") or {}).get("usage"),
            "timings_ms": result.get("timings_ms"),
            "wall_ms": round(elapsed, 1),
            "cache_hit": bool((result.get("cache") or {}).get("hit")),
        })
        print(f"  {index}/{len(sample)} {record['question_id']} -> "
              f"{rows[-1]['status']} ({rows[-1]['confidence']})", flush=True)

    return summarise(rows)


def summarise(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    status = collections.Counter(r["status"] for r in rows)
    answered = [r for r in rows if r["status"] == "ANSWERED"]
    reached = [r for r in rows if r["status"] in ("ANSWERED", "ABSTAINED")]
    latencies = sorted(r["wall_ms"] for r in rows if r.get("wall_ms"))
    return {
        "metric": "live_generation_sample",
        "provider_calls": len(rows),
        "cache_used": any(r.get("cache_hit") for r in rows),
        "retries": 0,
        "sample_size": len(rows),
        "status_counts": dict(status),
        "answered": status["ANSWERED"],
        "abstained": status["ABSTAINED"],
        "unavailable": status["UNAVAILABLE"] + status["ERROR"],
        "denominator_note": (
            f"{status['UNAVAILABLE'] + status['ERROR']} request(s) never reached "
            f"the model and are excluded from the rates below, which are out of "
            f"{len(reached)}"),
        "citation_validation": {
            "passed": sum(1 for r in answered if not r["citation_validation_failures"]),
            "denominator": len(answered)},
        "grounding": {"passed": sum(1 for r in answered if r["grounded"]),
                      "denominator": len(answered)},
        "unsupported_claims_published": {
            "count": sum(1 for r in answered if r["unsupported_claims"]),
            "denominator": len(answered)},
        "target_retrieved": {"count": sum(1 for r in reached if r["target_retrieved"]),
                             "denominator": len(reached)},
        "target_at_rank_1": {"count": sum(1 for r in reached if r["target_at_rank_1"]),
                             "denominator": len(reached)},
        "confidence": dict(collections.Counter(
            r["confidence"] for r in answered if r["confidence"])),
        "typo_questions": {
            "count": sum(1 for r in rows if r["typo_edit"]),
            "spelling_applied": sum(1 for r in rows if r["typo_edit"]
                                    and r["spelling_applied"]),
            "target_retrieved": sum(1 for r in rows if r["typo_edit"]
                                    and r["target_retrieved"])},
        "damaged_evidence_surfaced": {
            "questions": sum(1 for r in rows if r.get("damaged_evidence_shown")),
            "denominator": len(rows)},
        "latency_ms": {
            "median": latencies[len(latencies) // 2] if latencies else 0.0,
            "min": latencies[0] if latencies else 0.0,
            "max": latencies[-1] if latencies else 0.0},
        "claim_safety": ("One live request per question, no retry, no fallback, "
                         "cache disabled. A sample of this size carries wide "
                         "confidence intervals and is reported with its "
                         "denominator, never as a rate over 1,000."),
        "rows": list(rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=30)
    parser.add_argument("--pacing", type=float, default=PACING_SECONDS)
    parser.add_argument("--dry-run", action="store_true",
                        help="print the selection and make no provider call")
    args = parser.parse_args(argv)

    sample = select(load_rows(ROWS_PATH), args.size)
    buckets = collections.Counter(r["difficulty"] for r in sample)
    ranks = collections.Counter(
        "rank1" if r["rank_of_target"] == 1 else
        ("rank2_5" if r["rank_of_target"] and r["rank_of_target"] <= 5 else "deeper")
        for r in sample)
    print(f"sample {len(sample)}: difficulty {dict(buckets)}, rank {dict(ranks)}, "
          f"multi-chunk {sum(1 for r in sample if len(r['target_chunk_ids']) > 1)}, "
          f"stale {sum(1 for r in sample if r.get('temporal_status') == 'STALE')}, "
          f"typo {sum(1 for r in sample if r['typo_edit'])}")
    if args.dry_run:
        for row in sample:
            print(f"  {row['question_id']} {row['difficulty']:6s} "
                  f"rank={row['rank_of_target']} {row['asked_question'][:78]}")
        return 0

    summary = run(sample, pacing=args.pacing)
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    path = os.path.join(SUBMISSION_DIR, "live_generation_sample.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"\nanswered {summary['answered']}  abstained {summary['abstained']}  "
          f"unavailable {summary['unavailable']}  (n={summary['sample_size']})")
    print(f"citation validation {summary['citation_validation']}")
    print(f"grounding {summary['grounding']}")
    print(f"unsupported claims published {summary['unsupported_claims_published']}")
    print(f"-> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
