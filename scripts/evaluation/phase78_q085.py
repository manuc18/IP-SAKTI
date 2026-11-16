"""Phase 7.8 — one live request for Q085, the stale-evidence question.

    .venv/bin/python scripts/evaluation/phase78_q085.py --preflight
    .venv/bin/python scripts/evaluation/phase78_q085.py --dry-run
    .venv/bin/python scripts/evaluation/phase78_q085.py --run

Q085 was selected in Phase 7.3 as the stale-evidence slot and has never been
answered: its 7.3 request came back 429 before the model saw it. Every one of
its five passages is `STALE` at evidence level A, which makes it the only
question in the selection that exercises the temporal caution end to end — and
that caution is what Phase 7.8 has just rewritten, after Phase 7.7's first
published answer said "4 of 5 cited sources" over a single citation.

So this run tests two things at once, deliberately: whether prompt 0.3.0's
format holds on a second, differently-shaped question, and whether the caution
now counts what the answer actually cites.

Same pipeline as 7.5 and 7.7, imported rather than restated. One request per
invocation, and `--run` refuses if a result already exists.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if __package__ in (None, ""):
    sys.path.insert(0, PROJECT_ROOT)
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))

from ai.answering import AnswerPolicy, GroundedAnswerEngine, render_answer  # noqa: E402
from ai.answering.base import as_evidence_set  # noqa: E402
from ai.answering.citation_validator import VALIDATOR_VERSION  # noqa: E402
from ai.answering.grounding import (  # noqa: E402
    GROUNDING_VERSION,
    extract_markers,
    normalise_markers,
)
from ai.answering.policy import POLICY_VERSION  # noqa: E402
from ai.answering.providers import PROMPT_VERSION  # noqa: E402
from evaluation.phase73_five_questions import (  # noqa: E402
    BANNED_MODULES,
    TOP_K,
    build_pipeline,
    evaluate,
)
from evaluation.phase75_two_requests import (  # noqa: E402
    RecordingProvider,
    subdivision_observations,
)
from evaluation.phase77_q027 import analyse  # noqa: E402
from evaluation.validate_golden_dataset import GOLDEN_PATH, load_golden  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation", "results", "answering")
RESULTS_PATH = os.path.join(RESULTS_DIR, "phase7.8_q085.json")

QUESTION_ID = "Q085"

#: What Phase 7.3 computed for this question before Groq refused the request.
#: Recorded for comparison, not enforced: 7.3 never saw a completion, so there
#: is no answer baseline here — only a retrieval one.
BASELINE = {
    "phase": "7.3",
    "evidence_fingerprint": "a167a840c70e38d1",
    "lexical_candidate_rank": 6,
    "reranked_evidence_rank": 4,
    "outcome": "rate_limited (429); no completion was ever returned",
}


def preflight() -> int:
    ok = True
    for name in BANNED_MODULES:
        loaded = name in sys.modules
        print(f"  {name:24s} imported: {'YES  <-- FAIL' if loaded else 'no'}")
        ok &= not loaded
    key = bool(os.environ.get("GROQ_API_KEY"))
    print(f"  GROQ_API_KEY             {'present' if key else 'MISSING  <-- FAIL'}")
    ok &= key
    spent = os.path.exists(RESULTS_PATH)
    print(f"  phase 7.8 result exists  {'YES  <-- request already spent' if spent else 'no'}")
    ok &= not spent
    print(f"  versions                 prompt {PROMPT_VERSION}  policy "
          f"{POLICY_VERSION}  validator {VALIDATOR_VERSION}  grounding "
          f"{GROUNDING_VERSION}")
    ok &= (PROMPT_VERSION, POLICY_VERSION) == ("0.3.0", "0.3.0")
    return 0 if ok else 1


def run(*, dry_run: bool = False) -> int:
    from ai.answering.groq import GroqConfig, GroqProvider

    if not dry_run and os.path.exists(RESULTS_PATH):
        print("a Phase 7.8 result already exists; there is no second request.")
        return 1

    row = {r["question_id"]: r for r in load_golden(GOLDEN_PATH)}[QUESTION_ID]
    query = row["query"]
    chunks, generator, reranker, dense = build_pipeline()

    print("=" * 78)
    print(f"{QUESTION_ID}  {query}")

    pool = generator.generate(query)
    reranked = reranker.rerank(query, pool.candidates, TOP_K)
    evidence = as_evidence_set([r.to_retrieval_result() for r in reranked])
    print(f"  evidence fingerprint {evidence.fingerprint} "
          f"({'matches' if evidence.fingerprint == BASELINE['evidence_fingerprint'] else 'DIFFERS FROM'}"
          f" the 7.3 baseline)")

    config = GroqConfig.from_environment()
    if dry_run:
        from ai.answering import ExtractiveMockProvider

        provider = RecordingProvider(ExtractiveMockProvider())
    else:
        provider = RecordingProvider(GroqProvider(config=config))
    policy = AnswerPolicy()
    engine = GroundedAnswerEngine(provider=provider, policy=policy)

    started = time.perf_counter()
    answer = engine.answer(query, evidence)
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)

    raw = (provider.draft.text if provider.draft else "") or ""
    normalised = normalise_markers(raw)
    diagnostics = answer.diagnostics
    provider_diagnostics = dict(diagnostics.get("provider_diagnostics", {}))
    shape = analyse(normalised)
    published = not answer.insufficient_evidence

    cited = [{"marker": c.marker, "chunk_id": c.chunk_id,
              "citation_text": c.citation_text, "pages": list(c.pages),
              "temporal_status": c.temporal_status,
              "source_evidence_level": c.source_evidence_level}
             for c in answer.citations]
    distinct_sources = len({c["chunk_id"] for c in cited})

    print(f"  candidates={len(pool.candidates)}  supplied evidence={len(evidence)}  "
          f"{'ANSWERED' if published else 'ABSTAINED'}  grounded={answer.grounded}  "
          f"confidence={answer.confidence}  {elapsed_ms} ms")
    print(f"  provider calls={provider.calls}  finish_reason="
          f"{provider_diagnostics.get('finish_reason')}  "
          f"prompt={provider_diagnostics.get('prompt_tokens')} "
          f"completion={provider_diagnostics.get('completion_tokens')} "
          f"reasoning={provider_diagnostics.get('reasoning_tokens')}")
    print(f"  sentences={shape['sentences']}  propositions={shape['propositions']}"
          f"  cited={shape['cited_propositions']}  "
          f"cited sources={distinct_sources} of {len(evidence)} supplied")
    for caution in answer.cautions:
        print(f"    caution: {caution}")
    for failure in answer.insufficient_evidence_reasons:
        print(f"    ! {failure.code}: {failure.detail}")
    for failure in answer.validation_failures:
        print(f"    ! {failure.code}: {failure.detail}")

    report = {
        "phase": "7.8",
        "experiment_mode": "LEXICAL_ONLY",
        "purpose": "one controlled live request on the stale-evidence question: "
                   "does prompt 0.3.0 hold on a second question, and does the "
                   "rewritten temporal caution count the sources actually cited?",
        "requests_used": 0 if dry_run else 1,
        "retries": 0,
        "requested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question_id": QUESTION_ID,
        "query": query,
        "query_family": row["query_family"],
        "domain": row["domain"],
        "jurisdiction": row["jurisdiction"],
        "model": config.model,
        "temperature": config.temperature,
        "max_completion_tokens": config.max_completion_tokens,
        "timeout_seconds": config.timeout_seconds,
        "versions": {"prompt": PROMPT_VERSION, "policy": POLICY_VERSION,
                     "validator": VALIDATOR_VERSION,
                     "grounding": GROUNDING_VERSION},
        "dense_arm": "disabled",
        "reranker": reranker.name,
        "top_k": TOP_K,
        "policy": policy.to_dict(),
        "candidate_count": len(pool.candidates),
        "supplied_evidence_count": len(evidence),
        "evidence_fingerprint": evidence.fingerprint,
        "evidence": [
            {"marker": item.marker, "chunk_id": item.chunk_id,
             "citation": item.citation, "citable": item.citable,
             "temporal_status": item.temporal_status,
             "source_evidence_level": item.source_evidence_level}
            for item in evidence],
        "provider_status": ("completion returned" if raw.strip()
                            else provider_diagnostics.get("error", "no completion")),
        "provider_calls": provider.calls,
        "provider_diagnostics": provider_diagnostics,
        "latency_ms": elapsed_ms,
        "model_text_raw": raw,
        "text_after_normalisation": normalised,
        "marker_normalisation_applied": normalised != raw,
        "aliased_markers_in_raw": re.findall(r"【\d+】", raw),
        "citation_markers": extract_markers(normalised),
        "cited_sources": cited,
        "cited_source_count": distinct_sources,
        "cited_temporal_statuses": sorted({c["temporal_status"] for c in cited}),
        "supplied_temporal_statuses": sorted({e.temporal_status for e in evidence}),
        "cautions": list(answer.cautions),
        "shape": shape,
        "subdivision_observations": subdivision_observations(normalised, evidence),
        "citation_validation": diagnostics.get("validation"),
        "grounding": diagnostics.get("grounding"),
        "published": published,
        "final_answer": answer.answer,
        "abstention_reasons": [f.to_dict()
                               for f in answer.insufficient_evidence_reasons],
        "answer": answer.to_dict(),
        "rendered": render_answer(answer, show_confidence=True),
        "golden_target": {"documents": row["expected_documents"],
                          "locators": row["expected_locators"]},
        "scores": evaluate(row, pool.candidates, reranked, answer),
        "baseline_phase_7_3": BASELINE,
        "caveat": "Lexical-only candidate generation; retrieval figures are "
                  "lexical_candidate_recall and are not comparable with Phase "
                  "6. Nothing here measures legal accuracy.",
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = RESULTS_PATH.replace(".json", ".dry_run.json") if dry_run else RESULTS_PATH
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"→ {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)

    from ai.answering.groq import load_env_file

    load_env_file()
    if args.dry_run:
        return run(dry_run=True)
    if args.run:
        return run()
    print("PREFLIGHT — after importing the full pipeline:")
    return preflight()


if __name__ == "__main__":
    raise SystemExit(main())
