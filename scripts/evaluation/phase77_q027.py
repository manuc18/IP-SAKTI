"""Phase 7.7 — one question, one request, one thing to find out.

    .venv/bin/python scripts/evaluation/phase77_q027.py --preflight
    .venv/bin/python scripts/evaluation/phase77_q027.py --dry-run
    .venv/bin/python scripts/evaluation/phase77_q027.py --run

Q027 has now been asked twice: Phase 7.3 under prompt 0.1.0 and Phase 7.5 under
0.2.0. Both were refused, and both times the model wrote an uncited opening
sentence. 0.2.0 was the fix that enumerated the sentence kinds needing a marker
and it changed nothing, so 0.3.0 stopped asking for more citations and asked for
fewer sentences. This script spends the one reserved request finding out whether
that worked.

Everything about the run except the prompt is held fixed: the same query, the
same lexical-only candidate generation (**EXPERIMENT_MODE: LEXICAL_ONLY**), the
same reranker, the same policy. The evidence fingerprint is asserted against
Phase 7.5's `02813c7fcc2d8371` *before* the request goes out — an A/B where the
evidence moved is not an A/B, and finding that out afterwards would cost the
request.

`--run` refuses if a result already exists. There is no second request.
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
    is_claim,
    normalise_markers,
    split_sentences,
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
from evaluation.validate_golden_dataset import GOLDEN_PATH, load_golden  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation", "results", "answering")
RESULTS_PATH = os.path.join(RESULTS_DIR, "phase7.7_q027.json")

QUESTION_ID = "Q027"

#: Phase 7.5's evidence, by fingerprint. Asserted before the request, not after.
BASELINE_FINGERPRINT = "02813c7fcc2d8371"

#: What Phase 7.5 produced, for the A/B. Copied from
#: phase7.5_two_requests.json rather than remembered.
BASELINE = {
    "phase": "7.5",
    "prompt_version": "0.2.0",
    "sentences": 9,
    "propositions": 9,
    "cited_propositions": 6,
    "citation_validation": "FAIL",
    "grounding": "not reached",
    "published": False,
    "truncated": True,
}


def analyse(text: str) -> dict[str, Any]:
    """Sentence-level shape of an answer, by the pipeline's own definitions."""
    rows = []
    for i, sentence in enumerate(split_sentences(text), 1):
        markers = extract_markers(sentence)
        rows.append({"n": i, "is_proposition": is_claim(sentence),
                     "markers": markers, "sentence": sentence})
    props = [r for r in rows if r["is_proposition"]]
    return {
        "sentences": len(rows),
        "propositions": len(props),
        "cited_propositions": sum(1 for r in props if r["markers"]),
        "uncited_propositions": [r["sentence"] for r in props if not r["markers"]],
        "sentence_rows": rows,
        "within_one_to_four_sentence_contract": 1 <= len(rows) <= 4,
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
    print(f"  phase 7.7 result exists  {'YES  <-- request already spent' if spent else 'no'}")
    ok &= not spent
    print(f"  prompt version           {PROMPT_VERSION}")
    ok &= PROMPT_VERSION == "0.3.0"
    return 0 if ok else 1


def run(*, dry_run: bool = False) -> int:
    from ai.answering.groq import GroqConfig, GroqProvider

    if not dry_run and os.path.exists(RESULTS_PATH):
        print("a Phase 7.7 result already exists; there is no second request.")
        return 1

    row = {r["question_id"]: r for r in load_golden(GOLDEN_PATH)}[QUESTION_ID]
    query = row["query"]
    chunks, generator, reranker, dense = build_pipeline()

    print("=" * 78)
    print(f"{QUESTION_ID}  {query}")

    pool = generator.generate(query)
    reranked = reranker.rerank(query, pool.candidates, TOP_K)
    evidence = as_evidence_set([r.to_retrieval_result() for r in reranked])

    # Before the request, not after: an A/B whose evidence moved is not an A/B,
    # and discovering that afterwards would have cost the only request there is.
    if evidence.fingerprint != BASELINE_FINGERPRINT:
        print(f"  ABORT: evidence fingerprint is {evidence.fingerprint}, not the "
              f"Phase 7.5 baseline {BASELINE_FINGERPRINT}. No request was made.")
        return 1
    print(f"  evidence fingerprint {evidence.fingerprint} matches the 7.5 baseline")

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
    print(f"  candidates={len(pool.candidates)}  evidence={len(evidence)}  "
          f"{'ANSWERED' if published else 'ABSTAINED'}  grounded={answer.grounded}  "
          f"confidence={answer.confidence}  {elapsed_ms} ms")
    print(f"  provider calls={provider.calls}  finish_reason="
          f"{provider_diagnostics.get('finish_reason')}  "
          f"tokens={provider_diagnostics.get('completion_tokens')}/"
          f"{provider_diagnostics.get('prompt_tokens')}  "
          f"reasoning={provider_diagnostics.get('reasoning_tokens')}")
    print(f"  sentences={shape['sentences']}  propositions={shape['propositions']}"
          f"  cited={shape['cited_propositions']}  "
          f"normalisation_applied={normalised != raw}")
    for failure in answer.insufficient_evidence_reasons:
        print(f"    ! {failure.code}: {failure.detail}")
    for failure in answer.validation_failures:
        print(f"    ! {failure.code}: {failure.detail}")

    grounding = diagnostics.get("grounding")
    report = {
        "phase": "7.7",
        "experiment_mode": "LEXICAL_ONLY",
        "purpose": "one controlled live request: does prompt 0.3.0's 1-4 "
                   "sentence, every-sentence-cited contract change the "
                   "under-citation behaviour Phase 7.5 recorded on this exact "
                   "question and this exact evidence?",
        "requests_used": 1 if not dry_run else 0,
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
        "reranked_evidence_count": len(evidence),
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
        "shape": shape,
        "subdivision_observations": subdivision_observations(normalised, evidence),
        "citation_validation": diagnostics.get("validation"),
        "grounding": grounding,
        "published": published,
        "final_answer": answer.answer,
        "abstention_reasons": [f.to_dict()
                               for f in answer.insufficient_evidence_reasons],
        "answer": answer.to_dict(),
        "rendered": render_answer(answer, show_confidence=True),
        "golden_target": {"documents": row["expected_documents"],
                          "locators": row["expected_locators"]},
        "scores": evaluate(row, pool.candidates, reranked, answer),
        "baseline_phase_7_5": BASELINE,
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
