"""Phase 7.5 — controlled validation of the Phase 7.4 fixes. One question, one request.

    .venv/bin/python scripts/evaluation/phase75_two_requests.py --preflight
    .venv/bin/python scripts/evaluation/phase75_two_requests.py --question Q027
    .venv/bin/python scripts/evaluation/phase75_two_requests.py --question Q001

**One question per invocation, deliberately.** Phase 7.3 spent five requests in
a single process and three of them came back 429; the budget here is two, and a
loop is the wrong shape for a budget that small. One question per process means
the request count is bounded by how many times the command is typed, and the
second question cannot be spent by a crash in the first one's reporting code.

The pipeline is the Phase 7.3 pipeline, imported rather than restated, so
"same configuration" is a property of the code and not of my memory of it:
lexical candidate generation (**EXPERIMENT_MODE: LEXICAL_ONLY**, the dense arm
disabled for machine safety), the feature reranker, the answer policy at its
defaults, and the unmodified prompt.

`RecordingProvider` is the one addition. Phase 7.5 has to report what the model
literally wrote *and* what the pipeline published, and on the answered path the
engine records only the published text — `rejected_draft` exists solely on the
abstention path. Rather than add a diagnostic to the engine to make this run
easier to report on, the runner observes the draft on its way past. It
delegates, records, and returns the draft unchanged; no gate sees anything
different because of it.

The golden dataset is read for the query string and, after an answer exists, for
scoring. No expected chunk, locator or anchor term reaches retrieval, the
policy, or the model.
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
from ai.answering.grounding import normalise_markers  # noqa: E402
from ai.answering.policy import ref_matches_subdivision_of  # noqa: E402
from ai.answering.providers import (  # noqa: E402
    AnswerProvider,
    ProviderDraft,
    ProviderRequest,
)
from ai.retrieval.identifiers import LOCATOR_FIELD, parse_identifiers  # noqa: E402
from evaluation.phase73_five_questions import (  # noqa: E402
    BANNED_MODULES,
    TOP_K,
    build_pipeline,
    evaluate,
)
from evaluation.validate_golden_dataset import GOLDEN_PATH, load_golden  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation", "results", "answering")
RESULTS_PATH = os.path.join(RESULTS_DIR, "phase7.5_two_requests.json")

#: The two questions this phase validates, and what each one is a test of.
#: Chosen by Phase 7.3's observed failures, not by expected outcome.
SELECTION = {
    "Q027": ("F1", "under-citation: the opening topic sentence carried no marker "
                   "and the grounding gate refused the whole answer"),
    "Q001": ("F2+F3", "full-width marker 【1】 was invisible to the validator, and "
                      "the locator section 3(p) was rejected against the supplied "
                      "Section 3 chunk that contains clause (p)"),
}

#: Phase 7.4 fixed three things. Only two of them are visible from outside, and
#: this is the one that is not: whether the parent-supports-child rule was the
#: reason a locator was admitted. Recomputed here from the same function the
#: validator calls, against the same evidence, purely to report it.
def subdivision_observations(text: str, evidence_set) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    rows: list[dict[str, Any]] = []
    for ref in parse_identifiers(text or ""):
        if ref.kind not in LOCATOR_FIELD or not ref.subdivisions:
            continue
        key = (ref.kind, ref.number, ref.subdivisions)
        if key in seen:
            continue
        seen.add(key)
        admitted_by = [item.chunk_id for item in evidence_set
                       if ref_matches_subdivision_of(ref, item)]
        rows.append({
            "written": f"{ref.kind} {ref.number}" + "".join(
                f"({sub})" for sub in ref.subdivisions),
            "subdivisions": list(ref.subdivisions),
            "admitted_by_subdivision_rule": admitted_by,
        })
    return rows


class RecordingProvider(AnswerProvider):
    """Delegates to the real provider and keeps a copy of what came back."""

    def __init__(self, inner: AnswerProvider):
        self.inner = inner
        self.name = inner.name
        self.version = inner.version
        self.calls = 0
        self.draft: ProviderDraft | None = None

    def generate(self, request: ProviderRequest) -> ProviderDraft:
        self.calls += 1
        self.draft = self.inner.generate(request)
        return self.draft

    def config(self) -> dict[str, Any]:
        return self.inner.config()


def preflight() -> int:
    ok = True
    for name in BANNED_MODULES:
        loaded = name in sys.modules
        print(f"  {name:24s} imported: {'YES  <-- FAIL' if loaded else 'no'}")
        ok &= not loaded
    key = bool(os.environ.get("GROQ_API_KEY"))
    print(f"  GROQ_API_KEY             {'present' if key else 'MISSING  <-- FAIL'}")
    ok &= key
    done = [q for q in _load_report().get("questions", ())]
    print(f"  requests already spent   {len(done)} of 2 "
          f"({', '.join(q['question_id'] for q in done) or 'none'})")
    ok &= len(done) < 2
    return 0 if ok else 1


def _load_report() -> dict[str, Any]:
    if not os.path.exists(RESULTS_PATH):
        return {}
    with open(RESULTS_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def run(question_id: str, *, dry_run: bool = False) -> int:
    """`dry_run` swaps in the deterministic provider and makes no request.

    Phase 7.3 kept the same precaution and it earned its place: the budget is
    two requests and there is no third to spend on a crash in the reporting
    code, so every layer except the network is exercised first on exactly these
    queries.
    """
    from ai.answering.groq import GroqConfig, GroqProvider

    report = {} if dry_run else _load_report()
    spent = report.get("questions", [])
    if any(q["question_id"] == question_id for q in spent):
        print(f"{question_id} already has a recorded result; refusing to spend a "
              f"second request on it.")
        return 1
    if len(spent) >= 2:
        print("the two-request budget for this phase is already spent.")
        return 1

    rows = {r["question_id"]: r for r in load_golden(GOLDEN_PATH)}
    row = rows[question_id]
    query = row["query"]
    validates, because = SELECTION[question_id]

    chunks, generator, reranker, dense = build_pipeline()
    config = GroqConfig.from_environment()
    if dry_run:
        from ai.answering import ExtractiveMockProvider

        provider = RecordingProvider(ExtractiveMockProvider())
    else:
        provider = RecordingProvider(GroqProvider(config=config))
    policy = AnswerPolicy()
    engine = GroundedAnswerEngine(provider=provider, policy=policy)

    print("=" * 78)
    print(f"{question_id}  [validates {validates}]  {query}")

    pool = generator.generate(query)
    reranked = reranker.rerank(query, pool.candidates, TOP_K)
    evidence = as_evidence_set([r.to_retrieval_result() for r in reranked])

    started = time.perf_counter()
    answer = engine.answer(query, evidence)
    elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)

    raw = (provider.draft.text if provider.draft else "") or ""
    normalised = normalise_markers(raw)
    grounding = answer.diagnostics.get("grounding", {})
    sentences = grounding.get("sentences", [])

    verdict = "ABSTAINED" if answer.insufficient_evidence else "ANSWERED"
    print(f"  candidates={len(pool.candidates)}  evidence={len(evidence)}  "
          f"{verdict}  grounded={answer.grounded}  "
          f"confidence={answer.confidence}  {elapsed_ms} ms")
    print(f"  provider calls={provider.calls}  "
          f"normalisation_applied={normalised != raw}")
    for failure in answer.insufficient_evidence_reasons:
        print(f"    ! {failure.code}: {failure.detail}")
    for failure in answer.validation_failures:
        print(f"    ! {failure.code}: {failure.detail}")

    record = {
        "question_id": question_id,
        "validates": validates,
        "phase_7_3_failure": because,
        "query": query,
        "query_family": row["query_family"],
        "domain": row["domain"],
        "jurisdiction": row["jurisdiction"],
        "requested_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provider_calls": provider.calls,
        "candidate_count": len(pool.candidates),
        "reranked_evidence_count": len(evidence),
        "evidence_fingerprint": evidence.fingerprint,
        "evidence": [
            {"marker": item.marker, "chunk_id": item.chunk_id,
             "citation": item.citation, "citable": item.citable,
             "temporal_status": item.temporal_status,
             "source_evidence_level": item.source_evidence_level}
            for item in evidence],
        "latency_ms": elapsed_ms,
        "model_text_raw": raw,
        "text_after_normalisation": normalised,
        "marker_normalisation_applied": normalised != raw,
        "aliased_markers_in_raw": re.findall(r"【\d+】", raw),
        "published_answer": "" if answer.insufficient_evidence else answer.answer,
        "first_sentence": sentences[0] if sentences else None,
        "uncited_claims": list(grounding.get("unsupported_claims", ())),
        "subdivision_observations": subdivision_observations(normalised, evidence),
        "answer": answer.to_dict(),
        "rendered": render_answer(answer, show_confidence=True),
        "golden_target": {"documents": row["expected_documents"],
                          "locators": row["expected_locators"]},
        "scores": evaluate(row, pool.candidates, reranked, answer),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = {
        "phase": "7.5",
        "experiment_mode": "LEXICAL_ONLY",
        "purpose": "controlled validation of the Phase 7.4 answer-layer fixes "
                   "F1 (prompt contract), F2 (safe hierarchical locators) and "
                   "F3 (marker normalisation)",
        "request_budget": 2,
        "model": config.model,
        "temperature": config.temperature,
        "max_completion_tokens": config.max_completion_tokens,
        "timeout_seconds": config.timeout_seconds,
        "retries": 0,
        "dense_arm": "disabled",
        "reranker": reranker.name,
        "top_k": TOP_K,
        "prompt_version": provider.config().get("prompt_version"),
        "policy": policy.to_dict(),
        "caveat": "Lexical-only candidate generation. Retrieval figures are "
                  "lexical_candidate_recall and are NOT comparable with the "
                  "Phase 6 hybrid results. Nothing here measures legal accuracy.",
        "questions": spent + [record],
    }
    report["requests_used"] = len(report["questions"])
    report["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    path = RESULTS_PATH.replace(".json", ".dry_run.json") if dry_run else RESULTS_PATH
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print(f"→ {path}  (requests_used={report['requests_used']})")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--question", choices=sorted(SELECTION))
    parser.add_argument("--dry-run", action="store_true",
                        help="run the whole pipeline through the deterministic "
                             "provider; makes no network request")
    args = parser.parse_args(argv)

    from ai.answering.groq import load_env_file

    load_env_file()
    if args.question:
        return run(args.question, dry_run=args.dry_run)
    print("PREFLIGHT — after importing the full pipeline:")
    return preflight()


if __name__ == "__main__":
    raise SystemExit(main())
