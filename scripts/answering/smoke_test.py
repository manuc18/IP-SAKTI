"""Phase 7 §20 — the whole answer path, offline, in about a second.

    .venv/bin/python scripts/answering/smoke_test.py
    .venv/bin/python scripts/answering/smoke_test.py --fixture conflicting --verbose
    .venv/bin/python scripts/answering/smoke_test.py --live     # ONE Groq request

**Without `--live` this makes no network request of any kind** and constructs no
Groq provider. `--live` is the only path that leaves the machine, and it sends
exactly one request: one tiny fixture passage, one question, a ≤15 s timeout, no
retry, no retrieval, no reranking, no corpus load and no golden dataset. What it
sends is printed before it is sent, so the payload can be read rather than
trusted.

Loads deterministic fixtures, runs each through
``query → evidence → answer object → citation validation → rendered answer``,
and prints what each gate decided. No model is loaded, no index is opened, no
network call is made and no subprocess is started: the whole thing is JSON and
arithmetic, which is the point of the Phase 7 architecture rather than an
accident of the demo.

The fixtures are real corpus passages (see `tests/fixtures/answering/`), so what
this demonstrates is the actual behaviour on actual text, not on invented data.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if __package__ in (None, ""):
    sys.path.insert(0, PROJECT_ROOT)

from ai.answering import (  # noqa: E402
    AnswerPolicy,
    EvidenceSet,
    ExtractiveMockProvider,
    GroundedAnswerEngine,
    render_answer,
)
from ai.answering.providers import build_request  # noqa: E402
from ai.retrieval.base import RetrievalResult  # noqa: E402

FIXTURE_DIR = os.path.join(PROJECT_ROOT, "tests", "fixtures", "answering")

ORDER = ("citable_statutory", "multiple_citations", "stale_only", "conflicting",
         "non_citable_only", "locator_inconsistent", "empty")


def load_fixture(name: str) -> dict:
    with open(os.path.join(FIXTURE_DIR, f"{name}.json"), encoding="utf-8") as fh:
        return json.load(fh)


def as_results(rows: list[dict]) -> list[RetrievalResult]:
    """Rebuild Phase 4 results from the stored dicts. The published contract."""
    return [RetrievalResult(chunk_id=row["chunk_id"], document_id=row["document_id"],
                            score=row["score"], rank=row["rank"], text=row["text"],
                            locator=row["locator"], citable=row["citable"],
                            metadata=row["metadata"]) for row in rows]


def run_one(name: str, *, policy: AnswerPolicy, verbose: bool) -> dict:
    fixture = load_fixture(name)
    evidence = EvidenceSet.from_results(as_results(fixture["results"]))
    engine = GroundedAnswerEngine(provider=ExtractiveMockProvider(), policy=policy)
    answer = engine.answer(fixture["query"], evidence)

    print("=" * 78)
    print(f"FIXTURE  {name}")
    print(f"WHY      {fixture['description']}")
    print(f"QUERY    {fixture['query']}")
    print(f"EVIDENCE {len(evidence)} passage(s), fingerprint {evidence.fingerprint}")
    for item in evidence:
        print(f"         [{item.marker}] {item.citation}  "
              f"citable={item.citable}  {item.temporal_status}  "
              f"level {item.source_evidence_level}  pp. {list(item.physical_pages)}")
    verdict = "ABSTAINED" if answer.insufficient_evidence else "ANSWERED"
    print(f"VERDICT  {verdict}  grounded={answer.grounded}  "
          f"confidence={answer.confidence}  conflict={answer.conflict_detected}")
    for failure in answer.insufficient_evidence_reasons:
        print(f"         ! {failure.code}: {failure.detail}")
    for failure in answer.validation_failures:
        print(f"         ! {failure.code}: {failure.detail}")
    print("-" * 78)
    print(render_answer(answer, show_confidence=True), end="")
    if verbose:
        print("-" * 78)
        print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False))
    print()
    return answer.to_dict()


LIVE_QUESTION = "What does Section 3 of the supplied Patents Act evidence say?"
LIVE_FIXTURE = "citable_statutory"


def run_live(*, policy: AnswerPolicy, verbose: bool) -> int:
    """One Groq request, through the whole Phase 7 pipeline. Nothing else.

    Deliberately noisy about what it is about to do: the payload is printed
    before the call, so "only the query, the passage text and the instructions
    leave this machine" is something a reader verifies rather than believes.
    """
    from ai.answering.groq import GroqConfig, GroqProvider

    fixture = load_fixture(LIVE_FIXTURE)
    evidence = EvidenceSet.from_results(as_results(fixture["results"]))
    config = GroqConfig.from_environment()

    if not os.environ.get("GROQ_API_KEY"):
        print("GROQ_API_KEY is not set. Put it in .env (gitignored) or export it.")
        return 2

    print("=" * 78)
    print("LIVE — one Groq request. No retrieval, no reranking, no corpus load.")
    print(f"  endpoint  {config.endpoint}")
    print(f"  model     {config.model}")
    print(f"  timeout   {config.timeout_seconds}s   retries 0   stream False")
    print(f"  evidence  {len(evidence)} passage(s), fingerprint {evidence.fingerprint}")
    print(f"  question  {LIVE_QUESTION}")

    request = build_request(LIVE_QUESTION, evidence, policy)
    from ai.answering.groq import build_payload

    payload = build_payload(request, config)
    print("-" * 78)
    print("EXACTLY WHAT WILL BE SENT (the Authorization header is not shown "
          "and is never logged):")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print("-" * 78)

    provider = GroqProvider(config=config)
    engine = GroundedAnswerEngine(provider=provider, policy=policy)
    answer = engine.answer(LIVE_QUESTION, evidence)

    verdict = "ABSTAINED" if answer.insufficient_evidence else "ANSWERED"
    print(f"VERDICT  {verdict}  grounded={answer.grounded}  "
          f"confidence={answer.confidence}  provider={answer.provider}")
    # A live abstention is useless without its cause: `unauthorized`,
    # `rate_limited` and `empty_completion` are three unrelated problems and
    # `provider_abstained` alone does not tell them apart. Never a credential —
    # the provider scrubs its own strings, and nothing here reads the key.
    provider_diagnostics = answer.diagnostics.get("provider_diagnostics", {})
    if provider_diagnostics.get("error"):
        print(f"  PROVIDER  {provider_diagnostics['error']}: "
              f"{provider_diagnostics.get('detail', '')}")
    for failure in answer.insufficient_evidence_reasons:
        print(f"         ! {failure.code}: {failure.detail}")
    for failure in answer.validation_failures:
        print(f"         ! {failure.code}: {failure.detail}")

    grounding = answer.diagnostics.get("grounding", {})
    for claim in grounding.get("unsupported_claims", []):
        print(f"  UNCITED   {claim}")
    for marker in grounding.get("unresolved_markers", []):
        print(f"  BAD MARKER [{marker}] resolves to no supplied passage")
    rejected = answer.diagnostics.get("rejected_draft")
    if rejected:
        print("-" * 78)
        print("REJECTED DRAFT (what the model wrote; NOT published as an answer):")
        print(rejected)

    print("-" * 78)
    print(render_answer(answer, show_confidence=True), end="")
    if verbose:
        print("-" * 78)
        print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False))
    print("=" * 78)
    print("Exactly one Groq request was made. No retry, no fallback provider, "
          "no model was loaded locally.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture", action="append", default=[],
                        help=f"fixtures to run (default: all — {', '.join(ORDER)})")
    parser.add_argument("--stale-insufficient", action="store_true",
                        help="abstain when every citable passage is STALE "
                             "(AnswerPolicy.stale_only_is_insufficient)")
    parser.add_argument("--verbose", action="store_true",
                        help="also dump the full answer object")
    parser.add_argument("--live", action="store_true",
                        help="make ONE real Groq request with one tiny fixture "
                             "passage. Without this flag nothing leaves the "
                             "machine and no Groq provider is constructed.")
    args = parser.parse_args(argv)

    policy = AnswerPolicy(stale_only_is_insufficient=args.stale_insufficient)

    if args.live:
        return run_live(policy=policy, verbose=args.verbose)

    names = args.fixture or list(ORDER)
    answered = 0
    for name in names:
        record = run_one(name, policy=policy, verbose=args.verbose)
        answered += 0 if record["insufficient_evidence"] else 1

    print("=" * 78)
    print(f"{answered} answered, {len(names) - answered} abstained, "
          f"out of {len(names)} fixtures.")
    print("No model was loaded. No network request was made. "
          "No Groq provider was constructed. No background process was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
