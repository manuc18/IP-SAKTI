"""Phase 7.3 — five golden questions, end to end, one Groq request each.

    .venv/bin/python scripts/evaluation/phase73_five_questions.py --preflight
    .venv/bin/python scripts/evaluation/phase73_five_questions.py --run

**EXPERIMENT_MODE: LEXICAL_ONLY.** The production candidate stage is hybrid —
BGE-M3 dense unioned with BM25+identifier-boost. The dense arm is disabled here,
because loading it means `torch`, `sentence_transformers` and an MPS device, and
this phase requires that no local model is loaded at all. The consequence is
stated rather than buried: retrieval numbers from this run are
`lexical_candidate_recall`, not IP-SAKTI's retrieval recall, and they are not
comparable with the Phase 6 hybrid results. What this run evaluates is the
*answer, citation and grounding* layers on real corpus evidence.

The dense arm is switched off by handing `CandidateGenerator` a retriever that
returns nothing, rather than by reimplementing candidate generation. Everything
downstream — the interleave, `Candidate` construction, provenance copying — is
the real code path, and with one contributing engine the round-robin merge is
the identity function over that engine's ranking.

The golden dataset is opened for two things only, both *after* an answer exists:
choosing which five questions to ask, and scoring what came back. No expected
chunk, locator, document or anchor term is ever passed to retrieval, to the
policy, or to the model.
"""
from __future__ import annotations

import argparse
import json
import os
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
from ai.retrieval.base import Filters, RetrievalResult, Retriever, load_chunks  # noqa: E402
from ai.retrieval.bm25 import BM25Retriever  # noqa: E402
from ai.retrieval.candidates import CandidateGenerator  # noqa: E402
from ai.retrieval.identifiers import BM25IdentifierBoostRetriever  # noqa: E402
from ai.reranking.features import FeatureReranker  # noqa: E402
from evaluation.validate_golden_dataset import GOLDEN_PATH, load_golden  # noqa: E402

RESULTS_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation", "results", "answering")

#: The five questions, chosen structurally and before any answer was seen.
#: `family` is this experiment's label for the slot; `query_family` is the
#: dataset's own field and is reported separately, unmodified.
SELECTION = (
    ("Q001", "simple_statutory_provision",
     "first identifier question in file order; a single-provision Section lookup"),
    ("Q078", "conceptual_synthesis",
     "the only Indian conceptual question whose golden target spans two chunks, "
     "so it cannot be answered by returning one provision"),
    ("Q027", "identifier",
     "first Rule-identifier question in file order whose document was not "
     "already selected"),
    ("Q104", "treaty_international",
     "first international-jurisdiction question over the Nagoya Protocol"),
    ("Q085", "stale_evidence",
     "first drug-regulation question; every chunk of that instrument carries "
     "temporal_status STALE"),
)

DENSE_DEPTH = 50
LEXICAL_DEPTH = 50
TOP_K = 5

BANNED_MODULES = ("torch", "transformers", "sentence_transformers", "mlx")


class DisabledDenseRetriever(Retriever):
    """The dense arm, switched off and saying so.

    Not a stub that pretends to work: it returns nothing, records that it was
    asked, and the run reports the count. A silent no-op here would make a
    lexical-only run indistinguishable from a hybrid one in the results file.
    """

    name = "dense_disabled"

    def __init__(self) -> None:
        self.calls = 0

    def search(self, query: str, filters: Filters | None = None,
               top_k: int = 10) -> list[RetrievalResult]:
        self.calls += 1
        return []

    def config(self) -> dict[str, Any]:
        return {"retriever": self.name, "version": self.version,
                "disabled_reason": "Phase 7.3 machine safety: no torch, no "
                                   "sentence-transformers, no MPS/CUDA, no "
                                   "local model load"}


def preflight() -> int:
    """Everything that must be true before a single request is made."""
    ok = True
    for name in BANNED_MODULES:
        loaded = name in sys.modules
        print(f"  {name:24s} imported: {'YES  <-- FAIL' if loaded else 'no'}")
        ok &= not loaded
    print(f"  GROQ_API_KEY             {'present' if os.environ.get('GROQ_API_KEY') else 'MISSING  <-- FAIL'}")
    ok &= bool(os.environ.get("GROQ_API_KEY"))
    return 0 if ok else 1


def build_pipeline():
    chunks = load_chunks()
    bm25 = BM25Retriever(chunks)
    boost = BM25IdentifierBoostRetriever(bm25)
    dense = DisabledDenseRetriever()
    generator = CandidateGenerator(dense=dense, lexical=boost,
                                   mode="expanded", order="interleaved",
                                   dense_depth=DENSE_DEPTH,
                                   lexical_depth=LEXICAL_DEPTH,
                                   aliases=boost.aliases)
    reranker = FeatureReranker(idf=bm25.idf, aliases=boost.aliases)
    return chunks, generator, reranker, dense


def evaluate(row: dict[str, Any], candidates, reranked, answer) -> dict[str, Any]:
    """Engineering properties only. Never legal accuracy.

    The golden fields are read here and nowhere earlier.
    """
    expected = set(row["expected_chunks"])
    candidate_ids = [c.chunk_id for c in candidates]
    evidence_ids = [r.candidate.chunk_id for r in reranked]

    citations = answer.citations
    markers = [c.marker for c in citations]
    return {
        "lexical_candidate_recall": bool(expected & set(candidate_ids)),
        "lexical_candidate_rank": next(
            (i + 1 for i, cid in enumerate(candidate_ids) if cid in expected), None),
        "reranked_evidence_recall": bool(expected & set(evidence_ids)),
        "reranked_evidence_rank": next(
            (i + 1 for i, cid in enumerate(evidence_ids) if cid in expected), None),
        # "Valid" means resolved, not "correct": every citation must point at a
        # passage that was actually supplied, and no validation rule may have
        # tripped. Whether the passage supports the sentence is Part H's
        # evidence-alignment question and is judged by a human, not here.
        "citation_valid": (not answer.validation_failures)
                          and all(c.chunk_id in evidence_ids for c in citations),
        "citation_count": len(citations),
        "citation_markers": markers,
        "answer_grounded": bool(answer.grounded),
        "insufficient_evidence": bool(answer.insufficient_evidence),
    }


def run(argv_policy: AnswerPolicy, *, dry_run: bool = False) -> int:
    """`dry_run` swaps in the deterministic provider and makes no request.

    Worth the extra pass: the live budget is five requests and there is no sixth
    to spend on a crash in the reporting code, so every layer except the network
    is exercised first on exactly these five queries.
    """
    from ai.answering.groq import GroqConfig, GroqProvider

    rows = {r["question_id"]: r for r in load_golden(GOLDEN_PATH)}
    chunks, generator, reranker, dense = build_pipeline()
    config = GroqConfig.from_environment()
    if dry_run:
        from ai.answering import ExtractiveMockProvider

        provider = ExtractiveMockProvider()
    else:
        provider = GroqProvider(config=config)
    engine = GroundedAnswerEngine(provider=provider, policy=argv_policy)

    records: list[dict[str, Any]] = []
    for question_id, family, why in SELECTION:
        row = rows[question_id]
        query = row["query"]
        print("=" * 78)
        print(f"{question_id}  [{family}]  {query}")

        pool = generator.generate(query)
        reranked = reranker.rerank(query, pool.candidates, TOP_K)
        evidence = as_evidence_set([r.to_retrieval_result() for r in reranked])

        started = time.perf_counter()
        answer = engine.answer(query, evidence)
        elapsed_ms = round((time.perf_counter() - started) * 1000.0, 1)

        verdict = "ABSTAINED" if answer.insufficient_evidence else "ANSWERED"
        print(f"  candidates={len(pool.candidates)}  evidence={len(evidence)}  "
              f"{verdict}  grounded={answer.grounded}  "
              f"confidence={answer.confidence}  {elapsed_ms} ms")
        for failure in answer.insufficient_evidence_reasons:
            print(f"    ! {failure.code}: {failure.detail}")
        for failure in answer.validation_failures:
            print(f"    ! {failure.code}: {failure.detail}")

        scores = evaluate(row, pool.candidates, reranked, answer)
        record = {
            "question_id": question_id,
            "experiment_family": family,
            "selection_rationale": why,
            "query": query,
            "query_family": row["query_family"],
            "domain": row["domain"],
            "jurisdiction": row["jurisdiction"],
            "candidate_count": len(pool.candidates),
            "reranked_evidence_count": len(evidence),
            "evidence": [
                {"marker": item.marker, "chunk_id": item.chunk_id,
                 "citation": item.citation, "citable": item.citable,
                 "temporal_status": item.temporal_status,
                 "source_evidence_level": item.source_evidence_level}
                for item in evidence],
            "evidence_fingerprint": evidence.fingerprint,
            "latency_ms": elapsed_ms,
            "answer": answer.to_dict(),
            "rendered": render_answer(answer, show_confidence=True),
            "golden_target": {"documents": row["expected_documents"],
                              "locators": row["expected_locators"]},
            "scores": scores,
        }
        records.append(record)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    report = {
        "phase": "7.3",
        "experiment_mode": "LEXICAL_ONLY",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": config.model,
        "temperature": config.temperature,
        "max_completion_tokens": config.max_completion_tokens,
        "timeout_seconds": config.timeout_seconds,
        "retries": 0,
        "requests_attempted": len(records),
        "dense_arm": "disabled",
        "dense_calls_returning_empty": dense.calls,
        "reranker": reranker.name,
        "top_k": TOP_K,
        "policy": argv_policy.to_dict(),
        "caveat": "Lexical-only candidate generation. Retrieval figures here are "
                  "lexical_candidate_recall and are NOT comparable with the "
                  "Phase 6 hybrid results.",
        "questions": records,
    }
    report["provider"] = provider.name
    path = os.path.join(RESULTS_DIR,
                        "phase7.3_five_questions.dry_run.json" if dry_run
                        else "phase7.3_five_questions.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)
    print("=" * 78)
    print(f"→ {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preflight", action="store_true",
                        help="check the safety preconditions; make no request")
    parser.add_argument("--dry-run", action="store_true",
                        help="run all five through the deterministic provider; "
                             "makes no network request")
    parser.add_argument("--run", action="store_true",
                        help="run the five questions, one Groq request each")
    args = parser.parse_args(argv)

    from ai.answering.groq import load_env_file

    load_env_file()

    if args.dry_run:
        return run(AnswerPolicy(), dry_run=True)
    if args.preflight or not args.run:
        print("PREFLIGHT — after importing the full pipeline:")
        return preflight()
    return run(AnswerPolicy())


if __name__ == "__main__":
    raise SystemExit(main())
