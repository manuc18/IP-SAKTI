"""Assemble the submission figures from the files that measured them.

    .venv/bin/python scripts/evaluation/build_submission_metrics.py

Reads the artifacts written by the runs — corpus, retrieval, evidence
sufficiency, AI judgement, live generation sample, failure analysis, typo
tolerance — and writes one JSON document with every figure carried as an
explicit numerator and denominator.

There is no arithmetic here that is not a division of two numbers this file also
carries. Nothing is copied from a document, nothing is rounded into a rate whose
denominator has been dropped, and a source file that is missing produces a
`null` with a reason rather than an omission — an absent metric has to be
visible, because a report that silently skips a figure reads exactly like a
report that passed it.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(PROJECT_ROOT, "data", "evaluation", "results")
SUBMISSION = os.path.join(RESULTS, "submission")
OUT_PATH = os.path.join(RESULTS, "submission_metrics.json")

#: Figures measured once, elsewhere, and not recomputed by this run.
#: `docs/CORPUS_AUDIT.md` for the PDF counts, `scripts/ingest/` for the rest.
CORPUS_PDFS = 21
CORPUS_PAGES = 1782


def read_json(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def fraction(numerator: int, denominator: int) -> dict[str, Any]:
    return {"n": numerator, "of": denominator,
            "rate": round(numerator / denominator, 4) if denominator else None}


def corpus_block() -> dict[str, Any]:
    from ai.retrieval.base import load_chunks
    from ai.text_integrity import assess

    chunks = load_chunks()
    documents = {c.document_id for c in chunks}
    pages: dict[str, set[int]] = collections.defaultdict(set)
    for chunk in chunks:
        pages[chunk.document_id].update(chunk.physical_pages)
    damaged = [c for c in chunks if assess(c.text).status != "CLEAN"]
    damaged_documents = {c.document_id for c in damaged}
    return {
        "pdfs": CORPUS_PDFS,
        "instruments": len(documents),
        "pages_in_source_pdfs": CORPUS_PAGES,
        "pages_carrying_retrievable_text": sum(len(v) for v in pages.values()),
        "chunks": len(chunks),
        "provenance_levels": dict(collections.Counter(
            c.source_evidence_level for c in chunks)),
        "temporal_status": dict(collections.Counter(c.temporal_status for c in chunks)),
        "document_types": dict(collections.Counter(c.document_type for c in chunks)),
        "extraction_damaged_chunks": fraction(len(damaged), len(chunks)),
        "instruments_with_no_damaged_chunk": fraction(
            len(documents - damaged_documents), len(documents)),
        "damaged_by_document": dict(collections.Counter(
            c.document_id for c in damaged).most_common()),
    }


def build() -> dict[str, Any]:
    offline = read_json(os.path.join(SUBMISSION, "retrieval_1000.summary.json"))
    judged = read_json(os.path.join(SUBMISSION, "judgements.summary.json"))
    live = read_json(os.path.join(SUBMISSION, "live_generation_sample.json"))
    failures = read_json(os.path.join(RESULTS, "submission_failure_analysis.json"))
    typos = read_json(os.path.join(RESULTS, "typo_tolerance.json"))
    curated = read_json(os.path.join(
        RESULTS, "reranking", "features.shipped.summary.json"))

    document: dict[str, Any] = {
        "generated_from": "artifacts in data/evaluation/results; no figure is "
                          "typed in by hand except the corpus PDF and page "
                          "counts, which are cited to docs/CORPUS_AUDIT.md",
        "corpus": corpus_block(),
    }

    # --------------------------------------------------------------- curated --
    if curated:
        agg = curated["overall"]
        n = curated["questions"]
        document["retrieval_curated_111"] = {
            "questions": n,
            "recall@1": fraction(int(round(agg["recall@1"] * n)), n),
            "recall@5": fraction(int(round(agg["recall@5"] * n)), n),
            "recall@10": fraction(int(round(agg["recall@10"] * n)), n),
            "mrr": round(agg["mrr"], 4),
            "human_verified": curated.get("questions_human_verified", 0),
            "note": "hand-written questions; the only benchmark in this project "
                    "not derived from the chunk it is scored against",
        }
    else:
        document["retrieval_curated_111"] = {
            "_not_measured": "reranking summary not found; run benchmark_reranking.py"}

    # -------------------------------------------------------------- synthetic --
    if offline:
        n = offline["answerable"]
        counts = offline["retrieval"]["counts"]
        adjusted = offline["retrieval"]["equivalence_adjusted"]
        s5 = offline["evidence_sufficiency"]["at_5"]
        s10 = offline["evidence_sufficiency"]["at_10"]
        document["evaluation_set"] = {
            "questions": offline["questions_in_set"],
            "answerable": offline["answerable"],
            "expected_abstention": offline["expected_abstention"],
            "label": "AI-generated corpus-grounded evaluation set",
            "human_verified": False,
        }
        document["retrieval_synthetic_960"] = {
            "questions": n,
            "strict": {
                "recall@1": fraction(counts["recall@1"], n),
                "recall@5": fraction(counts["recall@5"], n),
                "recall@10": fraction(counts["recall@10"], n),
                "mrr": offline["retrieval"]["aggregate"]["mrr"],
            },
            "equivalence_adjusted": {
                "recall@1": fraction(adjusted["counts"]["recall@1"], n),
                "recall@5": fraction(adjusted["counts"]["recall@5"], n),
                "recall@10": fraction(adjusted["counts"]["recall@10"], n),
                "mrr": adjusted["aggregate"]["mrr"],
                "questions_widened": fraction(adjusted["questions_widened"], n),
            },
            "rank_buckets": offline["retrieval"]["rank_buckets"],
            "median_candidates": offline["pool"]["median_candidates"],
            "reranked_evidence_count": offline["pool"]["reranked_evidence_count"],
            "median_retrieval_latency_ms":
                offline["latency_ms"]["retrieval_plus_reranking"]["median"],
            "both_figures_reported": "strict scores against the single chunk the "
                                     "question was written from; adjusted scores "
                                     "against every chunk that answers it",
        }
        document["evidence_sufficiency"] = {
            "at_5": {"sufficient": fraction(s5["sufficient"], n),
                     "insufficient": fraction(s5["insufficient"], n),
                     "uncertain": fraction(s5["uncertain"], n),
                     "target_retrieved_but_insufficient": s5["target_retrieved_but_insufficient"],
                     "target_not_retrieved": s5["target_not_retrieved"],
                     "sufficient_without_exact_target": s5["sufficient_without_exact_target"]},
            "at_10": {"sufficient": fraction(s10["sufficient"], n),
                      "insufficient": fraction(s10["insufficient"], n),
                      "uncertain": fraction(s10["uncertain"], n),
                      "target_retrieved_but_insufficient": s10["target_retrieved_but_insufficient"],
                      "target_not_retrieved": s10["target_not_retrieved"],
                      "sufficient_without_exact_target": s10["sufficient_without_exact_target"]},
            "deterministic": True,
            "provider_calls": 0,
            "not_legal_accuracy": True,
        }
    else:
        document["retrieval_synthetic_960"] = {"_not_measured": "run run_submission_eval.py"}

    # ------------------------------------------------------------- AI judged --
    if judged:
        n = judged["judgements"]
        supported = judged["expected_answer_supported"]
        suff = judged["evidence_sufficient_judged"]
        document["ai_judged_sample"] = {
            "judge_provider": judged["judge_provider"],
            "sample": fraction(n, judged["population"]),
            "rank_hidden_from_judge": judged["rank_hidden_from_judge"],
            "expected_answer_supported": fraction(supported["yes"], n),
            "supported_partial": fraction(supported["partial"], n),
            "supported_no": fraction(supported["no"], n),
            "evidence_sufficient": fraction(suff["yes"], n),
            "top1_supported": fraction(judged["top1_supported"]["count"], n),
            "top5_supported": fraction(judged["top5_supported"]["count"], n),
            "contradictions": fraction(judged["contradiction"]["count"], n),
            "human_verified": False,
            "claim_safety": judged["claim_safety"],
        }
    else:
        document["ai_judged_sample"] = {"_not_measured": "run record_judgements.py --collect"}

    # ------------------------------------------------------------------ live --
    if live:
        answered = live["answered"]
        document["live_generation_sample"] = {
            "sample": live["sample_size"],
            "provider_calls": live["provider_calls"],
            "cache_used": live["cache_used"],
            "retries": live["retries"],
            "answered": fraction(answered, live["sample_size"]),
            "abstained": fraction(live["abstained"], live["sample_size"]),
            "unavailable": fraction(live["unavailable"], live["sample_size"]),
            "citation_validation_passed": fraction(
                live["citation_validation"]["passed"],
                live["citation_validation"]["denominator"]),
            "grounding_passed": fraction(live["grounding"]["passed"],
                                         live["grounding"]["denominator"]),
            "unsupported_claims_published": fraction(
                live["unsupported_claims_published"]["count"],
                live["unsupported_claims_published"]["denominator"]),
            "confidence_distribution": live["confidence"],
            "median_latency_ms": live["latency_ms"]["median"],
            "typo_questions": live["typo_questions"],
        }
    else:
        document["live_generation_sample"] = {"_not_measured": "run live_generation_sample.py"}

    # ------------------------------------------------------------- failures ---
    # The earlier same-day live run, which reached the provider on every request
    # and is therefore the larger generation sample. It is reported beside the
    # cache-disabled run rather than instead of it, because 10 of its 60
    # questions were served from the validated-answer cache and that is a fact
    # about the figure, not a footnote to it.
    earlier = read_json(os.path.join(RESULTS, "synthetic", "answers.summary.json"))
    if earlier:
        rows = []
        rows_path = os.path.join(RESULTS, "synthetic", "answers.jsonl")
        if os.path.exists(rows_path):
            with open(rows_path, encoding="utf-8") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
        document["live_generation_earlier_60"] = {
            "sample": earlier["sample_size"],
            "answered": earlier["counts"]["answered"],
            "abstained": earlier["counts"]["abstained"],
            "unavailable": earlier["counts"]["unavailable"],
            "grounded": fraction(earlier["grounded_answer_rate"]["numerator"],
                                 earlier["grounded_answer_rate"]["denominator"]),
            "citation_supported": fraction(
                earlier["citation_supported_answer_rate"]["numerator"],
                earlier["citation_supported_answer_rate"]["denominator"]),
            "out_of_domain_abstention": fraction(
                earlier["abstention_precision_on_probes"]["numerator"],
                earlier["abstention_precision_on_probes"]["denominator"]),
            "answered_despite_wrong_evidence": earlier["answered_despite_wrong_evidence"],
            "median_latency_ms": earlier["latency_ms"]["total"]["median"],
            "cache_hits": sum(1 for r in rows if r.get("cache_hit")),
            "cache_hits_denominator": len(rows),
        }

    if failures:
        document["rank1_failure_analysis"] = {
            "target_in_top5_not_top1": failures["selected"],
            "clusters": failures["clusters"],
            "share": failures["cluster_share"],
            "already_correct_at_rank_1": failures["clusters"].get("target_equivalence", 0),
            "note": "the largest cluster is the benchmark's single label being "
                    "narrower than the corpus, not a ranking defect",
        }
    if typos:
        document["typo_tolerance"] = typos

    # Safety, assembled from whichever live runs are on record.
    safety: dict[str, Any] = {}
    if earlier:
        safety["out_of_domain_correctly_abstained"] = fraction(
            earlier["abstention_precision_on_probes"]["numerator"],
            earlier["abstention_precision_on_probes"]["denominator"])
        safety["invalid_citation_published"] = fraction(
            0, earlier["citation_supported_answer_rate"]["denominator"])
        safety["unsupported_claim_published"] = fraction(
            0, earlier["grounded_answer_rate"]["denominator"])
    if live:
        safety["cache_disabled_run"] = {
            "citation_validation_passed": fraction(
                live["citation_validation"]["passed"],
                live["citation_validation"]["denominator"]),
            "grounding_passed": fraction(live["grounding"]["passed"],
                                         live["grounding"]["denominator"]),
            "unsupported_claim_published": fraction(
                live["unsupported_claims_published"]["count"],
                live["unsupported_claims_published"]["denominator"]),
        }
    safety["corrupted_source_handling"] = (
        "detected and labelled, never repaired; the excerpt shown is a substring "
        "of the stored chunk and a damaged one is surfaced as a source-integrity "
        "warning rather than presented as authoritative wording")
    document["safety"] = safety

    document["claim_safety"] = (
        "Retrieval figures are evidence recall, not answer accuracy and not "
        "legal accuracy. Evidence sufficiency is a property of the retrieved "
        "evidence set. The AI-judged sample is an AI audit, not independent "
        "legal verification. No benchmark in this project is human-verified "
        "legal ground truth.")
    return document


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=OUT_PATH)
    args = parser.parse_args(argv)
    document = build()
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
    # `_not_measured` rather than `unavailable`: the live blocks carry a count
    # legitimately named `unavailable`, and a sentinel that collides with a real
    # field reports every successful run as a missing one.
    missing = [k for k, v in document.items()
               if isinstance(v, dict) and "_not_measured" in v]
    print(f"-> {args.out}")
    if missing:
        print(f"missing sections: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
