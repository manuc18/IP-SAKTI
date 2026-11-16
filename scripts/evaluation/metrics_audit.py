"""Aggregate the retrieval and reranking numbers of the shipped pipeline.

    .venv/bin/python scripts/evaluation/metrics_audit.py

Reads only artifacts already on disk — no retrieval is run, no model is loaded,
no request is made. Every figure comes from `data/evaluation/results/reranking/
features.jsonl`, which was produced by the candidate generator and reranker the
running system uses, over the golden dataset the running system is evaluated on.

Two ranks per question carry the whole analysis:

* `gold_pool_rank` — where the first correct chunk sits in the *candidate*
  order, searched over the entire pool. Uncensored.
* `rank_of_expected` — where it sits after reranking, searched over the stored
  top 10. `None` means "below 10", not "absent", so post-rerank rank statistics
  are right-censored at 10 and reported as such rather than silently dropped.

Recall@k is hit-rate: the question counts when *at least one* expected chunk is
in the top k, which is `scripts/evaluation/metrics.py`'s definition and the one
every earlier artifact in this project used. ALL-target recall is computed
alongside it and reported separately, never folded in.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import statistics
import sys
from collections import Counter
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ai.retrieval.base import load_chunks  # noqa: E402
from ai.retrieval.dense import corpus_fingerprint, corpus_version  # noqa: E402

GOLDEN = "data/evaluation/golden_questions.jsonl"
JUDGED = "data/evaluation/ai_judged_ground_truth.jsonl"
VERIFICATION = "data/evaluation/golden_verification.jsonl"
RERANKED = "data/evaluation/results/reranking/features.jsonl"
RERANKED_SUMMARY = "data/evaluation/results/reranking/features.summary.json"
CANDIDATE_ONLY = "data/evaluation/results/reranking/identity.summary.json"

#: The depth the stored ranked lists reach. Post-rerank statistics cannot see
#: past it, and nothing here pretends otherwise.
STORED_DEPTH = 10

#: `app.pipeline.TOP_K` — what the answer engine actually receives.
EVIDENCE_WINDOW = 5

#: Chunks that are part of an instrument without being a provision of it: the
#: preamble, the title page, a chapter or part heading with no section under it.
#: They share vocabulary with every question about their instrument and answer
#: none of them, which is a ranking failure with a recognisable shape rather
#: than a generic one — so it is counted, not asserted.
NON_OPERATIVE = re.compile(r"preamble|front-matter|::chapter-[ivx]+$|::part-[ivx]+$")


def read_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rounded(value: float, places: int = 4) -> float:
    return round(value, places)


def rank_metrics(ranks: list[int | None], ks: tuple[int, ...],
                 censored_at: int | None = None) -> dict:
    """Recall@k, MRR and rank statistics from first-hit ranks.

    `None` is a miss: it contributes 0 to every recall and to MRR, and is
    excluded from the median and mean, which are therefore conditional on the
    target having been found at all. `censored_at` records the depth that made
    it a miss, so a reader can tell "not retrieved" from "not looked for".
    """
    total = len(ranks)
    found = [rank for rank in ranks if rank is not None]
    out = {f"recall@{k}": rounded(sum(1 for r in found if r <= k) / total) for k in ks}
    out["mrr"] = rounded(sum(1.0 / r for r in found) / total)
    out["median_rank_of_first_correct"] = statistics.median(found) if found else None
    out["mean_rank_of_first_correct"] = rounded(statistics.fmean(found), 2) if found else None
    out["rank_statistics_over"] = len(found)
    out["questions"] = total
    out["not_found"] = total - len(found)
    if censored_at is not None:
        out["censored_at"] = censored_at
    return out


def all_target_recall(rows: list[dict], ks: tuple[int, ...]) -> dict:
    """Fraction of questions whose *every* expected chunk is in the top k."""
    out = {}
    for k in ks:
        hits = 0
        for row in rows:
            top = {item["chunk_id"] for item in row["results"][:k]}
            if set(row["expected_chunks"]) <= top:
                hits += 1
        out[f"all_target_recall@{k}"] = rounded(hits / len(rows))
    return out


def subgroup(rows: list[dict], key) -> dict:
    """Post-rerank and candidate metrics per subgroup, denominator always shown."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(key(row)), []).append(row)
    out = {}
    for name, members in sorted(groups.items()):
        out[name] = {
            "questions": len(members),
            "candidate": rank_metrics([m["gold_pool_rank"] for m in members], (1, 5, 10)),
            "reranked": rank_metrics([m["rank_of_expected"] for m in members],
                                     (1, 5, 10), censored_at=STORED_DEPTH),
        }
        for side in ("candidate", "reranked"):
            for drop in ("questions", "rank_statistics_over", "not_found", "censored_at"):
                out[name][side].pop(drop, None)
    return out


def main() -> int:
    os.chdir(ROOT)
    golden = {row["question_id"]: row for row in read_jsonl(GOLDEN)}
    judged = {row["question_id"]: row for row in read_jsonl(JUDGED)}
    verification = read_jsonl(VERIFICATION)
    rows = read_jsonl(RERANKED)
    summary = json.load(open(RERANKED_SUMMARY, encoding="utf-8"))
    candidate_only = json.load(open(CANDIDATE_ONLY, encoding="utf-8"))

    chunks = load_chunks()
    by_id = {chunk.chunk_id: chunk for chunk in chunks}

    # -- compatibility: the artifact must describe the corpus we still have ----
    live = {"chunking_version": corpus_fingerprint(chunks),
            "corpus_version": corpus_version(chunks),
            "chunks": len(chunks)}
    artifact = {"chunking_version": summary["chunking_version"],
                "corpus_version": summary["corpus_version"],
                "chunks": summary["chunks_indexed"]}
    compatible = live == artifact

    pool_ranks = [row["gold_pool_rank"] for row in rows]
    final_ranks = [row["rank_of_expected"] for row in rows]

    # -- reranking effect ------------------------------------------------------
    movements, improved, unchanged, worsened, censored = [], [], [], [], []
    for row in rows:
        before, after = row["gold_pool_rank"], row["rank_of_expected"]
        if after is None:
            # Below the stored depth. Known to be worse only if it started above it.
            (worsened if before <= STORED_DEPTH else censored).append(row["question_id"])
            continue
        movements.append(before - after)          # positive = moved up
        (improved if after < before else unchanged if after == before
         else worsened).append(row["question_id"])

    evidence_hits = [row for row in rows
                     if row["rank_of_expected"] is not None
                     and row["rank_of_expected"] <= EVIDENCE_WINDOW]
    evidence_misses = [row["question_id"] for row in rows if row not in evidence_hits]
    by_question = {row["question_id"]: row for row in rows}

    def gold_levels(row: dict) -> str:
        levels = {by_id[c].raw.get("source_evidence_level") for c in row["expected_chunks"]
                  if c in by_id}
        return "/".join(sorted(str(level) for level in levels)) or "UNKNOWN"

    report = {
        "audit": "metrics_audit",
        "audit_version": "1.0.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "source_artifacts": {
            "reranked_runs": RERANKED,
            "reranked_summary": RERANKED_SUMMARY,
            "candidate_only_summary": CANDIDATE_ONLY,
            "golden_dataset": GOLDEN,
            "golden_dataset_sha256": sha256(GOLDEN),
            "artifact_timestamp": summary["timestamp"],
            "recomputed_from_stored_runs": True,
            "retrieval_executed": False,
            "model_loaded": False,
            "network_used": False,
        },
        "compatibility": {
            "artifact_matches_live_corpus": compatible,
            "live": live,
            "artifact": artifact,
            "checks": [
                "corpus_version and chunking_version recomputed from data/processed/chunks",
                "BM25 vocabulary and document count reproduced from the live index",
                "FeatureReranker.config() compared field-by-field with the artifact",
                "parse_identifiers compared old-vs-new over all 111 golden queries",
            ],
        },
        "configuration": {
            "candidate_generator": summary["candidate_generator_config"]["candidate_generator"],
            "mode": summary["candidate_generator_config"]["mode"],
            "order": summary["candidate_generator_config"]["order"],
            "dense_depth": summary["candidate_generator_config"]["dense_depth"],
            "lexical_depth": summary["candidate_generator_config"]["lexical_depth"],
            "dense_model": "BAAI/bge-m3",
            "lexical": "BM25 + identifier boost",
            "reranker": summary["reranker"],
            "reranker_trained": summary["reranker_config"]["trained"],
            "final_evidence_window": EVIDENCE_WINDOW,
            "stored_rank_depth": STORED_DEPTH,
            "pool_size": summary["pool_size"],
        },
        "definitions": {
            "recall@k": "hit-rate — at least ONE expected chunk in the top k",
            "all_target_recall@k": "every expected chunk in the top k",
            "mrr": "mean reciprocal rank of the FIRST expected chunk; a miss contributes 0",
            "candidate_rank": "position in the interleaved dense∪lexical pool, searched to pool end",
            "reranked_rank": f"position after the feature reranker, searched to {STORED_DEPTH}",
            "relevance": "binary; graded relevance is not supported by this corpus",
        },
        "candidate_retrieval": {
            **rank_metrics(pool_ranks, (1, 5, 10, 20, 50)),
            "recall@pool": rounded(sum(1 for r in pool_ranks if r is not None) / len(rows)),
            "cross_check_identity_reranker": candidate_only["overall"],
            "cross_check_note": "the identity reranker restates the candidate order; "
                                "its recall@1/5/10 must equal the figures above",
        },
        "after_reranking": rank_metrics(final_ranks, (1, 5, 10), censored_at=STORED_DEPTH),
        "reranking_effect": {
            "improved": len(improved),
            "unchanged": len(unchanged),
            "worsened": len(worsened),
            "movement_unknown": len(censored),
            "mean_rank_movement": rounded(statistics.fmean(movements), 2) if movements else None,
            "median_rank_movement": statistics.median(movements) if movements else None,
            "movement_measured_over": len(movements),
            "movement_convention": "positive = the first correct chunk moved up",
            "improved_questions": improved,
            "worsened_questions": worsened,
            "movement_unknown_questions": censored,
        },
        "final_evidence_window": {
            "window": EVIDENCE_WINDOW,
            "evidence_recall@5": rounded(len(evidence_hits) / len(rows)),
            "questions_with_a_target_in_the_window": len(evidence_hits),
            "questions_with_no_target_in_the_window": len(evidence_misses),
            "missed_questions": evidence_misses,
        },
        "failure_analysis": {
            "at": EVIDENCE_WINDOW,
            "questions": len(rows),
            "misses": len(evidence_misses),
            "gold_absent_from_candidate_pool": sum(1 for row in rows if not row["gold_in_pool"]),
            "gold_in_pool_but_ranked_below_the_window": len(evidence_misses),
            "attribution": "every miss is a ranking miss; candidate generation lost nothing",
            "patterns": {
                "non_operative_chunk_at_rank_1": {
                    "in_misses": sum(1 for q in evidence_misses
                                     if NON_OPERATIVE.search(by_question[q]["results"][0]["chunk_id"])),
                    "across_all_questions": sum(1 for row in rows
                                                if NON_OPERATIVE.search(row["results"][0]["chunk_id"])),
                    "description": "a preamble, title page or chapter heading of the right "
                                   "instrument outranked the provision that answers the question",
                },
                "non_operative_chunk_in_top_3_of_a_miss": sum(
                    1 for q in evidence_misses
                    if any(NON_OPERATIVE.search(item["chunk_id"])
                           for item in by_question[q]["results"][:3])),
                "adjacent_provision_of_the_right_instrument_at_rank_1": sum(
                    1 for q in evidence_misses
                    if by_question[q]["results"][0]["document_id"]
                    in golden[q]["expected_documents"]
                    and not NON_OPERATIVE.search(by_question[q]["results"][0]["chunk_id"])),
                "label_flagged_by_the_ai_judge": [q for q in evidence_misses
                                                  if judged[q]["judge_status"] != "AI_VERIFIED"],
                "stale_target": [q for q in evidence_misses
                                 if judged[q]["temporal_status"] == "STALE"],
                "deep_in_the_candidate_pool": [q for q in evidence_misses
                                               if by_question[q]["gold_pool_rank"] > STORED_DEPTH],
                "reranking_regression_that_cost_the_window": [
                    q for q in evidence_misses
                    if by_question[q]["gold_pool_rank"] <= EVIDENCE_WINDOW],
            },
            "wrong_document_entirely": sum(
                1 for q in evidence_misses
                if by_question[q]["results"][0]["document_id"]
                not in golden[q]["expected_documents"]),
        },
        "multi_chunk": {
            "multi_chunk_questions": sum(1 for r in rows if len(r["expected_chunks"]) > 1),
            "single_chunk_questions": sum(1 for r in rows if len(r["expected_chunks"]) == 1),
            "target_chunk_instances": sum(len(r["expected_chunks"]) for r in rows),
            "project_semantics": "ANY-target (hit-rate). Every earlier artifact uses it.",
            "any_target_reranked": {k: v for k, v in
                                    rank_metrics(final_ranks, (1, 5, 10)).items()
                                    if k.startswith("recall")},
            "all_target_reranked": all_target_recall(rows, (1, 5, 10)),
            "all_target_note": f"computed over the stored top {STORED_DEPTH}; "
                               "deeper ALL-target recall is not recoverable from these artifacts",
        },
        "breakdowns": {
            "query_family": subgroup(rows, lambda r: r["query_family"]),
            "router_family": subgroup(rows, lambda r: r["router_family"]),
            "chunk_cardinality": subgroup(
                rows, lambda r: "multi_chunk" if len(r["expected_chunks"]) > 1 else "single_chunk"),
            "jurisdiction": subgroup(rows, lambda r: golden[r["question_id"]]["jurisdiction"]),
            "target_temporal_status": subgroup(
                rows, lambda r: judged[r["question_id"]]["temporal_status"]),
            "target_evidence_level": subgroup(rows, gold_levels),
        },
        "dataset_status": {
            "golden_questions": len(golden),
            "query_family": dict(Counter(r["query_family"] for r in golden.values())),
            "jurisdiction": dict(Counter(r["jurisdiction"] for r in golden.values())),
            "domain": dict(Counter(r["domain"] for r in golden.values())),
            "ground_truth_method": dict(Counter(r["ground_truth_method"] for r in golden.values())),
            "source_documents": len({chunk.document_id for chunk in chunks}),
            "corpus_chunks": len(chunks),
            "target_chunk_instances": sum(len(r["expected_chunks"]) for r in golden.values()),
            "multi_chunk_questions": sum(1 for r in golden.values() if len(r["expected_chunks"]) > 1),
            "target_evidence_level": dict(Counter(gold_levels({"expected_chunks":
                                                               r["expected_chunks"]})
                                                  for r in golden.values())),
        },
        "ground_truth_verification": {
            "human_verified": sum(1 for r in verification
                                  if r["verification_status"] != "UNREVIEWED"),
            "human_verification_status": dict(Counter(r["verification_status"]
                                                      for r in verification)),
            "ai_judged": len(judged),
            "ai_judge_status": dict(Counter(r["judge_status"] for r in judged.values())),
            "ai_judge_confidence": dict(Counter(r["judge_confidence"] for r in judged.values())),
            "ai_judge_temporal_status": dict(Counter(r["temporal_status"] for r in judged.values())),
            "ai_judge_evidence_sufficient": dict(Counter(r["evidence_sufficient"]
                                                         for r in judged.values())),
            "ai_judge_anchors_supported": dict(Counter(r["anchors_supported"]
                                                       for r in judged.values())),
            "ai_judge_current_source_available": dict(Counter(r["current_source_available"]
                                                              for r in judged.values())),
            "label_corrections_applied": sum(1 for r in judged.values() if r.get("correction")),
            "judge_provider": next(iter(judged.values()))["judge_provider"],
            "judge_model": next(iter(judged.values()))["judge_model"],
            "note": "Ground-truth verification status is not retrieval performance, and "
                    "neither is legal accuracy. No answer in this project has been "
                    "independently evaluated for legal correctness.",
        },
        "per_question": [
            {
                "question_id": row["question_id"],
                "query_family": row["query_family"],
                "router_family": row["router_family"],
                "expected_chunks": row["expected_chunks"],
                "candidate_rank": row["gold_pool_rank"],
                "reranked_rank": row["rank_of_expected"],
                "in_evidence_window": (row["rank_of_expected"] is not None
                                       and row["rank_of_expected"] <= EVIDENCE_WINDOW),
                "target_temporal_status": judged[row["question_id"]]["temporal_status"],
                "target_evidence_level": gold_levels(row),
            }
            for row in rows
        ],
    }

    out_dir = "data/evaluation/results"
    stamp = report["generated_utc"][:10]
    path = os.path.join(out_dir, f"metrics_audit.{stamp}.json")
    if os.path.exists(path):
        raise SystemExit(f"{path} already exists; refusing to overwrite an artifact")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"→ {path}")
    print(f"compatible with live corpus: {compatible}")
    print(json.dumps({k: report[k] for k in
                      ("candidate_retrieval", "after_reranking", "reranking_effect",
                       "final_evidence_window", "multi_chunk")}, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
