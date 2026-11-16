"""Phase 4 — metrics, golden-dataset validation, and leakage prevention.

The metric tests use hand-computed values rather than a second implementation,
so a mistake in the formula cannot be reproduced identically in the assertion.

The leakage tests are the important ones. They check something stronger than
"we did not leak in this run": that there is no route by which an expected chunk
id, document or locator can reach a retriever's `search()`. A benchmark that
leaks does not report a worse number, it reports a better one, which is why the
guard has to be structural.
"""
from __future__ import annotations

import inspect
import json
import math
import os

import pytest

from ai.retrieval.base import Filters
from evaluation.benchmark_retrieval import (
    BenchmarkQuery,
    declared_filters,
    run_experiment,
)
from evaluation.metrics import (
    aggregate,
    first_hit_rank,
    ndcg_at_k,
    recall_at_k,
    reciprocal_rank,
    score_question,
)
from evaluation.validate_golden_dataset import GOLDEN_PATH, load_golden, validate

RANKED = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10"]


# ---------------------------------------------------------------- metrics ---

class TestRecallAtK:
    def test_a_hit_at_rank_one_counts_at_every_k(self):
        for k in (1, 5, 10):
            assert recall_at_k(RANKED, ["c1"], k) == 1.0

    def test_a_hit_at_rank_three_counts_only_from_five(self):
        assert recall_at_k(RANKED, ["c3"], 1) == 0.0
        assert recall_at_k(RANKED, ["c3"], 5) == 1.0
        assert recall_at_k(RANKED, ["c3"], 10) == 1.0

    def test_it_is_hit_rate_not_coverage(self):
        # Two expected chunks, one retrieved: still a hit.
        assert recall_at_k(RANKED, ["c2", "zzz"], 5) == 1.0

    def test_a_miss_scores_zero(self):
        assert recall_at_k(RANKED, ["nowhere"], 10) == 0.0

    def test_an_empty_ranking_scores_zero(self):
        assert recall_at_k([], ["c1"], 10) == 0.0


class TestMRR:
    def test_the_first_hit_sets_the_score(self):
        assert reciprocal_rank(RANKED, ["c4"]) == pytest.approx(0.25)
        assert reciprocal_rank(RANKED, ["c1"]) == 1.0

    def test_the_earliest_of_several_expected_chunks_wins(self):
        assert reciprocal_rank(RANKED, ["c7", "c2"]) == pytest.approx(0.5)

    def test_a_miss_contributes_zero(self):
        assert reciprocal_rank(RANKED, ["nowhere"]) == 0.0

    def test_first_hit_rank_reports_none_on_a_miss(self):
        assert first_hit_rank(RANKED, ["nowhere"]) is None
        assert first_hit_rank(RANKED, ["c3"]) == 3


class TestNDCG:
    def test_a_single_expected_chunk_at_rank_one_is_perfect(self):
        assert ndcg_at_k(RANKED, ["c1"], 5) == 1.0

    def test_it_discounts_by_log2_of_the_rank_plus_one(self):
        assert ndcg_at_k(RANKED, ["c3"], 5) == pytest.approx(1 / math.log2(4))

    def test_the_ideal_accounts_for_multiple_expected_chunks(self):
        # Two expected chunks at ranks 1 and 2 is the ideal ranking for them.
        assert ndcg_at_k(RANKED, ["c1", "c2"], 5) == pytest.approx(1.0)
        dcg = 1 / math.log2(2) + 1 / math.log2(4)
        ideal = 1 / math.log2(2) + 1 / math.log2(3)
        assert ndcg_at_k(RANKED, ["c1", "c3"], 5) == pytest.approx(dcg / ideal)

    def test_beyond_k_contributes_nothing(self):
        assert ndcg_at_k(RANKED, ["c9"], 5) == 0.0

    def test_no_expected_chunks_scores_zero_not_nan(self):
        assert ndcg_at_k(RANKED, [], 5) == 0.0


class TestAggregation:
    def test_it_macro_averages_over_questions(self):
        rows = [{"recall@1": 1.0}, {"recall@1": 0.0}, {"recall@1": 0.0}]
        assert aggregate(rows, ["recall@1"])["recall@1"] == pytest.approx(1 / 3)

    def test_an_empty_set_averages_to_zero(self):
        assert aggregate([], ["mrr"]) == {"mrr": 0.0}

    def test_document_level_metrics_are_reported_alongside_chunk_level(self):
        scores = score_question(["a::1"], ["a::2"], ["a"], ["a"])
        assert scores["recall@1"] == 0.0        # wrong provision
        assert scores["document_recall@1"] == 1.0  # right statute


# -------------------------------------------------------- leakage guards ---

class TestLeakagePrevention:
    def test_the_query_object_carries_only_what_a_user_would_type(self):
        assert set(BenchmarkQuery.__dataclass_fields__) == {
            "question_id", "query", "filters"}

    def test_filters_cannot_name_a_document(self):
        assert "document_id" not in Filters.__dataclass_fields__

    def test_declared_filters_read_the_query_and_nothing_else(self):
        signature = inspect.signature(declared_filters)
        assert list(signature.parameters) == ["query"]
        filters = declared_filters("How long does copyright last in a published work?")
        assert filters.domain == ["copyright"]
        assert declared_filters("what is the rule here").is_empty()

    def test_a_retriever_never_sees_the_expected_answer(self):
        """Runs a recording retriever over a record whose answer is a marker string."""
        seen: list[tuple[str, object, int]] = []

        class Recorder:
            name = "recorder"

            def search(self, query, filters=None, top_k=10):
                seen.append((query, filters, top_k))
                return []

            def config(self):
                return {"retriever": self.name}

        record = {
            "question_id": "Q999",
            "query": "how long does a patent last",
            "expected_chunks": ["LEAKED-CHUNK-MARKER"],
            "expected_documents": ["LEAKED-DOCUMENT-MARKER"],
            "expected_locators": ["LEAKED-LOCATOR-MARKER"],
            "query_family": "conceptual",
        }
        rows, _ = run_experiment(Recorder(), [record], top_k=10, filter_mode="none")
        assert seen == [("how long does a patent last", None, 10)]
        for query, filters, _ in seen:
            assert "LEAKED" not in query
            assert filters is None or "LEAKED" not in json.dumps(filters.describe())
        # The answer key is still recorded in the results file, for later analysis.
        assert rows[0]["expected_chunks"] == ["LEAKED-CHUNK-MARKER"]

    def test_declared_filters_are_derived_from_the_query_not_the_record(self):
        record_domain = "plant_varieties"
        derived = declared_filters("Which designs cannot be registered?")
        assert record_domain not in (derived.domain or [])


# ------------------------------------------------- the golden dataset itself ---

@pytest.fixture(scope="module")
def golden():
    if not os.path.exists(GOLDEN_PATH):
        pytest.skip("golden dataset not built")
    return load_golden(GOLDEN_PATH)


@pytest.fixture(scope="module")
def corpus_chunks():
    from ai.retrieval.base import load_chunks

    chunks = load_chunks()
    if not chunks:
        pytest.skip("pipeline output not present")
    return chunks


class TestGoldenDataset:
    def test_it_validates_against_the_current_corpus(self, golden, corpus_chunks):
        assert validate(golden, corpus_chunks) == []

    def test_nothing_is_marked_verified_without_a_human(self, golden):
        # Phase 4 §1: `verified` stays false until a person checks the provision.
        assert all(record["verified"] is False for record in golden)

    def test_question_ids_are_unique(self, golden):
        ids = [record["question_id"] for record in golden]
        assert len(ids) == len(set(ids))

    def test_a_broken_chunk_reference_fails_validation(self, golden, corpus_chunks):
        broken = dict(golden[0], question_id="QBAD",
                      expected_chunks=["NO-SUCH-DOCUMENT::nowhere"])
        checks = {check for _, check, _ in validate([broken], corpus_chunks)}
        assert "unknown_chunk" in checks

    def test_a_wrong_locator_fails_validation(self, golden, corpus_chunks):
        section_question = next(r for r in golden
                                if r["expected_locators"] == ["Section 3"])
        broken = dict(section_question, question_id="QBAD",
                      expected_locators=["Section 999"])
        checks = {check for _, check, _ in validate([broken], corpus_chunks)}
        assert "locator_mismatch" in checks

    def test_an_absent_anchor_fails_validation(self, golden, corpus_chunks):
        broken = dict(golden[0], question_id="QBAD",
                      anchor_terms=["a phrase that appears nowhere at all"])
        checks = {check for _, check, _ in validate([broken], corpus_chunks)}
        assert "anchor_absent" in checks

    def test_expecting_non_citable_evidence_must_be_declared(self, golden, corpus_chunks):
        unlocated = next((r for r in golden if r.get("allow_non_citable")), None)
        if unlocated is None:
            pytest.skip("no question expects non-citable evidence")
        broken = dict(unlocated, question_id="QBAD",
                      allow_non_citable=False, requires_citation=True)
        checks = {check for _, check, _ in validate([broken], corpus_chunks)}
        assert "unmarked_missing_locator" in checks
        assert "non_citable_evidence" in checks


class TestBenchmarkMetricsArePreserved:
    """Phase 5 §8/§12 — the Phase 4 figures must still be reproducible.

    Phase 5 re-ran A-D because the corpus changed under them, but the Phase 4
    numbers remain the published record of that phase. If the metric code drifts,
    a Phase 5 table and a Phase 4 table stop meaning the same thing while looking
    identical — so the archived rows are re-scored here and checked against the
    archived summary. This tests the metrics, not the retrievers: the stored
    rankings are the input.
    """

    ARCHIVE = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "evaluation", "results", "phase4-chunking-0.1.0")

    @pytest.mark.parametrize("retriever", ["bm25", "dense", "bm25_boost", "hybrid"])
    def test_archived_summaries_recompute_from_archived_rankings(self, retriever):
        rows_path = os.path.join(self.ARCHIVE, f"{retriever}.jsonl")
        summary_path = os.path.join(self.ARCHIVE, f"{retriever}.summary.json")
        if not (os.path.exists(rows_path) and os.path.exists(summary_path)):
            pytest.skip("Phase 4 archive not present")
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh)
        scores = []
        with open(rows_path, encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                ranked = [hit["chunk_id"] for hit in row["results"]]
                documents = [hit["document_id"] for hit in row["results"]]
                scores.append(score_question(ranked, row["expected_chunks"],
                                             documents, row["expected_documents"]))
        for metric in ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@5", "ndcg@10"):
            recomputed = aggregate(scores, [metric])[metric]
            assert recomputed == pytest.approx(summary["overall"][metric], abs=5e-5), metric

    def test_the_archive_records_the_corpus_it_was_measured_on(self):
        path = os.path.join(self.ARCHIVE, "bm25.summary.json")
        if not os.path.exists(path):
            pytest.skip("Phase 4 archive not present")
        with open(path, encoding="utf-8") as fh:
            summary = json.load(fh)
        # 3,031 chunks is the Phase 4 corpus. Phase 5 rebuilt it; the archive
        # must keep saying which one produced these numbers.
        assert summary["chunks_indexed"] == 3031
        assert summary["questions_human_verified"] == 0


class TestResultsPreserveCitability:
    """Phase 4 §17: a stored result must still say a chunk cannot be cited."""

    @pytest.mark.parametrize("retriever",
                             ["bm25", "bm25_boost", "dense", "hybrid",
                              "hybrid_boost", "routed"])
    def test_stored_results_report_citability_faithfully(self, retriever, corpus_chunks):
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "evaluation", "results", f"{retriever}.jsonl")
        if not os.path.exists(path):
            pytest.skip(f"no results for {retriever}")
        citable = {c.chunk_id: c.citable for c in corpus_chunks}
        unlocated = {c.chunk_id for c in corpus_chunks if c.locator_unavailable}
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                for hit in json.loads(line)["results"]:
                    assert hit["citable"] is citable[hit["chunk_id"]]
                    if hit["chunk_id"] in unlocated:
                        assert hit["citable"] is False

    def test_the_routed_run_records_the_path_every_answer_arrived_by(self):
        """Phase 5 §7 — a fallback that is not visible is a silent engine swap."""
        path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "data", "evaluation", "results", "routed.jsonl")
        if not os.path.exists(path):
            pytest.skip("no routed results")
        from ai.retrieval.router import PATHS, QUERY_TYPES
        with open(path, encoding="utf-8") as fh:
            rows = [json.loads(line) for line in fh if line.strip()]
        assert rows
        for row in rows:
            assert row["retrieval_path"] in PATHS
            assert row["routing"]["query_type"] in QUERY_TYPES
            assert row["routing"]["confidence"] == "deterministic"
            assert row["routing"]["reason"]
            for hit in row["results"]:
                assert hit["retrieval_path"] == row["retrieval_path"]
