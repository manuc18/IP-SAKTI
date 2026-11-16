"""Phase 4 — the retrieval layer.

The tests that matter most here are the ones about what retrieval must *not* do:
lose provenance, invent citability, or let a filter change the meaning of a
score. Ranking quality is measured by the benchmark, not asserted here.

A small synthetic corpus stands in for the real one wherever the behaviour under
test is a property of the code rather than of the documents, so the suite runs
without the 3,031-chunk index or the BGE-M3 weights.
"""
from __future__ import annotations

import os

import pytest

from ai.retrieval.base import (
    Chunk,
    Filters,
    apply_filters,
    load_chunks,
    result_from_chunk,
)
from ai.retrieval.bm25 import BM25Retriever
from ai.retrieval.hybrid import HybridRRFRetriever, reciprocal_rank_fusion
from ai.retrieval.identifiers import (
    BM25IdentifierBoostRetriever,
    document_aliases,
    identifier_boost,
    match_documents,
    parse_identifiers,
)
from ai.retrieval.tokenizer import tokenize

EMPTY_LOCATOR = {
    "part": None, "chapter": None, "section": None, "subsection": None, "clause": None,
    "rule": None, "sub_rule": None, "regulation": None, "article": None,
    "paragraph": None, "subparagraph": None, "schedule": None, "form": None,
    "annexure": None, "item": None,
}


def make_chunk(chunk_id: str, text: str, **overrides) -> Chunk:
    locator = dict(EMPTY_LOCATOR)
    locator.update(overrides.pop("locator", {}))
    fields = {
        "chunk_id": chunk_id,
        "document_id": overrides.pop("document_id", "DOC-A"),
        "text": text,
        "citation": overrides.pop("citation", chunk_id),
        "locator": locator,
        "locator_unavailable": overrides.pop("locator_unavailable", False),
        "citable": overrides.pop("citable", True),
        "retrievable": True,
        "domain": overrides.pop("domain", "patents"),
        "jurisdiction": overrides.pop("jurisdiction", "IN"),
        "document_type": overrides.pop("document_type", "act"),
        "temporal_status": overrides.pop("temporal_status", "CURRENT"),
        "source_evidence_level": overrides.pop("source_evidence_level", "A"),
        "source_status": "VERIFIED_OFFICIAL",
        "physical_pages": overrides.pop("physical_pages", [1]),
        "title": overrides.pop("title", "Test Act, 1970"),
        "short_title": overrides.pop("short_title", "Test Act, 1970"),
        "heading": None,
        "unit_type": "provision",
        "structural_path": "",
        "raw": {},
    }
    fields.update(overrides)
    return Chunk(**fields)


@pytest.fixture(scope="module")
def toy_chunks() -> list[Chunk]:
    return [
        make_chunk("DOC-A::s3", "What are not inventions. A mere discovery of a scientific "
                                "principle; traditional knowledge.",
                   locator={"section": "3"}, citation="Test Act — Section 3"),
        make_chunk("DOC-A::s53", "Term of patent. Every patent shall be for twenty years.",
                   locator={"section": "53"}, citation="Test Act — Section 53"),
        make_chunk("DOC-A::front", "Front matter with no locator at all.",
                   locator_unavailable=True, citable=False,
                   citation="Test Act — unlocated text"),
        make_chunk("DOC-B::r13", "Specifications shall be filed in the prescribed manner.",
                   document_id="DOC-B", locator={"rule": "13"}, domain="patents",
                   document_type="rules", temporal_status="STALE",
                   short_title="Test Rules, 2003", citation="Test Rules — Rule 13"),
        make_chunk("DOC-C::a27", "Patents shall be available for any inventions in all "
                                 "fields of technology.",
                   document_id="DOC-C", locator={"article": "27"}, domain="international_ip",
                   jurisdiction="international", document_type="treaty",
                   short_title="TRIPS", citation="TRIPS — Article 27"),
    ]


# ------------------------------------------------------------------ tokenizer ---

class TestLegalTokenizer:
    @pytest.mark.parametrize("text,expected", [
        ("Section 3(p)", "3(p)"),
        ("Article 27.1", "27.1"),
        ("Article 31bis", "31bis"),
        ("sub-section 15(7)", "15(7)"),
        ("Rule 13(2)", "13(2)"),
        ("section 117A", "117a"),
        ("section 33EEB", "33eeb"),
    ])
    def test_legal_identifiers_survive_as_single_tokens(self, text, expected):
        assert expected in tokenize(text)

    def test_the_leading_number_is_emitted_alongside_the_composite(self):
        # so a query for 3(p) still matches a provision that only writes "3."
        assert tokenize("section 3(p)") == ["section", "3(p)", "3"]

    def test_acronyms_are_preserved_including_ampersands(self):
        assert "ppv&fr" in tokenize("the PPV&FR Act")
        assert "trips" in tokenize("TRIPS Agreement")

    def test_nothing_is_stemmed(self):
        assert "inventions" in tokenize("inventions")
        assert "invention" not in tokenize("inventions")

    def test_the_corpus_hyphen_variants_agree(self):
        assert tokenize("sub-section") == tokenize("sub section") == ["sub", "section"]


# ----------------------------------------------------------------------- bm25 ---

class TestBM25:
    def test_it_ranks_the_matching_chunk_first(self, toy_chunks):
        results = BM25Retriever(toy_chunks).search("twenty years term of patent", top_k=3)
        assert results[0].chunk_id == "DOC-A::s53"

    def test_ranks_are_dense_and_start_at_one(self, toy_chunks):
        results = BM25Retriever(toy_chunks).search("patent", top_k=5)
        assert [r.rank for r in results] == list(range(1, len(results) + 1))

    def test_scores_are_non_increasing(self, toy_chunks):
        scores = [r.score for r in BM25Retriever(toy_chunks).search("patent invention", top_k=10)]
        assert scores == sorted(scores, reverse=True)

    def test_a_query_matching_nothing_returns_nothing(self, toy_chunks):
        assert BM25Retriever(toy_chunks).search("zzzznonexistent", top_k=5) == []

    def test_ranking_is_deterministic_across_index_builds(self, toy_chunks):
        first = [r.chunk_id for r in BM25Retriever(toy_chunks).search("patent", top_k=5)]
        second = [r.chunk_id for r in BM25Retriever(list(toy_chunks)).search("patent", top_k=5)]
        assert first == second

    def test_idf_does_not_depend_on_the_filter(self, toy_chunks):
        retriever = BM25Retriever(toy_chunks)
        unfiltered = {r.chunk_id: r.score for r in retriever.search("patent", top_k=10)}
        filtered = retriever.search("patent", Filters(document_type="act"), top_k=10)
        for result in filtered:
            assert result.score == pytest.approx(unfiltered[result.chunk_id])


# --------------------------------------------------------------------- filters ---

class TestMetadataFiltering:
    def test_each_field_restricts(self, toy_chunks):
        assert {c.chunk_id for c in apply_filters(toy_chunks, Filters(domain="international_ip"))} \
            == {"DOC-C::a27"}
        assert {c.chunk_id for c in apply_filters(toy_chunks, Filters(temporal_status="STALE"))} \
            == {"DOC-B::r13"}
        assert {c.chunk_id for c in apply_filters(toy_chunks, Filters(jurisdiction="IN"))} \
            == {"DOC-A::s3", "DOC-A::s53", "DOC-A::front", "DOC-B::r13"}

    def test_several_values_are_admitted(self, toy_chunks):
        selected = apply_filters(toy_chunks, Filters(document_type=["rules", "treaty"]))
        assert {c.chunk_id for c in selected} == {"DOC-B::r13", "DOC-C::a27"}

    def test_citable_is_tri_state(self, toy_chunks):
        assert len(apply_filters(toy_chunks, Filters(citable=True))) == 4
        assert [c.chunk_id for c in apply_filters(toy_chunks, Filters(citable=False))] \
            == ["DOC-A::front"]
        assert len(apply_filters(toy_chunks, Filters())) == len(toy_chunks)

    def test_the_same_filter_applies_to_every_retriever(self, toy_chunks):
        bm25 = BM25Retriever(toy_chunks)
        filters = Filters(domain="international_ip")
        for retriever in (bm25, BM25IdentifierBoostRetriever(bm25)):
            for result in retriever.search("patent invention", filters, top_k=10):
                assert result.metadata["domain"] == "international_ip"

    def test_there_is_no_document_id_filter(self):
        # Filtering to the expected document would make the benchmark meaningless.
        assert not hasattr(Filters(), "document_id")
        with pytest.raises(TypeError):
            Filters(document_id="DOC-A")


# ----------------------------------------------------------------- provenance ---

class TestProvenanceSurvivesRetrieval:
    def test_a_result_carries_the_chunk_provenance(self, toy_chunks):
        result = BM25Retriever(toy_chunks).search("twenty years", top_k=1)[0]
        assert result.locator == "Test Act — Section 53"
        assert result.metadata["physical_pages"] == [1]
        assert result.metadata["source_evidence_level"] == "A"
        assert result.metadata["temporal_status"] == "CURRENT"

    def test_citable_is_copied_never_decided(self, toy_chunks):
        unlocated = next(c for c in toy_chunks if c.locator_unavailable)
        result = result_from_chunk(unlocated, score=9.9, rank=1)
        assert result.citable is False
        assert result.metadata["locator_unavailable"] is True

    def test_a_non_citable_chunk_stays_retrievable_and_stays_non_citable(self, toy_chunks):
        results = BM25Retriever(toy_chunks).search("front matter locator", top_k=5)
        found = [r for r in results if r.chunk_id == "DOC-A::front"]
        assert found, "an unlocated chunk must still be retrievable"
        assert found[0].citable is False

    def test_the_published_result_shape(self, toy_chunks):
        payload = BM25Retriever(toy_chunks).search("patent", top_k=1)[0].to_dict()
        assert set(payload) == {"chunk_id", "document_id", "score", "rank", "text",
                                "locator", "citable", "metadata"}


# ------------------------------------------------------- identifier matching ---

class TestIdentifierParsing:
    @pytest.mark.parametrize("query,kind,number,subs", [
        ("section 3(p)", "section", "3", ("P",)),
        ("Section 117A", "section", "117A", ()),
        ("rule 13(2)", "rule", "13", ("2",)),
        ("Article 27.1", "article", "27", ("1",)),
        ("regulation 11", "regulation", "11", ()),
        ("sub-section (1) of section 39", "section", "39", ("1",)),
        ("s. 55", "section", "55", ()),
    ])
    def test_it_reads_the_citation(self, query, kind, number, subs):
        ref = parse_identifiers(query)[0]
        assert (ref.kind, ref.number, ref.subdivisions) == (kind, number, subs)

    def test_prose_without_a_citation_yields_nothing(self):
        assert parse_identifiers("how long does a patent last in India") == []

    def test_document_names_are_recognised(self, toy_chunks):
        aliases = document_aliases(toy_chunks)
        assert match_documents("under the Test Act, 1970", aliases) == {"DOC-A"}
        assert match_documents("what does TRIPS require", aliases) == {"DOC-C"}
        assert match_documents("a question about nothing in particular", aliases) == set()


class TestIdentifierBoost:
    def test_an_exact_provision_match_scores_full(self, toy_chunks):
        chunk = toy_chunks[1]  # section 53
        assert identifier_boost(chunk, parse_identifiers("section 53"), set()) == 1.0

    def test_the_parent_provision_scores_lower_than_an_exact_sub_unit(self, toy_chunks):
        chunk = toy_chunks[0]  # section 3, no subsection recorded
        assert identifier_boost(chunk, parse_identifiers("section 3(p)"), set()) == 0.6

    def test_a_different_unit_kind_does_not_match_on_number_alone(self, toy_chunks):
        rule_chunk = toy_chunks[3]  # rule 13
        assert identifier_boost(rule_chunk, parse_identifiers("section 13"), set()) == 0.0

    def test_naming_the_instrument_is_a_prior_not_a_decision(self, toy_chunks):
        assert identifier_boost(toy_chunks[1], [], {"DOC-A"}) == pytest.approx(0.35)

    def test_the_boost_lifts_the_cited_provision(self, toy_chunks):
        bm25 = BM25Retriever(toy_chunks)
        plain = [r.chunk_id for r in bm25.search("section 53 of the Test Act", top_k=5)]
        boosted = BM25IdentifierBoostRetriever(bm25).search(
            "section 53 of the Test Act", top_k=5)
        assert boosted[0].chunk_id == "DOC-A::s53"
        assert boosted[0].metadata["identifier_boost"] > 0
        assert plain is not None  # plain ranking recorded for comparison, not asserted

    def test_a_query_naming_nothing_is_left_as_bm25_ranked_it(self, toy_chunks):
        bm25 = BM25Retriever(toy_chunks)
        query = "twenty years"
        plain = [(r.chunk_id, round(r.score, 6)) for r in bm25.search(query, top_k=5)]
        boosted = [(r.chunk_id, round(r.score, 6))
                   for r in BM25IdentifierBoostRetriever(bm25).search(query, top_k=5)]
        assert plain == boosted


# --------------------------------------------------------------------- fusion ---

class TestReciprocalRankFusion:
    def test_a_chunk_ranked_well_by_both_wins(self):
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "c"]])
        assert [identifier for identifier, _ in fused][:2] == ["a", "b"] or \
               [identifier for identifier, _ in fused][:2] == ["b", "a"]
        assert fused[-1][0] == "c"

    def test_scores_are_the_sum_of_reciprocals(self):
        fused = dict(reciprocal_rank_fusion([["a"], ["a"]], k=60))
        assert fused["a"] == pytest.approx(2 / 61)

    def test_it_needs_no_comparable_scores(self):
        # Only the orderings are passed in; magnitudes never enter the fusion.
        fused = reciprocal_rank_fusion([["x", "y"], ["y", "x"]])
        assert fused[0][1] == pytest.approx(fused[1][1])

    def test_weights_must_match_the_rankings(self):
        with pytest.raises(ValueError):
            reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0])

    def test_the_hybrid_preserves_provenance_and_records_contributions(self, toy_chunks):
        bm25 = BM25Retriever(toy_chunks)
        hybrid = HybridRRFRetriever([bm25, BM25IdentifierBoostRetriever(bm25)])
        results = hybrid.search("section 53 twenty years", top_k=3)
        assert results[0].locator.startswith("Test Act")
        assert results[0].citable is True
        assert "contributing_ranks" in results[0].metadata

    def test_fusion_needs_at_least_one_retriever(self):
        with pytest.raises(ValueError):
            HybridRRFRetriever([])


# ------------------------------------------------------- against the real corpus ---

@pytest.fixture(scope="module")
def chunks() -> list[Chunk]:
    loaded = load_chunks()
    if not loaded:
        pytest.skip("pipeline output not present")
    return loaded


class TestAgainstTheProducedCorpus:
    """Skipped before the Phase 2 pipeline has been run."""

    def test_every_chunk_carries_a_temporal_status(self, chunks):
        assert {c.temporal_status for c in chunks} <= {"CURRENT", "STALE", "UNKNOWN"}

    def test_retrieval_never_makes_an_unlocated_chunk_citable(self, chunks):
        retriever = BM25Retriever(chunks)
        for query in ("ayurveda aahara recipes", "traditional knowledge", "compulsory licence"):
            for result in retriever.search(query, top_k=20):
                source = next(c for c in chunks if c.chunk_id == result.chunk_id)
                assert result.citable is source.citable
                if source.locator_unavailable:
                    assert result.citable is False

    def test_the_dense_index_is_cached_rather_than_rebuilt(self, chunks):
        from ai.retrieval.dense import EMBEDDING_DIR, corpus_fingerprint

        expected = os.path.join(EMBEDDING_DIR, f"bge-m3-{corpus_fingerprint(chunks)}.npy")
        if not os.path.exists(expected):
            pytest.skip("embeddings not built")
        assert os.path.getsize(expected) > 0
