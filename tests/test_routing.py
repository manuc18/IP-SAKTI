"""Phase 5 — deterministic query routing.

Two things are being asserted here, and they are different in kind.

The classification tests pin the *policy*: which family a query lands in, and
why. They use the six queries the Phase 5 specification gives as worked
examples, because a router that disagrees with its own specification is wrong
however good its numbers look.

The retriever tests pin the *plumbing*: that routing chooses an engine and
changes nothing else. A routed result must carry the same locator, citability
and provenance the chosen retriever produced, and it must say out loud which
path it came by — including when it fell back, which is the case a silent router
would hide.
"""
from __future__ import annotations

import pytest

from ai.retrieval.base import Filters, RetrievalResult
from ai.retrieval.router import (
    CONCEPTUAL,
    GENERAL,
    IDENTIFIER,
    PATHS,
    RoutedRetriever,
    classify,
    matched_regimes,
)


# --------------------------------------------------------------- classifier ---

class TestIdentifierDetection:
    @pytest.mark.parametrize("query", [
        "What does Section 3(p) of the Patents Act say?",
        "What is Article 27.1 of TRIPS?",
        "Explain Rule 13(2) of the Patents Rules.",
        "Does regulation 4 apply to exporters?",
        "sub-section (1) of section 39 of the Patents Act",
    ])
    def test_a_named_provision_routes_to_identifier(self, query):
        decision = classify(query)
        assert decision.query_type == IDENTIFIER
        assert decision.matched_identifiers

    def test_the_decision_names_the_evidence(self):
        decision = classify("What does Section 3(p) of the Patents Act say?")
        assert "3(P)" in decision.matched_identifiers[0].upper()
        assert "3(P)" in decision.reason.upper()

    def test_confidence_is_a_word_not_an_invented_number(self):
        # Phase 5 §5: there is no probability model behind a regex, so there is
        # no number to report.
        decision = classify("What is Article 27.1 of TRIPS?")
        assert decision.confidence == "deterministic"
        assert not isinstance(decision.confidence, float)

    def test_a_bare_year_is_not_a_provision(self):
        assert classify("What changed in the 2005 amendment?").query_type != IDENTIFIER


class TestConceptualRouting:
    @pytest.mark.parametrize("query", [
        "Can a new form of a known substance be patented?",
        "What rights arise from access to biological resources?",
        "When does fair dealing apply to a literary work?",
        "What is passing off?",
    ])
    def test_a_named_regime_without_a_provision_routes_to_conceptual(self, query):
        decision = classify(query)
        assert decision.query_type == CONCEPTUAL
        assert decision.matched_regimes

    def test_the_regime_lexicon_holds_bodies_of_law_not_subject_matter(self):
        # "Ayurvedic" names something several regimes regulate; on its own it
        # settles nothing, so it must not appear as a regime term.
        assert matched_regimes("an ayurvedic formulation") == []
        assert "patents" in matched_regimes("is this patentable")


class TestGeneralRouting:
    @pytest.mark.parametrize("query", [
        "How do I protect an Ayurvedic formulation?",
        "How to safeguard a family recipe?",
        "What is the process for protecting a formulation?",
        "Where do I go with this?",
    ])
    def test_a_procedural_question_routes_to_general(self, query):
        assert classify(query).query_type == GENERAL

    def test_a_query_naming_nothing_at_all_routes_to_general(self):
        decision = classify("What are my options here?")
        assert decision.query_type == GENERAL
        assert decision.matched_regimes == ()

    def test_a_procedural_opener_outranks_a_regime_term(self):
        # "How do I register a trade mark?" names a regime but still asks for a
        # procedure; the opener is checked first and that ordering is the policy.
        assert classify("How do I register a trade mark?").query_type == GENERAL

    def test_a_procedural_opener_never_outranks_a_named_provision(self):
        assert classify("How do I comply with Section 6 of the BDA?"
                        ).query_type == IDENTIFIER


class TestRoutingIsDeterministic:
    def test_the_same_query_always_routes_the_same_way(self):
        query = "What rights arise from access to biological resources?"
        first = classify(query)
        assert all(classify(query) == first for _ in range(5))

    def test_every_query_gets_a_reason(self):
        for query in ("Section 3", "is this patentable", "how do i start", ""):
            assert classify(query).reason


# ---------------------------------------------------------------- plumbing ---

class Stub:
    """A retriever that returns whatever it was constructed with."""

    def __init__(self, name, results):
        self.name = name
        self._results = results
        self.calls = []

    def search(self, query, filters=None, top_k=10):
        self.calls.append((query, filters, top_k))
        return list(self._results[:top_k])

    def config(self):
        return {"retriever": self.name}


def result(chunk_id, *, rank=1, boost=None, citable=True, locator="Act — Section 1"):
    metadata = {"physical_pages": [1], "temporal_status": "STALE",
                "source_evidence_level": "A", "locator_unavailable": False,
                "domain": "patents", "jurisdiction": "IN"}
    if boost is not None:
        metadata["identifier_boost"] = boost
    return RetrievalResult(chunk_id=chunk_id, document_id="DOC-A", score=1.0, rank=rank,
                           text="text", locator=locator, citable=citable,
                           metadata=metadata)


def build(identifier_results, dense_results, lexical_results=()):
    return RoutedRetriever(
        identifier=Stub("bm25_boost", identifier_results),
        dense=Stub("dense", dense_results),
        lexical=Stub("bm25", lexical_results),
    )


class TestRoutingPolicy:
    def test_an_identifier_query_goes_to_the_boosted_lexical_retriever(self):
        routed = build([result("hit", boost=1.0)], [result("other")])
        results = routed.search("What does Section 3(p) say?")
        assert results[0].chunk_id == "hit"
        assert results[0].metadata["routed_to"] == "bm25_boost"
        assert results[0].metadata["retrieval_path"] == "identifier"

    def test_a_conceptual_query_goes_to_the_dense_retriever(self):
        routed = build([result("lex", boost=1.0)], [result("dense-hit")])
        results = routed.search("Is a known substance patentable?")
        assert results[0].chunk_id == "dense-hit"
        assert results[0].metadata["retrieval_path"] == "conceptual_dense"

    def test_a_general_query_goes_to_the_dense_retriever(self):
        routed = build([result("lex", boost=1.0)], [result("dense-hit")])
        results = routed.search("How do I protect an Ayurvedic formulation?")
        assert results[0].metadata["retrieval_path"] == "general_dense"
        assert results[0].metadata["query_type"] == GENERAL

    def test_filters_and_top_k_reach_the_chosen_retriever_unchanged(self):
        routed = build([result("lex", boost=1.0)], [result("d")])
        filters = Filters(domain="patents")
        routed.search("Is this patentable?", filters, 7)
        assert routed.dense.calls == [("Is this patentable?", filters, 7)]
        assert routed.identifier.calls == []


class TestFallback:
    def test_an_identifier_query_falls_back_when_the_locator_matched_nothing(self):
        # The boost fired on no chunk: the corpus has nothing under the locator
        # the query named, so the lexical list is BM25 over a dead token.
        routed = build([result("lex", boost=0.0)], [result("dense-hit")])
        results = routed.search("What does Section 999 of the Patents Act say?")
        assert results[0].chunk_id == "dense-hit"
        assert results[0].metadata["retrieval_path"] == "identifier_fallback_dense"
        assert results[0].metadata["routed_to"] == "dense"

    def test_an_identifier_query_falls_back_when_nothing_came_back_at_all(self):
        routed = build([], [result("dense-hit")])
        results = routed.search("What does Section 3(p) say?")
        assert results[0].metadata["retrieval_path"] == "identifier_fallback_dense"

    def test_a_dense_miss_falls_back_to_bm25(self):
        routed = build([result("lex", boost=1.0)], [], [result("bm25-hit")])
        results = routed.search("Is a known substance patentable?")
        assert results[0].chunk_id == "bm25-hit"
        assert results[0].metadata["retrieval_path"] == "conceptual_fallback_lexical"

    def test_a_general_dense_miss_falls_back_to_bm25(self):
        routed = build([], [], [result("bm25-hit")])
        results = routed.search("How do I protect a formulation?")
        assert results[0].metadata["retrieval_path"] == "general_fallback_lexical"

    def test_the_identifier_route_is_kept_when_both_engines_are_empty(self):
        routed = build([], [], [])
        assert routed.search("What does Section 3(p) say?") == []

    def test_every_path_the_router_can_take_is_declared(self):
        routed = build([result("lex", boost=1.0)], [result("d")], [result("b")])
        taken = set()
        for query in ("What does Section 3(p) say?", "Is this patentable?",
                      "How do I file this?"):
            taken.add(routed.search(query)[0].metadata["retrieval_path"])
        assert taken <= set(PATHS)


class TestRoutingPreservesProvenance:
    """Phase 5 §10. Routing chooses an engine; it never edits a chunk."""

    def test_the_result_is_unchanged_apart_from_the_route(self):
        original = result("c1", citable=False, locator="CBD — unlocated text")
        routed = build([], [original])
        got = routed.search("Is this patentable?")[0]
        assert got.chunk_id == original.chunk_id
        assert got.locator == original.locator
        assert got.citable is False
        assert got.score == original.score
        for key, value in original.metadata.items():
            assert got.metadata[key] == value

    def test_the_router_cannot_make_a_chunk_citable(self):
        routed = build([], [result("c1", citable=False)])
        assert routed.search("Is this patentable?")[0].citable is False

    def test_temporal_status_and_evidence_level_survive_routing(self):
        routed = build([], [result("c1")])
        metadata = routed.search("Is this patentable?")[0].metadata
        assert metadata["temporal_status"] == "STALE"
        assert metadata["source_evidence_level"] == "A"
        assert metadata["locator_unavailable"] is False

    def test_the_config_declares_the_policy_and_the_fallbacks(self):
        config = build([], []).config()
        assert config["policy"][IDENTIFIER] == "bm25_boost"
        assert config["policy"][CONCEPTUAL] == "dense"
        assert config["policy"][GENERAL] == "dense"
        assert config["fallbacks"]["identifier_no_locator_match"] == "dense"
        assert "components" in config
