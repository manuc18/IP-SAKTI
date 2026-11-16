"""Phase 6 — candidate generation and reranking.

The tests that matter most are, again, the ones about what this layer must *not*
do: lose provenance, invent citability, show the reranker the answer key, or
report a cache hit as model inference. Ranking quality is measured by the
benchmark; nothing here asserts that a model is good.

A synthetic corpus stands in for the real one wherever the behaviour under test
is a property of the code rather than of the documents, so the suite runs
without the 3,073-chunk index or any model weights.
"""
from __future__ import annotations

import json

import pytest

from ai.reranking.base import IdentityReranker, Reranker, RerankedResult, order_by_score
from ai.reranking.features import (
    AUTHORITY_LAMBDA,
    NON_OPERATIVE_UNIT_TYPES,
    STRUCTURAL_PENALTY,
    WEIGHTS,
    AuthorityPolicyReranker,
    FeatureReranker,
    _percentile,
    authority_score,
    idf_coverage,
    locator_scores,
    operative_text,
    phrase_overlap,
)
from ai.reranking.format import (
    ALLOWED_FIELDS,
    format_candidate,
    leakage_report,
    locator_line,
)
from ai.retrieval.base import Chunk, Filters, RetrievalResult, result_from_chunk
from ai.retrieval.candidates import (
    Candidate,
    CandidateGenerator,
    CandidateSet,
    _interleave,
)
from ai.retrieval.identifiers import parse_identifiers
from ai.retrieval.router import CONCEPTUAL, IDENTIFIER
from evaluation.benchmark_reranking import (
    CANDIDATE_DEPTHS,
    candidate_metrics,
    conditional_mrr,
    percentiles,
)
from evaluation.rerank_errors import classify_failure, summarise_failures

EMPTY_LOCATOR = {
    "part": None, "chapter": None, "section": None, "subsection": None, "clause": None,
    "rule": None, "sub_rule": None, "regulation": None, "article": None,
    "paragraph": None, "subparagraph": None, "schedule": None, "form": None,
    "annexure": None, "item": None,
}


def make_chunk(chunk_id: str, text: str, **overrides) -> Chunk:
    locator = dict(EMPTY_LOCATOR)
    locator.update(overrides.pop("locator", {}))
    return Chunk(
        chunk_id=chunk_id,
        document_id=overrides.pop("document_id", "IN-PATENTS-ACT-1970"),
        text=text,
        citation=overrides.pop("citation", f"Patents Act, 1970 — {chunk_id}"),
        locator=locator,
        locator_unavailable=overrides.pop("locator_unavailable", False),
        citable=overrides.pop("citable", True),
        retrievable=True,
        domain=overrides.pop("domain", "patents"),
        jurisdiction="IN",
        document_type="act",
        temporal_status=overrides.pop("temporal_status", "CURRENT"),
        source_evidence_level=overrides.pop("source_evidence_level", "A"),
        source_status="VERIFIED_OFFICIAL",
        physical_pages=overrides.pop("physical_pages", [1]),
        title="The Patents Act, 1970",
        short_title=overrides.pop("short_title", "Patents Act, 1970"),
        heading=overrides.pop("heading", None),
        unit_type=overrides.pop("unit_type", "provision"),
        structural_path=overrides.pop("structural_path", ""),
        raw={"citability_reason": "locator established"},
    )


class StubRetriever:
    """Returns a fixed ranking, built through the real `result_from_chunk`."""

    def __init__(self, name, chunks, order, *, boosts=None):
        self.name = name
        self.chunks = {c.chunk_id: c for c in chunks}
        self.order = list(order)
        self.boosts = dict(boosts or {})

    def search(self, query, filters=None, top_k=10):
        out = []
        for rank, chunk_id in enumerate(self.order[:top_k], start=1):
            extra = ({"identifier_boost": self.boosts[chunk_id]}
                     if chunk_id in self.boosts else None)
            out.append(result_from_chunk(self.chunks[chunk_id],
                                         score=1.0 / rank, rank=rank, extra=extra))
        return out

    def config(self):
        return {"retriever": self.name}


@pytest.fixture
def corpus():
    return [
        make_chunk("c-s3", "What are not inventions. The following are not inventions "
                           "within the meaning of this Act, namely the mere discovery "
                           "of a new form of a known substance.",
                   locator={"section": "3"}, citation="Patents Act, 1970 — Section 3",
                   heading="What are not inventions"),
        make_chunk("c-s3-1", "sub-clause of section 3", locator={"section": "3",
                                                                 "clause": "D"},
                   citation="Patents Act, 1970 — Section 3(d)"),
        make_chunk("c-s64", "A patent may be revoked on the ground that the subject "
                            "matter falls within section 3 of this Act.",
                   locator={"section": "64"},
                   citation="Patents Act, 1970 — Section 64"),
        make_chunk("c-s53", "Term of patent shall be twenty years.",
                   locator={"section": "53"},
                   citation="Patents Act, 1970 — Section 53",
                   temporal_status="STALE", source_evidence_level="C"),
        make_chunk("c-unloc", "Unlocated preamble text.", citable=False,
                   locator_unavailable=True, citation="Patents Act, 1970"),
    ]


@pytest.fixture
def generator(corpus):
    dense = StubRetriever("dense", corpus, ["c-s3", "c-s64", "c-s53", "c-unloc"])
    lexical = StubRetriever("bm25_boost", corpus,
                            ["c-s64", "c-s3-1", "c-s3", "c-s53"],
                            boosts={"c-s3": 1.0, "c-s3-1": 1.0})
    return CandidateGenerator(dense=dense, lexical=lexical,
                              dense_depth=4, lexical_depth=4)


# ------------------------------------------------- candidate generation ---


def test_interleave_keeps_first_appearance_and_deduplicates():
    merged = _interleave([["a", "b", "c"], ["b", "d"]])
    assert merged == ["a", "b", "d", "c"]
    assert len(merged) == len(set(merged))


def test_candidate_pool_deduplicates_by_chunk_id(generator):
    pool = generator.generate("what is not an invention")
    ids = pool.ids()
    assert len(ids) == len(set(ids))
    assert set(ids) == {"c-s3", "c-s3-1", "c-s64", "c-s53", "c-unloc"}


def test_candidate_records_both_engines_ranks(generator):
    pool = generator.generate("what is not an invention")
    by_id = {c.chunk_id: c for c in pool.candidates}
    assert by_id["c-s3"].dense_rank == 1
    assert by_id["c-s3"].lexical_rank == 3
    assert by_id["c-s3"].sources == ("dense", "bm25_boost")
    # An engine that never returned the chunk records None, not a sentinel rank.
    assert by_id["c-s3-1"].dense_rank is None
    assert by_id["c-s3-1"].lexical_rank == 2
    assert by_id["c-unloc"].lexical_rank is None


def test_candidate_pool_depth_is_respected(corpus):
    dense = StubRetriever("dense", corpus, [c.chunk_id for c in corpus])
    lexical = StubRetriever("bm25_boost", corpus, [c.chunk_id for c in corpus][::-1])
    pool = CandidateGenerator(dense=dense, lexical=lexical,
                              dense_depth=2, lexical_depth=1).generate("q")
    assert pool.depths == {"dense": 2, "bm25_boost": 1}
    assert len(pool) == 3


def test_candidate_order_is_deterministic(generator):
    first = generator.generate("what is not an invention").ids()
    second = generator.generate("what is not an invention").ids()
    assert first == second
    assert [c.order for c in generator.generate("q").candidates] == [1, 2, 3, 4, 5]


def test_rrf_order_is_available_but_not_the_default(corpus, generator):
    assert generator.order == "interleaved"
    dense = StubRetriever("dense", corpus, ["c-s3", "c-s64"])
    lexical = StubRetriever("bm25_boost", corpus, ["c-s64", "c-s3"])
    fused = CandidateGenerator(dense=dense, lexical=lexical, order="rrf").generate("q")
    assert set(fused.ids()) == {"c-s3", "c-s64"}


def test_routed_mode_uses_one_engine_and_records_the_decision(corpus):
    dense = StubRetriever("dense", corpus, ["c-s3", "c-s64"])
    lexical = StubRetriever("bm25_boost", corpus, ["c-s3-1"])
    routed = CandidateGenerator(dense=dense, lexical=lexical, mode="routed")
    identifier = routed.generate("What does section 3(d) say?")
    assert identifier.routing.query_type == IDENTIFIER
    assert identifier.engines == ("bm25_boost",)
    conceptual = routed.generate("Is a new form of a known substance patentable?")
    assert conceptual.routing.query_type == CONCEPTUAL
    assert conceptual.engines == ("dense",)


def test_unknown_mode_and_order_are_rejected(corpus):
    dense = StubRetriever("dense", corpus, [])
    with pytest.raises(ValueError):
        CandidateGenerator(dense=dense, lexical=dense, mode="magic")
    with pytest.raises(ValueError):
        CandidateGenerator(dense=dense, lexical=dense, order="magic")


# ------------------------------------------------- provenance survives ---


def test_candidates_preserve_provenance(generator):
    pool = generator.generate("what is not an invention")
    by_id = {c.chunk_id: c for c in pool.candidates}
    stale = by_id["c-s53"]
    assert stale.metadata["temporal_status"] == "STALE"
    assert stale.metadata["source_evidence_level"] == "C"
    assert stale.metadata["physical_pages"] == [1]
    assert stale.document_id == "IN-PATENTS-ACT-1970"
    assert stale.metadata["locator_fields"]["section"] == "53"


def test_candidate_generation_never_sets_citability(generator):
    pool = generator.generate("what is not an invention")
    by_id = {c.chunk_id: c for c in pool.candidates}
    assert by_id["c-unloc"].citable is False
    assert by_id["c-unloc"].metadata["locator_unavailable"] is True
    assert by_id["c-s3"].citable is True


def test_authority_is_not_filtered_out_of_the_pool(generator):
    """Phase 6 §11: stale and Level C chunks stay in, so they can be studied."""
    pool = generator.generate("what is not an invention")
    levels = {c.metadata["source_evidence_level"] for c in pool.candidates}
    assert "C" in levels
    assert any(c.metadata["temporal_status"] == "STALE" for c in pool.candidates)


@pytest.mark.parametrize("reranker_factory", [
    lambda: IdentityReranker(),
    lambda: FeatureReranker(),
    lambda: AuthorityPolicyReranker(FeatureReranker()),
])
def test_reranking_preserves_citability_and_provenance(generator, reranker_factory):
    pool = generator.generate("what is not an invention")
    before = {c.chunk_id: (c.citable, c.metadata["temporal_status"],
                           c.metadata["source_evidence_level"],
                           c.metadata["physical_pages"], c.document_id, c.locator)
              for c in pool.candidates}
    reranked = reranker_factory().rerank("what is not an invention",
                                         pool.candidates, top_k=10)
    for result in reranked:
        assert (result.citable, result.metadata["temporal_status"],
                result.metadata["source_evidence_level"],
                result.metadata["physical_pages"], result.document_id,
                result.locator) == before[result.chunk_id]


def test_reranking_cannot_make_a_non_citable_chunk_citable(generator):
    pool = generator.generate("what is not an invention")
    reranked = FeatureReranker().rerank("q", pool.candidates, top_k=10)
    unlocated = next(r for r in reranked if r.chunk_id == "c-unloc")
    assert unlocated.citable is False
    assert unlocated.to_retrieval_result().citable is False


def test_reranked_result_round_trips_to_a_retrieval_result(generator):
    pool = generator.generate("q")
    result = IdentityReranker().rerank("q", pool.candidates, 1)[0]
    as_retrieval = result.to_retrieval_result()
    assert isinstance(as_retrieval, RetrievalResult)
    assert as_retrieval.chunk_id == result.chunk_id
    assert as_retrieval.metadata["rerank_rank"] == 1


# ------------------------------------------------------------ ordering ---


def test_identity_preserves_candidate_order(generator):
    pool = generator.generate("what is not an invention")
    reranked = IdentityReranker().rerank("q", pool.candidates, top_k=3)
    assert [r.chunk_id for r in reranked] == pool.ids()[:3]
    assert [r.rerank_rank for r in reranked] == [1, 2, 3]
    assert reranked[0].rerank_score > reranked[1].rerank_score


def test_order_by_score_breaks_ties_on_chunk_id(generator):
    pool = generator.generate("q")
    tied = [(c, 1.0, {}) for c in pool.candidates]
    ordered = order_by_score(tied, top_k=len(tied))
    assert [r.chunk_id for r in ordered] == sorted(c.chunk_id for c in pool.candidates)
    assert [r.rerank_rank for r in ordered] == list(range(1, len(tied) + 1))


def test_top_k_truncates_without_renumbering(generator):
    pool = generator.generate("q")
    reranked = FeatureReranker().rerank("q", pool.candidates, top_k=2)
    assert [r.rerank_rank for r in reranked] == [1, 2]


@pytest.mark.parametrize("reranker_factory", [
    lambda: IdentityReranker(),
    lambda: FeatureReranker(),
    lambda: AuthorityPolicyReranker(FeatureReranker()),
])
def test_empty_candidate_set_reranks_to_nothing(reranker_factory):
    assert reranker_factory().rerank("q", [], top_k=10) == []


# ----------------------------------------------------- feature behaviour ---


def test_rank_percentile_normalization():
    assert _percentile(1, 50) == 1.0
    assert _percentile(None, 50) == 0.0
    assert _percentile(50, 50) == pytest.approx(0.02)
    assert _percentile(1, 0) == 0.0


def test_min_max_is_the_selected_normalization_and_is_bounded(generator):
    """Selected over rank percentile on measured behaviour; see the module docs."""
    pool = generator.generate("q")
    reranker = FeatureReranker()
    assert reranker.normalization == "min_max"
    scored = reranker.score("q", pool.candidates)
    values = [s.features["dense_percentile"] for s in scored]
    assert min(values) >= 0.0 and max(values) <= 1.0
    assert reranker.config()["normalization"] == "min_max"


def test_rank_percentile_remains_available(generator):
    pool = generator.generate("q")
    reranker = FeatureReranker(normalization="rank_percentile")
    scored = reranker.score("q", pool.candidates)
    values = [s.features["dense_percentile"] for s in scored]
    assert min(values) >= 0.0 and max(values) <= 1.0


def test_unknown_normalization_is_rejected():
    with pytest.raises(ValueError):
        FeatureReranker(normalization="softmax")


def test_missing_engine_scores_zero_not_last_place(generator):
    pool = generator.generate("q")
    scored = {s.candidate.chunk_id: s for s in FeatureReranker(
        normalization="rank_percentile").score("q", pool.candidates)}
    assert scored["c-s3-1"].features["dense_percentile"] == 0.0
    assert scored["c-unloc"].features["lexical_percentile"] == 0.0


def test_idf_coverage_discounts_common_terms():
    idf = {"the": 0.01, "patentability": 6.0}
    covered = idf_coverage(["the", "patentability"], {"patentability"}, idf)
    uncovered = idf_coverage(["the", "patentability"], {"the"}, idf)
    assert covered > 0.99
    assert uncovered < 0.01
    # No IDF table: plain type coverage, stated rather than silently zero.
    assert idf_coverage(["a", "b"], {"a"}, None) == pytest.approx(0.5)
    assert idf_coverage([], set(), idf) == 0.0


def test_phrase_overlap_needs_two_consecutive_tokens():
    query = ["mere", "discovery", "of", "a", "new", "form"]
    assert phrase_overlap(query, ["the", "mere", "discovery", "of", "a", "new", "form"]) == 1.0
    assert phrase_overlap(query, ["discovery"]) == 0.0
    assert phrase_overlap(query, ["a", "new", "form"]) == pytest.approx(3 / 6)
    assert phrase_overlap(["one"], ["one"]) == 0.0


def test_authority_is_scored_separately_from_relevance(generator):
    pool = generator.generate("q")
    scored = {s.candidate.chunk_id: s for s in FeatureReranker().score("q", pool.candidates)}
    stale = scored["c-s53"]
    current = scored["c-s3"]
    assert stale.authority_score < current.authority_score
    # The relevance score is built only from query-to-text evidence, so the
    # authority difference must not appear in it.
    assert "authority" not in " ".join(stale.features)
    assert set(stale.features) == set(current.features)


def test_authority_score_reads_the_declared_tables():
    good = make_chunk("g", "x", source_evidence_level="A", temporal_status="CURRENT")
    poor = make_chunk("p", "x", source_evidence_level="C", temporal_status="STALE")
    to_candidate = lambda chunk: Candidate(  # noqa: E731
        chunk_id=chunk.chunk_id, document_id=chunk.document_id, text=chunk.text,
        locator=chunk.citation, citable=chunk.citable,
        metadata={"source_evidence_level": chunk.source_evidence_level,
                  "temporal_status": chunk.temporal_status},
        order=1)
    assert authority_score(to_candidate(good))[0] == pytest.approx(1.0)
    assert authority_score(to_candidate(poor))[0] == pytest.approx(0.6 * 0.3 + 0.4 * 0.4)


def test_weights_are_declared_and_sum_to_one():
    for family, table in WEIGHTS.items():
        assert sum(table.values()) == pytest.approx(1.0), family
    assert WEIGHTS[IDENTIFIER]["locator_exactness"] > WEIGHTS[CONCEPTUAL]["locator_exactness"]
    assert WEIGHTS[CONCEPTUAL]["dense_percentile"] > WEIGHTS[IDENTIFIER]["dense_percentile"]
    assert FeatureReranker().config()["trained"] is False


# ------------------------------------------------------- operative text ---


def _candidate(chunk_id, *, unit_type, dense, order):
    """One candidate, with only what the structural penalty and the engine
    features read."""
    return Candidate(
        chunk_id=chunk_id, document_id="IN-PATENTS-ACT-1970",
        text="the term of a patent is twenty years from the date of filing",
        locator="Patents Act, 1970", citable=True,
        metadata={"unit_type": unit_type, "locator_fields": {},
                  "source_evidence_level": "A", "temporal_status": "CURRENT"},
        order=order, sources=("dense",), dense_score=dense, dense_rank=order)


@pytest.fixture
def structural_pool():
    """A front-matter chunk the engines like *more* than the provision."""
    return [_candidate("c-front", unit_type="front_matter", dense=1.0, order=1),
            _candidate("c-prov", unit_type="provision", dense=0.6, order=2),
            _candidate("c-head", unit_type="heading", dense=0.4, order=3),
            _candidate("c-other", unit_type="provision", dense=0.0, order=4)]


def test_non_operative_unit_types_are_read_from_the_chunker(structural_pool):
    kinds = {c.chunk_id: operative_text(c) for c in structural_pool}
    assert kinds == {"c-front": 0.0, "c-prov": 1.0, "c-head": 0.0, "c-other": 1.0}
    assert NON_OPERATIVE_UNIT_TYPES == {"front_matter", "heading", "preamble"}


def test_front_matter_does_not_outrank_a_provision_it_merely_names(structural_pool):
    """The failure this exists for: a title page winning on document vocabulary."""
    query = "what is the term of a patent"
    without = FeatureReranker(structural_penalty=0.0).rerank(query, structural_pool)
    assert without[0].chunk_id == "c-front"

    with_penalty = FeatureReranker().rerank(query, structural_pool)
    assert with_penalty[0].chunk_id == "c-prov"


def test_the_penalty_is_a_demotion_and_never_a_filter(structural_pool):
    """Phase 3 decided against hiding text from search; this must not undo it."""
    results = FeatureReranker().rerank("what is the term of a patent", structural_pool)
    assert {r.chunk_id for r in results} == {c.chunk_id for c in structural_pool}


def test_ordering_among_operative_candidates_is_untouched(structural_pool):
    query = "what is the term of a patent"
    operative = lambda results: [r.chunk_id for r in results  # noqa: E731
                                 if r.chunk_id in ("c-prov", "c-other")]
    assert operative(FeatureReranker(structural_penalty=0.0).rerank(query, structural_pool)) \
        == operative(FeatureReranker().rerank(query, structural_pool))


def test_relevance_score_keeps_its_meaning_and_the_penalty_is_reported(structural_pool):
    scored = {s.candidate.chunk_id: s
              for s in FeatureReranker().score("q", structural_pool)}
    front, provision = scored["c-front"], scored["c-prov"]
    # The weighted sum is unchanged and still bounded; only the ranking score moves.
    assert 0.0 <= front.relevance_score <= 1.0
    assert front.structural_penalty == pytest.approx(STRUCTURAL_PENALTY)
    assert front.ranking_score == pytest.approx(front.relevance_score - STRUCTURAL_PENALTY)
    assert provision.structural_penalty == 0.0
    assert provision.ranking_score == pytest.approx(provision.relevance_score)


def test_unstructured_text_is_not_demoted():
    """`unstructured` means the chunker found no locator, not that the text is
    non-operative — a different claim, and one this module does not make."""
    assert operative_text(_candidate("c-u", unit_type="unstructured",
                                     dense=1.0, order=1)) == 1.0


def test_the_penalty_is_bounded_and_declared(structural_pool):
    config = FeatureReranker().config()["structural_penalty"]
    assert config["value"] == STRUCTURAL_PENALTY < 0.5
    assert config["filter"] is False
    assert sorted(NON_OPERATIVE_UNIT_TYPES) == config["applies_to"]
    with pytest.raises(ValueError):
        FeatureReranker(structural_penalty=-0.1)


# --------------------------------------------------- identifier handling ---


@pytest.mark.parametrize("locator,expected_exact,label", [
    ({"section": "3", "clause": "D"}, 1.0, "exact locator"),
    ({"section": "3"}, 0.6, "parent locator"),
    ({"section": "3", "clause": "P"}, 0.3, "sibling sub-unit"),
    ({"section": "64"}, 0.0, "unrelated locator"),
])
def test_locator_exactness_grades_the_four_cases(locator, expected_exact, label):
    refs = parse_identifiers("What does section 3(d) of the Patents Act say?")
    exact, _ = locator_scores(refs, locator, "some text")
    assert exact == pytest.approx(expected_exact), label


def test_referenced_locator_is_separated_from_being_the_provision():
    refs = parse_identifiers("What does section 3 exclude?")
    # The provision itself: exact, and not "referenced".
    exact, referenced = locator_scores(refs, {"section": "3"}, "The following are not…")
    assert (exact, referenced) == (1.0, 0.0)
    # A different provision whose text merely cites section 3.
    exact, referenced = locator_scores(
        refs, {"section": "64"}, "revoked on the ground that it falls within section 3")
    assert (exact, referenced) == (0.0, 1.0)
    # A different provision that does not cite it at all.
    exact, referenced = locator_scores(refs, {"section": "64"}, "unrelated text")
    assert (exact, referenced) == (0.0, 0.0)


def test_referenced_locator_carries_no_weight(generator):
    """It is a Phase 6 §10 diagnostic, not a ranking signal — yet."""
    for table in WEIGHTS.values():
        assert table["locator_referenced"] == 0.0


def test_identifier_query_puts_the_named_provision_first(generator):
    query = "What does section 3(d) of the Patents Act, 1970 provide?"
    pool = generator.generate(query)
    reranked = FeatureReranker(aliases={"IN-PATENTS-ACT-1970": ("patents act, 1970",)}
                               ).rerank(query, pool.candidates, top_k=5)
    assert reranked[0].chunk_id == "c-s3-1"


# ------------------------------------------------------ authority policy ---


def test_authority_policy_is_bounded_by_lambda(generator):
    pool = generator.generate("q")
    base = FeatureReranker()
    policy = AuthorityPolicyReranker(base, weight=AUTHORITY_LAMBDA)
    scored = policy.rerank("q", pool.candidates, top_k=len(pool.candidates))
    for result in scored:
        gap = (result.components["base_score_normalised"]
               - result.components["base_score_normalised"])
        assert gap == 0.0
    # A candidate can only overtake one whose normalised base score is within
    # lambda of it, which is the guarantee the class advertises.
    ordered = [(r.components["base_score_normalised"], r.rerank_rank) for r in scored]
    for (score_a, rank_a) in ordered:
        for (score_b, rank_b) in ordered:
            if rank_a < rank_b and score_a < score_b:
                assert score_b - score_a <= AUTHORITY_LAMBDA + 1e-9


def test_authority_policy_does_not_use_citability(generator):
    pool = generator.generate("q")
    config = AuthorityPolicyReranker(FeatureReranker()).config()
    assert config["citability_in_policy"] is False
    assert "citable" not in json.dumps(config["authority"])


def test_authority_policy_scores_the_whole_pool_before_truncating(generator):
    pool = generator.generate("q")
    policy = AuthorityPolicyReranker(FeatureReranker(), weight=10.0)
    top = policy.rerank("q", pool.candidates, top_k=1)
    assert len(top) == 1
    # With a large lambda the policy is free to promote from anywhere in the
    # pool; that it can is the proof the base saw every candidate.
    assert top[0].components["authority_score"] > 0.0


# ----------------------------------------------------- format and leakage ---


def test_candidate_format_is_the_documented_shape(generator):
    pool = generator.generate("q")
    candidate = next(c for c in pool.candidates if c.chunk_id == "c-s3")
    rendered = format_candidate(candidate)
    lines = rendered.split("\n")
    assert lines[0] == "Document: Patents Act, 1970"
    assert lines[1] == "Locator: Section 3"
    assert lines[2] == "Heading: What are not inventions"
    assert lines[3].startswith("Text: ")
    labels = [line.split(":", 1)[0].lower() for line in lines[:4]]
    assert labels == list(ALLOWED_FIELDS)


def test_candidate_format_omits_an_absent_heading(generator):
    pool = generator.generate("q")
    candidate = next(c for c in pool.candidates if c.chunk_id == "c-s64")
    assert "Heading:" not in format_candidate(candidate)


def test_locator_line_drops_the_repeated_document_name(generator):
    pool = generator.generate("q")
    candidate = next(c for c in pool.candidates if c.chunk_id == "c-s3")
    assert locator_line(candidate) == "Section 3"
    unlocated = next(c for c in pool.candidates if c.chunk_id == "c-unloc")
    assert locator_line(unlocated) == "(no locator established)"


def test_format_is_deterministic(generator):
    pool = generator.generate("q")
    candidate = pool.candidates[0]
    assert format_candidate(candidate) == format_candidate(candidate)


def test_rendered_candidates_never_contain_the_chunk_id(generator):
    pool = generator.generate("q")
    for candidate in pool.candidates:
        assert candidate.chunk_id not in format_candidate(candidate)


def test_rendered_candidates_never_contain_authority_metadata(generator):
    pool = generator.generate("q")
    for candidate in pool.candidates:
        rendered = format_candidate(candidate)
        assert "citable" not in rendered.lower()
        assert "STALE" not in rendered
        assert "source_evidence_level" not in rendered


def test_leakage_report_is_clean_on_the_real_format(generator):
    pool = generator.generate("Is a new form of a known substance patentable?")
    report = leakage_report(
        "Is a new form of a known substance patentable?", pool.candidates,
        forbidden_strings={"expected_chunks": ["c-s3"],
                           "expected_documents": ["IN-PATENTS-ACT-1970"],
                           "notes": ["s.3(d), the efficacy threshold."]})
    assert report["clean"], report["findings"]
    assert report["candidates_audited"] == len(pool)


def test_leakage_report_catches_an_injected_chunk_id(generator):
    pool = generator.generate("q")
    rendered = [format_candidate(c) for c in pool.candidates]
    rendered[0] += f"\nDebug: {pool.candidates[0].chunk_id}"
    report = leakage_report("q", pool.candidates, rendered=rendered)
    assert not report["clean"]
    checks = {finding["check"] for finding in report["findings"]}
    assert "chunk_id_in_rendered_candidate" in checks


def test_leakage_report_catches_a_stray_header_field(generator):
    """A new labelled field must not appear without being added to the contract."""
    pool = generator.generate("q")
    rendered = [format_candidate(c) for c in pool.candidates]
    rendered[0] = rendered[0].replace("Text: ", "Evidence level: A\nText: ", 1)
    report = leakage_report("q", pool.candidates, rendered=rendered)
    assert not report["clean"]
    assert any(f["check"] == "unknown_label_in_rendered_candidate"
               for f in report["findings"])


def test_leakage_report_catches_the_answer_key_in_the_query(generator):
    pool = generator.generate("q")
    report = leakage_report(
        "q c-s3", pool.candidates,
        forbidden_strings={"expected_chunks": ["c-s3"]})
    assert not report["clean"]
    assert any(f["check"] == "forbidden_string_in_query" for f in report["findings"])


def test_rerankers_receive_only_the_query_and_the_candidates():
    """Structural leakage prevention: the interface admits nothing else."""
    import inspect

    for cls in (IdentityReranker, FeatureReranker, AuthorityPolicyReranker):
        parameters = list(inspect.signature(cls.rerank).parameters)
        assert parameters == ["self", "query", "candidates", "top_k"], cls.__name__


def test_candidate_carries_no_golden_fields(generator):
    pool = generator.generate("q")
    serialised = json.dumps([c.to_dict() for c in pool.candidates])
    for forbidden in ("expected_chunks", "expected_locators", "anchor_terms",
                      "verified", "query_family", "ground_truth_method"):
        assert forbidden not in serialised


# ------------------------------------------------------------- metrics ---


def _row(**overrides):
    row = {
        "question_id": "Q001",
        "query": "What does section 3(p) of the Patents Act, 1970 exclude?",
        "expected_chunks": ["gold"],
        "expected_documents": ["IN-PATENTS-ACT-1970"],
        "expected_locators": ["Section 3"],
        "query_family": "identifier",
        "router_family": "IDENTIFIER",
        "gold_in_pool": True,
        "gold_pool_rank": 4,
        "rank_of_expected": 9,
        "candidates": {"pool_size": 60, "engines": ["dense", "bm25_boost"]},
        "results": [
            {"rerank_rank": 1, "chunk_id": "other", "locator": "Patents Act, 1970 — Section 64",
             "citable": True, "document_id": "IN-PATENTS-ACT-1970"},
        ],
    }
    row.update(overrides)
    return row


def test_candidate_recall_reports_every_requested_depth():
    pool = CandidateSet(query="q", mode="expanded", order="interleaved",
                        candidates=tuple(), depths={}, engines=())
    empty = candidate_metrics(pool, ["gold"])
    assert set(f"candidate_recall@{d}" for d in CANDIDATE_DEPTHS) <= set(empty)
    assert empty["gold_in_pool"] is False
    assert empty["gold_pool_rank"] is None
    assert all(empty[f"candidate_recall@{d}"] == 0.0 for d in CANDIDATE_DEPTHS)


def test_candidate_recall_is_depth_sensitive(corpus):
    dense = StubRetriever("dense", corpus, [c.chunk_id for c in corpus])
    pool = CandidateGenerator(dense=dense, lexical=StubRetriever("bm25_boost", corpus, []),
                              dense_depth=5, lexical_depth=0).generate("q")
    metrics = candidate_metrics(pool, ["c-s53"])  # 4th in the pool
    assert metrics["gold_pool_rank"] == 4
    assert metrics["candidate_recall@10"] == 1.0
    assert metrics["candidate_recall@pool"] == 1.0


def test_conditional_mrr_uses_only_queries_whose_gold_reached_the_pool():
    scores = [
        {"gold_in_pool": True, "mrr": 1.0, "recall@1": 1.0, "recall@5": 1.0},
        {"gold_in_pool": True, "mrr": 0.5, "recall@1": 0.0, "recall@5": 1.0},
        {"gold_in_pool": False, "mrr": 0.0, "recall@1": 0.0, "recall@5": 0.0},
    ]
    conditional = conditional_mrr(scores)
    assert conditional["questions"] == 2
    assert conditional["mrr"] == pytest.approx(0.75)
    assert conditional_mrr([])["questions"] == 0


def test_percentiles_reports_median_mean_and_p95():
    values = list(range(1, 101))  # 1..100
    stats = percentiles(values)
    assert stats["median_ms"] == pytest.approx(50.5)
    assert stats["mean_ms"] == pytest.approx(50.5)
    assert stats["p95_ms"] == pytest.approx(95.0)  # nearest-rank
    assert stats["n"] == 100
    assert percentiles([])["n"] == 0


def test_latency_is_measured_per_stage_not_only_in_total(generator):
    pool = generator.generate("q")
    assert pool.latency_ms >= 0.0
    assert set(pool.per_engine_latency_ms) == {"dense", "bm25_boost"}
    assert pool.summary()["latency_ms"] == round(pool.latency_ms, 2)


# ------------------------------------------------------ error analysis ---


def test_failure_type_a_is_a_candidate_generation_failure():
    failure = classify_failure(_row(gold_in_pool=False, gold_pool_rank=None,
                                    rank_of_expected=None),
                               scores_by_id={}, at=5)
    assert failure["type"] == "A"
    assert "never shown" in failure["detail"]


def test_failure_type_b_is_a_reranking_failure():
    failure = classify_failure(
        _row(results=[{"rerank_rank": 1, "chunk_id": "unrelated",
                       "locator": "Trade Marks Act, 1999 — Section 9",
                       "citable": True, "document_id": "IN-TM-ACT-1999"}]),
        scores_by_id={}, at=5)
    assert failure["type"] == "B"
    assert failure["gold_pool_rank"] == 4


def test_failure_type_c_names_the_competing_chunks():
    failure = classify_failure(
        _row(results=[{"rerank_rank": 1, "chunk_id": "sibling",
                       "locator": "Patents Act, 1970 — Section 3(d)",
                       "citable": True, "document_id": "IN-PATENTS-ACT-1970"}]),
        scores_by_id={}, at=5)
    assert failure["type"] == "C"
    assert failure["competing_chunks"][0]["chunk_id"] == "sibling"


def test_a_query_that_succeeded_is_not_a_failure():
    assert classify_failure(_row(rank_of_expected=1), scores_by_id={}, at=5) is None


def test_type_d_flags_a_family_disagreement():
    failure = classify_failure(
        _row(query="How does the Geographical Indications Act, 1999 define a "
                   "geographical indication?",
             query_family="identifier", router_family="CONCEPTUAL",
             expected_locators=["Section 2"]),
        scores_by_id={}, at=5)
    flags = {flag["flag"] for flag in failure["label_review_flags"]}
    assert "query_family_disagreement" in flags


def test_type_d_flags_a_label_that_ignores_the_cited_provision():
    failure = classify_failure(
        _row(query="What does section 3(p) say?", expected_locators=["Section 64"]),
        scores_by_id={}, at=5)
    flags = {flag["flag"] for flag in failure["label_review_flags"]}
    assert "query_cites_a_provision_the_label_does_not" in flags


def test_type_d_is_an_overlay_not_a_partition():
    failures = [
        {"type": "A", "question_id": "Q1", "label_review_flags": [{"flag": "x"}]},
        {"type": "B", "question_id": "Q2", "label_review_flags": []},
    ]
    summary = summarise_failures(failures, questions=10)
    assert summary["types"] == {"A": 1, "B": 1, "C": 0}
    assert sum(summary["types"].values()) == summary["failures"]
    assert summary["type_D_flagged"] == 1
    assert summary["candidate_generation_failures"] == 1
    assert summary["reranking_failures"] == 1


# -------------------------------------------------------------- config ---


def test_every_reranker_declares_a_reproducible_config(generator):
    for reranker in (IdentityReranker(), FeatureReranker(),
                     AuthorityPolicyReranker(FeatureReranker())):
        config = reranker.config()
        assert config["reranker"] == reranker.name
        assert "version" in config
        json.dumps(config)  # must be serialisable into a results file


def test_candidate_generator_config_records_both_engines(generator):
    config = generator.config()
    assert config["mode"] == "expanded"
    assert config["dense_depth"] == 4
    assert config["deduplication"] == "by chunk_id, first appearance wins"
    assert set(config["components"]) == {"dense", "lexical"}
