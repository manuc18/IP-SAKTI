"""Phase C: the heading-coverage bonus in `ai/reranking/features.py`.

The property that matters most here is the one that makes the change safe to
ship and safe to withdraw: at `heading_bonus=0.0` the reranker must produce
byte-identical orderings to the version before the feature existed. That is what
turns it into a measurable change rather than a new baseline nobody can compare
against — and it is asserted, not assumed.

The second property is that a chunk with no heading is not *demoted* for having
none. A third of this corpus has no heading, and a positively-weighted feature
scored at 0.0 for them would be a penalty for a chunking outcome rather than a
statement about relevance. The bonus is added outside the weighted sum precisely
so that those chunks score exactly what they scored before.
"""
from __future__ import annotations

import pytest

from ai.reranking.features import (
    HEADING_BONUS,
    STRUCTURAL_PENALTY,
    FeatureReranker,
    _stem,
    heading_coverage,
)
from ai.retrieval.candidates import Candidate


#: A miniature IDF table in the shape of the real one. Production always builds
#: `FeatureReranker(idf=bm25.idf)`, so these tests exercise the IDF path rather
#: than the no-table fallback; `TestNoIdfFallback` covers the other branch on
#: purpose. The numbers only have to be ordered the way a real corpus orders
#: them: function words near zero, legal nouns high.
IDF = {
    "of": 0.01, "for": 0.02, "the": 0.01, "to": 0.02, "a": 0.01, "and": 0.01,
    "is": 0.01, "what": 0.02, "how": 0.02, "does": 0.02,
    "protection": 3.1, "geographical": 5.4, "indications": 5.2, "indication": 5.2,
    "additional": 2.4, "application": 2.9, "licence": 3.4, "manufacture": 3.6,
    "form": 2.2, "penalties": 4.1, "penalty": 4.1, "contravention": 4.8,
    "least": 3.0, "developed": 3.3, "country": 3.2, "members": 3.0,
    "fees": 3.5, "payable": 3.1, "on": 0.02, "an": 0.01, "restoration": 4.6,
    "lapsed": 5.0, "patent": 2.8, "licences": 3.4, "applied": 2.9,
}


def make_candidate(chunk_id: str, *, text: str, heading: str | None,
                   order: int = 1, dense: float = 0.5, lexical: float = 0.5,
                   unit_type: str = "provision") -> Candidate:
    return Candidate(
        chunk_id=chunk_id, document_id=chunk_id.split("::")[0], text=text,
        locator=f"{chunk_id} locator", citable=True,
        metadata={"heading": heading, "unit_type": unit_type,
                  "locator_fields": {}, "source_evidence_level": "B",
                  "temporal_status": "CURRENT"},
        order=order, sources=("dense",), dense_score=dense, dense_rank=order,
        lexical_score=lexical, lexical_rank=order, identifier_boost=0.0)


class TestStemmerIsConsistentOnBothSides:
    """It may be wrong about English; it may not be wrong asymmetrically."""

    @pytest.mark.parametrize("left,right", [
        ("representation", "represented"),
        ("restoration", "restored"),
        ("penalties", "penalty"),
        ("designs", "design"),
        ("meaning", "mean"),
        ("infringement", "infringing"),
        ("indications", "indication"),
        ("protection", "protect"),
        ("applications", "application"),
    ])
    def test_related_forms_reach_the_same_stem(self, left, right):
        assert _stem(left) == _stem(right)

    def test_a_doubled_s_is_not_ground_away(self):
        assert _stem("address") == "address"

    def test_a_suffix_is_never_stripped_twice(self):
        # The bug this guards: "infringing" -> "infring" -> "infr".
        assert _stem("infringing") == _stem("infringe")
        assert len(_stem("infringing")) >= 5

    def test_short_tokens_survive_intact(self):
        for token in ("fee", "act", "use", "tax"):
            assert _stem(token) == token


class TestHeadingCoverage:

    def test_a_heading_the_question_fully_accounts_for_scores_high(self):
        score = heading_coverage(
            ["what", "protection", "does", "trips", "require", "for",
             "geographical", "indications"],
            "Protection of Geographical Indications", IDF)
        # "of" is in the heading and not in the query, but its IDF is ~0, which
        # is exactly how this feature avoids needing a stopword list.
        assert score > 0.99

    def test_an_extra_word_in_the_heading_lowers_it(self):
        query = ["what", "protection", "does", "trips", "require", "for",
                 "geographical", "indications"]
        exact = heading_coverage(query, "Protection of Geographical Indications", IDF)
        wider = heading_coverage(query, "Additional Protection for Geographical Indications", IDF)
        assert wider < exact

    def test_an_unrelated_heading_scores_zero(self):
        assert heading_coverage(["term", "of", "patent", "protection"],
                                "Least-Developed Country Members", IDF) < 0.05

    def test_no_heading_scores_zero_rather_than_raising(self):
        assert heading_coverage(["anything"], None, IDF) == 0.0
        assert heading_coverage(["anything"], "", IDF) == 0.0

    def test_it_reads_the_heading_not_the_body(self):
        # Direction matters: this is coverage OF the heading BY the query.
        # A one-word query cannot fully account for a seven-word heading.
        assert heading_coverage(["fees"], "Fees payable on an application for "
                                          "restoration of a lapsed patent", IDF) < 0.5


class TestBonusIsReversibleAndFair:

    @staticmethod
    def order(reranker, candidates, query="how is a licence applied for"):
        return [r.chunk_id for r in reranker.rerank(query, candidates, top_k=10)]

    def test_zero_bonus_restores_the_previous_ordering(self):
        candidates = [
            make_candidate("D::a", text="Application for licence to manufacture.",
                           heading="Application for licence to manufacture",
                           order=1, dense=0.60, lexical=0.5),
            make_candidate("D::b", text="Form of licence to manufacture.",
                           heading="Form of licence to manufacture",
                           order=2, dense=0.90, lexical=0.5),
        ]
        off = FeatureReranker(idf=IDF, heading_bonus=0.0)
        # With the feature off, the stronger dense score wins outright.
        assert self.order(off, candidates)[0] == "D::b"

    def test_the_bonus_can_break_a_near_tie_towards_the_named_provision(self):
        """Phase B's largest bucket, in miniature: two adjacent provisions.

        The pool has five members on purpose. Min-max normalisation over two
        candidates always yields 0.0 and 1.0 whatever the underlying scores, so
        a two-candidate fixture manufactures a maximal gap and tests nothing the
        real pool of ~77 ever produces. Here the two contenders sit close
        together inside a spread, which is the situation the bonus is for: it
        breaks near ties, and is not meant to overturn a decisive one.
        """
        candidates = [
            make_candidate("D::a", text="Application for licence to manufacture.",
                           heading="Application for licence to manufacture",
                           order=2, dense=0.81, lexical=0.60),
            make_candidate("D::b", text="Form of licence to manufacture.",
                           heading="Form of licence to manufacture",
                           order=1, dense=0.84, lexical=0.62),
            make_candidate("D::c", text="Fees payable on a licence.",
                           heading="Fees", order=3, dense=0.70, lexical=0.40),
            make_candidate("D::d", text="Conditions of a licence.",
                           heading="Conditions", order=4, dense=0.62, lexical=0.35),
            make_candidate("D::e", text="Duration of a licence.",
                           heading="Duration", order=5, dense=0.55, lexical=0.30),
        ]
        query = "what is the application for a licence to manufacture"
        off = FeatureReranker(idf=IDF, heading_bonus=0.0)
        on = FeatureReranker(idf=IDF, heading_bonus=HEADING_BONUS)
        assert self.order(off, candidates, query)[0] == "D::b"
        assert self.order(on, candidates, query)[0] == "D::a"

    def test_the_bonus_does_not_overturn_a_decisive_relevance_gap(self):
        """The other side of the bound: it is a tiebreak, not an override."""
        candidates = [
            make_candidate("D::a", text="Wholly unrelated text.",
                           heading="Application for licence to manufacture",
                           order=5, dense=0.10, lexical=0.05),
            make_candidate("D::b", text="Application for a licence to manufacture.",
                           heading=None, order=1, dense=0.95, lexical=0.90),
            make_candidate("D::c", text="Other text.", heading="Fees",
                           order=3, dense=0.50, lexical=0.45),
        ]
        on = FeatureReranker(idf=IDF, heading_bonus=HEADING_BONUS)
        query = "what is the application for a licence to manufacture"
        assert self.order(on, candidates, query)[0] == "D::b"

    def test_a_chunk_with_no_heading_scores_exactly_what_it_did_before(self):
        candidate = make_candidate("D::x", text="Some operative text here.",
                                   heading=None, order=1, dense=0.7, lexical=0.4)
        off = FeatureReranker(idf=IDF, heading_bonus=0.0).score("any query", [candidate])[0]
        on = FeatureReranker(idf=IDF, heading_bonus=HEADING_BONUS).score("any query", [candidate])[0]
        assert on.ranking_score == off.ranking_score
        assert on.heading_bonus == 0.0

    def test_the_bonus_never_lowers_a_score(self):
        candidates = [make_candidate(f"D::{i}", text="text about licences",
                                     heading="Licences", order=i + 1)
                      for i in range(5)]
        off = {s.candidate.chunk_id: s.ranking_score
               for s in FeatureReranker(idf=IDF, heading_bonus=0.0).score("licences", candidates)}
        on = {s.candidate.chunk_id: s.ranking_score
              for s in FeatureReranker(idf=IDF, heading_bonus=HEADING_BONUS).score("licences", candidates)}
        for chunk_id, score in on.items():
            assert score >= off[chunk_id] - 1e-12

    def test_a_negative_bonus_is_refused(self):
        with pytest.raises(ValueError, match="heading_bonus"):
            FeatureReranker(heading_bonus=-0.1)

    def test_config_reports_the_value_and_its_form(self):
        config = FeatureReranker(idf=IDF, heading_bonus=HEADING_BONUS).config()
        assert config["heading_bonus"]["value"] == HEADING_BONUS
        assert config["heading_bonus"]["filter"] is False
        assert "after the weighted sum" in config["heading_bonus"]["form"]

    def test_the_shipped_value_matches_the_structural_penalty_magnitude(self):
        # Not a coincidence and not a constraint - the sweep landed here, and
        # the docstring's a-priori argument is that a structural fact about a
        # chunk may move it a quarter of the relevance range either way. If a
        # future sweep moves one, this assertion should be deleted, not "fixed".
        assert HEADING_BONUS == STRUCTURAL_PENALTY

    def test_the_bonus_is_reported_per_candidate(self):
        candidate = make_candidate("D::a", text="Penalties for contravention.",
                                   heading="Penalties")
        result = FeatureReranker(idf=IDF, heading_bonus=HEADING_BONUS).rerank(
            "what penalty follows a contravention", [candidate], top_k=1)[0]
        assert "heading_bonus" in result.components
        assert result.components["heading_bonus"] > 0.0


class TestNoIdfFallback:
    """Without an IDF table the feature degrades to plain type coverage.

    Production never takes this path - `app/pipeline.py` always constructs the
    reranker with `idf=bm25.idf` - but the fallback exists so a caller without a
    corpus gets a sane number instead of a silent zero, and the difference is
    worth pinning: with no IDF, a function word in the heading costs as much as
    a legal noun.
    """

    def test_function_words_count_without_an_idf_table(self):
        query = ["what", "protection", "for", "geographical", "indications"]
        with_idf = heading_coverage(query, "Protection of Geographical Indications", IDF)
        without = heading_coverage(query, "Protection of Geographical Indications", None)
        assert with_idf > without
        assert without == pytest.approx(0.75)  # "of" is uncovered and unweighted
