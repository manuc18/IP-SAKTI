"""Phase 7.1 — citations must distinguish, not merely name.

The Drugs and Cosmetics Rules, 1945 contain Part VI rule 65 (conditions of
licence, pp. 71-72) and Part II rule 65 (a schedule entry, p. 533). Both
rendered as ``Drugs and Cosmetics Rules, 1945 — Rule 65``, so a reader given two
footnotes could not tell which provision each pointed at. The chunk ids always
separated them and the citation objects always resolved; the human-readable
string did not.

These tests fix the general rule, not the two rules that exposed it: no
document_id, rule number or Part appears in an assertion as a special case.
Everything is CPU-only and offline.
"""
import os
import sys

from ingest.citation import (
    DISAMBIGUATION_FIELDS,
    build_citation,
    disambiguate_citations,
)
from ingest.parsers.base import empty_locator

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.answering.citation_validator import build_citation as build_citation_object  # noqa: E402
from ai.answering.models import Evidence, EvidenceSet  # noqa: E402
from ai.answering.policy import detect_conflicts  # noqa: E402
from ai.retrieval.base import RetrievalResult  # noqa: E402

TITLE = "Drugs and Cosmetics Rules, 1945"


def locator(**values):
    base = empty_locator()
    base.update(values)
    return base


def chunk(chunk_id, *, unit_type="provision", pages=None, citable=True, **fields):
    """A chunk as `disambiguate_citations` sees it, with its citation prebuilt."""
    loc = locator(**fields)
    pages = pages or [1]
    citation, unavailable = build_citation(TITLE, loc, unit_type=unit_type, pages=pages)
    return {"chunk_id": chunk_id, "locator": loc, "unit_type": unit_type,
            "physical_pages": pages, "citation": citation,
            "locator_unavailable": unavailable, "citable": citable}


def colliding_pair():
    """The real corpus case: one rule number, two Parts, two different texts."""
    return [chunk("doc::part-vi-rule-65-c1", part="VI", rule="65", pages=[71, 72]),
            chunk("doc::part-ii-rule-65", part="II", rule="65", pages=[533])]


class TestCollisionIsDetectedAndResolved:
    def test_the_pair_collides_before_the_pass(self):
        chunks = colliding_pair()
        assert chunks[0]["citation"] == chunks[1]["citation"] == f"{TITLE} — Rule 65"

    def test_part_vi_rule_65_gains_its_part(self):
        chunks = colliding_pair()
        disambiguate_citations(chunks, short_title=TITLE)
        assert chunks[0]["citation"] == f"{TITLE} — Part VI, Rule 65"

    def test_part_ii_rule_65_gains_its_part(self):
        chunks = colliding_pair()
        disambiguate_citations(chunks, short_title=TITLE)
        assert chunks[1]["citation"] == f"{TITLE} — Part II, Rule 65"

    def test_the_two_citations_are_now_distinguishable(self):
        chunks = colliding_pair()
        changed = disambiguate_citations(chunks, short_title=TITLE)
        assert changed == 2
        assert len({c["citation"] for c in chunks}) == 2

    def test_chapter_separates_when_part_does_not(self):
        """The rule is a declared field order, not a hard-coded 'part'."""
        chunks = [chunk("doc::chapter-i-rule-1", chapter="I", rule="1"),
                  chunk("doc::chapter-xviii-rule-1", chapter="XVIII", rule="1")]
        disambiguate_citations(chunks, short_title=TITLE)
        assert {c["citation"] for c in chunks} == {
            f"{TITLE} — Chapter I, Rule 1", f"{TITLE} — Chapter XVIII, Rule 1"}

    def test_part_is_preferred_over_chapter_when_both_differ(self):
        chunks = [chunk("doc::a", part="VI", chapter="I", rule="65"),
                  chunk("doc::b", part="II", chapter="IX", rule="65")]
        disambiguate_citations(chunks, short_title=TITLE)
        assert all(c["citation"].split(" — ")[1].startswith("Part ") for c in chunks)

    def test_the_field_order_is_declared_not_incidental(self):
        assert DISAMBIGUATION_FIELDS == (("part", "Part"), ("chapter", "Chapter"))


class TestNonCollidingCitationsArePreserved:
    def test_a_normal_rule_65_style_citation_is_untouched(self):
        chunks = [chunk("doc::part-ix-rule-96", part="IX", rule="96"),
                  chunk("doc::part-vi-rule-65", part="VI", rule="65")]
        before = [c["citation"] for c in chunks]
        assert disambiguate_citations(chunks, short_title=TITLE) == 0
        assert [c["citation"] for c in chunks] == before
        assert chunks[0]["citation"] == f"{TITLE} — Rule 96"

    def test_a_provision_split_across_chunks_keeps_one_citation(self):
        """All the pieces really are Rule 65; separating them would be a lie."""
        chunks = [chunk("doc::part-vi-rule-65-c1", part="VI", rule="65"),
                  chunk("doc::part-vi-rule-65-c2", part="VI", rule="65"),
                  chunk("doc::part-vi-rule-65-c3", part="VI", rule="65")]
        assert disambiguate_citations(chunks, short_title=TITLE) == 0
        assert len({c["citation"] for c in chunks}) == 1

    def test_a_single_chunk_is_never_disambiguated(self):
        chunks = [chunk("doc::part-vi-rule-65", part="VI", rule="65")]
        assert disambiguate_citations(chunks, short_title=TITLE) == 0

    def test_a_colliding_chunk_without_the_field_keeps_its_citation(self):
        """Nothing truthful can be promoted for it, so nothing is invented."""
        chunks = [chunk("doc::part-vi-rule-65", part="VI", rule="65"),
                  chunk("doc::rule-65", rule="65")]
        disambiguate_citations(chunks, short_title=TITLE)
        assert chunks[0]["citation"] == f"{TITLE} — Part VI, Rule 65"
        assert chunks[1]["citation"] == f"{TITLE} — Rule 65"

    def test_sub_rule_tails_survive_disambiguation(self):
        chunks = [chunk("doc::a", part="VI", rule="65", sub_rule="4"),
                  chunk("doc::b", part="II", rule="65", sub_rule="4")]
        disambiguate_citations(chunks, short_title=TITLE)
        assert chunks[0]["citation"] == f"{TITLE} — Part VI, Rule 65(4)"

    def test_build_citation_never_adds_a_part_on_its_own(self):
        citation, _ = build_citation(TITLE, locator(part="VI", rule="65"))
        assert citation == f"{TITLE} — Rule 65"


class TestCitationObjectResolution:
    def _evidence(self, citation, chunk_id, pages, marker=1):
        result = RetrievalResult(
            chunk_id=chunk_id, document_id="IN-DC-RULES-1945", score=1.0, rank=1,
            text="Licences shall be subject to the conditions stated therein.",
            locator=citation, citable=True,
            metadata={"short_title": TITLE, "physical_pages": pages,
                      "locator_unavailable": False, "temporal_status": "STALE",
                      "source_evidence_level": "A",
                      "locator_fields": {"part": "VI", "rule": "65"},
                      "source_filename": "dc.pdf", "source_sha256": "deadbeef"})
        return Evidence.from_result(result, marker)

    def test_the_disambiguated_citation_resolves_to_a_citation_object(self):
        chunks = colliding_pair()
        disambiguate_citations(chunks, short_title=TITLE)
        evidence = self._evidence(chunks[0]["citation"], chunks[0]["chunk_id"],
                                  chunks[0]["physical_pages"])
        citation = build_citation_object(evidence)
        assert citation.citation_text == f"{TITLE} — Part VI, Rule 65"
        assert citation.chunk_id == "doc::part-vi-rule-65-c1"
        assert citation.pages == (71, 72)

    def test_the_part_survives_into_the_rendered_locator(self):
        chunks = colliding_pair()
        disambiguate_citations(chunks, short_title=TITLE)
        evidence = self._evidence(chunks[0]["citation"], chunks[0]["chunk_id"],
                                  chunks[0]["physical_pages"])
        assert evidence.locator_text == "Part VI, Rule 65"

    def _evidence_set(self, chunks):
        return EvidenceSet(tuple(
            self._evidence(c["citation"], c["chunk_id"], c["physical_pages"],
                           marker=index)
            for index, c in enumerate(chunks, start=1)))

    def test_disambiguated_citations_no_longer_read_as_ambiguous(self):
        chunks = colliding_pair()
        disambiguate_citations(chunks, short_title=TITLE)
        kinds = {c.kind for c in detect_conflicts(self._evidence_set(chunks))}
        assert "ambiguous_citation" not in kinds

    def test_colliding_citations_still_read_as_ambiguous_before_the_fix(self):
        """The Phase 7 detector fired on this pair; that is what was fixed."""
        kinds = {c.kind for c in detect_conflicts(self._evidence_set(colliding_pair()))}
        assert "ambiguous_citation" in kinds


class TestCitabilityInvariants:
    def test_disambiguation_never_changes_citability(self):
        chunks = colliding_pair()
        disambiguate_citations(chunks, short_title=TITLE)
        assert all(c["citable"] is True for c in chunks)

    def test_disambiguation_never_clears_locator_unavailable(self):
        chunks = colliding_pair()
        disambiguate_citations(chunks, short_title=TITLE)
        assert all(c["locator_unavailable"] is False for c in chunks)

    def test_an_unlocated_chunk_is_not_given_a_locator(self):
        """No locator field can separate them, so nothing is promoted."""
        chunks = [chunk("doc::u0", unit_type="unstructured", pages=[1], citable=False),
                  chunk("doc::u1", unit_type="unstructured", pages=[1], citable=False)]
        assert disambiguate_citations(chunks, short_title=TITLE) == 0
        assert all(c["locator_unavailable"] is True for c in chunks)
        assert all(c["citable"] is False for c in chunks)

    def test_a_non_citable_chunk_stays_non_citable_beside_a_citable_one(self):
        chunks = [chunk("doc::part-vi-rule-65", part="VI", rule="65"),
                  chunk("doc::u0", unit_type="unstructured", pages=[9], citable=False)]
        disambiguate_citations(chunks, short_title=TITLE)
        assert chunks[0]["citable"] is True
        assert chunks[1]["citable"] is False
        assert chunks[1]["locator_unavailable"] is True
