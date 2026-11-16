"""Citation locators, chunk splitting, provenance and the compendium split."""
import os

from ingest.chunk import split_unit
from ingest.citation import build_citation
from ingest.common import CHUNKS_DIR, logical_documents, read_jsonl
from ingest.parsers.base import empty_locator


def locator(**values):
    base = empty_locator()
    base.update(values)
    return base


class TestCitationLocator:
    def test_indian_section_with_a_clause(self):
        citation, unavailable = build_citation(
            "Patents Act, 1970", locator(section="3", clause="p")
        )
        assert citation == "Patents Act, 1970 — Section 3(p)"
        assert unavailable is False

    def test_indian_rule_with_a_sub_rule(self):
        citation, _ = build_citation("Patents Rules, 2003", locator(rule="13", sub_rule="2"))
        assert citation == "Patents Rules, 2003 — Rule 13(2)"

    def test_trips_uses_dotted_paragraphs(self):
        citation, _ = build_citation(
            "TRIPS", locator(article="27", paragraph="1"), citation_style="dot"
        )
        assert citation == "TRIPS — Article 27.1"

    def test_cbd_uses_parenthesised_paragraphs(self):
        citation, _ = build_citation(
            "CBD", locator(article="15", paragraph="7"), citation_style="paren"
        )
        assert citation == "CBD — Article 15(7)"

    def test_nagoya_uses_dotted_paragraphs(self):
        citation, _ = build_citation(
            "Nagoya Protocol", locator(article="5", paragraph="2"), citation_style="dot"
        )
        assert citation == "Nagoya Protocol — Article 5.2"

    def test_a_schedule_is_cited_as_a_schedule(self):
        citation, unavailable = build_citation(
            "Trade Marks Rules, 2017", locator(schedule="FIRST"), unit_type="schedule"
        )
        assert citation == "Trade Marks Rules, 2017 — First Schedule"
        assert unavailable is False

    def test_a_form_is_cited_as_a_form(self):
        citation, _ = build_citation(
            "Patents Rules, 2003", locator(form="1"), unit_type="form"
        )
        assert citation == "Patents Rules, 2003 — Form 1"

    def test_a_chapter_with_no_provision_is_still_citable(self):
        citation, unavailable = build_citation(
            "Copyright Act, 1957", locator(chapter="XI"), unit_type="heading"
        )
        assert citation == "Copyright Act, 1957 — Chapter XI"
        assert unavailable is False

    def test_an_empty_locator_is_declared_unavailable_and_never_invented(self):
        citation, unavailable = build_citation(
            "Food Safety and Standards (Ayurveda Aahara) Regulations, 2022",
            locator(), unit_type="unstructured", pages=[57],
        )
        assert unavailable is True
        assert "unlocated text" in citation
        assert "physical p. 57" in citation


class TestChunkSplitting:
    def test_a_short_provision_is_never_split(self):
        unit = {
            "lines": [[1, "13. Specifications.—(1) Every specification shall be in Form 2."]],
            "character_count": 62,
            "subdivisions": [],
        }
        assert len(split_unit(unit)) == 1

    def test_a_long_provision_splits_at_sub_provision_boundaries(self):
        filler = "x" * 300
        lines = [[1, "13. Specifications.—"]]
        subdivisions = []
        for index in range(1, 30):
            subdivisions.append([len(lines), "sub_rule", str(index)])
            lines.append([1, f"({index}) {filler}"])
        unit = {
            "lines": lines,
            "character_count": sum(len(line[1]) + 1 for line in lines),
            "subdivisions": subdivisions,
        }
        pieces = split_unit(unit)
        assert len(pieces) > 1
        # The opening piece covers the words before sub-rule (1) and so carries
        # no sub-locator; every later piece opens on a boundary and names it.
        assert pieces[0][1] == {}
        assert all(piece[1].get("sub_rule") for piece in pieces[1:])
        assert all(sum(len(line[1]) + 1 for line in piece) <= 6000 for piece, _ in pieces)

    def test_a_unit_with_no_sub_provisions_still_respects_the_ceiling(self):
        lines = [[1, "y" * 200] for _ in range(80)]
        unit = {
            "lines": lines,
            "character_count": sum(len(line[1]) + 1 for line in lines),
            "subdivisions": [],
        }
        pieces = split_unit(unit)
        assert len(pieces) > 1
        assert all(sum(len(line[1]) + 1 for line in piece) <= 6000 for piece, _ in pieces)


class TestCompendiumMapping:
    """One PDF, two citable instruments, non-overlapping page ranges."""

    def test_the_drugs_and_cosmetics_pdf_yields_two_logical_documents(self):
        documents = {d.document_id: d for d in logical_documents()}
        act = documents["IN-DC-ACT-1940"]
        rules = documents["IN-DC-RULES-1945"]
        assert act.parent_source_document_id == "IN-DC-ACT1940-RULES1945"
        assert rules.parent_source_document_id == "IN-DC-ACT1940-RULES1945"
        assert act.source_filename == rules.source_filename
        assert act.source_sha256 == rules.source_sha256

    def test_their_page_ranges_do_not_overlap(self):
        documents = {d.document_id: d for d in logical_documents()}
        act = documents["IN-DC-ACT-1940"]
        rules = documents["IN-DC-RULES-1945"]
        assert act.page_end < rules.page_start
        assert set(act.page_range).isdisjoint(rules.page_range)

    def test_each_is_cited_by_its_own_provision_label(self):
        documents = {d.document_id: d for d in logical_documents()}
        assert documents["IN-DC-ACT-1940"].provision_label == "Section"
        assert documents["IN-DC-RULES-1945"].provision_label == "Rule"

    def test_no_other_pdf_is_split(self):
        parents = [d for d in logical_documents() if d.parent_source_document_id]
        assert {d.document_id for d in parents} == {"IN-DC-ACT-1940", "IN-DC-RULES-1945"}


class TestPageToChunkProvenance:
    """Runs against the produced corpus; skipped before the pipeline has run."""

    def _chunks(self, document_id):
        path = os.path.join(CHUNKS_DIR, f"{document_id}.jsonl")
        if not os.path.exists(path):
            import pytest

            pytest.skip("pipeline output not present")
        return read_jsonl(path)

    def test_every_chunk_names_its_source_and_pages(self):
        for chunk in self._chunks("IN-PATENTS-ACT-1970"):
            assert chunk["source_sha256"]
            assert chunk["source_filename"]
            assert chunk["physical_pages"]

    def test_pages_stay_inside_the_logical_document(self):
        documents = {d.document_id: d for d in logical_documents()}
        for document_id in ("IN-DC-ACT-1940", "IN-DC-RULES-1945"):
            document = documents[document_id]
            for chunk in self._chunks(document_id):
                for page in chunk["physical_pages"]:
                    assert document.page_start <= page <= document.page_end

    def test_a_known_provision_resolves_to_its_page_and_citation(self):
        chunks = self._chunks("IN-PATENTS-ACT-1970")
        section_3 = [c for c in chunks if c["locator"]["section"] == "3"]
        assert section_3, "Patents Act section 3 should be chunked"
        chunk = section_3[0]
        assert chunk["citation"].startswith("Patents Act, 1970 — Section 3")
        assert chunk["locator_unavailable"] is False
        assert "not inventions" in chunk["text"]
