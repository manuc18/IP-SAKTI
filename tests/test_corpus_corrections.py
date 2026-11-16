"""Phase 5 — the two structural bugs Phase 4 found, and the guards against them.

Each bug gets tests at two levels, because fixing the corpus and preventing the
next one are different jobs.

*Parser* tests run on synthetic fixtures and pin the rule: a Schedule entry is
numbered as an entry, a treaty Article number set on its own line survives, a
heading with no recoverable number does not silently join the Article above it.

*Corpus* tests assert the repair actually landed in the built chunks, and are
skipped when the pipeline output is absent so the suite still runs on a clean
checkout.

*Validator* tests feed the validator the broken shapes on purpose. These are the
ones that matter in six months: they fail if anybody reintroduces a locator that
the source does not support.
"""
from __future__ import annotations

import json
import os

import pytest

from ingest.citation import build_citation
from ingest.clean import strip_page_numbers
from ingest.parsers import statutory, treaty
from ingest.parsers.base import LOCATOR_KEYS, empty_locator
from ingest.validate import locator_defects, unsupported_locator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHUNKS = os.path.join(ROOT, "data", "processed", "chunks")


def lines(text: str, page: int = 1):
    return [(page, line) for line in text.split("\n")]


def load(document_id: str):
    path = os.path.join(CHUNKS, f"{document_id}.jsonl")
    if not os.path.exists(path):
        pytest.skip("pipeline output not present")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


class Doc:
    """The two registry fields the locator checks read."""

    def __init__(self, parser="statutory", provision_label="Section"):
        self.parser = parser
        self.provision_label = provision_label
        self.document_id = "TEST-DOC"


# ------------------------------------------- Objective 1: Schedule locators ---

# A fixture must open its own numbering series, exactly as a real instrument
# does: the parser will not accept a high number as the first provision it sees.
SCHEDULE_FIXTURE = """CHAPTER I
PRELIMINARY
1. Short title.—This Act may be called the Test Act.
CHAPTER XXIII
MISCELLANEOUS
162. Repeal and savings.—The enactments specified in the Schedule are amended.
THE SCHEDULE
[See section 162]
AMENDMENTS TO THE INDIAN PATENTS AND DESIGNS ACT, 1911
1. Long title—Omit "Inventions and".
3. Section 1—In sub-section (1) omit "Indian Patents and".
"""


class TestScheduleEntriesAreNotSections:
    def test_a_schedule_entry_referencing_a_section_is_not_that_section(self):
        units = statutory.parse(lines(SCHEDULE_FIXTURE), "Section")
        entry = next(u for u in units if u.number == "3" and u.unit_type == "schedule_item")
        assert entry.locator["section"] is None
        assert entry.locator["item"] == "3"
        assert "Section 1" in entry.text          # it does reference a section
        assert entry.locator["schedule"] == "SCHEDULE"

    def test_the_real_section_keeps_its_own_locator(self):
        units = statutory.parse(lines(SCHEDULE_FIXTURE), "Section")
        section = [u for u in units if u.unit_type == "provision"][-1]
        assert section.number == "162"
        assert section.locator["section"] == "162"
        assert section.locator["schedule"] is None

    def test_a_schedule_entry_is_cited_through_its_container(self):
        locator = empty_locator()
        locator.update({"schedule": "SCHEDULE", "item": "3"})
        citation, unavailable = build_citation("Patents Act, 1970", locator,
                                               unit_type="schedule_item")
        assert citation == "Patents Act, 1970 — Schedule, item 3"
        assert unavailable is False

    def test_numbering_that_resumes_the_body_leaves_the_container(self):
        # The Drugs and Cosmetics Rules print "Form 37." inside rule 150E's text.
        # Rule 150F resumes the instrument's own sequence, which is the evidence
        # that the Form never started.
        text = ("1. Short title.—These rules may be called the Test Rules.\n"
                "150. Report of result.—Tests shall be reported.\n"
                "Form 37.\n"
                "150F. Inspection before grant.—Before an approval is granted.\n")
        units = statutory.parse(lines(text), "Rule")
        resumed = next(u for u in units if u.number == "150F")
        assert resumed.unit_type == "provision"
        assert resumed.locator["rule"] == "150F"
        assert resumed.locator["form"] is None

    def test_the_item_key_exists_in_the_locator_vocabulary(self):
        assert "item" in LOCATOR_KEYS


class TestPatentsActSchedule:
    def test_the_real_section_3_is_identifiable_as_section_3(self):
        chunks = load("IN-PATENTS-ACT-1970")
        section_3 = [c for c in chunks if c["locator"].get("section") == "3"]
        assert len(section_3) == 1
        assert section_3[0]["citation"] == "Patents Act, 1970 — Section 3"
        assert section_3[0]["citable"] is True

    def test_section_3p_is_identifiable_as_section_3p(self):
        # s.3(p) is a clause of s.3 and lives inside the single s.3 chunk; the
        # identifier boost resolves "3(p)" through that chunk's section locator.
        chunks = load("IN-PATENTS-ACT-1970")
        section_3 = next(c for c in chunks if c["locator"].get("section") == "3")
        assert "(p)" in section_3["text"]
        assert "traditional knowledge" in section_3["text"].lower()

    def test_no_schedule_entry_claims_to_be_a_section(self):
        chunks = load("IN-PATENTS-ACT-1970")
        masquerading = [c["chunk_id"] for c in chunks
                        if c["locator"].get("schedule") and c["locator"].get("section")]
        assert masquerading == []

    def test_the_schedule_entry_that_mentions_section_3_is_an_item(self):
        chunks = load("IN-PATENTS-ACT-1970")
        entry = next((c for c in chunks if c["locator"].get("item") == "3"), None)
        assert entry is not None
        assert entry["locator"]["section"] is None
        assert entry["citation"] == "Patents Act, 1970 — Schedule, item 3"

    def test_no_document_in_the_corpus_mixes_a_container_with_a_provision(self):
        import glob
        paths = sorted(glob.glob(os.path.join(CHUNKS, "*.jsonl")))
        if not paths:
            pytest.skip("pipeline output not present")
        offenders = []
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    chunk = json.loads(line)
                    locator = chunk["locator"]
                    if any(locator.get(k) for k in ("schedule", "form", "annexure")) \
                            and any(locator.get(k) for k in ("section", "rule", "regulation")):
                        offenders.append(chunk["chunk_id"])
        assert offenders == []


# ---------------------------------------------- Objective 2: treaty locators ---

class TestTreatyArticleNumbersSurviveCleaning:
    def test_a_number_under_an_article_heading_is_not_a_page_folio(self):
        # This is the whole CBD bug: the Article number sits on its own line and
        # looks exactly like the printed folio, so it was stripped.
        kept, removed = strip_page_numbers(
            ["Article", "4", "jurisDiCtional sCoPe"], physical_page=10)
        assert kept == ["Article", "4", "jurisDiCtional sCoPe"]
        assert removed == []

    def test_a_real_folio_is_still_stripped(self):
        kept, removed = strip_page_numbers(
            ["Some running text.", "6", "More text."], physical_page=10)
        assert kept == ["Some running text.", "More text."]
        assert removed == ["6"]

    def test_a_letter_spaced_number_is_read_not_guessed(self):
        # The Nagoya Protocol's Article 11 reaches the text layer as "1 1".
        units = treaty.parse(lines("Article\n1 1\nTRANSBOUNDARY COOPERATION\n"
                                   "1. In instances where the same resource is shared.\n"))
        article = next(u for u in units if u.unit_type == "article")
        assert article.number == "11"
        assert article.locator["article"] == "11"


class TestUnnumberedArticlesAreNotGuessed:
    def test_a_heading_with_no_number_opens_its_own_unnumbered_article(self):
        text = ("Article\n8\nIN-SITU CONSERVATION\nEach Party shall establish areas.\n"
                "Article\nSOMETHING ELSE ENTIRELY\nA different obligation applies.\n")
        units = treaty.parse(lines(text))
        articles = [u for u in units if u.unit_type == "article"]
        assert len(articles) == 2
        assert articles[0].locator["article"] == "8"
        # The second names no number, so it gets none — and it does not get
        # folded into Article 8, which would file its words under 8's citation.
        assert articles[1].number is None
        assert articles[1].locator["article"] is None
        assert "different obligation" in articles[1].text

    def test_an_unnumbered_article_is_not_citable(self):
        locator = empty_locator()
        citation, unavailable = build_citation("CBD", locator, citation_style="paren",
                                               unit_type="article", pages=[7])
        assert unavailable is True
        assert "unlocated text" in citation


class TestCBDArticleLocators:
    def test_every_article_of_the_convention_is_located(self):
        chunks = load("UN-CBD-1992")
        articles = {c["locator"]["article"] for c in chunks if c["locator"].get("article")}
        assert {str(n) for n in range(1, 43)} <= articles

    def test_article_15_is_citable_as_article_15(self):
        chunks = load("UN-CBD-1992")
        article_15 = [c for c in chunks if c["locator"].get("article") == "15"]
        assert article_15
        assert all(c["citable"] for c in article_15)
        assert any(c["citation"].startswith("CBD — Article 15") for c in article_15)

    def test_no_preamble_chunk_is_citable_as_a_paragraph(self):
        # The Phase 4 defect exactly: "CBD — Paragraph 3", a provision that does
        # not exist, on text that is really Article 15(4).
        chunks = load("UN-CBD-1992")
        invented = [c["chunk_id"] for c in chunks
                    if c["unit_type"] == "preamble" and c["citable"]]
        assert invented == []

    def test_unlocated_preamble_text_stays_retrievable(self):
        chunks = load("UN-CBD-1992")
        unlocated = [c for c in chunks if not c["citable"]]
        assert unlocated
        assert all(c["retrievable"] for c in unlocated)
        assert all(c["locator_unavailable"] for c in unlocated)


class TestNagoyaArticleLocators:
    def test_every_article_of_the_protocol_is_located(self):
        chunks = load("UN-NAGOYA-PROTOCOL-2010")
        articles = {c["locator"]["article"] for c in chunks if c["locator"].get("article")}
        assert {str(n) for n in range(1, 37)} <= articles

    def test_no_preamble_chunk_is_citable_as_a_paragraph(self):
        chunks = load("UN-NAGOYA-PROTOCOL-2010")
        assert [c["chunk_id"] for c in chunks
                if c["unit_type"] == "preamble" and c["citable"]] == []

    def test_article_5_is_citable_as_article_5(self):
        chunks = load("UN-NAGOYA-PROTOCOL-2010")
        article_5 = [c for c in chunks if c["locator"].get("article") == "5"]
        assert article_5
        assert all(c["citable"] for c in article_5)


# --------------------------------------- Objective 3: the validator's guards ---

class TestValidatorRejectsUnsupportedLocators:
    def test_a_schedule_entry_carrying_a_section_is_fatal(self):
        chunk = {"locator": {**empty_locator(), "schedule": "SCHEDULE", "section": "3"},
                 "unit_type": "provision", "citable": True}
        checks = {check for check, _ in locator_defects(chunk, Doc())}
        assert "schedule_masquerading_as_provision" in checks

    def test_a_schedule_entry_carrying_only_an_item_passes(self):
        chunk = {"locator": {**empty_locator(), "schedule": "SCHEDULE", "item": "3"},
                 "unit_type": "schedule_item", "citable": True}
        assert locator_defects(chunk, Doc()) == []

    def test_a_treaty_preamble_carrying_a_paragraph_is_fatal(self):
        chunk = {"locator": {**empty_locator(), "paragraph": "3"},
                 "unit_type": "preamble", "citable": True}
        checks = {check for check, _ in locator_defects(chunk, Doc(parser="treaty"))}
        assert "treaty_preamble_locator" in checks
        assert "unsupported_locator" in checks

    def test_a_treaty_article_carrying_a_paragraph_passes(self):
        chunk = {"locator": {**empty_locator(), "article": "15", "paragraph": "4"},
                 "unit_type": "article", "citable": True}
        assert locator_defects(chunk, Doc(parser="treaty")) == []

    def test_a_bare_sub_unit_locator_is_not_a_citation(self):
        chunk = {"locator": {**empty_locator(), "subsection": "2"},
                 "unit_type": "provision", "citable": True}
        checks = {check for check, _ in locator_defects(chunk, Doc())}
        assert "unsupported_locator" in checks

    def test_an_item_with_no_container_is_not_a_citation(self):
        chunk = {"locator": {**empty_locator(), "item": "3"},
                 "unit_type": "schedule_item", "citable": True}
        assert unsupported_locator(chunk, Doc()) is not None

    def test_an_unsupported_locator_is_tolerated_when_the_chunk_is_not_citable(self):
        # Retrievable-but-not-citable is the honest outcome, not a defect.
        chunk = {"locator": {**empty_locator(), "paragraph": "3"},
                 "unit_type": "article", "citable": False}
        checks = {check for check, _ in locator_defects(chunk, Doc(parser="treaty"))}
        assert "unsupported_locator" not in checks

    def test_a_chapter_only_locator_is_supported(self):
        chunk = {"locator": {**empty_locator(), "chapter": "VI"},
                 "unit_type": "heading", "citable": True}
        assert locator_defects(chunk, Doc()) == []


class TestCitabilityInvariantHolds:
    def test_every_unlocated_chunk_in_the_corpus_is_non_citable(self):
        import glob
        paths = sorted(glob.glob(os.path.join(CHUNKS, "*.jsonl")))
        if not paths:
            pytest.skip("pipeline output not present")
        for path in paths:
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    chunk = json.loads(line)
                    if chunk["locator_unavailable"]:
                        assert chunk["citable"] is False, chunk["chunk_id"]
                    assert chunk["retrievable"] is True
