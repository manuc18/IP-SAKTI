"""Cleaning: running heads and feet, contents pages, amendment footnotes."""
from ingest.clean import (
    collect_boilerplate,
    dot_leader_fraction,
    split_footnotes,
    strip_edges,
    strip_page_numbers,
    toc_like,
)
from ingest.common import classify_language

HEADER = "THE GAZETTE OF INDIA : EXTRAORDINARY  [PART II—SEC. 3(i)]"


def page(number: int, body: str) -> list[str]:
    return [HEADER, str(number), *body.split("\n"), "IndiaCode"]


class TestHeaderFooterRemoval:
    def test_repeated_edge_lines_become_boilerplate(self):
        pages = [page(n, f"Rule {n} text that is unique to this page.") for n in range(1, 11)]
        boilerplate = collect_boilerplate(pages)
        assert "THE GAZETTE OF INDIA : EXTRAORDINARY [PART II—SEC. #(i)]" in boilerplate

    def test_header_and_footer_are_stripped_but_body_is_not(self):
        # Bodies must differ in words, not only in digits: the boilerplate
        # detector normalises digits to "#" so that "page 4" and "page 5" are
        # recognised as the same running head.
        bodies = ["alpha", "bravo", "charlie", "delta", "echo",
                  "foxtrot", "golf", "hotel", "india", "juliet"]
        pages = [page(n, f"The licensee shall keep a {word} register.")
                 for n, word in enumerate(bodies, start=1)]
        boilerplate = collect_boilerplate(pages)
        kept, removed = strip_edges(pages[4], boilerplate)
        assert kept == ["The licensee shall keep a echo register."]
        assert "IndiaCode" in removed

    def test_a_line_recurring_only_twice_is_not_boilerplate(self):
        pages = [["unique %d" % n, "body"] for n in range(10)]
        pages[0][0] = pages[1][0] = "seen twice"
        assert "seen twice" not in collect_boilerplate(pages)

    def test_structural_headings_survive_the_header_stripper(self):
        # In TRIPS the Article heading *is* the top line of most pages.
        pages = [["Article %d" % n, "Body text of the article."] for n in range(1, 12)]
        boilerplate = collect_boilerplate(pages)
        assert "Article #" in boilerplate
        kept, _ = strip_edges(pages[3], boilerplate)
        assert kept[0] == "Article 4"

    def test_stray_folio_is_removed_from_mid_page(self):
        kept, removed = strip_page_numbers(["text above", "42", "text below"], 42)
        assert kept == ["text above", "text below"]
        assert removed == ["42"]

    def test_a_number_far_from_the_folio_is_left_alone(self):
        kept, _ = strip_page_numbers(["quantity", "500", "grams"], 42)
        assert kept == ["quantity", "500", "grams"]


class TestTableOfContentsDetection:
    def test_a_contents_page_scores_high(self):
        contents = [
            "ARRANGEMENT OF SECTIONS",
            "1.   Short title, extent and commencement.",
            "2.   Definitions and interpretation.",
            "3.   What are not inventions.",
            "4.   Inventions relating to atomic energy not patentable.",
            "11A. Publication of applications.",
        ]
        assert toc_like(contents) >= 0.5

    def test_a_body_page_does_not(self):
        body = [
            "3. What are not inventions.—The following are not inventions within the",
            "meaning of this Act, namely, an invention which is frivolous or which claims",
            "anything obviously contrary to well established natural laws;",
            "(b) an invention the primary or intended use of which would be contrary to",
            "public order or morality or which causes serious prejudice to human life;",
            "(c) the mere discovery of a scientific principle.",
        ]
        assert toc_like(body) < 0.5

    def test_dot_leaders_are_recognised(self):
        contents = [
            "EXECUTIVE SUMMARY ...................................................1",
            "INTRODUCTION .......................................................3",
            "OVERVIEW ...........................................................3",
            "OBJECTIVES .........................................................7",
        ]
        assert dot_leader_fraction(contents) >= 0.5


class TestAmendmentNotes:
    def test_footnote_block_is_lifted_off_the_page(self):
        body = [
            "(c) “artistic work” means a painting, a sculpture, a drawing (including a",
            "diagram, map, chart or plan), an engraving or a photograph, whether or not",
            "any such work possesses artistic quality; a work of architecture; and",
            "(iii) any other work of artistic craftsmanship;",
            "1 Subs. by Act 38 of 1994, s. 2, for “architectural work of art” (w.e.f. 10-5-1995).",
            "2 Ins. by Act 23 of 1983, s. 3 (w.e.f. 9-8-1984).",
        ]
        kept, notes = split_footnotes(body)
        assert kept[-1] == "(iii) any other work of artistic craftsmanship;"
        assert len(notes) == 2
        assert notes[0].startswith("1 Subs. by Act 38 of 1994")

    def test_a_wrapped_footnote_keeps_its_continuation(self):
        body = [
            "(f) “cinematograph film” means any work of visual recording and includes a",
            "sound recording accompanying such visual recording and “cinematograph”",
            "shall be construed as including any work produced by any process analogous",
            "to cinematography including video films;",
            "5 The words “on any medium produced through a process” omitted by Act 27",
            "of 2012, s. 2(i) (w.e.f. 21-6-2012).",
        ]
        _, notes = split_footnotes(body)
        assert len(notes) == 1
        assert notes[0].endswith("(w.e.f. 21-6-2012).")

    def test_an_ordinary_numbered_list_is_not_a_footnote(self):
        body = ["The licensee shall record:", "1. Name of the drug,", "2. Batch number."]
        kept, notes = split_footnotes(body)
        assert notes == []
        assert kept == body

    def test_footnote_numbering_must_ascend_down_the_page(self):
        body = [
            "(2) The licensing authority may require such further particulars to be",
            "supplied as he may consider necessary before granting the licence, and may",
            "refuse the application for reasons to be recorded in writing;",
            "3 Ins. by Act 5 of 2000, s. 4 (w.e.f. 1-1-2001).",
            "1 Subs. by Act 9 of 1999, s. 2 (w.e.f. 1-1-2000).",
        ]
        _, notes = split_footnotes(body)
        assert len(notes) == 1


class TestLanguageClassification:
    def test_english_is_clean(self):
        assert classify_language("The Controller shall refuse the application.") == "EN_CLEAN"

    def test_devanagari_dominant_native_text_is_reported_corrupt(self):
        assert classify_language("प्राजधकरण अजधजनयम की धारा") == "HI_CORRUPT"

    def test_devanagari_from_ocr_is_pending_not_corrupt(self):
        assert classify_language("प्राधिकरण अधिनियम की धारा", extraction_method="ocr") == \
            "HI_OCR_PENDING"

    def test_mixed_script_is_flagged(self):
        assert classify_language(
            "The Authority shall notify अजधसूचना in the Gazette of India today"
        ) == "EN_HI_MIXED"

    def test_empty_text_is_unknown(self):
        assert classify_language("   ") == "UNKNOWN"
