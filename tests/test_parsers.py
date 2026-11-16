"""Structural parsing, on synthetic fixtures rather than the corpus."""
from ingest.parsers import statutory, treaty
from ingest.parsers.base import accepts_next, extract_heading, parse_number


def lines(text: str, page: int = 1):
    return [(page, line) for line in text.split("\n")]


ACT = """CHAPTER II
INVENTIONS NOT PATENTABLE
3. What are not inventions.—The following are not inventions within the meaning of this Act,—
(a) an invention which is frivolous;
(p) an invention which, in effect, is traditional knowledge.
4. Inventions relating to atomic energy not patentable.—No patent shall be granted.
"""

# A fixture must open its own numbering series, exactly as a real instrument
# does: the parser will not accept a high number as the first provision it sees.
RULES = """1. Short title and commencement.—These rules may be called the Test Rules.
13. Specifications.—(1) Every specification shall be made in Form 2.
(2) A specification in respect of a divisional application shall contain a reference.
14. Amendments to specifications.—Any amendment shall be made in the manner specified.
"""


class TestSectionDetection:
    def test_sections_are_split_at_their_numbers(self):
        units = statutory.parse(lines(ACT), "Section")
        provisions = [u for u in units if u.unit_type == "provision"]
        assert [u.number for u in provisions] == ["3", "4"]

    def test_section_carries_its_chapter_and_heading(self):
        units = statutory.parse(lines(ACT), "Section")
        section_3 = next(u for u in units if u.number == "3")
        assert section_3.locator["chapter"] == "II"
        assert section_3.locator["section"] == "3"
        assert section_3.heading == "What are not inventions"

    def test_clauses_stay_with_their_section(self):
        units = statutory.parse(lines(ACT), "Section")
        section_3 = next(u for u in units if u.number == "3")
        assert "traditional knowledge" in section_3.text

    def test_lettered_sections_are_kept_distinct(self):
        text = ("1. Short title.—This Act may be called the Test Act.\n"
                "33E. Misbranded drugs.—A drug shall be deemed misbranded.\n"
                "33EEB. Regulation of manufacture.—No person shall manufacture.\n")
        units = statutory.parse(lines(text), "Section")
        assert [u.number for u in units if u.unit_type == "provision"] == \
            ["1", "33E", "33EEB"]

    def test_enumerated_list_inside_a_proviso_is_not_a_new_section(self):
        text = ("1. Short title.—This Act may be called the Test Act.\n"
                "62. Records to be kept.—The licensee shall record:\n"
                "1. Name of the drug,\n"
                "2. Batch number,\n"
                "3. Date of transaction.\n")
        units = statutory.parse(lines(text), "Section")
        assert [u.number for u in units if u.unit_type == "provision"] == ["1", "62"]


class TestSubsectionDetection:
    def test_subsections_are_found_in_a_section(self):
        units = statutory.parse(lines(ACT.replace("(a)", "(1)")), "Section")
        section_3 = next(u for u in units if u.number == "3")
        subs = statutory.subdivisions(section_3.lines, "section")
        assert [marker for _, _, marker in subs] == ["1"]
        assert subs[0][1] == "subsection"

    def test_subsections_must_ascend(self):
        block = lines("(1) first\n(5) fifth\n(2) stray back-reference\n")
        subs = statutory.subdivisions(block, "section")
        assert [marker for _, _, marker in subs] == ["1", "5"]


class TestRuleDetection:
    def test_rules_are_numbered_as_rules_not_sections(self):
        units = statutory.parse(lines(RULES), "Rule")
        rule_13 = next(u for u in units if u.number == "13")
        assert rule_13.locator["rule"] == "13"
        assert rule_13.locator["section"] is None

    def test_sub_rules_are_labelled_sub_rule(self):
        units = statutory.parse(lines(RULES), "Rule")
        rule_13 = next(u for u in units if u.number == "13")
        kinds = {kind for _, kind, _ in statutory.subdivisions(rule_13.lines, "rule")}
        assert kinds == {"sub_rule"}

    def test_a_form_reference_mid_sentence_does_not_open_a_form(self):
        text = ("1. Short title.—These rules may be called the Test Rules.\n"
                "24. Time for filing.—An application shall be made in\n"
                "Form 4 for extension of time along with the prescribed fee.\n"
                "25. Further provision.—Something else applies here.\n")
        units = statutory.parse(lines(text), "Rule")
        assert not [u for u in units if u.unit_type == "form"]
        assert [u.number for u in units if u.unit_type == "provision"] == ["1", "24", "25"]

    def test_a_standalone_form_heading_does_open_a_form(self):
        text = "FORM 1\nAPPLICATION FOR GRANT OF PATENT\n1. Applicant's reference\n"
        units = statutory.parse(lines(text), "Rule")
        assert [u.number for u in units if u.unit_type == "form"] == ["1"]

    def test_schedule_entries_are_numbered_as_items_not_rules(self):
        """Phase 5 §1. A Schedule restarts numbering at 1, and its entry 1 is not
        rule 1 of the instrument. Borrowing the provision key put the entry into
        direct competition with the real provision for identifier queries."""
        text = ("1. Short title.—These rules may be called the Test Rules.\n"
                "40. Last rule.—This is the final rule.\n"
                "THE FIRST SCHEDULE\n"
                "1. First schedule entry.—Text of the entry.\n")
        units = statutory.parse(lines(text), "Rule")
        entry = units[-1]
        assert entry.unit_type == "schedule_item"
        assert entry.locator["schedule"] == "FIRST"
        assert entry.locator["item"] == "1"
        assert entry.locator["rule"] is None


class TestArticleDetection:
    def test_inline_article_heading(self):
        text = ("Article 27\n\nPatentable Subject Matter\n"
                "1. Patents shall be available for any inventions.\n")
        units = treaty.parse(lines(text))
        article = next(u for u in units if u.unit_type == "article")
        assert article.number == "27"
        assert article.locator["article"] == "27"

    def test_article_number_on_its_own_line_is_joined_to_the_heading(self):
        # The CBD and the Nagoya Protocol set the rubric as a centred block.
        text = ("Article\n15\naCCess to genetiC resourCes\n"
                "1. Recognizing the sovereign rights of States over their natural resources.\n")
        units = treaty.parse(lines(text))
        article = next(u for u in units if u.unit_type == "article")
        assert article.number == "15"
        assert article.heading == "aCCess to genetiC resourCes"

    def test_text_before_the_first_article_is_the_preamble(self):
        text = "The Contracting Parties,\nConscious of the intrinsic value.\nArticle 1\nOBJECTIVES\n"
        units = treaty.parse(lines(text))
        assert units[0].unit_type == "preamble"

    def test_treaty_parts_and_sections_are_tracked(self):
        text = ("PART II\nSECTION 5: PATENTS\nArticle 27\nPatentable Subject Matter\n"
                "1. Patents shall be available.\n")
        units = treaty.parse(lines(text))
        article = next(u for u in units if u.unit_type == "article")
        assert article.locator["part"] == "II"
        assert article.locator["section"] == "5"


class TestParagraphDetection:
    def test_numbered_paragraphs_inside_an_article(self):
        text = ("Article 15\nACCESS\n"
                "1. Recognizing the sovereign rights of States.\n"
                "2. The authority to determine access rests with governments.\n"
                "3. Access shall be subject to prior informed consent.\n")
        units = treaty.parse(lines(text))
        article = next(u for u in units if u.unit_type == "article")
        paragraphs = treaty.subdivisions(article.lines)
        assert [marker for _, _, marker in paragraphs] == ["1", "2", "3"]
        assert {kind for _, kind, _ in paragraphs} == {"paragraph"}


class TestNumberGuards:
    def test_parse_number_splits_digits_from_letters(self):
        assert parse_number("33EEB") == (33, "EEB")

    def test_first_provision_must_be_low(self):
        assert accepts_next(None, (1, ""))
        assert not accepts_next(None, (17, ""))

    def test_numbering_must_ascend(self):
        assert accepts_next((11, ""), (11, "A"))
        assert not accepts_next((11, "A"), (11, ""))

    def test_bounded_jump(self):
        assert not accepts_next((12, ""), (99, ""))
        assert accepts_next((12, ""), (99, ""), max_jump=200)


class TestHeadingExtraction:
    def test_em_dash_marginal_note(self):
        assert extract_heading("What are not inventions.—The following are not") == \
            "What are not inventions"

    def test_whole_line_heading(self):
        assert extract_heading("Short title, extent and commencement.") == \
            "Short title, extent and commencement"

    def test_operative_text_is_not_a_heading(self):
        assert extract_heading("A licence shall be valid for three years. The holder") is None

    def test_a_citation_of_another_act_is_not_a_heading(self):
        assert extract_heading("The Patents (Amendment) Act, 2002 (38 of 2002).") is None
