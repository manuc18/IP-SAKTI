"""Phase 7.8 — the caution says "cited", so it must count what was cited.

Phase 7.7 published the first answer this pipeline has ever published. It cited
one source out of five supplied, and told the reader:

    4 of 5 cited sources are marked STALE and may have been amended.

Four sources were *supplied*. One was cited. The count came from
:func:`apply_policy`, which runs before the provider and therefore cannot know
what will be cited — so it counted the citable supplied passages and wrote
"cited" over the top. Nothing caught it for four phases because every earlier
live answer abstained and no caution was ever rendered.

The fix is not better wording in `apply_policy`. It is that `apply_policy` does
not have the fact, and stops claiming to: :func:`citation_cautions` is called by
the engine once the citations exist. These tests pin the counting, the
deduplication and the wording, and one of them walks the whole engine to check
that an answer over five passages and one citation says "The cited source".

Offline, CPU-only. No network, no model.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.answering.base import GroundedAnswerEngine  # noqa: E402
from ai.answering.models import Citation, EvidenceSet, SourceDocument  # noqa: E402
from ai.answering.policy import (  # noqa: E402
    POLICY_VERSION,
    AnswerPolicy,
    apply_policy,
    citation_cautions,
)

from test_answer_contract_74 import (  # noqa: E402
    ENGINE_QUERY,
    SECTION_3_TEXT,
    FixedDraftProvider,
    evidence,
)


def citation(chunk_id, *, marker=1, temporal="CURRENT", level="A"):
    return Citation(
        marker=marker, chunk_id=chunk_id, document_id="IN-PATENTS-ACT-1970",
        locator="Section 3", pages=(11,), citation_text="Patents Act, 1970 — Section 3",
        source_evidence_level=level, temporal_status=temporal,
        locator_unavailable=False,
        source_document=SourceDocument(document_id="IN-PATENTS-ACT-1970",
                                       filename="patents.pdf", sha256="0" * 64))


def stale_text(cautions):
    return next((c for c in cautions if "stale" in c.lower()), None)


class TestCitedSourceCounting:
    """The numbers in the sentence are the numbers in the citation list."""

    def test_no_cited_sources_produces_no_caution(self):
        assert citation_cautions([]) == ()
        assert citation_cautions(None) == ()

    def test_one_cited_current_source_produces_no_stale_caution(self):
        assert stale_text(citation_cautions([citation("a")])) is None

    def test_one_cited_stale_source_does_not_say_one_of_one(self):
        text = stale_text(citation_cautions([citation("a", temporal="STALE")]))
        assert text.startswith("The cited source is marked stale.")
        assert "1 of 1" not in text

    def test_two_cited_stale_sources_say_both(self):
        text = stale_text(citation_cautions([
            citation("a", marker=1, temporal="STALE"),
            citation("b", marker=2, temporal="STALE")]))
        assert text.startswith("Both cited sources are marked stale.")

    def test_two_of_three_cited_sources_stale(self):
        text = stale_text(citation_cautions([
            citation("a", marker=1, temporal="STALE"),
            citation("b", marker=2, temporal="STALE"),
            citation("c", marker=3)]))
        assert text.startswith("2 of 3 cited sources are marked stale.")

    def test_a_mixed_set_agrees_with_itself_grammatically(self):
        text = stale_text(citation_cautions([
            citation("a", marker=1, temporal="STALE"), citation("b", marker=2)]))
        # "1 of 2 cited sources *is*". A caution a reader trips over is a caution
        # a reader stops reading.
        assert text.startswith("1 of 2 cited sources is marked stale.")

    def test_all_stale_beyond_two_says_all(self):
        text = stale_text(citation_cautions([
            citation(c, marker=i, temporal="STALE")
            for i, c in enumerate("abc", start=1)]))
        assert text.startswith("All 3 cited sources are marked stale.")

    def test_duplicate_markers_on_one_source_count_once(self):
        # Two markers can resolve to the same chunk; they are one source, and
        # counting them twice would overstate both halves of "n of m".
        text = stale_text(citation_cautions([
            citation("a", marker=1, temporal="STALE"),
            citation("a", marker=2, temporal="STALE"),
            citation("b", marker=3)]))
        assert text.startswith("1 of 2 cited sources is marked stale.")

    def test_the_caution_carries_the_advice_not_just_the_count(self):
        text = stale_text(citation_cautions([citation("a", temporal="STALE")]))
        assert "verify against the current official text" in text

    def test_weak_provenance_is_counted_the_same_way(self):
        # The adjacent caution had the identical defect and is fixed with it.
        cautions = citation_cautions([citation("a", level="C")])
        assert any(c.startswith("The cited source is at evidence level C.")
                   for c in cautions)


class TestPolicyNoLongerClaimsToKnow:
    """`apply_policy` runs before the provider, so it does not say "cited"."""

    def test_policy_emits_no_cited_source_caution(self):
        supplied = EvidenceSet(tuple(
            evidence(marker=i, text=SECTION_3_TEXT) for i in range(1, 6)))
        report = apply_policy(ENGINE_QUERY, supplied, AnswerPolicy())
        for caution in report.cautions:
            assert "cited source" not in caution.lower(), caution

    def test_policy_version_records_the_change(self):
        assert POLICY_VERSION == "0.3.0"


class TestThroughTheWholeEngine:
    """The Phase 7.7 shape: five supplied, one cited."""

    @staticmethod
    def supplied(*statuses):
        return EvidenceSet(tuple(
            _with_status(evidence(marker=i, text=SECTION_3_TEXT), status)
            for i, status in enumerate(statuses, start=1)))

    @staticmethod
    def answer(supplied, draft):
        engine = GroundedAnswerEngine(provider=FixedDraftProvider(draft))
        return engine.answer(ENGINE_QUERY, supplied)

    def test_five_supplied_one_cited_stale_says_the_cited_source(self):
        # The literal Phase 7.7 shape: five passages reach the engine, four of
        # them stale, and the answer cites one — the stale one.
        answer = self.answer(
            self.supplied("STALE", "STALE", "STALE", "STALE", "CURRENT"),
            "Traditional knowledge is not an invention [1].")

        assert not answer.insufficient_evidence, answer.insufficient_evidence_reasons
        assert len(answer.citations) == 1
        text = stale_text(answer.cautions)
        assert text is not None
        assert text.startswith("The cited source is marked stale.")
        # The published bug, in the exact shape it was published in.
        assert not any("4 of 5" in c for c in answer.cautions)

    def test_five_supplied_four_stale_but_the_cited_one_is_current(self):
        # The case the old wording got most wrong: the reader was warned about
        # staleness that had nothing to do with the source under the answer.
        answer = self.answer(
            self.supplied("CURRENT", "STALE", "STALE", "STALE", "STALE"),
            "Traditional knowledge is not an invention [1].")
        assert not answer.insufficient_evidence
        assert len(answer.citations) == 1
        assert stale_text(answer.cautions) is None

    def test_an_invalid_marker_yields_no_cited_source_caution(self):
        # [9] resolves to nothing, so validation fails and the engine abstains.
        # An abstention has no cited sources and must claim none — the same
        # defect in its other direction.
        answer = self.answer(
            self.supplied(*["STALE"] * 5),
            "Traditional knowledge is not an invention [9].")
        assert answer.insufficient_evidence
        assert answer.citations == ()
        for caution in answer.cautions:
            assert "cited source" not in caution.lower(), caution


def _with_status(item, status):
    """An `Evidence` with `temporal_status` replaced, whatever its class shape."""
    from dataclasses import replace

    return replace(item, temporal_status=status)
