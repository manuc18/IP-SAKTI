"""Phase 7.3 Part A — an ellipsis is punctuation inside a sentence, not a full stop.

The bug these guard against was found in the first live Groq run, not in review.
The model wrote a faithful, correctly cited answer ending
``…does not result in an enhanced... [1]``. `split_sentences` read the ellipsis
as a sentence boundary, `[1]` became a sentence of its own, and the claim it was
supporting was reported as uncited — so a good answer was refused by the
grounding gate for a punctuation artefact.

The invariant under test is therefore not "sentences are split correctly" but
the narrower thing that actually matters here:

    A citation marker belonging to a claim stays attached to that claim.

`…` and `e.g.`/`i.e.` were already safe before the fix — the first because the
character is not in the terminator class, the second because `_ABBREVIATIONS`
covers it. They are tested anyway: this file is the record of which inputs must
keep working, and a case that passes for a reason nobody wrote down is a case
that regresses silently.
"""
from __future__ import annotations

from ai.answering.grounding import check_grounding, split_sentences
from ai.answering.models import Evidence, EvidenceSet


def _evidence_set() -> EvidenceSet:
    """One passage, marker [1], whose terms overlap the claims below."""
    return EvidenceSet((Evidence(
        marker=1, chunk_id="DOC::unit-1", document_id="DOC",
        document_name="Test Act, 2000", citation="Test Act, 2000 — Section 3",
        text="The mere discovery of a new form of a known substance which does "
             "not result in the enhancement of the known efficacy of that "
             "substance is not an invention.",
        citable=True, locator_unavailable=False, locator_fields={"section": "3"},
        temporal_status="CURRENT", source_evidence_level="A",
        physical_pages=(7,), source_filename="test.pdf", source_sha256="abc123",
        official_source_url="https://example.invalid/test.pdf",
    ),))


# 1 — ASCII ellipsis

def test_ascii_ellipsis_does_not_split_the_marker_away():
    assert split_sentences("Text ... [1]") == ["Text ... [1]"]


# 2 — Unicode ellipsis

def test_unicode_ellipsis_does_not_split_the_marker_away():
    assert split_sentences("Text … [1]") == ["Text … [1]"]


# 3 — ellipsis closing a truncated quotation, the shape that actually failed

def test_ellipsis_after_a_word_keeps_the_marker():
    assert split_sentences("Text says something... [1]") == [
        "Text says something... [1]"]


def test_the_exact_sentence_from_the_first_live_run_is_one_claim():
    text = ("The mere discovery of a new form of a known substance that does "
            "not result in an enhanced... [1]")
    assert split_sentences(text) == [text]


# 4 — a real sentence boundary must still be a boundary

def test_normal_sentence_boundaries_still_split():
    assert split_sentences("One. Two. Three.") == ["One.", "Two.", "Three."]


def test_marker_between_two_claims_still_splits_them():
    """The fix must not merge a document into a single sentence.

    `[1]` is written *after* the full stop here, so it lands with the second
    sentence rather than the first. That is a marker-placement problem and not
    an ellipsis one — it is recorded here rather than fixed, so that a later
    change to marker attachment has to confront this expectation deliberately.
    """
    assert split_sentences("First claim. [1] Second claim. [2]") == [
        "First claim.", "[1] Second claim.", "[2]"]


# 5 and 6 — abbreviations ending in a period

def test_eg_does_not_detach_the_marker():
    assert split_sentences("The Act excludes some subject matter, e.g. [1]") == [
        "The Act excludes some subject matter, e.g. [1]"]


def test_ie_does_not_detach_the_marker():
    assert split_sentences("The Act excludes some subject matter, i.e. [1]") == [
        "The Act excludes some subject matter, i.e. [1]"]


# 7 — the behaviour that matters: the grounding gate accepts the fixed prose

def test_ellipsis_claim_is_grounded_not_reported_as_uncited():
    text = ("The mere discovery of a new form of a known substance that does "
            "not result in an enhanced efficacy... [1]")
    report = check_grounding(text, _evidence_set())
    assert report.unsupported_claims == ()
    assert report.unresolved_markers == ()
    assert report.grounded


def test_an_uncited_claim_is_still_caught():
    """The fix must not make the gate permissive: it removed one false
    positive, and a gate that stopped reporting true positives would be a
    worse outcome than the bug."""
    text = ("Section 3 defines what is not an invention. The mere discovery of "
            "a new form of a known substance is excluded... [1]")
    report = check_grounding(text, _evidence_set())
    assert report.unsupported_claims == (
        "Section 3 defines what is not an invention.",)
    assert not report.grounded


def test_unresolved_marker_is_still_caught():
    report = check_grounding("A claim about a known substance... [4]",
                             _evidence_set())
    assert report.unresolved_markers == (4,)
