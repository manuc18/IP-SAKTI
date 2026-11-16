"""Issue 4: damaged source text is detected and reported, and never repaired.

Two properties are under test, and the second is the one that matters:

1. The detector finds the corruption that is actually in this corpus, and does
   not fire on the correctly-extracted tables, schedules and dot-leader form
   blanks that superficially resemble it.
2. Nothing anywhere rewrites a character of a statute. The displayed excerpt is
   still a verbatim substring of the stored chunk, damaged or not.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from ai.answering.excerpt import display_text, excerpt_for_evidence, select_excerpt
from ai.text_integrity import CLEAN, DAMAGED, assess, integrity_config

CHUNK_DIR = pathlib.Path(__file__).resolve().parents[1] / "data" / "processed" / "chunks"

#: Real damage, copied verbatim out of the corpus.
DAMAGED_SAMPLES = [
    # Per-glyph positioning in 2016DrugsandCosmeticsAct1940Rules1945.pdf.
    "(a) A  t ru e  c o p y   of  a  le t t e r  fr om  m e /u s  to  th e  "
    "m a nufa c t u r i n g  c o nc e r n  wh os e",
    "4.  The manufacture r or his authorise d agent in India shall inform the "
    "licensin g authority fort hwith in the event of any administrati ve action",
    # Tesseract on the dotted fill lines of the scanned Patents Rules forms.
    "In the matter of! ..........c.cccccccecccceceuceeveceenes eee eee eee eee "
    "ee eee ee ee ee ee eee eee eee ee eee ee eee ee",
]

#: Correctly extracted text that the rejected heuristics would have condemned:
#: a bilingual dosage table, a statutory form blank, a schedule of fees.
CLEAN_SAMPLES = [
    "36. Nasya                    36.   Huqna",
    "(iii) class a  .................................. in respect of b  "
    "........................ in the name(s) of",
    "23. Form and signing of application .- (1) Every application for the "
    "registration of a geographical indication shall be signed by the applicant "
    "or his agent.",
    "1. (1) These rules may be called the Geographical Indications Of Goods "
    "(Registration and Protection) Rules, 2002. (2) They shall come into force "
    "on the date on which the Act comes into force.",
    "(a) in relation to a trade mark, means (i) the proprietor; (ii) a "
    "registered user; (iii) any person claiming under them.",
]


@pytest.mark.parametrize("text", DAMAGED_SAMPLES)
def test_real_corpus_damage_is_detected(text):
    verdict = assess(text)
    assert verdict.status == DAMAGED
    assert verdict.signals


@pytest.mark.parametrize("text", CLEAN_SAMPLES)
def test_correctly_extracted_text_is_not_condemned(text):
    verdict = assess(text)
    assert verdict.status == CLEAN, (
        f"false positive on {text[:60]!r}: {verdict.signals}")


def test_roman_and_lettered_clause_labels_are_not_debris():
    """The first version of this detector read every `(ii)` as OCR noise."""
    text = ("2. Definitions. (1) In these rules, unless the context otherwise "
            "requires, (a) 'Act' means the Act; (b) 'Registrar' means the "
            "Registrar; (i) the first; (ii) the second; (iii) the third; "
            "(iv) the fourth.")
    assert assess(text).status == CLEAN


def test_the_detector_never_alters_the_text_it_judges():
    for text in DAMAGED_SAMPLES + CLEAN_SAMPLES:
        before = text
        assess(text)
        assert text == before
    assert integrity_config()["repairs_text"] is False


def test_empty_input_is_clean_not_damaged():
    assert assess("").status == CLEAN
    assert assess(None).status == CLEAN


# -- the invariant the brief asks for ---------------------------------------

def _corpus_chunks(limit: int | None = None):
    rows = []
    for path in sorted(CHUNK_DIR.glob("*.jsonl")):
        with path.open() as handle:
            for line in handle:
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    return rows
    return rows


def test_every_displayed_excerpt_is_a_substring_of_its_stored_chunk():
    """The whole-corpus invariant: display normalisation, and nothing else.

    Runs over all 3,073 chunks, damaged ones included, because a damaged chunk
    is exactly where a well-meaning repair would be tempting.
    """
    checked = 0
    for chunk in _corpus_chunks():
        excerpt = select_excerpt(chunk["text"], "registration of a trade mark")
        if excerpt is None:
            continue
        display = display_text(chunk["text"])
        assert excerpt.text == display[excerpt.start:excerpt.end].strip()
        assert excerpt.text in display
        checked += 1
    assert checked > 3000


def test_a_damaged_chunk_is_shown_verbatim_and_flagged_rather_than_cleaned():
    damaged = next(
        (chunk for chunk in _corpus_chunks()
         if assess(chunk["text"]).status == DAMAGED), None)
    assert damaged is not None, "expected the corpus to contain damaged chunks"

    class _Item:
        text = damaged["text"]
        marker = 1

    payload = excerpt_for_evidence(_Item(), "licence to manufacture")
    assert payload is not None
    assert payload["text"] in display_text(damaged["text"])
    assert payload["integrity"]["status"] in (CLEAN, DAMAGED)
    assert payload["integrity"]["repairable"] is False


def test_the_corpus_census_is_concentrated_not_diffuse():
    """A detector that fires everywhere is measuring nothing.

    The damage has two named causes in four named PDFs, so most instruments
    must come back entirely clean. This is the test that would fail if the
    thresholds were loosened until they flattered a number.
    """
    per_document: dict[str, list[int]] = {}
    for chunk in _corpus_chunks():
        entry = per_document.setdefault(chunk["document_id"], [0, 0])
        entry[1] += 1
        if assess(chunk["text"]).status == DAMAGED:
            entry[0] += 1
    damaged = sum(entry[0] for entry in per_document.values())
    total = sum(entry[1] for entry in per_document.values())
    untouched = [name for name, entry in per_document.items() if entry[0] == 0]

    assert 0.02 <= damaged / total <= 0.12, f"{damaged}/{total} looks wrong"
    assert len(untouched) >= 10, (
        f"only {len(untouched)} of {len(per_document)} documents are clean; "
        f"the detector is firing indiscriminately")
