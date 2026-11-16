"""The synthetic benchmark's single-label defect, and the rules that correct it.

Every test names a real question from the 1,000-question set and the real chunks
involved, because the rules are only defensible if they widen the cases they were
written for and leave everything else alone. The last two tests are the ones that
would catch a rule quietly becoming generous enough to flatter the benchmark.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "scripts"))

from ai.retrieval.base import load_chunks  # noqa: E402
from evaluation.target_equivalence import (  # noqa: E402
    EquivalenceIndex,
    equivalence_config,
    normalise_heading,
)


@pytest.fixture(scope="module")
def index() -> EquivalenceIndex:
    return EquivalenceIndex.build(load_chunks())


def test_the_index_actually_read_the_corpus(index):
    """The null-result guard.

    A first version of the adapter read every field off `metadata`, which
    `Chunk` does not have, so the index came back empty and the adjustment
    widened nothing — a bug that presents as a clean finding.
    """
    assert len(index.chunk) == 3073
    assert len(index.by_heading) > 1000
    assert len(index.by_locator) > 1500
    assert len(index.aliases) == 22


def test_two_halves_of_one_section_are_the_same_provision(index):
    """R1. The chunker split section 25; the question was written about it."""
    accepted, counts = index.acceptable(
        "How long do I get for opposition to the patent under the Patents Act, 1970?",
        ["IN-PATENTS-ACT-1970::chapter-v-section-25-c1"])
    assert "IN-PATENTS-ACT-1970::chapter-v-section-25-c2" in accepted
    assert counts["R1_same_provision"] == 1


def test_an_unnamed_instrument_makes_a_shared_heading_acceptable(index):
    """R2. "Security for costs" heads provisions in three instruments."""
    accepted, counts = index.acceptable(
        "May the authority act of its own motion in respect of security for costs?",
        ["IN-TM-RULES-2017::part-i-chapter-ii-rule-51"])
    assert "IN-PATENTS-ACT-1970::chapter-xxiii-section-150" in accepted
    assert counts["R2_same_heading"] >= 1


def test_naming_the_instrument_closes_the_shared_heading_rule(index):
    """R2's guard. If the question says which Act, the other Act is a miss."""
    accepted, counts = index.acceptable(
        "May the authority act of its own motion in respect of security for "
        "costs under the Trade Marks Rules, 2017?",
        ["IN-TM-RULES-2017::part-i-chapter-ii-rule-51"])
    assert counts["R2_same_heading"] == 0
    assert "IN-PATENTS-ACT-1970::chapter-xxiii-section-150" not in accepted


def test_a_provision_the_question_names_is_acceptable(index):
    """R3. 32 PPVFR-Rules questions name a section of the parent Act."""
    accepted, counts = index.acceptable(
        "What fee is payable for conducting tests under section 19 under the "
        "PPVFR Rules, 2003?",
        ["IN-PPVFR-RULES-2003::chapter-iii-rule-28"])
    assert "IN-PPVFR-ACT-2001::chapter-iii-section-19" in accepted
    assert counts["R3_named_provision"] == 1


def test_a_named_provision_does_not_reach_across_regimes(index):
    """R3's scope guard, and the reason it exists.

    Unscoped, "section 19" accepted section 19 of all eight Acts that have one,
    which would turn every identifier question into a free pass.
    """
    accepted, _ = index.acceptable(
        "What fee is payable for conducting tests under section 19 under the "
        "PPVFR Rules, 2003?",
        ["IN-PPVFR-RULES-2003::chapter-iii-rule-28"])
    for stranger in ("IN-COPYRIGHT-ACT-1957::chapter-iv-section-19",
                     "IN-PATENTS-ACT-1970::chapter-iv-section-19",
                     "IN-DC-ACT-1940::chapter-iv-section-19"):
        assert stranger not in accepted


def test_a_well_specified_question_is_not_widened_at_all(index):
    """The property that keeps the adjustment from being a free pass."""
    accepted, counts = index.acceptable(
        "How must an application for registration of a trade mark be signed "
        "under the Trade Marks Rules, 2017?",
        ["IN-TM-RULES-2017::part-i-chapter-ii-rule-23"])
    assert accepted == {"IN-TM-RULES-2017::part-i-chapter-ii-rule-23"}
    assert sum(counts.values()) == 0


def test_the_labels_are_always_acceptable(index):
    for target in ("IN-TM-RULES-2017::part-i-chapter-ii-rule-23",
                   "IN-GI-ACT-1999::chapter-ii-section-3",
                   "UN-CBD-1992::article-15"):
        accepted, _ = index.acceptable("what does this provide?", [target])
        assert target in accepted


def test_the_adjustment_stays_bounded_across_the_whole_benchmark(index):
    """The test that fails if a rule is loosened until the number looks good.

    Widening is meant to repair a labelling defect in a minority of questions,
    not to accept most of the corpus for most of them. If either bound trips,
    the adjusted figure has stopped measuring retrieval.
    """
    import json
    import pathlib

    path = (pathlib.Path(__file__).resolve().parents[1]
            / "data" / "evaluation" / "synthetic_1000.jsonl")
    if not path.exists():
        pytest.skip("synthetic benchmark not generated")

    sizes = []
    widened = 0
    total = 0
    for line in path.open():
        record = json.loads(line)
        if record["label_kind"] != "ANSWERABLE":
            continue
        total += 1
        accepted, _ = index.acceptable(record["question"],
                                       record["target_chunk_ids"])
        sizes.append(len(accepted))
        if len(accepted) > len(record["target_chunk_ids"]):
            widened += 1

    assert total == 960
    assert widened / total < 0.40, (
        f"{widened}/{total} questions widened; the rules have stopped being "
        f"corrections and become a discount")

    # Most questions must not be widened at all. A rule set that touches the
    # majority is not correcting a labelling defect, it is lowering the bar.
    assert sizes.count(1) / total >= 0.60, (
        f"only {sizes.count(1)}/{total} questions are left with a single "
        f"acceptable chunk")

    # And the widest set must still be one provision. Twenty is the size of the
    # largest legitimate unit in this corpus: a whole numbered Section of TRIPS,
    # which contains its Articles, and Rule 85 of the Drugs and Cosmetics Rules,
    # which the chunker split into ten parts. Anything beyond that is a rule
    # reaching across provisions, which is the failure this bound exists to
    # catch — it caught two of them while these rules were being written.
    assert max(sizes) <= 20, f"one question accepted {max(sizes)} chunks"


def test_heading_normalisation_folds_form_not_wording():
    assert normalise_heading("Form, etc; of Affidavits") == "form etc of affidavits"
    assert normalise_heading("Form etc. of affidavits") == "form etc of affidavits"
    assert normalise_heading("Sale at more than one place") != \
        normalise_heading("Sale at one place")
    assert normalise_heading(None) == ""


def test_the_config_says_where_the_rules_come_from():
    config = equivalence_config()
    assert set(config["rules"]) == {"R1_same_provision", "R2_same_heading",
                                    "R3_named_provision"}
    assert "never from the run being scored" in config["derived_from"]
