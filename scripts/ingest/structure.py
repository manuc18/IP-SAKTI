"""Stage 3 — structural parsing.

Cleaned pages become structural units with resolvable locators. Which parser
runs is a property of the instrument, recorded in the document registry: Indian
statutory grammar for Acts, Rules, Regulations and the IPR Policy; Article and
paragraph grammar for TRIPS, the CBD and the Nagoya Protocol.

Table-of-contents, empty and Hindi-only pages are excluded here rather than
deleted; they remain in ``data/processed/cleaned`` for audit.
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.clean import detect_table  # noqa: E402
from ingest.common import (  # noqa: E402
    CLEANED_DIR,
    STRUCTURED_DIR,
    LogicalDocument,
    classify_language,
    logical_documents,
    read_jsonl,
    write_jsonl,
)
from ingest.parsers import statutory, treaty  # noqa: E402
from ingest.parsers.base import Line  # noqa: E402

#: Only these page kinds carry operative text into the structural layer.
INGESTIBLE_CONTENT_TYPES = {"content"}

PARSERS = {"statutory": statutory, "treaty": treaty}

#: A provision this long that never subdivides is not a provision — the opener
#: that started it was almost certainly a table row or a list item.
UNRELIABLE_UNIT_CHARS = 20_000

#: Below this length a unit is too small for the prose test to mean anything
#: (``5. [Omitted.]`` is a real provision and three words long).
PROSE_TEST_MIN_CHARS = 400

#: Statutory prose is dense in function words; an index or a table of names is
#: not. Measured across this corpus the two populations do not overlap: genuine
#: provisions score 0.19-0.55, the scanned FSSAI recipe index scores 0.00-0.01.
#: This test is independent of page layout, which a line-length test is not.
PROSE_MIN_FUNCTION_WORD_RATIO = 0.15
FUNCTION_WORDS = frozenset(
    """the of and or to in a an by for shall be as any such that with under this is on
    from which may not it are was were been his her their its other case where who have
    has if at no all""".split()
)

TOKEN_RE = re.compile(r"[A-Za-z]+")


def function_word_ratio(text: str) -> float:
    tokens = TOKEN_RE.findall(text.lower())
    if not tokens:
        return 0.0
    return sum(1 for token in tokens if token in FUNCTION_WORDS) / len(tokens)


def looks_like_prose(text: str) -> bool:
    return function_word_ratio(text) >= PROSE_MIN_FUNCTION_WORD_RATIO


def downgrade_unreliable(record: dict[str, Any], subdivision_count: int) -> None:
    """Withdraw a locator the evidence does not support.

    A fabricated citation is worse than an honest ``locator_unavailable``. Two
    signals withdraw one: a "provision" that is really a table row (tabular text,
    no marginal heading — the scanned FSSAI recipe index is entirely this), and a
    "provision" that runs for tens of thousands of characters without ever
    subdividing.
    """
    if record["unit_type"] not in ("provision", "schedule_item"):
        return
    # OCR cannot tell a printed provision number from a table row number: both
    # are just digits followed by a full stop. On pages the audit proved are
    # scans, a provision locator is only trusted when a marginal heading — which
    # a table row never has — corroborates it.
    ocr_without_heading = "ocr" in record["extraction_methods"] and not record["heading"]
    tabular_row = record["contains_table"] and not record["heading"]
    runaway = record["character_count"] > UNRELIABLE_UNIT_CHARS and subdivision_count < 2
    not_prose = (
        record["character_count"] >= PROSE_TEST_MIN_CHARS
        and not looks_like_prose(record["text"])
    )
    if not (ocr_without_heading or tabular_row or runaway or not_prose):
        return
    record["unit_type"] = "unstructured"
    record["locator_downgraded_from"] = record["number"]
    record["locator_downgrade_reason"] = (
        "OCR-derived text without a corroborating marginal heading"
        if ocr_without_heading
        else "tabular text without a marginal heading"
        if tabular_row
        else "provision exceeds the reliable length with no sub-provisions"
        if runaway
        else "text is not prose — function-word density below the statutory floor"
    )
    record["number"] = None
    # The container — Schedule, Form, Chapter — is not withdrawn: it was read
    # from its own heading and is still true. Only the entry number goes.
    for key in ("section", "rule", "regulation", "paragraph", "item"):
        record["locator"][key] = None
    record["structural_path"] = ""


def document_lines(
    pages: list[dict[str, Any]]
) -> tuple[list[Line], dict[int, list[str]], dict[int, str]]:
    lines: list[Line] = []
    notes_by_page: dict[int, list[str]] = {}
    method_by_page: dict[int, str] = {}
    for page in pages:
        if page["content_type"] not in INGESTIBLE_CONTENT_TYPES:
            continue
        method_by_page[page["physical_page"]] = page["extraction_method"]
        if page["amendment_notes"]:
            notes_by_page[page["physical_page"]] = page["amendment_notes"]
        for text in page["cleaned_text"].split("\n"):
            lines.append((page["physical_page"], text))
    return lines, notes_by_page, method_by_page


def structure_document(doc: LogicalDocument) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pages = read_jsonl(os.path.join(CLEANED_DIR, f"{doc.document_id}.jsonl"))
    lines, notes_by_page, method_by_page = document_lines(pages)
    module = PARSERS[doc.parser]
    units = module.parse(lines, doc.provision_label)

    records: list[dict[str, Any]] = []
    for order, unit in enumerate(units):
        pages_covered = unit.pages
        # A page's footnote apparatus annotates the text printed on that page.
        # Without a resolvable superscript anchor in the text layer, attaching a
        # note to every unit on its page is the faithful mapping; inventing a
        # single referent would not be.
        notes = [note for page in pages_covered for note in notes_by_page.get(page, [])]
        text = unit.text
        subdivisions = module.subdivisions(
            unit.lines,
            statutory.LABEL_TO_KEY.get(doc.provision_label, "section")
            if doc.parser == "statutory"
            else "article",
        )
        record = {
            "document_id": doc.document_id,
            "parent_source_document_id": doc.parent_source_document_id,
            "source_filename": doc.source_filename,
            "source_sha256": doc.source_sha256,
            "unit_order": order,
            "unit_type": unit.unit_type,
            "number": unit.number,
            "heading": unit.heading,
            "structural_path": unit.structural_path,
            "locator": unit.locator,
            "text": text,
            "character_count": len(text),
            "physical_pages": pages_covered,
            "amendment_notes": notes,
            "contains_table": detect_table(text),
            "language_status": classify_language(text),
            "extraction_methods": sorted(
                {method_by_page.get(page, "native") for page in pages_covered}
            ),
            "lines": [[page, value] for page, value in unit.lines],
            "subdivisions": [[index, kind, marker] for index, kind, marker in subdivisions],
        }
        downgrade_unreliable(record, len(subdivisions))
        records.append(record)

    report = {
        "document_id": doc.document_id,
        "parser": doc.parser,
        "provision_label": doc.provision_label,
        "pages_ingested": len({p for p, _ in lines}),
        "pages_excluded": {
            key: value
            for key, value in Counter(
                page["content_type"] for page in pages
                if page["content_type"] not in INGESTIBLE_CONTENT_TYPES
            ).items()
        },
        "units": len(records),
        "units_by_type": dict(Counter(r["unit_type"] for r in records)),
        "units_with_locator": sum(
            1 for r in records if any(v for v in r["locator"].values())
        ),
        "units_locator_downgraded": sum(1 for r in records if "locator_downgrade_reason" in r),
        "characters": sum(r["character_count"] for r in records),
    }
    return records, report


def run(verbose: bool = True) -> dict[str, Any]:
    os.makedirs(STRUCTURED_DIR, exist_ok=True)
    summary: dict[str, Any] = {"documents": {}}
    for doc in logical_documents():
        records, report = structure_document(doc)
        write_jsonl(os.path.join(STRUCTURED_DIR, f"{doc.document_id}.jsonl"), records)
        with open(
            os.path.join(STRUCTURED_DIR, f"{doc.document_id}.structure_report.json"),
            "w", encoding="utf-8",
        ) as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        summary["documents"][doc.document_id] = report
        if verbose:
            types = report["units_by_type"]
            print(
                f"structure {doc.document_id:30s} units={report['units']:4d} "
                f"located={report['units_with_locator']:4d} "
                f"{ {k: v for k, v in sorted(types.items())} }",
                flush=True,
            )
    return summary


if __name__ == "__main__":
    run()
