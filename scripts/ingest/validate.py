"""Stage 5 — the validation gate.

Nothing downstream should trust these chunks until this passes. The checks are
split into two classes:

* **Fatal** — a defect that would put a wrong or unciteable statement in front of
  a user: missing provenance, a source PDF that no longer matches its audited
  digest, an empty chunk, mojibake, corrupt Devanagari leaking into the English
  retrieval text, a page that cannot be traced back to its PDF, or an invalid
  controlled-vocabulary value.
* **Advisory** — a defect worth reporting but not worth blocking on: chunks
  without a locator, duplicates, table-bearing chunks, boilerplate residue.

``VALIDATION_STATUS: PASS`` is only ever emitted when the fatal list is empty.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.common import (  # noqa: E402
    CHUNKS_DIR,
    CLEANED_DIR,
    DOCS,
    EXTRACTED_DIR,
    LANG_EN_CLEAN,
    STRUCTURED_DIR,
    DEVANAGARI,
    hindi_corruption_hits,
    legacy_font_hits,
    logical_documents,
    read_jsonl,
    sha256_file,
)
from ingest.parsers.statutory import LABEL_TO_KEY  # noqa: E402
from ingest.provenance import EVIDENCE_LEVELS  # noqa: E402

VALID_JURISDICTIONS = {"IN", "international"}
VALID_DOMAINS = {
    "biodiversity_abs", "copyright", "designs", "drug_regulation",
    "food_safety_ayurveda", "geographical_indications", "international_ip",
    "ip_policy", "patents", "plant_varieties", "trademarks",
}
VALID_DOCUMENT_TYPES = {
    "act", "act_and_rules_compendium", "policy", "regulations",
    "regulations_with_covering_order", "rules", "treaty",
}
#: The unit kinds that must carry a legal locator. Front matter and text whose
#: locator was withdrawn on evidence are exempt — that is the honest outcome.
LOCATOR_REQUIRED_UNIT_TYPES = {"provision", "article"}

#: Locator keys that name a container rather than a provision. A container is
#: read from its own heading ("THE SCHEDULE", "FORM 4"), so it is always its own
#: evidence.
CONTAINER_KEYS = ("schedule", "form", "annexure")
#: Locator keys that name a position *inside* a provision. One of these is only
#: a citation when the provision it sits inside is also known.
DEPENDENT_KEYS = ("subsection", "sub_rule", "clause", "paragraph", "subparagraph")


def provision_key_for(doc: Any) -> str:
    """The locator key this instrument's own provisions are numbered under."""
    if doc.parser == "treaty":
        return "article"
    return LABEL_TO_KEY.get(doc.provision_label, "section")


def unsupported_locator(chunk: dict[str, Any], doc: Any) -> str | None:
    """Why this chunk's locator is not supported by structure, or ``None``.

    Phase 5 §3. Every citation IP-SAKTI prints has to be checkable by a reader
    holding the PDF, and the failure mode this catches is the one Phase 4 found
    in the CBD: a numbered line inside unlocated running text became "CBD —
    Paragraph 3", a provision that does not exist. A locator naming a position
    inside a provision is evidence of nothing unless the provision is known too.
    """
    locator = chunk["locator"]
    containers = [key for key in CONTAINER_KEYS if locator.get(key)]
    provision_key = provision_key_for(doc)

    if locator.get("item") and not containers:
        return "item number with no Schedule, Form or Annexure to number within"
    if locator.get(provision_key) or containers:
        return None
    dependent = [key for key in DEPENDENT_KEYS
                 if key != provision_key and locator.get(key)]
    if dependent:
        return (f"{'/'.join(dependent)} with no {provision_key} to sit inside — "
                f"a sub-unit locator is not a citation on its own")
    return None


def locator_defects(chunk: dict[str, Any], doc: Any) -> list[tuple[str, str]]:
    """Phase 5 §3 — the three ways a locator can be wrong rather than absent.

    A wrong locator is more dangerous than a missing one. The Phase 3 invariant
    only guards chunks that admit they have no locator, so a confidently
    incorrect citation walks straight past it and reaches a reader as a checkable
    promise that does not check out.
    """
    found: list[tuple[str, str]] = []
    locator = chunk.get("locator") or {}
    containers = [key for key in CONTAINER_KEYS if locator.get(key)]
    provision_key = provision_key_for(doc)

    if doc.parser == "statutory" and containers and locator.get(provision_key):
        found.append((
            "schedule_masquerading_as_provision",
            f"{'/'.join(containers)} entry also carries {provision_key} "
            f"{locator[provision_key]}; a Schedule entry is not a {provision_key} "
            f"of the parent instrument"))

    if (doc.parser == "treaty"
            and chunk.get("unit_type") in ("preamble", "front_matter")
            and any(locator.get(key) for key in ("article", *DEPENDENT_KEYS))):
        found.append((
            "treaty_preamble_locator",
            "preamble text carries an Article or paragraph locator; treaty "
            "Articles must be parsed as Articles, never numbered off the preamble"))

    unsupported = unsupported_locator(chunk, doc)
    if unsupported and chunk.get("citable"):
        found.append(("unsupported_locator", unsupported))
    return found


CID_RE = re.compile(r"\(cid:\d+\)")
REPLACEMENT_CHAR = "�"
TOC_ENTRY_RE = re.compile(r"^\s*\d{1,3}[A-Z]{0,4}\s*\.\s+\S.{0,90}\.\s*$")

MIN_CHUNK_CHARS = 40


def normalise_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def toc_contamination(text: str) -> bool:
    """Does this chunk read like a contents listing rather than a provision?"""
    lines = [line for line in text.split("\n") if line.strip()]
    if len(lines) < 6:
        return False
    entries = sum(1 for line in lines if TOC_ENTRY_RE.match(line))
    return entries / len(lines) >= 0.60


def run(verbose: bool = True) -> dict[str, Any]:
    documents = logical_documents()
    fatal: list[dict[str, Any]] = []
    advisory: list[dict[str, Any]] = []
    per_document: dict[str, Any] = {}

    # Provenance: the source PDFs must still hash to the digests the Phase 1
    # audit recorded. Everything downstream is addressed by those digests.
    digest_cache: dict[str, str] = {}
    for doc in documents:
        if doc.source_filename in digest_cache:
            continue
        if not os.path.exists(doc.source_path):
            fatal.append({"check": "source_present", "document_id": doc.document_id,
                          "detail": f"missing source PDF {doc.source_path}"})
            digest_cache[doc.source_filename] = ""
            continue
        digest_cache[doc.source_filename] = sha256_file(doc.source_path)
        if digest_cache[doc.source_filename] != doc.source_sha256:
            fatal.append({"check": "source_sha256", "document_id": doc.document_id,
                          "detail": f"{doc.source_filename} no longer matches its audited digest"})

    totals = Counter()
    normalised_index: dict[str, list[str]] = defaultdict(list)
    boilerplate_hits = 0

    for doc in documents:
        cleaned = read_jsonl(os.path.join(CLEANED_DIR, f"{doc.document_id}.jsonl"))
        chunks = read_jsonl(os.path.join(CHUNKS_DIR, f"{doc.document_id}.jsonl"))
        structured = read_jsonl(os.path.join(STRUCTURED_DIR, f"{doc.document_id}.jsonl"))
        with open(os.path.join(CHUNKS_DIR, f"{doc.document_id}.chunk_report.json"),
                  encoding="utf-8") as fh:
            chunk_report = json.load(fh)
        with open(os.path.join(CLEANED_DIR, f"{doc.document_id}.clean_report.json"),
                  encoding="utf-8") as fh:
            clean_report = json.load(fh)

        known_pages = {page["physical_page"] for page in cleaned}
        boilerplate = set(clean_report["boilerplate_patterns"])
        doc_fatal_before = len(fatal)

        for chunk in chunks:
            cid = chunk.get("chunk_id", "<no id>")

            for field in ("chunk_id", "document_id", "source_sha256", "text"):
                if not chunk.get(field):
                    fatal.append({"check": "provenance_fields", "document_id": doc.document_id,
                                  "chunk_id": cid, "detail": f"missing {field}"})
            if chunk.get("source_sha256") != doc.source_sha256:
                fatal.append({"check": "provenance_fields", "document_id": doc.document_id,
                              "chunk_id": cid, "detail": "source_sha256 does not match the registry"})
            if not chunk.get("physical_pages"):
                fatal.append({"check": "provenance_pages", "document_id": doc.document_id,
                              "chunk_id": cid, "detail": "no physical page recorded"})
            if not chunk.get("locator_unavailable") and not any(chunk["locator"].values()):
                fatal.append({"check": "locator_declared", "document_id": doc.document_id,
                              "chunk_id": cid,
                              "detail": "empty locator not declared as locator_unavailable"})

            # Phase 3 §9 — hard safety invariant. A chunk whose locator could not
            # be established stays retrievable but must never be citable, because
            # a citation is a promise the reader can verify against the source.
            if "citable" not in chunk or "retrievable" not in chunk:
                fatal.append({"check": "citability_declared", "document_id": doc.document_id,
                              "chunk_id": cid, "detail": "citable/retrievable not declared"})
            elif chunk["locator_unavailable"] and chunk["citable"]:
                fatal.append({"check": "citability_invariant", "document_id": doc.document_id,
                              "chunk_id": cid,
                              "detail": "locator_unavailable chunk is marked citable"})
            elif not chunk["locator_unavailable"] and not chunk["citable"]:
                fatal.append({"check": "citability_invariant", "document_id": doc.document_id,
                              "chunk_id": cid,
                              "detail": "located chunk is marked not citable"})
            # Phase 5 §3 — a locator that is wrong rather than absent.
            for check, detail in locator_defects(chunk, doc):
                fatal.append({"check": check, "document_id": doc.document_id,
                              "chunk_id": cid, "detail": detail})

            if chunk.get("source_evidence_level") not in EVIDENCE_LEVELS:
                fatal.append({"check": "evidence_level", "document_id": doc.document_id,
                              "chunk_id": cid,
                              "detail": str(chunk.get("source_evidence_level"))})

            text = chunk.get("text", "")
            if len(text.strip()) < MIN_CHUNK_CHARS:
                fatal.append({"check": "empty_chunk", "document_id": doc.document_id,
                              "chunk_id": cid, "detail": f"{len(text.strip())} characters"})
            if CID_RE.search(text):
                fatal.append({"check": "cid_artefacts", "document_id": doc.document_id,
                              "chunk_id": cid, "detail": CID_RE.search(text).group(0)})
            if REPLACEMENT_CHAR in text:
                fatal.append({"check": "replacement_char", "document_id": doc.document_id,
                              "chunk_id": cid, "detail": "U+FFFD present"})
            devanagari = len(DEVANAGARI.findall(text))
            if devanagari:
                fatal.append({"check": "devanagari_in_english_text",
                              "document_id": doc.document_id, "chunk_id": cid,
                              "detail": f"{devanagari} Devanagari characters"})
            if hindi_corruption_hits(text) or legacy_font_hits(text):
                fatal.append({"check": "corrupt_hindi", "document_id": doc.document_id,
                              "chunk_id": cid, "detail": "known corrupt-Hindi marker present"})
            if chunk.get("language") != "en" or chunk.get("language_status") != LANG_EN_CLEAN:
                fatal.append({"check": "language", "document_id": doc.document_id,
                              "chunk_id": cid,
                              "detail": f"language={chunk.get('language')} "
                                        f"status={chunk.get('language_status')}"})

            if chunk.get("jurisdiction") not in VALID_JURISDICTIONS:
                fatal.append({"check": "jurisdiction", "document_id": doc.document_id,
                              "chunk_id": cid, "detail": str(chunk.get("jurisdiction"))})
            if chunk.get("domain") not in VALID_DOMAINS:
                fatal.append({"check": "domain", "document_id": doc.document_id,
                              "chunk_id": cid, "detail": str(chunk.get("domain"))})
            if chunk.get("document_type") not in VALID_DOCUMENT_TYPES:
                fatal.append({"check": "document_type", "document_id": doc.document_id,
                              "chunk_id": cid, "detail": str(chunk.get("document_type"))})

            for page in chunk.get("physical_pages", []):
                if not (doc.page_start <= page <= doc.page_end) or page not in known_pages:
                    fatal.append({"check": "traceability", "document_id": doc.document_id,
                                  "chunk_id": cid,
                                  "detail": f"page {page} is outside the document's pages"})

            if chunk["unit_type"] in LOCATOR_REQUIRED_UNIT_TYPES and chunk["locator_unavailable"]:
                advisory.append({"check": "missing_expected_locator",
                                 "document_id": doc.document_id, "chunk_id": cid,
                                 "detail": f"{chunk['unit_type']} without a locator"})
            if toc_contamination(text):
                advisory.append({"check": "toc_contamination", "document_id": doc.document_id,
                                 "chunk_id": cid, "detail": "reads as a contents listing"})
            for line in text.split("\n"):
                stripped = re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip()))
                if stripped and stripped in boilerplate:
                    boilerplate_hits += 1
                    advisory.append({"check": "boilerplate_residue",
                                     "document_id": doc.document_id, "chunk_id": cid,
                                     "detail": stripped[:80]})
                    break

            normalised_index[hashlib.sha256(
                normalise_text(text).encode("utf-8")
            ).hexdigest()].append(cid)

        pages_ocr = sum(1 for page in cleaned if page["extraction_method"] == "ocr")
        per_document[doc.document_id] = {
            "pages": len(cleaned),
            "pages_native": len(cleaned) - pages_ocr,
            "pages_ocr": pages_ocr,
            "pages_excluded": {
                key: value for key, value in
                Counter(p["content_type"] for p in cleaned).items() if key != "content"
            },
            "units": len(structured),
            "chunks": len(chunks),
            "rejected": chunk_report["rejected"],
            "chunks_missing_locator": chunk_report["chunks_missing_locator"],
            "chunks_with_tables": chunk_report["chunks_with_tables"],
            "chunks_with_amendment_notes": chunk_report["chunks_with_amendment_notes"],
            "chunks_from_ocr": chunk_report["chunks_from_ocr"],
            "hindi_lines_excluded": clean_report["hindi_lines_removed"],
            "hindi_pages_excluded": clean_report["pages_by_content_type"].get("hindi_only", 0),
            "fatal_findings": len(fatal) - doc_fatal_before,
        }
        totals["pages"] += len(cleaned)
        totals["pages_ocr"] += pages_ocr
        totals["chunks"] += len(chunks)
        totals["rejected"] += chunk_report["rejected"]
        totals["units"] += len(structured)
        totals["missing_locator"] += chunk_report["chunks_missing_locator"]
        totals["with_tables"] += chunk_report["chunks_with_tables"]
        totals["with_amendment_notes"] += chunk_report["chunks_with_amendment_notes"]
        totals["from_ocr"] += chunk_report["chunks_from_ocr"]
        totals["hindi_lines_excluded"] += clean_report["hindi_lines_removed"]

    # Locator withdrawals and OCR confidence are ingestion facts the report must
    # carry: they are the difference between "we have this text" and "we can cite
    # this text".
    downgrades: dict[str, int] = {}
    for doc in documents:
        path = os.path.join(STRUCTURED_DIR, f"{doc.document_id}.structure_report.json")
        with open(path, encoding="utf-8") as fh:
            count = json.load(fh).get("units_locator_downgraded", 0)
        if count:
            downgrades[doc.document_id] = count

    ocr_confidence: dict[str, dict[str, Any]] = {}
    for doc in documents:
        path = os.path.join(EXTRACTED_DIR, f"{doc.document_id}.ocr.jsonl")
        if not os.path.exists(path):
            continue
        records = read_jsonl(path)
        scores = [r["ocr_mean_confidence"] for r in records
                  if r.get("ocr_mean_confidence") is not None]
        ocr_confidence[doc.document_id] = {
            "pages": len(records),
            "lang": records[0]["ocr_lang"] if records else None,
            "mean_confidence": round(sum(scores) / len(scores), 2) if scores else None,
            "pages_with_no_recognised_text": len(records) - len(scores),
        }

    duplicates = {digest: ids for digest, ids in normalised_index.items() if len(ids) > 1}
    duplicate_chunks = sum(len(ids) for ids in duplicates.values())

    result = {
        "status": "PASS" if not fatal else "FAIL",
        "documents": len(documents),
        "totals": dict(totals),
        "duplicate_groups": len(duplicates),
        "duplicate_chunks": duplicate_chunks,
        "duplicate_examples": [ids for ids in list(duplicates.values())[:10]],
        "boilerplate_residue_chunks": boilerplate_hits,
        "locator_downgrades": downgrades,
        "ocr_confidence": ocr_confidence,
        "fatal": fatal,
        "advisory": advisory,
        "per_document": per_document,
    }
    if verbose:
        print(f"validate: {result['status']}  chunks={totals['chunks']} "
              f"fatal={len(fatal)} advisory={len(advisory)} duplicates={duplicate_chunks}")
    return result


def write_report(result: dict[str, Any]) -> str:
    os.makedirs(DOCS, exist_ok=True)
    path = os.path.join(DOCS, "INGESTION_VALIDATION.md")
    totals = result["totals"]
    advisory_counts = Counter(item["check"] for item in result["advisory"])
    fatal_counts = Counter(item["check"] for item in result["fatal"])

    lines = [
        "# IP-SAKTI — Phase 2 ingestion validation",
        "",
        f"Documents processed: **{result['documents']}** logical documents "
        f"from 21 source PDFs.",
        "",
        "## Totals",
        "",
        "| Measure | Count |",
        "| --- | ---: |",
        f"| Pages processed | {totals.get('pages', 0)} |",
        f"| Pages extracted natively | {totals.get('pages', 0) - totals.get('pages_ocr', 0)} |",
        f"| Pages OCR'd | {totals.get('pages_ocr', 0)} |",
        f"| Structural units parsed | {totals.get('units', 0)} |",
        f"| Chunks generated | {totals.get('chunks', 0)} |",
        f"| Chunks rejected | {totals.get('rejected', 0)} |",
        f"| Chunks missing a locator | {totals.get('missing_locator', 0)} |",
        f"| Chunks containing tables | {totals.get('with_tables', 0)} |",
        f"| Chunks carrying amendment notes | {totals.get('with_amendment_notes', 0)} |",
        f"| Chunks derived from OCR | {totals.get('from_ocr', 0)} |",
        f"| Hindi lines excluded from retrieval text | {totals.get('hindi_lines_excluded', 0)} |",
        f"| Duplicate chunk groups | {result['duplicate_groups']} |",
        f"| Chunks in a duplicate group | {result['duplicate_chunks']} |",
        f"| Chunks with provenance issues | {fatal_counts.get('provenance_fields', 0) + fatal_counts.get('provenance_pages', 0)} |",
        "",
        "## Validation failures",
        "",
    ]
    if result["fatal"]:
        lines += ["| Check | Document | Chunk | Detail |", "| --- | --- | --- | --- |"]
        for item in result["fatal"][:100]:
            lines.append(
                f"| {item['check']} | {item.get('document_id', '')} | "
                f"{item.get('chunk_id', '')} | {item.get('detail', '')} |"
            )
        if len(result["fatal"]) > 100:
            lines.append(f"| … | | | {len(result['fatal']) - 100} further findings |")
    else:
        lines.append("None. Every fatal check passed:")
        lines += [
            "",
            "* every chunk carries `document_id`, `source_sha256` and at least one physical page;",
            "* every source PDF still hashes to the digest recorded in the Phase 1 audit;",
            "* no empty chunks, no `(cid:N)` artefacts, no U+FFFD;",
            "* no Devanagari and no known corrupt-Hindi marker in any retrieval chunk;",
            "* every chunk resolves chunk → page → source PDF;",
            "* jurisdiction, domain and document type are all in the controlled vocabulary.",
        ]

    lines += ["", "## OCR", ""]
    if result["ocr_confidence"]:
        lines += ["| Document | Pages | Language | Mean confidence | Pages with no recognised text |",
                  "| --- | ---: | --- | ---: | ---: |"]
        for document_id, stats in result["ocr_confidence"].items():
            lines.append(
                f"| `{document_id}` | {stats['pages']} | {stats['lang']} | "
                f"{stats['mean_confidence']} | {stats['pages_with_no_recognised_text']} |"
            )
        lines += [
            "",
            "Rendered at 300 DPI. OCR output is stored in `*.ocr.jsonl` beside the "
            "native extraction and never overwrites it; the native record survives "
            "under `native_extraction` in the cleaned pages.",
        ]
    else:
        lines.append("No OCR output present.")

    lines += ["", "## Withdrawn locators", ""]
    if result["locator_downgrades"]:
        lines += [
            "A provision number that the evidence does not support is withdrawn "
            "rather than published: a fabricated citation is worse than an honest "
            "`locator_unavailable`. Three signals withdraw one — OCR-derived text "
            "with no corroborating marginal heading, tabular text with no heading, "
            "and text whose function-word density falls below the statutory floor "
            "(an index or a list of names, not prose).",
            "",
            "| Document | Units withdrawn |",
            "| --- | ---: |",
        ]
        for document_id, count in sorted(result["locator_downgrades"].items()):
            lines.append(f"| `{document_id}` | {count} |")
    else:
        lines.append("None.")

    lines += ["", "## Advisory findings", ""]
    if advisory_counts:
        lines += ["| Check | Count |", "| --- | ---: |"]
        for check, count in advisory_counts.most_common():
            lines.append(f"| {check} | {count} |")
    else:
        lines.append("None.")

    lines += [
        "",
        "## Per document",
        "",
        "| Document | Pages | OCR pages | Units | Chunks | Rejected | No locator | Tables | Amendment notes | Hindi lines dropped |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for document_id, stats in result["per_document"].items():
        lines.append(
            f"| `{document_id}` | {stats['pages']} | {stats['pages_ocr']} | {stats['units']} | "
            f"{stats['chunks']} | {stats['rejected']} | {stats['chunks_missing_locator']} | "
            f"{stats['chunks_with_tables']} | {stats['chunks_with_amendment_notes']} | "
            f"{stats['hindi_lines_excluded']} |"
        )

    if result["duplicate_groups"]:
        lines += [
            "",
            "## Duplicate chunks",
            "",
            "Reported, not deleted. Identical normalised text most often means a "
            "provision reprinted in a Schedule, or an omitted-section marker that "
            "reads identically wherever it appears.",
            "",
        ]
        for ids in result["duplicate_examples"]:
            lines.append(f"* {', '.join(f'`{i}`' for i in ids)}")

    lines += [
        "",
        "## Unresolved extraction issues",
        "",
        "### The Ayurveda Aahara Regulations are not in the Ayurveda Aahara PDF",
        "",
        "OCR of all 170 scanned pages (95,882 characters at 89.6% mean confidence) "
        "returns no regulatory prose whatsoever: zero occurrences of *short title*, "
        "*commencement*, *definitions*, *these regulations*, *Schedule A/B* or "
        "*licence*, and one occurrence of *shall* in the whole enclosure. Page 3 "
        "opens with the word *Index*; page 172 closes with a list of Ayurvedic "
        "reference works. The scanned enclosure is an **index of Ayurveda Aahara "
        "preparations and their authoritative textual references**, annexed to the "
        "2-page FSSAI covering Order of 25 July 2025.",
        "",
        "This corrects the Phase 1 audit, which inferred from the covering Order "
        "that the Regulations text and the Schedule A/B lists were behind the "
        "scan. They are not. **The Food Safety and Standards (Ayurveda Aahara) "
        "Regulations, 2022 are absent from the corpus** and must be sourced "
        "separately — which matters, because they are the most Ayurveda-specific "
        "instrument on the expected list. All 28 chunks from this document are "
        "correctly marked `locator_unavailable`.",
        "",
        "### Hindi is excluded from retrieval, not repaired",
        "",
        f"{totals.get('hindi_lines_excluded', 0)} Devanagari-bearing lines were "
        "dropped from the retrieval text across the five bilingual documents. Phase "
        "1 established that every one of them has a defective ToUnicode CMap: the "
        "output is well-formed Unicode in the correct block and the wrong "
        "characters, so no encoding check catches it. A line is dropped if it "
        "contains *any* Devanagari, because OCR of a bilingual scan interleaves the "
        "scripts within a line. Nothing is transliterated, translated or corrected — "
        "`raw_text` retains every dropped line verbatim for Phase 3.",
        "",
        "### Two consolidations are stale",
        "",
        "The Patents Rules, 2003 are consolidated only to 23 June 2017 and the "
        "Drugs and Cosmetics Rules, 1945 only to 31 December 2016. Their chunks are "
        "faithful to the PDFs and the PDFs are out of date.",
        "",
        "### Provenance remains unverified",
        "",
        "Every chunk carries `provenance_status: PROVENANCE_UNVERIFIED`, inherited "
        "from the Phase 1 audit: 15 of the 21 source PDFs are browser re-encodings "
        "whose publisher metadata was destroyed, and TRIPS carries no publisher "
        "imprint at all. Ingestion cannot fix this; only re-fetching each document "
        "from its publisher of record can.",
        "",
        "### Layout artefacts that survive extraction",
        "",
        "The Drugs and Cosmetics compendium sets marginal notes in a side column, "
        "which the text layer interleaves into the body — clauses occasionally "
        "appear out of reading order within a provision. This is a property of the "
        "PDF, not of the pipeline, and is left uncorrected rather than guessed at.",
        "",
        f"VALIDATION_STATUS: {result['status']}",
        "",
    ]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


if __name__ == "__main__":
    outcome = run()
    report_path = write_report(outcome)
    with open(os.path.join(DOCS, "ingestion_validation.json"), "w", encoding="utf-8") as fh:
        json.dump(outcome, fh, indent=2, ensure_ascii=False)
    print(f"wrote {report_path}")
    sys.exit(0 if outcome["status"] == "PASS" else 1)
