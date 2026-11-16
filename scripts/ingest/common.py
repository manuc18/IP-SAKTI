"""Shared paths, the logical-document registry, and small IO helpers.

The pipeline treats ``~/Downloads/IP-SAKTI Corpus`` as immutable. Nothing in this
package opens a corpus file in any mode other than ``rb``.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterator

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(PROJECT_ROOT, "data")
DOCS = os.path.join(PROJECT_ROOT, "docs")
METADATA_DIR = os.path.join(DATA, "metadata")
SOURCE_UPDATES_DIR = os.path.join(DATA, "source_updates")
MANIFEST_PATH = os.path.join(METADATA_DIR, "corpus_manifest.json")
EXTRACTED_DIR = os.path.join(DATA, "processed", "extracted")
CLEANED_DIR = os.path.join(DATA, "processed", "cleaned")
STRUCTURED_DIR = os.path.join(DATA, "processed", "structured")
CHUNKS_DIR = os.path.join(DATA, "processed", "chunks")

CORPUS_ROOT = os.environ.get(
    "IPSAKTI_CORPUS_ROOT", os.path.expanduser("~/Downloads/IP-SAKTI Corpus")
)

# ---------------------------------------------------------------- language ---

LANG_EN_CLEAN = "EN_CLEAN"
LANG_HI_CORRUPT = "HI_CORRUPT"
LANG_HI_OCR_PENDING = "HI_OCR_PENDING"
LANG_EN_HI_MIXED = "EN_HI_MIXED"
LANG_UNKNOWN = "UNKNOWN"

DEVANAGARI = re.compile(r"[ऀ-ॿ]")

#: Hindi legal tokens that a *correct* ToUnicode CMap produces, paired with the
#: mojibake the audit actually observed in this corpus. Presence of the bad form
#: is positive evidence of a defective CMap; the text is valid Unicode, so no
#: encoding check can catch it.
HINDI_CORRUPT_MARKERS = (
    "रािपत्र", "अजधजनयम", "प्राजधकरण", "जवजनयम", "जनयम", "सिस्ट्र्",
    "अजधसूचना", "संजिप्त", "जवजवधता",
)

#: Signature substrings of the legacy 8-bit Chanakya/Kruti Dev encodings, which
#: extract as ASCII garble rather than Devanagari.
LEGACY_FONT_MARKERS = (
    "Hkkjr dk jkti=k", "vlk/kj.k", "jpukRed", "¹Hkkx", "[k.M",
)


def devanagari_ratio(text: str) -> float:
    """Fraction of letters that are Devanagari, over Devanagari + ASCII letters."""
    dev = len(DEVANAGARI.findall(text))
    lat = sum(1 for c in text if c.isascii() and c.isalpha())
    total = dev + lat
    return (dev / total) if total else 0.0


def classify_language(text: str, *, extraction_method: str = "native") -> str:
    """Classify a page or chunk of text into one of the five ``language_status`` values.

    Hindi is never trusted in this corpus: every bilingual PDF audited in Phase 1
    had a defective ToUnicode CMap, so any Devanagari-dominant text is reported
    as ``HI_CORRUPT`` unless it came from OCR (``HI_OCR_PENDING``).
    """
    stripped = text.strip()
    if not stripped:
        return LANG_UNKNOWN
    ratio = devanagari_ratio(stripped)
    if ratio >= 0.60:
        return LANG_HI_OCR_PENDING if extraction_method == "ocr" else LANG_HI_CORRUPT
    if ratio >= 0.05:
        return LANG_EN_HI_MIXED
    return LANG_EN_CLEAN


def hindi_corruption_hits(text: str) -> int:
    return sum(text.count(m) for m in HINDI_CORRUPT_MARKERS)


def legacy_font_hits(text: str) -> int:
    return sum(text.count(m) for m in LEGACY_FONT_MARKERS)


# ------------------------------------------------------------------ helpers ---


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_jsonl(path: str, records: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def iter_jsonl_dir(directory: str) -> Iterator[tuple[str, list[dict[str, Any]]]]:
    for name in sorted(os.listdir(directory)):
        if name.endswith(".jsonl"):
            yield name[:-6], read_jsonl(os.path.join(directory, name))


# ------------------------------------------------------- document registry ---


@dataclass
class LogicalDocument:
    """One citable legal instrument.

    Usually 1:1 with a source PDF. The Drugs & Cosmetics compendium is the
    exception: one physical PDF carries two instruments, which must be citable
    independently, so it yields two logical documents sharing a
    ``parent_source_document_id``.
    """

    document_id: str
    parent_source_document_id: str | None
    source_filename: str
    source_sha256: str
    title: str
    short_title: str
    domain: str
    jurisdiction: str
    document_type: str
    authority: str | None
    parser: str  # "statutory" | "treaty"
    provision_label: str  # "Section" | "Rule" | "Regulation" | "Article" | "Paragraph"
    citation_style: str  # "indian" | "dot" | "paren"
    page_start: int
    page_end: int
    ocr_priority: str
    ocr_pages: list[int] = field(default_factory=list)
    ocr_lang: str = "eng"
    provenance_status: str = "PROVENANCE_UNVERIFIED"
    version: str | None = None
    reference_number: str | None = None
    source_url: str | None = None

    @property
    def source_path(self) -> str:
        return os.path.join(CORPUS_ROOT, self.source_filename)

    @property
    def page_range(self) -> range:
        return range(self.page_start, self.page_end + 1)


#: Physical page ranges for the one compendium PDF that carries two instruments.
#: Verified by the running header, which changes from "Drugs and Cosmetics Act,
#: 1940" to "Drugs and Cosmetics Rules, 1945" at physical page 39, where the
#: 21st-December-1945 notification and PART I / PRELIMINARY begin.
COMPENDIUM_SPLITS: dict[str, list[dict[str, Any]]] = {
    "IN-DC-ACT1940-RULES1945": [
        {
            "document_id": "IN-DC-ACT-1940",
            "title": "The Drugs and Cosmetics Act, 1940 (23 of 1940), as amended up to 31 December 2016",
            "short_title": "Drugs and Cosmetics Act, 1940",
            "document_type": "act",
            "provision_label": "Section",
            "reference_number": "Act 23 of 1940",
            "page_start": 1,
            "page_end": 38,
        },
        {
            "document_id": "IN-DC-RULES-1945",
            "title": "The Drugs and Cosmetics Rules, 1945, as amended up to 31 December 2016",
            "short_title": "Drugs and Cosmetics Rules, 1945",
            "document_type": "rules",
            "provision_label": "Rule",
            "reference_number": "Notification No. F. 28-10/45-H(1)",
            "page_start": 39,
            "page_end": 635,
        },
    ],
}

#: Per-document overrides. Short titles are the form used in citations; they are
#: taken from each document's own title page, not from its filename.
DOC_OVERRIDES: dict[str, dict[str, Any]] = {
    "IN-PATENTS-ACT-1970": {"short_title": "Patents Act, 1970", "provision_label": "Section"},
    "IN-PATENTS-RULES-2003": {"short_title": "Patents Rules, 2003", "provision_label": "Rule"},
    "IN-TM-ACT-1999": {"short_title": "Trade Marks Act, 1999", "provision_label": "Section"},
    "IN-TM-RULES-2017": {"short_title": "Trade Marks Rules, 2017", "provision_label": "Rule"},
    "IN-GI-ACT-1999": {
        "short_title": "Geographical Indications of Goods (Registration and Protection) Act, 1999",
        "provision_label": "Section",
    },
    "IN-GI-RULES-2002": {
        "short_title": "Geographical Indications of Goods (Registration and Protection) Rules, 2002",
        "provision_label": "Rule",
    },
    "IN-DESIGNS-ACT-2000": {"short_title": "Designs Act, 2000", "provision_label": "Section"},
    "IN-DESIGNS-RULES-2001": {"short_title": "Designs Rules, 2001", "provision_label": "Rule"},
    "IN-COPYRIGHT-ACT-1957": {"short_title": "Copyright Act, 1957", "provision_label": "Section"},
    "IN-COPYRIGHT-RULES-2013": {"short_title": "Copyright Rules, 2013", "provision_label": "Rule"},
    "IN-BDA-2002-AMD-2023": {
        "short_title": "Biological Diversity Act, 2002",
        "provision_label": "Section",
    },
    "IN-BD-RULES-2024": {"short_title": "Biological Diversity Rules, 2024", "provision_label": "Rule"},
    "IN-ABS-REGULATIONS-2025": {
        "short_title": "Access and Benefit Sharing Regulations, 2025",
        "provision_label": "Regulation",
    },
    "IN-PPVFR-ACT-2001": {
        "short_title": "Protection of Plant Varieties and Farmers' Rights Act, 2001",
        "provision_label": "Section",
    },
    "IN-PPVFR-RULES-2003": {
        "short_title": "Protection of Plant Varieties and Farmers' Rights Rules, 2003",
        "provision_label": "Rule",
    },
    "IN-FSSAI-AYURVEDA-AAHARA-2022": {
        "short_title": "Food Safety and Standards (Ayurveda Aahara) Regulations, 2022",
        "provision_label": "Regulation",
        "ocr_lang": "eng+hin",
    },
    "IN-NATIONAL-IPR-POLICY-2016": {
        "short_title": "National IPR Policy, 2016",
        "provision_label": "Paragraph",
    },
    "WTO-TRIPS": {
        "short_title": "TRIPS",
        "parser": "treaty",
        "provision_label": "Article",
        "citation_style": "dot",
    },
    "UN-CBD-1992": {
        "short_title": "CBD",
        "parser": "treaty",
        "provision_label": "Article",
        "citation_style": "paren",
    },
    "UN-NAGOYA-PROTOCOL-2010": {
        "short_title": "Nagoya Protocol",
        "parser": "treaty",
        "provision_label": "Article",
        "citation_style": "dot",
    },
}


def load_manifest() -> dict[str, Any]:
    with open(MANIFEST_PATH, encoding="utf-8") as fh:
        return json.load(fh)


def logical_documents(manifest: dict[str, Any] | None = None) -> list[LogicalDocument]:
    """Expand the 21 audited PDFs into the corpus's logical documents."""
    manifest = manifest or load_manifest()
    out: list[LogicalDocument] = []
    for entry in manifest["documents"]:
        source_id = entry["document_id"]
        splits = COMPENDIUM_SPLITS.get(source_id)
        parts = splits or [
            {
                "document_id": source_id,
                "title": entry.get("title_detected") or entry.get("expected_title") or source_id,
                "short_title": entry.get("expected_title") or source_id,
                "document_type": entry["document_type"],
                "provision_label": "Section",
                "reference_number": entry.get("reference_number"),
                "page_start": 1,
                "page_end": entry["page_count"],
            }
        ]
        for part in parts:
            over = DOC_OVERRIDES.get(part["document_id"], {})
            ocr_pages = [
                p
                for p in entry.get("ocr_pages", [])
                if part["page_start"] <= p <= part["page_end"]
            ]
            out.append(
                LogicalDocument(
                    document_id=part["document_id"],
                    parent_source_document_id=source_id if splits else None,
                    source_filename=entry["filename"],
                    source_sha256=entry["sha256"],
                    title=part["title"],
                    short_title=over.get("short_title", part["short_title"]),
                    domain=entry["domain"],
                    jurisdiction=entry["jurisdiction"],
                    document_type=part["document_type"],
                    authority=entry.get("authority"),
                    parser=over.get("parser", "statutory"),
                    provision_label=over.get("provision_label", part["provision_label"]),
                    citation_style=over.get("citation_style", "indian"),
                    page_start=part["page_start"],
                    page_end=part["page_end"],
                    ocr_priority=entry.get("ocr_priority", "NONE"),
                    # The audit flagged low-text pages in four further documents
                    # (cover art, portrait plates, colophons) and cleared them on
                    # inspection. Only CRITICAL/HIGH documents are OCR targets.
                    ocr_pages=ocr_pages if entry.get("ocr_priority") in ("CRITICAL", "HIGH") else [],
                    ocr_lang=over.get("ocr_lang", "eng"),
                    provenance_status=entry.get("provenance_status", "PROVENANCE_UNVERIFIED"),
                    version=entry.get("version"),
                    reference_number=part.get("reference_number") or entry.get("reference_number"),
                    source_url=entry.get("source_url"),
                )
            )
    return out


def document_index() -> dict[str, LogicalDocument]:
    return {d.document_id: d for d in logical_documents()}
