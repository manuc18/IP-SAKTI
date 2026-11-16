"""Stage 1 — native text extraction.

One record per physical page, per logical document. The corpus is opened
read-only and its SHA-256 is re-verified against the Phase 1 audit manifest
before a single page is read: a source file that has changed since the audit
fails the pipeline rather than silently contaminating the chunks.
"""
from __future__ import annotations

import os
import sys
from typing import Any

from pypdf import PdfReader

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.common import (  # noqa: E402
    EXTRACTED_DIR,
    LogicalDocument,
    classify_language,
    devanagari_ratio,
    hindi_corruption_hits,
    legacy_font_hits,
    logical_documents,
    sha256_file,
    write_jsonl,
)

#: Below this many characters a page carries no usable text layer. The Phase 1
#: audit inspected every such page individually: most are cover art, portrait
#: plates or colophons; only the pages listed in ``LogicalDocument.ocr_pages``
#: are genuine scans of statutory text.
MIN_TEXT_CHARS = 100


class SourceChanged(RuntimeError):
    pass


def extract_document(doc: LogicalDocument, reader: PdfReader) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for physical_page in doc.page_range:
        page = reader.pages[physical_page - 1]
        try:
            text = page.extract_text() or ""
        except Exception as exc:  # pragma: no cover - defensive; audit found none
            text = ""
            error = f"{type(exc).__name__}: {exc}"
        else:
            error = None
        stripped = text.strip()
        records.append(
            {
                "document_id": doc.document_id,
                "parent_source_document_id": doc.parent_source_document_id,
                "source_filename": doc.source_filename,
                "source_sha256": doc.source_sha256,
                "physical_page": physical_page,
                "extraction_method": "native",
                "text": text,
                "character_count": len(text),
                "is_empty": len(stripped) == 0,
                "is_thin": 0 < len(stripped) < MIN_TEXT_CHARS,
                "devanagari_ratio": round(devanagari_ratio(text), 4),
                "language_status": classify_language(text),
                "hindi_corruption_markers": hindi_corruption_hits(text),
                "legacy_font_markers": legacy_font_hits(text),
                "ocr_expected": physical_page in doc.ocr_pages,
                "extraction_error": error,
            }
        )
    return records


def run(verbose: bool = True) -> dict[str, Any]:
    docs = logical_documents()
    by_source: dict[str, list[LogicalDocument]] = {}
    for doc in docs:
        by_source.setdefault(doc.source_filename, []).append(doc)

    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    summary: dict[str, Any] = {"documents": {}, "pages": 0, "empty_pages": 0, "thin_pages": 0}

    for filename, group in sorted(by_source.items()):
        path = group[0].source_path
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        digest = sha256_file(path)
        if digest != group[0].source_sha256:
            raise SourceChanged(
                f"{filename}: sha256 {digest} does not match the audited digest "
                f"{group[0].source_sha256}. The corpus has changed since Phase 1; "
                f"re-run the audit before ingesting."
            )
        reader = PdfReader(path)
        for doc in group:
            records = extract_document(doc, reader)
            write_jsonl(os.path.join(EXTRACTED_DIR, f"{doc.document_id}.jsonl"), records)
            stats = {
                "pages": len(records),
                "characters": sum(r["character_count"] for r in records),
                "empty_pages": sum(1 for r in records if r["is_empty"]),
                "thin_pages": sum(1 for r in records if r["is_thin"]),
                "ocr_expected_pages": sum(1 for r in records if r["ocr_expected"]),
                "errors": sum(1 for r in records if r["extraction_error"]),
            }
            summary["documents"][doc.document_id] = stats
            summary["pages"] += stats["pages"]
            summary["empty_pages"] += stats["empty_pages"]
            summary["thin_pages"] += stats["thin_pages"]
            if verbose:
                print(
                    f"extract {doc.document_id:30s} pages={stats['pages']:4d} "
                    f"chars={stats['characters']:>9,d} empty={stats['empty_pages']:3d} "
                    f"ocr_expected={stats['ocr_expected_pages']:3d}",
                    flush=True,
                )
    return summary


if __name__ == "__main__":
    run()
