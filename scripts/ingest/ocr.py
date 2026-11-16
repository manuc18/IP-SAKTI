"""Stage 1b — OCR, restricted to the pages the Phase 1 audit proved are scans.

OCR never overwrites a native text layer. Results land in a sibling file
``{document_id}.ocr.jsonl``; ``clean.py`` merges them and records
``extraction_method: "ocr"`` while keeping the native extraction addressable.

Only two documents qualify:

* ``IN-FSSAI-AYURVEDA-AAHARA-2022`` pages 3-172 — the Regulations themselves are
  a scanned enclosure behind a 2-page covering Order (``eng+hin``).
* ``IN-PATENTS-RULES-2003`` pages 56-102 — the statutory Forms and fee Schedules
  (``eng``).
"""
from __future__ import annotations

import os
import shutil
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.common import (  # noqa: E402
    EXTRACTED_DIR,
    LogicalDocument,
    classify_language,
    devanagari_ratio,
    logical_documents,
    write_jsonl,
)

RENDER_DPI = 300


def missing_dependencies() -> list[str]:
    """Return human-readable names of anything OCR needs and cannot find."""
    missing: list[str] = []
    try:
        import pypdfium2  # noqa: F401
    except ImportError:
        missing.append("pypdfium2 (pip install pypdfium2) — renders PDF pages to images")
    try:
        import pytesseract  # noqa: F401
    except ImportError:
        missing.append("pytesseract (pip install pytesseract)")
    binary = shutil.which("tesseract")
    if binary is None:
        missing.append("tesseract binary (brew install tesseract tesseract-lang)")
    else:
        import subprocess

        langs = subprocess.run(
            [binary, "--list-langs"], capture_output=True, text=True
        ).stdout.split()
        needed = {lang for doc in ocr_targets() for lang in doc.ocr_lang.split("+")}
        for lang in sorted(needed - set(langs)):
            missing.append(f"tesseract language data '{lang}' (brew install tesseract-lang)")
    return missing


def ocr_targets() -> list[LogicalDocument]:
    return [d for d in logical_documents() if d.ocr_pages]


def ocr_document(doc: LogicalDocument, verbose: bool = True) -> list[dict[str, Any]]:
    import pypdfium2 as pdfium
    import pytesseract

    pdf = pdfium.PdfDocument(doc.source_path)
    scale = RENDER_DPI / 72.0
    records: list[dict[str, Any]] = []
    for physical_page in doc.ocr_pages:
        image = pdf[physical_page - 1].render(scale=scale).to_pil()
        data = pytesseract.image_to_data(
            image, lang=doc.ocr_lang, output_type=pytesseract.Output.DICT
        )
        words = [
            (word, int(conf))
            for word, conf in zip(data["text"], data["conf"])
            if word.strip() and int(conf) >= 0
        ]
        text = pytesseract.image_to_string(image, lang=doc.ocr_lang)
        mean_conf = round(sum(c for _, c in words) / len(words), 2) if words else None
        low_conf = sum(1 for _, c in words if c < 60)
        records.append(
            {
                "document_id": doc.document_id,
                "source_filename": doc.source_filename,
                "source_sha256": doc.source_sha256,
                "physical_page": physical_page,
                "extraction_method": "ocr",
                "ocr_engine": "tesseract",
                "ocr_lang": doc.ocr_lang,
                "ocr_dpi": RENDER_DPI,
                "ocr_mean_confidence": mean_conf,
                "ocr_word_count": len(words),
                "ocr_low_confidence_words": low_conf,
                "text": text,
                "character_count": len(text),
                "devanagari_ratio": round(devanagari_ratio(text), 4),
                "language_status": classify_language(text, extraction_method="ocr"),
            }
        )
        if verbose and physical_page % 10 == 0:
            print(f"  ocr {doc.document_id} page {physical_page} conf={mean_conf}", flush=True)
    pdf.close()
    return records


def run(verbose: bool = True) -> dict[str, Any]:
    missing = missing_dependencies()
    if missing:
        raise RuntimeError("OCR dependencies unavailable:\n  - " + "\n  - ".join(missing))
    os.makedirs(EXTRACTED_DIR, exist_ok=True)
    summary: dict[str, Any] = {"documents": {}, "pages": 0}
    for doc in ocr_targets():
        if verbose:
            print(
                f"ocr {doc.document_id}: {len(doc.ocr_pages)} pages, lang={doc.ocr_lang}",
                flush=True,
            )
        records = ocr_document(doc, verbose=verbose)
        write_jsonl(os.path.join(EXTRACTED_DIR, f"{doc.document_id}.ocr.jsonl"), records)
        confs = [r["ocr_mean_confidence"] for r in records if r["ocr_mean_confidence"] is not None]
        summary["documents"][doc.document_id] = {
            "pages": len(records),
            "lang": doc.ocr_lang,
            "characters": sum(r["character_count"] for r in records),
            "mean_confidence": round(sum(confs) / len(confs), 2) if confs else None,
            "pages_below_60_confidence": sum(
                1 for c in confs if c < 60
            ),
        }
        summary["pages"] += len(records)
    return summary


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
