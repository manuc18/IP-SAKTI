"""Stage 2 — deterministic, auditable page cleaning.

Every removal is positional and evidence-driven, never content-driven:

* **Running headers/footers** are only stripped from the first/last few lines of
  a page, and only when the same line (digits normalised to ``#``) recurs across
  a large fraction of the document's pages. Substantive legal text does not
  repeat in header position on a third of a statute's pages.
* **Amendment footnotes** are lifted out of the operative text into a separate
  ``amendment_notes`` list. They are never deleted — Phase 2 keeps operative text
  and amendment metadata separately addressable.
* **Devanagari lines** are dropped from the retrieval text because Phase 1 proved
  every bilingual PDF in this corpus has a defective ToUnicode CMap. They survive
  verbatim in ``raw_text``. Nothing is transliterated, translated or repaired.

``raw_text`` and ``cleaned_text`` are both persisted so any transformation can be
diffed after the fact.
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

from ingest.common import (  # noqa: E402
    CLEANED_DIR,
    DEVANAGARI,
    EXTRACTED_DIR,
    LANG_EN_CLEAN,
    LANG_HI_CORRUPT,
    LANG_HI_OCR_PENDING,
    LANG_UNKNOWN,
    LEGACY_FONT_MARKERS,
    LogicalDocument,
    classify_language,
    devanagari_ratio,
    logical_documents,
    read_jsonl,
    write_jsonl,
)

#: Contents-page detection thresholds: what starts a run, what continues it, and
#: how much of a document counts as front matter.
TOC_START_SCORE = 0.50
TOC_CONTINUE_SCORE = 0.40
TOC_FRONT_FRACTION = 0.15
TOC_FRONT_MATTER_FRACTION = 0.33

#: A page left with less than this after Hindi removal is table residue, not text.
THIN_RESIDUE_CHARS = 60

#: How many leading/trailing lines of a page may be considered header/footer.
EDGE_LINES = 4
#: A line must recur on at least this fraction of pages to count as boilerplate.
BOILERPLATE_MIN_FRACTION = 0.30
BOILERPLATE_MIN_PAGES = 3
BOILERPLATE_MAX_LEN = 160

#: Always boilerplate, wherever the repetition statistics land.
ALWAYS_STRIP = [
    re.compile(r"^\d{1,4}$"),                                  # bare page number
    re.compile(r"(?i)^page\s+\d{1,4}\s+of\s+\d{1,4}$"),
    re.compile(r"(?i)^indiacode$"),                            # India Code watermark
    re.compile(r"^[\*•\-–—_=~\.\s]+$"),                   # rule-off / dinkus
    re.compile(r"(?i)^national ipr policy__AA\.indd\s+\d+.*$"),  # InDesign job stamp
    re.compile(r"(?i)^\d+\s*/\s*\d+\s*(GI|GI/\d+)?$"),         # gazette press marks
]

BARE_NUMBER_RE = re.compile(r"^(\d{1,4})$")
#: How far the printed folio may drift from the physical page index — front
#: matter with roman numerals, or a cover sheet, shifts it by a few pages.
PAGE_NUMBER_TOLERANCE = 10

#: A centred structural heading whose number is set on its own line beneath the
#: word. The CBD and the Nagoya Protocol typeset every Article this way, so the
#: text layer emits ``Article`` / ``15`` / ``ACCESS TO GENETIC RESOURCES`` as
#: three lines. The middle line is a bare number and looks exactly like a folio
#: — which is how CBD Articles 1-39 and Nagoya Articles 1-20 lost their numbers
#: and, with them, their citations. The line above it is the evidence that tells
#: the two apart, and it is layout evidence, not a guess about the number.
CENTRED_HEADING_WORD = re.compile(
    r"(?i)^\s*(?:article|section|chapter|part|rule|regulation)\s*$")

FOOTNOTE_START = re.compile(r"^\s*(\d{1,2})\s*[\.\)]?\s+\S")
#: A wrapped footnote may spill onto at most this many continuation lines.
FOOTNOTE_MAX_CONTINUATION = 2
#: Reject a candidate block that would swallow this much of the page — a real
#: footnote apparatus is a minority of the page it annotates.
FOOTNOTE_MAX_PAGE_FRACTION = 0.45
FOOTNOTE_CUE = re.compile(
    r"(?i)\b(subs\.|ins\.|omitted|omit\.|rep\.|repealed|renumbered|added by|substituted|"
    r"inserted|w\.e\.f\.|vide\s+notif|notifn\.|g\.s\.r\.|s\.o\.\s*\d|gazette|"
    r"act\s+\d+\s+of\s+(?:19|20)\d{2}|by\s+s\.\s*\d+|for\s+the\s+words)\b"
)

TOC_TRIGGER = re.compile(
    r"(?im)^\s*(ARRANGEMENT\s+OF\s+(SECTIONS|RULES|REGULATIONS|ARTICLES|CLAUSES)|"
    r"TABLE\s+OF\s+CONTENTS|CONTENTS)\s*$"
)
NUMBERED_LINE = re.compile(r"^\s*\d{1,3}[A-Z]{0,2}\s*\.\s+\S")
DOT_LEADER = re.compile(r"[\.…]{3,}\s*\d{0,4}\s*$")

TABLE_LINE = re.compile(r"(\S {3,}\S.* {3,}\S)|\t|\|")

#: Structural headings are never boilerplate, even when they recur in header
#: position on many pages. In the CBD, the Nagoya Protocol and TRIPS the article
#: heading *is* the top line of the page; stripping it as a running header would
#: destroy the only citable structure those instruments have.
PROTECTED_HEADING = re.compile(
    r"(?i)^\s*(?:articles?\s*\d*|(?:chapter|part)\s+(?:[ivxlcdm]+|\d+)[A-Za-z]?|"
    r"(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
    r"eleventh|twelfth)?\s*schedule(?:\s*[-\u2013\u2014]?\s*[A-Z]|\s*[IVXLCDM]+)?|"
    r"form\s+\d+[A-Z]*|annex(?:ure)?(?:\s+(?:[IVXLCDM]+|\d+|[A-Z]))?)\s*$"
)


def normalise(line: str) -> str:
    return re.sub(r"\d+", "#", re.sub(r"\s+", " ", line.strip()))


def is_always_strip(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and any(p.match(stripped) for p in ALWAYS_STRIP)


def is_legacy_garble(line: str) -> bool:
    """Hindi set in a legacy 8-bit font (Chanakya/Kruti Dev) extracts as ASCII noise."""
    return sum(1 for m in LEGACY_FONT_MARKERS if m in line) >= 1


def is_devanagari_line(line: str) -> bool:
    """Any Devanagari at all disqualifies a line from the English retrieval text.

    A ratio threshold is not enough. OCR of a bilingual scan interleaves the two
    scripts *within* a line, leaving one or two Devanagari characters embedded in
    otherwise-Latin text — and in this corpus Devanagari is never trustworthy, so
    a line that mixes scripts cannot be trusted either. Whole lines are dropped
    rather than individual characters: removing characters would be repairing the
    text, which this phase does not do. ``raw_text`` keeps everything.
    """
    return bool(DEVANAGARI.search(line))


def collect_boilerplate(pages: list[list[str]]) -> set[str]:
    """Normalised lines that recur in header/footer position across the document."""
    head, tail = Counter(), Counter()
    for lines in pages:
        content = [ln for ln in lines if ln.strip()]
        for ln in content[:EDGE_LINES]:
            head[normalise(ln)] += 1
        for ln in content[-EDGE_LINES:]:
            tail[normalise(ln)] += 1
    threshold = max(BOILERPLATE_MIN_PAGES, int(BOILERPLATE_MIN_FRACTION * len(pages)))
    out = set()
    for counter in (head, tail):
        for norm, count in counter.items():
            if norm and count >= threshold and len(norm) <= BOILERPLATE_MAX_LEN:
                out.add(norm)
    return out


def strip_edges(lines: list[str], boilerplate: set[str]) -> tuple[list[str], list[str]]:
    """Remove boilerplate from the top and bottom of a page. Returns (kept, removed)."""
    removed: list[str] = []

    def drop(line: str) -> bool:
        if PROTECTED_HEADING.match(line.strip()):
            return False
        return (
            not line.strip()
            or is_always_strip(line)
            or normalise(line) in boilerplate
            or is_legacy_garble(line)
        )

    lo, hi = 0, len(lines)
    for _ in range(EDGE_LINES + 2):
        while lo < hi and not lines[lo].strip():
            lo += 1
        if lo < hi and drop(lines[lo]):
            removed.append(lines[lo].strip())
            lo += 1
        else:
            break
    for _ in range(EDGE_LINES + 2):
        while hi > lo and not lines[hi - 1].strip():
            hi -= 1
        if hi > lo and drop(lines[hi - 1]):
            removed.append(lines[hi - 1].strip())
            hi -= 1
        else:
            break
    return lines[lo:hi], [r for r in removed if r]


def strip_page_numbers(lines: list[str], physical_page: int) -> tuple[list[str], list[str]]:
    """Remove the printed folio wherever it sits on the page.

    ``strip_edges`` only reaches the outermost lines, but a folio can sit behind
    a running header or a blank, and then survives into the middle of a chunk
    where a provision spans a page break. A line consisting of nothing but a
    number close to this page's own number is that folio and nothing else.
    """
    kept: list[str] = []
    removed: list[str] = []
    previous = ""
    for line in lines:
        stripped = line.strip()
        match = BARE_NUMBER_RE.match(stripped)
        if (match
                and abs(int(match.group(1)) - physical_page) <= PAGE_NUMBER_TOLERANCE
                and not CENTRED_HEADING_WORD.match(previous)):
            removed.append(stripped)
            continue
        kept.append(line)
        if stripped:
            previous = stripped
    return kept, removed


def split_footnotes(lines: list[str]) -> tuple[list[str], list[str]]:
    """Lift the trailing amendment-footnote block off a page.

    Footnotes are numbered lines carrying a statutory-amendment cue ("Subs. by
    Act 10 of 2023 ... w.e.f. 1-4-2024"), possibly wrapped onto a continuation
    line. Three guards keep operative text out of the block: only a contiguous
    run at the very foot of the page is considered, the footnote numbers must
    ascend, and the block may not account for most of the page.
    """
    body_chars = sum(len(ln.strip()) for ln in lines)
    end = len(lines)
    while end > 0 and not lines[end - 1].strip():
        end -= 1
    notes: list[list[str]] = []
    numbers: list[int] = []
    taken_chars = 0

    while end > 0:
        tail: list[str] = []
        found = None
        for idx in range(end - 1, max(-1, end - FOOTNOTE_MAX_CONTINUATION - 2), -1):
            line = lines[idx]
            if not line.strip():
                break
            match = FOOTNOTE_START.match(line)
            if match and FOOTNOTE_CUE.search(" ".join([line] + tail)):
                found = (idx, int(match.group(1)), [line] + tail)
                break
            tail.insert(0, line)
        if found is None:
            break
        idx, number, block = found
        if numbers and number >= numbers[-1]:
            break  # numbering must ascend down the page
        chars = sum(len(ln.strip()) for ln in block)
        if body_chars and (taken_chars + chars) / body_chars > FOOTNOTE_MAX_PAGE_FRACTION:
            break
        notes.append(block)
        numbers.append(number)
        taken_chars += chars
        end = idx
        while end > 0 and not lines[end - 1].strip():
            end -= 1

    rendered = [re.sub(r"\s+", " ", " ".join(b).strip()) for b in reversed(notes)]
    return lines[:end], rendered


def toc_like(lines: list[str]) -> float:
    content = [ln for ln in lines if ln.strip()]
    if len(content) < 5:
        return 0.0
    hits = sum(
        1
        for ln in content
        if len(ln.strip()) <= 110 and (NUMBERED_LINE.match(ln) or DOT_LEADER.search(ln))
    )
    return hits / len(content)


def dot_leader_fraction(lines: list[str]) -> float:
    """Dot leaders are a contents page's unmistakable signature."""
    content = [ln for ln in lines if ln.strip()]
    if len(content) < 4:
        return 0.0
    return sum(1 for ln in content if DOT_LEADER.search(ln)) / len(content)


def detect_table(text: str) -> bool:
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if len(lines) < 4:
        return False
    hits = sum(1 for ln in lines if TABLE_LINE.search(ln))
    return hits >= 4 and hits / len(lines) >= 0.30


def load_pages(document_id: str) -> list[dict[str, Any]]:
    """Native pages, with audited scan pages replaced by their OCR record.

    The native record is preserved under ``native_extraction`` — OCR output never
    overwrites a text layer, it only supersedes an absent one.
    """
    native = read_jsonl(os.path.join(EXTRACTED_DIR, f"{document_id}.jsonl"))
    ocr_path = os.path.join(EXTRACTED_DIR, f"{document_id}.ocr.jsonl")
    if not os.path.exists(ocr_path):
        return native
    ocr = {r["physical_page"]: r for r in read_jsonl(ocr_path)}
    merged = []
    for rec in native:
        hit = ocr.get(rec["physical_page"])
        if hit is None:
            merged.append(rec)
            continue
        combined = dict(rec)
        combined["native_extraction"] = {
            "text": rec["text"],
            "character_count": rec["character_count"],
        }
        combined.update(
            {
                k: hit[k]
                for k in (
                    "text",
                    "character_count",
                    "extraction_method",
                    "ocr_engine",
                    "ocr_lang",
                    "ocr_dpi",
                    "ocr_mean_confidence",
                    "ocr_word_count",
                    "ocr_low_confidence_words",
                    "devanagari_ratio",
                    "language_status",
                )
            }
        )
        merged.append(combined)
    return merged


def clean_document(doc: LogicalDocument) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pages = load_pages(doc.document_id)
    page_lines = [rec["text"].split("\n") for rec in pages]
    boilerplate = collect_boilerplate(page_lines)

    # Table-of-contents front matter: a contiguous run of entry-dense pages
    # beginning at an explicit ARRANGEMENT OF SECTIONS / CONTENTS heading.
    toc_pages: set[int] = set()
    front_region = max(1, int(TOC_FRONT_FRACTION * len(pages)))
    scores = [toc_like(lines) for lines in page_lines]
    position = 0
    while position < len(pages):
        triggered = bool(TOC_TRIGGER.search(pages[position]["text"]))
        dot_leaders = dot_leader_fraction(page_lines[position]) >= 0.50
        # Dot leaders and an explicit "ARRANGEMENT OF SECTIONS" are unambiguous
        # wherever they appear. Entry density alone is not — a fee schedule looks
        # much the same — so it only starts a run in the front matter, and only
        # when the run is at least two pages long.
        strong = dot_leaders or (triggered and scores[position] >= TOC_CONTINUE_SCORE - 0.10)
        weak = position < front_region and scores[position] >= TOC_START_SCORE
        if not (strong or weak):
            position += 1
            continue
        end = position + 1
        while end < len(pages) and scores[end] >= TOC_CONTINUE_SCORE:
            end += 1
        if strong or end - position >= 2:
            toc_pages.update(pages[index]["physical_page"] for index in range(position, end))
        position = end

    # Only treat pages before the contents as front matter when the contents are
    # genuinely front matter; a late false positive must not swallow the body.
    first_toc_page = None
    if toc_pages:
        first = min(toc_pages)
        if pages[0]["physical_page"] + int(TOC_FRONT_MATTER_FRACTION * len(pages)) >= first:
            first_toc_page = first

    out: list[dict[str, Any]] = []
    for rec, lines in zip(pages, page_lines):
        kept, removed = strip_edges(lines, boilerplate)
        kept, page_numbers = strip_page_numbers(kept, rec["physical_page"])
        removed.extend(page_numbers)
        is_toc = rec["physical_page"] in toc_pages
        kept, notes = (kept, []) if is_toc else split_footnotes(kept)

        hindi_removed = [ln for ln in kept if is_devanagari_line(ln) or is_legacy_garble(ln)]
        kept = [ln for ln in kept if ln not in hindi_removed]
        cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()

        if is_toc:
            content_type = "table_of_contents"
        elif first_toc_page is not None and rec["physical_page"] < first_toc_page:
            # Anything printed before the arrangement of sections is front
            # matter: the title page and, in India Code prints, the numbered
            # list of amending Acts — which is numbered exactly like sections
            # and would otherwise start a phantom Section 1.
            content_type = "front_matter"
        elif cleaned and len(cleaned) < THIN_RESIDUE_CHARS and hindi_removed:
            # Column headers and rupee figures left behind by a dropped Hindi
            # table: syntactically English, semantically nothing.
            content_type = "thin_residue"
        elif not cleaned and hindi_removed:
            content_type = "hindi_only"
        elif not cleaned:
            content_type = "empty"
        else:
            content_type = "content"

        method = rec.get("extraction_method", "native")
        if cleaned:
            language_status = classify_language(cleaned, extraction_method=method)
        elif hindi_removed:
            language_status = (
                LANG_HI_OCR_PENDING if method == "ocr" else LANG_HI_CORRUPT
            )
        else:
            language_status = LANG_UNKNOWN

        entry = {
            "document_id": doc.document_id,
            "parent_source_document_id": doc.parent_source_document_id,
            "source_filename": doc.source_filename,
            "source_sha256": doc.source_sha256,
            "physical_page": rec["physical_page"],
            "extraction_method": method,
            "content_type": content_type,
            "language_status": language_status,
            "raw_text": rec["text"],
            "cleaned_text": cleaned,
            "raw_character_count": rec["character_count"],
            "cleaned_character_count": len(cleaned),
            "removed_boilerplate": removed,
            "amendment_notes": notes,
            "hindi_lines_removed": len(hindi_removed),
            "contains_table": detect_table(cleaned),
        }
        for key in ("ocr_mean_confidence", "ocr_lang", "native_extraction"):
            if key in rec:
                entry[key] = rec[key]
        out.append(entry)

    report = {
        "document_id": doc.document_id,
        "pages": len(out),
        "boilerplate_patterns": sorted(boilerplate),
        "table_of_contents_pages": sorted(toc_pages),
        "pages_by_content_type": dict(Counter(r["content_type"] for r in out)),
        "pages_by_language_status": dict(Counter(r["language_status"] for r in out)),
        "amendment_notes_extracted": sum(len(r["amendment_notes"]) for r in out),
        "hindi_lines_removed": sum(r["hindi_lines_removed"] for r in out),
        "pages_with_tables": sum(1 for r in out if r["contains_table"]),
        "characters_raw": sum(r["raw_character_count"] for r in out),
        "characters_cleaned": sum(r["cleaned_character_count"] for r in out),
    }
    return out, report


def run(verbose: bool = True) -> dict[str, Any]:
    os.makedirs(CLEANED_DIR, exist_ok=True)
    summary: dict[str, Any] = {"documents": {}}
    for doc in logical_documents():
        pages, report = clean_document(doc)
        write_jsonl(os.path.join(CLEANED_DIR, f"{doc.document_id}.jsonl"), pages)
        with open(
            os.path.join(CLEANED_DIR, f"{doc.document_id}.clean_report.json"), "w",
            encoding="utf-8",
        ) as fh:
            json.dump(report, fh, indent=2, ensure_ascii=False)
        summary["documents"][doc.document_id] = report
        if verbose:
            counts = report["pages_by_content_type"]
            print(
                f"clean {doc.document_id:30s} content={counts.get('content',0):4d} "
                f"toc={counts.get('table_of_contents',0):3d} hi={counts.get('hindi_only',0):3d} "
                f"empty={counts.get('empty',0):3d} notes={report['amendment_notes_extracted']:4d} "
                f"tables={report['pages_with_tables']:3d}",
                flush=True,
            )
    return summary


if __name__ == "__main__":
    run()
