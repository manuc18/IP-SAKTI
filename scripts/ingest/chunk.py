"""Stage 4 — structure-aware chunking.

A complete legal provision is the semantic unit: a whole Section, Rule,
Regulation or Article, with its provisos and Explanations attached, because a
proviso read apart from the provision it qualifies is worse than useless in
legal reasoning. Fixed-size windowing is never the primary strategy — it appears
only as the last resort inside a single sub-provision that is itself too long,
and even then the locator and page range travel with the fragment.

Schedules, Forms and Annexures are chunked as independent structural units,
which is how they are cited.
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

from ingest.citation import build_citation, disambiguate_citations  # noqa: E402
from ingest.clean import detect_table  # noqa: E402
from ingest.common import (  # noqa: E402
    CHUNKS_DIR,
    LANG_EN_CLEAN,
    STRUCTURED_DIR,
    LogicalDocument,
    classify_language,
    logical_documents,
    read_jsonl,
    write_jsonl,
)
from ingest.parsers.statutory import CLAUSE_RE  # noqa: E402
from ingest.provenance import citability, provenance_fields  # noqa: E402

#: Declared chunking version. Bumped whenever a change to the cleaning,
#: structural or chunking stages can alter a chunk's id, text or locator — which
#: makes every stored chunk id, embedding cache and benchmark result from an
#: earlier version non-comparable.
#:
#: 0.1.0  Phase 2 baseline.
#: 0.2.0  Phase 5. Treaty Article numbers are no longer stripped as page folios
#:        (CBD Articles 1-39, Nagoya Articles 1-20 recovered their locators);
#:        numbered entries inside a Schedule, Form or Annexure became
#:        `schedule_item` with an `item` locator instead of borrowing `section`.
CHUNKING_VERSION = "0.2.0"

#: Target ceiling for one chunk. Provisions below it are never split.
MAX_CHUNK_CHARS = 5_000
#: Anything shorter than this carries no retrievable content.
MIN_CHUNK_CHARS = 40
#: A bare "CHAPTER VI / OPPOSITION PROCEEDINGS" heading is scaffolding, not text.
MIN_HEADING_UNIT_CHARS = 200

SUB_LOCATOR_KEY = {
    "subsection": "subsection",
    "sub_rule": "sub_rule",
    "paragraph": "paragraph",
}

#: Locator keys that identify a provision in its own right. A sub-locator —
#: sub-section, sub-rule, paragraph — is a position *inside* one of these, so it
#: may only be attached to a chunk whose unit already carries one. Without that
#: rule a numbered line anywhere becomes a citation: the CBD's front matter reads
#: "3. Environmental protection — International cooperation", and splitting the
#: unlocated preamble on it produced a chunk citable as "CBD — Paragraph 3",
#: which is not a provision of the Convention at all.
PROVISION_LOCATOR_KEYS = ("section", "rule", "regulation", "article", "paragraph", "item")


def slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "-", value).strip("-").lower()
    return slug or "unit"


def _split_lines(lines: list[list[Any]], limit: int) -> list[list[list[Any]]]:
    """Last-resort split of an over-long block at line boundaries."""
    groups: list[list[list[Any]]] = []
    current: list[list[Any]] = []
    size = 0
    for line in lines:
        length = len(line[1]) + 1
        if current and size + length > limit:
            groups.append(current)
            current, size = [], 0
        current.append(line)
        size += length
    if current:
        groups.append(current)
    return groups


def _clause_groups(lines: list[list[Any]], limit: int) -> list[tuple[list[list[Any]], str | None]]:
    """Split a too-long sub-provision at its clause boundaries, then by lines."""
    boundaries = [index for index, line in enumerate(lines) if CLAUSE_RE.match(line[1])]
    if not boundaries:
        return [(group, None) for group in _split_lines(lines, limit)]
    starts = sorted({0, *boundaries})
    blocks: list[tuple[list[list[Any]], str | None]] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        block = lines[start:end]
        if not block:
            continue
        match = CLAUSE_RE.match(block[0][1])
        blocks.append((block, match.group(1) if match else None))

    out: list[tuple[list[list[Any]], str | None]] = []
    current: list[list[Any]] = []
    marker: str | None = None
    size = 0
    for block, block_marker in blocks:
        length = sum(len(line[1]) + 1 for line in block)
        if length > limit:
            # A single clause longer than the ceiling: nothing structural is
            # left to split on, so fall back to line boundaries.
            if current:
                out.append((current, marker))
                current, marker, size = [], None, 0
            out.extend((piece, block_marker) for piece in _split_lines(block, limit))
            continue
        if current and size + length > limit:
            out.append((current, marker))
            current, marker, size = [], None, 0
        if not current:
            marker = block_marker
        current.extend(block)
        size += length
    if current:
        out.append((current, marker))
    return out


def split_unit(unit: dict[str, Any]) -> list[tuple[list[list[Any]], dict[str, str | None]]]:
    """Split one structural unit into chunk-sized pieces with sub-locators."""
    lines = unit["lines"]
    text_length = unit["character_count"]
    if text_length <= MAX_CHUNK_CHARS:
        return [(lines, {})]

    subdivisions = unit.get("subdivisions") or []
    if not subdivisions:
        return [(group, {}) for group, _ in _clause_groups(lines, MAX_CHUNK_CHARS)]

    starts = sorted({0, *(index for index, _, _ in subdivisions)})
    markers = {index: (kind, marker) for index, kind, marker in subdivisions}
    blocks: list[tuple[list[list[Any]], dict[str, str | None]]] = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        block = lines[start:end]
        if not block:
            continue
        locator: dict[str, str | None] = {}
        if start in markers:
            kind, marker = markers[start]
            locator[SUB_LOCATOR_KEY.get(kind, "subsection")] = marker
        blocks.append((block, locator))

    out: list[tuple[list[list[Any]], dict[str, str | None]]] = []
    current: list[list[Any]] = []
    locator: dict[str, str | None] = {}
    size = 0
    for block, block_locator in blocks:
        length = sum(len(line[1]) + 1 for line in block)
        if length > MAX_CHUNK_CHARS:
            if current:
                out.append((current, locator))
                current, locator, size = [], {}, 0
            for piece, clause in _clause_groups(block, MAX_CHUNK_CHARS):
                piece_locator = dict(block_locator)
                if clause:
                    piece_locator["clause"] = clause
                out.append((piece, piece_locator))
            continue
        if current and size + length > MAX_CHUNK_CHARS:
            out.append((current, locator))
            current, locator, size = [], {}, 0
        if not current:
            locator = dict(block_locator)
        current.extend(block)
        size += length
    if current:
        out.append((current, locator))
    return out


def chunk_document(doc: LogicalDocument) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    units = read_jsonl(os.path.join(STRUCTURED_DIR, f"{doc.document_id}.jsonl"))
    chunks: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_ids: Counter[str] = Counter()
    provenance = provenance_fields(doc.document_id)

    for unit in units:
        if unit["unit_type"] == "heading" and unit["character_count"] < MIN_HEADING_UNIT_CHARS:
            rejected.append(
                {
                    "document_id": doc.document_id,
                    "unit_order": unit["unit_order"],
                    "reason": "structural heading with no body text",
                    "character_count": unit["character_count"],
                }
            )
            continue

        pieces = split_unit(unit)
        for index, (lines, sub_locator) in enumerate(pieces):
            text = re.sub(r"\n{3,}", "\n\n", "\n".join(line[1] for line in lines)).strip()
            pages = sorted({line[0] for line in lines})
            if len(text) < MIN_CHUNK_CHARS:
                rejected.append(
                    {
                        "document_id": doc.document_id,
                        "unit_order": unit["unit_order"],
                        "reason": "below the minimum retrievable length",
                        "character_count": len(text),
                    }
                )
                continue
            language_status = classify_language(text)
            if language_status != LANG_EN_CLEAN:
                rejected.append(
                    {
                        "document_id": doc.document_id,
                        "unit_order": unit["unit_order"],
                        "reason": f"non-English retrieval text ({language_status})",
                        "character_count": len(text),
                    }
                )
                continue

            locator = dict(unit["locator"])
            if any(locator.get(key) for key in PROVISION_LOCATOR_KEYS):
                locator.update(sub_locator)
            citation, unavailable = build_citation(
                doc.short_title,
                locator,
                citation_style=doc.citation_style,
                unit_type=unit["unit_type"],
                pages=pages,
            )
            methods = unit.get("extraction_methods") or ["native"]
            extraction_method = methods[0] if len(methods) == 1 else "mixed"

            # A unit with no structural path has no locator, so its id is built
            # from its position. The ``u`` matters: without it an unnumbered
            # Article at unit_order 8 would slug to ``article-8`` and collide
            # with the chunk for the real Article 8 — an id that reads like a
            # citation for text that has none.
            base = unit["structural_path"] or f"{unit['unit_type']}-u{unit['unit_order']}"
            slug = slugify(base)
            if len(pieces) > 1:
                slug = f"{slug}-c{index + 1}"
            chunk_id = f"{doc.document_id}::{slug}"
            seen_ids[chunk_id] += 1
            if seen_ids[chunk_id] > 1:
                chunk_id = f"{chunk_id}-{seen_ids[chunk_id]}"

            retrievable, citable, citability_reason = citability(
                locator_unavailable=unavailable
            )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "chunking_version": CHUNKING_VERSION,
                    "document_id": doc.document_id,
                    "parent_source_document_id": doc.parent_source_document_id,
                    "title": doc.title,
                    "short_title": doc.short_title,
                    "domain": doc.domain,
                    "jurisdiction": doc.jurisdiction,
                    "document_type": doc.document_type,
                    "authority": doc.authority,
                    "language": "en",
                    "language_status": language_status,
                    "text": text,
                    "character_count": len(text),
                    "physical_pages": pages,
                    "locator": locator,
                    "locator_unavailable": unavailable,
                    "citation": citation,
                    "structural_path": unit["structural_path"],
                    "unit_type": unit["unit_type"],
                    "heading": unit["heading"],
                    "chunk_index": index + 1,
                    "chunk_count": len(pieces),
                    "amendment_notes": unit["amendment_notes"],
                    "extraction_method": extraction_method,
                    "contains_table": detect_table(text),
                    "source_filename": doc.source_filename,
                    "source_sha256": doc.source_sha256,
                    "version": doc.version,
                    "reference_number": doc.reference_number,
                    "source_url": doc.source_url,
                    "provenance_status": doc.provenance_status,
                    "retrievable": retrievable,
                    "citable": citable,
                    "citability_reason": citability_reason,
                    **provenance,
                }
            )

    # A citation is only useful if it separates. Two provisions in different
    # Parts of the same instrument can carry the same rule number, and until
    # this pass they rendered the same string — see `disambiguate_citations`.
    # It runs here rather than inside the loop because a collision is a property
    # of the document, not of any one chunk.
    citations_disambiguated = disambiguate_citations(
        chunks, short_title=doc.short_title, citation_style=doc.citation_style)

    report = {
        "document_id": doc.document_id,
        "chunking_version": CHUNKING_VERSION,
        "units": len(units),
        "chunks": len(chunks),
        "citations_disambiguated": citations_disambiguated,
        "rejected": len(rejected),
        "rejected_by_reason": dict(Counter(r["reason"] for r in rejected)),
        "chunks_missing_locator": sum(1 for c in chunks if c["locator_unavailable"]),
        "chunks_citable": sum(1 for c in chunks if c["citable"]),
        "chunks_not_citable": sum(1 for c in chunks if not c["citable"]),
        "source_evidence_level": provenance["source_evidence_level"],
        "source_status": provenance["source_status"],
        "chunks_with_tables": sum(1 for c in chunks if c["contains_table"]),
        "chunks_with_amendment_notes": sum(1 for c in chunks if c["amendment_notes"]),
        "chunks_from_ocr": sum(1 for c in chunks if c["extraction_method"] != "native"),
        "characters": sum(c["character_count"] for c in chunks),
        "mean_chunk_characters": round(
            sum(c["character_count"] for c in chunks) / len(chunks), 1
        ) if chunks else 0,
        "max_chunk_characters": max((c["character_count"] for c in chunks), default=0),
    }
    return chunks, rejected, report


def run(verbose: bool = True) -> dict[str, Any]:
    os.makedirs(CHUNKS_DIR, exist_ok=True)
    summary: dict[str, Any] = {"documents": {}}
    for doc in logical_documents():
        chunks, rejected, report = chunk_document(doc)
        write_jsonl(os.path.join(CHUNKS_DIR, f"{doc.document_id}.jsonl"), chunks)
        with open(
            os.path.join(CHUNKS_DIR, f"{doc.document_id}.chunk_report.json"), "w",
            encoding="utf-8",
        ) as fh:
            json.dump({**report, "rejected_chunks": rejected}, fh, indent=2, ensure_ascii=False)
        summary["documents"][doc.document_id] = report
        if verbose:
            print(
                f"chunk {doc.document_id:30s} chunks={report['chunks']:4d} "
                f"rejected={report['rejected']:3d} no_locator={report['chunks_missing_locator']:3d} "
                f"mean={report['mean_chunk_characters']:7.1f} max={report['max_chunk_characters']:6d}",
                flush=True,
            )
    return summary


if __name__ == "__main__":
    run()
