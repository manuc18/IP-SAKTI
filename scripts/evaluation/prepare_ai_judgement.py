"""Pass A: build the AI judge's evidence packets. Phase 8.1, Parts 4, 9 and 10.

    .venv/bin/python scripts/evaluation/prepare_ai_judgement.py

Offline. No network, no model, no retrieval, no embeddings. Reads
`golden_questions.jsonl` and the processed chunks, and writes one bounded
evidence packet per question to `data/evaluation/ai_judge_packets.jsonl`.

**The packets are byte-stable.** Nothing in a packet is a timestamp and nothing
is a random draw, so running this twice over an unchanged corpus produces an
identical file, and each packet's `packet_sha256` is a fingerprint a later
judgement can be checked against. That is what makes "the judge saw exactly this
evidence" a verifiable claim rather than an assurance.

**What the judge is not given** matters as much as what it is. Three things are
withheld on purpose:

* `verified` — the judge is auditing the label, not being handed its status;
* the benchmark author's `notes` — that field is the label author's own
  justification for the label, and giving the auditor the defence's argument is
  the circularity Part 8 forbids;
* every retrieval and reranking result in the project — all of them were scored
  against the very label under audit.

The neighbouring passages come from document order, not from a ranked list, for
that last reason. It is a real bound: the judge can find a better target beside
the designated one and cannot find one in a different instrument. Part 5 of the
prompt tells it to answer AI_NEEDS_REVIEW in that case rather than guess.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai.retrieval.base import Chunk, load_chunks  # noqa: E402
from scripts.evaluation.prepare_ground_truth import (  # noqa: E402
    EVALUATION_DIR,
    corpus_fingerprint,
    read_jsonl,
    write_jsonl,
)
from scripts.evaluation.validate_golden_dataset import GOLDEN_PATH, load_golden  # noqa: E402

PACKETS_PATH = os.path.join(EVALUATION_DIR, "ai_judge_packets.jsonl")

#: The frozen provenance manifest. Written once beside the packets and treated
#: as read-only from then on: it is what lets a judgement made months later be
#: checked against exactly the evidence, corpus and benchmark version it claims.
#: Recomputing the hashes at judging time would only prove the code agrees with
#: itself; comparing against a frozen file is what detects that something moved.
MANIFEST_PATH = os.path.join(EVALUATION_DIR, "ai_judge_manifest.json")

PACKET_VERSION = "1.0.0"

#: How many chunks either side of the target, within the same instrument, the
#: judge is shown. Two is enough to see the provision before and after — which
#: is where a mis-designated locator almost always actually lives — without
#: turning the packet into a corpus dump the judge would skim.
NEIGHBOUR_SPAN = 2

#: Golden fields that must never reach a packet. `verified` and
#: `ground_truth_method` state the label's own provenance; `notes` argues for it.
WITHHELD_FROM_JUDGE = ("verified", "ground_truth_method", "notes")


def golden_sha256(path: str = GOLDEN_PATH) -> str:
    """The dataset under audit, identified by content.

    Recorded on every packet and every judgement. A judgement of Q027 is a
    judgement of one specific question text and one specific set of expected
    chunks; if either changes, the judgement is about something else.
    """
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def canonical_sha256(payload: Any) -> str:
    """Content hash of a JSON-serialisable value, stable across runs."""
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def neighbour_ids(chunk_id: str, ordered_ids: list[str],
                  span: int = NEIGHBOUR_SPAN) -> list[str]:
    """Structural neighbours: same instrument, adjacent in document order.

    Document order, not rank order. Every ranked list this project can produce
    was scored against the label being audited.
    """
    if chunk_id not in ordered_ids:
        return []
    document = chunk_id.split("::", 1)[0]
    index = ordered_ids.index(chunk_id)
    window = (ordered_ids[max(0, index - span):index]
              + ordered_ids[index + 1:index + 1 + span])
    return [cid for cid in window if cid.split("::", 1)[0] == document]


def passage(chunk: Chunk, role: str) -> dict[str, Any]:
    """One passage as the judge sees it.

    `locator_fields` is filtered to the keys that are actually set: a judge
    reading fifteen nulls to find `"rule": "23"` is being asked to do the
    parser's job.
    """
    return {
        "chunk_id": chunk.chunk_id,
        "role": role,
        "document_id": chunk.document_id,
        "citation": chunk.citation,
        "locator_fields": {k: v for k, v in chunk.locator.items() if v is not None},
        "unit_type": chunk.unit_type,
        "heading": chunk.heading,
        "temporal_status": chunk.temporal_status,
        "source_evidence_level": chunk.source_evidence_level,
        "source_status": chunk.source_status,
        "citable": chunk.citable,
        "locator_unavailable": chunk.locator_unavailable,
        "physical_pages": list(chunk.physical_pages),
        "text": chunk.text,
    }


def build_packet(record: dict[str, Any], by_id: dict[str, Chunk],
                 ordered_ids: list[str], fingerprint: str,
                 golden_hash: str) -> dict[str, Any]:
    """One bounded evidence packet. Deterministic; contains no timestamp."""
    targets = list(record["expected_chunks"])
    evidence: list[dict[str, Any]] = []
    seen: set[str] = set()

    for chunk_id in targets:
        chunk = by_id.get(chunk_id)
        if chunk is None:
            # Recorded rather than dropped. A packet whose target is missing is
            # exactly the case the judge should meet as AI_NEEDS_REVIEW, and
            # silently omitting it would present a target-less question as
            # though the corpus simply had nothing to say.
            evidence.append({"chunk_id": chunk_id, "role": "target",
                             "missing_from_corpus": True})
            continue
        evidence.append(passage(chunk, "target"))
        seen.add(chunk_id)

    for chunk_id in targets:
        for neighbour in neighbour_ids(chunk_id, ordered_ids):
            if neighbour in seen or neighbour in targets:
                continue
            evidence.append(passage(by_id[neighbour], "neighbour"))
            seen.add(neighbour)

    body = {
        "question_id": record["question_id"],
        "packet_version": PACKET_VERSION,
        "golden_dataset_sha256": golden_hash,
        "corpus_fingerprint": fingerprint,
        "question": {
            "text": record["query"],
            "query_family": record.get("query_family"),
            "domain": record.get("domain"),
            "jurisdiction": record.get("jurisdiction"),
        },
        # The claim under audit, labelled as a claim.
        "benchmark_claim": {
            "document_id": record["expected_documents"][0]
            if record["expected_documents"] else None,
            "all_document_ids": list(record["expected_documents"]),
            "locator": record["expected_locators"][0]
            if record["expected_locators"] else None,
            "all_locators": list(record["expected_locators"]),
            "chunk_ids": targets,
            "anchor_phrases": list(record.get("anchor_terms", [])),
            "requires_citation": record.get("requires_citation"),
            "allow_non_citable": record.get("allow_non_citable"),
        },
        "evidence": evidence,
        "evidence_source": "the designated chunks plus their structural "
                           "neighbours in the same instrument, taken in document "
                           "order; no retrieval or reranking output was used",
    }
    return dict(body, packet_sha256=canonical_sha256(body))


def build_all(records: list[dict[str, Any]], chunks: list[Chunk],
              golden_hash: str) -> list[dict[str, Any]]:
    by_id = {chunk.chunk_id: chunk for chunk in chunks}
    ordered_ids = [chunk.chunk_id for chunk in chunks]
    fingerprint = corpus_fingerprint(chunks)
    return [build_packet(record, by_id, ordered_ids, fingerprint, golden_hash)
            for record in records]


def audit_packets(packets: list[dict[str, Any]]) -> list[str]:
    """Refuse to ship a packet that leaks the label's own provenance."""
    problems: list[str] = []
    for packet in packets:
        blob = json.dumps(packet, ensure_ascii=False)
        for field in WITHHELD_FROM_JUDGE:
            if f'"{field}"' in blob:
                problems.append(f"{packet['question_id']}: withheld field "
                                f"{field!r} reached the packet")
        if packet["packet_sha256"] != canonical_sha256(
                {k: v for k, v in packet.items() if k != "packet_sha256"}):
            problems.append(f"{packet['question_id']}: packet hash does not match")
        if not any(item.get("role") == "target" for item in packet["evidence"]):
            problems.append(f"{packet['question_id']}: no target passage")
    return problems


def build_manifest(packets: list[dict[str, Any]], golden_hash: str,
                   fingerprint: str) -> dict[str, Any]:
    """The frozen record of what the judge is allowed to have seen."""
    return {
        "manifest_version": "1.0.0",
        "packet_version": PACKET_VERSION,
        "golden_dataset_sha256": golden_hash,
        "corpus_fingerprint": fingerprint,
        "packet_count": len(packets),
        "packet_sha256": {p["question_id"]: p["packet_sha256"]
                          for p in sorted(packets, key=lambda p: p["question_id"])},
    }


def summarise(packets: list[dict[str, Any]]) -> dict[str, Any]:
    sizes = [len(json.dumps(p, ensure_ascii=False)) for p in packets]
    passages = [len(p["evidence"]) for p in packets]
    return {
        "packets": len(packets),
        "packet_version": PACKET_VERSION,
        "passages_per_packet": {"min": min(passages), "max": max(passages),
                                "mean": round(sum(passages) / len(passages), 1)},
        "packet_chars": {"min": min(sizes), "max": max(sizes),
                         "mean": round(sum(sizes) / len(sizes))},
        "missing_targets": sum(
            1 for p in packets
            for e in p["evidence"] if e.get("missing_from_corpus")),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--golden", default=GOLDEN_PATH)
    parser.add_argument("--packets", default=PACKETS_PATH)
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    records = load_golden(args.golden)
    chunks = load_chunks()
    golden_hash = golden_sha256(args.golden)
    packets = build_all(records, chunks, golden_hash)

    problems = audit_packets(packets)
    if problems:
        for problem in problems:
            print(f"REFUSED {problem}")
        return 1

    manifest = build_manifest(packets, golden_hash, corpus_fingerprint(chunks))
    if not args.dry_run:
        write_jsonl(args.packets, packets)
        # Written only if absent. A manifest that silently re-freezes whatever
        # is on disk today is not a seatbelt; it is a rubber stamp.
        if os.path.exists(args.manifest):
            existing = json.load(open(args.manifest, encoding="utf-8"))
            if existing != manifest:
                print("REFUSED the frozen manifest disagrees with the packets "
                      "just built; delete it deliberately to re-freeze")
                return 1
        else:
            with open(args.manifest, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, indent=2, sort_keys=True)
                fh.write("\n")

    summary = summarise(packets)
    summary["golden_dataset_sha256"] = golden_hash
    summary["corpus_fingerprint"] = corpus_fingerprint(chunks)
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        for key, value in summary.items():
            print(f"{key:24s} {value}")
        if args.dry_run:
            print("(dry run - nothing written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
