"""Compact packets for the Claude Code judge, with the rank hidden.

    .venv/bin/python scripts/evaluation/build_judge_packets.py --sample 150

A packet carries the question, the expected answer, and the five retrieved
chunks — and nothing else. Not the corpus, not the reranker's working, not the
benchmark's own label, and above all **not the rank**.

Why the rank is hidden
----------------------

The judge is being asked whether a set of evidence supports an expected answer.
If the packet said "this one came first", the judge would be told the answer to
half the question before reading it, and every agreement between the ranker and
the judge would be unfalsifiable. So the five chunks are presented as A-E in an
order shuffled by a hash of the question id — deterministic, reproducible from
the id alone, and uncorrelated with the ranking.

The judge names the letters that carry the support. `record_judgements.py` maps
letters back to ranks afterwards, which is how `top1_supported` is computed
without the judge ever having seen which one rank 1 was.

Sampling
--------

Stratified over difficulty x whether the target was retrieved at rank 1, so the
sample contains the failures in proportion rather than by luck, and the strata
counts are written into the manifest as denominators.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai.retrieval.base import load_chunks  # noqa: E402

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation", "results", "submission")
ROWS_PATH = os.path.join(SUBMISSION_DIR, "retrieval_1000.jsonl")

#: Characters of each chunk shown to the judge. Long enough to carry a
#: provision, short enough that the sample fits in one reading session.
EXCERPT_CHARS = 700
#: Characters of the expected answer shown. The expected answers are verbatim
#: spans and some run to several hundred words; the judge needs the claim, not
#: the whole provision twice.
EXPECTED_CHARS = 450
LETTERS = "ABCDE"


def shuffle_order(question_id: str, count: int) -> list[int]:
    """A deterministic permutation of 0..count-1, derived from the id alone."""
    keyed = sorted(range(count),
                   key=lambda i: hashlib.sha256(
                       f"{question_id}:{i}".encode("utf-8")).hexdigest())
    return keyed


def truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " ..."


def stratify(rows: Sequence[dict[str, Any]], sample: int) -> list[dict[str, Any]]:
    """Proportional over (difficulty, rank-1 hit), stable within each stratum."""
    strata: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        rank = row["rank_of_target"]
        bucket = "rank1" if rank == 1 else ("rank2_5" if rank and rank <= 5
                                            else ("rank6_10" if rank and rank <= 10
                                                  else "missed"))
        strata[(row["difficulty"], bucket)].append(row)
    total = len(rows)
    picked: list[dict[str, Any]] = []
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda r: r["question_id"])
        take = max(1, round(sample * len(group) / total))
        step = max(1, len(group) // take)
        picked.extend(group[::step][:take])
    return sorted(picked, key=lambda r: r["question_id"])[:sample]


def build(rows: Sequence[dict[str, Any]], texts: dict[str, str],
          sample: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    answerable = [r for r in rows if r["label_kind"] == "ANSWERABLE"]
    chosen = stratify(answerable, sample) if sample else answerable

    packets: list[dict[str, Any]] = []
    key: list[dict[str, Any]] = []
    for row in chosen:
        top5 = row["top5_chunk_ids"]
        order = shuffle_order(row["question_id"], len(top5))
        items = []
        mapping = {}
        for slot, source_index in enumerate(order):
            chunk_id = top5[source_index]
            evidence = next((e for e in row["evidence"] if e["chunk_id"] == chunk_id), {})
            letter = LETTERS[slot]
            mapping[letter] = {"rank": source_index + 1, "chunk_id": chunk_id}
            items.append({
                "label": letter,
                "source": evidence.get("citation") or chunk_id,
                "heading": evidence.get("heading"),
                "page": (evidence.get("pages") or [None])[0],
                "text": truncate(texts.get(chunk_id, ""), EXCERPT_CHARS),
            })
        packets.append({
            "question_id": row["question_id"],
            "question": row["question"],
            "expected_answer": truncate(row["expected_answer"], EXPECTED_CHARS),
            "evidence": items,
        })
        key.append({"question_id": row["question_id"], "letter_to_rank": mapping,
                    "target_chunk_ids": row["target_chunk_ids"],
                    "acceptable_chunk_ids": row["acceptable_chunk_ids"],
                    "rank_of_target": row["rank_of_target"],
                    "difficulty": row["difficulty"],
                    "answer_type": row["answer_type"],
                    "domain": row["domain"]})

    counts: collections.Counter = collections.Counter()
    for row in chosen:
        rank = row["rank_of_target"]
        counts["rank1" if rank == 1 else ("rank2_5" if rank and rank <= 5
                                          else ("rank6_10" if rank and rank <= 10
                                                else "missed"))] += 1
    manifest = {
        "packets": len(packets),
        "population": len(answerable),
        "stratified_by": ["difficulty", "rank bucket of the target"],
        "strata_counts": dict(counts),
        "rank_hidden_from_judge": True,
        "evidence_order": "sha256(question_id:index), deterministic and "
                          "uncorrelated with rank",
        "excerpt_chars": EXCERPT_CHARS,
        "judge_provider": "claude_code_session",
        "claim_safety": ("AI audit of retrieved evidence against an "
                         "AI-generated expected answer. Not independent legal "
                         "verification and not human-verified ground truth."),
    }
    return packets, {"manifest": manifest, "key": key}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default=ROWS_PATH)
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--out", default=SUBMISSION_DIR)
    args = parser.parse_args(argv)

    with open(args.rows, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    texts = {chunk.chunk_id: chunk.text for chunk in load_chunks()}

    packets, side = build(rows, texts, args.sample)
    os.makedirs(args.out, exist_ok=True)
    packets_path = os.path.join(args.out, "judge_packets.jsonl")
    key_path = os.path.join(args.out, "judge_key.json")
    with open(packets_path, "w", encoding="utf-8") as handle:
        for packet in packets:
            handle.write(json.dumps(packet, ensure_ascii=False) + "\n")
    with open(key_path, "w", encoding="utf-8") as handle:
        json.dump(side, handle, indent=2, ensure_ascii=False)

    print(f"{len(packets)} packets -> {packets_path}")
    print(f"strata: {side['manifest']['strata_counts']}")
    print(f"key -> {key_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
