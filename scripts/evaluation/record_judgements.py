"""Split packets into batches, and fold the judge's answers back into metrics.

    # 1. write the batches a judging session reads
    .venv/bin/python scripts/evaluation/record_judgements.py --split 14

    # 2. after the batches have been judged and their .jsonl written
    .venv/bin/python scripts/evaluation/record_judgements.py --collect

The judge never sees a rank (see `build_judge_packets.py`), so it answers in
letters: which of A-E carry the support. This module holds the only copy of the
letter-to-rank map and applies it afterwards, which is what makes
`top1_supported` a measurement rather than a leading question.

A judgement is a line of JSON with these fields, and nothing is inferred for a
field the judge did not fill in:

``expected_answer_supported``  yes | partial | no
``supporting_labels``          the letters that carry it, possibly empty
``sufficient``                 yes | no | uncertain — could a reader answer from
                               the five chunks alone
``unsupported_claim``          true when the expected answer asserts something
                               no chunk states
``contradiction``              true when a chunk contradicts the expected answer
``missing_key_requirement``    true when the answer needs an element the
                               evidence does not carry
``note``                       one line, free text

Batches are independent and each writes its own file, so a batch that fails
costs that batch. `--collect` refuses to invent a verdict for a packet no batch
covered; it reports the shortfall in the denominator instead.

Later batches override earlier ones for the same question id, and that is used
for exactly one thing. The packets show a truncated excerpt of each chunk, and
on a handful of questions the truncation fell mid-provision — the judge was
shown a chunk that stopped before the sub-section the expected answer quoted and
correctly reported it absent. That is a defect in the packet, not in retrieval.
Those questions were re-issued in a final batch carrying the **full** chunk text
and their verdicts replace the truncated ones. The re-issue set is chosen by a
rule that cannot launder a real failure: only questions whose target chunk was
actually inside the top 5, where "the evidence does not contain it" and "the
packet did not show it" are the only two explanations and the second is
testable. Questions whose target was never retrieved are never re-issued.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
from typing import Any, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SUBMISSION_DIR = os.path.join(PROJECT_ROOT, "data", "evaluation", "results", "submission")
PACKETS_PATH = os.path.join(SUBMISSION_DIR, "judge_packets.jsonl")
KEY_PATH = os.path.join(SUBMISSION_DIR, "judge_key.json")
BATCH_DIR = os.path.join(SUBMISSION_DIR, "judge_batches")

JUDGE_PROVIDER = "claude_code_session"

INSTRUCTIONS = """\
You are judging retrieved legal evidence against an expected answer.

For each packet below, decide ONLY from the evidence shown. Do not use outside
legal knowledge. Do not guess at wording that is not present. The evidence
chunks are labelled A-E in no meaningful order.

Write one JSON object per packet, one per line, to the output file:

{"question_id": "...", "expected_answer_supported": "yes|partial|no",
 "supporting_labels": ["A"], "sufficient": "yes|no|uncertain",
 "unsupported_claim": false, "contradiction": false,
 "missing_key_requirement": false, "note": "one short line"}

Definitions:
- expected_answer_supported: does some chunk actually state what the expected
  answer says? "partial" when part of it is stated and part is not.
- supporting_labels: every label whose text carries that support. Empty list if
  none.
- sufficient: could a careful reader answer the question from these five chunks
  alone, without more of the statute?
- unsupported_claim: the expected answer asserts something no chunk states.
- contradiction: a chunk says something incompatible with the expected answer.
- missing_key_requirement: the answer needs an element (a time limit, a fee, a
  condition) that the evidence does not carry.

Judge every packet. Do not skip one because it is hard; mark it uncertain.
"""


def load_packets(path: str) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def render(packet: dict[str, Any]) -> str:
    lines = [f"### {packet['question_id']}",
             f"QUESTION: {packet['question']}",
             f"EXPECTED ANSWER: {packet['expected_answer']}",
             "EVIDENCE:"]
    for item in packet["evidence"]:
        head = f"  [{item['label']}] {item['source']}"
        if item.get("heading"):
            head += f" — {item['heading']}"
        if item.get("page"):
            head += f" (p.{item['page']})"
        lines.append(head)
        lines.append(f"      {item['text']}")
    return "\n".join(lines)


def split(packets: Sequence[dict[str, Any]], size: int) -> int:
    os.makedirs(BATCH_DIR, exist_ok=True)
    for stale in glob.glob(os.path.join(BATCH_DIR, "batch_*.txt")):
        os.remove(stale)
    count = 0
    for index in range(0, len(packets), size):
        group = packets[index:index + size]
        count += 1
        path = os.path.join(BATCH_DIR, f"batch_{count:02d}.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(INSTRUCTIONS)
            handle.write(f"\nOutput file: {BATCH_DIR}/batch_{count:02d}.jsonl\n")
            handle.write(f"Packets in this batch: {len(group)}\n\n")
            handle.write("\n\n".join(render(p) for p in group))
            handle.write("\n")
    return count


def collect(packets: Sequence[dict[str, Any]], key: dict[str, Any]) -> dict[str, Any]:
    verdicts: dict[str, dict[str, Any]] = {}
    for path in sorted(glob.glob(os.path.join(BATCH_DIR, "batch_*.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    verdicts[row["question_id"]] = row

    by_id = {entry["question_id"]: entry for entry in key["key"]}
    rows: list[dict[str, Any]] = []
    for packet in packets:
        qid = packet["question_id"]
        verdict = verdicts.get(qid)
        if verdict is None:
            continue
        mapping = by_id[qid]["letter_to_rank"]
        labels = verdict.get("supporting_labels") or []
        ranks = sorted(mapping[l]["rank"] for l in labels if l in mapping)
        rows.append({
            **verdict,
            "supporting_ranks": ranks,
            "top1_supported": 1 in ranks,
            "top5_supported": bool(ranks),
            "rank_of_target": by_id[qid]["rank_of_target"],
            "difficulty": by_id[qid]["difficulty"],
            "answer_type": by_id[qid]["answer_type"],
            "domain": by_id[qid]["domain"],
            "judge_provider": JUDGE_PROVIDER,
        })

    n = len(rows)
    tally = collections.Counter(r["expected_answer_supported"] for r in rows)
    suff = collections.Counter(r["sufficient"] for r in rows)
    summary = {
        "judge_provider": JUDGE_PROVIDER,
        "judgements": n,
        "packets_issued": len(packets),
        "packets_unjudged": len(packets) - n,
        "population": key["manifest"]["population"],
        "expected_answer_supported": {
            "yes": tally["yes"], "partial": tally["partial"], "no": tally["no"],
            "denominator": n},
        "evidence_sufficient_judged": {
            "yes": suff["yes"], "no": suff["no"], "uncertain": suff["uncertain"],
            "denominator": n},
        "top1_supported": {"count": sum(1 for r in rows if r["top1_supported"]),
                           "denominator": n},
        "top5_supported": {"count": sum(1 for r in rows if r["top5_supported"]),
                           "denominator": n},
        "unsupported_claim": {"count": sum(1 for r in rows if r.get("unsupported_claim")),
                              "denominator": n},
        "contradiction": {"count": sum(1 for r in rows if r.get("contradiction")),
                          "denominator": n},
        "missing_key_requirement": {
            "count": sum(1 for r in rows if r.get("missing_key_requirement")),
            "denominator": n},
        "agreement_with_deterministic_rank": {
            "judge_supports_top1_and_target_was_rank1": sum(
                1 for r in rows if r["top1_supported"] and r["rank_of_target"] == 1),
            "judge_supports_top1_but_target_was_not_rank1": sum(
                1 for r in rows if r["top1_supported"] and r["rank_of_target"] != 1),
            "note": "the second figure counts questions where the judge found "
                    "rank 1 supporting although the benchmark's single label sat "
                    "lower; those are labelling width, not ranking failure",
        },
        "rank_hidden_from_judge": True,
        "claim_safety": key["manifest"]["claim_safety"],
    }
    return {"summary": summary, "rows": rows}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--collect", action="store_true")
    args = parser.parse_args(argv)

    packets = load_packets(PACKETS_PATH)
    if args.split:
        count = split(packets, args.split)
        print(f"{len(packets)} packets -> {count} batches in {BATCH_DIR}")
        return 0
    if args.collect:
        with open(KEY_PATH, encoding="utf-8") as handle:
            key = json.load(handle)
        result = collect(packets, key)
        out_rows = os.path.join(SUBMISSION_DIR, "judgements.jsonl")
        out_summary = os.path.join(SUBMISSION_DIR, "judgements.summary.json")
        with open(out_rows, "w", encoding="utf-8") as handle:
            for row in result["rows"]:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        with open(out_summary, "w", encoding="utf-8") as handle:
            json.dump(result["summary"], handle, indent=2, ensure_ascii=False)
        s = result["summary"]
        print(f"judged {s['judgements']}/{s['packets_issued']} packets "
              f"(population {s['population']})")
        print(f"  supported yes/partial/no: {s['expected_answer_supported']}")
        print(f"  sufficient: {s['evidence_sufficient_judged']}")
        print(f"  top1 supported: {s['top1_supported']}")
        print(f"  unsupported claims: {s['unsupported_claim']}")
        print(f"-> {out_summary}")
        return 0

    parser.error("pass --split N or --collect")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
