"""Phase H, Metric B: judge packets for answer quality, and the scoring scheme.

    # build packets from an answer run
    .venv/bin/python scripts/evaluation/prepare_synthetic_judgement.py \
        --answers data/evaluation/results/synthetic/answers.jsonl

    # score judgements a Claude Code session wrote back
    .venv/bin/python scripts/evaluation/prepare_synthetic_judgement.py \
        --score data/evaluation/results/synthetic/judgements.jsonl

The judge is the Claude Code session, as it is for Phase 8 — see
`scripts/evaluation/record_claude_code_judgement.py`, whose provenance
conventions this module follows rather than reinvents. No Anthropic API key is
used and no socket is opened by anything here.

What the judge is not given, and why each omission matters
----------------------------------------------------------

A packet carries exactly four things: QUESTION, EXPECTED ANSWER, PIPELINE
ANSWER, CITED EVIDENCE. Everything else is withheld by `WITHHELD`, and the
withholding is asserted rather than intended — `build_packet` raises if a
forbidden key survives into a packet.

``rank_of_expected``      Telling the judge the pipeline found its target at
                          rank 1 tells it the answer is probably right. That is
                          Metric A leaking into Metric B, and it would make the
                          two numbers agree by construction.
``target_chunk_ids``      Same leak by a different route: a judge that can match
                          chunk ids is comparing identifiers, not reasoning about
                          whether the answer is supported.
``rerank_score``, scores  A confidence signal the judge has no business seeing;
                          it invites deference to the pipeline's own opinion.
``generation_metadata``   Names the template that wrote the question, which
                          telegraphs the expected shape of the answer.
``difficulty``            An expectation about how well the system should do,
                          handed over before it is asked how well it did.

The expected answer *is* supplied, because semantic equivalence cannot be judged
without it. The instruction compensates: the judge is asked whether the pipeline
answer is legally equivalent **on the cited evidence**, and is told in terms that
a differently-worded answer supported by the evidence is correct and a
string-matching one is not. That is a real residual risk and it is stated here
rather than papered over.

The scoring scheme, fixed before any judgement was read
-------------------------------------------------------

`SCORE_BY_CLASSIFICATION` maps the judge's categorical verdict to a number, and
the threshold is `PASS_THRESHOLD = 0.75`. Both are written down here, in advance,
for the reason the rest of this project writes its weights down in advance: a
threshold chosen after seeing the distribution is a threshold chosen to produce a
result. If a manual look at a sample says 0.75 is wrong, the right response is to
say so in the report and show both numbers — not to edit this constant until the
headline improves.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
from typing import Any, Mapping, Sequence

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_ANSWERS = os.path.join(PROJECT_ROOT, "data", "evaluation", "results",
                               "synthetic", "answers.jsonl")
DEFAULT_PACKETS = os.path.join(PROJECT_ROOT, "data", "evaluation", "results",
                               "synthetic", "judge_packets.jsonl")

PACKET_VERSION = "1.0.0"
JUDGE_PROVIDER = "claude_code_session"

#: Never allowed into a packet. Asserted, not trusted.
WITHHELD = ("rank_of_expected", "pool_rank_of_expected", "target_chunk_ids",
            "retrieval_correct", "retrieval_correct_at_1", "retrieved_chunk_ids",
            "rerank_score", "retrieval_score", "confidence", "difficulty",
            "generation_metadata", "metrics", "grounded")

CLASSIFICATIONS = ("correct", "partially_correct", "incorrect", "unsupported",
                   "appropriate_abstention", "inappropriate_abstention")

#: Declared in advance. See the module docstring.
SCORE_BY_CLASSIFICATION: dict[str, float] = {
    "correct": 1.00,                 # correct and fully supported by the evidence
    "partially_correct": 0.50,       # right as far as it goes, materially incomplete
    "incorrect": 0.00,               # wrong on the law as the evidence states it
    "unsupported": 0.00,             # may be right, but the cited evidence does not say it
    "appropriate_abstention": 1.00,  # declined, and declining was correct
    "inappropriate_abstention": 0.00,  # declined with sufficient evidence in hand
}

#: A minor omission the judge flags on an otherwise correct answer. Kept as a
#: modifier rather than a seventh category so the judge picks from a short list.
MINOR_OMISSION_SCORE = 0.75

PASS_THRESHOLD = 0.75

INSTRUCTIONS = """\
You are judging whether a question-answering system's answer is legally
equivalent to a reference answer, ON THE EVIDENCE SUPPLIED WITH IT.

You are given four things and nothing else: the QUESTION, an EXPECTED ANSWER
quoted from a legal corpus, the PIPELINE ANSWER, and the CITED EVIDENCE the
pipeline relied on.

Judge on substance, not wording. An answer that says the same thing in different
words is CORRECT. An answer that echoes the expected answer's phrasing but
states something the cited evidence does not support is UNSUPPORTED, not
correct. Do not reward string similarity and do not penalise paraphrase.

Use exactly one classification:

  correct                 states what the evidence supports, and answers the
                          question asked
  partially_correct       accurate but materially incomplete - a required
                          condition, time limit or exception is missing
  incorrect               contradicts the cited evidence or the expected answer
  unsupported             asserts something the cited evidence does not contain,
                          whether or not it happens to be true
  appropriate_abstention  declined to answer, and the evidence genuinely did not
                          support one
  inappropriate_abstention declined although the cited evidence did support an
                          answer

Also return:
  minor_omission   true only for an otherwise `correct` answer that omits
                   something small; it lowers the score to 0.75
  evidence_ids     the markers of the cited passages that actually support the
                   answer, as a list of integers - empty if none do
  reason           one or two sentences. Name the specific thing that decided
                   it, not a restatement of the classification.

If the pipeline abstained, judge whether abstaining was right given the cited
evidence, and use one of the two abstention classifications.
"""


def canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False).encode("utf-8")).hexdigest()


def build_packet(row: Mapping[str, Any]) -> dict[str, Any]:
    """One judgeable packet. Raises if anything withheld survives into it."""
    packet = {
        "question_id": row["question_id"],
        "packet_version": PACKET_VERSION,
        "question": row["question"],
        "expected_answer": row.get("expected_answer"),
        "pipeline_status": row.get("status"),
        "pipeline_answer": row.get("answer_text"),
        "pipeline_abstention_reasons": row.get("insufficient_reasons") or [],
        "cited_evidence": [
            {"marker": item["marker"],
             "citation": item.get("citation"),
             "temporal_status": item.get("temporal_status"),
             "passage": item.get("excerpt")}
            for item in (row.get("cited_evidence") or [])],
        "label_kind": row.get("label_kind"),
    }
    leaked = [key for key in WITHHELD if key in packet]
    if leaked:
        raise AssertionError(f"packet for {row['question_id']} leaks {leaked}")
    blob = json.dumps(packet, sort_keys=True, ensure_ascii=False)
    for key in WITHHELD:
        if f'"{key}"' in blob:
            raise AssertionError(f"packet for {row['question_id']} contains {key}")
    packet["packet_sha256"] = canonical_sha256(packet)
    return packet


def validate_judgement(judgement: Mapping[str, Any]) -> list[str]:
    problems: list[str] = []
    if judgement.get("classification") not in CLASSIFICATIONS:
        problems.append(f"classification must be one of {CLASSIFICATIONS}")
    if not str(judgement.get("reason") or "").strip():
        problems.append("reason is required")
    if not isinstance(judgement.get("evidence_ids", []), list):
        problems.append("evidence_ids must be a list")
    if judgement.get("minor_omission") not in (True, False, None):
        problems.append("minor_omission must be true, false or null")
    if not str(judgement.get("question_id") or "").strip():
        problems.append("question_id is required")
    return problems


def judge_score(judgement: Mapping[str, Any]) -> float:
    """The declared mapping, applied. No judgement-time discretion."""
    base = SCORE_BY_CLASSIFICATION[judgement["classification"]]
    if judgement["classification"] == "correct" and judgement.get("minor_omission"):
        return MINOR_OMISSION_SCORE
    return base


def score_judgements(judgements: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    problems = [p for j in judgements for p in validate_judgement(j)]
    if problems:
        raise SystemExit("invalid judgements:\n  " + "\n  ".join(problems[:20]))
    scores = [judge_score(j) for j in judgements]
    classifications = collections.Counter(j["classification"] for j in judgements)
    abstention_kinds = {"appropriate_abstention", "inappropriate_abstention"}
    substantive = [j for j in judgements
                   if j["classification"] not in abstention_kinds]
    substantive_scores = [judge_score(j) for j in substantive]
    passed = [s for s in scores if s >= PASS_THRESHOLD]
    return {
        "metric": "B_answer_quality",
        "judge_provider": JUDGE_PROVIDER,
        "judge_is_an_api_call": False,
        "judged": len(judgements),
        "pass_threshold": PASS_THRESHOLD,
        "score_scheme": SCORE_BY_CLASSIFICATION,
        "minor_omission_score": MINOR_OMISSION_SCORE,
        "classifications": dict(classifications),
        "pass_rate": {"numerator": len(passed), "denominator": len(scores)},
        "mean_judge_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "answer_correctness_rate": {
            "numerator": classifications["correct"],
            "denominator": len(substantive),
            "note": "correct / substantive answers; abstentions are scored "
                    "separately and are not in this denominator"},
        "mean_score_substantive_only": (
            round(sum(substantive_scores) / len(substantive_scores), 4)
            if substantive_scores else 0.0),
        "confidence_distribution": dict(collections.Counter(
            f"{judge_score(j):.2f}" for j in judgements)),
        "claim_safety":
            "Synthetic, AI-generated and AI-judged. The judge is a Claude Code "
            "session, not a lawyer, and shares the corpus's blind spots. Not "
            "human-verified legal ground truth.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--answers", default=DEFAULT_ANSWERS)
    parser.add_argument("--packets", default=DEFAULT_PACKETS)
    parser.add_argument("--score", help="score a judgements JSONL instead")
    parser.add_argument("--show-instructions", action="store_true")
    args = parser.parse_args(argv)

    if args.show_instructions:
        print(INSTRUCTIONS)
        return 0

    if args.score:
        with open(args.score, encoding="utf-8") as handle:
            judgements = [json.loads(line) for line in handle if line.strip()]
        summary = score_judgements(judgements)
        out = args.score.replace(".jsonl", ".summary.json")
        with open(out, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(json.dumps(summary, indent=2))
        print(f"\nwrote {out}")
        return 0

    with open(args.answers, encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    packets = [build_packet(row) for row in rows if row.get("status") in
               ("ANSWERED", "ABSTAINED")]
    os.makedirs(os.path.dirname(args.packets), exist_ok=True)
    with open(args.packets, "w", encoding="utf-8") as handle:
        for packet in packets:
            handle.write(json.dumps(packet, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(packets)} packets -> {args.packets}")
    print(f"withheld from every packet: {', '.join(WITHHELD)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
