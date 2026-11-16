"""Phase 8.2: record a judgement made by the current Claude Code session.

    .venv/bin/python scripts/evaluation/record_claude_code_judgement.py \
        --question Q027 --judgement path/to/judgement.json

The second execution path for the AI judge, beside `run_ai_judge.py`. The API
runner is unchanged and remains available; this one exists because the judge is
the Claude Code session itself, and a session is not something a Python script
can invoke.

**So this script is not the judge.** The judge reads
`data/evaluation/ai_judge_packets.jsonl` and the frozen instructions in
`docs/AI_GROUND_TRUTH_JUDGE_PROMPT.md`, and writes a JSON object. This script
checks that object and stores it. Everything it can check, it checks before
anything is written:

* the JSON validates against `run_ai_judge.validate_judgement` — the same
  validator the API path uses, not a second copy;
* the packet's own `packet_sha256` still matches its content, so a judgement
  cannot be attached to evidence that has since moved;
* the packet's corpus fingerprint matches the corpus on disk now;
* the packet's golden hash matches `golden_questions.jsonl` on disk now;
* every field Part 2 requires is present on the assembled record;
* no terminal judgement already exists for that question.

**Nothing here opens a socket.** There is no HTTP client, no API key, no
endpoint, and no import that reaches one — `run_ai_judge` is imported for its
schema and record builder, and importing it does not call anything.

**No invented measurements.** `judge_temperature` is `null`, because a Claude
Code session is not invoked through a temperature parameter and writing 0 would
be a false claim about how the judgement was produced. There are no token counts
and no latency: those are properties of an API call that did not happen.
`claude_code_elapsed_time` exists in the diagnostics for an environment that can
supply one honestly, and is `null` otherwise. It is never called model latency.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai.retrieval.base import load_chunks  # noqa: E402
from scripts.evaluation.prepare_ai_judgement import (  # noqa: E402
    MANIFEST_PATH,
    PACKETS_PATH,
    canonical_sha256,
    golden_sha256,
)
from scripts.evaluation.prepare_ground_truth import (  # noqa: E402
    corpus_fingerprint,
    read_jsonl,
    write_jsonl,
)
from scripts.evaluation.run_ai_judge import (  # noqa: E402
    JUDGEMENTS_PATH,
    TERMINAL,
    build_record,
    load_instructions,
    missing_record_fields,
    validate_judgement,
)

#: Not "anthropic". The distinction is the point of this module: an API call and
#: a Claude Code session are different provenance, and a record that blurred
#: them would be unreproducible in a way nothing downstream could detect.
JUDGE_PROVIDER = "claude_code_session"

#: A Claude Code session is not invoked with a temperature. `null` is the honest
#: value; `0.0` would assert a determinism control that was never applied.
JUDGE_TEMPERATURE = None

#: The session's identity is self-reported: it comes from the session's own
#: context, not from a server response that could corroborate it. Recorded as
#: such rather than silently presented as an attested model id. `--judge-model`
#: overrides it; "unknown" is the correct value where nothing is exposed.
DEFAULT_JUDGE_MODEL = "unknown"
MODEL_SOURCE = "session_self_report"

#: The prompt every judgement in this run must have been made under. Pinned as a
#: literal for the same reason the manifest exists: a check that recomputes the
#: value it verifies proves only that the extractor is deterministic.
FROZEN_PROMPT_VERSION = "1.0.0"
FROZEN_PROMPT_SHA256 = ("aa15a9c6d9b484403387c3b89dd8648f"
                        "ab0f8ed3bd1ec4f136203cc315399301")


def preflight(packet: dict[str, Any], judgement: dict[str, Any],
              judged: list[dict[str, Any]], *, golden_hash: str,
              fingerprint: str, force: bool,
              manifest: dict[str, Any] | None = None,
              prompt_sha: str | None = None) -> list[str]:
    """Everything that must hold before a judgement is stored. Part 13.

    Two layers, and they check different things. The self-consistency layer asks
    whether the packet still hashes to its own recorded hash and whether the
    corpus and golden file on disk are the ones it was built from. The frozen
    layer asks whether all of that matches `ai_judge_manifest.json`, written
    once and never rewritten — which is the only one of the two that can catch a
    packet file and a corpus that moved *together*. Without it, "judged against
    exactly this evidence" is a claim the code makes about itself.
    """
    problems = list(validate_judgement(judgement))
    question_id = packet["question_id"]

    body = {k: v for k, v in packet.items() if k != "packet_sha256"}
    if canonical_sha256(body) != packet["packet_sha256"]:
        problems.append("packet hash does not match the packet contents")
    if packet["corpus_fingerprint"] != fingerprint:
        problems.append(f"packet was built against corpus "
                        f"{packet['corpus_fingerprint']}, the corpus on disk is "
                        f"{fingerprint}")
    if packet["golden_dataset_sha256"] != golden_hash:
        problems.append("packet was built against a different "
                        "golden_questions.jsonl than the one on disk")

    if manifest is not None:
        frozen = manifest.get("packet_sha256", {}).get(question_id)
        if frozen is None:
            problems.append(f"{question_id} is not in the frozen manifest")
        elif frozen != packet["packet_sha256"]:
            problems.append(f"packet hash {packet['packet_sha256'][:12]} does not "
                            f"match the frozen manifest ({frozen[:12]})")
        if manifest.get("corpus_fingerprint") != fingerprint:
            problems.append(f"corpus fingerprint {fingerprint} does not match the "
                            f"frozen manifest ({manifest.get('corpus_fingerprint')})")
        if manifest.get("golden_dataset_sha256") != golden_hash:
            problems.append("golden_questions.jsonl does not match the frozen "
                            "manifest")
        if manifest.get("packet_version") != packet.get("packet_version"):
            problems.append(f"packet version {packet.get('packet_version')} does "
                            f"not match the frozen manifest "
                            f"({manifest.get('packet_version')})")

    if prompt_sha is not None and prompt_sha != FROZEN_PROMPT_SHA256:
        problems.append(f"judge prompt {prompt_sha[:12]} is not the frozen "
                        f"v{FROZEN_PROMPT_VERSION} prompt "
                        f"({FROZEN_PROMPT_SHA256[:12]})")

    existing = next((r for r in judged
                     if r["question_id"] == packet["question_id"]), None)
    if existing and existing.get("judge_status") in TERMINAL and not force:
        problems.append(f"{packet['question_id']} already carries a terminal "
                        f"judgement ({existing['judge_status']}); pass --force "
                        f"to replace it deliberately")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", required=True)
    parser.add_argument("--judgement", required=True,
                        help="path to the JSON object the session produced")
    parser.add_argument("--packets", default=PACKETS_PATH)
    parser.add_argument("--manifest", default=MANIFEST_PATH)
    parser.add_argument("--judgements", default=JUDGEMENTS_PATH)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL,
                        help="the session's model identity, if it is exposed; "
                             "'unknown' otherwise. Never invented.")
    parser.add_argument("--elapsed-seconds", type=float, default=None,
                        help="wall time of the session's reasoning, if the "
                             "environment can supply it honestly")
    parser.add_argument("--force", action="store_true",
                        help="replace an existing terminal judgement")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    packets = {p["question_id"]: p for p in read_jsonl(args.packets)}
    packet = packets.get(args.question)
    if packet is None:
        print(f"no packet for {args.question} in {args.packets}")
        return 1

    with open(args.judgement, encoding="utf-8") as fh:
        judgement = json.load(fh)

    judged = read_jsonl(args.judgements)
    prompt_version, _, prompt_sha = load_instructions()
    golden_hash = golden_sha256()
    fingerprint = corpus_fingerprint(load_chunks())

    if not os.path.exists(args.manifest):
        print(f"REFUSED no frozen manifest at {args.manifest}; run "
              f"prepare_ai_judgement.py first")
        return 2
    with open(args.manifest, encoding="utf-8") as fh:
        manifest = json.load(fh)

    problems = preflight(packet, judgement, judged, golden_hash=golden_hash,
                         fingerprint=fingerprint, force=args.force,
                         manifest=manifest, prompt_sha=prompt_sha)
    if problems:
        for problem in problems:
            print(f"REFUSED {problem}")
        return 2

    record = build_record(
        packet, judgement,
        model=args.judge_model,
        prompt_version=prompt_version,
        prompt_sha=prompt_sha,
        provider=JUDGE_PROVIDER,
        temperature=JUDGE_TEMPERATURE,
        diagnostics={
            "execution_mode": "claude_code_session",
            "judge_model_source": MODEL_SOURCE,
            "claude_code_elapsed_time": args.elapsed_seconds,
            "api_request_made": False,
            "requests": 0,
            "retries": 0,
            "passes": 1,
        })
    absent = missing_record_fields(record)
    if absent:
        print(f"REFUSED record is missing {absent}")
        return 2

    if args.dry_run:
        print(json.dumps(record, indent=2, ensure_ascii=False))
        print("\n(dry run - nothing written)")
        return 0

    by_qid = {r["question_id"]: r for r in judged}
    by_qid[record["question_id"]] = record
    write_jsonl(args.judgements, [by_qid[k] for k in sorted(by_qid)])

    print(f"{record['question_id']}: {record['judge_status']} "
          f"({record['judge_confidence']})")
    print(f"  provider    {record['judge_provider']}")
    print(f"  model       {record['judge_model']} ({MODEL_SOURCE})")
    print(f"  temperature {record['judge_temperature']}")
    print(f"  prompt      v{record['judge_prompt_version']} "
          f"({record['judge_prompt_sha256'][:12]})")
    print(f"  packet      {record['packet_sha256'][:12]}")
    print(f"  corpus      {record['corpus_fingerprint']}")
    print(f"  golden      {record['golden_dataset_sha256'][:12]}")
    print(f"  manifest    frozen, matched")
    print(f"  network requests: 0")
    print(f"\n{len(by_qid)} of {len(packets)} questions judged.")
    print("golden_questions.jsonl was not written to.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
