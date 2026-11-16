"""Pass B: the AI judge. Phase 8.1, Parts 3, 6, 9, 10 and 11.

    # offline, the default:
    .venv/bin/python scripts/evaluation/run_ai_judge.py --question Q027
    # one live request, only with the flag and only with a key:
    .venv/bin/python scripts/evaluation/run_ai_judge.py --question Q027 --live

**Dry run is the default and `--live` is the only way past it.** Without that
flag the script renders the prompt, validates it, exercises the parser against
a fixture and writes nothing to the judgement file; `_post` is the single
function in this module that opens a socket and it is unreachable on the offline
path. There is no retry, no fallback model and no batching: one question, one
request, sequentially, with `--max-requests` defaulting to 1 so that the way to
spend 111 requests is to ask for 111 requests.

**The judge is not the answer generator.** Different model, different provider,
different prompt, different question. It is never asked "is this the correct
answer?" — it is asked whether the supplied evidence justifies the benchmark's
designation of a document, locator and chunk as ground truth. Nothing in this
module imports `ai.answering`, including for conveniences like the dotenv
reader: the eight duplicated lines below are cheaper than an import edge from
the auditor to the thing it audits.

**The judge is another AI system.** Its verdicts are labelled `AI_VERIFIED` and
never `verified`. They are a deterministic second-pass audit of the benchmark's
labels, not a determination of legal accuracy, and no count of them adds up to
one.

**The key is never printed.** Read from the environment, placed in one header,
scrubbed from every diagnostic string this module can emit.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import Any, Callable

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from scripts.evaluation.prepare_ai_judgement import (  # noqa: E402
    PACKETS_PATH,
    canonical_sha256,
)
from scripts.evaluation.prepare_ground_truth import (  # noqa: E402
    EVALUATION_DIR,
    read_jsonl,
    write_jsonl,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JUDGEMENTS_PATH = os.path.join(EVALUATION_DIR, "ai_judged_ground_truth.jsonl")
PROMPT_PATH = os.path.join(PROJECT_ROOT, "docs", "AI_GROUND_TRUTH_JUDGE_PROMPT.md")

AI_GROUND_TRUTH_SCHEMA = "1.0.0"
JUDGE_PROMPT_VERSION = "1.0.0"

JUDGE_PROVIDER = "anthropic"
ENDPOINT = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2023-06-01"
API_KEY_ENV = "ANTHROPIC_API_KEY"

#: Configurable through ANTHROPIC_MODEL. Whatever is requested, the identifier
#: recorded on the judgement is the one the API reports back, not the one we
#: asked for: a judgement attributed to a model that did not produce it is worse
#: than an unattributed one.
DEFAULT_MODEL = "claude-opus-5"

#: Temperature 0 and an assistant prefill of "{". Those are the only
#: determinism controls used, and both are documented Messages API behaviour —
#: no invented parameter, per Part 6.
TEMPERATURE = 0.0
PREFILL = "{"

#: A judgement is a paragraph of reasoning and a fixed set of fields.
MAX_TOKENS = 1500

#: A packet can reach 20 KB and the reasoning is not trivial, so the answer
#: layer's 15-second ceiling is too tight here. Still bounded, still clamped:
#: AI_JUDGE_TIMEOUT_SECONDS may lower it and may not raise it.
MAX_TIMEOUT_SECONDS = 120.0
DEFAULT_TIMEOUT_SECONDS = 90.0

USER_AGENT = "ip-sakti/0.1.0 (ground-truth-judge; python-urllib)"

JUDGE_STATUSES = ("AI_VERIFIED", "AI_CORRECTED", "AI_REJECTED", "AI_NEEDS_REVIEW")
CONFIDENCES = ("HIGH", "MODERATE", "LOW")
TEMPORAL_STATUSES = ("CURRENT", "STALE", "UNKNOWN")

TRISTATE_FIELDS = ("question_valid", "document_correct", "locator_correct",
                   "evidence_sufficient", "anchors_supported",
                   "scope_within_corpus", "current_source_available")
LIST_FIELDS = ("retrieval_target_chunk_ids", "answer_source_chunk_ids")
CORRECTION_FIELDS = ("corrected_document", "corrected_locator",
                     "corrected_chunk_id", "reason")

#: Part 2's record, in order. Every one must be present on a stored judgement.
RECORD_FIELDS = (
    "question_id", "judge_status",
    "question_valid", "document_correct", "locator_correct",
    "evidence_sufficient", "anchors_supported", "scope_within_corpus",
    "retrieval_target_chunk_ids", "answer_source_chunk_ids",
    "authoritative_document_id", "current_source_available", "temporal_status",
    "original_document_id", "original_locator", "original_chunk_id",
    "judge_reasoning", "judge_confidence",
    "corpus_fingerprint", "golden_dataset_sha256",
    "judge_provider", "judge_model", "judge_temperature", "judge_prompt_version",
    "judged_at",
)

_MARKER = re.compile(
    r"<!--\s*BEGIN JUDGE INSTRUCTIONS v([0-9.]+)\s*-->\n(.*?)\n<!--\s*END JUDGE INSTRUCTIONS",
    re.DOTALL)


# -- the prompt ---------------------------------------------------------------

def load_instructions(path: str = PROMPT_PATH) -> tuple[str, str, str]:
    """`(version, text, sha256)` read out of the prompt document itself.

    The document is the prompt. Keeping a second copy in this module is how the
    two would come to disagree, with no way to tell which one produced a given
    judgement — so the hash goes on every record alongside the version, because
    a version number is a promise and a hash is a fact.
    """
    with open(path, encoding="utf-8") as fh:
        match = _MARKER.search(fh.read())
    if match is None:
        raise ValueError(f"no versioned judge instructions found in {path}")
    version, text = match.group(1), match.group(2).strip()
    if version != JUDGE_PROMPT_VERSION:
        raise ValueError(f"prompt document is v{version}, this script expects "
                         f"v{JUDGE_PROMPT_VERSION}")
    return version, text, canonical_sha256(text)


def render_user_message(packet: dict[str, Any]) -> str:
    """The packet as prose the judge reads. Deterministic in the packet alone."""
    claim = packet["benchmark_claim"]
    question = packet["question"]
    lines = [
        "AUDIT THIS BENCHMARK LABEL.",
        "",
        f"QUESTION: {question['text']}",
        f"query family: {question['query_family']} | domain: {question['domain']} "
        f"| jurisdiction: {question['jurisdiction']}",
        "",
        "THE BENCHMARK CLAIMS (this is the claim under audit, not a fact):",
        f"  document : {claim['document_id']}",
        f"  locator  : {claim['locator']}",
        f"  chunk(s) : {', '.join(claim['chunk_ids'])}",
        f"  anchors  : {', '.join(repr(a) for a in claim['anchor_phrases'])}",
        f"  requires an authoritative citation: {claim['requires_citation']}",
        "",
        f"EVIDENCE ({packet['evidence_source']}):",
    ]
    for item in packet["evidence"]:
        lines.append("")
        lines.append("-" * 70)
        if item.get("missing_from_corpus"):
            lines.append(f"[{item['role'].upper()}] {item['chunk_id']}")
            lines.append("THIS CHUNK IS NOT PRESENT IN THE CORPUS.")
            continue
        lines.append(f"[{item['role'].upper()}] {item['chunk_id']}")
        lines.append(f"citation: {item['citation']}")
        lines.append(f"locator: {json.dumps(item['locator_fields'], sort_keys=True)}")
        lines.append(f"unit: {item['unit_type']} | heading: {item['heading']}")
        lines.append(f"temporal: {item['temporal_status']} | provenance level: "
                     f"{item['source_evidence_level']} ({item['source_status']}) | "
                     f"citable: {item['citable']}")
        lines.append("")
        lines.append(item["text"].strip())
    lines += ["", "-" * 70, "",
              "Return the JSON object described in your instructions and nothing else."]
    return "\n".join(lines)


# -- validating what comes back -----------------------------------------------

def validate_judgement(payload: Any) -> list[str]:
    """Every way a judge response can be unusable, named. Empty list = usable."""
    problems: list[str] = []
    if not isinstance(payload, dict):
        return [f"response is {type(payload).__name__}, not an object"]

    status = payload.get("judge_status")
    if status not in JUDGE_STATUSES:
        problems.append(f"judge_status {status!r} is not one of {JUDGE_STATUSES}")

    for field in TRISTATE_FIELDS:
        if field not in payload:
            problems.append(f"missing field: {field}")
        elif payload[field] not in (True, False, None):
            problems.append(f"{field} is {payload[field]!r}, not true/false/null")

    for field in LIST_FIELDS:
        if field not in payload:
            problems.append(f"missing field: {field}")
        elif payload[field] is not None and not (
                isinstance(payload[field], list)
                and all(isinstance(x, str) for x in payload[field])):
            problems.append(f"{field} is not a list of strings or null")

    if "authoritative_document_id" not in payload:
        problems.append("missing field: authoritative_document_id")
    if payload.get("temporal_status") not in TEMPORAL_STATUSES:
        problems.append(f"temporal_status {payload.get('temporal_status')!r} is not "
                        f"one of {TEMPORAL_STATUSES}")
    if payload.get("judge_confidence") not in CONFIDENCES:
        problems.append(f"judge_confidence {payload.get('judge_confidence')!r} is not "
                        f"one of {CONFIDENCES}")
    if not str(payload.get("judge_reasoning") or "").strip():
        problems.append("judge_reasoning is empty")

    # Part 3: a correction without its replacement fields is not a correction.
    correction = payload.get("correction")
    if status == "AI_CORRECTED":
        if not isinstance(correction, dict):
            problems.append("AI_CORRECTED without a correction object")
        else:
            for field in CORRECTION_FIELDS:
                if not str(correction.get(field) or "").strip():
                    problems.append(f"AI_CORRECTED correction is missing {field}")
    elif correction not in (None, {}):
        problems.append(f"{status} must not carry a correction")
    return problems


def parse_response(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """The prefilled `{` back onto the front, then strict JSON. No repair pass.

    A response that needs coaxing into shape is a response we cannot claim was
    deterministic, so a malformed one is recorded as malformed.
    """
    candidate = text if text.lstrip().startswith("{") else PREFILL + text
    try:
        return json.loads(candidate), None
    except json.JSONDecodeError as error:
        return None, f"response was not valid JSON: {error}"


def build_record(packet: dict[str, Any], judgement: dict[str, Any], *,
                 model: str, prompt_version: str, prompt_sha: str,
                 diagnostics: dict[str, Any],
                 provider: str = JUDGE_PROVIDER,
                 temperature: float | None = TEMPERATURE) -> dict[str, Any]:
    """Part 2's record: the judge's verdict plus the provenance of the verdict.

    `provider` and `temperature` are parameters rather than constants because
    Phase 8.2 added a second execution path: the judge may be this Anthropic API
    client at temperature 0, or it may be a Claude Code session, which is not
    invoked through a temperature parameter at all and therefore records `null`.
    The defaults keep the API path byte-for-byte what it was.
    """
    claim = packet["benchmark_claim"]
    record = {
        "question_id": packet["question_id"],
        "ai_ground_truth_schema": AI_GROUND_TRUTH_SCHEMA,
        "judge_status": judgement["judge_status"],
    }
    for field in TRISTATE_FIELDS:
        record[field] = judgement.get(field)
    for field in LIST_FIELDS:
        record[field] = judgement.get(field)
    record.update({
        "authoritative_document_id": judgement.get("authoritative_document_id"),
        "temporal_status": judgement.get("temporal_status"),
        "correction": judgement.get("correction"),

        # The label as it stood, so a correction never has to be read against a
        # file that may since have moved on.
        "original_document_id": claim["document_id"],
        "original_locator": claim["locator"],
        "original_chunk_id": claim["chunk_ids"][0] if claim["chunk_ids"] else None,
        "original_chunk_ids": list(claim["chunk_ids"]),

        "judge_reasoning": judgement.get("judge_reasoning"),
        "judge_confidence": judgement.get("judge_confidence"),

        "corpus_fingerprint": packet["corpus_fingerprint"],
        "golden_dataset_sha256": packet["golden_dataset_sha256"],
        "packet_sha256": packet["packet_sha256"],

        "judge_provider": provider,
        "judge_model": model,
        "judge_temperature": temperature,
        "judge_prompt_version": prompt_version,
        "judge_prompt_sha256": prompt_sha,
        "judged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "diagnostics": diagnostics,
    })
    return record


def missing_record_fields(record: dict[str, Any]) -> list[str]:
    return [field for field in RECORD_FIELDS if field not in record]


# -- the one network call -----------------------------------------------------

def load_env_file(path: str | None = None) -> None:
    """Read `.env` without overriding anything already exported.

    Eight lines rather than an import from `ai.answering.groq`, which would give
    the auditor a dependency on the layer it audits for the sake of a dotenv
    reader.
    """
    if path is None:
        path = os.path.join(PROJECT_ROOT, ".env")
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def scrub(text: str, secret: str | None) -> str:
    """No diagnostic string may carry the key, whatever the API echoed back."""
    if secret and secret in text:
        text = text.replace(secret, "***")
    return text


def _post(payload: dict[str, Any], api_key: str, timeout: float) -> tuple[dict[str, Any], float]:
    """The only socket in this module. Unreachable without `--live`."""
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(ENDPOINT, data=body, method="POST", headers={
        "content-type": "application/json",
        "anthropic-version": ANTHROPIC_VERSION,
        "x-api-key": api_key,
        "user-agent": USER_AGENT,
    })
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    elapsed = (time.perf_counter() - started) * 1000.0
    return json.loads(raw.decode("utf-8")), elapsed


def judge_once(packet: dict[str, Any], instructions: str, *, model: str,
               api_key: str, timeout: float) -> tuple[dict[str, Any], dict[str, Any]]:
    """One question, one request, no retry. Returns `(raw_response, diagnostics)`."""
    payload = {
        "model": model,
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "system": instructions,
        "messages": [
            {"role": "user", "content": render_user_message(packet)},
            {"role": "assistant", "content": PREFILL},
        ],
    }
    try:
        response, elapsed_ms = _post(payload, api_key, timeout)
    except urllib.error.HTTPError as error:
        detail = scrub(error.read().decode("utf-8", "replace")[:600], api_key)
        return {}, {"error": f"http_{error.code}", "detail": detail,
                    "requests": 1, "retries": 0}
    except Exception as error:                       # noqa: BLE001 - recorded, not raised
        return {}, {"error": "request_failed",
                    "detail": scrub(str(error), api_key), "requests": 1, "retries": 0}

    usage = response.get("usage") or {}
    diagnostics = {
        "requests": 1,
        "retries": 0,
        "latency_ms": round(elapsed_ms, 1),
        "stop_reason": response.get("stop_reason"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "reported_model": response.get("model"),
    }
    return response, diagnostics


def response_text(response: dict[str, Any]) -> str:
    return "".join(block.get("text", "") for block in response.get("content", [])
                   if block.get("type") == "text")


# -- the run ------------------------------------------------------------------

TERMINAL = set(JUDGE_STATUSES)


def pending(packets: list[dict[str, Any]], judged: list[dict[str, Any]], *,
            question_id: str | None, start_from: str | None, limit: int | None,
            only_unjudged: bool) -> list[dict[str, Any]]:
    """Which packets this invocation would judge, in question order."""
    done = {record["question_id"] for record in judged
            if record.get("judge_status") in TERMINAL}
    chosen = []
    for packet in sorted(packets, key=lambda p: p["question_id"]):
        qid = packet["question_id"]
        if question_id and qid != question_id:
            continue
        if start_from and qid < start_from:
            continue
        if only_unjudged and qid in done:
            continue
        chosen.append(packet)
    return chosen[:limit] if limit is not None else chosen


FIXTURE_RESPONSE = {
    "judge_status": "AI_NEEDS_REVIEW",
    "question_valid": True,
    "document_correct": True,
    "locator_correct": None,
    "evidence_sufficient": False,
    "anchors_supported": True,
    "scope_within_corpus": None,
    "retrieval_target_chunk_ids": None,
    "answer_source_chunk_ids": None,
    "authoritative_document_id": None,
    "current_source_available": None,
    "temporal_status": "CURRENT",
    "correction": None,
    "judge_reasoning": "Offline fixture: exercises the parser and the record "
                       "builder without a request.",
    "judge_confidence": "LOW",
}


def dry_run(packets: list[dict[str, Any]], instructions: str, prompt_version: str,
            prompt_sha: str, model: str, verbose: bool) -> int:
    """Everything the live path does except the request. Part 11's checklist."""
    failures = 0
    for packet in packets:
        message = render_user_message(packet)
        parsed, error = parse_response(json.dumps(FIXTURE_RESPONSE)[1:])
        problems = [error] if error else validate_judgement(parsed)
        record = None
        if not problems:
            record = build_record(packet, parsed, model=model,
                                  prompt_version=prompt_version,
                                  prompt_sha=prompt_sha,
                                  diagnostics={"dry_run": True})
            problems = [f"record missing {f}" for f in missing_record_fields(record)]
        state = "ok" if not problems else "FAILED"
        print(f"  {packet['question_id']}  prompt {len(message):6d} chars  "
              f"{len(packet['evidence'])} passages  {state}")
        for problem in problems:
            print(f"      {problem}")
            failures += 1
        if verbose and record is not None:
            print(message)
    return failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--packets", default=PACKETS_PATH)
    parser.add_argument("--judgements", default=JUDGEMENTS_PATH)
    parser.add_argument("--question", help="judge exactly this question id")
    parser.add_argument("--start-from", metavar="QUESTION_ID")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only-status", choices=("UNJUDGED", "ALL"),
                        default="UNJUDGED")
    parser.add_argument("--max-requests", type=int, default=1,
                        help="hard ceiling on live requests this invocation "
                             "(default 1)")
    parser.add_argument("--model", default=os.environ.get("ANTHROPIC_MODEL",
                                                          DEFAULT_MODEL))
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--live", action="store_true",
                        help="actually make requests; without it nothing leaves "
                             "the machine")
    parser.add_argument("--show-prompt", action="store_true")
    args = parser.parse_args(argv)

    packets = read_jsonl(args.packets)
    if not packets:
        print(f"no packets at {args.packets}; run prepare_ai_judgement.py first")
        return 1
    judged = read_jsonl(args.judgements)
    prompt_version, instructions, prompt_sha = load_instructions()

    todo = pending(packets, judged, question_id=args.question,
                   start_from=args.start_from, limit=args.limit,
                   only_unjudged=args.only_status == "UNJUDGED")
    timeout = min(max(args.timeout, 1.0), MAX_TIMEOUT_SECONDS)

    print(f"packets {len(packets)} | already judged "
          f"{sum(1 for r in judged if r.get('judge_status') in TERMINAL)} | "
          f"selected {len(todo)}")
    print(f"model {args.model} | temperature {TEMPERATURE} | prompt "
          f"v{prompt_version} ({prompt_sha[:12]}) | timeout {timeout}s")

    if not args.live:
        print("\nDRY RUN - no network. Use --live to make requests.\n")
        failures = dry_run(todo, instructions, prompt_version, prompt_sha,
                           args.model, args.show_prompt)
        print(f"\n{len(todo)} packet(s) checked, {failures} failure(s)")
        return 1 if failures else 0

    load_env_file()
    api_key = os.environ.get(API_KEY_ENV)
    if not api_key:
        print(f"\n{API_KEY_ENV} is not set. No request was made.")
        return 2

    budget = max(0, args.max_requests)
    if len(todo) > budget:
        print(f"\n{len(todo)} selected but --max-requests is {budget}; judging the "
              f"first {budget}.")
        todo = todo[:budget]

    by_qid = {record["question_id"]: record for record in judged}
    made = 0
    for packet in todo:
        qid = packet["question_id"]
        if by_qid.get(qid, {}).get("judge_status") in TERMINAL:
            # Belt and braces: `pending` already filtered these, and a completed
            # record must not be overwritten even if a flag combination gets it
            # back into the list.
            print(f"{qid}: already judged, skipped")
            continue
        print(f"\n{qid}: requesting...")
        response, diagnostics = judge_once(packet, instructions, model=args.model,
                                           api_key=api_key, timeout=timeout)
        made += 1
        if diagnostics.get("error"):
            print(f"  FAILED {diagnostics['error']}: {diagnostics.get('detail','')}")
            print("  no retry, by design. Stopping.")
            return 3

        parsed, error = parse_response(response_text(response))
        problems = [error] if error else validate_judgement(parsed)
        if problems:
            print(f"  response was not usable: {problems}")
            print(f"  diagnostics: {diagnostics}")
            return 4

        record = build_record(packet, parsed,
                              model=diagnostics.get("reported_model") or args.model,
                              prompt_version=prompt_version, prompt_sha=prompt_sha,
                              diagnostics=diagnostics)
        absent = missing_record_fields(record)
        if absent:
            print(f"  record is missing {absent}")
            return 4

        by_qid[qid] = record
        # Checkpoint immediately: an interrupted run keeps every judgement it
        # already paid for, and resuming re-judges none of them.
        write_jsonl(args.judgements,
                    [by_qid[k] for k in sorted(by_qid)])
        print(f"  -> {record['judge_status']} ({record['judge_confidence']}) "
              f"in {diagnostics['latency_ms']} ms, "
              f"{diagnostics['input_tokens']} in / "
              f"{diagnostics['output_tokens']} out")

    print(f"\n{made} live request(s) made. "
          f"{len(by_qid)} of {len(packets)} questions judged.")
    print("golden_questions.jsonl was not written to.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
