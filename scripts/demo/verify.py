"""Health check and demo-question rehearsal against a running IP-SAKTI server.

    .venv/bin/python scripts/demo/verify.py                 # health + all scenarios
    .venv/bin/python scripts/demo/verify.py --health        # health only
    .venv/bin/python scripts/demo/verify.py --question "..."

Run this before the demo. It exercises the same HTTP path the browser uses, so
a pass here is a pass for the presentation — not a separate code path that only
resembles one. It sends one request per question and never retries: a failure
is reported as a failure.

An `ABSTAINED` verdict is not an error. Scenario 5 exists to produce one, and a
run where every question answers has not tested the safety property that
distinguishes this system from a chatbot.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_BASE = "http://127.0.0.1:8000"

#: Chosen for what each demonstrates, not for what each scores.
SCENARIOS = (
    ("1. Direct provision — trade marks",
     "How must an application for registration of a trade mark be signed?"),
    ("2. Different domain, and evidence not ranked first — patents",
     "Is the mere discovery of a new form of a known substance patentable in India?"),
    ("3. Multi-sentence, multi-citation — trade marks",
     "What is the duration of the registration of a trade mark and when must it "
     "be renewed?"),
    ("4. Source transparency — stale text and weak provenance earn cautions",
     "What is the term of copyright in a literary work published in the lifetime "
     "of the author?"),
    ("5. Corpus breadth — geographical indications",
     "Who may apply for the registration of a geographical indication?"),
    ("6. Safe abstention — outside the corpus",
     "What is the best pizza topping in Naples?"),
)


def request(base: str, path: str, payload: dict | None = None,
            timeout: float = 120.0) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(base + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


def health(base: str) -> int:
    try:
        status, body = request(base, "/api/health")
    except OSError as exc:
        print(f"UNREACHABLE  {base}  ({exc})")
        print("Start the server first:  .venv/bin/python -m app.server")
        return 2
    print(f"HEALTH {status}")
    print(f"  corpus      {body['corpus']['chunks']} chunks, "
          f"{body['corpus']['documents']} documents")
    print(f"  retrieval   dense={body['retrieval']['dense']} "
          f"lexical={body['retrieval']['lexical']} reranker={body['retrieval']['reranker']}")
    print(f"  provider    {body['answering']['provider']}")
    print(f"  startup     {body['startup_seconds']}s")
    for warning in body.get("warnings", []):
        print(f"  ! {warning}")
    return 0 if status == 200 else 1


def ask(base: str, label: str, question: str) -> dict:
    print("=" * 76)
    print(f"{label}\n  Q: {question}")
    started = time.perf_counter()
    status, body = request(base, "/api/ask", {"question": question})
    elapsed = round((time.perf_counter() - started) * 1000)

    if status != 200:
        print(f"  HTTP {status}: {body.get('error')}")
        return {"label": label, "question": question, "http_status": status,
                "error": body.get("error")}

    answer = body["answer"]
    if body["status"] == "UNAVAILABLE":
        failure = body.get("service_failure") or {}
        print(f"  verdict     UNAVAILABLE  ({failure.get('code')}) — provider "
              f"problem, not an evidence judgement")
        print(f"  detail      {failure.get('detail', '')[:160]}")
        return {"label": label, "question": question, "http_status": status,
                "verdict": "UNAVAILABLE", "service_error": failure.get("code"),
                "timings_ms": body["timings_ms"]}
    print(f"  verdict     {body['status']}  "
          f"grounded={answer['grounded']}  confidence={answer['confidence']}")
    print(f"  pipeline    {body['pipeline']['candidates']} candidates → "
          f"{body['pipeline']['evidence_count']} evidence · "
          f"{body['timings_ms']['total']} ms server / {elapsed} ms wall")

    if body["status"] == "ANSWERED":
        print(f"  answer      {answer['answer']}")
        for citation in answer["citations"]:
            print(f"    [{citation['marker']}] {citation['citation_text']} "
                  f"[{citation['temporal_status']}, level "
                  f"{citation['source_evidence_level']}]")
        for caution in answer["cautions"]:
            print(f"  caution     {caution}")
    else:
        for failure in (answer["insufficient_evidence_reasons"]
                        + answer["validation_failures"]):
            print(f"  abstained   {failure['code']}: {failure['detail']}")

    return {
        "label": label, "question": question, "http_status": status,
        "verdict": body["status"], "grounded": answer["grounded"],
        "confidence": answer["confidence"],
        "citations": [c["citation_text"] for c in answer["citations"]],
        "citation_count": len(answer["citations"]),
        "cautions": list(answer["cautions"]),
        "abstention_codes": [f["code"] for f in
                             answer["insufficient_evidence_reasons"]
                             + answer["validation_failures"]],
        "evidence": [{"marker": e["marker"], "document": e["document_name"],
                      "locator": e["locator_text"], "cited": e["cited"],
                      "temporal_status": e["temporal_status"]}
                     for e in body["evidence"]],
        "timings_ms": body["timings_ms"],
        "answer": answer["answer"] if body["status"] == "ANSWERED" else "",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--health", action="store_true", help="health check only")
    parser.add_argument("--question", help="ask one question instead of the scenarios")
    parser.add_argument("--out", help="write the run to this JSON file")
    parser.add_argument("--delay", type=float, default=25.0,
                        help="seconds between scenarios (default 25). The Groq "
                             "free tier caps tokens per minute, and a rehearsal "
                             "fired back-to-back reports rate limits rather than "
                             "answers. This paces requests; it never retries one.")
    args = parser.parse_args(argv)

    code = health(args.base)
    if code or args.health:
        return code

    if args.question:
        records = [ask(args.base, "ad hoc", args.question)]
    else:
        records = []
        for index, (label, question) in enumerate(SCENARIOS):
            if index:
                time.sleep(args.delay)
            records.append(ask(args.base, label, question))

    print("=" * 76)
    answered = sum(1 for r in records if r.get("verdict") == "ANSWERED")
    abstained = sum(1 for r in records if r.get("verdict") == "ABSTAINED")
    unavailable = sum(1 for r in records if r.get("verdict") == "UNAVAILABLE")
    errors = sum(1 for r in records if r.get("http_status") != 200)
    print(f"{len(records)} question(s): {answered} answered, {abstained} abstained, "
          f"{unavailable} provider-unavailable, {errors} error(s)")
    if unavailable:
        print("  ! provider-unavailable is a rate limit or transport failure, not a "
              "safety abstention. Raise --delay and re-run.")
    if records and all("timings_ms" in r for r in records):
        print("median total "
              f"{sorted(r['timings_ms']['total'] for r in records)[len(records) // 2]} ms")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"base": args.base, "records": records}, fh,
                      indent=2, ensure_ascii=False)
        print(f"→ {args.out}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
