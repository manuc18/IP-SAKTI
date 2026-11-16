"""Phase I: the ten user-facing behaviours, checked against a running server.

    .venv/bin/python scripts/demo/verify_phase_i.py
    .venv/bin/python scripts/demo/verify_phase_i.py --base http://127.0.0.1:8000

Companion to `scripts/demo/verify.py`, which rehearses the demo. This one checks
the *properties* Phase I names, one scenario per property, over the same HTTP
path the browser uses. Each scenario declares what must hold, and a scenario
fails only on the property it is there to test — a question that abstains is not
a failure unless abstention is the thing being contradicted.

Two of the checks are about what must NOT appear, and they are the ones worth
reading the output for:

* **No unsupported legal claim is published.** An ANSWERED response must be
  grounded, must carry at least one citation, and must have no validation
  failure and no unsupported claim. This restates the gates rather than
  trusting them, over the wire, on the real path.
* **No endless clarification loop.** A deliberately vague question may be asked
  one clarifying question. The same question sent back with `clarified: true`
  must never produce a second one. That is asserted directly, because the bound
  is the whole feature.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import os
import urllib.error
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ai.answering.excerpt import display_text  # noqa: E402

DEFAULT_BASE = "http://127.0.0.1:8000"

#: (label, question, required-property). Properties are checked by `CHECKS`.
SCENARIOS = (
    ("1. exact Rule question",
     "What does Rule 23 of the Trade Marks Rules, 2017 require?",
     "answered_or_abstained_cleanly"),
    ("2. exact Section question",
     "What does section 3(p) of the Patents Act, 1970 exclude from patentability?",
     "answered_or_abstained_cleanly"),
    ("3. conceptual legal question",
     "What is the term of copyright in a literary work published in the "
     "lifetime of the author?",
     "answered_or_abstained_cleanly"),
    ("4. ambiguous beginner question",
     "How can I trademark my own firm?",
     "inferred_not_interrupted"),
    ("5. practitioner-style question",
     "Under which provision may a compulsory licence be granted over a "
     "registered plant variety?",
     "answered_or_abstained_cleanly"),
    ("6. cross-reference question",
     "How do the Patents Act and the Patents Rules divide the time limit for "
     "requesting examination?",
     "answered_or_abstained_cleanly"),
    ("7. multi-part question",
     "Who may apply for registration of a geographical indication, and what "
     "must the application contain?",
     "answered_or_abstained_cleanly"),
    ("8. stale-source question",
     "What fee is payable when filing documents at the patent office?",
     "stale_is_disclosed"),
    ("9. unsupported / off-topic question",
     "What is the best pizza topping in Naples?",
     "must_abstain"),
    ("10. best evidence not rank 1 before reranking",
     "Is the mere discovery of a new form of a known substance patentable in India?",
     "answered_or_abstained_cleanly"),
    ("11. vague question with no legal anchor",
     "How can I protect my idea?",
     "one_clarification_then_never_again"),
)


def request(base: str, path: str, payload: dict | None = None,
            timeout: float = 180.0) -> tuple[int, dict]:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    req = urllib.request.Request(base + path, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b"{}")


# ------------------------------------------------------------------ checks ---

def _published_safely(body: dict) -> list[str]:
    """The invariants an ANSWERED response must satisfy. Never relaxed."""
    problems: list[str] = []
    answer = body.get("answer") or {}
    if not answer.get("grounded"):
        problems.append("published but not grounded")
    if not (answer.get("citations") or []):
        problems.append("published with no citation")
    if answer.get("validation_failures"):
        problems.append(f"published with validation failures: "
                        f"{[f.get('code') for f in answer['validation_failures']]}")
    if answer.get("unsupported_claims"):
        problems.append(f"published with {len(answer['unsupported_claims'])} "
                        f"unsupported claim(s)")
    cited = [e for e in body.get("evidence", []) if e.get("cited")]
    if not cited:
        problems.append("citations resolve to no evidence item")
    for item in cited:
        excerpt = item.get("excerpt") or {}
        if not excerpt.get("text"):
            problems.append(f"cited source [{item.get('marker')}] has no excerpt")
        elif excerpt["text"] not in display_text(item.get("text") or ""):
            # Phase F's guarantee, re-checked over the wire rather than trusted.
            # Normalised with the excerpt module's own `display_text`, which is
            # the text the offsets index; comparing against the raw chunk would
            # fail on OCR spacing the selector already collapsed, and would be a
            # bug in the check rather than in the excerpt.
            problems.append(f"excerpt for [{item.get('marker')}] is not a "
                            f"substring of the passage it quotes")
    return problems


def check(name: str, base: str, question: str, prop: str) -> tuple[bool, str, dict]:
    status, body = request(base, "/api/ask", {"question": question})
    verdict = body.get("status")

    if prop == "must_abstain":
        if verdict == "UNAVAILABLE":
            return False, f"provider unavailable ({body.get('service_failure', {}).get('code')})", body
        ok = verdict == "ABSTAINED"
        return ok, f"status={verdict} (required ABSTAINED)", body

    if prop == "inferred_not_interrupted":
        ok = verdict != "NEEDS_CLARIFICATION"
        detail = f"status={verdict}"
        if ok and verdict == "ANSWERED":
            problems = _published_safely(body)
            return (not problems), detail + ("; " + "; ".join(problems) if problems else
                                             "; inferred and published safely"), body
        return ok, detail + ("; a regime-bearing question must not be interrupted"
                             if not ok else ""), body

    if prop == "one_clarification_then_never_again":
        if verdict != "NEEDS_CLARIFICATION":
            return True, f"status={verdict} (no clarification needed; bound holds)", body
        status2, body2 = request(base, "/api/ask",
                                 {"question": question, "clarified": True})
        second = body2.get("status")
        ok = second != "NEEDS_CLARIFICATION"
        return ok, (f"first={verdict}, second={second} "
                    f"({'bounded' if ok else 'LOOPED - asked twice'})"), body

    if prop == "stale_is_disclosed":
        if verdict == "UNAVAILABLE":
            return False, "provider unavailable", body
        stale = [e for e in body.get("evidence", []) if e.get("cited")
                 and e.get("temporal_status") != "CURRENT"]
        cautions = (body.get("answer") or {}).get("cautions") or []
        if not stale:
            return True, "no stale source was cited; nothing to disclose", body
        ok = any("stale" in c.lower() or "current" in c.lower() for c in cautions)
        return ok, (f"{len(stale)} stale source(s) cited, "
                    f"{len(cautions)} caution(s) shown"), body

    # answered_or_abstained_cleanly
    if verdict == "UNAVAILABLE":
        return False, f"provider unavailable ({(body.get('service_failure') or {}).get('code')})", body
    if verdict == "ABSTAINED":
        reasons = [f.get("code") for f in
                   (body.get("answer") or {}).get("insufficient_evidence_reasons") or []]
        return True, f"ABSTAINED (safe) reasons={reasons}", body
    problems = _published_safely(body)
    return (not problems), ("ANSWERED, grounded, cited, excerpt verbatim"
                            if not problems else "; ".join(problems)), body


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=DEFAULT_BASE)
    parser.add_argument("--sleep", type=float, default=2.0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    rows = []
    failures = 0
    for index, (name, question, prop) in enumerate(SCENARIOS):
        t0 = time.perf_counter()
        try:
            ok, detail, body = check(name, args.base, question, prop)
        except Exception as exc:  # noqa: BLE001
            ok, detail, body = False, f"request failed: {exc}", {}
        elapsed = (time.perf_counter() - t0) * 1000
        failures += 0 if ok else 1
        mark = "PASS" if ok else "FAIL"
        print(f"[{mark}] {name}")
        print(f"       {question}")
        print(f"       {detail}  ({elapsed:.0f} ms)")
        answer = (body.get("answer") or {}).get("answer")
        if answer:
            print(f"       > {answer[:220]}")
        print()
        rows.append({"scenario": name, "question": question, "property": prop,
                     "pass": ok, "detail": detail,
                     "status": body.get("status"),
                     "wall_ms": round(elapsed, 1),
                     "answer": answer,
                     "cited": [{"marker": e["marker"], "citation": e.get("citation"),
                                "excerpt": (e.get("excerpt") or {}).get("text"),
                                "strategy": (e.get("excerpt") or {}).get("strategy"),
                                "temporal_status": e.get("temporal_status")}
                               for e in body.get("evidence", []) if e.get("cited")],
                     "clarification": body.get("clarification")})
        if args.sleep and index < len(SCENARIOS) - 1:
            time.sleep(args.sleep)

    print(f"{len(SCENARIOS) - failures}/{len(SCENARIOS)} scenarios passed")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump({"passed": len(SCENARIOS) - failures,
                       "total": len(SCENARIOS), "scenarios": rows},
                      handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        print(f"wrote {args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
