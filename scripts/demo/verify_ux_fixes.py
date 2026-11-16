"""Live verification of the four UX/quality fixes, over the real pipeline.

Not unit tests. This builds the pipeline the server builds, with the configured
provider, and puts the questions from the brief's required-cases list through it
one at a time — because the four fixes interact, and the only way to know that a
greeting is not spell-corrected into a legal query, or that a typo'd question
does not get intercepted by the clarifier, is to run them in the order a request
runs them.

Every case declares what must be true of the response. A case fails loudly; it
is never softened to make a total look better.

    .venv/bin/python scripts/demo/verify_ux_fixes.py
    .venv/bin/python scripts/demo/verify_ux_fixes.py --sleep 20   # free tier
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Callable

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ai.answering.excerpt import display_text  # noqa: E402
from ai.text_integrity import DAMAGED, assess  # noqa: E402

RESULTS_PATH = os.path.join(PROJECT_ROOT, "data", "evaluation", "results",
                            "synthetic", "ux_fixes.json")


def _social(result: dict[str, Any]) -> list[str]:
    problems = []
    if result["status"] != "SOCIAL":
        problems.append(f"status was {result['status']}, expected SOCIAL")
    if result.get("answer") is not None:
        problems.append("a greeting returned an answer object")
    if result.get("evidence"):
        problems.append("a greeting retrieved evidence")
    if not (result.get("rendered") or "").strip():
        problems.append("no reply text")
    return problems


def _answered(result: dict[str, Any]) -> list[str]:
    problems = []
    if result["status"] != "ANSWERED":
        problems.append(f"status was {result['status']}, expected ANSWERED")
        return problems
    answer = result["answer"]
    if not answer["grounded"]:
        problems.append("published without grounding")
    if not answer["citations"]:
        problems.append("published without citations")
    report = answer.get("confidence_report") or {}
    if not report.get("factors"):
        problems.append("confidence carries no stated reason")
    return problems


def _corrected(expected: str, domain: str) -> Callable[[dict[str, Any]], list[str]]:
    """What a spelling fix owes: the corrected query retrieves the right law.

    Deliberately not "the answer is published". Whether the generator can ground
    an answer to "what are the requirements for a patent?" is a property of that
    question's breadth, not of the typo, and asserting it here would make this
    case fail for a reason it is not testing — it did, on the first run. What the
    correction is responsible for is that the question reaches retrieval instead
    of the clarifier, and that retrieval lands in the right body of law.
    """
    def check(result: dict[str, Any]) -> list[str]:
        spelling = result.get("spelling") or {}
        problems = []
        if not spelling.get("applied"):
            problems.append("no correction was applied")
        elif expected not in spelling["search_query"]:
            problems.append(f"search query {spelling['search_query']!r} lacks "
                            f"{expected!r}")
        if result["question"] != spelling.get("original_query"):
            problems.append("the question shown is not the question asked")
        if result["status"] == "NEEDS_CLARIFICATION":
            problems.append("a misspelled question was sent to the clarifier")
        domains = {item.get("domain") for item in result.get("evidence", ())}
        if domain not in domains:
            problems.append(f"retrieved {sorted(d for d in domains if d)} — "
                            f"nothing from {domain}")
        return problems
    return check


def _unchanged_query(result: dict[str, Any]) -> list[str]:
    spelling = result.get("spelling") or {}
    if spelling.get("applied"):
        return [f"an off-topic question was rewritten to "
                f"{spelling['search_query']!r}"]
    return []


def _clarified(result: dict[str, Any]) -> list[str]:
    if result["status"] != "NEEDS_CLARIFICATION":
        return [f"status was {result['status']}, expected NEEDS_CLARIFICATION"]
    if not (result.get("clarification") or {}).get("options"):
        return ["a clarification with nothing to choose from"]
    return []


def _confidence_is(*allowed: str) -> Callable[[dict[str, Any]], list[str]]:
    def check(result: dict[str, Any]) -> list[str]:
        problems = _answered(result)
        if problems:
            return problems
        answer = result["answer"]
        if answer["confidence"] not in allowed:
            problems.append(f"confidence {answer['confidence']}, expected one "
                            f"of {allowed}")
        return problems
    return check


def _stale_is_capped(result: dict[str, Any]) -> list[str]:
    """A stale source may be MODERATE or LOW, never HIGH, and must say why."""
    problems = _answered(result)
    if problems:
        return problems
    answer = result["answer"]
    stale = [c for c in answer["citations"] if c["temporal_status"] == "STALE"]
    if not stale:
        return ["this case is meant to cite a stale instrument and cited none"]
    if answer["confidence"] == "HIGH":
        problems.append("a stale source reached HIGH")
    codes = {f["code"] for f in answer["confidence_report"]["factors"]}
    if not codes & {"all_sources_stale", "some_sources_stale"}:
        problems.append("staleness capped the level without being named")
    if not any("stale" in caution.lower() for caution in answer["cautions"]):
        problems.append("no caution mentions the stale source")
    return problems


def _integrity_is_reported(result: dict[str, Any]) -> list[str]:
    """Damaged evidence is flagged, shown verbatim, and never repaired."""
    problems = []
    if not result.get("evidence"):
        return ["no evidence retrieved"]
    for item in result["evidence"]:
        verdict = item.get("text_integrity") or {}
        if verdict.get("status") not in ("CLEAN", "EXTRACTION_DAMAGED"):
            problems.append(f"[{item['marker']}] carries no integrity verdict")
        if verdict.get("repairable") is not False:
            problems.append(f"[{item['marker']}] claims to be repairable")
        excerpt = item.get("excerpt")
        if not excerpt:
            continue
        if excerpt["text"] not in display_text(item["text"]):
            problems.append(f"[{item['marker']}] excerpt is not a substring of "
                            f"the passage beside it")
        if (excerpt.get("integrity") or {}).get("status") is None:
            problems.append(f"[{item['marker']}] excerpt carries no verdict")
    return problems


def _damaged_source_is_surfaced(result: dict[str, Any]) -> list[str]:
    """The garbled case: at least one retrieved passage must be flagged."""
    problems = _integrity_is_reported(result)
    flagged = [item for item in result.get("evidence", [])
               if (item.get("text_integrity") or {}).get("status") == DAMAGED]
    if not flagged:
        problems.append("expected at least one passage flagged as damaged")
    else:
        for item in flagged:
            # The flag must not have changed the text it describes.
            if assess(item["text"]).status != DAMAGED:
                problems.append(f"[{item['marker']}] flag disagrees with the text")
    return problems


#: (label, question, check). The list is the brief's required-cases section,
#: with the conflict and stale cases taken from questions measured to produce
#: those states rather than invented for the test.
CASES: list[tuple[str, str, Callable[[dict[str, Any]], list[str]]]] = [
    ("SOCIAL/greeting", "hi", _social),
    ("SOCIAL/greeting", "hello", _social),
    ("SOCIAL/small talk", "how are you?", _social),
    ("SOCIAL/thanks", "thanks", _social),
    ("LEGAL",
     "How must an application for registration of a trade mark be signed?",
     _answered),
    ("STRONG/identifier",
     "What does Rule 23 of the Trade Marks Rules, 2017 require?",
     _answered),
    ("TYPO", "How do I register a tradmark?",
     _corrected("trademark", "trademarks")),
    ("TYPO", "What are the requirements for a patant?",
     _corrected("patent", "patents")),
    ("AMBIGUOUS", "How can I protect my idea?", _clarified),
    ("CONFLICT",
     "Is it permissible to carry out fees and surcharge under the Trade Marks "
     "Act, 1999?",
     _confidence_is("LOW", "MODERATE")),
    ("STALE",
     "What does Rule 1 of the Geographical Indications of Goods (Registration "
     "and Protection) Rules, 2002 provide?",
     _stale_is_capped),
    ("GARBLED",
     "What must a licence to manufacture state under the Drugs and Cosmetics "
     "Rules, 1945?",
     _damaged_source_is_surfaced),
    ("OFF-TOPIC", "What is the best pizza topping in Naples?", _unchanged_query),
]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sleep", type=float, default=3.0,
                        help="seconds between provider calls")
    parser.add_argument("--out", default=RESULTS_PATH)
    args = parser.parse_args(argv)

    from app.pipeline import Pipeline

    pipeline = Pipeline(log=lambda message: print(message, flush=True))
    print(f"\n  {len(CASES)} cases, provider {pipeline.provider_name}\n")

    records: list[dict[str, Any]] = []
    for index, (label, question, check) in enumerate(CASES, start=1):
        started = time.perf_counter()
        try:
            result = pipeline.answer(question)
            problems = check(result)
        except Exception as exc:  # noqa: BLE001 - a crash is a failed case
            result, problems = {}, [f"{type(exc).__name__}: {exc}"]
        elapsed = round((time.perf_counter() - started) * 1000, 1)

        mark = "ok  " if not problems else "FAIL"
        print(f"  {index:2d}/{len(CASES)} {mark} {label:20s} "
              f"{result.get('status', 'ERROR'):20s} {elapsed:8.1f}ms  "
              f"{question[:46]}")
        for problem in problems:
            print(f"          - {problem}")

        records.append({
            "label": label,
            "question": question,
            "status": result.get("status"),
            "confidence": (result.get("answer") or {}).get("confidence"),
            "spelling": result.get("spelling"),
            "problems": problems,
            "passed": not problems,
            "wall_ms": elapsed,
        })
        if args.sleep and index < len(CASES):
            time.sleep(args.sleep)

    passed = sum(1 for record in records if record["passed"])
    summary = {
        "check": "ux_fixes",
        "provider": pipeline.provider_name,
        "passed": passed,
        "total": len(records),
        "denominator_note": f"{passed} of {len(records)} declared cases passed; "
                            f"each case asserts a property of the live response, "
                            f"not the wording of an answer",
        "cases": records,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"\n  {passed}/{len(records)} passed  ->  {args.out}\n")
    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
