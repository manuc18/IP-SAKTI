"""Fill the validated-answer cache from the real pipeline, before the demo.

    .venv/bin/python scripts/demo/warm_cache.py                  # dry run
    .venv/bin/python scripts/demo/warm_cache.py --warm           # send requests
    .venv/bin/python scripts/demo/warm_cache.py --warm --max-requests 2

Nothing here writes an answer. It asks the running server the questions
`scripts/demo/verify.py` already defines — the same list, imported rather than
copied — over the same HTTP endpoint the browser uses, and the server caches
whatever its own pipeline publishes. A question that abstains is not cached,
and a question the provider could not answer is not cached; both are reported
and neither is retried.

The default is a dry run because the point of this script is to spend a token
budget deliberately. `--warm` sends requests, `--max-requests` caps how many,
and `--delay` paces them, because firing six questions back-to-back at a
free-tier per-minute cap produces rate limits rather than cached answers.

A question that is already cached costs no provider tokens, so it neither
consumes the request budget nor triggers a delay. Re-running this after a
partial warm therefore picks up only what is still missing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, PROJECT_ROOT)

from verify import SCENARIOS, request  # noqa: E402

from app.cache import (  # noqa: E402
    CACHE_FORMAT,
    DEFAULT_SEED_PATH,
    ValidatedAnswerCache,
    is_publishable,
    normalize_question,
)

ENDPOINTS = {"demo": "/api/ask", "api": "/api/v1/query"}


def read_state(endpoint: str, body: dict) -> tuple[str, bool]:
    """`(status, cache_hit)` from either server's response shape.

    `/api/ask` returns the pipeline result as-is; `/api/v1/query` reshapes it
    and puts the same flag under `meta`. One function reads both so the script
    does not care which server is running.
    """
    if endpoint == "api":
        return body.get("status", "?"), bool((body.get("meta") or {})
                                             .get("cache", {}).get("hit"))
    return body.get("status", "?"), bool((body.get("cache") or {}).get("hit"))


def server_namespace(base: str) -> str | None:
    """The cache namespace the running server computed for itself.

    Read rather than recomputed. The namespace folds in the corpus fingerprint,
    the retrieval configuration and the answering versions, and only the process
    that built the pipeline knows all of them — so a seed keyed by a namespace
    this script guessed would load into silence.
    """
    try:
        _, body = request(base, "/api/health")
    except OSError:
        return None
    return ((body.get("cache") or body.get("engine", {}).get("cache") or {})
            .get("namespace"))


def write_seed(results: dict[str, dict], namespace: str, path: str) -> int:
    """Write the deployment seed from results the server actually published.

    Every entry is re-checked with the same gate the server uses. Nothing is
    authored here: the payload is the response body, verbatim, minus the
    per-request `cache` block that belongs to the request that fetched it and
    not to the answer.
    """
    keyer = ValidatedAnswerCache(namespace=namespace, path=None, seed_path=None)
    entries = {}
    for question, result in results.items():
        payload = {k: v for k, v in result.items() if k != "cache"}
        if not is_publishable(payload):
            print(f"   ! refused to seed (failed the cache gate): {question[:60]}")
            continue
        entries[keyer.key(question)] = {
            "namespace": namespace,
            "question": result.get("question") or question,
            "normalized": normalize_question(question),
            "cached_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "result": payload,
        }

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump({"format": CACHE_FORMAT, "namespace": namespace,
                   "entries": entries}, handle, ensure_ascii=False, indent=2)
    return len(entries)


def warm(base: str, endpoint: str, scenarios, *, send: bool, max_requests: int,
         delay: float) -> tuple[int, dict[str, dict]]:
    """Returns `(exit code, {question: published result})`."""
    path = ENDPOINTS[endpoint]
    spent = 0
    published: dict[str, dict] = {}
    failed: list[tuple[str, str]] = []
    print(f"{'warming' if send else 'DRY RUN — nothing will be sent'}  "
          f"{base}{path}  (budget: {max_requests} provider request(s))\n")

    for index, (label, question) in enumerate(scenarios, start=1):
        print(f"{index}. {label}")
        print(f"   Q: {question}")
        if not send:
            print("   would send one request; re-run with --warm\n")
            continue
        if spent >= max_requests:
            print("   skipped — request budget exhausted\n")
            failed.append((question, "request budget exhausted"))
            continue

        try:
            http_status, body = request(base, path, {"question": question})
        except OSError as exc:
            print(f"   UNREACHABLE  {exc}")
            print("   Start the server first:  .venv/bin/python -m app.server\n")
            return 2, published

        if http_status not in (200, 503):
            print(f"   ERROR  HTTP {http_status}\n")
            failed.append((question, f"HTTP {http_status}"))
            continue

        status, hit = read_state(endpoint, body)
        if status == "ANSWERED" and hit:
            published[question] = body
            print("   ALREADY CACHED — no provider request was made\n")
            continue

        spent += 1
        if status == "ANSWERED":
            published[question] = body
            print("   CACHED — answered, citations validated, grounding passed")
        elif status == "ABSTAINED":
            print("   NOT CACHED — safe abstention (this is correct; only "
                  "published answers are cached)")
            failed.append((question, "ABSTAINED"))
        elif status == "UNAVAILABLE":
            code = ((body.get("service_failure") or body.get("error", {})
                     .get("provider_error") or {}).get("code"))
            print(f"   NOT CACHED — provider unavailable ({code}); not retried")
            failed.append((question, f"UNAVAILABLE ({code})"))
        else:
            print(f"   NOT CACHED — status {status}")
            failed.append((question, f"status {status}"))

        remaining = len(scenarios) - index
        if remaining and spent < max_requests and delay > 0:
            print(f"   waiting {delay:.0f}s before the next request "
                  f"(free-tier per-minute cap)")
            time.sleep(delay)
        print()

    print("=" * 76)
    if not send:
        print(f"{len(scenarios)} question(s) would be warmed. Re-run with --warm.")
        return 0, published

    print(f"{len(published)}/{len(scenarios)} question(s) cached · "
          f"{spent} provider request(s) spent")
    if failed:
        print("\nnot cached:")
        for question, why in failed:
            print(f"  - {why:26s} {question}")
    return (1 if failed else 0), published


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", choices=tuple(ENDPOINTS), default="demo",
                        help="demo: POST /api/ask (what the web UI calls). "
                             "api: POST /api/v1/query")
    parser.add_argument("--warm", action="store_true",
                        help="actually send requests; without it nothing leaves "
                             "this machine")
    parser.add_argument("--seed-demo-cache", action="store_true",
                        help="warm, then write the deployment seed file that "
                             "ships in the Docker image. Implies --warm. Fails "
                             "unless every demo question was published, so a "
                             "half-empty seed cannot reach a deployment "
                             "unnoticed; --allow-partial overrides that.")
    parser.add_argument("--seed-path", default=DEFAULT_SEED_PATH)
    parser.add_argument("--allow-partial", action="store_true",
                        help="write the seed even if some questions failed")
    parser.add_argument("--max-requests", type=int, default=len(SCENARIOS),
                        help="hard cap on requests that reach the provider "
                             "(cache hits do not count)")
    parser.add_argument("--delay", type=float, default=25.0,
                        help="seconds between requests that reached the "
                             "provider (default 25)")
    parser.add_argument("--question", action="append",
                        help="warm this question instead of the demo scenarios; "
                             "repeatable")
    args = parser.parse_args(argv)

    if args.seed_demo_cache and args.endpoint != "demo":
        parser.error("--seed-demo-cache needs --endpoint demo: only /api/ask "
                     "returns the complete pipeline result the seed stores.")

    scenarios = ([(f"ad hoc {i}", q) for i, q in enumerate(args.question, start=1)]
                 if args.question else list(SCENARIOS))
    code, published = warm(args.base, args.endpoint, scenarios,
                           send=args.warm or args.seed_demo_cache,
                           max_requests=max(0, args.max_requests),
                           delay=args.delay)

    if not args.seed_demo_cache:
        if args.warm:
            print("Repeat clicks on cached questions now consume zero provider "
                  "tokens — until this server restarts. Use --seed-demo-cache "
                  "to write a seed that survives a redeploy.")
        return code
    if code == 2:
        return 2
    if code and not args.allow_partial:
        print("\nSeed NOT written: not every demo question was published. Fix "
              "the questions listed above, or pass --allow-partial to ship a "
              "seed that covers only what succeeded.")
        return 1

    namespace = server_namespace(args.base)
    if not namespace:
        print("\nSeed NOT written: could not read the server's cache namespace "
              "from /api/health.")
        return 1

    written = write_seed(published, namespace, args.seed_path)
    print(f"\n→ {args.seed_path}")
    print(f"  {written} validated answer(s), namespace {namespace}")
    print("  Commit this file and redeploy. Generate it in the configuration "
          "you deploy (--provider groq --no-dense); a seed built under a "
          "different one will load as zero entries and say so at startup.")
    return 0 if written else 1


if __name__ == "__main__":
    raise SystemExit(main())
