"""IP-SAKTI prototype server — standard library only.

    .venv/bin/python -m app.server
    .venv/bin/python -m app.server --port 8080 --provider extractive

One process, one port, no build step and no new dependency. The pipeline is
built before the socket opens, so a request never races a half-loaded model and
the startup log is the health check: if it printed `serving on`, the corpus,
the indexes and the provider are all in place.

Nothing here decides anything about an answer. It parses a question, hands it
to `app.pipeline`, and serialises what comes back. Citation validation,
grounding and abstention live in `ai/answering` and are not re-implemented,
re-interpreted or overridden at this layer.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app import pipeline as pipeline_module  # noqa: E402

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
MAX_BODY_BYTES = 64 * 1024

CONTENT_TYPES = {".html": "text/html; charset=utf-8",
                 ".css": "text/css; charset=utf-8",
                 ".js": "text/javascript; charset=utf-8",
                 ".svg": "image/svg+xml"}

PIPELINE: pipeline_module.Pipeline | None = None


class Handler(BaseHTTPRequestHandler):
    server_version = "IP-SAKTI/0.1"

    # -- plumbing -------------------------------------------------------------

    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _json(self, status: int, payload: dict[str, Any]) -> None:
        self._send(status, json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                   "application/json; charset=utf-8")

    def _error(self, status: int, message: str) -> None:
        """A message a person can act on. Tracebacks go to the log, not the UI."""
        self._json(status, {"error": message})

    def _static(self, name: str) -> None:
        path = os.path.normpath(os.path.join(STATIC_DIR, name))
        if not path.startswith(STATIC_DIR) or not os.path.isfile(path):
            return self._error(404, "Not found.")
        with open(path, "rb") as fh:
            body = fh.read()
        suffix = os.path.splitext(path)[1]
        self._send(200, body, CONTENT_TYPES.get(suffix, "application/octet-stream"))

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("  %s\n" % (fmt % args))

    # -- routes ---------------------------------------------------------------

    def do_GET(self) -> None:  # noqa: N802 - stdlib naming
        route = self.path.split("?", 1)[0]
        if route in ("/", "/index.html"):
            return self._static("index.html")
        if route == "/api/health":
            if PIPELINE is None:
                return self._json(503, {"status": "starting"})
            return self._json(200, PIPELINE.health())
        if route.startswith("/static/"):
            return self._static(route[len("/static/"):])
        return self._error(404, "Not found.")

    do_HEAD = do_GET

    def do_POST(self) -> None:  # noqa: N802 - stdlib naming
        if self.path.split("?", 1)[0] != "/api/ask":
            return self._error(404, "Not found.")
        if PIPELINE is None:
            return self._error(503, "The system is still starting up.")

        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            return self._error(400, "Malformed request.")
        if length > MAX_BODY_BYTES:
            return self._error(413, "Request too large.")

        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
            question = payload.get("question", "")
            # Phase E: the client sets this on the request it sends *after*
            # showing a clarifying question, and it is the only thing that stops
            # a second one being asked. Anything other than a literal true is
            # treated as false, so a malformed flag cannot silently disable the
            # feature.
            clarified = payload.get("clarified") is True
        except (json.JSONDecodeError, UnicodeDecodeError):
            return self._error(400, "Malformed request.")

        try:
            return self._json(200, PIPELINE.answer(
                question, allow_clarification=not clarified))
        except ValueError as exc:
            return self._error(400, str(exc))
        except Exception:  # noqa: BLE001 - the demo must not die on one question
            traceback.print_exc()
            return self._error(
                500,
                "The pipeline failed on that question. Nothing was published — "
                "no answer is shown rather than an unverified one.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    parser.add_argument("--provider", choices=("auto", "groq", "extractive"),
                        help="auto (default): Groq when GROQ_API_KEY is set, "
                             "otherwise the offline extractive provider")
    parser.add_argument("--no-dense", action="store_true",
                        help="skip BGE-M3; faster startup, weaker retrieval")
    parser.add_argument("--reranker", choices=("features", "cross_encoder"))
    args = parser.parse_args(argv)

    if args.provider:
        os.environ["IPSAKTI_PROVIDER"] = args.provider
    if args.reranker:
        os.environ["IPSAKTI_RERANKER"] = args.reranker
    if args.no_dense:
        os.environ["IPSAKTI_DENSE"] = "0"

    global PIPELINE
    def log(message: str) -> None:
        print(message, flush=True)

    log("IP-SAKTI — starting")
    PIPELINE = pipeline_module.build(log=log)
    for warning in PIPELINE.warnings:
        log(f"  ! {warning}")
    log(f"  ready in {PIPELINE.startup_seconds}s")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    log(f"\n  serving on http://{args.host}:{args.port}   (ctrl-c to stop)\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
