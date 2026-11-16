"""IP-SAKTI HTTP API — the integration surface for the frontend team.

    .venv/bin/python -m app.api_server
    .venv/bin/python -m app.api_server --port 8001 --provider extractive

A thin Flask application over the pipeline that `app/pipeline.py` already
builds and `app/server.py` already serves to the demo UI. Both servers call the
same `Pipeline.answer()`; this one returns JSON to a client that will render it
elsewhere, the other returns JSON to a page it ships itself. Neither contains a
second copy of the engine.

What this layer does, in order, for every request:

    validate → call the pipeline → classify the result → serialise → respond

What it does not do: retrieval, reranking, citation resolution, grounding,
confidence, or any decision about whether an answer may be published. Those
live in `ai/` and are not re-implemented, re-interpreted or overridden here. In
particular, a provider failure is reported as UNAVAILABLE and never as a safe
abstention — the pipeline draws that distinction and this layer preserves it.

The pipeline is built once, before the socket opens. Nothing on the request
path loads a model, reads the corpus or rebuilds an index.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from flask import Flask, jsonify, request  # noqa: E402
from werkzeug.exceptions import HTTPException  # noqa: E402

from app.config import API_VERSION, Config, SERVICE_NAME  # noqa: E402
from app.schemas import (  # noqa: E402
    RequestError,
    error_response,
    new_request_id,
    serialize_result,
    validate_query_request,
)

QUERY_ROUTE = f"/api/{API_VERSION}/query"


def create_app(config: Config | None = None, pipeline=None) -> Flask:
    """Build the Flask app around an already-constructed pipeline.

    `pipeline` is injected rather than imported so the API can be tested
    against a controlled result without loading BGE-M3, reading the corpus or
    making a network request. Passing `None` builds the real one, which is what
    :func:`main` does — after logging, so startup is visible.
    """
    config = config or Config.from_environment()
    if pipeline is None:
        from app import pipeline as pipeline_module

        pipeline = pipeline_module.build()

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = config.max_content_length
    # Keys leave in the order they were built: success, status, answer,
    # citations, sources. Alphabetical order would put `error` before `answer`
    # and read like a failure in every response the dev team eyeballs.
    app.json.sort_keys = False
    app.config["IPSAKTI_CONFIG"] = config
    app.config["IPSAKTI_PIPELINE"] = pipeline

    _register_cors(app, config)
    _register_routes(app, config, pipeline)
    _register_error_handlers(app)
    return app


# -- CORS ---------------------------------------------------------------------


def _register_cors(app: Flask, config: Config) -> None:
    """Hand-rolled and about fifteen lines, rather than a new dependency.

    Development-friendly by default (`*`), which is also why the server binds
    to 127.0.0.1. Set IPSAKTI_CORS_ORIGINS to narrow it, or to an empty string
    to send no CORS headers at all.
    """
    if not config.cors_origins:
        return

    allowed = set(config.cors_origins)
    wildcard = "*" in allowed

    @app.after_request
    def add_cors_headers(response):  # noqa: ANN001, ANN202
        origin = request.headers.get("Origin")
        if wildcard:
            response.headers["Access-Control-Allow-Origin"] = "*"
        elif origin and origin in allowed:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
        else:
            return response
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        response.headers["Access-Control-Max-Age"] = "600"
        return response


# -- routes -------------------------------------------------------------------


def _register_routes(app: Flask, config: Config, pipeline) -> None:

    @app.get("/")
    def root():  # noqa: ANN202
        return jsonify({
            "service": "IP-SAKTI API",
            "version": "prototype",
            "api_version": API_VERSION,
            "app_version": config.app_version,
            "endpoints": {
                "health": "GET /health",
                "query": f"POST {QUERY_ROUTE}",
            },
        })

    @app.get("/health")
    def health():  # noqa: ANN202
        """Liveness only: no retrieval, no model call, no network request.

        `Pipeline.health()` reads counters and version strings that were fixed
        at startup. It is safe to poll.
        """
        return jsonify({"status": "ok", "service": SERVICE_NAME,
                        "api_version": API_VERSION, "engine": pipeline.health()})

    @app.post(QUERY_ROUTE)
    def query():  # noqa: ANN202
        request_id = new_request_id(None)
        try:
            if not request.is_json:
                raise RequestError(
                    "INVALID_CONTENT_TYPE",
                    "Request body must be JSON with header "
                    "'Content-Type: application/json'.")
            payload = request.get_json(silent=True)
            if payload is None:
                raise RequestError("INVALID_JSON",
                                   "Request body is not valid JSON.")

            question, request_id = validate_query_request(
                payload,
                max_question_length=config.max_question_length,
                active_provider=getattr(pipeline, "provider_name", "unknown"))
        except RequestError as exc:
            return jsonify(error_response(exc.code, exc.message,
                                          status="INVALID_REQUEST",
                                          request_id=request_id)), exc.http_status

        try:
            result = pipeline.answer(question)
        except ValueError as exc:
            # The pipeline's own input guard. Reachable only if its limits and
            # this layer's ever disagree; it stays a 400 either way.
            return jsonify(error_response("INVALID_QUESTION", str(exc),
                                          status="INVALID_REQUEST",
                                          request_id=request_id,
                                          question=question)), 400
        except Exception:  # noqa: BLE001 - one bad question must not end the demo
            app.logger.exception("pipeline failed [request_id=%s]", request_id)
            return jsonify(error_response(
                "INTERNAL_ERROR",
                "An unexpected error occurred. Nothing was published — no "
                "answer is returned rather than an unverified one.",
                request_id=request_id, question=question)), 500

        body, status = serialize_result(result, request_id)
        return jsonify(body), status


# -- errors -------------------------------------------------------------------


def _register_error_handlers(app: Flask) -> None:
    """Every failure leaves as the same JSON envelope. No HTML, no traceback."""

    @app.errorhandler(404)
    def not_found(_error):  # noqa: ANN001, ANN202
        return jsonify(error_response(
            "NOT_FOUND",
            f"Unknown endpoint. Try GET /health or POST {QUERY_ROUTE}.",
        )), 404

    @app.errorhandler(405)
    def method_not_allowed(_error):  # noqa: ANN001, ANN202
        return jsonify(error_response(
            "METHOD_NOT_ALLOWED",
            f"That method is not supported on this endpoint. "
            f"The query endpoint is POST {QUERY_ROUTE}.",
        )), 405

    @app.errorhandler(413)
    def payload_too_large(_error):  # noqa: ANN001, ANN202
        return jsonify(error_response(
            "REQUEST_TOO_LARGE",
            f"Request body exceeds {app.config['MAX_CONTENT_LENGTH']} bytes.",
            status="INVALID_REQUEST",
        )), 413

    @app.errorhandler(HTTPException)
    def http_exception(error):  # noqa: ANN001, ANN202
        return jsonify(error_response(
            "HTTP_ERROR", error.description or "Request could not be handled.",
        )), error.code or 500

    @app.errorhandler(Exception)
    def unexpected(error):  # noqa: ANN001, ANN202
        """The detail goes to the log. The client gets a code and a sentence."""
        app.logger.exception("unhandled error: %s", type(error).__name__)
        return jsonify(error_response(
            "INTERNAL_ERROR", "An unexpected error occurred.",
        )), 500


# -- entry point --------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--provider", choices=("auto", "groq", "extractive"),
                        help="auto (default): Groq when GROQ_API_KEY is set, "
                             "otherwise the offline extractive provider")
    parser.add_argument("--reranker", choices=("features", "cross_encoder"))
    parser.add_argument("--no-dense", action="store_true",
                        help="skip BGE-M3; faster startup, weaker retrieval")
    parser.add_argument("--debug", action="store_true",
                        help="verbose Flask errors; the reloader stays off, "
                             "because reloading rebuilds the whole engine")
    args = parser.parse_args(argv)

    # Set before Config.from_environment() reads them, so CLI flags and the
    # environment go through one code path instead of two.
    if args.provider:
        os.environ["IPSAKTI_PROVIDER"] = args.provider
    if args.reranker:
        os.environ["IPSAKTI_RERANKER"] = args.reranker
    if args.no_dense:
        os.environ["IPSAKTI_DENSE"] = "0"
    if args.debug:
        os.environ["IPSAKTI_DEBUG"] = "1"

    config = Config.from_environment()
    if args.host:
        config = replace(config, host=args.host)
    if args.port:
        config = replace(config, port=args.port)

    from app import pipeline as pipeline_module

    def log(message: str) -> None:
        print(message, flush=True)

    log("IP-SAKTI API — starting")
    pipeline = pipeline_module.build(log=log)
    for warning in pipeline.warnings:
        log(f"  ! {warning}")
    log(f"  ready in {pipeline.startup_seconds}s")

    app = create_app(config, pipeline)
    log(f"\nIP-SAKTI API running\nhttp://{config.host}:{config.port}\n")
    log(f"POST {QUERY_ROUTE}")
    log("GET  /health\n")

    # Flask's development server: fine for a prototype and for the demo, and
    # documented in docs/API_HANDOVER.md as the thing to replace with a WSGI
    # server (gunicorn, waitress) before this is deployed anywhere real.
    app.run(host=config.host, port=config.port, debug=config.debug,
            use_reloader=False, threaded=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
