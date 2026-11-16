"""Configuration for the API server, in one place.

Every value is read from the environment, and every default is the one the
prototype already ran with — the API layer introduces no new tuning surface.
Three of these names (`IPSAKTI_PROVIDER`, `IPSAKTI_RERANKER`, `IPSAKTI_DENSE`)
are the same variables `app.pipeline` already reads, so the demo UI server and
the API server are configured identically rather than by two rival mechanisms.

`GROQ_API_KEY` is deliberately absent from this file. Credentials stay where
`ai/answering/groq.py` already reads them, they are never copied into config
objects, and no endpoint returns them.

There is no `TIMEOUT`. The pipeline answers under a single lock and offers no
cancellation point, so a timeout here could only report a deadline it cannot
enforce while the work continued behind it. A request runs to completion; the
one live network call is `ai/answering/groq.py`'s, with its own timeout.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from app.pipeline import APP_VERSION, MAX_QUESTION_CHARS

API_VERSION = "v1"
SERVICE_NAME = "ip-sakti-api"

#: An accepted client-supplied request id. Bounded and boring on purpose: the
#: value is echoed into responses and logs, so it may not carry newlines or
#: control characters.
MAX_REQUEST_ID_CHARS = 64


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in ("0", "false", "no", "off")


def _env_int(default: int, *names: str) -> int:
    for name in names:
        raw = os.environ.get(name)
        if raw:
            try:
                return int(raw)
            except ValueError:
                continue
    return default


@dataclass(frozen=True)
class Config:
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    #: "auto" | "groq" | "extractive" — resolved once, at startup, by
    #: `app.pipeline`. Never per request: §21, no silent provider switching.
    provider: str = "auto"
    reranker: str = "features"
    use_dense: bool = True
    max_question_length: int = MAX_QUESTION_CHARS
    #: 64 KiB, matching the demo server. Larger bodies are refused before the
    #: question-length check ever sees them.
    max_content_length: int = 64 * 1024
    #: ("*",) is a prototype default, and is why the API binds to 127.0.0.1.
    #: Set IPSAKTI_CORS_ORIGINS to a comma-separated list to narrow it, or to
    #: an empty string to send no CORS headers at all.
    cors_origins: tuple[str, ...] = ("*",)
    app_version: str = APP_VERSION

    @classmethod
    def from_environment(cls) -> "Config":
        raw_origins = os.environ.get("IPSAKTI_CORS_ORIGINS")
        if raw_origins is None:
            origins: tuple[str, ...] = ("*",)
        else:
            origins = tuple(o.strip() for o in raw_origins.split(",") if o.strip())
        return cls(
            host=os.environ.get("IPSAKTI_API_HOST") or os.environ.get("HOST") or "127.0.0.1",
            port=_env_int(8000, "IPSAKTI_API_PORT", "PORT"),
            debug=_env_flag("IPSAKTI_DEBUG", False),
            provider=os.environ.get("IPSAKTI_PROVIDER", "auto"),
            reranker=os.environ.get("IPSAKTI_RERANKER", "features"),
            use_dense=_env_flag("IPSAKTI_DENSE", True),
            cors_origins=origins,
        )
