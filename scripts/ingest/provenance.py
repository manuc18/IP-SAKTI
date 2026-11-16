"""Source provenance and citability.

Two separate questions are answered here, and the whole point of the module is
that they stay separate:

*Can this chunk be retrieved?*  Almost always yes. Text whose locator could not
be established is still evidence; suppressing it from search would hide real
statutory material from the reader.

*Can this chunk be cited as authority?*  Only if a locator was established.
A citation is a promise that the reader can turn to that provision in the
source and find these words. When the ingestion pipeline withdrew the locator
(`locator_unavailable`), no such promise can be made, so `citable` is False.
This is a hard invariant, enforced again in `validate.py`: a future answer
engine must never be able to cite a chunk whose locator is unknown.

Source evidence level (`A`..`D`) is a *third*, orthogonal axis: it says how well
we know where the PDF came from, not whether the text inside it can be located.
It deliberately does not gate `citable` — a correctly located Section 3 of the
Patents Act is citable even while the PDF's provenance is still being resolved —
but it travels with every chunk so that a caller can apply its own floor.
"""
from __future__ import annotations

import json
import os
from typing import Any

from ingest.common import METADATA_DIR

REGISTRY_PATH = os.path.join(METADATA_DIR, "provenance_registry.json")

#: Categorical evidence levels (Phase 3 §10). Not a probability, not a score:
#: each level is a statement about what was actually verified.
EVIDENCE_LEVELS = {
    "A": "authoritative publisher source directly verified",
    "B": "authoritative identity established but local copy not cryptographically matched",
    "C": "credible copy but publisher provenance unresolved",
    "D": "unverified / unknown",
}

#: Fields lifted from the registry onto every chunk of a document.
CARRIED_FIELDS = (
    "source_status",
    "source_evidence_level",
    "temporal_status",
    "official_source_url",
    "consolidated_as_of",
    "effective_from",
    "effective_to",
    "source_version_id",
    "retrieved_at",
    "replacement_required",
)

_DEFAULTS: dict[str, Any] = {
    "source_status": "PROVENANCE_UNVERIFIED",
    "source_evidence_level": "D",
    "temporal_status": "UNKNOWN",
    "official_source_url": None,
    "consolidated_as_of": None,
    "effective_from": None,
    "effective_to": None,
    "source_version_id": None,
    "retrieved_at": None,
    "replacement_required": False,
}

_cache: dict[str, dict[str, Any]] | None = None


def load_registry() -> dict[str, dict[str, Any]]:
    """Return the registry keyed by ``document_id``.

    A missing registry is not an error: the ingestion pipeline must remain
    runnable before Phase 3 has been executed, in which case every document
    falls back to evidence level D.
    """
    global _cache
    if _cache is not None:
        return _cache
    if not os.path.exists(REGISTRY_PATH):
        _cache = {}
        return _cache
    with open(REGISTRY_PATH, encoding="utf-8") as fh:
        payload = json.load(fh)
    _cache = {entry["document_id"]: entry for entry in payload.get("sources", [])}
    return _cache


def provenance_fields(document_id: str) -> dict[str, Any]:
    """Registry fields for one document, defaulted where no evidence exists."""
    entry = load_registry().get(document_id, {})
    return {key: entry.get(key, _DEFAULTS[key]) for key in CARRIED_FIELDS}


def citability(*, locator_unavailable: bool) -> tuple[bool, bool, str | None]:
    """Return ``(retrievable, citable, reason)`` for one chunk.

    The only thing that makes a chunk uncitable is an unestablished locator.
    Staleness and unresolved publisher provenance are reported through
    `source_status` and `source_evidence_level`; they are the caller's policy
    decision, not a fact about whether these words can be pointed at.
    """
    if locator_unavailable:
        return True, False, "locator_unavailable"
    return True, True, None
