"""Phase 3 — provenance registry and the citability invariant.

The invariant under test is the one that protects the reader: a chunk whose
locator the pipeline could not establish stays retrievable but must never be
citable. It is asserted three times — in the function that decides it, in the
produced corpus, and in the validator — because it is the single rule a future
answer engine could most easily violate by accident.
"""
import json
import os

import pytest

from ingest.common import CHUNKS_DIR, logical_documents, read_jsonl
from ingest.provenance import (
    EVIDENCE_LEVELS,
    REGISTRY_PATH,
    citability,
    load_registry,
    provenance_fields,
)


class TestCitabilityRule:
    def test_a_located_chunk_is_retrievable_and_citable(self):
        retrievable, citable, reason = citability(locator_unavailable=False)
        assert (retrievable, citable, reason) == (True, True, None)

    def test_an_unlocated_chunk_is_retrievable_but_not_citable(self):
        retrievable, citable, reason = citability(locator_unavailable=True)
        assert retrievable is True
        assert citable is False
        assert reason == "locator_unavailable"


class TestRegistry:
    def _registry(self):
        if not os.path.exists(REGISTRY_PATH):
            pytest.skip("provenance registry not built")
        with open(REGISTRY_PATH, encoding="utf-8") as fh:
            return json.load(fh)

    def test_every_logical_document_has_a_registry_entry(self):
        entries = load_registry()
        if not entries:
            pytest.skip("provenance registry not built")
        for document in logical_documents():
            assert document.document_id in entries, document.document_id

    def test_local_sha256_matches_the_document_registry(self):
        entries = load_registry()
        if not entries:
            pytest.skip("provenance registry not built")
        for document in logical_documents():
            assert entries[document.document_id]["local_sha256"] == document.source_sha256

    def test_evidence_levels_are_from_the_controlled_vocabulary(self):
        payload = self._registry()
        for entry in payload["sources"]:
            assert entry["source_evidence_level"] in EVIDENCE_LEVELS
            assert entry["source_evidence_level_reason"], entry["document_id"]

    def test_a_missing_document_is_never_marked_verified(self):
        payload = self._registry()
        for entry in payload["sources"]:
            if entry["source_status"] == "MISSING":
                assert entry["source_evidence_level"] == "D"
                assert entry["replacement_required"] is True

    def test_defaults_apply_when_a_document_is_unknown(self):
        fields = provenance_fields("NO-SUCH-DOCUMENT")
        assert fields["source_evidence_level"] == "D"
        assert fields["source_status"] == "PROVENANCE_UNVERIFIED"


class TestCitabilityInTheProducedCorpus:
    """Runs against the produced corpus; skipped before the pipeline has run."""

    def _all_chunks(self):
        rows = []
        for document in logical_documents():
            path = os.path.join(CHUNKS_DIR, f"{document.document_id}.jsonl")
            if not os.path.exists(path):
                pytest.skip("pipeline output not present")
            rows.extend(read_jsonl(path))
        return rows

    def test_no_chunk_without_a_locator_is_citable(self):
        offenders = [
            chunk["chunk_id"]
            for chunk in self._all_chunks()
            if chunk["locator_unavailable"] and chunk["citable"]
        ]
        assert offenders == []

    def test_every_located_chunk_is_citable(self):
        offenders = [
            chunk["chunk_id"]
            for chunk in self._all_chunks()
            if not chunk["locator_unavailable"] and not chunk["citable"]
        ]
        assert offenders == []

    def test_every_chunk_is_retrievable_and_carries_an_evidence_level(self):
        for chunk in self._all_chunks():
            assert chunk["retrievable"] is True
            assert chunk["source_evidence_level"] in EVIDENCE_LEVELS
