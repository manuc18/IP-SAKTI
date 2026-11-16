#!/usr/bin/env python3
"""Standalone bridge script for IP-SAKTI Express backend.

Does NOT modify any file in ai/, scripts/, or tests/.
Imports existing answer engine and returns normalized JSON to stdout.
"""
import json
import os
import sys

# Add project root to sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No question provided"}))
        sys.exit(1)

    question = sys.argv[1].strip()
    if not question:
        print(json.dumps({"error": "Empty question"}))
        sys.exit(1)

    try:
        from ai.answering import AnswerPolicy, EvidenceSet, ExtractiveMockProvider, GroundedAnswerEngine
        from ai.retrieval.base import RetrievalResult

        fixture_dir = os.path.join(PROJECT_ROOT, "tests", "fixtures", "answering")

        # Select matching fixture or fallback
        q_lower = question.lower()
        if "trademark" in q_lower or "tm-a" in q_lower or "rule 23" in q_lower:
            fixture_name = "multiple_citations.json"
        elif "section 3" in q_lower or "patent" in q_lower:
            fixture_name = "citable_statutory.json"
        elif "stale" in q_lower or "drugs" in q_lower or "cosmetics" in q_lower:
            fixture_name = "stale_only.json"
        elif "conflict" in q_lower or "rule 65" in q_lower:
            fixture_name = "conflicting.json"
        elif "lunar" in q_lower or "martian" in q_lower or "space" in q_lower:
            fixture_name = "empty.json"
        else:
            # Default to statutory fixture for grounded evaluation
            fixture_name = "citable_statutory.json"

        fixture_path = os.path.join(fixture_dir, fixture_name)
        if not os.path.exists(fixture_path):
            fixture_path = os.path.join(fixture_dir, "citable_statutory.json")

        with open(fixture_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        # Convert fixture rows to RetrievalResults
        results = [
            RetrievalResult(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                score=r["score"],
                rank=r["rank"],
                text=r["text"],
                locator=r["locator"],
                citable=r["citable"],
                metadata=r["metadata"]
            )
            for r in data.get("results", [])
        ]

        evidence = EvidenceSet.from_results(results)
        policy = AnswerPolicy()
        engine = GroundedAnswerEngine(provider=ExtractiveMockProvider(), policy=policy)
        answer = engine.answer(question, evidence)

        # Format output
        output = {
            "success": True,
            "insufficient_evidence": answer.insufficient_evidence,
            "answer_text": answer.answer,
            "confidence": answer.confidence if not answer.insufficient_evidence else None,
            "citations": [
                {
                    "id": c.marker,
                    "marker": f"[{c.marker}]",
                    "document": c.source_document.document_id if hasattr(c, "source_document") else c.document_id,
                    "locator": c.locator,
                    "page": c.pages[0] if c.pages else 1,
                    "source_status": c.temporal_status,
                    "provenance_level": c.source_evidence_level,
                    "evidence_text": evidence.by_marker(c.marker).text if evidence.by_marker(c.marker) else ""
                }
                for c in answer.citations
            ],
            "cautions": list(answer.cautions),
            "diagnostics": {
                "grounded": answer.grounded,
                "conflict_detected": answer.conflict_detected,
                "provider": answer.provider,
                "evidence_fingerprint": answer.evidence_fingerprint
            }
        }
        print(json.dumps(output))

    except Exception as exc:
        print(json.dumps({"error": str(exc)}))
        sys.exit(0)

if __name__ == "__main__":
    main()
