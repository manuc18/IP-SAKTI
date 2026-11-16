"""IP-SAKTI Phase 2 ingestion pipeline.

    python scripts/ingest/run_pipeline.py --extract --clean --structure --chunk --validate
    python scripts/ingest/run_pipeline.py --all

Stages are independent and idempotent; each reads the previous stage's output
from ``data/processed`` and can be re-run on its own. ``--all`` refuses to start
when the OCR dependencies are missing rather than quietly producing a corpus
with 217 pages of statutory text missing from it.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest import chunk, clean, extract, ocr, structure, validate  # noqa: E402
from ingest.common import DATA, PROJECT_ROOT  # noqa: E402

STAGES = ("audit", "extract", "ocr", "clean", "structure", "chunk", "validate")


def run_audit() -> None:
    output = os.path.join(DATA, "metadata", "raw_audit.json")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    subprocess.run(
        [sys.executable, os.path.join(PROJECT_ROOT, "scripts", "audit_corpus.py"), output],
        check=True,
    )
    print(f"audit → {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    for stage in STAGES:
        parser.add_argument(f"--{stage}", action="store_true", help=f"run the {stage} stage")
    parser.add_argument("--all", action="store_true", help="run every stage in order")
    args = parser.parse_args(argv)

    selected = [stage for stage in STAGES if getattr(args, stage)]
    if args.all:
        selected = list(STAGES)
        missing = ocr.missing_dependencies()
        if missing:
            print("Refusing to run --all: OCR dependencies are unavailable.")
            for item in missing:
                print(f"  - {item}")
            print(
                "\nOCR covers 217 pages the audit proved are scans "
                "(Ayurveda Aahara 3-172, Patents Rules 56-102). Install the "
                "dependencies, or run the stages individually and omit --ocr, "
                "accepting that those pages will be absent."
            )
            return 2
    if not selected:
        parser.print_help()
        return 1

    for stage in selected:
        started = time.time()
        print(f"\n=== {stage} ===", flush=True)
        if stage == "audit":
            run_audit()
        elif stage == "extract":
            extract.run()
        elif stage == "ocr":
            missing = ocr.missing_dependencies()
            if missing:
                print("skipping OCR — dependencies unavailable:")
                for item in missing:
                    print(f"  - {item}")
                continue
            print(json.dumps(ocr.run(), indent=2))
        elif stage == "clean":
            clean.run()
        elif stage == "structure":
            structure.run()
        elif stage == "chunk":
            chunk.run()
        elif stage == "validate":
            outcome = validate.run()
            path = validate.write_report(outcome)
            with open(os.path.join(PROJECT_ROOT, "docs", "ingestion_validation.json"),
                      "w", encoding="utf-8") as fh:
                json.dump(outcome, fh, indent=2, ensure_ascii=False)
            print(f"wrote {path}")
            if outcome["status"] != "PASS":
                print("VALIDATION_STATUS: FAIL")
                return 1
        print(f"--- {stage} finished in {time.time() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
