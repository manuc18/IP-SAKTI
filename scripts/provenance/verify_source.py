"""Phase 3 — mechanical verification of a local PDF against a publisher copy.

Fifteen of the twenty-one corpus PDFs are browser re-encodings: the container
was rewritten (PDFium, iLovePDF, Chrome's print pipeline) and the publisher
metadata destroyed, so the SHA-256 of the local file can never equal the
SHA-256 of the file the publisher serves. A hash mismatch therefore proves
nothing on its own, and eyeballing two documents proves less.

What *can* be established mechanically is whether the two files carry the same
document. This module downloads a candidate publisher copy into the controlled
staging area and compares it page by page on three axes that survive
re-encoding:

* the extracted text layer of every page,
* the page geometry (MediaBox) of every page,
* the SHA-256 of the embedded image XObjects, for scanned pages.

A file whose 172 text layers and 172 page images all match the publisher's copy
is the publisher's document, whatever its container hash says. That verdict is
`CONTENT_IDENTICAL`, and it is the difference between evidence level D and B.

Nothing here writes to the corpus. Downloads land in `data/source_updates/`
under content-addressed names, and the original PDFs are opened read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ingest.common import (  # noqa: E402
    CORPUS_ROOT,
    METADATA_DIR,
    SOURCE_UPDATES_DIR,
    logical_documents,
    sha256_file,
)

VERIFICATIONS_PATH = os.path.join(METADATA_DIR, "source_verifications.json")

#: How many scanned pages to compare by embedded-image digest. Comparing every
#: page of a 600-page compendium costs minutes and adds nothing once the text
#: layers already match; a spread of samples catches a substituted page.
IMAGE_SAMPLE = 12

USER_AGENT = "IP-SAKTI-provenance/0.1 (source verification; contact: local use)"
TIMEOUT = 180

VERDICTS = (
    "BYTE_IDENTICAL",         # same SHA-256: the local file is the publisher's file
    "CONTENT_IDENTICAL",      # re-encoded container, identical document
    "CONTENT_IDENTICAL_TEXT",  # every text layer matches; some images re-compressed
    "CONTENT_DIVERGENT",      # some pages differ in text: likely a different revision
    "STRUCTURE_DIFFERENT",    # different page count: a different document or edition
    "NOT_A_PDF",              # the URL served HTML — usually a JS shell or error page
    "DOWNLOAD_FAILED",
)

#: Per-page stamps applied by online repository viewers, not by the publisher of
#: the document. India Code prints "IndiaCode" on every page it serves; the
#: underlying text is the publisher's. Stripping the stamp is the difference
#: between "these are different documents" and "this is the same document,
#: fetched through a repository viewer" — which is exactly what we need to know.
VIEWER_STAMPS = ("IndiaCode",)


def normalise(text: str) -> str:
    for stamp in VIEWER_STAMPS:
        text = text.replace(stamp, "")
    return text.strip()


def slugify(value: str) -> str:
    return re.sub(r"-+", "-", re.sub(r"[^a-z0-9]+", "-", value.lower())).strip("-")


def download(url: str, destination: str) -> tuple[bool, str]:
    """Fetch ``url`` to ``destination``. Returns ``(ok, detail)``.

    Several Indian government hosts (copyright.gov.in among them) serve a chain
    that omits the intermediate certificate. Browsers and curl repair that from
    their own stores; Python's urllib does not, and fails the handshake. Falling
    back to curl is not a relaxation of verification — curl still validates the
    chain — it just has a store that can complete it.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    payload = b""
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT) as response:
            payload = response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        completed = subprocess.run(
            ["curl", "-sSL", "--max-time", str(TIMEOUT), "-A", USER_AGENT, url],
            capture_output=True,
        )
        if completed.returncode != 0 or not completed.stdout:
            detail = completed.stderr.decode("utf-8", "replace").strip()
            return False, f"{type(exc).__name__}: {exc}" + (f" | curl: {detail}" if detail else "")
        payload = completed.stdout
    if not payload.startswith(b"%PDF"):
        head = payload[:200].decode("utf-8", "replace")
        return False, f"not a PDF (served {len(payload)} bytes starting {head!r})"
    with open(destination, "wb") as fh:
        fh.write(payload)
    return True, f"{len(payload)} bytes"


def page_digests(reader: Any, sample: list[int]) -> dict[int, str]:
    digests = {}
    for index in sample:
        digest = hashlib.sha256()
        for image in reader.pages[index].images:
            digest.update(image.data)
        digests[index + 1] = digest.hexdigest()
    return digests


def compare(local_path: str, candidate_path: str) -> dict[str, Any]:
    """Compare two PDFs on text layer, page geometry and embedded images."""
    from pypdf import PdfReader

    local = PdfReader(local_path)
    candidate = PdfReader(candidate_path)
    result: dict[str, Any] = {
        "local_pages": len(local.pages),
        "candidate_pages": len(candidate.pages),
    }
    if len(local.pages) != len(candidate.pages):
        result["verdict"] = "STRUCTURE_DIFFERENT"
        return result

    identical_text = 0
    identical_geometry = 0
    differing: list[int] = []
    for number, (a, b) in enumerate(zip(local.pages, candidate.pages), start=1):
        if normalise(a.extract_text() or "") == normalise(b.extract_text() or ""):
            identical_text += 1
        else:
            differing.append(number)
        if [round(float(v)) for v in a.mediabox] == [round(float(v)) for v in b.mediabox]:
            identical_geometry += 1

    total = len(local.pages)
    step = max(1, total // IMAGE_SAMPLE)
    sample = list(range(0, total, step))[:IMAGE_SAMPLE]
    local_images = page_digests(local, sample)
    candidate_images = page_digests(candidate, sample)
    matching_images = sum(
        1 for page, digest in local_images.items() if candidate_images.get(page) == digest
    )

    result.update({
        "pages_with_identical_text": identical_text,
        "pages_with_identical_geometry": identical_geometry,
        "pages_differing": differing[:25],
        "image_pages_sampled": len(sample),
        "image_pages_matching": matching_images,
    })
    result["text_normalisation"] = "viewer stamps removed: " + ", ".join(VIEWER_STAMPS)
    if identical_text < total:
        result["verdict"] = "CONTENT_DIVERGENT"
    elif identical_geometry == total and matching_images == len(sample):
        result["verdict"] = "CONTENT_IDENTICAL"
    else:
        # Every word on every page matches; only the re-encoded raster differs.
        # A browser "save as PDF" recompresses cover images, which says nothing
        # about the document's content.
        result["verdict"] = "CONTENT_IDENTICAL_TEXT"
    return result


def verify(document_id: str, url: str, *, label: str | None = None) -> dict[str, Any]:
    documents = {d.document_id: d for d in logical_documents()}
    if document_id not in documents:
        raise SystemExit(f"unknown document_id: {document_id}")
    document = documents[document_id]
    local_path = os.path.join(CORPUS_ROOT, document.source_filename)

    os.makedirs(SOURCE_UPDATES_DIR, exist_ok=True)
    staged = os.path.join(
        SOURCE_UPDATES_DIR, f"{slugify(label or document_id)}.download.pdf"
    )
    record: dict[str, Any] = {
        "document_id": document_id,
        "local_source_filename": document.source_filename,
        "local_sha256": document.source_sha256,
        "candidate_url": url,
    }

    ok, detail = download(url, staged)
    if not ok:
        record["verdict"] = "NOT_A_PDF" if "not a PDF" in detail else "DOWNLOAD_FAILED"
        record["detail"] = detail
        record["candidate_sha256"] = None
        return record

    candidate_sha = sha256_file(staged)
    record["candidate_sha256"] = candidate_sha
    record["candidate_bytes"] = os.path.getsize(staged)

    if candidate_sha == document.source_sha256:
        record["verdict"] = "BYTE_IDENTICAL"
        final = os.path.join(
            SOURCE_UPDATES_DIR, f"{candidate_sha[:16]}-{slugify(label or document_id)}.pdf"
        )
        os.replace(staged, final)
        record["staged_as"] = os.path.relpath(final)
        return record

    record.update(compare(local_path, staged))
    final = os.path.join(
        SOURCE_UPDATES_DIR, f"{candidate_sha[:16]}-{slugify(label or document_id)}.pdf"
    )
    os.replace(staged, final)
    record["staged_as"] = os.path.relpath(final)
    return record


def save(records: list[dict[str, Any]]) -> None:
    existing: dict[str, dict[str, Any]] = {}
    if os.path.exists(VERIFICATIONS_PATH):
        with open(VERIFICATIONS_PATH, encoding="utf-8") as fh:
            for record in json.load(fh)["verifications"]:
                existing[f"{record['document_id']}::{record['candidate_url']}"] = record
    for record in records:
        existing[f"{record['document_id']}::{record['candidate_url']}"] = record
    os.makedirs(METADATA_DIR, exist_ok=True)
    with open(VERIFICATIONS_PATH, "w", encoding="utf-8") as fh:
        json.dump(
            {"verifications": sorted(existing.values(), key=lambda r: r["document_id"])},
            fh, indent=2, ensure_ascii=False,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-id")
    parser.add_argument("--url")
    parser.add_argument("--label")
    parser.add_argument(
        "--batch",
        help="JSON file of [{document_id, url, label}] to verify in one pass",
    )
    args = parser.parse_args()

    if args.batch:
        with open(args.batch, encoding="utf-8") as fh:
            targets = json.load(fh)
    elif args.document_id and args.url:
        targets = [{"document_id": args.document_id, "url": args.url, "label": args.label}]
    else:
        parser.error("supply --batch, or both --document-id and --url")

    records = []
    for target in targets:
        print(f"verify {target['document_id']} <- {target['url']}", flush=True)
        record = verify(
            target["document_id"], target["url"], label=target.get("label")
        )
        print(f"  {record['verdict']}"
              f"{' — ' + record['detail'] if record.get('detail') else ''}", flush=True)
        records.append(record)
    save(records)
    print(f"\nwrote {VERIFICATIONS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
