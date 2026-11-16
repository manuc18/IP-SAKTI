# `data/source_updates/` — controlled staging for acquired sources

Nothing here has replaced anything. Files land in this directory so that a
replacement can be examined, compared and approved *before* it displaces a
source, and so that the displaced source remains readable afterwards. We need a
provenance chain, not a disappearing history.

## Rules

* `~/Downloads/IP-SAKTI Corpus` is immutable. No file in it is renamed,
  replaced, overwritten or deleted — by this directory or anything else.
* Filenames are content-addressed: the first 16 hex characters of the file's
  SHA-256, then a human-readable label. The digest is the identity; the label is
  for people.
* A file here supersedes a corpus source only after an explicit approval step
  and a fresh ingestion run. Placing a file here is not approval.
* `data/metadata/source_verifications.json` records what each file was compared
  against and what the comparison found. That file is written by
  `scripts/provenance/verify_source.py`, not by hand.

## What is here

**Publisher copies fetched for verification.** Each was compared page by page
against the corpus file it corresponds to. Where the verdict was
`CONTENT_IDENTICAL`, the corpus file is the publisher's document in a rewritten
container — see `docs/PROVENANCE_REPORT.md` §4.

**Acquired replacements, awaiting approval.**

| File | Supersedes | Why |
|---|---|---|
| `…-dc-rules-1945-cdsco-2024-07-01.pdf` | the Rules portion of `2016DrugsandCosmeticsAct1940Rules1945.pdf` | CDSCO consolidation to G.S.R. 360(E) of 1 July 2024; 963 pages against our 597 |

**Candidates held but not accepted.**

| File | Intended for | Why not accepted |
|---|---|---|
| `a690fc8c03db07c1-fssai-ayurveda-aahara-regulations-2022-legitquest.pdf` | the missing Ayurveda Aahara Regulations, 2022 | Right text, wrong source, and incomplete. A LegitQuest commercial-database print-to-PDF, which §2 excludes as a publisher of record; and it stops inside Schedule D, so Schedule E — the mandatory product logo — is referenced but never reproduced. Evidence level C, `accepted: false`. |

**Reference copies** that are not replacements for anything, kept because they
settle a question in the report — the FSSAI Ayurveda Aahara Compendium, for
instance, which establishes that FSSAI's compendium is a 647-page recipe index
and not the Regulations.

## Regenerating

```bash
.venv/bin/python scripts/provenance/verify_source.py --batch <targets.json>
.venv/bin/python scripts/provenance/build_registry.py
```
